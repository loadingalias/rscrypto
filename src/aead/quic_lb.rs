//! The QUIC-LB connection-ID cipher of
//! [draft-ietf-quic-load-balancers-22](https://datatracker.ietf.org/doc/draft-ietf-quic-load-balancers-22/),
//! section 5.4.
//!
//! The cipher encrypts the server ID and nonce of a QUIC-LB connection ID with AES-128: one block
//! when they total 16 octets, and a four-pass Feistel network for every other total. The first
//! octet, the configuration rotation, and the nonce choice are protocol-layer concerns.

use core::fmt;

use super::aes;
use crate::traits::ct;

/// Length of the QUIC-LB key in octets.
const KEY_LEN: usize = 16;
/// Largest server-ID and nonce total (section 5.3), and the length of every cipher buffer.
const MAX_LEN: usize = 19;
/// Total that selects the single-pass form (section 5.4.1).
const SINGLE_PASS_LEN: usize = 16;

/// A 128-bit QUIC-LB connection-ID key.
///
/// The key clears on drop. It is neither `Clone` nor `Copy`, and its `Debug` output names only the
/// type.
pub struct QuicLbKey([u8; KEY_LEN]);

impl QuicLbKey {
  /// Key length in bytes.
  pub const LENGTH: usize = KEY_LEN;

  /// Construct a key from the configuration's 16 octets.
  #[inline]
  #[must_use]
  pub const fn from_bytes(bytes: [u8; KEY_LEN]) -> Self {
    Self(bytes)
  }
}

impl fmt::Debug for QuicLbKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("QuicLbKey(****)")
  }
}

impl Drop for QuicLbKey {
  #[inline]
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

/// The QUIC-LB cipher for one server-ID length and nonce length.
///
/// The plaintext is the server ID followed by the nonce, `server_id_len + nonce_len` octets in
/// all, at the start of a buffer of [`Self::MAX_PLAINTEXT_LEN`] octets:
///
/// - [`Self::encrypt`] replaces that plaintext prefix with its ciphertext. The octets after it are
///   unchanged, and the output does not depend on them.
/// - [`Self::decrypt_server_id`] leaves the server ID in the first `server_id_len` octets and sets
///   every later octet of the buffer to zero. The output does not depend on the octets after the
///   ciphertext prefix.
///
/// Neither operation allocates or fails. No branch or memory index depends on the key, the
/// plaintext, or the ciphertext. The number of AES passes and the masks depend only on the
/// lengths. Decryption needs three AES passes when the server ID is no longer than the nonce, and
/// four otherwise (section 5.5.2).
///
/// The context owns expanded AES-128 schedules, which clear on drop. It is neither `Clone` nor
/// `Copy`. On alloc-enabled RISC-V without a hardware AES backend, construction keeps the boxed
/// fixslice schedules of the AES fallback. No-alloc builds store them inline.
pub struct QuicLbCipher {
  encrypt: aes::Aes128EncKey,
  /// The inverse cipher schedule, present only for the single-pass form.
  decrypt: Option<aes::Aes128DecKey>,
  layout: Layout,
  server_id_len: u8,
  nonce_len: u8,
}

impl fmt::Debug for QuicLbCipher {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("QuicLbCipher")
      .field("server_id_len", &self.server_id_len)
      .field("nonce_len", &self.nonce_len)
      .finish_non_exhaustive()
  }
}

impl QuicLbCipher {
  /// Shortest server ID in octets (section 5.3).
  pub const MIN_SERVER_ID_LEN: usize = 1;
  /// Shortest nonce in octets (section 5.3).
  pub const MIN_NONCE_LEN: usize = 4;
  /// Largest server-ID and nonce total in octets, and the length of every cipher buffer.
  pub const MAX_PLAINTEXT_LEN: usize = MAX_LEN;

  /// Construct the cipher for `key`, `server_id_len`, and `nonce_len`.
  ///
  /// Returns `None` unless the server ID is at least [`Self::MIN_SERVER_ID_LEN`] octets, the nonce
  /// is at least [`Self::MIN_NONCE_LEN`] octets, and their total is at most
  /// [`Self::MAX_PLAINTEXT_LEN`] octets.
  #[must_use]
  pub fn new(key: &QuicLbKey, server_id_len: usize, nonce_len: usize) -> Option<Self> {
    if server_id_len < Self::MIN_SERVER_ID_LEN
      || nonce_len < Self::MIN_NONCE_LEN
      || server_id_len.checked_add(nonce_len)? > MAX_LEN
    {
      return None;
    }
    let len = server_id_len.strict_add(nonce_len);
    Some(Self {
      encrypt: aes::aes128_expand_key(&key.0),
      decrypt: (len == SINGLE_PASS_LEN).then(|| aes::aes128_expand_decrypt_key(&key.0)),
      layout: Layout::new(len),
      server_id_len: u8::try_from(server_id_len).ok()?,
      nonce_len: u8::try_from(nonce_len).ok()?,
    })
  }

  /// Server-ID length in octets.
  #[inline]
  #[must_use]
  pub const fn server_id_len(&self) -> usize {
    self.server_id_len as usize
  }

  /// Nonce length in octets.
  #[inline]
  #[must_use]
  pub const fn nonce_len(&self) -> usize {
    self.nonce_len as usize
  }

  /// Plaintext length: the server-ID length plus the nonce length.
  #[inline]
  const fn len(&self) -> usize {
    self.server_id_len().strict_add(self.nonce_len())
  }

  /// Encrypt the server ID and nonce at the start of `block` in place.
  ///
  /// The octets after `server_id_len + nonce_len` are unchanged, and the output does not depend on
  /// them.
  pub fn encrypt(&self, block: &mut [u8; MAX_LEN]) {
    let len = self.len();
    if len == SINGLE_PASS_LEN {
      let (head, _) = block
        .split_first_chunk_mut::<SINGLE_PASS_LEN>()
        .expect("a 19-octet buffer");
      aes::aes128_encrypt_block(&self.encrypt, head);
      return;
    }
    let mut halves = self.layout.split(block);
    self.passes(&mut halves, 1, 4);
    self.layout.join(&halves, block);
  }

  /// Decrypt the ciphertext at the start of `block` and leave the server ID in its first
  /// `server_id_len` octets.
  ///
  /// The output does not depend on the octets after `server_id_len + nonce_len`. Every octet after
  /// the server ID is set to zero.
  pub fn decrypt_server_id(&self, block: &mut [u8; MAX_LEN]) {
    if let Some(decrypt) = &self.decrypt {
      let (head, _) = block
        .split_first_chunk_mut::<SINGLE_PASS_LEN>()
        .expect("a 19-octet buffer");
      aes::aes128_decrypt_block(decrypt, head);
    } else {
      let mut halves = self.layout.split(block);
      // When the server ID fits in the left half, pass 1 recovers only nonce octets.
      let last = if self.server_id_len > self.nonce_len { 1 } else { 2 };
      self.passes(&mut halves, 4, last);
      self.layout.join(&halves, block);
    }
    ct::zeroize(&mut block[self.server_id_len()..]);
  }

  /// Run the Feistel passes from `first` to `last`, in either direction, as one sequence of
  /// dependent AES blocks.
  #[inline]
  fn passes(&self, halves: &mut Halves, first: u8, last: u8) {
    aes::aes128_serial(
      &self.encrypt,
      Feistel {
        halves,
        layout: &self.layout,
        first,
        last,
      },
    );
  }
}

/// The Feistel passes of section 5.4.2.3 over two halves, from pass `first` to pass `last`.
struct Feistel<'a> {
  halves: &'a mut Halves,
  layout: &'a Layout,
  first: u8,
  last: u8,
}

impl aes::Aes128Serial for Feistel<'_> {
  type Output = ();

  /// Each pass expands one half, encrypts it with `encrypt`, truncates the result to the half
  /// length, and XORs it into the other half. Odd passes read the left half and write the right.
  /// The halves, the masks, the expanded block, and its encryption are local values, which the
  /// fused backends keep in vector registers. The halves write back once, and clear on drop.
  #[inline(always)]
  fn run(self, mut encrypt: impl FnMut([u8; 16]) -> [u8; 16]) {
    let Self {
      halves,
      layout,
      first,
      last,
    } = self;
    let (mut left, mut right) = (halves.left, halves.right);
    let (left_mask, right_mask) = (layout.left_mask, layout.right_mask);
    let len = layout.len;
    let mut pass = first;
    loop {
      if pass % 2 == 1 {
        right = feistel_pass(right, left, right_mask, len, pass, &mut encrypt);
      } else {
        left = feistel_pass(left, right, left_mask, len, pass, &mut encrypt);
      }
      if pass == last {
        break;
      }
      pass = if first < last {
        pass.strict_add(1)
      } else {
        pass.strict_sub(1)
      };
    }
    (halves.left, halves.right) = (left, right);
  }
}

/// `target ^ (AES(expand(len, pass, source)) & mask)`: one pass of section 5.4.2.3.
///
/// expand places the half, then zero padding, then the length and the pass number. A half is zero
/// after its first `len / 2` octets, rounded up, and so after octet 10, so OR with a block that
/// holds only the length and the pass number writes them in place. Whole-block OR keeps the data
/// flow in one vector register.
#[inline(always)]
fn feistel_pass(
  target: [u8; 16],
  source: [u8; 16],
  mask: [u8; 16],
  len: u8,
  pass: u8,
  encrypt: &mut impl FnMut([u8; 16]) -> [u8; 16],
) -> [u8; 16] {
  let mut tweak = [0u8; 16];
  tweak[14] = len;
  tweak[15] = pass;
  let output = encrypt(core::array::from_fn(|i| source[i] | tweak[i]));
  core::array::from_fn(|i| target[i] ^ (output[i] & mask[i]))
}

/// The two Feistel halves of a plaintext or ciphertext, each in the first `len / 2` octets
/// (rounded up) of a 16-octet block, with zero padding.
struct Halves {
  left: [u8; 16],
  right: [u8; 16],
}

/// Where the halves sit in the 19-octet buffer, fixed by the plaintext length at construction.
///
/// The buffer is read and written only through its two 16-octet windows, octets 0..16 and 3..19,
/// so every access has a fixed length and no runtime copy length exists. The right half starts at
/// octet `len - half`, which is at most 9, and ends before octet `len`, which is at most 19, so
/// it lies inside one window.
struct Layout {
  /// Ones over the left half, without the low four bits of its last octet for odd lengths.
  left_mask: [u8; 16],
  /// Ones over the right half, without the high four bits of its first octet for odd lengths.
  right_mask: [u8; 16],
  /// Whether the right half lies in the window at octet 3 (otherwise the window at octet 0).
  right_in_tail: bool,
  /// Bit offset of the right half inside its window.
  right_shift: u32,
  /// Bit offset of the right half from the start of the buffer.
  right_start: u32,
  /// Left and right shifts that move the right half from bit 0 to its place in the window at
  /// octet 3. One of the two is zero.
  tail_shifts: (u32, u32),
  /// Ones over the octets of the window at octet 0 that are below `len`.
  head_mask: u128,
  /// Ones over the octets of the window at octet 3 that are below `len`.
  tail_mask: u128,
  /// Plaintext length, the `length` argument of expand (section 5.4.2.2).
  len: u8,
}

/// The first `octets` octets of a little-endian `u128` window set to ones.
const fn low_octets(octets: usize) -> u128 {
  if octets >= 16 {
    u128::MAX
  } else {
    (1u128 << octets.strict_mul(8)).strict_sub(1)
  }
}

/// The 16-octet window at octet 0 of the buffer, as a little-endian integer.
#[inline(always)]
fn head(block: &[u8; MAX_LEN]) -> u128 {
  // Octets 0..16 of the 19-octet buffer.
  u128::from_le_bytes(*block.first_chunk::<16>().expect("a 19-octet buffer"))
}

/// The 16-octet window at octet 3 of the buffer, as a little-endian integer.
#[inline(always)]
fn tail(block: &[u8; MAX_LEN]) -> u128 {
  // Octets 3..19 of the 19-octet buffer.
  u128::from_le_bytes(*block.last_chunk::<16>().expect("a 19-octet buffer"))
}

impl Layout {
  /// The layout of a `len`-octet plaintext, 5 to 19 octets, other than 16: halves of `len / 2`
  /// octets, rounded up (section 5.4.2.3, step 2). For odd lengths, the halves share the middle
  /// octet: the left half keeps its high four bits and the right half its low four bits.
  fn new(len: usize) -> Self {
    let half = len.div_ceil(2);
    let start = len.strict_sub(half);
    let odd = len % 2 == 1;
    let mut left_mask = [0u8; 16];
    let mut right_mask = [0u8; 16];
    left_mask[..half].fill(0xff);
    right_mask[..half].fill(0xff);
    if odd {
      left_mask[half.strict_sub(1)] = 0xf0;
      right_mask[0] = 0x0f;
    }
    let right_in_tail = start >= 3;
    let window = if right_in_tail { 3 } else { 0 };
    Self {
      left_mask,
      right_mask,
      right_in_tail,
      right_shift: u32::try_from(start.strict_sub(window).strict_mul(8)).expect("small shift"),
      right_start: u32::try_from(start.strict_mul(8)).expect("small shift"),
      tail_shifts: if right_in_tail {
        (
          u32::try_from(start.strict_sub(3).strict_mul(8)).expect("small shift"),
          0,
        )
      } else {
        (
          0,
          u32::try_from(3usize.strict_sub(start).strict_mul(8)).expect("small shift"),
        )
      },
      head_mask: low_octets(len),
      tail_mask: low_octets(len.saturating_sub(3)),
      len: u8::try_from(len).expect("QUIC-LB length fits one octet"),
    }
  }

  /// Split the plaintext or ciphertext at the start of `block` into its halves.
  #[inline]
  fn split(&self, block: &[u8; MAX_LEN]) -> Halves {
    let head = head(block);
    let window = if self.right_in_tail { tail(block) } else { head };
    Halves {
      left: (head & u128::from_le_bytes(self.left_mask)).to_le_bytes(),
      right: ((window >> self.right_shift) & u128::from_le_bytes(self.right_mask)).to_le_bytes(),
    }
  }

  /// Write `halves` over the first `len` octets of `block` (section 5.4.2.3, step 12). For odd
  /// lengths, the shared middle octet is the left half's high bits and the right half's low bits.
  /// The octets from `len` on keep their values.
  #[inline]
  fn join(&self, halves: &Halves, block: &mut [u8; MAX_LEN]) {
    let left = u128::from_le_bytes(halves.left);
    let right = u128::from_le_bytes(halves.right);
    // The right half shifted to its place in each window: octet `start` of the buffer is octet
    // `start` of the head window and octet `start - 3` of the tail window.
    let head_value = left | (right << self.right_start);
    let tail_value = (left >> 24) | ((right << self.tail_shifts.0) >> self.tail_shifts.1);
    let head_new = (head(block) & !self.head_mask) | (head_value & self.head_mask);
    let tail_new = (tail(block) & !self.tail_mask) | (tail_value & self.tail_mask);
    // Octets 0..16, then octets 3..19. The windows agree on octets 3..16.
    *block.first_chunk_mut::<16>().expect("a 19-octet buffer") = head_new.to_le_bytes();
    *block.last_chunk_mut::<16>().expect("a 19-octet buffer") = tail_new.to_le_bytes();
  }
}

impl Drop for Halves {
  fn drop(&mut self) {
    ct::zeroize(&mut self.left);
    ct::zeroize(&mut self.right);
  }
}
