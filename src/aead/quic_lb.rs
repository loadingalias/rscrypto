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
/// - [`Self::encrypt`] reads and writes only that plaintext prefix. It does not read or write the
///   octets after it.
/// - [`Self::decrypt_server_id`] reads only the ciphertext prefix. It leaves the server ID in the
///   first `server_id_len` octets and sets every later octet of the buffer to zero.
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
    let single_pass = server_id_len.strict_add(nonce_len) == SINGLE_PASS_LEN;
    Some(Self {
      encrypt: aes::aes128_expand_key(&key.0),
      decrypt: single_pass.then(|| aes::aes128_expand_decrypt_key(&key.0)),
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
  /// The octets after `server_id_len + nonce_len` are not read or written.
  pub fn encrypt(&self, block: &mut [u8; MAX_LEN]) {
    let len = self.len();
    if len == SINGLE_PASS_LEN {
      let (head, _) = block
        .split_first_chunk_mut::<SINGLE_PASS_LEN>()
        .expect("a 19-octet buffer");
      aes::aes128_encrypt_block(&self.encrypt, head);
      return;
    }
    let mut halves = Halves::split(block, len);
    self.passes(&mut halves, 1, 4);
    halves.join(block);
  }

  /// Decrypt the ciphertext at the start of `block` and leave the server ID in its first
  /// `server_id_len` octets.
  ///
  /// The octets after `server_id_len + nonce_len` are not read. Every octet after the server ID is
  /// set to zero.
  pub fn decrypt_server_id(&self, block: &mut [u8; MAX_LEN]) {
    let len = self.len();
    if let Some(decrypt) = &self.decrypt {
      let (head, _) = block
        .split_first_chunk_mut::<SINGLE_PASS_LEN>()
        .expect("a 19-octet buffer");
      aes::aes128_decrypt_block(decrypt, head);
    } else {
      let mut halves = Halves::split(block, len);
      // When the server ID fits in the left half, pass 1 recovers only nonce octets.
      let last = if self.server_id_len > self.nonce_len { 1 } else { 2 };
      self.passes(&mut halves, 4, last);
      halves.join(block);
    }
    ct::zeroize(&mut block[self.server_id_len()..]);
  }

  /// Run the Feistel passes from `first` to `last`, in either direction, as one sequence of
  /// dependent AES blocks.
  #[inline]
  fn passes(&self, halves: &mut Halves, first: u8, last: u8) {
    aes::aes128_serial(&self.encrypt, Feistel { halves, first, last });
  }
}

/// The Feistel passes of section 5.4.2.3 over two halves, from pass `first` to pass `last`.
struct Feistel<'a> {
  halves: &'a mut Halves,
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
    let Self { halves, first, last } = self;
    let (mut left, mut right) = (halves.left, halves.right);
    let (left_mask, right_mask) = (halves.left_mask, halves.right_mask);
    let len = halves.len;
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

/// The two Feistel halves of a plaintext or ciphertext, each in the first `half` octets of a
/// 16-octet block with zero padding, and the masks that keep each half in place.
struct Halves {
  left: [u8; 16],
  right: [u8; 16],
  /// Ones over the left half, without the low four bits of its last octet for odd lengths.
  left_mask: [u8; 16],
  /// Ones over the right half, without the high four bits of its first octet for odd lengths.
  right_mask: [u8; 16],
  /// Plaintext length, the `length` argument of expand (section 5.4.2.2).
  len: u8,
}

impl Halves {
  /// Split the first `len` octets of `block` into halves of `len / 2` octets, rounded up
  /// (section 5.4.2.3, step 2). For odd lengths, the halves share the middle octet: the left half
  /// keeps its high four bits and the right half its low four bits.
  #[inline]
  fn split(block: &[u8; MAX_LEN], len: usize) -> Self {
    let half = len.div_ceil(2);
    let odd = len % 2 == 1;
    let mut halves = Self {
      left: [0; 16],
      right: [0; 16],
      left_mask: [0; 16],
      right_mask: [0; 16],
      len: u8::try_from(len).expect("QUIC-LB length fits one octet"),
    };
    halves.left[..half].copy_from_slice(&block[..half]);
    halves.right[..half].copy_from_slice(&block[len.strict_sub(half)..len]);
    halves.left_mask[..half].fill(0xff);
    halves.right_mask[..half].fill(0xff);
    if odd {
      halves.left_mask[half.strict_sub(1)] = 0xf0;
      halves.right_mask[0] = 0x0f;
    }
    for (byte, mask) in halves.left.iter_mut().zip(&halves.left_mask) {
      *byte &= mask;
    }
    for (byte, mask) in halves.right.iter_mut().zip(&halves.right_mask) {
      *byte &= mask;
    }
    halves
  }

  /// Write the halves back over the first `len` octets of `block` (section 5.4.2.3, step 12).
  /// For odd lengths, the shared middle octet is the left half's high bits and the right half's
  /// low bits.
  #[inline]
  fn join(&self, block: &mut [u8; MAX_LEN]) {
    let len = usize::from(self.len);
    let half = len.div_ceil(2);
    let start = len.strict_sub(half);
    block[start..len].copy_from_slice(&self.right[..half]);
    block[..start].copy_from_slice(&self.left[..start]);
    if half > start {
      block[start] |= self.left[start];
    }
  }
}

impl Drop for Halves {
  fn drop(&mut self) {
    ct::zeroize(&mut self.left);
    ct::zeroize(&mut self.right);
  }
}
