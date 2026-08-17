//! Ascon-AEAD128 authenticated encryption (NIST SP 800-232).
//!
//! Pure Rust, `no_std` implementation with fixed-work, table-free source
//! structure, a 128-bit key, a 128-bit nonce, and a 128-bit authentication
//! tag. Generated-code timing claims remain configuration- and
//! release-evidence-bound; see `ct.toml`.

use core::fmt;

use super::{AeadBufferError, Nonce128, OpenError, SealError};
use crate::{
  backend::ascon::{permute_8_portable, permute_12_portable},
  traits::{Aead, ct},
};

const KEY_SIZE: usize = 16;
const NONCE_SIZE: usize = Nonce128::LENGTH;
const TAG_SIZE: usize = 16;
const RATE: usize = 16;

/// Ascon-AEAD128 IV (SP 800-232): k=128, r=128, a=12, b=8.
const IV: u64 = 0x0000_1000_808c_0001;
const DOMAIN_SEPARATOR: u64 = 0x8000_0000_0000_0000;

/// Little-endian padding: set the first free byte at position `n`.
#[inline(always)]
fn pad(n: usize) -> u64 {
  let shift = u32::try_from(n.strict_mul(8)).expect("Ascon tail position must fit the word width");
  0x01_u64.strict_shl(shift)
}

/// Clear the lowest `n` bytes of `word`.
#[inline(always)]
fn clear(word: u64, n: usize) -> u64 {
  if n == 0 {
    return word;
  }
  let shift = u32::try_from(n.strict_mul(8)).expect("Ascon tail length must fit the word width");
  word & u64::MAX.strict_shl(shift)
}

/// Load up to 8 bytes little-endian into a u64, zero-padding on the right.
#[inline(always)]
fn load_bytes(data: &[u8]) -> u64 {
  debug_assert!(data.len() <= 8);
  let mut buf = [0u8; 8];
  buf[..data.len()].copy_from_slice(data);
  u64::from_le_bytes(buf)
}

// Key

define_aead_key_type!(AsconAead128Key, KEY_SIZE, "Ascon-AEAD128 128-bit secret key.");

impl AsconAead128Key {
  /// Key halves as little-endian u64 words.
  #[inline]
  fn words(&self) -> (u64, u64) {
    let mut hi = [0u8; 8];
    let mut lo = [0u8; 8];
    hi.copy_from_slice(&self.0[..8]);
    lo.copy_from_slice(&self.0[8..]);
    (u64::from_le_bytes(hi), u64::from_le_bytes(lo))
  }
}

// Tag

define_aead_tag_type!(AsconAead128Tag, TAG_SIZE, "Ascon-AEAD128 128-bit authentication tag.");

// AEAD

/// Ascon-AEAD128 authenticated encryption with associated data.
///
/// NIST SP 800-232 lightweight AEAD with a 128-bit key, 128-bit nonce,
/// and 128-bit authentication tag. Built on the Ascon permutation with
/// rate = 128 bits, PA = 12 rounds, PB = 8 rounds.
///
/// # Per-key requirements
///
/// For conformance, SP 800-232 requires single-purpose keys generated
/// according to SP 800-133 with an approved random bit generator supporting
/// at least 128-bit security. It also requires a distinct nonce for every
/// encryption under one key. Across all encryption and decryption operations
/// under that key, including each nonce, no more than 2^54 bytes may be
/// processed. With this type's full 128-bit tags, no more than 2^96 failed
/// decryptions may occur under one key. Rotate the key at the data limit and
/// when the failed-decryption limit is reached.
///
/// The caller must enforce these deployment-wide limits. This type does not
/// keep counters because the same key may be used by multiple instances,
/// processes, or devices. Randomly generated nonces are not recorded; a
/// deployment that requires deterministic proof of uniqueness needs a
/// protocol-owned nonce issuer.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "getrandom")]
/// # {
/// use rscrypto::{Aead, AsconAead128, AsconAead128Key};
///
/// let key = AsconAead128Key::from_bytes([0u8; 16]);
/// let aead = AsconAead128::new(&key);
///
/// let mut buf = *b"hello";
/// let (nonce, tag) = aead.seal_random_in_place(b"", &mut buf)?;
/// aead.decrypt_in_place(&nonce, b"", &mut buf, &tag)?;
/// assert_eq!(&buf, b"hello");
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Tampering is reported as an opaque verification failure.
///
/// ```
/// # #[cfg(feature = "getrandom")]
/// # {
/// use rscrypto::{Aead, AsconAead128, AsconAead128Key, aead::OpenError};
///
/// let key = AsconAead128Key::from_bytes([0u8; 16]);
/// let aead = AsconAead128::new(&key);
///
/// let mut sealed = [0u8; 5 + AsconAead128::TAG_SIZE];
/// let nonce = aead.seal_random(b"", b"hello", &mut sealed)?;
/// sealed[0] ^= 1;
///
/// let mut opened = [0u8; 5];
/// assert_eq!(
///   aead.decrypt(&nonce, b"", &sealed, &mut opened),
///   Err(OpenError::verification())
/// );
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct AsconAead128 {
  key: AsconAead128Key,
}

impl fmt::Debug for AsconAead128 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("AsconAead128").finish_non_exhaustive()
  }
}

impl AsconAead128 {
  /// Key length in bytes.
  pub const KEY_SIZE: usize = KEY_SIZE;

  /// Nonce length in bytes.
  pub const NONCE_SIZE: usize = NONCE_SIZE;

  /// Tag length in bytes.
  pub const TAG_SIZE: usize = TAG_SIZE;

  /// Construct a new Ascon-AEAD128 instance from `key`.
  #[inline]
  #[must_use]
  pub fn new(key: &AsconAead128Key) -> Self {
    <Self as Aead>::new(key)
  }

  /// Rebuild a typed tag from raw tag bytes.
  #[inline]
  pub fn tag_from_slice(bytes: &[u8]) -> Result<AsconAead128Tag, AeadBufferError> {
    <Self as Aead>::tag_from_slice(bytes)
  }

  /// Decrypt `buffer` in place and verify the detached authentication tag.
  #[inline]
  pub fn decrypt_in_place(
    &self,
    nonce: &Nonce128,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &AsconAead128Tag,
  ) -> Result<(), OpenError> {
    <Self as Aead>::decrypt_in_place(self, nonce, aad, buffer, tag)
  }

  /// Decrypt a combined `ciphertext || tag` into `out`.
  #[inline]
  pub fn decrypt(
    &self,
    nonce: &Nonce128,
    aad: &[u8],
    ciphertext_and_tag: &[u8],
    out: &mut [u8],
  ) -> Result<(), OpenError> {
    <Self as Aead>::decrypt(self, nonce, aad, ciphertext_and_tag, out)
  }

  /// Initialize the 320-bit state from IV, key, and nonce.
  #[inline]
  fn initialize(&self, nonce: &Nonce128) -> [u64; 5] {
    let (k0, k1) = self.key.words();

    let n = nonce.as_bytes();
    let mut n0_buf = [0u8; 8];
    let mut n1_buf = [0u8; 8];
    n0_buf.copy_from_slice(&n[..8]);
    n1_buf.copy_from_slice(&n[8..]);
    let n0 = u64::from_le_bytes(n0_buf);
    let n1 = u64::from_le_bytes(n1_buf);

    let mut s = [IV, k0, k1, n0, n1];
    permute_12_portable(&mut s);
    s[3] ^= k0;
    s[4] ^= k1;
    s
  }

  /// Absorb associated data into the state.
  fn process_aad(s: &mut [u64; 5], aad: &[u8]) {
    if !aad.is_empty() {
      let (chunks, mut rest) = aad.as_chunks::<RATE>();
      for chunk in chunks {
        s[0] ^= load_bytes(&chunk[..8]);
        s[1] ^= load_bytes(&chunk[8..]);
        permute_8_portable(s);
      }

      let sidx = if rest.len() >= 8 {
        s[0] ^= load_bytes(&rest[..8]);
        rest = &rest[8..];
        1
      } else {
        0
      };
      s[sidx] ^= pad(rest.len());
      if !rest.is_empty() {
        s[sidx] ^= load_bytes(rest);
      }
      permute_8_portable(s);
    }

    s[4] ^= DOMAIN_SEPARATOR;
  }

  /// Finalize the state and extract the 128-bit tag.
  fn finalize(&self, s: &mut [u64; 5]) -> [u8; TAG_SIZE] {
    let (k0, k1) = self.key.words();

    s[2] ^= k0;
    s[3] ^= k1;
    permute_12_portable(s);
    s[3] ^= k0;
    s[4] ^= k1;

    let mut tag = [0u8; TAG_SIZE];
    tag[..8].copy_from_slice(&s[3].to_le_bytes());
    tag[8..].copy_from_slice(&s[4].to_le_bytes());
    tag
  }
}

impl Aead for AsconAead128 {
  const KEY_SIZE: usize = KEY_SIZE;
  const NONCE_SIZE: usize = NONCE_SIZE;
  const TAG_SIZE: usize = TAG_SIZE;

  type Key = AsconAead128Key;
  type Nonce = Nonce128;
  type Tag = AsconAead128Tag;

  fn new(key: &Self::Key) -> Self {
    Self {
      key: key.duplicate_secret(),
    }
  }

  fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
    if bytes.len() != TAG_SIZE {
      return Err(AeadBufferError::new());
    }
    let mut tag = [0u8; TAG_SIZE];
    tag.copy_from_slice(bytes);
    Ok(AsconAead128Tag::from_bytes(tag))
  }

  fn __encrypt_in_place_with_nonce(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    _token: crate::traits::aead::SealToken,
  ) -> Result<Self::Tag, SealError> {
    let mut s = self.initialize(nonce);
    Self::process_aad(&mut s, aad);

    let (blocks, mut tail) = buffer.as_chunks_mut::<RATE>();
    for block in blocks {
      s[0] ^= load_bytes(&block[..8]);
      block[..8].copy_from_slice(&s[0].to_le_bytes());
      s[1] ^= load_bytes(&block[8..]);
      block[8..].copy_from_slice(&s[1].to_le_bytes());
      permute_8_portable(&mut s);
    }

    let sidx = if tail.len() >= 8 {
      s[0] ^= load_bytes(&tail[..8]);
      tail[..8].copy_from_slice(&s[0].to_le_bytes());
      tail = &mut tail[8..];
      1
    } else {
      0
    };
    s[sidx] ^= pad(tail.len());
    if !tail.is_empty() {
      s[sidx] ^= load_bytes(tail);
      tail.copy_from_slice(&s[sidx].to_le_bytes()[..tail.len()]);
    }

    let tag = self.finalize(&mut s);
    ct::zeroize_words(&mut s);
    Ok(AsconAead128Tag::from_bytes(tag))
  }

  fn decrypt_in_place(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), OpenError> {
    let mut s = self.initialize(nonce);
    Self::process_aad(&mut s, aad);

    let (blocks, mut tail) = buffer.as_chunks_mut::<RATE>();
    for block in blocks {
      let c0 = load_bytes(&block[..8]);
      block[..8].copy_from_slice(&(s[0] ^ c0).to_le_bytes());
      s[0] = c0;
      let c1 = load_bytes(&block[8..]);
      block[8..].copy_from_slice(&(s[1] ^ c1).to_le_bytes());
      s[1] = c1;
      permute_8_portable(&mut s);
    }

    let sidx = if tail.len() >= 8 {
      let c0 = load_bytes(&tail[..8]);
      tail[..8].copy_from_slice(&(s[0] ^ c0).to_le_bytes());
      s[0] = c0;
      tail = &mut tail[8..];
      1
    } else {
      0
    };
    s[sidx] ^= pad(tail.len());
    if !tail.is_empty() {
      let c = load_bytes(tail);
      s[sidx] ^= c;
      tail.copy_from_slice(&s[sidx].to_le_bytes()[..tail.len()]);
      s[sidx] = clear(s[sidx], tail.len()) ^ c;
    }

    let expected = self.finalize(&mut s);
    ct::zeroize_words(&mut s);
    if !ct::fixed_eq(&expected, tag.as_bytes()).declassify() {
      ct::zeroize(buffer);
      return Err(OpenError::verification());
    }

    Ok(())
  }
}

#[cfg(feature = "diag")]
/// Compare a portable Ascon-AEAD128 tag computation with an expected diagnostic tag.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn diag_ascon_aead128_tag_portable(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; NONCE_SIZE],
  block: &[u8; RATE],
  expected: &[u8; TAG_SIZE],
) -> ct::CtDecision {
  let key = AsconAead128Key::from_bytes(*key);
  let nonce = Nonce128::from_bytes(*nonce);
  let cipher = AsconAead128::new(&key);
  let mut s = cipher.initialize(&nonce);
  AsconAead128::process_aad(&mut s, b"binsec");
  s[0] ^= load_bytes(&block[..8]);
  s[1] ^= load_bytes(&block[8..]);
  permute_8_portable(&mut s);
  let tag = cipher.finalize(&mut s);
  ct::zeroize_words(&mut s);
  ct::fixed_eq(&tag, expected)
}

#[cfg(test)]
mod tests {
  use alloc::{vec, vec::Vec};

  use ascon_aead::aead::{Aead as _, KeyInit, Payload, array::Array};

  use super::*;
  use crate::aead::expert::AeadWithNonce;

  fn assert_matches_oracle(key: [u8; 16], nonce: [u8; 16], aad: &[u8], plaintext: &[u8]) {
    let aead = AsconAead128::new(&AsconAead128Key::from_bytes(key));
    let nonce_typed = Nonce128::from_bytes(nonce);
    let oracle = ascon_aead::AsconAead128::new_from_slice(&key).expect("16-byte Ascon oracle key must be accepted");
    let oracle_nonce = Array(nonce);

    let mut ours = plaintext.to_vec();
    let tag = aead
      .encrypt_in_place(&nonce_typed, aad, &mut ours)
      .expect("rscrypto Ascon encryption must succeed");
    let mut ours_combined = ours.clone();
    ours_combined.extend_from_slice(tag.as_bytes());
    let expected = oracle
      .encrypt(&oracle_nonce, Payload { msg: plaintext, aad })
      .expect("oracle Ascon encryption must succeed");
    assert_eq!(ours_combined, expected, "encryption mismatch");

    let mut ours_buf = ours.clone();
    aead
      .decrypt_in_place(&nonce_typed, aad, &mut ours_buf, &tag)
      .expect("rscrypto Ascon self-decryption must succeed");
    assert_eq!(ours_buf, plaintext, "self decrypt mismatch");

    let (oracle_ct, oracle_tag) = expected.split_at(expected.len().strict_sub(TAG_SIZE));
    let mut oracle_buf = oracle_ct.to_vec();
    aead
      .decrypt_in_place(
        &nonce_typed,
        aad,
        &mut oracle_buf,
        &AsconAead128Tag::from_bytes(
          oracle_tag
            .try_into()
            .expect("oracle Ascon output must end in a 16-byte tag"),
        ),
      )
      .expect("rscrypto must decrypt the oracle Ascon ciphertext");
    assert_eq!(oracle_buf, plaintext, "oracle decrypt mismatch");
  }

  /// Round-trip: encrypt then decrypt recovers the original plaintext.
  #[test]
  fn round_trip_empty() {
    let key = AsconAead128Key::from_bytes([0u8; 16]);
    let nonce = Nonce128::from_bytes([0u8; 16]);
    let aead = AsconAead128::new(&key);

    let mut buf = [];
    let tag = aead
      .encrypt_in_place(&nonce, b"", &mut buf)
      .expect("empty Ascon encryption must succeed");
    aead
      .decrypt_in_place(&nonce, b"", &mut buf, &tag)
      .expect("empty Ascon decryption must succeed");
  }

  #[test]
  fn round_trip_with_data() {
    let key = AsconAead128Key::from_bytes([0x42; 16]);
    let nonce = Nonce128::from_bytes([0x13; 16]);
    let aead = AsconAead128::new(&key);
    let plaintext = b"the quick brown fox jumps over the lazy dog";

    let mut buf = *plaintext;
    let tag = aead
      .encrypt_in_place(&nonce, b"header", &mut buf)
      .expect("Ascon encryption with AAD must succeed");
    assert_ne!(&buf[..], &plaintext[..]);

    aead
      .decrypt_in_place(&nonce, b"header", &mut buf, &tag)
      .expect("Ascon decryption with AAD must succeed");
    assert_eq!(&buf[..], &plaintext[..]);
  }

  #[test]
  fn round_trip_with_aad_only() {
    let key = AsconAead128Key::from_bytes([0xFF; 16]);
    let nonce = Nonce128::from_bytes([0xAA; 16]);
    let aead = AsconAead128::new(&key);

    let mut buf = [];
    let tag = aead
      .encrypt_in_place(&nonce, b"associated data only", &mut buf)
      .expect("AAD-only Ascon encryption must succeed");
    aead
      .decrypt_in_place(&nonce, b"associated data only", &mut buf, &tag)
      .expect("AAD-only Ascon decryption must succeed");
  }

  #[test]
  fn buffer_zeroed_on_auth_failure() {
    let key = AsconAead128Key::from_bytes([0x42; 16]);
    let nonce = Nonce128::from_bytes([0x13; 16]);
    let aead = AsconAead128::new(&key);

    let mut buf = *b"zero me on failure";
    let tag = aead
      .encrypt_in_place(&nonce, b"aad", &mut buf)
      .expect("Ascon test setup encryption must succeed");

    let mut bad_tag = tag.to_bytes();
    bad_tag[0] ^= 0xFF;
    let bad_tag = AsconAead128Tag::from_bytes(bad_tag);

    assert_eq!(
      aead.decrypt_in_place(&nonce, b"aad", &mut buf, &bad_tag),
      Err(OpenError::verification())
    );
    assert!(buf.iter().all(|&b| b == 0), "buffer not zeroed on auth failure");
  }

  #[test]
  fn tampered_ciphertext_fails() {
    let key = AsconAead128Key::from_bytes([1; 16]);
    let nonce = Nonce128::from_bytes([2; 16]);
    let aead = AsconAead128::new(&key);

    let mut buf = *b"secret";
    let tag = aead
      .encrypt_in_place(&nonce, b"", &mut buf)
      .expect("Ascon test setup encryption must succeed");

    buf[0] ^= 1;
    assert_eq!(
      aead.decrypt_in_place(&nonce, b"", &mut buf, &tag),
      Err(OpenError::verification())
    );
    // Buffer must be zeroized on failure.
    assert_eq!(&buf, &[0u8; 6]);
  }

  #[test]
  fn tampered_tag_fails() {
    let key = AsconAead128Key::from_bytes([3; 16]);
    let nonce = Nonce128::from_bytes([4; 16]);
    let aead = AsconAead128::new(&key);

    let mut buf = *b"data";
    let tag = aead
      .encrypt_in_place(&nonce, b"aad", &mut buf)
      .expect("Ascon test setup encryption must succeed");

    let mut bad_tag_bytes = tag.to_bytes();
    bad_tag_bytes[15] ^= 1;
    let bad_tag = AsconAead128Tag::from_bytes(bad_tag_bytes);

    assert_eq!(
      aead.decrypt_in_place(&nonce, b"aad", &mut buf, &bad_tag),
      Err(OpenError::verification())
    );
    assert_eq!(&buf, &[0u8; 4]);
  }

  #[test]
  fn wrong_aad_fails() {
    let key = AsconAead128Key::from_bytes([5; 16]);
    let nonce = Nonce128::from_bytes([6; 16]);
    let aead = AsconAead128::new(&key);

    let mut buf = *b"msg";
    let tag = aead
      .encrypt_in_place(&nonce, b"correct", &mut buf)
      .expect("Ascon test setup encryption must succeed");

    assert_eq!(
      aead.decrypt_in_place(&nonce, b"wrong", &mut buf, &tag),
      Err(OpenError::verification())
    );
  }

  #[test]
  fn wrong_nonce_fails() {
    let key = AsconAead128Key::from_bytes([9; 16]);
    let nonce = Nonce128::from_bytes([10; 16]);
    let aead = AsconAead128::new(&key);

    let mut buf = *b"nonce test";
    let tag = aead
      .encrypt_in_place(&nonce, b"aad", &mut buf)
      .expect("Ascon test setup encryption must succeed");

    let wrong_nonce = Nonce128::from_bytes([11; 16]);
    assert_eq!(
      aead.decrypt_in_place(&wrong_nonce, b"aad", &mut buf, &tag),
      Err(OpenError::verification())
    );
  }

  #[test]
  fn combined_encrypt_decrypt_round_trip() {
    let key = AsconAead128Key::from_bytes([7; 16]);
    let nonce = Nonce128::from_bytes([8; 16]);
    let aead = AsconAead128::new(&key);
    let pt = b"combined mode";

    let mut sealed = vec![0u8; pt.len().strict_add(TAG_SIZE)];
    aead
      .encrypt(&nonce, b"h", pt.as_slice(), &mut sealed)
      .expect("combined Ascon encryption must succeed");

    let mut opened = vec![0u8; pt.len()];
    aead
      .decrypt(&nonce, b"h", &sealed, &mut opened)
      .expect("combined Ascon decryption must succeed");
    assert_eq!(&opened, &pt[..]);
  }

  #[test]
  fn tag_from_slice_rejects_wrong_length() {
    assert_eq!(
      AsconAead128::tag_from_slice(&[0u8; 15]).expect_err("short Ascon tag must be rejected"),
      AeadBufferError::new()
    );
    assert_eq!(
      AsconAead128::tag_from_slice(&[0u8; 17]).expect_err("long Ascon tag must be rejected"),
      AeadBufferError::new()
    );
    let tag = AsconAead128::tag_from_slice(&[0u8; 16]).expect("16-byte Ascon tag must be accepted");
    assert_eq!(tag.as_bytes(), &[0u8; 16]);
  }

  #[test]
  fn multi_block_round_trip() {
    let key = AsconAead128Key::from_bytes([0xAB; 16]);
    let nonce = Nonce128::from_bytes([0xCD; 16]);
    let aead = AsconAead128::new(&key);

    // 100 bytes = 12 full blocks + 4-byte tail.
    let plaintext = [0x77u8; 100];
    let mut buf = plaintext;
    let tag = aead
      .encrypt_in_place(&nonce, b"multi-block aad that is longer than one rate block", &mut buf)
      .expect("multi-block Ascon encryption must succeed");
    aead
      .decrypt_in_place(
        &nonce,
        b"multi-block aad that is longer than one rate block",
        &mut buf,
        &tag,
      )
      .expect("multi-block Ascon decryption must succeed");
    assert_eq!(buf, plaintext);
  }

  #[test]
  fn exact_rate_boundary() {
    let key = AsconAead128Key::from_bytes([0x10; 16]);
    let nonce = Nonce128::from_bytes([0x20; 16]);
    let aead = AsconAead128::new(&key);

    // Exactly 8 bytes = 1 full block, 0-byte tail.
    let plaintext = [0x55u8; 8];
    let mut buf = plaintext;
    let tag = aead
      .encrypt_in_place(&nonce, b"", &mut buf)
      .expect("one-word Ascon encryption must succeed");
    aead
      .decrypt_in_place(&nonce, b"", &mut buf, &tag)
      .expect("one-word Ascon decryption must succeed");
    assert_eq!(buf, plaintext);

    // Exactly 16 bytes = 2 full blocks, 0-byte tail.
    let plaintext16 = [0x66u8; 16];
    let mut buf16 = plaintext16;
    let tag16 = aead
      .encrypt_in_place(&nonce, b"", &mut buf16)
      .expect("one-rate Ascon encryption must succeed");
    aead
      .decrypt_in_place(&nonce, b"", &mut buf16, &tag16)
      .expect("one-rate Ascon decryption must succeed");
    assert_eq!(buf16, plaintext16);
  }

  #[test]
  fn differential_empty_inputs_match_oracle() {
    assert_matches_oracle([0u8; 16], [0u8; 16], b"", b"");
  }

  #[test]
  fn differential_crash_case_matches_oracle() {
    assert_matches_oracle(
      [
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x0A, 0xFF, 0xFF, 0xFF, 0x3D,
      ],
      [
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x0A, 0xFF, 0xFF, 0xFF, 0x3D, 0xFF, 0xFF, 0x0A,
      ],
      &[0xFF, 0xFF, 0xFF],
      b"",
    );
  }

  #[test]
  fn differential_exact_rate_boundaries_match_oracle() {
    let key = [
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    ];
    let nonce = [
      0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
    ];
    let aad: Vec<u8> = (0x30..0x40).collect();
    let pt: Vec<u8> = (0x20..0x30).collect();
    assert_matches_oracle(key, nonce, &aad[..8], &pt[..8]);
    assert_matches_oracle(key, nonce, &aad, &pt);
  }

  #[test]
  fn differential_multiblock_matches_oracle() {
    let key = [0x42; 16];
    let nonce = [0x24; 16];
    let aad: Vec<u8> = (0u8..48).collect();
    let pt: Vec<u8> = (0u8..97).map(|i| i.wrapping_mul(17)).collect();
    assert_matches_oracle(key, nonce, &aad, &pt);
  }
}
