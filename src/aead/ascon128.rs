#![allow(clippy::indexing_slicing)]

//! Ascon-AEAD128 authenticated encryption (NIST SP 800-232).
//!
//! Pure Rust, constant-time, `no_std` implementation with 128-bit key,
//! 128-bit nonce, and 128-bit authentication tag.

use core::fmt;

use super::{AeadBufferError, Nonce128, OpenError};
use crate::{
  backend::ascon::{permute_8_portable, permute_12_portable},
  traits::{Aead, VerificationError, ct},
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
const fn pad(n: usize) -> u64 {
  0x01_u64 << (8 * n)
}

/// Clear the lowest `n` bytes of `word`.
#[inline(always)]
const fn clear(word: u64, n: usize) -> u64 {
  if n == 0 {
    return word;
  }
  word & (u64::MAX << (8 * n))
}

/// Load up to 8 bytes little-endian into a u64, zero-padding on the right.
#[inline(always)]
fn load_bytes(data: &[u8]) -> u64 {
  debug_assert!(data.len() <= 8);
  let mut buf = [0u8; 8];
  buf[..data.len()].copy_from_slice(data);
  u64::from_le_bytes(buf)
}

// ---------------------------------------------------------------------------
// Key
// ---------------------------------------------------------------------------

/// Ascon-AEAD128 128-bit secret key.
#[derive(Clone)]
pub struct AsconAead128Key([u8; Self::LENGTH]);

impl PartialEq for AsconAead128Key {
  fn eq(&self, other: &Self) -> bool {
    ct::constant_time_eq(&self.0, &other.0)
  }
}

impl Eq for AsconAead128Key {}

impl AsconAead128Key {
  /// Key length in bytes.
  pub const LENGTH: usize = KEY_SIZE;

  /// Construct a key from raw bytes.
  #[inline]
  #[must_use]
  pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
    Self(bytes)
  }

  /// Return the raw key bytes.
  #[inline]
  #[must_use]
  pub fn to_bytes(&self) -> [u8; Self::LENGTH] {
    self.0
  }

  /// Borrow the raw key bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }

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

impl AsRef<[u8]> for AsconAead128Key {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

impl_ct_eq!(AsconAead128Key);

impl fmt::Debug for AsconAead128Key {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("AsconAead128Key(****)")
  }
}

impl AsconAead128Key {
  /// Construct a key by filling bytes from the provided closure.
  ///
  /// ```rust
  /// # use rscrypto::AsconAead128Key;
  /// let key = AsconAead128Key::generate(|buf| buf.fill(0xA5));
  /// assert_eq!(key.as_bytes(), &[0xA5; AsconAead128Key::LENGTH]);
  /// ```
  #[inline]
  #[must_use]
  pub fn generate(fill: impl FnOnce(&mut [u8; Self::LENGTH])) -> Self {
    let mut bytes = [0u8; Self::LENGTH];
    fill(&mut bytes);
    Self(bytes)
  }

  impl_getrandom!();
}

impl_hex_fmt_secret!(AsconAead128Key);
impl_serde_bytes!(AsconAead128Key);

impl Drop for AsconAead128Key {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

// ---------------------------------------------------------------------------
// Tag
// ---------------------------------------------------------------------------

/// Ascon-AEAD128 128-bit authentication tag.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsconAead128Tag([u8; Self::LENGTH]);

impl AsconAead128Tag {
  /// Tag length in bytes.
  pub const LENGTH: usize = TAG_SIZE;

  /// Construct a tag from raw bytes.
  #[inline]
  #[must_use]
  pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
    Self(bytes)
  }

  /// Return the raw tag bytes.
  #[inline]
  #[must_use]
  pub const fn to_bytes(self) -> [u8; Self::LENGTH] {
    self.0
  }

  /// Borrow the raw tag bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }
}

impl Default for AsconAead128Tag {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl AsRef<[u8]> for AsconAead128Tag {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

impl_ct_eq!(AsconAead128Tag);

impl fmt::Debug for AsconAead128Tag {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "AsconAead128Tag(")?;
    crate::hex::fmt_hex_lower(&self.0, f)?;
    write!(f, ")")
  }
}

impl_hex_fmt!(AsconAead128Tag);
impl_serde_bytes!(AsconAead128Tag);

// ---------------------------------------------------------------------------
// AEAD
// ---------------------------------------------------------------------------

/// Ascon-AEAD128 authenticated encryption with associated data.
///
/// NIST SP 800-232 lightweight AEAD with a 128-bit key, 128-bit nonce,
/// and 128-bit authentication tag. Built on the Ascon permutation with
/// rate = 128 bits, PA = 12 rounds, PB = 8 rounds.
///
/// # Examples
///
/// ```
/// use rscrypto::{Aead, AsconAead128, AsconAead128Key, AsconAead128Tag, aead::Nonce128};
///
/// let key = AsconAead128Key::from_bytes([0u8; 16]);
/// let nonce = Nonce128::from_bytes([0u8; 16]);
/// let aead = AsconAead128::new(&key);
///
/// let mut buf = *b"hello";
/// let tag = aead.encrypt_in_place(&nonce, b"", &mut buf);
/// aead.decrypt_in_place(&nonce, b"", &mut buf, &tag)?;
/// assert_eq!(&buf, b"hello");
/// # Ok::<(), rscrypto::VerificationError>(())
/// ```
#[derive(Clone)]
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

  /// Encrypt `buffer` in place and return the detached authentication tag.
  #[inline]
  #[must_use]
  pub fn encrypt_in_place(&self, nonce: &Nonce128, aad: &[u8], buffer: &mut [u8]) -> AsconAead128Tag {
    <Self as Aead>::encrypt_in_place(self, nonce, aad, buffer)
  }

  /// Decrypt `buffer` in place and verify the detached authentication tag.
  #[inline]
  pub fn decrypt_in_place(
    &self,
    nonce: &Nonce128,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &AsconAead128Tag,
  ) -> Result<(), VerificationError> {
    <Self as Aead>::decrypt_in_place(self, nonce, aad, buffer, tag)
  }

  /// Encrypt `plaintext` into `out` as `ciphertext || tag`.
  #[inline]
  pub fn encrypt(&self, nonce: &Nonce128, aad: &[u8], plaintext: &[u8], out: &mut [u8]) -> Result<(), AeadBufferError> {
    <Self as Aead>::encrypt(self, nonce, aad, plaintext, out)
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

  // -----------------------------------------------------------------------
  // Internal helpers
  // -----------------------------------------------------------------------

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
      let mut chunks = aad.chunks_exact(RATE);
      for chunk in chunks.by_ref() {
        s[0] ^= load_bytes(&chunk[..8]);
        s[1] ^= load_bytes(&chunk[8..]);
        permute_8_portable(s);
      }

      let mut rest = chunks.remainder();
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

// ---------------------------------------------------------------------------
// Aead trait implementation
// ---------------------------------------------------------------------------

impl Aead for AsconAead128 {
  const KEY_SIZE: usize = KEY_SIZE;
  const NONCE_SIZE: usize = NONCE_SIZE;
  const TAG_SIZE: usize = TAG_SIZE;

  type Key = AsconAead128Key;
  type Nonce = Nonce128;
  type Tag = AsconAead128Tag;

  fn new(key: &Self::Key) -> Self {
    Self { key: key.clone() }
  }

  fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
    if bytes.len() != TAG_SIZE {
      return Err(AeadBufferError::new());
    }
    let mut tag = [0u8; TAG_SIZE];
    tag.copy_from_slice(bytes);
    Ok(AsconAead128Tag::from_bytes(tag))
  }

  fn encrypt_in_place(&self, nonce: &Self::Nonce, aad: &[u8], buffer: &mut [u8]) -> Self::Tag {
    let mut s = self.initialize(nonce);
    Self::process_aad(&mut s, aad);

    let mut blocks = buffer.chunks_exact_mut(RATE);
    for block in blocks.by_ref() {
      s[0] ^= load_bytes(&block[..8]);
      block[..8].copy_from_slice(&s[0].to_le_bytes());
      s[1] ^= load_bytes(&block[8..]);
      block[8..].copy_from_slice(&s[1].to_le_bytes());
      permute_8_portable(&mut s);
    }

    let mut tail = blocks.into_remainder();
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

    AsconAead128Tag::from_bytes(self.finalize(&mut s))
  }

  fn decrypt_in_place(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), VerificationError> {
    let mut s = self.initialize(nonce);
    Self::process_aad(&mut s, aad);

    let mut blocks = buffer.chunks_exact_mut(RATE);
    for block in blocks.by_ref() {
      let c0 = load_bytes(&block[..8]);
      block[..8].copy_from_slice(&(s[0] ^ c0).to_le_bytes());
      s[0] = c0;
      let c1 = load_bytes(&block[8..]);
      block[8..].copy_from_slice(&(s[1] ^ c1).to_le_bytes());
      s[1] = c1;
      permute_8_portable(&mut s);
    }

    let mut tail = blocks.into_remainder();
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
    if !ct::constant_time_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Err(VerificationError::new());
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use alloc::{vec, vec::Vec};

  use ascon_aead::aead::{Aead as _, KeyInit, Payload, generic_array::GenericArray};

  use super::*;

  fn assert_matches_oracle(key: [u8; 16], nonce: [u8; 16], aad: &[u8], plaintext: &[u8]) {
    let aead = AsconAead128::new(&AsconAead128Key::from_bytes(key));
    let nonce_typed = Nonce128::from_bytes(nonce);
    let oracle = ascon_aead::AsconAead128::new_from_slice(&key).unwrap();
    let oracle_nonce = GenericArray::from_slice(&nonce);

    let mut ours = plaintext.to_vec();
    let tag = aead.encrypt_in_place(&nonce_typed, aad, &mut ours);
    let mut ours_combined = ours.clone();
    ours_combined.extend_from_slice(tag.as_bytes());
    let expected = oracle.encrypt(oracle_nonce, Payload { msg: plaintext, aad }).unwrap();
    assert_eq!(ours_combined, expected, "encryption mismatch");

    let mut ours_buf = ours.clone();
    aead.decrypt_in_place(&nonce_typed, aad, &mut ours_buf, &tag).unwrap();
    assert_eq!(ours_buf, plaintext, "self decrypt mismatch");

    let (oracle_ct, oracle_tag) = expected.split_at(expected.len().strict_sub(TAG_SIZE));
    let mut oracle_buf = oracle_ct.to_vec();
    aead
      .decrypt_in_place(
        &nonce_typed,
        aad,
        &mut oracle_buf,
        &AsconAead128Tag::from_bytes(oracle_tag.try_into().unwrap()),
      )
      .unwrap();
    assert_eq!(oracle_buf, plaintext, "oracle decrypt mismatch");
  }

  /// Round-trip: encrypt then decrypt recovers the original plaintext.
  #[test]
  fn round_trip_empty() {
    let key = AsconAead128Key::from_bytes([0u8; 16]);
    let nonce = Nonce128::from_bytes([0u8; 16]);
    let aead = AsconAead128::new(&key);

    let mut buf = [];
    let tag = aead.encrypt_in_place(&nonce, b"", &mut buf);
    aead.decrypt_in_place(&nonce, b"", &mut buf, &tag).unwrap();
  }

  #[test]
  fn round_trip_with_data() {
    let key = AsconAead128Key::from_bytes([0x42; 16]);
    let nonce = Nonce128::from_bytes([0x13; 16]);
    let aead = AsconAead128::new(&key);
    let plaintext = b"the quick brown fox jumps over the lazy dog";

    let mut buf = *plaintext;
    let tag = aead.encrypt_in_place(&nonce, b"header", &mut buf);
    assert_ne!(&buf[..], &plaintext[..]);

    aead.decrypt_in_place(&nonce, b"header", &mut buf, &tag).unwrap();
    assert_eq!(&buf[..], &plaintext[..]);
  }

  #[test]
  fn round_trip_with_aad_only() {
    let key = AsconAead128Key::from_bytes([0xFF; 16]);
    let nonce = Nonce128::from_bytes([0xAA; 16]);
    let aead = AsconAead128::new(&key);

    let mut buf = [];
    let tag = aead.encrypt_in_place(&nonce, b"associated data only", &mut buf);
    aead
      .decrypt_in_place(&nonce, b"associated data only", &mut buf, &tag)
      .unwrap();
  }

  #[test]
  fn tampered_ciphertext_fails() {
    let key = AsconAead128Key::from_bytes([1; 16]);
    let nonce = Nonce128::from_bytes([2; 16]);
    let aead = AsconAead128::new(&key);

    let mut buf = *b"secret";
    let tag = aead.encrypt_in_place(&nonce, b"", &mut buf);

    buf[0] ^= 1;
    let result = aead.decrypt_in_place(&nonce, b"", &mut buf, &tag);
    assert!(result.is_err());
    // Buffer must be zeroized on failure.
    assert_eq!(&buf, &[0u8; 6]);
  }

  #[test]
  fn tampered_tag_fails() {
    let key = AsconAead128Key::from_bytes([3; 16]);
    let nonce = Nonce128::from_bytes([4; 16]);
    let aead = AsconAead128::new(&key);

    let mut buf = *b"data";
    let tag = aead.encrypt_in_place(&nonce, b"aad", &mut buf);

    let mut bad_tag_bytes = tag.to_bytes();
    bad_tag_bytes[15] ^= 1;
    let bad_tag = AsconAead128Tag::from_bytes(bad_tag_bytes);

    let result = aead.decrypt_in_place(&nonce, b"aad", &mut buf, &bad_tag);
    assert!(result.is_err());
    assert_eq!(&buf, &[0u8; 4]);
  }

  #[test]
  fn wrong_aad_fails() {
    let key = AsconAead128Key::from_bytes([5; 16]);
    let nonce = Nonce128::from_bytes([6; 16]);
    let aead = AsconAead128::new(&key);

    let mut buf = *b"msg";
    let tag = aead.encrypt_in_place(&nonce, b"correct", &mut buf);

    let result = aead.decrypt_in_place(&nonce, b"wrong", &mut buf, &tag);
    assert!(result.is_err());
  }

  #[test]
  fn wrong_nonce_fails() {
    let key = AsconAead128Key::from_bytes([9; 16]);
    let nonce = Nonce128::from_bytes([10; 16]);
    let aead = AsconAead128::new(&key);

    let mut buf = *b"nonce test";
    let tag = aead.encrypt_in_place(&nonce, b"aad", &mut buf);

    let wrong_nonce = Nonce128::from_bytes([11; 16]);
    let result = aead.decrypt_in_place(&wrong_nonce, b"aad", &mut buf, &tag);
    assert!(result.is_err());
  }

  #[test]
  fn combined_encrypt_decrypt_round_trip() {
    let key = AsconAead128Key::from_bytes([7; 16]);
    let nonce = Nonce128::from_bytes([8; 16]);
    let aead = AsconAead128::new(&key);
    let pt = b"combined mode";

    let mut sealed = vec![0u8; pt.len().strict_add(TAG_SIZE)];
    aead.encrypt(&nonce, b"h", pt.as_slice(), &mut sealed).unwrap();

    let mut opened = vec![0u8; pt.len()];
    aead.decrypt(&nonce, b"h", &sealed, &mut opened).unwrap();
    assert_eq!(&opened, &pt[..]);
  }

  #[test]
  fn tag_from_slice_rejects_wrong_length() {
    assert!(AsconAead128::tag_from_slice(&[0u8; 15]).is_err());
    assert!(AsconAead128::tag_from_slice(&[0u8; 17]).is_err());
    assert!(AsconAead128::tag_from_slice(&[0u8; 16]).is_ok());
  }

  #[test]
  fn multi_block_round_trip() {
    let key = AsconAead128Key::from_bytes([0xAB; 16]);
    let nonce = Nonce128::from_bytes([0xCD; 16]);
    let aead = AsconAead128::new(&key);

    // 100 bytes = 12 full blocks + 4-byte tail.
    let plaintext = [0x77u8; 100];
    let mut buf = plaintext;
    let tag = aead.encrypt_in_place(&nonce, b"multi-block aad that is longer than one rate block", &mut buf);
    aead
      .decrypt_in_place(
        &nonce,
        b"multi-block aad that is longer than one rate block",
        &mut buf,
        &tag,
      )
      .unwrap();
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
    let tag = aead.encrypt_in_place(&nonce, b"", &mut buf);
    aead.decrypt_in_place(&nonce, b"", &mut buf, &tag).unwrap();
    assert_eq!(buf, plaintext);

    // Exactly 16 bytes = 2 full blocks, 0-byte tail.
    let plaintext16 = [0x66u8; 16];
    let mut buf16 = plaintext16;
    let tag16 = aead.encrypt_in_place(&nonce, b"", &mut buf16);
    aead.decrypt_in_place(&nonce, b"", &mut buf16, &tag16).unwrap();
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
    let aad: Vec<u8> = (0..48).map(|i| i as u8).collect();
    let pt: Vec<u8> = (0u8..97).map(|i| i.wrapping_mul(17)).collect();
    assert_matches_oracle(key, nonce, &aad, &pt);
  }
}
