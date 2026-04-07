#![allow(clippy::indexing_slicing)]

//! Ascon-AEAD128 authenticated encryption (NIST SP 800-232).
//!
//! Pure Rust, constant-time, `no_std` implementation with 128-bit key,
//! 128-bit nonce, and 128-bit authentication tag.

use core::fmt;

use super::{AeadBufferError, Nonce128, OpenError};
use crate::{
  hashes::crypto::ascon::{permute_6_portable, permute_12_portable},
  traits::{Aead, VerificationError, ct},
};

const KEY_SIZE: usize = 16;
const NONCE_SIZE: usize = Nonce128::LENGTH;
const TAG_SIZE: usize = 16;
const RATE: usize = 8;

/// Ascon-AEAD128 IV (SP 800-232): k=128, r=64, a=12, b=6.
const IV: u64 = 0x80400c0600000000;

/// Big-endian padding: set bit 7 of byte position `n` (0 ≤ n < RATE).
#[inline(always)]
const fn pad(n: usize) -> u64 {
  let shift = 8u32.strict_mul(7u32.strict_sub(n as u32));
  0x80_u64 << shift
}

/// Big-endian mask covering the first `n` bytes (MSB-aligned).
#[inline(always)]
const fn byte_mask(n: usize) -> u64 {
  if n == 0 {
    return 0;
  }
  let shift = 8u32.strict_mul(8u32.strict_sub(n as u32));
  u64::MAX << shift
}

/// Load up to 8 bytes big-endian into a u64, zero-padding on the right.
#[inline(always)]
fn load_bytes(data: &[u8]) -> u64 {
  debug_assert!(data.len() <= 8);
  let mut buf = [0u8; 8];
  buf[..data.len()].copy_from_slice(data);
  u64::from_be_bytes(buf)
}

// ---------------------------------------------------------------------------
// Key
// ---------------------------------------------------------------------------

/// Ascon-AEAD128 128-bit secret key.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AsconAead128Key([u8; Self::LENGTH]);

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

  /// Key halves as big-endian u64 words.
  #[inline]
  fn words(&self) -> (u64, u64) {
    let mut hi = [0u8; 8];
    let mut lo = [0u8; 8];
    hi.copy_from_slice(&self.0[..8]);
    lo.copy_from_slice(&self.0[8..]);
    (u64::from_be_bytes(hi), u64::from_be_bytes(lo))
  }
}

impl Default for AsconAead128Key {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl AsRef<[u8]> for AsconAead128Key {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

impl fmt::Debug for AsconAead128Key {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("AsconAead128Key(****)")
  }
}

impl AsconAead128Key {
  /// Construct a key by filling bytes from the provided closure.
  ///
  /// ```ignore
  /// let key = AsconAead128Key::generate(|buf| getrandom::fill(buf).unwrap());
  /// ```
  #[inline]
  #[must_use]
  pub fn generate(fill: impl FnOnce(&mut [u8; Self::LENGTH])) -> Self {
    let mut bytes = [0u8; Self::LENGTH];
    fill(&mut bytes);
    Self(bytes)
  }
}

impl_hex_fmt_secret!(AsconAead128Key);

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

impl fmt::Debug for AsconAead128Tag {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "AsconAead128Tag(")?;
    crate::hex::fmt_hex_lower(&self.0, f)?;
    write!(f, ")")
  }
}

impl_hex_fmt!(AsconAead128Tag);

// ---------------------------------------------------------------------------
// AEAD
// ---------------------------------------------------------------------------

/// Ascon-AEAD128 authenticated encryption with associated data.
///
/// NIST SP 800-232 lightweight AEAD with a 128-bit key, 128-bit nonce,
/// and 128-bit authentication tag. Built on the Ascon permutation with
/// rate = 64 bits, PA = 12 rounds, PB = 6 rounds.
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
/// aead.decrypt_in_place(&nonce, b"", &mut buf, &tag).unwrap();
/// assert_eq!(&buf, b"hello");
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
    let (k_hi, k_lo) = self.key.words();

    let n = nonce.as_bytes();
    let mut n_hi_buf = [0u8; 8];
    let mut n_lo_buf = [0u8; 8];
    n_hi_buf.copy_from_slice(&n[..8]);
    n_lo_buf.copy_from_slice(&n[8..]);
    let n_hi = u64::from_be_bytes(n_hi_buf);
    let n_lo = u64::from_be_bytes(n_lo_buf);

    let mut s = [IV, k_hi, k_lo, n_hi, n_lo];
    permute_12_portable(&mut s);
    s[3] ^= k_hi;
    s[4] ^= k_lo;
    s
  }

  /// Absorb associated data into the state.
  fn process_aad(s: &mut [u64; 5], aad: &[u8]) {
    if !aad.is_empty() {
      let mut chunks = aad.chunks_exact(RATE);
      for chunk in chunks.by_ref() {
        let mut block = [0u8; RATE];
        block.copy_from_slice(chunk);
        s[0] ^= u64::from_be_bytes(block);
        permute_6_portable(s);
      }

      // Last partial block (padded). Empty remainder is handled correctly:
      // load_bytes returns 0, pad(0) sets the MSB.
      let rest = chunks.remainder();
      s[0] ^= load_bytes(rest);
      s[0] ^= pad(rest.len());
      permute_6_portable(s);
    }

    // Domain separator (unconditional).
    s[4] ^= 1;
  }

  /// Finalize the state and extract the 128-bit tag.
  fn finalize(&self, s: &mut [u64; 5]) -> [u8; TAG_SIZE] {
    let (k_hi, k_lo) = self.key.words();

    s[1] ^= k_hi;
    s[2] ^= k_lo;
    permute_12_portable(s);
    s[3] ^= k_hi;
    s[4] ^= k_lo;

    let mut tag = [0u8; TAG_SIZE];
    tag[..8].copy_from_slice(&s[3].to_be_bytes());
    tag[8..].copy_from_slice(&s[4].to_be_bytes());
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

    let full_blocks = buffer.len() / RATE;
    let tail_start = full_blocks.strict_mul(RATE);

    // Encrypt full blocks.
    for i in 0..full_blocks {
      let start = i.strict_mul(RATE);
      let end = start.strict_add(RATE);
      let mut block = [0u8; RATE];
      block.copy_from_slice(&buffer[start..end]);
      s[0] ^= u64::from_be_bytes(block);
      block = s[0].to_be_bytes();
      buffer[start..end].copy_from_slice(&block);
      permute_6_portable(&mut s);
    }

    // Last partial block.
    let tail_len = buffer.len().strict_sub(tail_start);
    let pt_tail = load_bytes(&buffer[tail_start..]);
    s[0] ^= pt_tail;
    let ct_bytes = s[0].to_be_bytes();
    buffer[tail_start..].copy_from_slice(&ct_bytes[..tail_len]);
    s[0] ^= pad(tail_len);

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

    let full_blocks = buffer.len() / RATE;
    let tail_start = full_blocks.strict_mul(RATE);

    // Decrypt full blocks.
    for i in 0..full_blocks {
      let start = i.strict_mul(RATE);
      let end = start.strict_add(RATE);
      let mut block = [0u8; RATE];
      block.copy_from_slice(&buffer[start..end]);
      let ct_word = u64::from_be_bytes(block);
      block = (s[0] ^ ct_word).to_be_bytes();
      buffer[start..end].copy_from_slice(&block);
      s[0] = ct_word;
      permute_6_portable(&mut s);
    }

    // Last partial block.
    let tail_len = buffer.len().strict_sub(tail_start);
    let ct_word = load_bytes(&buffer[tail_start..]);
    let pt_bytes = (s[0] ^ ct_word).to_be_bytes();
    buffer[tail_start..].copy_from_slice(&pt_bytes[..tail_len]);
    // Reconstruct the state: replace rate bytes with ciphertext, add padding.
    s[0] &= !byte_mask(tail_len);
    s[0] |= ct_word;
    s[0] ^= pad(tail_len);

    // Finalize and verify.
    let expected = self.finalize(&mut s);
    if !ct::constant_time_eq(&expected, tag.as_bytes()) {
      // Zeroize unverified plaintext before returning.
      ct::zeroize(buffer);
      return Err(VerificationError::new());
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
      .step_by(2)
      .map(|i| u8::from_str_radix(&s[i..i.strict_add(2)], 16).unwrap())
      .collect()
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

  // -- Official test vectors (Ascon-128 NIST LWC KAT) --
  //
  // Source: ascon/ascon-c crypto_aead/asconaead128/LWC_AEAD_KAT_128_128.txt
  //
  // genkat format:
  //   Key   = 000102030405060708090A0B0C0D0E0F  (offset 0x00)
  //   Nonce = 101112131415161718191A1B1C1D1E1F  (offset 0x10)
  //   PT    = 20, 2021, 202122, ...             (offset 0x20)
  //   AD    = 30, 3031, 303132, ...             (offset 0x30)
  //   CT    = ciphertext || 16-byte tag

  fn kat_key() -> AsconAead128Key {
    AsconAead128Key::from_bytes([
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    ])
  }

  fn kat_nonce() -> Nonce128 {
    Nonce128::from_bytes([
      0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
    ])
  }

  fn kat_verify(pt: &[u8], ad: &[u8], expected_ct_tag: &str) {
    let aead = AsconAead128::new(&kat_key());
    let nonce = kat_nonce();
    let expected = hex(expected_ct_tag);
    let ct_len = pt.len();

    // Encrypt and check ciphertext + tag.
    let mut buf = pt.to_vec();
    let tag = aead.encrypt_in_place(&nonce, ad, &mut buf);
    assert_eq!(&buf, &expected[..ct_len], "ciphertext mismatch");
    assert_eq!(tag.as_bytes(), &expected[ct_len..], "tag mismatch");

    // Decrypt and verify round-trip.
    aead.decrypt_in_place(&nonce, ad, &mut buf, &tag).unwrap();
    assert_eq!(&buf, pt, "plaintext recovery mismatch");
  }

  /// Count 1: empty PT, empty AD.
  #[test]
  fn kat_count_1() {
    kat_verify(b"", b"", "38CCA290D1F2EF3DF9C8531946499037");
  }

  /// Count 2: empty PT, 1-byte AD.
  #[test]
  fn kat_count_2() {
    kat_verify(b"", &[0x30], "01FF81046C47E4D78D0389A321FFD48E");
  }

  /// Count 9: empty PT, 8-byte AD (one full rate block).
  #[test]
  fn kat_count_9() {
    let ad: Vec<u8> = (0x30..0x38).collect();
    kat_verify(b"", &ad, "E24B128B544E22871026615786C75E9E");
  }

  /// Count 17: empty PT, 16-byte AD (two rate blocks).
  #[test]
  fn kat_count_17() {
    let ad: Vec<u8> = (0x30..0x40).collect();
    kat_verify(b"", &ad, "5C1472BC958B7CEB24FCEBA70C81297F");
  }

  /// Count 33: empty PT, 32-byte AD (four rate blocks).
  #[test]
  fn kat_count_33() {
    let ad: Vec<u8> = (0x30..0x50).collect();
    kat_verify(b"", &ad, "016B13352D9202913C7CC98F9AC11659");
  }

  /// Count 34: 1-byte PT, empty AD.
  #[test]
  fn kat_count_34() {
    kat_verify(&[0x20], b"", "EB9B92A2B8149DE23C431A5FEC6E110422");
  }

  /// Count 35: 1-byte PT, 1-byte AD.
  #[test]
  fn kat_count_35() {
    kat_verify(&[0x20], &[0x30], "5E89608365697A23A0377D13FDBA0EF80E");
  }

  /// Count 67: 2-byte PT, empty AD.
  #[test]
  fn kat_count_67() {
    kat_verify(&[0x20, 0x21], b"", "EB856FACB4BC06CE8A41F20478F1719EE3BE");
  }

  /// Count 265: 8-byte PT (one full rate block), empty AD.
  #[test]
  fn kat_count_265() {
    let pt: Vec<u8> = (0x20..0x28).collect();
    kat_verify(&pt, b"", "EB851929173E2CC2ACD21F198E532C9F15EBF55DEC80AAE2");
  }

  /// Count 281: 8-byte PT, 16-byte AD (both multi-block).
  #[test]
  fn kat_count_281() {
    let pt: Vec<u8> = (0x20..0x28).collect();
    let ad: Vec<u8> = (0x30..0x40).collect();
    kat_verify(&pt, &ad, "42CD8C3F62DECC6BBBDCE06638AAF57D7C17C4438C6251EF");
  }
}
