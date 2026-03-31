#![allow(clippy::indexing_slicing)]

//! AES-256-GCM-SIV public AEAD surface (RFC 8452).

use core::fmt;

use super::{AeadBufferError, Nonce96, OpenError, aes, polyval};
use crate::traits::{Aead, VerificationError, ct};

const KEY_SIZE: usize = 32;
const TAG_SIZE: usize = 16;
const NONCE_SIZE: usize = Nonce96::LENGTH;

/// Maximum plaintext length: 2^36 - 32 bytes (per RFC 8452 §5).
/// Beyond this the 32-bit CTR counter wraps, causing keystream reuse.
const MAX_PLAINTEXT_LEN: u64 = (1u64 << 36).strict_sub(32);

/// AES-256-GCM-SIV secret key (32 bytes).
#[derive(Clone)]
pub struct Aes256GcmSivKey([u8; Self::LENGTH]);

impl PartialEq for Aes256GcmSivKey {
  fn eq(&self, other: &Self) -> bool {
    ct::constant_time_eq(&self.0, &other.0)
  }
}

impl Eq for Aes256GcmSivKey {}

impl Aes256GcmSivKey {
  /// Key length in bytes.
  pub const LENGTH: usize = KEY_SIZE;

  /// Construct a typed key from raw bytes.
  #[inline]
  #[must_use]
  pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
    Self(bytes)
  }

  /// Return the key bytes.
  #[inline]
  #[must_use]
  pub fn to_bytes(&self) -> [u8; Self::LENGTH] {
    self.0
  }

  /// Borrow the key bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }
}

impl Default for Aes256GcmSivKey {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl AsRef<[u8]> for Aes256GcmSivKey {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

impl fmt::Debug for Aes256GcmSivKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Aes256GcmSivKey").finish_non_exhaustive()
  }
}

impl Drop for Aes256GcmSivKey {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

/// AES-256-GCM-SIV authentication tag (16 bytes).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Aes256GcmSivTag([u8; Self::LENGTH]);

impl Aes256GcmSivTag {
  /// Tag length in bytes.
  pub const LENGTH: usize = TAG_SIZE;

  /// Construct a typed tag from raw bytes.
  #[inline]
  #[must_use]
  pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
    Self(bytes)
  }

  /// Return the tag bytes.
  #[inline]
  #[must_use]
  pub const fn to_bytes(self) -> [u8; Self::LENGTH] {
    self.0
  }

  /// Borrow the tag bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }
}

impl Default for Aes256GcmSivTag {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl AsRef<[u8]> for Aes256GcmSivTag {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

impl fmt::Debug for Aes256GcmSivTag {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_tuple("Aes256GcmSivTag").field(&self.0).finish()
  }
}

/// AES-256-GCM-SIV AEAD (RFC 8452).
///
/// Nonce-misuse resistant authenticated encryption. On nonce reuse, only
/// the authentication guarantee degrades — confidentiality is preserved
/// up to a multi-message distinguishing bound.
#[derive(Clone)]
pub struct Aes256GcmSiv {
  key: Aes256GcmSivKey,
}

impl fmt::Debug for Aes256GcmSiv {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Aes256GcmSiv").finish_non_exhaustive()
  }
}

impl Aes256GcmSiv {
  /// Key length in bytes.
  pub const KEY_SIZE: usize = KEY_SIZE;

  /// Nonce length in bytes.
  pub const NONCE_SIZE: usize = NONCE_SIZE;

  /// Tag length in bytes.
  pub const TAG_SIZE: usize = TAG_SIZE;

  /// Construct a new AES-256-GCM-SIV instance from `key`.
  #[inline]
  #[must_use]
  pub fn new(key: &Aes256GcmSivKey) -> Self {
    <Self as Aead>::new(key)
  }

  /// Rebuild a typed tag from raw tag bytes.
  #[inline]
  pub fn tag_from_slice(bytes: &[u8]) -> Result<Aes256GcmSivTag, AeadBufferError> {
    <Self as Aead>::tag_from_slice(bytes)
  }

  /// Encrypt `buffer` in place and return the detached authentication tag.
  #[inline]
  #[must_use]
  pub fn encrypt_in_place(&self, nonce: &Nonce96, aad: &[u8], buffer: &mut [u8]) -> Aes256GcmSivTag {
    <Self as Aead>::encrypt_in_place(self, nonce, aad, buffer)
  }

  /// Decrypt `buffer` in place and verify the detached authentication tag.
  #[inline]
  pub fn decrypt_in_place(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Aes256GcmSivTag,
  ) -> Result<(), VerificationError> {
    <Self as Aead>::decrypt_in_place(self, nonce, aad, buffer, tag)
  }

  /// Encrypt `plaintext` into `out` as `ciphertext || tag`.
  #[inline]
  pub fn encrypt(&self, nonce: &Nonce96, aad: &[u8], plaintext: &[u8], out: &mut [u8]) -> Result<(), AeadBufferError> {
    <Self as Aead>::encrypt(self, nonce, aad, plaintext, out)
  }

  /// Decrypt a combined `ciphertext || tag` into `out`.
  #[inline]
  pub fn decrypt(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    ciphertext_and_tag: &[u8],
    out: &mut [u8],
  ) -> Result<(), OpenError> {
    <Self as Aead>::decrypt(self, nonce, aad, ciphertext_and_tag, out)
  }
}

// ---------------------------------------------------------------------------
// RFC 8452 construction internals
// ---------------------------------------------------------------------------

/// Derive per-message authentication and encryption keys from the master key
/// and nonce (RFC 8452 §4, AES-256 variant).
///
/// Returns (auth_key [16 bytes], enc_key [32 bytes]).
#[inline]
fn derive_keys(master_key: &[u8; KEY_SIZE], nonce: &Nonce96) -> ([u8; 16], [u8; 32]) {
  let ek = aes::aes256_expand_key(master_key);
  let nonce_bytes = nonce.as_bytes();

  let mut auth_key = [0u8; 16];
  let mut enc_key = [0u8; 32];

  // 6 AES block encryptions: counter (LE32) || nonce (96 bits)
  let mut counter_block = [0u8; 16];
  counter_block[4..16].copy_from_slice(nonce_bytes);

  let mut i = 0u32;
  while i < 6 {
    counter_block[0..4].copy_from_slice(&i.to_le_bytes());
    let mut block = counter_block;
    aes::aes256_encrypt_block(&ek, &mut block);

    // Take only the first 8 bytes of each encrypted block.
    match i {
      0 => auth_key[0..8].copy_from_slice(&block[0..8]),
      1 => auth_key[8..16].copy_from_slice(&block[0..8]),
      2 => enc_key[0..8].copy_from_slice(&block[0..8]),
      3 => enc_key[8..16].copy_from_slice(&block[0..8]),
      4 => enc_key[16..24].copy_from_slice(&block[0..8]),
      5 => enc_key[24..32].copy_from_slice(&block[0..8]),
      _ => unreachable!(),
    }
    ct::zeroize(&mut block);
    i = i.strict_add(1);
  }

  (auth_key, enc_key)
}

/// Compute the POLYVAL-based authentication tag (RFC 8452 §5 steps 1-3).
///
/// Accepts the pre-expanded encryption key to avoid redundant key expansion
/// (the same expanded key is reused for AES-CTR in the caller).
#[inline]
fn compute_tag(
  auth_key: &[u8; 16],
  enc_ek: &aes::Aes256EncKey,
  nonce: &Nonce96,
  aad: &[u8],
  plaintext: &[u8],
) -> [u8; TAG_SIZE] {
  // POLYVAL over: padded AAD || padded plaintext || length block
  let mut pv = polyval::Polyval::new(auth_key);

  // Process AAD (padded to 16-byte blocks).
  pv.update_padded(aad);

  // Process plaintext (padded to 16-byte blocks).
  pv.update_padded(plaintext);

  // Length block: [aad_bits as u64 LE || plaintext_bits as u64 LE].
  let aad_bits = (aad.len() as u64).strict_mul(8);
  let pt_bits = (plaintext.len() as u64).strict_mul(8);
  let mut length_block = [0u8; 16];
  length_block[0..8].copy_from_slice(&aad_bits.to_le_bytes());
  length_block[8..16].copy_from_slice(&pt_bits.to_le_bytes());
  pv.update_block(&length_block);

  let mut s = pv.finalize();

  // XOR nonce into the first 12 bytes.
  let nonce_bytes = nonce.as_bytes();
  let mut j = 0usize;
  while j < 12 {
    s[j] ^= nonce_bytes[j];
    j = j.strict_add(1);
  }

  // Clear the MSB of the last byte.
  s[15] &= 0x7f;

  // Encrypt with AES to get the tag.
  aes::aes256_encrypt_block(enc_ek, &mut s);

  s
}

// ---------------------------------------------------------------------------
// Aead trait implementation
// ---------------------------------------------------------------------------

impl Aead for Aes256GcmSiv {
  const KEY_SIZE: usize = KEY_SIZE;
  const NONCE_SIZE: usize = NONCE_SIZE;
  const TAG_SIZE: usize = TAG_SIZE;

  type Key = Aes256GcmSivKey;
  type Nonce = Nonce96;
  type Tag = Aes256GcmSivTag;

  fn new(key: &Self::Key) -> Self {
    Self { key: key.clone() }
  }

  fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
    if bytes.len() != TAG_SIZE {
      return Err(AeadBufferError::new());
    }
    let mut tag = [0u8; TAG_SIZE];
    tag.copy_from_slice(bytes);
    Ok(Aes256GcmSivTag::from_bytes(tag))
  }

  fn encrypt_in_place(&self, nonce: &Self::Nonce, aad: &[u8], buffer: &mut [u8]) -> Self::Tag {
    assert!(
      (buffer.len() as u64) <= MAX_PLAINTEXT_LEN,
      "AES-256-GCM-SIV plaintext exceeds 2^36 - 32 bytes"
    );

    let (mut auth_key, mut enc_key) = derive_keys(self.key.as_bytes(), nonce);
    let ek = aes::aes256_expand_key(&enc_key);

    // Compute tag over plaintext (before encryption -- nonce-misuse resistance).
    let tag_bytes = compute_tag(&auth_key, &ek, nonce, aad, buffer);

    // AES-CTR encryption: counter_block = tag with MSB of byte 15 set.
    let mut counter_block = tag_bytes;
    counter_block[15] |= 0x80;
    aes::aes256_ctr32_encrypt(&ek, &counter_block, buffer);

    ct::zeroize(&mut auth_key);
    ct::zeroize(&mut enc_key);

    Aes256GcmSivTag::from_bytes(tag_bytes)
  }

  fn decrypt_in_place(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), VerificationError> {
    assert!(
      (buffer.len() as u64) <= MAX_PLAINTEXT_LEN,
      "AES-256-GCM-SIV ciphertext exceeds 2^36 - 32 bytes"
    );

    let (mut auth_key, mut enc_key) = derive_keys(self.key.as_bytes(), nonce);
    let ek = aes::aes256_expand_key(&enc_key);

    // AES-CTR decryption first (before verification -- SIV pattern).
    let mut counter_block = tag.0;
    counter_block[15] |= 0x80;
    aes::aes256_ctr32_encrypt(&ek, &counter_block, buffer);

    // Recompute expected tag over the decrypted plaintext.
    let expected = compute_tag(&auth_key, &ek, nonce, aad, buffer);

    ct::zeroize(&mut auth_key);
    ct::zeroize(&mut enc_key);

    if !ct::constant_time_eq(&expected, tag.as_bytes()) {
      // Verification failed -- zero the (unauthenticated) plaintext.
      ct::zeroize(buffer);
      return Err(VerificationError::new());
    }

    Ok(())
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  /// RFC 8452 Appendix C.2, test case 1: empty plaintext, empty AAD.
  #[test]
  fn aes256gcmsiv_empty() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let expected_ct_tag = hex_vec("07f5f4169bbf55a8400cd47ea6fd400f");

    let cipher = Aes256GcmSiv::new(&key);
    let mut out = vec![0u8; expected_ct_tag.len()];
    cipher.encrypt(&nonce, &[], &[], &mut out).unwrap();
    assert_eq!(out, expected_ct_tag);

    // Decrypt.
    let mut pt_out = vec![0u8; 0];
    cipher.decrypt(&nonce, &[], &expected_ct_tag, &mut pt_out).unwrap();
    assert!(pt_out.is_empty());
  }

  /// RFC 8452 Appendix C.2: AAD=01, plaintext=0200000000000000.
  #[test]
  fn aes256gcmsiv_aad_and_plaintext() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let aad = hex_vec("01");
    let plaintext = hex_vec("0200000000000000");
    let expected_ct_tag = hex_vec("1de22967237a813291213f267e3b452f02d01ae33e4ec854");

    let cipher = Aes256GcmSiv::new(&key);

    // Encrypt.
    let mut out = vec![0u8; plaintext.len().strict_add(TAG_SIZE)];
    cipher.encrypt(&nonce, &aad, &plaintext, &mut out).unwrap();
    assert_eq!(out, expected_ct_tag);

    // Decrypt.
    let mut pt_out = vec![0u8; plaintext.len()];
    cipher.decrypt(&nonce, &aad, &expected_ct_tag, &mut pt_out).unwrap();
    assert_eq!(pt_out, plaintext);
  }

  /// RFC 8452 Appendix C.2: longer AAD and plaintext.
  #[test]
  fn aes256gcmsiv_longer_aad_and_plaintext() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let aad = hex_vec("010000000000000000000000000000000200");
    let plaintext = hex_vec("0300000000000000000000000000000004000000");
    let expected_ct_tag = hex_vec("43dd0163cdb48f9fe3212bf61b201976067f342bb879ad976d8242acc188ab59cabfe307");

    let cipher = Aes256GcmSiv::new(&key);

    let mut out = vec![0u8; plaintext.len().strict_add(TAG_SIZE)];
    cipher.encrypt(&nonce, &aad, &plaintext, &mut out).unwrap();
    assert_eq!(out, expected_ct_tag);

    let mut pt_out = vec![0u8; plaintext.len()];
    cipher.decrypt(&nonce, &aad, &expected_ct_tag, &mut pt_out).unwrap();
    assert_eq!(pt_out, plaintext);
  }

  /// Decryption with wrong tag should fail.
  #[test]
  fn aes256gcmsiv_bad_tag() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let mut bad_ct_tag = hex_vec("07f5f4169bbf55a8400cd47ea6fd400f");
    // Flip a bit in the tag.
    bad_ct_tag[0] ^= 1;

    let cipher = Aes256GcmSiv::new(&key);
    let mut pt_out = vec![0u8; 0];
    let result = cipher.decrypt(&nonce, &[], &bad_ct_tag, &mut pt_out);
    assert!(result.is_err());
  }

  /// Decryption with wrong AAD should fail.
  #[test]
  fn aes256gcmsiv_wrong_aad_rejected() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let ct_tag = hex_vec("1de22967237a813291213f267e3b452f02d01ae33e4ec854");

    let cipher = Aes256GcmSiv::new(&key);
    let mut pt_out = vec![0u8; 8]; // plaintext was 8 bytes
    // Wrong AAD: 0x02 instead of 0x01.
    let result = cipher.decrypt(&nonce, &[0x02], &ct_tag, &mut pt_out);
    assert!(result.is_err());
  }

  /// Decryption with wrong nonce should fail.
  #[test]
  fn aes256gcmsiv_wrong_nonce_rejected() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let aad = hex_vec("01");
    let ct_tag = hex_vec("1de22967237a813291213f267e3b452f02d01ae33e4ec854");

    let cipher = Aes256GcmSiv::new(&key);
    let mut pt_out = vec![0u8; 8]; // plaintext was 8 bytes
    // Wrong nonce: 0x04 instead of 0x03.
    let wrong_nonce = Nonce96::from_bytes(hex12("040000000000000000000000"));
    let result = cipher.decrypt(&wrong_nonce, &aad, &ct_tag, &mut pt_out);
    assert!(result.is_err());
  }

  /// Ciphertext body tampering should fail verification.
  #[test]
  fn aes256gcmsiv_ciphertext_tampering_rejected() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let aad = hex_vec("01");
    let plaintext = hex_vec("0200000000000000");
    let mut ct_tag = hex_vec("1de22967237a813291213f267e3b452f02d01ae33e4ec854");

    // Flip a bit in the ciphertext body (not the tag).
    ct_tag[0] ^= 1;

    let cipher = Aes256GcmSiv::new(&key);
    let mut pt_out = vec![0u8; plaintext.len()];
    let result = cipher.decrypt(&nonce, &aad, &ct_tag, &mut pt_out);
    assert!(result.is_err());
  }

  /// On authentication failure, the output buffer must be zeroed.
  #[test]
  fn aes256gcmsiv_buffer_zeroed_on_auth_failure() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let aad = hex_vec("01");
    let plaintext = hex_vec("0200000000000000");

    let cipher = Aes256GcmSiv::new(&key);
    let mut out = vec![0u8; plaintext.len().strict_add(TAG_SIZE)];
    cipher.encrypt(&nonce, &aad, &plaintext, &mut out).unwrap();

    // Corrupt the tag.
    let last = out.len().strict_sub(1);
    out[last] ^= 0xff;

    let mut pt_out = vec![0xffu8; plaintext.len()]; // fill with non-zero
    let result = cipher.decrypt(&nonce, &aad, &out, &mut pt_out);
    assert!(result.is_err());
    // Buffer must be zeroed even though decryption was attempted.
    assert!(pt_out.iter().all(|&b| b == 0), "buffer not zeroed on auth failure");
  }

  /// Detached encrypt/decrypt round-trip.
  #[test]
  fn aes256gcmsiv_detached_round_trip() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    ));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let aad = hex_vec("01");
    let plaintext = hex_vec("0200000000000000");

    let cipher = Aes256GcmSiv::new(&key);

    let mut buf = plaintext.clone();
    let tag = cipher.encrypt_in_place(&nonce, &aad, &mut buf);

    // buf is now ciphertext, tag is separate.
    assert_ne!(buf, plaintext);

    cipher.decrypt_in_place(&nonce, &aad, &mut buf, &tag).unwrap();
    assert_eq!(buf, plaintext);
  }

  /// `tag_from_slice` rejects wrong-length input.
  #[test]
  fn aes256gcmsiv_tag_from_slice_rejects_bad_length() {
    assert!(Aes256GcmSiv::tag_from_slice(&[0u8; 15]).is_err());
    assert!(Aes256GcmSiv::tag_from_slice(&[0u8; 17]).is_err());
    assert!(Aes256GcmSiv::tag_from_slice(&[0u8; 0]).is_err());
    assert!(Aes256GcmSiv::tag_from_slice(&[0u8; 16]).is_ok());
  }

  /// RFC 8452 Appendix C.2 vector with a different key (empty PT+AAD).
  #[test]
  fn aes256gcmsiv_different_key_vector() {
    let key = Aes256GcmSivKey::from_bytes(hex32(
      "e66021d5eb8e4f4066d4adb9c33560e4f46e44bb3da0015c94f7088736864200",
    ));
    let nonce = Nonce96::from_bytes(hex12("e0eaf5284d884a0e77d31646"));
    let expected_ct_tag = hex_vec("169fbb2fbf389a995f6390af22228a62");

    let cipher = Aes256GcmSiv::new(&key);

    let mut out = vec![0u8; expected_ct_tag.len()];
    cipher.encrypt(&nonce, &[], &[], &mut out).unwrap();
    assert_eq!(out, expected_ct_tag);

    let mut pt_out = vec![0u8; 0];
    cipher.decrypt(&nonce, &[], &expected_ct_tag, &mut pt_out).unwrap();
  }

  // --- Hex helpers ---

  fn hex32(hex: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..32 {
      out[i] = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
    }
    out
  }

  fn hex12(hex: &str) -> [u8; 12] {
    let mut out = [0u8; 12];
    for i in 0..12 {
      out[i] = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
    }
    out
  }

  fn hex_vec(hex: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(hex.len() / 2);
    let mut i = 0;
    while i < hex.len() {
      out.push(u8::from_str_radix(&hex[i..i + 2], 16).unwrap());
      i += 2;
    }
    out
  }
}
