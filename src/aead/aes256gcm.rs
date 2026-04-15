#![allow(clippy::indexing_slicing)]

//! AES-256-GCM public AEAD surface (NIST SP 800-38D).

use core::fmt;

use super::{AeadBufferError, Nonce96, OpenError, aes, ghash, polyval};
use crate::traits::{Aead, VerificationError, ct};

const KEY_SIZE: usize = 32;
const TAG_SIZE: usize = 16;
const NONCE_SIZE: usize = Nonce96::LENGTH;

/// Maximum plaintext length per NIST SP 800-38D: 2^39 - 256 bits = 2^36 - 32 bytes.
/// In practice the portable CTR uses a 32-bit counter, limiting to (2^32 - 2) blocks.
const MAX_PLAINTEXT_LEN: u64 = ((1u64 << 32).strict_sub(2)).strict_mul(16); // ~64 GiB

/// AES-256-GCM secret key (32 bytes).
#[derive(Clone)]
pub struct Aes256GcmKey([u8; Self::LENGTH]);

impl PartialEq for Aes256GcmKey {
  fn eq(&self, other: &Self) -> bool {
    ct::constant_time_eq(&self.0, &other.0)
  }
}

impl Eq for Aes256GcmKey {}

impl Aes256GcmKey {
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

impl Default for Aes256GcmKey {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl AsRef<[u8]> for Aes256GcmKey {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

impl crate::traits::ConstantTimeEq for Aes256GcmKey {
  #[inline]
  fn ct_eq(&self, other: &Self) -> bool {
    crate::traits::ct::constant_time_eq(&self.0, &other.0)
  }
}

impl fmt::Debug for Aes256GcmKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("Aes256GcmKey(****)")
  }
}

impl Aes256GcmKey {
  /// Construct a key by filling bytes from the provided closure.
  ///
  /// ```rust
  /// # use rscrypto::Aes256GcmKey;
  /// let key = Aes256GcmKey::generate(|buf| buf.fill(0xA5));
  /// assert_eq!(key.as_bytes(), &[0xA5; Aes256GcmKey::LENGTH]);
  /// ```
  #[inline]
  #[must_use]
  pub fn generate(fill: impl FnOnce(&mut [u8; Self::LENGTH])) -> Self {
    let mut bytes = [0u8; Self::LENGTH];
    fill(&mut bytes);
    Self(bytes)
  }

  /// Generate a random key using the operating system's CSPRNG.
  ///
  /// # Panics
  ///
  /// Panics if the platform entropy source is unavailable.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[inline]
  #[must_use]
  pub fn random() -> Self {
    let mut bytes = [0u8; Self::LENGTH];
    match getrandom::fill(&mut bytes) {
      Ok(()) => {}
      Err(e) => panic!("getrandom failed: {e}"),
    }
    Self(bytes)
  }
}

impl_hex_fmt_secret!(Aes256GcmKey);
impl_serde_bytes!(Aes256GcmKey);

impl Drop for Aes256GcmKey {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

/// AES-256-GCM authentication tag (16 bytes).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Aes256GcmTag([u8; Self::LENGTH]);

impl Aes256GcmTag {
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

impl Default for Aes256GcmTag {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl AsRef<[u8]> for Aes256GcmTag {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

impl crate::traits::ConstantTimeEq for Aes256GcmTag {
  #[inline]
  fn ct_eq(&self, other: &Self) -> bool {
    crate::traits::ct::constant_time_eq(&self.0, &other.0)
  }
}

impl fmt::Debug for Aes256GcmTag {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Aes256GcmTag(")?;
    crate::hex::fmt_hex_lower(&self.0, f)?;
    write!(f, ")")
  }
}

impl_hex_fmt!(Aes256GcmTag);
impl_serde_bytes!(Aes256GcmTag);

/// AES-256-GCM AEAD (NIST SP 800-38D).
///
/// The standard TLS / interop workhorse. Requires strict nonce uniqueness:
/// reusing a nonce with the same key leaks the GHASH key, enabling forgery.
/// For new designs where nonce discipline is uncertain, prefer
/// `Aes256GcmSiv`.
#[derive(Clone)]
pub struct Aes256Gcm {
  /// Pre-expanded AES-256 round keys.
  ek: aes::Aes256EncKey,
  /// GHASH hash key H = AES_K(0^128).
  h: [u8; 16],
  /// Precomputed H powers [H^4, H^3, H^2, H] in the POLYVAL domain
  /// for 4-block wide GHASH processing.
  h_powers_rev: [u128; 4],
}

impl fmt::Debug for Aes256Gcm {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Aes256Gcm").finish_non_exhaustive()
  }
}

impl Aes256Gcm {
  /// Key length in bytes.
  pub const KEY_SIZE: usize = KEY_SIZE;

  /// Nonce length in bytes.
  pub const NONCE_SIZE: usize = NONCE_SIZE;

  /// Tag length in bytes.
  pub const TAG_SIZE: usize = TAG_SIZE;

  /// Construct a new AES-256-GCM instance from `key`.
  #[inline]
  #[must_use]
  pub fn new(key: &Aes256GcmKey) -> Self {
    <Self as Aead>::new(key)
  }

  /// Rebuild a typed tag from raw tag bytes.
  #[inline]
  pub fn tag_from_slice(bytes: &[u8]) -> Result<Aes256GcmTag, AeadBufferError> {
    <Self as Aead>::tag_from_slice(bytes)
  }

  /// Encrypt `buffer` in place and return the detached authentication tag.
  #[inline]
  #[must_use]
  pub fn encrypt_in_place(&self, nonce: &Nonce96, aad: &[u8], buffer: &mut [u8]) -> Aes256GcmTag {
    <Self as Aead>::encrypt_in_place(self, nonce, aad, buffer)
  }

  /// Decrypt `buffer` in place and verify the detached authentication tag.
  #[inline]
  pub fn decrypt_in_place(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Aes256GcmTag,
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
// GCM construction internals (NIST SP 800-38D)
// ---------------------------------------------------------------------------

/// Build the initial counter block J0 for a 96-bit IV.
///
/// J0 = IV || 0x00000001 (NIST SP 800-38D § 7.1, when len(IV) = 96).
#[inline]
fn make_j0(nonce: &Nonce96) -> [u8; 16] {
  let mut j0 = [0u8; 16];
  j0[..12].copy_from_slice(nonce.as_bytes());
  j0[15] = 0x01; // Counter starts at 1.
  j0
}

/// Compute the GCM authentication tag.
///
/// tag = AES(K, J0) XOR GHASH(H, AAD, ciphertext)
///
/// GHASH processes: padded AAD || padded ciphertext || length block,
/// where the length block is [len(A) in bits as u64 BE || len(C) in bits as u64 BE].
#[inline]
fn compute_tag(ek: &aes::Aes256EncKey, h: &[u8; 16], j0: &[u8; 16], aad: &[u8], ciphertext: &[u8]) -> [u8; TAG_SIZE] {
  // GHASH(H, A, C)
  let mut gh = ghash::Ghash::new(h);

  // Process AAD (zero-padded to 128-bit blocks).
  gh.update_padded(aad);

  // Process ciphertext (zero-padded to 128-bit blocks).
  gh.update_padded(ciphertext);

  // Length block: [len(A) in bits as u64 BE || len(C) in bits as u64 BE].
  let aad_bits = (aad.len() as u64).strict_mul(8);
  let ct_bits = (ciphertext.len() as u64).strict_mul(8);
  let mut length_block = [0u8; 16];
  length_block[0..8].copy_from_slice(&aad_bits.to_be_bytes());
  length_block[8..16].copy_from_slice(&ct_bits.to_be_bytes());
  gh.update_block(&length_block);

  let mut s = gh.finalize();

  // tag = GHASH_result XOR AES(K, J0)
  let mut encrypted_j0 = *j0;
  aes::aes256_encrypt_block(ek, &mut encrypted_j0);

  let mut i = 0usize;
  while i < 16 {
    s[i] ^= encrypted_j0[i];
    i = i.strict_add(1);
  }

  ct::zeroize(&mut encrypted_j0);
  s
}

/// Compute the GCM authentication tag using 4-block wide GHASH.
///
/// Same semantics as `compute_tag` but processes ciphertext in 4-block
/// (64-byte) chunks via `accumulate_4blocks`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn compute_tag_wide(
  ek: &aes::Aes256EncKey,
  h: &[u8; 16],
  h_powers_rev: &[u128; 4],
  j0: &[u8; 16],
  aad: &[u8],
  ciphertext: &[u8],
) -> [u8; TAG_SIZE] {
  let h_polyval = ghash::h_to_polyval(h);

  // Process AAD (zero-padded, block by block — usually small).
  let mut acc = {
    let mut gh = ghash::Ghash::new(h);
    gh.update_padded(aad);
    gh.finalize_u128()
  };

  // Process ciphertext in 4-block wide chunks.
  let mut offset = 0usize;
  while offset.strict_add(64) <= ciphertext.len() {
    let mut blocks = [0u128; 4];
    let mut i = 0usize;
    while i < 4 {
      let base = offset.strict_add(i.strict_mul(16));
      let mut block = [0u8; 16];
      block.copy_from_slice(&ciphertext[base..base.strict_add(16)]);
      blocks[i] = u128::from_be_bytes(block);
      i = i.strict_add(1);
    }
    acc = polyval::accumulate_4blocks(acc, h_polyval, h_powers_rev, &blocks);
    offset = offset.strict_add(64);
  }

  // Remaining single blocks.
  while offset.strict_add(16) <= ciphertext.len() {
    let mut block = [0u8; 16];
    block.copy_from_slice(&ciphertext[offset..offset.strict_add(16)]);
    acc ^= u128::from_be_bytes(block);
    acc = polyval::clmul128_reduce(acc, h_polyval);
    offset = offset.strict_add(16);
  }

  // Partial tail block.
  let remaining = ciphertext.len().strict_sub(offset);
  if remaining > 0 {
    let mut block = [0u8; 16];
    block[..remaining].copy_from_slice(&ciphertext[offset..]);
    acc ^= u128::from_be_bytes(block);
    acc = polyval::clmul128_reduce(acc, h_polyval);
  }

  // Length block.
  let aad_bits = (aad.len() as u64).strict_mul(8);
  let ct_bits = (ciphertext.len() as u64).strict_mul(8);
  let mut length_block = [0u8; 16];
  length_block[0..8].copy_from_slice(&aad_bits.to_be_bytes());
  length_block[8..16].copy_from_slice(&ct_bits.to_be_bytes());
  acc ^= u128::from_be_bytes(length_block);
  acc = polyval::clmul128_reduce(acc, h_polyval);

  // Finalize: convert back to GHASH convention.
  let mut s = acc.to_be_bytes();

  // tag = GHASH_result XOR AES(K, J0)
  let mut encrypted_j0 = *j0;
  aes::aes256_encrypt_block(ek, &mut encrypted_j0);
  let mut i = 0usize;
  while i < 16 {
    s[i] ^= encrypted_j0[i];
    i = i.strict_add(1);
  }
  ct::zeroize(&mut encrypted_j0);
  s
}

/// Increment the 32-bit big-endian counter in bytes 12..15.
#[inline]
fn inc32(block: &mut [u8; 16]) {
  let ctr = u32::from_be_bytes([block[12], block[13], block[14], block[15]]);
  block[12..16].copy_from_slice(&ctr.wrapping_add(1).to_be_bytes());
}

// ---------------------------------------------------------------------------
// Aead trait implementation
// ---------------------------------------------------------------------------

impl Aead for Aes256Gcm {
  const KEY_SIZE: usize = KEY_SIZE;
  const NONCE_SIZE: usize = NONCE_SIZE;
  const TAG_SIZE: usize = TAG_SIZE;

  type Key = Aes256GcmKey;
  type Nonce = Nonce96;
  type Tag = Aes256GcmTag;

  fn new(key: &Self::Key) -> Self {
    let ek = aes::aes256_expand_key(&key.0);

    // H = AES_K(0^128)
    let mut h = [0u8; 16];
    aes::aes256_encrypt_block(&ek, &mut h);

    // Precompute H powers for 4-block wide GHASH.
    // GHASH loads H as BE, then applies mulX_POLYVAL to get the POLYVAL-domain key.
    let h_polyval = ghash::h_to_polyval(&h);
    let powers = polyval::precompute_powers(h_polyval);
    // Reverse: [H, H^2, H^3, H^4] -> [H^4, H^3, H^2, H]
    let h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];

    Self { ek, h, h_powers_rev }
  }

  fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
    if bytes.len() != TAG_SIZE {
      return Err(AeadBufferError::new());
    }
    let mut tag = [0u8; TAG_SIZE];
    tag.copy_from_slice(bytes);
    Ok(Aes256GcmTag::from_bytes(tag))
  }

  fn encrypt_in_place(&self, nonce: &Self::Nonce, aad: &[u8], buffer: &mut [u8]) -> Self::Tag {
    assert!(
      (buffer.len() as u64) <= MAX_PLAINTEXT_LEN,
      "AES-256-GCM plaintext exceeds maximum length"
    );

    let j0 = make_j0(nonce);
    let mut ctr_block = j0;
    inc32(&mut ctr_block);

    // Wide path: VAES-512 CTR + VPCLMULQDQ GHASH when available.
    #[cfg(target_arch = "x86_64")]
    {
      use crate::platform::caps;
      let c = crate::platform::caps();
      if c.has(caps::x86::VAES_READY) && c.has(caps::x86::VPCLMUL_READY) {
        // SAFETY: VAES + VPCLMULQDQ availability verified via CPUID.
        unsafe { aes::aes256_ctr32_encrypt_be_wide(&self.ek, &ctr_block, buffer) };
        let tag_bytes = compute_tag_wide(&self.ek, &self.h, &self.h_powers_rev, &j0, aad, buffer);
        return Aes256GcmTag::from_bytes(tag_bytes);
      }
    }

    // Scalar path: single-block AES-NI/CE/portable + single-block GHASH.
    aes::aes256_ctr32_encrypt_be(&self.ek, &ctr_block, buffer);
    let tag_bytes = compute_tag(&self.ek, &self.h, &j0, aad, buffer);
    Aes256GcmTag::from_bytes(tag_bytes)
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
      "AES-256-GCM ciphertext exceeds maximum length"
    );

    let j0 = make_j0(nonce);

    // Wide path: VPCLMULQDQ GHASH + VAES-512 CTR when available.
    #[cfg(target_arch = "x86_64")]
    {
      use crate::platform::caps;
      let c = crate::platform::caps();
      if c.has(caps::x86::VAES_READY) && c.has(caps::x86::VPCLMUL_READY) {
        // Verify tag BEFORE decryption (authenticate-then-decrypt).
        let expected = compute_tag_wide(&self.ek, &self.h, &self.h_powers_rev, &j0, aad, buffer);
        if !ct::constant_time_eq(&expected, tag.as_bytes()) {
          return Err(VerificationError::new());
        }
        let mut ctr_block = j0;
        inc32(&mut ctr_block);
        // SAFETY: VAES availability verified via CPUID.
        unsafe { aes::aes256_ctr32_encrypt_be_wide(&self.ek, &ctr_block, buffer) };
        return Ok(());
      }
    }

    // Scalar path: authenticate then decrypt.
    let expected = compute_tag(&self.ek, &self.h, &j0, aad, buffer);
    if !ct::constant_time_eq(&expected, tag.as_bytes()) {
      return Err(VerificationError::new());
    }
    let mut ctr_block = j0;
    inc32(&mut ctr_block);
    aes::aes256_ctr32_encrypt_be(&self.ek, &ctr_block, buffer);
    Ok(())
  }
}

// Aes256Gcm stores the expanded key which contains sensitive material.
impl Drop for Aes256Gcm {
  fn drop(&mut self) {
    ct::zeroize(&mut self.h);
    // SAFETY: [u128; 4] is layout-compatible with [u8; 64].
    ct::zeroize(unsafe { core::slice::from_raw_parts_mut(self.h_powers_rev.as_mut_ptr().cast::<u8>(), 64) });
    // ek's Drop is handled by Aes256EncKey's own Drop impl.
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use alloc::{vec, vec::Vec};

  use super::*;

  // NIST SP 800-38D Test Case 13: AES-256-GCM, empty plaintext, empty AAD.
  // Key:  0000...00 (32 bytes)
  // IV:   000000000000000000000000
  // PT:   (empty)
  // AAD:  (empty)
  // CT:   (empty)
  // Tag:  530f8afbc74536b9a963b4f1c4cb738b
  #[test]
  fn nist_test_case_13_empty() {
    let key = Aes256GcmKey::from_bytes([0u8; 32]);
    let nonce = Nonce96::from_bytes([0u8; 12]);

    let cipher = Aes256Gcm::new(&key);
    let expected_tag = hex16("530f8afbc74536b9a963b4f1c4cb738b");

    // Encrypt.
    let mut buf = vec![];
    let tag = cipher.encrypt_in_place(&nonce, &[], &mut buf);
    assert_eq!(tag.0, expected_tag, "Tag mismatch on encrypt");

    // Decrypt.
    cipher.decrypt_in_place(&nonce, &[], &mut buf, &tag).unwrap();
  }

  // NIST SP 800-38D Test Case 14: AES-256-GCM, 16-byte plaintext, empty AAD.
  // Key:  0000...00 (32 bytes)
  // IV:   000000000000000000000000
  // PT:   00000000000000000000000000000000
  // AAD:  (empty)
  // CT:   cea7403d4d606b6e074ec5d3baf39d18
  // Tag:  d0d1c8a799996bf0265b98b5d48ab919
  #[test]
  fn nist_test_case_14_one_block() {
    let key = Aes256GcmKey::from_bytes([0u8; 32]);
    let nonce = Nonce96::from_bytes([0u8; 12]);

    let cipher = Aes256Gcm::new(&key);
    let expected_ct = hex_vec("cea7403d4d606b6e074ec5d3baf39d18");
    let expected_tag = hex16("d0d1c8a799996bf0265b98b5d48ab919");

    // Encrypt.
    let mut buf = vec![0u8; 16];
    let tag = cipher.encrypt_in_place(&nonce, &[], &mut buf);
    assert_eq!(buf, expected_ct, "Ciphertext mismatch");
    assert_eq!(tag.0, expected_tag, "Tag mismatch");

    // Decrypt.
    cipher.decrypt_in_place(&nonce, &[], &mut buf, &tag).unwrap();
    assert_eq!(buf, vec![0u8; 16], "Plaintext mismatch after decrypt");
  }

  // NIST SP 800-38D Test Case 15: AES-256-GCM with non-zero key and longer data.
  // Key:  feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308
  // IV:   cafebabefacedbaddecaf888
  // PT:   d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255
  // AAD:  (empty)
  // CT:   522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662898015ad
  // Tag:  b094dac5d93471bdec1a502270e3cc6c
  #[test]
  fn nist_test_case_15_multi_block() {
    let key = Aes256GcmKey::from_bytes(hex32(
      "feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308",
    ));
    let nonce = Nonce96::from_bytes(hex12("cafebabefacedbaddecaf888"));
    let plaintext = hex_vec(
      "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255",
    );
    let expected_ct = hex_vec(
      "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662898015ad",
    );
    let expected_tag = hex16("b094dac5d93471bdec1a502270e3cc6c");

    let cipher = Aes256Gcm::new(&key);

    // Encrypt.
    let mut buf = plaintext.clone();
    let tag = cipher.encrypt_in_place(&nonce, &[], &mut buf);
    assert_eq!(buf, expected_ct, "Ciphertext mismatch");
    assert_eq!(tag.0, expected_tag, "Tag mismatch");

    // Decrypt.
    cipher.decrypt_in_place(&nonce, &[], &mut buf, &tag).unwrap();
    assert_eq!(buf, plaintext, "Plaintext mismatch after decrypt");
  }

  // NIST SP 800-38D Test Case 16: AES-256-GCM with AAD.
  // Key:  feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308
  // IV:   cafebabefacedbaddecaf888
  // PT:   d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39
  // AAD:  feedfacedeadbeeffeedfacedeadbeefabaddad2
  // CT:   522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662
  // Tag:  76fc6ece0f4e1768cddf8853bb2d551b
  #[test]
  fn nist_test_case_16_with_aad() {
    let key = Aes256GcmKey::from_bytes(hex32(
      "feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308",
    ));
    let nonce = Nonce96::from_bytes(hex12("cafebabefacedbaddecaf888"));
    let aad = hex_vec("feedfacedeadbeeffeedfacedeadbeefabaddad2");
    let plaintext = hex_vec(
      "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39",
    );
    let expected_ct = hex_vec(
      "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662",
    );
    let expected_tag = hex16("76fc6ece0f4e1768cddf8853bb2d551b");

    let cipher = Aes256Gcm::new(&key);

    // Encrypt.
    let mut buf = plaintext.clone();
    let tag = cipher.encrypt_in_place(&nonce, &aad, &mut buf);
    assert_eq!(buf, expected_ct, "Ciphertext mismatch");
    assert_eq!(tag.0, expected_tag, "Tag mismatch");

    // Decrypt.
    cipher.decrypt_in_place(&nonce, &aad, &mut buf, &tag).unwrap();
    assert_eq!(buf, plaintext, "Plaintext mismatch after decrypt");
  }

  /// Decryption with wrong tag should fail.
  #[test]
  fn bad_tag_rejected() {
    let key = Aes256GcmKey::from_bytes([0u8; 32]);
    let nonce = Nonce96::from_bytes([0u8; 12]);
    let cipher = Aes256Gcm::new(&key);

    let mut buf = vec![0u8; 16];
    let mut tag = cipher.encrypt_in_place(&nonce, &[], &mut buf);
    tag.0[0] ^= 1;

    let result = cipher.decrypt_in_place(&nonce, &[], &mut buf, &tag);
    assert!(result.is_err());
  }

  /// Decryption with wrong AAD should fail.
  #[test]
  fn wrong_aad_rejected() {
    let key = Aes256GcmKey::from_bytes(hex32(
      "feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308",
    ));
    let nonce = Nonce96::from_bytes(hex12("cafebabefacedbaddecaf888"));
    let aad = hex_vec("feedfacedeadbeeffeedfacedeadbeefabaddad2");
    let plaintext = hex_vec("d9313225f88406e5a55909c5aff5269a");
    let cipher = Aes256Gcm::new(&key);

    let mut buf = plaintext.clone();
    let tag = cipher.encrypt_in_place(&nonce, &aad, &mut buf);

    // Wrong AAD.
    let result = cipher.decrypt_in_place(&nonce, b"wrong aad", &mut buf, &tag);
    assert!(result.is_err());
  }

  /// Ciphertext tampering should fail verification.
  #[test]
  fn ciphertext_tampering_rejected() {
    let key = Aes256GcmKey::from_bytes([0x42u8; 32]);
    let nonce = Nonce96::from_bytes([0x07u8; 12]);
    let cipher = Aes256Gcm::new(&key);

    let mut buf = vec![0u8; 32];
    let tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buf);

    buf[0] ^= 1;
    let result = cipher.decrypt_in_place(&nonce, b"aad", &mut buf, &tag);
    assert!(result.is_err());
  }

  /// Detached encrypt/decrypt round-trip.
  #[test]
  fn detached_round_trip() {
    let key = Aes256GcmKey::from_bytes([0x42u8; 32]);
    let nonce = Nonce96::from_bytes([0x07u8; 12]);
    let aad = b"associated data";
    let plaintext = b"Hello, AES-256-GCM!";
    let cipher = Aes256Gcm::new(&key);

    let mut buf = plaintext.to_vec();
    let tag = cipher.encrypt_in_place(&nonce, aad, &mut buf);
    assert_ne!(&buf[..], &plaintext[..]);

    cipher.decrypt_in_place(&nonce, aad, &mut buf, &tag).unwrap();
    assert_eq!(&buf[..], &plaintext[..]);
  }

  /// Combined encrypt/decrypt round-trip.
  #[test]
  fn combined_round_trip() {
    let key = Aes256GcmKey::from_bytes([0x42u8; 32]);
    let nonce = Nonce96::from_bytes([0x07u8; 12]);
    let aad = b"associated data";
    let plaintext = b"Hello, AES-256-GCM!";
    let cipher = Aes256Gcm::new(&key);

    let mut out = vec![0u8; plaintext.len().strict_add(TAG_SIZE)];
    cipher.encrypt(&nonce, aad, plaintext, &mut out).unwrap();

    let mut pt_out = vec![0u8; plaintext.len()];
    cipher.decrypt(&nonce, aad, &out, &mut pt_out).unwrap();
    assert_eq!(&pt_out[..], &plaintext[..]);
  }

  /// `tag_from_slice` rejects wrong-length input.
  #[test]
  fn tag_from_slice_rejects_bad_length() {
    assert!(Aes256Gcm::tag_from_slice(&[0u8; 15]).is_err());
    assert!(Aes256Gcm::tag_from_slice(&[0u8; 17]).is_err());
    assert!(Aes256Gcm::tag_from_slice(&[0u8; 0]).is_err());
    assert!(Aes256Gcm::tag_from_slice(&[0u8; 16]).is_ok());
  }

  // --- Hex helpers ---

  fn hex16(hex: &str) -> [u8; 16] {
    let mut out = [0u8; 16];
    for i in 0..16 {
      out[i] = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
    }
    out
  }

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
