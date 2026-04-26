#![allow(clippy::indexing_slicing)]

//! AES-256-GCM public AEAD surface (NIST SP 800-38D).

use core::fmt;

#[cfg(target_arch = "x86_64")]
use super::targets::{AeadBackend, AeadPrimitive, select_backend};
use super::{AeadBufferError, LengthOverflow, Nonce96, OpenError, SealError, aes, ghash, polyval};
use crate::traits::{Aead, ct};

const KEY_SIZE: usize = 32;
const TAG_SIZE: usize = 16;
const NONCE_SIZE: usize = Nonce96::LENGTH;

/// Maximum plaintext length per NIST SP 800-38D: 2^39 - 256 bits = 2^36 - 32 bytes.
/// In practice the portable CTR uses a 32-bit counter, limiting to (2^32 - 2) blocks.
const MAX_PLAINTEXT_LEN: u64 = ((1u64 << 32).strict_sub(2)).strict_mul(16); // ~64 GiB

#[cfg(target_arch = "x86_64")]
const X86_VAES_GCM_MIN_LEN: usize = 32;

define_aead_key_type!(Aes256GcmKey, KEY_SIZE, "AES-256-GCM secret key (32 bytes).");

define_aead_tag_type!(Aes256GcmTag, TAG_SIZE, "AES-256-GCM authentication tag (16 bytes).");

/// AES-256-GCM AEAD (NIST SP 800-38D).
///
/// The standard TLS / interop workhorse.
///
/// # Nonce Uniqueness
///
/// **A nonce must never be reused with the same key.** Nonce reuse under
/// AES-GCM is catastrophic: it leaks the GHASH authentication key,
/// enabling both plaintext recovery (via crib-dragging) and universal
/// forgery of future messages. There is no recovery — the key must be
/// discarded.
///
/// SP 800-38D §8.3 places limits on IV usage under one key:
/// - random nonces: at most 2^32 encryptions per key (for collision-limiting risk),
/// - deterministic nonces/counter mode: at most 2^48 messages per key.
///
/// For high-volume use, prefer a deterministic construction with explicit
/// key rotation near 2^48 calls, or one of:
///
/// - [`crate::aead::NonceCounter`] — 32-bit fixed prefix + monotonic counter for deterministic
///   nonce issuance without allocations.
/// - [`Aes256GcmSiv`](crate::Aes256GcmSiv) — nonce-misuse resistant (degrades to deterministic
///   encryption on reuse, but never leaks the auth key).
/// - [`XChaCha20Poly1305`](crate::XChaCha20Poly1305) — 192-bit nonce makes random generation safe
///   for ~2^96 messages per key.
///
/// # Examples
///
/// ```
/// use rscrypto::{Aead, Aes256Gcm, Aes256GcmKey, aead::Nonce96};
///
/// let key = Aes256GcmKey::from_bytes([0x42; 32]);
/// let nonce = Nonce96::from_bytes([0x24; 12]);
/// let cipher = Aes256Gcm::new(&key);
///
/// let mut buf = *b"hello";
/// let tag = cipher.encrypt_in_place(&nonce, b"", &mut buf)?;
/// cipher.decrypt_in_place(&nonce, b"", &mut buf, &tag)?;
/// assert_eq!(&buf, b"hello");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Tampering is reported as an opaque verification failure.
///
/// ```
/// use rscrypto::{
///   Aead, Aes256Gcm, Aes256GcmKey,
///   aead::{Nonce96, OpenError},
/// };
///
/// let key = Aes256GcmKey::from_bytes([0x42; 32]);
/// let nonce = Nonce96::from_bytes([0x24; 12]);
/// let cipher = Aes256Gcm::new(&key);
///
/// let mut sealed = [0u8; 5 + Aes256Gcm::TAG_SIZE];
/// cipher.encrypt(&nonce, b"", b"hello", &mut sealed)?;
/// sealed[0] ^= 1;
///
/// let mut opened = [0u8; 5];
/// assert_eq!(
///   cipher.decrypt(&nonce, b"", &sealed, &mut opened),
///   Err(OpenError::verification())
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Security
///
/// On x86_64 (AES-NI), aarch64 (AES-CE), and s390x (CPACF), all AES
/// operations use constant-time hardware instructions. On RISC-V without
/// hardware AES extensions (Zkne / Zvkned), encryption falls back to the
/// constant-time portable implementation. That path is slower, but it
/// avoids secret-indexed lookup tables.
#[derive(Clone)]
pub struct Aes256Gcm {
  /// Pre-expanded AES-256 round keys.
  ek: aes::Aes256EncKey,
  /// GHASH hash key H = AES_K(0^128).
  h: [u8; 16],
  /// Precomputed H powers [H^4, H^3, H^2, H] in the POLYVAL domain
  /// for 4-block wide GHASH processing.
  h_powers_rev: [u128; 4],
  #[cfg(target_arch = "x86_64")]
  backend: AeadBackend,
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
  pub fn encrypt_in_place(&self, nonce: &Nonce96, aad: &[u8], buffer: &mut [u8]) -> Result<Aes256GcmTag, SealError> {
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
  ) -> Result<(), OpenError> {
    <Self as Aead>::decrypt_in_place(self, nonce, aad, buffer, tag)
  }

  /// Encrypt `plaintext` into `out` as `ciphertext || tag`.
  #[inline]
  pub fn encrypt(&self, nonce: &Nonce96, aad: &[u8], plaintext: &[u8], out: &mut [u8]) -> Result<(), SealError> {
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
fn compute_tag(
  ek: &aes::Aes256EncKey,
  h_polyval: u128,
  j0: &[u8; 16],
  aad: &[u8],
  ciphertext: &[u8],
) -> Result<[u8; TAG_SIZE], LengthOverflow> {
  let mut acc = 0u128;
  acc = ghash_update_padded(acc, h_polyval, aad);
  acc = ghash_update_padded(acc, h_polyval, ciphertext);

  let length_block = super::AeadByteLengths::from_usize(aad.len(), ciphertext.len()).to_be_bits_block();
  acc ^= u128::from_be_bytes(length_block);
  acc = polyval::clmul128_reduce(acc, h_polyval);

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
  Ok(s)
}

#[inline]
fn ghash_update_padded(mut acc: u128, h_polyval: u128, data: &[u8]) -> u128 {
  let (blocks, remainder) = data.as_chunks::<16>();
  for block in blocks {
    acc ^= u128::from_be_bytes(*block);
    acc = polyval::clmul128_reduce(acc, h_polyval);
  }

  if !remainder.is_empty() {
    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    acc ^= u128::from_be_bytes(block);
    acc = polyval::clmul128_reduce(acc, h_polyval);
  }

  acc
}

/// Compute the GCM authentication tag using 4-block wide GHASH.
///
/// Same semantics as `compute_tag` but processes ciphertext in 4-block
/// (64-byte) chunks via `accumulate_4blocks`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn compute_tag_wide(
  ek: &aes::Aes256EncKey,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
  j0: &[u8; 16],
  aad: &[u8],
  ciphertext: &[u8],
) -> Result<[u8; TAG_SIZE], LengthOverflow> {
  let mut acc = ghash_update_padded(0, h_polyval, aad);

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
  let length_block = super::AeadByteLengths::from_usize(aad.len(), ciphertext.len()).to_be_bits_block();
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
  Ok(s)
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

#[cfg(target_arch = "x86_64")]
#[inline]
fn resolve_backend() -> AeadBackend {
  select_backend(
    AeadPrimitive::Aes256Gcm,
    crate::platform::arch(),
    crate::platform::caps(),
  )
}

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

    Self {
      ek,
      h,
      h_powers_rev,
      #[cfg(target_arch = "x86_64")]
      backend: resolve_backend(),
    }
  }

  fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
    if bytes.len() != TAG_SIZE {
      return Err(AeadBufferError::new());
    }
    let mut tag = [0u8; TAG_SIZE];
    tag.copy_from_slice(bytes);
    Ok(Aes256GcmTag::from_bytes(tag))
  }

  fn encrypt_in_place(&self, nonce: &Self::Nonce, aad: &[u8], buffer: &mut [u8]) -> Result<Self::Tag, SealError> {
    super::seal_bounded_length_as_u64(buffer.len(), MAX_PLAINTEXT_LEN)?;

    let j0 = make_j0(nonce);
    let mut ctr_block = j0;
    inc32(&mut ctr_block);

    // Wide path: VAES-512 CTR + VPCLMULQDQ GHASH when available.
    #[cfg(target_arch = "x86_64")]
    if self.backend == AeadBackend::X86VaesVpclmul && buffer.len() >= X86_VAES_GCM_MIN_LEN {
      // SAFETY: VAES + VPCLMULQDQ availability verified during backend resolution.
      unsafe { aes::aes256_ctr32_encrypt_be_wide(&self.ek, &ctr_block, buffer) };
      let tag_bytes = compute_tag_wide(&self.ek, self.h_powers_rev[3], &self.h_powers_rev, &j0, aad, buffer)
        .map_err(|_| SealError::too_large())?;
      return Ok(Aes256GcmTag::from_bytes(tag_bytes));
    }

    // Scalar path: single-block AES-NI/CE/portable + single-block GHASH.
    aes::aes256_ctr32_encrypt_be(&self.ek, &ctr_block, buffer);
    let tag_bytes =
      compute_tag(&self.ek, self.h_powers_rev[3], &j0, aad, buffer).map_err(|_| SealError::too_large())?;
    Ok(Aes256GcmTag::from_bytes(tag_bytes))
  }

  fn decrypt_in_place(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), OpenError> {
    super::open_bounded_length_as_u64(buffer.len(), MAX_PLAINTEXT_LEN)?;

    let j0 = make_j0(nonce);

    // Wide path: VPCLMULQDQ GHASH + VAES-512 CTR when available.
    #[cfg(target_arch = "x86_64")]
    if self.backend == AeadBackend::X86VaesVpclmul && buffer.len() >= X86_VAES_GCM_MIN_LEN {
      // Verify tag BEFORE decryption (authenticate-then-decrypt).
      let expected = compute_tag_wide(&self.ek, self.h_powers_rev[3], &self.h_powers_rev, &j0, aad, buffer)
        .map_err(|_| OpenError::too_large())?;
      if !ct::constant_time_eq(&expected, tag.as_bytes()) {
        ct::zeroize(buffer);
        return Err(OpenError::verification());
      }
      let mut ctr_block = j0;
      inc32(&mut ctr_block);
      // SAFETY: VAES availability verified during backend resolution.
      unsafe { aes::aes256_ctr32_encrypt_be_wide(&self.ek, &ctr_block, buffer) };
      return Ok(());
    }

    // Scalar path: authenticate then decrypt.
    let expected = compute_tag(&self.ek, self.h_powers_rev[3], &j0, aad, buffer).map_err(|_| OpenError::too_large())?;
    if !ct::constant_time_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Err(OpenError::verification());
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
    let tag = cipher.encrypt_in_place(&nonce, &[], &mut buf).unwrap();
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
    let tag = cipher.encrypt_in_place(&nonce, &[], &mut buf).unwrap();
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
    let tag = cipher.encrypt_in_place(&nonce, &[], &mut buf).unwrap();
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
    let tag = cipher.encrypt_in_place(&nonce, &aad, &mut buf).unwrap();
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
    let mut tag = cipher.encrypt_in_place(&nonce, &[], &mut buf).unwrap();
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
    let tag = cipher.encrypt_in_place(&nonce, &aad, &mut buf).unwrap();

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
    let tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buf).unwrap();

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
    let tag = cipher.encrypt_in_place(&nonce, aad, &mut buf).unwrap();
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

  /// Decryption with wrong nonce must fail.
  #[test]
  fn wrong_nonce_rejected() {
    let key = Aes256GcmKey::from_bytes([0x42u8; 32]);
    let nonce = Nonce96::from_bytes([0x07u8; 12]);
    let cipher = Aes256Gcm::new(&key);

    let mut buf = *b"hello gcm";
    let tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buf).unwrap();

    let wrong_nonce = Nonce96::from_bytes([0x08u8; 12]);
    let result = cipher.decrypt_in_place(&wrong_nonce, b"aad", &mut buf, &tag);
    assert!(result.is_err());
  }

  /// On authentication failure, the output buffer must be zeroed.
  #[test]
  fn buffer_zeroed_on_auth_failure() {
    let key = Aes256GcmKey::from_bytes([0x42u8; 32]);
    let nonce = Nonce96::from_bytes([0x07u8; 12]);
    let cipher = Aes256Gcm::new(&key);

    let plaintext = *b"zero me on failure";
    let mut buf = plaintext;
    let tag = cipher.encrypt_in_place(&nonce, b"aad", &mut buf).unwrap();

    // Corrupt the tag.
    let mut bad_tag = tag.to_bytes();
    bad_tag[0] ^= 0xFF;
    let bad_tag = Aes256GcmTag::from_bytes(bad_tag);

    let result = cipher.decrypt_in_place(&nonce, b"aad", &mut buf, &bad_tag);
    assert!(result.is_err());
    assert!(buf.iter().all(|&b| b == 0), "buffer not zeroed on auth failure");
  }
}
