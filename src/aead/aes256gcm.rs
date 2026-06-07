#![allow(clippy::indexing_slicing)]

//! AES-256-GCM public AEAD surface (NIST SP 800-38D).

use core::fmt;

use super::{
  AeadBufferError, LengthOverflow, Nonce96, OpenError, SealError, aes, ghash, polyval,
  targets::{AeadBackend, AeadPrimitive, select_backend},
};
use crate::traits::{Aead, ct};

const KEY_SIZE: usize = 32;
const TAG_SIZE: usize = 16;
const NONCE_SIZE: usize = Nonce96::LENGTH;

/// Maximum plaintext length per NIST SP 800-38D: 2^39 - 256 bits = 2^36 - 32 bytes.
/// In practice the portable CTR uses a 32-bit counter, limiting to (2^32 - 2) blocks.
const MAX_PLAINTEXT_LEN: u64 = ((1u64 << 32).strict_sub(2)).strict_mul(16); // ~64 GiB

#[cfg(target_arch = "x86_64")]
const X86_VAES_GCM_MIN_LEN: usize = 64;
#[cfg(target_arch = "x86_64")]
const X86_AESNI_GCM_MIN_LEN: usize = 64;

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
  /// Precomputed H powers [H^8, H^7, ..., H] for 8-block GHASH windows.
  h_powers_rev_8: [u128; 8],
  /// Precomputed H powers [H^16, H^15, ..., H] for 16-block GHASH windows.
  #[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux"))
  ))]
  h_powers_rev_16: [u128; 16],
  /// Precomputed H powers [H^32, H^31, ..., H] for x86 32-block GHASH windows.
  #[cfg(target_arch = "x86_64")]
  h_powers_rev_32: [u128; 32],
  /// Precomputed H powers [H^64, H^63, ..., H] for x86 64-block GHASH windows.
  #[cfg(target_arch = "x86_64")]
  h_powers_rev_64: [u128; 64],
  /// Precomputed H powers [H^128, H^127, ..., H] for x86 128-block GHASH windows.
  #[cfg(target_arch = "x86_64")]
  h_powers_rev_128: [u128; 128],
  /// Precomputed `(lo64 ^ hi64)` for each 16-block GHASH power.
  #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
  h_powers_rev_16_mid: [u128; 16],
  /// Pair-packed 16-block GHASH powers for AArch64 assembly.
  #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
  h_powers_rev_16_pair: [u128; 24],
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

  #[cfg(target_arch = "x86_64")]
  #[inline]
  fn x86_gcm_tables<'a>(&'a self) -> aes::X86GcmTables<'a> {
    aes::X86GcmTables {
      h_polyval: self.h_powers_rev[3],
      h_powers_rev: &self.h_powers_rev,
      h_powers_rev_16: &self.h_powers_rev_16,
      #[cfg(target_os = "linux")]
      h_powers_rev_32: &self.h_powers_rev_32,
      #[cfg(target_os = "linux")]
      h_powers_rev_64: &self.h_powers_rev_64,
      #[cfg(target_os = "linux")]
      h_powers_rev_128: &self.h_powers_rev_128,
    }
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

// GCM construction internals (NIST SP 800-38D)

/// Build the initial counter block J0 and first CTR block for a 96-bit IV.
///
/// J0 = IV || 0x00000001 (NIST SP 800-38D § 7.1, when len(IV) = 96).
#[inline]
fn make_j0_and_ctr(nonce: &Nonce96) -> ([u8; 16], [u8; 16]) {
  let mut j0 = [0u8; 16];
  j0[..12].copy_from_slice(nonce.as_bytes());
  j0[15] = 0x01; // Counter starts at 1.
  let mut ctr_block = j0;
  ctr_block[15] = 0x02;
  (j0, ctr_block)
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

  let length_block = super::AeadByteLengths::try_new_bit_lengths(aad.len(), ciphertext.len())?.to_be_bits_block();
  acc ^= u128::from_be_bytes(length_block);
  acc = polyval::clmul128_reduce(acc, h_polyval);

  Ok(encrypt_j0_tag(ek, j0, acc))
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

#[inline]
fn ghash_update_padded_wide(mut acc: u128, h_polyval: u128, h_powers_rev: &[u128; 4], data: &[u8]) -> u128 {
  let (full_blocks, remainder) = data.as_chunks::<16>();
  let mut chunks = full_blocks.chunks_exact(4);

  for chunk in &mut chunks {
    let blocks = [
      u128::from_be_bytes(chunk[0]),
      u128::from_be_bytes(chunk[1]),
      u128::from_be_bytes(chunk[2]),
      u128::from_be_bytes(chunk[3]),
    ];
    acc = polyval::accumulate_4blocks(acc, h_polyval, h_powers_rev, &blocks);
  }

  for block in chunks.remainder() {
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

#[cfg(target_arch = "x86_64")]
#[inline]
fn ghash_collect_padded_block(blocks: &mut [u128; 4], block_count: &mut usize, block: u128) -> bool {
  if *block_count == 4 {
    return false;
  }
  blocks[*block_count] = block;
  *block_count = (*block_count).strict_add(1);
  true
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn ghash_collect_padded(blocks: &mut [u128; 4], block_count: &mut usize, data: &[u8]) -> bool {
  let (full_blocks, remainder) = data.as_chunks::<16>();
  for block in full_blocks {
    if !ghash_collect_padded_block(blocks, block_count, u128::from_be_bytes(*block)) {
      return false;
    }
  }

  if !remainder.is_empty() {
    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    return ghash_collect_padded_block(blocks, block_count, u128::from_be_bytes(block));
  }

  true
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn compute_tag_short_wide(
  ek: &aes::Aes256EncKey,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
  j0: &[u8; 16],
  aad: &[u8],
  ciphertext: &[u8],
) -> Result<Option<[u8; TAG_SIZE]>, LengthOverflow> {
  let mut sequence = [0u128; 4];
  let mut block_count = 0usize;

  if !ghash_collect_padded(&mut sequence, &mut block_count, aad) {
    return Ok(None);
  }
  if !ghash_collect_padded(&mut sequence, &mut block_count, ciphertext) {
    return Ok(None);
  }

  let length_block = super::AeadByteLengths::try_new_bit_lengths(aad.len(), ciphertext.len())?.to_be_bits_block();
  if !ghash_collect_padded_block(&mut sequence, &mut block_count, u128::from_be_bytes(length_block)) {
    return Ok(None);
  }

  if block_count == 1 && sequence[0] == 0 {
    return Ok(Some(encrypt_j0_tag(ek, j0, 0)));
  }

  let mut blocks = [0u128; 4];
  let start = 4usize.strict_sub(block_count);
  blocks[start..].copy_from_slice(&sequence[..block_count]);
  let acc = polyval::accumulate_4blocks(0, h_polyval, h_powers_rev, &blocks);
  Ok(Some(encrypt_j0_tag(ek, j0, acc)))
}

#[inline(always)]
fn encrypt_j0_tag(ek: &aes::Aes256EncKey, j0: &[u8; 16], acc: u128) -> [u8; TAG_SIZE] {
  let mut encrypted_j0 = *j0;
  aes::aes256_encrypt_block(ek, &mut encrypted_j0);
  let tag = (acc ^ u128::from_be_bytes(encrypted_j0)).to_be_bytes();
  ct::zeroize(&mut encrypted_j0);
  tag
}

#[cfg(feature = "diag")]
#[must_use]
pub fn diag_aes256gcm_ctr32_be(cipher: &Aes256Gcm, nonce: &Nonce96, plaintext: &[u8; 44]) -> [u8; 16] {
  let (_, ctr_block) = make_j0_and_ctr(nonce);
  let mut buffer = *plaintext;
  aes::aes256_ctr32_encrypt_be(&cipher.ek, &ctr_block, &mut buffer);
  diag_fold16(&buffer)
}

#[cfg(feature = "diag")]
#[must_use]
pub fn diag_aes256gcm_ghash(cipher: &Aes256Gcm, aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
  let h_polyval = cipher.h_powers_rev[3];
  let mut acc = 0u128;
  if should_use_wide_ghash(cipher.backend, aad.len(), ciphertext.len()) {
    acc = ghash_update_padded_wide(acc, h_polyval, &cipher.h_powers_rev, aad);
    acc = ghash_update_padded_wide(acc, h_polyval, &cipher.h_powers_rev, ciphertext);
  } else {
    acc = ghash_update_padded(acc, h_polyval, aad);
    acc = ghash_update_padded(acc, h_polyval, ciphertext);
  }
  let length_block = match super::AeadByteLengths::try_new_bit_lengths(aad.len(), ciphertext.len()) {
    Ok(lengths) => lengths.to_be_bits_block(),
    Err(_) => return [0u8; 16],
  };
  acc ^= u128::from_be_bytes(length_block);
  acc = polyval::clmul128_reduce(acc, h_polyval);
  acc.to_be_bytes()
}

#[cfg(feature = "diag")]
#[must_use]
pub fn diag_aes256gcm_tag_aes(cipher: &Aes256Gcm, nonce: &Nonce96, acc: &[u8; 16]) -> [u8; 16] {
  let (j0, _) = make_j0_and_ctr(nonce);
  encrypt_j0_tag(&cipher.ek, &j0, u128::from_be_bytes(*acc))
}

#[cfg(feature = "diag")]
fn diag_fold16(data: &[u8]) -> [u8; 16] {
  let (blocks, tail) = data.as_chunks::<16>();
  let mut acc = 0u128;
  for block in blocks {
    acc ^= u128::from_ne_bytes(*block);
  }
  if !tail.is_empty() {
    let mut block = [0u8; 16];
    block[..tail.len()].copy_from_slice(tail);
    acc ^= u128::from_ne_bytes(block);
  }
  acc.to_ne_bytes()
}

/// Compute the GCM authentication tag using 4-block wide GHASH.
///
/// Same semantics as `compute_tag` but processes AAD and ciphertext in
/// 4-block (64-byte) chunks via `accumulate_4blocks`.
#[inline]
fn compute_tag_wide(
  ek: &aes::Aes256EncKey,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
  j0: &[u8; 16],
  aad: &[u8],
  ciphertext: &[u8],
) -> Result<[u8; TAG_SIZE], LengthOverflow> {
  let mut acc = 0u128;
  acc = ghash_update_padded_wide(acc, h_polyval, h_powers_rev, aad);
  acc = ghash_update_padded_wide(acc, h_polyval, h_powers_rev, ciphertext);

  // Length block.
  let length_block = super::AeadByteLengths::try_new_bit_lengths(aad.len(), ciphertext.len())?.to_be_bits_block();
  acc ^= u128::from_be_bytes(length_block);
  acc = polyval::clmul128_reduce(acc, h_polyval);

  Ok(encrypt_j0_tag(ek, j0, acc))
}

/// Update GHASH using VPCLMUL without scalar block packing.
///
/// # Safety
/// Caller must ensure VPCLMULQDQ, PCLMULQDQ, AVX-512F/VL/BW/DQ, and SSE2 are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,pclmulqdq,sse2")]
unsafe fn ghash_update_padded_wide_x86(mut acc: u128, h_polyval: u128, h_powers_rev: &[u128; 4], data: &[u8]) -> u128 {
  let (chunks, tail) = data.as_chunks::<64>();
  for chunk in chunks {
    // SAFETY: direct-byte VPCLMUL GHASH aggregation because:
    // 1. This function's caller guarantees all required x86 target features.
    // 2. `chunk` is exactly four initialized 16-byte GHASH blocks.
    acc = unsafe { polyval::x86_aggregate_4blocks_be_bytes_inline(acc, h_powers_rev, chunk) };
  }

  let (full_blocks, remainder) = tail.as_chunks::<16>();
  for block in full_blocks {
    acc ^= u128::from_be_bytes(*block);
    // SAFETY: x86 carryless multiply because:
    // 1. This function's caller guarantees PCLMULQDQ and SSE2 availability.
    // 2. `acc` and `h_polyval` are initialized GHASH field elements.
    acc = unsafe { polyval::x86_clmul128_reduce_inline(acc, h_polyval) };
  }

  if !remainder.is_empty() {
    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    acc ^= u128::from_be_bytes(block);
    // SAFETY: x86 carryless multiply because:
    // 1. This function's caller guarantees PCLMULQDQ and SSE2 availability.
    // 2. `acc` and `h_polyval` are initialized GHASH field elements.
    acc = unsafe { polyval::x86_clmul128_reduce_inline(acc, h_polyval) };
  }

  acc
}

/// Update GHASH using PMULL without leaving the aarch64 target-feature scope.
///
/// # Safety
/// Caller must ensure AES-CE and PMULL are available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes,neon")]
unsafe fn ghash_update_padded_wide_aarch64(
  mut acc: u128,
  h_polyval: u128,
  h_powers_rev: &[u128; 4],
  data: &[u8],
) -> u128 {
  let (full_blocks, remainder) = data.as_chunks::<16>();
  let mut chunks = full_blocks.chunks_exact(4);

  for chunk in &mut chunks {
    let blocks = [
      u128::from_be_bytes(chunk[0]),
      u128::from_be_bytes(chunk[1]),
      u128::from_be_bytes(chunk[2]),
      u128::from_be_bytes(chunk[3]),
    ];
    // SAFETY: PMULL 4-block GHASH aggregation because:
    // 1. This function's caller must guarantee AES-CE/PMULL availability.
    // 2. `h_powers_rev` and `blocks` are fixed 4-lane arrays with valid initialized values.
    acc = unsafe { polyval::aarch64_aggregate_4blocks_inline(acc, h_powers_rev, &blocks) };
  }

  for block in chunks.remainder() {
    acc ^= u128::from_be_bytes(*block);
    // SAFETY: PMULL carryless multiply because:
    // 1. This function's caller must guarantee AES-CE/PMULL availability.
    // 2. `acc` and `h_polyval` are initialized GHASH field elements.
    acc = unsafe { polyval::aarch64_clmul128_reduce_inline(acc, h_polyval) };
  }

  if !remainder.is_empty() {
    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    acc ^= u128::from_be_bytes(block);
    // SAFETY: PMULL carryless multiply because:
    // 1. This function's caller must guarantee AES-CE/PMULL availability.
    // 2. `acc` and `h_polyval` are initialized GHASH field elements.
    acc = unsafe { polyval::aarch64_clmul128_reduce_inline(acc, h_polyval) };
  }

  acc
}

/// Update GHASH using POWER8 crypto without leaving the target-feature scope.
///
/// # Safety
/// Caller must ensure POWER8 crypto is available.
#[cfg(target_arch = "powerpc64")]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
unsafe fn ghash_update_padded_wide_ppc(mut acc: u128, h_polyval: u128, h_powers_rev: &[u128; 4], data: &[u8]) -> u128 {
  let (full_blocks, remainder) = data.as_chunks::<16>();
  let mut chunks = full_blocks.chunks_exact(4);

  for chunk in &mut chunks {
    let blocks = [
      u128::from_be_bytes(chunk[0]),
      u128::from_be_bytes(chunk[1]),
      u128::from_be_bytes(chunk[2]),
      u128::from_be_bytes(chunk[3]),
    ];
    // SAFETY: POWER8 4-block GHASH aggregation because:
    // 1. This function's caller must guarantee POWER8 crypto availability.
    // 2. `h_powers_rev` and `blocks` are fixed 4-lane arrays with valid initialized values.
    acc = unsafe { polyval::ppc_aggregate_4blocks_inline(acc, h_powers_rev, &blocks) };
  }

  for block in chunks.remainder() {
    acc ^= u128::from_be_bytes(*block);
    // SAFETY: POWER8 carryless multiply because:
    // 1. This function's caller must guarantee POWER8 crypto availability.
    // 2. `acc` and `h_polyval` are initialized GHASH field elements.
    acc = unsafe { polyval::ppc_clmul128_reduce_inline(acc, h_polyval) };
  }

  if !remainder.is_empty() {
    let mut block = [0u8; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    acc ^= u128::from_be_bytes(block);
    // SAFETY: POWER8 carryless multiply because:
    // 1. This function's caller must guarantee POWER8 crypto availability.
    // 2. `acc` and `h_polyval` are initialized GHASH field elements.
    acc = unsafe { polyval::ppc_clmul128_reduce_inline(acc, h_polyval) };
  }

  acc
}

#[inline]
fn should_use_wide_ghash(backend: AeadBackend, aad_len: usize, ciphertext_len: usize) -> bool {
  const MIN_WIDE_INPUT: usize = 64;

  if aad_len < MIN_WIDE_INPUT && ciphertext_len < MIN_WIDE_INPUT {
    return false;
  }

  match backend {
    AeadBackend::X86VaesVpclmul
    | AeadBackend::X86AesniPclmul
    | AeadBackend::Aarch64AesPmull
    | AeadBackend::Aarch64Sve2AesPmull
    | AeadBackend::Power8Crypto => true,
    AeadBackend::S390xMsa => crate::platform::caps().has(crate::platform::caps::s390x::VECTOR),
    _ => false,
  }
}

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

    // Precompute H powers for 4/8-block wide GHASH.
    // GHASH loads H as BE, then applies mulX_POLYVAL to get the POLYVAL-domain key.
    let h_polyval = ghash::h_to_polyval(&h);
    let powers = polyval::precompute_powers_8(h_polyval);
    // Reverse: [H, H^2, H^3, H^4] -> [H^4, H^3, H^2, H]
    let h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    let h_powers_rev_8 = [
      powers[7], powers[6], powers[5], powers[4], powers[3], powers[2], powers[1], powers[0],
    ];
    #[cfg(any(
      target_arch = "x86_64",
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux"))
    ))]
    let h_powers_rev_16 = {
      let powers = polyval::precompute_powers_16(h_polyval);
      core::array::from_fn(|i| powers[15usize.strict_sub(i)])
    };
    #[cfg(target_arch = "x86_64")]
    let h_powers_rev_32 = {
      let powers = polyval::precompute_powers_32(h_polyval);
      core::array::from_fn(|i| powers[31usize.strict_sub(i)])
    };
    #[cfg(target_arch = "x86_64")]
    let h_powers_rev_64 = {
      let powers = polyval::precompute_powers_64(h_polyval);
      core::array::from_fn(|i| powers[63usize.strict_sub(i)])
    };
    #[cfg(target_arch = "x86_64")]
    let h_powers_rev_128 = {
      let powers = polyval::precompute_powers_128(h_polyval);
      core::array::from_fn(|i| powers[127usize.strict_sub(i)])
    };
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
    let h_powers_rev_16_mid = polyval::precompute_powers_16_mid(&h_powers_rev_16);
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
    let h_powers_rev_16_pair = polyval::precompute_powers_16_pair(&h_powers_rev_16);
    let backend = resolve_backend();

    Self {
      ek,
      h,
      h_powers_rev,
      h_powers_rev_8,
      #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux"))
      ))]
      h_powers_rev_16,
      #[cfg(target_arch = "x86_64")]
      h_powers_rev_32,
      #[cfg(target_arch = "x86_64")]
      h_powers_rev_64,
      #[cfg(target_arch = "x86_64")]
      h_powers_rev_128,
      #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
      h_powers_rev_16_mid,
      #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
      h_powers_rev_16_pair,
      backend,
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
    let length_block = super::seal_bit_lengths(aad.len(), buffer.len())?.to_be_bits_block();
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "powerpc64")))]
    let _ = length_block;

    let (j0, ctr_block) = make_j0_and_ctr(nonce);

    // Wide path: VAES-512 CTR + VPCLMULQDQ GHASH when available.
    #[cfg(target_arch = "x86_64")]
    if self.backend == AeadBackend::X86VaesVpclmul && buffer.len() >= X86_VAES_GCM_MIN_LEN {
      let tables = self.x86_gcm_tables();
      let h_polyval = tables.h_polyval;
      // SAFETY: x86 GHASH AAD path because:
      // 1. Backend resolution selected X86VaesVpclmul only after CPUID confirmed VPCLMULQDQ.
      // 2. `aad` is a valid byte slice; padding is handled inside the helper.
      let mut acc = unsafe { ghash_update_padded_wide_x86(0, h_polyval, tables.h_powers_rev, aad) };
      // SAFETY: fused x86 AES-GCM sealing because:
      // 1. Backend resolution selected X86VaesVpclmul only after CPUID confirmed VAES, VPCLMULQDQ, and
      //    AES-NI.
      // 2. The helper encrypts `buffer` in place and folds the resulting ciphertext into GHASH.
      acc = unsafe { aes::aes256_ctr32_encrypt_be_wide_ghash(&self.ek, &ctr_block, buffer, acc, tables) };
      acc ^= u128::from_be_bytes(length_block);
      // SAFETY: x86 GHASH final multiply because:
      // 1. Backend resolution selected X86VaesVpclmul only after CPUID confirmed PCLMULQDQ.
      // 2. `acc` and `h_polyval` are initialized GHASH field elements.
      acc = unsafe { polyval::x86_clmul128_reduce_inline(acc, h_polyval) };
      let tag_bytes = encrypt_j0_tag(&self.ek, &j0, acc);
      return Ok(Aes256GcmTag::from_bytes(tag_bytes));
    }

    #[cfg(target_arch = "x86_64")]
    if self.backend == AeadBackend::X86AesniPclmul
      && buffer.len() >= X86_AESNI_GCM_MIN_LEN
      && crate::platform::caps().has(crate::platform::caps::x86::PCLMUL_READY)
    {
      let h_polyval = self.h_powers_rev[3];
      let mut acc = ghash_update_padded_wide(0, h_polyval, &self.h_powers_rev, aad);
      // SAFETY: fused x86 AES-NI/PCLMUL sealing because:
      // 1. Backend resolution selected X86AesniPclmul only after CPUID confirmed AES-NI and PCLMULQDQ.
      // 2. The extra PCLMUL_READY check confirms SSSE3 for in-register GHASH byte reversal.
      // 3. The helper encrypts `buffer` in place and folds ciphertext into GHASH.
      acc = unsafe {
        aes::aes256_ctr32_encrypt_be_aesni_pclmul_ghash(
          &self.ek,
          &ctr_block,
          buffer,
          acc,
          h_polyval,
          &self.h_powers_rev,
        )
      };
      acc ^= u128::from_be_bytes(length_block);
      // SAFETY: x86 GHASH final multiply because PCLMUL_READY was checked above.
      acc = unsafe { polyval::x86_clmul128_reduce_inline(acc, h_polyval) };
      let tag_bytes = encrypt_j0_tag(&self.ek, &j0, acc);
      return Ok(Aes256GcmTag::from_bytes(tag_bytes));
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(
      self.backend,
      AeadBackend::Aarch64AesPmull | AeadBackend::Aarch64Sve2AesPmull
    ) {
      let h_polyval = self.h_powers_rev[3];
      // SAFETY: aarch64 GHASH AAD path because:
      // 1. Backend resolution selected an AES/PMULL backend only after runtime detection confirmed AES-CE
      //    and PMULL.
      // 2. `aad` is a valid byte slice; padding is handled inside the helper.
      let mut acc = unsafe { ghash_update_padded_wide_aarch64(0, h_polyval, &self.h_powers_rev, aad) };
      // SAFETY: fused intrinsic AArch64 AES-GCM sealing because:
      // 1. Backend resolution selected an AES/PMULL backend only after runtime detection confirmed AES-CE
      //    and PMULL.
      // 2. The helper encrypts `buffer` in place and folds the resulting ciphertext into GHASH.
      let tables = aes::Aarch64GcmTables {
        h_polyval,
        h_powers_rev: &self.h_powers_rev,
        h_powers_rev_8: &self.h_powers_rev_8,
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        h_powers_rev_16: &self.h_powers_rev_16,
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        h_powers_rev_16_mid: &self.h_powers_rev_16_mid,
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        h_powers_rev_16_pair: &self.h_powers_rev_16_pair,
      };
      // SAFETY: backend selection confirmed AES-CE/PMULL, and the helper encrypts the valid buffer
      // in place while folding ciphertext into GHASH.
      acc = unsafe { aes::aes256_ctr32_encrypt_be_aarch64_ghash(&self.ek, &ctr_block, buffer, acc, &tables) };
      acc ^= u128::from_be_bytes(length_block);
      // SAFETY: aarch64 GHASH final multiply because:
      // 1. Backend resolution selected an AES/PMULL backend only after runtime detection confirmed PMULL.
      // 2. `acc` and `h_polyval` are initialized GHASH field elements.
      acc = unsafe { polyval::aarch64_clmul128_reduce_inline(acc, h_polyval) };
      let tag_bytes = encrypt_j0_tag(&self.ek, &j0, acc);
      return Ok(Aes256GcmTag::from_bytes(tag_bytes));
    }

    #[cfg(target_arch = "powerpc64")]
    if self.backend == AeadBackend::Power8Crypto {
      let h_polyval = self.h_powers_rev[3];
      // SAFETY: POWER8 GHASH AAD path because:
      // 1. Backend resolution selected `Power8Crypto` only after runtime detection confirmed POWER8
      //    crypto.
      // 2. `aad` is a valid byte slice; padding is handled inside the helper.
      let mut acc = unsafe { ghash_update_padded_wide_ppc(0, h_polyval, &self.h_powers_rev, aad) };
      // SAFETY: fused POWER AES-GCM sealing because:
      // 1. Backend resolution selected `Power8Crypto` only after runtime detection confirmed POWER8
      //    crypto.
      // 2. The helper encrypts `buffer` in place and folds ciphertext into GHASH.
      acc = unsafe {
        aes::aes256_ctr32_encrypt_be_ppc_ghash(&self.ek, &ctr_block, buffer, acc, h_polyval, &self.h_powers_rev)
      };
      acc ^= u128::from_be_bytes(length_block);
      // SAFETY: POWER8 carryless multiply because backend resolution confirmed POWER8 crypto.
      acc = unsafe { polyval::ppc_clmul128_reduce_inline(acc, h_polyval) };
      let tag_bytes = encrypt_j0_tag(&self.ek, &j0, acc);
      return Ok(Aes256GcmTag::from_bytes(tag_bytes));
    }

    // Scalar path: single-block AES-NI/CE/portable + single-block GHASH.
    aes::aes256_ctr32_encrypt_be(&self.ek, &ctr_block, buffer);
    #[cfg(target_arch = "x86_64")]
    if self.backend == AeadBackend::X86VaesVpclmul
      && let Some(tag_bytes) =
        compute_tag_short_wide(&self.ek, self.h_powers_rev[3], &self.h_powers_rev, &j0, aad, buffer)
          .map_err(|_| SealError::too_large())?
    {
      return Ok(Aes256GcmTag::from_bytes(tag_bytes));
    }
    if should_use_wide_ghash(self.backend, aad.len(), buffer.len()) {
      let tag_bytes = compute_tag_wide(&self.ek, self.h_powers_rev[3], &self.h_powers_rev, &j0, aad, buffer)
        .map_err(|_| SealError::too_large())?;
      return Ok(Aes256GcmTag::from_bytes(tag_bytes));
    }
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
    let length_block = super::open_bit_lengths(aad.len(), buffer.len())?.to_be_bits_block();
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "powerpc64")))]
    let _ = length_block;

    let (j0, ctr_block) = make_j0_and_ctr(nonce);

    // Wide path: VPCLMULQDQ GHASH + VAES-512 CTR when available.
    #[cfg(target_arch = "x86_64")]
    if self.backend == AeadBackend::X86VaesVpclmul && buffer.len() >= X86_VAES_GCM_MIN_LEN {
      let tables = self.x86_gcm_tables();
      let h_polyval = tables.h_polyval;
      // SAFETY: x86 GHASH AAD path because:
      // 1. Backend resolution selected X86VaesVpclmul only after CPUID confirmed VPCLMULQDQ.
      // 2. `aad` is a valid byte slice; padding is handled inside the helper.
      let mut acc = unsafe { ghash_update_padded_wide_x86(0, h_polyval, tables.h_powers_rev, aad) };
      // SAFETY: fused x86 AES-GCM open because:
      // 1. Backend resolution selected X86VaesVpclmul only after CPUID confirmed VAES, VPCLMULQDQ, and
      //    AES-NI.
      // 2. The helper GHASHes ciphertext bytes before decrypting each chunk in place.
      acc = unsafe { aes::aes256_ctr32_decrypt_be_wide_ghash(&self.ek, &ctr_block, buffer, acc, tables) };
      acc ^= u128::from_be_bytes(length_block);
      // SAFETY: x86 GHASH final multiply because:
      // 1. Backend resolution selected X86VaesVpclmul only after CPUID confirmed PCLMULQDQ.
      // 2. `acc` and `h_polyval` are initialized GHASH field elements.
      acc = unsafe { polyval::x86_clmul128_reduce_inline(acc, h_polyval) };
      let expected = encrypt_j0_tag(&self.ek, &j0, acc);
      if !ct::constant_time_eq(&expected, tag.as_bytes()) {
        ct::zeroize(buffer);
        return Err(OpenError::verification());
      }
      return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    if self.backend == AeadBackend::X86AesniPclmul
      && buffer.len() >= X86_AESNI_GCM_MIN_LEN
      && crate::platform::caps().has(crate::platform::caps::x86::PCLMUL_READY)
    {
      let h_polyval = self.h_powers_rev[3];
      let mut acc = ghash_update_padded_wide(0, h_polyval, &self.h_powers_rev, aad);
      // SAFETY: fused x86 AES-NI/PCLMUL open because:
      // 1. Backend resolution selected X86AesniPclmul only after CPUID confirmed AES-NI and PCLMULQDQ.
      // 2. The extra PCLMUL_READY check confirms SSSE3 for in-register GHASH byte reversal.
      // 3. The helper GHASHes ciphertext before decrypting each chunk in place.
      acc = unsafe {
        aes::aes256_ctr32_decrypt_be_aesni_pclmul_ghash(
          &self.ek,
          &ctr_block,
          buffer,
          acc,
          h_polyval,
          &self.h_powers_rev,
        )
      };
      acc ^= u128::from_be_bytes(length_block);
      // SAFETY: x86 GHASH final multiply because PCLMUL_READY was checked above.
      acc = unsafe { polyval::x86_clmul128_reduce_inline(acc, h_polyval) };
      let expected = encrypt_j0_tag(&self.ek, &j0, acc);
      if !ct::constant_time_eq(&expected, tag.as_bytes()) {
        ct::zeroize(buffer);
        return Err(OpenError::verification());
      }
      return Ok(());
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(
      self.backend,
      AeadBackend::Aarch64AesPmull | AeadBackend::Aarch64Sve2AesPmull
    ) {
      let h_polyval = self.h_powers_rev[3];
      // SAFETY: aarch64 GHASH AAD path because:
      // 1. Backend resolution selected an AES/PMULL backend only after runtime detection confirmed AES-CE
      //    and PMULL.
      // 2. `aad` is a valid byte slice; padding is handled inside the helper.
      let mut acc = unsafe { ghash_update_padded_wide_aarch64(0, h_polyval, &self.h_powers_rev, aad) };
      // SAFETY: fused intrinsic AArch64 AES-GCM open because:
      // 1. Backend resolution selected an AES/PMULL backend only after runtime detection confirmed AES-CE
      //    and PMULL.
      // 2. The helper GHASHes ciphertext bytes before decrypting each chunk in place.
      let tables = aes::Aarch64GcmTables {
        h_polyval,
        h_powers_rev: &self.h_powers_rev,
        h_powers_rev_8: &self.h_powers_rev_8,
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        h_powers_rev_16: &self.h_powers_rev_16,
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        h_powers_rev_16_mid: &self.h_powers_rev_16_mid,
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        h_powers_rev_16_pair: &self.h_powers_rev_16_pair,
      };
      // SAFETY: backend selection confirmed AES-CE/PMULL, and the helper GHASHes ciphertext before
      // decrypting the valid buffer in place.
      acc = unsafe { aes::aes256_ctr32_decrypt_be_aarch64_ghash(&self.ek, &ctr_block, buffer, acc, &tables) };
      acc ^= u128::from_be_bytes(length_block);
      // SAFETY: aarch64 GHASH final multiply because:
      // 1. Backend resolution selected an AES/PMULL backend only after runtime detection confirmed PMULL.
      // 2. `acc` and `h_polyval` are initialized GHASH field elements.
      acc = unsafe { polyval::aarch64_clmul128_reduce_inline(acc, h_polyval) };
      let expected = encrypt_j0_tag(&self.ek, &j0, acc);
      if !ct::constant_time_eq(&expected, tag.as_bytes()) {
        ct::zeroize(buffer);
        return Err(OpenError::verification());
      }
      return Ok(());
    }

    #[cfg(target_arch = "powerpc64")]
    if self.backend == AeadBackend::Power8Crypto {
      let h_polyval = self.h_powers_rev[3];
      // SAFETY: POWER8 GHASH AAD path because:
      // 1. Backend resolution selected `Power8Crypto` only after runtime detection confirmed POWER8
      //    crypto.
      // 2. `aad` is a valid byte slice; padding is handled inside the helper.
      let mut acc = unsafe { ghash_update_padded_wide_ppc(0, h_polyval, &self.h_powers_rev, aad) };
      // SAFETY: fused POWER AES-GCM open because:
      // 1. Backend resolution selected `Power8Crypto` only after runtime detection confirmed POWER8
      //    crypto.
      // 2. The helper GHASHes ciphertext before decrypting the valid buffer in place.
      acc = unsafe {
        aes::aes256_ctr32_decrypt_be_ppc_ghash(&self.ek, &ctr_block, buffer, acc, h_polyval, &self.h_powers_rev)
      };
      acc ^= u128::from_be_bytes(length_block);
      // SAFETY: POWER8 carryless multiply because backend resolution confirmed POWER8 crypto.
      acc = unsafe { polyval::ppc_clmul128_reduce_inline(acc, h_polyval) };
      let expected = encrypt_j0_tag(&self.ek, &j0, acc);
      if !ct::constant_time_eq(&expected, tag.as_bytes()) {
        ct::zeroize(buffer);
        return Err(OpenError::verification());
      }
      return Ok(());
    }

    // Scalar path: authenticate then decrypt.
    #[cfg(target_arch = "x86_64")]
    if self.backend == AeadBackend::X86VaesVpclmul
      && let Some(expected) =
        compute_tag_short_wide(&self.ek, self.h_powers_rev[3], &self.h_powers_rev, &j0, aad, buffer)
          .map_err(|_| OpenError::too_large())?
    {
      if !ct::constant_time_eq(&expected, tag.as_bytes()) {
        ct::zeroize(buffer);
        return Err(OpenError::verification());
      }
      aes::aes256_ctr32_encrypt_be(&self.ek, &ctr_block, buffer);
      return Ok(());
    }
    if should_use_wide_ghash(self.backend, aad.len(), buffer.len()) {
      let expected = compute_tag_wide(&self.ek, self.h_powers_rev[3], &self.h_powers_rev, &j0, aad, buffer)
        .map_err(|_| OpenError::too_large())?;
      if !ct::constant_time_eq(&expected, tag.as_bytes()) {
        ct::zeroize(buffer);
        return Err(OpenError::verification());
      }
      aes::aes256_ctr32_encrypt_be(&self.ek, &ctr_block, buffer);
      return Ok(());
    }
    let expected = compute_tag(&self.ek, self.h_powers_rev[3], &j0, aad, buffer).map_err(|_| OpenError::too_large())?;
    if !ct::constant_time_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Err(OpenError::verification());
    }
    aes::aes256_ctr32_encrypt_be(&self.ek, &ctr_block, buffer);
    Ok(())
  }
}

// Aes256Gcm stores the expanded key which contains sensitive material.
impl Drop for Aes256Gcm {
  fn drop(&mut self) {
    ct::zeroize(&mut self.h);
    // SAFETY: H-power byte view because:
    // 1. `[u128; 4]` is a contiguous initialized 64-byte array.
    // 2. `self` is mutably borrowed during drop, so no alias observes the byte view.
    ct::zeroize(unsafe { core::slice::from_raw_parts_mut(self.h_powers_rev.as_mut_ptr().cast::<u8>(), 64) });
    // SAFETY: H-power byte view because:
    // 1. `[u128; 8]` is a contiguous initialized 128-byte array.
    // 2. `self` is mutably borrowed during drop, so no alias observes the byte view.
    ct::zeroize(unsafe { core::slice::from_raw_parts_mut(self.h_powers_rev_8.as_mut_ptr().cast::<u8>(), 128) });
    #[cfg(any(
      target_arch = "x86_64",
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux"))
    ))]
    {
      // SAFETY: H-power byte view because:
      // 1. `[u128; 16]` is a contiguous initialized 256-byte array.
      // 2. `self` is mutably borrowed during drop, so no alias observes the byte view.
      ct::zeroize(unsafe { core::slice::from_raw_parts_mut(self.h_powers_rev_16.as_mut_ptr().cast::<u8>(), 256) });
    }
    #[cfg(target_arch = "x86_64")]
    {
      // SAFETY: H-power byte view because:
      // 1. `[u128; 32]` is a contiguous initialized 512-byte array.
      // 2. `self` is mutably borrowed during drop, so no alias observes the byte view.
      ct::zeroize(unsafe { core::slice::from_raw_parts_mut(self.h_powers_rev_32.as_mut_ptr().cast::<u8>(), 512) });
      // SAFETY: H-power byte view because:
      // 1. `[u128; 64]` is a contiguous initialized 1024-byte array.
      // 2. `self` is mutably borrowed during drop, so no alias observes the byte view.
      ct::zeroize(unsafe { core::slice::from_raw_parts_mut(self.h_powers_rev_64.as_mut_ptr().cast::<u8>(), 1024) });
      // SAFETY: H-power byte view because:
      // 1. `[u128; 128]` is a contiguous initialized 2048-byte array.
      // 2. `self` is mutably borrowed during drop, so no alias observes the byte view.
      ct::zeroize(unsafe { core::slice::from_raw_parts_mut(self.h_powers_rev_128.as_mut_ptr().cast::<u8>(), 2048) });
    }
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
    {
      // SAFETY: H-power byte view because:
      // 1. `[u128; 16]` is a contiguous initialized 256-byte array.
      // 2. `self` is mutably borrowed during drop, so no alias observes the byte view.
      ct::zeroize(unsafe { core::slice::from_raw_parts_mut(self.h_powers_rev_16_mid.as_mut_ptr().cast::<u8>(), 256) });
      // SAFETY: H-power byte view because:
      // 1. `[u128; 24]` is a contiguous initialized 384-byte array.
      // 2. `self` is mutably borrowed during drop, so no alias observes the byte view.
      ct::zeroize(unsafe { core::slice::from_raw_parts_mut(self.h_powers_rev_16_pair.as_mut_ptr().cast::<u8>(), 384) });
    }
    // ek's Drop is handled by Aes256EncKey's own Drop impl.
  }
}

// Tests

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
