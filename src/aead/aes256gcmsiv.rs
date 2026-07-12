#![allow(clippy::indexing_slicing)]

//! AES-256-GCM-SIV public AEAD surface (RFC 8452).

use core::fmt;

#[cfg(target_arch = "x86_64")]
use super::polyval::{accumulate_padded_x86, precompute_powers, precompute_powers_16};
use super::{
  AeadBufferError, Nonce96, OpenError, SealError, aes, polyval,
  targets::{AeadBackend, AeadPrimitive, select_backend},
};
use crate::traits::{Aead, ct};

const KEY_SIZE: usize = 32;
const TAG_SIZE: usize = 16;
const NONCE_SIZE: usize = Nonce96::LENGTH;

/// Maximum plaintext length: 2^36 - 32 bytes (per RFC 8452 §5).
/// Beyond this the 32-bit CTR counter wraps, causing keystream reuse.
const MAX_PLAINTEXT_LEN: u64 = (1u64 << 36).strict_sub(32);

define_aead_key_type!(Aes256GcmSivKey, KEY_SIZE, "AES-256-GCM-SIV secret key (32 bytes).");

define_aead_tag_type!(
  Aes256GcmSivTag,
  TAG_SIZE,
  "AES-256-GCM-SIV authentication tag (16 bytes)."
);

/// AES-256-GCM-SIV AEAD (RFC 8452).
///
/// Nonce-misuse resistant authenticated encryption. On nonce reuse, only
/// the authentication guarantee degrades — confidentiality is preserved
/// up to a multi-message distinguishing bound.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "getrandom")]
/// # {
/// use rscrypto::{Aead, Aes256GcmSiv, Aes256GcmSivKey};
///
/// let key = Aes256GcmSivKey::from_bytes([0x42; 32]);
/// let cipher = Aes256GcmSiv::new(&key);
///
/// let mut buf = *b"hello";
/// let (nonce, tag) = cipher.seal_random_in_place(b"", &mut buf)?;
/// cipher.decrypt_in_place(&nonce, b"", &mut buf, &tag)?;
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
/// use rscrypto::{Aead, Aes256GcmSiv, Aes256GcmSivKey, aead::OpenError};
///
/// let key = Aes256GcmSivKey::from_bytes([0x42; 32]);
/// let cipher = Aes256GcmSiv::new(&key);
///
/// let mut sealed = [0u8; 5 + Aes256GcmSiv::TAG_SIZE];
/// let nonce = cipher.seal_random(b"", b"hello", &mut sealed)?;
/// sealed[0] ^= 1;
///
/// let mut opened = [0u8; 5];
/// assert_eq!(
///   cipher.decrypt(&nonce, b"", &sealed, &mut opened),
///   Err(OpenError::verification())
/// );
/// # }
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
pub struct Aes256GcmSiv {
  master_ek: aes::Aes256EncKey,
  #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
  backend: AeadBackend,
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
  pub fn encrypt_in_place(&self, nonce: &Nonce96, aad: &[u8], buffer: &mut [u8]) -> Result<Aes256GcmSivTag, SealError> {
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

// RFC 8452 construction internals

/// Derive per-message authentication and encryption keys from the cached
/// master-key schedule and nonce (RFC 8452 §4, AES-256 variant).
///
/// Returns (auth_key [16 bytes], enc_key [32 bytes]).
///
/// Prepares all 6 counter blocks up front and encrypts them in a single
/// batch call. On s390x this collapses 6 individual KM invocations into
/// one, eliminating per-block overhead that dominates at small sizes.
#[inline]
fn derive_keys(master_ek: &aes::Aes256EncKey, nonce: &Nonce96) -> ([u8; 16], [u8; 32]) {
  let nonce_bytes = nonce.as_bytes();

  // Build all 6 counter blocks: counter (LE32) || nonce (96 bits).
  let mut blocks = [[0u8; 16]; 6];
  let mut i = 0u32;
  while i < 6 {
    blocks[i as usize][0..4].copy_from_slice(&i.to_le_bytes());
    blocks[i as usize][4..16].copy_from_slice(nonce_bytes);
    i = i.strict_add(1);
  }

  // Encrypt all 6 blocks (single KM call on s390x, per-block elsewhere).
  aes::aes256_encrypt_blocks_ecb(master_ek, &mut blocks);

  // Extract the first 8 bytes of each encrypted block.
  let mut auth_key = [0u8; 16];
  let mut enc_key = [0u8; 32];
  auth_key[0..8].copy_from_slice(&blocks[0][0..8]);
  auth_key[8..16].copy_from_slice(&blocks[1][0..8]);
  enc_key[0..8].copy_from_slice(&blocks[2][0..8]);
  enc_key[8..16].copy_from_slice(&blocks[3][0..8]);
  enc_key[16..24].copy_from_slice(&blocks[4][0..8]);
  enc_key[24..32].copy_from_slice(&blocks[5][0..8]);

  ct::zeroize(blocks.as_flattened_mut());
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
  let length_block = super::AeadByteLengths::from_usize(aad.len(), plaintext.len()).to_le_bits_block();
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

#[cfg(feature = "diag")]
#[must_use]
pub fn diag_aes256gcmsiv_derive_keys(cipher: &Aes256GcmSiv, nonce: &Nonce96) -> ([u8; 16], [u8; 32]) {
  derive_keys(&cipher.master_ek, nonce)
}

#[cfg(feature = "diag")]
#[must_use]
pub fn diag_aes256gcmsiv_raw_tag_aes(enc_key: &[u8; 32], block: &[u8; 16]) -> [u8; 16] {
  let mut out = *block;
  #[cfg(target_arch = "s390x")]
  {
    // SAFETY: diagnostic s390x CT runs execute on the native MSA runner.
    unsafe { aes::s390x_encrypt_block_raw_inline(enc_key, &mut out) };
  }
  #[cfg(not(target_arch = "s390x"))]
  {
    let ek = aes::aes256_expand_key(enc_key);
    aes::aes256_encrypt_block(&ek, &mut out);
  }
  out
}

#[cfg(feature = "diag")]
#[must_use]
pub fn diag_aes256gcmsiv_ctr32(enc_key: &[u8; 32], tag: &[u8; 16], plaintext: &[u8; 44]) -> [u8; 16] {
  let mut counter_block = *tag;
  counter_block[15] |= 0x80;
  let mut buffer = *plaintext;
  #[cfg(target_arch = "s390x")]
  {
    let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
    let mut offset = 0usize;
    while offset < buffer.len() {
      counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
      let mut keystream = counter_block;
      // SAFETY: diagnostic s390x CT runs execute on the native MSA runner.
      unsafe { aes::s390x_encrypt_block_raw_inline(enc_key, &mut keystream) };

      let remaining = buffer.len().strict_sub(offset);
      if remaining >= 16 {
        let mut d = [0u8; 16];
        d.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
        let xored = u128::from_ne_bytes(d) ^ u128::from_ne_bytes(keystream);
        buffer[offset..offset.strict_add(16)].copy_from_slice(&xored.to_ne_bytes());
        offset = offset.strict_add(16);
      } else {
        let mut i = 0usize;
        while i < remaining {
          buffer[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }
  }
  #[cfg(not(target_arch = "s390x"))]
  {
    let ek = aes::aes256_expand_key(enc_key);
    aes::aes256_ctr32_encrypt(&ek, &counter_block, &mut buffer);
  }
  diag_fold16(&buffer)
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

#[cfg(target_arch = "riscv64")]
#[derive(Clone, Copy)]
enum RiscvPolyvalBackend {
  Portable,
  Scalar,
  Vector,
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn reduce_riscv_portable(a: u128, b: u128) -> u128 {
  polyval::portable_clmul128_reduce_inline(a, b)
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn reduce_riscv_scalar(a: u128, b: u128) -> u128 {
  // SAFETY: caller only selects this reducer after runtime detection confirms
  // Zbc or Zbkc support.
  unsafe { polyval::riscv_scalar_clmul128_reduce_inline(a, b) }
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn reduce_riscv_vector(a: u128, b: u128) -> u128 {
  // SAFETY: caller only selects this reducer after runtime detection confirms
  // Zvbc support.
  unsafe { polyval::riscv_vector_clmul128_reduce_inline(a, b) }
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn compute_tag_riscv_with_reduce(
  auth_key: &[u8; 16],
  enc_ek: &aes::Aes256EncKey,
  nonce: &Nonce96,
  aad: &[u8],
  plaintext: &[u8],
  reduce: impl Fn(u128, u128) -> u128,
) -> [u8; TAG_SIZE] {
  let h = u128::from_le_bytes(*auth_key);
  let mut acc: u128 = 0;

  let mut offset = 0usize;
  while offset.strict_add(16) <= aad.len() {
    let mut block = [0u8; 16];
    block.copy_from_slice(&aad[offset..offset.strict_add(16)]);
    acc ^= u128::from_le_bytes(block);
    acc = reduce(acc, h);
    offset = offset.strict_add(16);
  }
  if offset < aad.len() {
    let mut block = [0u8; 16];
    block[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
    acc ^= u128::from_le_bytes(block);
    acc = reduce(acc, h);
  }

  offset = 0;
  while offset.strict_add(16) <= plaintext.len() {
    let mut block = [0u8; 16];
    block.copy_from_slice(&plaintext[offset..offset.strict_add(16)]);
    acc ^= u128::from_le_bytes(block);
    acc = reduce(acc, h);
    offset = offset.strict_add(16);
  }
  if offset < plaintext.len() {
    let mut block = [0u8; 16];
    block[..plaintext.len().strict_sub(offset)].copy_from_slice(&plaintext[offset..]);
    acc ^= u128::from_le_bytes(block);
    acc = reduce(acc, h);
  }

  let length_block = super::AeadByteLengths::from_usize(aad.len(), plaintext.len()).to_le_bits_block();
  acc ^= u128::from_le_bytes(length_block);
  acc = reduce(acc, h);

  let mut s = acc.to_le_bytes();
  let nonce_bytes = nonce.as_bytes();
  let mut j = 0usize;
  while j < 12 {
    s[j] ^= nonce_bytes[j];
    j = j.strict_add(1);
  }
  s[15] &= 0x7f;
  aes::aes256_encrypt_block(enc_ek, &mut s);
  s
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn expand_key_riscv_for_backend(key: &[u8; 32], backend: AeadBackend) -> aes::Aes256EncKey {
  match backend {
    AeadBackend::Riscv64VectorCrypto => aes::aes256_expand_key_riscv_vector(key),
    AeadBackend::Riscv64ScalarCrypto => aes::aes256_expand_key_riscv_scalar(key),
    AeadBackend::Riscv64Vperm => aes::aes256_expand_key_riscv_vperm(key),
    AeadBackend::Portable => aes::aes256_expand_key_riscv_ttable(key),
    _ => aes::aes256_expand_key_riscv_ttable(key),
  }
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn expand_message_key_riscv(enc_key: &[u8; 32], backend: AeadBackend) -> aes::Aes256EncKey {
  expand_key_riscv_for_backend(enc_key, backend)
}

#[inline]
fn resolve_backend() -> AeadBackend {
  select_backend(
    AeadPrimitive::Aes256GcmSiv,
    crate::platform::arch(),
    crate::platform::caps(),
  )
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn riscv_polyval_backend(backend: AeadBackend) -> RiscvPolyvalBackend {
  match backend {
    AeadBackend::Riscv64VectorCrypto => RiscvPolyvalBackend::Vector,
    AeadBackend::Riscv64ScalarCrypto => RiscvPolyvalBackend::Scalar,
    AeadBackend::Portable | AeadBackend::Riscv64Vperm => {
      let caps = crate::platform::caps();
      if caps.has(crate::platform::caps::riscv::ZBC) || caps.has(crate::platform::caps::riscv::ZBKC) {
        RiscvPolyvalBackend::Scalar
      } else {
        RiscvPolyvalBackend::Portable
      }
    }
    _ => RiscvPolyvalBackend::Portable,
  }
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn compute_tag_riscv(
  auth_key: &[u8; 16],
  enc_ek: &aes::Aes256EncKey,
  nonce: &Nonce96,
  aad: &[u8],
  plaintext: &[u8],
  backend: AeadBackend,
) -> [u8; TAG_SIZE] {
  match riscv_polyval_backend(backend) {
    RiscvPolyvalBackend::Portable => {
      compute_tag_riscv_with_reduce(auth_key, enc_ek, nonce, aad, plaintext, reduce_riscv_portable)
    }
    RiscvPolyvalBackend::Scalar => {
      compute_tag_riscv_with_reduce(auth_key, enc_ek, nonce, aad, plaintext, reduce_riscv_scalar)
    }
    RiscvPolyvalBackend::Vector => {
      compute_tag_riscv_with_reduce(auth_key, enc_ek, nonce, aad, plaintext, reduce_riscv_vector)
    }
  }
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn encrypt_riscv(
  master_ek: &aes::Aes256EncKey,
  backend: AeadBackend,
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  let (mut auth_key, mut enc_key) = derive_keys(master_ek, nonce);
  let ek = expand_message_key_riscv(&enc_key, backend);
  let tag_bytes = compute_tag_riscv(&auth_key, &ek, nonce, aad, buffer, backend);
  let mut counter_block = tag_bytes;
  counter_block[15] |= 0x80;
  aes::aes256_ctr32_encrypt(&ek, &counter_block, buffer);
  ct::zeroize(&mut auth_key);
  ct::zeroize(&mut enc_key);
  tag_bytes
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn decrypt_riscv(
  master_ek: &aes::Aes256EncKey,
  backend: AeadBackend,
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  tag: &Aes256GcmSivTag,
) -> Result<(), crate::traits::VerificationError> {
  let (mut auth_key, mut enc_key) = derive_keys(master_ek, nonce);
  let ek = expand_message_key_riscv(&enc_key, backend);
  let mut counter_block = tag.0;
  counter_block[15] |= 0x80;
  aes::aes256_ctr32_encrypt(&ek, &counter_block, buffer);

  let expected = compute_tag_riscv(&auth_key, &ek, nonce, aad, buffer, backend);
  ct::zeroize(&mut auth_key);
  ct::zeroize(&mut enc_key);

  if !ct::constant_time_eq(&expected, tag.as_bytes()) {
    ct::zeroize(buffer);
    return Err(crate::traits::VerificationError::new());
  }
  Ok(())
}

/// Compute the POLYVAL-based authentication tag using 4-block wide processing.
///
/// Same semantics as `compute_tag` but processes data in 4-block (64-byte)
/// chunks via `accumulate_4blocks`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn compute_tag_wide(
  auth_key: &[u8; 16],
  enc_ek: &aes::Aes256EncKey,
  nonce: &Nonce96,
  aad: &[u8],
  plaintext: &[u8],
) -> [u8; TAG_SIZE] {
  let h = u128::from_le_bytes(*auth_key);
  let (h_powers_rev, h_powers_rev_16) = if aad.len() >= 256 || plaintext.len() >= 256 {
    let powers_16 = precompute_powers_16(h);
    (
      [powers_16[3], powers_16[2], powers_16[1], powers_16[0]],
      Some(core::array::from_fn(|i| powers_16[15usize.strict_sub(i)])),
    )
  } else {
    let powers = precompute_powers(h);
    ([powers[3], powers[2], powers[1], powers[0]], None)
  };

  let mut acc: u128 = 0;
  acc = accumulate_padded_x86(acc, h, &h_powers_rev, h_powers_rev_16.as_ref(), aad);
  acc = accumulate_padded_x86(acc, h, &h_powers_rev, h_powers_rev_16.as_ref(), plaintext);

  // Length block: [aad_bits as u64 LE || pt_bits as u64 LE].
  let length_block = super::AeadByteLengths::from_usize(aad.len(), plaintext.len()).to_le_bits_block();
  acc ^= u128::from_le_bytes(length_block);
  acc = polyval::clmul128_reduce(acc, h);

  // Finalize: convert to bytes, XOR nonce, clear MSB, encrypt.
  let mut s = acc.to_le_bytes();
  let nonce_bytes = nonce.as_bytes();
  let mut j = 0usize;
  while j < 12 {
    s[j] ^= nonce_bytes[j];
    j = j.strict_add(1);
  }
  s[15] &= 0x7f;
  aes::aes256_encrypt_block(enc_ek, &mut s);
  s
}

// aarch64 fused encrypt/decrypt (single #[target_feature] scope)
//
// On aarch64 the "scalar path" crosses a #[target_feature] boundary for every
// AES-CE encrypt_block and PMULL clmul128_reduce call, causing register spills
// that dominate at small message sizes. The fused paths below inline the entire
// GCM-SIV construction — key derivation, POLYVAL, tag finalization, AES-CTR —
// into one #[target_feature(enable = "aes,neon")] scope, matching the AEGIS
// pattern. The compiler keeps AES round keys and POLYVAL state in NEON
// registers across the whole operation.

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes,neon")]
unsafe fn encrypt_fused_aarch64(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 32],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  // SAFETY: fused AArch64 AES-256-GCM-SIV encryption because:
  // 1. This function has `#[target_feature(enable = "aes,neon")]`.
  // 2. The caller verifies AES-CE availability before dispatching here.
  // 3. `auth_key`, `enc_key_bytes`, nonce, AAD, and buffer are initialized caller-owned inputs.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    // --- 1. Expand derived encryption key ---
    let enc_ek = aes::aarch64_expand_key_inline(enc_key_bytes);
    ct::zeroize(enc_key_bytes);

    // --- 2. POLYVAL tag computation (RFC 8452 §5) ---
    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    // Precompute H powers only when at least one 4-block chunk will run.
    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

    // Process AAD in 4-block (64-byte) chunks.
    let mut offset = 0usize;
    while offset.strict_add(64) <= aad.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&aad[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::aarch64_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining AAD single blocks.
    while offset.strict_add(16) <= aad.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&aad[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::aarch64_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < aad.len() {
      let mut block = [0u8; 16];
      block[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::aarch64_clmul128_reduce_inline(acc, h);
    }

    // Process plaintext in 4-block (64-byte) chunks.
    offset = 0;
    while offset.strict_add(64) <= buffer.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&buffer[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::aarch64_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining plaintext single blocks.
    while offset.strict_add(16) <= buffer.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::aarch64_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < buffer.len() {
      let mut block = [0u8; 16];
      block[..buffer.len().strict_sub(offset)].copy_from_slice(&buffer[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::aarch64_clmul128_reduce_inline(acc, h);
    }

    // Length block: [aad_bits as u64 LE || pt_bits as u64 LE].
    let length_block = super::AeadByteLengths::from_usize(aad.len(), buffer.len()).to_le_bits_block();
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::aarch64_clmul128_reduce_inline(acc, h);

    // --- 4. Tag finalization ---
    let mut tag = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      tag[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    tag[15] &= 0x7f;
    aes::aarch64_encrypt_block_inline(&enc_ek, &mut tag);

    // --- 5. AES-CTR encryption ---
    let mut counter_block = tag;
    counter_block[15] |= 0x80;
    let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
    let iv_suffix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&counter_block[4..16]);
      buf
    };
    offset = 0;
    while offset.strict_add(128) <= buffer.len() {
      let end = offset.strict_add(128);
      aes::aarch64_ctr32_le_xor_8blocks_inline(&enc_ek, &iv_suffix, ctr, &mut buffer[offset..end]);
      ctr = ctr.wrapping_add(8);
      offset = end;
    }
    while offset.strict_add(64) <= buffer.len() {
      let end = offset.strict_add(64);
      aes::aarch64_ctr32_le_xor_4blocks_inline(&enc_ek, &iv_suffix, ctr, &mut buffer[offset..end]);
      ctr = ctr.wrapping_add(4);
      offset = end;
    }
    while offset < buffer.len() {
      counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
      let mut keystream = counter_block;
      aes::aarch64_encrypt_block_inline(&enc_ek, &mut keystream);

      let remaining = buffer.len().strict_sub(offset);
      if remaining >= 16 {
        let mut d = [0u8; 16];
        d.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
        let xored = u128::from_ne_bytes(d) ^ u128::from_ne_bytes(keystream);
        buffer[offset..offset.strict_add(16)].copy_from_slice(&xored.to_ne_bytes());
        offset = offset.strict_add(16);
      } else {
        let mut i = 0usize;
        while i < remaining {
          buffer[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    tag
  }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes,neon")]
unsafe fn decrypt_fused_aarch64(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 32],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  tag: &Aes256GcmSivTag,
) -> Result<(), crate::traits::VerificationError> {
  // SAFETY: fused AArch64 AES-256-GCM-SIV decryption because:
  // 1. This function has `#[target_feature(enable = "aes,neon")]`.
  // 2. The caller verifies AES-CE availability before dispatching here.
  // 3. `auth_key`, `enc_key_bytes`, nonce, AAD, buffer, and tag are initialized caller-owned inputs.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    // --- 1. Expand derived encryption key ---
    let enc_ek = aes::aarch64_expand_key_inline(enc_key_bytes);
    ct::zeroize(enc_key_bytes);

    // --- 2. AES-CTR decryption (SIV: decrypt before verify) ---
    let mut counter_block = tag.0;
    counter_block[15] |= 0x80;
    let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
    let iv_suffix: [u8; 12] = {
      let mut buf = [0u8; 12];
      buf.copy_from_slice(&counter_block[4..16]);
      buf
    };
    let mut offset = 0usize;
    while offset.strict_add(128) <= buffer.len() {
      let end = offset.strict_add(128);
      aes::aarch64_ctr32_le_xor_8blocks_inline(&enc_ek, &iv_suffix, ctr, &mut buffer[offset..end]);
      ctr = ctr.wrapping_add(8);
      offset = end;
    }
    while offset.strict_add(64) <= buffer.len() {
      let end = offset.strict_add(64);
      aes::aarch64_ctr32_le_xor_4blocks_inline(&enc_ek, &iv_suffix, ctr, &mut buffer[offset..end]);
      ctr = ctr.wrapping_add(4);
      offset = end;
    }
    while offset < buffer.len() {
      counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
      let mut keystream = counter_block;
      aes::aarch64_encrypt_block_inline(&enc_ek, &mut keystream);

      let remaining = buffer.len().strict_sub(offset);
      if remaining >= 16 {
        let mut d = [0u8; 16];
        d.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
        let xored = u128::from_ne_bytes(d) ^ u128::from_ne_bytes(keystream);
        buffer[offset..offset.strict_add(16)].copy_from_slice(&xored.to_ne_bytes());
        offset = offset.strict_add(16);
      } else {
        let mut i = 0usize;
        while i < remaining {
          buffer[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    // --- 3. POLYVAL tag computation over decrypted plaintext ---
    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    // Precompute H powers only when at least one 4-block chunk will run.
    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

    // Process AAD in 4-block (64-byte) chunks.
    offset = 0;
    while offset.strict_add(64) <= aad.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&aad[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::aarch64_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining AAD single blocks.
    while offset.strict_add(16) <= aad.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&aad[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::aarch64_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < aad.len() {
      let mut block = [0u8; 16];
      block[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::aarch64_clmul128_reduce_inline(acc, h);
    }

    // Process decrypted plaintext in 4-block (64-byte) chunks.
    offset = 0;
    while offset.strict_add(64) <= buffer.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&buffer[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::aarch64_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining plaintext single blocks.
    while offset.strict_add(16) <= buffer.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::aarch64_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < buffer.len() {
      let mut block = [0u8; 16];
      block[..buffer.len().strict_sub(offset)].copy_from_slice(&buffer[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::aarch64_clmul128_reduce_inline(acc, h);
    }

    // Length block.
    let length_block = super::AeadByteLengths::from_usize(aad.len(), buffer.len()).to_le_bits_block();
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::aarch64_clmul128_reduce_inline(acc, h);

    // --- 4. Tag finalization + verification ---
    let mut expected = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      expected[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    expected[15] &= 0x7f;
    aes::aarch64_encrypt_block_inline(&enc_ek, &mut expected);

    if !ct::constant_time_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Err(crate::traits::VerificationError::new());
    }
    Ok(())
  }
}

// powerpc64 fused encrypt/decrypt (single #[target_feature] scope)

#[cfg(target_arch = "powerpc64")]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
unsafe fn encrypt_fused_ppc(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 32],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  // SAFETY: fused POWER8 AES-256-GCM-SIV encryption because:
  // 1. This function has POWER8 crypto target features enabled.
  // 2. The caller verifies POWER8 crypto availability before dispatching here.
  // 3. `auth_key`, `enc_key_bytes`, nonce, AAD, and buffer are initialized caller-owned inputs.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    // --- 1. Expand derived encryption key ---
    let enc_ek = aes::ppc_expand_key_inline(enc_key_bytes);
    ct::zeroize(enc_key_bytes);

    // --- 2. POLYVAL tag computation (RFC 8452 §5) ---
    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    // Precompute H powers only when at least one 4-block chunk will run.
    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

    // Process AAD in 4-block (64-byte) chunks.
    let mut offset = 0usize;
    while offset.strict_add(64) <= aad.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&aad[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::ppc_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining AAD single blocks.
    while offset.strict_add(16) <= aad.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&aad[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::ppc_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < aad.len() {
      let mut block = [0u8; 16];
      block[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::ppc_clmul128_reduce_inline(acc, h);
    }

    // Process plaintext in 4-block (64-byte) chunks.
    offset = 0;
    while offset.strict_add(64) <= buffer.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&buffer[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::ppc_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining plaintext single blocks.
    while offset.strict_add(16) <= buffer.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::ppc_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < buffer.len() {
      let mut block = [0u8; 16];
      block[..buffer.len().strict_sub(offset)].copy_from_slice(&buffer[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::ppc_clmul128_reduce_inline(acc, h);
    }

    // Length block: [aad_bits as u64 LE || pt_bits as u64 LE].
    let length_block = super::AeadByteLengths::from_usize(aad.len(), buffer.len()).to_le_bits_block();
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::ppc_clmul128_reduce_inline(acc, h);

    // --- 4. Tag finalization ---
    let mut tag = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      tag[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    tag[15] &= 0x7f;
    aes::ppc_encrypt_block_inline(&enc_ek, &mut tag);

    // --- 5. AES-CTR encryption ---
    let mut counter_block = tag;
    counter_block[15] |= 0x80;
    let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
    offset = 0;
    while offset < buffer.len() {
      counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
      let mut keystream = counter_block;
      aes::ppc_encrypt_block_inline(&enc_ek, &mut keystream);

      let remaining = buffer.len().strict_sub(offset);
      if remaining >= 16 {
        let mut d = [0u8; 16];
        d.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
        let xored = u128::from_ne_bytes(d) ^ u128::from_ne_bytes(keystream);
        buffer[offset..offset.strict_add(16)].copy_from_slice(&xored.to_ne_bytes());
        offset = offset.strict_add(16);
      } else {
        let mut i = 0usize;
        while i < remaining {
          buffer[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    tag
  }
}

#[cfg(target_arch = "powerpc64")]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
unsafe fn decrypt_fused_ppc(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 32],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  tag: &Aes256GcmSivTag,
) -> Result<(), crate::traits::VerificationError> {
  // SAFETY: fused POWER8 AES-256-GCM-SIV decryption because:
  // 1. This function has POWER8 crypto target features enabled.
  // 2. The caller verifies POWER8 crypto availability before dispatching here.
  // 3. `auth_key`, `enc_key_bytes`, nonce, AAD, buffer, and tag are initialized caller-owned inputs.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    // --- 1. Expand derived encryption key ---
    let enc_ek = aes::ppc_expand_key_inline(enc_key_bytes);
    ct::zeroize(enc_key_bytes);

    // --- 2. AES-CTR decryption (SIV: decrypt before verify) ---
    let mut counter_block = tag.0;
    counter_block[15] |= 0x80;
    let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
    let mut offset = 0usize;
    while offset < buffer.len() {
      counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
      let mut keystream = counter_block;
      aes::ppc_encrypt_block_inline(&enc_ek, &mut keystream);

      let remaining = buffer.len().strict_sub(offset);
      if remaining >= 16 {
        let mut d = [0u8; 16];
        d.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
        let xored = u128::from_ne_bytes(d) ^ u128::from_ne_bytes(keystream);
        buffer[offset..offset.strict_add(16)].copy_from_slice(&xored.to_ne_bytes());
        offset = offset.strict_add(16);
      } else {
        let mut i = 0usize;
        while i < remaining {
          buffer[offset.strict_add(i)] ^= keystream[i];
          i = i.strict_add(1);
        }
        offset = offset.strict_add(remaining);
      }
      ctr = ctr.wrapping_add(1);
    }

    // --- 3. POLYVAL tag computation over decrypted plaintext ---
    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    // Precompute H powers only when at least one 4-block chunk will run.
    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

    // Process AAD in 4-block (64-byte) chunks.
    offset = 0;
    while offset.strict_add(64) <= aad.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&aad[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::ppc_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining AAD single blocks.
    while offset.strict_add(16) <= aad.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&aad[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::ppc_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < aad.len() {
      let mut block = [0u8; 16];
      block[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::ppc_clmul128_reduce_inline(acc, h);
    }

    // Process decrypted plaintext in 4-block (64-byte) chunks.
    offset = 0;
    while offset.strict_add(64) <= buffer.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&buffer[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::ppc_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining plaintext single blocks.
    while offset.strict_add(16) <= buffer.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::ppc_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < buffer.len() {
      let mut block = [0u8; 16];
      block[..buffer.len().strict_sub(offset)].copy_from_slice(&buffer[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::ppc_clmul128_reduce_inline(acc, h);
    }

    // Length block.
    let length_block = super::AeadByteLengths::from_usize(aad.len(), buffer.len()).to_le_bits_block();
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::ppc_clmul128_reduce_inline(acc, h);

    // --- 4. Tag finalization + verification ---
    let mut expected = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      expected[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    expected[15] &= 0x7f;
    aes::ppc_encrypt_block_inline(&enc_ek, &mut expected);

    if !ct::constant_time_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Err(crate::traits::VerificationError::new());
    }
    Ok(())
  }
}

// s390x fused encrypt/decrypt (#[target_feature(enable = "vector")] for POLYVAL)

#[cfg(target_arch = "s390x")]
unsafe fn s390x_ctr32_le_xor_raw(enc_key_bytes: &[u8; 32], counter_block: &mut [u8; 16], buffer: &mut [u8]) {
  let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
  let mut offset = 0usize;

  while offset < buffer.len() {
    let remaining = buffer.len().strict_sub(offset);
    let block_count = aes::ctr_tail_block_count(remaining);
    let mut keystream = [[0u8; 16]; 4];
    let mut i = 0u32;
    while (i as usize) < block_count {
      keystream[i as usize][0..4].copy_from_slice(&ctr.wrapping_add(i).to_le_bytes());
      keystream[i as usize][4..16].copy_from_slice(&counter_block[4..16]);
      i = i.strict_add(1);
    }

    let flat_len = block_count.strict_mul(16);
    // SAFETY: `keystream` is a contiguous four-block array. `flat_len`
    // is `block_count * 16`, and `block_count <= 4`.
    let flat = unsafe { core::slice::from_raw_parts_mut(keystream.as_mut_ptr().cast::<u8>(), flat_len) };
    // SAFETY: caller guarantees MSA; `flat` spans exactly `block_count` initialized blocks.
    unsafe { aes::s390x_encrypt_blocks_raw_inline(enc_key_bytes, flat, block_count) };

    let processed = aes::xor_keystream_tail(buffer, offset, &keystream, block_count);
    offset = offset.strict_add(processed);
    ctr = ctr.wrapping_add(block_count as u32);
  }

  counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
}

#[cfg(target_arch = "s390x")]
#[target_feature(enable = "vector")]
unsafe fn encrypt_fused_s390x(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 32],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  // SAFETY: fused s390x AES-256-GCM-SIV encryption because:
  // 1. This path only runs after z/Vector + MSA availability is verified.
  // 2. `auth_key` and `enc_key_bytes` are fixed-size initialized derived keys.
  // 3. Nonce, AAD, and buffer are initialized caller-owned inputs.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    // --- 2. POLYVAL tag computation (RFC 8452 §5) ---
    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    // Precompute H powers only when at least one 4-block chunk will run.
    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

    // Process AAD in 4-block (64-byte) chunks.
    let mut offset = 0usize;
    while offset.strict_add(64) <= aad.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&aad[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::s390x_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining AAD single blocks.
    while offset.strict_add(16) <= aad.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&aad[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::s390x_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < aad.len() {
      let mut block = [0u8; 16];
      block[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::s390x_clmul128_reduce_inline(acc, h);
    }

    // Process plaintext in 4-block (64-byte) chunks.
    offset = 0;
    while offset.strict_add(64) <= buffer.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&buffer[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::s390x_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining plaintext single blocks.
    while offset.strict_add(16) <= buffer.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::s390x_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < buffer.len() {
      let mut block = [0u8; 16];
      block[..buffer.len().strict_sub(offset)].copy_from_slice(&buffer[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::s390x_clmul128_reduce_inline(acc, h);
    }

    // Length block: [aad_bits as u64 LE || pt_bits as u64 LE].
    let length_block = super::AeadByteLengths::from_usize(aad.len(), buffer.len()).to_le_bits_block();
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::s390x_clmul128_reduce_inline(acc, h);

    // --- 4. Tag finalization ---
    let mut tag = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      tag[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    tag[15] &= 0x7f;
    aes::s390x_encrypt_block_raw_inline(enc_key_bytes, &mut tag);

    // --- 5. AES-CTR encryption ---
    let mut counter_block = tag;
    counter_block[15] |= 0x80;
    s390x_ctr32_le_xor_raw(enc_key_bytes, &mut counter_block, buffer);

    ct::zeroize(enc_key_bytes);
    tag
  }
}

#[cfg(target_arch = "s390x")]
#[target_feature(enable = "vector")]
unsafe fn decrypt_fused_s390x(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 32],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  tag: &Aes256GcmSivTag,
) -> Result<(), crate::traits::VerificationError> {
  // SAFETY: fused s390x AES-256-GCM-SIV decryption because:
  // 1. This path only runs after z/Vector + MSA availability is verified.
  // 2. `auth_key` and `enc_key_bytes` are fixed-size initialized derived keys.
  // 3. Nonce, AAD, buffer, and tag are initialized caller-owned inputs.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    // --- 2. AES-CTR decryption (SIV: decrypt before verify) ---
    let mut counter_block = tag.0;
    counter_block[15] |= 0x80;
    s390x_ctr32_le_xor_raw(enc_key_bytes, &mut counter_block, buffer);

    // --- 3. POLYVAL tag computation over decrypted plaintext ---
    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    // Precompute H powers only when at least one 4-block chunk will run.
    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

    // Process AAD in 4-block (64-byte) chunks.
    let mut offset = 0usize;
    while offset.strict_add(64) <= aad.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&aad[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::s390x_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining AAD single blocks.
    while offset.strict_add(16) <= aad.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&aad[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::s390x_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < aad.len() {
      let mut block = [0u8; 16];
      block[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::s390x_clmul128_reduce_inline(acc, h);
    }

    // Process decrypted plaintext in 4-block (64-byte) chunks.
    offset = 0;
    while offset.strict_add(64) <= buffer.len() {
      let mut b = [0u128; 4];
      let mut i = 0usize;
      while i < 4 {
        let base = offset.strict_add(i.strict_mul(16));
        let mut block = [0u8; 16];
        block.copy_from_slice(&buffer[base..base.strict_add(16)]);
        b[i] = u128::from_le_bytes(block);
        i = i.strict_add(1);
      }
      acc = polyval::s390x_aggregate_4blocks_inline(acc, &h_powers_rev, &b);
      offset = offset.strict_add(64);
    }
    // Remaining plaintext single blocks.
    while offset.strict_add(16) <= buffer.len() {
      let mut block = [0u8; 16];
      block.copy_from_slice(&buffer[offset..offset.strict_add(16)]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::s390x_clmul128_reduce_inline(acc, h);
      offset = offset.strict_add(16);
    }
    if offset < buffer.len() {
      let mut block = [0u8; 16];
      block[..buffer.len().strict_sub(offset)].copy_from_slice(&buffer[offset..]);
      acc ^= u128::from_le_bytes(block);
      acc = polyval::s390x_clmul128_reduce_inline(acc, h);
    }

    // Length block.
    let length_block = super::AeadByteLengths::from_usize(aad.len(), buffer.len()).to_le_bits_block();
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::s390x_clmul128_reduce_inline(acc, h);

    // --- 4. Tag finalization + verification ---
    let mut expected = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      expected[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    expected[15] &= 0x7f;
    aes::s390x_encrypt_block_raw_inline(enc_key_bytes, &mut expected);
    ct::zeroize(enc_key_bytes);

    if !ct::constant_time_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Err(crate::traits::VerificationError::new());
    }
    Ok(())
  }
}

impl Aead for Aes256GcmSiv {
  const KEY_SIZE: usize = KEY_SIZE;
  const NONCE_SIZE: usize = NONCE_SIZE;
  const TAG_SIZE: usize = TAG_SIZE;

  type Key = Aes256GcmSivKey;
  type Nonce = Nonce96;
  type Tag = Aes256GcmSivTag;

  fn new(key: &Self::Key) -> Self {
    let backend = resolve_backend();

    Self {
      #[cfg(target_arch = "riscv64")]
      master_ek: expand_key_riscv_for_backend(key.as_bytes(), backend),
      #[cfg(not(target_arch = "riscv64"))]
      master_ek: aes::aes256_expand_key(key.as_bytes()),
      backend,
    }
  }

  fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
    if bytes.len() != TAG_SIZE {
      return Err(AeadBufferError::new());
    }
    let mut tag = [0u8; TAG_SIZE];
    tag.copy_from_slice(bytes);
    Ok(Aes256GcmSivTag::from_bytes(tag))
  }

  fn encrypt_in_place(&self, nonce: &Self::Nonce, aad: &[u8], buffer: &mut [u8]) -> Result<Self::Tag, SealError> {
    super::seal_bounded_length_as_u64(buffer.len(), MAX_PLAINTEXT_LEN)?;
    super::seal_bit_lengths(aad.len(), buffer.len())?;

    // Wide path: VPCLMULQDQ POLYVAL + VAES-512 CTR when available.
    #[cfg(target_arch = "x86_64")]
    if self.backend == AeadBackend::X86VaesVpclmul {
      let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
      let ek = aes::aes256_expand_key(&enc_key);
      let tag_bytes = compute_tag_wide(&auth_key, &ek, nonce, aad, buffer);
      let mut counter_block = tag_bytes;
      counter_block[15] |= 0x80;
      // SAFETY: VAES availability verified during backend resolution.
      unsafe { aes::aes256_ctr32_encrypt_wide(&ek, &counter_block, buffer) };
      ct::zeroize(&mut auth_key);
      ct::zeroize(&mut enc_key);
      return Ok(Aes256GcmSivTag::from_bytes(tag_bytes));
    }

    // Fused path: entire encrypt in a single #[target_feature] scope.
    #[cfg(target_arch = "aarch64")]
    if matches!(
      self.backend,
      AeadBackend::Aarch64AesPmull | AeadBackend::Aarch64Sve2AesPmull
    ) {
      // SAFETY: Direct AArch64 AES-256 GCM-SIV KDF because:
      // 1. Backend resolution selected an AArch64 AES+PMULL backend.
      // 2. The selected backend constructs `self.master_ek` with AES-CE round keys.
      // 3. `nonce.as_bytes()` is exactly the 96-bit GCM-SIV nonce.
      let (mut auth_key, mut enc_key) =
        unsafe { aes::aarch64_gcmsiv_derive_keys_inline(&self.master_ek, nonce.as_bytes()) };
      // SAFETY: AES-CE availability verified during backend resolution.
      let tag_bytes = unsafe { encrypt_fused_aarch64(&mut auth_key, &mut enc_key, nonce, aad, buffer) };
      return Ok(Aes256GcmSivTag::from_bytes(tag_bytes));
    }

    // Fused path: POWER8 crypto.
    #[cfg(target_arch = "powerpc64")]
    if self.backend == AeadBackend::Power8Crypto {
      let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
      // SAFETY: POWER8 crypto availability verified during backend resolution.
      let tag_bytes = unsafe { encrypt_fused_ppc(&mut auth_key, &mut enc_key, nonce, aad, buffer) };
      return Ok(Aes256GcmSivTag::from_bytes(tag_bytes));
    }

    // Fused path: s390x z/Vector + MSA.
    #[cfg(target_arch = "s390x")]
    if self.backend == AeadBackend::S390xMsa {
      let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
      // SAFETY: z/Vector + MSA availability verified during backend resolution.
      let tag_bytes = unsafe { encrypt_fused_s390x(&mut auth_key, &mut enc_key, nonce, aad, buffer) };
      return Ok(Aes256GcmSivTag::from_bytes(tag_bytes));
    }

    #[cfg(target_arch = "riscv64")]
    {
      match self.backend {
        AeadBackend::Portable
        | AeadBackend::Riscv64VectorCrypto
        | AeadBackend::Riscv64ScalarCrypto
        | AeadBackend::Riscv64Vperm => {
          let tag_bytes = encrypt_riscv(&self.master_ek, self.backend, nonce, aad, buffer);
          return Ok(Aes256GcmSivTag::from_bytes(tag_bytes));
        }
        _ => {}
      }
    }

    // Scalar path.
    let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
    let ek = aes::aes256_expand_key(&enc_key);
    let tag_bytes = compute_tag(&auth_key, &ek, nonce, aad, buffer);
    let mut counter_block = tag_bytes;
    counter_block[15] |= 0x80;
    aes::aes256_ctr32_encrypt(&ek, &counter_block, buffer);

    ct::zeroize(&mut auth_key);
    ct::zeroize(&mut enc_key);

    Ok(Aes256GcmSivTag::from_bytes(tag_bytes))
  }

  fn decrypt_in_place(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), OpenError> {
    super::open_bounded_length_as_u64(buffer.len(), MAX_PLAINTEXT_LEN)?;
    super::open_bit_lengths(aad.len(), buffer.len())?;

    // Wide path: VAES-512 CTR + VPCLMULQDQ POLYVAL when available.
    #[cfg(target_arch = "x86_64")]
    if self.backend == AeadBackend::X86VaesVpclmul {
      let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
      let ek = aes::aes256_expand_key(&enc_key);
      // Decrypt first (SIV pattern).
      let mut counter_block = tag.0;
      counter_block[15] |= 0x80;
      // SAFETY: VAES availability verified during backend resolution.
      unsafe { aes::aes256_ctr32_encrypt_wide(&ek, &counter_block, buffer) };

      // Verify tag over decrypted plaintext.
      let expected = compute_tag_wide(&auth_key, &ek, nonce, aad, buffer);
      ct::zeroize(&mut auth_key);
      ct::zeroize(&mut enc_key);
      if !ct::constant_time_eq(&expected, tag.as_bytes()) {
        ct::zeroize(buffer);
        return Err(OpenError::verification());
      }
      return Ok(());
    }

    // Fused path: entire decrypt in a single #[target_feature] scope.
    #[cfg(target_arch = "aarch64")]
    if matches!(
      self.backend,
      AeadBackend::Aarch64AesPmull | AeadBackend::Aarch64Sve2AesPmull
    ) {
      // SAFETY: Direct AArch64 AES-256 GCM-SIV KDF because:
      // 1. Backend resolution selected an AArch64 AES+PMULL backend.
      // 2. The selected backend constructs `self.master_ek` with AES-CE round keys.
      // 3. `nonce.as_bytes()` is exactly the 96-bit GCM-SIV nonce.
      let (mut auth_key, mut enc_key) =
        unsafe { aes::aarch64_gcmsiv_derive_keys_inline(&self.master_ek, nonce.as_bytes()) };
      // SAFETY: AES-CE availability verified during backend resolution.
      return unsafe { decrypt_fused_aarch64(&mut auth_key, &mut enc_key, nonce, aad, buffer, tag) }
        .map_err(OpenError::from);
    }

    // Fused path: POWER8 crypto.
    #[cfg(target_arch = "powerpc64")]
    if self.backend == AeadBackend::Power8Crypto {
      let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
      // SAFETY: POWER8 crypto availability verified during backend resolution.
      return unsafe { decrypt_fused_ppc(&mut auth_key, &mut enc_key, nonce, aad, buffer, tag) }
        .map_err(OpenError::from);
    }

    // Fused path: s390x z/Vector + MSA.
    #[cfg(target_arch = "s390x")]
    if self.backend == AeadBackend::S390xMsa {
      let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
      // SAFETY: z/Vector + MSA availability verified during backend resolution.
      return unsafe { decrypt_fused_s390x(&mut auth_key, &mut enc_key, nonce, aad, buffer, tag) }
        .map_err(OpenError::from);
    }

    #[cfg(target_arch = "riscv64")]
    {
      match self.backend {
        AeadBackend::Portable
        | AeadBackend::Riscv64VectorCrypto
        | AeadBackend::Riscv64ScalarCrypto
        | AeadBackend::Riscv64Vperm => {
          return decrypt_riscv(&self.master_ek, self.backend, nonce, aad, buffer, tag).map_err(OpenError::from);
        }
        _ => {}
      }
    }

    // Scalar path: decrypt then verify.
    let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
    let ek = aes::aes256_expand_key(&enc_key);
    let mut counter_block = tag.0;
    counter_block[15] |= 0x80;
    aes::aes256_ctr32_encrypt(&ek, &counter_block, buffer);

    let expected = compute_tag(&auth_key, &ek, nonce, aad, buffer);
    ct::zeroize(&mut auth_key);
    ct::zeroize(&mut enc_key);

    if !ct::constant_time_eq(&expected, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Err(OpenError::verification());
    }

    Ok(())
  }
}

// Tests

#[cfg(test)]
mod tests {
  use alloc::{vec, vec::Vec};

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
    let tag = cipher.encrypt_in_place(&nonce, &aad, &mut buf).unwrap();

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
