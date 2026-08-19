//! AES-128-GCM-SIV public AEAD surface (RFC 8452).

use core::fmt;

#[cfg(target_arch = "x86_64")]
use super::polyval::{accumulate_padded_x86, precompute_powers, precompute_powers_16};
#[cfg(any(
  target_arch = "aarch64",
  target_arch = "powerpc64",
  target_arch = "riscv64",
  target_arch = "s390x",
  target_arch = "x86_64",
))]
use super::targets::{AeadBackend, AeadPrimitive, select_backend};
use super::{AeadBufferError, Nonce96, OpenError, SealError, aes, polyval};
use crate::traits::{Aead, ct};

const KEY_SIZE: usize = 16;
const TAG_SIZE: usize = 16;
const NONCE_SIZE: usize = Nonce96::LENGTH;

/// Maximum plaintext and additional-data length: 2^36 bytes (RFC 8452 §6).
const MAX_INPUT_LEN: u64 = 1u64 << 36;

define_aead_key_type!(Aes128GcmSivKey, KEY_SIZE, "AES-128-GCM-SIV secret key (16 bytes).");

define_aead_tag_type!(
  Aes128GcmSivTag,
  TAG_SIZE,
  "AES-128-GCM-SIV authentication tag (16 bytes)."
);

/// AES-128-GCM-SIV AEAD (RFC 8452).
///
/// 128-bit-key counterpart of [`Aes256GcmSiv`](crate::Aes256GcmSiv).
/// Nonce-misuse resistant authenticated encryption: on nonce reuse only
/// the authentication guarantee degrades — confidentiality is preserved
/// up to a multi-message distinguishing bound.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "getrandom")]
/// # {
/// use rscrypto::{Aead, Aes128GcmSiv, Aes128GcmSivKey};
///
/// let key = Aes128GcmSivKey::from_bytes([0x42; 16]);
/// let cipher = Aes128GcmSiv::new(&key);
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
/// use rscrypto::{Aead, Aes128GcmSiv, Aes128GcmSivKey, aead::OpenError};
///
/// let key = Aes128GcmSivKey::from_bytes([0x42; 16]);
/// let cipher = Aes128GcmSiv::new(&key);
///
/// let mut sealed = [0u8; 5 + Aes128GcmSiv::TAG_SIZE];
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
/// On x86_64 (AES-NI), aarch64 (AES-CE), and s390x (CPACF), AES operations use
/// dedicated hardware instructions. On RISC-V without hardware AES extensions
/// (Zkne / Zvkned), encryption falls back to a fixed-work portable / fixslice
/// source implementation that avoids secret-indexed lookup tables. These
/// source and ISA properties are necessary, not sufficient: generated-code
/// timing claims are configuration- and release-evidence-bound; see `ct.toml`.
pub struct Aes128GcmSiv {
  master_ek: aes::Aes128EncKey,
  #[cfg(any(
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "riscv64",
    target_arch = "s390x",
    target_arch = "x86_64",
  ))]
  backend: AeadBackend,
}

impl fmt::Debug for Aes128GcmSiv {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Aes128GcmSiv").finish_non_exhaustive()
  }
}

impl Aes128GcmSiv {
  /// Key length in bytes.
  pub const KEY_SIZE: usize = KEY_SIZE;

  /// Nonce length in bytes.
  pub const NONCE_SIZE: usize = NONCE_SIZE;

  /// Tag length in bytes.
  pub const TAG_SIZE: usize = TAG_SIZE;

  /// Construct a new AES-128-GCM-SIV instance from `key`.
  #[inline]
  #[must_use]
  pub fn new(key: &Aes128GcmSivKey) -> Self {
    <Self as Aead>::new(key)
  }

  /// Rebuild a typed tag from raw tag bytes.
  #[inline]
  pub fn tag_from_slice(bytes: &[u8]) -> Result<Aes128GcmSivTag, AeadBufferError> {
    <Self as Aead>::tag_from_slice(bytes)
  }

  /// Decrypt `buffer` in place and verify the detached authentication tag.
  #[inline]
  pub fn decrypt_in_place(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Aes128GcmSivTag,
  ) -> Result<(), OpenError> {
    <Self as Aead>::decrypt_in_place(self, nonce, aad, buffer, tag)
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

// RFC 8452 construction internals (AES-128 variant)

/// Derive per-message authentication and encryption keys from the cached
/// master-key schedule and nonce (RFC 8452 §4, AES-128 variant).
///
/// Returns (auth_key [16 bytes], enc_key [16 bytes]).
///
/// AES-128 requires only **4** ECB blocks (vs 6 for AES-256): blocks 0..1
/// supply the auth key and blocks 2..3 supply the (16-byte) enc key.
/// All four counter blocks are encrypted in a single batch call, which
/// collapses to one KM invocation on s390x.
#[inline]
fn derive_keys(master_ek: &aes::Aes128EncKey, nonce: &Nonce96) -> ([u8; 16], [u8; 16]) {
  let nonce_bytes = nonce.as_bytes();

  // Build 4 counter blocks: counter (LE32) || nonce (96 bits).
  let mut blocks = [[0u8; 16]; 4];
  let mut i = 0u32;
  while i < 4 {
    blocks[i as usize][0..4].copy_from_slice(&i.to_le_bytes());
    blocks[i as usize][4..16].copy_from_slice(nonce_bytes);
    i = i.strict_add(1);
  }

  // Encrypt all 4 blocks (single KM call on s390x, per-block elsewhere).
  aes::aes128_encrypt_blocks_ecb(master_ek, &mut blocks);

  // Extract the first 8 bytes of each encrypted block.
  let mut auth_key = [0u8; 16];
  let mut enc_key = [0u8; 16];
  auth_key[0..8].copy_from_slice(&blocks[0][0..8]);
  auth_key[8..16].copy_from_slice(&blocks[1][0..8]);
  enc_key[0..8].copy_from_slice(&blocks[2][0..8]);
  enc_key[8..16].copy_from_slice(&blocks[3][0..8]);

  ct::zeroize(blocks.as_flattened_mut());
  (auth_key, enc_key)
}

/// Compute the POLYVAL-based authentication tag (RFC 8452 §5 steps 1-3).
#[inline]
fn compute_tag(
  auth_key: &[u8; 16],
  enc_ek: &aes::Aes128EncKey,
  nonce: &Nonce96,
  aad: &[u8],
  plaintext: &[u8],
) -> [u8; TAG_SIZE] {
  let mut pv = polyval::Polyval::new(auth_key);

  pv.update_padded(aad);
  pv.update_padded(plaintext);

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
  aes::aes128_encrypt_block(enc_ek, &mut s);

  s
}

#[cfg(feature = "diag")]
/// Derive the per-nonce authentication and encryption keys for diagnostic comparison.
#[must_use]
pub fn diag_aes128gcmsiv_derive_keys(cipher: &Aes128GcmSiv, nonce: &Nonce96) -> ([u8; 16], [u8; 16]) {
  derive_keys(&cipher.master_ek, nonce)
}

#[cfg(feature = "diag")]
/// Return the AES-128-GCM-SIV POLYVAL digest before nonce and AES tag finalization.
#[must_use]
pub fn diag_aes128gcmsiv_polyval_digest(auth_key: &[u8; 16], aad: &[u8], plaintext: &[u8]) -> [u8; 16] {
  let mut pv = polyval::Polyval::new(auth_key);
  pv.update_padded(aad);
  pv.update_padded(plaintext);
  let length_block = super::AeadByteLengths::from_usize(aad.len(), plaintext.len()).to_le_bits_block();
  pv.update_block(&length_block);
  pv.finalize()
}

#[cfg(feature = "diag")]
/// Encrypt one diagnostic tag block with a raw AES-128 key.
#[must_use]
pub fn diag_aes128gcmsiv_raw_tag_aes(enc_key: &[u8; 16], block: &[u8; 16]) -> [u8; 16] {
  let mut out = *block;
  #[cfg(target_arch = "s390x")]
  {
    // SAFETY: diagnostic s390x CT runs execute on the native MSA runner.
    unsafe { aes::s390x_encrypt_block_raw_128_inline(enc_key, &mut out) };
  }
  #[cfg(not(target_arch = "s390x"))]
  {
    let ek = aes::aes128_expand_key(enc_key);
    aes::aes128_encrypt_block(&ek, &mut out);
  }
  out
}

#[cfg(feature = "diag")]
/// Exercise AES-128 counter-mode encryption and fold the fixed diagnostic output to one block.
#[must_use]
pub fn diag_aes128gcmsiv_ctr32(enc_key: &[u8; 16], tag: &[u8; 16], plaintext: &[u8; 44]) -> [u8; 16] {
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
      unsafe { aes::s390x_encrypt_block_raw_128_inline(enc_key, &mut keystream) };

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
    let ek = aes::aes128_expand_key(enc_key);
    aes::aes128_ctr32_encrypt(&ek, &counter_block, &mut buffer);
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
  enc_ek: &aes::Aes128EncKey,
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
  aes::aes128_encrypt_block(enc_ek, &mut s);
  s
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn expand_key_riscv_for_backend(key: &[u8; 16], backend: AeadBackend) -> aes::Aes128EncKey {
  match backend {
    AeadBackend::Riscv64VectorCrypto => aes::aes128_expand_key_riscv_vector(key),
    AeadBackend::Riscv64ScalarCrypto => aes::aes128_expand_key_riscv_scalar(key),
    AeadBackend::Portable => aes::aes128_expand_key_riscv_ttable(key),
    _ => aes::aes128_expand_key_riscv_ttable(key),
  }
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn expand_message_key_riscv(enc_key: &[u8; 16], backend: AeadBackend) -> aes::Aes128EncKey {
  expand_key_riscv_for_backend(enc_key, backend)
}

#[cfg(any(
  target_arch = "aarch64",
  target_arch = "powerpc64",
  target_arch = "riscv64",
  target_arch = "s390x",
  target_arch = "x86_64",
))]
#[inline]
fn resolve_backend() -> AeadBackend {
  select_backend(
    AeadPrimitive::Aes128GcmSiv,
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
    AeadBackend::Portable => {
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
  enc_ek: &aes::Aes128EncKey,
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
  master_ek: &aes::Aes128EncKey,
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
  aes::aes128_ctr32_encrypt(&ek, &counter_block, buffer);
  ct::zeroize(&mut auth_key);
  ct::zeroize(&mut enc_key);
  tag_bytes
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn decrypt_riscv(
  master_ek: &aes::Aes128EncKey,
  backend: AeadBackend,
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  tag: &Aes128GcmSivTag,
) -> Result<(), crate::traits::VerificationError> {
  let (mut auth_key, mut enc_key) = derive_keys(master_ek, nonce);
  let ek = expand_message_key_riscv(&enc_key, backend);
  let mut counter_block = tag.0;
  counter_block[15] |= 0x80;
  aes::aes128_ctr32_encrypt(&ek, &counter_block, buffer);

  let expected = compute_tag_riscv(&auth_key, &ek, nonce, aad, buffer, backend);
  ct::zeroize(&mut auth_key);
  ct::zeroize(&mut enc_key);

  if !ct::fixed_eq(&expected, tag.as_bytes()).declassify() {
    ct::zeroize(buffer);
    return Err(crate::traits::VerificationError::new());
  }
  Ok(())
}

/// Compute the POLYVAL-based authentication tag using 4-block wide processing.
#[cfg(target_arch = "x86_64")]
#[inline]
fn compute_tag_wide(
  auth_key: &[u8; 16],
  enc_ek: &aes::Aes128EncKey,
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

  let length_block = super::AeadByteLengths::from_usize(aad.len(), plaintext.len()).to_le_bits_block();
  acc ^= u128::from_le_bytes(length_block);
  acc = polyval::clmul128_reduce(acc, h);

  let mut s = acc.to_le_bytes();
  let nonce_bytes = nonce.as_bytes();
  let mut j = 0usize;
  while j < 12 {
    s[j] ^= nonce_bytes[j];
    j = j.strict_add(1);
  }
  s[15] &= 0x7f;
  aes::aes128_encrypt_block(enc_ek, &mut s);
  s
}

// aarch64 fused encrypt/decrypt (single #[target_feature] scope)

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes,neon")]
/// Seal one message through the fused AArch64 AES-128-GCM-SIV path.
///
/// # Safety
///
/// The current CPU must support AArch64 AES, NEON, and PMULL. Callers must establish those
/// capabilities through validated backend selection before entering this function.
unsafe fn encrypt_fused_aarch64(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 16],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  // SAFETY: fused AArch64 AES-128-GCM-SIV encryption because:
  // 1. This function has `#[target_feature(enable = "aes,neon")]`.
  // 2. The caller verifies AES-CE availability before dispatching here.
  // 3. `auth_key`, `enc_key_bytes`, nonce, AAD, and buffer are initialized caller-owned inputs.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    let enc_ek = aes::aarch64_expand_key_128_inline(enc_key_bytes);
    ct::zeroize(enc_key_bytes);

    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

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

    let length_block = super::AeadByteLengths::from_usize(aad.len(), buffer.len()).to_le_bits_block();
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::aarch64_clmul128_reduce_inline(acc, h);

    let mut tag = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      tag[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    tag[15] &= 0x7f;
    aes::aarch64_encrypt_block_128_inline(&enc_ek, &mut tag);

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
      aes::aarch64_ctr32_le_xor_8blocks_128_inline(&enc_ek, &iv_suffix, ctr, &mut buffer[offset..end]);
      ctr = ctr.wrapping_add(8);
      offset = end;
    }
    while offset.strict_add(64) <= buffer.len() {
      let end = offset.strict_add(64);
      aes::aarch64_ctr32_le_xor_4blocks_128_inline(&enc_ek, &iv_suffix, ctr, &mut buffer[offset..end]);
      ctr = ctr.wrapping_add(4);
      offset = end;
    }
    while offset < buffer.len() {
      counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
      let mut keystream = counter_block;
      aes::aarch64_encrypt_block_128_inline(&enc_ek, &mut keystream);

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
/// Open one message through the fused AArch64 AES-128-GCM-SIV path.
///
/// # Safety
///
/// The current CPU must support AArch64 AES, NEON, and PMULL. Callers must establish those
/// capabilities through validated backend selection before entering this function.
unsafe fn decrypt_fused_aarch64(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 16],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  tag: &Aes128GcmSivTag,
) -> Result<(), crate::traits::VerificationError> {
  // SAFETY: fused AArch64 AES-128-GCM-SIV decryption because:
  // 1. This function has `#[target_feature(enable = "aes,neon")]`.
  // 2. The caller verifies AES-CE availability before dispatching here.
  // 3. `auth_key`, `enc_key_bytes`, nonce, AAD, buffer, and tag are initialized caller-owned inputs.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    let enc_ek = aes::aarch64_expand_key_128_inline(enc_key_bytes);
    ct::zeroize(enc_key_bytes);

    // SIV: decrypt then verify.
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
      aes::aarch64_ctr32_le_xor_8blocks_128_inline(&enc_ek, &iv_suffix, ctr, &mut buffer[offset..end]);
      ctr = ctr.wrapping_add(8);
      offset = end;
    }
    while offset.strict_add(64) <= buffer.len() {
      let end = offset.strict_add(64);
      aes::aarch64_ctr32_le_xor_4blocks_128_inline(&enc_ek, &iv_suffix, ctr, &mut buffer[offset..end]);
      ctr = ctr.wrapping_add(4);
      offset = end;
    }
    while offset < buffer.len() {
      counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
      let mut keystream = counter_block;
      aes::aarch64_encrypt_block_128_inline(&enc_ek, &mut keystream);

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

    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

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

    let length_block = super::AeadByteLengths::from_usize(aad.len(), buffer.len()).to_le_bits_block();
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::aarch64_clmul128_reduce_inline(acc, h);

    let mut expected = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      expected[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    expected[15] &= 0x7f;
    aes::aarch64_encrypt_block_128_inline(&enc_ek, &mut expected);

    if !ct::fixed_eq(&expected, tag.as_bytes()).declassify() {
      ct::zeroize(buffer);
      return Err(crate::traits::VerificationError::new());
    }
    Ok(())
  }
}

// powerpc64 fused encrypt/decrypt (single #[target_feature] scope)

/// Encrypt with the fused POWER8 AES-128-GCM-SIV backend.
///
/// # Safety
///
/// The executing CPU must support AltiVec, VSX, POWER8 vector, and POWER8 crypto.
#[cfg(target_arch = "powerpc64")]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
unsafe fn encrypt_fused_ppc(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 16],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  // SAFETY: fused POWER8 AES-128-GCM-SIV encryption because:
  // 1. This function has POWER8 crypto target features enabled.
  // 2. The caller verifies POWER8 crypto availability before dispatching here.
  // 3. `auth_key`, `enc_key_bytes`, nonce, AAD, and buffer are initialized caller-owned inputs.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    let enc_ek = aes::ppc_expand_key_128_inline(enc_key_bytes);
    ct::zeroize(enc_key_bytes);

    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

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

    let length_block = super::AeadByteLengths::from_usize(aad.len(), buffer.len()).to_le_bits_block();
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::ppc_clmul128_reduce_inline(acc, h);

    let mut tag = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      tag[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    tag[15] &= 0x7f;
    aes::ppc_encrypt_block_128_inline(&enc_ek, &mut tag);

    let mut counter_block = tag;
    counter_block[15] |= 0x80;
    let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
    offset = 0;
    while offset < buffer.len() {
      counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
      let mut keystream = counter_block;
      aes::ppc_encrypt_block_128_inline(&enc_ek, &mut keystream);

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

/// Decrypt and authenticate with the fused POWER8 AES-128-GCM-SIV backend.
///
/// # Safety
///
/// The executing CPU must support AltiVec, VSX, POWER8 vector, and POWER8 crypto.
#[cfg(target_arch = "powerpc64")]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
unsafe fn decrypt_fused_ppc(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 16],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  tag: &Aes128GcmSivTag,
) -> Result<(), crate::traits::VerificationError> {
  // SAFETY: fused POWER8 AES-128-GCM-SIV decryption because:
  // 1. This function has POWER8 crypto target features enabled.
  // 2. The caller verifies POWER8 crypto availability before dispatching here.
  // 3. `auth_key`, `enc_key_bytes`, nonce, AAD, buffer, and tag are initialized caller-owned inputs.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    let enc_ek = aes::ppc_expand_key_128_inline(enc_key_bytes);
    ct::zeroize(enc_key_bytes);

    let mut counter_block = tag.0;
    counter_block[15] |= 0x80;
    let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
    let mut offset = 0usize;
    while offset < buffer.len() {
      counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
      let mut keystream = counter_block;
      aes::ppc_encrypt_block_128_inline(&enc_ek, &mut keystream);

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

    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

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

    let length_block = super::AeadByteLengths::from_usize(aad.len(), buffer.len()).to_le_bits_block();
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::ppc_clmul128_reduce_inline(acc, h);

    let mut expected = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      expected[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    expected[15] &= 0x7f;
    aes::ppc_encrypt_block_128_inline(&enc_ek, &mut expected);

    if !ct::fixed_eq(&expected, tag.as_bytes()).declassify() {
      ct::zeroize(buffer);
      return Err(crate::traits::VerificationError::new());
    }
    Ok(())
  }
}

// s390x fused encrypt/decrypt

/// XOR an AES-128 counter stream into `buffer` using s390x CPACF.
///
/// # Safety
///
/// The executing CPU must support MSA AES instructions.
#[cfg(target_arch = "s390x")]
unsafe fn s390x_ctr32_le_xor_raw(enc_key_bytes: &[u8; 16], counter_block: &mut [u8; 16], buffer: &mut [u8]) {
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
    unsafe { aes::s390x_encrypt_blocks_raw_128_inline(enc_key_bytes, flat, block_count) };

    let processed = aes::xor_keystream_tail(buffer, offset, &keystream, block_count);
    offset = offset.strict_add(processed);
    ctr = ctr.wrapping_add(u32::from(block_count.to_le_bytes()[0]));
  }

  counter_block[0..4].copy_from_slice(&ctr.to_le_bytes());
}

/// Encrypt with the fused s390x AES-128-GCM-SIV backend.
///
/// # Safety
///
/// The executing CPU must support the vector facility and MSA AES instructions.
#[cfg(target_arch = "s390x")]
#[target_feature(enable = "vector")]
unsafe fn encrypt_fused_s390x(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 16],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  // SAFETY: fused s390x AES-128-GCM-SIV encryption because:
  // 1. This path only runs after z/Vector + MSA availability is verified.
  // 2. `auth_key` and `enc_key_bytes` are fixed-size initialized derived keys.
  // 3. Nonce, AAD, and buffer are initialized caller-owned inputs.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

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

    let length_block = super::AeadByteLengths::from_usize(aad.len(), buffer.len()).to_le_bits_block();
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::s390x_clmul128_reduce_inline(acc, h);

    let mut tag = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      tag[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    tag[15] &= 0x7f;
    aes::s390x_encrypt_block_raw_128_inline(enc_key_bytes, &mut tag);

    let mut counter_block = tag;
    counter_block[15] |= 0x80;
    s390x_ctr32_le_xor_raw(enc_key_bytes, &mut counter_block, buffer);

    ct::zeroize(enc_key_bytes);
    tag
  }
}

/// Decrypt and authenticate with the fused s390x AES-128-GCM-SIV backend.
///
/// # Safety
///
/// The executing CPU must support the vector facility and MSA AES instructions.
#[cfg(target_arch = "s390x")]
#[target_feature(enable = "vector")]
unsafe fn decrypt_fused_s390x(
  auth_key: &mut [u8; 16],
  enc_key_bytes: &mut [u8; 16],
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  tag: &Aes128GcmSivTag,
) -> Result<(), crate::traits::VerificationError> {
  // SAFETY: fused s390x AES-128-GCM-SIV decryption because:
  // 1. This path only runs after z/Vector + MSA availability is verified.
  // 2. `auth_key` and `enc_key_bytes` are fixed-size initialized derived keys.
  // 3. Nonce, AAD, buffer, and tag are initialized caller-owned inputs.
  unsafe {
    let nonce_bytes = nonce.as_bytes();

    let mut counter_block = tag.0;
    counter_block[15] |= 0x80;
    s390x_ctr32_le_xor_raw(enc_key_bytes, &mut counter_block, buffer);

    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);
    let mut acc: u128 = 0;

    let mut h_powers_rev = [0u128; 4];
    if aad.len() >= 64 || buffer.len() >= 64 {
      let powers = polyval::precompute_powers(h);
      h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    }

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

    let length_block = super::AeadByteLengths::from_usize(aad.len(), buffer.len()).to_le_bits_block();
    acc ^= u128::from_le_bytes(length_block);
    acc = polyval::s390x_clmul128_reduce_inline(acc, h);

    let mut expected = acc.to_le_bytes();
    let mut j = 0usize;
    while j < 12 {
      expected[j] ^= nonce_bytes[j];
      j = j.strict_add(1);
    }
    expected[15] &= 0x7f;
    aes::s390x_encrypt_block_raw_128_inline(enc_key_bytes, &mut expected);
    ct::zeroize(enc_key_bytes);

    if !ct::fixed_eq(&expected, tag.as_bytes()).declassify() {
      ct::zeroize(buffer);
      return Err(crate::traits::VerificationError::new());
    }
    Ok(())
  }
}

impl Aead for Aes128GcmSiv {
  const KEY_SIZE: usize = KEY_SIZE;
  const NONCE_SIZE: usize = NONCE_SIZE;
  const TAG_SIZE: usize = TAG_SIZE;

  type Key = Aes128GcmSivKey;
  type Nonce = Nonce96;
  type Tag = Aes128GcmSivTag;

  fn new(key: &Self::Key) -> Self {
    #[cfg(any(
      target_arch = "aarch64",
      target_arch = "powerpc64",
      target_arch = "riscv64",
      target_arch = "s390x",
      target_arch = "x86_64",
    ))]
    let backend = resolve_backend();

    Self {
      #[cfg(target_arch = "riscv64")]
      master_ek: expand_key_riscv_for_backend(key.as_bytes(), backend),
      #[cfg(not(target_arch = "riscv64"))]
      master_ek: aes::aes128_expand_key(key.as_bytes()),
      #[cfg(any(
        target_arch = "aarch64",
        target_arch = "powerpc64",
        target_arch = "riscv64",
        target_arch = "s390x",
        target_arch = "x86_64",
      ))]
      backend,
    }
  }

  fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
    if bytes.len() != TAG_SIZE {
      return Err(AeadBufferError::new());
    }
    let mut tag = [0u8; TAG_SIZE];
    tag.copy_from_slice(bytes);
    Ok(Aes128GcmSivTag::from_bytes(tag))
  }

  fn __encrypt_in_place_with_nonce(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    _token: crate::traits::aead::SealToken,
  ) -> Result<Self::Tag, SealError> {
    super::seal_bounded_length_as_u64(aad.len(), MAX_INPUT_LEN)?;
    super::seal_bounded_length_as_u64(buffer.len(), MAX_INPUT_LEN)?;
    super::seal_bit_lengths(aad.len(), buffer.len())?;

    // Wide path: VPCLMULQDQ POLYVAL + VAES-512 CTR when available.
    #[cfg(target_arch = "x86_64")]
    if self.backend == AeadBackend::X86VaesVpclmul {
      let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
      let ek = aes::aes128_expand_key(&enc_key);
      let tag_bytes = compute_tag_wide(&auth_key, &ek, nonce, aad, buffer);
      let mut counter_block = tag_bytes;
      counter_block[15] |= 0x80;
      // SAFETY: VAES availability verified during backend resolution.
      unsafe { aes::aes128_ctr32_encrypt_wide(&ek, &counter_block, buffer) };
      ct::zeroize(&mut auth_key);
      ct::zeroize(&mut enc_key);
      return Ok(Aes128GcmSivTag::from_bytes(tag_bytes));
    }

    // Fused path: entire encrypt in a single #[target_feature] scope.
    #[cfg(target_arch = "aarch64")]
    if matches!(
      self.backend,
      AeadBackend::Aarch64AesPmull | AeadBackend::Aarch64Sve2AesPmull
    ) {
      // SAFETY: Direct AArch64 AES-128 GCM-SIV KDF because:
      // 1. Backend resolution selected an AArch64 AES+PMULL backend.
      // 2. The selected backend constructs `self.master_ek` with AES-CE round keys.
      // 3. `nonce.as_bytes()` is exactly the 96-bit GCM-SIV nonce.
      let (mut auth_key, mut enc_key) =
        unsafe { aes::aarch64_gcmsiv_derive_keys_128_inline(&self.master_ek, nonce.as_bytes()) };
      // SAFETY: AES-CE availability verified during backend resolution.
      let tag_bytes = unsafe { encrypt_fused_aarch64(&mut auth_key, &mut enc_key, nonce, aad, buffer) };
      return Ok(Aes128GcmSivTag::from_bytes(tag_bytes));
    }

    // Fused path: POWER8 crypto.
    #[cfg(target_arch = "powerpc64")]
    if self.backend == AeadBackend::Power8Crypto {
      let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
      // SAFETY: POWER8 crypto availability verified during backend resolution.
      let tag_bytes = unsafe { encrypt_fused_ppc(&mut auth_key, &mut enc_key, nonce, aad, buffer) };
      return Ok(Aes128GcmSivTag::from_bytes(tag_bytes));
    }

    // Fused path: s390x z/Vector + MSA.
    #[cfg(target_arch = "s390x")]
    if self.backend == AeadBackend::S390xMsa {
      let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
      // SAFETY: z/Vector + MSA availability verified during backend resolution.
      let tag_bytes = unsafe { encrypt_fused_s390x(&mut auth_key, &mut enc_key, nonce, aad, buffer) };
      return Ok(Aes128GcmSivTag::from_bytes(tag_bytes));
    }

    #[cfg(target_arch = "riscv64")]
    {
      match self.backend {
        AeadBackend::Portable | AeadBackend::Riscv64VectorCrypto | AeadBackend::Riscv64ScalarCrypto => {
          let tag_bytes = encrypt_riscv(&self.master_ek, self.backend, nonce, aad, buffer);
          return Ok(Aes128GcmSivTag::from_bytes(tag_bytes));
        }
        _ => {}
      }
    }

    // Scalar path.
    let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
    let ek = aes::aes128_expand_key(&enc_key);
    let tag_bytes = compute_tag(&auth_key, &ek, nonce, aad, buffer);
    let mut counter_block = tag_bytes;
    counter_block[15] |= 0x80;
    aes::aes128_ctr32_encrypt(&ek, &counter_block, buffer);

    ct::zeroize(&mut auth_key);
    ct::zeroize(&mut enc_key);

    Ok(Aes128GcmSivTag::from_bytes(tag_bytes))
  }

  fn decrypt_in_place(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), OpenError> {
    super::open_bounded_length_as_u64(aad.len(), MAX_INPUT_LEN)?;
    super::open_bounded_length_as_u64(buffer.len(), MAX_INPUT_LEN)?;
    super::open_bit_lengths(aad.len(), buffer.len())?;

    // Wide path: VAES-512 CTR + VPCLMULQDQ POLYVAL when available.
    #[cfg(target_arch = "x86_64")]
    if self.backend == AeadBackend::X86VaesVpclmul {
      let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
      let ek = aes::aes128_expand_key(&enc_key);
      // Decrypt first (SIV pattern).
      let mut counter_block = tag.0;
      counter_block[15] |= 0x80;
      // SAFETY: VAES availability verified during backend resolution.
      unsafe { aes::aes128_ctr32_encrypt_wide(&ek, &counter_block, buffer) };

      // Verify tag over decrypted plaintext.
      let expected = compute_tag_wide(&auth_key, &ek, nonce, aad, buffer);
      ct::zeroize(&mut auth_key);
      ct::zeroize(&mut enc_key);
      if !ct::fixed_eq(&expected, tag.as_bytes()).declassify() {
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
      // SAFETY: Direct AArch64 AES-128 GCM-SIV KDF because:
      // 1. Backend resolution selected an AArch64 AES+PMULL backend.
      // 2. The selected backend constructs `self.master_ek` with AES-CE round keys.
      // 3. `nonce.as_bytes()` is exactly the 96-bit GCM-SIV nonce.
      let (mut auth_key, mut enc_key) =
        unsafe { aes::aarch64_gcmsiv_derive_keys_128_inline(&self.master_ek, nonce.as_bytes()) };
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
        AeadBackend::Portable | AeadBackend::Riscv64VectorCrypto | AeadBackend::Riscv64ScalarCrypto => {
          return decrypt_riscv(&self.master_ek, self.backend, nonce, aad, buffer, tag).map_err(OpenError::from);
        }
        _ => {}
      }
    }

    // Scalar path: decrypt then verify.
    let (mut auth_key, mut enc_key) = derive_keys(&self.master_ek, nonce);
    let ek = aes::aes128_expand_key(&enc_key);
    let mut counter_block = tag.0;
    counter_block[15] |= 0x80;
    aes::aes128_ctr32_encrypt(&ek, &counter_block, buffer);

    let expected = compute_tag(&auth_key, &ek, nonce, aad, buffer);
    ct::zeroize(&mut auth_key);
    ct::zeroize(&mut enc_key);

    if !ct::fixed_eq(&expected, tag.as_bytes()).declassify() {
      ct::zeroize(buffer);
      return Err(OpenError::verification());
    }

    Ok(())
  }
}

// Tests (RFC 8452 Appendix C.1 vectors)

#[cfg(test)]
mod tests {
  use alloc::vec;

  use super::*;
  use crate::aead::{
    expert::AeadWithNonce,
    test_vectors::{hex_vec, hex12, hex16},
  };

  /// RFC 8452 Appendix C.1, test 1: empty plaintext, empty AAD.
  #[test]
  fn aes128gcmsiv_empty() {
    let key = Aes128GcmSivKey::from_bytes(hex16("01000000000000000000000000000000"));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let expected_ct_tag = hex_vec("dc20e2d83f25705bb49e439eca56de25");

    let cipher = Aes128GcmSiv::new(&key);
    let mut out = vec![0u8; expected_ct_tag.len()];
    cipher
      .encrypt(&nonce, &[], &[], &mut out)
      .expect("RFC 8452 empty AES-128-GCM-SIV encryption must succeed");
    assert_eq!(out, expected_ct_tag);

    let mut pt_out = vec![0u8; 0];
    cipher
      .decrypt(&nonce, &[], &expected_ct_tag, &mut pt_out)
      .expect("RFC 8452 empty AES-128-GCM-SIV decryption must succeed");
    assert!(pt_out.is_empty());
  }

  /// RFC 8452 Appendix C.1, test 2: 8-byte plaintext, empty AAD.
  #[test]
  fn aes128gcmsiv_pt8_no_aad() {
    let key = Aes128GcmSivKey::from_bytes(hex16("01000000000000000000000000000000"));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let plaintext = hex_vec("0100000000000000");
    let expected_ct_tag = hex_vec("b5d839330ac7b786578782fff6013b815b287c22493a364c");

    let cipher = Aes128GcmSiv::new(&key);
    let mut out = vec![0u8; plaintext.len().strict_add(TAG_SIZE)];
    cipher
      .encrypt(&nonce, &[], &plaintext, &mut out)
      .expect("RFC 8452 AES-128-GCM-SIV encryption must succeed");
    assert_eq!(out, expected_ct_tag);

    let mut pt_out = vec![0u8; plaintext.len()];
    cipher
      .decrypt(&nonce, &[], &expected_ct_tag, &mut pt_out)
      .expect("RFC 8452 AES-128-GCM-SIV decryption must succeed");
    assert_eq!(pt_out, plaintext);
  }

  /// RFC 8452 Appendix C.1, test 8: AAD=01, plaintext=0200000000000000.
  #[test]
  fn aes128gcmsiv_aad_and_plaintext() {
    let key = Aes128GcmSivKey::from_bytes(hex16("01000000000000000000000000000000"));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let aad = hex_vec("01");
    let plaintext = hex_vec("0200000000000000");
    let expected_ct_tag = hex_vec("1e6daba35669f4273b0a1a2560969cdf790d99759abd1508");

    let cipher = Aes128GcmSiv::new(&key);

    let mut out = vec![0u8; plaintext.len().strict_add(TAG_SIZE)];
    cipher
      .encrypt(&nonce, &aad, &plaintext, &mut out)
      .expect("RFC 8452 AES-128-GCM-SIV encryption with AAD must succeed");
    assert_eq!(out, expected_ct_tag);

    let mut pt_out = vec![0u8; plaintext.len()];
    cipher
      .decrypt(&nonce, &aad, &expected_ct_tag, &mut pt_out)
      .expect("RFC 8452 AES-128-GCM-SIV decryption with AAD must succeed");
    assert_eq!(pt_out, plaintext);
  }

  /// RFC 8452 Appendix C.1, longer multi-block test (PT=4 blocks, AAD=01).
  #[test]
  fn aes128gcmsiv_multi_block_with_aad() {
    let key = Aes128GcmSivKey::from_bytes(hex16("01000000000000000000000000000000"));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let aad = hex_vec("01");
    let plaintext = hex_vec(
      "0200000000000000000000000000000003000000000000000000000000000000040000000000000000000000000000000500000000000000000000000000000000",
    );
    let cipher = Aes128GcmSiv::new(&key);
    let mut out = vec![0u8; plaintext.len().strict_add(TAG_SIZE)];
    cipher
      .encrypt(&nonce, &aad, &plaintext, &mut out)
      .expect("multi-block AES-128-GCM-SIV encryption must succeed");

    let mut pt_out = vec![0u8; plaintext.len()];
    cipher
      .decrypt(&nonce, &aad, &out, &mut pt_out)
      .expect("multi-block AES-128-GCM-SIV decryption must succeed");
    assert_eq!(pt_out, plaintext);
  }

  /// Decryption with wrong tag should fail.
  #[test]
  fn aes128gcmsiv_bad_tag() {
    let key = Aes128GcmSivKey::from_bytes(hex16("01000000000000000000000000000000"));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let mut bad_ct_tag = hex_vec("dc20e2d83f25705bb49e439eca56de25");
    bad_ct_tag[0] ^= 1;

    let cipher = Aes128GcmSiv::new(&key);
    let mut pt_out = vec![0u8; 0];
    assert_eq!(
      cipher.decrypt(&nonce, &[], &bad_ct_tag, &mut pt_out),
      Err(OpenError::verification())
    );
  }

  /// Decryption with wrong AAD should fail.
  #[test]
  fn aes128gcmsiv_wrong_aad_rejected() {
    let key = Aes128GcmSivKey::from_bytes(hex16("01000000000000000000000000000000"));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let ct_tag = hex_vec("1e6daba35669f4273b0a1a2560969cdf790d99759abd1508");

    let cipher = Aes128GcmSiv::new(&key);
    let mut pt_out = vec![0u8; 8];
    assert_eq!(
      cipher.decrypt(&nonce, &[0x02], &ct_tag, &mut pt_out),
      Err(OpenError::verification())
    );
  }

  /// Decryption with wrong nonce should fail.
  #[test]
  fn aes128gcmsiv_wrong_nonce_rejected() {
    let key = Aes128GcmSivKey::from_bytes(hex16("01000000000000000000000000000000"));
    let aad = hex_vec("01");
    let ct_tag = hex_vec("1e6daba35669f4273b0a1a2560969cdf790d99759abd1508");

    let cipher = Aes128GcmSiv::new(&key);
    let mut pt_out = vec![0u8; 8];
    let wrong_nonce = Nonce96::from_bytes(hex12("040000000000000000000000"));
    assert_eq!(
      cipher.decrypt(&wrong_nonce, &aad, &ct_tag, &mut pt_out),
      Err(OpenError::verification())
    );
  }

  /// Ciphertext tampering should fail verification.
  #[test]
  fn aes128gcmsiv_ciphertext_tampering_rejected() {
    let key = Aes128GcmSivKey::from_bytes(hex16("01000000000000000000000000000000"));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let aad = hex_vec("01");
    let plaintext = hex_vec("0200000000000000");
    let mut ct_tag = hex_vec("1e6daba35669f4273b0a1a2560969cdf790d99759abd1508");

    ct_tag[0] ^= 1;

    let cipher = Aes128GcmSiv::new(&key);
    let mut pt_out = vec![0u8; plaintext.len()];
    assert_eq!(
      cipher.decrypt(&nonce, &aad, &ct_tag, &mut pt_out),
      Err(OpenError::verification())
    );
  }

  /// On authentication failure, the output buffer must be zeroed.
  #[test]
  fn aes128gcmsiv_buffer_zeroed_on_auth_failure() {
    let key = Aes128GcmSivKey::from_bytes(hex16("01000000000000000000000000000000"));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let aad = hex_vec("01");
    let plaintext = hex_vec("0200000000000000");

    let cipher = Aes128GcmSiv::new(&key);
    let mut out = vec![0u8; plaintext.len().strict_add(TAG_SIZE)];
    cipher
      .encrypt(&nonce, &aad, &plaintext, &mut out)
      .expect("AES-128-GCM-SIV test setup encryption must succeed");

    let last = out.len().strict_sub(1);
    out[last] ^= 0xff;

    let mut pt_out = vec![0xffu8; plaintext.len()];
    assert_eq!(
      cipher.decrypt(&nonce, &aad, &out, &mut pt_out),
      Err(OpenError::verification())
    );
    assert!(pt_out.iter().all(|&b| b == 0), "buffer not zeroed on auth failure");
  }

  /// Detached encrypt/decrypt round-trip.
  #[test]
  fn aes128gcmsiv_detached_round_trip() {
    let key = Aes128GcmSivKey::from_bytes(hex16("01000000000000000000000000000000"));
    let nonce = Nonce96::from_bytes(hex12("030000000000000000000000"));
    let aad = hex_vec("01");
    let plaintext = hex_vec("0200000000000000");

    let cipher = Aes128GcmSiv::new(&key);

    let mut buf = plaintext.clone();
    let tag = cipher
      .encrypt_in_place(&nonce, &aad, &mut buf)
      .expect("AES-128-GCM-SIV detached encryption must succeed");

    assert_ne!(buf, plaintext);

    cipher
      .decrypt_in_place(&nonce, &aad, &mut buf, &tag)
      .expect("AES-128-GCM-SIV detached decryption must succeed");
    assert_eq!(buf, plaintext);
  }

  /// `tag_from_slice` rejects wrong-length input.
  #[test]
  fn aes128gcmsiv_tag_from_slice_rejects_bad_length() {
    assert_eq!(
      Aes128GcmSiv::tag_from_slice(&[0u8; 15]).expect_err("short AES-128-GCM-SIV tag must be rejected"),
      AeadBufferError::new()
    );
    assert_eq!(
      Aes128GcmSiv::tag_from_slice(&[0u8; 17]).expect_err("long AES-128-GCM-SIV tag must be rejected"),
      AeadBufferError::new()
    );
    assert_eq!(
      Aes128GcmSiv::tag_from_slice(&[]).expect_err("empty AES-128-GCM-SIV tag must be rejected"),
      AeadBufferError::new()
    );
    let tag = Aes128GcmSiv::tag_from_slice(&[0u8; 16]).expect("16-byte AES-128-GCM-SIV tag must be accepted");
    assert_eq!(tag.as_bytes(), &[0u8; 16]);
  }

  #[test]
  #[cfg(target_pointer_width = "64")]
  fn aes128gcmsiv_input_limit_matches_rfc8452() {
    for len in [MAX_INPUT_LEN.strict_sub(1), MAX_INPUT_LEN] {
      let platform_len = usize::try_from(len).expect("RFC 8452 input limit fits 64-bit usize");
      assert_eq!(
        super::super::try_bounded_length_as_u64(platform_len, MAX_INPUT_LEN),
        Ok(len)
      );
    }
    let too_large = usize::try_from(MAX_INPUT_LEN.strict_add(1)).expect("RFC 8452 input limit fits 64-bit usize");
    assert_eq!(
      super::super::try_bounded_length_as_u64(too_large, MAX_INPUT_LEN),
      Err(super::super::LengthOverflow)
    );
  }
}
