//! AES-GCM-SIV public AEAD surface (RFC 8452).
//!
//! One generic core, [`GcmSiv`], holds the mode for both key sizes. [`Aes128GcmSiv`] and
//! [`Aes256GcmSiv`] are concrete wrappers over it, so each key size compiles its own code.

use core::fmt;

#[cfg(any(
  target_arch = "aarch64",
  all(target_arch = "powerpc64", not(feature = "portable-only")),
  all(target_arch = "riscv64", not(feature = "portable-only")),
  all(target_arch = "s390x", not(feature = "portable-only")),
  all(target_arch = "x86_64", target_feature = "sse2")
))]
use super::targets::{AeadBackend, AeadPrimitive, select_backend};
use super::{
  AeadBufferError, Nonce96, OpenError, SealError,
  aes::{self, AesKey},
  polyval,
};
use crate::traits::{Aead, ct};

const TAG_SIZE: usize = 16;
const NONCE_SIZE: usize = Nonce96::LENGTH;

/// Maximum plaintext and additional-data length: 2^36 bytes (RFC 8452 §6).
const MAX_INPUT_LEN: u64 = 1u64 << 36;

/// The AES-GCM-SIV state of one key: its master AES schedule and the selected backend.
struct GcmSiv<K> {
  master_ek: K,
  #[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "powerpc64", not(feature = "portable-only")),
    all(target_arch = "riscv64", not(feature = "portable-only")),
    all(target_arch = "s390x", not(feature = "portable-only")),
    all(target_arch = "x86_64", target_feature = "sse2")
  ))]
  backend: AeadBackend,
}

impl<K: AesKey> GcmSiv<K> {
  // The wrapper's `Aead::new` holds this body, as the per-key-size constructor did.
  #[inline(always)]
  fn new(
    key: &K::Bytes,
    #[cfg(any(
      target_arch = "aarch64",
      all(target_arch = "powerpc64", not(feature = "portable-only")),
      all(target_arch = "riscv64", not(feature = "portable-only")),
      all(target_arch = "s390x", not(feature = "portable-only")),
      all(target_arch = "x86_64", target_feature = "sse2")
    ))]
    primitive: AeadPrimitive,
  ) -> Self {
    #[cfg(any(
      target_arch = "aarch64",
      all(target_arch = "powerpc64", not(feature = "portable-only")),
      all(target_arch = "riscv64", not(feature = "portable-only")),
      all(target_arch = "s390x", not(feature = "portable-only")),
      all(target_arch = "x86_64", target_feature = "sse2")
    ))]
    let backend = select_backend(primitive, crate::platform::arch(), crate::platform::caps());

    Self {
      #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
      master_ek: expand_riscv(key, backend),
      #[cfg(not(all(target_arch = "riscv64", not(feature = "portable-only"))))]
      master_ek: K::expand(key),
      #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "powerpc64", not(feature = "portable-only")),
        all(target_arch = "riscv64", not(feature = "portable-only")),
        all(target_arch = "s390x", not(feature = "portable-only")),
        all(target_arch = "x86_64", target_feature = "sse2")
      ))]
      backend,
    }
  }

  // The wrapper's `Aead` method holds this body, as the per-key-size method did.
  #[inline(always)]
  fn seal(&self, nonce: &Nonce96, aad: &[u8], buffer: &mut [u8]) -> Result<[u8; TAG_SIZE], SealError> {
    super::seal_bounded_length_as_u64(aad.len(), MAX_INPUT_LEN)?;
    super::seal_bounded_length_as_u64(buffer.len(), MAX_INPUT_LEN)?;
    super::seal_bit_lengths(aad.len(), buffer.len())?;

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if self.backend == AeadBackend::X86VaesVpclmul {
      // SAFETY: backend resolution selected this variant only after CPUID and OS-state checks
      // confirmed AES-NI, VAES, PCLMULQDQ, VPCLMULQDQ, and the required AVX-512 features.
      if let Some(tag) = unsafe { crypt_x86::<K, false>(&self.master_ek, nonce, aad, buffer, &[0; TAG_SIZE]) } {
        return Ok(tag);
      }
    }

    // Fused path: entire encrypt in a single #[target_feature] scope.
    #[cfg(target_arch = "aarch64")]
    if matches!(
      self.backend,
      AeadBackend::Aarch64AesPmull | AeadBackend::Aarch64Sve2AesPmull
    ) {
      // SAFETY: Direct AArch64 GCM-SIV KDF because:
      // 1. Backend resolution selected an AArch64 AES+PMULL backend.
      // 2. The method falls back to the portable KDF when `master_ek` is not an AES-CE schedule.
      // 3. `nonce.as_bytes()` is exactly the 96-bit GCM-SIV nonce.
      let (mut auth_key, mut enc_key) = unsafe { self.master_ek.aarch64_gcmsiv_derive_keys(nonce.as_bytes()) };
      // SAFETY: AES-CE availability verified during backend resolution.
      return Ok(unsafe { crypt_aarch64::<K, false>(&mut auth_key, &mut enc_key, nonce, aad, buffer, &[0; TAG_SIZE]) });
    }

    // Fused path: POWER8 crypto.
    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    if self.backend == AeadBackend::Power8Crypto {
      let (mut auth_key, mut enc_key) = self.master_ek.gcmsiv_derive_keys(nonce.as_bytes());
      // SAFETY: POWER8 crypto availability verified during backend resolution.
      return Ok(unsafe { crypt_ppc::<K, false>(&mut auth_key, &mut enc_key, nonce, aad, buffer, &[0; TAG_SIZE]) });
    }

    // Fused path: s390x z/Vector + MSA.
    #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
    if self.backend == AeadBackend::S390xMsa {
      let (mut auth_key, mut enc_key) = self.master_ek.gcmsiv_derive_keys(nonce.as_bytes());
      // SAFETY: z/Vector + MSA availability verified during backend resolution.
      return Ok(unsafe { crypt_s390x::<K, false>(&mut auth_key, &mut enc_key, nonce, aad, buffer, &[0; TAG_SIZE]) });
    }

    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    if matches!(
      self.backend,
      AeadBackend::Portable | AeadBackend::Riscv64VectorCrypto | AeadBackend::Riscv64ScalarCrypto
    ) {
      return Ok(seal_riscv(&self.master_ek, self.backend, nonce, aad, buffer));
    }

    // Scalar path.
    let (mut auth_key, mut enc_key) = self.master_ek.gcmsiv_derive_keys(nonce.as_bytes());
    let ek = K::expand(&enc_key);
    let tag = compute_tag(&auth_key, &ek, nonce, aad, buffer);
    ek.ctr32_encrypt(&counter_block(&tag), buffer);

    ct::zeroize(&mut auth_key);
    ct::zeroize(enc_key.as_mut());

    Ok(tag)
  }

  // The wrapper's `Aead` method holds this body, as the per-key-size method did.
  #[inline(always)]
  fn open(&self, nonce: &Nonce96, aad: &[u8], buffer: &mut [u8], tag: &[u8; TAG_SIZE]) -> Result<(), OpenError> {
    super::open_bounded_length_as_u64(aad.len(), MAX_INPUT_LEN)?;
    super::open_bounded_length_as_u64(buffer.len(), MAX_INPUT_LEN)?;
    super::open_bit_lengths(aad.len(), buffer.len())?;

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if self.backend == AeadBackend::X86VaesVpclmul {
      // SAFETY: backend resolution selected this variant only after CPUID and OS-state checks
      // confirmed AES-NI, VAES, PCLMULQDQ, VPCLMULQDQ, and the required AVX-512 features.
      if let Some(expected) = unsafe { crypt_x86::<K, true>(&self.master_ek, nonce, aad, buffer, tag) } {
        return verify(&expected, tag, buffer);
      }
    }

    // Fused path: entire decrypt in a single #[target_feature] scope.
    #[cfg(target_arch = "aarch64")]
    if matches!(
      self.backend,
      AeadBackend::Aarch64AesPmull | AeadBackend::Aarch64Sve2AesPmull
    ) {
      // SAFETY: Direct AArch64 GCM-SIV KDF because:
      // 1. Backend resolution selected an AArch64 AES+PMULL backend.
      // 2. The method falls back to the portable KDF when `master_ek` is not an AES-CE schedule.
      // 3. `nonce.as_bytes()` is exactly the 96-bit GCM-SIV nonce.
      let (mut auth_key, mut enc_key) = unsafe { self.master_ek.aarch64_gcmsiv_derive_keys(nonce.as_bytes()) };
      // SAFETY: AES-CE availability verified during backend resolution.
      let expected = unsafe { crypt_aarch64::<K, true>(&mut auth_key, &mut enc_key, nonce, aad, buffer, tag) };
      return verify(&expected, tag, buffer);
    }

    // Fused path: POWER8 crypto.
    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    if self.backend == AeadBackend::Power8Crypto {
      let (mut auth_key, mut enc_key) = self.master_ek.gcmsiv_derive_keys(nonce.as_bytes());
      // SAFETY: POWER8 crypto availability verified during backend resolution.
      let expected = unsafe { crypt_ppc::<K, true>(&mut auth_key, &mut enc_key, nonce, aad, buffer, tag) };
      return verify(&expected, tag, buffer);
    }

    // Fused path: s390x z/Vector + MSA.
    #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
    if self.backend == AeadBackend::S390xMsa {
      let (mut auth_key, mut enc_key) = self.master_ek.gcmsiv_derive_keys(nonce.as_bytes());
      // SAFETY: z/Vector + MSA availability verified during backend resolution.
      let expected = unsafe { crypt_s390x::<K, true>(&mut auth_key, &mut enc_key, nonce, aad, buffer, tag) };
      return verify(&expected, tag, buffer);
    }

    #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
    if matches!(
      self.backend,
      AeadBackend::Portable | AeadBackend::Riscv64VectorCrypto | AeadBackend::Riscv64ScalarCrypto
    ) {
      return open_riscv(&self.master_ek, self.backend, nonce, aad, buffer, tag);
    }

    // Scalar path: decrypt then verify.
    let (mut auth_key, mut enc_key) = self.master_ek.gcmsiv_derive_keys(nonce.as_bytes());
    let ek = K::expand(&enc_key);
    ek.ctr32_encrypt(&counter_block(tag), buffer);

    let expected = compute_tag(&auth_key, &ek, nonce, aad, buffer);
    ct::zeroize(&mut auth_key);
    ct::zeroize(enc_key.as_mut());

    verify(&expected, tag, buffer)
  }

  /// Derive the per-nonce keys through the production dispatch, for diagnostic comparison.
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  #[inline(always)]
  fn diag_derive_keys(&self, nonce: &Nonce96) -> ([u8; 16], K::Bytes) {
    #[cfg(target_arch = "aarch64")]
    if matches!(
      self.backend,
      AeadBackend::Aarch64AesPmull | AeadBackend::Aarch64Sve2AesPmull
    ) {
      // SAFETY: mirror the production AArch64 GCM-SIV key-derivation dispatch because:
      // 1. Backend resolution selected an AArch64 AES+PMULL backend.
      // 2. The method falls back to the portable KDF when `master_ek` is not an AES-CE schedule.
      // 3. `nonce.as_bytes()` is exactly the 96-bit GCM-SIV nonce.
      return unsafe { self.master_ek.aarch64_gcmsiv_derive_keys(nonce.as_bytes()) };
    }
    self.master_ek.gcmsiv_derive_keys(nonce.as_bytes())
  }
}

// RFC 8452 construction internals

/// The initial CTR block of a tag: the tag with the most significant bit of its last byte set
/// (RFC 8452 section 4).
#[inline(always)]
fn counter_block(tag: &[u8; TAG_SIZE]) -> [u8; 16] {
  let mut block = *tag;
  block[15] |= 0x80;
  block
}

/// Check an expected tag against the received one; on a mismatch, clear `buffer`.
#[inline(always)]
fn verify(expected: &[u8; TAG_SIZE], tag: &[u8; TAG_SIZE], buffer: &mut [u8]) -> Result<(), OpenError> {
  if !ct::fixed_eq(expected, tag).declassify() {
    ct::zeroize(buffer);
    return Err(OpenError::verification());
  }
  Ok(())
}

/// Turn the POLYVAL digest `acc` into the AES input of the tag: XOR the nonce into its first
/// 12 bytes and clear the most significant bit of its last byte (RFC 8452 section 4, step 5).
#[inline(always)]
fn tag_input_from_digest(acc: u128, nonce: &Nonce96) -> [u8; TAG_SIZE] {
  let mut s = acc.to_le_bytes();
  for (byte, nonce_byte) in s.iter_mut().zip(nonce.as_bytes()) {
    *byte ^= nonce_byte;
  }
  s[15] &= 0x7f;
  s
}

/// Absorb `data` into a POLYVAL accumulator, one 16-byte block per reduction; zero bytes pad the
/// final partial block.
#[cfg(any(
  target_arch = "aarch64",
  all(target_arch = "powerpc64", not(feature = "portable-only")),
  all(target_arch = "riscv64", not(feature = "portable-only")),
  all(target_arch = "s390x", not(feature = "portable-only"))
))]
#[inline(always)]
fn absorb_padded(mut acc: u128, h: u128, data: &[u8], reduce: &impl Fn(u128, u128) -> u128) -> u128 {
  let (blocks, tail) = data.as_chunks::<16>();
  for block in blocks {
    acc = reduce(acc ^ u128::from_le_bytes(*block), h);
  }
  if !tail.is_empty() {
    let mut block = [0u8; 16];
    block[..tail.len()].copy_from_slice(tail);
    acc = reduce(acc ^ u128::from_le_bytes(block), h);
  }
  acc
}

/// Absorb the length block and return the AES input of the tag.
#[cfg(any(
  target_arch = "aarch64",
  all(target_arch = "powerpc64", not(feature = "portable-only")),
  all(target_arch = "riscv64", not(feature = "portable-only")),
  all(target_arch = "s390x", not(feature = "portable-only"))
))]
#[inline(always)]
fn finish_tag_input(
  acc: u128,
  h: u128,
  nonce: &Nonce96,
  aad_len: usize,
  message_len: usize,
  reduce: &impl Fn(u128, u128) -> u128,
) -> [u8; TAG_SIZE] {
  let length_block = super::AeadByteLengths::from_usize(aad_len, message_len).to_le_bits_block();
  tag_input_from_digest(reduce(acc ^ u128::from_le_bytes(length_block), h), nonce)
}

/// The AES input of the tag over `aad` and `message` (RFC 8452 section 4, steps 3 to 5), with the
/// POLYVAL key `h`. Each whole 64-byte chunk takes one four-block `aggregate`; every other block
/// takes one `reduce`. The fused backends call it with their carryless-multiply kernels, inside
/// their own target-feature scope.
#[cfg(any(
  target_arch = "aarch64",
  all(target_arch = "powerpc64", not(feature = "portable-only")),
  all(target_arch = "s390x", not(feature = "portable-only"))
))]
#[inline(always)]
fn tag_input_wide(
  h: u128,
  nonce: &Nonce96,
  aad: &[u8],
  message: &[u8],
  reduce: impl Fn(u128, u128) -> u128,
  aggregate: impl Fn(u128, &[u128; 4], &[u128; 4]) -> u128,
) -> [u8; TAG_SIZE] {
  // Precompute H powers only when at least one 4-block chunk will run.
  let mut h_powers_rev = [0u128; 4];
  if aad.len() >= 64 || message.len() >= 64 {
    let powers = polyval::precompute_powers(h);
    h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
  }

  let mut acc = 0u128;
  for part in [aad, message] {
    let (chunks, rest) = part.as_chunks::<64>();
    for chunk in chunks {
      let blocks = chunk.as_chunks::<16>().0;
      acc = aggregate(
        acc,
        &h_powers_rev,
        &core::array::from_fn(|i| u128::from_le_bytes(blocks[i])),
      );
    }
    acc = absorb_padded(acc, h, rest, &reduce);
  }
  finish_tag_input(acc, h, nonce, aad.len(), message.len(), &reduce)
}

/// XOR the GCM-SIV keystream into `data`, one `encrypt` call per block, from counter `ctr`. The
/// counter is the little-endian word in bytes 0..4 of `counter_block` and wraps modulo 2^32.
#[cfg(any(
  target_arch = "aarch64",
  all(target_arch = "powerpc64", not(feature = "portable-only")),
  all(
    rscrypto_internal,
    feature = "diag",
    target_arch = "s390x",
    not(feature = "portable-only")
  )
))]
#[inline(always)]
fn ctr32_le_xor_blocks(
  counter_block: &[u8; 16],
  mut ctr: u32,
  data: &mut [u8],
  mut encrypt: impl FnMut(&mut [u8; 16]),
) {
  let mut block = *counter_block;
  let (chunks, tail) = data.as_chunks_mut::<16>();
  for chunk in chunks {
    block[..4].copy_from_slice(&ctr.to_le_bytes());
    let mut keystream = block;
    encrypt(&mut keystream);
    *chunk = (u128::from_ne_bytes(*chunk) ^ u128::from_ne_bytes(keystream)).to_ne_bytes();
    ctr = ctr.wrapping_add(1);
  }
  if !tail.is_empty() {
    block[..4].copy_from_slice(&ctr.to_le_bytes());
    let mut keystream = block;
    encrypt(&mut keystream);
    for (byte, stream) in tail.iter_mut().zip(keystream) {
      *byte ^= stream;
    }
  }
}

/// Compute the POLYVAL-based authentication tag (RFC 8452 §5 steps 1-3).
///
/// Accepts the pre-expanded encryption key to avoid redundant key expansion
/// (the same expanded key is reused for AES-CTR in the caller).
#[inline]
fn compute_tag<K: AesKey>(
  auth_key: &[u8; 16],
  enc_ek: &K,
  nonce: &Nonce96,
  aad: &[u8],
  plaintext: &[u8],
) -> [u8; TAG_SIZE] {
  let mut pv = polyval::Polyval::new(auth_key);

  // POLYVAL over: padded AAD || padded plaintext || length block
  pv.update_padded(aad);
  pv.update_padded(plaintext);

  // Length block: [aad_bits as u64 LE || plaintext_bits as u64 LE].
  let length_block = super::AeadByteLengths::from_usize(aad.len(), plaintext.len()).to_le_bits_block();
  pv.update_block(&length_block);

  let mut s = tag_input_from_digest(u128::from_le_bytes(pv.finalize()), nonce);
  enc_ek.encrypt_block(&mut s);
  s
}

#[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
#[derive(Clone, Copy)]
enum RiscvPolyvalBackend {
  Portable,
  Scalar,
  Vector,
}

#[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
#[inline]
fn reduce_riscv_portable(a: u128, b: u128) -> u128 {
  polyval::portable_clmul128_reduce_inline(a, b)
}

#[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
#[inline]
fn reduce_riscv_scalar(a: u128, b: u128) -> u128 {
  // SAFETY: caller only selects this reducer after runtime detection confirms
  // Zbc or Zbkc support.
  unsafe { polyval::riscv_scalar_clmul128_reduce_inline(a, b) }
}

#[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
#[inline]
fn reduce_riscv_vector(a: u128, b: u128) -> u128 {
  // SAFETY: caller only selects this reducer after runtime detection confirms
  // Zvbc support.
  unsafe { polyval::riscv_vector_clmul128_reduce_inline(a, b) }
}

#[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
#[inline]
fn compute_tag_riscv_with_reduce<K: AesKey>(
  auth_key: &[u8; 16],
  enc_ek: &K,
  nonce: &Nonce96,
  aad: &[u8],
  plaintext: &[u8],
  reduce: impl Fn(u128, u128) -> u128,
) -> [u8; TAG_SIZE] {
  let h = u128::from_le_bytes(*auth_key);
  let acc = absorb_padded(absorb_padded(0, h, aad, &reduce), h, plaintext, &reduce);
  let mut tag = finish_tag_input(acc, h, nonce, aad.len(), plaintext.len(), &reduce);
  enc_ek.encrypt_block(&mut tag);
  tag
}

/// Expand a key into the schedule of the selected RISC-V backend.
#[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
#[inline]
fn expand_riscv<K: AesKey>(key: &K::Bytes, backend: AeadBackend) -> K {
  match backend {
    AeadBackend::Riscv64VectorCrypto => K::expand_riscv_vector(key),
    AeadBackend::Riscv64ScalarCrypto => K::expand_riscv_scalar(key),
    _ => K::expand_riscv_fixslice(key),
  }
}

#[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
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

#[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
#[inline]
fn compute_tag_riscv<K: AesKey>(
  auth_key: &[u8; 16],
  enc_ek: &K,
  nonce: &Nonce96,
  aad: &[u8],
  plaintext: &[u8],
  backend: AeadBackend,
) -> [u8; TAG_SIZE] {
  match riscv_polyval_backend(backend) {
    RiscvPolyvalBackend::Portable => {
      match polyval::polyval_portable_wide(u128::from_le_bytes(*auth_key), aad, plaintext) {
        Some(mut acc) => {
          let mut tag = tag_input_from_digest(acc, nonce);
          enc_ek.encrypt_block(&mut tag);
          ct::zeroize_words(core::slice::from_mut(&mut acc));
          tag
        }
        None => compute_tag_riscv_with_reduce(auth_key, enc_ek, nonce, aad, plaintext, reduce_riscv_portable),
      }
    }
    RiscvPolyvalBackend::Scalar => {
      compute_tag_riscv_with_reduce(auth_key, enc_ek, nonce, aad, plaintext, reduce_riscv_scalar)
    }
    RiscvPolyvalBackend::Vector => {
      compute_tag_riscv_with_reduce(auth_key, enc_ek, nonce, aad, plaintext, reduce_riscv_vector)
    }
  }
}

#[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
#[inline]
fn seal_riscv<K: AesKey>(
  master_ek: &K,
  backend: AeadBackend,
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
) -> [u8; TAG_SIZE] {
  let (mut auth_key, mut enc_key) = master_ek.gcmsiv_derive_keys(nonce.as_bytes());
  let ek = expand_riscv::<K>(&enc_key, backend);
  let tag = compute_tag_riscv(&auth_key, &ek, nonce, aad, buffer, backend);
  ek.ctr32_encrypt(&counter_block(&tag), buffer);
  ct::zeroize(&mut auth_key);
  ct::zeroize(enc_key.as_mut());
  tag
}

#[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
#[inline]
fn open_riscv<K: AesKey>(
  master_ek: &K,
  backend: AeadBackend,
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  tag: &[u8; TAG_SIZE],
) -> Result<(), OpenError> {
  let (mut auth_key, mut enc_key) = master_ek.gcmsiv_derive_keys(nonce.as_bytes());
  let ek = expand_riscv::<K>(&enc_key, backend);
  ek.ctr32_encrypt(&counter_block(tag), buffer);

  let expected = compute_tag_riscv(&auth_key, &ek, nonce, aad, buffer, backend);
  ct::zeroize(&mut auth_key);
  ct::zeroize(enc_key.as_mut());

  verify(&expected, tag, buffer)
}

// Fused seal and open: one target-feature scope per backend
//
// Each function below seals (`OPEN = false`) or opens (`OPEN = true`) one message in one
// target-feature scope: the per-message key schedule, POLYVAL, the tag block, and AES-CTR. A
// boundary per block would spill AES round keys and POLYVAL state, which dominates at small
// message sizes. Closures defined in these functions inherit their target features. Sealing
// computes the tag and then encrypts with it. Opening decrypts with the `received` tag first, as
// SIV requires. Each function returns the tag over the plaintext, which the caller compares with
// `received` when it opens.

/// Seal or open one message through one x86 VAES/VPCLMULQDQ scope, with the per-nonce key
/// derivation inside it. The derived keys and their schedule are cleared before return.
///
/// Returns `None` without modifying `buffer` when the cached master schedule does not use the
/// AES-NI layout of the selected x86 backend.
///
/// # Safety
///
/// The current CPU and OS must support AES-NI, SSE2, SSSE3, AVX-512F, AVX-512VL, AVX-512BW,
/// AVX-512DQ, VAES, PCLMULQDQ, and VPCLMULQDQ. Callers must establish those capabilities through
/// validated backend selection before entering this function.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[target_feature(enable = "aes,sse2,ssse3,avx512f,avx512vl,avx512bw,avx512dq,vaes,pclmulqdq,vpclmulqdq")]
unsafe fn crypt_x86<K: AesKey, const OPEN: bool>(
  master_ek: &K,
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  received: &[u8; TAG_SIZE],
) -> Option<[u8; TAG_SIZE]> {
  // SAFETY: fused x86 AES-GCM-SIV because:
  // 1. This function enables every target feature required by the AES and POLYVAL helpers.
  // 2. The KDF helper returns `None` unless `master_ek` holds AES-NI round keys.
  // 3. The CTR helper accepts any buffer length.
  unsafe {
    let (mut auth_key, mut enc_key) = master_ek.x86_gcmsiv_derive_keys(nonce.as_bytes())?;
    let enc_ek = K::x86_expand(&enc_key);
    ct::zeroize(enc_key.as_mut());

    if OPEN {
      K::x86_ctr32_le_xor(&enc_ek, &counter_block(received), buffer);
    }
    let mut tag = polyval::x86_gcmsiv_tag_input(&mut auth_key, nonce.as_bytes(), aad, buffer);
    K::x86_encrypt_block(&enc_ek, &mut tag);
    if !OPEN {
      K::x86_ctr32_le_xor(&enc_ek, &counter_block(&tag), buffer);
    }
    Some(tag)
  }
}

/// Seal or open one message through the fused AArch64 AES-CE and PMULL scope.
///
/// # Safety
///
/// The current CPU must support AArch64 AES, NEON, and PMULL. Callers must establish those
/// capabilities through validated backend selection before entering this function.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes,neon")]
unsafe fn crypt_aarch64<K: AesKey, const OPEN: bool>(
  auth_key: &mut [u8; 16],
  enc_key: &mut K::Bytes,
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  received: &[u8; TAG_SIZE],
) -> [u8; TAG_SIZE] {
  // SAFETY: fused AArch64 AES-GCM-SIV because:
  // 1. This function has `#[target_feature(enable = "aes,neon")]`, and the caller establishes
  //    AES-CE and PMULL.
  // 2. Each CTR batch receives exactly the 128 or 64 bytes that it requires.
  // 3. `auth_key`, `enc_key`, nonce, AAD, buffer, and tag are initialized caller-owned inputs.
  unsafe {
    let enc_ek = K::aarch64_expand(enc_key);
    ct::zeroize(enc_key.as_mut());
    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);

    // Eight-block batches, then at most one four-block batch, then single blocks.
    let ctr = |tag: &[u8; TAG_SIZE], buffer: &mut [u8]| {
      let counter_block = counter_block(tag);
      let mut ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
      let iv_suffix: &[u8; 12] = counter_block
        .last_chunk()
        .expect("a 16-byte counter block ends in 12 bytes");
      let (chunks, rest) = buffer.as_chunks_mut::<128>();
      for chunk in chunks {
        K::aarch64_ctr32_le_xor_8blocks(&enc_ek, iv_suffix, ctr, chunk);
        ctr = ctr.wrapping_add(8);
      }
      let (chunks, rest) = rest.as_chunks_mut::<64>();
      for chunk in chunks {
        K::aarch64_ctr32_le_xor_4blocks(&enc_ek, iv_suffix, ctr, chunk);
        ctr = ctr.wrapping_add(4);
      }
      ctr32_le_xor_blocks(&counter_block, ctr, rest, |block| {
        K::aarch64_encrypt_block(&enc_ek, block)
      });
    };

    if OPEN {
      ctr(received, buffer);
    }
    let mut tag = tag_input_wide(
      h,
      nonce,
      aad,
      buffer,
      |a, b| polyval::aarch64_clmul128_reduce_inline(a, b),
      |acc, powers, blocks| polyval::aarch64_aggregate_4blocks_inline(acc, powers, blocks),
    );
    K::aarch64_encrypt_block(&enc_ek, &mut tag);
    if !OPEN {
      ctr(&tag, buffer);
    }
    tag
  }
}

/// Seal or open one message through the fused POWER8 crypto scope.
///
/// # Safety
///
/// The executing CPU must support AltiVec, VSX, POWER8 vector, and POWER8 crypto.
#[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
unsafe fn crypt_ppc<K: AesKey, const OPEN: bool>(
  auth_key: &mut [u8; 16],
  enc_key: &mut K::Bytes,
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  received: &[u8; TAG_SIZE],
) -> [u8; TAG_SIZE] {
  // SAFETY: fused POWER8 AES-GCM-SIV because:
  // 1. This function has POWER8 crypto target features enabled.
  // 2. The caller verifies POWER8 crypto availability before dispatching here.
  // 3. `auth_key`, `enc_key`, nonce, AAD, buffer, and tag are initialized caller-owned inputs.
  unsafe {
    let enc_ek = K::ppc_expand(enc_key);
    ct::zeroize(enc_key.as_mut());
    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);

    let ctr_blocks = |counter_block: &[u8; 16], buffer: &mut [u8]| {
      let ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
      ctr32_le_xor_blocks(counter_block, ctr, buffer, |block| K::ppc_encrypt_block(&enc_ek, block));
    };

    if OPEN {
      let counter_block = counter_block(received);
      // One-block chains overlap well below four blocks; longer messages need the explicit
      // four-block interleave (opening 1 MiB took three times as long as sealing without it).
      if buffer.len() >= 64 {
        K::ppc_ctr32_le_xor(&enc_ek, &counter_block, buffer);
      } else {
        ctr_blocks(&counter_block, buffer);
      }
    }
    let mut tag = tag_input_wide(
      h,
      nonce,
      aad,
      buffer,
      |a, b| polyval::ppc_clmul128_reduce_inline(a, b),
      |acc, powers, blocks| polyval::ppc_aggregate_4blocks_inline(acc, powers, blocks),
    );
    K::ppc_encrypt_block(&enc_ek, &mut tag);
    if !OPEN {
      ctr_blocks(&counter_block(&tag), buffer);
    }
    tag
  }
}

/// XOR an AES counter stream into `buffer` using s390x CPACF and the raw key.
///
/// # Safety
///
/// The executing CPU must support MSA AES instructions.
#[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
unsafe fn s390x_ctr32_le_xor_raw<K: AesKey>(enc_key: &K::Bytes, counter_block: &[u8; 16], buffer: &mut [u8]) {
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

    let flat = &mut keystream.as_flattened_mut()[..block_count.strict_mul(16)];
    // SAFETY: caller guarantees MSA; `flat` spans exactly `block_count <= 4` initialized blocks.
    unsafe { K::s390x_encrypt_blocks_raw(enc_key, flat, block_count) };

    let processed = aes::xor_keystream_tail(buffer, offset, &keystream, block_count);
    offset = offset.strict_add(processed);
    ctr = ctr.wrapping_add(u32::from(block_count.to_le_bytes()[0]));
  }
}

/// Seal or open one message through the fused s390x scope: z/Vector POLYVAL and KM with the raw
/// per-message key, which needs no key schedule.
///
/// # Safety
///
/// The executing CPU must support the vector facility and MSA AES instructions.
#[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
#[target_feature(enable = "vector")]
unsafe fn crypt_s390x<K: AesKey, const OPEN: bool>(
  auth_key: &mut [u8; 16],
  enc_key: &mut K::Bytes,
  nonce: &Nonce96,
  aad: &[u8],
  buffer: &mut [u8],
  received: &[u8; TAG_SIZE],
) -> [u8; TAG_SIZE] {
  // SAFETY: fused s390x AES-GCM-SIV because:
  // 1. This path only runs after z/Vector + MSA availability is verified.
  // 2. `auth_key` and `enc_key` are fixed-size initialized derived keys.
  // 3. Nonce, AAD, buffer, and tag are initialized caller-owned inputs.
  unsafe {
    let h = u128::from_le_bytes(*auth_key);
    ct::zeroize(auth_key);

    if OPEN {
      s390x_ctr32_le_xor_raw::<K>(enc_key, &counter_block(received), buffer);
    }
    let mut tag = tag_input_wide(
      h,
      nonce,
      aad,
      buffer,
      |a, b| polyval::s390x_clmul128_reduce_inline(a, b),
      |acc, powers, blocks| polyval::s390x_aggregate_4blocks_inline(acc, powers, blocks),
    );
    K::s390x_encrypt_block_raw(enc_key, &mut tag);
    if !OPEN {
      s390x_ctr32_le_xor_raw::<K>(enc_key, &counter_block(&tag), buffer);
    }
    ct::zeroize(enc_key.as_mut());
    tag
  }
}

/// Encrypt one diagnostic tag block with a raw key.
#[cfg(all(rscrypto_internal, feature = "diag"))]
#[inline(always)]
fn diag_raw_tag_aes<K: AesKey>(enc_key: &K::Bytes, block: &[u8; 16]) -> [u8; 16] {
  let mut out = *block;
  #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
  {
    // SAFETY: diagnostic s390x CT runs execute on the native MSA runner.
    unsafe { K::s390x_encrypt_block_raw(enc_key, &mut out) };
  }
  #[cfg(not(all(target_arch = "s390x", not(feature = "portable-only"))))]
  K::expand(enc_key).encrypt_block(&mut out);
  out
}

/// Exercise counter-mode encryption with a raw key and fold the fixed diagnostic output to one
/// block.
#[cfg(all(rscrypto_internal, feature = "diag"))]
#[inline(always)]
fn diag_ctr32<K: AesKey>(enc_key: &K::Bytes, tag: &[u8; 16], plaintext: &[u8; 44]) -> [u8; 16] {
  let counter_block = counter_block(tag);
  let mut buffer = *plaintext;
  #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
  {
    let ctr = u32::from_le_bytes([counter_block[0], counter_block[1], counter_block[2], counter_block[3]]);
    ctr32_le_xor_blocks(&counter_block, ctr, &mut buffer, |block| {
      // SAFETY: diagnostic s390x CT runs execute on the native MSA runner.
      unsafe { K::s390x_encrypt_block_raw(enc_key, block) }
    });
  }
  #[cfg(not(all(target_arch = "s390x", not(feature = "portable-only"))))]
  K::expand(enc_key).ctr32_encrypt(&counter_block, &mut buffer);
  super::diag_fold16(&buffer)
}

/// Define one public AES-GCM-SIV type over [`GcmSiv`], with its key and tag types.
macro_rules! define_aes_gcm_siv {
  (
    $(#[$summary:meta])*
    $name:ident($aes:ty), $variant:literal,
    key: $key:ident($key_len:literal, $key_doc:literal),
    tag: $tag:ident($tag_doc:literal)
  ) => {
    define_aead_key_type!($key, $key_len, $key_doc);

    define_aead_tag_type!($tag, TAG_SIZE, $tag_doc);

    $(#[$summary])*
    ///
    /// # Examples
    ///
    #[doc = concat!(
      "```\n",
      "# #[cfg(feature = \"getrandom\")]\n",
      "# {\n",
      "use rscrypto::{Aead, ", stringify!($name), ", ", stringify!($key), "};\n\n",
      "let key = ", stringify!($key), "::from_bytes([0x42; ", stringify!($key_len), "]);\n",
      "let cipher = ", stringify!($name), "::new(&key);\n\n",
      "let mut buf = *b\"hello\";\n",
      "let (nonce, tag) = cipher.seal_random_in_place(b\"\", &mut buf)?;\n",
      "cipher.decrypt_in_place(&nonce, b\"\", &mut buf, &tag)?;\n",
      "assert_eq!(&buf, b\"hello\");\n",
      "# }\n",
      "# Ok::<(), Box<dyn std::error::Error>>(())\n",
      "```",
    )]
    ///
    /// Tampering is reported as an opaque verification failure.
    ///
    #[doc = concat!(
      "```\n",
      "# #[cfg(feature = \"getrandom\")]\n",
      "# {\n",
      "use rscrypto::{Aead, ", stringify!($name), ", ", stringify!($key), ", aead::OpenError};\n\n",
      "let key = ", stringify!($key), "::from_bytes([0x42; ", stringify!($key_len), "]);\n",
      "let cipher = ", stringify!($name), "::new(&key);\n\n",
      "let mut sealed = [0u8; 5 + ", stringify!($name), "::TAG_SIZE];\n",
      "let nonce = cipher.seal_random(b\"\", b\"hello\", &mut sealed)?;\n",
      "sealed[0] ^= 1;\n\n",
      "let mut opened = [0u8; 5];\n",
      "assert_eq!(\n",
      "  cipher.decrypt(&nonce, b\"\", &sealed, &mut opened),\n",
      "  Err(OpenError::verification())\n",
      ");\n",
      "# }\n",
      "# Ok::<(), Box<dyn std::error::Error>>(())\n",
      "```",
    )]
    ///
    /// # Security
    ///
    /// On x86_64 (AES-NI), aarch64 (AES-CE), and s390x (CPACF), AES operations use
    /// dedicated hardware instructions. On RISC-V without hardware AES extensions
    /// (Zkne / Zvkned), encryption falls back to a fixed-work portable / fixslice
    /// source implementation that avoids secret-indexed lookup tables. These
    /// source and ISA properties are necessary, not sufficient: generated-code
    /// timing claims are configuration- and release-evidence-bound; see `ct.toml`.
    pub struct $name {
      inner: GcmSiv<$aes>,
    }

    impl fmt::Debug for $name {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(stringify!($name)).finish_non_exhaustive()
      }
    }

    impl $name {
      /// Key length in bytes.
      pub const KEY_SIZE: usize = $key_len;

      /// Nonce length in bytes.
      pub const NONCE_SIZE: usize = NONCE_SIZE;

      /// Tag length in bytes.
      pub const TAG_SIZE: usize = TAG_SIZE;

      #[doc = concat!("Construct a new ", $variant, " instance from `key`.")]
      #[inline]
      #[must_use]
      pub fn new(key: &$key) -> Self {
        <Self as Aead>::new(key)
      }

      /// Rebuild a typed tag from raw tag bytes.
      #[inline]
      pub fn tag_from_slice(bytes: &[u8]) -> Result<$tag, AeadBufferError> {
        <Self as Aead>::tag_from_slice(bytes)
      }

      /// Decrypt `buffer` in place and verify the detached authentication tag.
      #[inline]
      pub fn decrypt_in_place(&self, nonce: &Nonce96, aad: &[u8], buffer: &mut [u8], tag: &$tag) -> Result<(), OpenError> {
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

    impl crate::traits::RandomNonceAead for $name {}

    impl Aead for $name {
      const KEY_SIZE: usize = $key_len;
      const NONCE_SIZE: usize = NONCE_SIZE;
      const TAG_SIZE: usize = TAG_SIZE;

      type Key = $key;
      type Nonce = Nonce96;
      type Tag = $tag;

      fn new(key: &Self::Key) -> Self {
        #[cfg(any(
          target_arch = "aarch64",
          all(target_arch = "powerpc64", not(feature = "portable-only")),
          all(target_arch = "riscv64", not(feature = "portable-only")),
          all(target_arch = "s390x", not(feature = "portable-only")),
          all(target_arch = "x86_64", target_feature = "sse2")
        ))]
        let inner = GcmSiv::new(key.as_bytes(), AeadPrimitive::$name);
        #[cfg(not(any(
          target_arch = "aarch64",
          all(target_arch = "powerpc64", not(feature = "portable-only")),
          all(target_arch = "riscv64", not(feature = "portable-only")),
          all(target_arch = "s390x", not(feature = "portable-only")),
          all(target_arch = "x86_64", target_feature = "sse2")
        )))]
        let inner = GcmSiv::new(key.as_bytes());
        Self { inner }
      }

      fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
        let tag: [u8; TAG_SIZE] = bytes.try_into().map_err(|_| AeadBufferError::new())?;
        Ok($tag::from_bytes(tag))
      }

      fn __encrypt_in_place_with_nonce(
        &self,
        nonce: &Self::Nonce,
        aad: &[u8],
        buffer: &mut [u8],
        _token: crate::traits::aead::SealToken,
      ) -> Result<Self::Tag, SealError> {
        self.inner.seal(nonce, aad, buffer).map($tag::from_bytes)
      }

      fn decrypt_in_place(
        &self,
        nonce: &Self::Nonce,
        aad: &[u8],
        buffer: &mut [u8],
        tag: &Self::Tag,
      ) -> Result<(), OpenError> {
        self.inner.open(nonce, aad, buffer, tag.as_bytes())
      }
    }
  };
}

define_aes_gcm_siv! {
  /// AES-128-GCM-SIV AEAD (RFC 8452).
  ///
  /// 128-bit-key counterpart of [`Aes256GcmSiv`](crate::Aes256GcmSiv).
  /// Nonce-misuse resistant authenticated encryption: on nonce reuse only
  /// the authentication guarantee degrades — confidentiality is preserved
  /// up to a multi-message distinguishing bound.
  Aes128GcmSiv(aes::Aes128EncKey), "AES-128-GCM-SIV",
  key: Aes128GcmSivKey(16, "AES-128-GCM-SIV secret key (16 bytes)."),
  tag: Aes128GcmSivTag("AES-128-GCM-SIV authentication tag (16 bytes).")
}

define_aes_gcm_siv! {
  /// AES-256-GCM-SIV AEAD (RFC 8452).
  ///
  /// Nonce-misuse resistant authenticated encryption. On nonce reuse, only
  /// the authentication guarantee degrades — confidentiality is preserved
  /// up to a multi-message distinguishing bound.
  Aes256GcmSiv(aes::Aes256EncKey), "AES-256-GCM-SIV",
  key: Aes256GcmSivKey(32, "AES-256-GCM-SIV secret key (32 bytes)."),
  tag: Aes256GcmSivTag("AES-256-GCM-SIV authentication tag (16 bytes).")
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
/// Derive the AES-128-GCM-SIV per-nonce keys for diagnostic comparison.
#[must_use]
pub fn diag_aes128gcmsiv_derive_keys(cipher: &Aes128GcmSiv, nonce: &Nonce96) -> ([u8; 16], [u8; 16]) {
  cipher.inner.diag_derive_keys(nonce)
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
/// Derive the AES-256-GCM-SIV per-nonce keys for diagnostic comparison.
#[must_use]
pub fn diag_aes256gcmsiv_derive_keys(cipher: &Aes256GcmSiv, nonce: &Nonce96) -> ([u8; 16], [u8; 32]) {
  cipher.inner.diag_derive_keys(nonce)
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
/// Return the AES-GCM-SIV POLYVAL digest before nonce and AES tag finalization.
#[must_use]
pub fn diag_aes128gcmsiv_polyval_digest(auth_key: &[u8; 16], aad: &[u8], plaintext: &[u8]) -> [u8; 16] {
  let mut pv = polyval::Polyval::new(auth_key);
  pv.update_padded(aad);
  pv.update_padded(plaintext);
  let length_block = super::AeadByteLengths::from_usize(aad.len(), plaintext.len()).to_le_bits_block();
  pv.update_block(&length_block);
  pv.finalize()
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
/// Encrypt one diagnostic tag block with a raw AES-128 key.
#[must_use]
pub fn diag_aes128gcmsiv_raw_tag_aes(enc_key: &[u8; 16], block: &[u8; 16]) -> [u8; 16] {
  diag_raw_tag_aes::<aes::Aes128EncKey>(enc_key, block)
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
/// Encrypt one diagnostic tag block with a raw AES-256 key.
#[must_use]
pub fn diag_aes256gcmsiv_raw_tag_aes(enc_key: &[u8; 32], block: &[u8; 16]) -> [u8; 16] {
  diag_raw_tag_aes::<aes::Aes256EncKey>(enc_key, block)
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
/// Exercise AES-128 counter-mode encryption and fold the fixed diagnostic output to one block.
#[must_use]
pub fn diag_aes128gcmsiv_ctr32(enc_key: &[u8; 16], tag: &[u8; 16], plaintext: &[u8; 44]) -> [u8; 16] {
  diag_ctr32::<aes::Aes128EncKey>(enc_key, tag, plaintext)
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
/// Exercise AES-256 counter-mode encryption and fold the fixed diagnostic output to one block.
#[must_use]
pub fn diag_aes256gcmsiv_ctr32(enc_key: &[u8; 32], tag: &[u8; 16], plaintext: &[u8; 44]) -> [u8; 16] {
  diag_ctr32::<aes::Aes256EncKey>(enc_key, tag, plaintext)
}

#[cfg(test)]
mod tests {
  use alloc::vec;

  use super::*;
  use crate::aead::{
    expert::AeadWithNonce,
    test_vectors::{hex_vec, hex12, hex16, hex32},
  };

  #[track_caller]
  fn assert_vector<A: Aead<Nonce = Nonce96>>(cipher: &A, nonce: &str, aad: &str, plaintext: &str, sealed: &str) {
    let nonce = Nonce96::from_bytes(hex12(nonce));
    let aad = hex_vec(aad);
    let plaintext = hex_vec(plaintext);
    let sealed = hex_vec(sealed);
    let mut out = vec![0u8; sealed.len()];
    cipher
      .encrypt(&nonce, &aad, &plaintext, &mut out)
      .expect("RFC 8452 vector must seal");
    assert_eq!(out, sealed, "ciphertext and tag");
    let mut opened = vec![0u8; plaintext.len()];
    cipher
      .decrypt(&nonce, &aad, &sealed, &mut opened)
      .expect("RFC 8452 vector must open");
    assert_eq!(opened, plaintext, "plaintext");
  }

  const NONCE: &str = "030000000000000000000000";

  /// RFC 8452 Appendix C.1.
  #[test]
  fn aes128gcmsiv_rfc8452_vectors() {
    let cipher = Aes128GcmSiv::new(&Aes128GcmSivKey::from_bytes(hex16("01000000000000000000000000000000")));
    assert_vector(&cipher, NONCE, "", "", "dc20e2d83f25705bb49e439eca56de25");
    assert_vector(
      &cipher,
      NONCE,
      "",
      "0100000000000000",
      "b5d839330ac7b786578782fff6013b815b287c22493a364c",
    );
    assert_vector(
      &cipher,
      NONCE,
      "01",
      "0200000000000000",
      "1e6daba35669f4273b0a1a2560969cdf790d99759abd1508",
    );
  }

  /// RFC 8452 Appendix C.2.
  #[test]
  fn aes256gcmsiv_rfc8452_vectors() {
    let cipher = Aes256GcmSiv::new(&Aes256GcmSivKey::from_bytes(hex32(
      "0100000000000000000000000000000000000000000000000000000000000000",
    )));
    assert_vector(&cipher, NONCE, "", "", "07f5f4169bbf55a8400cd47ea6fd400f");
    assert_vector(
      &cipher,
      NONCE,
      "01",
      "0200000000000000",
      "1de22967237a813291213f267e3b452f02d01ae33e4ec854",
    );
    assert_vector(
      &cipher,
      NONCE,
      "010000000000000000000000000000000200",
      "0300000000000000000000000000000004000000",
      "43dd0163cdb48f9fe3212bf61b201976067f342bb879ad976d8242acc188ab59cabfe307",
    );
    let cipher = Aes256GcmSiv::new(&Aes256GcmSivKey::from_bytes(hex32(
      "e66021d5eb8e4f4066d4adb9c33560e4f46e44bb3da0015c94f7088736864200",
    )));
    assert_vector(
      &cipher,
      "e0eaf5284d884a0e77d31646",
      "",
      "",
      "169fbb2fbf389a995f6390af22228a62",
    );
  }

  #[track_caller]
  fn assert_rejects_wrong_nonce_and_tag_length<A: Aead<Nonce = Nonce96>>(cipher: &A) {
    let nonce = Nonce96::from_bytes([0x03; 12]);
    let mut buffer = *b"hello siv";
    let tag = cipher
      .encrypt_in_place(&nonce, b"aad", &mut buffer)
      .expect("AES-GCM-SIV fixture must seal");
    assert_eq!(
      cipher.decrypt_in_place(&Nonce96::from_bytes([0x04; 12]), b"aad", &mut buffer, &tag),
      Err(OpenError::verification())
    );

    for len in [0, 15, 17] {
      assert_eq!(A::tag_from_slice(&[0u8; 17][..len]).err(), Some(AeadBufferError::new()));
    }
    let tag = A::tag_from_slice(&[0x5a; TAG_SIZE]).expect("a 16-byte AES-GCM-SIV tag must be accepted");
    assert_eq!(tag.as_ref(), [0x5a; TAG_SIZE]);
  }

  #[test]
  fn rejects_wrong_nonce_and_tag_length() {
    assert_rejects_wrong_nonce_and_tag_length(&Aes128GcmSiv::new(&Aes128GcmSivKey::from_bytes([0x01; 16])));
    assert_rejects_wrong_nonce_and_tag_length(&Aes256GcmSiv::new(&Aes256GcmSivKey::from_bytes([0x01; 32])));
  }

  #[test]
  #[cfg(target_pointer_width = "64")]
  fn input_limit_matches_rfc8452() {
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
