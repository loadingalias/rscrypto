//! AES-GCM public AEAD surface (NIST SP 800-38D).
//!
//! One generic core, [`Gcm`], holds the mode for both key sizes. [`Aes128Gcm`] and [`Aes256Gcm`]
//! are concrete wrappers over it, so each key size compiles its own code.

use core::fmt;

#[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
use super::ghash::ghash_update_padded_wide_ppc;
use super::ghash::{GhashAccumulator, ghash_update_padded, ghash_update_padded_wide};
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
use super::ghash::{ghash_collect_padded, ghash_collect_padded_block, ghash_update_padded_wide_x86};
use super::{
  AeadBufferError, LengthOverflow, Nonce96, OpenError, SealError,
  aes::{self, AesKey},
  ghash, polyval,
  targets::{AeadBackend, AeadPrimitive, select_backend},
};
use crate::traits::{Aead, AeadGather, ct};

const TAG_SIZE: usize = 16;
const NONCE_SIZE: usize = Nonce96::LENGTH;

/// Maximum plaintext length per NIST SP 800-38D: 2^39 - 256 bits = 2^36 - 32 bytes.
/// In practice the portable CTR uses a 32-bit counter, limiting to (2^32 - 2) blocks.
const MAX_PLAINTEXT_LEN: u64 = ((1u64 << 32).strict_sub(2)).strict_mul(16); // ~64 GiB

#[cfg(all(
  target_arch = "x86_64",
  target_feature = "sse2",
  any(target_os = "linux", target_os = "windows")
))]
const X86_VAES_GCM_MIN_LEN: usize = 64;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
const X86_AESNI_GCM_MIN_LEN: usize = 64;

/// The AES-GCM state of one key: its AES schedule, the GHASH key powers, and the selected backend.
struct Gcm<K> {
  /// Pre-expanded AES round keys.
  ek: K,
  /// Precomputed H powers [H^4, H^3, H^2, H] in the POLYVAL domain
  /// for 4-block wide GHASH processing.
  #[cfg(not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", target_feature = "sse2")
  )))]
  h_powers_rev: [u128; 4],
  /// Precomputed H powers [H^8, H^7, ..., H] for 8-block GHASH windows.
  #[cfg(not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", target_feature = "sse2")
  )))]
  h_powers_rev_8: [u128; 8],
  /// Precomputed H powers [H^16, H^15, ..., H] for 16-block GHASH windows.
  #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
  h_powers_rev_16: [u128; 16],
  /// Reversed H powers; every x86 GHASH window is a suffix of this table.
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  powers: polyval::X86GcmPowers,
  /// Pair-packed 16-block GHASH powers for AArch64 assembly.
  #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
  h_powers_rev_16_pair: [u128; 24],
  backend: AeadBackend,
}

impl<K: AesKey> Gcm<K> {
  // The wrapper's `Aead::new` holds this body, as the per-key-size constructor did.
  #[inline(always)]
  fn new(key: &K::Bytes, primitive: AeadPrimitive) -> Self {
    let ek = K::expand(key);

    // H = AES_K(0^128)
    let mut h = [0u8; 16];
    ek.encrypt_block(&mut h);

    // Precompute H powers for 4/8-block wide GHASH.
    // GHASH loads H as BE, then applies mulX_POLYVAL to get the POLYVAL-domain key.
    let h_polyval = ghash::h_to_polyval(&h);
    ct::zeroize(&mut h);
    // One chain of powers feeds every window table; each table is a reversed prefix.
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    let powers = polyval::X86GcmPowers::new(h_polyval);
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
    let powers = polyval::precompute_powers_chained::<16>(h_polyval);
    #[cfg(not(any(
      all(target_arch = "x86_64", target_feature = "sse2"),
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux"))
    )))]
    let powers = polyval::precompute_powers_chained::<8>(h_polyval);
    #[cfg(not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    let h_powers_rev = [powers[3], powers[2], powers[1], powers[0]];
    #[cfg(not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    let h_powers_rev_8 = core::array::from_fn(|i| powers[7usize.strict_sub(i)]);
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
    let h_powers_rev_16 = core::array::from_fn(|i| powers[15usize.strict_sub(i)]);
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
    let h_powers_rev_16_pair = polyval::precompute_powers_16_pair(&h_powers_rev_16);
    let backend = select_backend(primitive, crate::platform::arch(), crate::platform::caps());

    Self {
      ek,
      #[cfg(not(any(
        all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
        all(target_arch = "x86_64", target_feature = "sse2")
      )))]
      h_powers_rev,
      #[cfg(not(any(
        all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
        all(target_arch = "x86_64", target_feature = "sse2")
      )))]
      h_powers_rev_8,
      #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
      h_powers_rev_16,
      #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
      powers,
      #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
      h_powers_rev_16_pair,
      backend,
    }
  }

  // The smaller GHASH windows are suffixes of the 16-power authority on these targets.
  #[inline]
  fn h_powers_4(&self) -> &[u128; 4] {
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
    {
      &self.h_powers_rev_16.as_chunks::<4>().0[3]
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
      self.powers.window::<4>()
    }
    #[cfg(not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    {
      &self.h_powers_rev
    }
  }

  #[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "powerpc64", not(feature = "portable-only"))
  ))]
  #[inline]
  fn h_powers_8(&self) -> &[u128; 8] {
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
    {
      &self.h_powers_rev_16.as_chunks::<8>().0[1]
    }
    #[cfg(not(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux"))))]
    {
      &self.h_powers_rev_8
    }
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  #[inline]
  fn x86_gcm_tables(&self) -> aes::X86GcmTables<'_> {
    aes::X86GcmTables {
      h_polyval: self.h_powers_4()[3],
      h_powers_rev: self.h_powers_4(),
      h_powers_rev_16: self.powers.window::<16>(),
      #[cfg(any(target_os = "linux", target_os = "windows"))]
      h_powers_rev_32: self.powers.window::<32>(),
      #[cfg(any(target_os = "linux", target_os = "windows"))]
      h_powers_rev_64: self.powers.window::<64>(),
      #[cfg(any(target_os = "linux", target_os = "windows"))]
      h_powers_rev_128: self.powers.window::<128>(),
    }
  }

  #[cfg(target_arch = "aarch64")]
  #[inline]
  fn aarch64_gcm_tables(&self) -> aes::Aarch64GcmTables<'_> {
    aes::Aarch64GcmTables {
      h_powers_rev: self.h_powers_4(),
      h_powers_rev_8: self.h_powers_8(),
      #[cfg(any(target_os = "macos", target_os = "linux"))]
      h_powers_rev_16: &self.h_powers_rev_16,
      #[cfg(any(target_os = "macos", target_os = "linux"))]
      h_powers_rev_16_pair: &self.h_powers_rev_16_pair,
    }
  }

  /// Compute the GCM authentication tag.
  ///
  /// tag = AES(K, J0) XOR GHASH(H, AAD, ciphertext)
  ///
  /// GHASH processes: padded AAD || padded ciphertext || length block,
  /// where the length block is [len(A) in bits as u64 BE || len(C) in bits as u64 BE].
  #[inline]
  fn compute_tag(&self, j0: &[u8; 16], aad: &[u8], ciphertext: &[u8]) -> Result<[u8; TAG_SIZE], LengthOverflow> {
    let h_powers_rev = self.h_powers_4();
    let h_polyval = h_powers_rev[3];
    let length_block = super::AeadByteLengths::try_new_bit_lengths(aad.len(), ciphertext.len())?.to_be_bits_block();
    let mut acc = GhashAccumulator(0);

    #[cfg(target_arch = "riscv64")]
    if polyval::clmul128_reduce_is_portable() {
      acc.0 = super::ghash::ghash_transcript_portable(h_powers_rev, aad, ciphertext, u128::from_be_bytes(length_block));
      return Ok(self.encrypt_j0_tag(j0, acc.0));
    }

    acc.0 = ghash_update_padded(acc.0, h_polyval, aad);
    acc.0 = ghash_update_padded(acc.0, h_polyval, ciphertext);
    acc.0 ^= u128::from_be_bytes(length_block);
    acc.0 = polyval::clmul128_reduce(acc.0, h_polyval);

    Ok(self.encrypt_j0_tag(j0, acc.0))
  }

  /// Compute the GCM authentication tag using 4-block wide GHASH.
  ///
  /// Same semantics as `compute_tag` but processes AAD and ciphertext in
  /// 4-block (64-byte) chunks via `accumulate_4blocks`.
  #[inline]
  fn compute_tag_wide(&self, j0: &[u8; 16], aad: &[u8], ciphertext: &[u8]) -> Result<[u8; TAG_SIZE], LengthOverflow> {
    let h_powers_rev = self.h_powers_4();
    let h_polyval = h_powers_rev[3];
    let mut acc = GhashAccumulator(0);
    acc.0 = ghash_update_padded_wide(acc.0, h_polyval, h_powers_rev, aad);
    acc.0 = ghash_update_padded_wide(acc.0, h_polyval, h_powers_rev, ciphertext);

    // Length block.
    let length_block = super::AeadByteLengths::try_new_bit_lengths(aad.len(), ciphertext.len())?.to_be_bits_block();
    acc.0 ^= u128::from_be_bytes(length_block);
    acc.0 = polyval::clmul128_reduce(acc.0, h_polyval);

    Ok(self.encrypt_j0_tag(j0, acc.0))
  }

  /// Compute the tag of a transcript of at most four GHASH blocks with one aggregate, or return
  /// `None` for a longer transcript.
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  #[inline]
  fn compute_tag_short_wide(
    &self,
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
      return Ok(Some(self.encrypt_j0_tag(j0, 0)));
    }

    let mut blocks = [0u128; 4];
    let start = 4usize.strict_sub(block_count);
    blocks[start..].copy_from_slice(&sequence[..block_count]);
    let h_powers_rev = self.h_powers_4();
    let acc = GhashAccumulator(polyval::accumulate_4blocks(0, h_powers_rev[3], h_powers_rev, &blocks));
    Ok(Some(self.encrypt_j0_tag(j0, acc.0)))
  }

  #[inline(always)]
  fn encrypt_j0_tag(&self, j0: &[u8; 16], acc: u128) -> [u8; TAG_SIZE] {
    let mut encrypted_j0 = *j0;
    self.ek.encrypt_block(&mut encrypted_j0);
    let tag = (acc ^ u128::from_be_bytes(encrypted_j0)).to_be_bytes();
    ct::zeroize(&mut encrypted_j0);
    tag
  }

  #[inline]
  fn should_use_wide_ghash(&self, aad_len: usize, ciphertext_len: usize) -> bool {
    const MIN_WIDE_INPUT: usize = 64;

    if aad_len < MIN_WIDE_INPUT && ciphertext_len < MIN_WIDE_INPUT {
      return false;
    }

    match self.backend {
      AeadBackend::X86VaesVpclmul
      | AeadBackend::X86AesniPclmul
      | AeadBackend::Aarch64AesPmull
      | AeadBackend::Aarch64Sve2AesPmull
      | AeadBackend::Power8Crypto => true,
      AeadBackend::S390xMsa => crate::platform::caps().has(crate::platform::caps::s390x::VECTOR),
      _ => false,
    }
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

  // The wrapper's `Aead` method holds this body, as the per-key-size method did.
  #[inline(always)]
  fn seal(&self, nonce: &Nonce96, aad: &[u8], buffer: &mut [u8]) -> Result<[u8; TAG_SIZE], SealError> {
    super::seal_bounded_length_as_u64(buffer.len(), MAX_PLAINTEXT_LEN)?;
    let length_block = super::seal_bit_lengths(aad.len(), buffer.len())?.to_be_bits_block();
    #[cfg(not(any(
      all(target_arch = "x86_64", target_feature = "sse2"),
      target_arch = "aarch64",
      all(target_arch = "powerpc64", not(feature = "portable-only"))
    )))]
    let _ = length_block;

    let (j0, ctr_block) = make_j0_and_ctr(nonce);

    // VAES path: the widest batch kernels, then one pass over the final
    // blocks that also computes the tag.
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if self.backend == AeadBackend::X86VaesVpclmul {
      let tables = self.x86_gcm_tables();
      // SAFETY: the selected backend establishes VPCLMULQDQ and its AVX-512 prerequisites.
      let acc =
        GhashAccumulator(unsafe { ghash_update_padded_wide_x86(0, tables.h_polyval, tables.h_powers_rev, aad) });
      let data = buffer.as_mut_ptr();
      // SAFETY: X86VaesVpclmul establishes AES, SSE2, SSSE3, AVX-512F/VL/BW/DQ,
      // VAES, VPCLMULQDQ, and PCLMULQDQ. `buffer` is readable and writable for
      // its validated length and is sealed in place.
      if let Some(tag) = unsafe {
        self.ek.gcm_seal_x86(
          &j0,
          &ctr_block,
          acc.0,
          data,
          buffer.len(),
          &[],
          data,
          &length_block,
          tables,
        )
      } {
        return Ok(tag);
      }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if self.backend == AeadBackend::X86AesniPclmul
      && buffer.len() >= X86_AESNI_GCM_MIN_LEN
      && crate::platform::caps().has(crate::platform::caps::x86::PCLMUL_READY)
    {
      let h_polyval = self.h_powers_4()[3];
      let mut acc = GhashAccumulator(ghash_update_padded_wide(0, h_polyval, self.h_powers_4(), aad));
      // SAFETY: fused x86 AES-NI/PCLMUL sealing because:
      // 1. Backend resolution selected X86AesniPclmul only after CPUID confirmed AES-NI and PCLMULQDQ.
      // 2. The extra PCLMUL_READY check confirms SSSE3 for in-register GHASH byte reversal.
      // 3. The helper encrypts `buffer` in place and folds ciphertext into GHASH.
      acc.0 = unsafe {
        self
          .ek
          .ctr32_encrypt_be_aesni_pclmul_ghash(&ctr_block, buffer, acc.0, h_polyval, self.h_powers_4())
      };
      acc.0 ^= u128::from_be_bytes(length_block);
      // SAFETY: x86 GHASH final multiply because PCLMUL_READY was checked above.
      acc.0 = unsafe { polyval::x86_clmul128_reduce_inline(acc.0, h_polyval) };
      return Ok(self.encrypt_j0_tag(&j0, acc.0));
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(
      self.backend,
      AeadBackend::Aarch64AesPmull | AeadBackend::Aarch64Sve2AesPmull
    ) {
      let tables = self.aarch64_gcm_tables();
      let mut acc = GhashAccumulator(0);
      // SAFETY: aarch64 GHASH AAD path because:
      // 1. Backend resolution established AES-CE and PMULL.
      // 2. `aad` is a valid byte slice; padding is handled inside the helper.
      unsafe { aes::aarch64_gcm_update(&tables, &mut acc.0, aad) };
      // SAFETY: backend selection confirmed AES-CE/PMULL, and the helper encrypts the valid buffer
      // in place while folding ciphertext into GHASH.
      unsafe {
        self
          .ek
          .ctr32_encrypt_be_aarch64_ghash(&ctr_block, buffer, &mut acc.0, &tables)
      };
      acc.0 ^= u128::from_be_bytes(length_block);
      // SAFETY: aarch64 GHASH final multiply because:
      // 1. Backend resolution selected an AES/PMULL backend only after runtime detection confirmed PMULL.
      // 2. `acc` and the borrowed table contain initialized GHASH field elements.
      acc.0 = unsafe { aes::aarch64_gcm_multiply(&tables, acc.0) };
      return Ok(self.encrypt_j0_tag(&j0, acc.0));
    }

    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    if self.backend == AeadBackend::Power8Crypto {
      // SAFETY: backend resolution selected `Power8Crypto` only after runtime detection confirmed
      // POWER8 crypto; the helper leaves `buffer` unchanged when it declines the message shape.
      let short = unsafe {
        self
          .ek
          .gcm_short_ppc::<false>(&j0, aad, buffer, &length_block, self.h_powers_4())
      };
      if let Some(tag) = short {
        return Ok(tag);
      }
      let h_polyval = self.h_powers_4()[3];
      // SAFETY: POWER8 GHASH AAD path because:
      // 1. Backend resolution selected `Power8Crypto` only after runtime detection confirmed POWER8
      //    crypto.
      // 2. `aad` is a valid byte slice; padding is handled inside the helper.
      let mut acc = GhashAccumulator(unsafe { ghash_update_padded_wide_ppc(0, h_polyval, self.h_powers_4(), aad) });
      // SAFETY: fused POWER AES-GCM sealing because:
      // 1. Backend resolution selected `Power8Crypto` only after runtime detection confirmed POWER8
      //    crypto.
      // 2. The helper encrypts `buffer` in place and folds ciphertext into GHASH.
      acc.0 = unsafe {
        self.ek.ctr32_encrypt_be_ppc_ghash(
          &ctr_block,
          buffer,
          acc.0,
          h_polyval,
          self.h_powers_4(),
          self.h_powers_8(),
        )
      };
      acc.0 ^= u128::from_be_bytes(length_block);
      // SAFETY: POWER8 carryless multiply because backend resolution confirmed POWER8 crypto.
      acc.0 = unsafe { polyval::ppc_clmul128_reduce_inline(acc.0, h_polyval) };
      return Ok(self.encrypt_j0_tag(&j0, acc.0));
    }

    // Scalar path: single-block AES-NI/CE/portable + single-block GHASH.
    self.ek.ctr32_encrypt_be(&ctr_block, buffer);
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if self.backend == AeadBackend::X86VaesVpclmul
      && let Some(tag) = self
        .compute_tag_short_wide(&j0, aad, buffer)
        .map_err(|_| SealError::too_large())?
    {
      return Ok(tag);
    }
    let tag = if self.should_use_wide_ghash(aad.len(), buffer.len()) {
      self.compute_tag_wide(&j0, aad, buffer)
    } else {
      self.compute_tag(&j0, aad, buffer)
    };
    tag.map_err(|_| SealError::too_large())
  }

  // The wrapper's `Aead` method holds this body, as the per-key-size method did.
  #[inline(always)]
  fn open(&self, nonce: &Nonce96, aad: &[u8], buffer: &mut [u8], tag: &[u8; TAG_SIZE]) -> Result<(), OpenError> {
    super::open_bounded_length_as_u64(buffer.len(), MAX_PLAINTEXT_LEN)?;
    let length_block = super::open_bit_lengths(aad.len(), buffer.len())?.to_be_bits_block();
    #[cfg(not(any(
      all(target_arch = "x86_64", target_feature = "sse2"),
      target_arch = "aarch64",
      all(target_arch = "powerpc64", not(feature = "portable-only"))
    )))]
    let _ = length_block;

    let (j0, ctr_block) = make_j0_and_ctr(nonce);

    // VAES path: the widest batch kernels, then one pass over the final
    // blocks whose plaintext is written only after the tag check.
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if self.backend == AeadBackend::X86VaesVpclmul {
      let tables = self.x86_gcm_tables();
      // SAFETY: the selected backend establishes VPCLMULQDQ and its AVX-512 prerequisites.
      let acc =
        GhashAccumulator(unsafe { ghash_update_padded_wide_x86(0, tables.h_polyval, tables.h_powers_rev, aad) });
      // SAFETY: X86VaesVpclmul establishes every kernel feature. `buffer` is
      // readable and writable for its validated length.
      if let Some(matched) = unsafe {
        self.ek.gcm_open_x86(
          &j0,
          &ctr_block,
          acc.0,
          buffer.as_mut_ptr(),
          buffer.len(),
          &length_block,
          tables,
          tag,
        )
      } {
        // The kernels zeroed `buffer` on a mismatch.
        return if matched {
          Ok(())
        } else {
          Err(OpenError::verification())
        };
      }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if self.backend == AeadBackend::X86AesniPclmul
      && buffer.len() >= X86_AESNI_GCM_MIN_LEN
      && crate::platform::caps().has(crate::platform::caps::x86::PCLMUL_READY)
    {
      let h_polyval = self.h_powers_4()[3];
      let mut acc = GhashAccumulator(ghash_update_padded_wide(0, h_polyval, self.h_powers_4(), aad));
      // SAFETY: fused x86 AES-NI/PCLMUL open because:
      // 1. Backend resolution selected X86AesniPclmul only after CPUID confirmed AES-NI and PCLMULQDQ.
      // 2. The extra PCLMUL_READY check confirms SSSE3 for in-register GHASH byte reversal.
      // 3. The helper GHASHes ciphertext before decrypting each chunk in place.
      acc.0 = unsafe {
        self
          .ek
          .ctr32_decrypt_be_aesni_pclmul_ghash(&ctr_block, buffer, acc.0, h_polyval, self.h_powers_4())
      };
      acc.0 ^= u128::from_be_bytes(length_block);
      // SAFETY: x86 GHASH final multiply because PCLMUL_READY was checked above.
      acc.0 = unsafe { polyval::x86_clmul128_reduce_inline(acc.0, h_polyval) };
      return Self::verify(&self.encrypt_j0_tag(&j0, acc.0), tag, buffer);
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(
      self.backend,
      AeadBackend::Aarch64AesPmull | AeadBackend::Aarch64Sve2AesPmull
    ) {
      let tables = self.aarch64_gcm_tables();
      let mut acc = GhashAccumulator(0);
      // SAFETY: aarch64 GHASH AAD path because:
      // 1. Backend resolution established AES-CE and PMULL.
      // 2. `aad` is a valid byte slice; padding is handled inside the helper.
      unsafe { aes::aarch64_gcm_update(&tables, &mut acc.0, aad) };
      // SAFETY: backend selection confirmed AES-CE/PMULL, and the helper GHASHes ciphertext before
      // decrypting the valid buffer in place.
      unsafe {
        self
          .ek
          .ctr32_decrypt_be_aarch64_ghash(&ctr_block, buffer, &mut acc.0, &tables)
      };
      acc.0 ^= u128::from_be_bytes(length_block);
      // SAFETY: aarch64 GHASH final multiply because:
      // 1. Backend resolution selected an AES/PMULL backend only after runtime detection confirmed PMULL.
      // 2. `acc` and the borrowed table contain initialized GHASH field elements.
      acc.0 = unsafe { aes::aarch64_gcm_multiply(&tables, acc.0) };
      return Self::verify(&self.encrypt_j0_tag(&j0, acc.0), tag, buffer);
    }

    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    if self.backend == AeadBackend::Power8Crypto {
      // SAFETY: backend resolution selected `Power8Crypto` only after runtime detection confirmed
      // POWER8 crypto; the helper leaves `buffer` unchanged when it declines the message shape.
      let short = unsafe {
        self
          .ek
          .gcm_short_ppc::<true>(&j0, aad, buffer, &length_block, self.h_powers_4())
      };
      if let Some(expected) = short {
        return Self::verify(&expected, tag, buffer);
      }
      let h_polyval = self.h_powers_4()[3];
      // SAFETY: POWER8 GHASH AAD path because:
      // 1. Backend resolution selected `Power8Crypto` only after runtime detection confirmed POWER8
      //    crypto.
      // 2. `aad` is a valid byte slice; padding is handled inside the helper.
      let mut acc = GhashAccumulator(unsafe { ghash_update_padded_wide_ppc(0, h_polyval, self.h_powers_4(), aad) });
      // SAFETY: fused POWER AES-GCM open because:
      // 1. Backend resolution selected `Power8Crypto` only after runtime detection confirmed POWER8
      //    crypto.
      // 2. The helper GHASHes ciphertext before decrypting the valid buffer in place.
      acc.0 = unsafe {
        self.ek.ctr32_decrypt_be_ppc_ghash(
          &ctr_block,
          buffer,
          acc.0,
          h_polyval,
          self.h_powers_4(),
          self.h_powers_8(),
        )
      };
      acc.0 ^= u128::from_be_bytes(length_block);
      // SAFETY: POWER8 carryless multiply because backend resolution confirmed POWER8 crypto.
      acc.0 = unsafe { polyval::ppc_clmul128_reduce_inline(acc.0, h_polyval) };
      return Self::verify(&self.encrypt_j0_tag(&j0, acc.0), tag, buffer);
    }

    // Scalar path: authenticate then decrypt.
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if self.backend == AeadBackend::X86VaesVpclmul
      && let Some(expected) = self
        .compute_tag_short_wide(&j0, aad, buffer)
        .map_err(|_| OpenError::too_large())?
    {
      Self::verify(&expected, tag, buffer)?;
      self.ek.ctr32_encrypt_be(&ctr_block, buffer);
      return Ok(());
    }
    let expected = if self.should_use_wide_ghash(aad.len(), buffer.len()) {
      self.compute_tag_wide(&j0, aad, buffer)
    } else {
      self.compute_tag(&j0, aad, buffer)
    };
    Self::verify(&expected.map_err(|_| OpenError::too_large())?, tag, buffer)?;
    self.ek.ctr32_encrypt_be(&ctr_block, buffer);
    Ok(())
  }

  // The wrapper's `AeadGather` method holds this body.
  #[inline(always)]
  fn open_segments(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    segments: &mut [&mut [u8]],
    tag: &[u8; TAG_SIZE],
  ) -> Result<(), OpenError> {
    if let Some(buffer) = super::segments::contiguous(segments) {
      return self.open(nonce, aad, buffer, tag);
    }
    let len = super::segments::total_len(segments).ok_or_else(OpenError::too_large)?;
    super::open_bounded_length_as_u64(len, MAX_PLAINTEXT_LEN)?;
    let length_block = super::open_bit_lengths(aad.len(), len)?.to_be_bits_block();
    let (j0, _) = make_j0_and_ctr(nonce);

    // VAES: the batch kernels open each segment's whole batches and AES-NI its
    // other whole blocks; the short kernel opens the rest and checks the tag.
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if self.backend == AeadBackend::X86VaesVpclmul {
      let tables = self.x86_gcm_tables();
      let state = OpenState {
        // SAFETY: the selected backend establishes VPCLMULQDQ and its AVX-512 prerequisites.
        acc: GhashAccumulator(unsafe { ghash_update_padded_wide_x86(0, tables.h_polyval, tables.h_powers_rev, aad) }),
        counter: 2,
      };
      let matched = super::segments::open::<16, _>(
        segments,
        len,
        state,
        |state, data| {
          #[cfg(any(target_os = "linux", target_os = "windows"))]
          // SAFETY: X86VaesVpclmul establishes every kernel feature, and `data` is
          // an exclusive range whose counters the validated length bounds.
          let done = unsafe {
            self.ek.gcm_open_bulk_x86(
              &state.counter_block(nonce),
              data.as_mut_ptr(),
              data.len(),
              &mut state.acc.0,
              tables,
            )
          }
          .unwrap_or(0);
          #[cfg(not(any(target_os = "linux", target_os = "windows")))]
          let done = 0;
          state.advance(done);
          let rest = &mut data[done..];
          // SAFETY: X86VaesVpclmul establishes AES-NI, PCLMULQDQ, SSE2, and SSSE3.
          state.acc.0 = unsafe {
            self.ek.ctr32_decrypt_be_aesni_pclmul_ghash(
              &state.counter_block(nonce),
              rest,
              state.acc.0,
              tables.h_polyval,
              tables.h_powers_rev,
            )
          };
          state.advance(rest.len());
        },
        |state, rest| {
          // SAFETY: X86VaesVpclmul establishes every kernel feature. `rest` is the
          // exclusive rest of the validated message, and `acc` covers everything before it.
          // This backend's key has the AES-NI schedule; as in the gather seal, a key
          // without it fails closed.
          unsafe {
            self.ek.gcm_open_x86(
              &j0,
              &state.counter_block(nonce),
              state.acc.0,
              rest.as_mut_ptr(),
              rest.len(),
              &length_block,
              tables,
              tag,
            )
          }
          .unwrap_or(false)
        },
      );
      return super::segments::verdict(segments, matched);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if self.backend == AeadBackend::X86AesniPclmul
      && len >= X86_AESNI_GCM_MIN_LEN
      && crate::platform::caps().has(crate::platform::caps::x86::PCLMUL_READY)
    {
      let h_polyval = self.h_powers_4()[3];
      let acc = ghash_update_padded_wide(0, h_polyval, self.h_powers_4(), aad);
      let matched = self.open_segments_fused(
        nonce,
        &j0,
        segments,
        len,
        &length_block,
        acc,
        tag,
        // SAFETY: backend resolution selected X86AesniPclmul after CPUID confirmed
        // AES-NI and PCLMULQDQ, and PCLMUL_READY confirms SSSE3.
        |acc, counter_block, data| unsafe {
          self
            .ek
            .ctr32_decrypt_be_aesni_pclmul_ghash(counter_block, data, acc, h_polyval, self.h_powers_4())
        },
        // SAFETY: PCLMUL_READY was checked above.
        |acc| unsafe { polyval::x86_clmul128_reduce_inline(acc, h_polyval) },
      );
      return super::segments::verdict(segments, matched);
    }

    #[cfg(target_arch = "aarch64")]
    if matches!(
      self.backend,
      AeadBackend::Aarch64AesPmull | AeadBackend::Aarch64Sve2AesPmull
    ) {
      let tables = self.aarch64_gcm_tables();
      let mut acc = GhashAccumulator(0);
      // SAFETY: backend resolution established AES-CE and PMULL; `aad` is a valid slice.
      unsafe { aes::aarch64_gcm_update(&tables, &mut acc.0, aad) };
      let matched = self.open_segments_fused(
        nonce,
        &j0,
        segments,
        len,
        &length_block,
        acc.0,
        tag,
        |mut acc, counter_block, data| {
          // SAFETY: backend resolution established AES-CE and PMULL; the helper
          // GHASHes each ciphertext block before it decrypts it in place.
          unsafe {
            self
              .ek
              .ctr32_decrypt_be_aarch64_ghash(counter_block, data, &mut acc, &tables)
          };
          acc
        },
        // SAFETY: backend resolution established PMULL.
        |acc| unsafe { aes::aarch64_gcm_multiply(&tables, acc) },
      );
      return super::segments::verdict(segments, matched);
    }

    #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only")))]
    if self.backend == AeadBackend::Power8Crypto {
      let h_polyval = self.h_powers_4()[3];
      // SAFETY: backend resolution selected `Power8Crypto` after runtime detection confirmed POWER8 crypto.
      let acc = unsafe { ghash_update_padded_wide_ppc(0, h_polyval, self.h_powers_4(), aad) };
      let matched = self.open_segments_fused(
        nonce,
        &j0,
        segments,
        len,
        &length_block,
        acc,
        tag,
        // SAFETY: backend resolution confirmed POWER8 crypto; the helper GHASHes
        // ciphertext before it decrypts the range in place.
        |acc, counter_block, data| unsafe {
          self.ek.ctr32_decrypt_be_ppc_ghash(
            counter_block,
            data,
            acc,
            h_polyval,
            self.h_powers_4(),
            self.h_powers_8(),
          )
        },
        // SAFETY: backend resolution confirmed POWER8 crypto.
        |acc| unsafe { polyval::ppc_clmul128_reduce_inline(acc, h_polyval) },
      );
      return super::segments::verdict(segments, matched);
    }

    // GHASH, then CTR, over each range.
    let h_powers_rev = self.h_powers_4();
    let h_polyval = h_powers_rev[3];
    let wide = self.should_use_wide_ghash(aad.len(), len);
    let ghash = |acc, data: &[u8]| {
      if wide {
        ghash_update_padded_wide(acc, h_polyval, h_powers_rev, data)
      } else {
        ghash_update_padded(acc, h_polyval, data)
      }
    };
    let matched = self.open_segments_fused(
      nonce,
      &j0,
      segments,
      len,
      &length_block,
      ghash(0, aad),
      tag,
      |acc, counter_block, data| {
        let acc = ghash(acc, data);
        self.ek.ctr32_encrypt_be(counter_block, data);
        acc
      },
      |acc| polyval::clmul128_reduce(acc, h_polyval),
    );
    super::segments::verdict(segments, matched)
  }

  /// Open `segments` with `fold`, which GHASHes a contiguous ciphertext range
  /// and decrypts it in place from a counter block, and `reduce`, the final
  /// GHASH multiply. Returns whether the tag matched.
  #[expect(
    clippy::too_many_arguments,
    reason = "a message boundary: nonce, segments, lengths, GHASH state, tag, and kernels"
  )]
  #[inline(always)]
  fn open_segments_fused(
    &self,
    nonce: &Nonce96,
    j0: &[u8; 16],
    segments: &mut [&mut [u8]],
    len: usize,
    length_block: &[u8; 16],
    acc: u128,
    tag: &[u8; TAG_SIZE],
    fold: impl Fn(u128, &[u8; 16], &mut [u8]) -> u128,
    reduce: impl FnOnce(u128) -> u128,
  ) -> bool {
    let state = OpenState {
      acc: GhashAccumulator(acc),
      counter: 2,
    };
    super::segments::open::<16, _>(
      segments,
      len,
      state,
      |state, data| {
        state.acc.0 = fold(state.acc.0, &state.counter_block(nonce), data);
        state.advance(data.len());
      },
      |mut state, rest| {
        state.acc.0 = fold(state.acc.0, &state.counter_block(nonce), rest) ^ u128::from_be_bytes(*length_block);
        state.acc.0 = reduce(state.acc.0);
        ct::fixed_eq(&self.encrypt_j0_tag(j0, state.acc.0), tag).declassify()
      },
    )
  }

  // The wrapper's `AeadGather` method holds this body, as the per-key-size method did.
  #[inline(always)]
  fn seal_gather<'input, 'output>(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    plaintext: impl IntoIterator<Item = &'input [u8]>,
    out: &'output mut [core::mem::MaybeUninit<u8>],
  ) -> Result<([u8; TAG_SIZE], &'output mut [u8]), SealError> {
    validate_gather_lengths(aad.len(), out.len())?;
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if self.backend == AeadBackend::X86VaesVpclmul && !out.is_empty() {
      let length_block = super::seal_bit_lengths(aad.len(), out.len())?.to_be_bits_block();
      let (j0, ctr_block) = make_j0_and_ctr(nonce);
      let tables = self.x86_gcm_tables();
      // SAFETY: the selected backend establishes VPCLMULQDQ and its AVX-512 prerequisites.
      let acc =
        GhashAccumulator(unsafe { ghash_update_padded_wide_x86(0, tables.h_polyval, tables.h_powers_rev, aad) });
      // A TLS record's payload and content type seal like one contiguous
      // message; other shapes take the general path.
      return super::gather::head_tail_spans(
        plaintext,
        out,
        aes::X86_SHORT_GCM_MAX_TAIL,
        |head, tail, dst| {
          // SAFETY: X86VaesVpclmul establishes every kernel feature. `head` and
          // the tail spans are shared input, disjoint from the exclusive output,
          // whose length the driver checked against the validated message; the
          // tail holds at most X86_SHORT_GCM_MAX_TAIL bytes.
          unsafe {
            self.ek.gcm_seal_x86(
              &j0,
              &ctr_block,
              acc.0,
              head.as_ptr(),
              head.len(),
              tail,
              dst,
              &length_block,
              tables,
            )
          }
          .ok_or_else(SealError::buffer)
        },
        |rest, out| self.seal_gather_general(nonce, aad, rest, out),
      );
    }
    self.seal_gather_general(nonce, aad, plaintext, out)
  }

  // Always inlined, so targets without the record path compile main's gather
  // entry unchanged.
  #[inline(always)]
  fn seal_gather_general<'input, 'output>(
    &self,
    nonce: &Nonce96,
    aad: &[u8],
    plaintext: impl IntoIterator<Item = &'input [u8]>,
    out: &'output mut [core::mem::MaybeUninit<u8>],
  ) -> Result<([u8; TAG_SIZE], &'output mut [u8]), SealError> {
    let (j0, _) = make_j0_and_ctr(nonce);
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
    if matches!(
      self.backend,
      AeadBackend::Aarch64AesPmull | AeadBackend::Aarch64Sve2AesPmull
    ) && out.len() >= 128
    {
      let length_block = super::seal_bit_lengths(aad.len(), out.len())?.to_be_bits_block();
      let tables = self.aarch64_gcm_tables();
      let mut acc = GhashAccumulator(0);
      // SAFETY: the selected backend establishes AES-CE/PMULL/NEON; AAD is a valid slice.
      unsafe { aes::aarch64_gcm_update(&tables, &mut acc.0, aad) };
      let ciphertext = super::gather::fused::<16, 128, _>(
        plaintext,
        out,
        2,
        &mut acc,
        |acc, counter, input, output| {
          let mut ctr = [0u8; 16];
          ctr[..12].copy_from_slice(nonce.as_bytes());
          ctr[12..].copy_from_slice(&counter.to_be_bytes());
          // SAFETY: the selected backend establishes AES/PMULL/NEON. The gather
          // driver supplies matching disjoint input/output and the validated
          // total length bounds every counter. The helper checks EOR3 separately.
          Ok(unsafe {
            self
              .ek
              .ctr32_encrypt_be_gather_aarch64_ghash(&ctr, input, output, &mut acc.0, &tables)
          })
        },
        |counter, stream| {
          stream[..12].copy_from_slice(nonce.as_bytes());
          stream[12..].copy_from_slice(&counter.to_be_bytes());
          self.ek.encrypt_block(stream);
        },
        |acc, ciphertext| {
          // SAFETY: the selected backend establishes AES/PMULL/NEON.
          unsafe { aes::aarch64_gcm_update(&tables, &mut acc.0, ciphertext) };
        },
      )?;
      acc.0 ^= u128::from_be_bytes(length_block);
      // SAFETY: the selected backend establishes PMULL for this fixed field multiply.
      acc.0 = unsafe { aes::aarch64_gcm_multiply(&tables, acc.0) };
      return Ok((self.encrypt_j0_tag(&j0, acc.0), ciphertext));
    }

    #[cfg(all(
      target_arch = "x86_64",
      target_feature = "sse2",
      any(target_os = "linux", target_os = "windows")
    ))]
    if self.backend == AeadBackend::X86VaesVpclmul && out.len() >= X86_VAES_GCM_MIN_LEN {
      let length_block = super::seal_bit_lengths(aad.len(), out.len())?.to_be_bits_block();
      let tables = self.x86_gcm_tables();
      // SAFETY: the selected backend establishes VPCLMULQDQ and its AVX-512 prerequisites.
      let mut acc =
        GhashAccumulator(unsafe { ghash_update_padded_wide_x86(0, tables.h_polyval, tables.h_powers_rev, aad) });
      // Match the established whole-message cadence; a stripped partial tail
      // must not select a wider workspace than the in-place operation.
      let whole_message_aligned = out.len().is_multiple_of(2048);
      let ciphertext = super::gather::fused::<16, 256, _>(
        plaintext,
        out,
        2,
        &mut acc,
        |acc, counter, input, output| {
          let mut ctr = [0u8; 16];
          ctr[..12].copy_from_slice(nonce.as_bytes());
          ctr[12..].copy_from_slice(&counter.to_be_bytes());
          // SAFETY: X86VaesVpclmul establishes the complete AES/VAES/VPCLMUL
          // capability set. The gather driver passes matching disjoint input
          // and output ranges; the validated total length bounds every counter.
          Ok(unsafe {
            self
              .ek
              .ctr32_encrypt_be_gather_wide_ghash(&ctr, input, whole_message_aligned, output, &mut acc.0, tables)
          })
        },
        |counter, stream| {
          stream[..12].copy_from_slice(nonce.as_bytes());
          stream[12..].copy_from_slice(&counter.to_be_bytes());
          self.ek.encrypt_block(stream);
        },
        |acc, ciphertext| {
          acc.0 = ghash_update_padded(acc.0, tables.h_polyval, ciphertext);
        },
      )?;
      acc.0 ^= u128::from_be_bytes(length_block);
      // SAFETY: the selected backend establishes PCLMULQDQ for this fixed field multiply.
      acc.0 = unsafe { polyval::x86_clmul128_reduce_inline(acc.0, tables.h_polyval) };
      return Ok((self.encrypt_j0_tag(&j0, acc.0), ciphertext));
    }

    let ciphertext = super::gather::xor::<16, 512>(plaintext, out, 2, |counter, stream| {
      let (blocks, _) = stream.as_chunks_mut::<16>();
      for (index, block) in blocks.iter_mut().enumerate() {
        block[..12].copy_from_slice(nonce.as_bytes());
        let index = u32::try_from(index).map_err(|_| SealError::too_large())?;
        block[12..].copy_from_slice(&counter.strict_add(index).to_be_bytes());
      }
      self.ek.encrypt_blocks_ecb(blocks);
      Ok(())
    })?;
    let tag = if self.should_use_wide_ghash(aad.len(), ciphertext.len()) {
      self.compute_tag_wide(&j0, aad, ciphertext)
    } else {
      self.compute_tag(&j0, aad, ciphertext)
    };
    match tag {
      Ok(tag) => Ok((tag, ciphertext)),
      Err(_) => {
        ct::zeroize(ciphertext);
        Err(SealError::too_large())
      }
    }
  }

  /// Exercise counter-mode encryption and fold the fixed diagnostic output to one block.
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  #[inline(always)]
  fn diag_ctr32_be(&self, nonce: &Nonce96, plaintext: &[u8; 44]) -> [u8; 16] {
    let (_, ctr_block) = make_j0_and_ctr(nonce);
    let mut buffer = *plaintext;
    self.ek.ctr32_encrypt_be(&ctr_block, &mut buffer);
    super::diag_fold16(&buffer)
  }

  /// Return the GHASH accumulator for diagnostic backend comparison.
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  #[inline(always)]
  fn diag_ghash(&self, aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
    let h_polyval = self.h_powers_4()[3];
    let mut acc = GhashAccumulator(0);
    if self.should_use_wide_ghash(aad.len(), ciphertext.len()) {
      acc.0 = ghash_update_padded_wide(acc.0, h_polyval, self.h_powers_4(), aad);
      acc.0 = ghash_update_padded_wide(acc.0, h_polyval, self.h_powers_4(), ciphertext);
    } else {
      acc.0 = ghash_update_padded(acc.0, h_polyval, aad);
      acc.0 = ghash_update_padded(acc.0, h_polyval, ciphertext);
    }
    let length_block = match super::AeadByteLengths::try_new_bit_lengths(aad.len(), ciphertext.len()) {
      Ok(lengths) => lengths.to_be_bits_block(),
      Err(_) => return [0u8; 16],
    };
    acc.0 ^= u128::from_be_bytes(length_block);
    acc.0 = polyval::clmul128_reduce(acc.0, h_polyval);
    acc.0.to_be_bytes()
  }

  /// Encrypt a diagnostic GHASH accumulator into the final tag.
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  #[inline(always)]
  fn diag_tag_aes(&self, nonce: &Nonce96, acc: &[u8; 16]) -> [u8; 16] {
    let (j0, _) = make_j0_and_ctr(nonce);
    self.encrypt_j0_tag(&j0, u128::from_be_bytes(*acc))
  }
}

// The context stores the expanded key and the GHASH key powers, which are secret.
impl<K> Drop for Gcm<K> {
  fn drop(&mut self) {
    #[cfg(not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    {
      ct::zeroize_words(&mut self.h_powers_rev);
      ct::zeroize_words(&mut self.h_powers_rev_8);
    }
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
    {
      ct::zeroize_words(&mut self.h_powers_rev_16);
      ct::zeroize_words(&mut self.h_powers_rev_16_pair);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    self.powers.clear();
    // The key schedule's own Drop clears `ek`.
  }
}

/// The GHASH state and the next block counter of a segmented open.
struct OpenState {
  acc: GhashAccumulator,
  counter: u32,
}

impl OpenState {
  /// The counter block of the next ciphertext block.
  #[inline(always)]
  fn counter_block(&self, nonce: &Nonce96) -> [u8; 16] {
    let mut block = [0u8; 16];
    block[..12].copy_from_slice(nonce.as_bytes());
    block[12..].copy_from_slice(&self.counter.to_be_bytes());
    block
  }

  /// Count the whole blocks of `len` opened bytes. GCM's inc32 is modulo 2^32,
  /// and the validated message length keeps every counter distinct.
  #[inline(always)]
  fn advance(&mut self, len: usize) {
    let blocks = u32::try_from(len.strict_div(16)).expect("validated GCM length fits the 32-bit counter");
    self.counter = self.counter.wrapping_add(blocks);
  }
}

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

#[inline]
fn validate_gather_lengths(aad_len: usize, plaintext_len: usize) -> Result<(), SealError> {
  super::seal_bounded_length_as_u64(plaintext_len, MAX_PLAINTEXT_LEN)?;
  super::seal_bit_lengths(aad_len, plaintext_len)?;
  Ok(())
}

/// Define one public AES-GCM type over [`Gcm`], with its key and tag types.
macro_rules! define_aes_gcm {
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
    /// # Nonce Uniqueness
    ///
    /// **A nonce must never be reused with the same key.** Nonce reuse under
    /// AES-GCM is catastrophic: it leaks the GHASH authentication key,
    /// enabling both plaintext recovery (via crib-dragging) and universal
    /// forgery of future messages. There is no recovery — the key must be
    /// discarded.
    ///
    /// Choose one nonce construction per key. For random nonces, limit all senders together to 2^32 encryptions per key.
    /// See [NIST SP 800-38D, sections 8.2 and 8.3](https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-38d.pdf#page=27).
    ///
    /// - [`crate::aead::NonceCounter`] issues deterministic nonces with a 32-bit prefix.
    ///   This crate's deterministic profile limits each key to 2^48 invocations.
    ///   Each counter enforces the limit for one stream.
    /// - [`Aes256GcmSiv`](crate::Aes256GcmSiv) — nonce-misuse resistant (degrades to deterministic
    ///   encryption on reuse, but never leaks the auth key).
    /// - `XChaCha20Poly1305` supports independent, uniformly random 192-bit nonces.
    ///   At 2^64 messages per key, the nonce collision probability is below 2^-65.
    ///   At 2^96 messages, it is about 39%.
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
      inner: Gcm<$aes>,
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

      // Allow optimized GNU std callers to coalesce the returned context's storage.
      #[cfg_attr(
        all(
          target_arch = "aarch64",
          any(target_os = "linux", target_os = "windows"),
          target_env = "gnu",
          feature = "std",
          not(debug_assertions)
        ),
        inline(always)
      )]
      fn new(key: &Self::Key) -> Self {
        Self {
          inner: Gcm::new(key.as_bytes(), AeadPrimitive::$name),
        }
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

    impl AeadGather for $name {
      fn __validate_gather_lengths(aad_len: usize, plaintext_len: usize) -> Result<(), SealError> {
        validate_gather_lengths(aad_len, plaintext_len)
      }

      fn __encrypt_gather_with_nonce<'input, 'output>(
        &self,
        nonce: &Self::Nonce,
        aad: &[u8],
        plaintext: impl IntoIterator<Item = &'input [u8]>,
        out: &'output mut [core::mem::MaybeUninit<u8>],
        _token: crate::traits::aead::SealToken,
      ) -> Result<(Self::Tag, &'output mut [u8]), SealError> {
        let (tag, ciphertext) = self.inner.seal_gather(nonce, aad, plaintext, out)?;
        Ok(($tag::from_bytes(tag), ciphertext))
      }

      fn __decrypt_segments_in_place(
        &self,
        nonce: &Self::Nonce,
        aad: &[u8],
        segments: &mut [&mut [u8]],
        tag: &Self::Tag,
      ) -> Result<(), OpenError> {
        self.inner.open_segments(nonce, aad, segments, tag.as_bytes())
      }
    }
  };
}

define_aes_gcm! {
  /// AES-128-GCM AEAD (NIST SP 800-38D).
  ///
  /// The 128-bit-key counterpart of [`Aes256Gcm`](crate::Aes256Gcm), with the
  /// same nonce rules, 96-bit nonce, and combined and detached surfaces.
  Aes128Gcm(aes::Aes128EncKey), "AES-128-GCM",
  key: Aes128GcmKey(16, "AES-128-GCM secret key (16 bytes)."),
  tag: Aes128GcmTag("AES-128-GCM authentication tag (16 bytes).")
}

define_aes_gcm! {
  /// AES-256-GCM AEAD (NIST SP 800-38D).
  ///
  /// The standard TLS / interop workhorse.
  Aes256Gcm(aes::Aes256EncKey), "AES-256-GCM",
  key: Aes256GcmKey(32, "AES-256-GCM secret key (32 bytes)."),
  tag: Aes256GcmTag("AES-256-GCM authentication tag (16 bytes).")
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
/// Exercise AES-128 counter-mode encryption and fold the fixed diagnostic output to one block.
#[must_use]
pub fn diag_aes128gcm_ctr32_be(cipher: &Aes128Gcm, nonce: &Nonce96, plaintext: &[u8; 44]) -> [u8; 16] {
  cipher.inner.diag_ctr32_be(nonce, plaintext)
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
/// Return the AES-128-GCM GHASH accumulator for diagnostic backend comparison.
#[unsafe(no_mangle)]
#[inline(never)]
#[must_use]
pub fn diag_aes128gcm_ghash(cipher: &Aes128Gcm, aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
  cipher.inner.diag_ghash(aad, ciphertext)
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
/// Encrypt a diagnostic GHASH accumulator into the final AES-128-GCM tag.
#[must_use]
pub fn diag_aes128gcm_tag_aes(cipher: &Aes128Gcm, nonce: &Nonce96, acc: &[u8; 16]) -> [u8; 16] {
  cipher.inner.diag_tag_aes(nonce, acc)
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
/// Exercise AES-256 counter-mode encryption and fold the fixed diagnostic output to one block.
#[must_use]
pub fn diag_aes256gcm_ctr32_be(cipher: &Aes256Gcm, nonce: &Nonce96, plaintext: &[u8; 44]) -> [u8; 16] {
  cipher.inner.diag_ctr32_be(nonce, plaintext)
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
/// Return the AES-256-GCM GHASH accumulator for diagnostic backend comparison.
#[unsafe(no_mangle)]
#[inline(never)]
#[must_use]
pub fn diag_aes256gcm_ghash(cipher: &Aes256Gcm, aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
  cipher.inner.diag_ghash(aad, ciphertext)
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
/// Encrypt a diagnostic GHASH accumulator into the final AES-256-GCM tag.
#[must_use]
pub fn diag_aes256gcm_tag_aes(cipher: &Aes256Gcm, nonce: &Nonce96, acc: &[u8; 16]) -> [u8; 16] {
  cipher.inner.diag_tag_aes(nonce, acc)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::aead::{
    expert::AeadWithNonce,
    test_vectors::{hex_vec, hex12, hex16, hex32},
  };

  #[track_caller]
  fn assert_vector<A: Aead<Nonce = Nonce96>>(
    cipher: &A,
    nonce: &str,
    aad: &str,
    plaintext: &str,
    sealed: &str,
    tag: &str,
  ) {
    let nonce = Nonce96::from_bytes(hex12(nonce));
    let aad = hex_vec(aad);
    let plaintext = hex_vec(plaintext);
    let mut buffer = plaintext.clone();
    let computed = cipher
      .encrypt_in_place(&nonce, &aad, &mut buffer)
      .expect("NIST AES-GCM vector must seal");
    assert_eq!(buffer, hex_vec(sealed), "ciphertext");
    assert_eq!(computed.as_ref(), hex16(tag), "tag");
    cipher
      .decrypt_in_place(&nonce, &aad, &mut buffer, &computed)
      .expect("NIST AES-GCM vector must open");
    assert_eq!(buffer, plaintext, "plaintext");
  }

  const PLAINTEXT_60: &str = "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39";
  const PLAINTEXT_64: &str = "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255";
  const AAD_20: &str = "feedfacedeadbeeffeedfacedeadbeefabaddad2";
  const NONCE: &str = "cafebabefacedbaddecaf888";

  /// NIST SP 800-38D test cases 1 to 4.
  #[test]
  fn aes128gcm_nist_vectors() {
    let zero = Aes128Gcm::new(&Aes128GcmKey::from_bytes([0u8; 16]));
    assert_vector(
      &zero,
      "000000000000000000000000",
      "",
      "",
      "",
      "58e2fccefa7e3061367f1d57a4e7455a",
    );
    assert_vector(
      &zero,
      "000000000000000000000000",
      "",
      "00000000000000000000000000000000",
      "0388dace60b6a392f328c2b971b2fe78",
      "ab6e47d42cec13bdf53a67b21257bddf",
    );
    let cipher = Aes128Gcm::new(&Aes128GcmKey::from_bytes(hex16("feffe9928665731c6d6a8f9467308308")));
    assert_vector(
      &cipher,
      NONCE,
      "",
      PLAINTEXT_64,
      "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091473f5985",
      "4d5c2af327cd64a62cf35abd2ba6fab4",
    );
    assert_vector(
      &cipher,
      NONCE,
      AAD_20,
      PLAINTEXT_60,
      "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091",
      "5bc94fbc3221a5db94fae95ae7121a47",
    );
  }

  /// NIST SP 800-38D test cases 13 to 16.
  #[test]
  fn aes256gcm_nist_vectors() {
    let zero = Aes256Gcm::new(&Aes256GcmKey::from_bytes([0u8; 32]));
    assert_vector(
      &zero,
      "000000000000000000000000",
      "",
      "",
      "",
      "530f8afbc74536b9a963b4f1c4cb738b",
    );
    assert_vector(
      &zero,
      "000000000000000000000000",
      "",
      "00000000000000000000000000000000",
      "cea7403d4d606b6e074ec5d3baf39d18",
      "d0d1c8a799996bf0265b98b5d48ab919",
    );
    let cipher = Aes256Gcm::new(&Aes256GcmKey::from_bytes(hex32(
      "feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308",
    )));
    assert_vector(
      &cipher,
      NONCE,
      "",
      PLAINTEXT_64,
      "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662898015ad",
      "b094dac5d93471bdec1a502270e3cc6c",
    );
    assert_vector(
      &cipher,
      NONCE,
      AAD_20,
      PLAINTEXT_60,
      "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662",
      "76fc6ece0f4e1768cddf8853bb2d551b",
    );
  }

  #[track_caller]
  fn assert_rejects_wrong_nonce_and_tag_length<A: Aead<Nonce = Nonce96>>(cipher: &A) {
    let nonce = Nonce96::from_bytes([0x07; 12]);
    let mut buffer = *b"hello gcm";
    let tag = cipher
      .encrypt_in_place(&nonce, b"aad", &mut buffer)
      .expect("AES-GCM fixture must seal");
    assert_eq!(
      cipher.decrypt_in_place(&Nonce96::from_bytes([0x08; 12]), b"aad", &mut buffer, &tag),
      Err(OpenError::verification())
    );

    for len in [0, 15, 17] {
      assert_eq!(A::tag_from_slice(&[0u8; 17][..len]).err(), Some(AeadBufferError::new()));
    }
    let tag = A::tag_from_slice(&[0x5a; TAG_SIZE]).expect("a 16-byte AES-GCM tag must be accepted");
    assert_eq!(tag.as_ref(), [0x5a; TAG_SIZE]);
  }

  #[test]
  fn rejects_wrong_nonce_and_tag_length() {
    assert_rejects_wrong_nonce_and_tag_length(&Aes128Gcm::new(&Aes128GcmKey::from_bytes([0x42; 16])));
    assert_rejects_wrong_nonce_and_tag_length(&Aes256Gcm::new(&Aes256GcmKey::from_bytes([0x42; 32])));
  }
}
