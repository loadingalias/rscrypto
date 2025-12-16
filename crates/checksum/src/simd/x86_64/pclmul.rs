//! PCLMULQDQ (128-bit) CRC folding engine.
//!
//! This is a "fold + Barrett reduction" implementation for reflected CRCs.
//! Supports both 32-bit (CRC32, CRC32C) and 64-bit (CRC64) variants.
//!
//! The engine is designed to be shared with VPCLMULQDQ (AVX-512)
//! and PMULL (aarch64) variants.

#![allow(unsafe_code)]
// Rust 2024 requires explicit `unsafe {}` even inside `unsafe fn` bodies.
// This module is intrinsically "unsafe-heavy", so we allow it at file scope.
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

pub(crate) trait Crc32FoldSpec {
  const COEFF_64: (u64, u64);
  const COEFF_48: (u64, u64);
  const COEFF_32: (u64, u64);
  const COEFF_16: (u64, u64);
  const FOLD_WIDTH: (u64, u64);
  const BARRETT: (u64, u64);

  fn portable(crc: u32, data: &[u8]) -> u32;
}

pub(crate) struct Crc32cSpec;

impl Crc32FoldSpec for Crc32cSpec {
  const COEFF_64: (u64, u64) = crate::constants::crc32c::fold::COEFF_64;
  const COEFF_48: (u64, u64) = crate::constants::crc32c::fold::COEFF_48;
  const COEFF_32: (u64, u64) = crate::constants::crc32c::fold::COEFF_32;
  const COEFF_16: (u64, u64) = crate::constants::crc32c::fold::COEFF_16;
  const FOLD_WIDTH: (u64, u64) = crate::constants::crc32c::fold::FOLD_WIDTH;
  const BARRETT: (u64, u64) = crate::constants::crc32c::fold::BARRETT;

  #[inline]
  fn portable(crc: u32, data: &[u8]) -> u32 {
    crate::crc32c::portable::compute(crc, data)
  }
}

pub(crate) struct Crc32Spec;

impl Crc32FoldSpec for Crc32Spec {
  const COEFF_64: (u64, u64) = crate::constants::crc32::fold::COEFF_64;
  const COEFF_48: (u64, u64) = crate::constants::crc32::fold::COEFF_48;
  const COEFF_32: (u64, u64) = crate::constants::crc32::fold::COEFF_32;
  const COEFF_16: (u64, u64) = crate::constants::crc32::fold::COEFF_16;
  const FOLD_WIDTH: (u64, u64) = crate::constants::crc32::fold::FOLD_WIDTH;
  const BARRETT: (u64, u64) = crate::constants::crc32::fold::BARRETT;

  #[inline]
  fn portable(crc: u32, data: &[u8]) -> u32 {
    crate::crc32::portable::compute(crc, data)
  }
}

#[inline(always)]
unsafe fn load_128(ptr: *const u8) -> __m128i {
  // SAFETY: caller ensures `ptr` is valid for 16 bytes.
  // For reflected CRCs on little-endian, load data as-is without byte reflection.
  #[allow(clippy::cast_ptr_alignment)]
  _mm_loadu_si128(ptr as *const __m128i)
}

#[inline(always)]
pub(crate) unsafe fn fold16_128(current: __m128i, coeff: __m128i, data: __m128i) -> __m128i {
  // CRC-32 folding uses "cross" carryless multiplications (see Intel whitepaper):
  // - 0x10: current.high64 * coeff.low64
  // - 0x01: current.low64  * coeff.high64
  let h = _mm_clmulepi64_si128(current, coeff, 0x10);
  let l = _mm_clmulepi64_si128(current, coeff, 0x01);
  _mm_xor_si128(_mm_xor_si128(h, l), data)
}

#[inline(always)]
pub(crate) unsafe fn finalize_reduced_128<S: Crc32FoldSpec>(mut x: __m128i) -> u32 {
  // Fold 16 bytes -> 4 bytes (CRC-32 width), then Barrett reduce.
  let (high, low) = S::FOLD_WIDTH;
  let coeff_low = _mm_set_epi64x(0, low as i64);
  let coeff_high = _mm_set_epi64x(high as i64, 0);

  x = _mm_xor_si128(_mm_clmulepi64_si128(x, coeff_low, 0x00), _mm_srli_si128(x, 8));

  let mask2 = _mm_set_epi64x(0xFFFF_FFFF_0000_0000u64 as i64, 0xFFFF_FFFF_FFFF_FFFFu64 as i64);
  let masked = _mm_and_si128(x, mask2);
  let shifted = _mm_slli_si128(x, 12);
  let clmul = _mm_clmulepi64_si128(shifted, coeff_high, 0x11);
  x = _mm_xor_si128(clmul, masked);

  let (poly, mu) = S::BARRETT;
  let mu_poly = _mm_set_epi64x(poly as i64, mu as i64);
  let clmul1 = _mm_clmulepi64_si128(x, mu_poly, 0x00);
  let clmul2 = _mm_clmulepi64_si128(clmul1, mu_poly, 0x10);
  let xorred = _mm_xor_si128(x, clmul2);

  let mut out = [0u64; 2];
  #[allow(clippy::cast_ptr_alignment)]
  _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, xorred);
  out[1] as u32
}

/// Compute a reflected 32-bit CRC using PCLMULQDQ.
///
/// # Safety
/// Caller must ensure the CPU supports `pclmulqdq` and `sse2`.
#[target_feature(enable = "pclmulqdq,sse2")]
unsafe fn compute_pclmul_unchecked_impl<S: Crc32FoldSpec>(crc: u32, data: &[u8]) -> u32 {
  if data.len() < 64 {
    return S::portable(crc, data);
  }

  // Load coefficients. For CRC32, we use "cross" multiplication (0x10/0x01),
  // so coeff layout is: [high=COEFF.0, low=COEFF.1]
  let coeff_64 = {
    let (high, low) = S::COEFF_64;
    _mm_set_epi64x(high as i64, low as i64)
  };
  let coeff_48 = {
    let (high, low) = S::COEFF_48;
    _mm_set_epi64x(high as i64, low as i64)
  };
  let coeff_32 = {
    let (high, low) = S::COEFF_32;
    _mm_set_epi64x(high as i64, low as i64)
  };
  let coeff_16 = {
    let (high, low) = S::COEFF_16;
    _mm_set_epi64x(high as i64, low as i64)
  };

  let bulk_len = data.len() & !63;
  let (bulk, rem) = data.split_at(bulk_len);

  let mut blocks = bulk.chunks_exact(64);
  let first = blocks.next().unwrap();
  let base = first.as_ptr();

  // Load data WITHOUT byte reflection (correct for reflected CRCs on little-endian).
  let mut x0 = load_128(base);
  let mut x1 = load_128(base.add(16));
  let mut x2 = load_128(base.add(32));
  let mut x3 = load_128(base.add(48));

  // XOR initial CRC into low 32 bits of the first block.
  x0 = _mm_xor_si128(x0, _mm_cvtsi32_si128(crc as i32));

  for block in blocks {
    let base = block.as_ptr();
    let y0 = load_128(base);
    let y1 = load_128(base.add(16));
    let y2 = load_128(base.add(32));
    let y3 = load_128(base.add(48));

    x0 = fold16_128(x0, coeff_64, y0);
    x1 = fold16_128(x1, coeff_64, y1);
    x2 = fold16_128(x2, coeff_64, y2);
    x3 = fold16_128(x3, coeff_64, y3);
  }

  // Fold 4 accumulators -> 1.
  let mut folded = x3;
  folded = fold16_128(x2, coeff_16, folded);
  folded = fold16_128(x1, coeff_32, folded);
  folded = fold16_128(x0, coeff_48, folded);

  let crc = finalize_reduced_128::<S>(folded);
  S::portable(crc, rem)
}

/// Compute CRC32-C using `pclmulqdq` when enabled at compile time.
#[cfg(all(target_feature = "pclmulqdq", target_feature = "ssse3"))]
#[inline]
pub(crate) fn compute_pclmul_enabled(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: this function is only compiled when the required target features are enabled.
  unsafe { compute_pclmul_unchecked_impl::<Crc32cSpec>(crc, data) }
}

/// Compute CRC32-C using PCLMULQDQ when selected by runtime feature detection.
#[cfg(feature = "std")]
#[inline]
pub(crate) fn compute_pclmul_runtime(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: callers must gate this with `is_x86_feature_detected!("pclmulqdq")`
  // and `is_x86_feature_detected!("ssse3")`.
  unsafe { compute_pclmul_unchecked_impl::<Crc32cSpec>(crc, data) }
}

/// Compute CRC32 (ISO-HDLC) using `pclmulqdq` when enabled at compile time.
#[cfg(all(target_feature = "pclmulqdq", target_feature = "ssse3"))]
#[inline]
pub(crate) fn compute_pclmul_crc32_enabled(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: this function is only compiled when the required target features are enabled.
  unsafe { compute_pclmul_unchecked_impl::<Crc32Spec>(crc, data) }
}

/// Compute CRC32 (ISO-HDLC) using PCLMULQDQ when selected by runtime feature detection.
#[cfg(feature = "std")]
#[inline]
pub(crate) fn compute_pclmul_crc32_runtime(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: callers must gate this with `is_x86_feature_detected!("pclmulqdq")`
  // and `is_x86_feature_detected!("ssse3")`.
  unsafe { compute_pclmul_unchecked_impl::<Crc32Spec>(crc, data) }
}

// ============================================================================
// CRC64 Support
// ============================================================================

/// Trait for CRC64 polynomial-specific folding constants.
pub(crate) trait Crc64FoldSpec {
  const COEFF_64: (u64, u64);
  const COEFF_48: (u64, u64);
  const COEFF_32: (u64, u64);
  const COEFF_16: (u64, u64);
  const FOLD_WIDTH: (u64, u64);
  const BARRETT: (u64, u64);

  fn portable(crc: u64, data: &[u8]) -> u64;
}

/// CRC64/XZ (ECMA polynomial) specification.
pub(crate) struct Crc64XzSpec;

impl Crc64FoldSpec for Crc64XzSpec {
  const COEFF_64: (u64, u64) = crate::constants::crc64::fold::COEFF_64;
  const COEFF_48: (u64, u64) = crate::constants::crc64::fold::COEFF_48;
  const COEFF_32: (u64, u64) = crate::constants::crc64::fold::COEFF_32;
  const COEFF_16: (u64, u64) = crate::constants::crc64::fold::COEFF_16;
  const FOLD_WIDTH: (u64, u64) = crate::constants::crc64::fold::FOLD_WIDTH;
  const BARRETT: (u64, u64) = crate::constants::crc64::fold::BARRETT;

  #[inline]
  fn portable(crc: u64, data: &[u8]) -> u64 {
    crate::crc64::xz::compute_portable(crc, data)
  }
}

/// CRC64/NVME specification.
pub(crate) struct Crc64NvmeSpec;

impl Crc64FoldSpec for Crc64NvmeSpec {
  const COEFF_64: (u64, u64) = crate::constants::crc64_nvme::fold::COEFF_64;
  const COEFF_48: (u64, u64) = crate::constants::crc64_nvme::fold::COEFF_48;
  const COEFF_32: (u64, u64) = crate::constants::crc64_nvme::fold::COEFF_32;
  const COEFF_16: (u64, u64) = crate::constants::crc64_nvme::fold::COEFF_16;
  const FOLD_WIDTH: (u64, u64) = crate::constants::crc64_nvme::fold::FOLD_WIDTH;
  const BARRETT: (u64, u64) = crate::constants::crc64_nvme::fold::BARRETT;

  #[inline]
  fn portable(crc: u64, data: &[u8]) -> u64 {
    crate::crc64::nvme::compute_portable(crc, data)
  }
}

/// Fold 16 bytes using parallel multiplication for CRC64.
///
/// This uses 0x00 (lo×lo) and 0x11 (hi×hi) selectors, matching crc-fast-rust.
#[inline(always)]
pub(crate) unsafe fn fold16_128_parallel(current: __m128i, coeff: __m128i, data: __m128i) -> __m128i {
  // CRC-64 folding uses "parallel" carryless multiplications:
  // - 0x00: current.low64 * coeff.low64
  // - 0x11: current.high64 * coeff.high64
  let lo = _mm_clmulepi64_si128(current, coeff, 0x00);
  let hi = _mm_clmulepi64_si128(current, coeff, 0x11);
  _mm_xor_si128(_mm_xor_si128(lo, hi), data)
}

/// Barrett reduction from 128-bit folded value to 64-bit CRC.
///
/// For reflected CRC64, following the crc-fast-rust algorithm:
/// 1. Fold 128 bits to 64 bits using coeff.hi × state.lo (selector 0x01)
/// 2. Apply Barrett reduction: multiply by poly first, then by mu
#[inline(always)]
pub(crate) unsafe fn finalize_reduced_128_64<S: Crc64FoldSpec>(x: __m128i) -> u64 {
  let (_, key16) = S::FOLD_WIDTH;
  let (poly_simd, mu) = S::BARRETT;

  // Step 1: Fold 128 bits to 64 bits (crc-fast-rust fold_width for reflected)
  // h = coeff.lo × state.hi (using 0x01: src1.hi × src2.lo, but we swap operand order)
  // Actually: coeff[lo=0, hi=key16] × state, selector 0x01 gives: coeff.hi × state.lo = key16 ×
  // state.lo Then XOR with state >> 64
  let k_vec = _mm_set_epi64x(key16 as i64, 0);
  let h = _mm_clmulepi64_si128(k_vec, x, 0x01); // k_vec.hi × x.lo = key16 × x.lo
  let shifted = _mm_srli_si128(x, 8); // x >> 64
  let folded = _mm_xor_si128(h, shifted);

  // Step 2: Barrett reduction (crc-fast-rust order: poly first, then mu)
  // mu_poly = [poly (lo), mu (hi)]
  let mu_poly = _mm_set_epi64x(mu as i64, poly_simd as i64);

  // clmul1 = folded.lo × poly (selector 0x00)
  let clmul1 = _mm_clmulepi64_si128(folded, mu_poly, 0x00);

  // clmul2 = clmul1.lo × mu (selector 0x10: clmul1.lo × mu_poly.hi)
  let clmul2 = _mm_clmulepi64_si128(clmul1, mu_poly, 0x10);

  // Handle the x^64 term: (clmul1 << 64)
  let clmul1_shifted = _mm_slli_si128(clmul1, 8);

  // result = clmul2 ⊕ (clmul1 << 64) ⊕ folded
  let result = _mm_xor_si128(_mm_xor_si128(clmul2, clmul1_shifted), folded);

  // Extract HIGH 64 bits (this is the final CRC)
  _mm_extract_epi64(result, 1) as u64
}

/// Compute a reflected 64-bit CRC using PCLMULQDQ.
///
/// This implementation follows crc-fast-rust's approach:
/// - Uses parallel multiplication (0x00/0x11) for folding
/// - Does NOT use byte reflection for reflected CRCs
/// - XORs initial CRC into low 64 bits
///
/// # Safety
/// Caller must ensure the CPU supports `pclmulqdq` and `sse2`.
#[target_feature(enable = "pclmulqdq,sse2")]
unsafe fn compute_pclmul_crc64_unchecked_impl<S: Crc64FoldSpec>(crc: u64, data: &[u8]) -> u64 {
  if data.len() < 64 {
    return S::portable(crc, data);
  }

  // For CRC64 with parallel multiplication (0x00/0x11):
  // - 0x00: current.lo × coeff.lo
  // - 0x11: current.hi × coeff.hi
  // COEFF tuple is (high_key, low_key).
  // To match ARM's load_coeff_crc64 which puts high in lane 0 and low in lane 1,
  // we use _mm_set_epi64x(low, high) since it puts first arg in HIGH lane and second in LOW.
  let coeff_64 = {
    let (high, low) = S::COEFF_64;
    _mm_set_epi64x(low as i64, high as i64)
  };
  let coeff_48 = {
    let (high, low) = S::COEFF_48;
    _mm_set_epi64x(low as i64, high as i64)
  };
  let coeff_32 = {
    let (high, low) = S::COEFF_32;
    _mm_set_epi64x(low as i64, high as i64)
  };
  let coeff_16 = {
    let (high, low) = S::COEFF_16;
    _mm_set_epi64x(low as i64, high as i64)
  };

  let bulk_len = data.len() & !63;
  let (bulk, rem) = data.split_at(bulk_len);

  let mut blocks = bulk.chunks_exact(64);
  let first = blocks.next().unwrap();
  let base = first.as_ptr();

  // Load data WITHOUT byte reflection (matching crc-fast-rust for reflected CRCs)
  #[allow(clippy::cast_ptr_alignment)]
  let mut x0 = _mm_loadu_si128(base as *const __m128i);
  #[allow(clippy::cast_ptr_alignment)]
  let mut x1 = _mm_loadu_si128(base.add(16) as *const __m128i);
  #[allow(clippy::cast_ptr_alignment)]
  let mut x2 = _mm_loadu_si128(base.add(32) as *const __m128i);
  #[allow(clippy::cast_ptr_alignment)]
  let mut x3 = _mm_loadu_si128(base.add(48) as *const __m128i);

  // XOR initial CRC into low 64 bits of the first block.
  x0 = _mm_xor_si128(x0, _mm_set_epi64x(0, crc as i64));

  for block in blocks {
    let base = block.as_ptr();
    #[allow(clippy::cast_ptr_alignment)]
    let y0 = _mm_loadu_si128(base as *const __m128i);
    #[allow(clippy::cast_ptr_alignment)]
    let y1 = _mm_loadu_si128(base.add(16) as *const __m128i);
    #[allow(clippy::cast_ptr_alignment)]
    let y2 = _mm_loadu_si128(base.add(32) as *const __m128i);
    #[allow(clippy::cast_ptr_alignment)]
    let y3 = _mm_loadu_si128(base.add(48) as *const __m128i);

    // Use parallel multiplication for CRC64
    x0 = fold16_128_parallel(x0, coeff_64, y0);
    x1 = fold16_128_parallel(x1, coeff_64, y1);
    x2 = fold16_128_parallel(x2, coeff_64, y2);
    x3 = fold16_128_parallel(x3, coeff_64, y3);
  }

  // Fold 4 accumulators -> 1 using parallel multiplication
  let mut folded = x3;
  folded = fold16_128_parallel(x2, coeff_16, folded);
  folded = fold16_128_parallel(x1, coeff_32, folded);
  folded = fold16_128_parallel(x0, coeff_48, folded);

  let crc = finalize_reduced_128_64::<S>(folded);
  S::portable(crc, rem)
}

/// Compute CRC64/XZ using `pclmulqdq` when enabled at compile time.
#[cfg(all(target_feature = "pclmulqdq", target_feature = "ssse3"))]
#[inline]
pub(crate) fn compute_pclmul_crc64_xz_enabled(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: this function is only compiled when the required target features are enabled.
  unsafe { compute_pclmul_crc64_unchecked_impl::<Crc64XzSpec>(crc, data) }
}

/// Compute CRC64/XZ using PCLMULQDQ when selected by runtime feature detection.
#[cfg(feature = "std")]
#[inline]
pub(crate) fn compute_pclmul_crc64_xz_runtime(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: callers must gate this with `is_x86_feature_detected!("pclmulqdq")`
  // and `is_x86_feature_detected!("ssse3")`.
  unsafe { compute_pclmul_crc64_unchecked_impl::<Crc64XzSpec>(crc, data) }
}

/// Compute CRC64/NVME using `pclmulqdq` when enabled at compile time.
#[cfg(all(target_feature = "pclmulqdq", target_feature = "ssse3"))]
#[inline]
pub(crate) fn compute_pclmul_crc64_nvme_enabled(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: this function is only compiled when the required target features are enabled.
  unsafe { compute_pclmul_crc64_unchecked_impl::<Crc64NvmeSpec>(crc, data) }
}

/// Compute CRC64/NVME using PCLMULQDQ when selected by runtime feature detection.
#[cfg(feature = "std")]
#[inline]
pub(crate) fn compute_pclmul_crc64_nvme_runtime(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: callers must gate this with `is_x86_feature_detected!("pclmulqdq")`
  // and `is_x86_feature_detected!("ssse3")`.
  unsafe { compute_pclmul_crc64_unchecked_impl::<Crc64NvmeSpec>(crc, data) }
}
