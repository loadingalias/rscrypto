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

/// Trait for CRC32 polynomial-specific folding constants.
///
/// Uses PMULL_KEY constants (same as ARM) with parallel multiplication.
pub(crate) trait Crc32FoldSpec {
  /// Key for 64-byte folding distance (main loop with 4 accumulators).
  const KEY_64: (u64, u64);
  /// Key for 48-byte folding distance (reduction: x0 → folded).
  const KEY_48: (u64, u64);
  /// Key for 32-byte folding distance (reduction: x1 → folded).
  const KEY_32: (u64, u64);
  /// Key for 16-byte folding distance (reduction: x2 → folded).
  const KEY_16: (u64, u64);

  fn portable(crc: u32, data: &[u8]) -> u32;
}

pub(crate) struct Crc32cSpec;

impl Crc32FoldSpec for Crc32cSpec {
  const KEY_64: (u64, u64) = crate::constants::crc32c::fold::PMULL_KEY_64;
  const KEY_48: (u64, u64) = crate::constants::crc32c::fold::PMULL_KEY_48;
  const KEY_32: (u64, u64) = crate::constants::crc32c::fold::PMULL_KEY_32;
  const KEY_16: (u64, u64) = crate::constants::crc32c::fold::PMULL_KEY_16;

  #[inline]
  fn portable(crc: u32, data: &[u8]) -> u32 {
    crate::crc32c::portable::compute(crc, data)
  }
}

pub(crate) struct Crc32Spec;

impl Crc32FoldSpec for Crc32Spec {
  const KEY_64: (u64, u64) = crate::constants::crc32::fold::PMULL_KEY_64;
  const KEY_48: (u64, u64) = crate::constants::crc32::fold::PMULL_KEY_48;
  const KEY_32: (u64, u64) = crate::constants::crc32::fold::PMULL_KEY_32;
  const KEY_16: (u64, u64) = crate::constants::crc32::fold::PMULL_KEY_16;

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

/// Fold 16 bytes using parallel multiplication (0x00/0x11) for CRC32/CRC32C.
///
/// This matches the ARM PMULL approach:
/// - 0x00: current.lo × coeff.lo
/// - 0x11: current.hi × coeff.hi
///
/// Coefficient layout: coeff.lo = KEY.0, coeff.hi = KEY.1
/// (achieved by loading with `_mm_set_epi64x(KEY.1, KEY.0)`)
#[inline(always)]
pub(crate) unsafe fn fold16_128(current: __m128i, coeff: __m128i, data: __m128i) -> __m128i {
  let lo = _mm_clmulepi64_si128(current, coeff, 0x00);
  let hi = _mm_clmulepi64_si128(current, coeff, 0x11);
  _mm_xor_si128(_mm_xor_si128(lo, hi), data)
}

/// Load a PMULL key as an __m128i with correct lane layout.
///
/// ARM's `vld1q_u64([KEY.0, KEY.1])` puts KEY.0 in lane 0 and KEY.1 in lane 1.
/// x86's `_mm_set_epi64x(a, b)` puts a in lane 1 and b in lane 0.
/// So we need `_mm_set_epi64x(KEY.1, KEY.0)` to match ARM's layout.
#[inline(always)]
unsafe fn load_key(key: (u64, u64)) -> __m128i {
  _mm_set_epi64x(key.1 as i64, key.0 as i64)
}

/// Compute a reflected 32-bit CRC using PCLMULQDQ.
///
/// This implementation uses the same approach as ARM PMULL:
/// - Parallel multiplication (0x00/0x11 selectors)
/// - PMULL_KEY constants
/// - Hardware CRC instruction for finalization (via portable fallback)
///
/// # Safety
/// Caller must ensure the CPU supports `pclmulqdq` and `sse2`.
#[target_feature(enable = "pclmulqdq,sse2")]
unsafe fn compute_pclmul_unchecked_impl<S: Crc32FoldSpec>(crc: u32, data: &[u8]) -> u32 {
  if data.len() < 64 {
    return S::portable(crc, data);
  }

  // Load coefficients with ARM-compatible layout (KEY.0 in lane 0, KEY.1 in lane 1).
  let key_64 = load_key(S::KEY_64);
  let key_48 = load_key(S::KEY_48);
  let key_32 = load_key(S::KEY_32);
  let key_16 = load_key(S::KEY_16);

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

    x0 = fold16_128(x0, key_64, y0);
    x1 = fold16_128(x1, key_64, y1);
    x2 = fold16_128(x2, key_64, y2);
    x3 = fold16_128(x3, key_64, y3);
  }

  // Fold 4 accumulators -> 1.
  let mut folded = x3;
  folded = fold16_128(x2, key_16, folded);
  folded = fold16_128(x1, key_32, folded);
  folded = fold16_128(x0, key_48, folded);

  // Finalize: extract 128 bits and process with portable (which uses hardware CRC if available).
  // This matches ARM's approach of using __crc32cd for finalization.
  let lo = _mm_extract_epi64(folded, 0) as u64;
  let hi = _mm_extract_epi64(folded, 1) as u64;

  // Process the 16 bytes from the folded accumulator, then the remainder.
  let mut final_buf = [0u8; 16];
  final_buf[0..8].copy_from_slice(&lo.to_le_bytes());
  final_buf[8..16].copy_from_slice(&hi.to_le_bytes());

  let crc = S::portable(0, &final_buf);
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
/// For reflected CRC64, following the standard Barrett reduction algorithm:
/// 1. Fold 128 bits to 64 bits using key16 × state.lo
/// 2. Apply Barrett reduction: multiply by mu first, then by poly
#[inline(always)]
pub(crate) unsafe fn finalize_reduced_128_64<S: Crc64FoldSpec>(x: __m128i) -> u64 {
  let (_, key16) = S::FOLD_WIDTH;
  let (poly_simd, mu) = S::BARRETT;

  // Step 1: Fold 128 bits to 64 bits
  // h = key16 × x.lo, then XOR with x >> 64
  let k_vec = _mm_set_epi64x(key16 as i64, 0);
  let h = _mm_clmulepi64_si128(k_vec, x, 0x01); // k_vec.hi × x.lo = key16 × x.lo
  let shifted = _mm_srli_si128(x, 8); // x >> 64
  let folded = _mm_xor_si128(h, shifted);

  // Step 2: Barrett reduction (mu first, then poly - matching ARM implementation)
  // mu_poly = [mu (lo), poly (hi)]
  let mu_poly = _mm_set_epi64x(poly_simd as i64, mu as i64);

  // clmul1 = folded.lo × mu (selector 0x00)
  let clmul1 = _mm_clmulepi64_si128(folded, mu_poly, 0x00);

  // clmul2 = clmul1.lo × poly (selector 0x10: clmul1.lo × mu_poly.hi)
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
