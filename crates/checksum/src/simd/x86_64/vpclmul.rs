//! VPCLMULQDQ (AVX-512) CRC folding engine.
//!
//! This implementation uses AVX-512 + VPCLMULQDQ to fold 64-byte blocks using
//! four 128-bit lanes in parallel (packed into one 512-bit register), then
//! reduces to the final CRC width using the PCLMUL finalize path.
//!
//! Supports both 32-bit (CRC32, CRC32C) and 64-bit (CRC64) variants.

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

use crate::simd::x86_64::pclmul::{Crc32FoldSpec, Crc64FoldSpec};

#[inline(always)]
unsafe fn load_block(ptr: *const u8) -> __m512i {
  // SAFETY: caller ensures `ptr` is valid for 64 bytes.
  // For reflected CRCs on little-endian, load data as-is without byte reflection.
  #[allow(clippy::cast_ptr_alignment)]
  _mm512_loadu_si512(ptr as *const __m512i)
}

#[inline(always)]
unsafe fn store_lanes(x: __m512i) -> [__m128i; 4] {
  let mut lanes = [_mm_setzero_si128(); 4];
  // SAFETY: `[__m128i; 4]` is 64 bytes and is suitable for an unaligned store.
  #[allow(clippy::cast_ptr_alignment)]
  _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, x);
  lanes
}

/// Fold 64 bytes using parallel multiplication (0x00/0x11).
///
/// Used for both CRC32/CRC32C and CRC64, matching ARM PMULL approach.
#[inline(always)]
unsafe fn fold64(x: __m512i, coeff: __m512i, data: __m512i) -> __m512i {
  let lo = _mm512_clmulepi64_epi128(x, coeff, 0x00);
  let hi = _mm512_clmulepi64_epi128(x, coeff, 0x11);
  _mm512_xor_si512(_mm512_xor_si512(lo, hi), data)
}

/// Load a PMULL key as an __m128i with correct lane layout.
#[inline(always)]
unsafe fn load_key_128(key: (u64, u64)) -> __m128i {
  _mm_set_epi64x(key.1 as i64, key.0 as i64)
}

/// Compute a reflected 32-bit CRC using VPCLMULQDQ.
///
/// Uses the same parallel multiplication approach as ARM PMULL.
///
/// # Safety
/// Caller must ensure the CPU supports the required AVX-512/VPCLMUL features.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,vpclmulqdq,pclmulqdq")]
unsafe fn compute_vpclmul_unchecked_impl<S: Crc32FoldSpec>(crc: u32, data: &[u8]) -> u32 {
  if data.len() < 64 {
    return S::portable(crc, data);
  }

  let bulk_len = data.len() & !63;
  let (bulk, rem) = data.split_at(bulk_len);

  // Load key with ARM-compatible layout and broadcast to 512 bits.
  let key_64 = _mm512_broadcast_i64x2(load_key_128(S::KEY_64));

  let base = bulk.as_ptr();
  // Load data WITHOUT byte reflection (correct for reflected CRCs on little-endian).
  let mut x = load_block(base);
  let crc_vec = _mm512_zextsi128_si512(_mm_cvtsi32_si128(crc as i32));
  x = _mm512_xor_si512(x, crc_vec);

  let mut offset = 64usize;
  while offset + 256 <= bulk_len {
    let y0 = load_block(base.add(offset));
    let y1 = load_block(base.add(offset + 64));
    let y2 = load_block(base.add(offset + 128));
    let y3 = load_block(base.add(offset + 192));

    x = fold64(x, key_64, y0);
    x = fold64(x, key_64, y1);
    x = fold64(x, key_64, y2);
    x = fold64(x, key_64, y3);

    offset += 256;
  }

  while offset < bulk_len {
    let y = load_block(base.add(offset));
    x = fold64(x, key_64, y);
    offset += 64;
  }

  let lanes = store_lanes(x);

  // Load reduction keys with ARM-compatible layout.
  let key_48 = load_key_128(S::KEY_48);
  let key_32 = load_key_128(S::KEY_32);
  let key_16 = load_key_128(S::KEY_16);

  // Reduce 4×128 -> 1×128 using parallel multiplication.
  let mut folded = lanes[3];
  folded = crate::simd::x86_64::pclmul::fold16_128(lanes[2], key_16, folded);
  folded = crate::simd::x86_64::pclmul::fold16_128(lanes[1], key_32, folded);
  folded = crate::simd::x86_64::pclmul::fold16_128(lanes[0], key_48, folded);

  // Finalize using portable (which uses hardware CRC if available).
  let lo = _mm_extract_epi64(folded, 0) as u64;
  let hi = _mm_extract_epi64(folded, 1) as u64;

  let mut final_buf = [0u8; 16];
  final_buf[0..8].copy_from_slice(&lo.to_le_bytes());
  final_buf[8..16].copy_from_slice(&hi.to_le_bytes());

  let crc = S::portable(0, &final_buf);
  S::portable(crc, rem)
}

#[cfg(all(
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512bw",
  target_feature = "vpclmulqdq",
  target_feature = "pclmulqdq"
))]
#[inline]
pub(crate) fn compute_vpclmul_enabled(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: this function is only compiled when the required target features are enabled.
  unsafe { compute_vpclmul_unchecked_impl::<crate::simd::x86_64::pclmul::Crc32cSpec>(crc, data) }
}

/// Runtime-dispatched entrypoint used when `vpclmulqdq` is available.
#[cfg(feature = "std")]
#[inline]
pub(crate) fn compute_vpclmul_runtime(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: callers must gate with `is_x86_feature_detected!` for the required features.
  unsafe { compute_vpclmul_unchecked_impl::<crate::simd::x86_64::pclmul::Crc32cSpec>(crc, data) }
}

#[cfg(all(
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512bw",
  target_feature = "vpclmulqdq",
  target_feature = "pclmulqdq"
))]
#[inline]
pub(crate) fn compute_vpclmul_crc32_enabled(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: this function is only compiled when the required target features are enabled.
  unsafe { compute_vpclmul_unchecked_impl::<crate::simd::x86_64::pclmul::Crc32Spec>(crc, data) }
}

/// Runtime-dispatched entrypoint used when `vpclmulqdq` is available.
#[cfg(feature = "std")]
#[inline]
pub(crate) fn compute_vpclmul_crc32_runtime(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: callers must gate with `is_x86_feature_detected!` for the required features.
  unsafe { compute_vpclmul_unchecked_impl::<crate::simd::x86_64::pclmul::Crc32Spec>(crc, data) }
}

// CRC64 Support

/// Compute a reflected 64-bit CRC using VPCLMULQDQ.
///
/// # Safety
/// Caller must ensure the CPU supports the required AVX-512/VPCLMUL features.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,vpclmulqdq,pclmulqdq")]
unsafe fn compute_vpclmul_crc64_unchecked_impl<S: Crc64FoldSpec>(crc: u64, data: &[u8]) -> u64 {
  if data.len() < 64 {
    return S::portable(crc, data);
  }

  let bulk_len = data.len() & !63;
  let (bulk, rem) = data.split_at(bulk_len);

  // CRC64 uses parallel multiplication (0x00/0x11).
  // To match ARM's coefficient layout (high in lane 0, low in lane 1),
  // we swap: _mm_set_epi64x(low, high) since it puts first arg in HIGH lane.
  let coeff_64 = {
    let (high, low) = S::COEFF_64;
    _mm_set_epi64x(low as i64, high as i64)
  };
  let coeff_64 = _mm512_broadcast_i64x2(coeff_64);

  let base = bulk.as_ptr();
  // Load data WITHOUT byte reflection (correct for reflected CRCs on little-endian).
  let mut x = load_block(base);
  // XOR initial CRC into low 64 bits
  let crc_vec = _mm512_zextsi128_si512(_mm_set_epi64x(0, crc as i64));
  x = _mm512_xor_si512(x, crc_vec);

  let mut offset = 64usize;
  while offset + 256 <= bulk_len {
    let y0 = load_block(base.add(offset));
    let y1 = load_block(base.add(offset + 64));
    let y2 = load_block(base.add(offset + 128));
    let y3 = load_block(base.add(offset + 192));

    x = fold64(x, coeff_64, y0);
    x = fold64(x, coeff_64, y1);
    x = fold64(x, coeff_64, y2);
    x = fold64(x, coeff_64, y3);

    offset += 256;
  }

  while offset < bulk_len {
    let y = load_block(base.add(offset));
    x = fold64(x, coeff_64, y);
    offset += 64;
  }

  let lanes = store_lanes(x);

  // CRC64 coefficients with swapped loading (to match ARM).
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

  // Reduce 4×128 -> 1×128 using parallel multiplication (matching PCLMUL CRC64).
  let mut folded = lanes[3];
  folded = crate::simd::x86_64::pclmul::fold16_128_parallel(lanes[2], coeff_16, folded);
  folded = crate::simd::x86_64::pclmul::fold16_128_parallel(lanes[1], coeff_32, folded);
  folded = crate::simd::x86_64::pclmul::fold16_128_parallel(lanes[0], coeff_48, folded);

  let crc = crate::simd::x86_64::pclmul::finalize_reduced_128_64::<S>(folded);
  S::portable(crc, rem)
}

/// Compute CRC64/XZ using VPCLMULQDQ when enabled at compile time.
#[cfg(all(
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512bw",
  target_feature = "vpclmulqdq",
  target_feature = "pclmulqdq"
))]
#[inline]
pub(crate) fn compute_vpclmul_crc64_xz_enabled(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: this function is only compiled when the required target features are enabled.
  unsafe { compute_vpclmul_crc64_unchecked_impl::<crate::simd::x86_64::pclmul::Crc64XzSpec>(crc, data) }
}

/// Runtime-dispatched entrypoint for CRC64/XZ.
#[cfg(feature = "std")]
#[inline]
pub(crate) fn compute_vpclmul_crc64_xz_runtime(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: callers must gate with `is_x86_feature_detected!` for the required features.
  unsafe { compute_vpclmul_crc64_unchecked_impl::<crate::simd::x86_64::pclmul::Crc64XzSpec>(crc, data) }
}

/// Compute CRC64/NVME using VPCLMULQDQ when enabled at compile time.
#[cfg(all(
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512bw",
  target_feature = "vpclmulqdq",
  target_feature = "pclmulqdq"
))]
#[inline]
pub(crate) fn compute_vpclmul_crc64_nvme_enabled(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: this function is only compiled when the required target features are enabled.
  unsafe { compute_vpclmul_crc64_unchecked_impl::<crate::simd::x86_64::pclmul::Crc64NvmeSpec>(crc, data) }
}

/// Runtime-dispatched entrypoint for CRC64/NVME.
#[cfg(feature = "std")]
#[inline]
pub(crate) fn compute_vpclmul_crc64_nvme_runtime(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: callers must gate with `is_x86_feature_detected!` for the required features.
  unsafe { compute_vpclmul_crc64_unchecked_impl::<crate::simd::x86_64::pclmul::Crc64NvmeSpec>(crc, data) }
}
