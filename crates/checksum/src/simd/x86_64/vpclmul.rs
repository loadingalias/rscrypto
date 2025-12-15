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
  #[allow(clippy::cast_ptr_alignment)]
  _mm512_loadu_si512(ptr as *const __m512i)
}

#[inline(always)]
unsafe fn load_reflected_block(ptr: *const u8, smask: __m512i) -> __m512i {
  // SAFETY: caller ensures `ptr` is valid for 64 bytes.
  _mm512_shuffle_epi8(load_block(ptr), smask)
}

#[inline(always)]
unsafe fn store_lanes(x: __m512i) -> [__m128i; 4] {
  let mut lanes = [_mm_setzero_si128(); 4];
  // SAFETY: `[__m128i; 4]` is 64 bytes and is suitable for an unaligned store.
  #[allow(clippy::cast_ptr_alignment)]
  _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, x);
  lanes
}

#[inline(always)]
unsafe fn fold64(x: __m512i, coeff_64: __m512i, data: __m512i) -> __m512i {
  let h = _mm512_clmulepi64_epi128(x, coeff_64, 0x10);
  let l = _mm512_clmulepi64_epi128(x, coeff_64, 0x01);
  _mm512_xor_si512(_mm512_xor_si512(h, l), data)
}

/// Compute a reflected 32-bit CRC using VPCLMULQDQ.
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

  // Byte-reflection shuffle mask (mirrors the PCLMUL engine).
  let smask_128 = _mm_set_epi64x(0x0809_0a0b_0c0d_0e0f_u64 as i64, 0x0001_0203_0405_0607_u64 as i64);
  let smask = _mm512_broadcast_i64x2(smask_128);

  let coeff_64 = {
    let (high, low) = S::COEFF_64;
    _mm_set_epi64x(high as i64, low as i64)
  };
  let coeff_64 = _mm512_broadcast_i64x2(coeff_64);

  let base = bulk.as_ptr();
  let mut x = load_reflected_block(base, smask);
  let crc_vec = _mm512_zextsi128_si512(_mm_cvtsi32_si128(crc as i32));
  x = _mm512_xor_si512(x, crc_vec);

  let mut offset = 64usize;
  while offset + 256 <= bulk_len {
    let y0 = load_reflected_block(base.add(offset), smask);
    let y1 = load_reflected_block(base.add(offset + 64), smask);
    let y2 = load_reflected_block(base.add(offset + 128), smask);
    let y3 = load_reflected_block(base.add(offset + 192), smask);

    x = fold64(x, coeff_64, y0);
    x = fold64(x, coeff_64, y1);
    x = fold64(x, coeff_64, y2);
    x = fold64(x, coeff_64, y3);

    offset += 256;
  }

  while offset < bulk_len {
    let y = load_reflected_block(base.add(offset), smask);
    x = fold64(x, coeff_64, y);
    offset += 64;
  }

  let lanes = store_lanes(x);

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

  // Reduce 4×128 -> 1×128.
  let mut folded = lanes[3];
  folded = crate::simd::x86_64::pclmul::fold16_128(lanes[2], coeff_16, folded);
  folded = crate::simd::x86_64::pclmul::fold16_128(lanes[1], coeff_32, folded);
  folded = crate::simd::x86_64::pclmul::fold16_128(lanes[0], coeff_48, folded);

  let crc = crate::simd::x86_64::pclmul::finalize_reduced_128::<S>(folded);
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

// ============================================================================
// CRC64 Support
// ============================================================================

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

  // Byte-reflection shuffle mask.
  let smask_128 = _mm_set_epi64x(0x0809_0a0b_0c0d_0e0f_u64 as i64, 0x0001_0203_0405_0607_u64 as i64);
  let smask = _mm512_broadcast_i64x2(smask_128);

  // CRC64 coefficients loaded the same as CRC32 (no swap).
  let coeff_64 = {
    let (high, low) = S::COEFF_64;
    _mm_set_epi64x(high as i64, low as i64)
  };
  let coeff_64 = _mm512_broadcast_i64x2(coeff_64);

  let base = bulk.as_ptr();
  let mut x = load_reflected_block(base, smask);
  // XOR initial CRC into low 64 bits
  let crc_vec = _mm512_zextsi128_si512(_mm_set_epi64x(0, crc as i64));
  x = _mm512_xor_si512(x, crc_vec);

  let mut offset = 64usize;
  while offset + 256 <= bulk_len {
    let y0 = load_reflected_block(base.add(offset), smask);
    let y1 = load_reflected_block(base.add(offset + 64), smask);
    let y2 = load_reflected_block(base.add(offset + 128), smask);
    let y3 = load_reflected_block(base.add(offset + 192), smask);

    x = fold64(x, coeff_64, y0);
    x = fold64(x, coeff_64, y1);
    x = fold64(x, coeff_64, y2);
    x = fold64(x, coeff_64, y3);

    offset += 256;
  }

  while offset < bulk_len {
    let y = load_reflected_block(base.add(offset), smask);
    x = fold64(x, coeff_64, y);
    offset += 64;
  }

  let lanes = store_lanes(x);

  // CRC64 coefficients loaded the same as CRC32 (no swap).
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

  // Reduce 4×128 -> 1×128.
  let mut folded = lanes[3];
  folded = crate::simd::x86_64::pclmul::fold16_128(lanes[2], coeff_16, folded);
  folded = crate::simd::x86_64::pclmul::fold16_128(lanes[1], coeff_32, folded);
  folded = crate::simd::x86_64::pclmul::fold16_128(lanes[0], coeff_48, folded);

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
