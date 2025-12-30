//! x86_64 carryless-multiply CRC-16 kernels (PCLMULQDQ).
//!
//! These kernels implement reflected CRC-16 polynomials by lifting the 16-bit
//! state into the "width32" folding/reduction strategy (same structure as the
//! CRC-32 PCLMUL kernels, but with CRC-16-specific constants).
//!
//! # Safety
//!
//! Uses `unsafe` for x86 SIMD intrinsics. Callers must ensure SSSE3 + PCLMULQDQ
//! are available before executing these kernels (the dispatcher does this).
#![allow(unsafe_code)]
#![allow(clippy::indexing_slicing)]
#![allow(unsafe_op_in_unsafe_fn)]

use core::{
  arch::x86_64::*,
  ops::{BitXor, BitXorAssign},
};

#[rustfmt::skip]
const KEYS_CRC16_1021_REFLECTED: [u64; 23] = [
  0x0000000000000000,
  0x00000000000189ae,
  0x0000000000008e10,
  0x00000000000160be,
  0x000000000001bed8,
  0x00000000000189ae,
  0x00000000000114aa,
  0x000000011c581911,
  0x0000000000010811,
  0x000000000001ce5e,
  0x000000000001c584,
  0x000000000001db50,
  0x000000000000b8f2,
  0x0000000000000842,
  0x000000000000b072,
  0x0000000000014ff2,
  0x0000000000019a3c,
  0x0000000000000e3a,
  0x0000000000004d7a,
  0x0000000000005b44,
  0x0000000000007762,
  0x0000000000019208,
  0x0000000000002df8,
];

#[rustfmt::skip]
const KEYS_CRC16_8005_REFLECTED: [u64; 23] = [
  0x0000000000000000,
  0x0000000000018cc2,
  0x000000000001d0c2,
  0x0000000000014cc2,
  0x000000000001dc02,
  0x0000000000018cc2,
  0x000000000001bc02,
  0x00000001cfffbfff,
  0x0000000000014003,
  0x000000000000bcac,
  0x000000000001a674,
  0x000000000001ac00,
  0x0000000000019b6e,
  0x000000000001d33e,
  0x000000000001c462,
  0x000000000000bffa,
  0x000000000001b0c2,
  0x00000000000186ae,
  0x000000000001ad6e,
  0x000000000001d55e,
  0x000000000001ec02,
  0x000000000001d99e,
  0x000000000001bcc2,
];

#[repr(transparent)]
#[derive(Copy, Clone)]
struct Simd128(__m128i);

impl BitXor for Simd128 {
  type Output = Self;

  #[inline]
  fn bitxor(self, other: Self) -> Self {
    // SAFETY: `_mm_xor_si128` is available on all x86_64 (SSE2 baseline).
    unsafe { Self(_mm_xor_si128(self.0, other.0)) }
  }
}

impl BitXorAssign for Simd128 {
  #[inline]
  fn bitxor_assign(&mut self, other: Self) {
    *self = *self ^ other;
  }
}

impl Simd128 {
  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn new(high: u64, low: u64) -> Self {
    Self(_mm_set_epi64x(high as i64, low as i64))
  }

  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn shift_right_8(self) -> Self {
    Self(_mm_srli_si128::<8>(self.0))
  }

  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn shift_left_12(self) -> Self {
    Self(_mm_slli_si128::<12>(self.0))
  }

  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn and(self, mask: Self) -> Self {
    Self(_mm_and_si128(self.0, mask.0))
  }

  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn fold_16_reflected(self, coeff: Self, data_to_xor: Self) -> Self {
    let h = _mm_clmulepi64_si128::<0x10>(self.0, coeff.0);
    let l = _mm_clmulepi64_si128::<0x01>(self.0, coeff.0);
    Self(_mm_xor_si128(_mm_xor_si128(h, l), data_to_xor.0))
  }

  /// Fold 16 bytes down to the "width32" reduction state (reflected mode).
  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn fold_width32_reflected(self, high: u64, low: u64) -> Self {
    let coeff_low = Self::new(0, low);
    let coeff_high = Self::new(high, 0);

    // 16B -> 8B
    let clmul = _mm_clmulepi64_si128::<0x00>(self.0, coeff_low.0);
    let shifted = self.shift_right_8();
    let mut state = Self(_mm_xor_si128(clmul, shifted.0));

    // 8B -> 4B
    let mask2 = Self::new(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_0000_0000);
    let masked = state.and(mask2);
    let shifted = state.shift_left_12();
    let clmul = _mm_clmulepi64_si128::<0x11>(shifted.0, coeff_high.0);
    state = Self(_mm_xor_si128(clmul, masked.0));

    state
  }

  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn barrett_width32_reflected(self, poly: u64, mu: u64) -> u32 {
    let polymu = Self::new(poly, mu);
    let clmul1 = _mm_clmulepi64_si128::<0x00>(self.0, polymu.0);
    let clmul2 = _mm_clmulepi64_si128::<0x10>(clmul1, polymu.0);
    let xorred = _mm_xor_si128(self.0, clmul2);

    let hi = _mm_srli_si128::<8>(xorred);
    _mm_cvtsi128_si64(hi) as u32
  }
}

#[inline]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn finalize_lanes_width32_reflected(x: [Simd128; 8], keys: &[u64; 23]) -> u32 {
  let mut res = x[7];
  res = x[0].fold_16_reflected(Simd128::new(keys[10], keys[9]), res);
  res = x[1].fold_16_reflected(Simd128::new(keys[12], keys[11]), res);
  res = x[2].fold_16_reflected(Simd128::new(keys[14], keys[13]), res);
  res = x[3].fold_16_reflected(Simd128::new(keys[16], keys[15]), res);
  res = x[4].fold_16_reflected(Simd128::new(keys[18], keys[17]), res);
  res = x[5].fold_16_reflected(Simd128::new(keys[20], keys[19]), res);
  res = x[6].fold_16_reflected(Simd128::new(keys[2], keys[1]), res);

  res = res.fold_width32_reflected(keys[6], keys[5]);
  res.barrett_width32_reflected(keys[8], keys[7])
}

#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn update_simd_width32_reflected(
  state: u32,
  first: &[Simd128; 8],
  rest: &[[Simd128; 8]],
  keys: &[u64; 23],
) -> u32 {
  let mut x = *first;

  x[0] ^= Simd128::new(0, state as u64);

  let coeff_128b = Simd128::new(keys[4], keys[3]);
  for chunk in rest {
    x[0] = x[0].fold_16_reflected(coeff_128b, chunk[0]);
    x[1] = x[1].fold_16_reflected(coeff_128b, chunk[1]);
    x[2] = x[2].fold_16_reflected(coeff_128b, chunk[2]);
    x[3] = x[3].fold_16_reflected(coeff_128b, chunk[3]);
    x[4] = x[4].fold_16_reflected(coeff_128b, chunk[4]);
    x[5] = x[5].fold_16_reflected(coeff_128b, chunk[5]);
    x[6] = x[6].fold_16_reflected(coeff_128b, chunk[6]);
    x[7] = x[7].fold_16_reflected(coeff_128b, chunk[7]);
  }

  finalize_lanes_width32_reflected(x, keys)
}

#[inline]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn crc16_width32_pclmul_small(
  mut state: u16,
  data: &[u8],
  keys: &[u64; 23],
  portable: fn(u16, &[u8]) -> u16,
) -> u16 {
  let mut buf = data.as_ptr();
  let mut len = data.len();

  if len < 16 {
    return portable(state, data);
  }

  let coeff_16b = Simd128::new(keys[2], keys[1]);

  let mut x0 = Simd128(_mm_loadu_si128(buf as *const __m128i));
  x0 ^= Simd128::new(0, state as u64);
  buf = buf.add(16);
  len = len.strict_sub(16);

  while len >= 16 {
    let chunk = Simd128(_mm_loadu_si128(buf as *const __m128i));
    x0 = x0.fold_16_reflected(coeff_16b, chunk);
    buf = buf.add(16);
    len = len.strict_sub(16);
  }

  let x0 = x0.fold_width32_reflected(keys[6], keys[5]);
  state = x0.barrett_width32_reflected(keys[8], keys[7]) as u16;

  let tail = core::slice::from_raw_parts(buf, len);
  portable(state, tail)
}

#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn crc16_width32_pclmul(mut state: u16, data: &[u8], keys: &[u64; 23], portable: fn(u16, &[u8]) -> u16) -> u16 {
  let (left, middle, right) = data.align_to::<[Simd128; 8]>();
  let Some((first, rest)) = middle.split_first() else {
    return crc16_width32_pclmul_small(state, data, keys, portable);
  };

  state = portable(state, left);
  let state32 = update_simd_width32_reflected(state as u32, first, rest, keys);
  state = state32 as u16;
  portable(state, right)
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX-512 VPCLMULQDQ Tier
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq")]
unsafe fn clmul10_vpclmul(a: __m512i, b: __m512i) -> __m512i {
  _mm512_clmulepi64_epi128(a, b, 0x10)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq")]
unsafe fn clmul01_vpclmul(a: __m512i, b: __m512i) -> __m512i {
  _mm512_clmulepi64_epi128(a, b, 0x01)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq")]
unsafe fn fold_16_reflected_vpclmul(state: __m512i, coeff: __m512i, data: __m512i) -> __m512i {
  _mm512_ternarylogic_epi64(clmul10_vpclmul(state, coeff), clmul01_vpclmul(state, coeff), data, 0x96)
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn broadcast_coeff_128b(high: u64, low: u64) -> __m512i {
  _mm512_set_epi64(
    high as i64,
    low as i64,
    high as i64,
    low as i64,
    high as i64,
    low as i64,
    high as i64,
    low as i64,
  )
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn state_mask_lane0(state: u32) -> __m512i {
  _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, state as i64)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn update_simd_width32_reflected_vpclmul(
  state: u32,
  first: &[Simd128; 8],
  rest: &[[Simd128; 8]],
  keys: &[u64; 23],
) -> u32 {
  let ptr = first.as_ptr() as *const u8;
  let mut x0 = _mm512_loadu_si512(ptr as *const __m512i);
  let mut x1 = _mm512_loadu_si512(ptr.add(64) as *const __m512i);

  x0 = _mm512_xor_si512(x0, state_mask_lane0(state));

  let coeff_128b = broadcast_coeff_128b(keys[4], keys[3]);
  for chunk in rest {
    let ptr = chunk.as_ptr() as *const u8;
    let y0 = _mm512_loadu_si512(ptr as *const __m512i);
    let y1 = _mm512_loadu_si512(ptr.add(64) as *const __m512i);
    x0 = fold_16_reflected_vpclmul(x0, coeff_128b, y0);
    x1 = fold_16_reflected_vpclmul(x1, coeff_128b, y1);
  }

  let mut lanes0 = [Simd128(_mm_setzero_si128()); 4];
  let mut lanes1 = [Simd128(_mm_setzero_si128()); 4];
  _mm512_storeu_si512(lanes0.as_mut_ptr() as *mut __m512i, x0);
  _mm512_storeu_si512(lanes1.as_mut_ptr() as *mut __m512i, x1);

  let x = [
    lanes0[0], lanes0[1], lanes0[2], lanes0[3], lanes1[0], lanes1[1], lanes1[2], lanes1[3],
  ];

  finalize_lanes_width32_reflected(x, keys)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn crc16_width32_vpclmul(mut state: u16, data: &[u8], keys: &[u64; 23], portable: fn(u16, &[u8]) -> u16) -> u16 {
  let (left, middle, right) = data.align_to::<[Simd128; 8]>();
  let Some((first, rest)) = middle.split_first() else {
    return crc16_width32_pclmul_small(state, data, keys, portable);
  };

  state = portable(state, left);
  let state32 = update_simd_width32_reflected_vpclmul(state as u32, first, rest, keys);
  state = state32 as u16;
  portable(state, right)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public Safe Kernels (matching CRC-64 pure fn(u16, &[u8]) -> u16 signature)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-16/CCITT PCLMULQDQ kernel.
///
/// # Safety
///
/// Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
#[inline]
pub fn crc16_ccitt_pclmul_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
  unsafe {
    crc16_width32_pclmul(
      crc,
      data,
      &KEYS_CRC16_1021_REFLECTED,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/CCITT VPCLMULQDQ kernel (AVX-512).
///
/// # Safety
///
/// Dispatcher verifies VPCLMULQDQ + AVX-512 before selecting this kernel.
#[inline]
pub fn crc16_ccitt_vpclmul_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies VPCLMULQDQ + AVX-512 before selecting this kernel.
  unsafe {
    crc16_width32_vpclmul(
      crc,
      data,
      &KEYS_CRC16_1021_REFLECTED,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/IBM PCLMULQDQ kernel.
///
/// # Safety
///
/// Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
#[inline]
pub fn crc16_ibm_pclmul_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
  unsafe { crc16_width32_pclmul(crc, data, &KEYS_CRC16_8005_REFLECTED, super::portable::crc16_ibm_slice8) }
}

/// CRC-16/IBM VPCLMULQDQ kernel (AVX-512).
///
/// # Safety
///
/// Dispatcher verifies VPCLMULQDQ + AVX-512 before selecting this kernel.
#[inline]
pub fn crc16_ibm_vpclmul_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies VPCLMULQDQ + AVX-512 before selecting this kernel.
  unsafe { crc16_width32_vpclmul(crc, data, &KEYS_CRC16_8005_REFLECTED, super::portable::crc16_ibm_slice8) }
}
