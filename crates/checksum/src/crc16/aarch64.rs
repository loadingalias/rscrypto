//! aarch64 carryless-multiply CRC-16 kernels (PMULL).
//!
//! These kernels implement reflected CRC-16 polynomials by lifting the 16-bit
//! state into the "width32" folding/reduction strategy (same structure as the
//! CRC-32 folding kernels, but with CRC-16-specific constants).
//!
//! # Safety
//!
//! Uses `unsafe` for ARM SIMD intrinsics. Callers must ensure PMULL is
//! available before executing these kernels (the dispatcher does this).
#![allow(unsafe_code)]
#![allow(clippy::indexing_slicing)]
#![allow(unsafe_op_in_unsafe_fn)]

use core::{
  arch::aarch64::*,
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
struct Simd(uint8x16_t);

impl BitXor for Simd {
  type Output = Self;

  #[inline]
  fn bitxor(self, other: Self) -> Self {
    // SAFETY: `veorq_u8` is available with NEON.
    unsafe { Self(veorq_u8(self.0, other.0)) }
  }
}

impl BitXorAssign for Simd {
  #[inline]
  fn bitxor_assign(&mut self, other: Self) {
    *self = *self ^ other;
  }
}

impl Simd {
  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn new(high: u64, low: u64) -> Self {
    Self(vcombine_u8(vcreate_u8(low), vcreate_u8(high)))
  }

  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn load(ptr: *const u8) -> Self {
    Self(vld1q_u8(ptr))
  }

  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn and(self, mask: Self) -> Self {
    Self(vandq_u8(self.0, mask.0))
  }

  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn shift_right_8(self) -> Self {
    Self(vextq_u8(self.0, vdupq_n_u8(0), 8))
  }

  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn shift_left_12(self) -> Self {
    let low_32 = vgetq_lane_u32(vreinterpretq_u32_u8(self.0), 0);
    let result = vsetq_lane_u32(low_32, vdupq_n_u32(0), 3);
    Self(vreinterpretq_u8_u32(result))
  }

  #[inline]
  #[target_feature(enable = "aes")]
  unsafe fn clmul00(self, other: Self) -> Self {
    let a = vgetq_lane_p64(vreinterpretq_p64_u8(self.0), 0);
    let b = vgetq_lane_p64(vreinterpretq_p64_u8(other.0), 0);
    Self(vreinterpretq_u8_p128(vmull_p64(a, b)))
  }

  #[inline]
  #[target_feature(enable = "aes")]
  unsafe fn clmul01(self, other: Self) -> Self {
    let a = vgetq_lane_p64(vreinterpretq_p64_u8(self.0), 1);
    let b = vgetq_lane_p64(vreinterpretq_p64_u8(other.0), 0);
    Self(vreinterpretq_u8_p128(vmull_p64(a, b)))
  }

  #[inline]
  #[target_feature(enable = "aes")]
  unsafe fn clmul10(self, other: Self) -> Self {
    let a = vgetq_lane_p64(vreinterpretq_p64_u8(self.0), 0);
    let b = vgetq_lane_p64(vreinterpretq_p64_u8(other.0), 1);
    Self(vreinterpretq_u8_p128(vmull_p64(a, b)))
  }

  #[inline]
  #[target_feature(enable = "aes")]
  unsafe fn clmul11(self, other: Self) -> Self {
    let a = vgetq_lane_p64(vreinterpretq_p64_u8(self.0), 1);
    let b = vgetq_lane_p64(vreinterpretq_p64_u8(other.0), 1);
    Self(vreinterpretq_u8_p128(vmull_p64(a, b)))
  }

  #[inline]
  #[target_feature(enable = "aes")]
  unsafe fn fold_16_reflected(self, coeff: Self, data_to_xor: Self) -> Self {
    let h = self.clmul10(coeff);
    let l = self.clmul01(coeff);
    h ^ l ^ data_to_xor
  }

  #[inline]
  #[target_feature(enable = "aes", enable = "neon")]
  unsafe fn fold_width32_reflected(self, high: u64, low: u64) -> Self {
    let coeff_low = Self::new(0, low);
    let coeff_high = Self::new(high, 0);

    let clmul = self.clmul00(coeff_low);
    let shifted = self.shift_right_8();
    let mut state = clmul ^ shifted;

    let mask2 = Self::new(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_0000_0000);
    let masked = state.and(mask2);
    let shifted = state.shift_left_12();
    let clmul = shifted.clmul11(coeff_high);
    state = clmul ^ masked;

    state
  }

  #[inline]
  #[target_feature(enable = "aes", enable = "neon")]
  unsafe fn barrett_width32_reflected(self, poly: u64, mu: u64) -> u32 {
    let polymu = Self::new(poly, mu);
    let clmul1 = self.clmul00(polymu);
    let clmul2 = clmul1.clmul10(polymu);
    let xorred = self ^ clmul2;

    let hi = xorred.shift_right_8();
    vgetq_lane_u32(vreinterpretq_u32_u8(hi.0), 0)
  }
}

#[inline]
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn finalize_lanes_width32_reflected(x: [Simd; 8], keys: &[u64; 23]) -> u32 {
  let mut res = x[7];
  res = x[0].fold_16_reflected(Simd::new(keys[10], keys[9]), res);
  res = x[1].fold_16_reflected(Simd::new(keys[12], keys[11]), res);
  res = x[2].fold_16_reflected(Simd::new(keys[14], keys[13]), res);
  res = x[3].fold_16_reflected(Simd::new(keys[16], keys[15]), res);
  res = x[4].fold_16_reflected(Simd::new(keys[18], keys[17]), res);
  res = x[5].fold_16_reflected(Simd::new(keys[20], keys[19]), res);
  res = x[6].fold_16_reflected(Simd::new(keys[2], keys[1]), res);

  res = res.fold_width32_reflected(keys[6], keys[5]);
  res.barrett_width32_reflected(keys[8], keys[7])
}

#[inline]
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn update_simd_width32_reflected(state: u32, first: &[Simd; 8], rest: &[[Simd; 8]], keys: &[u64; 23]) -> u32 {
  let mut x = *first;

  x[0] ^= Simd::new(0, state as u64);

  let coeff_128b = Simd::new(keys[4], keys[3]);
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
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn crc16_width32_pmull_small(
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

  let coeff_16b = Simd::new(keys[2], keys[1]);

  let mut x0 = Simd::load(buf);
  x0 ^= Simd::new(0, state as u64);
  buf = buf.add(16);
  len = len.strict_sub(16);

  while len >= 16 {
    let chunk = Simd::load(buf);
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
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn crc16_width32_pmull(mut state: u16, data: &[u8], keys: &[u64; 23], portable: fn(u16, &[u8]) -> u16) -> u16 {
  let (left, middle, right) = data.align_to::<[Simd; 8]>();
  let Some((first, rest)) = middle.split_first() else {
    return crc16_width32_pmull_small(state, data, keys, portable);
  };

  state = portable(state, left);
  let state32 = update_simd_width32_reflected(state as u32, first, rest, keys);
  state = state32 as u16;
  portable(state, right)
}

#[inline]
pub fn crc16_ccitt_pmull_safe(crc: u16, data: &[u8], portable: fn(u16, &[u8]) -> u16) -> u16 {
  // SAFETY: Dispatcher verifies PMULL before selecting this kernel.
  unsafe { crc16_width32_pmull(crc, data, &KEYS_CRC16_1021_REFLECTED, portable) }
}

#[inline]
pub fn crc16_ibm_pmull_safe(crc: u16, data: &[u8], portable: fn(u16, &[u8]) -> u16) -> u16 {
  // SAFETY: Dispatcher verifies PMULL before selecting this kernel.
  unsafe { crc16_width32_pmull(crc, data, &KEYS_CRC16_8005_REFLECTED, portable) }
}
