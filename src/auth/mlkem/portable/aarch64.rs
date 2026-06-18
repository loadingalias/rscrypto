#![allow(clippy::indexing_slicing)] // Fixed-size ML-KEM native tables and assembly ABI.
#![allow(unsafe_code)]

use core::arch::{
  aarch64::{
    int16x8_t, vaddq_s16, vandq_s16, vdupq_n_s16, vld1q_s16, vreinterpretq_u16_s16, vshrq_n_s16, vst1q_u16, vsubq_s16,
  },
  global_asm,
};

use super::{N, Poly, Q_I16, Q_I32};

const Q_TIMES_4_I32: i32 = Q_I32 * 4;

#[cfg(target_os = "macos")]
global_asm!(include_str!("../asm/rscrypto_mlkem_ntt_aarch64_apple_darwin.s"));
#[cfg(target_os = "linux")]
global_asm!(include_str!("../asm/rscrypto_mlkem_ntt_aarch64_linux.s"));

#[repr(align(16))]
struct AlignedI16<const N: usize>([i16; N]);

static NTT_ZETAS_LAYER_12345: AlignedI16<80> = AlignedI16([
  -1600, -15749, -749, -7373, -40, -394, -687, -6762, 630, 6201, -1432, -14095, 848, 8347, 0, 0, 1062, 10453, 296,
  2914, -882, -8682, 0, 0, -1410, -13879, 1339, 13180, 1476, 14529, 0, 0, 193, 1900, -283, -2786, 56, 551, 0, 0, 797,
  7845, -1089, -10719, 1333, 13121, 0, 0, -543, -5345, 1426, 14036, -1235, -12156, 0, 0, -69, -679, 535, 5266, -447,
  -4400, 0, 0, 569, 5601, -936, -9213, -450, -4429, 0, 0, -1583, -15582, -1355, -13338, 821, 8081, 0, 0,
]);

static NTT_ZETAS_LAYER_67: AlignedI16<384> = AlignedI16([
  289, 289, 331, 331, -76, -76, -1573, -1573, 2845, 2845, 3258, 3258, -748, -748, -15483, -15483, 17, 17, 583, 583,
  1637, 1637, -1041, -1041, 167, 167, 5739, 5739, 16113, 16113, -10247, -10247, -568, -568, -680, -680, 723, 723, 1100,
  1100, -5591, -5591, -6693, -6693, 7117, 7117, 10828, 10828, 1197, 1197, -1025, -1025, -1052, -1052, -1274, -1274,
  11782, 11782, -10089, -10089, -10355, -10355, -12540, -12540, 1409, 1409, -48, -48, 756, 756, -314, -314, 13869,
  13869, -472, -472, 7441, 7441, -3091, -3091, -667, -667, 233, 233, -1173, -1173, -279, -279, -6565, -6565, 2293,
  2293, -11546, -11546, -2746, -2746, 650, 650, -1352, -1352, -816, -816, 632, 632, 6398, 6398, -13308, -13308, -8032,
  -8032, 6221, 6221, -1626, -1626, -540, -540, -1482, -1482, 1461, 1461, -16005, -16005, -5315, -5315, -14588, -14588,
  14381, 14381, 1651, 1651, -1540, -1540, 952, 952, -642, -642, 16251, 16251, -15159, -15159, 9371, 9371, -6319, -6319,
  -464, -464, 33, 33, 1320, 1320, -1414, -1414, -4567, -4567, 325, 325, 12993, 12993, -13918, -13918, 939, 939, -892,
  -892, 733, 733, 268, 268, 9243, 9243, -8780, -8780, 7215, 7215, 2638, 2638, -1021, -1021, -941, -941, -992, -992,
  641, 641, -10050, -10050, -9262, -9262, -9764, -9764, 6309, 6309, -1010, -1010, 1435, 1435, 807, 807, 452, 452,
  -9942, -9942, 14125, 14125, 7943, 7943, 4449, 4449, 1584, 1584, -1292, -1292, 375, 375, -1239, -1239, 15592, 15592,
  -12717, -12717, 3691, 3691, -12196, -12196, -1031, -1031, -109, -109, -780, -780, 1645, 1645, -10148, -10148, -1073,
  -1073, -7678, -7678, 16192, 16192, 1438, 1438, -461, -461, 1534, 1534, -927, -927, 14155, 14155, -4538, -4538, 15099,
  15099, -9125, -9125, 1063, 1063, -556, -556, -1230, -1230, -863, -863, 10463, 10463, -5473, -5473, -12107, -12107,
  -8495, -8495, 319, 319, 757, 757, 561, 561, -735, -735, 3140, 3140, 7451, 7451, 5522, 5522, -7235, -7235, -682, -682,
  -712, -712, 1481, 1481, 648, 648, -6713, -6713, -7008, -7008, 14578, 14578, 6378, 6378, -525, -525, 403, 403, 1143,
  1143, -554, -554, -5168, -5168, 3967, 3967, 11251, 11251, -5453, -5453, 1092, 1092, 1026, 1026, -1179, -1179, 886,
  886, 10749, 10749, 10099, 10099, -11605, -11605, 8721, 8721, -855, -855, -219, -219, 1227, 1227, 910, 910, -8416,
  -8416, -2156, -2156, 12078, 12078, 8957, 8957, -1607, -1607, -1455, -1455, -1219, -1219, 885, 885, -15818, -15818,
  -14322, -14322, -11999, -11999, 8711, 8711, 1212, 1212, 1029, 1029, -394, -394, -1175, -1175, 11930, 11930, 10129,
  10129, -3878, -3878, -11566, -11566,
]);

#[cfg(target_os = "macos")]
unsafe extern "C" {
  fn rscrypto_mlkem_ntt_aarch64_apple_darwin(poly: *mut i16, zetas12345: *const i16, zetas67: *const i16);
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
  fn rscrypto_mlkem_ntt_aarch64_linux(poly: *mut i16, zetas12345: *const i16, zetas67: *const i16);
}

#[inline]
pub(super) unsafe fn ntt_asm(poly: &mut Poly) {
  // SAFETY: forwarded from this function's caller; the raw assembly ABI invariants are documented in
  // `ntt_asm_raw`.
  unsafe {
    ntt_asm_raw(poly);
  }

  // SAFETY: post-assembly NEON normalization because:
  // 1. This module compiles only for aarch64 targets, where Advanced SIMD is baseline.
  // 2. `poly` is a fixed 256-coefficient polynomial; the loop in `normalize_ntt_asm` touches only
  //    in-bounds 8-coefficient chunks.
  // 3. The assembly emits signed representatives bounded by [-4Q, 4Q), so the fixed three-step
  //    reduction canonicalizes every lane without data-dependent branches or division.
  unsafe {
    normalize_ntt_asm(poly);
  }
}

#[inline]
unsafe fn ntt_asm_raw(poly: &mut Poly) {
  #[cfg(target_os = "macos")]
  {
    // SAFETY: ML-KEM aarch64 NTT assembly call because:
    // 1. `poly` is a fixed 256-lane u16 array with the same layout and size as the assembly's i16
    //    polynomial ABI.
    // 2. The twiddle tables are fixed 16-byte-aligned `i16` arrays with the exact lengths required by
    //    the assembly ABI.
    // 3. This module is compiled only for aarch64 macOS with baseline Advanced SIMD support.
    // 4. The assembly memory access schedule depends only on public ML-KEM dimensions.
    unsafe {
      rscrypto_mlkem_ntt_aarch64_apple_darwin(
        poly.as_mut_ptr().cast::<i16>(),
        NTT_ZETAS_LAYER_12345.0.as_ptr(),
        NTT_ZETAS_LAYER_67.0.as_ptr(),
      );
    }
  }

  #[cfg(target_os = "linux")]
  {
    // SAFETY: ML-KEM aarch64 NTT assembly call because:
    // 1. `poly` is a fixed 256-lane u16 array with the same layout and size as the assembly's i16
    //    polynomial ABI.
    // 2. The twiddle tables are fixed 16-byte-aligned `i16` arrays with the exact lengths required by
    //    the assembly ABI.
    // 3. This module is compiled only for aarch64 Linux with baseline Advanced SIMD support.
    // 4. The assembly memory access schedule depends only on public ML-KEM dimensions.
    unsafe {
      rscrypto_mlkem_ntt_aarch64_linux(
        poly.as_mut_ptr().cast::<i16>(),
        NTT_ZETAS_LAYER_12345.0.as_ptr(),
        NTT_ZETAS_LAYER_67.0.as_ptr(),
      );
    }
  }
}

#[cfg(test)]
pub(super) unsafe fn test_ntt_asm_raw(poly: &mut Poly) {
  // SAFETY: test-only direct access to the same raw assembly entry point used by `ntt_asm`.
  unsafe {
    ntt_asm_raw(poly);
  }
}

#[target_feature(enable = "neon")]
unsafe fn normalize_ntt_asm(poly: &mut Poly) {
  let q = vdupq_n_s16(Q_I16);
  let q4 = vdupq_n_s16(Q_TIMES_4_I32 as i16);

  for i in (0..N).step_by(8) {
    // SAFETY: fixed-size aarch64 normalization because:
    // 1. `i` advances by 8 while `i < N == 256`, so each load/store touches `i..i + 8`.
    // 2. The raw assembly output is an i16 polynomial in the same memory as `poly`.
    // 3. The reduction schedule is fixed and independent of coefficient values.
    unsafe {
      let value = vld1q_s16(poly.as_ptr().add(i).cast::<i16>());
      let negative = vshrq_n_s16::<15>(value);
      let mut value = vaddq_s16(value, vandq_s16(negative, q4));
      value = sub_q_if_ge_i16x8(value, q);
      value = sub_q_if_ge_i16x8(value, q);
      value = sub_q_if_ge_i16x8(value, q);
      vst1q_u16(poly.as_mut_ptr().add(i), vreinterpretq_u16_s16(value));
    }
  }
}

#[target_feature(enable = "neon")]
fn sub_q_if_ge_i16x8(value: int16x8_t, q: int16x8_t) -> int16x8_t {
  let reduced = vsubq_s16(value, q);
  let negative = vshrq_n_s16::<15>(reduced);
  vaddq_s16(reduced, vandq_s16(negative, q))
}

#[cfg(test)]
#[inline(always)]
fn normalize_ntt_asm_coeff(coeff: u16) -> u16 {
  let signed = i32::from(coeff as i16);
  debug_assert!((-Q_TIMES_4_I32..Q_TIMES_4_I32).contains(&signed));

  let mut value = signed + ((signed >> 31) & Q_TIMES_4_I32);
  value = sub_q_if_ge_i32(value);
  value = sub_q_if_ge_i32(value);
  value = sub_q_if_ge_i32(value);
  value as u16
}

#[cfg(test)]
#[inline(always)]
fn sub_q_if_ge_i32(value: i32) -> i32 {
  let reduced = value - Q_I32;
  reduced + ((reduced >> 31) & Q_I32)
}

#[cfg(test)]
pub(super) fn test_normalize_ntt_asm_coeff(value: i16) -> u16 {
  normalize_ntt_asm_coeff(value as u16)
}
