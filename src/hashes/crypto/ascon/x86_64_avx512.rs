//! Ascon-p[12] x86_64 AVX-512 kernel.
//!
//! This keeps the current single-state dispatcher intact by broadcasting each
//! state word across a 256-bit register. AVX-512VL gives us `VPTERNLOGQ` for
//! the S-box updates and `VPROLQ` for the linear layer without forcing a new
//! batch-oriented API.
//!
//! # Safety
//!
//! All functions require the `avx512f,avx512vl` target features.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
macro_rules! ror_epi64 {
  ($value:expr, $left:literal) => {{ _mm256_rol_epi64::<$left>($value) }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! ror_epi64x8 {
  ($value:expr, $right:literal, $left:literal) => {{ _mm512_or_si512(_mm512_srli_epi64::<$right>($value), _mm512_slli_epi64::<$left>($value)) }};
}

/// Apply the Ascon-p[12] permutation using duplicated-lane AVX-512 registers.
///
/// # Safety
///
/// Caller must ensure the `avx512f` and `avx512vl` CPU features are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl")]
#[inline]
unsafe fn permute_12_x86_avx512_impl(state: &mut [u64; 5]) {
  let mut x0 = _mm256_set1_epi64x(state[0] as i64);
  let mut x1 = _mm256_set1_epi64x(state[1] as i64);
  let mut x2 = _mm256_set1_epi64x(state[2] as i64);
  let mut x3 = _mm256_set1_epi64x(state[3] as i64);
  let mut x4 = _mm256_set1_epi64x(state[4] as i64);
  let ones = _mm256_set1_epi64x(-1);

  for &c in &super::RC {
    x2 = _mm256_xor_si256(x2, _mm256_set1_epi64x(c as i64));

    x0 = _mm256_xor_si256(x0, x4);
    x4 = _mm256_xor_si256(x4, x3);
    x2 = _mm256_xor_si256(x2, x1);

    let y0 = x0;
    let y1 = x1;
    let y2 = x2;
    let y3 = x3;
    let y4 = x4;

    x0 = _mm256_ternarylogic_epi64(y0, y1, y2, 0xD2);
    x1 = _mm256_ternarylogic_epi64(y1, y2, y3, 0xD2);
    x2 = _mm256_ternarylogic_epi64(y2, y3, y4, 0xD2);
    x3 = _mm256_ternarylogic_epi64(y3, y4, y0, 0xD2);
    x4 = _mm256_ternarylogic_epi64(y4, y0, y1, 0xD2);

    x1 = _mm256_xor_si256(x1, x0);
    x0 = _mm256_xor_si256(x0, x4);
    x3 = _mm256_xor_si256(x3, x2);
    x2 = _mm256_xor_si256(x2, ones);

    let y0 = x0;
    let y1 = x1;
    let y2 = x2;
    let y3 = x3;
    let y4 = x4;

    x0 = _mm256_xor_si256(y0, _mm256_xor_si256(ror_epi64!(y0, 45), ror_epi64!(y0, 36)));
    x1 = _mm256_xor_si256(y1, _mm256_xor_si256(ror_epi64!(y1, 3), ror_epi64!(y1, 25)));
    x2 = _mm256_xor_si256(y2, _mm256_xor_si256(ror_epi64!(y2, 63), ror_epi64!(y2, 58)));
    x3 = _mm256_xor_si256(y3, _mm256_xor_si256(ror_epi64!(y3, 54), ror_epi64!(y3, 47)));
    x4 = _mm256_xor_si256(y4, _mm256_xor_si256(ror_epi64!(y4, 57), ror_epi64!(y4, 23)));
  }

  state[0] = _mm256_extract_epi64::<0>(x0) as u64;
  state[1] = _mm256_extract_epi64::<0>(x1) as u64;
  state[2] = _mm256_extract_epi64::<0>(x2) as u64;
  state[3] = _mm256_extract_epi64::<0>(x3) as u64;
  state[4] = _mm256_extract_epi64::<0>(x4) as u64;
}

/// Apply the Ascon-p[12] permutation using x86_64 AVX-512.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn permute_12_x86_avx512(state: &mut [u64; 5]) {
  // SAFETY: Dispatch verifies x86::AVX512F + x86::AVX512VL before selecting this kernel.
  unsafe { permute_12_x86_avx512_impl(state) }
}

/// Apply the Ascon-p[12] permutation to eight independent states in parallel.
///
/// # Safety
///
/// Caller must ensure the `avx512f` and `avx512vl` CPU features are available.
#[cfg(target_arch = "x86_64")]
#[cfg_attr(not(any(test, feature = "std")), allow(dead_code))]
#[target_feature(enable = "avx512f,avx512vl")]
#[inline]
unsafe fn permute_12_x86_avx512_x8_impl(states: &mut [[u64; 5]; 8]) {
  let mut x0 = _mm512_set_epi64(
    states[7][0] as i64,
    states[6][0] as i64,
    states[5][0] as i64,
    states[4][0] as i64,
    states[3][0] as i64,
    states[2][0] as i64,
    states[1][0] as i64,
    states[0][0] as i64,
  );
  let mut x1 = _mm512_set_epi64(
    states[7][1] as i64,
    states[6][1] as i64,
    states[5][1] as i64,
    states[4][1] as i64,
    states[3][1] as i64,
    states[2][1] as i64,
    states[1][1] as i64,
    states[0][1] as i64,
  );
  let mut x2 = _mm512_set_epi64(
    states[7][2] as i64,
    states[6][2] as i64,
    states[5][2] as i64,
    states[4][2] as i64,
    states[3][2] as i64,
    states[2][2] as i64,
    states[1][2] as i64,
    states[0][2] as i64,
  );
  let mut x3 = _mm512_set_epi64(
    states[7][3] as i64,
    states[6][3] as i64,
    states[5][3] as i64,
    states[4][3] as i64,
    states[3][3] as i64,
    states[2][3] as i64,
    states[1][3] as i64,
    states[0][3] as i64,
  );
  let mut x4 = _mm512_set_epi64(
    states[7][4] as i64,
    states[6][4] as i64,
    states[5][4] as i64,
    states[4][4] as i64,
    states[3][4] as i64,
    states[2][4] as i64,
    states[1][4] as i64,
    states[0][4] as i64,
  );
  let ones = _mm512_set1_epi64(-1);

  for &c in &super::RC {
    x2 = _mm512_xor_si512(x2, _mm512_set1_epi64(c as i64));

    x0 = _mm512_xor_si512(x0, x4);
    x4 = _mm512_xor_si512(x4, x3);
    x2 = _mm512_xor_si512(x2, x1);

    let y0 = x0;
    let y1 = x1;
    let y2 = x2;
    let y3 = x3;
    let y4 = x4;

    x0 = _mm512_ternarylogic_epi64(y0, y1, y2, 0xD2);
    x1 = _mm512_ternarylogic_epi64(y1, y2, y3, 0xD2);
    x2 = _mm512_ternarylogic_epi64(y2, y3, y4, 0xD2);
    x3 = _mm512_ternarylogic_epi64(y3, y4, y0, 0xD2);
    x4 = _mm512_ternarylogic_epi64(y4, y0, y1, 0xD2);

    x1 = _mm512_xor_si512(x1, x0);
    x0 = _mm512_xor_si512(x0, x4);
    x3 = _mm512_xor_si512(x3, x2);
    x2 = _mm512_xor_si512(x2, ones);

    let y0 = x0;
    let y1 = x1;
    let y2 = x2;
    let y3 = x3;
    let y4 = x4;

    x0 = _mm512_xor_si512(y0, _mm512_xor_si512(ror_epi64x8!(y0, 19, 45), ror_epi64x8!(y0, 28, 36)));
    x1 = _mm512_xor_si512(y1, _mm512_xor_si512(ror_epi64x8!(y1, 61, 3), ror_epi64x8!(y1, 39, 25)));
    x2 = _mm512_xor_si512(y2, _mm512_xor_si512(ror_epi64x8!(y2, 1, 63), ror_epi64x8!(y2, 6, 58)));
    x3 = _mm512_xor_si512(y3, _mm512_xor_si512(ror_epi64x8!(y3, 10, 54), ror_epi64x8!(y3, 17, 47)));
    x4 = _mm512_xor_si512(y4, _mm512_xor_si512(ror_epi64x8!(y4, 7, 57), ror_epi64x8!(y4, 41, 23)));
  }

  let mut lanes0 = [0u64; 8];
  let mut lanes1 = [0u64; 8];
  let mut lanes2 = [0u64; 8];
  let mut lanes3 = [0u64; 8];
  let mut lanes4 = [0u64; 8];
  // SAFETY: arrays are valid for 64-byte stores and hold exactly eight u64 lanes each.
  unsafe {
    _mm512_storeu_si512(lanes0.as_mut_ptr().cast(), x0);
    _mm512_storeu_si512(lanes1.as_mut_ptr().cast(), x1);
    _mm512_storeu_si512(lanes2.as_mut_ptr().cast(), x2);
    _mm512_storeu_si512(lanes3.as_mut_ptr().cast(), x3);
    _mm512_storeu_si512(lanes4.as_mut_ptr().cast(), x4);
  }
  for lane in 0..8 {
    states[lane][0] = lanes0[lane];
    states[lane][1] = lanes1[lane];
    states[lane][2] = lanes2[lane];
    states[lane][3] = lanes3[lane];
    states[lane][4] = lanes4[lane];
  }
}

/// Apply the Ascon-p[12] permutation to eight independent states in parallel.
#[cfg(target_arch = "x86_64")]
#[cfg_attr(not(any(test, feature = "std")), allow(dead_code))]
#[inline]
pub(crate) fn permute_12_x86_avx512_x8(states: &mut [[u64; 5]; 8]) {
  // SAFETY: Dispatch verifies x86::AVX512F + x86::AVX512VL before selecting this kernel.
  unsafe { permute_12_x86_avx512_x8_impl(states) }
}
