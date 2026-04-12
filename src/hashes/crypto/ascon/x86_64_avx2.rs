//! Ascon-p[12] x86_64 AVX2 kernel.
//!
//! The current Ascon dispatcher operates on a single `[u64; 5]` state, so this
//! kernel uses duplicated-lane `__m256i` registers rather than a 4-state batch
//! API. That still gives us a real AVX2 implementation for the existing call
//! surface and keeps the eventual multi-state throughput work separate.
//!
//! # Safety
//!
//! All functions require the `avx2` target feature.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
macro_rules! ror_epi64 {
  ($value:expr, $right:literal, $left:literal) => {{ _mm256_or_si256(_mm256_srli_epi64::<$right>($value), _mm256_slli_epi64::<$left>($value)) }};
}

/// Apply the Ascon-p[12] permutation using duplicated-lane AVX2 registers.
///
/// # Safety
///
/// Caller must ensure the `avx2` CPU feature is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn permute_12_x86_avx2_impl(state: &mut [u64; 5]) {
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

    x0 = _mm256_xor_si256(y0, _mm256_andnot_si256(y1, y2));
    x1 = _mm256_xor_si256(y1, _mm256_andnot_si256(y2, y3));
    x2 = _mm256_xor_si256(y2, _mm256_andnot_si256(y3, y4));
    x3 = _mm256_xor_si256(y3, _mm256_andnot_si256(y4, y0));
    x4 = _mm256_xor_si256(y4, _mm256_andnot_si256(y0, y1));

    x1 = _mm256_xor_si256(x1, x0);
    x0 = _mm256_xor_si256(x0, x4);
    x3 = _mm256_xor_si256(x3, x2);
    x2 = _mm256_xor_si256(x2, ones);

    let y0 = x0;
    let y1 = x1;
    let y2 = x2;
    let y3 = x3;
    let y4 = x4;

    x0 = _mm256_xor_si256(y0, _mm256_xor_si256(ror_epi64!(y0, 19, 45), ror_epi64!(y0, 28, 36)));
    x1 = _mm256_xor_si256(y1, _mm256_xor_si256(ror_epi64!(y1, 61, 3), ror_epi64!(y1, 39, 25)));
    x2 = _mm256_xor_si256(y2, _mm256_xor_si256(ror_epi64!(y2, 1, 63), ror_epi64!(y2, 6, 58)));
    x3 = _mm256_xor_si256(y3, _mm256_xor_si256(ror_epi64!(y3, 10, 54), ror_epi64!(y3, 17, 47)));
    x4 = _mm256_xor_si256(y4, _mm256_xor_si256(ror_epi64!(y4, 7, 57), ror_epi64!(y4, 41, 23)));
  }

  state[0] = _mm256_extract_epi64::<0>(x0) as u64;
  state[1] = _mm256_extract_epi64::<0>(x1) as u64;
  state[2] = _mm256_extract_epi64::<0>(x2) as u64;
  state[3] = _mm256_extract_epi64::<0>(x3) as u64;
  state[4] = _mm256_extract_epi64::<0>(x4) as u64;
}

/// Apply the Ascon-p[12] permutation using x86_64 AVX2.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn permute_12_x86_avx2(state: &mut [u64; 5]) {
  // SAFETY: Dispatch verifies x86::AVX2 before selecting this kernel.
  unsafe { permute_12_x86_avx2_impl(state) }
}

/// Apply the Ascon-p[12] permutation to four independent states in parallel.
///
/// # Safety
///
/// Caller must ensure the `avx2` CPU feature is available.
#[cfg(target_arch = "x86_64")]
#[cfg_attr(not(any(test, feature = "std")), allow(dead_code))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn permute_12_x86_avx2_x4_impl(states: &mut [[u64; 5]; 4]) {
  let mut x0 = _mm256_set_epi64x(
    states[3][0] as i64,
    states[2][0] as i64,
    states[1][0] as i64,
    states[0][0] as i64,
  );
  let mut x1 = _mm256_set_epi64x(
    states[3][1] as i64,
    states[2][1] as i64,
    states[1][1] as i64,
    states[0][1] as i64,
  );
  let mut x2 = _mm256_set_epi64x(
    states[3][2] as i64,
    states[2][2] as i64,
    states[1][2] as i64,
    states[0][2] as i64,
  );
  let mut x3 = _mm256_set_epi64x(
    states[3][3] as i64,
    states[2][3] as i64,
    states[1][3] as i64,
    states[0][3] as i64,
  );
  let mut x4 = _mm256_set_epi64x(
    states[3][4] as i64,
    states[2][4] as i64,
    states[1][4] as i64,
    states[0][4] as i64,
  );
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

    x0 = _mm256_xor_si256(y0, _mm256_andnot_si256(y1, y2));
    x1 = _mm256_xor_si256(y1, _mm256_andnot_si256(y2, y3));
    x2 = _mm256_xor_si256(y2, _mm256_andnot_si256(y3, y4));
    x3 = _mm256_xor_si256(y3, _mm256_andnot_si256(y4, y0));
    x4 = _mm256_xor_si256(y4, _mm256_andnot_si256(y0, y1));

    x1 = _mm256_xor_si256(x1, x0);
    x0 = _mm256_xor_si256(x0, x4);
    x3 = _mm256_xor_si256(x3, x2);
    x2 = _mm256_xor_si256(x2, ones);

    let y0 = x0;
    let y1 = x1;
    let y2 = x2;
    let y3 = x3;
    let y4 = x4;

    x0 = _mm256_xor_si256(y0, _mm256_xor_si256(ror_epi64!(y0, 19, 45), ror_epi64!(y0, 28, 36)));
    x1 = _mm256_xor_si256(y1, _mm256_xor_si256(ror_epi64!(y1, 61, 3), ror_epi64!(y1, 39, 25)));
    x2 = _mm256_xor_si256(y2, _mm256_xor_si256(ror_epi64!(y2, 1, 63), ror_epi64!(y2, 6, 58)));
    x3 = _mm256_xor_si256(y3, _mm256_xor_si256(ror_epi64!(y3, 10, 54), ror_epi64!(y3, 17, 47)));
    x4 = _mm256_xor_si256(y4, _mm256_xor_si256(ror_epi64!(y4, 7, 57), ror_epi64!(y4, 41, 23)));
  }

  states[0][0] = _mm256_extract_epi64::<0>(x0) as u64;
  states[1][0] = _mm256_extract_epi64::<1>(x0) as u64;
  states[2][0] = _mm256_extract_epi64::<2>(x0) as u64;
  states[3][0] = _mm256_extract_epi64::<3>(x0) as u64;
  states[0][1] = _mm256_extract_epi64::<0>(x1) as u64;
  states[1][1] = _mm256_extract_epi64::<1>(x1) as u64;
  states[2][1] = _mm256_extract_epi64::<2>(x1) as u64;
  states[3][1] = _mm256_extract_epi64::<3>(x1) as u64;
  states[0][2] = _mm256_extract_epi64::<0>(x2) as u64;
  states[1][2] = _mm256_extract_epi64::<1>(x2) as u64;
  states[2][2] = _mm256_extract_epi64::<2>(x2) as u64;
  states[3][2] = _mm256_extract_epi64::<3>(x2) as u64;
  states[0][3] = _mm256_extract_epi64::<0>(x3) as u64;
  states[1][3] = _mm256_extract_epi64::<1>(x3) as u64;
  states[2][3] = _mm256_extract_epi64::<2>(x3) as u64;
  states[3][3] = _mm256_extract_epi64::<3>(x3) as u64;
  states[0][4] = _mm256_extract_epi64::<0>(x4) as u64;
  states[1][4] = _mm256_extract_epi64::<1>(x4) as u64;
  states[2][4] = _mm256_extract_epi64::<2>(x4) as u64;
  states[3][4] = _mm256_extract_epi64::<3>(x4) as u64;
}

/// Apply the Ascon-p[12] permutation to four independent states in parallel.
#[cfg(target_arch = "x86_64")]
#[cfg_attr(not(any(test, feature = "std")), allow(dead_code))]
#[inline]
pub(crate) fn permute_12_x86_avx2_x4(states: &mut [[u64; 5]; 4]) {
  // SAFETY: Dispatch verifies x86::AVX2 before selecting this kernel.
  unsafe { permute_12_x86_avx2_x4_impl(states) }
}
