//! Ascon-p[12] aarch64 NEON kernel.
//!
//! This is the current SIMD-compatible single-state path for the existing
//! `fn(&mut [u64; 5])` dispatcher: each word is broadcast into both lanes of a
//! `uint64x2_t`, the permutation runs lane-wise, and lane 0 is written back.
//! It avoids API churn while giving us a real NEON kernel slot and keeps the
//! follow-up multi-state batch path separate.
//!
//! # Safety
//!
//! All functions require the `neon` target feature.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
macro_rules! ror_u64x2 {
  ($value:expr, $right:literal, $left:literal) => {{ vorrq_u64(vshrq_n_u64::<$right>($value), vshlq_n_u64::<$left>($value)) }};
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn combine_lanes(a: u64, b: u64) -> uint64x2_t {
  vcombine_u64(vcreate_u64(a), vcreate_u64(b))
}

/// Apply the Ascon-p[12] permutation using duplicated-lane NEON registers.
///
/// # Safety
///
/// Caller must ensure the `neon` CPU feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn permute_12_aarch64_neon_impl(state: &mut [u64; 5]) {
  let mut x0 = vdupq_n_u64(state[0]);
  let mut x1 = vdupq_n_u64(state[1]);
  let mut x2 = vdupq_n_u64(state[2]);
  let mut x3 = vdupq_n_u64(state[3]);
  let mut x4 = vdupq_n_u64(state[4]);
  let ones = vdupq_n_u64(u64::MAX);

  for &c in &super::RC {
    x2 = veorq_u64(x2, vdupq_n_u64(c));

    x0 = veorq_u64(x0, x4);
    x4 = veorq_u64(x4, x3);
    x2 = veorq_u64(x2, x1);

    let y0 = x0;
    let y1 = x1;
    let y2 = x2;
    let y3 = x3;
    let y4 = x4;

    x0 = veorq_u64(y0, vbicq_u64(y2, y1));
    x1 = veorq_u64(y1, vbicq_u64(y3, y2));
    x2 = veorq_u64(y2, vbicq_u64(y4, y3));
    x3 = veorq_u64(y3, vbicq_u64(y0, y4));
    x4 = veorq_u64(y4, vbicq_u64(y1, y0));

    x1 = veorq_u64(x1, x0);
    x0 = veorq_u64(x0, x4);
    x3 = veorq_u64(x3, x2);
    x2 = veorq_u64(x2, ones);

    let y0 = x0;
    let y1 = x1;
    let y2 = x2;
    let y3 = x3;
    let y4 = x4;

    x0 = veorq_u64(y0, veorq_u64(ror_u64x2!(y0, 19, 45), ror_u64x2!(y0, 28, 36)));
    x1 = veorq_u64(y1, veorq_u64(ror_u64x2!(y1, 61, 3), ror_u64x2!(y1, 39, 25)));
    x2 = veorq_u64(y2, veorq_u64(ror_u64x2!(y2, 1, 63), ror_u64x2!(y2, 6, 58)));
    x3 = veorq_u64(y3, veorq_u64(ror_u64x2!(y3, 10, 54), ror_u64x2!(y3, 17, 47)));
    x4 = veorq_u64(y4, veorq_u64(ror_u64x2!(y4, 7, 57), ror_u64x2!(y4, 41, 23)));
  }

  state[0] = vgetq_lane_u64(x0, 0);
  state[1] = vgetq_lane_u64(x1, 0);
  state[2] = vgetq_lane_u64(x2, 0);
  state[3] = vgetq_lane_u64(x3, 0);
  state[4] = vgetq_lane_u64(x4, 0);
}

/// Apply the Ascon-p[12] permutation using aarch64 NEON.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn permute_12_aarch64_neon(state: &mut [u64; 5]) {
  // SAFETY: Dispatch verifies aarch64::NEON before selecting this kernel.
  unsafe { permute_12_aarch64_neon_impl(state) }
}

/// Apply the Ascon-p[12] permutation to two independent states in parallel.
///
/// # Safety
///
/// Caller must ensure the `neon` CPU feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn permute_12_aarch64_neon_x2_impl(states: &mut [[u64; 5]; 2]) {
  // SAFETY: `combine_lanes` only requires the active `neon` target feature,
  // which is guaranteed by this function's contract and attribute.
  let (mut x0, mut x1, mut x2, mut x3, mut x4) = unsafe {
    (
      combine_lanes(states[0][0], states[1][0]),
      combine_lanes(states[0][1], states[1][1]),
      combine_lanes(states[0][2], states[1][2]),
      combine_lanes(states[0][3], states[1][3]),
      combine_lanes(states[0][4], states[1][4]),
    )
  };
  let ones = vdupq_n_u64(u64::MAX);

  for &c in &super::RC {
    x2 = veorq_u64(x2, vdupq_n_u64(c));

    x0 = veorq_u64(x0, x4);
    x4 = veorq_u64(x4, x3);
    x2 = veorq_u64(x2, x1);

    let y0 = x0;
    let y1 = x1;
    let y2 = x2;
    let y3 = x3;
    let y4 = x4;

    x0 = veorq_u64(y0, vbicq_u64(y2, y1));
    x1 = veorq_u64(y1, vbicq_u64(y3, y2));
    x2 = veorq_u64(y2, vbicq_u64(y4, y3));
    x3 = veorq_u64(y3, vbicq_u64(y0, y4));
    x4 = veorq_u64(y4, vbicq_u64(y1, y0));

    x1 = veorq_u64(x1, x0);
    x0 = veorq_u64(x0, x4);
    x3 = veorq_u64(x3, x2);
    x2 = veorq_u64(x2, ones);

    let y0 = x0;
    let y1 = x1;
    let y2 = x2;
    let y3 = x3;
    let y4 = x4;

    x0 = veorq_u64(y0, veorq_u64(ror_u64x2!(y0, 19, 45), ror_u64x2!(y0, 28, 36)));
    x1 = veorq_u64(y1, veorq_u64(ror_u64x2!(y1, 61, 3), ror_u64x2!(y1, 39, 25)));
    x2 = veorq_u64(y2, veorq_u64(ror_u64x2!(y2, 1, 63), ror_u64x2!(y2, 6, 58)));
    x3 = veorq_u64(y3, veorq_u64(ror_u64x2!(y3, 10, 54), ror_u64x2!(y3, 17, 47)));
    x4 = veorq_u64(y4, veorq_u64(ror_u64x2!(y4, 7, 57), ror_u64x2!(y4, 41, 23)));
  }

  states[0][0] = vgetq_lane_u64(x0, 0);
  states[1][0] = vgetq_lane_u64(x0, 1);
  states[0][1] = vgetq_lane_u64(x1, 0);
  states[1][1] = vgetq_lane_u64(x1, 1);
  states[0][2] = vgetq_lane_u64(x2, 0);
  states[1][2] = vgetq_lane_u64(x2, 1);
  states[0][3] = vgetq_lane_u64(x3, 0);
  states[1][3] = vgetq_lane_u64(x3, 1);
  states[0][4] = vgetq_lane_u64(x4, 0);
  states[1][4] = vgetq_lane_u64(x4, 1);
}

/// Apply the Ascon-p[12] permutation to two independent states in parallel.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn permute_12_aarch64_neon_x2(states: &mut [[u64; 5]; 2]) {
  // SAFETY: Dispatch verifies aarch64::NEON before selecting this kernel.
  unsafe { permute_12_aarch64_neon_x2_impl(states) }
}
