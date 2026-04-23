use core::simd::i64x2;

use super::{State, compute_block_scalar_reduction};

#[inline]
pub(super) fn compute_block(state: &mut State, block: &[u8; 16], partial: bool) {
  // SAFETY: Backend selection guarantees the z/Vector facility before this wrapper is chosen.
  unsafe { compute_block_impl(state, block, partial) }
}

#[target_feature(enable = "vector")]
unsafe fn compute_block_impl(state: &mut State, block: &[u8; 16], partial: bool) {
  compute_block_scalar_reduction(state, block, partial, |lhs, rhs| {
    // SAFETY: target_feature ensures z/Vector availability for this entire wrapper.
    unsafe { sum4_mul(lhs, rhs) }
  });
}

/// Vectorized 4-element dot product: `vmlof` + `vag`.
///
/// Computes `lhs[0]*rhs[0] + lhs[1]*rhs[1] + lhs[2]*rhs[2] + lhs[3]*rhs[3]`
/// using two 128-bit multiply-odd and one 128-bit add.
#[inline]
#[target_feature(enable = "vector")]
unsafe fn sum4_mul(lhs: [u32; 4], rhs: [u32; 4]) -> u64 {
  // SAFETY: z/Vector facility available via target_feature.
  unsafe {
    let a_lo = i64x2::from_array([i64::from(lhs[0]), i64::from(lhs[1])]);
    let b_lo = i64x2::from_array([i64::from(rhs[0]), i64::from(rhs[1])]);
    let prod_lo = vmlof(a_lo, b_lo);

    let a_hi = i64x2::from_array([i64::from(lhs[2]), i64::from(lhs[3])]);
    let b_hi = i64x2::from_array([i64::from(rhs[2]), i64::from(rhs[3])]);
    let prod_hi = vmlof(a_hi, b_hi);

    let sum = vag(prod_lo, prod_hi);
    let lanes = sum.to_array();
    (lanes[0] as u64).wrapping_add(lanes[1] as u64)
  }
}

/// Multiply odd-indexed u32 lanes → u64: `vmlof`.
#[inline]
#[target_feature(enable = "vector")]
unsafe fn vmlof(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z/Vector facility available via target_feature.
  unsafe {
    core::arch::asm!(
      "vmlof {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Add u64 lanes: `vag`.
#[inline]
#[target_feature(enable = "vector")]
unsafe fn vag(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z/Vector facility available via target_feature.
  unsafe {
    core::arch::asm!(
      "vag {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}
