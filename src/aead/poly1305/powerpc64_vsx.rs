use core::simd::i64x2;

use super::{State, compute_block_scalar_reduction};

#[inline]
pub(super) fn compute_block(state: &mut State, block: &[u8; 16], partial: bool) {
  // SAFETY: Backend selection guarantees POWER vector support before this wrapper is chosen.
  unsafe { compute_block_impl(state, block, partial) }
}

#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn compute_block_impl(state: &mut State, block: &[u8; 16], partial: bool) {
  compute_block_scalar_reduction(state, block, partial, |lhs, rhs| {
    // SAFETY: target_feature ensures VSX availability for this entire wrapper.
    unsafe { sum4_mul(lhs, rhs) }
  });
}

/// Vectorized 4-element dot product: `vmulouw` + `vaddudm`.
///
/// Computes `lhs[0]*rhs[0] + lhs[1]*rhs[1] + lhs[2]*rhs[2] + lhs[3]*rhs[3]`
/// using two 128-bit multiply-odd and one 128-bit add.
#[inline(always)]
unsafe fn sum4_mul(lhs: [u32; 4], rhs: [u32; 4]) -> u64 {
  // SAFETY: POWER8+ VSX available via enclosing target_feature.
  unsafe {
    let a_lo = i64x2::from_array([i64::from(lhs[0]), i64::from(lhs[1])]);
    let b_lo = i64x2::from_array([i64::from(rhs[0]), i64::from(rhs[1])]);
    let prod_lo = vmulouw(a_lo, b_lo);

    let a_hi = i64x2::from_array([i64::from(lhs[2]), i64::from(lhs[3])]);
    let b_hi = i64x2::from_array([i64::from(rhs[2]), i64::from(rhs[3])]);
    let prod_hi = vmulouw(a_hi, b_hi);

    let sum = vaddudm(prod_lo, prod_hi);
    let lanes = sum.to_array();
    (lanes[0] as u64).wrapping_add(lanes[1] as u64)
  }
}

/// Multiply low 32 bits of each u64 lane → u64: `vmulouw`.
#[inline(always)]
unsafe fn vmulouw(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: POWER8+ VSX available via enclosing target_feature.
  unsafe {
    core::arch::asm!(
      "vmulouw {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Add u64 lanes: `vaddudm`.
#[inline(always)]
unsafe fn vaddudm(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: POWER8+ VSX available via enclosing target_feature.
  unsafe {
    core::arch::asm!(
      "vaddudm {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}
