use core::simd::i64x2;

use super::{State, compute_block_scalar_reduction};

#[inline]
pub(super) fn compute_block(state: &mut State, block: &[u8; 16], partial: bool) {
  // SAFETY: backend selection guarantees the RISC-V V extension before this wrapper is chosen.
  unsafe { compute_block_impl(state, block, partial) }
}

#[target_feature(enable = "v")]
unsafe fn compute_block_impl(state: &mut State, block: &[u8; 16], partial: bool) {
  compute_block_scalar_reduction(state, block, partial, |lhs, rhs| {
    // SAFETY: target_feature ensures RVV availability for this entire wrapper.
    unsafe { sum4_mul(lhs, rhs) }
  });
}

/// Vectorized 4-element dot product using two 64-bit RVV lane multiplies.
#[target_feature(enable = "v")]
unsafe fn sum4_mul(lhs: [u32; 4], rhs: [u32; 4]) -> u64 {
  let a_lo = i64x2::from_array([i64::from(lhs[0]), i64::from(lhs[1])]);
  let b_lo = i64x2::from_array([i64::from(rhs[0]), i64::from(rhs[1])]);
  let prod_lo = a_lo * b_lo;

  let a_hi = i64x2::from_array([i64::from(lhs[2]), i64::from(lhs[3])]);
  let b_hi = i64x2::from_array([i64::from(rhs[2]), i64::from(rhs[3])]);
  let prod_hi = a_hi * b_hi;

  let sum = prod_lo + prod_hi;
  let lanes = sum.to_array();
  (lanes[0] as u64).wrapping_add(lanes[1] as u64)
}
