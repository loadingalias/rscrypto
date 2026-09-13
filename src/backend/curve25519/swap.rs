//! Conditional swap shared by the portable X25519 ladder and its internal evidence probe.

/// Swap radix-51 limbs when the low bit of `swap` is set.
#[inline(always)]
pub(crate) fn conditional_swap(lhs: &mut [u64; 5], rhs: &mut [u64; 5], swap: u8) {
  let mask = 0u64.wrapping_sub(u64::from(swap & 1));
  for (lhs_limb, rhs_limb) in lhs.iter_mut().zip(rhs.iter_mut()) {
    let diff = mask & (*lhs_limb ^ *rhs_limb);
    *lhs_limb ^= diff;
    *rhs_limb ^= diff;
  }
}

/// Exercise the portable X25519 ladder's conditional swap.
#[cfg(all(rscrypto_internal, feature = "diag"))]
#[inline(always)]
pub fn diag_curve25519_conditional_swap(lhs: &mut [u64; 5], rhs: &mut [u64; 5], swap: u8) {
  conditional_swap(lhs, rhs, swap);
}
