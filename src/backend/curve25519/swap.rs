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

#[cfg(all(test, not(miri), feature = "portable-only", feature = "x25519"))]
mod binsec {
  use super::*;
  use crate::binsec::*;
  use core::ptr;

  #[unsafe(no_mangle)]
  #[inline(never)]
  pub(super) extern "C" fn ct_binsec_curve25519_conditional_swap() -> ! {
    // SAFETY: These pointers reference fixed harness globals with static storage.
    let mut lhs = unsafe { array_from_global(&raw const CT_BINSEC_CURVE25519_LHS) };
    // SAFETY: These pointers reference fixed harness globals with static storage.
    let mut rhs = unsafe { array_from_global(&raw const CT_BINSEC_CURVE25519_RHS) };
    // SAFETY: This pointer references a fixed harness global with static storage.
    let swap = unsafe { ptr::read_volatile(ptr::addr_of!(CT_BINSEC_CURVE25519_SWAP)) };

    conditional_swap(&mut lhs, &mut rhs, swap);

    let mut acc = 0u64;
    for limb in lhs.into_iter().chain(rhs) {
      acc ^= limb;
    }
    let folded = acc | (acc >> 8) | (acc >> 16) | (acc >> 24) | (acc >> 32) | (acc >> 40) | (acc >> 48) | (acc >> 56);
    ct_binsec_done(folded.to_le_bytes()[0])
  }
}
