//! Original four-lane ML-DSA arithmetic for the baseline z/Vector facility.
//!
//! Coefficients keep the portable canonical representation. Widening even/odd
//! multiplies compute Montgomery products without scalar lane multiplication.

use core::arch::s390x::{vec_add, vec_sub, vector_unsigned_int};
use core::simd::{i64x2, simd_swizzle, u32x4};

use super::super::{NEG_Q_INVERSE, Q};

/// One subtraction for four inputs below 2q.
///
/// # Safety
/// The executing CPU must support z/Vector.
#[inline]
#[target_feature(enable = "vector")]
pub(super) unsafe fn reduce(value: u32x4) -> u32x4 {
  let out: i64x2;
  // SAFETY: Each transmute preserves one 128-bit vector register; all integer
  // bit patterns are valid. VSF subtracts q modulo 2^32. Since value < 2q <
  // 2^24, VESRAF produces all ones exactly when that subtraction borrowed.
  // VN and VAF add q back in those lanes. Every operand is a register, and the
  // instructions access no memory, stack, or condition code. Early outputs
  // cannot overlap the still-live input or q registers.
  unsafe {
    core::arch::asm!(
      "vsf {difference}, {value}, {q}",
      "vesraf {mask}, {difference}, 31",
      "vn {mask}, {mask}, {q}",
      "vaf {out}, {difference}, {mask}",
      value = in(vreg) core::mem::transmute::<u32x4, i64x2>(value),
      q = in(vreg) core::mem::transmute::<u32x4, i64x2>(u32x4::splat(Q)),
      difference = out(vreg) _,
      mask = out(vreg) _,
      out = lateout(vreg) out,
      options(nomem, nostack, pure, preserves_flags),
    );
    core::mem::transmute(out)
  }
}

/// Canonical Montgomery products for operands below 2q.
///
/// # Safety
/// The executing CPU must support z/Vector.
#[inline]
#[target_feature(enable = "vector")]
pub(super) unsafe fn multiply(a: u32x4, b: u32x4) -> u32x4 {
  let even: i64x2;
  let odd: i64x2;
  // SAFETY: z/Vector is established by the caller. Every transmute preserves
  // one 128-bit register and all integer bit patterns are valid. VMLF computes
  // the low products modulo 2^32; VMLEF/VMLOF widen unsigned even/odd u32 lanes.
  // VAG adds the correction in u64 lanes, bounded below 2^56 as proved by
  // super::super::montgomery. Early outputs preserve all live inputs. These baseline
  // vector instructions access no memory, stack, or condition code.
  // Explicit instructions prevent LLVM from scalarizing widening intrinsics
  // into secret-fed MSGR/MSGFI instructions.
  unsafe {
    core::arch::asm!(
      "vmlf {m}, {a}, {b}",
      "vmlf {m}, {m}, {inverse}",
      "vmlef {even}, {a}, {b}",
      "vmlof {odd}, {a}, {b}",
      "vmlef {correction}, {m}, {q}",
      "vag {even}, {even}, {correction}",
      "vmlof {correction}, {m}, {q}",
      "vag {odd}, {odd}, {correction}",
      a = in(vreg) core::mem::transmute::<u32x4, i64x2>(a),
      b = in(vreg) core::mem::transmute::<u32x4, i64x2>(b),
      q = in(vreg) core::mem::transmute::<u32x4, i64x2>(u32x4::splat(Q)),
      inverse = in(vreg) core::mem::transmute::<u32x4, i64x2>(u32x4::splat(NEG_Q_INVERSE)),
      m = out(vreg) _,
      correction = out(vreg) _,
      even = out(vreg) even,
      odd = out(vreg) odd,
      options(nomem, nostack, pure, preserves_flags),
    );
    // s390x is big-endian: each u64 high word occupies the even u32 lane.
    // Interleave the high words to recover the original coefficient order.
    let even: u32x4 = core::mem::transmute(even);
    let odd: u32x4 = core::mem::transmute(odd);
    reduce(simd_swizzle!(even, odd, [0, 4, 2, 6]))
  }
}

/// # Safety
/// The executing CPU must support z/Vector.
#[inline]
#[target_feature(enable = "vector")]
pub(super) unsafe fn add_lanes(a: u32x4, b: u32x4) -> u32x4 {
  // SAFETY: z/Vector is established. Both vector representations have identical
  // integer lanes. VAF performs wrapping addition with no scalar overflow check.
  unsafe {
    core::mem::transmute(vec_add(
      core::mem::transmute::<u32x4, vector_unsigned_int>(a),
      core::mem::transmute::<u32x4, vector_unsigned_int>(b),
    ))
  }
}

/// # Safety
/// The executing CPU must support z/Vector.
#[inline]
#[target_feature(enable = "vector")]
pub(super) unsafe fn sub_lanes(a: u32x4, b: u32x4) -> u32x4 {
  // SAFETY: z/Vector is established. Both vector representations have identical
  // integer lanes. VSF performs wrapping subtraction with no scalar overflow check.
  unsafe {
    core::mem::transmute(vec_sub(
      core::mem::transmute::<u32x4, vector_unsigned_int>(a),
      core::mem::transmute::<u32x4, vector_unsigned_int>(b),
    ))
  }
}
