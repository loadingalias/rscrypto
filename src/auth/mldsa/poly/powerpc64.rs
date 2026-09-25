//! Original four-lane ML-DSA arithmetic for little-endian POWER8 and later.
//!
//! Explicit integer-vector operations preserve the canonical representation
//! without relying on LLVM's scalarization or conditional-selection choices.

use core::simd::{i64x2, simd_swizzle, u32x4};

use super::super::{NEG_Q_INVERSE, Q};

/// Reduce four inputs below 2q with one masked subtraction.
///
/// # Safety
/// The caller must establish Altivec, VSX, and POWER8 vector support.
#[inline]
#[target_feature(enable = "altivec,vsx,power8-vector")]
pub(super) unsafe fn reduce(value: u32x4) -> u32x4 {
  let out: i64x2;
  // SAFETY: Each transmute preserves a 128-bit vector register. VSUBUWM
  // subtracts q modulo 2^32; since value < 2q < 2^24, VSRAW yields all ones
  // precisely for a borrow. VAND and VADDUWM restore q in those lanes.
  // Early outputs cannot overlap still-live inputs. No instruction touches
  // memory, the stack, or the condition register; all require only POWER8.
  unsafe {
    core::arch::asm!(
      "vsubuwm {difference}, {value}, {q}",
      "vsraw {mask}, {difference}, {shift}",
      "vand {mask}, {mask}, {q}",
      "vadduwm {out}, {difference}, {mask}",
      value = in(vreg) core::mem::transmute::<u32x4, i64x2>(value),
      q = in(vreg) core::mem::transmute::<u32x4, i64x2>(u32x4::splat(Q)),
      shift = in(vreg) core::mem::transmute::<u32x4, i64x2>(u32x4::splat(31)),
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
/// The caller must establish Altivec, VSX, and POWER8 vector support.
#[inline]
#[target_feature(enable = "altivec,vsx,power8-vector")]
pub(super) unsafe fn multiply(a: u32x4, b: u32x4) -> u32x4 {
  let even: i64x2;
  let odd: i64x2;
  // SAFETY: All transmuted values have identical 128-bit register layouts.
  // VMULUWM forms the low product/correction modulo 2^32. VMULEUW/VMULOUW
  // widen unsigned words. The widened sums fit below 2^56, as proved by
  // super::super::montgomery. Early outputs preserve live inputs. These
  // POWER8 operations do not access memory, stack, or the condition register.
  unsafe {
    core::arch::asm!(
      "vmuluwm {m}, {a}, {b}",
      "vmuluwm {m}, {m}, {inverse}",
      "vmuleuw {even}, {a}, {b}",
      "vmulouw {odd}, {a}, {b}",
      "vmuleuw {correction}, {m}, {q}",
      "vaddudm {even}, {even}, {correction}",
      "vmulouw {correction}, {m}, {q}",
      "vaddudm {odd}, {odd}, {correction}",
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
    // Little-endian POWER reverses architectural word numbering relative to
    // Rust's memory-order lanes. Architectural odd products contain Rust lanes
    // 0/2, and even products contain 1/3. Each high word is at u32 index 1 or 3.
    let even: u32x4 = core::mem::transmute(even);
    let odd: u32x4 = core::mem::transmute(odd);
    reduce(simd_swizzle!(odd, even, [1, 5, 3, 7]))
  }
}

/// # Safety
/// The caller must establish Altivec, VSX, and POWER8 vector support.
#[inline]
#[target_feature(enable = "altivec,vsx,power8-vector")]
pub(super) unsafe fn add_lanes(a: u32x4, b: u32x4) -> u32x4 {
  let out: i64x2;
  // SAFETY: The inputs and output are 128-bit registers with all bit patterns
  // valid. VADDUWM wraps each word modulo 2^32 and has no memory or flag effects.
  unsafe {
    core::arch::asm!(
      "vadduwm {out}, {a}, {b}",
      a = in(vreg) core::mem::transmute::<u32x4, i64x2>(a),
      b = in(vreg) core::mem::transmute::<u32x4, i64x2>(b),
      out = lateout(vreg) out,
      options(nomem, nostack, pure, preserves_flags),
    );
    core::mem::transmute(out)
  }
}

/// # Safety
/// The caller must establish Altivec, VSX, and POWER8 vector support.
#[inline]
#[target_feature(enable = "altivec,vsx,power8-vector")]
pub(super) unsafe fn sub_lanes(a: u32x4, b: u32x4) -> u32x4 {
  let out: i64x2;
  // SAFETY: The register representations have identical layouts. VSUBUWM
  // wraps each word modulo 2^32 and has no memory, stack, or flag effects.
  unsafe {
    core::arch::asm!(
      "vsubuwm {out}, {a}, {b}",
      a = in(vreg) core::mem::transmute::<u32x4, i64x2>(a),
      b = in(vreg) core::mem::transmute::<u32x4, i64x2>(b),
      out = lateout(vreg) out,
      options(nomem, nostack, pure, preserves_flags),
    );
    core::mem::transmute(out)
  }
}
