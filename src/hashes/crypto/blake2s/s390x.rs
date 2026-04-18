//! Blake2s z/Vector-accelerated compression for s390x.
//!
//! Blake2s works on a 4x4 matrix of `u32`, so each row fits exactly in one
//! 128-bit vector register. z/Vector gives us word-wise add/XOR/rotate, while
//! row diagonalization uses `vperm` because Blake2s needs 32-bit lane shuffles,
//! not the 64-bit pair shuffles Blake2b uses.
//!
//! All four Blake2s rotations (16, 12, 8, 7) are rotate-RIGHT values per
//! RFC 7693 §3.1. z/Arch `verll` is a rotate-LEFT instruction, so the helper
//! `rotr32_via_verll::<ROR>` passes `32 - ROR` as the immediate and the call
//! sites read the spec constants directly. `ror 16` is self-symmetric
//! (rol 16 == ror 16 on u32), which is the only reason this file produced
//! one correct rotation out of four before the ROR-aware helper landed.

#![allow(unsafe_code)]
#![allow(clippy::cast_possible_truncation, clippy::indexing_slicing)]

use core::simd::i64x2;

use super::kernels::{SIGMA, init_v, load_msg};

const VPERM_ROT_LEFT_1: [u8; 16] = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3];
const VPERM_ROT_LEFT_2: [u8; 16] = [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7];
const VPERM_ROT_LEFT_3: [u8; 16] = [12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

struct DiagMasks {
  rot1: i64x2,
  rot2: i64x2,
  rot3: i64x2,
}

#[inline(always)]
fn from_u32x4(words: [u32; 4]) -> i64x2 {
  // SAFETY: [u32; 4] and i64x2 are both 16 bytes with identical alignment.
  unsafe { core::mem::transmute(words) }
}

#[cfg(test)]
#[inline(always)]
fn to_u32x4(v: i64x2) -> [u32; 4] {
  // SAFETY: i64x2 and [u32; 4] are both 16 bytes with identical alignment.
  unsafe { core::mem::transmute(v) }
}

#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn vaf(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z/Vector facility available via target_feature.
  unsafe {
    core::arch::asm!(
      "vaf {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn vx(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z/Vector facility available via target_feature.
  unsafe {
    core::arch::asm!(
      "vx {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Element rotate-RIGHT u32 via z/Arch `verll` (M4=2, fullword elements).
///
/// `verll` is a rotate-LEFT instruction; on u32, `rol(32 - ROR) == ror(ROR)`.
/// The conversion happens in the `const` `bits` operand so emitted asm is a
/// single `verll` with the correct immediate — zero runtime cost.
#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn rotr32_via_verll<const ROR: u32>(x: i64x2) -> i64x2 {
  const {
    assert!(ROR > 0 && ROR < 32, "ROR must be in 1..32 for u32 rotate-right");
  }
  let out: i64x2;
  // SAFETY: z/Vector facility available via target_feature; immediate is
  // compile-time bounded to 1..32 by the const assertion above.
  unsafe {
    core::arch::asm!(
      "verll {out}, {x}, {bits}, 2",
      out = lateout(vreg) out,
      x = in(vreg) x,
      bits = const 32 - ROR,
      options(nomem, nostack, pure)
    );
  }
  out
}

#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn vperm(a: i64x2, b: i64x2, mask: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z/Vector facility available via target_feature.
  unsafe {
    core::arch::asm!(
      "vperm {out}, {a}, {b}, {mask}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      mask = in(vreg) mask,
      options(nomem, nostack, pure)
    );
  }
  out
}

#[inline(always)]
unsafe fn load_perm_mask(mask: &[u8; 16]) -> i64x2 {
  // SAFETY: `mask` is a fully initialized 16-byte array.
  unsafe { core::ptr::read_unaligned(mask.as_ptr().cast::<i64x2>()) }
}

impl DiagMasks {
  #[inline(always)]
  unsafe fn new() -> Self {
    Self {
      // SAFETY: each constant is a fully initialized 16-byte shuffle mask.
      rot1: unsafe { load_perm_mask(&VPERM_ROT_LEFT_1) },
      // SAFETY: each constant is a fully initialized 16-byte shuffle mask.
      rot2: unsafe { load_perm_mask(&VPERM_ROT_LEFT_2) },
      // SAFETY: each constant is a fully initialized 16-byte shuffle mask.
      rot3: unsafe { load_perm_mask(&VPERM_ROT_LEFT_3) },
    }
  }
}

#[inline(always)]
unsafe fn load_msg_quad(m: &[u32; 16], i0: u8, i1: u8, i2: u8, i3: u8) -> i64x2 {
  from_u32x4([m[i0 as usize], m[i1 as usize], m[i2 as usize], m[i3 as usize]])
}

#[inline(always)]
unsafe fn vload_u32_quad(p: *const u32) -> i64x2 {
  // SAFETY: caller ensures `p` is valid for 4 consecutive u32 values.
  unsafe { core::ptr::read_unaligned(p.cast::<i64x2>()) }
}

#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn g4(a: &mut i64x2, b: &mut i64x2, c: &mut i64x2, d: &mut i64x2, mx: i64x2, my: i64x2) {
  // SAFETY: caller guarantees z/Vector and passes disjoint row registers.
  unsafe {
    *a = vaf(vaf(*a, *b), mx);
    *d = rotr32_via_verll::<16>(vx(*d, *a));
    *c = vaf(*c, *d);
    *b = rotr32_via_verll::<12>(vx(*b, *c));
    *a = vaf(vaf(*a, *b), my);
    *d = rotr32_via_verll::<8>(vx(*d, *a));
    *c = vaf(*c, *d);
    *b = rotr32_via_verll::<7>(vx(*b, *c));
  }
}

#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn diagonalize(b: &mut i64x2, c: &mut i64x2, d: &mut i64x2, masks: &DiagMasks) {
  // SAFETY: caller guarantees z/Vector and passes disjoint row registers.
  unsafe {
    *b = vperm(*b, *b, masks.rot1);
    *c = vperm(*c, *c, masks.rot2);
    *d = vperm(*d, *d, masks.rot3);
  }
}

#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn undiagonalize(b: &mut i64x2, c: &mut i64x2, d: &mut i64x2, masks: &DiagMasks) {
  // SAFETY: caller guarantees z/Vector and passes disjoint row registers.
  unsafe {
    *b = vperm(*b, *b, masks.rot3);
    *c = vperm(*c, *c, masks.rot2);
    *d = vperm(*d, *d, masks.rot1);
  }
}

/// Blake2s z/Vector-accelerated compress.
///
/// # Safety
///
/// Caller must ensure the z13+ vector facility is available.
#[target_feature(enable = "vector")]
pub(super) unsafe fn compress_vector(h: &mut [u32; 8], block: &[u8; 64], t: u64, last: bool) {
  // SAFETY: caller guarantees z/Vector and valid state/block buffers.
  unsafe {
    let m = load_msg(block);
    let v = init_v(h, t, last);
    let masks = DiagMasks::new();

    let mut a = vload_u32_quad(v.as_ptr());
    let mut b = vload_u32_quad(v.as_ptr().add(4));
    let mut c = vload_u32_quad(v.as_ptr().add(8));
    let mut d = vload_u32_quad(v.as_ptr().add(12));

    for round in 0..10u8 {
      let s = &SIGMA[round as usize];

      let mx = load_msg_quad(&m, s[0], s[2], s[4], s[6]);
      let my = load_msg_quad(&m, s[1], s[3], s[5], s[7]);
      g4(&mut a, &mut b, &mut c, &mut d, mx, my);

      diagonalize(&mut b, &mut c, &mut d, &masks);

      let mx = load_msg_quad(&m, s[8], s[10], s[12], s[14]);
      let my = load_msg_quad(&m, s[9], s[11], s[13], s[15]);
      g4(&mut a, &mut b, &mut c, &mut d, mx, my);

      undiagonalize(&mut b, &mut c, &mut d, &masks);
    }

    let h0 = vload_u32_quad(h.as_ptr());
    let h1 = vload_u32_quad(h.as_ptr().add(4));
    let r0 = vx(h0, vx(a, c));
    let r1 = vx(h1, vx(b, d));
    core::ptr::write_unaligned(h.as_mut_ptr().cast::<i64x2>(), r0);
    core::ptr::write_unaligned(h.as_mut_ptr().add(4).cast::<i64x2>(), r1);
  }
}

#[cfg(test)]
mod tests {
  use super::{DiagMasks, diagonalize, rotr32_via_verll, to_u32x4, undiagonalize};

  #[target_feature(enable = "vector")]
  unsafe fn assert_diagonalize_matches_blake2s_lane_rotation() {
    // SAFETY: the helper itself requires the s390x vector facility.
    let masks = unsafe { DiagMasks::new() };
    let mut b = super::from_u32x4([4, 5, 6, 7]);
    let mut c = super::from_u32x4([8, 9, 10, 11]);
    let mut d = super::from_u32x4([12, 13, 14, 15]);
    let original = (b, c, d);

    // SAFETY: the helper itself requires the s390x vector facility.
    unsafe {
      diagonalize(&mut b, &mut c, &mut d, &masks);
    }
    assert_eq!(to_u32x4(b), [5, 6, 7, 4]);
    assert_eq!(to_u32x4(c), [10, 11, 8, 9]);
    assert_eq!(to_u32x4(d), [15, 12, 13, 14]);

    // SAFETY: the helper itself requires the s390x vector facility.
    unsafe {
      undiagonalize(&mut b, &mut c, &mut d, &masks);
    }
    assert_eq!((b, c, d), original);
  }

  #[test]
  fn diagonalize_matches_blake2s_lane_rotation() {
    assert!(
      std::arch::is_s390x_feature_detected!("vector"),
      "s390x vector facility is required for Blake2s diagonalize tests"
    );
    // SAFETY: the runtime check above guarantees the vector facility is available.
    unsafe { assert_diagonalize_matches_blake2s_lane_rotation() };
  }

  /// Seeds chosen to exercise every bit position and asymmetric patterns so a
  /// wrong-direction or wrong-amount rotation surfaces in the comparison.
  const ROT_SEEDS: [u32; 4] = [0x0123_4567, 0x89AB_CDEF, 0xDEAD_BEEF, 0x0000_00FF];

  #[target_feature(enable = "vector")]
  unsafe fn assert_rotr_matches_portable() {
    let input = super::from_u32x4(ROT_SEEDS);

    // SAFETY: target_feature on the enclosing function guarantees z/Vector.
    let r16 = unsafe { rotr32_via_verll::<16>(input) };
    let r12 = unsafe { rotr32_via_verll::<12>(input) };
    let r8 = unsafe { rotr32_via_verll::<8>(input) };
    let r7 = unsafe { rotr32_via_verll::<7>(input) };

    let expect = |ror: u32| -> [u32; 4] {
      [
        ROT_SEEDS[0].rotate_right(ror),
        ROT_SEEDS[1].rotate_right(ror),
        ROT_SEEDS[2].rotate_right(ror),
        ROT_SEEDS[3].rotate_right(ror),
      ]
    };

    assert_eq!(to_u32x4(r16), expect(16), "ror16 mismatch");
    assert_eq!(to_u32x4(r12), expect(12), "ror12 mismatch");
    assert_eq!(to_u32x4(r8), expect(8), "ror8 mismatch");
    assert_eq!(to_u32x4(r7), expect(7), "ror7 mismatch");
  }

  #[test]
  fn rotr32_via_verll_matches_u32_rotate_right() {
    assert!(
      std::arch::is_s390x_feature_detected!("vector"),
      "s390x vector facility is required for Blake2s rotation tests"
    );
    // SAFETY: the runtime check above guarantees the vector facility is available.
    unsafe { assert_rotr_matches_portable() };
  }
}
