//! Blake2b z/Vector-accelerated compression for s390x.
//!
//! Each row of the 4x4 u64 working matrix is split across two `i64x2`
//! registers (lo = lanes 0-1, hi = lanes 2-3). Diagonalization uses `vpdi`
//! (permute doubleword immediate) and lane swaps.
//!
//! All four Blake2b rotations (32, 24, 16, 63) map to single `verllg`
//! (element rotate-left doubleword) instructions via the ROL equivalents:
//!   ROR 32 = ROL 32, ROR 24 = ROL 40, ROR 16 = ROL 48, ROR 63 = ROL 1.
//!
//! # Endianness
//!
//! s390x is big-endian. The `m: [u64; 16]` array from `load_msg` contains
//! native-endian u64 values (already byte-swapped from little-endian wire
//! format). We load pairs directly from the array — no further byte-swap
//! is needed.
//!
//! # Safety
//!
//! Requires z13+ vector facility. Caller must verify `s390x::VECTOR`.

#![allow(unsafe_code)]
#![allow(clippy::cast_possible_truncation, clippy::indexing_slicing)]

use core::simd::i64x2;

use super::kernels::{SIGMA, init_v, load_msg};

// ─── Vector primitive operations (inline asm, z13+) ───────────────────────

/// Vector add u64 lanes: `vag`.
#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn vag(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
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

/// Vector XOR: `vx`.
#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn vx(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
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

/// Element rotate left u64 by immediate: `verllg`.
/// ROR 32 = ROL 32.
#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn verllg_32(x: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "verllg {out}, {x}, 32",
      out = lateout(vreg) out,
      x = in(vreg) x,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Element rotate left u64 by 40 (= ROR 24): `verllg`.
#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn verllg_40(x: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "verllg {out}, {x}, 40",
      out = lateout(vreg) out,
      x = in(vreg) x,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Element rotate left u64 by 48 (= ROR 16): `verllg`.
#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn verllg_48(x: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "verllg {out}, {x}, 48",
      out = lateout(vreg) out,
      x = in(vreg) x,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Element rotate left u64 by 1 (= ROR 63): `verllg`.
#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn verllg_1(x: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "verllg {out}, {x}, 1",
      out = lateout(vreg) out,
      x = in(vreg) x,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// `vpdi` helper for `[a[1], b[0]]`.
///
/// `vpdi` selects each output doubleword from the concatenation `[a[0], a[1], b[0], b[1]]`.
/// The high two bits of the immediate select output lane 0, and the low two bits select output
/// lane 1. `xxh3` already relies on this encoding with `vpdi ..., 4` as an in-register lane swap.
#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn vpdi_a1_b0(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "vpdi {out}, {a}, {b}, 6",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// `vpdi` helper for `[b[1], a[0]]`.
#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn vpdi_b1_a0(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "vpdi {out}, {a}, {b}, 12",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

// ─── Load helpers ─────────────────────────────────────────────────────────

/// Load a pair of u64 values from the message array into a vector register.
///
/// The message array is native-endian u64s (already decoded by `load_msg`).
#[inline(always)]
unsafe fn load_msg_pair(m: &[u64; 16], i0: u8, i1: u8) -> i64x2 {
  i64x2::from_array([m[i0 as usize] as i64, m[i1 as usize] as i64])
}

/// Load 2 consecutive u64 values from a `[u64]` slice via pointer.
#[inline(always)]
unsafe fn vload_u64_pair(p: *const u64) -> i64x2 {
  // SAFETY: caller ensures p is valid for 2 × u64.
  unsafe { core::ptr::read_unaligned(p as *const i64x2) }
}

// ─── G function on SIMD register pairs ────────────────────────────────────

/// Blake2b G mixing on SIMD rows (2-wide).
///
/// Each row is (lo, hi) = 2 × i64x2 for lanes [0,1] and [2,3].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "vector")]
unsafe fn g2(
  a0: &mut i64x2,
  a1: &mut i64x2,
  b0: &mut i64x2,
  b1: &mut i64x2,
  c0: &mut i64x2,
  c1: &mut i64x2,
  d0: &mut i64x2,
  d1: &mut i64x2,
  mx0: i64x2,
  mx1: i64x2,
  my0: i64x2,
  my1: i64x2,
) {
  // SAFETY: caller guarantees the s390x vector facility is available and all
  // inputs are local vector registers with no aliasing beyond the provided refs.
  unsafe {
    *a0 = vag(vag(*a0, *b0), mx0);
    *a1 = vag(vag(*a1, *b1), mx1);
    *d0 = verllg_32(vx(*d0, *a0));
    *d1 = verllg_32(vx(*d1, *a1));
    *c0 = vag(*c0, *d0);
    *c1 = vag(*c1, *d1);
    *b0 = verllg_40(vx(*b0, *c0));
    *b1 = verllg_40(vx(*b1, *c1));
    *a0 = vag(vag(*a0, *b0), my0);
    *a1 = vag(vag(*a1, *b1), my1);
    *d0 = verllg_48(vx(*d0, *a0));
    *d1 = verllg_48(vx(*d1, *a1));
    *c0 = vag(*c0, *d0);
    *c1 = vag(*c1, *d1);
    *b0 = verllg_1(vx(*b0, *c0));
    *b1 = verllg_1(vx(*b1, *c1));
  }
}

// ─── Diagonalize / Un-diagonalize ─────────────────────────────────────────

/// Diagonalize: rotate row B left by 1, row C by 2 (swap lo/hi), row D right by 1.
///
/// Using `vpdi` to rearrange doubleword elements between register pairs:
///   B rotate-left-1:  (b0=[0,1], b1=[2,3]) -> ([1,2], [3,0])
///   C swap:           (c0, c1) -> (c1, c0)
///   D rotate-right-1: (d0=[0,1], d1=[2,3]) -> ([3,0], [1,2])
#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn diagonalize(b0: &mut i64x2, b1: &mut i64x2, c0: &mut i64x2, c1: &mut i64x2, d0: &mut i64x2, d1: &mut i64x2) {
  // SAFETY: caller guarantees the s390x vector facility is available and all
  // inputs are local vector registers with no aliasing beyond the provided refs.
  unsafe {
    let tb0 = *b0;
    let tb1 = *b1;
    *b0 = vpdi_a1_b0(tb0, tb1);
    *b1 = vpdi_b1_a0(tb0, tb1);
    core::mem::swap(c0, c1);
    let td0 = *d0;
    let td1 = *d1;
    *d0 = vpdi_b1_a0(td0, td1);
    *d1 = vpdi_a1_b0(td0, td1);
  }
}

/// Un-diagonalize: reverse the rotations.
#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn undiagonalize(
  b0: &mut i64x2,
  b1: &mut i64x2,
  c0: &mut i64x2,
  c1: &mut i64x2,
  d0: &mut i64x2,
  d1: &mut i64x2,
) {
  // SAFETY: caller guarantees the s390x vector facility is available and all
  // inputs are local vector registers with no aliasing beyond the provided refs.
  unsafe {
    let tb0 = *b0;
    let tb1 = *b1;
    *b0 = vpdi_b1_a0(tb0, tb1);
    *b1 = vpdi_a1_b0(tb0, tb1);
    core::mem::swap(c0, c1);
    let td0 = *d0;
    let td1 = *d1;
    *d0 = vpdi_a1_b0(td0, td1);
    *d1 = vpdi_b1_a0(td0, td1);
  }
}

// ─── Compress entry point ─────────────────────────────────────────────────

/// Blake2b z/Vector-accelerated compress.
///
/// # Safety
///
/// Caller must ensure the z13+ vector facility is available.
#[target_feature(enable = "vector")]
pub(super) unsafe fn compress_vector(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  // SAFETY: caller guarantees the s390x vector facility is available and both
  // `h` and `block` are valid buffers for the full Blake2b compression step.
  unsafe {
    let m = load_msg(block);
    let v = init_v(h, t, last);
    let mut a0 = vload_u64_pair(v.as_ptr());
    let mut a1 = vload_u64_pair(v.as_ptr().add(2));
    let mut b0 = vload_u64_pair(v.as_ptr().add(4));
    let mut b1 = vload_u64_pair(v.as_ptr().add(6));
    let mut c0 = vload_u64_pair(v.as_ptr().add(8));
    let mut c1 = vload_u64_pair(v.as_ptr().add(10));
    let mut d0 = vload_u64_pair(v.as_ptr().add(12));
    let mut d1 = vload_u64_pair(v.as_ptr().add(14));

    for round in 0..12u8 {
      let s = &SIGMA[(round % 10) as usize];
      let mx0 = load_msg_pair(&m, s[0], s[2]);
      let mx1 = load_msg_pair(&m, s[4], s[6]);
      let my0 = load_msg_pair(&m, s[1], s[3]);
      let my1 = load_msg_pair(&m, s[5], s[7]);
      g2(
        &mut a0, &mut a1, &mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1, mx0, mx1, my0, my1,
      );
      diagonalize(&mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1);
      let mx0 = load_msg_pair(&m, s[8], s[10]);
      let mx1 = load_msg_pair(&m, s[12], s[14]);
      let my0 = load_msg_pair(&m, s[9], s[11]);
      let my1 = load_msg_pair(&m, s[13], s[15]);
      g2(
        &mut a0, &mut a1, &mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1, mx0, mx1, my0, my1,
      );
      undiagonalize(&mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1);
    }

    let h0 = vload_u64_pair(h.as_ptr());
    let h1 = vload_u64_pair(h.as_ptr().add(2));
    let h2 = vload_u64_pair(h.as_ptr().add(4));
    let h3 = vload_u64_pair(h.as_ptr().add(6));
    let r0 = vx(h0, vx(a0, c0));
    let r1 = vx(h1, vx(a1, c1));
    let r2 = vx(h2, vx(b0, d0));
    let r3 = vx(h3, vx(b1, d1));
    core::ptr::write_unaligned(h.as_mut_ptr() as *mut i64x2, r0);
    core::ptr::write_unaligned(h.as_mut_ptr().add(2) as *mut i64x2, r1);
    core::ptr::write_unaligned(h.as_mut_ptr().add(4) as *mut i64x2, r2);
    core::ptr::write_unaligned(h.as_mut_ptr().add(6) as *mut i64x2, r3);
  }
}

#[cfg(test)]
mod tests {
  use core::simd::i64x2;

  use super::{diagonalize, undiagonalize, vpdi_a1_b0, vpdi_b1_a0};

  #[target_feature(enable = "vector")]
  unsafe fn assert_vpdi_lane_selectors() {
    let a = i64x2::from_array([10, 11]);
    let b = i64x2::from_array([20, 21]);
    // SAFETY: the helper itself requires the s390x vector facility.
    unsafe {
      assert_eq!(vpdi_a1_b0(a, b).to_array(), [11, 20]);
      assert_eq!(vpdi_b1_a0(a, b).to_array(), [21, 10]);
    }
  }

  #[test]
  fn vpdi_lane_selectors_match_expected_pairs() {
    assert!(
      std::arch::is_s390x_feature_detected!("vector"),
      "s390x vector facility is required for Blake2b vpdi lane tests"
    );
    // SAFETY: the runtime check above guarantees the vector facility is available.
    unsafe { assert_vpdi_lane_selectors() };
  }

  #[target_feature(enable = "vector")]
  unsafe fn assert_diagonalize_round_trip() {
    let mut b0 = i64x2::from_array([4, 5]);
    let mut b1 = i64x2::from_array([6, 7]);
    let mut c0 = i64x2::from_array([8, 9]);
    let mut c1 = i64x2::from_array([10, 11]);
    let mut d0 = i64x2::from_array([12, 13]);
    let mut d1 = i64x2::from_array([14, 15]);
    let original = (b0, b1, c0, c1, d0, d1);

    // SAFETY: the helper itself requires the s390x vector facility.
    unsafe { diagonalize(&mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1) };
    assert_eq!(b0.to_array(), [5, 6]);
    assert_eq!(b1.to_array(), [7, 4]);
    assert_eq!(c0.to_array(), [10, 11]);
    assert_eq!(c1.to_array(), [8, 9]);
    assert_eq!(d0.to_array(), [15, 12]);
    assert_eq!(d1.to_array(), [13, 14]);

    // SAFETY: the helper itself requires the s390x vector facility.
    unsafe { undiagonalize(&mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1) };
    assert_eq!((b0, b1, c0, c1, d0, d1), original);
  }

  #[test]
  fn diagonalize_round_trip_restores_rows() {
    assert!(
      std::arch::is_s390x_feature_detected!("vector"),
      "s390x vector facility is required for Blake2b diagonalize tests"
    );
    // SAFETY: the runtime check above guarantees the vector facility is available.
    unsafe { assert_diagonalize_round_trip() };
  }
}
