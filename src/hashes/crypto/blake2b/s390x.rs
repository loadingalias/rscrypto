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

/// Permute doubleword immediate: `vpdi` with M3 mask.
///
/// `vpdi v1, v2, v3, M3` selects one doubleword from each source:
///   - bit 0 of M3: selects element from v2 (0=high, 1=low)
///   - bit 2 of M3: selects element from v3 (0=high, 1=low)
///
/// M3=0: [v2[0], v3[0]]  (both high elements)
/// M3=1: [v2[1], v3[0]]  (low of a, high of b)
/// M3=4: [v2[0], v3[1]]  (high of a, low of b)
/// M3=5: [v2[1], v3[1]]  (both low elements)
#[inline(always)]
#[target_feature(enable = "vector")]
unsafe fn vpdi_0(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "vpdi {out}, {a}, {b}, 0",
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
unsafe fn vpdi_1(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "vpdi {out}, {a}, {b}, 1",
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
unsafe fn vpdi_4(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "vpdi {out}, {a}, {b}, 4",
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
unsafe fn vpdi_5(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z13+ vector facility via target_feature.
  unsafe {
    core::arch::asm!(
      "vpdi {out}, {a}, {b}, 5",
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
  // a += b + mx
  *a0 = vag(vag(*a0, *b0), mx0);
  *a1 = vag(vag(*a1, *b1), mx1);
  // d = (d ^ a) >>> 32
  *d0 = verllg_32(vx(*d0, *a0));
  *d1 = verllg_32(vx(*d1, *a1));
  // c += d
  *c0 = vag(*c0, *d0);
  *c1 = vag(*c1, *d1);
  // b = (b ^ c) >>> 24
  *b0 = verllg_40(vx(*b0, *c0));
  *b1 = verllg_40(vx(*b1, *c1));
  // a += b + my
  *a0 = vag(vag(*a0, *b0), my0);
  *a1 = vag(vag(*a1, *b1), my1);
  // d = (d ^ a) >>> 16
  *d0 = verllg_48(vx(*d0, *a0));
  *d1 = verllg_48(vx(*d1, *a1));
  // c += d
  *c0 = vag(*c0, *d0);
  *c1 = vag(*c1, *d1);
  // b = (b ^ c) >>> 63
  *b0 = verllg_1(vx(*b0, *c0));
  *b1 = verllg_1(vx(*b1, *c1));
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
  // B: rotate left 1: (v4,v5,v6,v7) -> (v5,v6,v7,v4)
  // vpdi_1(b0, b1) = [b0[1], b1[0]] = [v5, v6]
  // vpdi_1(b1, b0) = [b1[1], b0[0]] = [v7, v4]
  let tb0 = *b0;
  let tb1 = *b1;
  *b0 = vpdi_1(tb0, tb1);
  *b1 = vpdi_1(tb1, tb0);

  // C: rotate left 2 = swap lo/hi
  core::mem::swap(c0, c1);

  // D: rotate left 3 = rotate right 1: (v12,v13,v14,v15) -> (v15,v12,v13,v14)
  // vpdi_4(d1, d0) = [d1[0], d0[1]] = [v14, v13] -- wrong, need [v15, v12]
  // vpdi_1(d1, d0) = [d1[1], d0[0]] = [v15, v12]
  // vpdi_1(d0, d1) = [d0[1], d1[0]] = [v13, v14]
  let td0 = *d0;
  let td1 = *d1;
  *d0 = vpdi_1(td1, td0);
  *d1 = vpdi_1(td0, td1);
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
  // B: rotate right 1 (undo left 1)
  // Current: b0=[v5,v6], b1=[v7,v4]
  // Want:    b0=[v4,v5], b1=[v6,v7]
  // vpdi_1(b1, b0) = [b1[1], b0[0]] = [v4, v5]
  // vpdi_1(b0, b1) = [b0[1], b1[0]] = [v6, v7]
  let tb0 = *b0;
  let tb1 = *b1;
  *b0 = vpdi_1(tb1, tb0);
  *b1 = vpdi_1(tb0, tb1);

  // C: swap back
  core::mem::swap(c0, c1);

  // D: rotate left 1 (undo right 1)
  // Current: d0=[v15,v12], d1=[v13,v14]
  // Want:    d0=[v12,v13], d1=[v14,v15]
  // vpdi_1(d0, d1) = [d0[1], d1[0]] = [v12, v13]
  // vpdi_1(d1, d0) = [d1[1], d0[0]] = [v14, v15]
  let td0 = *d0;
  let td1 = *d1;
  *d0 = vpdi_1(td0, td1);
  *d1 = vpdi_1(td1, td0);
}

// ─── Compress entry point ─────────────────────────────────────────────────

/// Blake2b z/Vector-accelerated compress.
///
/// # Safety
///
/// Caller must ensure the z13+ vector facility is available.
#[target_feature(enable = "vector")]
pub(super) unsafe fn compress_vector(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  let m = load_msg(block);
  let v = init_v(h, t, last);

  // Pack into 2-wide SIMD rows: (lo, hi) for each row
  // SAFETY: v is a [u64; 16] — pointer arithmetic is within bounds.
  let mut a0 = unsafe { vload_u64_pair(v.as_ptr()) }; // v[0], v[1]
  let mut a1 = unsafe { vload_u64_pair(v.as_ptr().add(2)) }; // v[2], v[3]
  let mut b0 = unsafe { vload_u64_pair(v.as_ptr().add(4)) }; // v[4], v[5]
  let mut b1 = unsafe { vload_u64_pair(v.as_ptr().add(6)) }; // v[6], v[7]
  let mut c0 = unsafe { vload_u64_pair(v.as_ptr().add(8)) }; // v[8], v[9]
  let mut c1 = unsafe { vload_u64_pair(v.as_ptr().add(10)) }; // v[10], v[11]
  let mut d0 = unsafe { vload_u64_pair(v.as_ptr().add(12)) }; // v[12], v[13]
  let mut d1 = unsafe { vload_u64_pair(v.as_ptr().add(14)) }; // v[14], v[15]

  // 12 rounds
  for round in 0..12u8 {
    let s = &SIGMA[(round % 10) as usize];

    // Column step
    let mx0 = load_msg_pair(&m, s[0], s[2]);
    let mx1 = load_msg_pair(&m, s[4], s[6]);
    let my0 = load_msg_pair(&m, s[1], s[3]);
    let my1 = load_msg_pair(&m, s[5], s[7]);

    g2(
      &mut a0, &mut a1, &mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1, mx0, mx1, my0, my1,
    );

    diagonalize(&mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1);

    // Diagonal step
    let mx0 = load_msg_pair(&m, s[8], s[10]);
    let mx1 = load_msg_pair(&m, s[12], s[14]);
    let my0 = load_msg_pair(&m, s[9], s[11]);
    let my1 = load_msg_pair(&m, s[13], s[15]);

    g2(
      &mut a0, &mut a1, &mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1, mx0, mx1, my0, my1,
    );

    undiagonalize(&mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1);
  }

  // Finalize: h[i] ^= v[i] ^ v[i+8]
  // SAFETY: h is a [u64; 8] — pointer arithmetic is within bounds.
  let h0 = unsafe { vload_u64_pair(h.as_ptr()) };
  let h1 = unsafe { vload_u64_pair(h.as_ptr().add(2)) };
  let h2 = unsafe { vload_u64_pair(h.as_ptr().add(4)) };
  let h3 = unsafe { vload_u64_pair(h.as_ptr().add(6)) };

  let r0 = vx(h0, vx(a0, c0));
  let r1 = vx(h1, vx(a1, c1));
  let r2 = vx(h2, vx(b0, d0));
  let r3 = vx(h3, vx(b1, d1));

  // SAFETY: h is a [u64; 8] — pointer arithmetic is within bounds.
  unsafe {
    core::ptr::write_unaligned(h.as_mut_ptr() as *mut i64x2, r0);
    core::ptr::write_unaligned(h.as_mut_ptr().add(2) as *mut i64x2, r1);
    core::ptr::write_unaligned(h.as_mut_ptr().add(4) as *mut i64x2, r2);
    core::ptr::write_unaligned(h.as_mut_ptr().add(6) as *mut i64x2, r3);
  }
}
