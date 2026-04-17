//! Blake2b NEON-accelerated compression for AArch64.
//!
//! Each row of the 4×4 u64 working matrix is split across two `uint64x2_t`
//! registers (lo = lanes 0-1, hi = lanes 2-3). Diagonalization uses `vextq_u64`.
//! NEON is baseline on AArch64 — always available, no feature detection needed.

#![allow(clippy::cast_possible_truncation, clippy::indexing_slicing)]

use core::arch::aarch64::*;

use super::kernels::init_v;

// ─── Rotation helpers ──────────────────────────────────────────────────────

/// Rotate right by 32: swap 32-bit halves within each 64-bit lane.
#[inline(always)]
unsafe fn ror32(x: uint64x2_t) -> uint64x2_t {
  // SAFETY: reinterpret and byte-reverse operate entirely on the input NEON
  // register value without memory access or additional preconditions.
  unsafe { vreinterpretq_u64_u32(vrev64q_u32(vreinterpretq_u32_u64(x))) }
}

/// Rotate right by 24: byte shuffle.
#[inline(always)]
unsafe fn ror24(x: uint64x2_t) -> uint64x2_t {
  static ROT24: [u8; 16] = [3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10];
  // SAFETY: the shuffle table is a fixed 16-byte constant and the NEON table
  // lookup reads only from the provided register/table values.
  unsafe {
    let tbl = vld1q_u8(ROT24.as_ptr());
    vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(x), tbl))
  }
}

/// Rotate right by 16: byte shuffle.
#[inline(always)]
unsafe fn ror16(x: uint64x2_t) -> uint64x2_t {
  static ROT16: [u8; 16] = [2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9];
  // SAFETY: the shuffle table is a fixed 16-byte constant and the NEON table
  // lookup reads only from the provided register/table values.
  unsafe {
    let tbl = vld1q_u8(ROT16.as_ptr());
    vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(x), tbl))
  }
}

/// Rotate right by 63: shift-right-insert.
#[inline(always)]
unsafe fn ror63(x: uint64x2_t) -> uint64x2_t {
  // SAFETY: the shift/insert sequence operates only on the provided register
  // value and requires no memory access or aliasing assumptions.
  unsafe { vsriq_n_u64(vaddq_u64(x, x), x, 63) }
}

// ─── G function on SIMD register pairs ──────────────────────────────────

/// Perform the Blake2b G mixing on SIMD rows (2-wide).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn g2(
  a0: &mut uint64x2_t,
  a1: &mut uint64x2_t,
  b0: &mut uint64x2_t,
  b1: &mut uint64x2_t,
  c0: &mut uint64x2_t,
  c1: &mut uint64x2_t,
  d0: &mut uint64x2_t,
  d1: &mut uint64x2_t,
  mx0: uint64x2_t,
  mx1: uint64x2_t,
  my0: uint64x2_t,
  my1: uint64x2_t,
) {
  // SAFETY: all operations stay within NEON registers and the mutable
  // references point to disjoint working-row vectors owned by the caller.
  unsafe {
    // a += b + mx
    *a0 = vaddq_u64(vaddq_u64(*a0, *b0), mx0);
    *a1 = vaddq_u64(vaddq_u64(*a1, *b1), mx1);
    // d = (d ^ a) >>> 32
    *d0 = ror32(veorq_u64(*d0, *a0));
    *d1 = ror32(veorq_u64(*d1, *a1));
    // c += d
    *c0 = vaddq_u64(*c0, *d0);
    *c1 = vaddq_u64(*c1, *d1);
    // b = (b ^ c) >>> 24
    *b0 = ror24(veorq_u64(*b0, *c0));
    *b1 = ror24(veorq_u64(*b1, *c1));
    // a += b + my
    *a0 = vaddq_u64(vaddq_u64(*a0, *b0), my0);
    *a1 = vaddq_u64(vaddq_u64(*a1, *b1), my1);
    // d = (d ^ a) >>> 16
    *d0 = ror16(veorq_u64(*d0, *a0));
    *d1 = ror16(veorq_u64(*d1, *a1));
    // c += d
    *c0 = vaddq_u64(*c0, *d0);
    *c1 = vaddq_u64(*c1, *d1);
    // b = (b ^ c) >>> 63
    *b0 = ror63(veorq_u64(*b0, *c0));
    *b1 = ror63(veorq_u64(*b1, *c1));
  }
}

// ─── Diagonalize / Un-diagonalize ──────────────────────────────────────

#[inline(always)]
unsafe fn diagonalize(
  b0: &mut uint64x2_t,
  b1: &mut uint64x2_t,
  c0: &mut uint64x2_t,
  c1: &mut uint64x2_t,
  d0: &mut uint64x2_t,
  d1: &mut uint64x2_t,
) {
  // SAFETY: diagonalization permutes only the caller-provided NEON registers;
  // no memory is accessed and the mutable references are disjoint.
  unsafe {
    // B: rotate left 1: (v4,v5,v6,v7) → (v5,v6,v7,v4)
    let tb0 = *b0;
    let tb1 = *b1;
    *b0 = vextq_u64(tb0, tb1, 1);
    *b1 = vextq_u64(tb1, tb0, 1);

    // C: rotate left 2 = swap lo/hi
    core::mem::swap(c0, c1);

    // D: rotate left 3 = rotate right 1
    let td0 = *d0;
    let td1 = *d1;
    *d0 = vextq_u64(td1, td0, 1);
    *d1 = vextq_u64(td0, td1, 1);
  }
}

#[inline(always)]
unsafe fn undiagonalize(
  b0: &mut uint64x2_t,
  b1: &mut uint64x2_t,
  c0: &mut uint64x2_t,
  c1: &mut uint64x2_t,
  d0: &mut uint64x2_t,
  d1: &mut uint64x2_t,
) {
  // SAFETY: undiagonalization permutes only the caller-provided NEON registers;
  // no memory is accessed and the mutable references are disjoint.
  unsafe {
    // B: rotate right 1
    let tb0 = *b0;
    let tb1 = *b1;
    *b0 = vextq_u64(tb1, tb0, 1);
    *b1 = vextq_u64(tb0, tb1, 1);

    // C: swap back
    core::mem::swap(c0, c1);

    // D: rotate left 1
    let td0 = *d0;
    let td1 = *d1;
    *d0 = vextq_u64(td0, td1, 1);
    *d1 = vextq_u64(td1, td0, 1);
  }
}

#[inline(always)]
unsafe fn load_u64x2(block: &[u8; 128], offset: usize) -> uint64x2_t {
  // SAFETY: `offset` is one of the fixed 16-byte chunk starts inside the
  // 128-byte Blake2b block, and `vld1q_u8` has no alignment requirement.
  unsafe { vreinterpretq_u64_u8(vld1q_u8(block.as_ptr().add(offset))) }
}

#[inline(always)]
unsafe fn lo2(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
  // SAFETY: the lane extract/combine operations stay within the provided
  // NEON registers and do not access memory.
  unsafe { vcombine_u64(vget_low_u64(a), vget_low_u64(b)) }
}

#[inline(always)]
unsafe fn hi2(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
  // SAFETY: the lane extract/combine operations stay within the provided
  // NEON registers and do not access memory.
  unsafe { vcombine_u64(vget_high_u64(a), vget_high_u64(b)) }
}

#[inline(always)]
unsafe fn lo_hi(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
  // SAFETY: the lane extract/combine operations stay within the provided
  // NEON registers and do not access memory.
  unsafe { vcombine_u64(vget_low_u64(a), vget_high_u64(b)) }
}

#[inline(always)]
unsafe fn ext1(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
  // SAFETY: the extract intrinsic operates only on the provided registers and
  // the immediate lane count is in range.
  unsafe { vextq_u64(a, b, 1) }
}

macro_rules! blake2b_round {
  (
    $a0:ident, $a1:ident, $b0:ident, $b1:ident, $c0:ident, $c1:ident, $d0:ident, $d1:ident;
    $mx0:expr, $mx1:expr; $my0:expr, $my1:expr;
    $nx0:expr, $nx1:expr; $ny0:expr, $ny1:expr
  ) => {{
    let mx0 = $mx0;
    let mx1 = $mx1;
    let my0 = $my0;
    let my1 = $my1;
    g2(
      &mut $a0, &mut $a1, &mut $b0, &mut $b1, &mut $c0, &mut $c1, &mut $d0, &mut $d1, mx0, mx1, my0, my1,
    );
    diagonalize(&mut $b0, &mut $b1, &mut $c0, &mut $c1, &mut $d0, &mut $d1);

    let mx0 = $nx0;
    let mx1 = $nx1;
    let my0 = $ny0;
    let my1 = $ny1;
    g2(
      &mut $a0, &mut $a1, &mut $b0, &mut $b1, &mut $c0, &mut $c1, &mut $d0, &mut $d1, mx0, mx1, my0, my1,
    );
    undiagonalize(&mut $b0, &mut $b1, &mut $c0, &mut $c1, &mut $d0, &mut $d1);
  }};
}

// ─── Compress entry point ──────────────────────────────────────────────

/// Blake2b NEON-accelerated compress.
///
/// # Safety
///
/// Caller must ensure NEON is available (baseline on AArch64).
#[target_feature(enable = "neon")]
pub(super) unsafe fn compress_neon(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  // SAFETY: NEON is required by the target_feature on this function; all
  // pointer-based loads/stores stay within the provided Blake2b state/block.
  unsafe {
    let v = init_v(h, t, last);
    let m0 = load_u64x2(block, 0);
    let m1 = load_u64x2(block, 16);
    let m2 = load_u64x2(block, 32);
    let m3 = load_u64x2(block, 48);
    let m4 = load_u64x2(block, 64);
    let m5 = load_u64x2(block, 80);
    let m6 = load_u64x2(block, 96);
    let m7 = load_u64x2(block, 112);

    // Pack into 2-wide SIMD rows
    let mut a0 = vld1q_u64(v.as_ptr());
    let mut a1 = vld1q_u64(v.as_ptr().add(2));
    let mut b0 = vld1q_u64(v.as_ptr().add(4));
    let mut b1 = vld1q_u64(v.as_ptr().add(6));
    let mut c0 = vld1q_u64(v.as_ptr().add(8));
    let mut c1 = vld1q_u64(v.as_ptr().add(10));
    let mut d0 = vld1q_u64(v.as_ptr().add(12));
    let mut d1 = vld1q_u64(v.as_ptr().add(14));

    blake2b_round!(
      a0, a1, b0, b1, c0, c1, d0, d1;
      lo2(m0, m1), lo2(m2, m3); hi2(m0, m1), hi2(m2, m3);
      lo2(m4, m5), lo2(m6, m7); hi2(m4, m5), hi2(m6, m7)
    );

    blake2b_round!(
      a0, a1, b0, b1, c0, c1, d0, d1;
      lo2(m7, m2), hi2(m4, m6); lo2(m5, m4), ext1(m7, m3);
      ext1(m0, m0), hi2(m5, m2); lo2(m6, m1), hi2(m3, m1)
    );

    blake2b_round!(
      a0, a1, b0, b1, c0, c1, d0, d1;
      ext1(m5, m6), hi2(m2, m7); lo2(m4, m0), lo_hi(m1, m6);
      lo_hi(m5, m1), hi2(m3, m4); lo2(m7, m3), ext1(m0, m2)
    );

    blake2b_round!(
      a0, a1, b0, b1, c0, c1, d0, d1;
      hi2(m3, m1), hi2(m6, m5); hi2(m4, m0), lo2(m6, m7);
      lo_hi(m1, m2), lo_hi(m2, m7); lo2(m3, m5), lo2(m0, m4)
    );

    blake2b_round!(
      a0, a1, b0, b1, c0, c1, d0, d1;
      hi2(m4, m2), lo2(m1, m5); lo_hi(m0, m3), lo_hi(m2, m7);
      lo_hi(m7, m5), lo_hi(m3, m1); ext1(m0, m6), lo_hi(m4, m6)
    );

    blake2b_round!(
      a0, a1, b0, b1, c0, c1, d0, d1;
      lo2(m1, m3), lo2(m0, m4); lo2(m6, m5), hi2(m5, m1);
      lo_hi(m2, m3), hi2(m7, m0); hi2(m6, m2), lo_hi(m7, m4)
    );

    blake2b_round!(
      a0, a1, b0, b1, c0, c1, d0, d1;
      lo_hi(m6, m0), lo2(m7, m2); hi2(m2, m7), ext1(m6, m5);
      lo2(m0, m3), ext1(m4, m4); hi2(m3, m1), lo_hi(m1, m5)
    );

    blake2b_round!(
      a0, a1, b0, b1, c0, c1, d0, d1;
      hi2(m6, m3), lo_hi(m6, m1); ext1(m5, m7), hi2(m0, m4);
      hi2(m2, m7), lo2(m4, m1); lo2(m0, m2), lo2(m3, m5)
    );

    blake2b_round!(
      a0, a1, b0, b1, c0, c1, d0, d1;
      lo2(m3, m7), ext1(m5, m0); hi2(m7, m4), ext1(m1, m4);
      m6, ext1(m0, m5); lo_hi(m1, m3), m2
    );

    blake2b_round!(
      a0, a1, b0, b1, c0, c1, d0, d1;
      lo2(m5, m4), hi2(m3, m0); lo2(m1, m2), lo_hi(m3, m2);
      hi2(m7, m4), hi2(m1, m6); ext1(m5, m7), lo2(m6, m0)
    );

    blake2b_round!(
      a0, a1, b0, b1, c0, c1, d0, d1;
      lo2(m0, m1), lo2(m2, m3); hi2(m0, m1), hi2(m2, m3);
      lo2(m4, m5), lo2(m6, m7); hi2(m4, m5), hi2(m6, m7)
    );

    blake2b_round!(
      a0, a1, b0, b1, c0, c1, d0, d1;
      lo2(m7, m2), hi2(m4, m6); lo2(m5, m4), ext1(m7, m3);
      ext1(m0, m0), hi2(m5, m2); lo2(m6, m1), hi2(m3, m1)
    );

    // Finalize: h[i] ^= v[i] ^ v[i+8]
    let h0 = vld1q_u64(h.as_ptr());
    let h1 = vld1q_u64(h.as_ptr().add(2));
    let h2 = vld1q_u64(h.as_ptr().add(4));
    let h3 = vld1q_u64(h.as_ptr().add(6));

    vst1q_u64(h.as_mut_ptr(), veorq_u64(h0, veorq_u64(a0, c0)));
    vst1q_u64(h.as_mut_ptr().add(2), veorq_u64(h1, veorq_u64(a1, c1)));
    vst1q_u64(h.as_mut_ptr().add(4), veorq_u64(h2, veorq_u64(b0, d0)));
    vst1q_u64(h.as_mut_ptr().add(6), veorq_u64(h3, veorq_u64(b1, d1)));
  }
}
