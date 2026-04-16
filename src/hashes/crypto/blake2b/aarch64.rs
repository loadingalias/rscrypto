//! Blake2b NEON-accelerated compression for AArch64.
//!
//! Each row of the 4×4 u64 working matrix is split across two `uint64x2_t`
//! registers (lo = lanes 0-1, hi = lanes 2-3). Diagonalization uses `vextq_u64`.
//! NEON is baseline on AArch64 — always available, no feature detection needed.

#![allow(clippy::cast_possible_truncation, clippy::indexing_slicing)]

use core::arch::aarch64::*;

use super::kernels::{SIGMA, load_msg, init_v};

// ─── Rotation helpers ──────────────────────────────────────────────────────

/// Rotate right by 32: swap 32-bit halves within each 64-bit lane.
#[inline(always)]
unsafe fn ror32(x: uint64x2_t) -> uint64x2_t {
  unsafe { vreinterpretq_u64_u32(vrev64q_u32(vreinterpretq_u32_u64(x))) }
}

/// Rotate right by 24: byte shuffle.
#[inline(always)]
unsafe fn ror24(x: uint64x2_t) -> uint64x2_t {
  static ROT24: [u8; 16] = [3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10];
  unsafe {
    let tbl = vld1q_u8(ROT24.as_ptr());
    vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(x), tbl))
  }
}

/// Rotate right by 16: byte shuffle.
#[inline(always)]
unsafe fn ror16(x: uint64x2_t) -> uint64x2_t {
  static ROT16: [u8; 16] = [2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9];
  unsafe {
    let tbl = vld1q_u8(ROT16.as_ptr());
    vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(x), tbl))
  }
}

/// Rotate right by 63: shift-right-insert.
#[inline(always)]
unsafe fn ror63(x: uint64x2_t) -> uint64x2_t {
  unsafe { vsriq_n_u64(vaddq_u64(x, x), x, 63) }
}

// ─── G function on SIMD register pairs ──────────────────────────────────

/// Perform the Blake2b G mixing on SIMD rows (2-wide).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn g2(
  a0: &mut uint64x2_t, a1: &mut uint64x2_t,
  b0: &mut uint64x2_t, b1: &mut uint64x2_t,
  c0: &mut uint64x2_t, c1: &mut uint64x2_t,
  d0: &mut uint64x2_t, d1: &mut uint64x2_t,
  mx0: uint64x2_t, mx1: uint64x2_t,
  my0: uint64x2_t, my1: uint64x2_t,
) {
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
  b0: &mut uint64x2_t, b1: &mut uint64x2_t,
  c0: &mut uint64x2_t, c1: &mut uint64x2_t,
  d0: &mut uint64x2_t, d1: &mut uint64x2_t,
) {
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
  b0: &mut uint64x2_t, b1: &mut uint64x2_t,
  c0: &mut uint64x2_t, c1: &mut uint64x2_t,
  d0: &mut uint64x2_t, d1: &mut uint64x2_t,
) {
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

// ─── Compress entry point ──────────────────────────────────────────────

/// Blake2b NEON-accelerated compress.
///
/// # Safety
///
/// Caller must ensure NEON is available (baseline on AArch64).
#[target_feature(enable = "neon")]
pub(super) unsafe fn compress_neon(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  unsafe {
    let m = load_msg(block);
    let v = init_v(h, t, last);

    // Pack into 2-wide SIMD rows
    let mut a0 = vld1q_u64(v.as_ptr());
    let mut a1 = vld1q_u64(v.as_ptr().add(2));
    let mut b0 = vld1q_u64(v.as_ptr().add(4));
    let mut b1 = vld1q_u64(v.as_ptr().add(6));
    let mut c0 = vld1q_u64(v.as_ptr().add(8));
    let mut c1 = vld1q_u64(v.as_ptr().add(10));
    let mut d0 = vld1q_u64(v.as_ptr().add(12));
    let mut d1 = vld1q_u64(v.as_ptr().add(14));

    for round in 0..12u8 {
      let s = &SIGMA[(round % 10) as usize];

      // Column step
      let mx0 = load_msg_pair(&m, s[0], s[2]);
      let mx1 = load_msg_pair(&m, s[4], s[6]);
      let my0 = load_msg_pair(&m, s[1], s[3]);
      let my1 = load_msg_pair(&m, s[5], s[7]);
      g2(&mut a0, &mut a1, &mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1, mx0, mx1, my0, my1);

      diagonalize(&mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1);

      // Diagonal step
      let mx0 = load_msg_pair(&m, s[8], s[10]);
      let mx1 = load_msg_pair(&m, s[12], s[14]);
      let my0 = load_msg_pair(&m, s[9], s[11]);
      let my1 = load_msg_pair(&m, s[13], s[15]);
      g2(&mut a0, &mut a1, &mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1, mx0, mx1, my0, my1);

      undiagonalize(&mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1);
    }

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

#[inline(always)]
unsafe fn load_msg_pair(m: &[u64; 16], i0: u8, i1: u8) -> uint64x2_t {
  let pair = [m[i0 as usize], m[i1 as usize]];
  unsafe { vld1q_u64(pair.as_ptr()) }
}
