//! Blake2s NEON-accelerated compression for AArch64.
//!
//! Each row of the 4×4 u32 working matrix fits in one `uint32x4_t` register.
//! Rotation constants (16, 12, 8, 7) are the same as ChaCha20.
//! NEON is baseline on AArch64.

#![allow(clippy::cast_possible_truncation, clippy::indexing_slicing)]

use core::arch::aarch64::*;

use super::kernels::{SIGMA, load_msg, init_v};

// ─── Rotation helpers ──────────────────────────────────────────────────────

/// Rotate right by 16: swap 16-bit halves within each 32-bit lane.
#[inline(always)]
unsafe fn ror16(x: uint32x4_t) -> uint32x4_t {
  unsafe { vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(x))) }
}

/// Rotate right by 12: shift+or.
#[inline(always)]
unsafe fn ror12(x: uint32x4_t) -> uint32x4_t {
  unsafe { vsriq_n_u32(vshlq_n_u32(x, 20), x, 12) }
}

/// Rotate right by 8: byte shuffle.
#[inline(always)]
unsafe fn ror8(x: uint32x4_t) -> uint32x4_t {
  static ROT8: [u8; 16] = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12];
  unsafe {
    let tbl = vld1q_u8(ROT8.as_ptr());
    vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(x), tbl))
  }
}

/// Rotate right by 7: shift+or.
#[inline(always)]
unsafe fn ror7(x: uint32x4_t) -> uint32x4_t {
  unsafe { vsriq_n_u32(vshlq_n_u32(x, 25), x, 7) }
}

// ─── G function on full rows ────────────────────────────────────────────

#[inline(always)]
unsafe fn g4(
  a: &mut uint32x4_t, b: &mut uint32x4_t,
  c: &mut uint32x4_t, d: &mut uint32x4_t,
  mx: uint32x4_t, my: uint32x4_t,
) {
  unsafe {
    *a = vaddq_u32(vaddq_u32(*a, *b), mx);
    *d = ror16(veorq_u32(*d, *a));
    *c = vaddq_u32(*c, *d);
    *b = ror12(veorq_u32(*b, *c));
    *a = vaddq_u32(vaddq_u32(*a, *b), my);
    *d = ror8(veorq_u32(*d, *a));
    *c = vaddq_u32(*c, *d);
    *b = ror7(veorq_u32(*b, *c));
  }
}

// ─── Diagonalize / Un-diagonalize ──────────────────────────────────────

#[inline(always)]
unsafe fn diagonalize(b: &mut uint32x4_t, c: &mut uint32x4_t, d: &mut uint32x4_t) {
  unsafe {
    // B: rotate left 1 lane
    *b = vextq_u32(*b, *b, 1);
    // C: rotate left 2 lanes
    *c = vextq_u32(*c, *c, 2);
    // D: rotate left 3 lanes = rotate right 1 lane
    *d = vextq_u32(*d, *d, 3);
  }
}

#[inline(always)]
unsafe fn undiagonalize(b: &mut uint32x4_t, c: &mut uint32x4_t, d: &mut uint32x4_t) {
  unsafe {
    *b = vextq_u32(*b, *b, 3);
    *c = vextq_u32(*c, *c, 2);
    *d = vextq_u32(*d, *d, 1);
  }
}

// ─── Compress entry point ──────────────────────────────────────────────

/// Blake2s NEON-accelerated compress.
///
/// # Safety
///
/// Caller must ensure NEON is available (baseline on AArch64).
#[target_feature(enable = "neon")]
pub(super) unsafe fn compress_neon(h: &mut [u32; 8], block: &[u8; 64], t: u64, last: bool) {
  unsafe {
    let m = load_msg(block);
    let v = init_v(h, t, last);

    let mut a = vld1q_u32(v.as_ptr());
    let mut b = vld1q_u32(v.as_ptr().add(4));
    let mut c = vld1q_u32(v.as_ptr().add(8));
    let mut d = vld1q_u32(v.as_ptr().add(12));

    for round in 0..10u8 {
      let s = &SIGMA[round as usize];

      // Column step
      let mx = load_msg_quad(&m, s[0], s[2], s[4], s[6]);
      let my = load_msg_quad(&m, s[1], s[3], s[5], s[7]);
      g4(&mut a, &mut b, &mut c, &mut d, mx, my);

      diagonalize(&mut b, &mut c, &mut d);

      // Diagonal step
      let mx = load_msg_quad(&m, s[8], s[10], s[12], s[14]);
      let my = load_msg_quad(&m, s[9], s[11], s[13], s[15]);
      g4(&mut a, &mut b, &mut c, &mut d, mx, my);

      undiagonalize(&mut b, &mut c, &mut d);
    }

    // Finalize: h[i] ^= v[i] ^ v[i+8]
    let h0 = vld1q_u32(h.as_ptr());
    let h1 = vld1q_u32(h.as_ptr().add(4));
    vst1q_u32(h.as_mut_ptr(), veorq_u32(h0, veorq_u32(a, c)));
    vst1q_u32(h.as_mut_ptr().add(4), veorq_u32(h1, veorq_u32(b, d)));
  }
}

#[inline(always)]
unsafe fn load_msg_quad(m: &[u32; 16], i0: u8, i1: u8, i2: u8, i3: u8) -> uint32x4_t {
  let quad = [m[i0 as usize], m[i1 as usize], m[i2 as usize], m[i3 as usize]];
  unsafe { vld1q_u32(quad.as_ptr()) }
}
