//! Blake2s NEON-accelerated compression for AArch64.
//!
//! Each row of the 4×4 u32 working matrix fits in one `uint32x4_t` register.
//! Rotation constants (16, 12, 8, 7) are the same as ChaCha20.
//! NEON is baseline on AArch64.

#![allow(clippy::cast_possible_truncation)]

use core::arch::aarch64::*;

use super::kernels::init_v;

// ─── Rotation helpers ──────────────────────────────────────────────────────

/// Rotate right by 16: swap 16-bit halves within each 32-bit lane.
#[inline(always)]
unsafe fn ror16(x: uint32x4_t) -> uint32x4_t {
  // SAFETY: reinterpret and byte-reverse operate entirely on the input NEON
  // register value without memory access or additional preconditions.
  unsafe { vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(x))) }
}

/// Rotate right by 12: shift+or.
#[inline(always)]
unsafe fn ror12(x: uint32x4_t) -> uint32x4_t {
  // SAFETY: the shift/insert sequence operates only on the provided register
  // value and requires no memory access or aliasing assumptions.
  unsafe { vsriq_n_u32(vshlq_n_u32(x, 20), x, 12) }
}

/// Rotate right by 8: byte shuffle.
#[inline(always)]
unsafe fn ror8(x: uint32x4_t) -> uint32x4_t {
  // SAFETY: the shift/or sequence operates only on the provided register
  // value and requires no memory access or aliasing assumptions.
  unsafe { vorrq_u32(vshrq_n_u32(x, 8), vshlq_n_u32(x, 24)) }
}

/// Rotate right by 7: shift+or.
#[inline(always)]
unsafe fn ror7(x: uint32x4_t) -> uint32x4_t {
  // SAFETY: the shift/insert sequence operates only on the provided register
  // value and requires no memory access or aliasing assumptions.
  unsafe { vsriq_n_u32(vshlq_n_u32(x, 25), x, 7) }
}

// ─── G function on full rows ────────────────────────────────────────────

#[inline(always)]
unsafe fn g4(
  a: &mut uint32x4_t,
  b: &mut uint32x4_t,
  c: &mut uint32x4_t,
  d: &mut uint32x4_t,
  mx: uint32x4_t,
  my: uint32x4_t,
) {
  // SAFETY: all operations stay within NEON registers and the mutable
  // references point to disjoint working-row vectors owned by the caller.
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
  // SAFETY: diagonalization permutes only the caller-provided NEON registers;
  // no memory is accessed and the mutable references are disjoint.
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
  // SAFETY: undiagonalization permutes only the caller-provided NEON registers;
  // no memory is accessed and the mutable references are disjoint.
  unsafe {
    *b = vextq_u32(*b, *b, 3);
    *c = vextq_u32(*c, *c, 2);
    *d = vextq_u32(*d, *d, 1);
  }
}

#[inline(always)]
unsafe fn load_u32x2(block: &[u8; 64], offset: usize) -> uint32x2_t {
  // SAFETY: `offset` is one of the fixed 8-byte chunk starts inside the
  // 64-byte Blake2s block, and `vld1_u8` has no alignment requirement.
  unsafe { vreinterpret_u32_u8(vld1_u8(block.as_ptr().add(offset))) }
}

#[inline(always)]
unsafe fn zip_lo(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
  // SAFETY: the zip intrinsic operates only on the provided NEON registers.
  unsafe { vzip1_u32(a, b) }
}

#[inline(always)]
unsafe fn zip_hi(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
  // SAFETY: the zip intrinsic operates only on the provided NEON registers.
  unsafe { vzip2_u32(a, b) }
}

#[inline(always)]
unsafe fn ext1(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
  // SAFETY: the extract intrinsic operates only on the provided NEON
  // registers and the immediate lane count is in range.
  unsafe { vext_u32(a, b, 1) }
}

#[inline(always)]
unsafe fn rev64_2(x: uint32x2_t) -> uint32x2_t {
  // SAFETY: the byte-lane reverse operates only on the provided NEON register.
  unsafe { vrev64_u32(x) }
}

#[inline(always)]
unsafe fn concat(a: uint32x2_t, b: uint32x2_t) -> uint32x4_t {
  // SAFETY: concatenation operates only on the provided NEON registers.
  unsafe { vcombine_u32(a, b) }
}

macro_rules! blake2s_round {
  ($a:ident, $b:ident, $c:ident, $d:ident; $mx1:expr, $my1:expr; $mx2:expr, $my2:expr) => {{
    let mx = $mx1;
    let my = $my1;
    // SAFETY: the caller supplies valid NEON row registers and message vectors.
    g4(&mut $a, &mut $b, &mut $c, &mut $d, mx, my);
    // SAFETY: row permutations stay within the working matrix registers.
    diagonalize(&mut $b, &mut $c, &mut $d);

    let mx = $mx2;
    let my = $my2;
    // SAFETY: the caller supplies valid NEON row registers and message vectors.
    g4(&mut $a, &mut $b, &mut $c, &mut $d, mx, my);
    // SAFETY: row permutations stay within the working matrix registers.
    undiagonalize(&mut $b, &mut $c, &mut $d);
  }};
}

// ─── Compress entry point ──────────────────────────────────────────────

/// Blake2s NEON-accelerated compress.
///
/// # Safety
///
/// Caller must ensure NEON is available (baseline on AArch64).
#[target_feature(enable = "neon")]
pub(super) unsafe fn compress_neon(h: &mut [u32; 8], block: &[u8; 64], t: u64, last: bool) {
  // SAFETY: NEON is required by the target_feature on this function; all
  // pointer-based loads/stores stay within the provided Blake2s state/block.
  unsafe {
    let v = init_v(h, t, last);
    let m0 = load_u32x2(block, 0);
    let m1 = load_u32x2(block, 8);
    let m2 = load_u32x2(block, 16);
    let m3 = load_u32x2(block, 24);
    let m4 = load_u32x2(block, 32);
    let m5 = load_u32x2(block, 40);
    let m6 = load_u32x2(block, 48);
    let m7 = load_u32x2(block, 56);

    let mut a = vld1q_u32(v.as_ptr());
    let mut b = vld1q_u32(v.as_ptr().add(4));
    let mut c = vld1q_u32(v.as_ptr().add(8));
    let mut d = vld1q_u32(v.as_ptr().add(12));

    blake2s_round!(
      a,
      b,
      c,
      d;
      concat(zip_lo(m0, m1), zip_lo(m2, m3)),
      concat(zip_hi(m0, m1), zip_hi(m2, m3));
      concat(zip_lo(m4, m5), zip_lo(m6, m7)),
      concat(zip_hi(m4, m5), zip_hi(m6, m7))
    );

    blake2s_round!(
      a,
      b,
      c,
      d;
      concat(zip_lo(m7, m2), zip_hi(m4, m6)),
      concat(zip_lo(m5, m4), ext1(m7, m3));
      concat(rev64_2(m0), zip_hi(m5, m2)),
      concat(zip_lo(m6, m1), zip_hi(m3, m1))
    );

    blake2s_round!(
      a,
      b,
      c,
      d;
      concat(ext1(m5, m6), zip_hi(m2, m7)),
      concat(zip_lo(m4, m0), rev64_2(ext1(m6, m1)));
      concat(rev64_2(ext1(m1, m5)), zip_hi(m3, m4)),
      concat(zip_lo(m7, m3), ext1(m0, m2))
    );

    blake2s_round!(
      a,
      b,
      c,
      d;
      concat(zip_hi(m3, m1), zip_hi(m6, m5)),
      concat(zip_hi(m4, m0), zip_lo(m6, m7));
      concat(rev64_2(ext1(m2, m1)), rev64_2(ext1(m7, m2))),
      concat(zip_lo(m3, m5), zip_lo(m0, m4))
    );

    blake2s_round!(
      a,
      b,
      c,
      d;
      concat(zip_hi(m4, m2), zip_lo(m1, m5)),
      concat(rev64_2(ext1(m3, m0)), rev64_2(ext1(m7, m2)));
      concat(rev64_2(ext1(m5, m7)), rev64_2(ext1(m1, m3))),
      concat(ext1(m0, m6), rev64_2(ext1(m6, m4)))
    );

    blake2s_round!(
      a,
      b,
      c,
      d;
      concat(zip_lo(m1, m3), zip_lo(m0, m4)),
      concat(zip_lo(m6, m5), zip_hi(m5, m1));
      concat(rev64_2(ext1(m3, m2)), zip_hi(m7, m0)),
      concat(zip_hi(m6, m2), rev64_2(ext1(m4, m7)))
    );

    blake2s_round!(
      a,
      b,
      c,
      d;
      concat(rev64_2(ext1(m0, m6)), zip_lo(m7, m2)),
      concat(zip_hi(m2, m7), ext1(m6, m5));
      concat(zip_lo(m0, m3), rev64_2(m4)),
      concat(zip_hi(m3, m1), rev64_2(ext1(m5, m1)))
    );

    blake2s_round!(
      a,
      b,
      c,
      d;
      concat(zip_hi(m6, m3), rev64_2(ext1(m1, m6))),
      concat(ext1(m5, m7), zip_hi(m0, m4));
      concat(zip_hi(m2, m7), zip_lo(m4, m1)),
      concat(zip_lo(m0, m2), zip_lo(m3, m5))
    );

    blake2s_round!(
      a,
      b,
      c,
      d;
      concat(zip_lo(m3, m7), ext1(m5, m0)),
      concat(zip_hi(m7, m4), ext1(m1, m4));
      concat(m6, ext1(m0, m5)),
      concat(rev64_2(ext1(m3, m1)), m2)
    );

    blake2s_round!(
      a,
      b,
      c,
      d;
      concat(zip_lo(m5, m4), zip_hi(m3, m0)),
      concat(zip_lo(m1, m2), rev64_2(ext1(m2, m3)));
      concat(zip_hi(m7, m4), zip_hi(m1, m6)),
      concat(ext1(m5, m7), zip_lo(m6, m0))
    );

    // Finalize: h[i] ^= v[i] ^ v[i+8]
    let h0 = vld1q_u32(h.as_ptr());
    let h1 = vld1q_u32(h.as_ptr().add(4));
    vst1q_u32(h.as_mut_ptr(), veorq_u32(h0, veorq_u32(a, c)));
    vst1q_u32(h.as_mut_ptr().add(4), veorq_u32(h1, veorq_u32(b, d)));
  }
}
