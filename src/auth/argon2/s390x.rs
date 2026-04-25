//! IBM Z (s390x) z/Vector BlaMka compression kernel for Argon2.
//!
//! Mirrors the Blake2b s390x kernel layout: u64 add via `vag`, XOR via
//! `vx`, and the four BlaMka rotations (32, 24, 16, 63) via `verllg`
//! immediates (`32`, `40`, `48`, `1` for ROR-equivalent ROL counts).
//!
//! # BlaMka multiply
//!
//! `2 · lsb(a) · lsb(b)` lane-wise. The base z13 vector facility has
//! no native `u64 × u64 → u64` instruction (only word-width
//! multiply-even/odd at 32×32→64). Rather than encode the cross-lane
//! VPMULUDQ-style pattern, the multiply step extracts each lane to
//! scalar and reuses the host's 64-bit `mlgr`/`msgr` integer multiply
//! — the dependency chain is short and the resulting code stays
//! readable. If a future z16+ kernel needs the SIMD multiply, replace
//! `bla_mul` only.
//!
//! # Endianness
//!
//! s390x is big-endian. Memory layout for `[u64; 2]` aligned through
//! `core::ptr::read_unaligned`/`write_unaligned` already accounts for
//! native byte order, so no explicit byte swap is needed.
//!
//! # Safety
//!
//! Requires the z13+ vector facility. Caller must verify
//! `s390x::VECTOR`.

#![cfg(target_arch = "s390x")]
#![allow(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use core::simd::i64x2;

use super::BLOCK_WORDS;

// ─── Inline-asm primitives (z13+ vector facility) ──────────────────────────

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

/// `verllg` ROL by 32 = ROR 32.
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

/// `verllg` ROL by 40 = ROR 24.
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

/// `verllg` ROL by 48 = ROR 16.
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

/// `verllg` ROL by 1 = ROR 63.
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

#[inline(always)]
fn pair_a1_b0(a: i64x2, b: i64x2) -> i64x2 {
  core::simd::simd_swizzle!(a, b, [1, 2])
}

#[inline(always)]
fn pair_b1_a0(a: i64x2, b: i64x2) -> i64x2 {
  core::simd::simd_swizzle!(a, b, [3, 0])
}

#[inline(always)]
unsafe fn vload_pair(p: *const u64) -> i64x2 {
  // SAFETY: caller ensures p is valid for 2 × u64.
  unsafe { core::ptr::read_unaligned(p as *const i64x2) }
}

#[inline(always)]
unsafe fn vstore_pair(p: *mut u64, v: i64x2) {
  // SAFETY: caller ensures p is valid for 2 × u64.
  unsafe { core::ptr::write_unaligned(p as *mut i64x2, v) }
}

/// `2 · lsb(a) · lsb(b)` lane-wise — scalar fallback (z13 has no native
/// u64×u64→u64 vector multiply; the masked u32 product fits in u64 so
/// the math is exact).
#[inline(always)]
fn bla_mul(a: i64x2, b: i64x2) -> i64x2 {
  let aa = a.to_array();
  let bb = b.to_array();
  const MASK: u64 = 0xffff_ffff;
  let r0 = ((aa[0] as u64) & MASK)
    .wrapping_mul((bb[0] as u64) & MASK)
    .wrapping_shl(1);
  let r1 = ((aa[1] as u64) & MASK)
    .wrapping_mul((bb[1] as u64) & MASK)
    .wrapping_shl(1);
  i64x2::from_array([r0 as i64, r1 as i64])
}

// ─── 4-way P-round ─────────────────────────────────────────────────────────

#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "vector")]
unsafe fn p_round(
  a_lo: &mut i64x2,
  a_hi: &mut i64x2,
  b_lo: &mut i64x2,
  b_hi: &mut i64x2,
  c_lo: &mut i64x2,
  c_hi: &mut i64x2,
  d_lo: &mut i64x2,
  d_hi: &mut i64x2,
) {
  // SAFETY: vector facility inherited.
  unsafe {
    gb(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi, d_lo, d_hi);

    let tb_lo = *b_lo;
    let tb_hi = *b_hi;
    *b_lo = pair_a1_b0(tb_lo, tb_hi);
    *b_hi = pair_b1_a0(tb_lo, tb_hi);

    core::mem::swap(c_lo, c_hi);

    let td_lo = *d_lo;
    let td_hi = *d_hi;
    *d_lo = pair_b1_a0(td_lo, td_hi);
    *d_hi = pair_a1_b0(td_lo, td_hi);

    gb(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi, d_lo, d_hi);

    let tb_lo = *b_lo;
    let tb_hi = *b_hi;
    *b_lo = pair_b1_a0(tb_lo, tb_hi);
    *b_hi = pair_a1_b0(tb_lo, tb_hi);

    core::mem::swap(c_lo, c_hi);

    let td_lo = *d_lo;
    let td_hi = *d_hi;
    *d_lo = pair_a1_b0(td_lo, td_hi);
    *d_hi = pair_b1_a0(td_lo, td_hi);
  }
}

// ─── 4-way BlaMka G ────────────────────────────────────────────────────────

#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "vector")]
unsafe fn gb(
  a_lo: &mut i64x2,
  a_hi: &mut i64x2,
  b_lo: &mut i64x2,
  b_hi: &mut i64x2,
  c_lo: &mut i64x2,
  c_hi: &mut i64x2,
  d_lo: &mut i64x2,
  d_hi: &mut i64x2,
) {
  // SAFETY: vector facility inherited.
  unsafe {
    // Step 1
    let p_lo = bla_mul(*a_lo, *b_lo);
    let p_hi = bla_mul(*a_hi, *b_hi);
    *a_lo = vag(vag(*a_lo, *b_lo), p_lo);
    *a_hi = vag(vag(*a_hi, *b_hi), p_hi);
    *d_lo = verllg_32(vx(*d_lo, *a_lo));
    *d_hi = verllg_32(vx(*d_hi, *a_hi));

    // Step 2
    let p_lo = bla_mul(*c_lo, *d_lo);
    let p_hi = bla_mul(*c_hi, *d_hi);
    *c_lo = vag(vag(*c_lo, *d_lo), p_lo);
    *c_hi = vag(vag(*c_hi, *d_hi), p_hi);
    *b_lo = verllg_40(vx(*b_lo, *c_lo));
    *b_hi = verllg_40(vx(*b_hi, *c_hi));

    // Step 3
    let p_lo = bla_mul(*a_lo, *b_lo);
    let p_hi = bla_mul(*a_hi, *b_hi);
    *a_lo = vag(vag(*a_lo, *b_lo), p_lo);
    *a_hi = vag(vag(*a_hi, *b_hi), p_hi);
    *d_lo = verllg_48(vx(*d_lo, *a_lo));
    *d_hi = verllg_48(vx(*d_hi, *a_hi));

    // Step 4
    let p_lo = bla_mul(*c_lo, *d_lo);
    let p_hi = bla_mul(*c_hi, *d_hi);
    *c_lo = vag(vag(*c_lo, *d_lo), p_lo);
    *c_hi = vag(vag(*c_hi, *d_hi), p_hi);
    *b_lo = verllg_1(vx(*b_lo, *c_lo));
    *b_hi = verllg_1(vx(*b_hi, *c_hi));
  }
}

// ─── Compress entry point ──────────────────────────────────────────────────

/// IBM Z z/Vector BlaMka compression kernel.
///
/// # Safety
///
/// - `target_arch = "s390x"` enforced at compile time (module gate).
/// - Caller must have the z13+ vector facility — enforced via `#[target_feature(enable =
///   "vector")]`.
/// - The three `[u64; BLOCK_WORDS]` buffers are fixed-size arrays.
#[target_feature(enable = "vector")]
pub(super) unsafe fn compress_vector(
  dst: &mut [u64; BLOCK_WORDS],
  x: &[u64; BLOCK_WORDS],
  y: &[u64; BLOCK_WORDS],
  xor_into: bool,
) {
  // SAFETY: vector facility enabled by target_feature; pointer-based
  // loads / stores stay within fixed-size arrays.
  unsafe {
    let mut r = [0u64; BLOCK_WORDS];
    let mut q = [0u64; BLOCK_WORDS];
    let mut i = 0;
    while i < BLOCK_WORDS {
      let xv = vload_pair(x.as_ptr().add(i));
      let yv = vload_pair(y.as_ptr().add(i));
      let rv = vx(xv, yv);
      vstore_pair(r.as_mut_ptr().add(i), rv);
      vstore_pair(q.as_mut_ptr().add(i), rv);
      i += 2;
    }

    // Row pass.
    let mut row = 0usize;
    while row < 8 {
      let base = row * 16;
      let mut a_lo = vload_pair(q.as_ptr().add(base));
      let mut a_hi = vload_pair(q.as_ptr().add(base + 2));
      let mut b_lo = vload_pair(q.as_ptr().add(base + 4));
      let mut b_hi = vload_pair(q.as_ptr().add(base + 6));
      let mut c_lo = vload_pair(q.as_ptr().add(base + 8));
      let mut c_hi = vload_pair(q.as_ptr().add(base + 10));
      let mut d_lo = vload_pair(q.as_ptr().add(base + 12));
      let mut d_hi = vload_pair(q.as_ptr().add(base + 14));

      p_round(
        &mut a_lo, &mut a_hi, &mut b_lo, &mut b_hi, &mut c_lo, &mut c_hi, &mut d_lo, &mut d_hi,
      );

      vstore_pair(q.as_mut_ptr().add(base), a_lo);
      vstore_pair(q.as_mut_ptr().add(base + 2), a_hi);
      vstore_pair(q.as_mut_ptr().add(base + 4), b_lo);
      vstore_pair(q.as_mut_ptr().add(base + 6), b_hi);
      vstore_pair(q.as_mut_ptr().add(base + 8), c_lo);
      vstore_pair(q.as_mut_ptr().add(base + 10), c_hi);
      vstore_pair(q.as_mut_ptr().add(base + 12), d_lo);
      vstore_pair(q.as_mut_ptr().add(base + 14), d_hi);
      row += 1;
    }

    // Column pass.
    let mut col = 0usize;
    while col < 8 {
      let base = col * 2;
      let mut a_lo = vload_pair(q.as_ptr().add(base));
      let mut a_hi = vload_pair(q.as_ptr().add(base + 16));
      let mut b_lo = vload_pair(q.as_ptr().add(base + 32));
      let mut b_hi = vload_pair(q.as_ptr().add(base + 48));
      let mut c_lo = vload_pair(q.as_ptr().add(base + 64));
      let mut c_hi = vload_pair(q.as_ptr().add(base + 80));
      let mut d_lo = vload_pair(q.as_ptr().add(base + 96));
      let mut d_hi = vload_pair(q.as_ptr().add(base + 112));

      p_round(
        &mut a_lo, &mut a_hi, &mut b_lo, &mut b_hi, &mut c_lo, &mut c_hi, &mut d_lo, &mut d_hi,
      );

      vstore_pair(q.as_mut_ptr().add(base), a_lo);
      vstore_pair(q.as_mut_ptr().add(base + 16), a_hi);
      vstore_pair(q.as_mut_ptr().add(base + 32), b_lo);
      vstore_pair(q.as_mut_ptr().add(base + 48), b_hi);
      vstore_pair(q.as_mut_ptr().add(base + 64), c_lo);
      vstore_pair(q.as_mut_ptr().add(base + 80), c_hi);
      vstore_pair(q.as_mut_ptr().add(base + 96), d_lo);
      vstore_pair(q.as_mut_ptr().add(base + 112), d_hi);
      col += 1;
    }

    // Final XOR with R, fused with dst store/xor.
    let mut i = 0;
    while i < BLOCK_WORDS {
      let qv = vload_pair(q.as_ptr().add(i));
      let rv = vload_pair(r.as_ptr().add(i));
      let f = vx(qv, rv);
      if xor_into {
        let cur = vload_pair(dst.as_ptr().add(i));
        vstore_pair(dst.as_mut_ptr().add(i), vx(cur, f));
      } else {
        vstore_pair(dst.as_mut_ptr().add(i), f);
      }
      i += 2;
    }
  }
}
