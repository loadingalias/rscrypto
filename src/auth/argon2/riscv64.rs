//! RISC-V Vector (RVV) BlaMka compression kernel for Argon2.
//!
//! 2-u64 scalar pairs that the compiler auto-lowers to RVV at
//! `VL=2 / SEW=64` when the V extension is active. This is the same
//! contract the Blake2b RVV kernel uses; on cores without RVV the
//! sequence still inlines to plain scalar 64-bit ops.
//!
//! # Vectorisation topology
//!
//! Per RFC 9106 §3.6 a P-round operates on 16 u64 words laid out as:
//!
//! ```text
//! v = [ a0 a1 a2 a3 | b0 b1 b2 b3 | c0 c1 c2 c3 | d0 d1 d2 d3 ]
//! ```
//!
//! Column step: `GB(a_i, b_i, c_i, d_i)` for `i ∈ 0..4` runs at 4-way
//! parallelism with each `[u64; 2]` pair holding two of the four lanes
//! per row (a/b/c/d × {lo, hi}).
//!
//! Diagonal step: rotate `b` by 1, `c` by 2, `d` by 3 within each
//! 4-lane row. Implemented as plain index swaps; the compiler keeps
//! these in registers under `-O2`.
//!
//! # BlaMka multiply
//!
//! `2 · lsb(a) · lsb(b)` lane-wise via masked `wrapping_mul`. The
//! product of two 32-bit values fits in u64, so masking with
//! `0xffffffff` and one `wrapping_mul(2)` is exact.
//!
//! # Rotations
//!
//! All four (32, 24, 16, 63) use scalar `u64::rotate_right`. On RVV
//! Zvbb-equipped cores these vectorise to native vector rotates; on
//! older cores they expand to shift+or pairs that the engine
//! pipelines independently per lane.

#![cfg(target_arch = "riscv64")]
#![allow(clippy::cast_possible_truncation)]

use super::BLOCK_WORDS;

type Pair = [u64; 2];

/// RISC-V V extension BlaMka compression kernel.
///
/// # Safety
///
/// - `target_arch = "riscv64"` enforced at compile time (module gate).
/// - Caller must have the V extension available — enforced via `#[target_feature(enable = "v")]`.
/// - The three `[u64; BLOCK_WORDS]` buffers are fixed-size arrays; the kernel reads/writes only
///   within their bounds.
#[target_feature(enable = "v")]
pub(super) unsafe fn compress_rvv(
  dst: &mut [u64; BLOCK_WORDS],
  x: &[u64; BLOCK_WORDS],
  y: &[u64; BLOCK_WORDS],
  xor_into: bool,
) {
  // SAFETY: V extension is enabled by this function's `#[target_feature]`.
  // Memory accesses are explicit indices into fixed-size arrays.
  let mut r = [0u64; BLOCK_WORDS];
  let mut q = [0u64; BLOCK_WORDS];
  let mut i = 0;
  while i < BLOCK_WORDS {
    r[i] = x[i] ^ y[i];
    r[i + 1] = x[i + 1] ^ y[i + 1];
    q[i] = r[i];
    q[i + 1] = r[i + 1];
    i += 2;
  }

  // Row pass.
  let mut row = 0usize;
  while row < 8 {
    let base = row * 16;
    let mut a_lo: Pair = [q[base], q[base + 1]];
    let mut a_hi: Pair = [q[base + 2], q[base + 3]];
    let mut b_lo: Pair = [q[base + 4], q[base + 5]];
    let mut b_hi: Pair = [q[base + 6], q[base + 7]];
    let mut c_lo: Pair = [q[base + 8], q[base + 9]];
    let mut c_hi: Pair = [q[base + 10], q[base + 11]];
    let mut d_lo: Pair = [q[base + 12], q[base + 13]];
    let mut d_hi: Pair = [q[base + 14], q[base + 15]];

    p_round(
      &mut a_lo, &mut a_hi, &mut b_lo, &mut b_hi, &mut c_lo, &mut c_hi, &mut d_lo, &mut d_hi,
    );

    q[base] = a_lo[0];
    q[base + 1] = a_lo[1];
    q[base + 2] = a_hi[0];
    q[base + 3] = a_hi[1];
    q[base + 4] = b_lo[0];
    q[base + 5] = b_lo[1];
    q[base + 6] = b_hi[0];
    q[base + 7] = b_hi[1];
    q[base + 8] = c_lo[0];
    q[base + 9] = c_lo[1];
    q[base + 10] = c_hi[0];
    q[base + 11] = c_hi[1];
    q[base + 12] = d_lo[0];
    q[base + 13] = d_lo[1];
    q[base + 14] = d_hi[0];
    q[base + 15] = d_hi[1];
    row += 1;
  }

  // Column pass.
  let mut col = 0usize;
  while col < 8 {
    let base = col * 2;
    let mut a_lo: Pair = [q[base], q[base + 1]];
    let mut a_hi: Pair = [q[base + 16], q[base + 17]];
    let mut b_lo: Pair = [q[base + 32], q[base + 33]];
    let mut b_hi: Pair = [q[base + 48], q[base + 49]];
    let mut c_lo: Pair = [q[base + 64], q[base + 65]];
    let mut c_hi: Pair = [q[base + 80], q[base + 81]];
    let mut d_lo: Pair = [q[base + 96], q[base + 97]];
    let mut d_hi: Pair = [q[base + 112], q[base + 113]];

    p_round(
      &mut a_lo, &mut a_hi, &mut b_lo, &mut b_hi, &mut c_lo, &mut c_hi, &mut d_lo, &mut d_hi,
    );

    q[base] = a_lo[0];
    q[base + 1] = a_lo[1];
    q[base + 16] = a_hi[0];
    q[base + 17] = a_hi[1];
    q[base + 32] = b_lo[0];
    q[base + 33] = b_lo[1];
    q[base + 48] = b_hi[0];
    q[base + 49] = b_hi[1];
    q[base + 64] = c_lo[0];
    q[base + 65] = c_lo[1];
    q[base + 80] = c_hi[0];
    q[base + 81] = c_hi[1];
    q[base + 96] = d_lo[0];
    q[base + 97] = d_lo[1];
    q[base + 112] = d_hi[0];
    q[base + 113] = d_hi[1];
    col += 1;
  }

  // Final XOR with R, fused with dst store/xor.
  let mut i = 0;
  while i < BLOCK_WORDS {
    let f0 = q[i] ^ r[i];
    let f1 = q[i + 1] ^ r[i + 1];
    if xor_into {
      dst[i] ^= f0;
      dst[i + 1] ^= f1;
    } else {
      dst[i] = f0;
      dst[i + 1] = f1;
    }
    i += 2;
  }
}

// ─── 4-way P-round ─────────────────────────────────────────────────────────

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn p_round(
  a_lo: &mut Pair,
  a_hi: &mut Pair,
  b_lo: &mut Pair,
  b_hi: &mut Pair,
  c_lo: &mut Pair,
  c_hi: &mut Pair,
  d_lo: &mut Pair,
  d_hi: &mut Pair,
) {
  // Column step.
  gb(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi, d_lo, d_hi);

  // Diagonalise: rotate b by 1, c by 2, d by 3 across the 4-lane row.
  let tb_lo = *b_lo;
  let tb_hi = *b_hi;
  *b_lo = [tb_lo[1], tb_hi[0]];
  *b_hi = [tb_hi[1], tb_lo[0]];

  core::mem::swap(c_lo, c_hi);

  let td_lo = *d_lo;
  let td_hi = *d_hi;
  *d_lo = [td_hi[1], td_lo[0]];
  *d_hi = [td_lo[1], td_hi[0]];

  // Diagonal step.
  gb(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi, d_lo, d_hi);

  // Undo diagonalisation.
  let tb_lo = *b_lo;
  let tb_hi = *b_hi;
  *b_lo = [tb_hi[1], tb_lo[0]];
  *b_hi = [tb_lo[1], tb_hi[0]];

  core::mem::swap(c_lo, c_hi);

  let td_lo = *d_lo;
  let td_hi = *d_hi;
  *d_lo = [td_lo[1], td_hi[0]];
  *d_hi = [td_hi[1], td_lo[0]];
}

// ─── 4-way BlaMka G ────────────────────────────────────────────────────────

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn gb(
  a_lo: &mut Pair,
  a_hi: &mut Pair,
  b_lo: &mut Pair,
  b_hi: &mut Pair,
  c_lo: &mut Pair,
  c_hi: &mut Pair,
  d_lo: &mut Pair,
  d_hi: &mut Pair,
) {
  // Step 1: a = a + b + 2·lsb(a)·lsb(b)
  let p_lo = bla_mul(*a_lo, *b_lo);
  let p_hi = bla_mul(*a_hi, *b_hi);
  *a_lo = vadd(vadd(*a_lo, *b_lo), p_lo);
  *a_hi = vadd(vadd(*a_hi, *b_hi), p_hi);
  *d_lo = ror(vxor(*d_lo, *a_lo), 32);
  *d_hi = ror(vxor(*d_hi, *a_hi), 32);

  // Step 2: c = c + d + 2·lsb(c)·lsb(d)
  let p_lo = bla_mul(*c_lo, *d_lo);
  let p_hi = bla_mul(*c_hi, *d_hi);
  *c_lo = vadd(vadd(*c_lo, *d_lo), p_lo);
  *c_hi = vadd(vadd(*c_hi, *d_hi), p_hi);
  *b_lo = ror(vxor(*b_lo, *c_lo), 24);
  *b_hi = ror(vxor(*b_hi, *c_hi), 24);

  // Step 3: a = a + b + 2·lsb(a)·lsb(b)
  let p_lo = bla_mul(*a_lo, *b_lo);
  let p_hi = bla_mul(*a_hi, *b_hi);
  *a_lo = vadd(vadd(*a_lo, *b_lo), p_lo);
  *a_hi = vadd(vadd(*a_hi, *b_hi), p_hi);
  *d_lo = ror(vxor(*d_lo, *a_lo), 16);
  *d_hi = ror(vxor(*d_hi, *a_hi), 16);

  // Step 4: c = c + d + 2·lsb(c)·lsb(d)
  let p_lo = bla_mul(*c_lo, *d_lo);
  let p_hi = bla_mul(*c_hi, *d_hi);
  *c_lo = vadd(vadd(*c_lo, *d_lo), p_lo);
  *c_hi = vadd(vadd(*c_hi, *d_hi), p_hi);
  *b_lo = ror(vxor(*b_lo, *c_lo), 63);
  *b_hi = ror(vxor(*b_hi, *c_hi), 63);
}

// ─── Micro-ops (compiler-vectorised at VL=2/SEW=64 under -C target-feature=+v) ─

#[inline(always)]
fn vadd(a: Pair, b: Pair) -> Pair {
  [a[0].wrapping_add(b[0]), a[1].wrapping_add(b[1])]
}

#[inline(always)]
fn vxor(a: Pair, b: Pair) -> Pair {
  [a[0] ^ b[0], a[1] ^ b[1]]
}

#[inline(always)]
fn ror(x: Pair, n: u32) -> Pair {
  [x[0].rotate_right(n), x[1].rotate_right(n)]
}

/// `2 · lsb(a) · lsb(b)` lane-wise. Masked u32 multiply fits in u64 so
/// the result is exact without a 128-bit-wide product.
#[inline(always)]
fn bla_mul(a: Pair, b: Pair) -> Pair {
  const MASK: u64 = 0xffff_ffff;
  [
    (a[0] & MASK).wrapping_mul(b[0] & MASK).wrapping_shl(1),
    (a[1] & MASK).wrapping_mul(b[1] & MASK).wrapping_shl(1),
  ]
}
