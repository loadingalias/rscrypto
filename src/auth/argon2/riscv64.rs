//! RISC-V Vector (RVV) BlaMka compression kernel for Argon2.
//!
//! Uses `core::simd::u64x2` pairs to expose two independent lanes per
//! operation. Dispatch requires the RISC-V V extension for this backend;
//! exact RVV lowering remains target- and toolchain-specific evidence.
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
//! parallelism with each `u64x2` pair holding two of the four lanes
//! per row (a/b/c/d × {lo, hi}).
//!
//! Diagonal step: rotate `b` by 1, `c` by 2, `d` by 3 within each
//! 4-lane row using cross-pair swizzles.
//!
//! # BlaMka multiply
//!
//! `2 · lsb(a) · lsb(b)` lane-wise via a masked SIMD multiply. The product of
//! two 32-bit values fits in u64, so the operation is exact modulo the lane
//! representation.
//!
//! # Rotations
//!
//! All four rotations use lane-wise SIMD shifts and OR. Instruction selection
//! is left to the target compiler.

#![cfg(target_arch = "riscv64")]

use core::simd::u64x2;

use super::BLOCK_WORDS;

type Pair = u64x2;

struct RvvState {
  a_lo: Pair,
  a_hi: Pair,
  b_lo: Pair,
  b_hi: Pair,
  c_lo: Pair,
  c_hi: Pair,
  d_lo: Pair,
  d_hi: Pair,
}

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
  let mut r = [0u64; BLOCK_WORDS];
  let mut q = [0u64; BLOCK_WORDS];
  let mut i = 0usize;
  while i < BLOCK_WORDS {
    let rv = vxor(load_pair(x, i), load_pair(y, i));
    store_pair(&mut r, i, rv);
    store_pair(&mut q, i, rv);
    i = i.strict_add(2);
  }

  // Row pass.
  let mut row = 0usize;
  while row < 8 {
    let base = row.strict_mul(16);
    let mut state = RvvState {
      a_lo: load_pair(&q, base),
      a_hi: load_pair(&q, base.strict_add(2)),
      b_lo: load_pair(&q, base.strict_add(4)),
      b_hi: load_pair(&q, base.strict_add(6)),
      c_lo: load_pair(&q, base.strict_add(8)),
      c_hi: load_pair(&q, base.strict_add(10)),
      d_lo: load_pair(&q, base.strict_add(12)),
      d_hi: load_pair(&q, base.strict_add(14)),
    };

    p_round(&mut state);

    store_pair(&mut q, base, state.a_lo);
    store_pair(&mut q, base.strict_add(2), state.a_hi);
    store_pair(&mut q, base.strict_add(4), state.b_lo);
    store_pair(&mut q, base.strict_add(6), state.b_hi);
    store_pair(&mut q, base.strict_add(8), state.c_lo);
    store_pair(&mut q, base.strict_add(10), state.c_hi);
    store_pair(&mut q, base.strict_add(12), state.d_lo);
    store_pair(&mut q, base.strict_add(14), state.d_hi);
    row = row.strict_add(1);
  }

  // Column pass.
  let mut col = 0usize;
  while col < 8 {
    let base = col.strict_mul(2);
    let mut state = RvvState {
      a_lo: load_pair(&q, base),
      a_hi: load_pair(&q, base.strict_add(16)),
      b_lo: load_pair(&q, base.strict_add(32)),
      b_hi: load_pair(&q, base.strict_add(48)),
      c_lo: load_pair(&q, base.strict_add(64)),
      c_hi: load_pair(&q, base.strict_add(80)),
      d_lo: load_pair(&q, base.strict_add(96)),
      d_hi: load_pair(&q, base.strict_add(112)),
    };

    p_round(&mut state);

    store_pair(&mut q, base, state.a_lo);
    store_pair(&mut q, base.strict_add(16), state.a_hi);
    store_pair(&mut q, base.strict_add(32), state.b_lo);
    store_pair(&mut q, base.strict_add(48), state.b_hi);
    store_pair(&mut q, base.strict_add(64), state.c_lo);
    store_pair(&mut q, base.strict_add(80), state.c_hi);
    store_pair(&mut q, base.strict_add(96), state.d_lo);
    store_pair(&mut q, base.strict_add(112), state.d_hi);
    col = col.strict_add(1);
  }

  // Final XOR with R, fused with dst store/xor.
  let mut i = 0usize;
  while i < BLOCK_WORDS {
    let result = vxor(load_pair(&q, i), load_pair(&r, i));
    let result = if xor_into {
      vxor(load_pair(dst, i), result)
    } else {
      result
    };
    store_pair(dst, i, result);
    i = i.strict_add(2);
  }
}

#[inline(always)]
fn load_pair(buf: &[u64; BLOCK_WORDS], idx: usize) -> Pair {
  u64x2::from_array([buf[idx], buf[idx.strict_add(1)]])
}

#[inline(always)]
fn store_pair(buf: &mut [u64; BLOCK_WORDS], idx: usize, pair: Pair) {
  let lanes = pair.to_array();
  buf[idx] = lanes[0];
  buf[idx.strict_add(1)] = lanes[1];
}

// ─── 4-way P-round ─────────────────────────────────────────────────────────

#[inline(always)]
fn p_round(state: &mut RvvState) {
  // Column step.
  gb(state);

  // Diagonalise: rotate b by 1, c by 2, d by 3 across the 4-lane row.
  let b_lo = state.b_lo;
  let b_hi = state.b_hi;
  state.b_lo = pair_a1_b0(b_lo, b_hi);
  state.b_hi = pair_b1_a0(b_lo, b_hi);

  core::mem::swap(&mut state.c_lo, &mut state.c_hi);

  let d_lo = state.d_lo;
  let d_hi = state.d_hi;
  state.d_lo = pair_b1_a0(d_lo, d_hi);
  state.d_hi = pair_a1_b0(d_lo, d_hi);

  // Diagonal step.
  gb(state);

  // Undo diagonalisation.
  let b_lo = state.b_lo;
  let b_hi = state.b_hi;
  state.b_lo = pair_b1_a0(b_lo, b_hi);
  state.b_hi = pair_a1_b0(b_lo, b_hi);

  core::mem::swap(&mut state.c_lo, &mut state.c_hi);

  let d_lo = state.d_lo;
  let d_hi = state.d_hi;
  state.d_lo = pair_a1_b0(d_lo, d_hi);
  state.d_hi = pair_b1_a0(d_lo, d_hi);
}

#[inline(always)]
fn pair_a1_b0(a: Pair, b: Pair) -> Pair {
  core::simd::simd_swizzle!(a, b, [1, 2])
}

#[inline(always)]
fn pair_b1_a0(a: Pair, b: Pair) -> Pair {
  core::simd::simd_swizzle!(a, b, [3, 0])
}

// ─── 4-way BlaMka G ────────────────────────────────────────────────────────

#[inline(always)]
fn gb(state: &mut RvvState) {
  // Step 1: a = a + b + 2·lsb(a)·lsb(b)
  let p_lo = bla_mul(state.a_lo, state.b_lo);
  let p_hi = bla_mul(state.a_hi, state.b_hi);
  state.a_lo = bla_add(state.a_lo, state.b_lo, p_lo);
  state.a_hi = bla_add(state.a_hi, state.b_hi, p_hi);
  state.d_lo = ror::<32>(vxor(state.d_lo, state.a_lo));
  state.d_hi = ror::<32>(vxor(state.d_hi, state.a_hi));

  // Step 2: c = c + d + 2·lsb(c)·lsb(d)
  let p_lo = bla_mul(state.c_lo, state.d_lo);
  let p_hi = bla_mul(state.c_hi, state.d_hi);
  state.c_lo = bla_add(state.c_lo, state.d_lo, p_lo);
  state.c_hi = bla_add(state.c_hi, state.d_hi, p_hi);
  state.b_lo = ror::<24>(vxor(state.b_lo, state.c_lo));
  state.b_hi = ror::<24>(vxor(state.b_hi, state.c_hi));

  // Step 3: a = a + b + 2·lsb(a)·lsb(b)
  let p_lo = bla_mul(state.a_lo, state.b_lo);
  let p_hi = bla_mul(state.a_hi, state.b_hi);
  state.a_lo = bla_add(state.a_lo, state.b_lo, p_lo);
  state.a_hi = bla_add(state.a_hi, state.b_hi, p_hi);
  state.d_lo = ror::<16>(vxor(state.d_lo, state.a_lo));
  state.d_hi = ror::<16>(vxor(state.d_hi, state.a_hi));

  // Step 4: c = c + d + 2·lsb(c)·lsb(d)
  let p_lo = bla_mul(state.c_lo, state.d_lo);
  let p_hi = bla_mul(state.c_hi, state.d_hi);
  state.c_lo = bla_add(state.c_lo, state.d_lo, p_lo);
  state.c_hi = bla_add(state.c_hi, state.d_hi, p_hi);
  state.b_lo = ror::<63>(vxor(state.b_lo, state.c_lo));
  state.b_hi = ror::<63>(vxor(state.b_hi, state.c_hi));
}

// ─── Micro-ops ─────────────────────────────────────────────────────────────

#[inline(always)]
fn vxor(a: Pair, b: Pair) -> Pair {
  core::ops::BitXor::bitxor(a, b)
}

#[inline(always)]
fn ror<const N: u32>(value: Pair) -> Pair {
  const { assert!(N > 0 && N < 64) }
  let right = core::ops::Shr::shr(value, u64x2::splat(u64::from(N)));
  let left = core::ops::Shl::shl(value, u64x2::splat(u64::from(64u32.strict_sub(N))));
  core::ops::BitOr::bitor(right, left)
}

#[inline(always)]
fn bla_add(a: Pair, b: Pair, product: Pair) -> Pair {
  simd_wrapping_add(simd_wrapping_add(a, b), product)
}

/// `2 · lsb(a) · lsb(b)` lane-wise. Masked u32 multiply fits in u64 so
/// the result is exact without a 128-bit-wide product.
#[inline(always)]
fn bla_mul(a: Pair, b: Pair) -> Pair {
  let mask = u64x2::splat(0xffff_ffff);
  let product = simd_wrapping_mul(a & mask, b & mask);
  core::ops::Shl::shl(product, u64x2::splat(1))
}

#[inline(always)]
fn simd_wrapping_add(a: Pair, b: Pair) -> Pair {
  core::ops::Add::add(a, b)
}

#[inline(always)]
fn simd_wrapping_mul(a: Pair, b: Pair) -> Pair {
  core::ops::Mul::mul(a, b)
}
