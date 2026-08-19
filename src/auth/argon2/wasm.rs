//! WebAssembly SIMD128 BlaMka compression kernel for Argon2.
//!
//! 128-bit `v128` vectors hold 2 u64 lanes each. The 16-u64 P-round state
//! is packed across 8 `v128` registers (a/b/c/d × {lo, hi} = 8 × 2-u64
//! pairs), one row of 4 GB-lanes per "row pair" in lo/hi order. This
//! mirrors the aarch64 NEON kernel topology so all 128-bit-vector
//! backends share a single mental model.
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
//! parallelism (one GB per SIMD lane across `a`, `b`, `c`, `d`).
//!
//! Diagonal step: rotate `b` by 1, `c` by 2, `d` by 3 within each 4-lane
//! row — `i64x2_shuffle` for cross-pair lane exchange and a direct
//! lo/hi swap for `c`.
//!
//! # BlaMka multiply
//!
//! `2 · lsb(a) · lsb(b)` lane-wise. wasm SIMD128 has no native 32×32→64
//! multiply, but `i64x2_mul` produces the low 64 bits of the full
//! 128-bit product. For operands ≤ `2^32 − 1`, the full product fits
//! in u64, so masking with `0xffffffff` and using `i64x2_mul` is exact.
//!
//! # Rotations
//!
//! - ROR 32: byte shuffle (4-byte halves swap within each u64 lane).
//! - ROR 16: byte shuffle (2-byte rotation within each u64 lane).
//! - ROR 24: shift-right + shift-left + OR.
//! - ROR 63 ≡ ROL 1: shift-right + shift-left + OR.

#![cfg(target_arch = "wasm32")]

use core::arch::wasm32::{
  i8x16_shuffle, i64x2_add, i64x2_mul, i64x2_shuffle, u64x2, u64x2_extract_lane, u64x2_shl, u64x2_shr, u64x2_splat,
  v128, v128_and, v128_or, v128_xor,
};

use super::BLOCK_WORDS;

struct RoundState {
  a: [v128; 2],
  b: [v128; 2],
  c: [v128; 2],
  d: [v128; 2],
}

#[inline(always)]
fn load_pair<const N: usize>(words: &[u64; N], offset: usize) -> v128 {
  u64x2(words[offset], words[offset.strict_add(1)])
}

#[inline(always)]
fn store_pair<const N: usize>(words: &mut [u64; N], offset: usize, value: v128) {
  words[offset] = u64x2_extract_lane::<0>(value);
  words[offset.strict_add(1)] = u64x2_extract_lane::<1>(value);
}

/// WebAssembly SIMD128 BlaMka compression kernel.
///
/// # Safety
///
/// - `target_arch = "wasm32"` is enforced at compile time (module gate).
/// - Caller must have `simd128` available — enforced via `#[target_feature(enable = "simd128")]`.
/// - The three `[u64; BLOCK_WORDS]` buffers are fixed-size arrays; the kernel reads/writes only
///   within their bounds.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn compress_simd128(
  dst: &mut [u64; BLOCK_WORDS],
  x: &[u64; BLOCK_WORDS],
  y: &[u64; BLOCK_WORDS],
  xor_into: bool,
) {
  // R = X XOR Y, materialised to scratch for re-reads during the
  // row + column passes plus the final XOR.
  let mut r = [0u64; BLOCK_WORDS];
  let mut q = [0u64; BLOCK_WORDS];
  let mut i = 0usize;
  while i < BLOCK_WORDS {
    let rv = v128_xor(load_pair(x, i), load_pair(y, i));
    store_pair(&mut r, i, rv);
    store_pair(&mut q, i, rv);
    i = i.strict_add(2);
  }

  // Row pass: 8 P-rounds on contiguous 16-u64 chunks of q[].
  let mut row = 0usize;
  while row < 8 {
    let base = row.strict_mul(16);
    let mut state = RoundState {
      a: [load_pair(&q, base), load_pair(&q, base.strict_add(2))],
      b: [load_pair(&q, base.strict_add(4)), load_pair(&q, base.strict_add(6))],
      c: [load_pair(&q, base.strict_add(8)), load_pair(&q, base.strict_add(10))],
      d: [load_pair(&q, base.strict_add(12)), load_pair(&q, base.strict_add(14))],
    };

    p_round(&mut state);

    store_pair(&mut q, base, state.a[0]);
    store_pair(&mut q, base.strict_add(2), state.a[1]);
    store_pair(&mut q, base.strict_add(4), state.b[0]);
    store_pair(&mut q, base.strict_add(6), state.b[1]);
    store_pair(&mut q, base.strict_add(8), state.c[0]);
    store_pair(&mut q, base.strict_add(10), state.c[1]);
    store_pair(&mut q, base.strict_add(12), state.d[0]);
    store_pair(&mut q, base.strict_add(14), state.d[1]);
    row = row.strict_add(1);
  }

  // Column pass: 8 P-rounds on stride-16 u64 sequences. Each lane of
  // 2 u64 is loaded directly from the natural row-major positions —
  // see RFC 9106 §3.6 column-step indexing.
  let mut col = 0usize;
  while col < 8 {
    let base = col.strict_mul(2);
    let mut state = RoundState {
      a: [load_pair(&q, base), load_pair(&q, base.strict_add(16))],
      b: [load_pair(&q, base.strict_add(32)), load_pair(&q, base.strict_add(48))],
      c: [load_pair(&q, base.strict_add(64)), load_pair(&q, base.strict_add(80))],
      d: [load_pair(&q, base.strict_add(96)), load_pair(&q, base.strict_add(112))],
    };

    p_round(&mut state);

    store_pair(&mut q, base, state.a[0]);
    store_pair(&mut q, base.strict_add(16), state.a[1]);
    store_pair(&mut q, base.strict_add(32), state.b[0]);
    store_pair(&mut q, base.strict_add(48), state.b[1]);
    store_pair(&mut q, base.strict_add(64), state.c[0]);
    store_pair(&mut q, base.strict_add(80), state.c[1]);
    store_pair(&mut q, base.strict_add(96), state.d[0]);
    store_pair(&mut q, base.strict_add(112), state.d[1]);
    col = col.strict_add(1);
  }

  // Final XOR with R, fused with dst store/xor.
  let mut i = 0usize;
  while i < BLOCK_WORDS {
    let f = v128_xor(load_pair(&q, i), load_pair(&r, i));
    let output = if xor_into { v128_xor(load_pair(dst, i), f) } else { f };
    store_pair(dst, i, output);
    i = i.strict_add(2);
  }
}

// ─── 4-way P-round ─────────────────────────────────────────────────────────

#[inline(always)]
fn p_round(state: &mut RoundState) {
  // Column step.
  gb(state);

  // Diagonalise: rotate b by 1, c by 2, d by 3 across the 4-lane row.
  let tb_lo = state.b[0];
  let tb_hi = state.b[1];
  state.b[0] = i64x2_shuffle::<1, 2>(tb_lo, tb_hi);
  state.b[1] = i64x2_shuffle::<1, 2>(tb_hi, tb_lo);

  state.c.swap(0, 1);

  let td_lo = state.d[0];
  let td_hi = state.d[1];
  state.d[0] = i64x2_shuffle::<1, 2>(td_hi, td_lo);
  state.d[1] = i64x2_shuffle::<1, 2>(td_lo, td_hi);

  // Diagonal step.
  gb(state);

  // Undo diagonalisation.
  let tb_lo = state.b[0];
  let tb_hi = state.b[1];
  state.b[0] = i64x2_shuffle::<1, 2>(tb_hi, tb_lo);
  state.b[1] = i64x2_shuffle::<1, 2>(tb_lo, tb_hi);

  state.c.swap(0, 1);

  let td_lo = state.d[0];
  let td_hi = state.d[1];
  state.d[0] = i64x2_shuffle::<1, 2>(td_lo, td_hi);
  state.d[1] = i64x2_shuffle::<1, 2>(td_hi, td_lo);
}

// ─── 4-way BlaMka G ────────────────────────────────────────────────────────

#[inline(always)]
fn gb(state: &mut RoundState) {
  // Step 1: a = a + b + 2·lsb(a)·lsb(b)
  let p_lo = bla_mul(state.a[0], state.b[0]);
  let p_hi = bla_mul(state.a[1], state.b[1]);
  state.a[0] = i64x2_add(i64x2_add(state.a[0], state.b[0]), p_lo);
  state.a[1] = i64x2_add(i64x2_add(state.a[1], state.b[1]), p_hi);
  state.d[0] = ror32(v128_xor(state.d[0], state.a[0]));
  state.d[1] = ror32(v128_xor(state.d[1], state.a[1]));

  // Step 2: c = c + d + 2·lsb(c)·lsb(d)
  let p_lo = bla_mul(state.c[0], state.d[0]);
  let p_hi = bla_mul(state.c[1], state.d[1]);
  state.c[0] = i64x2_add(i64x2_add(state.c[0], state.d[0]), p_lo);
  state.c[1] = i64x2_add(i64x2_add(state.c[1], state.d[1]), p_hi);
  state.b[0] = ror24(v128_xor(state.b[0], state.c[0]));
  state.b[1] = ror24(v128_xor(state.b[1], state.c[1]));

  // Step 3: a = a + b + 2·lsb(a)·lsb(b)
  let p_lo = bla_mul(state.a[0], state.b[0]);
  let p_hi = bla_mul(state.a[1], state.b[1]);
  state.a[0] = i64x2_add(i64x2_add(state.a[0], state.b[0]), p_lo);
  state.a[1] = i64x2_add(i64x2_add(state.a[1], state.b[1]), p_hi);
  state.d[0] = ror16(v128_xor(state.d[0], state.a[0]));
  state.d[1] = ror16(v128_xor(state.d[1], state.a[1]));

  // Step 4: c = c + d + 2·lsb(c)·lsb(d)
  let p_lo = bla_mul(state.c[0], state.d[0]);
  let p_hi = bla_mul(state.c[1], state.d[1]);
  state.c[0] = i64x2_add(i64x2_add(state.c[0], state.d[0]), p_lo);
  state.c[1] = i64x2_add(i64x2_add(state.c[1], state.d[1]), p_hi);
  state.b[0] = ror63(v128_xor(state.b[0], state.c[0]));
  state.b[1] = ror63(v128_xor(state.b[1], state.c[1]));
}

// ─── Micro-ops ─────────────────────────────────────────────────────────────

/// `2 · lsb(a) · lsb(b)` lane-wise, exploiting the fact that
/// `(2^32 − 1)^2 < 2^64` so masking + low-64 multiply is exact.
#[inline(always)]
fn bla_mul(a: v128, b: v128) -> v128 {
  let mask = u64x2_splat(0xffff_ffff);
  let al = v128_and(a, mask);
  let bl = v128_and(b, mask);
  u64x2_shl(i64x2_mul(al, bl), 1)
}

/// ROR 32 via byte shuffle (swap 4-byte halves within each u64 lane).
#[inline(always)]
fn ror32(x: v128) -> v128 {
  i8x16_shuffle::<4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11>(x, x)
}

/// ROR 24 via shift-right + shift-left + OR.
///
/// (No native 24-bit byte shuffle pattern in simd128 — Blake2b uses the
/// same fallback. The shift-pair is a fixed-cost 3-instruction sequence
/// that the V8 / Wasmtime engine pipelines well.)
#[inline(always)]
fn ror24(x: v128) -> v128 {
  v128_or(u64x2_shr(x, 24), u64x2_shl(x, 40))
}

/// ROR 16 via byte shuffle (rotate 2 bytes within each u64 lane).
#[inline(always)]
fn ror16(x: v128) -> v128 {
  i8x16_shuffle::<2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9>(x, x)
}

/// ROR 63 ≡ ROL 1 = `(x >> 63) | (x << 1)`.
#[inline(always)]
fn ror63(x: v128) -> v128 {
  v128_or(u64x2_shr(x, 63), u64x2_shl(x, 1))
}
