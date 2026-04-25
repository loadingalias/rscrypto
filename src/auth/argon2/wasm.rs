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
#![allow(clippy::cast_possible_truncation)]

use core::arch::wasm32::{
  i8x16_shuffle, i64x2_add, i64x2_mul, i64x2_shuffle, u64x2_shl, u64x2_shr, u64x2_splat, v128, v128_and, v128_load,
  v128_or, v128_store, v128_xor,
};

use super::BLOCK_WORDS;

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
  // SAFETY: simd128 is enabled by this function's `#[target_feature]`
  // attribute, so all `v128`/`u64x2_*` ops below are valid to call.
  unsafe {
    // R = X XOR Y, materialised to scratch for re-reads during the
    // row + column passes plus the final XOR.
    let mut r = [0u64; BLOCK_WORDS];
    let mut q = [0u64; BLOCK_WORDS];
    let mut i = 0;
    while i < BLOCK_WORDS {
      let xv = v128_load(x.as_ptr().add(i).cast());
      let yv = v128_load(y.as_ptr().add(i).cast());
      let rv = v128_xor(xv, yv);
      v128_store(r.as_mut_ptr().add(i).cast(), rv);
      v128_store(q.as_mut_ptr().add(i).cast(), rv);
      i += 2;
    }

    // Row pass: 8 P-rounds on contiguous 16-u64 chunks of q[].
    let mut row = 0usize;
    while row < 8 {
      let base = row * 16;
      let mut a_lo = v128_load(q.as_ptr().add(base).cast());
      let mut a_hi = v128_load(q.as_ptr().add(base + 2).cast());
      let mut b_lo = v128_load(q.as_ptr().add(base + 4).cast());
      let mut b_hi = v128_load(q.as_ptr().add(base + 6).cast());
      let mut c_lo = v128_load(q.as_ptr().add(base + 8).cast());
      let mut c_hi = v128_load(q.as_ptr().add(base + 10).cast());
      let mut d_lo = v128_load(q.as_ptr().add(base + 12).cast());
      let mut d_hi = v128_load(q.as_ptr().add(base + 14).cast());

      p_round(
        &mut a_lo, &mut a_hi, &mut b_lo, &mut b_hi, &mut c_lo, &mut c_hi, &mut d_lo, &mut d_hi,
      );

      v128_store(q.as_mut_ptr().add(base).cast(), a_lo);
      v128_store(q.as_mut_ptr().add(base + 2).cast(), a_hi);
      v128_store(q.as_mut_ptr().add(base + 4).cast(), b_lo);
      v128_store(q.as_mut_ptr().add(base + 6).cast(), b_hi);
      v128_store(q.as_mut_ptr().add(base + 8).cast(), c_lo);
      v128_store(q.as_mut_ptr().add(base + 10).cast(), c_hi);
      v128_store(q.as_mut_ptr().add(base + 12).cast(), d_lo);
      v128_store(q.as_mut_ptr().add(base + 14).cast(), d_hi);
      row += 1;
    }

    // Column pass: 8 P-rounds on stride-16 u64 sequences. Each lane of
    // 2 u64 is loaded directly from the natural row-major positions —
    // see RFC 9106 §3.6 column-step indexing.
    let mut col = 0usize;
    while col < 8 {
      let base = col * 2;
      let mut a_lo = v128_load(q.as_ptr().add(base).cast());
      let mut a_hi = v128_load(q.as_ptr().add(base + 16).cast());
      let mut b_lo = v128_load(q.as_ptr().add(base + 32).cast());
      let mut b_hi = v128_load(q.as_ptr().add(base + 48).cast());
      let mut c_lo = v128_load(q.as_ptr().add(base + 64).cast());
      let mut c_hi = v128_load(q.as_ptr().add(base + 80).cast());
      let mut d_lo = v128_load(q.as_ptr().add(base + 96).cast());
      let mut d_hi = v128_load(q.as_ptr().add(base + 112).cast());

      p_round(
        &mut a_lo, &mut a_hi, &mut b_lo, &mut b_hi, &mut c_lo, &mut c_hi, &mut d_lo, &mut d_hi,
      );

      v128_store(q.as_mut_ptr().add(base).cast(), a_lo);
      v128_store(q.as_mut_ptr().add(base + 16).cast(), a_hi);
      v128_store(q.as_mut_ptr().add(base + 32).cast(), b_lo);
      v128_store(q.as_mut_ptr().add(base + 48).cast(), b_hi);
      v128_store(q.as_mut_ptr().add(base + 64).cast(), c_lo);
      v128_store(q.as_mut_ptr().add(base + 80).cast(), c_hi);
      v128_store(q.as_mut_ptr().add(base + 96).cast(), d_lo);
      v128_store(q.as_mut_ptr().add(base + 112).cast(), d_hi);
      col += 1;
    }

    // Final XOR with R, fused with dst store/xor.
    let mut i = 0;
    while i < BLOCK_WORDS {
      let qv = v128_load(q.as_ptr().add(i).cast());
      let rv = v128_load(r.as_ptr().add(i).cast());
      let f = v128_xor(qv, rv);
      if xor_into {
        let cur = v128_load(dst.as_ptr().add(i).cast());
        v128_store(dst.as_mut_ptr().add(i).cast(), v128_xor(cur, f));
      } else {
        v128_store(dst.as_mut_ptr().add(i).cast(), f);
      }
      i += 2;
    }
  }
}

// ─── 4-way P-round ─────────────────────────────────────────────────────────

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn p_round(
  a_lo: &mut v128,
  a_hi: &mut v128,
  b_lo: &mut v128,
  b_hi: &mut v128,
  c_lo: &mut v128,
  c_hi: &mut v128,
  d_lo: &mut v128,
  d_hi: &mut v128,
) {
  // Column step.
  gb(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi, d_lo, d_hi);

  // Diagonalise: rotate b by 1, c by 2, d by 3 across the 4-lane row.
  let tb_lo = *b_lo;
  let tb_hi = *b_hi;
  *b_lo = i64x2_shuffle::<1, 2>(tb_lo, tb_hi);
  *b_hi = i64x2_shuffle::<1, 2>(tb_hi, tb_lo);

  core::mem::swap(c_lo, c_hi);

  let td_lo = *d_lo;
  let td_hi = *d_hi;
  *d_lo = i64x2_shuffle::<1, 2>(td_hi, td_lo);
  *d_hi = i64x2_shuffle::<1, 2>(td_lo, td_hi);

  // Diagonal step.
  gb(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi, d_lo, d_hi);

  // Undo diagonalisation.
  let tb_lo = *b_lo;
  let tb_hi = *b_hi;
  *b_lo = i64x2_shuffle::<1, 2>(tb_hi, tb_lo);
  *b_hi = i64x2_shuffle::<1, 2>(tb_lo, tb_hi);

  core::mem::swap(c_lo, c_hi);

  let td_lo = *d_lo;
  let td_hi = *d_hi;
  *d_lo = i64x2_shuffle::<1, 2>(td_lo, td_hi);
  *d_hi = i64x2_shuffle::<1, 2>(td_hi, td_lo);
}

// ─── 4-way BlaMka G ────────────────────────────────────────────────────────

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn gb(
  a_lo: &mut v128,
  a_hi: &mut v128,
  b_lo: &mut v128,
  b_hi: &mut v128,
  c_lo: &mut v128,
  c_hi: &mut v128,
  d_lo: &mut v128,
  d_hi: &mut v128,
) {
  // Step 1: a = a + b + 2·lsb(a)·lsb(b)
  let p_lo = bla_mul(*a_lo, *b_lo);
  let p_hi = bla_mul(*a_hi, *b_hi);
  *a_lo = i64x2_add(i64x2_add(*a_lo, *b_lo), p_lo);
  *a_hi = i64x2_add(i64x2_add(*a_hi, *b_hi), p_hi);
  *d_lo = ror32(v128_xor(*d_lo, *a_lo));
  *d_hi = ror32(v128_xor(*d_hi, *a_hi));

  // Step 2: c = c + d + 2·lsb(c)·lsb(d)
  let p_lo = bla_mul(*c_lo, *d_lo);
  let p_hi = bla_mul(*c_hi, *d_hi);
  *c_lo = i64x2_add(i64x2_add(*c_lo, *d_lo), p_lo);
  *c_hi = i64x2_add(i64x2_add(*c_hi, *d_hi), p_hi);
  *b_lo = ror24(v128_xor(*b_lo, *c_lo));
  *b_hi = ror24(v128_xor(*b_hi, *c_hi));

  // Step 3: a = a + b + 2·lsb(a)·lsb(b)
  let p_lo = bla_mul(*a_lo, *b_lo);
  let p_hi = bla_mul(*a_hi, *b_hi);
  *a_lo = i64x2_add(i64x2_add(*a_lo, *b_lo), p_lo);
  *a_hi = i64x2_add(i64x2_add(*a_hi, *b_hi), p_hi);
  *d_lo = ror16(v128_xor(*d_lo, *a_lo));
  *d_hi = ror16(v128_xor(*d_hi, *a_hi));

  // Step 4: c = c + d + 2·lsb(c)·lsb(d)
  let p_lo = bla_mul(*c_lo, *d_lo);
  let p_hi = bla_mul(*c_hi, *d_hi);
  *c_lo = i64x2_add(i64x2_add(*c_lo, *d_lo), p_lo);
  *c_hi = i64x2_add(i64x2_add(*c_hi, *d_hi), p_hi);
  *b_lo = ror63(v128_xor(*b_lo, *c_lo));
  *b_hi = ror63(v128_xor(*b_hi, *c_hi));
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
