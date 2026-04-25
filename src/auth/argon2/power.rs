//! POWER VSX BlaMka compression kernel for Argon2.
//!
//! Uses `core::simd::u64x2` row pairs and `simd_swizzle!` for cross-pair
//! lane exchange. This stays on the portable-simd surface so the kernel
//! is endian-clean on both POWER8/9/10 little-endian and historical
//! big-endian deployments — VSX intrinsics differ subtly between
//! endianness modes, but the `core::simd` lowering is unambiguous.
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
//! (a/b/c/d × {lo, hi}).
//!
//! # BlaMka multiply
//!
//! `2 · lsb(a) · lsb(b)` lane-wise via masked `u64x2` multiply. POWER's
//! VPMULUDQ-equivalent is `vmulouw` / `vmuleuw` — `core::simd` lowers
//! the masked-multiply pattern to those instructions when it can prove
//! the upper 32 bits are zero, which the explicit `& 0xffffffff` mask
//! makes plain.
//!
//! # Rotations
//!
//! Lane-wise u64 rotate via shift-right + shift-left + OR vector
//! sequence. POWER's `vec_rl` is the native u64 rotate instruction;
//! `core::simd` lowers shift-or pairs to it directly under
//! `target_feature = "vsx"`.

#![cfg(target_arch = "powerpc64")]
#![allow(clippy::cast_possible_truncation)]

use core::simd::u64x2;

use super::BLOCK_WORDS;

type Pair = [u64x2; 2];

/// POWER VSX BlaMka compression kernel.
///
/// # Safety
///
/// - `target_arch = "powerpc64"` enforced at compile time (module gate).
/// - Caller must have VSX available — enforced via `#[target_feature(enable = "vsx")]`.
/// - The three `[u64; BLOCK_WORDS]` buffers are fixed-size arrays.
#[target_feature(enable = "vsx")]
pub(super) unsafe fn compress_vsx(
  dst: &mut [u64; BLOCK_WORDS],
  x: &[u64; BLOCK_WORDS],
  y: &[u64; BLOCK_WORDS],
  xor_into: bool,
) {
  // SAFETY: VSX is enabled by this function's `#[target_feature]`.
  // `core::simd::u64x2` arithmetic is always wrapping per the SIMD
  // contract, so no further unsafe is needed for the algorithm itself.
  let mut r = [0u64; BLOCK_WORDS];
  let mut q = [0u64; BLOCK_WORDS];
  let mut i = 0;
  while i < BLOCK_WORDS {
    let xv = u64x2::from_array([x[i], x[i + 1]]);
    let yv = u64x2::from_array([y[i], y[i + 1]]);
    let rv = (xv ^ yv).to_array();
    r[i] = rv[0];
    r[i + 1] = rv[1];
    q[i] = rv[0];
    q[i + 1] = rv[1];
    i += 2;
  }

  // Row pass.
  let mut row = 0usize;
  while row < 8 {
    let base = row * 16;
    let mut a: Pair = [load_pair(&q, base), load_pair(&q, base + 2)];
    let mut b: Pair = [load_pair(&q, base + 4), load_pair(&q, base + 6)];
    let mut c: Pair = [load_pair(&q, base + 8), load_pair(&q, base + 10)];
    let mut d: Pair = [load_pair(&q, base + 12), load_pair(&q, base + 14)];

    p_round(&mut a, &mut b, &mut c, &mut d);

    store_pair(&mut q, base, a[0]);
    store_pair(&mut q, base + 2, a[1]);
    store_pair(&mut q, base + 4, b[0]);
    store_pair(&mut q, base + 6, b[1]);
    store_pair(&mut q, base + 8, c[0]);
    store_pair(&mut q, base + 10, c[1]);
    store_pair(&mut q, base + 12, d[0]);
    store_pair(&mut q, base + 14, d[1]);
    row += 1;
  }

  // Column pass.
  let mut col = 0usize;
  while col < 8 {
    let base = col * 2;
    let mut a: Pair = [load_pair(&q, base), load_pair(&q, base + 16)];
    let mut b: Pair = [load_pair(&q, base + 32), load_pair(&q, base + 48)];
    let mut c: Pair = [load_pair(&q, base + 64), load_pair(&q, base + 80)];
    let mut d: Pair = [load_pair(&q, base + 96), load_pair(&q, base + 112)];

    p_round(&mut a, &mut b, &mut c, &mut d);

    store_pair(&mut q, base, a[0]);
    store_pair(&mut q, base + 16, a[1]);
    store_pair(&mut q, base + 32, b[0]);
    store_pair(&mut q, base + 48, b[1]);
    store_pair(&mut q, base + 64, c[0]);
    store_pair(&mut q, base + 80, c[1]);
    store_pair(&mut q, base + 96, d[0]);
    store_pair(&mut q, base + 112, d[1]);
    col += 1;
  }

  // Final XOR with R, fused with dst store/xor.
  let mut i = 0;
  while i < BLOCK_WORDS {
    let qv = load_pair(&q, i);
    let rv = load_pair(&r, i);
    let f = (qv ^ rv).to_array();
    if xor_into {
      dst[i] ^= f[0];
      dst[i + 1] ^= f[1];
    } else {
      dst[i] = f[0];
      dst[i + 1] = f[1];
    }
    i += 2;
  }
}

#[inline(always)]
fn load_pair(buf: &[u64; BLOCK_WORDS], idx: usize) -> u64x2 {
  u64x2::from_array([buf[idx], buf[idx + 1]])
}

#[inline(always)]
fn store_pair(buf: &mut [u64; BLOCK_WORDS], idx: usize, v: u64x2) {
  let a = v.to_array();
  buf[idx] = a[0];
  buf[idx + 1] = a[1];
}

// ─── 4-way P-round ─────────────────────────────────────────────────────────

#[inline(always)]
fn p_round(a: &mut Pair, b: &mut Pair, c: &mut Pair, d: &mut Pair) {
  // Column step.
  gb(a, b, c, d);

  // Diagonalise: rotate b by 1, c by 2, d by 3 across the 4-lane row.
  let tb0 = b[0];
  let tb1 = b[1];
  b[0] = pair_a1_b0(tb0, tb1);
  b[1] = pair_b1_a0(tb0, tb1);

  c.swap(0, 1);

  let td0 = d[0];
  let td1 = d[1];
  d[0] = pair_b1_a0(td0, td1);
  d[1] = pair_a1_b0(td0, td1);

  // Diagonal step.
  gb(a, b, c, d);

  // Undo diagonalisation.
  let tb0 = b[0];
  let tb1 = b[1];
  b[0] = pair_b1_a0(tb0, tb1);
  b[1] = pair_a1_b0(tb0, tb1);

  c.swap(0, 1);

  let td0 = d[0];
  let td1 = d[1];
  d[0] = pair_a1_b0(td0, td1);
  d[1] = pair_b1_a0(td0, td1);
}

/// `simd_swizzle!(a, b, [1, 2])` = `[a[1], b[0]]` — the cross-pair
/// "low half of next lane" used by the b-rotation in diagonalise.
#[inline(always)]
fn pair_a1_b0(a: u64x2, b: u64x2) -> u64x2 {
  core::simd::simd_swizzle!(a, b, [1, 2])
}

/// `simd_swizzle!(a, b, [3, 0])` = `[b[1], a[0]]` — the cross-pair
/// "high half of previous lane" used by the b-rotation in diagonalise.
#[inline(always)]
fn pair_b1_a0(a: u64x2, b: u64x2) -> u64x2 {
  core::simd::simd_swizzle!(a, b, [3, 0])
}

// ─── 4-way BlaMka G ────────────────────────────────────────────────────────

#[inline(always)]
fn gb(a: &mut Pair, b: &mut Pair, c: &mut Pair, d: &mut Pair) {
  // Step 1: a = a + b + 2·lsb(a)·lsb(b)
  let p0 = bla_mul(a[0], b[0]);
  let p1 = bla_mul(a[1], b[1]);
  a[0] = a[0] + b[0] + p0;
  a[1] = a[1] + b[1] + p1;
  d[0] = ror::<32>(d[0] ^ a[0]);
  d[1] = ror::<32>(d[1] ^ a[1]);

  // Step 2: c = c + d + 2·lsb(c)·lsb(d)
  let p0 = bla_mul(c[0], d[0]);
  let p1 = bla_mul(c[1], d[1]);
  c[0] = c[0] + d[0] + p0;
  c[1] = c[1] + d[1] + p1;
  b[0] = ror::<24>(b[0] ^ c[0]);
  b[1] = ror::<24>(b[1] ^ c[1]);

  // Step 3: a = a + b + 2·lsb(a)·lsb(b)
  let p0 = bla_mul(a[0], b[0]);
  let p1 = bla_mul(a[1], b[1]);
  a[0] = a[0] + b[0] + p0;
  a[1] = a[1] + b[1] + p1;
  d[0] = ror::<16>(d[0] ^ a[0]);
  d[1] = ror::<16>(d[1] ^ a[1]);

  // Step 4: c = c + d + 2·lsb(c)·lsb(d)
  let p0 = bla_mul(c[0], d[0]);
  let p1 = bla_mul(c[1], d[1]);
  c[0] = c[0] + d[0] + p0;
  c[1] = c[1] + d[1] + p1;
  b[0] = ror::<63>(b[0] ^ c[0]);
  b[1] = ror::<63>(b[1] ^ c[1]);
}

// ─── Micro-ops ─────────────────────────────────────────────────────────────

/// Lane-wise u64 right-rotate. `core::simd` lowers `(x >> n) | (x << (64-n))`
/// to a single VSX `vrld`-pattern rotate when the shift count is a
/// compile-time constant.
#[inline(always)]
fn ror<const N: u32>(v: u64x2) -> u64x2 {
  const { assert!(N > 0 && N < 64) }
  let s = u64x2::splat(N as u64);
  let r = u64x2::splat((64 - N) as u64);
  (v >> s) | (v << r)
}

/// `2 · lsb(a) · lsb(b)` lane-wise via masked `u64x2 *`. The product
/// of two u32 values fits in u64, so the mask makes the multiply
/// exact without spilling through wider types.
#[inline(always)]
fn bla_mul(a: u64x2, b: u64x2) -> u64x2 {
  let mask = u64x2::splat(0xffff_ffff);
  let al = a & mask;
  let bl = b & mask;
  (al * bl) << u64x2::splat(1)
}
