//! Blake2b WebAssembly SIMD128-accelerated compression.
//!
//! Each row of the 4x4 u64 working matrix is split across two `v128`
//! registers (lo = lanes 0-1, hi = lanes 2-3). Diagonalization uses
//! `i64x2_shuffle` for lane rearrangement.
//!
//! Rotations use shift-right + shift-left + OR, except ROR 32 which maps
//! to a byte shuffle (swap 4-byte halves within each 8-byte lane).
//!
//! # Safety
//!
//! Requires WASM SIMD128. Caller must verify `wasm::SIMD128`.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

use super::kernels::{SIGMA, init_v, load_msg};

// ─── Rotation helpers ─────────────────────────────────────────────────────

/// Rotate right by 32: byte shuffle swapping 4-byte halves within each u64 lane.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn ror32(x: v128) -> v128 {
  // Swap bytes [0..3] <-> [4..7] and [8..11] <-> [12..15] within each u64 lane
  i8x16_shuffle::<4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11>(x, x)
}

/// Rotate right by 24: (x >> 24) | (x << 40).
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn ror24(x: v128) -> v128 {
  v128_or(u64x2_shr(x, 24), u64x2_shl(x, 40))
}

/// Rotate right by 16: byte shuffle swapping 2-byte halves.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn ror16(x: v128) -> v128 {
  // Rotate each u64 lane right by 2 bytes
  i8x16_shuffle::<2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9>(x, x)
}

/// Rotate right by 63: (x >> 63) | (x << 1).
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn ror63(x: v128) -> v128 {
  v128_or(u64x2_shr(x, 63), u64x2_shl(x, 1))
}

// ─── G function on SIMD register pairs ────────────────────────────────────

/// Blake2b G mixing on SIMD rows (2-wide).
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn g2(state: &mut WorkingState, mx: [v128; 2], my: [v128; 2]) {
  // a += b + mx
  state.a[0] = i64x2_add(i64x2_add(state.a[0], state.b[0]), mx[0]);
  state.a[1] = i64x2_add(i64x2_add(state.a[1], state.b[1]), mx[1]);
  // d = (d ^ a) >>> 32
  state.d[0] = ror32(v128_xor(state.d[0], state.a[0]));
  state.d[1] = ror32(v128_xor(state.d[1], state.a[1]));
  // c += d
  state.c[0] = i64x2_add(state.c[0], state.d[0]);
  state.c[1] = i64x2_add(state.c[1], state.d[1]);
  // b = (b ^ c) >>> 24
  state.b[0] = ror24(v128_xor(state.b[0], state.c[0]));
  state.b[1] = ror24(v128_xor(state.b[1], state.c[1]));
  // a += b + my
  state.a[0] = i64x2_add(i64x2_add(state.a[0], state.b[0]), my[0]);
  state.a[1] = i64x2_add(i64x2_add(state.a[1], state.b[1]), my[1]);
  // d = (d ^ a) >>> 16
  state.d[0] = ror16(v128_xor(state.d[0], state.a[0]));
  state.d[1] = ror16(v128_xor(state.d[1], state.a[1]));
  // c += d
  state.c[0] = i64x2_add(state.c[0], state.d[0]);
  state.c[1] = i64x2_add(state.c[1], state.d[1]);
  // b = (b ^ c) >>> 63
  state.b[0] = ror63(v128_xor(state.b[0], state.c[0]));
  state.b[1] = ror63(v128_xor(state.b[1], state.c[1]));
}

#[cfg(target_arch = "wasm32")]
struct WorkingState {
  a: [v128; 2],
  b: [v128; 2],
  c: [v128; 2],
  d: [v128; 2],
}

// ─── Diagonalize / Un-diagonalize ─────────────────────────────────────────

/// Diagonalize: rotate row B left by 1, row C by 2 (swap), row D right by 1.
///
/// `i64x2_shuffle` indices: 0,1 = lanes from first operand, 2,3 = from second.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn diagonalize(state: &mut WorkingState) {
  // B: rotate left 1: (v4,v5,v6,v7) -> (v5,v6,v7,v4)
  let tb0 = state.b[0];
  let tb1 = state.b[1];
  state.b[0] = i64x2_shuffle::<1, 2>(tb0, tb1); // [b0[1], b1[0]] = [v5, v6]
  state.b[1] = i64x2_shuffle::<1, 2>(tb1, tb0); // [b1[1], b0[0]] = [v7, v4]

  // C: rotate left 2 = swap lo/hi
  state.c.swap(0, 1);

  // D: rotate left 3 = rotate right 1: (v12,v13,v14,v15) -> (v15,v12,v13,v14)
  let td0 = state.d[0];
  let td1 = state.d[1];
  state.d[0] = i64x2_shuffle::<1, 2>(td1, td0); // [d1[1], d0[0]] = [v15, v12]
  state.d[1] = i64x2_shuffle::<1, 2>(td0, td1); // [d0[1], d1[0]] = [v13, v14]
}

/// Un-diagonalize: reverse the rotations.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn undiagonalize(state: &mut WorkingState) {
  // B: rotate right 1 (undo left 1)
  let tb0 = state.b[0];
  let tb1 = state.b[1];
  state.b[0] = i64x2_shuffle::<1, 2>(tb1, tb0);
  state.b[1] = i64x2_shuffle::<1, 2>(tb0, tb1);

  // C: swap back
  state.c.swap(0, 1);

  // D: rotate left 1 (undo right 1)
  let td0 = state.d[0];
  let td1 = state.d[1];
  state.d[0] = i64x2_shuffle::<1, 2>(td0, td1);
  state.d[1] = i64x2_shuffle::<1, 2>(td1, td0);
}

// ─── Load helpers ─────────────────────────────────────────────────────────

/// Create a v128 from two message words by index.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn load_msg_pair(m: &[u64; 16], i0: u8, i1: u8) -> v128 {
  u64x2(m[i0 as usize], m[i1 as usize])
}

/// Load two consecutive words into a SIMD vector.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn load_u64_pair<const N: usize>(words: &[u64; N], offset: usize) -> v128 {
  u64x2(words[offset], words[offset.strict_add(1)])
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn store_u64_pair(words: &mut [u64; 8], offset: usize, value: v128) {
  words[offset] = u64x2_extract_lane::<0>(value);
  words[offset.strict_add(1)] = u64x2_extract_lane::<1>(value);
}

// ─── Compress entry point ─────────────────────────────────────────────────

/// Blake2b WASM SIMD128-accelerated compress.
///
/// # Safety
///
/// Caller must ensure WASM SIMD128 is available.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub(super) unsafe fn compress_simd128(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  let m = load_msg(block);
  let v = init_v(h, t, last);

  // Pack into 2-wide SIMD rows: (lo, hi) for each row.
  let mut state = WorkingState {
    a: [load_u64_pair(&v, 0), load_u64_pair(&v, 2)],
    b: [load_u64_pair(&v, 4), load_u64_pair(&v, 6)],
    c: [load_u64_pair(&v, 8), load_u64_pair(&v, 10)],
    d: [load_u64_pair(&v, 12), load_u64_pair(&v, 14)],
  };

  // 12 rounds
  for round in 0..12u8 {
    let s = &SIGMA[(round % 10) as usize];

    // Column step
    let mx0 = load_msg_pair(&m, s[0], s[2]);
    let mx1 = load_msg_pair(&m, s[4], s[6]);
    let my0 = load_msg_pair(&m, s[1], s[3]);
    let my1 = load_msg_pair(&m, s[5], s[7]);

    g2(&mut state, [mx0, mx1], [my0, my1]);

    diagonalize(&mut state);

    // Diagonal step
    let mx0 = load_msg_pair(&m, s[8], s[10]);
    let mx1 = load_msg_pair(&m, s[12], s[14]);
    let my0 = load_msg_pair(&m, s[9], s[11]);
    let my1 = load_msg_pair(&m, s[13], s[15]);

    g2(&mut state, [mx0, mx1], [my0, my1]);

    undiagonalize(&mut state);
  }

  // Finalize: h[i] ^= v[i] ^ v[i+8]
  let h0 = load_u64_pair(h, 0);
  let h1 = load_u64_pair(h, 2);
  let h2 = load_u64_pair(h, 4);
  let h3 = load_u64_pair(h, 6);

  let r0 = v128_xor(h0, v128_xor(state.a[0], state.c[0]));
  let r1 = v128_xor(h1, v128_xor(state.a[1], state.c[1]));
  let r2 = v128_xor(h2, v128_xor(state.b[0], state.d[0]));
  let r3 = v128_xor(h3, v128_xor(state.b[1], state.d[1]));

  store_u64_pair(h, 0, r0);
  store_u64_pair(h, 2, r1);
  store_u64_pair(h, 4, r2);
  store_u64_pair(h, 6, r3);
}
