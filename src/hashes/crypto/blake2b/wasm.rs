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

#![allow(clippy::cast_possible_truncation, clippy::indexing_slicing)]

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
#[allow(clippy::too_many_arguments)]
fn g2(
  a0: &mut v128,
  a1: &mut v128,
  b0: &mut v128,
  b1: &mut v128,
  c0: &mut v128,
  c1: &mut v128,
  d0: &mut v128,
  d1: &mut v128,
  mx0: v128,
  mx1: v128,
  my0: v128,
  my1: v128,
) {
  // a += b + mx
  *a0 = i64x2_add(i64x2_add(*a0, *b0), mx0);
  *a1 = i64x2_add(i64x2_add(*a1, *b1), mx1);
  // d = (d ^ a) >>> 32
  *d0 = ror32(v128_xor(*d0, *a0));
  *d1 = ror32(v128_xor(*d1, *a1));
  // c += d
  *c0 = i64x2_add(*c0, *d0);
  *c1 = i64x2_add(*c1, *d1);
  // b = (b ^ c) >>> 24
  *b0 = ror24(v128_xor(*b0, *c0));
  *b1 = ror24(v128_xor(*b1, *c1));
  // a += b + my
  *a0 = i64x2_add(i64x2_add(*a0, *b0), my0);
  *a1 = i64x2_add(i64x2_add(*a1, *b1), my1);
  // d = (d ^ a) >>> 16
  *d0 = ror16(v128_xor(*d0, *a0));
  *d1 = ror16(v128_xor(*d1, *a1));
  // c += d
  *c0 = i64x2_add(*c0, *d0);
  *c1 = i64x2_add(*c1, *d1);
  // b = (b ^ c) >>> 63
  *b0 = ror63(v128_xor(*b0, *c0));
  *b1 = ror63(v128_xor(*b1, *c1));
}

// ─── Diagonalize / Un-diagonalize ─────────────────────────────────────────

/// Diagonalize: rotate row B left by 1, row C by 2 (swap), row D right by 1.
///
/// `i64x2_shuffle` indices: 0,1 = lanes from first operand, 2,3 = from second.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn diagonalize(b0: &mut v128, b1: &mut v128, c0: &mut v128, c1: &mut v128, d0: &mut v128, d1: &mut v128) {
  // B: rotate left 1: (v4,v5,v6,v7) -> (v5,v6,v7,v4)
  let tb0 = *b0;
  let tb1 = *b1;
  *b0 = i64x2_shuffle::<1, 2>(tb0, tb1); // [b0[1], b1[0]] = [v5, v6]
  *b1 = i64x2_shuffle::<1, 2>(tb1, tb0); // [b1[1], b0[0]] = [v7, v4]

  // C: rotate left 2 = swap lo/hi
  core::mem::swap(c0, c1);

  // D: rotate left 3 = rotate right 1: (v12,v13,v14,v15) -> (v15,v12,v13,v14)
  let td0 = *d0;
  let td1 = *d1;
  *d0 = i64x2_shuffle::<1, 2>(td1, td0); // [d1[1], d0[0]] = [v15, v12]
  *d1 = i64x2_shuffle::<1, 2>(td0, td1); // [d0[1], d1[0]] = [v13, v14]
}

/// Un-diagonalize: reverse the rotations.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn undiagonalize(b0: &mut v128, b1: &mut v128, c0: &mut v128, c1: &mut v128, d0: &mut v128, d1: &mut v128) {
  // B: rotate right 1 (undo left 1)
  let tb0 = *b0;
  let tb1 = *b1;
  *b0 = i64x2_shuffle::<1, 2>(tb1, tb0);
  *b1 = i64x2_shuffle::<1, 2>(tb0, tb1);

  // C: swap back
  core::mem::swap(c0, c1);

  // D: rotate left 1 (undo right 1)
  let td0 = *d0;
  let td1 = *d1;
  *d0 = i64x2_shuffle::<1, 2>(td0, td1);
  *d1 = i64x2_shuffle::<1, 2>(td1, td0);
}

// ─── Load helpers ─────────────────────────────────────────────────────────

/// Create a v128 from two message words by index.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn load_msg_pair(m: &[u64; 16], i0: u8, i1: u8) -> v128 {
  u64x2(m[i0 as usize], m[i1 as usize])
}

/// Load 2 consecutive u64 values from a pointer as v128.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn vload_u64_pair(p: *const u64) -> v128 {
  // SAFETY: caller ensures p is valid for 2 x u64 (16 bytes).
  unsafe { v128_load(p as *const v128) }
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

  // Pack into 2-wide SIMD rows: (lo, hi) for each row
  // SAFETY: v is a [u64; 16] — pointer arithmetic is within bounds.
  let mut a0 = unsafe { vload_u64_pair(v.as_ptr()) }; // v[0], v[1]
  let mut a1 = unsafe { vload_u64_pair(v.as_ptr().add(2)) }; // v[2], v[3]
  let mut b0 = unsafe { vload_u64_pair(v.as_ptr().add(4)) }; // v[4], v[5]
  let mut b1 = unsafe { vload_u64_pair(v.as_ptr().add(6)) }; // v[6], v[7]
  let mut c0 = unsafe { vload_u64_pair(v.as_ptr().add(8)) }; // v[8], v[9]
  let mut c1 = unsafe { vload_u64_pair(v.as_ptr().add(10)) }; // v[10], v[11]
  let mut d0 = unsafe { vload_u64_pair(v.as_ptr().add(12)) }; // v[12], v[13]
  let mut d1 = unsafe { vload_u64_pair(v.as_ptr().add(14)) }; // v[14], v[15]

  // 12 rounds
  for round in 0..12u8 {
    let s = &SIGMA[(round % 10) as usize];

    // Column step
    let mx0 = load_msg_pair(&m, s[0], s[2]);
    let mx1 = load_msg_pair(&m, s[4], s[6]);
    let my0 = load_msg_pair(&m, s[1], s[3]);
    let my1 = load_msg_pair(&m, s[5], s[7]);

    g2(
      &mut a0, &mut a1, &mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1, mx0, mx1, my0, my1,
    );

    diagonalize(&mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1);

    // Diagonal step
    let mx0 = load_msg_pair(&m, s[8], s[10]);
    let mx1 = load_msg_pair(&m, s[12], s[14]);
    let my0 = load_msg_pair(&m, s[9], s[11]);
    let my1 = load_msg_pair(&m, s[13], s[15]);

    g2(
      &mut a0, &mut a1, &mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1, mx0, mx1, my0, my1,
    );

    undiagonalize(&mut b0, &mut b1, &mut c0, &mut c1, &mut d0, &mut d1);
  }

  // Finalize: h[i] ^= v[i] ^ v[i+8]
  // SAFETY: h is a [u64; 8] — pointer arithmetic is within bounds.
  let h0 = unsafe { vload_u64_pair(h.as_ptr()) };
  let h1 = unsafe { vload_u64_pair(h.as_ptr().add(2)) };
  let h2 = unsafe { vload_u64_pair(h.as_ptr().add(4)) };
  let h3 = unsafe { vload_u64_pair(h.as_ptr().add(6)) };

  let r0 = v128_xor(h0, v128_xor(a0, c0));
  let r1 = v128_xor(h1, v128_xor(a1, c1));
  let r2 = v128_xor(h2, v128_xor(b0, d0));
  let r3 = v128_xor(h3, v128_xor(b1, d1));

  // SAFETY: h is a [u64; 8] — pointer arithmetic is within bounds.
  unsafe {
    v128_store(h.as_mut_ptr() as *mut v128, r0);
    v128_store(h.as_mut_ptr().add(2) as *mut v128, r1);
    v128_store(h.as_mut_ptr().add(4) as *mut v128, r2);
    v128_store(h.as_mut_ptr().add(6) as *mut v128, r3);
  }
}
