//! Blake2s WebAssembly SIMD128 accelerated compression.
//!
//! Blake2s is a natural fit for `simd128`: each 4-word row lives in one
//! `v128`, diagonalization is lane shuffling, and the 32-bit rotates map to
//! byte shuffles plus shift/or.

#![allow(clippy::indexing_slicing)]

use core::arch::wasm32::*;

use super::kernels::{SIGMA, init_v, load_msg};

#[inline(always)]
fn ror16(x: v128) -> v128 {
  i8x16_shuffle::<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>(x, x)
}

#[inline(always)]
fn ror12(x: v128) -> v128 {
  v128_or(u32x4_shr(x, 12), u32x4_shl(x, 20))
}

#[inline(always)]
fn ror8(x: v128) -> v128 {
  i8x16_shuffle::<1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12>(x, x)
}

#[inline(always)]
fn ror7(x: v128) -> v128 {
  v128_or(u32x4_shr(x, 7), u32x4_shl(x, 25))
}

#[inline(always)]
fn g4(a: &mut v128, b: &mut v128, c: &mut v128, d: &mut v128, mx: v128, my: v128) {
  *a = u32x4_add(u32x4_add(*a, *b), mx);
  *d = ror16(v128_xor(*d, *a));
  *c = u32x4_add(*c, *d);
  *b = ror12(v128_xor(*b, *c));
  *a = u32x4_add(u32x4_add(*a, *b), my);
  *d = ror8(v128_xor(*d, *a));
  *c = u32x4_add(*c, *d);
  *b = ror7(v128_xor(*b, *c));
}

#[inline(always)]
fn diagonalize(b: &mut v128, c: &mut v128, d: &mut v128) {
  *b = u32x4_shuffle::<1, 2, 3, 0>(*b, *b);
  *c = u32x4_shuffle::<2, 3, 0, 1>(*c, *c);
  *d = u32x4_shuffle::<3, 0, 1, 2>(*d, *d);
}

#[inline(always)]
fn undiagonalize(b: &mut v128, c: &mut v128, d: &mut v128) {
  *b = u32x4_shuffle::<3, 0, 1, 2>(*b, *b);
  *c = u32x4_shuffle::<2, 3, 0, 1>(*c, *c);
  *d = u32x4_shuffle::<1, 2, 3, 0>(*d, *d);
}

#[inline(always)]
fn load_msg_quad(m: &[u32; 16], i0: u8, i1: u8, i2: u8, i3: u8) -> v128 {
  u32x4(m[i0 as usize], m[i1 as usize], m[i2 as usize], m[i3 as usize])
}

#[inline(always)]
unsafe fn vload_u32_quad(p: *const u32) -> v128 {
  // SAFETY: caller ensures `p` is valid for 16 bytes / 4 u32 lanes.
  unsafe { v128_load(p.cast()) }
}

/// Blake2s WASM SIMD128 compress.
///
/// # Safety
///
/// Caller must ensure WASM SIMD128 is available.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn compress_simd128(h: &mut [u32; 8], block: &[u8; 64], t: u64, last: bool) {
  let m = load_msg(block);
  let v = init_v(h, t, last);

  let mut a = unsafe { vload_u32_quad(v.as_ptr()) };
  let mut b = unsafe { vload_u32_quad(v.as_ptr().add(4)) };
  let mut c = unsafe { vload_u32_quad(v.as_ptr().add(8)) };
  let mut d = unsafe { vload_u32_quad(v.as_ptr().add(12)) };

  for round in 0..10u8 {
    let s = &SIGMA[round as usize];

    let mx = load_msg_quad(&m, s[0], s[2], s[4], s[6]);
    let my = load_msg_quad(&m, s[1], s[3], s[5], s[7]);
    g4(&mut a, &mut b, &mut c, &mut d, mx, my);

    diagonalize(&mut b, &mut c, &mut d);

    let mx = load_msg_quad(&m, s[8], s[10], s[12], s[14]);
    let my = load_msg_quad(&m, s[9], s[11], s[13], s[15]);
    g4(&mut a, &mut b, &mut c, &mut d, mx, my);

    undiagonalize(&mut b, &mut c, &mut d);
  }

  let h0 = unsafe { vload_u32_quad(h.as_ptr()) };
  let h1 = unsafe { vload_u32_quad(h.as_ptr().add(4)) };

  unsafe {
    v128_store(h.as_mut_ptr().cast(), v128_xor(h0, v128_xor(a, c)));
    v128_store(h.as_mut_ptr().add(4).cast(), v128_xor(h1, v128_xor(b, d)));
  }
}
