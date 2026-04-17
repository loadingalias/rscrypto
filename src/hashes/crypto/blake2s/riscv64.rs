//! Blake2s RISC-V V accelerated compression.
//!
//! This mirrors the Blake2b RVV approach in this repo: keep the row structure
//! explicit and gate the backend on `#[target_feature(enable = "v")]`, letting
//! the compiler lower the hot pair/quad operations appropriately for current
//! RISC-V vector hardware.

#![allow(clippy::indexing_slicing)]

use super::kernels::{SIGMA, init_v, load_msg};

#[inline(always)]
fn ror16(x: [u32; 4]) -> [u32; 4] {
  [
    x[0].rotate_right(16),
    x[1].rotate_right(16),
    x[2].rotate_right(16),
    x[3].rotate_right(16),
  ]
}

#[inline(always)]
fn ror12(x: [u32; 4]) -> [u32; 4] {
  [
    x[0].rotate_right(12),
    x[1].rotate_right(12),
    x[2].rotate_right(12),
    x[3].rotate_right(12),
  ]
}

#[inline(always)]
fn ror8(x: [u32; 4]) -> [u32; 4] {
  [
    x[0].rotate_right(8),
    x[1].rotate_right(8),
    x[2].rotate_right(8),
    x[3].rotate_right(8),
  ]
}

#[inline(always)]
fn ror7(x: [u32; 4]) -> [u32; 4] {
  [
    x[0].rotate_right(7),
    x[1].rotate_right(7),
    x[2].rotate_right(7),
    x[3].rotate_right(7),
  ]
}

#[inline(always)]
fn vadd(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
  [
    a[0].wrapping_add(b[0]),
    a[1].wrapping_add(b[1]),
    a[2].wrapping_add(b[2]),
    a[3].wrapping_add(b[3]),
  ]
}

#[inline(always)]
fn vxor(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
  [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
}

#[inline(always)]
fn g4(a: &mut [u32; 4], b: &mut [u32; 4], c: &mut [u32; 4], d: &mut [u32; 4], mx: [u32; 4], my: [u32; 4]) {
  *a = vadd(vadd(*a, *b), mx);
  *d = ror16(vxor(*d, *a));
  *c = vadd(*c, *d);
  *b = ror12(vxor(*b, *c));
  *a = vadd(vadd(*a, *b), my);
  *d = ror8(vxor(*d, *a));
  *c = vadd(*c, *d);
  *b = ror7(vxor(*b, *c));
}

#[inline(always)]
fn diagonalize(b: &mut [u32; 4], c: &mut [u32; 4], d: &mut [u32; 4]) {
  *b = [b[1], b[2], b[3], b[0]];
  *c = [c[2], c[3], c[0], c[1]];
  *d = [d[3], d[0], d[1], d[2]];
}

#[inline(always)]
fn undiagonalize(b: &mut [u32; 4], c: &mut [u32; 4], d: &mut [u32; 4]) {
  *b = [b[3], b[0], b[1], b[2]];
  *c = [c[2], c[3], c[0], c[1]];
  *d = [d[1], d[2], d[3], d[0]];
}

#[inline(always)]
fn load_msg_quad(m: &[u32; 16], i0: u8, i1: u8, i2: u8, i3: u8) -> [u32; 4] {
  [m[i0 as usize], m[i1 as usize], m[i2 as usize], m[i3 as usize]]
}

/// Blake2s RVV-selected compress.
///
/// # Safety
///
/// Caller must ensure the RISC-V V extension is available.
#[target_feature(enable = "v")]
pub(super) unsafe fn compress_rvv(h: &mut [u32; 8], block: &[u8; 64], t: u64, last: bool) {
  let m = load_msg(block);
  let v = init_v(h, t, last);

  let mut a = [v[0], v[1], v[2], v[3]];
  let mut b = [v[4], v[5], v[6], v[7]];
  let mut c = [v[8], v[9], v[10], v[11]];
  let mut d = [v[12], v[13], v[14], v[15]];

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

  h[0] ^= a[0] ^ c[0];
  h[1] ^= a[1] ^ c[1];
  h[2] ^= a[2] ^ c[2];
  h[3] ^= a[3] ^ c[3];
  h[4] ^= b[0] ^ d[0];
  h[5] ^= b[1] ^ d[1];
  h[6] ^= b[2] ^ d[2];
  h[7] ^= b[3] ^ d[3];
}
