//! Blake2s POWER VSX-accelerated compression for powerpc64.
//!
//! This backend keeps the working state in canonical `u32` words and expresses
//! the row operations with `core::simd::u32x4`. That keeps the implementation
//! endian-agnostic across both big- and little-endian POWER while still
//! letting LLVM lower the row math and lane shuffles to VSX.

#![allow(clippy::indexing_slicing)]

use core::simd::u32x4;

use super::kernels::{SIGMA, init_v, load_msg};

#[inline(always)]
fn rotr32<const N: u32>(v: u32x4) -> u32x4 {
  const { assert!(N > 0 && N < 32) }
  let s0 = u32x4::splat(N);
  let s1 = u32x4::splat(32 - N);
  (v >> s0) | (v << s1)
}

#[inline(always)]
fn rot_lanes_left_1(v: u32x4) -> u32x4 {
  core::simd::simd_swizzle!(v, [1, 2, 3, 0])
}

#[inline(always)]
fn rot_lanes_left_2(v: u32x4) -> u32x4 {
  core::simd::simd_swizzle!(v, [2, 3, 0, 1])
}

#[inline(always)]
fn rot_lanes_left_3(v: u32x4) -> u32x4 {
  core::simd::simd_swizzle!(v, [3, 0, 1, 2])
}

#[inline(always)]
fn load_msg_quad(m: &[u32; 16], i0: u8, i1: u8, i2: u8, i3: u8) -> u32x4 {
  u32x4::from_array([m[i0 as usize], m[i1 as usize], m[i2 as usize], m[i3 as usize]])
}

#[inline(always)]
fn g4(a: &mut u32x4, b: &mut u32x4, c: &mut u32x4, d: &mut u32x4, mx: u32x4, my: u32x4) {
  *a += *b;
  *a += mx;
  *d ^= *a;
  *d = rotr32::<16>(*d);
  *c += *d;
  *b ^= *c;
  *b = rotr32::<12>(*b);
  *a += *b;
  *a += my;
  *d ^= *a;
  *d = rotr32::<8>(*d);
  *c += *d;
  *b ^= *c;
  *b = rotr32::<7>(*b);
}

#[inline(always)]
fn diagonalize(b: &mut u32x4, c: &mut u32x4, d: &mut u32x4) {
  *b = rot_lanes_left_1(*b);
  *c = rot_lanes_left_2(*c);
  *d = rot_lanes_left_3(*d);
}

#[inline(always)]
fn undiagonalize(b: &mut u32x4, c: &mut u32x4, d: &mut u32x4) {
  *b = rot_lanes_left_3(*b);
  *c = rot_lanes_left_2(*c);
  *d = rot_lanes_left_1(*d);
}

/// Blake2s POWER VSX-accelerated compress.
///
/// # Safety
///
/// Caller must ensure POWER8+ with VSX is available.
#[target_feature(enable = "vsx")]
pub(super) unsafe fn compress_vsx(h: &mut [u32; 8], block: &[u8; 64], t: u64, last: bool) {
  let m = load_msg(block);
  let v = init_v(h, t, last);

  let mut a = u32x4::from_array([v[0], v[1], v[2], v[3]]);
  let mut b = u32x4::from_array([v[4], v[5], v[6], v[7]]);
  let mut c = u32x4::from_array([v[8], v[9], v[10], v[11]]);
  let mut d = u32x4::from_array([v[12], v[13], v[14], v[15]]);

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

  let h0 = u32x4::from_array([h[0], h[1], h[2], h[3]]);
  let h1 = u32x4::from_array([h[4], h[5], h[6], h[7]]);
  let r0 = h0 ^ a ^ c;
  let r1 = h1 ^ b ^ d;
  h[..4].copy_from_slice(&r0.to_array());
  h[4..].copy_from_slice(&r1.to_array());
}

#[cfg(test)]
mod tests {
  use core::simd::u32x4;

  use super::{diagonalize, undiagonalize};

  #[test]
  fn diagonalize_matches_blake2s_lane_rotation() {
    if !crate::platform::caps().has(crate::platform::caps::power::VSX) {
      return;
    }

    let mut b = u32x4::from_array([4, 5, 6, 7]);
    let mut c = u32x4::from_array([8, 9, 10, 11]);
    let mut d = u32x4::from_array([12, 13, 14, 15]);
    let original = (b, c, d);

    diagonalize(&mut b, &mut c, &mut d);
    assert_eq!(b.to_array(), [5, 6, 7, 4]);
    assert_eq!(c.to_array(), [10, 11, 8, 9]);
    assert_eq!(d.to_array(), [15, 12, 13, 14]);

    undiagonalize(&mut b, &mut c, &mut d);
    assert_eq!((b, c, d), original);
  }
}
