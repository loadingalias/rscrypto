//! Blake2b POWER VSX-accelerated compression for powerpc64.
//!
//! This backend keeps the working state in canonical `u64` words and uses
//! `core::simd::u64x2` row pairs plus lane swizzles for diagonalization. That
//! avoids any dependence on POWER byte-permute register layout, so the kernel
//! is correct on both big- and little-endian POWER while still mapping cleanly
//! to VSX-sized 128-bit vectors.

#![allow(clippy::indexing_slicing)]

use core::simd::u64x2;

use super::kernels::{SIGMA, init_v, load_msg};

type RowPair = [u64x2; 2];

#[inline(always)]
fn rotr64<const N: u32>(v: u64x2) -> u64x2 {
  const { assert!(N > 0 && N < 64) }
  let s0 = u64x2::splat(N as u64);
  let s1 = u64x2::splat((64 - N) as u64);
  (v >> s0) | (v << s1)
}

#[inline(always)]
fn load_msg_pair(m: &[u64; 16], i0: u8, i1: u8) -> u64x2 {
  u64x2::from_array([m[i0 as usize], m[i1 as usize]])
}

#[inline(always)]
fn pair_a1_b0(a: u64x2, b: u64x2) -> u64x2 {
  core::simd::simd_swizzle!(a, b, [1, 2])
}

#[inline(always)]
fn pair_b1_a0(a: u64x2, b: u64x2) -> u64x2 {
  core::simd::simd_swizzle!(a, b, [3, 0])
}

#[inline(always)]
fn g2(a: &mut RowPair, b: &mut RowPair, c: &mut RowPair, d: &mut RowPair, mx: RowPair, my: RowPair) {
  a[0] += b[0];
  a[0] += mx[0];
  a[1] += b[1];
  a[1] += mx[1];
  d[0] ^= a[0];
  d[1] ^= a[1];
  d[0] = rotr64::<32>(d[0]);
  d[1] = rotr64::<32>(d[1]);
  c[0] += d[0];
  c[1] += d[1];
  b[0] ^= c[0];
  b[1] ^= c[1];
  b[0] = rotr64::<24>(b[0]);
  b[1] = rotr64::<24>(b[1]);
  a[0] += b[0];
  a[0] += my[0];
  a[1] += b[1];
  a[1] += my[1];
  d[0] ^= a[0];
  d[1] ^= a[1];
  d[0] = rotr64::<16>(d[0]);
  d[1] = rotr64::<16>(d[1]);
  c[0] += d[0];
  c[1] += d[1];
  b[0] ^= c[0];
  b[1] ^= c[1];
  b[0] = rotr64::<63>(b[0]);
  b[1] = rotr64::<63>(b[1]);
}

#[inline(always)]
fn diagonalize(b: &mut RowPair, c: &mut RowPair, d: &mut RowPair) {
  let tb0 = b[0];
  let tb1 = b[1];
  b[0] = pair_a1_b0(tb0, tb1);
  b[1] = pair_b1_a0(tb0, tb1);
  c.swap(0, 1);
  let td0 = d[0];
  let td1 = d[1];
  d[0] = pair_b1_a0(td0, td1);
  d[1] = pair_a1_b0(td0, td1);
}

#[inline(always)]
fn undiagonalize(b: &mut RowPair, c: &mut RowPair, d: &mut RowPair) {
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

/// Blake2b POWER VSX-accelerated compress.
///
/// # Safety
///
/// Caller must ensure POWER8+ with VSX is available.
#[target_feature(enable = "vsx")]
pub(super) unsafe fn compress_vsx(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  let m = load_msg(block);
  let v = init_v(h, t, last);

  let mut a = [u64x2::from_array([v[0], v[1]]), u64x2::from_array([v[2], v[3]])];
  let mut b = [u64x2::from_array([v[4], v[5]]), u64x2::from_array([v[6], v[7]])];
  let mut c = [u64x2::from_array([v[8], v[9]]), u64x2::from_array([v[10], v[11]])];
  let mut d = [u64x2::from_array([v[12], v[13]]), u64x2::from_array([v[14], v[15]])];

  for round in 0..12u8 {
    let s = &SIGMA[(round % 10) as usize];
    let mx = [load_msg_pair(&m, s[0], s[2]), load_msg_pair(&m, s[4], s[6])];
    let my = [load_msg_pair(&m, s[1], s[3]), load_msg_pair(&m, s[5], s[7])];
    g2(&mut a, &mut b, &mut c, &mut d, mx, my);
    diagonalize(&mut b, &mut c, &mut d);

    let mx = [load_msg_pair(&m, s[8], s[10]), load_msg_pair(&m, s[12], s[14])];
    let my = [load_msg_pair(&m, s[9], s[11]), load_msg_pair(&m, s[13], s[15])];
    g2(&mut a, &mut b, &mut c, &mut d, mx, my);
    undiagonalize(&mut b, &mut c, &mut d);
  }

  let h0 = u64x2::from_array([h[0], h[1]]);
  let h1 = u64x2::from_array([h[2], h[3]]);
  let h2 = u64x2::from_array([h[4], h[5]]);
  let h3 = u64x2::from_array([h[6], h[7]]);
  let r0 = h0 ^ a[0] ^ c[0];
  let r1 = h1 ^ a[1] ^ c[1];
  let r2 = h2 ^ b[0] ^ d[0];
  let r3 = h3 ^ b[1] ^ d[1];
  h[..2].copy_from_slice(&r0.to_array());
  h[2..4].copy_from_slice(&r1.to_array());
  h[4..6].copy_from_slice(&r2.to_array());
  h[6..].copy_from_slice(&r3.to_array());
}

#[cfg(test)]
mod tests {
  use core::simd::u64x2;

  use super::{RowPair, diagonalize, pair_a1_b0, pair_b1_a0, undiagonalize};

  #[test]
  fn lane_selectors_match_expected_pairs() {
    if !crate::platform::caps().has(crate::platform::caps::power::VSX) {
      return;
    }

    let a = u64x2::from_array([10, 11]);
    let b = u64x2::from_array([20, 21]);
    assert_eq!(pair_a1_b0(a, b).to_array(), [11, 20]);
    assert_eq!(pair_b1_a0(a, b).to_array(), [21, 10]);
  }

  #[test]
  fn diagonalize_round_trip_restores_rows() {
    if !crate::platform::caps().has(crate::platform::caps::power::VSX) {
      return;
    }

    let mut b: RowPair = [u64x2::from_array([4, 5]), u64x2::from_array([6, 7])];
    let mut c: RowPair = [u64x2::from_array([8, 9]), u64x2::from_array([10, 11])];
    let mut d: RowPair = [u64x2::from_array([12, 13]), u64x2::from_array([14, 15])];
    let original = (b, c, d);

    diagonalize(&mut b, &mut c, &mut d);
    assert_eq!(b[0].to_array(), [5, 6]);
    assert_eq!(b[1].to_array(), [7, 4]);
    assert_eq!(c[0].to_array(), [10, 11]);
    assert_eq!(c[1].to_array(), [8, 9]);
    assert_eq!(d[0].to_array(), [15, 12]);
    assert_eq!(d[1].to_array(), [13, 14]);

    undiagonalize(&mut b, &mut c, &mut d);
    assert_eq!((b, c, d), original);
  }
}
