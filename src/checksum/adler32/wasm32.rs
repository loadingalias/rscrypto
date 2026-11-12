//! wasm32 SIMD128 Adler-32 kernel.
//!
//! The module is compiled only when the artifact requires SIMD128, so the
//! engine that runs the artifact supports these instructions. It computes the
//! same sums as the portable kernel, 64 bytes at a time: pairwise widening adds
//! for the byte sum, widening multiplies by the position weights for the
//! weighted sum, and the running total of the byte sum before each block.
//! Loads accept byte-aligned input; no relaxed-SIMD instructions are used.

use core::arch::wasm32::{
  u16x8_add, u16x8_extadd_pairwise_u8x16, u16x8_extmul_high_u8x16, u16x8_extmul_low_u8x16, u32x4_add,
  u32x4_extadd_pairwise_u16x8, u32x4_extract_lane, u32x4_splat, v128, v128_load,
};

use super::portable::{self, reduce};

/// Canonical name of the SIMD128 kernel.
pub(in crate::checksum) const KERNEL_NAME: &str = "wasm32/simd128";

/// Vectors of 16 bytes per block of the main loop.
const VECTORS: usize = 4;
/// Blocks between reductions.
const CHUNK_BLOCKS: usize = 256;
/// Every 32-bit sum of a chunk stays at most `u32::MAX` across all lanes, so each lane does
/// too: a block of `16 * VECTORS` bytes adds at most `255 * 16 * VECTORS` to the byte sum, the
/// running total of the byte sum before each block stays at most that times `CHUNK_BLOCKS^2 /
/// 2`, and the weighted total at most `255 * CHUNK_BLOCKS * Σ weights`. A 16-bit lane holds
/// either the byte pairs of a block, at most `2 * 255 * VECTORS`, or two bytes times weights of
/// at most `16 * VECTORS`.
const CHUNK_SUMS_FIT: () = {
  let (blocks, block) = (CHUNK_BLOCKS as u64, 16 * VECTORS as u64);
  assert!(255 * block * blocks * (blocks + 1) / 2 <= u32::MAX as u64);
  assert!(255 * blocks * block * (block + 1) / 2 <= u32::MAX as u64);
  assert!(2 * 255 * block <= u16::MAX as u64);
};

/// Load 16 bytes.
#[inline(always)]
fn load(bytes: &[u8; 16]) -> v128 {
  // SAFETY: `bytes` refers to 16 initialized bytes, so the 16-byte read stays in bounds, and
  // `v128_load` accepts any alignment.
  unsafe { v128_load(bytes.as_ptr().cast()) }
}

/// Position weights `16·N - i` of an `N`-vector block.
#[expect(clippy::cast_possible_truncation, reason = "a weight is at most 16·N = 64")]
const fn weights<const N: usize>() -> [[u8; 16]; N] {
  let mut out = [[0u8; 16]; N];
  let mut i = 0;
  while i < N.strict_mul(16) {
    out[i / 16][i % 16] = N.strict_mul(16).strict_sub(i) as u8;
    i = i.strict_add(1);
  }
  out
}

/// Sum of the four lanes.
#[inline(always)]
fn horizontal_sum(value: v128) -> u32 {
  u32x4_extract_lane::<0>(value)
    .strict_add(u32x4_extract_lane::<1>(value))
    .strict_add(u32x4_extract_lane::<2>(value))
    .strict_add(u32x4_extract_lane::<3>(value))
}

/// Add the whole `16·N`-byte blocks of `vectors` to the reduced sums `a` and `b`.
///
/// For `k` blocks, `a` gains the byte sum and `b` gains `16·N·k·a`, plus `16·N` times the
/// running total of the byte sum before each block, plus each byte times its position
/// weight `16·N - i`.
#[inline(always)]
fn blocks<const N: usize>(mut a: u32, mut b: u32, vectors: &[[u8; 16]]) -> (u32, u32) {
  // Under `CHUNK_SUMS_FIT`, no lane addition or horizontal sum below wraps, so each wrapping
  // lane add equals the exact add.
  let () = CHUNK_SUMS_FIT;
  let weights = const { weights::<N>() }.map(|weight| load(&weight));
  for chunk in vectors.chunks(N.strict_mul(CHUNK_BLOCKS)) {
    let mut sum = u32x4_splat(0);
    let mut running = u32x4_splat(0);
    // One weighted accumulator per vector keeps their additions independent.
    let mut weighted = [u32x4_splat(0); N];
    for block in chunk.as_chunks::<N>().0 {
      running = u32x4_add(running, sum);
      let mut pairs = u32x4_splat(0);
      for ((vector, &weight), weighted) in block.iter().zip(&weights).zip(&mut weighted) {
        let bytes = load(vector);
        pairs = u16x8_add(pairs, u16x8_extadd_pairwise_u8x16(bytes));
        let products = u16x8_add(
          u16x8_extmul_low_u8x16(bytes, weight),
          u16x8_extmul_high_u8x16(bytes, weight),
        );
        *weighted = u32x4_add(*weighted, u32x4_extadd_pairwise_u16x8(products));
      }
      sum = u32x4_add(sum, u32x4_extadd_pairwise_u16x8(pairs));
    }
    let weighted = weighted
      .into_iter()
      .fold(u32x4_splat(0), |total, part| u32x4_add(total, part));
    let len = (chunk.len() as u64).strict_mul(16);
    let b_wide = u64::from(b)
      .strict_add(len.strict_mul(u64::from(a)))
      .strict_add((N as u64).strict_mul(16).strict_mul(u64::from(horizontal_sum(running))))
      .strict_add(u64::from(horizontal_sum(weighted)));
    a = reduce(u64::from(a).strict_add(u64::from(horizontal_sum(sum))));
    b = reduce(b_wide);
  }
  (a, b)
}

/// Add `data` to the Adler-32 value `adler`, whose two sums are reduced.
pub(in crate::checksum) fn update(adler: u32, data: &[u8]) -> u32 {
  let (vectors, bytes) = data.as_chunks::<16>();
  let whole = vectors.len().strict_sub(vectors.len() % VECTORS);
  let (large, small) = vectors.split_at(whole);
  let (a, b) = blocks::<VECTORS>(adler & 0xffff, adler >> 16, large);
  let (a, b) = blocks::<1>(a, b, small);
  portable::update((b << 16) | a, bytes)
}
