//! aarch64 NEON Adler-32 kernel.
//!
//! The kernel computes the same sums as the portable kernel, 64 bytes at a
//! time: pairwise widening adds for the byte sum, 16-bit column sums for the
//! position weights, and the running total of the byte sum before each block.
//! Inputs of fewer than 256 bytes in blocks weight each block directly instead.
//! With the dot-product extension, [`update_dotprod`] takes one `UDOT` for the
//! byte sum and one for the weighted sum of each 16 bytes. Every aarch64
//! target in the catalog enables NEON at compile time, and this module compiles
//! only when `target_feature = "neon"` holds, so [`update_safe`] needs no
//! runtime check.

use core::arch::aarch64::{
  uint8x16_t, vaddq_u32, vaddvq_u32, vaddw_high_u8, vaddw_u8, vdotq_u32, vdupq_n_u8, vdupq_n_u16, vdupq_n_u32,
  vget_low_u8, vget_low_u16, vld1q_u8, vmlal_high_u8, vmlal_high_u16, vmovl_high_u8, vmovl_u8, vmull_u8, vmull_u16,
  vpadalq_u8, vpadalq_u16, vpaddlq_u8, vpaddq_u32,
};

use super::portable::{MOD, reduce};

/// Canonical name of the NEON kernel.
pub(in crate::checksum) const KERNEL_NAME: &str = "aarch64/neon";
/// Canonical name of the NEON kernel with the dot-product extension.
pub(in crate::checksum) const DOTPROD_KERNEL_NAME: &str = "aarch64/neon-dotprod";

/// Vectors of 16 bytes per block of the main loop.
const VECTORS: usize = 4;
/// Below this many vectors in blocks, weighting each block directly costs less than column
/// sums.
const FEW_VECTORS: usize = 16;
/// Blocks between reductions.
const CHUNK_BLOCKS: usize = 256;
/// Every 16-bit column sum of a chunk stays at most `u16::MAX`, and every 32-bit sum of the
/// chunk at most `u32::MAX` across all lanes, so each lane does too: a block of
/// `16 * VECTORS` bytes adds at most `255 * 16 * VECTORS` to the byte sum, the running total of
/// the byte sum before each block stays at most that times `CHUNK_BLOCKS^2 / 2`, and the
/// weighted total at most `255 * CHUNK_BLOCKS * Σ weights`. A 16-bit product lane of
/// [`few_blocks`] holds two bytes times weights of at most `16 * VECTORS`.
const CHUNK_SUMS_FIT: () = assert!(
  255 * CHUNK_BLOCKS <= u16::MAX as usize
    && 255 * 16 * VECTORS * CHUNK_BLOCKS * (CHUNK_BLOCKS + 1) / 2 <= u32::MAX as usize
    && 255 * CHUNK_BLOCKS * (16 * VECTORS) * (16 * VECTORS + 1) / 2 <= u32::MAX as usize
    && 2 * 255 * 16 * VECTORS <= u16::MAX as usize
);

/// Load 16 bytes.
///
/// # Safety
///
/// The CPU must support NEON.
#[inline]
#[target_feature(enable = "neon")]
fn load(bytes: &[u8; 16]) -> uint8x16_t {
  // SAFETY: `bytes` refers to 16 initialized bytes, so the 16-byte read stays in bounds, and
  // `vld1q_u8` has no alignment requirement.
  unsafe { vld1q_u8(bytes.as_ptr()) }
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

/// Add the whole `16·N`-byte blocks of `vectors` to the reduced sums `a` and `b`.
///
/// For `k` blocks, `a` gains the byte sum and `b` gains `16·N·k·a`, plus `16·N` times the
/// running total of the byte sum before each block, plus each byte times its position
/// weight `16·N - i`. Each block adds its bytes into 16-bit column sums, one per position;
/// the column sums meet the weights once per chunk.
///
/// # Safety
///
/// The CPU must support NEON.
// Out of line, so short inputs do not pay for saving the vector registers that this loop uses.
#[inline(never)]
#[target_feature(enable = "neon")]
fn blocks<const N: usize>(mut a: u32, mut b: u32, vectors: &[[u8; 16]]) -> (u32, u32) {
  // Under `CHUNK_SUMS_FIT`, no lane addition or horizontal sum below wraps, so each wrapping
  // lane add equals the exact add.
  let () = CHUNK_SUMS_FIT;
  for chunk in vectors.chunks(N.strict_mul(CHUNK_BLOCKS)) {
    let mut sum = vdupq_n_u32(0);
    let mut running = vdupq_n_u32(0);
    let mut columns = [[vdupq_n_u16(0); 2]; N];
    for block in chunk.as_chunks::<N>().0 {
      running = vaddq_u32(running, sum);
      let bytes = block.each_ref().map(|vector| load(vector));
      for (&bytes, column) in bytes.iter().zip(&mut columns) {
        column[0] = vaddw_u8(column[0], vget_low_u8(bytes));
        column[1] = vaddw_high_u8(column[1], bytes);
      }
      let mut pairs = vpaddlq_u8(bytes[0]);
      for &bytes in &bytes[1..] {
        pairs = vpadalq_u8(pairs, bytes);
      }
      sum = vpadalq_u16(sum, pairs);
    }
    // One product per column vector, joined by pairwise adds, which the compiler cannot fuse
    // into one long multiply-accumulate chain. Each pairwise add keeps the sum of all lanes.
    let weights = const { weights::<N>() };
    let mut products = [[vdupq_n_u32(0); 2]; N];
    for ((column, weight), product) in columns.iter().zip(&weights).zip(&mut products) {
      let weight = load(weight);
      let halves = [vmovl_u8(vget_low_u8(weight)), vmovl_high_u8(weight)];
      for ((&column, &weight), product) in column.iter().zip(&halves).zip(product) {
        *product = vmlal_high_u16(vmull_u16(vget_low_u16(column), vget_low_u16(weight)), column, weight);
      }
    }
    let terms = products.as_flattened_mut();
    let mut len = terms.len();
    while len > 1 {
      len /= 2;
      for i in 0..len {
        let pair = i.strict_mul(2);
        terms[i] = vpaddq_u32(terms[pair], terms[pair.strict_add(1)]);
      }
    }
    let len = (chunk.len() as u64).strict_mul(16);
    let b_wide = u64::from(b)
      .strict_add(len.strict_mul(u64::from(a)))
      .strict_add((N as u64).strict_mul(16).strict_mul(u64::from(vaddvq_u32(running))))
      .strict_add(u64::from(vaddvq_u32(terms[0])));
    a = reduce(u64::from(a).strict_add(u64::from(vaddvq_u32(sum))));
    b = reduce(b_wide);
  }
  (a, b)
}

/// Add the whole `16·N`-byte blocks of `vectors` to the reduced sums `a` and `b`, weighting
/// each block's bytes directly.
///
/// This computes the same sums as [`blocks`] with more work per block but none per chunk,
/// which is cheaper for a few blocks.
///
/// # Safety
///
/// The CPU must support NEON.
#[inline]
#[target_feature(enable = "neon")]
fn few_blocks<const N: usize>(mut a: u32, mut b: u32, vectors: &[[u8; 16]]) -> (u32, u32) {
  // Under `CHUNK_SUMS_FIT`, no lane addition or horizontal sum below wraps, and a 16-bit
  // product lane holds two bytes times weights of at most `16·N`.
  let () = CHUNK_SUMS_FIT;
  let weights = const { weights::<N>() };
  let weights = weights.each_ref().map(|weight| load(weight));
  for chunk in vectors.chunks(N.strict_mul(CHUNK_BLOCKS)) {
    let mut sum = vdupq_n_u32(0);
    let mut running = vdupq_n_u32(0);
    // One weighted accumulator per vector keeps their additions independent.
    let mut weighted = [vdupq_n_u32(0); N];
    for block in chunk.as_chunks::<N>().0 {
      running = vaddq_u32(running, sum);
      let bytes = block.each_ref().map(|vector| load(vector));
      for ((&bytes, &weight), weighted) in bytes.iter().zip(&weights).zip(&mut weighted) {
        let products = vmlal_high_u8(vmull_u8(vget_low_u8(bytes), vget_low_u8(weight)), bytes, weight);
        *weighted = vpadalq_u16(*weighted, products);
      }
      let mut pairs = vpaddlq_u8(bytes[0]);
      for &bytes in &bytes[1..] {
        pairs = vpadalq_u8(pairs, bytes);
      }
      sum = vpadalq_u16(sum, pairs);
    }
    let weighted = weighted
      .into_iter()
      .fold(vdupq_n_u32(0), |total, part| vaddq_u32(total, part));
    let len = (chunk.len() as u64).strict_mul(16);
    let b_wide = u64::from(b)
      .strict_add(len.strict_mul(u64::from(a)))
      .strict_add((N as u64).strict_mul(16).strict_mul(u64::from(vaddvq_u32(running))))
      .strict_add(u64::from(vaddvq_u32(weighted)));
    a = reduce(u64::from(a).strict_add(u64::from(vaddvq_u32(sum))));
    b = reduce(b_wide);
  }
  (a, b)
}

/// Add `data` to the Adler-32 value `adler`, whose two sums are reduced.
pub(in crate::checksum) fn update_safe(adler: u32, data: &[u8]) -> u32 {
  // SAFETY: This module compiles only with `target_feature = "neon"`, so every CPU that runs
  // this code has NEON, the only feature that `update` enables.
  unsafe { update(adler, data) }
}

/// Add `data` to `adler` with NEON.
///
/// # Safety
///
/// The CPU must support NEON.
#[target_feature(enable = "neon")]
fn update(adler: u32, data: &[u8]) -> u32 {
  let (vectors, bytes) = data.as_chunks::<16>();
  let whole = vectors.len().strict_sub(vectors.len() % VECTORS);
  let (large, small) = vectors.split_at(whole);
  let (a, b) = if large.len() < FEW_VECTORS {
    few_blocks::<VECTORS>(adler & 0xffff, adler >> 16, large)
  } else {
    blocks::<VECTORS>(adler & 0xffff, adler >> 16, large)
  };
  let (mut a, mut b) = few_blocks::<1>(a, b, small);
  // At most 15 bytes remain, so neither sum can pass 2^32.
  for &byte in bytes {
    a = a.strict_add(u32::from(byte));
    b = b.strict_add(a);
  }
  if !bytes.is_empty() {
    a %= MOD;
    b %= MOD;
  }
  (b << 16) | a
}

/// Add `data` to the Adler-32 value `adler`, whose two sums are reduced, with the
/// dot-product extension.
///
/// Each 16 bytes take one `UDOT` against all-one bytes for the byte sum and one against the
/// position weights for the weighted sum.
///
/// # Safety
///
/// The CPU must support NEON and the dot-product extension (`FEAT_DotProd`).
#[target_feature(enable = "neon,dotprod")]
pub(in crate::checksum) fn update_dotprod(adler: u32, data: &[u8]) -> u32 {
  let (vectors, bytes) = data.as_chunks::<16>();
  let whole = vectors.len().strict_sub(vectors.len() % VECTORS);
  let (large, small) = vectors.split_at(whole);
  let (a, b) = dot_blocks::<VECTORS>(adler & 0xffff, adler >> 16, large);
  let (mut a, mut b) = dot_blocks::<1>(a, b, small);
  // At most 15 bytes remain, so neither sum can pass 2^32.
  for &byte in bytes {
    a = a.strict_add(u32::from(byte));
    b = b.strict_add(a);
  }
  if !bytes.is_empty() {
    a %= MOD;
    b %= MOD;
  }
  (b << 16) | a
}

/// [`few_blocks`] with `UDOT` for the byte sums and the weighted sums.
///
/// # Safety
///
/// The CPU must support NEON and the dot-product extension (`FEAT_DotProd`).
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn dot_blocks<const N: usize>(mut a: u32, mut b: u32, vectors: &[[u8; 16]]) -> (u32, u32) {
  // Under `CHUNK_SUMS_FIT`, no lane addition or horizontal sum below wraps, so each wrapping
  // lane add equals the exact add.
  let () = CHUNK_SUMS_FIT;
  let weights = const { weights::<N>() };
  let weights = weights.each_ref().map(|weight| load(weight));
  let ones = vdupq_n_u8(1);
  for chunk in vectors.chunks(N.strict_mul(CHUNK_BLOCKS)) {
    let mut sum = vdupq_n_u32(0);
    let mut running = vdupq_n_u32(0);
    // One weighted accumulator per vector keeps their dot products independent.
    let mut weighted = [vdupq_n_u32(0); N];
    for block in chunk.as_chunks::<N>().0 {
      running = vaddq_u32(running, sum);
      let bytes = block.each_ref().map(|vector| load(vector));
      let mut block_sum = vdupq_n_u32(0);
      for ((&bytes, &weight), weighted) in bytes.iter().zip(&weights).zip(&mut weighted) {
        *weighted = vdotq_u32(*weighted, bytes, weight);
        block_sum = vdotq_u32(block_sum, bytes, ones);
      }
      sum = vaddq_u32(sum, block_sum);
    }
    let weighted = weighted
      .into_iter()
      .fold(vdupq_n_u32(0), |total, part| vaddq_u32(total, part));
    let len = (chunk.len() as u64).strict_mul(16);
    let b_wide = u64::from(b)
      .strict_add(len.strict_mul(u64::from(a)))
      .strict_add((N as u64).strict_mul(16).strict_mul(u64::from(vaddvq_u32(running))))
      .strict_add(u64::from(vaddvq_u32(weighted)));
    a = reduce(u64::from(a).strict_add(u64::from(vaddvq_u32(sum))));
    b = reduce(b_wide);
  }
  (a, b)
}
