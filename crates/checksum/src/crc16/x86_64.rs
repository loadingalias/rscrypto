//! x86_64 carryless-multiply CRC-16 kernels (PCLMULQDQ).
//!
//! These kernels implement reflected CRC-16 polynomials by lifting the 16-bit
//! state into the "width32" folding/reduction strategy (same structure as the
//! CRC-32 PCLMUL kernels, but with CRC-16-specific constants).
//!
//! # Safety
//!
//! Uses `unsafe` for x86 SIMD intrinsics. Callers must ensure SSSE3 + PCLMULQDQ
//! are available before executing these kernels (the dispatcher does this).
#![allow(unsafe_code)]
#![allow(clippy::indexing_slicing)]
#![allow(unsafe_op_in_unsafe_fn)]

use core::{
  arch::x86_64::*,
  ops::{BitXor, BitXorAssign},
};

use super::keys::{
  CRC16_CCITT_KEYS_REFLECTED, CRC16_CCITT_STREAM_REFLECTED, CRC16_IBM_KEYS_REFLECTED, CRC16_IBM_STREAM_REFLECTED,
  Width32StreamConstants,
};

#[repr(transparent)]
#[derive(Copy, Clone)]
struct Simd128(__m128i);

impl BitXor for Simd128 {
  type Output = Self;

  #[inline]
  fn bitxor(self, other: Self) -> Self {
    // SAFETY: `_mm_xor_si128` is available on all x86_64 (SSE2 baseline).
    unsafe { Self(_mm_xor_si128(self.0, other.0)) }
  }
}

impl BitXorAssign for Simd128 {
  #[inline]
  fn bitxor_assign(&mut self, other: Self) {
    *self = *self ^ other;
  }
}

impl Simd128 {
  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn new(high: u64, low: u64) -> Self {
    Self(_mm_set_epi64x(high as i64, low as i64))
  }

  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn shift_right_8(self) -> Self {
    Self(_mm_srli_si128::<8>(self.0))
  }

  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn shift_left_12(self) -> Self {
    Self(_mm_slli_si128::<12>(self.0))
  }

  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn and(self, mask: Self) -> Self {
    Self(_mm_and_si128(self.0, mask.0))
  }

  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn fold_16_reflected(self, coeff: Self, data_to_xor: Self) -> Self {
    let h = _mm_clmulepi64_si128::<0x10>(self.0, coeff.0);
    let l = _mm_clmulepi64_si128::<0x01>(self.0, coeff.0);
    Self(_mm_xor_si128(_mm_xor_si128(h, l), data_to_xor.0))
  }

  /// Fold 16 bytes down to the "width32" reduction state (reflected mode).
  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn fold_width32_reflected(self, high: u64, low: u64) -> Self {
    let coeff_low = Self::new(0, low);
    let coeff_high = Self::new(high, 0);

    // 16B -> 8B
    let clmul = _mm_clmulepi64_si128::<0x00>(self.0, coeff_low.0);
    let shifted = self.shift_right_8();
    let mut state = Self(_mm_xor_si128(clmul, shifted.0));

    // 8B -> 4B
    let mask2 = Self::new(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_0000_0000);
    let masked = state.and(mask2);
    let shifted = state.shift_left_12();
    let clmul = _mm_clmulepi64_si128::<0x11>(shifted.0, coeff_high.0);
    state = Self(_mm_xor_si128(clmul, masked.0));

    state
  }

  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn barrett_width32_reflected(self, poly: u64, mu: u64) -> u32 {
    let polymu = Self::new(poly, mu);
    let clmul1 = _mm_clmulepi64_si128::<0x00>(self.0, polymu.0);
    let clmul2 = _mm_clmulepi64_si128::<0x10>(clmul1, polymu.0);
    let xorred = _mm_xor_si128(self.0, clmul2);

    let hi = _mm_srli_si128::<8>(xorred);
    _mm_cvtsi128_si64(hi) as u32
  }
}

#[inline]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn finalize_lanes_width32_reflected(x: [Simd128; 8], keys: &[u64; 23]) -> u32 {
  let mut res = x[7];
  res = x[0].fold_16_reflected(Simd128::new(keys[10], keys[9]), res);
  res = x[1].fold_16_reflected(Simd128::new(keys[12], keys[11]), res);
  res = x[2].fold_16_reflected(Simd128::new(keys[14], keys[13]), res);
  res = x[3].fold_16_reflected(Simd128::new(keys[16], keys[15]), res);
  res = x[4].fold_16_reflected(Simd128::new(keys[18], keys[17]), res);
  res = x[5].fold_16_reflected(Simd128::new(keys[20], keys[19]), res);
  res = x[6].fold_16_reflected(Simd128::new(keys[2], keys[1]), res);

  res = res.fold_width32_reflected(keys[6], keys[5]);
  res.barrett_width32_reflected(keys[8], keys[7])
}

#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn update_simd_width32_reflected(
  state: u32,
  first: &[Simd128; 8],
  rest: &[[Simd128; 8]],
  keys: &[u64; 23],
) -> u32 {
  let mut x = *first;

  x[0] ^= Simd128::new(0, state as u64);

  let coeff_128b = Simd128::new(keys[4], keys[3]);
  for chunk in rest {
    x[0] = x[0].fold_16_reflected(coeff_128b, chunk[0]);
    x[1] = x[1].fold_16_reflected(coeff_128b, chunk[1]);
    x[2] = x[2].fold_16_reflected(coeff_128b, chunk[2]);
    x[3] = x[3].fold_16_reflected(coeff_128b, chunk[3]);
    x[4] = x[4].fold_16_reflected(coeff_128b, chunk[4]);
    x[5] = x[5].fold_16_reflected(coeff_128b, chunk[5]);
    x[6] = x[6].fold_16_reflected(coeff_128b, chunk[6]);
    x[7] = x[7].fold_16_reflected(coeff_128b, chunk[7]);
  }

  finalize_lanes_width32_reflected(x, keys)
}

// ─────────────────────────────────────────────────────────────────────────────
// PCLMULQDQ multi-stream (2/4/7/8-way, 128B blocks)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn fold_block_128_reflected(x: &mut [Simd128; 8], chunk: &[Simd128; 8], coeff: Simd128) {
  x[0] = x[0].fold_16_reflected(coeff, chunk[0]);
  x[1] = x[1].fold_16_reflected(coeff, chunk[1]);
  x[2] = x[2].fold_16_reflected(coeff, chunk[2]);
  x[3] = x[3].fold_16_reflected(coeff, chunk[3]);
  x[4] = x[4].fold_16_reflected(coeff, chunk[4]);
  x[5] = x[5].fold_16_reflected(coeff, chunk[5]);
  x[6] = x[6].fold_16_reflected(coeff, chunk[6]);
  x[7] = x[7].fold_16_reflected(coeff, chunk[7]);
}

#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn update_simd_width32_reflected_2way(
  state: u32,
  blocks: &[[Simd128; 8]],
  fold_256b: (u64, u64),
  keys: &[u64; 23],
) -> u32 {
  use crate::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  debug_assert!(!blocks.is_empty());

  if blocks.len() < 2 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_width32_reflected(state, first, rest, keys);
  }

  let coeff_256 = Simd128::new(fold_256b.0, fold_256b.1);
  let coeff_128 = Simd128::new(keys[4], keys[3]);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];

  s0[0] ^= Simd128::new(0, state as u64);

  // Double-unrolled main loop: process 4 blocks (512B) per iteration.
  const BLOCK_SIZE: usize = 128;
  const DOUBLE_GROUP: usize = 4; // 2 × 2-way = 4 blocks = 512B

  let mut i: usize = 2;
  let aligned = (blocks.len() / DOUBLE_GROUP) * DOUBLE_GROUP;

  while i.strict_add(DOUBLE_GROUP) <= aligned {
    let prefetch_idx = i.strict_add(LARGE_BLOCK_DISTANCE / BLOCK_SIZE);
    if prefetch_idx < blocks.len() {
      prefetch_read_l1(blocks[prefetch_idx].as_ptr().cast::<u8>());
    }

    // First iteration (blocks i, i+1)
    fold_block_128_reflected(&mut s0, &blocks[i], coeff_256);
    fold_block_128_reflected(&mut s1, &blocks[i.strict_add(1)], coeff_256);

    // Second iteration (blocks i+2, i+3)
    fold_block_128_reflected(&mut s0, &blocks[i.strict_add(2)], coeff_256);
    fold_block_128_reflected(&mut s1, &blocks[i.strict_add(3)], coeff_256);

    i = i.strict_add(DOUBLE_GROUP);
  }

  // Handle remaining pairs.
  let even = blocks.len() & !1usize;
  while i < even {
    fold_block_128_reflected(&mut s0, &blocks[i], coeff_256);
    fold_block_128_reflected(&mut s1, &blocks[i.strict_add(1)], coeff_256);
    i = i.strict_add(2);
  }

  // Merge: A·s0 ⊕ s1 (A = shift by 128B).
  let mut combined = s1;
  combined[0] = s0[0].fold_16_reflected(coeff_128, combined[0]);
  combined[1] = s0[1].fold_16_reflected(coeff_128, combined[1]);
  combined[2] = s0[2].fold_16_reflected(coeff_128, combined[2]);
  combined[3] = s0[3].fold_16_reflected(coeff_128, combined[3]);
  combined[4] = s0[4].fold_16_reflected(coeff_128, combined[4]);
  combined[5] = s0[5].fold_16_reflected(coeff_128, combined[5]);
  combined[6] = s0[6].fold_16_reflected(coeff_128, combined[6]);
  combined[7] = s0[7].fold_16_reflected(coeff_128, combined[7]);

  if even != blocks.len() {
    fold_block_128_reflected(&mut combined, &blocks[even], coeff_128);
  }

  finalize_lanes_width32_reflected(combined, keys)
}

#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn update_simd_width32_reflected_4way(
  state: u32,
  blocks: &[[Simd128; 8]],
  fold_512b: (u64, u64),
  combine: &[(u64, u64); 3],
  keys: &[u64; 23],
) -> u32 {
  use crate::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  debug_assert!(!blocks.is_empty());

  if blocks.len() < 4 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_width32_reflected(state, first, rest, keys);
  }

  let coeff_512 = Simd128::new(fold_512b.0, fold_512b.1);
  let coeff_128 = Simd128::new(keys[4], keys[3]);
  let c384 = Simd128::new(combine[0].0, combine[0].1);
  let c256 = Simd128::new(combine[1].0, combine[1].1);
  let c128 = Simd128::new(combine[2].0, combine[2].1);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];
  let mut s2 = blocks[2];
  let mut s3 = blocks[3];

  s0[0] ^= Simd128::new(0, state as u64);

  // Double-unrolled main loop: process 8 blocks (1KB) per iteration.
  const BLOCK_SIZE: usize = 128;
  const DOUBLE_GROUP: usize = 8; // 2 × 4-way = 8 blocks = 1KB

  let mut i: usize = 4;
  let aligned = (blocks.len() / DOUBLE_GROUP) * DOUBLE_GROUP;

  while i.strict_add(DOUBLE_GROUP) <= aligned {
    let prefetch_idx = i.strict_add(LARGE_BLOCK_DISTANCE / BLOCK_SIZE);
    if prefetch_idx < blocks.len() {
      prefetch_read_l1(blocks[prefetch_idx].as_ptr().cast::<u8>());
    }

    // First iteration (blocks i..i+3)
    fold_block_128_reflected(&mut s0, &blocks[i], coeff_512);
    fold_block_128_reflected(&mut s1, &blocks[i.strict_add(1)], coeff_512);
    fold_block_128_reflected(&mut s2, &blocks[i.strict_add(2)], coeff_512);
    fold_block_128_reflected(&mut s3, &blocks[i.strict_add(3)], coeff_512);

    // Second iteration (blocks i+4..i+7)
    fold_block_128_reflected(&mut s0, &blocks[i.strict_add(4)], coeff_512);
    fold_block_128_reflected(&mut s1, &blocks[i.strict_add(5)], coeff_512);
    fold_block_128_reflected(&mut s2, &blocks[i.strict_add(6)], coeff_512);
    fold_block_128_reflected(&mut s3, &blocks[i.strict_add(7)], coeff_512);

    i = i.strict_add(DOUBLE_GROUP);
  }

  // Handle remaining quads.
  let quad_aligned = (blocks.len() / 4) * 4;
  while i < quad_aligned {
    fold_block_128_reflected(&mut s0, &blocks[i], coeff_512);
    fold_block_128_reflected(&mut s1, &blocks[i.strict_add(1)], coeff_512);
    fold_block_128_reflected(&mut s2, &blocks[i.strict_add(2)], coeff_512);
    fold_block_128_reflected(&mut s3, &blocks[i.strict_add(3)], coeff_512);
    i = i.strict_add(4);
  }

  // Merge: A^3·s0 ⊕ A^2·s1 ⊕ A·s2 ⊕ s3.
  let mut acc = s3;
  acc[0] = s2[0].fold_16_reflected(c128, acc[0]);
  acc[1] = s2[1].fold_16_reflected(c128, acc[1]);
  acc[2] = s2[2].fold_16_reflected(c128, acc[2]);
  acc[3] = s2[3].fold_16_reflected(c128, acc[3]);
  acc[4] = s2[4].fold_16_reflected(c128, acc[4]);
  acc[5] = s2[5].fold_16_reflected(c128, acc[5]);
  acc[6] = s2[6].fold_16_reflected(c128, acc[6]);
  acc[7] = s2[7].fold_16_reflected(c128, acc[7]);

  acc[0] = s1[0].fold_16_reflected(c256, acc[0]);
  acc[1] = s1[1].fold_16_reflected(c256, acc[1]);
  acc[2] = s1[2].fold_16_reflected(c256, acc[2]);
  acc[3] = s1[3].fold_16_reflected(c256, acc[3]);
  acc[4] = s1[4].fold_16_reflected(c256, acc[4]);
  acc[5] = s1[5].fold_16_reflected(c256, acc[5]);
  acc[6] = s1[6].fold_16_reflected(c256, acc[6]);
  acc[7] = s1[7].fold_16_reflected(c256, acc[7]);

  acc[0] = s0[0].fold_16_reflected(c384, acc[0]);
  acc[1] = s0[1].fold_16_reflected(c384, acc[1]);
  acc[2] = s0[2].fold_16_reflected(c384, acc[2]);
  acc[3] = s0[3].fold_16_reflected(c384, acc[3]);
  acc[4] = s0[4].fold_16_reflected(c384, acc[4]);
  acc[5] = s0[5].fold_16_reflected(c384, acc[5]);
  acc[6] = s0[6].fold_16_reflected(c384, acc[6]);
  acc[7] = s0[7].fold_16_reflected(c384, acc[7]);

  for block in &blocks[aligned..] {
    fold_block_128_reflected(&mut acc, block, coeff_128);
  }

  finalize_lanes_width32_reflected(acc, keys)
}

#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn update_simd_width32_reflected_7way(
  state: u32,
  blocks: &[[Simd128; 8]],
  fold_896b: (u64, u64),
  combine: &[(u64, u64); 6],
  keys: &[u64; 23],
) -> u32 {
  use crate::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  debug_assert!(!blocks.is_empty());

  if blocks.len() < 7 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_width32_reflected(state, first, rest, keys);
  }

  let aligned = (blocks.len() / 7) * 7;

  let coeff_896 = Simd128::new(fold_896b.0, fold_896b.1);
  let coeff_128 = Simd128::new(keys[4], keys[3]);
  let c768 = Simd128::new(combine[0].0, combine[0].1);
  let c640 = Simd128::new(combine[1].0, combine[1].1);
  let c512 = Simd128::new(combine[2].0, combine[2].1);
  let c384 = Simd128::new(combine[3].0, combine[3].1);
  let c256 = Simd128::new(combine[4].0, combine[4].1);
  let c128 = Simd128::new(combine[5].0, combine[5].1);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];
  let mut s2 = blocks[2];
  let mut s3 = blocks[3];
  let mut s4 = blocks[4];
  let mut s5 = blocks[5];
  let mut s6 = blocks[6];

  s0[0] ^= Simd128::new(0, state as u64);

  const BLOCK_SIZE: usize = 128;

  let mut i: usize = 7;
  while i < aligned {
    let prefetch_idx = i.strict_add(LARGE_BLOCK_DISTANCE / BLOCK_SIZE);
    if prefetch_idx < blocks.len() {
      prefetch_read_l1(blocks[prefetch_idx].as_ptr().cast::<u8>());
    }

    fold_block_128_reflected(&mut s0, &blocks[i], coeff_896);
    fold_block_128_reflected(&mut s1, &blocks[i.strict_add(1)], coeff_896);
    fold_block_128_reflected(&mut s2, &blocks[i.strict_add(2)], coeff_896);
    fold_block_128_reflected(&mut s3, &blocks[i.strict_add(3)], coeff_896);
    fold_block_128_reflected(&mut s4, &blocks[i.strict_add(4)], coeff_896);
    fold_block_128_reflected(&mut s5, &blocks[i.strict_add(5)], coeff_896);
    fold_block_128_reflected(&mut s6, &blocks[i.strict_add(6)], coeff_896);
    i = i.strict_add(7);
  }

  // Merge: A^6·s0 ⊕ A^5·s1 ⊕ A^4·s2 ⊕ A^3·s3 ⊕ A^2·s4 ⊕ A·s5 ⊕ s6.
  let mut acc = s6;
  acc[0] = s5[0].fold_16_reflected(c128, acc[0]);
  acc[1] = s5[1].fold_16_reflected(c128, acc[1]);
  acc[2] = s5[2].fold_16_reflected(c128, acc[2]);
  acc[3] = s5[3].fold_16_reflected(c128, acc[3]);
  acc[4] = s5[4].fold_16_reflected(c128, acc[4]);
  acc[5] = s5[5].fold_16_reflected(c128, acc[5]);
  acc[6] = s5[6].fold_16_reflected(c128, acc[6]);
  acc[7] = s5[7].fold_16_reflected(c128, acc[7]);

  acc[0] = s4[0].fold_16_reflected(c256, acc[0]);
  acc[1] = s4[1].fold_16_reflected(c256, acc[1]);
  acc[2] = s4[2].fold_16_reflected(c256, acc[2]);
  acc[3] = s4[3].fold_16_reflected(c256, acc[3]);
  acc[4] = s4[4].fold_16_reflected(c256, acc[4]);
  acc[5] = s4[5].fold_16_reflected(c256, acc[5]);
  acc[6] = s4[6].fold_16_reflected(c256, acc[6]);
  acc[7] = s4[7].fold_16_reflected(c256, acc[7]);

  acc[0] = s3[0].fold_16_reflected(c384, acc[0]);
  acc[1] = s3[1].fold_16_reflected(c384, acc[1]);
  acc[2] = s3[2].fold_16_reflected(c384, acc[2]);
  acc[3] = s3[3].fold_16_reflected(c384, acc[3]);
  acc[4] = s3[4].fold_16_reflected(c384, acc[4]);
  acc[5] = s3[5].fold_16_reflected(c384, acc[5]);
  acc[6] = s3[6].fold_16_reflected(c384, acc[6]);
  acc[7] = s3[7].fold_16_reflected(c384, acc[7]);

  acc[0] = s2[0].fold_16_reflected(c512, acc[0]);
  acc[1] = s2[1].fold_16_reflected(c512, acc[1]);
  acc[2] = s2[2].fold_16_reflected(c512, acc[2]);
  acc[3] = s2[3].fold_16_reflected(c512, acc[3]);
  acc[4] = s2[4].fold_16_reflected(c512, acc[4]);
  acc[5] = s2[5].fold_16_reflected(c512, acc[5]);
  acc[6] = s2[6].fold_16_reflected(c512, acc[6]);
  acc[7] = s2[7].fold_16_reflected(c512, acc[7]);

  acc[0] = s1[0].fold_16_reflected(c640, acc[0]);
  acc[1] = s1[1].fold_16_reflected(c640, acc[1]);
  acc[2] = s1[2].fold_16_reflected(c640, acc[2]);
  acc[3] = s1[3].fold_16_reflected(c640, acc[3]);
  acc[4] = s1[4].fold_16_reflected(c640, acc[4]);
  acc[5] = s1[5].fold_16_reflected(c640, acc[5]);
  acc[6] = s1[6].fold_16_reflected(c640, acc[6]);
  acc[7] = s1[7].fold_16_reflected(c640, acc[7]);

  acc[0] = s0[0].fold_16_reflected(c768, acc[0]);
  acc[1] = s0[1].fold_16_reflected(c768, acc[1]);
  acc[2] = s0[2].fold_16_reflected(c768, acc[2]);
  acc[3] = s0[3].fold_16_reflected(c768, acc[3]);
  acc[4] = s0[4].fold_16_reflected(c768, acc[4]);
  acc[5] = s0[5].fold_16_reflected(c768, acc[5]);
  acc[6] = s0[6].fold_16_reflected(c768, acc[6]);
  acc[7] = s0[7].fold_16_reflected(c768, acc[7]);

  for block in &blocks[aligned..] {
    fold_block_128_reflected(&mut acc, block, coeff_128);
  }

  finalize_lanes_width32_reflected(acc, keys)
}

#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn update_simd_width32_reflected_8way(
  state: u32,
  blocks: &[[Simd128; 8]],
  fold_1024b: (u64, u64),
  combine: &[(u64, u64); 7],
  keys: &[u64; 23],
) -> u32 {
  use crate::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  debug_assert!(!blocks.is_empty());

  if blocks.len() < 8 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_width32_reflected(state, first, rest, keys);
  }

  let aligned = (blocks.len() / 8) * 8;

  let coeff_1024 = Simd128::new(fold_1024b.0, fold_1024b.1);
  let coeff_128 = Simd128::new(keys[4], keys[3]);
  let c896 = Simd128::new(combine[0].0, combine[0].1);
  let c768 = Simd128::new(combine[1].0, combine[1].1);
  let c640 = Simd128::new(combine[2].0, combine[2].1);
  let c512 = Simd128::new(combine[3].0, combine[3].1);
  let c384 = Simd128::new(combine[4].0, combine[4].1);
  let c256 = Simd128::new(combine[5].0, combine[5].1);
  let c128 = Simd128::new(combine[6].0, combine[6].1);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];
  let mut s2 = blocks[2];
  let mut s3 = blocks[3];
  let mut s4 = blocks[4];
  let mut s5 = blocks[5];
  let mut s6 = blocks[6];
  let mut s7 = blocks[7];

  s0[0] ^= Simd128::new(0, state as u64);

  const BLOCK_SIZE: usize = 128;

  let mut i: usize = 8;
  while i < aligned {
    let prefetch_idx = i.strict_add(LARGE_BLOCK_DISTANCE / BLOCK_SIZE);
    if prefetch_idx < blocks.len() {
      prefetch_read_l1(blocks[prefetch_idx].as_ptr().cast::<u8>());
    }

    fold_block_128_reflected(&mut s0, &blocks[i], coeff_1024);
    fold_block_128_reflected(&mut s1, &blocks[i.strict_add(1)], coeff_1024);
    fold_block_128_reflected(&mut s2, &blocks[i.strict_add(2)], coeff_1024);
    fold_block_128_reflected(&mut s3, &blocks[i.strict_add(3)], coeff_1024);
    fold_block_128_reflected(&mut s4, &blocks[i.strict_add(4)], coeff_1024);
    fold_block_128_reflected(&mut s5, &blocks[i.strict_add(5)], coeff_1024);
    fold_block_128_reflected(&mut s6, &blocks[i.strict_add(6)], coeff_1024);
    fold_block_128_reflected(&mut s7, &blocks[i.strict_add(7)], coeff_1024);
    i = i.strict_add(8);
  }

  // Merge: A^7·s0 ⊕ A^6·s1 ⊕ A^5·s2 ⊕ A^4·s3 ⊕ A^3·s4 ⊕ A^2·s5 ⊕ A·s6 ⊕ s7.
  let mut acc = s7;
  acc[0] = s6[0].fold_16_reflected(c128, acc[0]);
  acc[1] = s6[1].fold_16_reflected(c128, acc[1]);
  acc[2] = s6[2].fold_16_reflected(c128, acc[2]);
  acc[3] = s6[3].fold_16_reflected(c128, acc[3]);
  acc[4] = s6[4].fold_16_reflected(c128, acc[4]);
  acc[5] = s6[5].fold_16_reflected(c128, acc[5]);
  acc[6] = s6[6].fold_16_reflected(c128, acc[6]);
  acc[7] = s6[7].fold_16_reflected(c128, acc[7]);

  acc[0] = s5[0].fold_16_reflected(c256, acc[0]);
  acc[1] = s5[1].fold_16_reflected(c256, acc[1]);
  acc[2] = s5[2].fold_16_reflected(c256, acc[2]);
  acc[3] = s5[3].fold_16_reflected(c256, acc[3]);
  acc[4] = s5[4].fold_16_reflected(c256, acc[4]);
  acc[5] = s5[5].fold_16_reflected(c256, acc[5]);
  acc[6] = s5[6].fold_16_reflected(c256, acc[6]);
  acc[7] = s5[7].fold_16_reflected(c256, acc[7]);

  acc[0] = s4[0].fold_16_reflected(c384, acc[0]);
  acc[1] = s4[1].fold_16_reflected(c384, acc[1]);
  acc[2] = s4[2].fold_16_reflected(c384, acc[2]);
  acc[3] = s4[3].fold_16_reflected(c384, acc[3]);
  acc[4] = s4[4].fold_16_reflected(c384, acc[4]);
  acc[5] = s4[5].fold_16_reflected(c384, acc[5]);
  acc[6] = s4[6].fold_16_reflected(c384, acc[6]);
  acc[7] = s4[7].fold_16_reflected(c384, acc[7]);

  acc[0] = s3[0].fold_16_reflected(c512, acc[0]);
  acc[1] = s3[1].fold_16_reflected(c512, acc[1]);
  acc[2] = s3[2].fold_16_reflected(c512, acc[2]);
  acc[3] = s3[3].fold_16_reflected(c512, acc[3]);
  acc[4] = s3[4].fold_16_reflected(c512, acc[4]);
  acc[5] = s3[5].fold_16_reflected(c512, acc[5]);
  acc[6] = s3[6].fold_16_reflected(c512, acc[6]);
  acc[7] = s3[7].fold_16_reflected(c512, acc[7]);

  acc[0] = s2[0].fold_16_reflected(c640, acc[0]);
  acc[1] = s2[1].fold_16_reflected(c640, acc[1]);
  acc[2] = s2[2].fold_16_reflected(c640, acc[2]);
  acc[3] = s2[3].fold_16_reflected(c640, acc[3]);
  acc[4] = s2[4].fold_16_reflected(c640, acc[4]);
  acc[5] = s2[5].fold_16_reflected(c640, acc[5]);
  acc[6] = s2[6].fold_16_reflected(c640, acc[6]);
  acc[7] = s2[7].fold_16_reflected(c640, acc[7]);

  acc[0] = s1[0].fold_16_reflected(c768, acc[0]);
  acc[1] = s1[1].fold_16_reflected(c768, acc[1]);
  acc[2] = s1[2].fold_16_reflected(c768, acc[2]);
  acc[3] = s1[3].fold_16_reflected(c768, acc[3]);
  acc[4] = s1[4].fold_16_reflected(c768, acc[4]);
  acc[5] = s1[5].fold_16_reflected(c768, acc[5]);
  acc[6] = s1[6].fold_16_reflected(c768, acc[6]);
  acc[7] = s1[7].fold_16_reflected(c768, acc[7]);

  acc[0] = s0[0].fold_16_reflected(c896, acc[0]);
  acc[1] = s0[1].fold_16_reflected(c896, acc[1]);
  acc[2] = s0[2].fold_16_reflected(c896, acc[2]);
  acc[3] = s0[3].fold_16_reflected(c896, acc[3]);
  acc[4] = s0[4].fold_16_reflected(c896, acc[4]);
  acc[5] = s0[5].fold_16_reflected(c896, acc[5]);
  acc[6] = s0[6].fold_16_reflected(c896, acc[6]);
  acc[7] = s0[7].fold_16_reflected(c896, acc[7]);

  for block in &blocks[aligned..] {
    fold_block_128_reflected(&mut acc, block, coeff_128);
  }

  finalize_lanes_width32_reflected(acc, keys)
}

#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn crc16_width32_pclmul_stream(
  mut state: u16,
  data: &[u8],
  keys: &[u64; 23],
  stream: &Width32StreamConstants,
  streams: u8,
  portable: fn(u16, &[u8]) -> u16,
) -> u16 {
  let (left, middle, right) = data.align_to::<[Simd128; 8]>();
  let Some((first, rest)) = middle.split_first() else {
    return crc16_width32_pclmul_small(state, data, keys, portable);
  };

  state = portable(state, left);
  let state32 = match streams {
    8 => update_simd_width32_reflected_8way(state as u32, middle, stream.fold_1024b, &stream.combine_8way, keys),
    7 => update_simd_width32_reflected_7way(state as u32, middle, stream.fold_896b, &stream.combine_7way, keys),
    4 => update_simd_width32_reflected_4way(state as u32, middle, stream.fold_512b, &stream.combine_4way, keys),
    2 => update_simd_width32_reflected_2way(state as u32, middle, stream.fold_256b, keys),
    _ => update_simd_width32_reflected(state as u32, first, rest, keys),
  };
  state = state32 as u16;
  portable(state, right)
}

#[inline]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn crc16_width32_pclmul_small(
  mut state: u16,
  data: &[u8],
  keys: &[u64; 23],
  portable: fn(u16, &[u8]) -> u16,
) -> u16 {
  let mut buf = data.as_ptr();
  let mut len = data.len();

  if len < 16 {
    return portable(state, data);
  }

  let coeff_16b = Simd128::new(keys[2], keys[1]);

  let mut x0 = Simd128(_mm_loadu_si128(buf as *const __m128i));
  x0 ^= Simd128::new(0, state as u64);
  buf = buf.add(16);
  len = len.strict_sub(16);

  while len >= 16 {
    let chunk = Simd128(_mm_loadu_si128(buf as *const __m128i));
    x0 = x0.fold_16_reflected(coeff_16b, chunk);
    buf = buf.add(16);
    len = len.strict_sub(16);
  }

  let x0 = x0.fold_width32_reflected(keys[6], keys[5]);
  state = x0.barrett_width32_reflected(keys[8], keys[7]) as u16;

  let tail = core::slice::from_raw_parts(buf, len);
  portable(state, tail)
}

#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn crc16_width32_pclmul(mut state: u16, data: &[u8], keys: &[u64; 23], portable: fn(u16, &[u8]) -> u16) -> u16 {
  let (left, middle, right) = data.align_to::<[Simd128; 8]>();
  let Some((first, rest)) = middle.split_first() else {
    return crc16_width32_pclmul_small(state, data, keys, portable);
  };

  state = portable(state, left);
  let state32 = update_simd_width32_reflected(state as u32, first, rest, keys);
  state = state32 as u16;
  portable(state, right)
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX-512 VPCLMULQDQ Tier
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq")]
unsafe fn clmul10_vpclmul(a: __m512i, b: __m512i) -> __m512i {
  _mm512_clmulepi64_epi128(a, b, 0x10)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq")]
unsafe fn clmul01_vpclmul(a: __m512i, b: __m512i) -> __m512i {
  _mm512_clmulepi64_epi128(a, b, 0x01)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq")]
unsafe fn fold_16_reflected_vpclmul(state: __m512i, coeff: __m512i, data: __m512i) -> __m512i {
  _mm512_ternarylogic_epi64(clmul10_vpclmul(state, coeff), clmul01_vpclmul(state, coeff), data, 0x96)
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn broadcast_coeff_128b(high: u64, low: u64) -> __m512i {
  _mm512_set_epi64(
    high as i64,
    low as i64,
    high as i64,
    low as i64,
    high as i64,
    low as i64,
    high as i64,
    low as i64,
  )
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn state_mask_lane0(state: u32) -> __m512i {
  _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, state as i64)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn update_simd_width32_reflected_vpclmul(
  state: u32,
  first: &[Simd128; 8],
  rest: &[[Simd128; 8]],
  keys: &[u64; 23],
) -> u32 {
  let ptr = first.as_ptr() as *const u8;
  let mut x0 = _mm512_loadu_si512(ptr as *const __m512i);
  let mut x1 = _mm512_loadu_si512(ptr.add(64) as *const __m512i);

  x0 = _mm512_xor_si512(x0, state_mask_lane0(state));

  let coeff_128b = broadcast_coeff_128b(keys[4], keys[3]);
  for chunk in rest {
    let ptr = chunk.as_ptr() as *const u8;
    let y0 = _mm512_loadu_si512(ptr as *const __m512i);
    let y1 = _mm512_loadu_si512(ptr.add(64) as *const __m512i);
    x0 = fold_16_reflected_vpclmul(x0, coeff_128b, y0);
    x1 = fold_16_reflected_vpclmul(x1, coeff_128b, y1);
  }

  let mut lanes0 = [Simd128(_mm_setzero_si128()); 4];
  let mut lanes1 = [Simd128(_mm_setzero_si128()); 4];
  _mm512_storeu_si512(lanes0.as_mut_ptr() as *mut __m512i, x0);
  _mm512_storeu_si512(lanes1.as_mut_ptr() as *mut __m512i, x1);

  let x = [
    lanes0[0], lanes0[1], lanes0[2], lanes0[3], lanes1[0], lanes1[1], lanes1[2], lanes1[3],
  ];

  finalize_lanes_width32_reflected(x, keys)
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn vpclmul_coeff(pair: (u64, u64)) -> __m512i {
  broadcast_coeff_128b(pair.0, pair.1)
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_128b_block(block: &[Simd128; 8]) -> (__m512i, __m512i) {
  let ptr = block.as_ptr() as *const u8;
  (
    _mm512_loadu_si512(ptr as *const __m512i),
    _mm512_loadu_si512(ptr.add(64) as *const __m512i),
  )
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn finalize_vpclmul_state(x0: __m512i, x1: __m512i, keys: &[u64; 23]) -> u32 {
  let lanes0 = [
    Simd128(_mm512_extracti32x4_epi32::<0>(x0)),
    Simd128(_mm512_extracti32x4_epi32::<1>(x0)),
    Simd128(_mm512_extracti32x4_epi32::<2>(x0)),
    Simd128(_mm512_extracti32x4_epi32::<3>(x0)),
  ];
  let lanes1 = [
    Simd128(_mm512_extracti32x4_epi32::<0>(x1)),
    Simd128(_mm512_extracti32x4_epi32::<1>(x1)),
    Simd128(_mm512_extracti32x4_epi32::<2>(x1)),
    Simd128(_mm512_extracti32x4_epi32::<3>(x1)),
  ];

  let x = [
    lanes0[0], lanes0[1], lanes0[2], lanes0[3], lanes1[0], lanes1[1], lanes1[2], lanes1[3],
  ];
  finalize_lanes_width32_reflected(x, keys)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn update_simd_width32_reflected_vpclmul_2way(
  state: u32,
  blocks: &[[Simd128; 8]],
  fold_256b: (u64, u64),
  keys: &[u64; 23],
) -> u32 {
  use crate::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  debug_assert!(!blocks.is_empty());

  if blocks.len() < 2 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_width32_reflected_vpclmul(state, first, rest, keys);
  }

  let (mut x0_0, mut x1_0) = load_128b_block(&blocks[0]);
  let (mut x0_1, mut x1_1) = load_128b_block(&blocks[1]);

  x0_0 = _mm512_xor_si512(x0_0, state_mask_lane0(state));

  let coeff_256 = vpclmul_coeff(fold_256b);
  let coeff_128 = vpclmul_coeff((keys[4], keys[3]));

  // Double-unrolled main loop: process 4 blocks (512B) per iteration.
  const BLOCK_SIZE: usize = 128;
  const DOUBLE_GROUP: usize = 4; // 2 × 2-way = 4 blocks = 512B

  let mut i: usize = 2;
  let aligned = (blocks.len() / DOUBLE_GROUP) * DOUBLE_GROUP;

  while i.strict_add(DOUBLE_GROUP) <= aligned {
    let prefetch_idx = i.strict_add(LARGE_BLOCK_DISTANCE / BLOCK_SIZE);
    if prefetch_idx < blocks.len() {
      prefetch_read_l1(blocks[prefetch_idx].as_ptr().cast::<u8>());
    }

    // First iteration (blocks i, i+1)
    let (y0, y1) = load_128b_block(&blocks[i]);
    x0_0 = fold_16_reflected_vpclmul(x0_0, coeff_256, y0);
    x1_0 = fold_16_reflected_vpclmul(x1_0, coeff_256, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(1)]);
    x0_1 = fold_16_reflected_vpclmul(x0_1, coeff_256, y0);
    x1_1 = fold_16_reflected_vpclmul(x1_1, coeff_256, y1);

    // Second iteration (blocks i+2, i+3)
    let (y0, y1) = load_128b_block(&blocks[i.strict_add(2)]);
    x0_0 = fold_16_reflected_vpclmul(x0_0, coeff_256, y0);
    x1_0 = fold_16_reflected_vpclmul(x1_0, coeff_256, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(3)]);
    x0_1 = fold_16_reflected_vpclmul(x0_1, coeff_256, y0);
    x1_1 = fold_16_reflected_vpclmul(x1_1, coeff_256, y1);

    i = i.strict_add(DOUBLE_GROUP);
  }

  // Handle remaining pairs.
  let even = blocks.len() & !1usize;
  while i < even {
    let (y0, y1) = load_128b_block(&blocks[i]);
    x0_0 = fold_16_reflected_vpclmul(x0_0, coeff_256, y0);
    x1_0 = fold_16_reflected_vpclmul(x1_0, coeff_256, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(1)]);
    x0_1 = fold_16_reflected_vpclmul(x0_1, coeff_256, y0);
    x1_1 = fold_16_reflected_vpclmul(x1_1, coeff_256, y1);

    i = i.strict_add(2);
  }

  let mut x0 = fold_16_reflected_vpclmul(x0_0, coeff_128, x0_1);
  let mut x1 = fold_16_reflected_vpclmul(x1_0, coeff_128, x1_1);

  if even != blocks.len() {
    let (y0, y1) = load_128b_block(&blocks[even]);
    x0 = fold_16_reflected_vpclmul(x0, coeff_128, y0);
    x1 = fold_16_reflected_vpclmul(x1, coeff_128, y1);
  }

  finalize_vpclmul_state(x0, x1, keys)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn update_simd_width32_reflected_vpclmul_4way(
  state: u32,
  blocks: &[[Simd128; 8]],
  fold_512b: (u64, u64),
  combine: &[(u64, u64); 3],
  keys: &[u64; 23],
) -> u32 {
  use crate::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  debug_assert!(!blocks.is_empty());

  if blocks.len() < 4 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_width32_reflected_vpclmul(state, first, rest, keys);
  }

  let (mut x0_0, mut x1_0) = load_128b_block(&blocks[0]);
  let (mut x0_1, mut x1_1) = load_128b_block(&blocks[1]);
  let (mut x0_2, mut x1_2) = load_128b_block(&blocks[2]);
  let (mut x0_3, mut x1_3) = load_128b_block(&blocks[3]);

  x0_0 = _mm512_xor_si512(x0_0, state_mask_lane0(state));

  let coeff_512 = vpclmul_coeff(fold_512b);
  let coeff_128 = vpclmul_coeff((keys[4], keys[3]));
  let c384 = vpclmul_coeff(combine[0]);
  let c256 = vpclmul_coeff(combine[1]);
  let c128 = vpclmul_coeff(combine[2]);

  // Double-unrolled main loop: process 8 blocks (1KB) per iteration.
  const BLOCK_SIZE: usize = 128;
  const DOUBLE_GROUP: usize = 8; // 2 × 4-way = 8 blocks = 1KB

  let mut i: usize = 4;
  let aligned = (blocks.len() / DOUBLE_GROUP) * DOUBLE_GROUP;

  while i.strict_add(DOUBLE_GROUP) <= aligned {
    let prefetch_idx = i.strict_add(LARGE_BLOCK_DISTANCE / BLOCK_SIZE);
    if prefetch_idx < blocks.len() {
      prefetch_read_l1(blocks[prefetch_idx].as_ptr().cast::<u8>());
    }

    // First iteration (blocks i..i+3)
    let (y0, y1) = load_128b_block(&blocks[i]);
    x0_0 = fold_16_reflected_vpclmul(x0_0, coeff_512, y0);
    x1_0 = fold_16_reflected_vpclmul(x1_0, coeff_512, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(1)]);
    x0_1 = fold_16_reflected_vpclmul(x0_1, coeff_512, y0);
    x1_1 = fold_16_reflected_vpclmul(x1_1, coeff_512, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(2)]);
    x0_2 = fold_16_reflected_vpclmul(x0_2, coeff_512, y0);
    x1_2 = fold_16_reflected_vpclmul(x1_2, coeff_512, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(3)]);
    x0_3 = fold_16_reflected_vpclmul(x0_3, coeff_512, y0);
    x1_3 = fold_16_reflected_vpclmul(x1_3, coeff_512, y1);

    // Second iteration (blocks i+4..i+7)
    let (y0, y1) = load_128b_block(&blocks[i.strict_add(4)]);
    x0_0 = fold_16_reflected_vpclmul(x0_0, coeff_512, y0);
    x1_0 = fold_16_reflected_vpclmul(x1_0, coeff_512, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(5)]);
    x0_1 = fold_16_reflected_vpclmul(x0_1, coeff_512, y0);
    x1_1 = fold_16_reflected_vpclmul(x1_1, coeff_512, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(6)]);
    x0_2 = fold_16_reflected_vpclmul(x0_2, coeff_512, y0);
    x1_2 = fold_16_reflected_vpclmul(x1_2, coeff_512, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(7)]);
    x0_3 = fold_16_reflected_vpclmul(x0_3, coeff_512, y0);
    x1_3 = fold_16_reflected_vpclmul(x1_3, coeff_512, y1);

    i = i.strict_add(DOUBLE_GROUP);
  }

  // Handle remaining quads.
  let quad_aligned = (blocks.len() / 4) * 4;
  while i < quad_aligned {
    let (y0, y1) = load_128b_block(&blocks[i]);
    x0_0 = fold_16_reflected_vpclmul(x0_0, coeff_512, y0);
    x1_0 = fold_16_reflected_vpclmul(x1_0, coeff_512, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(1)]);
    x0_1 = fold_16_reflected_vpclmul(x0_1, coeff_512, y0);
    x1_1 = fold_16_reflected_vpclmul(x1_1, coeff_512, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(2)]);
    x0_2 = fold_16_reflected_vpclmul(x0_2, coeff_512, y0);
    x1_2 = fold_16_reflected_vpclmul(x1_2, coeff_512, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(3)]);
    x0_3 = fold_16_reflected_vpclmul(x0_3, coeff_512, y0);
    x1_3 = fold_16_reflected_vpclmul(x1_3, coeff_512, y1);

    i = i.strict_add(4);
  }

  let mut x0 = fold_16_reflected_vpclmul(x0_2, c128, x0_3);
  let mut x1 = fold_16_reflected_vpclmul(x1_2, c128, x1_3);
  x0 = fold_16_reflected_vpclmul(x0_1, c256, x0);
  x1 = fold_16_reflected_vpclmul(x1_1, c256, x1);
  x0 = fold_16_reflected_vpclmul(x0_0, c384, x0);
  x1 = fold_16_reflected_vpclmul(x1_0, c384, x1);

  for block in &blocks[quad_aligned..] {
    let (y0, y1) = load_128b_block(block);
    x0 = fold_16_reflected_vpclmul(x0, coeff_128, y0);
    x1 = fold_16_reflected_vpclmul(x1, coeff_128, y1);
  }

  finalize_vpclmul_state(x0, x1, keys)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn update_simd_width32_reflected_vpclmul_7way(
  state: u32,
  blocks: &[[Simd128; 8]],
  fold_896b: (u64, u64),
  combine: &[(u64, u64); 6],
  keys: &[u64; 23],
) -> u32 {
  use crate::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  debug_assert!(!blocks.is_empty());

  if blocks.len() < 7 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_width32_reflected_vpclmul(state, first, rest, keys);
  }

  let aligned = (blocks.len() / 7) * 7;

  let (mut x0_0, mut x1_0) = load_128b_block(&blocks[0]);
  let (mut x0_1, mut x1_1) = load_128b_block(&blocks[1]);
  let (mut x0_2, mut x1_2) = load_128b_block(&blocks[2]);
  let (mut x0_3, mut x1_3) = load_128b_block(&blocks[3]);
  let (mut x0_4, mut x1_4) = load_128b_block(&blocks[4]);
  let (mut x0_5, mut x1_5) = load_128b_block(&blocks[5]);
  let (mut x0_6, mut x1_6) = load_128b_block(&blocks[6]);

  x0_0 = _mm512_xor_si512(x0_0, state_mask_lane0(state));

  let coeff_896 = vpclmul_coeff(fold_896b);
  let coeff_128 = vpclmul_coeff((keys[4], keys[3]));
  let c768 = vpclmul_coeff(combine[0]);
  let c640 = vpclmul_coeff(combine[1]);
  let c512 = vpclmul_coeff(combine[2]);
  let c384 = vpclmul_coeff(combine[3]);
  let c256 = vpclmul_coeff(combine[4]);
  let c128 = vpclmul_coeff(combine[5]);

  const BLOCK_SIZE: usize = 128;

  let mut i: usize = 7;
  while i < aligned {
    let prefetch_idx = i.strict_add(LARGE_BLOCK_DISTANCE / BLOCK_SIZE);
    if prefetch_idx < blocks.len() {
      prefetch_read_l1(blocks[prefetch_idx].as_ptr().cast::<u8>());
    }

    let (y0, y1) = load_128b_block(&blocks[i]);
    x0_0 = fold_16_reflected_vpclmul(x0_0, coeff_896, y0);
    x1_0 = fold_16_reflected_vpclmul(x1_0, coeff_896, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(1)]);
    x0_1 = fold_16_reflected_vpclmul(x0_1, coeff_896, y0);
    x1_1 = fold_16_reflected_vpclmul(x1_1, coeff_896, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(2)]);
    x0_2 = fold_16_reflected_vpclmul(x0_2, coeff_896, y0);
    x1_2 = fold_16_reflected_vpclmul(x1_2, coeff_896, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(3)]);
    x0_3 = fold_16_reflected_vpclmul(x0_3, coeff_896, y0);
    x1_3 = fold_16_reflected_vpclmul(x1_3, coeff_896, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(4)]);
    x0_4 = fold_16_reflected_vpclmul(x0_4, coeff_896, y0);
    x1_4 = fold_16_reflected_vpclmul(x1_4, coeff_896, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(5)]);
    x0_5 = fold_16_reflected_vpclmul(x0_5, coeff_896, y0);
    x1_5 = fold_16_reflected_vpclmul(x1_5, coeff_896, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(6)]);
    x0_6 = fold_16_reflected_vpclmul(x0_6, coeff_896, y0);
    x1_6 = fold_16_reflected_vpclmul(x1_6, coeff_896, y1);

    i = i.strict_add(7);
  }

  let mut x0 = fold_16_reflected_vpclmul(x0_5, c128, x0_6);
  let mut x1 = fold_16_reflected_vpclmul(x1_5, c128, x1_6);
  x0 = fold_16_reflected_vpclmul(x0_4, c256, x0);
  x1 = fold_16_reflected_vpclmul(x1_4, c256, x1);
  x0 = fold_16_reflected_vpclmul(x0_3, c384, x0);
  x1 = fold_16_reflected_vpclmul(x1_3, c384, x1);
  x0 = fold_16_reflected_vpclmul(x0_2, c512, x0);
  x1 = fold_16_reflected_vpclmul(x1_2, c512, x1);
  x0 = fold_16_reflected_vpclmul(x0_1, c640, x0);
  x1 = fold_16_reflected_vpclmul(x1_1, c640, x1);
  x0 = fold_16_reflected_vpclmul(x0_0, c768, x0);
  x1 = fold_16_reflected_vpclmul(x1_0, c768, x1);

  for block in &blocks[aligned..] {
    let (y0, y1) = load_128b_block(block);
    x0 = fold_16_reflected_vpclmul(x0, coeff_128, y0);
    x1 = fold_16_reflected_vpclmul(x1, coeff_128, y1);
  }

  finalize_vpclmul_state(x0, x1, keys)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn update_simd_width32_reflected_vpclmul_8way(
  state: u32,
  blocks: &[[Simd128; 8]],
  fold_1024b: (u64, u64),
  combine: &[(u64, u64); 7],
  keys: &[u64; 23],
) -> u32 {
  use crate::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  debug_assert!(!blocks.is_empty());

  if blocks.len() < 8 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_width32_reflected_vpclmul(state, first, rest, keys);
  }

  let aligned = (blocks.len() / 8) * 8;

  let (mut x0_0, mut x1_0) = load_128b_block(&blocks[0]);
  let (mut x0_1, mut x1_1) = load_128b_block(&blocks[1]);
  let (mut x0_2, mut x1_2) = load_128b_block(&blocks[2]);
  let (mut x0_3, mut x1_3) = load_128b_block(&blocks[3]);
  let (mut x0_4, mut x1_4) = load_128b_block(&blocks[4]);
  let (mut x0_5, mut x1_5) = load_128b_block(&blocks[5]);
  let (mut x0_6, mut x1_6) = load_128b_block(&blocks[6]);
  let (mut x0_7, mut x1_7) = load_128b_block(&blocks[7]);

  x0_0 = _mm512_xor_si512(x0_0, state_mask_lane0(state));

  let coeff_1024 = vpclmul_coeff(fold_1024b);
  let coeff_128 = vpclmul_coeff((keys[4], keys[3]));
  let c896 = vpclmul_coeff(combine[0]);
  let c768 = vpclmul_coeff(combine[1]);
  let c640 = vpclmul_coeff(combine[2]);
  let c512 = vpclmul_coeff(combine[3]);
  let c384 = vpclmul_coeff(combine[4]);
  let c256 = vpclmul_coeff(combine[5]);
  let c128 = vpclmul_coeff(combine[6]);

  const BLOCK_SIZE: usize = 128;

  let mut i: usize = 8;
  while i < aligned {
    let prefetch_idx = i.strict_add(LARGE_BLOCK_DISTANCE / BLOCK_SIZE);
    if prefetch_idx < blocks.len() {
      prefetch_read_l1(blocks[prefetch_idx].as_ptr().cast::<u8>());
    }

    let (y0, y1) = load_128b_block(&blocks[i]);
    x0_0 = fold_16_reflected_vpclmul(x0_0, coeff_1024, y0);
    x1_0 = fold_16_reflected_vpclmul(x1_0, coeff_1024, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(1)]);
    x0_1 = fold_16_reflected_vpclmul(x0_1, coeff_1024, y0);
    x1_1 = fold_16_reflected_vpclmul(x1_1, coeff_1024, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(2)]);
    x0_2 = fold_16_reflected_vpclmul(x0_2, coeff_1024, y0);
    x1_2 = fold_16_reflected_vpclmul(x1_2, coeff_1024, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(3)]);
    x0_3 = fold_16_reflected_vpclmul(x0_3, coeff_1024, y0);
    x1_3 = fold_16_reflected_vpclmul(x1_3, coeff_1024, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(4)]);
    x0_4 = fold_16_reflected_vpclmul(x0_4, coeff_1024, y0);
    x1_4 = fold_16_reflected_vpclmul(x1_4, coeff_1024, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(5)]);
    x0_5 = fold_16_reflected_vpclmul(x0_5, coeff_1024, y0);
    x1_5 = fold_16_reflected_vpclmul(x1_5, coeff_1024, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(6)]);
    x0_6 = fold_16_reflected_vpclmul(x0_6, coeff_1024, y0);
    x1_6 = fold_16_reflected_vpclmul(x1_6, coeff_1024, y1);

    let (y0, y1) = load_128b_block(&blocks[i.strict_add(7)]);
    x0_7 = fold_16_reflected_vpclmul(x0_7, coeff_1024, y0);
    x1_7 = fold_16_reflected_vpclmul(x1_7, coeff_1024, y1);

    i = i.strict_add(8);
  }

  let mut x0 = fold_16_reflected_vpclmul(x0_6, c128, x0_7);
  let mut x1 = fold_16_reflected_vpclmul(x1_6, c128, x1_7);
  x0 = fold_16_reflected_vpclmul(x0_5, c256, x0);
  x1 = fold_16_reflected_vpclmul(x1_5, c256, x1);
  x0 = fold_16_reflected_vpclmul(x0_4, c384, x0);
  x1 = fold_16_reflected_vpclmul(x1_4, c384, x1);
  x0 = fold_16_reflected_vpclmul(x0_3, c512, x0);
  x1 = fold_16_reflected_vpclmul(x1_3, c512, x1);
  x0 = fold_16_reflected_vpclmul(x0_2, c640, x0);
  x1 = fold_16_reflected_vpclmul(x1_2, c640, x1);
  x0 = fold_16_reflected_vpclmul(x0_1, c768, x0);
  x1 = fold_16_reflected_vpclmul(x1_1, c768, x1);
  x0 = fold_16_reflected_vpclmul(x0_0, c896, x0);
  x1 = fold_16_reflected_vpclmul(x1_0, c896, x1);

  for block in &blocks[aligned..] {
    let (y0, y1) = load_128b_block(block);
    x0 = fold_16_reflected_vpclmul(x0, coeff_128, y0);
    x1 = fold_16_reflected_vpclmul(x1, coeff_128, y1);
  }

  finalize_vpclmul_state(x0, x1, keys)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn crc16_width32_vpclmul_stream(
  mut state: u16,
  data: &[u8],
  keys: &[u64; 23],
  stream: &Width32StreamConstants,
  streams: u8,
  portable: fn(u16, &[u8]) -> u16,
) -> u16 {
  let (left, middle, right) = data.align_to::<[Simd128; 8]>();
  let Some((first, rest)) = middle.split_first() else {
    return crc16_width32_pclmul_small(state, data, keys, portable);
  };

  state = portable(state, left);
  let state32 = match streams {
    8 => {
      update_simd_width32_reflected_vpclmul_8way(state as u32, middle, stream.fold_1024b, &stream.combine_8way, keys)
    }
    7 => update_simd_width32_reflected_vpclmul_7way(state as u32, middle, stream.fold_896b, &stream.combine_7way, keys),
    4 => update_simd_width32_reflected_vpclmul_4way(state as u32, middle, stream.fold_512b, &stream.combine_4way, keys),
    2 => update_simd_width32_reflected_vpclmul_2way(state as u32, middle, stream.fold_256b, keys),
    _ => update_simd_width32_reflected_vpclmul(state as u32, first, rest, keys),
  };
  state = state32 as u16;
  portable(state, right)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn crc16_width32_vpclmul(mut state: u16, data: &[u8], keys: &[u64; 23], portable: fn(u16, &[u8]) -> u16) -> u16 {
  let (left, middle, right) = data.align_to::<[Simd128; 8]>();
  let Some((first, rest)) = middle.split_first() else {
    return crc16_width32_pclmul_small(state, data, keys, portable);
  };

  state = portable(state, left);
  let state32 = update_simd_width32_reflected_vpclmul(state as u32, first, rest, keys);
  state = state32 as u16;
  portable(state, right)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public Safe Kernels (matching CRC-64 pure fn(u16, &[u8]) -> u16 signature)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-16/CCITT PCLMULQDQ kernel.
///
/// # Safety
///
/// Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
#[inline]
pub fn crc16_ccitt_pclmul_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
  unsafe {
    crc16_width32_pclmul(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/CCITT PCLMULQDQ small-buffer kernel.
///
/// Optimized for inputs smaller than a folding block (128 bytes).
///
/// # Safety
///
/// Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
#[inline]
pub fn crc16_ccitt_pclmul_small_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
  unsafe {
    crc16_width32_pclmul_small(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/CCITT PCLMULQDQ kernel (2-way multi-stream).
#[inline]
pub fn crc16_ccitt_pclmul_2way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_pclmul_stream(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      &CRC16_CCITT_STREAM_REFLECTED,
      2,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/CCITT PCLMULQDQ kernel (4-way multi-stream).
#[inline]
pub fn crc16_ccitt_pclmul_4way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_pclmul_stream(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      &CRC16_CCITT_STREAM_REFLECTED,
      4,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/CCITT PCLMULQDQ kernel (7-way multi-stream).
#[inline]
pub fn crc16_ccitt_pclmul_7way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_pclmul_stream(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      &CRC16_CCITT_STREAM_REFLECTED,
      7,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/CCITT PCLMULQDQ kernel (8-way multi-stream).
#[inline]
pub fn crc16_ccitt_pclmul_8way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_pclmul_stream(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      &CRC16_CCITT_STREAM_REFLECTED,
      8,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/CCITT VPCLMULQDQ kernel (AVX-512).
///
/// # Safety
///
/// Dispatcher verifies VPCLMULQDQ + AVX-512 before selecting this kernel.
#[inline]
pub fn crc16_ccitt_vpclmul_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies VPCLMULQDQ + AVX-512 before selecting this kernel.
  unsafe {
    crc16_width32_vpclmul(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/CCITT VPCLMULQDQ kernel (2-way multi-stream).
#[inline]
pub fn crc16_ccitt_vpclmul_2way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_vpclmul_stream(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      &CRC16_CCITT_STREAM_REFLECTED,
      2,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/CCITT VPCLMULQDQ kernel (4-way multi-stream).
#[inline]
pub fn crc16_ccitt_vpclmul_4way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_vpclmul_stream(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      &CRC16_CCITT_STREAM_REFLECTED,
      4,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/CCITT VPCLMULQDQ kernel (7-way multi-stream).
#[inline]
pub fn crc16_ccitt_vpclmul_7way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_vpclmul_stream(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      &CRC16_CCITT_STREAM_REFLECTED,
      7,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/CCITT VPCLMULQDQ kernel (8-way multi-stream).
#[inline]
pub fn crc16_ccitt_vpclmul_8way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_vpclmul_stream(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      &CRC16_CCITT_STREAM_REFLECTED,
      8,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/IBM PCLMULQDQ kernel.
///
/// # Safety
///
/// Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
#[inline]
pub fn crc16_ibm_pclmul_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
  unsafe { crc16_width32_pclmul(crc, data, &CRC16_IBM_KEYS_REFLECTED, super::portable::crc16_ibm_slice8) }
}

/// CRC-16/IBM PCLMULQDQ small-buffer kernel.
///
/// Optimized for inputs smaller than a folding block (128 bytes).
///
/// # Safety
///
/// Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
#[inline]
pub fn crc16_ibm_pclmul_small_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
  unsafe { crc16_width32_pclmul_small(crc, data, &CRC16_IBM_KEYS_REFLECTED, super::portable::crc16_ibm_slice8) }
}

/// CRC-16/IBM PCLMULQDQ kernel (2-way multi-stream).
#[inline]
pub fn crc16_ibm_pclmul_2way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_pclmul_stream(
      crc,
      data,
      &CRC16_IBM_KEYS_REFLECTED,
      &CRC16_IBM_STREAM_REFLECTED,
      2,
      super::portable::crc16_ibm_slice8,
    )
  }
}

/// CRC-16/IBM PCLMULQDQ kernel (4-way multi-stream).
#[inline]
pub fn crc16_ibm_pclmul_4way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_pclmul_stream(
      crc,
      data,
      &CRC16_IBM_KEYS_REFLECTED,
      &CRC16_IBM_STREAM_REFLECTED,
      4,
      super::portable::crc16_ibm_slice8,
    )
  }
}

/// CRC-16/IBM PCLMULQDQ kernel (7-way multi-stream).
#[inline]
pub fn crc16_ibm_pclmul_7way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_pclmul_stream(
      crc,
      data,
      &CRC16_IBM_KEYS_REFLECTED,
      &CRC16_IBM_STREAM_REFLECTED,
      7,
      super::portable::crc16_ibm_slice8,
    )
  }
}

/// CRC-16/IBM PCLMULQDQ kernel (8-way multi-stream).
#[inline]
pub fn crc16_ibm_pclmul_8way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_pclmul_stream(
      crc,
      data,
      &CRC16_IBM_KEYS_REFLECTED,
      &CRC16_IBM_STREAM_REFLECTED,
      8,
      super::portable::crc16_ibm_slice8,
    )
  }
}

/// CRC-16/IBM VPCLMULQDQ kernel (AVX-512).
///
/// # Safety
///
/// Dispatcher verifies VPCLMULQDQ + AVX-512 before selecting this kernel.
#[inline]
pub fn crc16_ibm_vpclmul_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies VPCLMULQDQ + AVX-512 before selecting this kernel.
  unsafe { crc16_width32_vpclmul(crc, data, &CRC16_IBM_KEYS_REFLECTED, super::portable::crc16_ibm_slice8) }
}

/// CRC-16/IBM VPCLMULQDQ kernel (2-way multi-stream).
#[inline]
pub fn crc16_ibm_vpclmul_2way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_vpclmul_stream(
      crc,
      data,
      &CRC16_IBM_KEYS_REFLECTED,
      &CRC16_IBM_STREAM_REFLECTED,
      2,
      super::portable::crc16_ibm_slice8,
    )
  }
}

/// CRC-16/IBM VPCLMULQDQ kernel (4-way multi-stream).
#[inline]
pub fn crc16_ibm_vpclmul_4way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_vpclmul_stream(
      crc,
      data,
      &CRC16_IBM_KEYS_REFLECTED,
      &CRC16_IBM_STREAM_REFLECTED,
      4,
      super::portable::crc16_ibm_slice8,
    )
  }
}

/// CRC-16/IBM VPCLMULQDQ kernel (7-way multi-stream).
#[inline]
pub fn crc16_ibm_vpclmul_7way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_vpclmul_stream(
      crc,
      data,
      &CRC16_IBM_KEYS_REFLECTED,
      &CRC16_IBM_STREAM_REFLECTED,
      7,
      super::portable::crc16_ibm_slice8,
    )
  }
}

/// CRC-16/IBM VPCLMULQDQ kernel (8-way multi-stream).
#[inline]
pub fn crc16_ibm_vpclmul_8way_safe(crc: u16, data: &[u8]) -> u16 {
  unsafe {
    crc16_width32_vpclmul_stream(
      crc,
      data,
      &CRC16_IBM_KEYS_REFLECTED,
      &CRC16_IBM_STREAM_REFLECTED,
      8,
      super::portable::crc16_ibm_slice8,
    )
  }
}
