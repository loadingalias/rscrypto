//! x86_64 hardware-accelerated CRC-64 kernels (XZ + NVME).
//!
//! This is a PCLMULQDQ implementation derived from the Intel/TiKV folding
//! algorithm (also used by `crc64fast` / `crc64fast-nvme`).
//!
//! # Safety
//!
//! Uses `unsafe` for x86 SIMD intrinsics. Callers must ensure PCLMULQDQ is
//! available before executing the accelerated path (the dispatcher does this).
#![allow(unsafe_code)]
#![allow(dead_code)] // Kernels wired up via dispatcher
// SAFETY: All indexing is over fixed-size arrays with in-bounds constant indices.
#![allow(clippy::indexing_slicing)]
// This module is intrinsics-heavy; keep unsafe blocks readable.
#![allow(unsafe_op_in_unsafe_fn)]

use core::{
  arch::x86_64::*,
  ops::{BitXor, BitXorAssign},
};

use crate::common::clmul::{CRC64_NVME_STREAM, CRC64_XZ_STREAM, Crc64ClmulConstants};

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
struct Simd(__m128i);

impl BitXor for Simd {
  type Output = Self;

  #[inline]
  fn bitxor(self, other: Self) -> Self {
    // SAFETY: `_mm_xor_si128` is available on all x86_64 (SSE2 baseline).
    unsafe { Self(_mm_xor_si128(self.0, other.0)) }
  }
}

impl BitXorAssign for Simd {
  #[inline]
  fn bitxor_assign(&mut self, other: Self) {
    *self = *self ^ other;
  }
}

impl Simd {
  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn new(high: u64, low: u64) -> Self {
    Self(_mm_set_epi64x(high as i64, low as i64))
  }

  /// Fold 16 bytes: `(coeff.low ⊗ self.low) ⊕ (coeff.high ⊗ self.high)`.
  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn fold_16(self, coeff: Self) -> Self {
    let h = _mm_clmulepi64_si128::<0x11>(self.0, coeff.0);
    let l = _mm_clmulepi64_si128::<0x00>(self.0, coeff.0);
    Self(_mm_xor_si128(h, l))
  }

  /// Fold 8 bytes: `self.high ⊕ (coeff ⊗ self.low)`.
  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn fold_8(self, coeff: u64) -> Self {
    let coeff = Self::new(0, coeff);
    let h = _mm_clmulepi64_si128::<0x00>(self.0, coeff.0);
    let l = _mm_srli_si128::<8>(self.0);
    Self(_mm_xor_si128(h, l))
  }

  /// Barrett reduction to finalize the CRC.
  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn barrett(self, poly: u64, mu: u64) -> u64 {
    let polymu = Self::new(poly, mu);
    let t1 = _mm_clmulepi64_si128::<0x00>(self.0, polymu.0);
    let h = _mm_slli_si128::<8>(t1);
    let l = _mm_clmulepi64_si128::<0x10>(t1, polymu.0);
    let reduced = Self(_mm_xor_si128(_mm_xor_si128(h, l), self.0));

    // Extract high 64 bits without requiring SSE4.1.
    let hi = _mm_srli_si128::<8>(reduced.0);
    _mm_cvtsi128_si64(hi) as u64
  }
}

#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn update_simd(state: u64, first: &[Simd; 8], rest: &[[Simd; 8]], consts: &Crc64ClmulConstants) -> u64 {
  let mut x = *first;

  // XOR the initial CRC into the first lane.
  x[0] ^= Simd::new(0, state);

  // 128-byte folding.
  let coeff = Simd::new(consts.fold_128b.0, consts.fold_128b.1);
  for chunk in rest {
    // Manually unrolled for better ILP and to avoid iterator overhead.
    let t0 = x[0].fold_16(coeff);
    let t1 = x[1].fold_16(coeff);
    let t2 = x[2].fold_16(coeff);
    let t3 = x[3].fold_16(coeff);
    let t4 = x[4].fold_16(coeff);
    let t5 = x[5].fold_16(coeff);
    let t6 = x[6].fold_16(coeff);
    let t7 = x[7].fold_16(coeff);

    x[0] = chunk[0] ^ t0;
    x[1] = chunk[1] ^ t1;
    x[2] = chunk[2] ^ t2;
    x[3] = chunk[3] ^ t3;
    x[4] = chunk[4] ^ t4;
    x[5] = chunk[5] ^ t5;
    x[6] = chunk[6] ^ t6;
    x[7] = chunk[7] ^ t7;
  }

  fold_tail(x, consts)
}

#[inline(always)]
unsafe fn fold_tail(x: [Simd; 8], consts: &Crc64ClmulConstants) -> u64 {
  // Tail reduction (8×16B → 1×16B), unrolled for throughput.
  let c0 = Simd::new(consts.tail_fold_16b[0].0, consts.tail_fold_16b[0].1);
  let c1 = Simd::new(consts.tail_fold_16b[1].0, consts.tail_fold_16b[1].1);
  let c2 = Simd::new(consts.tail_fold_16b[2].0, consts.tail_fold_16b[2].1);
  let c3 = Simd::new(consts.tail_fold_16b[3].0, consts.tail_fold_16b[3].1);
  let c4 = Simd::new(consts.tail_fold_16b[4].0, consts.tail_fold_16b[4].1);
  let c5 = Simd::new(consts.tail_fold_16b[5].0, consts.tail_fold_16b[5].1);
  let c6 = Simd::new(consts.tail_fold_16b[6].0, consts.tail_fold_16b[6].1);

  let mut acc = x[7];
  acc ^= x[0].fold_16(c0);
  acc ^= x[1].fold_16(c1);
  acc ^= x[2].fold_16(c2);
  acc ^= x[3].fold_16(c3);
  acc ^= x[4].fold_16(c4);
  acc ^= x[5].fold_16(c5);
  acc ^= x[6].fold_16(c6);

  acc.fold_8(consts.fold_8b).barrett(consts.poly, consts.mu)
}

// ─────────────────────────────────────────────────────────────────────────────
// PCLMULQDQ multi-stream (2-way, 128B blocks)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn fold_block_128(x: &mut [Simd; 8], chunk: &[Simd; 8], coeff: Simd) {
  let t0 = x[0].fold_16(coeff);
  let t1 = x[1].fold_16(coeff);
  let t2 = x[2].fold_16(coeff);
  let t3 = x[3].fold_16(coeff);
  let t4 = x[4].fold_16(coeff);
  let t5 = x[5].fold_16(coeff);
  let t6 = x[6].fold_16(coeff);
  let t7 = x[7].fold_16(coeff);

  x[0] = chunk[0] ^ t0;
  x[1] = chunk[1] ^ t1;
  x[2] = chunk[2] ^ t2;
  x[3] = chunk[3] ^ t3;
  x[4] = chunk[4] ^ t4;
  x[5] = chunk[5] ^ t5;
  x[6] = chunk[6] ^ t6;
  x[7] = chunk[7] ^ t7;
}

#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn update_simd_2way(
  state: u64,
  blocks: &[[Simd; 8]],
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
) -> u64 {
  debug_assert!(blocks.len() >= 2);

  let coeff_256 = Simd::new(fold_256b.0, fold_256b.1);
  let coeff_128 = Simd::new(consts.fold_128b.0, consts.fold_128b.1);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];

  // Inject CRC into stream 0 (block 0).
  s0[0] ^= Simd::new(0, state);

  // Process the largest even prefix with 2-way striping.
  let mut i = 2;
  let even = blocks.len() & !1usize;
  while i < even {
    fold_block_128(&mut s0, &blocks[i], coeff_256);
    fold_block_128(&mut s1, &blocks[i + 1], coeff_256);
    i = i.strict_add(2);
  }

  // Merge streams: A·s0 ⊕ s1 (A = shift by 128B).
  let mut combined = s1;
  combined[0] ^= s0[0].fold_16(coeff_128);
  combined[1] ^= s0[1].fold_16(coeff_128);
  combined[2] ^= s0[2].fold_16(coeff_128);
  combined[3] ^= s0[3].fold_16(coeff_128);
  combined[4] ^= s0[4].fold_16(coeff_128);
  combined[5] ^= s0[5].fold_16(coeff_128);
  combined[6] ^= s0[6].fold_16(coeff_128);
  combined[7] ^= s0[7].fold_16(coeff_128);

  // Handle any remaining block (odd tail) sequentially.
  if even != blocks.len() {
    fold_block_128(&mut combined, &blocks[even], coeff_128);
  }

  fold_tail(combined, consts)
}

#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn update_simd_4way(
  state: u64,
  blocks: &[[Simd; 8]],
  fold_512b: (u64, u64),
  combine: &[(u64, u64); 3],
  consts: &Crc64ClmulConstants,
) -> u64 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 4 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd(state, first, rest, consts);
  }

  let aligned = (blocks.len() / 4) * 4;

  let coeff_512 = Simd::new(fold_512b.0, fold_512b.1);
  let coeff_128 = Simd::new(consts.fold_128b.0, consts.fold_128b.1);
  let c384 = Simd::new(combine[0].0, combine[0].1);
  let c256 = Simd::new(combine[1].0, combine[1].1);
  let c128 = Simd::new(combine[2].0, combine[2].1);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];
  let mut s2 = blocks[2];
  let mut s3 = blocks[3];

  // Inject CRC into stream 0.
  s0[0] ^= Simd::new(0, state);

  let mut i = 4;
  while i < aligned {
    fold_block_128(&mut s0, &blocks[i], coeff_512);
    fold_block_128(&mut s1, &blocks[i + 1], coeff_512);
    fold_block_128(&mut s2, &blocks[i + 2], coeff_512);
    fold_block_128(&mut s3, &blocks[i + 3], coeff_512);
    i = i.strict_add(4);
  }

  // Merge: A^3·s0 ⊕ A^2·s1 ⊕ A·s2 ⊕ s3.
  let mut combined = s3;
  combined[0] ^= s2[0].fold_16(c128);
  combined[1] ^= s2[1].fold_16(c128);
  combined[2] ^= s2[2].fold_16(c128);
  combined[3] ^= s2[3].fold_16(c128);
  combined[4] ^= s2[4].fold_16(c128);
  combined[5] ^= s2[5].fold_16(c128);
  combined[6] ^= s2[6].fold_16(c128);
  combined[7] ^= s2[7].fold_16(c128);

  combined[0] ^= s1[0].fold_16(c256);
  combined[1] ^= s1[1].fold_16(c256);
  combined[2] ^= s1[2].fold_16(c256);
  combined[3] ^= s1[3].fold_16(c256);
  combined[4] ^= s1[4].fold_16(c256);
  combined[5] ^= s1[5].fold_16(c256);
  combined[6] ^= s1[6].fold_16(c256);
  combined[7] ^= s1[7].fold_16(c256);

  combined[0] ^= s0[0].fold_16(c384);
  combined[1] ^= s0[1].fold_16(c384);
  combined[2] ^= s0[2].fold_16(c384);
  combined[3] ^= s0[3].fold_16(c384);
  combined[4] ^= s0[4].fold_16(c384);
  combined[5] ^= s0[5].fold_16(c384);
  combined[6] ^= s0[6].fold_16(c384);
  combined[7] ^= s0[7].fold_16(c384);

  for block in &blocks[aligned..] {
    fold_block_128(&mut combined, block, coeff_128);
  }

  fold_tail(combined, consts)
}

#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn update_simd_7way(
  state: u64,
  blocks: &[[Simd; 8]],
  fold_896b: (u64, u64),
  combine: &[(u64, u64); 6],
  consts: &Crc64ClmulConstants,
) -> u64 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 7 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd(state, first, rest, consts);
  }

  let aligned = (blocks.len() / 7) * 7;

  let coeff_896 = Simd::new(fold_896b.0, fold_896b.1);
  let coeff_128 = Simd::new(consts.fold_128b.0, consts.fold_128b.1);

  let c768 = Simd::new(combine[0].0, combine[0].1);
  let c640 = Simd::new(combine[1].0, combine[1].1);
  let c512 = Simd::new(combine[2].0, combine[2].1);
  let c384 = Simd::new(combine[3].0, combine[3].1);
  let c256 = Simd::new(combine[4].0, combine[4].1);
  let c128 = Simd::new(combine[5].0, combine[5].1);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];
  let mut s2 = blocks[2];
  let mut s3 = blocks[3];
  let mut s4 = blocks[4];
  let mut s5 = blocks[5];
  let mut s6 = blocks[6];

  // Inject CRC into stream 0.
  s0[0] ^= Simd::new(0, state);

  let mut i = 7;
  while i < aligned {
    fold_block_128(&mut s0, &blocks[i], coeff_896);
    fold_block_128(&mut s1, &blocks[i + 1], coeff_896);
    fold_block_128(&mut s2, &blocks[i + 2], coeff_896);
    fold_block_128(&mut s3, &blocks[i + 3], coeff_896);
    fold_block_128(&mut s4, &blocks[i + 4], coeff_896);
    fold_block_128(&mut s5, &blocks[i + 5], coeff_896);
    fold_block_128(&mut s6, &blocks[i + 6], coeff_896);
    i = i.strict_add(7);
  }

  // Merge: A^6·s0 ⊕ A^5·s1 ⊕ … ⊕ A·s5 ⊕ s6.
  let mut combined = s6;
  combined[0] ^= s5[0].fold_16(c128);
  combined[1] ^= s5[1].fold_16(c128);
  combined[2] ^= s5[2].fold_16(c128);
  combined[3] ^= s5[3].fold_16(c128);
  combined[4] ^= s5[4].fold_16(c128);
  combined[5] ^= s5[5].fold_16(c128);
  combined[6] ^= s5[6].fold_16(c128);
  combined[7] ^= s5[7].fold_16(c128);

  combined[0] ^= s4[0].fold_16(c256);
  combined[1] ^= s4[1].fold_16(c256);
  combined[2] ^= s4[2].fold_16(c256);
  combined[3] ^= s4[3].fold_16(c256);
  combined[4] ^= s4[4].fold_16(c256);
  combined[5] ^= s4[5].fold_16(c256);
  combined[6] ^= s4[6].fold_16(c256);
  combined[7] ^= s4[7].fold_16(c256);

  combined[0] ^= s3[0].fold_16(c384);
  combined[1] ^= s3[1].fold_16(c384);
  combined[2] ^= s3[2].fold_16(c384);
  combined[3] ^= s3[3].fold_16(c384);
  combined[4] ^= s3[4].fold_16(c384);
  combined[5] ^= s3[5].fold_16(c384);
  combined[6] ^= s3[6].fold_16(c384);
  combined[7] ^= s3[7].fold_16(c384);

  combined[0] ^= s2[0].fold_16(c512);
  combined[1] ^= s2[1].fold_16(c512);
  combined[2] ^= s2[2].fold_16(c512);
  combined[3] ^= s2[3].fold_16(c512);
  combined[4] ^= s2[4].fold_16(c512);
  combined[5] ^= s2[5].fold_16(c512);
  combined[6] ^= s2[6].fold_16(c512);
  combined[7] ^= s2[7].fold_16(c512);

  combined[0] ^= s1[0].fold_16(c640);
  combined[1] ^= s1[1].fold_16(c640);
  combined[2] ^= s1[2].fold_16(c640);
  combined[3] ^= s1[3].fold_16(c640);
  combined[4] ^= s1[4].fold_16(c640);
  combined[5] ^= s1[5].fold_16(c640);
  combined[6] ^= s1[6].fold_16(c640);
  combined[7] ^= s1[7].fold_16(c640);

  combined[0] ^= s0[0].fold_16(c768);
  combined[1] ^= s0[1].fold_16(c768);
  combined[2] ^= s0[2].fold_16(c768);
  combined[3] ^= s0[3].fold_16(c768);
  combined[4] ^= s0[4].fold_16(c768);
  combined[5] ^= s0[5].fold_16(c768);
  combined[6] ^= s0[6].fold_16(c768);
  combined[7] ^= s0[7].fold_16(c768);

  for block in &blocks[aligned..] {
    fold_block_128(&mut combined, block, coeff_128);
  }

  fold_tail(combined, consts)
}

#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn update_simd_8way(
  state: u64,
  blocks: &[[Simd; 8]],
  fold_1024b: (u64, u64),
  combine: &[(u64, u64); 7],
  consts: &Crc64ClmulConstants,
) -> u64 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 8 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd(state, first, rest, consts);
  }

  let aligned = (blocks.len() / 8) * 8;

  let coeff_1024 = Simd::new(fold_1024b.0, fold_1024b.1);
  let coeff_128 = Simd::new(consts.fold_128b.0, consts.fold_128b.1);

  let c896 = Simd::new(combine[0].0, combine[0].1);
  let c768 = Simd::new(combine[1].0, combine[1].1);
  let c640 = Simd::new(combine[2].0, combine[2].1);
  let c512 = Simd::new(combine[3].0, combine[3].1);
  let c384 = Simd::new(combine[4].0, combine[4].1);
  let c256 = Simd::new(combine[5].0, combine[5].1);
  let c128 = Simd::new(combine[6].0, combine[6].1);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];
  let mut s2 = blocks[2];
  let mut s3 = blocks[3];
  let mut s4 = blocks[4];
  let mut s5 = blocks[5];
  let mut s6 = blocks[6];
  let mut s7 = blocks[7];

  // Inject CRC into stream 0.
  s0[0] ^= Simd::new(0, state);

  let mut i = 8;
  while i < aligned {
    fold_block_128(&mut s0, &blocks[i], coeff_1024);
    fold_block_128(&mut s1, &blocks[i + 1], coeff_1024);
    fold_block_128(&mut s2, &blocks[i + 2], coeff_1024);
    fold_block_128(&mut s3, &blocks[i + 3], coeff_1024);
    fold_block_128(&mut s4, &blocks[i + 4], coeff_1024);
    fold_block_128(&mut s5, &blocks[i + 5], coeff_1024);
    fold_block_128(&mut s6, &blocks[i + 6], coeff_1024);
    fold_block_128(&mut s7, &blocks[i + 7], coeff_1024);
    i = i.strict_add(8);
  }

  // Merge: A^7·s0 ⊕ A^6·s1 ⊕ … ⊕ A·s6 ⊕ s7.
  let mut combined = s7;
  combined[0] ^= s6[0].fold_16(c128);
  combined[1] ^= s6[1].fold_16(c128);
  combined[2] ^= s6[2].fold_16(c128);
  combined[3] ^= s6[3].fold_16(c128);
  combined[4] ^= s6[4].fold_16(c128);
  combined[5] ^= s6[5].fold_16(c128);
  combined[6] ^= s6[6].fold_16(c128);
  combined[7] ^= s6[7].fold_16(c128);

  combined[0] ^= s5[0].fold_16(c256);
  combined[1] ^= s5[1].fold_16(c256);
  combined[2] ^= s5[2].fold_16(c256);
  combined[3] ^= s5[3].fold_16(c256);
  combined[4] ^= s5[4].fold_16(c256);
  combined[5] ^= s5[5].fold_16(c256);
  combined[6] ^= s5[6].fold_16(c256);
  combined[7] ^= s5[7].fold_16(c256);

  combined[0] ^= s4[0].fold_16(c384);
  combined[1] ^= s4[1].fold_16(c384);
  combined[2] ^= s4[2].fold_16(c384);
  combined[3] ^= s4[3].fold_16(c384);
  combined[4] ^= s4[4].fold_16(c384);
  combined[5] ^= s4[5].fold_16(c384);
  combined[6] ^= s4[6].fold_16(c384);
  combined[7] ^= s4[7].fold_16(c384);

  combined[0] ^= s3[0].fold_16(c512);
  combined[1] ^= s3[1].fold_16(c512);
  combined[2] ^= s3[2].fold_16(c512);
  combined[3] ^= s3[3].fold_16(c512);
  combined[4] ^= s3[4].fold_16(c512);
  combined[5] ^= s3[5].fold_16(c512);
  combined[6] ^= s3[6].fold_16(c512);
  combined[7] ^= s3[7].fold_16(c512);

  combined[0] ^= s2[0].fold_16(c640);
  combined[1] ^= s2[1].fold_16(c640);
  combined[2] ^= s2[2].fold_16(c640);
  combined[3] ^= s2[3].fold_16(c640);
  combined[4] ^= s2[4].fold_16(c640);
  combined[5] ^= s2[5].fold_16(c640);
  combined[6] ^= s2[6].fold_16(c640);
  combined[7] ^= s2[7].fold_16(c640);

  combined[0] ^= s1[0].fold_16(c768);
  combined[1] ^= s1[1].fold_16(c768);
  combined[2] ^= s1[2].fold_16(c768);
  combined[3] ^= s1[3].fold_16(c768);
  combined[4] ^= s1[4].fold_16(c768);
  combined[5] ^= s1[5].fold_16(c768);
  combined[6] ^= s1[6].fold_16(c768);
  combined[7] ^= s1[7].fold_16(c768);

  combined[0] ^= s0[0].fold_16(c896);
  combined[1] ^= s0[1].fold_16(c896);
  combined[2] ^= s0[2].fold_16(c896);
  combined[3] ^= s0[3].fold_16(c896);
  combined[4] ^= s0[4].fold_16(c896);
  combined[5] ^= s0[5].fold_16(c896);
  combined[6] ^= s0[6].fold_16(c896);
  combined[7] ^= s0[7].fold_16(c896);

  for block in &blocks[aligned..] {
    fold_block_128(&mut combined, block, coeff_128);
  }

  fold_tail(combined, consts)
}

#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn crc64_pclmul(mut state: u64, bytes: &[u8], consts: &Crc64ClmulConstants, tables: &[[u64; 256]; 8]) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if let Some((first, rest)) = middle.split_first() {
    state = super::portable::crc64_slice8(state, left, tables);
    state = update_simd(state, first, rest, consts);
    super::portable::crc64_slice8(state, right, tables)
  } else {
    super::portable::crc64_slice8(state, bytes, tables)
  }
}

#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn crc64_pclmul_2way(
  mut state: u64,
  bytes: &[u8],
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 8],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return super::portable::crc64_slice8(state, bytes, tables);
  }

  state = super::portable::crc64_slice8(state, left, tables);

  if middle.len() >= 2 {
    state = update_simd_2way(state, middle, fold_256b, consts);
  } else if let Some((first, rest)) = middle.split_first() {
    state = update_simd(state, first, rest, consts);
  }

  super::portable::crc64_slice8(state, right, tables)
}

#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn crc64_pclmul_4way(
  mut state: u64,
  bytes: &[u8],
  fold_512b: (u64, u64),
  combine: &[(u64, u64); 3],
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 8],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return super::portable::crc64_slice8(state, bytes, tables);
  }

  state = super::portable::crc64_slice8(state, left, tables);
  state = update_simd_4way(state, middle, fold_512b, combine, consts);
  super::portable::crc64_slice8(state, right, tables)
}

#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn crc64_pclmul_7way(
  mut state: u64,
  bytes: &[u8],
  fold_896b: (u64, u64),
  combine: &[(u64, u64); 6],
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 8],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return super::portable::crc64_slice8(state, bytes, tables);
  }

  state = super::portable::crc64_slice8(state, left, tables);
  state = update_simd_7way(state, middle, fold_896b, combine, consts);
  super::portable::crc64_slice8(state, right, tables)
}

#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn crc64_pclmul_8way(
  mut state: u64,
  bytes: &[u8],
  fold_1024b: (u64, u64),
  combine: &[(u64, u64); 7],
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 8],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return super::portable::crc64_slice8(state, bytes, tables);
  }

  state = super::portable::crc64_slice8(state, left, tables);
  state = update_simd_8way(state, middle, fold_1024b, combine, consts);
  super::portable::crc64_slice8(state, right, tables)
}

/// Small-buffer CLMUL path: fold one 16-byte lane at a time.
///
/// This targets the regime where full 128-byte folding has too much setup cost,
/// but CLMUL still outperforms table CRC (typically ~16..127 bytes depending on CPU).
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn crc64_pclmul_small(
  mut state: u64,
  bytes: &[u8],
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 8],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<Simd>();

  // Prefix: portable until 16B alignment.
  state = super::portable::crc64_slice8(state, left, tables);

  // If we don't have any full 16B lane, finish portably.
  let Some((first, rest)) = middle.split_first() else {
    return super::portable::crc64_slice8(state, right, tables);
  };

  let mut acc = *first;
  acc ^= Simd::new(0, state);

  // Shift-by-16B folding coefficient (K_127, K_191).
  let coeff_16b = Simd::new(consts.tail_fold_16b[6].0, consts.tail_fold_16b[6].1);

  for chunk in rest {
    acc = *chunk ^ acc.fold_16(coeff_16b);
  }

  // Reduce 16B → 8B → u64, then finish any tail bytes portably.
  state = acc.fold_8(consts.fold_8b).barrett(consts.poly, consts.mu);
  super::portable::crc64_slice8(state, right, tables)
}

// ─────────────────────────────────────────────────────────────────────────────
// VPCLMULQDQ (AVX-512) folding
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn fold16_4x(x: __m512i, coeff: __m512i) -> __m512i {
  let h = _mm512_clmulepi64_epi128::<0x11>(x, coeff);
  let l = _mm512_clmulepi64_epi128::<0x00>(x, coeff);
  _mm512_xor_si512(h, l)
}

/// Fold and XOR with data using VPTERNLOGD (3-way XOR in one instruction).
///
/// Computes: `data ^ clmul_hi(x, coeff) ^ clmul_lo(x, coeff)`
///
/// This saves one XOR instruction per fold operation compared to the
/// two-step `data ^ fold16_4x(x, coeff)` pattern. The ternary logic
/// immediate 0x96 encodes XOR(a, XOR(b, c)) = a ^ b ^ c.
#[inline]
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn fold16_4x_ternlog(x: __m512i, data: __m512i, coeff: __m512i) -> __m512i {
  let h = _mm512_clmulepi64_epi128::<0x11>(x, coeff);
  let l = _mm512_clmulepi64_epi128::<0x00>(x, coeff);
  // VPTERNLOGD: 3-way XOR (imm8 = 0x96 = a ^ b ^ c)
  _mm512_ternarylogic_epi64::<0x96>(data, h, l)
}

#[inline]
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn vpclmul_coeff(pair: (u64, u64)) -> __m512i {
  _mm512_set_epi64(
    pair.0 as i64,
    pair.1 as i64,
    pair.0 as i64,
    pair.1 as i64,
    pair.0 as i64,
    pair.1 as i64,
    pair.0 as i64,
    pair.1 as i64,
  )
}

#[inline]
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn load_128b_block(block: &[Simd; 8]) -> (__m512i, __m512i) {
  let ptr = block as *const [Simd; 8] as *const u8;
  // 8×16B lanes packed as 2×64B vectors (4 lanes each).
  let y0 = _mm512_loadu_si512(ptr.cast::<__m512i>());
  let y1 = _mm512_loadu_si512(ptr.add(64).cast::<__m512i>());
  (y0, y1)
}

#[inline]
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn finalize_vpclmul_state(x0: __m512i, x1: __m512i, consts: &Crc64ClmulConstants) -> u64 {
  // Reuse the well-tested 128-bit tail fold + Barrett reduction by extracting
  // the 8×16B lanes directly (avoids a store+reload round-trip).
  let lanes: [Simd; 8] = [
    Simd(_mm512_extracti32x4_epi32::<0>(x0)),
    Simd(_mm512_extracti32x4_epi32::<1>(x0)),
    Simd(_mm512_extracti32x4_epi32::<2>(x0)),
    Simd(_mm512_extracti32x4_epi32::<3>(x0)),
    Simd(_mm512_extracti32x4_epi32::<0>(x1)),
    Simd(_mm512_extracti32x4_epi32::<1>(x1)),
    Simd(_mm512_extracti32x4_epi32::<2>(x1)),
    Simd(_mm512_extracti32x4_epi32::<3>(x1)),
  ];
  fold_tail(lanes, consts)
}

#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn update_simd_vpclmul(state: u64, first: &[Simd; 8], rest: &[[Simd; 8]], consts: &Crc64ClmulConstants) -> u64 {
  let first_ptr = first as *const [Simd; 8] as *const u8;

  // 8×16B lanes packed as 2×64B vectors (4 lanes each).
  let mut x0 = _mm512_loadu_si512(first_ptr.cast::<__m512i>());
  let mut x1 = _mm512_loadu_si512(first_ptr.add(64).cast::<__m512i>());

  // XOR the initial CRC into lane 0 (low 64 bits of the first 16-byte lane).
  let crc_mask = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, state as i64);
  x0 = _mm512_xor_si512(x0, crc_mask);

  // Broadcast 128-byte fold coefficient across the 4×128-bit lanes of each vector.
  let coeff = vpclmul_coeff(consts.fold_128b);

  for chunk in rest {
    let ptr = chunk as *const [Simd; 8] as *const u8;
    let y0 = _mm512_loadu_si512(ptr.cast::<__m512i>());
    let y1 = _mm512_loadu_si512(ptr.add(64).cast::<__m512i>());

    // VPTERNLOGD: fold + XOR in one instruction per vector
    x0 = fold16_4x_ternlog(x0, y0, coeff);
    x1 = fold16_4x_ternlog(x1, y1, coeff);
  }

  finalize_vpclmul_state(x0, x1, consts)
}

#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn crc64_vpclmul(mut state: u64, bytes: &[u8], consts: &Crc64ClmulConstants, tables: &[[u64; 256]; 8]) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if let Some((first, rest)) = middle.split_first() {
    state = super::portable::crc64_slice8(state, left, tables);
    state = update_simd_vpclmul(state, first, rest, consts);
    super::portable::crc64_slice8(state, right, tables)
  } else {
    super::portable::crc64_slice8(state, bytes, tables)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// VPCLMULQDQ multi-stream (2/4/7-way, 128B blocks)
// ─────────────────────────────────────────────────────────────────────────────

#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn update_simd_vpclmul_2way(
  state: u64,
  blocks: &[[Simd; 8]],
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
) -> u64 {
  debug_assert!(!blocks.is_empty());

  // Fallback to the single-stream kernel when we don't have enough blocks.
  if blocks.len() < 2 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_vpclmul(state, first, rest, consts);
  }

  let even = blocks.len() & !1usize;

  let (mut x0_0, mut x1_0) = load_128b_block(&blocks[0]);
  let (mut x0_1, mut x1_1) = load_128b_block(&blocks[1]);

  // Inject CRC into stream 0.
  let crc_mask = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, state as i64);
  x0_0 = _mm512_xor_si512(x0_0, crc_mask);

  let coeff_256 = vpclmul_coeff(fold_256b);
  let coeff_128 = vpclmul_coeff(consts.fold_128b);

  let mut i = 2;
  while i < even {
    let (y0, y1) = load_128b_block(&blocks[i]);
    x0_0 = fold16_4x_ternlog(x0_0, y0, coeff_256);
    x1_0 = fold16_4x_ternlog(x1_0, y1, coeff_256);

    let (y0, y1) = load_128b_block(&blocks[i + 1]);
    x0_1 = fold16_4x_ternlog(x0_1, y0, coeff_256);
    x1_1 = fold16_4x_ternlog(x1_1, y1, coeff_256);

    i = i.strict_add(2);
  }

  // Merge: A·s0 ⊕ s1 (A = shift by 128B) using VPTERNLOGD.
  let mut x0 = fold16_4x_ternlog(x0_0, x0_1, coeff_128);
  let mut x1 = fold16_4x_ternlog(x1_0, x1_1, coeff_128);

  // Odd tail (one remaining block).
  if even != blocks.len() {
    let (y0, y1) = load_128b_block(&blocks[even]);
    x0 = fold16_4x_ternlog(x0, y0, coeff_128);
    x1 = fold16_4x_ternlog(x1, y1, coeff_128);
  }

  finalize_vpclmul_state(x0, x1, consts)
}

#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn update_simd_vpclmul_4way(
  state: u64,
  blocks: &[[Simd; 8]],
  fold_512b: (u64, u64),
  combine: &[(u64, u64); 3],
  consts: &Crc64ClmulConstants,
) -> u64 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 4 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_vpclmul(state, first, rest, consts);
  }

  let aligned = (blocks.len() / 4) * 4;

  let (mut x0_0, mut x1_0) = load_128b_block(&blocks[0]);
  let (mut x0_1, mut x1_1) = load_128b_block(&blocks[1]);
  let (mut x0_2, mut x1_2) = load_128b_block(&blocks[2]);
  let (mut x0_3, mut x1_3) = load_128b_block(&blocks[3]);

  let crc_mask = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, state as i64);
  x0_0 = _mm512_xor_si512(x0_0, crc_mask);

  let coeff_512 = vpclmul_coeff(fold_512b);
  let coeff_128 = vpclmul_coeff(consts.fold_128b);
  let c384 = vpclmul_coeff(combine[0]);
  let c256 = vpclmul_coeff(combine[1]);
  let c128 = vpclmul_coeff(combine[2]);

  let mut i = 4;
  while i < aligned {
    let (y0, y1) = load_128b_block(&blocks[i]);
    x0_0 = fold16_4x_ternlog(x0_0, y0, coeff_512);
    x1_0 = fold16_4x_ternlog(x1_0, y1, coeff_512);

    let (y0, y1) = load_128b_block(&blocks[i + 1]);
    x0_1 = fold16_4x_ternlog(x0_1, y0, coeff_512);
    x1_1 = fold16_4x_ternlog(x1_1, y1, coeff_512);

    let (y0, y1) = load_128b_block(&blocks[i + 2]);
    x0_2 = fold16_4x_ternlog(x0_2, y0, coeff_512);
    x1_2 = fold16_4x_ternlog(x1_2, y1, coeff_512);

    let (y0, y1) = load_128b_block(&blocks[i + 3]);
    x0_3 = fold16_4x_ternlog(x0_3, y0, coeff_512);
    x1_3 = fold16_4x_ternlog(x1_3, y1, coeff_512);

    i = i.strict_add(4);
  }

  // Merge: A^3·s0 ⊕ A^2·s1 ⊕ A·s2 ⊕ s3 using VPTERNLOGD.
  let mut x0 = fold16_4x_ternlog(x0_2, x0_3, c128);
  let mut x1 = fold16_4x_ternlog(x1_2, x1_3, c128);
  x0 = fold16_4x_ternlog(x0_1, x0, c256);
  x1 = fold16_4x_ternlog(x1_1, x1, c256);
  x0 = fold16_4x_ternlog(x0_0, x0, c384);
  x1 = fold16_4x_ternlog(x1_0, x1, c384);

  // Remaining blocks processed sequentially.
  for block in &blocks[aligned..] {
    let (y0, y1) = load_128b_block(block);
    x0 = fold16_4x_ternlog(x0, y0, coeff_128);
    x1 = fold16_4x_ternlog(x1, y1, coeff_128);
  }

  finalize_vpclmul_state(x0, x1, consts)
}

#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn update_simd_vpclmul_7way(
  state: u64,
  blocks: &[[Simd; 8]],
  fold_896b: (u64, u64),
  combine: &[(u64, u64); 6],
  consts: &Crc64ClmulConstants,
) -> u64 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 7 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_vpclmul(state, first, rest, consts);
  }

  let aligned = (blocks.len() / 7) * 7;

  let (mut x0_0, mut x1_0) = load_128b_block(&blocks[0]);
  let (mut x0_1, mut x1_1) = load_128b_block(&blocks[1]);
  let (mut x0_2, mut x1_2) = load_128b_block(&blocks[2]);
  let (mut x0_3, mut x1_3) = load_128b_block(&blocks[3]);
  let (mut x0_4, mut x1_4) = load_128b_block(&blocks[4]);
  let (mut x0_5, mut x1_5) = load_128b_block(&blocks[5]);
  let (mut x0_6, mut x1_6) = load_128b_block(&blocks[6]);

  let crc_mask = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, state as i64);
  x0_0 = _mm512_xor_si512(x0_0, crc_mask);

  let coeff_896 = vpclmul_coeff(fold_896b);
  let coeff_128 = vpclmul_coeff(consts.fold_128b);

  let c768 = vpclmul_coeff(combine[0]);
  let c640 = vpclmul_coeff(combine[1]);
  let c512 = vpclmul_coeff(combine[2]);
  let c384 = vpclmul_coeff(combine[3]);
  let c256 = vpclmul_coeff(combine[4]);
  let c128 = vpclmul_coeff(combine[5]);

  let mut i = 7;
  while i < aligned {
    // VPTERNLOGD: fold + XOR in one instruction per vector (7×2 = 14 per iteration)
    let (y0, y1) = load_128b_block(&blocks[i]);
    x0_0 = fold16_4x_ternlog(x0_0, y0, coeff_896);
    x1_0 = fold16_4x_ternlog(x1_0, y1, coeff_896);

    let (y0, y1) = load_128b_block(&blocks[i + 1]);
    x0_1 = fold16_4x_ternlog(x0_1, y0, coeff_896);
    x1_1 = fold16_4x_ternlog(x1_1, y1, coeff_896);

    let (y0, y1) = load_128b_block(&blocks[i + 2]);
    x0_2 = fold16_4x_ternlog(x0_2, y0, coeff_896);
    x1_2 = fold16_4x_ternlog(x1_2, y1, coeff_896);

    let (y0, y1) = load_128b_block(&blocks[i + 3]);
    x0_3 = fold16_4x_ternlog(x0_3, y0, coeff_896);
    x1_3 = fold16_4x_ternlog(x1_3, y1, coeff_896);

    let (y0, y1) = load_128b_block(&blocks[i + 4]);
    x0_4 = fold16_4x_ternlog(x0_4, y0, coeff_896);
    x1_4 = fold16_4x_ternlog(x1_4, y1, coeff_896);

    let (y0, y1) = load_128b_block(&blocks[i + 5]);
    x0_5 = fold16_4x_ternlog(x0_5, y0, coeff_896);
    x1_5 = fold16_4x_ternlog(x1_5, y1, coeff_896);

    let (y0, y1) = load_128b_block(&blocks[i + 6]);
    x0_6 = fold16_4x_ternlog(x0_6, y0, coeff_896);
    x1_6 = fold16_4x_ternlog(x1_6, y1, coeff_896);

    i = i.strict_add(7);
  }

  // Merge: A^6·s0 ⊕ A^5·s1 ⊕ … ⊕ A·s5 ⊕ s6 using VPTERNLOGD.
  let mut x0 = fold16_4x_ternlog(x0_5, x0_6, c128);
  let mut x1 = fold16_4x_ternlog(x1_5, x1_6, c128);
  x0 = fold16_4x_ternlog(x0_4, x0, c256);
  x1 = fold16_4x_ternlog(x1_4, x1, c256);
  x0 = fold16_4x_ternlog(x0_3, x0, c384);
  x1 = fold16_4x_ternlog(x1_3, x1, c384);
  x0 = fold16_4x_ternlog(x0_2, x0, c512);
  x1 = fold16_4x_ternlog(x1_2, x1, c512);
  x0 = fold16_4x_ternlog(x0_1, x0, c640);
  x1 = fold16_4x_ternlog(x1_1, x1, c640);
  x0 = fold16_4x_ternlog(x0_0, x0, c768);
  x1 = fold16_4x_ternlog(x1_0, x1, c768);

  for block in &blocks[aligned..] {
    let (y0, y1) = load_128b_block(block);
    x0 = fold16_4x_ternlog(x0, y0, coeff_128);
    x1 = fold16_4x_ternlog(x1, y1, coeff_128);
  }

  finalize_vpclmul_state(x0, x1, consts)
}

#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn update_simd_vpclmul_8way(
  state: u64,
  blocks: &[[Simd; 8]],
  fold_1024b: (u64, u64),
  combine: &[(u64, u64); 7],
  consts: &Crc64ClmulConstants,
) -> u64 {
  debug_assert!(!blocks.is_empty());

  if blocks.len() < 8 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_vpclmul(state, first, rest, consts);
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

  let crc_mask = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, state as i64);
  x0_0 = _mm512_xor_si512(x0_0, crc_mask);

  let coeff_1024 = vpclmul_coeff(fold_1024b);
  let coeff_128 = vpclmul_coeff(consts.fold_128b);

  let c896 = vpclmul_coeff(combine[0]);
  let c768 = vpclmul_coeff(combine[1]);
  let c640 = vpclmul_coeff(combine[2]);
  let c512 = vpclmul_coeff(combine[3]);
  let c384 = vpclmul_coeff(combine[4]);
  let c256 = vpclmul_coeff(combine[5]);
  let c128 = vpclmul_coeff(combine[6]);

  let mut i = 8;
  while i < aligned {
    // VPTERNLOGD: fold + XOR in one instruction per vector (8×2 = 16 per iteration)
    let (y0, y1) = load_128b_block(&blocks[i]);
    x0_0 = fold16_4x_ternlog(x0_0, y0, coeff_1024);
    x1_0 = fold16_4x_ternlog(x1_0, y1, coeff_1024);

    let (y0, y1) = load_128b_block(&blocks[i + 1]);
    x0_1 = fold16_4x_ternlog(x0_1, y0, coeff_1024);
    x1_1 = fold16_4x_ternlog(x1_1, y1, coeff_1024);

    let (y0, y1) = load_128b_block(&blocks[i + 2]);
    x0_2 = fold16_4x_ternlog(x0_2, y0, coeff_1024);
    x1_2 = fold16_4x_ternlog(x1_2, y1, coeff_1024);

    let (y0, y1) = load_128b_block(&blocks[i + 3]);
    x0_3 = fold16_4x_ternlog(x0_3, y0, coeff_1024);
    x1_3 = fold16_4x_ternlog(x1_3, y1, coeff_1024);

    let (y0, y1) = load_128b_block(&blocks[i + 4]);
    x0_4 = fold16_4x_ternlog(x0_4, y0, coeff_1024);
    x1_4 = fold16_4x_ternlog(x1_4, y1, coeff_1024);

    let (y0, y1) = load_128b_block(&blocks[i + 5]);
    x0_5 = fold16_4x_ternlog(x0_5, y0, coeff_1024);
    x1_5 = fold16_4x_ternlog(x1_5, y1, coeff_1024);

    let (y0, y1) = load_128b_block(&blocks[i + 6]);
    x0_6 = fold16_4x_ternlog(x0_6, y0, coeff_1024);
    x1_6 = fold16_4x_ternlog(x1_6, y1, coeff_1024);

    let (y0, y1) = load_128b_block(&blocks[i + 7]);
    x0_7 = fold16_4x_ternlog(x0_7, y0, coeff_1024);
    x1_7 = fold16_4x_ternlog(x1_7, y1, coeff_1024);

    i = i.strict_add(8);
  }

  // Merge: A^7·s0 ⊕ A^6·s1 ⊕ … ⊕ A·s6 ⊕ s7 using VPTERNLOGD.
  let mut x0 = fold16_4x_ternlog(x0_6, x0_7, c128);
  let mut x1 = fold16_4x_ternlog(x1_6, x1_7, c128);
  x0 = fold16_4x_ternlog(x0_5, x0, c256);
  x1 = fold16_4x_ternlog(x1_5, x1, c256);
  x0 = fold16_4x_ternlog(x0_4, x0, c384);
  x1 = fold16_4x_ternlog(x1_4, x1, c384);
  x0 = fold16_4x_ternlog(x0_3, x0, c512);
  x1 = fold16_4x_ternlog(x1_3, x1, c512);
  x0 = fold16_4x_ternlog(x0_2, x0, c640);
  x1 = fold16_4x_ternlog(x1_2, x1, c640);
  x0 = fold16_4x_ternlog(x0_1, x0, c768);
  x1 = fold16_4x_ternlog(x1_1, x1, c768);
  x0 = fold16_4x_ternlog(x0_0, x0, c896);
  x1 = fold16_4x_ternlog(x1_0, x1, c896);

  for block in &blocks[aligned..] {
    let (y0, y1) = load_128b_block(block);
    x0 = fold16_4x_ternlog(x0, y0, coeff_128);
    x1 = fold16_4x_ternlog(x1, y1, coeff_128);
  }

  finalize_vpclmul_state(x0, x1, consts)
}

#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn crc64_vpclmul_2way(
  mut state: u64,
  bytes: &[u8],
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 8],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return super::portable::crc64_slice8(state, bytes, tables);
  }

  state = super::portable::crc64_slice8(state, left, tables);
  state = update_simd_vpclmul_2way(state, middle, fold_256b, consts);
  super::portable::crc64_slice8(state, right, tables)
}

#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn crc64_vpclmul_4way(
  mut state: u64,
  bytes: &[u8],
  fold_512b: (u64, u64),
  combine: &[(u64, u64); 3],
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 8],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return super::portable::crc64_slice8(state, bytes, tables);
  }

  state = super::portable::crc64_slice8(state, left, tables);
  state = update_simd_vpclmul_4way(state, middle, fold_512b, combine, consts);
  super::portable::crc64_slice8(state, right, tables)
}

#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn crc64_vpclmul_7way(
  mut state: u64,
  bytes: &[u8],
  fold_896b: (u64, u64),
  combine: &[(u64, u64); 6],
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 8],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return super::portable::crc64_slice8(state, bytes, tables);
  }

  state = super::portable::crc64_slice8(state, left, tables);
  state = update_simd_vpclmul_7way(state, middle, fold_896b, combine, consts);
  super::portable::crc64_slice8(state, right, tables)
}

#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
unsafe fn crc64_vpclmul_8way(
  mut state: u64,
  bytes: &[u8],
  fold_1024b: (u64, u64),
  combine: &[(u64, u64); 7],
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 8],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return super::portable::crc64_slice8(state, bytes, tables);
  }

  state = super::portable::crc64_slice8(state, left, tables);
  state = update_simd_vpclmul_8way(state, middle, fold_1024b, combine, consts);
  super::portable::crc64_slice8(state, right, tables)
}

/// CRC-64-XZ using PCLMULQDQ folding.
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc64_xz_pclmul(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pclmul(
      crc,
      data,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-XZ using PCLMULQDQ (small-buffer lane folding).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc64_xz_pclmul_small(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pclmul_small(
      crc,
      data,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-XZ using PCLMULQDQ folding (2-way ILP variant).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc64_xz_pclmul_2way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pclmul_2way(
      crc,
      data,
      CRC64_XZ_STREAM.fold_256b,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-XZ using PCLMULQDQ folding (4-way ILP variant).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc64_xz_pclmul_4way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pclmul_4way(
      crc,
      data,
      CRC64_XZ_STREAM.fold_512b,
      &CRC64_XZ_STREAM.combine_4way,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-XZ using PCLMULQDQ folding (7-way ILP variant).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc64_xz_pclmul_7way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pclmul_7way(
      crc,
      data,
      CRC64_XZ_STREAM.fold_896b,
      &CRC64_XZ_STREAM.combine_7way,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-XZ using PCLMULQDQ folding (8-way ILP variant).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc64_xz_pclmul_8way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pclmul_8way(
      crc,
      data,
      CRC64_XZ_STREAM.fold_1024b,
      &CRC64_XZ_STREAM.combine_8way,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-XZ using VPCLMULQDQ (AVX-512) folding.
///
/// # Safety
///
/// Requires VPCLMULQDQ + AVX-512. Caller must verify via
/// `platform::caps().has(x86::VPCLMUL_READY)`.
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
pub unsafe fn crc64_xz_vpclmul(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_vpclmul(
      crc,
      data,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-XZ using VPCLMULQDQ (2-way ILP variant).
///
/// # Safety
///
/// Requires VPCLMULQDQ + AVX-512. Caller must verify via
/// `platform::caps().has(x86::VPCLMUL_READY)`.
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
pub unsafe fn crc64_xz_vpclmul_2way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_vpclmul_2way(
      crc,
      data,
      CRC64_XZ_STREAM.fold_256b,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-XZ using VPCLMULQDQ (4-way ILP variant).
///
/// # Safety
///
/// Requires VPCLMULQDQ + AVX-512. Caller must verify via
/// `platform::caps().has(x86::VPCLMUL_READY)`.
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
pub unsafe fn crc64_xz_vpclmul_4way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_vpclmul_4way(
      crc,
      data,
      CRC64_XZ_STREAM.fold_512b,
      &CRC64_XZ_STREAM.combine_4way,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-XZ using VPCLMULQDQ (7-way ILP variant).
///
/// # Safety
///
/// Requires VPCLMULQDQ + AVX-512. Caller must verify via
/// `platform::caps().has(x86::VPCLMUL_READY)`.
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
pub unsafe fn crc64_xz_vpclmul_7way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_vpclmul_7way(
      crc,
      data,
      CRC64_XZ_STREAM.fold_896b,
      &CRC64_XZ_STREAM.combine_7way,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-XZ using VPCLMULQDQ (8-way ILP variant).
///
/// # Safety
///
/// Requires VPCLMULQDQ + AVX-512. Caller must verify via
/// `platform::caps().has(x86::VPCLMUL_READY)`.
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
pub unsafe fn crc64_xz_vpclmul_8way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_vpclmul_8way(
      crc,
      data,
      CRC64_XZ_STREAM.fold_1024b,
      &CRC64_XZ_STREAM.combine_8way,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-NVME using PCLMULQDQ folding.
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc64_nvme_pclmul(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pclmul(
      crc,
      data,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

/// CRC-64-NVME using PCLMULQDQ (small-buffer lane folding).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc64_nvme_pclmul_small(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pclmul_small(
      crc,
      data,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

/// CRC-64-NVME using PCLMULQDQ folding (2-way ILP variant).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc64_nvme_pclmul_2way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pclmul_2way(
      crc,
      data,
      CRC64_NVME_STREAM.fold_256b,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

/// CRC-64-NVME using PCLMULQDQ folding (4-way ILP variant).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc64_nvme_pclmul_4way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pclmul_4way(
      crc,
      data,
      CRC64_NVME_STREAM.fold_512b,
      &CRC64_NVME_STREAM.combine_4way,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

/// CRC-64-NVME using PCLMULQDQ folding (7-way ILP variant).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc64_nvme_pclmul_7way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pclmul_7way(
      crc,
      data,
      CRC64_NVME_STREAM.fold_896b,
      &CRC64_NVME_STREAM.combine_7way,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

/// CRC-64-NVME using PCLMULQDQ folding (8-way ILP variant).
///
/// # Safety
///
/// Requires PCLMULQDQ. Caller must verify via `platform::caps().has(x86::PCLMUL_READY)`.
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
pub unsafe fn crc64_nvme_pclmul_8way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pclmul_8way(
      crc,
      data,
      CRC64_NVME_STREAM.fold_1024b,
      &CRC64_NVME_STREAM.combine_8way,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

/// CRC-64-NVME using VPCLMULQDQ (AVX-512) folding.
///
/// # Safety
///
/// Requires VPCLMULQDQ + AVX-512. Caller must verify via
/// `platform::caps().has(x86::VPCLMUL_READY)`.
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
pub unsafe fn crc64_nvme_vpclmul(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_vpclmul(
      crc,
      data,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

/// CRC-64-NVME using VPCLMULQDQ (2-way ILP variant).
///
/// # Safety
///
/// Requires VPCLMULQDQ + AVX-512. Caller must verify via
/// `platform::caps().has(x86::VPCLMUL_READY)`.
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
pub unsafe fn crc64_nvme_vpclmul_2way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_vpclmul_2way(
      crc,
      data,
      CRC64_NVME_STREAM.fold_256b,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

/// CRC-64-NVME using VPCLMULQDQ (4-way ILP variant).
///
/// # Safety
///
/// Requires VPCLMULQDQ + AVX-512. Caller must verify via
/// `platform::caps().has(x86::VPCLMUL_READY)`.
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
pub unsafe fn crc64_nvme_vpclmul_4way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_vpclmul_4way(
      crc,
      data,
      CRC64_NVME_STREAM.fold_512b,
      &CRC64_NVME_STREAM.combine_4way,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

/// CRC-64-NVME using VPCLMULQDQ (7-way ILP variant).
///
/// # Safety
///
/// Requires VPCLMULQDQ + AVX-512. Caller must verify via
/// `platform::caps().has(x86::VPCLMUL_READY)`.
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
pub unsafe fn crc64_nvme_vpclmul_7way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_vpclmul_7way(
      crc,
      data,
      CRC64_NVME_STREAM.fold_896b,
      &CRC64_NVME_STREAM.combine_7way,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

/// CRC-64-NVME using VPCLMULQDQ (8-way ILP variant).
///
/// # Safety
///
/// Requires VPCLMULQDQ + AVX-512. Caller must verify via
/// `platform::caps().has(x86::VPCLMUL_READY)`.
#[target_feature(enable = "avx512f", enable = "vpclmulqdq")]
pub unsafe fn crc64_nvme_vpclmul_8way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_vpclmul_8way(
      crc,
      data,
      CRC64_NVME_STREAM.fold_1024b,
      &CRC64_NVME_STREAM.combine_8way,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Wrappers (safe interface)
// ─────────────────────────────────────────────────────────────────────────────

/// Safe wrapper for CRC-64-XZ PCLMUL kernel.
#[inline]
pub fn crc64_xz_pclmul_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel.
  unsafe { crc64_xz_pclmul(crc, data) }
}

/// Safe wrapper for CRC-64-XZ PCLMUL small-buffer kernel.
#[inline]
pub fn crc64_xz_pclmul_small_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel.
  unsafe { crc64_xz_pclmul_small(crc, data) }
}

/// Safe wrapper for CRC-64-XZ PCLMUL 2-way kernel.
#[inline]
pub fn crc64_xz_pclmul_2way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel.
  unsafe { crc64_xz_pclmul_2way(crc, data) }
}

/// Safe wrapper for CRC-64-XZ PCLMUL 4-way kernel.
#[inline]
pub fn crc64_xz_pclmul_4way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel.
  unsafe { crc64_xz_pclmul_4way(crc, data) }
}

/// Safe wrapper for CRC-64-XZ PCLMUL 7-way kernel.
#[inline]
pub fn crc64_xz_pclmul_7way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel.
  unsafe { crc64_xz_pclmul_7way(crc, data) }
}

/// Safe wrapper for CRC-64-XZ PCLMUL 8-way kernel.
#[inline]
pub fn crc64_xz_pclmul_8way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel.
  unsafe { crc64_xz_pclmul_8way(crc, data) }
}

/// Safe wrapper for CRC-64-XZ VPCLMUL kernel.
#[inline]
pub fn crc64_xz_vpclmul_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies VPCLMULQDQ + AVX-512 before selecting this kernel.
  unsafe { crc64_xz_vpclmul(crc, data) }
}

/// Safe wrapper for CRC-64-XZ VPCLMUL 2-way kernel.
#[inline]
pub fn crc64_xz_vpclmul_2way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Callers must verify VPCLMUL_READY before selecting this kernel.
  unsafe { crc64_xz_vpclmul_2way(crc, data) }
}

/// Safe wrapper for CRC-64-XZ VPCLMUL 4-way kernel.
#[inline]
pub fn crc64_xz_vpclmul_4way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Callers must verify VPCLMUL_READY before selecting this kernel.
  unsafe { crc64_xz_vpclmul_4way(crc, data) }
}

/// Safe wrapper for CRC-64-XZ VPCLMUL 7-way kernel.
#[inline]
pub fn crc64_xz_vpclmul_7way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Callers must verify VPCLMUL_READY before selecting this kernel.
  unsafe { crc64_xz_vpclmul_7way(crc, data) }
}

/// Safe wrapper for CRC-64-XZ VPCLMUL 8-way kernel.
#[inline]
pub fn crc64_xz_vpclmul_8way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Callers must verify VPCLMUL_READY before selecting this kernel.
  unsafe { crc64_xz_vpclmul_8way(crc, data) }
}

/// Safe wrapper for CRC-64-NVME PCLMUL kernel.
#[inline]
pub fn crc64_nvme_pclmul_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel.
  unsafe { crc64_nvme_pclmul(crc, data) }
}

/// Safe wrapper for CRC-64-NVME PCLMUL small-buffer kernel.
#[inline]
pub fn crc64_nvme_pclmul_small_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel.
  unsafe { crc64_nvme_pclmul_small(crc, data) }
}

/// Safe wrapper for CRC-64-NVME PCLMUL 2-way kernel.
#[inline]
pub fn crc64_nvme_pclmul_2way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel.
  unsafe { crc64_nvme_pclmul_2way(crc, data) }
}

/// Safe wrapper for CRC-64-NVME PCLMUL 4-way kernel.
#[inline]
pub fn crc64_nvme_pclmul_4way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel.
  unsafe { crc64_nvme_pclmul_4way(crc, data) }
}

/// Safe wrapper for CRC-64-NVME PCLMUL 7-way kernel.
#[inline]
pub fn crc64_nvme_pclmul_7way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel.
  unsafe { crc64_nvme_pclmul_7way(crc, data) }
}

/// Safe wrapper for CRC-64-NVME PCLMUL 8-way kernel.
#[inline]
pub fn crc64_nvme_pclmul_8way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PCLMULQDQ before selecting this kernel.
  unsafe { crc64_nvme_pclmul_8way(crc, data) }
}

/// Safe wrapper for CRC-64-NVME VPCLMUL kernel.
#[inline]
pub fn crc64_nvme_vpclmul_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies VPCLMULQDQ + AVX-512 before selecting this kernel.
  unsafe { crc64_nvme_vpclmul(crc, data) }
}

/// Safe wrapper for CRC-64-NVME VPCLMUL 2-way kernel.
#[inline]
pub fn crc64_nvme_vpclmul_2way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Callers must verify VPCLMUL_READY before selecting this kernel.
  unsafe { crc64_nvme_vpclmul_2way(crc, data) }
}

/// Safe wrapper for CRC-64-NVME VPCLMUL 4-way kernel.
#[inline]
pub fn crc64_nvme_vpclmul_4way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Callers must verify VPCLMUL_READY before selecting this kernel.
  unsafe { crc64_nvme_vpclmul_4way(crc, data) }
}

/// Safe wrapper for CRC-64-NVME VPCLMUL 7-way kernel.
#[inline]
pub fn crc64_nvme_vpclmul_7way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Callers must verify VPCLMUL_READY before selecting this kernel.
  unsafe { crc64_nvme_vpclmul_7way(crc, data) }
}

/// Safe wrapper for CRC-64-NVME VPCLMUL 8-way kernel.
#[inline]
pub fn crc64_nvme_vpclmul_8way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Callers must verify VPCLMUL_READY before selecting this kernel.
  unsafe { crc64_nvme_vpclmul_8way(crc, data) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// Tests require SIMD intrinsics that Miri cannot interpret.
#[cfg(all(test, not(miri)))]
mod tests {
  extern crate alloc;
  extern crate std;

  use alloc::vec::Vec;

  use super::*;

  fn make_data(len: usize) -> Vec<u8> {
    (0..len)
      .map(|i| (i as u8).wrapping_mul(17).wrapping_add((i >> 3) as u8))
      .collect()
  }

  #[test]
  fn test_crc64_xz_pclmul_matches_vector() {
    if !std::arch::is_x86_feature_detected!("pclmulqdq") {
      return;
    }
    let crc = crc64_xz_pclmul_safe(!0, b"123456789") ^ !0;
    assert_eq!(crc, 0x995D_C9BB_DF19_39FA);
  }

  #[test]
  fn test_crc64_nvme_pclmul_matches_vector() {
    if !std::arch::is_x86_feature_detected!("pclmulqdq") {
      return;
    }
    let crc = crc64_nvme_pclmul_safe(!0, b"123456789") ^ !0;
    assert_eq!(crc, 0xAE8B_1486_0A79_9888);
  }

  #[test]
  fn test_crc64_xz_pclmul_matches_portable_various_lengths() {
    if !std::arch::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    for len in [
      0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 129, 255, 256, 511, 512, 1024,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      let pclmul = crc64_xz_pclmul_safe(!0, &data) ^ !0;
      assert_eq!(pclmul, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_pclmul_matches_portable_various_lengths() {
    if !std::arch::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    for len in [
      0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 129, 255, 256, 511, 512, 1024,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      let pclmul = crc64_nvme_pclmul_safe(!0, &data) ^ !0;
      assert_eq!(pclmul, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_xz_pclmul_small_matches_portable_all_lengths_0_127() {
    if !std::arch::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    for len in 0..128 {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      let pclmul_small = crc64_xz_pclmul_small_safe(!0, &data) ^ !0;
      assert_eq!(pclmul_small, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_pclmul_small_matches_portable_all_lengths_0_127() {
    if !std::arch::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    for len in 0..128 {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      let pclmul_small = crc64_nvme_pclmul_small_safe(!0, &data) ^ !0;
      assert_eq!(pclmul_small, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_xz_pclmul_2way_matches_portable_various_lengths() {
    if !std::arch::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    for len in [0usize, 1, 7, 16, 63, 64, 127, 128, 255, 256, 1024, 4096, 16 * 1024] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      let pclmul_2way = crc64_xz_pclmul_2way_safe(!0, &data) ^ !0;
      assert_eq!(pclmul_2way, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_pclmul_2way_matches_portable_various_lengths() {
    if !std::arch::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    for len in [0usize, 1, 7, 16, 63, 64, 127, 128, 255, 256, 1024, 4096, 16 * 1024] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      let pclmul_2way = crc64_nvme_pclmul_2way_safe(!0, &data) ^ !0;
      assert_eq!(pclmul_2way, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_xz_pclmul_multiway_matches_portable_various_lengths() {
    if !std::arch::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    for len in [0usize, 1, 7, 16, 63, 64, 127, 128, 255, 256, 512, 1024, 4096, 16 * 1024] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      let vp2 = crc64_xz_pclmul_2way_safe(!0, &data) ^ !0;
      let vp4 = crc64_xz_pclmul_4way_safe(!0, &data) ^ !0;
      let vp7 = crc64_xz_pclmul_7way_safe(!0, &data) ^ !0;
      assert_eq!(vp2, portable, "2-way mismatch at len={len}");
      assert_eq!(vp4, portable, "4-way mismatch at len={len}");
      assert_eq!(vp7, portable, "7-way mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_pclmul_multiway_matches_portable_various_lengths() {
    if !std::arch::is_x86_feature_detected!("pclmulqdq") {
      return;
    }

    for len in [0usize, 1, 7, 16, 63, 64, 127, 128, 255, 256, 512, 1024, 4096, 16 * 1024] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      let vp2 = crc64_nvme_pclmul_2way_safe(!0, &data) ^ !0;
      let vp4 = crc64_nvme_pclmul_4way_safe(!0, &data) ^ !0;
      let vp7 = crc64_nvme_pclmul_7way_safe(!0, &data) ^ !0;
      assert_eq!(vp2, portable, "2-way mismatch at len={len}");
      assert_eq!(vp4, portable, "4-way mismatch at len={len}");
      assert_eq!(vp7, portable, "7-way mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_xz_vpclmul_multiway_matches_portable_various_lengths() {
    if !(std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("vpclmulqdq")) {
      return;
    }

    for len in [0usize, 1, 7, 16, 63, 64, 127, 128, 255, 256, 512, 1024, 4096, 16 * 1024] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      let vp2 = crc64_xz_vpclmul_2way_safe(!0, &data) ^ !0;
      let vp4 = crc64_xz_vpclmul_4way_safe(!0, &data) ^ !0;
      let vp7 = crc64_xz_vpclmul_7way_safe(!0, &data) ^ !0;
      assert_eq!(vp2, portable, "2-way mismatch at len={len}");
      assert_eq!(vp4, portable, "4-way mismatch at len={len}");
      assert_eq!(vp7, portable, "7-way mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_vpclmul_multiway_matches_portable_various_lengths() {
    if !(std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("vpclmulqdq")) {
      return;
    }

    for len in [0usize, 1, 7, 16, 63, 64, 127, 128, 255, 256, 512, 1024, 4096, 16 * 1024] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      let vp2 = crc64_nvme_vpclmul_2way_safe(!0, &data) ^ !0;
      let vp4 = crc64_nvme_vpclmul_4way_safe(!0, &data) ^ !0;
      let vp7 = crc64_nvme_vpclmul_7way_safe(!0, &data) ^ !0;
      assert_eq!(vp2, portable, "2-way mismatch at len={len}");
      assert_eq!(vp4, portable, "4-way mismatch at len={len}");
      assert_eq!(vp7, portable, "7-way mismatch at len={len}");
    }
  }
}
