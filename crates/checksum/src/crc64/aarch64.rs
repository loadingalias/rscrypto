//! aarch64 hardware-accelerated CRC-64 kernels (XZ + NVME).
//!
//! This is a PMULL implementation derived from the Intel/TiKV folding
//! algorithm (also used by `crc64fast` / `crc64fast-nvme`).
//!
//! # Safety
//!
//! Uses `unsafe` for ARM SIMD intrinsics. Callers must ensure PMULL is
//! available before executing the accelerated path (the dispatcher does this).
#![allow(unsafe_code)]
#![allow(dead_code)] // Kernels wired up via dispatcher
// SAFETY: All indexing is over fixed-size arrays with in-bounds constant indices.
#![allow(clippy::indexing_slicing)]
// This module is intrinsics-heavy; keep unsafe blocks readable.
#![allow(unsafe_op_in_unsafe_fn)]

use core::{
  arch::aarch64::*,
  ops::{BitXor, BitXorAssign},
};

use crate::common::clmul::{CRC64_NVME_STREAM, CRC64_XZ_STREAM, Crc64ClmulConstants};

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
struct Simd(uint8x16_t);

#[allow(non_camel_case_types)]
type poly64_t = u64;

impl Simd {
  #[inline]
  #[target_feature(enable = "neon", enable = "aes")]
  unsafe fn from_mul(a: poly64_t, b: poly64_t) -> Self {
    let mul = vmull_p64(a, b);
    Self(vreinterpretq_u8_p128(mul))
  }

  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn into_poly64s(self) -> [poly64_t; 2] {
    let x = vreinterpretq_p64_u8(self.0);
    [vgetq_lane_p64(x, 0), vgetq_lane_p64(x, 1)]
  }

  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn high_64(self) -> poly64_t {
    let x = vreinterpretq_p64_u8(self.0);
    vgetq_lane_p64(x, 1)
  }

  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn low_64(self) -> poly64_t {
    let x = vreinterpretq_p64_u8(self.0);
    vgetq_lane_p64(x, 0)
  }

  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn new(high: u64, low: u64) -> Self {
    Self(vcombine_u8(vcreate_u8(low), vcreate_u8(high)))
  }

  /// Fold 16 bytes: `(coeff.low ⊗ self.low) ⊕ (coeff.high ⊗ self.high)`.
  #[inline]
  #[target_feature(enable = "neon", enable = "aes")]
  unsafe fn fold_16(self, coeff: Self) -> Self {
    let [x0, x1] = self.into_poly64s();
    let [c0, c1] = coeff.into_poly64s();
    let h = Self::from_mul(c0, x0);
    let l = Self::from_mul(c1, x1);
    h ^ l
  }

  /// Fold 16 bytes using pre-extracted coefficient halves (low, high).
  ///
  /// Equivalent to `self.fold_16(Simd::new(high, low))` but avoids repeatedly
  /// extracting the coefficient lanes inside hot loops.
  #[inline]
  #[target_feature(enable = "neon", enable = "aes")]
  unsafe fn fold_16_pair(self, coeff_low: poly64_t, coeff_high: poly64_t) -> Self {
    let [x0, x1] = self.into_poly64s();
    let h = Self::from_mul(coeff_low, x0);
    let l = Self::from_mul(coeff_high, x1);
    h ^ l
  }

  /// Fold 8 bytes: `self.high ⊕ (coeff ⊗ self.low)`.
  #[inline]
  #[target_feature(enable = "neon", enable = "aes")]
  unsafe fn fold_8(self, coeff: u64) -> Self {
    let [x0, x1] = self.into_poly64s();
    let h = Self::from_mul(coeff, x0);
    let l = Self::new(0, x1);
    h ^ l
  }

  /// Barrett reduction to finalize the CRC.
  #[inline]
  #[target_feature(enable = "neon", enable = "aes")]
  unsafe fn barrett(self, poly: u64, mu: u64) -> u64 {
    let t1 = Self::from_mul(self.low_64(), mu).low_64();
    let l = Self::from_mul(t1, poly);
    let reduced = (self ^ l).high_64();
    reduced ^ t1
  }
}

impl BitXor for Simd {
  type Output = Self;

  #[inline]
  fn bitxor(self, other: Self) -> Self {
    // SAFETY: `veorq_u8` is available with NEON.
    unsafe { Self(veorq_u8(self.0, other.0)) }
  }
}

impl BitXorAssign for Simd {
  #[inline]
  fn bitxor_assign(&mut self, other: Self) {
    *self = *self ^ other;
  }
}

#[target_feature(enable = "aes", enable = "neon")]
unsafe fn update_simd(state: u64, first: &[Simd; 8], rest: &[[Simd; 8]], consts: &Crc64ClmulConstants) -> u64 {
  let mut x = *first;

  // XOR the initial CRC into the first lane.
  x[0] ^= Simd::new(0, state);

  // 128-byte folding.
  let coeff_low = consts.fold_128b.1;
  let coeff_high = consts.fold_128b.0;
  for chunk in rest {
    fold_block_128(&mut x, chunk, coeff_low, coeff_high);
  }

  fold_tail(x, consts)
}

#[target_feature(enable = "aes", enable = "neon", enable = "sha3")]
unsafe fn update_simd_eor3(state: u64, first: &[Simd; 8], rest: &[[Simd; 8]], consts: &Crc64ClmulConstants) -> u64 {
  let mut x = *first;

  // XOR the initial CRC into the first lane.
  x[0] ^= Simd::new(0, state);

  // 128-byte folding.
  let coeff_low = consts.fold_128b.1;
  let coeff_high = consts.fold_128b.0;
  for chunk in rest {
    fold_block_128_eor3(&mut x, chunk, coeff_low, coeff_high);
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
// PMULL multi-stream (2-way/3-way, 128B blocks)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn fold_block_128(x: &mut [Simd; 8], chunk: &[Simd; 8], coeff_low: u64, coeff_high: u64) {
  x[0] = chunk[0] ^ x[0].fold_16_pair(coeff_low, coeff_high);
  x[1] = chunk[1] ^ x[1].fold_16_pair(coeff_low, coeff_high);
  x[2] = chunk[2] ^ x[2].fold_16_pair(coeff_low, coeff_high);
  x[3] = chunk[3] ^ x[3].fold_16_pair(coeff_low, coeff_high);
  x[4] = chunk[4] ^ x[4].fold_16_pair(coeff_low, coeff_high);
  x[5] = chunk[5] ^ x[5].fold_16_pair(coeff_low, coeff_high);
  x[6] = chunk[6] ^ x[6].fold_16_pair(coeff_low, coeff_high);
  x[7] = chunk[7] ^ x[7].fold_16_pair(coeff_low, coeff_high);
}

/// Fold a 128-byte block using EOR3 (3-way XOR in one instruction).
///
/// This reduces the XOR dependency chain: instead of `chunk ^ (h ^ l)` (2 XORs),
/// we use `veor3(chunk, h, l)` (1 instruction).
#[inline]
#[target_feature(enable = "aes", enable = "neon", enable = "sha3")]
unsafe fn fold_block_128_eor3(x: &mut [Simd; 8], chunk: &[Simd; 8], coeff_low: u64, coeff_high: u64) {
  x[0] = fold_lane_eor3(x[0], chunk[0], coeff_low, coeff_high);
  x[1] = fold_lane_eor3(x[1], chunk[1], coeff_low, coeff_high);
  x[2] = fold_lane_eor3(x[2], chunk[2], coeff_low, coeff_high);
  x[3] = fold_lane_eor3(x[3], chunk[3], coeff_low, coeff_high);
  x[4] = fold_lane_eor3(x[4], chunk[4], coeff_low, coeff_high);
  x[5] = fold_lane_eor3(x[5], chunk[5], coeff_low, coeff_high);
  x[6] = fold_lane_eor3(x[6], chunk[6], coeff_low, coeff_high);
  x[7] = fold_lane_eor3(x[7], chunk[7], coeff_low, coeff_high);
}

/// Fold a single 16-byte lane using EOR3.
///
/// Computes: `data ^ pmull(coeff_low, x.low) ^ pmull(coeff_high, x.high)`
/// using a single 3-way XOR instruction.
#[inline]
#[target_feature(enable = "aes", enable = "neon", enable = "sha3")]
unsafe fn fold_lane_eor3(x: Simd, data: Simd, coeff_low: poly64_t, coeff_high: poly64_t) -> Simd {
  let [x0, x1] = x.into_poly64s();
  let h = Simd::from_mul(coeff_low, x0);
  let l = Simd::from_mul(coeff_high, x1);
  // EOR3: 3-way XOR in one instruction (ARMv8.2-SHA3)
  Simd(veor3q_u8(data.0, h.0, l.0))
}

#[target_feature(enable = "aes", enable = "neon")]
unsafe fn update_simd_2way(
  state: u64,
  blocks: &[[Simd; 8]],
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
) -> u64 {
  debug_assert!(blocks.len() >= 2);

  let coeff_256_low = fold_256b.1;
  let coeff_256_high = fold_256b.0;
  let coeff_128_low = consts.fold_128b.1;
  let coeff_128_high = consts.fold_128b.0;

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];

  // Inject CRC into stream 0 (block 0).
  s0[0] ^= Simd::new(0, state);

  // Process the largest even prefix with 2-way striping.
  let mut i = 2;
  let even = blocks.len() & !1usize;
  while i < even {
    fold_block_128(&mut s0, &blocks[i], coeff_256_low, coeff_256_high);
    fold_block_128(&mut s1, &blocks[i + 1], coeff_256_low, coeff_256_high);
    i = i.strict_add(2);
  }

  // Merge streams: A·s0 ⊕ s1 (A = shift by 128B).
  let mut combined = s1;
  combined[0] ^= s0[0].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[1] ^= s0[1].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[2] ^= s0[2].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[3] ^= s0[3].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[4] ^= s0[4].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[5] ^= s0[5].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[6] ^= s0[6].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[7] ^= s0[7].fold_16_pair(coeff_128_low, coeff_128_high);

  // Handle any remaining block (odd tail) sequentially.
  if even != blocks.len() {
    fold_block_128(&mut combined, &blocks[even], coeff_128_low, coeff_128_high);
  }

  fold_tail(combined, consts)
}

#[target_feature(enable = "aes", enable = "neon")]
unsafe fn update_simd_3way(
  state: u64,
  blocks: &[[Simd; 8]],
  fold_384b: (u64, u64),
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
) -> u64 {
  if blocks.len() < 3 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd(state, first, rest, consts);
  }

  let aligned = (blocks.len() / 3) * 3;

  let coeff_384_low = fold_384b.1;
  let coeff_384_high = fold_384b.0;
  let coeff_256_low = fold_256b.1;
  let coeff_256_high = fold_256b.0;
  let coeff_128_low = consts.fold_128b.1;
  let coeff_128_high = consts.fold_128b.0;

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];
  let mut s2 = blocks[2];

  // Inject CRC into stream 0 (block 0).
  s0[0] ^= Simd::new(0, state);

  // Process the largest multiple-of-3 prefix with 3-way striping.
  let mut i = 3;
  while i < aligned {
    fold_block_128(&mut s0, &blocks[i], coeff_384_low, coeff_384_high);
    fold_block_128(&mut s1, &blocks[i + 1], coeff_384_low, coeff_384_high);
    fold_block_128(&mut s2, &blocks[i + 2], coeff_384_low, coeff_384_high);
    i = i.strict_add(3);
  }

  // Merge: A^2·s0 ⊕ A·s1 ⊕ s2 (A = shift by 128B).
  let mut combined = s2;
  combined[0] ^= s1[0].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[1] ^= s1[1].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[2] ^= s1[2].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[3] ^= s1[3].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[4] ^= s1[4].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[5] ^= s1[5].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[6] ^= s1[6].fold_16_pair(coeff_128_low, coeff_128_high);
  combined[7] ^= s1[7].fold_16_pair(coeff_128_low, coeff_128_high);

  combined[0] ^= s0[0].fold_16_pair(coeff_256_low, coeff_256_high);
  combined[1] ^= s0[1].fold_16_pair(coeff_256_low, coeff_256_high);
  combined[2] ^= s0[2].fold_16_pair(coeff_256_low, coeff_256_high);
  combined[3] ^= s0[3].fold_16_pair(coeff_256_low, coeff_256_high);
  combined[4] ^= s0[4].fold_16_pair(coeff_256_low, coeff_256_high);
  combined[5] ^= s0[5].fold_16_pair(coeff_256_low, coeff_256_high);
  combined[6] ^= s0[6].fold_16_pair(coeff_256_low, coeff_256_high);
  combined[7] ^= s0[7].fold_16_pair(coeff_256_low, coeff_256_high);

  // Handle any remaining blocks sequentially.
  for block in &blocks[aligned..] {
    fold_block_128(&mut combined, block, coeff_128_low, coeff_128_high);
  }

  fold_tail(combined, consts)
}

// ─────────────────────────────────────────────────────────────────────────────
// PMULL+EOR3 Multi-Stream (2-way/3-way with EOR3, 128B blocks)
// ─────────────────────────────────────────────────────────────────────────────

/// 2-way striping with EOR3 folding.
///
/// Combines the ILP benefit of 2-way striping (major) with EOR3's reduced
/// XOR dependency chain (minor). This is the optimal path on Apple M1+ and
/// AWS Graviton3+ for large buffers.
#[target_feature(enable = "aes", enable = "neon", enable = "sha3")]
unsafe fn update_simd_eor3_2way(
  state: u64,
  blocks: &[[Simd; 8]],
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
) -> u64 {
  debug_assert!(blocks.len() >= 2);

  let coeff_256_low = fold_256b.1;
  let coeff_256_high = fold_256b.0;
  let coeff_128_low = consts.fold_128b.1;
  let coeff_128_high = consts.fold_128b.0;

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];

  // Inject CRC into stream 0 (block 0).
  s0[0] ^= Simd::new(0, state);

  // Process the largest even prefix with 2-way striping using EOR3.
  let mut i = 2;
  let even = blocks.len() & !1usize;
  while i < even {
    fold_block_128_eor3(&mut s0, &blocks[i], coeff_256_low, coeff_256_high);
    fold_block_128_eor3(&mut s1, &blocks[i + 1], coeff_256_low, coeff_256_high);
    i = i.strict_add(2);
  }

  // Merge streams using EOR3: combined = s1 ^ (A·s0) where A = shift by 128B.
  // fold_lane_eor3(x, data, coeff_low, coeff_high) = data ^ pmull(coeff_low, x.low) ^
  // pmull(coeff_high, x.high)
  let mut combined: [Simd; 8] = [
    fold_lane_eor3(s0[0], s1[0], coeff_128_low, coeff_128_high),
    fold_lane_eor3(s0[1], s1[1], coeff_128_low, coeff_128_high),
    fold_lane_eor3(s0[2], s1[2], coeff_128_low, coeff_128_high),
    fold_lane_eor3(s0[3], s1[3], coeff_128_low, coeff_128_high),
    fold_lane_eor3(s0[4], s1[4], coeff_128_low, coeff_128_high),
    fold_lane_eor3(s0[5], s1[5], coeff_128_low, coeff_128_high),
    fold_lane_eor3(s0[6], s1[6], coeff_128_low, coeff_128_high),
    fold_lane_eor3(s0[7], s1[7], coeff_128_low, coeff_128_high),
  ];

  // Handle any remaining block (odd tail) sequentially with EOR3.
  if even != blocks.len() {
    fold_block_128_eor3(&mut combined, &blocks[even], coeff_128_low, coeff_128_high);
  }

  fold_tail(combined, consts)
}

/// 3-way striping with EOR3 folding.
///
/// Combines the ILP benefit of 3-way striping (major) with EOR3's reduced
/// XOR dependency chain (minor). This is the optimal path on Apple M1+ and
/// AWS Graviton3+ for very large buffers (32KB+).
#[target_feature(enable = "aes", enable = "neon", enable = "sha3")]
unsafe fn update_simd_eor3_3way(
  state: u64,
  blocks: &[[Simd; 8]],
  fold_384b: (u64, u64),
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
) -> u64 {
  if blocks.len() < 3 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_eor3(state, first, rest, consts);
  }

  let aligned = (blocks.len() / 3) * 3;

  let coeff_384_low = fold_384b.1;
  let coeff_384_high = fold_384b.0;
  let coeff_256_low = fold_256b.1;
  let coeff_256_high = fold_256b.0;
  let coeff_128_low = consts.fold_128b.1;
  let coeff_128_high = consts.fold_128b.0;

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];
  let mut s2 = blocks[2];

  // Inject CRC into stream 0 (block 0).
  s0[0] ^= Simd::new(0, state);

  // Process the largest multiple-of-3 prefix with 3-way striping using EOR3.
  let mut i = 3;
  while i < aligned {
    fold_block_128_eor3(&mut s0, &blocks[i], coeff_384_low, coeff_384_high);
    fold_block_128_eor3(&mut s1, &blocks[i + 1], coeff_384_low, coeff_384_high);
    fold_block_128_eor3(&mut s2, &blocks[i + 2], coeff_384_low, coeff_384_high);
    i = i.strict_add(3);
  }

  // Merge: A^2·s0 ⊕ A·s1 ⊕ s2 (A = shift by 128B).
  // First: combined = s2 ^ (A·s1)
  let mut combined: [Simd; 8] = [
    fold_lane_eor3(s1[0], s2[0], coeff_128_low, coeff_128_high),
    fold_lane_eor3(s1[1], s2[1], coeff_128_low, coeff_128_high),
    fold_lane_eor3(s1[2], s2[2], coeff_128_low, coeff_128_high),
    fold_lane_eor3(s1[3], s2[3], coeff_128_low, coeff_128_high),
    fold_lane_eor3(s1[4], s2[4], coeff_128_low, coeff_128_high),
    fold_lane_eor3(s1[5], s2[5], coeff_128_low, coeff_128_high),
    fold_lane_eor3(s1[6], s2[6], coeff_128_low, coeff_128_high),
    fold_lane_eor3(s1[7], s2[7], coeff_128_low, coeff_128_high),
  ];

  // Then: combined ^= (A^2·s0)
  combined[0] = fold_lane_eor3(s0[0], combined[0], coeff_256_low, coeff_256_high);
  combined[1] = fold_lane_eor3(s0[1], combined[1], coeff_256_low, coeff_256_high);
  combined[2] = fold_lane_eor3(s0[2], combined[2], coeff_256_low, coeff_256_high);
  combined[3] = fold_lane_eor3(s0[3], combined[3], coeff_256_low, coeff_256_high);
  combined[4] = fold_lane_eor3(s0[4], combined[4], coeff_256_low, coeff_256_high);
  combined[5] = fold_lane_eor3(s0[5], combined[5], coeff_256_low, coeff_256_high);
  combined[6] = fold_lane_eor3(s0[6], combined[6], coeff_256_low, coeff_256_high);
  combined[7] = fold_lane_eor3(s0[7], combined[7], coeff_256_low, coeff_256_high);

  // Handle any remaining blocks sequentially with EOR3.
  for block in &blocks[aligned..] {
    fold_block_128_eor3(&mut combined, block, coeff_128_low, coeff_128_high);
  }

  fold_tail(combined, consts)
}

#[target_feature(enable = "aes", enable = "neon", enable = "sha3")]
unsafe fn crc64_pmull_eor3_2way(
  mut state: u64,
  bytes: &[u8],
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 8],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if middle.len() < 2 {
    return crc64_pmull_eor3(state, bytes, consts, tables);
  }

  state = super::portable::crc64_slice8(state, left, tables);
  state = update_simd_eor3_2way(state, middle, fold_256b, consts);
  super::portable::crc64_slice8(state, right, tables)
}

#[target_feature(enable = "aes", enable = "neon", enable = "sha3")]
unsafe fn crc64_pmull_eor3_3way(
  mut state: u64,
  bytes: &[u8],
  fold_384b: (u64, u64),
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 8],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if middle.len() < 3 {
    return crc64_pmull_eor3_2way(state, bytes, fold_256b, consts, tables);
  }

  state = super::portable::crc64_slice8(state, left, tables);
  state = update_simd_eor3_3way(state, middle, fold_384b, fold_256b, consts);
  super::portable::crc64_slice8(state, right, tables)
}

#[target_feature(enable = "aes", enable = "neon")]
unsafe fn crc64_pmull_2way(
  mut state: u64,
  bytes: &[u8],
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 8],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if middle.len() < 2 {
    return crc64_pmull(state, bytes, consts, tables);
  }

  state = super::portable::crc64_slice8(state, left, tables);
  state = update_simd_2way(state, middle, fold_256b, consts);
  super::portable::crc64_slice8(state, right, tables)
}

#[target_feature(enable = "aes", enable = "neon")]
unsafe fn crc64_pmull_3way(
  mut state: u64,
  bytes: &[u8],
  fold_384b: (u64, u64),
  fold_256b: (u64, u64),
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 8],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if middle.len() < 3 {
    return crc64_pmull_2way(state, bytes, fold_256b, consts, tables);
  }

  state = super::portable::crc64_slice8(state, left, tables);
  state = update_simd_3way(state, middle, fold_384b, fold_256b, consts);
  super::portable::crc64_slice8(state, right, tables)
}

#[target_feature(enable = "aes", enable = "neon")]
unsafe fn crc64_pmull(mut state: u64, bytes: &[u8], consts: &Crc64ClmulConstants, tables: &[[u64; 256]; 8]) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if let Some((first, rest)) = middle.split_first() {
    state = super::portable::crc64_slice8(state, left, tables);
    state = update_simd(state, first, rest, consts);
    super::portable::crc64_slice8(state, right, tables)
  } else {
    super::portable::crc64_slice8(state, bytes, tables)
  }
}

/// PMULL+EOR3 path: uses 3-way XOR for faster folding.
///
/// Available on ARMv8.2+ with SHA3 extension (Apple M1+, AWS Graviton3+).
#[target_feature(enable = "aes", enable = "neon", enable = "sha3")]
unsafe fn crc64_pmull_eor3(
  mut state: u64,
  bytes: &[u8],
  consts: &Crc64ClmulConstants,
  tables: &[[u64; 256]; 8],
) -> u64 {
  let (left, middle, right) = bytes.align_to::<[Simd; 8]>();
  if let Some((first, rest)) = middle.split_first() {
    state = super::portable::crc64_slice8(state, left, tables);
    state = update_simd_eor3(state, first, rest, consts);
    super::portable::crc64_slice8(state, right, tables)
  } else {
    super::portable::crc64_slice8(state, bytes, tables)
  }
}

/// Small-buffer PMULL path: fold one 16-byte lane at a time.
///
/// This targets the regime where full 128-byte folding has too much setup cost,
/// but PMULL still outperforms table CRC (typically ~16..127 bytes depending on CPU).
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn crc64_pmull_small(
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
  let coeff_16b_low = consts.tail_fold_16b[6].1;
  let coeff_16b_high = consts.tail_fold_16b[6].0;

  for chunk in rest {
    acc = *chunk ^ acc.fold_16_pair(coeff_16b_low, coeff_16b_high);
  }

  // Reduce 16B → 8B → u64, then finish any tail bytes portably.
  state = acc.fold_8(consts.fold_8b).barrett(consts.poly, consts.mu);
  super::portable::crc64_slice8(state, right, tables)
}

/// CRC-64-XZ using PMULL folding.
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_READY)`.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_xz_pmull(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull(
      crc,
      data,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-XZ using PMULL (small-buffer lane folding).
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_READY)`.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_xz_pmull_small(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull_small(
      crc,
      data,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-XZ using a tuned "SVE2 PMULL" tier (2-way striping).
///
/// This is still implemented with NEON+PMULL intrinsics, but is intended for
/// high-throughput Armv9/SVE2-class CPUs where additional ILP helps.
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::SVE2_PMULL)` and `PMULL_READY` before selecting.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_xz_sve2_pmull_2way(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_xz_pmull_2way(crc, data) }
}

/// CRC-64-XZ using a tuned "SVE2 PMULL" tier (3-way striping).
///
/// This is still implemented with NEON+PMULL intrinsics, but is intended for
/// high-throughput Armv9/SVE2-class CPUs where additional ILP helps.
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::SVE2_PMULL)` and `PMULL_READY` before selecting.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_xz_sve2_pmull_3way(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_xz_pmull_3way(crc, data) }
}

/// CRC-64-NVME using PMULL folding.
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_READY)`.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_nvme_pmull(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull(
      crc,
      data,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

/// CRC-64-NVME using PMULL (small-buffer lane folding).
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_READY)`.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_nvme_pmull_small(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull_small(
      crc,
      data,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

/// CRC-64-NVME using a tuned "SVE2 PMULL" tier (2-way striping).
///
/// This is still implemented with NEON+PMULL intrinsics, but is intended for
/// high-throughput Armv9/SVE2-class CPUs where additional ILP helps.
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::SVE2_PMULL)` and `PMULL_READY` before selecting.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_nvme_sve2_pmull_2way(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_nvme_pmull_2way(crc, data) }
}

/// CRC-64-NVME using a tuned "SVE2 PMULL" tier (3-way striping).
///
/// This is still implemented with NEON+PMULL intrinsics, but is intended for
/// high-throughput Armv9/SVE2-class CPUs where additional ILP helps.
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::SVE2_PMULL)` and `PMULL_READY` before selecting.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_nvme_sve2_pmull_3way(crc: u64, data: &[u8]) -> u64 {
  unsafe { crc64_nvme_pmull_3way(crc, data) }
}

// ─────────────────────────────────────────────────────────────────────────────
// PMULL+EOR3 Kernels (ARMv8.2-SHA3)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-64-XZ using PMULL+EOR3 folding.
///
/// Uses the 3-way XOR instruction (EOR3) from ARMv8.2-SHA3 to reduce the
/// folding dependency chain. Available on Apple M1+, AWS Graviton3+, etc.
///
/// # Safety
///
/// Requires PMULL+SHA3 (crypto/aes + sha3). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_EOR3_READY)`.
#[target_feature(enable = "aes", enable = "neon", enable = "sha3")]
pub unsafe fn crc64_xz_pmull_eor3(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull_eor3(
      crc,
      data,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-NVME using PMULL+EOR3 folding.
///
/// Uses the 3-way XOR instruction (EOR3) from ARMv8.2-SHA3 to reduce the
/// folding dependency chain. Available on Apple M1+, AWS Graviton3+, etc.
///
/// # Safety
///
/// Requires PMULL+SHA3 (crypto/aes + sha3). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_EOR3_READY)`.
#[target_feature(enable = "aes", enable = "neon", enable = "sha3")]
pub unsafe fn crc64_nvme_pmull_eor3(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull_eor3(
      crc,
      data,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

/// CRC-64-XZ using PMULL+EOR3 folding with 2-way striping.
///
/// Combines the ILP benefit of 2-way striping with EOR3's reduced
/// XOR dependency chain. This is optimal for Apple M1+ and AWS Graviton3+
/// for large buffers.
///
/// # Safety
///
/// Requires PMULL+SHA3 (crypto/aes + sha3). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_EOR3_READY)`.
#[target_feature(enable = "aes", enable = "neon", enable = "sha3")]
pub unsafe fn crc64_xz_pmull_eor3_2way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull_eor3_2way(
      crc,
      data,
      CRC64_XZ_STREAM.fold_256b,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-XZ using PMULL+EOR3 folding with 3-way striping.
///
/// Combines the ILP benefit of 3-way striping with EOR3's reduced
/// XOR dependency chain. This is optimal for Apple M1+ and AWS Graviton3+
/// for very large buffers (32KB+).
///
/// # Safety
///
/// Requires PMULL+SHA3 (crypto/aes + sha3). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_EOR3_READY)`.
#[target_feature(enable = "aes", enable = "neon", enable = "sha3")]
pub unsafe fn crc64_xz_pmull_eor3_3way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull_eor3_3way(
      crc,
      data,
      CRC64_XZ_STREAM.fold_384b,
      CRC64_XZ_STREAM.fold_256b,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-NVME using PMULL+EOR3 folding with 2-way striping.
///
/// Combines the ILP benefit of 2-way striping with EOR3's reduced
/// XOR dependency chain. This is optimal for Apple M1+ and AWS Graviton3+
/// for large buffers.
///
/// # Safety
///
/// Requires PMULL+SHA3 (crypto/aes + sha3). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_EOR3_READY)`.
#[target_feature(enable = "aes", enable = "neon", enable = "sha3")]
pub unsafe fn crc64_nvme_pmull_eor3_2way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull_eor3_2way(
      crc,
      data,
      CRC64_NVME_STREAM.fold_256b,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

/// CRC-64-NVME using PMULL+EOR3 folding with 3-way striping.
///
/// Combines the ILP benefit of 3-way striping with EOR3's reduced
/// XOR dependency chain. This is optimal for Apple M1+ and AWS Graviton3+
/// for very large buffers (32KB+).
///
/// # Safety
///
/// Requires PMULL+SHA3 (crypto/aes + sha3). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_EOR3_READY)`.
#[target_feature(enable = "aes", enable = "neon", enable = "sha3")]
pub unsafe fn crc64_nvme_pmull_eor3_3way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull_eor3_3way(
      crc,
      data,
      CRC64_NVME_STREAM.fold_384b,
      CRC64_NVME_STREAM.fold_256b,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Wrappers (safe interface)
// ─────────────────────────────────────────────────────────────────────────────

/// Safe wrapper for CRC-64-XZ PMULL kernel.
#[inline]
pub fn crc64_xz_pmull_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL (crypto/aes) before selecting this kernel.
  unsafe { crc64_xz_pmull(crc, data) }
}

/// Safe wrapper for CRC-64-XZ PMULL small-buffer kernel.
#[inline]
pub fn crc64_xz_pmull_small_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL (crypto/aes) before selecting this kernel.
  unsafe { crc64_xz_pmull_small(crc, data) }
}

/// Safe wrapper for CRC-64-XZ tuned "SVE2 PMULL" tier (single-stream).
#[inline]
pub fn crc64_xz_sve2_pmull_safe(crc: u64, data: &[u8]) -> u64 {
  crc64_xz_pmull_safe(crc, data)
}

/// Safe wrapper for CRC-64-XZ tuned "SVE2 PMULL" tier (small-buffer).
#[inline]
pub fn crc64_xz_sve2_pmull_small_safe(crc: u64, data: &[u8]) -> u64 {
  crc64_xz_pmull_small_safe(crc, data)
}

/// Safe wrapper for CRC-64-XZ tuned SVE2 2-way PMULL kernel.
#[inline]
pub fn crc64_xz_sve2_pmull_2way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies SVE2_PMULL + PMULL before selecting this kernel.
  unsafe { crc64_xz_sve2_pmull_2way(crc, data) }
}

/// Safe wrapper for CRC-64-XZ tuned SVE2 3-way PMULL kernel.
#[inline]
pub fn crc64_xz_sve2_pmull_3way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies SVE2_PMULL + PMULL before selecting this kernel.
  unsafe { crc64_xz_sve2_pmull_3way(crc, data) }
}

/// Safe wrapper for CRC-64-NVME PMULL kernel.
#[inline]
pub fn crc64_nvme_pmull_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL (crypto/aes) before selecting this kernel.
  unsafe { crc64_nvme_pmull(crc, data) }
}

/// Safe wrapper for CRC-64-NVME PMULL small-buffer kernel.
#[inline]
pub fn crc64_nvme_pmull_small_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL (crypto/aes) before selecting this kernel.
  unsafe { crc64_nvme_pmull_small(crc, data) }
}

/// Safe wrapper for CRC-64-NVME tuned "SVE2 PMULL" tier (single-stream).
#[inline]
pub fn crc64_nvme_sve2_pmull_safe(crc: u64, data: &[u8]) -> u64 {
  crc64_nvme_pmull_safe(crc, data)
}

/// Safe wrapper for CRC-64-NVME tuned "SVE2 PMULL" tier (small-buffer).
#[inline]
pub fn crc64_nvme_sve2_pmull_small_safe(crc: u64, data: &[u8]) -> u64 {
  crc64_nvme_pmull_small_safe(crc, data)
}

/// Safe wrapper for CRC-64-NVME tuned SVE2 2-way PMULL kernel.
#[inline]
pub fn crc64_nvme_sve2_pmull_2way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies SVE2_PMULL + PMULL before selecting this kernel.
  unsafe { crc64_nvme_sve2_pmull_2way(crc, data) }
}

/// Safe wrapper for CRC-64-NVME tuned SVE2 3-way PMULL kernel.
#[inline]
pub fn crc64_nvme_sve2_pmull_3way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies SVE2_PMULL + PMULL before selecting this kernel.
  unsafe { crc64_nvme_sve2_pmull_3way(crc, data) }
}

/// Safe wrapper for CRC-64-XZ PMULL+EOR3 kernel.
#[inline]
pub fn crc64_xz_pmull_eor3_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL_EOR3_READY before selecting this kernel.
  unsafe { crc64_xz_pmull_eor3(crc, data) }
}

/// Safe wrapper for CRC-64-NVME PMULL+EOR3 kernel.
#[inline]
pub fn crc64_nvme_pmull_eor3_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL_EOR3_READY before selecting this kernel.
  unsafe { crc64_nvme_pmull_eor3(crc, data) }
}

/// Safe wrapper for CRC-64-XZ PMULL+EOR3 2-way kernel.
#[inline]
pub fn crc64_xz_pmull_eor3_2way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL_EOR3_READY before selecting this kernel.
  unsafe { crc64_xz_pmull_eor3_2way(crc, data) }
}

/// Safe wrapper for CRC-64-XZ PMULL+EOR3 3-way kernel.
#[inline]
pub fn crc64_xz_pmull_eor3_3way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL_EOR3_READY before selecting this kernel.
  unsafe { crc64_xz_pmull_eor3_3way(crc, data) }
}

/// Safe wrapper for CRC-64-NVME PMULL+EOR3 2-way kernel.
#[inline]
pub fn crc64_nvme_pmull_eor3_2way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL_EOR3_READY before selecting this kernel.
  unsafe { crc64_nvme_pmull_eor3_2way(crc, data) }
}

/// Safe wrapper for CRC-64-NVME PMULL+EOR3 3-way kernel.
#[inline]
pub fn crc64_nvme_pmull_eor3_3way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL_EOR3_READY before selecting this kernel.
  unsafe { crc64_nvme_pmull_eor3_3way(crc, data) }
}

/// CRC-64-XZ using PMULL folding (2-way striping).
///
/// This is still NEON+PMULL, but splits the main loop into two independent
/// streams to improve ILP on some microarchitectures (including Apple M-series).
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_READY)`.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_xz_pmull_2way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull_2way(
      crc,
      data,
      CRC64_XZ_STREAM.fold_256b,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-XZ using PMULL folding (3-way striping).
///
/// This is still NEON+PMULL, but splits the main loop into three independent
/// streams to improve ILP on some microarchitectures.
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_READY)`.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_xz_pmull_3way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull_3way(
      crc,
      data,
      CRC64_XZ_STREAM.fold_384b,
      CRC64_XZ_STREAM.fold_256b,
      &crate::common::clmul::CRC64_XZ_CLMUL,
      &super::kernel_tables::XZ_TABLES_8,
    )
  }
}

/// CRC-64-NVME using PMULL folding (2-way striping).
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_READY)`.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_nvme_pmull_2way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull_2way(
      crc,
      data,
      CRC64_NVME_STREAM.fold_256b,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

/// CRC-64-NVME using PMULL folding (3-way striping).
///
/// # Safety
///
/// Requires PMULL (crypto/aes). Caller must verify via
/// `platform::caps().has(aarch64::PMULL_READY)`.
#[target_feature(enable = "aes", enable = "neon")]
pub unsafe fn crc64_nvme_pmull_3way(crc: u64, data: &[u8]) -> u64 {
  unsafe {
    crc64_pmull_3way(
      crc,
      data,
      CRC64_NVME_STREAM.fold_384b,
      CRC64_NVME_STREAM.fold_256b,
      &crate::common::clmul::CRC64_NVME_CLMUL,
      &super::kernel_tables::NVME_TABLES_8,
    )
  }
}

/// Safe wrapper for CRC-64-XZ PMULL 2-way kernel.
#[inline]
pub fn crc64_xz_pmull_2way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL (crypto/aes) before selecting this kernel.
  unsafe { crc64_xz_pmull_2way(crc, data) }
}

/// Safe wrapper for CRC-64-XZ PMULL 3-way kernel.
#[inline]
pub fn crc64_xz_pmull_3way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL (crypto/aes) before selecting this kernel.
  unsafe { crc64_xz_pmull_3way(crc, data) }
}

/// Safe wrapper for CRC-64-NVME PMULL 2-way kernel.
#[inline]
pub fn crc64_nvme_pmull_2way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL (crypto/aes) before selecting this kernel.
  unsafe { crc64_nvme_pmull_2way(crc, data) }
}

/// Safe wrapper for CRC-64-NVME PMULL 3-way kernel.
#[inline]
pub fn crc64_nvme_pmull_3way_safe(crc: u64, data: &[u8]) -> u64 {
  // SAFETY: Dispatcher verifies PMULL (crypto/aes) before selecting this kernel.
  unsafe { crc64_nvme_pmull_3way(crc, data) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// Tests require SIMD intrinsics that Miri cannot interpret.
#[cfg(all(test, not(miri)))]
mod tests {
  use super::*;

  const TEST_DATA: &[u8] = b"123456789";

  fn make_data(len: usize) -> alloc::vec::Vec<u8> {
    (0..len)
      .map(|i| (i as u8).wrapping_mul(17).wrapping_add((i >> 3) as u8))
      .collect()
  }

  #[test]
  fn test_crc64_xz_pmull_matches_vector() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    // SAFETY: We just checked AES/PMULL is available.
    let crc = unsafe { crc64_xz_pmull(!0, TEST_DATA) } ^ !0;
    assert_eq!(crc, 0x995D_C9BB_DF19_39FA);
  }

  #[test]
  fn test_crc64_nvme_pmull_matches_vector() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    // SAFETY: We just checked AES/PMULL is available.
    let crc = unsafe { crc64_nvme_pmull(!0, TEST_DATA) } ^ !0;
    assert_eq!(crc, 0xAE8B_1486_0A79_9888);
  }

  #[test]
  fn test_crc64_xz_pmull_matches_portable_various_lengths() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    for len in [
      0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 129, 255, 256, 511, 512, 1024,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      // SAFETY: We just checked AES/PMULL is available.
      let pmull = unsafe { crc64_xz_pmull(!0, &data) } ^ !0;
      assert_eq!(pmull, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_pmull_matches_portable_various_lengths() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    for len in [
      0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 129, 255, 256, 511, 512, 1024,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      // SAFETY: We just checked AES/PMULL is available.
      let pmull = unsafe { crc64_nvme_pmull(!0, &data) } ^ !0;
      assert_eq!(pmull, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_xz_pmull_small_matches_portable_all_lengths_0_511() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    for len in 0..super::super::policy::CRC64_SMALL_KERNEL_MAX_BYTES_DEFAULT {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      let pmull_small = crc64_xz_pmull_small_safe(!0, &data) ^ !0;
      assert_eq!(pmull_small, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_pmull_small_matches_portable_all_lengths_0_511() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    for len in 0..super::super::policy::CRC64_SMALL_KERNEL_MAX_BYTES_DEFAULT {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      let pmull_small = crc64_nvme_pmull_small_safe(!0, &data) ^ !0;
      assert_eq!(pmull_small, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_xz_sve2_pmull_2way_matches_portable_various_lengths() {
    let caps = platform::caps();
    if !caps.has(platform::caps::aarch64::SVE2_PMULL) || !caps.has(platform::caps::aarch64::PMULL_READY) {
      return;
    }

    for len in [
      0usize, 1, 7, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1024, 4096,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      let sve2 = crc64_xz_sve2_pmull_2way_safe(!0, &data) ^ !0;
      assert_eq!(sve2, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_xz_sve2_pmull_3way_matches_portable_various_lengths() {
    let caps = platform::caps();
    if !caps.has(platform::caps::aarch64::SVE2_PMULL) || !caps.has(platform::caps::aarch64::PMULL_READY) {
      return;
    }

    for len in [
      0usize, 1, 7, 16, 31, 32, 63, 64, 127, 128, 255, 256, 257, 383, 384, 511, 512, 513, 639, 640, 767, 768, 1024,
      4096,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      let sve2 = crc64_xz_sve2_pmull_3way_safe(!0, &data) ^ !0;
      assert_eq!(sve2, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_sve2_pmull_2way_matches_portable_various_lengths() {
    let caps = platform::caps();
    if !caps.has(platform::caps::aarch64::SVE2_PMULL) || !caps.has(platform::caps::aarch64::PMULL_READY) {
      return;
    }

    for len in [
      0usize, 1, 7, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1024, 4096,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      let sve2 = crc64_nvme_sve2_pmull_2way_safe(!0, &data) ^ !0;
      assert_eq!(sve2, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_sve2_pmull_3way_matches_portable_various_lengths() {
    let caps = platform::caps();
    if !caps.has(platform::caps::aarch64::SVE2_PMULL) || !caps.has(platform::caps::aarch64::PMULL_READY) {
      return;
    }

    for len in [
      0usize, 1, 7, 16, 31, 32, 63, 64, 127, 128, 255, 256, 257, 383, 384, 511, 512, 513, 639, 640, 767, 768, 1024,
      4096,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      let sve2 = crc64_nvme_sve2_pmull_3way_safe(!0, &data) ^ !0;
      assert_eq!(sve2, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_xz_pmull_2way_matches_portable_various_lengths() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    for len in [
      0usize, 1, 7, 16, 31, 32, 63, 64, 127, 128, 255, 256, 257, 383, 384, 511, 512, 513, 1024, 4096,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      let pmull_2way = crc64_xz_pmull_2way_safe(!0, &data) ^ !0;
      assert_eq!(pmull_2way, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_xz_pmull_3way_matches_portable_various_lengths() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    for len in [
      0usize, 1, 7, 16, 31, 32, 63, 64, 127, 128, 255, 256, 257, 383, 384, 511, 512, 513, 639, 640, 767, 768, 1024,
      4096,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      let pmull_3way = crc64_xz_pmull_3way_safe(!0, &data) ^ !0;
      assert_eq!(pmull_3way, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_pmull_2way_matches_portable_various_lengths() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    for len in [
      0usize, 1, 7, 16, 31, 32, 63, 64, 127, 128, 255, 256, 257, 383, 384, 511, 512, 513, 1024, 4096,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      let pmull_2way = crc64_nvme_pmull_2way_safe(!0, &data) ^ !0;
      assert_eq!(pmull_2way, portable, "mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_pmull_3way_matches_portable_various_lengths() {
    if !std::arch::is_aarch64_feature_detected!("aes") {
      return;
    }

    for len in [
      0usize, 1, 7, 16, 31, 32, 63, 64, 127, 128, 255, 256, 257, 383, 384, 511, 512, 513, 639, 640, 767, 768, 1024,
      4096,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      let pmull_3way = crc64_nvme_pmull_3way_safe(!0, &data) ^ !0;
      assert_eq!(pmull_3way, portable, "mismatch at len={len}");
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // PMULL+EOR3 tests (ARMv8.2-SHA3)
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_crc64_xz_pmull_eor3_matches_vector() {
    if !std::arch::is_aarch64_feature_detected!("aes") || !std::arch::is_aarch64_feature_detected!("sha3") {
      return;
    }

    // SAFETY: We just checked AES+SHA3 is available.
    let crc = crc64_xz_pmull_eor3_safe(!0, TEST_DATA) ^ !0;
    assert_eq!(crc, 0x995D_C9BB_DF19_39FA);
  }

  #[test]
  fn test_crc64_nvme_pmull_eor3_matches_vector() {
    if !std::arch::is_aarch64_feature_detected!("aes") || !std::arch::is_aarch64_feature_detected!("sha3") {
      return;
    }

    // SAFETY: We just checked AES+SHA3 is available.
    let crc = crc64_nvme_pmull_eor3_safe(!0, TEST_DATA) ^ !0;
    assert_eq!(crc, 0xAE8B_1486_0A79_9888);
  }

  #[test]
  fn test_crc64_xz_pmull_eor3_matches_portable_various_lengths() {
    if !std::arch::is_aarch64_feature_detected!("aes") || !std::arch::is_aarch64_feature_detected!("sha3") {
      return;
    }

    // Test lengths that exercise all code paths:
    // - < 128: falls through to portable (no EOR3 benefit)
    // - >= 128: uses EOR3 folding
    // - various alignments and block boundaries
    for len in [
      0usize, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 129, 255, 256, 257, 383, 384, 511, 512, 513, 639, 640, 767,
      768, 1024, 2048, 4096, 8192,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      let eor3 = crc64_xz_pmull_eor3_safe(!0, &data) ^ !0;
      assert_eq!(eor3, portable, "XZ EOR3 mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_pmull_eor3_matches_portable_various_lengths() {
    if !std::arch::is_aarch64_feature_detected!("aes") || !std::arch::is_aarch64_feature_detected!("sha3") {
      return;
    }

    for len in [
      0usize, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 129, 255, 256, 257, 383, 384, 511, 512, 513, 639, 640, 767,
      768, 1024, 2048, 4096, 8192,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      let eor3 = crc64_nvme_pmull_eor3_safe(!0, &data) ^ !0;
      assert_eq!(eor3, portable, "NVME EOR3 mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_xz_pmull_eor3_matches_standard_pmull() {
    if !std::arch::is_aarch64_feature_detected!("aes") || !std::arch::is_aarch64_feature_detected!("sha3") {
      return;
    }

    // EOR3 should produce identical results to standard PMULL - just faster
    for len in [128, 256, 512, 1024, 4096] {
      let data = make_data(len);
      let pmull = crc64_xz_pmull_safe(!0, &data);
      let eor3 = crc64_xz_pmull_eor3_safe(!0, &data);
      assert_eq!(eor3, pmull, "EOR3 vs PMULL mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_pmull_eor3_matches_standard_pmull() {
    if !std::arch::is_aarch64_feature_detected!("aes") || !std::arch::is_aarch64_feature_detected!("sha3") {
      return;
    }

    // EOR3 should produce identical results to standard PMULL - just faster
    for len in [128, 256, 512, 1024, 4096] {
      let data = make_data(len);
      let pmull = crc64_nvme_pmull_safe(!0, &data);
      let eor3 = crc64_nvme_pmull_eor3_safe(!0, &data);
      assert_eq!(eor3, pmull, "EOR3 vs PMULL mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_pmull_eor3_streaming() {
    if !std::arch::is_aarch64_feature_detected!("aes") || !std::arch::is_aarch64_feature_detected!("sha3") {
      return;
    }

    // Test that streaming updates produce correct results with EOR3
    let data = make_data(1024);
    let oneshot = crc64_xz_pmull_eor3_safe(!0, &data);

    // Split at various points
    for split in [128, 256, 384, 512, 640, 768, 896] {
      let state1 = crc64_xz_pmull_eor3_safe(!0, &data[..split]);
      let state2 = crc64_xz_pmull_eor3_safe(state1, &data[split..]);
      assert_eq!(state2, oneshot, "streaming mismatch at split={split}");
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // PMULL+EOR3 Multi-Stream tests (2-way and 3-way)
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_crc64_xz_pmull_eor3_2way_matches_portable_various_lengths() {
    if !std::arch::is_aarch64_feature_detected!("aes") || !std::arch::is_aarch64_feature_detected!("sha3") {
      return;
    }

    // Test lengths that exercise all code paths:
    // - < 256: falls back to single-stream EOR3 (< 2 blocks)
    // - >= 256: uses 2-way striping with EOR3
    // - various alignments and block boundaries
    for len in [
      0usize, 1, 7, 16, 31, 64, 127, 128, 129, 255, 256, 257, 383, 384, 511, 512, 513, 639, 640, 767, 768, 1024, 2048,
      4096, 8192, 16384, 32768,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      let eor3_2way = crc64_xz_pmull_eor3_2way_safe(!0, &data) ^ !0;
      assert_eq!(eor3_2way, portable, "XZ EOR3-2way mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_xz_pmull_eor3_3way_matches_portable_various_lengths() {
    if !std::arch::is_aarch64_feature_detected!("aes") || !std::arch::is_aarch64_feature_detected!("sha3") {
      return;
    }

    // Test lengths that exercise all code paths:
    // - < 384: falls back to 2-way (< 3 blocks)
    // - >= 384: uses 3-way striping with EOR3
    // - various alignments and block boundaries
    for len in [
      0usize, 1, 7, 16, 31, 64, 127, 128, 129, 255, 256, 257, 383, 384, 385, 511, 512, 513, 639, 640, 767, 768, 1024,
      2048, 4096, 8192, 16384, 32768, 65536,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_xz(!0, &data) ^ !0;
      let eor3_3way = crc64_xz_pmull_eor3_3way_safe(!0, &data) ^ !0;
      assert_eq!(eor3_3way, portable, "XZ EOR3-3way mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_pmull_eor3_2way_matches_portable_various_lengths() {
    if !std::arch::is_aarch64_feature_detected!("aes") || !std::arch::is_aarch64_feature_detected!("sha3") {
      return;
    }

    for len in [
      0usize, 1, 7, 16, 31, 64, 127, 128, 129, 255, 256, 257, 383, 384, 511, 512, 513, 639, 640, 767, 768, 1024, 2048,
      4096, 8192, 16384, 32768,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      let eor3_2way = crc64_nvme_pmull_eor3_2way_safe(!0, &data) ^ !0;
      assert_eq!(eor3_2way, portable, "NVME EOR3-2way mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_pmull_eor3_3way_matches_portable_various_lengths() {
    if !std::arch::is_aarch64_feature_detected!("aes") || !std::arch::is_aarch64_feature_detected!("sha3") {
      return;
    }

    for len in [
      0usize, 1, 7, 16, 31, 64, 127, 128, 129, 255, 256, 257, 383, 384, 385, 511, 512, 513, 639, 640, 767, 768, 1024,
      2048, 4096, 8192, 16384, 32768, 65536,
    ] {
      let data = make_data(len);
      let portable = super::super::portable::crc64_slice8_nvme(!0, &data) ^ !0;
      let eor3_3way = crc64_nvme_pmull_eor3_3way_safe(!0, &data) ^ !0;
      assert_eq!(eor3_3way, portable, "NVME EOR3-3way mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_xz_pmull_eor3_multiway_matches_single_stream() {
    if !std::arch::is_aarch64_feature_detected!("aes") || !std::arch::is_aarch64_feature_detected!("sha3") {
      return;
    }

    // Multi-stream EOR3 should produce identical results to single-stream EOR3
    for len in [256, 384, 512, 1024, 4096, 16384, 32768] {
      let data = make_data(len);
      let single = crc64_xz_pmull_eor3_safe(!0, &data);
      let two_way = crc64_xz_pmull_eor3_2way_safe(!0, &data);
      let three_way = crc64_xz_pmull_eor3_3way_safe(!0, &data);
      assert_eq!(two_way, single, "EOR3 2-way vs single mismatch at len={len}");
      assert_eq!(three_way, single, "EOR3 3-way vs single mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_nvme_pmull_eor3_multiway_matches_single_stream() {
    if !std::arch::is_aarch64_feature_detected!("aes") || !std::arch::is_aarch64_feature_detected!("sha3") {
      return;
    }

    // Multi-stream EOR3 should produce identical results to single-stream EOR3
    for len in [256, 384, 512, 1024, 4096, 16384, 32768] {
      let data = make_data(len);
      let single = crc64_nvme_pmull_eor3_safe(!0, &data);
      let two_way = crc64_nvme_pmull_eor3_2way_safe(!0, &data);
      let three_way = crc64_nvme_pmull_eor3_3way_safe(!0, &data);
      assert_eq!(two_way, single, "EOR3 2-way vs single mismatch at len={len}");
      assert_eq!(three_way, single, "EOR3 3-way vs single mismatch at len={len}");
    }
  }

  #[test]
  fn test_crc64_pmull_eor3_2way_streaming() {
    if !std::arch::is_aarch64_feature_detected!("aes") || !std::arch::is_aarch64_feature_detected!("sha3") {
      return;
    }

    // Test that streaming updates produce correct results with EOR3 2-way
    let data = make_data(4096);
    let oneshot = crc64_xz_pmull_eor3_2way_safe(!0, &data);

    // Split at various points
    for split in [128, 256, 384, 512, 1024, 2048, 3072] {
      let state1 = crc64_xz_pmull_eor3_2way_safe(!0, &data[..split]);
      let state2 = crc64_xz_pmull_eor3_2way_safe(state1, &data[split..]);
      assert_eq!(state2, oneshot, "2-way streaming mismatch at split={split}");
    }
  }

  #[test]
  fn test_crc64_pmull_eor3_3way_streaming() {
    if !std::arch::is_aarch64_feature_detected!("aes") || !std::arch::is_aarch64_feature_detected!("sha3") {
      return;
    }

    // Test that streaming updates produce correct results with EOR3 3-way
    let data = make_data(8192);
    let oneshot = crc64_xz_pmull_eor3_3way_safe(!0, &data);

    // Split at various points including at block boundaries
    for split in [128, 256, 384, 512, 768, 1024, 2048, 4096, 6144] {
      let state1 = crc64_xz_pmull_eor3_3way_safe(!0, &data[..split]);
      let state2 = crc64_xz_pmull_eor3_3way_safe(state1, &data[split..]);
      assert_eq!(state2, oneshot, "3-way streaming mismatch at split={split}");
    }
  }
}
