//! aarch64 carryless-multiply CRC-24/OPENPGP kernels (PMULL).
//!
//! CRC-24/OPENPGP is MSB-first in the portable implementation. These kernels
//! reuse the existing "width32" folding/reduction structure by computing the
//! equivalent **reflected** CRC-24 over per-byte bit-reversed input.
//!
//! # Safety
//!
//! Uses `unsafe` for ARM SIMD intrinsics. Callers must ensure PMULL is
//! available before executing these kernels (the dispatcher does this).
#![allow(unsafe_code)]
#![allow(clippy::indexing_slicing)]
#![allow(unsafe_op_in_unsafe_fn)]

use core::{
  arch::aarch64::*,
  ops::{BitXor, BitXorAssign},
};

use super::{
  keys::{CRC24_OPENPGP_KEYS_REFLECTED, CRC24_OPENPGP_STREAM_REFLECTED},
  reflected::{crc24_reflected_update_bitrev_bytes, from_reflected_state, to_reflected_state},
};

// ─────────────────────────────────────────────────────────────────────────────
// SIMD helpers
// ─────────────────────────────────────────────────────────────────────────────

#[repr(transparent)]
#[derive(Copy, Clone)]
struct Simd(uint8x16_t);

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

impl Simd {
  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn new(high: u64, low: u64) -> Self {
    Self(vcombine_u8(vcreate_u8(low), vcreate_u8(high)))
  }

  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn load(ptr: *const u8) -> Self {
    Self(vld1q_u8(ptr))
  }

  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn and(self, mask: Self) -> Self {
    Self(vandq_u8(self.0, mask.0))
  }

  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn shift_right_8(self) -> Self {
    Self(vextq_u8(self.0, vdupq_n_u8(0), 8))
  }

  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn shift_left_12(self) -> Self {
    let low_32 = vgetq_lane_u32(vreinterpretq_u32_u8(self.0), 0);
    let result = vsetq_lane_u32(low_32, vdupq_n_u32(0), 3);
    Self(vreinterpretq_u8_u32(result))
  }

  /// Reverse bits within each byte (u8::reverse_bits), lane-wise.
  #[inline]
  #[target_feature(enable = "neon")]
  unsafe fn bitrev_bytes(self) -> Self {
    Self(vrbitq_u8(self.0))
  }

  #[inline]
  #[target_feature(enable = "aes")]
  unsafe fn clmul00(self, other: Self) -> Self {
    let a = vgetq_lane_p64(vreinterpretq_p64_u8(self.0), 0);
    let b = vgetq_lane_p64(vreinterpretq_p64_u8(other.0), 0);
    Self(vreinterpretq_u8_p128(vmull_p64(a, b)))
  }

  #[inline]
  #[target_feature(enable = "aes")]
  unsafe fn clmul01(self, other: Self) -> Self {
    let a = vgetq_lane_p64(vreinterpretq_p64_u8(self.0), 1);
    let b = vgetq_lane_p64(vreinterpretq_p64_u8(other.0), 0);
    Self(vreinterpretq_u8_p128(vmull_p64(a, b)))
  }

  #[inline]
  #[target_feature(enable = "aes")]
  unsafe fn clmul10(self, other: Self) -> Self {
    let a = vgetq_lane_p64(vreinterpretq_p64_u8(self.0), 0);
    let b = vgetq_lane_p64(vreinterpretq_p64_u8(other.0), 1);
    Self(vreinterpretq_u8_p128(vmull_p64(a, b)))
  }

  #[inline]
  #[target_feature(enable = "aes")]
  unsafe fn clmul11(self, other: Self) -> Self {
    let a = vgetq_lane_p64(vreinterpretq_p64_u8(self.0), 1);
    let b = vgetq_lane_p64(vreinterpretq_p64_u8(other.0), 1);
    Self(vreinterpretq_u8_p128(vmull_p64(a, b)))
  }

  #[inline]
  #[target_feature(enable = "aes")]
  unsafe fn fold_16_reflected(self, coeff: Self, data_to_xor: Self) -> Self {
    let h = self.clmul10(coeff);
    let l = self.clmul01(coeff);
    h ^ l ^ data_to_xor
  }

  #[inline]
  #[target_feature(enable = "aes", enable = "neon")]
  unsafe fn fold_width32_reflected(self, high: u64, low: u64) -> Self {
    let coeff_low = Self::new(0, low);
    let coeff_high = Self::new(high, 0);

    let clmul = self.clmul00(coeff_low);
    let shifted = self.shift_right_8();
    let mut state = clmul ^ shifted;

    let mask2 = Self::new(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_0000_0000);
    let masked = state.and(mask2);
    let shifted = state.shift_left_12();
    let clmul = shifted.clmul11(coeff_high);
    state = clmul ^ masked;

    state
  }

  #[inline]
  #[target_feature(enable = "aes", enable = "neon")]
  unsafe fn barrett_width32_reflected(self, poly: u64, mu: u64) -> u32 {
    let polymu = Self::new(poly, mu);
    let clmul1 = self.clmul00(polymu);
    let clmul2 = clmul1.clmul10(polymu);
    let xorred = self ^ clmul2;

    let hi = xorred.shift_right_8();
    vgetq_lane_u32(vreinterpretq_u32_u8(hi.0), 0)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8-lane width32 update (128B blocks)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn finalize_lanes_width32_reflected(x: [Simd; 8], keys: &[u64; 23]) -> u32 {
  let mut res = x[7];
  res = x[0].fold_16_reflected(Simd::new(keys[10], keys[9]), res);
  res = x[1].fold_16_reflected(Simd::new(keys[12], keys[11]), res);
  res = x[2].fold_16_reflected(Simd::new(keys[14], keys[13]), res);
  res = x[3].fold_16_reflected(Simd::new(keys[16], keys[15]), res);
  res = x[4].fold_16_reflected(Simd::new(keys[18], keys[17]), res);
  res = x[5].fold_16_reflected(Simd::new(keys[20], keys[19]), res);
  res = x[6].fold_16_reflected(Simd::new(keys[2], keys[1]), res);

  res = res.fold_width32_reflected(keys[6], keys[5]);
  res.barrett_width32_reflected(keys[8], keys[7])
}

#[inline]
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn bitrev_block(block: &[Simd; 8]) -> [Simd; 8] {
  [
    block[0].bitrev_bytes(),
    block[1].bitrev_bytes(),
    block[2].bitrev_bytes(),
    block[3].bitrev_bytes(),
    block[4].bitrev_bytes(),
    block[5].bitrev_bytes(),
    block[6].bitrev_bytes(),
    block[7].bitrev_bytes(),
  ]
}

#[inline]
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn fold_block_128_width32_reflected_bitrev_bytes(x: &mut [Simd; 8], chunk: &[Simd; 8], coeff: Simd) {
  let y0 = chunk[0].bitrev_bytes();
  let y1 = chunk[1].bitrev_bytes();
  let y2 = chunk[2].bitrev_bytes();
  let y3 = chunk[3].bitrev_bytes();
  let y4 = chunk[4].bitrev_bytes();
  let y5 = chunk[5].bitrev_bytes();
  let y6 = chunk[6].bitrev_bytes();
  let y7 = chunk[7].bitrev_bytes();

  x[0] = x[0].fold_16_reflected(coeff, y0);
  x[1] = x[1].fold_16_reflected(coeff, y1);
  x[2] = x[2].fold_16_reflected(coeff, y2);
  x[3] = x[3].fold_16_reflected(coeff, y3);
  x[4] = x[4].fold_16_reflected(coeff, y4);
  x[5] = x[5].fold_16_reflected(coeff, y5);
  x[6] = x[6].fold_16_reflected(coeff, y6);
  x[7] = x[7].fold_16_reflected(coeff, y7);
}

#[target_feature(enable = "aes", enable = "neon")]
unsafe fn update_simd_width32_reflected_bitrev_bytes_2way(
  state: u32,
  blocks: &[[Simd; 8]],
  fold_256b: (u64, u64),
  keys: &[u64; 23],
) -> u32 {
  use crate::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  debug_assert!(blocks.len() >= 2);

  let coeff_256b = Simd::new(fold_256b.0, fold_256b.1);
  let coeff_128b = Simd::new(keys[4], keys[3]);
  let zero = Simd::new(0, 0);

  let mut s0 = bitrev_block(&blocks[0]);
  let mut s1 = bitrev_block(&blocks[1]);

  // Inject CRC into stream 0 (block 0).
  s0[0] ^= Simd::new(0, state as u64);

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
    fold_block_128_width32_reflected_bitrev_bytes(&mut s0, &blocks[i], coeff_256b);
    fold_block_128_width32_reflected_bitrev_bytes(&mut s1, &blocks[i.strict_add(1)], coeff_256b);

    // Second iteration (blocks i+2, i+3)
    fold_block_128_width32_reflected_bitrev_bytes(&mut s0, &blocks[i.strict_add(2)], coeff_256b);
    fold_block_128_width32_reflected_bitrev_bytes(&mut s1, &blocks[i.strict_add(3)], coeff_256b);

    i = i.strict_add(DOUBLE_GROUP);
  }

  // Handle remaining pairs.
  let even = blocks.len() & !1usize;
  while i < even {
    fold_block_128_width32_reflected_bitrev_bytes(&mut s0, &blocks[i], coeff_256b);
    fold_block_128_width32_reflected_bitrev_bytes(&mut s1, &blocks[i.strict_add(1)], coeff_256b);
    i = i.strict_add(2);
  }

  // Merge streams: A·s0 ⊕ s1 (A = shift by 128B).
  let mut combined = s1;
  combined[0] ^= s0[0].fold_16_reflected(coeff_128b, zero);
  combined[1] ^= s0[1].fold_16_reflected(coeff_128b, zero);
  combined[2] ^= s0[2].fold_16_reflected(coeff_128b, zero);
  combined[3] ^= s0[3].fold_16_reflected(coeff_128b, zero);
  combined[4] ^= s0[4].fold_16_reflected(coeff_128b, zero);
  combined[5] ^= s0[5].fold_16_reflected(coeff_128b, zero);
  combined[6] ^= s0[6].fold_16_reflected(coeff_128b, zero);
  combined[7] ^= s0[7].fold_16_reflected(coeff_128b, zero);

  // Handle any remaining block (odd tail) sequentially.
  if even != blocks.len() {
    fold_block_128_width32_reflected_bitrev_bytes(&mut combined, &blocks[even], coeff_128b);
  }

  finalize_lanes_width32_reflected(combined, keys)
}

#[target_feature(enable = "aes", enable = "neon")]
unsafe fn update_simd_width32_reflected_bitrev_bytes_3way(
  state: u32,
  blocks: &[[Simd; 8]],
  fold_384b: (u64, u64),
  fold_256b: (u64, u64),
  keys: &[u64; 23],
) -> u32 {
  use crate::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  if blocks.len() < 3 {
    let Some((first, rest)) = blocks.split_first() else {
      return state;
    };
    return update_simd_width32_reflected_bitrev_bytes(state, first, rest, keys);
  }

  let coeff_384b = Simd::new(fold_384b.0, fold_384b.1);
  let coeff_256b = Simd::new(fold_256b.0, fold_256b.1);
  let coeff_128b = Simd::new(keys[4], keys[3]);
  let zero = Simd::new(0, 0);

  let mut s0 = bitrev_block(&blocks[0]);
  let mut s1 = bitrev_block(&blocks[1]);
  let mut s2 = bitrev_block(&blocks[2]);

  // Inject CRC into stream 0 (block 0).
  s0[0] ^= Simd::new(0, state as u64);

  // Double-unrolled main loop: process 6 blocks (768B) per iteration.
  const BLOCK_SIZE: usize = 128;
  const DOUBLE_GROUP: usize = 6; // 2 × 3-way = 6 blocks = 768B

  let mut i: usize = 3;
  let aligned = (blocks.len() / DOUBLE_GROUP) * DOUBLE_GROUP;

  while i.strict_add(DOUBLE_GROUP) <= aligned {
    let prefetch_idx = i.strict_add(LARGE_BLOCK_DISTANCE / BLOCK_SIZE);
    if prefetch_idx < blocks.len() {
      prefetch_read_l1(blocks[prefetch_idx].as_ptr().cast::<u8>());
    }

    // First iteration (blocks i, i+1, i+2)
    fold_block_128_width32_reflected_bitrev_bytes(&mut s0, &blocks[i], coeff_384b);
    fold_block_128_width32_reflected_bitrev_bytes(&mut s1, &blocks[i.strict_add(1)], coeff_384b);
    fold_block_128_width32_reflected_bitrev_bytes(&mut s2, &blocks[i.strict_add(2)], coeff_384b);

    // Second iteration (blocks i+3, i+4, i+5)
    fold_block_128_width32_reflected_bitrev_bytes(&mut s0, &blocks[i.strict_add(3)], coeff_384b);
    fold_block_128_width32_reflected_bitrev_bytes(&mut s1, &blocks[i.strict_add(4)], coeff_384b);
    fold_block_128_width32_reflected_bitrev_bytes(&mut s2, &blocks[i.strict_add(5)], coeff_384b);

    i = i.strict_add(DOUBLE_GROUP);
  }

  // Handle remaining triplets.
  let triple_aligned = (blocks.len() / 3) * 3;
  while i < triple_aligned {
    fold_block_128_width32_reflected_bitrev_bytes(&mut s0, &blocks[i], coeff_384b);
    fold_block_128_width32_reflected_bitrev_bytes(&mut s1, &blocks[i.strict_add(1)], coeff_384b);
    fold_block_128_width32_reflected_bitrev_bytes(&mut s2, &blocks[i.strict_add(2)], coeff_384b);
    i = i.strict_add(3);
  }

  // Merge: A^2·s0 ⊕ A·s1 ⊕ s2 (A = shift by 128B).
  let mut combined = s2;

  combined[0] ^= s1[0].fold_16_reflected(coeff_128b, zero);
  combined[1] ^= s1[1].fold_16_reflected(coeff_128b, zero);
  combined[2] ^= s1[2].fold_16_reflected(coeff_128b, zero);
  combined[3] ^= s1[3].fold_16_reflected(coeff_128b, zero);
  combined[4] ^= s1[4].fold_16_reflected(coeff_128b, zero);
  combined[5] ^= s1[5].fold_16_reflected(coeff_128b, zero);
  combined[6] ^= s1[6].fold_16_reflected(coeff_128b, zero);
  combined[7] ^= s1[7].fold_16_reflected(coeff_128b, zero);

  combined[0] ^= s0[0].fold_16_reflected(coeff_256b, zero);
  combined[1] ^= s0[1].fold_16_reflected(coeff_256b, zero);
  combined[2] ^= s0[2].fold_16_reflected(coeff_256b, zero);
  combined[3] ^= s0[3].fold_16_reflected(coeff_256b, zero);
  combined[4] ^= s0[4].fold_16_reflected(coeff_256b, zero);
  combined[5] ^= s0[5].fold_16_reflected(coeff_256b, zero);
  combined[6] ^= s0[6].fold_16_reflected(coeff_256b, zero);
  combined[7] ^= s0[7].fold_16_reflected(coeff_256b, zero);

  // Handle any remaining blocks sequentially.
  for block in &blocks[triple_aligned..] {
    fold_block_128_width32_reflected_bitrev_bytes(&mut combined, block, coeff_128b);
  }

  finalize_lanes_width32_reflected(combined, keys)
}

#[inline]
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn update_simd_width32_reflected_bitrev_bytes(
  state: u32,
  first: &[Simd; 8],
  rest: &[[Simd; 8]],
  keys: &[u64; 23],
) -> u32 {
  let mut x = *first;

  x[0] = x[0].bitrev_bytes();
  x[1] = x[1].bitrev_bytes();
  x[2] = x[2].bitrev_bytes();
  x[3] = x[3].bitrev_bytes();
  x[4] = x[4].bitrev_bytes();
  x[5] = x[5].bitrev_bytes();
  x[6] = x[6].bitrev_bytes();
  x[7] = x[7].bitrev_bytes();

  x[0] ^= Simd::new(0, state as u64);

  let coeff_128b = Simd::new(keys[4], keys[3]);
  for chunk in rest {
    let y0 = chunk[0].bitrev_bytes();
    let y1 = chunk[1].bitrev_bytes();
    let y2 = chunk[2].bitrev_bytes();
    let y3 = chunk[3].bitrev_bytes();
    let y4 = chunk[4].bitrev_bytes();
    let y5 = chunk[5].bitrev_bytes();
    let y6 = chunk[6].bitrev_bytes();
    let y7 = chunk[7].bitrev_bytes();

    x[0] = x[0].fold_16_reflected(coeff_128b, y0);
    x[1] = x[1].fold_16_reflected(coeff_128b, y1);
    x[2] = x[2].fold_16_reflected(coeff_128b, y2);
    x[3] = x[3].fold_16_reflected(coeff_128b, y3);
    x[4] = x[4].fold_16_reflected(coeff_128b, y4);
    x[5] = x[5].fold_16_reflected(coeff_128b, y5);
    x[6] = x[6].fold_16_reflected(coeff_128b, y6);
    x[7] = x[7].fold_16_reflected(coeff_128b, y7);
  }

  finalize_lanes_width32_reflected(x, keys)
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-stream kernel entry points
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn crc24_width32_pmull_small(mut state: u32, data: &[u8], keys: &[u64; 23]) -> u32 {
  let mut buf = data.as_ptr();
  let mut len = data.len();

  if len < 16 {
    return crc24_reflected_update_bitrev_bytes(state, data);
  }

  let coeff_16b = Simd::new(keys[2], keys[1]);

  let mut x0 = Simd::load(buf).bitrev_bytes();
  x0 ^= Simd::new(0, state as u64);
  buf = buf.add(16);
  len = len.strict_sub(16);

  while len >= 16 {
    let chunk = Simd::load(buf).bitrev_bytes();
    x0 = x0.fold_16_reflected(coeff_16b, chunk);
    buf = buf.add(16);
    len = len.strict_sub(16);
  }

  let x0 = x0.fold_width32_reflected(keys[6], keys[5]);
  state = x0.barrett_width32_reflected(keys[8], keys[7]);

  let tail = core::slice::from_raw_parts(buf, len);
  crc24_reflected_update_bitrev_bytes(state, tail)
}

#[inline]
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn crc24_width32_pmull(mut state: u32, data: &[u8], keys: &[u64; 23]) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd; 8]>();
  let Some((first, rest)) = middle.split_first() else {
    return crc24_width32_pmull_small(state, data, keys);
  };

  state = crc24_reflected_update_bitrev_bytes(state, left);
  let state32 = update_simd_width32_reflected_bitrev_bytes(state, first, rest, keys);
  state = state32;
  crc24_reflected_update_bitrev_bytes(state, right)
}

#[inline]
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn crc24_width32_pmull_2way(mut state: u32, data: &[u8], fold_256b: (u64, u64), keys: &[u64; 23]) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return crc24_width32_pmull_small(state, data, keys);
  }

  state = crc24_reflected_update_bitrev_bytes(state, left);
  if middle.len() >= 2 {
    state = update_simd_width32_reflected_bitrev_bytes_2way(state, middle, fold_256b, keys);
  } else if let Some((first, rest)) = middle.split_first() {
    state = update_simd_width32_reflected_bitrev_bytes(state, first, rest, keys);
  }
  crc24_reflected_update_bitrev_bytes(state, right)
}

#[inline]
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn crc24_width32_pmull_3way(
  mut state: u32,
  data: &[u8],
  fold_384b: (u64, u64),
  fold_256b: (u64, u64),
  keys: &[u64; 23],
) -> u32 {
  let (left, middle, right) = data.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return crc24_width32_pmull_small(state, data, keys);
  }

  state = crc24_reflected_update_bitrev_bytes(state, left);
  state = update_simd_width32_reflected_bitrev_bytes_3way(state, middle, fold_384b, fold_256b, keys);
  crc24_reflected_update_bitrev_bytes(state, right)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public Safe Kernel
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-24/OPENPGP PMULL kernel.
///
/// # Safety
///
/// Dispatcher verifies PMULL before selecting this kernel.
#[inline]
pub fn crc24_openpgp_pmull_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies PMULL before selecting this kernel.
  state = unsafe { crc24_width32_pmull(state, data, &CRC24_OPENPGP_KEYS_REFLECTED) };
  from_reflected_state(state)
}

/// CRC-24/OPENPGP PMULL small-buffer kernel.
///
/// Optimized for inputs smaller than a folding block (128 bytes).
///
/// # Safety
///
/// Dispatcher verifies PMULL before selecting this kernel.
#[inline]
pub fn crc24_openpgp_pmull_small_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies PMULL before selecting this kernel.
  state = unsafe { crc24_width32_pmull_small(state, data, &CRC24_OPENPGP_KEYS_REFLECTED) };
  from_reflected_state(state)
}

/// CRC-24/OPENPGP PMULL kernel (2-way striping).
///
/// # Safety
///
/// Dispatcher verifies PMULL before selecting this kernel.
#[inline]
pub fn crc24_openpgp_pmull_2way_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies PMULL before selecting this kernel.
  state = unsafe {
    crc24_width32_pmull_2way(
      state,
      data,
      CRC24_OPENPGP_STREAM_REFLECTED.fold_256b,
      &CRC24_OPENPGP_KEYS_REFLECTED,
    )
  };
  from_reflected_state(state)
}

/// CRC-24/OPENPGP PMULL kernel (3-way striping).
///
/// # Safety
///
/// Dispatcher verifies PMULL before selecting this kernel.
#[inline]
pub fn crc24_openpgp_pmull_3way_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies PMULL before selecting this kernel.
  state = unsafe {
    crc24_width32_pmull_3way(
      state,
      data,
      CRC24_OPENPGP_STREAM_REFLECTED.fold_384b,
      CRC24_OPENPGP_STREAM_REFLECTED.fold_256b,
      &CRC24_OPENPGP_KEYS_REFLECTED,
    )
  };
  from_reflected_state(state)
}
