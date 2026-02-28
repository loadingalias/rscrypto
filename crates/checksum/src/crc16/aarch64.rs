//! aarch64 carryless-multiply CRC-16 kernels (PMULL).
//!
//! These kernels implement reflected CRC-16 polynomials by lifting the 16-bit
//! state into the "width32" folding/reduction strategy (same structure as the
//! CRC-32 folding kernels, but with CRC-16-specific constants).
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

use super::keys::{
  CRC16_CCITT_KEYS_REFLECTED, CRC16_CCITT_STREAM_REFLECTED, CRC16_IBM_KEYS_REFLECTED, CRC16_IBM_STREAM_REFLECTED,
};

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
unsafe fn fold_block_128_width32_reflected(x: &mut [Simd; 8], chunk: &[Simd; 8], coeff: Simd) {
  x[0] = x[0].fold_16_reflected(coeff, chunk[0]);
  x[1] = x[1].fold_16_reflected(coeff, chunk[1]);
  x[2] = x[2].fold_16_reflected(coeff, chunk[2]);
  x[3] = x[3].fold_16_reflected(coeff, chunk[3]);
  x[4] = x[4].fold_16_reflected(coeff, chunk[4]);
  x[5] = x[5].fold_16_reflected(coeff, chunk[5]);
  x[6] = x[6].fold_16_reflected(coeff, chunk[6]);
  x[7] = x[7].fold_16_reflected(coeff, chunk[7]);
}

#[target_feature(enable = "aes", enable = "neon")]
unsafe fn update_simd_width32_reflected_2way(
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

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];

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
    fold_block_128_width32_reflected(&mut s0, &blocks[i], coeff_256b);
    fold_block_128_width32_reflected(&mut s1, &blocks[i.strict_add(1)], coeff_256b);

    // Second iteration (blocks i+2, i+3)
    fold_block_128_width32_reflected(&mut s0, &blocks[i.strict_add(2)], coeff_256b);
    fold_block_128_width32_reflected(&mut s1, &blocks[i.strict_add(3)], coeff_256b);

    i = i.strict_add(DOUBLE_GROUP);
  }

  // Handle remaining pairs.
  let even = blocks.len() & !1usize;
  while i < even {
    fold_block_128_width32_reflected(&mut s0, &blocks[i], coeff_256b);
    fold_block_128_width32_reflected(&mut s1, &blocks[i.strict_add(1)], coeff_256b);
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
    fold_block_128_width32_reflected(&mut combined, &blocks[even], coeff_128b);
  }

  finalize_lanes_width32_reflected(combined, keys)
}

#[target_feature(enable = "aes", enable = "neon")]
unsafe fn update_simd_width32_reflected_3way(
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
    return update_simd_width32_reflected(state, first, rest, keys);
  }

  let coeff_384b = Simd::new(fold_384b.0, fold_384b.1);
  let coeff_256b = Simd::new(fold_256b.0, fold_256b.1);
  let coeff_128b = Simd::new(keys[4], keys[3]);
  let zero = Simd::new(0, 0);

  let mut s0 = blocks[0];
  let mut s1 = blocks[1];
  let mut s2 = blocks[2];

  // Inject CRC into stream 0 (block 0).
  s0[0] ^= Simd::new(0, state as u64);

  // Double-unrolled main loop: process 6 blocks (768B) per iteration.
  const BLOCK_SIZE: usize = 128;
  const DOUBLE_GROUP: usize = 6; // 2 × 3-way = 6 blocks = 768B
  const PREFETCH_BLOCKS: usize = LARGE_BLOCK_DISTANCE / BLOCK_SIZE;

  let blocks_ptr = blocks.as_ptr();
  let blocks_end = blocks_ptr.add(blocks.len());
  let mut ptr = blocks_ptr.add(3);
  let double_end = blocks_ptr.add(3 + ((blocks.len() - 3) / DOUBLE_GROUP) * DOUBLE_GROUP);

  while ptr < double_end {
    let prefetch_ptr = ptr.add(PREFETCH_BLOCKS);
    if prefetch_ptr < blocks_end {
      prefetch_read_l1(prefetch_ptr.cast::<u8>());
    }

    // First iteration (blocks i, i+1, i+2)
    fold_block_128_width32_reflected(&mut s0, &*ptr, coeff_384b);
    fold_block_128_width32_reflected(&mut s1, &*ptr.add(1), coeff_384b);
    fold_block_128_width32_reflected(&mut s2, &*ptr.add(2), coeff_384b);

    // Second iteration (blocks i+3, i+4, i+5)
    fold_block_128_width32_reflected(&mut s0, &*ptr.add(3), coeff_384b);
    fold_block_128_width32_reflected(&mut s1, &*ptr.add(4), coeff_384b);
    fold_block_128_width32_reflected(&mut s2, &*ptr.add(5), coeff_384b);

    ptr = ptr.add(DOUBLE_GROUP);
  }

  // Handle remaining triplets.
  let triple_end = blocks_ptr.add((blocks.len() / 3) * 3);
  while ptr < triple_end {
    fold_block_128_width32_reflected(&mut s0, &*ptr, coeff_384b);
    fold_block_128_width32_reflected(&mut s1, &*ptr.add(1), coeff_384b);
    fold_block_128_width32_reflected(&mut s2, &*ptr.add(2), coeff_384b);
    ptr = ptr.add(3);
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
  while ptr < blocks_end {
    fold_block_128_width32_reflected(&mut combined, &*ptr, coeff_128b);
    ptr = ptr.add(1);
  }

  finalize_lanes_width32_reflected(combined, keys)
}

#[inline]
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn update_simd_width32_reflected(state: u32, first: &[Simd; 8], rest: &[[Simd; 8]], keys: &[u64; 23]) -> u32 {
  use crate::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  let mut x = *first;

  x[0] ^= Simd::new(0, state as u64);

  let coeff_128b = Simd::new(keys[4], keys[3]);

  // Double-unrolled folding for large buffers improves ILP on Neoverse-class cores.
  const BLOCK_SIZE: usize = 128;
  const DOUBLE_GROUP: usize = 2; // 2 × 1-way = 2 blocks = 256B
  const PREFETCH_BLOCKS: usize = LARGE_BLOCK_DISTANCE / BLOCK_SIZE;

  let rest_ptr = rest.as_ptr();
  let rest_end = rest_ptr.add(rest.len());
  let mut ptr = rest_ptr;
  let double_end = rest_ptr.add((rest.len() / DOUBLE_GROUP) * DOUBLE_GROUP);

  while ptr < double_end {
    let prefetch_ptr = ptr.add(PREFETCH_BLOCKS);
    if prefetch_ptr < rest_end {
      prefetch_read_l1(prefetch_ptr.cast::<u8>());
    }

    let chunk0 = &*ptr;
    x[0] = x[0].fold_16_reflected(coeff_128b, chunk0[0]);
    x[1] = x[1].fold_16_reflected(coeff_128b, chunk0[1]);
    x[2] = x[2].fold_16_reflected(coeff_128b, chunk0[2]);
    x[3] = x[3].fold_16_reflected(coeff_128b, chunk0[3]);
    x[4] = x[4].fold_16_reflected(coeff_128b, chunk0[4]);
    x[5] = x[5].fold_16_reflected(coeff_128b, chunk0[5]);
    x[6] = x[6].fold_16_reflected(coeff_128b, chunk0[6]);
    x[7] = x[7].fold_16_reflected(coeff_128b, chunk0[7]);

    let chunk1 = &*ptr.add(1);
    x[0] = x[0].fold_16_reflected(coeff_128b, chunk1[0]);
    x[1] = x[1].fold_16_reflected(coeff_128b, chunk1[1]);
    x[2] = x[2].fold_16_reflected(coeff_128b, chunk1[2]);
    x[3] = x[3].fold_16_reflected(coeff_128b, chunk1[3]);
    x[4] = x[4].fold_16_reflected(coeff_128b, chunk1[4]);
    x[5] = x[5].fold_16_reflected(coeff_128b, chunk1[5]);
    x[6] = x[6].fold_16_reflected(coeff_128b, chunk1[6]);
    x[7] = x[7].fold_16_reflected(coeff_128b, chunk1[7]);

    ptr = ptr.add(DOUBLE_GROUP);
  }

  while ptr < rest_end {
    let chunk = &*ptr;
    x[0] = x[0].fold_16_reflected(coeff_128b, chunk[0]);
    x[1] = x[1].fold_16_reflected(coeff_128b, chunk[1]);
    x[2] = x[2].fold_16_reflected(coeff_128b, chunk[2]);
    x[3] = x[3].fold_16_reflected(coeff_128b, chunk[3]);
    x[4] = x[4].fold_16_reflected(coeff_128b, chunk[4]);
    x[5] = x[5].fold_16_reflected(coeff_128b, chunk[5]);
    x[6] = x[6].fold_16_reflected(coeff_128b, chunk[6]);
    x[7] = x[7].fold_16_reflected(coeff_128b, chunk[7]);
    ptr = ptr.add(1);
  }

  finalize_lanes_width32_reflected(x, keys)
}

#[inline]
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn crc16_width32_pmull_small(
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

  let coeff_16b = Simd::new(keys[2], keys[1]);

  let mut x0 = Simd::load(buf);
  x0 ^= Simd::new(0, state as u64);
  buf = buf.add(16);
  len = len.strict_sub(16);

  while len >= 16 {
    let chunk = Simd::load(buf);
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
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn crc16_width32_pmull(mut state: u16, data: &[u8], keys: &[u64; 23], portable: fn(u16, &[u8]) -> u16) -> u16 {
  let (left, middle, right) = data.align_to::<[Simd; 8]>();
  let Some((first, rest)) = middle.split_first() else {
    return crc16_width32_pmull_small(state, data, keys, portable);
  };

  state = portable(state, left);
  let state32 = update_simd_width32_reflected(state as u32, first, rest, keys);
  state = state32 as u16;
  portable(state, right)
}

#[inline]
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn crc16_width32_pmull_2way(
  mut state: u16,
  data: &[u8],
  keys: &[u64; 23],
  fold_256b: (u64, u64),
  portable: fn(u16, &[u8]) -> u16,
) -> u16 {
  let (left, middle, right) = data.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return crc16_width32_pmull_small(state, data, keys, portable);
  }

  state = portable(state, left);
  let state32 = if middle.len() >= 2 {
    update_simd_width32_reflected_2way(state as u32, middle, fold_256b, keys)
  } else {
    let Some((first, rest)) = middle.split_first() else {
      return crc16_width32_pmull_small(state, data, keys, portable);
    };
    update_simd_width32_reflected(state as u32, first, rest, keys)
  };
  state = state32 as u16;
  portable(state, right)
}

#[inline]
#[target_feature(enable = "aes", enable = "neon")]
unsafe fn crc16_width32_pmull_3way(
  mut state: u16,
  data: &[u8],
  keys: &[u64; 23],
  fold_384b: (u64, u64),
  fold_256b: (u64, u64),
  portable: fn(u16, &[u8]) -> u16,
) -> u16 {
  let (left, middle, right) = data.align_to::<[Simd; 8]>();
  if middle.is_empty() {
    return crc16_width32_pmull_small(state, data, keys, portable);
  }

  state = portable(state, left);
  let state32 = update_simd_width32_reflected_3way(state as u32, middle, fold_384b, fold_256b, keys);
  state = state32 as u16;
  portable(state, right)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public Safe Kernels (matching CRC-64 pure fn(u16, &[u8]) -> u16 signature)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-16/CCITT PMULL kernel.
///
/// # Safety
///
/// Dispatcher verifies PMULL before selecting this kernel.
#[inline]
pub fn crc16_ccitt_pmull_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies PMULL before selecting this kernel.
  unsafe {
    crc16_width32_pmull(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/CCITT PMULL small-buffer kernel.
///
/// Optimized for inputs smaller than a folding block (128 bytes).
///
/// # Safety
///
/// Dispatcher verifies PMULL before selecting this kernel.
#[inline]
pub fn crc16_ccitt_pmull_small_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies PMULL before selecting this kernel.
  unsafe {
    crc16_width32_pmull_small(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/CCITT PMULL kernel (2-way striping).
///
/// # Safety
///
/// Dispatcher verifies PMULL before selecting this kernel.
#[inline]
pub fn crc16_ccitt_pmull_2way_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies PMULL before selecting this kernel.
  unsafe {
    crc16_width32_pmull_2way(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      CRC16_CCITT_STREAM_REFLECTED.fold_256b,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/CCITT PMULL kernel (3-way striping).
///
/// # Safety
///
/// Dispatcher verifies PMULL before selecting this kernel.
#[inline]
pub fn crc16_ccitt_pmull_3way_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies PMULL before selecting this kernel.
  unsafe {
    crc16_width32_pmull_3way(
      crc,
      data,
      &CRC16_CCITT_KEYS_REFLECTED,
      CRC16_CCITT_STREAM_REFLECTED.fold_384b,
      CRC16_CCITT_STREAM_REFLECTED.fold_256b,
      super::portable::crc16_ccitt_slice8,
    )
  }
}

/// CRC-16/IBM PMULL kernel.
///
/// # Safety
///
/// Dispatcher verifies PMULL before selecting this kernel.
#[inline]
pub fn crc16_ibm_pmull_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies PMULL before selecting this kernel.
  unsafe { crc16_width32_pmull(crc, data, &CRC16_IBM_KEYS_REFLECTED, super::portable::crc16_ibm_slice8) }
}

/// CRC-16/IBM PMULL small-buffer kernel.
///
/// Optimized for inputs smaller than a folding block (128 bytes).
///
/// # Safety
///
/// Dispatcher verifies PMULL before selecting this kernel.
#[inline]
pub fn crc16_ibm_pmull_small_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies PMULL before selecting this kernel.
  unsafe { crc16_width32_pmull_small(crc, data, &CRC16_IBM_KEYS_REFLECTED, super::portable::crc16_ibm_slice8) }
}

/// CRC-16/IBM PMULL kernel (2-way striping).
///
/// # Safety
///
/// Dispatcher verifies PMULL before selecting this kernel.
#[inline]
pub fn crc16_ibm_pmull_2way_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies PMULL before selecting this kernel.
  unsafe {
    crc16_width32_pmull_2way(
      crc,
      data,
      &CRC16_IBM_KEYS_REFLECTED,
      CRC16_IBM_STREAM_REFLECTED.fold_256b,
      super::portable::crc16_ibm_slice8,
    )
  }
}

/// CRC-16/IBM PMULL kernel (3-way striping).
///
/// # Safety
///
/// Dispatcher verifies PMULL before selecting this kernel.
#[inline]
pub fn crc16_ibm_pmull_3way_safe(crc: u16, data: &[u8]) -> u16 {
  // SAFETY: Dispatcher verifies PMULL before selecting this kernel.
  unsafe {
    crc16_width32_pmull_3way(
      crc,
      data,
      &CRC16_IBM_KEYS_REFLECTED,
      CRC16_IBM_STREAM_REFLECTED.fold_384b,
      CRC16_IBM_STREAM_REFLECTED.fold_256b,
      super::portable::crc16_ibm_slice8,
    )
  }
}
