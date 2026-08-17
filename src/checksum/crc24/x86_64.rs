//! x86_64 carryless-multiply CRC-24/OPENPGP kernels (PCLMULQDQ).
//!
//! CRC-24/OPENPGP is MSB-first in the portable implementation. These kernels
//! reuse the existing "width32" folding/reduction structure by computing the
//! equivalent **reflected** CRC-24 over per-byte bit-reversed input.
//!
//! This keeps the core folding code identical to CRC-16 (width32) and CRC-32,
//! while still producing the correct MSB-first OpenPGP checksum.
//!
//! # Safety
//!
//! The baseline kernels require SSE2, SSSE3, and PCLMULQDQ. The wide kernels
//! additionally require AVX-512F/VL/BW/DQ and VPCLMULQDQ. The private safe
//! wrappers are installed only through capability-gated dispatcher tables.

use core::{
  arch::x86_64::*,
  ops::{BitXor, BitXorAssign},
};

use super::{
  keys::{CRC24_OPENPGP_KEYS_REFLECTED, CRC24_OPENPGP_STREAM_REFLECTED, Crc24StreamConstants},
  reflected::{crc24_reflected_update_bitrev_bytes, from_reflected_state, to_reflected_state},
};

// SIMD helpers

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
  /// Loads 16 bytes without requiring alignment.
  ///
  /// # Safety
  ///
  /// `ptr` must be valid to read 16 initialized bytes. The source may be
  /// unaligned because the bytes are copied into aligned local storage.
  #[inline]
  unsafe fn load_unaligned(ptr: *const u8) -> Self {
    let mut value = core::mem::MaybeUninit::<Self>::uninit();

    // SAFETY: The caller guarantees a readable 16-byte source. `value` is an
    // aligned, non-overlapping 16-byte destination, and every bit pattern is
    // valid for the integer vector inside `Simd128`.
    unsafe {
      core::ptr::copy_nonoverlapping(ptr, value.as_mut_ptr().cast::<u8>(), 16);
      value.assume_init()
    }
  }

  /// Creates a vector from its high and low 64-bit lanes.
  ///
  /// # Safety
  ///
  /// The current CPU must support SSE2.
  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn new(high: u64, low: u64) -> Self {
    Self(_mm_set_epi64x(high.cast_signed(), low.cast_signed()))
  }

  /// Shifts the vector right by eight bytes, filling the high bytes with zero.
  ///
  /// # Safety
  ///
  /// The current CPU must support SSE2.
  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn shift_right_8(self) -> Self {
    Self(_mm_srli_si128::<8>(self.0))
  }

  /// Shifts the vector left by 12 bytes, filling the low bytes with zero.
  ///
  /// # Safety
  ///
  /// The current CPU must support SSE2.
  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn shift_left_12(self) -> Self {
    Self(_mm_slli_si128::<12>(self.0))
  }

  /// Computes the bitwise AND of two vectors.
  ///
  /// # Safety
  ///
  /// The current CPU must support SSE2.
  #[inline]
  #[target_feature(enable = "sse2")]
  unsafe fn and(self, mask: Self) -> Self {
    Self(_mm_and_si128(self.0, mask.0))
  }

  /// Reverses the bits within each byte.
  ///
  /// # Safety
  ///
  /// The current CPU must support SSE2 and SSSE3.
  #[inline]
  #[target_feature(enable = "sse2", enable = "ssse3")]
  unsafe fn bitrev_bytes(self) -> Self {
    let mask = _mm_set1_epi8(0x0f);
    let lo = _mm_and_si128(self.0, mask);
    let hi = _mm_and_si128(_mm_srli_epi16::<4>(self.0), mask);

    // Nibble bit-reversal LUT: 0b0000..0b1111 -> reversed bits.
    let lut = _mm_setr_epi8(0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15);
    let lo_rev = _mm_shuffle_epi8(lut, lo);
    let hi_rev = _mm_shuffle_epi8(lut, hi);
    let lo_shift = _mm_slli_epi16::<4>(lo_rev);
    Self(_mm_or_si128(lo_shift, hi_rev))
  }

  /// Folds one reflected 16-byte lane and XORs the supplied input lane.
  ///
  /// # Safety
  ///
  /// The current CPU must support SSE2 and PCLMULQDQ.
  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn fold_16_reflected(self, coeff: Self, data_to_xor: Self) -> Self {
    let h = _mm_clmulepi64_si128::<0x10>(self.0, coeff.0);
    let l = _mm_clmulepi64_si128::<0x01>(self.0, coeff.0);
    Self(_mm_xor_si128(_mm_xor_si128(h, l), data_to_xor.0))
  }

  /// Folds a reflected CRC state from 128 bits to the width-32 reduction state.
  ///
  /// # Safety
  ///
  /// The current CPU must support SSE2 and PCLMULQDQ.
  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn fold_width32_reflected(self, high: u64, low: u64) -> Self {
    // SAFETY: All intrinsics and target-feature-gated calls require SSE2+PCLMULQDQ, ensured by this
    // function's #[target_feature] attribute.
    unsafe {
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
  }

  /// Applies Barrett reduction and returns the width-32 state.
  ///
  /// # Safety
  ///
  /// The current CPU must support SSE2 and PCLMULQDQ.
  #[inline]
  #[target_feature(enable = "sse2", enable = "pclmulqdq")]
  unsafe fn barrett_width32_reflected(self, poly: u64, mu: u64) -> u32 {
    // SAFETY: All intrinsics and target-feature-gated calls require SSE2+PCLMULQDQ, ensured by this
    // function's #[target_feature] attribute.
    unsafe {
      let polymu = Self::new(poly, mu);
      let clmul1 = _mm_clmulepi64_si128::<0x00>(self.0, polymu.0);
      let clmul2 = _mm_clmulepi64_si128::<0x10>(clmul1, polymu.0);
      let xorred = _mm_xor_si128(self.0, clmul2);

      let hi = _mm_srli_si128::<8>(xorred);
      _mm_cvtsi128_si32(hi).cast_unsigned()
    }
  }
}

// 8-lane width32 update (128B blocks)

/// Combines eight folded lanes and applies width-32 Barrett reduction.
///
/// # Safety
///
/// The current CPU must support SSE2 and PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2", enable = "pclmulqdq")]
unsafe fn finalize_lanes_width32_reflected(x: [Simd128; 8], keys: &[u64; 23]) -> u32 {
  // SAFETY: All intrinsics and target-feature-gated calls require SSE2+PCLMULQDQ, ensured by this
  // function's #[target_feature] attribute.
  unsafe {
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
}

/// Bit-reverses and folds one or more 128-byte blocks into a width-32 CRC state.
///
/// # Safety
///
/// The current CPU must support SSE2, SSSE3, and PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn update_simd_width32_reflected_bitrev_bytes(
  state: u32,
  first: &[Simd128; 8],
  rest: &[[Simd128; 8]],
  keys: &[u64; 23],
) -> u32 {
  // SAFETY: All intrinsics and target-feature-gated calls require SSE2+SSSE3+PCLMULQDQ, ensured by
  // this function's #[target_feature] attribute.
  unsafe {
    let mut x = *first;

    x[0] = x[0].bitrev_bytes();
    x[1] = x[1].bitrev_bytes();
    x[2] = x[2].bitrev_bytes();
    x[3] = x[3].bitrev_bytes();
    x[4] = x[4].bitrev_bytes();
    x[5] = x[5].bitrev_bytes();
    x[6] = x[6].bitrev_bytes();
    x[7] = x[7].bitrev_bytes();

    x[0] ^= Simd128::new(0, state as u64);

    let coeff_128b = Simd128::new(keys[4], keys[3]);
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
}

// PCLMULQDQ multi-stream (2/4/7/8-way, 128B blocks)

/// Bit-reverses and folds one 128-byte block into eight parallel lanes.
///
/// # Safety
///
/// The current CPU must support SSE2, SSSE3, and PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn fold_block_128_reflected_bitrev(x: &mut [Simd128; 8], chunk: &[Simd128; 8], coeff: Simd128) {
  // SAFETY: All intrinsics and target-feature-gated calls require SSE2+SSSE3+PCLMULQDQ, ensured by
  // this function's #[target_feature] attribute.
  unsafe {
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
}

/// Bit-reverses and folds 128-byte blocks through two PCLMULQDQ streams.
///
/// # Safety
///
/// The current CPU must support SSE2, SSSE3, and PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn update_simd_width32_reflected_bitrev_bytes_2way(
  state: u32,
  blocks: &[[Simd128; 8]],
  fold_256b: (u64, u64),
  keys: &[u64; 23],
) -> u32 {
  use crate::checksum::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  // SAFETY: All intrinsics and target-feature-gated calls require SSE2+SSSE3+PCLMULQDQ, ensured by
  // this function's #[target_feature] attribute. align_to is sound because Simd128 is
  // repr(transparent) over the SIMD type.
  unsafe {
    debug_assert!(!blocks.is_empty());

    if blocks.len() < 2 {
      let Some((first, rest)) = blocks.split_first() else {
        return state;
      };
      return update_simd_width32_reflected_bitrev_bytes(state, first, rest, keys);
    }

    let coeff_256 = Simd128::new(fold_256b.0, fold_256b.1);
    let coeff_128 = Simd128::new(keys[4], keys[3]);

    let mut s0 = blocks[0];
    let mut s1 = blocks[1];

    s0[0] = s0[0].bitrev_bytes();
    s0[1] = s0[1].bitrev_bytes();
    s0[2] = s0[2].bitrev_bytes();
    s0[3] = s0[3].bitrev_bytes();
    s0[4] = s0[4].bitrev_bytes();
    s0[5] = s0[5].bitrev_bytes();
    s0[6] = s0[6].bitrev_bytes();
    s0[7] = s0[7].bitrev_bytes();

    s1[0] = s1[0].bitrev_bytes();
    s1[1] = s1[1].bitrev_bytes();
    s1[2] = s1[2].bitrev_bytes();
    s1[3] = s1[3].bitrev_bytes();
    s1[4] = s1[4].bitrev_bytes();
    s1[5] = s1[5].bitrev_bytes();
    s1[6] = s1[6].bitrev_bytes();
    s1[7] = s1[7].bitrev_bytes();

    s0[0] ^= Simd128::new(0, state as u64);

    // Double-unrolled main loop: process 4 blocks (512B) per iteration.
    const BLOCK_SIZE: usize = 128;
    const DOUBLE_GROUP: usize = 4; // 2 × 2-way = 4 blocks = 512B

    let mut i: usize = 2;
    let aligned = blocks.len().strict_sub(blocks.len().strict_rem(DOUBLE_GROUP));

    while i.strict_add(DOUBLE_GROUP) <= aligned {
      let prefetch_idx = i.strict_add(LARGE_BLOCK_DISTANCE / BLOCK_SIZE);
      if prefetch_idx < blocks.len() {
        prefetch_read_l1(blocks[prefetch_idx].as_ptr().cast::<u8>());
      }

      // First iteration (blocks i, i+1)
      fold_block_128_reflected_bitrev(&mut s0, &blocks[i], coeff_256);
      fold_block_128_reflected_bitrev(&mut s1, &blocks[i.strict_add(1)], coeff_256);

      // Second iteration (blocks i+2, i+3)
      fold_block_128_reflected_bitrev(&mut s0, &blocks[i.strict_add(2)], coeff_256);
      fold_block_128_reflected_bitrev(&mut s1, &blocks[i.strict_add(3)], coeff_256);

      i = i.strict_add(DOUBLE_GROUP);
    }

    // Handle remaining pairs.
    let even = blocks.len() & !1usize;
    while i < even {
      fold_block_128_reflected_bitrev(&mut s0, &blocks[i], coeff_256);
      fold_block_128_reflected_bitrev(&mut s1, &blocks[i.strict_add(1)], coeff_256);
      i = i.strict_add(2);
    }

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
      fold_block_128_reflected_bitrev(&mut combined, &blocks[even], coeff_128);
    }

    finalize_lanes_width32_reflected(combined, keys)
  }
}

/// Bit-reverses and folds 128-byte blocks through four PCLMULQDQ streams.
///
/// # Safety
///
/// The current CPU must support SSE2, SSSE3, and PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn update_simd_width32_reflected_bitrev_bytes_4way(
  state: u32,
  blocks: &[[Simd128; 8]],
  fold_512b: (u64, u64),
  combine: &[(u64, u64); 3],
  keys: &[u64; 23],
) -> u32 {
  use crate::checksum::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  // SAFETY: All intrinsics and target-feature-gated calls require SSE2+SSSE3+PCLMULQDQ, ensured by
  // this function's #[target_feature] attribute.
  unsafe {
    debug_assert!(!blocks.is_empty());

    if blocks.len() < 4 {
      let Some((first, rest)) = blocks.split_first() else {
        return state;
      };
      return update_simd_width32_reflected_bitrev_bytes(state, first, rest, keys);
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

    for s in [&mut s0, &mut s1, &mut s2, &mut s3] {
      s[0] = s[0].bitrev_bytes();
      s[1] = s[1].bitrev_bytes();
      s[2] = s[2].bitrev_bytes();
      s[3] = s[3].bitrev_bytes();
      s[4] = s[4].bitrev_bytes();
      s[5] = s[5].bitrev_bytes();
      s[6] = s[6].bitrev_bytes();
      s[7] = s[7].bitrev_bytes();
    }

    s0[0] ^= Simd128::new(0, state as u64);

    // Double-unrolled main loop: process 8 blocks (1KB) per iteration.
    const BLOCK_SIZE: usize = 128;
    const DOUBLE_GROUP: usize = 8; // 2 × 4-way = 8 blocks = 1KB

    let mut i: usize = 4;
    let aligned = blocks.len().strict_sub(blocks.len().strict_rem(DOUBLE_GROUP));

    while i.strict_add(DOUBLE_GROUP) <= aligned {
      let prefetch_idx = i.strict_add(LARGE_BLOCK_DISTANCE / BLOCK_SIZE);
      if prefetch_idx < blocks.len() {
        prefetch_read_l1(blocks[prefetch_idx].as_ptr().cast::<u8>());
      }

      // First iteration (blocks i..i+3)
      fold_block_128_reflected_bitrev(&mut s0, &blocks[i], coeff_512);
      fold_block_128_reflected_bitrev(&mut s1, &blocks[i.strict_add(1)], coeff_512);
      fold_block_128_reflected_bitrev(&mut s2, &blocks[i.strict_add(2)], coeff_512);
      fold_block_128_reflected_bitrev(&mut s3, &blocks[i.strict_add(3)], coeff_512);

      // Second iteration (blocks i+4..i+7)
      fold_block_128_reflected_bitrev(&mut s0, &blocks[i.strict_add(4)], coeff_512);
      fold_block_128_reflected_bitrev(&mut s1, &blocks[i.strict_add(5)], coeff_512);
      fold_block_128_reflected_bitrev(&mut s2, &blocks[i.strict_add(6)], coeff_512);
      fold_block_128_reflected_bitrev(&mut s3, &blocks[i.strict_add(7)], coeff_512);

      i = i.strict_add(DOUBLE_GROUP);
    }

    // Handle remaining quads.
    let quad_aligned = blocks.len().strict_sub(blocks.len().strict_rem(4));
    while i < quad_aligned {
      fold_block_128_reflected_bitrev(&mut s0, &blocks[i], coeff_512);
      fold_block_128_reflected_bitrev(&mut s1, &blocks[i.strict_add(1)], coeff_512);
      fold_block_128_reflected_bitrev(&mut s2, &blocks[i.strict_add(2)], coeff_512);
      fold_block_128_reflected_bitrev(&mut s3, &blocks[i.strict_add(3)], coeff_512);
      i = i.strict_add(4);
    }

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

    for block in &blocks[quad_aligned..] {
      fold_block_128_reflected_bitrev(&mut acc, block, coeff_128);
    }

    finalize_lanes_width32_reflected(acc, keys)
  }
}

/// Bit-reverses and folds 128-byte blocks through seven PCLMULQDQ streams.
///
/// # Safety
///
/// The current CPU must support SSE2, SSSE3, and PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn update_simd_width32_reflected_bitrev_bytes_7way(
  state: u32,
  blocks: &[[Simd128; 8]],
  fold_896b: (u64, u64),
  combine: &[(u64, u64); 6],
  keys: &[u64; 23],
) -> u32 {
  use crate::checksum::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  // SAFETY: All intrinsics and target-feature-gated calls require SSE2+SSSE3+PCLMULQDQ, ensured by
  // this function's #[target_feature] attribute.
  unsafe {
    debug_assert!(!blocks.is_empty());

    if blocks.len() < 7 {
      let Some((first, rest)) = blocks.split_first() else {
        return state;
      };
      return update_simd_width32_reflected_bitrev_bytes(state, first, rest, keys);
    }

    let aligned = blocks.len().strict_sub(blocks.len().strict_rem(7));

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

    for s in [&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, &mut s6] {
      s[0] = s[0].bitrev_bytes();
      s[1] = s[1].bitrev_bytes();
      s[2] = s[2].bitrev_bytes();
      s[3] = s[3].bitrev_bytes();
      s[4] = s[4].bitrev_bytes();
      s[5] = s[5].bitrev_bytes();
      s[6] = s[6].bitrev_bytes();
      s[7] = s[7].bitrev_bytes();
    }

    s0[0] ^= Simd128::new(0, state as u64);

    const BLOCK_SIZE: usize = 128;

    let mut i: usize = 7;
    while i < aligned {
      let prefetch_idx = i.strict_add(LARGE_BLOCK_DISTANCE / BLOCK_SIZE);
      if prefetch_idx < blocks.len() {
        prefetch_read_l1(blocks[prefetch_idx].as_ptr().cast::<u8>());
      }

      fold_block_128_reflected_bitrev(&mut s0, &blocks[i], coeff_896);
      fold_block_128_reflected_bitrev(&mut s1, &blocks[i.strict_add(1)], coeff_896);
      fold_block_128_reflected_bitrev(&mut s2, &blocks[i.strict_add(2)], coeff_896);
      fold_block_128_reflected_bitrev(&mut s3, &blocks[i.strict_add(3)], coeff_896);
      fold_block_128_reflected_bitrev(&mut s4, &blocks[i.strict_add(4)], coeff_896);
      fold_block_128_reflected_bitrev(&mut s5, &blocks[i.strict_add(5)], coeff_896);
      fold_block_128_reflected_bitrev(&mut s6, &blocks[i.strict_add(6)], coeff_896);
      i = i.strict_add(7);
    }

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
      fold_block_128_reflected_bitrev(&mut acc, block, coeff_128);
    }

    finalize_lanes_width32_reflected(acc, keys)
  }
}

/// Bit-reverses and folds 128-byte blocks through eight PCLMULQDQ streams.
///
/// # Safety
///
/// The current CPU must support SSE2, SSSE3, and PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn update_simd_width32_reflected_bitrev_bytes_8way(
  state: u32,
  blocks: &[[Simd128; 8]],
  fold_1024b: (u64, u64),
  combine: &[(u64, u64); 7],
  keys: &[u64; 23],
) -> u32 {
  use crate::checksum::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  // SAFETY: All intrinsics and target-feature-gated calls require SSE2+SSSE3+PCLMULQDQ, ensured by
  // this function's #[target_feature] attribute.
  unsafe {
    debug_assert!(!blocks.is_empty());

    if blocks.len() < 8 {
      let Some((first, rest)) = blocks.split_first() else {
        return state;
      };
      return update_simd_width32_reflected_bitrev_bytes(state, first, rest, keys);
    }

    let aligned = blocks.len().strict_sub(blocks.len().strict_rem(8));

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

    for s in [&mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, &mut s6, &mut s7] {
      s[0] = s[0].bitrev_bytes();
      s[1] = s[1].bitrev_bytes();
      s[2] = s[2].bitrev_bytes();
      s[3] = s[3].bitrev_bytes();
      s[4] = s[4].bitrev_bytes();
      s[5] = s[5].bitrev_bytes();
      s[6] = s[6].bitrev_bytes();
      s[7] = s[7].bitrev_bytes();
    }

    s0[0] ^= Simd128::new(0, state as u64);

    const BLOCK_SIZE: usize = 128;

    let mut i: usize = 8;
    while i < aligned {
      let prefetch_idx = i.strict_add(LARGE_BLOCK_DISTANCE / BLOCK_SIZE);
      if prefetch_idx < blocks.len() {
        prefetch_read_l1(blocks[prefetch_idx].as_ptr().cast::<u8>());
      }

      fold_block_128_reflected_bitrev(&mut s0, &blocks[i], coeff_1024);
      fold_block_128_reflected_bitrev(&mut s1, &blocks[i.strict_add(1)], coeff_1024);
      fold_block_128_reflected_bitrev(&mut s2, &blocks[i.strict_add(2)], coeff_1024);
      fold_block_128_reflected_bitrev(&mut s3, &blocks[i.strict_add(3)], coeff_1024);
      fold_block_128_reflected_bitrev(&mut s4, &blocks[i.strict_add(4)], coeff_1024);
      fold_block_128_reflected_bitrev(&mut s5, &blocks[i.strict_add(5)], coeff_1024);
      fold_block_128_reflected_bitrev(&mut s6, &blocks[i.strict_add(6)], coeff_1024);
      fold_block_128_reflected_bitrev(&mut s7, &blocks[i.strict_add(7)], coeff_1024);
      i = i.strict_add(8);
    }

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
      fold_block_128_reflected_bitrev(&mut acc, block, coeff_128);
    }

    finalize_lanes_width32_reflected(acc, keys)
  }
}

/// Updates CRC-24 with a selected multi-stream PCLMULQDQ kernel.
///
/// # Safety
///
/// The current CPU must support SSE2, SSSE3, and PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn crc24_width32_pclmul_stream(
  mut state: u32,
  data: &[u8],
  keys: &[u64; 23],
  stream: &Crc24StreamConstants,
  streams: u8,
) -> u32 {
  // SAFETY: All intrinsics and target-feature-gated calls require SSE2+SSSE3+PCLMULQDQ, ensured by
  // this function's #[target_feature] attribute. align_to is sound because Simd128 is
  // repr(transparent) over the SIMD type.
  unsafe {
    let (left, middle, right) = data.align_to::<[Simd128; 8]>();
    let Some((first, rest)) = middle.split_first() else {
      return crc24_width32_pclmul_small(state, data, keys);
    };

    state = crc24_reflected_update_bitrev_bytes(state, left);
    let state32 = match streams {
      8 => {
        update_simd_width32_reflected_bitrev_bytes_8way(state, middle, stream.fold_1024b, &stream.combine_8way, keys)
      }
      7 => update_simd_width32_reflected_bitrev_bytes_7way(state, middle, stream.fold_896b, &stream.combine_7way, keys),
      4 => update_simd_width32_reflected_bitrev_bytes_4way(state, middle, stream.fold_512b, &stream.combine_4way, keys),
      2 => update_simd_width32_reflected_bitrev_bytes_2way(state, middle, stream.fold_256b, keys),
      _ => update_simd_width32_reflected_bitrev_bytes(state, first, rest, keys),
    };
    state = state32;
    crc24_reflected_update_bitrev_bytes(state, right)
  }
}

// Single-stream kernel entry points

/// Updates CRC-24 with the single-stream PCLMULQDQ kernel.
///
/// # Safety
///
/// The current CPU must support SSE2, SSSE3, and PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn crc24_width32_pclmul_small(mut state: u32, data: &[u8], keys: &[u64; 23]) -> u32 {
  // SAFETY: All intrinsics and target-feature-gated calls require SSE2+SSSE3+PCLMULQDQ, ensured by
  // this function's #[target_feature] attribute. Pointer arithmetic stays within bounds via loop
  // guards; from_raw_parts len is bounded by remaining data.
  unsafe {
    let mut buf = data.as_ptr();
    let mut len = data.len();

    if len < 16 {
      return crc24_reflected_update_bitrev_bytes(state, data);
    }

    let coeff_16b = Simd128::new(keys[2], keys[1]);

    let mut x0 = Simd128::load_unaligned(buf).bitrev_bytes();
    x0 ^= Simd128::new(0, state as u64);
    buf = buf.add(16);
    len = len.strict_sub(16);

    while len >= 16 {
      let chunk = Simd128::load_unaligned(buf).bitrev_bytes();
      x0 = x0.fold_16_reflected(coeff_16b, chunk);
      buf = buf.add(16);
      len = len.strict_sub(16);
    }

    let x0 = x0.fold_width32_reflected(keys[6], keys[5]);
    state = x0.barrett_width32_reflected(keys[8], keys[7]);

    let tail = core::slice::from_raw_parts(buf, len);
    crc24_reflected_update_bitrev_bytes(state, tail)
  }
}

/// Updates CRC-24 with the baseline PCLMULQDQ kernel.
///
/// # Safety
///
/// The current CPU must support SSE2, SSSE3, and PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2", enable = "ssse3", enable = "pclmulqdq")]
unsafe fn crc24_width32_pclmul(mut state: u32, data: &[u8], keys: &[u64; 23]) -> u32 {
  // SAFETY: All intrinsics and target-feature-gated calls require SSE2+SSSE3+PCLMULQDQ, ensured by
  // this function's #[target_feature] attribute. align_to is sound because Simd128 is
  // repr(transparent) over the SIMD type.
  unsafe {
    let (left, middle, right) = data.align_to::<[Simd128; 8]>();
    let Some((first, rest)) = middle.split_first() else {
      return crc24_width32_pclmul_small(state, data, keys);
    };

    state = crc24_reflected_update_bitrev_bytes(state, left);
    let state32 = update_simd_width32_reflected_bitrev_bytes(state, first, rest, keys);
    state = state32;
    crc24_reflected_update_bitrev_bytes(state, right)
  }
}

// AVX-512 VPCLMULQDQ Tier

/// Loads 64 bytes without requiring alignment.
///
/// # Safety
///
/// `ptr` must be valid to read 64 initialized bytes. The source may be
/// unaligned because the bytes are copied into aligned local storage.
#[inline]
unsafe fn load_unaligned_512(ptr: *const u8) -> __m512i {
  let mut value = core::mem::MaybeUninit::<__m512i>::uninit();

  // SAFETY: The caller guarantees a readable 64-byte source. `value` is an
  // aligned, non-overlapping 64-byte destination, and every bit pattern is
  // valid for an integer vector.
  unsafe {
    core::ptr::copy_nonoverlapping(ptr, value.as_mut_ptr().cast::<u8>(), 64);
    value.assume_init()
  }
}

/// Stores 64 bytes without requiring alignment.
///
/// # Safety
///
/// `ptr` must be valid to write 64 bytes and must not overlap `value`.
#[inline]
unsafe fn store_unaligned_512(ptr: *mut u8, value: __m512i) {
  // SAFETY: The caller guarantees a writable, non-overlapping 64-byte
  // destination. `value` provides exactly 64 initialized source bytes.
  unsafe { core::ptr::copy_nonoverlapping(core::ptr::from_ref(&value).cast::<u8>(), ptr, 64) }
}

/// Multiplies the high lane of each 128-bit element in `a` by the low lane in `b`.
///
/// # Safety
///
/// The current CPU must support AVX-512F, AVX-512VL, AVX-512BW, AVX-512DQ, and
/// VPCLMULQDQ.
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq")]
unsafe fn clmul10_vpclmul(a: __m512i, b: __m512i) -> __m512i {
  _mm512_clmulepi64_epi128(a, b, 0x10)
}

/// Multiplies the low lane of each 128-bit element in `a` by the high lane in `b`.
///
/// # Safety
///
/// The current CPU must support AVX-512F, AVX-512VL, AVX-512BW, AVX-512DQ, and
/// VPCLMULQDQ.
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq")]
unsafe fn clmul01_vpclmul(a: __m512i, b: __m512i) -> __m512i {
  _mm512_clmulepi64_epi128(a, b, 0x01)
}

/// Folds four reflected 16-byte lanes and XORs their supplied input lanes.
///
/// # Safety
///
/// The current CPU must support AVX-512F, AVX-512VL, AVX-512BW, AVX-512DQ, and
/// VPCLMULQDQ.
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq")]
unsafe fn fold_16_reflected_vpclmul(state: __m512i, coeff: __m512i, data: __m512i) -> __m512i {
  // SAFETY: All intrinsics and target-feature-gated calls require
  // AVX512F+AVX512VL+AVX512BW+AVX512DQ+VPCLMULQDQ, ensured by this function's #[target_feature]
  // attribute.
  unsafe { _mm512_ternarylogic_epi64(clmul10_vpclmul(state, coeff), clmul01_vpclmul(state, coeff), data, 0x96) }
}

/// Broadcasts a pair of 64-bit coefficients across four 128-bit lanes.
///
/// # Safety
///
/// The current CPU must support AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn broadcast_coeff_128b(high: u64, low: u64) -> __m512i {
  _mm512_set_epi64(
    high.cast_signed(),
    low.cast_signed(),
    high.cast_signed(),
    low.cast_signed(),
    high.cast_signed(),
    low.cast_signed(),
    high.cast_signed(),
    low.cast_signed(),
  )
}

/// Places the width-32 CRC state in the low 32 bits of lane zero.
///
/// # Safety
///
/// The current CPU must support AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn state_mask_lane0(state: u32) -> __m512i {
  _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, state as i64)
}

/// Reverses the bits within each byte of a 512-bit vector.
///
/// # Safety
///
/// The current CPU must support AVX-512F and AVX-512BW.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn bitrev_bytes_vpclmul(x: __m512i) -> __m512i {
  let mask = _mm512_set1_epi8(0x0f);
  let lo = _mm512_and_si512(x, mask);
  let hi = _mm512_and_si512(_mm512_srli_epi16(x, 4), mask);

  let lut128 = _mm_setr_epi8(0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15);
  let lut = _mm512_broadcast_i32x4(lut128);
  let lo_rev = _mm512_shuffle_epi8(lut, lo);
  let hi_rev = _mm512_shuffle_epi8(lut, hi);
  _mm512_or_si512(_mm512_slli_epi16(lo_rev, 4), hi_rev)
}

/// Bit-reverses and folds 128-byte blocks into a width-32 VPCLMULQDQ state.
///
/// # Safety
///
/// The current CPU must support all target features enabled on this function.
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn update_simd_width32_reflected_vpclmul_bitrev_bytes(
  state: u32,
  first: &[Simd128; 8],
  rest: &[[Simd128; 8]],
  keys: &[u64; 23],
) -> u32 {
  // SAFETY: All intrinsics and target-feature-gated calls require
  // AVX512F+AVX512VL+AVX512BW+AVX512DQ+VPCLMULQDQ+SSSE3+PCLMULQDQ+SSE2, ensured by this function's
  // #[target_feature] attribute. Pointer arithmetic stays within bounds via first/rest structure.
  unsafe {
    let ptr = first.as_ptr().cast::<u8>();
    let mut x0 = load_unaligned_512(ptr);
    let mut x1 = load_unaligned_512(ptr.add(64));

    x0 = bitrev_bytes_vpclmul(x0);
    x1 = bitrev_bytes_vpclmul(x1);

    x0 = _mm512_xor_si512(x0, state_mask_lane0(state));

    let coeff_128b = broadcast_coeff_128b(keys[4], keys[3]);
    for chunk in rest {
      let ptr = chunk.as_ptr().cast::<u8>();
      let y0 = bitrev_bytes_vpclmul(load_unaligned_512(ptr));
      let y1 = bitrev_bytes_vpclmul(load_unaligned_512(ptr.add(64)));
      x0 = fold_16_reflected_vpclmul(x0, coeff_128b, y0);
      x1 = fold_16_reflected_vpclmul(x1, coeff_128b, y1);
    }

    let mut lanes0 = [Simd128(_mm_setzero_si128()); 4];
    let mut lanes1 = [Simd128(_mm_setzero_si128()); 4];
    store_unaligned_512(lanes0.as_mut_ptr().cast::<u8>(), x0);
    store_unaligned_512(lanes1.as_mut_ptr().cast::<u8>(), x1);

    let x = [
      lanes0[0], lanes0[1], lanes0[2], lanes0[3], lanes1[0], lanes1[1], lanes1[2], lanes1[3],
    ];

    finalize_lanes_width32_reflected(x, keys)
  }
}

/// Broadcasts one pair of folding coefficients across four 128-bit lanes.
///
/// # Safety
///
/// The current CPU must support AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn vpclmul_coeff(pair: (u64, u64)) -> __m512i {
  // SAFETY: Caller ensures AVX512F is available; delegating to the target-feature-gated callee.
  unsafe { broadcast_coeff_128b(pair.0, pair.1) }
}

/// Loads, bit-reverses, and returns one 128-byte block as two vectors.
///
/// # Safety
///
/// The current CPU must support AVX-512F and AVX-512BW.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_128b_block_bitrev(block: &[Simd128; 8]) -> (__m512i, __m512i) {
  // SAFETY: All intrinsics and target-feature-gated calls require AVX512F, ensured by this function's
  // #[target_feature] attribute. Pointer arithmetic stays within bounds: block is 128 bytes,
  // loading at offsets 0 and 64.
  unsafe {
    let ptr = block.as_ptr().cast::<u8>();
    let y0 = bitrev_bytes_vpclmul(load_unaligned_512(ptr));
    let y1 = bitrev_bytes_vpclmul(load_unaligned_512(ptr.add(64)));
    (y0, y1)
  }
}

/// Combines two four-lane VPCLMULQDQ states and applies width-32 reduction.
///
/// # Safety
///
/// The current CPU must support AVX-512F, SSE2, and PCLMULQDQ.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn finalize_vpclmul_state(x0: __m512i, x1: __m512i, keys: &[u64; 23]) -> u32 {
  // SAFETY: All intrinsics and target-feature-gated calls require AVX512F, ensured by this function's
  // #[target_feature] attribute.
  unsafe {
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
}

/// Bit-reverses and folds 128-byte blocks through two VPCLMULQDQ streams.
///
/// # Safety
///
/// The current CPU must support all target features enabled on this function.
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn update_simd_width32_reflected_vpclmul_bitrev_bytes_2way(
  state: u32,
  blocks: &[[Simd128; 8]],
  fold_256b: (u64, u64),
  keys: &[u64; 23],
) -> u32 {
  use crate::checksum::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  // SAFETY: All intrinsics and target-feature-gated calls require
  // AVX512F+AVX512VL+AVX512BW+AVX512DQ+VPCLMULQDQ+SSSE3+PCLMULQDQ+SSE2, ensured by this function's
  // #[target_feature] attribute.
  unsafe {
    debug_assert!(!blocks.is_empty());

    if blocks.len() < 2 {
      let Some((first, rest)) = blocks.split_first() else {
        return state;
      };
      return update_simd_width32_reflected_vpclmul_bitrev_bytes(state, first, rest, keys);
    }

    let (mut x0_0, mut x1_0) = load_128b_block_bitrev(&blocks[0]);
    let (mut x0_1, mut x1_1) = load_128b_block_bitrev(&blocks[1]);

    x0_0 = _mm512_xor_si512(x0_0, state_mask_lane0(state));

    let coeff_256 = vpclmul_coeff(fold_256b);
    let coeff_128 = vpclmul_coeff((keys[4], keys[3]));

    // Double-unrolled main loop: process 4 blocks (512B) per iteration.
    const BLOCK_SIZE: usize = 128;
    const DOUBLE_GROUP: usize = 4; // 2 × 2-way = 4 blocks = 512B

    let mut i: usize = 2;
    let aligned = blocks.len().strict_sub(blocks.len().strict_rem(DOUBLE_GROUP));

    while i.strict_add(DOUBLE_GROUP) <= aligned {
      let prefetch_idx = i.strict_add(LARGE_BLOCK_DISTANCE / BLOCK_SIZE);
      if prefetch_idx < blocks.len() {
        prefetch_read_l1(blocks[prefetch_idx].as_ptr().cast::<u8>());
      }

      // First iteration (blocks i, i+1)
      let (y0, y1) = load_128b_block_bitrev(&blocks[i]);
      x0_0 = fold_16_reflected_vpclmul(x0_0, coeff_256, y0);
      x1_0 = fold_16_reflected_vpclmul(x1_0, coeff_256, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(1)]);
      x0_1 = fold_16_reflected_vpclmul(x0_1, coeff_256, y0);
      x1_1 = fold_16_reflected_vpclmul(x1_1, coeff_256, y1);

      // Second iteration (blocks i+2, i+3)
      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(2)]);
      x0_0 = fold_16_reflected_vpclmul(x0_0, coeff_256, y0);
      x1_0 = fold_16_reflected_vpclmul(x1_0, coeff_256, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(3)]);
      x0_1 = fold_16_reflected_vpclmul(x0_1, coeff_256, y0);
      x1_1 = fold_16_reflected_vpclmul(x1_1, coeff_256, y1);

      i = i.strict_add(DOUBLE_GROUP);
    }

    // Handle remaining pairs.
    let even = blocks.len() & !1usize;
    while i < even {
      let (y0, y1) = load_128b_block_bitrev(&blocks[i]);
      x0_0 = fold_16_reflected_vpclmul(x0_0, coeff_256, y0);
      x1_0 = fold_16_reflected_vpclmul(x1_0, coeff_256, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(1)]);
      x0_1 = fold_16_reflected_vpclmul(x0_1, coeff_256, y0);
      x1_1 = fold_16_reflected_vpclmul(x1_1, coeff_256, y1);

      i = i.strict_add(2);
    }

    let mut x0 = fold_16_reflected_vpclmul(x0_0, coeff_128, x0_1);
    let mut x1 = fold_16_reflected_vpclmul(x1_0, coeff_128, x1_1);

    if even != blocks.len() {
      let (y0, y1) = load_128b_block_bitrev(&blocks[even]);
      x0 = fold_16_reflected_vpclmul(x0, coeff_128, y0);
      x1 = fold_16_reflected_vpclmul(x1, coeff_128, y1);
    }

    finalize_vpclmul_state(x0, x1, keys)
  }
}

/// Bit-reverses and folds 128-byte blocks through four VPCLMULQDQ streams.
///
/// # Safety
///
/// The current CPU must support all target features enabled on this function.
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn update_simd_width32_reflected_vpclmul_bitrev_bytes_4way(
  state: u32,
  blocks: &[[Simd128; 8]],
  fold_512b: (u64, u64),
  combine: &[(u64, u64); 3],
  keys: &[u64; 23],
) -> u32 {
  use crate::checksum::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  // SAFETY: All intrinsics and target-feature-gated calls require
  // AVX512F+AVX512VL+AVX512BW+AVX512DQ+VPCLMULQDQ+SSSE3+PCLMULQDQ+SSE2, ensured by this function's
  // #[target_feature] attribute.
  unsafe {
    debug_assert!(!blocks.is_empty());

    if blocks.len() < 4 {
      let Some((first, rest)) = blocks.split_first() else {
        return state;
      };
      return update_simd_width32_reflected_vpclmul_bitrev_bytes(state, first, rest, keys);
    }

    let (mut x0_0, mut x1_0) = load_128b_block_bitrev(&blocks[0]);
    let (mut x0_1, mut x1_1) = load_128b_block_bitrev(&blocks[1]);
    let (mut x0_2, mut x1_2) = load_128b_block_bitrev(&blocks[2]);
    let (mut x0_3, mut x1_3) = load_128b_block_bitrev(&blocks[3]);

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
    let aligned = blocks.len().strict_sub(blocks.len().strict_rem(DOUBLE_GROUP));

    while i.strict_add(DOUBLE_GROUP) <= aligned {
      let prefetch_idx = i.strict_add(LARGE_BLOCK_DISTANCE / BLOCK_SIZE);
      if prefetch_idx < blocks.len() {
        prefetch_read_l1(blocks[prefetch_idx].as_ptr().cast::<u8>());
      }

      // First iteration (blocks i..i+3)
      let (y0, y1) = load_128b_block_bitrev(&blocks[i]);
      x0_0 = fold_16_reflected_vpclmul(x0_0, coeff_512, y0);
      x1_0 = fold_16_reflected_vpclmul(x1_0, coeff_512, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(1)]);
      x0_1 = fold_16_reflected_vpclmul(x0_1, coeff_512, y0);
      x1_1 = fold_16_reflected_vpclmul(x1_1, coeff_512, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(2)]);
      x0_2 = fold_16_reflected_vpclmul(x0_2, coeff_512, y0);
      x1_2 = fold_16_reflected_vpclmul(x1_2, coeff_512, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(3)]);
      x0_3 = fold_16_reflected_vpclmul(x0_3, coeff_512, y0);
      x1_3 = fold_16_reflected_vpclmul(x1_3, coeff_512, y1);

      // Second iteration (blocks i+4..i+7)
      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(4)]);
      x0_0 = fold_16_reflected_vpclmul(x0_0, coeff_512, y0);
      x1_0 = fold_16_reflected_vpclmul(x1_0, coeff_512, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(5)]);
      x0_1 = fold_16_reflected_vpclmul(x0_1, coeff_512, y0);
      x1_1 = fold_16_reflected_vpclmul(x1_1, coeff_512, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(6)]);
      x0_2 = fold_16_reflected_vpclmul(x0_2, coeff_512, y0);
      x1_2 = fold_16_reflected_vpclmul(x1_2, coeff_512, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(7)]);
      x0_3 = fold_16_reflected_vpclmul(x0_3, coeff_512, y0);
      x1_3 = fold_16_reflected_vpclmul(x1_3, coeff_512, y1);

      i = i.strict_add(DOUBLE_GROUP);
    }

    // Handle remaining quads.
    let quad_aligned = blocks.len().strict_sub(blocks.len().strict_rem(4));
    while i < quad_aligned {
      let (y0, y1) = load_128b_block_bitrev(&blocks[i]);
      x0_0 = fold_16_reflected_vpclmul(x0_0, coeff_512, y0);
      x1_0 = fold_16_reflected_vpclmul(x1_0, coeff_512, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(1)]);
      x0_1 = fold_16_reflected_vpclmul(x0_1, coeff_512, y0);
      x1_1 = fold_16_reflected_vpclmul(x1_1, coeff_512, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(2)]);
      x0_2 = fold_16_reflected_vpclmul(x0_2, coeff_512, y0);
      x1_2 = fold_16_reflected_vpclmul(x1_2, coeff_512, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(3)]);
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
      let (y0, y1) = load_128b_block_bitrev(block);
      x0 = fold_16_reflected_vpclmul(x0, coeff_128, y0);
      x1 = fold_16_reflected_vpclmul(x1, coeff_128, y1);
    }

    finalize_vpclmul_state(x0, x1, keys)
  }
}

/// Bit-reverses and folds 128-byte blocks through seven VPCLMULQDQ streams.
///
/// # Safety
///
/// The current CPU must support all target features enabled on this function.
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn update_simd_width32_reflected_vpclmul_bitrev_bytes_7way(
  state: u32,
  blocks: &[[Simd128; 8]],
  fold_896b: (u64, u64),
  combine: &[(u64, u64); 6],
  keys: &[u64; 23],
) -> u32 {
  use crate::checksum::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  // SAFETY: All intrinsics and target-feature-gated calls require
  // AVX512F+AVX512VL+AVX512BW+AVX512DQ+VPCLMULQDQ+SSSE3+PCLMULQDQ+SSE2, ensured by this function's
  // #[target_feature] attribute.
  unsafe {
    debug_assert!(!blocks.is_empty());

    if blocks.len() < 7 {
      let Some((first, rest)) = blocks.split_first() else {
        return state;
      };
      return update_simd_width32_reflected_vpclmul_bitrev_bytes(state, first, rest, keys);
    }

    let aligned = blocks.len().strict_sub(blocks.len().strict_rem(7));

    let (mut x0_0, mut x1_0) = load_128b_block_bitrev(&blocks[0]);
    let (mut x0_1, mut x1_1) = load_128b_block_bitrev(&blocks[1]);
    let (mut x0_2, mut x1_2) = load_128b_block_bitrev(&blocks[2]);
    let (mut x0_3, mut x1_3) = load_128b_block_bitrev(&blocks[3]);
    let (mut x0_4, mut x1_4) = load_128b_block_bitrev(&blocks[4]);
    let (mut x0_5, mut x1_5) = load_128b_block_bitrev(&blocks[5]);
    let (mut x0_6, mut x1_6) = load_128b_block_bitrev(&blocks[6]);

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

      let (y0, y1) = load_128b_block_bitrev(&blocks[i]);
      x0_0 = fold_16_reflected_vpclmul(x0_0, coeff_896, y0);
      x1_0 = fold_16_reflected_vpclmul(x1_0, coeff_896, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(1)]);
      x0_1 = fold_16_reflected_vpclmul(x0_1, coeff_896, y0);
      x1_1 = fold_16_reflected_vpclmul(x1_1, coeff_896, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(2)]);
      x0_2 = fold_16_reflected_vpclmul(x0_2, coeff_896, y0);
      x1_2 = fold_16_reflected_vpclmul(x1_2, coeff_896, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(3)]);
      x0_3 = fold_16_reflected_vpclmul(x0_3, coeff_896, y0);
      x1_3 = fold_16_reflected_vpclmul(x1_3, coeff_896, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(4)]);
      x0_4 = fold_16_reflected_vpclmul(x0_4, coeff_896, y0);
      x1_4 = fold_16_reflected_vpclmul(x1_4, coeff_896, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(5)]);
      x0_5 = fold_16_reflected_vpclmul(x0_5, coeff_896, y0);
      x1_5 = fold_16_reflected_vpclmul(x1_5, coeff_896, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(6)]);
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
      let (y0, y1) = load_128b_block_bitrev(block);
      x0 = fold_16_reflected_vpclmul(x0, coeff_128, y0);
      x1 = fold_16_reflected_vpclmul(x1, coeff_128, y1);
    }

    finalize_vpclmul_state(x0, x1, keys)
  }
}

/// Bit-reverses and folds 128-byte blocks through eight VPCLMULQDQ streams.
///
/// # Safety
///
/// The current CPU must support all target features enabled on this function.
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn update_simd_width32_reflected_vpclmul_bitrev_bytes_8way(
  state: u32,
  blocks: &[[Simd128; 8]],
  fold_1024b: (u64, u64),
  combine: &[(u64, u64); 7],
  keys: &[u64; 23],
) -> u32 {
  use crate::checksum::common::prefetch::{LARGE_BLOCK_DISTANCE, prefetch_read_l1};

  // SAFETY: All intrinsics and target-feature-gated calls require
  // AVX512F+AVX512VL+AVX512BW+AVX512DQ+VPCLMULQDQ+SSSE3+PCLMULQDQ+SSE2, ensured by this function's
  // #[target_feature] attribute.
  unsafe {
    debug_assert!(!blocks.is_empty());

    if blocks.len() < 8 {
      let Some((first, rest)) = blocks.split_first() else {
        return state;
      };
      return update_simd_width32_reflected_vpclmul_bitrev_bytes(state, first, rest, keys);
    }

    let aligned = blocks.len().strict_sub(blocks.len().strict_rem(8));

    let (mut x0_0, mut x1_0) = load_128b_block_bitrev(&blocks[0]);
    let (mut x0_1, mut x1_1) = load_128b_block_bitrev(&blocks[1]);
    let (mut x0_2, mut x1_2) = load_128b_block_bitrev(&blocks[2]);
    let (mut x0_3, mut x1_3) = load_128b_block_bitrev(&blocks[3]);
    let (mut x0_4, mut x1_4) = load_128b_block_bitrev(&blocks[4]);
    let (mut x0_5, mut x1_5) = load_128b_block_bitrev(&blocks[5]);
    let (mut x0_6, mut x1_6) = load_128b_block_bitrev(&blocks[6]);
    let (mut x0_7, mut x1_7) = load_128b_block_bitrev(&blocks[7]);

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

      let (y0, y1) = load_128b_block_bitrev(&blocks[i]);
      x0_0 = fold_16_reflected_vpclmul(x0_0, coeff_1024, y0);
      x1_0 = fold_16_reflected_vpclmul(x1_0, coeff_1024, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(1)]);
      x0_1 = fold_16_reflected_vpclmul(x0_1, coeff_1024, y0);
      x1_1 = fold_16_reflected_vpclmul(x1_1, coeff_1024, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(2)]);
      x0_2 = fold_16_reflected_vpclmul(x0_2, coeff_1024, y0);
      x1_2 = fold_16_reflected_vpclmul(x1_2, coeff_1024, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(3)]);
      x0_3 = fold_16_reflected_vpclmul(x0_3, coeff_1024, y0);
      x1_3 = fold_16_reflected_vpclmul(x1_3, coeff_1024, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(4)]);
      x0_4 = fold_16_reflected_vpclmul(x0_4, coeff_1024, y0);
      x1_4 = fold_16_reflected_vpclmul(x1_4, coeff_1024, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(5)]);
      x0_5 = fold_16_reflected_vpclmul(x0_5, coeff_1024, y0);
      x1_5 = fold_16_reflected_vpclmul(x1_5, coeff_1024, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(6)]);
      x0_6 = fold_16_reflected_vpclmul(x0_6, coeff_1024, y0);
      x1_6 = fold_16_reflected_vpclmul(x1_6, coeff_1024, y1);

      let (y0, y1) = load_128b_block_bitrev(&blocks[i.strict_add(7)]);
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
      let (y0, y1) = load_128b_block_bitrev(block);
      x0 = fold_16_reflected_vpclmul(x0, coeff_128, y0);
      x1 = fold_16_reflected_vpclmul(x1, coeff_128, y1);
    }

    finalize_vpclmul_state(x0, x1, keys)
  }
}

/// Updates CRC-24 with a selected multi-stream VPCLMULQDQ kernel.
///
/// # Safety
///
/// The current CPU must support all target features enabled on this function.
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn crc24_width32_vpclmul_stream(
  mut state: u32,
  data: &[u8],
  keys: &[u64; 23],
  stream: &Crc24StreamConstants,
  streams: u8,
) -> u32 {
  // SAFETY: All intrinsics and target-feature-gated calls require
  // AVX512F+AVX512VL+AVX512BW+AVX512DQ+VPCLMULQDQ+SSSE3+PCLMULQDQ+SSE2, ensured by this function's
  // #[target_feature] attribute. align_to is sound because Simd128 is repr(transparent) over the
  // SIMD type.
  unsafe {
    let (left, middle, right) = data.align_to::<[Simd128; 8]>();
    let Some((first, rest)) = middle.split_first() else {
      return crc24_width32_pclmul_small(state, data, keys);
    };

    state = crc24_reflected_update_bitrev_bytes(state, left);
    let state32 = match streams {
      8 => update_simd_width32_reflected_vpclmul_bitrev_bytes_8way(
        state,
        middle,
        stream.fold_1024b,
        &stream.combine_8way,
        keys,
      ),
      7 => update_simd_width32_reflected_vpclmul_bitrev_bytes_7way(
        state,
        middle,
        stream.fold_896b,
        &stream.combine_7way,
        keys,
      ),
      4 => update_simd_width32_reflected_vpclmul_bitrev_bytes_4way(
        state,
        middle,
        stream.fold_512b,
        &stream.combine_4way,
        keys,
      ),
      2 => update_simd_width32_reflected_vpclmul_bitrev_bytes_2way(state, middle, stream.fold_256b, keys),
      _ => update_simd_width32_reflected_vpclmul_bitrev_bytes(state, first, rest, keys),
    };
    state = state32;
    crc24_reflected_update_bitrev_bytes(state, right)
  }
}

/// Updates CRC-24 with the baseline VPCLMULQDQ kernel.
///
/// # Safety
///
/// The current CPU must support all target features enabled on this function.
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,vpclmulqdq,ssse3,pclmulqdq,sse2")]
unsafe fn crc24_width32_vpclmul(mut state: u32, data: &[u8], keys: &[u64; 23]) -> u32 {
  // SAFETY: All intrinsics and target-feature-gated calls require
  // AVX512F+AVX512VL+AVX512BW+AVX512DQ+VPCLMULQDQ+SSSE3+PCLMULQDQ+SSE2, ensured by this function's
  // #[target_feature] attribute. align_to is sound because Simd128 is repr(transparent) over the
  // SIMD type.
  unsafe {
    let (left, middle, right) = data.align_to::<[Simd128; 8]>();
    let Some((first, rest)) = middle.split_first() else {
      return crc24_width32_pclmul_small(state, data, keys);
    };

    state = crc24_reflected_update_bitrev_bytes(state, left);
    let state32 = update_simd_width32_reflected_vpclmul_bitrev_bytes(state, first, rest, keys);
    state = state32;
    crc24_reflected_update_bitrev_bytes(state, right)
  }
}

// Private safe kernel adapters matching the dispatcher's function signature.

/// CRC-24/OPENPGP PCLMULQDQ kernel.
///
/// The dispatcher selects this private kernel only after verifying SSSE3 and
/// PCLMULQDQ support.
#[inline]
pub(super) fn crc24_openpgp_pclmul_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
  state = unsafe { crc24_width32_pclmul(state, data, &CRC24_OPENPGP_KEYS_REFLECTED) };
  from_reflected_state(state)
}

/// CRC-24/OPENPGP PCLMULQDQ small-buffer kernel.
///
/// Optimized for inputs smaller than a folding block (128 bytes).
///
/// The dispatcher selects this private kernel only after verifying SSSE3 and
/// PCLMULQDQ support.
#[inline]
pub(super) fn crc24_openpgp_pclmul_small_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
  state = unsafe { crc24_width32_pclmul_small(state, data, &CRC24_OPENPGP_KEYS_REFLECTED) };
  from_reflected_state(state)
}

/// CRC-24/OPENPGP PCLMULQDQ kernel (2-way multi-stream).
#[inline]
pub(super) fn crc24_openpgp_pclmul_2way_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
  state = unsafe {
    crc24_width32_pclmul_stream(
      state,
      data,
      &CRC24_OPENPGP_KEYS_REFLECTED,
      &CRC24_OPENPGP_STREAM_REFLECTED,
      2,
    )
  };
  from_reflected_state(state)
}

/// CRC-24/OPENPGP PCLMULQDQ kernel (4-way multi-stream).
#[inline]
pub(super) fn crc24_openpgp_pclmul_4way_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
  state = unsafe {
    crc24_width32_pclmul_stream(
      state,
      data,
      &CRC24_OPENPGP_KEYS_REFLECTED,
      &CRC24_OPENPGP_STREAM_REFLECTED,
      4,
    )
  };
  from_reflected_state(state)
}

/// CRC-24/OPENPGP PCLMULQDQ kernel (7-way multi-stream).
#[inline]
pub(super) fn crc24_openpgp_pclmul_7way_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
  state = unsafe {
    crc24_width32_pclmul_stream(
      state,
      data,
      &CRC24_OPENPGP_KEYS_REFLECTED,
      &CRC24_OPENPGP_STREAM_REFLECTED,
      7,
    )
  };
  from_reflected_state(state)
}

/// CRC-24/OPENPGP PCLMULQDQ kernel (8-way multi-stream).
#[inline]
pub(super) fn crc24_openpgp_pclmul_8way_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies SSSE3 + PCLMULQDQ before selecting this kernel.
  state = unsafe {
    crc24_width32_pclmul_stream(
      state,
      data,
      &CRC24_OPENPGP_KEYS_REFLECTED,
      &CRC24_OPENPGP_STREAM_REFLECTED,
      8,
    )
  };
  from_reflected_state(state)
}

/// CRC-24/OPENPGP VPCLMULQDQ kernel (AVX-512).
///
/// The dispatcher selects this private kernel only after verifying VPCLMULQDQ
/// and AVX-512 support.
#[inline]
pub(super) fn crc24_openpgp_vpclmul_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies VPCLMULQDQ + AVX-512 before selecting this kernel.
  state = unsafe { crc24_width32_vpclmul(state, data, &CRC24_OPENPGP_KEYS_REFLECTED) };
  from_reflected_state(state)
}

/// CRC-24/OPENPGP VPCLMULQDQ kernel (2-way multi-stream).
#[inline]
pub(super) fn crc24_openpgp_vpclmul_2way_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies VPCLMULQDQ + AVX-512 before selecting this kernel.
  state = unsafe {
    crc24_width32_vpclmul_stream(
      state,
      data,
      &CRC24_OPENPGP_KEYS_REFLECTED,
      &CRC24_OPENPGP_STREAM_REFLECTED,
      2,
    )
  };
  from_reflected_state(state)
}

/// CRC-24/OPENPGP VPCLMULQDQ kernel (4-way multi-stream).
#[inline]
pub(super) fn crc24_openpgp_vpclmul_4way_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies VPCLMULQDQ + AVX-512 before selecting this kernel.
  state = unsafe {
    crc24_width32_vpclmul_stream(
      state,
      data,
      &CRC24_OPENPGP_KEYS_REFLECTED,
      &CRC24_OPENPGP_STREAM_REFLECTED,
      4,
    )
  };
  from_reflected_state(state)
}

/// CRC-24/OPENPGP VPCLMULQDQ kernel (7-way multi-stream).
#[inline]
pub(super) fn crc24_openpgp_vpclmul_7way_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies VPCLMULQDQ + AVX-512 before selecting this kernel.
  state = unsafe {
    crc24_width32_vpclmul_stream(
      state,
      data,
      &CRC24_OPENPGP_KEYS_REFLECTED,
      &CRC24_OPENPGP_STREAM_REFLECTED,
      7,
    )
  };
  from_reflected_state(state)
}

/// CRC-24/OPENPGP VPCLMULQDQ kernel (8-way multi-stream).
#[inline]
pub(super) fn crc24_openpgp_vpclmul_8way_safe(crc: u32, data: &[u8]) -> u32 {
  let mut state = to_reflected_state(crc);
  // SAFETY: Dispatcher verifies VPCLMULQDQ + AVX-512 before selecting this kernel.
  state = unsafe {
    crc24_width32_vpclmul_stream(
      state,
      data,
      &CRC24_OPENPGP_KEYS_REFLECTED,
      &CRC24_OPENPGP_STREAM_REFLECTED,
      8,
    )
  };
  from_reflected_state(state)
}

#[cfg(test)]
mod tests {
  extern crate std;

  use alloc::vec::Vec;

  use super::*;

  const LENS: &[usize] = &[0, 1, 7, 15, 16, 31, 63, 64, 127, 128, 255, 256, 1023, 1024, 4096];
  const OFFSETS: &[usize] = &[0, 1, 7, 15];
  const STATES: &[u32] = &[0, 0x00b7_04ce, 0x005a_a5a5, 0x00ff_ffff];

  fn data() -> Vec<u8> {
    (0u16..4111)
      .map(|i| {
        let [low, high] = i.to_le_bytes();
        low.wrapping_mul(61).wrapping_add(high)
      })
      .collect()
  }

  fn assert_kernel(name: &str, kernel: fn(u32, &[u8]) -> u32) {
    let input = data();
    for &state in STATES {
      for &offset in OFFSETS {
        for &len in LENS {
          let slice = &input[offset..offset.strict_add(len)];
          assert_eq!(
            kernel(state, slice),
            super::super::portable::crc24_openpgp_slice8(state, slice),
            "{name} state={state:#010x} offset={offset} len={len}"
          );
        }
      }
    }
  }

  #[test]
  fn pclmul_kernels_match_portable() {
    if !crate::platform::caps().has(crate::platform::caps::x86::PCLMUL_READY) {
      return;
    }

    for (name, kernel) in [
      ("openpgp/pclmul", crc24_openpgp_pclmul_safe as fn(u32, &[u8]) -> u32),
      ("openpgp/pclmul-small", crc24_openpgp_pclmul_small_safe),
      ("openpgp/pclmul-2way", crc24_openpgp_pclmul_2way_safe),
      ("openpgp/pclmul-4way", crc24_openpgp_pclmul_4way_safe),
      ("openpgp/pclmul-7way", crc24_openpgp_pclmul_7way_safe),
      ("openpgp/pclmul-8way", crc24_openpgp_pclmul_8way_safe),
    ] {
      assert_kernel(name, kernel);
    }
  }

  #[test]
  fn vpclmul_kernels_match_portable() {
    if !crate::platform::caps().has(crate::platform::caps::x86::VPCLMUL_READY) {
      return;
    }

    for (name, kernel) in [
      ("openpgp/vpclmul", crc24_openpgp_vpclmul_safe as fn(u32, &[u8]) -> u32),
      ("openpgp/vpclmul-2way", crc24_openpgp_vpclmul_2way_safe),
      ("openpgp/vpclmul-4way", crc24_openpgp_vpclmul_4way_safe),
      ("openpgp/vpclmul-7way", crc24_openpgp_vpclmul_7way_safe),
      ("openpgp/vpclmul-8way", crc24_openpgp_vpclmul_8way_safe),
    ] {
      assert_kernel(name, kernel);
    }
  }
}
