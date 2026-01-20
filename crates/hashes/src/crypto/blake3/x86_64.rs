//! BLAKE3 x86_64 SIMD kernels.
//!
//! This module provides SIMD-accelerated compression functions for BLAKE3 using
//! x86_64 intrinsics (SSSE3, AVX2, AVX-512).
//!
//! # Safety
//!
//! All functions in this module are marked `unsafe` and require specific CPU
//! features to be present. Callers must verify CPU capabilities before calling.

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::undocumented_unsafe_blocks)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

pub(crate) mod avx2;
pub(crate) mod avx512;
pub(crate) mod sse41;

use super::{BLOCK_LEN, CHUNK_LEN, CHUNK_START, IV, OUT_LEN, PARENT, first_8_words, words16_from_le_bytes_64};

// ─────────────────────────────────────────────────────────────────────────────
// Rotation shuffle masks for SSSE3
// ─────────────────────────────────────────────────────────────────────────────

/// Shuffle mask for 16-bit rotation right of 32-bit lanes using `pshufb`.
/// Rotates each u32 lane right by 16 bits: bytes [1,0,3,2] for each lane.
#[cfg(target_arch = "x86_64")]
const ROT16_SHUFFLE: [i8; 16] = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13];

/// Shuffle mask for 8-bit rotation right of 32-bit lanes using `pshufb`.
/// Rotates each u32 lane right by 8 bits: bytes [0,3,2,1] for each lane.
#[cfg(target_arch = "x86_64")]
const ROT8_SHUFFLE: [i8; 16] = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12];

// ─────────────────────────────────────────────────────────────────────────────
// SSSE3 Implementation
// ─────────────────────────────────────────────────────────────────────────────

/// BLAKE3 compress function using SSSE3 intrinsics.
///
/// # Safety
///
/// Caller must ensure SSSE3 is available (`target_feature = "ssse3"`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
pub unsafe fn compress_ssse3(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  // Load rotation shuffle masks
  let rot16 = _mm_loadu_si128(ROT16_SHUFFLE.as_ptr().cast());
  let rot8 = _mm_loadu_si128(ROT8_SHUFFLE.as_ptr().cast());

  // Load state into 4 vectors (rows of the BLAKE3 state matrix)
  // row0 = [v0, v1, v2, v3] = cv[0..4]
  // row1 = [v4, v5, v6, v7] = cv[4..8]
  // row2 = [v8, v9, v10, v11] = IV[0..4]
  // row3 = [v12, v13, v14, v15] = [counter_lo, counter_hi, block_len, flags]
  let mut row0 = _mm_loadu_si128(chaining_value.as_ptr().cast());
  let mut row1 = _mm_loadu_si128(chaining_value.as_ptr().add(4).cast());
  let mut row2 = _mm_loadu_si128(IV.as_ptr().cast());
  let mut row3 = _mm_set_epi32(flags as i32, block_len as i32, (counter >> 32) as i32, counter as i32);

  // Load message words
  let m0 = _mm_loadu_si128(block_words.as_ptr().cast());
  let m1 = _mm_loadu_si128(block_words.as_ptr().add(4).cast());
  let m2 = _mm_loadu_si128(block_words.as_ptr().add(8).cast());
  let m3 = _mm_loadu_si128(block_words.as_ptr().add(12).cast());

  // The G function operates on columns and diagonals.
  // We implement it with a "row-wise" representation and use shuffles
  // to rotate rows for diagonal operations.

  // Macro for the quarter-round G function on a single column
  macro_rules! g {
    ($a:expr, $b:expr, $c:expr, $d:expr, $mx:expr, $my:expr) => {{
      // a = a + b + mx
      $a = _mm_add_epi32($a, $b);
      $a = _mm_add_epi32($a, $mx);
      // d = (d ^ a) >>> 16
      $d = _mm_xor_si128($d, $a);
      $d = _mm_shuffle_epi8($d, rot16);
      // c = c + d
      $c = _mm_add_epi32($c, $d);
      // b = (b ^ c) >>> 12
      $b = _mm_xor_si128($b, $c);
      $b = _mm_or_si128(_mm_srli_epi32($b, 12), _mm_slli_epi32($b, 20));
      // a = a + b + my
      $a = _mm_add_epi32($a, $b);
      $a = _mm_add_epi32($a, $my);
      // d = (d ^ a) >>> 8
      $d = _mm_xor_si128($d, $a);
      $d = _mm_shuffle_epi8($d, rot8);
      // c = c + d
      $c = _mm_add_epi32($c, $d);
      // b = (b ^ c) >>> 7
      $b = _mm_xor_si128($b, $c);
      $b = _mm_or_si128(_mm_srli_epi32($b, 7), _mm_slli_epi32($b, 25));
    }};
  }

  // One round: column step + diagonal step
  // After column mixing, rotate rows to set up diagonal mixing
  macro_rules! round {
    ($m0:expr, $m1:expr, $m2:expr, $m3:expr) => {{
      // Column step: mix columns [0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]
      g!(row0, row1, row2, row3, $m0, $m1);

      // Diagonalize: rotate rows to prepare for diagonal mixing
      // row1 rotates left by 1: [v4,v5,v6,v7] -> [v5,v6,v7,v4]
      // row2 rotates left by 2: [v8,v9,v10,v11] -> [v10,v11,v8,v9]
      // row3 rotates left by 3: [v12,v13,v14,v15] -> [v15,v12,v13,v14]
      row1 = _mm_shuffle_epi32(row1, 0b00_11_10_01); // rotate left 1
      row2 = _mm_shuffle_epi32(row2, 0b01_00_11_10); // rotate left 2
      row3 = _mm_shuffle_epi32(row3, 0b10_01_00_11); // rotate left 3

      // Diagonal step: now the "columns" are actually diagonals
      g!(row0, row1, row2, row3, $m2, $m3);

      // Undiagonalize: rotate rows back
      row1 = _mm_shuffle_epi32(row1, 0b10_01_00_11); // rotate right 1 = left 3
      row2 = _mm_shuffle_epi32(row2, 0b01_00_11_10); // rotate right 2 = left 2
      row3 = _mm_shuffle_epi32(row3, 0b00_11_10_01); // rotate right 3 = left 1
    }};
  }

  // Helper to permute message for each round
  // BLAKE3 message schedule (7 rounds with fixed permutations)
  // Round 0: m[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
  let (mx0, my0, mx1, my1) = permute_round_0(m0, m1, m2, m3);
  round!(mx0, my0, mx1, my1);

  // Round 1: m[2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
  let (mx0, my0, mx1, my1) = permute_round_1(m0, m1, m2, m3);
  round!(mx0, my0, mx1, my1);

  // Round 2: m[3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1]
  let (mx0, my0, mx1, my1) = permute_round_2(m0, m1, m2, m3);
  round!(mx0, my0, mx1, my1);

  // Round 3: m[10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6]
  let (mx0, my0, mx1, my1) = permute_round_3(m0, m1, m2, m3);
  round!(mx0, my0, mx1, my1);

  // Round 4: m[12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4]
  let (mx0, my0, mx1, my1) = permute_round_4(m0, m1, m2, m3);
  round!(mx0, my0, mx1, my1);

  // Round 5: m[9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7]
  let (mx0, my0, mx1, my1) = permute_round_5(m0, m1, m2, m3);
  round!(mx0, my0, mx1, my1);

  // Round 6: m[11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13]
  let (mx0, my0, mx1, my1) = permute_round_6(m0, m1, m2, m3);
  round!(mx0, my0, mx1, my1);

  // Finalization: XOR the two halves
  // low half: row0 ^= row2, row1 ^= row3
  // high half: row2 ^= cv[0..4], row3 ^= cv[4..8]
  let cv_lo = _mm_loadu_si128(chaining_value.as_ptr().cast());
  let cv_hi = _mm_loadu_si128(chaining_value.as_ptr().add(4).cast());

  row0 = _mm_xor_si128(row0, row2);
  row1 = _mm_xor_si128(row1, row3);
  row2 = _mm_xor_si128(row2, cv_lo);
  row3 = _mm_xor_si128(row3, cv_hi);

  // Store result
  let mut out = [0u32; 16];
  _mm_storeu_si128(out.as_mut_ptr().cast(), row0);
  _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), row1);
  _mm_storeu_si128(out.as_mut_ptr().add(8).cast(), row2);
  _mm_storeu_si128(out.as_mut_ptr().add(12).cast(), row3);
  out
}

// ─────────────────────────────────────────────────────────────────────────────
// Message permutation helpers
// ─────────────────────────────────────────────────────────────────────────────
//
// BLAKE3 uses a fixed message schedule where each round permutes the 16 message
// words. We pre-compute the permutations as shuffle operations.

/// Round 0: identity permutation
/// m[0,1,2,3], m[4,5,6,7], m[8,9,10,11], m[12,13,14,15]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn permute_round_0(m0: __m128i, m1: __m128i, m2: __m128i, m3: __m128i) -> (__m128i, __m128i, __m128i, __m128i) {
  // Column: m[0,2,4,6], m[1,3,5,7]
  // Diagonal: m[8,10,12,14], m[9,11,13,15]
  // For round 0, the schedule is [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
  // Column step uses: (m0,m1), (m2,m3), (m4,m5), (m6,m7)
  // Diagonal step uses: (m8,m9), (m10,m11), (m12,m13), (m14,m15)
  //
  // In vectorized form:
  // mx0 = [m0, m2, m4, m6] for first G in column
  // my0 = [m1, m3, m5, m7] for second G in column
  // mx1 = [m8, m10, m12, m14] for first G in diagonal
  // my1 = [m9, m11, m13, m15] for second G in diagonal

  let t0: __m128i = _mm_shuffle_epi32(m0, 0b10_00_10_00); // [m0, m0, m2, m2]
  let t1 = _mm_shuffle_epi32(m1, 0b10_00_10_00); // [m4, m4, m6, m6]
  let mx0 = _mm_unpacklo_epi64(t0, t1); // [m0, m2, m4, m6]

  let t0 = _mm_shuffle_epi32(m0, 0b11_01_11_01); // [m1, m1, m3, m3]
  let t1 = _mm_shuffle_epi32(m1, 0b11_01_11_01); // [m5, m5, m7, m7]
  let my0 = _mm_unpacklo_epi64(t0, t1); // [m1, m3, m5, m7]

  let t0 = _mm_shuffle_epi32(m2, 0b10_00_10_00); // [m8, m8, m10, m10]
  let t1 = _mm_shuffle_epi32(m3, 0b10_00_10_00); // [m12, m12, m14, m14]
  let mx1 = _mm_unpacklo_epi64(t0, t1); // [m8, m10, m12, m14]

  let t0 = _mm_shuffle_epi32(m2, 0b11_01_11_01); // [m9, m9, m11, m11]
  let t1 = _mm_shuffle_epi32(m3, 0b11_01_11_01); // [m13, m13, m15, m15]
  let my1 = _mm_unpacklo_epi64(t0, t1); // [m9, m11, m13, m15]

  (mx0, my0, mx1, my1)
}

/// Round 1: m[2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn permute_round_1(m0: __m128i, m1: __m128i, m2: __m128i, m3: __m128i) -> (__m128i, __m128i, __m128i, __m128i) {
  // Schedule: [2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
  // Column: (2,6), (3,10), (7,0), (4,13) -> mx0=[2,3,7,4], my0=[6,10,0,13]
  // Diagonal: (1,11), (12,5), (9,14), (15,8) -> mx1=[1,12,9,15], my1=[11,5,14,8]
  let mx0 = blend_words(m0, m1, m2, m3, 2, 3, 7, 4);
  let my0 = blend_words(m0, m1, m2, m3, 6, 10, 0, 13);
  let mx1 = blend_words(m0, m1, m2, m3, 1, 12, 9, 15);
  let my1 = blend_words(m0, m1, m2, m3, 11, 5, 14, 8);
  (mx0, my0, mx1, my1)
}

/// Round 2: m[3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn permute_round_2(m0: __m128i, m1: __m128i, m2: __m128i, m3: __m128i) -> (__m128i, __m128i, __m128i, __m128i) {
  let mx0 = blend_words(m0, m1, m2, m3, 3, 10, 13, 7);
  let my0 = blend_words(m0, m1, m2, m3, 4, 12, 2, 14);
  let mx1 = blend_words(m0, m1, m2, m3, 6, 9, 11, 8);
  let my1 = blend_words(m0, m1, m2, m3, 5, 0, 15, 1);
  (mx0, my0, mx1, my1)
}

/// Round 3: m[10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn permute_round_3(m0: __m128i, m1: __m128i, m2: __m128i, m3: __m128i) -> (__m128i, __m128i, __m128i, __m128i) {
  let mx0 = blend_words(m0, m1, m2, m3, 10, 12, 14, 13);
  let my0 = blend_words(m0, m1, m2, m3, 7, 9, 3, 15);
  let mx1 = blend_words(m0, m1, m2, m3, 4, 11, 5, 1);
  let my1 = blend_words(m0, m1, m2, m3, 0, 2, 8, 6);
  (mx0, my0, mx1, my1)
}

/// Round 4: m[12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn permute_round_4(m0: __m128i, m1: __m128i, m2: __m128i, m3: __m128i) -> (__m128i, __m128i, __m128i, __m128i) {
  let mx0 = blend_words(m0, m1, m2, m3, 12, 9, 15, 14);
  let my0 = blend_words(m0, m1, m2, m3, 13, 11, 10, 8);
  let mx1 = blend_words(m0, m1, m2, m3, 7, 5, 0, 6);
  let my1 = blend_words(m0, m1, m2, m3, 2, 3, 1, 4);
  (mx0, my0, mx1, my1)
}

/// Round 5: m[9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn permute_round_5(m0: __m128i, m1: __m128i, m2: __m128i, m3: __m128i) -> (__m128i, __m128i, __m128i, __m128i) {
  let mx0 = blend_words(m0, m1, m2, m3, 9, 11, 8, 15);
  let my0 = blend_words(m0, m1, m2, m3, 14, 5, 12, 1);
  let mx1 = blend_words(m0, m1, m2, m3, 13, 0, 2, 4);
  let my1 = blend_words(m0, m1, m2, m3, 3, 10, 6, 7);
  (mx0, my0, mx1, my1)
}

/// Round 6: m[11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn permute_round_6(m0: __m128i, m1: __m128i, m2: __m128i, m3: __m128i) -> (__m128i, __m128i, __m128i, __m128i) {
  let mx0 = blend_words(m0, m1, m2, m3, 11, 5, 1, 8);
  let my0 = blend_words(m0, m1, m2, m3, 15, 0, 9, 6);
  let mx1 = blend_words(m0, m1, m2, m3, 14, 2, 3, 7);
  let my1 = blend_words(m0, m1, m2, m3, 10, 12, 4, 13);
  (mx0, my0, mx1, my1)
}

/// Helper to blend 4 specific words from the message vectors.
/// Each index is 0-15, mapping to m0[0-3], m1[4-7], m2[8-11], m3[12-15].
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn blend_words(
  m0: __m128i,
  m1: __m128i,
  m2: __m128i,
  m3: __m128i,
  i0: usize,
  i1: usize,
  i2: usize,
  i3: usize,
) -> __m128i {
  // Extract individual words - this is not the most efficient but is correct
  // A production implementation would use more clever shuffles
  let words = [
    _mm_extract_epi32(m0, 0) as u32,
    _mm_extract_epi32(m0, 1) as u32,
    _mm_extract_epi32(m0, 2) as u32,
    _mm_extract_epi32(m0, 3) as u32,
    _mm_extract_epi32(m1, 0) as u32,
    _mm_extract_epi32(m1, 1) as u32,
    _mm_extract_epi32(m1, 2) as u32,
    _mm_extract_epi32(m1, 3) as u32,
    _mm_extract_epi32(m2, 0) as u32,
    _mm_extract_epi32(m2, 1) as u32,
    _mm_extract_epi32(m2, 2) as u32,
    _mm_extract_epi32(m2, 3) as u32,
    _mm_extract_epi32(m3, 0) as u32,
    _mm_extract_epi32(m3, 1) as u32,
    _mm_extract_epi32(m3, 2) as u32,
    _mm_extract_epi32(m3, 3) as u32,
  ];
  _mm_set_epi32(words[i3] as i32, words[i2] as i32, words[i1] as i32, words[i0] as i32)
}

/// Unused helper - kept for reference
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
unsafe fn interleave_low_high(a: __m128i, b: __m128i) -> __m128i {
  // Interleave: [a0, a2, b0, b2]
  let a_even = _mm_shuffle_epi32(a, 0b10_00_10_00); // [a0, a0, a2, a2]
  let b_even = _mm_shuffle_epi32(b, 0b10_00_10_00); // [b0, b0, b2, b2]
  _mm_unpacklo_epi32(a_even, b_even) // [a0, b0, a2, b2] - not quite right
}

// ─────────────────────────────────────────────────────────────────────────────
// High-level kernel wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// SSSE3 chunk compression: process multiple 64-byte blocks.
///
/// # Safety
///
/// Caller must ensure SSSE3 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
pub unsafe fn chunk_compress_blocks_ssse3(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
  let (block_slices, remainder) = blocks.as_chunks::<BLOCK_LEN>();
  debug_assert!(remainder.is_empty());
  for block_bytes in block_slices {
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    let block_words = words16_from_le_bytes_64(block_bytes);
    *chaining_value = first_8_words(compress_ssse3(
      chaining_value,
      &block_words,
      chunk_counter,
      BLOCK_LEN as u32,
      flags | start,
    ));
    *blocks_compressed = blocks_compressed.wrapping_add(1);
  }
}

/// SSSE3 parent CV computation.
///
/// # Safety
///
/// Caller must ensure SSSE3 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
pub unsafe fn parent_cv_ssse3(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  let mut block_words = [0u32; 16];
  block_words[..8].copy_from_slice(&left_child_cv);
  block_words[8..].copy_from_slice(&right_child_cv);
  first_8_words(compress_ssse3(
    &key_words,
    &block_words,
    0,
    BLOCK_LEN as u32,
    PARENT | flags,
  ))
}

/// Hash many contiguous full chunks using the SSSE3 single-block compressor.
///
/// This is a correctness-preserving throughput baseline until a true hash4/hash8
/// multi-lane x86 kernel is implemented.
///
/// # Safety
///
/// Caller must ensure SSSE3 is available, and the input/output pointers are valid.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
pub unsafe fn hash_many_contiguous_ssse3(
  input: *const u8,
  num_chunks: usize,
  key: &[u32; 8],
  mut counter: u64,
  flags: u32,
  out: *mut u8,
) {
  debug_assert!(num_chunks != 0);

  for chunk_idx in 0..num_chunks {
    let mut cv = *key;

    for block_idx in 0..(CHUNK_LEN / BLOCK_LEN) {
      let mut block = [0u8; BLOCK_LEN];
      let src = unsafe { input.add(chunk_idx * CHUNK_LEN + block_idx * BLOCK_LEN) };
      unsafe { core::ptr::copy_nonoverlapping(src, block.as_mut_ptr(), BLOCK_LEN) };
      let block_words = words16_from_le_bytes_64(&block);

      let start = if block_idx == 0 { CHUNK_START } else { 0 };
      let end = if block_idx + 1 == (CHUNK_LEN / BLOCK_LEN) {
        super::CHUNK_END
      } else {
        0
      };
      cv = first_8_words(compress_ssse3(
        &cv,
        &block_words,
        counter,
        BLOCK_LEN as u32,
        flags | start | end,
      ));
    }

    for (j, &word) in cv.iter().enumerate() {
      let bytes = word.to_le_bytes();
      unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), out.add(chunk_idx * OUT_LEN + j * 4), 4) };
    }

    counter = counter.wrapping_add(1);
  }
}

// ─────────────────────────────────────────────────
