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

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
pub(crate) mod asm;
pub(crate) mod avx2;
pub(crate) mod avx512;
pub(crate) mod sse41;

use super::{BLOCK_LEN, CHUNK_LEN, CHUNK_START, IV, OUT_LEN, PARENT, first_8_words, words16_from_le_bytes_64};

// Shared helpers for SIMD kernels.

#[inline(always)]
pub(crate) const fn counter_low(counter: u64) -> u32 {
  counter as u32
}

#[inline(always)]
pub(crate) const fn counter_high(counter: u64) -> u32 {
  (counter >> 32) as u32
}

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
// CV-only compression helpers (avoid `[u32; 16]` materialization)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn load_msg_vecs(block: *const u8) -> (__m128i, __m128i, __m128i, __m128i) {
  // SAFETY: caller guarantees `block` is valid for 64 bytes.
  unsafe {
    let m0 = _mm_loadu_si128(block.cast());
    let m1 = _mm_loadu_si128(block.add(16).cast());
    let m2 = _mm_loadu_si128(block.add(32).cast());
    let m3 = _mm_loadu_si128(block.add(48).cast());
    (m0, m1, m2, m3)
  }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1,ssse3")]
pub(crate) unsafe fn compress_cv_sse41_bytes(
  chaining_value: &[u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  let (m0, m1, m2, m3) = load_msg_vecs(block);
  let [row0, row1, row2, row3] = compress_pre_sse41_impl(chaining_value, m0, m1, m2, m3, counter, block_len, flags);
  let row0 = _mm_xor_si128(row0, row2);
  let row1 = _mm_xor_si128(row1, row3);

  let mut out = [0u32; 8];
  _mm_storeu_si128(out.as_mut_ptr().cast(), row0);
  _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), row1);
  out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,sse4.1,ssse3")]
pub(crate) unsafe fn compress_cv_avx2_bytes(
  chaining_value: &[u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  let (m0, m1, m2, m3) = load_msg_vecs(block);
  compress_cv_avx2(chaining_value, m0, m1, m2, m3, counter, block_len, flags)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2,sse4.1,ssse3")]
pub(crate) unsafe fn compress_cv_avx512_bytes(
  chaining_value: &[u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  let (m0, m1, m2, m3) = load_msg_vecs(block);
  compress_cv_avx512(chaining_value, m0, m1, m2, m3, counter, block_len, flags)
}

// ─────────────────────────────────────────────────────────────────────────────
// SSE4.1 / AVX2 / AVX-512 per-block compressor (world-class schedule)
// ─────────────────────────────────────────────────────────────────────────────
//
// This schedule is adapted from the upstream BLAKE3 project's high-performance
// `rust_sse41` compressor (CC0-1.0 / Apache-2.0 / LLVM-exception). It avoids
// the expensive per-round gather/shuffle machinery in the SSSE3 row-wise
// implementation by keeping the message in a permuted form and applying a
// fixed, low-instruction permutation each round.

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn rot16_sse41(a: __m128i) -> __m128i {
  unsafe { _mm_or_si128(_mm_srli_epi32(a, 16), _mm_slli_epi32(a, 16)) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn rot12_sse41(a: __m128i) -> __m128i {
  unsafe { _mm_or_si128(_mm_srli_epi32(a, 12), _mm_slli_epi32(a, 20)) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn rot8_sse41(a: __m128i) -> __m128i {
  unsafe { _mm_or_si128(_mm_srli_epi32(a, 8), _mm_slli_epi32(a, 24)) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn rot7_sse41(a: __m128i) -> __m128i {
  unsafe { _mm_or_si128(_mm_srli_epi32(a, 7), _mm_slli_epi32(a, 25)) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn g1_sse41(row0: &mut __m128i, row1: &mut __m128i, row2: &mut __m128i, row3: &mut __m128i, m: __m128i) {
  unsafe {
    *row0 = _mm_add_epi32(_mm_add_epi32(*row0, m), *row1);
    *row3 = _mm_xor_si128(*row3, *row0);
    *row3 = rot16_sse41(*row3);
    *row2 = _mm_add_epi32(*row2, *row3);
    *row1 = _mm_xor_si128(*row1, *row2);
    *row1 = rot12_sse41(*row1);
  }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn g2_sse41(row0: &mut __m128i, row1: &mut __m128i, row2: &mut __m128i, row3: &mut __m128i, m: __m128i) {
  unsafe {
    *row0 = _mm_add_epi32(_mm_add_epi32(*row0, m), *row1);
    *row3 = _mm_xor_si128(*row3, *row0);
    *row3 = rot8_sse41(*row3);
    *row2 = _mm_add_epi32(*row2, *row3);
    *row1 = _mm_xor_si128(*row1, *row2);
    *row1 = rot7_sse41(*row1);
  }
}

macro_rules! _MM_SHUFFLE {
  ($z:expr, $y:expr, $x:expr, $w:expr) => {
    ($z << 6) | ($y << 4) | ($x << 2) | $w
  };
}

macro_rules! shuffle2 {
  ($a:expr, $b:expr, $c:expr) => {
    _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps($a), _mm_castsi128_ps($b), $c))
  };
}

// Leave row1 unrotated and diagonalize the other rows.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn diagonalize_sse41(row0: &mut __m128i, row2: &mut __m128i, row3: &mut __m128i) {
  unsafe {
    *row0 = _mm_shuffle_epi32(*row0, _MM_SHUFFLE!(2, 1, 0, 3));
    *row3 = _mm_shuffle_epi32(*row3, _MM_SHUFFLE!(1, 0, 3, 2));
    *row2 = _mm_shuffle_epi32(*row2, _MM_SHUFFLE!(0, 3, 2, 1));
  }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn undiagonalize_sse41(row0: &mut __m128i, row2: &mut __m128i, row3: &mut __m128i) {
  unsafe {
    *row0 = _mm_shuffle_epi32(*row0, _MM_SHUFFLE!(0, 3, 2, 1));
    *row3 = _mm_shuffle_epi32(*row3, _MM_SHUFFLE!(1, 0, 3, 2));
    *row2 = _mm_shuffle_epi32(*row2, _MM_SHUFFLE!(2, 1, 0, 3));
  }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn compress_pre_sse41_impl(
  chaining_value: &[u32; 8],
  mut m0: __m128i,
  mut m1: __m128i,
  mut m2: __m128i,
  mut m3: __m128i,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [__m128i; 4] {
  unsafe {
    let mut row0 = _mm_loadu_si128(chaining_value.as_ptr().cast());
    let mut row1 = _mm_loadu_si128(chaining_value.as_ptr().add(4).cast());
    let mut row2 = _mm_setr_epi32(IV[0] as i32, IV[1] as i32, IV[2] as i32, IV[3] as i32);
    let mut row3 = _mm_setr_epi32(
      counter as u32 as i32,
      (counter >> 32) as u32 as i32,
      block_len as i32,
      flags as i32,
    );

    let mut t0;
    let mut t1;
    let mut t2;
    let mut t3;
    let mut tt;

    // Round 1
    t0 = shuffle2!(m0, m1, _MM_SHUFFLE!(2, 0, 2, 0));
    g1_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t0);
    t1 = shuffle2!(m0, m1, _MM_SHUFFLE!(3, 1, 3, 1));
    g2_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t1);
    diagonalize_sse41(&mut row0, &mut row2, &mut row3);
    t2 = shuffle2!(m2, m3, _MM_SHUFFLE!(2, 0, 2, 0));
    t2 = _mm_shuffle_epi32(t2, _MM_SHUFFLE!(2, 1, 0, 3));
    g1_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t2);
    t3 = shuffle2!(m2, m3, _MM_SHUFFLE!(3, 1, 3, 1));
    t3 = _mm_shuffle_epi32(t3, _MM_SHUFFLE!(2, 1, 0, 3));
    g2_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t3);
    undiagonalize_sse41(&mut row0, &mut row2, &mut row3);
    m0 = t0;
    m1 = t1;
    m2 = t2;
    m3 = t3;

    macro_rules! next_round_update {
      () => {{
        t0 = shuffle2!(m0, m1, _MM_SHUFFLE!(3, 1, 1, 2));
        t0 = _mm_shuffle_epi32(t0, _MM_SHUFFLE!(0, 3, 2, 1));
        g1_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t0);
        t1 = shuffle2!(m2, m3, _MM_SHUFFLE!(3, 3, 2, 2));
        tt = _mm_shuffle_epi32(m0, _MM_SHUFFLE!(0, 0, 3, 3));
        t1 = _mm_blend_epi16(tt, t1, 0xCC);
        g2_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t1);
        diagonalize_sse41(&mut row0, &mut row2, &mut row3);
        t2 = _mm_unpacklo_epi64(m3, m1);
        tt = _mm_blend_epi16(t2, m2, 0xC0);
        t2 = _mm_shuffle_epi32(tt, _MM_SHUFFLE!(1, 3, 2, 0));
        g1_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t2);
        t3 = _mm_unpackhi_epi32(m1, m3);
        tt = _mm_unpacklo_epi32(m2, t3);
        t3 = _mm_shuffle_epi32(tt, _MM_SHUFFLE!(0, 1, 3, 2));
        g2_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t3);
        undiagonalize_sse41(&mut row0, &mut row2, &mut row3);
        m0 = t0;
        m1 = t1;
        m2 = t2;
        m3 = t3;
      }};
    }

    macro_rules! next_round_final {
      () => {{
        t0 = shuffle2!(m0, m1, _MM_SHUFFLE!(3, 1, 1, 2));
        t0 = _mm_shuffle_epi32(t0, _MM_SHUFFLE!(0, 3, 2, 1));
        g1_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t0);
        t1 = shuffle2!(m2, m3, _MM_SHUFFLE!(3, 3, 2, 2));
        tt = _mm_shuffle_epi32(m0, _MM_SHUFFLE!(0, 0, 3, 3));
        t1 = _mm_blend_epi16(tt, t1, 0xCC);
        g2_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t1);
        diagonalize_sse41(&mut row0, &mut row2, &mut row3);
        t2 = _mm_unpacklo_epi64(m3, m1);
        tt = _mm_blend_epi16(t2, m2, 0xC0);
        t2 = _mm_shuffle_epi32(tt, _MM_SHUFFLE!(1, 3, 2, 0));
        g1_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t2);
        t3 = _mm_unpackhi_epi32(m1, m3);
        tt = _mm_unpacklo_epi32(m2, t3);
        t3 = _mm_shuffle_epi32(tt, _MM_SHUFFLE!(0, 1, 3, 2));
        g2_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t3);
        undiagonalize_sse41(&mut row0, &mut row2, &mut row3);
      }};
    }

    // Rounds 2..6
    next_round_update!();
    next_round_update!();
    next_round_update!();
    next_round_update!();
    next_round_update!();
    // Round 7
    next_round_final!();

    [row0, row1, row2, row3]
  }
}

#[cfg(target_arch = "x86_64")]
macro_rules! compress_cv_common_body {
  ($cv:expr, $m0:expr, $m1:expr, $m2:expr, $m3:expr, $counter:expr, $block_len:expr, $flags:expr) => {{
    let chaining_value = $cv;
    let m0 = $m0;
    let m1 = $m1;
    let m2 = $m2;
    let m3 = $m3;
    let counter = $counter;
    let block_len = $block_len;
    let flags = $flags;

    let rot16 = _mm_loadu_si128(ROT16_SHUFFLE.as_ptr().cast());
    let rot8 = _mm_loadu_si128(ROT8_SHUFFLE.as_ptr().cast());

    let cv_lo = _mm_loadu_si128(chaining_value.as_ptr().cast());
    let cv_hi = _mm_loadu_si128(chaining_value.as_ptr().add(4).cast());

    let mut row0 = cv_lo;
    let mut row1 = cv_hi;
    let mut row2 = _mm_loadu_si128(IV.as_ptr().cast());
    let mut row3 = _mm_set_epi32(flags as i32, block_len as i32, (counter >> 32) as i32, counter as i32);

    macro_rules! g {
      ($a:expr, $b:expr, $c:expr, $d:expr, $mx:expr, $my:expr) => {{
        $a = _mm_add_epi32($a, $b);
        $a = _mm_add_epi32($a, $mx);
        $d = _mm_xor_si128($d, $a);
        $d = _mm_shuffle_epi8($d, rot16);
        $c = _mm_add_epi32($c, $d);
        $b = _mm_xor_si128($b, $c);
        $b = _mm_or_si128(_mm_srli_epi32($b, 12), _mm_slli_epi32($b, 20));
        $a = _mm_add_epi32($a, $b);
        $a = _mm_add_epi32($a, $my);
        $d = _mm_xor_si128($d, $a);
        $d = _mm_shuffle_epi8($d, rot8);
        $c = _mm_add_epi32($c, $d);
        $b = _mm_xor_si128($b, $c);
        $b = _mm_or_si128(_mm_srli_epi32($b, 7), _mm_slli_epi32($b, 25));
      }};
    }

    macro_rules! round {
      ($mx0:expr, $my0:expr, $mx1:expr, $my1:expr) => {{
        g!(row0, row1, row2, row3, $mx0, $my0);

        row1 = _mm_shuffle_epi32(row1, 0b00_11_10_01);
        row2 = _mm_shuffle_epi32(row2, 0b01_00_11_10);
        row3 = _mm_shuffle_epi32(row3, 0b10_01_00_11);

        g!(row0, row1, row2, row3, $mx1, $my1);

        row1 = _mm_shuffle_epi32(row1, 0b10_01_00_11);
        row2 = _mm_shuffle_epi32(row2, 0b01_00_11_10);
        row3 = _mm_shuffle_epi32(row3, 0b00_11_10_01);
      }};
    }

    let (mx0, my0, mx1, my1) = permute_round_0(m0, m1, m2, m3);
    round!(mx0, my0, mx1, my1);
    let (mx0, my0, mx1, my1) = permute_round_1(m0, m1, m2, m3);
    round!(mx0, my0, mx1, my1);
    let (mx0, my0, mx1, my1) = permute_round_2(m0, m1, m2, m3);
    round!(mx0, my0, mx1, my1);
    let (mx0, my0, mx1, my1) = permute_round_3(m0, m1, m2, m3);
    round!(mx0, my0, mx1, my1);
    let (mx0, my0, mx1, my1) = permute_round_4(m0, m1, m2, m3);
    round!(mx0, my0, mx1, my1);
    let (mx0, my0, mx1, my1) = permute_round_5(m0, m1, m2, m3);
    round!(mx0, my0, mx1, my1);
    let (mx0, my0, mx1, my1) = permute_round_6(m0, m1, m2, m3);
    round!(mx0, my0, mx1, my1);

    row0 = _mm_xor_si128(row0, row2);
    row1 = _mm_xor_si128(row1, row3);

    let mut out = [0u32; 8];
    _mm_storeu_si128(out.as_mut_ptr().cast(), row0);
    _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), row1);
    out
  }};
}

/// SSSE3 compress that returns only the chaining value (first 8 words).
///
/// This avoids storing the full 16-word output and avoids building a
/// temporary `[u32; 16]` message array when the caller already has bytes.
///
/// # Safety
/// Caller must ensure SSSE3 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn compress_cv_ssse3(
  chaining_value: &[u32; 8],
  m0: __m128i,
  m1: __m128i,
  m2: __m128i,
  m3: __m128i,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  compress_cv_common_body!(chaining_value, m0, m1, m2, m3, counter, block_len, flags)
}

/// AVX2+SSSE3 compress that returns only the chaining value (first 8 words).
///
/// This mirrors `compress_cv_ssse3`, but is compiled under AVX2 to encourage
/// VEX-encoded 128-bit operations for better mixed-workload behavior.
///
/// # Safety
/// Caller must ensure AVX2 + SSSE3 are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,sse4.1,ssse3")]
unsafe fn compress_cv_avx2(
  chaining_value: &[u32; 8],
  m0: __m128i,
  m1: __m128i,
  m2: __m128i,
  m3: __m128i,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  let [row0, row1, row2, row3] = compress_pre_sse41_impl(chaining_value, m0, m1, m2, m3, counter, block_len, flags);
  let row0 = _mm_xor_si128(row0, row2);
  let row1 = _mm_xor_si128(row1, row3);
  let mut out = [0u32; 8];
  _mm_storeu_si128(out.as_mut_ptr().cast(), row0);
  _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), row1);
  out
}

/// AVX-512+AVX2+SSSE3 compress that returns only the chaining value (first 8 words).
///
/// # Safety
/// Caller must ensure the declared AVX-512 + AVX2 + SSSE3 features are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2,sse4.1,ssse3")]
unsafe fn compress_cv_avx512(
  chaining_value: &[u32; 8],
  m0: __m128i,
  m1: __m128i,
  m2: __m128i,
  m3: __m128i,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  let [row0, row1, row2, row3] = compress_pre_sse41_impl(chaining_value, m0, m1, m2, m3, counter, block_len, flags);
  let row0 = _mm_xor_si128(row0, row2);
  let row1 = _mm_xor_si128(row1, row3);
  let mut out = [0u32; 8];
  _mm_storeu_si128(out.as_mut_ptr().cast(), row0);
  _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), row1);
  out
}

/// BLAKE3 compress function using AVX2-enabled codegen.
///
/// Note: This is still a single-block (dependency-chained) compressor. AVX2
/// doesn't unlock extra parallelism inside the algorithm, but it does allow
/// the compiler to use VEX-encoded integer ops and avoid AVX<->SSE transition
/// penalties when interleaving with AVX2 throughput kernels.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,sse4.1,ssse3")]
pub unsafe fn compress_avx2(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  let m0 = _mm_loadu_si128(block_words.as_ptr().cast());
  let m1 = _mm_loadu_si128(block_words.as_ptr().add(4).cast());
  let m2 = _mm_loadu_si128(block_words.as_ptr().add(8).cast());
  let m3 = _mm_loadu_si128(block_words.as_ptr().add(12).cast());
  let [mut row0, mut row1, mut row2, mut row3] =
    compress_pre_sse41_impl(chaining_value, m0, m1, m2, m3, counter, block_len, flags);

  let cv_lo = _mm_loadu_si128(chaining_value.as_ptr().cast());
  let cv_hi = _mm_loadu_si128(chaining_value.as_ptr().add(4).cast());

  row0 = _mm_xor_si128(row0, row2);
  row1 = _mm_xor_si128(row1, row3);
  row2 = _mm_xor_si128(row2, cv_lo);
  row3 = _mm_xor_si128(row3, cv_hi);

  let mut out = [0u32; 16];
  _mm_storeu_si128(out.as_mut_ptr().cast(), row0);
  _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), row1);
  _mm_storeu_si128(out.as_mut_ptr().add(8).cast(), row2);
  _mm_storeu_si128(out.as_mut_ptr().add(12).cast(), row3);
  out
}

/// BLAKE3 compress function using SSE4.1-enabled codegen.
///
/// This is a thin wrapper around the SSSE3 row-wise compressor. SSE4.1 implies
/// (and our dispatcher requires) SSSE3, and this keeps the per-block hot path
/// fully SIMD without maintaining duplicate implementations.
///
/// # Safety
/// Caller must ensure SSE4.1 + SSSE3 are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1,ssse3")]
pub unsafe fn compress_sse41(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  let m0 = _mm_loadu_si128(block_words.as_ptr().cast());
  let m1 = _mm_loadu_si128(block_words.as_ptr().add(4).cast());
  let m2 = _mm_loadu_si128(block_words.as_ptr().add(8).cast());
  let m3 = _mm_loadu_si128(block_words.as_ptr().add(12).cast());
  let [mut row0, mut row1, mut row2, mut row3] =
    compress_pre_sse41_impl(chaining_value, m0, m1, m2, m3, counter, block_len, flags);

  let cv_lo = _mm_loadu_si128(chaining_value.as_ptr().cast());
  let cv_hi = _mm_loadu_si128(chaining_value.as_ptr().add(4).cast());

  row0 = _mm_xor_si128(row0, row2);
  row1 = _mm_xor_si128(row1, row3);
  row2 = _mm_xor_si128(row2, cv_lo);
  row3 = _mm_xor_si128(row3, cv_hi);

  let mut out = [0u32; 16];
  _mm_storeu_si128(out.as_mut_ptr().cast(), row0);
  _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), row1);
  _mm_storeu_si128(out.as_mut_ptr().add(8).cast(), row2);
  _mm_storeu_si128(out.as_mut_ptr().add(12).cast(), row3);
  out
}

/// SSE4.1 chunk compression: process multiple 64-byte blocks.
///
/// # Safety
/// Caller must ensure SSE4.1 + SSSE3 are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1,ssse3")]
pub unsafe fn chunk_compress_blocks_sse41(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);

  #[inline(always)]
  unsafe fn compress_cv_sse41_block(cv: &[u32; 8], block: *const u8, counter: u64, flags: u32) -> [u32; 8] {
    let (m0, m1, m2, m3) = unsafe { load_msg_vecs(block) };
    let [row0, row1, row2, row3] =
      unsafe { compress_pre_sse41_impl(cv, m0, m1, m2, m3, counter, BLOCK_LEN as u32, flags) };
    let row0 = unsafe { _mm_xor_si128(row0, row2) };
    let row1 = unsafe { _mm_xor_si128(row1, row3) };
    let mut out = [0u32; 8];
    unsafe {
      _mm_storeu_si128(out.as_mut_ptr().cast(), row0);
      _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), row1);
    }
    out
  }

  if blocks.len() == BLOCK_LEN {
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    *chaining_value = unsafe { compress_cv_sse41_block(chaining_value, blocks.as_ptr(), chunk_counter, flags | start) };
    *blocks_compressed = blocks_compressed.wrapping_add(1);
    return;
  }

  let (block_slices, remainder) = blocks.as_chunks::<BLOCK_LEN>();
  debug_assert!(remainder.is_empty());
  for block_bytes in block_slices {
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    *chaining_value =
      unsafe { compress_cv_sse41_block(chaining_value, block_bytes.as_ptr(), chunk_counter, flags | start) };
    *blocks_compressed = blocks_compressed.wrapping_add(1);
  }
}

/// SSE4.1 parent CV computation.
///
/// # Safety
/// Caller must ensure SSE4.1 + SSSE3 are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1,ssse3")]
pub unsafe fn parent_cv_sse41(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  let m0 = _mm_loadu_si128(left_child_cv.as_ptr().cast());
  let m1 = _mm_loadu_si128(left_child_cv.as_ptr().add(4).cast());
  let m2 = _mm_loadu_si128(right_child_cv.as_ptr().cast());
  let m3 = _mm_loadu_si128(right_child_cv.as_ptr().add(4).cast());
  let [row0, row1, row2, row3] =
    compress_pre_sse41_impl(&key_words, m0, m1, m2, m3, 0, BLOCK_LEN as u32, PARENT | flags);
  let row0 = _mm_xor_si128(row0, row2);
  let row1 = _mm_xor_si128(row1, row3);
  let mut out = [0u32; 8];
  _mm_storeu_si128(out.as_mut_ptr().cast(), row0);
  _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), row1);
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
  let mx0 = gather4::<2, 3, 7, 4>(m0, m1, m2, m3);
  let my0 = gather4::<6, 10, 0, 13>(m0, m1, m2, m3);
  let mx1 = gather4::<1, 12, 9, 15>(m0, m1, m2, m3);
  let my1 = gather4::<11, 5, 14, 8>(m0, m1, m2, m3);
  (mx0, my0, mx1, my1)
}

/// Round 2: m[3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn permute_round_2(m0: __m128i, m1: __m128i, m2: __m128i, m3: __m128i) -> (__m128i, __m128i, __m128i, __m128i) {
  let mx0 = gather4::<3, 10, 13, 7>(m0, m1, m2, m3);
  let my0 = gather4::<4, 12, 2, 14>(m0, m1, m2, m3);
  let mx1 = gather4::<6, 9, 11, 8>(m0, m1, m2, m3);
  let my1 = gather4::<5, 0, 15, 1>(m0, m1, m2, m3);
  (mx0, my0, mx1, my1)
}

/// Round 3: m[10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn permute_round_3(m0: __m128i, m1: __m128i, m2: __m128i, m3: __m128i) -> (__m128i, __m128i, __m128i, __m128i) {
  let mx0 = gather4::<10, 12, 14, 13>(m0, m1, m2, m3);
  let my0 = gather4::<7, 9, 3, 15>(m0, m1, m2, m3);
  let mx1 = gather4::<4, 11, 5, 1>(m0, m1, m2, m3);
  let my1 = gather4::<0, 2, 8, 6>(m0, m1, m2, m3);
  (mx0, my0, mx1, my1)
}

/// Round 4: m[12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn permute_round_4(m0: __m128i, m1: __m128i, m2: __m128i, m3: __m128i) -> (__m128i, __m128i, __m128i, __m128i) {
  let mx0 = gather4::<12, 9, 15, 14>(m0, m1, m2, m3);
  let my0 = gather4::<13, 11, 10, 8>(m0, m1, m2, m3);
  let mx1 = gather4::<7, 5, 0, 6>(m0, m1, m2, m3);
  let my1 = gather4::<2, 3, 1, 4>(m0, m1, m2, m3);
  (mx0, my0, mx1, my1)
}

/// Round 5: m[9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn permute_round_5(m0: __m128i, m1: __m128i, m2: __m128i, m3: __m128i) -> (__m128i, __m128i, __m128i, __m128i) {
  let mx0 = gather4::<9, 11, 8, 15>(m0, m1, m2, m3);
  let my0 = gather4::<14, 5, 12, 1>(m0, m1, m2, m3);
  let mx1 = gather4::<13, 0, 2, 4>(m0, m1, m2, m3);
  let my1 = gather4::<3, 10, 6, 7>(m0, m1, m2, m3);
  (mx0, my0, mx1, my1)
}

/// Round 6: m[11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn permute_round_6(m0: __m128i, m1: __m128i, m2: __m128i, m3: __m128i) -> (__m128i, __m128i, __m128i, __m128i) {
  let mx0 = gather4::<11, 5, 1, 8>(m0, m1, m2, m3);
  let my0 = gather4::<15, 0, 9, 6>(m0, m1, m2, m3);
  let mx1 = gather4::<14, 2, 3, 7>(m0, m1, m2, m3);
  let my1 = gather4::<10, 12, 4, 13>(m0, m1, m2, m3);
  (mx0, my0, mx1, my1)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn bcast_word<const I: usize>(m0: __m128i, m1: __m128i, m2: __m128i, m3: __m128i) -> __m128i {
  debug_assert!(I < 16);
  match I {
    0 => _mm_shuffle_epi32(m0, 0x00),
    1 => _mm_shuffle_epi32(m0, 0x55),
    2 => _mm_shuffle_epi32(m0, 0xAA),
    3 => _mm_shuffle_epi32(m0, 0xFF),
    4 => _mm_shuffle_epi32(m1, 0x00),
    5 => _mm_shuffle_epi32(m1, 0x55),
    6 => _mm_shuffle_epi32(m1, 0xAA),
    7 => _mm_shuffle_epi32(m1, 0xFF),
    8 => _mm_shuffle_epi32(m2, 0x00),
    9 => _mm_shuffle_epi32(m2, 0x55),
    10 => _mm_shuffle_epi32(m2, 0xAA),
    11 => _mm_shuffle_epi32(m2, 0xFF),
    12 => _mm_shuffle_epi32(m3, 0x00),
    13 => _mm_shuffle_epi32(m3, 0x55),
    14 => _mm_shuffle_epi32(m3, 0xAA),
    15 => _mm_shuffle_epi32(m3, 0xFF),
    _ => core::hint::unreachable_unchecked(),
  }
}

/// Gather 4 specific message words into one vector, without scalar extraction.
///
/// Output lanes are `[A, B, C, D]` (u32 lane order).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn gather4<const A: usize, const B: usize, const C: usize, const D: usize>(
  m0: __m128i,
  m1: __m128i,
  m2: __m128i,
  m3: __m128i,
) -> __m128i {
  debug_assert!(A < 16 && B < 16 && C < 16 && D < 16);
  let ab = _mm_unpacklo_epi32(bcast_word::<A>(m0, m1, m2, m3), bcast_word::<B>(m0, m1, m2, m3));
  let cd = _mm_unpacklo_epi32(bcast_word::<C>(m0, m1, m2, m3), bcast_word::<D>(m0, m1, m2, m3));
  _mm_unpacklo_epi64(ab, cd)
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

  // Hot path for streaming callers that feed one full block at a time.
  if blocks.len() == BLOCK_LEN {
    // SAFETY: `blocks` is exactly one full block.
    let block_bytes: &[u8; BLOCK_LEN] = unsafe { &*(blocks.as_ptr().cast()) };
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    let (m0, m1, m2, m3) = unsafe { load_msg_vecs(block_bytes.as_ptr()) };
    *chaining_value = unsafe {
      compress_cv_ssse3(
        chaining_value,
        m0,
        m1,
        m2,
        m3,
        chunk_counter,
        BLOCK_LEN as u32,
        flags | start,
      )
    };
    *blocks_compressed = blocks_compressed.wrapping_add(1);
    return;
  }

  let (block_slices, remainder) = blocks.as_chunks::<BLOCK_LEN>();
  debug_assert!(remainder.is_empty());
  for block_bytes in block_slices {
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    let (m0, m1, m2, m3) = unsafe { load_msg_vecs(block_bytes.as_ptr()) };
    *chaining_value = unsafe {
      compress_cv_ssse3(
        chaining_value,
        m0,
        m1,
        m2,
        m3,
        chunk_counter,
        BLOCK_LEN as u32,
        flags | start,
      )
    };
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
  let m0 = _mm_loadu_si128(left_child_cv.as_ptr().cast());
  let m1 = _mm_loadu_si128(left_child_cv.as_ptr().add(4).cast());
  let m2 = _mm_loadu_si128(right_child_cv.as_ptr().cast());
  let m3 = _mm_loadu_si128(right_child_cv.as_ptr().add(4).cast());
  unsafe { compress_cv_ssse3(&key_words, m0, m1, m2, m3, 0, BLOCK_LEN as u32, PARENT | flags) }
}

/// AVX2 chunk compression: process multiple 64-byte blocks.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,sse4.1,ssse3")]
pub unsafe fn chunk_compress_blocks_avx2(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  // For now, keep the AVX2 per-block path identical to SSSE3 but with AVX2
  // enabled so the compiler can emit VEX-encoded 128-bit ops and avoid
  // AVX<->SSE transition penalties in mixed workloads.
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);

  if blocks.len() == BLOCK_LEN {
    // SAFETY: `blocks` is exactly one full block.
    let block_bytes: &[u8; BLOCK_LEN] = unsafe { &*(blocks.as_ptr().cast()) };
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    let (m0, m1, m2, m3) = unsafe { load_msg_vecs(block_bytes.as_ptr()) };
    // SAFETY: this function is AVX2+SSSE3-gated.
    *chaining_value = unsafe {
      compress_cv_avx2(
        chaining_value,
        m0,
        m1,
        m2,
        m3,
        chunk_counter,
        BLOCK_LEN as u32,
        flags | start,
      )
    };
    *blocks_compressed = blocks_compressed.wrapping_add(1);
    return;
  }

  let (block_slices, remainder) = blocks.as_chunks::<BLOCK_LEN>();
  debug_assert!(remainder.is_empty());
  for block_bytes in block_slices {
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    let (m0, m1, m2, m3) = unsafe { load_msg_vecs(block_bytes.as_ptr()) };
    // SAFETY: this function is AVX2+SSSE3-gated.
    *chaining_value = unsafe {
      compress_cv_avx2(
        chaining_value,
        m0,
        m1,
        m2,
        m3,
        chunk_counter,
        BLOCK_LEN as u32,
        flags | start,
      )
    };
    *blocks_compressed = blocks_compressed.wrapping_add(1);
  }
}

/// AVX2 parent CV computation.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,sse4.1,ssse3")]
pub unsafe fn parent_cv_avx2(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  let m0 = _mm_loadu_si128(left_child_cv.as_ptr().cast());
  let m1 = _mm_loadu_si128(left_child_cv.as_ptr().add(4).cast());
  let m2 = _mm_loadu_si128(right_child_cv.as_ptr().cast());
  let m3 = _mm_loadu_si128(right_child_cv.as_ptr().add(4).cast());
  unsafe { compress_cv_avx2(&key_words, m0, m1, m2, m3, 0, BLOCK_LEN as u32, PARENT | flags) }
}

/// BLAKE3 compress function using AVX-512-enabled codegen.
///
/// Like the AVX2 entrypoint, this is still a single-block compressor. The
/// primary benefit is avoiding transition penalties when the surrounding
/// workload uses AVX-512 throughput kernels.
///
/// # Safety
/// Caller must ensure AVX-512 + AVX2 are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2,sse4.1,ssse3")]
pub unsafe fn compress_avx512(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  // SAFETY: caller guarantees AVX2 (and thus `compress_avx2`'s requirements).
  unsafe { compress_avx2(chaining_value, block_words, counter, block_len, flags) }
}

/// AVX-512 chunk compression: process multiple 64-byte blocks.
///
/// # Safety
/// Caller must ensure AVX-512 + AVX2 are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2,sse4.1,ssse3")]
pub unsafe fn chunk_compress_blocks_avx512(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);

  if blocks.len() == BLOCK_LEN {
    // SAFETY: `blocks` is exactly one full block.
    let block_bytes: &[u8; BLOCK_LEN] = unsafe { &*(blocks.as_ptr().cast()) };
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    let (m0, m1, m2, m3) = unsafe { load_msg_vecs(block_bytes.as_ptr()) };
    *chaining_value = unsafe {
      compress_cv_avx512(
        chaining_value,
        m0,
        m1,
        m2,
        m3,
        chunk_counter,
        BLOCK_LEN as u32,
        flags | start,
      )
    };
    *blocks_compressed = blocks_compressed.wrapping_add(1);
    return;
  }

  let (block_slices, remainder) = blocks.as_chunks::<BLOCK_LEN>();
  debug_assert!(remainder.is_empty());
  for block_bytes in block_slices {
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    let (m0, m1, m2, m3) = unsafe { load_msg_vecs(block_bytes.as_ptr()) };
    *chaining_value = unsafe {
      compress_cv_avx512(
        chaining_value,
        m0,
        m1,
        m2,
        m3,
        chunk_counter,
        BLOCK_LEN as u32,
        flags | start,
      )
    };
    *blocks_compressed = blocks_compressed.wrapping_add(1);
  }
}

/// AVX-512 parent CV computation.
///
/// # Safety
/// Caller must ensure AVX-512 + AVX2 are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2,sse4.1,ssse3")]
pub unsafe fn parent_cv_avx512(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  let m0 = _mm_loadu_si128(left_child_cv.as_ptr().cast());
  let m1 = _mm_loadu_si128(left_child_cv.as_ptr().add(4).cast());
  let m2 = _mm_loadu_si128(right_child_cv.as_ptr().cast());
  let m3 = _mm_loadu_si128(right_child_cv.as_ptr().add(4).cast());
  unsafe { compress_cv_avx512(&key_words, m0, m1, m2, m3, 0, BLOCK_LEN as u32, PARENT | flags) }
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
