//! BLAKE3 aarch64 NEON kernels.
//!
//! This module provides SIMD-accelerated compression functions for BLAKE3 using
//! aarch64 NEON intrinsics.
//!
//! # Optimizations
//!
//! - Uses `vsriq_n_u32` (shift-right-insert) for rot12/rot7 instead of shift+OR
//! - Uses `vrev32q_u16` for rot16 (single instruction)
//! - Uses `vqtbl1q_u8` (table lookup) for rot8 (single instruction)
//! - Implements 4-way parallel chunk hashing via `hash4_neon`
//!
//! # Safety
//!
//! All functions in this module are marked `unsafe` and require NEON
//! to be present. Callers must verify CPU capabilities before calling.

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
use super::MSG_SCHEDULE;
use super::{BLOCK_LEN, CHUNK_LEN, CHUNK_START, IV, OUT_LEN, PARENT, first_8_words, words16_from_le_bytes_64};

/// SIMD degree for NEON (4 parallel lanes).
#[allow(dead_code)]
pub const SIMD_DEGREE: usize = 4;

// ─────────────────────────────────────────────────────────────────────────────
// Rotation helpers for NEON - Optimized versions
// ─────────────────────────────────────────────────────────────────────────────

/// Static shuffle table for 8-bit rotation.
/// Each u32 [b0, b1, b2, b3] becomes [b1, b2, b3, b0].
static ROT8_TABLE: [u8; 16] = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12];

/// Rotate right by 16 bits (each u32 lane).
/// Uses vrev32q_u16 which reverses 16-bit elements within 32-bit containers.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn rotr16(v: uint32x4_t) -> uint32x4_t {
  let v16 = vreinterpretq_u16_u32(v);
  let rotated = vrev32q_u16(v16);
  vreinterpretq_u32_u16(rotated)
}

/// Rotate right by 12 bits (each u32 lane).
/// Uses shift+OR which allows parallel execution on superscalar cores.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn rotr12(v: uint32x4_t) -> uint32x4_t {
  // rotr(x, 12) = (x >> 12) | (x << 20)
  //
  // Use "shift-left-insert" to form the OR without an extra instruction.
  // This maps to a single `vsli` on aarch64.
  vsliq_n_u32(vshrq_n_u32(v, 12), v, 20)
}

/// Rotate right by 8 bits (each u32 lane).
/// Uses vqtbl1q_u8 (table lookup) for byte-level rotation - single cycle.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn rotr8(v: uint32x4_t) -> uint32x4_t {
  // Load shuffle table from static - compiler will hoist this out of loops
  let tbl: uint8x16_t = vld1q_u8(ROT8_TABLE.as_ptr());
  let bytes = vreinterpretq_u8_u32(v);
  let rotated = vqtbl1q_u8(bytes, tbl);
  vreinterpretq_u32_u8(rotated)
}

/// Rotate right by 7 bits (each u32 lane).
/// Uses shift+OR which allows parallel execution on superscalar cores.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn rotr7(v: uint32x4_t) -> uint32x4_t {
  // rotr(x, 7) = (x >> 7) | (x << 25)
  //
  // Use "shift-left-insert" to form the OR without an extra instruction.
  vsliq_n_u32(vshrq_n_u32(v, 7), v, 25)
}

// ─────────────────────────────────────────────────────────────────────────────
// Lane rotation helpers (for diagonalization)
// ─────────────────────────────────────────────────────────────────────────────

/// Rotate lanes left by 1: [a, b, c, d] -> [b, c, d, a]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn rot_lanes_left_1(v: uint32x4_t) -> uint32x4_t {
  vextq_u32(v, v, 1)
}

/// Rotate lanes left by 2: [a, b, c, d] -> [c, d, a, b]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn rot_lanes_left_2(v: uint32x4_t) -> uint32x4_t {
  vextq_u32(v, v, 2)
}

/// Rotate lanes left by 3: [a, b, c, d] -> [d, a, b, c]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn rot_lanes_left_3(v: uint32x4_t) -> uint32x4_t {
  vextq_u32(v, v, 3)
}

// ─────────────────────────────────────────────────────────────────────────────
// Transpose operations
// ─────────────────────────────────────────────────────────────────────────────

/// Transpose 4 uint32x4_t vectors (4x4 matrix transpose).
///
/// Input:
/// ```text
///   vecs[0] = [a0, a1, a2, a3]
///   vecs[1] = [b0, b1, b2, b3]
///   vecs[2] = [c0, c1, c2, c3]
///   vecs[3] = [d0, d1, d2, d3]
/// ```
///
/// Output:
/// ```text
///   vecs[0] = [a0, b0, c0, d0]
///   vecs[1] = [a1, b1, c1, d1]
///   vecs[2] = [a2, b2, c2, d2]
///   vecs[3] = [a3, b3, c3, d3]
/// ```
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(dead_code)]
unsafe fn transpose_vecs(vecs: &mut [uint32x4_t; 4]) {
  // Step 1: Transpose 2x2 sub-matrices using vtrnq_u32
  let rows01 = vtrnq_u32(vecs[0], vecs[1]);
  let rows23 = vtrnq_u32(vecs[2], vecs[3]);

  // Step 2: Swap top-right and bottom-left 2x2 blocks
  vecs[0] = vcombine_u32(vget_low_u32(rows01.0), vget_low_u32(rows23.0));
  vecs[1] = vcombine_u32(vget_low_u32(rows01.1), vget_low_u32(rows23.1));
  vecs[2] = vcombine_u32(vget_high_u32(rows01.0), vget_high_u32(rows23.0));
  vecs[3] = vcombine_u32(vget_high_u32(rows01.1), vget_high_u32(rows23.1));
}

/// Load and transpose message words from 4 input blocks.
/// After transpose, m[i] contains word i from all 4 inputs.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load_msg_vecs_transposed(inputs: [*const u8; 4], block_offset: usize, block_len: usize) -> [uint32x4_t; 16] {
  debug_assert!(
    cfg!(target_endian = "little"),
    "aarch64 NEON implementation assumes little-endian"
  );

  #[inline(always)]
  unsafe fn loadu_128(src: *const u8) -> uint32x4_t {
    // `vld1q_u8` has no alignment requirements.
    vreinterpretq_u32_u8(vld1q_u8(src))
  }

  // Fast path: full 64-byte block.
  if block_len == BLOCK_LEN {
    let mut out = [vdupq_n_u32(0); 16];

    // Load four 16-byte chunks per input and transpose each 4x4.
    for lane_block in 0..4 {
      let off = block_offset + lane_block * 16;
      let mut vecs = [
        loadu_128(inputs[0].add(off)),
        loadu_128(inputs[1].add(off)),
        loadu_128(inputs[2].add(off)),
        loadu_128(inputs[3].add(off)),
      ];
      transpose_vecs(&mut vecs);
      out[lane_block * 4 + 0] = vecs[0];
      out[lane_block * 4 + 1] = vecs[1];
      out[lane_block * 4 + 2] = vecs[2];
      out[lane_block * 4 + 3] = vecs[3];
    }

    return out;
  }

  // Slow path: pad partial blocks to 64 bytes before loading.
  let mut msg_words = [[0u32; 4]; 16];
  for (lane, &input) in inputs.iter().enumerate() {
    let mut block = [0u8; BLOCK_LEN];
    if block_len != 0 {
      core::ptr::copy_nonoverlapping(input.add(block_offset), block.as_mut_ptr(), block_len);
    }
    let words = words16_from_le_bytes_64(&block);
    for j in 0..16 {
      msg_words[j][lane] = words[j];
    }
  }

  let mut out = [vdupq_n_u32(0); 16];
  for i in 0..16 {
    out[i] = vld1q_u32(msg_words[i].as_ptr());
  }
  out
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-block compression (NEON accelerated)
// ─────────────────────────────────────────────────────────────────────────────

/// BLAKE3 compress function using NEON intrinsics.
///
/// # Safety
///
/// Caller must ensure NEON is available (`target_feature = "neon"`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn compress_neon(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  // Load state into 4 vectors (rows of the BLAKE3 state matrix)
  let mut row0 = vld1q_u32(chaining_value.as_ptr());
  let mut row1 = vld1q_u32(chaining_value.as_ptr().add(4));
  let mut row2 = vld1q_u32(IV.as_ptr());

  // Build row3 from counter, block_len, and flags
  let counter_lo = counter as u32;
  let counter_hi = (counter >> 32) as u32;
  let row3_arr: [u32; 4] = [counter_lo, counter_hi, block_len, flags];
  let mut row3 = vld1q_u32(row3_arr.as_ptr());

  // Load message words
  let m0 = vld1q_u32(block_words.as_ptr());
  let m1 = vld1q_u32(block_words.as_ptr().add(4));
  let m2 = vld1q_u32(block_words.as_ptr().add(8));
  let m3 = vld1q_u32(block_words.as_ptr().add(12));

  // G function macro
  macro_rules! g {
    ($a:expr, $b:expr, $c:expr, $d:expr, $mx:expr, $my:expr) => {{
      $a = vaddq_u32($a, $b);
      $a = vaddq_u32($a, $mx);
      $d = veorq_u32($d, $a);
      $d = rotr16($d);
      $c = vaddq_u32($c, $d);
      $b = veorq_u32($b, $c);
      $b = rotr12($b);
      $a = vaddq_u32($a, $b);
      $a = vaddq_u32($a, $my);
      $d = veorq_u32($d, $a);
      $d = rotr8($d);
      $c = vaddq_u32($c, $d);
      $b = veorq_u32($b, $c);
      $b = rotr7($b);
    }};
  }

  // One round: column step + diagonal step
  macro_rules! round {
    ($mx0:expr, $my0:expr, $mx1:expr, $my1:expr) => {{
      // Column step
      g!(row0, row1, row2, row3, $mx0, $my0);

      // Diagonalize
      row1 = rot_lanes_left_1(row1);
      row2 = rot_lanes_left_2(row2);
      row3 = rot_lanes_left_3(row3);

      // Diagonal step
      g!(row0, row1, row2, row3, $mx1, $my1);

      // Undiagonalize
      row1 = rot_lanes_left_3(row1);
      row2 = rot_lanes_left_2(row2);
      row3 = rot_lanes_left_1(row3);
    }};
  }

  // Execute all 7 rounds with the BLAKE3 message schedule
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

  // Finalization
  let cv_lo = vld1q_u32(chaining_value.as_ptr());
  let cv_hi = vld1q_u32(chaining_value.as_ptr().add(4));

  row0 = veorq_u32(row0, row2);
  row1 = veorq_u32(row1, row3);
  row2 = veorq_u32(row2, cv_lo);
  row3 = veorq_u32(row3, cv_hi);

  // Store result
  let mut out = [0u32; 16];
  vst1q_u32(out.as_mut_ptr(), row0);
  vst1q_u32(out.as_mut_ptr().add(4), row1);
  vst1q_u32(out.as_mut_ptr().add(8), row2);
  vst1q_u32(out.as_mut_ptr().add(12), row3);
  out
}

// ─────────────────────────────────────────────────────────────────────────────
// Message permutation helpers (for single-block compression)
// ─────────────────────────────────────────────────────────────────────────────

/// Helper to extract a single word from the message vectors.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn extract_word(m0: uint32x4_t, m1: uint32x4_t, m2: uint32x4_t, m3: uint32x4_t, idx: usize) -> u32 {
  match idx {
    0 => vgetq_lane_u32(m0, 0),
    1 => vgetq_lane_u32(m0, 1),
    2 => vgetq_lane_u32(m0, 2),
    3 => vgetq_lane_u32(m0, 3),
    4 => vgetq_lane_u32(m1, 0),
    5 => vgetq_lane_u32(m1, 1),
    6 => vgetq_lane_u32(m1, 2),
    7 => vgetq_lane_u32(m1, 3),
    8 => vgetq_lane_u32(m2, 0),
    9 => vgetq_lane_u32(m2, 1),
    10 => vgetq_lane_u32(m2, 2),
    11 => vgetq_lane_u32(m2, 3),
    12 => vgetq_lane_u32(m3, 0),
    13 => vgetq_lane_u32(m3, 1),
    14 => vgetq_lane_u32(m3, 2),
    _ => vgetq_lane_u32(m3, 3),
  }
}

/// Build a vector from 4 specific message word indices.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn blend_words(
  m0: uint32x4_t,
  m1: uint32x4_t,
  m2: uint32x4_t,
  m3: uint32x4_t,
  i0: usize,
  i1: usize,
  i2: usize,
  i3: usize,
) -> uint32x4_t {
  let arr: [u32; 4] = [
    extract_word(m0, m1, m2, m3, i0),
    extract_word(m0, m1, m2, m3, i1),
    extract_word(m0, m1, m2, m3, i2),
    extract_word(m0, m1, m2, m3, i3),
  ];
  vld1q_u32(arr.as_ptr())
}

/// Round 0: identity permutation
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn permute_round_0(
  m0: uint32x4_t,
  m1: uint32x4_t,
  m2: uint32x4_t,
  m3: uint32x4_t,
) -> (uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t) {
  let mx0 = blend_words(m0, m1, m2, m3, 0, 2, 4, 6);
  let my0 = blend_words(m0, m1, m2, m3, 1, 3, 5, 7);
  let mx1 = blend_words(m0, m1, m2, m3, 8, 10, 12, 14);
  let my1 = blend_words(m0, m1, m2, m3, 9, 11, 13, 15);
  (mx0, my0, mx1, my1)
}

/// Round 1: m[2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn permute_round_1(
  m0: uint32x4_t,
  m1: uint32x4_t,
  m2: uint32x4_t,
  m3: uint32x4_t,
) -> (uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t) {
  let mx0 = blend_words(m0, m1, m2, m3, 2, 3, 7, 4);
  let my0 = blend_words(m0, m1, m2, m3, 6, 10, 0, 13);
  let mx1 = blend_words(m0, m1, m2, m3, 1, 12, 9, 15);
  let my1 = blend_words(m0, m1, m2, m3, 11, 5, 14, 8);
  (mx0, my0, mx1, my1)
}

/// Round 2: m[3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn permute_round_2(
  m0: uint32x4_t,
  m1: uint32x4_t,
  m2: uint32x4_t,
  m3: uint32x4_t,
) -> (uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t) {
  let mx0 = blend_words(m0, m1, m2, m3, 3, 10, 13, 7);
  let my0 = blend_words(m0, m1, m2, m3, 4, 12, 2, 14);
  let mx1 = blend_words(m0, m1, m2, m3, 6, 9, 11, 8);
  let my1 = blend_words(m0, m1, m2, m3, 5, 0, 15, 1);
  (mx0, my0, mx1, my1)
}

/// Round 3: m[10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn permute_round_3(
  m0: uint32x4_t,
  m1: uint32x4_t,
  m2: uint32x4_t,
  m3: uint32x4_t,
) -> (uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t) {
  let mx0 = blend_words(m0, m1, m2, m3, 10, 12, 14, 13);
  let my0 = blend_words(m0, m1, m2, m3, 7, 9, 3, 15);
  let mx1 = blend_words(m0, m1, m2, m3, 4, 11, 5, 1);
  let my1 = blend_words(m0, m1, m2, m3, 0, 2, 8, 6);
  (mx0, my0, mx1, my1)
}

/// Round 4: m[12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn permute_round_4(
  m0: uint32x4_t,
  m1: uint32x4_t,
  m2: uint32x4_t,
  m3: uint32x4_t,
) -> (uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t) {
  let mx0 = blend_words(m0, m1, m2, m3, 12, 9, 15, 14);
  let my0 = blend_words(m0, m1, m2, m3, 13, 11, 10, 8);
  let mx1 = blend_words(m0, m1, m2, m3, 7, 5, 0, 6);
  let my1 = blend_words(m0, m1, m2, m3, 2, 3, 1, 4);
  (mx0, my0, mx1, my1)
}

/// Round 5: m[9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn permute_round_5(
  m0: uint32x4_t,
  m1: uint32x4_t,
  m2: uint32x4_t,
  m3: uint32x4_t,
) -> (uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t) {
  let mx0 = blend_words(m0, m1, m2, m3, 9, 11, 8, 15);
  let my0 = blend_words(m0, m1, m2, m3, 14, 5, 12, 1);
  let mx1 = blend_words(m0, m1, m2, m3, 13, 0, 2, 4);
  let my1 = blend_words(m0, m1, m2, m3, 3, 10, 6, 7);
  (mx0, my0, mx1, my1)
}

/// Round 6: m[11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn permute_round_6(
  m0: uint32x4_t,
  m1: uint32x4_t,
  m2: uint32x4_t,
  m3: uint32x4_t,
) -> (uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t) {
  let mx0 = blend_words(m0, m1, m2, m3, 11, 5, 1, 8);
  let my0 = blend_words(m0, m1, m2, m3, 15, 0, 9, 6);
  let mx1 = blend_words(m0, m1, m2, m3, 14, 2, 3, 7);
  let my1 = blend_words(m0, m1, m2, m3, 10, 12, 4, 13);
  (mx0, my0, mx1, my1)
}

// ─────────────────────────────────────────────────────────────────────────────
// 4-way parallel hashing (hash4_neon)
// ─────────────────────────────────────────────────────────────────────────────

/// Parallel G function for 4 independent chunks.
///
/// In this model:
/// - v[0..4] are the four rows of state from 4 different chunks
/// - Each vector lane i corresponds to chunk i
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn g4(v: &mut [uint32x4_t; 16], a: usize, b: usize, c: usize, d: usize, mx: uint32x4_t, my: uint32x4_t) {
  // a = a + b + mx
  v[a] = vaddq_u32(v[a], v[b]);
  v[a] = vaddq_u32(v[a], mx);
  // d = (d ^ a) >>> 16
  v[d] = veorq_u32(v[d], v[a]);
  v[d] = rotr16(v[d]);
  // c = c + d
  v[c] = vaddq_u32(v[c], v[d]);
  // b = (b ^ c) >>> 12
  v[b] = veorq_u32(v[b], v[c]);
  v[b] = rotr12(v[b]);
  // a = a + b + my
  v[a] = vaddq_u32(v[a], v[b]);
  v[a] = vaddq_u32(v[a], my);
  // d = (d ^ a) >>> 8
  v[d] = veorq_u32(v[d], v[a]);
  v[d] = rotr8(v[d]);
  // c = c + d
  v[c] = vaddq_u32(v[c], v[d]);
  // b = (b ^ c) >>> 7
  v[b] = veorq_u32(v[b], v[c]);
  v[b] = rotr7(v[b]);
}

/// One round of the parallel compression function for 4 chunks.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn round4(v: &mut [uint32x4_t; 16], m: &[uint32x4_t; 16], r: usize) {
  // `r` is always 0..7 in all callers.
  debug_assert!(r < MSG_SCHEDULE.len());
  // Avoid bounds checks when indexing `m` with schedule-driven indices.
  let s = unsafe { MSG_SCHEDULE.get_unchecked(r) };

  // Column step: G(0,4,8,12), G(1,5,9,13), G(2,6,10,14), G(3,7,11,15)
  g4(v, 0, 4, 8, 12, unsafe { *m.get_unchecked(s[0]) }, unsafe {
    *m.get_unchecked(s[1])
  });
  g4(v, 1, 5, 9, 13, unsafe { *m.get_unchecked(s[2]) }, unsafe {
    *m.get_unchecked(s[3])
  });
  g4(v, 2, 6, 10, 14, unsafe { *m.get_unchecked(s[4]) }, unsafe {
    *m.get_unchecked(s[5])
  });
  g4(v, 3, 7, 11, 15, unsafe { *m.get_unchecked(s[6]) }, unsafe {
    *m.get_unchecked(s[7])
  });

  // Diagonal step: G(0,5,10,15), G(1,6,11,12), G(2,7,8,13), G(3,4,9,14)
  g4(v, 0, 5, 10, 15, unsafe { *m.get_unchecked(s[8]) }, unsafe {
    *m.get_unchecked(s[9])
  });
  g4(v, 1, 6, 11, 12, unsafe { *m.get_unchecked(s[10]) }, unsafe {
    *m.get_unchecked(s[11])
  });
  g4(v, 2, 7, 8, 13, unsafe { *m.get_unchecked(s[12]) }, unsafe {
    *m.get_unchecked(s[13])
  });
  g4(v, 3, 4, 9, 14, unsafe { *m.get_unchecked(s[14]) }, unsafe {
    *m.get_unchecked(s[15])
  });
}

/// Hash 4 complete chunks in parallel.
///
/// Each chunk is CHUNK_LEN (1024) bytes = 16 blocks.
///
/// # Safety
///
/// Caller must ensure NEON is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn hash4_neon(
  inputs: [*const u8; 4],
  input_len: usize,
  key: &[u32; 8],
  counter: u64,
  increment_counter: bool,
  flags: u32,
  flags_start: u32,
  flags_end: u32,
  out: &mut [[u8; OUT_LEN]; 4],
) {
  debug_assert!(input_len > 0);
  debug_assert!(input_len <= CHUNK_LEN);

  let num_blocks = (input_len + BLOCK_LEN - 1) / BLOCK_LEN;

  #[inline(always)]
  unsafe fn storeu_128(src: uint32x4_t, dest: *mut u8) {
    vst1q_u8(dest, vreinterpretq_u8_u32(src));
  }

  // CV vectors, lane = chunk.
  let mut h_vecs = [
    vdupq_n_u32(key[0]),
    vdupq_n_u32(key[1]),
    vdupq_n_u32(key[2]),
    vdupq_n_u32(key[3]),
    vdupq_n_u32(key[4]),
    vdupq_n_u32(key[5]),
    vdupq_n_u32(key[6]),
    vdupq_n_u32(key[7]),
  ];

  // Counter vectors.
  let (d0, d1, d2, d3) = if increment_counter {
    (0u64, 1, 2, 3)
  } else {
    (0u64, 0, 0, 0)
  };
  let counter_low_vec = vld1q_u32(
    [
      (counter.wrapping_add(d0) as u32),
      (counter.wrapping_add(d1) as u32),
      (counter.wrapping_add(d2) as u32),
      (counter.wrapping_add(d3) as u32),
    ]
    .as_ptr(),
  );
  let counter_high_vec = vld1q_u32(
    [
      ((counter.wrapping_add(d0) >> 32) as u32),
      ((counter.wrapping_add(d1) >> 32) as u32),
      ((counter.wrapping_add(d2) >> 32) as u32),
      ((counter.wrapping_add(d3) >> 32) as u32),
    ]
    .as_ptr(),
  );

  // Process each block
  let mut block_flags = flags | flags_start;
  for block_idx in 0..num_blocks {
    let block_offset = block_idx * BLOCK_LEN;
    let is_last = block_idx == num_blocks - 1;

    // Calculate block length for last block
    let block_len = if is_last && (input_len % BLOCK_LEN != 0) {
      (input_len % BLOCK_LEN) as u32
    } else {
      BLOCK_LEN as u32
    };

    if is_last {
      block_flags |= flags_end;
    }

    // Load and transpose message blocks (pads the last block if needed).
    let msg = load_msg_vecs_transposed(inputs, block_offset, block_len as usize);

    let block_len_vec = vdupq_n_u32(block_len);
    let block_flags_vec = vdupq_n_u32(block_flags);

    let mut v = [
      h_vecs[0],
      h_vecs[1],
      h_vecs[2],
      h_vecs[3],
      h_vecs[4],
      h_vecs[5],
      h_vecs[6],
      h_vecs[7],
      vdupq_n_u32(IV[0]),
      vdupq_n_u32(IV[1]),
      vdupq_n_u32(IV[2]),
      vdupq_n_u32(IV[3]),
      counter_low_vec,
      counter_high_vec,
      block_len_vec,
      block_flags_vec,
    ];

    round4(&mut v, &msg, 0);
    round4(&mut v, &msg, 1);
    round4(&mut v, &msg, 2);
    round4(&mut v, &msg, 3);
    round4(&mut v, &msg, 4);
    round4(&mut v, &msg, 5);
    round4(&mut v, &msg, 6);

    h_vecs[0] = veorq_u32(v[0], v[8]);
    h_vecs[1] = veorq_u32(v[1], v[9]);
    h_vecs[2] = veorq_u32(v[2], v[10]);
    h_vecs[3] = veorq_u32(v[3], v[11]);
    h_vecs[4] = veorq_u32(v[4], v[12]);
    h_vecs[5] = veorq_u32(v[5], v[13]);
    h_vecs[6] = veorq_u32(v[6], v[14]);
    h_vecs[7] = veorq_u32(v[7], v[15]);

    block_flags = flags;
  }

  // Transpose the CV vectors so we can store each output contiguously.
  let mut lo = [h_vecs[0], h_vecs[1], h_vecs[2], h_vecs[3]];
  let mut hi = [h_vecs[4], h_vecs[5], h_vecs[6], h_vecs[7]];
  transpose_vecs(&mut lo);
  transpose_vecs(&mut hi);

  for lane in 0..4 {
    let dst = out[lane].as_mut_ptr();
    storeu_128(lo[lane], dst.add(0));
    storeu_128(hi[lane], dst.add(16));
  }
}

/// Hash many chunks, dispatching to hash4 as much as possible.
///
/// This is the main entry point for parallel hashing of large inputs.
///
/// # Safety
///
/// Caller must ensure NEON is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn hash_many_neon(
  inputs: &[&[u8]],
  key: &[u32; 8],
  mut counter: u64,
  increment_counter: bool,
  flags: u32,
  flags_start: u32,
  flags_end: u32,
  out: &mut [u8],
) {
  debug_assert!(!inputs.is_empty());
  debug_assert_eq!(out.len(), inputs.len() * OUT_LEN);

  let mut input_idx = 0;
  let mut out_offset = 0;

  // Process groups of 4 chunks using hash4
  while input_idx + 4 <= inputs.len() {
    let input_ptrs = [
      inputs[input_idx].as_ptr(),
      inputs[input_idx + 1].as_ptr(),
      inputs[input_idx + 2].as_ptr(),
      inputs[input_idx + 3].as_ptr(),
    ];

    // All inputs in a batch must have the same length for hash4
    let len = inputs[input_idx].len();
    debug_assert!(inputs[input_idx + 1].len() == len);
    debug_assert!(inputs[input_idx + 2].len() == len);
    debug_assert!(inputs[input_idx + 3].len() == len);

    // SAFETY: `out` is sized to `inputs.len() * OUT_LEN`, and `out_offset`
    // advances by `4 * OUT_LEN` per iteration.
    let out_buf: &mut [[u8; OUT_LEN]; 4] =
      unsafe { &mut *(out.as_mut_ptr().add(out_offset) as *mut [[u8; OUT_LEN]; 4]) };
    hash4_neon(
      input_ptrs,
      len,
      key,
      counter,
      increment_counter,
      flags,
      flags_start,
      flags_end,
      out_buf,
    );

    out_offset += 4 * OUT_LEN;

    input_idx += 4;
    if increment_counter {
      counter += 4;
    }
  }

  // Process remaining chunks one at a time using single-block compress
  while input_idx < inputs.len() {
    let input = inputs[input_idx];
    let mut cv = *key;

    let num_blocks = (input.len() + BLOCK_LEN - 1) / BLOCK_LEN;

    for block_idx in 0..num_blocks {
      let block_offset = block_idx * BLOCK_LEN;
      let is_first = block_idx == 0;
      let is_last = block_idx == num_blocks - 1;

      let block_end = core::cmp::min(block_offset + BLOCK_LEN, input.len());
      let block_len = (block_end - block_offset) as u32;

      // Prepare block (pad if necessary)
      let mut block = [0u8; BLOCK_LEN];
      block[..block_len as usize].copy_from_slice(&input[block_offset..block_end]);

      let block_words = words16_from_le_bytes_64(&block);

      let mut block_flags = flags;
      if is_first {
        block_flags |= flags_start;
      }
      if is_last {
        block_flags |= flags_end;
      }

      cv = first_8_words(super::compress(&cv, &block_words, counter, block_len, block_flags));
    }

    // Write output
    for (j, &word) in cv.iter().enumerate() {
      let bytes = word.to_le_bytes();
      out[out_offset + j * 4..out_offset + (j + 1) * 4].copy_from_slice(&bytes);
    }
    out_offset += OUT_LEN;

    input_idx += 1;
    if increment_counter {
      counter += 1;
    }
  }
}

/// Hash many contiguous full chunks, dispatching to hash4 as much as possible.
///
/// This is the hot path for large inputs in the rscrypto hasher framework.
///
/// # Safety
///
/// Caller must ensure NEON is available, and the input/output pointers are valid.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn hash_many_contiguous_neon(
  input: *const u8,
  num_chunks: usize,
  key: &[u32; 8],
  mut counter: u64,
  flags: u32,
  out: *mut u8,
) {
  debug_assert!(num_chunks != 0);

  let mut idx = 0usize;

  while idx + 4 <= num_chunks {
    let input_ptrs = [
      unsafe { input.add((idx + 0) * CHUNK_LEN) },
      unsafe { input.add((idx + 1) * CHUNK_LEN) },
      unsafe { input.add((idx + 2) * CHUNK_LEN) },
      unsafe { input.add((idx + 3) * CHUNK_LEN) },
    ];

    // SAFETY: the caller guarantees `out` is valid for `num_chunks * OUT_LEN`
    // bytes. Here we write exactly `4 * OUT_LEN` bytes starting at
    // `idx * OUT_LEN`.
    let out_buf: &mut [[u8; OUT_LEN]; 4] = unsafe { &mut *(out.add(idx * OUT_LEN) as *mut [[u8; OUT_LEN]; 4]) };
    hash4_neon(
      input_ptrs,
      CHUNK_LEN,
      key,
      counter,
      true,
      flags,
      CHUNK_START,
      super::CHUNK_END,
      out_buf,
    );

    idx += 4;
    counter = counter.wrapping_add(4);
  }

  let remaining = num_chunks - idx;
  if remaining == 0 {
    return;
  }

  // `hash4_neon` is our world-class fast path. For a remainder of 2–3 chunks,
  // it's better to run one extra (duplicate) lane than to fall back to the
  // scalar compressor for most streaming workloads (notably 4096B updates).
  if remaining >= 2 {
    let last_ptr = unsafe { input.add((idx + remaining - 1) * CHUNK_LEN) };
    let input_ptrs = [
      unsafe { input.add((idx + 0) * CHUNK_LEN) },
      unsafe { input.add((idx + 1) * CHUNK_LEN) },
      if remaining >= 3 {
        unsafe { input.add((idx + 2) * CHUNK_LEN) }
      } else {
        last_ptr
      },
      last_ptr,
    ];

    let mut out_buf = [[0u8; OUT_LEN]; 4];
    hash4_neon(
      input_ptrs,
      CHUNK_LEN,
      key,
      counter,
      true,
      flags,
      CHUNK_START,
      super::CHUNK_END,
      &mut out_buf,
    );

    for lane in 0..remaining {
      unsafe { core::ptr::copy_nonoverlapping(out_buf[lane].as_ptr(), out.add((idx + lane) * OUT_LEN), OUT_LEN) };
    }
    return;
  }

  // Single-chunk remainder: scalar is fine here (and avoids the fixed overhead
  // of setting up a 4-lane state).
  debug_assert_eq!(remaining, 1);
  let mut cv = *key;
  for block_idx in 0..(CHUNK_LEN / BLOCK_LEN) {
    let src = unsafe { input.add(idx * CHUNK_LEN + block_idx * BLOCK_LEN) };
    // SAFETY: caller guarantees `input` is valid for `num_chunks * CHUNK_LEN`.
    let block_bytes: &[u8; BLOCK_LEN] = unsafe { &*(src as *const [u8; BLOCK_LEN]) };
    let block_words = words16_from_le_bytes_64(block_bytes);

    let start = if block_idx == 0 { CHUNK_START } else { 0 };
    let end = if block_idx + 1 == (CHUNK_LEN / BLOCK_LEN) {
      super::CHUNK_END
    } else {
      0
    };
    cv = first_8_words(super::compress(
      &cv,
      &block_words,
      counter,
      BLOCK_LEN as u32,
      flags | start | end,
    ));
  }

  for (j, &word) in cv.iter().enumerate() {
    let bytes = word.to_le_bytes();
    unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), out.add(idx * OUT_LEN + j * 4), 4) };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// High-level kernel wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// NEON chunk compression: process multiple 64-byte blocks.
///
/// # Safety
///
/// Caller must ensure NEON is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub unsafe fn chunk_compress_blocks_neon(
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
    *chaining_value = first_8_words(compress_neon(
      chaining_value,
      &block_words,
      chunk_counter,
      BLOCK_LEN as u32,
      flags | start,
    ));
    *blocks_compressed = blocks_compressed.wrapping_add(1);
  }
}

/// NEON parent CV computation.
///
/// # Safety
///
/// Caller must ensure NEON is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub unsafe fn parent_cv_neon(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  let mut block_words = [0u32; 16];
  block_words[..8].copy_from_slice(&left_child_cv);
  block_words[8..].copy_from_slice(&right_child_cv);
  first_8_words(compress_neon(
    &key_words,
    &block_words,
    0,
    BLOCK_LEN as u32,
    PARENT | flags,
  ))
}
