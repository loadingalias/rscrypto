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

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod asm;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
use super::{BLOCK_LEN, CHUNK_LEN, CHUNK_START, IV, MSG_SCHEDULE, OUT_LEN, PARENT, words16_from_le_bytes_64};

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
unsafe fn rotr8_tbl(v: uint32x4_t, tbl: uint8x16_t) -> uint32x4_t {
  let bytes = vreinterpretq_u8_u32(v);
  let rotated = vqtbl1q_u8(bytes, tbl);
  vreinterpretq_u32_u8(rotated)
}

/// Rotate right by 8 bits (each u32 lane), shift/or variant.
///
/// Some cores have relatively high-latency `tbl`/`vtbl` paths; for the per-block
/// compressor (latency-sensitive), prefer a shift/or implementation.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn rotr8(v: uint32x4_t) -> uint32x4_t {
  // rotr(x, 8) = (x >> 8) | (x << 24)
  vsliq_n_u32(vshrq_n_u32(v, 8), v, 24)
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
  const {
    assert!(
      cfg!(target_endian = "little"),
      "aarch64 NEON implementation assumes little-endian"
    );
  }

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
      out[lane_block * 4] = vecs[0];
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

/// Helper: take the low 64-bit lane (2x u32) from each input and concatenate.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn concat_low64_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
  let a64 = vreinterpretq_u64_u32(a);
  let b64 = vreinterpretq_u64_u32(b);
  vreinterpretq_u32_u64(vcombine_u64(vget_low_u64(a64), vget_low_u64(b64)))
}

/// Message word permutation for BLAKE3.
///
/// This applies the fixed permutation:
/// `[2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]`.
///
/// Each round uses the same access pattern for message words, and the message
/// vectors are permuted between rounds. This avoids the expensive per-round
/// gather/extract machinery that dominated the previous per-block compressor.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn permute_msg(m0: uint32x4_t, m1: uint32x4_t, m2: uint32x4_t, m3: uint32x4_t) -> [uint32x4_t; 4] {
  // Rotations within each 4-word vector.
  let m0r1 = vextq_u32(m0, m0, 1);
  let m0r2 = vextq_u32(m0, m0, 2);
  let m0r3 = vextq_u32(m0, m0, 3);

  let m1r1 = vextq_u32(m1, m1, 1);
  let m1r2 = vextq_u32(m1, m1, 2);
  let m1r3 = vextq_u32(m1, m1, 3);

  let m2r1 = vextq_u32(m2, m2, 1);
  let m2r2 = vextq_u32(m2, m2, 2);
  let m2r3 = vextq_u32(m2, m2, 3);

  let m3r1 = vextq_u32(m3, m3, 1);
  let m3r2 = vextq_u32(m3, m3, 2);
  let m3r3 = vextq_u32(m3, m3, 3);

  // Build each output vector as two 64-bit lanes (2x u32).
  let p0 = concat_low64_u32(vzip1q_u32(m0r2, m1r2), vzip1q_u32(m0r3, m2r2)); // [2,6,3,10]
  let p1 = concat_low64_u32(vzip1q_u32(m1r3, m0), vzip1q_u32(m1, m3r1)); // [7,0,4,13]
  let p2 = concat_low64_u32(vzip1q_u32(m0r1, m2r3), vzip1q_u32(m3, m1r1)); // [1,11,12,5]
  let p3 = concat_low64_u32(vzip1q_u32(m2r1, m3r2), vzip1q_u32(m3r3, m2)); // [9,14,15,8]
  [p0, p1, p2, p3]
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn compress_neon_core(
  chaining_value: &[u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> (uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t, uint32x4_t) {
  const {
    assert!(
      cfg!(target_endian = "little"),
      "aarch64 NEON implementation assumes little-endian"
    );
  }

  // Load state into 4 vectors (rows of the BLAKE3 state matrix)
  let mut row0 = vld1q_u32(chaining_value.as_ptr());
  let mut row1 = vld1q_u32(chaining_value.as_ptr().add(4));
  let mut row2 = vld1q_u32(IV.as_ptr());

  // Build row3 from counter, block_len, and flags.
  let counter_lo = counter as u32;
  let counter_hi = (counter >> 32) as u32;
  let row3_arr: [u32; 4] = [counter_lo, counter_hi, block_len, flags];
  let mut row3 = vld1q_u32(row3_arr.as_ptr());

  // Load message words into 4 vectors (standard order).
  let mut m0 = vld1q_u32(block.cast());
  let mut m1 = vld1q_u32(block.add(16).cast());
  let mut m2 = vld1q_u32(block.add(32).cast());
  let mut m3 = vld1q_u32(block.add(48).cast());

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

  // Round 0 uses the message as-loaded. Between rounds, permute the message.
  // For each round, build `(mx0,my0)` from words 0..7 and `(mx1,my1)` from 8..15.
  for r in 0..7 {
    let mx0 = vuzp1q_u32(m0, m1);
    let my0 = vuzp2q_u32(m0, m1);
    let mx1 = vuzp1q_u32(m2, m3);
    let my1 = vuzp2q_u32(m2, m3);
    round!(mx0, my0, mx1, my1);

    if r != 6 {
      let [p0, p1, p2, p3] = permute_msg(m0, m1, m2, m3);
      m0 = p0;
      m1 = p1;
      m2 = p2;
      m3 = p3;
    }
  }

  let cv_lo = vld1q_u32(chaining_value.as_ptr());
  let cv_hi = vld1q_u32(chaining_value.as_ptr().add(4));
  (row0, row1, row2, row3, cv_lo, cv_hi)
}

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
  // SAFETY: `block_words` is exactly 16 u32s, i.e. 64 bytes.
  unsafe { compress_neon_bytes(chaining_value, block_words.as_ptr().cast(), counter, block_len, flags) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn compress_neon_bytes(
  chaining_value: &[u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  let (mut row0, mut row1, mut row2, mut row3, cv_lo, cv_hi) =
    compress_neon_core(chaining_value, block, counter, block_len, flags);

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
// 4-way parallel hashing (hash4_neon)
// ─────────────────────────────────────────────────────────────────────────────

/// Parallel G function for 4 independent chunks.
///
/// In this model:
/// - v[0..4] are the four rows of state from 4 different chunks
/// - Each vector lane i corresponds to chunk i
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn g4(
  v: &mut [uint32x4_t; 16],
  a: usize,
  b: usize,
  c: usize,
  d: usize,
  mx: uint32x4_t,
  my: uint32x4_t,
  rot8_tbl: uint8x16_t,
) {
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
  v[d] = rotr8_tbl(v[d], rot8_tbl);
  // c = c + d
  v[c] = vaddq_u32(v[c], v[d]);
  // b = (b ^ c) >>> 7
  v[b] = veorq_u32(v[b], v[c]);
  v[b] = rotr7(v[b]);
}

/// One round of the parallel compression function for 4 chunks.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn round4(v: &mut [uint32x4_t; 16], m: &[uint32x4_t; 16], r: usize, rot8_tbl: uint8x16_t) {
  // `r` is always 0..7 in all callers.
  debug_assert!(r < MSG_SCHEDULE.len());
  // Avoid bounds checks when indexing `m` with schedule-driven indices.
  // SAFETY: `r < MSG_SCHEDULE.len()` (asserted above).
  let s = unsafe { MSG_SCHEDULE.get_unchecked(r) };

  // Column step: G(0,4,8,12), G(1,5,9,13), G(2,6,10,14), G(3,7,11,15)
  // SAFETY: `s` contains fixed schedule indices in 0..16, and `m` is `[T; 16]`.
  unsafe {
    g4(v, 0, 4, 8, 12, *m.get_unchecked(s[0]), *m.get_unchecked(s[1]), rot8_tbl);
    g4(v, 1, 5, 9, 13, *m.get_unchecked(s[2]), *m.get_unchecked(s[3]), rot8_tbl);
    g4(
      v,
      2,
      6,
      10,
      14,
      *m.get_unchecked(s[4]),
      *m.get_unchecked(s[5]),
      rot8_tbl,
    );
    g4(
      v,
      3,
      7,
      11,
      15,
      *m.get_unchecked(s[6]),
      *m.get_unchecked(s[7]),
      rot8_tbl,
    );

    // Diagonal step: G(0,5,10,15), G(1,6,11,12), G(2,7,8,13), G(3,4,9,14)
    g4(
      v,
      0,
      5,
      10,
      15,
      *m.get_unchecked(s[8]),
      *m.get_unchecked(s[9]),
      rot8_tbl,
    );
    g4(
      v,
      1,
      6,
      11,
      12,
      *m.get_unchecked(s[10]),
      *m.get_unchecked(s[11]),
      rot8_tbl,
    );
    g4(
      v,
      2,
      7,
      8,
      13,
      *m.get_unchecked(s[12]),
      *m.get_unchecked(s[13]),
      rot8_tbl,
    );
    g4(
      v,
      3,
      4,
      9,
      14,
      *m.get_unchecked(s[14]),
      *m.get_unchecked(s[15]),
      rot8_tbl,
    );
  }
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

  let num_blocks = input_len.div_ceil(BLOCK_LEN);

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
  let rot8_tbl = vld1q_u8(ROT8_TABLE.as_ptr());
  for block_idx in 0..num_blocks {
    let block_offset = block_idx * BLOCK_LEN;
    let is_last = block_idx == num_blocks - 1;

    // Calculate block length for last block
    let block_len = if is_last && !input_len.is_multiple_of(BLOCK_LEN) {
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

    round4(&mut v, &msg, 0, rot8_tbl);
    round4(&mut v, &msg, 1, rot8_tbl);
    round4(&mut v, &msg, 2, rot8_tbl);
    round4(&mut v, &msg, 3, rot8_tbl);
    round4(&mut v, &msg, 4, rot8_tbl);
    round4(&mut v, &msg, 5, rot8_tbl);
    round4(&mut v, &msg, 6, rot8_tbl);

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

/// Hash exactly one full chunk (1024B) and return the *root* hash bytes.
///
/// This is a dedicated aarch64 fast path for `len == 1024` oneshot hashing.
///
/// # Safety
///
/// - `input` must point to exactly `CHUNK_LEN` readable bytes.
#[inline]
pub unsafe fn root_hash_one_chunk_root_aarch64(input: *const u8, key: &[u32; 8], flags: u32) -> [u8; OUT_LEN] {
  #[cfg(any(target_os = "linux", target_os = "macos"))]
  {
    let mut out = [0u8; OUT_LEN];
    #[cfg(target_os = "linux")]
    asm::rscrypto_blake3_hash1_chunk_root_aarch64_unix_linux(input, key.as_ptr(), flags, out.as_mut_ptr());
    #[cfg(target_os = "macos")]
    asm::rscrypto_blake3_hash1_chunk_root_aarch64_apple_darwin(input, key.as_ptr(), flags, out.as_mut_ptr());
    out
  }

  #[cfg(not(any(target_os = "linux", target_os = "macos")))]
  {
    // Fallback: reuse the 4-lane NEON chunk kernel (duplicates lanes).
    root_hash_one_chunk_neon(input, key, flags)
  }
}

/// Hash exactly one full chunk (1024B) and write the resulting *chunk CV* bytes.
///
/// This is used to avoid per-block compression in remainder paths.
///
/// # Safety
///
/// - `input` must point to exactly `CHUNK_LEN` readable bytes.
/// - `out` must point to at least `OUT_LEN` writable bytes.
#[inline]
pub unsafe fn chunk_cv_one_chunk_aarch64_out(input: *const u8, key: &[u32; 8], counter: u64, flags: u32, out: *mut u8) {
  #[cfg(any(target_os = "linux", target_os = "macos"))]
  {
    #[cfg(target_os = "linux")]
    asm::rscrypto_blake3_hash1_chunk_cv_aarch64_unix_linux(input, key.as_ptr(), counter, flags, out);
    #[cfg(target_os = "macos")]
    asm::rscrypto_blake3_hash1_chunk_cv_aarch64_apple_darwin(input, key.as_ptr(), counter, flags, out);
  }

  #[cfg(not(any(target_os = "linux", target_os = "macos")))]
  {
    // Fallback: per-block NEON compressor.
    let mut cv = *key;
    for block_idx in 0..(CHUNK_LEN / BLOCK_LEN) {
      let block_bytes: &[u8; BLOCK_LEN] = {
        let src = input.add(block_idx * BLOCK_LEN);
        &*(src as *const [u8; BLOCK_LEN])
      };

      let start = if block_idx == 0 { CHUNK_START } else { 0 };
      let end = if block_idx + 1 == (CHUNK_LEN / BLOCK_LEN) {
        super::CHUNK_END
      } else {
        0
      };
      cv = compress_cv_neon_bytes(
        &cv,
        block_bytes.as_ptr(),
        counter,
        BLOCK_LEN as u32,
        flags | start | end,
      );
    }

    for (j, &word) in cv.iter().enumerate() {
      let bytes = word.to_le_bytes();
      core::ptr::copy_nonoverlapping(bytes.as_ptr(), out.add(j * 4), 4);
    }
  }
}

/// Hash exactly one full chunk (1024B) and write the *pre-final-block* CV plus the last block
/// bytes.
///
/// This produces the internal state needed to build an `OutputState` for a full chunk:
/// - `out_cv` receives the chaining value after blocks 0..14.
/// - `out_last_block` receives the raw bytes of block 15 (64B).
///
/// # Safety
///
/// - `input` must point to exactly `CHUNK_LEN` readable bytes.
/// - `out_cv` must point to at least 8 writable `u32`s.
/// - `out_last_block` must point to at least `BLOCK_LEN` writable bytes.
#[inline]
pub unsafe fn chunk_state_one_chunk_aarch64_out(
  input: *const u8,
  key: &[u32; 8],
  counter: u64,
  flags: u32,
  out_cv: *mut u32,
  out_last_block: *mut u8,
) {
  #[cfg(any(target_os = "linux", target_os = "macos"))]
  {
    #[cfg(target_os = "linux")]
    asm::rscrypto_blake3_hash1_chunk_state_aarch64_unix_linux(
      input,
      key.as_ptr(),
      counter,
      flags,
      out_cv,
      out_last_block,
    );
    #[cfg(target_os = "macos")]
    asm::rscrypto_blake3_hash1_chunk_state_aarch64_apple_darwin(
      input,
      key.as_ptr(),
      counter,
      flags,
      out_cv,
      out_last_block,
    );
  }

  #[cfg(not(any(target_os = "linux", target_os = "macos")))]
  {
    // Fallback: per-block NEON compressor for blocks 0..14, then copy the final block bytes.
    let mut cv = *key;
    for block_idx in 0..15 {
      let block_bytes: &[u8; BLOCK_LEN] = {
        let src = input.add(block_idx * BLOCK_LEN);
        &*(src as *const [u8; BLOCK_LEN])
      };

      let start = if block_idx == 0 { CHUNK_START } else { 0 };
      cv = compress_cv_neon_bytes(&cv, block_bytes.as_ptr(), counter, BLOCK_LEN as u32, flags | start);
    }

    // Store cv.
    core::ptr::copy_nonoverlapping(cv.as_ptr(), out_cv, 8);

    // Copy final block bytes.
    core::ptr::copy_nonoverlapping(input.add(15 * BLOCK_LEN), out_last_block, BLOCK_LEN);
  }
}

/// Hash exactly one full chunk (1024B) and return the *root* hash bytes.
///
/// This is a specialized fast path for the aarch64 "exactly one chunk" cliff:
/// we reuse the 4-lane chunk kernel, duplicating the input pointer across lanes
/// but setting `ROOT` on the last block so the resulting CV is directly the
/// root hash.
///
/// # Safety
///
/// Caller must ensure NEON is available, and `input` is valid for `CHUNK_LEN`
/// readable bytes.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn root_hash_one_chunk_neon(input: *const u8, key: &[u32; 8], flags: u32) -> [u8; OUT_LEN] {
  let inputs = [input, input, input, input];
  let mut out = [[0u8; OUT_LEN]; 4];
  hash4_neon(
    inputs,
    CHUNK_LEN,
    key,
    0,
    false,
    flags,
    CHUNK_START,
    super::CHUNK_END | super::ROOT,
    &mut out,
  );
  out[0]
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
    // SAFETY: caller guarantees `input` is valid for `num_chunks * CHUNK_LEN`.
    let input_ptrs = unsafe {
      [
        input.add(idx * CHUNK_LEN),
        input.add((idx + 1) * CHUNK_LEN),
        input.add((idx + 2) * CHUNK_LEN),
        input.add((idx + 3) * CHUNK_LEN),
      ]
    };

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
    // SAFETY: caller guarantees `input` is valid for `num_chunks * CHUNK_LEN`.
    let input_ptrs = unsafe {
      let last_ptr = input.add((idx + remaining - 1) * CHUNK_LEN);
      [
        input.add(idx * CHUNK_LEN),
        input.add((idx + 1) * CHUNK_LEN),
        if remaining >= 3 {
          input.add((idx + 2) * CHUNK_LEN)
        } else {
          last_ptr
        },
        last_ptr,
      ]
    };

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

    for (lane, lane_out) in out_buf.iter().take(remaining).enumerate() {
      // SAFETY: `lane < remaining` and `remaining <= num_chunks - idx`, so
      // `out.add((idx + lane) * OUT_LEN)` stays within `num_chunks * OUT_LEN`.
      unsafe { core::ptr::copy_nonoverlapping(lane_out.as_ptr(), out.add((idx + lane) * OUT_LEN), OUT_LEN) };
    }
    return;
  }

  // Single-chunk remainder: avoid falling back to the portable compressor.
  // This shows up in both large-input tails and "exactly one chunk" cases.
  debug_assert_eq!(remaining, 1);

  // SAFETY: caller guarantees `input` is valid for `num_chunks * CHUNK_LEN`, and
  // `out` is valid for `num_chunks * OUT_LEN`.
  unsafe {
    let src = input.add(idx * CHUNK_LEN);
    let dst = out.add(idx * OUT_LEN);
    chunk_cv_one_chunk_aarch64_out(src, key, counter, flags, dst);
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

  // aarch64 per-block hot path:
  // Use the scalar assembly backend for tight block-compress loops. This avoids
  // the current Rust/NEON per-block cliffs (message permutation overhead), and
  // is especially important for server-class cores (Neoverse/Graviton) and
  // streaming workloads with many full blocks.
  #[cfg(target_os = "linux")]
  {
    if !blocks.is_empty() {
      let num_blocks = blocks.len() / BLOCK_LEN;
      debug_assert_ne!(num_blocks, 0);
      // SAFETY: `blocks` is `num_blocks * 64` bytes, `chaining_value` is 8 u32
      // words, `blocks_compressed` is a valid pointer to a `u8`, and this is
      // only compiled on Linux aarch64 where the symbol is available.
      unsafe {
        asm::rscrypto_blake3_chunk_compress_blocks_aarch64_unix_linux(
          blocks.as_ptr(),
          chaining_value.as_mut_ptr(),
          chunk_counter,
          flags,
          blocks_compressed as *mut u8,
          num_blocks,
        );
      }
      return;
    }
  }

  let (block_slices, remainder) = blocks.as_chunks::<BLOCK_LEN>();
  debug_assert!(remainder.is_empty());
  for block_bytes in block_slices {
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    *chaining_value = compress_cv_neon_bytes(
      chaining_value,
      block_bytes.as_ptr(),
      chunk_counter,
      BLOCK_LEN as u32,
      flags | start,
    );
    *blocks_compressed = blocks_compressed.wrapping_add(1);
  }
}

/// NEON CV-only compression.
///
/// Returns the 8-word chaining value result (row0^row2, row1^row3), avoiding
/// materializing the full 16-word compression output.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn compress_cv_neon_bytes(
  chaining_value: &[u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  let (mut row0, mut row1, row2, row3, _cv_lo, _cv_hi) =
    compress_neon_core(chaining_value, block, counter, block_len, flags);
  row0 = veorq_u32(row0, row2);
  row1 = veorq_u32(row1, row3);

  let mut out = [0u32; 8];
  vst1q_u32(out.as_mut_ptr(), row0);
  vst1q_u32(out.as_mut_ptr().add(4), row1);
  out
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
  compress_cv_neon_bytes(
    &key_words,
    block_words.as_ptr().cast(),
    0,
    BLOCK_LEN as u32,
    PARENT | flags,
  )
}

/// Generate 4 root output blocks (64 bytes each) in parallel.
///
/// Each lane uses an independent `output_block_counter` (`counter + lane`), but
/// shares the same `chaining_value`, `block_words`, `block_len`, and `flags`.
///
/// # Safety
/// Caller must ensure NEON is available and that `out` is valid for `4 * 64`
/// writable bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn root_output_blocks4_neon(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
  out: *mut u8,
) {
  #[inline(always)]
  unsafe fn storeu_128(src: uint32x4_t, dest: *mut u8) {
    vst1q_u8(dest, vreinterpretq_u8_u32(src))
  }

  let cv_vecs = [
    vdupq_n_u32(chaining_value[0]),
    vdupq_n_u32(chaining_value[1]),
    vdupq_n_u32(chaining_value[2]),
    vdupq_n_u32(chaining_value[3]),
    vdupq_n_u32(chaining_value[4]),
    vdupq_n_u32(chaining_value[5]),
    vdupq_n_u32(chaining_value[6]),
    vdupq_n_u32(chaining_value[7]),
  ];

  let m = [
    vdupq_n_u32(block_words[0]),
    vdupq_n_u32(block_words[1]),
    vdupq_n_u32(block_words[2]),
    vdupq_n_u32(block_words[3]),
    vdupq_n_u32(block_words[4]),
    vdupq_n_u32(block_words[5]),
    vdupq_n_u32(block_words[6]),
    vdupq_n_u32(block_words[7]),
    vdupq_n_u32(block_words[8]),
    vdupq_n_u32(block_words[9]),
    vdupq_n_u32(block_words[10]),
    vdupq_n_u32(block_words[11]),
    vdupq_n_u32(block_words[12]),
    vdupq_n_u32(block_words[13]),
    vdupq_n_u32(block_words[14]),
    vdupq_n_u32(block_words[15]),
  ];

  let counter_low_vec = vld1q_u32(
    [
      counter as u32,
      counter.wrapping_add(1) as u32,
      counter.wrapping_add(2) as u32,
      counter.wrapping_add(3) as u32,
    ]
    .as_ptr(),
  );
  let counter_high_vec = vld1q_u32(
    [
      (counter >> 32) as u32,
      (counter.wrapping_add(1) >> 32) as u32,
      (counter.wrapping_add(2) >> 32) as u32,
      (counter.wrapping_add(3) >> 32) as u32,
    ]
    .as_ptr(),
  );

  let block_len_vec = vdupq_n_u32(block_len);
  let flags_vec = vdupq_n_u32(flags);

  let iv0 = vdupq_n_u32(IV[0]);
  let iv1 = vdupq_n_u32(IV[1]);
  let iv2 = vdupq_n_u32(IV[2]);
  let iv3 = vdupq_n_u32(IV[3]);

  let rot8_tbl = vld1q_u8(ROT8_TABLE.as_ptr());
  let mut v = [
    cv_vecs[0],
    cv_vecs[1],
    cv_vecs[2],
    cv_vecs[3],
    cv_vecs[4],
    cv_vecs[5],
    cv_vecs[6],
    cv_vecs[7],
    iv0,
    iv1,
    iv2,
    iv3,
    counter_low_vec,
    counter_high_vec,
    block_len_vec,
    flags_vec,
  ];

  round4(&mut v, &m, 0, rot8_tbl);
  round4(&mut v, &m, 1, rot8_tbl);
  round4(&mut v, &m, 2, rot8_tbl);
  round4(&mut v, &m, 3, rot8_tbl);
  round4(&mut v, &m, 4, rot8_tbl);
  round4(&mut v, &m, 5, rot8_tbl);
  round4(&mut v, &m, 6, rot8_tbl);

  let out_words = [
    veorq_u32(v[0], v[8]),
    veorq_u32(v[1], v[9]),
    veorq_u32(v[2], v[10]),
    veorq_u32(v[3], v[11]),
    veorq_u32(v[4], v[12]),
    veorq_u32(v[5], v[13]),
    veorq_u32(v[6], v[14]),
    veorq_u32(v[7], v[15]),
    veorq_u32(v[8], cv_vecs[0]),
    veorq_u32(v[9], cv_vecs[1]),
    veorq_u32(v[10], cv_vecs[2]),
    veorq_u32(v[11], cv_vecs[3]),
    veorq_u32(v[12], cv_vecs[4]),
    veorq_u32(v[13], cv_vecs[5]),
    veorq_u32(v[14], cv_vecs[6]),
    veorq_u32(v[15], cv_vecs[7]),
  ];

  let mut g0 = [out_words[0], out_words[1], out_words[2], out_words[3]];
  let mut g1 = [out_words[4], out_words[5], out_words[6], out_words[7]];
  let mut g2 = [out_words[8], out_words[9], out_words[10], out_words[11]];
  let mut g3 = [out_words[12], out_words[13], out_words[14], out_words[15]];
  transpose_vecs(&mut g0);
  transpose_vecs(&mut g1);
  transpose_vecs(&mut g2);
  transpose_vecs(&mut g3);

  for lane in 0..4 {
    let base = out.add(lane * 64);
    storeu_128(g0[lane], base);
    storeu_128(g1[lane], base.add(16));
    storeu_128(g2[lane], base.add(32));
    storeu_128(g3[lane], base.add(48));
  }
}
