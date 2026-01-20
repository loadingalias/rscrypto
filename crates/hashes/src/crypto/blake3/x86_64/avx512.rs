//! BLAKE3 x86_64 AVX-512 throughput kernel (16-way).

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]

use core::arch::x86_64::*;

use super::super::{BLOCK_LEN, CHUNK_LEN, IV, MSG_SCHEDULE, OUT_LEN};

pub const DEGREE: usize = 16;

#[inline(always)]
unsafe fn add(a: __m512i, b: __m512i) -> __m512i {
  unsafe { _mm512_add_epi32(a, b) }
}

#[inline(always)]
unsafe fn xor(a: __m512i, b: __m512i) -> __m512i {
  unsafe { _mm512_xor_si512(a, b) }
}

#[inline(always)]
unsafe fn set1(x: u32) -> __m512i {
  unsafe { _mm512_set1_epi32(x as i32) }
}

#[inline(always)]
unsafe fn rot16(x: __m512i) -> __m512i {
  unsafe { _mm512_or_si512(_mm512_srli_epi32(x, 16), _mm512_slli_epi32(x, 16)) }
}

#[inline(always)]
unsafe fn rot12(x: __m512i) -> __m512i {
  unsafe { _mm512_or_si512(_mm512_srli_epi32(x, 12), _mm512_slli_epi32(x, 20)) }
}

#[inline(always)]
unsafe fn rot8(x: __m512i) -> __m512i {
  unsafe { _mm512_or_si512(_mm512_srli_epi32(x, 8), _mm512_slli_epi32(x, 24)) }
}

#[inline(always)]
unsafe fn rot7(x: __m512i) -> __m512i {
  unsafe { _mm512_or_si512(_mm512_srli_epi32(x, 7), _mm512_slli_epi32(x, 25)) }
}

#[inline(always)]
unsafe fn round(v: &mut [__m512i; 16], m: &[__m512i; 16], r: usize) {
  unsafe {
    v[0] = add(v[0], m[MSG_SCHEDULE[r][0]]);
    v[1] = add(v[1], m[MSG_SCHEDULE[r][2]]);
    v[2] = add(v[2], m[MSG_SCHEDULE[r][4]]);
    v[3] = add(v[3], m[MSG_SCHEDULE[r][6]]);
    v[0] = add(v[0], v[4]);
    v[1] = add(v[1], v[5]);
    v[2] = add(v[2], v[6]);
    v[3] = add(v[3], v[7]);
    v[12] = xor(v[12], v[0]);
    v[13] = xor(v[13], v[1]);
    v[14] = xor(v[14], v[2]);
    v[15] = xor(v[15], v[3]);
    v[12] = rot16(v[12]);
    v[13] = rot16(v[13]);
    v[14] = rot16(v[14]);
    v[15] = rot16(v[15]);
    v[8] = add(v[8], v[12]);
    v[9] = add(v[9], v[13]);
    v[10] = add(v[10], v[14]);
    v[11] = add(v[11], v[15]);
    v[4] = xor(v[4], v[8]);
    v[5] = xor(v[5], v[9]);
    v[6] = xor(v[6], v[10]);
    v[7] = xor(v[7], v[11]);
    v[4] = rot12(v[4]);
    v[5] = rot12(v[5]);
    v[6] = rot12(v[6]);
    v[7] = rot12(v[7]);
    v[0] = add(v[0], m[MSG_SCHEDULE[r][1]]);
    v[1] = add(v[1], m[MSG_SCHEDULE[r][3]]);
    v[2] = add(v[2], m[MSG_SCHEDULE[r][5]]);
    v[3] = add(v[3], m[MSG_SCHEDULE[r][7]]);
    v[0] = add(v[0], v[4]);
    v[1] = add(v[1], v[5]);
    v[2] = add(v[2], v[6]);
    v[3] = add(v[3], v[7]);
    v[12] = xor(v[12], v[0]);
    v[13] = xor(v[13], v[1]);
    v[14] = xor(v[14], v[2]);
    v[15] = xor(v[15], v[3]);
    v[12] = rot8(v[12]);
    v[13] = rot8(v[13]);
    v[14] = rot8(v[14]);
    v[15] = rot8(v[15]);
    v[8] = add(v[8], v[12]);
    v[9] = add(v[9], v[13]);
    v[10] = add(v[10], v[14]);
    v[11] = add(v[11], v[15]);
    v[4] = xor(v[4], v[8]);
    v[5] = xor(v[5], v[9]);
    v[6] = xor(v[6], v[10]);
    v[7] = xor(v[7], v[11]);
    v[4] = rot7(v[4]);
    v[5] = rot7(v[5]);
    v[6] = rot7(v[6]);
    v[7] = rot7(v[7]);

    v[0] = add(v[0], m[MSG_SCHEDULE[r][8]]);
    v[1] = add(v[1], m[MSG_SCHEDULE[r][10]]);
    v[2] = add(v[2], m[MSG_SCHEDULE[r][12]]);
    v[3] = add(v[3], m[MSG_SCHEDULE[r][14]]);
    v[0] = add(v[0], v[5]);
    v[1] = add(v[1], v[6]);
    v[2] = add(v[2], v[7]);
    v[3] = add(v[3], v[4]);
    v[15] = xor(v[15], v[0]);
    v[12] = xor(v[12], v[1]);
    v[13] = xor(v[13], v[2]);
    v[14] = xor(v[14], v[3]);
    v[15] = rot16(v[15]);
    v[12] = rot16(v[12]);
    v[13] = rot16(v[13]);
    v[14] = rot16(v[14]);
    v[10] = add(v[10], v[15]);
    v[11] = add(v[11], v[12]);
    v[8] = add(v[8], v[13]);
    v[9] = add(v[9], v[14]);
    v[5] = xor(v[5], v[10]);
    v[6] = xor(v[6], v[11]);
    v[7] = xor(v[7], v[8]);
    v[4] = xor(v[4], v[9]);
    v[5] = rot12(v[5]);
    v[6] = rot12(v[6]);
    v[7] = rot12(v[7]);
    v[4] = rot12(v[4]);
    v[0] = add(v[0], m[MSG_SCHEDULE[r][9]]);
    v[1] = add(v[1], m[MSG_SCHEDULE[r][11]]);
    v[2] = add(v[2], m[MSG_SCHEDULE[r][13]]);
    v[3] = add(v[3], m[MSG_SCHEDULE[r][15]]);
    v[0] = add(v[0], v[5]);
    v[1] = add(v[1], v[6]);
    v[2] = add(v[2], v[7]);
    v[3] = add(v[3], v[4]);
    v[15] = xor(v[15], v[0]);
    v[12] = xor(v[12], v[1]);
    v[13] = xor(v[13], v[2]);
    v[14] = xor(v[14], v[3]);
    v[15] = rot8(v[15]);
    v[12] = rot8(v[12]);
    v[13] = rot8(v[13]);
    v[14] = rot8(v[14]);
    v[10] = add(v[10], v[15]);
    v[11] = add(v[11], v[12]);
    v[8] = add(v[8], v[13]);
    v[9] = add(v[9], v[14]);
    v[5] = xor(v[5], v[10]);
    v[6] = xor(v[6], v[11]);
    v[7] = xor(v[7], v[8]);
    v[4] = xor(v[4], v[9]);
    v[5] = rot7(v[5]);
    v[6] = rot7(v[6]);
    v[7] = rot7(v[7]);
    v[4] = rot7(v[4]);
  }
}

#[inline(always)]
unsafe fn counter_vec(counter: u64, increment_counter: bool) -> (__m512i, __m512i) {
  let mask = if increment_counter { !0u64 } else { 0u64 };
  unsafe {
    let lo = _mm512_setr_epi32(
      counter_low(counter) as i32,
      counter_low(counter + (mask & 1)) as i32,
      counter_low(counter + (mask & 2)) as i32,
      counter_low(counter + (mask & 3)) as i32,
      counter_low(counter + (mask & 4)) as i32,
      counter_low(counter + (mask & 5)) as i32,
      counter_low(counter + (mask & 6)) as i32,
      counter_low(counter + (mask & 7)) as i32,
      counter_low(counter + (mask & 8)) as i32,
      counter_low(counter + (mask & 9)) as i32,
      counter_low(counter + (mask & 10)) as i32,
      counter_low(counter + (mask & 11)) as i32,
      counter_low(counter + (mask & 12)) as i32,
      counter_low(counter + (mask & 13)) as i32,
      counter_low(counter + (mask & 14)) as i32,
      counter_low(counter + (mask & 15)) as i32,
    );
    let hi = _mm512_setr_epi32(
      counter_high(counter) as i32,
      counter_high(counter + (mask & 1)) as i32,
      counter_high(counter + (mask & 2)) as i32,
      counter_high(counter + (mask & 3)) as i32,
      counter_high(counter + (mask & 4)) as i32,
      counter_high(counter + (mask & 5)) as i32,
      counter_high(counter + (mask & 6)) as i32,
      counter_high(counter + (mask & 7)) as i32,
      counter_high(counter + (mask & 8)) as i32,
      counter_high(counter + (mask & 9)) as i32,
      counter_high(counter + (mask & 10)) as i32,
      counter_high(counter + (mask & 11)) as i32,
      counter_high(counter + (mask & 12)) as i32,
      counter_high(counter + (mask & 13)) as i32,
      counter_high(counter + (mask & 14)) as i32,
      counter_high(counter + (mask & 15)) as i32,
    );
    (lo, hi)
  }
}

#[inline(always)]
const fn counter_low(counter: u64) -> u32 {
  counter as u32
}

#[inline(always)]
const fn counter_high(counter: u64) -> u32 {
  (counter >> 32) as u32
}

/// Hash 16 contiguous independent inputs in parallel.
///
/// This is optimized for the contiguous chunk hashing hot path, where inputs
/// are arranged as `CHUNK_LEN`-byte blocks back-to-back.
///
/// # Safety
/// Caller must ensure AVX-512 is available, and `input`/`out` are valid for
/// `DEGREE * CHUNK_LEN` and `DEGREE * OUT_LEN` bytes respectively.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
pub unsafe fn hash16_contiguous(input: *const u8, key: &[u32; 8], counter: u64, flags: u32, out: *mut u8) {
  unsafe {
    let mut h_vecs = [
      set1(key[0]),
      set1(key[1]),
      set1(key[2]),
      set1(key[3]),
      set1(key[4]),
      set1(key[5]),
      set1(key[6]),
      set1(key[7]),
    ];

    let (counter_low_vec, counter_high_vec) = counter_vec(counter, true);
    let stride_words: i32 = (CHUNK_LEN / 4) as i32; // u32 words per chunk
    let input_words: *const i32 = input.cast();
    let out_words: *mut i32 = out.cast();

    let blocks = CHUNK_LEN / BLOCK_LEN;
    for block in 0..blocks {
      let mut block_flags = flags;
      if block == 0 {
        block_flags |= super::super::CHUNK_START;
      }
      if block + 1 == blocks {
        block_flags |= super::super::CHUNK_END;
      }

      let block_len_vec = set1(BLOCK_LEN as u32);
      let block_flags_vec = set1(block_flags);

      // Gather message words across 16 chunks.
      let block_word_base: i32 = (block * (BLOCK_LEN / 4)) as i32;
      let lane_base = _mm512_setr_epi32(
        0,
        stride_words,
        2 * stride_words,
        3 * stride_words,
        4 * stride_words,
        5 * stride_words,
        6 * stride_words,
        7 * stride_words,
        8 * stride_words,
        9 * stride_words,
        10 * stride_words,
        11 * stride_words,
        12 * stride_words,
        13 * stride_words,
        14 * stride_words,
        15 * stride_words,
      );

      let mut m = [set1(0); 16];
      for (word, m_word) in m.iter_mut().enumerate() {
        let idx = _mm512_add_epi32(lane_base, _mm512_set1_epi32(block_word_base + word as i32));
        *m_word = _mm512_i32gather_epi32(idx, input_words, 4);
      }

      let mut v = [
        h_vecs[0],
        h_vecs[1],
        h_vecs[2],
        h_vecs[3],
        h_vecs[4],
        h_vecs[5],
        h_vecs[6],
        h_vecs[7],
        set1(IV[0]),
        set1(IV[1]),
        set1(IV[2]),
        set1(IV[3]),
        counter_low_vec,
        counter_high_vec,
        block_len_vec,
        block_flags_vec,
      ];

      round(&mut v, &m, 0);
      round(&mut v, &m, 1);
      round(&mut v, &m, 2);
      round(&mut v, &m, 3);
      round(&mut v, &m, 4);
      round(&mut v, &m, 5);
      round(&mut v, &m, 6);

      h_vecs[0] = xor(v[0], v[8]);
      h_vecs[1] = xor(v[1], v[9]);
      h_vecs[2] = xor(v[2], v[10]);
      h_vecs[3] = xor(v[3], v[11]);
      h_vecs[4] = xor(v[4], v[12]);
      h_vecs[5] = xor(v[5], v[13]);
      h_vecs[6] = xor(v[6], v[14]);
      h_vecs[7] = xor(v[7], v[15]);
    }

    // Scatter output as u32 words into `[chunk][word]` order.
    // Output stride is OUT_LEN bytes = 8 u32 words per chunk.
    let out_stride_words: i32 = (OUT_LEN / 4) as i32;
    let out_lane_base = _mm512_setr_epi32(
      0,
      out_stride_words,
      2 * out_stride_words,
      3 * out_stride_words,
      4 * out_stride_words,
      5 * out_stride_words,
      6 * out_stride_words,
      7 * out_stride_words,
      8 * out_stride_words,
      9 * out_stride_words,
      10 * out_stride_words,
      11 * out_stride_words,
      12 * out_stride_words,
      13 * out_stride_words,
      14 * out_stride_words,
      15 * out_stride_words,
    );

    for (word, &word_vec) in h_vecs.iter().enumerate() {
      let idx = _mm512_add_epi32(out_lane_base, _mm512_set1_epi32(word as i32));
      _mm512_i32scatter_epi32(out_words, idx, word_vec, 4);
    }
  }
}
