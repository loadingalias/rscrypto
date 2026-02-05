//! BLAKE3 x86_64 SSE4.1 throughput kernel (4-way).

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]

use core::arch::x86_64::*;

use super::{
  super::{BLOCK_LEN, IV, MSG_SCHEDULE},
  counter_high, counter_low,
};

pub const DEGREE: usize = 4;

#[inline(always)]
unsafe fn loadu(src: *const u8) -> __m128i {
  unsafe { _mm_loadu_si128(src.cast()) }
}

#[inline(always)]
unsafe fn storeu(src: __m128i, dest: *mut u8) {
  unsafe { _mm_storeu_si128(dest.cast(), src) }
}

#[inline(always)]
unsafe fn add(a: __m128i, b: __m128i) -> __m128i {
  unsafe { _mm_add_epi32(a, b) }
}

#[inline(always)]
unsafe fn xor(a: __m128i, b: __m128i) -> __m128i {
  unsafe { _mm_xor_si128(a, b) }
}

#[inline(always)]
unsafe fn set1(x: u32) -> __m128i {
  unsafe { _mm_set1_epi32(x as i32) }
}

#[inline(always)]
unsafe fn set4(a: u32, b: u32, c: u32, d: u32) -> __m128i {
  unsafe { _mm_setr_epi32(a as i32, b as i32, c as i32, d as i32) }
}

#[inline(always)]
unsafe fn rot12(a: __m128i) -> __m128i {
  unsafe { _mm_or_si128(_mm_srli_epi32(a, 12), _mm_slli_epi32(a, 20)) }
}

#[inline(always)]
unsafe fn rot7(a: __m128i) -> __m128i {
  unsafe { _mm_or_si128(_mm_srli_epi32(a, 7), _mm_slli_epi32(a, 25)) }
}

#[inline(always)]
unsafe fn round(v: &mut [__m128i; 16], m: &[__m128i; 16], r: usize, rot16_mask: __m128i, rot8_mask: __m128i) {
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
    v[12] = _mm_shuffle_epi8(v[12], rot16_mask);
    v[13] = _mm_shuffle_epi8(v[13], rot16_mask);
    v[14] = _mm_shuffle_epi8(v[14], rot16_mask);
    v[15] = _mm_shuffle_epi8(v[15], rot16_mask);
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
    v[12] = _mm_shuffle_epi8(v[12], rot8_mask);
    v[13] = _mm_shuffle_epi8(v[13], rot8_mask);
    v[14] = _mm_shuffle_epi8(v[14], rot8_mask);
    v[15] = _mm_shuffle_epi8(v[15], rot8_mask);
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
    v[15] = _mm_shuffle_epi8(v[15], rot16_mask);
    v[12] = _mm_shuffle_epi8(v[12], rot16_mask);
    v[13] = _mm_shuffle_epi8(v[13], rot16_mask);
    v[14] = _mm_shuffle_epi8(v[14], rot16_mask);
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
    v[15] = _mm_shuffle_epi8(v[15], rot8_mask);
    v[12] = _mm_shuffle_epi8(v[12], rot8_mask);
    v[13] = _mm_shuffle_epi8(v[13], rot8_mask);
    v[14] = _mm_shuffle_epi8(v[14], rot8_mask);
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
unsafe fn transpose_vecs(vecs: &mut [__m128i; DEGREE]) {
  unsafe {
    let ab_01 = _mm_unpacklo_epi32(vecs[0], vecs[1]);
    let ab_23 = _mm_unpackhi_epi32(vecs[0], vecs[1]);
    let cd_01 = _mm_unpacklo_epi32(vecs[2], vecs[3]);
    let cd_23 = _mm_unpackhi_epi32(vecs[2], vecs[3]);

    let abcd_0 = _mm_unpacklo_epi64(ab_01, cd_01);
    let abcd_1 = _mm_unpackhi_epi64(ab_01, cd_01);
    let abcd_2 = _mm_unpacklo_epi64(ab_23, cd_23);
    let abcd_3 = _mm_unpackhi_epi64(ab_23, cd_23);

    vecs[0] = abcd_0;
    vecs[1] = abcd_1;
    vecs[2] = abcd_2;
    vecs[3] = abcd_3;
  }
}

#[inline(always)]
unsafe fn transpose_msg_vecs(inputs: &[*const u8; DEGREE], block_offset: usize) -> [__m128i; 16] {
  unsafe {
    let stride = 4 * DEGREE;
    let mut quarter0 = [
      loadu(inputs[0].add(block_offset)),
      loadu(inputs[1].add(block_offset)),
      loadu(inputs[2].add(block_offset)),
      loadu(inputs[3].add(block_offset)),
    ];
    let mut quarter1 = [
      loadu(inputs[0].add(block_offset + stride)),
      loadu(inputs[1].add(block_offset + stride)),
      loadu(inputs[2].add(block_offset + stride)),
      loadu(inputs[3].add(block_offset + stride)),
    ];
    let mut quarter2 = [
      loadu(inputs[0].add(block_offset + 2 * stride)),
      loadu(inputs[1].add(block_offset + 2 * stride)),
      loadu(inputs[2].add(block_offset + 2 * stride)),
      loadu(inputs[3].add(block_offset + 2 * stride)),
    ];
    let mut quarter3 = [
      loadu(inputs[0].add(block_offset + 3 * stride)),
      loadu(inputs[1].add(block_offset + 3 * stride)),
      loadu(inputs[2].add(block_offset + 3 * stride)),
      loadu(inputs[3].add(block_offset + 3 * stride)),
    ];

    for &input in inputs.iter() {
      _mm_prefetch(input.wrapping_add(block_offset + 256).cast::<i8>(), _MM_HINT_T0);
    }

    transpose_vecs(&mut quarter0);
    transpose_vecs(&mut quarter1);
    transpose_vecs(&mut quarter2);
    transpose_vecs(&mut quarter3);

    [
      quarter0[0],
      quarter0[1],
      quarter0[2],
      quarter0[3],
      quarter1[0],
      quarter1[1],
      quarter1[2],
      quarter1[3],
      quarter2[0],
      quarter2[1],
      quarter2[2],
      quarter2[3],
      quarter3[0],
      quarter3[1],
      quarter3[2],
      quarter3[3],
    ]
  }
}

#[inline(always)]
unsafe fn load_counters(counter: u64, increment_counter: bool) -> (__m128i, __m128i) {
  let mask = if increment_counter { !0u64 } else { 0u64 };
  unsafe {
    (
      set4(
        counter_low(counter),
        counter_low(counter + (mask & 1)),
        counter_low(counter + (mask & 2)),
        counter_low(counter + (mask & 3)),
      ),
      set4(
        counter_high(counter),
        counter_high(counter + (mask & 1)),
        counter_high(counter + (mask & 2)),
        counter_high(counter + (mask & 3)),
      ),
    )
  }
}

/// Hash `DEGREE` independent inputs in parallel.
///
/// # Safety
/// Caller must ensure SSE4.1 is available and that all input pointers are valid
/// for `blocks * BLOCK_LEN` bytes.
#[target_feature(enable = "sse4.1,ssse3")]
pub unsafe fn hash4(
  inputs: &[*const u8; DEGREE],
  blocks: usize,
  key: &[u32; 8],
  counter: u64,
  increment_counter: bool,
  flags: u32,
  flags_start: u32,
  flags_end: u32,
  out: *mut u8,
) {
  unsafe {
    let rot16_mask = _mm_setr_epi8(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
    let rot8_mask = _mm_setr_epi8(1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);

    let block_len_vec = set1(BLOCK_LEN as u32);
    let iv0 = set1(IV[0]);
    let iv1 = set1(IV[1]);
    let iv2 = set1(IV[2]);
    let iv3 = set1(IV[3]);

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

    let (counter_low_vec, counter_high_vec) = load_counters(counter, increment_counter);

    for block in 0..blocks {
      let mut block_flags = flags;
      if block == 0 {
        block_flags |= flags_start;
      }
      if block + 1 == blocks {
        block_flags |= flags_end;
      }

      let block_flags_vec = set1(block_flags);
      let msg_vecs = transpose_msg_vecs(inputs, block * BLOCK_LEN);

      let mut v = [
        h_vecs[0],
        h_vecs[1],
        h_vecs[2],
        h_vecs[3],
        h_vecs[4],
        h_vecs[5],
        h_vecs[6],
        h_vecs[7],
        iv0,
        iv1,
        iv2,
        iv3,
        counter_low_vec,
        counter_high_vec,
        block_len_vec,
        block_flags_vec,
      ];

      round(&mut v, &msg_vecs, 0, rot16_mask, rot8_mask);
      round(&mut v, &msg_vecs, 1, rot16_mask, rot8_mask);
      round(&mut v, &msg_vecs, 2, rot16_mask, rot8_mask);
      round(&mut v, &msg_vecs, 3, rot16_mask, rot8_mask);
      round(&mut v, &msg_vecs, 4, rot16_mask, rot8_mask);
      round(&mut v, &msg_vecs, 5, rot16_mask, rot8_mask);
      round(&mut v, &msg_vecs, 6, rot16_mask, rot8_mask);

      h_vecs[0] = xor(v[0], v[8]);
      h_vecs[1] = xor(v[1], v[9]);
      h_vecs[2] = xor(v[2], v[10]);
      h_vecs[3] = xor(v[3], v[11]);
      h_vecs[4] = xor(v[4], v[12]);
      h_vecs[5] = xor(v[5], v[13]);
      h_vecs[6] = xor(v[6], v[14]);
      h_vecs[7] = xor(v[7], v[15]);
    }

    let mut lo = [h_vecs[0], h_vecs[1], h_vecs[2], h_vecs[3]];
    let mut hi = [h_vecs[4], h_vecs[5], h_vecs[6], h_vecs[7]];
    transpose_vecs(&mut lo);
    transpose_vecs(&mut hi);

    // After transpose, `lo[i]` and `hi[i]` each contain one u32 word from all
    // lanes. Store in the layout expected by the rest of the implementation:
    // `OUT_LEN` bytes per chunk.
    let stride = 4 * DEGREE;
    storeu(lo[0], out);
    storeu(hi[0], out.add(stride));
    storeu(lo[1], out.add(2 * stride));
    storeu(hi[1], out.add(3 * stride));
    storeu(lo[2], out.add(4 * stride));
    storeu(hi[2], out.add(5 * stride));
    storeu(lo[3], out.add(6 * stride));
    storeu(hi[3], out.add(7 * stride));
  }
}

/// Generate 4 root output blocks (64 bytes each) in parallel.
///
/// Each lane uses an independent `output_block_counter` (`counter + lane`), but
/// shares the same `chaining_value`, `block_words`, `block_len`, and `flags`.
///
/// # Safety
/// Caller must ensure SSE4.1+SSSE3 is available and that `out` is valid for
/// `4 * 64` writable bytes.
#[target_feature(enable = "sse4.1,ssse3")]
pub unsafe fn root_output_blocks4(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
  out: *mut u8,
) {
  unsafe {
    let rot16_mask = _mm_setr_epi8(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
    let rot8_mask = _mm_setr_epi8(1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);

    let cv_vecs = [
      set1(chaining_value[0]),
      set1(chaining_value[1]),
      set1(chaining_value[2]),
      set1(chaining_value[3]),
      set1(chaining_value[4]),
      set1(chaining_value[5]),
      set1(chaining_value[6]),
      set1(chaining_value[7]),
    ];

    let msg_vecs = [
      set1(block_words[0]),
      set1(block_words[1]),
      set1(block_words[2]),
      set1(block_words[3]),
      set1(block_words[4]),
      set1(block_words[5]),
      set1(block_words[6]),
      set1(block_words[7]),
      set1(block_words[8]),
      set1(block_words[9]),
      set1(block_words[10]),
      set1(block_words[11]),
      set1(block_words[12]),
      set1(block_words[13]),
      set1(block_words[14]),
      set1(block_words[15]),
    ];

    let (counter_low_vec, counter_high_vec) = load_counters(counter, true);
    let block_len_vec = set1(block_len);
    let flags_vec = set1(flags);

    let iv0 = set1(IV[0]);
    let iv1 = set1(IV[1]);
    let iv2 = set1(IV[2]);
    let iv3 = set1(IV[3]);

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

    round(&mut v, &msg_vecs, 0, rot16_mask, rot8_mask);
    round(&mut v, &msg_vecs, 1, rot16_mask, rot8_mask);
    round(&mut v, &msg_vecs, 2, rot16_mask, rot8_mask);
    round(&mut v, &msg_vecs, 3, rot16_mask, rot8_mask);
    round(&mut v, &msg_vecs, 4, rot16_mask, rot8_mask);
    round(&mut v, &msg_vecs, 5, rot16_mask, rot8_mask);
    round(&mut v, &msg_vecs, 6, rot16_mask, rot8_mask);

    let out_words = [
      xor(v[0], v[8]),
      xor(v[1], v[9]),
      xor(v[2], v[10]),
      xor(v[3], v[11]),
      xor(v[4], v[12]),
      xor(v[5], v[13]),
      xor(v[6], v[14]),
      xor(v[7], v[15]),
      xor(v[8], cv_vecs[0]),
      xor(v[9], cv_vecs[1]),
      xor(v[10], cv_vecs[2]),
      xor(v[11], cv_vecs[3]),
      xor(v[12], cv_vecs[4]),
      xor(v[13], cv_vecs[5]),
      xor(v[14], cv_vecs[6]),
      xor(v[15], cv_vecs[7]),
    ];

    let mut g0 = [out_words[0], out_words[1], out_words[2], out_words[3]];
    let mut g1 = [out_words[4], out_words[5], out_words[6], out_words[7]];
    let mut g2 = [out_words[8], out_words[9], out_words[10], out_words[11]];
    let mut g3 = [out_words[12], out_words[13], out_words[14], out_words[15]];
    transpose_vecs(&mut g0);
    transpose_vecs(&mut g1);
    transpose_vecs(&mut g2);
    transpose_vecs(&mut g3);

    for lane in 0..DEGREE {
      let base = out.add(lane * 64);
      storeu(g0[lane], base);
      storeu(g1[lane], base.add(16));
      storeu(g2[lane], base.add(32));
      storeu(g3[lane], base.add(48));
    }
  }
}
