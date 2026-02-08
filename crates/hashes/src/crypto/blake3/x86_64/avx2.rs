//! BLAKE3 x86_64 AVX2 throughput kernel (8-way).

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]
// On Linux we currently prefer the upstream asm implementation; keep the
// intrinsic fallback compiled but don't let `-D warnings` turn it into a build
// failure.
#![cfg_attr(
  any(target_os = "linux", target_os = "macos", target_os = "windows"),
  allow(dead_code, unused_imports)
)]

use core::arch::x86_64::*;

use super::{
  super::{BLOCK_LEN, IV, MSG_SCHEDULE},
  counter_high, counter_low,
};

pub const DEGREE: usize = 8;

#[inline(always)]
unsafe fn loadu(src: *const u8) -> __m256i {
  unsafe { _mm256_loadu_si256(src.cast()) }
}

#[inline(always)]
unsafe fn storeu(src: __m256i, dest: *mut u8) {
  unsafe { _mm256_storeu_si256(dest.cast(), src) }
}

#[inline(always)]
unsafe fn add(a: __m256i, b: __m256i) -> __m256i {
  unsafe { _mm256_add_epi32(a, b) }
}

#[inline(always)]
unsafe fn xor(a: __m256i, b: __m256i) -> __m256i {
  unsafe { _mm256_xor_si256(a, b) }
}

#[inline(always)]
unsafe fn set1(x: u32) -> __m256i {
  unsafe { _mm256_set1_epi32(x as i32) }
}

#[inline(always)]
unsafe fn set8(a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32) -> __m256i {
  unsafe {
    _mm256_setr_epi32(
      a as i32, b as i32, c as i32, d as i32, e as i32, f as i32, g as i32, h as i32,
    )
  }
}

#[inline(always)]
unsafe fn rot12(x: __m256i) -> __m256i {
  unsafe { _mm256_or_si256(_mm256_srli_epi32(x, 12), _mm256_slli_epi32(x, 20)) }
}

#[inline(always)]
unsafe fn rot7(x: __m256i) -> __m256i {
  unsafe { _mm256_or_si256(_mm256_srli_epi32(x, 7), _mm256_slli_epi32(x, 25)) }
}

#[inline(always)]
unsafe fn round(v: &mut [__m256i; 16], m: &[__m256i; 16], r: usize, rot16_mask: __m256i, rot8_mask: __m256i) {
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
    v[12] = _mm256_shuffle_epi8(v[12], rot16_mask);
    v[13] = _mm256_shuffle_epi8(v[13], rot16_mask);
    v[14] = _mm256_shuffle_epi8(v[14], rot16_mask);
    v[15] = _mm256_shuffle_epi8(v[15], rot16_mask);
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
    v[12] = _mm256_shuffle_epi8(v[12], rot8_mask);
    v[13] = _mm256_shuffle_epi8(v[13], rot8_mask);
    v[14] = _mm256_shuffle_epi8(v[14], rot8_mask);
    v[15] = _mm256_shuffle_epi8(v[15], rot8_mask);
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
    v[15] = _mm256_shuffle_epi8(v[15], rot16_mask);
    v[12] = _mm256_shuffle_epi8(v[12], rot16_mask);
    v[13] = _mm256_shuffle_epi8(v[13], rot16_mask);
    v[14] = _mm256_shuffle_epi8(v[14], rot16_mask);
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
    v[15] = _mm256_shuffle_epi8(v[15], rot8_mask);
    v[12] = _mm256_shuffle_epi8(v[12], rot8_mask);
    v[13] = _mm256_shuffle_epi8(v[13], rot8_mask);
    v[14] = _mm256_shuffle_epi8(v[14], rot8_mask);
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
unsafe fn interleave128(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
  unsafe {
    (
      _mm256_permute2x128_si256(a, b, 0x20),
      _mm256_permute2x128_si256(a, b, 0x31),
    )
  }
}

#[inline(always)]
pub(super) unsafe fn transpose8x8(vecs: &mut [__m256i; 8]) {
  unsafe {
    let ab_0145 = _mm256_unpacklo_epi32(vecs[0], vecs[1]);
    let ab_2367 = _mm256_unpackhi_epi32(vecs[0], vecs[1]);
    let cd_0145 = _mm256_unpacklo_epi32(vecs[2], vecs[3]);
    let cd_2367 = _mm256_unpackhi_epi32(vecs[2], vecs[3]);
    let ef_0145 = _mm256_unpacklo_epi32(vecs[4], vecs[5]);
    let ef_2367 = _mm256_unpackhi_epi32(vecs[4], vecs[5]);
    let gh_0145 = _mm256_unpacklo_epi32(vecs[6], vecs[7]);
    let gh_2367 = _mm256_unpackhi_epi32(vecs[6], vecs[7]);

    let abcd_04 = _mm256_unpacklo_epi64(ab_0145, cd_0145);
    let abcd_15 = _mm256_unpackhi_epi64(ab_0145, cd_0145);
    let abcd_26 = _mm256_unpacklo_epi64(ab_2367, cd_2367);
    let abcd_37 = _mm256_unpackhi_epi64(ab_2367, cd_2367);
    let efgh_04 = _mm256_unpacklo_epi64(ef_0145, gh_0145);
    let efgh_15 = _mm256_unpackhi_epi64(ef_0145, gh_0145);
    let efgh_26 = _mm256_unpacklo_epi64(ef_2367, gh_2367);
    let efgh_37 = _mm256_unpackhi_epi64(ef_2367, gh_2367);

    let (abcdefgh_0, abcdefgh_4) = interleave128(abcd_04, efgh_04);
    let (abcdefgh_1, abcdefgh_5) = interleave128(abcd_15, efgh_15);
    let (abcdefgh_2, abcdefgh_6) = interleave128(abcd_26, efgh_26);
    let (abcdefgh_3, abcdefgh_7) = interleave128(abcd_37, efgh_37);

    vecs[0] = abcdefgh_0;
    vecs[1] = abcdefgh_1;
    vecs[2] = abcdefgh_2;
    vecs[3] = abcdefgh_3;
    vecs[4] = abcdefgh_4;
    vecs[5] = abcdefgh_5;
    vecs[6] = abcdefgh_6;
    vecs[7] = abcdefgh_7;
  }
}

#[inline(always)]
unsafe fn transpose_msg_vecs(inputs: &[*const u8; DEGREE], block_offset: usize) -> [__m256i; 16] {
  unsafe {
    let stride = 4 * DEGREE;
    let mut half0 = [
      loadu(inputs[0].add(block_offset)),
      loadu(inputs[1].add(block_offset)),
      loadu(inputs[2].add(block_offset)),
      loadu(inputs[3].add(block_offset)),
      loadu(inputs[4].add(block_offset)),
      loadu(inputs[5].add(block_offset)),
      loadu(inputs[6].add(block_offset)),
      loadu(inputs[7].add(block_offset)),
    ];
    let mut half1 = [
      loadu(inputs[0].add(block_offset + stride)),
      loadu(inputs[1].add(block_offset + stride)),
      loadu(inputs[2].add(block_offset + stride)),
      loadu(inputs[3].add(block_offset + stride)),
      loadu(inputs[4].add(block_offset + stride)),
      loadu(inputs[5].add(block_offset + stride)),
      loadu(inputs[6].add(block_offset + stride)),
      loadu(inputs[7].add(block_offset + stride)),
    ];

    for &input in inputs.iter() {
      _mm_prefetch(input.wrapping_add(block_offset + 256).cast::<i8>(), _MM_HINT_T0);
    }

    transpose8x8(&mut half0);
    transpose8x8(&mut half1);

    [
      half0[0], half0[1], half0[2], half0[3], half0[4], half0[5], half0[6], half0[7], half1[0], half1[1], half1[2],
      half1[3], half1[4], half1[5], half1[6], half1[7],
    ]
  }
}

#[inline(always)]
unsafe fn load_counters(counter: u64, increment_counter: bool) -> (__m256i, __m256i) {
  let mask = if increment_counter { !0u64 } else { 0u64 };
  unsafe {
    (
      set8(
        counter_low(counter),
        counter_low(counter.wrapping_add(mask & 1)),
        counter_low(counter.wrapping_add(mask & 2)),
        counter_low(counter.wrapping_add(mask & 3)),
        counter_low(counter.wrapping_add(mask & 4)),
        counter_low(counter.wrapping_add(mask & 5)),
        counter_low(counter.wrapping_add(mask & 6)),
        counter_low(counter.wrapping_add(mask & 7)),
      ),
      set8(
        counter_high(counter),
        counter_high(counter.wrapping_add(mask & 1)),
        counter_high(counter.wrapping_add(mask & 2)),
        counter_high(counter.wrapping_add(mask & 3)),
        counter_high(counter.wrapping_add(mask & 4)),
        counter_high(counter.wrapping_add(mask & 5)),
        counter_high(counter.wrapping_add(mask & 6)),
        counter_high(counter.wrapping_add(mask & 7)),
      ),
    )
  }
}

/// Hash `DEGREE` independent inputs in parallel.
///
/// # Safety
/// Caller must ensure AVX2 is available and that all input pointers are valid
/// for `blocks * BLOCK_LEN` bytes.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
#[target_feature(enable = "avx2")]
pub unsafe fn hash8(
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
  debug_assert!(flags <= u8::MAX as u32);
  debug_assert!(flags_start <= u8::MAX as u32);
  debug_assert!(flags_end <= u8::MAX as u32);
  unsafe {
    super::asm::rscrypto_blake3_hash_many_avx2(
      inputs.as_ptr(),
      DEGREE,
      blocks,
      key.as_ptr(),
      counter,
      increment_counter,
      flags as u8,
      flags_start as u8,
      flags_end as u8,
      out,
    );
  }
}

/// Hash `DEGREE` independent inputs in parallel.
///
/// # Safety
/// Caller must ensure AVX2 is available and that all input pointers are valid
/// for `blocks * BLOCK_LEN` bytes.
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
#[target_feature(enable = "avx2")]
pub unsafe fn hash8(
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
    let rot16_mask = _mm256_setr_epi8(
      2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
    );
    let rot8_mask = _mm256_setr_epi8(
      1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12,
    );

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

    // Unlike SSE4.1, this transpose yields output vecs already ordered by word.
    transpose8x8(&mut h_vecs);

    let stride = 4 * DEGREE;
    storeu(h_vecs[0], out);
    storeu(h_vecs[1], out.add(stride));
    storeu(h_vecs[2], out.add(2 * stride));
    storeu(h_vecs[3], out.add(3 * stride));
    storeu(h_vecs[4], out.add(4 * stride));
    storeu(h_vecs[5], out.add(5 * stride));
    storeu(h_vecs[6], out.add(6 * stride));
    storeu(h_vecs[7], out.add(7 * stride));
  }
}

/// Generate 8 root output blocks (64 bytes each) in parallel.
///
/// Each lane uses an independent `output_block_counter` (`counter + lane`), but
/// shares the same `chaining_value`, `block_words`, `block_len`, and `flags`.
///
/// # Safety
/// Caller must ensure AVX2 is available and that `out` is valid for `8 * 64`
/// writable bytes.
#[target_feature(enable = "avx2")]
pub unsafe fn root_output_blocks8(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
  out: *mut u8,
) {
  unsafe {
    let rot16_mask = _mm256_setr_epi8(
      2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
    );
    let rot8_mask = _mm256_setr_epi8(
      1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12,
    );

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

    let mut out_lo = [
      xor(v[0], v[8]),
      xor(v[1], v[9]),
      xor(v[2], v[10]),
      xor(v[3], v[11]),
      xor(v[4], v[12]),
      xor(v[5], v[13]),
      xor(v[6], v[14]),
      xor(v[7], v[15]),
    ];
    let mut out_hi = [
      xor(v[8], cv_vecs[0]),
      xor(v[9], cv_vecs[1]),
      xor(v[10], cv_vecs[2]),
      xor(v[11], cv_vecs[3]),
      xor(v[12], cv_vecs[4]),
      xor(v[13], cv_vecs[5]),
      xor(v[14], cv_vecs[6]),
      xor(v[15], cv_vecs[7]),
    ];

    transpose8x8(&mut out_lo);
    transpose8x8(&mut out_hi);

    for lane in 0..DEGREE {
      let base = out.add(lane * 64);
      storeu(out_lo[lane], base);
      storeu(out_hi[lane], base.add(32));
    }
  }
}

/// Generate 1 root output block (64 bytes).
/// Delegates to SSE4.1 implementation (AVX2 is overkill for single block).
///
/// # Safety
/// Caller must ensure AVX2 is available and that `out` is valid for `64` writable bytes.
#[target_feature(enable = "avx2")]
pub unsafe fn root_output_blocks1(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
  out: *mut u8,
) {
  // AVX2 implies SSE4.1, so delegate to the SSE4.1 implementation
  unsafe { super::sse41::root_output_blocks1(chaining_value, block_words, counter, block_len, flags, out) }
}

/// Generate 2 root output blocks (128 bytes) with consecutive counters.
/// Delegates to SSE4.1 implementation.
///
/// # Safety
/// Caller must ensure AVX2 is available and that `out` is valid for `128` writable bytes.
#[target_feature(enable = "avx2")]
pub unsafe fn root_output_blocks2(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
  out: *mut u8,
) {
  // AVX2 implies SSE4.1, so delegate to the SSE4.1 implementation
  unsafe { super::sse41::root_output_blocks2(chaining_value, block_words, counter, block_len, flags, out) }
}
