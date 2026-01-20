//! BLAKE3 x86_64 AVX2 throughput kernel (8-way).

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]

use core::arch::x86_64::*;

use super::super::{BLOCK_LEN, IV, MSG_SCHEDULE, OUT_LEN};

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
const fn counter_low(counter: u64) -> u32 {
  counter as u32
}

#[inline(always)]
const fn counter_high(counter: u64) -> u32 {
  (counter >> 32) as u32
}

#[inline(always)]
unsafe fn rot16(x: __m256i) -> __m256i {
  unsafe { _mm256_or_si256(_mm256_srli_epi32(x, 16), _mm256_slli_epi32(x, 16)) }
}

#[inline(always)]
unsafe fn rot12(x: __m256i) -> __m256i {
  unsafe { _mm256_or_si256(_mm256_srli_epi32(x, 12), _mm256_slli_epi32(x, 20)) }
}

#[inline(always)]
unsafe fn rot8(x: __m256i) -> __m256i {
  unsafe { _mm256_or_si256(_mm256_srli_epi32(x, 8), _mm256_slli_epi32(x, 24)) }
}

#[inline(always)]
unsafe fn rot7(x: __m256i) -> __m256i {
  unsafe { _mm256_or_si256(_mm256_srli_epi32(x, 7), _mm256_slli_epi32(x, 25)) }
}

#[inline(always)]
unsafe fn round(v: &mut [__m256i; 16], m: &[__m256i; 16], r: usize) {
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
unsafe fn interleave128(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
  unsafe {
    (
      _mm256_permute2x128_si256(a, b, 0x20),
      _mm256_permute2x128_si256(a, b, 0x31),
    )
  }
}

#[inline(always)]
unsafe fn transpose_vecs(vecs: &mut [__m256i; DEGREE]) {
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
    let mut half0 = [
      loadu(inputs[0].add(block_offset + 0 * 4 * DEGREE)),
      loadu(inputs[1].add(block_offset + 0 * 4 * DEGREE)),
      loadu(inputs[2].add(block_offset + 0 * 4 * DEGREE)),
      loadu(inputs[3].add(block_offset + 0 * 4 * DEGREE)),
      loadu(inputs[4].add(block_offset + 0 * 4 * DEGREE)),
      loadu(inputs[5].add(block_offset + 0 * 4 * DEGREE)),
      loadu(inputs[6].add(block_offset + 0 * 4 * DEGREE)),
      loadu(inputs[7].add(block_offset + 0 * 4 * DEGREE)),
    ];
    let mut half1 = [
      loadu(inputs[0].add(block_offset + 1 * 4 * DEGREE)),
      loadu(inputs[1].add(block_offset + 1 * 4 * DEGREE)),
      loadu(inputs[2].add(block_offset + 1 * 4 * DEGREE)),
      loadu(inputs[3].add(block_offset + 1 * 4 * DEGREE)),
      loadu(inputs[4].add(block_offset + 1 * 4 * DEGREE)),
      loadu(inputs[5].add(block_offset + 1 * 4 * DEGREE)),
      loadu(inputs[6].add(block_offset + 1 * 4 * DEGREE)),
      loadu(inputs[7].add(block_offset + 1 * 4 * DEGREE)),
    ];

    for i in 0..DEGREE {
      _mm_prefetch(inputs[i].wrapping_add(block_offset + 256).cast::<i8>(), _MM_HINT_T0);
    }

    transpose_vecs(&mut half0);
    transpose_vecs(&mut half1);

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
        counter_low(counter + (mask & 0)),
        counter_low(counter + (mask & 1)),
        counter_low(counter + (mask & 2)),
        counter_low(counter + (mask & 3)),
        counter_low(counter + (mask & 4)),
        counter_low(counter + (mask & 5)),
        counter_low(counter + (mask & 6)),
        counter_low(counter + (mask & 7)),
      ),
      set8(
        counter_high(counter + (mask & 0)),
        counter_high(counter + (mask & 1)),
        counter_high(counter + (mask & 2)),
        counter_high(counter + (mask & 3)),
        counter_high(counter + (mask & 4)),
        counter_high(counter + (mask & 5)),
        counter_high(counter + (mask & 6)),
        counter_high(counter + (mask & 7)),
      ),
    )
  }
}

/// Hash `DEGREE` independent inputs in parallel.
///
/// # Safety
/// Caller must ensure AVX2 is available and that all input pointers are valid
/// for `blocks * BLOCK_LEN` bytes.
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

      let block_len_vec = set1(BLOCK_LEN as u32);
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
        set1(IV[0]),
        set1(IV[1]),
        set1(IV[2]),
        set1(IV[3]),
        counter_low_vec,
        counter_high_vec,
        block_len_vec,
        block_flags_vec,
      ];

      round(&mut v, &msg_vecs, 0);
      round(&mut v, &msg_vecs, 1);
      round(&mut v, &msg_vecs, 2);
      round(&mut v, &msg_vecs, 3);
      round(&mut v, &msg_vecs, 4);
      round(&mut v, &msg_vecs, 5);
      round(&mut v, &msg_vecs, 6);

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
    transpose_vecs(&mut h_vecs);

    storeu(h_vecs[0], out.add(0 * 4 * DEGREE));
    storeu(h_vecs[1], out.add(1 * 4 * DEGREE));
    storeu(h_vecs[2], out.add(2 * 4 * DEGREE));
    storeu(h_vecs[3], out.add(3 * 4 * DEGREE));
    storeu(h_vecs[4], out.add(4 * 4 * DEGREE));
    storeu(h_vecs[5], out.add(5 * 4 * DEGREE));
    storeu(h_vecs[6], out.add(6 * 4 * DEGREE));
    storeu(h_vecs[7], out.add(7 * 4 * DEGREE));
  }
}
