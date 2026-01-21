//! BLAKE3 x86_64 AVX-512 throughput kernel (16-way).

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]
// On Linux we currently prefer the upstream asm implementation; keep the
// intrinsic fallback compiled but don't let `-D warnings` turn it into a build
// failure.
#![cfg_attr(target_os = "linux", allow(dead_code, unused_imports))]

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
unsafe fn loadu256(src: *const u8) -> __m256i {
  unsafe { _mm256_loadu_si256(src.cast()) }
}

#[inline(always)]
unsafe fn storeu256(src: __m256i, dest: *mut u8) {
  unsafe { _mm256_storeu_si256(dest.cast(), src) }
}

#[inline(always)]
unsafe fn rot16(x: __m512i) -> __m512i {
  // 32-bit rotate by 16. Prefer `vprold` over shift/or.
  unsafe { _mm512_rol_epi32(x, 16) }
}

#[inline(always)]
unsafe fn rot12(x: __m512i) -> __m512i {
  // Rotate right by 12 == rotate left by 20.
  unsafe { _mm512_rol_epi32(x, 20) }
}

#[inline(always)]
unsafe fn rot8(x: __m512i) -> __m512i {
  // Rotate right by 8 == rotate left by 24.
  unsafe { _mm512_rol_epi32(x, 24) }
}

#[inline(always)]
unsafe fn rot7(x: __m512i) -> __m512i {
  // Rotate right by 7 == rotate left by 25.
  unsafe { _mm512_rol_epi32(x, 25) }
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
unsafe fn transpose8x8(vecs: &mut [__m256i; 8]) {
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
unsafe fn transpose_msg_vecs8(inputs: &[*const u8; 8], block_offset: usize) -> [__m256i; 16] {
  unsafe {
    let stride = 4 * 8;
    let mut half0 = [
      loadu256(inputs[0].add(block_offset)),
      loadu256(inputs[1].add(block_offset)),
      loadu256(inputs[2].add(block_offset)),
      loadu256(inputs[3].add(block_offset)),
      loadu256(inputs[4].add(block_offset)),
      loadu256(inputs[5].add(block_offset)),
      loadu256(inputs[6].add(block_offset)),
      loadu256(inputs[7].add(block_offset)),
    ];
    let mut half1 = [
      loadu256(inputs[0].add(block_offset + stride)),
      loadu256(inputs[1].add(block_offset + stride)),
      loadu256(inputs[2].add(block_offset + stride)),
      loadu256(inputs[3].add(block_offset + stride)),
      loadu256(inputs[4].add(block_offset + stride)),
      loadu256(inputs[5].add(block_offset + stride)),
      loadu256(inputs[6].add(block_offset + stride)),
      loadu256(inputs[7].add(block_offset + stride)),
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
unsafe fn transpose_msg_vecs16(inputs: &[*const u8; 16], block_offset: usize) -> [__m512i; 16] {
  unsafe {
    let lo_ptrs = [
      inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7],
    ];
    let hi_ptrs = [
      inputs[8], inputs[9], inputs[10], inputs[11], inputs[12], inputs[13], inputs[14], inputs[15],
    ];

    let lo = transpose_msg_vecs8(&lo_ptrs, block_offset);
    let hi = transpose_msg_vecs8(&hi_ptrs, block_offset);

    let mut out = [set1(0); 16];
    for i in 0..16 {
      let mut v = _mm512_castsi256_si512(lo[i]);
      v = _mm512_inserti64x4(v, hi[i], 1);
      out[i] = v;
    }

    out
  }
}

/// Hash 16 contiguous independent inputs in parallel.
///
/// This is optimized for the contiguous chunk hashing hot path, where inputs
/// are arranged as `CHUNK_LEN`-byte blocks back-to-back.
///
/// # Safety
/// Caller must ensure AVX-512 is available, and `input`/`out` are valid for
/// `DEGREE * CHUNK_LEN` and `DEGREE * OUT_LEN` bytes respectively.
#[cfg(target_os = "linux")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,avx2")]
pub unsafe fn hash16_contiguous(input: *const u8, key: &[u32; 8], counter: u64, flags: u32, out: *mut u8) {
  // Delegate to the upstream-grade AVX-512 asm implementation on Linux.
  //
  // The upstream function accepts an array of input pointers, so we build it
  // from the contiguous layout.
  debug_assert!(flags <= u8::MAX as u32);
  unsafe {
    let inputs = [
      input,
      input.add(CHUNK_LEN),
      input.add(2 * CHUNK_LEN),
      input.add(3 * CHUNK_LEN),
      input.add(4 * CHUNK_LEN),
      input.add(5 * CHUNK_LEN),
      input.add(6 * CHUNK_LEN),
      input.add(7 * CHUNK_LEN),
      input.add(8 * CHUNK_LEN),
      input.add(9 * CHUNK_LEN),
      input.add(10 * CHUNK_LEN),
      input.add(11 * CHUNK_LEN),
      input.add(12 * CHUNK_LEN),
      input.add(13 * CHUNK_LEN),
      input.add(14 * CHUNK_LEN),
      input.add(15 * CHUNK_LEN),
    ];
    let flags_start = (flags | super::super::CHUNK_START) as u8;
    let flags_end = (flags | super::super::CHUNK_END) as u8;
    super::asm_linux::rscrypto_blake3_hash_many_avx512(
      inputs.as_ptr(),
      DEGREE,
      CHUNK_LEN / BLOCK_LEN,
      key.as_ptr(),
      counter,
      true,
      flags as u8,
      flags_start,
      flags_end,
      out,
    );
  }
}

/// Hash 16 contiguous independent inputs in parallel.
///
/// This is optimized for the contiguous chunk hashing hot path, where inputs
/// are arranged as `CHUNK_LEN`-byte blocks back-to-back.
///
/// # Safety
/// Caller must ensure AVX-512 is available, and `input`/`out` are valid for
/// `DEGREE * CHUNK_LEN` and `DEGREE * OUT_LEN` bytes respectively.
#[cfg(not(target_os = "linux"))]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,avx2")]
pub unsafe fn hash16_contiguous(input: *const u8, key: &[u32; 8], counter: u64, flags: u32, out: *mut u8) {
  unsafe {
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

    let inputs = [
      input,
      input.add(CHUNK_LEN),
      input.add(2 * CHUNK_LEN),
      input.add(3 * CHUNK_LEN),
      input.add(4 * CHUNK_LEN),
      input.add(5 * CHUNK_LEN),
      input.add(6 * CHUNK_LEN),
      input.add(7 * CHUNK_LEN),
      input.add(8 * CHUNK_LEN),
      input.add(9 * CHUNK_LEN),
      input.add(10 * CHUNK_LEN),
      input.add(11 * CHUNK_LEN),
      input.add(12 * CHUNK_LEN),
      input.add(13 * CHUNK_LEN),
      input.add(14 * CHUNK_LEN),
      input.add(15 * CHUNK_LEN),
    ];

    let (counter_low_vec, counter_high_vec) = counter_vec(counter, true);

    let blocks = CHUNK_LEN / BLOCK_LEN;
    for block in 0..blocks {
      let mut block_flags = flags;
      if block == 0 {
        block_flags |= super::super::CHUNK_START;
      }
      if block + 1 == blocks {
        block_flags |= super::super::CHUNK_END;
      }

      let block_flags_vec = set1(block_flags);

      let m = transpose_msg_vecs16(&inputs, block * BLOCK_LEN);

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

    // Convert word-major vectors into `[chunk][word]` order without scatter.
    let mut lo = [_mm256_setzero_si256(); 8];
    let mut hi = [_mm256_setzero_si256(); 8];
    for i in 0..8 {
      lo[i] = _mm512_castsi512_si256(h_vecs[i]);
      hi[i] = _mm512_extracti64x4_epi64(h_vecs[i], 1);
    }

    transpose8x8(&mut lo);
    transpose8x8(&mut hi);

    for chunk in 0..8 {
      storeu256(lo[chunk], out.add(chunk * OUT_LEN));
      storeu256(hi[chunk], out.add((chunk + 8) * OUT_LEN));
    }
  }
}
