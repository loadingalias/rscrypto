//! BLAKE3 x86_64 AVX-512 throughput kernel (16-way).

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
  super::{BLOCK_LEN, CHUNK_LEN, IV, MSG_SCHEDULE, OUT_LEN},
  avx2::transpose8x8,
  counter_high, counter_low,
};

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
      counter_low(counter.wrapping_add(mask & 1)) as i32,
      counter_low(counter.wrapping_add(mask & 2)) as i32,
      counter_low(counter.wrapping_add(mask & 3)) as i32,
      counter_low(counter.wrapping_add(mask & 4)) as i32,
      counter_low(counter.wrapping_add(mask & 5)) as i32,
      counter_low(counter.wrapping_add(mask & 6)) as i32,
      counter_low(counter.wrapping_add(mask & 7)) as i32,
      counter_low(counter.wrapping_add(mask & 8)) as i32,
      counter_low(counter.wrapping_add(mask & 9)) as i32,
      counter_low(counter.wrapping_add(mask & 10)) as i32,
      counter_low(counter.wrapping_add(mask & 11)) as i32,
      counter_low(counter.wrapping_add(mask & 12)) as i32,
      counter_low(counter.wrapping_add(mask & 13)) as i32,
      counter_low(counter.wrapping_add(mask & 14)) as i32,
      counter_low(counter.wrapping_add(mask & 15)) as i32,
    );
    let hi = _mm512_setr_epi32(
      counter_high(counter) as i32,
      counter_high(counter.wrapping_add(mask & 1)) as i32,
      counter_high(counter.wrapping_add(mask & 2)) as i32,
      counter_high(counter.wrapping_add(mask & 3)) as i32,
      counter_high(counter.wrapping_add(mask & 4)) as i32,
      counter_high(counter.wrapping_add(mask & 5)) as i32,
      counter_high(counter.wrapping_add(mask & 6)) as i32,
      counter_high(counter.wrapping_add(mask & 7)) as i32,
      counter_high(counter.wrapping_add(mask & 8)) as i32,
      counter_high(counter.wrapping_add(mask & 9)) as i32,
      counter_high(counter.wrapping_add(mask & 10)) as i32,
      counter_high(counter.wrapping_add(mask & 11)) as i32,
      counter_high(counter.wrapping_add(mask & 12)) as i32,
      counter_high(counter.wrapping_add(mask & 13)) as i32,
      counter_high(counter.wrapping_add(mask & 14)) as i32,
      counter_high(counter.wrapping_add(mask & 15)) as i32,
    );
    (lo, hi)
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
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
// Match upstream: AVX-512 detection is based on `avx512f` + `avx512vl`.
// The Linux backend delegates to upstream-grade asm, so we intentionally do
// not require BW/DQ here.
#[target_feature(enable = "avx512f,avx512vl,avx2")]
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
    super::asm::rscrypto_blake3_hash_many_avx512(
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
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
#[target_feature(enable = "avx512f,avx512vl,avx512dq,avx2")]
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

/// Generate 16 root output blocks (64 bytes each) in parallel.
///
/// Each lane uses an independent `output_block_counter` (`counter + lane`), but
/// shares the same `chaining_value`, `block_words`, `block_len`, and `flags`.
///
/// # Safety
/// Caller must ensure AVX-512 is available and that `out` is valid for `16 * 64`
/// writable bytes.
#[target_feature(enable = "avx512f,avx512vl,avx2")]
pub unsafe fn root_output_blocks16(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
  out: *mut u8,
) {
  #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
  {
    debug_assert!(flags <= u8::MAX as u32);
    debug_assert!(block_len <= u8::MAX as u32);
    // SAFETY: AVX-512 is available (checked by dispatch), and `out` is valid
    // for `16 * 64` writable bytes per this function's contract.
    unsafe {
      super::asm::rscrypto_blake3_xof_many_avx512(
        chaining_value.as_ptr(),
        block_words.as_ptr().cast(),
        block_len as u8,
        counter,
        flags as u8,
        out,
        16,
      );
    }
  }

  #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
  unsafe {
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

    let m = [
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

    let (counter_low_vec, counter_high_vec) = counter_vec(counter, true);
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

    round(&mut v, &m, 0);
    round(&mut v, &m, 1);
    round(&mut v, &m, 2);
    round(&mut v, &m, 3);
    round(&mut v, &m, 4);
    round(&mut v, &m, 5);
    round(&mut v, &m, 6);

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

    let mut lo0 = [_mm256_setzero_si256(); 8];
    let mut hi0 = [_mm256_setzero_si256(); 8];
    let mut lo1 = [_mm256_setzero_si256(); 8];
    let mut hi1 = [_mm256_setzero_si256(); 8];

    for i in 0..8 {
      lo0[i] = _mm512_castsi512_si256(out_words[i]);
      hi0[i] = _mm512_extracti64x4_epi64(out_words[i], 1);
      lo1[i] = _mm512_castsi512_si256(out_words[i + 8]);
      hi1[i] = _mm512_extracti64x4_epi64(out_words[i + 8], 1);
    }

    transpose8x8(&mut lo0);
    transpose8x8(&mut hi0);
    transpose8x8(&mut lo1);
    transpose8x8(&mut hi1);

    for lane in 0..8 {
      let base = out.add(lane * 64);
      storeu256(lo0[lane], base);
      storeu256(lo1[lane], base.add(32));
    }
    for lane in 0..8 {
      let base = out.add((lane + 8) * 64);
      storeu256(hi0[lane], base);
      storeu256(hi1[lane], base.add(32));
    }
  }
}

/// Generate 1 root output block (64 bytes).
/// On supported platforms, uses AVX-512 assembly for lower latency.
///
/// # Safety
/// Caller must ensure AVX-512 is available and that `out` is valid for `64` writable bytes.
#[target_feature(enable = "avx512f,avx512vl,avx2")]
pub unsafe fn root_output_blocks1(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
  out: *mut u8,
) {
  #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
  // SAFETY: AVX-512 is available (checked by dispatch), and `out` is valid
  // for 64 writable bytes per this function's contract.
  unsafe {
    super::asm::compress_xof_avx512(
      chaining_value,
      block_words.as_ptr().cast(),
      counter,
      block_len,
      flags,
      out,
    );
  }
  // Fallback for other platforms: delegate to SSE4.1 implementation
  #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
  unsafe {
    super::sse41::root_output_blocks1(chaining_value, block_words, counter, block_len, flags, out)
  }
}

/// Generate 2 root output blocks (128 bytes) with consecutive counters.
/// On supported platforms, uses AVX-512 assembly for lower latency.
///
/// # Safety
/// Caller must ensure AVX-512 is available and that `out` is valid for `128` writable bytes.
#[target_feature(enable = "avx512f,avx512vl,avx2")]
pub unsafe fn root_output_blocks2(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
  out: *mut u8,
) {
  #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
  // SAFETY: AVX-512 is available (checked by dispatch), and `out` is valid
  // for 128 writable bytes per this function's contract.
  unsafe {
    let block = block_words.as_ptr().cast();
    super::asm::compress_xof_avx512(chaining_value, block, counter, block_len, flags, out);
    super::asm::compress_xof_avx512(
      chaining_value,
      block,
      counter.wrapping_add(1),
      block_len,
      flags,
      out.add(64),
    );
  }
  // Fallback for other platforms: delegate to SSE4.1 implementation
  #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
  unsafe {
    super::sse41::root_output_blocks2(chaining_value, block_words, counter, block_len, flags, out)
  }
}

/// Compress one BLAKE3 block with a latency-oriented schedule.
///
/// This uses the same dependency-chain schedule as the SSE4.1/AVX2 single-block
/// path (no 16-lane broadcast + lane extraction), while keeping this entrypoint
/// AVX-512-gated for mixed-workload dispatching.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512VL + AVX2 + SSE4.1 + SSSE3 are available.
#[target_feature(enable = "avx512f,avx512vl,avx2,sse4.1,ssse3")]
pub(crate) unsafe fn compress_block(
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
    super::compress_pre_sse41_impl(chaining_value, m0, m1, m2, m3, counter, block_len, flags);

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

/// Compress one BLAKE3 block and return only the chaining value (8 words).
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512VL + AVX2 + SSE4.1 + SSSE3 are available.
#[target_feature(enable = "avx512f,avx512vl,avx2,sse4.1,ssse3")]
pub(crate) unsafe fn compress_cv_block(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  let m0 = _mm_loadu_si128(block_words.as_ptr().cast());
  let m1 = _mm_loadu_si128(block_words.as_ptr().add(4).cast());
  let m2 = _mm_loadu_si128(block_words.as_ptr().add(8).cast());
  let m3 = _mm_loadu_si128(block_words.as_ptr().add(12).cast());
  let [row0, row1, row2, row3] =
    super::compress_pre_sse41_impl(chaining_value, m0, m1, m2, m3, counter, block_len, flags);
  let row0 = _mm_xor_si128(row0, row2);
  let row1 = _mm_xor_si128(row1, row3);
  let mut out = [0u32; 8];
  _mm_storeu_si128(out.as_mut_ptr().cast(), row0);
  _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), row1);
  out
}
