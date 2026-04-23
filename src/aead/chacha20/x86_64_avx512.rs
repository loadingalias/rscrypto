use core::arch::x86_64::{
  __m512i, _mm512_add_epi32, _mm512_loadu_si512, _mm512_rol_epi32, _mm512_set1_epi32, _mm512_setr_epi32,
  _mm512_shuffle_i32x4, _mm512_storeu_si512, _mm512_unpackhi_epi32, _mm512_unpackhi_epi64, _mm512_unpacklo_epi32,
  _mm512_unpacklo_epi64, _mm512_xor_si512,
};

use super::{BLOCK_SIZE, KEY_SIZE, NONCE_SIZE, load_u32_le, xor_keystream_portable};

const BLOCKS_PER_BATCH: usize = 16;

#[inline]
pub(super) fn xor_keystream(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
  // SAFETY: Backend selection guarantees the AVX-512 feature set required by this kernel.
  unsafe { xor_keystream_impl(key, initial_counter, nonce, buffer) }
}

#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
unsafe fn xor_keystream_impl(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
  let mut counter = initial_counter;
  let mut batches = buffer.chunks_exact_mut(BLOCK_SIZE * BLOCKS_PER_BATCH);
  for chunk in &mut batches {
    debug_assert!(counter.checked_add((BLOCKS_PER_BATCH - 1) as u32).is_some());

    let mut x0 = _mm512_set1_epi32(0x6170_7865u32 as i32);
    let mut x1 = _mm512_set1_epi32(0x3320_646eu32 as i32);
    let mut x2 = _mm512_set1_epi32(0x7962_2d32u32 as i32);
    let mut x3 = _mm512_set1_epi32(0x6b20_6574u32 as i32);
    let mut x4 = _mm512_set1_epi32(load_u32_le(&key[0..4]) as i32);
    let mut x5 = _mm512_set1_epi32(load_u32_le(&key[4..8]) as i32);
    let mut x6 = _mm512_set1_epi32(load_u32_le(&key[8..12]) as i32);
    let mut x7 = _mm512_set1_epi32(load_u32_le(&key[12..16]) as i32);
    let mut x8 = _mm512_set1_epi32(load_u32_le(&key[16..20]) as i32);
    let mut x9 = _mm512_set1_epi32(load_u32_le(&key[20..24]) as i32);
    let mut x10 = _mm512_set1_epi32(load_u32_le(&key[24..28]) as i32);
    let mut x11 = _mm512_set1_epi32(load_u32_le(&key[28..32]) as i32);
    let mut x12 = _mm512_setr_epi32(
      counter as i32,
      counter.wrapping_add(1) as i32,
      counter.wrapping_add(2) as i32,
      counter.wrapping_add(3) as i32,
      counter.wrapping_add(4) as i32,
      counter.wrapping_add(5) as i32,
      counter.wrapping_add(6) as i32,
      counter.wrapping_add(7) as i32,
      counter.wrapping_add(8) as i32,
      counter.wrapping_add(9) as i32,
      counter.wrapping_add(10) as i32,
      counter.wrapping_add(11) as i32,
      counter.wrapping_add(12) as i32,
      counter.wrapping_add(13) as i32,
      counter.wrapping_add(14) as i32,
      counter.wrapping_add(15) as i32,
    );
    let mut x13 = _mm512_set1_epi32(load_u32_le(&nonce[0..4]) as i32);
    let mut x14 = _mm512_set1_epi32(load_u32_le(&nonce[4..8]) as i32);
    let mut x15 = _mm512_set1_epi32(load_u32_le(&nonce[8..12]) as i32);

    let o0 = x0;
    let o1 = x1;
    let o2 = x2;
    let o3 = x3;
    let o4 = x4;
    let o5 = x5;
    let o6 = x6;
    let o7 = x7;
    let o8 = x8;
    let o9 = x9;
    let o10 = x10;
    let o11 = x11;
    let o12 = x12;
    let o13 = x13;
    let o14 = x14;
    let o15 = x15;

    let mut round = 0usize;
    while round < 10 {
      quarter_round(&mut x0, &mut x4, &mut x8, &mut x12);
      quarter_round(&mut x1, &mut x5, &mut x9, &mut x13);
      quarter_round(&mut x2, &mut x6, &mut x10, &mut x14);
      quarter_round(&mut x3, &mut x7, &mut x11, &mut x15);

      quarter_round(&mut x0, &mut x5, &mut x10, &mut x15);
      quarter_round(&mut x1, &mut x6, &mut x11, &mut x12);
      quarter_round(&mut x2, &mut x7, &mut x8, &mut x13);
      quarter_round(&mut x3, &mut x4, &mut x9, &mut x14);

      round = round.strict_add(1);
    }

    x0 = _mm512_add_epi32(x0, o0);
    x1 = _mm512_add_epi32(x1, o1);
    x2 = _mm512_add_epi32(x2, o2);
    x3 = _mm512_add_epi32(x3, o3);
    x4 = _mm512_add_epi32(x4, o4);
    x5 = _mm512_add_epi32(x5, o5);
    x6 = _mm512_add_epi32(x6, o6);
    x7 = _mm512_add_epi32(x7, o7);
    x8 = _mm512_add_epi32(x8, o8);
    x9 = _mm512_add_epi32(x9, o9);
    x10 = _mm512_add_epi32(x10, o10);
    x11 = _mm512_add_epi32(x11, o11);
    x12 = _mm512_add_epi32(x12, o12);
    x13 = _mm512_add_epi32(x13, o13);
    x14 = _mm512_add_epi32(x14, o14);
    x15 = _mm512_add_epi32(x15, o15);

    // 16×16 u32 matrix transpose: convert from word-major (x_i = word i for all
    // 16 blocks) to block-major (each register = one complete 64-byte block).
    //
    // Stage 1: 32-bit interleave — pairwise unpack adjacent state-word registers.
    // SAFETY: AVX-512 intrinsics are valid under the enclosing target_feature.
    let s1_0 = _mm512_unpacklo_epi32(x0, x1);
    let s1_1 = _mm512_unpackhi_epi32(x0, x1);
    let s1_2 = _mm512_unpacklo_epi32(x2, x3);
    let s1_3 = _mm512_unpackhi_epi32(x2, x3);
    let s1_4 = _mm512_unpacklo_epi32(x4, x5);
    let s1_5 = _mm512_unpackhi_epi32(x4, x5);
    let s1_6 = _mm512_unpacklo_epi32(x6, x7);
    let s1_7 = _mm512_unpackhi_epi32(x6, x7);
    let s1_8 = _mm512_unpacklo_epi32(x8, x9);
    let s1_9 = _mm512_unpackhi_epi32(x8, x9);
    let s1_10 = _mm512_unpacklo_epi32(x10, x11);
    let s1_11 = _mm512_unpackhi_epi32(x10, x11);
    let s1_12 = _mm512_unpacklo_epi32(x12, x13);
    let s1_13 = _mm512_unpackhi_epi32(x12, x13);
    let s1_14 = _mm512_unpacklo_epi32(x14, x15);
    let s1_15 = _mm512_unpackhi_epi32(x14, x15);

    // Stage 2: 64-bit interleave — pair up stage-1 results to get 4 consecutive
    // words from one block in each 128-bit lane.
    let s2_0 = _mm512_unpacklo_epi64(s1_0, s1_2);
    let s2_1 = _mm512_unpackhi_epi64(s1_0, s1_2);
    let s2_2 = _mm512_unpacklo_epi64(s1_1, s1_3);
    let s2_3 = _mm512_unpackhi_epi64(s1_1, s1_3);
    let s2_4 = _mm512_unpacklo_epi64(s1_4, s1_6);
    let s2_5 = _mm512_unpackhi_epi64(s1_4, s1_6);
    let s2_6 = _mm512_unpacklo_epi64(s1_5, s1_7);
    let s2_7 = _mm512_unpackhi_epi64(s1_5, s1_7);
    let s2_8 = _mm512_unpacklo_epi64(s1_8, s1_10);
    let s2_9 = _mm512_unpackhi_epi64(s1_8, s1_10);
    let s2_10 = _mm512_unpacklo_epi64(s1_9, s1_11);
    let s2_11 = _mm512_unpackhi_epi64(s1_9, s1_11);
    let s2_12 = _mm512_unpacklo_epi64(s1_12, s1_14);
    let s2_13 = _mm512_unpackhi_epi64(s1_12, s1_14);
    let s2_14 = _mm512_unpacklo_epi64(s1_13, s1_15);
    let s2_15 = _mm512_unpackhi_epi64(s1_13, s1_15);

    // Stage 3: 128-bit lane shuffle — assemble complete 64-byte blocks from
    // four word-groups distributed across s2 registers.
    //
    // After stage 2, s2[m] lane L holds 4 consecutive words of block (4L + offset).
    // We combine word-groups {0-3, 4-7, 8-11, 12-15} for each block using
    // shuffle_i32x4 which selects 128-bit lanes from two source registers.
    //
    // For blocks {0,4,8,12} — from s2[0], s2[4], s2[8], s2[12]:
    let ab_04_lo = _mm512_shuffle_i32x4::<0b_01_00_01_00>(s2_0, s2_4);
    let cd_04_lo = _mm512_shuffle_i32x4::<0b_01_00_01_00>(s2_8, s2_12);
    let ab_04_hi = _mm512_shuffle_i32x4::<0b_11_10_11_10>(s2_0, s2_4);
    let cd_04_hi = _mm512_shuffle_i32x4::<0b_11_10_11_10>(s2_8, s2_12);
    let blk0 = _mm512_shuffle_i32x4::<0b_10_00_10_00>(ab_04_lo, cd_04_lo);
    let blk4 = _mm512_shuffle_i32x4::<0b_11_01_11_01>(ab_04_lo, cd_04_lo);
    let blk8 = _mm512_shuffle_i32x4::<0b_10_00_10_00>(ab_04_hi, cd_04_hi);
    let blk12 = _mm512_shuffle_i32x4::<0b_11_01_11_01>(ab_04_hi, cd_04_hi);

    // For blocks {1,5,9,13} — from s2[1], s2[5], s2[9], s2[13]:
    let ab_15_lo = _mm512_shuffle_i32x4::<0b_01_00_01_00>(s2_1, s2_5);
    let cd_15_lo = _mm512_shuffle_i32x4::<0b_01_00_01_00>(s2_9, s2_13);
    let ab_15_hi = _mm512_shuffle_i32x4::<0b_11_10_11_10>(s2_1, s2_5);
    let cd_15_hi = _mm512_shuffle_i32x4::<0b_11_10_11_10>(s2_9, s2_13);
    let blk1 = _mm512_shuffle_i32x4::<0b_10_00_10_00>(ab_15_lo, cd_15_lo);
    let blk5 = _mm512_shuffle_i32x4::<0b_11_01_11_01>(ab_15_lo, cd_15_lo);
    let blk9 = _mm512_shuffle_i32x4::<0b_10_00_10_00>(ab_15_hi, cd_15_hi);
    let blk13 = _mm512_shuffle_i32x4::<0b_11_01_11_01>(ab_15_hi, cd_15_hi);

    // For blocks {2,6,10,14} — from s2[2], s2[6], s2[10], s2[14]:
    let ab_26_lo = _mm512_shuffle_i32x4::<0b_01_00_01_00>(s2_2, s2_6);
    let cd_26_lo = _mm512_shuffle_i32x4::<0b_01_00_01_00>(s2_10, s2_14);
    let ab_26_hi = _mm512_shuffle_i32x4::<0b_11_10_11_10>(s2_2, s2_6);
    let cd_26_hi = _mm512_shuffle_i32x4::<0b_11_10_11_10>(s2_10, s2_14);
    let blk2 = _mm512_shuffle_i32x4::<0b_10_00_10_00>(ab_26_lo, cd_26_lo);
    let blk6 = _mm512_shuffle_i32x4::<0b_11_01_11_01>(ab_26_lo, cd_26_lo);
    let blk10 = _mm512_shuffle_i32x4::<0b_10_00_10_00>(ab_26_hi, cd_26_hi);
    let blk14 = _mm512_shuffle_i32x4::<0b_11_01_11_01>(ab_26_hi, cd_26_hi);

    // For blocks {3,7,11,15} — from s2[3], s2[7], s2[11], s2[15]:
    let ab_37_lo = _mm512_shuffle_i32x4::<0b_01_00_01_00>(s2_3, s2_7);
    let cd_37_lo = _mm512_shuffle_i32x4::<0b_01_00_01_00>(s2_11, s2_15);
    let ab_37_hi = _mm512_shuffle_i32x4::<0b_11_10_11_10>(s2_3, s2_7);
    let cd_37_hi = _mm512_shuffle_i32x4::<0b_11_10_11_10>(s2_11, s2_15);
    let blk3 = _mm512_shuffle_i32x4::<0b_10_00_10_00>(ab_37_lo, cd_37_lo);
    let blk7 = _mm512_shuffle_i32x4::<0b_11_01_11_01>(ab_37_lo, cd_37_lo);
    let blk11 = _mm512_shuffle_i32x4::<0b_10_00_10_00>(ab_37_hi, cd_37_hi);
    let blk15 = _mm512_shuffle_i32x4::<0b_11_01_11_01>(ab_37_hi, cd_37_hi);

    // XOR each transposed block with plaintext and store in-place.
    let ptr = chunk.as_mut_ptr();
    // SAFETY: each `chunk` slice has exactly BLOCKS_PER_BATCH * BLOCK_SIZE = 1024
    // bytes, so all 16 × 64-byte load/store pairs are in bounds.
    unsafe {
      xor_block(ptr, 0, blk0);
      xor_block(ptr, 1, blk1);
      xor_block(ptr, 2, blk2);
      xor_block(ptr, 3, blk3);
      xor_block(ptr, 4, blk4);
      xor_block(ptr, 5, blk5);
      xor_block(ptr, 6, blk6);
      xor_block(ptr, 7, blk7);
      xor_block(ptr, 8, blk8);
      xor_block(ptr, 9, blk9);
      xor_block(ptr, 10, blk10);
      xor_block(ptr, 11, blk11);
      xor_block(ptr, 12, blk12);
      xor_block(ptr, 13, blk13);
      xor_block(ptr, 14, blk14);
      xor_block(ptr, 15, blk15);
    }

    counter = counter.wrapping_add(BLOCKS_PER_BATCH as u32);
  }

  let remainder = batches.into_remainder();
  if !remainder.is_empty() {
    xor_keystream_portable(key, counter, nonce, remainder);
  }
}

/// Load 64 bytes of plaintext at block offset `idx`, XOR with keystream block, store.
#[inline(always)]
unsafe fn xor_block(buf: *mut u8, idx: usize, keystream: __m512i) {
  // SAFETY: caller guarantees `buf` points to a 1024-byte chunk and `idx < 16`.
  unsafe {
    let p = buf.add(idx.strict_mul(BLOCK_SIZE)).cast::<__m512i>();
    let plaintext = _mm512_loadu_si512(p);
    _mm512_storeu_si512(p, _mm512_xor_si512(plaintext, keystream));
  }
}

#[inline(always)]
fn quarter_round(a: &mut __m512i, b: &mut __m512i, c: &mut __m512i, d: &mut __m512i) {
  // SAFETY: this helper is only called from the AVX-512 kernel, so all
  // `_mm512_*` intrinsics are valid for the current compilation target.
  unsafe {
    *a = _mm512_add_epi32(*a, *b);
    *d = _mm512_rol_epi32::<16>(_mm512_xor_si512(*d, *a));
    *c = _mm512_add_epi32(*c, *d);
    *b = _mm512_rol_epi32::<12>(_mm512_xor_si512(*b, *c));
    *a = _mm512_add_epi32(*a, *b);
    *d = _mm512_rol_epi32::<8>(_mm512_xor_si512(*d, *a));
    *c = _mm512_add_epi32(*c, *d);
    *b = _mm512_rol_epi32::<7>(_mm512_xor_si512(*b, *c));
  }
}
