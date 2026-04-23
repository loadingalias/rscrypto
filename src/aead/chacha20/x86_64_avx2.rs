use core::arch::x86_64::{
  __m256i, _mm256_add_epi32, _mm256_loadu_si256, _mm256_or_si256, _mm256_permute2x128_si256, _mm256_set_epi8,
  _mm256_set1_epi32, _mm256_setr_epi32, _mm256_shuffle_epi8, _mm256_slli_epi32, _mm256_srli_epi32, _mm256_storeu_si256,
  _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64, _mm256_xor_si256,
};

use super::{BLOCK_SIZE, KEY_SIZE, NONCE_SIZE, load_u32_le, xor_keystream_portable};

const BLOCKS_PER_BATCH: usize = 8;

#[inline]
pub(super) fn xor_keystream(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
  // SAFETY: Backend selection guarantees AVX2 is available before this wrapper is chosen.
  unsafe { xor_keystream_impl(key, initial_counter, nonce, buffer) }
}

#[target_feature(enable = "avx2")]
unsafe fn xor_keystream_impl(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
  // vpshufb masks for byte-aligned rotations (16-bit and 8-bit).
  let rot16 = _mm256_set_epi8(
    13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2,
  );
  let rot8 = _mm256_set_epi8(
    14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3, 14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3,
  );

  let mut counter = initial_counter;
  let mut batches = buffer.chunks_exact_mut(BLOCK_SIZE * BLOCKS_PER_BATCH);
  for chunk in &mut batches {
    debug_assert!(counter.checked_add((BLOCKS_PER_BATCH - 1) as u32).is_some());

    let mut x0 = _mm256_set1_epi32(0x6170_7865u32 as i32);
    let mut x1 = _mm256_set1_epi32(0x3320_646eu32 as i32);
    let mut x2 = _mm256_set1_epi32(0x7962_2d32u32 as i32);
    let mut x3 = _mm256_set1_epi32(0x6b20_6574u32 as i32);
    let mut x4 = _mm256_set1_epi32(load_u32_le(&key[0..4]) as i32);
    let mut x5 = _mm256_set1_epi32(load_u32_le(&key[4..8]) as i32);
    let mut x6 = _mm256_set1_epi32(load_u32_le(&key[8..12]) as i32);
    let mut x7 = _mm256_set1_epi32(load_u32_le(&key[12..16]) as i32);
    let mut x8 = _mm256_set1_epi32(load_u32_le(&key[16..20]) as i32);
    let mut x9 = _mm256_set1_epi32(load_u32_le(&key[20..24]) as i32);
    let mut x10 = _mm256_set1_epi32(load_u32_le(&key[24..28]) as i32);
    let mut x11 = _mm256_set1_epi32(load_u32_le(&key[28..32]) as i32);
    let mut x12 = _mm256_setr_epi32(
      counter as i32,
      counter.wrapping_add(1) as i32,
      counter.wrapping_add(2) as i32,
      counter.wrapping_add(3) as i32,
      counter.wrapping_add(4) as i32,
      counter.wrapping_add(5) as i32,
      counter.wrapping_add(6) as i32,
      counter.wrapping_add(7) as i32,
    );
    let mut x13 = _mm256_set1_epi32(load_u32_le(&nonce[0..4]) as i32);
    let mut x14 = _mm256_set1_epi32(load_u32_le(&nonce[4..8]) as i32);
    let mut x15 = _mm256_set1_epi32(load_u32_le(&nonce[8..12]) as i32);

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
      quarter_round(&mut x0, &mut x4, &mut x8, &mut x12, rot16, rot8);
      quarter_round(&mut x1, &mut x5, &mut x9, &mut x13, rot16, rot8);
      quarter_round(&mut x2, &mut x6, &mut x10, &mut x14, rot16, rot8);
      quarter_round(&mut x3, &mut x7, &mut x11, &mut x15, rot16, rot8);

      quarter_round(&mut x0, &mut x5, &mut x10, &mut x15, rot16, rot8);
      quarter_round(&mut x1, &mut x6, &mut x11, &mut x12, rot16, rot8);
      quarter_round(&mut x2, &mut x7, &mut x8, &mut x13, rot16, rot8);
      quarter_round(&mut x3, &mut x4, &mut x9, &mut x14, rot16, rot8);

      round = round.strict_add(1);
    }

    x0 = _mm256_add_epi32(x0, o0);
    x1 = _mm256_add_epi32(x1, o1);
    x2 = _mm256_add_epi32(x2, o2);
    x3 = _mm256_add_epi32(x3, o3);
    x4 = _mm256_add_epi32(x4, o4);
    x5 = _mm256_add_epi32(x5, o5);
    x6 = _mm256_add_epi32(x6, o6);
    x7 = _mm256_add_epi32(x7, o7);
    x8 = _mm256_add_epi32(x8, o8);
    x9 = _mm256_add_epi32(x9, o9);
    x10 = _mm256_add_epi32(x10, o10);
    x11 = _mm256_add_epi32(x11, o11);
    x12 = _mm256_add_epi32(x12, o12);
    x13 = _mm256_add_epi32(x13, o13);
    x14 = _mm256_add_epi32(x14, o14);
    x15 = _mm256_add_epi32(x15, o15);

    // 16×8 matrix transpose: convert from word-major (x_i = word i for 8 blocks)
    // to block-major (each pair of YMM registers = one 64-byte block).
    //
    // Stage 1: 32-bit interleave.
    // SAFETY: AVX2 intrinsics are valid under the enclosing target_feature.
    let s1_0 = _mm256_unpacklo_epi32(x0, x1);
    let s1_1 = _mm256_unpackhi_epi32(x0, x1);
    let s1_2 = _mm256_unpacklo_epi32(x2, x3);
    let s1_3 = _mm256_unpackhi_epi32(x2, x3);
    let s1_4 = _mm256_unpacklo_epi32(x4, x5);
    let s1_5 = _mm256_unpackhi_epi32(x4, x5);
    let s1_6 = _mm256_unpacklo_epi32(x6, x7);
    let s1_7 = _mm256_unpackhi_epi32(x6, x7);
    let s1_8 = _mm256_unpacklo_epi32(x8, x9);
    let s1_9 = _mm256_unpackhi_epi32(x8, x9);
    let s1_10 = _mm256_unpacklo_epi32(x10, x11);
    let s1_11 = _mm256_unpackhi_epi32(x10, x11);
    let s1_12 = _mm256_unpacklo_epi32(x12, x13);
    let s1_13 = _mm256_unpackhi_epi32(x12, x13);
    let s1_14 = _mm256_unpacklo_epi32(x14, x15);
    let s1_15 = _mm256_unpackhi_epi32(x14, x15);

    // Stage 2: 64-bit interleave. After this, each 128-bit lane holds 4 consecutive
    // words from one block (lo lane = blocks 0-3, hi lane = blocks 4-7).
    let s2_0 = _mm256_unpacklo_epi64(s1_0, s1_2);
    let s2_1 = _mm256_unpackhi_epi64(s1_0, s1_2);
    let s2_2 = _mm256_unpacklo_epi64(s1_1, s1_3);
    let s2_3 = _mm256_unpackhi_epi64(s1_1, s1_3);
    let s2_4 = _mm256_unpacklo_epi64(s1_4, s1_6);
    let s2_5 = _mm256_unpackhi_epi64(s1_4, s1_6);
    let s2_6 = _mm256_unpacklo_epi64(s1_5, s1_7);
    let s2_7 = _mm256_unpackhi_epi64(s1_5, s1_7);
    let s2_8 = _mm256_unpacklo_epi64(s1_8, s1_10);
    let s2_9 = _mm256_unpackhi_epi64(s1_8, s1_10);
    let s2_10 = _mm256_unpacklo_epi64(s1_9, s1_11);
    let s2_11 = _mm256_unpackhi_epi64(s1_9, s1_11);
    let s2_12 = _mm256_unpacklo_epi64(s1_12, s1_14);
    let s2_13 = _mm256_unpackhi_epi64(s1_12, s1_14);
    let s2_14 = _mm256_unpacklo_epi64(s1_13, s1_15);
    let s2_15 = _mm256_unpackhi_epi64(s1_13, s1_15);

    // Stage 3: 128-bit lane permute. Separate lo/hi lanes to form complete blocks.
    // Each block = 2 YMM registers (32 + 32 = 64 bytes).
    // Process block pairs (j, j+4) and XOR+store immediately to ease register pressure.
    let ptr = chunk.as_mut_ptr();
    // SAFETY: each `chunk` slice has exactly BLOCKS_PER_BATCH * BLOCK_SIZE = 512
    // bytes, so all 8 × 64-byte load/store pairs are in bounds.
    unsafe {
      // Blocks 0 and 4 — from s2[0], s2[4], s2[8], s2[12]
      xor_block_pair(ptr, 0, 4, s2_0, s2_4, s2_8, s2_12);
      // Blocks 1 and 5 — from s2[1], s2[5], s2[9], s2[13]
      xor_block_pair(ptr, 1, 5, s2_1, s2_5, s2_9, s2_13);
      // Blocks 2 and 6 — from s2[2], s2[6], s2[10], s2[14]
      xor_block_pair(ptr, 2, 6, s2_2, s2_6, s2_10, s2_14);
      // Blocks 3 and 7 — from s2[3], s2[7], s2[11], s2[15]
      xor_block_pair(ptr, 3, 7, s2_3, s2_7, s2_11, s2_15);
    }

    counter = counter.wrapping_add(BLOCKS_PER_BATCH as u32);
  }

  let remainder = batches.into_remainder();
  if !remainder.is_empty() {
    xor_keystream_portable(key, counter, nonce, remainder);
  }
}

/// Permute stage-2 results into two complete blocks (lo_idx and hi_idx) and
/// XOR+store them in-place. Each block is 64 bytes = 2 × YMM.
#[inline(always)]
unsafe fn xor_block_pair(
  buf: *mut u8,
  lo_idx: usize,
  hi_idx: usize,
  w03: __m256i,
  w47: __m256i,
  w811: __m256i,
  w1215: __m256i,
) {
  // SAFETY: caller guarantees `buf` points to a 512-byte chunk and indices < 8.
  unsafe {
    // lo_idx block: select lo 128-bit lanes (0x20 = [a_lo, b_lo])
    let blk_lo_a = _mm256_permute2x128_si256::<0x20>(w03, w47);
    let blk_lo_b = _mm256_permute2x128_si256::<0x20>(w811, w1215);
    // hi_idx block: select hi 128-bit lanes (0x31 = [a_hi, b_hi])
    let blk_hi_a = _mm256_permute2x128_si256::<0x31>(w03, w47);
    let blk_hi_b = _mm256_permute2x128_si256::<0x31>(w811, w1215);

    // XOR and store lo_idx block
    let p_lo = buf.add(lo_idx.strict_mul(BLOCK_SIZE));
    let pt0 = _mm256_loadu_si256(p_lo.cast());
    let pt1 = _mm256_loadu_si256(p_lo.add(32).cast());
    _mm256_storeu_si256(p_lo.cast(), _mm256_xor_si256(pt0, blk_lo_a));
    _mm256_storeu_si256(p_lo.add(32).cast(), _mm256_xor_si256(pt1, blk_lo_b));

    // XOR and store hi_idx block
    let p_hi = buf.add(hi_idx.strict_mul(BLOCK_SIZE));
    let pt2 = _mm256_loadu_si256(p_hi.cast());
    let pt3 = _mm256_loadu_si256(p_hi.add(32).cast());
    _mm256_storeu_si256(p_hi.cast(), _mm256_xor_si256(pt2, blk_hi_a));
    _mm256_storeu_si256(p_hi.add(32).cast(), _mm256_xor_si256(pt3, blk_hi_b));
  }
}

#[inline(always)]
fn quarter_round(a: &mut __m256i, b: &mut __m256i, c: &mut __m256i, d: &mut __m256i, rot16: __m256i, rot8: __m256i) {
  // SAFETY: this helper is only reached from the AVX2-enabled backend.
  unsafe {
    *a = _mm256_add_epi32(*a, *b);
    *d = _mm256_shuffle_epi8(_mm256_xor_si256(*d, *a), rot16);
    *c = _mm256_add_epi32(*c, *d);
    *b = rotl::<12, 20>(_mm256_xor_si256(*b, *c));
    *a = _mm256_add_epi32(*a, *b);
    *d = _mm256_shuffle_epi8(_mm256_xor_si256(*d, *a), rot8);
    *c = _mm256_add_epi32(*c, *d);
    *b = rotl::<7, 25>(_mm256_xor_si256(*b, *c));
  }
}

#[inline(always)]
fn rotl<const LEFT: i32, const RIGHT: i32>(value: __m256i) -> __m256i {
  const { assert!(LEFT + RIGHT == 32, "rotation amounts must sum to 32") };
  // SAFETY: only reached from the AVX2-enabled backend.
  unsafe { _mm256_or_si256(_mm256_slli_epi32(value, LEFT), _mm256_srli_epi32(value, RIGHT)) }
}
