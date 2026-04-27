use core::arch::x86_64::{
  __m128i, _mm_add_epi32, _mm_loadu_si128, _mm_or_si128, _mm_set_epi8, _mm_set1_epi32, _mm_setr_epi32,
  _mm_shuffle_epi8, _mm_slli_epi32, _mm_srli_epi32, _mm_storeu_si128, _mm_unpackhi_epi32, _mm_unpackhi_epi64,
  _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm_xor_si128,
};

use super::{BLOCK_SIZE, KEY_SIZE, NONCE_SIZE, load_u32_le};

pub(super) const BLOCKS_PER_BATCH: usize = 4;

#[inline]
pub(super) unsafe fn xor_blocks(
  key: &[u8; KEY_SIZE],
  initial_counter: u32,
  nonce: &[u8; NONCE_SIZE],
  chunk: &mut [u8],
) {
  debug_assert_eq!(chunk.len(), BLOCK_SIZE * BLOCKS_PER_BATCH);
  // SAFETY: callers only reach this helper from x86 SIMD backends whose
  // dispatch gates guarantee AVX/SSSE3-capable hardware.
  unsafe { xor_blocks_impl(key, initial_counter, nonce, chunk) }
}

#[target_feature(enable = "ssse3,avx")]
unsafe fn xor_blocks_impl(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], chunk: &mut [u8]) {
  let rot16 = _mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
  let rot8 = _mm_set_epi8(14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3);

  let mut x0 = _mm_set1_epi32(0x6170_7865u32 as i32);
  let mut x1 = _mm_set1_epi32(0x3320_646eu32 as i32);
  let mut x2 = _mm_set1_epi32(0x7962_2d32u32 as i32);
  let mut x3 = _mm_set1_epi32(0x6b20_6574u32 as i32);
  let mut x4 = _mm_set1_epi32(load_u32_le(&key[0..4]) as i32);
  let mut x5 = _mm_set1_epi32(load_u32_le(&key[4..8]) as i32);
  let mut x6 = _mm_set1_epi32(load_u32_le(&key[8..12]) as i32);
  let mut x7 = _mm_set1_epi32(load_u32_le(&key[12..16]) as i32);
  let mut x8 = _mm_set1_epi32(load_u32_le(&key[16..20]) as i32);
  let mut x9 = _mm_set1_epi32(load_u32_le(&key[20..24]) as i32);
  let mut x10 = _mm_set1_epi32(load_u32_le(&key[24..28]) as i32);
  let mut x11 = _mm_set1_epi32(load_u32_le(&key[28..32]) as i32);
  let mut x12 = _mm_setr_epi32(
    initial_counter as i32,
    initial_counter.wrapping_add(1) as i32,
    initial_counter.wrapping_add(2) as i32,
    initial_counter.wrapping_add(3) as i32,
  );
  let mut x13 = _mm_set1_epi32(load_u32_le(&nonce[0..4]) as i32);
  let mut x14 = _mm_set1_epi32(load_u32_le(&nonce[4..8]) as i32);
  let mut x15 = _mm_set1_epi32(load_u32_le(&nonce[8..12]) as i32);

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

  x0 = _mm_add_epi32(x0, o0);
  x1 = _mm_add_epi32(x1, o1);
  x2 = _mm_add_epi32(x2, o2);
  x3 = _mm_add_epi32(x3, o3);
  x4 = _mm_add_epi32(x4, o4);
  x5 = _mm_add_epi32(x5, o5);
  x6 = _mm_add_epi32(x6, o6);
  x7 = _mm_add_epi32(x7, o7);
  x8 = _mm_add_epi32(x8, o8);
  x9 = _mm_add_epi32(x9, o9);
  x10 = _mm_add_epi32(x10, o10);
  x11 = _mm_add_epi32(x11, o11);
  x12 = _mm_add_epi32(x12, o12);
  x13 = _mm_add_epi32(x13, o13);
  x14 = _mm_add_epi32(x14, o14);
  x15 = _mm_add_epi32(x15, o15);

  let b03 = transpose_words(x0, x1, x2, x3);
  let b47 = transpose_words(x4, x5, x6, x7);
  let b811 = transpose_words(x8, x9, x10, x11);
  let b1215 = transpose_words(x12, x13, x14, x15);

  let ptr = chunk.as_mut_ptr();
  // SAFETY: `chunk` is exactly four 64-byte blocks.
  unsafe {
    xor_block(ptr, 0, b03[0], b47[0], b811[0], b1215[0]);
    xor_block(ptr, 1, b03[1], b47[1], b811[1], b1215[1]);
    xor_block(ptr, 2, b03[2], b47[2], b811[2], b1215[2]);
    xor_block(ptr, 3, b03[3], b47[3], b811[3], b1215[3]);
  }
}

#[inline(always)]
fn transpose_words(w0: __m128i, w1: __m128i, w2: __m128i, w3: __m128i) -> [__m128i; 4] {
  // SAFETY: all operations are lane-local integer shuffles.
  unsafe {
    let s0 = _mm_unpacklo_epi32(w0, w1);
    let s1 = _mm_unpackhi_epi32(w0, w1);
    let s2 = _mm_unpacklo_epi32(w2, w3);
    let s3 = _mm_unpackhi_epi32(w2, w3);
    [
      _mm_unpacklo_epi64(s0, s2),
      _mm_unpackhi_epi64(s0, s2),
      _mm_unpacklo_epi64(s1, s3),
      _mm_unpackhi_epi64(s1, s3),
    ]
  }
}

#[inline(always)]
unsafe fn xor_block(buf: *mut u8, idx: usize, w03: __m128i, w47: __m128i, w811: __m128i, w1215: __m128i) {
  // SAFETY: caller guarantees `buf` points to four full ChaCha20 blocks and
  // `idx < 4`, so all four 16-byte load/store pairs are in bounds.
  unsafe {
    let p = buf.add(idx.strict_mul(BLOCK_SIZE));
    xor_store(p, w03);
    xor_store(p.add(16), w47);
    xor_store(p.add(32), w811);
    xor_store(p.add(48), w1215);
  }
}

#[inline(always)]
unsafe fn xor_store(ptr: *mut u8, keystream: __m128i) {
  // SAFETY: caller guarantees `ptr..ptr+16` is in bounds for unaligned access.
  unsafe {
    let plaintext = _mm_loadu_si128(ptr.cast());
    _mm_storeu_si128(ptr.cast(), _mm_xor_si128(plaintext, keystream));
  }
}

#[inline(always)]
fn quarter_round(a: &mut __m128i, b: &mut __m128i, c: &mut __m128i, d: &mut __m128i, rot16: __m128i, rot8: __m128i) {
  // SAFETY: this helper is only reached from the SSSE3-enabled backend.
  unsafe {
    *a = _mm_add_epi32(*a, *b);
    *d = _mm_shuffle_epi8(_mm_xor_si128(*d, *a), rot16);
    *c = _mm_add_epi32(*c, *d);
    *b = rotl::<12, 20>(_mm_xor_si128(*b, *c));
    *a = _mm_add_epi32(*a, *b);
    *d = _mm_shuffle_epi8(_mm_xor_si128(*d, *a), rot8);
    *c = _mm_add_epi32(*c, *d);
    *b = rotl::<7, 25>(_mm_xor_si128(*b, *c));
  }
}

#[inline(always)]
fn rotl<const LEFT: i32, const RIGHT: i32>(value: __m128i) -> __m128i {
  const { assert!(LEFT + RIGHT == 32, "rotation amounts must sum to 32") };
  // SAFETY: only reached from the SSSE3-enabled backend.
  unsafe { _mm_or_si128(_mm_slli_epi32(value, LEFT), _mm_srli_epi32(value, RIGHT)) }
}
