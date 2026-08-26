use core::arch::wasm32::*;

use super::{BLOCK_SIZE, KEY_SIZE, NONCE_SIZE, load_u32_le, xor_keystream_portable};

const BLOCKS_PER_BATCH: usize = 4;

/// Generate and XOR a ChaCha20 stream with the wasm SIMD128 kernel.
///
/// # Safety
///
/// The caller must ensure that SIMD128 is available and that `buffer`'s 64-byte block count fits the counter range
/// starting at `initial_counter`.
#[inline]
pub(super) unsafe fn xor_keystream(
  key: &[u8; KEY_SIZE],
  initial_counter: u32,
  nonce: &[u8; NONCE_SIZE],
  buffer: &mut [u8],
) {
  // SAFETY: Production validates the counter range and selects this wrapper only when simd128 is available; direct
  // test and diagnostic callers establish the same conditions.
  unsafe { xor_keystream_impl(key, initial_counter, nonce, buffer) }
}

/// Generate and XOR a ChaCha20 stream in four-block SIMD128 batches.
///
/// # Safety
///
/// The caller must ensure that SIMD128 is available and that `buffer`'s 64-byte block count fits the counter range
/// starting at `initial_counter`.
#[target_feature(enable = "simd128")]
unsafe fn xor_keystream_impl(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
  let mut counter = initial_counter;
  let (batches, remainder) = buffer.as_chunks_mut::<{ BLOCK_SIZE * BLOCKS_PER_BATCH }>();
  for chunk in batches {
    debug_assert!(counter.checked_add(3).is_some());

    let mut x0 = u32x4_splat(0x6170_7865);
    let mut x1 = u32x4_splat(0x3320_646e);
    let mut x2 = u32x4_splat(0x7962_2d32);
    let mut x3 = u32x4_splat(0x6b20_6574);
    let mut x4 = u32x4_splat(load_u32_le(&key[0..4]));
    let mut x5 = u32x4_splat(load_u32_le(&key[4..8]));
    let mut x6 = u32x4_splat(load_u32_le(&key[8..12]));
    let mut x7 = u32x4_splat(load_u32_le(&key[12..16]));
    let mut x8 = u32x4_splat(load_u32_le(&key[16..20]));
    let mut x9 = u32x4_splat(load_u32_le(&key[20..24]));
    let mut x10 = u32x4_splat(load_u32_le(&key[24..28]));
    let mut x11 = u32x4_splat(load_u32_le(&key[28..32]));
    let mut x12 = u32x4(
      counter,
      counter.wrapping_add(1),
      counter.wrapping_add(2),
      counter.wrapping_add(3),
    );
    let mut x13 = u32x4_splat(load_u32_le(&nonce[0..4]));
    let mut x14 = u32x4_splat(load_u32_le(&nonce[4..8]));
    let mut x15 = u32x4_splat(load_u32_le(&nonce[8..12]));

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

    x0 = u32x4_add(x0, o0);
    x1 = u32x4_add(x1, o1);
    x2 = u32x4_add(x2, o2);
    x3 = u32x4_add(x3, o3);
    x4 = u32x4_add(x4, o4);
    x5 = u32x4_add(x5, o5);
    x6 = u32x4_add(x6, o6);
    x7 = u32x4_add(x7, o7);
    x8 = u32x4_add(x8, o8);
    x9 = u32x4_add(x9, o9);
    x10 = u32x4_add(x10, o10);
    x11 = u32x4_add(x11, o11);
    x12 = u32x4_add(x12, o12);
    x13 = u32x4_add(x13, o13);
    x14 = u32x4_add(x14, o14);
    x15 = u32x4_add(x15, o15);

    let words = [
      lanes(x0),
      lanes(x1),
      lanes(x2),
      lanes(x3),
      lanes(x4),
      lanes(x5),
      lanes(x6),
      lanes(x7),
      lanes(x8),
      lanes(x9),
      lanes(x10),
      lanes(x11),
      lanes(x12),
      lanes(x13),
      lanes(x14),
      lanes(x15),
    ];

    let mut block_index = 0usize;
    while block_index < BLOCKS_PER_BATCH {
      let mut word_index = 0usize;
      while word_index < 16 {
        let offset = block_index.strict_mul(BLOCK_SIZE).strict_add(word_index.strict_mul(4));
        let keystream = words[word_index][block_index].to_le_bytes();
        chunk[offset..offset.strict_add(4)]
          .iter_mut()
          .zip(keystream)
          .for_each(|(dst, src)| *dst ^= src);
        word_index = word_index.strict_add(1);
      }
      block_index = block_index.strict_add(1);
    }

    counter = counter.wrapping_add(4);
  }

  if !remainder.is_empty() {
    xor_keystream_portable(key, counter, nonce, remainder);
  }
}

#[inline(always)]
fn lanes(value: v128) -> [u32; BLOCKS_PER_BATCH] {
  [
    u32x4_extract_lane::<0>(value),
    u32x4_extract_lane::<1>(value),
    u32x4_extract_lane::<2>(value),
    u32x4_extract_lane::<3>(value),
  ]
}

#[inline(always)]
fn quarter_round(a: &mut v128, b: &mut v128, c: &mut v128, d: &mut v128) {
  *a = u32x4_add(*a, *b);
  *d = rotl(v128_xor(*d, *a), 16);
  *c = u32x4_add(*c, *d);
  *b = rotl(v128_xor(*b, *c), 12);
  *a = u32x4_add(*a, *b);
  *d = rotl(v128_xor(*d, *a), 8);
  *c = u32x4_add(*c, *d);
  *b = rotl(v128_xor(*b, *c), 7);
}

#[inline(always)]
fn rotl(value: v128, bits: u32) -> v128 {
  v128_or(u32x4_shl(value, bits), u32x4_shr(value, 32u32.wrapping_sub(bits)))
}
