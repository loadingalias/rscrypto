use core::arch::aarch64::{
  uint32x4_t, vaddq_u32, vcombine_u32, vdupq_n_u32, veorq_u32, vget_high_u32, vget_low_u32, vld1q_u32,
  vreinterpretq_u16_u32, vreinterpretq_u32_u16, vrev32q_u16, vshrq_n_u32, vsliq_n_u32, vst1q_u32, vzip1q_u32,
  vzip2q_u32,
};

use super::{BLOCK_SIZE, KEY_SIZE, NONCE_SIZE, load_u32_le, xor_keystream_portable};

const BLOCKS_PER_BATCH: usize = 4;

#[inline]
pub(super) fn xor_keystream(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
  // SAFETY: Backend selection guarantees NEON is available before this wrapper is chosen because:
  // 1. Runtime dispatch selects this module only after `aarch64::NEON` is present.
  // 2. `xor_keystream_impl` is annotated with `#[target_feature(enable = "neon")]`.
  unsafe { xor_keystream_impl(key, initial_counter, nonce, buffer) }
}

#[target_feature(enable = "neon")]
unsafe fn xor_keystream_impl(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
  let c0 = vdupq_n_u32(0x6170_7865);
  let c1 = vdupq_n_u32(0x3320_646e);
  let c2 = vdupq_n_u32(0x7962_2d32);
  let c3 = vdupq_n_u32(0x6b20_6574);
  let k4 = vdupq_n_u32(load_u32_le(&key[0..4]));
  let k5 = vdupq_n_u32(load_u32_le(&key[4..8]));
  let k6 = vdupq_n_u32(load_u32_le(&key[8..12]));
  let k7 = vdupq_n_u32(load_u32_le(&key[12..16]));
  let k8 = vdupq_n_u32(load_u32_le(&key[16..20]));
  let k9 = vdupq_n_u32(load_u32_le(&key[20..24]));
  let k10 = vdupq_n_u32(load_u32_le(&key[24..28]));
  let k11 = vdupq_n_u32(load_u32_le(&key[28..32]));
  let n13 = vdupq_n_u32(load_u32_le(&nonce[0..4]));
  let n14 = vdupq_n_u32(load_u32_le(&nonce[4..8]));
  let n15 = vdupq_n_u32(load_u32_le(&nonce[8..12]));

  let mut counter = initial_counter;
  let mut double_batches = buffer.chunks_exact_mut(BLOCK_SIZE * BLOCKS_PER_BATCH * 2);
  for chunk in &mut double_batches {
    debug_assert!(counter.checked_add((BLOCKS_PER_BATCH * 2 - 1) as u32).is_some());

    let mut x0 = c0;
    let mut x1 = c1;
    let mut x2 = c2;
    let mut x3 = c3;
    let mut x4 = k4;
    let mut x5 = k5;
    let mut x6 = k6;
    let mut x7 = k7;
    let mut x8 = k8;
    let mut x9 = k9;
    let mut x10 = k10;
    let mut x11 = k11;
    let counter_words = [
      counter,
      counter.wrapping_add(1),
      counter.wrapping_add(2),
      counter.wrapping_add(3),
    ];
    // SAFETY: `counter_words` is a properly aligned local array with four initialized `u32` lanes.
    let mut x12 = unsafe { vld1q_u32(counter_words.as_ptr()) };
    let mut x13 = n13;
    let mut x14 = n14;
    let mut x15 = n15;

    let mut y0 = c0;
    let mut y1 = c1;
    let mut y2 = c2;
    let mut y3 = c3;
    let mut y4 = k4;
    let mut y5 = k5;
    let mut y6 = k6;
    let mut y7 = k7;
    let mut y8 = k8;
    let mut y9 = k9;
    let mut y10 = k10;
    let mut y11 = k11;
    let counter_words_hi = [
      counter.wrapping_add(4),
      counter.wrapping_add(5),
      counter.wrapping_add(6),
      counter.wrapping_add(7),
    ];
    // SAFETY: `counter_words_hi` is a properly aligned local array with four initialized `u32` lanes.
    let mut y12 = unsafe { vld1q_u32(counter_words_hi.as_ptr()) };
    let mut y13 = n13;
    let mut y14 = n14;
    let mut y15 = n15;

    let ox12 = x12;
    let oy12 = y12;

    macro_rules! double_round_pair {
      () => {{
        quarter_round(&mut x0, &mut x4, &mut x8, &mut x12);
        quarter_round(&mut y0, &mut y4, &mut y8, &mut y12);
        quarter_round(&mut x1, &mut x5, &mut x9, &mut x13);
        quarter_round(&mut y1, &mut y5, &mut y9, &mut y13);
        quarter_round(&mut x2, &mut x6, &mut x10, &mut x14);
        quarter_round(&mut y2, &mut y6, &mut y10, &mut y14);
        quarter_round(&mut x3, &mut x7, &mut x11, &mut x15);
        quarter_round(&mut y3, &mut y7, &mut y11, &mut y15);

        quarter_round(&mut x0, &mut x5, &mut x10, &mut x15);
        quarter_round(&mut y0, &mut y5, &mut y10, &mut y15);
        quarter_round(&mut x1, &mut x6, &mut x11, &mut x12);
        quarter_round(&mut y1, &mut y6, &mut y11, &mut y12);
        quarter_round(&mut x2, &mut x7, &mut x8, &mut x13);
        quarter_round(&mut y2, &mut y7, &mut y8, &mut y13);
        quarter_round(&mut x3, &mut x4, &mut x9, &mut x14);
        quarter_round(&mut y3, &mut y4, &mut y9, &mut y14);
      }};
    }

    double_round_pair!();
    double_round_pair!();
    double_round_pair!();
    double_round_pair!();
    double_round_pair!();
    double_round_pair!();
    double_round_pair!();
    double_round_pair!();
    double_round_pair!();
    double_round_pair!();

    x0 = vaddq_u32(x0, c0);
    x1 = vaddq_u32(x1, c1);
    x2 = vaddq_u32(x2, c2);
    x3 = vaddq_u32(x3, c3);
    x4 = vaddq_u32(x4, k4);
    x5 = vaddq_u32(x5, k5);
    x6 = vaddq_u32(x6, k6);
    x7 = vaddq_u32(x7, k7);
    x8 = vaddq_u32(x8, k8);
    x9 = vaddq_u32(x9, k9);
    x10 = vaddq_u32(x10, k10);
    x11 = vaddq_u32(x11, k11);
    x12 = vaddq_u32(x12, ox12);
    x13 = vaddq_u32(x13, n13);
    x14 = vaddq_u32(x14, n14);
    x15 = vaddq_u32(x15, n15);

    y0 = vaddq_u32(y0, c0);
    y1 = vaddq_u32(y1, c1);
    y2 = vaddq_u32(y2, c2);
    y3 = vaddq_u32(y3, c3);
    y4 = vaddq_u32(y4, k4);
    y5 = vaddq_u32(y5, k5);
    y6 = vaddq_u32(y6, k6);
    y7 = vaddq_u32(y7, k7);
    y8 = vaddq_u32(y8, k8);
    y9 = vaddq_u32(y9, k9);
    y10 = vaddq_u32(y10, k10);
    y11 = vaddq_u32(y11, k11);
    y12 = vaddq_u32(y12, oy12);
    y13 = vaddq_u32(y13, n13);
    y14 = vaddq_u32(y14, n14);
    y15 = vaddq_u32(y15, n15);

    let ptr = chunk.as_mut_ptr();
    // SAFETY: vector transpose and XOR stores because:
    // 1. `chunk` is exactly eight ChaCha20 blocks from `chunks_exact_mut`.
    // 2. The first four-block group starts at `ptr`; the second starts at `ptr + 256`.
    // 3. NEON is guaranteed by the enclosing `#[target_feature(enable = "neon")]`.
    unsafe {
      xor_store_word_group(ptr, 0, x0, x1, x2, x3);
      xor_store_word_group(ptr, 4, x4, x5, x6, x7);
      xor_store_word_group(ptr, 8, x8, x9, x10, x11);
      xor_store_word_group(ptr, 12, x12, x13, x14, x15);

      let hi = ptr.add(BLOCK_SIZE * BLOCKS_PER_BATCH);
      xor_store_word_group(hi, 0, y0, y1, y2, y3);
      xor_store_word_group(hi, 4, y4, y5, y6, y7);
      xor_store_word_group(hi, 8, y8, y9, y10, y11);
      xor_store_word_group(hi, 12, y12, y13, y14, y15);
    }

    counter = counter.wrapping_add((BLOCKS_PER_BATCH * 2) as u32);
  }

  let mut batches = double_batches
    .into_remainder()
    .chunks_exact_mut(BLOCK_SIZE * BLOCKS_PER_BATCH);
  for chunk in &mut batches {
    debug_assert!(counter.checked_add((BLOCKS_PER_BATCH - 1) as u32).is_some());

    let mut x0 = c0;
    let mut x1 = c1;
    let mut x2 = c2;
    let mut x3 = c3;
    let mut x4 = k4;
    let mut x5 = k5;
    let mut x6 = k6;
    let mut x7 = k7;
    let mut x8 = k8;
    let mut x9 = k9;
    let mut x10 = k10;
    let mut x11 = k11;
    let counter_words = [
      counter,
      counter.wrapping_add(1),
      counter.wrapping_add(2),
      counter.wrapping_add(3),
    ];
    // SAFETY: `counter_words` is a properly aligned local array with four initialized `u32` lanes.
    let mut x12 = unsafe { vld1q_u32(counter_words.as_ptr()) };
    let mut x13 = n13;
    let mut x14 = n14;
    let mut x15 = n15;

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

    macro_rules! double_round {
      () => {{
        quarter_round(&mut x0, &mut x4, &mut x8, &mut x12);
        quarter_round(&mut x1, &mut x5, &mut x9, &mut x13);
        quarter_round(&mut x2, &mut x6, &mut x10, &mut x14);
        quarter_round(&mut x3, &mut x7, &mut x11, &mut x15);

        quarter_round(&mut x0, &mut x5, &mut x10, &mut x15);
        quarter_round(&mut x1, &mut x6, &mut x11, &mut x12);
        quarter_round(&mut x2, &mut x7, &mut x8, &mut x13);
        quarter_round(&mut x3, &mut x4, &mut x9, &mut x14);
      }};
    }

    double_round!();
    double_round!();
    double_round!();
    double_round!();
    double_round!();
    double_round!();
    double_round!();
    double_round!();
    double_round!();
    double_round!();

    x0 = vaddq_u32(x0, o0);
    x1 = vaddq_u32(x1, o1);
    x2 = vaddq_u32(x2, o2);
    x3 = vaddq_u32(x3, o3);
    x4 = vaddq_u32(x4, o4);
    x5 = vaddq_u32(x5, o5);
    x6 = vaddq_u32(x6, o6);
    x7 = vaddq_u32(x7, o7);
    x8 = vaddq_u32(x8, o8);
    x9 = vaddq_u32(x9, o9);
    x10 = vaddq_u32(x10, o10);
    x11 = vaddq_u32(x11, o11);
    x12 = vaddq_u32(x12, o12);
    x13 = vaddq_u32(x13, o13);
    x14 = vaddq_u32(x14, o14);
    x15 = vaddq_u32(x15, o15);

    let ptr = chunk.as_mut_ptr();
    // SAFETY: vector transpose and XOR stores because:
    // 1. `chunk` is exactly `BLOCKS_PER_BATCH * BLOCK_SIZE` bytes from `chunks_exact_mut`.
    // 2. Each call stores four 16-byte word groups at offsets inside the 256-byte chunk.
    // 3. NEON is guaranteed by the enclosing `#[target_feature(enable = "neon")]`.
    unsafe {
      xor_store_word_group(ptr, 0, x0, x1, x2, x3);
      xor_store_word_group(ptr, 4, x4, x5, x6, x7);
      xor_store_word_group(ptr, 8, x8, x9, x10, x11);
      xor_store_word_group(ptr, 12, x12, x13, x14, x15);
    }

    counter = counter.wrapping_add(BLOCKS_PER_BATCH as u32);
  }

  let remainder = batches.into_remainder();
  if !remainder.is_empty() {
    xor_keystream_portable(key, counter, nonce, remainder);
  }
}

/// Transpose four word-major ChaCha vectors into four block-major 16-byte word groups.
#[inline(always)]
unsafe fn xor_store_word_group(
  chunk: *mut u8,
  word_start: usize,
  w0: uint32x4_t,
  w1: uint32x4_t,
  w2: uint32x4_t,
  w3: uint32x4_t,
) {
  // SAFETY: NEON transpose and stores because:
  // 1. Caller guarantees `chunk` points to a full four-block chunk.
  // 2. `word_start` is one of 0, 4, 8, or 12, so every 16-byte store is inside each 64-byte block.
  // 3. `vld1q_u32`/`vst1q_u32` allow unaligned buffer addresses.
  unsafe {
    let lo01 = vzip1q_u32(w0, w1);
    let hi01 = vzip2q_u32(w0, w1);
    let lo23 = vzip1q_u32(w2, w3);
    let hi23 = vzip2q_u32(w2, w3);

    let block0 = vcombine_u32(vget_low_u32(lo01), vget_low_u32(lo23));
    let block1 = vcombine_u32(vget_high_u32(lo01), vget_high_u32(lo23));
    let block2 = vcombine_u32(vget_low_u32(hi01), vget_low_u32(hi23));
    let block3 = vcombine_u32(vget_high_u32(hi01), vget_high_u32(hi23));

    xor_store_block_words(chunk, 0, word_start, block0);
    xor_store_block_words(chunk, 1, word_start, block1);
    xor_store_block_words(chunk, 2, word_start, block2);
    xor_store_block_words(chunk, 3, word_start, block3);
  }
}

#[inline(always)]
unsafe fn xor_store_block_words(chunk: *mut u8, block_index: usize, word_start: usize, keystream: uint32x4_t) {
  // SAFETY: in-place 16-byte XOR/store because:
  // 1. Caller guarantees `chunk` points to a full four-block chunk.
  // 2. `block_index < 4` and `word_start <= 12`, so `block_index * 64 + word_start * 4 + 16` stays
  //    within the 256-byte chunk.
  // 3. `vld1q_u32`/`vst1q_u32` support unaligned addresses and the pointer does not escape.
  unsafe {
    let offset = block_index.strict_mul(BLOCK_SIZE).strict_add(word_start.strict_mul(4));
    let ptr = chunk.add(offset).cast::<u32>();
    let plaintext = vld1q_u32(ptr);
    vst1q_u32(ptr, veorq_u32(plaintext, keystream));
  }
}

#[inline(always)]
fn quarter_round(a: &mut uint32x4_t, b: &mut uint32x4_t, c: &mut uint32x4_t, d: &mut uint32x4_t) {
  // SAFETY: this helper is only reached from the NEON-enabled backend.
  unsafe {
    *a = vaddq_u32(*a, *b);
    *d = rotl16(veorq_u32(*d, *a));
    *c = vaddq_u32(*c, *d);
    *b = rotl12(veorq_u32(*b, *c));
    *a = vaddq_u32(*a, *b);
    *d = rotl8(veorq_u32(*d, *a));
    *c = vaddq_u32(*c, *d);
    *b = rotl7(veorq_u32(*b, *c));
  }
}

#[inline(always)]
fn rotl16(value: uint32x4_t) -> uint32x4_t {
  // SAFETY: fixed-lane NEON rotate because:
  // 1. This helper is only reached from the NEON-enabled backend.
  // 2. Reversing 16-bit halves inside each 32-bit lane is exactly a 16-bit rotate.
  // 3. The reinterpret operations preserve bits and accept every `uint32x4_t` value.
  unsafe { vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(value))) }
}

#[inline(always)]
fn rotl12(value: uint32x4_t) -> uint32x4_t {
  // SAFETY: fixed-lane NEON rotate because:
  // 1. This helper is only reached from the NEON-enabled backend.
  // 2. `vsli` forms `(value >> 20) | (value << 12)` without an extra OR.
  // 3. The immediate shifts are compile-time constants in the valid 1..31 range.
  unsafe { vsliq_n_u32(vshrq_n_u32(value, 20), value, 12) }
}

#[inline(always)]
fn rotl8(value: uint32x4_t) -> uint32x4_t {
  // SAFETY: fixed-lane NEON rotate because:
  // 1. This helper is only reached from the NEON-enabled backend.
  // 2. `vsli` forms `(value >> 24) | (value << 8)` without an extra OR.
  // 3. The immediate shifts are compile-time constants in the valid 1..31 range.
  unsafe { vsliq_n_u32(vshrq_n_u32(value, 24), value, 8) }
}

#[inline(always)]
fn rotl7(value: uint32x4_t) -> uint32x4_t {
  // SAFETY: fixed-lane NEON rotate because:
  // 1. This helper is only reached from the NEON-enabled backend.
  // 2. `vsli` forms `(value >> 25) | (value << 7)` without an extra OR.
  // 3. The immediate shifts are compile-time constants in the valid 1..31 range.
  unsafe { vsliq_n_u32(vshrq_n_u32(value, 25), value, 7) }
}
