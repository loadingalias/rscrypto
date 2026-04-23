use core::arch::aarch64::{
  uint32x4_t, vaddq_u32, vdupq_n_u32, veorq_u32, vld1q_u32, vorrq_u32, vshlq_n_u32, vshrq_n_u32, vst1q_u32,
};

use super::{BLOCK_SIZE, KEY_SIZE, NONCE_SIZE, load_u32_le, xor_keystream_portable};

const BLOCKS_PER_BATCH: usize = 4;

#[inline]
pub(super) fn xor_keystream(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
  // SAFETY: Backend selection guarantees NEON is available before this wrapper is chosen.
  unsafe { xor_keystream_impl(key, initial_counter, nonce, buffer) }
}

#[target_feature(enable = "neon")]
unsafe fn xor_keystream_impl(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
  let mut counter = initial_counter;
  let mut batches = buffer.chunks_exact_mut(BLOCK_SIZE * BLOCKS_PER_BATCH);
  for chunk in &mut batches {
    debug_assert!(counter.checked_add((BLOCKS_PER_BATCH - 1) as u32).is_some());

    let mut x0 = vdupq_n_u32(0x6170_7865);
    let mut x1 = vdupq_n_u32(0x3320_646e);
    let mut x2 = vdupq_n_u32(0x7962_2d32);
    let mut x3 = vdupq_n_u32(0x6b20_6574);
    let mut x4 = vdupq_n_u32(load_u32_le(&key[0..4]));
    let mut x5 = vdupq_n_u32(load_u32_le(&key[4..8]));
    let mut x6 = vdupq_n_u32(load_u32_le(&key[8..12]));
    let mut x7 = vdupq_n_u32(load_u32_le(&key[12..16]));
    let mut x8 = vdupq_n_u32(load_u32_le(&key[16..20]));
    let mut x9 = vdupq_n_u32(load_u32_le(&key[20..24]));
    let mut x10 = vdupq_n_u32(load_u32_le(&key[24..28]));
    let mut x11 = vdupq_n_u32(load_u32_le(&key[28..32]));
    let counter_words = [
      counter,
      counter.wrapping_add(1),
      counter.wrapping_add(2),
      counter.wrapping_add(3),
    ];
    // SAFETY: `counter_words` is a properly aligned local array with four initialized `u32` lanes.
    let mut x12 = unsafe { vld1q_u32(counter_words.as_ptr()) };
    let mut x13 = vdupq_n_u32(load_u32_le(&nonce[0..4]));
    let mut x14 = vdupq_n_u32(load_u32_le(&nonce[4..8]));
    let mut x15 = vdupq_n_u32(load_u32_le(&nonce[8..12]));

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

    let mut words = [[0u32; BLOCKS_PER_BATCH]; 16];
    // SAFETY: each destination is a valid four-lane `u32` array.
    unsafe {
      vst1q_u32(words[0].as_mut_ptr(), x0);
      vst1q_u32(words[1].as_mut_ptr(), x1);
      vst1q_u32(words[2].as_mut_ptr(), x2);
      vst1q_u32(words[3].as_mut_ptr(), x3);
      vst1q_u32(words[4].as_mut_ptr(), x4);
      vst1q_u32(words[5].as_mut_ptr(), x5);
      vst1q_u32(words[6].as_mut_ptr(), x6);
      vst1q_u32(words[7].as_mut_ptr(), x7);
      vst1q_u32(words[8].as_mut_ptr(), x8);
      vst1q_u32(words[9].as_mut_ptr(), x9);
      vst1q_u32(words[10].as_mut_ptr(), x10);
      vst1q_u32(words[11].as_mut_ptr(), x11);
      vst1q_u32(words[12].as_mut_ptr(), x12);
      vst1q_u32(words[13].as_mut_ptr(), x13);
      vst1q_u32(words[14].as_mut_ptr(), x14);
      vst1q_u32(words[15].as_mut_ptr(), x15);
    }

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

    counter = counter.wrapping_add(BLOCKS_PER_BATCH as u32);
  }

  let remainder = batches.into_remainder();
  if !remainder.is_empty() {
    xor_keystream_portable(key, counter, nonce, remainder);
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
  // SAFETY: this helper is only reached from the NEON-enabled backend and only uses fixed shift
  // amounts.
  unsafe { vorrq_u32(vshlq_n_u32(value, 16), vshrq_n_u32(value, 16)) }
}

#[inline(always)]
fn rotl12(value: uint32x4_t) -> uint32x4_t {
  // SAFETY: this helper is only reached from the NEON-enabled backend and only uses fixed shift
  // amounts.
  unsafe { vorrq_u32(vshlq_n_u32(value, 12), vshrq_n_u32(value, 20)) }
}

#[inline(always)]
fn rotl8(value: uint32x4_t) -> uint32x4_t {
  // SAFETY: this helper is only reached from the NEON-enabled backend and only uses fixed shift
  // amounts.
  unsafe { vorrq_u32(vshlq_n_u32(value, 8), vshrq_n_u32(value, 24)) }
}

#[inline(always)]
fn rotl7(value: uint32x4_t) -> uint32x4_t {
  // SAFETY: this helper is only reached from the NEON-enabled backend and only uses fixed shift
  // amounts.
  unsafe { vorrq_u32(vshlq_n_u32(value, 7), vshrq_n_u32(value, 25)) }
}
