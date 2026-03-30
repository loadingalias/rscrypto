#![allow(clippy::indexing_slicing)]

//! Portable ChaCha20 and HChaCha20 core.

use core::mem;

#[cfg(feature = "std")]
use crate::backend::cache::OnceCache;
use crate::{
  aead::targets::{AeadPrimitive, select_backend},
  platform::{Arch, Caps},
};

pub(crate) const KEY_SIZE: usize = 32;
pub(crate) const NONCE_SIZE: usize = 12;
pub(crate) const HCHACHA_NONCE_SIZE: usize = 16;
pub(crate) const BLOCK_SIZE: usize = 64;
pub(crate) const POLY1305_KEY_SIZE: usize = 32;

const CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

type XorKeystreamFn = fn(&[u8; KEY_SIZE], u32, &[u8; NONCE_SIZE], &mut [u8]);

#[cfg(feature = "std")]
static XOR_KEYSTREAM_DISPATCH: OnceCache<XorKeystreamFn> = OnceCache::new();

#[inline(always)]
fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
  state[a] = state[a].wrapping_add(state[b]);
  state[d] ^= state[a];
  state[d] = state[d].rotate_left(16);

  state[c] = state[c].wrapping_add(state[d]);
  state[b] ^= state[c];
  state[b] = state[b].rotate_left(12);

  state[a] = state[a].wrapping_add(state[b]);
  state[d] ^= state[a];
  state[d] = state[d].rotate_left(8);

  state[c] = state[c].wrapping_add(state[d]);
  state[b] ^= state[c];
  state[b] = state[b].rotate_left(7);
}

#[inline(always)]
fn rounds(state: &mut [u32; 16]) {
  let mut round = 0usize;
  while round < 10 {
    quarter_round(state, 0, 4, 8, 12);
    quarter_round(state, 1, 5, 9, 13);
    quarter_round(state, 2, 6, 10, 14);
    quarter_round(state, 3, 7, 11, 15);

    quarter_round(state, 0, 5, 10, 15);
    quarter_round(state, 1, 6, 11, 12);
    quarter_round(state, 2, 7, 8, 13);
    quarter_round(state, 3, 4, 9, 14);

    round = round.strict_add(1);
  }
}

#[inline]
fn load_u32_le(input: &[u8]) -> u32 {
  let mut bytes = [0u8; mem::size_of::<u32>()];
  bytes.copy_from_slice(input);
  u32::from_le_bytes(bytes)
}

#[inline]
fn init_state(key: &[u8; KEY_SIZE], counter: u32, nonce: &[u8; NONCE_SIZE]) -> [u32; 16] {
  [
    CONSTANTS[0],
    CONSTANTS[1],
    CONSTANTS[2],
    CONSTANTS[3],
    load_u32_le(&key[0..4]),
    load_u32_le(&key[4..8]),
    load_u32_le(&key[8..12]),
    load_u32_le(&key[12..16]),
    load_u32_le(&key[16..20]),
    load_u32_le(&key[20..24]),
    load_u32_le(&key[24..28]),
    load_u32_le(&key[28..32]),
    counter,
    load_u32_le(&nonce[0..4]),
    load_u32_le(&nonce[4..8]),
    load_u32_le(&nonce[8..12]),
  ]
}

#[inline]
fn init_hchacha_state(key: &[u8; KEY_SIZE], nonce: &[u8; HCHACHA_NONCE_SIZE]) -> [u32; 16] {
  [
    CONSTANTS[0],
    CONSTANTS[1],
    CONSTANTS[2],
    CONSTANTS[3],
    load_u32_le(&key[0..4]),
    load_u32_le(&key[4..8]),
    load_u32_le(&key[8..12]),
    load_u32_le(&key[12..16]),
    load_u32_le(&key[16..20]),
    load_u32_le(&key[20..24]),
    load_u32_le(&key[24..28]),
    load_u32_le(&key[28..32]),
    load_u32_le(&nonce[0..4]),
    load_u32_le(&nonce[4..8]),
    load_u32_le(&nonce[8..12]),
    load_u32_le(&nonce[12..16]),
  ]
}

/// Produce a single ChaCha20 block.
#[must_use]
pub(crate) fn block(key: &[u8; KEY_SIZE], counter: u32, nonce: &[u8; NONCE_SIZE]) -> [u8; BLOCK_SIZE] {
  let initial = init_state(key, counter, nonce);
  let mut state = initial;
  rounds(&mut state);

  let mut index = 0usize;
  while index < state.len() {
    state[index] = state[index].wrapping_add(initial[index]);
    index = index.strict_add(1);
  }

  let mut out = [0u8; BLOCK_SIZE];
  for (chunk, word) in out.chunks_exact_mut(4).zip(state) {
    chunk.copy_from_slice(&word.to_le_bytes());
  }
  out
}

/// Derive a one-time Poly1305 key from the ChaCha20 block with counter 0.
#[must_use]
pub(crate) fn poly1305_key_gen(key: &[u8; KEY_SIZE], nonce: &[u8; NONCE_SIZE]) -> [u8; POLY1305_KEY_SIZE] {
  let mut out = [0u8; POLY1305_KEY_SIZE];
  out.copy_from_slice(&block(key, 0, nonce)[..POLY1305_KEY_SIZE]);
  out
}

/// XOR the ChaCha20 keystream into `buffer` starting from `initial_counter`.
pub(crate) fn xor_keystream(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
  xor_keystream_resolved()(key, initial_counter, nonce, buffer);
}

#[inline]
fn xor_keystream_resolved() -> XorKeystreamFn {
  #[cfg(feature = "std")]
  {
    XOR_KEYSTREAM_DISPATCH.get_or_init(resolve_xor_keystream)
  }

  #[cfg(not(feature = "std"))]
  {
    resolve_xor_keystream()
  }
}

#[inline]
fn resolve_xor_keystream() -> XorKeystreamFn {
  match select_backend(AeadPrimitive::XChaCha20Poly1305, Arch::current(), current_caps()) {
    #[cfg(target_arch = "x86_64")]
    crate::aead::targets::AeadBackend::X86Avx2 => x86_avx2::xor_keystream,
    #[cfg(target_arch = "aarch64")]
    crate::aead::targets::AeadBackend::Aarch64Neon => aarch64_neon::xor_keystream,
    _ => xor_keystream_portable,
  }
}

#[inline]
fn current_caps() -> Caps {
  #[cfg(feature = "std")]
  {
    crate::platform::caps()
  }

  #[cfg(not(feature = "std"))]
  {
    crate::platform::caps_static()
  }
}

fn xor_keystream_portable(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
  let mut counter = initial_counter;
  for chunk in buffer.chunks_mut(BLOCK_SIZE) {
    let block = block(key, counter, nonce);
    for (dst, src) in chunk.iter_mut().zip(block.iter().copied()) {
      *dst ^= src;
    }
    counter = match counter.checked_add(1) {
      Some(next) => next,
      None => panic!("ChaCha20 block counter overflow"),
    };
  }
}

/// HChaCha20 subkey derivation for XChaCha20.
#[must_use]
pub(crate) fn hchacha20(key: &[u8; KEY_SIZE], nonce: &[u8; HCHACHA_NONCE_SIZE]) -> [u8; KEY_SIZE] {
  let mut state = init_hchacha_state(key, nonce);
  rounds(&mut state);

  let mut out = [0u8; KEY_SIZE];
  for (chunk, word) in out.chunks_exact_mut(4).zip([
    state[0], state[1], state[2], state[3], state[12], state[13], state[14], state[15],
  ]) {
    chunk.copy_from_slice(&word.to_le_bytes());
  }
  out
}

#[cfg(target_arch = "x86_64")]
mod x86_avx2 {
  use core::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_or_si256, _mm256_set1_epi32, _mm256_setr_epi32, _mm256_slli_epi32,
    _mm256_srli_epi32, _mm256_storeu_si256, _mm256_xor_si256,
  };

  use super::{BLOCK_SIZE, KEY_SIZE, NONCE_SIZE, load_u32_le, xor_keystream_portable};

  const BLOCKS_PER_BATCH: usize = 8;

  #[inline]
  pub(super) fn xor_keystream(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
    // SAFETY: Backend selection guarantees AVX2 is available before this wrapper is chosen.
    unsafe { xor_keystream_impl(key, initial_counter, nonce, buffer) }
  }

  #[target_feature(enable = "avx2")]
  unsafe fn xor_keystream_impl(
    key: &[u8; KEY_SIZE],
    initial_counter: u32,
    nonce: &[u8; NONCE_SIZE],
    buffer: &mut [u8],
  ) {
    let mut counter = initial_counter;
    let mut batches = buffer.chunks_exact_mut(BLOCK_SIZE * BLOCKS_PER_BATCH);
    for chunk in &mut batches {
      if counter.checked_add((BLOCKS_PER_BATCH - 1) as u32).is_none() {
        panic!("ChaCha20 block counter overflow");
      }

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

      let mut words = [[0u32; BLOCKS_PER_BATCH]; 16];
      // SAFETY: each destination is a valid eight-lane `u32` array and storeu supports unaligned
      // pointers.
      unsafe {
        _mm256_storeu_si256(words[0].as_mut_ptr() as *mut __m256i, x0);
        _mm256_storeu_si256(words[1].as_mut_ptr() as *mut __m256i, x1);
        _mm256_storeu_si256(words[2].as_mut_ptr() as *mut __m256i, x2);
        _mm256_storeu_si256(words[3].as_mut_ptr() as *mut __m256i, x3);
        _mm256_storeu_si256(words[4].as_mut_ptr() as *mut __m256i, x4);
        _mm256_storeu_si256(words[5].as_mut_ptr() as *mut __m256i, x5);
        _mm256_storeu_si256(words[6].as_mut_ptr() as *mut __m256i, x6);
        _mm256_storeu_si256(words[7].as_mut_ptr() as *mut __m256i, x7);
        _mm256_storeu_si256(words[8].as_mut_ptr() as *mut __m256i, x8);
        _mm256_storeu_si256(words[9].as_mut_ptr() as *mut __m256i, x9);
        _mm256_storeu_si256(words[10].as_mut_ptr() as *mut __m256i, x10);
        _mm256_storeu_si256(words[11].as_mut_ptr() as *mut __m256i, x11);
        _mm256_storeu_si256(words[12].as_mut_ptr() as *mut __m256i, x12);
        _mm256_storeu_si256(words[13].as_mut_ptr() as *mut __m256i, x13);
        _mm256_storeu_si256(words[14].as_mut_ptr() as *mut __m256i, x14);
        _mm256_storeu_si256(words[15].as_mut_ptr() as *mut __m256i, x15);
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
  fn quarter_round(a: &mut __m256i, b: &mut __m256i, c: &mut __m256i, d: &mut __m256i) {
    // SAFETY: this helper is only reached from the AVX2-enabled backend.
    unsafe {
      *a = _mm256_add_epi32(*a, *b);
      *d = rotl(_mm256_xor_si256(*d, *a), 16);
      *c = _mm256_add_epi32(*c, *d);
      *b = rotl(_mm256_xor_si256(*b, *c), 12);
      *a = _mm256_add_epi32(*a, *b);
      *d = rotl(_mm256_xor_si256(*d, *a), 8);
      *c = _mm256_add_epi32(*c, *d);
      *b = rotl(_mm256_xor_si256(*b, *c), 7);
    }
  }

  #[inline(always)]
  fn rotl(value: __m256i, bits: i32) -> __m256i {
    // SAFETY: this helper is only reached from the AVX2-enabled backend and only uses fixed shift
    // amounts supported by the intrinsic immediates.
    unsafe {
      match bits {
        16 => _mm256_or_si256(_mm256_slli_epi32(value, 16), _mm256_srli_epi32(value, 16)),
        12 => _mm256_or_si256(_mm256_slli_epi32(value, 12), _mm256_srli_epi32(value, 20)),
        8 => _mm256_or_si256(_mm256_slli_epi32(value, 8), _mm256_srli_epi32(value, 24)),
        7 => _mm256_or_si256(_mm256_slli_epi32(value, 7), _mm256_srli_epi32(value, 25)),
        _ => unreachable!("ChaCha20 only rotates by fixed amounts"),
      }
    }
  }
}

#[cfg(target_arch = "aarch64")]
mod aarch64_neon {
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
  unsafe fn xor_keystream_impl(
    key: &[u8; KEY_SIZE],
    initial_counter: u32,
    nonce: &[u8; NONCE_SIZE],
    buffer: &mut [u8],
  ) {
    let mut counter = initial_counter;
    let mut batches = buffer.chunks_exact_mut(BLOCK_SIZE * BLOCKS_PER_BATCH);
    for chunk in &mut batches {
      if counter.checked_add((BLOCKS_PER_BATCH - 1) as u32).is_none() {
        panic!("ChaCha20 block counter overflow");
      }

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
      *d = rotl(veorq_u32(*d, *a), 16);
      *c = vaddq_u32(*c, *d);
      *b = rotl(veorq_u32(*b, *c), 12);
      *a = vaddq_u32(*a, *b);
      *d = rotl(veorq_u32(*d, *a), 8);
      *c = vaddq_u32(*c, *d);
      *b = rotl(veorq_u32(*b, *c), 7);
    }
  }

  #[inline(always)]
  fn rotl(value: uint32x4_t, bits: i32) -> uint32x4_t {
    // SAFETY: this helper is only reached from the NEON-enabled backend and only uses fixed shift
    // amounts.
    unsafe {
      match bits {
        16 => vorrq_u32(vshlq_n_u32(value, 16), vshrq_n_u32(value, 16)),
        12 => vorrq_u32(vshlq_n_u32(value, 12), vshrq_n_u32(value, 20)),
        8 => vorrq_u32(vshlq_n_u32(value, 8), vshrq_n_u32(value, 24)),
        7 => vorrq_u32(vshlq_n_u32(value, 7), vshrq_n_u32(value, 25)),
        _ => unreachable!("ChaCha20 only rotates by fixed amounts"),
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::{KEY_SIZE, NONCE_SIZE, block, hchacha20, xor_keystream, xor_keystream_portable};
  #[cfg(target_arch = "aarch64")]
  use crate::platform::caps::aarch64;
  #[cfg(target_arch = "x86_64")]
  use crate::platform::caps::x86;

  #[test]
  fn chacha20_block_matches_rfc_8439_section_2_3_2() {
    let key = [
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12,
      0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    ];
    let nonce = [0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00];
    let expected = [
      0x10, 0xf1, 0xe7, 0xe4, 0xd1, 0x3b, 0x59, 0x15, 0x50, 0x0f, 0xdd, 0x1f, 0xa3, 0x20, 0x71, 0xc4, 0xc7, 0xd1, 0xf4,
      0xc7, 0x33, 0xc0, 0x68, 0x03, 0x04, 0x22, 0xaa, 0x9a, 0xc3, 0xd4, 0x6c, 0x4e, 0xd2, 0x82, 0x64, 0x46, 0x07, 0x9f,
      0xaa, 0x09, 0x14, 0xc2, 0xd7, 0x05, 0xd9, 0x8b, 0x02, 0xa2, 0xb5, 0x12, 0x9c, 0xd1, 0xde, 0x16, 0x4e, 0xb9, 0xcb,
      0xd0, 0x83, 0xe8, 0xa2, 0x50, 0x3c, 0x4e,
    ];

    assert_eq!(block(&key, 1, &nonce), expected);
  }

  #[test]
  fn hchacha20_matches_xchacha_draft_vector() {
    let key = [
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12,
      0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    ];
    let nonce = [
      0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00, 0x31, 0x41, 0x59, 0x27,
    ];
    let expected = [
      0x82, 0x41, 0x3b, 0x42, 0x27, 0xb2, 0x7b, 0xfe, 0xd3, 0x0e, 0x42, 0x50, 0x8a, 0x87, 0x7d, 0x73, 0xa0, 0xf9, 0xe4,
      0xd5, 0x8a, 0x74, 0xa8, 0x53, 0xc1, 0x2e, 0xc4, 0x13, 0x26, 0xd3, 0xec, 0xdc,
    ];

    assert_eq!(hchacha20(&key, &nonce), expected);
  }

  #[test]
  fn xor_keystream_is_symmetric() {
    let key = [0x42; KEY_SIZE];
    let nonce = [0x24; NONCE_SIZE];
    let plaintext = *b"chacha20 portable core";
    let mut ciphertext = plaintext;

    xor_keystream(&key, 1, &nonce, &mut ciphertext);
    assert_ne!(ciphertext, plaintext);

    xor_keystream(&key, 1, &nonce, &mut ciphertext);
    assert_eq!(ciphertext, plaintext);
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn avx2_backend_matches_portable_when_available() {
    if !crate::platform::caps().has(x86::AVX2) {
      return;
    }

    let key = [0x55; KEY_SIZE];
    let nonce = [0x33; NONCE_SIZE];
    for len in [0usize, 1, 63, 64, 65, 255, 256, 257, 511, 512, 513, 1024] {
      let mut portable = vec![0u8; len];
      let mut accelerated = vec![0u8; len];
      let mut index = 0usize;
      while index < len {
        let value = index.strict_mul(17).strict_add(9) as u8;
        portable[index] = value;
        accelerated[index] = value;
        index = index.strict_add(1);
      }

      xor_keystream_portable(&key, 7, &nonce, &mut portable);
      super::x86_avx2::xor_keystream(&key, 7, &nonce, &mut accelerated);
      assert_eq!(accelerated, portable);
    }
  }

  #[test]
  #[cfg(target_arch = "aarch64")]
  fn neon_backend_matches_portable() {
    if !crate::platform::caps().has(aarch64::NEON) {
      return;
    }

    let key = [0x66; KEY_SIZE];
    let nonce = [0x11; NONCE_SIZE];
    for len in [0usize, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 768] {
      let mut portable = vec![0u8; len];
      let mut accelerated = vec![0u8; len];
      let mut index = 0usize;
      while index < len {
        let value = index.strict_mul(29).strict_add(3) as u8;
        portable[index] = value;
        accelerated[index] = value;
        index = index.strict_add(1);
      }

      xor_keystream_portable(&key, 11, &nonce, &mut portable);
      super::aarch64_neon::xor_keystream(&key, 11, &nonce, &mut accelerated);
      assert_eq!(accelerated, portable);
    }
  }
}
