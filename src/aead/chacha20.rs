#![allow(clippy::indexing_slicing)]

//! Portable ChaCha20 and HChaCha20 core.

use core::mem;
#[cfg(target_arch = "riscv64")]
use core::simd::u32x4;

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
static XCHACHA20POLY1305_XOR_KEYSTREAM_DISPATCH: OnceCache<XorKeystreamFn> = OnceCache::new();
#[cfg(feature = "std")]
static CHACHA20POLY1305_XOR_KEYSTREAM_DISPATCH: OnceCache<XorKeystreamFn> = OnceCache::new();

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

#[cfg(not(target_arch = "aarch64"))]
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
  #[cfg(target_arch = "aarch64")]
  // SAFETY: NEON is always available on aarch64 (mandatory since ARMv8).
  unsafe {
    aarch64_neon::poly1305_key_gen_neon(key, nonce)
  }

  #[cfg(not(target_arch = "aarch64"))]
  {
    let mut out = [0u8; POLY1305_KEY_SIZE];
    out.copy_from_slice(&block(key, 0, nonce)[..POLY1305_KEY_SIZE]);
    out
  }
}

/// XOR the ChaCha20 keystream into `buffer` starting from `initial_counter`.
pub(crate) fn xor_keystream(
  primitive: AeadPrimitive,
  key: &[u8; KEY_SIZE],
  initial_counter: u32,
  nonce: &[u8; NONCE_SIZE],
  buffer: &mut [u8],
) {
  xor_keystream_resolved(primitive)(key, initial_counter, nonce, buffer);
}

#[inline]
fn xor_keystream_resolved(primitive: AeadPrimitive) -> XorKeystreamFn {
  #[cfg(feature = "std")]
  {
    match primitive {
      AeadPrimitive::XChaCha20Poly1305 => {
        XCHACHA20POLY1305_XOR_KEYSTREAM_DISPATCH.get_or_init(|| resolve_xor_keystream(primitive))
      }
      AeadPrimitive::ChaCha20Poly1305 => {
        CHACHA20POLY1305_XOR_KEYSTREAM_DISPATCH.get_or_init(|| resolve_xor_keystream(primitive))
      }
      _ => resolve_xor_keystream(primitive),
    }
  }

  #[cfg(not(feature = "std"))]
  {
    resolve_xor_keystream(primitive)
  }
}

#[inline]
fn resolve_xor_keystream(primitive: AeadPrimitive) -> XorKeystreamFn {
  match select_backend(primitive, Arch::current(), current_caps()) {
    #[cfg(target_arch = "wasm32")]
    crate::aead::targets::AeadBackend::WasmSimd128 => wasm_simd128::xor_keystream,
    #[cfg(target_arch = "x86_64")]
    crate::aead::targets::AeadBackend::X86Avx512 => x86_avx512::xor_keystream,
    #[cfg(target_arch = "x86_64")]
    crate::aead::targets::AeadBackend::X86Avx2 => x86_avx2::xor_keystream,
    #[cfg(target_arch = "aarch64")]
    crate::aead::targets::AeadBackend::Aarch64Neon => aarch64_neon::xor_keystream,
    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    crate::aead::targets::AeadBackend::PowerVector => power_vsx::xor_keystream,
    #[cfg(target_arch = "s390x")]
    crate::aead::targets::AeadBackend::S390xVector => s390x_vector::xor_keystream,
    #[cfg(target_arch = "riscv64")]
    crate::aead::targets::AeadBackend::Riscv64Vector => riscv64_vector::xor_keystream,
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

#[cfg(target_arch = "riscv64")]
#[inline(always)]
fn simd_u32x4_rotl(value: u32x4, bits: u32) -> u32x4 {
  (value << u32x4::splat(bits)) | (value >> u32x4::splat(32u32.wrapping_sub(bits)))
}

#[cfg(target_arch = "riscv64")]
#[inline(always)]
fn simd_u32x4_quarter_round(a: &mut u32x4, b: &mut u32x4, c: &mut u32x4, d: &mut u32x4) {
  *a += *b;
  *d ^= *a;
  *d = simd_u32x4_rotl(*d, 16);

  *c += *d;
  *b ^= *c;
  *b = simd_u32x4_rotl(*b, 12);

  *a += *b;
  *d ^= *a;
  *d = simd_u32x4_rotl(*d, 8);

  *c += *d;
  *b ^= *c;
  *b = simd_u32x4_rotl(*b, 7);
}

#[cfg(target_arch = "riscv64")]
unsafe fn xor_keystream_u32x4_impl(
  key: &[u8; KEY_SIZE],
  initial_counter: u32,
  nonce: &[u8; NONCE_SIZE],
  buffer: &mut [u8],
) {
  const BLOCKS_PER_BATCH: usize = 4;

  let mut counter = initial_counter;
  let mut batches = buffer.chunks_exact_mut(BLOCK_SIZE * BLOCKS_PER_BATCH);
  for chunk in &mut batches {
    if counter.checked_add((BLOCKS_PER_BATCH - 1) as u32).is_none() {
      panic!("ChaCha20 block counter overflow");
    }

    let mut x0 = u32x4::splat(0x6170_7865);
    let mut x1 = u32x4::splat(0x3320_646e);
    let mut x2 = u32x4::splat(0x7962_2d32);
    let mut x3 = u32x4::splat(0x6b20_6574);
    let mut x4 = u32x4::splat(load_u32_le(&key[0..4]));
    let mut x5 = u32x4::splat(load_u32_le(&key[4..8]));
    let mut x6 = u32x4::splat(load_u32_le(&key[8..12]));
    let mut x7 = u32x4::splat(load_u32_le(&key[12..16]));
    let mut x8 = u32x4::splat(load_u32_le(&key[16..20]));
    let mut x9 = u32x4::splat(load_u32_le(&key[20..24]));
    let mut x10 = u32x4::splat(load_u32_le(&key[24..28]));
    let mut x11 = u32x4::splat(load_u32_le(&key[28..32]));
    let mut x12 = u32x4::from_array([
      counter,
      counter.wrapping_add(1),
      counter.wrapping_add(2),
      counter.wrapping_add(3),
    ]);
    let mut x13 = u32x4::splat(load_u32_le(&nonce[0..4]));
    let mut x14 = u32x4::splat(load_u32_le(&nonce[4..8]));
    let mut x15 = u32x4::splat(load_u32_le(&nonce[8..12]));

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
      simd_u32x4_quarter_round(&mut x0, &mut x4, &mut x8, &mut x12);
      simd_u32x4_quarter_round(&mut x1, &mut x5, &mut x9, &mut x13);
      simd_u32x4_quarter_round(&mut x2, &mut x6, &mut x10, &mut x14);
      simd_u32x4_quarter_round(&mut x3, &mut x7, &mut x11, &mut x15);

      simd_u32x4_quarter_round(&mut x0, &mut x5, &mut x10, &mut x15);
      simd_u32x4_quarter_round(&mut x1, &mut x6, &mut x11, &mut x12);
      simd_u32x4_quarter_round(&mut x2, &mut x7, &mut x8, &mut x13);
      simd_u32x4_quarter_round(&mut x3, &mut x4, &mut x9, &mut x14);

      round = round.strict_add(1);
    }

    x0 += o0;
    x1 += o1;
    x2 += o2;
    x3 += o3;
    x4 += o4;
    x5 += o5;
    x6 += o6;
    x7 += o7;
    x8 += o8;
    x9 += o9;
    x10 += o10;
    x11 += o11;
    x12 += o12;
    x13 += o13;
    x14 += o14;
    x15 += o15;

    let words = [
      x0.to_array(),
      x1.to_array(),
      x2.to_array(),
      x3.to_array(),
      x4.to_array(),
      x5.to_array(),
      x6.to_array(),
      x7.to_array(),
      x8.to_array(),
      x9.to_array(),
      x10.to_array(),
      x11.to_array(),
      x12.to_array(),
      x13.to_array(),
      x14.to_array(),
      x15.to_array(),
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

    counter = counter.wrapping_add(BLOCKS_PER_BATCH as u32);
  }

  let remainder = batches.into_remainder();
  if !remainder.is_empty() {
    xor_keystream_portable(key, counter, nonce, remainder);
  }
}

/// HChaCha20 subkey derivation for XChaCha20.
#[must_use]
pub(crate) fn hchacha20(key: &[u8; KEY_SIZE], nonce: &[u8; HCHACHA_NONCE_SIZE]) -> [u8; KEY_SIZE] {
  #[cfg(target_arch = "aarch64")]
  // SAFETY: NEON is always available on aarch64 (mandatory since ARMv8).
  unsafe {
    aarch64_neon::hchacha20_neon(key, nonce)
  }

  #[cfg(not(target_arch = "aarch64"))]
  {
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
}

#[cfg(target_arch = "x86_64")]
mod x86_avx512 {
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
}

#[cfg(target_arch = "x86_64")]
mod x86_avx2 {
  use core::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_loadu_si256, _mm256_or_si256, _mm256_permute2x128_si256, _mm256_set_epi8,
    _mm256_set1_epi32, _mm256_setr_epi32, _mm256_shuffle_epi8, _mm256_slli_epi32, _mm256_srli_epi32,
    _mm256_storeu_si256, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
    _mm256_xor_si256,
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
}

#[cfg(target_arch = "aarch64")]
mod aarch64_neon {
  use core::arch::aarch64::{
    uint32x4_t, vaddq_u32, vdupq_n_u32, veorq_u32, vextq_u32, vld1q_u32, vorrq_u32, vshlq_n_u32, vshrq_n_u32, vst1q_u32,
  };

  use super::{BLOCK_SIZE, CONSTANTS, HCHACHA_NONCE_SIZE, KEY_SIZE, NONCE_SIZE, POLY1305_KEY_SIZE, load_u32_le};

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

    // Use NEON single-block for remainder instead of portable scalar.
    let remainder = batches.into_remainder();
    if !remainder.is_empty() {
      for chunk in remainder.chunks_mut(BLOCK_SIZE) {
        // SAFETY: NEON availability guaranteed by the backend dispatch that selected this path.
        let blk = unsafe { block_neon(key, counter, nonce) };
        for (dst, src) in chunk.iter_mut().zip(blk.iter().copied()) {
          *dst ^= src;
        }
        counter = match counter.checked_add(1) {
          Some(next) => next,
          None => panic!("ChaCha20 block counter overflow"),
        };
      }
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

  // ─── Single-block NEON core ───────────────────────────────────────────
  //
  // State packed as 4 × uint32x4_t (row-major):
  //   v0 = [s0, s1, s2, s3]    (constants)
  //   v1 = [s4, s5, s6, s7]    (key lo)
  //   v2 = [s8, s9, s10, s11]  (key hi)
  //   v3 = [s12, s13, s14, s15] (counter + nonce)
  //
  // Column round operates on columns directly.
  // Diagonal round rotates v1 left by 1, v2 by 2, v3 by 3, then reverses.

  /// 10 double-rounds on a row-major single-block NEON state.
  #[inline(always)]
  fn chacha20_rounds_single(v0: &mut uint32x4_t, v1: &mut uint32x4_t, v2: &mut uint32x4_t, v3: &mut uint32x4_t) {
    let mut round = 0usize;
    while round < 10 {
      // Column round: QR(0,4,8,12), QR(1,5,9,13), QR(2,6,10,14), QR(3,7,11,15)
      quarter_round(v0, v1, v2, v3);

      // Diagonal round: rotate rows to align diagonals into columns
      // SAFETY: vextq_u32 is NEON; callers guarantee NEON is available.
      unsafe {
        *v1 = vextq_u32(*v1, *v1, 1); // [s5, s6, s7, s4]
        *v2 = vextq_u32(*v2, *v2, 2); // [s10, s11, s8, s9]
        *v3 = vextq_u32(*v3, *v3, 3); // [s15, s12, s13, s14]
      }

      quarter_round(v0, v1, v2, v3);

      // Restore original lane order
      // SAFETY: same as above.
      unsafe {
        *v1 = vextq_u32(*v1, *v1, 3); // rotate right 1
        *v2 = vextq_u32(*v2, *v2, 2); // rotate right 2
        *v3 = vextq_u32(*v3, *v3, 1); // rotate right 3
      }

      round = round.strict_add(1);
    }
  }

  /// Load the initial ChaCha20 state into 4 NEON vectors.
  #[inline(always)]
  fn load_state(key: &[u8; KEY_SIZE], w12: u32, w13: u32, w14: u32, w15: u32) -> [uint32x4_t; 4] {
    // SAFETY: vld1q_u32 is NEON; callers guarantee NEON is available.
    unsafe {
      [
        vld1q_u32([CONSTANTS[0], CONSTANTS[1], CONSTANTS[2], CONSTANTS[3]].as_ptr()),
        vld1q_u32(
          [
            load_u32_le(&key[0..4]),
            load_u32_le(&key[4..8]),
            load_u32_le(&key[8..12]),
            load_u32_le(&key[12..16]),
          ]
          .as_ptr(),
        ),
        vld1q_u32(
          [
            load_u32_le(&key[16..20]),
            load_u32_le(&key[20..24]),
            load_u32_le(&key[24..28]),
            load_u32_le(&key[28..32]),
          ]
          .as_ptr(),
        ),
        vld1q_u32([w12, w13, w14, w15].as_ptr()),
      ]
    }
  }

  /// Serialize 4 state vectors to LE bytes.
  #[inline(always)]
  fn store_state(v: &[uint32x4_t; 4], out: &mut [u8; BLOCK_SIZE]) {
    let mut words = [0u32; 16];
    // SAFETY: vst1q_u32 is NEON; callers guarantee NEON is available.
    unsafe {
      vst1q_u32(words[0..].as_mut_ptr(), v[0]);
      vst1q_u32(words[4..].as_mut_ptr(), v[1]);
      vst1q_u32(words[8..].as_mut_ptr(), v[2]);
      vst1q_u32(words[12..].as_mut_ptr(), v[3]);
    }
    let mut i = 0usize;
    while i < 16 {
      let off = i.strict_mul(4);
      out[off..off.strict_add(4)].copy_from_slice(&words[i].to_le_bytes());
      i = i.strict_add(1);
    }
  }

  /// Generate a single ChaCha20 block using NEON.
  #[target_feature(enable = "neon")]
  pub(super) unsafe fn block_neon(
    key: &[u8; KEY_SIZE],
    counter: u32,
    nonce: &[u8; super::NONCE_SIZE],
  ) -> [u8; BLOCK_SIZE] {
    let [mut v0, mut v1, mut v2, mut v3] = load_state(
      key,
      counter,
      load_u32_le(&nonce[0..4]),
      load_u32_le(&nonce[4..8]),
      load_u32_le(&nonce[8..12]),
    );

    let (o0, o1, o2, o3) = (v0, v1, v2, v3);
    chacha20_rounds_single(&mut v0, &mut v1, &mut v2, &mut v3);

    // Add original state (vaddq_u32 is safe under target_feature gate).
    v0 = vaddq_u32(v0, o0);
    v1 = vaddq_u32(v1, o1);
    v2 = vaddq_u32(v2, o2);
    v3 = vaddq_u32(v3, o3);

    let mut out = [0u8; BLOCK_SIZE];
    store_state(&[v0, v1, v2, v3], &mut out);
    out
  }

  /// HChaCha20 using NEON: subkey derivation for XChaCha20.
  #[target_feature(enable = "neon")]
  pub(super) unsafe fn hchacha20_neon(key: &[u8; KEY_SIZE], nonce: &[u8; HCHACHA_NONCE_SIZE]) -> [u8; KEY_SIZE] {
    let [mut v0, mut v1, mut v2, mut v3] = load_state(
      key,
      load_u32_le(&nonce[0..4]),
      load_u32_le(&nonce[4..8]),
      load_u32_le(&nonce[8..12]),
      load_u32_le(&nonce[12..16]),
    );

    // HChaCha20: rounds only, no add-back of initial state
    chacha20_rounds_single(&mut v0, &mut v1, &mut v2, &mut v3);

    // Extract words 0-3 (from v0) and 12-15 (from v3)
    let mut out = [0u8; KEY_SIZE];
    let mut words = [0u32; 8];
    // SAFETY: vst1q_u32 is NEON; target_feature gate guarantees availability.
    unsafe {
      vst1q_u32(words[0..].as_mut_ptr(), v0);
      vst1q_u32(words[4..].as_mut_ptr(), v3);
    }
    let mut i = 0usize;
    while i < 8 {
      let off = i.strict_mul(4);
      out[off..off.strict_add(4)].copy_from_slice(&words[i].to_le_bytes());
      i = i.strict_add(1);
    }
    out
  }

  /// Derive a one-time Poly1305 key using NEON single-block.
  #[target_feature(enable = "neon")]
  pub(super) unsafe fn poly1305_key_gen_neon(
    key: &[u8; KEY_SIZE],
    nonce: &[u8; super::NONCE_SIZE],
  ) -> [u8; POLY1305_KEY_SIZE] {
    // SAFETY: block_neon requires NEON, which target_feature gate provides.
    let blk = unsafe { block_neon(key, 0, nonce) };
    let mut out = [0u8; POLY1305_KEY_SIZE];
    out.copy_from_slice(&blk[..POLY1305_KEY_SIZE]);
    out
  }
}

#[cfg(target_arch = "wasm32")]
mod wasm_simd128 {
  use core::arch::wasm32::*;

  use super::{BLOCK_SIZE, KEY_SIZE, NONCE_SIZE, load_u32_le, xor_keystream_portable};

  const BLOCKS_PER_BATCH: usize = 4;

  #[inline]
  pub(super) fn xor_keystream(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
    // SAFETY: Backend selection guarantees simd128 is available before this wrapper is chosen.
    unsafe { xor_keystream_impl(key, initial_counter, nonce, buffer) }
  }

  #[target_feature(enable = "simd128")]
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

      let mut words = [[0u32; BLOCKS_PER_BATCH]; 16];
      // SAFETY: each destination is a valid four-lane `u32` array for one unaligned `v128` store.
      unsafe {
        v128_store(words[0].as_mut_ptr() as *mut v128, x0);
        v128_store(words[1].as_mut_ptr() as *mut v128, x1);
        v128_store(words[2].as_mut_ptr() as *mut v128, x2);
        v128_store(words[3].as_mut_ptr() as *mut v128, x3);
        v128_store(words[4].as_mut_ptr() as *mut v128, x4);
        v128_store(words[5].as_mut_ptr() as *mut v128, x5);
        v128_store(words[6].as_mut_ptr() as *mut v128, x6);
        v128_store(words[7].as_mut_ptr() as *mut v128, x7);
        v128_store(words[8].as_mut_ptr() as *mut v128, x8);
        v128_store(words[9].as_mut_ptr() as *mut v128, x9);
        v128_store(words[10].as_mut_ptr() as *mut v128, x10);
        v128_store(words[11].as_mut_ptr() as *mut v128, x11);
        v128_store(words[12].as_mut_ptr() as *mut v128, x12);
        v128_store(words[13].as_mut_ptr() as *mut v128, x13);
        v128_store(words[14].as_mut_ptr() as *mut v128, x14);
        v128_store(words[15].as_mut_ptr() as *mut v128, x15);
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
}

#[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
mod power_vsx {
  use core::simd::i64x2;

  use super::{BLOCK_SIZE, KEY_SIZE, NONCE_SIZE, load_u32_le, xor_keystream_portable};

  const BLOCKS_PER_BATCH: usize = 4;

  #[inline]
  pub(super) fn xor_keystream(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
    // SAFETY: Backend selection guarantees POWER vector support before this wrapper is chosen.
    unsafe { xor_keystream_impl(key, initial_counter, nonce, buffer) }
  }

  /// Precomputed `vrlw` shift vectors for the four ChaCha20 rotations.
  struct RotShifts {
    rot16: i64x2,
    rot12: i64x2,
    rot8: i64x2,
    rot7: i64x2,
  }

  impl RotShifts {
    fn new() -> Self {
      Self {
        rot16: splat(16),
        rot12: splat(12),
        rot8: splat(8),
        rot7: splat(7),
      }
    }
  }

  #[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
  unsafe fn xor_keystream_impl(
    key: &[u8; KEY_SIZE],
    initial_counter: u32,
    nonce: &[u8; NONCE_SIZE],
    buffer: &mut [u8],
  ) {
    let rot = RotShifts::new();

    let mut counter = initial_counter;
    let mut batches = buffer.chunks_exact_mut(BLOCK_SIZE * BLOCKS_PER_BATCH);
    for chunk in &mut batches {
      if counter.checked_add((BLOCKS_PER_BATCH - 1) as u32).is_none() {
        panic!("ChaCha20 block counter overflow");
      }

      let mut x0 = splat(0x6170_7865);
      let mut x1 = splat(0x3320_646e);
      let mut x2 = splat(0x7962_2d32);
      let mut x3 = splat(0x6b20_6574);
      let mut x4 = splat(load_u32_le(&key[0..4]));
      let mut x5 = splat(load_u32_le(&key[4..8]));
      let mut x6 = splat(load_u32_le(&key[8..12]));
      let mut x7 = splat(load_u32_le(&key[12..16]));
      let mut x8 = splat(load_u32_le(&key[16..20]));
      let mut x9 = splat(load_u32_le(&key[20..24]));
      let mut x10 = splat(load_u32_le(&key[24..28]));
      let mut x11 = splat(load_u32_le(&key[28..32]));
      let mut x12 = from_u32x4([
        counter,
        counter.wrapping_add(1),
        counter.wrapping_add(2),
        counter.wrapping_add(3),
      ]);
      let mut x13 = splat(load_u32_le(&nonce[0..4]));
      let mut x14 = splat(load_u32_le(&nonce[4..8]));
      let mut x15 = splat(load_u32_le(&nonce[8..12]));

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
        // SAFETY: target_feature ensures VSX availability.
        unsafe {
          quarter_round(&mut x0, &mut x4, &mut x8, &mut x12, &rot);
          quarter_round(&mut x1, &mut x5, &mut x9, &mut x13, &rot);
          quarter_round(&mut x2, &mut x6, &mut x10, &mut x14, &rot);
          quarter_round(&mut x3, &mut x7, &mut x11, &mut x15, &rot);

          quarter_round(&mut x0, &mut x5, &mut x10, &mut x15, &rot);
          quarter_round(&mut x1, &mut x6, &mut x11, &mut x12, &rot);
          quarter_round(&mut x2, &mut x7, &mut x8, &mut x13, &rot);
          quarter_round(&mut x3, &mut x4, &mut x9, &mut x14, &rot);
        }

        round = round.strict_add(1);
      }

      // SAFETY: target_feature ensures VSX availability.
      unsafe {
        x0 = vadduwm(x0, o0);
        x1 = vadduwm(x1, o1);
        x2 = vadduwm(x2, o2);
        x3 = vadduwm(x3, o3);
        x4 = vadduwm(x4, o4);
        x5 = vadduwm(x5, o5);
        x6 = vadduwm(x6, o6);
        x7 = vadduwm(x7, o7);
        x8 = vadduwm(x8, o8);
        x9 = vadduwm(x9, o9);
        x10 = vadduwm(x10, o10);
        x11 = vadduwm(x11, o11);
        x12 = vadduwm(x12, o12);
        x13 = vadduwm(x13, o13);
        x14 = vadduwm(x14, o14);
        x15 = vadduwm(x15, o15);
      }

      let words = [
        to_u32x4(x0),
        to_u32x4(x1),
        to_u32x4(x2),
        to_u32x4(x3),
        to_u32x4(x4),
        to_u32x4(x5),
        to_u32x4(x6),
        to_u32x4(x7),
        to_u32x4(x8),
        to_u32x4(x9),
        to_u32x4(x10),
        to_u32x4(x11),
        to_u32x4(x12),
        to_u32x4(x13),
        to_u32x4(x14),
        to_u32x4(x15),
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

      counter = counter.wrapping_add(BLOCKS_PER_BATCH as u32);
    }

    let remainder = batches.into_remainder();
    if !remainder.is_empty() {
      xor_keystream_portable(key, counter, nonce, remainder);
    }
  }

  #[inline(always)]
  fn from_u32x4(words: [u32; 4]) -> i64x2 {
    // SAFETY: [u32; 4] and i64x2 are both 16 bytes with no alignment mismatch.
    unsafe { core::mem::transmute(words) }
  }

  #[inline(always)]
  fn to_u32x4(v: i64x2) -> [u32; 4] {
    // SAFETY: i64x2 and [u32; 4] are both 16 bytes.
    unsafe { core::mem::transmute(v) }
  }

  #[inline(always)]
  fn splat(val: u32) -> i64x2 {
    from_u32x4([val; 4])
  }

  /// Vector add unsigned word modulo: `vadduwm`.
  #[inline(always)]
  unsafe fn vadduwm(a: i64x2, b: i64x2) -> i64x2 {
    let out: i64x2;
    // SAFETY: POWER8+ VSX available via enclosing target_feature.
    unsafe {
      core::arch::asm!(
        "vadduwm {out}, {a}, {b}",
        out = lateout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(nomem, nostack, pure)
      );
    }
    out
  }

  /// Vector XOR: `vxor`.
  #[inline(always)]
  unsafe fn vxor(a: i64x2, b: i64x2) -> i64x2 {
    let out: i64x2;
    // SAFETY: POWER8+ VSX available via enclosing target_feature.
    unsafe {
      core::arch::asm!(
        "vxor {out}, {a}, {b}",
        out = lateout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(nomem, nostack, pure)
      );
    }
    out
  }

  /// Vector rotate left word: `vrlw`.
  #[inline(always)]
  unsafe fn vrlw(value: i64x2, shift: i64x2) -> i64x2 {
    let out: i64x2;
    // SAFETY: POWER8+ VSX available via enclosing target_feature.
    unsafe {
      core::arch::asm!(
        "vrlw {out}, {val}, {shift}",
        out = lateout(vreg) out,
        val = in(vreg) value,
        shift = in(vreg) shift,
        options(nomem, nostack, pure)
      );
    }
    out
  }

  #[inline(always)]
  unsafe fn quarter_round(a: &mut i64x2, b: &mut i64x2, c: &mut i64x2, d: &mut i64x2, rot: &RotShifts) {
    // SAFETY: POWER8+ VSX available via enclosing target_feature.
    unsafe {
      *a = vadduwm(*a, *b);
      *d = vrlw(vxor(*d, *a), rot.rot16);

      *c = vadduwm(*c, *d);
      *b = vrlw(vxor(*b, *c), rot.rot12);

      *a = vadduwm(*a, *b);
      *d = vrlw(vxor(*d, *a), rot.rot8);

      *c = vadduwm(*c, *d);
      *b = vrlw(vxor(*b, *c), rot.rot7);
    }
  }
}

#[cfg(target_arch = "s390x")]
mod s390x_vector {
  use core::simd::i64x2;

  use super::{BLOCK_SIZE, KEY_SIZE, NONCE_SIZE, load_u32_le, xor_keystream_portable};

  const BLOCKS_PER_BATCH: usize = 4;

  #[inline]
  pub(super) fn xor_keystream(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
    // SAFETY: Backend selection guarantees the z/Vector facility before this wrapper is chosen.
    unsafe { xor_keystream_impl(key, initial_counter, nonce, buffer) }
  }

  #[target_feature(enable = "vector")]
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

      let mut x0 = splat(0x6170_7865);
      let mut x1 = splat(0x3320_646e);
      let mut x2 = splat(0x7962_2d32);
      let mut x3 = splat(0x6b20_6574);
      let mut x4 = splat(load_u32_le(&key[0..4]));
      let mut x5 = splat(load_u32_le(&key[4..8]));
      let mut x6 = splat(load_u32_le(&key[8..12]));
      let mut x7 = splat(load_u32_le(&key[12..16]));
      let mut x8 = splat(load_u32_le(&key[16..20]));
      let mut x9 = splat(load_u32_le(&key[20..24]));
      let mut x10 = splat(load_u32_le(&key[24..28]));
      let mut x11 = splat(load_u32_le(&key[28..32]));
      let mut x12 = from_u32x4([
        counter,
        counter.wrapping_add(1),
        counter.wrapping_add(2),
        counter.wrapping_add(3),
      ]);
      let mut x13 = splat(load_u32_le(&nonce[0..4]));
      let mut x14 = splat(load_u32_le(&nonce[4..8]));
      let mut x15 = splat(load_u32_le(&nonce[8..12]));

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
        // SAFETY: target_feature ensures z/Vector availability.
        unsafe {
          quarter_round(&mut x0, &mut x4, &mut x8, &mut x12);
          quarter_round(&mut x1, &mut x5, &mut x9, &mut x13);
          quarter_round(&mut x2, &mut x6, &mut x10, &mut x14);
          quarter_round(&mut x3, &mut x7, &mut x11, &mut x15);

          quarter_round(&mut x0, &mut x5, &mut x10, &mut x15);
          quarter_round(&mut x1, &mut x6, &mut x11, &mut x12);
          quarter_round(&mut x2, &mut x7, &mut x8, &mut x13);
          quarter_round(&mut x3, &mut x4, &mut x9, &mut x14);
        }

        round = round.strict_add(1);
      }

      // SAFETY: target_feature ensures z/Vector availability.
      unsafe {
        x0 = vaf(x0, o0);
        x1 = vaf(x1, o1);
        x2 = vaf(x2, o2);
        x3 = vaf(x3, o3);
        x4 = vaf(x4, o4);
        x5 = vaf(x5, o5);
        x6 = vaf(x6, o6);
        x7 = vaf(x7, o7);
        x8 = vaf(x8, o8);
        x9 = vaf(x9, o9);
        x10 = vaf(x10, o10);
        x11 = vaf(x11, o11);
        x12 = vaf(x12, o12);
        x13 = vaf(x13, o13);
        x14 = vaf(x14, o14);
        x15 = vaf(x15, o15);
      }

      let words = [
        to_u32x4(x0),
        to_u32x4(x1),
        to_u32x4(x2),
        to_u32x4(x3),
        to_u32x4(x4),
        to_u32x4(x5),
        to_u32x4(x6),
        to_u32x4(x7),
        to_u32x4(x8),
        to_u32x4(x9),
        to_u32x4(x10),
        to_u32x4(x11),
        to_u32x4(x12),
        to_u32x4(x13),
        to_u32x4(x14),
        to_u32x4(x15),
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

      counter = counter.wrapping_add(BLOCKS_PER_BATCH as u32);
    }

    let remainder = batches.into_remainder();
    if !remainder.is_empty() {
      xor_keystream_portable(key, counter, nonce, remainder);
    }
  }

  #[inline(always)]
  fn from_u32x4(words: [u32; 4]) -> i64x2 {
    // SAFETY: [u32; 4] and i64x2 are both 16 bytes with no alignment mismatch.
    unsafe { core::mem::transmute(words) }
  }

  #[inline(always)]
  fn to_u32x4(v: i64x2) -> [u32; 4] {
    // SAFETY: i64x2 and [u32; 4] are both 16 bytes.
    unsafe { core::mem::transmute(v) }
  }

  #[inline(always)]
  fn splat(val: u32) -> i64x2 {
    from_u32x4([val; 4])
  }

  /// Vector add fullword: `vaf`.
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn vaf(a: i64x2, b: i64x2) -> i64x2 {
    let out: i64x2;
    // SAFETY: z/Vector facility available via target_feature.
    unsafe {
      core::arch::asm!(
        "vaf {out}, {a}, {b}",
        out = lateout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(nomem, nostack, pure)
      );
    }
    out
  }

  /// Vector exclusive OR: `vx`.
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn vx(a: i64x2, b: i64x2) -> i64x2 {
    let out: i64x2;
    // SAFETY: z/Vector facility available via target_feature.
    unsafe {
      core::arch::asm!(
        "vx {out}, {a}, {b}",
        out = lateout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(nomem, nostack, pure)
      );
    }
    out
  }

  /// Vector element rotate left logical fullword: `verll` with M4=2.
  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn verllf<const BITS: u32>(a: i64x2) -> i64x2 {
    let out: i64x2;
    // SAFETY: z/Vector facility available via target_feature.
    unsafe {
      core::arch::asm!(
        "verll {out}, {a}, {bits}, 2",
        out = lateout(vreg) out,
        a = in(vreg) a,
        bits = const BITS,
        options(nomem, nostack, pure)
      );
    }
    out
  }

  #[inline]
  #[target_feature(enable = "vector")]
  unsafe fn quarter_round(a: &mut i64x2, b: &mut i64x2, c: &mut i64x2, d: &mut i64x2) {
    // SAFETY: z/Vector facility available via enclosing target_feature.
    unsafe {
      *a = vaf(*a, *b);
      *d = verllf::<16>(vx(*d, *a));

      *c = vaf(*c, *d);
      *b = verllf::<12>(vx(*b, *c));

      *a = vaf(*a, *b);
      *d = verllf::<8>(vx(*d, *a));

      *c = vaf(*c, *d);
      *b = verllf::<7>(vx(*b, *c));
    }
  }
}

#[cfg(target_arch = "riscv64")]
mod riscv64_vector {
  use super::{KEY_SIZE, NONCE_SIZE, xor_keystream_u32x4_impl};

  #[inline]
  pub(super) fn xor_keystream(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
    // SAFETY: Backend selection guarantees the vector extension before this wrapper is chosen.
    unsafe { xor_keystream_impl(key, initial_counter, nonce, buffer) }
  }

  #[target_feature(enable = "v")]
  unsafe fn xor_keystream_impl(
    key: &[u8; KEY_SIZE],
    initial_counter: u32,
    nonce: &[u8; NONCE_SIZE],
    buffer: &mut [u8],
  ) {
    // SAFETY: The wrapper only reaches this function when the RISC-V vector extension is available.
    unsafe { xor_keystream_u32x4_impl(key, initial_counter, nonce, buffer) }
  }
}

#[cfg(test)]
mod tests {
  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
  use super::xor_keystream_portable;
  use super::{KEY_SIZE, NONCE_SIZE, block, hchacha20, xor_keystream};
  use crate::aead::targets::AeadPrimitive;
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

    xor_keystream(AeadPrimitive::ChaCha20Poly1305, &key, 1, &nonce, &mut ciphertext);
    assert_ne!(ciphertext, plaintext);

    xor_keystream(AeadPrimitive::XChaCha20Poly1305, &key, 1, &nonce, &mut ciphertext);
    assert_eq!(ciphertext, plaintext);
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn avx512_backend_matches_portable_when_available() {
    if !crate::platform::caps().has(x86::AVX512_READY) {
      return;
    }

    let key = [0x71; KEY_SIZE];
    let nonce = [0x19; NONCE_SIZE];
    for len in [
      0usize, 1, 63, 64, 65, 255, 256, 257, 511, 512, 513, 1024, 1536, 2048, 4096, 8192,
    ] {
      let mut portable = vec![0u8; len];
      let mut accelerated = vec![0u8; len];
      let mut index = 0usize;
      while index < len {
        let value = index.strict_mul(13).strict_add(5) as u8;
        portable[index] = value;
        accelerated[index] = value;
        index = index.strict_add(1);
      }

      xor_keystream_portable(&key, 3, &nonce, &mut portable);
      super::x86_avx512::xor_keystream(&key, 3, &nonce, &mut accelerated);
      assert_eq!(accelerated, portable, "AVX-512 mismatch at len={len}");
    }
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn avx2_backend_matches_portable_when_available() {
    if !crate::platform::caps().has(x86::AVX2) {
      return;
    }

    let key = [0x55; KEY_SIZE];
    let nonce = [0x33; NONCE_SIZE];
    for len in [
      0usize, 1, 63, 64, 65, 255, 256, 257, 511, 512, 513, 1024, 2048, 4096, 8192,
    ] {
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
      assert_eq!(accelerated, portable, "AVX2 mismatch at len={len}");
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
