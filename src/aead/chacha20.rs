#![allow(clippy::indexing_slicing)]

//! Portable ChaCha20 and HChaCha20 core.

use core::mem;
#[cfg(target_arch = "riscv64")]
use core::simd::u32x4;

use crate::{
  aead::{
    LengthOverflow,
    targets::{AeadPrimitive, select_backend},
  },
  backend::cache::OnceCache,
  platform::{Arch, Caps},
};

pub(crate) const KEY_SIZE: usize = 32;
pub(crate) const NONCE_SIZE: usize = 12;
#[cfg(feature = "xchacha20poly1305")]
pub(crate) const HCHACHA_NONCE_SIZE: usize = 16;
pub(crate) const BLOCK_SIZE: usize = 64;
pub(crate) const POLY1305_KEY_SIZE: usize = 32;

const CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

type XorKeystreamFn = fn(&[u8; KEY_SIZE], u32, &[u8; NONCE_SIZE], &mut [u8]);

static XCHACHA20POLY1305_XOR_KEYSTREAM_DISPATCH: OnceCache<XorKeystreamFn> = OnceCache::new();
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

#[inline]
#[cfg(feature = "xchacha20poly1305")]
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
pub(crate) fn xor_keystream(
  primitive: AeadPrimitive,
  key: &[u8; KEY_SIZE],
  initial_counter: u32,
  nonce: &[u8; NONCE_SIZE],
  buffer: &mut [u8],
) -> Result<(), LengthOverflow> {
  let blocks = {
    let len = u64::try_from(buffer.len()).map_err(|_| LengthOverflow)?;
    if len == 0 {
      0
    } else {
      let block_size = u64::try_from(BLOCK_SIZE).map_err(|_| LengthOverflow)?;
      len.strict_sub(1).strict_div(block_size).strict_add(1)
    }
  };

  if blocks > 0 {
    let last_counter = u64::from(initial_counter).strict_add(blocks).strict_sub(1);
    if last_counter > u64::from(u32::MAX) {
      return Err(LengthOverflow);
    }
  }

  xor_keystream_resolved(primitive)(key, initial_counter, nonce, buffer);
  Ok(())
}

#[inline]
fn xor_keystream_resolved(primitive: AeadPrimitive) -> XorKeystreamFn {
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
    counter = counter.wrapping_add(1);
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
    debug_assert!(
      counter.checked_add((BLOCKS_PER_BATCH - 1) as u32).is_some(),
      "ChaCha20 block counter overflow"
    );

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
#[cfg(feature = "xchacha20poly1305")]
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

#[cfg(target_arch = "aarch64")]
#[path = "chacha20/aarch64_neon.rs"]
mod aarch64_neon;
#[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
#[path = "chacha20/powerpc64_vsx.rs"]
mod power_vsx;
#[cfg(target_arch = "riscv64")]
#[path = "chacha20/riscv64_vector.rs"]
mod riscv64_vector;
#[cfg(target_arch = "s390x")]
#[path = "chacha20/s390x_vector.rs"]
mod s390x_vector;
#[cfg(target_arch = "wasm32")]
#[path = "chacha20/wasm32_simd128.rs"]
mod wasm_simd128;
#[cfg(target_arch = "x86_64")]
#[path = "chacha20/x86_64_avx2.rs"]
mod x86_avx2;
#[cfg(target_arch = "x86_64")]
#[path = "chacha20/x86_64_avx512.rs"]
mod x86_avx512;

// ─── Forced-kernel diag entrypoints ────────────────────────────────────────
//
// Mirrors the `diag_compress_*` pattern in `src/auth/argon2/mod.rs`. Tests in
// `tests/aead_kernel_equivalence.rs` invoke each backend's `xor_keystream`
// directly to assert byte-identity with the portable oracle, regardless of
// which backend the runtime dispatcher would otherwise pick. This is what
// catches CRIT-1-class silent kernel divergence (e.g. POWER VSX rotation
// inversion) on every CI runner that has the corresponding target feature.

/// Run the **portable** ChaCha20 XOR-keystream regardless of host caps.
#[cfg(feature = "diag")]
pub fn diag_chacha20_xor_keystream_portable(
  key: &[u8; KEY_SIZE],
  initial_counter: u32,
  nonce: &[u8; NONCE_SIZE],
  buffer: &mut [u8],
) {
  xor_keystream_portable(key, initial_counter, nonce, buffer);
}

/// Run the aarch64 NEON ChaCha20 XOR-keystream.
///
/// # Safety
///
/// Caller must verify the host has `aarch64::NEON`. Compile-time gated to
/// `target_arch = "aarch64"`.
#[cfg(all(feature = "diag", target_arch = "aarch64"))]
pub fn diag_chacha20_xor_keystream_aarch64_neon(
  key: &[u8; KEY_SIZE],
  initial_counter: u32,
  nonce: &[u8; NONCE_SIZE],
  buffer: &mut [u8],
) {
  aarch64_neon::xor_keystream(key, initial_counter, nonce, buffer);
}

/// Run the x86_64 AVX2 ChaCha20 XOR-keystream.
///
/// # Safety
///
/// Caller must verify the host has `x86::AVX2`.
#[cfg(all(feature = "diag", target_arch = "x86_64"))]
pub fn diag_chacha20_xor_keystream_x86_avx2(
  key: &[u8; KEY_SIZE],
  initial_counter: u32,
  nonce: &[u8; NONCE_SIZE],
  buffer: &mut [u8],
) {
  x86_avx2::xor_keystream(key, initial_counter, nonce, buffer);
}

/// Run the x86_64 AVX-512 ChaCha20 XOR-keystream.
///
/// # Safety
///
/// Caller must verify the host has `x86::AVX512F + AVX512VL + AVX512BW`.
#[cfg(all(feature = "diag", target_arch = "x86_64"))]
pub fn diag_chacha20_xor_keystream_x86_avx512(
  key: &[u8; KEY_SIZE],
  initial_counter: u32,
  nonce: &[u8; NONCE_SIZE],
  buffer: &mut [u8],
) {
  x86_avx512::xor_keystream(key, initial_counter, nonce, buffer);
}

/// Run the POWER VSX ChaCha20 XOR-keystream.
///
/// # Safety
///
/// Caller must verify the host is `powerpc64le` with VSX. The portable
/// kernel — which has been the correctness oracle since commit `2631aefa`
/// fixed the rotation-amount bug here — must produce identical bytes.
#[cfg(all(feature = "diag", target_arch = "powerpc64", target_endian = "little"))]
pub fn diag_chacha20_xor_keystream_power_vsx(
  key: &[u8; KEY_SIZE],
  initial_counter: u32,
  nonce: &[u8; NONCE_SIZE],
  buffer: &mut [u8],
) {
  power_vsx::xor_keystream(key, initial_counter, nonce, buffer);
}

/// Run the s390x z/Vector ChaCha20 XOR-keystream.
///
/// # Safety
///
/// Caller must verify the host has `s390x::VECTOR`. Same correctness-oracle
/// invariant as POWER VSX above.
#[cfg(all(feature = "diag", target_arch = "s390x"))]
pub fn diag_chacha20_xor_keystream_s390x_vector(
  key: &[u8; KEY_SIZE],
  initial_counter: u32,
  nonce: &[u8; NONCE_SIZE],
  buffer: &mut [u8],
) {
  s390x_vector::xor_keystream(key, initial_counter, nonce, buffer);
}

/// Run the riscv64 RVV ChaCha20 XOR-keystream.
///
/// # Safety
///
/// Caller must verify the host has `riscv::V`.
#[cfg(all(feature = "diag", target_arch = "riscv64"))]
pub fn diag_chacha20_xor_keystream_riscv64_vector(
  key: &[u8; KEY_SIZE],
  initial_counter: u32,
  nonce: &[u8; NONCE_SIZE],
  buffer: &mut [u8],
) {
  riscv64_vector::xor_keystream(key, initial_counter, nonce, buffer);
}

/// Run the wasm32 simd128 ChaCha20 XOR-keystream.
#[cfg(all(feature = "diag", target_arch = "wasm32"))]
pub fn diag_chacha20_xor_keystream_wasm_simd128(
  key: &[u8; KEY_SIZE],
  initial_counter: u32,
  nonce: &[u8; NONCE_SIZE],
  buffer: &mut [u8],
) {
  wasm_simd128::xor_keystream(key, initial_counter, nonce, buffer);
}
#[cfg(test)]
mod tests {
  #[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "powerpc64", target_endian = "little"),
    target_arch = "s390x"
  ))]
  use alloc::vec;

  #[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "powerpc64", target_endian = "little"),
    target_arch = "s390x"
  ))]
  use super::xor_keystream_portable;
  use super::{KEY_SIZE, NONCE_SIZE, block, hchacha20, xor_keystream};
  use crate::aead::targets::AeadPrimitive;
  #[cfg(target_arch = "aarch64")]
  use crate::platform::caps::aarch64;
  #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
  use crate::platform::caps::power;
  #[cfg(target_arch = "s390x")]
  use crate::platform::caps::s390x;
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

    xor_keystream(AeadPrimitive::ChaCha20Poly1305, &key, 1, &nonce, &mut ciphertext).unwrap();
    assert_ne!(ciphertext, plaintext);

    xor_keystream(AeadPrimitive::XChaCha20Poly1305, &key, 1, &nonce, &mut ciphertext).unwrap();
    assert_eq!(ciphertext, plaintext);
  }

  #[test]
  fn xor_keystream_boundary_respects_u32_counter() {
    let key = [0u8; KEY_SIZE];
    let nonce = [0u8; NONCE_SIZE];

    let mut one_block = [0u8; 64];
    assert!(xor_keystream(AeadPrimitive::ChaCha20Poly1305, &key, u32::MAX, &nonce, &mut one_block).is_ok());

    let mut two_blocks = [0u8; 65];
    assert!(xor_keystream(AeadPrimitive::ChaCha20Poly1305, &key, u32::MAX, &nonce, &mut two_blocks).is_err());
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

  #[test]
  #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
  fn power_vsx_backend_matches_portable() {
    if !crate::platform::caps().has(power::POWER8_VECTOR) {
      return;
    }

    let key = [0x39; KEY_SIZE];
    let nonce = [0x52; NONCE_SIZE];
    for len in [0usize, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 768] {
      let mut portable = vec![0u8; len];
      let mut accelerated = vec![0u8; len];
      let mut index = 0usize;
      while index < len {
        let value = index.strict_mul(23).strict_add(11) as u8;
        portable[index] = value;
        accelerated[index] = value;
        index = index.strict_add(1);
      }

      xor_keystream_portable(&key, 5, &nonce, &mut portable);
      super::power_vsx::xor_keystream(&key, 5, &nonce, &mut accelerated);
      assert_eq!(accelerated, portable, "POWER VSX mismatch at len={len}");
    }
  }

  #[test]
  #[cfg(target_arch = "s390x")]
  fn s390x_backend_matches_portable() {
    if !crate::platform::caps().has(s390x::MSA) {
      return;
    }

    let key = [0x2b; KEY_SIZE];
    let nonce = [0x64; NONCE_SIZE];
    for len in [0usize, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 768] {
      let mut portable = vec![0u8; len];
      let mut accelerated = vec![0u8; len];
      let mut index = 0usize;
      while index < len {
        let value = index.strict_mul(31).strict_add(7) as u8;
        portable[index] = value;
        accelerated[index] = value;
        index = index.strict_add(1);
      }

      xor_keystream_portable(&key, 9, &nonce, &mut portable);
      super::s390x_vector::xor_keystream(&key, 9, &nonce, &mut accelerated);
      assert_eq!(accelerated, portable, "s390x vector mismatch at len={len}");
    }
  }
}
