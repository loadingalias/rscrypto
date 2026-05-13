//! AES-round fast hash (**NOT CRYPTO**).
//!
//! `AesHash64` and `AesHash128` are hardware-oriented, non-cryptographic
//! hashes for in-process hash tables, sharding, and fingerprints. They use
//! AES rounds as a fast avalanche primitive when AES-NI/AES-CE is available,
//! with a byte-identical portable reference for every target.
//!
//! The >=49B AES compression schedule is adapted from gxhash 3.5.0
//! (MIT, Copyright (c) 2023 Olivier Giniaux) with rscrypto-local dispatch,
//! scalar short-input handling, and a portable reference path.

#![allow(clippy::indexing_slicing)] // AES reference and fixed-size block code.

use crate::traits::FastHash;

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "portable-only"),
  not(miri)
))]
mod x86;

#[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
mod aarch64;

const BLOCK_SIZE: usize = 16;
const LONG_CHUNK_SIZE: usize = 8 * BLOCK_SIZE;
const HASH64_GX_LONG_THRESHOLD: usize = 256 * 1024;

const RAPID_SECRETS: [u64; 3] = [0x2d35_8dcc_aa6c_78a5, 0x8bb8_4b93_962e_acc9, 0x4b33_a62e_d433_d4a3];
const RAPID_HI_SEED: u64 = 0x9e37_79b9_7f4a_7c15;
const RAPID_PRECOMPUTED_SEED_0: u64 = rapid_seed_cpp(0);
const RAPID_PRECOMPUTED_SEED_HI: u64 = rapid_seed_cpp(RAPID_HI_SEED);
const RAPID_EMPTY_HASH_SEED_0: u64 = rapid_fast_empty(RAPID_PRECOMPUTED_SEED_0);
const RAPID_EMPTY_HASH_SEED_HI: u64 = rapid_fast_empty(RAPID_PRECOMPUTED_SEED_HI);
const AESHASH128_EMPTY_SEED_0: u128 =
  (RAPID_EMPTY_HASH_SEED_0 as u128) | (((RAPID_EMPTY_HASH_SEED_0 ^ RAPID_EMPTY_HASH_SEED_HI) as u128) << 64);

#[rustfmt::skip]
const GX_KEYS: [u32; 12] = [
  0xf278_4542, 0xb09d_3e21, 0x89c2_22e5, 0xfc3b_c28e,
  0x03fc_e279, 0xcb6b_2e9b, 0xb361_dc58, 0x3913_2bd9,
  0xd001_2e32, 0x689d_2b7d, 0x5544_b1b7, 0xc78b_122b,
];

/// AES-round accelerated 64-bit fast hash.
#[derive(Clone, Debug, Default)]
pub struct AesHash64;

/// AES-round accelerated 128-bit fast hash.
#[derive(Clone, Debug, Default)]
pub struct AesHash128;

#[rustfmt::skip]
const KEYS: [[u8; BLOCK_SIZE]; 8] = [
  [0x72, 0x73, 0x63, 0x72, 0x79, 0x70, 0x74, 0x6f, 0x2f, 0x61, 0x65, 0x73, 0x68, 0x61, 0x73, 0x68],
  [0x9d, 0x38, 0xf1, 0x8b, 0xe3, 0x6c, 0x42, 0x15, 0x47, 0xa9, 0xd2, 0x70, 0x1c, 0x5b, 0x83, 0xee],
  [0x4f, 0x27, 0x91, 0xd6, 0x0b, 0xfa, 0x35, 0xc8, 0xa6, 0x5d, 0x18, 0x73, 0xe2, 0x94, 0xbc, 0x09],
  [0xc3, 0x0e, 0x56, 0xa4, 0x7b, 0x19, 0xd8, 0x2f, 0x95, 0xe1, 0x6a, 0x40, 0x3d, 0xb7, 0x0c, 0xf2],
  [0x16, 0xbb, 0x5e, 0x29, 0xd1, 0x84, 0x03, 0x77, 0xea, 0x4c, 0xaf, 0x90, 0x62, 0x38, 0xcd, 0x11],
  [0x58, 0xe7, 0x24, 0xac, 0x9a, 0x31, 0xdf, 0x06, 0x73, 0xc5, 0x48, 0xb2, 0x0f, 0x9d, 0x61, 0xfa],
  [0xaf, 0x04, 0xc9, 0x72, 0x3e, 0xd6, 0x1b, 0x85, 0x50, 0x27, 0xfa, 0x9c, 0xb4, 0x68, 0x0d, 0xe1],
  [0x3a, 0x8f, 0xd0, 0x57, 0xc6, 0x21, 0x9b, 0x74, 0x0e, 0xb5, 0x68, 0xa2, 0xf1, 0x4c, 0x33, 0xdd],
];

#[rustfmt::skip]
const SBOX: [u8; 256] = [
  0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
  0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
  0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
  0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
  0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
  0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
  0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
  0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
  0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
  0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
  0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
  0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
  0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
  0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
  0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
  0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

impl AesHash64 {
  /// Hash `data` to 64 bits using the default seed.
  #[inline(always)]
  pub fn hash(data: &[u8]) -> u64 {
    let len = data.len();
    if len >= HASH64_GX_LONG_THRESHOLD {
      return hash64_seeded_non_exact(0, data);
    }
    if len == 4 * BLOCK_SIZE {
      return hash64_fold(0, data);
    }
    if len == 2 * BLOCK_SIZE {
      return hash64_32(0, data);
    }
    if len == 0 {
      return RAPID_EMPTY_HASH_SEED_0;
    }
    hash64_seeded_non_exact(0, data)
  }

  /// Hash `data` to 64 bits using `seed`.
  #[inline(always)]
  pub fn hash_with_seed(seed: u64, data: &[u8]) -> u64 {
    hash64_seeded(seed, data)
  }
}

impl AesHash128 {
  /// Hash `data` to 128 bits using the default seed.
  #[inline(always)]
  pub fn hash(data: &[u8]) -> u128 {
    if data.is_empty() {
      return AESHASH128_EMPTY_SEED_0;
    }
    hash128_seeded(0, data)
  }

  /// Hash `data` to 128 bits using `seed`.
  #[inline(always)]
  pub fn hash_with_seed(seed: u64, data: &[u8]) -> u128 {
    hash128_seeded(seed, data)
  }
}

impl FastHash for AesHash64 {
  const OUTPUT_SIZE: usize = 8;
  type Output = u64;
  type Seed = u64;

  #[inline(always)]
  fn hash(data: &[u8]) -> Self::Output {
    AesHash64::hash(data)
  }

  #[inline(always)]
  fn hash_with_seed(seed: Self::Seed, data: &[u8]) -> Self::Output {
    AesHash64::hash_with_seed(seed, data)
  }
}

impl FastHash for AesHash128 {
  const OUTPUT_SIZE: usize = 16;
  type Output = u128;
  type Seed = u64;

  #[inline(always)]
  fn hash(data: &[u8]) -> Self::Output {
    AesHash128::hash(data)
  }

  #[inline(always)]
  fn hash_with_seed(seed: Self::Seed, data: &[u8]) -> Self::Output {
    AesHash128::hash_with_seed(seed, data)
  }
}

#[inline(always)]
fn hash64_seeded(seed: u64, data: &[u8]) -> u64 {
  let len = data.len();
  if len == 4 * BLOCK_SIZE {
    return hash64_fold(seed, data);
  }
  if len == 2 * BLOCK_SIZE {
    return hash64_32(seed, data);
  }
  hash64_seeded_non_exact(seed, data)
}

#[inline(always)]
fn hash64_seeded_non_exact(seed: u64, data: &[u8]) -> u64 {
  let len = data.len();
  if len <= BLOCK_SIZE {
    return short_hash64(seed, data);
  }
  if hash64_long_len(len) {
    return hash64_long(seed, data);
  }
  if hash64_low_lane_len(len) {
    return hash64_low(seed, data);
  }
  fold_hash128(hash128(seed, data))
}

#[inline(always)]
fn hash128_seeded(seed: u64, data: &[u8]) -> u128 {
  if data.len() <= BLOCK_SIZE {
    return short_hash128(seed, data);
  }
  hash128(seed, data)
}

#[inline(always)]
#[allow(unreachable_code)]
fn hash64_low_lane_len(len: usize) -> bool {
  len >= LONG_CHUNK_SIZE && len.is_multiple_of(LONG_CHUNK_SIZE)
}

#[inline(always)]
fn hash64_long_len(len: usize) -> bool {
  len >= HASH64_GX_LONG_THRESHOLD && len.is_multiple_of(LONG_CHUNK_SIZE)
}

#[inline(never)]
fn hash64_long(seed: u64, data: &[u8]) -> u64 {
  debug_assert!(hash64_long_len(data.len()));
  hash64_low(seed, data)
}

#[inline(always)]
#[allow(unreachable_code)]
fn hash64_32(seed: u64, data: &[u8]) -> u64 {
  debug_assert_eq!(data.len(), 2 * BLOCK_SIZE);

  #[cfg(all(
    target_arch = "aarch64",
    target_os = "macos",
    not(feature = "portable-only"),
    not(miri)
  ))]
  {
    // SAFETY: every aarch64-apple-darwin target is Apple Silicon with NEON and AES-CE.
    return unsafe { aarch64::hash64_32_static(seed, data) };
  }

  #[cfg(all(
    target_arch = "aarch64",
    target_feature = "aes",
    target_feature = "neon",
    not(feature = "portable-only"),
    not(miri)
  ))]
  {
    // SAFETY: compile-time target features guarantee AES-CE + NEON.
    return unsafe { aarch64::hash64_32_static(seed, data) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "aes",
    target_feature = "sse2",
    not(feature = "portable-only"),
    not(miri)
  ))]
  {
    // SAFETY: compile-time target features guarantee AES-NI + SSE2.
    return unsafe { x86::hash64_32_static(seed, data) };
  }

  hash64_32_runtime(seed, data)
}

#[inline(never)]
fn hash64_32_runtime(seed: u64, data: &[u8]) -> u64 {
  debug_assert_eq!(data.len(), 2 * BLOCK_SIZE);

  #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
  {
    let caps = crate::platform::caps();
    if caps.has(crate::platform::caps::aarch64::AES | crate::platform::caps::aarch64::NEON) {
      // SAFETY: runtime dispatch confirmed AES-CE + NEON before entering the target-feature function.
      return unsafe { aarch64::hash64_32(seed, data) };
    }
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "portable-only"),
    not(miri)
  ))]
  {
    let caps = crate::platform::caps();
    if caps.has(crate::platform::caps::x86::AESNI | crate::platform::caps::x86::SSE2) {
      // SAFETY: runtime dispatch confirmed AES-NI + SSE2 before entering the target-feature function.
      return unsafe { x86::hash64_32(seed, data) };
    }
  }

  portable::hash64_32(seed, data)
}

#[inline(always)]
#[allow(unreachable_code)]
fn hash64_fold(seed: u64, data: &[u8]) -> u64 {
  debug_assert_eq!(data.len(), 4 * BLOCK_SIZE);

  #[cfg(all(
    target_arch = "aarch64",
    target_os = "macos",
    not(feature = "portable-only"),
    not(miri)
  ))]
  {
    // SAFETY: every aarch64-apple-darwin target is Apple Silicon with NEON and AES-CE.
    return unsafe { aarch64::hash64_fold_static(seed, data) };
  }

  #[cfg(all(
    target_arch = "aarch64",
    target_feature = "aes",
    target_feature = "neon",
    not(feature = "portable-only"),
    not(miri)
  ))]
  {
    // SAFETY: compile-time target features guarantee AES-CE + NEON.
    return unsafe { aarch64::hash64_fold_static(seed, data) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "aes",
    target_feature = "sse2",
    not(feature = "portable-only"),
    not(miri)
  ))]
  {
    // SAFETY: compile-time target features guarantee AES-NI + SSE2.
    return unsafe { x86::hash64_fold_static(seed, data) };
  }

  hash64_fold_runtime(seed, data)
}

#[inline(never)]
fn hash64_fold_runtime(seed: u64, data: &[u8]) -> u64 {
  debug_assert_eq!(data.len(), 4 * BLOCK_SIZE);

  #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
  {
    let caps = crate::platform::caps();
    if caps.has(crate::platform::caps::aarch64::AES | crate::platform::caps::aarch64::NEON) {
      // SAFETY: runtime dispatch confirmed AES-CE + NEON before entering the target-feature function.
      return unsafe { aarch64::hash64_fold(seed, data) };
    }
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "portable-only"),
    not(miri)
  ))]
  {
    let caps = crate::platform::caps();
    if caps.has(crate::platform::caps::x86::AESNI | crate::platform::caps::x86::SSE2) {
      // SAFETY: runtime dispatch confirmed AES-NI + SSE2 before entering the target-feature function.
      return unsafe { x86::hash64_fold(seed, data) };
    }
  }

  portable::hash64_64(seed, data)
}

#[inline(always)]
#[allow(unreachable_code)]
fn hash64_low(seed: u64, data: &[u8]) -> u64 {
  debug_assert!(hash64_low_lane_len(data.len()));

  #[cfg(all(
    target_arch = "aarch64",
    target_os = "macos",
    not(feature = "portable-only"),
    not(miri)
  ))]
  {
    // SAFETY: every aarch64-apple-darwin target is Apple Silicon with NEON and AES-CE.
    return unsafe { aarch64::hash64_low_static(seed, data) };
  }

  #[cfg(all(
    target_arch = "aarch64",
    target_feature = "aes",
    target_feature = "neon",
    not(feature = "portable-only"),
    not(miri)
  ))]
  {
    // SAFETY: compile-time target features guarantee AES-CE + NEON.
    return unsafe { aarch64::hash64_low_static(seed, data) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "aes",
    target_feature = "sse2",
    not(feature = "portable-only"),
    not(miri)
  ))]
  {
    // SAFETY: compile-time target features guarantee AES-NI + SSE2.
    return unsafe { x86::hash64_low_static(seed, data) };
  }

  hash64_low_runtime(seed, data)
}

#[inline(never)]
fn hash64_low_runtime(seed: u64, data: &[u8]) -> u64 {
  debug_assert!(hash64_low_lane_len(data.len()));

  #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
  {
    let caps = crate::platform::caps();
    if caps.has(crate::platform::caps::aarch64::AES | crate::platform::caps::aarch64::NEON) {
      // SAFETY: runtime dispatch confirmed AES-CE + NEON before entering the target-feature function.
      return unsafe { aarch64::hash64_low(seed, data) };
    }
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "portable-only"),
    not(miri)
  ))]
  {
    let caps = crate::platform::caps();
    if caps.has(crate::platform::caps::x86::AESNI | crate::platform::caps::x86::SSE2) {
      // SAFETY: runtime dispatch confirmed AES-NI + SSE2 before entering the target-feature function.
      return unsafe { x86::hash64_low(seed, data) };
    }
  }

  portable::hash64_low(seed, data)
}

#[inline(always)]
#[allow(unreachable_code)]
fn hash128(seed: u64, data: &[u8]) -> u128 {
  #[cfg(all(
    target_arch = "aarch64",
    target_os = "macos",
    not(feature = "portable-only"),
    not(miri)
  ))]
  {
    // SAFETY: every aarch64-apple-darwin target is Apple Silicon with NEON and AES-CE.
    return unsafe { aarch64::hash128_static(seed, data) };
  }

  #[cfg(all(
    target_arch = "aarch64",
    target_feature = "aes",
    target_feature = "neon",
    not(feature = "portable-only"),
    not(miri)
  ))]
  {
    // SAFETY: compile-time target features guarantee AES-CE + NEON.
    return unsafe { aarch64::hash128_static(seed, data) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "aes",
    target_feature = "sse2",
    not(feature = "portable-only"),
    not(miri)
  ))]
  {
    // SAFETY: compile-time target features guarantee AES-NI + SSE2.
    return unsafe { x86::hash128_static(seed, data) };
  }

  hash128_runtime(seed, data)
}

#[inline(never)]
fn hash128_runtime(seed: u64, data: &[u8]) -> u128 {
  #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
  {
    let caps = crate::platform::caps();
    if caps.has(crate::platform::caps::aarch64::AES | crate::platform::caps::aarch64::NEON) {
      // SAFETY: runtime dispatch confirmed AES-CE + NEON before entering the target-feature function.
      return unsafe { aarch64::hash128(seed, data) };
    }
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "portable-only"),
    not(miri)
  ))]
  {
    let caps = crate::platform::caps();
    if caps.has(crate::platform::caps::x86::AESNI | crate::platform::caps::x86::SSE2) {
      // SAFETY: runtime dispatch confirmed AES-NI + SSE2 before entering the target-feature function.
      return unsafe { x86::hash128(seed, data) };
    }
  }

  portable::hash128(seed, data)
}

#[inline(always)]
pub(super) const fn seed_lo(seed: u64, lane: u64) -> u64 {
  seed ^ lane.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

#[inline(always)]
pub(super) const fn seed_hi(seed: u64, lane: u64) -> u64 {
  seed.rotate_left(32) ^ 0xd1b5_4a32_d192_ed03u64.wrapping_mul(lane.wrapping_add(1))
}

#[inline(always)]
pub(super) const fn len_lo(seed: u64, len: usize) -> u64 {
  (len as u64) ^ seed.rotate_left(17) ^ 0xa24b_aed4_963e_e407
}

#[inline(always)]
pub(super) const fn len_hi(seed: u64, len: usize) -> u64 {
  (len as u64).rotate_left(32) ^ seed.wrapping_mul(0x9fb2_1c65_1e98_df25) ^ 0x9e37_79b9_7f4a_7c15
}

#[inline(always)]
pub(super) fn short_hash128(seed: u64, data: &[u8]) -> u128 {
  debug_assert!(data.len() <= BLOCK_SIZE);
  if data.is_empty() {
    if seed == 0 {
      return AESHASH128_EMPTY_SEED_0;
    }
    let h = rapid_fast_empty(rapid_seed_cpp(seed));
    return (h as u128) | (((h ^ rapid_fast_empty(rapid_seed_cpp(seed ^ RAPID_HI_SEED))) as u128) << 64);
  }
  if data.len() == 1 {
    return short_one(seed, data[0]);
  }

  let seed = if seed == 0 {
    RAPID_PRECOMPUTED_SEED_0
  } else {
    rapid_seed_cpp(seed)
  };
  let (a, b, seed) = rapid_small_parts(data, seed);
  let (lo, hi) = mul_wide(a ^ RAPID_SECRETS[0], b ^ seed);
  let lo = lo ^ hi.rotate_left(23) ^ seed;
  let hi = hi ^ lo.rotate_right(17) ^ (data.len() as u64).wrapping_mul(RAPID_HI_SEED);
  (lo as u128) | ((hi as u128) << 64)
}

#[inline(always)]
fn short_hash64(seed: u64, data: &[u8]) -> u64 {
  short_hash128(seed, data) as u64
}

#[inline(always)]
fn fold_hash128(hash: u128) -> u64 {
  let lo = hash as u64;
  let hi = (hash >> 64) as u64;
  lo ^ hi.rotate_left(32)
}

#[inline(always)]
fn short_one(seed: u64, byte: u8) -> u128 {
  let x = (byte as u64).wrapping_mul(0x0101_0101_0101_0101) ^ seed.rotate_left(17);
  let lo = x.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ RAPID_SECRETS[0];
  let hi = lo.rotate_left(32) ^ x.rotate_right(17) ^ RAPID_SECRETS[1] ^ seed.rotate_left(41);
  (lo as u128) | ((hi as u128) << 64)
}

#[inline(always)]
fn rapid_small_parts(data: &[u8], mut seed: u64) -> (u64, u64, u64) {
  debug_assert!(data.len() <= BLOCK_SIZE);
  let mut a = 0u64;
  let mut b = 0u64;

  if data.len() == 1 {
    let byte = data[0] as u64;
    return ((byte << 45) | byte, byte, seed.wrapping_add(1));
  }

  if data.len() >= 8 {
    a = read_u64_le(data, 0);
    b = read_u64_le(data, data.len() - 8);
  } else if data.len() >= 4 {
    a = read_u32_le(data, 0) as u64;
    b = read_u32_le(data, data.len() - 4) as u64;
  } else if !data.is_empty() {
    a = ((data[0] as u64) << 45) | data[data.len() - 1] as u64;
    b = data[data.len() >> 1] as u64;
  }

  seed = seed.wrapping_add(data.len() as u64);
  (a, b, seed)
}

#[inline(always)]
const fn rapid_seed_cpp(seed: u64) -> u64 {
  seed ^ rapid_mix(seed ^ RAPID_SECRETS[2], RAPID_SECRETS[1])
}

#[inline(always)]
const fn rapid_fast_empty(seed: u64) -> u64 {
  rapid_mix(RAPID_SECRETS[0], seed)
}

#[inline(always)]
const fn rapid_mix(a: u64, b: u64) -> u64 {
  let (lo, hi) = mul_wide(a, b);
  lo ^ hi
}

#[inline(always)]
const fn mul_wide(a: u64, b: u64) -> (u64, u64) {
  let r = (a as u128).wrapping_mul(b as u128);
  (r as u64, (r >> 64) as u64)
}

#[inline(always)]
fn read_u32_le(input: &[u8], offset: usize) -> u32 {
  debug_assert!(offset + 4 <= input.len());
  // SAFETY: caller ensures `offset + 4 <= input.len()`, and `read_unaligned` permits unaligned input.
  let v = unsafe { core::ptr::read_unaligned(input.as_ptr().add(offset) as *const u32) };
  u32::from_le(v)
}

#[inline(always)]
fn read_u64_le(input: &[u8], offset: usize) -> u64 {
  debug_assert!(offset + 8 <= input.len());
  // SAFETY: caller ensures `offset + 8 <= input.len()`, and `read_unaligned` permits unaligned input.
  let v = unsafe { core::ptr::read_unaligned(input.as_ptr().add(offset) as *const u64) };
  u64::from_le(v)
}

mod portable {
  use super::{
    BLOCK_SIZE, GX_KEYS, HASH64_GX_LONG_THRESHOLD, KEYS, LONG_CHUNK_SIZE, RAPID_HI_SEED, SBOX, len_hi, len_lo,
    rapid_mix, read_u64_le, seed_hi, seed_lo, short_hash128,
  };

  #[inline]
  pub(super) fn hash128(seed: u64, data: &[u8]) -> u128 {
    if data.len() <= BLOCK_SIZE {
      return short_hash128(seed, data);
    }
    if data.len() <= 2 * BLOCK_SIZE {
      return hash_17to32(seed, data);
    }
    if data.len() <= 4 * BLOCK_SIZE {
      return hash_33to64(seed, data);
    }
    gx_hash128(seed, data)
  }

  #[inline(always)]
  pub(super) fn hash64_low(seed: u64, data: &[u8]) -> u64 {
    debug_assert!(super::hash64_low_lane_len(data.len()));
    if data.len() >= HASH64_GX_LONG_THRESHOLD {
      return block_to_u128(gx_finalize(aesenc(gx_compress_all(data), gx_seed(seed)))) as u64;
    }
    hash128(seed, data) as u64
  }

  #[inline(always)]
  pub(super) fn hash64_32(seed: u64, data: &[u8]) -> u64 {
    debug_assert_eq!(data.len(), 2 * BLOCK_SIZE);
    let a = read_u64_le(data, 0) ^ read_u64_le(data, 2 * BLOCK_SIZE - 16) ^ seed.rotate_left(17);
    let b = read_u64_le(data, 8) ^ read_u64_le(data, 2 * BLOCK_SIZE - 8) ^ RAPID_HI_SEED ^ (2 * BLOCK_SIZE as u64);
    rapid_mix(a, b)
  }

  #[inline(always)]
  pub(super) fn hash64_64(seed: u64, data: &[u8]) -> u64 {
    debug_assert_eq!(data.len(), 4 * BLOCK_SIZE);
    let mix01 = xor(load_block(data, 0), load_block(data, BLOCK_SIZE));
    let mix23 = xor(load_block(data, 2 * BLOCK_SIZE), load_block(data, 3 * BLOCK_SIZE));
    let mix = xor(mix01, mix23);
    let folded = xor(mix, rotate_bytes::<8>(mix));
    block_to_u128(aesenclast(folded, len_block(seed, 4 * BLOCK_SIZE))) as u64
  }

  #[inline(always)]
  fn hash_17to32(seed: u64, data: &[u8]) -> u128 {
    let tail = if data.len() == 2 * BLOCK_SIZE {
      load_block(data, BLOCK_SIZE)
    } else {
      partial_block(&data[BLOCK_SIZE..], data.len())
    };
    let h = aesenc(
      xor(load_block(data, 0), seed_block(seed, 0)),
      xor(tail, len_block(seed, data.len())),
    );
    block_to_u128(h)
  }

  #[inline(always)]
  fn hash_33to64(seed: u64, data: &[u8]) -> u128 {
    if data.len() == 4 * BLOCK_SIZE {
      return block_to_u128(hash_64(seed, data));
    }

    if data.len() <= 3 * BLOCK_SIZE {
      let tail = if data.len() == 3 * BLOCK_SIZE {
        load_block(data, 2 * BLOCK_SIZE)
      } else {
        partial_block(&data[2 * BLOCK_SIZE..], data.len())
      };
      let h0 = aesenc(xor(load_block(data, 0), tail), len_block(seed, data.len()));
      let h1 = aesenc(xor(load_block(data, BLOCK_SIZE), seed_block(seed, 1)), KEYS[1]);
      return block_to_u128(xor(h0, h1));
    }

    let tail = if data.len() == 4 * BLOCK_SIZE {
      load_block(data, 3 * BLOCK_SIZE)
    } else {
      partial_block(&data[3 * BLOCK_SIZE..], data.len())
    };
    let h0 = aesenc(
      xor(load_block(data, 0), load_block(data, 2 * BLOCK_SIZE)),
      seed_block(seed, 0),
    );
    let h1 = aesenc(xor(load_block(data, BLOCK_SIZE), tail), len_block(seed, data.len()));
    block_to_u128(aesenclast(xor(h0, h1), KEYS[7]))
  }

  #[inline(always)]
  fn hash_64(seed: u64, data: &[u8]) -> [u8; BLOCK_SIZE] {
    let a = hash_64_mix(data);
    let b = xor(seed_block(seed, 0), len_block(seed, 4 * BLOCK_SIZE));
    aesenc(a, b)
  }

  #[inline(always)]
  fn hash_64_mix(data: &[u8]) -> [u8; BLOCK_SIZE] {
    let v0 = load_block(data, 0);
    let v1 = load_block(data, BLOCK_SIZE);
    let v2 = load_block(data, 2 * BLOCK_SIZE);
    let v3 = load_block(data, 3 * BLOCK_SIZE);
    xor(
      xor(v0, rotate_bytes::<3>(v1)),
      xor(rotate_bytes::<7>(v2), rotate_bytes::<11>(v3)),
    )
  }

  #[inline]
  pub(super) fn gx_hash128(seed: u64, data: &[u8]) -> u128 {
    if data.len() >= LONG_CHUNK_SIZE && data.len().is_multiple_of(LONG_CHUNK_SIZE) {
      return fast_aligned_hash128(seed, data);
    }
    let compressed = gx_compress_all(data);
    block_to_u128(gx_finalize(aesenc(compressed, gx_seed(seed))))
  }

  #[inline(always)]
  fn fast_aligned_hash128(seed: u64, data: &[u8]) -> u128 {
    let mut offset = 0usize;
    let mut lane = xor(gx_seed(seed), gx_key(0));

    while offset < data.len() {
      let v0 = load_block(data, offset);
      let v1 = load_block(data, offset + 16);
      let v2 = load_block(data, offset + 32);
      let v3 = load_block(data, offset + 48);
      let v4 = load_block(data, offset + 64);
      let v5 = load_block(data, offset + 80);
      let v6 = load_block(data, offset + 96);
      let v7 = load_block(data, offset + 112);

      let a = aesenc(xor(v0, v4), xor(v1, v5));
      let b = aesenc(xor(v2, v6), xor(v3, v7));
      let tweak = u32x4((offset as u32) ^ (data.len() as u32));
      lane = aesenclast(xor(xor(a, b), tweak), lane);

      offset += LONG_CHUNK_SIZE;
    }

    block_to_u128(gx_finalize(lane))
  }

  #[inline(always)]
  fn gx_compress_all(data: &[u8]) -> [u8; BLOCK_SIZE] {
    let len = data.len();
    debug_assert!(len > 3 * BLOCK_SIZE);

    let mut offset;
    let mut hash_vector;

    let extra = len % BLOCK_SIZE;
    if extra == 0 {
      hash_vector = load_block(data, 0);
      offset = BLOCK_SIZE;
    } else {
      hash_vector = gx_partial(&data[..extra], extra);
      offset = extra;
    }

    let mut v0 = load_block(data, offset);
    offset += BLOCK_SIZE;

    if len > 2 * BLOCK_SIZE {
      let v = load_block(data, offset);
      offset += BLOCK_SIZE;
      v0 = aesenc(v0, v);

      if len > 3 * BLOCK_SIZE {
        let v = load_block(data, offset);
        offset += BLOCK_SIZE;
        v0 = aesenc(v0, v);

        if len > 4 * BLOCK_SIZE {
          hash_vector = gx_compress_many(data, offset, hash_vector, len);
        }
      }
    }

    aesenclast(hash_vector, aesenc(aesenc(v0, gx_key(0)), gx_key(4)))
  }

  #[inline(always)]
  fn gx_compress_many(
    data: &[u8],
    mut offset: usize,
    mut hash_vector: [u8; BLOCK_SIZE],
    len: usize,
  ) -> [u8; BLOCK_SIZE] {
    let remaining = len - offset;
    let unrollable_blocks = remaining / LONG_CHUNK_SIZE * 8;
    let remaining = remaining - unrollable_blocks * BLOCK_SIZE;
    let scalar_end = offset + remaining;

    while offset < scalar_end {
      hash_vector = aesenc(hash_vector, load_block(data, offset));
      offset += BLOCK_SIZE;
    }

    gx_compress_8(data, offset, hash_vector, len)
  }

  #[inline(always)]
  fn gx_compress_8(data: &[u8], mut offset: usize, hash_vector: [u8; BLOCK_SIZE], len: usize) -> [u8; BLOCK_SIZE] {
    let mut t1 = [0u8; BLOCK_SIZE];
    let mut t2 = [0u8; BLOCK_SIZE];
    let mut lane1 = hash_vector;
    let mut lane2 = hash_vector;

    while offset < len {
      let v0 = load_block(data, offset);
      let v1 = load_block(data, offset + 16);
      let v2 = load_block(data, offset + 32);
      let v3 = load_block(data, offset + 48);
      let v4 = load_block(data, offset + 64);
      let v5 = load_block(data, offset + 80);
      let v6 = load_block(data, offset + 96);
      let v7 = load_block(data, offset + 112);
      offset += LONG_CHUNK_SIZE;

      let mut tmp1 = aesenc(v0, v2);
      let mut tmp2 = aesenc(v1, v3);
      tmp1 = aesenc(tmp1, v4);
      tmp2 = aesenc(tmp2, v5);
      tmp1 = aesenc(tmp1, v6);
      tmp2 = aesenc(tmp2, v7);

      t1 = add_bytes(t1, gx_key(0));
      t2 = add_bytes(t2, gx_key(4));
      lane1 = aesenclast(aesenc(tmp1, t1), lane1);
      lane2 = aesenclast(aesenc(tmp2, t2), lane2);
    }

    let len_vec = u32x4(len as u32);
    lane1 = add_bytes(lane1, len_vec);
    lane2 = add_bytes(lane2, len_vec);
    aesenc(lane1, lane2)
  }

  #[inline(always)]
  fn gx_finalize(hash: [u8; BLOCK_SIZE]) -> [u8; BLOCK_SIZE] {
    let hash = aesenc(hash, gx_key(0));
    let hash = aesenc(hash, gx_key(4));
    aesenclast(hash, gx_key(8))
  }

  #[inline(always)]
  fn gx_partial(data: &[u8], len: usize) -> [u8; BLOCK_SIZE] {
    debug_assert!(data.len() <= BLOCK_SIZE);
    let mut out = [0u8; BLOCK_SIZE];
    out[..data.len()].copy_from_slice(data);
    add_bytes(out, [len as u8; BLOCK_SIZE])
  }

  #[inline(always)]
  fn gx_seed(seed: u64) -> [u8; BLOCK_SIZE] {
    u64x2(seed, seed)
  }

  #[inline(always)]
  fn gx_key(offset: usize) -> [u8; BLOCK_SIZE] {
    debug_assert!(offset + 4 <= GX_KEYS.len());
    let mut out = [0u8; BLOCK_SIZE];
    out[..4].copy_from_slice(&GX_KEYS[offset].to_le_bytes());
    out[4..8].copy_from_slice(&GX_KEYS[offset + 1].to_le_bytes());
    out[8..12].copy_from_slice(&GX_KEYS[offset + 2].to_le_bytes());
    out[12..].copy_from_slice(&GX_KEYS[offset + 3].to_le_bytes());
    out
  }

  #[inline(always)]
  fn u32x4(v: u32) -> [u8; BLOCK_SIZE] {
    let bytes = v.to_le_bytes();
    [
      bytes[0], bytes[1], bytes[2], bytes[3], bytes[0], bytes[1], bytes[2], bytes[3], bytes[0], bytes[1], bytes[2],
      bytes[3], bytes[0], bytes[1], bytes[2], bytes[3],
    ]
  }

  #[inline(always)]
  fn seed_block(seed: u64, lane: u64) -> [u8; BLOCK_SIZE] {
    u64x2(seed_lo(seed, lane), seed_hi(seed, lane))
  }

  #[inline(always)]
  fn len_block(seed: u64, len: usize) -> [u8; BLOCK_SIZE] {
    u64x2(len_lo(seed, len), len_hi(seed, len))
  }

  #[inline(always)]
  fn load_block(data: &[u8], offset: usize) -> [u8; BLOCK_SIZE] {
    debug_assert!(offset + BLOCK_SIZE <= data.len());
    let mut out = [0u8; BLOCK_SIZE];
    out.copy_from_slice(&data[offset..offset + BLOCK_SIZE]);
    out
  }

  #[inline(always)]
  fn partial_block(data: &[u8], total_len: usize) -> [u8; BLOCK_SIZE] {
    debug_assert!(data.len() <= BLOCK_SIZE);
    let mut out = [0u8; BLOCK_SIZE];
    out[..data.len()].copy_from_slice(data);
    if data.len() < BLOCK_SIZE {
      out[data.len()] = 0x80;
    }
    let len = total_len as u64;
    let len_bytes = len.to_le_bytes();
    for i in 0..8 {
      out[8 + i] ^= len_bytes[i];
    }
    out
  }

  #[inline(always)]
  fn u64x2(lo: u64, hi: u64) -> [u8; BLOCK_SIZE] {
    let mut out = [0u8; BLOCK_SIZE];
    out[..8].copy_from_slice(&lo.to_le_bytes());
    out[8..].copy_from_slice(&hi.to_le_bytes());
    out
  }

  #[inline(always)]
  fn block_to_u128(block: [u8; BLOCK_SIZE]) -> u128 {
    u128::from_le_bytes(block)
  }

  #[inline(always)]
  fn xor(a: [u8; BLOCK_SIZE], b: [u8; BLOCK_SIZE]) -> [u8; BLOCK_SIZE] {
    let mut out = [0u8; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
      out[i] = a[i] ^ b[i];
    }
    out
  }

  #[inline(always)]
  fn add_bytes(a: [u8; BLOCK_SIZE], b: [u8; BLOCK_SIZE]) -> [u8; BLOCK_SIZE] {
    let mut out = [0u8; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
      out[i] = a[i].wrapping_add(b[i]);
    }
    out
  }

  #[inline(always)]
  fn rotate_bytes<const N: usize>(a: [u8; BLOCK_SIZE]) -> [u8; BLOCK_SIZE] {
    let mut out = [0u8; BLOCK_SIZE];
    let mut i = 0usize;
    while i < BLOCK_SIZE {
      out[i] = a[(i + N) & (BLOCK_SIZE - 1)];
      i += 1;
    }
    out
  }

  #[inline(always)]
  fn aesenc(state: [u8; BLOCK_SIZE], key: [u8; BLOCK_SIZE]) -> [u8; BLOCK_SIZE] {
    let sr = sub_shift(state);
    mix_columns(sr, key)
  }

  #[inline(always)]
  fn aesenclast(state: [u8; BLOCK_SIZE], key: [u8; BLOCK_SIZE]) -> [u8; BLOCK_SIZE] {
    let sr = sub_shift(state);
    xor(sr, key)
  }

  #[inline(always)]
  fn sub_shift(s: [u8; BLOCK_SIZE]) -> [u8; BLOCK_SIZE] {
    [
      SBOX[s[0] as usize],
      SBOX[s[5] as usize],
      SBOX[s[10] as usize],
      SBOX[s[15] as usize],
      SBOX[s[4] as usize],
      SBOX[s[9] as usize],
      SBOX[s[14] as usize],
      SBOX[s[3] as usize],
      SBOX[s[8] as usize],
      SBOX[s[13] as usize],
      SBOX[s[2] as usize],
      SBOX[s[7] as usize],
      SBOX[s[12] as usize],
      SBOX[s[1] as usize],
      SBOX[s[6] as usize],
      SBOX[s[11] as usize],
    ]
  }

  #[inline(always)]
  fn xtime(x: u8) -> u8 {
    (x << 1) ^ ((x >> 7).wrapping_mul(0x1b))
  }

  #[inline(always)]
  fn mix_columns(s: [u8; BLOCK_SIZE], key: [u8; BLOCK_SIZE]) -> [u8; BLOCK_SIZE] {
    let mut out = [0u8; BLOCK_SIZE];
    let mut i = 0usize;
    while i < BLOCK_SIZE {
      let a0 = s[i];
      let a1 = s[i + 1];
      let a2 = s[i + 2];
      let a3 = s[i + 3];
      let t = a0 ^ a1 ^ a2 ^ a3;
      out[i] = a0 ^ t ^ xtime(a0 ^ a1) ^ key[i];
      out[i + 1] = a1 ^ t ^ xtime(a1 ^ a2) ^ key[i + 1];
      out[i + 2] = a2 ^ t ^ xtime(a2 ^ a3) ^ key[i + 2];
      out[i + 3] = a3 ^ t ^ xtime(a3 ^ a0) ^ key[i + 3];
      i += 4;
    }
    out
  }
}

#[cfg(test)]
mod tests {
  use alloc::vec::Vec;

  use super::{AesHash64, AesHash128, HASH64_GX_LONG_THRESHOLD};

  #[test]
  fn default_seed_matches_seed_zero() {
    for len in [
      0usize,
      1,
      15,
      16,
      17,
      31,
      32,
      33,
      63,
      64,
      65,
      255,
      1024,
      HASH64_GX_LONG_THRESHOLD,
    ] {
      let data = bytes(len);
      assert_eq!(AesHash64::hash(&data), AesHash64::hash_with_seed(0, &data));
      assert_eq!(AesHash128::hash(&data), AesHash128::hash_with_seed(0, &data));
    }
  }

  #[test]
  fn hash64_seed_changes_output() {
    for len in [0usize, 1, 16, 17, 32, 64, 65, 4096, HASH64_GX_LONG_THRESHOLD] {
      let data = bytes(len);
      assert_ne!(
        AesHash64::hash_with_seed(0, &data),
        AesHash64::hash_with_seed(42, &data)
      );
    }
  }

  #[test]
  fn seed_changes_output() {
    let data = bytes(256);
    assert_ne!(
      AesHash128::hash_with_seed(0, &data),
      AesHash128::hash_with_seed(1, &data)
    );
  }

  #[test]
  fn every_byte_affects_hash() {
    for len in 1..192usize {
      assert_every_byte_affects_hash(len);
    }
    for len in [256usize, 1024] {
      assert_every_byte_affects_hash(len);
    }
  }

  fn assert_every_byte_affects_hash(len: usize) {
    let data = bytes(len);
    let reference64 = AesHash64::hash(&data);
    let reference = AesHash128::hash(&data);
    for i in 0..len {
      let mut changed = data.clone();
      changed[i] ^= 0xa5;
      assert_ne!(
        reference64,
        AesHash64::hash(&changed),
        "byte {i} not consumed by AesHash64 for len {len}"
      );
      assert_ne!(
        reference,
        AesHash128::hash(&changed),
        "byte {i} not consumed for len {len}"
      );
    }
  }

  #[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"),
    not(feature = "portable-only"),
    not(miri)
  ))]
  fn portable_hash64(seed: u64, data: &[u8]) -> u64 {
    if data.is_empty() && seed == 0 {
      return super::RAPID_EMPTY_HASH_SEED_0;
    }
    if data.len() <= super::BLOCK_SIZE {
      return super::short_hash64(seed, data);
    }
    if data.len() == 2 * super::BLOCK_SIZE {
      return super::portable::hash64_32(seed, data);
    }
    if data.len() == 4 * super::BLOCK_SIZE {
      return super::portable::hash64_64(seed, data);
    }
    if super::hash64_low_lane_len(data.len()) {
      return super::portable::hash64_low(seed, data);
    }
    super::fold_hash128(super::portable::hash128(seed, data))
  }

  #[test]
  #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
  fn aarch64_aes_matches_portable_when_available() {
    if !crate::platform::caps().has(crate::platform::caps::aarch64::AES | crate::platform::caps::aarch64::NEON) {
      return;
    }
    for len in [
      0usize,
      1,
      15,
      16,
      17,
      31,
      32,
      33,
      63,
      64,
      65,
      256,
      511,
      4096,
      HASH64_GX_LONG_THRESHOLD,
    ] {
      let data = bytes(len);
      let portable64 = portable_hash64(0x1234_5678_9abc_def0, &data);
      let portable = super::portable::hash128(0x1234_5678_9abc_def0, &data);
      assert_eq!(
        AesHash64::hash_with_seed(0x1234_5678_9abc_def0, &data),
        portable64,
        "64-bit len {len}"
      );
      // SAFETY: runtime check above confirms AES-CE + NEON.
      let accelerated = unsafe { super::aarch64::hash128(0x1234_5678_9abc_def0, &data) };
      assert_eq!(accelerated, portable, "len {len}");
    }
  }

  #[test]
  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "portable-only"),
    not(miri)
  ))]
  fn x86_aes_matches_portable_when_available() {
    if !crate::platform::caps().has(crate::platform::caps::x86::AESNI | crate::platform::caps::x86::SSE2) {
      return;
    }
    for len in [
      0usize,
      1,
      15,
      16,
      17,
      31,
      32,
      33,
      63,
      64,
      65,
      256,
      511,
      4096,
      HASH64_GX_LONG_THRESHOLD,
    ] {
      let data = bytes(len);
      let portable64 = portable_hash64(0x1234_5678_9abc_def0, &data);
      let portable = super::portable::hash128(0x1234_5678_9abc_def0, &data);
      assert_eq!(
        AesHash64::hash_with_seed(0x1234_5678_9abc_def0, &data),
        portable64,
        "64-bit len {len}"
      );
      // SAFETY: runtime check above confirms AES-NI + SSE2.
      let accelerated = unsafe { super::x86::hash128(0x1234_5678_9abc_def0, &data) };
      assert_eq!(accelerated, portable, "len {len}");
    }
  }

  fn bytes(len: usize) -> Vec<u8> {
    let mut state = len as u64 ^ 0xa076_1d64_78bd_642f;
    (0..len)
      .map(|_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 56) as u8
      })
      .collect()
  }
}
