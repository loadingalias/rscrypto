//! rapidhash V3 (**NOT CRYPTO**).
//!
//! Portable scalar implementation (no SIMD yet).
//!
//! This implements the stable, C++-compatible V3 algorithm with the default secret set.
//! The `FastHash` seed is interpreted as the C++ seed value.

#![allow(clippy::indexing_slicing)] // Tight block parsing

use traits::FastHash;

#[doc(hidden)]
pub mod dispatch;
#[doc(hidden)]
pub mod dispatch_tables;
pub(crate) mod kernels;

#[derive(Clone, Default)]
pub struct RapidHash64;

#[derive(Clone, Default)]
pub struct RapidHash128;

// rapidhash v3 default secrets (C++ compatible)
const DEFAULT_SECRETS: [u64; 7] = [
  0x2d35_8dcc_aa6c_78a5,
  0x8bb8_4b93_962e_acc9,
  0x4b33_a62e_d433_d4a3,
  0x4d5a_2da5_1de1_aa47,
  0xa076_1d64_78bd_642f,
  0xe703_7ed1_a0b4_28db,
  0x90ed_1765_281c_388c,
];

#[inline(always)]
fn read_u32_le(input: &[u8], offset: usize) -> u32 {
  debug_assert!(offset + 4 <= input.len());
  // SAFETY: caller ensures `offset + 4 <= input.len()`, and `read_unaligned` supports unaligned
  // loads.
  let v = unsafe { core::ptr::read_unaligned(input.as_ptr().add(offset) as *const u32) };
  u32::from_le(v)
}

#[inline(always)]
fn read_u64_le(input: &[u8], offset: usize) -> u64 {
  debug_assert!(offset + 8 <= input.len());
  // SAFETY: caller ensures `offset + 8 <= input.len()`, and `read_unaligned` supports unaligned
  // loads.
  let v = unsafe { core::ptr::read_unaligned(input.as_ptr().add(offset) as *const u64) };
  u64::from_le(v)
}

#[inline(always)]
const fn rapid_mum(a: u64, b: u64) -> (u64, u64) {
  let r = (a as u128).wrapping_mul(b as u128);
  (r as u64, (r >> 64) as u64)
}

#[inline(always)]
const fn rapid_mix(a: u64, b: u64) -> u64 {
  let r = (a as u128).wrapping_mul(b as u128);
  (r as u64) ^ ((r >> 64) as u64)
}

#[inline(always)]
const fn rapidhash_seed_cpp(seed: u64) -> u64 {
  seed ^ rapid_mix(seed ^ DEFAULT_SECRETS[2], DEFAULT_SECRETS[1])
}

#[inline(always)]
fn rapidhash_finish(a: u64, b: u64, remainder: u64) -> u64 {
  rapid_mix(a ^ 0xaaaa_aaaa_aaaa_aaaa, b ^ DEFAULT_SECRETS[1] ^ remainder)
}

#[inline(always)]
fn rapidhash_v3_with_seed(data: &[u8], seed: u64) -> u64 {
  rapidhash_v3_core(data, rapidhash_seed_cpp(seed))
}

#[inline(always)]
fn rapidhash_v3_core(data: &[u8], seed: u64) -> u64 {
  if data.len() <= 16 {
    rapidhash_v3_core_small(data, seed)
  } else {
    rapidhash_v3_core_large(data, seed)
  }
}

#[inline(always)]
fn rapidhash_v3_core_small(data: &[u8], mut seed: u64) -> u64 {
  let mut a = 0u64;
  let mut b = 0u64;

  if data.len() >= 4 {
    seed ^= data.len() as u64;
    if data.len() >= 8 {
      let plast = data.len() - 8;
      a ^= read_u64_le(data, 0);
      b ^= read_u64_le(data, plast);
    } else {
      let plast = data.len() - 4;
      a ^= read_u32_le(data, 0) as u64;
      b ^= read_u32_le(data, plast) as u64;
    }
  } else if !data.is_empty() {
    a ^= ((data[0] as u64) << 45) | (data[data.len() - 1] as u64);
    b ^= data[data.len() >> 1] as u64;
  }

  let remainder = data.len() as u64;
  a ^= DEFAULT_SECRETS[1];
  b ^= seed;

  (a, b) = rapid_mum(a, b);
  rapidhash_finish(a, b, remainder)
}

#[inline]
fn rapidhash_v3_core_large(data: &[u8], mut seed: u64) -> u64 {
  debug_assert!(data.len() > 16);

  let mut a = 0u64;
  let mut b = 0u64;
  let mut slice = data;

  if slice.len() > 112 {
    let mut see1 = seed;
    let mut see2 = seed;
    let mut see3 = seed;
    let mut see4 = seed;
    let mut see5 = seed;
    let mut see6 = seed;

    while slice.len() > 224 {
      seed = rapid_mix(read_u64_le(slice, 0) ^ DEFAULT_SECRETS[0], read_u64_le(slice, 8) ^ seed);
      see1 = rapid_mix(
        read_u64_le(slice, 16) ^ DEFAULT_SECRETS[1],
        read_u64_le(slice, 24) ^ see1,
      );
      see2 = rapid_mix(
        read_u64_le(slice, 32) ^ DEFAULT_SECRETS[2],
        read_u64_le(slice, 40) ^ see2,
      );
      see3 = rapid_mix(
        read_u64_le(slice, 48) ^ DEFAULT_SECRETS[3],
        read_u64_le(slice, 56) ^ see3,
      );
      see4 = rapid_mix(
        read_u64_le(slice, 64) ^ DEFAULT_SECRETS[4],
        read_u64_le(slice, 72) ^ see4,
      );
      see5 = rapid_mix(
        read_u64_le(slice, 80) ^ DEFAULT_SECRETS[5],
        read_u64_le(slice, 88) ^ see5,
      );
      see6 = rapid_mix(
        read_u64_le(slice, 96) ^ DEFAULT_SECRETS[6],
        read_u64_le(slice, 104) ^ see6,
      );

      seed = rapid_mix(
        read_u64_le(slice, 112) ^ DEFAULT_SECRETS[0],
        read_u64_le(slice, 120) ^ seed,
      );
      see1 = rapid_mix(
        read_u64_le(slice, 128) ^ DEFAULT_SECRETS[1],
        read_u64_le(slice, 136) ^ see1,
      );
      see2 = rapid_mix(
        read_u64_le(slice, 144) ^ DEFAULT_SECRETS[2],
        read_u64_le(slice, 152) ^ see2,
      );
      see3 = rapid_mix(
        read_u64_le(slice, 160) ^ DEFAULT_SECRETS[3],
        read_u64_le(slice, 168) ^ see3,
      );
      see4 = rapid_mix(
        read_u64_le(slice, 176) ^ DEFAULT_SECRETS[4],
        read_u64_le(slice, 184) ^ see4,
      );
      see5 = rapid_mix(
        read_u64_le(slice, 192) ^ DEFAULT_SECRETS[5],
        read_u64_le(slice, 200) ^ see5,
      );
      see6 = rapid_mix(
        read_u64_le(slice, 208) ^ DEFAULT_SECRETS[6],
        read_u64_le(slice, 216) ^ see6,
      );

      slice = &slice[224..];
    }

    if slice.len() > 112 {
      seed = rapid_mix(read_u64_le(slice, 0) ^ DEFAULT_SECRETS[0], read_u64_le(slice, 8) ^ seed);
      see1 = rapid_mix(
        read_u64_le(slice, 16) ^ DEFAULT_SECRETS[1],
        read_u64_le(slice, 24) ^ see1,
      );
      see2 = rapid_mix(
        read_u64_le(slice, 32) ^ DEFAULT_SECRETS[2],
        read_u64_le(slice, 40) ^ see2,
      );
      see3 = rapid_mix(
        read_u64_le(slice, 48) ^ DEFAULT_SECRETS[3],
        read_u64_le(slice, 56) ^ see3,
      );
      see4 = rapid_mix(
        read_u64_le(slice, 64) ^ DEFAULT_SECRETS[4],
        read_u64_le(slice, 72) ^ see4,
      );
      see5 = rapid_mix(
        read_u64_le(slice, 80) ^ DEFAULT_SECRETS[5],
        read_u64_le(slice, 88) ^ see5,
      );
      see6 = rapid_mix(
        read_u64_le(slice, 96) ^ DEFAULT_SECRETS[6],
        read_u64_le(slice, 104) ^ see6,
      );
      slice = &slice[112..];
    }

    seed ^= see1;
    see2 ^= see3;
    see4 ^= see5;
    seed ^= see6;
    see2 ^= see4;
    seed ^= see2;
  }

  if slice.len() > 16 {
    seed = rapid_mix(read_u64_le(slice, 0) ^ DEFAULT_SECRETS[2], read_u64_le(slice, 8) ^ seed);
    if slice.len() > 32 {
      seed = rapid_mix(
        read_u64_le(slice, 16) ^ DEFAULT_SECRETS[2],
        read_u64_le(slice, 24) ^ seed,
      );
      if slice.len() > 48 {
        seed = rapid_mix(
          read_u64_le(slice, 32) ^ DEFAULT_SECRETS[1],
          read_u64_le(slice, 40) ^ seed,
        );
        if slice.len() > 64 {
          seed = rapid_mix(
            read_u64_le(slice, 48) ^ DEFAULT_SECRETS[1],
            read_u64_le(slice, 56) ^ seed,
          );
          if slice.len() > 80 {
            seed = rapid_mix(
              read_u64_le(slice, 64) ^ DEFAULT_SECRETS[2],
              read_u64_le(slice, 72) ^ seed,
            );
            if slice.len() > 96 {
              seed = rapid_mix(
                read_u64_le(slice, 80) ^ DEFAULT_SECRETS[1],
                read_u64_le(slice, 88) ^ seed,
              );
            }
          }
        }
      }
    }
  }

  let remainder = slice.len() as u64;
  a ^= read_u64_le(data, data.len() - 16) ^ remainder;
  b ^= read_u64_le(data, data.len() - 8);

  a ^= DEFAULT_SECRETS[1];
  b ^= seed;

  (a, b) = rapid_mum(a, b);
  rapidhash_finish(a, b, remainder)
}

impl FastHash for RapidHash64 {
  const OUTPUT_SIZE: usize = 8;
  type Output = u64;
  type Seed = u64;

  #[inline]
  fn hash_with_seed(seed: Self::Seed, data: &[u8]) -> Self::Output {
    dispatch::hash64_with_seed(seed, data)
  }
}

impl FastHash for RapidHash128 {
  const OUTPUT_SIZE: usize = 16;
  type Output = u128;
  type Seed = u64;

  #[inline]
  fn hash_with_seed(seed: Self::Seed, data: &[u8]) -> Self::Output {
    dispatch::hash128_with_seed(seed, data)
  }
}

#[cfg(test)]
mod tests {
  use proptest::prelude::*;

  use super::{RapidHash64, RapidHash128};

  #[test]
  fn smoke_empty_matches_oracle() {
    let seed = 0u64;
    let ours = <RapidHash64 as traits::FastHash>::hash_with_seed(seed, b"");
    let secrets = rapidhash::v3::RapidSecrets::seed_cpp(seed);
    let theirs = rapidhash::v3::rapidhash_v3_seeded(b"", &secrets);
    assert_eq!(ours, theirs);
  }

  proptest! {
    #[test]
    fn rapidhash_v3_64_matches_oracle(seed in any::<u64>(), data in proptest::collection::vec(any::<u8>(), 0..2048)) {
      let ours = <RapidHash64 as traits::FastHash>::hash_with_seed(seed, &data);
      let secrets = rapidhash::v3::RapidSecrets::seed_cpp(seed);
      let theirs = rapidhash::v3::rapidhash_v3_seeded(&data, &secrets);
      prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn rapidhash128_is_two_independent_64bit_hashes(seed in any::<u64>(), data in proptest::collection::vec(any::<u8>(), 0..2048)) {
      let out = <RapidHash128 as traits::FastHash>::hash_with_seed(seed, &data);
      let lo = out as u64;
      let hi = (out >> 64) as u64;

      prop_assert_eq!(lo, <RapidHash64 as traits::FastHash>::hash_with_seed(seed, &data));
      prop_assert_eq!(hi, <RapidHash64 as traits::FastHash>::hash_with_seed(seed ^ 0x9E37_79B9_7F4A_7C15, &data));
    }
  }
}
