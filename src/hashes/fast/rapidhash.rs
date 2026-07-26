//! RapidHash V3 (**NOT CRYPTO**).
//!
//! The portable, safe Rust implementation is the sole semantic authority.

#![allow(clippy::indexing_slicing)]

use crate::traits::FastHash;

mod stream;

pub use stream::{RapidHasher, RapidRandomState, RapidSeededState, RapidStreamHasher};

const DEFAULT_SECRETS: [u64; 7] = [
  0x2d35_8dcc_aa6c_78a5,
  0x8bb8_4b93_962e_acc9,
  0x4b33_a62e_d433_d4a3,
  0x4d5a_2da5_1de1_aa47,
  0xa076_1d64_78bd_642f,
  0xe703_7ed1_a0b4_28db,
  0x90ed_1765_281c_388c,
];

struct RapidSecrets {
  seed: u64,
  words: [u64; 7],
}

impl RapidSecrets {
  #[inline(always)]
  const fn cpp(seed: u64) -> Self {
    Self {
      seed: rapidhash_seed_cpp(seed),
      words: DEFAULT_SECRETS,
    }
  }

  #[inline]
  const fn derived(seed: u64, word0: u64) -> Self {
    let mut words = [0; 7];
    words[0] = word0;
    words[1] = premix_seed(words[0], 1);
    words[2] = premix_seed(words[1], 2);
    words[3] = premix_seed(words[2], 3);
    words[4] = premix_seed(words[3], 4);
    words[5] = premix_seed(words[4], 5);
    words[6] = premix_seed(words[5], 6);
    Self { seed, words }
  }
}

const DEFAULT_RAPID_SECRETS: RapidSecrets = RapidSecrets::cpp(0);

/// Standard, portable RapidHash V3 with a 64-bit output.
#[derive(Clone, Debug, Default)]
pub struct RapidHash64;

impl RapidHash64 {
  /// Hash `data` with the C++-compatible default seed.
  #[inline]
  #[must_use]
  pub const fn hash(data: &[u8]) -> u64 {
    rapidhash_core(data, DEFAULT_RAPID_SECRETS.seed, &DEFAULT_RAPID_SECRETS.words)
  }

  /// Hash `data` with the C++-compatible V3 seed schedule.
  #[inline]
  #[must_use]
  pub const fn hash_with_seed(seed: u64, data: &[u8]) -> u64 {
    let state = RapidSecrets::cpp(seed);
    rapidhash_core(data, state.seed, &state.words)
  }
}

impl FastHash for RapidHash64 {
  const OUTPUT_SIZE: usize = 8;
  type Output = u64;
  type Seed = u64;

  #[inline]
  fn hash(data: &[u8]) -> Self::Output {
    Self::hash(data)
  }

  #[inline]
  fn hash_with_seed(seed: Self::Seed, data: &[u8]) -> Self::Output {
    Self::hash_with_seed(seed, data)
  }
}

#[inline(always)]
const fn read_u32_le(input: &[u8], offset: usize) -> u32 {
  let (_, tail) = input.split_at(offset);
  let Some(bytes) = tail.first_chunk::<4>() else {
    panic!("RapidHash u32 read exceeds input");
  };
  u32::from_le_bytes(*bytes)
}

#[inline(always)]
const fn read_u64_le(input: &[u8], offset: usize) -> u64 {
  let (_, tail) = input.split_at(offset);
  let Some(bytes) = tail.first_chunk::<8>() else {
    panic!("RapidHash u64 read exceeds input");
  };
  u64::from_le_bytes(*bytes)
}

#[inline(always)]
const fn rapid_mum(a: u64, b: u64) -> (u64, u64) {
  let product = (a as u128).wrapping_mul(b as u128);
  (product as u64, (product >> 64) as u64)
}

#[inline(always)]
const fn rapid_mix(a: u64, b: u64) -> u64 {
  let product = (a as u128).wrapping_mul(b as u128);
  (product as u64) ^ ((product >> 64) as u64)
}

#[inline(always)]
const fn rapidhash_seed_cpp(seed: u64) -> u64 {
  seed ^ rapid_mix(seed ^ DEFAULT_SECRETS[2], DEFAULT_SECRETS[1])
}

#[inline]
const fn premix_seed(mut seed: u64, index: usize) -> u64 {
  seed ^= rapid_mix(seed ^ DEFAULT_SECRETS[2], DEFAULT_SECRETS[index]);

  if seed & (0xffff << 48) == 0 {
    seed |= 1 << 63;
  }
  if seed & (0xffff << 24) == 0 {
    seed |= 1 << 31;
  }
  if seed & 0xffff == 0 {
    seed |= 1;
  }
  seed
}

#[inline(always)]
const fn rapidhash_finish(a: u64, b: u64, remainder: u64, secrets: &[u64; 7]) -> u64 {
  rapid_mix(a ^ 0xaaaa_aaaa_aaaa_aaaa, b ^ secrets[1] ^ remainder)
}

#[inline(always)]
const fn rapidhash_core(data: &[u8], mut seed: u64, secrets: &[u64; 7]) -> u64 {
  let mut a = 0u64;
  let mut b = 0u64;

  if data.len() <= 16 {
    if data.len() >= 4 {
      seed ^= data.len() as u64;
      if data.len() >= 8 {
        a = read_u64_le(data, 0);
        b = read_u64_le(data, data.len().strict_sub(8));
      } else {
        a = read_u32_le(data, 0) as u64;
        b = read_u32_le(data, data.len().strict_sub(4)) as u64;
      }
    } else if !data.is_empty() {
      a = ((data[0] as u64) << 45) | data[data.len().strict_sub(1)] as u64;
      b = data[data.len() >> 1] as u64;
    }

    a ^= secrets[1];
    b ^= seed;
    (a, b) = rapid_mum(a, b);
    return rapidhash_finish(a, b, data.len() as u64, secrets);
  }

  if data.len() <= 112 {
    return rapidhash_tail(data, 0, seed, secrets);
  }

  rapidhash_core_large(data, seed, secrets)
}

// Keep the large-input schedule out of short-input callers.
#[inline(never)]
const fn rapidhash_core_large(data: &[u8], mut seed: u64, secrets: &[u64; 7]) -> u64 {
  let mut offset = 0usize;
  if data.len() > 112 {
    let mut see1 = seed;
    let mut see2 = seed;
    let mut see3 = seed;
    let mut see4 = seed;
    let mut see5 = seed;
    let mut see6 = seed;

    while data.len().strict_sub(offset) > 224 {
      // Validate the span once so the fixed-size safe reads need no per-load bounds checks.
      let (_, remaining) = data.split_at(offset);
      let (block, _) = remaining.split_at(224);
      let Some(first) = block.first_chunk::<112>() else {
        panic!("RapidHash block is shorter than 112 bytes");
      };
      let Some(second) = block.last_chunk::<112>() else {
        panic!("RapidHash block is shorter than 224 bytes");
      };
      seed = rapid_mix(read_u64_le(first, 0) ^ secrets[0], read_u64_le(first, 8) ^ seed);
      see1 = rapid_mix(read_u64_le(first, 16) ^ secrets[1], read_u64_le(first, 24) ^ see1);
      see2 = rapid_mix(read_u64_le(first, 32) ^ secrets[2], read_u64_le(first, 40) ^ see2);
      see3 = rapid_mix(read_u64_le(first, 48) ^ secrets[3], read_u64_le(first, 56) ^ see3);
      see4 = rapid_mix(read_u64_le(first, 64) ^ secrets[4], read_u64_le(first, 72) ^ see4);
      see5 = rapid_mix(read_u64_le(first, 80) ^ secrets[5], read_u64_le(first, 88) ^ see5);
      see6 = rapid_mix(read_u64_le(first, 96) ^ secrets[6], read_u64_le(first, 104) ^ see6);

      seed = rapid_mix(read_u64_le(second, 0) ^ secrets[0], read_u64_le(second, 8) ^ seed);
      see1 = rapid_mix(read_u64_le(second, 16) ^ secrets[1], read_u64_le(second, 24) ^ see1);
      see2 = rapid_mix(read_u64_le(second, 32) ^ secrets[2], read_u64_le(second, 40) ^ see2);
      see3 = rapid_mix(read_u64_le(second, 48) ^ secrets[3], read_u64_le(second, 56) ^ see3);
      see4 = rapid_mix(read_u64_le(second, 64) ^ secrets[4], read_u64_le(second, 72) ^ see4);
      see5 = rapid_mix(read_u64_le(second, 80) ^ secrets[5], read_u64_le(second, 88) ^ see5);
      see6 = rapid_mix(read_u64_le(second, 96) ^ secrets[6], read_u64_le(second, 104) ^ see6);
      offset = offset.strict_add(224);
    }

    if data.len().strict_sub(offset) > 112 {
      let (_, remaining) = data.split_at(offset);
      let Some(block) = remaining.first_chunk::<112>() else {
        panic!("RapidHash block is shorter than 112 bytes");
      };
      seed = rapid_mix(read_u64_le(block, 0) ^ secrets[0], read_u64_le(block, 8) ^ seed);
      see1 = rapid_mix(read_u64_le(block, 16) ^ secrets[1], read_u64_le(block, 24) ^ see1);
      see2 = rapid_mix(read_u64_le(block, 32) ^ secrets[2], read_u64_le(block, 40) ^ see2);
      see3 = rapid_mix(read_u64_le(block, 48) ^ secrets[3], read_u64_le(block, 56) ^ see3);
      see4 = rapid_mix(read_u64_le(block, 64) ^ secrets[4], read_u64_le(block, 72) ^ see4);
      see5 = rapid_mix(read_u64_le(block, 80) ^ secrets[5], read_u64_le(block, 88) ^ see5);
      see6 = rapid_mix(read_u64_le(block, 96) ^ secrets[6], read_u64_le(block, 104) ^ see6);
      offset = offset.strict_add(112);
    }

    seed ^= see1;
    see2 ^= see3;
    see4 ^= see5;
    seed ^= see6;
    see2 ^= see4;
    seed ^= see2;
  }

  rapidhash_tail(data, offset, seed, secrets)
}

#[inline(always)]
const fn rapidhash_tail(data: &[u8], offset: usize, mut seed: u64, secrets: &[u64; 7]) -> u64 {
  let remainder = data.len().strict_sub(offset);
  if remainder > 16 {
    seed = rapid_mix(
      read_u64_le(data, offset) ^ secrets[2],
      read_u64_le(data, offset.strict_add(8)) ^ seed,
    );
    if remainder > 32 {
      seed = rapid_mix(
        read_u64_le(data, offset.strict_add(16)) ^ secrets[2],
        read_u64_le(data, offset.strict_add(24)) ^ seed,
      );
      if remainder > 48 {
        seed = rapid_mix(
          read_u64_le(data, offset.strict_add(32)) ^ secrets[1],
          read_u64_le(data, offset.strict_add(40)) ^ seed,
        );
        if remainder > 64 {
          seed = rapid_mix(
            read_u64_le(data, offset.strict_add(48)) ^ secrets[1],
            read_u64_le(data, offset.strict_add(56)) ^ seed,
          );
          if remainder > 80 {
            seed = rapid_mix(
              read_u64_le(data, offset.strict_add(64)) ^ secrets[2],
              read_u64_le(data, offset.strict_add(72)) ^ seed,
            );
            if remainder > 96 {
              seed = rapid_mix(
                read_u64_le(data, offset.strict_add(80)) ^ secrets[1],
                read_u64_le(data, offset.strict_add(88)) ^ seed,
              );
            }
          }
        }
      }
    }
  }

  let mut a = read_u64_le(data, data.len().strict_sub(16)) ^ remainder as u64;
  let mut b = read_u64_le(data, data.len().strict_sub(8));
  a ^= secrets[1];
  b ^= seed;
  (a, b) = rapid_mum(a, b);
  rapidhash_finish(a, b, remainder as u64, secrets)
}

#[cfg(test)]
mod tests {
  use alloc::vec::Vec;

  #[cfg(not(miri))]
  use proptest::prelude::*;

  use super::RapidHash64;

  const CONST_EMPTY: u64 = RapidHash64::hash(b"");
  const CONST_SEEDED: u64 = RapidHash64::hash_with_seed(42, b"const RapidHash");

  fn data(len: usize) -> Vec<u8> {
    (0..len).map(|i| i.wrapping_mul(131).wrapping_add(17) as u8).collect()
  }

  fn reference(seed: u64, data: &[u8]) -> u64 {
    let secrets = rapidhash::v3::RapidSecrets::seed_cpp(seed);
    rapidhash::v3::rapidhash_v3_seeded(data, &secrets)
  }

  #[test]
  fn const_hashes_match_runtime_and_reference() {
    assert_eq!(CONST_EMPTY, reference(0, b""));
    assert_eq!(CONST_SEEDED, reference(42, b"const RapidHash"));
  }

  #[test]
  fn boundary_lengths_match_reference() {
    #[cfg(not(miri))]
    let lengths: Vec<usize> = (0..=512).collect();
    #[cfg(miri)]
    let lengths = [0, 1, 3, 4, 7, 8, 16, 17, 32, 48, 64, 80, 96, 112, 113, 224, 225, 512];

    for seed in [0, 1, u64::MAX, 0x243f_6a88_85a3_08d3] {
      for len in lengths.iter().copied() {
        let input = data(len);
        assert_eq!(
          RapidHash64::hash_with_seed(seed, &input),
          reference(seed, &input),
          "seed={seed:#x}, len={len}"
        );
      }
    }
  }

  #[cfg(not(miri))]
  proptest! {
    #[test]
    fn matches_reference(seed in any::<u64>(), data in proptest::collection::vec(any::<u8>(), 0..8192)) {
      prop_assert_eq!(RapidHash64::hash_with_seed(seed, &data), reference(seed, &data));
    }
  }
}
