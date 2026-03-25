//! rapidhash V3 (**NOT CRYPTO**).
//!
//! Portable scalar implementation.
//!
//! Two variants are provided:
//!
//! - **`RapidHash64` / `RapidHash128`** — the standard V3 algorithm with avalanche finisher (extra
//!   `rapid_mix`). C++-compatible output. Use when you need stable, cross-language hash values.
//!
//! - **`RapidHashFast64` / `RapidHashFast128`** — V3 core with avalanche disabled. Saves one
//!   128-bit multiply at finalization. Use when you only need a fast in-process hash (e.g. hash
//!   maps) and don't need C++ compatibility.

#![allow(clippy::indexing_slicing)] // Tight block parsing

use crate::traits::FastHash;

#[doc(hidden)]
pub mod dispatch;
#[doc(hidden)]
pub mod dispatch_tables;
pub(crate) mod kernels;

/// Standard V3 rapidhash (64-bit) with avalanche finisher.
#[derive(Clone, Debug, Default)]
pub struct RapidHash64;

/// Standard V3 rapidhash (128-bit) with avalanche finisher.
#[derive(Clone, Debug, Default)]
pub struct RapidHash128;

/// Fast V3 rapidhash (64-bit) — avalanche disabled for maximum throughput.
#[derive(Clone, Debug, Default)]
pub struct RapidHashFast64;

/// Fast V3 rapidhash (128-bit) — avalanche disabled for maximum throughput.
#[derive(Clone, Debug, Default)]
pub struct RapidHashFast128;

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

/// Hint that `cond` is likely true. Uses `#[cold]` to nudge branch layout.
#[inline(always)]
fn likely(cond: bool) -> bool {
  if !cond {
    cold_path();
  }
  cond
}

/// Hint that `cond` is likely false.
#[inline(always)]
fn unlikely(cond: bool) -> bool {
  if cond {
    cold_path();
  }
  cond
}

#[cold]
#[inline(always)]
fn cold_path() {}

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
  rapidhash_core::<true>(data, rapidhash_seed_cpp(seed))
}

#[inline(always)]
fn rapidhash_fast_with_seed(data: &[u8], seed: u64) -> u64 {
  rapidhash_fast_core(data, rapidhash_seed_cpp(seed))
}

#[inline(always)]
fn rapidhash_core<const AVALANCHE: bool>(data: &[u8], mut seed: u64) -> u64 {
  let mut a = 0u64;
  let mut b = 0u64;

  if likely(data.len() <= 16) {
    // Small path: 0-16 bytes. Fully inline for minimum overhead.
    if data.len() >= 4 {
      seed ^= data.len() as u64;
      if data.len() >= 8 {
        a = read_u64_le(data, 0);
        b = read_u64_le(data, data.len() - 8);
      } else {
        a = read_u32_le(data, 0) as u64;
        b = read_u32_le(data, data.len() - 4) as u64;
      }
    } else if !data.is_empty() {
      a = ((data[0] as u64) << 45) | (data[data.len() - 1] as u64);
      b = data[data.len() >> 1] as u64;
    }
  } else {
    // Medium/large path: >16 bytes. The 7-stream bulk loop lives in a
    // separate non-inlined function to keep the entry point lean.
    // SAFETY: we just verified data.len() > 16.
    unsafe { return rapidhash_core_large::<AVALANCHE>(data, seed) };
  }

  let remainder = data.len() as u64;
  a ^= DEFAULT_SECRETS[1];
  b ^= seed;
  (a, b) = rapid_mum(a, b);
  rapidhash_final::<AVALANCHE>(a, b, remainder)
}

#[inline(always)]
fn rapidhash_final<const AVALANCHE: bool>(a: u64, b: u64, remainder: u64) -> u64 {
  if AVALANCHE {
    rapidhash_finish(a, b, remainder)
  } else {
    a ^ b
  }
}

/// Handles inputs >16 bytes. Kept as a separate function so the small-path
/// entry point stays lean for inlining. LLVM decides whether to inline this.
///
/// # Safety
///
/// Caller must guarantee `data.len() > 16`.
#[inline]
unsafe fn rapidhash_core_large<const AVALANCHE: bool>(data: &[u8], mut seed: u64) -> u64 {
  // SAFETY: caller guarantees data.len() > 16. This eliminates redundant
  // bounds checks in the tail-read section below.
  unsafe { core::hint::assert_unchecked(data.len() > 16) };

  let mut a = 0u64;
  let mut b = 0u64;
  let mut slice = data;

  if unlikely(slice.len() > 112) {
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

      let (_, rest) = slice.split_at(224);
      slice = rest;
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
      let (_, rest) = slice.split_at(112);
      slice = rest;
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
  rapidhash_final::<AVALANCHE>(a, b, remainder)
}

// ---------------------------------------------------------------------------
// Inner-algorithm core for RapidHashFast64 / RapidHashFast128
// ---------------------------------------------------------------------------
//
// Distinct from V3: size-tuned dispatch (3-stream mid-range, 7-stream large),
// cold-path separation for codegen quality, and inner-module finalization.
// Output is NOT V3-compatible — use RapidHash64/128 for cross-language hashes.

/// Inputs above this use the 7-stream large kernel. Below it, the 3-stream
/// mid-range loop has lower register pressure and avoids stack spills.
const FAST_COLD_CUTOFF: usize = 400;

#[inline(always)]
fn rapidhash_fast_finish(a: u64, b: u64, seed: u64) -> u64 {
  rapid_mix(a ^ DEFAULT_SECRETS[0], b ^ seed)
}

/// Inner-algorithm entry point for the Fast variants.
///
/// 0–16B path is always inlined. Larger inputs dispatch to `#[cold]` paths
/// that are never inlined, keeping this entry point lean for the common case.
#[inline(always)]
fn rapidhash_fast_core(data: &[u8], mut seed: u64) -> u64 {
  if likely(data.len() <= 16) {
    let mut a = 0u64;
    let mut b = 0u64;

    if likely(data.len() >= 8) {
      a = read_u64_le(data, 0);
      b = read_u64_le(data, data.len() - 8);
    } else if likely(data.len() >= 4) {
      a = read_u32_le(data, 0) as u64;
      b = read_u32_le(data, data.len() - 4) as u64;
    } else if !data.is_empty() {
      a = ((data[0] as u64) << 45) | data[data.len() - 1] as u64;
      b = data[data.len() >> 1] as u64;
    }

    seed = seed.wrapping_add(data.len() as u64);
    rapidhash_fast_finish(a, b, seed)
  } else {
    // SAFETY: data.len() > 16 verified above.
    unsafe { rapidhash_fast_core_medium(data, seed) }
  }
}

/// Final ≤48B tail processing shared by medium and large paths.
///
/// Reads up to two 16B pairs from `slice` (the unprocessed remainder), then
/// reads the final 16B from `data` (the original input) for overlap handling.
///
/// # Safety
///
/// Caller must guarantee `data.len() > 16`.
#[inline(always)]
unsafe fn rapidhash_fast_tail(mut seed: u64, slice: &[u8], data: &[u8]) -> u64 {
  // SAFETY: caller guarantees data.len() > 16.
  unsafe { core::hint::assert_unchecked(data.len() > 16) };

  if likely(slice.len() > 16) {
    seed = rapid_mix(read_u64_le(slice, 0) ^ DEFAULT_SECRETS[0], read_u64_le(slice, 8) ^ seed);
    if likely(slice.len() > 32) {
      seed = rapid_mix(
        read_u64_le(slice, 16) ^ DEFAULT_SECRETS[0],
        read_u64_le(slice, 24) ^ seed,
      );
    }
  }

  let a = read_u64_le(data, data.len() - 16);
  let b = read_u64_le(data, data.len() - 8);
  seed = seed.wrapping_add(data.len() as u64);
  rapidhash_fast_finish(a, b, seed)
}

/// Cold medium path: 17B–~400B. 3-stream accumulation for 49B+.
///
/// `#[cold] #[inline(never)]` keeps the small-path entry point lean for
/// inlining. The 17–48B range short-circuits to the tail handler to avoid
/// the 3-stream register setup (stack spill on x86-64, branch-predictor
/// noise on aarch64).
///
/// # Safety
///
/// Caller must guarantee `data.len() > 16`.
#[cold]
#[inline(never)]
unsafe fn rapidhash_fast_core_medium(data: &[u8], mut seed: u64) -> u64 {
  // SAFETY: caller guarantees data.len() > 16.
  unsafe { core::hint::assert_unchecked(data.len() > 16) };

  // Short-circuit 17–48B to the tail handler. On x86-64 this avoids
  // 3-stream register setup/stack spill. On aarch64 it gives the branch
  // predictor a cleaner layout for the >48B path.
  if likely(data.len() <= 48) {
    // SAFETY: data.len() > 16 guaranteed by caller.
    return unsafe { rapidhash_fast_tail(seed, data, data) };
  }

  let mut slice = data;
  if unlikely(data.len() > 48) {
    if unlikely(data.len() > FAST_COLD_CUTOFF) {
      // SAFETY: data.len() > FAST_COLD_CUTOFF > 16.
      return unsafe { rapidhash_fast_core_large(data, seed) };
    }

    // 3-stream mid-range: 49–400B. Three independent accumulators allow
    // out-of-order execution to overlap the multiply latency.
    let mut see1 = seed;
    let mut see2 = seed;

    while slice.len() >= 48 {
      seed = rapid_mix(read_u64_le(slice, 0) ^ DEFAULT_SECRETS[0], read_u64_le(slice, 8) ^ seed);
      see1 = rapid_mix(
        read_u64_le(slice, 16) ^ DEFAULT_SECRETS[1],
        read_u64_le(slice, 24) ^ see1,
      );
      see2 = rapid_mix(
        read_u64_le(slice, 32) ^ DEFAULT_SECRETS[2],
        read_u64_le(slice, 40) ^ see2,
      );
      let (_, rest) = slice.split_at(48);
      slice = rest;
    }

    seed ^= see1 ^ see2;
  }

  // SAFETY: data.len() > 16 guaranteed by caller.
  unsafe { rapidhash_fast_tail(seed, slice, data) }
}

/// Cold large path: >400B. 7-stream bulk with 3-stream residual.
///
/// # Safety
///
/// Caller must guarantee `data.len() > FAST_COLD_CUTOFF`.
#[cold]
#[inline(never)]
unsafe fn rapidhash_fast_core_large(data: &[u8], mut seed: u64) -> u64 {
  // SAFETY: caller guarantees data.len() > FAST_COLD_CUTOFF.
  unsafe { core::hint::assert_unchecked(data.len() > FAST_COLD_CUTOFF) };

  let mut slice = data;
  let mut see1 = seed;
  let mut see2 = seed;
  let mut see3 = seed;
  let mut see4 = seed;
  let mut see5 = seed;
  let mut see6 = seed;

  // 7-stream bulk: 224B per iteration (2 × 112B half-blocks).
  while slice.len() >= 224 {
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

    let (_, rest) = slice.split_at(224);
    slice = rest;
  }

  // Single 112B half-block if enough remains.
  if likely(slice.len() >= 112) {
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
    let (_, rest) = slice.split_at(112);
    slice = rest;
  }

  // 3-stream residual: up to 2 × 48B blocks drain the remainder before the
  // tail, avoiding the serial dependency chain of the V3 nested-if tail.
  if slice.len() >= 48 {
    seed = rapid_mix(read_u64_le(slice, 0) ^ DEFAULT_SECRETS[0], read_u64_le(slice, 8) ^ seed);
    see1 = rapid_mix(
      read_u64_le(slice, 16) ^ DEFAULT_SECRETS[1],
      read_u64_le(slice, 24) ^ see1,
    );
    see2 = rapid_mix(
      read_u64_le(slice, 32) ^ DEFAULT_SECRETS[2],
      read_u64_le(slice, 40) ^ see2,
    );
    let (_, rest) = slice.split_at(48);
    slice = rest;

    if slice.len() >= 48 {
      seed = rapid_mix(read_u64_le(slice, 0) ^ DEFAULT_SECRETS[0], read_u64_le(slice, 8) ^ seed);
      see1 = rapid_mix(
        read_u64_le(slice, 16) ^ DEFAULT_SECRETS[1],
        read_u64_le(slice, 24) ^ see1,
      );
      see2 = rapid_mix(
        read_u64_le(slice, 32) ^ DEFAULT_SECRETS[2],
        read_u64_le(slice, 40) ^ see2,
      );
      let (_, rest) = slice.split_at(48);
      slice = rest;
    }
  }

  // Merge all accumulators into seed.
  see3 ^= see4;
  see5 ^= see6;
  seed ^= see1;
  see3 ^= see2;
  seed ^= see5;
  seed ^= see3;

  // SAFETY: data.len() > FAST_COLD_CUTOFF > 16.
  unsafe { rapidhash_fast_tail(seed, slice, data) }
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

impl FastHash for RapidHashFast64 {
  const OUTPUT_SIZE: usize = 8;
  type Output = u64;
  type Seed = u64;

  #[inline]
  fn hash_with_seed(seed: Self::Seed, data: &[u8]) -> Self::Output {
    dispatch::hash64_fast_with_seed(seed, data)
  }
}

impl FastHash for RapidHashFast128 {
  const OUTPUT_SIZE: usize = 16;
  type Output = u128;
  type Seed = u64;

  #[inline]
  fn hash_with_seed(seed: Self::Seed, data: &[u8]) -> Self::Output {
    dispatch::hash128_fast_with_seed(seed, data)
  }
}

#[cfg(test)]
mod tests {
  use proptest::prelude::*;

  use super::{RapidHash64, RapidHash128, RapidHashFast64, RapidHashFast128};

  // -- V3 standard variant oracle (C++-compatible output) --

  #[test]
  fn smoke_v3_empty_matches_oracle() {
    let seed = 0u64;
    let ours = <RapidHash64 as crate::traits::FastHash>::hash_with_seed(seed, b"");
    let secrets = rapidhash::v3::RapidSecrets::seed_cpp(seed);
    let theirs = rapidhash::v3::rapidhash_v3_seeded(b"", &secrets);
    assert_eq!(ours, theirs);
  }

  // -- Fast variant oracle (competitor's inner module) --

  #[test]
  #[cfg(target_endian = "little")]
  fn smoke_fast_empty_matches_oracle() {
    use core::hash::Hasher;
    let seed = 0u64;
    let ours = <RapidHashFast64 as crate::traits::FastHash>::hash_with_seed(seed, b"");
    let mut h = rapidhash::fast::RapidHasher::new(seed);
    h.write(b"");
    let theirs = h.finish();
    assert_eq!(ours, theirs);
  }

  proptest! {
    #[test]
    fn rapidhash_v3_64_matches_oracle(seed in any::<u64>(), data in proptest::collection::vec(any::<u8>(), 0..2048)) {
      let ours = <RapidHash64 as crate::traits::FastHash>::hash_with_seed(seed, &data);
      let secrets = rapidhash::v3::RapidSecrets::seed_cpp(seed);
      let theirs = rapidhash::v3::rapidhash_v3_seeded(&data, &secrets);
      prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn rapidhash128_is_two_independent_64bit_hashes(seed in any::<u64>(), data in proptest::collection::vec(any::<u8>(), 0..2048)) {
      let out = <RapidHash128 as crate::traits::FastHash>::hash_with_seed(seed, &data);
      let lo = out as u64;
      let hi = (out >> 64) as u64;

      prop_assert_eq!(lo, <RapidHash64 as crate::traits::FastHash>::hash_with_seed(seed, &data));
      prop_assert_eq!(hi, <RapidHash64 as crate::traits::FastHash>::hash_with_seed(seed ^ 0x9E37_79B9_7F4A_7C15, &data));
    }

    /// The fast variant uses the inner-algorithm core (3-stream mid-range,
    /// inner finalization). Oracle: `rapidhash::fast::RapidHasher` (the
    /// competitor's optimized inner module with AVALANCHE=false, SPONGE=true).
    ///
    /// LE-only: the competitor's inner module uses non-portable native-endian
    /// reads (`read_u64_np`), while we use LE reads for consistent hashing
    /// across platforms. On BE targets the outputs diverge by design.
    #[test]
    #[cfg(target_endian = "little")]
    fn rapidhash_fast_64_matches_inner_oracle(seed in any::<u64>(), data in proptest::collection::vec(any::<u8>(), 0..8192)) {
      use core::hash::Hasher;
      let ours = <RapidHashFast64 as crate::traits::FastHash>::hash_with_seed(seed, &data);
      let mut h = rapidhash::fast::RapidHasher::new(seed);
      h.write(&data);
      let theirs = h.finish();
      prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn rapidhash_fast_128_is_two_independent_64bit_hashes(seed in any::<u64>(), data in proptest::collection::vec(any::<u8>(), 0..2048)) {
      let out = <RapidHashFast128 as crate::traits::FastHash>::hash_with_seed(seed, &data);
      let lo = out as u64;
      let hi = (out >> 64) as u64;

      prop_assert_eq!(lo, <RapidHashFast64 as crate::traits::FastHash>::hash_with_seed(seed, &data));
      prop_assert_eq!(hi, <RapidHashFast64 as crate::traits::FastHash>::hash_with_seed(seed ^ 0x9E37_79B9_7F4A_7C15, &data));
    }
  }
}
