//! Arithmetic shared by the vector kernels that sum 64-byte blocks: x86-64, POWER, and IBM Z.
//!
//! Over a chunk of blocks `x[0..n)`, starting from the reduced sums `(a, b)`, Adler-32 gives
//! `a + Σ x[j]` and `b + n·a + Σ (n - j)·x[j]`. Writing byte `i` of block `t` among `k` blocks as
//! `x[t][i]`, its weight `n - j` is `64·(k - 1 - t) + (64 - i)`. A kernel's lanes therefore keep
//! three sums:
//!
//! - `bytes`: `Σ x`;
//! - `prefix`: before each block, the `bytes` of the earlier blocks, so it ends at
//!   `Σ_t (k - 1 - t)·Σ_i x[t][i]`;
//! - `weighted`: `Σ_t Σ_i (64 - i)·x[t][i]`.
//!
//! A chunk of [`CHUNK_BLOCKS`] blocks then folds into `(a, b)` once, modulo 65521, and bytes after
//! the last whole block go to the portable kernel. Each kernel proves that its own lanes hold
//! these sums without overflow for a whole chunk.

use super::portable;

/// Bytes per block. The weights of a block, 64 down to 1, fit in a signed or unsigned byte.
pub(super) const BLOCK: usize = 64;
/// Blocks per chunk, between two reductions modulo 65521.
pub(super) const CHUNK_BLOCKS: usize = 1024;

/// The weight of each byte of a block: 64 for its first byte, down to 1 for its last.
pub(super) static WEIGHTS: [u8; BLOCK] = {
  let mut weights = [0; BLOCK];
  let mut weight: u8 = 64;
  let mut i = 0;
  while i < BLOCK {
    weights[i] = weight;
    weight = weight.strict_sub(1);
    i = i.strict_add(1);
  }
  weights
};

/// Fold one chunk of `len` bytes into the reduced Adler-32 value `adler`.
///
/// `bytes` is the chunk's byte sum, and `weighted` is `64·prefix + weighted` from the lanes, in
/// the notation of the module documentation.
#[inline(always)]
pub(super) fn fold(adler: u32, len: usize, (bytes, weighted): (u64, u64)) -> u32 {
  // A chunk adds at most 255 per byte to `a`, so `a` stays below 2^25.
  let a = u32::try_from(bytes)
    .expect("a chunk's byte sum fits in 32 bits")
    .strict_add(adler & 0xffff);
  // `len` is at most 64 KiB and the old `a` below 2^16, so `len·a` stays below 2^32; `weighted` is
  // at most `255·Σ (n - j)` over the chunk, below 2^40. A short input keeps `b` below 2^32, where
  // a 32-bit remainder costs less than a 64-bit one.
  let b = u64::from(adler >> 16)
    .strict_add((len as u64).strict_mul(u64::from(adler & 0xffff)))
    .strict_add(weighted);
  let b = match u32::try_from(b) {
    Ok(b) => b % portable::MOD,
    Err(_) => portable::reduce(b),
  };
  (b << 16) | (a % portable::MOD)
}

/// Add the bytes after the last whole block, which no vector holds.
#[inline(always)]
pub(super) fn finish(adler: u32, tail: &[u8]) -> u32 {
  // The portable kernel ends with two reductions, which an empty tail does not need.
  if tail.is_empty() {
    adler
  } else {
    portable::update(adler, tail)
  }
}

/// Differential checks against the portable kernel that every block kernel runs on its target.
#[cfg(test)]
pub(super) mod checks {
  extern crate std;

  use alloc::{vec, vec::Vec};

  use super::{BLOCK, CHUNK_BLOCKS};
  use crate::checksum::adler32::{config::Adler32Force, portable};

  /// Bytes per chunk, the span of one deferred reduction.
  const CHUNK: usize = BLOCK * CHUNK_BLOCKS;

  /// A kernel's entry point.
  pub(in crate::checksum::adler32) type Kernel = fn(u32, &[u8]) -> u32;

  /// Deterministic pseudo-random bytes from a 64-bit xorshift generator.
  fn random_bytes(len: usize) -> Vec<u8> {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    (0..len)
      .map(|_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state.to_le_bytes()[0]
      })
      .collect()
  }

  /// Starting values: the initial value, and the largest reduced sums.
  const STARTS: [u32; 2] = [1, (65_520 << 16) | 65_520];

  /// Every length to 320 bytes covers each tail after zero to five blocks. The rest sit at
  /// multiples of zlib's 5552-byte reduction span and at the kernels' chunk boundaries.
  fn lengths() -> impl Iterator<Item = usize> {
    (0..=320).chain([
      5551,
      5552,
      5553,
      4 * 5552 + 1,
      CHUNK - 1,
      CHUNK,
      CHUNK + 1,
      CHUNK + BLOCK - 1,
      2 * CHUNK + 33,
      3 * CHUNK + 4093,
    ])
  }

  /// `kernel` matches the portable kernel at every length, alignment, and start; when streamed
  /// across its reductions; and for all-0xff input, which makes every lane sum its largest, at
  /// the longest span a 32-bit lane accumulates.
  pub(in crate::checksum::adler32) fn assert_matches_portable(name: &str, kernel: Kernel) {
    let data = random_bytes(3 * CHUNK + 4093 + BLOCK);
    for len in lengths() {
      for offset in 0..BLOCK {
        let input = &data[offset..offset.strict_add(len)];
        for start in STARTS {
          assert_eq!(
            kernel(start, input),
            portable::update(start, input),
            "{name}: len={len}, offset={offset}, start={start:#010x}"
          );
        }
      }
    }

    let data = &data[..3 * CHUNK + 4093];
    let expected = portable::update(1, data);
    for split in [63, 64, 65, 5552, CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK - 65] {
      let (head, tail) = data.split_at(split);
      assert_eq!(kernel(kernel(1, head), tail), expected, "{name}: split={split}");
    }
    for size in [1, 63, 4093, CHUNK + 7] {
      let streamed = data.chunks(size).fold(1, kernel);
      assert_eq!(streamed, expected, "{name}: chunk={size}");
    }

    for len in [CHUNK, CHUNK + 1, 2 * CHUNK, 2 * CHUNK + BLOCK + 31] {
      let data = vec![0xff; len];
      for start in STARTS {
        assert_eq!(
          kernel(start, &data),
          portable::update(start, &data),
          "{name}: all 0xff, len={len}, start={start:#010x}"
        );
      }
    }
  }

  /// With `RSCRYPTO_ADLER32_FORCE` naming `force`, the public API must run that kernel. The
  /// effective force must equal the request, so a host that lacks the kernel fails here instead
  /// of falling back to automatic selection.
  pub(in crate::checksum::adler32) fn assert_forced_kernel_runs(force: Adler32Force, name: &str) {
    use crate::{checksum::Adler32, traits::Checksum};

    let Ok(requested) = std::env::var("RSCRYPTO_ADLER32_FORCE") else {
      return;
    };
    if !requested.trim().eq_ignore_ascii_case(force.as_str()) {
      return;
    }
    assert_eq!(Adler32::config().effective_force, force, "this CPU cannot run {name}");
    assert_eq!(Adler32::kernel_name_for_len(CHUNK), name);

    let data = random_bytes(3 * CHUNK + 4093);
    for len in lengths() {
      let input = &data[..len];
      let expected = portable::update(1, input);
      assert_eq!(Adler32::checksum(input), expected, "{name}: len={len}");
      let mut streamed = Adler32::new();
      for part in input.chunks(4093) {
        streamed.update(part);
      }
      assert_eq!(streamed.finalize(), expected, "{name}: streamed len={len}");
    }
  }
}
