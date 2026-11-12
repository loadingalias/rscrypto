//! x86-64 Adler-32 kernels: SSSE3, AVX2, and AVX-512 VNNI.
//!
//! Every kernel sums 64-byte blocks in vector lanes. Over a chunk of blocks `x[0..n)`, starting
//! from the reduced sums `(a, b)`, Adler-32 gives `a + Σ x[j]` and `b + n·a + Σ (n - j)·x[j]`.
//! Writing byte `i` of block `t` among `k` blocks as `x[t][i]`, its weight `n - j` is
//! `64·(k - 1 - t) + (64 - i)`. The lanes therefore keep three sums:
//!
//! - `bytes`: `Σ x`, from `psadbw`;
//! - `prefix`: before each block, the `bytes` of the earlier blocks, so it ends at
//!   `Σ_t (k - 1 - t)·Σ_i x[t][i]`;
//! - `weighted`: `Σ_t Σ_i (64 - i)·x[t][i]`, from `pmaddubsw` and `pmaddwd`, or from `vpdpbusd`.
//!
//! A chunk of [`CHUNK_BLOCKS`] blocks then folds into `(a, b)` once, modulo 65521. Bytes after the
//! last whole block go to the portable kernel.
//!
//! # Safety
//!
//! Each kernel is a `#[target_feature]` function; callers must establish its features. The
//! `_safe` wrappers rely on the dispatcher to select them only on CPUs that have those features:
//! [`KERNELS`] pairs each one with the capabilities that it requires.

use core::arch::x86_64::*;

use super::{config::Adler32Force, portable};
use crate::{
  checksum::dispatchers::Adler32Fn,
  platform::{Caps, caps::x86},
};

/// One x86-64 kernel: its force, required capabilities, entry point, and canonical name.
pub(in crate::checksum) struct Kernel {
  pub(in crate::checksum) force: Adler32Force,
  pub(in crate::checksum) requires: Caps,
  /// Callable only on a CPU whose cached detection reports `requires`.
  pub(in crate::checksum) update: Adler32Fn,
  pub(in crate::checksum) name: &'static str,
}

/// The x86-64 kernels, fastest first. Each `requires` holds every capability whose instructions
/// its kernel's `#[target_feature]` set enables; detection reports a capability only with the
/// features that it implies.
pub(in crate::checksum) const KERNELS: [Kernel; 3] = [
  Kernel {
    force: Adler32Force::Avx512Vnni,
    requires: x86::AVX512F.union(x86::AVX512BW).union(x86::AVX512VNNI),
    update: adler32_avx512_vnni_safe,
    name: "x86_64/avx512-vnni",
  },
  Kernel {
    force: Adler32Force::Avx2,
    requires: x86::AVX2,
    update: adler32_avx2_safe,
    name: "x86_64/avx2",
  },
  Kernel {
    force: Adler32Force::Ssse3,
    requires: x86::SSSE3,
    update: adler32_ssse3_safe,
    name: "x86_64/ssse3",
  },
];

/// The fastest kernel that a CPU with `caps` can run, or `None` without SSSE3.
#[inline]
pub(in crate::checksum) fn select(caps: Caps) -> Option<&'static Kernel> {
  KERNELS.iter().find(|kernel| caps.has(kernel.requires))
}

/// Bytes per block. The block's weights, 64 down to 1, fit in an `i8`, and `pmaddubsw` adds two
/// adjacent products of at most `255·(64 + 63) = 32385`, below `i16::MAX`.
const BLOCK: usize = 64;
/// Blocks per chunk, between two reductions modulo 65521.
///
/// The `bytes` and `prefix` lanes are 64-bit and cannot overflow. A 32-bit `weighted` lane gains
/// at most `255·616` per block: SSSE3 adds four products of a 16-byte vector to each lane, whose
/// largest weights are 64 to 61, 48 to 45, 32 to 29, and 16 to 13, which total 616.
const CHUNK_BLOCKS: usize = 1024;
const _: () = assert!(255 * 616 * CHUNK_BLOCKS <= i32::MAX as usize);

/// Inputs below this length take [`adler32_avx2_short`] on every AVX2 and VNNI CPU. It folds its
/// lanes without chunks, in 32 bits: such an input's byte sum stays below `255·192` and its
/// weighted sum below `255·Σ (n - j) < 2^23`, so `b` stays below 2^25 from any reduced start.
/// The 256-bit and 512-bit chunk setup and reductions cost more than these few blocks.
const SHORT_LEN: usize = 4 * BLOCK;

/// The weight of each byte of a block: 64 for its first byte, down to 1 for its last.
static WEIGHTS: [i8; BLOCK] = {
  let mut weights = [0; BLOCK];
  let mut weight: i8 = 64;
  let mut i = 0;
  while i < BLOCK {
    weights[i] = weight;
    weight = weight.strict_sub(1);
    i = i.strict_add(1);
  }
  weights
};

/// The halves of [`WEIGHTS`] for 32-byte vectors.
fn weight_halves() -> &'static [[i8; 32]] {
  WEIGHTS.as_chunks::<32>().0
}

/// The quarters of [`WEIGHTS`] for 16-byte vectors.
fn weight_quarters() -> &'static [[i8; 16]] {
  WEIGHTS.as_chunks::<16>().0
}

/// Fold one chunk of `len` bytes into the reduced Adler-32 value `adler`.
///
/// `bytes` is the chunk's byte sum, and `weighted` is `64·prefix + weighted` from the lanes, in
/// the notation of the module documentation.
#[inline(always)]
fn fold(adler: u32, len: usize, (bytes, weighted): (u64, u64)) -> u32 {
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
fn finish(adler: u32, tail: &[u8]) -> u32 {
  // The portable kernel ends with two reductions, which an empty tail does not need.
  if tail.is_empty() {
    adler
  } else {
    portable::update(adler, tail)
  }
}

// SSSE3

/// # Safety
/// Calling outside an SSSE3-enabled function requires SSSE3 support.
#[inline]
#[target_feature(enable = "ssse3")]
fn load128(bytes: &[u8; 16]) -> __m128i {
  // SAFETY: `bytes` is 16 readable bytes, and this load has no alignment requirement.
  unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) }
}

/// # Safety
/// Calling outside an SSSE3-enabled function requires SSSE3 support.
#[inline]
#[target_feature(enable = "ssse3")]
fn load_weights128(weights: &[i8; 16]) -> __m128i {
  // SAFETY: `weights` is 16 readable bytes, and this load has no alignment requirement.
  unsafe { _mm_loadu_si128(weights.as_ptr().cast()) }
}

/// The sums of the 64-bit lanes of `bytes` and of `weighted`.
///
/// # Safety
/// Calling outside an SSSE3-enabled function requires SSSE3 support.
#[inline]
#[target_feature(enable = "ssse3")]
fn sum_lanes128(bytes: __m128i, weighted: __m128i) -> (u64, u64) {
  let sums = _mm_add_epi64(_mm_unpacklo_epi64(bytes, weighted), _mm_unpackhi_epi64(bytes, weighted));
  (
    _mm_cvtsi128_si64(sums).cast_unsigned(),
    _mm_cvtsi128_si64(_mm_unpackhi_epi64(sums, sums)).cast_unsigned(),
  )
}

/// Widen the four 32-bit lanes of `v` into two 64-bit sums of lane pairs.
///
/// # Safety
/// Calling outside an SSSE3-enabled function requires SSSE3 support.
#[inline]
#[target_feature(enable = "ssse3")]
fn widen_u32x4(v: __m128i) -> __m128i {
  _mm_add_epi64(_mm_srli_epi64::<32>(v), _mm_and_si128(v, _mm_set1_epi64x(0xffff_ffff)))
}

/// # Safety
/// Calling outside an SSSE3-enabled function requires SSSE3 support.
#[target_feature(enable = "ssse3")]
fn adler32_ssse3(mut adler: u32, data: &[u8]) -> u32 {
  let (blocks, tail) = data.as_chunks::<BLOCK>();
  let quarters = weight_quarters();
  let weights = [
    load_weights128(&quarters[0]),
    load_weights128(&quarters[1]),
    load_weights128(&quarters[2]),
    load_weights128(&quarters[3]),
  ];
  let zero = _mm_setzero_si128();
  let ones = _mm_set1_epi16(1);
  for chunk in blocks.chunks(CHUNK_BLOCKS) {
    let mut bytes = zero;
    let mut prefix = zero;
    let mut weighted = zero;
    for block in chunk {
      prefix = _mm_add_epi64(prefix, bytes);
      let (mut block_bytes, mut block_weighted) = (zero, zero);
      for (quarter, &weight) in block.as_chunks::<16>().0.iter().zip(&weights) {
        let v = load128(quarter);
        block_bytes = _mm_add_epi64(block_bytes, _mm_sad_epu8(v, zero));
        block_weighted = _mm_add_epi32(block_weighted, _mm_madd_epi16(_mm_maddubs_epi16(v, weight), ones));
      }
      bytes = _mm_add_epi64(bytes, block_bytes);
      weighted = _mm_add_epi32(weighted, block_weighted);
    }
    let weighted = _mm_add_epi64(_mm_slli_epi64::<6>(prefix), widen_u32x4(weighted));
    adler = fold(adler, chunk.len().strict_mul(BLOCK), sum_lanes128(bytes, weighted));
  }
  finish(adler, tail)
}

/// Safe wrapper for the SSSE3 kernel.
#[inline]
pub(in crate::checksum) fn adler32_ssse3_safe(adler: u32, data: &[u8]) -> u32 {
  // SAFETY: The dispatcher selects this kernel only when cached detection reports SSSE3.
  unsafe { adler32_ssse3(adler, data) }
}

// AVX2

/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn load256(bytes: &[u8; 32]) -> __m256i {
  // SAFETY: `bytes` is 32 readable bytes, and this load has no alignment requirement.
  unsafe { _mm256_loadu_si256(bytes.as_ptr().cast()) }
}

/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn load_weights256(weights: &[i8; 32]) -> __m256i {
  // SAFETY: `weights` is 32 readable bytes, and this load has no alignment requirement.
  unsafe { _mm256_loadu_si256(weights.as_ptr().cast()) }
}

/// The sums of the 64-bit lanes of `bytes` and of `weighted`.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn sum_lanes256(bytes: __m256i, weighted: __m256i) -> (u64, u64) {
  let sums = _mm256_add_epi64(
    _mm256_unpacklo_epi64(bytes, weighted),
    _mm256_unpackhi_epi64(bytes, weighted),
  );
  let sums = _mm_add_epi64(_mm256_castsi256_si128(sums), _mm256_extracti128_si256::<1>(sums));
  (
    _mm_cvtsi128_si64(sums).cast_unsigned(),
    _mm_extract_epi64::<1>(sums).cast_unsigned(),
  )
}

/// The sums of the 32-bit lanes of `bytes` and of `weighted`.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn sum_lanes32x8(bytes: __m256i, weighted: __m256i) -> (u32, u32) {
  let sums = _mm256_hadd_epi32(bytes, weighted);
  let sums = _mm_add_epi32(_mm256_castsi256_si128(sums), _mm256_extracti128_si256::<1>(sums));
  let sums = _mm_hadd_epi32(sums, sums);
  (
    _mm_cvtsi128_si32(sums).cast_unsigned(),
    _mm_extract_epi32::<1>(sums).cast_unsigned(),
  )
}

/// One block's byte sums, in 64-bit lanes, and weighted sums, in 32-bit lanes.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn block_sums256(block: &[u8; BLOCK], weights: &[__m256i; 2]) -> (__m256i, __m256i) {
  let zero = _mm256_setzero_si256();
  let ones = _mm256_set1_epi16(1);
  let (mut bytes, mut weighted) = (zero, zero);
  for (half, &weight) in block.as_chunks::<32>().0.iter().zip(weights) {
    let v = load256(half);
    bytes = _mm256_add_epi64(bytes, _mm256_sad_epu8(v, zero));
    weighted = _mm256_add_epi32(weighted, _mm256_madd_epi16(_mm256_maddubs_epi16(v, weight), ones));
  }
  (bytes, weighted)
}

/// Widen the eight 32-bit lanes of `v` into four 64-bit sums of lane pairs.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
fn widen_u32x8(v: __m256i) -> __m256i {
  _mm256_add_epi64(
    _mm256_srli_epi64::<32>(v),
    _mm256_and_si256(v, _mm256_set1_epi64x(0xffff_ffff)),
  )
}

/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[target_feature(enable = "avx2")]
fn adler32_avx2(mut adler: u32, data: &[u8]) -> u32 {
  let (blocks, tail) = data.as_chunks::<BLOCK>();
  let halves = weight_halves();
  let weights = [load_weights256(&halves[0]), load_weights256(&halves[1])];
  let zero = _mm256_setzero_si256();
  for chunk in blocks.chunks(CHUNK_BLOCKS) {
    let mut bytes = zero;
    let mut prefix = zero;
    let mut weighted = zero;
    for block in chunk {
      prefix = _mm256_add_epi64(prefix, bytes);
      let (block_bytes, block_weighted) = block_sums256(block, &weights);
      bytes = _mm256_add_epi64(bytes, block_bytes);
      weighted = _mm256_add_epi32(weighted, block_weighted);
    }
    let weighted = _mm256_add_epi64(_mm256_slli_epi64::<6>(prefix), widen_u32x8(weighted));
    adler = fold(adler, chunk.len().strict_mul(BLOCK), sum_lanes256(bytes, weighted));
  }
  finish(adler, tail)
}

/// Adler-32 of an input shorter than [`SHORT_LEN`], without chunks.
///
/// # Safety
/// Calling outside an AVX2-enabled function requires AVX2 support.
#[target_feature(enable = "avx2")]
fn adler32_avx2_short(adler: u32, data: &[u8]) -> u32 {
  debug_assert!(data.len() < SHORT_LEN);
  let (blocks, tail) = data.as_chunks::<BLOCK>();
  let halves = weight_halves();
  let weights = [load_weights256(&halves[0]), load_weights256(&halves[1])];
  let zero = _mm256_setzero_si256();
  // The `bytes` lanes hold sums below 2^32 in the low halves of 64-bit lanes, so 32-bit lane
  // arithmetic keeps their high halves zero.
  let (mut bytes, mut prefix, mut weighted) = (zero, zero, zero);
  // `b` gains the starting `a` once per byte.
  let mut start_a_terms = 0u32;
  for block in blocks {
    prefix = _mm256_add_epi32(prefix, bytes);
    let (block_bytes, block_weighted) = block_sums256(block, &weights);
    bytes = _mm256_add_epi32(bytes, block_bytes);
    weighted = _mm256_add_epi32(weighted, block_weighted);
    start_a_terms = start_a_terms.strict_add((adler & 0xffff).strict_mul(64));
  }
  let (bytes, weighted) = sum_lanes32x8(bytes, _mm256_add_epi32(_mm256_slli_epi32::<6>(prefix), weighted));
  let a = (adler & 0xffff).strict_add(bytes);
  let b = (adler >> 16).strict_add(start_a_terms).strict_add(weighted);
  finish(((b % portable::MOD) << 16) | (a % portable::MOD), tail)
}

/// Safe wrapper for the AVX2 kernel.
///
/// The length check sits here, ahead of both `#[target_feature]` functions, so a short input
/// pays only the compare and a jump.
#[inline]
pub(in crate::checksum) fn adler32_avx2_safe(adler: u32, data: &[u8]) -> u32 {
  if data.len() < SHORT_LEN {
    // SAFETY: The dispatcher selects this kernel only when cached detection reports AVX2.
    return unsafe { adler32_avx2_short(adler, data) };
  }
  // SAFETY: The dispatcher selects this kernel only when cached detection reports AVX2.
  unsafe { adler32_avx2(adler, data) }
}

// AVX-512 VNNI

/// # Safety
/// Calling outside an AVX-512F-enabled function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn load512(bytes: &[u8; 64]) -> __m512i {
  // SAFETY: `bytes` is 64 readable bytes, and this load has no alignment requirement.
  unsafe { _mm512_loadu_si512(bytes.as_ptr().cast()) }
}

/// # Safety
/// Calling outside an AVX-512F-enabled function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn load_weights512() -> __m512i {
  // SAFETY: `WEIGHTS` is 64 readable bytes, and this load has no alignment requirement.
  unsafe { _mm512_loadu_si512(WEIGHTS.as_ptr().cast()) }
}

/// The sums of the 64-bit lanes of `bytes` and of `weighted`.
///
/// # Safety
/// Calling outside an AVX-512F-enabled function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn sum_lanes512(bytes: __m512i, weighted: __m512i) -> (u64, u64) {
  let sums = _mm512_add_epi64(
    _mm512_unpacklo_epi64(bytes, weighted),
    _mm512_unpackhi_epi64(bytes, weighted),
  );
  let sums = _mm256_add_epi64(_mm512_castsi512_si256(sums), _mm512_extracti64x4_epi64::<1>(sums));
  let sums = _mm_add_epi64(_mm256_castsi256_si128(sums), _mm256_extracti128_si256::<1>(sums));
  (
    _mm_cvtsi128_si64(sums).cast_unsigned(),
    _mm_extract_epi64::<1>(sums).cast_unsigned(),
  )
}

/// Widen the sixteen 32-bit lanes of `v` into eight 64-bit sums of lane pairs.
///
/// # Safety
/// Calling outside an AVX-512F-enabled function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn widen_u32x16(v: __m512i) -> __m512i {
  _mm512_add_epi64(
    _mm512_srli_epi64::<32>(v),
    _mm512_and_si512(v, _mm512_set1_epi64(0xffff_ffff)),
  )
}

/// Independent `weighted` accumulators of the VNNI kernel. `vpdpbusd` accumulates in place, so
/// one accumulator would serialize every block on its latency.
const VNNI_LANES: usize = 4;

/// # Safety
/// Calling outside a function that enables AVX-512BW and AVX-512 VNNI requires both.
#[target_feature(enable = "avx512bw,avx512vnni")]
fn adler32_avx512_vnni(mut adler: u32, data: &[u8]) -> u32 {
  let (blocks, tail) = data.as_chunks::<BLOCK>();
  let weights = load_weights512();
  let zero = _mm512_setzero_si512();
  for chunk in blocks.chunks(CHUNK_BLOCKS) {
    let mut bytes = zero;
    let mut prefix = zero;
    let mut weighted = [zero; VNNI_LANES];
    let (groups, rest) = chunk.as_chunks::<VNNI_LANES>();
    for group in groups {
      for (block, lane) in group.iter().zip(&mut weighted) {
        let v = load512(block);
        prefix = _mm512_add_epi64(prefix, bytes);
        bytes = _mm512_add_epi64(bytes, _mm512_sad_epu8(v, zero));
        *lane = _mm512_dpbusd_epi32(*lane, v, weights);
      }
    }
    // The last few blocks share one accumulator, which keeps every accumulator in a register.
    for block in rest {
      let v = load512(block);
      prefix = _mm512_add_epi64(prefix, bytes);
      bytes = _mm512_add_epi64(bytes, _mm512_sad_epu8(v, zero));
      weighted[0] = _mm512_dpbusd_epi32(weighted[0], v, weights);
    }
    let weighted = weighted.into_iter().fold(zero, |sum, lane| _mm512_add_epi32(sum, lane));
    let weighted = _mm512_add_epi64(_mm512_slli_epi64::<6>(prefix), widen_u32x16(weighted));
    adler = fold(adler, chunk.len().strict_mul(BLOCK), sum_lanes512(bytes, weighted));
  }
  finish(adler, tail)
}

/// Safe wrapper for the AVX-512 VNNI kernel, which runs [`adler32_avx2_short`] on short inputs.
///
/// The length check sits here, ahead of both `#[target_feature]` functions, so a short input
/// pays only the compare and a jump.
#[inline]
pub(in crate::checksum) fn adler32_avx512_vnni_safe(adler: u32, data: &[u8]) -> u32 {
  if data.len() < SHORT_LEN {
    // SAFETY: The dispatcher selects this kernel only when cached detection reports AVX-512F,
    // and detection reports AVX-512F only with AVX2, which Rust's `avx512f` implies.
    return unsafe { adler32_avx2_short(adler, data) };
  }
  // SAFETY: The dispatcher selects this kernel only when cached detection reports AVX-512F,
  // AVX-512BW, and AVX-512 VNNI.
  unsafe { adler32_avx512_vnni(adler, data) }
}

#[cfg(test)]
mod tests {
  extern crate std;

  use alloc::{vec, vec::Vec};

  use super::*;

  /// Bytes per chunk, the span of one deferred reduction.
  const CHUNK: usize = BLOCK * CHUNK_BLOCKS;

  type Kernel = fn(u32, &[u8]) -> u32;

  /// Each kernel that this CPU can run, with its name.
  fn kernels() -> Vec<(&'static str, Kernel)> {
    let mut kernels: Vec<(&'static str, Kernel)> = Vec::new();
    if cfg!(miri) {
      return kernels;
    }
    if std::arch::is_x86_feature_detected!("ssse3") {
      kernels.push(("ssse3", adler32_ssse3_safe));
    }
    if std::arch::is_x86_feature_detected!("avx2") {
      kernels.push(("avx2", adler32_avx2_safe));
    }
    if std::arch::is_x86_feature_detected!("avx512f")
      && std::arch::is_x86_feature_detected!("avx512bw")
      && std::arch::is_x86_feature_detected!("avx512vnni")
    {
      kernels.push(("avx512-vnni", adler32_avx512_vnni_safe));
    }
    kernels
  }

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

  #[test]
  fn kernels_match_portable_at_every_length_and_alignment() {
    let data = random_bytes(3 * CHUNK + 4093 + BLOCK);
    for (name, kernel) in kernels() {
      for len in lengths() {
        for offset in 0..BLOCK {
          let input = &data[offset..offset + len];
          for start in STARTS {
            assert_eq!(
              kernel(start, input),
              portable::update(start, input),
              "{name}: len={len}, offset={offset}, start={start:#010x}"
            );
          }
        }
      }
    }
  }

  #[test]
  fn kernels_match_portable_when_streamed_across_reductions() {
    let data = random_bytes(3 * CHUNK + 4093);
    let expected = portable::update(1, &data);
    for (name, kernel) in kernels() {
      for split in [63, 64, 65, 5552, CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK - 65] {
        let (head, tail) = data.split_at(split);
        assert_eq!(kernel(kernel(1, head), tail), expected, "{name}: split={split}");
      }
      for size in [1, 63, 4093, CHUNK + 7] {
        let streamed = data.chunks(size).fold(1, kernel);
        assert_eq!(streamed, expected, "{name}: chunk={size}");
      }
    }
  }

  /// All-0xff input makes every lane sum its largest; a full chunk is the longest span that a
  /// 32-bit `weighted` lane accumulates before a reduction.
  #[test]
  fn kernels_match_portable_for_all_0xff_at_the_longest_span() {
    for len in [CHUNK, CHUNK + 1, 2 * CHUNK, 2 * CHUNK + BLOCK + 31] {
      let data = vec![0xff; len];
      for start in STARTS {
        let expected = portable::update(start, &data);
        for (name, kernel) in kernels() {
          assert_eq!(kernel(start, &data), expected, "{name}: len={len}, start={start:#010x}");
        }
      }
    }
  }

  /// With `RSCRYPTO_ADLER32_FORCE` naming an x86-64 tier, the public API must run that tier.
  /// The effective force must equal the request, so a host that lacks the tier fails here
  /// instead of falling back to automatic selection.
  #[test]
  fn the_forced_tier_runs_through_the_public_api() {
    use crate::{checksum::Adler32, traits::Checksum};

    let Ok(requested) = std::env::var("RSCRYPTO_ADLER32_FORCE") else {
      return;
    };
    let Some(kernel) = KERNELS
      .iter()
      .find(|kernel| requested.trim().eq_ignore_ascii_case(kernel.force.as_str()))
    else {
      return;
    };
    assert_eq!(
      Adler32::config().effective_force,
      kernel.force,
      "this CPU cannot run {}",
      kernel.name
    );
    assert_eq!(Adler32::kernel_name_for_len(CHUNK), kernel.name);

    let data = random_bytes(3 * CHUNK + 4093);
    for len in lengths() {
      let input = &data[..len];
      let expected = portable::update(1, input);
      assert_eq!(Adler32::checksum(input), expected, "{}: len={len}", kernel.name);
      let mut streamed = Adler32::new();
      for part in input.chunks(4093) {
        streamed.update(part);
      }
      assert_eq!(streamed.finalize(), expected, "{}: streamed len={len}", kernel.name);
    }
  }
}
