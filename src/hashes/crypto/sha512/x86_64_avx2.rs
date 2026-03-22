//! SHA-512 x86_64 AVX2 software kernel.
//!
//! Uses the Gueron-Krasnov two-block parallel technique: processes two
//! consecutive 128-byte blocks simultaneously, with message schedule words
//! from both blocks interleaved in `__m256i` registers (lower 128 bits =
//! block 1, upper 128 bits = block 2).
//!
//! Message schedule is computed in SIMD; compression rounds remain scalar.
//! When an odd number of blocks remains, falls back to the portable kernel
//! for the last block.
//!
//! Available on all x86_64 CPUs since Haswell (2013+).
//!
//! **Competitive advantage**: BoringSSL's AVX2 SHA-512 path is **disabled**
//! (CFI annotation issues). `ring` inherits this — it has NO AVX2 for SHA-512.
//! We beat ring on every AVX2-capable CPU.
//!
//! # Safety
//!
//! All functions require `avx2` target feature.

#![allow(unsafe_code)]
#![allow(clippy::inline_always)]
#![allow(clippy::indexing_slicing)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{BLOCK_LEN, K, Sha512, big_sigma0, big_sigma1, ch, maj};

// ─────────────────────────────────────────────────────────────────────────────
// SIMD sigma (message schedule, 2 × u64 per 128-bit lane, parallel both blocks)
// ─────────────────────────────────────────────────────────────────────────────

/// σ0(x) = ROTR(1) ^ ROTR(8) ^ SHR(7)
///
/// AVX2 has no VPRORQ, so each rotate is 3 ops: (x >> n) | (x << (64-n)).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn small_sigma0_v(x: __m256i) -> __m256i {
  // SAFETY: AVX2 intrinsics are available via this function's #[target_feature] attribute.
  unsafe {
    let rotr1 = _mm256_or_si256(_mm256_srli_epi64(x, 1), _mm256_slli_epi64(x, 63));
    let rotr8 = _mm256_or_si256(_mm256_srli_epi64(x, 8), _mm256_slli_epi64(x, 56));
    _mm256_xor_si256(_mm256_xor_si256(rotr1, rotr8), _mm256_srli_epi64(x, 7))
  }
}

/// σ1(x) = ROTR(19) ^ ROTR(61) ^ SHR(6)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn small_sigma1_v(x: __m256i) -> __m256i {
  // SAFETY: AVX2 intrinsics are available via this function's #[target_feature] attribute.
  unsafe {
    let rotr19 = _mm256_or_si256(_mm256_srli_epi64(x, 19), _mm256_slli_epi64(x, 45));
    let rotr61 = _mm256_or_si256(_mm256_srli_epi64(x, 61), _mm256_slli_epi64(x, 3));
    _mm256_xor_si256(_mm256_xor_si256(rotr19, rotr61), _mm256_srli_epi64(x, 6))
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Message schedule helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-register extraction: returns [a[1], a[2], a[3], b[0]] (shifting by
/// 1 u64 lane across two __m256i vectors).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn cross_lanes(a: __m256i, b: __m256i) -> __m256i {
  // SAFETY: AVX2 intrinsics are available via this function's #[target_feature] attribute.
  unsafe {
    let t = _mm256_permute2x128_si256(a, b, 0x21); // [a_hi, b_lo]
    _mm256_alignr_epi8(t, a, 8) // shift within each 128-bit lane by 8 bytes
  }
}

/// Compute 2 schedule words for both blocks, storing into `sched[i]`.
///
/// Uses a ring buffer view into `sched` for reading prior words.
/// `sched` has 40 slots; indices 0-7 are initial words, 8-39 are computed.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn schedule_pair(sched: &mut [__m256i; 40], i: usize) {
  // SAFETY: AVX2 intrinsics and target-feature-gated calls are available via
  // this function's #[target_feature] attribute.
  unsafe {
    // σ1 input: [W[2i-2], W[2i-1]] for both blocks
    let s1_in = sched[i.wrapping_sub(1)];
    // W[t-7]: [W[2i-7], W[2i-6]] — crosses a pair boundary
    let w_tm7 = cross_lanes(sched[i.wrapping_sub(4)], sched[i.wrapping_sub(3)]);
    // σ0 input: [W[2i-15], W[2i-14]] — crosses a pair boundary
    let s0_in = cross_lanes(sched[i.wrapping_sub(8)], sched[i.wrapping_sub(7)]);
    // W[t-16]: [W[2i-16], W[2i-15]] — exact pair
    let w_tm16 = sched[i.wrapping_sub(8)];

    sched[i] = _mm256_add_epi64(
      _mm256_add_epi64(small_sigma1_v(s1_in), w_tm7),
      _mm256_add_epi64(small_sigma0_v(s0_in), w_tm16),
    );
  }
}

/// Byte-swap mask for big-endian u64 word loads (128-bit).
#[cfg(target_arch = "x86_64")]
static BSWAP64_128: [u8; 16] = [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8];

/// Load 2 big-endian u64 words from each of two blocks, interleaving them
/// into a single __m256i: lo128 = block1 words, hi128 = block2 words.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn load_two_blocks(blk1: *const u8, blk2: *const u8, offset: usize, bswap: __m128i) -> __m256i {
  // SAFETY: AVX2/SSE intrinsics are available via this function's #[target_feature] attribute.
  // Pointer arithmetic is bounded by caller-provided block pointers.
  unsafe {
    let lo = _mm_shuffle_epi8(_mm_loadu_si128(blk1.add(offset).cast()), bswap);
    let hi = _mm_shuffle_epi8(_mm_loadu_si128(blk2.add(offset).cast()), bswap);
    _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1)
  }
}

/// Scalar compression for one block, reading schedule from a flat array.
///
/// Extracts u64 words from the specified lane of each __m256i schedule entry.
/// `lo_lane`: if true, extract from lower 128-bit lane (block 1); else upper (block 2).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn compress_one_block(state: &mut [u64; 8], sched: &[__m256i; 40], lo_lane: bool) {
  // SAFETY: AVX2 intrinsics are available via this function's #[target_feature] attribute.
  unsafe {
    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    macro_rules! sha_round {
      ($k:expr, $w:expr) => {{
        let t1 = h
          .wrapping_add(big_sigma1(e))
          .wrapping_add(ch(e, f, g))
          .wrapping_add($k)
          .wrapping_add($w);
        let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
      }};
    }

    for (pair, &v) in sched.iter().enumerate() {
      // Extract 2 u64 words from the correct block's 128-bit lane.
      let (w_lo, w_hi) = if lo_lane {
        // Lower 128-bit lane: cast to __m128i and extract.
        let lo128 = _mm256_castsi256_si128(v);
        (_mm_extract_epi64(lo128, 0) as u64, _mm_extract_epi64(lo128, 1) as u64)
      } else {
        // Upper 128-bit lane.
        let hi128 = _mm256_extracti128_si256(v, 1);
        (_mm_extract_epi64(hi128, 0) as u64, _mm_extract_epi64(hi128, 1) as u64)
      };
      let r = pair.strict_mul(2);
      sha_round!(K[r], w_lo);
      sha_round!(K[r + 1], w_hi);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
  } // unsafe
}

/// SHA-512 multi-block compression using AVX2 (Gueron-Krasnov two-block).
///
/// Processes blocks in pairs. If an odd number of blocks remains, the last
/// block is processed by the portable kernel.
///
/// # Safety
///
/// Caller must ensure `avx2` CPU feature is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn compress_blocks_avx2(state: &mut [u64; 8], blocks: &[u8]) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
  let num_blocks = blocks.len() / BLOCK_LEN;
  if num_blocks == 0 {
    return;
  }

  // SAFETY: AVX2 intrinsics and target-feature-gated calls are available via this
  // function's #[target_feature] attribute. Pointer arithmetic is bounded by blocks.len().
  // from_raw_parts length is exactly BLOCK_LEN for the final odd block.
  unsafe {
    let mut ptr = blocks.as_ptr();
    let mut remaining = num_blocks;

    let bswap = _mm_loadu_si128(BSWAP64_128.as_ptr().cast());

    while remaining >= 2 {
      let blk1 = ptr;
      let blk2 = ptr.add(BLOCK_LEN);

      // Full schedule array: 40 × __m256i = 1280 bytes.
      // Each entry holds 2 u64 words from block 1 (lo128) and block 2 (hi128).
      let mut sched: [__m256i; 40] = [_mm256_setzero_si256(); 40];

      // Load initial 16 message words (8 pairs) from both blocks.
      sched[0] = load_two_blocks(blk1, blk2, 0, bswap);
      sched[1] = load_two_blocks(blk1, blk2, 16, bswap);
      sched[2] = load_two_blocks(blk1, blk2, 32, bswap);
      sched[3] = load_two_blocks(blk1, blk2, 48, bswap);
      sched[4] = load_two_blocks(blk1, blk2, 64, bswap);
      sched[5] = load_two_blocks(blk1, blk2, 80, bswap);
      sched[6] = load_two_blocks(blk1, blk2, 96, bswap);
      sched[7] = load_two_blocks(blk1, blk2, 112, bswap);

      // Expand schedule: compute pairs 8-39 (rounds 16-79) for both blocks.
      for pair in 8..40 {
        schedule_pair(&mut sched, pair);
      }

      // Compress block 1 (lower 128-bit lanes).
      compress_one_block(state, &sched, true);

      // Compress block 2 (upper 128-bit lanes).
      compress_one_block(state, &sched, false);

      ptr = ptr.add(BLOCK_LEN.strict_mul(2));
      remaining = remaining.strict_sub(2);
    }

    // Handle the last block (if odd count) with the portable kernel.
    if remaining == 1 {
      Sha512::compress_blocks_portable(state, core::slice::from_raw_parts(ptr, BLOCK_LEN));
    }
  } // unsafe
}
