//! SHA-512 x86_64 AVX2 + BMI2 stitched dual-block kernel.
//!
//! Uses the Gueron-Krasnov two-block parallel technique with **stitched**
//! schedule/compress: SIMD message schedule expansion (on FP ports) is
//! interleaved with scalar compression rounds (on ALU ports), allowing the
//! out-of-order engine to overlap both pipelines.
//!
//! Architecture:
//! - **Block 1**: stitched — SIMD schedule expands into a ring buffer while scalar rounds consume
//!   words from the lower 128-bit lane.
//! - **Block 2**: pure scalar — reads pre-computed `W[t] + K[t]` values that were extracted from
//!   the upper lane during block 1's pass.
//!
//! BMI2 (`bmi2` target feature) enables the compiler to emit `RORX` for scalar
//! 64-bit rotations — a non-destructive, flag-free rotate that improves
//! register allocation and throughput.
//!
//! Deferred-Σ0 optimization: the `Σ0(a) + Maj(a,b,c)` addition is postponed
//! to the start of the next round, shortening the critical dependency chain
//! (e → Σ1 → Ch → T1 → e') by removing Σ0 from it.
//!
//! Available on all x86_64 CPUs since Haswell (2013+).
//!
//! **Competitive advantage**: BoringSSL's AVX2 SHA-512 path is **disabled**
//! (CFI annotation issues). `ring` inherits this — it has NO AVX2 for SHA-512.
//!
//! # Safety
//!
//! All functions require `avx2` and `bmi2` target features.

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

/// Compute 2 schedule words for both blocks using a ring buffer.
///
/// `w` is an 8-slot ring buffer; indices are masked with `& 7`.
/// Called for pairs i = 8..39 (schedule words 16..79).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn schedule_pair(w: &mut [__m256i; 8], i: usize) {
  // SAFETY: AVX2 intrinsics and target-feature-gated calls are available via
  // this function's #[target_feature] attribute.
  unsafe {
    // σ1 input: [W[2i-2], W[2i-1]] for both blocks
    let s1_in = w[i.wrapping_sub(1) & 7];
    // W[t-7]: [W[2i-7], W[2i-6]] — crosses a pair boundary
    let w_tm7 = cross_lanes(w[i.wrapping_sub(4) & 7], w[i.wrapping_sub(3) & 7]);
    // σ0 input: [W[2i-15], W[2i-14]] — crosses a pair boundary
    let s0_in = cross_lanes(w[i.wrapping_sub(8) & 7], w[i.wrapping_sub(7) & 7]);
    // W[t-16]: [W[2i-16], W[2i-15]] — exact pair
    let w_tm16 = w[i.wrapping_sub(8) & 7];

    w[i & 7] = _mm256_add_epi64(
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

/// Extract two u64 words from the lower 128-bit lane of a __m256i.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn extract_lo(v: __m256i) -> (u64, u64) {
  // SAFETY: AVX2/SSE intrinsics are available via this function's #[target_feature] attribute.
  unsafe {
    let lo128 = _mm256_castsi256_si128(v);
    (_mm_extract_epi64(lo128, 0) as u64, _mm_extract_epi64(lo128, 1) as u64)
  }
}

/// Extract two u64 words from the upper 128-bit lane of a __m256i.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn extract_hi(v: __m256i) -> (u64, u64) {
  // SAFETY: AVX2 intrinsics are available via this function's #[target_feature] attribute.
  unsafe {
    let hi128 = _mm256_extracti128_si256(v, 1);
    (_mm_extract_epi64(hi128, 0) as u64, _mm_extract_epi64(hi128, 1) as u64)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stitched dual-block compression
// ─────────────────────────────────────────────────────────────────────────────

/// SHA-512 multi-block compression using AVX2 + BMI2 (stitched dual-block).
///
/// Processes blocks in pairs. During block 1's compression, SIMD schedule
/// expansion is interleaved with scalar rounds, allowing FP and ALU ports
/// to work in parallel. Block 2's `W[t]+K[t]` values are stored to a side
/// buffer during block 1's pass, then consumed in a pure-scalar second pass.
///
/// If an odd number of blocks remains, the last block is processed by the
/// portable kernel.
///
/// # Safety
///
/// Caller must ensure `avx2` and `bmi2` CPU features are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2")]
pub(crate) unsafe fn compress_blocks_avx2(state: &mut [u64; 8], blocks: &[u8]) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
  let num_blocks = blocks.len() / BLOCK_LEN;
  if num_blocks == 0 {
    return;
  }

  // SAFETY: AVX2/BMI2 intrinsics and target-feature-gated calls are available via this
  // function's #[target_feature] attribute. Pointer arithmetic is bounded by blocks.len().
  // from_raw_parts length is exactly BLOCK_LEN for the final odd block.
  unsafe {
    let mut ptr = blocks.as_ptr();
    let mut remaining = num_blocks;

    let bswap = _mm_loadu_si128(BSWAP64_128.as_ptr().cast());

    while remaining >= 2 {
      let blk1 = ptr;
      let blk2 = ptr.add(BLOCK_LEN);

      // Ring-buffer schedule: 8 × __m256i (256 bytes).
      // Each entry holds 2 u64 words from block 1 (lo128) and block 2 (hi128).
      let mut w: [__m256i; 8] = [
        load_two_blocks(blk1, blk2, 0, bswap),
        load_two_blocks(blk1, blk2, 16, bswap),
        load_two_blocks(blk1, blk2, 32, bswap),
        load_two_blocks(blk1, blk2, 48, bswap),
        load_two_blocks(blk1, blk2, 64, bswap),
        load_two_blocks(blk1, blk2, 80, bswap),
        load_two_blocks(blk1, blk2, 96, bswap),
        load_two_blocks(blk1, blk2, 112, bswap),
      ];

      // Block 2 pre-computed W[t]+K[t] buffer (640 bytes).
      let mut t2_buf = [0u64; 80];

      // ── Block 1: stitched (SIMD schedule + scalar rounds) ──────────────
      let mut a = state[0];
      let mut b = state[1];
      let mut c = state[2];
      let mut d = state[3];
      let mut e = state[4];
      let mut f = state[5];
      let mut g = state[6];
      let mut h = state[7];

      // Deferred Σ0(a) + Maj(a,b,c) from previous round.
      let mut t2_deferred: u64 = 0;

      // Deferred-Σ0 round: applies the previous round's T2 at the start,
      // computes this round's T2 but defers its addition to the next round.
      // This shortens the critical path: e → Σ1 → Ch → T1 → e'.
      macro_rules! round {
        ($wk:expr) => {{
          a = a.wrapping_add(t2_deferred);
          let t1 = h
            .wrapping_add(big_sigma1(e))
            .wrapping_add(ch(e, f, g))
            .wrapping_add($wk);
          t2_deferred = big_sigma0(a).wrapping_add(maj(a, b, c));
          h = g;
          g = f;
          f = e;
          e = d.wrapping_add(t1);
          d = c;
          c = b;
          b = a;
          a = t1;
        }};
      }

      // Rounds 0-15: consume loaded words, store block 2 values.
      for pair in 0..8usize {
        let (lo0, lo1) = extract_lo(w[pair]);
        let (hi0, hi1) = extract_hi(w[pair]);
        let r = pair.strict_mul(2);
        t2_buf[r] = hi0.wrapping_add(K[r]);
        t2_buf[r.strict_add(1)] = hi1.wrapping_add(K[r.strict_add(1)]);
        round!(lo0.wrapping_add(K[r]));
        round!(lo1.wrapping_add(K[r.strict_add(1)]));
      }

      // Rounds 16-79: interleaved SIMD schedule expansion + scalar rounds.
      for pair in 8..40usize {
        schedule_pair(&mut w, pair);
        let slot = pair & 7;
        let (lo0, lo1) = extract_lo(w[slot]);
        let (hi0, hi1) = extract_hi(w[slot]);
        let r = pair.strict_mul(2);
        t2_buf[r] = hi0.wrapping_add(K[r]);
        t2_buf[r.strict_add(1)] = hi1.wrapping_add(K[r.strict_add(1)]);
        round!(lo0.wrapping_add(K[r]));
        round!(lo1.wrapping_add(K[r.strict_add(1)]));
      }

      // Apply final deferred Σ0.
      a = a.wrapping_add(t2_deferred);

      state[0] = state[0].wrapping_add(a);
      state[1] = state[1].wrapping_add(b);
      state[2] = state[2].wrapping_add(c);
      state[3] = state[3].wrapping_add(d);
      state[4] = state[4].wrapping_add(e);
      state[5] = state[5].wrapping_add(f);
      state[6] = state[6].wrapping_add(g);
      state[7] = state[7].wrapping_add(h);

      // ── Block 2: pure scalar from pre-computed buffer ──────────────────
      a = state[0];
      b = state[1];
      c = state[2];
      d = state[3];
      e = state[4];
      f = state[5];
      g = state[6];
      h = state[7];
      t2_deferred = 0;

      for r in 0..80usize {
        round!(t2_buf[r]);
      }

      a = a.wrapping_add(t2_deferred);

      state[0] = state[0].wrapping_add(a);
      state[1] = state[1].wrapping_add(b);
      state[2] = state[2].wrapping_add(c);
      state[3] = state[3].wrapping_add(d);
      state[4] = state[4].wrapping_add(e);
      state[5] = state[5].wrapping_add(f);
      state[6] = state[6].wrapping_add(g);
      state[7] = state[7].wrapping_add(h);

      ptr = ptr.add(BLOCK_LEN.strict_mul(2));
      remaining = remaining.strict_sub(2);
    }

    // Handle the last block (if odd count) with the portable kernel.
    if remaining == 1 {
      Sha512::compress_blocks_portable(state, core::slice::from_raw_parts(ptr, BLOCK_LEN));
    }
  } // unsafe
}
