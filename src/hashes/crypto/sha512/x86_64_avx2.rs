//! SHA-512 x86_64 AVX2 + BMI2 stitched kernel.
//!
//! Uses the Gueron-Krasnov two-block parallel technique with **stitched**
//! schedule/compress: SIMD message schedule expansion (on FP ports) is
//! interleaved with scalar compression rounds (on ALU ports), allowing the
//! out-of-order engine to overlap both pipelines.
//!
//! Architecture:
//! - **Dual-block** (≥ 2 blocks): 256-bit schedule (`__m256i` ring buffer, 3-op rotates), stitched
//!   with scalar rounds. Block 2's `W[t]+K[t]` stored during block 1's pass.
//! - **Single-block** (odd trailing): 128-bit schedule (`__m128i` ring buffer, 3-op rotates),
//!   stitched with scalar rounds. No portable fallback — fully self-contained.
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

use super::{BLOCK_LEN, K, big_sigma0, big_sigma1, ch, maj};

// ─────────────────────────────────────────────────────────────────────────────
// 256-bit SIMD sigma (dual-block schedule, 2 × u64 per 128-bit lane)
// ─────────────────────────────────────────────────────────────────────────────

/// σ0(x) = ROTR(1) ^ ROTR(8) ^ SHR(7)  — 256-bit, 3-op rotates.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn small_sigma0_256(x: __m256i) -> __m256i {
  // SAFETY: AVX2 intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    let rotr1 = _mm256_or_si256(_mm256_srli_epi64(x, 1), _mm256_slli_epi64(x, 63));
    let rotr8 = _mm256_or_si256(_mm256_srli_epi64(x, 8), _mm256_slli_epi64(x, 56));
    _mm256_xor_si256(_mm256_xor_si256(rotr1, rotr8), _mm256_srli_epi64(x, 7))
  }
}

/// σ1(x) = ROTR(19) ^ ROTR(61) ^ SHR(6)  — 256-bit, 3-op rotates.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn small_sigma1_256(x: __m256i) -> __m256i {
  // SAFETY: AVX2 intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    let rotr19 = _mm256_or_si256(_mm256_srli_epi64(x, 19), _mm256_slli_epi64(x, 45));
    let rotr61 = _mm256_or_si256(_mm256_srli_epi64(x, 61), _mm256_slli_epi64(x, 3));
    _mm256_xor_si256(_mm256_xor_si256(rotr19, rotr61), _mm256_srli_epi64(x, 6))
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 128-bit SIMD sigma (single-block schedule, 2 × u64 per register)
// ─────────────────────────────────────────────────────────────────────────────

/// σ0(x) = ROTR(1) ^ ROTR(8) ^ SHR(7)  — 128-bit, 3-op rotates.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn small_sigma0_128(x: __m128i) -> __m128i {
  // SAFETY: SSE2 intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    let rotr1 = _mm_or_si128(_mm_srli_epi64(x, 1), _mm_slli_epi64(x, 63));
    let rotr8 = _mm_or_si128(_mm_srli_epi64(x, 8), _mm_slli_epi64(x, 56));
    _mm_xor_si128(_mm_xor_si128(rotr1, rotr8), _mm_srli_epi64(x, 7))
  }
}

/// σ1(x) = ROTR(19) ^ ROTR(61) ^ SHR(6)  — 128-bit, 3-op rotates.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn small_sigma1_128(x: __m128i) -> __m128i {
  // SAFETY: SSE2 intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    let rotr19 = _mm_or_si128(_mm_srli_epi64(x, 19), _mm_slli_epi64(x, 45));
    let rotr61 = _mm_or_si128(_mm_srli_epi64(x, 61), _mm_slli_epi64(x, 3));
    _mm_xor_si128(_mm_xor_si128(rotr19, rotr61), _mm_srli_epi64(x, 6))
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dual-block message schedule helpers (256-bit)
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-register extraction (256-bit): [a[1], a[2], a[3], b[0]].
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn cross_lanes_256(a: __m256i, b: __m256i) -> __m256i {
  // SAFETY: AVX2 intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    let t = _mm256_permute2x128_si256(a, b, 0x21); // [a_hi, b_lo]
    _mm256_alignr_epi8(t, a, 8)
  }
}

/// Compute 2 schedule words for both blocks (256-bit ring buffer).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn schedule_pair_256(w: &mut [__m256i; 8], i: usize) {
  // SAFETY: AVX2 intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    let s1_in = w[i.wrapping_sub(1) & 7];
    let w_tm7 = cross_lanes_256(w[i.wrapping_sub(4) & 7], w[i.wrapping_sub(3) & 7]);
    let s0_in = cross_lanes_256(w[i.wrapping_sub(8) & 7], w[i.wrapping_sub(7) & 7]);
    let w_tm16 = w[i.wrapping_sub(8) & 7];

    w[i & 7] = _mm256_add_epi64(
      _mm256_add_epi64(small_sigma1_256(s1_in), w_tm7),
      _mm256_add_epi64(small_sigma0_256(s0_in), w_tm16),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-block message schedule helpers (128-bit)
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-register extraction (128-bit): [a[1], b[0]].
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn cross_lanes_128(a: __m128i, b: __m128i) -> __m128i {
  // SAFETY: SSSE3 intrinsics are available via the caller's #[target_feature] attribute.
  unsafe { _mm_alignr_epi8(b, a, 8) }
}

/// Compute 2 schedule words for a single block (128-bit ring buffer).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn schedule_pair_128(w: &mut [__m128i; 8], i: usize) {
  // SAFETY: SSE2/SSSE3 intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    let s1_in = w[i.wrapping_sub(1) & 7];
    let w_tm7 = cross_lanes_128(w[i.wrapping_sub(4) & 7], w[i.wrapping_sub(3) & 7]);
    let s0_in = cross_lanes_128(w[i.wrapping_sub(8) & 7], w[i.wrapping_sub(7) & 7]);
    let w_tm16 = w[i.wrapping_sub(8) & 7];

    w[i & 7] = _mm_add_epi64(
      _mm_add_epi64(small_sigma1_128(s1_in), w_tm7),
      _mm_add_epi64(small_sigma0_128(s0_in), w_tm16),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Common helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Byte-swap mask for big-endian u64 word loads (128-bit).
#[cfg(target_arch = "x86_64")]
static BSWAP64_128: [u8; 16] = [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8];

/// Load 2 big-endian u64 words from each of two blocks into a __m256i.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn load_two_blocks(blk1: *const u8, blk2: *const u8, offset: usize, bswap: __m128i) -> __m256i {
  // SAFETY: AVX2/SSE intrinsics are available; pointer arithmetic bounded by caller.
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
  // SAFETY: SSE intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    let lo128 = _mm256_castsi256_si128(v);
    (_mm_extract_epi64(lo128, 0) as u64, _mm_extract_epi64(lo128, 1) as u64)
  }
}

/// Extract two u64 words from the upper 128-bit lane of a __m256i.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn extract_hi(v: __m256i) -> (u64, u64) {
  // SAFETY: AVX2 intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    let hi128 = _mm256_extracti128_si256(v, 1);
    (_mm_extract_epi64(hi128, 0) as u64, _mm_extract_epi64(hi128, 1) as u64)
  }
}

/// Extract two u64 words from a __m128i.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn extract_128(v: __m128i) -> (u64, u64) {
  // SAFETY: SSE intrinsics are available via the caller's #[target_feature] attribute.
  unsafe { (_mm_extract_epi64(v, 0) as u64, _mm_extract_epi64(v, 1) as u64) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

/// SHA-512 multi-block compression using AVX2 + BMI2.
///
/// - **≥ 2 blocks**: stitched dual-block (256-bit SIMD schedule + scalar rounds).
/// - **1 block**: stitched single-block (128-bit SIMD schedule + scalar rounds).
/// - **No portable fallback** — fully self-contained.
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
  unsafe {
    let mut ptr = blocks.as_ptr();
    let mut remaining = num_blocks;

    let bswap = _mm_loadu_si128(BSWAP64_128.as_ptr().cast());

    // Pre-allocate the block-2 round-value buffer outside the loop.
    // All 80 entries are written (rounds 0-15 + 16-79) before block 2 reads them,
    // so the initial value is irrelevant. Hoisting avoids a 640-byte memset per
    // dual-block iteration.
    let mut t2_buf = [0u64; 80];

    // ── Dual-block loop ──────────────────────────────────────────────────
    while remaining >= 2 {
      // Prefetch next block pair into L1D to hide L2 latency at large sizes.
      if remaining >= 4 {
        _mm_prefetch::<_MM_HINT_T0>(ptr.add(BLOCK_LEN.strict_mul(2)).cast());
        _mm_prefetch::<_MM_HINT_T0>(ptr.add(BLOCK_LEN.strict_mul(2).strict_add(64)).cast());
        _mm_prefetch::<_MM_HINT_T0>(ptr.add(BLOCK_LEN.strict_mul(3)).cast());
        _mm_prefetch::<_MM_HINT_T0>(ptr.add(BLOCK_LEN.strict_mul(3).strict_add(64)).cast());
      }

      let blk1 = ptr;
      let blk2 = ptr.add(BLOCK_LEN);

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

      let mut a = state[0];
      let mut b = state[1];
      let mut c = state[2];
      let mut d = state[3];
      let mut e = state[4];
      let mut f = state[5];
      let mut g = state[6];
      let mut h = state[7];
      let mut t2_deferred: u64 = 0;

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

      // Rounds 0-15: loaded words.
      for (pair, &wv) in w.iter().enumerate() {
        let (lo0, lo1) = extract_lo(wv);
        let (hi0, hi1) = extract_hi(wv);
        let r = pair.strict_mul(2);
        t2_buf[r] = hi0.wrapping_add(K[r]);
        t2_buf[r.strict_add(1)] = hi1.wrapping_add(K[r.strict_add(1)]);
        round!(lo0.wrapping_add(K[r]));
        round!(lo1.wrapping_add(K[r.strict_add(1)]));
      }

      // Rounds 16-79: interleaved schedule + rounds.
      for pair in 8..40usize {
        schedule_pair_256(&mut w, pair);
        let (lo0, lo1) = extract_lo(w[pair & 7]);
        let (hi0, hi1) = extract_hi(w[pair & 7]);
        let r = pair.strict_mul(2);
        t2_buf[r] = hi0.wrapping_add(K[r]);
        t2_buf[r.strict_add(1)] = hi1.wrapping_add(K[r.strict_add(1)]);
        round!(lo0.wrapping_add(K[r]));
        round!(lo1.wrapping_add(K[r.strict_add(1)]));
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

      // Block 2: pure scalar from pre-computed buffer.
      a = state[0];
      b = state[1];
      c = state[2];
      d = state[3];
      e = state[4];
      f = state[5];
      g = state[6];
      h = state[7];
      t2_deferred = 0;

      for &wk in &t2_buf {
        round!(wk);
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

    // ── Single-block (odd trailing) ──────────────────────────────────────
    if remaining == 1 {
      let mut wv: [__m128i; 8] = [
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(16).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(32).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(48).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(64).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(80).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(96).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(112).cast()), bswap),
      ];

      let mut a = state[0];
      let mut b = state[1];
      let mut c = state[2];
      let mut d = state[3];
      let mut e = state[4];
      let mut f = state[5];
      let mut g = state[6];
      let mut h = state[7];
      let mut t2_deferred: u64 = 0;

      macro_rules! round1 {
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

      // Rounds 0-15: loaded words.
      for (pair, &v) in wv.iter().enumerate() {
        let (w0, w1) = extract_128(v);
        let r = pair.strict_mul(2);
        round1!(w0.wrapping_add(K[r]));
        round1!(w1.wrapping_add(K[r.strict_add(1)]));
      }

      // Rounds 16-79: interleaved schedule + rounds.
      for pair in 8..40usize {
        schedule_pair_128(&mut wv, pair);
        let (w0, w1) = extract_128(wv[pair & 7]);
        let r = pair.strict_mul(2);
        round1!(w0.wrapping_add(K[r]));
        round1!(w1.wrapping_add(K[r.strict_add(1)]));
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
    }
  } // unsafe
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotation-based schedule (eliminates ring-buffer index computation)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute one schedule pair using **array rotation** (128-bit, single-block).
///
/// Same approach as [`schedule_rotate_256`] but for 128-bit registers.
/// Eliminates ring-buffer index computation (`wrapping_sub`, `& 7`) in favour
/// of fixed-offset array accesses + physical rotation. The 7 register moves
/// are zero-cost on modern x86 (register renaming).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn schedule_rotate_128(x: &mut [__m128i; 8], k: __m128i) -> __m128i {
  // SAFETY: SSE2/SSSE3 intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    let w_tm15 = _mm_alignr_epi8(x[1], x[0], 8);
    let w_tm7 = _mm_alignr_epi8(x[5], x[4], 8);

    x[0] = _mm_add_epi64(
      _mm_add_epi64(x[0], w_tm7),
      _mm_add_epi64(small_sigma0_128(w_tm15), small_sigma1_128(x[7])),
    );

    let new_val = x[0];
    x[0] = x[1];
    x[1] = x[2];
    x[2] = x[3];
    x[3] = x[4];
    x[4] = x[5];
    x[5] = x[6];
    x[6] = x[7];
    x[7] = new_val;

    _mm_add_epi64(x[7], k)
  }
}

/// Compute one schedule pair using **array rotation** (256-bit, dual-block).
///
/// Produces raw W[t:t+1] for both blocks, stores in `x[7]` (after rotation),
/// and returns `W[t:t+1] + k` (with K constants pre-added for both blocks).
///
/// Unlike [`schedule_pair_256`], this function physically rotates the `x[]`
/// array so that adjacent schedule words are always in adjacent registers.
/// This lets `_mm256_alignr_epi8` extract cross-register values without
/// `_mm256_permute2x128_si256` (3-cycle latency on Zen 5). The 7 register
/// moves from rotation are zero-cost on modern x86 (register renaming).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn schedule_rotate_256(x: &mut [__m256i; 8], k: __m256i) -> __m256i {
  // SAFETY: AVX2 intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    // σ0 input: W[t-15:t-14] — straddles x[0] and x[1].
    let w_tm15 = _mm256_alignr_epi8(x[1], x[0], 8);
    // W[t-7:t-6] — straddles x[4] and x[5].
    let w_tm7 = _mm256_alignr_epi8(x[5], x[4], 8);

    // W[t:t+1] = W[t-16:t-15] + W[t-7:t-6] + σ0(W[t-15:t-14]) + σ1(W[t-2:t-1])
    x[0] = _mm256_add_epi64(
      _mm256_add_epi64(x[0], w_tm7),
      _mm256_add_epi64(small_sigma0_256(w_tm15), small_sigma1_256(x[7])),
    );

    // Rotate: x[0] (newest) → x[7], shift everything left.
    // Zero-cost on modern x86 via register renaming.
    let new_val = x[0];
    x[0] = x[1];
    x[1] = x[2];
    x[2] = x[3];
    x[3] = x[4];
    x[4] = x[5];
    x[5] = x[6];
    x[6] = x[7];
    x[7] = new_val;

    _mm256_add_epi64(x[7], k)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Decoupled kernel (schedule one-ahead of rounds, rotation-based schedule)
// ─────────────────────────────────────────────────────────────────────────────

/// SHA-512 multi-block compression using AVX2 + BMI2, **decoupled schedule
/// with rotation-based message expansion**.
///
/// Two key optimisations over the stitched kernel:
///
/// 1. **Decoupled schedule/rounds** — rounds consume W+K values from the *previous* 16-round group
///    while the SIMD schedule computes the *next* group. Gives the OOO engine 16 independent scalar
///    rounds to overlap with SIMD schedule latency.
///
/// 2. **Rotation-based schedule** — the `w[]` array is physically rotated after each schedule
///    update so adjacent words stay in adjacent registers. This lets `_mm256_alignr_epi8` extract
///    cross-register values directly, eliminating `_mm256_permute2x128_si256` (3-cycle latency on
///    Zen 5). The 7 register moves from rotation are zero-cost (register renaming).
///
/// Uses the **standard round** (Σ0 and Σ1 computed independently within
/// each round for maximum within-round parallelism).
///
/// # Safety
///
/// Caller must ensure `avx2` and `bmi2` CPU features are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2")]
pub(crate) unsafe fn compress_blocks_avx2_decoupled(state: &mut [u64; 8], blocks: &[u8]) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
  let num_blocks = blocks.len() / BLOCK_LEN;
  if num_blocks == 0 {
    return;
  }

  // SAFETY: AVX2/BMI2 intrinsics and target-feature-gated calls are available via this
  // function's #[target_feature] attribute. Pointer arithmetic is bounded by blocks.len().
  unsafe {
    let mut ptr = blocks.as_ptr();
    let mut remaining = num_blocks;

    let bswap = _mm_loadu_si128(BSWAP64_128.as_ptr().cast());

    // Round-value buffer: holds one group of 16 W[t]+K[t] values for block 1.
    // All 16 entries are written (load or schedule) before rounds consume them,
    // so the initial value is irrelevant. Hoisted to avoid stack reallocation.
    let mut rv = [0u64; 16];

    // Block-2 round-value buffer: holds all 80 W[t]+K[t] for block 2.
    let mut t2_buf = [0u64; 80];

    // ── Dual-block loop ──────────────────────────────────────────────────
    while remaining >= 2 {
      // Prefetch next block pair into L1D to hide L2 latency at large sizes.
      if remaining >= 4 {
        _mm_prefetch::<_MM_HINT_T0>(ptr.add(BLOCK_LEN.strict_mul(2)).cast());
        _mm_prefetch::<_MM_HINT_T0>(ptr.add(BLOCK_LEN.strict_mul(2).strict_add(64)).cast());
        _mm_prefetch::<_MM_HINT_T0>(ptr.add(BLOCK_LEN.strict_mul(3)).cast());
        _mm_prefetch::<_MM_HINT_T0>(ptr.add(BLOCK_LEN.strict_mul(3).strict_add(64)).cast());
      }

      let blk1 = ptr;
      let blk2 = ptr.add(BLOCK_LEN);

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

      // Extract initial W[0-15]+K[0-15] for both blocks.
      // K is added as a vector op (1 load + 1 broadcast + 1 add vs 4 scalar adds).
      for (i, &wv) in w.iter().enumerate() {
        let r = i.strict_mul(2);
        let k_pair: __m128i = _mm_loadu_si128(K.as_ptr().add(r).cast());
        let wk = _mm256_add_epi64(wv, _mm256_set_m128i(k_pair, k_pair));
        let (lo0, lo1) = extract_lo(wk);
        let (hi0, hi1) = extract_hi(wk);
        rv[r] = lo0;
        rv[r.strict_add(1)] = lo1;
        t2_buf[r] = hi0;
        t2_buf[r.strict_add(1)] = hi1;
      }

      let mut a = state[0];
      let mut b = state[1];
      let mut c = state[2];
      let mut d = state[3];
      let mut e = state[4];
      let mut f = state[5];
      let mut g = state[6];
      let mut h = state[7];

      macro_rules! round {
        ($wk:expr) => {{
          let t1 = h
            .wrapping_add(big_sigma1(e))
            .wrapping_add(ch(e, f, g))
            .wrapping_add($wk);
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

      // Decoupled loop: 4 outer × 8 inner = 64 rounds + 64 schedule words.
      // Rounds consume rv[] from the PREVIOUS group (load or prior outer).
      // Schedule computes the NEXT group via rotation (no cross-lane permute).
      for outer in 0..4usize {
        for j in 0..8usize {
          // Consume 2 rounds from current rv[] (previous group).
          let r = j.strict_mul(2);
          round!(rv[r]);
          round!(rv[r.strict_add(1)]);

          // Compute next schedule pair via rotation (independent of rounds).
          let kr = 8usize.strict_add(outer.strict_mul(8)).strict_add(j).strict_mul(2);
          let k_pair: __m128i = _mm_loadu_si128(K.as_ptr().add(kr).cast());
          let wk = schedule_rotate_256(&mut w, _mm256_set_m128i(k_pair, k_pair));
          let (lo0, lo1) = extract_lo(wk);
          let (hi0, hi1) = extract_hi(wk);

          // Store for next group (overwrite consumed rv[] slots).
          rv[r] = lo0;
          rv[r.strict_add(1)] = lo1;
          t2_buf[kr] = hi0;
          t2_buf[kr.strict_add(1)] = hi1;
        }
      }

      // Final 16 rounds (64-79) from last rv[] — pure scalar.
      for &wk in &rv {
        round!(wk);
      }

      state[0] = state[0].wrapping_add(a);
      state[1] = state[1].wrapping_add(b);
      state[2] = state[2].wrapping_add(c);
      state[3] = state[3].wrapping_add(d);
      state[4] = state[4].wrapping_add(e);
      state[5] = state[5].wrapping_add(f);
      state[6] = state[6].wrapping_add(g);
      state[7] = state[7].wrapping_add(h);

      // Block 2: pure scalar from pre-computed buffer.
      a = state[0];
      b = state[1];
      c = state[2];
      d = state[3];
      e = state[4];
      f = state[5];
      g = state[6];
      h = state[7];

      for &wk in &t2_buf {
        round!(wk);
      }

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

    // ── Single-block (odd trailing) ──────────────────────────────────────
    // Uses rotation-based 128-bit schedule (same approach as 256-bit) and
    // vector K addition for consistency with the dual-block path.
    if remaining == 1 {
      let mut wv: [__m128i; 8] = [
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(16).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(32).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(48).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(64).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(80).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(96).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(112).cast()), bswap),
      ];

      // Extract initial W[0-15]+K[0-15] with vector K addition.
      for (i, &v) in wv.iter().enumerate() {
        let r = i.strict_mul(2);
        let k_pair: __m128i = _mm_loadu_si128(K.as_ptr().add(r).cast());
        let wk = _mm_add_epi64(v, k_pair);
        let (w0, w1) = extract_128(wk);
        rv[r] = w0;
        rv[r.strict_add(1)] = w1;
      }

      let mut a = state[0];
      let mut b = state[1];
      let mut c = state[2];
      let mut d = state[3];
      let mut e = state[4];
      let mut f = state[5];
      let mut g = state[6];
      let mut h = state[7];

      macro_rules! round1 {
        ($wk:expr) => {{
          let t1 = h
            .wrapping_add(big_sigma1(e))
            .wrapping_add(ch(e, f, g))
            .wrapping_add($wk);
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

      // Decoupled loop: 4 outer × 8 inner = 64 rounds + 64 schedule words.
      // Rotation-based schedule with vector K addition.
      for outer in 0..4usize {
        for j in 0..8usize {
          let r = j.strict_mul(2);
          round1!(rv[r]);
          round1!(rv[r.strict_add(1)]);

          let kr = 8usize.strict_add(outer.strict_mul(8)).strict_add(j).strict_mul(2);
          let k_pair: __m128i = _mm_loadu_si128(K.as_ptr().add(kr).cast());
          let wk = schedule_rotate_128(&mut wv, k_pair);
          let (w0, w1) = extract_128(wk);
          rv[r] = w0;
          rv[r.strict_add(1)] = w1;
        }
      }

      // Final 16 rounds (64-79).
      for &wk in &rv {
        round1!(wk);
      }

      state[0] = state[0].wrapping_add(a);
      state[1] = state[1].wrapping_add(b);
      state[2] = state[2].wrapping_add(c);
      state[3] = state[3].wrapping_add(d);
      state[4] = state[4].wrapping_add(e);
      state[5] = state[5].wrapping_add(f);
      state[6] = state[6].wrapping_add(g);
      state[7] = state[7].wrapping_add(h);
    }
  } // unsafe
}

// ─────────────────────────────────────────────────────────────────────────────
// Standard-round variant (non-deferred Σ0)
// ─────────────────────────────────────────────────────────────────────────────

/// SHA-512 multi-block compression using AVX2 + BMI2, **standard round**.
///
/// Identical architecture to [`compress_blocks_avx2`] (stitched dual-block
/// SIMD schedule + scalar rounds, block-2 replay from pre-computed buffer)
/// but uses the standard SHA-512 round function where `a = T1 + Σ0(a) + Maj`
/// is computed immediately rather than deferred to the next round.
///
/// The standard round has more within-round parallelism (Σ0 and Σ1 are
/// independent), which benefits wide-pipeline CPUs (Zen 5, 6-wide dispatch).
/// The deferred variant has a shorter serial dependency chain, which benefits
/// narrow-pipeline CPUs (Zen 4, 4-wide dispatch).
///
/// # Safety
///
/// Caller must ensure `avx2` and `bmi2` CPU features are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2")]
pub(crate) unsafe fn compress_blocks_avx2_std(state: &mut [u64; 8], blocks: &[u8]) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
  let num_blocks = blocks.len() / BLOCK_LEN;
  if num_blocks == 0 {
    return;
  }

  // SAFETY: AVX2/BMI2 intrinsics and target-feature-gated calls are available via this
  // function's #[target_feature] attribute. Pointer arithmetic is bounded by blocks.len().
  unsafe {
    let mut ptr = blocks.as_ptr();
    let mut remaining = num_blocks;

    let bswap = _mm_loadu_si128(BSWAP64_128.as_ptr().cast());

    let mut t2_buf = [0u64; 80];

    // ── Dual-block loop ──────────────────────────────────────────────────
    while remaining >= 2 {
      // Prefetch next block pair into L1D to hide L2 latency at large sizes.
      if remaining >= 4 {
        _mm_prefetch::<_MM_HINT_T0>(ptr.add(BLOCK_LEN.strict_mul(2)).cast());
        _mm_prefetch::<_MM_HINT_T0>(ptr.add(BLOCK_LEN.strict_mul(2).strict_add(64)).cast());
        _mm_prefetch::<_MM_HINT_T0>(ptr.add(BLOCK_LEN.strict_mul(3)).cast());
        _mm_prefetch::<_MM_HINT_T0>(ptr.add(BLOCK_LEN.strict_mul(3).strict_add(64)).cast());
      }

      let blk1 = ptr;
      let blk2 = ptr.add(BLOCK_LEN);

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

      let mut a = state[0];
      let mut b = state[1];
      let mut c = state[2];
      let mut d = state[3];
      let mut e = state[4];
      let mut f = state[5];
      let mut g = state[6];
      let mut h = state[7];

      macro_rules! round_std {
        ($wk:expr) => {{
          let t1 = h
            .wrapping_add(big_sigma1(e))
            .wrapping_add(ch(e, f, g))
            .wrapping_add($wk);
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

      // Rounds 0-15: loaded words.
      for (pair, &wv) in w.iter().enumerate() {
        let (lo0, lo1) = extract_lo(wv);
        let (hi0, hi1) = extract_hi(wv);
        let r = pair.strict_mul(2);
        t2_buf[r] = hi0.wrapping_add(K[r]);
        t2_buf[r.strict_add(1)] = hi1.wrapping_add(K[r.strict_add(1)]);
        round_std!(lo0.wrapping_add(K[r]));
        round_std!(lo1.wrapping_add(K[r.strict_add(1)]));
      }

      // Rounds 16-79: interleaved schedule + rounds.
      for pair in 8..40usize {
        schedule_pair_256(&mut w, pair);
        let (lo0, lo1) = extract_lo(w[pair & 7]);
        let (hi0, hi1) = extract_hi(w[pair & 7]);
        let r = pair.strict_mul(2);
        t2_buf[r] = hi0.wrapping_add(K[r]);
        t2_buf[r.strict_add(1)] = hi1.wrapping_add(K[r.strict_add(1)]);
        round_std!(lo0.wrapping_add(K[r]));
        round_std!(lo1.wrapping_add(K[r.strict_add(1)]));
      }

      state[0] = state[0].wrapping_add(a);
      state[1] = state[1].wrapping_add(b);
      state[2] = state[2].wrapping_add(c);
      state[3] = state[3].wrapping_add(d);
      state[4] = state[4].wrapping_add(e);
      state[5] = state[5].wrapping_add(f);
      state[6] = state[6].wrapping_add(g);
      state[7] = state[7].wrapping_add(h);

      // Block 2: pure scalar from pre-computed buffer.
      a = state[0];
      b = state[1];
      c = state[2];
      d = state[3];
      e = state[4];
      f = state[5];
      g = state[6];
      h = state[7];

      for &wk in &t2_buf {
        round_std!(wk);
      }

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

    // ── Single-block (odd trailing) ──────────────────────────────────────
    if remaining == 1 {
      let mut wv: [__m128i; 8] = [
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(16).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(32).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(48).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(64).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(80).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(96).cast()), bswap),
        _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(112).cast()), bswap),
      ];

      let mut a = state[0];
      let mut b = state[1];
      let mut c = state[2];
      let mut d = state[3];
      let mut e = state[4];
      let mut f = state[5];
      let mut g = state[6];
      let mut h = state[7];

      macro_rules! round1_std {
        ($wk:expr) => {{
          let t1 = h
            .wrapping_add(big_sigma1(e))
            .wrapping_add(ch(e, f, g))
            .wrapping_add($wk);
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

      // Rounds 0-15: loaded words.
      for (pair, &v) in wv.iter().enumerate() {
        let (w0, w1) = extract_128(v);
        let r = pair.strict_mul(2);
        round1_std!(w0.wrapping_add(K[r]));
        round1_std!(w1.wrapping_add(K[r.strict_add(1)]));
      }

      // Rounds 16-79: interleaved schedule + rounds.
      for pair in 8..40usize {
        schedule_pair_128(&mut wv, pair);
        let (w0, w1) = extract_128(wv[pair & 7]);
        let r = pair.strict_mul(2);
        round1_std!(w0.wrapping_add(K[r]));
        round1_std!(w1.wrapping_add(K[r.strict_add(1)]));
      }

      state[0] = state[0].wrapping_add(a);
      state[1] = state[1].wrapping_add(b);
      state[2] = state[2].wrapping_add(c);
      state[3] = state[3].wrapping_add(d);
      state[4] = state[4].wrapping_add(e);
      state[5] = state[5].wrapping_add(f);
      state[6] = state[6].wrapping_add(g);
      state[7] = state[7].wrapping_add(h);
    }
  } // unsafe
}
