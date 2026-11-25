//! SHA-512 x86_64 AVX2 + BMI2 decoupled kernel.
//!
//! Uses the Gueron-Krasnov two-block parallel technique with **decoupled**
//! schedule/compress and **rotation-based** message expansion.
//!
//! Architecture:
//! - **Dual-block** (≥ 2 blocks): 256-bit schedule (`__m256i`, `_mm256_alignr_epi8` rotation, 3-op
//!   rotates), decoupled from scalar rounds. Block 2's `W[t]+K[t]` stored during block 1's pass.
//! - **Single-block** (odd trailing): 128-bit schedule (`__m128i`, `_mm_alignr_epi8` rotation, 3-op
//!   rotates), decoupled from scalar rounds. No portable fallback — fully self-contained.
//!
//! BMI2 (`bmi2` target feature) enables the compiler to emit `RORX` for scalar
//! 64-bit rotations — a non-destructive, flag-free rotate that improves
//! register allocation and throughput.
//!
//! Available on all x86_64 CPUs since Haswell (2013+).
//!
//! **Competitive advantage**: BoringSSL's AVX2 SHA-512 path is **disabled**
//! (CFI annotation issues). `ring` inherits this — it has NO AVX2 for SHA-512.
//!
//! # Safety
//!
//! All functions require `avx2` and `bmi2` target features.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{BLOCK_LEN, K, big_sigma0, big_sigma1, ch, maj};

// 256-bit SIMD sigma (dual-block schedule, 2 × u64 per 128-bit lane)

/// σ0(x) = ROTR(1) ^ ROTR(8) ^ SHR(7)  — 256-bit, 3-op rotates.
///
/// # Safety
/// Caller must execute this helper only from this module's AVX2 target-feature scope.
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
///
/// # Safety
/// Caller must execute this helper only from this module's AVX2 target-feature scope.
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

// 128-bit SIMD sigma (single-block schedule, 2 × u64 per register)

/// σ0(x) = ROTR(1) ^ ROTR(8) ^ SHR(7)  — 128-bit, 3-op rotates.
///
/// # Safety
/// Caller must execute this helper only from this module's AVX2 target-feature scope.
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
///
/// # Safety
/// Caller must execute this helper only from this module's AVX2 target-feature scope.
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

// Common helpers

/// Byte-swap mask for big-endian u64 word loads (128-bit).
#[cfg(target_arch = "x86_64")]
static BSWAP64_128: [u8; 16] = [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8];

/// Load 2 big-endian u64 words from each of two blocks into a __m256i.
///
/// # Safety
/// Caller must execute this helper only from this module's AVX2 target-feature scope. Both block
/// pointers must remain valid for 16 readable bytes starting at `offset`.
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
///
/// # Safety
/// Caller must execute this helper only from this module's AVX2 target-feature scope.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn extract_lo(v: __m256i) -> (u64, u64) {
  // SAFETY: SSE intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    let lo128 = _mm256_castsi256_si128(v);
    (
      _mm_extract_epi64(lo128, 0).cast_unsigned(),
      _mm_extract_epi64(lo128, 1).cast_unsigned(),
    )
  }
}

/// Extract two u64 words from the upper 128-bit lane of a __m256i.
///
/// # Safety
/// Caller must execute this helper only from this module's AVX2 target-feature scope.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn extract_hi(v: __m256i) -> (u64, u64) {
  // SAFETY: AVX2 intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    let hi128 = _mm256_extracti128_si256(v, 1);
    (
      _mm_extract_epi64(hi128, 0).cast_unsigned(),
      _mm_extract_epi64(hi128, 1).cast_unsigned(),
    )
  }
}

/// Extract two u64 words from a __m128i.
///
/// # Safety
/// Caller must execute this helper only from this module's AVX2 target-feature scope.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn extract_128(v: __m128i) -> (u64, u64) {
  // SAFETY: SSE intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    (
      _mm_extract_epi64(v, 0).cast_unsigned(),
      _mm_extract_epi64(v, 1).cast_unsigned(),
    )
  }
}

// Rotation-based schedule (eliminates ring-buffer index computation)

/// Compute one schedule pair using **array rotation** (128-bit, single-block).
///
/// Same approach as [`schedule_rotate_256`] but for 128-bit registers.
/// Eliminates ring-buffer index computation (`wrapping_sub`, `& 7`) in favour
/// of fixed-offset array accesses and physical rotation.
///
/// # Safety
/// Caller must execute this helper only from this module's AVX2 target-feature scope.
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
/// Physically rotates the `x[]` array so that adjacent schedule words are
/// always in adjacent registers. This lets `_mm256_alignr_epi8` extract
/// cross-register values without `_mm256_permute2x128_si256`.
///
/// # Safety
/// Caller must execute this helper only from this module's AVX2 target-feature scope.
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
    // Rotate the logical ring to keep adjacent words in adjacent registers.
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

// Decoupled kernel (schedule one-ahead of rounds, rotation-based schedule)

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
///    cross-register values directly, eliminating `_mm256_permute2x128_si256`.
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
  sha512_decoupled_blocks!(state, blocks, prefetch = true);
}
