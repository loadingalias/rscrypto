//! SHA-512 x86_64 AVX-512VL software kernel.
//!
//! Processes the message schedule using `__m128i` (128-bit registers) with
//! AVX-512VL operations:
//! - **VPRORQ** (`_mm_ror_epi64`): native 64-bit vector rotate in 1 op (vs 3 ops with
//!   shift+shift+or on AVX2)
//!
//! Compression rounds remain in scalar GPRs (sequential dependency).
//! Uses a 2-word-at-a-time ring buffer (8 × `__m128i`) for the schedule.
//!
//! Available on Intel Ice Lake+ (2019+) and AMD Zen 4+ (2022+).
//! 256-bit register width avoids frequency throttling.
//!
//! Expected: ~15-20% improvement over portable from reduced rotation overhead.
//!
//! # Safety
//!
//! All functions require `avx512f` and `avx512vl` target features.

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::inline_always)]
#![allow(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::indexing_slicing)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{BLOCK_LEN, K, big_sigma0, big_sigma1, ch, maj};

// ─────────────────────────────────────────────────────────────────────────────
// SIMD sigma functions using VPRORQ (AVX-512VL)
// ─────────────────────────────────────────────────────────────────────────────

/// σ0(x) = ROTR(1) ^ ROTR(8) ^ SHR(7)  — 1 op per rotate with VPRORQ.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn small_sigma0_v(x: __m128i) -> __m128i {
  _mm_xor_si128(
    _mm_xor_si128(_mm_ror_epi64(x, 1), _mm_ror_epi64(x, 8)),
    _mm_srli_epi64(x, 7),
  )
}

/// σ1(x) = ROTR(19) ^ ROTR(61) ^ SHR(6)  — 1 op per rotate with VPRORQ.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn small_sigma1_v(x: __m128i) -> __m128i {
  _mm_xor_si128(
    _mm_xor_si128(_mm_ror_epi64(x, 19), _mm_ror_epi64(x, 61)),
    _mm_srli_epi64(x, 6),
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Message schedule helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-register extraction: [a[1], b[0]] (cross-pair lane extraction).
/// Uses PALIGNR (SSSE3): concat(b, a) >> 8 bytes = [a[1], b[0]].
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn cross_lanes(a: __m128i, b: __m128i) -> __m128i {
  _mm_alignr_epi8(b, a, 8)
}

/// Compute 2 message schedule words using SIMD with VPRORQ.
///
/// Ring buffer `w` has 8 slots, each holding `[W[2j], W[2j+1]]`.
/// Called for pairs i = 8..39 (schedule words 16..79).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn schedule_pair(w: &mut [__m128i; 8], i: usize) {
  // σ1 input: [W[2i-2], W[2i-1]]
  let s1_in = w[i.wrapping_sub(1) & 7];
  // W[t-7]: [W[2i-7], W[2i-6]] — crosses a pair boundary
  let w_tm7 = cross_lanes(w[i.wrapping_sub(4) & 7], w[i.wrapping_sub(3) & 7]);
  // σ0 input: [W[2i-15], W[2i-14]] — crosses a pair boundary
  let s0_in = cross_lanes(w[i.wrapping_sub(8) & 7], w[i.wrapping_sub(7) & 7]);
  // W[t-16]: [W[2i-16], W[2i-15]] — exact pair
  let w_tm16 = w[i.wrapping_sub(8) & 7];

  w[i & 7] = _mm_add_epi64(
    _mm_add_epi64(small_sigma1_v(s1_in), w_tm7),
    _mm_add_epi64(small_sigma0_v(s0_in), w_tm16),
  );
}

/// Byte-swap mask for converting little-endian loads to big-endian u64 words.
#[cfg(target_arch = "x86_64")]
static BSWAP64_128: [u8; 16] = [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8];

/// SHA-512 multi-block compression using AVX-512VL.
///
/// Message schedule uses VPRORQ for efficient 64-bit rotations.
/// Compression rounds remain scalar.
///
/// # Safety
///
/// Caller must ensure `avx512f` and `avx512vl` CPU features are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl")]
pub(crate) unsafe fn compress_blocks_avx512vl(state: &mut [u64; 8], blocks: &[u8]) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
  if blocks.is_empty() {
    return;
  }

  let bswap = _mm_loadu_si128(BSWAP64_128.as_ptr().cast());
  let num_blocks = blocks.len() / BLOCK_LEN;
  let mut ptr = blocks.as_ptr();

  for _ in 0..num_blocks {
    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    // Load and byte-swap 16 message words into 8 × __m128i ring buffer.
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

    // Rounds 0-15: extract from loaded vectors.
    for (pair, &v) in wv.iter().enumerate() {
      let w_lo = _mm_extract_epi64(v, 0) as u64;
      let w_hi = _mm_extract_epi64(v, 1) as u64;
      let r = pair.strict_mul(2);
      sha_round!(K[r], w_lo);
      sha_round!(K[r + 1], w_hi);
    }

    // Rounds 16-79: expand schedule with VPRORQ, then extract for scalar rounds.
    for pair in 8..40usize {
      schedule_pair(&mut wv, pair);
      let v = wv[pair & 7];
      let w_lo = _mm_extract_epi64(v, 0) as u64;
      let w_hi = _mm_extract_epi64(v, 1) as u64;
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

    ptr = ptr.add(128);
  }
}
