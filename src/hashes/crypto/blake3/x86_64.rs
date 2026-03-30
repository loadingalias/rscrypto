//! BLAKE3 x86_64 SIMD kernels.
//!
//! This module provides SIMD-accelerated compression functions for BLAKE3 using
//! x86_64 intrinsics (SSE4.1, AVX2, AVX-512).
//!
//! Runtime strategy contract:
//! - AVX-512 > AVX2 > SSE4.1 > scalar (portable)
//!
//! # Safety
//!
//! All functions in this module are marked `unsafe` and require specific CPU
//! features to be present. Callers must verify CPU capabilities before calling.

#![allow(unsafe_code)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
pub(crate) mod asm;
pub(crate) mod avx2;
pub(crate) mod avx512;
pub(crate) mod sse41;

use super::{BLOCK_LEN, CHUNK_START, IV, PARENT};

// Shared helpers for SIMD kernels.

#[inline(always)]
pub(crate) const fn counter_low(counter: u64) -> u32 {
  counter as u32
}

#[inline(always)]
pub(crate) const fn counter_high(counter: u64) -> u32 {
  (counter >> 32) as u32
}

// ─────────────────────────────────────────────────────────────────────────────
// CV-only compression helpers (avoid `[u32; 16]` materialization)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn load_msg_vecs(block: *const u8) -> (__m128i, __m128i, __m128i, __m128i) {
  // SAFETY: Caller guarantees block pointer is valid for 64 bytes. Intrinsics require SSSE3 via
  // caller's #[target_feature].
  unsafe {
    let m0 = _mm_loadu_si128(block.cast());
    let m1 = _mm_loadu_si128(block.add(16).cast());
    let m2 = _mm_loadu_si128(block.add(32).cast());
    let m3 = _mm_loadu_si128(block.add(48).cast());
    (m0, m1, m2, m3)
  }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1,ssse3")]
pub(crate) unsafe fn compress_in_place_sse41_bytes(
  chaining_value: &mut [u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
) {
  // SAFETY: SSE4.1/SSSE3 intrinsics are available via this function's #[target_feature] attribute.
  unsafe {
    let (m0, m1, m2, m3) = load_msg_vecs(block);
    let [row0, row1, row2, row3] = compress_pre_sse41_impl(chaining_value, m0, m1, m2, m3, counter, block_len, flags);
    _mm_storeu_si128(chaining_value.as_mut_ptr().cast(), _mm_xor_si128(row0, row2));
    _mm_storeu_si128(chaining_value.as_mut_ptr().add(4).cast(), _mm_xor_si128(row1, row3));
  }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1,ssse3")]
pub(crate) unsafe fn compress_cv_sse41_bytes(
  chaining_value: &[u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  // SAFETY: SSE4.1/SSSE3 intrinsics are available via this function's #[target_feature] attribute.
  unsafe {
    let (m0, m1, m2, m3) = load_msg_vecs(block);
    let [row0, row1, row2, row3] = compress_pre_sse41_impl(chaining_value, m0, m1, m2, m3, counter, block_len, flags);
    let row0 = _mm_xor_si128(row0, row2);
    let row1 = _mm_xor_si128(row1, row3);

    let mut out = [0u32; 8];
    _mm_storeu_si128(out.as_mut_ptr().cast(), row0);
    _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), row1);
    out
  }
}

pub(crate) unsafe fn compress_cv_avx2_bytes(
  chaining_value: &[u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  // SAFETY: AVX2/SSE4.1/SSSE3 intrinsics are available via caller's target_feature guarantee.
  unsafe {
    let (m0, m1, m2, m3) = load_msg_vecs(block);
    compress_cv_avx2(chaining_value, m0, m1, m2, m3, counter, block_len, flags)
  }
}

// On ASM-supported platforms, we prefer the handwritten assembly. This intrinsics
// version is kept as fallback for other x86_64 platforms (e.g., FreeBSD, illumos).
#[cfg(target_arch = "x86_64")]
#[cfg_attr(
  any(target_os = "linux", target_os = "macos", target_os = "windows"),
  allow(dead_code)
)]
#[target_feature(enable = "avx512f,avx512vl,avx2,sse4.1,ssse3")]
pub(crate) unsafe fn compress_in_place_avx512_bytes(
  chaining_value: &mut [u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
) {
  // SAFETY: AVX-512/AVX2/SSE4.1/SSSE3 intrinsics are available via this function's #[target_feature]
  // attribute.
  unsafe {
    let (m0, m1, m2, m3) = load_msg_vecs(block);
    let [row0, row1, row2, row3] = compress_pre_sse41_impl(chaining_value, m0, m1, m2, m3, counter, block_len, flags);
    _mm_storeu_si128(chaining_value.as_mut_ptr().cast(), _mm_xor_si128(row0, row2));
    _mm_storeu_si128(chaining_value.as_mut_ptr().add(4).cast(), _mm_xor_si128(row1, row3));
  }
}

#[cfg(target_arch = "x86_64")]
#[cfg_attr(
  any(target_os = "linux", target_os = "macos", target_os = "windows"),
  allow(dead_code)
)]
#[target_feature(enable = "avx512f,avx512vl,avx2,sse4.1,ssse3")]
pub(crate) unsafe fn compress_cv_avx512_bytes(
  chaining_value: &[u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  // SAFETY: AVX-512/AVX2/SSE4.1/SSSE3 intrinsics are available via this function's #[target_feature]
  // attribute.
  unsafe {
    let (m0, m1, m2, m3) = load_msg_vecs(block);
    compress_cv_avx512(chaining_value, m0, m1, m2, m3, counter, block_len, flags)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SSE4.1 / AVX2 / AVX-512 per-block compressor (world-class schedule)
// ─────────────────────────────────────────────────────────────────────────────
//
// This schedule is adapted from the upstream BLAKE3 project's high-performance
// `rust_sse41` compressor (CC0-1.0 / Apache-2.0 / LLVM-exception). It avoids
// the expensive per-round gather/shuffle machinery in the SSSE3 row-wise
// implementation by keeping the message in a permuted form and applying a
// fixed, low-instruction permutation each round.

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn rot16_sse41(a: __m128i) -> __m128i {
  // SAFETY: SSE2 intrinsics are available via caller's #[target_feature] attribute.
  unsafe { _mm_or_si128(_mm_srli_epi32(a, 16), _mm_slli_epi32(a, 16)) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn rot12_sse41(a: __m128i) -> __m128i {
  // SAFETY: SSE2 intrinsics are available via caller's #[target_feature] attribute.
  unsafe { _mm_or_si128(_mm_srli_epi32(a, 12), _mm_slli_epi32(a, 20)) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn rot8_sse41(a: __m128i) -> __m128i {
  // SAFETY: SSE2 intrinsics are available via caller's #[target_feature] attribute.
  unsafe { _mm_or_si128(_mm_srli_epi32(a, 8), _mm_slli_epi32(a, 24)) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn rot7_sse41(a: __m128i) -> __m128i {
  // SAFETY: SSE2 intrinsics are available via caller's #[target_feature] attribute.
  unsafe { _mm_or_si128(_mm_srli_epi32(a, 7), _mm_slli_epi32(a, 25)) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn g1_sse41(row0: &mut __m128i, row1: &mut __m128i, row2: &mut __m128i, row3: &mut __m128i, m: __m128i) {
  // SAFETY: SSE4.1/SSSE3 intrinsics are available via caller's #[target_feature] attribute.
  unsafe {
    *row0 = _mm_add_epi32(_mm_add_epi32(*row0, m), *row1);
    *row3 = _mm_xor_si128(*row3, *row0);
    *row3 = rot16_sse41(*row3);
    *row2 = _mm_add_epi32(*row2, *row3);
    *row1 = _mm_xor_si128(*row1, *row2);
    *row1 = rot12_sse41(*row1);
  }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn g2_sse41(row0: &mut __m128i, row1: &mut __m128i, row2: &mut __m128i, row3: &mut __m128i, m: __m128i) {
  // SAFETY: SSE4.1/SSSE3 intrinsics are available via caller's #[target_feature] attribute.
  unsafe {
    *row0 = _mm_add_epi32(_mm_add_epi32(*row0, m), *row1);
    *row3 = _mm_xor_si128(*row3, *row0);
    *row3 = rot8_sse41(*row3);
    *row2 = _mm_add_epi32(*row2, *row3);
    *row1 = _mm_xor_si128(*row1, *row2);
    *row1 = rot7_sse41(*row1);
  }
}

macro_rules! _MM_SHUFFLE {
  ($z:expr, $y:expr, $x:expr, $w:expr) => {
    ($z << 6) | ($y << 4) | ($x << 2) | $w
  };
}

macro_rules! shuffle2 {
  ($a:expr, $b:expr, $c:expr) => {
    _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps($a), _mm_castsi128_ps($b), $c))
  };
}

// Leave row1 unrotated and diagonalize the other rows.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn diagonalize_sse41(row0: &mut __m128i, row2: &mut __m128i, row3: &mut __m128i) {
  // SAFETY: SSE2 intrinsics are available via caller's #[target_feature] attribute.
  unsafe {
    *row0 = _mm_shuffle_epi32(*row0, _MM_SHUFFLE!(2, 1, 0, 3));
    *row3 = _mm_shuffle_epi32(*row3, _MM_SHUFFLE!(1, 0, 3, 2));
    *row2 = _mm_shuffle_epi32(*row2, _MM_SHUFFLE!(0, 3, 2, 1));
  }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn undiagonalize_sse41(row0: &mut __m128i, row2: &mut __m128i, row3: &mut __m128i) {
  // SAFETY: SSE2 intrinsics are available via caller's #[target_feature] attribute.
  unsafe {
    *row0 = _mm_shuffle_epi32(*row0, _MM_SHUFFLE!(0, 3, 2, 1));
    *row3 = _mm_shuffle_epi32(*row3, _MM_SHUFFLE!(1, 0, 3, 2));
    *row2 = _mm_shuffle_epi32(*row2, _MM_SHUFFLE!(2, 1, 0, 3));
  }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn compress_pre_sse41_impl(
  chaining_value: &[u32; 8],
  mut m0: __m128i,
  mut m1: __m128i,
  mut m2: __m128i,
  mut m3: __m128i,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [__m128i; 4] {
  // SAFETY: SSE4.1/SSSE3 intrinsics are available via caller's #[target_feature] attribute.
  unsafe {
    let mut row0 = _mm_loadu_si128(chaining_value.as_ptr().cast());
    let mut row1 = _mm_loadu_si128(chaining_value.as_ptr().add(4).cast());
    let mut row2 = _mm_setr_epi32(
      IV[0].cast_signed(),
      IV[1].cast_signed(),
      IV[2].cast_signed(),
      IV[3].cast_signed(),
    );
    let mut row3 = _mm_setr_epi32(
      (counter as u32).cast_signed(),
      ((counter >> 32) as u32).cast_signed(),
      block_len.cast_signed(),
      flags.cast_signed(),
    );

    let mut t0;
    let mut t1;
    let mut t2;
    let mut t3;
    let mut tt;

    // Round 1
    t0 = shuffle2!(m0, m1, _MM_SHUFFLE!(2, 0, 2, 0));
    g1_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t0);
    t1 = shuffle2!(m0, m1, _MM_SHUFFLE!(3, 1, 3, 1));
    g2_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t1);
    diagonalize_sse41(&mut row0, &mut row2, &mut row3);
    t2 = shuffle2!(m2, m3, _MM_SHUFFLE!(2, 0, 2, 0));
    t2 = _mm_shuffle_epi32(t2, _MM_SHUFFLE!(2, 1, 0, 3));
    g1_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t2);
    t3 = shuffle2!(m2, m3, _MM_SHUFFLE!(3, 1, 3, 1));
    t3 = _mm_shuffle_epi32(t3, _MM_SHUFFLE!(2, 1, 0, 3));
    g2_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t3);
    undiagonalize_sse41(&mut row0, &mut row2, &mut row3);
    m0 = t0;
    m1 = t1;
    m2 = t2;
    m3 = t3;

    macro_rules! next_round_update {
      () => {{
        t0 = shuffle2!(m0, m1, _MM_SHUFFLE!(3, 1, 1, 2));
        t0 = _mm_shuffle_epi32(t0, _MM_SHUFFLE!(0, 3, 2, 1));
        g1_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t0);
        t1 = shuffle2!(m2, m3, _MM_SHUFFLE!(3, 3, 2, 2));
        tt = _mm_shuffle_epi32(m0, _MM_SHUFFLE!(0, 0, 3, 3));
        t1 = _mm_blend_epi16(tt, t1, 0xCC);
        g2_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t1);
        diagonalize_sse41(&mut row0, &mut row2, &mut row3);
        t2 = _mm_unpacklo_epi64(m3, m1);
        tt = _mm_blend_epi16(t2, m2, 0xC0);
        t2 = _mm_shuffle_epi32(tt, _MM_SHUFFLE!(1, 3, 2, 0));
        g1_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t2);
        t3 = _mm_unpackhi_epi32(m1, m3);
        tt = _mm_unpacklo_epi32(m2, t3);
        t3 = _mm_shuffle_epi32(tt, _MM_SHUFFLE!(0, 1, 3, 2));
        g2_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t3);
        undiagonalize_sse41(&mut row0, &mut row2, &mut row3);
        m0 = t0;
        m1 = t1;
        m2 = t2;
        m3 = t3;
      }};
    }

    macro_rules! next_round_final {
      () => {{
        t0 = shuffle2!(m0, m1, _MM_SHUFFLE!(3, 1, 1, 2));
        t0 = _mm_shuffle_epi32(t0, _MM_SHUFFLE!(0, 3, 2, 1));
        g1_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t0);
        t1 = shuffle2!(m2, m3, _MM_SHUFFLE!(3, 3, 2, 2));
        tt = _mm_shuffle_epi32(m0, _MM_SHUFFLE!(0, 0, 3, 3));
        t1 = _mm_blend_epi16(tt, t1, 0xCC);
        g2_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t1);
        diagonalize_sse41(&mut row0, &mut row2, &mut row3);
        t2 = _mm_unpacklo_epi64(m3, m1);
        tt = _mm_blend_epi16(t2, m2, 0xC0);
        t2 = _mm_shuffle_epi32(tt, _MM_SHUFFLE!(1, 3, 2, 0));
        g1_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t2);
        t3 = _mm_unpackhi_epi32(m1, m3);
        tt = _mm_unpacklo_epi32(m2, t3);
        t3 = _mm_shuffle_epi32(tt, _MM_SHUFFLE!(0, 1, 3, 2));
        g2_sse41(&mut row0, &mut row1, &mut row2, &mut row3, t3);
        undiagonalize_sse41(&mut row0, &mut row2, &mut row3);
      }};
    }

    // Rounds 2..6
    next_round_update!();
    next_round_update!();
    next_round_update!();
    next_round_update!();
    next_round_update!();
    // Round 7
    next_round_final!();

    [row0, row1, row2, row3]
  }
}

/// AVX2+SSSE3 compress that returns only the chaining value (first 8 words).
///
/// This mirrors `compress_cv_ssse3`, but is compiled under AVX2 to encourage
/// VEX-encoded 128-bit operations for better mixed-workload behavior.
///
/// # Safety
/// Caller must ensure AVX2 + SSSE3 are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,sse4.1,ssse3")]
unsafe fn compress_cv_avx2(
  chaining_value: &[u32; 8],
  m0: __m128i,
  m1: __m128i,
  m2: __m128i,
  m3: __m128i,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  // SAFETY: AVX2/SSE4.1/SSSE3 intrinsics are available via this function's #[target_feature]
  // attribute.
  unsafe {
    let [row0, row1, row2, row3] = compress_pre_sse41_impl(chaining_value, m0, m1, m2, m3, counter, block_len, flags);
    let row0 = _mm_xor_si128(row0, row2);
    let row1 = _mm_xor_si128(row1, row3);
    let mut out = [0u32; 8];
    _mm_storeu_si128(out.as_mut_ptr().cast(), row0);
    _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), row1);
    out
  }
}

/// AVX-512+AVX2+SSSE3 compress that returns only the chaining value (first 8 words).
///
/// # Safety
/// Caller must ensure the declared AVX-512 + AVX2 + SSSE3 features are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2,sse4.1,ssse3")]
unsafe fn compress_cv_avx512(
  chaining_value: &[u32; 8],
  m0: __m128i,
  m1: __m128i,
  m2: __m128i,
  m3: __m128i,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  // SAFETY: AVX-512/AVX2/SSE4.1/SSSE3 intrinsics are available via this function's #[target_feature]
  // attribute.
  unsafe {
    let [row0, row1, row2, row3] = compress_pre_sse41_impl(chaining_value, m0, m1, m2, m3, counter, block_len, flags);
    let row0 = _mm_xor_si128(row0, row2);
    let row1 = _mm_xor_si128(row1, row3);
    let mut out = [0u32; 8];
    _mm_storeu_si128(out.as_mut_ptr().cast(), row0);
    _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), row1);
    out
  }
}

/// BLAKE3 compress function using AVX2-enabled codegen.
///
/// Note: This is still a single-block (dependency-chained) compressor. AVX2
/// doesn't unlock extra parallelism inside the algorithm, but it does allow
/// the compiler to use VEX-encoded integer ops and avoid AVX<->SSE transition
/// penalties when interleaving with AVX2 throughput kernels.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,sse4.1,ssse3")]
pub(crate) unsafe fn compress_avx2(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  // SAFETY: AVX2/SSE4.1/SSSE3 intrinsics are available via this function's #[target_feature]
  // attribute.
  unsafe {
    let m0 = _mm_loadu_si128(block_words.as_ptr().cast());
    let m1 = _mm_loadu_si128(block_words.as_ptr().add(4).cast());
    let m2 = _mm_loadu_si128(block_words.as_ptr().add(8).cast());
    let m3 = _mm_loadu_si128(block_words.as_ptr().add(12).cast());
    let [mut row0, mut row1, mut row2, mut row3] =
      compress_pre_sse41_impl(chaining_value, m0, m1, m2, m3, counter, block_len, flags);

    let cv_lo = _mm_loadu_si128(chaining_value.as_ptr().cast());
    let cv_hi = _mm_loadu_si128(chaining_value.as_ptr().add(4).cast());

    row0 = _mm_xor_si128(row0, row2);
    row1 = _mm_xor_si128(row1, row3);
    row2 = _mm_xor_si128(row2, cv_lo);
    row3 = _mm_xor_si128(row3, cv_hi);

    let mut out = [0u32; 16];
    _mm_storeu_si128(out.as_mut_ptr().cast(), row0);
    _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), row1);
    _mm_storeu_si128(out.as_mut_ptr().add(8).cast(), row2);
    _mm_storeu_si128(out.as_mut_ptr().add(12).cast(), row3);
    out
  }
}

/// BLAKE3 compress function using SSE4.1-enabled codegen.
///
/// This is a thin wrapper around the SSSE3 row-wise compressor. SSE4.1 implies
/// (and our dispatcher requires) SSSE3, and this keeps the per-block hot path
/// fully SIMD without maintaining duplicate implementations.
///
/// # Safety
/// Caller must ensure SSE4.1 + SSSE3 are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1,ssse3")]
pub(crate) unsafe fn compress_sse41(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  // SAFETY: SSE4.1/SSSE3 intrinsics are available via this function's #[target_feature] attribute.
  unsafe {
    let m0 = _mm_loadu_si128(block_words.as_ptr().cast());
    let m1 = _mm_loadu_si128(block_words.as_ptr().add(4).cast());
    let m2 = _mm_loadu_si128(block_words.as_ptr().add(8).cast());
    let m3 = _mm_loadu_si128(block_words.as_ptr().add(12).cast());
    let [mut row0, mut row1, mut row2, mut row3] =
      compress_pre_sse41_impl(chaining_value, m0, m1, m2, m3, counter, block_len, flags);

    let cv_lo = _mm_loadu_si128(chaining_value.as_ptr().cast());
    let cv_hi = _mm_loadu_si128(chaining_value.as_ptr().add(4).cast());

    row0 = _mm_xor_si128(row0, row2);
    row1 = _mm_xor_si128(row1, row3);
    row2 = _mm_xor_si128(row2, cv_lo);
    row3 = _mm_xor_si128(row3, cv_hi);

    let mut out = [0u32; 16];
    _mm_storeu_si128(out.as_mut_ptr().cast(), row0);
    _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), row1);
    _mm_storeu_si128(out.as_mut_ptr().add(8).cast(), row2);
    _mm_storeu_si128(out.as_mut_ptr().add(12).cast(), row3);
    out
  }
}

/// SSE4.1 chunk compression: process multiple 64-byte blocks.
///
/// # Safety
/// Caller must ensure SSE4.1 + SSSE3 are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1,ssse3")]
pub(crate) unsafe fn chunk_compress_blocks_sse41(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  // SAFETY: SSE4.1/SSSE3 intrinsics are available via this function's #[target_feature] attribute.
  // Calls to unsafe asm/intrinsics helpers are valid under the same guarantee.
  unsafe {
    debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);

    if blocks.len() == BLOCK_LEN {
      let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
      #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
      {
        asm::compress_in_place_sse41_mut(
          chaining_value,
          blocks.as_ptr(),
          chunk_counter,
          BLOCK_LEN as u32,
          flags | start,
        );
      }
      #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
      {
        compress_in_place_sse41_bytes(
          chaining_value,
          blocks.as_ptr(),
          chunk_counter,
          BLOCK_LEN as u32,
          flags | start,
        );
      }
      *blocks_compressed = blocks_compressed.wrapping_add(1);
      return;
    }

    let (block_slices, remainder) = blocks.as_chunks::<BLOCK_LEN>();
    debug_assert!(remainder.is_empty());
    for block_bytes in block_slices {
      let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
      #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
      {
        asm::compress_in_place_sse41_mut(
          chaining_value,
          block_bytes.as_ptr(),
          chunk_counter,
          BLOCK_LEN as u32,
          flags | start,
        );
      }
      #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
      {
        compress_in_place_sse41_bytes(
          chaining_value,
          block_bytes.as_ptr(),
          chunk_counter,
          BLOCK_LEN as u32,
          flags | start,
        );
      }
      *blocks_compressed = blocks_compressed.wrapping_add(1);
    }
  }
}

/// SSE4.1 parent CV computation.
///
/// # Safety
/// Caller must ensure SSE4.1 + SSSE3 are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1,ssse3")]
pub(crate) unsafe fn parent_cv_sse41(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  // SAFETY: SSE4.1/SSSE3 intrinsics are available via this function's #[target_feature] attribute.
  unsafe {
    let m0 = _mm_loadu_si128(left_child_cv.as_ptr().cast());
    let m1 = _mm_loadu_si128(left_child_cv.as_ptr().add(4).cast());
    let m2 = _mm_loadu_si128(right_child_cv.as_ptr().cast());
    let m3 = _mm_loadu_si128(right_child_cv.as_ptr().add(4).cast());
    let [row0, row1, row2, row3] =
      compress_pre_sse41_impl(&key_words, m0, m1, m2, m3, 0, BLOCK_LEN as u32, PARENT | flags);
    let row0 = _mm_xor_si128(row0, row2);
    let row1 = _mm_xor_si128(row1, row3);
    let mut out = [0u32; 8];
    _mm_storeu_si128(out.as_mut_ptr().cast(), row0);
    _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), row1);
    out
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// High-level kernel wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// AVX2 chunk compression: process multiple 64-byte blocks.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,sse4.1,ssse3")]
pub(crate) unsafe fn chunk_compress_blocks_avx2(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  // SAFETY: AVX2/SSE4.1/SSSE3 intrinsics are available via this function's #[target_feature]
  // attribute. Calls to unsafe asm/intrinsics helpers are valid under the same guarantee.
  unsafe {
    debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);

    if blocks.len() == BLOCK_LEN {
      let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
      #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
      {
        asm::compress_in_place_avx2_mut(
          chaining_value,
          blocks.as_ptr(),
          chunk_counter,
          BLOCK_LEN as u32,
          flags | start,
        );
      }
      #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
      {
        *chaining_value = compress_cv_avx2_bytes(
          chaining_value,
          blocks.as_ptr(),
          chunk_counter,
          BLOCK_LEN as u32,
          flags | start,
        );
      }
      *blocks_compressed = blocks_compressed.wrapping_add(1);
      return;
    }

    let (block_slices, remainder) = blocks.as_chunks::<BLOCK_LEN>();
    debug_assert!(remainder.is_empty());
    for block_bytes in block_slices {
      let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
      #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
      {
        asm::compress_in_place_avx2_mut(
          chaining_value,
          block_bytes.as_ptr(),
          chunk_counter,
          BLOCK_LEN as u32,
          flags | start,
        );
      }
      #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
      {
        *chaining_value = compress_cv_avx2_bytes(
          chaining_value,
          block_bytes.as_ptr(),
          chunk_counter,
          BLOCK_LEN as u32,
          flags | start,
        );
      }
      *blocks_compressed = blocks_compressed.wrapping_add(1);
    }
  }
}

/// AVX2 parent CV computation.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,sse4.1,ssse3")]
pub(crate) unsafe fn parent_cv_avx2(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  // SAFETY: AVX2/SSE4.1/SSSE3 intrinsics are available via this function's #[target_feature]
  // attribute.
  unsafe {
    let m0 = _mm_loadu_si128(left_child_cv.as_ptr().cast());
    let m1 = _mm_loadu_si128(left_child_cv.as_ptr().add(4).cast());
    let m2 = _mm_loadu_si128(right_child_cv.as_ptr().cast());
    let m3 = _mm_loadu_si128(right_child_cv.as_ptr().add(4).cast());
    compress_cv_avx2(&key_words, m0, m1, m2, m3, 0, BLOCK_LEN as u32, PARENT | flags)
  }
}

/// BLAKE3 compress function using AVX-512-enabled codegen.
///
/// Like the AVX2 entrypoint, this is still a single-block compressor. The
/// primary benefit is avoiding transition penalties when the surrounding
/// workload uses AVX-512 throughput kernels.
///
/// # Safety
/// Caller must ensure AVX-512 + AVX2 are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2,sse4.1,ssse3")]
pub(crate) unsafe fn compress_avx512(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  // SAFETY: AVX-512/AVX2/SSE4.1/SSSE3 intrinsics are available via this function's #[target_feature]
  // attribute.
  unsafe { avx512::compress_block(chaining_value, block_words, counter, block_len, flags) }
}

/// AVX-512 chunk compression: process multiple 64-byte blocks.
///
/// # Safety
/// Caller must ensure AVX-512 + AVX2 are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2,sse4.1,ssse3")]
pub(crate) unsafe fn chunk_compress_blocks_avx512(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  // SAFETY: AVX-512/AVX2/SSE4.1/SSSE3 intrinsics are available via this function's #[target_feature]
  // attribute. Calls to unsafe asm/intrinsics helpers are valid under the same guarantee.
  unsafe {
    debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);

    if blocks.len() == BLOCK_LEN {
      let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
      #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
      {
        *chaining_value = asm::compress_in_place_avx512(
          chaining_value,
          blocks.as_ptr(),
          chunk_counter,
          BLOCK_LEN as u32,
          flags | start,
        );
      }
      #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
      {
        *chaining_value = compress_cv_avx512_bytes(
          chaining_value,
          blocks.as_ptr(),
          chunk_counter,
          BLOCK_LEN as u32,
          flags | start,
        );
      }
      *blocks_compressed = blocks_compressed.wrapping_add(1);
      return;
    }

    let (block_slices, remainder) = blocks.as_chunks::<BLOCK_LEN>();
    debug_assert!(remainder.is_empty());
    for block_bytes in block_slices {
      let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
      #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
      {
        *chaining_value = asm::compress_in_place_avx512(
          chaining_value,
          block_bytes.as_ptr(),
          chunk_counter,
          BLOCK_LEN as u32,
          flags | start,
        );
      }
      #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
      {
        *chaining_value = compress_cv_avx512_bytes(
          chaining_value,
          block_bytes.as_ptr(),
          chunk_counter,
          BLOCK_LEN as u32,
          flags | start,
        );
      }
      *blocks_compressed = blocks_compressed.wrapping_add(1);
    }
  }
}

/// AVX-512 parent CV computation.
///
/// # Safety
/// Caller must ensure AVX-512 + AVX2 are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2,sse4.1,ssse3")]
pub(crate) unsafe fn parent_cv_avx512(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  // SAFETY: AVX-512/AVX2/SSE4.1/SSSE3 intrinsics are available via this function's #[target_feature]
  // attribute.
  unsafe {
    let m0 = _mm_loadu_si128(left_child_cv.as_ptr().cast());
    let m1 = _mm_loadu_si128(left_child_cv.as_ptr().add(4).cast());
    let m2 = _mm_loadu_si128(right_child_cv.as_ptr().cast());
    let m3 = _mm_loadu_si128(right_child_cv.as_ptr().add(4).cast());
    compress_cv_avx512(&key_words, m0, m1, m2, m3, 0, BLOCK_LEN as u32, PARENT | flags)
  }
}

// ─────────────────────────────────────────────────
