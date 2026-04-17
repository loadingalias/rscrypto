//! Blake2s SIMD compression for x86_64.
//!
//! The Blake2s working state is naturally a 4×4 matrix of `u32`, so each row
//! fits exactly in one 128-bit SIMD register. The AVX2 path uses register-wise
//! shift/or rotates; the AVX-512VL path upgrades those rotates to `VPRORD`.
//!
//! This is a real SIMD backend, not a dispatch stub.

#![allow(clippy::cast_possible_wrap, clippy::indexing_slicing)]

use core::arch::x86_64::*;

use super::kernels::{SIGMA, init_v, load_msg};

#[inline(always)]
unsafe fn ror16_avx2(x: __m128i) -> __m128i {
  // SAFETY: SSE2 intrinsics operate only on the provided SIMD register.
  unsafe { _mm_or_si128(_mm_srli_epi32(x, 16), _mm_slli_epi32(x, 16)) }
}

#[inline(always)]
unsafe fn ror12_avx2(x: __m128i) -> __m128i {
  // SAFETY: SSE2 intrinsics operate only on the provided SIMD register.
  unsafe { _mm_or_si128(_mm_srli_epi32(x, 12), _mm_slli_epi32(x, 20)) }
}

#[inline(always)]
unsafe fn ror8_avx2(x: __m128i) -> __m128i {
  // SAFETY: SSE2 intrinsics operate only on the provided SIMD register.
  unsafe { _mm_or_si128(_mm_srli_epi32(x, 8), _mm_slli_epi32(x, 24)) }
}

#[inline(always)]
unsafe fn ror7_avx2(x: __m128i) -> __m128i {
  // SAFETY: SSE2 intrinsics operate only on the provided SIMD register.
  unsafe { _mm_or_si128(_mm_srli_epi32(x, 7), _mm_slli_epi32(x, 25)) }
}

#[inline(always)]
unsafe fn g_avx2(a: &mut __m128i, b: &mut __m128i, c: &mut __m128i, d: &mut __m128i, mx: __m128i, my: __m128i) {
  // SAFETY: all operations stay within SIMD registers and mutate disjoint rows.
  unsafe {
    *a = _mm_add_epi32(_mm_add_epi32(*a, *b), mx);
    *d = ror16_avx2(_mm_xor_si128(*d, *a));
    *c = _mm_add_epi32(*c, *d);
    *b = ror12_avx2(_mm_xor_si128(*b, *c));
    *a = _mm_add_epi32(_mm_add_epi32(*a, *b), my);
    *d = ror8_avx2(_mm_xor_si128(*d, *a));
    *c = _mm_add_epi32(*c, *d);
    *b = ror7_avx2(_mm_xor_si128(*b, *c));
  }
}

#[inline(always)]
unsafe fn diagonalize(b: &mut __m128i, c: &mut __m128i, d: &mut __m128i) {
  // SAFETY: shuffle permutes only the provided SIMD registers.
  unsafe {
    *b = _mm_shuffle_epi32(*b, 0x39);
    *c = _mm_shuffle_epi32(*c, 0x4E);
    *d = _mm_shuffle_epi32(*d, 0x93);
  }
}

#[inline(always)]
unsafe fn undiagonalize(b: &mut __m128i, c: &mut __m128i, d: &mut __m128i) {
  // SAFETY: shuffle permutes only the provided SIMD registers.
  unsafe {
    *b = _mm_shuffle_epi32(*b, 0x93);
    *c = _mm_shuffle_epi32(*c, 0x4E);
    *d = _mm_shuffle_epi32(*d, 0x39);
  }
}

#[inline(always)]
unsafe fn load_msg_quad(m: &[u32; 16], i0: u8, i1: u8, i2: u8, i3: u8) -> __m128i {
  // SAFETY: `_mm_set_epi32` constructs a register from in-bounds message words.
  unsafe {
    _mm_set_epi32(
      m[i3 as usize] as i32,
      m[i2 as usize] as i32,
      m[i1 as usize] as i32,
      m[i0 as usize] as i32,
    )
  }
}

/// Blake2s AVX2-selected SIMD compress.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn compress_avx2(h: &mut [u32; 8], block: &[u8; 64], t: u64, last: bool) {
  // SAFETY: the target feature guarantees the SIMD instructions used here are available.
  unsafe {
    let m = load_msg(block);
    let v = init_v(h, t, last);

    let mut a = _mm_loadu_si128(v.as_ptr().cast());
    let mut b = _mm_loadu_si128(v.as_ptr().add(4).cast());
    let mut c = _mm_loadu_si128(v.as_ptr().add(8).cast());
    let mut d = _mm_loadu_si128(v.as_ptr().add(12).cast());

    for round in 0..10u8 {
      let s = &SIGMA[round as usize];

      let mx = load_msg_quad(&m, s[0], s[2], s[4], s[6]);
      let my = load_msg_quad(&m, s[1], s[3], s[5], s[7]);
      g_avx2(&mut a, &mut b, &mut c, &mut d, mx, my);

      diagonalize(&mut b, &mut c, &mut d);

      let mx = load_msg_quad(&m, s[8], s[10], s[12], s[14]);
      let my = load_msg_quad(&m, s[9], s[11], s[13], s[15]);
      g_avx2(&mut a, &mut b, &mut c, &mut d, mx, my);

      undiagonalize(&mut b, &mut c, &mut d);
    }

    let h0 = _mm_loadu_si128(h.as_ptr().cast());
    let h1 = _mm_loadu_si128(h.as_ptr().add(4).cast());

    _mm_storeu_si128(h.as_mut_ptr().cast(), _mm_xor_si128(h0, _mm_xor_si128(a, c)));
    _mm_storeu_si128(h.as_mut_ptr().add(4).cast(), _mm_xor_si128(h1, _mm_xor_si128(b, d)));
  }
}

#[inline(always)]
unsafe fn g_avx512vl(a: &mut __m128i, b: &mut __m128i, c: &mut __m128i, d: &mut __m128i, mx: __m128i, my: __m128i) {
  // SAFETY: all operations stay within SIMD registers and the required target
  // features are enabled by the caller.
  unsafe {
    *a = _mm_add_epi32(_mm_add_epi32(*a, *b), mx);
    *d = _mm_ror_epi32(_mm_xor_si128(*d, *a), 16);
    *c = _mm_add_epi32(*c, *d);
    *b = _mm_ror_epi32(_mm_xor_si128(*b, *c), 12);
    *a = _mm_add_epi32(_mm_add_epi32(*a, *b), my);
    *d = _mm_ror_epi32(_mm_xor_si128(*d, *a), 8);
    *c = _mm_add_epi32(*c, *d);
    *b = _mm_ror_epi32(_mm_xor_si128(*b, *c), 7);
  }
}

/// Blake2s AVX-512VL-selected SIMD compress.
///
/// # Safety
///
/// Caller must ensure AVX-512F and AVX-512VL are available.
#[target_feature(enable = "avx512f,avx512vl")]
pub(super) unsafe fn compress_avx512vl(h: &mut [u32; 8], block: &[u8; 64], t: u64, last: bool) {
  // SAFETY: the target feature guarantees the SIMD instructions used here are available.
  unsafe {
    let m = load_msg(block);
    let v = init_v(h, t, last);

    let mut a = _mm_loadu_si128(v.as_ptr().cast());
    let mut b = _mm_loadu_si128(v.as_ptr().add(4).cast());
    let mut c = _mm_loadu_si128(v.as_ptr().add(8).cast());
    let mut d = _mm_loadu_si128(v.as_ptr().add(12).cast());

    for round in 0..10u8 {
      let s = &SIGMA[round as usize];

      let mx = load_msg_quad(&m, s[0], s[2], s[4], s[6]);
      let my = load_msg_quad(&m, s[1], s[3], s[5], s[7]);
      g_avx512vl(&mut a, &mut b, &mut c, &mut d, mx, my);

      diagonalize(&mut b, &mut c, &mut d);

      let mx = load_msg_quad(&m, s[8], s[10], s[12], s[14]);
      let my = load_msg_quad(&m, s[9], s[11], s[13], s[15]);
      g_avx512vl(&mut a, &mut b, &mut c, &mut d, mx, my);

      undiagonalize(&mut b, &mut c, &mut d);
    }

    let h0 = _mm_loadu_si128(h.as_ptr().cast());
    let h1 = _mm_loadu_si128(h.as_ptr().add(4).cast());

    _mm_storeu_si128(h.as_mut_ptr().cast(), _mm_xor_si128(h0, _mm_xor_si128(a, c)));
    _mm_storeu_si128(h.as_mut_ptr().add(4).cast(), _mm_xor_si128(h1, _mm_xor_si128(b, d)));
  }
}
