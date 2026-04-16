//! Blake2b AVX2 and AVX-512VL accelerated compression for x86_64.
//!
//! Each row of the 4x4 u64 working matrix is packed into a single `__m256i`
//! (4 lanes of u64). Diagonalization uses `_mm256_permute4x64_epi64`.
//!
//! Two kernels:
//! - **AVX2** (`compress_avx2`): rotations via shuffle/shift.
//! - **AVX-512VL** (`compress_avx512vl`): rotations via `VPRORQ` (`_mm256_ror_epi64`), eliminating
//!   the byte-shuffle tables entirely.
//!
//! # Safety
//!
//! `compress_avx2` requires `avx2`. `compress_avx512vl` requires `avx512f` + `avx512vl`.

#![allow(clippy::cast_possible_truncation, clippy::indexing_slicing)]

use core::arch::x86_64::*;

use super::kernels::{SIGMA, init_v, load_msg};

// ═════════════════════════════════════════════════════════════════════════════
// AVX2 kernel
// ═════════════════════════════════════════════════════════════════════════════

// ─── AVX2 rotation helpers ───────────────────────────────────────────────────

/// ROR 32: swap 32-bit halves within each 64-bit lane.
#[inline(always)]
unsafe fn ror32_avx2(x: __m256i) -> __m256i {
  // SAFETY: AVX2 intrinsics are available via the caller's #[target_feature] attribute.
  // shuffle_epi32 with 0xB1 = (2,3,0,1) swaps adjacent 32-bit pairs.
  unsafe { _mm256_shuffle_epi32(x, 0xB1) }
}

/// ROR 24: byte shuffle within each 128-bit lane.
#[inline(always)]
unsafe fn ror24_avx2(x: __m256i) -> __m256i {
  // SAFETY: AVX2 intrinsics are available via the caller's #[target_feature] attribute.
  // Each 64-bit lane rotates its bytes right by 3 (= rotate_right(24)).
  #[repr(align(32))]
  struct Align32([u8; 32]);
  static ROT24: Align32 = Align32([
    3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10, 3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10,
  ]);
  unsafe {
    let mask = _mm256_load_si256(ROT24.0.as_ptr().cast());
    _mm256_shuffle_epi8(x, mask)
  }
}

/// ROR 16: byte shuffle within each 128-bit lane.
#[inline(always)]
unsafe fn ror16_avx2(x: __m256i) -> __m256i {
  // SAFETY: AVX2 intrinsics are available via the caller's #[target_feature] attribute.
  // Each 64-bit lane rotates its bytes right by 2 (= rotate_right(16)).
  #[repr(align(32))]
  struct Align32([u8; 32]);
  static ROT16: Align32 = Align32([
    2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9, 2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9,
  ]);
  unsafe {
    let mask = _mm256_load_si256(ROT16.0.as_ptr().cast());
    _mm256_shuffle_epi8(x, mask)
  }
}

/// ROR 63: (x >> 63) | (x << 1) = (x + x) ^ (x >> 63).
#[inline(always)]
unsafe fn ror63_avx2(x: __m256i) -> __m256i {
  // SAFETY: AVX2 intrinsics are available via the caller's #[target_feature] attribute.
  unsafe { _mm256_xor_si256(_mm256_srli_epi64(x, 63), _mm256_add_epi64(x, x)) }
}

// ─── AVX2 G function ─────────────────────────────────────────────────────────

/// Blake2b quarter-round G on 4-wide AVX2 rows.
#[inline(always)]
unsafe fn g_avx2(a: &mut __m256i, b: &mut __m256i, c: &mut __m256i, d: &mut __m256i, mx: __m256i, my: __m256i) {
  // SAFETY: AVX2 intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    // a += b + mx
    *a = _mm256_add_epi64(_mm256_add_epi64(*a, *b), mx);
    // d = (d ^ a) >>> 32
    *d = ror32_avx2(_mm256_xor_si256(*d, *a));
    // c += d
    *c = _mm256_add_epi64(*c, *d);
    // b = (b ^ c) >>> 24
    *b = ror24_avx2(_mm256_xor_si256(*b, *c));
    // a += b + my
    *a = _mm256_add_epi64(_mm256_add_epi64(*a, *b), my);
    // d = (d ^ a) >>> 16
    *d = ror16_avx2(_mm256_xor_si256(*d, *a));
    // c += d
    *c = _mm256_add_epi64(*c, *d);
    // b = (b ^ c) >>> 63
    *b = ror63_avx2(_mm256_xor_si256(*b, *c));
  }
}

// ─── Diagonalize / un-diagonalize ────────────────────────────────────────────

/// Diagonalize: rotate B left by 1, C left by 2, D left by 3.
#[inline(always)]
unsafe fn diagonalize(b: &mut __m256i, c: &mut __m256i, d: &mut __m256i) {
  // SAFETY: AVX2 intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    // B: left 1 = (1,2,3,0) → imm8 = 0b_00_11_10_01 = 0x39
    *b = _mm256_permute4x64_epi64(*b, 0x39);
    // C: left 2 = swap pairs = (2,3,0,1) → imm8 = 0b_01_00_11_10 = 0x4E
    *c = _mm256_permute4x64_epi64(*c, 0x4E);
    // D: left 3 = right 1 = (3,0,1,2) → imm8 = 0b_10_01_00_11 = 0x93
    *d = _mm256_permute4x64_epi64(*d, 0x93);
  }
}

/// Un-diagonalize: reverse the rotations (B right 1, C swap, D left 1).
#[inline(always)]
unsafe fn undiagonalize(b: &mut __m256i, c: &mut __m256i, d: &mut __m256i) {
  // SAFETY: AVX2 intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    // B: right 1 = (3,0,1,2) → imm8 = 0x93
    *b = _mm256_permute4x64_epi64(*b, 0x93);
    // C: swap (same as diagonalize)
    *c = _mm256_permute4x64_epi64(*c, 0x4E);
    // D: left 1 = (1,2,3,0) → imm8 = 0x39
    *d = _mm256_permute4x64_epi64(*d, 0x39);
  }
}

// ─── AVX2 compress entry point ───────────────────────────────────────────────

/// Blake2b AVX2-accelerated compress.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn compress_avx2(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  // SAFETY: AVX2 intrinsics are available via this function's #[target_feature] attribute.
  unsafe {
    let m = load_msg(block);
    let v = init_v(h, t, last);

    // Pack the 16-word working vector into 4 rows of __m256i
    let mut a = _mm256_loadu_si256(v.as_ptr().cast()); // v[0..4]
    let mut b = _mm256_loadu_si256(v.as_ptr().add(4).cast()); // v[4..8]
    let mut c = _mm256_loadu_si256(v.as_ptr().add(8).cast()); // v[8..12]
    let mut d = _mm256_loadu_si256(v.as_ptr().add(12).cast()); // v[12..16]

    // 12 rounds of column + diagonal mixing
    for round in 0..12u8 {
      let s = &SIGMA[(round % 10) as usize];

      // Column step: G on (0,4,8,12), (1,5,9,13), (2,6,10,14), (3,7,11,15)
      // _mm256_set_epi64x takes args in HIGH→LOW order: lane3, lane2, lane1, lane0
      let mx = _mm256_set_epi64x(
        m[s[6] as usize] as i64,
        m[s[4] as usize] as i64,
        m[s[2] as usize] as i64,
        m[s[0] as usize] as i64,
      );
      let my = _mm256_set_epi64x(
        m[s[7] as usize] as i64,
        m[s[5] as usize] as i64,
        m[s[3] as usize] as i64,
        m[s[1] as usize] as i64,
      );

      g_avx2(&mut a, &mut b, &mut c, &mut d, mx, my);

      diagonalize(&mut b, &mut c, &mut d);

      // Diagonal step: G on (0,5,10,15), (1,6,11,12), (2,7,8,13), (3,4,9,14)
      let mx = _mm256_set_epi64x(
        m[s[14] as usize] as i64,
        m[s[12] as usize] as i64,
        m[s[10] as usize] as i64,
        m[s[8] as usize] as i64,
      );
      let my = _mm256_set_epi64x(
        m[s[15] as usize] as i64,
        m[s[13] as usize] as i64,
        m[s[11] as usize] as i64,
        m[s[9] as usize] as i64,
      );

      g_avx2(&mut a, &mut b, &mut c, &mut d, mx, my);

      undiagonalize(&mut b, &mut c, &mut d);
    }

    // Finalize: h[i] ^= v[i] ^ v[i+8]
    // a = v[0..4], b = v[4..8], c = v[8..12], d = v[12..16]
    // h[0..4] ^= a ^ c, h[4..8] ^= b ^ d
    let h0 = _mm256_loadu_si256(h.as_ptr().cast());
    let h1 = _mm256_loadu_si256(h.as_ptr().add(4).cast());

    _mm256_storeu_si256(h.as_mut_ptr().cast(), _mm256_xor_si256(h0, _mm256_xor_si256(a, c)));
    _mm256_storeu_si256(
      h.as_mut_ptr().add(4).cast(),
      _mm256_xor_si256(h1, _mm256_xor_si256(b, d)),
    );
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// AVX-512VL kernel
// ═════════════════════════════════════════════════════════════════════════════

// ─── AVX-512VL G function ────────────────────────────────────────────────────

/// Blake2b quarter-round G on 4-wide AVX-512VL rows.
///
/// All rotations use `VPRORQ` (`_mm256_ror_epi64`) — no shuffle tables needed.
#[inline(always)]
unsafe fn g_avx512vl(a: &mut __m256i, b: &mut __m256i, c: &mut __m256i, d: &mut __m256i, mx: __m256i, my: __m256i) {
  // SAFETY: AVX-512VL intrinsics are available via the caller's #[target_feature] attribute.
  unsafe {
    // a += b + mx
    *a = _mm256_add_epi64(_mm256_add_epi64(*a, *b), mx);
    // d = (d ^ a) >>> 32
    *d = _mm256_ror_epi64(_mm256_xor_si256(*d, *a), 32);
    // c += d
    *c = _mm256_add_epi64(*c, *d);
    // b = (b ^ c) >>> 24
    *b = _mm256_ror_epi64(_mm256_xor_si256(*b, *c), 24);
    // a += b + my
    *a = _mm256_add_epi64(_mm256_add_epi64(*a, *b), my);
    // d = (d ^ a) >>> 16
    *d = _mm256_ror_epi64(_mm256_xor_si256(*d, *a), 16);
    // c += d
    *c = _mm256_add_epi64(*c, *d);
    // b = (b ^ c) >>> 63
    *b = _mm256_ror_epi64(_mm256_xor_si256(*b, *c), 63);
  }
}

// ─── AVX-512VL compress entry point ──────────────────────────────────────────

/// Blake2b AVX-512VL-accelerated compress.
///
/// Same structure as AVX2 but uses `VPRORQ` for all rotations, eliminating
/// the byte-shuffle lookup tables.
///
/// # Safety
///
/// Caller must ensure AVX-512F and AVX-512VL are available.
#[target_feature(enable = "avx512f,avx512vl")]
pub(super) unsafe fn compress_avx512vl(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  // SAFETY: AVX-512VL intrinsics are available via this function's #[target_feature] attribute.
  // diagonalize/undiagonalize use only AVX2 permutes, which are available under AVX-512VL.
  unsafe {
    let m = load_msg(block);
    let v = init_v(h, t, last);

    // Pack the 16-word working vector into 4 rows of __m256i
    let mut a = _mm256_loadu_si256(v.as_ptr().cast()); // v[0..4]
    let mut b = _mm256_loadu_si256(v.as_ptr().add(4).cast()); // v[4..8]
    let mut c = _mm256_loadu_si256(v.as_ptr().add(8).cast()); // v[8..12]
    let mut d = _mm256_loadu_si256(v.as_ptr().add(12).cast()); // v[12..16]

    // 12 rounds of column + diagonal mixing
    for round in 0..12u8 {
      let s = &SIGMA[(round % 10) as usize];

      // Column step
      let mx = _mm256_set_epi64x(
        m[s[6] as usize] as i64,
        m[s[4] as usize] as i64,
        m[s[2] as usize] as i64,
        m[s[0] as usize] as i64,
      );
      let my = _mm256_set_epi64x(
        m[s[7] as usize] as i64,
        m[s[5] as usize] as i64,
        m[s[3] as usize] as i64,
        m[s[1] as usize] as i64,
      );

      g_avx512vl(&mut a, &mut b, &mut c, &mut d, mx, my);

      diagonalize(&mut b, &mut c, &mut d);

      // Diagonal step
      let mx = _mm256_set_epi64x(
        m[s[14] as usize] as i64,
        m[s[12] as usize] as i64,
        m[s[10] as usize] as i64,
        m[s[8] as usize] as i64,
      );
      let my = _mm256_set_epi64x(
        m[s[15] as usize] as i64,
        m[s[13] as usize] as i64,
        m[s[11] as usize] as i64,
        m[s[9] as usize] as i64,
      );

      g_avx512vl(&mut a, &mut b, &mut c, &mut d, mx, my);

      undiagonalize(&mut b, &mut c, &mut d);
    }

    // Finalize: h[i] ^= v[i] ^ v[i+8]
    let h0 = _mm256_loadu_si256(h.as_ptr().cast());
    let h1 = _mm256_loadu_si256(h.as_ptr().add(4).cast());

    _mm256_storeu_si256(h.as_mut_ptr().cast(), _mm256_xor_si256(h0, _mm256_xor_si256(a, c)));
    _mm256_storeu_si256(
      h.as_mut_ptr().add(4).cast(),
      _mm256_xor_si256(h1, _mm256_xor_si256(b, d)),
    );
  }
}
