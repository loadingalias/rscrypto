//! x86_64 AVX2 and AVX-512 BlaMka compression kernels for Argon2.
//!
//! Two kernels live in this file:
//!
//! - [`compress_avx2`] — 4-way parallel BlaMka P-round across YMM registers. Each (a, b, c, d)
//!   GB-lane of 4 u64s lives in one `__m256i`. Rotations via byte-shuffle masks and shift-or;
//!   BlaMka multiply via `VPMULUDQ`.
//!
//! - [`compress_avx512`] — Asymmetric ZMM/YMM design: 8-way 2-row batched for the row pass (where
//!   16-u64 chunks are contiguous in memory) and 4-way YMM with native `VPRORQ` rotations for the
//!   column pass (where loads are stride-16 in memory and a clean ZMM batch is not free). Layout
//!   transforms via `VSHUFI64X2`. The asymmetric design keeps the row pass at full 8-way ZMM
//!   throughput while avoiding the cost of per-iteration 4× `VPGATHERQQ`-style loads in the column
//!   pass.
//!
//! # Vectorisation topology (shared)
//!
//! A BlaMka P-round operates on 16 u64 words laid out as:
//!
//! ```text
//! v = [ a0 a1 a2 a3 | b0 b1 b2 b3 | c0 c1 c2 c3 | d0 d1 d2 d3 ]
//! ```
//!
//! Column step: `GB(a_i, b_i, c_i, d_i)` for `i ∈ 0..4` — already 4-way
//! parallel with lane `i` in SIMD position `i`.
//!
//! Diagonal step: `GB(a0,b1,c2,d3)`, `GB(a1,b2,c3,d0)`,
//! `GB(a2,b3,c0,d1)`, `GB(a3,b0,c1,d2)` — rotate the `b` lane by 1, `c`
//! lane by 2, `d` lane by 3 (via `VPERMQ` / `VPERMI2Q`), run the same
//! 4-way GB, then rotate back.
//!
//! # BlaMka multiply
//!
//! `2 · lsb(a) · lsb(b)` vectorises cleanly: `VPMULUDQ` returns the
//! u64 product of the low 32 bits of each 64-bit lane pair. A single
//! left shift by 1 supplies the `2 · …` factor.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::cast_possible_truncation)]

use core::arch::x86_64::{
  __m256i, __m512i, _mm_loadu_si128, _mm_storeu_si128, _mm256_add_epi64, _mm256_castsi128_si256,
  _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_inserti128_si256, _mm256_load_si256, _mm256_loadu_si256,
  _mm256_mul_epu32, _mm256_or_si256, _mm256_permute4x64_epi64, _mm256_ror_epi64, _mm256_shuffle_epi8,
  _mm256_shuffle_epi32, _mm256_slli_epi64, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_xor_si256, _mm512_add_epi64,
  _mm512_loadu_si512, _mm512_mul_epu32, _mm512_permutex_epi64, _mm512_ror_epi64, _mm512_shuffle_i64x2,
  _mm512_slli_epi64, _mm512_storeu_si512, _mm512_xor_si512,
};

use super::BLOCK_WORDS;

// ─── Rotation masks ─────────────────────────────────────────────────────────
//
// AVX2 ROR-24 / ROR-16 are byte shuffles. Each 64-bit lane within the YMM
// rotates its 8 bytes by 3 (for ROR-24) or by 2 (for ROR-16). The masks are
// duplicated across the two 128-bit halves of the YMM (`_mm256_shuffle_epi8`
// is per-128-bit-lane, like NEON `vqtbl1q_u8`).

#[repr(align(32))]
struct Align32([u8; 32]);

static ROT24_MASK: Align32 = Align32([
  3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10, 3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10,
]);

static ROT16_MASK: Align32 = Align32([
  2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9, 2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9,
]);

// ═══════════════════════════════════════════════════════════════════════════
// AVX2 kernel
// ═══════════════════════════════════════════════════════════════════════════

/// AVX2 BlaMka compression kernel.
///
/// 4-way parallel P-round per row (or per column) across 4 YMMs.
///
/// # Safety
///
/// - `target_arch = "x86_64"` enforced at compile time (module gate).
/// - Caller must have AVX2 available — enforced via `#[target_feature]`.
/// - The three `[u64; BLOCK_WORDS]` buffers are fixed-size arrays; the kernel reads/writes only
///   within their bounds.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn compress_avx2(
  dst: &mut [u64; BLOCK_WORDS],
  x: &[u64; BLOCK_WORDS],
  y: &[u64; BLOCK_WORDS],
  xor_into: bool,
) {
  // SAFETY: AVX2 is enabled by this function's `#[target_feature]`
  // attribute, so all `_mm256_*` intrinsics below are valid to call.
  unsafe {
    let mut r = [0u64; BLOCK_WORDS];
    let mut q = [0u64; BLOCK_WORDS];

    // R = X XOR Y, written to both `r` (preserved for the final XOR) and
    // `q` (working buffer). Using YMM stores is a 4× win over per-u64
    // stores even before the row/column passes start.
    let mut i = 0;
    while i < BLOCK_WORDS {
      let xv = _mm256_loadu_si256(x.as_ptr().add(i).cast());
      let yv = _mm256_loadu_si256(y.as_ptr().add(i).cast());
      let rv = _mm256_xor_si256(xv, yv);
      _mm256_storeu_si256(r.as_mut_ptr().add(i).cast(), rv);
      _mm256_storeu_si256(q.as_mut_ptr().add(i).cast(), rv);
      i += 4;
    }

    // Row pass: 8 P-rounds on contiguous 16-u64 chunks of q[].
    let mut row = 0usize;
    while row < 8 {
      let base = row * 16;
      let mut a = _mm256_loadu_si256(q.as_ptr().add(base).cast());
      let mut b = _mm256_loadu_si256(q.as_ptr().add(base + 4).cast());
      let mut c = _mm256_loadu_si256(q.as_ptr().add(base + 8).cast());
      let mut d = _mm256_loadu_si256(q.as_ptr().add(base + 12).cast());

      p_round_avx2(&mut a, &mut b, &mut c, &mut d);

      _mm256_storeu_si256(q.as_mut_ptr().add(base).cast(), a);
      _mm256_storeu_si256(q.as_mut_ptr().add(base + 4).cast(), b);
      _mm256_storeu_si256(q.as_mut_ptr().add(base + 8).cast(), c);
      _mm256_storeu_si256(q.as_mut_ptr().add(base + 12).cast(), d);
      row += 1;
    }

    // Column pass: 8 P-rounds on stride-16 u64 sequences. Each YMM holds
    // 2 contiguous u64 pairs from 2 different rows (low half = row 2k,
    // high half = row 2k+1) — see RFC 9106 §3.6 column-step indexing.
    let mut col = 0usize;
    while col < 8 {
      let base = col * 2;
      let mut a = load_col_pair_avx2(&q, base, base + 16);
      let mut b = load_col_pair_avx2(&q, base + 32, base + 48);
      let mut c = load_col_pair_avx2(&q, base + 64, base + 80);
      let mut d = load_col_pair_avx2(&q, base + 96, base + 112);

      p_round_avx2(&mut a, &mut b, &mut c, &mut d);

      store_col_pair_avx2(&mut q, base, base + 16, a);
      store_col_pair_avx2(&mut q, base + 32, base + 48, b);
      store_col_pair_avx2(&mut q, base + 64, base + 80, c);
      store_col_pair_avx2(&mut q, base + 96, base + 112, d);
      col += 1;
    }

    // Final XOR with R, fused with the dst store/xor.
    let mut i = 0;
    while i < BLOCK_WORDS {
      let qv = _mm256_loadu_si256(q.as_ptr().add(i).cast());
      let rv = _mm256_loadu_si256(r.as_ptr().add(i).cast());
      let f = _mm256_xor_si256(qv, rv);
      if xor_into {
        let cur = _mm256_loadu_si256(dst.as_ptr().add(i).cast());
        _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), _mm256_xor_si256(cur, f));
      } else {
        _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), f);
      }
      i += 4;
    }
  }
}

/// Load a column-pass GB-lane: 4 u64s = `[q[lo..lo+2], q[hi..hi+2]]`.
///
/// # Safety
///
/// Inherits AVX2 from the caller. `lo` and `hi` are passed by the column
/// pass driver as values in `{0, 2, 4, …, 14}` plus row offsets in
/// `{0, 16, 32, …, 112}`, so each ends + 2 ≤ `BLOCK_WORDS = 128`.
#[inline(always)]
unsafe fn load_col_pair_avx2(q: &[u64; BLOCK_WORDS], lo: usize, hi: usize) -> __m256i {
  // SAFETY: AVX2 inherited; `lo` and `hi` are bounded by the caller as
  // documented above, so the 16-byte loads stay in-bounds.
  unsafe {
    let lo_v = _mm_loadu_si128(q.as_ptr().add(lo).cast());
    let hi_v = _mm_loadu_si128(q.as_ptr().add(hi).cast());
    _mm256_inserti128_si256(_mm256_castsi128_si256(lo_v), hi_v, 1)
  }
}

/// Store a column-pass GB-lane back to its stride-16 positions.
///
/// # Safety
///
/// Same bounds argument as [`load_col_pair_avx2`].
#[inline(always)]
unsafe fn store_col_pair_avx2(q: &mut [u64; BLOCK_WORDS], lo: usize, hi: usize, v: __m256i) {
  // SAFETY: AVX2 inherited; bounds witnessed by the caller (see
  // `load_col_pair_avx2`).
  unsafe {
    _mm_storeu_si128(q.as_mut_ptr().add(lo).cast(), _mm256_castsi256_si128(v));
    _mm_storeu_si128(q.as_mut_ptr().add(hi).cast(), _mm256_extracti128_si256(v, 1));
  }
}

/// One BlaMka P-round on 4 YMM rows (4-way parallel GBs).
///
/// # Safety
///
/// Inherits AVX2 from the caller; all ops below are register-only.
#[inline(always)]
unsafe fn p_round_avx2(a: &mut __m256i, b: &mut __m256i, c: &mut __m256i, d: &mut __m256i) {
  // SAFETY: AVX2 inherited.
  unsafe {
    // Column step.
    gb_avx2(a, b, c, d);
    // Diagonalise: rotate b by 1, c by 2, d by 3 across the 4-lane row.
    *b = _mm256_permute4x64_epi64(*b, 0x39);
    *c = _mm256_permute4x64_epi64(*c, 0x4E);
    *d = _mm256_permute4x64_epi64(*d, 0x93);
    // Diagonal step.
    gb_avx2(a, b, c, d);
    // Undo diagonalisation.
    *b = _mm256_permute4x64_epi64(*b, 0x93);
    *c = _mm256_permute4x64_epi64(*c, 0x4E);
    *d = _mm256_permute4x64_epi64(*d, 0x39);
  }
}

/// 4-way parallel BlaMka G on YMM rows.
///
/// # Safety
///
/// Inherits AVX2 from the caller; register-only.
#[inline(always)]
unsafe fn gb_avx2(a: &mut __m256i, b: &mut __m256i, c: &mut __m256i, d: &mut __m256i) {
  // SAFETY: AVX2 inherited.
  unsafe {
    // a += b + 2·lsb(a)·lsb(b)
    let p = _mm256_mul_epu32(*a, *b);
    *a = _mm256_add_epi64(_mm256_add_epi64(*a, *b), _mm256_slli_epi64(p, 1));
    // d = (d ^ a) ROR 32
    *d = ror32_avx2(_mm256_xor_si256(*d, *a));

    // c += d + 2·lsb(c)·lsb(d)
    let p = _mm256_mul_epu32(*c, *d);
    *c = _mm256_add_epi64(_mm256_add_epi64(*c, *d), _mm256_slli_epi64(p, 1));
    // b = (b ^ c) ROR 24
    *b = ror24_avx2(_mm256_xor_si256(*b, *c));

    // a += b + 2·lsb(a)·lsb(b)
    let p = _mm256_mul_epu32(*a, *b);
    *a = _mm256_add_epi64(_mm256_add_epi64(*a, *b), _mm256_slli_epi64(p, 1));
    // d = (d ^ a) ROR 16
    *d = ror16_avx2(_mm256_xor_si256(*d, *a));

    // c += d + 2·lsb(c)·lsb(d)
    let p = _mm256_mul_epu32(*c, *d);
    *c = _mm256_add_epi64(_mm256_add_epi64(*c, *d), _mm256_slli_epi64(p, 1));
    // b = (b ^ c) ROR 63 ≡ ROL 1
    *b = ror63_avx2(_mm256_xor_si256(*b, *c));
  }
}

#[inline(always)]
unsafe fn ror32_avx2(x: __m256i) -> __m256i {
  // SAFETY: AVX2 inherited; shuffle imm 0xB1 swaps adjacent u32 halves.
  unsafe { _mm256_shuffle_epi32(x, 0xB1) }
}

#[inline(always)]
unsafe fn ror24_avx2(x: __m256i) -> __m256i {
  // SAFETY: AVX2 inherited; ROT24_MASK is 32-byte aligned static data.
  unsafe {
    let mask = _mm256_load_si256(ROT24_MASK.0.as_ptr().cast());
    _mm256_shuffle_epi8(x, mask)
  }
}

#[inline(always)]
unsafe fn ror16_avx2(x: __m256i) -> __m256i {
  // SAFETY: AVX2 inherited; ROT16_MASK is 32-byte aligned static data.
  unsafe {
    let mask = _mm256_load_si256(ROT16_MASK.0.as_ptr().cast());
    _mm256_shuffle_epi8(x, mask)
  }
}

#[inline(always)]
unsafe fn ror63_avx2(x: __m256i) -> __m256i {
  // SAFETY: AVX2 inherited; (x << 1) | (x >> 63).
  unsafe { _mm256_or_si256(_mm256_add_epi64(x, x), _mm256_srli_epi64(x, 63)) }
}

// ═══════════════════════════════════════════════════════════════════════════
// AVX-512 kernel (8-way ZMM row pass + 4-way YMM-VL column pass)
// ═══════════════════════════════════════════════════════════════════════════

/// AVX-512 BlaMka compression kernel.
///
/// Asymmetric layout: the row pass batches 2 contiguous 16-u64 rows per
/// P-round and runs an 8-way ZMM kernel; the column pass stays 4-way YMM
/// (with `VPRORQ` rotations) because column loads are stride-16 in
/// memory and a 2-column ZMM batch would require multiple `VPGATHERQQ`-
/// or `VPERMI2Q`-style operations whose cost dominates the SIMD width
/// gain.
///
/// # Safety
///
/// - `target_arch = "x86_64"` enforced at compile time.
/// - Caller must have AVX-512F + AVX-512VL available — enforced via `#[target_feature]`. F gives
///   ZMM-wide ops (used in the row pass); VL gives YMM-form `_mm256_ror_epi64` (used in the column
///   pass).
/// - The three `[u64; BLOCK_WORDS]` buffers are fixed-size arrays.
#[target_feature(enable = "avx512f,avx512vl")]
pub(super) unsafe fn compress_avx512(
  dst: &mut [u64; BLOCK_WORDS],
  x: &[u64; BLOCK_WORDS],
  y: &[u64; BLOCK_WORDS],
  xor_into: bool,
) {
  // SAFETY: AVX-512F + AVX-512VL enabled by this function's
  // `#[target_feature]`.
  unsafe {
    let mut r = [0u64; BLOCK_WORDS];
    let mut q = [0u64; BLOCK_WORDS];

    // R = X XOR Y at ZMM width.
    let mut i = 0;
    while i < BLOCK_WORDS {
      let xv = _mm512_loadu_si512(x.as_ptr().add(i).cast());
      let yv = _mm512_loadu_si512(y.as_ptr().add(i).cast());
      let rv = _mm512_xor_si512(xv, yv);
      _mm512_storeu_si512(r.as_mut_ptr().add(i).cast(), rv);
      _mm512_storeu_si512(q.as_mut_ptr().add(i).cast(), rv);
      i += 8;
    }

    // Row pass: 4 iterations × 2 rows per iter = 8 P-rounds done at
    // 8-way ZMM width.
    //
    // Layout transform per iter:
    //   r0_lo = q[r0_off..r0_off+8]   = [r0.a0..3, r0.b0..3]
    //   r0_hi = q[r0_off+8..r0_off+16]= [r0.c0..3, r0.d0..3]
    //   r1_lo = q[r0_off+16..r0_off+24] = [r1.a0..3, r1.b0..3]
    //   r1_hi = q[r0_off+24..r0_off+32] = [r1.c0..3, r1.d0..3]
    //
    //   a = [r0.a0..3, r1.a0..3]
    //   b = [r0.b0..3, r1.b0..3]
    //   c = [r0.c0..3, r1.c0..3]
    //   d = [r0.d0..3, r1.d0..3]
    //
    // Each `VSHUFI64X2` picks 4 of 8 contiguous 128-bit lanes (4 from
    // src1, 4 from src2) using a single imm8. The 0x44 imm picks lanes
    // 0,1 from each source; 0xEE picks lanes 2,3.
    let mut iter = 0;
    while iter < 4 {
      let off = iter * 32;

      let r0_lo = _mm512_loadu_si512(q.as_ptr().add(off).cast());
      let r0_hi = _mm512_loadu_si512(q.as_ptr().add(off + 8).cast());
      let r1_lo = _mm512_loadu_si512(q.as_ptr().add(off + 16).cast());
      let r1_hi = _mm512_loadu_si512(q.as_ptr().add(off + 24).cast());

      let mut a = _mm512_shuffle_i64x2(r0_lo, r1_lo, 0x44);
      let mut b = _mm512_shuffle_i64x2(r0_lo, r1_lo, 0xEE);
      let mut c = _mm512_shuffle_i64x2(r0_hi, r1_hi, 0x44);
      let mut d = _mm512_shuffle_i64x2(r0_hi, r1_hi, 0xEE);

      p_round_avx512(&mut a, &mut b, &mut c, &mut d);

      // Reverse permute and store back. The shuffle pattern is symmetric:
      // `r0_lo` recombines from `(a, b)` with the same 0x44 imm, etc.
      let r0_lo_out = _mm512_shuffle_i64x2(a, b, 0x44);
      let r0_hi_out = _mm512_shuffle_i64x2(c, d, 0x44);
      let r1_lo_out = _mm512_shuffle_i64x2(a, b, 0xEE);
      let r1_hi_out = _mm512_shuffle_i64x2(c, d, 0xEE);

      _mm512_storeu_si512(q.as_mut_ptr().add(off).cast(), r0_lo_out);
      _mm512_storeu_si512(q.as_mut_ptr().add(off + 8).cast(), r0_hi_out);
      _mm512_storeu_si512(q.as_mut_ptr().add(off + 16).cast(), r1_lo_out);
      _mm512_storeu_si512(q.as_mut_ptr().add(off + 24).cast(), r1_hi_out);
      iter += 1;
    }

    // Column pass: 4-way YMM with native `VPRORQ` rotations.
    //
    // Two-column ZMM batching would reduce the iteration count from 8 to
    // 4 but each iter would need 16 × `_mm_loadu_si128` + 8 × `_mm256_set_m128i`
    // + 4 × `_mm512_inserti64x4` to assemble its 4 ZMMs from stride-16
    // memory positions — measurably more expensive than the 4-way
    // single-column kernel below, which only needs 4 × `_mm_loadu_si128` +
    // 2 × insert per GB-lane.
    let mut col = 0usize;
    while col < 8 {
      let base = col * 2;
      let mut a = load_col_pair_avx2(&q, base, base + 16);
      let mut b = load_col_pair_avx2(&q, base + 32, base + 48);
      let mut c = load_col_pair_avx2(&q, base + 64, base + 80);
      let mut d = load_col_pair_avx2(&q, base + 96, base + 112);

      p_round_avx512vl(&mut a, &mut b, &mut c, &mut d);

      store_col_pair_avx2(&mut q, base, base + 16, a);
      store_col_pair_avx2(&mut q, base + 32, base + 48, b);
      store_col_pair_avx2(&mut q, base + 64, base + 80, c);
      store_col_pair_avx2(&mut q, base + 96, base + 112, d);
      col += 1;
    }

    // Final XOR with R, fused with dst store/xor at ZMM width.
    let mut i = 0;
    while i < BLOCK_WORDS {
      let qv = _mm512_loadu_si512(q.as_ptr().add(i).cast());
      let rv = _mm512_loadu_si512(r.as_ptr().add(i).cast());
      let f = _mm512_xor_si512(qv, rv);
      if xor_into {
        let cur = _mm512_loadu_si512(dst.as_ptr().add(i).cast());
        _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast(), _mm512_xor_si512(cur, f));
      } else {
        _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast(), f);
      }
      i += 8;
    }
  }
}

/// 8-way ZMM P-round (2 batched rows of 16 u64 each).
///
/// # Safety
///
/// Inherits AVX-512F from the caller. `_mm512_permutex_epi64` permutes
/// each 256-bit lane independently with the same imm8 — the natural
/// shape for 2-row batched diagonalisation.
#[inline(always)]
unsafe fn p_round_avx512(a: &mut __m512i, b: &mut __m512i, c: &mut __m512i, d: &mut __m512i) {
  // SAFETY: AVX-512F inherited.
  unsafe {
    gb_avx512(a, b, c, d);
    *b = _mm512_permutex_epi64(*b, 0x39);
    *c = _mm512_permutex_epi64(*c, 0x4E);
    *d = _mm512_permutex_epi64(*d, 0x93);
    gb_avx512(a, b, c, d);
    *b = _mm512_permutex_epi64(*b, 0x93);
    *c = _mm512_permutex_epi64(*c, 0x4E);
    *d = _mm512_permutex_epi64(*d, 0x39);
  }
}

/// 8-way parallel BlaMka G on ZMM rows.
///
/// # Safety
///
/// Inherits AVX-512F from the caller; register-only.
#[inline(always)]
unsafe fn gb_avx512(a: &mut __m512i, b: &mut __m512i, c: &mut __m512i, d: &mut __m512i) {
  // SAFETY: AVX-512F inherited.
  unsafe {
    let p = _mm512_mul_epu32(*a, *b);
    *a = _mm512_add_epi64(_mm512_add_epi64(*a, *b), _mm512_slli_epi64(p, 1));
    *d = _mm512_ror_epi64(_mm512_xor_si512(*d, *a), 32);

    let p = _mm512_mul_epu32(*c, *d);
    *c = _mm512_add_epi64(_mm512_add_epi64(*c, *d), _mm512_slli_epi64(p, 1));
    *b = _mm512_ror_epi64(_mm512_xor_si512(*b, *c), 24);

    let p = _mm512_mul_epu32(*a, *b);
    *a = _mm512_add_epi64(_mm512_add_epi64(*a, *b), _mm512_slli_epi64(p, 1));
    *d = _mm512_ror_epi64(_mm512_xor_si512(*d, *a), 16);

    let p = _mm512_mul_epu32(*c, *d);
    *c = _mm512_add_epi64(_mm512_add_epi64(*c, *d), _mm512_slli_epi64(p, 1));
    *b = _mm512_ror_epi64(_mm512_xor_si512(*b, *c), 63);
  }
}

/// 4-way YMM P-round using EVEX-encoded `VPRORQ` rotations.
///
/// # Safety
///
/// Inherits AVX-512VL from the caller (required for 256-bit `VPRORQ`).
#[inline(always)]
unsafe fn p_round_avx512vl(a: &mut __m256i, b: &mut __m256i, c: &mut __m256i, d: &mut __m256i) {
  // SAFETY: AVX-512VL inherited.
  unsafe {
    gb_avx512vl(a, b, c, d);
    *b = _mm256_permute4x64_epi64(*b, 0x39);
    *c = _mm256_permute4x64_epi64(*c, 0x4E);
    *d = _mm256_permute4x64_epi64(*d, 0x93);
    gb_avx512vl(a, b, c, d);
    *b = _mm256_permute4x64_epi64(*b, 0x93);
    *c = _mm256_permute4x64_epi64(*c, 0x4E);
    *d = _mm256_permute4x64_epi64(*d, 0x39);
  }
}

/// 4-way parallel BlaMka G on YMM rows with native rotations.
///
/// # Safety
///
/// Inherits AVX-512VL from the caller; register-only.
#[inline(always)]
unsafe fn gb_avx512vl(a: &mut __m256i, b: &mut __m256i, c: &mut __m256i, d: &mut __m256i) {
  // SAFETY: AVX-512VL inherited.
  unsafe {
    let p = _mm256_mul_epu32(*a, *b);
    *a = _mm256_add_epi64(_mm256_add_epi64(*a, *b), _mm256_slli_epi64(p, 1));
    *d = _mm256_ror_epi64(_mm256_xor_si256(*d, *a), 32);

    let p = _mm256_mul_epu32(*c, *d);
    *c = _mm256_add_epi64(_mm256_add_epi64(*c, *d), _mm256_slli_epi64(p, 1));
    *b = _mm256_ror_epi64(_mm256_xor_si256(*b, *c), 24);

    let p = _mm256_mul_epu32(*a, *b);
    *a = _mm256_add_epi64(_mm256_add_epi64(*a, *b), _mm256_slli_epi64(p, 1));
    *d = _mm256_ror_epi64(_mm256_xor_si256(*d, *a), 16);

    let p = _mm256_mul_epu32(*c, *d);
    *c = _mm256_add_epi64(_mm256_add_epi64(*c, *d), _mm256_slli_epi64(p, 1));
    *b = _mm256_ror_epi64(_mm256_xor_si256(*b, *c), 63);
  }
}
