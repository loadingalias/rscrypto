//! AVX-512 IFMA vectorized field arithmetic for Ed25519.
//!
//! Four field elements over GF(2^255 - 19) packed as `[__m256i; 5]` in
//! radix-2^51 (5 limbs per element, one limb per u64 lane). Multiplications
//! use the dual-accumulator IFMA pattern (`vpmadd52luq` / `vpmadd52huq`)
//! which captures the full 104-bit product without the 52-bit truncation
//! that broke the radix-26/25 approach.
//!
//! # Lane layout
//!
//! ```text
//! self.0[i] = (a_i, b_i, c_i, d_i)   // 4 × u64 lanes
//! ```
//!
//! where `a_i..d_i` is limb `i` of four independent field elements A, B, C, D.
//!
//! # Arithmetic convention
//!
//! Field arithmetic is modular math (mod 2^255 - 19). Per CLAUDE.md rules,
//! `wrapping_*` is the correct choice for intentional modular arithmetic.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{
  field::FieldElement,
  field_avx2::{Lanes, Shuffle},
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MASK51: i64 = (1i64 << 51) - 1;
const MASK52: i64 = (1i64 << 52) - 1;

/// Subtraction bias: 2p in radix-51. Limb 0 accounts for the -19 term.
const BIAS_0: i64 = 2 * ((1i64 << 51) - 19);
const BIAS_N: i64 = 2 * ((1i64 << 51) - 1);

// ---------------------------------------------------------------------------
// Type
// ---------------------------------------------------------------------------

/// Four field elements packed for AVX-512 IFMA parallel processing.
///
/// Uses radix-2^51: 5 limbs per element stored in u64 lanes. Four elements
/// are interleaved across 5 × `__m256i` registers (4 × u64 each).
#[derive(Clone, Copy)]
#[cfg(target_arch = "x86_64")]
pub(crate) struct FieldElement51x4(pub(crate) [__m256i; 5]);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Multiply a u64x4 vector by 19 using shift-and-add.
///
/// `x * 19 = (x << 4) + (x << 1) + x = 16x + 2x + x`
///
/// No AVX-512DQ dependency (avoids `_mm256_mullo_epi64`).
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn mul19(x: __m256i) -> __m256i {
  let x16 = _mm256_slli_epi64::<4>(x);
  let x2 = _mm256_slli_epi64::<1>(x);
  _mm256_add_epi64(_mm256_add_epi64(x16, x2), x)
}

/// IFMA fused multiply-accumulate (low 52 bits): `acc += (a * b)[51:0]`.
///
/// # Safety
///
/// Caller must ensure AVX-512 IFMA + VL are available.
#[inline]
#[target_feature(enable = "avx2,avx512ifma,avx512vl")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn madd52lo(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
  _mm256_madd52lo_epu64(acc, a, b)
}

/// IFMA fused multiply-accumulate (high 52 bits): `acc += (a * b)[103:52]`.
///
/// # Safety
///
/// Caller must ensure AVX-512 IFMA + VL are available.
#[inline]
#[target_feature(enable = "avx2,avx512ifma,avx512vl")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn madd52hi(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
  _mm256_madd52hi_epu64(acc, a, b)
}

/// Select `val` when `bit` is 1, zero when `bit` is 0.
///
/// Converts a 0/1 integer mask into a full u64-lane mask and AND-selects.
/// `bit` must contain only 0 or 1 in each u64 lane.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn select_by_bit(bit: __m256i, val: __m256i) -> __m256i {
  // 0 → 0, 1 → 0xFFFF_FFFF_FFFF_FFFF
  let mask = _mm256_sub_epi64(_mm256_setzero_si256(), bit);
  _mm256_and_si256(mask, val)
}

// ---------------------------------------------------------------------------
// FieldElement51x4 implementation
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
impl FieldElement51x4 {
  /// The zero element (all four lanes).
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn zero() -> Self {
    Self([_mm256_setzero_si256(); 5])
  }

  /// Pack four scalar field elements into IFMA vectorized form.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn new(a: &FieldElement, b: &FieldElement, c: &FieldElement, d: &FieldElement) -> Self {
    let al = a.limbs();
    let bl = b.limbs();
    let cl = c.limbs();
    let dl = d.limbs();

    Self([
      _mm256_set_epi64x(dl[0] as i64, cl[0] as i64, bl[0] as i64, al[0] as i64),
      _mm256_set_epi64x(dl[1] as i64, cl[1] as i64, bl[1] as i64, al[1] as i64),
      _mm256_set_epi64x(dl[2] as i64, cl[2] as i64, bl[2] as i64, al[2] as i64),
      _mm256_set_epi64x(dl[3] as i64, cl[3] as i64, bl[3] as i64, al[3] as i64),
      _mm256_set_epi64x(dl[4] as i64, cl[4] as i64, bl[4] as i64, al[4] as i64),
    ])
  }

  /// Unpack back to four scalar field elements.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn split(&self) -> [FieldElement; 4] {
    let mut al = [0u64; 5];
    let mut bl = [0u64; 5];
    let mut cl = [0u64; 5];
    let mut dl = [0u64; 5];

    for (((a_out, b_out), (c_out, d_out)), vec) in al
      .iter_mut()
      .zip(bl.iter_mut())
      .zip(cl.iter_mut().zip(dl.iter_mut()))
      .zip(self.0.iter())
    {
      let mut tmp = [0u64; 4];
      _mm256_storeu_si256(tmp.as_mut_ptr().cast(), *vec);
      *a_out = tmp[0];
      *b_out = tmp[1];
      *c_out = tmp[2];
      *d_out = tmp[3];
    }

    [
      FieldElement::from_limbs(al),
      FieldElement::from_limbs(bl),
      FieldElement::from_limbs(cl),
      FieldElement::from_limbs(dl),
    ]
  }

  /// Lazy addition (no carry propagation).
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn add(&self, rhs: &Self) -> Self {
    Self([
      _mm256_add_epi64(self.0[0], rhs.0[0]),
      _mm256_add_epi64(self.0[1], rhs.0[1]),
      _mm256_add_epi64(self.0[2], rhs.0[2]),
      _mm256_add_epi64(self.0[3], rhs.0[3]),
      _mm256_add_epi64(self.0[4], rhs.0[4]),
    ])
  }

  /// Subtraction with 2p bias to avoid underflow.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn sub(&self, rhs: &Self) -> Self {
    let bias_0 = _mm256_set1_epi64x(BIAS_0);
    let bias_n = _mm256_set1_epi64x(BIAS_N);

    Self([
      _mm256_sub_epi64(_mm256_add_epi64(self.0[0], bias_0), rhs.0[0]),
      _mm256_sub_epi64(_mm256_add_epi64(self.0[1], bias_n), rhs.0[1]),
      _mm256_sub_epi64(_mm256_add_epi64(self.0[2], bias_n), rhs.0[2]),
      _mm256_sub_epi64(_mm256_add_epi64(self.0[3], bias_n), rhs.0[3]),
      _mm256_sub_epi64(_mm256_add_epi64(self.0[4], bias_n), rhs.0[4]),
    ])
  }

  /// Negate all four lanes: `2p - self`.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn negate_lazy(&self) -> Self {
    Self::zero().sub(self)
  }

  /// Rearrange the four lane positions according to `pattern`.
  ///
  /// Uses `_mm256_permute4x64_epi64` (u64 granularity).
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn shuffle(&self, pattern: Shuffle) -> Self {
    // _mm256_permute4x64_epi64 requires a compile-time immediate.
    // IMM8 = (d_src << 6) | (c_src << 4) | (b_src << 2) | a_src
    macro_rules! do_shuffle {
      ($imm:expr) => {
        Self([
          _mm256_permute4x64_epi64::<$imm>(self.0[0]),
          _mm256_permute4x64_epi64::<$imm>(self.0[1]),
          _mm256_permute4x64_epi64::<$imm>(self.0[2]),
          _mm256_permute4x64_epi64::<$imm>(self.0[3]),
          _mm256_permute4x64_epi64::<$imm>(self.0[4]),
        ])
      };
    }

    match pattern {
      Shuffle::ABCD => do_shuffle!(0b11_10_01_00),
      Shuffle::BADC => do_shuffle!(0b10_11_00_01),
      Shuffle::BACD => do_shuffle!(0b11_10_00_01),
      Shuffle::ABDC => do_shuffle!(0b10_11_01_00),
      Shuffle::AAAA => do_shuffle!(0b00_00_00_00),
      Shuffle::BBBB => do_shuffle!(0b01_01_01_01),
      Shuffle::CACA => do_shuffle!(0b00_10_00_10),
      Shuffle::DBBD => do_shuffle!(0b11_01_01_11),
      Shuffle::ADDA => do_shuffle!(0b00_11_11_00),
      Shuffle::CBCB => do_shuffle!(0b01_10_01_10),
      Shuffle::ABAB => do_shuffle!(0b01_00_01_00),
    }
  }

  /// Select lanes from `self` or `other` according to `lanes`.
  ///
  /// Lanes specified in `lanes` come from `other`; remaining from `self`.
  /// Uses u64-aligned blend masks for `_mm256_blend_epi32`.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn blend(&self, other: &Self, lanes: Lanes) -> Self {
    // u64 lane → u32 pair: A={0,1}, B={2,3}, C={4,5}, D={6,7}
    macro_rules! do_blend {
      ($imm:expr) => {
        Self([
          _mm256_blend_epi32::<$imm>(self.0[0], other.0[0]),
          _mm256_blend_epi32::<$imm>(self.0[1], other.0[1]),
          _mm256_blend_epi32::<$imm>(self.0[2], other.0[2]),
          _mm256_blend_epi32::<$imm>(self.0[3], other.0[3]),
          _mm256_blend_epi32::<$imm>(self.0[4], other.0[4]),
        ])
      };
    }

    match lanes {
      Lanes::A => do_blend!(0b0000_0011),
      Lanes::B => do_blend!(0b0000_1100),
      Lanes::C => do_blend!(0b0011_0000),
      Lanes::D => do_blend!(0b1100_0000),
      Lanes::AB => do_blend!(0b0000_1111),
      Lanes::AC => do_blend!(0b0011_0011),
      Lanes::AD => do_blend!(0b1100_0011),
      Lanes::BC => do_blend!(0b0011_1100),
      Lanes::CD => do_blend!(0b1111_0000),
      Lanes::ABCD => do_blend!(0b1111_1111),
    }
  }

  /// Compute `(B-A, A+B, D-C, C+D)` in one pass.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn diff_sum(&self) -> Self {
    let swapped = self.shuffle(Shuffle::BADC); // (B, A, D, C)
    let negated = self.negate_lazy(); // (-A, -B, -C, -D)
    let neg_ac = self.blend(&negated, Lanes::AC); // (-A, B, -C, D)
    swapped.add(&neg_ac) // (B-A, A+B, D-C, C+D)
  }

  /// Carry-propagate to bring all limbs within 51 bits.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX-512 IFMA + VL are available.
  #[inline]
  #[target_feature(enable = "avx2,avx512ifma,avx512vl")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn reduce(mut self) -> Self {
    let mask = _mm256_set1_epi64x(MASK51);
    let r19 = _mm256_set1_epi64x(19);

    // Forward carry chain: limb 0 → 1 → 2 → 3 → 4
    let c0 = _mm256_srli_epi64::<51>(self.0[0]);
    self.0[0] = _mm256_and_si256(self.0[0], mask);
    self.0[1] = _mm256_add_epi64(self.0[1], c0);

    let c1 = _mm256_srli_epi64::<51>(self.0[1]);
    self.0[1] = _mm256_and_si256(self.0[1], mask);
    self.0[2] = _mm256_add_epi64(self.0[2], c1);

    let c2 = _mm256_srli_epi64::<51>(self.0[2]);
    self.0[2] = _mm256_and_si256(self.0[2], mask);
    self.0[3] = _mm256_add_epi64(self.0[3], c2);

    let c3 = _mm256_srli_epi64::<51>(self.0[3]);
    self.0[3] = _mm256_and_si256(self.0[3], mask);
    self.0[4] = _mm256_add_epi64(self.0[4], c3);

    // Wraparound: carry from limb 4 × 19 → limb 0
    let c4 = _mm256_srli_epi64::<51>(self.0[4]);
    self.0[4] = _mm256_and_si256(self.0[4], mask);
    self.0[0] = madd52lo(self.0[0], c4, r19);

    // Final carry from limb 0 → 1 (at most 1 bit of overflow)
    let c0f = _mm256_srli_epi64::<51>(self.0[0]);
    self.0[0] = _mm256_and_si256(self.0[0], mask);
    self.0[1] = _mm256_add_epi64(self.0[1], c0f);

    self
  }

  /// Multiply two vectorized field elements using AVX-512 IFMA.
  ///
  /// Full 5×5 schoolbook with dual accumulators (`madd52lo` + `madd52hi`),
  /// post-reduction of upper limbs, and carry propagation.
  ///
  /// # Precondition
  ///
  /// Both operands must have limbs ≤ 52 bits (fitting IFMA's input window).
  /// This is satisfied by `reduce()` output (≤51 bits), `negate_lazy()` on
  /// reduced values (≤52 bits), or `new()` from scalar field elements (≤51 bits).
  /// It is NOT satisfied after `diff_sum()` on reduced values (up to 53 bits) —
  /// callers must `reduce()` such intermediates before calling `mul`.
  ///
  /// Production code uses [`mul_unreduced`](Self::mul_unreduced) which accepts
  /// up to 53-bit inputs, eliminating the need for `reduce()` before multiply.
  /// Retained for differential testing.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX-512 IFMA + VL are available.
  #[cfg(test)]
  #[target_feature(enable = "avx2,avx512ifma,avx512vl")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn mul(&self, rhs: &Self) -> Self {
    let zero = _mm256_setzero_si256();
    let f = &self.0;
    let g = &rhs.0;

    // -----------------------------------------------------------------------
    // Phase 1: 5×5 schoolbook with dual accumulators (50 IFMA ops)
    //
    // Interleaved lo/hi scheduling: each product term issues madd52lo then
    // madd52hi before the next term. This exposes the independent lo and hi
    // dependency chains to Intel's dual-issue IFMA ports (0 and 1), letting
    // the OOO engine overlap them instead of serialising all-lo-then-all-hi.
    // -----------------------------------------------------------------------

    // z0 = f0*g0
    let lo0 = madd52lo(zero, f[0], g[0]);
    let hi0 = madd52hi(zero, f[0], g[0]);

    // z1 = f0*g1 + f1*g0
    let mut lo1 = madd52lo(zero, f[0], g[1]);
    let mut hi1 = madd52hi(zero, f[0], g[1]);
    lo1 = madd52lo(lo1, f[1], g[0]);
    hi1 = madd52hi(hi1, f[1], g[0]);

    // z2 = f0*g2 + f1*g1 + f2*g0
    let mut lo2 = madd52lo(zero, f[0], g[2]);
    let mut hi2 = madd52hi(zero, f[0], g[2]);
    lo2 = madd52lo(lo2, f[1], g[1]);
    hi2 = madd52hi(hi2, f[1], g[1]);
    lo2 = madd52lo(lo2, f[2], g[0]);
    hi2 = madd52hi(hi2, f[2], g[0]);

    // z3 = f0*g3 + f1*g2 + f2*g1 + f3*g0
    let mut lo3 = madd52lo(zero, f[0], g[3]);
    let mut hi3 = madd52hi(zero, f[0], g[3]);
    lo3 = madd52lo(lo3, f[1], g[2]);
    hi3 = madd52hi(hi3, f[1], g[2]);
    lo3 = madd52lo(lo3, f[2], g[1]);
    hi3 = madd52hi(hi3, f[2], g[1]);
    lo3 = madd52lo(lo3, f[3], g[0]);
    hi3 = madd52hi(hi3, f[3], g[0]);

    // z4 = f0*g4 + f1*g3 + f2*g2 + f3*g1 + f4*g0
    let mut lo4 = madd52lo(zero, f[0], g[4]);
    let mut hi4 = madd52hi(zero, f[0], g[4]);
    lo4 = madd52lo(lo4, f[1], g[3]);
    hi4 = madd52hi(hi4, f[1], g[3]);
    lo4 = madd52lo(lo4, f[2], g[2]);
    hi4 = madd52hi(hi4, f[2], g[2]);
    lo4 = madd52lo(lo4, f[3], g[1]);
    hi4 = madd52hi(hi4, f[3], g[1]);
    lo4 = madd52lo(lo4, f[4], g[0]);
    hi4 = madd52hi(hi4, f[4], g[0]);

    // z5 = f1*g4 + f2*g3 + f3*g2 + f4*g1
    let mut lo5 = madd52lo(zero, f[1], g[4]);
    let mut hi5 = madd52hi(zero, f[1], g[4]);
    lo5 = madd52lo(lo5, f[2], g[3]);
    hi5 = madd52hi(hi5, f[2], g[3]);
    lo5 = madd52lo(lo5, f[3], g[2]);
    hi5 = madd52hi(hi5, f[3], g[2]);
    lo5 = madd52lo(lo5, f[4], g[1]);
    hi5 = madd52hi(hi5, f[4], g[1]);

    // z6 = f2*g4 + f3*g3 + f4*g2
    let mut lo6 = madd52lo(zero, f[2], g[4]);
    let mut hi6 = madd52hi(zero, f[2], g[4]);
    lo6 = madd52lo(lo6, f[3], g[3]);
    hi6 = madd52hi(hi6, f[3], g[3]);
    lo6 = madd52lo(lo6, f[4], g[2]);
    hi6 = madd52hi(hi6, f[4], g[2]);

    // z7 = f3*g4 + f4*g3
    let mut lo7 = madd52lo(zero, f[3], g[4]);
    let mut hi7 = madd52hi(zero, f[3], g[4]);
    lo7 = madd52lo(lo7, f[4], g[3]);
    hi7 = madd52hi(hi7, f[4], g[3]);

    // z8 = f4*g4
    let lo8 = madd52lo(zero, f[4], g[4]);
    let hi8 = madd52hi(zero, f[4], g[4]);

    // -----------------------------------------------------------------------
    // Phase 2: Combine lo + hi into 10 actual limbs
    //
    // z_k = lo_k + 2 * hi_{k-1}
    // The factor 2 comes from: bits [103:52] have value v * 2^52 = v * 2 * 2^51,
    // so they contribute v * 2 to the next limb.
    // -----------------------------------------------------------------------

    let z0 = lo0;
    let z1 = _mm256_add_epi64(_mm256_add_epi64(lo1, hi0), hi0);
    let z2 = _mm256_add_epi64(_mm256_add_epi64(lo2, hi1), hi1);
    let z3 = _mm256_add_epi64(_mm256_add_epi64(lo3, hi2), hi2);
    let z4 = _mm256_add_epi64(_mm256_add_epi64(lo4, hi3), hi3);
    let z5 = _mm256_add_epi64(_mm256_add_epi64(lo5, hi4), hi4);
    let z6 = _mm256_add_epi64(_mm256_add_epi64(lo6, hi5), hi5);
    let z7 = _mm256_add_epi64(_mm256_add_epi64(lo7, hi6), hi6);
    let z8 = _mm256_add_epi64(_mm256_add_epi64(lo8, hi7), hi7);
    let z9 = _mm256_add_epi64(hi8, hi8);

    // -----------------------------------------------------------------------
    // Phase 3: Reduce upper limbs by ×19 and fold into lower 5
    //
    // 2^(51*5) = 2^255 ≡ 19 (mod p)
    // -----------------------------------------------------------------------

    let z5_19 = mul19(z5);
    let z6_19 = mul19(z6);
    let z7_19 = mul19(z7);
    let z8_19 = mul19(z8);
    let z9_19 = mul19(z9);

    Self([
      _mm256_add_epi64(z0, z5_19),
      _mm256_add_epi64(z1, z6_19),
      _mm256_add_epi64(z2, z7_19),
      _mm256_add_epi64(z3, z8_19),
      _mm256_add_epi64(z4, z9_19),
    ])
    .reduce()
  }

  /// Multiply two vectorized field elements that may have up to 53-bit limbs.
  ///
  /// This is the key optimization for IFMA point operations: `diff_sum()`
  /// produces limbs up to 53 bits (51-bit reduced + 2-bit bias from add/sub),
  /// but IFMA's `vpmadd52luq` truncates inputs to 52 bits. The standard
  /// `mul()` therefore requires a `reduce()` call (17 insns, ~18 cycle
  /// latency) before each multiply after `diff_sum()`.
  ///
  /// `mul_unreduced()` handles the 53rd bit by decomposing each operand
  /// into a 52-bit low part (suitable for IFMA) and a 0/1 high bit, then
  /// computing corrections for the overflow:
  ///
  ///   f * g = f_lo * g_lo + f_hi * g_lo + f_lo * g_hi + f_hi * g_hi
  ///
  /// The `f_lo * g_lo` term uses the standard 50-IFMA schoolbook.
  /// The cross terms (`f_hi * g_lo`, `f_lo * g_hi`) use `select_by_bit`
  /// to conditionally add values based on the 0/1 overflow bits.
  /// The `f_hi * g_hi` term produces at most 4 per accumulator and is
  /// folded with a simple shift.
  ///
  /// # Precondition
  ///
  /// Both operands must have limbs ≤ 53 bits. This is satisfied by
  /// `diff_sum()` on reduced values, or `add()` of two 52-bit values.
  ///
  /// # Overflow safety
  ///
  /// With 52-bit inputs to IFMA, the lo/hi accumulators after 5 terms
  /// are at most ~5 × 2^52 ≈ 2^54.3. The corrections add at most
  /// 5 × (2^52 - 1) per hi accumulator. Combined z_k values stay
  /// within ~2^57, and the ×19 fold produces values up to ~2^61.3,
  /// well within u64.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX-512 IFMA + VL are available.
  #[target_feature(enable = "avx2,avx512ifma,avx512vl")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn mul_unreduced(&self, rhs: &Self) -> Self {
    let zero = _mm256_setzero_si256();
    let mask52 = _mm256_set1_epi64x(MASK52);

    // Pre-mask inputs to 52 bits for IFMA, extract overflow bits (0 or 1).
    let f_lo: [__m256i; 5] = [
      _mm256_and_si256(self.0[0], mask52),
      _mm256_and_si256(self.0[1], mask52),
      _mm256_and_si256(self.0[2], mask52),
      _mm256_and_si256(self.0[3], mask52),
      _mm256_and_si256(self.0[4], mask52),
    ];
    let f_hi: [__m256i; 5] = [
      _mm256_srli_epi64::<52>(self.0[0]),
      _mm256_srli_epi64::<52>(self.0[1]),
      _mm256_srli_epi64::<52>(self.0[2]),
      _mm256_srli_epi64::<52>(self.0[3]),
      _mm256_srli_epi64::<52>(self.0[4]),
    ];
    let g_lo: [__m256i; 5] = [
      _mm256_and_si256(rhs.0[0], mask52),
      _mm256_and_si256(rhs.0[1], mask52),
      _mm256_and_si256(rhs.0[2], mask52),
      _mm256_and_si256(rhs.0[3], mask52),
      _mm256_and_si256(rhs.0[4], mask52),
    ];
    let g_hi: [__m256i; 5] = [
      _mm256_srli_epi64::<52>(rhs.0[0]),
      _mm256_srli_epi64::<52>(rhs.0[1]),
      _mm256_srli_epi64::<52>(rhs.0[2]),
      _mm256_srli_epi64::<52>(rhs.0[3]),
      _mm256_srli_epi64::<52>(rhs.0[4]),
    ];

    // -----------------------------------------------------------------------
    // Phase 1: 5×5 schoolbook on 52-bit inputs (50 IFMA ops)
    // Same interleaved lo/hi scheduling as mul().
    // -----------------------------------------------------------------------

    // z0 = f0*g0
    let lo0 = madd52lo(zero, f_lo[0], g_lo[0]);
    let hi0 = madd52hi(zero, f_lo[0], g_lo[0]);

    // z1 = f0*g1 + f1*g0
    let mut lo1 = madd52lo(zero, f_lo[0], g_lo[1]);
    let mut hi1 = madd52hi(zero, f_lo[0], g_lo[1]);
    lo1 = madd52lo(lo1, f_lo[1], g_lo[0]);
    hi1 = madd52hi(hi1, f_lo[1], g_lo[0]);

    // z2 = f0*g2 + f1*g1 + f2*g0
    let mut lo2 = madd52lo(zero, f_lo[0], g_lo[2]);
    let mut hi2 = madd52hi(zero, f_lo[0], g_lo[2]);
    lo2 = madd52lo(lo2, f_lo[1], g_lo[1]);
    hi2 = madd52hi(hi2, f_lo[1], g_lo[1]);
    lo2 = madd52lo(lo2, f_lo[2], g_lo[0]);
    hi2 = madd52hi(hi2, f_lo[2], g_lo[0]);

    // z3 = f0*g3 + f1*g2 + f2*g1 + f3*g0
    let mut lo3 = madd52lo(zero, f_lo[0], g_lo[3]);
    let mut hi3 = madd52hi(zero, f_lo[0], g_lo[3]);
    lo3 = madd52lo(lo3, f_lo[1], g_lo[2]);
    hi3 = madd52hi(hi3, f_lo[1], g_lo[2]);
    lo3 = madd52lo(lo3, f_lo[2], g_lo[1]);
    hi3 = madd52hi(hi3, f_lo[2], g_lo[1]);
    lo3 = madd52lo(lo3, f_lo[3], g_lo[0]);
    hi3 = madd52hi(hi3, f_lo[3], g_lo[0]);

    // z4 = f0*g4 + f1*g3 + f2*g2 + f3*g1 + f4*g0
    let mut lo4 = madd52lo(zero, f_lo[0], g_lo[4]);
    let mut hi4 = madd52hi(zero, f_lo[0], g_lo[4]);
    lo4 = madd52lo(lo4, f_lo[1], g_lo[3]);
    hi4 = madd52hi(hi4, f_lo[1], g_lo[3]);
    lo4 = madd52lo(lo4, f_lo[2], g_lo[2]);
    hi4 = madd52hi(hi4, f_lo[2], g_lo[2]);
    lo4 = madd52lo(lo4, f_lo[3], g_lo[1]);
    hi4 = madd52hi(hi4, f_lo[3], g_lo[1]);
    lo4 = madd52lo(lo4, f_lo[4], g_lo[0]);
    hi4 = madd52hi(hi4, f_lo[4], g_lo[0]);

    // z5 = f1*g4 + f2*g3 + f3*g2 + f4*g1
    let mut lo5 = madd52lo(zero, f_lo[1], g_lo[4]);
    let mut hi5 = madd52hi(zero, f_lo[1], g_lo[4]);
    lo5 = madd52lo(lo5, f_lo[2], g_lo[3]);
    hi5 = madd52hi(hi5, f_lo[2], g_lo[3]);
    lo5 = madd52lo(lo5, f_lo[3], g_lo[2]);
    hi5 = madd52hi(hi5, f_lo[3], g_lo[2]);
    lo5 = madd52lo(lo5, f_lo[4], g_lo[1]);
    hi5 = madd52hi(hi5, f_lo[4], g_lo[1]);

    // z6 = f2*g4 + f3*g3 + f4*g2
    let mut lo6 = madd52lo(zero, f_lo[2], g_lo[4]);
    let mut hi6 = madd52hi(zero, f_lo[2], g_lo[4]);
    lo6 = madd52lo(lo6, f_lo[3], g_lo[3]);
    hi6 = madd52hi(hi6, f_lo[3], g_lo[3]);
    lo6 = madd52lo(lo6, f_lo[4], g_lo[2]);
    hi6 = madd52hi(hi6, f_lo[4], g_lo[2]);

    // z7 = f3*g4 + f4*g3
    let mut lo7 = madd52lo(zero, f_lo[3], g_lo[4]);
    let mut hi7 = madd52hi(zero, f_lo[3], g_lo[4]);
    lo7 = madd52lo(lo7, f_lo[4], g_lo[3]);
    hi7 = madd52hi(hi7, f_lo[4], g_lo[3]);

    // z8 = f4*g4
    let lo8 = madd52lo(zero, f_lo[4], g_lo[4]);
    let hi8 = madd52hi(zero, f_lo[4], g_lo[4]);

    // -----------------------------------------------------------------------
    // Phase 1b: Corrections for overflow bits (f_hi * g_lo + f_lo * g_hi)
    //
    // For each product index k, add contributions from overflow bits:
    //   hi_k += sum_{i+j=k} select(f_hi[i], g_lo[j]) + select(g_hi[j], f_lo[i])
    //
    // Since f_hi[i] is 0 or 1, select(f_hi[i], g_lo[j]) is either 0 or g_lo[j].
    // These corrections go into hi accumulators because the overflow bit
    // has weight 2^52 = 2 × 2^51, matching the hi accumulator's weight.
    // -----------------------------------------------------------------------

    // k=0: i=0,j=0
    let hi0 = _mm256_add_epi64(hi0, select_by_bit(f_hi[0], g_lo[0]));
    let hi0 = _mm256_add_epi64(hi0, select_by_bit(g_hi[0], f_lo[0]));

    // k=1: (i,j) = (0,1), (1,0)
    let mut hi1 = _mm256_add_epi64(hi1, select_by_bit(f_hi[0], g_lo[1]));
    hi1 = _mm256_add_epi64(hi1, select_by_bit(g_hi[1], f_lo[0]));
    hi1 = _mm256_add_epi64(hi1, select_by_bit(f_hi[1], g_lo[0]));
    hi1 = _mm256_add_epi64(hi1, select_by_bit(g_hi[0], f_lo[1]));

    // k=2: (0,2), (1,1), (2,0)
    let mut hi2 = _mm256_add_epi64(hi2, select_by_bit(f_hi[0], g_lo[2]));
    hi2 = _mm256_add_epi64(hi2, select_by_bit(g_hi[2], f_lo[0]));
    hi2 = _mm256_add_epi64(hi2, select_by_bit(f_hi[1], g_lo[1]));
    hi2 = _mm256_add_epi64(hi2, select_by_bit(g_hi[1], f_lo[1]));
    hi2 = _mm256_add_epi64(hi2, select_by_bit(f_hi[2], g_lo[0]));
    hi2 = _mm256_add_epi64(hi2, select_by_bit(g_hi[0], f_lo[2]));

    // k=3: (0,3), (1,2), (2,1), (3,0)
    let mut hi3 = _mm256_add_epi64(hi3, select_by_bit(f_hi[0], g_lo[3]));
    hi3 = _mm256_add_epi64(hi3, select_by_bit(g_hi[3], f_lo[0]));
    hi3 = _mm256_add_epi64(hi3, select_by_bit(f_hi[1], g_lo[2]));
    hi3 = _mm256_add_epi64(hi3, select_by_bit(g_hi[2], f_lo[1]));
    hi3 = _mm256_add_epi64(hi3, select_by_bit(f_hi[2], g_lo[1]));
    hi3 = _mm256_add_epi64(hi3, select_by_bit(g_hi[1], f_lo[2]));
    hi3 = _mm256_add_epi64(hi3, select_by_bit(f_hi[3], g_lo[0]));
    hi3 = _mm256_add_epi64(hi3, select_by_bit(g_hi[0], f_lo[3]));

    // k=4: (0,4), (1,3), (2,2), (3,1), (4,0)
    let mut hi4 = _mm256_add_epi64(hi4, select_by_bit(f_hi[0], g_lo[4]));
    hi4 = _mm256_add_epi64(hi4, select_by_bit(g_hi[4], f_lo[0]));
    hi4 = _mm256_add_epi64(hi4, select_by_bit(f_hi[1], g_lo[3]));
    hi4 = _mm256_add_epi64(hi4, select_by_bit(g_hi[3], f_lo[1]));
    hi4 = _mm256_add_epi64(hi4, select_by_bit(f_hi[2], g_lo[2]));
    hi4 = _mm256_add_epi64(hi4, select_by_bit(g_hi[2], f_lo[2]));
    hi4 = _mm256_add_epi64(hi4, select_by_bit(f_hi[3], g_lo[1]));
    hi4 = _mm256_add_epi64(hi4, select_by_bit(g_hi[1], f_lo[3]));
    hi4 = _mm256_add_epi64(hi4, select_by_bit(f_hi[4], g_lo[0]));
    hi4 = _mm256_add_epi64(hi4, select_by_bit(g_hi[0], f_lo[4]));

    // k=5: (1,4), (2,3), (3,2), (4,1)
    let mut hi5 = _mm256_add_epi64(hi5, select_by_bit(f_hi[1], g_lo[4]));
    hi5 = _mm256_add_epi64(hi5, select_by_bit(g_hi[4], f_lo[1]));
    hi5 = _mm256_add_epi64(hi5, select_by_bit(f_hi[2], g_lo[3]));
    hi5 = _mm256_add_epi64(hi5, select_by_bit(g_hi[3], f_lo[2]));
    hi5 = _mm256_add_epi64(hi5, select_by_bit(f_hi[3], g_lo[2]));
    hi5 = _mm256_add_epi64(hi5, select_by_bit(g_hi[2], f_lo[3]));
    hi5 = _mm256_add_epi64(hi5, select_by_bit(f_hi[4], g_lo[1]));
    hi5 = _mm256_add_epi64(hi5, select_by_bit(g_hi[1], f_lo[4]));

    // k=6: (2,4), (3,3), (4,2)
    let mut hi6 = _mm256_add_epi64(hi6, select_by_bit(f_hi[2], g_lo[4]));
    hi6 = _mm256_add_epi64(hi6, select_by_bit(g_hi[4], f_lo[2]));
    hi6 = _mm256_add_epi64(hi6, select_by_bit(f_hi[3], g_lo[3]));
    hi6 = _mm256_add_epi64(hi6, select_by_bit(g_hi[3], f_lo[3]));
    hi6 = _mm256_add_epi64(hi6, select_by_bit(f_hi[4], g_lo[2]));
    hi6 = _mm256_add_epi64(hi6, select_by_bit(g_hi[2], f_lo[4]));

    // k=7: (3,4), (4,3)
    let mut hi7 = _mm256_add_epi64(hi7, select_by_bit(f_hi[3], g_lo[4]));
    hi7 = _mm256_add_epi64(hi7, select_by_bit(g_hi[4], f_lo[3]));
    hi7 = _mm256_add_epi64(hi7, select_by_bit(f_hi[4], g_lo[3]));
    hi7 = _mm256_add_epi64(hi7, select_by_bit(g_hi[3], f_lo[4]));

    // k=8: (4,4)
    let hi8 = _mm256_add_epi64(hi8, select_by_bit(f_hi[4], g_lo[4]));
    let hi8 = _mm256_add_epi64(hi8, select_by_bit(g_hi[4], f_lo[4]));

    // -----------------------------------------------------------------------
    // Phase 2: Combine lo + hi into 10 actual limbs
    //
    // z_k = lo_k + 2 * hi_{k-1}
    // -----------------------------------------------------------------------

    let z0 = lo0;
    let z1 = _mm256_add_epi64(_mm256_add_epi64(lo1, hi0), hi0);
    let z2 = _mm256_add_epi64(_mm256_add_epi64(lo2, hi1), hi1);
    let z3 = _mm256_add_epi64(_mm256_add_epi64(lo3, hi2), hi2);
    let z4 = _mm256_add_epi64(_mm256_add_epi64(lo4, hi3), hi3);
    let z5 = _mm256_add_epi64(_mm256_add_epi64(lo5, hi4), hi4);
    let z6 = _mm256_add_epi64(_mm256_add_epi64(lo6, hi5), hi5);
    let z7 = _mm256_add_epi64(_mm256_add_epi64(lo7, hi6), hi6);
    let z8 = _mm256_add_epi64(_mm256_add_epi64(lo8, hi7), hi7);
    let z9 = _mm256_add_epi64(hi8, hi8);

    // -----------------------------------------------------------------------
    // Phase 2b: f_hi * g_hi corrections
    //
    // When both f_hi[i] and g_hi[j] are 1, the product is 2^52 * 2^52 =
    // 2^104, which has weight 2^(104 - 51*(i+j)) in the output. This maps
    // to 4 × 2^(51*(k+2)) where k = i+j, i.e. add 4 to z_{k+2}.
    // Since f_hi and g_hi are 0/1, (f_hi[i] AND g_hi[j]) is 0/1, so the
    // correction is 4 * (f_hi[i] AND g_hi[j]) added to z_{k+2}.
    //
    // k=0: → z2, k=1: → z3, ..., k=6: → z8, k=7: → z9 (k=8 overflows
    // but z_{10} doesn't exist; it would fold via ×19 but the magnitude
    // is negligible: at most 4).
    // -----------------------------------------------------------------------

    let four = _mm256_set1_epi64x(4);

    // k=0 → z2: (0,0) — 1 term
    let hh_0 = _mm256_and_si256(f_hi[0], g_hi[0]);
    let z2 = _mm256_add_epi64(z2, _mm256_and_si256(_mm256_sub_epi64(zero, hh_0), four));

    // k=1 → z3: (0,1), (1,0) — 2 terms
    let hh_1 = _mm256_add_epi64(_mm256_and_si256(f_hi[0], g_hi[1]), _mm256_and_si256(f_hi[1], g_hi[0]));
    let z3 = _mm256_add_epi64(z3, _mm256_slli_epi64::<2>(hh_1));

    // k=2 → z4: (0,2), (1,1), (2,0) — 3 terms
    let hh_2 = _mm256_add_epi64(
      _mm256_add_epi64(_mm256_and_si256(f_hi[0], g_hi[2]), _mm256_and_si256(f_hi[1], g_hi[1])),
      _mm256_and_si256(f_hi[2], g_hi[0]),
    );
    let z4 = _mm256_add_epi64(z4, _mm256_slli_epi64::<2>(hh_2));

    // k=3 → z5: (0,3), (1,2), (2,1), (3,0) — 4 terms
    let hh_3 = _mm256_add_epi64(
      _mm256_add_epi64(_mm256_and_si256(f_hi[0], g_hi[3]), _mm256_and_si256(f_hi[1], g_hi[2])),
      _mm256_add_epi64(_mm256_and_si256(f_hi[2], g_hi[1]), _mm256_and_si256(f_hi[3], g_hi[0])),
    );
    let z5 = _mm256_add_epi64(z5, _mm256_slli_epi64::<2>(hh_3));

    // k=4 → z6: (0,4), (1,3), (2,2), (3,1), (4,0) — 5 terms
    let hh_4 = _mm256_add_epi64(
      _mm256_add_epi64(_mm256_and_si256(f_hi[0], g_hi[4]), _mm256_and_si256(f_hi[1], g_hi[3])),
      _mm256_add_epi64(
        _mm256_add_epi64(_mm256_and_si256(f_hi[2], g_hi[2]), _mm256_and_si256(f_hi[3], g_hi[1])),
        _mm256_and_si256(f_hi[4], g_hi[0]),
      ),
    );
    let z6 = _mm256_add_epi64(z6, _mm256_slli_epi64::<2>(hh_4));

    // k=5 → z7: (1,4), (2,3), (3,2), (4,1) — 4 terms
    let hh_5 = _mm256_add_epi64(
      _mm256_add_epi64(_mm256_and_si256(f_hi[1], g_hi[4]), _mm256_and_si256(f_hi[2], g_hi[3])),
      _mm256_add_epi64(_mm256_and_si256(f_hi[3], g_hi[2]), _mm256_and_si256(f_hi[4], g_hi[1])),
    );
    let z7 = _mm256_add_epi64(z7, _mm256_slli_epi64::<2>(hh_5));

    // k=6 → z8: (2,4), (3,3), (4,2) — 3 terms
    let hh_6 = _mm256_add_epi64(
      _mm256_add_epi64(_mm256_and_si256(f_hi[2], g_hi[4]), _mm256_and_si256(f_hi[3], g_hi[3])),
      _mm256_and_si256(f_hi[4], g_hi[2]),
    );
    let z8 = _mm256_add_epi64(z8, _mm256_slli_epi64::<2>(hh_6));

    // k=7 → z9: (3,4), (4,3) — 2 terms
    let hh_7 = _mm256_add_epi64(_mm256_and_si256(f_hi[3], g_hi[4]), _mm256_and_si256(f_hi[4], g_hi[3]));
    let z9 = _mm256_add_epi64(z9, _mm256_slli_epi64::<2>(hh_7));

    // k=8 → would be z10 (doesn't exist). f_hi[4]*g_hi[4] produces at
    // most 4 per lane. Fold via ×19 into z5. Max contribution: 4*19 = 76.
    let hh_8 = _mm256_and_si256(f_hi[4], g_hi[4]);
    let hh_8_x4 = _mm256_slli_epi64::<2>(hh_8);
    let z5 = _mm256_add_epi64(z5, mul19(hh_8_x4));

    // -----------------------------------------------------------------------
    // Phase 3: Reduce upper limbs by ×19 and fold into lower 5
    // -----------------------------------------------------------------------

    let z5_19 = mul19(z5);
    let z6_19 = mul19(z6);
    let z7_19 = mul19(z7);
    let z8_19 = mul19(z8);
    let z9_19 = mul19(z9);

    Self([
      _mm256_add_epi64(z0, z5_19),
      _mm256_add_epi64(z1, z6_19),
      _mm256_add_epi64(z2, z7_19),
      _mm256_add_epi64(z3, z8_19),
      _mm256_add_epi64(z4, z9_19),
    ])
    .reduce()
  }

  /// Multiply by a "small" constant accepting up to 53-bit self limbs.
  ///
  /// Same as `mul_small()` but handles the extra bit from unreduced inputs
  /// (e.g., after `diff_sum()` + `blend()`). The `small` operand has only
  /// limb 0 non-zero and is at most 18 bits, so only one-sided corrections
  /// are needed per limb.
  ///
  /// # Precondition
  ///
  /// - `small.0[1..5]` must all be zero (only limb 0 non-zero).
  /// - `self` must have limbs ≤ 53 bits.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX-512 IFMA + VL are available.
  #[target_feature(enable = "avx2,avx512ifma,avx512vl")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn mul_small_unreduced(&self, small: &Self) -> Self {
    let zero = _mm256_setzero_si256();
    let mask52 = _mm256_set1_epi64x(MASK52);
    let c = small.0[0]; // The only non-zero limb (≤18 bits, fits in 52 bits)

    // Pre-mask self to 52 bits; extract overflow bits.
    let f_lo: [__m256i; 5] = [
      _mm256_and_si256(self.0[0], mask52),
      _mm256_and_si256(self.0[1], mask52),
      _mm256_and_si256(self.0[2], mask52),
      _mm256_and_si256(self.0[3], mask52),
      _mm256_and_si256(self.0[4], mask52),
    ];
    let f_hi: [__m256i; 5] = [
      _mm256_srli_epi64::<52>(self.0[0]),
      _mm256_srli_epi64::<52>(self.0[1]),
      _mm256_srli_epi64::<52>(self.0[2]),
      _mm256_srli_epi64::<52>(self.0[3]),
      _mm256_srli_epi64::<52>(self.0[4]),
    ];

    // 5 multiplies: f_lo[k] * c — interleaved lo/hi.
    let lo0 = madd52lo(zero, f_lo[0], c);
    let hi0 = madd52hi(zero, f_lo[0], c);
    let lo1 = madd52lo(zero, f_lo[1], c);
    let hi1 = madd52hi(zero, f_lo[1], c);
    let lo2 = madd52lo(zero, f_lo[2], c);
    let hi2 = madd52hi(zero, f_lo[2], c);
    let lo3 = madd52lo(zero, f_lo[3], c);
    let hi3 = madd52hi(zero, f_lo[3], c);
    let lo4 = madd52lo(zero, f_lo[4], c);
    let hi4 = madd52hi(zero, f_lo[4], c);

    // Corrections: f_hi[k] * c. Since f_hi is 0/1 and c ≤ 18 bits,
    // select_by_bit(f_hi[k], c) is either 0 or c. These go into hi
    // because the overflow bit has weight 2^52 = 2 × 2^51.
    let hi0 = _mm256_add_epi64(hi0, select_by_bit(f_hi[0], c));
    let hi1 = _mm256_add_epi64(hi1, select_by_bit(f_hi[1], c));
    let hi2 = _mm256_add_epi64(hi2, select_by_bit(f_hi[2], c));
    let hi3 = _mm256_add_epi64(hi3, select_by_bit(f_hi[3], c));
    let hi4 = _mm256_add_epi64(hi4, select_by_bit(f_hi[4], c));

    // Recombine: z_k = lo_k + 2 * hi_{k-1}
    // Wrap-around: 2 * hi_4 * 19 folds into limb 0.
    let hi4_x2 = _mm256_add_epi64(hi4, hi4);

    Self([
      _mm256_add_epi64(lo0, mul19(hi4_x2)),
      _mm256_add_epi64(_mm256_add_epi64(lo1, hi0), hi0),
      _mm256_add_epi64(_mm256_add_epi64(lo2, hi1), hi1),
      _mm256_add_epi64(_mm256_add_epi64(lo3, hi2), hi2),
      _mm256_add_epi64(_mm256_add_epi64(lo4, hi3), hi3),
    ])
    .reduce()
  }

  /// Square (51-bit inputs, pre-doubled cross terms). Retained for testing.
  ///
  /// Production code uses `square_and_negate_d_wide()` which accepts 52-bit inputs.
  #[cfg(test)]
  /// Inputs with 52-bit limbs would overflow to 53 bits after pre-doubling.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX-512 IFMA + VL are available.
  #[target_feature(enable = "avx2,avx512ifma,avx512vl")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn square(&self) -> Self {
    let zero = _mm256_setzero_si256();
    let f = &self.0;

    // Pre-double for cross terms (f_i is at most 51 bits → f_i_2 ≤ 52 bits,
    // which fits in IFMA's 52-bit input window).
    let f0_2 = _mm256_add_epi64(f[0], f[0]);
    let f1_2 = _mm256_add_epi64(f[1], f[1]);
    let f2_2 = _mm256_add_epi64(f[2], f[2]);
    let f3_2 = _mm256_add_epi64(f[3], f[3]);

    // Interleaved lo/hi scheduling (same rationale as mul).

    // z0 = f0*f0
    let lo0 = madd52lo(zero, f[0], f[0]);
    let hi0 = madd52hi(zero, f[0], f[0]);

    // z1 = 2*f0*f1
    let lo1 = madd52lo(zero, f0_2, f[1]);
    let hi1 = madd52hi(zero, f0_2, f[1]);

    // z2 = 2*f0*f2 + f1*f1
    let mut lo2 = madd52lo(zero, f0_2, f[2]);
    let mut hi2 = madd52hi(zero, f0_2, f[2]);
    lo2 = madd52lo(lo2, f[1], f[1]);
    hi2 = madd52hi(hi2, f[1], f[1]);

    // z3 = 2*f0*f3 + 2*f1*f2
    let mut lo3 = madd52lo(zero, f0_2, f[3]);
    let mut hi3 = madd52hi(zero, f0_2, f[3]);
    lo3 = madd52lo(lo3, f1_2, f[2]);
    hi3 = madd52hi(hi3, f1_2, f[2]);

    // z4 = 2*f0*f4 + 2*f1*f3 + f2*f2
    let mut lo4 = madd52lo(zero, f0_2, f[4]);
    let mut hi4 = madd52hi(zero, f0_2, f[4]);
    lo4 = madd52lo(lo4, f1_2, f[3]);
    hi4 = madd52hi(hi4, f1_2, f[3]);
    lo4 = madd52lo(lo4, f[2], f[2]);
    hi4 = madd52hi(hi4, f[2], f[2]);

    // z5 = 2*f1*f4 + 2*f2*f3
    let mut lo5 = madd52lo(zero, f1_2, f[4]);
    let mut hi5 = madd52hi(zero, f1_2, f[4]);
    lo5 = madd52lo(lo5, f2_2, f[3]);
    hi5 = madd52hi(hi5, f2_2, f[3]);

    // z6 = 2*f2*f4 + f3*f3
    let mut lo6 = madd52lo(zero, f2_2, f[4]);
    let mut hi6 = madd52hi(zero, f2_2, f[4]);
    lo6 = madd52lo(lo6, f[3], f[3]);
    hi6 = madd52hi(hi6, f[3], f[3]);

    // z7 = 2*f3*f4
    let lo7 = madd52lo(zero, f3_2, f[4]);
    let hi7 = madd52hi(zero, f3_2, f[4]);

    // z8 = f4*f4
    let lo8 = madd52lo(zero, f[4], f[4]);
    let hi8 = madd52hi(zero, f[4], f[4]);

    // Combine lo + hi into 10 actual limbs
    let z0 = lo0;
    let z1 = _mm256_add_epi64(_mm256_add_epi64(lo1, hi0), hi0);
    let z2 = _mm256_add_epi64(_mm256_add_epi64(lo2, hi1), hi1);
    let z3 = _mm256_add_epi64(_mm256_add_epi64(lo3, hi2), hi2);
    let z4 = _mm256_add_epi64(_mm256_add_epi64(lo4, hi3), hi3);
    let z5 = _mm256_add_epi64(_mm256_add_epi64(lo5, hi4), hi4);
    let z6 = _mm256_add_epi64(_mm256_add_epi64(lo6, hi5), hi5);
    let z7 = _mm256_add_epi64(_mm256_add_epi64(lo7, hi6), hi6);
    let z8 = _mm256_add_epi64(_mm256_add_epi64(lo8, hi7), hi7);
    let z9 = _mm256_add_epi64(hi8, hi8);

    // Reduce upper limbs by ×19
    let z5_19 = mul19(z5);
    let z6_19 = mul19(z6);
    let z7_19 = mul19(z7);
    let z8_19 = mul19(z8);
    let z9_19 = mul19(z9);

    Self([
      _mm256_add_epi64(z0, z5_19),
      _mm256_add_epi64(z1, z6_19),
      _mm256_add_epi64(z2, z7_19),
      _mm256_add_epi64(z3, z8_19),
      _mm256_add_epi64(z4, z9_19),
    ])
    .reduce()
  }

  // square_and_negate_d (51-bit) removed — superseded by square_and_negate_d_wide.

  /// Square 52-bit inputs, returning 5 folded limbs before reduction.
  ///
  /// Shared helper for [`square_and_negate_d_wide`](Self::square_and_negate_d_wide).
  /// Performs the IFMA schoolbook, lo+hi recombination, and ×19 fold,
  /// returning the 5 combined limbs ready for optional D-lane negation
  /// and final carry propagation via `reduce()`.
  ///
  /// # Precondition
  ///
  /// All limbs must be ≤ 52 bits.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX-512 IFMA + VL are available.
  #[target_feature(enable = "avx2,avx512ifma,avx512vl")]
  #[allow(unsafe_op_in_unsafe_fn)]
  unsafe fn square_wide_fold(&self) -> [__m256i; 5] {
    let zero = _mm256_setzero_si256();
    let f = &self.0;

    // Interleaved lo/hi, double-accumulate for cross terms.

    // z0 = f0*f0
    let lo0 = madd52lo(zero, f[0], f[0]);
    let hi0 = madd52hi(zero, f[0], f[0]);

    // z1 = 2*f0*f1 (double-accumulate)
    let mut lo1 = madd52lo(zero, f[0], f[1]);
    let mut hi1 = madd52hi(zero, f[0], f[1]);
    lo1 = madd52lo(lo1, f[0], f[1]);
    hi1 = madd52hi(hi1, f[0], f[1]);

    // z2 = 2*f0*f2 + f1*f1
    let mut lo2 = madd52lo(zero, f[0], f[2]);
    let mut hi2 = madd52hi(zero, f[0], f[2]);
    lo2 = madd52lo(lo2, f[0], f[2]);
    hi2 = madd52hi(hi2, f[0], f[2]);
    lo2 = madd52lo(lo2, f[1], f[1]);
    hi2 = madd52hi(hi2, f[1], f[1]);

    // z3 = 2*f0*f3 + 2*f1*f2
    let mut lo3 = madd52lo(zero, f[0], f[3]);
    let mut hi3 = madd52hi(zero, f[0], f[3]);
    lo3 = madd52lo(lo3, f[0], f[3]);
    hi3 = madd52hi(hi3, f[0], f[3]);
    lo3 = madd52lo(lo3, f[1], f[2]);
    hi3 = madd52hi(hi3, f[1], f[2]);
    lo3 = madd52lo(lo3, f[1], f[2]);
    hi3 = madd52hi(hi3, f[1], f[2]);

    // z4 = 2*f0*f4 + 2*f1*f3 + f2*f2
    let mut lo4 = madd52lo(zero, f[0], f[4]);
    let mut hi4 = madd52hi(zero, f[0], f[4]);
    lo4 = madd52lo(lo4, f[0], f[4]);
    hi4 = madd52hi(hi4, f[0], f[4]);
    lo4 = madd52lo(lo4, f[1], f[3]);
    hi4 = madd52hi(hi4, f[1], f[3]);
    lo4 = madd52lo(lo4, f[1], f[3]);
    hi4 = madd52hi(hi4, f[1], f[3]);
    lo4 = madd52lo(lo4, f[2], f[2]);
    hi4 = madd52hi(hi4, f[2], f[2]);

    // z5 = 2*f1*f4 + 2*f2*f3
    let mut lo5 = madd52lo(zero, f[1], f[4]);
    let mut hi5 = madd52hi(zero, f[1], f[4]);
    lo5 = madd52lo(lo5, f[1], f[4]);
    hi5 = madd52hi(hi5, f[1], f[4]);
    lo5 = madd52lo(lo5, f[2], f[3]);
    hi5 = madd52hi(hi5, f[2], f[3]);
    lo5 = madd52lo(lo5, f[2], f[3]);
    hi5 = madd52hi(hi5, f[2], f[3]);

    // z6 = 2*f2*f4 + f3*f3
    let mut lo6 = madd52lo(zero, f[2], f[4]);
    let mut hi6 = madd52hi(zero, f[2], f[4]);
    lo6 = madd52lo(lo6, f[2], f[4]);
    hi6 = madd52hi(hi6, f[2], f[4]);
    lo6 = madd52lo(lo6, f[3], f[3]);
    hi6 = madd52hi(hi6, f[3], f[3]);

    // z7 = 2*f3*f4 (double-accumulate)
    let mut lo7 = madd52lo(zero, f[3], f[4]);
    let mut hi7 = madd52hi(zero, f[3], f[4]);
    lo7 = madd52lo(lo7, f[3], f[4]);
    hi7 = madd52hi(hi7, f[3], f[4]);

    // z8 = f4*f4
    let lo8 = madd52lo(zero, f[4], f[4]);
    let hi8 = madd52hi(zero, f[4], f[4]);

    // Recombine, fold ×19, reduce — identical to square().
    let z0 = lo0;
    let z1 = _mm256_add_epi64(_mm256_add_epi64(lo1, hi0), hi0);
    let z2 = _mm256_add_epi64(_mm256_add_epi64(lo2, hi1), hi1);
    let z3 = _mm256_add_epi64(_mm256_add_epi64(lo3, hi2), hi2);
    let z4 = _mm256_add_epi64(_mm256_add_epi64(lo4, hi3), hi3);
    let z5 = _mm256_add_epi64(_mm256_add_epi64(lo5, hi4), hi4);
    let z6 = _mm256_add_epi64(_mm256_add_epi64(lo6, hi5), hi5);
    let z7 = _mm256_add_epi64(_mm256_add_epi64(lo7, hi6), hi6);
    let z8 = _mm256_add_epi64(_mm256_add_epi64(lo8, hi7), hi7);
    let z9 = _mm256_add_epi64(hi8, hi8);

    let z5_19 = mul19(z5);
    let z6_19 = mul19(z6);
    let z7_19 = mul19(z7);
    let z8_19 = mul19(z8);
    let z9_19 = mul19(z9);

    [
      _mm256_add_epi64(z0, z5_19),
      _mm256_add_epi64(z1, z6_19),
      _mm256_add_epi64(z2, z7_19),
      _mm256_add_epi64(z3, z8_19),
      _mm256_add_epi64(z4, z9_19),
    ]
  }

  /// Square 52-bit inputs and negate lane D in the accumulator domain.
  ///
  /// Negates D in the **u64 domain** (after ×19 fold, before reduce)
  /// using a `p × 2^10` bias. The negated D lane emerges fully reduced
  /// (≤51 bits), matching the other lanes — unlike the post-reduce
  /// `2p − D` approach which leaves D at ≤52 bits.
  ///
  /// This tighter bound is critical: with all lanes ≤51 bits, pre-reducing
  /// `neg_s2` keeps S8 ≤ 53 bits for `mul_unreduced`, and S9 ≤ 52.6 bits.
  ///
  /// # Precondition
  ///
  /// All limbs must be ≤ 52 bits.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX-512 IFMA + VL are available.
  #[inline]
  #[target_feature(enable = "avx2,avx512ifma,avx512vl")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn square_and_negate_d_wide(&self) -> Self {
    let mut folded = self.square_wide_fold();

    // Negate D lane in u64 domain using p × 2^10 bias.
    //
    // p in radix-51: (2^51 − 19, 2^51 − 1, 2^51 − 1, 2^51 − 1, 2^51 − 1)
    // Bias = p_k × 2^10 ≈ 2^61: large enough to prevent underflow (max
    // accumulator after fold ≈ 2^60.2), small enough to fit in u64.
    // After reduce, the negated D lane is ≤51 bits — same as A, B, C.
    const D_BIAS_0: i64 = ((1i64 << 51) - 19) << 10;
    const D_BIAS_N: i64 = ((1i64 << 51) - 1) << 10;
    const D_BLEND: i32 = 0b1100_0000; // u32 positions 6-7 = u64 lane 3

    // _mm256_set_epi64x(d, c, b, a): only lane 3 (D) gets the bias.
    let bias_0 = _mm256_set_epi64x(D_BIAS_0, 0, 0, 0);
    let bias_n = _mm256_set_epi64x(D_BIAS_N, 0, 0, 0);

    macro_rules! neg_d {
      ($idx:expr, $bias:expr) => {
        let negated = _mm256_sub_epi64($bias, folded[$idx]);
        folded[$idx] = _mm256_blend_epi32::<D_BLEND>(folded[$idx], negated);
      };
    }

    neg_d!(0, bias_0);
    neg_d!(1, bias_n);
    neg_d!(2, bias_n);
    neg_d!(3, bias_n);
    neg_d!(4, bias_n);

    Self(folded).reduce()
  }

  /// Multiply by a "small" constant where only limb 0 is non-zero.
  ///
  /// The Hamburg scaling constants `(d2, d2, 2·d2, 2·d1)` fit in 18 bits,
  /// so their radix-51 representation has non-zero content only in limb 0.
  /// This needs only 10 IFMA ops (5 lo + 5 hi) instead of the full 50-op
  /// schoolbook, saving ~40 IFMA ops per constant multiply.
  ///
  /// # Precondition
  ///
  /// - `small.0[1..5]` must all be zero (only limb 0 non-zero).
  /// - `self` must have limbs ≤ 52 bits (same as `mul`).
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX-512 IFMA + VL are available.
  #[target_feature(enable = "avx2,avx512ifma,avx512vl")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn mul_small(&self, small: &Self) -> Self {
    let zero = _mm256_setzero_si256();
    let f = &self.0;
    let c = small.0[0]; // The only non-zero limb

    // 5 multiplies: f[k] * c — interleaved lo/hi.
    // Each pair is independent so the OOO engine overlaps freely.
    let lo0 = madd52lo(zero, f[0], c);
    let hi0 = madd52hi(zero, f[0], c);
    let lo1 = madd52lo(zero, f[1], c);
    let hi1 = madd52hi(zero, f[1], c);
    let lo2 = madd52lo(zero, f[2], c);
    let hi2 = madd52hi(zero, f[2], c);
    let lo3 = madd52lo(zero, f[3], c);
    let hi3 = madd52hi(zero, f[3], c);
    let lo4 = madd52lo(zero, f[4], c);
    let hi4 = madd52hi(zero, f[4], c);

    // Recombine: z_k = lo_k + 2 * hi_{k-1}
    // Wrap-around: 2 * hi_4 * 19 folds into limb 0.
    // With 18-bit constants and 51-bit inputs, hi values are ≤ 17 bits,
    // so 2*hi*19 ≤ ~5M — negligible overflow risk.
    let hi4_x2 = _mm256_add_epi64(hi4, hi4);

    Self([
      _mm256_add_epi64(lo0, mul19(hi4_x2)),
      _mm256_add_epi64(_mm256_add_epi64(lo1, hi0), hi0),
      _mm256_add_epi64(_mm256_add_epi64(lo2, hi1), hi1),
      _mm256_add_epi64(_mm256_add_epi64(lo3, hi2), hi2),
      _mm256_add_epi64(_mm256_add_epi64(lo4, hi3), hi3),
    ])
    .reduce()
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
  use super::*;
  use crate::auth::ed25519::field::FieldElement;

  fn test_field_elements() -> [FieldElement; 4] {
    let a = FieldElement::from_limbs([
      1_234_567_890_123,
      987_654_321_012,
      111_222_333_444,
      555_666_777_888,
      999_000_111_222,
    ]);
    let b = FieldElement::from_limbs([
      2_100_000_000_000,
      1_800_000_000_000,
      1_500_000_000_000,
      1_200_000_000_000,
      900_000_000_000,
    ]);
    let c = FieldElement::from_limbs([42, 0, 0, 0, 0]);
    let d = FieldElement::from_limbs([
      (1u64 << 51) - 20,
      (1u64 << 51) - 1,
      (1u64 << 51) - 1,
      (1u64 << 51) - 1,
      (1u64 << 51) - 1,
    ]);
    [a, b, c, d]
  }

  #[test]
  fn new_split_roundtrip() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    let [a, b, c, d] = test_field_elements();

    // SAFETY: AVX2 checked above.
    unsafe {
      let packed = FieldElement51x4::new(&a, &b, &c, &d);
      let [ra, rb, rc, rd] = packed.split();
      assert_eq!(ra.limbs(), a.limbs(), "lane A roundtrip");
      assert_eq!(rb.limbs(), b.limbs(), "lane B roundtrip");
      assert_eq!(rc.limbs(), c.limbs(), "lane C roundtrip");
      assert_eq!(rd.limbs(), d.limbs(), "lane D roundtrip");
    }
  }

  #[test]
  fn add_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    let [a, b, c, d] = test_field_elements();
    let [e, f, g, h] = [
      FieldElement::from_limbs([100, 200, 300, 400, 500]),
      FieldElement::from_limbs([600, 700, 800, 900, 1000]),
      FieldElement::from_limbs([1, 1, 1, 1, 1]),
      FieldElement::from_limbs([10, 20, 30, 40, 50]),
    ];

    // SAFETY: AVX2 checked above.
    unsafe {
      let lhs = FieldElement51x4::new(&a, &b, &c, &d);
      let rhs = FieldElement51x4::new(&e, &f, &g, &h);
      let sum = lhs.add(&rhs);
      let [ra, rb, rc, rd] = sum.split();

      assert_eq!(ra, a.add(&e), "lane A add");
      assert_eq!(rb, b.add(&f), "lane B add");
      assert_eq!(rc, c.add(&g), "lane C add");
      assert_eq!(rd, d.add(&h), "lane D add");
    }
  }

  #[test]
  fn mul_matches_scalar() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    let [a, b, c, d] = test_field_elements();
    let [e, f, g, h] = [
      FieldElement::from_limbs([100_000_000, 200_000_000, 300_000_000, 400_000_000, 500_000_000]),
      FieldElement::from_limbs([600_000_000, 700_000_000, 800_000_000, 900_000_000, 1_000_000_000]),
      FieldElement::from_limbs([17, 0, 0, 0, 0]),
      FieldElement::from_limbs([
        (1u64 << 51) - 100,
        (1u64 << 51) - 1,
        (1u64 << 51) - 1,
        (1u64 << 51) - 1,
        (1u64 << 51) - 1,
      ]),
    ];

    // SAFETY: AVX-512 IFMA checked above.
    unsafe {
      let lhs = FieldElement51x4::new(&a, &b, &c, &d);
      let rhs = FieldElement51x4::new(&e, &f, &g, &h);
      let product = lhs.mul(&rhs);
      let [ra, rb, rc, rd] = product.split();

      // Normalize both sides for comparison.
      let expected_a = a.mul(&e).normalize();
      let expected_b = b.mul(&f).normalize();
      let expected_c = c.mul(&g).normalize();
      let expected_d = d.mul(&h).normalize();

      assert_eq!(ra.normalize(), expected_a, "lane A mul");
      assert_eq!(rb.normalize(), expected_b, "lane B mul");
      assert_eq!(rc.normalize(), expected_c, "lane C mul");
      assert_eq!(rd.normalize(), expected_d, "lane D mul");
    }
  }

  #[test]
  fn square_matches_mul_self() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    let [a, b, c, d] = test_field_elements();

    // SAFETY: AVX-512 IFMA checked above.
    unsafe {
      let packed = FieldElement51x4::new(&a, &b, &c, &d);
      let sq = packed.square();
      let mul_self = packed.mul(&packed);

      let [sq_a, sq_b, sq_c, sq_d] = sq.split();
      let [ms_a, ms_b, ms_c, ms_d] = mul_self.split();

      assert_eq!(sq_a.normalize(), ms_a.normalize(), "lane A square vs mul");
      assert_eq!(sq_b.normalize(), ms_b.normalize(), "lane B square vs mul");
      assert_eq!(sq_c.normalize(), ms_c.normalize(), "lane C square vs mul");
      assert_eq!(sq_d.normalize(), ms_d.normalize(), "lane D square vs mul");
    }
  }

  #[test]
  fn square_matches_scalar() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    let [a, b, c, d] = test_field_elements();

    // SAFETY: AVX-512 IFMA checked above.
    unsafe {
      let packed = FieldElement51x4::new(&a, &b, &c, &d);
      let sq = packed.square();
      let [ra, rb, rc, rd] = sq.split();

      assert_eq!(ra.normalize(), a.square().normalize(), "lane A square");
      assert_eq!(rb.normalize(), b.square().normalize(), "lane B square");
      assert_eq!(rc.normalize(), c.square().normalize(), "lane C square");
      assert_eq!(rd.normalize(), d.square().normalize(), "lane D square");
    }
  }

  #[test]
  fn mul_small_matches_full_mul() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    let [a, b, c, d] = test_field_elements();

    // Hamburg-like small constants: only limb 0 is non-zero.
    let small = FieldElement::from_limbs([121_666, 0, 0, 0, 0]);
    let small2 = FieldElement::from_limbs([243_332, 0, 0, 0, 0]);

    // SAFETY: AVX-512 IFMA checked above.
    unsafe {
      let packed = FieldElement51x4::new(&a, &b, &c, &d);
      let constants = FieldElement51x4::new(&small, &small, &small2, &small);

      let via_small = packed.mul_small(&constants);
      let via_full = packed.mul(&constants);

      let [sa, sb, sc, sd] = via_small.split();
      let [fa, fb, fc, fd] = via_full.split();

      assert_eq!(sa.normalize(), fa.normalize(), "lane A mul_small vs mul");
      assert_eq!(sb.normalize(), fb.normalize(), "lane B mul_small vs mul");
      assert_eq!(sc.normalize(), fc.normalize(), "lane C mul_small vs mul");
      assert_eq!(sd.normalize(), fd.normalize(), "lane D mul_small vs mul");
    }
  }

  #[test]
  fn shuffle_badc() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    let [a, b, c, d] = test_field_elements();

    // SAFETY: AVX2 checked above.
    unsafe {
      let packed = FieldElement51x4::new(&a, &b, &c, &d);
      let shuffled = packed.shuffle(Shuffle::BADC);
      let [ra, rb, rc, rd] = shuffled.split();

      assert_eq!(ra.limbs(), b.limbs(), "BADC lane 0 = B");
      assert_eq!(rb.limbs(), a.limbs(), "BADC lane 1 = A");
      assert_eq!(rc.limbs(), d.limbs(), "BADC lane 2 = D");
      assert_eq!(rd.limbs(), c.limbs(), "BADC lane 3 = C");
    }
  }

  /// Helper: create field elements with 53-bit limbs (simulating diff_sum output).
  fn test_53bit_field_elements() -> [FieldElement; 4] {
    // 53-bit max: (1 << 53) - 1 = 9_007_199_254_740_991
    let a = FieldElement::from_limbs([
      (1u64 << 53) - 1,
      (1u64 << 53) - 100,
      (1u64 << 52) + 12345,
      (1u64 << 53) - 999,
      (1u64 << 52) + 1,
    ]);
    let b = FieldElement::from_limbs([
      (1u64 << 52) + 7,
      (1u64 << 53) - 42,
      (1u64 << 53) - 1,
      (1u64 << 52) + 100_000,
      (1u64 << 53) - 7777,
    ]);
    let c = FieldElement::from_limbs([
      (1u64 << 51) + 1,
      (1u64 << 52),
      (1u64 << 53) - 2,
      (1u64 << 51),
      (1u64 << 52) + 42,
    ]);
    let d = FieldElement::from_limbs([
      (1u64 << 53) - 1,
      (1u64 << 53) - 1,
      (1u64 << 53) - 1,
      (1u64 << 53) - 1,
      (1u64 << 53) - 1,
    ]);
    [a, b, c, d]
  }

  #[test]
  fn mul_unreduced_matches_reduce_then_mul() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    let [a, b, c, d] = test_53bit_field_elements();
    let [e, f, g, h] = [
      FieldElement::from_limbs([
        (1u64 << 53) - 5,
        (1u64 << 52) + 100,
        (1u64 << 51) + 999,
        (1u64 << 53) - 1,
        (1u64 << 52) + 50_000,
      ]),
      FieldElement::from_limbs([
        (1u64 << 52),
        (1u64 << 53) - 3,
        (1u64 << 52) + 7,
        (1u64 << 51) + 42,
        (1u64 << 53) - 100,
      ]),
      FieldElement::from_limbs([
        (1u64 << 53) - 1,
        (1u64 << 53) - 1,
        (1u64 << 53) - 1,
        (1u64 << 53) - 1,
        (1u64 << 53) - 1,
      ]),
      FieldElement::from_limbs([
        (1u64 << 51) + 10,
        (1u64 << 51) + 20,
        (1u64 << 51) + 30,
        (1u64 << 51) + 40,
        (1u64 << 51) + 50,
      ]),
    ];

    // SAFETY: AVX-512 IFMA checked above.
    unsafe {
      let lhs = FieldElement51x4::new(&a, &b, &c, &d);
      let rhs = FieldElement51x4::new(&e, &f, &g, &h);

      // mul_unreduced: accepts 53-bit inputs directly
      let unreduced_product = lhs.mul_unreduced(&rhs);
      let [ua, ub, uc, ud] = unreduced_product.split();

      // Reference: reduce both operands to ≤51 bits, then use standard mul
      let lhs_reduced = lhs.reduce();
      let rhs_reduced = rhs.reduce();
      let reduced_product = lhs_reduced.mul(&rhs_reduced);
      let [ra, rb, rc, rd] = reduced_product.split();

      assert_eq!(ua.normalize(), ra.normalize(), "lane A mul_unreduced vs reduce+mul");
      assert_eq!(ub.normalize(), rb.normalize(), "lane B mul_unreduced vs reduce+mul");
      assert_eq!(uc.normalize(), rc.normalize(), "lane C mul_unreduced vs reduce+mul");
      assert_eq!(ud.normalize(), rd.normalize(), "lane D mul_unreduced vs reduce+mul");
    }
  }

  #[test]
  fn mul_unreduced_max_limbs() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    // Worst case: all limbs at maximum 53-bit value.
    let max = FieldElement::from_limbs([
      (1u64 << 53) - 1,
      (1u64 << 53) - 1,
      (1u64 << 53) - 1,
      (1u64 << 53) - 1,
      (1u64 << 53) - 1,
    ]);

    // SAFETY: AVX-512 IFMA checked above.
    unsafe {
      let packed = FieldElement51x4::new(&max, &max, &max, &max);

      let unreduced = packed.mul_unreduced(&packed);
      let [ua, ub, uc, ud] = unreduced.split();

      let reduced = packed.reduce();
      let reference = reduced.mul(&reduced);
      let [ra, rb, rc, rd] = reference.split();

      assert_eq!(ua.normalize(), ra.normalize(), "lane A max-limb mul_unreduced");
      assert_eq!(ub.normalize(), rb.normalize(), "lane B max-limb mul_unreduced");
      assert_eq!(uc.normalize(), rc.normalize(), "lane C max-limb mul_unreduced");
      assert_eq!(ud.normalize(), rd.normalize(), "lane D max-limb mul_unreduced");
    }
  }

  #[test]
  fn mul_unreduced_with_diff_sum_output() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    // Simulate the actual use case: diff_sum() on reduced field elements.
    let [a, b, c, d] = test_field_elements();

    // SAFETY: AVX-512 IFMA checked above.
    unsafe {
      let packed = FieldElement51x4::new(&a, &b, &c, &d);
      let ds = packed.diff_sum(); // produces up to 53-bit limbs

      // mul_unreduced should handle diff_sum output directly
      let result = ds.mul_unreduced(&packed);
      let [ua, ub, uc, ud] = result.split();

      // Reference: reduce first, then mul
      let ds_reduced = ds.reduce();
      let reference = ds_reduced.mul(&packed);
      let [ra, rb, rc, rd] = reference.split();

      assert_eq!(ua.normalize(), ra.normalize(), "lane A diff_sum mul_unreduced");
      assert_eq!(ub.normalize(), rb.normalize(), "lane B diff_sum mul_unreduced");
      assert_eq!(uc.normalize(), rc.normalize(), "lane C diff_sum mul_unreduced");
      assert_eq!(ud.normalize(), rd.normalize(), "lane D diff_sum mul_unreduced");
    }
  }

  #[test]
  fn mul_small_unreduced_matches_reduce_then_mul_small() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    let [a, b, c, d] = test_53bit_field_elements();

    // Hamburg-like small constants: only limb 0 is non-zero.
    let small = FieldElement::from_limbs([121_666, 0, 0, 0, 0]);
    let small2 = FieldElement::from_limbs([243_332, 0, 0, 0, 0]);

    // SAFETY: AVX-512 IFMA checked above.
    unsafe {
      let packed = FieldElement51x4::new(&a, &b, &c, &d);
      let constants = FieldElement51x4::new(&small, &small, &small2, &small);

      let via_unreduced = packed.mul_small_unreduced(&constants);
      let [ua, ub, uc, ud] = via_unreduced.split();

      let packed_reduced = packed.reduce();
      let via_reduced = packed_reduced.mul_small(&constants);
      let [ra, rb, rc, rd] = via_reduced.split();

      assert_eq!(ua.normalize(), ra.normalize(), "lane A mul_small_unreduced");
      assert_eq!(ub.normalize(), rb.normalize(), "lane B mul_small_unreduced");
      assert_eq!(uc.normalize(), rc.normalize(), "lane C mul_small_unreduced");
      assert_eq!(ud.normalize(), rd.normalize(), "lane D mul_small_unreduced");
    }
  }

  #[test]
  fn mul_unreduced_51bit_inputs_matches_mul() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    // With 51-bit inputs (already reduced), mul_unreduced should produce
    // the same result as mul since overflow bits are all zero.
    let [a, b, c, d] = test_field_elements();
    let [e, f, g, h] = [
      FieldElement::from_limbs([100_000_000, 200_000_000, 300_000_000, 400_000_000, 500_000_000]),
      FieldElement::from_limbs([600_000_000, 700_000_000, 800_000_000, 900_000_000, 1_000_000_000]),
      FieldElement::from_limbs([17, 0, 0, 0, 0]),
      FieldElement::from_limbs([
        (1u64 << 51) - 100,
        (1u64 << 51) - 1,
        (1u64 << 51) - 1,
        (1u64 << 51) - 1,
        (1u64 << 51) - 1,
      ]),
    ];

    // SAFETY: AVX-512 IFMA checked above.
    unsafe {
      let lhs = FieldElement51x4::new(&a, &b, &c, &d);
      let rhs = FieldElement51x4::new(&e, &f, &g, &h);

      let via_mul = lhs.mul(&rhs);
      let via_unreduced = lhs.mul_unreduced(&rhs);

      let [ma, mb, mc, md] = via_mul.split();
      let [ua, ub, uc, ud] = via_unreduced.split();

      assert_eq!(ua.normalize(), ma.normalize(), "lane A mul vs mul_unreduced (51-bit)");
      assert_eq!(ub.normalize(), mb.normalize(), "lane B mul vs mul_unreduced (51-bit)");
      assert_eq!(uc.normalize(), mc.normalize(), "lane C mul vs mul_unreduced (51-bit)");
      assert_eq!(ud.normalize(), md.normalize(), "lane D mul vs mul_unreduced (51-bit)");
    }
  }

  #[test]
  fn square_and_negate_d_wide_matches_scalar() {
    if !is_x86_feature_detected!("avx512ifma") {
      return;
    }
    let [a, b, c, d] = test_field_elements();

    // SAFETY: AVX-512 IFMA checked above.
    unsafe {
      let packed = FieldElement51x4::new(&a, &b, &c, &d);
      let result = packed.square_and_negate_d_wide();
      let [ra, rb, rc, rd] = result.split();

      assert_eq!(ra.normalize(), a.square().normalize(), "lane A sq_neg_d_wide");
      assert_eq!(rb.normalize(), b.square().normalize(), "lane B sq_neg_d_wide");
      assert_eq!(rc.normalize(), c.square().normalize(), "lane C sq_neg_d_wide");
      assert_eq!(rd.normalize(), d.square().neg().normalize(), "lane D should be negated");
    }
  }
}
