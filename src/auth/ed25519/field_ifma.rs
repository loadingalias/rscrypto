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
//! Field arithmetic is modular math (mod 2^255 - 19). Wrapping arithmetic is
//! reserved for operations whose field representation is intentionally modular.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{
  field::FieldElement,
  field_avx2::{Lanes, Shuffle},
};

// Constants

const MASK51: i64 = (1i64 << 51) - 1;
#[cfg(feature = "ed25519")]
const MASK52: i64 = (1i64 << 52) - 1;

/// Subtraction bias: 2p in radix-51. Limb 0 accounts for the -19 term.
const BIAS_0: i64 = 2 * ((1i64 << 51) - 19);
const BIAS_N: i64 = 2 * ((1i64 << 51) - 1);

// Type

/// Four field elements packed for AVX-512 IFMA parallel processing.
///
/// Uses radix-2^51: 5 limbs per element stored in u64 lanes. Four elements
/// are interleaved across 5 × `__m256i` registers (4 × u64 each).
#[derive(Clone, Copy)]
#[cfg(target_arch = "x86_64")]
pub(crate) struct FieldElement51x4(pub(crate) [__m256i; 5]);

// Helpers

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
fn mul19(x: __m256i) -> __m256i {
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
fn madd52lo(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
  _mm256_madd52lo_epu64(acc, a, b)
}

/// IFMA fused multiply-accumulate (high 52 bits): `acc += (a * b)[103:52]`.
///
/// # Safety
///
/// Caller must ensure AVX-512 IFMA + VL are available.
#[inline]
#[target_feature(enable = "avx2,avx512ifma,avx512vl")]
fn madd52hi(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
  _mm256_madd52hi_epu64(acc, a, b)
}

#[inline(always)]
fn u64_as_i64_bits(value: u64) -> i64 {
  i64::from_ne_bytes(value.to_ne_bytes())
}

// FieldElement51x4 implementation

#[cfg(target_arch = "x86_64")]
impl FieldElement51x4 {
  /// The zero element (all four lanes).
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  pub(crate) fn zero() -> Self {
    Self([_mm256_setzero_si256(); 5])
  }

  /// Pack four scalar field elements into IFMA vectorized form.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  pub(crate) fn new(a: &FieldElement, b: &FieldElement, c: &FieldElement, d: &FieldElement) -> Self {
    let al = a.limbs();
    let bl = b.limbs();
    let cl = c.limbs();
    let dl = d.limbs();

    Self([
      _mm256_set_epi64x(
        u64_as_i64_bits(dl[0]),
        u64_as_i64_bits(cl[0]),
        u64_as_i64_bits(bl[0]),
        u64_as_i64_bits(al[0]),
      ),
      _mm256_set_epi64x(
        u64_as_i64_bits(dl[1]),
        u64_as_i64_bits(cl[1]),
        u64_as_i64_bits(bl[1]),
        u64_as_i64_bits(al[1]),
      ),
      _mm256_set_epi64x(
        u64_as_i64_bits(dl[2]),
        u64_as_i64_bits(cl[2]),
        u64_as_i64_bits(bl[2]),
        u64_as_i64_bits(al[2]),
      ),
      _mm256_set_epi64x(
        u64_as_i64_bits(dl[3]),
        u64_as_i64_bits(cl[3]),
        u64_as_i64_bits(bl[3]),
        u64_as_i64_bits(al[3]),
      ),
      _mm256_set_epi64x(
        u64_as_i64_bits(dl[4]),
        u64_as_i64_bits(cl[4]),
        u64_as_i64_bits(bl[4]),
        u64_as_i64_bits(al[4]),
      ),
    ])
  }

  /// Unpack back to four scalar field elements.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  pub(crate) fn split(&self) -> [FieldElement; 4] {
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
      // SAFETY: AVX2 is active in this function, and `tmp` provides 32 writable bytes for the unaligned store.
      unsafe { _mm256_storeu_si256(tmp.as_mut_ptr().cast(), *vec) };
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
  pub(crate) fn add(&self, rhs: &Self) -> Self {
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
  pub(crate) fn sub(&self, rhs: &Self) -> Self {
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
  pub(crate) fn negate_lazy(&self) -> Self {
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
  pub(crate) fn shuffle(&self, pattern: Shuffle) -> Self {
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
      Shuffle::SwapPairs => do_shuffle!(0b10_11_00_01),
      Shuffle::SwapAB => do_shuffle!(0b11_10_00_01),
      Shuffle::SwapCD => do_shuffle!(0b10_11_01_00),
      #[cfg(feature = "ed25519")]
      Shuffle::BroadcastA => do_shuffle!(0b00_00_00_00),
      #[cfg(feature = "ed25519")]
      Shuffle::BroadcastB => do_shuffle!(0b01_01_01_01),
      #[cfg(feature = "ed25519")]
      Shuffle::AlternateCA => do_shuffle!(0b00_10_00_10),
      #[cfg(feature = "ed25519")]
      Shuffle::OuterDInnerB => do_shuffle!(0b11_01_01_11),
      Shuffle::OuterAInnerD => do_shuffle!(0b00_11_11_00),
      Shuffle::AlternateCB => do_shuffle!(0b01_10_01_10),
      #[cfg(feature = "ed25519")]
      Shuffle::RepeatAB => do_shuffle!(0b01_00_01_00),
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
  pub(crate) fn blend(&self, other: &Self, lanes: Lanes) -> Self {
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
      #[cfg(feature = "ed25519")]
      Lanes::C => do_blend!(0b0011_0000),
      Lanes::D => do_blend!(0b1100_0000),
      Lanes::AB => do_blend!(0b0000_1111),
      Lanes::AC => do_blend!(0b0011_0011),
      #[cfg(feature = "ed25519")]
      Lanes::AD => do_blend!(0b1100_0011),
      #[cfg(feature = "ed25519")]
      Lanes::BC => do_blend!(0b0011_1100),
      #[cfg(feature = "ed25519")]
      Lanes::ExceptA => do_blend!(0b1111_1100),
    }
  }

  /// Select `other` where `mask` is all-ones and `self` where `mask` is zero.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  pub(crate) fn select_mask(&self, other: &Self, mask: u64) -> Self {
    let mask = _mm256_set1_epi64x(u64_as_i64_bits(mask));
    Self([
      _mm256_xor_si256(
        self.0[0],
        _mm256_and_si256(mask, _mm256_xor_si256(self.0[0], other.0[0])),
      ),
      _mm256_xor_si256(
        self.0[1],
        _mm256_and_si256(mask, _mm256_xor_si256(self.0[1], other.0[1])),
      ),
      _mm256_xor_si256(
        self.0[2],
        _mm256_and_si256(mask, _mm256_xor_si256(self.0[2], other.0[2])),
      ),
      _mm256_xor_si256(
        self.0[3],
        _mm256_and_si256(mask, _mm256_xor_si256(self.0[3], other.0[3])),
      ),
      _mm256_xor_si256(
        self.0[4],
        _mm256_and_si256(mask, _mm256_xor_si256(self.0[4], other.0[4])),
      ),
    ])
  }

  /// Compute `(B-A, A+B, D-C, C+D)` in one pass.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  pub(crate) fn diff_sum(&self) -> Self {
    let swapped = self.shuffle(Shuffle::SwapPairs); // (B, A, D, C)
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
  pub(crate) fn reduce(mut self) -> Self {
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

  /// Fold one radix-51 carry per limb before a 52-bit IFMA multiplication.
  ///
  /// # Safety
  ///
  /// The caller must establish AVX2, AVX-512 IFMA and VL.
  #[inline]
  #[target_feature(enable = "avx2,avx512ifma,avx512vl")]
  pub(crate) fn reduce_for_mul(&self) -> Self {
    let mask = _mm256_set1_epi64x(MASK51);
    let low = self.0.map(|limb| _mm256_and_si256(limb, mask));
    let carry = self.0.map(|limb| _mm256_srli_epi64::<51>(limb));
    // Each carry is at most 8191. Even the folded top carry leaves limb zero below 2^52.
    Self([
      madd52lo(low[0], carry[4], _mm256_set1_epi64x(19)),
      _mm256_add_epi64(low[1], carry[0]),
      _mm256_add_epi64(low[2], carry[1]),
      _mm256_add_epi64(low[3], carry[2]),
      _mm256_add_epi64(low[4], carry[3]),
    ])
  }

  /// Reduce public limbs only when an IFMA operand exceeds 52 bits.
  ///
  /// # Safety
  ///
  /// The caller must establish AVX2, AVX-512 IFMA and VL. Every lane must hold public data.
  #[cfg(feature = "ed25519")]
  #[inline]
  #[target_feature(enable = "avx2,avx512ifma,avx512vl")]
  pub(crate) fn public_mul_input(self) -> Self {
    let [a, b, c, d, e] = self.0;
    let ab = _mm256_or_si256(a, b);
    let cd = _mm256_or_si256(c, d);
    let bits = _mm256_or_si256(_mm256_or_si256(ab, cd), e);
    if _mm256_testz_si256(bits, _mm256_set1_epi64x(!MASK52)) != 0 {
      self
    } else {
      self.reduce_for_mul()
    }
  }

  /// Multiply two vectorized field elements using AVX-512 IFMA.
  ///
  /// Full 5×5 schoolbook with dual accumulators (`madd52lo` + `madd52hi`),
  /// post-reduction of upper limbs, and carry propagation.
  ///
  /// # Precondition
  ///
  /// Each operand must fit in 52 bits per limb. `reduce_for_mul()` establishes this bound.
  /// Apply it to a `diff_sum()` result before multiplication. The sum can exceed 52 bits.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX-512 IFMA + VL are available.
  #[target_feature(enable = "avx2,avx512ifma,avx512vl")]
  pub(crate) fn mul(&self, rhs: &Self) -> Self {
    let zero = _mm256_setzero_si256();
    let f = &self.0;
    let g = &rhs.0;

    // Phase 1: 5×5 schoolbook with dual accumulators (50 IFMA ops)
    //
    // Interleaved lo/hi scheduling: each product term issues madd52lo then
    // madd52hi before the next term. This exposes the independent lo and hi
    // dependency chains to Intel's dual-issue IFMA ports (0 and 1), letting
    // the OOO engine overlap them instead of serialising all-lo-then-all-hi.

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

    // Phase 2: Combine lo + hi into 10 actual limbs
    //
    // z_k = lo_k + 2 * hi_{k-1}
    // The factor 2 comes from: bits [103:52] have value v * 2^52 = v * 2 * 2^51,
    // so they contribute v * 2 to the next limb.

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

    // Phase 3: Reduce upper limbs by ×19 and fold into lower 5
    //
    // 2^(51*5) = 2^255 ≡ 19 (mod p)

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
    .reduce_for_mul()
  }

  /// Square a field element with five unsigned 52-bit limbs.
  ///
  /// # Safety
  ///
  /// The caller must establish AVX2, AVX-512 IFMA and VL.
  #[inline]
  #[target_feature(enable = "avx2,avx512ifma,avx512vl")]
  #[cfg(feature = "ed25519")]
  pub(crate) fn square(&self) -> Self {
    Self(self.square_wide_fold()).reduce_for_mul()
  }

  /// Fold the square of five unsigned 52-bit limbs modulo 2^255 - 19.
  ///
  /// # Safety
  ///
  /// The caller must establish AVX2, AVX-512 IFMA and VL.
  #[inline]
  #[target_feature(enable = "avx2,avx512ifma,avx512vl")]
  #[cfg(feature = "ed25519")]
  fn square_wide_fold(&self) -> [__m256i; 5] {
    let f = &self.0;
    let zero = _mm256_setzero_si256();
    let twice = |v| _mm256_add_epi64(v, v);
    // IFMA consumes 52 bits per operand. Double the cross products after multiplication.
    macro_rules! square_half {
      ($madd:ident) => {{
        let c2 = $madd(zero, f[0], f[2]);
        let c3 = $madd($madd(zero, f[0], f[3]), f[1], f[2]);
        let c4 = $madd($madd(zero, f[0], f[4]), f[1], f[3]);
        let c5 = $madd($madd(zero, f[1], f[4]), f[2], f[3]);
        let c6 = $madd(zero, f[2], f[4]);
        [
          $madd(zero, f[0], f[0]),
          twice($madd(zero, f[0], f[1])),
          $madd(twice(c2), f[1], f[1]),
          twice(c3),
          $madd(twice(c4), f[2], f[2]),
          twice(c5),
          $madd(twice(c6), f[3], f[3]),
          twice($madd(zero, f[3], f[4])),
          $madd(zero, f[4], f[4]),
        ]
      }};
    }
    let lo = square_half!(madd52lo);
    let hi = square_half!(madd52hi);
    // Each high half starts at bit 52. Each radix limb starts at bit 51.
    let limb = |index: usize| _mm256_add_epi64(lo[index], twice(hi[index.strict_sub(1)]));
    [
      _mm256_add_epi64(lo[0], mul19(limb(5))),
      _mm256_add_epi64(limb(1), mul19(limb(6))),
      _mm256_add_epi64(limb(2), mul19(limb(7))),
      _mm256_add_epi64(limb(3), mul19(limb(8))),
      _mm256_add_epi64(limb(4), mul19(twice(hi[8]))),
    ]
  }

  /// Multiply by a "small" constant where only limb 0 is non-zero.
  ///
  /// The Hamburg scaling constants `(d2, d2, 2·d2, 2·d1)` fit in 18 bits,
  /// so their radix-51 representation has non-zero content only in limb 0.
  /// This uses 5 low and 5 high IFMA operations instead of the full
  /// schoolbook multiplication.
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
  pub(crate) fn mul_small(&self, small: &Self) -> Self {
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

// Tests

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
#[cfg(feature = "ed25519")]
mod tests {
  use super::{FieldElement, *};

  fn avx512ifma_available_for_tests() -> bool {
    !cfg!(miri) && std::arch::is_x86_feature_detected!("avx512ifma")
  }

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
    if !std::arch::is_x86_feature_detected!("avx2") {
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
    if !std::arch::is_x86_feature_detected!("avx2") {
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
    if !avx512ifma_available_for_tests() {
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
    if !avx512ifma_available_for_tests() {
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
    if !avx512ifma_available_for_tests() {
      return;
    }
    let max = (1u64 << 52).strict_sub(1);
    let a = FieldElement::from_limbs([max; 5]);
    let b = FieldElement::from_limbs([0, max, 0, max, 0]);
    let c = FieldElement::from_limbs([max, 0, max, 0, max]);
    let d = FieldElement::from_limbs([1 << 51; 5]);

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
    if !avx512ifma_available_for_tests() {
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
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    let [a, b, c, d] = test_field_elements();

    // SAFETY: AVX2 checked above.
    unsafe {
      let packed = FieldElement51x4::new(&a, &b, &c, &d);
      let shuffled = packed.shuffle(Shuffle::SwapPairs);
      let [ra, rb, rc, rd] = shuffled.split();

      assert_eq!(ra.limbs(), b.limbs(), "BADC lane 0 = B");
      assert_eq!(rb.limbs(), a.limbs(), "BADC lane 1 = A");
      assert_eq!(rc.limbs(), d.limbs(), "BADC lane 2 = D");
      assert_eq!(rd.limbs(), c.limbs(), "BADC lane 3 = C");
    }
  }
}
