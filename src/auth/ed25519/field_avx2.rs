//! AVX2 vectorized field arithmetic for Ed25519.
//!
//! Four field elements over GF(2²⁵⁵ − 19) are processed in parallel using
//! `__m256i` registers with a radix-10×(26/25) representation. This enables
//! 4-way parallel field multiply/square via `vpmuludq`, matching the HWCD'08
//! parallel point formulas that operate uniformly across (X, Y, Z, T).
//!
//! # Lane layout
//!
//! Each `__m256i` holds 8 × u32 encoding one (even, odd) limb pair from each
//! of four independent field elements A, B, C, D:
//!
//! ```text
//! self.0[i] = (a_{2i}, b_{2i}, a_{2i+1}, b_{2i+1}, c_{2i}, d_{2i}, c_{2i+1}, d_{2i+1})
//! ```
//!
//! Even-indexed limbs use 26 bits, odd-indexed limbs use 25 bits (matching
//! the alternating radix needed for `vpmuludq`'s 32-bit input constraint).
//!
//! # Arithmetic convention
//!
//! Field arithmetic is modular math (mod 2²⁵⁵ − 19). Per CLAUDE.md rules,
//! `wrapping_*` is the correct choice for intentional modular arithmetic.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::field::FieldElement;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LOW_25_BITS: i64 = (1 << 25) - 1;
const LOW_26_BITS: i64 = (1 << 26) - 1;

/// D-lane blend mask: positions 5 and 7 in each `__m256i` (u32 view).
const D_BLEND: i32 = 0b1010_0000u8 as i32;

/// Four field elements packed for AVX2 parallel processing.
///
/// Uses radix 10×(26/25): 10 limbs per element with alternating 26-bit
/// (even) and 25-bit (odd) widths. Four elements are interleaved across
/// 5 × `__m256i` registers.
#[derive(Clone, Copy)]
#[cfg(target_arch = "x86_64")]
pub(crate) struct FieldElement2625x4(pub(crate) [__m256i; 5]);

// ---------------------------------------------------------------------------
// Shuffle patterns for _mm256_permutevar8x32_epi32
// ---------------------------------------------------------------------------

/// Lane rearrangement patterns for `shuffle`.
#[derive(Clone, Copy)]
#[repr(u8)]
#[allow(clippy::upper_case_acronyms, dead_code)] // Lane labels, complete API.
pub(crate) enum Shuffle {
  /// Identity: (A, B, C, D) → (A, B, C, D)
  ABCD,
  /// Swap pairs: (A, B, C, D) → (B, A, D, C)
  BADC,
  /// Swap left pair only: (A, B, C, D) → (B, A, C, D)
  BACD,
  /// Swap right pair only: (A, B, C, D) → (A, B, D, C)
  ABDC,
  /// Broadcast A: (A, B, C, D) → (A, A, A, A)
  AAAA,
  /// Broadcast B: (A, B, C, D) → (B, B, B, B)
  BBBB,
  /// (A, B, C, D) → (C, A, C, A)
  CACA,
  /// (A, B, C, D) → (D, B, B, D)
  DBBD,
  /// (A, B, C, D) → (A, D, D, A)
  ADDA,
  /// (A, B, C, D) → (C, B, C, B)
  CBCB,
  /// (A, B, C, D) → (A, B, A, B)
  ABAB,
}

impl Shuffle {
  /// Get the permutation control vector for `_mm256_permutevar8x32_epi32`.
  #[inline(always)]
  fn control(self) -> [i32; 8] {
    match self {
      Self::ABCD => [0, 1, 2, 3, 4, 5, 6, 7],
      Self::BADC => [1, 0, 3, 2, 5, 4, 7, 6],
      Self::BACD => [1, 0, 3, 2, 4, 5, 6, 7],
      Self::ABDC => [0, 1, 2, 3, 5, 4, 7, 6],
      Self::AAAA => [0, 0, 2, 2, 0, 0, 2, 2],
      Self::BBBB => [1, 1, 3, 3, 1, 1, 3, 3],
      Self::CACA => [4, 0, 6, 2, 4, 0, 6, 2],
      Self::DBBD => [5, 1, 7, 3, 1, 5, 3, 7],
      Self::ADDA => [0, 5, 2, 7, 5, 0, 7, 2],
      Self::CBCB => [4, 1, 6, 3, 4, 1, 6, 3],
      Self::ABAB => [0, 1, 2, 3, 0, 1, 2, 3],
    }
  }
}

// ---------------------------------------------------------------------------
// Blend lane selectors for _mm256_blend_epi32
// ---------------------------------------------------------------------------

/// Lane selection masks for `blend`.
///
/// Each variant selects which 32-bit lanes come from the second operand.
/// Lane positions: `[a_even, b_even, a_odd, b_odd, c_even, d_even, c_odd, d_odd]`.
#[derive(Clone, Copy)]
#[repr(u8)]
#[allow(clippy::upper_case_acronyms, dead_code)] // Lane labels, complete API.
pub(crate) enum Lanes {
  /// Select A lanes: positions 0, 2
  A = 0b0000_0101,
  /// Select B lanes: positions 1, 3
  B = 0b0000_1010,
  /// Select C lanes: positions 4, 6
  C = 0b0101_0000,
  /// Select D lanes: positions 5, 7
  D = 0b1010_0000,
  /// Select A and B lanes: positions 0-3
  AB = 0b0000_1111,
  /// Select A and C lanes: positions 0, 2, 4, 6
  AC = 0b0101_0101,
  /// Select A and D lanes: positions 0, 2, 5, 7
  AD = 0b1010_0101,
  /// Select B and C lanes: positions 1, 3, 4, 6
  BC = 0b0101_1010,
  /// Select B, C, and D lanes: positions 1, 3, 4, 5, 6, 7
  BCD = 0b1111_1010,
  /// Select C and D lanes: positions 4-7
  CD = 0b1111_0000,
  /// Select all lanes
  ABCD = 0b1111_1111,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Zero-extend packed u32 pairs into two u64x4 vectors suitable for `vpmuludq`.
///
/// Input: `(a_even, b_even, a_odd, b_odd, c_even, d_even, c_odd, d_odd)`
/// Returns:
/// - lo: `(a_even, 0, b_even, 0, c_even, 0, d_even, 0)` — even limbs
/// - hi: `(a_odd, 0, b_odd, 0, c_odd, 0, d_odd, 0)` — odd limbs
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn unpack_pair(v: __m256i) -> (__m256i, __m256i) {
  let zero = _mm256_setzero_si256();
  let lo = _mm256_unpacklo_epi32(v, zero);
  let hi = _mm256_unpackhi_epi32(v, zero);
  (lo, hi)
}

/// Pack two u64x4 vectors (even/odd limbs) back into interleaved u32x8.
///
/// Inverse of `unpack_pair`. Truncates each u64 lane to u32.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn repack_pair(lo: __m256i, hi: __m256i) -> __m256i {
  // Shuffle to pack the low u32 of each 64-bit lane into consecutive positions.
  // _mm256_shuffle_epi32 with imm [0, 2, 0, 2] = 0b10_00_10_00 packs positions
  // 0 and 2 within each 128-bit lane.
  let lo_packed = _mm256_shuffle_epi32::<0b10_00_10_00>(lo);
  let hi_packed = _mm256_shuffle_epi32::<0b10_00_10_00>(hi);
  // Blend: positions 2,3,6,7 from hi, rest from lo.
  _mm256_blend_epi32::<0b1100_1100>(lo_packed, hi_packed)
}

/// `vpmuludq` wrapper: multiply low 32-bit of each 64-bit lane, producing u64.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn mul32(a: __m256i, b: __m256i) -> __m256i {
  _mm256_mul_epu32(a, b)
}

/// Add two u64x4 vectors.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn add64(a: __m256i, b: __m256i) -> __m256i {
  _mm256_add_epi64(a, b)
}

// ---------------------------------------------------------------------------
// FieldElement2625x4 implementation
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
impl FieldElement2625x4 {
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

  /// Pack four scalar field elements into a vectorized representation.
  ///
  /// Each 5×51 element is split into 10×(26/25) limbs and interleaved.
  /// The result is reduced to ensure limbs fit their nominal widths.
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

    let mask = LOW_26_BITS as u64;
    let out = [
      Self::pack_limb_pair(al[0], bl[0], cl[0], dl[0], mask),
      Self::pack_limb_pair(al[1], bl[1], cl[1], dl[1], mask),
      Self::pack_limb_pair(al[2], bl[2], cl[2], dl[2], mask),
      Self::pack_limb_pair(al[3], bl[3], cl[3], dl[3], mask),
      Self::pack_limb_pair(al[4], bl[4], cl[4], dl[4], mask),
    ];

    // Odd limbs from a non-reduced FieldElement may exceed 25 bits.
    Self(out).reduce()
  }

  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  unsafe fn pack_limb_pair(al: u64, bl: u64, cl: u64, dl: u64, mask: u64) -> __m256i {
    _mm256_setr_epi32(
      (al & mask) as i32,
      (bl & mask) as i32,
      (al >> 26) as i32,
      (bl >> 26) as i32,
      (cl & mask) as i32,
      (dl & mask) as i32,
      (cl >> 26) as i32,
      (dl >> 26) as i32,
    )
  }

  /// Unpack into four independent scalar field elements.
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

    for ((((a_out, b_out), c_out), d_out), vec) in al
      .iter_mut()
      .zip(bl.iter_mut())
      .zip(cl.iter_mut())
      .zip(dl.iter_mut())
      .zip(self.0.iter())
    {
      let mut tmp = [0u32; 8];
      _mm256_storeu_si256(tmp.as_mut_ptr().cast(), *vec);

      *a_out = u64::from(tmp[0]) | (u64::from(tmp[2]) << 26);
      *b_out = u64::from(tmp[1]) | (u64::from(tmp[3]) << 26);
      *c_out = u64::from(tmp[4]) | (u64::from(tmp[6]) << 26);
      *d_out = u64::from(tmp[5]) | (u64::from(tmp[7]) << 26);
    }

    [
      FieldElement::from_limbs(al),
      FieldElement::from_limbs(bl),
      FieldElement::from_limbs(cl),
      FieldElement::from_limbs(dl),
    ]
  }

  // -------------------------------------------------------------------------
  // Lane-wise arithmetic
  // -------------------------------------------------------------------------

  /// Lane-wise addition (lazy — no carry propagation).
  ///
  /// Each 32-bit element is added independently. Limbs may exceed their
  /// nominal widths after this operation.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn add(&self, rhs: &Self) -> Self {
    Self([
      _mm256_add_epi32(self.0[0], rhs.0[0]),
      _mm256_add_epi32(self.0[1], rhs.0[1]),
      _mm256_add_epi32(self.0[2], rhs.0[2]),
      _mm256_add_epi32(self.0[3], rhs.0[3]),
      _mm256_add_epi32(self.0[4], rhs.0[4]),
    ])
  }

  /// Lane-wise subtraction with 2p bias.
  ///
  /// Computes `self + 2p - rhs` to ensure non-negative results. The bias
  /// is sufficient for reduced inputs (limbs within nominal widths).
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn sub(&self, rhs: &Self) -> Self {
    // 2p in radix 10×(26/25):
    // p = 2^255 - 19
    // Limb 0: 2 * ((1 << 26) - 19) = 2^27 - 38
    // Even limbs 2,4,6,8: 2 * ((1 << 26) - 1) = 2^27 - 2
    // Odd limbs 1,3,5,7,9: 2 * ((1 << 25) - 1) = 2^26 - 2
    let bias_0 = _mm256_setr_epi32(
      (2 * ((1i64 << 26) - 19)) as i32,
      (2 * ((1i64 << 26) - 19)) as i32,
      (2 * ((1i64 << 25) - 1)) as i32,
      (2 * ((1i64 << 25) - 1)) as i32,
      (2 * ((1i64 << 26) - 19)) as i32,
      (2 * ((1i64 << 26) - 19)) as i32,
      (2 * ((1i64 << 25) - 1)) as i32,
      (2 * ((1i64 << 25) - 1)) as i32,
    );
    let bias_n = _mm256_setr_epi32(
      (2 * ((1i64 << 26) - 1)) as i32,
      (2 * ((1i64 << 26) - 1)) as i32,
      (2 * ((1i64 << 25) - 1)) as i32,
      (2 * ((1i64 << 25) - 1)) as i32,
      (2 * ((1i64 << 26) - 1)) as i32,
      (2 * ((1i64 << 26) - 1)) as i32,
      (2 * ((1i64 << 25) - 1)) as i32,
      (2 * ((1i64 << 25) - 1)) as i32,
    );

    Self([
      _mm256_sub_epi32(_mm256_add_epi32(self.0[0], bias_0), rhs.0[0]),
      _mm256_sub_epi32(_mm256_add_epi32(self.0[1], bias_n), rhs.0[1]),
      _mm256_sub_epi32(_mm256_add_epi32(self.0[2], bias_n), rhs.0[2]),
      _mm256_sub_epi32(_mm256_add_epi32(self.0[3], bias_n), rhs.0[3]),
      _mm256_sub_epi32(_mm256_add_epi32(self.0[4], bias_n), rhs.0[4]),
    ])
  }

  /// Negate lanes lazily: compute `2p - self`.
  ///
  /// Precondition: `self` must have excess bits b < 1.0.
  /// Postcondition: result has b < 1.0.
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

  // -------------------------------------------------------------------------
  // Data movement
  // -------------------------------------------------------------------------

  /// Rearrange field element lanes according to the given pattern.
  ///
  /// Each `__m256i` is permuted at the 32-bit level using
  /// `_mm256_permutevar8x32_epi32`.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn shuffle(&self, pattern: Shuffle) -> Self {
    let ctrl = pattern.control();
    let c = _mm256_setr_epi32(ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4], ctrl[5], ctrl[6], ctrl[7]);
    Self([
      _mm256_permutevar8x32_epi32(self.0[0], c),
      _mm256_permutevar8x32_epi32(self.0[1], c),
      _mm256_permutevar8x32_epi32(self.0[2], c),
      _mm256_permutevar8x32_epi32(self.0[3], c),
      _mm256_permutevar8x32_epi32(self.0[4], c),
    ])
  }

  /// Select lanes from two sources using a blend mask.
  ///
  /// For each bit set in `lanes`, the corresponding 32-bit lane comes from
  /// `other`; otherwise from `self`.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn blend(&self, other: &Self, lanes: Lanes) -> Self {
    // _mm256_blend_epi32 requires a compile-time immediate, so dispatch
    // on the enum variant.
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
      Lanes::A => do_blend!(0b0000_0101),
      Lanes::B => do_blend!(0b0000_1010),
      Lanes::C => do_blend!(0b0101_0000),
      Lanes::D => do_blend!(0b1010_0000),
      Lanes::AB => do_blend!(0b0000_1111),
      Lanes::AC => do_blend!(0b0101_0101),
      Lanes::AD => do_blend!(0b1010_0101),
      Lanes::BC => do_blend!(0b0101_1010),
      Lanes::BCD => do_blend!(0b1111_1010),
      Lanes::CD => do_blend!(0b1111_0000),
      Lanes::ABCD => do_blend!(0b1111_1111),
    }
  }

  /// Compute `(B−A, A+B, D−C, C+D)` — the key building block for HWCD
  /// parallel point formulas.
  ///
  /// Precondition: b < 0.01. Postcondition: b < 1.6.
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

  // -------------------------------------------------------------------------
  // Carry reduction
  // -------------------------------------------------------------------------

  /// Full carry propagation across 10 limbs.
  ///
  /// Splits into two parallel carry chains (limbs 0-3 and 4-7) for
  /// instruction-level parallelism, then propagates the final carry from
  /// limb 9 back to limb 0 with the ×19 wraparound.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn reduce(&self) -> Self {
    // Unpack to 10 × u64x4 for carry propagation.
    let (z0, z1) = unpack_pair(self.0[0]);
    let (z2, z3) = unpack_pair(self.0[1]);
    let (z4, z5) = unpack_pair(self.0[2]);
    let (z6, z7) = unpack_pair(self.0[3]);
    let (z8, z9) = unpack_pair(self.0[4]);

    let mut z = [z0, z1, z2, z3, z4, z5, z6, z7, z8, z9];
    Self::reduce64(&mut z)
  }

  /// Carry-propagate 10 × u64x4 accumulators and repack to 5 × u32x8.
  ///
  /// Used after multiplication and squaring where accumulators are 64-bit.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  unsafe fn reduce64(z: &mut [__m256i; 10]) -> Self {
    let mask_26 = _mm256_set1_epi64x(LOW_26_BITS);
    let mask_25 = _mm256_set1_epi64x(LOW_25_BITS);
    let v19 = _mm256_set1_epi64x(19);

    macro_rules! carry_even {
      ($i:expr) => {
        let carry = _mm256_srli_epi64::<26>(z[$i]);
        z[$i] = _mm256_and_si256(z[$i], mask_26);
        z[$i + 1] = add64(z[$i + 1], carry);
      };
    }
    macro_rules! carry_odd {
      ($i:expr) => {
        let carry = _mm256_srli_epi64::<25>(z[$i]);
        z[$i] = _mm256_and_si256(z[$i], mask_25);
        z[$i + 1] = add64(z[$i + 1], carry);
      };
    }

    // Two parallel carry chains: 0→3 and 4→7
    carry_even!(0);
    carry_even!(4);
    carry_odd!(1);
    carry_odd!(5);
    carry_even!(2);
    carry_even!(6);
    carry_odd!(3);
    carry_odd!(7);

    // Second pass on 4 to flush carry from 3
    carry_even!(4);
    carry_even!(8);

    // Final carry from limb 9 with ×19 wraparound.
    // The carry from z[9] >> 25 can be large (up to ~2^39), exceeding the
    // 32-bit input range for a single vpmuludq. Split into two parts:
    //   c = z[9] >> 25
    //   c0 = c & LOW_26_BITS (fits 26 bits → safe for vpmuludq)
    //   c1 = c >> 26 (fits ~13 bits)
    //   z[0] += c0 * 19
    //   z[1] += c1 * 19
    let carry9 = _mm256_srli_epi64::<25>(z[9]);
    z[9] = _mm256_and_si256(z[9], mask_25);

    let c0 = _mm256_and_si256(carry9, mask_26);
    let c1 = _mm256_srli_epi64::<26>(carry9);

    z[0] = add64(z[0], _mm256_mul_epu32(c0, v19));
    z[1] = add64(z[1], _mm256_mul_epu32(c1, v19));

    // One final carry from z[0] to z[1]
    carry_even!(0);

    // Repack from 10 × u64x4 to 5 × u32x8
    Self([
      repack_pair(z[0], z[1]),
      repack_pair(z[2], z[3]),
      repack_pair(z[4], z[5]),
      repack_pair(z[6], z[7]),
      repack_pair(z[8], z[9]),
    ])
  }

  // -------------------------------------------------------------------------
  // Multiplication (4-way parallel schoolbook)
  // -------------------------------------------------------------------------

  /// Multiply two vectorized field elements (4 independent multiplications).
  ///
  /// Uses 100 `vpmuludq` instructions for the schoolbook product plus 9 for
  /// the ×19 pre-multiplications and 5 additions for pre-doubling odd terms.
  ///
  /// Precondition: one operand must have b < 2.5, the other b < 1.75.
  /// The b < 1.75 constraint ensures `19 * limb` fits in 32 bits.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn mul(&self, rhs: &Self) -> Self {
    let v19 = _mm256_set1_epi64x(19);

    // Unpack self into 10 �� u64x4 (zero-extended for vpmuludq)
    let (x0, x1) = unpack_pair(self.0[0]);
    let (x2, x3) = unpack_pair(self.0[1]);
    let (x4, x5) = unpack_pair(self.0[2]);
    let (x6, x7) = unpack_pair(self.0[3]);
    let (x8, x9) = unpack_pair(self.0[4]);

    // Pre-double odd-index x terms (for even z_k radix alignment)
    let x1_2 = add64(x1, x1);
    let x3_2 = add64(x3, x3);
    let x5_2 = add64(x5, x5);
    let x7_2 = add64(x7, x7);
    let x9_2 = add64(x9, x9);

    // Unpack rhs
    let (y0, y1) = unpack_pair(rhs.0[0]);
    let (y2, y3) = unpack_pair(rhs.0[1]);
    let (y4, y5) = unpack_pair(rhs.0[2]);
    let (y6, y7) = unpack_pair(rhs.0[3]);
    let (y8, y9) = unpack_pair(rhs.0[4]);

    // Pre-multiply rhs limbs by 19 for wrap-around reduction (2^255 ≡ 19 mod p)
    let y1_19 = mul32(y1, v19);
    let y2_19 = mul32(y2, v19);
    let y3_19 = mul32(y3, v19);
    let y4_19 = mul32(y4, v19);
    let y5_19 = mul32(y5, v19);
    let y6_19 = mul32(y6, v19);
    let y7_19 = mul32(y7, v19);
    let y8_19 = mul32(y8, v19);
    let y9_19 = mul32(y9, v19);

    // Schoolbook multiplication.
    // For even z_k: odd x indices use pre-doubled values (x_i_2).
    // For odd z_k: no doubling.
    // Terms with i+j >= 10 use y_j_19 (reduction mod p).

    let z0 = add64(
      add64(
        add64(mul32(x0, y0), mul32(x1_2, y9_19)),
        add64(mul32(x2, y8_19), mul32(x3_2, y7_19)),
      ),
      add64(
        add64(mul32(x4, y6_19), mul32(x5_2, y5_19)),
        add64(
          add64(mul32(x6, y4_19), mul32(x7_2, y3_19)),
          add64(mul32(x8, y2_19), mul32(x9_2, y1_19)),
        ),
      ),
    );

    let z1 = add64(
      add64(
        add64(mul32(x0, y1), mul32(x1, y0)),
        add64(mul32(x2, y9_19), mul32(x3, y8_19)),
      ),
      add64(
        add64(mul32(x4, y7_19), mul32(x5, y6_19)),
        add64(
          add64(mul32(x6, y5_19), mul32(x7, y4_19)),
          add64(mul32(x8, y3_19), mul32(x9, y2_19)),
        ),
      ),
    );

    let z2 = add64(
      add64(
        add64(mul32(x0, y2), mul32(x1_2, y1)),
        add64(mul32(x2, y0), mul32(x3_2, y9_19)),
      ),
      add64(
        add64(mul32(x4, y8_19), mul32(x5_2, y7_19)),
        add64(
          add64(mul32(x6, y6_19), mul32(x7_2, y5_19)),
          add64(mul32(x8, y4_19), mul32(x9_2, y3_19)),
        ),
      ),
    );

    let z3 = add64(
      add64(add64(mul32(x0, y3), mul32(x1, y2)), add64(mul32(x2, y1), mul32(x3, y0))),
      add64(
        add64(mul32(x4, y9_19), mul32(x5, y8_19)),
        add64(
          add64(mul32(x6, y7_19), mul32(x7, y6_19)),
          add64(mul32(x8, y5_19), mul32(x9, y4_19)),
        ),
      ),
    );

    let z4 = add64(
      add64(
        add64(mul32(x0, y4), mul32(x1_2, y3)),
        add64(mul32(x2, y2), mul32(x3_2, y1)),
      ),
      add64(
        add64(mul32(x4, y0), mul32(x5_2, y9_19)),
        add64(
          add64(mul32(x6, y8_19), mul32(x7_2, y7_19)),
          add64(mul32(x8, y6_19), mul32(x9_2, y5_19)),
        ),
      ),
    );

    let z5 = add64(
      add64(add64(mul32(x0, y5), mul32(x1, y4)), add64(mul32(x2, y3), mul32(x3, y2))),
      add64(
        add64(mul32(x4, y1), mul32(x5, y0)),
        add64(
          add64(mul32(x6, y9_19), mul32(x7, y8_19)),
          add64(mul32(x8, y7_19), mul32(x9, y6_19)),
        ),
      ),
    );

    let z6 = add64(
      add64(
        add64(mul32(x0, y6), mul32(x1_2, y5)),
        add64(mul32(x2, y4), mul32(x3_2, y3)),
      ),
      add64(
        add64(mul32(x4, y2), mul32(x5_2, y1)),
        add64(
          add64(mul32(x6, y0), mul32(x7_2, y9_19)),
          add64(mul32(x8, y8_19), mul32(x9_2, y7_19)),
        ),
      ),
    );

    let z7 = add64(
      add64(add64(mul32(x0, y7), mul32(x1, y6)), add64(mul32(x2, y5), mul32(x3, y4))),
      add64(
        add64(mul32(x4, y3), mul32(x5, y2)),
        add64(
          add64(mul32(x6, y1), mul32(x7, y0)),
          add64(mul32(x8, y9_19), mul32(x9, y8_19)),
        ),
      ),
    );

    let z8 = add64(
      add64(
        add64(mul32(x0, y8), mul32(x1_2, y7)),
        add64(mul32(x2, y6), mul32(x3_2, y5)),
      ),
      add64(
        add64(mul32(x4, y4), mul32(x5_2, y3)),
        add64(
          add64(mul32(x6, y2), mul32(x7_2, y1)),
          add64(mul32(x8, y0), mul32(x9_2, y9_19)),
        ),
      ),
    );

    let z9 = add64(
      add64(add64(mul32(x0, y9), mul32(x1, y8)), add64(mul32(x2, y7), mul32(x3, y6))),
      add64(
        add64(mul32(x4, y5), mul32(x5, y4)),
        add64(add64(mul32(x6, y3), mul32(x7, y2)), add64(mul32(x8, y1), mul32(x9, y0))),
      ),
    );

    let mut z = [z0, z1, z2, z3, z4, z5, z6, z7, z8, z9];
    Self::reduce64(&mut z)
  }

  // -------------------------------------------------------------------------
  // Squaring (with symmetry optimization)
  // -------------------------------------------------------------------------

  /// Compute the square accumulators in the u64 domain (before reduction).
  ///
  /// Returns 10 `u64x4` vectors ready for `reduce64`. Factored out so that
  /// `square()` and `square_and_negate_d()` can share the schoolbook.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  unsafe fn square_accum(&self) -> [__m256i; 10] {
    let v19 = _mm256_set1_epi64x(19);

    let (x0, x1) = unpack_pair(self.0[0]);
    let (x2, x3) = unpack_pair(self.0[1]);
    let (x4, x5) = unpack_pair(self.0[2]);
    let (x6, x7) = unpack_pair(self.0[3]);
    let (x8, x9) = unpack_pair(self.0[4]);

    // Pre-doubled terms for cross products (×2 from squaring symmetry)
    let x0_2 = add64(x0, x0);
    let x1_2 = add64(x1, x1);
    let x2_2 = add64(x2, x2);
    let x3_2 = add64(x3, x3);
    let x4_2 = add64(x4, x4);
    let x5_2 = add64(x5, x5);
    let x6_2 = add64(x6, x6);
    let x7_2 = add64(x7, x7);

    // Pre-multiply by 19 for wrap-around terms
    let x5_19 = mul32(x5, v19);
    let x6_19 = mul32(x6, v19);
    let x7_19 = mul32(x7, v19);
    let x8_19 = mul32(x8, v19);
    let x9_19 = mul32(x9, v19);

    // ×4 for odd×odd cross terms on even z_k:
    //   ×2 (symmetry: cross term appears twice)
    //   ×2 (radix alignment: 2^(25+25) vs 2^26 = 2^(26-1) overshoot)
    // When paired with y_19 terms, gives ×4·19 = ×76 total.
    let x1_4 = _mm256_slli_epi64::<2>(x1);
    let x3_4 = _mm256_slli_epi64::<2>(x3);
    let x5_4 = _mm256_slli_epi64::<2>(x5);
    let x7_4 = _mm256_slli_epi64::<2>(x7);

    // z0 (even): (0,0)×1 + (1,9)×76 + (2,8)×2·19 + (3,7)×76 + (4,6)×2·19 + (5,5)×2·19
    let z0 = add64(
      add64(
        add64(mul32(x0, x0), mul32(x1_4, x9_19)),
        add64(mul32(x2_2, x8_19), mul32(x3_4, x7_19)),
      ),
      add64(mul32(x4_2, x6_19), mul32(x5_2, x5_19)),
    );

    // z1 (odd): (0,1)×2 + (2,9)×2·19 + (3,8)×2·19 + (4,7)×2·19 + (5,6)×2·19
    let z1 = add64(
      add64(mul32(x0_2, x1), mul32(x2_2, x9_19)),
      add64(add64(mul32(x3_2, x8_19), mul32(x4_2, x7_19)), mul32(x5_2, x6_19)),
    );

    // z2 (even): (0,2)×2 + (1,1)×2 + (3,9)×76 + (4,8)×2·19 + (5,7)×76 + (6,6)×1·19
    let z2 = add64(
      add64(
        add64(mul32(x0_2, x2), mul32(x1_2, x1)),
        add64(mul32(x3_4, x9_19), mul32(x4_2, x8_19)),
      ),
      add64(mul32(x5_4, x7_19), mul32(x6, x6_19)),
    );

    // z3 (odd): (0,3)×2 + (1,2)×2 + (4,9)×2·19 + (5,8)×2·19 + (6,7)×2·19
    let z3 = add64(
      add64(mul32(x0_2, x3), mul32(x1_2, x2)),
      add64(add64(mul32(x4_2, x9_19), mul32(x5_2, x8_19)), mul32(x6_2, x7_19)),
    );

    // z4 (even): (0,4)×2 + (1,3)×4 + (2,2)×1 + (5,9)×76 + (6,8)×2·19 + (7,7)×2·19
    let z4 = add64(
      add64(
        add64(mul32(x0_2, x4), mul32(x1_4, x3)),
        add64(mul32(x2, x2), mul32(x5_4, x9_19)),
      ),
      add64(mul32(x6_2, x8_19), mul32(x7_2, x7_19)),
    );

    // z5 (odd): (0,5)×2 + (1,4)×2 + (2,3)×2 + (6,9)×2·19 + (7,8)×2·19
    let z5 = add64(
      add64(mul32(x0_2, x5), mul32(x1_2, x4)),
      add64(mul32(x2_2, x3), add64(mul32(x6_2, x9_19), mul32(x7_2, x8_19))),
    );

    // z6 (even): (0,6)×2 + (1,5)×4 + (2,4)×2 + (3,3)×2 + (7,9)×76 + (8,8)×1·19
    let z6 = add64(
      add64(
        add64(mul32(x0_2, x6), mul32(x1_4, x5)),
        add64(mul32(x2_2, x4), mul32(x3_2, x3)),
      ),
      add64(mul32(x7_4, x9_19), mul32(x8, x8_19)),
    );

    // z7 (odd): (0,7)×2 + (1,6)×2 + (2,5)×2 + (3,4)×2 + (8,9)×2·19
    let x8_2 = add64(x8, x8);
    let z7 = add64(
      add64(mul32(x0_2, x7), mul32(x1_2, x6)),
      add64(add64(mul32(x2_2, x5), mul32(x3_2, x4)), mul32(x8_2, x9_19)),
    );

    // z8 (even): (0,8)×2 + (1,7)×4 + (2,6)×2 + (3,5)×4 + (4,4)×1 + (9,9)×2·19
    let x9_2 = add64(x9, x9);
    let z8 = add64(
      add64(
        add64(mul32(x0_2, x8), mul32(x1_4, x7)),
        add64(mul32(x2_2, x6), mul32(x3_4, x5)),
      ),
      add64(mul32(x4, x4), mul32(x9_2, x9_19)),
    );

    // z9 (odd): (0,9)×2 + (1,8)×2 + (2,7)×2 + (3,6)×2 + (4,5)×2
    let z9 = add64(
      add64(mul32(x0_2, x9), mul32(x1_2, x8)),
      add64(add64(mul32(x2_2, x7), mul32(x3_2, x6)), mul32(x4_2, x5)),
    );

    [z0, z1, z2, z3, z4, z5, z6, z7, z8, z9]
  }

  /// Square the vectorized field element (4 independent squarings).
  ///
  /// Production code uses `square_and_negate_d()` directly; this standalone
  /// wrapper exists for differential testing.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[cfg(test)]
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn square(&self) -> Self {
    let mut z = self.square_accum();
    Self::reduce64(&mut z)
  }

  /// Square and negate the D lane in a single fused operation.
  ///
  /// Computes `(A², B², C², −D²)` — needed for the HWCD doubling formula
  /// where the negation keeps intermediate bounds within `vpmuludq`'s
  /// 32-bit input range.
  ///
  /// The D-lane negation is performed in the **u64 accumulator domain**
  /// before `reduce64`, using a `p × 2^37` bias. This ensures the negated
  /// D lane emerges fully reduced (b < 0.007), matching the other lanes.
  /// Without this, a post-reduce `2p − D` negation adds a ~2^27 bias that
  /// pushes subsequent doubling adds past the ×19 overflow threshold.
  ///
  /// # Safety
  ///
  /// Caller must ensure AVX2 is available.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  pub(crate) unsafe fn square_and_negate_d(&self) -> Self {
    let mut z = self.square_accum();
    Self::negate_d_accum(&mut z);
    Self::reduce64(&mut z)
  }

  /// Negate the D lane in each u64 accumulator using a `p × 2^37` bias.
  ///
  /// Replaces `z[k]_D` with `(p_k × 2^37) − z[k]_D` for each limb `k`.
  /// The bias is large enough to prevent underflow (max accumulator ≈ 2^60,
  /// min bias ≈ 2^62) and small enough to fit in u64. The subsequent
  /// `reduce64` carry chain processes the biased values normally, producing
  /// a fully reduced negation with b < 0.007.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn)]
  unsafe fn negate_d_accum(z: &mut [__m256i; 10]) {
    // p × 2^37 per limb (radix-26/25):
    let bias_even_0 = _mm256_set1_epi64x(((1i64 << 26) - 19) << 37);
    let bias_even = _mm256_set1_epi64x(((1i64 << 26) - 1) << 37);
    let bias_odd = _mm256_set1_epi64x(((1i64 << 25) - 1) << 37);

    // D lane = u64 lane 3 = u32 positions 6-7.
    const D_U64: i32 = 0b1100_0000;

    macro_rules! neg_d {
      ($idx:expr, $bias:expr) => {
        let negated = _mm256_sub_epi64($bias, z[$idx]);
        z[$idx] = _mm256_blend_epi32::<D_U64>(z[$idx], negated);
      };
    }

    neg_d!(0, bias_even_0);
    neg_d!(1, bias_odd);
    neg_d!(2, bias_even);
    neg_d!(3, bias_odd);
    neg_d!(4, bias_even);
    neg_d!(5, bias_odd);
    neg_d!(6, bias_even);
    neg_d!(7, bias_odd);
    neg_d!(8, bias_even);
    neg_d!(9, bias_odd);
  }

  /// Negate the D lane of each limb via `2p − D` in the packed u32 domain.
  ///
  /// **Not used by AVX2 `square_and_negate_d`** (which negates in u64 domain
  /// for tighter bounds). Retained for potential external callers.
  #[inline]
  #[target_feature(enable = "avx2")]
  #[allow(unsafe_op_in_unsafe_fn, dead_code)]
  unsafe fn negate_d_lane(fe: &mut Self) {
    let p2_limb0_even = (2i64.wrapping_mul((1i64 << 26) - 19)) as i32;
    let p2_limb_even = (2i64.wrapping_mul((1i64 << 26) - 1)) as i32;
    let p2_limb_odd = (2i64.wrapping_mul((1i64 << 25) - 1)) as i32;

    let bias_0 = _mm256_setr_epi32(0, 0, 0, 0, 0, p2_limb0_even, 0, p2_limb_odd);
    let bias_n = _mm256_setr_epi32(0, 0, 0, 0, 0, p2_limb_even, 0, p2_limb_odd);

    let neg0 = _mm256_sub_epi32(bias_0, fe.0[0]);
    fe.0[0] = _mm256_blend_epi32::<D_BLEND>(fe.0[0], neg0);

    let neg1 = _mm256_sub_epi32(bias_n, fe.0[1]);
    fe.0[1] = _mm256_blend_epi32::<D_BLEND>(fe.0[1], neg1);

    let neg2 = _mm256_sub_epi32(bias_n, fe.0[2]);
    fe.0[2] = _mm256_blend_epi32::<D_BLEND>(fe.0[2], neg2);

    let neg3 = _mm256_sub_epi32(bias_n, fe.0[3]);
    fe.0[3] = _mm256_blend_epi32::<D_BLEND>(fe.0[3], neg3);

    let neg4 = _mm256_sub_epi32(bias_n, fe.0[4]);
    fe.0[4] = _mm256_blend_epi32::<D_BLEND>(fe.0[4], neg4);
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
  use super::{FieldElement, *};

  fn test_field_elements() -> [FieldElement; 4] {
    let a = FieldElement::from_limbs([
      1_234_567_890_123,
      987_654_321_012,
      111_222_333_444,
      555_666_777_888,
      999_000_111_222,
    ]);
    let b = FieldElement::from_limbs([
      2_111_222_333_444,
      1_555_666_777_888,
      333_444_555_666,
      777_888_999_000,
      100_200_300_400,
    ]);
    let c = FieldElement::from_limbs([
      42_000_000_001,
      123_456_789_012,
      2_000_000_000_000,
      1_500_000_000_000,
      750_000_000_000,
    ]);
    let d = FieldElement::from_limbs([
      1_999_999_999_999,
      888_777_666_555,
      444_333_222_111,
      1_111_222_333_444,
      2_222_111_000_999,
    ]);
    [a, b, c, d]
  }

  fn small_field_elements() -> [FieldElement; 4] {
    [
      FieldElement::from_limbs([100, 200, 300, 400, 500]),
      FieldElement::from_limbs([600, 700, 800, 900, 1000]),
      FieldElement::from_limbs([1100, 1200, 1300, 1400, 1500]),
      FieldElement::from_limbs([1600, 1700, 1800, 1900, 2000]),
    ]
  }

  #[test]
  fn pack_unpack_roundtrip() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    let [a, b, c, d] = test_field_elements();

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let packed = FieldElement2625x4::new(&a, &b, &c, &d);
      let [ra, rb, rc, rd] = packed.split();

      assert_eq!(ra.normalize(), a.normalize(), "A roundtrip failed");
      assert_eq!(rb.normalize(), b.normalize(), "B roundtrip failed");
      assert_eq!(rc.normalize(), c.normalize(), "C roundtrip failed");
      assert_eq!(rd.normalize(), d.normalize(), "D roundtrip failed");
    }
  }

  #[test]
  fn add_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    let [a, b, c, d] = test_field_elements();
    let [e, f, g, h] = small_field_elements();

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let lhs = FieldElement2625x4::new(&a, &b, &c, &d);
      let rhs = FieldElement2625x4::new(&e, &f, &g, &h);
      let sum = lhs.add(&rhs).reduce();
      let [ra, rb, rc, rd] = sum.split();

      assert_eq!(ra.normalize(), a.add(&e).normalize(), "A add failed");
      assert_eq!(rb.normalize(), b.add(&f).normalize(), "B add failed");
      assert_eq!(rc.normalize(), c.add(&g).normalize(), "C add failed");
      assert_eq!(rd.normalize(), d.add(&h).normalize(), "D add failed");
    }
  }

  #[test]
  fn sub_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    let [a, b, c, d] = test_field_elements();
    let [e, f, g, h] = small_field_elements();

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let lhs = FieldElement2625x4::new(&a, &b, &c, &d);
      let rhs = FieldElement2625x4::new(&e, &f, &g, &h);
      let diff = lhs.sub(&rhs).reduce();
      let [ra, rb, rc, rd] = diff.split();

      assert_eq!(ra.normalize(), a.sub(&e).normalize(), "A sub failed");
      assert_eq!(rb.normalize(), b.sub(&f).normalize(), "B sub failed");
      assert_eq!(rc.normalize(), c.sub(&g).normalize(), "C sub failed");
      assert_eq!(rd.normalize(), d.sub(&h).normalize(), "D sub failed");
    }
  }

  #[test]
  fn mul_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    let [a, b, c, d] = test_field_elements();
    let [e, f, g, h] = small_field_elements();

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let lhs = FieldElement2625x4::new(&a, &b, &c, &d);
      let rhs = FieldElement2625x4::new(&e, &f, &g, &h);
      let product = lhs.mul(&rhs);
      let [ra, rb, rc, rd] = product.split();

      assert_eq!(ra.normalize(), a.mul(&e).normalize(), "A mul failed");
      assert_eq!(rb.normalize(), b.mul(&f).normalize(), "B mul failed");
      assert_eq!(rc.normalize(), c.mul(&g).normalize(), "C mul failed");
      assert_eq!(rd.normalize(), d.mul(&h).normalize(), "D mul failed");
    }
  }

  #[test]
  fn square_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    let [a, b, c, d] = test_field_elements();

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let packed = FieldElement2625x4::new(&a, &b, &c, &d);
      let squared = packed.square();
      let [ra, rb, rc, rd] = squared.split();

      assert_eq!(ra.normalize(), a.square().normalize(), "A square failed");
      assert_eq!(rb.normalize(), b.square().normalize(), "B square failed");
      assert_eq!(rc.normalize(), c.square().normalize(), "C square failed");
      assert_eq!(rd.normalize(), d.square().normalize(), "D square failed");
    }
  }

  #[test]
  fn square_matches_mul_self() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    let [a, b, c, d] = test_field_elements();

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let packed = FieldElement2625x4::new(&a, &b, &c, &d);
      let sq = packed.square();
      let mm = packed.mul(&packed);
      let [sa, sb, sc, sd] = sq.split();
      let [ma, mb, mc, md] = mm.split();

      assert_eq!(sa.normalize(), ma.normalize(), "A square vs mul mismatch");
      assert_eq!(sb.normalize(), mb.normalize(), "B square vs mul mismatch");
      assert_eq!(sc.normalize(), mc.normalize(), "C square vs mul mismatch");
      assert_eq!(sd.normalize(), md.normalize(), "D square vs mul mismatch");
    }
  }

  #[test]
  fn square_and_negate_d_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    let [a, b, c, d] = test_field_elements();

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let packed = FieldElement2625x4::new(&a, &b, &c, &d);
      let result = packed.square_and_negate_d();
      let [ra, rb, rc, rd] = result.split();

      assert_eq!(ra.normalize(), a.square().normalize(), "A square_neg_d failed");
      assert_eq!(rb.normalize(), b.square().normalize(), "B square_neg_d failed");
      assert_eq!(rc.normalize(), c.square().normalize(), "C square_neg_d failed");
      assert_eq!(rd.normalize(), d.square().neg().normalize(), "D should be negated");
    }
  }

  #[test]
  fn shuffle_badc_swaps_pairs() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    let [a, b, c, d] = test_field_elements();

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let packed = FieldElement2625x4::new(&a, &b, &c, &d);
      let shuffled = packed.shuffle(Shuffle::BADC);
      let [ra, rb, rc, rd] = shuffled.split();

      assert_eq!(ra.normalize(), b.normalize(), "BADC: A should be B");
      assert_eq!(rb.normalize(), a.normalize(), "BADC: B should be A");
      assert_eq!(rc.normalize(), d.normalize(), "BADC: C should be D");
      assert_eq!(rd.normalize(), c.normalize(), "BADC: D should be C");
    }
  }

  #[test]
  fn diff_sum_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    let [a, b, c, d] = small_field_elements();

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let packed = FieldElement2625x4::new(&a, &b, &c, &d);
      let ds = packed.diff_sum().reduce();
      let [r0, r1, r2, r3] = ds.split();

      assert_eq!(r0.normalize(), b.sub(&a).normalize(), "diff_sum[0] = B-A");
      assert_eq!(r1.normalize(), a.add(&b).normalize(), "diff_sum[1] = A+B");
      assert_eq!(r2.normalize(), d.sub(&c).normalize(), "diff_sum[2] = D-C");
      assert_eq!(r3.normalize(), c.add(&d).normalize(), "diff_sum[3] = C+D");
    }
  }

  #[test]
  fn blend_ab_cd() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    let [a, b, c, d] = test_field_elements();
    let [e, f, g, h] = [
      FieldElement::from_limbs([1, 2, 3, 4, 5]),
      FieldElement::from_limbs([6, 7, 8, 9, 10]),
      FieldElement::from_limbs([11, 12, 13, 14, 15]),
      FieldElement::from_limbs([16, 17, 18, 19, 20]),
    ];

    // SAFETY: AVX2 availability checked by the runtime guard above.
    unsafe {
      let lhs = FieldElement2625x4::new(&a, &b, &c, &d);
      let rhs = FieldElement2625x4::new(&e, &f, &g, &h);

      let blended = lhs.blend(&rhs, Lanes::AB);
      let [ra, rb, rc, rd] = blended.split();

      assert_eq!(ra.normalize(), e.normalize(), "AB blend: A should be from rhs");
      assert_eq!(rb.normalize(), f.normalize(), "AB blend: B should be from rhs");
      assert_eq!(rc.normalize(), c.normalize(), "AB blend: C should be from self");
      assert_eq!(rd.normalize(), d.normalize(), "AB blend: D should be from self");
    }
  }

  // Cross-validation: IFMA (radix-51) mul/square match AVX2 (radix-26/25).

  #[test]
  fn ifma_mul_matches_avx2() {
    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("avx512ifma") {
      return;
    }
    let [a, b, c, d] = test_field_elements();
    let [e, f, g, h] = small_field_elements();

    // SAFETY: AVX2 + IFMA availability checked by the runtime guard above.
    unsafe {
      use super::super::field_ifma::FieldElement51x4;

      let avx2_lhs = FieldElement2625x4::new(&a, &b, &c, &d);
      let avx2_rhs = FieldElement2625x4::new(&e, &f, &g, &h);
      let avx2_result = avx2_lhs.mul(&avx2_rhs);
      let [a2, b2, c2, d2] = avx2_result.split();

      let ifma_lhs = FieldElement51x4::new(&a, &b, &c, &d);
      let ifma_rhs = FieldElement51x4::new(&e, &f, &g, &h);
      let ifma_result = ifma_lhs.mul(&ifma_rhs);
      let [ai, bi, ci, di] = ifma_result.split();

      assert_eq!(a2.normalize(), ai.normalize(), "IFMA mul A mismatch");
      assert_eq!(b2.normalize(), bi.normalize(), "IFMA mul B mismatch");
      assert_eq!(c2.normalize(), ci.normalize(), "IFMA mul C mismatch");
      assert_eq!(d2.normalize(), di.normalize(), "IFMA mul D mismatch");
    }
  }

  #[test]
  fn ifma_square_matches_avx2() {
    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("avx512ifma") {
      return;
    }
    let [a, b, c, d] = test_field_elements();

    // SAFETY: AVX2 + IFMA availability checked by the runtime guard above.
    unsafe {
      use super::super::field_ifma::FieldElement51x4;

      let avx2_packed = FieldElement2625x4::new(&a, &b, &c, &d);
      let avx2_result = avx2_packed.square();
      let [a2, b2, c2, d2] = avx2_result.split();

      let ifma_packed = FieldElement51x4::new(&a, &b, &c, &d);
      let ifma_result = ifma_packed.square();
      let [ai, bi, ci, di] = ifma_result.split();

      assert_eq!(a2.normalize(), ai.normalize(), "IFMA square A mismatch");
      assert_eq!(b2.normalize(), bi.normalize(), "IFMA square B mismatch");
      assert_eq!(c2.normalize(), ci.normalize(), "IFMA square C mismatch");
      assert_eq!(d2.normalize(), di.normalize(), "IFMA square D mismatch");
    }
  }
}
