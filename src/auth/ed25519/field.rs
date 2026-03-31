//! Internal Ed25519 field arithmetic over `2^255 - 19`.
//!
//! This is the portable 5×51 radix baseline. All point arithmetic sits on
//! top of these operations.
//!
//! # Arithmetic convention
//!
//! Field arithmetic is modular math (mod 2²⁵⁵ − 19). Per CLAUDE.md rules,
//! `wrapping_*` is the correct choice for intentional modular arithmetic.
//! Intermediate u128 accumulators are sized so that overflow is provably
//! impossible — wrapping semantics are used for consistency, not because
//! wrap-around actually occurs.
//!
//! # Lazy reduction
//!
//! `add` is lazy: no carry propagation, limbs may exceed 51 bits.
//! `sub` uses a 16p bias and a single carry-propagation round.
//! `mul` and `square` reduce via two carry-propagation rounds on u128
//! accumulators. This matches the dalek/ref10 strategy and keeps the
//! hot-path field operations as cheap as possible.

use core::fmt;

use super::constants::FIELD_LIMBS;

const RADIX_BITS: u32 = 51;
const MASK51: u64 = (1u64 << RADIX_BITS) - 1;
const MODULUS_0: u64 = (1u64 << RADIX_BITS) - 19;
const MODULUS_N: u64 = MASK51;

/// Subtraction bias: 16p in radix-51. Large enough that `self + bias − rhs`
/// never underflows even when limbs are up to ~55 bits from lazy addition
/// chains.
const SUB_BIAS_0: u64 = ((1u64 << RADIX_BITS) - 19).wrapping_mul(16);
const SUB_BIAS_N: u64 = MASK51.wrapping_mul(16);
const SQRT_M1: FieldElement = FieldElement::from_limbs([
  1_718_705_420_411_056,
  234_908_883_556_509,
  2_233_514_472_574_048,
  2_117_202_627_021_982,
  765_476_049_583_133,
]);

/// Internal field element representation for arithmetic mod `2^255 - 19`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FieldElement([u64; FIELD_LIMBS]);

impl FieldElement {
  /// Additive identity.
  pub(crate) const ZERO: Self = Self([0, 0, 0, 0, 0]);

  /// Multiplicative identity.
  pub(crate) const ONE: Self = Self([1, 0, 0, 0, 0]);

  /// Construct a field element from raw radix-51 limbs.
  #[inline]
  #[must_use]
  pub(crate) const fn from_limbs(limbs: [u64; FIELD_LIMBS]) -> Self {
    Self(limbs)
  }

  /// Construct a small field element without reduction.
  #[inline]
  #[must_use]
  pub(crate) const fn from_small(value: u64) -> Self {
    Self([value, 0, 0, 0, 0])
  }

  /// Borrow the raw radix-51 limbs.
  #[inline]
  #[must_use]
  pub(crate) const fn limbs(&self) -> &[u64; FIELD_LIMBS] {
    &self.0
  }

  /// Add two field elements (lazy — no carry propagation).
  #[inline]
  #[must_use]
  pub(crate) fn add(&self, rhs: &Self) -> Self {
    Self([
      self.0[0].wrapping_add(rhs.0[0]),
      self.0[1].wrapping_add(rhs.0[1]),
      self.0[2].wrapping_add(rhs.0[2]),
      self.0[3].wrapping_add(rhs.0[3]),
      self.0[4].wrapping_add(rhs.0[4]),
    ])
  }

  /// Subtract two field elements modulo `2^255 - 19`.
  ///
  /// Adds a 16p bias to guarantee non-negative intermediates, then
  /// propagates carries once to keep limbs bounded.
  #[inline]
  #[must_use]
  pub(crate) fn sub(&self, rhs: &Self) -> Self {
    Self(carry_propagate([
      self.0[0].wrapping_add(SUB_BIAS_0).wrapping_sub(rhs.0[0]),
      self.0[1].wrapping_add(SUB_BIAS_N).wrapping_sub(rhs.0[1]),
      self.0[2].wrapping_add(SUB_BIAS_N).wrapping_sub(rhs.0[2]),
      self.0[3].wrapping_add(SUB_BIAS_N).wrapping_sub(rhs.0[3]),
      self.0[4].wrapping_add(SUB_BIAS_N).wrapping_sub(rhs.0[4]),
    ]))
  }

  /// Multiply two field elements.
  ///
  /// Precomputes `g[i] * 19` at the u64 level so that the reduction trick
  /// `2^255 ≡ 19 (mod p)` uses a single u64×u64→u128 widening multiply
  /// instead of a u128×u128 triple multiply.
  #[inline]
  #[must_use]
  pub(crate) fn mul(&self, rhs: &Self) -> Self {
    let [f0, f1, f2, f3, f4] = self.0;
    let [g0, g1, g2, g3, g4] = rhs.0;

    let g1_19 = g1.wrapping_mul(19);
    let g2_19 = g2.wrapping_mul(19);
    let g3_19 = g3.wrapping_mul(19);
    let g4_19 = g4.wrapping_mul(19);

    let h0 = m(f0, g0).wrapping_add(m(f1, g4_19)).wrapping_add(m(f2, g3_19)).wrapping_add(m(f3, g2_19)).wrapping_add(m(f4, g1_19));
    let h1 = m(f0, g1).wrapping_add(m(f1, g0)).wrapping_add(m(f2, g4_19)).wrapping_add(m(f3, g3_19)).wrapping_add(m(f4, g2_19));
    let h2 = m(f0, g2).wrapping_add(m(f1, g1)).wrapping_add(m(f2, g0)).wrapping_add(m(f3, g4_19)).wrapping_add(m(f4, g3_19));
    let h3 = m(f0, g3).wrapping_add(m(f1, g2)).wrapping_add(m(f2, g1)).wrapping_add(m(f3, g0)).wrapping_add(m(f4, g4_19));
    let h4 = m(f0, g4).wrapping_add(m(f1, g3)).wrapping_add(m(f2, g2)).wrapping_add(m(f3, g1)).wrapping_add(m(f4, g0));

    Self(reduce_wide([h0, h1, h2, h3, h4]))
  }

  /// Square a field element.
  ///
  /// Dedicated formula exploiting `f[i]*f[j] == f[j]*f[i]` symmetry:
  /// 15 wide multiplies instead of 25 in the general `mul` path.
  #[inline]
  #[must_use]
  pub(crate) fn square(&self) -> Self {
    let [f0, f1, f2, f3, f4] = self.0;

    let f0_2 = f0.wrapping_mul(2);
    let f1_2 = f1.wrapping_mul(2);
    let f1_38 = f1.wrapping_mul(38);
    let f2_38 = f2.wrapping_mul(38);
    let f3_38 = f3.wrapping_mul(38);
    let f3_19 = f3.wrapping_mul(19);
    let f4_19 = f4.wrapping_mul(19);

    let h0 = m(f0, f0).wrapping_add(m(f1_38, f4)).wrapping_add(m(f2_38, f3));
    let h1 = m(f0_2, f1).wrapping_add(m(f2_38, f4)).wrapping_add(m(f3_19, f3));
    let h2 = m(f0_2, f2).wrapping_add(m(f1, f1)).wrapping_add(m(f3_38, f4));
    let h3 = m(f0_2, f3).wrapping_add(m(f1_2, f2)).wrapping_add(m(f4_19, f4));
    let h4 = m(f0_2, f4).wrapping_add(m(f1_2, f3)).wrapping_add(m(f2, f2));

    Self(reduce_wide([h0, h1, h2, h3, h4]))
  }

  /// Negate the field element modulo `2^255 - 19`.
  #[inline]
  #[must_use]
  pub(crate) fn neg(&self) -> Self {
    Self::ZERO.sub(self)
  }

  /// Multiplicative inverse using exponentiation by `p - 2`.
  #[must_use]
  pub(crate) fn invert(&self) -> Self {
    let (t19, t3) = self.pow22501();
    t19.pow2k(5).mul(&t3)
  }

  /// Canonically reduce the field element.
  #[inline]
  #[must_use]
  pub(crate) fn normalize(&self) -> Self {
    // Two carry rounds to fully reduce potentially wide limbs from lazy
    // add chains, then conditional subtraction of the modulus.
    let reduced = carry_propagate(carry_propagate(self.0));
    let candidate = subtract_modulus(reduced);
    let use_candidate = u64::MAX.wrapping_mul(u64::from(candidate.1));
    let keep_reduced = !use_candidate;
    let [r0, r1, r2, r3, r4] = reduced;
    let [c0, c1, c2, c3, c4] = candidate.0;

    Self([
      (c0 & use_candidate) | (r0 & keep_reduced),
      (c1 & use_candidate) | (r1 & keep_reduced),
      (c2 & use_candidate) | (r2 & keep_reduced),
      (c3 & use_candidate) | (r3 & keep_reduced),
      (c4 & use_candidate) | (r4 & keep_reduced),
    ])
  }

  /// Decode a canonical 32-byte field element.
  #[must_use]
  pub(crate) fn from_bytes(bytes: &[u8; 32]) -> Option<Self> {
    let mut acc = 0u128;
    let mut acc_bits = 0u32;
    let mut byte_iter = bytes.iter().copied();
    let mut limbs = [0u64; FIELD_LIMBS];

    for limb in limbs.iter_mut() {
      while acc_bits < RADIX_BITS {
        if let Some(byte) = byte_iter.next() {
          acc |= u128::from(byte) << acc_bits;
          acc_bits = acc_bits.wrapping_add(8);
        } else {
          break;
        }
      }

      *limb = (acc & u128::from(MASK51)) as u64;
      acc >>= RADIX_BITS;
      acc_bits = acc_bits.wrapping_sub(RADIX_BITS);
    }

    if acc != 0 || byte_iter.next().is_some() {
      return None;
    }

    let element = Self(limbs).normalize();
    if element.to_bytes() == *bytes {
      Some(element)
    } else {
      None
    }
  }

  /// Encode the field element canonically.
  #[must_use]
  pub(crate) fn to_bytes(self) -> [u8; 32] {
    let canonical = self.normalize();
    let mut out = [0u8; 32];
    let mut acc = 0u128;
    let mut acc_bits = 0u32;
    let mut out_iter = out.iter_mut();

    for &limb in canonical.0.iter() {
      acc |= u128::from(limb) << acc_bits;
      acc_bits = acc_bits.wrapping_add(RADIX_BITS);

      while acc_bits >= 8 {
        if let Some(byte) = out_iter.next() {
          *byte = acc as u8;
        }
        acc >>= 8;
        acc_bits = acc_bits.wrapping_sub(8);
      }
    }

    if let Some(byte) = out_iter.next() {
      *byte = acc as u8;
    }

    out
  }

  /// Return `true` when the canonical field element is zero.
  #[must_use]
  pub(crate) fn is_zero(&self) -> bool {
    self.normalize().0.iter().all(|&limb| limb == 0)
  }

  /// Return the low-bit sign of the canonical encoding.
  #[must_use]
  pub(crate) fn is_negative(&self) -> bool {
    ((*self).to_bytes()[0] & 1) == 1
  }

  /// Square root in `GF(2^255 - 19)` when one exists.
  #[must_use]
  pub(crate) fn sqrt(&self) -> Option<Self> {
    self.sqrt_ratio_i(&Self::ONE).and_then(|candidate| {
      if candidate.square().normalize() == self.normalize() {
        Some(candidate)
      } else {
        None
      }
    })
  }

  #[must_use]
  fn pow2k(&self, mut k: u32) -> Self {
    debug_assert!(k > 0);

    let mut acc = *self;
    while k > 0 {
      acc = acc.square();
      k = k.wrapping_sub(1);
    }
    acc
  }

  #[must_use]
  fn pow22501(&self) -> (Self, Self) {
    let t0 = self.square();
    let t1 = t0.square().square();
    let t2 = self.mul(&t1);
    let t3 = t0.mul(&t2);
    let t4 = t3.square();
    let t5 = t2.mul(&t4);
    let t6 = t5.pow2k(5);
    let t7 = t6.mul(&t5);
    let t8 = t7.pow2k(10);
    let t9 = t8.mul(&t7);
    let t10 = t9.pow2k(20);
    let t11 = t10.mul(&t9);
    let t12 = t11.pow2k(10);
    let t13 = t12.mul(&t7);
    let t14 = t13.pow2k(50);
    let t15 = t14.mul(&t13);
    let t16 = t15.pow2k(100);
    let t17 = t16.mul(&t15);
    let t18 = t17.pow2k(50);
    let t19 = t18.mul(&t13);

    (t19, t3)
  }

  #[must_use]
  fn pow_p58(&self) -> Self {
    let (t19, _) = self.pow22501();
    self.mul(&t19.pow2k(2))
  }

  #[must_use]
  pub(crate) fn sqrt_ratio_i(&self, denominator: &Self) -> Option<Self> {
    let numerator = self.normalize();
    let denominator = denominator.normalize();
    let v3 = denominator.square().mul(&denominator);
    let v7 = v3.square().mul(&denominator);
    let r = numerator.mul(&v3).mul(&numerator.mul(&v7).pow_p58());
    let check = denominator.mul(&r.square()).normalize();

    if check == numerator {
      Some(if r.is_negative() { r.neg() } else { r })
    } else if check == numerator.neg().normalize() {
      let adjusted = r.mul(&SQRT_M1).normalize();
      Some(if adjusted.is_negative() {
        adjusted.neg()
      } else {
        adjusted
      })
    } else {
      None
    }
  }
}

impl Default for FieldElement {
  #[inline]
  fn default() -> Self {
    Self::ZERO
  }
}

impl fmt::Debug for FieldElement {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_tuple("FieldElement").field(&self.normalize().0).finish()
  }
}

/// Widening multiply: u64 × u64 → u128.
#[inline(always)]
#[must_use]
fn m(a: u64, b: u64) -> u128 {
  (a as u128).wrapping_mul(b as u128)
}

/// Two-round carry propagation on u128 accumulators from mul/square.
#[inline]
#[must_use]
fn reduce_wide(wide: [u128; FIELD_LIMBS]) -> [u64; FIELD_LIMBS] {
  let [mut h0, mut h1, mut h2, mut h3, mut h4] = wide;
  let mask = u128::from(MASK51);

  // Round 1: the wrap-around carry (h4 → h0 × 19) may leave h0 wide.
  h1 = h1.wrapping_add(h0 >> RADIX_BITS); h0 &= mask;
  h2 = h2.wrapping_add(h1 >> RADIX_BITS); h1 &= mask;
  h3 = h3.wrapping_add(h2 >> RADIX_BITS); h2 &= mask;
  h4 = h4.wrapping_add(h3 >> RADIX_BITS); h3 &= mask;
  h0 = h0.wrapping_add((h4 >> RADIX_BITS).wrapping_mul(19)); h4 &= mask;

  // Round 2: flush the residual carry from round 1.
  h1 = h1.wrapping_add(h0 >> RADIX_BITS); h0 &= mask;
  h2 = h2.wrapping_add(h1 >> RADIX_BITS); h1 &= mask;
  h3 = h3.wrapping_add(h2 >> RADIX_BITS); h2 &= mask;
  h4 = h4.wrapping_add(h3 >> RADIX_BITS); h3 &= mask;
  h0 = h0.wrapping_add((h4 >> RADIX_BITS).wrapping_mul(19)); h4 &= mask;

  [h0 as u64, h1 as u64, h2 as u64, h3 as u64, h4 as u64]
}

/// Single-round carry propagation on u64 limbs (sub output, normalize).
#[inline]
#[must_use]
fn carry_propagate(limbs: [u64; FIELD_LIMBS]) -> [u64; FIELD_LIMBS] {
  let [mut l0, mut l1, mut l2, mut l3, mut l4] = limbs;

  l1 = l1.wrapping_add(l0 >> RADIX_BITS); l0 &= MASK51;
  l2 = l2.wrapping_add(l1 >> RADIX_BITS); l1 &= MASK51;
  l3 = l3.wrapping_add(l2 >> RADIX_BITS); l2 &= MASK51;
  l4 = l4.wrapping_add(l3 >> RADIX_BITS); l3 &= MASK51;
  l0 = l0.wrapping_add((l4 >> RADIX_BITS).wrapping_mul(19)); l4 &= MASK51;

  [l0, l1, l2, l3, l4]
}

#[inline]
#[must_use]
fn subtract_modulus(limbs: [u64; FIELD_LIMBS]) -> ([u64; FIELD_LIMBS], bool) {
  let [f0, f1, f2, f3, f4] = limbs;
  let (s0, borrow0) = sub_limb(f0, MODULUS_0, false);
  let (s1, borrow1) = sub_limb(f1, MODULUS_N, borrow0);
  let (s2, borrow2) = sub_limb(f2, MODULUS_N, borrow1);
  let (s3, borrow3) = sub_limb(f3, MODULUS_N, borrow2);
  let (s4, borrow4) = sub_limb(f4, MODULUS_N, borrow3);

  ([s0, s1, s2, s3, s4], !borrow4)
}

#[inline]
#[must_use]
fn sub_limb(lhs: u64, rhs: u64, borrow: bool) -> (u64, bool) {
  let subtrahend = u128::from(rhs).wrapping_add(u128::from(u8::from(borrow)));
  let lhs_wide = u128::from(lhs);

  if lhs_wide >= subtrahend {
    ((lhs_wide.wrapping_sub(subtrahend)) as u64, false)
  } else {
    ((lhs_wide.wrapping_add(u128::from(1u64 << RADIX_BITS)).wrapping_sub(subtrahend)) as u64, true)
  }
}

#[cfg(test)]
mod tests {
  use super::{FieldElement, MASK51};

  fn from_u128(mut value: u128) -> FieldElement {
    let mut limbs = [0u64; 5];
    for limb in &mut limbs {
      *limb = (value & u128::from(MASK51)) as u64;
      value >>= 51;
    }
    FieldElement::from_limbs(limbs)
  }

  fn bytes_from_u128(value: u128) -> [u8; 32] {
    let mut out = [0u8; 32];
    let raw = value.to_le_bytes();
    out[..raw.len()].copy_from_slice(&raw);
    out
  }

  #[test]
  fn add_carries_across_radix51_boundary() {
    let lhs = FieldElement::from_limbs([MASK51, 0, 0, 0, 0]);
    let rhs = FieldElement::ONE;

    assert_eq!(lhs.add(&rhs).normalize().limbs(), &[0, 1, 0, 0, 0]);
  }

  #[test]
  fn modulus_minus_one_wraps_with_one() {
    let modulus_minus_one = FieldElement::from_limbs([MASK51 - 19, MASK51, MASK51, MASK51, MASK51]);

    assert_eq!(
      modulus_minus_one.add(&FieldElement::ONE).to_bytes(),
      FieldElement::ZERO.to_bytes()
    );
  }

  #[test]
  fn multiply_and_square_match_for_small_values() {
    let value = from_u128(9_876_543_210);

    assert_eq!(value.square(), value.mul(&value));
  }

  #[test]
  fn small_arithmetic_matches_u128_reference() {
    let lhs = from_u128(1_234_567_890_123);
    let rhs = from_u128(9_876_543_210);

    assert_eq!(lhs.add(&rhs).to_bytes(), bytes_from_u128(1_244_444_433_333));
    assert_eq!(lhs.sub(&rhs).to_bytes(), bytes_from_u128(1_224_691_346_913));
    assert_eq!(
      lhs.mul(&rhs).to_bytes(),
      bytes_from_u128(12_193_263_112_478_341_714_830)
    );
  }

  #[test]
  fn canonical_roundtrip_preserves_small_values() {
    let element = from_u128(0x0102_0304_0506_0708_1112_1314_1516_1718);
    let bytes = element.to_bytes();

    assert_eq!(FieldElement::from_bytes(&bytes), Some(element.normalize()));
  }

  #[test]
  fn from_bytes_rejects_non_canonical_inputs() {
    let modulus = [
      0xED, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F,
    ];
    let top_bit_set = [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x80,
    ];

    assert_eq!(FieldElement::from_bytes(&modulus), None);
    assert_eq!(FieldElement::from_bytes(&top_bit_set), None);
  }

  #[test]
  fn inversion_roundtrips_small_nonzero_values() {
    let value = from_u128(123_456_789);
    let identity = value.mul(&value.invert()).normalize();

    assert_eq!(identity, FieldElement::ONE);
  }

  #[test]
  fn negation_cancels_addition() {
    let value = from_u128(42_424_242);

    assert!(value.add(&value.neg()).is_zero());
  }

  #[test]
  fn sqrt_recovers_square() {
    let value = from_u128(7_654_321);
    let square = value.square();
    let root = square.sqrt();

    assert_eq!(
      root.map(|candidate| candidate.square().normalize()),
      Some(square.normalize())
    );
  }
}
