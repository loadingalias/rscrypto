//! Internal Ed25519 field arithmetic over `2^255 - 19`.
//!
//! This is the portable 5x51 radix baseline. It is the boring core that all
//! later point arithmetic should sit on top of.

use core::fmt;

use super::constants::FIELD_LIMBS;

const RADIX_BITS: u32 = 51;
const RADIX: u64 = 1u64 << RADIX_BITS;
const MASK51: u64 = RADIX - 1;
const INVERT_EXPONENT: [u8; 32] = [
  0xEB, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F,
];
const SQRT_EXPONENT: [u8; 32] = [
  0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F,
];
const MODULUS_0: u64 = RADIX - 19;
const MODULUS_N: u64 = MASK51;
const SUB_BIAS_0: u64 = RADIX.strict_mul(2).strict_sub(38);
const SUB_BIAS_N: u64 = RADIX.strict_mul(2).strict_sub(2);
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

  /// Add two field elements.
  #[inline]
  #[must_use]
  pub(crate) fn add(&self, rhs: &Self) -> Self {
    let [f0, f1, f2, f3, f4] = self.0;
    let [g0, g1, g2, g3, g4] = rhs.0;

    Self(reduce_limbs([
      f0.strict_add(g0),
      f1.strict_add(g1),
      f2.strict_add(g2),
      f3.strict_add(g3),
      f4.strict_add(g4),
    ]))
  }

  /// Subtract two field elements modulo `2^255 - 19`.
  #[inline]
  #[must_use]
  pub(crate) fn sub(&self, rhs: &Self) -> Self {
    let [f0, f1, f2, f3, f4] = self.0;
    let [g0, g1, g2, g3, g4] = rhs.0;

    Self(reduce_limbs([
      f0.strict_add(SUB_BIAS_0).strict_sub(g0),
      f1.strict_add(SUB_BIAS_N).strict_sub(g1),
      f2.strict_add(SUB_BIAS_N).strict_sub(g2),
      f3.strict_add(SUB_BIAS_N).strict_sub(g3),
      f4.strict_add(SUB_BIAS_N).strict_sub(g4),
    ]))
  }

  /// Multiply two field elements.
  #[inline]
  #[must_use]
  pub(crate) fn mul(&self, rhs: &Self) -> Self {
    let [f0, f1, f2, f3, f4] = self.0;
    let [g0, g1, g2, g3, g4] = rhs.0;

    let h0 = mul_acc([(f0, g0, 1), (f1, g4, 19), (f2, g3, 19), (f3, g2, 19), (f4, g1, 19)]);
    let h1 = mul_acc([(f0, g1, 1), (f1, g0, 1), (f2, g4, 19), (f3, g3, 19), (f4, g2, 19)]);
    let h2 = mul_acc([(f0, g2, 1), (f1, g1, 1), (f2, g0, 1), (f3, g4, 19), (f4, g3, 19)]);
    let h3 = mul_acc([(f0, g3, 1), (f1, g2, 1), (f2, g1, 1), (f3, g0, 1), (f4, g4, 19)]);
    let h4 = mul_acc([(f0, g4, 1), (f1, g3, 1), (f2, g2, 1), (f3, g1, 1), (f4, g0, 1)]);

    Self(reduce_wide([h0, h1, h2, h3, h4]))
  }

  /// Square a field element.
  #[inline]
  #[must_use]
  pub(crate) fn square(&self) -> Self {
    self.mul(self)
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
    self.pow(&INVERT_EXPONENT)
  }

  /// Canonically reduce the field element.
  #[inline]
  #[must_use]
  pub(crate) fn normalize(&self) -> Self {
    let reduced = reduce_limbs(self.0);
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
          acc_bits = acc_bits.strict_add(8);
        } else {
          break;
        }
      }

      *limb = (acc & u128::from(MASK51)) as u64;
      acc >>= RADIX_BITS;
      acc_bits = acc_bits.strict_sub(RADIX_BITS);
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
      acc_bits = acc_bits.strict_add(RADIX_BITS);

      while acc_bits >= 8 {
        if let Some(byte) = out_iter.next() {
          *byte = acc as u8;
        }
        acc >>= 8;
        acc_bits = acc_bits.strict_sub(8);
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
    let canonical = self.normalize();
    let mut candidate = canonical.pow(&SQRT_EXPONENT).normalize();

    if candidate.square().normalize() == canonical {
      return Some(candidate);
    }

    candidate = candidate.mul(&SQRT_M1).normalize();
    if candidate.square().normalize() == canonical {
      Some(candidate)
    } else {
      None
    }
  }

  #[must_use]
  fn pow(&self, exponent: &[u8; 32]) -> Self {
    let mut acc = Self::ONE;

    for byte in exponent.iter().rev().copied() {
      let mut shift = 8u32;
      while shift > 0 {
        shift = shift.strict_sub(1);
        acc = acc.square();
        if ((byte >> shift) & 1) == 1 {
          acc = acc.mul(self);
        }
      }
    }

    acc
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

#[inline]
#[must_use]
fn mul_acc(terms: [(u64, u64, u64); FIELD_LIMBS]) -> u128 {
  let mut acc = 0u128;
  for (left, right, scale) in terms {
    acc = acc.strict_add(
      u128::from(left)
        .strict_mul(u128::from(right))
        .strict_mul(u128::from(scale)),
    );
  }
  acc
}

#[inline]
#[must_use]
fn reduce_wide(wide: [u128; FIELD_LIMBS]) -> [u64; FIELD_LIMBS] {
  let [mut h0, mut h1, mut h2, mut h3, mut h4] = wide;
  let mask = u128::from(MASK51);

  propagate_carries(&mut h0, &mut h1, &mut h2, &mut h3, &mut h4, mask);
  propagate_carries(&mut h0, &mut h1, &mut h2, &mut h3, &mut h4, mask);

  [h0 as u64, h1 as u64, h2 as u64, h3 as u64, h4 as u64]
}

#[inline]
#[must_use]
fn reduce_limbs(limbs: [u64; FIELD_LIMBS]) -> [u64; FIELD_LIMBS] {
  let [h0, h1, h2, h3, h4] = limbs;
  reduce_wide([
    u128::from(h0),
    u128::from(h1),
    u128::from(h2),
    u128::from(h3),
    u128::from(h4),
  ])
}

#[inline]
fn propagate_carries(h0: &mut u128, h1: &mut u128, h2: &mut u128, h3: &mut u128, h4: &mut u128, mask: u128) {
  let carry0 = *h0 >> RADIX_BITS;
  *h1 = h1.strict_add(carry0);
  *h0 &= mask;

  let carry1 = *h1 >> RADIX_BITS;
  *h2 = h2.strict_add(carry1);
  *h1 &= mask;

  let carry2 = *h2 >> RADIX_BITS;
  *h3 = h3.strict_add(carry2);
  *h2 &= mask;

  let carry3 = *h3 >> RADIX_BITS;
  *h4 = h4.strict_add(carry3);
  *h3 &= mask;

  let carry4 = *h4 >> RADIX_BITS;
  *h0 = h0.strict_add(carry4.strict_mul(19));
  *h4 &= mask;
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
  let subtrahend = u128::from(rhs).strict_add(u128::from(u8::from(borrow)));
  let lhs = u128::from(lhs);

  if lhs >= subtrahend {
    ((lhs.strict_sub(subtrahend)) as u64, false)
  } else {
    ((lhs.strict_add(u128::from(RADIX)).strict_sub(subtrahend)) as u64, true)
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
