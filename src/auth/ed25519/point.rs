//! Internal Ed25519 point arithmetic.
//!
//! This is the portable extended Edwards baseline using complete addition and
//! a correctness-first scalar-multiplication path.

use core::fmt;

use super::field::FieldElement;

const EDWARDS_D: FieldElement = FieldElement::from_limbs([
  929_955_233_495_203,
  466_365_720_129_213,
  1_662_059_464_998_953,
  2_033_849_074_728_123,
  1_442_794_654_840_575,
]);
const EDWARDS_D2: FieldElement = FieldElement::from_limbs([
  1_859_910_466_990_425,
  932_731_440_258_426,
  1_072_319_116_312_658,
  1_815_898_335_770_999,
  633_789_495_995_903,
]);
const BASEPOINT_X: FieldElement = FieldElement::from_limbs([
  1_738_742_601_995_546,
  1_146_398_526_822_698,
  2_070_867_633_025_821,
  562_264_141_797_630,
  587_772_402_128_613,
]);
const BASEPOINT_Y: FieldElement = FieldElement::from_limbs([
  1_801_439_850_948_184,
  1_351_079_888_211_148,
  450_359_962_737_049,
  900_719_925_474_099,
  1_801_439_850_948_198,
]);

/// Internal extended Edwards point `(X, Y, Z, T)`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExtendedPoint {
  x: FieldElement,
  y: FieldElement,
  z: FieldElement,
  t: FieldElement,
}

impl ExtendedPoint {
  /// Extended-coordinate identity point.
  #[must_use]
  pub(crate) const fn identity() -> Self {
    Self {
      x: FieldElement::ZERO,
      y: FieldElement::ONE,
      z: FieldElement::ONE,
      t: FieldElement::ZERO,
    }
  }

  /// Construct an extended point from affine coordinates.
  #[must_use]
  pub(crate) fn from_affine(x: FieldElement, y: FieldElement) -> Self {
    Self {
      x,
      y,
      z: FieldElement::ONE,
      t: x.mul(&y),
    }
  }

  /// Add two extended Edwards points.
  #[must_use]
  pub(crate) fn add(&self, rhs: &Self) -> Self {
    let a = self.y.sub(&self.x).mul(&rhs.y.sub(&rhs.x));
    let b = self.y.add(&self.x).mul(&rhs.y.add(&rhs.x));
    let c = self.t.mul(&rhs.t).mul(&EDWARDS_D2);
    let zz = self.z.mul(&rhs.z);
    let d = zz.add(&zz);
    let e = b.sub(&a);
    let f = d.sub(&c);
    let g = d.add(&c);
    let h = b.add(&a);

    Self {
      x: e.mul(&f),
      y: g.mul(&h),
      z: f.mul(&g),
      t: e.mul(&h),
    }
  }

  /// Double an extended Edwards point.
  #[must_use]
  pub(crate) fn double(&self) -> Self {
    self.add(self)
  }

  /// Compress the point into the standard Ed25519 encoding.
  #[must_use]
  pub(crate) fn to_bytes(self) -> Option<[u8; 32]> {
    let (x, y) = self.to_affine()?;
    let mut bytes = y.to_bytes();
    if x.is_negative() {
      bytes[31] |= 0x80;
    }
    Some(bytes)
  }

  /// Decode a compressed Ed25519 point.
  #[must_use]
  pub(crate) fn from_bytes(bytes: &[u8; 32]) -> Option<Self> {
    let sign = (bytes[31] >> 7) != 0;
    let mut y_bytes = *bytes;
    y_bytes[31] &= 0x7F;
    let y = FieldElement::from_bytes(&y_bytes)?;
    let y2 = y.square();
    let numerator = y2.sub(&FieldElement::ONE);
    let denominator = EDWARDS_D.mul(&y2).add(&FieldElement::ONE);
    let x2 = numerator.mul(&denominator.invert()).normalize();
    let mut x = x2.sqrt()?;

    if x.is_zero() && sign {
      return None;
    }
    if x.is_negative() != sign {
      x = x.neg();
    }

    Some(Self::from_affine(x.normalize(), y.normalize()))
  }

  /// Standard Ed25519 basepoint.
  #[must_use]
  pub(crate) fn basepoint() -> Self {
    Self::from_affine(BASEPOINT_X, BASEPOINT_Y)
  }

  /// Scalar multiplication by a little-endian 32-byte scalar.
  #[must_use]
  pub(crate) fn scalar_mul(&self, scalar: &[u8; 32]) -> Self {
    let mut acc = Self::identity();

    for byte in scalar.iter().rev().copied() {
      let mut shift = 8u32;
      while shift > 0 {
        shift = shift.strict_sub(1);
        acc = acc.double();
        if ((byte >> shift) & 1) == 1 {
          acc = acc.add(self);
        }
      }
    }

    acc
  }

  /// Fixed-base multiplication for the Ed25519 basepoint.
  #[must_use]
  pub(crate) fn scalar_mul_basepoint(scalar: &[u8; 32]) -> Self {
    Self::basepoint().scalar_mul(scalar)
  }

  /// Multiply by the Edwards cofactor.
  #[must_use]
  pub(crate) fn mul_by_cofactor(&self) -> Self {
    self.double().double().double()
  }

  /// Convert the point to affine coordinates when `Z != 0`.
  #[must_use]
  pub(crate) fn to_affine(self) -> Option<(FieldElement, FieldElement)> {
    if self.z.is_zero() {
      return None;
    }

    let inv_z = self.z.invert();
    Some((self.x.mul(&inv_z).normalize(), self.y.mul(&inv_z).normalize()))
  }

  /// Borrow the extended-coordinate components.
  #[must_use]
  pub(crate) const fn components(&self) -> (&FieldElement, &FieldElement, &FieldElement, &FieldElement) {
    (&self.x, &self.y, &self.z, &self.t)
  }
}

impl Default for ExtendedPoint {
  fn default() -> Self {
    Self::identity()
  }
}

impl fmt::Debug for ExtendedPoint {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("ExtendedPoint").finish_non_exhaustive()
  }
}

#[cfg(test)]
mod tests {
  use super::ExtendedPoint;
  use crate::auth::ed25519::{Ed25519SecretKey, field::FieldElement, hash::ExpandedSecret};

  fn basepoint() -> ExtendedPoint {
    ExtendedPoint::basepoint()
  }

  fn decode_hex_32(hex: &str) -> [u8; 32] {
    let bytes = hex.as_bytes();
    let mut out = [0u8; 32];

    for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
      *dst = hex_value(chunk[0]) << 4 | hex_value(chunk[1]);
    }

    out
  }

  fn hex_value(byte: u8) -> u8 {
    match byte {
      b'0'..=b'9' => byte - b'0',
      b'a'..=b'f' => byte - b'a' + 10,
      b'A'..=b'F' => byte - b'A' + 10,
      _ => panic!("invalid hex"),
    }
  }

  #[test]
  fn identity_has_expected_affine_coordinates() {
    let affine = ExtendedPoint::identity().to_affine();

    assert_eq!(affine, Some((FieldElement::ZERO, FieldElement::ONE)));
  }

  #[test]
  fn affine_constructor_sets_t_to_xy() {
    let point = basepoint();
    let (x, y, z, t) = point.components();

    assert_eq!(*z, FieldElement::ONE);
    assert_eq!(*t, x.mul(y));
  }

  #[test]
  fn identity_is_neutral_for_addition() {
    let point = basepoint();
    let identity = ExtendedPoint::identity();

    assert_eq!(point.add(&identity).to_affine(), point.to_affine());
    assert_eq!(identity.add(&point).to_affine(), point.to_affine());
  }

  #[test]
  fn doubling_matches_add_self() {
    let point = basepoint();

    assert_eq!(point.double().to_affine(), point.add(&point).to_affine());
  }

  #[test]
  fn basepoint_roundtrips_compressed_encoding() {
    let expected = decode_hex_32("5866666666666666666666666666666666666666666666666666666666666666");
    let encoded = basepoint().to_bytes();

    assert_eq!(encoded, Some(expected));
    assert_eq!(
      encoded
        .and_then(|bytes| ExtendedPoint::from_bytes(&bytes))
        .and_then(|point| point.to_bytes()),
      Some(expected)
    );
  }

  #[test]
  fn compressed_identity_with_sign_bit_set_is_rejected() {
    let mut bytes = [0u8; 32];
    bytes[0] = 1;
    bytes[31] = 0x80;

    assert_eq!(ExtendedPoint::from_bytes(&bytes), None);
  }

  #[test]
  fn scalar_mul_basepoint_zero_is_identity() {
    let point = ExtendedPoint::scalar_mul_basepoint(&[0u8; 32]);

    assert_eq!(point.to_affine(), Some((FieldElement::ZERO, FieldElement::ONE)));
  }

  #[test]
  fn scalar_mul_basepoint_one_is_basepoint() {
    let mut scalar = [0u8; 32];
    scalar[0] = 1;

    assert_eq!(
      ExtendedPoint::scalar_mul_basepoint(&scalar).to_bytes(),
      basepoint().to_bytes()
    );
  }

  #[test]
  fn cofactor_mul_identity_stays_identity() {
    let point = ExtendedPoint::identity().mul_by_cofactor();

    assert_eq!(point.to_affine(), Some((FieldElement::ZERO, FieldElement::ONE)));
  }

  #[test]
  fn rfc8032_public_key_derivation_matches_vector_1() {
    let secret = Ed25519SecretKey::from_bytes(decode_hex_32(
      "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
    ));
    let expanded = ExpandedSecret::from_secret_key(&secret);
    let public = ExtendedPoint::scalar_mul_basepoint(expanded.scalar_bytes()).to_bytes();
    let expected = decode_hex_32("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");

    assert_eq!(public, Some(expected));
  }
}
