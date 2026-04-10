//! Internal Ed25519 point arithmetic.
//!
//! This is the portable extended Edwards baseline using complete addition and
//! a correctness-first scalar-multiplication path.

use core::fmt;

use super::{field::FieldElement, scalar};

#[path = "basepoint_tables.rs"]
mod basepoint_tables;
#[path = "basepoint_wnaf5_table.rs"]
mod basepoint_wnaf5_table;

pub(crate) use self::{basepoint_tables::BASEPOINT_RADIX16_TABLE, basepoint_wnaf5_table::BASEPOINT_WNAF5_TABLE};

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

/// Cached affine precompute for mixed Edwards addition (7M per add).
///
/// Used for the static basepoint table where points are precomputed at
/// build time with Z = 1.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct CachedPoint {
  y_plus_x: FieldElement,
  y_minus_x: FieldElement,
  t2d: FieldElement,
}

impl CachedPoint {
  /// Borrow the cached-point components.
  #[cfg(target_arch = "x86_64")]
  #[must_use]
  pub(crate) const fn components(&self) -> (&FieldElement, &FieldElement, &FieldElement) {
    (&self.y_plus_x, &self.y_minus_x, &self.t2d)
  }

  #[must_use]
  fn neg(&self) -> Self {
    Self {
      y_plus_x: self.y_minus_x,
      y_minus_x: self.y_plus_x,
      t2d: self.t2d.neg(),
    }
  }
}

/// Projective cached precompute for non-affine mixed addition (8M per add).
///
/// Stores `(Y+X, Y-X, Z, 2dT)` in projective form. Costs 1 more field
/// multiply per addition than affine `CachedPoint`, but avoids the
/// expensive field inversion required to convert runtime tables to affine.
#[derive(Clone, Copy)]
struct ProjectiveCachedPoint {
  y_plus_x: FieldElement,
  y_minus_x: FieldElement,
  z: FieldElement,
  t2d: FieldElement,
}

impl ProjectiveCachedPoint {
  const IDENTITY: Self = Self {
    y_plus_x: FieldElement::ONE,
    y_minus_x: FieldElement::ONE,
    z: FieldElement::ONE,
    t2d: FieldElement::ZERO,
  };

  #[must_use]
  fn from_extended(p: &ExtendedPoint) -> Self {
    Self {
      y_plus_x: p.y.add(&p.x),
      y_minus_x: p.y.sub(&p.x),
      z: p.z,
      t2d: p.t.mul(&EDWARDS_D2),
    }
  }

  #[must_use]
  fn neg(&self) -> Self {
    Self {
      y_plus_x: self.y_minus_x,
      y_minus_x: self.y_plus_x,
      z: self.z,
      t2d: self.t2d.neg(),
    }
  }
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

  /// Add a cached affine point using the mixed-add path.
  #[must_use]
  pub(crate) fn add_cached(&self, rhs: &CachedPoint) -> Self {
    let a = self.y.sub(&self.x).mul(&rhs.y_minus_x);
    let b = self.y.add(&self.x).mul(&rhs.y_plus_x);
    let c = self.t.mul(&rhs.t2d);
    let d = self.z.add(&self.z);
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

  /// Add a projective cached point (8M — 1 more than affine cached, but
  /// avoids the inversion needed to build affine runtime tables).
  #[must_use]
  fn add_projective_cached(&self, rhs: &ProjectiveCachedPoint) -> Self {
    let a = self.y.sub(&self.x).mul(&rhs.y_minus_x);
    let b = self.y.add(&self.x).mul(&rhs.y_plus_x);
    let c = self.t.mul(&rhs.t2d);
    let d = self.z.mul(&rhs.z);
    let d2 = d.add(&d);
    let e = b.sub(&a);
    let f = d2.sub(&c);
    let g = d2.add(&c);
    let h = b.add(&a);

    Self {
      x: e.mul(&f),
      y: g.mul(&h),
      z: f.mul(&g),
      t: e.mul(&h),
    }
  }

  /// Double an extended Edwards point.
  ///
  /// Dedicated `dbl-2008-hwcd` formula for `a = -1`: 4 squarings + 4
  /// multiplications, no curve-constant multiply. The general `add(self)`
  /// path costs 4 squarings + 5 multiplications and can't exploit squaring
  /// symmetry in the compiler.
  #[must_use]
  pub(crate) fn double(&self) -> Self {
    let a = self.x.square();
    let b = self.y.square();
    let zz = self.z.square();
    let c = zz.add(&zz); // 2·Z²
    let d = a.neg(); // a·X² = -X² since a = -1
    let e = self.x.add(&self.y).square().sub(&a).sub(&b); // (X+Y)² - X² - Y²
    let g = d.add(&b);
    let f = g.sub(&c);
    let h = d.sub(&b);

    Self {
      x: e.mul(&f),
      y: g.mul(&h),
      z: f.mul(&g),
      t: e.mul(&h),
    }
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
    let mut x = numerator.sqrt_ratio_i(&denominator)?;

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

  /// Variable-base signed radix-16 multiplication using a projective runtime
  /// table (no field inversion).
  #[must_use]
  pub(crate) fn scalar_mul_vartime(&self, scalar: &[u8; 32]) -> Self {
    let digits = scalar::as_radix_16(scalar);
    let table = projective_cached_multiples(self);
    let mut acc = Self::identity();

    for digit in digits.iter().rev().copied() {
      acc = acc.double().double().double().double();
      if digit != 0 {
        acc = add_signed_projective_cached(acc, &table, digit);
      }
    }

    acc
  }

  /// Fixed-base multiplication for the Ed25519 basepoint.
  ///
  /// Uses a positional radix-16 precompute table. Each signed nibble is added
  /// directly against its `16^i * B` cached multiple, eliminating the
  /// repeated-doubling work from the fixed-base path.
  #[must_use]
  pub(crate) fn scalar_mul_basepoint(scalar: &[u8; 32]) -> Self {
    let digits = scalar::as_radix_16(scalar);
    let mut acc = Self::identity();

    for (position, digit) in digits.iter().copied().enumerate() {
      if digit != 0
        && let Some(table) = BASEPOINT_RADIX16_TABLE.get(position)
      {
        acc = add_signed_cached(acc, table, digit);
      }
    }

    acc
  }

  // Radix-16 Straus removed — superseded by straus_wnaf_basepoint_vartime.

  /// Multiply by the Edwards cofactor.
  #[must_use]
  pub(crate) fn mul_by_cofactor(&self) -> Self {
    self.double().double().double()
  }

  /// Whether this point lies in the low-order torsion subgroup.
  #[must_use]
  pub(crate) fn is_small_order(&self) -> bool {
    self.mul_by_cofactor().equals_projective(&Self::identity())
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

  /// Compare two extended points without converting to affine coordinates.
  #[must_use]
  pub(crate) fn equals_projective(&self, rhs: &Self) -> bool {
    if self.z.is_zero() || rhs.z.is_zero() {
      return false;
    }

    self.x.mul(&rhs.z).normalize() == rhs.x.mul(&self.z).normalize()
      && self.y.mul(&rhs.z).normalize() == rhs.y.mul(&self.z).normalize()
  }

  /// Construct from raw field elements (no validation).
  #[cfg(target_arch = "x86_64")]
  #[must_use]
  pub(crate) const fn from_raw(x: FieldElement, y: FieldElement, z: FieldElement, t: FieldElement) -> Self {
    Self { x, y, z, t }
  }

  /// Borrow the extended-coordinate components.
  #[must_use]
  pub(crate) const fn components(&self) -> (&FieldElement, &FieldElement, &FieldElement, &FieldElement) {
    (&self.x, &self.y, &self.z, &self.t)
  }
}

/// Add a signed digit from an affine cached table (basepoint table).
#[inline]
#[must_use]
fn add_signed_cached(acc: ExtendedPoint, table: &[CachedPoint; 8], digit: i8) -> ExtendedPoint {
  let index = usize::from(digit.unsigned_abs()).wrapping_sub(1);
  let Some(point) = table.get(index).copied() else {
    return acc;
  };

  if digit > 0 {
    acc.add_cached(&point)
  } else {
    acc.add_cached(&point.neg())
  }
}

/// Add a signed digit from a projective cached table (runtime table).
#[inline]
#[must_use]
fn add_signed_projective_cached(acc: ExtendedPoint, table: &[ProjectiveCachedPoint; 8], digit: i8) -> ExtendedPoint {
  let index = usize::from(digit.unsigned_abs()).wrapping_sub(1);
  let Some(point) = table.get(index).copied() else {
    return acc;
  };

  if digit > 0 {
    acc.add_projective_cached(&point)
  } else {
    acc.add_projective_cached(&point.neg())
  }
}

/// Build a projective cached table of `[1P, 2P, ..., 8P]` without any
/// field inversion. Each entry stores `(Y+X, Y-X, Z, 2dT)` in extended
/// projective coordinates.
#[must_use]
fn projective_cached_multiples(point: &ExtendedPoint) -> [ProjectiveCachedPoint; 8] {
  let mut out = [ProjectiveCachedPoint::IDENTITY; 8];
  let mut acc = *point;
  out[0] = ProjectiveCachedPoint::from_extended(&acc);

  for entry in out.iter_mut().skip(1) {
    acc = acc.add(point);
    *entry = ProjectiveCachedPoint::from_extended(&acc);
  }

  out
}

/// Build odd multiples `[1P, 3P, 5P, ..., (2n-1)P]` in projective cached format.
#[must_use]
fn odd_projective_cached_multiples<const N: usize>(point: &ExtendedPoint) -> [ProjectiveCachedPoint; N] {
  let p2 = point.double();
  let mut out = [ProjectiveCachedPoint::IDENTITY; N];
  let mut acc = *point;
  out[0] = ProjectiveCachedPoint::from_extended(&acc);

  for dst in out.iter_mut().skip(1) {
    acc = acc.add(&p2);
    *dst = ProjectiveCachedPoint::from_extended(&acc);
  }

  out
}

/// Add a signed wNAF digit from an odd-multiples projective cached table.
#[inline]
#[must_use]
fn add_wnaf_digit_projective(acc: ExtendedPoint, table: &[ProjectiveCachedPoint], digit: i8) -> ExtendedPoint {
  let index = usize::from((digit.unsigned_abs().wrapping_sub(1)) / 2);
  let Some(point) = table.get(index).copied() else {
    return acc;
  };

  if digit > 0 {
    acc.add_projective_cached(&point)
  } else {
    acc.add_projective_cached(&point.neg())
  }
}

/// Add a signed wNAF digit from an affine cached table.
#[inline]
#[must_use]
fn add_wnaf_digit_cached(acc: ExtendedPoint, table: &[CachedPoint], digit: i8) -> ExtendedPoint {
  let index = usize::from((digit.unsigned_abs().wrapping_sub(1)) / 2);
  let Some(point) = table.get(index).copied() else {
    return acc;
  };

  if digit > 0 {
    acc.add_cached(&point)
  } else {
    acc.add_cached(&point.neg())
  }
}

/// wNAF-based portable Straus: `[s]B + [h]A`.
///
/// Uses wNAF(5) for both scalars (8-entry odd-multiples tables).
#[must_use]
#[allow(clippy::indexing_slicing)] // i bounded by top < 256, naf arrays are [i8; 256]
pub(crate) fn straus_wnaf_basepoint_vartime(s: &[u8; 32], h: &[u8; 32], a: &ExtendedPoint) -> ExtendedPoint {
  let s_naf = scalar::non_adjacent_form(s, 5);
  let h_naf = scalar::non_adjacent_form(h, 5);

  let a_table: [ProjectiveCachedPoint; 8] = odd_projective_cached_multiples(a);

  let top = s_naf
    .iter()
    .enumerate()
    .rev()
    .find(|&(_, &d)| d != 0)
    .map_or(0, |(i, _)| i)
    .max(
      h_naf
        .iter()
        .enumerate()
        .rev()
        .find(|&(_, &d)| d != 0)
        .map_or(0, |(i, _)| i),
    );

  let mut acc = ExtendedPoint::identity();

  for i in (0..=top).rev() {
    acc = acc.double();
    if s_naf[i] != 0 {
      acc = add_wnaf_digit_cached(acc, &BASEPOINT_WNAF5_TABLE, s_naf[i]);
    }
    if h_naf[i] != 0 {
      acc = add_wnaf_digit_projective(acc, &a_table, h_naf[i]);
    }
  }

  acc
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

  /// Generates the wNAF(8) basepoint table for IFMA verify.
  /// Run with `--nocapture` to print the table source.
  #[test]
  fn gen_basepoint_wnaf8_ifma_table() {
    let bp = ExtendedPoint::basepoint();
    let bp2 = bp.double();

    let d2_fe = FieldElement::from_small(121_666);
    let d2_2_fe = FieldElement::from_small(121_666u64.wrapping_mul(2));
    let d1_2_fe = FieldElement::from_small(121_665u64.wrapping_mul(2));

    let mut acc = bp;
    eprintln!("pub(crate) static BASEPOINT_WNAF8_IFMA_RAW: [[[i64; 4]; 5]; 64] = [");
    for i in 0..64u32 {
      if i > 0 {
        acc = acc.add(&bp2);
      }
      let (x, y, z, t) = acc.components();

      let a = d2_fe.mul(&y.sub(x)).normalize();
      let b = d2_fe.mul(&y.add(x)).normalize();
      let c = d2_2_fe.mul(z).normalize();
      let d = d1_2_fe.mul(t).neg().normalize();

      let al = a.limbs();
      let bl = b.limbs();
      let cl = c.limbs();
      let dl = d.limbs();

      eprintln!("  // {}B", 2 * i + 1);
      eprintln!("  [");
      for k in 0..5 {
        eprintln!(
          "    [{}, {}, {}, {}],",
          al[k] as i64, bl[k] as i64, cl[k] as i64, dl[k] as i64
        );
      }
      eprintln!("  ],");
    }
    eprintln!("];");
  }

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
  fn straus_wnaf_basepoint_matches_fixed_base_when_variable_scalar_is_zero() {
    let mut s = [0u8; 32];
    s[0] = 7;
    let h = [0u8; 32];
    let result = super::straus_wnaf_basepoint_vartime(&s, &h, &ExtendedPoint::identity());
    let expected = ExtendedPoint::scalar_mul_basepoint(&s);

    assert!(result.equals_projective(&expected));
  }

  #[test]
  fn basepoint_wnaf5_table_matches_odd_basepoint_multiples() {
    for (i, cached) in super::BASEPOINT_WNAF5_TABLE.iter().enumerate() {
      let scalar = (2 * i + 1) as u8;
      let mut scalar_bytes = [0u8; 32];
      scalar_bytes[0] = scalar;

      let point = ExtendedPoint::identity().add_cached(cached);
      let expected = ExtendedPoint::scalar_mul_basepoint(&scalar_bytes);

      assert!(
        point.equals_projective(&expected),
        "wNAF table entry for {}B should match scalar multiplication",
        scalar
      );
    }
  }

  #[test]
  fn small_order_detection_matches_identity_and_basepoint() {
    assert!(ExtendedPoint::identity().is_small_order());
    assert!(!basepoint().is_small_order());
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
