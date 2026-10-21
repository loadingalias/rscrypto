//! Point arithmetic for public inputs: ECDSA verification and public-key comb tables.
//!
//! Peer keys, signatures, digests and the tables derived from them are public, so this module
//! may branch on point values and scalar digits. It runs the generic arithmetic's point formulas
//! (`a = -3`) over a curve-specific [`VerifyField`] whose operations inline into the loops,
//! instead of dispatching each field operation through the generic modulus. The field keeps the
//! generic Montgomery representation, so tables match limb for limb and verification makes the
//! same decisions.

use super::{Affine, COMB_TABLE_SIZE, COMB_WIDTH, Curve, FieldElement, Jacobian, Modulus, Uint, comb_digit, comb_rows};

/// A field in the generic Montgomery representation (`R = 2^(64·L)`), with canonical limbs.
pub(super) trait VerifyField<const L: usize>: Copy + PartialEq {
  const ZERO: Self;
  /// One in Montgomery form: `R mod p`.
  const ONE: Self;
  /// The generic modulus, for conversions and the table inversion.
  const MODULUS: &'static Modulus<L>;

  fn from_limbs(limbs: [u64; L]) -> Self;
  fn limbs(self) -> [u64; L];
  fn add(self, rhs: Self) -> Self;
  fn sub(self, rhs: Self) -> Self;
  fn mul(self, rhs: Self) -> Self;
  fn square(self) -> Self;

  #[inline(always)]
  fn is_zero(self) -> bool {
    self.limbs() == [0; L]
  }

  #[inline(always)]
  fn double(self) -> Self {
    self.add(self)
  }

  #[inline(always)]
  fn triple(self) -> Self {
    self.double().add(self)
  }

  #[inline(always)]
  fn from_element(value: FieldElement<L>) -> Self {
    Self::from_limbs(value.value.0)
  }

  #[inline(always)]
  fn to_element(self) -> FieldElement<L> {
    FieldElement::from_montgomery(Uint(self.limbs()), Self::MODULUS)
  }
}

#[derive(Clone, Copy)]
struct Point<F> {
  x: F,
  y: F,
  z: F,
  infinity: bool,
}

#[derive(Clone, Copy)]
struct AffinePoint<F> {
  x: F,
  y: F,
}

impl<F> AffinePoint<F> {
  #[inline(always)]
  fn from_affine<const L: usize>(point: Affine<L>) -> Self
  where
    F: VerifyField<L>,
  {
    Self {
      x: F::from_element(point.x),
      y: F::from_element(point.y),
    }
  }

  #[inline(always)]
  fn to_affine<const L: usize>(self) -> Affine<L>
  where
    F: VerifyField<L>,
  {
    Affine {
      x: self.x.to_element(),
      y: self.y.to_element(),
    }
  }

  /// The signing-comb generator entry for a public digit.
  #[inline(always)]
  fn generator<const L: usize>(curve: &Curve<L>, digit: usize) -> Self
  where
    F: VerifyField<L>,
  {
    Self {
      x: F::from_limbs(curve.signing_generator_comb_x[digit].0),
      y: F::from_limbs(curve.signing_generator_comb_y[digit].0),
    }
  }
}

impl<F> Point<F> {
  #[inline(always)]
  fn infinity<const L: usize>() -> Self
  where
    F: VerifyField<L>,
  {
    Self {
      x: F::ZERO,
      y: F::ZERO,
      z: F::ZERO,
      infinity: true,
    }
  }

  #[inline(always)]
  fn from_affine<const L: usize>(point: AffinePoint<F>) -> Self
  where
    F: VerifyField<L>,
  {
    Self {
      x: point.x,
      y: point.y,
      z: F::ONE,
      infinity: false,
    }
  }

  /// `2·self` with `a = -3` (dbl-2001-b), as `Jacobian::double` computes it.
  #[inline(always)]
  fn double<const L: usize>(self) -> Self
  where
    F: VerifyField<L>,
  {
    if self.infinity || self.y.is_zero() {
      return Self::infinity();
    }
    let delta = self.z.square();
    let gamma = self.y.square();
    let beta = self.x.mul(gamma);
    let alpha = self.x.sub(delta).mul(self.x.add(delta)).triple();
    let x3 = alpha.square().sub(beta.double().double().double());
    let z3 = self.y.add(self.z).square().sub(gamma).sub(delta);
    let y3 = alpha
      .mul(beta.double().double().sub(x3))
      .sub(gamma.square().double().double().double());
    Self {
      x: x3,
      y: y3,
      z: z3,
      infinity: false,
    }
  }

  /// `self + rhs` for affine `rhs` (madd-2007-bl), as `Jacobian::add_mixed` computes it,
  /// including the doubling and inverse cases.
  #[inline(always)]
  fn add_mixed<const L: usize>(self, rhs: AffinePoint<F>) -> Self
  where
    F: VerifyField<L>,
  {
    if self.infinity {
      return Self::from_affine(rhs);
    }
    let z1z1 = self.z.square();
    let u2 = rhs.x.mul(z1z1);
    let s2 = rhs.y.mul(self.z).mul(z1z1);
    let h = u2.sub(self.x);
    let r = s2.sub(self.y).double();
    if h.is_zero() {
      return if r.is_zero() { self.double() } else { Self::infinity() };
    }
    let hh = h.square();
    let i = hh.double().double();
    let j = h.mul(i);
    let v = self.x.mul(i);
    let x3 = r.square().sub(j).sub(v.double());
    let y3 = r.mul(v.sub(x3)).sub(self.y.mul(j).double());
    let z3 = self.z.add(h).square().sub(z1z1).sub(hh);
    Self {
      x: x3,
      y: y3,
      z: z3,
      infinity: false,
    }
  }

  /// `self + rhs` for Jacobian `rhs` (add-2007-bl), including the infinity, doubling and inverse
  /// cases.
  fn add<const L: usize>(self, rhs: Self) -> Self
  where
    F: VerifyField<L>,
  {
    if self.infinity {
      return rhs;
    }
    if rhs.infinity {
      return self;
    }
    let z1z1 = self.z.square();
    let z2z2 = rhs.z.square();
    let u1 = self.x.mul(z2z2);
    let u2 = rhs.x.mul(z1z1);
    let s1 = self.y.mul(rhs.z).mul(z2z2);
    let s2 = rhs.y.mul(self.z).mul(z1z1);
    let h = u2.sub(u1);
    let r = s2.sub(s1).double();
    if h.is_zero() {
      return if r.is_zero() { self.double() } else { Self::infinity() };
    }
    let i = h.double().square();
    let j = h.mul(i);
    let v = u1.mul(i);
    let x3 = r.square().sub(j).sub(v.double());
    let y3 = r.mul(v.sub(x3)).sub(s1.mul(j).double());
    let z3 = self.z.add(rhs.z).square().sub(z1z1).sub(z2z2).mul(h);
    Self {
      x: x3,
      y: y3,
      z: z3,
      infinity: false,
    }
  }

  #[inline(always)]
  fn negate<const L: usize>(self) -> Self
  where
    F: VerifyField<L>,
  {
    Self {
      y: F::ZERO.sub(self.y),
      ..self
    }
  }

  #[inline(always)]
  fn to_jacobian<const L: usize>(self) -> Jacobian<L>
  where
    F: VerifyField<L>,
  {
    Jacobian {
      x: self.x.to_element(),
      y: self.y.to_element(),
      z: self.z.to_element(),
      infinity: self.infinity,
    }
  }
}

/// Affine forms of finite points, with one field inversion for the whole table.
fn normalize<F: VerifyField<L>, const L: usize, const N: usize>(table: [Point<F>; N]) -> [AffinePoint<F>; N] {
  let mut prefixes = [F::ONE; N];
  let mut acc = F::ONE;
  for (prefix, point) in prefixes.iter_mut().zip(&table) {
    *prefix = acc;
    acc = acc.mul(point.z);
  }
  let mut acc_inverse = F::from_element(acc.to_element().inv());
  let mut out = [AffinePoint { x: F::ZERO, y: F::ZERO }; N];
  for ((point, prefix), output) in table.iter().zip(prefixes).zip(out.iter_mut()).rev() {
    let z_inverse = acc_inverse.mul(prefix);
    acc_inverse = acc_inverse.mul(point.z);
    let z2 = z_inverse.square();
    let z3 = z2.mul(z_inverse);
    *output = AffinePoint {
      x: point.x.mul(z2),
      y: point.y.mul(z3),
    };
  }
  out
}

/// The public-key comb table of `super::precompute_comb_table`: entry `mask` is the sum of the
/// columns `2^(rows·c)·point` whose bit `c` is set, and entry zero is `point`.
pub(super) fn comb_table<F: VerifyField<L>, const L: usize>(point: Affine<L>) -> [Affine<L>; COMB_TABLE_SIZE] {
  let rows = comb_rows::<L>();
  let base = AffinePoint::<F>::from_affine(point);
  let mut columns = [Point::from_affine(base); COMB_WIDTH];
  let mut current = Point::from_affine(base);
  for column in columns.iter_mut().skip(1) {
    for _ in 0..rows {
      current = current.double();
    }
    *column = current;
  }
  let columns = normalize(columns);

  let mut table = [Point::from_affine(base); COMB_TABLE_SIZE];
  for (mask, entry) in table.iter_mut().enumerate().skip(1) {
    let mut acc = Point::infinity();
    for (column, &column_point) in columns.iter().enumerate() {
      if mask & (1usize << column) != 0 {
        acc = acc.add_mixed(column_point);
      }
    }
    *entry = acc;
  }
  normalize(table).map(AffinePoint::to_affine)
}

/// `generator_scalar·G + point_scalar·Q` with the signing generator comb and the public-key comb
/// table of Q, using the digits of `super::scalar_mul_two`.
///
/// The generator and key terms accumulate in separate chains, combined by one addition, so the
/// core overlaps their doublings and additions instead of serializing every addition on one
/// accumulator.
pub(super) fn scalar_mul_two<F: VerifyField<L>, const L: usize>(
  curve: &Curve<L>,
  generator_scalar: Uint<L>,
  point_scalar: Uint<L>,
  point_table: &[Affine<L>; COMB_TABLE_SIZE],
) -> Jacobian<L> {
  let rows = comb_rows::<L>();
  let table = point_table.map(AffinePoint::<F>::from_affine);
  let mut generator = Point::infinity();
  let mut key = Point::infinity();
  for row in (0..rows).rev() {
    if row < curve.signing_comb_rows {
      generator = generator.double();
      let digit = comb_digit(generator_scalar, row, curve.signing_comb_rows, curve.signing_comb_width);
      if digit != 0 {
        generator = generator.add_mixed(AffinePoint::generator(curve, digit));
      }
    }
    key = key.double();
    let digit = comb_digit(point_scalar, row, rows, COMB_WIDTH);
    if digit != 0 {
      key = key.add_mixed(table[digit]);
    }
  }
  generator.add(key).to_jacobian()
}

/// Width-5 non-adjacent form of a scalar below `2^(64·L)`, for `L <= 6`: digit `i` is zero or
/// odd in `[-15, 15]`, `Σ digit_i·2^i` equals the scalar, and digits from `64·L + 1` on are zero.
/// The scalar is public.
pub(super) fn naf5<const L: usize>(scalar: Uint<L>) -> [i8; 385] {
  const { assert!(L <= 6) };
  let mut limbs = [0u64; 7];
  limbs[..L].copy_from_slice(&scalar.0);
  let mut digits = [0i8; 385];
  for digit in &mut digits[..L.strict_mul(64).strict_add(1)] {
    if limbs[0] & 1 == 1 {
      let window = limbs[0] & 31;
      if window < 16 {
        // Subtract the positive digit; the low limb is at least `window`.
        limbs[0] = limbs[0].strict_sub(window);
        *digit = i8::try_from(window).expect("a five-bit window fits i8");
      } else {
        // Add 32 - window, carrying through the limbs; the scalar stays below 2^(64·L + 1).
        let mut carry = 32u64.strict_sub(window);
        for limb in &mut limbs {
          let (sum, overflow) = limb.overflowing_add(carry);
          *limb = sum;
          carry = u64::from(overflow);
        }
        *digit = i8::try_from(window).expect("a five-bit window fits i8").strict_sub(32);
      }
    }
    // Shift the scalar right by one bit.
    for index in 0..6 {
      limbs[index] = (limbs[index] >> 1) | (limbs[index.strict_add(1)] << 63);
    }
    limbs[6] >>= 1;
  }
  digits
}

/// `generator_scalar·G + point_scalar·Q` for a point Q used once, without its comb table.
///
/// The key term runs a width-5 NAF of `point_scalar` over the Jacobian odd multiples
/// Q, 3Q, ..., 15Q, which cost one doubling and seven additions; the generator term runs the
/// signing comb in an independent chain, as in [`scalar_mul_two`].
pub(super) fn scalar_mul_two_single<F: VerifyField<L>, const L: usize>(
  curve: &Curve<L>,
  generator_scalar: Uint<L>,
  point_scalar: Uint<L>,
  point: Affine<L>,
) -> Jacobian<L> {
  let base = AffinePoint::<F>::from_affine(point);
  let twice = Point::from_affine(base).double();
  let mut odd = [Point::from_affine(base); 8];
  for index in 1..odd.len() {
    odd[index] = odd[index.strict_sub(1)].add(twice);
  }

  let digits = naf5(point_scalar);
  let mut generator = Point::infinity();
  let mut key = Point::infinity();
  for (row, &digit) in digits[..L.strict_mul(64).strict_add(1)].iter().enumerate().rev() {
    if row < curve.signing_comb_rows {
      generator = generator.double();
      let comb = comb_digit(generator_scalar, row, curve.signing_comb_rows, curve.signing_comb_width);
      if comb != 0 {
        generator = generator.add_mixed(AffinePoint::generator(curve, comb));
      }
    }
    key = key.double();
    if digit != 0 {
      let entry = odd[usize::from(digit.unsigned_abs() >> 1)];
      key = key.add(if digit > 0 { entry } else { entry.negate() });
    }
  }
  generator.add(key).to_jacobian()
}

#[cfg(test)]
mod tests {
  use super::{Uint, naf5};

  fn assert_naf5_reconstructs<const L: usize>(scalar: [u64; L]) {
    let digits = naf5(Uint(scalar));
    // Rebuild Σ digit·2^i from the top digit down as a signed seven-limb value.
    let mut acc = [0u64; 7];
    for &digit in digits.iter().rev() {
      let mut carry = 0u64;
      for limb in &mut acc {
        let next = *limb >> 63;
        *limb = (*limb << 1) | carry;
        carry = next;
      }
      let mut borrow_or_carry = u64::from(digit.unsigned_abs());
      for limb in &mut acc {
        let (value, flag) = if digit >= 0 {
          limb.overflowing_add(borrow_or_carry)
        } else {
          limb.overflowing_sub(borrow_or_carry)
        };
        *limb = value;
        borrow_or_carry = u64::from(flag);
      }
    }
    let mut expected = [0u64; 7];
    expected[..L].copy_from_slice(&scalar);
    assert_eq!(acc, expected, "{scalar:x?}");
    let mut last_nonzero: Option<usize> = None;
    for (index, &digit) in digits.iter().enumerate() {
      if digit != 0 {
        assert!(digit % 2 != 0 && (-15..=15).contains(&digit), "{scalar:x?}");
        assert!(index <= L.strict_mul(64), "{scalar:x?}");
        if let Some(last) = last_nonzero {
          assert!(
            index.strict_sub(last) >= 5,
            "nonzero digits are at least five apart: {scalar:x?}"
          );
        }
        last_nonzero = Some(index);
      }
    }
  }

  #[test]
  fn naf5_digits_reconstruct_the_scalar() {
    for scalar in [
      [0u64; 4],
      [1, 0, 0, 0],
      [31, 0, 0, 0],
      [u64::MAX; 4],
      [
        0x0123_4567_89ab_cdef,
        0xfedc_ba98_7654_3210,
        0x8000_0000_0000_0001,
        0x7fff_ffff_ffff_fffe,
      ],
    ] {
      assert_naf5_reconstructs(scalar);
    }
    for scalar in [
      [0u64; 6],
      [u64::MAX; 6],
      [
        0x0123_4567_89ab_cdef,
        0,
        u64::MAX,
        0x8000_0000_0000_0001,
        17,
        0x7fff_ffff_ffff_fffe,
      ],
    ] {
      assert_naf5_reconstructs(scalar);
    }
  }
}
