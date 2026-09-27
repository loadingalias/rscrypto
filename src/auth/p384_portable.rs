//! Portable P-384 arithmetic and encoding authority for ephemeral ECDH.
//!
//! AArch64 builds route field multiplication, squaring, addition,
//! subtraction, small multiples, and the fixed-base comb scan through
//! `p384_aarch64.rs`, and field inversion through `p384_divsteps.rs`. The
//! portable functions here remain the semantic authority for those kernels.
//!
//! Field elements are six little-endian 64-bit limbs in Montgomery form with
//! `R = 2^384`. The modulus `p = 2^384 - 2^128 - 2^96 + 2^32 - 1` gives
//! `-p^-1 mod 2^64 = 2^32 + 1`, so each reduction step derives its quotient
//! and `q * p` from shifts and additions instead of general multiplications.
//!
//! After scalar validation, secret scalars select table entries only through
//! full-table masked scans and never choose a branch or memory address. This
//! is a source-level property; timing claims require target binary evidence.

use core::cmp::Ordering;

use super::ecdsa_generator_tables::{
  P384_SIGNING_COMB_WIDTH, P384_SIGNING_GENERATOR_COMB_X, P384_SIGNING_GENERATOR_COMB_Y,
};
use crate::traits::ct;

#[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
#[path = "p384_aarch64.rs"]
mod aarch64;
#[cfg(any(
  all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
  all(target_arch = "x86_64", not(feature = "portable-only"), not(miri))
))]
#[path = "p384_divsteps.rs"]
mod divsteps;
#[cfg(all(target_arch = "x86_64", not(feature = "portable-only"), not(miri)))]
#[path = "p384_x86_64.rs"]
mod x86_64;

pub(super) const FIELD_BYTES: usize = 48;
pub(super) const SEC1_BYTES: usize = 97;
const LIMBS: usize = 6;
const WIDE_LIMBS: usize = 2 * LIMBS;
/// Signed radix-32 digits cover 385 bits so the top digit absorbs the final carry.
const WINDOW_BITS: usize = 5;
const WINDOW_DIGITS: usize = 77;
/// Precomputed multiples `1P..=16P` for signed digits in `[-15, 16]`.
const WINDOW_TABLE_SIZE: usize = 16;
const COMB_ROWS: usize = 48;

const _: () = assert!(WINDOW_DIGITS * WINDOW_BITS >= 385);
const _: () = assert!(COMB_ROWS * P384_SIGNING_COMB_WIDTH == 384);

const FIELD_MODULUS: Uint = Uint([
  0x0000_0000_ffff_ffff,
  0xffff_ffff_0000_0000,
  0xffff_ffff_ffff_fffe,
  0xffff_ffff_ffff_ffff,
  0xffff_ffff_ffff_ffff,
  0xffff_ffff_ffff_ffff,
]);
const SCALAR_MODULUS: Uint = Uint([
  0xecec_196a_ccc5_2973,
  0x581a_0db2_48b0_a77a,
  0xc763_4d81_f437_2ddf,
  0xffff_ffff_ffff_ffff,
  0xffff_ffff_ffff_ffff,
  0xffff_ffff_ffff_ffff,
]);
/// `R^2 mod p`, used to enter the Montgomery domain.
const FIELD_R2: Uint = Uint([
  0xffff_fffe_0000_0001,
  0x0000_0002_0000_0000,
  0xffff_fffe_0000_0000,
  0x0000_0002_0000_0000,
  0x0000_0000_0000_0001,
  0x0000_0000_0000_0000,
]);
/// `R mod p = 2^128 + 2^96 - 2^32 + 1`.
const FIELD_ONE_MONTGOMERY: Uint = Uint([
  0xffff_ffff_0000_0001,
  0x0000_0000_ffff_ffff,
  0x0000_0000_0000_0001,
  0x0000_0000_0000_0000,
  0x0000_0000_0000_0000,
  0x0000_0000_0000_0000,
]);
const CURVE_B_MONTGOMERY: Uint = Uint([
  0x0811_8871_9d41_2dcc,
  0xf729_add8_7a4c_32ec,
  0x77f2_209b_1920_022e,
  0xe337_4bee_9493_8ae2,
  0xb62b_21f4_1f02_2094,
  0xcd08_114b_604f_bff9,
]);

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
struct Uint([u64; LIMBS]);

impl Uint {
  const ZERO: Self = Self([0; LIMBS]);

  fn from_be_bytes(bytes: &[u8; FIELD_BYTES]) -> Self {
    let mut limbs = [0u64; LIMBS];
    for (limb, chunk) in limbs.iter_mut().zip(bytes.rchunks_exact(8)) {
      let mut word = [0u8; 8];
      word.copy_from_slice(chunk);
      *limb = u64::from_be_bytes(word);
    }
    Self(limbs)
  }

  fn write_be(self, out: &mut [u8; FIELD_BYTES]) {
    for (chunk, limb) in out.rchunks_exact_mut(8).zip(self.0) {
      chunk.copy_from_slice(&limb.to_be_bytes());
    }
  }

  /// Variable-time comparison for public values and rejection sampling.
  fn cmp_vartime(&self, other: &Self) -> Ordering {
    for (&left, &right) in self.0.iter().zip(other.0.iter()).rev() {
      if left != right {
        return left.cmp(&right);
      }
    }
    Ordering::Equal
  }

  #[inline(always)]
  fn add_raw(self, rhs: Self) -> (Self, u64) {
    let mut out = [0u64; LIMBS];
    let mut carry = 0u64;
    for ((dst, left), right) in out.iter_mut().zip(self.0).zip(rhs.0) {
      (*dst, carry) = adc_limb(left, right, carry);
    }
    (Self(out), carry)
  }

  #[inline(always)]
  fn sub_raw(self, rhs: Self) -> (Self, u64) {
    let mut out = [0u64; LIMBS];
    let mut borrow = 0u64;
    for ((dst, left), right) in out.iter_mut().zip(self.0).zip(rhs.0) {
      (*dst, borrow) = sbb_limb(left, right, borrow);
    }
    (Self(out), borrow)
  }

  /// Return `p` when `mask` is all ones and zero when it is all zeros.
  #[inline(always)]
  fn masked_field_modulus(mask: u64) -> Self {
    #[cfg(any(
      target_arch = "riscv32",
      target_arch = "riscv64",
      target_arch = "s390x",
      target_arch = "x86_64"
    ))]
    // SECURITY: RISC-V LLVM lowered the masked low limb to a branch around a
    // move. Keep the secret-derived mask opaque, as in `Self::select`.
    let mask = core::hint::black_box(mask);
    let p = FIELD_MODULUS.0;
    Self([p[0] & mask, p[1] & mask, p[2] & mask, mask, mask, mask])
  }

  #[inline(always)]
  #[cfg(any(
    test,
    not(any(
      all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
      all(target_arch = "x86_64", not(feature = "portable-only"), not(miri))
    ))
  ))]
  fn add_mod(self, rhs: Self) -> Self {
    let (sum, carry) = self.add_raw(rhs);
    let (reduced, borrow) = sum.sub_raw(FIELD_MODULUS);
    Self::select(sum, reduced, mask_nonzero(carry) | mask_zero(borrow))
  }

  #[inline(always)]
  #[cfg(any(
    test,
    not(any(
      all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
      all(target_arch = "x86_64", not(feature = "portable-only"), not(miri))
    ))
  ))]
  fn sub_mod(self, rhs: Self) -> Self {
    let (difference, borrow) = self.sub_raw(rhs);
    difference
      .add_raw(Self::masked_field_modulus(0u64.wrapping_sub(borrow)))
      .0
  }

  #[inline(always)]
  fn zero_mask(self) -> u64 {
    mask_zero(self.0.into_iter().fold(0u64, |acc, limb| acc | limb))
  }

  #[inline(always)]
  fn select(left: Self, right: Self, mask: u64) -> Self {
    #[cfg(any(
      target_arch = "riscv32",
      target_arch = "riscv64",
      target_arch = "s390x",
      target_arch = "x86_64"
    ))]
    // SECURITY: Keep the mask opaque so the tested LLVM builds retain bitwise
    // selection instead of branching on a secret-derived mask. Binary CT evidence is
    // still required; black_box is not a language-level constant-time guarantee.
    let mask = core::hint::black_box(mask);
    let mut out = [0u64; LIMBS];
    for ((dst, left), right) in out.iter_mut().zip(left.0).zip(right.0) {
      *dst = left ^ (mask & (left ^ right));
    }
    Self(out)
  }

  /// Return `candidate` when `mask` is all ones and zero when it is all zeros.
  #[inline(always)]
  fn masked(candidate: &[u64; LIMBS], mask: u64) -> Self {
    Self(candidate.map(|limb| limb & mask))
  }

  /// OR `candidate` into `self` when `mask` is all ones.
  #[inline(always)]
  fn accumulate_masked(&mut self, candidate: &[u64; LIMBS], mask: u64) {
    for (dst, &limb) in self.0.iter_mut().zip(candidate) {
      *dst |= limb & mask;
    }
  }

  fn zeroize_no_fence(&mut self) {
    ct::zeroize_words_no_fence(&mut self.0);
  }
}

/// Canonical nonzero secret scalar below the group order.
struct Scalar(Uint);

impl Scalar {
  fn from_bytes(bytes: &[u8; FIELD_BYTES]) -> Self {
    Self(Uint::from_be_bytes(bytes))
  }

  #[inline(always)]
  fn bit(&self, index: usize) -> usize {
    let limb = self.0.0.get(index / 64).copied().unwrap_or(0);
    usize::from(((limb >> (index % 64)) & 1).to_le_bytes()[0])
  }

  /// Recode into signed radix-32 digits in `[-15, 16]`, stored as two's complement.
  fn signed_window_digits(&self) -> WindowDigits {
    let mut digits = WindowDigits([0u8; WINDOW_DIGITS]);
    let mut carry = 0u64;
    for (index, digit) in digits.0.iter_mut().enumerate() {
      let bit = index.strict_mul(WINDOW_BITS);
      let limb = bit / 64;
      let shift = bit % 64;
      let mut window = self.0.0.get(limb).copied().unwrap_or(0) >> shift;
      // `shift` and `limb` depend only on the public digit index.
      if shift > 64 - WINDOW_BITS {
        window |= self.0.0.get(limb.strict_add(1)).copied().unwrap_or(0) << (64usize.strict_sub(shift));
      }
      let value = (window & 0x1f).strict_add(carry);
      let negative = 16u64.wrapping_sub(value) >> 63;
      *digit = value.wrapping_sub(negative << WINDOW_BITS).to_le_bytes()[0];
      carry = negative;
    }
    debug_assert_eq!(carry, 0, "the top digit of a 384-bit scalar is at most 16");
    digits
  }

  /// Return the fixed-base comb digit for `row`.
  #[inline(always)]
  fn comb_digit(&self, row: usize) -> usize {
    let mut digit = 0usize;
    for column in 0..P384_SIGNING_COMB_WIDTH {
      digit |= self.bit(row.strict_add(column.strict_mul(COMB_ROWS))) << column;
    }
    digit
  }
}

impl Drop for Scalar {
  fn drop(&mut self) {
    ct::zeroize_words(&mut self.0.0);
  }
}

struct WindowDigits([u8; WINDOW_DIGITS]);

impl Drop for WindowDigits {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
struct FieldElement(Uint);

impl FieldElement {
  const ZERO: Self = Self(Uint::ZERO);
  const ONE: Self = Self(FIELD_ONE_MONTGOMERY);

  fn from_uint(value: Uint) -> Self {
    Self(montgomery_mul(value, FIELD_R2))
  }

  const fn from_montgomery(value: Uint) -> Self {
    Self(value)
  }

  fn to_uint(self) -> Uint {
    let mut wide = [0u64; WIDE_LIMBS];
    wide[..LIMBS].copy_from_slice(&self.0.0);
    montgomery_reduce(wide)
  }

  #[inline(always)]
  fn add(self, rhs: Self) -> Self {
    #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
    {
      Self(Uint(aarch64::add_mod(self.0.0, rhs.0.0)))
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "portable-only"), not(miri)))]
    {
      Self(Uint(x86_64::add_mod(self.0.0, &rhs.0.0)))
    }

    #[cfg(not(any(
      all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
      all(target_arch = "x86_64", not(feature = "portable-only"), not(miri))
    )))]
    {
      Self(self.0.add_mod(rhs.0))
    }
  }

  #[inline(always)]
  fn sub(self, rhs: Self) -> Self {
    #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
    {
      Self(Uint(aarch64::sub_mod(self.0.0, rhs.0.0)))
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "portable-only"), not(miri)))]
    {
      Self(Uint(x86_64::sub_mod(self.0.0, &rhs.0.0)))
    }

    #[cfg(not(any(
      all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
      all(target_arch = "x86_64", not(feature = "portable-only"), not(miri))
    )))]
    {
      Self(self.0.sub_mod(rhs.0))
    }
  }

  #[inline(always)]
  fn negate(self) -> Self {
    Self::ZERO.sub(self)
  }

  #[inline(always)]
  fn mul(self, rhs: Self) -> Self {
    Self(montgomery_mul(self.0, rhs.0))
  }

  #[inline(always)]
  fn square(self) -> Self {
    Self(montgomery_square(self.0))
  }

  #[inline(always)]
  #[cfg(any(
    test,
    not(any(
      all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
      all(target_arch = "x86_64", not(feature = "portable-only"), not(miri))
    ))
  ))]
  fn square_repeated(mut self, count: usize) -> Self {
    for _ in 0..count {
      self = self.square();
    }
    self
  }

  #[inline(always)]
  fn double(self) -> Self {
    self.add(self)
  }

  #[inline(always)]
  fn triple(self) -> Self {
    #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
    {
      Self(Uint(aarch64::mul_small(self.0.0, 3)))
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "portable-only"), not(miri)))]
    if has_bmi2_adx() {
      // SAFETY: `has_bmi2_adx` confirmed BMI2 on this CPU.
      return Self(Uint(unsafe { x86_64::mul_small_bmi2(self.0.0, 3) }));
    }

    #[cfg(not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri))))]
    {
      self.double().add(self)
    }
  }

  #[inline(always)]
  fn times4(self) -> Self {
    #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
    {
      Self(Uint(aarch64::mul_small(self.0.0, 4)))
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "portable-only"), not(miri)))]
    if has_bmi2_adx() {
      // SAFETY: `has_bmi2_adx` confirmed BMI2 on this CPU.
      return Self(Uint(unsafe { x86_64::mul_small_bmi2(self.0.0, 4) }));
    }

    #[cfg(not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri))))]
    {
      self.double().double()
    }
  }

  #[inline(always)]
  fn times8(self) -> Self {
    #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
    {
      Self(Uint(aarch64::mul_small(self.0.0, 8)))
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "portable-only"), not(miri)))]
    if has_bmi2_adx() {
      // SAFETY: `has_bmi2_adx` confirmed BMI2 on this CPU.
      return Self(Uint(unsafe { x86_64::mul_small_bmi2(self.0.0, 8) }));
    }

    #[cfg(not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri))))]
    {
      self.double().double().double()
    }
  }

  /// Return the inverse of a nonzero element and zero for zero.
  fn invert(self) -> Self {
    #[cfg(any(
      all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
      all(target_arch = "x86_64", not(feature = "portable-only"), not(miri))
    ))]
    {
      Self(Uint(divsteps::invert_montgomery(&self.0.0)))
    }

    #[cfg(not(any(
      all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
      all(target_arch = "x86_64", not(feature = "portable-only"), not(miri))
    )))]
    {
      self.invert_fermat()
    }
  }

  /// Return `self^(p - 2)`, the inverse of a nonzero element and zero for zero.
  #[cfg(any(
    test,
    not(any(
      all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
      all(target_arch = "x86_64", not(feature = "portable-only"), not(miri))
    ))
  ))]
  fn invert_fermat(self) -> Self {
    // Write x_k = self^(2^k - 1). The binary form of
    // p - 2 = 2^384 - 2^128 - 2^96 + 2^32 - 3 is, from the top:
    // 255 ones, one zero, 32 ones, 64 zeros, 30 ones, then 01.
    // This fixed chain uses 386 squarings and 14 multiplications.
    let x1 = self;
    let x2 = x1.square().mul(x1);
    let x3 = x2.square().mul(x1);
    let x6 = x3.square_repeated(3).mul(x3);
    let x12 = x6.square_repeated(6).mul(x6);
    let x15 = x12.square_repeated(3).mul(x3);
    let x30 = x15.square_repeated(15).mul(x15);
    let x32 = x30.square_repeated(2).mul(x2);
    let x60 = x30.square_repeated(30).mul(x30);
    let x120 = x60.square_repeated(60).mul(x60);
    let x240 = x120.square_repeated(120).mul(x120);
    let x255 = x240.square_repeated(15).mul(x15);
    x255
      .square_repeated(33)
      .mul(x32)
      .square_repeated(94)
      .mul(x30)
      .square_repeated(2)
      .mul(x1)
  }

  #[inline(always)]
  fn select(left: Self, right: Self, mask: u64) -> Self {
    Self(Uint::select(left.0, right.0, mask))
  }

  #[inline(always)]
  fn zero_mask(self) -> u64 {
    self.0.zero_mask()
  }
}

#[derive(Clone, Copy)]
struct Affine {
  x: FieldElement,
  y: FieldElement,
}

impl Affine {
  fn is_on_curve(self) -> bool {
    let lhs = self.y.square();
    let rhs = self
      .x
      .square()
      .mul(self.x)
      .sub(self.x.triple())
      .add(FieldElement::from_montgomery(CURVE_B_MONTGOMERY));
    lhs == rhs
  }

  fn encode_sec1(self) -> [u8; SEC1_BYTES] {
    let mut bytes = [0u8; SEC1_BYTES];
    bytes[0] = 0x04;
    let (x, y) = bytes[1..].split_at_mut(FIELD_BYTES);
    let mut coordinate = [0u8; FIELD_BYTES];
    self.x.to_uint().write_be(&mut coordinate);
    x.copy_from_slice(&coordinate);
    self.y.to_uint().write_be(&mut coordinate);
    y.copy_from_slice(&coordinate);
    bytes
  }

  /// Select a fixed-base comb entry and report whether `digit` is zero.
  ///
  /// Entry zero duplicates the generator; the returned mask marks it as the
  /// point at infinity.
  fn select_generator_comb(digit: usize) -> (Self, u64) {
    #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
    let (x, y) = {
      let (x, y) = aarch64::select_comb(&P384_SIGNING_GENERATOR_COMB_X, &P384_SIGNING_GENERATOR_COMB_Y, digit);
      (Uint(x), Uint(y))
    };
    #[cfg(not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri))))]
    let (x, y) = Self::select_generator_comb_limbs(digit);
    (
      Self {
        x: FieldElement::from_montgomery(x),
        y: FieldElement::from_montgomery(y),
      },
      mask_equal_usize(digit, 0),
    )
  }

  /// Portable full-table scan behind [`Self::select_generator_comb`].
  #[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
  fn select_generator_comb_limbs(digit: usize) -> (Uint, Uint) {
    let mut x = Uint::ZERO;
    let mut y = Uint::ZERO;
    for (index, (candidate_x, candidate_y)) in P384_SIGNING_GENERATOR_COMB_X
      .iter()
      .zip(P384_SIGNING_GENERATOR_COMB_Y.iter())
      .enumerate()
    {
      // SECURITY: Keep the equality mask opaque so LLVM retains the full
      // table scan instead of loading from a secret-derived address.
      let mask = core::hint::black_box(mask_equal_usize(digit, index));
      x.accumulate_masked(&candidate_x.0, mask);
      y.accumulate_masked(&candidate_y.0, mask);
    }
    (x, y)
  }
}

/// Jacobian point `(X / Z^2, Y / Z^3)`. Any point with `Z = 0` is infinity.
///
/// `repr(C)` over transparent limb arrays lays the point out as 18 contiguous
/// limbs, which the x86-64 in-place doubling addresses directly.
#[derive(Clone, Copy)]
#[repr(C)]
struct Jacobian {
  x: FieldElement,
  y: FieldElement,
  z: FieldElement,
}

impl Jacobian {
  const INFINITY: Self = Self {
    x: FieldElement::ZERO,
    y: FieldElement::ZERO,
    z: FieldElement::ZERO,
  };

  fn from_affine(point: Affine) -> Self {
    Self {
      x: point.x,
      y: point.y,
      z: FieldElement::ONE,
    }
  }

  #[inline(always)]
  fn infinity_mask(&self) -> u64 {
    self.z.zero_mask()
  }

  #[inline(always)]
  fn select(left: Self, right: Self, mask: u64) -> Self {
    Self {
      x: FieldElement::select(left.x, right.x, mask),
      y: FieldElement::select(left.y, right.y, mask),
      z: FieldElement::select(left.z, right.z, mask),
    }
  }

  /// Return the double of this point; see [`Self::double_formula`].
  fn double(self) -> Self {
    let mut point = self;
    point.double_in_place();
    point
  }

  /// Replace this point with its double.
  #[inline(always)]
  fn double_in_place(&mut self) {
    #[cfg(all(target_arch = "x86_64", not(feature = "portable-only"), not(miri)))]
    if has_bmi2_adx() {
      const _: () = assert!(size_of::<Jacobian>() == 3 * LIMBS * 8 && align_of::<Jacobian>() == 8);
      // SAFETY: `has_bmi2_adx` confirmed BMI2 and ADX. `Jacobian` is
      // `repr(C)` over three `repr(transparent)` six-limb arrays, so it is
      // exactly 18 initialized, 8-byte-aligned limbs, and the exclusive borrow
      // of `self` covers the whole cast.
      unsafe { x86_64::point_double_bmi2_adx(&mut *core::ptr::from_mut(self).cast::<[u64; 3 * LIMBS]>()) };
      return;
    }
    *self = self.double_formula();
  }

  /// Double with `a = -3` (dbl-2001-b, 3M + 5S). This is the portable
  /// authority; the x86-64 BMI2/ADX block computes the same polynomials.
  ///
  /// P-384 has prime order, so no finite point has `Y = 0`. Infinity maps to
  /// `Z3 = 2 * Y * Z = 0`, so this formula has no exceptional inputs.
  fn double_formula(self) -> Self {
    // Statements alternate the critical delta -> alpha -> x -> y chain with
    // independent products so adjacent kernels can overlap.
    let delta = self.z.square();
    let gamma = self.y.square();
    let alpha = self.x.sub(delta).mul(self.x.add(delta));
    let beta = self.x.mul(gamma);
    let alpha = alpha.triple();
    let alpha_squared = alpha.square();
    let yz_squared = self.y.add(self.z).square();
    let beta4 = beta.times4();
    let x = alpha_squared.sub(beta.times8());
    let gamma_squared = gamma.square();
    let y = alpha.mul(beta4.sub(x));
    let z = yz_squared.sub(gamma).sub(delta);
    let y = y.sub(gamma_squared.times8());
    Self { x, y, z }
  }

  /// Add a cached table entry (add-2007-bl with cached `Z2^2` and `Z2^3`,
  /// 10M + 4S).
  ///
  /// Returns the formula result and a mask that is set when `U1 = U2` and
  /// `S1 = S2`. The result is correct for distinct finite points, including
  /// opposites, which yield `Z3 = 0`. Callers select the result for infinity
  /// operands and, when reachable, equal operands.
  #[inline(always)]
  fn add_formula(self, rhs: CachedJacobian) -> (Self, u64) {
    // Statements alternate the critical u -> h -> i -> j -> x -> y chain
    // with independent products so adjacent kernels can overlap.
    let z1z1 = self.z.square();
    let u1 = self.x.mul(rhs.zz);
    let s1 = self.y.mul(rhs.zzz);
    let u2 = rhs.point.x.mul(z1z1);
    let z1_cubed = self.z.mul(z1z1);
    let h = u2.sub(u1);
    let s2 = rhs.point.y.mul(z1_cubed);
    let i = h.double().square();
    let z = self.z.add(rhs.point.z).square();
    let j = h.mul(i);
    let v = u1.mul(i);
    let r = s2.sub(s1).double();
    let r_squared = r.square();
    let s1j = s1.mul(j);
    let x = r_squared.sub(j).sub(v.double());
    let z = z.sub(z1z1).sub(rhs.zz).mul(h);
    let y = r.mul(v.sub(x)).sub(s1j.double());
    (Self { x, y, z }, h.zero_mask() & r.zero_mask())
  }

  /// Add points that are known not to be equal finite points.
  #[inline(always)]
  fn add_distinct(self, rhs: CachedJacobian) -> Self {
    let (sum, _) = self.add_formula(rhs);
    let sum = Self::select(sum, rhs.point, self.infinity_mask());
    Self::select(sum, self, rhs.point.infinity_mask())
  }

  /// Add any two points.
  fn add_complete(self, rhs: CachedJacobian) -> Self {
    let (sum, equal) = self.add_formula(rhs);
    let finite = !(self.infinity_mask() | rhs.point.infinity_mask());
    let sum = Self::select(sum, self.double(), equal & finite);
    let sum = Self::select(sum, rhs.point, self.infinity_mask());
    Self::select(sum, self, rhs.point.infinity_mask())
  }

  /// Add a finite affine point (madd-2007-bl, 7M + 4S) when the finite
  /// operands are known to be neither equal nor opposite.
  #[inline(always)]
  fn add_mixed_formula(self, rhs: Affine) -> Self {
    let z1z1 = self.z.square();
    let u2 = rhs.x.mul(z1z1);
    let s2 = rhs.y.mul(self.z);
    let h = u2.sub(self.x);
    let hh = h.square();
    let s2 = s2.mul(z1z1);
    let i = hh.times4();
    let z = self.z.add(h).square();
    let j = h.mul(i);
    let v = self.x.mul(i);
    let r = s2.sub(self.y).double();
    let r_squared = r.square();
    let y1j = self.y.mul(j);
    let x = r_squared.sub(j).sub(v.double());
    let y = r.mul(v.sub(x)).sub(y1j.double());
    let z = z.sub(z1z1).sub(hh);
    Self { x, y, z }
  }

  /// Mixed addition that also handles either operand being infinity.
  #[inline(always)]
  fn add_mixed_distinct(self, rhs: Affine, rhs_infinity_mask: u64) -> Self {
    let sum = Self::select(
      self.add_mixed_formula(rhs),
      Self::from_affine(rhs),
      self.infinity_mask(),
    );
    Self::select(sum, self, rhs_infinity_mask)
  }

  fn to_affine(self) -> Affine {
    let z_inverse = self.z.invert();
    let z_inverse_squared = z_inverse.square();
    Affine {
      x: self.x.mul(z_inverse_squared),
      y: self.y.mul(z_inverse_squared).mul(z_inverse),
    }
  }

  fn affine_x(self) -> FieldElement {
    self.x.mul(self.z.invert().square())
  }
}

/// Window table entry: a Jacobian point with cached `Z^2` and `Z^3`.
#[derive(Clone, Copy)]
struct CachedJacobian {
  point: Jacobian,
  zz: FieldElement,
  zzz: FieldElement,
}

impl CachedJacobian {
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  const INFINITY: Self = Self {
    point: Jacobian::INFINITY,
    zz: FieldElement::ZERO,
    zzz: FieldElement::ZERO,
  };

  fn new(point: Jacobian) -> Self {
    let zz = point.z.square();
    Self {
      point,
      zz,
      zzz: zz.mul(point.z),
    }
  }

  /// Select `sign(digit) * table[|digit| - 1]`, or infinity for digit zero.
  fn select_signed(table: &[Self; WINDOW_TABLE_SIZE], digit: u8) -> Self {
    let sign = 0u8.wrapping_sub(digit >> 7);
    let magnitude = usize::from((digit ^ sign).wrapping_sub(sign));
    // Start from the masked first entry instead of zero. AArch64 LLVM keeps
    // the accumulators in SIMD registers and would copy zero into each with
    // `fmov`, which BINSEC cannot interpret.
    // SECURITY: Keep every equality mask opaque so LLVM retains the full
    // table scan instead of loading from a secret-derived address.
    let first = &table[0];
    let mask = core::hint::black_box(mask_equal_usize(magnitude, 1));
    let mut x = Uint::masked(&first.point.x.0.0, mask);
    let mut y = Uint::masked(&first.point.y.0.0, mask);
    let mut z = Uint::masked(&first.point.z.0.0, mask);
    let mut zz = Uint::masked(&first.zz.0.0, mask);
    let mut zzz = Uint::masked(&first.zzz.0.0, mask);
    for (candidate, entry) in table[1..].iter().zip(2..) {
      let mask = core::hint::black_box(mask_equal_usize(magnitude, entry));
      x.accumulate_masked(&candidate.point.x.0.0, mask);
      y.accumulate_masked(&candidate.point.y.0.0, mask);
      z.accumulate_masked(&candidate.point.z.0.0, mask);
      zz.accumulate_masked(&candidate.zz.0.0, mask);
      zzz.accumulate_masked(&candidate.zzz.0.0, mask);
    }
    let y = FieldElement::from_montgomery(y);
    // SECURITY: Keep the sign mask opaque. Otherwise LLVM can lower this
    // select to a branch on the digit's sign bit.
    let negate = core::hint::black_box(0u64.wrapping_sub(u64::from(sign & 1)));
    Self {
      point: Jacobian {
        x: FieldElement::from_montgomery(x),
        y: FieldElement::select(y, y.negate(), negate),
        z: FieldElement::from_montgomery(z),
      },
      zz: FieldElement::from_montgomery(zz),
      zzz: FieldElement::from_montgomery(zzz),
    }
  }
}

/// Secret-dependent Jacobian accumulator that clears its coordinates on drop.
struct SecretJacobian(Jacobian);

impl Drop for SecretJacobian {
  fn drop(&mut self) {
    self.0.x.0.zeroize_no_fence();
    self.0.y.0.zeroize_no_fence();
    self.0.z.0.zeroize_no_fence();
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

/// Multiply the generator with the fixed-base signing comb.
fn scalar_mul_generator(scalar: &Scalar) -> SecretJacobian {
  let mut acc = SecretJacobian(Jacobian::INFINITY);
  for row in (0..COMB_ROWS).rev() {
    acc.0.double_in_place();
    let (selected, infinity) = Affine::select_generator_comb(scalar.comb_digit(row));
    // The comb writes the scalar as sum(2^row * S_row), where S_row has bits
    // only at positions `column * 48`. After the loop double, the accumulator
    // coefficient has bits only at `column * 48 + offset` with
    // 1 <= offset < 48, so for a nonzero digit it differs from S_row as an
    // integer. Both are below the prime group order, so they are not equal
    // modulo the order. Opposites would require their sum, which is a prefix
    // of the canonical scalar, to equal the order.
    acc.0 = acc.0.add_mixed_distinct(selected, infinity);
  }
  acc
}

/// Precompute `1P..=16P` for a validated finite public point.
fn precompute_window_table(point: Affine) -> [CachedJacobian; WINDOW_TABLE_SIZE] {
  let base = Jacobian::from_affine(point);
  let mut multiples = [base; WINDOW_TABLE_SIZE];
  multiples[1] = base.double();
  for index in 2..WINDOW_TABLE_SIZE {
    // `index * P + P` with 2 <= index < 16 never meets an exceptional case
    // in a group of prime order far above 17.
    multiples[index] = multiples[index.strict_sub(1)].add_mixed_formula(point);
  }
  multiples.map(CachedJacobian::new)
}

/// Multiply a public point by a secret scalar with a fixed signed window.
fn scalar_mul_window(scalar: &Scalar, table: &[CachedJacobian; WINDOW_TABLE_SIZE]) -> SecretJacobian {
  let digits = scalar.signed_window_digits();
  let mut acc = SecretJacobian(CachedJacobian::select_signed(table, digits.0[WINDOW_DIGITS - 1]).point);
  for row in (0..WINDOW_DIGITS - 1).rev() {
    for _ in 0..WINDOW_BITS {
      acc.0.double_in_place();
    }
    let selected = CachedJacobian::select_signed(table, digits.0[row]);
    if row == 0 {
      acc.0 = acc.0.add_complete(selected);
    } else {
      // Before digit `row`, the accumulator is m * P with m = 32 * S, where S
      // is the recoded prefix and S <= floor(k / 32^(row + 1)) + 1. For
      // row >= 1, 0 <= m < n / 32 + 32, and a nonzero m is at least 32. A
      // selected digit has magnitude at most 16, so m cannot equal it or its
      // negation modulo n unless an operand is infinity. Only the final
      // addition can meet equal operands.
      acc.0 = acc.0.add_distinct(selected);
    }
  }
  acc
}

/// Validated finite public point on P-384.
#[derive(Clone, Copy)]
pub(super) struct PublicPoint(Affine);

impl PublicPoint {
  /// Parse canonical uncompressed SEC1 and reject off-curve points.
  pub(super) fn from_sec1_bytes(bytes: &[u8]) -> Option<Self> {
    if bytes.len() != SEC1_BYTES || bytes.first().copied() != Some(0x04) {
      return None;
    }
    let (x_bytes, y_bytes) = bytes.get(1..)?.split_at(FIELD_BYTES);
    let x = Uint::from_be_bytes(x_bytes.try_into().ok()?);
    let y = Uint::from_be_bytes(y_bytes.try_into().ok()?);
    if x.cmp_vartime(&FIELD_MODULUS).is_ge() || y.cmp_vartime(&FIELD_MODULUS).is_ge() {
      return None;
    }
    let point = Affine {
      x: FieldElement::from_uint(x),
      y: FieldElement::from_uint(y),
    };
    point.is_on_curve().then_some(Self(point))
  }

  pub(super) fn to_sec1_bytes(self) -> [u8; SEC1_BYTES] {
    self.0.encode_sec1()
  }
}

/// Return whether `bytes` encode a scalar in `[1, n - 1]`.
///
/// This comparison is variable time. It is used only for rejection sampling,
/// whose outcome is outside the constant-time claim.
pub(super) fn scalar_is_canonical_nonzero(bytes: &[u8; FIELD_BYTES]) -> bool {
  let candidate = Uint::from_be_bytes(bytes);
  candidate.zero_mask() == 0 && candidate.cmp_vartime(&SCALAR_MODULUS).is_lt()
}

/// Derive the public point for a canonical nonzero scalar.
pub(super) fn public_key_from_scalar(bytes: &[u8; FIELD_BYTES]) -> PublicPoint {
  let scalar = Scalar::from_bytes(bytes);
  PublicPoint(scalar_mul_generator(&scalar).0.to_affine())
}

/// Write the ECC CDH x-coordinate for a canonical nonzero scalar.
pub(super) fn agree(bytes: &[u8; FIELD_BYTES], public: PublicPoint, shared: &mut [u8; FIELD_BYTES]) {
  let table = precompute_window_table(public.0);
  let scalar = Scalar::from_bytes(bytes);
  let product = scalar_mul_window(&scalar, &table);
  let mut x = product.0.affine_x().to_uint();
  x.write_be(shared);
  x.zeroize_no_fence();
}

/// Return the production fixed-base comb selection as Montgomery limbs.
#[cfg(all(rscrypto_internal, feature = "diag"))]
pub(super) fn diag_select_generator_limb_digest(digit: u8) -> [u64; 2 * LIMBS] {
  let (selected, _) = Affine::select_generator_comb(usize::from(digit));
  let mut output = [0u64; 2 * LIMBS];
  output[..LIMBS].copy_from_slice(&selected.x.0.0);
  output[LIMBS..].copy_from_slice(&selected.y.0.0);
  output
}

/// Return the production signed-window selection over a fixed public table.
///
/// The table holds distinct canonical limb patterns rather than curve points;
/// the evidence concerns only the selector's data flow.
#[cfg(all(rscrypto_internal, feature = "diag"))]
pub(super) fn diag_select_window_limb_digest(digit: u8) -> [u64; 5 * LIMBS] {
  let mut table = [CachedJacobian::INFINITY; WINDOW_TABLE_SIZE];
  for (index, entry) in (1u64..).zip(table.iter_mut()) {
    let pattern = |lane: u64| FieldElement::from_montgomery(Uint([index.strict_mul(lane); LIMBS]));
    *entry = CachedJacobian {
      point: Jacobian {
        x: pattern(0x0101),
        y: pattern(0x0303),
        z: pattern(0x0505),
      },
      zz: pattern(0x0707),
      zzz: pattern(0x0909),
    };
  }
  let selected = CachedJacobian::select_signed(&table, digit);
  let mut output = [0u64; 5 * LIMBS];
  for (chunk, coordinate) in output.chunks_exact_mut(LIMBS).zip([
    selected.point.x,
    selected.point.y,
    selected.point.z,
    selected.zz,
    selected.zzz,
  ]) {
    chunk.copy_from_slice(&coordinate.0.0);
  }
  output
}

#[inline(always)]
fn mask_nonzero(value: u64) -> u64 {
  0u64.wrapping_sub((value | value.wrapping_neg()) >> 63)
}

#[inline(always)]
fn mask_zero(value: u64) -> u64 {
  !mask_nonzero(value)
}

#[inline(always)]
fn mask_equal_usize(left: usize, right: usize) -> u64 {
  mask_zero((left ^ right) as u64)
}

#[cfg(target_arch = "s390x")]
#[inline(always)]
fn low_u32(value: u64) -> u32 {
  let [b0, b1, b2, b3, _, _, _, _] = value.to_le_bytes();
  u32::from_le_bytes([b0, b1, b2, b3])
}

#[cfg(target_arch = "s390x")]
#[inline(always)]
fn low_u32_signed(value: i64) -> u32 {
  let [b0, b1, b2, b3, _, _, _, _] = value.to_le_bytes();
  u32::from_le_bytes([b0, b1, b2, b3])
}

#[inline(always)]
fn adc_limb(left: u64, right: u64, carry: u64) -> (u64, u64) {
  #[cfg(target_arch = "s390x")]
  {
    // Split limbs so s390x carry extraction stays arithmetic.
    let low = u64::from(low_u32(left))
      .strict_add(u64::from(low_u32(right)))
      .strict_add(carry);
    let high = (left >> 32).strict_add(right >> 32).strict_add(low >> 32);
    (u64::from(low_u32(low)) | (u64::from(low_u32(high)) << 32), high >> 32)
  }

  #[cfg(not(target_arch = "s390x"))]
  {
    let (sum, carry0) = left.overflowing_add(right);
    let (sum, carry1) = sum.overflowing_add(carry);
    (sum, u64::from(carry0 | carry1))
  }
}

#[inline(always)]
fn sbb_limb(left: u64, right: u64, borrow: u64) -> (u64, u64) {
  #[cfg(target_arch = "s390x")]
  {
    let low = i64::from(low_u32(left))
      .strict_sub(i64::from(low_u32(right)))
      .strict_sub(i64::from(low_u32(borrow)));
    let low_borrow = (low >> 63) & 1;
    let high = i64::from(low_u32(left >> 32))
      .strict_sub(i64::from(low_u32(right >> 32)))
      .strict_sub(low_borrow);
    (
      u64::from(low_u32_signed(low)) | (u64::from(low_u32_signed(high)) << 32),
      u64::from(low_u32_signed((high >> 63) & 1)),
    )
  }

  #[cfg(not(target_arch = "s390x"))]
  {
    let (difference, borrow0) = left.overflowing_sub(right);
    let (difference, borrow1) = difference.overflowing_sub(borrow);
    (difference, u64::from(borrow0 | borrow1))
  }
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "s390x"))]
#[inline(never)]
fn ct_mul_u64_wide(left: u64, right: u64) -> (u64, u64) {
  let mut product_low = 0u64;
  let mut product_high = 0u64;
  let mut multiplicand_low = left;
  let mut multiplicand_high = 0u64;
  let mut multiplier = right;
  for _ in 0..(u64::BITS / 4) {
    // These targets lower ordinary wide multiplication through instructions or
    // helpers whose latency is operand-dependent. Keep the radix-16 authority
    // fixed-work and prevent LLVM from recognizing it as an ordinary product.
    let digit = core::hint::black_box(multiplier & 0xf);
    for bit in 0..4 {
      let mask = 0u64.wrapping_sub((digit >> bit) & 1);
      let (next_low, carry) = adc_limb(product_low, multiplicand_low & mask, 0);
      let (next_high, _) = adc_limb(product_high, multiplicand_high & mask, carry);
      product_low = next_low;
      product_high = next_high;
      multiplicand_high = (multiplicand_high << 1) | (multiplicand_low >> 63);
      multiplicand_low <<= 1;
    }
    multiplier >>= 4;
  }
  (product_low, product_high)
}

/// RV64 multiplicand with its top bit forced set.
///
/// Some RV64 multipliers finish early for operands with leading zeros.
/// Multiplying normalized operands and subtracting the correction terms keeps
/// the hardware multiply operand width fixed.
#[cfg(any(test, target_arch = "riscv64"))]
#[derive(Clone, Copy)]
struct Riscv64MulLimb {
  normalized: u64,
  padding_mask: u64,
  shift_low: u64,
  shift_high: u64,
}

#[cfg(any(test, target_arch = "riscv64"))]
impl Riscv64MulLimb {
  #[inline(always)]
  fn new(value: u64) -> Self {
    const HIGH_BIT: u64 = 1 << 63;
    let padding_mask = 0u64.wrapping_sub((value >> 63) ^ 1);
    Self {
      normalized: core::hint::black_box(value | HIGH_BIT),
      padding_mask,
      shift_low: value << 63,
      shift_high: value >> 1,
    }
  }
}

#[cfg(any(test, target_arch = "riscv64"))]
#[inline(always)]
fn ct_mul_riscv64_limbs(left: Riscv64MulLimb, right: Riscv64MulLimb) -> (u64, u64) {
  const HIGH_BIT: u64 = 1 << 63;

  let product = u128::from(left.normalized).strict_mul(u128::from(right.normalized));
  let (product_low, product_high) = split_u128(product);

  // Subtract the two conditional 2^63 cross terms and their 2^126
  // intersection to recover the original product.
  let (product_low, borrow) = sbb_limb(product_low, right.shift_low & left.padding_mask, 0);
  let (product_high, borrow) = sbb_limb(product_high, right.shift_high & left.padding_mask, borrow);
  debug_assert_eq!(borrow, 0);
  let (product_low, borrow) = sbb_limb(product_low, left.shift_low & right.padding_mask, 0);
  let (product_high, borrow) = sbb_limb(product_high, left.shift_high & right.padding_mask, borrow);
  debug_assert_eq!(borrow, 0);
  let intersection = (HIGH_BIT >> 1) & left.padding_mask & right.padding_mask;
  let (product_high, borrow) = sbb_limb(product_high, intersection, 0);
  debug_assert_eq!(borrow, 0);

  (product_low, product_high)
}

#[inline(always)]
#[cfg(not(target_arch = "riscv64"))]
#[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
fn mul_u64_wide(left: u64, right: u64) -> (u64, u64) {
  #[cfg(any(target_arch = "riscv32", target_arch = "s390x"))]
  {
    ct_mul_u64_wide(left, right)
  }

  #[cfg(not(any(target_arch = "riscv32", target_arch = "s390x")))]
  {
    split_u128(u128::from(left).strict_mul(u128::from(right)))
  }
}

#[inline(always)]
#[cfg(any(test, not(any(target_arch = "riscv32", target_arch = "s390x"))))]
#[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
fn split_u128(value: u128) -> (u64, u64) {
  let [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15] = value.to_le_bytes();
  (
    u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]),
    u64::from_le_bytes([b8, b9, b10, b11, b12, b13, b14, b15]),
  )
}

/// Return `acc + product + carry` as a low limb and a high limb.
#[inline(always)]
#[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
fn mac_wide(acc: u64, product_low: u64, product_high: u64, carry: u64) -> (u64, u64) {
  let (result, carry0) = adc_limb(product_low, acc, 0);
  let (result, carry1) = adc_limb(result, carry, 0);
  let (high, overflow0) = adc_limb(product_high, carry0, 0);
  let (high, overflow1) = adc_limb(high, carry1, 0);
  debug_assert_eq!(overflow0 | overflow1, 0);
  (result, high)
}

#[inline(always)]
#[cfg(not(target_arch = "riscv64"))]
#[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
fn mac_limb(acc: u64, left: u64, right: u64, carry: u64) -> (u64, u64) {
  let (product_low, product_high) = mul_u64_wide(left, right);
  mac_wide(acc, product_low, product_high, carry)
}

#[cfg(any(test, target_arch = "riscv64"))]
#[inline(always)]
fn mac_riscv64_limb(acc: u64, left: Riscv64MulLimb, right: Riscv64MulLimb, carry: u64) -> (u64, u64) {
  let (product_low, product_high) = ct_mul_riscv64_limbs(left, right);
  mac_wide(acc, product_low, product_high, carry)
}

#[cfg(any(test, target_arch = "riscv64"))]
#[inline(always)]
fn riscv64_mul_limbs(value: Uint) -> [Riscv64MulLimb; LIMBS] {
  value.0.map(Riscv64MulLimb::new)
}

/// Accumulate `left * right` into `wide[ROW..=ROW + 6]`, overwriting the top limb.
///
/// Rows, squaring passes, and reduction steps take their position as a const
/// parameter so every accumulator index is a compile-time constant. LLVM then
/// keeps the accumulator in registers instead of carrying a rolled loop
/// through stack memory.
#[inline(always)]
#[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
fn multiply_row<T: Copy, const ROW: usize>(
  wide: &mut [u64; WIDE_LIMBS],
  left: T,
  right: &[T; LIMBS],
  mac: &impl Fn(u64, T, T, u64) -> (u64, u64),
) {
  let mut carry = 0u64;
  for (column, &right) in right.iter().enumerate() {
    let index = ROW.strict_add(column);
    (wide[index], carry) = mac(wide[index], left, right, carry);
  }
  wide[ROW.strict_add(LIMBS)] = carry;
}

/// Schoolbook 384 x 384 -> 768-bit product.
#[inline(always)]
#[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
fn multiply_wide<T: Copy>(
  left: [T; LIMBS],
  right: [T; LIMBS],
  mac: impl Fn(u64, T, T, u64) -> (u64, u64),
) -> [u64; WIDE_LIMBS] {
  let mut wide = [0u64; WIDE_LIMBS];
  multiply_row::<T, 0>(&mut wide, left[0], &right, &mac);
  multiply_row::<T, 1>(&mut wide, left[1], &right, &mac);
  multiply_row::<T, 2>(&mut wide, left[2], &right, &mac);
  multiply_row::<T, 3>(&mut wide, left[3], &right, &mac);
  multiply_row::<T, 4>(&mut wide, left[4], &right, &mac);
  multiply_row::<T, 5>(&mut wide, left[5], &right, &mac);
  wide
}

/// Accumulate the cross products `value[ROW] * value[ROW + 1..]`.
#[inline(always)]
#[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
fn square_cross_row<T: Copy, const ROW: usize>(
  wide: &mut [u64; WIDE_LIMBS],
  value: &[T; LIMBS],
  mac: &impl Fn(u64, T, T, u64) -> (u64, u64),
) {
  let mut carry = 0u64;
  for column in ROW.strict_add(1)..LIMBS {
    let index = ROW.strict_add(column);
    (wide[index], carry) = mac(wide[index], value[ROW], value[column], carry);
  }
  wide[ROW.strict_add(LIMBS)] = carry;
}

/// 384-bit square with 15 cross products and 6 diagonal products.
#[inline(always)]
#[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
fn square_wide<T: Copy>(
  value: [T; LIMBS],
  mul: impl Fn(T, T) -> (u64, u64),
  mac: impl Fn(u64, T, T, u64) -> (u64, u64),
) -> [u64; WIDE_LIMBS] {
  let mut wide = [0u64; WIDE_LIMBS];
  square_cross_row::<T, 0>(&mut wide, &value, &mac);
  square_cross_row::<T, 1>(&mut wide, &value, &mac);
  square_cross_row::<T, 2>(&mut wide, &value, &mac);
  square_cross_row::<T, 3>(&mut wide, &value, &mac);
  square_cross_row::<T, 4>(&mut wide, &value, &mac);

  // The cross-product sum is below 2^767, so doubling cannot overflow.
  let mut top = 0u64;
  for limb in &mut wide {
    let next = *limb >> 63;
    *limb = (*limb << 1) | top;
    top = next;
  }
  debug_assert_eq!(top, 0);

  let mut carry = 0u64;
  for (index, &limb) in value.iter().enumerate() {
    let (low, high) = mul(limb, limb);
    let low_index = index.strict_mul(2);
    let high_index = low_index.strict_add(1);
    (wide[low_index], carry) = adc_limb(wide[low_index], low, carry);
    (wide[high_index], carry) = adc_limb(wide[high_index], high, carry);
  }
  debug_assert_eq!(carry, 0);
  wide
}

#[inline(always)]
fn montgomery_mul(left: Uint, right: Uint) -> Uint {
  #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
  {
    Uint(aarch64::montgomery_mul(left.0, right.0))
  }

  #[cfg(all(target_arch = "x86_64", not(feature = "portable-only"), not(miri)))]
  {
    if has_bmi2_adx() {
      // SAFETY: `has_bmi2_adx` confirmed BMI2 and ADX on this CPU.
      return Uint(unsafe { x86_64::montgomery_mul_bmi2_adx(&left.0, &right.0) });
    }
    montgomery_mul_portable(left, right)
  }

  #[cfg(not(any(
    all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
    all(target_arch = "x86_64", not(feature = "portable-only"), not(miri))
  )))]
  {
    montgomery_mul_portable(left, right)
  }
}

/// Return whether the cached CPU capabilities include BMI2 and ADX.
#[cfg(all(target_arch = "x86_64", not(feature = "portable-only"), not(miri)))]
#[inline(always)]
fn has_bmi2_adx() -> bool {
  use crate::platform::caps::x86;
  // A build that already requires BMI2 and ADX folds the check away.
  cfg!(all(target_feature = "bmi2", target_feature = "adx")) || crate::platform::caps().has(x86::BMI2.union(x86::ADX))
}

#[inline(always)]
fn montgomery_square(value: Uint) -> Uint {
  #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
  {
    Uint(aarch64::montgomery_square(value.0))
  }

  #[cfg(all(target_arch = "x86_64", not(feature = "portable-only"), not(miri)))]
  {
    if has_bmi2_adx() {
      // SAFETY: `has_bmi2_adx` confirmed BMI2 and ADX on this CPU.
      return Uint(unsafe { x86_64::montgomery_square_bmi2_adx(&value.0) });
    }
    montgomery_square_portable(value)
  }

  #[cfg(not(any(
    all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
    all(target_arch = "x86_64", not(feature = "portable-only"), not(miri))
  )))]
  {
    montgomery_square_portable(value)
  }
}

#[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
fn montgomery_mul_portable(left: Uint, right: Uint) -> Uint {
  #[cfg(target_arch = "riscv64")]
  {
    montgomery_reduce(multiply_wide(
      riscv64_mul_limbs(left),
      riscv64_mul_limbs(right),
      mac_riscv64_limb,
    ))
  }

  #[cfg(not(target_arch = "riscv64"))]
  {
    montgomery_reduce(multiply_wide(left.0, right.0, mac_limb))
  }
}

#[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
fn montgomery_square_portable(value: Uint) -> Uint {
  #[cfg(target_arch = "riscv64")]
  {
    montgomery_reduce(square_wide(
      riscv64_mul_limbs(value),
      ct_mul_riscv64_limbs,
      mac_riscv64_limb,
    ))
  }

  #[cfg(not(target_arch = "riscv64"))]
  {
    montgomery_reduce(square_wide(value.0, mul_u64_wide, mac_limb))
  }
}

/// Return `q * p` for the Montgomery quotient `q = limb * (2^32 + 1) mod 2^64`.
///
/// With `c = 2^128 + 2^96 - 2^32 + 1`, `q * p = q * 2^384 - q * c`, and
/// `q * c = q * 2^32 * (2^64 + 2^96) - q * (2^32 - 1)`. The low result limb
/// is `-limb mod 2^64`, which clears the reduced limb.
#[inline(always)]
fn quotient_times_modulus(limb: u64) -> [u64; LIMBS + 1] {
  let quotient = limb.wrapping_add(limb << 32);
  let shifted_low = quotient << 32;
  let shifted_high = quotient >> 32;
  // A = q * (2^32 - 1) is nonnegative, so its high limb cannot underflow.
  let (a0, borrow) = sbb_limb(shifted_low, quotient, 0);
  let a1 = shifted_high.wrapping_sub(borrow);
  // L = q * 2^32 * (2^64 + 2^96) has limbs [0, shifted_low, l2, l3].
  let (l2, l3) = adc_limb(quotient, shifted_high, 0);
  // q * p = q * 2^384 + A - L; the complete value is nonnegative.
  let (m1, borrow) = sbb_limb(a1, shifted_low, 0);
  let (m2, borrow) = sbb_limb(0, l2, borrow);
  let (m3, borrow) = sbb_limb(0, l3, borrow);
  let fill = 0u64.wrapping_sub(borrow);
  let m6 = quotient.wrapping_sub(borrow);
  [a0, m1, m2, m3, fill, fill, m6]
}

/// Clear `limbs[STEP]` by adding `q * p * 2^(64 * STEP)`.
///
/// `pending` carries the overflow of `limbs[STEP + 6]` into the next step.
#[inline(always)]
fn montgomery_reduce_step<const STEP: usize>(limbs: &mut [u64; WIDE_LIMBS], pending: &mut u64) {
  let product = quotient_times_modulus(limbs[STEP]);
  let mut carry = 0u64;
  for (offset, &term) in product.iter().enumerate().take(LIMBS) {
    let index = STEP.strict_add(offset);
    (limbs[index], carry) = adc_limb(limbs[index], term, carry);
  }
  debug_assert_eq!(limbs[STEP], 0);
  let top = STEP.strict_add(LIMBS);
  let (limb, carry0) = adc_limb(limbs[top], product[LIMBS], carry);
  let (limb, carry1) = adc_limb(limb, *pending, 0);
  limbs[top] = limb;
  *pending = carry0.strict_add(carry1);
}

/// Montgomery-reduce a value below `p * R` to a canonical field element.
#[inline(always)]
fn montgomery_reduce(wide: [u64; WIDE_LIMBS]) -> Uint {
  let mut limbs = wide;
  let mut pending = 0u64;
  montgomery_reduce_step::<0>(&mut limbs, &mut pending);
  montgomery_reduce_step::<1>(&mut limbs, &mut pending);
  montgomery_reduce_step::<2>(&mut limbs, &mut pending);
  montgomery_reduce_step::<3>(&mut limbs, &mut pending);
  montgomery_reduce_step::<4>(&mut limbs, &mut pending);
  montgomery_reduce_step::<5>(&mut limbs, &mut pending);
  let mut reduced = [0u64; LIMBS];
  reduced.copy_from_slice(&limbs[LIMBS..]);
  subtract_modulus_once(Uint(reduced), pending)
}

/// Reduce a value below `2p`, given as six limbs plus a high bit.
#[inline(always)]
fn subtract_modulus_once(value: Uint, high: u64) -> Uint {
  debug_assert!(high <= 1);
  let (reduced, borrow) = value.sub_raw(FIELD_MODULUS);
  // A set high bit always authorizes subtraction.
  let keep = 0u64.wrapping_sub(borrow & (high ^ 1));
  reduced.add_raw(Uint::masked_field_modulus(keep)).0
}

#[cfg(all(test, feature = "p384-ecdh"))]
mod tests {
  use super::{Affine, CachedJacobian, FieldElement, Jacobian, Scalar, Uint};

  const GENERATOR_X: Uint = Uint([
    0x3a54_5e38_7276_0ab7,
    0x5502_f25d_bf55_296c,
    0x59f7_41e0_8254_2a38,
    0x6e1d_3b62_8ba7_9b98,
    0x8eb1_c71e_f320_ad74,
    0xaa87_ca22_be8b_0537,
  ]);
  const GENERATOR_Y: Uint = Uint([
    0x7a43_1d7c_90ea_0e5f,
    0x0a60_b1ce_1d7e_819d,
    0xe9da_3113_b5f0_b8c0,
    0xf8f4_1dbd_289a_147c,
    0x5d9e_98bf_9292_dc29,
    0x3617_de4a_9626_2c6f,
  ]);
  const P_MINUS_ONE: Uint = Uint([
    0x0000_0000_ffff_fffe,
    0xffff_ffff_0000_0000,
    0xffff_ffff_ffff_fffe,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
  ]);

  fn generator() -> Affine {
    Affine {
      x: FieldElement::from_uint(GENERATOR_X),
      y: FieldElement::from_uint(GENERATOR_Y),
    }
  }

  #[test]
  fn target_shaped_wide_multiply_matches_u128() {
    let edges = [
      0,
      1,
      0x5555_5555_5555_5555,
      0x7fff_ffff_ffff_ffff,
      0x8000_0000_0000_0000,
      0xaaaa_aaaa_aaaa_aaaa,
      u64::MAX - 1,
      u64::MAX,
    ];
    for left in edges {
      for right in edges {
        let expected = super::split_u128(u128::from(left).strict_mul(u128::from(right)));
        assert_eq!(super::ct_mul_u64_wide(left, right), expected);
        assert_eq!(
          super::ct_mul_riscv64_limbs(super::Riscv64MulLimb::new(left), super::Riscv64MulLimb::new(right)),
          expected
        );
      }
    }

    for left in [Uint::ZERO, GENERATOR_X, GENERATOR_Y, P_MINUS_ONE] {
      let riscv_square = super::montgomery_reduce(super::square_wide(
        super::riscv64_mul_limbs(left),
        super::ct_mul_riscv64_limbs,
        super::mac_riscv64_limb,
      ));
      assert!(riscv_square == super::montgomery_square(left));
      for right in [Uint::ZERO, GENERATOR_X, GENERATOR_Y, P_MINUS_ONE] {
        let riscv_product = super::montgomery_reduce(super::multiply_wide(
          super::riscv64_mul_limbs(left),
          super::riscv64_mul_limbs(right),
          super::mac_riscv64_limb,
        ));
        assert!(riscv_product == super::montgomery_mul(left, right));
      }
    }
  }

  /// Compare the selected field kernels with the portable authority on boundary
  /// values and pseudo-random canonical elements, including values just
  /// below `p` that exercise the final conditional subtraction.
  #[cfg(any(
    all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
    all(target_arch = "x86_64", not(feature = "portable-only"), not(miri))
  ))]
  #[test]
  fn accelerated_field_kernels_match_portable_authority() {
    fn splitmix(state: &mut u64) -> u64 {
      *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
      let mut z = *state;
      z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
      z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
      z ^ (z >> 31)
    }
    fn canonical(state: &mut u64) -> Uint {
      loop {
        let mut limbs = [0u64; 6];
        for limb in &mut limbs {
          *limb = splitmix(state);
        }
        // Force some candidates next to `p` and to sparse or dense limbs.
        match splitmix(state) % 4 {
          0 => limbs[2..].fill(u64::MAX),
          1 => limbs[..3].fill(0),
          _ => {}
        }
        let value = Uint(limbs);
        if value.cmp_vartime(&super::FIELD_MODULUS).is_lt() {
          return value;
        }
      }
    }

    // Computed with Python integers: B1 = R / A and B2 = 2R / A mod p, and
    // SQUARE_ROOT_OF_TWO^2 = 2R mod p, where R = 2^384.
    const SMALL_PRODUCT_A: Uint = Uint([
      0x4567_89ab_cdef_0123,
      0xf567_89ab_cdef_0123,
      0x7d8e_9fa0_b1c2_d3e4,
      0x3b8e_1d2c_3f4a_5b6c,
      0xf33a_1e8c_91c4_2f6a,
      0x05b4_77d8_f196_476e,
    ]);
    const SMALL_PRODUCT_B1: Uint = Uint([
      0x3211_3afc_c048_1d23,
      0x7b7c_b075_049f_0682,
      0x59c9_fa68_4dd4_9e2e,
      0x375f_31ff_f904_2d1d,
      0xda7b_c49b_6ae6_2a84,
      0xcf1e_3ec2_a9b9_eb54,
    ]);
    const SMALL_PRODUCT_B2: Uint = Uint([
      0x6422_75f8_8090_3a47,
      0xf6f9_60eb_093e_0d04,
      0xb393_f4d0_9ba9_3c5d,
      0x6ebe_63ff_f208_5a3a,
      0xb4f7_8936_d5cc_5508,
      0x9e3c_7d85_5373_d6a9,
    ]);
    const SQUARE_ROOT_OF_TWO: Uint = Uint([
      0x01e9_994f_4084_06c4,
      0xb485_1561_9734_a613,
      0x1f60_ec57_b5d7_a6ff,
      0x7154_1b27_c534_cedb,
      0x972f_548b_2adb_0681,
      0x10ec_f788_79b9_9d71,
    ]);
    let (p_minus_two, _) = P_MINUS_ONE.sub_raw(Uint([1, 0, 0, 0, 0, 0]));
    let edges = [
      Uint::ZERO,
      Uint([1, 0, 0, 0, 0, 0]),
      Uint([0, 0, 0, 0, 0, 1 << 63]),
      super::FIELD_ONE_MONTGOMERY,
      super::FIELD_R2,
      GENERATOR_X,
      GENERATOR_Y,
      p_minus_two,
      P_MINUS_ONE,
      // `0x5555... * 3` is all ones, so tripling carries through limbs 1..=5.
      // Random limbs reach that carry with probability about 2^-62.
      Uint([
        u64::MAX,
        0x5555_5555_5555_5555,
        0x5555_5555_5555_5555,
        0x5555_5555_5555_5555,
        0x5555_5555_5555_5555,
        0x5555_5555_5555_5555,
      ]),
      // `(p + 1) / 3`: tripling yields `p + 1`, so the final subtraction of `p`
      // borrows through every limb.
      Uint([
        0xaaaa_aaab_0000_0000,
        0xffff_ffff_aaaa_aaaa,
        0x5555_5555_5555_5554,
        0x5555_5555_5555_5555,
        0x5555_5555_5555_5555,
        0x5555_5555_5555_5555,
      ]),
      // `A` with `B1` and `B2` multiply to 1 and 2, and `SQUARE_ROOT_OF_TWO`
      // squares to 2 (all in the Montgomery domain). A small result can leave
      // the unreduced value in `[2^384 - c, 2^384)`, the only range in which
      // adding `c = 2^384 - p` carries through limbs 3..=5; uniform operands
      // reach it with probability about 2^-256.
      SMALL_PRODUCT_A,
      SMALL_PRODUCT_B1,
      SMALL_PRODUCT_B2,
      SQUARE_ROOT_OF_TWO,
    ];
    for left in edges {
      assert!(super::montgomery_square(left) == super::montgomery_square_portable(left));
      for right in edges {
        assert!(super::montgomery_mul(left, right) == super::montgomery_mul_portable(left, right));
        assert_field_add_sub_match(left, right);
      }
    }

    let mut state = 0x0384_0384_0384_0384;
    for _ in 0..100_000 {
      let left = canonical(&mut state);
      let right = canonical(&mut state);
      assert!(super::montgomery_mul(left, right) == super::montgomery_mul_portable(left, right));
      assert!(super::montgomery_square(left) == super::montgomery_square_portable(left));
      assert_field_add_sub_match(left, right);
    }

    // The fused x86-64 doubling uses a different operation order and linear
    // combinations; it must match the formula on any canonical coordinates,
    // including infinity and points off the curve.
    #[cfg(all(target_arch = "x86_64", not(feature = "portable-only"), not(miri)))]
    if super::has_bmi2_adx() {
      let assert_double_matches = |x: Uint, y: Uint, z: Uint| {
        let point = Jacobian {
          x: FieldElement(x),
          y: FieldElement(y),
          z: FieldElement(z),
        };
        let expected = point.double_formula();
        let mut fused = point;
        fused.double_in_place();
        assert!(fused.x == expected.x && fused.y == expected.y && fused.z == expected.z);
      };
      for x in edges {
        for y in edges {
          for z in [Uint::ZERO, super::FIELD_ONE_MONTGOMERY, P_MINUS_ONE] {
            assert_double_matches(x, y, z);
          }
        }
      }
      for _ in 0..20_000 {
        assert_double_matches(canonical(&mut state), canonical(&mut state), canonical(&mut state));
      }
    }
  }

  #[cfg(any(
    all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
    all(target_arch = "x86_64", not(feature = "portable-only"), not(miri))
  ))]
  #[test]
  fn divstep_inversion_matches_fermat_chain() {
    let mut state = 0x0384_1111_2222_3333u64;
    let (p_minus_two, _) = P_MINUS_ONE.sub_raw(Uint([1, 0, 0, 0, 0, 0]));
    let mut inputs = vec![
      Uint::ZERO,
      Uint([1, 0, 0, 0, 0, 0]),
      super::FIELD_ONE_MONTGOMERY,
      super::FIELD_R2,
      P_MINUS_ONE,
      p_minus_two,
      GENERATOR_X,
      GENERATOR_Y,
    ];
    while inputs.len() < 3000 {
      let mut limbs = [0u64; 6];
      for limb in &mut limbs {
        state = state
          .wrapping_mul(6_364_136_223_846_793_005)
          .wrapping_add(1_442_695_040_888_963_407);
        *limb = state;
      }
      let candidate = Uint(limbs);
      if candidate.cmp_vartime(&super::FIELD_MODULUS).is_lt() {
        inputs.push(candidate);
      }
    }
    for value in inputs {
      let value = FieldElement::from_montgomery(value);
      assert!(value.invert() == value.invert_fermat());
    }
  }

  #[cfg(any(
    all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
    all(target_arch = "x86_64", not(feature = "portable-only"), not(miri))
  ))]
  fn assert_field_add_sub_match(left: Uint, right: Uint) {
    let double = left.add_mod(left);
    assert!(FieldElement(left).triple().0 == double.add_mod(left));
    assert!(FieldElement(left).times4().0 == double.add_mod(double));
    assert!(FieldElement(left).times8().0 == double.add_mod(double).add_mod(double.add_mod(double)));
    let (sum, difference) = (
      FieldElement(left).add(FieldElement(right)),
      FieldElement(left).sub(FieldElement(right)),
    );
    assert!(sum.0 == left.add_mod(right));
    assert!(difference.0 == left.sub_mod(right));
  }

  #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
  #[test]
  fn aarch64_comb_selection_matches_portable_scan() {
    // Digits past the table select nothing on both paths.
    for digit in (0..=256).chain([usize::MAX, 1 << 63]) {
      let (selected, _) = Affine::select_generator_comb(digit);
      let (x, y) = Affine::select_generator_comb_limbs(digit);
      assert!(selected.x.0 == x && selected.y.0 == y, "digit {digit}");
    }
  }

  #[test]
  fn field_multiplication_matches_independent_vectors() {
    // Expected products were computed with Python integers as a * b mod p.
    let cases = [
      (P_MINUS_ONE, P_MINUS_ONE, Uint([1, 0, 0, 0, 0, 0])),
      (
        GENERATOR_X,
        GENERATOR_Y,
        Uint([
          0xb3c8_c48b_2ad3_f025,
          0x482e_90de_9d3c_cd96,
          0x84dc_5d3c_c441_cb0a,
          0x8219_71a9_9c25_0daf,
          0x3cb2_9c4b_55af_5783,
          0x332e_5593_89c9_7031,
        ]),
      ),
      (
        GENERATOR_X,
        GENERATOR_X,
        Uint([
          0xe222_2412_aca8_b019,
          0x92e8_614f_ee38_a288,
          0xffd8_419f_e7b1_3f6a,
          0xc335_3aca_34a3_80e1,
          0x6728_217d_f5bc_7c1f,
          0x046a_f925_fa51_ac49,
        ]),
      ),
    ];
    for (left, right, expected) in cases {
      let product = FieldElement::from_uint(left).mul(FieldElement::from_uint(right));
      assert!(product.to_uint() == expected);
      if left == right {
        assert!(FieldElement::from_uint(left).square().to_uint() == expected);
      }
    }
  }

  #[test]
  fn field_inversion_is_multiplicative_inverse() {
    assert!(FieldElement::ZERO.invert() == FieldElement::ZERO);
    assert!(FieldElement::ZERO.invert_fermat() == FieldElement::ZERO);
    for value in [Uint([1, 0, 0, 0, 0, 0]), GENERATOR_X, GENERATOR_Y, P_MINUS_ONE] {
      let value = FieldElement::from_uint(value);
      assert!(value.mul(value.invert()) == FieldElement::ONE);
    }
  }

  #[test]
  fn generator_comb_table_uses_this_montgomery_domain() {
    let (selected, infinity) = Affine::select_generator_comb(1);
    assert_eq!(infinity, 0);
    assert!(selected.x == generator().x && selected.y == generator().y);
    assert!(generator().is_on_curve());
  }

  #[test]
  fn fixed_base_and_window_multiplication_agree() {
    let table = super::precompute_window_table(generator());
    let mut order_minus_one = [0xffu8; 48];
    order_minus_one[24..].copy_from_slice(&[
      0xc7, 0x63, 0x4d, 0x81, 0xf4, 0x37, 0x2d, 0xdf, 0x58, 0x1a, 0x0d, 0xb2, 0x48, 0xb0, 0xa7, 0x7a, 0xec, 0xec, 0x19,
      0x6a, 0xcc, 0xc5, 0x29, 0x72,
    ]);
    let mut one = [0u8; 48];
    one[47] = 1;
    for bytes in [one, [0x11; 48], [0x42; 48], [0x7f; 48], [0xa5; 48], order_minus_one] {
      let scalar = Scalar::from_bytes(&bytes);
      let fixed = super::scalar_mul_generator(&scalar).0.to_affine().encode_sec1();
      let window = super::scalar_mul_window(&scalar, &table).0.to_affine().encode_sec1();
      assert_eq!(fixed, window, "scalar {bytes:02x?}");
    }
  }

  #[test]
  fn complete_addition_handles_equal_opposite_and_infinity() {
    let point = Jacobian::from_affine(generator());
    let cached = CachedJacobian::new(point);
    let doubled = point.double().to_affine().encode_sec1();
    assert_eq!(point.add_complete(cached).to_affine().encode_sec1(), doubled);
    let mut negated = point;
    negated.y = negated.y.negate();
    assert_ne!(negated.add_complete(cached).infinity_mask(), 0);
    assert_eq!(
      Jacobian::INFINITY.add_complete(cached).to_affine().encode_sec1(),
      generator().encode_sec1()
    );
    assert_eq!(
      point
        .add_complete(CachedJacobian::new(Jacobian::INFINITY))
        .to_affine()
        .encode_sec1(),
      generator().encode_sec1()
    );
    assert_ne!(Jacobian::INFINITY.double().infinity_mask(), 0);

    // Cached entries with Z != 1: 2G + 2G = 4G and 2G + G = 3G, checked
    // against doubling and the independent mixed-addition formula.
    let two = point.double();
    let three = two.add_mixed_formula(generator());
    assert_eq!(
      point.add_complete(CachedJacobian::new(two)).to_affine().encode_sec1(),
      three.to_affine().encode_sec1()
    );
    assert_eq!(
      two.add_complete(CachedJacobian::new(two)).to_affine().encode_sec1(),
      two.double().to_affine().encode_sec1()
    );
    assert_eq!(
      three.add_distinct(CachedJacobian::new(point)).to_affine().encode_sec1(),
      two.double().to_affine().encode_sec1()
    );
  }
}
