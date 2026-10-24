//! RSA public operations that enter the Montgomery domain by long division.
//!
//! A key imported from its encoding has no precomputed `R² mod n`. Instead of computing it, a
//! one-time public operation divides `x·R` by the modulus, which costs about half of one
//! Montgomery multiplication and replaces both `R² mod n` and the multiplication by it. Scratch
//! from [`RsaPublicKey::public_scratch`] is for reuse, so it computes `R² mod n` once, with two
//! such divisions, and each operation then enters by one Montgomery multiplication. The division
//! is variable time in the representative and the modulus, which are public in signature
//! verification and encryption. Private-key operations keep the precomputed `R² mod n` through
//! [`RsaPublicKey::prepared_public_operation_with_scratch`], so their code never reaches it.

#[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
use super::rsa_aarch64_mont;
use super::{
  RsaPublicArithmeticScratch, RsaPublicExponent, RsaPublicKey, RsaPublicModulus, RsaPublicOpError, RsaPublicScratch,
  clear_output_on_error, cmp_limbs, copy_limbs, limbs_from_be_into, limbs_to_be, mont_mac, mont_mul, mont_mul_cios,
  mont_mul_cios_in_place_left, mont_mul_in_place_left, mont_square_cios_in_place, mont_square_in_place, split_u128,
  sub_modulus_in_place, use_public_cios_montgomery,
};

impl RsaPublicKey {
  /// [`Self::public_operation_with_scratch`] through the precomputed `R² mod n` that the public
  /// half of a private key holds. Private-key blinding and fault checks call it, so their code
  /// never reaches [`public_montgomery_form_vartime`].
  pub(super) fn prepared_public_operation_with_scratch(
    &self,
    input: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaPublicOpError> {
    let result = self.modulus.public_operation(self.exponent, input, out, scratch);
    clear_output_on_error(result, out)
  }

  /// [`RsaPublicModulus::public_operation_by_division`] with this key's exponent. It keeps the
  /// call in [`Self::public_operation_with_scratch`] to one line, so no later panic location in
  /// `rsa.rs` moves.
  #[inline(always)]
  pub(super) fn public_operation_by_division(
    &self,
    input: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaPublicOpError> {
    self
      .modulus
      .public_operation_by_division(self.exponent, input, out, scratch.arithmetic_scratch())
  }
}

impl RsaPublicScratch {
  /// [`Self::new`] with `key`'s `R² mod n` bound in the scratch, unless the modulus carries its
  /// own. `R mod n` is the Montgomery form of 1, and `R² mod n` is the Montgomery form of that.
  pub(super) fn with_montgomery_r2(key: &RsaPublicKey) -> Self {
    let mut scratch = Self::new(key);
    if key.modulus.montgomery_r2().is_none() {
      let RsaPublicArithmeticScratch {
        modulus, r2, t, tmp, ..
      } = scratch.arithmetic_scratch();
      // The modulus slot is the divisions' working space until it binds the result.
      tmp.fill(0);
      tmp[0] = 1;
      public_montgomery_form_vartime(r2, tmp, &key.modulus.limbs, t, modulus);
      tmp.copy_from_slice(r2);
      public_montgomery_form_vartime(r2, tmp, &key.modulus.limbs, t, modulus);
      modulus.copy_from_slice(&key.modulus.limbs);
    }
    scratch
  }
}

impl RsaPublicModulus {
  /// `out = input^e mod n` for a public representative `input < n`.
  ///
  /// The operation enters the Montgomery domain by one Montgomery multiplication when an
  /// `R² mod n` is at hand: the modulus's own, for the public half of a private key, or one
  /// bound to this modulus in the scratch. Otherwise it enters by
  /// [`public_montgomery_form_vartime`].
  pub(super) fn public_operation_by_division(
    &self,
    exponent: RsaPublicExponent,
    input: &[u8],
    out: &mut [u8],
    scratch: RsaPublicArithmeticScratch<'_>,
  ) -> Result<(), RsaPublicOpError> {
    let limbs = self.limbs.len();
    let RsaPublicArithmeticScratch {
      modulus,
      r2,
      t,
      x,
      tmp,
      base,
      acc,
    } = scratch;
    if modulus.len() != limbs || r2.len() != limbs {
      return Err(RsaPublicOpError::InvalidScratch);
    }
    if input.len() != self.bytes.len() || out.len() != self.bytes.len() {
      return Err(RsaPublicOpError::InvalidLength);
    }
    if x.len() != limbs
      || base.len() != limbs
      || acc.len() != limbs
      || tmp.len() != limbs
      || t.len() != limbs.strict_mul(2).strict_add(2)
    {
      return Err(RsaPublicOpError::InvalidScratch);
    }

    limbs_from_be_into(input, x);
    if cmp_limbs(x, &self.limbs) != core::cmp::Ordering::Less {
      return Err(RsaPublicOpError::RepresentativeOutOfRange);
    }

    let cios = use_public_cios_montgomery(self);
    let bound = (*modulus == *self.limbs).then_some(&*r2);
    match (self.montgomery_r2().or(bound), cios) {
      (Some(r2), true) => mont_mul_cios(base, x, r2, self, t),
      (Some(r2), false) => mont_mul(base, x, r2, self, t),
      (None, _) => public_montgomery_form_vartime(base, x, &self.limbs, t, tmp),
    }
    copy_limbs(acc, base);
    let exponent = exponent.as_u64();

    #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
    if cios
      && exponent == 65_537
      && rsa_aarch64_mont::supports(limbs)
      && t.len() >= rsa_aarch64_mont::scratch_words(limbs)
    {
      for _ in 0..16 {
        rsa_aarch64_mont::square_in_place(acc, &self.limbs, &self.n0_256, t);
      }
      rsa_aarch64_mont::mul_in_place_left(acc, x, &self.limbs, &self.n0_256, t);
      limbs_to_be(acc, out);
      return Ok(());
    }

    if cios {
      public_power(
        exponent,
        x,
        base,
        acc,
        tmp,
        self,
        t,
        mont_square_cios_in_place,
        mont_mul_cios_in_place_left,
      );
    } else {
      public_power(
        exponent,
        x,
        base,
        acc,
        tmp,
        self,
        t,
        mont_square_in_place,
        mont_mul_in_place_left,
      );
    }
    limbs_to_be(acc, out);
    Ok(())
  }
}

/// `acc = x^e mod n` from `acc = base = x·R mod n`, by square-and-multiply in the Montgomery
/// domain with one width's squaring and multiplication.
///
/// [`RsaPublicExponent`] is odd, so the last step multiplies by `x` itself rather than by
/// `base`: `(x^(e-1)·R)·x·R⁻¹ = x^e mod n` leaves the Montgomery domain without a separate
/// reduction.
#[expect(
  clippy::too_many_arguments,
  reason = "the exponent, operands, scratch, modulus, and one width's two kernels form one exponentiation"
)]
#[inline(always)]
fn public_power(
  exponent: u64,
  x: &[u64],
  base: &[u64],
  acc: &mut [u64],
  tmp: &mut [u64],
  modulus: &RsaPublicModulus,
  t: &mut [u64],
  square: impl Fn(&mut [u64], &mut [u64], &RsaPublicModulus, &mut [u64]),
  multiply: impl Fn(&mut [u64], &[u64], &mut [u64], &RsaPublicModulus, &mut [u64]),
) {
  debug_assert_eq!(exponent & 1, 1);
  let top_bit = 63u32.strict_sub(exponent.leading_zeros());
  for bit in (1..top_bit).rev() {
    square(acc, tmp, modulus, t);
    if (exponent >> bit) & 1 == 1 {
      multiply(acc, base, tmp, modulus, t);
    }
  }
  square(acc, tmp, modulus, t);
  multiply(acc, x, tmp, modulus, t);
}

/// Write the Montgomery form `x·R mod n` of a public representative `x < n`, for `R = 2^(64·L)`
/// and the odd `L`-limb public modulus `n`, by long division of `x·R` by `n`.
///
/// Knuth's Algorithm D (TAOCP vol. 2, 3rd ed., §4.3.1) clears one limb of `x·R` per row. Each
/// row's quotient limb comes from the top three limbs of the running remainder and the top two
/// limbs of the normalized modulus by Möller and Granlund's 3-by-2 division ("Improved Division
/// by Invariant Integers", IEEE Trans. Comput. 60(2), 2011, Algorithm 5), which uses a
/// precomputed reciprocal and no division per row. That estimate is the true quotient limb or
/// one more, so a row needs at most one correction. A row adds `q·(2^(64·L) − n)` instead of
/// subtracting `q·n`, so each limb is the one-carry multiply-accumulate of a Montgomery row.
///
/// The `L` rows cost `L²` multiply-accumulates, about half of one Montgomery multiplication. The
/// quotient estimates, the corrections, and the reciprocal's division depend on `x` and `n`:
/// this is variable time, for public values only. `window` needs `2·L` limbs and `complement`
/// `L`.
pub(super) fn public_montgomery_form_vartime(
  out: &mut [u64],
  x: &[u64],
  modulus: &[u64],
  window: &mut [u64],
  complement: &mut [u64],
) {
  let limbs = modulus.len();
  debug_assert_eq!(out.len(), limbs);
  debug_assert_eq!(x.len(), limbs);
  debug_assert_eq!(cmp_limbs(x, modulus), core::cmp::Ordering::Less);
  debug_assert_eq!(modulus.first().copied().unwrap_or(0) & 1, 1);
  let Some(&top) = modulus.last() else {
    return;
  };
  debug_assert_ne!(top, 0);
  let shift = top.leading_zeros();

  if limbs == 1 {
    out[0] = split_u128((u128::from(x[0]) << 64).strict_rem(u128::from(top))).0;
    return;
  }

  // Shifting `n` and `x` left until the modulus's top bit is set scales the remainder by the
  // same power of two, which the final right shift removes.
  let window = &mut window[..limbs.strict_mul(2)];
  let complement = &mut complement[..limbs];
  shift_limbs_left_into(complement, modulus, shift);
  let divisor_high = complement[limbs.strict_sub(1)];
  let divisor_low = complement[limbs.strict_sub(2)];
  // `2^(64·L) − (n << shift)`. Limb 0 of `n << shift` holds the odd low bit of `n`, so it is
  // nonzero and negation borrows from no other limb.
  complement[0] = complement[0].wrapping_neg();
  for limb in &mut complement[1..] {
    *limb = !*limb;
  }
  let reciprocal = reciprocal_3by2(divisor_high, divisor_low);

  let (low, high) = window.split_at_mut(limbs);
  low.fill(0);
  shift_limbs_left_into(high, x, shift);

  // Row `row` divides the `L + 1` limbs from `row`, which are below `(n << shift)·2^64`, by
  // `n << shift`, and leaves the remainder in the `L` limbs from `row`.
  for row in (0..limbs).rev() {
    let top_index = row.strict_add(limbs);
    let u2 = window[top_index];
    let u1 = window[top_index.strict_sub(1)];
    let u0 = window[top_index.strict_sub(2)];
    // The running value is below the divisor times 2^64, so (u2, u1) ≤ (d1, d0), and equality
    // makes the quotient limb 2^64 − 1, outside the 3-by-2 division's domain.
    let quotient = if u2 == divisor_high && u1 == divisor_low {
      u64::MAX
    } else {
      div_3by2(u2, u1, u0, divisor_high, divisor_low, reciprocal)
    };
    let mut carry = 0u64;
    for (limb, &complement_limb) in window[row..top_index].iter_mut().zip(complement.iter()) {
      (*limb, carry) = mont_mac(*limb, quotient, complement_limb, carry);
    }
    // The row's top limb is now u2 + carry − quotient: 0, or −1 when the estimate was one too
    // large. Then the low limbs hold the negative remainder plus 2^(64·L), which is at least the
    // complement, and subtracting the complement adds the divisor back.
    if u2.wrapping_add(carry) != quotient {
      let borrow = sub_modulus_in_place(&mut window[row..top_index], complement);
      debug_assert_eq!(borrow, 0);
    }
  }

  shift_limbs_right_into(out, &window[..limbs], shift);
}

/// `⌊(2^192 − 1) / d⌋ − 2^64` for the normalized divisor `d = d1·2^64 + d0`, `d1 ≥ 2^63`: the
/// reciprocal for [`div_3by2`]. Möller and Granlund, Algorithm 6: start from the two-by-one
/// reciprocal of `d1`, which is never too small, and step it down while `(2^64 + v)·d` reaches
/// `2^192`, first for `d1·2^64 + d0·2^64` and then for the rest of `v·d0`.
fn reciprocal_3by2(d1: u64, d0: u64) -> u64 {
  debug_assert_eq!(d1 >> 63, 1);
  // ⌊(2^128 − 1) / d1⌋ lies in [2^64, 2^65), so its low limb is the two-by-one reciprocal.
  let mut v = split_u128(u128::MAX.strict_div(u128::from(d1))).0;
  // `(2^64 + v)·d1 = (2^64 − 1)·2^64 + p`; adding `d0·2^64` carries out of 2^192 when `p + d0`
  // wraps, and each decrement takes `d1` back off the middle limb.
  let mut p = d1.wrapping_mul(v).wrapping_add(d0);
  if p < d0 {
    v = v.wrapping_sub(1);
    if p >= d1 {
      v = v.wrapping_sub(1);
      p = p.wrapping_sub(d1);
    }
    p = p.wrapping_sub(d1);
  }
  // Add `v·d0`; a carry out of 2^192 needs one or two more decrements of the whole divisor.
  let (t0, t1) = split_u128(u128::from(v).strict_mul(u128::from(d0)));
  p = p.wrapping_add(t1);
  if p < t1 {
    v = v.wrapping_sub(1);
    if (p, t0) >= (d1, d0) {
      v = v.wrapping_sub(1);
    }
  }
  v
}

/// `⌊(u2·2^128 + u1·2^64 + u0) / (d1·2^64 + d0)⌋` for `(u2, u1) < (d1, d0)` and the reciprocal
/// `v` of the normalized divisor from [`reciprocal_3by2`]: Möller and Granlund, Algorithm 5. The
/// quotient fits one limb. The first adjustment applies about half the time, so it is masked
/// rather than branched.
#[inline(always)]
fn div_3by2(u2: u64, u1: u64, u0: u64, d1: u64, d0: u64, v: u64) -> u64 {
  let divisor = (u128::from(d1) << 64) | u128::from(d0);
  // `(2^64 + v)·u2 + u1` stays below 2^128 under the precondition.
  let (q0, q1) = split_u128(
    u128::from(v)
      .strict_mul(u128::from(u2))
      .wrapping_add((u128::from(u2) << 64) | u128::from(u1)),
  );
  let r1 = u1.wrapping_sub(q1.wrapping_mul(d1));
  let mut remainder = ((u128::from(r1) << 64) | u128::from(u0))
    .wrapping_sub(u128::from(d0).strict_mul(u128::from(q1)))
    .wrapping_sub(divisor);
  let over = u64::from(split_u128(remainder).1 >= q0);
  let quotient = q1.wrapping_add(1).wrapping_sub(over);
  remainder = remainder.wrapping_add(divisor & 0u128.wrapping_sub(u128::from(over)));
  quotient.wrapping_add(u64::from(remainder >= divisor))
}

/// `out = value·2^shift` limb by limb, for `shift < 64` and a `value` whose top `shift` bits are
/// zero.
fn shift_limbs_left_into(out: &mut [u64], value: &[u64], shift: u32) {
  debug_assert_eq!(out.len(), value.len());
  if shift == 0 {
    out.copy_from_slice(value);
    return;
  }
  let carry_shift = 64u32.strict_sub(shift);
  let mut carry = 0u64;
  for (out_limb, &limb) in out.iter_mut().zip(value) {
    *out_limb = (limb << shift) | carry;
    carry = limb >> carry_shift;
  }
  debug_assert_eq!(carry, 0);
}

/// `out = ⌊value / 2^shift⌋` limb by limb, for `shift < 64`.
fn shift_limbs_right_into(out: &mut [u64], value: &[u64], shift: u32) {
  debug_assert_eq!(out.len(), value.len());
  if shift == 0 {
    out.copy_from_slice(value);
    return;
  }
  let carry_shift = 64u32.strict_sub(shift);
  let mut carry = 0u64;
  for (out_limb, &limb) in out.iter_mut().zip(value).rev() {
    *out_limb = (limb >> shift) | carry;
    carry = limb << carry_shift;
  }
}

#[cfg(test)]
mod tests {
  use alloc::{vec, vec::Vec};

  use super::{
    super::{
      RsaPublicExponent, RsaPublicKey, RsaPublicModulus, RsaPublicOpError, RsaPublicScratch, limbs_to_be,
      unsigned_be_bit_len,
    },
    div_3by2, public_montgomery_form_vartime, reciprocal_3by2,
  };

  fn limbs_to_biguint(limbs: &[u64]) -> rsa::BigUint {
    let bytes: Vec<u8> = limbs.iter().flat_map(|limb| limb.to_le_bytes()).collect();
    rsa::BigUint::from_bytes_le(&bytes)
  }

  fn biguint_to_limbs(value: &rsa::BigUint, words: usize) -> Vec<u64> {
    let mut bytes = value.to_bytes_le();
    assert!(bytes.len() <= words.strict_mul(8), "value exceeds {words} limbs");
    bytes.resize(words.strict_mul(8), 0);
    bytes
      .as_chunks::<8>()
      .0
      .iter()
      .map(|chunk| u64::from_le_bytes(*chunk))
      .collect()
  }

  fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
  }

  /// Odd `words`-limb moduli covering the normalization shifts and divisor shapes of
  /// `public_montgomery_form_vartime`: random limbs under top limbs whose leading bit sits at
  /// several positions; `2^(64·words) − 1`; moduli one above a power of two, with the top bit at
  /// either end of the top limb; and the extreme top-two-limb pairs over random low limbs.
  fn test_moduli(words: usize, state: &mut u64) -> Vec<Vec<u64>> {
    let random = |state: &mut u64| (0..words).map(|_| xorshift(state)).collect::<Vec<u64>>();
    let top = words.strict_sub(1);
    let mut moduli = Vec::new();
    for top_bit in [63u32, 62, 40, 7, 0] {
      let mut limbs = random(state);
      limbs[top] = (1u64 << top_bit) | (xorshift(state) & (1u64 << top_bit).wrapping_sub(1));
      moduli.push(limbs);
    }
    moduli.push(vec![u64::MAX; words]);
    for top_limb in [1u64 << 63, 1] {
      let mut limbs = vec![0u64; words];
      limbs[top] = top_limb;
      moduli.push(limbs);
    }
    if words >= 2 {
      for (high, low) in [(u64::MAX, u64::MAX), (1u64 << 63, 0)] {
        let mut limbs = random(state);
        limbs[top] = high;
        limbs[top.strict_sub(1)] = low;
        moduli.push(limbs);
      }
    }
    for limbs in &mut moduli {
      limbs[0] |= 1;
    }
    // A one-limb modulus needs room for the range-end inputs.
    moduli.retain(|limbs| limbs.len() > 1 || limbs[0] > 2);
    moduli
  }

  /// Representatives below `modulus`: the ends of the range, random values, and, for four or more
  /// limbs, a value whose first quotient estimate is one too large, so that the row's correction
  /// runs.
  #[expect(clippy::arithmetic_side_effects, reason = "big-integer oracle arithmetic")]
  fn test_inputs(modulus: &[u64], state: &mut u64) -> Vec<Vec<u64>> {
    let words = modulus.len();
    let n = limbs_to_biguint(modulus);
    let one = rsa::BigUint::from(1u8);
    let two = rsa::BigUint::from(2u8) % &n;
    let mut inputs = vec![rsa::BigUint::from(0u8), one.clone(), two.clone(), &n - &one, &n - &two];
    for _ in 0..4 {
      let random: Vec<u64> = (0..words).map(|_| xorshift(state)).collect();
      inputs.push(limbs_to_biguint(&random) % &n);
    }
    if words >= 4 {
      // With `d` the top two limbs of the normalized modulus, the first row divides
      // `(2^64 − 1)·d·2^(64·(words − 2))`: its top three limbs give the estimate 2^64 − 1, but the
      // nonzero low limbs of `n << shift` make the true quotient limb smaller.
      let shift = usize::try_from(modulus[words.strict_sub(1)].leading_zeros()).expect("shift fits usize");
      let normalized = &n << shift;
      let divisor_top = &normalized >> words.strict_sub(2).strict_mul(64);
      let shifted = (rsa::BigUint::from(u64::MAX) * &divisor_top) << words.strict_sub(3).strict_mul(64);
      let input = &shifted >> shift;
      assert_eq!(&input << shift, shifted);
      assert!(input < n);
      let row = &shifted << 64usize;
      let estimate = (&row >> words.strict_sub(2).strict_mul(64)) / &divisor_top;
      assert!(
        estimate > &row / &normalized,
        "the constructed row must need its correction"
      );
      inputs.push(input);
    }
    inputs.iter().map(|input| biguint_to_limbs(input, words)).collect()
  }

  #[test]
  fn montgomery_form_matches_big_integer_reduction() {
    // Every limb count through RSA-8192 and RSA-16384, under each normalization shift and divisor
    // shape, against an independent big-integer reduction of `x·2^(64·L)`.
    let widths: Vec<usize> = if cfg!(miri) {
      vec![1, 2, 3, 4]
    } else {
      (1..=129).chain([256]).collect()
    };
    let mut state = 0x3c6e_f372_fe94_f82bu64;
    for words in widths {
      for modulus in test_moduli(words, &mut state) {
        let n = limbs_to_biguint(&modulus);
        let r = rsa::BigUint::from(1u8) << words.strict_mul(64);
        for input in test_inputs(&modulus, &mut state) {
          let expected = biguint_to_limbs(&((limbs_to_biguint(&input) * &r) % &n), words);
          let mut out = vec![0u64; words];
          let mut window = vec![0u64; words.strict_mul(2)];
          let mut complement = vec![0u64; words];
          public_montgomery_form_vartime(&mut out, &input, &modulus, &mut window, &mut complement);
          assert_eq!(out, expected, "{words} limbs, modulus {modulus:x?}, input {input:x?}");
        }
      }
    }
  }

  #[test]
  fn reciprocal_and_quotient_match_big_integer_division() {
    let mut state = 0xa54f_f53a_5f1d_36f1u64;
    let b = rsa::BigUint::from(1u8) << 64usize;
    let mut divisors = vec![
      (1u64 << 63, 0u64),
      (1u64 << 63, 1),
      (1u64 << 63, u64::MAX),
      (u64::MAX, 0),
      (u64::MAX, u64::MAX),
      ((1u64 << 63) | 1, u64::MAX),
    ];
    for _ in 0..if cfg!(miri) { 4 } else { 2000 } {
      divisors.push((xorshift(&mut state) | (1 << 63), xorshift(&mut state)));
    }
    for (d1, d0) in divisors {
      let divisor = rsa::BigUint::from(d1) * &b + rsa::BigUint::from(d0);
      let v = reciprocal_3by2(d1, d0);
      let full = ((&b * &b * &b) - rsa::BigUint::from(1u8)) / &divisor;
      assert_eq!(rsa::BigUint::from(v) + &b, full, "reciprocal of ({d1:#x}, {d0:#x})");

      // Numerators with (u2, u1) < (d1, d0): zero, the largest, and random ones on both sides of
      // u2 = d1.
      let largest = if d0 == 0 {
        (d1.wrapping_sub(1), u64::MAX, u64::MAX)
      } else {
        (d1, d0.wrapping_sub(1), u64::MAX)
      };
      let mut numerators = vec![(0, 0, 0), largest];
      for _ in 0..if cfg!(miri) { 2 } else { 16 } {
        numerators.push((
          xorshift(&mut state).strict_rem(d1),
          xorshift(&mut state),
          xorshift(&mut state),
        ));
        numerators.push((d1, xorshift(&mut state).strict_rem(d0.max(1)), xorshift(&mut state)));
      }
      for (u2, u1, u0) in numerators {
        if (u2, u1) >= (d1, d0) {
          continue;
        }
        let numerator = (rsa::BigUint::from(u2) * &b + rsa::BigUint::from(u1)) * &b + rsa::BigUint::from(u0);
        assert_eq!(
          rsa::BigUint::from(div_3by2(u2, u1, u0, d1, d0, v)),
          &numerator / &divisor,
          "({u2:#x}, {u1:#x}, {u0:#x}) / ({d1:#x}, {d0:#x})"
        );
      }
    }
  }

  /// Canonical big-endian bytes and bit length of a test modulus.
  fn modulus_bytes(limbs: &[u64]) -> (Vec<u8>, usize) {
    let mut padded = vec![0u8; limbs.len().strict_mul(8)];
    limbs_to_be(limbs, &mut padded);
    let bytes = padded[padded.iter().position(|&byte| byte != 0).unwrap_or(0)..].to_vec();
    let bits = unsigned_be_bit_len(&bytes);
    (bytes, bits)
  }

  #[test]
  fn public_operation_matches_big_integer_power_on_every_entry_path() {
    // Widths reach each Montgomery kernel family: the x86-64 and AArch64 assembly widths, the
    // portable CIOS widths beside them, and product scanning above 128 limbs. Odd exponents of
    // every length end with the multiplication by the plain representative that leaves the
    // Montgomery domain. An imported key enters by division with fresh scratch or scratch bound
    // to another modulus of its width, and by its bound `R² mod n` with its own `public_scratch`;
    // a private key's public half enters by its precomputed `R² mod n`, also on the private-key
    // operations' path.
    let widths: &[usize] = if cfg!(miri) {
      &[2, 3]
    } else {
      &[1, 2, 3, 4, 8, 12, 16, 24, 31, 32, 33, 48, 64, 65, 128, 129]
    };
    let mut state = 0x510e_527f_ade6_82d1u64;
    for &words in widths {
      let exponents: &[u64] = if words <= 32 {
        &[3, 5, 17, 65_537, (1 << 32) | 1, u64::MAX]
      } else {
        &[3, 65_537]
      };
      let moduli = test_moduli(words, &mut state);
      for (index, modulus_limbs) in moduli.iter().enumerate().take(4) {
        let (bytes, bits) = modulus_bytes(modulus_limbs);
        let (foreign_bytes, foreign_bits) = modulus_bytes(&moduli[index.strict_add(1)]);
        let n = limbs_to_biguint(modulus_limbs);
        for &exponent in exponents {
          let imported = RsaPublicKey {
            modulus: RsaPublicModulus::new(&bytes, bits),
            exponent: RsaPublicExponent(exponent),
          };
          let prepared = RsaPublicKey {
            modulus: RsaPublicModulus::new_with_montgomery_r2(&bytes, bits),
            exponent: RsaPublicExponent(exponent),
          };
          let foreign = RsaPublicKey {
            modulus: RsaPublicModulus::new(&foreign_bytes, foreign_bits),
            exponent: RsaPublicExponent(exponent),
          };
          let mut fresh = RsaPublicScratch::new(&imported);
          let mut prepared_fresh = RsaPublicScratch::new(&prepared);
          let mut bound = imported.public_scratch();
          let mut foreign_bound = foreign.public_scratch();
          let mut private = prepared.public_scratch();
          let mut out = vec![0u8; bytes.len()];
          for input_limbs in test_inputs(modulus_limbs, &mut state) {
            let mut input = vec![0u8; bytes.len()];
            limbs_to_be(&input_limbs, &mut input);
            let power = limbs_to_biguint(&input_limbs).modpow(&rsa::BigUint::from(exponent), &n);
            let mut expected = vec![0u8; bytes.len()];
            limbs_to_be(&biguint_to_limbs(&power, words), &mut expected);

            for (key, scratch, path) in [
              (&imported, &mut fresh, "imported, fresh scratch"),
              (&imported, &mut bound, "imported, bound scratch"),
              (&imported, &mut foreign_bound, "imported, foreign scratch"),
              (&prepared, &mut prepared_fresh, "prepared"),
            ] {
              key
                .public_operation_with_scratch(&input, &mut out, scratch)
                .expect("public operation");
              assert_eq!(out, expected, "{path}, {words} limbs, e = {exponent:#x}");
            }
            prepared
              .prepared_public_operation_with_scratch(&input, &mut out, &mut private)
              .expect("private-key public operation");
            assert_eq!(out, expected, "private-key path, {words} limbs, e = {exponent:#x}");
          }
        }
      }
    }
  }

  #[test]
  fn public_scratch_binds_montgomery_r2_only_for_imported_keys() {
    let widths: &[usize] = if cfg!(miri) {
      &[2, 3]
    } else {
      &[1, 2, 3, 4, 31, 32, 33, 64, 128, 129]
    };
    let mut state = 0x9b05_688c_2b3e_6c1fu64;
    for &words in widths {
      for modulus_limbs in test_moduli(words, &mut state) {
        let (bytes, bits) = modulus_bytes(&modulus_limbs);
        let r2 = (rsa::BigUint::from(1u8) << words.strict_mul(128)) % limbs_to_biguint(&modulus_limbs);
        let mut expected = modulus_limbs;
        expected.extend(biguint_to_limbs(&r2, words));
        let imported = RsaPublicKey {
          modulus: RsaPublicModulus::new(&bytes, bits),
          exponent: RsaPublicExponent(65_537),
        };
        assert_eq!(
          &*imported.public_scratch().montgomery,
          expected.as_slice(),
          "{words} limbs"
        );
        assert!(
          RsaPublicScratch::new(&imported)
            .montgomery
            .iter()
            .all(|&limb| limb == 0)
        );
        let prepared = RsaPublicKey {
          modulus: RsaPublicModulus::new_with_montgomery_r2(&bytes, bits),
          exponent: RsaPublicExponent(65_537),
        };
        assert!(prepared.public_scratch().montgomery.iter().all(|&limb| limb == 0));
      }
    }
  }

  #[test]
  fn public_operation_rejects_scratch_of_another_width() {
    let key = RsaPublicKey {
      modulus: RsaPublicModulus::new(&[0xc5u8; 256], 2048),
      exponent: RsaPublicExponent(65_537),
    };
    let narrower = RsaPublicKey {
      modulus: RsaPublicModulus::new(&[0xc5u8; 248], 1984),
      exponent: RsaPublicExponent(65_537),
    };
    let mut out = [0x5au8; 256];
    assert_eq!(
      key.public_operation_with_scratch(&[1u8; 256], &mut out, &mut narrower.public_scratch()),
      Err(RsaPublicOpError::InvalidScratch)
    );
    assert_eq!(out, [0u8; 256], "a failed operation clears its output");
  }
}
