#![allow(clippy::identity_op, clippy::indexing_slicing)]

//! Internal Ed25519 scalar arithmetic mod the group order `L`.
//!
//! This is the correctness-first baseline for signing and verification. It
//! keeps the representation fixed at four little-endian `u64` limbs and uses
//! simple modular double/add reduction rather than trying to be clever early.

use core::cmp::Ordering;

use super::constants::{SCALAR_LIMBS, SECRET_KEY_LENGTH};

/// Internal scalar representation for arithmetic mod `L`.
pub(crate) type Scalar = [u64; SCALAR_LIMBS];

const ZERO: Scalar = [0, 0, 0, 0];
const ONE: Scalar = [1, 0, 0, 0];
const ORDER: Scalar = [
  6_346_243_789_798_364_141,
  1_503_914_060_200_516_822,
  0,
  1_152_921_504_606_846_976,
];
const RADIX52_MASK: u64 = (1u64 << 52) - 1;

// Montgomery-friendly 5x52-limb constants for the Ed25519 subgroup order.
const ORDER52: Scalar52 = Scalar52([
  0x0002_631a_5cf5_d3ed,
  0x000d_ea2f_79cd_6581,
  0x0000_0000_0014_def9,
  0x0000_0000_0000_0000,
  0x0000_1000_0000_0000,
]);
const R52: Scalar52 = Scalar52([
  0x000f_48bd_6721_e6ed,
  0x0003_bab5_ac67_e45a,
  0x000f_ffff_eb35_e51b,
  0x000f_ffff_ffff_ffff,
  0x0000_0fff_ffff_ffff,
]);
const RR52: Scalar52 = Scalar52([
  0x0009_d265_e952_d13b,
  0x000d_63c7_15be_a69f,
  0x0005_be65_cb68_7604,
  0x0003_dcee_c73d_217f,
  0x0000_0941_1b7c_309a,
]);
const LFACTOR52: u64 = 0x0005_1da3_1254_7e1b;

#[derive(Clone, Copy)]
struct Scalar52([u64; 5]);

#[inline(always)]
#[must_use]
fn wide_mul(lhs: u64, rhs: u64) -> u128 {
  u128::from(lhs) * u128::from(rhs)
}

impl Scalar52 {
  #[rustfmt::skip]
  #[must_use]
  fn from_bytes(bytes: &[u8; 32]) -> Self {
    let words = read_words_le_32(bytes);

    Self([
        words[0] & RADIX52_MASK,
      ((words[0] >> 52) | (words[1] << 12)) & RADIX52_MASK,
      ((words[1] >> 40) | (words[2] << 24)) & RADIX52_MASK,
      ((words[2] >> 28) | (words[3] << 36)) & RADIX52_MASK,
       (words[3] >> 16) & ((1u64 << 48) - 1),
    ])
  }

  #[rustfmt::skip]
  #[must_use]
  fn from_bytes_wide(bytes: &[u8; 64]) -> Self {
    let words = read_words_le_64(bytes);

    let lo = Self([
        words[0] & RADIX52_MASK,
      ((words[0] >> 52) | (words[1] << 12)) & RADIX52_MASK,
      ((words[1] >> 40) | (words[2] << 24)) & RADIX52_MASK,
      ((words[2] >> 28) | (words[3] << 36)) & RADIX52_MASK,
      ((words[3] >> 16) | (words[4] << 48)) & RADIX52_MASK,
    ]);
    let hi = Self([
       (words[4] >>  4) & RADIX52_MASK,
      ((words[4] >> 56) | (words[5] <<  8)) & RADIX52_MASK,
      ((words[5] >> 44) | (words[6] << 20)) & RADIX52_MASK,
      ((words[6] >> 32) | (words[7] << 32)) & RADIX52_MASK,
       (words[7] >> 20),
    ]);

    let lo = Self::montgomery_mul(&lo, &R52);
    let hi = Self::montgomery_mul(&hi, &RR52);
    Self::add(&hi, &lo)
  }

  #[rustfmt::skip]
  #[must_use]
  fn as_bytes(self) -> [u8; 32] {
    let limbs = self.0;
    [
      ( limbs[0] >>  0) as u8,
      ( limbs[0] >>  8) as u8,
      ( limbs[0] >> 16) as u8,
      ( limbs[0] >> 24) as u8,
      ( limbs[0] >> 32) as u8,
      ( limbs[0] >> 40) as u8,
      ((limbs[0] >> 48) | (limbs[1] <<  4)) as u8,
      ( limbs[1] >>  4) as u8,
      ( limbs[1] >> 12) as u8,
      ( limbs[1] >> 20) as u8,
      ( limbs[1] >> 28) as u8,
      ( limbs[1] >> 36) as u8,
      ( limbs[1] >> 44) as u8,
      ( limbs[2] >>  0) as u8,
      ( limbs[2] >>  8) as u8,
      ( limbs[2] >> 16) as u8,
      ( limbs[2] >> 24) as u8,
      ( limbs[2] >> 32) as u8,
      ( limbs[2] >> 40) as u8,
      ((limbs[2] >> 48) | (limbs[3] <<  4)) as u8,
      ( limbs[3] >>  4) as u8,
      ( limbs[3] >> 12) as u8,
      ( limbs[3] >> 20) as u8,
      ( limbs[3] >> 28) as u8,
      ( limbs[3] >> 36) as u8,
      ( limbs[3] >> 44) as u8,
      ( limbs[4] >>  0) as u8,
      ( limbs[4] >>  8) as u8,
      ( limbs[4] >> 16) as u8,
      ( limbs[4] >> 24) as u8,
      ( limbs[4] >> 32) as u8,
      ( limbs[4] >> 40) as u8,
    ]
  }

  #[must_use]
  fn add(lhs: &Self, rhs: &Self) -> Self {
    let mut out = [0u64; 5];
    let mut carry = 0u64;

    for (dst, (&left, &right)) in out.iter_mut().zip(lhs.0.iter().zip(rhs.0.iter())) {
      carry = left.strict_add(right).strict_add(carry >> 52);
      *dst = carry & RADIX52_MASK;
    }

    Self::sub(&Self(out), &ORDER52)
  }

  #[must_use]
  fn sub(lhs: &Self, rhs: &Self) -> Self {
    #[inline]
    fn barrier(value: u64) -> u64 {
      // SAFETY: `value` is a local `u64`; reading it through a volatile pointer
      // preserves the arithmetic shape without violating aliasing or lifetime rules.
      unsafe { core::ptr::read_volatile(&value) }
    }

    let mut out = [0u64; 5];
    let mut borrow = 0u64;

    for (dst, (&left, &right)) in out.iter_mut().zip(lhs.0.iter().zip(rhs.0.iter())) {
      borrow = left.wrapping_sub(right.strict_add(borrow >> 63));
      *dst = borrow & RADIX52_MASK;
    }

    let underflow_mask = ((borrow >> 63) ^ 1).wrapping_sub(1);
    let mut carry = 0u64;
    for (i, limb) in out.iter_mut().enumerate() {
      carry = (carry >> 52)
        .strict_add(*limb)
        .strict_add(ORDER52.0[i] & barrier(underflow_mask));
      *limb = carry & RADIX52_MASK;
    }

    Self(out)
  }

  #[rustfmt::skip]
  #[must_use]
  fn mul_internal(lhs: &Self, rhs: &Self) -> [u128; 9] {
    [
      wide_mul(lhs.0[0], rhs.0[0]),
      wide_mul(lhs.0[0], rhs.0[1]) + wide_mul(lhs.0[1], rhs.0[0]),
      wide_mul(lhs.0[0], rhs.0[2]) + wide_mul(lhs.0[1], rhs.0[1]) + wide_mul(lhs.0[2], rhs.0[0]),
      wide_mul(lhs.0[0], rhs.0[3]) + wide_mul(lhs.0[1], rhs.0[2]) + wide_mul(lhs.0[2], rhs.0[1]) + wide_mul(lhs.0[3], rhs.0[0]),
      wide_mul(lhs.0[0], rhs.0[4]) + wide_mul(lhs.0[1], rhs.0[3]) + wide_mul(lhs.0[2], rhs.0[2]) + wide_mul(lhs.0[3], rhs.0[1]) + wide_mul(lhs.0[4], rhs.0[0]),
      wide_mul(lhs.0[1], rhs.0[4]) + wide_mul(lhs.0[2], rhs.0[3]) + wide_mul(lhs.0[3], rhs.0[2]) + wide_mul(lhs.0[4], rhs.0[1]),
      wide_mul(lhs.0[2], rhs.0[4]) + wide_mul(lhs.0[3], rhs.0[3]) + wide_mul(lhs.0[4], rhs.0[2]),
      wide_mul(lhs.0[3], rhs.0[4]) + wide_mul(lhs.0[4], rhs.0[3]),
      wide_mul(lhs.0[4], rhs.0[4]),
    ]
  }

  #[inline(always)]
  #[must_use]
  fn montgomery_reduce(limbs: &[u128; 9]) -> Self {
    #[inline(always)]
    fn part1(sum: u128) -> (u128, u64) {
      let p = (sum as u64).wrapping_mul(LFACTOR52) & RADIX52_MASK;
      ((sum.strict_add(wide_mul(p, ORDER52.0[0]))) >> 52, p)
    }

    #[inline(always)]
    fn part2(sum: u128) -> (u128, u64) {
      let word = (sum as u64) & RADIX52_MASK;
      (sum >> 52, word)
    }

    let (carry, n0) = part1(limbs[0]);
    let (carry, n1) = part1(carry.strict_add(limbs[1]).strict_add(wide_mul(n0, ORDER52.0[1])));
    let (carry, n2) = part1(
      carry
        .strict_add(limbs[2])
        .strict_add(wide_mul(n0, ORDER52.0[2]))
        .strict_add(wide_mul(n1, ORDER52.0[1])),
    );
    let (carry, n3) = part1(
      carry
        .strict_add(limbs[3])
        .strict_add(wide_mul(n1, ORDER52.0[2]))
        .strict_add(wide_mul(n2, ORDER52.0[1])),
    );
    let (carry, n4) = part1(
      carry
        .strict_add(limbs[4])
        .strict_add(wide_mul(n0, ORDER52.0[4]))
        .strict_add(wide_mul(n2, ORDER52.0[2]))
        .strict_add(wide_mul(n3, ORDER52.0[1])),
    );

    let (carry, r0) = part2(
      carry
        .strict_add(limbs[5])
        .strict_add(wide_mul(n1, ORDER52.0[4]))
        .strict_add(wide_mul(n3, ORDER52.0[2]))
        .strict_add(wide_mul(n4, ORDER52.0[1])),
    );
    let (carry, r1) = part2(
      carry
        .strict_add(limbs[6])
        .strict_add(wide_mul(n2, ORDER52.0[4]))
        .strict_add(wide_mul(n4, ORDER52.0[2])),
    );
    let (carry, r2) = part2(carry.strict_add(limbs[7]).strict_add(wide_mul(n3, ORDER52.0[4])));
    let (carry, r3) = part2(carry.strict_add(limbs[8]).strict_add(wide_mul(n4, ORDER52.0[4])));
    let r4 = carry as u64;

    Self::sub(&Self([r0, r1, r2, r3, r4]), &ORDER52)
  }

  #[must_use]
  fn montgomery_mul(lhs: &Self, rhs: &Self) -> Self {
    Self::montgomery_reduce(&Self::mul_internal(lhs, rhs))
  }

  #[must_use]
  fn mul(lhs: &Self, rhs: &Self) -> Self {
    let product = Self::montgomery_reduce(&Self::mul_internal(lhs, rhs));
    Self::montgomery_reduce(&Self::mul_internal(&product, &RR52))
  }
}

/// Clamp the lower half of the expanded secret-key digest per RFC 8032.
#[inline]
pub(crate) fn clamp_secret_scalar(bytes: &mut [u8; SECRET_KEY_LENGTH]) {
  if let Some((first, tail)) = bytes.split_first_mut() {
    *first &= 248;
    if let Some(last) = tail.last_mut() {
      *last &= 63;
      *last |= 64;
    }
  }
}

/// Decode a 32-byte little-endian scalar into the portable limb layout.
#[inline]
#[must_use]
pub(crate) fn decode_words_le(bytes: &[u8; SECRET_KEY_LENGTH]) -> Scalar {
  let mut limbs = [0u64; SCALAR_LIMBS];
  for (limb, chunk) in limbs.iter_mut().zip(bytes.as_slice().chunks_exact(8)) {
    *limb = read_u64_le(chunk);
  }
  limbs
}

/// Decode a canonical scalar encoding, rejecting values `>= L`.
#[must_use]
pub(crate) fn from_canonical_bytes(bytes: &[u8; SECRET_KEY_LENGTH]) -> Option<Scalar> {
  let words = decode_words_le(bytes);
  if compare(&words, &ORDER) == Ordering::Less {
    Some(words)
  } else {
    None
  }
}

/// Encode a scalar canonically.
#[must_use]
pub(crate) fn to_bytes(words: &Scalar) -> [u8; SECRET_KEY_LENGTH] {
  let mut out = [0u8; SECRET_KEY_LENGTH];
  for (chunk, limb) in out.as_mut_slice().chunks_exact_mut(8).zip(words.iter().copied()) {
    let limb_bytes = limb.to_le_bytes();
    for (dst, src) in chunk.iter_mut().zip(limb_bytes.iter().copied()) {
      *dst = src;
    }
  }
  out
}

/// Reduce an arbitrary byte string modulo the Ed25519 group order.
#[must_use]
pub(crate) fn reduce_bytes_mod_order(bytes: &[u8]) -> Scalar {
  if bytes.len() == 64 {
    let mut wide = [0u8; 64];
    wide.copy_from_slice(bytes);
    return scalar52_to_words(Scalar52::from_bytes_wide(&wide));
  }

  reduce_bytes_mod_order_fallback(bytes)
}

/// Add two scalars modulo `L`.
#[must_use]
pub(crate) fn add_mod(lhs: &Scalar, rhs: &Scalar) -> Scalar {
  let (sum, _) = add_raw(lhs, rhs);
  maybe_sub_order(sum)
}

/// Multiply two scalars modulo `L`.
#[must_use]
pub(crate) fn mul_mod(lhs: &Scalar, rhs: &Scalar) -> Scalar {
  scalar52_to_words(Scalar52::mul(&scalar52_from_words(lhs), &scalar52_from_words(rhs)))
}

/// Compute `lhs * rhs + acc (mod L)`.
#[must_use]
pub(crate) fn mul_add_mod(lhs: &Scalar, rhs: &Scalar, acc: &Scalar) -> Scalar {
  let product = Scalar52::mul(&scalar52_from_words(lhs), &scalar52_from_words(rhs));
  let sum = Scalar52::add(&product, &scalar52_from_words(acc));
  scalar52_to_words(sum)
}

/// Negate a scalar modulo `L`: returns `L - s` when `s != 0`, else `0`.
#[must_use]
pub(crate) fn negate_mod(s: &Scalar) -> Scalar {
  if *s == ZERO {
    ZERO
  } else {
    let (result, _) = sub_raw(&ORDER, s);
    result
  }
}

/// Decompose a scalar encoding into signed radix-16 digits in `[-8, 8]`.
#[must_use]
#[allow(clippy::indexing_slicing)]
pub(crate) fn as_radix_16(bytes: &[u8; SECRET_KEY_LENGTH]) -> [i8; 64] {
  debug_assert!(bytes[31] <= 127);

  let mut digits = [0i8; 64];

  for (i, byte) in bytes.iter().copied().enumerate() {
    digits[2 * i] = (byte & 0x0F) as i8;
    digits[2 * i + 1] = ((byte >> 4) & 0x0F) as i8;
  }

  for i in 0..63 {
    let carry = (digits[i] + 8) >> 4;
    digits[i] -= carry << 4;
    digits[i + 1] += carry;
  }

  digits
}

#[inline]
#[must_use]
fn read_u64_le(chunk: &[u8]) -> u64 {
  let mut bytes = [0u8; core::mem::size_of::<u64>()];
  for (dst, src) in bytes.iter_mut().zip(chunk.iter().copied()) {
    *dst = src;
  }
  u64::from_le_bytes(bytes)
}

#[inline]
#[must_use]
fn read_words_le_32(bytes: &[u8; 32]) -> [u64; 4] {
  let mut out = [0u64; 4];
  for (dst, chunk) in out.iter_mut().zip(bytes.as_slice().chunks_exact(8)) {
    *dst = read_u64_le(chunk);
  }
  out
}

#[inline]
#[must_use]
fn read_words_le_64(bytes: &[u8; 64]) -> [u64; 8] {
  let mut out = [0u64; 8];
  for (dst, chunk) in out.iter_mut().zip(bytes.as_slice().chunks_exact(8)) {
    *dst = read_u64_le(chunk);
  }
  out
}

#[inline]
#[must_use]
fn reduce_bytes_mod_order_fallback(bytes: &[u8]) -> Scalar {
  let mut acc = ZERO;
  for byte in bytes.iter().rev().copied() {
    let mut shift = 8u32;
    while shift > 0 {
      shift = shift.strict_sub(1);
      acc = add_mod(&acc, &acc);
      if ((byte >> shift) & 1) == 1 {
        acc = add_mod(&acc, &ONE);
      }
    }
  }
  acc
}

#[inline]
#[must_use]
fn scalar52_from_words(words: &Scalar) -> Scalar52 {
  Scalar52::from_bytes(&to_bytes(words))
}

#[inline]
#[must_use]
fn scalar52_to_words(value: Scalar52) -> Scalar {
  decode_words_le(&value.as_bytes())
}

#[inline]
#[must_use]
fn compare(lhs: &Scalar, rhs: &Scalar) -> Ordering {
  for (&left, &right) in lhs.iter().zip(rhs.iter()).rev() {
    if left < right {
      return Ordering::Less;
    }
    if left > right {
      return Ordering::Greater;
    }
  }
  Ordering::Equal
}

#[inline]
#[must_use]
fn add_raw(lhs: &Scalar, rhs: &Scalar) -> (Scalar, u64) {
  let mut out = ZERO;
  let mut carry = 0u128;

  for (dst, (&left, &right)) in out.iter_mut().zip(lhs.iter().zip(rhs.iter())) {
    let sum = u128::from(left).strict_add(u128::from(right)).strict_add(carry);
    *dst = sum as u64;
    carry = sum >> 64;
  }

  (out, carry as u64)
}

#[inline]
#[must_use]
fn sub_raw(lhs: &Scalar, rhs: &Scalar) -> (Scalar, bool) {
  let mut out = ZERO;
  let mut borrow = false;

  for (dst, (&left, &right)) in out.iter_mut().zip(lhs.iter().zip(rhs.iter())) {
    let (mid, borrow1) = left.overflowing_sub(right);
    let (diff, borrow2) = mid.overflowing_sub(u64::from(borrow));
    *dst = diff;
    borrow = borrow1 || borrow2;
  }

  (out, borrow)
}

#[inline]
#[must_use]
fn maybe_sub_order(words: Scalar) -> Scalar {
  if compare(&words, &ORDER) != Ordering::Less {
    let (reduced, _) = sub_raw(&words, &ORDER);
    reduced
  } else {
    words
  }
}

#[cfg(test)]
mod tests {
  use super::{
    ORDER, Scalar, add_mod, as_radix_16, clamp_secret_scalar, decode_words_le, from_canonical_bytes, mul_add_mod,
    reduce_bytes_mod_order, reduce_bytes_mod_order_fallback, to_bytes,
  };

  fn from_u64(value: u64) -> Scalar {
    [value, 0, 0, 0]
  }

  #[test]
  fn clamp_secret_scalar_matches_rfc_8032_bit_rules() {
    let mut bytes = [0xFFu8; 32];
    clamp_secret_scalar(&mut bytes);

    assert_eq!(bytes[0], 0xF8);
    assert_eq!(bytes[31], 0x7F);
  }

  #[test]
  fn decode_words_le_preserves_little_endian_limb_order() {
    let bytes = [
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12,
      0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
    ];

    assert_eq!(
      decode_words_le(&bytes),
      [
        0x0706_0504_0302_0100,
        0x0F0E_0D0C_0B0A_0908,
        0x1716_1514_1312_1110,
        0x1F1E_1D1C_1B1A_1918,
      ]
    );
  }

  #[test]
  fn canonical_scalar_rejects_group_order() {
    let order_bytes = to_bytes(&ORDER);

    assert_eq!(from_canonical_bytes(&order_bytes), None);
  }

  #[test]
  fn reduction_keeps_small_values() {
    let bytes = [7u8];

    assert_eq!(reduce_bytes_mod_order(&bytes), from_u64(7));
  }

  #[test]
  fn wide_reduction_matches_known_vector() {
    let reduced = reduce_bytes_mod_order(&[0xFFu8; 64]);

    assert_eq!(
      to_bytes(&reduced),
      [
        0x00, 0x0F, 0x9C, 0x44, 0xE3, 0x11, 0x06, 0xA4, 0x47, 0x93, 0x85, 0x68, 0xA7, 0x1B, 0x0E, 0xD0, 0x65, 0xBE,
        0xF5, 0x17, 0xD2, 0x73, 0xEC, 0xCE, 0x3D, 0x9A, 0x30, 0x7C, 0x1B, 0x41, 0x99, 0x03,
      ]
    );
  }

  #[test]
  fn wide_reduction_matches_fallback_for_fixed_input() {
    let bytes = core::array::from_fn::<_, 64, _>(|i| (i as u8).wrapping_mul(17).wrapping_add(9));

    assert_eq!(reduce_bytes_mod_order(&bytes), reduce_bytes_mod_order_fallback(&bytes));
  }

  #[test]
  fn radix16_recenters_nibbles_into_signed_digits() {
    let mut bytes = [0u8; 32];
    bytes[0] = 0x19;
    let digits = as_radix_16(&bytes);

    assert_eq!(digits[0], -7);
    assert_eq!(digits[1], 2);
    assert!(digits[2..].iter().all(|digit| *digit == 0));
    assert!(digits[..63].iter().all(|digit| (-8..8).contains(digit)));
    assert!((-8..=8).contains(&digits[63]));
  }

  #[test]
  fn modular_addition_wraps_order_boundary() {
    let near_order = [ORDER[0] - 1, ORDER[1], ORDER[2], ORDER[3]];
    let wrapped = add_mod(&near_order, &from_u64(1));

    assert_eq!(wrapped, [0, 0, 0, 0]);
  }

  #[test]
  fn multiply_add_matches_small_reference() {
    let lhs = from_u64(17);
    let rhs = from_u64(19);
    let acc = from_u64(23);

    assert_eq!(mul_add_mod(&lhs, &rhs, &acc), from_u64(346));
  }
}
