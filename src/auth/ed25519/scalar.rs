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

/// Add two scalars modulo `L`.
#[must_use]
pub(crate) fn add_mod(lhs: &Scalar, rhs: &Scalar) -> Scalar {
  let (sum, _) = add_raw(lhs, rhs);
  maybe_sub_order(sum)
}

/// Multiply two scalars modulo `L`.
#[must_use]
pub(crate) fn mul_mod(lhs: &Scalar, rhs: &Scalar) -> Scalar {
  let mut acc = ZERO;
  let mut term = *lhs;

  for mut word in rhs.iter().copied() {
    let mut bit = 64u32;
    while bit > 0 {
      if (word & 1) == 1 {
        acc = add_mod(&acc, &term);
      }
      term = add_mod(&term, &term);
      word >>= 1;
      bit = bit.strict_sub(1);
    }
  }

  acc
}

/// Compute `lhs * rhs + acc (mod L)`.
#[must_use]
pub(crate) fn mul_add_mod(lhs: &Scalar, rhs: &Scalar, acc: &Scalar) -> Scalar {
  add_mod(&mul_mod(lhs, rhs), acc)
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
    ORDER, Scalar, add_mod, clamp_secret_scalar, decode_words_le, from_canonical_bytes, mul_add_mod,
    reduce_bytes_mod_order, to_bytes,
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
