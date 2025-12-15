//! CRC combination operations.
//!
//! This module implements the combine operation for CRC checksums, which
//! computes `crc(A || B)` from `crc(A)`, `crc(B)`, and `len(B)` in O(log n) time.
//!
//! # Mathematical Background
//!
//! CRC can be expressed as polynomial arithmetic in GF(2). Concatenation can be
//! computed using linear algebra: multiplying the first CRC by `x^(8*len(B))`
//! (modulo the generator polynomial) and XORing the second CRC.
//!
//! This implementation uses GF(2) matrix exponentiation (O(log n)), similar to
//! zlib's well-known approach, but written in a more direct "square-and-multiply"
//! style over the *byte* length.

use crate::constants;

/// Combine two CRC32 values for a specific reflected polynomial.
///
/// Given `crc_a = crc(A)` and `crc_b = crc(B)`, computes `crc(A || B)`.
///
/// # Arguments
///
/// * `crc_a` - CRC32-C of first part
/// * `crc_b` - CRC32-C of second part
/// * `len_b` - Length of second part in bytes
///
/// # Algorithm
///
/// Uses GF(2) polynomial exponentiation with square-and-multiply.
/// Time complexity: O(log(len_b))
///
/// Based on zlib's crc32_combine algorithm.
#[inline]
pub fn crc32_combine_poly(poly_reflected: u32, crc_a: u32, crc_b: u32, len_b: usize) -> u32 {
  if len_b == 0 {
    return crc_a;
  }

  // Operator for multiplying by x (one zero bit) modulo the reflected
  // polynomial.
  let mut op = gf2_matrix_one_bit(poly_reflected);

  // Convert to operator for one zero byte (x^8) by squaring three times:
  // x -> x^2 -> x^4 -> x^8.
  op = gf2_matrix_square(&op);
  op = gf2_matrix_square(&op);
  op = gf2_matrix_square(&op);

  // Apply (x^8)^(len_b) to crc_a using square-and-multiply on the byte length.
  let mut crc = crc_a;
  let mut len = len_b as u64;
  while len != 0 {
    if (len & 1) != 0 {
      crc = gf2_matrix_times(&op, crc);
    }
    len >>= 1;
    if len == 0 {
      break;
    }
    op = gf2_matrix_square(&op);
  }

  crc ^ crc_b
}

/// Combine two CRC32-C values.
#[inline]
pub fn crc32c_combine(crc_a: u32, crc_b: u32, len_b: usize) -> u32 {
  crc32_combine_poly(constants::crc32c::POLYNOMIAL, crc_a, crc_b, len_b)
}

/// Combine two CRC32 (ISO-HDLC) values.
#[inline]
pub fn crc32_combine(crc_a: u32, crc_b: u32, len_b: usize) -> u32 {
  crc32_combine_poly(constants::crc32::POLYNOMIAL, crc_a, crc_b, len_b)
}

/// Combine two CRC16/IBM values.
#[inline]
pub fn crc16_ibm_combine(crc_a: u16, crc_b: u16, len_b: usize) -> u16 {
  // CRC16/IBM has `crc(empty) == 0`, so no affine offset is required.
  crc_combine_poly_u64(
    constants::crc16_ibm::POLYNOMIAL as u64,
    16,
    true,
    0,
    crc_a as u64,
    crc_b as u64,
    len_b,
  ) as u16
}

/// Combine two CRC16/CCITT-FALSE values.
#[inline]
pub fn crc16_ccitt_false_combine(crc_a: u16, crc_b: u16, len_b: usize) -> u16 {
  // CRC16/CCITT-FALSE has `crc(empty) == 0xFFFF`, so the CRC function is affine.
  crc_combine_poly_u64(
    constants::crc16_ccitt_false::POLYNOMIAL as u64,
    16,
    false,
    0xFFFF,
    crc_a as u64,
    crc_b as u64,
    len_b,
  ) as u16
}

/// Combine two CRC24/OpenPGP values.
#[inline]
pub fn crc24_openpgp_combine(crc_a: u32, crc_b: u32, len_b: usize) -> u32 {
  // CRC24/OpenPGP has `crc(empty) == 0xB704CE`, so the CRC function is affine.
  crc_combine_poly_u64(
    constants::crc24_openpgp::POLYNOMIAL as u64,
    24,
    false,
    0xB7_04CE,
    crc_a as u64,
    crc_b as u64,
    len_b,
  ) as u32
}

/// Combine two CRC64/XZ values (ECMA polynomial).
#[inline]
pub fn crc64_combine(crc_a: u64, crc_b: u64, len_b: usize) -> u64 {
  crc_combine_poly_u64(constants::crc64::POLYNOMIAL, 64, true, 0, crc_a, crc_b, len_b)
}

/// Combine two CRC64/NVME values.
#[inline]
pub fn crc64_nvme_combine(crc_a: u64, crc_b: u64, len_b: usize) -> u64 {
  crc_combine_poly_u64(constants::crc64_nvme::POLYNOMIAL, 64, true, 0, crc_a, crc_b, len_b)
}

/// Multiply a 32×32 GF(2) matrix by a 32-bit vector.
#[inline]
fn gf2_matrix_times(mat: &[u32; 32], vec: u32) -> u32 {
  let mut sum = 0u32;
  let mut v = vec;

  // Iterate set bits only (faster than checking all 32 bits).
  while v != 0 {
    let bit = v.trailing_zeros() as usize;
    sum ^= mat[bit];
    v &= v - 1;
  }

  sum
}

/// Square a 32×32 GF(2) matrix: `square = mat * mat`.
#[inline]
fn gf2_matrix_square(mat: &[u32; 32]) -> [u32; 32] {
  let mut square = [0u32; 32];
  for i in 0..32 {
    square[i] = gf2_matrix_times(mat, mat[i]);
  }
  square
}

/// Construct the 32×32 operator matrix for multiplying by `x` (one zero bit)
/// modulo the provided reflected polynomial.
#[inline]
fn gf2_matrix_one_bit(poly: u32) -> [u32; 32] {
  let mut mat = [0u32; 32];
  mat[0] = poly;
  let mut row = 1u32;
  for slot in mat.iter_mut().skip(1) {
    *slot = row;
    row <<= 1;
  }
  mat
}

#[inline]
fn mask_width_u64(width: u8) -> u64 {
  if width >= 64 {
    return u64::MAX;
  }
  (1u64 << width) - 1
}

#[inline]
fn step_one_bit_u64(reflected: bool, poly: u64, width: u8, crc: u64) -> u64 {
  if reflected {
    let mask = 0u64.wrapping_sub(crc & 1);
    (crc >> 1) ^ (poly & mask)
  } else {
    let mask = mask_width_u64(width);
    let top = 1u64 << (width as u32 - 1);
    if (crc & top) != 0 {
      ((crc << 1) ^ poly) & mask
    } else {
      (crc << 1) & mask
    }
  }
}

#[inline]
fn gf2_matrix_one_bit_u64(reflected: bool, poly: u64, width: u8) -> [u64; 64] {
  let mut mat = [0u64; 64];
  let mut i = 0u8;
  while i < width {
    let v = 1u64 << i;
    mat[i as usize] = step_one_bit_u64(reflected, poly, width, v);
    i = i.wrapping_add(1);
  }
  mat
}

/// Multiply a width×width GF(2) matrix by a width-bit vector (both in u64).
#[inline]
fn gf2_matrix_times_u64(mat: &[u64; 64], vec: u64) -> u64 {
  let mut sum = 0u64;
  let mut v = vec;

  while v != 0 {
    let bit = v.trailing_zeros() as usize;
    sum ^= mat[bit];
    v &= v - 1;
  }

  sum
}

#[inline]
fn gf2_matrix_square_u64(mat: &[u64; 64], width: u8) -> [u64; 64] {
  let mut square = [0u64; 64];
  let mut i = 0u8;
  while i < width {
    square[i as usize] = gf2_matrix_times_u64(mat, mat[i as usize]);
    i = i.wrapping_add(1);
  }
  square
}

/// Combine two CRC values for the specified polynomial and bit ordering.
///
/// This supports CRC widths up to 64 bits and handles affine CRC parameter
/// sets by converting to/from a linear form using `crc_empty`.
#[inline]
fn crc_combine_poly_u64(
  poly: u64,
  width: u8,
  reflected: bool,
  crc_empty: u64,
  crc_a: u64,
  crc_b: u64,
  len_b: usize,
) -> u64 {
  if len_b == 0 {
    return crc_a;
  }

  let mask = mask_width_u64(width);
  let crc_empty = crc_empty & mask;

  let crc_a = crc_a & mask;
  let crc_b = crc_b & mask;

  // Convert from affine CRC values to a linear form: f(x) = L(x) ^ c.
  let mut crc = crc_a ^ crc_empty;
  let crc_b = crc_b ^ crc_empty;

  // Operator for multiplying by x (one zero bit) modulo the polynomial.
  let mut op = gf2_matrix_one_bit_u64(reflected, poly, width);

  // Convert to operator for one zero byte (x^8) by squaring three times.
  op = gf2_matrix_square_u64(&op, width);
  op = gf2_matrix_square_u64(&op, width);
  op = gf2_matrix_square_u64(&op, width);

  // Apply (x^8)^(len_b) to crc_a using square-and-multiply on the byte length.
  let mut len = len_b as u64;
  while len != 0 {
    if (len & 1) != 0 {
      crc = gf2_matrix_times_u64(&op, crc);
    }
    len >>= 1;
    if len == 0 {
      break;
    }
    op = gf2_matrix_square_u64(&op, width);
  }

  // Combine in linear form, then re-apply the affine offset.
  ((crc ^ crc_b) ^ crc_empty) & mask
}

#[cfg(test)]
mod tests {
  extern crate std;

  use alloc::vec;

  use super::*;
  use crate::{crc16, crc24, crc32::Crc32, crc32c::Crc32c, crc64};

  #[test]
  fn test_combine_simple() {
    let a = b"hello ";
    let b = b"world";
    let ab = b"hello world";

    let crc_a = Crc32c::checksum(a);
    let crc_b = Crc32c::checksum(b);
    let crc_ab = Crc32c::checksum(ab);

    assert_eq!(crc32c_combine(crc_a, crc_b, b.len()), crc_ab);
  }

  #[test]
  fn test_combine_empty_second() {
    let a = b"hello";
    let crc_a = Crc32c::checksum(a);

    // Combining with empty should return first CRC
    assert_eq!(crc32c_combine(crc_a, 0, 0), crc_a);
  }

  #[test]
  fn test_combine_various_lengths() {
    let data = b"The quick brown fox jumps over the lazy dog";

    for split in 1..data.len() {
      let (a, b) = data.split_at(split);
      let crc_a = Crc32c::checksum(a);
      let crc_b = Crc32c::checksum(b);
      let crc_ab = Crc32c::checksum(data);

      assert_eq!(
        crc32c_combine(crc_a, crc_b, b.len()),
        crc_ab,
        "failed at split {}",
        split
      );
    }
  }

  #[test]
  fn test_combine_large() {
    let a = vec![0xABu8; 4096];
    let b = vec![0xCDu8; 4096];
    let mut ab = a.clone();
    ab.extend_from_slice(&b);

    let crc_a = Crc32c::checksum(&a);
    let crc_b = Crc32c::checksum(&b);
    let crc_ab = Crc32c::checksum(&ab);

    assert_eq!(crc32c_combine(crc_a, crc_b, b.len()), crc_ab);
  }

  #[test]
  fn test_crc32_combine_simple() {
    let a = b"hello ";
    let b = b"world";
    let ab = b"hello world";

    let crc_a = Crc32::checksum(a);
    let crc_b = Crc32::checksum(b);
    let crc_ab = Crc32::checksum(ab);

    assert_eq!(crc32_combine(crc_a, crc_b, b.len()), crc_ab);
  }

  #[test]
  fn test_crc32_combine_various_lengths() {
    let data = b"The quick brown fox jumps over the lazy dog";

    for split in 1..data.len() {
      let (a, b) = data.split_at(split);
      let crc_a = Crc32::checksum(a);
      let crc_b = Crc32::checksum(b);
      let crc_ab = Crc32::checksum(data);

      assert_eq!(
        crc32_combine(crc_a, crc_b, b.len()),
        crc_ab,
        "failed at split {}",
        split
      );
    }
  }

  #[test]
  fn test_crc16_ibm_combine_simple() {
    let a = b"hello ";
    let b = b"world";
    let ab = b"hello world";

    let crc_a = crc16::Crc16Ibm::checksum(a);
    let crc_b = crc16::Crc16Ibm::checksum(b);
    let crc_ab = crc16::Crc16Ibm::checksum(ab);

    assert_eq!(crc16_ibm_combine(crc_a, crc_b, b.len()), crc_ab);
  }

  #[test]
  fn test_crc16_ccitt_false_combine_simple() {
    let a = b"hello ";
    let b = b"world";
    let ab = b"hello world";

    let crc_a = crc16::Crc16CcittFalse::checksum(a);
    let crc_b = crc16::Crc16CcittFalse::checksum(b);
    let crc_ab = crc16::Crc16CcittFalse::checksum(ab);

    assert_eq!(crc16_ccitt_false_combine(crc_a, crc_b, b.len()), crc_ab);
  }

  #[test]
  fn test_crc24_openpgp_combine_simple() {
    let a = b"hello ";
    let b = b"world";
    let ab = b"hello world";

    let crc_a = crc24::Crc24::checksum(a);
    let crc_b = crc24::Crc24::checksum(b);
    let crc_ab = crc24::Crc24::checksum(ab);

    assert_eq!(crc24_openpgp_combine(crc_a, crc_b, b.len()), crc_ab);
  }

  #[test]
  fn test_crc64_combine_simple() {
    let a = b"hello ";
    let b = b"world";
    let ab = b"hello world";

    let crc_a = crc64::Crc64::checksum(a);
    let crc_b = crc64::Crc64::checksum(b);
    let crc_ab = crc64::Crc64::checksum(ab);

    assert_eq!(crc64_combine(crc_a, crc_b, b.len()), crc_ab);
  }
}
