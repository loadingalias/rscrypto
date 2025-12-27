//! GF(2) matrix operations for CRC combination.
//!
//! When you have `crc(A)` and `crc(B)`, you can compute `crc(A || B)` without
//! reprocessing `A`. This is done in O(log n) time using matrix exponentiation
//! over GF(2).
//!
//! # Mathematical Background
//!
//! For reflected CRCs:
//! ```text
//! crc(A || B) = crc(A) * x^(8*len(B)) mod G(x) XOR crc(B)
//! ```
//!
//! The multiplication by `x^(8*len(B))` is computed as matrix multiplication
//! where the matrix represents the effect of shifting by `8*len(B)` bits.

// SAFETY: All array indexing in this module uses bounded loop indices (0..N where N is the
// array size). Clippy cannot prove this in const fn contexts, but the bounds are statically
// guaranteed by the loop conditions.
#![allow(clippy::indexing_slicing)]

// ─────────────────────────────────────────────────────────────────────────────
// GF(2) Matrix Types (64-bit CRC)
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// GF(2) Matrix Types (32-bit CRC)
// ─────────────────────────────────────────────────────────────────────────────

/// A 32x32 GF(2) matrix represented as 32 u32 values.
#[derive(Clone, Copy)]
pub struct Gf2Matrix32([u32; 32]);

impl Gf2Matrix32 {
  /// Create the identity matrix.
  #[must_use]
  pub const fn identity() -> Self {
    let mut m = [0u32; 32];
    let mut i: u32 = 0;
    while i < 32 {
      m[i as usize] = 1u32 << i;
      i = i.strict_add(1);
    }
    Self(m)
  }

  /// Multiply matrix by a vector.
  #[inline]
  #[must_use]
  pub const fn mul_vec(self, vec: u32) -> u32 {
    let mut result = 0u32;
    let mut i: u32 = 0;
    while i < 32 {
      if vec & (1u32 << i) != 0 {
        result ^= self.0[i as usize];
      }
      i = i.strict_add(1);
    }
    result
  }

  /// Multiply two matrices.
  #[must_use]
  pub const fn mul_mat(self, other: Self) -> Self {
    let mut result = [0u32; 32];
    let mut i: u32 = 0;
    while i < 32 {
      result[i as usize] = self.mul_vec(other.0[i as usize]);
      i = i.strict_add(1);
    }
    Self(result)
  }

  /// Square the matrix.
  #[inline]
  #[must_use]
  pub const fn square(self) -> Self {
    self.mul_mat(self)
  }
}

/// Generate the "shift by 1 bit" matrix for a CRC-32 polynomial.
#[must_use]
pub const fn generate_shift1_matrix_32(poly: u32) -> Gf2Matrix32 {
  let mut m = [0u32; 32];
  m[0] = poly;
  let mut j: u32 = 1;
  while j < 32 {
    m[j as usize] = 1u32 << (j - 1);
    j = j.strict_add(1);
  }
  Gf2Matrix32(m)
}

/// Generate the "shift by 8 bits" matrix for a CRC-32 polynomial.
#[must_use]
pub const fn generate_shift8_matrix_32(poly: u32) -> Gf2Matrix32 {
  let shift1 = generate_shift1_matrix_32(poly);
  let shift2 = shift1.square();
  let shift4 = shift2.square();

  shift4.square()
}

/// Combine two CRC-32 values.
#[must_use]
pub const fn combine_crc32(crc_a: u32, crc_b: u32, len_b: usize, shift8_matrix: Gf2Matrix32) -> u32 {
  if len_b == 0 {
    return crc_a;
  }

  let mut mat = shift8_matrix;
  let mut result_mat = Gf2Matrix32::identity();
  let mut remaining = len_b;

  while remaining > 0 {
    if remaining & 1 != 0 {
      result_mat = result_mat.mul_mat(mat);
    }
    mat = mat.square();
    remaining >>= 1;
  }

  result_mat.mul_vec(crc_a) ^ crc_b
}

/// A 64x64 GF(2) matrix represented as 64 u64 values.
#[derive(Clone, Copy)]
pub struct Gf2Matrix64([u64; 64]);

impl Gf2Matrix64 {
  /// Create the identity matrix.
  #[must_use]
  pub const fn identity() -> Self {
    let mut m = [0u64; 64];
    let mut i = 0;
    while i < 64 {
      m[i] = 1 << i;
      i += 1;
    }
    Self(m)
  }

  /// Multiply matrix by a vector.
  #[inline]
  #[must_use]
  pub const fn mul_vec(self, vec: u64) -> u64 {
    let mut result = 0u64;
    let mut i = 0;
    while i < 64 {
      if vec & (1 << i) != 0 {
        result ^= self.0[i];
      }
      i += 1;
    }
    result
  }

  /// Multiply two matrices.
  #[must_use]
  pub const fn mul_mat(self, other: Self) -> Self {
    let mut result = [0u64; 64];
    let mut i = 0;
    while i < 64 {
      result[i] = self.mul_vec(other.0[i]);
      i += 1;
    }
    Self(result)
  }

  /// Square the matrix.
  #[inline]
  #[must_use]
  pub const fn square(self) -> Self {
    self.mul_mat(self)
  }
}

/// Generate the "shift by 1 bit" matrix for a CRC-64 polynomial.
#[must_use]
pub const fn generate_shift1_matrix_64(poly: u64) -> Gf2Matrix64 {
  let mut m = [0u64; 64];
  m[0] = poly;
  let mut j = 1;
  while j < 64 {
    m[j] = 1 << (j - 1);
    j += 1;
  }
  Gf2Matrix64(m)
}

/// Generate the "shift by 8 bits" matrix for a CRC-64 polynomial.
#[must_use]
pub const fn generate_shift8_matrix_64(poly: u64) -> Gf2Matrix64 {
  let shift1 = generate_shift1_matrix_64(poly);
  let shift2 = shift1.square();
  let shift4 = shift2.square();

  shift4.square()
}

/// Combine two CRC-64 values.
#[must_use]
pub const fn combine_crc64(crc_a: u64, crc_b: u64, len_b: usize, shift8_matrix: Gf2Matrix64) -> u64 {
  if len_b == 0 {
    return crc_a;
  }

  let mut mat = shift8_matrix;
  let mut result_mat = Gf2Matrix64::identity();
  let mut remaining = len_b;

  while remaining > 0 {
    if remaining & 1 != 0 {
      result_mat = result_mat.mul_mat(mat);
    }
    mat = mat.square();
    remaining >>= 1;
  }

  result_mat.mul_vec(crc_a) ^ crc_b
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;
  use crate::common::tables::{CRC32_IEEE_POLY, CRC64_XZ_POLY};

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-32 Tests
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_identity_matrix_32() {
    let id = Gf2Matrix32::identity();
    for i in 0..32 {
      let v = 1u32 << i;
      assert_eq!(id.mul_vec(v), v);
    }
  }

  #[test]
  fn test_shift1_matrix_32() {
    let poly = CRC32_IEEE_POLY;
    let m = generate_shift1_matrix_32(poly);

    assert_eq!(m.mul_vec(0), 0);
    assert_eq!(m.mul_vec(1), poly);
    assert_eq!(m.mul_vec(2), 1);
  }

  #[test]
  fn test_combine_zero_length_32() {
    let shift8 = generate_shift8_matrix_32(CRC32_IEEE_POLY);
    let crc_a = 0x1234_5678;
    let crc_b = 0xDEAD_BEEF;
    assert_eq!(combine_crc32(crc_a, crc_b, 0, shift8), crc_a);
  }

  #[test]
  fn test_shift8_matrix_crc32() {
    let poly = CRC32_IEEE_POLY;
    let shift1 = generate_shift1_matrix_32(poly);
    let shift8 = generate_shift8_matrix_32(poly);

    let mut m = shift1;
    for _ in 0..7 {
      m = m.mul_mat(shift1);
    }

    for i in 0..32usize {
      assert_eq!(shift8.0[i], m.0[i], "Row {i} differs");
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-64 Tests
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_identity_matrix_64() {
    let id = Gf2Matrix64::identity();
    for i in 0..64 {
      let v = 1u64 << i;
      assert_eq!(id.mul_vec(v), v);
    }
  }

  #[test]
  fn test_shift1_matrix_64() {
    let poly = CRC64_XZ_POLY;
    let m = generate_shift1_matrix_64(poly);

    // Shifting 0 should give 0
    assert_eq!(m.mul_vec(0), 0);

    // Shifting 1 (LSB set) should give poly (after the shift and XOR)
    assert_eq!(m.mul_vec(1), poly);

    // Shifting 2 (bit 1 set) should give 1 (bit 1 moves to bit 0)
    assert_eq!(m.mul_vec(2), 1);
  }

  #[test]
  fn test_combine_zero_length() {
    let shift8 = generate_shift8_matrix_64(CRC64_XZ_POLY);
    let crc_a = 0x1234_5678_9ABC_DEF0;
    let crc_b = 0xDEAD_BEEF_CAFE_BABE;

    // Combining with zero-length B should return crc_a
    assert_eq!(combine_crc64(crc_a, crc_b, 0, shift8), crc_a);
  }

  #[test]
  fn test_shift8_matrix_crc64() {
    let poly = CRC64_XZ_POLY;
    let shift1 = generate_shift1_matrix_64(poly);
    let shift8 = generate_shift8_matrix_64(poly);

    let mut m = shift1;
    for _ in 0..7 {
      m = m.mul_mat(shift1);
    }

    for i in 0..64 {
      assert_eq!(shift8.0[i], m.0[i], "Row {i} differs");
    }
  }
}
