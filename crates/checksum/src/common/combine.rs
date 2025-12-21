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
// GF(2) Matrix Types (32-bit CRC)
// ─────────────────────────────────────────────────────────────────────────────

/// A 32x32 GF(2) matrix represented as 32 u32 values.
///
/// Each u32 is a row of the matrix, where bit i represents column i.
#[derive(Clone, Copy)]
pub struct Gf2Matrix32([u32; 32]);

impl Gf2Matrix32 {
  /// Create the identity matrix.
  #[must_use]
  pub const fn identity() -> Self {
    let mut m = [0u32; 32];
    let mut i = 0;
    while i < 32 {
      m[i] = 1 << i;
      i += 1;
    }
    Self(m)
  }

  /// Multiply matrix by a vector (u32 treated as column vector).
  #[inline]
  #[must_use]
  pub const fn mul_vec(self, vec: u32) -> u32 {
    let mut result = 0u32;
    let mut i = 0;
    while i < 32 {
      if vec & (1 << i) != 0 {
        result ^= self.0[i];
      }
      i += 1;
    }
    result
  }

  /// Multiply two matrices (self * other).
  #[must_use]
  pub const fn mul_mat(self, other: Self) -> Self {
    let mut result = [0u32; 32];
    let mut i = 0;
    while i < 32 {
      result[i] = self.mul_vec(other.0[i]);
      i += 1;
    }
    Self(result)
  }

  /// Square the matrix (self * self).
  #[inline]
  #[must_use]
  pub const fn square(self) -> Self {
    self.mul_mat(self)
  }
}

/// Generate the "shift by 1 bit" matrix for a given CRC polynomial.
///
/// For a reflected CRC with polynomial P (reflected form), shifting by 1 bit means:
/// - If the LSB is 0, shift right by 1
/// - If the LSB is 1, shift right by 1, then XOR with P
///
/// The matrix M is such that M * crc = crc_shifted_by_1_bit.
#[must_use]
pub const fn generate_shift1_matrix_32(poly: u32) -> Gf2Matrix32 {
  let mut m = [0u32; 32];

  // For bit position i, determine what happens after one bit shift
  // If we shift crc by 1 bit (reflected): new_crc = (crc >> 1) ^ (poly if crc & 1 else 0)
  // Considering bit j of the input:
  // - If j > 0, bit j contributes to bit j-1 of output
  // - If j = 0, it determines whether we XOR with poly

  // Column 0: if bit 0 is set, result is poly >> 1 (since crc >> 1 contributes nothing at bit 0,
  // but if original bit 0 was set, we XOR with poly after shifting)
  m[0] = poly;

  // Columns 1-31: bit j of input becomes bit j-1 of output (from the >> 1 operation)
  let mut j = 1;
  while j < 32 {
    m[j] = 1 << (j - 1);
    j += 1;
  }

  Gf2Matrix32(m)
}

/// Generate the "shift by 8 bits" matrix (one byte) for a CRC-32 polynomial.
///
/// This is the fundamental building block for combine - it represents what
/// happens to a CRC when 8 zero bits are appended.
#[must_use]
pub const fn generate_shift8_matrix_32(poly: u32) -> Gf2Matrix32 {
  let shift1 = generate_shift1_matrix_32(poly);
  let shift2 = shift1.square();
  let shift4 = shift2.square();

  shift4.square()
}

/// Combine two CRC-32 values.
///
/// Given `crc_a = crc(A)` and `crc_b = crc(B)`, computes `crc(A || B)`.
///
/// # Arguments
///
/// * `crc_a` - CRC of the first part (already finalized/inverted as needed)
/// * `crc_b` - CRC of the second part (already finalized/inverted as needed)
/// * `len_b` - Length of the second part in bytes
/// * `shift8_matrix` - Pre-computed "shift by 8 bits" matrix for the polynomial
///
/// # Algorithm
///
/// Uses square-and-multiply to compute `crc_a * x^(8*len_b)` in O(log len_b) time.
#[must_use]
pub const fn combine_crc32(crc_a: u32, crc_b: u32, len_b: usize, shift8_matrix: Gf2Matrix32) -> u32 {
  if len_b == 0 {
    return crc_a;
  }

  // Start with the shift-by-8-bits matrix
  let mut mat = shift8_matrix;
  let mut result_mat = Gf2Matrix32::identity();
  let mut remaining = len_b;

  // Square-and-multiply to get the matrix for shifting by 8*len_b bits
  while remaining > 0 {
    if remaining & 1 != 0 {
      result_mat = result_mat.mul_mat(mat);
    }
    mat = mat.square();
    remaining >>= 1;
  }

  // Apply the matrix to crc_a and XOR with crc_b
  result_mat.mul_vec(crc_a) ^ crc_b
}

// ─────────────────────────────────────────────────────────────────────────────
// GF(2) Matrix Types (64-bit CRC)
// ─────────────────────────────────────────────────────────────────────────────

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
// GF(2) Matrix Types (24-bit CRC - MSB-first)
// ─────────────────────────────────────────────────────────────────────────────

/// A 24x24 GF(2) matrix for CRC-24 (MSB-first) stored in u32 values.
///
/// Only the lower 24 bits of each u32 are used.
#[derive(Clone, Copy)]
pub struct Gf2Matrix24([u32; 24]);

impl Gf2Matrix24 {
  /// Create the identity matrix.
  #[must_use]
  pub const fn identity() -> Self {
    let mut m = [0u32; 24];
    let mut i = 0;
    while i < 24 {
      m[i] = 1 << i;
      i += 1;
    }
    Self(m)
  }

  /// Multiply matrix by a vector (lower 24 bits of u32).
  #[inline]
  #[must_use]
  pub const fn mul_vec(self, vec: u32) -> u32 {
    let mut result = 0u32;
    let mut i = 0;
    while i < 24 {
      if vec & (1 << i) != 0 {
        result ^= self.0[i];
      }
      i += 1;
    }
    result & 0xFF_FFFF
  }

  /// Multiply two matrices.
  #[must_use]
  pub const fn mul_mat(self, other: Self) -> Self {
    let mut result = [0u32; 24];
    let mut i = 0;
    while i < 24 {
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

/// Generate the "shift by 1 bit" matrix for a CRC-24 polynomial (MSB-first).
///
/// For MSB-first CRC with polynomial P, shifting by 1 bit means:
/// - Shift left by 1
/// - If bit 23 (MSB) was set, XOR with P
#[must_use]
pub const fn generate_shift1_matrix_24_msb(poly: u32) -> Gf2Matrix24 {
  let mut m = [0u32; 24];

  // For MSB-first, bit j shifts to bit j+1, except bit 23 which wraps with poly
  // Column j represents what happens when bit j is set in the input
  let mut j = 0;
  while j < 23 {
    // Bit j shifts to position j+1
    m[j] = 1 << (j + 1);
    j += 1;
  }
  // Bit 23 (MSB): shifts out and XORs with polynomial
  m[23] = poly;

  Gf2Matrix24(m)
}

/// Generate the "shift by 8 bits" matrix for a CRC-24 polynomial (MSB-first).
#[must_use]
pub const fn generate_shift8_matrix_24_msb(poly: u32) -> Gf2Matrix24 {
  let shift1 = generate_shift1_matrix_24_msb(poly);
  let shift2 = shift1.square();
  let shift4 = shift2.square();

  shift4.square()
}

/// Combine two CRC-24 values (MSB-first).
///
/// For MSB-first CRCs with init value, the formula is:
/// `crc(A || B) = shift(crc_a XOR init, len_b) XOR crc_b`
#[must_use]
pub const fn combine_crc24_msb(crc_a: u32, crc_b: u32, len_b: usize, init: u32, shift8_matrix: Gf2Matrix24) -> u32 {
  if len_b == 0 {
    return crc_a;
  }

  // XOR with init to get "raw" state
  let adjusted_a = crc_a ^ init;

  let mut mat = shift8_matrix;
  let mut result_mat = Gf2Matrix24::identity();
  let mut remaining = len_b;

  while remaining > 0 {
    if remaining & 1 != 0 {
      result_mat = result_mat.mul_mat(mat);
    }
    mat = mat.square();
    remaining >>= 1;
  }

  (result_mat.mul_vec(adjusted_a) ^ crc_b) & 0xFF_FFFF
}

// ─────────────────────────────────────────────────────────────────────────────
// GF(2) Matrix Types (16-bit CRC)
// ─────────────────────────────────────────────────────────────────────────────

/// A 16x16 GF(2) matrix represented as 16 u16 values.
#[derive(Clone, Copy)]
pub struct Gf2Matrix16([u16; 16]);

impl Gf2Matrix16 {
  /// Create the identity matrix.
  #[must_use]
  pub const fn identity() -> Self {
    let mut m = [0u16; 16];
    let mut i = 0;
    while i < 16 {
      m[i] = 1 << i;
      i += 1;
    }
    Self(m)
  }

  /// Multiply matrix by a vector.
  #[inline]
  #[must_use]
  pub const fn mul_vec(self, vec: u16) -> u16 {
    let mut result = 0u16;
    let mut i = 0;
    while i < 16 {
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
    let mut result = [0u16; 16];
    let mut i = 0;
    while i < 16 {
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

/// Generate the "shift by 1 bit" matrix for a CRC-16 polynomial.
#[must_use]
pub const fn generate_shift1_matrix_16(poly: u16) -> Gf2Matrix16 {
  let mut m = [0u16; 16];
  m[0] = poly;
  let mut j = 1;
  while j < 16 {
    m[j] = 1 << (j - 1);
    j += 1;
  }
  Gf2Matrix16(m)
}

/// Generate the "shift by 8 bits" matrix for a CRC-16 polynomial.
#[must_use]
pub const fn generate_shift8_matrix_16(poly: u16) -> Gf2Matrix16 {
  let shift1 = generate_shift1_matrix_16(poly);
  let shift2 = shift1.square();
  let shift4 = shift2.square();

  shift4.square()
}

/// Combine two CRC-16 values.
#[must_use]
pub const fn combine_crc16(crc_a: u16, crc_b: u16, len_b: usize, shift8_matrix: Gf2Matrix16) -> u16 {
  if len_b == 0 {
    return crc_a;
  }

  let mut mat = shift8_matrix;
  let mut result_mat = Gf2Matrix16::identity();
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
  use crate::common::tables::{CRC32_IEEE_POLY, CRC32C_POLY, CRC64_XZ_POLY};

  #[test]
  fn test_identity_matrix_32() {
    let id = Gf2Matrix32::identity();
    for i in 0..32 {
      let v = 1u32 << i;
      assert_eq!(id.mul_vec(v), v);
    }
  }

  #[test]
  fn test_identity_matrix_64() {
    let id = Gf2Matrix64::identity();
    for i in 0..64 {
      let v = 1u64 << i;
      assert_eq!(id.mul_vec(v), v);
    }
  }

  #[test]
  fn test_shift1_matrix_32() {
    let poly = CRC32_IEEE_POLY;
    let m = generate_shift1_matrix_32(poly);

    // Shifting 0 should give 0
    assert_eq!(m.mul_vec(0), 0);

    // Shifting 1 (LSB set) should give poly (after the shift and XOR)
    assert_eq!(m.mul_vec(1), poly);

    // Shifting 2 (bit 1 set) should give 1 (bit 1 moves to bit 0)
    assert_eq!(m.mul_vec(2), 1);
  }

  #[test]
  fn test_combine_zero_length() {
    let shift8 = generate_shift8_matrix_32(CRC32_IEEE_POLY);
    let crc_a = 0x12345678;
    let crc_b = 0xDEADBEEF;

    // Combining with zero-length B should return crc_a
    assert_eq!(combine_crc32(crc_a, crc_b, 0, shift8), crc_a);
  }

  #[test]
  fn test_shift8_matrix_crc32c() {
    // Verify that the shift8 matrix is the 8th power of shift1
    let poly = CRC32C_POLY;
    let shift1 = generate_shift1_matrix_32(poly);
    let shift8 = generate_shift8_matrix_32(poly);

    // Compute shift1^8 manually
    let mut m = shift1;
    for _ in 0..7 {
      m = m.mul_mat(shift1);
    }

    // Compare all rows
    for i in 0..32 {
      assert_eq!(shift8.0[i], m.0[i], "Row {i} differs");
    }
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
