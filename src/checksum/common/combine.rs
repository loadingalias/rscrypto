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
// Unified GF(2) matrix macro
// ─────────────────────────────────────────────────────────────────────────────

/// Defines a full GF(2) matrix type together with its shift and combine functions.
///
/// # Variants
///
/// - `full_width, reflected` -- dimension equals backing-type bit width, reflected CRC
///   (shift-right: poly at row 0, `m[j] = 1 << (j-1)` for `j >= 1`).
///
/// - `full_width, reflected, combine_with_init_xorout` -- same matrix, but the combine
///   function takes an extra `init_xorout` parameter.
///
/// - `sub_width, msb_first, combine_with_init_xorout` -- dimension < backing-type bits,
///   a `MASK` constant is generated, MSB-first shift (shift-left: `m[j] = 1 << (j+1)`,
///   poly at last row), and the combine function applies masking and `init_xorout`.
macro_rules! define_gf2_combine {
  // ── full-width, reflected, simple combine ───────────────────────────────
  (
    name: $Name:ident,
    backing: $T:ty,
    dim: $DIM:expr,
    doc_matrix: $doc_matrix:expr,

    shift1_fn: $shift1_fn:ident,
    doc_shift1: $doc_shift1:expr,

    shift8_fn: $shift8_fn:ident,
    doc_shift8: $doc_shift8:expr,

    combine_fn: $combine_fn:ident,
    doc_combine: $doc_combine:expr,

    full_width, reflected
  ) => {
    #[doc = $doc_matrix]
    #[derive(Clone, Copy)]
    pub struct $Name([$T; $DIM]);

    impl $Name {
      /// Create the identity matrix.
      #[must_use]
      pub const fn identity() -> Self {
        let mut m = [0 as $T; $DIM];
        let mut i: u32 = 0;
        while i < $DIM as u32 {
          m[i as usize] = (1 as $T).strict_shl(i);
          i = i.strict_add(1);
        }
        Self(m)
      }

      /// Multiply matrix by a vector.
      #[inline]
      #[must_use]
      pub const fn mul_vec(self, vec: $T) -> $T {
        let mut result = 0 as $T;
        let mut i: u32 = 0;
        while i < $DIM as u32 {
          if vec & (1 as $T).strict_shl(i) != 0 {
            result ^= self.0[i as usize];
          }
          i = i.strict_add(1);
        }
        result
      }

      /// Multiply two matrices.
      #[must_use]
      pub const fn mul_mat(self, other: Self) -> Self {
        let mut result = [0 as $T; $DIM];
        let mut i: u32 = 0;
        while i < $DIM as u32 {
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

    #[doc = $doc_shift1]
    #[must_use]
    pub const fn $shift1_fn(poly: $T) -> $Name {
      let mut m = [0 as $T; $DIM];
      m[0] = poly;
      let mut j: u32 = 1;
      while j < $DIM as u32 {
        m[j as usize] = (1 as $T).strict_shl(j.strict_sub(1));
        j = j.strict_add(1);
      }
      $Name(m)
    }

    #[doc = $doc_shift8]
    #[must_use]
    pub const fn $shift8_fn(poly: $T) -> $Name {
      let shift1 = $shift1_fn(poly);
      let shift2 = shift1.square();
      let shift4 = shift2.square();
      shift4.square()
    }

    #[doc = $doc_combine]
    #[must_use]
    pub const fn $combine_fn(
      crc_a: $T,
      crc_b: $T,
      len_b: usize,
      shift8_matrix: $Name,
    ) -> $T {
      if len_b == 0 {
        return crc_a;
      }

      let mut mat = shift8_matrix;
      let mut result_mat = $Name::identity();
      let mut remaining = len_b;

      while remaining > 0 {
        if remaining & 1 != 0 {
          result_mat = result_mat.mul_mat(mat);
        }
        mat = mat.square();
        remaining = remaining.strict_shr(1);
      }

      result_mat.mul_vec(crc_a) ^ crc_b
    }
  };

  // ── full-width, reflected, combine with init_xorout ─────────────────────
  (
    name: $Name:ident,
    backing: $T:ty,
    dim: $DIM:expr,
    doc_matrix: $doc_matrix:expr,

    shift1_fn: $shift1_fn:ident,
    doc_shift1: $doc_shift1:expr,

    shift8_fn: $shift8_fn:ident,
    doc_shift8: $doc_shift8:expr,

    combine_fn: $combine_fn:ident,
    doc_combine: $doc_combine:expr,

    full_width, reflected, combine_with_init_xorout
  ) => {
    #[doc = $doc_matrix]
    #[derive(Clone, Copy)]
    pub struct $Name([$T; $DIM]);

    impl $Name {
      /// Create the identity matrix.
      #[must_use]
      pub const fn identity() -> Self {
        let mut m = [0 as $T; $DIM];
        let mut i: u32 = 0;
        while i < $DIM as u32 {
          m[i as usize] = (1 as $T).strict_shl(i);
          i = i.strict_add(1);
        }
        Self(m)
      }

      /// Multiply matrix by a vector.
      #[inline]
      #[must_use]
      pub const fn mul_vec(self, vec: $T) -> $T {
        let mut result = 0 as $T;
        let mut i: u32 = 0;
        while i < $DIM as u32 {
          if vec & (1 as $T).strict_shl(i) != 0 {
            result ^= self.0[i as usize];
          }
          i = i.strict_add(1);
        }
        result
      }

      /// Multiply two matrices.
      #[must_use]
      pub const fn mul_mat(self, other: Self) -> Self {
        let mut result = [0 as $T; $DIM];
        let mut i: u32 = 0;
        while i < $DIM as u32 {
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

    #[doc = $doc_shift1]
    #[must_use]
    pub const fn $shift1_fn(poly: $T) -> $Name {
      let mut m = [0 as $T; $DIM];
      m[0] = poly;
      let mut j: u32 = 1;
      while j < $DIM as u32 {
        m[j as usize] = (1 as $T).strict_shl(j.strict_sub(1));
        j = j.strict_add(1);
      }
      $Name(m)
    }

    #[doc = $doc_shift8]
    #[must_use]
    pub const fn $shift8_fn(poly: $T) -> $Name {
      let shift1 = $shift1_fn(poly);
      let shift2 = shift1.square();
      let shift4 = shift2.square();
      shift4.square()
    }

    #[doc = $doc_combine]
    ///
    /// This works for any init/xorout as long as the caller supplies
    /// `init_xorout = init ^ xorout` for the CRC variant being combined.
    #[must_use]
    pub const fn $combine_fn(
      crc_a: $T,
      crc_b: $T,
      len_b: usize,
      shift8_matrix: $Name,
      init_xorout: $T,
    ) -> $T {
      if len_b == 0 {
        return crc_a;
      }

      let mut mat = shift8_matrix;
      let mut result_mat = $Name::identity();
      let mut remaining = len_b;

      while remaining > 0 {
        if remaining & 1 != 0 {
          result_mat = result_mat.mul_mat(mat);
        }
        mat = mat.square();
        remaining = remaining.strict_shr(1);
      }

      result_mat.mul_vec(crc_a) ^ crc_b ^ result_mat.mul_vec(init_xorout)
    }
  };

  // ── sub-width, msb-first, combine with init_xorout + masking ────────────
  (
    name: $Name:ident,
    backing: $T:ty,
    dim: $DIM:expr,
    mask: $MASK:expr,
    doc_matrix: $doc_matrix:expr,

    shift1_fn: $shift1_fn:ident,
    doc_shift1: $doc_shift1:expr,

    shift8_fn: $shift8_fn:ident,
    doc_shift8: $doc_shift8:expr,

    combine_fn: $combine_fn:ident,
    doc_combine: $doc_combine:expr,

    sub_width, msb_first, combine_with_init_xorout
  ) => {
    #[doc = $doc_matrix]
    #[derive(Clone, Copy)]
    pub struct $Name([$T; $DIM]);

    impl $Name {
      /// Bit mask that keeps only the low `DIM` bits of the backing type.
      pub const MASK: $T = $MASK;

      /// Create the identity matrix.
      #[must_use]
      pub const fn identity() -> Self {
        let mut m = [0 as $T; $DIM];
        let mut i: u32 = 0;
        while i < $DIM as u32 {
          m[i as usize] = (1 as $T).strict_shl(i) & Self::MASK;
          i = i.strict_add(1);
        }
        Self(m)
      }

      /// Multiply matrix by a vector (low bits only).
      #[inline]
      #[must_use]
      pub const fn mul_vec(self, vec: $T) -> $T {
        let vec = vec & Self::MASK;
        let mut result = 0 as $T;
        let mut i: u32 = 0;
        while i < $DIM as u32 {
          if vec & ((1 as $T).strict_shl(i) & Self::MASK) != 0 {
            result ^= self.0[i as usize];
          }
          i = i.strict_add(1);
        }
        result & Self::MASK
      }

      /// Multiply two matrices.
      #[must_use]
      pub const fn mul_mat(self, other: Self) -> Self {
        let mut result = [0 as $T; $DIM];
        let mut i: u32 = 0;
        while i < $DIM as u32 {
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

    #[doc = $doc_shift1]
    #[must_use]
    pub const fn $shift1_fn(poly: $T) -> $Name {
      let poly = poly & $Name::MASK;
      let mut m = [0 as $T; $DIM];

      // Shift-left: bit i moves to i+1; top bit reduces by poly.
      let mut j: u32 = 0;
      while j < ($DIM as u32).strict_sub(1) {
        m[j as usize] = (1 as $T).strict_shl(j.strict_add(1)) & $Name::MASK;
        j = j.strict_add(1);
      }
      m[$DIM - 1] = poly;

      $Name(m)
    }

    #[doc = $doc_shift8]
    #[must_use]
    pub const fn $shift8_fn(poly: $T) -> $Name {
      let shift1 = $shift1_fn(poly);
      let shift2 = shift1.square();
      let shift4 = shift2.square();
      shift4.square()
    }

    #[doc = $doc_combine]
    ///
    /// This works for any init/xorout as long as the caller supplies
    /// `init_xorout = init ^ xorout` for the CRC variant being combined.
    #[must_use]
    pub const fn $combine_fn(
      crc_a: $T,
      crc_b: $T,
      len_b: usize,
      shift8_matrix: $Name,
      init_xorout: $T,
    ) -> $T {
      if len_b == 0 {
        return crc_a & $Name::MASK;
      }

      let mut mat = shift8_matrix;
      let mut result_mat = $Name::identity();
      let mut remaining = len_b;

      while remaining > 0 {
        if remaining & 1 != 0 {
          result_mat = result_mat.mul_mat(mat);
        }
        mat = mat.square();
        remaining = remaining.strict_shr(1);
      }

      (result_mat.mul_vec(crc_a)
        ^ (crc_b & $Name::MASK)
        ^ result_mat.mul_vec(init_xorout))
        & $Name::MASK
    }
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// GF(2) Matrix Types (16-bit CRC)
// ─────────────────────────────────────────────────────────────────────────────

define_gf2_combine! {
  name: Gf2Matrix16,
  backing: u16,
  dim: 16,
  doc_matrix: "A 16x16 GF(2) matrix represented as 16 u16 values.",

  shift1_fn: generate_shift1_matrix_16,
  doc_shift1: "Generate the \"shift by 1 bit\" matrix for a reflected CRC-16 polynomial.",

  shift8_fn: generate_shift8_matrix_16,
  doc_shift8: "Generate the \"shift by 8 bits\" matrix for a reflected CRC-16 polynomial.",

  combine_fn: combine_crc16,
  doc_combine: "Combine two CRC-16 values.",

  full_width, reflected, combine_with_init_xorout
}

// ─────────────────────────────────────────────────────────────────────────────
// GF(2) Matrix Types (24-bit CRC)
// ─────────────────────────────────────────────────────────────────────────────

define_gf2_combine! {
  name: Gf2Matrix24,
  backing: u32,
  dim: 24,
  mask: 0x00FF_FFFF,
  doc_matrix: "A 24x24 GF(2) matrix represented as 24 u32 values (low 24 bits used).",

  shift1_fn: generate_shift1_matrix_24,
  doc_shift1: "Generate the \"shift by 1 bit\" matrix for an MSB-first CRC-24 polynomial.",

  shift8_fn: generate_shift8_matrix_24,
  doc_shift8: "Generate the \"shift by 8 bits\" matrix for an MSB-first CRC-24 polynomial.",

  combine_fn: combine_crc24,
  doc_combine: "Combine two CRC-24 values (low 24 bits).",

  sub_width, msb_first, combine_with_init_xorout
}

// ─────────────────────────────────────────────────────────────────────────────
// GF(2) Matrix Types (32-bit CRC)
// ─────────────────────────────────────────────────────────────────────────────

define_gf2_combine! {
  name: Gf2Matrix32,
  backing: u32,
  dim: 32,
  doc_matrix: "A 32x32 GF(2) matrix represented as 32 u32 values.",

  shift1_fn: generate_shift1_matrix_32,
  doc_shift1: "Generate the \"shift by 1 bit\" matrix for a CRC-32 polynomial.",

  shift8_fn: generate_shift8_matrix_32,
  doc_shift8: "Generate the \"shift by 8 bits\" matrix for a CRC-32 polynomial.",

  combine_fn: combine_crc32,
  doc_combine: "Combine two CRC-32 values.",

  full_width, reflected
}

/// Compute the matrix that applies a shift of `len_bytes` bytes for a reflected CRC-32.
///
/// This returns `M = (shift8_matrix)^(len_bytes)` where multiplication is in
/// GF(2), so `M.mul_vec(crc)` computes `crc * x^(8*len_bytes) mod G(x)`.
#[inline]
#[must_use]
#[cfg(target_arch = "aarch64")]
pub const fn pow_shift8_matrix_32(len_bytes: usize, shift8_matrix: Gf2Matrix32) -> Gf2Matrix32 {
  if len_bytes == 0 {
    return Gf2Matrix32::identity();
  }

  let mut mat = shift8_matrix;
  let mut result_mat = Gf2Matrix32::identity();
  let mut remaining = len_bytes;

  while remaining > 0 {
    if remaining & 1 != 0 {
      result_mat = result_mat.mul_mat(mat);
    }
    mat = mat.square();
    remaining = remaining.strict_shr(1);
  }

  result_mat
}

// ─────────────────────────────────────────────────────────────────────────────
// GF(2) Matrix Types (64-bit CRC)
// ─────────────────────────────────────────────────────────────────────────────

define_gf2_combine! {
  name: Gf2Matrix64,
  backing: u64,
  dim: 64,
  doc_matrix: "A 64x64 GF(2) matrix represented as 64 u64 values.",

  shift1_fn: generate_shift1_matrix_64,
  doc_shift1: "Generate the \"shift by 1 bit\" matrix for a CRC-64 polynomial.",

  shift8_fn: generate_shift8_matrix_64,
  doc_shift8: "Generate the \"shift by 8 bits\" matrix for a CRC-64 polynomial.",

  combine_fn: combine_crc64,
  doc_combine: "Combine two CRC-64 values.",

  full_width, reflected
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;
  use crate::checksum::common::tables::{CRC16_CCITT_POLY, CRC24_OPENPGP_POLY, CRC32_IEEE_POLY, CRC64_XZ_POLY};

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
  // CRC-16 Tests
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_combine_crc16_x25_split() {
    // CRC-16/X25 (IBM-SDLC): init=0xFFFF, xorout=0xFFFF, refin/refout=true.
    const INIT_XOROUT: u16 = 0;
    let shift8 = generate_shift8_matrix_16(CRC16_CCITT_POLY);

    fn crc16_x25(data: &[u8]) -> u16 {
      let mut crc: u16 = 0xFFFF;
      for &b in data {
        crc ^= b as u16;
        for _ in 0..8 {
          if crc & 1 != 0 {
            crc = (crc >> 1) ^ CRC16_CCITT_POLY;
          } else {
            crc >>= 1;
          }
        }
      }
      crc ^ 0xFFFF
    }

    let data = b"123456789";
    let (a, b) = data.split_at(4);
    let crc_a = crc16_x25(a);
    let crc_b = crc16_x25(b);
    let combined = combine_crc16(crc_a, crc_b, b.len(), shift8, INIT_XOROUT);
    assert_eq!(combined, crc16_x25(data));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-24 Tests
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_combine_crc24_openpgp_split() {
    // CRC-24/OPENPGP: init=0xB704CE, xorout=0, refin/refout=false.
    const INIT_XOROUT: u32 = 0x00B7_04CE;
    let shift8 = generate_shift8_matrix_24(CRC24_OPENPGP_POLY);

    fn crc24_openpgp(data: &[u8]) -> u32 {
      let poly_aligned = CRC24_OPENPGP_POLY << 8;
      let mut state: u32 = 0x00B7_04CE << 8;
      for &byte in data {
        state ^= (byte as u32) << 24;
        for _ in 0..8 {
          if state & 0x8000_0000 != 0 {
            state = (state << 1) ^ poly_aligned;
          } else {
            state <<= 1;
          }
        }
      }
      (state >> 8) & 0x00FF_FFFF
    }

    let data = b"123456789";
    let (a, b) = data.split_at(4);
    let crc_a = crc24_openpgp(a);
    let crc_b = crc24_openpgp(b);
    let combined = combine_crc24(crc_a, crc_b, b.len(), shift8, INIT_XOROUT);
    assert_eq!(combined, crc24_openpgp(data));
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
