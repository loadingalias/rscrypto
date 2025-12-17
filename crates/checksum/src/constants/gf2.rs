//! GF(2) polynomial arithmetic for CRC constant generation.
//!
//! This module provides const-time GF(2) operations used to compute
//! folding coefficients, Barrett reduction constants, and other SIMD
//! acceleration parameters at compile time.
//!
//! All functions are generic over polynomial width (32-bit or 64-bit CRCs).

// Bit Reflection (used by all CRC variants)

/// Reflect (bit-reverse) the lower `width` bits of `value`.
///
/// For example, `reflect_bits_u64(0b1010, 4)` returns `0b0101`.
#[inline]
pub const fn reflect_bits_u64(value: u64, width: u32) -> u64 {
  let mut result = 0u64;
  let mut i = 0u32;
  while i < width {
    if (value >> i) & 1 != 0 {
      result |= 1 << (width - 1 - i);
    }
    i += 1;
  }
  result
}

/// Reflect (bit-reverse) the lower `width` bits of a 128-bit value.
#[inline]
pub const fn reflect_bits_u128(value: u128, width: u32) -> u128 {
  let mut result = 0u128;
  let mut i = 0u32;
  while i < width {
    if (value >> i) & 1 != 0 {
      result |= 1u128 << (width - 1 - i);
    }
    i += 1;
  }
  result
}

// Polynomial Degree and Division

/// Returns the degree of a polynomial (position of highest set bit).
///
/// For example, `deg_u128(0b1000)` returns `3`.
#[inline]
pub const fn deg_u128(x: u128) -> u32 {
  // Degree is (bit width - leading zeros - 1), but we need to handle x=0.
  // For our use cases x is never 0, but we handle it gracefully.
  if x == 0 {
    return 0;
  }
  (128u32 - x.leading_zeros()) - 1
}

/// GF(2) polynomial division: `dividend / divisor`, returning quotient.
///
/// Both operands and result are treated as polynomials over GF(2).
/// Used for computing Barrett reduction constants.
#[inline]
pub const fn gf2_div_128(dividend: u128, divisor: u128) -> u128 {
  let mut dd = dividend;
  let mut q: u128 = 0;

  while dd != 0 && deg_u128(dd) >= deg_u128(divisor) {
    let shift = deg_u128(dd) - deg_u128(divisor);
    q ^= 1u128 << shift;
    dd ^= divisor << shift;
  }

  q
}

// 32-bit CRC Operations (CRC32, CRC32C)

/// Multiply two polynomials in GF(2) and reduce mod `poly_full`.
///
/// For 32-bit CRCs, `poly_full` is the 33-bit polynomial with explicit x^32 term
/// (e.g., `0x1_04C1_1DB7` for CRC32, `0x1_1EDC_6F41` for CRC32C).
///
/// Both inputs are at most 32-bit polynomials; output is at most 32 bits.
#[inline]
pub const fn gf2_mul_mod_32(a: u64, b: u64, poly_full: u64) -> u64 {
  // Carryless multiply (schoolbook, max 32x32 -> 63 bits).
  let mut product: u64 = 0;
  let mut i = 0;
  while i < 32 {
    if (b >> i) & 1 != 0 {
      product ^= a << i;
    }
    i += 1;
  }

  // Reduce mod poly_full - product can be up to 63 bits.
  // We reduce from high bits down.
  let mut result = product;
  let mut bit = 63i32;
  while bit >= 32 {
    if (result >> (bit as u32)) & 1 != 0 {
      // Subtract poly_full * x^(bit-32)
      result ^= poly_full << ((bit - 32) as u32);
    }
    bit -= 1;
  }

  result & 0xFFFF_FFFF
}

/// Compute x^n mod poly_full in GF(2) for 32-bit CRC polynomials.
///
/// Uses square-and-multiply algorithm. Result is a 32-bit polynomial
/// coefficient represented as u64 for use with PCLMUL/PMULL.
#[inline]
pub const fn xpow_mod_32(n: u32, poly_full: u64) -> u64 {
  if n < 32 {
    return 1u64 << n;
  }

  // Square-and-multiply: x^n = x^(n/2)^2 * x^(n mod 2)
  let mut result: u64 = 1; // x^0
  let mut base: u64 = 2; // x^1
  let mut exp = n;

  while exp > 0 {
    if exp & 1 != 0 {
      result = gf2_mul_mod_32(result, base, poly_full);
    }
    base = gf2_mul_mod_32(base, base, poly_full);
    exp >>= 1;
  }

  result
}

// 64-bit CRC Operations (CRC64/XZ, CRC64/NVME)

/// Multiply two 64-bit polynomials in GF(2) and reduce mod `poly_full`.
///
/// For 64-bit CRCs, `poly_full` is the 65-bit polynomial with explicit x^64 term
/// (e.g., `(1u128 << 64) | normal_poly`).
///
/// Both inputs are at most 64-bit polynomials; output is at most 64 bits.
#[inline]
pub const fn gf2_mul_mod_64(a: u64, b: u64, poly_full: u128) -> u64 {
  // Carryless multiply (schoolbook, max 64x64 -> 127 bits).
  let mut product: u128 = 0;
  let mut i = 0;
  while i < 64 {
    if (b >> i) & 1 != 0 {
      product ^= (a as u128) << i;
    }
    i += 1;
  }

  // Reduce mod poly_full - product can be up to 127 bits.
  let mut result = product;
  let mut bit: i32 = 127;
  while bit >= 64 {
    if (result >> (bit as u32)) & 1 != 0 {
      result ^= poly_full << ((bit - 64) as u32);
    }
    bit -= 1;
  }

  result as u64
}

/// Compute x^n mod poly_full in GF(2) for 64-bit CRC polynomials.
///
/// Uses square-and-multiply algorithm.
#[inline]
pub const fn xpow_mod_64(n: u32, poly_full: u128) -> u64 {
  if n < 64 {
    return 1u64 << n;
  }

  let mut result: u64 = 1;
  let mut base: u64 = 2;
  let mut exp = n;

  while exp > 0 {
    if exp & 1 != 0 {
      result = gf2_mul_mod_64(result, base, poly_full);
    }
    base = gf2_mul_mod_64(base, base, poly_full);
    exp >>= 1;
  }

  result
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_reflect_bits_u64() {
    assert_eq!(reflect_bits_u64(0b1010, 4), 0b0101);
    assert_eq!(reflect_bits_u64(0b1000_0000, 8), 0b0000_0001);
    assert_eq!(reflect_bits_u64(0xFFFF_FFFF, 32), 0xFFFF_FFFF);
  }

  #[test]
  fn test_reflect_bits_u128() {
    assert_eq!(reflect_bits_u128(0b1010, 4), 0b0101);
    assert_eq!(reflect_bits_u128(1u128 << 64, 65), 1u128);
  }

  #[test]
  fn test_deg_u128() {
    assert_eq!(deg_u128(1), 0);
    assert_eq!(deg_u128(0b1000), 3);
    assert_eq!(deg_u128(1u128 << 64), 64);
  }

  #[test]
  fn test_xpow_mod_32_crc32c() {
    // CRC32C polynomial with explicit x^32 term
    const POLY_CRC32C: u64 = 0x1_1EDC_6F41;

    // x^0 = 1
    assert_eq!(xpow_mod_32(0, POLY_CRC32C), 1);
    // x^1 = 2
    assert_eq!(xpow_mod_32(1, POLY_CRC32C), 2);
    // x^31 = 2^31 (still below modulus)
    assert_eq!(xpow_mod_32(31, POLY_CRC32C), 1u64 << 31);
    // x^32 should be reduced
    assert!(xpow_mod_32(32, POLY_CRC32C) < (1u64 << 32));
  }

  #[test]
  fn test_xpow_mod_64_crc64() {
    // CRC64/XZ polynomial with explicit x^64 term
    const POLY_CRC64: u128 = (1u128 << 64) | 0x42F0_E1EB_A9EA_3693;

    // x^0 = 1
    assert_eq!(xpow_mod_64(0, POLY_CRC64), 1);
    // x^63 = 2^63 (still below modulus)
    assert_eq!(xpow_mod_64(63, POLY_CRC64), 1u64 << 63);
    // x^64 should be reduced
    assert_ne!(xpow_mod_64(64, POLY_CRC64), 0);
  }
}
