//! CRC64 constants (XZ / ECMA polynomial).
//!
//! Polynomial: 0x42F0E1EBA9EA3693 (reflected: 0xC96C5795D7870F42)
//!
//! This matches the common "XZ" parameters:
//! - init = 0xFFFF_FFFF_FFFF_FFFF
//! - xorout = 0xFFFF_FFFF_FFFF_FFFF
//! - refin/refout = true

/// CRC64 polynomial in reflected (bit-reversed) form.
pub const POLYNOMIAL: u64 = 0xC96C_5795_D787_0F42;

/// CRC64 polynomial in normal (non-reflected) form (no implicit x^64 term).
pub const POLYNOMIAL_NORMAL: u64 = 0x42F0_E1EB_A9EA_3693;

/// Slicing-by-8 lookup tables.
///
/// Total size: 8 * 256 * 8 = 16KB.
/// The tables are 64-byte aligned for optimal cache behavior.
#[cfg(not(feature = "no-tables"))]
pub static TABLES: super::Aligned64<[[u64; 256]; 8]> =
  super::Aligned64(super::tables::generate_slicing_tables_64(POLYNOMIAL));

/// Folding and reduction constants for reflected CRC64 (XZ).
#[allow(dead_code)] // Used in SIMD implementations.
pub mod fold {
  use super::super::gf2;

  /// CRC64 polynomial with explicit x^64 term (normal form).
  ///
  /// Normal form polynomial: 0x42F0E1EBA9EA3693
  const POLY_FULL: u128 = (1u128 << 64) | (super::POLYNOMIAL_NORMAL as u128);

  /// Compute x^n mod P(x) in GF(2) for CRC64 polynomial.
  const fn xpow_mod(n: u32) -> u64 {
    gf2::xpow_mod_64(n, POLY_FULL)
  }

  /// Compute a reflected folding key for a given byte distance.
  pub const fn key(distance_bytes: u32) -> u64 {
    let bits = distance_bytes * 8;
    // Reflected CRC64 uses exponent `bits - 1`.
    let v = xpow_mod(bits - 1);
    gf2::reflect_bits_u64(v, 64)
  }

  /// Compute fold coefficient for a distance (bytes) as `(high, low)`.
  ///
  /// For reflected CRC64 folding, the coefficient pair is:
  /// - `high = key(distance + 8)`
  /// - `low  = key(distance)`
  pub const fn coeff(distance_bytes: u32) -> (u64, u64) {
    (key(distance_bytes + 8), key(distance_bytes))
  }

  pub const KEY_16: u64 = key(16);
  pub const KEY_64: u64 = key(64);

  pub const COEFF_192: (u64, u64) = coeff(192);
  pub const COEFF_176: (u64, u64) = coeff(176);
  pub const COEFF_160: (u64, u64) = coeff(160);
  pub const COEFF_144: (u64, u64) = coeff(144);
  pub const COEFF_128: (u64, u64) = coeff(128);
  pub const COEFF_112: (u64, u64) = coeff(112);
  pub const COEFF_96: (u64, u64) = coeff(96);
  pub const COEFF_80: (u64, u64) = coeff(80);
  pub const COEFF_64: (u64, u64) = coeff(64);
  pub const COEFF_48: (u64, u64) = coeff(48);
  pub const COEFF_32: (u64, u64) = coeff(32);
  pub const COEFF_16: (u64, u64) = coeff(16);

  // ============================================================================
  // PMULL keys for aarch64 (different encoding than x86_64 PCLMULQDQ)
  // ============================================================================

  /// Compute PMULL folding key for aarch64.
  ///
  /// For reflected CRC64 with PMULL (lo*lo, hi*hi) multiplication pattern,
  /// the key pair must match x86 COEFF semantics (cross-multiplication):
  /// - `k0 = reflect64(x^(D*8+63) mod P(x))` - for low lane (matches x86 coeff.high)
  /// - `k1 = reflect64(x^(D*8-1) mod P(x))` - for high lane (matches x86 coeff.low)
  pub const fn compute_pmull_key(distance_bytes: u32) -> (u64, u64) {
    let bits = distance_bytes * 8;
    let k0 = gf2::reflect_bits_u64(xpow_mod(bits + 63), 64);
    let k1 = gf2::reflect_bits_u64(xpow_mod(bits - 1), 64);
    (k0, k1)
  }

  pub const PMULL_KEY_16: (u64, u64) = compute_pmull_key(16);
  pub const PMULL_KEY_32: (u64, u64) = compute_pmull_key(32);
  pub const PMULL_KEY_64: (u64, u64) = compute_pmull_key(64);
  pub const PMULL_KEY_144: (u64, u64) = compute_pmull_key(144);
  pub const PMULL_KEY_192: (u64, u64) = compute_pmull_key(192);

  /// Fold-width reduction constants (16 bytes -> 8 bytes) for reflected CRC64.
  ///
  /// Matches `crc-fast` keys[6], keys[5] for reflected CRC64:
  /// - `high = 0`
  /// - `low  = key(16)`
  pub const FOLD_WIDTH: (u64, u64) = (0, KEY_16);

  /// Barrett reduction constants `(poly_simd, mu)` for reflected CRC64.
  pub const BARRETT: (u64, u64) = {
    // `poly_simd` is the low 64 bits of the 65-bit-reflected polynomial.
    let poly_ref = gf2::reflect_bits_u128(POLY_FULL, 65);
    let poly_simd = (poly_ref & ((1u128 << 64) - 1)) as u64;

    // `mu` is reflect64(floor(x^127 / P(x))).
    let q = gf2::gf2_div_128(1u128 << 127, POLY_FULL) as u64;
    let mu = gf2::reflect_bits_u64(q, 64);

    (poly_simd, mu)
  };
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_polynomial() {
    assert_eq!(POLYNOMIAL, 0xC96C_5795_D787_0F42);
  }

  #[test]
  fn test_barrett_constants() {
    // Verified against `crc-fast` (CRC-64/XZ reflected).
    assert_eq!(fold::BARRETT.0, 0x92D8_AF2B_AF0E_1E85);
    assert_eq!(fold::BARRETT.1, 0x9C3E_466C_1729_63D5);
  }

  #[test]
  fn test_keys() {
    // Verified against `crc-fast` keys for CRC-64/XZ.
    assert_eq!(fold::KEY_16, 0xDABE_95AF_C787_5F40);
    assert_eq!(fold::KEY_64, 0x081F_6054_A784_2DF4);
  }
}
