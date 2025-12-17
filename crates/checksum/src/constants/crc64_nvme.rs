//! CRC64/NVME constants.
//!
//! Polynomial: 0xAD93D23594C93659 (reflected: 0x9A6C9329AC4BC9B5)
//!
//! Used by NVMe storage specification and AWS S3.

/// CRC64/NVME polynomial in reflected (bit-reversed) form.
pub const POLYNOMIAL: u64 = 0x9A6C_9329_AC4B_C9B5;

/// CRC64/NVME polynomial in normal (non-reflected) form.
pub const POLYNOMIAL_NORMAL: u64 = 0xAD93_D235_94C9_3659;

/// Slicing-by-8 lookup tables.
///
/// Total size: 8 * 256 * 8 = 16KB.
/// The tables are 64-byte aligned for optimal cache behavior.
#[cfg(not(feature = "no-tables"))]
pub static TABLES: super::Aligned64<[[u64; 256]; 8]> =
  super::Aligned64(super::tables::generate_slicing_tables_64(POLYNOMIAL));

/// Folding and reduction constants for reflected CRC64/NVME.
#[allow(dead_code)] // Used in SIMD implementations.
pub mod fold {
  use super::super::gf2;

  /// CRC64/NVME polynomial with explicit x^64 term (normal form).
  const POLY_FULL: u128 = (1u128 << 64) | (super::POLYNOMIAL_NORMAL as u128);

  /// Compute x^n mod P(x) in GF(2) for CRC64/NVME polynomial.
  const fn xpow_mod(n: u32) -> u64 {
    gf2::xpow_mod_64(n, POLY_FULL)
  }

  /// Compute a reflected folding key for a given byte distance.
  pub const fn key(distance_bytes: u32) -> u64 {
    let bits = distance_bytes * 8;
    let v = xpow_mod(bits - 1);
    gf2::reflect_bits_u64(v, 64)
  }

  /// Compute fold coefficient for a distance (bytes) as `(high, low)`.
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

  // PMULL keys for aarch64

  /// Compute PMULL folding key for aarch64.
  ///
  /// For reflected CRC64 with PMULL (lo*lo, hi*hi) multiplication pattern,
  /// the key pair must match x86 COEFF semantics (cross-multiplication):
  /// - `k0 = reflect64(x^(D*8+63) mod P(x))` - for low lane
  /// - `k1 = reflect64(x^(D*8-1) mod P(x))` - for high lane
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

  /// Fold-width reduction constants (16 bytes -> 8 bytes) for reflected CRC64/NVME.
  pub const FOLD_WIDTH: (u64, u64) = (0, KEY_16);

  /// Barrett reduction constants `(poly_simd, mu)` for reflected CRC64/NVME.
  pub const BARRETT: (u64, u64) = {
    let poly_ref = gf2::reflect_bits_u128(POLY_FULL, 65);
    let poly_simd = (poly_ref & ((1u128 << 64) - 1)) as u64;

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
    assert_eq!(POLYNOMIAL, 0x9A6C_9329_AC4B_C9B5);
  }

  #[test]
  fn test_barrett_constants() {
    // Barrett reduction constants are computed from the polynomial.
    // poly_simd = low 64 bits of reflect65(x^64 + P(x))
    // mu = reflect64(floor(x^127 / (x^64 + P(x))))
    let (poly_simd, mu) = fold::BARRETT;
    // Sanity check: these should be non-zero 64-bit values
    assert_ne!(poly_simd, 0);
    assert_ne!(mu, 0);
  }
}
