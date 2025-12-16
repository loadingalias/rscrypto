//! CRC32 (ISO-HDLC) constants.
//!
//! Polynomial: 0x04C11DB7 (reflected: 0xEDB88320)
//! Used by: Ethernet, gzip, PNG, zip, SATA, zlib

/// CRC32 (ISO-HDLC) polynomial in reflected (bit-reversed) form.
///
/// The normal form is 0x04C11DB7, but we use reflected form for LSB-first
/// processing on little-endian machines.
pub const POLYNOMIAL: u32 = 0xEDB8_8320;

/// Slicing-by-8 lookup tables.
///
/// Total size: 8 * 256 * 4 = 8KB
/// The tables are 64-byte aligned for optimal cache behavior.
#[cfg(not(feature = "no-tables"))]
pub static TABLES: super::Aligned64<[[u32; 256]; 8]> =
  super::Aligned64(super::tables::generate_slicing_tables_32(POLYNOMIAL));

/// Folding and reduction constants for CRC32 (reflected, ISO-HDLC).
#[allow(dead_code)] // Used in SIMD implementations.
pub mod fold {
  use super::super::gf2;

  /// CRC32 polynomial with explicit x^32 term (normal form).
  ///
  /// Normal form polynomial: 0x04C11DB7
  const POLY_FULL: u64 = 0x1_04C1_1DB7;

  /// Compute x^n mod P(x) in GF(2) for CRC32 polynomial.
  const fn xpow_mod(n: u32) -> u64 {
    gf2::xpow_mod_32(n, POLY_FULL)
  }

  /// Reflect (bit-reverse) a 32-bit value.
  const fn reflect32(x: u32) -> u32 {
    gf2::reflect_bits_u64(x as u64, 32) as u32
  }

  /// Reflect (bit-reverse) a 33-bit value.
  const fn reflect33(x: u64) -> u64 {
    gf2::reflect_bits_u64(x, 33)
  }

  /// Compute fold coefficient for a given byte distance.
  pub const fn compute_fold_coeff(distance_bytes: u32) -> (u64, u64) {
    let bits = distance_bytes * 8;

    let k_hi_raw = xpow_mod(bits + 32);
    let k_lo_raw = xpow_mod(bits - 32);

    let k_hi = (reflect32(k_hi_raw as u32) as u64) << 1;
    let k_lo = (reflect32(k_lo_raw as u32) as u64) << 1;

    (k_hi, k_lo)
  }

  /// Compute a PMULL folding key for the aarch64 PMULL engines.
  ///
  /// The aarch64 PMULL kernels use the same key schedule as `fast-crc32`:
  ///
  /// - `k0 = (x^(D+31) mod P(x))'`
  /// - `k1 = (x^(D-33) mod P(x))'`
  ///
  /// where `D = distance_bytes * 8` and `'` denotes bit-reflection (LSB-first).
  pub const fn compute_pmull_key(distance_bytes: u32) -> (u64, u64) {
    let bits = distance_bytes * 8;
    let k0 = reflect32(xpow_mod(bits + 31) as u32) as u64;
    let k1 = reflect32(xpow_mod(bits - 33) as u32) as u64;
    (k0, k1)
  }

  pub const PMULL_KEY_16: (u64, u64) = compute_pmull_key(16);
  pub const PMULL_KEY_32: (u64, u64) = compute_pmull_key(32);
  pub const PMULL_KEY_48: (u64, u64) = compute_pmull_key(48);
  pub const PMULL_KEY_64: (u64, u64) = compute_pmull_key(64);
  pub const PMULL_KEY_144: (u64, u64) = compute_pmull_key(144);
  pub const PMULL_KEY_192: (u64, u64) = compute_pmull_key(192);

  /// Fold coefficient for a 64-byte (512-bit) distance.
  pub const COEFF_64: (u64, u64) = compute_fold_coeff(64);

  /// Fold coefficient for a 48-byte distance.
  pub const COEFF_48: (u64, u64) = compute_fold_coeff(48);

  /// Fold coefficient for a 32-byte distance.
  pub const COEFF_32: (u64, u64) = compute_fold_coeff(32);

  /// Fold coefficient for a 16-byte distance.
  pub const COEFF_16: (u64, u64) = compute_fold_coeff(16);

  /// Fold-width reduction constants, used to fold 16 bytes down to 4 bytes.
  ///
  /// `high = x^64 mod P(x)` (reflected << 1)
  /// `low  = x^96 mod P(x)` (reflected << 1)
  pub const FOLD_WIDTH: (u64, u64) = {
    let high = (reflect32(xpow_mod(64) as u32) as u64) << 1;
    let low = (reflect32(xpow_mod(96) as u32) as u64) << 1;
    (high, low)
  };

  /// Barrett reduction constants `(poly, mu)` for reflected CRC32.
  pub const BARRETT: (u64, u64) = {
    let poly = reflect33(POLY_FULL);
    let q = gf2::gf2_div_128(1u128 << 64, POLY_FULL as u128) as u64;
    let mu = reflect33(q);
    (poly, mu)
  };
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_polynomial() {
    assert_eq!(POLYNOMIAL, 0xEDB8_8320);
  }

  #[test]
  fn test_barrett_constants() {
    // These are the standard reflected Barrett constants for CRC32 (IEEE).
    assert_eq!(fold::BARRETT.0, 0x0001_DB71_0641);
    assert_eq!(fold::BARRETT.1, 0x0001_F701_1641);
  }

  #[test]
  fn test_fold_constants_bounds() {
    // Fold coefficients are 32-bit values shifted left by 1, so they must fit in 33 bits.
    for (hi, lo) in [
      fold::COEFF_16,
      fold::COEFF_32,
      fold::COEFF_48,
      fold::COEFF_64,
      fold::FOLD_WIDTH,
    ] {
      assert!(hi < (1u64 << 33));
      assert!(lo < (1u64 << 33));
      assert_ne!(hi, 0);
      assert_ne!(lo, 0);
    }
  }

  #[cfg(not(feature = "no-tables"))]
  #[test]
  fn test_tables_consistency() {
    for t in 1..8 {
      for (i, &prev) in TABLES.0[t - 1].iter().enumerate() {
        let expected = (prev >> 8) ^ TABLES.0[0][(prev & 0xFF) as usize];
        assert_eq!(TABLES.0[t][i], expected);
      }
    }
  }

  #[test]
  fn test_pmull_keys() {
    assert_eq!(fold::PMULL_KEY_16, (0x0000_0000_ae68_9191, 0x0000_0000_ccaa_009e));
    assert_eq!(fold::PMULL_KEY_32, (0x0000_0000_f1da_05aa, 0x0000_0000_8125_6527));
    assert_eq!(fold::PMULL_KEY_64, (0x0000_0000_8f35_2d95, 0x0000_0000_1d95_13d7));
    assert_eq!(fold::PMULL_KEY_144, (0x0000_0000_26b7_0c3d, 0x0000_0000_3f41_287a));
    assert_eq!(fold::PMULL_KEY_192, (0x0000_0000_596c_8d81, 0x0000_0000_f5e4_8c85));
  }
}
