//! CRC32-C (Castagnoli) constants.
//!
//! Polynomial: 0x1EDC6F41 (reflected: 0x82F63B78)
//! Used by: iSCSI, SCTP, Btrfs, ext4, RocksDB, LevelDB

/// CRC32-C polynomial in reflected (bit-reversed) form.
///
/// The normal form is 0x1EDC6F41, but we use reflected form for LSB-first
/// processing which is more efficient on little-endian machines.
pub const POLYNOMIAL: u32 = 0x82F63B78;

/// Slicing-by-8 lookup tables.
///
/// These 8 tables enable processing 8 bytes at a time, achieving ~4x speedup
/// over byte-at-a-time processing on modern CPUs.
///
/// Total size: 8 * 256 * 4 = 8KB
///
/// The tables are 64-byte aligned for optimal cache behavior.
#[cfg(not(feature = "no-tables"))]
pub static TABLES: super::Aligned64<[[u32; 256]; 8]> =
  super::Aligned64(super::tables::generate_slicing_tables_32(POLYNOMIAL));

/// PCLMULQDQ folding constants for CRC32-C.
///
/// These constants are used for hardware-accelerated CRC computation using
/// carryless multiplication instructions (PCLMULQDQ on x86, PMULL on ARM).
///
/// Reference: Intel's "Fast CRC Computation Using PCLMULQDQ Instruction"
#[allow(dead_code)] // Used in Phase 2/3 SIMD implementations
pub mod fold {
  use super::super::gf2;

  /// CRC32-C polynomial with explicit x^32 term for modular arithmetic.
  /// Normal form: 0x1_1EDC_6F41
  const POLY_FULL: u64 = 0x1_1EDC_6F41;

  /// Compute x^n mod P(x) in GF(2) for CRC32-C polynomial.
  const fn xpow_mod(n: u32) -> u64 {
    gf2::xpow_mod_32(n, POLY_FULL)
  }

  /// Reflect (bit-reverse) a 32-bit value.
  const fn reflect32(x: u32) -> u32 {
    gf2::reflect_bits_u64(x as u64, 32) as u32
  }

  /// Compute fold coefficient for a given byte distance.
  ///
  /// Returns (k_hi, k_lo) where:
  /// - k_hi = x^(distance*8 + 32) mod P(x), reflected << 1
  /// - k_lo = x^(distance*8 - 32) mod P(x), reflected << 1
  ///
  /// The shift-left-by-1 is required by the PMULL instruction encoding.
  ///
  /// The +32/-32 offset accounts for how the fold operation works with
  /// the PMULL instruction and 128-bit accumulators.
  pub const fn compute_fold_coeff(distance_bytes: u32) -> (u64, u64) {
    let bits = distance_bytes * 8;

    // Compute x^(bits+32) for k_hi, x^(bits-32) for k_lo
    let k_hi_raw = xpow_mod(bits + 32);
    let k_lo_raw = xpow_mod(bits - 32);

    // Reflect and shift left by 1
    let k_hi = (reflect32(k_hi_raw as u32) as u64) << 1;
    let k_lo = (reflect32(k_lo_raw as u32) as u64) << 1;

    (k_hi, k_lo)
  }

  /// Compute a PMULL folding key for the aarch64 PMULL engines.
  ///
  /// The PMULL kernels in `crate::simd::aarch64::pmull` use a slightly different
  /// key schedule than the classic PCLMULQDQ folding constants:
  ///
  /// - `k0 = (x^(D+31) mod P(x))'`
  /// - `k1 = (x^(D-33) mod P(x))'`
  ///
  /// where `D = distance_bytes * 8` and `'` denotes bit-reflection (LSB-first).
  ///
  /// This matches the constants emitted by the `corsix/fast-crc32` generator
  /// for the Apple M-series fusion kernels.
  pub const fn compute_pmull_key(distance_bytes: u32) -> (u64, u64) {
    let bits = distance_bytes * 8;
    let k0 = reflect32(xpow_mod(bits + 31) as u32) as u64;
    let k1 = reflect32(xpow_mod(bits - 33) as u32) as u64;
    (k0, k1)
  }

  pub const PMULL_KEY_16: (u64, u64) = compute_pmull_key(16);
  pub const PMULL_KEY_32: (u64, u64) = compute_pmull_key(32);
  pub const PMULL_KEY_64: (u64, u64) = compute_pmull_key(64);
  pub const PMULL_KEY_144: (u64, u64) = compute_pmull_key(144);
  pub const PMULL_KEY_192: (u64, u64) = compute_pmull_key(192);
  /// Folding and reduction keys for CRC32-C (reflected, iSCSI).
  ///
  /// This is a compact "key schedule" used by high-throughput PCLMUL/PMULL
  /// implementations (fold + Barrett reduction).
  ///
  /// Indices are 1-based in the original literature/implementations; index 0 is
  /// an unused placeholder to make the mapping obvious.
  ///
  /// Values match Intel whitepaper-derived implementations.
  pub const KEYS_REFLECTED: [u64; 23] = [
    0x0000_0000_0000_0000, // unused placeholder to match 1-based indexing
    0x0000_0001_4cd0_0bd6, // (2^(32* 3) mod P(x))' << 1
    0x0000_0000_f20c_0dfe, // (2^(32* 5) mod P(x))' << 1
    0x0000_0000_0d3b_6092, // (2^(32*31) mod P(x))' << 1
    0x0000_0000_6992_cea2, // (2^(32*33) mod P(x))' << 1
    0x0000_0001_4cd0_0bd6, // (2^(32* 3) mod P(x))' << 1
    0x0000_0000_dd45_aab8, // (2^(32* 2) mod P(x))' << 1
    0x0000_0000_dea7_13f1, // (floor(2^64/P(x)))'
    0x0000_0001_05ec_76f1, // (P(x))'
    0x0000_0001_4237_f5e6, // (2^(32*27) mod P(x))' << 1
    0x0000_0000_2ad9_1c30, // (2^(32*29) mod P(x))' << 1
    0x0000_0001_02f9_b8a2, // (2^(32*23) mod P(x))' << 1
    0x0000_0001_c173_3996, // (2^(32*25) mod P(x))' << 1
    0x0000_0000_39d3_b296, // (2^(32*19) mod P(x))' << 1
    0x0000_0000_083a_6eec, // (2^(32*21) mod P(x))' << 1
    0x0000_0000_9e4a_ddf8, // (2^(32*15) mod P(x))' << 1
    0x0000_0000_740e_ef02, // (2^(32*17) mod P(x))' << 1
    0x0000_0001_d82c_63da, // (2^(32*11) mod P(x))' << 1
    0x0000_0000_1c29_1d04, // (2^(32*13) mod P(x))' << 1
    0x0000_0000_ba4f_c28e, // (2^(32* 7) mod P(x))' << 1
    0x0000_0001_384a_a63a, // (2^(32* 9) mod P(x))' << 1
    0x0000_0000_b9e0_2b86,
    0x0000_0000_dcb1_7aa4,
  ];

  // ============================================================================
  // 9-accumulator (v9) fold constants for Apple M-series optimization
  // 144 bytes per iteration = 9 accumulators × 16 bytes
  // ============================================================================

  /// Fold coefficient for 144-byte (1152-bit) distance.
  ///
  /// Used for 9-way folding main loop (9 accumulators × 16 bytes = 144 bytes/iter).
  /// k_hi = x^(1152+64) mod P(x), k_lo = x^1152 mod P(x), both reflected and <<1.
  pub const COEFF_144: (u64, u64) = compute_fold_coeff(144);

  /// Fold coefficient for 128-byte distance (reduction: x0 into final).
  pub const COEFF_128: (u64, u64) = compute_fold_coeff(128);

  /// Fold coefficient for 112-byte distance (reduction: x1 into final).
  pub const COEFF_112: (u64, u64) = compute_fold_coeff(112);

  /// Fold coefficient for 96-byte distance (reduction: x2 into final).
  pub const COEFF_96: (u64, u64) = compute_fold_coeff(96);

  /// Fold coefficient for 80-byte distance (reduction: x3 into final).
  pub const COEFF_80: (u64, u64) = compute_fold_coeff(80);

  // ============================================================================
  // 4-accumulator fold constants (verified against Intel whitepaper)
  // These use the KEYS_REFLECTED array which has been validated.
  // ============================================================================

  /// Fold coefficient for a 64-byte (512-bit) distance.
  ///
  /// Used for 4-way folding (4 accumulators).
  pub const COEFF_64: (u64, u64) = (KEYS_REFLECTED[16], KEYS_REFLECTED[15]);

  /// Fold coefficient for a 48-byte distance.
  pub const COEFF_48: (u64, u64) = (KEYS_REFLECTED[18], KEYS_REFLECTED[17]);

  /// Fold coefficient for a 32-byte distance.
  pub const COEFF_32: (u64, u64) = (KEYS_REFLECTED[20], KEYS_REFLECTED[19]);

  /// Fold coefficient for a 16-byte distance.
  pub const COEFF_16: (u64, u64) = (KEYS_REFLECTED[2], KEYS_REFLECTED[1]);

  /// Fold-width reduction constants, used to fold 16 bytes down to 4 bytes.
  ///
  /// `high = keys[6]`, `low = keys[5]`.
  pub const FOLD_WIDTH: (u64, u64) = (KEYS_REFLECTED[6], KEYS_REFLECTED[5]);

  /// Barrett reduction constants `(poly, mu)` for reflected CRC32-C.
  pub const BARRETT: (u64, u64) = (KEYS_REFLECTED[8], KEYS_REFLECTED[7]);

  /// Fold by 512 bits (64 bytes) - for AVX-512 VPCLMULQDQ
  /// k1 = x^(512+64) mod P(x), k2 = x^(512) mod P(x)
  pub const K1_K2_512: (u64, u64) = (0x1a0f717c4, 0x0170076fa);

  /// Fold by 256 bits (32 bytes) - for AVX2
  /// k1 = x^(256+64) mod P(x), k2 = x^256 mod P(x)
  pub const K1_K2_256: (u64, u64) = (0x0e417f38a, 0x08f158014);

  /// Fold by 128 bits (16 bytes)
  /// k1 = x^(128+64) mod P(x), k2 = x^128 mod P(x)
  pub const K1_K2_128: (u64, u64) = (0x0740eef02, 0x09e4addf8);

  /// Final fold constants (64-bit to 32-bit reduction)
  /// k5 = x^96 mod P(x), k6 = x^64 mod P(x)
  pub const K5_K6: (u64, u64) = (0x0f20c0dfe, 0x14cd00bd6);

  /// Barrett reduction constants
  /// mu = x^64 / P(x), poly = P(x) with explicit x^32 term
  pub const MU_POLY: (u64, u64) = (0x0dea713f1, 0x105ec76f1);
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_polynomial() {
    // Verify the reflected polynomial is correct
    // 0x1EDC6F41 reflected in 32 bits = 0x82F63B78
    assert_eq!(POLYNOMIAL, 0x82F63B78);
  }

  #[test]
  fn test_fold_constants_bounds() {
    // Verify all fold constants fit in 33 bits (32-bit value << 1)
    // This is required by the PMULL instruction encoding.
    assert!(fold::COEFF_144.0 < (1u64 << 33));
    assert!(fold::COEFF_144.1 < (1u64 << 33));
    assert!(fold::COEFF_128.0 < (1u64 << 33));
    assert!(fold::COEFF_128.1 < (1u64 << 33));
    assert!(fold::COEFF_64.0 < (1u64 << 33));
    assert!(fold::COEFF_64.1 < (1u64 << 33));
    assert!(fold::COEFF_16.0 < (1u64 << 33));
    assert!(fold::COEFF_16.1 < (1u64 << 33));

    // Verify non-zero
    assert_ne!(fold::COEFF_144.0, 0);
    assert_ne!(fold::COEFF_144.1, 0);
  }

  #[test]
  fn test_computed_constants_match_keys() {
    // Verify computed constants match the Intel whitepaper-derived KEYS_REFLECTED values.
    // This validates our constant generation formula.
    // Tuple format: (k_hi, k_lo) = (x^(D+32), x^(D-32))

    // COEFF_16: D=128, k_hi=x^160 (key[2]), k_lo=x^96 (key[1])
    let computed_16 = fold::compute_fold_coeff(16);
    assert_eq!(computed_16, fold::COEFF_16, "COEFF_16 mismatch");

    // COEFF_64: D=512, k_hi=x^544 (key[16]), k_lo=x^480 (key[15])
    let computed_64 = fold::compute_fold_coeff(64);
    assert_eq!(computed_64, fold::COEFF_64, "COEFF_64 mismatch");

    // COEFF_48: D=384, k_hi=x^416 (key[18]), k_lo=x^352 (key[17])
    let computed_48 = fold::compute_fold_coeff(48);
    assert_eq!(computed_48, fold::COEFF_48, "COEFF_48 mismatch");

    // COEFF_32: D=256, k_hi=x^288 (key[20]), k_lo=x^224 (key[19])
    let computed_32 = fold::compute_fold_coeff(32);
    assert_eq!(computed_32, fold::COEFF_32, "COEFF_32 mismatch");
  }

  #[cfg(not(feature = "no-tables"))]
  #[test]
  fn test_table_0_entry() {
    // table[0][0] should be 0 (CRC of 0x00 with initial 0)
    assert_eq!(TABLES.0[0][0], 0);

    // table[0][1] is CRC of processing 0x01 through 8 bit iterations
    // After first bit (LSB=1): 0 ^ POLYNOMIAL = 0x82F63B78
    // Then 7 more iterations with various LSB values
    // Verified against Linux kernel and crc32c reference implementations
    assert_eq!(TABLES.0[0][1], 0xF26B8303);

    // table[0][255] - verified against reference implementations
    assert_eq!(TABLES.0[0][255], 0xAD7D5351);
  }

  #[cfg(not(feature = "no-tables"))]
  #[test]
  fn test_tables_consistency() {
    // Verify tables are related correctly
    // tables[t][i] = (tables[t-1][i] >> 8) ^ tables[0][tables[t-1][i] & 0xFF]
    for t in 1..8 {
      for (i, &prev) in TABLES.0[t - 1].iter().enumerate() {
        let expected = (prev >> 8) ^ TABLES.0[0][(prev & 0xFF) as usize];
        assert_eq!(TABLES.0[t][i], expected);
      }
    }
  }

  #[test]
  fn test_pmull_keys_match_aarch64_kernels() {
    // These values must match the constants used by `simd::aarch64::pmull`.
    assert_eq!(fold::PMULL_KEY_16, (0x0000_0000_f20c_0dfe, 0x0000_0000_493c_7d27));
    assert_eq!(fold::PMULL_KEY_32, (0x0000_0000_3da6_d0cb, 0x0000_0000_ba4f_c28e));
    assert_eq!(fold::PMULL_KEY_64, (0x0000_0000_740e_ef02, 0x0000_0000_9e4a_ddf8));
    assert_eq!(fold::PMULL_KEY_144, (0x0000_0000_7e90_8048, 0x0000_0000_c96c_fdc0));
    assert_eq!(fold::PMULL_KEY_192, (0x0000_0000_a87a_b8a8, 0x0000_0000_ab7a_ff2a));
  }
}
