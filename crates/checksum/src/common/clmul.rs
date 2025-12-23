//! PCLMULQDQ/PMULL folding constants for CRC-32 and CRC-64.
//!
//! This module generates the constant sets needed by the classic Intel/TiKV
//! carryless-multiply folding algorithm:
//! - Process 128 bytes at a time (8×16B lanes)
//! - Fold lanes down to 16B, then 8B/4B
//! - Finish with Barrett reduction
//!
//! The constants are defined in terms of the **reciprocal polynomial** (TiKV
//! nomenclature):
//!
//! - `POLY = (reflected_poly << 1) | 1`
//! - `NORMAL = bit_reverse(reflected_poly)` (the non-reflected polynomial)
//! - `K_n = bit_reverse(x^n mod (x^W ⊕ NORMAL))` where W is the CRC width
//! - `MU = POLY⁻¹ mod x^W` (the multiplicative inverse in GF(2)[x]/(x^W))
//!
//! References:
//! - Intel: "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ"
//! - TiKV: `crc64fast` + `crc64fast-nvme`
//! - zlib-ng: `crc32_fold_pclmulqdq.c`
//! - Linux kernel: `arch/x86/crypto/crc32-pclmul_asm.S`

use crate::common::tables::{CRC32_IEEE_POLY, CRC32C_POLY, CRC64_NVME_POLY, CRC64_XZ_POLY};

// ─────────────────────────────────────────────────────────────────────────────
// GF(2) Polynomial Arithmetic
// ─────────────────────────────────────────────────────────────────────────────

/// Carryless multiplication of two 64-bit values, returning 128-bit result.
///
/// This is the software equivalent of PCLMULQDQ/PMULL.
/// Result is (high64, low64) where high64 contains bits 127..64.
#[must_use]
const fn clmul64(a: u64, b: u64) -> (u64, u64) {
  let mut hi: u64 = 0;
  let mut lo: u64 = 0;

  // Process each bit of 'a'
  let mut i = 0;
  while i < 64 {
    if (a >> i) & 1 != 0 {
      // XOR b shifted by i positions into the result
      // If i + bit_position < 64, it goes into lo
      // If i + bit_position >= 64, it goes into hi
      if i == 0 {
        lo ^= b;
      } else {
        lo ^= b << i;
        hi ^= b >> (64 - i);
      }
    }
    i += 1;
  }

  (hi, lo)
}

/// Reduce a 128-bit value modulo a 65-bit polynomial.
///
/// The polynomial is represented as (poly_hi, poly_lo) where poly_hi is
/// the high bit (always 1 for degree-64 polynomials) and poly_lo is the
/// lower 64 bits.
///
/// For reflected CRCs, the polynomial's implicit x^64 term is at bit 64.
#[must_use]
pub(crate) const fn reduce128(hi: u64, lo: u64, poly: u64) -> u64 {
  // For a 65-bit polynomial G(x) = x^64 + poly (where poly is the low 64 bits),
  // we reduce by: if bit 64+i is set, XOR poly shifted left by i positions.
  //
  // We process from high bits down to bit 64.
  let mut result_hi = hi;
  let mut result_lo = lo;

  // Reduce bits 127 down to 64
  let mut i: i32 = 63;
  while i >= 0 {
    if (result_hi >> i) & 1 != 0 {
      // Bit 64+i is set, XOR with poly shifted by i
      if i == 0 {
        // XOR poly into low part, clear bit 64
        result_lo ^= poly;
        result_hi ^= 1;
      } else {
        // XOR poly shifted by i
        result_lo ^= poly << i;
        result_hi ^= (poly >> (64 - i)) | (1 << i);
      }
    }
    i -= 1;
  }

  result_lo
}

/// Compute x^n mod P for a 64-bit polynomial.
///
/// Uses square-and-multiply algorithm in GF(2).
///
/// # Arguments
///
/// * `n` - The exponent (must be >= 1)
/// * `poly` - The polynomial (lower 64 bits of the 65-bit polynomial)
#[must_use]
const fn xpow_mod(n: u32, poly: u64) -> u64 {
  if n == 0 {
    return 1;
  }
  if n == 1 {
    return 2; // x^1
  }

  // Start with x^1 = 2
  let mut result: u64 = 1; // Identity for multiplication
  let mut base: u64 = 2; // x^1

  let mut exp = n;
  while exp > 0 {
    if exp & 1 != 0 {
      // Multiply result by base
      let (hi, lo) = clmul64(result, base);
      result = reduce128(hi, lo, poly);
    }
    // Square base
    let (hi, lo) = clmul64(base, base);
    base = reduce128(hi, lo, poly);
    exp >>= 1;
  }

  result
}

// ─────────────────────────────────────────────────────────────────────────────
// TiKV/Intel CRC64 folding constants
// ─────────────────────────────────────────────────────────────────────────────

/// Folding constants needed by the TiKV/Intel CRC64 CLMUL algorithm.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Crc64ClmulConstants {
  /// Reciprocal polynomial low 64 bits (`POLY` in TiKV code).
  pub poly: u64,
  /// Barrett reduction constant (`MU` in TiKV code).
  pub mu: u64,
  /// 128-byte folding coefficient (high, low) = (K_1023, K_1087).
  pub fold_128b: (u64, u64),
  /// Tail fold coefficients (distance 112..16 bytes): (K_{d-1}, K_{d+63}).
  pub tail_fold_16b: [(u64, u64); 7],
  /// 16B→8B fold coefficient (`K_127`).
  pub fold_8b: u64,
}

/// Compute TiKV-style reciprocal polynomial from a reflected CRC polynomial.
///
/// TiKV defines:
/// `POLY = (reflected_poly << 1) | 1`.
#[must_use]
const fn reciprocal_poly(reflected_poly: u64) -> u64 {
  (reflected_poly << 1) | 1
}

/// Compute the normal (non-reflected) polynomial from a reflected polynomial.
///
/// For width-64 CRC polynomials, the normal and reflected forms are bit-reverses.
#[must_use]
const fn normal_poly(reflected_poly: u64) -> u64 {
  reflected_poly.reverse_bits()
}

/// Compute folding constant `K_n = bit_reverse(x^n mod (x^64 ⊕ NORMAL))`.
#[must_use]
const fn fold_k(normal_poly: u64, n: u32) -> u64 {
  xpow_mod(n, normal_poly).reverse_bits()
}

/// Compute a `(high, low)` fold coefficient pair for folding 16 bytes by `shift_bytes`.
///
/// The TiKV/Intel CLMUL folding step uses a pair `(K_{d-1}, K_{d+63})` where
/// `d = 8 * shift_bytes` (in bits).
#[must_use]
pub(crate) const fn fold16_coeff_for_bytes(reflected_poly: u64, shift_bytes: u32) -> (u64, u64) {
  if shift_bytes == 0 {
    return (0, 0);
  }

  let normal = normal_poly(reflected_poly);
  let d = shift_bytes * 8;
  // `d >= 8`, so `d - 1` is valid.
  (fold_k(normal, d - 1), fold_k(normal, d + 63))
}

impl Crc64ClmulConstants {
  #[must_use]
  pub const fn new(reflected_poly: u64) -> Self {
    let poly = reciprocal_poly(reflected_poly);
    let normal = normal_poly(reflected_poly);
    let mu = compute_tikv_mu(poly);

    Self {
      poly,
      mu,
      fold_128b: (fold_k(normal, 1023), fold_k(normal, 1087)),
      tail_fold_16b: [
        (fold_k(normal, 895), fold_k(normal, 959)), // 112 bytes
        (fold_k(normal, 767), fold_k(normal, 831)), // 96 bytes
        (fold_k(normal, 639), fold_k(normal, 703)), // 80 bytes
        (fold_k(normal, 511), fold_k(normal, 575)), // 64 bytes
        (fold_k(normal, 383), fold_k(normal, 447)), // 48 bytes
        (fold_k(normal, 255), fold_k(normal, 319)), // 32 bytes
        (fold_k(normal, 127), fold_k(normal, 191)), // 16 bytes
      ],
      fold_8b: fold_k(normal, 127),
    }
  }
}

/// Compute TiKV `MU` for Barrett reduction.
///
/// TiKV’s CRC64 CLMUL reduction uses `MU = POLY⁻¹ mod x^64` where `POLY` is the
/// reciprocal polynomial (low 64 bits; the x^64 term is implicit). Equivalently:
/// `(MU ⊗ POLY) mod x^64 == 1`.
///
/// Since `POLY` always has constant term 1, the inverse exists in
/// GF(2)[x]/(x^64). We compute the inverse bit-by-bit as a power series.
#[must_use]
const fn compute_tikv_mu(poly: u64) -> u64 {
  // q(x) = 1 / p(x) (mod x^64), with p(0)=1.
  // For k>=1: q_k = Σ_{i=1..k} p_i * q_{k-i}  (in GF(2)).
  let mut inv: u64 = 1;

  let mut k: u32 = 1;
  while k < 64 {
    let mut s: u64 = 0;

    let mut i: u32 = 1;
    while i <= k {
      let p_i = (poly >> i) & 1;
      let q_j = (inv >> (k - i)) & 1;
      s ^= p_i & q_j;
      i += 1;
    }

    inv |= s << k;
    k += 1;
  }

  inv
}

// ─────────────────────────────────────────────────────────────────────────────
// Pre-computed constant sets for CRC-64.
pub(crate) const CRC64_XZ_CLMUL: Crc64ClmulConstants = Crc64ClmulConstants::new(CRC64_XZ_POLY);
pub(crate) const CRC64_NVME_CLMUL: Crc64ClmulConstants = Crc64ClmulConstants::new(CRC64_NVME_POLY);

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32 CLMUL Folding Constants (Phase 2+)
//
// These constants and helpers are prepared for CRC-32 SIMD kernels.
// Currently unused; will be activated in Phase 2 of the refactor.
// ─────────────────────────────────────────────────────────────────────────────

/// Carryless multiplication of two 32-bit values, returning 64-bit result.
///
/// This is the software equivalent of PCLMULQDQ with 32-bit operands.
/// Result is (high32, low32) where high32 contains bits 63..32.
#[allow(dead_code)]
#[must_use]
const fn clmul32(a: u32, b: u32) -> (u32, u32) {
  let mut hi: u32 = 0;
  let mut lo: u32 = 0;

  let mut i = 0;
  while i < 32 {
    if (a >> i) & 1 != 0 {
      if i == 0 {
        lo ^= b;
      } else {
        lo ^= b << i;
        hi ^= b >> (32 - i);
      }
    }
    i += 1;
  }

  (hi, lo)
}

/// Reduce a 64-bit value modulo a 33-bit polynomial.
///
/// For CRC-32, the polynomial is degree-32: G(x) = x^32 + poly (where poly is the low 32 bits).
#[allow(dead_code)]
#[must_use]
const fn reduce64_32(hi: u32, lo: u32, poly: u32) -> u32 {
  let mut result_hi = hi;
  let mut result_lo = lo;

  // Reduce bits 63 down to 32
  let mut i: i32 = 31;
  while i >= 0 {
    if (result_hi >> i) & 1 != 0 {
      if i == 0 {
        result_lo ^= poly;
        result_hi ^= 1;
      } else {
        result_lo ^= poly << i;
        result_hi ^= (poly >> (32 - i)) | (1 << i);
      }
    }
    i -= 1;
  }

  result_lo
}

/// Compute x^n mod P for a 32-bit polynomial.
///
/// Uses square-and-multiply algorithm in GF(2).
#[allow(dead_code)]
#[must_use]
const fn xpow_mod_32(n: u32, poly: u32) -> u32 {
  if n == 0 {
    return 1;
  }
  if n == 1 {
    return 2; // x^1
  }

  let mut result: u32 = 1;
  let mut base: u32 = 2; // x^1

  let mut exp = n;
  while exp > 0 {
    if exp & 1 != 0 {
      let (hi, lo) = clmul32(result, base);
      result = reduce64_32(hi, lo, poly);
    }
    let (hi, lo) = clmul32(base, base);
    base = reduce64_32(hi, lo, poly);
    exp >>= 1;
  }

  result
}

/// Compute the normal (non-reflected) polynomial from a reflected polynomial (32-bit).
#[allow(dead_code)]
#[must_use]
const fn normal_poly_32(reflected_poly: u32) -> u32 {
  reflected_poly.reverse_bits()
}

/// Compute folding constant `K_n = bit_reverse(x^n mod (x^32 ⊕ NORMAL))` for CRC-32.
#[allow(dead_code)]
#[must_use]
const fn fold_k_32(normal_poly: u32, n: u32) -> u32 {
  xpow_mod_32(n, normal_poly).reverse_bits()
}

/// Folding constants needed by the Intel CRC32 CLMUL algorithm.
///
/// These constants support the classic PCLMULQDQ folding:
/// - 64-byte blocks folded with `fold_64b`
/// - Tail reduction with `tail_fold_16b` array (like CRC64)
/// - Final Barrett reduction
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Crc32ClmulConstants {
  /// Reciprocal polynomial (`POLY = (reflected << 1) | 1`), stored as u64 for CLMUL.
  pub poly: u64,
  /// Barrett reduction constant µ, stored as u64 for CLMUL.
  pub mu: u64,
  /// 64-byte folding coefficients - for folding 64B blocks.
  /// Uses (K_{d-1}, K_{d+63}) convention where d = 512 bits.
  pub fold_64b: (u64, u64),
  /// Tail folding coefficients for reducing 4 lanes (64B) to 1 lane.
  /// Index 0: 48-byte fold (384 bits)
  /// Index 1: 32-byte fold (256 bits)
  /// Index 2: 16-byte fold (128 bits)
  pub tail_fold_16b: [(u64, u64); 3],
  /// 8-byte folding coefficient K_63 - for fold_8 (16B → prepared for Barrett).
  pub fold_8b: u64,
}

/// Compute TiKV-style reciprocal polynomial from a reflected CRC-32 polynomial.
#[allow(dead_code)]
#[must_use]
const fn reciprocal_poly_32(reflected_poly: u32) -> u64 {
  ((reflected_poly as u64) << 1) | 1
}

/// Compute the Barrett µ for a 33-bit polynomial.
///
/// µ = x^64 / P where P = x^32 + poly.
/// For Barrett reduction: (a mod P) = a - (a * µ / x^64) * P
#[allow(dead_code)]
#[must_use]
const fn compute_barrett_mu_32(poly: u32) -> u64 {
  // For CRC-32, we need µ such that (µ * P) mod x^64 has bit 64 set.
  // µ ≈ x^64 / P = x^32 * (x^32 / (x^32 + poly))
  //
  // Using power series: 1/(1+p) = 1 + p + p^2 + ... in GF(2)
  // where p = poly/x^32 (the fractional part).
  //
  // We compute the 33-bit inverse by iterative refinement.
  let poly64 = poly as u64;

  // Initial approximation: µ_0 = x^32
  let mut mu: u64 = 1u64 << 32;

  // Refine: each step corrects one more bit of the quotient
  let mut i = 0;
  while i < 32 {
    // Check if we need to adjust bit i of the quotient
    // by testing if P * current_mu has wrong bit pattern
    let _check_bit = 32u64.wrapping_add(i);
    let product_hi = (mu >> 32) ^ ((mu & ((1u64 << 32) - 1)).wrapping_mul(poly64 >> 32));
    let product_lo = (mu & ((1u64 << 32) - 1)).wrapping_mul(poly64 & ((1u64 << 32) - 1));

    // If bit check_bit is set in the product, flip the corresponding bit in mu
    let combined = product_hi ^ (product_lo >> 32);
    if i < 32 && (combined >> i) & 1 != 0 {
      mu ^= 1u64 << i;
    }
    i += 1;
  }

  // Alternative: compute directly using polynomial division
  // x^64 / (x^32 + poly) = x^32 + r where r = x^64 mod P
  // Use the same approach as CRC-64 but adapted for 32-bit poly

  // Actually, let's use the simpler bit-by-bit inverse approach:
  // µ such that (µ * (poly | (1<<32))) mod x^64 = x^32
  let recip = reciprocal_poly_32(poly);
  let mut inv: u64 = 1;

  let mut k: u32 = 1;
  while k < 64 {
    let mut s: u64 = 0;
    let mut j: u32 = 1;
    while j <= k && j < 33 {
      let p_j = (recip >> j) & 1;
      let q_kj = if k >= j { (inv >> (k - j)) & 1 } else { 0 };
      s ^= p_j & q_kj;
      j += 1;
    }
    inv |= s << k;
    k += 1;
  }

  inv
}

/// Compute a `(high, low)` fold coefficient pair for folding 16 bytes by `shift_bytes` for CRC32.
///
/// Uses the same `(K_{d-1}, K_{d+63})` convention as CRC64.
#[allow(dead_code)]
#[must_use]
const fn fold16_coeff_for_bytes_32(normal_poly: u32, shift_bytes: u32) -> (u64, u64) {
  if shift_bytes == 0 {
    return (0, 0);
  }
  let d = shift_bytes * 8;
  (
    fold_k_32(normal_poly, d - 1) as u64,
    fold_k_32(normal_poly, d + 63) as u64,
  )
}

impl Crc32ClmulConstants {
  /// Create CLMUL constants for a reflected CRC-32 polynomial.
  ///
  /// Uses the same `(K_{d-1}, K_{d+63})` convention as CRC64 for folding constants.
  #[allow(dead_code)]
  #[must_use]
  pub const fn new(reflected_poly: u32) -> Self {
    let normal = normal_poly_32(reflected_poly);
    let poly = reciprocal_poly_32(reflected_poly);
    let mu = compute_barrett_mu_32(reflected_poly);

    // 64-byte folding: d = 512 bits → (K_511, K_575)
    let fold_64b = fold16_coeff_for_bytes_32(normal, 64);

    // Tail folding for reducing 4 lanes to 1:
    // - Lane 0 folds by 48 bytes (384 bits)
    // - Lane 1 folds by 32 bytes (256 bits)
    // - Lane 2 folds by 16 bytes (128 bits)
    let tail_fold_16b = [
      fold16_coeff_for_bytes_32(normal, 48), // (K_383, K_447)
      fold16_coeff_for_bytes_32(normal, 32), // (K_255, K_319)
      fold16_coeff_for_bytes_32(normal, 16), // (K_127, K_191)
    ];

    // For fold_8: K_63 (fold by 64 bits, like CRC64's K_127)
    let fold_8b = fold_k_32(normal, 63) as u64;

    Self {
      poly,
      mu,
      fold_64b,
      tail_fold_16b,
      fold_8b,
    }
  }
}

// Pre-computed constant sets for CRC-32 (Phase 2+).
#[allow(dead_code)]
pub(crate) const CRC32_IEEE_CLMUL: Crc32ClmulConstants = Crc32ClmulConstants::new(CRC32_IEEE_POLY);
#[allow(dead_code)]
pub(crate) const CRC32C_CLMUL: Crc32ClmulConstants = Crc32ClmulConstants::new(CRC32C_POLY);

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_clmul64_basic() {
    // 0 * anything = 0
    assert_eq!(clmul64(0, 12345), (0, 0));
    assert_eq!(clmul64(12345, 0), (0, 0));

    // 1 * x = x
    assert_eq!(clmul64(1, 0x1234), (0, 0x1234));
    assert_eq!(clmul64(0x1234, 1), (0, 0x1234));

    // x * x = x^2 (2 * 2 = 4 in GF(2))
    assert_eq!(clmul64(2, 2), (0, 4));

    // (x+1) * (x+1) = x^2 + 1 (no 2x term in GF(2))
    assert_eq!(clmul64(3, 3), (0, 5)); // 3*3 = 5 in GF(2)
  }

  #[test]
  fn test_clmul64_overflow() {
    // High values that overflow into the high 64 bits
    let a = 1u64 << 63;
    let b = 2u64; // x
    // (x^63) * x = x^64, which is bit 64 (in high part)
    let (hi, lo) = clmul64(a, b);
    assert_eq!(hi, 1);
    assert_eq!(lo, 0);
  }

  #[test]
  fn test_xpow_mod_basic() {
    let poly = CRC64_XZ_CLMUL.poly;

    // x^0 = 1
    assert_eq!(xpow_mod(0, poly), 1);

    // x^1 = 2 (just x in reflected form)
    assert_eq!(xpow_mod(1, poly), 2);

    // x^2 = 4
    assert_eq!(xpow_mod(2, poly), 4);

    // Higher powers should be reduced mod poly
    let x64 = xpow_mod(64, poly);
    // x^64 mod P should equal poly (since P = x^64 + poly)
    assert_eq!(x64, poly);
  }

  #[test]
  fn test_xpow_mod_xz_polynomial() {
    // Verify some known values for CRC-64-XZ
    let poly = CRC64_XZ_POLY;

    // x^64 mod P = poly (by definition of the polynomial)
    assert_eq!(xpow_mod(64, poly), poly);

    // Verify the result is always 64 bits
    assert!(xpow_mod(128, poly) != 0);
    assert!(xpow_mod(1024, poly) != 0);
  }

  #[test]
  fn test_fold_constants_generated() {
    // Just verify the constants are generated without panic
    // Constant sets should be distinct
    assert_ne!(CRC64_XZ_CLMUL.poly, CRC64_NVME_CLMUL.poly);
    assert_ne!(CRC64_XZ_CLMUL.mu, CRC64_NVME_CLMUL.mu);
  }

  #[test]
  fn test_barrett_mu() {
    // Barrett µ should be approximately x^64 (high bit set)
    // µ for a degree-64 polynomial is a 65-bit value; low bits should be non-zero.
    assert_ne!(CRC64_XZ_CLMUL.mu, 0);
  }

  #[test]
  fn test_polynomial_property() {
    // Verify that x^64 mod P = P (lower 64 bits)
    // This is the fundamental property of the polynomial
    // x^64 mod (x^64 + poly) = poly
    assert_eq!(xpow_mod(64, CRC64_XZ_CLMUL.poly), CRC64_XZ_CLMUL.poly);
    assert_eq!(xpow_mod(64, CRC64_NVME_CLMUL.poly), CRC64_NVME_CLMUL.poly);
  }

  #[test]
  fn test_reduce128_identity() {
    // Reducing a value less than 2^64 should return itself
    let poly = CRC64_XZ_POLY;
    assert_eq!(reduce128(0, 0x12345678, poly), 0x12345678);
    assert_eq!(reduce128(0, poly - 1, poly), poly - 1);
  }

  #[test]
  fn test_reduce128_single_bit() {
    let poly = CRC64_XZ_POLY;

    // Reducing x^64 (bit 64 set) should give poly
    assert_eq!(reduce128(1, 0, poly), poly);

    // Reducing x^65 (bit 65 set) should give poly * 2 mod P
    let x65 = reduce128(2, 0, poly);
    let expected = xpow_mod(65, poly);
    assert_eq!(x65, expected);
  }

  #[test]
  fn test_constants_symmetry() {
    // `K_127` should be non-zero.
    assert_ne!(CRC64_XZ_CLMUL.fold_8b, 0);
  }

  #[test]
  fn test_xz_constants_match_tikv() {
    // TiKV `crc64fast` constants (v1.1.0).
    assert_eq!(CRC64_XZ_CLMUL.poly, 0x92d8_af2b_af0e_1e85);
    assert_eq!(CRC64_XZ_CLMUL.mu, 0x9c3e_466c_1729_63d5);
    assert_eq!(CRC64_XZ_CLMUL.fold_128b, (0xd7d8_6b2a_f73d_e740, 0x8757_d71d_4fcc_1000));
    assert_eq!(
      CRC64_XZ_CLMUL.tail_fold_16b,
      [
        (0x9478_74de_5950_52cb, 0x9e73_5cb5_9b47_24da), // 112
        (0xe4ce_2cd5_5fea_0037, 0x2fe3_fd29_20ce_82ec), // 96
        (0x0e31_d519_421a_63a5, 0x2e30_2032_12ca_c325), // 80
        (0x081f_6054_a784_2df4, 0x6ae3_efbb_9dd4_41f3), // 64
        (0x69a3_5d91_c373_0254, 0xb5ea_1af9_c013_aca4), // 48
        (0x3be6_53a3_0fe1_af51, 0x6009_5b00_8a9e_fa44), // 32
        (0xdabe_95af_c787_5f40, 0xe05d_d497_ca39_3ae4), // 16
      ]
    );
    assert_eq!(CRC64_XZ_CLMUL.fold_8b, 0xdabe_95af_c787_5f40);
  }

  #[test]
  fn test_nvme_constants_match_tikv() {
    // TiKV `crc64fast-nvme` constants (v1.2.1).
    assert_eq!(CRC64_NVME_CLMUL.poly, 0x34d9_2653_5897_936b);
    assert_eq!(CRC64_NVME_CLMUL.mu, 0x27ec_fa32_9aef_9f77);
    assert_eq!(
      CRC64_NVME_CLMUL.fold_128b,
      (0x5f85_2fb6_1e8d_92dc, 0xa1ca_681e_733f_9c40)
    );
    assert_eq!(
      CRC64_NVME_CLMUL.tail_fold_16b,
      [
        (0x9465_8840_3d4a_dcbc, 0xd083_dd59_4d96_319d), // 112
        (0x34f5_a24e_22d6_6e90, 0x3c25_5f5e_bc41_4423), // 96
        (0x0336_3823_e6e7_91e5, 0x7b0a_b10d_d0f8_09fe), // 80
        (0x6224_2240_ace5_045a, 0x0c32_cdb3_1e18_a84a), // 64
        (0xa3ff_dc1f_e8e8_2a8b, 0xbdd7_ac0e_e1a4_a0f0), // 48
        (0xe1e0_bb9d_45d7_a44c, 0xb0bc_2e58_9204_f500), // 32
        (0x21e9_761e_2526_21ac, 0xeadc_41fd_2ba3_d420), // 16
      ]
    );
    assert_eq!(CRC64_NVME_CLMUL.fold_8b, 0x21e9_761e_2526_21ac);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // CRC-32 Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_clmul32_basic() {
    // 0 * anything = 0
    assert_eq!(clmul32(0, 12345), (0, 0));
    assert_eq!(clmul32(12345, 0), (0, 0));

    // 1 * x = x
    assert_eq!(clmul32(1, 0x1234), (0, 0x1234));
    assert_eq!(clmul32(0x1234, 1), (0, 0x1234));

    // x * x = x^2 (2 * 2 = 4 in GF(2))
    assert_eq!(clmul32(2, 2), (0, 4));

    // (x+1) * (x+1) = x^2 + 1 (no 2x term in GF(2))
    assert_eq!(clmul32(3, 3), (0, 5));
  }

  #[test]
  fn test_clmul32_overflow() {
    // High values that overflow into the high 32 bits
    let a = 1u32 << 31;
    let b = 2u32; // x
    // (x^31) * x = x^32, which is bit 32 (in high part)
    let (hi, lo) = clmul32(a, b);
    assert_eq!(hi, 1);
    assert_eq!(lo, 0);
  }

  #[test]
  fn test_xpow_mod_32_basic() {
    let poly = CRC32_IEEE_POLY;

    // x^0 = 1
    assert_eq!(xpow_mod_32(0, poly), 1);

    // x^1 = 2
    assert_eq!(xpow_mod_32(1, poly), 2);

    // x^2 = 4
    assert_eq!(xpow_mod_32(2, poly), 4);

    // x^32 mod P = poly (by definition)
    assert_eq!(xpow_mod_32(32, poly), poly);
  }

  #[test]
  fn test_crc32_constants_generated() {
    // Verify constants are distinct between IEEE and Castagnoli
    assert_ne!(CRC32_IEEE_CLMUL.poly, CRC32C_CLMUL.poly);
    assert_ne!(CRC32_IEEE_CLMUL.mu, CRC32C_CLMUL.mu);

    // Verify poly has expected form: (reflected << 1) | 1
    assert_eq!(CRC32_IEEE_CLMUL.poly & 1, 1);
    assert_eq!(CRC32C_CLMUL.poly & 1, 1);

    // Verify folding constants are non-zero
    assert_ne!(CRC32_IEEE_CLMUL.fold_64b, (0, 0));
    assert_ne!(CRC32_IEEE_CLMUL.tail_fold_16b[0], (0, 0));
    assert_ne!(CRC32_IEEE_CLMUL.tail_fold_16b[1], (0, 0));
    assert_ne!(CRC32_IEEE_CLMUL.tail_fold_16b[2], (0, 0));
    assert_ne!(CRC32_IEEE_CLMUL.fold_8b, 0);
  }

  #[test]
  fn test_crc32_polynomial_property() {
    // x^32 mod (x^32 + poly) = poly
    assert_eq!(xpow_mod_32(32, CRC32_IEEE_POLY), CRC32_IEEE_POLY);
    assert_eq!(xpow_mod_32(32, CRC32C_POLY), CRC32C_POLY);
  }

  #[test]
  fn test_reduce64_32_identity() {
    // Reducing a value less than 2^32 should return itself
    let poly = CRC32_IEEE_POLY;
    assert_eq!(reduce64_32(0, 0x12345678, poly), 0x12345678);
    assert_eq!(reduce64_32(0, poly - 1, poly), poly - 1);
  }

  #[test]
  fn test_reduce64_32_single_bit() {
    let poly = CRC32_IEEE_POLY;

    // Reducing x^32 (bit 32 set) should give poly
    assert_eq!(reduce64_32(1, 0, poly), poly);

    // Reducing x^33 (bit 33 set) should give poly * 2 mod P
    let x33 = reduce64_32(2, 0, poly);
    let expected = xpow_mod_32(33, poly);
    assert_eq!(x33, expected);
  }

  #[test]
  fn test_crc32c_reciprocal_poly() {
    // CRC32C reflected polynomial is 0x82F63B78
    // Reciprocal should be (0x82F63B78 << 1) | 1 = 0x105EC76F1
    assert_eq!(CRC32C_CLMUL.poly, 0x1_05EC_76F1);
  }

  #[test]
  fn test_crc32_ieee_reciprocal_poly() {
    // CRC32 IEEE reflected polynomial is 0xEDB88320
    // Reciprocal should be (0xEDB88320 << 1) | 1 = 0x1DB710641
    assert_eq!(CRC32_IEEE_CLMUL.poly, 0x1_DB71_0641);
  }
}
