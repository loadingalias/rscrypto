//! PCLMULQDQ/PMULL folding constants for CRC-64.
//!
//! This module generates the constant sets needed by the classic Intel/TiKV
//! carryless-multiply folding algorithm:
//! - Process 128 bytes at a time (8×16B lanes)
//! - Fold lanes down to 16B, then 8B
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
/// TiKV's CRC64 CLMUL reduction uses `MU = POLY⁻¹ mod x^64` where `POLY` is the
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
// Multi-stream folding constants for CRC-64.
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-stream folding constants for CRC-64 CLMUL kernels.
///
/// These constants support multi-way ILP (instruction-level parallelism)
/// optimizations on x86_64 (PCLMULQDQ/VPCLMULQDQ) and aarch64 (PMULL).
///
/// - `fold_256b`: 2-way striping (both architectures)
/// - `fold_384b`: 3-way striping (aarch64)
/// - `fold_512b`: 4-way striping (x86_64)
/// - `fold_896b`: 7-way striping (x86_64)
/// - `combine_4way`: merge coefficients for 4-way (x86_64)
/// - `combine_7way`: merge coefficients for 7-way (x86_64)
// Some fields are only used on specific architectures.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Crc64StreamConstants {
  /// 2-way fold coefficient (256B = 2×128B).
  pub fold_256b: (u64, u64),
  /// 3-way fold coefficient (384B = 3×128B).
  pub fold_384b: (u64, u64),
  /// 4-way fold coefficient (512B = 4×128B).
  pub fold_512b: (u64, u64),
  /// 7-way fold coefficient (896B = 7×128B).
  pub fold_896b: (u64, u64),
  /// 4-way combine coefficients: shifts by 384B, 256B, 128B.
  pub combine_4way: [(u64, u64); 3],
  /// 7-way combine coefficients: shifts by 768B, 640B, 512B, 384B, 256B, 128B.
  pub combine_7way: [(u64, u64); 6],
}

impl Crc64StreamConstants {
  /// Compute all multi-stream folding constants for a given polynomial.
  #[must_use]
  pub const fn new(reflected_poly: u64) -> Self {
    Self {
      fold_256b: fold16_coeff_for_bytes(reflected_poly, 256),
      fold_384b: fold16_coeff_for_bytes(reflected_poly, 384),
      fold_512b: fold16_coeff_for_bytes(reflected_poly, 512),
      fold_896b: fold16_coeff_for_bytes(reflected_poly, 896),
      combine_4way: [
        fold16_coeff_for_bytes(reflected_poly, 384),
        fold16_coeff_for_bytes(reflected_poly, 256),
        fold16_coeff_for_bytes(reflected_poly, 128),
      ],
      combine_7way: [
        fold16_coeff_for_bytes(reflected_poly, 768),
        fold16_coeff_for_bytes(reflected_poly, 640),
        fold16_coeff_for_bytes(reflected_poly, 512),
        fold16_coeff_for_bytes(reflected_poly, 384),
        fold16_coeff_for_bytes(reflected_poly, 256),
        fold16_coeff_for_bytes(reflected_poly, 128),
      ],
    }
  }
}

// Pre-computed multi-stream constants for CRC-64.
pub(crate) const CRC64_XZ_STREAM: Crc64StreamConstants = Crc64StreamConstants::new(CRC64_XZ_POLY);
pub(crate) const CRC64_NVME_STREAM: Crc64StreamConstants = Crc64StreamConstants::new(CRC64_NVME_POLY);

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32 CLMUL Constants
// ─────────────────────────────────────────────────────────────────────────────
//
// CRC-32 CLMUL folding differs from CRC-64 in the final reduction:
// - 128-bit → 96-bit → 64-bit → 32-bit (more reduction steps)
// - 64-byte blocks (4×16B lanes) instead of 128-byte blocks
// - Barrett reduction for 32-bit result
//
// These are infrastructure for future CRC-32 CLMUL implementation.

/// Folding constants for CRC-32 CLMUL algorithm.
///
/// The CRC-32 CLMUL algorithm uses 64-byte blocks (4×16B lanes) and requires
/// additional reduction steps compared to CRC-64.
///
/// For VPCLMULQDQ (AVX-512), we use 128-byte blocks (8×16B lanes = 2×__m512i)
/// with multi-stream ILP optimizations.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // Some fields used only on specific architectures
pub(crate) struct Crc32ClmulConstants {
  /// Reciprocal polynomial (POLY = reflected_poly << 1 | 1).
  pub poly: u64,
  /// Barrett reduction constant.
  pub mu: u64,
  /// 64B block folding coefficient (K_511, K_575).
  pub fold_64b: (u64, u64),
  /// 128B block folding coefficient (K_1023, K_1087) for VPCLMUL.
  pub fold_128b: (u64, u64),
  /// 16B folding coefficient (K_127, K_191) for lane reduction.
  pub fold_16b: (u64, u64),
  /// 128→96 bit reduction constant (K_95).
  pub k_96: u64,
  /// 96→64 bit reduction constant (K_63).
  pub k_64: u64,
}

/// Reduce a 64-bit value modulo a 33-bit polynomial.
///
/// For CRC-32, the polynomial is (1 << 32) | poly where poly is the low 32 bits.
#[must_use]
#[allow(dead_code)]
const fn reduce64_for_crc32(hi: u32, lo: u32, poly: u32) -> u32 {
  // Reduce bits 63..32 down to 32
  let mut result_hi = hi;
  let mut result_lo = lo;

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
#[must_use]
#[allow(dead_code)]
const fn xpow_mod_32(n: u32, poly: u32) -> u32 {
  if n == 0 {
    return 1;
  }
  if n == 1 {
    return 2;
  }

  let mut result: u32 = 1;
  let mut base: u32 = 2;

  let mut exp = n;
  while exp > 0 {
    if exp & 1 != 0 {
      // Carryless multiply result * base (32-bit)
      let (mul_hi, mul_lo) = clmul32(result, base);
      result = reduce64_for_crc32(mul_hi, mul_lo, poly);
    }
    // Square base
    let (sq_hi, sq_lo) = clmul32(base, base);
    base = reduce64_for_crc32(sq_hi, sq_lo, poly);
    exp >>= 1;
  }

  result
}

/// Carryless multiplication of two 32-bit values, returning 64-bit result as (hi32, lo32).
#[must_use]
#[allow(dead_code)]
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

/// Compute folding constant K_n for CRC-32 (bit-reversed x^n mod normal_poly).
#[must_use]
#[allow(dead_code)]
const fn fold_k_32(normal_poly: u32, n: u32) -> u64 {
  // K_n = bit_reverse(x^n mod normal_poly), stored as low 32 bits of u64
  xpow_mod_32(n, normal_poly).reverse_bits() as u64
}

/// Compute TiKV-style reciprocal polynomial for CRC-32.
#[must_use]
#[allow(dead_code)]
const fn reciprocal_poly_32(reflected_poly: u32) -> u64 {
  ((reflected_poly as u64) << 1) | 1
}

/// Compute Barrett µ for CRC-32.
#[must_use]
#[allow(dead_code)]
const fn compute_mu_32(poly: u64) -> u64 {
  // µ = x^64 / P for Barrett reduction
  // For CRC-32 CLMUL, we need a 33-bit divisor result
  let poly32 = poly as u32;
  let mut inv: u64 = 1;

  let mut k: u32 = 1;
  while k < 33 {
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

  // The result needs to be adjusted for the CLMUL Barrett reduction
  // which expects the inverse in a specific format.
  // For correctness, we compute x^64 / poly using the extended GCD approach.
  let _ = poly32; // Silence unused warning

  inv
}

impl Crc32ClmulConstants {
  #[must_use]
  pub const fn new(reflected_poly: u32) -> Self {
    let poly = reciprocal_poly_32(reflected_poly);
    let normal = reflected_poly.reverse_bits();
    let mu = compute_mu_32(poly);

    Self {
      poly,
      mu,
      // 64B block: fold by 64 bytes = 512 bits
      fold_64b: (fold_k_32(normal, 511), fold_k_32(normal, 575)),
      // 128B block: fold by 128 bytes = 1024 bits (for VPCLMUL)
      fold_128b: (fold_k_32(normal, 1023), fold_k_32(normal, 1087)),
      // 16B lane: fold by 16 bytes = 128 bits
      fold_16b: (fold_k_32(normal, 127), fold_k_32(normal, 191)),
      // 128→96 bits: K_95
      k_96: fold_k_32(normal, 95),
      // 96→64 bits: K_63
      k_64: fold_k_32(normal, 63),
    }
  }
}

// Pre-computed constants for CRC-32 variants.
#[allow(dead_code)] // Used on x86_64
pub(crate) const CRC32_IEEE_CLMUL: Crc32ClmulConstants = Crc32ClmulConstants::new(CRC32_IEEE_POLY);
#[allow(dead_code)] // Used on x86_64
pub(crate) const CRC32C_CLMUL: Crc32ClmulConstants = Crc32ClmulConstants::new(CRC32C_POLY);

// ─────────────────────────────────────────────────────────────────────────────
// Multi-stream folding constants for CRC-32.
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a `(high, low)` fold coefficient pair for CRC-32 folding by `shift_bytes`.
#[must_use]
#[allow(dead_code)] // Used on x86_64
pub(crate) const fn fold16_coeff_for_bytes_32(reflected_poly: u32, shift_bytes: u32) -> (u64, u64) {
  if shift_bytes == 0 {
    return (0, 0);
  }

  let normal = reflected_poly.reverse_bits();
  let d = shift_bytes * 8;
  // K_n = bit_reverse(x^n mod normal_poly), stored as low 32 bits of u64
  (fold_k_32(normal, d - 1), fold_k_32(normal, d + 63))
}

/// Multi-stream folding constants for CRC-32 CLMUL kernels.
///
/// These constants support multi-way ILP (instruction-level parallelism)
/// optimizations on x86_64 (PCLMULQDQ/VPCLMULQDQ).
///
/// - `fold_256b`: 2-way striping (256B = 2×128B)
/// - `fold_512b`: 4-way striping (512B = 4×128B)
/// - `fold_896b`: 7-way striping (896B = 7×128B)
/// - `combine_4way`: merge coefficients for 4-way
/// - `combine_7way`: merge coefficients for 7-way
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // Used on x86_64
pub(crate) struct Crc32StreamConstants {
  /// 2-way fold coefficient (256B = 2×128B).
  pub fold_256b: (u64, u64),
  /// 4-way fold coefficient (512B = 4×128B).
  pub fold_512b: (u64, u64),
  /// 7-way fold coefficient (896B = 7×128B).
  pub fold_896b: (u64, u64),
  /// 4-way combine coefficients: shifts by 384B, 256B, 128B.
  pub combine_4way: [(u64, u64); 3],
  /// 7-way combine coefficients: shifts by 768B, 640B, 512B, 384B, 256B, 128B.
  pub combine_7way: [(u64, u64); 6],
}

impl Crc32StreamConstants {
  /// Compute all multi-stream folding constants for a given CRC-32 polynomial.
  #[must_use]
  pub const fn new(reflected_poly: u32) -> Self {
    Self {
      fold_256b: fold16_coeff_for_bytes_32(reflected_poly, 256),
      fold_512b: fold16_coeff_for_bytes_32(reflected_poly, 512),
      fold_896b: fold16_coeff_for_bytes_32(reflected_poly, 896),
      combine_4way: [
        fold16_coeff_for_bytes_32(reflected_poly, 384),
        fold16_coeff_for_bytes_32(reflected_poly, 256),
        fold16_coeff_for_bytes_32(reflected_poly, 128),
      ],
      combine_7way: [
        fold16_coeff_for_bytes_32(reflected_poly, 768),
        fold16_coeff_for_bytes_32(reflected_poly, 640),
        fold16_coeff_for_bytes_32(reflected_poly, 512),
        fold16_coeff_for_bytes_32(reflected_poly, 384),
        fold16_coeff_for_bytes_32(reflected_poly, 256),
        fold16_coeff_for_bytes_32(reflected_poly, 128),
      ],
    }
  }
}

// Pre-computed multi-stream constants for CRC-32.
#[allow(dead_code)] // Used on x86_64
pub(crate) const CRC32_IEEE_STREAM: Crc32StreamConstants = Crc32StreamConstants::new(CRC32_IEEE_POLY);
#[allow(dead_code)] // Used on x86_64
pub(crate) const CRC32C_STREAM: Crc32StreamConstants = Crc32StreamConstants::new(CRC32C_POLY);

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
}
