//! Carryless-multiply folding constants for reflected CRC-32 polynomials.
//!
//! This module mirrors the CRC64 CLMUL constants generator but targets the
//! CRC32 folding/reduction strategy used by the 128-byte block folding kernels.

use crate::common::tables::{CRC32_IEEE_POLY, CRC32C_POLY};

// ─────────────────────────────────────────────────────────────────────────────
// GF(2) polynomial arithmetic (CRC-32 width)
// ─────────────────────────────────────────────────────────────────────────────

/// Carryless multiplication of two 64-bit values, returning 128-bit result (hi, lo).
#[must_use]
const fn clmul64(a: u64, b: u64) -> (u64, u64) {
  let mut hi: u64 = 0;
  let mut lo: u64 = 0;

  let mut i: u32 = 0;
  while i < 64 {
    if (a >> i) & 1 != 0 {
      if i == 0 {
        lo ^= b;
      } else {
        lo ^= b << i;
        hi ^= b >> (64u32.strict_sub(i));
      }
    }
    i = i.strict_add(1);
  }

  (hi, lo)
}

/// Reduce a 128-bit value modulo a degree-32 polynomial `x^32 + poly` (poly is normal form).
#[must_use]
const fn reduce128_crc32(hi: u64, lo: u64, poly: u32) -> u32 {
  let poly_full: u128 = (1u128 << 32) | (poly as u128);
  let mut val: u128 = ((hi as u128) << 64) | (lo as u128);

  let mut bit: i32 = 127;
  while bit >= 32 {
    let b = bit as u32;
    if ((val >> b) & 1) != 0 {
      val ^= poly_full << b.strict_sub(32);
    }
    bit = bit.strict_sub(1);
  }

  val as u32
}

/// Compute `x^n mod (x^32 + poly)` in GF(2), where `poly` is the **normal** CRC polynomial
/// without the `x^32` term.
#[must_use]
const fn xpow_mod_crc32(mut n: u32, poly: u32) -> u32 {
  if n == 0 {
    return 1;
  }
  if n == 1 {
    return 2;
  }

  let mut result: u32 = 1;
  let mut base: u32 = 2;

  while n > 0 {
    if n & 1 != 0 {
      let (hi, lo) = clmul64(result as u64, base as u64);
      result = reduce128_crc32(hi, lo, poly);
    }
    let (hi, lo) = clmul64(base as u64, base as u64);
    base = reduce128_crc32(hi, lo, poly);
    n >>= 1;
  }

  result
}

/// Reverse the low 33 bits of `v`.
#[must_use]
const fn reverse33(v: u64) -> u64 {
  let mask = (1u64.strict_shl(33)).strict_sub(1);
  (v & mask).reverse_bits() >> 31
}

/// Compute folding constant `K_n` for reflected CRC-32 polynomials.
///
/// `K_n = reverse33(x^n mod (x^32 ⊕ NORMAL))`.
#[must_use]
const fn fold_k_crc32(reflected_poly: u32, n: u32) -> u64 {
  let normal_poly = reflected_poly.reverse_bits();
  let rem = xpow_mod_crc32(n, normal_poly) as u64;
  reverse33(rem)
}

/// Compute the TiKV-style reciprocal polynomial for reflected CRC-32.
///
/// `POLY = (reflected_poly << 1) | 1` (33-bit value).
#[must_use]
const fn reciprocal_poly_crc32(reflected_poly: u32) -> u64 {
  ((reflected_poly as u64) << 1) | 1
}

/// Compute `MU` for reflected CRC-32 folding reduction.
///
/// CRC32 kernels operate on 33-bit polynomials (TiKV reciprocal form). We compute
/// the inverse series modulo `x^33` (33 low bits).
#[must_use]
const fn compute_mu33(poly: u64) -> u64 {
  let mut inv: u64 = 1;

  let mut k: u32 = 1;
  while k < 33 {
    let mut s: u64 = 0;

    let mut i: u32 = 1;
    while i <= k {
      let p_i = (poly >> i) & 1;
      let q_j = (inv >> (k - i)) & 1;
      s ^= p_i & q_j;
      i = i.strict_add(1);
    }

    inv |= s << k;
    k = k.strict_add(1);
  }

  inv
}

/// Compute a `(high, low)` fold coefficient pair for folding 16 bytes by `shift_bytes`.
///
/// Returns `(K_{d+32}, K_{d-32})` where `d = 8 * shift_bytes`.
#[must_use]
pub(super) const fn fold16_coeff_for_bytes_crc32(reflected_poly: u32, shift_bytes: u32) -> (u64, u64) {
  if shift_bytes == 0 {
    return (0, 0);
  }
  let d = shift_bytes.strict_mul(8);
  if d < 32 {
    return (0, 0);
  }

  (
    fold_k_crc32(reflected_poly, d.strict_add(32)),
    fold_k_crc32(reflected_poly, d.strict_sub(32)),
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Constant sets for the folding kernels
// ─────────────────────────────────────────────────────────────────────────────

/// Folding constants for a reflected CRC-32 polynomial.
#[derive(Clone, Copy, Debug)]
pub(super) struct Crc32ClmulConstants {
  pub poly: u64,
  pub mu: u64,
  pub fold_128b: (u64, u64),
  pub tail_fold_16b: [(u64, u64); 7],
  /// `(high, low)` pair for `fold_width_crc32_reflected`.
  pub fold_width: (u64, u64),
}

impl Crc32ClmulConstants {
  #[must_use]
  pub const fn new(reflected_poly: u32) -> Self {
    let poly = reciprocal_poly_crc32(reflected_poly);
    let mu = compute_mu33(poly);

    Self {
      poly,
      mu,
      fold_128b: fold16_coeff_for_bytes_crc32(reflected_poly, 128),
      tail_fold_16b: [
        fold16_coeff_for_bytes_crc32(reflected_poly, 112),
        fold16_coeff_for_bytes_crc32(reflected_poly, 96),
        fold16_coeff_for_bytes_crc32(reflected_poly, 80),
        fold16_coeff_for_bytes_crc32(reflected_poly, 64),
        fold16_coeff_for_bytes_crc32(reflected_poly, 48),
        fold16_coeff_for_bytes_crc32(reflected_poly, 32),
        fold16_coeff_for_bytes_crc32(reflected_poly, 16),
      ],
      // 16B → 8B uses K_96 (matches the 16B fold low coeff), 8B → 4B uses K_64.
      fold_width: (fold_k_crc32(reflected_poly, 64), fold_k_crc32(reflected_poly, 96)),
    }
  }
}

/// Multi-stream folding constants for CRC-32 CLMUL kernels.
#[derive(Clone, Copy, Debug)]
pub(super) struct Crc32StreamConstants {
  pub fold_256b: (u64, u64),
  pub fold_512b: (u64, u64),
  pub fold_1024b: (u64, u64),
  pub combine_4way: [(u64, u64); 3],
  pub combine_8way: [(u64, u64); 7],
}

impl Crc32StreamConstants {
  #[must_use]
  pub const fn new(reflected_poly: u32) -> Self {
    Self {
      fold_256b: fold16_coeff_for_bytes_crc32(reflected_poly, 256),
      fold_512b: fold16_coeff_for_bytes_crc32(reflected_poly, 512),
      fold_1024b: fold16_coeff_for_bytes_crc32(reflected_poly, 1024),
      combine_4way: [
        fold16_coeff_for_bytes_crc32(reflected_poly, 384),
        fold16_coeff_for_bytes_crc32(reflected_poly, 256),
        fold16_coeff_for_bytes_crc32(reflected_poly, 128),
      ],
      combine_8way: [
        fold16_coeff_for_bytes_crc32(reflected_poly, 896),
        fold16_coeff_for_bytes_crc32(reflected_poly, 768),
        fold16_coeff_for_bytes_crc32(reflected_poly, 640),
        fold16_coeff_for_bytes_crc32(reflected_poly, 512),
        fold16_coeff_for_bytes_crc32(reflected_poly, 384),
        fold16_coeff_for_bytes_crc32(reflected_poly, 256),
        fold16_coeff_for_bytes_crc32(reflected_poly, 128),
      ],
    }
  }
}

pub(super) const CRC32_IEEE_CLMUL: Crc32ClmulConstants = Crc32ClmulConstants::new(CRC32_IEEE_POLY);
pub(super) const CRC32C_CLMUL: Crc32ClmulConstants = Crc32ClmulConstants::new(CRC32C_POLY);

pub(super) const CRC32_IEEE_STREAM: Crc32StreamConstants = Crc32StreamConstants::new(CRC32_IEEE_POLY);
pub(super) const CRC32C_STREAM: Crc32StreamConstants = Crc32StreamConstants::new(CRC32C_POLY);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn mu_is_33_bit() {
    assert!(CRC32_IEEE_CLMUL.mu < (1u64 << 33));
    assert!(CRC32C_CLMUL.mu < (1u64 << 33));
  }

  #[test]
  fn poly_is_reciprocal() {
    assert_eq!(CRC32_IEEE_CLMUL.poly, ((CRC32_IEEE_POLY as u64) << 1) | 1);
    assert_eq!(CRC32C_CLMUL.poly, ((CRC32C_POLY as u64) << 1) | 1);
  }
}
