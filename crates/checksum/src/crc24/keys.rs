//! Carryless-multiply folding constants for CRC-24/OPENPGP (width32 strategy).
//!
//! CRC-24/OPENPGP is specified as an MSB-first CRC-24, but the portable
//! implementation internally expands the CRC into the top 24 bits of a 32-bit
//! register (low 8 bits cleared). This makes CRC-24 acceleration natural in a
//! **32-bit folding domain**:
//!
//! - Treat the CRC as a CRC-32 over the expanded register with `poly32 = CRC24_POLY << 8` (low 8
//!   bits cleared).
//! - Compute the equivalent reflected CRC32 over per-byte bit-reversed input.
//! - Convert back to the OpenPGP 24-bit representation (`state >> 8`).
//!
//! This module provides the constant key schedule used by both x86_64 (PCLMUL)
//! and aarch64 (PMULL) implementations.
//!
//! The key layout matches the CRC-32 "width32" layout: 23 u64 values, where
//! pairs are interpreted as 128-bit coefficients (high, low).

use crate::common::tables::CRC24_OPENPGP_POLY;

/// CRC-24/OPENPGP polynomial in reflected CRC-32 form (LSB-first).
///
/// This is `reverse_bits(poly32)` where `poly32 = CRC24_OPENPGP_POLY << 8`.
pub(crate) const CRC24_OPENPGP_POLY_REFLECTED: u32 = (CRC24_OPENPGP_POLY << 8).reverse_bits();

/// Key schedule for CRC-24/OPENPGP in the width32 folding strategy.
#[rustfmt::skip]
pub(crate) const CRC24_OPENPGP_KEYS_REFLECTED: [u64; 23] = build_keys(CRC24_OPENPGP_POLY_REFLECTED);

/// Multi-stream folding constants for CRC-24/OPENPGP in the width32 strategy.
///
/// These constants enable multi-way striping (2-way / 3-way / 4-way / 7-way / 8-way)
/// for carryless-multiply kernels.
#[allow(dead_code)] // Some fields are only used on specific architectures.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Crc24StreamConstants {
  pub fold_256b: (u64, u64),
  pub fold_384b: (u64, u64),
  pub fold_512b: (u64, u64),
  pub fold_896b: (u64, u64),
  pub fold_1024b: (u64, u64),
  pub combine_4way: [(u64, u64); 3],
  pub combine_7way: [(u64, u64); 6],
  pub combine_8way: [(u64, u64); 7],
}

impl Crc24StreamConstants {
  #[must_use]
  pub const fn new(reflected_poly: u32) -> Self {
    Self {
      fold_256b: fold16_coeff_for_bytes(reflected_poly, 256),
      fold_384b: fold16_coeff_for_bytes(reflected_poly, 384),
      fold_512b: fold16_coeff_for_bytes(reflected_poly, 512),
      fold_896b: fold16_coeff_for_bytes(reflected_poly, 896),
      fold_1024b: fold16_coeff_for_bytes(reflected_poly, 1024),
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
      combine_8way: [
        fold16_coeff_for_bytes(reflected_poly, 896),
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

pub(crate) const CRC24_OPENPGP_STREAM_REFLECTED: Crc24StreamConstants =
  Crc24StreamConstants::new(CRC24_OPENPGP_POLY_REFLECTED);

// ─────────────────────────────────────────────────────────────────────────────
// Constant Generation (compile-time)
// ─────────────────────────────────────────────────────────────────────────────

/// Carryless multiplication of two 64-bit values, returning the 128-bit result (hi, lo).
#[must_use]
const fn clmul64(a: u64, b: u64) -> (u64, u64) {
  let mut hi: u64 = 0;
  let mut lo: u64 = 0;

  let mut i: u32 = 0;
  while i < 64 {
    if (a.strict_shr(i)) & 1 != 0 {
      if i == 0 {
        lo ^= b;
      } else {
        lo ^= b.strict_shl(i);
        hi ^= b.strict_shr(64u32.strict_sub(i));
      }
    }
    i = i.strict_add(1);
  }

  (hi, lo)
}

/// Reduce a 128-bit value modulo a degree-32 polynomial `x^32 + poly`.
#[must_use]
const fn reduce128(hi: u64, lo: u64, poly: u32) -> u32 {
  let poly_full: u128 = (1u128.strict_shl(32)) | (poly as u128);
  let mut val: u128 = (hi as u128).strict_shl(64) | (lo as u128);

  let mut bit: i32 = 127;
  while bit >= 32 {
    let b = bit as u32;
    if ((val.strict_shr(b)) & 1) != 0 {
      val ^= poly_full.strict_shl(b.strict_sub(32));
    }
    bit = bit.strict_sub(1);
  }

  val as u32
}

/// Compute x^n mod (x^width + poly) in GF(2) where `poly` is the normal CRC polynomial
/// without the x^width term.
#[must_use]
const fn xpow_mod(mut n: u32, poly: u32) -> u32 {
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
      result = reduce128(hi, lo, poly);
    }
    let (hi, lo) = clmul64(base as u64, base as u64);
    base = reduce128(hi, lo, poly);
    n = n.strict_shr(1);
  }

  result
}

/// Reverse the low 33 bits of `v`.
#[must_use]
const fn reverse33(v: u64) -> u64 {
  let mask = (1u64.strict_shl(33)).strict_sub(1);
  (v & mask).reverse_bits().strict_shr(31)
}

/// Compute Intel/TiKV-style folding constant K_n for CRC-32 in reflected mode.
#[must_use]
const fn fold_k(reflected_poly: u32, n: u32) -> u64 {
  let normal_poly = reflected_poly.reverse_bits();
  let rem = xpow_mod(n, normal_poly) as u64;
  reverse33(rem)
}

/// Compute a `(high, low)` fold coefficient pair for folding 16 bytes by `shift_bytes`.
///
/// Returns `(K_{d+32}, K_{d-32})` where `d = 8 * shift_bytes`.
#[must_use]
const fn fold16_coeff_for_bytes(reflected_poly: u32, shift_bytes: u32) -> (u64, u64) {
  if shift_bytes == 0 {
    return (0, 0);
  }
  let d = shift_bytes.strict_mul(8);
  if d < 32 {
    return (0, 0);
  }
  (
    fold_k(reflected_poly, d.strict_add(32)),
    fold_k(reflected_poly, d.strict_sub(32)),
  )
}

/// Compute TiKV-style `MU` for Barrett reduction: `MU = POLY⁻¹ mod x^64`.
#[must_use]
const fn compute_mu(poly: u64) -> u64 {
  let mut inv: u64 = 1;

  let mut k: u32 = 1;
  while k < 64 {
    let mut s: u64 = 0;
    let mut i: u32 = 1;
    while i <= k {
      let p_i = (poly.strict_shr(i)) & 1;
      let q_j = (inv.strict_shr(k.strict_sub(i))) & 1;
      s ^= p_i & q_j;
      i = i.strict_add(1);
    }
    inv |= s.strict_shl(k);
    k = k.strict_add(1);
  }

  inv
}

#[must_use]
const fn build_keys(reflected_poly: u32) -> [u64; 23] {
  let poly: u64 = (reflected_poly as u64).strict_shl(1) | 1;
  let mu: u64 = compute_mu(poly);

  let (k16_hi, k16_lo) = fold16_coeff_for_bytes(reflected_poly, 16);
  let (k128_hi, k128_lo) = fold16_coeff_for_bytes(reflected_poly, 128);
  let (k256_hi, k256_lo) = fold16_coeff_for_bytes(reflected_poly, 256);

  let (k112_hi, k112_lo) = fold16_coeff_for_bytes(reflected_poly, 112);
  let (k96_hi, k96_lo) = fold16_coeff_for_bytes(reflected_poly, 96);
  let (k80_hi, k80_lo) = fold16_coeff_for_bytes(reflected_poly, 80);
  let (k64_hi, k64_lo) = fold16_coeff_for_bytes(reflected_poly, 64);
  let (k48_hi, k48_lo) = fold16_coeff_for_bytes(reflected_poly, 48);
  let (k32_hi, k32_lo) = fold16_coeff_for_bytes(reflected_poly, 32);

  // Width reduction constants:
  // - 16B→8B uses K_{64+32} = K_96
  // - 8B→4B uses K_{32+32} = K_64
  let fold_8b: u64 = fold_k(reflected_poly, 96);
  let fold_4b: u64 = fold_k(reflected_poly, 64);

  [
    0,       // 0: unused placeholder to match existing key layouts
    k16_lo,  // 1: K_{128-32}  (16 bytes)
    k16_hi,  // 2: K_{128+32}  (16 bytes)
    k128_lo, // 3: K_{1024-32} (128 bytes)
    k128_hi, // 4: K_{1024+32} (128 bytes)
    fold_8b, // 5: K_96
    fold_4b, // 6: K_64
    mu,      // 7: MU
    poly,    // 8: POLY
    k112_lo, // 9:  K_{896-32} (112 bytes)
    k112_hi, // 10: K_{896+32} (112 bytes)
    k96_lo,  // 11: K_{768-32} (96 bytes)
    k96_hi,  // 12: K_{768+32} (96 bytes)
    k80_lo,  // 13: K_{640-32} (80 bytes)
    k80_hi,  // 14: K_{640+32} (80 bytes)
    k64_lo,  // 15: K_{512-32} (64 bytes)
    k64_hi,  // 16: K_{512+32} (64 bytes)
    k48_lo,  // 17: K_{384-32} (48 bytes)
    k48_hi,  // 18: K_{384+32} (48 bytes)
    k32_lo,  // 19: K_{256-32} (32 bytes)
    k32_hi,  // 20: K_{256+32} (32 bytes)
    k256_lo, // 21: K_{2048-32}
    k256_hi, // 22: K_{2048+32}
  ]
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_crc24_openpgp_reflected_poly_matches_expanded_crc32_form() {
    assert_eq!(CRC24_OPENPGP_POLY_REFLECTED, 0x00DF_3261);
  }

  #[test]
  fn test_crc24_openpgp_key_schedule_sanity() {
    let keys = &CRC24_OPENPGP_KEYS_REFLECTED;
    assert_eq!(keys[2], fold_k(CRC24_OPENPGP_POLY_REFLECTED, 160));
    assert_eq!(keys[1], fold_k(CRC24_OPENPGP_POLY_REFLECTED, 96));

    // Width reduction constants (16B→8B and 8B→4B).
    assert_eq!(keys[5], fold_k(CRC24_OPENPGP_POLY_REFLECTED, 96));
    assert_eq!(keys[6], fold_k(CRC24_OPENPGP_POLY_REFLECTED, 64));
  }
}
