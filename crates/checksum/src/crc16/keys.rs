//! Carryless-multiply folding constants for CRC-16 (width32 strategy).
//!
//! CRC-16 acceleration reuses the same "width32" folding/reduction structure
//! as the reflected CRC-32 CLMUL kernels, but with CRC-16-specific constants.
//!
//! These constants are shared across all accelerated backends (x86_64 PCLMUL /
//! VPCLMUL, aarch64 PMULL, and other carryless-multiply platforms).

use crate::common::tables::{CRC16_CCITT_POLY, CRC16_IBM_POLY};

/// Key schedule for CRC-16/CCITT (X.25 / IBM-SDLC), reflected polynomial.
#[rustfmt::skip]
pub(crate) const CRC16_CCITT_KEYS_REFLECTED: [u64; 23] = [
  0x0000000000000000,
  0x00000000000189ae,
  0x0000000000008e10,
  0x00000000000160be,
  0x000000000001bed8,
  0x00000000000189ae,
  0x00000000000114aa,
  0x000000011c581911,
  0x0000000000010811,
  0x000000000001ce5e,
  0x000000000001c584,
  0x000000000001db50,
  0x000000000000b8f2,
  0x0000000000000842,
  0x000000000000b072,
  0x0000000000014ff2,
  0x0000000000019a3c,
  0x0000000000000e3a,
  0x0000000000004d7a,
  0x0000000000005b44,
  0x0000000000007762,
  0x0000000000019208,
  0x0000000000002df8,
];

/// Key schedule for CRC-16/IBM (ARC), reflected polynomial.
#[rustfmt::skip]
pub(crate) const CRC16_IBM_KEYS_REFLECTED: [u64; 23] = [
  0x0000000000000000,
  0x0000000000018cc2,
  0x000000000001d0c2,
  0x0000000000014cc2,
  0x000000000001dc02,
  0x0000000000018cc2,
  0x000000000001bc02,
  0x00000001cfffbfff,
  0x0000000000014003,
  0x000000000000bcac,
  0x000000000001a674,
  0x000000000001ac00,
  0x0000000000019b6e,
  0x000000000001d33e,
  0x000000000001c462,
  0x000000000000bffa,
  0x000000000001b0c2,
  0x00000000000186ae,
  0x000000000001ad6e,
  0x000000000001d55e,
  0x000000000001ec02,
  0x000000000001d99e,
  0x000000000001bcc2,
];

/// Multi-stream folding constants for width32 CRCs.
///
/// These are shared across all CRCs implemented via the width32 strategy
/// (CRC-16 and CRC-24 today). They describe how to:
/// - advance a stream by `N * 128B` per iteration (striping)
/// - merge streams back together (combine coefficients)
#[allow(dead_code)] // Some fields are only used on specific architectures.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Width32StreamConstants {
  /// 2-way fold coefficient (256B = 2×128B).
  pub fold_256b: (u64, u64),
  /// 3-way fold coefficient (384B = 3×128B).
  pub fold_384b: (u64, u64),
  /// 4-way fold coefficient (512B = 4×128B).
  pub fold_512b: (u64, u64),
  /// 7-way fold coefficient (896B = 7×128B).
  pub fold_896b: (u64, u64),
  /// 8-way fold coefficient (1024B = 8×128B).
  pub fold_1024b: (u64, u64),
  /// 4-way combine coefficients: shifts by 384B, 256B, 128B.
  pub combine_4way: [(u64, u64); 3],
  /// 7-way combine coefficients: shifts by 768B, 640B, 512B, 384B, 256B, 128B.
  pub combine_7way: [(u64, u64); 6],
  /// 8-way combine coefficients: shifts by 896B, 768B, 640B, 512B, 384B, 256B, 128B.
  pub combine_8way: [(u64, u64); 7],
}

impl Width32StreamConstants {
  /// Compute all multi-stream folding constants for a given reflected polynomial.
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

pub(crate) const CRC16_CCITT_STREAM_REFLECTED: Width32StreamConstants =
  Width32StreamConstants::new(CRC16_CCITT_POLY as u32);
pub(crate) const CRC16_IBM_STREAM_REFLECTED: Width32StreamConstants =
  Width32StreamConstants::new(CRC16_IBM_POLY as u32);

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
