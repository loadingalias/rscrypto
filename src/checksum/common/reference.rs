//! Bitwise reference implementations for all CRC widths.
//!
//! This module provides the canonical "source of truth" for CRC computation.
//! These implementations process one bit at a time, making them:
//!
//! - **Obviously correct**: The algorithm directly mirrors the mathematical definition
//! - **Audit-friendly**: ~10 lines of code per width, no lookup tables
//! - **Const-evaluable**: Can verify check values at compile time
//!
//! All optimized implementations (slice-by-N, SIMD) must produce identical
//! results to these reference functions.
//!
//! # CRC Model
//!
//! These implementations follow the Rocksoft model (CRC RevEng catalog):
//!
//! | Parameter | Description |
//! |-----------|-------------|
//! | `width`   | CRC width in bits (16, 24, 32, 64) |
//! | `poly`    | Generator polynomial (reflected for LSB-first CRCs) |
//! | `init`    | Initial register value |
//! | `refin`   | Reflect input bytes (true for most CRCs) |
//! | `refout`  | Reflect output before final XOR (true for most CRCs) |
//! | `xorout`  | Final XOR value |
//!
//! # Performance
//!
//! These are intentionally slow (~8 operations per bit). Use for:
//! - Correctness verification
//! - Test oracles
//! - Generating expected values
//! - Auditing algorithm correctness
//!
//! For production throughput, use the auto-selected implementations.

// SAFETY: All array indexing uses bounded loop indices (0..data.len()).
// Clippy cannot prove this in const fn contexts, but bounds are statically guaranteed.
#![allow(clippy::indexing_slicing)]

// ─────────────────────────────────────────────────────────────────────────────
// CRC-16 Reference Implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Bitwise CRC-16 computation (reflected, LSB-first).
///
/// Processes input one bit at a time. This is the canonical reference
/// against which all CRC-16 optimizations are verified.
///
/// # Arguments
///
/// * `poly` - Reflected polynomial (e.g., 0x8408 for CRC-16-CCITT)
/// * `init` - Initial register value (typically 0xFFFF or 0x0000)
/// * `data` - Input bytes
///
/// # Returns
///
/// The raw CRC register state (caller applies final XOR if needed).
#[must_use]
pub const fn crc16_bitwise(poly: u16, init: u16, data: &[u8]) -> u16 {
  let mut crc = init;
  let mut i: usize = 0;
  while i < data.len() {
    crc ^= data[i] as u16;
    let mut bit: u32 = 0;
    while bit < 8 {
      crc = if crc & 1 != 0 { (crc >> 1) ^ poly } else { crc >> 1 };
      bit = bit.strict_add(1);
    }
    i = i.strict_add(1);
  }
  crc
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-24 Reference Implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Bitwise CRC-24 computation (non-reflected, MSB-first).
///
/// CRC-24 (OpenPGP) uses MSB-first processing, unlike the reflected CRCs.
/// The polynomial and CRC are aligned to the top of a 32-bit register.
///
/// # Arguments
///
/// * `poly` - Normal polynomial (e.g., 0x864CFB for CRC-24-OPENPGP)
/// * `init` - Initial register value (0xB704CE for OpenPGP)
/// * `data` - Input bytes
///
/// # Returns
///
/// The CRC value in the low 24 bits.
#[must_use]
pub const fn crc24_bitwise(poly: u32, init: u32, data: &[u8]) -> u32 {
  // Work in expanded form: CRC in top 24 bits of u32
  let poly_expanded = poly.strict_shl(8);
  let mut crc = (init & 0x00FF_FFFF).strict_shl(8);

  let mut i: usize = 0;
  while i < data.len() {
    crc ^= (data[i] as u32).strict_shl(24);
    let mut bit: u32 = 0;
    while bit < 8 {
      crc = if crc & 0x8000_0000 != 0 {
        crc.strict_shl(1) ^ poly_expanded
      } else {
        crc.strict_shl(1)
      };
      bit = bit.strict_add(1);
    }
    i = i.strict_add(1);
  }

  // Return in low 24 bits
  (crc >> 8) & 0x00FF_FFFF
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32 Reference Implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Bitwise CRC-32 computation (reflected, LSB-first).
///
/// Processes input one bit at a time. This is the canonical reference
/// against which all CRC-32 optimizations are verified.
///
/// # Arguments
///
/// * `poly` - Reflected polynomial (e.g., 0xEDB88320 for CRC-32-IEEE)
/// * `init` - Initial register value (typically 0xFFFFFFFF)
/// * `data` - Input bytes
///
/// # Returns
///
/// The raw CRC register state (caller applies final XOR if needed).
#[must_use]
pub const fn crc32_bitwise(poly: u32, init: u32, data: &[u8]) -> u32 {
  let mut crc = init;
  let mut i: usize = 0;
  while i < data.len() {
    crc ^= data[i] as u32;
    let mut bit: u32 = 0;
    while bit < 8 {
      crc = if crc & 1 != 0 { (crc >> 1) ^ poly } else { crc >> 1 };
      bit = bit.strict_add(1);
    }
    i = i.strict_add(1);
  }
  crc
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-64 Reference Implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Bitwise CRC-64 computation (reflected, LSB-first).
///
/// Processes input one bit at a time. This is the canonical reference
/// against which all CRC-64 optimizations are verified.
///
/// # Arguments
///
/// * `poly` - Reflected polynomial (e.g., 0xC96C5795D7870F42 for CRC-64-XZ)
/// * `init` - Initial register value (typically 0xFFFFFFFFFFFFFFFF)
/// * `data` - Input bytes
///
/// # Returns
///
/// The raw CRC register state (caller applies final XOR if needed).
#[must_use]
pub const fn crc64_bitwise(poly: u64, init: u64, data: &[u8]) -> u64 {
  let mut crc = init;
  let mut i: usize = 0;
  while i < data.len() {
    crc ^= data[i] as u64;
    let mut bit: u32 = 0;
    while bit < 8 {
      crc = if crc & 1 != 0 { (crc >> 1) ^ poly } else { crc >> 1 };
      bit = bit.strict_add(1);
    }
    i = i.strict_add(1);
  }
  crc
}

// ─────────────────────────────────────────────────────────────────────────────
// Compile-Time Verification
// ─────────────────────────────────────────────────────────────────────────────

// These const assertions verify the reference implementations against known
// check values at compile time. If these fail, the build fails.

use super::tables::{
  CRC16_CCITT_POLY, CRC24_OPENPGP_POLY, CRC32_IEEE_POLY, CRC32C_POLY, CRC64_NVME_POLY, CRC64_XZ_POLY,
};

/// Standard test input for CRC check values.
const CHECK_INPUT: &[u8] = b"123456789";

// CRC-16-CCITT: init=0xFFFF, xorout=0xFFFF
// Check value: 0x906E (per CRC RevEng catalog for CRC-16/CCITT-FALSE reflected variant)
const _: () = {
  let raw = crc16_bitwise(CRC16_CCITT_POLY, !0u16, CHECK_INPUT);
  let check = raw ^ !0u16;
  assert!(check == 0x906E);
};

// CRC-24-OPENPGP: init=0xB704CE, xorout=0x000000
// Check value: 0x21CF02
const _: () = {
  let check = crc24_bitwise(CRC24_OPENPGP_POLY, 0x00B7_04CE, CHECK_INPUT);
  assert!(check == 0x0021_CF02);
};

// CRC-32-IEEE: init=0xFFFFFFFF, xorout=0xFFFFFFFF
// Check value: 0xCBF43926
const _: () = {
  let raw = crc32_bitwise(CRC32_IEEE_POLY, !0u32, CHECK_INPUT);
  let check = raw ^ !0u32;
  assert!(check == 0xCBF4_3926);
};

// CRC-32C (Castagnoli): init=0xFFFFFFFF, xorout=0xFFFFFFFF
// Check value: 0xE3069283
const _: () = {
  let raw = crc32_bitwise(CRC32C_POLY, !0u32, CHECK_INPUT);
  let check = raw ^ !0u32;
  assert!(check == 0xE306_9283);
};

// CRC-64-XZ: init=0xFFFFFFFFFFFFFFFF, xorout=0xFFFFFFFFFFFFFFFF
// Check value: 0x995DC9BBDF1939FA
const _: () = {
  let raw = crc64_bitwise(CRC64_XZ_POLY, !0u64, CHECK_INPUT);
  let check = raw ^ !0u64;
  assert!(check == 0x995D_C9BB_DF19_39FA);
};

// CRC-64-NVME: init=0xFFFFFFFFFFFFFFFF, xorout=0xFFFFFFFFFFFFFFFF
// Check value: 0xAE8B14860A799888
const _: () = {
  let raw = crc64_bitwise(CRC64_NVME_POLY, !0u64, CHECK_INPUT);
  let check = raw ^ !0u64;
  assert!(check == 0xAE8B_1486_0A79_9888);
};

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16 Tests
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc16_empty() {
    // Empty input should return init XOR xorout
    let raw = crc16_bitwise(CRC16_CCITT_POLY, !0u16, &[]);
    assert_eq!(raw ^ !0u16, 0);
  }

  #[test]
  fn crc16_single_bytes() {
    // Verify single-byte CRCs are consistent
    for byte in 0u8..=255 {
      let crc = crc16_bitwise(CRC16_CCITT_POLY, !0u16, &[byte]);
      // Just verify it doesn't panic and produces a value
      let _ = crc;
    }
  }

  #[test]
  fn crc16_incremental() {
    // Verify incremental computation matches one-shot
    let data = b"The quick brown fox jumps over the lazy dog";
    let oneshot = crc16_bitwise(CRC16_CCITT_POLY, !0u16, data);

    for split in 1..data.len() {
      let first = crc16_bitwise(CRC16_CCITT_POLY, !0u16, &data[..split]);
      let second = crc16_bitwise(CRC16_CCITT_POLY, first, &data[split..]);
      assert_eq!(second, oneshot, "Incremental mismatch at split {split}");
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-24 Tests
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc24_empty() {
    // Empty input should return init (no xorout for OpenPGP)
    let crc = crc24_bitwise(CRC24_OPENPGP_POLY, 0x00B7_04CE, &[]);
    assert_eq!(crc, 0x00B7_04CE);
  }

  #[test]
  fn crc24_incremental() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let oneshot = crc24_bitwise(CRC24_OPENPGP_POLY, 0x00B7_04CE, data);

    for split in 1..data.len() {
      let first = crc24_bitwise(CRC24_OPENPGP_POLY, 0x00B7_04CE, &data[..split]);
      let second = crc24_bitwise(CRC24_OPENPGP_POLY, first, &data[split..]);
      assert_eq!(second, oneshot, "Incremental mismatch at split {split}");
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-32 Tests
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc32_empty() {
    let raw = crc32_bitwise(CRC32_IEEE_POLY, !0u32, &[]);
    assert_eq!(raw ^ !0u32, 0);
  }

  #[test]
  fn crc32_single_bytes() {
    for byte in 0u8..=255 {
      let crc = crc32_bitwise(CRC32_IEEE_POLY, !0u32, &[byte]);
      let _ = crc;
    }
  }

  #[test]
  fn crc32_incremental() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let oneshot = crc32_bitwise(CRC32_IEEE_POLY, !0u32, data);

    for split in 1..data.len() {
      let first = crc32_bitwise(CRC32_IEEE_POLY, !0u32, &data[..split]);
      let second = crc32_bitwise(CRC32_IEEE_POLY, first, &data[split..]);
      assert_eq!(second, oneshot, "Incremental mismatch at split {split}");
    }
  }

  #[test]
  fn crc32c_check_value() {
    let raw = crc32_bitwise(CRC32C_POLY, !0u32, CHECK_INPUT);
    assert_eq!(raw ^ !0u32, 0xE306_9283);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-64 Tests
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc64_empty() {
    let raw = crc64_bitwise(CRC64_XZ_POLY, !0u64, &[]);
    assert_eq!(raw ^ !0u64, 0);

    let raw = crc64_bitwise(CRC64_NVME_POLY, !0u64, &[]);
    assert_eq!(raw ^ !0u64, 0);
  }

  #[test]
  fn crc64_single_bytes() {
    for byte in 0u8..=255 {
      let crc = crc64_bitwise(CRC64_XZ_POLY, !0u64, &[byte]);
      let _ = crc;
    }
  }

  #[test]
  fn crc64_incremental() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let oneshot = crc64_bitwise(CRC64_XZ_POLY, !0u64, data);

    for split in 1..data.len() {
      let first = crc64_bitwise(CRC64_XZ_POLY, !0u64, &data[..split]);
      let second = crc64_bitwise(CRC64_XZ_POLY, first, &data[split..]);
      assert_eq!(second, oneshot, "Incremental mismatch at split {split}");
    }
  }

  #[test]
  fn crc64_nvme_check_value() {
    let raw = crc64_bitwise(CRC64_NVME_POLY, !0u64, CHECK_INPUT);
    assert_eq!(raw ^ !0u64, 0xAE8B_1486_0A79_9888);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Cross-Width Consistency
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn all_widths_handle_large_input() {
    // Verify all widths can handle larger inputs without panic
    let data: [u8; 1024] = core::array::from_fn(|i| (i as u8).wrapping_mul(17));

    let _ = crc16_bitwise(CRC16_CCITT_POLY, !0u16, &data);
    let _ = crc24_bitwise(CRC24_OPENPGP_POLY, 0x00B7_04CE, &data);
    let _ = crc32_bitwise(CRC32_IEEE_POLY, !0u32, &data);
    let _ = crc64_bitwise(CRC64_XZ_POLY, !0u64, &data);
  }
}
