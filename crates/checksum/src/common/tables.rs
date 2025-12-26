//! Const-fn CRC lookup table generation for all widths.
//!
//! This module provides compile-time table generation for CRC-16, CRC-24,
//! and CRC-64. Tables are computed using `const fn` and embedded
//! directly in the binary.
//!
//! # Table Strategies
//!
//! Each CRC width supports multiple slice-by-N strategies:
//!
//! | Width | Slice-by-4 | Slice-by-8 | Slice-by-16 |
//! |-------|------------|------------|-------------|
//! | 16-bit | 4×256×u16 | 8×256×u16 | - |
//! | 24-bit | 4×256×u32 | 8×256×u32 | - |
//! | 64-bit | - | 8×256×u64 | 16×256×u64 |
//!
//! Higher slice counts provide better throughput at the cost of table size.

// SAFETY: All array indexing in this module uses bounded loop indices (0..256, 0..N).
// Clippy cannot prove this in const fn contexts, but bounds are statically guaranteed.
#![allow(clippy::indexing_slicing)]

// ─────────────────────────────────────────────────────────────────────────────
// CRC-16 Table Generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a single CRC-16 lookup table entry.
///
/// Uses bit-by-bit computation with the reflected polynomial.
#[cfg(test)] // Will be used when CRC-16 module is added
#[must_use]
pub const fn crc16_table_entry(poly: u16, index: u8) -> u16 {
  let mut crc = index as u16;
  let mut i = 0;
  while i < 8 {
    if crc & 1 != 0 {
      crc = (crc >> 1) ^ poly;
    } else {
      crc >>= 1;
    }
    i += 1;
  }
  crc
}

/// Generate 4 CRC-16 lookup tables for slice-by-4 computation.
///
/// # Arguments
///
/// * `poly` - The reflected polynomial
#[cfg(test)] // Will be used when CRC-16 module is added
#[must_use]
pub const fn generate_crc16_tables_4(poly: u16) -> [[u16; 256]; 4] {
  let mut tables = [[0u16; 256]; 4];

  let mut i = 0u16;
  while i < 256 {
    tables[0][i as usize] = crc16_table_entry(poly, i as u8);
    i += 1;
  }

  let mut k = 1usize;
  while k < 4 {
    i = 0;
    while i < 256 {
      let prev = tables[k - 1][i as usize];
      tables[k][i as usize] = tables[0][(prev & 0xFF) as usize] ^ (prev >> 8);
      i += 1;
    }
    k += 1;
  }

  tables
}

/// Generate 8 CRC-16 lookup tables for slice-by-8 computation.
///
/// # Arguments
///
/// * `poly` - The reflected polynomial
#[cfg(test)] // Will be used when CRC-16 module is added
#[must_use]
pub const fn generate_crc16_tables_8(poly: u16) -> [[u16; 256]; 8] {
  let mut tables = [[0u16; 256]; 8];

  let mut i = 0u16;
  while i < 256 {
    tables[0][i as usize] = crc16_table_entry(poly, i as u8);
    i += 1;
  }

  let mut k = 1usize;
  while k < 8 {
    i = 0;
    while i < 256 {
      let prev = tables[k - 1][i as usize];
      tables[k][i as usize] = tables[0][(prev & 0xFF) as usize] ^ (prev >> 8);
      i += 1;
    }
    k += 1;
  }

  tables
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-24 Table Generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a single CRC-24 lookup table entry.
///
/// Uses bit-by-bit computation with the reflected polynomial.
/// The result is stored in the low 24 bits of a u32.
#[cfg(test)] // Will be used when CRC-24 module is added
#[must_use]
pub const fn crc24_table_entry(poly: u32, index: u8) -> u32 {
  let mut crc = index as u32;
  let mut i = 0;
  while i < 8 {
    if crc & 1 != 0 {
      crc = (crc >> 1) ^ poly;
    } else {
      crc >>= 1;
    }
    i += 1;
  }
  crc & 0x00FF_FFFF // Ensure only 24 bits
}

/// Generate 4 CRC-24 lookup tables for slice-by-4 computation.
///
/// # Arguments
///
/// * `poly` - The reflected polynomial (low 24 bits)
#[cfg(test)] // Will be used when CRC-24 module is added
#[must_use]
pub const fn generate_crc24_tables_4(poly: u32) -> [[u32; 256]; 4] {
  let mut tables = [[0u32; 256]; 4];

  let mut i = 0u16;
  while i < 256 {
    tables[0][i as usize] = crc24_table_entry(poly, i as u8);
    i += 1;
  }

  let mut k = 1usize;
  while k < 4 {
    i = 0;
    while i < 256 {
      let prev = tables[k - 1][i as usize];
      tables[k][i as usize] = (tables[0][(prev & 0xFF) as usize] ^ (prev >> 8)) & 0x00FF_FFFF;
      i += 1;
    }
    k += 1;
  }

  tables
}

/// Generate 8 CRC-24 lookup tables for slice-by-8 computation.
///
/// # Arguments
///
/// * `poly` - The reflected polynomial (low 24 bits)
#[cfg(test)] // Will be used when CRC-24 module is added
#[must_use]
pub const fn generate_crc24_tables_8(poly: u32) -> [[u32; 256]; 8] {
  let mut tables = [[0u32; 256]; 8];

  let mut i = 0u16;
  while i < 256 {
    tables[0][i as usize] = crc24_table_entry(poly, i as u8);
    i += 1;
  }

  let mut k = 1usize;
  while k < 8 {
    i = 0;
    while i < 256 {
      let prev = tables[k - 1][i as usize];
      tables[k][i as usize] = (tables[0][(prev & 0xFF) as usize] ^ (prev >> 8)) & 0x00FF_FFFF;
      i += 1;
    }
    k += 1;
  }

  tables
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-64 Table Generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a single CRC-64 lookup table entry.
///
/// Uses bit-by-bit computation with the reflected polynomial.
#[must_use]
pub const fn crc64_table_entry(poly: u64, index: u8) -> u64 {
  let mut crc = index as u64;
  let mut i = 0;
  while i < 8 {
    if crc & 1 != 0 {
      crc = (crc >> 1) ^ poly;
    } else {
      crc >>= 1;
    }
    i += 1;
  }
  crc
}

/// Generate 8 CRC-64 lookup tables for slice-by-8 computation.
///
/// # Arguments
///
/// * `poly` - The reflected polynomial
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", test))]
#[must_use]
pub const fn generate_crc64_tables_8(poly: u64) -> [[u64; 256]; 8] {
  let mut tables = [[0u64; 256]; 8];

  let mut i = 0u16;
  while i < 256 {
    tables[0][i as usize] = crc64_table_entry(poly, i as u8);
    i += 1;
  }

  let mut k = 1usize;
  while k < 8 {
    i = 0;
    while i < 256 {
      let prev = tables[k - 1][i as usize];
      tables[k][i as usize] = tables[0][(prev & 0xFF) as usize] ^ (prev >> 8);
      i += 1;
    }
    k += 1;
  }

  tables
}

/// Generate 16 CRC-64 lookup tables for slice-by-16 computation.
///
/// This is an extension of the slice-by-8 strategy that processes 16 bytes per
/// iteration. It can improve throughput on targets without SIMD acceleration.
///
/// # Arguments
///
/// * `poly` - The reflected polynomial
#[must_use]
pub const fn generate_crc64_tables_16(poly: u64) -> [[u64; 256]; 16] {
  let mut tables = [[0u64; 256]; 16];

  let mut i = 0u16;
  while i < 256 {
    tables[0][i as usize] = crc64_table_entry(poly, i as u8);
    i += 1;
  }

  let mut k = 1usize;
  while k < 16 {
    i = 0;
    while i < 256 {
      let prev = tables[k - 1][i as usize];
      tables[k][i as usize] = tables[0][(prev & 0xFF) as usize] ^ (prev >> 8);
      i += 1;
    }
    k += 1;
  }

  tables
}

// ─────────────────────────────────────────────────────────────────────────────
// Polynomial Constants (Reflected Form)
// ─────────────────────────────────────────────────────────────────────────────

// CRC-16 Polynomials

/// CRC-16-CCITT polynomial (0x1021) in reflected form.
/// Used by X.25, V.41, HDLC, XMODEM, Bluetooth, PACTOR, SD, etc.
#[cfg(test)] // Will be used when CRC-16 module is added
pub const CRC16_CCITT_POLY: u16 = 0x8408;

/// CRC-16-IBM polynomial (0x8005) in reflected form.
/// Used by Modbus, USB, ANSI X3.28, etc.
#[cfg(test)] // Will be used when CRC-16 module is added
#[allow(dead_code)]
pub const CRC16_IBM_POLY: u16 = 0xA001;

// CRC-24 Polynomials

/// CRC-24-OPENPGP polynomial (0x864CFB) in reflected form.
/// Used by OpenPGP (RFC 4880).
#[cfg(test)] // Will be used when CRC-24 module is added
pub const CRC24_OPENPGP_POLY: u32 = 0x00DF_3261;

// CRC-64 Polynomials

/// CRC-64-XZ polynomial (0x42F0E1EBA9EA3693) in reflected form.
/// Used by XZ Utils, 7-Zip, LZMA.
pub const CRC64_XZ_POLY: u64 = 0xC96C_5795_D787_0F42;

/// CRC-64-NVME polynomial (0xAD93D23594C93659) in reflected form.
/// Used by NVMe specification.
pub const CRC64_NVME_POLY: u64 = 0x9A6C_9329_AC4B_C9B5;

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
  fn test_crc16_tables_4_consistency() {
    let tables = generate_crc16_tables_4(CRC16_CCITT_POLY);

    assert_eq!(tables[0][0], 0);
    assert_ne!(tables[0][1], 0);

    for k in 1..4 {
      for i in 0..256 {
        let prev = tables[k - 1][i];
        let expected = tables[0][(prev & 0xFF) as usize] ^ (prev >> 8);
        assert_eq!(tables[k][i], expected);
      }
    }
  }

  #[test]
  fn test_crc16_tables_8_consistency() {
    let tables4 = generate_crc16_tables_4(CRC16_CCITT_POLY);
    let tables8 = generate_crc16_tables_8(CRC16_CCITT_POLY);

    // First 4 tables must match
    for k in 0..4 {
      assert_eq!(tables8[k], tables4[k]);
    }

    for k in 1..8 {
      for i in 0..256 {
        let prev = tables8[k - 1][i];
        let expected = tables8[0][(prev & 0xFF) as usize] ^ (prev >> 8);
        assert_eq!(tables8[k][i], expected);
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-24 Tests
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_crc24_tables_4_consistency() {
    let tables = generate_crc24_tables_4(CRC24_OPENPGP_POLY);

    assert_eq!(tables[0][0], 0);
    assert_ne!(tables[0][1], 0);

    for k in 1..4 {
      for i in 0..256 {
        let prev = tables[k - 1][i];
        let expected = (tables[0][(prev & 0xFF) as usize] ^ (prev >> 8)) & 0x00FF_FFFF;
        assert_eq!(tables[k][i], expected);
        // Verify 24-bit constraint
        assert_eq!(tables[k][i] & 0xFF00_0000, 0);
      }
    }
  }

  #[test]
  fn test_crc24_tables_8_consistency() {
    let tables4 = generate_crc24_tables_4(CRC24_OPENPGP_POLY);
    let tables8 = generate_crc24_tables_8(CRC24_OPENPGP_POLY);

    // First 4 tables must match
    for k in 0..4 {
      assert_eq!(tables8[k], tables4[k]);
    }

    for k in 1..8 {
      for i in 0..256 {
        let prev = tables8[k - 1][i];
        let expected = (tables8[0][(prev & 0xFF) as usize] ^ (prev >> 8)) & 0x00FF_FFFF;
        assert_eq!(tables8[k][i], expected);
        // Verify 24-bit constraint
        assert_eq!(tables8[k][i] & 0xFF00_0000, 0);
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-64 Tests
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_crc64_tables_8_consistency() {
    let tables = generate_crc64_tables_8(CRC64_XZ_POLY);

    assert_eq!(tables[0][0], 0);
    assert_ne!(tables[0][1], 0);

    for k in 1..8 {
      for i in 0..256 {
        let prev = tables[k - 1][i];
        let expected = tables[0][(prev & 0xFF) as usize] ^ (prev >> 8);
        assert_eq!(tables[k][i], expected);
      }
    }
  }

  #[test]
  fn test_crc64_tables_16_consistency() {
    let tables8 = generate_crc64_tables_8(CRC64_XZ_POLY);
    let tables16 = generate_crc64_tables_16(CRC64_XZ_POLY);

    assert_eq!(tables16[0][0], 0);
    assert_ne!(tables16[0][1], 0);

    // First 8 tables must match
    for k in 0..8 {
      assert_eq!(tables16[k], tables8[k]);
    }

    for k in 1..16 {
      for i in 0..256 {
        let prev = tables16[k - 1][i];
        let expected = tables16[0][(prev & 0xFF) as usize] ^ (prev >> 8);
        assert_eq!(tables16[k][i], expected);
      }
    }
  }

  #[test]
  fn test_crc64_nvme_tables_consistency() {
    // Verify CRC-64-NVME uses a different polynomial
    let xz = generate_crc64_tables_8(CRC64_XZ_POLY);
    let nvme = generate_crc64_tables_8(CRC64_NVME_POLY);

    // Tables should be different
    assert_ne!(xz[0], nvme[0]);
  }
}
