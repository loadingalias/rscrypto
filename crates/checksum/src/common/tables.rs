//! Const-fn CRC lookup table generation.
//!
//! This module provides compile-time table generation for all CRC sizes.
//! Tables are computed using `const fn` and embedded directly in the binary.
//!
//! # Table Strategies
//!
//! - **CRC-16/CRC-24**: Single 256-entry table (byte-at-a-time)
//! - **CRC-32/CRC-64**: 8×256-entry tables (slice-by-8), optionally 16×256 (slice-by-16)

// SAFETY: All array indexing in this module uses bounded loop indices (0..256, 0..8).
// Clippy cannot prove this in const fn contexts, but bounds are statically guaranteed.
#![allow(clippy::indexing_slicing)]

// ─────────────────────────────────────────────────────────────────────────────
// CRC-16 Table Generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a single CRC-16 lookup table entry.
///
/// Uses bit-by-bit computation with the reflected polynomial.
const fn crc16_table_entry(poly: u16, index: u8) -> u16 {
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

/// Generate a 256-entry CRC-16 lookup table.
///
/// # Arguments
///
/// * `poly` - The reflected polynomial (e.g., 0x8408 for CCITT, 0xA001 for IBM)
#[must_use]
pub const fn generate_crc16_table(poly: u16) -> [u16; 256] {
  let mut table = [0u16; 256];
  let mut i = 0u16;
  while i < 256 {
    table[i as usize] = crc16_table_entry(poly, i as u8);
    i += 1;
  }
  table
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-24 Table Generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a single CRC-24 lookup table entry.
///
/// CRC-24 is computed MSB-first (non-reflected).
const fn crc24_table_entry(poly: u32, index: u8) -> u32 {
  let mut crc = (index as u32) << 16;
  let mut i = 0;
  while i < 8 {
    if crc & 0x80_0000 != 0 {
      crc = (crc << 1) ^ poly;
    } else {
      crc <<= 1;
    }
    i += 1;
  }
  crc & 0xFF_FFFF
}

/// Generate a 256-entry CRC-24 lookup table.
///
/// # Arguments
///
/// * `poly` - The polynomial (e.g., 0x864CFB for OpenPGP)
#[must_use]
pub const fn generate_crc24_table(poly: u32) -> [u32; 256] {
  let mut table = [0u32; 256];
  let mut i = 0u16;
  while i < 256 {
    table[i as usize] = crc24_table_entry(poly, i as u8);
    i += 1;
  }
  table
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32 Table Generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a single CRC-32 lookup table entry.
///
/// Uses bit-by-bit computation with the reflected polynomial.
const fn crc32_table_entry(poly: u32, index: u8) -> u32 {
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
  crc
}

/// Generate 8 CRC-32 lookup tables for slice-by-8 computation.
///
/// The first table (index 0) is the standard byte-at-a-time table.
/// Tables 1-7 enable processing 8 bytes in parallel with reduced memory access.
///
/// # Arguments
///
/// * `poly` - The reflected polynomial
///
/// # Note
///
/// Production code uses `generate_crc32_tables_16` for better throughput.
/// This function is retained for testing slice-by-8 as a correctness baseline.
#[cfg(test)]
#[must_use]
pub const fn generate_crc32_tables_8(poly: u32) -> [[u32; 256]; 8] {
  let mut tables = [[0u32; 256]; 8];

  // Generate table 0 (standard byte-at-a-time)
  let mut i = 0u16;
  while i < 256 {
    tables[0][i as usize] = crc32_table_entry(poly, i as u8);
    i += 1;
  }

  // Generate tables 1-7 using the recurrence relation:
  // table[k][i] = table[0][table[k-1][i] & 0xFF] ^ (table[k-1][i] >> 8)
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

/// Generate 16 CRC-32 lookup tables for slice-by-16 computation.
///
/// This is an extension of the slice-by-8 strategy that processes 16 bytes per
/// iteration. It can improve throughput on targets without SIMD acceleration
/// by approximately 1.5-2x compared to slice-by-8.
///
/// # Arguments
///
/// * `poly` - The reflected polynomial
#[must_use]
pub const fn generate_crc32_tables_16(poly: u32) -> [[u32; 256]; 16] {
  let mut tables = [[0u32; 256]; 16];

  // Generate table 0 (standard byte-at-a-time)
  let mut i = 0u16;
  while i < 256 {
    tables[0][i as usize] = crc32_table_entry(poly, i as u8);
    i += 1;
  }

  // Generate tables 1-15 using the recurrence relation
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
// CRC-64 Table Generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a single CRC-64 lookup table entry.
///
/// Uses bit-by-bit computation with the reflected polynomial.
const fn crc64_table_entry(poly: u64, index: u8) -> u64 {
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

  // Generate table 0 (standard byte-at-a-time)
  let mut i = 0u16;
  while i < 256 {
    tables[0][i as usize] = crc64_table_entry(poly, i as u8);
    i += 1;
  }

  // Generate tables 1-7 using the recurrence relation
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

  // Generate table 0 (standard byte-at-a-time)
  let mut i = 0u16;
  while i < 256 {
    tables[0][i as usize] = crc64_table_entry(poly, i as u8);
    i += 1;
  }

  // Generate tables 1-15 using the recurrence relation
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

/// CRC-16-CCITT polynomial (0x1021) in reflected form.
pub const CRC16_CCITT_POLY: u16 = 0x8408;

/// CRC-16-IBM/ANSI polynomial (0x8005) in reflected form.
pub const CRC16_IBM_POLY: u16 = 0xA001;

/// CRC-24 OpenPGP polynomial (0x864CFB) - NOT reflected (MSB-first).
pub const CRC24_OPENPGP_POLY: u32 = 0x86_4CFB;

/// CRC-32 IEEE polynomial (0x04C11DB7) in reflected form.
pub const CRC32_IEEE_POLY: u32 = 0xEDB8_8320;

/// CRC-32C Castagnoli polynomial (0x1EDC6F41) in reflected form.
pub const CRC32C_POLY: u32 = 0x82F6_3B78;

/// CRC-64-XZ polynomial (0x42F0E1EBA9EA3693) in reflected form.
pub const CRC64_XZ_POLY: u64 = 0xC96C_5795_D787_0F42;

/// CRC-64-NVME polynomial (0xAD93D23594C93659) in reflected form.
pub const CRC64_NVME_POLY: u64 = 0x9A6C_9329_AC4B_C9B5;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_crc16_table_generation() {
    let table = generate_crc16_table(CRC16_CCITT_POLY);
    // Verify table is non-trivial (not all zeros)
    assert_ne!(table[1], 0);
    // Table[0] should be 0 (CRC of 0x00 is 0)
    assert_eq!(table[0], 0);
    // Table[255] should be non-zero
    assert_ne!(table[255], 0);
  }

  #[test]
  fn test_crc24_table_generation() {
    let table = generate_crc24_table(CRC24_OPENPGP_POLY);
    assert_ne!(table[1], 0);
    // All entries should be 24-bit values
    for &entry in &table {
      assert!(entry <= 0xFF_FFFF);
    }
  }

  #[test]
  fn test_crc32_table_generation() {
    let tables = generate_crc32_tables_8(CRC32_IEEE_POLY);
    // Table 0 is the standard byte-at-a-time table
    // Known value: CRC-32 table[1] for IEEE polynomial
    // This is the CRC of a single 0x01 byte XORed into CRC 0
    assert_eq!(tables[0][0], 0);
    assert_ne!(tables[0][1], 0);
  }

  #[test]
  fn test_crc32_tables_8_consistency() {
    let tables = generate_crc32_tables_8(CRC32_IEEE_POLY);

    // Verify recurrence: table[k][i] = table[0][table[k-1][i] & 0xFF] ^ (table[k-1][i] >> 8)
    for k in 1..8 {
      for i in 0..256 {
        let prev = tables[k - 1][i];
        let expected = tables[0][(prev & 0xFF) as usize] ^ (prev >> 8);
        assert_eq!(tables[k][i], expected);
      }
    }
  }

  #[test]
  fn test_crc32_tables_16_consistency() {
    let tables8 = generate_crc32_tables_8(CRC32_IEEE_POLY);
    let tables16 = generate_crc32_tables_16(CRC32_IEEE_POLY);

    // Table 0 is the standard byte-at-a-time table
    assert_eq!(tables16[0][0], 0);
    assert_ne!(tables16[0][1], 0);

    // First 8 tables must match the slice-by-8 generation
    for k in 0..8 {
      assert_eq!(tables16[k], tables8[k]);
    }

    // Verify recurrence for all 16 tables
    for k in 1..16 {
      for i in 0..256 {
        let prev = tables16[k - 1][i];
        let expected = tables16[0][(prev & 0xFF) as usize] ^ (prev >> 8);
        assert_eq!(tables16[k][i], expected);
      }
    }
  }

  #[test]
  fn test_crc64_tables_8_consistency() {
    let tables = generate_crc64_tables_8(CRC64_XZ_POLY);

    // Table 0 is the standard byte-at-a-time table
    assert_eq!(tables[0][0], 0);
    assert_ne!(tables[0][1], 0);

    // Verify recurrence: table[k][i] = table[0][table[k-1][i] & 0xFF] ^ (table[k-1][i] >> 8)
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

    // Table 0 is the standard byte-at-a-time table
    assert_eq!(tables16[0][0], 0);
    assert_ne!(tables16[0][1], 0);

    // First 8 tables must match the slice-by-8 generation.
    for k in 0..8 {
      assert_eq!(tables16[k], tables8[k]);
    }

    // Verify recurrence: table[k][i] = table[0][table[k-1][i] & 0xFF] ^ (table[k-1][i] >> 8)
    for k in 1..16 {
      for i in 0..256 {
        let prev = tables16[k - 1][i];
        let expected = tables16[0][(prev & 0xFF) as usize] ^ (prev >> 8);
        assert_eq!(tables16[k][i], expected);
      }
    }
  }
}
