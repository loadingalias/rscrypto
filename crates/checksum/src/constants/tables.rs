//! Generic slicing-by-8 lookup table generation.
//!
//! This module provides const-time table generation for CRC algorithms
//! of various widths (16, 32, 64 bits). The slicing-by-8 technique
//! processes 8 bytes at a time, achieving ~4x speedup over byte-at-a-time.
//!
//! # Table Structure
//!
//! Each CRC variant uses 8 tables of 256 entries:
//! - Table 0: CRC contribution of each byte value
//! - Tables 1-7: CRC contribution of bytes at positions 1-7 earlier in stream
//!
//! Total sizes:
//! - CRC16: 8 × 256 × 2 = 4KB
//! - CRC32: 8 × 256 × 4 = 8KB
//! - CRC64: 8 × 256 × 8 = 16KB

// ============================================================================
// 16-bit CRC Tables (CRC16/IBM, etc.)
// ============================================================================

/// Generate base CRC16 lookup table (table 0) for a reflected polynomial.
#[cfg(not(feature = "no-tables"))]
pub const fn generate_table_0_16(poly: u16) -> [u16; 256] {
  let mut table = [0u16; 256];
  let mut i = 0usize;

  while i < 256 {
    let mut crc = i as u16;
    let mut j = 0;
    while j < 8 {
      if crc & 1 != 0 {
        crc = (crc >> 1) ^ poly;
      } else {
        crc >>= 1;
      }
      j += 1;
    }
    table[i] = crc;
    i += 1;
  }

  table
}

/// Generate all 8 slicing-by-8 tables for 16-bit CRC.
///
/// Tables 1-7 are derived by applying the CRC transform repeatedly.
/// This enables processing 8 bytes in parallel using XOR chains.
#[cfg(not(feature = "no-tables"))]
pub const fn generate_slicing_tables_16(poly: u16) -> [[u16; 256]; 8] {
  let table0 = generate_table_0_16(poly);
  let mut tables = [[0u16; 256]; 8];

  // Copy table 0
  let mut i = 0;
  while i < 256 {
    tables[0][i] = table0[i];
    i += 1;
  }

  // Generate tables 1-7
  let mut t = 1;
  while t < 8 {
    let mut i = 0;
    while i < 256 {
      let prev = tables[t - 1][i];
      tables[t][i] = (prev >> 8) ^ table0[(prev & 0xFF) as usize];
      i += 1;
    }
    t += 1;
  }

  tables
}

// ============================================================================
// 32-bit CRC Tables (CRC32, CRC32C)
// ============================================================================

/// Generate base CRC32 lookup table (table 0) for a reflected polynomial.
#[cfg(not(feature = "no-tables"))]
pub const fn generate_table_0_32(poly: u32) -> [u32; 256] {
  let mut table = [0u32; 256];
  let mut i = 0usize;

  while i < 256 {
    let mut crc = i as u32;
    let mut j = 0;
    while j < 8 {
      if crc & 1 != 0 {
        crc = (crc >> 1) ^ poly;
      } else {
        crc >>= 1;
      }
      j += 1;
    }
    table[i] = crc;
    i += 1;
  }

  table
}

/// Generate all 8 slicing-by-8 tables for 32-bit CRC.
#[cfg(not(feature = "no-tables"))]
pub const fn generate_slicing_tables_32(poly: u32) -> [[u32; 256]; 8] {
  let table0 = generate_table_0_32(poly);
  let mut tables = [[0u32; 256]; 8];

  let mut i = 0;
  while i < 256 {
    tables[0][i] = table0[i];
    i += 1;
  }

  let mut t = 1;
  while t < 8 {
    let mut i = 0;
    while i < 256 {
      let prev = tables[t - 1][i];
      tables[t][i] = (prev >> 8) ^ table0[(prev & 0xFF) as usize];
      i += 1;
    }
    t += 1;
  }

  tables
}

// ============================================================================
// 64-bit CRC Tables (CRC64/XZ, CRC64/NVME)
// ============================================================================

/// Generate base CRC64 lookup table (table 0) for a reflected polynomial.
#[cfg(not(feature = "no-tables"))]
pub const fn generate_table_0_64(poly: u64) -> [u64; 256] {
  let mut table = [0u64; 256];
  let mut i = 0usize;

  while i < 256 {
    let mut crc = i as u64;
    let mut j = 0;
    while j < 8 {
      if crc & 1 != 0 {
        crc = (crc >> 1) ^ poly;
      } else {
        crc >>= 1;
      }
      j += 1;
    }
    table[i] = crc;
    i += 1;
  }

  table
}

/// Generate all 8 slicing-by-8 tables for 64-bit CRC.
#[cfg(not(feature = "no-tables"))]
pub const fn generate_slicing_tables_64(poly: u64) -> [[u64; 256]; 8] {
  let table0 = generate_table_0_64(poly);
  let mut tables = [[0u64; 256]; 8];

  let mut i = 0;
  while i < 256 {
    tables[0][i] = table0[i];
    i += 1;
  }

  let mut t = 1;
  while t < 8 {
    let mut i = 0;
    while i < 256 {
      let prev = tables[t - 1][i];
      tables[t][i] = (prev >> 8) ^ table0[(prev & 0xFF) as usize];
      i += 1;
    }
    t += 1;
  }

  tables
}

#[cfg(test)]
mod tests {
  #[cfg(not(feature = "no-tables"))]
  use super::*;

  #[test]
  #[cfg(not(feature = "no-tables"))]
  fn test_table_0_32_crc32c() {
    // CRC32C polynomial (reflected)
    const POLY: u32 = 0x82F6_3B78;
    let table = generate_table_0_32(POLY);

    // table[0] should be 0 (CRC of 0x00)
    assert_eq!(table[0], 0);
    // table[1] is CRC of 0x01 - verified against reference
    assert_eq!(table[1], 0xF26B_8303);
    // table[255] - verified against reference
    assert_eq!(table[255], 0xAD7D_5351);
  }

  #[test]
  #[cfg(not(feature = "no-tables"))]
  fn test_slicing_tables_consistency_32() {
    const POLY: u32 = 0x82F6_3B78;
    let tables = generate_slicing_tables_32(POLY);

    // Verify tables are related correctly
    for t in 1..8 {
      for i in 0..256 {
        let prev = tables[t - 1][i];
        let expected = (prev >> 8) ^ tables[0][(prev & 0xFF) as usize];
        assert_eq!(tables[t][i], expected);
      }
    }
  }

  #[test]
  #[cfg(not(feature = "no-tables"))]
  fn test_table_0_64_crc64_xz() {
    // CRC64/XZ polynomial (reflected)
    const POLY: u64 = 0xC96C_5795_D787_0F42;
    let table = generate_table_0_64(POLY);

    // table[0] should be 0
    assert_eq!(table[0], 0);
    // Verify non-trivial entry
    assert_ne!(table[1], 0);
    assert_ne!(table[255], 0);
  }

  #[test]
  #[cfg(not(feature = "no-tables"))]
  fn test_table_0_16_crc16_ibm() {
    // CRC16/IBM polynomial (reflected)
    const POLY: u16 = 0xA001;
    let table = generate_table_0_16(POLY);

    // table[0] should be 0
    assert_eq!(table[0], 0);
    // Verify non-trivial entries
    assert_ne!(table[1], 0);
  }
}
