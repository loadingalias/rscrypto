//! Portable CRC implementations using lookup table algorithms.
//!
//! This module provides generic slice-by-N implementations for all CRC widths:
//! - CRC-16: slice-by-4, slice-by-8
//! - CRC-64: slice-by-8, slice-by-16
//!
//! # Algorithm Overview
//!
//! Slice-by-N processes N bytes per iteration using N precomputed lookup tables.
//! Each table contains 256 entries representing the CRC contribution of a single
//! byte at a specific position in the input stream.
//!
//! The algorithm XORs the current CRC with N input bytes, then combines
//! N table lookups (one per byte position) using XOR. This achieves ~N×
//! throughput compared to byte-at-a-time processing.
//!
//! # Performance Characteristics
//!
//! | Width | Algorithm | Bytes/iter | Tables | Throughput |
//! |-------|-----------|------------|--------|------------|
//! | 16-bit | slice-by-4 | 4 | 4×256×u16 | ~1.5 GB/s |
//! | 16-bit | slice-by-8 | 8 | 8×256×u16 | ~2.5 GB/s |
//! | 64-bit | slice-by-8 | 8 | 8×256×u64 | ~2.0 GB/s |
//! | 64-bit | slice-by-16 | 16 | 16×256×u64 | ~3.0 GB/s |

// SAFETY: All array indexing in this module uses bounded indices:
// - chunks_exact guarantees chunk sizes
// - Table indices use `& 0xFF` (0..255) or explicit byte extraction
// Clippy cannot prove this in const fn contexts, but bounds are statically guaranteed.
#![allow(clippy::indexing_slicing)]

// ─────────────────────────────────────────────────────────────────────────────
// CRC-16 Portable Implementations
// ─────────────────────────────────────────────────────────────────────────────

/// Update CRC-16 state using slice-by-4 algorithm.
///
/// Processes 4 bytes per iteration (2× the CRC width in bytes).
///
/// # Arguments
///
/// * `crc` - Current CRC state (pre-inverted if applicable)
/// * `data` - Input data
/// * `tables` - 4 lookup tables (256 entries each)
#[cfg(test)] // Will be used when CRC-16 module is added
#[inline]
pub(crate) fn slice4_16(mut crc: u16, data: &[u8], tables: &[[u16; 256]; 4]) -> u16 {
  let (chunks, remainder) = data.as_chunks::<4>();

  for chunk in chunks {
    // Read 4 bytes: first 2 XOR with CRC, next 2 are separate
    let a = u16::from_le_bytes([chunk[0], chunk[1]]) ^ crc;
    let b = u16::from_le_bytes([chunk[2], chunk[3]]);

    crc = tables[3][(a & 0xFF) as usize]
      ^ tables[2][((a >> 8) & 0xFF) as usize]
      ^ tables[1][(b & 0xFF) as usize]
      ^ tables[0][((b >> 8) & 0xFF) as usize];
  }

  // Process remaining bytes with byte-at-a-time
  for &byte in remainder {
    let index = ((crc ^ (byte as u16)) & 0xFF) as usize;
    crc = tables[0][index] ^ (crc >> 8);
  }

  crc
}

/// Update CRC-16 state using slice-by-8 algorithm.
///
/// Processes 8 bytes per iteration (4× the CRC width in bytes).
///
/// # Arguments
///
/// * `crc` - Current CRC state (pre-inverted if applicable)
/// * `data` - Input data
/// * `tables` - 8 lookup tables (256 entries each)
#[cfg(test)] // Will be used when CRC-16 module is added
#[inline]
pub(crate) fn slice8_16(mut crc: u16, data: &[u8], tables: &[[u16; 256]; 8]) -> u16 {
  let (chunks, remainder) = data.as_chunks::<8>();

  for chunk in chunks {
    // Read 8 bytes as 4 u16 values
    let a = u16::from_le_bytes([chunk[0], chunk[1]]) ^ crc;
    let b = u16::from_le_bytes([chunk[2], chunk[3]]);
    let c = u16::from_le_bytes([chunk[4], chunk[5]]);
    let d = u16::from_le_bytes([chunk[6], chunk[7]]);

    crc = tables[7][(a & 0xFF) as usize]
      ^ tables[6][((a >> 8) & 0xFF) as usize]
      ^ tables[5][(b & 0xFF) as usize]
      ^ tables[4][((b >> 8) & 0xFF) as usize]
      ^ tables[3][(c & 0xFF) as usize]
      ^ tables[2][((c >> 8) & 0xFF) as usize]
      ^ tables[1][(d & 0xFF) as usize]
      ^ tables[0][((d >> 8) & 0xFF) as usize];
  }

  // Process remaining bytes with byte-at-a-time
  for &byte in remainder {
    let index = ((crc ^ (byte as u16)) & 0xFF) as usize;
    crc = tables[0][index] ^ (crc >> 8);
  }

  crc
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-64 Portable Implementations
// ─────────────────────────────────────────────────────────────────────────────

/// Update CRC-64 state using slice-by-8 algorithm.
///
/// Processes 8 bytes per iteration (1× the CRC width in bytes).
///
/// # Arguments
///
/// * `crc` - Current CRC state (pre-inverted)
/// * `data` - Input data
/// * `tables` - 8 lookup tables (256 entries each)
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", test))]
#[inline]
pub fn slice8_64(mut crc: u64, data: &[u8], tables: &[[u64; 256]; 8]) -> u64 {
  let (chunks, remainder) = data.as_chunks::<8>();

  for chunk in chunks {
    let val = u64::from_le_bytes(*chunk) ^ crc;

    crc = tables[7][(val & 0xFF) as usize]
      ^ tables[6][((val >> 8) & 0xFF) as usize]
      ^ tables[5][((val >> 16) & 0xFF) as usize]
      ^ tables[4][((val >> 24) & 0xFF) as usize]
      ^ tables[3][((val >> 32) & 0xFF) as usize]
      ^ tables[2][((val >> 40) & 0xFF) as usize]
      ^ tables[1][((val >> 48) & 0xFF) as usize]
      ^ tables[0][(val >> 56) as usize];
  }

  // Process remaining bytes (0-7) with byte-at-a-time
  for &byte in remainder {
    let index = ((crc ^ (byte as u64)) & 0xFF) as usize;
    crc = tables[0][index] ^ (crc >> 8);
  }

  crc
}

/// Update CRC-64 state using slice-by-16 algorithm.
///
/// Processes 16 bytes per iteration (2× the CRC width in bytes).
/// Optimal for larger buffers where cache is warm.
///
/// # Arguments
///
/// * `crc` - Current CRC state (pre-inverted)
/// * `data` - Input data
/// * `tables` - 16 lookup tables (256 entries each)
#[inline]
pub fn slice16_64(mut crc: u64, data: &[u8], tables: &[[u64; 256]; 16]) -> u64 {
  let (chunks8, remainder) = data.as_chunks::<8>();
  let mut pairs = chunks8.chunks_exact(2);

  for pair in pairs.by_ref() {
    let a = u64::from_le_bytes(pair[0]) ^ crc;
    let b = u64::from_le_bytes(pair[1]);

    crc = tables[15][(a & 0xFF) as usize]
      ^ tables[14][((a >> 8) & 0xFF) as usize]
      ^ tables[13][((a >> 16) & 0xFF) as usize]
      ^ tables[12][((a >> 24) & 0xFF) as usize]
      ^ tables[11][((a >> 32) & 0xFF) as usize]
      ^ tables[10][((a >> 40) & 0xFF) as usize]
      ^ tables[9][((a >> 48) & 0xFF) as usize]
      ^ tables[8][(a >> 56) as usize]
      ^ tables[7][(b & 0xFF) as usize]
      ^ tables[6][((b >> 8) & 0xFF) as usize]
      ^ tables[5][((b >> 16) & 0xFF) as usize]
      ^ tables[4][((b >> 24) & 0xFF) as usize]
      ^ tables[3][((b >> 32) & 0xFF) as usize]
      ^ tables[2][((b >> 40) & 0xFF) as usize]
      ^ tables[1][((b >> 48) & 0xFF) as usize]
      ^ tables[0][(b >> 56) as usize];
  }

  // Handle an odd 8-byte tail
  if let [chunk] = pairs.remainder() {
    let val = u64::from_le_bytes(*chunk) ^ crc;
    crc = tables[7][(val & 0xFF) as usize]
      ^ tables[6][((val >> 8) & 0xFF) as usize]
      ^ tables[5][((val >> 16) & 0xFF) as usize]
      ^ tables[4][((val >> 24) & 0xFF) as usize]
      ^ tables[3][((val >> 32) & 0xFF) as usize]
      ^ tables[2][((val >> 40) & 0xFF) as usize]
      ^ tables[1][((val >> 48) & 0xFF) as usize]
      ^ tables[0][(val >> 56) as usize];
  }

  // Process remaining bytes (< 8) with byte-at-a-time
  for &byte in remainder {
    let index = ((crc ^ (byte as u64)) & 0xFF) as usize;
    crc = tables[0][index] ^ (crc >> 8);
  }

  crc
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16 Tests
  // ─────────────────────────────────────────────────────────────────────────

  /// Generate CRC-16 tables for testing (CCITT polynomial 0x8408 reflected).
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

  const fn generate_crc16_tables_4(poly: u16) -> [[u16; 256]; 4] {
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

  const fn generate_crc16_tables_8(poly: u16) -> [[u16; 256]; 8] {
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

  const CRC16_CCITT_POLY: u16 = 0x8408; // Reflected

  #[test]
  fn test_slice4_16_empty() {
    let tables = generate_crc16_tables_4(CRC16_CCITT_POLY);
    assert_eq!(slice4_16(!0, &[], &tables), !0);
  }

  #[test]
  fn test_slice8_16_empty() {
    let tables = generate_crc16_tables_8(CRC16_CCITT_POLY);
    assert_eq!(slice8_16(!0, &[], &tables), !0);
  }

  #[test]
  fn test_slice4_16_matches_slice8_16() {
    let tables4 = generate_crc16_tables_4(CRC16_CCITT_POLY);
    let tables8 = generate_crc16_tables_8(CRC16_CCITT_POLY);
    let data = b"The quick brown fox jumps over the lazy dog";

    let a = slice4_16(!0, data, &tables4);
    let b = slice8_16(!0, data, &tables8);
    assert_eq!(a, b);
  }

  #[test]
  fn test_slice4_16_incremental() {
    let tables = generate_crc16_tables_4(CRC16_CCITT_POLY);
    let data = b"hello world, this is a test";
    let full = slice4_16(!0, data, &tables);

    for split in [1, 3, 4, 5, 7, 8, 10, 15] {
      if split < data.len() {
        let crc1 = slice4_16(!0, &data[..split], &tables);
        let crc2 = slice4_16(crc1, &data[split..], &tables);
        assert_eq!(crc2, full, "Incremental failed at split {split}");
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-64 Tests
  // ─────────────────────────────────────────────────────────────────────────

  /// Generate CRC-64 tables for testing.
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

  const fn generate_crc64_tables_8(poly: u64) -> [[u64; 256]; 8] {
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

  const fn generate_crc64_tables_16(poly: u64) -> [[u64; 256]; 16] {
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

  const CRC64_XZ_POLY: u64 = 0xC96C_5795_D787_0F42; // Reflected

  #[test]
  fn test_slice8_64_empty() {
    let tables = generate_crc64_tables_8(CRC64_XZ_POLY);
    assert_eq!(slice8_64(!0, &[], &tables), !0);
  }

  #[test]
  fn test_slice16_64_empty() {
    let tables = generate_crc64_tables_16(CRC64_XZ_POLY);
    assert_eq!(slice16_64(!0, &[], &tables), !0);
  }

  #[test]
  fn test_slice8_64_matches_slice16_64() {
    let tables8 = generate_crc64_tables_8(CRC64_XZ_POLY);
    let tables16 = generate_crc64_tables_16(CRC64_XZ_POLY);
    let data = b"The quick brown fox jumps over the lazy dog";

    let a = slice8_64(!0, data, &tables8);
    let b = slice16_64(!0, data, &tables16);
    assert_eq!(a, b);
  }

  #[test]
  fn test_slice16_64_incremental() {
    let tables = generate_crc64_tables_16(CRC64_XZ_POLY);
    let data = b"hello world, this is a longer test string";
    let full = slice16_64(!0, data, &tables);

    for split in [1, 7, 8, 9, 15, 16, 17, 20] {
      if split < data.len() {
        let crc1 = slice16_64(!0, &data[..split], &tables);
        let crc2 = slice16_64(crc1, &data[split..], &tables);
        assert_eq!(crc2, full, "Incremental failed at split {split}");
      }
    }
  }

  #[test]
  fn test_crc64_xz_test_vector() {
    // "123456789" should produce 0x995DC9BBDF1939FA for CRC-64-XZ
    let tables = generate_crc64_tables_16(CRC64_XZ_POLY);
    let crc = slice16_64(!0, b"123456789", &tables) ^ !0;
    assert_eq!(crc, 0x995D_C9BB_DF19_39FA);
  }
}
