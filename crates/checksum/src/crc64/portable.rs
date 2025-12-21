//! Portable CRC-64 implementation using slice-by-8.

// SAFETY: chunks_exact(8) guarantees 8-byte chunks; table indices use `& 0xFF` (0..255).
#![allow(clippy::indexing_slicing)]

use super::kernel_tables;

// ─────────────────────────────────────────────────────────────────────────────
// Polynomial-specific wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-64-XZ slice-by-8 computation.
#[inline]
pub fn crc64_slice8_xz(crc: u64, data: &[u8]) -> u64 {
  crc64_slice8(crc, data, &kernel_tables::XZ_TABLES)
}

/// CRC-64-NVME slice-by-8 computation.
#[inline]
pub fn crc64_slice8_nvme(crc: u64, data: &[u8]) -> u64 {
  crc64_slice8(crc, data, &kernel_tables::NVME_TABLES)
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic slice-by-8
// ─────────────────────────────────────────────────────────────────────────────

/// Update CRC-64 state using slice-by-8 algorithm.
///
/// # Arguments
///
/// * `crc` - Current CRC state (pre-inverted)
/// * `data` - Input data
/// * `tables` - 8 lookup tables (256 entries each)
#[inline]
pub fn crc64_slice8(mut crc: u64, data: &[u8], tables: &[[u64; 256]; 8]) -> u64 {
  let mut chunks = data.chunks_exact(8);

  for chunk in chunks.by_ref() {
    // Read 8 bytes as u64 (little-endian) and XOR with current CRC.
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(chunk);
    let val = u64::from_le_bytes(bytes) ^ crc;

    // Lookup all 8 bytes using 8 different tables
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
  for &byte in chunks.remainder() {
    let index = ((crc ^ (byte as u64)) & 0xFF) as usize;
    crc = tables[0][index] ^ (crc >> 8);
  }

  crc
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::common::tables::{CRC64_XZ_POLY, generate_crc64_tables_8};

  #[test]
  fn test_slice8_empty() {
    let tables = generate_crc64_tables_8(CRC64_XZ_POLY);
    let crc = crc64_slice8(!0, &[], &tables);
    assert_eq!(crc, !0);
  }

  #[test]
  fn test_slice8_consistency_with_byte_at_a_time() {
    let tables = generate_crc64_tables_8(CRC64_XZ_POLY);
    let data = b"The quick brown fox jumps over the lazy dog";

    // Compute with slice-by-8
    let slice8_result = crc64_slice8(!0, data, &tables);

    // Compute byte-at-a-time using table[0]
    let mut byte_result = !0u64;
    for &b in data.iter() {
      let index = ((byte_result ^ (b as u64)) & 0xFF) as usize;
      byte_result = tables[0][index] ^ (byte_result >> 8);
    }

    assert_eq!(slice8_result, byte_result);
  }

  #[test]
  fn test_slice8_incremental() {
    let tables = generate_crc64_tables_8(CRC64_XZ_POLY);
    let data = b"hello world, this is a longer test string";

    let full = crc64_slice8(!0, data, &tables);

    for split in [1, 7, 8, 9, 15, 16, 17, 20] {
      if split < data.len() {
        let crc1 = crc64_slice8(!0, &data[..split], &tables);
        let crc2 = crc64_slice8(crc1, &data[split..], &tables);
        assert_eq!(crc2, full, "Incremental failed at split {split}");
      }
    }
  }
}
