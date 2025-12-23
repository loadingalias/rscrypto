//! Portable CRC-32 implementation using slice-by-8 and slice-by-16.
//!
//! Slice-by-8 processes 8 bytes per iteration, achieving ~2 GB/s on modern CPUs.
//! Slice-by-16 processes 16 bytes per iteration for ~1.5-2x better throughput.
//! These are the fallbacks when hardware acceleration is unavailable.

// SAFETY: chunks_exact guarantees proper chunk sizes; table indices use `& 0xFF` (0..255).
#![allow(clippy::indexing_slicing)]

/// Update CRC-32 state using slice-by-8 algorithm.
///
/// Processes 8 bytes at a time for improved throughput compared to
/// byte-at-a-time table lookup.
///
/// # Arguments
///
/// * `crc` - Current CRC state (pre-inverted)
/// * `data` - Input data
/// * `tables` - 8 lookup tables (256 entries each)
#[inline]
#[cfg(test)]
pub fn crc32_slice8(mut crc: u32, data: &[u8], tables: &[[u32; 256]; 8]) -> u32 {
  let mut chunks = data.chunks_exact(8);

  for chunk in chunks.by_ref() {
    // Read 8 bytes as two u32s (little-endian)
    let lo = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    let hi = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);

    // XOR first 4 bytes with current CRC
    let val_lo = lo ^ crc;

    // Lookup all 8 bytes in parallel using 8 different tables
    crc = tables[7][(val_lo & 0xFF) as usize]
      ^ tables[6][((val_lo >> 8) & 0xFF) as usize]
      ^ tables[5][((val_lo >> 16) & 0xFF) as usize]
      ^ tables[4][(val_lo >> 24) as usize]
      ^ tables[3][(hi & 0xFF) as usize]
      ^ tables[2][((hi >> 8) & 0xFF) as usize]
      ^ tables[1][((hi >> 16) & 0xFF) as usize]
      ^ tables[0][(hi >> 24) as usize];
  }

  // Process remaining bytes (0-7) with byte-at-a-time
  for &byte in chunks.remainder() {
    let index = ((crc ^ (byte as u32)) & 0xFF) as usize;
    crc = tables[0][index] ^ (crc >> 8);
  }

  crc
}

/// Update CRC-32 state using slice-by-16 algorithm.
///
/// Processes 16 bytes at a time for improved throughput (~1.5-2x faster than
/// slice-by-8 on modern CPUs). Uses 16 lookup tables (16 KB total).
///
/// # Arguments
///
/// * `crc` - Current CRC state (pre-inverted)
/// * `data` - Input data
/// * `tables` - 16 lookup tables (256 entries each)
#[inline]
pub fn crc32_slice16(mut crc: u32, data: &[u8], tables: &[[u32; 256]; 16]) -> u32 {
  let mut chunks = data.chunks_exact(16);

  for chunk in chunks.by_ref() {
    // Read 16 bytes as four u32s (little-endian)
    let w0 = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    let w1 = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
    let w2 = u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
    let w3 = u32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]);

    // XOR first 4 bytes with current CRC
    let v0 = w0 ^ crc;

    // Lookup all 16 bytes using 16 different tables
    crc = tables[15][(v0 & 0xFF) as usize]
      ^ tables[14][((v0 >> 8) & 0xFF) as usize]
      ^ tables[13][((v0 >> 16) & 0xFF) as usize]
      ^ tables[12][(v0 >> 24) as usize]
      ^ tables[11][(w1 & 0xFF) as usize]
      ^ tables[10][((w1 >> 8) & 0xFF) as usize]
      ^ tables[9][((w1 >> 16) & 0xFF) as usize]
      ^ tables[8][(w1 >> 24) as usize]
      ^ tables[7][(w2 & 0xFF) as usize]
      ^ tables[6][((w2 >> 8) & 0xFF) as usize]
      ^ tables[5][((w2 >> 16) & 0xFF) as usize]
      ^ tables[4][(w2 >> 24) as usize]
      ^ tables[3][(w3 & 0xFF) as usize]
      ^ tables[2][((w3 >> 8) & 0xFF) as usize]
      ^ tables[1][((w3 >> 16) & 0xFF) as usize]
      ^ tables[0][(w3 >> 24) as usize];
  }

  // Handle remainder (0-15 bytes) with byte-at-a-time using table[0]
  for &byte in chunks.remainder() {
    let index = ((crc ^ (byte as u32)) & 0xFF) as usize;
    crc = tables[0][index] ^ (crc >> 8);
  }

  crc
}

#[cfg(test)]
mod tests {
  extern crate alloc;
  use alloc::vec::Vec;

  use super::*;
  use crate::common::tables::{CRC32_IEEE_POLY, CRC32C_POLY, generate_crc32_tables_8, generate_crc32_tables_16};

  // ─────────────────────────────────────────────────────────────────────────────
  // Slice-by-8 tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_slice8_empty() {
    let tables = generate_crc32_tables_8(CRC32_IEEE_POLY);
    let crc = crc32_slice8(!0, &[], &tables);
    assert_eq!(crc, !0);
  }

  #[test]
  fn test_slice8_short_data() {
    let tables = generate_crc32_tables_8(CRC32_IEEE_POLY);

    // Test with 1-7 bytes (all remainder cases)
    for len in 1..8 {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();
      let crc = crc32_slice8(!0, &data, &tables);
      assert_ne!(crc, !0, "CRC should change for non-empty data");
    }
  }

  #[test]
  fn test_slice8_exact_multiple() {
    let tables = generate_crc32_tables_8(CRC32_IEEE_POLY);

    // Test with exactly 8, 16, 24 bytes
    for len in [8, 16, 24] {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();
      let crc = crc32_slice8(!0, &data, &tables);
      assert_ne!(crc, !0);
    }
  }

  #[test]
  fn test_slice8_consistency_with_byte_at_a_time() {
    let tables = generate_crc32_tables_8(CRC32C_POLY);
    let data = b"The quick brown fox jumps over the lazy dog";

    // Compute with slice-by-8
    let slice8_result = crc32_slice8(!0, data, &tables);

    // Compute byte-at-a-time using table[0]
    let mut byte_result = !0u32;
    for &b in data.iter() {
      let index = ((byte_result ^ (b as u32)) & 0xFF) as usize;
      byte_result = tables[0][index] ^ (byte_result >> 8);
    }

    assert_eq!(slice8_result, byte_result);
  }

  #[test]
  fn test_slice8_incremental() {
    let tables = generate_crc32_tables_8(CRC32_IEEE_POLY);
    let data = b"hello world, this is a longer test string";

    // Full computation
    let full = crc32_slice8(!0, data, &tables);

    // Split at various points and verify consistency
    for split in [1, 7, 8, 9, 15, 16, 17, 20] {
      if split < data.len() {
        let crc1 = crc32_slice8(!0, &data[..split], &tables);
        let crc2 = crc32_slice8(crc1, &data[split..], &tables);
        assert_eq!(crc2, full, "Incremental failed at split {split}");
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Slice-by-16 tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_slice16_empty() {
    let tables = generate_crc32_tables_16(CRC32_IEEE_POLY);
    let crc = crc32_slice16(!0, &[], &tables);
    assert_eq!(crc, !0);
  }

  #[test]
  fn test_slice16_short_data() {
    let tables = generate_crc32_tables_16(CRC32_IEEE_POLY);

    // Test with 1-15 bytes (all remainder cases)
    for len in 1..16 {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();
      let crc = crc32_slice16(!0, &data, &tables);
      assert_ne!(crc, !0, "CRC should change for non-empty data (len={len})");
    }
  }

  #[test]
  fn test_slice16_exact_multiple() {
    let tables = generate_crc32_tables_16(CRC32_IEEE_POLY);

    // Test with exactly 16, 32, 48 bytes
    for len in [16, 32, 48] {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();
      let crc = crc32_slice16(!0, &data, &tables);
      assert_ne!(crc, !0);
    }
  }

  #[test]
  fn test_slice16_matches_slice8() {
    let tables8 = generate_crc32_tables_8(CRC32_IEEE_POLY);
    let tables16 = generate_crc32_tables_16(CRC32_IEEE_POLY);

    let data = b"The quick brown fox jumps over the lazy dog";
    let slice8_result = crc32_slice8(!0, data, &tables8);
    let slice16_result = crc32_slice16(!0, data, &tables16);
    assert_eq!(slice8_result, slice16_result);
  }

  #[test]
  fn test_slice16_matches_slice8_castagnoli() {
    let tables8 = generate_crc32_tables_8(CRC32C_POLY);
    let tables16 = generate_crc32_tables_16(CRC32C_POLY);

    let data = b"The quick brown fox jumps over the lazy dog";
    let slice8_result = crc32_slice8(!0, data, &tables8);
    let slice16_result = crc32_slice16(!0, data, &tables16);
    assert_eq!(slice8_result, slice16_result);
  }

  #[test]
  fn test_slice16_consistency_with_byte_at_a_time() {
    let tables = generate_crc32_tables_16(CRC32C_POLY);
    let data = b"The quick brown fox jumps over the lazy dog";

    // Compute with slice-by-16
    let slice16_result = crc32_slice16(!0, data, &tables);

    // Compute byte-at-a-time using table[0]
    let mut byte_result = !0u32;
    for &b in data.iter() {
      let index = ((byte_result ^ (b as u32)) & 0xFF) as usize;
      byte_result = tables[0][index] ^ (byte_result >> 8);
    }

    assert_eq!(slice16_result, byte_result);
  }

  #[test]
  fn test_slice16_incremental() {
    let tables = generate_crc32_tables_16(CRC32_IEEE_POLY);
    let data = b"hello world, this is a longer test string for slice16";

    // Full computation
    let full = crc32_slice16(!0, data, &tables);

    // Split at various points and verify consistency
    for split in [1, 7, 8, 9, 15, 16, 17, 20, 31, 32, 33] {
      if split < data.len() {
        let crc1 = crc32_slice16(!0, &data[..split], &tables);
        let crc2 = crc32_slice16(crc1, &data[split..], &tables);
        assert_eq!(crc2, full, "Incremental failed at split {split}");
      }
    }
  }

  #[test]
  fn test_slice16_various_lengths() {
    let tables8 = generate_crc32_tables_8(CRC32_IEEE_POLY);
    let tables16 = generate_crc32_tables_16(CRC32_IEEE_POLY);

    // Test a range of lengths to ensure slice16 matches slice8
    for len in 0..=64 {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();
      let slice8_result = crc32_slice8(!0, &data, &tables8);
      let slice16_result = crc32_slice16(!0, &data, &tables16);
      assert_eq!(slice8_result, slice16_result, "Mismatch at length {len}");
    }
  }
}
