//! Portable CRC-64 implementation (slice-by-8 and slice-by-16).
//!
//! This module provides polynomial-specific wrappers around the generic
//! slice-by-N implementations in [`crate::common::portable`].

use super::kernel_tables;
use crate::common::portable;

// ─────────────────────────────────────────────────────────────────────────────
// Polynomial-specific wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-64-XZ slice-by-8 computation.
#[inline]
#[cfg(test)]
#[allow(dead_code)] // Used by arch-specific tests that don't run under Miri
pub fn crc64_slice8_xz(crc: u64, data: &[u8]) -> u64 {
  crc64_slice8(crc, data, &kernel_tables::XZ_TABLES_8)
}

/// CRC-64-NVME slice-by-8 computation.
#[inline]
#[cfg(test)]
#[allow(dead_code)] // Used by arch-specific tests that don't run under Miri
pub fn crc64_slice8_nvme(crc: u64, data: &[u8]) -> u64 {
  crc64_slice8(crc, data, &kernel_tables::NVME_TABLES_8)
}

/// CRC-64-XZ slice-by-16 computation.
#[inline]
pub fn crc64_slice16_xz(crc: u64, data: &[u8]) -> u64 {
  crc64_slice16(crc, data, &kernel_tables::XZ_TABLES_16)
}

/// CRC-64-NVME slice-by-16 computation.
#[inline]
pub fn crc64_slice16_nvme(crc: u64, data: &[u8]) -> u64 {
  crc64_slice16(crc, data, &kernel_tables::NVME_TABLES_16)
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic slice-by-8/16 (delegating to common::portable)
// ─────────────────────────────────────────────────────────────────────────────

/// Update CRC-64 state using slice-by-8 algorithm.
///
/// # Arguments
///
/// * `crc` - Current CRC state (pre-inverted)
/// * `data` - Input data
/// * `tables` - 8 lookup tables (256 entries each)
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", test))]
#[inline]
pub fn crc64_slice8(crc: u64, data: &[u8], tables: &[[u64; 256]; 8]) -> u64 {
  portable::slice8_64(crc, data, tables)
}

/// Update CRC-64 state using slice-by-16 algorithm.
///
/// # Arguments
///
/// * `crc` - Current CRC state (pre-inverted)
/// * `data` - Input data
/// * `tables` - 16 lookup tables (256 entries each)
#[inline]
pub fn crc64_slice16(crc: u64, data: &[u8], tables: &[[u64; 256]; 16]) -> u64 {
  portable::slice16_64(crc, data, tables)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::common::tables::{CRC64_XZ_POLY, generate_crc64_tables_8, generate_crc64_tables_16};

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

  #[test]
  fn test_slice16_empty() {
    let tables = generate_crc64_tables_16(CRC64_XZ_POLY);
    let crc = crc64_slice16(!0, &[], &tables);
    assert_eq!(crc, !0);
  }

  #[test]
  fn test_slice16_matches_slice8() {
    let tables8 = generate_crc64_tables_8(CRC64_XZ_POLY);
    let tables16 = generate_crc64_tables_16(CRC64_XZ_POLY);

    let data = b"The quick brown fox jumps over the lazy dog";
    let a = crc64_slice8(!0, data, &tables8);
    let b = crc64_slice16(!0, data, &tables16);
    assert_eq!(a, b);
  }

  #[test]
  fn test_slice16_incremental() {
    let tables = generate_crc64_tables_16(CRC64_XZ_POLY);
    let data = b"hello world, this is a longer test string";

    let full = crc64_slice16(!0, data, &tables);

    for split in [1, 7, 8, 9, 15, 16, 17, 20] {
      if split < data.len() {
        let crc1 = crc64_slice16(!0, &data[..split], &tables);
        let crc2 = crc64_slice16(crc1, &data[split..], &tables);
        assert_eq!(crc2, full, "Incremental failed at split {split}");
      }
    }
  }
}
