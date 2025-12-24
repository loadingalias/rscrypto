//! Portable CRC-32 implementations using table-driven algorithms.
//!
//! These implementations work on all platforms without hardware acceleration.
//! They serve as:
//! - Fallback when SIMD isn't available or the buffer is too small
//! - Reference implementations for testing

use crate::common::{
  portable::{slice8_32, slice16_32},
  tables::{CRC32_IEEE_POLY, CRC32C_POLY, generate_crc32_tables_8, generate_crc32_tables_16},
};

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32 IEEE (Ethernet/ZIP/PNG)
// ─────────────────────────────────────────────────────────────────────────────

/// Slice-by-8 lookup tables for CRC-32-IEEE.
#[allow(dead_code)]
const CRC32_IEEE_TABLES_8: [[u32; 256]; 8] = generate_crc32_tables_8(CRC32_IEEE_POLY);

/// Slice-by-16 lookup tables for CRC-32-IEEE.
const CRC32_IEEE_TABLES_16: [[u32; 256]; 16] = generate_crc32_tables_16(CRC32_IEEE_POLY);

/// CRC-32-IEEE using slice-by-8 algorithm.
///
/// Processes 8 bytes per iteration. Good balance of speed and table size.
#[inline]
#[allow(dead_code)]
pub fn crc32_ieee_slice8(crc: u32, data: &[u8]) -> u32 {
  slice8_32(crc, data, &CRC32_IEEE_TABLES_8)
}

/// CRC-32-IEEE using slice-by-16 algorithm.
///
/// Processes 16 bytes per iteration. Fastest portable implementation.
#[inline]
pub fn crc32_ieee_slice16(crc: u32, data: &[u8]) -> u32 {
  slice16_32(crc, data, &CRC32_IEEE_TABLES_16)
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32C (Castagnoli) - iSCSI/ext4/Btrfs
// ─────────────────────────────────────────────────────────────────────────────

/// Slice-by-8 lookup tables for CRC-32C.
#[allow(dead_code)]
const CRC32C_TABLES_8: [[u32; 256]; 8] = generate_crc32_tables_8(CRC32C_POLY);

/// Slice-by-16 lookup tables for CRC-32C.
const CRC32C_TABLES_16: [[u32; 256]; 16] = generate_crc32_tables_16(CRC32C_POLY);

/// CRC-32C using slice-by-8 algorithm.
///
/// Processes 8 bytes per iteration. Good balance of speed and table size.
#[inline]
#[allow(dead_code)]
pub fn crc32c_slice8(crc: u32, data: &[u8]) -> u32 {
  slice8_32(crc, data, &CRC32C_TABLES_8)
}

/// CRC-32C using slice-by-16 algorithm.
///
/// Processes 16 bytes per iteration. Fastest portable implementation.
#[inline]
pub fn crc32c_slice16(crc: u32, data: &[u8]) -> u32 {
  slice16_32(crc, data, &CRC32C_TABLES_16)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  extern crate alloc;
  use alloc::vec::Vec;

  use super::*;

  // Known test vectors
  const CRC32_IEEE_EMPTY: u32 = 0x0000_0000;
  const CRC32_IEEE_HELLO_WORLD: u32 = 0x0D4A_1185; // "hello world"

  const CRC32C_EMPTY: u32 = 0x0000_0000;
  const CRC32C_HELLO_WORLD: u32 = 0xC99465AA; // "hello world"

  fn compute_crc32_ieee(data: &[u8]) -> u32 {
    crc32_ieee_slice16(!0, data) ^ !0
  }

  fn compute_crc32c(data: &[u8]) -> u32 {
    crc32c_slice16(!0, data) ^ !0
  }

  #[test]
  fn test_crc32_ieee_empty() {
    assert_eq!(compute_crc32_ieee(b""), CRC32_IEEE_EMPTY);
  }

  #[test]
  fn test_crc32_ieee_hello_world() {
    assert_eq!(compute_crc32_ieee(b"hello world"), CRC32_IEEE_HELLO_WORLD);
  }

  #[test]
  fn test_crc32c_empty() {
    assert_eq!(compute_crc32c(b""), CRC32C_EMPTY);
  }

  #[test]
  fn test_crc32c_hello_world() {
    assert_eq!(compute_crc32c(b"hello world"), CRC32C_HELLO_WORLD);
  }

  #[test]
  fn test_slice8_vs_slice16_ieee() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let crc8 = crc32_ieee_slice8(!0, data) ^ !0;
    let crc16 = crc32_ieee_slice16(!0, data) ^ !0;
    assert_eq!(crc8, crc16);
  }

  #[test]
  fn test_slice8_vs_slice16_crc32c() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let crc8 = crc32c_slice8(!0, data) ^ !0;
    let crc16 = crc32c_slice16(!0, data) ^ !0;
    assert_eq!(crc8, crc16);
  }

  #[test]
  fn test_incremental_ieee() {
    let full = compute_crc32_ieee(b"hello world");

    // Compute incrementally
    let mut state = !0u32;
    state = crc32_ieee_slice16(state, b"hello ");
    state = crc32_ieee_slice16(state, b"world");
    let incremental = state ^ !0;

    assert_eq!(incremental, full);
  }

  #[test]
  fn test_incremental_crc32c() {
    let full = compute_crc32c(b"hello world");

    // Compute incrementally
    let mut state = !0u32;
    state = crc32c_slice16(state, b"hello ");
    state = crc32c_slice16(state, b"world");
    let incremental = state ^ !0;

    assert_eq!(incremental, full);
  }

  #[test]
  fn test_various_lengths_ieee() {
    // Test various lengths to exercise alignment and remainder handling
    for len in 0..=64 {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();
      let crc8 = crc32_ieee_slice8(!0, &data) ^ !0;
      let crc16 = crc32_ieee_slice16(!0, &data) ^ !0;
      assert_eq!(crc8, crc16, "Mismatch at length {len}");
    }
  }

  #[test]
  fn test_various_lengths_crc32c() {
    for len in 0..=64 {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();
      let crc8 = crc32c_slice8(!0, &data) ^ !0;
      let crc16 = crc32c_slice16(!0, &data) ^ !0;
      assert_eq!(crc8, crc16, "Mismatch at length {len}");
    }
  }
}
