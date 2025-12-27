//! Portable CRC-32 implementations (slice-by-8 and slice-by-16).
//!
//! This module provides polynomial-specific wrappers around the generic
//! slice-by-N implementations in [`crate::common::portable`].

use super::kernel_tables;
use crate::common::portable;

// ─────────────────────────────────────────────────────────────────────────────
// Polynomial-specific wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32 (IEEE) slice-by-16 computation.
#[inline]
pub fn crc32_slice16_ieee(crc: u32, data: &[u8]) -> u32 {
  crc32_slice16(crc, data, &kernel_tables::IEEE_TABLES_16)
}

/// CRC-32C (Castagnoli) slice-by-16 computation.
#[inline]
pub fn crc32c_slice16(crc: u32, data: &[u8]) -> u32 {
  crc32_slice16(crc, data, &kernel_tables::CRC32C_TABLES_16)
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic slice-by-16 (delegating to common::portable)
// ─────────────────────────────────────────────────────────────────────────────

/// Update CRC-32 state using slice-by-16 algorithm.
#[inline]
pub fn crc32_slice16(crc: u32, data: &[u8], tables: &[[u32; 256]; 16]) -> u32 {
  portable::slice16_32(crc, data, tables)
}
