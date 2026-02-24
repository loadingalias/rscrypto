//! Portable CRC-32 implementations (slice-by-8 and slice-by-16).
//!
//! This module provides polynomial-specific wrappers around the generic
//! slice-by-N implementations in [`crate::common::portable`].

use super::kernel_tables;
use crate::common::portable;

// ─────────────────────────────────────────────────────────────────────────────
// Polynomial-specific wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// Canonical kernel name for byte-at-a-time table lookup kernels.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(crate) const BYTEWISE_KERNEL_NAME: &str = "portable/bytewise";

/// CRC-32 (IEEE) byte-at-a-time lookup computation.
///
/// This is typically faster than slice-by-16 for tiny buffers because it uses a
/// single 256-entry table.
#[inline]
pub fn crc32_bytewise_ieee(crc: u32, data: &[u8]) -> u32 {
  crc32_bytewise(crc, data, &kernel_tables::IEEE_TABLES_16[0])
}

/// CRC-32C (Castagnoli) byte-at-a-time lookup computation.
///
/// This is typically faster than slice-by-16 for tiny buffers because it uses a
/// single 256-entry table.
#[inline]
pub fn crc32c_bytewise(crc: u32, data: &[u8]) -> u32 {
  crc32_bytewise(crc, data, &kernel_tables::CRC32C_TABLES_16[0])
}

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
// Generic bytewise/slice-by-16 (delegating to common::portable)
// ─────────────────────────────────────────────────────────────────────────────

/// Update CRC-32 state using a byte-at-a-time lookup table.
#[inline]
#[allow(clippy::indexing_slicing)] // index is 0..=255 by mask, table is [u32; 256]
pub fn crc32_bytewise(mut crc: u32, data: &[u8], table: &[u32; 256]) -> u32 {
  for &b in data {
    let index = ((crc ^ (b as u32)) & 0xFF) as usize;
    crc = table[index] ^ (crc >> 8);
  }
  crc
}

/// Update CRC-32 state using slice-by-16 algorithm.
#[inline]
pub fn crc32_slice16(crc: u32, data: &[u8], tables: &[[u32; 256]; 16]) -> u32 {
  portable::slice16_32(crc, data, tables)
}
