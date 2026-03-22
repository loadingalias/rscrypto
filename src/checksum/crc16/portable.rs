//! Portable CRC-16 implementations (slice-by-4 and slice-by-8).

use super::kernel_tables;
use crate::checksum::common::portable;

/// CRC-16/CCITT (X25 / IBM-SDLC) slice-by-4 computation.
///
/// This is a legacy kernel kept for benchmarking and testing. Prefer `crc16_ccitt_slice8`.
#[cfg(any(test, feature = "testing"))]
#[doc(hidden)]
#[inline]
pub(crate) fn crc16_ccitt_slice4(crc: u16, data: &[u8]) -> u16 {
  portable::slice4_16(crc, data, &kernel_tables::CCITT_TABLES_4)
}

/// CRC-16/CCITT (X25 / IBM-SDLC) slice-by-8 computation.
#[inline]
pub fn crc16_ccitt_slice8(crc: u16, data: &[u8]) -> u16 {
  portable::slice8_16(crc, data, &kernel_tables::CCITT_TABLES_8)
}

/// CRC-16/IBM (ARC) slice-by-4 computation.
///
/// This is a legacy kernel kept for benchmarking and testing. Prefer `crc16_ibm_slice8`.
#[cfg(any(test, feature = "testing"))]
#[doc(hidden)]
#[inline]
pub(crate) fn crc16_ibm_slice4(crc: u16, data: &[u8]) -> u16 {
  portable::slice4_16(crc, data, &kernel_tables::IBM_TABLES_4)
}

/// CRC-16/IBM (ARC) slice-by-8 computation.
#[inline]
pub fn crc16_ibm_slice8(crc: u16, data: &[u8]) -> u16 {
  portable::slice8_16(crc, data, &kernel_tables::IBM_TABLES_8)
}

// ─────────────────────────────────────────────────────────────────────────────
// Byte-at-a-time (fast-path for tiny buffers)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-16/CCITT byte-at-a-time lookup computation.
///
/// This is typically faster than slice-by-8 for tiny buffers because it uses a
/// single 256-entry table.
#[inline]
pub fn crc16_ccitt_bytewise(crc: u16, data: &[u8]) -> u16 {
  crc16_bytewise(crc, data, &kernel_tables::CCITT_TABLES_8[0])
}

/// CRC-16/IBM byte-at-a-time lookup computation.
///
/// This is typically faster than slice-by-8 for tiny buffers because it uses a
/// single 256-entry table.
#[inline]
pub fn crc16_ibm_bytewise(crc: u16, data: &[u8]) -> u16 {
  crc16_bytewise(crc, data, &kernel_tables::IBM_TABLES_8[0])
}

/// Update CRC-16 state using a byte-at-a-time lookup table.
#[inline]
#[allow(clippy::indexing_slicing)] // index is 0..=255 by mask, table is [u16; 256]
fn crc16_bytewise(mut crc: u16, data: &[u8], table: &[u16; 256]) -> u16 {
  for &b in data {
    let index = ((crc ^ (b as u16)) & 0xFF) as usize;
    crc = table[index] ^ (crc >> 8);
  }
  crc
}
