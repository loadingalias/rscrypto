//! Portable CRC-24 implementations.

use super::kernel_tables;
use crate::checksum::common::portable;

/// CRC-24/OPENPGP slice-by-8 computation.
#[inline]
pub fn crc24_openpgp_slice8(crc: u32, data: &[u8]) -> u32 {
  portable::slice8_24(crc, data, &kernel_tables::OPENPGP_TABLES_8)
}

// Byte-at-a-time (fast-path for tiny buffers)

/// CRC-24/OpenPGP byte-at-a-time lookup computation (MSB-first).
///
/// Uses one 256-entry table rather than the slice-by-8 table set.
#[inline(always)]
#[allow(clippy::indexing_slicing)] // index is 0..=255 by byte cast, table is [u32; 256]
pub fn crc24_openpgp_bytewise(crc: u32, data: &[u8]) -> u32 {
  const MASK24: u32 = 0x00FF_FFFF;
  let mut state = (crc & MASK24) << 8;
  for &byte in data {
    let index = (((state >> 24) as u8) ^ byte) as usize;
    state = kernel_tables::OPENPGP_TABLES_8[0][index] ^ (state << 8);
  }
  (state >> 8) & MASK24
}
