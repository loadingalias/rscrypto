//! Portable CRC-24 implementations (slice-by-4 and slice-by-8).

use super::kernel_tables;
use crate::common::portable;

/// CRC-24/OPENPGP slice-by-4 computation.
#[inline]
pub fn crc24_openpgp_slice4(crc: u32, data: &[u8]) -> u32 {
  portable::slice4_24(crc, data, &kernel_tables::OPENPGP_TABLES_4)
}

/// CRC-24/OPENPGP slice-by-8 computation.
#[inline]
pub fn crc24_openpgp_slice8(crc: u32, data: &[u8]) -> u32 {
  portable::slice8_24(crc, data, &kernel_tables::OPENPGP_TABLES_8)
}
