//! Portable CRC-16 implementations (slice-by-4 and slice-by-8).

use super::kernel_tables;
use crate::common::portable;

/// CRC-16/CCITT (X25 / IBM-SDLC) slice-by-4 computation.
#[inline]
pub fn crc16_ccitt_slice4(crc: u16, data: &[u8]) -> u16 {
  portable::slice4_16(crc, data, &kernel_tables::CCITT_TABLES_4)
}

/// CRC-16/CCITT (X25 / IBM-SDLC) slice-by-8 computation.
#[inline]
pub fn crc16_ccitt_slice8(crc: u16, data: &[u8]) -> u16 {
  portable::slice8_16(crc, data, &kernel_tables::CCITT_TABLES_8)
}

/// CRC-16/IBM (ARC) slice-by-4 computation.
#[inline]
pub fn crc16_ibm_slice4(crc: u16, data: &[u8]) -> u16 {
  portable::slice4_16(crc, data, &kernel_tables::IBM_TABLES_4)
}

/// CRC-16/IBM (ARC) slice-by-8 computation.
#[inline]
pub fn crc16_ibm_slice8(crc: u16, data: &[u8]) -> u16 {
  portable::slice8_16(crc, data, &kernel_tables::IBM_TABLES_8)
}
