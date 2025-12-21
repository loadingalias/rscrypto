//! Portable CRC-16 implementation (byte-at-a-time).

// SAFETY: Table index is `(crc ^ byte) & 0xFF`, always in 0..255.
#![allow(clippy::indexing_slicing)]

/// Update CRC-16 state with data using byte-at-a-time table lookup.
///
/// # Arguments
///
/// * `crc` - Current CRC state
/// * `data` - Input data
/// * `table` - 256-entry lookup table for the polynomial
#[inline]
pub fn crc16_update(mut crc: u16, data: &[u8], table: &[u16; 256]) -> u16 {
  for &byte in data {
    let index = (crc ^ (byte as u16)) & 0xFF;
    crc = table[index as usize] ^ (crc >> 8);
  }
  crc
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::common::tables::{CRC16_CCITT_POLY, generate_crc16_table};

  #[test]
  fn test_crc16_update_empty() {
    let table = generate_crc16_table(CRC16_CCITT_POLY);
    let crc = crc16_update(0xFFFF, &[], &table);
    assert_eq!(crc, 0xFFFF);
  }

  #[test]
  fn test_crc16_update_single_byte() {
    let table = generate_crc16_table(CRC16_CCITT_POLY);
    let crc = crc16_update(0xFFFF, &[0x00], &table);
    // Single byte should change the CRC
    assert_ne!(crc, 0xFFFF);
  }
}
