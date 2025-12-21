//! Portable CRC-24 implementation (byte-at-a-time, MSB-first).

// SAFETY: Table index is `((crc >> 16) ^ byte) & 0xFF`, always in 0..255.
#![allow(clippy::indexing_slicing)]

/// Update CRC-24 state with data using byte-at-a-time table lookup.
///
/// CRC-24 uses MSB-first (non-reflected) processing.
///
/// # Arguments
///
/// * `crc` - Current CRC state (24-bit value in low bits of u32)
/// * `data` - Input data
/// * `table` - 256-entry lookup table for the polynomial
#[inline]
pub fn crc24_update(mut crc: u32, data: &[u8], table: &[u32; 256]) -> u32 {
  for &byte in data {
    let index = ((crc >> 16) ^ (byte as u32)) & 0xFF;
    crc = ((crc << 8) ^ table[index as usize]) & 0xFF_FFFF;
  }
  crc
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::common::tables::{CRC24_OPENPGP_POLY, generate_crc24_table};

  #[test]
  fn test_crc24_update_empty() {
    let table = generate_crc24_table(CRC24_OPENPGP_POLY);
    let init = 0xB704CE;
    let crc = crc24_update(init, &[], &table);
    assert_eq!(crc, init);
  }

  #[test]
  fn test_crc24_update_single_byte() {
    let table = generate_crc24_table(CRC24_OPENPGP_POLY);
    let init = 0xB704CE;
    let crc = crc24_update(init, &[0x00], &table);
    // Single byte should change the CRC
    assert_ne!(crc, init);
    // Result should be 24 bits
    assert!(crc <= 0xFF_FFFF);
  }
}
