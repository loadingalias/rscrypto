//! CRC16/CCITT-FALSE constants.
//!
//! Polynomial: 0x1021 (normal, MSB-first)
//! Used by: X.25, HDLC, Bluetooth, SD cards

/// CRC16/CCITT-FALSE polynomial (normal form).
pub const POLYNOMIAL: u16 = 0x1021;

/// Single-byte lookup table for CRC16/CCITT-FALSE (MSB-first).
///
/// Total size: 256 * 2 = 512B.
#[cfg(not(feature = "no-tables"))]
pub const TABLE: [u16; 256] = generate_table();

#[cfg(not(feature = "no-tables"))]
const fn generate_table() -> [u16; 256] {
  let mut table = [0u16; 256];
  let mut i = 0usize;

  while i < 256 {
    // Align byte into the top 8 bits of the 16-bit register.
    let mut crc = (i as u16) << 8;
    let mut j = 0;
    while j < 8 {
      if (crc & 0x8000) != 0 {
        crc = (crc << 1) ^ POLYNOMIAL;
      } else {
        crc <<= 1;
      }
      j += 1;
    }
    table[i] = crc;
    i += 1;
  }

  table
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_polynomial() {
    assert_eq!(POLYNOMIAL, 0x1021);
  }
}
