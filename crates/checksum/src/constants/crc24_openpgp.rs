//! CRC24/OpenPGP constants.
//!
//! Polynomial: 0x864CFB (normal, MSB-first)
//! Used by: OpenPGP, IETF protocols

/// CRC24/OpenPGP polynomial (normal form).
pub const POLYNOMIAL: u32 = 0x86_4C_FB;

/// Single-byte lookup table for CRC24/OpenPGP (MSB-first).
///
/// Total size: 256 * 4 = 1KB.
#[cfg(not(feature = "no-tables"))]
pub const TABLE: [u32; 256] = generate_table();

#[cfg(not(feature = "no-tables"))]
const fn generate_table() -> [u32; 256] {
  let mut table = [0u32; 256];
  let mut i = 0usize;

  while i < 256 {
    // Align byte into the top 8 bits of the 24-bit register.
    let mut crc = (i as u32) << 16;
    let mut j = 0;
    while j < 8 {
      if (crc & 0x80_0000) != 0 {
        crc = (crc << 1) ^ POLYNOMIAL;
      } else {
        crc <<= 1;
      }
      crc &= 0xFF_FFFF;
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
    assert_eq!(POLYNOMIAL, 0x86_4C_FB);
  }
}
