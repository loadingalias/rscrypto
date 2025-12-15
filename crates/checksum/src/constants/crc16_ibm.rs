//! CRC16/IBM constants.
//!
//! Polynomial: 0x8005 (reflected: 0xA001)
//! Used by: Modbus, USB, legacy protocols

/// CRC16/IBM polynomial in reflected (bit-reversed) form.
pub const POLYNOMIAL: u16 = 0xA001;

/// Slicing-by-8 lookup tables.
///
/// Total size: 8 * 256 * 2 = 4KB.
/// The tables are 64-byte aligned for optimal cache behavior.
#[cfg(not(feature = "no-tables"))]
pub static TABLES: super::Aligned64<[[u16; 256]; 8]> =
  super::Aligned64(super::tables::generate_slicing_tables_16(POLYNOMIAL));

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_polynomial() {
    assert_eq!(POLYNOMIAL, 0xA001);
  }
}
