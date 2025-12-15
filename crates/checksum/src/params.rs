//! CRC algorithm parameters.
//!
//! This module defines the parameters for various CRC algorithms following
//! the conventions from the [CRC Catalogue](https://reveng.sourceforge.io/crc-catalogue/).

/// CRC algorithm parameters.
///
/// This struct captures all the parameters needed to define a CRC algorithm.
/// The parameters follow the conventions from the CRC Catalogue.
///
/// # Parameters
///
/// - `width`: Number of bits in the CRC (8, 16, 24, 32, or 64)
/// - `polynomial`: The generator polynomial (without the implicit high bit)
/// - `initial`: Initial value for the CRC register
/// - `reflect_in`: If true, reflect each input byte before processing
/// - `reflect_out`: If true, reflect the final CRC before XOR
/// - `xor_out`: Value to XOR with the final CRC
///
/// # Reflection
///
/// "Reflected" means bit-reversed. Most common CRCs (CRC32, CRC32C) use
/// reflected input and output, which maps to LSB-first processing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrcParams {
  /// Width in bits (8, 16, 24, 32, or 64).
  pub width: u8,
  /// Generator polynomial (without implicit high bit).
  pub polynomial: u64,
  /// Initial value for the CRC register.
  pub initial: u64,
  /// Reflect input bytes before processing.
  pub reflect_in: bool,
  /// Reflect final CRC before XOR.
  pub reflect_out: bool,
  /// XOR value applied to final CRC.
  pub xor_out: u64,
}

impl CrcParams {
  /// CRC32-C (Castagnoli) - iSCSI, SCTP, Btrfs, ext4, RocksDB, LevelDB
  ///
  /// This polynomial was specifically designed to have good error detection
  /// properties for data storage and networking.
  ///
  /// Hardware support: SSE4.2 `crc32` instruction, ARMv8 CRC extension.
  pub const CRC32C: Self = Self {
    width: 32,
    polynomial: 0x1EDC6F41,
    initial: 0xFFFFFFFF,
    reflect_in: true,
    reflect_out: true,
    xor_out: 0xFFFFFFFF,
  };

  /// CRC32 (ISO 3309) - Ethernet, gzip, PNG, zip, SATA
  ///
  /// The most widely used CRC32 variant. Used in virtually all file formats
  /// and network protocols that use CRC32.
  ///
  /// Note: No dedicated hardware instruction (unlike CRC32-C), but can be
  /// accelerated via PCLMULQDQ/PMULL.
  pub const CRC32_ISO: Self = Self {
    width: 32,
    polynomial: 0x04C11DB7,
    initial: 0xFFFFFFFF,
    reflect_in: true,
    reflect_out: true,
    xor_out: 0xFFFFFFFF,
  };

  /// CRC64 (ECMA-182) - XZ compression, PostgreSQL, Redis
  ///
  /// 64-bit CRC used in data storage and compression.
  pub const CRC64_ECMA: Self = Self {
    width: 64,
    polynomial: 0x42F0E1EBA9EA3693,
    initial: 0xFFFFFFFFFFFFFFFF,
    reflect_in: true,
    reflect_out: true,
    xor_out: 0xFFFFFFFFFFFFFFFF,
  };

  /// CRC64/NVME - NVMe storage specification
  pub const CRC64_NVME: Self = Self {
    width: 64,
    polynomial: 0xAD93D23594C93659,
    initial: 0xFFFFFFFFFFFFFFFF,
    reflect_in: true,
    reflect_out: true,
    xor_out: 0xFFFFFFFFFFFFFFFF,
  };

  /// CRC16/IBM - Modbus, USB, many legacy protocols
  pub const CRC16_IBM: Self = Self {
    width: 16,
    polynomial: 0x8005,
    initial: 0x0000,
    reflect_in: true,
    reflect_out: true,
    xor_out: 0x0000,
  };

  /// CRC16/CCITT - X.25, HDLC, Bluetooth, SD cards
  pub const CRC16_CCITT: Self = Self {
    width: 16,
    polynomial: 0x1021,
    initial: 0xFFFF,
    reflect_in: false,
    reflect_out: false,
    xor_out: 0x0000,
  };

  /// CRC8/MAXIM - 1-Wire, iButton, sensor networks
  pub const CRC8_MAXIM: Self = Self {
    width: 8,
    polynomial: 0x31,
    initial: 0x00,
    reflect_in: true,
    reflect_out: true,
    xor_out: 0x00,
  };

  /// CRC24/OpenPGP - OpenPGP, IETF protocols
  pub const CRC24_OPENPGP: Self = Self {
    width: 24,
    polynomial: 0x864CFB,
    initial: 0xB704CE,
    reflect_in: false,
    reflect_out: false,
    xor_out: 0x000000,
  };

  /// Returns the reflected polynomial (bit-reversed).
  ///
  /// For reflected CRCs, the polynomial is processed in bit-reversed form.
  #[must_use]
  pub const fn polynomial_reflected(&self) -> u64 {
    reflect_bits(self.polynomial, self.width)
  }
}

/// Reflect (bit-reverse) the lower `width` bits of `value`.
#[must_use]
const fn reflect_bits(value: u64, width: u8) -> u64 {
  let mut result = 0u64;
  let mut i = 0u8;
  while i < width {
    if (value >> i) & 1 != 0 {
      result |= 1 << (width.wrapping_sub(1).wrapping_sub(i));
    }
    i = i.wrapping_add(1);
  }
  result
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_crc32c_polynomial_reflected() {
    // CRC32-C polynomial 0x1EDC6F41 reflected is 0x82F63B78
    assert_eq!(CrcParams::CRC32C.polynomial_reflected(), 0x82F63B78);
  }

  #[test]
  fn test_crc32_polynomial_reflected() {
    // CRC32 (ISO) polynomial 0x04C11DB7 reflected is 0xEDB88320
    assert_eq!(CrcParams::CRC32_ISO.polynomial_reflected(), 0xEDB88320);
  }

  #[test]
  fn test_reflect_bits() {
    assert_eq!(reflect_bits(0b1010, 4), 0b0101);
    assert_eq!(reflect_bits(0b1100, 4), 0b0011);
    assert_eq!(reflect_bits(0xFF, 8), 0xFF);
    assert_eq!(reflect_bits(0x80, 8), 0x01);
  }
}
