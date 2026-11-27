//! CRC-32/ISO-HDLC (IEEE) and CRC-32C (Castagnoli).

use crate::checksum::crc::api::define_crc;

define_crc! {
  /// CRC-32 (IEEE) checksum.
  ///
  /// Used by Ethernet, gzip, zip, PNG, etc.
  ///
  /// # Properties
  ///
  /// - **Polynomial**: 0x04C11DB7 (normal), 0xEDB88320 (reflected)
  /// - **Initial value**: 0xFFFFFFFF
  /// - **Final XOR**: 0xFFFFFFFF
  /// - **Reflect input/output**: Yes
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::checksum::{Checksum, Crc32};
  ///
  /// let crc = Crc32::checksum(b"123456789");
  /// assert_eq!(crc, 0xCBF4_3926);
  /// ```
  pub struct Crc32;
  word: u32,
  output_size: 4,
  spec: Spec::new(32, 0x04C1_1DB7, true, 0xFFFF_FFFF, 0xFFFF_FFFF),
  hw: Hw::Crc32,
  buffered: BufferedCrc32,
}

/// Explicit name for the IEEE CRC-32 variant (alias of [`Crc32`]).
pub type Crc32Ieee = Crc32;

define_crc! {
  /// CRC-32C (Castagnoli) checksum.
  ///
  /// Used by iSCSI, SCTP, ext4, Btrfs, SSE4.2 `crc32`, etc.
  ///
  /// # Properties
  ///
  /// - **Polynomial**: 0x1EDC6F41 (normal), 0x82F63B78 (reflected)
  /// - **Initial value**: 0xFFFFFFFF
  /// - **Final XOR**: 0xFFFFFFFF
  /// - **Reflect input/output**: Yes
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::checksum::{Checksum, Crc32C};
  ///
  /// let crc = Crc32C::checksum(b"123456789");
  /// assert_eq!(crc, 0xE306_9283);
  /// ```
  pub struct Crc32C;
  word: u32,
  output_size: 4,
  spec: Spec::new(32, 0x1EDC_6F41, true, 0xFFFF_FFFF, 0xFFFF_FFFF),
  hw: Hw::Crc32c,
  buffered: BufferedCrc32C,
}

/// Explicit name for the Castagnoli CRC-32C variant (alias of [`Crc32C`]).
pub type Crc32Castagnoli = Crc32C;
