//! CRC-16/X25 and CRC-16/ARC.

use crate::checksum::crc::api::define_crc;

define_crc! {
  /// CRC-16/X25 checksum (also known as IBM-SDLC).
  ///
  /// # Properties
  ///
  /// - **Polynomial**: 0x1021 (normal), 0x8408 (reflected)
  /// - **Initial value**: 0xFFFF
  /// - **Final XOR**: 0xFFFF
  /// - **Reflect input/output**: Yes
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::checksum::{Checksum, Crc16Ccitt};
  ///
  /// let crc = Crc16Ccitt::checksum(b"123456789");
  /// assert_eq!(crc, 0x906E);
  /// ```
  pub struct Crc16Ccitt;
  word: u16,
  output_size: 2,
  spec: Spec::new(16, 0x1021, true, 0xFFFF, 0xFFFF),
  hw: Hw::None,
  buffered: BufferedCrc16Ccitt,
}

define_crc! {
  /// CRC-16/ARC checksum (also known as CRC-16/IBM).
  ///
  /// # Properties
  ///
  /// - **Polynomial**: 0x8005 (normal), 0xA001 (reflected)
  /// - **Initial value**: 0x0000
  /// - **Final XOR**: 0x0000
  /// - **Reflect input/output**: Yes
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::checksum::{Checksum, Crc16Ibm};
  ///
  /// let crc = Crc16Ibm::checksum(b"123456789");
  /// assert_eq!(crc, 0xBB3D);
  /// ```
  pub struct Crc16Ibm;
  word: u16,
  output_size: 2,
  spec: Spec::new(16, 0x8005, true, 0, 0),
  hw: Hw::None,
  buffered: BufferedCrc16Ibm,
}
