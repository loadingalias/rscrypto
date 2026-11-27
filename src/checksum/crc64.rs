//! CRC-64/XZ and CRC-64/NVME.

use crate::checksum::crc::api::define_crc;

define_crc! {
  /// CRC-64-XZ checksum (ECMA-182).
  ///
  /// Used in XZ Utils, 7-Zip, and other compression tools.
  ///
  /// # Properties
  ///
  /// - **Polynomial**: 0x42F0E1EBA9EA3693 (normal), 0xC96C5795D7870F42 (reflected)
  /// - **Initial value**: 0xFFFFFFFFFFFFFFFF
  /// - **Final XOR**: 0xFFFFFFFFFFFFFFFF
  /// - **Reflect input/output**: Yes
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::{Checksum, Crc64};
  ///
  /// let crc = Crc64::checksum(b"123456789");
  /// assert_eq!(crc, 0x995DC9BBDF1939FA); // "123456789" test vector
  /// ```
  pub struct Crc64;
  word: u64,
  output_size: 8,
  spec: Spec::new(64, 0x42F0_E1EB_A9EA_3693, true, !0, !0),
  hw: Hw::None,
  buffered: BufferedCrc64,
}

/// Explicit name for the XZ CRC-64 variant (alias of [`Crc64`]).
pub type Crc64Xz = Crc64;

define_crc! {
  /// CRC-64-NVME checksum.
  ///
  /// Used in the NVMe specification for data integrity.
  ///
  /// # Properties
  ///
  /// - **Polynomial**: 0xAD93D23594C93659 (normal), 0x9A6C9329AC4BC9B5 (reflected)
  /// - **Initial value**: 0xFFFFFFFFFFFFFFFF
  /// - **Final XOR**: 0xFFFFFFFFFFFFFFFF
  /// - **Reflect input/output**: Yes
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::checksum::{Checksum, Crc64Nvme};
  ///
  /// let crc = Crc64Nvme::checksum(b"123456789");
  /// assert_eq!(crc, 0xAE8B14860A799888); // "123456789" test vector
  /// ```
  pub struct Crc64Nvme;
  word: u64,
  output_size: 8,
  spec: Spec::new(64, 0xAD93_D235_94C9_3659, true, !0, !0),
  hw: Hw::None,
  buffered: BufferedCrc64Nvme,
}
