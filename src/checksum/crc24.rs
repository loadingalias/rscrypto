//! CRC-24/OPENPGP.

use crate::checksum::crc::api::define_crc;

define_crc! {
  /// CRC-24/OPENPGP checksum.
  ///
  /// Used by OpenPGP (RFC 4880) for ASCII armor / Radix-64 integrity. The value is in the low 24
  /// bits of the `u32`.
  ///
  /// # Properties
  ///
  /// - **Polynomial**: 0x864CFB (normal)
  /// - **Initial value**: 0xB704CE
  /// - **Final XOR**: 0x000000
  /// - **Reflect input/output**: No
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::checksum::{Checksum, Crc24OpenPgp};
  ///
  /// let crc = Crc24OpenPgp::checksum(b"123456789");
  /// assert_eq!(crc, 0x21CF02);
  /// ```
  pub struct Crc24OpenPgp;
  word: u32,
  output_size: 3,
  spec: Spec::new(24, 0x86_4CFB, false, 0xB7_04CE, 0),
  hw: Hw::None,
  env: "RSCRYPTO_CRC24_FORCE",
  buffered: BufferedCrc24OpenPgp,
}
