//! CRC16/IBM (aka CRC-16/ARC) checksum.
//!
//! Parameters (CRC Catalogue):
//! - width: 16
//! - poly: 0x8005 (reflected: 0xA001)
//! - init: 0x0000
//! - refin/refout: true
//! - xorout: 0x0000
//!
//! # Usage
//!
//! ```
//! use checksum::Crc16Ibm;
//!
//! let crc = Crc16Ibm::checksum(b"123456789");
//! assert_eq!(crc, 0xBB3D);
//! ```

use traits::{Checksum, ChecksumCombine};

#[cfg(not(feature = "no-tables"))]
use crate::constants::crc16_ibm::TABLES;
#[cfg(not(feature = "no-tables"))]
macro_rules! table {
  ($idx:expr) => {
    TABLES.0[$idx]
  };
}

/// CRC16/IBM checksum.
///
/// This struct implements streaming CRC16 computation. CRC16/IBM is used
/// in Modbus, USB, SDLC, and many legacy protocols.
#[derive(Clone, Debug)]
pub struct Crc16Ibm {
  /// Current CRC state.
  state: u16,
  /// Initial value for reset.
  initial: u16,
}

impl Crc16Ibm {
  /// Initial value for CRC16/IBM (zero).
  const INIT: u16 = 0x0000;
  const XOR_OUT: u16 = 0x0000;

  #[inline]
  #[must_use]
  pub const fn new() -> Self {
    Self {
      state: Self::INIT,
      initial: Self::INIT,
    }
  }

  /// Create a new hasher that will resume from a previous CRC.
  #[inline]
  #[must_use]
  pub const fn resume(crc: u16) -> Self {
    Self {
      state: crc ^ Self::XOR_OUT,
      initial: crc ^ Self::XOR_OUT,
    }
  }

  /// Compute CRC16/IBM of `data` in one shot.
  #[inline]
  #[must_use]
  pub fn checksum(data: &[u8]) -> u16 {
    dispatch(Self::INIT, data) ^ Self::XOR_OUT
  }

  #[inline]
  pub fn update(&mut self, data: &[u8]) {
    self.state = dispatch(self.state, data);
  }

  #[inline]
  #[must_use]
  pub const fn finalize(&self) -> u16 {
    self.state ^ Self::XOR_OUT
  }

  #[inline]
  pub fn reset(&mut self) {
    self.state = self.initial;
  }

  #[inline]
  #[must_use]
  pub const fn state(&self) -> u16 {
    self.finalize()
  }

  /// Combine two CRC16/IBM values: `crc(A || B)` from `crc(A)`, `crc(B)`, `len(B)`.
  #[inline]
  #[must_use]
  pub fn combine(crc_a: u16, crc_b: u16, len_b: usize) -> u16 {
    crate::combine::crc16_ibm_combine(crc_a, crc_b, len_b)
  }
}

impl Default for Crc16Ibm {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl Checksum for Crc16Ibm {
  const OUTPUT_SIZE: usize = 2;
  type Output = u16;

  #[inline]
  fn new() -> Self {
    Crc16Ibm::new()
  }

  #[inline]
  fn with_initial(initial: Self::Output) -> Self {
    Self {
      state: initial ^ Self::XOR_OUT,
      initial: initial ^ Self::XOR_OUT,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    Crc16Ibm::update(self, data);
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    Crc16Ibm::finalize(self)
  }

  #[inline]
  fn reset(&mut self) {
    Crc16Ibm::reset(self);
  }

  #[inline]
  fn checksum(data: &[u8]) -> Self::Output {
    Crc16Ibm::checksum(data)
  }
}

impl ChecksumCombine for Crc16Ibm {
  #[inline]
  fn combine(crc_a: Self::Output, crc_b: Self::Output, len_b: usize) -> Self::Output {
    Crc16Ibm::combine(crc_a, crc_b, len_b)
  }
}

/// Returns the CRC16/IBM backend used by this build.
#[doc(hidden)]
#[inline]
#[must_use]
pub fn selected_backend() -> &'static str {
  #[cfg(feature = "no-tables")]
  return "portable/bitwise";

  #[cfg(not(feature = "no-tables"))]
  "portable/slicing-by-8"
}

#[cfg(feature = "std")]
impl std::io::Write for Crc16Ibm {
  #[inline]
  fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
    self.update(buf);
    Ok(buf.len())
  }

  #[inline]
  fn flush(&mut self) -> std::io::Result<()> {
    Ok(())
  }
}

#[inline]
fn dispatch(crc: u16, data: &[u8]) -> u16 {
  compute_portable(crc, data)
}

/// Compute CRC16/IBM over `data`, returning the updated *raw* CRC state.
#[inline]
#[allow(dead_code)]
fn compute_portable(crc: u16, data: &[u8]) -> u16 {
  #[cfg(feature = "no-tables")]
  {
    let mut crc = crc;
    for &b in data {
      crc = compute_byte(crc, b);
    }
    crc
  }

  #[cfg(not(feature = "no-tables"))]
  {
    let mut crc = crc;
    let mut chunks = data.chunks_exact(8);

    for chunk in chunks.by_ref() {
      let bytes: [u8; 8] = chunk.try_into().unwrap();
      let d = u64::from_le_bytes(bytes);

      // CRC16 is 2 bytes wide, so XOR the first 2 bytes of the chunk into the register.
      let lo = (crc as u64) ^ (d & 0xFFFF);
      let rest = d >> 16;

      crc = table!(7)[(lo & 0xFF) as usize]
        ^ table!(6)[((lo >> 8) & 0xFF) as usize]
        ^ table!(5)[(rest & 0xFF) as usize]
        ^ table!(4)[((rest >> 8) & 0xFF) as usize]
        ^ table!(3)[((rest >> 16) & 0xFF) as usize]
        ^ table!(2)[((rest >> 24) & 0xFF) as usize]
        ^ table!(1)[((rest >> 32) & 0xFF) as usize]
        ^ table!(0)[((rest >> 40) & 0xFF) as usize];
    }

    for &byte in chunks.remainder() {
      crc = (crc >> 8) ^ table!(0)[((crc ^ (byte as u16)) & 0xFF) as usize];
    }

    crc
  }
}

#[inline]
#[allow(dead_code)]
const fn compute_byte(crc: u16, byte: u8) -> u16 {
  #[cfg(feature = "no-tables")]
  {
    use crate::constants::crc16_ibm::POLYNOMIAL;
    let mut crc = crc ^ (byte as u16);
    let mut i = 0;
    while i < 8 {
      let mask = 0u16.wrapping_sub(crc & 1);
      crc = (crc >> 1) ^ (POLYNOMIAL & mask);
      i += 1;
    }
    crc
  }

  #[cfg(not(feature = "no-tables"))]
  {
    (crc >> 8) ^ table!(0)[((crc ^ (byte as u16)) & 0xFF) as usize]
  }
}

#[cfg(test)]
mod tests {
  extern crate std;

  use super::*;

  #[test]
  fn test_check_string() {
    assert_eq!(Crc16Ibm::checksum(b"123456789"), 0xBB3D);
  }

  #[test]
  fn test_empty() {
    assert_eq!(Crc16Ibm::checksum(b""), 0x0000);
  }

  #[test]
  fn test_zeros() {
    // CRC16/IBM of 32 zero bytes
    assert_eq!(Crc16Ibm::checksum(&[0u8; 32]), 0x0000);
  }

  #[test]
  fn test_ones() {
    // CRC16/IBM of 32 0xFF bytes
    assert_eq!(Crc16Ibm::checksum(&[0xFFu8; 32]), 0xA401);
  }

  #[test]
  fn test_incremental() {
    let mut h = Crc16Ibm::new();
    h.update(b"1234");
    h.update(b"56789");
    assert_eq!(h.finalize(), 0xBB3D);
  }

  #[test]
  fn test_resume() {
    let data = b"hello world";
    let (a, b) = data.split_at(6);

    let crc_a = Crc16Ibm::checksum(a);
    let mut h = Crc16Ibm::resume(crc_a);
    h.update(b);
    assert_eq!(h.finalize(), Crc16Ibm::checksum(data));
  }

  #[test]
  fn test_reset() {
    let mut h = Crc16Ibm::new();
    h.update(b"garbage");
    h.reset();
    h.update(b"123456789");
    assert_eq!(h.finalize(), 0xBB3D);
  }

  #[test]
  fn test_clone() {
    let mut h = Crc16Ibm::new();
    h.update(b"1234");

    let mut clone = h.clone();
    h.update(b"56789");
    clone.update(b"56789");

    assert_eq!(h.finalize(), clone.finalize());
  }

  #[test]
  fn test_trait_impl() {
    fn check_trait<T: Checksum>() {}
    fn check_combine<T: ChecksumCombine>() {}

    check_trait::<Crc16Ibm>();
    check_combine::<Crc16Ibm>();
  }
}
