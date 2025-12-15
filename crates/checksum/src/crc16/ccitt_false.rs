//! CRC16/CCITT-FALSE checksum.
//!
//! Parameters (CRC Catalogue):
//! - width: 16
//! - poly: 0x1021
//! - init: 0xFFFF
//! - refin/refout: false
//! - xorout: 0x0000

use traits::{Checksum, ChecksumCombine};

#[cfg(not(feature = "no-tables"))]
use crate::constants::crc16_ccitt_false::TABLE;

/// CRC16/CCITT-FALSE checksum.
#[derive(Clone, Debug)]
pub struct Crc16CcittFalse {
  /// Current CRC state.
  state: u16,
  /// Initial value for reset.
  initial: u16,
}

impl Crc16CcittFalse {
  /// Initial value for CRC16/CCITT-FALSE (all ones).
  const INIT: u16 = 0xFFFF;
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

  /// Compute CRC16/CCITT-FALSE of `data` in one shot.
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

  /// Combine two CRC16/CCITT-FALSE values: `crc(A || B)` from `crc(A)`, `crc(B)`, `len(B)`.
  #[inline]
  #[must_use]
  pub fn combine(crc_a: u16, crc_b: u16, len_b: usize) -> u16 {
    crate::combine::crc16_ccitt_false_combine(crc_a, crc_b, len_b)
  }
}

impl Default for Crc16CcittFalse {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl Checksum for Crc16CcittFalse {
  const OUTPUT_SIZE: usize = 2;
  type Output = u16;

  #[inline]
  fn new() -> Self {
    Crc16CcittFalse::new()
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
    Crc16CcittFalse::update(self, data);
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    Crc16CcittFalse::finalize(self)
  }

  #[inline]
  fn reset(&mut self) {
    Crc16CcittFalse::reset(self);
  }

  #[inline]
  fn checksum(data: &[u8]) -> Self::Output {
    Crc16CcittFalse::checksum(data)
  }
}

impl ChecksumCombine for Crc16CcittFalse {
  #[inline]
  fn combine(crc_a: Self::Output, crc_b: Self::Output, len_b: usize) -> Self::Output {
    Crc16CcittFalse::combine(crc_a, crc_b, len_b)
  }
}

/// Returns the CRC16/CCITT-FALSE backend used by this build.
#[doc(hidden)]
#[inline]
#[must_use]
pub fn selected_backend() -> &'static str {
  #[cfg(feature = "no-tables")]
  return "portable/bitwise";

  #[cfg(not(feature = "no-tables"))]
  "portable/table"
}

#[cfg(feature = "std")]
impl std::io::Write for Crc16CcittFalse {
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

/// Compute CRC16/CCITT-FALSE over `data`, returning the updated *raw* CRC state.
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
    for &byte in data {
      let idx = ((crc >> 8) ^ u16::from(byte)) & 0xFF;
      crc = (crc << 8) ^ TABLE[idx as usize];
    }
    crc
  }
}

#[inline]
#[allow(dead_code)]
const fn compute_byte(crc: u16, byte: u8) -> u16 {
  use crate::constants::crc16_ccitt_false::POLYNOMIAL;

  let mut crc = crc ^ ((byte as u16) << 8);
  let mut i = 0;
  while i < 8 {
    if (crc & 0x8000) != 0 {
      crc = (crc << 1) ^ POLYNOMIAL;
    } else {
      crc <<= 1;
    }
    i += 1;
  }
  crc
}

#[cfg(test)]
mod tests {
  extern crate std;

  use super::*;

  #[test]
  fn test_check_string() {
    assert_eq!(Crc16CcittFalse::checksum(b"123456789"), 0x29B1);
  }

  #[test]
  fn test_empty() {
    assert_eq!(Crc16CcittFalse::checksum(b""), 0xFFFF);
  }

  #[test]
  fn test_zeros() {
    // CRC16/CCITT-FALSE of 32 zero bytes
    assert_eq!(Crc16CcittFalse::checksum(&[0u8; 32]), 0xF14C);
  }

  #[test]
  fn test_ones() {
    // CRC16/CCITT-FALSE of 32 0xFF bytes
    assert_eq!(Crc16CcittFalse::checksum(&[0xFFu8; 32]), 0x75F8);
  }

  #[test]
  fn test_incremental() {
    let mut h = Crc16CcittFalse::new();
    h.update(b"1234");
    h.update(b"56789");
    assert_eq!(h.finalize(), 0x29B1);
  }

  #[test]
  fn test_resume() {
    let data = b"hello world";
    let (a, b) = data.split_at(6);

    let crc_a = Crc16CcittFalse::checksum(a);
    let mut h = Crc16CcittFalse::resume(crc_a);
    h.update(b);
    assert_eq!(h.finalize(), Crc16CcittFalse::checksum(data));
  }

  #[test]
  fn test_reset() {
    let mut h = Crc16CcittFalse::new();
    h.update(b"garbage");
    h.reset();
    h.update(b"123456789");
    assert_eq!(h.finalize(), 0x29B1);
  }

  #[test]
  fn test_clone() {
    let mut h = Crc16CcittFalse::new();
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

    check_trait::<Crc16CcittFalse>();
    check_combine::<Crc16CcittFalse>();
  }
}
