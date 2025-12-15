//! CRC24/OpenPGP checksum.
//!
//! Parameters (CRC Catalogue):
//! - width: 24
//! - poly: 0x864CFB
//! - init: 0xB704CE
//! - refin/refout: false
//! - xorout: 0x000000

use traits::{Checksum, ChecksumCombine};

#[cfg(not(feature = "no-tables"))]
use crate::constants::crc24_openpgp::TABLE;

/// CRC24/OpenPGP checksum.
///
/// This checksum is 24 bits wide. The returned `u32` only uses the lower 24 bits.
#[derive(Clone, Debug)]
pub struct Crc24 {
  /// Current CRC state (24-bit value in lower bits).
  state: u32,
  /// Initial value for reset.
  initial: u32,
}

impl Crc24 {
  const WIDTH_MASK: u32 = 0xFF_FFFF;
  /// Initial value for CRC24/OpenPGP.
  const INIT: u32 = 0xB7_04_CE;
  const XOR_OUT: u32 = 0x00_00_00;

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
  pub const fn resume(crc: u32) -> Self {
    let s = (crc ^ Self::XOR_OUT) & Self::WIDTH_MASK;
    Self { state: s, initial: s }
  }

  /// Compute CRC24/OpenPGP of `data` in one shot.
  #[inline]
  #[must_use]
  pub fn checksum(data: &[u8]) -> u32 {
    (dispatch(Self::INIT, data) ^ Self::XOR_OUT) & Self::WIDTH_MASK
  }

  #[inline]
  pub fn update(&mut self, data: &[u8]) {
    self.state = dispatch(self.state, data);
  }

  #[inline]
  #[must_use]
  pub const fn finalize(&self) -> u32 {
    (self.state ^ Self::XOR_OUT) & Self::WIDTH_MASK
  }

  #[inline]
  pub fn reset(&mut self) {
    self.state = self.initial;
  }

  #[inline]
  #[must_use]
  pub const fn state(&self) -> u32 {
    self.finalize()
  }

  /// Combine two CRC24/OpenPGP values: `crc(A || B)` from `crc(A)`, `crc(B)`, `len(B)`.
  #[inline]
  #[must_use]
  pub fn combine(crc_a: u32, crc_b: u32, len_b: usize) -> u32 {
    crate::combine::crc24_openpgp_combine(crc_a, crc_b, len_b)
  }
}

impl Default for Crc24 {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl Checksum for Crc24 {
  const OUTPUT_SIZE: usize = 3;
  type Output = u32;

  #[inline]
  fn new() -> Self {
    Crc24::new()
  }

  #[inline]
  fn with_initial(initial: Self::Output) -> Self {
    let s = (initial ^ Self::XOR_OUT) & Self::WIDTH_MASK;
    Self { state: s, initial: s }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    Crc24::update(self, data);
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    Crc24::finalize(self)
  }

  #[inline]
  fn reset(&mut self) {
    Crc24::reset(self);
  }

  #[inline]
  fn checksum(data: &[u8]) -> Self::Output {
    Crc24::checksum(data)
  }
}

impl ChecksumCombine for Crc24 {
  #[inline]
  fn combine(crc_a: Self::Output, crc_b: Self::Output, len_b: usize) -> Self::Output {
    Crc24::combine(crc_a, crc_b, len_b)
  }
}

#[inline]
#[must_use]
pub(super) fn selected_backend() -> &'static str {
  #[cfg(feature = "no-tables")]
  return "portable/bitwise";

  #[cfg(not(feature = "no-tables"))]
  "portable/table"
}

#[cfg(feature = "std")]
impl std::io::Write for Crc24 {
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
fn dispatch(crc: u32, data: &[u8]) -> u32 {
  compute_portable(crc, data)
}

#[inline]
#[allow(dead_code)]
fn compute_portable(crc: u32, data: &[u8]) -> u32 {
  #[cfg(feature = "no-tables")]
  {
    let mut crc = crc & Crc24::WIDTH_MASK;
    for &b in data {
      crc = compute_byte(crc, b);
    }
    crc
  }

  #[cfg(not(feature = "no-tables"))]
  {
    let mut crc = crc & Crc24::WIDTH_MASK;
    for &byte in data {
      let idx = ((crc >> 16) ^ u32::from(byte)) & 0xFF;
      crc = ((crc << 8) & Crc24::WIDTH_MASK) ^ TABLE[idx as usize];
    }
    crc
  }
}

#[inline]
#[allow(dead_code)]
const fn compute_byte(crc: u32, byte: u8) -> u32 {
  use crate::constants::crc24_openpgp::POLYNOMIAL;

  let mut crc = (crc & Crc24::WIDTH_MASK) ^ ((byte as u32) << 16);
  let mut i = 0;
  while i < 8 {
    if (crc & 0x80_0000) != 0 {
      crc = (crc << 1) ^ POLYNOMIAL;
    } else {
      crc <<= 1;
    }
    crc &= Crc24::WIDTH_MASK;
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
    assert_eq!(Crc24::checksum(b"123456789"), 0x21_CF_02);
  }

  #[test]
  fn test_empty() {
    assert_eq!(Crc24::checksum(b""), 0xB7_04_CE);
  }

  #[test]
  fn test_zeros() {
    // CRC24/OpenPGP of 32 zero bytes
    assert_eq!(Crc24::checksum(&[0u8; 32]), 0x77_B8_50);
  }

  #[test]
  fn test_ones() {
    // CRC24/OpenPGP of 32 0xFF bytes
    assert_eq!(Crc24::checksum(&[0xFFu8; 32]), 0x28_B9_F1);
  }

  #[test]
  fn test_incremental() {
    let mut h = Crc24::new();
    h.update(b"1234");
    h.update(b"56789");
    assert_eq!(h.finalize(), 0x21_CF_02);
  }

  #[test]
  fn test_resume() {
    let data = b"hello world";
    let (a, b) = data.split_at(6);

    let crc_a = Crc24::checksum(a);
    let mut h = Crc24::resume(crc_a);
    h.update(b);
    assert_eq!(h.finalize(), Crc24::checksum(data));
  }

  #[test]
  fn test_reset() {
    let mut h = Crc24::new();
    h.update(b"garbage");
    h.reset();
    h.update(b"123456789");
    assert_eq!(h.finalize(), 0x21_CF_02);
  }

  #[test]
  fn test_clone() {
    let mut h = Crc24::new();
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

    check_trait::<Crc24>();
    check_combine::<Crc24>();
  }

  #[test]
  fn test_24bit_mask() {
    // Verify results are always 24-bit (lower 24 bits of u32)
    let crc = Crc24::checksum(b"test");
    assert_eq!(crc & 0xFF_00_00_00, 0, "CRC24 should be 24-bit only");
  }
}
