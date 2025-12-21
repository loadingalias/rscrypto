//! CRC-16 implementations.
//!
//! This module provides:
//! - [`Crc16Ccitt`] - CRC-16-CCITT (polynomial 0x1021)
//! - [`Crc16Ibm`] - CRC-16-IBM/ANSI (polynomial 0x8005)

mod portable;

use backend::dispatch::Selected;
use traits::{Checksum, ChecksumCombine};

use crate::{
  common::{
    combine::{Gf2Matrix16, combine_crc16, generate_shift8_matrix_16},
    tables::{CRC16_CCITT_POLY, CRC16_IBM_POLY, generate_crc16_table},
  },
  dispatchers::{Crc16Dispatcher, Crc16Fn},
};

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Wrappers
// ─────────────────────────────────────────────────────────────────────────────
//
// These wrap the portable implementations to match the Crc16Fn signature.
// Each wrapper bakes in the appropriate polynomial table.

/// Portable kernel tables (pre-computed at compile time).
mod kernel_tables {
  use super::*;
  pub static CCITT_TABLE: [u16; 256] = generate_crc16_table(CRC16_CCITT_POLY);
  pub static IBM_TABLE: [u16; 256] = generate_crc16_table(CRC16_IBM_POLY);
}

/// CRC-16-CCITT portable kernel wrapper.
fn crc16_ccitt_portable(crc: u16, data: &[u8]) -> u16 {
  portable::crc16_update(crc, data, &kernel_tables::CCITT_TABLE)
}

/// CRC-16-IBM portable kernel wrapper.
fn crc16_ibm_portable(crc: u16, data: &[u8]) -> u16 {
  portable::crc16_update(crc, data, &kernel_tables::IBM_TABLE)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Selection
// ─────────────────────────────────────────────────────────────────────────────

/// Select the best CRC-16-CCITT kernel for the current platform.
fn select_crc16_ccitt() -> Selected<Crc16Fn> {
  Selected::new("portable", crc16_ccitt_portable)
}

/// Select the best CRC-16-IBM kernel for the current platform.
fn select_crc16_ibm() -> Selected<Crc16Fn> {
  Selected::new("portable", crc16_ibm_portable)
}

/// Static dispatcher for CRC-16-CCITT.
static CRC16_CCITT_DISPATCHER: Crc16Dispatcher = Crc16Dispatcher::new(select_crc16_ccitt);

/// Static dispatcher for CRC-16-IBM.
static CRC16_IBM_DISPATCHER: Crc16Dispatcher = Crc16Dispatcher::new(select_crc16_ibm);

// ─────────────────────────────────────────────────────────────────────────────
// CRC-16-CCITT
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-16-CCITT checksum.
///
/// Uses polynomial 0x1021 (X.25, HDLC, Bluetooth, XMODEM variant).
///
/// # Properties
///
/// - **Polynomial**: 0x1021 (normal form), 0x8408 (reflected)
/// - **Initial value**: 0xFFFF
/// - **Final XOR**: 0xFFFF
/// - **Reflect input/output**: Yes
///
/// # Example
///
/// ```ignore
/// use checksum::{Crc16Ccitt, Checksum};
///
/// let crc = Crc16Ccitt::checksum(b"123456789");
/// assert_eq!(crc, 0x906E); // "123456789" test vector
/// ```
#[derive(Clone, Default)]
pub struct Crc16Ccitt {
  state: u16,
}

impl Crc16Ccitt {
  /// Pre-computed shift-by-8 matrix for combine.
  const SHIFT8_MATRIX: Gf2Matrix16 = generate_shift8_matrix_16(CRC16_CCITT_POLY);

  /// Create a hasher to resume from a previous CRC value.
  ///
  /// Use this to continue computation from a previously computed CRC.
  #[inline]
  #[must_use]
  pub const fn resume(crc: u16) -> Self {
    Self { state: crc ^ 0xFFFF }
  }

  /// Get the name of the currently selected backend.
  #[must_use]
  pub fn backend_name() -> &'static str {
    CRC16_CCITT_DISPATCHER.backend_name()
  }
}

impl Checksum for Crc16Ccitt {
  const OUTPUT_SIZE: usize = 2;
  type Output = u16;

  #[inline]
  fn new() -> Self {
    Self { state: 0xFFFF }
  }

  #[inline]
  fn with_initial(initial: u16) -> Self {
    Self {
      state: initial ^ 0xFFFF,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.state = CRC16_CCITT_DISPATCHER.call(self.state, data);
  }

  #[inline]
  fn finalize(&self) -> u16 {
    self.state ^ 0xFFFF
  }

  #[inline]
  fn reset(&mut self) {
    self.state = 0xFFFF;
  }
}

impl ChecksumCombine for Crc16Ccitt {
  fn combine(crc_a: u16, crc_b: u16, len_b: usize) -> u16 {
    // For CRCs with init=final_xor (like 0xFFFF/0xFFFF), the combine
    // can be computed directly on finalized values.
    // result = shift(crc_a, len_b) XOR crc_b
    combine_crc16(crc_a, crc_b, len_b, Self::SHIFT8_MATRIX)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-16-IBM
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-16-IBM/ANSI checksum.
///
/// Uses polynomial 0x8005 (USB, Modbus, many protocols).
///
/// # Properties
///
/// - **Polynomial**: 0x8005 (normal form), 0xA001 (reflected)
/// - **Initial value**: 0x0000
/// - **Final XOR**: 0x0000
/// - **Reflect input/output**: Yes
///
/// # Example
///
/// ```ignore
/// use checksum::{Crc16Ibm, Checksum};
///
/// let crc = Crc16Ibm::checksum(b"123456789");
/// assert_eq!(crc, 0xBB3D); // "123456789" test vector
/// ```
#[derive(Clone, Default)]
pub struct Crc16Ibm {
  state: u16,
}

impl Crc16Ibm {
  /// Pre-computed shift-by-8 matrix for combine.
  const SHIFT8_MATRIX: Gf2Matrix16 = generate_shift8_matrix_16(CRC16_IBM_POLY);

  /// Create a hasher to resume from a previous CRC value.
  #[inline]
  #[must_use]
  pub const fn resume(crc: u16) -> Self {
    Self { state: crc }
  }

  /// Get the name of the currently selected backend.
  #[must_use]
  pub fn backend_name() -> &'static str {
    CRC16_IBM_DISPATCHER.backend_name()
  }
}

impl Checksum for Crc16Ibm {
  const OUTPUT_SIZE: usize = 2;
  type Output = u16;

  #[inline]
  fn new() -> Self {
    Self { state: 0 }
  }

  #[inline]
  fn with_initial(initial: u16) -> Self {
    Self { state: initial }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.state = CRC16_IBM_DISPATCHER.call(self.state, data);
  }

  #[inline]
  fn finalize(&self) -> u16 {
    self.state
  }

  #[inline]
  fn reset(&mut self) {
    self.state = 0;
  }
}

impl ChecksumCombine for Crc16Ibm {
  fn combine(crc_a: u16, crc_b: u16, len_b: usize) -> u16 {
    combine_crc16(crc_a, crc_b, len_b, Self::SHIFT8_MATRIX)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  const TEST_DATA: &[u8] = b"123456789";

  #[test]
  fn test_crc16_ccitt_checksum() {
    // Standard test vector for CRC-16-CCITT (X.25 / HDLC)
    // Note: Different variants have different init/xor values
    let crc = Crc16Ccitt::checksum(TEST_DATA);
    // CRC-16/X-25 check value
    assert_eq!(crc, 0x906E);
  }

  #[test]
  fn test_crc16_ibm_checksum() {
    // Standard test vector for CRC-16-IBM (CRC-16/ARC)
    let crc = Crc16Ibm::checksum(TEST_DATA);
    assert_eq!(crc, 0xBB3D);
  }

  #[test]
  fn test_crc16_ccitt_streaming() {
    let oneshot = Crc16Ccitt::checksum(TEST_DATA);

    let mut hasher = Crc16Ccitt::new();
    hasher.update(&TEST_DATA[..5]);
    hasher.update(&TEST_DATA[5..]);
    assert_eq!(hasher.finalize(), oneshot);
  }

  #[test]
  fn test_crc16_ibm_streaming() {
    let oneshot = Crc16Ibm::checksum(TEST_DATA);

    let mut hasher = Crc16Ibm::new();
    hasher.update(&TEST_DATA[..3]);
    hasher.update(&TEST_DATA[3..7]);
    hasher.update(&TEST_DATA[7..]);
    assert_eq!(hasher.finalize(), oneshot);
  }

  #[test]
  fn test_crc16_ccitt_combine() {
    let data = b"hello world";
    let (a, b) = data.split_at(6);

    let crc_a = Crc16Ccitt::checksum(a);
    let crc_b = Crc16Ccitt::checksum(b);
    let combined = Crc16Ccitt::combine(crc_a, crc_b, b.len());

    assert_eq!(combined, Crc16Ccitt::checksum(data));
  }

  #[test]
  fn test_crc16_ibm_combine() {
    let data = b"hello world";
    let (a, b) = data.split_at(6);

    let crc_a = Crc16Ibm::checksum(a);
    let crc_b = Crc16Ibm::checksum(b);
    let combined = Crc16Ibm::combine(crc_a, crc_b, b.len());

    assert_eq!(combined, Crc16Ibm::checksum(data));
  }

  #[test]
  fn test_crc16_empty() {
    // CRC of empty data with init=0xFFFF and final XOR=0xFFFF should be 0
    // But actually: init ^ final_xor when no data processed
    let crc = Crc16Ccitt::checksum(&[]);
    assert_eq!(crc, 0); // 0xFFFF ^ 0xFFFF = 0

    let crc = Crc16Ibm::checksum(&[]);
    assert_eq!(crc, 0); // init=0, no final xor
  }

  #[test]
  fn test_crc16_reset() {
    let mut hasher = Crc16Ccitt::new();
    hasher.update(b"some data");
    hasher.reset();
    hasher.update(TEST_DATA);
    assert_eq!(hasher.finalize(), Crc16Ccitt::checksum(TEST_DATA));
  }

  #[test]
  fn test_backend_name_not_empty() {
    assert!(!Crc16Ccitt::backend_name().is_empty());
    assert!(!Crc16Ibm::backend_name().is_empty());
  }
}
