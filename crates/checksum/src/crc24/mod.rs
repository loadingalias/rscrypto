//! CRC-24 implementation (OpenPGP Radix-64).
//!
//! CRC-24 is used in OpenPGP ASCII armor format (RFC 4880).

mod portable;

use backend::dispatch::Selected;
use traits::{Checksum, ChecksumCombine};

use crate::{
  common::{
    combine::{Gf2Matrix24, combine_crc24_msb, generate_shift8_matrix_24_msb},
    tables::{CRC24_OPENPGP_POLY, generate_crc24_table},
  },
  dispatchers::{Crc24Dispatcher, Crc24Fn},
};

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Wrappers
// ─────────────────────────────────────────────────────────────────────────────
//
// These wrap the portable implementation to match the Crc24Fn signature.

/// Portable kernel table (pre-computed at compile time).
mod kernel_tables {
  use super::*;
  pub static OPENPGP_TABLE: [u32; 256] = generate_crc24_table(CRC24_OPENPGP_POLY);
}

/// CRC-24 (OpenPGP) portable kernel wrapper.
fn crc24_portable(crc: u32, data: &[u8]) -> u32 {
  portable::crc24_update(crc, data, &kernel_tables::OPENPGP_TABLE)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Selection
// ─────────────────────────────────────────────────────────────────────────────

/// Select the best CRC-24 kernel for the current platform.
fn select_crc24() -> Selected<Crc24Fn> {
  Selected::new("portable", crc24_portable)
}

/// Static dispatcher for CRC-24.
static CRC24_DISPATCHER: Crc24Dispatcher = Crc24Dispatcher::new(select_crc24);

// ─────────────────────────────────────────────────────────────────────────────
// CRC-24 (OpenPGP)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-24 checksum (OpenPGP Radix-64).
///
/// Used in OpenPGP ASCII armor format as defined in RFC 4880.
///
/// # Properties
///
/// - **Polynomial**: 0x864CFB
/// - **Initial value**: 0xB704CE
/// - **Final XOR**: 0x000000
/// - **Reflect input/output**: No (MSB-first)
/// - **Width**: 24 bits (stored in low 24 bits of u32)
///
/// # Example
///
/// ```ignore
/// use checksum::{Crc24, Checksum};
///
/// let crc = Crc24::checksum(b"123456789");
/// assert_eq!(crc, 0x21CF02); // "123456789" test vector
/// ```
#[derive(Clone, Default)]
pub struct Crc24 {
  state: u32,
}

impl Crc24 {
  /// Pre-computed shift-by-8 matrix for O(log n) combine.
  const SHIFT8_MATRIX: Gf2Matrix24 = generate_shift8_matrix_24_msb(CRC24_OPENPGP_POLY);

  /// Initial CRC value as defined in RFC 4880.
  const INIT: u32 = 0xB704CE;

  /// Create a hasher to resume from a previous CRC value.
  #[inline]
  #[must_use]
  pub const fn resume(crc: u32) -> Self {
    Self { state: crc & 0xFF_FFFF }
  }

  /// Get the name of the currently selected backend.
  #[must_use]
  pub fn backend_name() -> &'static str {
    CRC24_DISPATCHER.backend_name()
  }
}

impl Checksum for Crc24 {
  const OUTPUT_SIZE: usize = 3; // 24 bits = 3 bytes
  type Output = u32; // Stored in low 24 bits

  #[inline]
  fn new() -> Self {
    Self { state: Self::INIT }
  }

  #[inline]
  fn with_initial(initial: u32) -> Self {
    Self {
      state: initial & 0xFF_FFFF,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.state = CRC24_DISPATCHER.call(self.state, data);
  }

  #[inline]
  fn finalize(&self) -> u32 {
    self.state & 0xFF_FFFF
  }

  #[inline]
  fn reset(&mut self) {
    self.state = Self::INIT;
  }
}

impl ChecksumCombine for Crc24 {
  fn combine(crc_a: u32, crc_b: u32, len_b: usize) -> u32 {
    // O(log n) matrix-based combine for MSB-first CRC-24
    combine_crc24_msb(crc_a, crc_b, len_b, Self::INIT, Self::SHIFT8_MATRIX)
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
  fn test_crc24_checksum() {
    // Standard test vector for CRC-24/OPENPGP
    let crc = Crc24::checksum(TEST_DATA);
    assert_eq!(crc, 0x21CF02);
  }

  #[test]
  fn test_crc24_streaming() {
    let oneshot = Crc24::checksum(TEST_DATA);

    let mut hasher = Crc24::new();
    hasher.update(&TEST_DATA[..5]);
    hasher.update(&TEST_DATA[5..]);
    assert_eq!(hasher.finalize(), oneshot);
  }

  #[test]
  fn test_crc24_combine() {
    let data = b"hello world";
    let (a, b) = data.split_at(6);

    let crc_a = Crc24::checksum(a);
    let crc_b = Crc24::checksum(b);
    let combined = Crc24::combine(crc_a, crc_b, b.len());

    assert_eq!(combined, Crc24::checksum(data));
  }

  #[test]
  fn test_crc24_output_is_24_bits() {
    let crc = Crc24::checksum(TEST_DATA);
    assert!(crc <= 0xFF_FFFF);
  }

  #[test]
  fn test_crc24_reset() {
    let mut hasher = Crc24::new();
    hasher.update(b"some data");
    hasher.reset();
    hasher.update(TEST_DATA);
    assert_eq!(hasher.finalize(), Crc24::checksum(TEST_DATA));
  }

  #[test]
  fn test_backend_name_not_empty() {
    assert!(!Crc24::backend_name().is_empty());
  }
}
