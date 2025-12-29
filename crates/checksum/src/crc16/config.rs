//! CRC-16 configuration and tunables.
//!
//! CRC-16 is currently portable-only. This module exists to keep API parity
//! with CRC-32/CRC-64 and to allow future tuning without breaking changes.

/// Tunables for CRC-16 portable kernel selection.
#[derive(Clone, Copy, Debug)]
pub struct Crc16Tunables {
  /// Minimum `len` in bytes to use slice-by-8 (otherwise slice-by-4).
  pub slice4_to_slice8: usize,
}

impl Default for Crc16Tunables {
  fn default() -> Self {
    Self { slice4_to_slice8: 64 }
  }
}

/// Effective CRC-16 configuration.
#[derive(Clone, Copy, Debug)]
#[derive(Default)]
pub struct Crc16Config {
  /// Active tunables.
  pub tunables: Crc16Tunables,
}


#[inline]
#[must_use]
pub(crate) fn get() -> Crc16Config {
  Crc16Config::default()
}
