//! CRC-24 configuration and tunables.
//!
//! CRC-24 is currently portable-only. This module exists to keep API parity
//! with CRC-32/CRC-64 and to allow future tuning without breaking changes.

/// Tunables for CRC-24 portable kernel selection.
#[derive(Clone, Copy, Debug)]
pub struct Crc24Tunables {
  /// Minimum `len` in bytes to use slice-by-8 (otherwise slice-by-4).
  pub slice4_to_slice8: usize,
}

impl Default for Crc24Tunables {
  fn default() -> Self {
    Self { slice4_to_slice8: 64 }
  }
}

/// Effective CRC-24 configuration.
#[derive(Clone, Copy, Debug)]
#[derive(Default)]
pub struct Crc24Config {
  /// Active tunables.
  pub tunables: Crc24Tunables,
}


#[inline]
#[must_use]
pub(crate) fn get() -> Crc24Config {
  Crc24Config::default()
}
