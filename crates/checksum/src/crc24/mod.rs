//! CRC-24 implementations (portable-only, initial release).
//!
//! This module provides:
//! - [`Crc24OpenPgp`] - CRC-24/OPENPGP (RFC 4880)
//!
//! # Quick Start
//!
//! ```rust
//! use checksum::{Checksum, ChecksumCombine, Crc24OpenPgp};
//!
//! let data = b"123456789";
//! assert_eq!(Crc24OpenPgp::checksum(data), 0x21CF02);
//!
//! let (a, b) = data.split_at(4);
//! let combined = Crc24OpenPgp::combine(
//!   Crc24OpenPgp::checksum(a),
//!   Crc24OpenPgp::checksum(b),
//!   b.len(),
//! );
//! assert_eq!(combined, Crc24OpenPgp::checksum(data));
//! ```

pub(crate) mod config;
mod kernels;
mod portable;
mod tuned_defaults;

use backend::dispatch::Selected;
#[allow(unused_imports)]
pub use config::{Crc24Config, Crc24Force, Crc24Tunables};
// Re-export traits for test modules (`use super::*`).
#[allow(unused_imports)]
pub(super) use traits::{Checksum, ChecksumCombine};

use crate::{
  common::{
    combine::{Gf2Matrix24, combine_crc24, generate_shift8_matrix_24},
    tables::{CRC24_OPENPGP_POLY, generate_crc24_tables_4, generate_crc24_tables_8},
  },
  dispatchers::{Crc24Dispatcher, Crc24Fn},
};

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Tables (compile-time)
// ─────────────────────────────────────────────────────────────────────────────

mod kernel_tables {
  use super::*;
  pub static OPENPGP_TABLES_4: [[u32; 256]; 4] = generate_crc24_tables_4(CRC24_OPENPGP_POLY);
  pub static OPENPGP_TABLES_8: [[u32; 256]; 8] = generate_crc24_tables_8(CRC24_OPENPGP_POLY);
}

// ─────────────────────────────────────────────────────────────────────────────
// Portable Kernel Wrapper (auto selection)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn crc24_openpgp_portable_auto(crc: u32, data: &[u8]) -> u32 {
  if data.len() < crc24_slice4_to_slice8() {
    portable::crc24_openpgp_slice4(crc, data)
  } else {
    portable::crc24_openpgp_slice8(crc, data)
  }
}

#[cfg(feature = "std")]
use std::sync::OnceLock;

#[cfg(feature = "std")]
static CRC24_SLICE4_TO_SLICE8: OnceLock<usize> = OnceLock::new();

#[inline]
#[must_use]
fn crc24_slice4_to_slice8() -> usize {
  #[cfg(feature = "std")]
  {
    *CRC24_SLICE4_TO_SLICE8.get_or_init(|| config::get().tunables.slice4_to_slice8)
  }
  #[cfg(not(feature = "std"))]
  {
    config::get().tunables.slice4_to_slice8
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Introspection (portable-only)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[must_use]
pub(crate) fn crc24_selected_kernel_name(len: usize) -> &'static str {
  let cfg = config::get();
  match cfg.effective_force {
    config::Crc24Force::Slice4 => kernels::PORTABLE_SLICE4,
    config::Crc24Force::Slice8 => kernels::PORTABLE_SLICE8,
    _ => kernels::portable_name_for_len(len, cfg.tunables.slice4_to_slice8),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Selection (portable-only)
// ─────────────────────────────────────────────────────────────────────────────

fn select_crc24_openpgp() -> Selected<Crc24Fn> {
  let cfg = config::get();
  #[cfg(feature = "std")]
  let _ = CRC24_SLICE4_TO_SLICE8.get_or_init(|| cfg.tunables.slice4_to_slice8);

  match cfg.effective_force {
    config::Crc24Force::Slice4 => Selected::new(kernels::PORTABLE_SLICE4, portable::crc24_openpgp_slice4),
    config::Crc24Force::Slice8 => Selected::new(kernels::PORTABLE_SLICE8, portable::crc24_openpgp_slice8),
    _ => Selected::new(kernels::PORTABLE_AUTO, crc24_openpgp_portable_auto),
  }
}

static CRC24_OPENPGP_DISPATCHER: Crc24Dispatcher = Crc24Dispatcher::new(select_crc24_openpgp);

// ─────────────────────────────────────────────────────────────────────────────
// CRC-24 Types
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-24/OPENPGP checksum.
///
/// Used by OpenPGP (RFC 4880) for ASCII armor / Radix-64 integrity.
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
/// use checksum::{Checksum, Crc24OpenPgp};
///
/// let crc = Crc24OpenPgp::checksum(b"123456789");
/// assert_eq!(crc, 0x21CF02);
/// ```
#[derive(Clone)]
pub struct Crc24OpenPgp {
  state: u32,
  kernel: Crc24Fn,
  initialized: bool,
}

impl Crc24OpenPgp {
  const MASK: u32 = 0x00FF_FFFF;
  const INIT: u32 = 0x00B7_04CE;
  const XOROUT: u32 = 0x0000_0000;
  const INIT_XOROUT: u32 = Self::INIT ^ Self::XOROUT;

  /// Pre-computed shift-by-8 matrix for combine.
  const SHIFT8_MATRIX: Gf2Matrix24 = generate_shift8_matrix_24(CRC24_OPENPGP_POLY);

  /// Create a hasher to resume from a previous CRC value.
  #[inline]
  #[must_use]
  pub const fn resume(crc: u32) -> Self {
    Self {
      state: (crc ^ Self::XOROUT) & Self::MASK,
      kernel: crc24_openpgp_portable_auto,
      initialized: false,
    }
  }

  /// Get the name of the currently selected backend.
  #[must_use]
  pub fn backend_name() -> &'static str {
    CRC24_OPENPGP_DISPATCHER.backend_name()
  }

  /// Get the effective CRC-24 configuration.
  #[must_use]
  pub fn config() -> Crc24Config {
    config::get()
  }

  /// Convenience accessor for the active CRC-24 tunables.
  #[must_use]
  pub fn tunables() -> Crc24Tunables {
    Self::config().tunables
  }

  /// Returns the kernel name that the selector would choose for `len`.
  #[must_use]
  pub fn kernel_name_for_len(len: usize) -> &'static str {
    crc24_selected_kernel_name(len)
  }
}

impl traits::Checksum for Crc24OpenPgp {
  const OUTPUT_SIZE: usize = 3;
  type Output = u32;

  #[inline]
  fn new() -> Self {
    Self {
      state: Self::INIT,
      kernel: CRC24_OPENPGP_DISPATCHER.kernel(),
      initialized: true,
    }
  }

  #[inline]
  fn with_initial(initial: u32) -> Self {
    Self {
      state: (initial ^ Self::XOROUT) & Self::MASK,
      kernel: CRC24_OPENPGP_DISPATCHER.kernel(),
      initialized: true,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    if !self.initialized {
      self.kernel = CRC24_OPENPGP_DISPATCHER.kernel();
      self.initialized = true;
    }
    self.state = (self.kernel)(self.state, data) & Self::MASK;
  }

  #[inline]
  fn finalize(&self) -> u32 {
    (self.state ^ Self::XOROUT) & Self::MASK
  }

  #[inline]
  fn reset(&mut self) {
    self.state = Self::INIT;
  }
}

impl Default for Crc24OpenPgp {
  fn default() -> Self {
    <Self as traits::Checksum>::new()
  }
}

impl traits::ChecksumCombine for Crc24OpenPgp {
  fn combine(crc_a: u32, crc_b: u32, len_b: usize) -> u32 {
    combine_crc24(crc_a, crc_b, len_b, Self::SHIFT8_MATRIX, Self::INIT_XOROUT)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffered CRC-24 Wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Buffer size for buffered CRC-24 wrappers.
#[cfg(feature = "alloc")]
const BUFFERED_CRC24_BUFFER_SIZE: usize = 256;

#[cfg(feature = "alloc")]
#[inline]
#[must_use]
fn crc24_buffered_threshold() -> usize {
  config::get().tunables.slice4_to_slice8.max(64)
}

#[cfg(feature = "alloc")]
define_buffered_crc! {
  /// A buffering wrapper around [`Crc24OpenPgp`] for streaming small chunks.
  pub struct BufferedCrc24OpenPgp<Crc24OpenPgp> {
    buffer_size: BUFFERED_CRC24_BUFFER_SIZE,
    threshold_fn: crc24_buffered_threshold,
  }
}

// Proptest uses file I/O for failure persistence that Miri cannot interpret.
#[cfg(all(test, not(miri)))]
mod proptests;

#[cfg(test)]
mod tests {
  extern crate std;

  use super::*;

  #[test]
  fn test_vectors_crc24_openpgp() {
    assert_eq!(Crc24OpenPgp::checksum(b"123456789"), 0x0021_CF02);
    assert_eq!(Crc24OpenPgp::checksum(b""), 0x00B7_04CE);
  }

  #[test]
  fn test_combine_all_splits_openpgp() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let full = Crc24OpenPgp::checksum(data);
    for split in 0..=data.len() {
      let (a, b) = data.split_at(split);
      let combined = Crc24OpenPgp::combine(Crc24OpenPgp::checksum(a), Crc24OpenPgp::checksum(b), b.len());
      assert_eq!(combined, full, "split={split}");
    }
  }

  #[cfg(feature = "alloc")]
  #[test]
  fn test_buffered_openpgp_matches_unbuffered() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let expected = Crc24OpenPgp::checksum(data);

    let mut buffered = BufferedCrc24OpenPgp::new();
    for chunk in data.chunks(3) {
      buffered.update(chunk);
    }
    assert_eq!(buffered.finalize(), expected);
  }
}
