//! CRC-16 implementations (portable-only, initial release).
//!
//! This module provides:
//! - [`Crc16Ccitt`] - CRC-16/X25 (also known as IBM-SDLC)
//! - [`Crc16Ibm`] - CRC-16/ARC (also known as CRC-16/IBM)
//!
//! # Quick Start
//!
//! ```rust
//! use checksum::{Checksum, ChecksumCombine, Crc16Ccitt, Crc16Ibm};
//!
//! let data = b"123456789";
//! assert_eq!(Crc16Ccitt::checksum(data), 0x906E);
//! assert_eq!(Crc16Ibm::checksum(data), 0xBB3D);
//!
//! let (a, b) = data.split_at(4);
//! let combined = Crc16Ccitt::combine(Crc16Ccitt::checksum(a), Crc16Ccitt::checksum(b), b.len());
//! assert_eq!(combined, Crc16Ccitt::checksum(data));
//! ```

pub(crate) mod config;
mod kernels;
mod portable;
mod tuned_defaults;

use backend::dispatch::Selected;
#[allow(unused_imports)]
pub use config::{Crc16Config, Crc16Force, Crc16Tunables};
// Re-export traits for test modules (`use super::*`).
#[allow(unused_imports)]
pub(super) use traits::{Checksum, ChecksumCombine};

use crate::{
  common::{
    combine::{Gf2Matrix16, combine_crc16, generate_shift8_matrix_16},
    tables::{CRC16_CCITT_POLY, CRC16_IBM_POLY, generate_crc16_tables_4, generate_crc16_tables_8},
  },
  dispatchers::{Crc16Dispatcher, Crc16Fn},
};

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Tables (compile-time)
// ─────────────────────────────────────────────────────────────────────────────

mod kernel_tables {
  use super::*;

  pub static CCITT_TABLES_4: [[u16; 256]; 4] = generate_crc16_tables_4(CRC16_CCITT_POLY);
  pub static CCITT_TABLES_8: [[u16; 256]; 8] = generate_crc16_tables_8(CRC16_CCITT_POLY);

  pub static IBM_TABLES_4: [[u16; 256]; 4] = generate_crc16_tables_4(CRC16_IBM_POLY);
  pub static IBM_TABLES_8: [[u16; 256]; 8] = generate_crc16_tables_8(CRC16_IBM_POLY);
}

// ─────────────────────────────────────────────────────────────────────────────
// Portable Kernel Wrappers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn crc16_ccitt_portable_auto(crc: u16, data: &[u8]) -> u16 {
  if data.len() < crc16_slice4_to_slice8() {
    portable::crc16_ccitt_slice4(crc, data)
  } else {
    portable::crc16_ccitt_slice8(crc, data)
  }
}

#[inline]
fn crc16_ibm_portable_auto(crc: u16, data: &[u8]) -> u16 {
  if data.len() < crc16_slice4_to_slice8() {
    portable::crc16_ibm_slice4(crc, data)
  } else {
    portable::crc16_ibm_slice8(crc, data)
  }
}

#[cfg(feature = "std")]
use std::sync::OnceLock;

#[cfg(feature = "std")]
static CRC16_SLICE4_TO_SLICE8: OnceLock<usize> = OnceLock::new();

#[inline]
#[must_use]
fn crc16_slice4_to_slice8() -> usize {
  #[cfg(feature = "std")]
  {
    *CRC16_SLICE4_TO_SLICE8.get_or_init(|| config::get().tunables.slice4_to_slice8)
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
pub(crate) fn crc16_selected_kernel_name(len: usize) -> &'static str {
  let cfg = config::get();
  match cfg.effective_force {
    config::Crc16Force::Slice4 => kernels::PORTABLE_SLICE4,
    config::Crc16Force::Slice8 => kernels::PORTABLE_SLICE8,
    _ => kernels::portable_name_for_len(len, cfg.tunables.slice4_to_slice8),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Selection (portable-only)
// ─────────────────────────────────────────────────────────────────────────────

fn select_crc16_ccitt() -> Selected<Crc16Fn> {
  let cfg = config::get();
  #[cfg(feature = "std")]
  let _ = CRC16_SLICE4_TO_SLICE8.get_or_init(|| cfg.tunables.slice4_to_slice8);

  match cfg.effective_force {
    config::Crc16Force::Slice4 => Selected::new(kernels::PORTABLE_SLICE4, portable::crc16_ccitt_slice4),
    config::Crc16Force::Slice8 => Selected::new(kernels::PORTABLE_SLICE8, portable::crc16_ccitt_slice8),
    _ => Selected::new(kernels::PORTABLE_AUTO, crc16_ccitt_portable_auto),
  }
}

fn select_crc16_ibm() -> Selected<Crc16Fn> {
  let cfg = config::get();
  #[cfg(feature = "std")]
  let _ = CRC16_SLICE4_TO_SLICE8.get_or_init(|| cfg.tunables.slice4_to_slice8);

  match cfg.effective_force {
    config::Crc16Force::Slice4 => Selected::new(kernels::PORTABLE_SLICE4, portable::crc16_ibm_slice4),
    config::Crc16Force::Slice8 => Selected::new(kernels::PORTABLE_SLICE8, portable::crc16_ibm_slice8),
    _ => Selected::new(kernels::PORTABLE_AUTO, crc16_ibm_portable_auto),
  }
}

static CRC16_CCITT_DISPATCHER: Crc16Dispatcher = Crc16Dispatcher::new(select_crc16_ccitt);
static CRC16_IBM_DISPATCHER: Crc16Dispatcher = Crc16Dispatcher::new(select_crc16_ibm);

// ─────────────────────────────────────────────────────────────────────────────
// CRC-16 Types
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-16/X25 checksum (also known as IBM-SDLC).
///
/// # Properties
///
/// - **Polynomial**: 0x1021 (normal), 0x8408 (reflected)
/// - **Initial value**: 0xFFFF
/// - **Final XOR**: 0xFFFF
/// - **Reflect input/output**: Yes
///
/// # Examples
///
/// ```rust
/// use checksum::{Checksum, Crc16Ccitt};
///
/// let crc = Crc16Ccitt::checksum(b"123456789");
/// assert_eq!(crc, 0x906E);
/// ```
#[derive(Clone)]
pub struct Crc16Ccitt {
  state: u16,
  kernel: Crc16Fn,
  initialized: bool,
}

impl Crc16Ccitt {
  const INIT: u16 = 0xFFFF;
  const XOROUT: u16 = 0xFFFF;
  const INIT_XOROUT: u16 = Self::INIT ^ Self::XOROUT;

  /// Pre-computed shift-by-8 matrix for combine.
  const SHIFT8_MATRIX: Gf2Matrix16 = generate_shift8_matrix_16(CRC16_CCITT_POLY);

  /// Create a hasher to resume from a previous CRC value.
  #[inline]
  #[must_use]
  pub const fn resume(crc: u16) -> Self {
    Self {
      state: crc ^ Self::XOROUT,
      kernel: crc16_ccitt_portable_auto,
      initialized: false,
    }
  }

  /// Get the name of the currently selected backend.
  #[must_use]
  pub fn backend_name() -> &'static str {
    CRC16_CCITT_DISPATCHER.backend_name()
  }

  /// Get the effective CRC-16 configuration.
  #[must_use]
  pub fn config() -> Crc16Config {
    config::get()
  }

  /// Convenience accessor for the active CRC-16 tunables.
  #[must_use]
  pub fn tunables() -> Crc16Tunables {
    Self::config().tunables
  }

  /// Returns the kernel name that the selector would choose for `len`.
  #[must_use]
  pub fn kernel_name_for_len(len: usize) -> &'static str {
    crc16_selected_kernel_name(len)
  }
}

impl traits::Checksum for Crc16Ccitt {
  const OUTPUT_SIZE: usize = 2;
  type Output = u16;

  #[inline]
  fn new() -> Self {
    Self {
      state: Self::INIT,
      kernel: CRC16_CCITT_DISPATCHER.kernel(),
      initialized: true,
    }
  }

  #[inline]
  fn with_initial(initial: u16) -> Self {
    Self {
      state: initial ^ Self::XOROUT,
      kernel: CRC16_CCITT_DISPATCHER.kernel(),
      initialized: true,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    if !self.initialized {
      self.kernel = CRC16_CCITT_DISPATCHER.kernel();
      self.initialized = true;
    }
    self.state = (self.kernel)(self.state, data);
  }

  #[inline]
  fn finalize(&self) -> u16 {
    self.state ^ Self::XOROUT
  }

  #[inline]
  fn reset(&mut self) {
    self.state = Self::INIT;
  }
}

impl Default for Crc16Ccitt {
  fn default() -> Self {
    <Self as traits::Checksum>::new()
  }
}

impl traits::ChecksumCombine for Crc16Ccitt {
  fn combine(crc_a: u16, crc_b: u16, len_b: usize) -> u16 {
    combine_crc16(crc_a, crc_b, len_b, Self::SHIFT8_MATRIX, Self::INIT_XOROUT)
  }
}

/// CRC-16/ARC checksum (also known as CRC-16/IBM).
///
/// # Properties
///
/// - **Polynomial**: 0x8005 (normal), 0xA001 (reflected)
/// - **Initial value**: 0x0000
/// - **Final XOR**: 0x0000
/// - **Reflect input/output**: Yes
///
/// # Examples
///
/// ```rust
/// use checksum::{Checksum, Crc16Ibm};
///
/// let crc = Crc16Ibm::checksum(b"123456789");
/// assert_eq!(crc, 0xBB3D);
/// ```
#[derive(Clone)]
pub struct Crc16Ibm {
  state: u16,
  kernel: Crc16Fn,
  initialized: bool,
}

impl Crc16Ibm {
  const INIT: u16 = 0x0000;
  const XOROUT: u16 = 0x0000;
  const INIT_XOROUT: u16 = Self::INIT ^ Self::XOROUT;

  /// Pre-computed shift-by-8 matrix for combine.
  const SHIFT8_MATRIX: Gf2Matrix16 = generate_shift8_matrix_16(CRC16_IBM_POLY);

  /// Create a hasher to resume from a previous CRC value.
  #[inline]
  #[must_use]
  pub const fn resume(crc: u16) -> Self {
    Self {
      state: crc ^ Self::XOROUT,
      kernel: crc16_ibm_portable_auto,
      initialized: false,
    }
  }

  /// Get the name of the currently selected backend.
  #[must_use]
  pub fn backend_name() -> &'static str {
    CRC16_IBM_DISPATCHER.backend_name()
  }

  /// Get the effective CRC-16 configuration.
  #[must_use]
  pub fn config() -> Crc16Config {
    config::get()
  }

  /// Convenience accessor for the active CRC-16 tunables.
  #[must_use]
  pub fn tunables() -> Crc16Tunables {
    Self::config().tunables
  }

  /// Returns the kernel name that the selector would choose for `len`.
  #[must_use]
  pub fn kernel_name_for_len(len: usize) -> &'static str {
    crc16_selected_kernel_name(len)
  }
}

impl traits::Checksum for Crc16Ibm {
  const OUTPUT_SIZE: usize = 2;
  type Output = u16;

  #[inline]
  fn new() -> Self {
    Self {
      state: Self::INIT,
      kernel: CRC16_IBM_DISPATCHER.kernel(),
      initialized: true,
    }
  }

  #[inline]
  fn with_initial(initial: u16) -> Self {
    Self {
      state: initial ^ Self::XOROUT,
      kernel: CRC16_IBM_DISPATCHER.kernel(),
      initialized: true,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    if !self.initialized {
      self.kernel = CRC16_IBM_DISPATCHER.kernel();
      self.initialized = true;
    }
    self.state = (self.kernel)(self.state, data);
  }

  #[inline]
  fn finalize(&self) -> u16 {
    self.state ^ Self::XOROUT
  }

  #[inline]
  fn reset(&mut self) {
    self.state = Self::INIT;
  }
}

impl Default for Crc16Ibm {
  fn default() -> Self {
    <Self as traits::Checksum>::new()
  }
}

impl traits::ChecksumCombine for Crc16Ibm {
  fn combine(crc_a: u16, crc_b: u16, len_b: usize) -> u16 {
    combine_crc16(crc_a, crc_b, len_b, Self::SHIFT8_MATRIX, Self::INIT_XOROUT)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffered CRC-16 Wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// Buffer size for buffered CRC-16 wrappers.
#[cfg(feature = "alloc")]
const BUFFERED_CRC16_BUFFER_SIZE: usize = 256;

#[cfg(feature = "alloc")]
#[inline]
#[must_use]
fn crc16_buffered_threshold() -> usize {
  config::get().tunables.slice4_to_slice8.max(64)
}

#[cfg(feature = "alloc")]
define_buffered_crc! {
  /// A buffering wrapper around [`Crc16Ccitt`] for streaming small chunks.
  pub struct BufferedCrc16Ccitt<Crc16Ccitt> {
    buffer_size: BUFFERED_CRC16_BUFFER_SIZE,
    threshold_fn: crc16_buffered_threshold,
  }
}

#[cfg(feature = "alloc")]
define_buffered_crc! {
  /// A buffering wrapper around [`Crc16Ibm`] for streaming small chunks.
  pub struct BufferedCrc16Ibm<Crc16Ibm> {
    buffer_size: BUFFERED_CRC16_BUFFER_SIZE,
    threshold_fn: crc16_buffered_threshold,
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
  fn test_vectors_crc16_ccitt_x25() {
    assert_eq!(Crc16Ccitt::checksum(b"123456789"), 0x906E);
  }

  #[test]
  fn test_vectors_crc16_ibm_arc() {
    assert_eq!(Crc16Ibm::checksum(b"123456789"), 0xBB3D);
  }

  #[test]
  fn test_combine_all_splits_ccitt() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let full = Crc16Ccitt::checksum(data);
    for split in 0..=data.len() {
      let (a, b) = data.split_at(split);
      let combined = Crc16Ccitt::combine(Crc16Ccitt::checksum(a), Crc16Ccitt::checksum(b), b.len());
      assert_eq!(combined, full, "split={split}");
    }
  }

  #[test]
  fn test_streaming_resume_ibm() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let mid = data.len() / 2;
    let (a, b) = data.split_at(mid);
    let oneshot = Crc16Ibm::checksum(data);

    let crc_a = Crc16Ibm::checksum(a);
    let mut resumed = Crc16Ibm::resume(crc_a);
    resumed.update(b);
    assert_eq!(resumed.finalize(), oneshot);
  }

  #[cfg(feature = "alloc")]
  #[test]
  fn test_buffered_ccitt_matches_unbuffered() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let expected = Crc16Ccitt::checksum(data);

    let mut buffered = BufferedCrc16Ccitt::new();
    for chunk in data.chunks(3) {
      buffered.update(chunk);
    }
    assert_eq!(buffered.finalize(), expected);
  }

  #[cfg(feature = "alloc")]
  #[test]
  fn test_buffered_ibm_matches_unbuffered() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let expected = Crc16Ibm::checksum(data);

    let mut buffered = BufferedCrc16Ibm::new();
    for chunk in data.chunks(3) {
      buffered.update(chunk);
    }
    assert_eq!(buffered.finalize(), expected);
  }
}
