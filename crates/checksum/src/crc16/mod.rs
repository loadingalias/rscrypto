//! CRC-16 implementations with optional hardware acceleration.
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
pub(crate) mod kernels;
pub(crate) mod policy;
pub(crate) mod portable;
mod tuned_defaults;

#[cfg(feature = "alloc")]
pub mod kernel_test;

use backend::dispatch::Selected;
#[allow(unused_imports)]
pub use config::{Crc16Config, Crc16Force, Crc16Tunables};
// Re-export traits for test modules (`use super::*`).
#[allow(unused_imports)]
pub(super) use traits::{Checksum, ChecksumCombine};

use crate::{
  common::{
    combine::{Gf2Matrix16, combine_crc16, generate_shift8_matrix_16},
    reference::crc16_bitwise,
    tables::{CRC16_CCITT_POLY, CRC16_IBM_POLY, generate_crc16_tables_4, generate_crc16_tables_8},
  },
  dispatchers::{Crc16Dispatcher, Crc16Fn},
};

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

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
// Reference Kernel Wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// Bitwise reference implementation for CRC-16/CCITT.
#[inline]
fn crc16_ccitt_reference(crc: u16, data: &[u8]) -> u16 {
  crc16_bitwise(CRC16_CCITT_POLY, crc, data)
}

/// Bitwise reference implementation for CRC-16/IBM.
#[inline]
fn crc16_ibm_reference(crc: u16, data: &[u8]) -> u16 {
  crc16_bitwise(CRC16_IBM_POLY, crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Portable Kernel Wrappers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn crc16_ccitt_portable_auto(crc: u16, data: &[u8]) -> u16 {
  let threshold = config::get().tunables.slice4_to_slice8;
  if data.len() < threshold {
    portable::crc16_ccitt_slice4(crc, data)
  } else {
    portable::crc16_ccitt_slice8(crc, data)
  }
}

#[inline]
fn crc16_ibm_portable_auto(crc: u16, data: &[u8]) -> u16 {
  let threshold = config::get().tunables.slice4_to_slice8;
  if data.len() < threshold {
    portable::crc16_ibm_slice4(crc, data)
  } else {
    portable::crc16_ibm_slice8(crc, data)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cached Policy and Kernels (works on both std and no_std)
// ─────────────────────────────────────────────────────────────────────────────

use backend::PolicyCache;

#[cfg(target_arch = "x86_64")]
static CRC16_CCITT_CACHED: PolicyCache<policy::Crc16Policy, policy::Crc16Kernels> = PolicyCache::new();
#[cfg(target_arch = "x86_64")]
static CRC16_IBM_CACHED: PolicyCache<policy::Crc16Policy, policy::Crc16Kernels> = PolicyCache::new();

#[cfg(target_arch = "aarch64")]
static CRC16_CCITT_CACHED: PolicyCache<policy::Crc16Policy, policy::Crc16Kernels> = PolicyCache::new();
#[cfg(target_arch = "aarch64")]
static CRC16_IBM_CACHED: PolicyCache<policy::Crc16Policy, policy::Crc16Kernels> = PolicyCache::new();

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
static CRC16_CCITT_CACHED: PolicyCache<policy::Crc16Policy, policy::Crc16Kernels> = PolicyCache::new();
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
static CRC16_IBM_CACHED: PolicyCache<policy::Crc16Policy, policy::Crc16Kernels> = PolicyCache::new();

#[cfg(target_arch = "x86_64")]
fn init_ccitt_policy() -> (policy::Crc16Policy, policy::Crc16Kernels) {
  let cfg = config::get();
  let caps = platform::caps();
  let tune = platform::tune();
  let pol = policy::Crc16Policy::from_config(&cfg, caps, &tune, policy::Crc16Variant::Ccitt);
  let kernels = policy::build_ccitt_kernels_x86(
    &pol,
    crc16_ccitt_reference,
    portable::crc16_ccitt_slice4,
    portable::crc16_ccitt_slice8,
  );
  (pol, kernels)
}

#[cfg(target_arch = "x86_64")]
fn init_ibm_policy() -> (policy::Crc16Policy, policy::Crc16Kernels) {
  let cfg = config::get();
  let caps = platform::caps();
  let tune = platform::tune();
  let pol = policy::Crc16Policy::from_config(&cfg, caps, &tune, policy::Crc16Variant::Ibm);
  let kernels = policy::build_ibm_kernels_x86(
    &pol,
    crc16_ibm_reference,
    portable::crc16_ibm_slice4,
    portable::crc16_ibm_slice8,
  );
  (pol, kernels)
}

#[cfg(target_arch = "aarch64")]
fn init_ccitt_policy() -> (policy::Crc16Policy, policy::Crc16Kernels) {
  let cfg = config::get();
  let caps = platform::caps();
  let tune = platform::tune();
  let pol = policy::Crc16Policy::from_config(&cfg, caps, &tune, policy::Crc16Variant::Ccitt);
  let kernels = policy::build_ccitt_kernels_aarch64(
    &pol,
    crc16_ccitt_reference,
    portable::crc16_ccitt_slice4,
    portable::crc16_ccitt_slice8,
  );
  (pol, kernels)
}

#[cfg(target_arch = "aarch64")]
fn init_ibm_policy() -> (policy::Crc16Policy, policy::Crc16Kernels) {
  let cfg = config::get();
  let caps = platform::caps();
  let tune = platform::tune();
  let pol = policy::Crc16Policy::from_config(&cfg, caps, &tune, policy::Crc16Variant::Ibm);
  let kernels = policy::build_ibm_kernels_aarch64(
    &pol,
    crc16_ibm_reference,
    portable::crc16_ibm_slice4,
    portable::crc16_ibm_slice8,
  );
  (pol, kernels)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn init_ccitt_policy() -> (policy::Crc16Policy, policy::Crc16Kernels) {
  let cfg = config::get();
  let caps = platform::caps();
  let tune = platform::tune();
  let pol = policy::Crc16Policy::from_config(&cfg, caps, &tune, policy::Crc16Variant::Ccitt);
  let kernels = policy::build_ccitt_kernels_generic(
    crc16_ccitt_reference,
    portable::crc16_ccitt_slice4,
    portable::crc16_ccitt_slice8,
  );
  (pol, kernels)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn init_ibm_policy() -> (policy::Crc16Policy, policy::Crc16Kernels) {
  let cfg = config::get();
  let caps = platform::caps();
  let tune = platform::tune();
  let pol = policy::Crc16Policy::from_config(&cfg, caps, &tune, policy::Crc16Variant::Ibm);
  let kernels = policy::build_ibm_kernels_generic(
    crc16_ibm_reference,
    portable::crc16_ibm_slice4,
    portable::crc16_ibm_slice8,
  );
  (pol, kernels)
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto Dispatch Functions (policy-based)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn crc16_ccitt_auto(crc: u16, data: &[u8]) -> u16 {
  let (pol, kernels) = CRC16_CCITT_CACHED.get_or_init(init_ccitt_policy);
  policy::policy_dispatch(&pol, &kernels, crc, data)
}

#[inline]
fn crc16_ibm_auto(crc: u16, data: &[u8]) -> u16 {
  let (pol, kernels) = CRC16_IBM_CACHED.get_or_init(init_ibm_policy);
  policy::policy_dispatch(&pol, &kernels, crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Introspection
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
#[must_use]
pub(crate) fn crc16_selected_kernel_name(len: usize) -> &'static str {
  let (pol, _) = CRC16_CCITT_CACHED.get_or_init(init_ccitt_policy);
  pol.kernel_name(len)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Selection
// ─────────────────────────────────────────────────────────────────────────────

fn select_crc16_ccitt() -> Selected<Crc16Fn> {
  // ALL modes go through policy dispatch.
  // The policy respects effective_force internally.
  Selected::new("auto", crc16_ccitt_auto)
}

fn select_crc16_ibm() -> Selected<Crc16Fn> {
  // ALL modes go through policy dispatch.
  // The policy respects effective_force internally.
  Selected::new("auto", crc16_ibm_auto)
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
  let t = config::get().tunables;
  t.slice4_to_slice8.max(t.portable_to_clmul).max(64)
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

// ─────────────────────────────────────────────────────────────────────────────
// Cross-Check Tests: All accelerated kernels vs bitwise reference
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod cross_check {
  extern crate alloc;
  extern crate std;

  use alloc::vec::Vec;

  use super::*;

  // ─────────────────────────────────────────────────────────────────────────
  // Test Data Generation
  // ─────────────────────────────────────────────────────────────────────────

  /// Lengths covering SIMD boundaries, alignment edges, and common sizes.
  const TEST_LENGTHS: &[usize] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, // Tiny
    16, 17, 31, 32, 33, 63, 64, 65, // SSE/NEON boundaries
    127, 128, 129, 255, 256, 257, // Cache line boundaries
    511, 512, 513, 1023, 1024, 1025, // Larger buffers
    2047, 2048, 2049, 4095, 4096, 4097, // Page boundaries
    8192, 16384, 32768, 65536, // Large buffers
  ];

  /// Chunk sizes for streaming tests.
  const STREAMING_CHUNK_SIZES: &[usize] = &[1, 3, 7, 13, 17, 31, 37, 61, 127, 251];

  /// Generate deterministic test data of a given length.
  fn generate_test_data(len: usize) -> Vec<u8> {
    (0..len)
      .map(|i| (i as u64).wrapping_mul(17).wrapping_add(i as u64) as u8)
      .collect()
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16/CCITT Cross-Check Tests
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn ccitt_all_lengths() {
    for &len in TEST_LENGTHS {
      let data = generate_test_data(len);
      let reference = crc16_ccitt_reference(!0u16, &data) ^ !0u16;
      let actual = Crc16Ccitt::checksum(&data);
      assert_eq!(actual, reference, "CRC-16/CCITT mismatch at len={len}");
    }
  }

  #[test]
  fn ccitt_all_single_bytes() {
    for byte in 0u8..=255 {
      let data = [byte];
      let reference = crc16_ccitt_reference(!0u16, &data) ^ !0u16;
      let actual = Crc16Ccitt::checksum(&data);
      assert_eq!(actual, reference, "CRC-16/CCITT mismatch for byte={byte:#04X}");
    }
  }

  #[test]
  fn ccitt_streaming_all_chunk_sizes() {
    let data = generate_test_data(4096);
    let reference = crc16_ccitt_reference(!0u16, &data) ^ !0u16;

    for &chunk_size in STREAMING_CHUNK_SIZES {
      let mut hasher = Crc16Ccitt::new();
      for chunk in data.chunks(chunk_size) {
        hasher.update(chunk);
      }
      let actual = hasher.finalize();
      assert_eq!(
        actual, reference,
        "CRC-16/CCITT streaming mismatch with chunk_size={chunk_size}"
      );
    }
  }

  #[test]
  fn ccitt_combine_all_splits() {
    let data = generate_test_data(1024);
    let reference = crc16_ccitt_reference(!0u16, &data) ^ !0u16;

    for split in [0, 1, 15, 16, 17, 127, 128, 129, 511, 512, 513, 1023, 1024] {
      if split > data.len() {
        continue;
      }
      let (a, b) = data.split_at(split);
      let crc_a = Crc16Ccitt::checksum(a);
      let crc_b = Crc16Ccitt::checksum(b);
      let combined = Crc16Ccitt::combine(crc_a, crc_b, b.len());
      assert_eq!(combined, reference, "CRC-16/CCITT combine mismatch at split={split}");
    }
  }

  #[test]
  fn ccitt_unaligned_offsets() {
    let data = generate_test_data(4096 + 64);

    for offset in 1..=16 {
      let slice = &data[offset..offset + 4096];
      let reference = crc16_ccitt_reference(!0u16, slice) ^ !0u16;
      let actual = Crc16Ccitt::checksum(slice);
      assert_eq!(actual, reference, "CRC-16/CCITT unaligned mismatch at offset={offset}");
    }
  }

  #[test]
  fn ccitt_byte_at_a_time_streaming() {
    let data = generate_test_data(256);
    let reference = crc16_ccitt_reference(!0u16, &data) ^ !0u16;

    let mut hasher = Crc16Ccitt::new();
    for &byte in &data {
      hasher.update(&[byte]);
    }
    assert_eq!(hasher.finalize(), reference, "CRC-16/CCITT byte-at-a-time mismatch");
  }

  #[test]
  fn ccitt_reference_kernel_accessible() {
    let data = b"123456789";
    let expected = 0x906E_u16;
    let reference = crc16_ccitt_reference(!0u16, data) ^ !0u16;
    assert_eq!(reference, expected, "Reference kernel check value mismatch");
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16/IBM Cross-Check Tests
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn ibm_all_lengths() {
    for &len in TEST_LENGTHS {
      let data = generate_test_data(len);
      let reference = crc16_ibm_reference(0u16, &data);
      let actual = Crc16Ibm::checksum(&data);
      assert_eq!(actual, reference, "CRC-16/IBM mismatch at len={len}");
    }
  }

  #[test]
  fn ibm_all_single_bytes() {
    for byte in 0u8..=255 {
      let data = [byte];
      let reference = crc16_ibm_reference(0u16, &data);
      let actual = Crc16Ibm::checksum(&data);
      assert_eq!(actual, reference, "CRC-16/IBM mismatch for byte={byte:#04X}");
    }
  }

  #[test]
  fn ibm_streaming_all_chunk_sizes() {
    let data = generate_test_data(4096);
    let reference = crc16_ibm_reference(0u16, &data);

    for &chunk_size in STREAMING_CHUNK_SIZES {
      let mut hasher = Crc16Ibm::new();
      for chunk in data.chunks(chunk_size) {
        hasher.update(chunk);
      }
      let actual = hasher.finalize();
      assert_eq!(
        actual, reference,
        "CRC-16/IBM streaming mismatch with chunk_size={chunk_size}"
      );
    }
  }

  #[test]
  fn ibm_combine_all_splits() {
    let data = generate_test_data(1024);
    let reference = crc16_ibm_reference(0u16, &data);

    for split in [0, 1, 15, 16, 17, 127, 128, 129, 511, 512, 513, 1023, 1024] {
      if split > data.len() {
        continue;
      }
      let (a, b) = data.split_at(split);
      let crc_a = Crc16Ibm::checksum(a);
      let crc_b = Crc16Ibm::checksum(b);
      let combined = Crc16Ibm::combine(crc_a, crc_b, b.len());
      assert_eq!(combined, reference, "CRC-16/IBM combine mismatch at split={split}");
    }
  }

  #[test]
  fn ibm_unaligned_offsets() {
    let data = generate_test_data(4096 + 64);

    for offset in 1..=16 {
      let slice = &data[offset..offset + 4096];
      let reference = crc16_ibm_reference(0u16, slice);
      let actual = Crc16Ibm::checksum(slice);
      assert_eq!(actual, reference, "CRC-16/IBM unaligned mismatch at offset={offset}");
    }
  }

  #[test]
  fn ibm_byte_at_a_time_streaming() {
    let data = generate_test_data(256);
    let reference = crc16_ibm_reference(0u16, &data);

    let mut hasher = Crc16Ibm::new();
    for &byte in &data {
      hasher.update(&[byte]);
    }
    assert_eq!(hasher.finalize(), reference, "CRC-16/IBM byte-at-a-time mismatch");
  }

  #[test]
  fn ibm_reference_kernel_accessible() {
    let data = b"123456789";
    let expected = 0xBB3D_u16;
    let reference = crc16_ibm_reference(0u16, data);
    assert_eq!(reference, expected, "Reference kernel check value mismatch");
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Portable Kernel Explicit Tests
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn ccitt_portable_matches_reference() {
    for &len in TEST_LENGTHS {
      let data = generate_test_data(len);
      let reference = crc16_ccitt_reference(!0u16, &data) ^ !0u16;
      let portable = portable::crc16_ccitt_slice8(!0u16, &data) ^ !0u16;
      assert_eq!(portable, reference, "CRC-16/CCITT portable mismatch at len={len}");
    }
  }

  #[test]
  fn ibm_portable_matches_reference() {
    for &len in TEST_LENGTHS {
      let data = generate_test_data(len);
      let reference = crc16_ibm_reference(0u16, &data);
      let portable = portable::crc16_ibm_slice8(0u16, &data);
      assert_eq!(portable, reference, "CRC-16/IBM portable mismatch at len={len}");
    }
  }
}
