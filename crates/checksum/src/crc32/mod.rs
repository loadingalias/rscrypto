//! CRC-32 implementations.
//!
//! This module provides:
//! - [`Crc32`] - CRC-32 IEEE (Ethernet, ZIP, PNG)
//! - [`Crc32c`] - CRC-32C Castagnoli (iSCSI, ext4, Btrfs)
//!
//! # Hardware Acceleration
//!
//! On supported platforms, hardware acceleration is automatically used:
//! - x86_64: SSE4.2 crc32 instruction (CRC-32C), PCLMULQDQ, VPCLMULQDQ
//! - aarch64: CRC32 extension, PMULL

mod portable;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

use backend::dispatch::Selected;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use backend::{candidates, dispatch::select};
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use platform::Caps;
use traits::{Checksum, ChecksumCombine};

use crate::{
  common::{
    combine::{Gf2Matrix32, combine_crc32, generate_shift8_matrix_32},
    tables::{CRC32_IEEE_POLY, CRC32C_POLY, generate_crc32_tables_8},
  },
  dispatchers::{Crc32Dispatcher, Crc32Fn},
};

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Wrappers
// ─────────────────────────────────────────────────────────────────────────────
//
// These wrap the portable/arch-specific implementations to match the Crc32Fn
// signature. Each wrapper bakes in the appropriate polynomial tables.

/// Portable kernel tables (pre-computed at compile time).
mod kernel_tables {
  use super::*;
  pub static IEEE_TABLES: [[u32; 256]; 8] = generate_crc32_tables_8(CRC32_IEEE_POLY);
  pub static CASTAGNOLI_TABLES: [[u32; 256]; 8] = generate_crc32_tables_8(CRC32C_POLY);
}

/// CRC-32 IEEE portable kernel wrapper.
fn crc32_ieee_portable(crc: u32, data: &[u8]) -> u32 {
  portable::crc32_slice8(crc, data, &kernel_tables::IEEE_TABLES)
}

/// CRC-32C portable kernel wrapper.
fn crc32c_portable(crc: u32, data: &[u8]) -> u32 {
  portable::crc32_slice8(crc, data, &kernel_tables::CASTAGNOLI_TABLES)
}

// aarch64 kernel wrappers
#[cfg(target_arch = "aarch64")]
fn crc32_ieee_aarch64(crc: u32, data: &[u8]) -> u32 {
  aarch64::crc32_crc_safe(crc, data)
}

#[cfg(target_arch = "aarch64")]
fn crc32c_aarch64(crc: u32, data: &[u8]) -> u32 {
  aarch64::crc32c_crc_safe(crc, data)
}

// x86_64 kernel wrappers
#[cfg(target_arch = "x86_64")]
fn crc32c_x86_64_sse42(crc: u32, data: &[u8]) -> u32 {
  x86_64::crc32c_sse42_safe(crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Selection
// ─────────────────────────────────────────────────────────────────────────────

/// Select the best CRC-32 IEEE kernel for the current platform.
#[cfg(target_arch = "aarch64")]
fn select_crc32_ieee() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  select(
    caps,
    candidates![
      "aarch64/crc" => platform::caps::aarch64::CRC => crc32_ieee_aarch64,
      "portable/slice8" => Caps::NONE => crc32_ieee_portable,
    ],
  )
}

#[cfg(target_arch = "x86_64")]
fn select_crc32_ieee() -> Selected<Crc32Fn> {
  // No native x86_64 CRC32 IEEE instruction; PCLMULQDQ not yet implemented
  Selected::new("portable/slice8", crc32_ieee_portable)
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn select_crc32_ieee() -> Selected<Crc32Fn> {
  Selected::new("portable/slice8", crc32_ieee_portable)
}

/// Select the best CRC-32C kernel for the current platform.
#[cfg(target_arch = "aarch64")]
fn select_crc32c() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  select(
    caps,
    candidates![
      "aarch64/crc" => platform::caps::aarch64::CRC => crc32c_aarch64,
      "portable/slice8" => Caps::NONE => crc32c_portable,
    ],
  )
}

#[cfg(target_arch = "x86_64")]
fn select_crc32c() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  select(
    caps,
    candidates![
      "x86_64/sse42" => platform::caps::x86::SSE42 => crc32c_x86_64_sse42,
      "portable/slice8" => Caps::NONE => crc32c_portable,
    ],
  )
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn select_crc32c() -> Selected<Crc32Fn> {
  Selected::new("portable/slice8", crc32c_portable)
}

/// Static dispatcher for CRC-32 IEEE.
static CRC32_IEEE_DISPATCHER: Crc32Dispatcher = Crc32Dispatcher::new(select_crc32_ieee);

/// Static dispatcher for CRC-32C.
static CRC32C_DISPATCHER: Crc32Dispatcher = Crc32Dispatcher::new(select_crc32c);

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32 IEEE
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32 checksum (IEEE 802.3 / ISO-HDLC).
///
/// Used in Ethernet FCS, ZIP, gzip, PNG, and many other formats.
///
/// # Properties
///
/// - **Polynomial**: 0x04C11DB7 (normal), 0xEDB88320 (reflected)
/// - **Initial value**: 0xFFFFFFFF
/// - **Final XOR**: 0xFFFFFFFF
/// - **Reflect input/output**: Yes
///
/// # Hardware Acceleration
///
/// - **aarch64**: CRC32 extension (`crc32w`, `crc32b`)
/// - **x86_64**: PCLMULQDQ folding, VPCLMULQDQ (AVX-512)
///
/// # Example
///
/// ```ignore
/// use checksum::{Crc32, Checksum};
///
/// let crc = Crc32::checksum(b"123456789");
/// assert_eq!(crc, 0xCBF43926); // "123456789" test vector
/// ```
#[derive(Clone, Default)]
pub struct Crc32 {
  state: u32,
}

impl Crc32 {
  /// Pre-computed shift-by-8 matrix for combine.
  const SHIFT8_MATRIX: Gf2Matrix32 = generate_shift8_matrix_32(CRC32_IEEE_POLY);

  /// Create a hasher to resume from a previous CRC value.
  #[inline]
  #[must_use]
  pub const fn resume(crc: u32) -> Self {
    Self { state: crc ^ !0 }
  }

  /// Get the name of the currently selected backend.
  ///
  /// Returns the implementation name (e.g., "portable/slice8", "aarch64/crc").
  #[must_use]
  pub fn backend_name() -> &'static str {
    CRC32_IEEE_DISPATCHER.backend_name()
  }
}

impl Checksum for Crc32 {
  const OUTPUT_SIZE: usize = 4;
  type Output = u32;

  #[inline]
  fn new() -> Self {
    Self { state: !0 }
  }

  #[inline]
  fn with_initial(initial: u32) -> Self {
    Self { state: initial ^ !0 }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.state = CRC32_IEEE_DISPATCHER.call(self.state, data);
  }

  #[inline]
  fn finalize(&self) -> u32 {
    self.state ^ !0
  }

  #[inline]
  fn reset(&mut self) {
    self.state = !0;
  }
}

impl ChecksumCombine for Crc32 {
  fn combine(crc_a: u32, crc_b: u32, len_b: usize) -> u32 {
    combine_crc32(crc_a, crc_b, len_b, Self::SHIFT8_MATRIX)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32C (Castagnoli)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32C checksum (Castagnoli polynomial).
///
/// Used in iSCSI, ext4, Btrfs, SCTP, and other modern protocols.
/// Has better error detection properties than CRC-32 IEEE.
///
/// # Properties
///
/// - **Polynomial**: 0x1EDC6F41 (normal), 0x82F63B78 (reflected)
/// - **Initial value**: 0xFFFFFFFF
/// - **Final XOR**: 0xFFFFFFFF
/// - **Reflect input/output**: Yes
///
/// # Hardware Acceleration
///
/// - **x86_64**: SSE4.2 `crc32` instruction (~20 GB/s)
/// - **aarch64**: CRC32 extension with `crc32cw`/`crc32cb` (~20 GB/s)
/// - **x86_64**: PCLMULQDQ (~15 GB/s), VPCLMULQDQ (~40 GB/s)
///
/// # Example
///
/// ```ignore
/// use checksum::{Crc32c, Checksum};
///
/// let crc = Crc32c::checksum(b"123456789");
/// assert_eq!(crc, 0xE3069283); // "123456789" test vector
/// ```
#[derive(Clone, Default)]
pub struct Crc32c {
  state: u32,
}

impl Crc32c {
  /// Pre-computed shift-by-8 matrix for combine.
  const SHIFT8_MATRIX: Gf2Matrix32 = generate_shift8_matrix_32(CRC32C_POLY);

  /// Create a hasher to resume from a previous CRC value.
  #[inline]
  #[must_use]
  pub const fn resume(crc: u32) -> Self {
    Self { state: crc ^ !0 }
  }

  /// Get the name of the currently selected backend.
  ///
  /// Returns the implementation name (e.g., "portable/slice8", "x86_64/sse42").
  #[must_use]
  pub fn backend_name() -> &'static str {
    CRC32C_DISPATCHER.backend_name()
  }
}

impl Checksum for Crc32c {
  const OUTPUT_SIZE: usize = 4;
  type Output = u32;

  #[inline]
  fn new() -> Self {
    Self { state: !0 }
  }

  #[inline]
  fn with_initial(initial: u32) -> Self {
    Self { state: initial ^ !0 }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.state = CRC32C_DISPATCHER.call(self.state, data);
  }

  #[inline]
  fn finalize(&self) -> u32 {
    self.state ^ !0
  }

  #[inline]
  fn reset(&mut self) {
    self.state = !0;
  }
}

impl ChecksumCombine for Crc32c {
  fn combine(crc_a: u32, crc_b: u32, len_b: usize) -> u32 {
    combine_crc32(crc_a, crc_b, len_b, Self::SHIFT8_MATRIX)
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
  fn test_crc32_ieee_checksum() {
    let crc = Crc32::checksum(TEST_DATA);
    assert_eq!(crc, 0xCBF43926);
  }

  #[test]
  fn test_crc32c_checksum() {
    let crc = Crc32c::checksum(TEST_DATA);
    assert_eq!(crc, 0xE3069283);
  }

  #[test]
  fn test_crc32_streaming() {
    let oneshot = Crc32::checksum(TEST_DATA);

    let mut hasher = Crc32::new();
    hasher.update(&TEST_DATA[..5]);
    hasher.update(&TEST_DATA[5..]);
    assert_eq!(hasher.finalize(), oneshot);
  }

  #[test]
  fn test_crc32c_streaming() {
    let oneshot = Crc32c::checksum(TEST_DATA);

    let mut hasher = Crc32c::new();
    for chunk in TEST_DATA.chunks(3) {
      hasher.update(chunk);
    }
    assert_eq!(hasher.finalize(), oneshot);
  }

  #[test]
  fn test_crc32_combine() {
    let data = b"hello world";
    let (a, b) = data.split_at(6);

    let crc_a = Crc32::checksum(a);
    let crc_b = Crc32::checksum(b);
    let combined = Crc32::combine(crc_a, crc_b, b.len());

    assert_eq!(combined, Crc32::checksum(data));
  }

  #[test]
  fn test_crc32c_combine() {
    let data = b"hello world";
    let (a, b) = data.split_at(6);

    let crc_a = Crc32c::checksum(a);
    let crc_b = Crc32c::checksum(b);
    let combined = Crc32c::combine(crc_a, crc_b, b.len());

    assert_eq!(combined, Crc32c::checksum(data));
  }

  #[test]
  fn test_crc32_empty() {
    let crc = Crc32::checksum(&[]);
    assert_eq!(crc, 0);

    let crc = Crc32c::checksum(&[]);
    assert_eq!(crc, 0);
  }

  #[test]
  fn test_crc32_reset() {
    let mut hasher = Crc32c::new();
    hasher.update(b"some data");
    hasher.reset();
    hasher.update(TEST_DATA);
    assert_eq!(hasher.finalize(), Crc32c::checksum(TEST_DATA));
  }

  #[test]
  fn test_crc32_combine_all_splits() {
    for split in 0..=TEST_DATA.len() {
      let (a, b) = TEST_DATA.split_at(split);
      let crc_a = Crc32c::checksum(a);
      let crc_b = Crc32c::checksum(b);
      let combined = Crc32c::combine(crc_a, crc_b, b.len());
      assert_eq!(combined, Crc32c::checksum(TEST_DATA), "Failed at split {split}");
    }
  }

  #[test]
  fn test_crc32_resume() {
    let mut h1 = Crc32c::new();
    h1.update(&TEST_DATA[..5]);
    let partial = h1.finalize();

    let mut h2 = Crc32c::resume(partial);
    h2.update(&TEST_DATA[5..]);
    assert_eq!(h2.finalize(), Crc32c::checksum(TEST_DATA));
  }

  #[test]
  fn test_backend_name_not_empty() {
    assert!(!Crc32::backend_name().is_empty());
    assert!(!Crc32c::backend_name().is_empty());
  }
}
