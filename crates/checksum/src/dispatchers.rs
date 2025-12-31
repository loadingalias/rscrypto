//! CRC dispatcher types for checksum algorithms.
//!
//! This module provides pre-defined dispatcher types for various CRC widths.
//! Each dispatcher caches kernel selection for efficient repeated calls.
//!
//! # Usage
//!
//! ```rust
//! use backend::dispatch::Selected;
//! use checksum::dispatchers::{Crc32Dispatcher, Crc32Fn};
//!
//! fn portable(crc: u32, data: &[u8]) -> u32 {
//!   crc.wrapping_add(data.len() as u32)
//! }
//!
//! fn select_crc32() -> Selected<Crc32Fn> {
//!   Selected::new("portable", portable)
//! }
//!
//! static DISPATCHER: Crc32Dispatcher = Crc32Dispatcher::new(select_crc32);
//!
//! let got = DISPATCHER.call(0, b"abc");
//! assert_eq!(got, portable(0, b"abc"));
//! assert_eq!(DISPATCHER.backend_name(), "portable");
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// CRC-16 Dispatchers
// ─────────────────────────────────────────────────────────────────────────────

/// Function signature for CRC-16 kernels.
///
/// Used by CRC-16-CCITT, CRC-16-IBM, CRC-16-USB, and other 16-bit CRC variants.
///
/// # Arguments
///
/// * `state` - Current CRC state (typically initialized to 0xFFFF or 0x0000)
/// * `data` - Input data to process
///
/// # Returns
///
/// Updated CRC state after processing the input data.
pub type Crc16Fn = fn(u16, &[u8]) -> u16;

backend::define_dispatcher!(
  /// Dispatcher for CRC-16 kernels.
  ///
  /// Caches the selected kernel on first access. Thread-safe.
  ///
  /// # Performance
  ///
  /// - First call: ~1μs (kernel selection + caching)
  /// - Subsequent calls: ~3ns (cached function pointer)
  Crc16Dispatcher, Crc16Fn, u16
);

// ─────────────────────────────────────────────────────────────────────────────
// CRC-24 Dispatchers
// ─────────────────────────────────────────────────────────────────────────────

/// Function signature for CRC-24 kernels.
///
/// Used by CRC-24 (OpenPGP/Radix-64). The result is a 24-bit value stored
/// in the low 24 bits of a u32.
///
/// # Arguments
///
/// * `state` - Current CRC state (only low 24 bits are used)
/// * `data` - Input data to process
///
/// # Returns
///
/// Updated CRC state with the result in the low 24 bits.
pub type Crc24Fn = fn(u32, &[u8]) -> u32;

backend::define_dispatcher!(
  /// Dispatcher for CRC-24 kernels.
  ///
  /// Caches the selected kernel on first access. Thread-safe.
  Crc24Dispatcher, Crc24Fn, u32
);

// ─────────────────────────────────────────────────────────────────────────────
// CRC-64 Dispatchers
// ─────────────────────────────────────────────────────────────────────────────

/// Function signature for CRC-64 kernels.
///
/// Used by:
/// - CRC-64-XZ (ECMA-182) - XZ Utils, 7-Zip
/// - CRC-64-NVME - NVMe specification
/// - CRC-64-GO-ISO - Go standard library
///
/// # Hardware Acceleration
///
/// - **x86_64**: PCLMULQDQ, VPCLMULQDQ
/// - **aarch64**: PMULL, PMULL+EOR3
///
/// # Arguments
///
/// * `state` - Current CRC state (typically initialized to 0xFFFFFFFFFFFFFFFF)
/// * `data` - Input data to process
///
/// # Returns
///
/// Updated CRC state after processing the input data.
pub type Crc64Fn = fn(u64, &[u8]) -> u64;

backend::define_dispatcher!(
  /// Dispatcher for CRC-64 kernels.
  ///
  /// Caches the selected kernel on first access. Thread-safe.
  ///
  /// # Performance Characteristics
  ///
  /// | Backend | Throughput |
  /// |---------|------------|
  /// | VPCLMULQDQ | ~35 GB/s |
  /// | PCLMULQDQ | ~12 GB/s |
  /// | PMULL+EOR3 | ~15 GB/s |
  /// | PMULL | ~10 GB/s |
  /// | Portable | ~2 GB/s |
  Crc64Dispatcher, Crc64Fn, u64
);

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32 Dispatchers
// ─────────────────────────────────────────────────────────────────────────────

/// Function signature for CRC-32 kernels.
///
/// Used by:
/// - CRC-32 (IEEE)
/// - CRC-32C (Castagnoli)
///
/// # Hardware Acceleration
///
/// - **x86_64**: SSE4.2 `crc32` (CRC-32C only)
/// - **aarch64**: ARMv8 CRC extension (CRC-32 + CRC-32C)
///
/// # Arguments
///
/// * `state` - Current CRC state (typically initialized to 0xFFFFFFFF)
/// * `data` - Input data to process
///
/// # Returns
///
/// Updated CRC state after processing the input data.
pub type Crc32Fn = fn(u32, &[u8]) -> u32;

backend::define_dispatcher!(
  /// Dispatcher for CRC-32 kernels.
  ///
  /// Caches the selected kernel on first access. Thread-safe.
  Crc32Dispatcher, Crc32Fn, u32
);

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use backend::dispatch::Selected;

  use super::*;

  // Test kernel functions
  fn portable_crc16(_state: u16, _data: &[u8]) -> u16 {
    0xBEEF
  }
  fn portable_crc24(_state: u32, _data: &[u8]) -> u32 {
    0x00ABCDEF
  }
  fn portable_crc64(_state: u64, _data: &[u8]) -> u64 {
    0xDEAD_BEEF_CAFE_BABE
  }
  fn portable_crc32(_state: u32, _data: &[u8]) -> u32 {
    0xC0FF_EE00
  }

  fn test_crc16_selector() -> Selected<Crc16Fn> {
    Selected::new("portable", portable_crc16)
  }

  fn test_crc24_selector() -> Selected<Crc24Fn> {
    Selected::new("portable", portable_crc24)
  }

  fn test_crc64_selector() -> Selected<Crc64Fn> {
    Selected::new("portable", portable_crc64)
  }

  fn test_crc32_selector() -> Selected<Crc32Fn> {
    Selected::new("portable", portable_crc32)
  }

  #[test]
  fn test_crc16_dispatcher() {
    static DISPATCH: Crc16Dispatcher = Crc16Dispatcher::new(test_crc16_selector);
    assert_eq!(DISPATCH.get().name, "portable");
    assert_eq!(DISPATCH.call(0, &[]), 0xBEEF);
  }

  #[test]
  fn test_crc24_dispatcher() {
    static DISPATCH: Crc24Dispatcher = Crc24Dispatcher::new(test_crc24_selector);
    assert_eq!(DISPATCH.get().name, "portable");
    assert_eq!(DISPATCH.call(0, &[]), 0x00ABCDEF);
  }

  #[test]
  fn test_crc64_dispatcher() {
    static DISPATCH: Crc64Dispatcher = Crc64Dispatcher::new(test_crc64_selector);
    assert_eq!(DISPATCH.get().name, "portable");
    assert_eq!(DISPATCH.call(0, &[]), 0xDEAD_BEEF_CAFE_BABE);
  }

  #[test]
  fn test_crc32_dispatcher() {
    static DISPATCH: Crc32Dispatcher = Crc32Dispatcher::new(test_crc32_selector);
    assert_eq!(DISPATCH.get().name, "portable");
    assert_eq!(DISPATCH.call(0, &[]), 0xC0FF_EE00);
  }
}
