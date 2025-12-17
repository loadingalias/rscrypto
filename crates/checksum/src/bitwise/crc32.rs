//! Table-less CRC32 (ISO-HDLC) implementation using branchless bitwise computation.
//!
//! This module provides a zero-table CRC32 implementation optimized for
//! environments where lookup tables are too expensive.
//!
//! # Algorithm
//!
//! Uses branchless bitwise polynomial reduction:
//!
//! ```text
//! for each bit:
//!   mask = 0 - (crc & 1)    // 0x00000000 or 0xFFFFFFFF
//!   crc = (crc >> 1) ^ (POLYNOMIAL & mask)
//! ```
//!
//! This avoids branch mispredictions and enables better CPU pipelining.
//!
//! # Performance
//!
//! ~200 MB/s on modern CPUs (compared to ~25-100 GB/s with SIMD).
//! For large messages, prefer [`crate::Crc32`] which uses SIMD when available.

use crate::constants::crc32::POLYNOMIAL;

// Main API

/// Compute CRC32 over a byte slice without using lookup tables.
///
/// # Example
///
/// ```
/// use checksum::bitwise::crc32::compute;
///
/// let crc = compute(0xFFFF_FFFF, b"123456789") ^ 0xFFFF_FFFF;
/// assert_eq!(crc, 0xCBF4_3926);
/// ```
#[inline]
pub fn compute(mut crc: u32, data: &[u8]) -> u32 {
  // Process 4 bytes at a time for better instruction-level parallelism.
  let mut chunks = data.chunks_exact(4);

  for chunk in chunks.by_ref() {
    crc = compute_byte(crc, chunk[0]);
    crc = compute_byte(crc, chunk[1]);
    crc = compute_byte(crc, chunk[2]);
    crc = compute_byte(crc, chunk[3]);
  }

  for &byte in chunks.remainder() {
    crc = compute_byte(crc, byte);
  }

  crc
}

/// Compute CRC32 for a single byte using branchless bitwise reduction.
///
/// This is a `const fn` to allow compile-time CRC computation.
#[inline]
pub const fn compute_byte(mut crc: u32, byte: u8) -> u32 {
  crc ^= byte as u32;

  // Unrolled loop with branchless masking.
  let mask = 0u32.wrapping_sub(crc & 1);
  crc = (crc >> 1) ^ (POLYNOMIAL & mask);

  let mask = 0u32.wrapping_sub(crc & 1);
  crc = (crc >> 1) ^ (POLYNOMIAL & mask);

  let mask = 0u32.wrapping_sub(crc & 1);
  crc = (crc >> 1) ^ (POLYNOMIAL & mask);

  let mask = 0u32.wrapping_sub(crc & 1);
  crc = (crc >> 1) ^ (POLYNOMIAL & mask);

  let mask = 0u32.wrapping_sub(crc & 1);
  crc = (crc >> 1) ^ (POLYNOMIAL & mask);

  let mask = 0u32.wrapping_sub(crc & 1);
  crc = (crc >> 1) ^ (POLYNOMIAL & mask);

  let mask = 0u32.wrapping_sub(crc & 1);
  crc = (crc >> 1) ^ (POLYNOMIAL & mask);

  let mask = 0u32.wrapping_sub(crc & 1);
  crc = (crc >> 1) ^ (POLYNOMIAL & mask);

  crc
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_check_value() {
    // Standard CRC32 check value: "123456789" -> 0xCBF43926
    let crc = compute(0xFFFF_FFFF, b"123456789") ^ 0xFFFF_FFFF;
    assert_eq!(crc, 0xCBF4_3926);
  }

  #[test]
  fn test_empty() {
    let crc = compute(0xFFFF_FFFF, b"") ^ 0xFFFF_FFFF;
    assert_eq!(crc, 0x0000_0000);
  }

  #[test]
  fn test_single_byte() {
    let crc = compute(0xFFFF_FFFF, &[0x00]) ^ 0xFFFF_FFFF;
    assert_eq!(crc, 0xD202_EF8D);
  }

  #[test]
  fn test_incremental() {
    let data = b"hello world";
    let oneshot = compute(0xFFFF_FFFF, data) ^ 0xFFFF_FFFF;

    for split in 0..=data.len() {
      let (a, b) = data.split_at(split);
      let mut crc = compute(0xFFFF_FFFF, a);
      crc = compute(crc, b);
      crc ^= 0xFFFF_FFFF;
      assert_eq!(crc, oneshot, "mismatch at split {}", split);
    }
  }

  #[test]
  fn test_const_computation() {
    const CRC_OF_ZERO: u32 = compute_byte(0xFFFF_FFFF, 0x00);
    assert_eq!(CRC_OF_ZERO ^ 0xFFFF_FFFF, 0xD202_EF8D);
  }
}
