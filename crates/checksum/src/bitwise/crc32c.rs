//! Table-less CRC32-C (Castagnoli) implementation.
//!
//! This module provides a zero-table CRC32-C implementation using optimized
//! bitwise computation with branchless masking.
//!
//! # Algorithm
//!
//! This module uses optimized branchless bitwise computation:
//!
//! - **Branchless conditional XOR** using wrapping arithmetic (no branch mispredictions)
//! - **Unrolled loop** for better instruction pipelining
//! - **4-byte chunk processing** to reduce loop overhead
//!
//! # Performance
//!
//! ~200 MB/s on modern CPUs (compared to ~25-100 GB/s with SIMD).
//! For large messages, prefer [`crate::Crc32c`] which uses SIMD when available.

use crate::constants::crc32c::POLYNOMIAL;

// ============================================================================
// Main API
// ============================================================================

/// Compute CRC32-C over a byte slice without using lookup tables.
///
/// This uses branchless bitwise computation optimized for modern CPUs.
/// The implementation avoids branches, enabling better CPU pipelining.
///
/// # Example
///
/// ```
/// use checksum::bitwise::crc32c::compute;
///
/// let crc = compute(0xFFFF_FFFF, b"123456789") ^ 0xFFFF_FFFF;
/// assert_eq!(crc, 0xE306_9283);
/// ```
#[inline]
pub fn compute(mut crc: u32, data: &[u8]) -> u32 {
  // Process 4 bytes at a time for better instruction-level parallelism.
  // The inner loop is still serial per byte, but grouping reduces loop overhead.
  let mut chunks = data.chunks_exact(4);

  for chunk in chunks.by_ref() {
    // SAFETY: chunks_exact guarantees exactly 4 bytes
    crc = compute_byte(crc, chunk[0]);
    crc = compute_byte(crc, chunk[1]);
    crc = compute_byte(crc, chunk[2]);
    crc = compute_byte(crc, chunk[3]);
  }

  // Handle remaining bytes
  for &byte in chunks.remainder() {
    crc = compute_byte(crc, byte);
  }

  crc
}

/// Compute CRC32-C for a single byte using branchless bitwise reduction.
///
/// This is a `const fn` to allow compile-time CRC computation for known data.
///
/// # Algorithm
///
/// For each of the 8 bits in the byte:
/// 1. Create a conditional mask: `mask = 0` if LSB is 0, `mask = 0xFFFFFFFF` if LSB is 1
/// 2. XOR with `(POLYNOMIAL & mask)` - this conditionally applies the polynomial
/// 3. Shift right by 1
///
/// The use of `wrapping_sub` creates the mask without branching:
/// - `0u32.wrapping_sub(0)` = 0x00000000
/// - `0u32.wrapping_sub(1)` = 0xFFFFFFFF
///
/// This branchless approach eliminates branch mispredictions and enables
/// better pipelining on modern out-of-order CPUs.
#[inline]
pub const fn compute_byte(mut crc: u32, byte: u8) -> u32 {
  crc ^= byte as u32;

  // Unrolled loop for better pipelining on modern CPUs.
  // Each iteration processes one bit using branchless masking.
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
    // Standard CRC32-C check value: "123456789" -> 0xE3069283
    let crc = compute(0xFFFF_FFFF, b"123456789") ^ 0xFFFF_FFFF;
    assert_eq!(crc, 0xE306_9283);
  }

  #[test]
  fn test_empty() {
    let crc = compute(0xFFFF_FFFF, b"") ^ 0xFFFF_FFFF;
    assert_eq!(crc, 0x0000_0000);
  }

  #[test]
  fn test_single_byte() {
    let crc = compute(0xFFFF_FFFF, &[0x00]) ^ 0xFFFF_FFFF;
    assert_eq!(crc, 0x527D_5351);
  }

  #[test]
  fn test_incremental() {
    let data = b"hello world";
    let oneshot = compute(0xFFFF_FFFF, data) ^ 0xFFFF_FFFF;

    // Incremental should match oneshot
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
    // Verify that compute_byte works at const time
    const CRC_OF_ZERO: u32 = compute_byte(0xFFFF_FFFF, 0x00);
    // After processing 0x00 with initial CRC 0xFFFFFFFF, we get 0x527D5351 ^ 0xFFFFFFFF
    assert_eq!(CRC_OF_ZERO ^ 0xFFFF_FFFF, 0x527D_5351);
  }

  #[test]
  fn test_various_patterns() {
    // All zeros
    let crc = compute(0xFFFF_FFFF, &[0x00; 16]) ^ 0xFFFF_FFFF;
    assert_ne!(crc, 0); // Should produce non-zero CRC

    // All ones
    let crc1 = compute(0xFFFF_FFFF, &[0xFF; 16]) ^ 0xFFFF_FFFF;
    assert_ne!(crc1, 0);

    // Alternating pattern
    let pattern: [u8; 16] = [
      0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55,
    ];
    let crc2 = compute(0xFFFF_FFFF, &pattern) ^ 0xFFFF_FFFF;
    assert_ne!(crc2, 0);

    // All different
    assert_ne!(crc, crc1);
    assert_ne!(crc1, crc2);
  }
}
