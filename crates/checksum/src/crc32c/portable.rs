//! Portable CRC32-C implementation using slicing-by-8.
//!
//! This implementation achieves ~500 MB/s on modern CPUs by processing
//! 8 bytes at a time using 8 precomputed lookup tables.
//!
//! # Algorithm
//!
//! The slicing-by-8 algorithm (also called "slicing-by-N" or "Sarwate's algorithm
//! extended") processes N bytes simultaneously using N lookup tables.
//!
//! For each 8-byte chunk:
//! 1. XOR the chunk with the current CRC
//! 2. Look up each byte in its corresponding table
//! 3. XOR all 8 table entries to get the new CRC
//!
//! This hides the latency of table lookups by doing them in parallel.

#[cfg(not(feature = "no-tables"))]
use crate::constants::crc32c::TABLES;
#[cfg(not(feature = "no-tables"))]
macro_rules! table {
  ($idx:expr) => {
    TABLES.0[$idx]
  };
}

/// Compute CRC32-C using slicing-by-8.
///
/// # Arguments
///
/// * `crc` - Current CRC state (NOT pre/post XORed - raw register value)
/// * `data` - Data to process
///
/// # Returns
///
/// Updated CRC state (raw register value, caller must apply final XOR)
#[inline]
#[allow(dead_code)] // Unused when compile-time hardware CRC is enabled.
pub fn compute(crc: u32, data: &[u8]) -> u32 {
  #[cfg(feature = "no-tables")]
  {
    crate::bitwise::crc32c::compute(crc, data)
  }

  #[cfg(not(feature = "no-tables"))]
  {
    let mut crc = crc;
    let mut chunks = data.chunks_exact(8);

    // Process 8 bytes at a time
    for chunk in chunks.by_ref() {
      // Read 8 bytes as little-endian u64
      // SAFETY: chunks_exact guarantees exactly 8 bytes
      let bytes: [u8; 8] = chunk.try_into().unwrap();
      let d = u64::from_le_bytes(bytes);

      // XOR lower 32 bits with current CRC
      let lo = (crc as u64) ^ (d & 0xFFFF_FFFF);
      let hi = d >> 32;

      // Parallel table lookups for all 8 bytes
      // The tables are ordered so that table[7] corresponds to byte 0 (LSB)
      // and table[0] corresponds to byte 7 (MSB)
      let b0 = lo as u8 as usize;
      let b1 = (lo >> 8) as u8 as usize;
      let b2 = (lo >> 16) as u8 as usize;
      let b3 = (lo >> 24) as u8 as usize;
      let b4 = hi as u8 as usize;
      let b5 = (hi >> 8) as u8 as usize;
      let b6 = (hi >> 16) as u8 as usize;
      let b7 = (hi >> 24) as u8 as usize;

      crc = table!(7)[b0]
        ^ table!(6)[b1]
        ^ table!(5)[b2]
        ^ table!(4)[b3]
        ^ table!(3)[b4]
        ^ table!(2)[b5]
        ^ table!(1)[b6]
        ^ table!(0)[b7];
    }

    // Process remaining bytes one at a time
    for &byte in chunks.remainder() {
      let idx = (crc as u8 ^ byte) as usize;
      crc = (crc >> 8) ^ table!(0)[idx];
    }

    crc
  }
}

/// Compute CRC32-C for a single byte.
///
/// Useful for processing unaligned prefixes/suffixes.
#[inline]
#[allow(dead_code)] // Used in SIMD implementations for cleanup
pub const fn compute_byte(crc: u32, byte: u8) -> u32 {
  #[cfg(feature = "no-tables")]
  {
    crate::bitwise::crc32c::compute_byte(crc, byte)
  }

  #[cfg(not(feature = "no-tables"))]
  {
    let idx = (crc as u8 ^ byte) as usize;
    (crc >> 8) ^ table!(0)[idx]
  }
}

#[cfg(test)]
mod tests {
  extern crate std;

  use alloc::vec;

  use super::*;

  /// Standard CRC32-C test vector: "123456789" -> 0xE3069283
  const CHECK_VALUE: u32 = 0xE306_9283;

  #[test]
  fn test_check_string() {
    let data = b"123456789";
    // CRC32-C uses initial value 0xFFFFFFFF and final XOR 0xFFFFFFFF
    let crc = compute(0xFFFF_FFFF, data) ^ 0xFFFF_FFFF;
    assert_eq!(crc, CHECK_VALUE);
  }

  #[test]
  fn test_empty() {
    // Empty input with standard init/finalize
    let crc = compute(0xFFFF_FFFF, b"") ^ 0xFFFF_FFFF;
    assert_eq!(crc, 0x0000_0000);
  }

  #[test]
  fn test_zeros() {
    // 32 zero bytes
    let data = [0u8; 32];
    let crc = compute(0xFFFF_FFFF, &data) ^ 0xFFFF_FFFF;
    assert_eq!(crc, 0x8A91_36AA);
  }

  #[test]
  fn test_ones() {
    // 32 0xFF bytes
    let data = [0xFFu8; 32];
    let crc = compute(0xFFFF_FFFF, &data) ^ 0xFFFF_FFFF;
    assert_eq!(crc, 0x62A8_AB43);
  }

  #[test]
  fn test_incremental_matches_oneshot() {
    let data = b"hello world, this is a test of incremental CRC";

    // One-shot
    let oneshot = compute(0xFFFF_FFFF, data) ^ 0xFFFF_FFFF;

    // Incremental (split at various points)
    for split in 0..data.len() {
      let (a, b) = data.split_at(split);
      let mut crc = compute(0xFFFF_FFFF, a);
      crc = compute(crc, b);
      crc ^= 0xFFFF_FFFF;
      assert_eq!(crc, oneshot, "mismatch at split point {}", split);
    }
  }

  #[test]
  fn test_single_byte() {
    // Single byte 0x00
    let crc = compute(0xFFFF_FFFF, &[0x00]) ^ 0xFFFF_FFFF;
    assert_eq!(crc, 0x527D_5351);
  }

  #[test]
  fn test_various_lengths() {
    // Ensure no panics for various lengths (alignment edge cases)
    for len in 0..=128 {
      let data = vec![0xABu8; len];
      let _ = compute(0xFFFF_FFFF, &data);
    }
  }
}
