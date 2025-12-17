//! WebAssembly CRC implementation.
//!
//! # Limitations
//!
//! WebAssembly SIMD does NOT have:
//! - Carryless multiplication (PCLMULQDQ/PMULL) - the key to fast CRC on x86/ARM
//! - Hardware CRC32 instructions
//!
//! Therefore, we use the portable implementation which provides:
//! - **Default**: Slicing-by-8 with 8 KB lookup tables (~500 MB/s)
//! - **`no-tables` feature**: Bitwise table-less algorithm (~200 MB/s)
//!
//! For comparison, native platforms with SIMD achieve 25-100 GB/s.

#![cfg(target_arch = "wasm32")]

// CRC32-C (Castagnoli)

/// Compute CRC32-C for wasm32 targets.
///
/// Uses the portable implementation (slicing-by-8 or bitwise depending on features).
#[inline]
pub(crate) fn compute_crc32c(crc: u32, data: &[u8]) -> u32 {
  crate::crc32c::portable::compute(crc, data)
}

// CRC32 (ISO-HDLC)

/// Compute CRC32 for wasm32 targets.
#[inline]
pub(crate) fn compute_crc32(crc: u32, data: &[u8]) -> u32 {
  crate::crc32::portable::compute(crc, data)
}

#[cfg(test)]
mod tests {
  extern crate std;

  use std::vec;

  use super::*;

  #[test]
  fn test_crc32c_check_value() {
    let data = b"123456789";
    let crc = compute_crc32c(0xFFFF_FFFF, data) ^ 0xFFFF_FFFF;
    assert_eq!(crc, 0xE306_9283);
  }

  #[test]
  fn test_crc32c_various_sizes() {
    for size in [16, 64, 256, 1024, 4096] {
      let data = vec![0xCDu8; size];
      let crc = compute_crc32c(0xFFFF_FFFF, &data) ^ 0xFFFF_FFFF;
      // Verify against bitwise (known correct)
      let expected = crate::bitwise::crc32c::compute(0xFFFF_FFFF, &data) ^ 0xFFFF_FFFF;
      assert_eq!(crc, expected, "CRC32-C mismatch at size {}", size);
    }
  }

  #[test]
  fn test_crc32_check_value() {
    let data = b"123456789";
    let crc = compute_crc32(0xFFFF_FFFF, data) ^ 0xFFFF_FFFF;
    assert_eq!(crc, 0xCBF4_3926);
  }

  #[test]
  fn test_crc32_various_sizes() {
    for size in [16, 64, 256, 1024, 4096] {
      let data = vec![0xCDu8; size];
      let crc = compute_crc32(0xFFFF_FFFF, &data) ^ 0xFFFF_FFFF;
      let expected = crate::bitwise::crc32::compute(0xFFFF_FFFF, &data) ^ 0xFFFF_FFFF;
      assert_eq!(crc, expected, "CRC32 mismatch at size {}", size);
    }
  }
}
