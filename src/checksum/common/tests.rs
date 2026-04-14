//! Generic test harnesses for CRC algorithms.
//!
//! This module provides property-based test harnesses that work with any CRC type
//! implementing the [`Checksum`] and [`ChecksumCombine`] traits. These tests verify
//! fundamental invariants that must hold for all correct CRC implementations.
//!
//! # Invariants Tested
//!
//! 1. **Combine property**: `crc(A || B) == combine(crc(A), crc(B), len(B))`
//! 2. **SIMD/Portable equivalence**: SIMD kernels produce identical results to portable
//! 3. **Streaming consistency**: Incremental updates produce the same result as one-shot
//! 4. **Idempotent finalize**: `finalize()` can be called multiple times
//!
//! Used via [`define_crc_property_tests!`] macro in algorithm modules.

use crate::traits::{Checksum, ChecksumCombine};

/// Generic test harness for CRC algorithms.
///
/// Provides a suite of property-based tests that verify fundamental CRC invariants.
/// These tests can be run against any type implementing `Checksum + ChecksumCombine`.
pub struct CrcTestHarness<C> {
  _phantom: core::marker::PhantomData<C>,
}

impl<C> CrcTestHarness<C>
where
  C: Checksum + ChecksumCombine,
  C::Output: Eq + core::fmt::Debug,
{
  // ─────────────────────────────────────────────────────────────────────────
  // Combine Property Tests
  // ─────────────────────────────────────────────────────────────────────────

  /// Test that combine produces crc(A || B) from crc(A) and crc(B).
  ///
  /// This is the fundamental combine property that enables parallel CRC computation.
  #[inline]
  pub fn test_combine_property(data: &[u8], split: usize) {
    let split = if data.is_empty() { 0 } else { split % data.len() };
    let (a, b) = data.split_at(split);

    let crc_a = C::checksum(a);
    let crc_b = C::checksum(b);
    let combined = C::combine(crc_a, crc_b, b.len());

    let expected = C::checksum(data);
    assert_eq!(combined, expected, "combine(crc(A), crc(B), len(B)) != crc(A || B)");
  }

  /// Test combine property at all possible split points for a given buffer.
  #[inline]
  #[cfg_attr(test, allow(dead_code))]
  pub fn test_combine_all_splits(data: &[u8]) {
    let full = C::checksum(data);

    for split in 0..=data.len() {
      let (a, b) = data.split_at(split);
      let crc_a = C::checksum(a);
      let crc_b = C::checksum(b);
      let combined = C::combine(crc_a, crc_b, b.len());

      assert_eq!(combined, full, "combine failed at split point {split}/{}", data.len());
    }
  }

  /// Test combine with empty second part (identity case).
  #[inline]
  pub fn test_combine_empty_suffix(data: &[u8]) {
    let crc_data = C::checksum(data);
    let crc_empty = C::checksum(&[]);
    let combined = C::combine(crc_data, crc_empty, 0);

    assert_eq!(combined, crc_data, "combine(crc(A), crc(''), 0) != crc(A)");
  }

  /// Test combine with empty first part.
  #[inline]
  pub fn test_combine_empty_prefix(data: &[u8]) {
    let crc_empty = C::checksum(&[]);
    let crc_data = C::checksum(data);
    let combined = C::combine(crc_empty, crc_data, data.len());

    assert_eq!(combined, crc_data, "combine(crc(''), crc(B), len(B)) != crc(B)");
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Streaming Consistency Tests
  // ─────────────────────────────────────────────────────────────────────────

  /// Test that streaming updates produce the same result as one-shot.
  #[inline]
  #[cfg_attr(test, allow(dead_code))]
  pub fn test_streaming_equals_oneshot(data: &[u8]) {
    let oneshot = C::checksum(data);

    let mut hasher = C::new();
    hasher.update(data);
    let streaming = hasher.finalize();

    assert_eq!(streaming, oneshot, "streaming != oneshot");
  }

  /// Test streaming with byte-at-a-time updates.
  #[inline]
  pub fn test_streaming_byte_at_a_time(data: &[u8]) {
    let oneshot = C::checksum(data);

    let mut hasher = C::new();
    for &byte in data {
      hasher.update(&[byte]);
    }
    let streaming = hasher.finalize();

    assert_eq!(streaming, oneshot, "byte-at-a-time streaming != oneshot");
  }

  /// Test streaming across a specific chunk size boundary.
  #[inline]
  pub fn test_streaming_chunked(data: &[u8], chunk_size: usize) {
    if chunk_size == 0 {
      return;
    }

    let oneshot = C::checksum(data);

    let mut hasher = C::new();
    for chunk in data.chunks(chunk_size) {
      hasher.update(chunk);
    }
    let streaming = hasher.finalize();

    assert_eq!(streaming, oneshot, "streaming with chunk_size={chunk_size} != oneshot");
  }

  /// Test that finalize is idempotent (can be called multiple times).
  #[inline]
  pub fn test_finalize_idempotent(data: &[u8]) {
    let mut hasher = C::new();
    hasher.update(data);

    let first = hasher.finalize();
    let second = hasher.finalize();
    let third = hasher.finalize();

    assert_eq!(first, second, "finalize() not idempotent (1st != 2nd)");
    assert_eq!(second, third, "finalize() not idempotent (2nd != 3rd)");
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Reset Tests
  // ─────────────────────────────────────────────────────────────────────────

  /// Test that reset returns the hasher to its initial state.
  #[inline]
  pub fn test_reset(data: &[u8]) {
    let fresh = C::checksum(data);

    let mut hasher = C::new();
    hasher.update(b"garbage data that should be discarded");
    hasher.reset();
    hasher.update(data);
    let after_reset = hasher.finalize();

    assert_eq!(after_reset, fresh, "reset did not restore initial state");
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Combine + Streaming Integration Tests
  // ─────────────────────────────────────────────────────────────────────────

  /// Test that streaming and combine work together correctly.
  ///
  /// Computes a CRC by:
  /// 1. Streaming first half
  /// 2. Computing second half separately
  /// 3. Combining the results
  #[inline]
  pub fn test_streaming_and_combine(data: &[u8]) {
    if data.is_empty() {
      return;
    }

    let full = C::checksum(data);
    let mid = data.len() / 2;
    let (first_half, second_half) = data.split_at(mid);

    // Stream the first half
    let mut hasher = C::new();
    hasher.update(first_half);
    let crc_first = hasher.finalize();

    // One-shot the second half
    let crc_second = C::checksum(second_half);

    // Combine
    let combined = C::combine(crc_first, crc_second, second_half.len());

    assert_eq!(combined, full, "streaming + combine != full CRC");
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Edge Cases
  // ─────────────────────────────────────────────────────────────────────────

  /// Test empty input.
  #[inline]
  pub fn test_empty_input() {
    let oneshot = C::checksum(&[]);

    let hasher = C::new();
    let streaming = hasher.finalize();

    assert_eq!(streaming, oneshot, "empty streaming != empty oneshot");
  }

  /// Test single byte inputs for all byte values.
  #[inline]
  pub fn test_single_bytes() {
    for byte in 0u8..=255 {
      let oneshot = C::checksum(&[byte]);

      let mut hasher = C::new();
      hasher.update(&[byte]);
      let streaming = hasher.finalize();

      assert_eq!(streaming, oneshot, "single byte {byte} mismatch");
    }
  }
}

/// Helper macro to generate property tests for a CRC type.
///
/// Creates a test module with proptest-based tests.
#[macro_export]
macro_rules! define_crc_property_tests {
  ($mod_name:ident, $crc_type:ty) => {
    #[cfg(all(test, not(miri)))]
    mod $mod_name {
      use proptest::prelude::*;
      use $crate::checksum::common::tests::CrcTestHarness;

      use super::*;

      proptest! {
        /// crc(A || B) == combine(crc(A), crc(B), len(B))
        #[test]
        fn combine_property(data in proptest::collection::vec(any::<u8>(), 0..1024), split in 0usize..1024) {
          CrcTestHarness::<$crc_type>::test_combine_property(&data, split);
        }

        /// Streaming byte-at-a-time equals one-shot
        #[test]
        fn streaming_byte_at_a_time(data in proptest::collection::vec(any::<u8>(), 0..256)) {
          CrcTestHarness::<$crc_type>::test_streaming_byte_at_a_time(&data);
        }

        /// Streaming with various chunk sizes equals one-shot
        #[test]
        fn streaming_chunked(data in proptest::collection::vec(any::<u8>(), 0..512), chunk_size in 1usize..64) {
          CrcTestHarness::<$crc_type>::test_streaming_chunked(&data, chunk_size);
        }

        /// Finalize is idempotent
        #[test]
        fn finalize_idempotent(data in proptest::collection::vec(any::<u8>(), 0..256)) {
          CrcTestHarness::<$crc_type>::test_finalize_idempotent(&data);
        }

        /// Reset restores initial state
        #[test]
        fn reset_works(data in proptest::collection::vec(any::<u8>(), 0..256)) {
          CrcTestHarness::<$crc_type>::test_reset(&data);
        }

        /// Streaming + combine integration
        #[test]
        fn streaming_and_combine(data in proptest::collection::vec(any::<u8>(), 1..512)) {
          CrcTestHarness::<$crc_type>::test_streaming_and_combine(&data);
        }
      }

      #[test]
      fn test_empty_input() {
        CrcTestHarness::<$crc_type>::test_empty_input();
      }

      #[test]
      fn test_single_bytes() {
        CrcTestHarness::<$crc_type>::test_single_bytes();
      }

      #[test]
      fn test_combine_empty_suffix() {
        CrcTestHarness::<$crc_type>::test_combine_empty_suffix(b"test data");
      }

      #[test]
      fn test_combine_empty_prefix() {
        CrcTestHarness::<$crc_type>::test_combine_empty_prefix(b"test data");
      }
    }
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared Cross-Check Test Infrastructure
// ─────────────────────────────────────────────────────────────────────────────

/// Lengths covering SIMD boundaries, alignment edges, and common sizes.
///
/// Used by all CRC cross-check test modules to exercise edge cases around
/// lane widths, cache line boundaries, and page boundaries.
pub const TEST_LENGTHS: &[usize] = &[
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, // Tiny
  16, 17, 31, 32, 33, 63, 64, 65, // SSE/NEON boundaries
  127, 128, 129, 255, 256, 257, // Cache line boundaries
  511, 512, 513, 1023, 1024, 1025, // Larger buffers
  2047, 2048, 2049, 4095, 4096, 4097, // Page boundaries
  8192, 16384, 32768, 65536, // Large buffers
];

/// Prime-sized chunk patterns for streaming cross-check tests.
pub const STREAMING_CHUNK_SIZES: &[usize] = &[1, 3, 7, 13, 17, 31, 37, 61, 127, 251];

/// Generate deterministic test data of a given length.
///
/// Uses a simple mixing function to produce non-trivial byte patterns
/// that avoid accidentally passing due to regularity.
#[cfg_attr(test, allow(dead_code))]
pub fn generate_test_data(len: usize) -> alloc::vec::Vec<u8> {
  (0..len)
    .map(|i| (i as u64).wrapping_mul(17).wrapping_add(i as u64) as u8)
    .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests for the test harness itself
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod harness_self_tests {
  // Test harness using CRC-64-XZ as the reference implementation
  #[cfg(feature = "crc64")]
  use super::*;
  #[cfg(feature = "crc64")]
  use crate::checksum::Crc64Xz;

  #[test]
  #[cfg(feature = "crc64")]
  fn harness_test_combine_all_splits() {
    CrcTestHarness::<Crc64Xz>::test_combine_all_splits(b"hello world");
  }

  #[test]
  #[cfg(feature = "crc64")]
  fn harness_test_streaming_equals_oneshot() {
    CrcTestHarness::<Crc64Xz>::test_streaming_equals_oneshot(b"The quick brown fox");
  }

  #[test]
  #[cfg(feature = "crc64")]
  fn harness_test_empty_input() {
    CrcTestHarness::<Crc64Xz>::test_empty_input();
  }

  #[test]
  #[cfg(feature = "crc64")]
  fn harness_test_single_bytes() {
    CrcTestHarness::<Crc64Xz>::test_single_bytes();
  }

  #[test]
  #[cfg(feature = "crc64")]
  fn harness_test_finalize_idempotent() {
    CrcTestHarness::<Crc64Xz>::test_finalize_idempotent(b"test data");
  }

  #[test]
  #[cfg(feature = "crc64")]
  fn harness_test_reset() {
    CrcTestHarness::<Crc64Xz>::test_reset(b"fresh data");
  }
}
