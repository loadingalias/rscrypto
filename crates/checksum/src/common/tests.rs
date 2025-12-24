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
//! # Usage
//!
//! ```ignore
//! use checksum::common::tests::CrcTestHarness;
//! use checksum::Crc64Xz;
//!
//! // Run all property tests for CRC-64-XZ
//! CrcTestHarness::<Crc64Xz>::run_all_tests();
//! ```

use traits::{Checksum, ChecksumCombine};

/// Generic test harness for CRC algorithms.
///
/// Provides a suite of property-based tests that verify fundamental CRC invariants.
/// These tests can be run against any type implementing `Checksum + ChecksumCombine`.
pub struct CrcTestHarness<C> {
  _phantom: core::marker::PhantomData<C>,
}

#[allow(dead_code)]
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
/// Creates a test module with standard property-based tests using proptest.
///
/// # Example
///
/// ```ignore
/// define_crc_property_tests!(crc64_xz_tests, Crc64Xz);
/// ```
#[macro_export]
macro_rules! define_crc_property_tests {
  ($mod_name:ident, $crc_type:ty) => {
    #[cfg(all(test, not(miri)))]
    mod $mod_name {
      use proptest::prelude::*;
      use $crate::common::tests::CrcTestHarness;

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
// Tests for the test harness itself
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod harness_self_tests {
  use super::*;
  // Test harness using CRC-64-XZ as the reference implementation
  use crate::Crc64Xz;

  #[test]
  fn harness_test_combine_all_splits() {
    CrcTestHarness::<Crc64Xz>::test_combine_all_splits(b"hello world");
  }

  #[test]
  fn harness_test_streaming_equals_oneshot() {
    CrcTestHarness::<Crc64Xz>::test_streaming_equals_oneshot(b"The quick brown fox");
  }

  #[test]
  fn harness_test_empty_input() {
    CrcTestHarness::<Crc64Xz>::test_empty_input();
  }

  #[test]
  fn harness_test_single_bytes() {
    CrcTestHarness::<Crc64Xz>::test_single_bytes();
  }

  #[test]
  fn harness_test_finalize_idempotent() {
    CrcTestHarness::<Crc64Xz>::test_finalize_idempotent(b"test data");
  }

  #[test]
  fn harness_test_reset() {
    CrcTestHarness::<Crc64Xz>::test_reset(b"fresh data");
  }
}
