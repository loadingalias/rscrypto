//! Unified property tests for all CRC implementations.
//!
//! This module provides rigorous property-based tests that verify two fundamental
//! invariants across ALL CRC variants:
//!
//! 1. **Combine correctness**: `crc(A || B) == combine(crc(A), crc(B), len(B))`
//!    - Verified against the bitwise reference implementation (mathematical truth)
//!    - Tests random splits at all positions
//!
//! 2. **Chunking equivalence**: Any chunking of input through streaming API equals oneshot
//!    - Tests arbitrary, variable-size chunk patterns
//!    - Proves streaming implementation is correct regardless of update boundaries
//!
//! These tests use independent reference crates as the oracle, establishing
//! that production code matches known-good external implementations.

// Proptest uses getcwd() which fails under Miri isolation.
#![cfg(not(miri))]
#![cfg(feature = "checksums")]

use crc::Crc as RefCrc;
use crc_fast::CrcAlgorithm;
use proptest::prelude::*;
use rscrypto::{Checksum, ChecksumCombine, Crc16Ccitt, Crc16Ibm, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme};

const REF_CRC24_OPENPGP: RefCrc<u32> = RefCrc::<u32>::new(&crc::CRC_24_OPENPGP);

// ─────────────────────────────────────────────────────────────────────────────
// Combine Correctness Tests
// ─────────────────────────────────────────────────────────────────────────────
//
// These tests prove the fundamental combine property:
//   crc(A || B) == combine(crc(A), crc(B), len(B))
//
// We verify against independent reference crates. This proves the combine
// operation matches external implementations, not just internal consistency.
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
  #![proptest_config(ProptestConfig::with_cases(256))]

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16 Combine Correctness
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc16_ccitt_combine_correctness(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    split in any::<usize>()
  ) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    // Production: combine partial checksums
    let crc_a = Crc16Ccitt::checksum(a);
    let crc_b = Crc16Ccitt::checksum(b);
    let combined = Crc16Ccitt::combine(crc_a, crc_b, b.len());

    // Reference: independent crc-fast implementation over full data
    let expected = crc_fast::checksum(CrcAlgorithm::Crc16IbmSdlc, &data) as u16;

    prop_assert_eq!(combined, expected,
      "combine(crc(A), crc(B), len(B)) != crc(A||B) at split {}/{}",
      split, data.len());
  }

  #[test]
  fn crc16_ibm_combine_correctness(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    split in any::<usize>()
  ) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc16Ibm::checksum(a);
    let crc_b = Crc16Ibm::checksum(b);
    let combined = Crc16Ibm::combine(crc_a, crc_b, b.len());

    let expected = crc_fast::checksum(CrcAlgorithm::Crc16Arc, &data) as u16;

    prop_assert_eq!(combined, expected,
      "combine(crc(A), crc(B), len(B)) != crc(A||B) at split {}/{}",
      split, data.len());
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-24 Combine Correctness
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc24_openpgp_combine_correctness(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    split in any::<usize>()
  ) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc24OpenPgp::checksum(a);
    let crc_b = Crc24OpenPgp::checksum(b);
    let combined = Crc24OpenPgp::combine(crc_a, crc_b, b.len());

    let expected = REF_CRC24_OPENPGP.checksum(&data) & 0x00FF_FFFF;

    prop_assert_eq!(combined, expected,
      "combine(crc(A), crc(B), len(B)) != crc(A||B) at split {}/{}",
      split, data.len());
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-32 Combine Correctness
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc32_ieee_combine_correctness(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    split in any::<usize>()
  ) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc32::checksum(a);
    let crc_b = Crc32::checksum(b);
    let combined = Crc32::combine(crc_a, crc_b, b.len());

    let expected = crc_fast::checksum(CrcAlgorithm::Crc32IsoHdlc, &data) as u32;

    prop_assert_eq!(combined, expected,
      "combine(crc(A), crc(B), len(B)) != crc(A||B) at split {}/{}",
      split, data.len());
  }

  #[test]
  fn crc32c_combine_correctness(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    split in any::<usize>()
  ) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc32C::checksum(a);
    let crc_b = Crc32C::checksum(b);
    let combined = Crc32C::combine(crc_a, crc_b, b.len());

    let expected = crc_fast::checksum(CrcAlgorithm::Crc32Iscsi, &data) as u32;

    prop_assert_eq!(combined, expected,
      "combine(crc(A), crc(B), len(B)) != crc(A||B) at split {}/{}",
      split, data.len());
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-64 Combine Correctness
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc64_xz_combine_correctness(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    split in any::<usize>()
  ) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc64::checksum(a);
    let crc_b = Crc64::checksum(b);
    let combined = Crc64::combine(crc_a, crc_b, b.len());

    let expected = crc_fast::checksum(CrcAlgorithm::Crc64Xz, &data);

    prop_assert_eq!(combined, expected,
      "combine(crc(A), crc(B), len(B)) != crc(A||B) at split {}/{}",
      split, data.len());
  }

  #[test]
  fn crc64_nvme_combine_correctness(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    split in any::<usize>()
  ) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc64Nvme::checksum(a);
    let crc_b = Crc64Nvme::checksum(b);
    let combined = Crc64Nvme::combine(crc_a, crc_b, b.len());

    let expected = crc_fast::checksum(CrcAlgorithm::Crc64Nvme, &data);

    prop_assert_eq!(combined, expected,
      "combine(crc(A), crc(B), len(B)) != crc(A||B) at split {}/{}",
      split, data.len());
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunking Equivalence Tests
// ─────────────────────────────────────────────────────────────────────────────
//
// These tests prove that any chunking of input through the streaming API
// produces the same result as a one-shot computation. This is a fundamental
// correctness property: the streaming implementation must be independent of
// how the input is partitioned into update() calls.
//
// We use arbitrary chunk patterns (variable-size chunks) to stress-test
// boundary conditions in SIMD kernels and buffering logic.
// ─────────────────────────────────────────────────────────────────────────────

/// Apply an arbitrary chunk pattern to data and feed it to a hasher.
///
/// The chunk pattern is a sequence of chunk sizes. We iterate through the
/// pattern cyclically until all data is consumed.
fn apply_chunking<C: Checksum>(data: &[u8], chunk_pattern: &[usize]) -> C::Output {
  let mut hasher = C::new();

  if chunk_pattern.is_empty() || data.is_empty() {
    hasher.update(data);
    return hasher.finalize();
  }

  let mut offset = 0;
  let mut pattern_idx = 0;

  while offset < data.len() {
    let chunk_size = chunk_pattern[pattern_idx].max(1); // Ensure at least 1 byte
    let end = (offset.strict_add(chunk_size)).min(data.len());
    hasher.update(&data[offset..end]);
    offset = end;
    pattern_idx = pattern_idx.strict_add(1) % chunk_pattern.len();
  }

  hasher.finalize()
}

/// Apply an arbitrary chunk pattern, but feed all produced chunks in one `update_vectored` call.
fn apply_chunking_vectored<C: Checksum>(data: &[u8], chunk_pattern: &[usize]) -> C::Output {
  let mut hasher = C::new();

  if chunk_pattern.is_empty() || data.is_empty() {
    hasher.update(data);
    return hasher.finalize();
  }

  let mut chunks: Vec<&[u8]> = Vec::new();
  let mut offset = 0;
  let mut pattern_idx = 0;

  while offset < data.len() {
    let chunk_size = chunk_pattern[pattern_idx].max(1); // Ensure at least 1 byte
    let end = (offset.strict_add(chunk_size)).min(data.len());
    chunks.push(&data[offset..end]);
    offset = end;
    pattern_idx = pattern_idx.strict_add(1) % chunk_pattern.len();
  }

  hasher.update_vectored(&chunks);
  hasher.finalize()
}

proptest! {
  #![proptest_config(ProptestConfig::with_cases(256))]

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16 Chunking Equivalence
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc16_ccitt_chunking_equivalence(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    chunk_pattern in proptest::collection::vec(1usize..=512, 1..=32)
  ) {
    let oneshot = Crc16Ccitt::checksum(&data);
    let streamed = apply_chunking::<Crc16Ccitt>(&data, &chunk_pattern);
    let vectored = apply_chunking_vectored::<Crc16Ccitt>(&data, &chunk_pattern);
    prop_assert_eq!(streamed, oneshot, "chunking pattern {:?} produced different result", chunk_pattern);
    prop_assert_eq!(vectored, oneshot, "vectored chunking pattern {:?} produced different result", chunk_pattern);
  }

  #[test]
  fn crc16_ibm_chunking_equivalence(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    chunk_pattern in proptest::collection::vec(1usize..=512, 1..=32)
  ) {
    let oneshot = Crc16Ibm::checksum(&data);
    let streamed = apply_chunking::<Crc16Ibm>(&data, &chunk_pattern);
    let vectored = apply_chunking_vectored::<Crc16Ibm>(&data, &chunk_pattern);
    prop_assert_eq!(streamed, oneshot, "chunking pattern {:?} produced different result", chunk_pattern);
    prop_assert_eq!(vectored, oneshot, "vectored chunking pattern {:?} produced different result", chunk_pattern);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-24 Chunking Equivalence
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc24_openpgp_chunking_equivalence(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    chunk_pattern in proptest::collection::vec(1usize..=512, 1..=32)
  ) {
    let oneshot = Crc24OpenPgp::checksum(&data);
    let streamed = apply_chunking::<Crc24OpenPgp>(&data, &chunk_pattern);
    let vectored = apply_chunking_vectored::<Crc24OpenPgp>(&data, &chunk_pattern);
    prop_assert_eq!(streamed, oneshot, "chunking pattern {:?} produced different result", chunk_pattern);
    prop_assert_eq!(vectored, oneshot, "vectored chunking pattern {:?} produced different result", chunk_pattern);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-32 Chunking Equivalence
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc32_ieee_chunking_equivalence(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    chunk_pattern in proptest::collection::vec(1usize..=512, 1..=32)
  ) {
    let oneshot = Crc32::checksum(&data);
    let streamed = apply_chunking::<Crc32>(&data, &chunk_pattern);
    let vectored = apply_chunking_vectored::<Crc32>(&data, &chunk_pattern);
    prop_assert_eq!(streamed, oneshot, "chunking pattern {:?} produced different result", chunk_pattern);
    prop_assert_eq!(vectored, oneshot, "vectored chunking pattern {:?} produced different result", chunk_pattern);
  }

  #[test]
  fn crc32c_chunking_equivalence(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    chunk_pattern in proptest::collection::vec(1usize..=512, 1..=32)
  ) {
    let oneshot = Crc32C::checksum(&data);
    let streamed = apply_chunking::<Crc32C>(&data, &chunk_pattern);
    let vectored = apply_chunking_vectored::<Crc32C>(&data, &chunk_pattern);
    prop_assert_eq!(streamed, oneshot, "chunking pattern {:?} produced different result", chunk_pattern);
    prop_assert_eq!(vectored, oneshot, "vectored chunking pattern {:?} produced different result", chunk_pattern);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-64 Chunking Equivalence
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc64_xz_chunking_equivalence(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    chunk_pattern in proptest::collection::vec(1usize..=512, 1..=32)
  ) {
    let oneshot = Crc64::checksum(&data);
    let streamed = apply_chunking::<Crc64>(&data, &chunk_pattern);
    let vectored = apply_chunking_vectored::<Crc64>(&data, &chunk_pattern);
    prop_assert_eq!(streamed, oneshot, "chunking pattern {:?} produced different result", chunk_pattern);
    prop_assert_eq!(vectored, oneshot, "vectored chunking pattern {:?} produced different result", chunk_pattern);
  }

  #[test]
  fn crc64_nvme_chunking_equivalence(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    chunk_pattern in proptest::collection::vec(1usize..=512, 1..=32)
  ) {
    let oneshot = Crc64Nvme::checksum(&data);
    let streamed = apply_chunking::<Crc64Nvme>(&data, &chunk_pattern);
    let vectored = apply_chunking_vectored::<Crc64Nvme>(&data, &chunk_pattern);
    prop_assert_eq!(streamed, oneshot, "chunking pattern {:?} produced different result", chunk_pattern);
    prop_assert_eq!(vectored, oneshot, "vectored chunking pattern {:?} produced different result", chunk_pattern);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge Case Tests
// ─────────────────────────────────────────────────────────────────────────────
//
// Focused tests for boundary conditions that property tests might not hit
// frequently enough: empty data, single bytes, powers of two, etc.
// ─────────────────────────────────────────────────────────────────────────────

/// Test combine at all split points for small data.
/// This exhaustively tests edge cases that random sampling might miss.
macro_rules! test_combine_all_splits {
  ($name:ident, $crc_type:ty, $expected:expr) => {
    #[test]
    fn $name() {
      let expected_for = $expected;

      // Test various small sizes including edge cases
      for size in [0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256] {
        let data: Vec<u8> = (0..size).map(|i| (i as u8).wrapping_mul(17)).collect();

        for split in 0..=data.len() {
          let (a, b) = data.split_at(split);

          let crc_a = <$crc_type>::checksum(a);
          let crc_b = <$crc_type>::checksum(b);
          let combined = <$crc_type>::combine(crc_a, crc_b, b.len());

          let expected = expected_for(&data);

          assert_eq!(
            combined,
            expected,
            "{}: combine failed at split {}/{} for size {}",
            stringify!($crc_type),
            split,
            data.len(),
            size
          );
        }
      }
    }
  };
}

test_combine_all_splits!(crc16_ccitt_combine_all_splits, Crc16Ccitt, |data: &[u8]| {
  crc_fast::checksum(CrcAlgorithm::Crc16IbmSdlc, data) as u16
});
test_combine_all_splits!(crc16_ibm_combine_all_splits, Crc16Ibm, |data: &[u8]| {
  crc_fast::checksum(CrcAlgorithm::Crc16Arc, data) as u16
});
test_combine_all_splits!(crc24_openpgp_combine_all_splits, Crc24OpenPgp, |data: &[u8]| {
  REF_CRC24_OPENPGP.checksum(data) & 0x00FF_FFFF
});
test_combine_all_splits!(crc32_ieee_combine_all_splits, Crc32, |data: &[u8]| {
  crc_fast::checksum(CrcAlgorithm::Crc32IsoHdlc, data) as u32
});
test_combine_all_splits!(crc32c_combine_all_splits, Crc32C, |data: &[u8]| {
  crc_fast::checksum(CrcAlgorithm::Crc32Iscsi, data) as u32
});
test_combine_all_splits!(crc64_xz_combine_all_splits, Crc64, |data: &[u8]| {
  crc_fast::checksum(CrcAlgorithm::Crc64Xz, data)
});
test_combine_all_splits!(crc64_nvme_combine_all_splits, Crc64Nvme, |data: &[u8]| {
  crc_fast::checksum(CrcAlgorithm::Crc64Nvme, data)
});

/// Test chunking with specific patterns known to stress SIMD boundaries.
macro_rules! test_chunking_edge_cases {
  ($name:ident, $crc_type:ty) => {
    #[test]
    fn $name() {
      // Test data that spans common SIMD register sizes
      for size in [
        0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 512, 1024,
      ] {
        let data: Vec<u8> = (0..size).map(|i| (i as u8).wrapping_mul(23)).collect();
        let expected = <$crc_type>::checksum(&data);

        // Byte-at-a-time
        let result = apply_chunking::<$crc_type>(&data, &[1]);
        assert_eq!(
          result,
          expected,
          "{}: byte-at-a-time failed for size {}",
          stringify!($crc_type),
          size
        );

        // Powers of two
        for chunk_size in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
          let result = apply_chunking::<$crc_type>(&data, &[chunk_size]);
          assert_eq!(
            result,
            expected,
            "{}: chunk size {} failed for size {}",
            stringify!($crc_type),
            chunk_size,
            size
          );
        }

        // Alternating small/large chunks (stresses buffering)
        let result = apply_chunking::<$crc_type>(&data, &[1, 64, 3, 128, 7, 256]);
        assert_eq!(
          result,
          expected,
          "{}: alternating pattern failed for size {}",
          stringify!($crc_type),
          size
        );

        // Prime-sized chunks (misalign with SIMD boundaries)
        let result = apply_chunking::<$crc_type>(&data, &[7, 13, 17, 23, 31]);
        assert_eq!(
          result,
          expected,
          "{}: prime pattern failed for size {}",
          stringify!($crc_type),
          size
        );
      }
    }
  };
}

test_chunking_edge_cases!(crc16_ccitt_chunking_edge_cases, Crc16Ccitt);
test_chunking_edge_cases!(crc16_ibm_chunking_edge_cases, Crc16Ibm);
test_chunking_edge_cases!(crc24_openpgp_chunking_edge_cases, Crc24OpenPgp);
test_chunking_edge_cases!(crc32_ieee_chunking_edge_cases, Crc32);
test_chunking_edge_cases!(crc32c_chunking_edge_cases, Crc32C);
test_chunking_edge_cases!(crc64_xz_chunking_edge_cases, Crc64);
test_chunking_edge_cases!(crc64_nvme_chunking_edge_cases, Crc64Nvme);

// ─────────────────────────────────────────────────────────────────────────────
// Resume Correctness Tests
// ─────────────────────────────────────────────────────────────────────────────
//
// The resume() API allows continuing a CRC computation from a previous
// checksum value. This tests that:
//   resume(crc(A)).update(B).finalize() == crc(A || B)
//
// This is equivalent to the combine property but uses streaming instead of
// the mathematical combine function.
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
  #![proptest_config(ProptestConfig::with_cases(256))]

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16 Resume Correctness
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc16_ccitt_resume_correctness(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    split in any::<usize>()
  ) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc16Ccitt::checksum(a);
    let mut resumed = Crc16Ccitt::resume(crc_a);
    resumed.update(b);
    let result = resumed.finalize();

    let expected = Crc16Ccitt::checksum(&data);
    prop_assert_eq!(result, expected, "resume(crc(A)).update(B) != crc(A||B)");
  }

  #[test]
  fn crc16_ibm_resume_correctness(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    split in any::<usize>()
  ) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc16Ibm::checksum(a);
    let mut resumed = Crc16Ibm::resume(crc_a);
    resumed.update(b);
    let result = resumed.finalize();

    let expected = Crc16Ibm::checksum(&data);
    prop_assert_eq!(result, expected, "resume(crc(A)).update(B) != crc(A||B)");
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-24 Resume Correctness
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc24_openpgp_resume_correctness(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    split in any::<usize>()
  ) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc24OpenPgp::checksum(a);
    let mut resumed = Crc24OpenPgp::resume(crc_a);
    resumed.update(b);
    let result = resumed.finalize();

    let expected = Crc24OpenPgp::checksum(&data);
    prop_assert_eq!(result, expected, "resume(crc(A)).update(B) != crc(A||B)");
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-32 Resume Correctness
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc32_ieee_resume_correctness(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    split in any::<usize>()
  ) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc32::checksum(a);
    let mut resumed = Crc32::resume(crc_a);
    resumed.update(b);
    let result = resumed.finalize();

    let expected = Crc32::checksum(&data);
    prop_assert_eq!(result, expected, "resume(crc(A)).update(B) != crc(A||B)");
  }

  #[test]
  fn crc32c_resume_correctness(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    split in any::<usize>()
  ) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc32C::checksum(a);
    let mut resumed = Crc32C::resume(crc_a);
    resumed.update(b);
    let result = resumed.finalize();

    let expected = Crc32C::checksum(&data);
    prop_assert_eq!(result, expected, "resume(crc(A)).update(B) != crc(A||B)");
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-64 Resume Correctness
  // ─────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc64_xz_resume_correctness(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    split in any::<usize>()
  ) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc64::checksum(a);
    let mut resumed = Crc64::resume(crc_a);
    resumed.update(b);
    let result = resumed.finalize();

    let expected = Crc64::checksum(&data);
    prop_assert_eq!(result, expected, "resume(crc(A)).update(B) != crc(A||B)");
  }

  #[test]
  fn crc64_nvme_resume_correctness(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    split in any::<usize>()
  ) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc64Nvme::checksum(a);
    let mut resumed = Crc64Nvme::resume(crc_a);
    resumed.update(b);
    let result = resumed.finalize();

    let expected = Crc64Nvme::checksum(&data);
    prop_assert_eq!(result, expected, "resume(crc(A)).update(B) != crc(A||B)");
  }
}
