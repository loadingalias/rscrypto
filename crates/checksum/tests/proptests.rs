//! Property-based tests for CRC implementations.
//!
//! These tests verify invariants that must hold for all inputs, not just
//! specific test vectors. Uses proptest for randomized input generation.

use checksum::{Crc16CcittFalse, Crc16Ibm, Crc24, Crc32, Crc32c, Crc64, Crc64Nvme};
use proptest::prelude::*;
use traits::Checksum;

// Test Strategies

/// Generate arbitrary byte vectors up to 8KB.
fn arb_data() -> impl Strategy<Value = Vec<u8>> {
  prop::collection::vec(any::<u8>(), 0..8192)
}

/// Generate multiple split points for chunked testing.
fn arb_splits(len: usize, count: usize) -> impl Strategy<Value = Vec<usize>> {
  prop::collection::vec(0..=len, count).prop_map(move |mut splits| {
    splits.sort();
    splits.push(len);
    splits.dedup();
    splits
  })
}

// Generic Property Tests

/// Test that incremental updates produce the same result as one-shot.
fn prop_incremental_equals_oneshot<C: Checksum + Default + Clone>(data: &[u8], split: usize) -> bool {
  let split = split.min(data.len());
  let (a, b) = data.split_at(split);

  let oneshot = C::checksum(data);

  let mut incremental = C::new();
  incremental.update(a);
  incremental.update(b);

  incremental.finalize() == oneshot
}

/// Test that multiple incremental updates produce the same result.
fn prop_multi_incremental<C: Checksum + Default + Clone>(data: &[u8], splits: &[usize]) -> bool {
  let oneshot = C::checksum(data);

  let mut hasher = C::new();
  let mut prev = 0;
  for &split in splits {
    let split = split.min(data.len());
    if split > prev {
      hasher.update(&data[prev..split]);
      prev = split;
    }
  }
  if prev < data.len() {
    hasher.update(&data[prev..]);
  }

  hasher.finalize() == oneshot
}

/// Test that reset returns hasher to initial state.
fn prop_reset_works<C: Checksum + Default + Clone>(data: &[u8]) -> bool {
  let mut hasher = C::new();
  hasher.update(data);
  hasher.reset();
  hasher.update(data);

  hasher.finalize() == C::checksum(data)
}

// CRC32-C Property Tests

proptest! {
  #![proptest_config(ProptestConfig::with_cases(1000))]

  #[test]
  fn crc32c_incremental_equals_oneshot(data in arb_data(), split in 0..8192usize) {
    prop_assert!(prop_incremental_equals_oneshot::<Crc32c>(&data, split));
  }

  #[test]
  fn crc32c_multi_incremental(data in arb_data(), splits in arb_splits(8192, 5)) {
    prop_assert!(prop_multi_incremental::<Crc32c>(&data, &splits));
  }

  #[test]
  fn crc32c_reset(data in arb_data()) {
    prop_assert!(prop_reset_works::<Crc32c>(&data));
  }

  #[test]
  fn crc32c_combine_correctness(
    data in arb_data(),
    split in 0..8192usize
  ) {
    let split = split.min(data.len());
    let (a, b) = data.split_at(split);

    let crc_a = Crc32c::checksum(a);
    let crc_b = Crc32c::checksum(b);
    let crc_ab = Crc32c::checksum(&data);

    let combined = Crc32c::combine(crc_a, crc_b, b.len());
    prop_assert_eq!(combined, crc_ab);
  }

  #[test]
  fn crc32c_resume_correctness(
    data in arb_data(),
    split in 0..8192usize
  ) {
    let split = split.min(data.len());
    let (a, b) = data.split_at(split);

    let crc_a = Crc32c::checksum(a);
    let mut resumed = Crc32c::resume(crc_a);
    resumed.update(b);

    prop_assert_eq!(resumed.finalize(), Crc32c::checksum(&data));
  }

  #[test]
  fn crc32c_combine_associative(
    data in arb_data(),
    split1 in 0..4096usize,
    split2 in 0..4096usize
  ) {
    let split1 = split1.min(data.len());
    let split2 = (split1 + split2).min(data.len());

    let a = &data[..split1];
    let b = &data[split1..split2];
    let c = &data[split2..];

    let crc_a = Crc32c::checksum(a);
    let crc_b = Crc32c::checksum(b);
    let crc_c = Crc32c::checksum(c);

    // (crc_a || crc_b) || crc_c
    let ab = Crc32c::combine(crc_a, crc_b, b.len());
    let abc = Crc32c::combine(ab, crc_c, c.len());

    prop_assert_eq!(abc, Crc32c::checksum(&data));
  }
}

// CRC32 Property Tests

proptest! {
  #![proptest_config(ProptestConfig::with_cases(1000))]

  #[test]
  fn crc32_incremental_equals_oneshot(data in arb_data(), split in 0..8192usize) {
    prop_assert!(prop_incremental_equals_oneshot::<Crc32>(&data, split));
  }

  #[test]
  fn crc32_combine_correctness(
    data in arb_data(),
    split in 0..8192usize
  ) {
    let split = split.min(data.len());
    let (a, b) = data.split_at(split);

    let crc_a = Crc32::checksum(a);
    let crc_b = Crc32::checksum(b);
    let combined = Crc32::combine(crc_a, crc_b, b.len());

    prop_assert_eq!(combined, Crc32::checksum(&data));
  }

  #[test]
  fn crc32_resume_correctness(
    data in arb_data(),
    split in 0..8192usize
  ) {
    let split = split.min(data.len());
    let (a, b) = data.split_at(split);

    let crc_a = Crc32::checksum(a);
    let mut resumed = Crc32::resume(crc_a);
    resumed.update(b);

    prop_assert_eq!(resumed.finalize(), Crc32::checksum(&data));
  }
}

// CRC64 Property Tests

proptest! {
  #![proptest_config(ProptestConfig::with_cases(1000))]

  #[test]
  fn crc64_incremental_equals_oneshot(data in arb_data(), split in 0..8192usize) {
    prop_assert!(prop_incremental_equals_oneshot::<Crc64>(&data, split));
  }

  #[test]
  fn crc64_combine_correctness(
    data in arb_data(),
    split in 0..8192usize
  ) {
    let split = split.min(data.len());
    let (a, b) = data.split_at(split);

    let crc_a = Crc64::checksum(a);
    let crc_b = Crc64::checksum(b);
    let combined = Crc64::combine(crc_a, crc_b, b.len());

    prop_assert_eq!(combined, Crc64::checksum(&data));
  }

  #[test]
  fn crc64_resume_correctness(
    data in arb_data(),
    split in 0..8192usize
  ) {
    let split = split.min(data.len());
    let (a, b) = data.split_at(split);

    let crc_a = Crc64::checksum(a);
    let mut resumed = Crc64::resume(crc_a);
    resumed.update(b);

    prop_assert_eq!(resumed.finalize(), Crc64::checksum(&data));
  }
}

// CRC64/NVME Property Tests

proptest! {
  #![proptest_config(ProptestConfig::with_cases(1000))]

  #[test]
  fn crc64_nvme_incremental_equals_oneshot(data in arb_data(), split in 0..8192usize) {
    prop_assert!(prop_incremental_equals_oneshot::<Crc64Nvme>(&data, split));
  }

  #[test]
  fn crc64_nvme_combine_correctness(
    data in arb_data(),
    split in 0..8192usize
  ) {
    let split = split.min(data.len());
    let (a, b) = data.split_at(split);

    let crc_a = Crc64Nvme::checksum(a);
    let crc_b = Crc64Nvme::checksum(b);
    let combined = Crc64Nvme::combine(crc_a, crc_b, b.len());

    prop_assert_eq!(combined, Crc64Nvme::checksum(&data));
  }
}

// CRC16 Property Tests

proptest! {
  #![proptest_config(ProptestConfig::with_cases(1000))]

  #[test]
  fn crc16_ibm_incremental_equals_oneshot(data in arb_data(), split in 0..8192usize) {
    prop_assert!(prop_incremental_equals_oneshot::<Crc16Ibm>(&data, split));
  }

  #[test]
  fn crc16_ibm_combine_correctness(
    data in arb_data(),
    split in 0..8192usize
  ) {
    let split = split.min(data.len());
    let (a, b) = data.split_at(split);

    let crc_a = Crc16Ibm::checksum(a);
    let crc_b = Crc16Ibm::checksum(b);
    let combined = Crc16Ibm::combine(crc_a, crc_b, b.len());

    prop_assert_eq!(combined, Crc16Ibm::checksum(&data));
  }

  #[test]
  fn crc16_ccitt_false_incremental_equals_oneshot(data in arb_data(), split in 0..8192usize) {
    prop_assert!(prop_incremental_equals_oneshot::<Crc16CcittFalse>(&data, split));
  }

  #[test]
  fn crc16_ccitt_false_combine_correctness(
    data in arb_data(),
    split in 0..8192usize
  ) {
    let split = split.min(data.len());
    let (a, b) = data.split_at(split);

    let crc_a = Crc16CcittFalse::checksum(a);
    let crc_b = Crc16CcittFalse::checksum(b);
    let combined = Crc16CcittFalse::combine(crc_a, crc_b, b.len());

    prop_assert_eq!(combined, Crc16CcittFalse::checksum(&data));
  }
}

// CRC24 Property Tests

proptest! {
  #![proptest_config(ProptestConfig::with_cases(1000))]

  #[test]
  fn crc24_incremental_equals_oneshot(data in arb_data(), split in 0..8192usize) {
    prop_assert!(prop_incremental_equals_oneshot::<Crc24>(&data, split));
  }

  #[test]
  fn crc24_combine_correctness(
    data in arb_data(),
    split in 0..8192usize
  ) {
    let split = split.min(data.len());
    let (a, b) = data.split_at(split);

    let crc_a = Crc24::checksum(a);
    let crc_b = Crc24::checksum(b);
    let combined = Crc24::combine(crc_a, crc_b, b.len());

    prop_assert_eq!(combined, Crc24::checksum(&data));
  }
}
