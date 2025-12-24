//! Property-based tests for CRC-32 implementations.
//!
//! These tests validate our implementations against reference crates:
//! - `crc32fast`: Reference for CRC-32-IEEE (Ethernet/ZIP/PNG)
//! - `crc-fast` (Crc32Iscsi): Reference for CRC-32C (iSCSI/ext4/Btrfs)

extern crate std;

use crc_fast::CrcAlgorithm;
use proptest::prelude::*;

use super::*;
use crate::{Checksum, ChecksumCombine};

proptest! {
  // ─────────────────────────────────────────────────────────────────────────────
  // CRC-32-IEEE Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc32_ieee_matches_portable(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc32::checksum(&data);
    let portable = portable::crc32_ieee_slice16(!0, &data) ^ !0;
    prop_assert_eq!(ours, portable);
  }

  #[test]
  fn crc32_ieee_matches_crc32fast(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc32::checksum(&data);
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&data);
    let reference = hasher.finalize();
    prop_assert_eq!(ours, reference);
  }

  #[test]
  fn crc32_ieee_matches_crc_fast(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc32::checksum(&data);
    let reference = crc_fast::checksum(CrcAlgorithm::Crc32IsoHdlc, &data) as u32;
    prop_assert_eq!(ours, reference);
  }

  #[test]
  fn crc32_ieee_streaming_and_combine(data in proptest::collection::vec(any::<u8>(), 0..=4096), split in any::<usize>(), chunk in 1usize..=257) {
    let split = split % (data.len() + 1);
    let (a, b) = data.split_at(split);

    let oneshot = Crc32::checksum(&data);

    // Streaming with chunks
    let mut hasher = Crc32::new();
    for part in a.chunks(chunk) {
      hasher.update(part);
    }
    for part in b.chunks(chunk) {
      hasher.update(part);
    }
    prop_assert_eq!(hasher.finalize(), oneshot);

    // Combine
    let crc_a = Crc32::checksum(a);
    let crc_b = Crc32::checksum(b);
    let combined = Crc32::combine(crc_a, crc_b, b.len());
    prop_assert_eq!(combined, oneshot);

    // Resume
    let mut resumed = Crc32::resume(crc_a);
    resumed.update(b);
    prop_assert_eq!(resumed.finalize(), oneshot);
  }

  #[test]
  fn crc32_ieee_streaming_matches_crc32fast(data in proptest::collection::vec(any::<u8>(), 0..=4096), chunk in 1usize..=257) {
    let mut ours = Crc32::new();
    let mut reference = crc32fast::Hasher::new();

    for part in data.chunks(chunk) {
      ours.update(part);
      reference.update(part);
    }

    prop_assert_eq!(ours.finalize(), reference.finalize());
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // CRC-32C Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc32c_matches_portable(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc32c::checksum(&data);
    let portable = portable::crc32c_slice16(!0, &data) ^ !0;
    prop_assert_eq!(ours, portable);
  }

  #[test]
  fn crc32c_matches_crc_fast_iscsi(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc32c::checksum(&data);
    let reference = crc_fast::checksum(CrcAlgorithm::Crc32Iscsi, &data) as u32;
    prop_assert_eq!(ours, reference);
  }

  #[test]
  fn crc32c_streaming_and_combine(data in proptest::collection::vec(any::<u8>(), 0..=4096), split in any::<usize>(), chunk in 1usize..=257) {
    let split = split % (data.len() + 1);
    let (a, b) = data.split_at(split);

    let oneshot = Crc32c::checksum(&data);

    // Streaming with chunks
    let mut hasher = Crc32c::new();
    for part in a.chunks(chunk) {
      hasher.update(part);
    }
    for part in b.chunks(chunk) {
      hasher.update(part);
    }
    prop_assert_eq!(hasher.finalize(), oneshot);

    // Combine
    let crc_a = Crc32c::checksum(a);
    let crc_b = Crc32c::checksum(b);
    let combined = Crc32c::combine(crc_a, crc_b, b.len());
    prop_assert_eq!(combined, oneshot);

    // Resume
    let mut resumed = Crc32c::resume(crc_a);
    resumed.update(b);
    prop_assert_eq!(resumed.finalize(), oneshot);
  }

  #[test]
  fn crc32c_streaming_matches_crc_fast(data in proptest::collection::vec(any::<u8>(), 0..=4096), chunk in 1usize..=257) {
    let mut ours = Crc32c::new();
    let mut reference = crc_fast::Digest::new(CrcAlgorithm::Crc32Iscsi);

    for part in data.chunks(chunk) {
      ours.update(part);
      reference.update(part);
    }

    prop_assert_eq!(ours.finalize(), reference.finalize() as u32);
  }
}
