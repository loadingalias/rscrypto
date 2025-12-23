//! Property-based tests for CRC-32 implementations.
//!
//! These tests verify:
//! 1. Dispatched kernels match portable implementation
//! 2. Our results match the `crc-fast` reference crate
//! 3. Streaming, combine, and resume all produce consistent results

extern crate std;

use crc_fast::CrcAlgorithm;
use proptest::prelude::*;

use super::*;

proptest! {
  /// Verify dispatched CRC32C matches the portable slice-by-16 implementation.
  #[test]
  fn crc32c_matches_portable(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let dispatched = Crc32c::checksum(&data);
    let portable = portable::crc32_slice16(!0, &data, &kernel_tables::CASTAGNOLI_TABLES_16) ^ !0;
    prop_assert_eq!(dispatched, portable);
  }

  /// Verify dispatched CRC32 IEEE matches the portable slice-by-16 implementation.
  #[test]
  fn crc32_ieee_matches_portable(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let dispatched = Crc32::checksum(&data);
    let portable = portable::crc32_slice16(!0, &data, &kernel_tables::IEEE_TABLES_16) ^ !0;
    prop_assert_eq!(dispatched, portable);
  }

  /// Verify our CRC32C matches the `crc-fast` reference (CRC32/ISCSI = Castagnoli).
  #[test]
  fn crc32c_matches_crc_fast(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc32c::checksum(&data);
    let reference = crc_fast::checksum(CrcAlgorithm::Crc32Iscsi, &data) as u32;
    prop_assert_eq!(ours, reference);
  }

  /// Verify our CRC32 IEEE matches the `crc-fast` reference (CRC32/ISO-HDLC = IEEE).
  #[test]
  fn crc32_ieee_matches_crc_fast(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc32::checksum(&data);
    let reference = crc_fast::checksum(CrcAlgorithm::Crc32IsoHdlc, &data) as u32;
    prop_assert_eq!(ours, reference);
  }

  /// Verify CRC32C streaming, combine, and resume all produce consistent results.
  #[test]
  fn crc32c_streaming_and_combine(data in proptest::collection::vec(any::<u8>(), 0..=4096), split in any::<usize>(), chunk in 1usize..=257) {
    let split = split % (data.len() + 1);
    let (a, b) = data.split_at(split);

    let oneshot = Crc32c::checksum(&data);

    // Streaming must match oneshot
    let mut hasher = Crc32c::new();
    for part in a.chunks(chunk) {
      hasher.update(part);
    }
    for part in b.chunks(chunk) {
      hasher.update(part);
    }
    prop_assert_eq!(hasher.finalize(), oneshot);

    // Combine must match oneshot
    let crc_a = Crc32c::checksum(a);
    let crc_b = Crc32c::checksum(b);
    let combined = Crc32c::combine(crc_a, crc_b, b.len());
    prop_assert_eq!(combined, oneshot);

    // Resume must match oneshot
    let mut resumed = Crc32c::resume(crc_a);
    resumed.update(b);
    prop_assert_eq!(resumed.finalize(), oneshot);
  }

  /// Verify CRC32 IEEE streaming, combine, and resume all produce consistent results.
  #[test]
  fn crc32_ieee_streaming_and_combine(data in proptest::collection::vec(any::<u8>(), 0..=4096), split in any::<usize>(), chunk in 1usize..=257) {
    let split = split % (data.len() + 1);
    let (a, b) = data.split_at(split);

    let oneshot = Crc32::checksum(&data);

    // Streaming must match oneshot
    let mut hasher = Crc32::new();
    for part in a.chunks(chunk) {
      hasher.update(part);
    }
    for part in b.chunks(chunk) {
      hasher.update(part);
    }
    prop_assert_eq!(hasher.finalize(), oneshot);

    // Combine must match oneshot
    let crc_a = Crc32::checksum(a);
    let crc_b = Crc32::checksum(b);
    let combined = Crc32::combine(crc_a, crc_b, b.len());
    prop_assert_eq!(combined, oneshot);

    // Resume must match oneshot
    let mut resumed = Crc32::resume(crc_a);
    resumed.update(b);
    prop_assert_eq!(resumed.finalize(), oneshot);
  }
}
