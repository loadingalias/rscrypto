extern crate std;

use crc_fast::CrcAlgorithm;
use proptest::prelude::*;

use super::*;

proptest! {
  #[test]
  fn crc32_matches_portable(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc32::checksum(&data);
    let portable = portable::crc32_slice16_ieee(!0, &data) ^ !0;
    prop_assert_eq!(ours, portable);
  }

  #[test]
  fn crc32c_matches_portable(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc32C::checksum(&data);
    let portable = portable::crc32c_slice16(!0, &data) ^ !0;
    prop_assert_eq!(ours, portable);
  }

  // NOTE: crc32_streaming_and_combine and crc32c_streaming_and_combine tests removed -
  // now covered by unified tests in common/proptests.rs (combine_correctness,
  // chunking_equivalence, resume_correctness)

  // ─────────────────────────────────────────────────────────────────────────────
  // Cross-validation against crc-fast-rust
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc32_matches_crc_fast_rust(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc32::checksum(&data);
    let reference = crc_fast::checksum(CrcAlgorithm::Crc32IsoHdlc, &data) as u32;
    prop_assert_eq!(ours, reference);
  }

  #[test]
  fn crc32c_matches_crc_fast_rust(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc32C::checksum(&data);
    let reference = crc_fast::checksum(CrcAlgorithm::Crc32Iscsi, &data) as u32;
    prop_assert_eq!(ours, reference);
  }

  #[test]
  fn crc32_streaming_matches_crc_fast_rust(data in proptest::collection::vec(any::<u8>(), 0..=4096), chunk in 1usize..=257) {
    let mut ours = Crc32::new();
    let mut reference = crc_fast::Digest::new(CrcAlgorithm::Crc32IsoHdlc);

    for part in data.chunks(chunk) {
      ours.update(part);
      reference.update(part);
    }

    prop_assert_eq!(ours.finalize(), reference.finalize() as u32);
  }

  #[test]
  fn crc32c_streaming_matches_crc_fast_rust(data in proptest::collection::vec(any::<u8>(), 0..=4096), chunk in 1usize..=257) {
    let mut ours = Crc32C::new();
    let mut reference = crc_fast::Digest::new(CrcAlgorithm::Crc32Iscsi);

    for part in data.chunks(chunk) {
      ours.update(part);
      reference.update(part);
    }

    prop_assert_eq!(ours.finalize(), reference.finalize() as u32);
  }

  #[test]
  fn crc32_combine_matches_crc_fast_rust(data in proptest::collection::vec(any::<u8>(), 0..=4096), split in any::<usize>()) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc32::checksum(a);
    let crc_b = Crc32::checksum(b);
    let combined = Crc32::combine(crc_a, crc_b, b.len());

    let ref_crc_a = crc_fast::checksum(CrcAlgorithm::Crc32IsoHdlc, a);
    let ref_crc_b = crc_fast::checksum(CrcAlgorithm::Crc32IsoHdlc, b);
    let ref_combined = crc_fast::checksum_combine(CrcAlgorithm::Crc32IsoHdlc, ref_crc_a, ref_crc_b, b.len() as u64) as u32;

    prop_assert_eq!(combined, ref_combined);
  }

  #[test]
  fn crc32c_combine_matches_crc_fast_rust(data in proptest::collection::vec(any::<u8>(), 0..=4096), split in any::<usize>()) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc32C::checksum(a);
    let crc_b = Crc32C::checksum(b);
    let combined = Crc32C::combine(crc_a, crc_b, b.len());

    let ref_crc_a = crc_fast::checksum(CrcAlgorithm::Crc32Iscsi, a);
    let ref_crc_b = crc_fast::checksum(CrcAlgorithm::Crc32Iscsi, b);
    let ref_combined = crc_fast::checksum_combine(CrcAlgorithm::Crc32Iscsi, ref_crc_a, ref_crc_b, b.len() as u64) as u32;

    prop_assert_eq!(combined, ref_combined);
  }
}

#[test]
fn test_vectors_crc32() {
  assert_eq!(Crc32::checksum(b"123456789"), 0xCBF4_3926);
}

#[test]
fn test_vectors_crc32c() {
  assert_eq!(Crc32C::checksum(b"123456789"), 0xE306_9283);
}
