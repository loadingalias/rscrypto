extern crate std;

use crc_fast::CrcAlgorithm;
use crc64fast as ref_crc64fast;
use crc64fast_nvme as ref_crc64fast_nvme;
use proptest::prelude::*;

use super::*;

proptest! {
  #[test]
  fn crc64_xz_matches_portable(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc64::checksum(&data);
    let portable = portable::crc64_slice16_xz(!0, &data) ^ !0;
    prop_assert_eq!(ours, portable);
  }

  #[test]
  fn crc64_nvme_matches_portable(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc64Nvme::checksum(&data);
    let portable = portable::crc64_slice16_nvme(!0, &data) ^ !0;
    prop_assert_eq!(ours, portable);
  }

  #[test]
  fn crc64_xz_matches_crc64fast(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc64::checksum(&data);
    let mut digest = ref_crc64fast::Digest::new();
    digest.write(&data);
    let reference = digest.sum64();
    prop_assert_eq!(ours, reference);
  }

  #[test]
  fn crc64_nvme_matches_crc64fast_nvme(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc64Nvme::checksum(&data);
    let mut digest = ref_crc64fast_nvme::Digest::new();
    digest.write(&data);
    let reference = digest.sum64();
    prop_assert_eq!(ours, reference);
  }

  // NOTE: crc64_xz_streaming_and_combine test removed - now covered by unified
  // tests in common/proptests.rs (combine_correctness, chunking_equivalence, resume_correctness)

  #[test]
  fn crc64_xz_streaming_matches_crc64fast(data in proptest::collection::vec(any::<u8>(), 0..=4096), chunk in 1usize..=257) {
    let mut ours = Crc64::new();
    let mut reference = ref_crc64fast::Digest::new();

    for part in data.chunks(chunk) {
      ours.update(part);
      reference.write(part);
    }

    prop_assert_eq!(ours.finalize(), reference.sum64());
  }

  // NOTE: crc64_nvme_streaming_and_combine test removed - now covered by unified
  // tests in common/proptests.rs (combine_correctness, chunking_equivalence, resume_correctness)

  #[test]
  fn crc64_nvme_streaming_matches_crc64fast_nvme(data in proptest::collection::vec(any::<u8>(), 0..=4096), chunk in 1usize..=257) {
    let mut ours = Crc64Nvme::new();
    let mut reference = ref_crc64fast_nvme::Digest::new();

    for part in data.chunks(chunk) {
      ours.update(part);
      reference.write(part);
    }

    prop_assert_eq!(ours.finalize(), reference.sum64());
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Cross-validation against crc-fast-rust
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc64_xz_matches_crc_fast_rust(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc64::checksum(&data);
    let reference = crc_fast::checksum(CrcAlgorithm::Crc64Xz, &data);
    prop_assert_eq!(ours, reference);
  }

  #[test]
  fn crc64_nvme_matches_crc_fast_rust(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc64Nvme::checksum(&data);
    let reference = crc_fast::checksum(CrcAlgorithm::Crc64Nvme, &data);
    prop_assert_eq!(ours, reference);
  }

  #[test]
  fn crc64_xz_streaming_matches_crc_fast_rust(data in proptest::collection::vec(any::<u8>(), 0..=4096), chunk in 1usize..=257) {
    let mut ours = Crc64::new();
    let mut reference = crc_fast::Digest::new(CrcAlgorithm::Crc64Xz);

    for part in data.chunks(chunk) {
      ours.update(part);
      reference.update(part);
    }

    prop_assert_eq!(ours.finalize(), reference.finalize());
  }

  #[test]
  fn crc64_nvme_streaming_matches_crc_fast_rust(data in proptest::collection::vec(any::<u8>(), 0..=4096), chunk in 1usize..=257) {
    let mut ours = Crc64Nvme::new();
    let mut reference = crc_fast::Digest::new(CrcAlgorithm::Crc64Nvme);

    for part in data.chunks(chunk) {
      ours.update(part);
      reference.update(part);
    }

    prop_assert_eq!(ours.finalize(), reference.finalize());
  }

  #[test]
  fn crc64_xz_combine_matches_crc_fast_rust(data in proptest::collection::vec(any::<u8>(), 0..=4096), split in any::<usize>()) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc64::checksum(a);
    let crc_b = Crc64::checksum(b);
    let combined = Crc64::combine(crc_a, crc_b, b.len());

    let ref_crc_a = crc_fast::checksum(CrcAlgorithm::Crc64Xz, a);
    let ref_crc_b = crc_fast::checksum(CrcAlgorithm::Crc64Xz, b);
    let ref_combined = crc_fast::checksum_combine(CrcAlgorithm::Crc64Xz, ref_crc_a, ref_crc_b, b.len() as u64);

    prop_assert_eq!(combined, ref_combined);
  }

  #[test]
  fn crc64_nvme_combine_matches_crc_fast_rust(data in proptest::collection::vec(any::<u8>(), 0..=4096), split in any::<usize>()) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc64Nvme::checksum(a);
    let crc_b = Crc64Nvme::checksum(b);
    let combined = Crc64Nvme::combine(crc_a, crc_b, b.len());

    let ref_crc_a = crc_fast::checksum(CrcAlgorithm::Crc64Nvme, a);
    let ref_crc_b = crc_fast::checksum(CrcAlgorithm::Crc64Nvme, b);
    let ref_combined = crc_fast::checksum_combine(CrcAlgorithm::Crc64Nvme, ref_crc_a, ref_crc_b, b.len() as u64);

    prop_assert_eq!(combined, ref_combined);
  }
}
