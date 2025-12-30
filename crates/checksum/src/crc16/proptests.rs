extern crate std;

use crc_fast::CrcAlgorithm;
use proptest::prelude::*;

use super::*;

proptest! {
  #[test]
  fn crc16_ccitt_matches_portable(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc16Ccitt::checksum(&data);
    let portable = portable::crc16_ccitt_slice8(0xFFFF, &data) ^ 0xFFFF;
    prop_assert_eq!(ours, portable);
  }

  #[test]
  fn crc16_ibm_matches_portable(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc16Ibm::checksum(&data);
    let portable = portable::crc16_ibm_slice8(0, &data);
    prop_assert_eq!(ours, portable);
  }

  // NOTE: crc16_ccitt_streaming_and_combine and crc16_ibm_streaming_and_combine tests removed -
  // now covered by unified tests in common/proptests.rs (combine_correctness,
  // chunking_equivalence, resume_correctness)

  // ─────────────────────────────────────────────────────────────────────────────
  // Cross-validation against crc-fast-rust
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn crc16_ccitt_matches_crc_fast_rust(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc16Ccitt::checksum(&data);
    let reference = crc_fast::checksum(CrcAlgorithm::Crc16IbmSdlc, &data) as u16;
    prop_assert_eq!(ours, reference);
  }

  #[test]
  fn crc16_ibm_matches_crc_fast_rust(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc16Ibm::checksum(&data);
    let reference = crc_fast::checksum(CrcAlgorithm::Crc16Arc, &data) as u16;
    prop_assert_eq!(ours, reference);
  }

  #[test]
  fn crc16_ccitt_streaming_matches_crc_fast_rust(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    chunk in 1usize..=257
  ) {
    let mut ours = Crc16Ccitt::new();
    let mut reference = crc_fast::Digest::new(CrcAlgorithm::Crc16IbmSdlc);

    for part in data.chunks(chunk) {
      ours.update(part);
      reference.update(part);
    }

    prop_assert_eq!(ours.finalize(), reference.finalize() as u16);
  }

  #[test]
  fn crc16_ibm_streaming_matches_crc_fast_rust(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    chunk in 1usize..=257
  ) {
    let mut ours = Crc16Ibm::new();
    let mut reference = crc_fast::Digest::new(CrcAlgorithm::Crc16Arc);

    for part in data.chunks(chunk) {
      ours.update(part);
      reference.update(part);
    }

    prop_assert_eq!(ours.finalize(), reference.finalize() as u16);
  }

  #[test]
  fn crc16_ccitt_combine_matches_crc_fast_rust(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    split in any::<usize>()
  ) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc16Ccitt::checksum(a);
    let crc_b = Crc16Ccitt::checksum(b);
    let combined = Crc16Ccitt::combine(crc_a, crc_b, b.len());

    let ref_crc_a = crc_fast::checksum(CrcAlgorithm::Crc16IbmSdlc, a);
    let ref_crc_b = crc_fast::checksum(CrcAlgorithm::Crc16IbmSdlc, b);
    let ref_combined =
      crc_fast::checksum_combine(CrcAlgorithm::Crc16IbmSdlc, ref_crc_a, ref_crc_b, b.len() as u64) as u16;

    prop_assert_eq!(combined, ref_combined);
  }

  #[test]
  fn crc16_ibm_combine_matches_crc_fast_rust(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    split in any::<usize>()
  ) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc16Ibm::checksum(a);
    let crc_b = Crc16Ibm::checksum(b);
    let combined = Crc16Ibm::combine(crc_a, crc_b, b.len());

    let ref_crc_a = crc_fast::checksum(CrcAlgorithm::Crc16Arc, a);
    let ref_crc_b = crc_fast::checksum(CrcAlgorithm::Crc16Arc, b);
    let ref_combined =
      crc_fast::checksum_combine(CrcAlgorithm::Crc16Arc, ref_crc_a, ref_crc_b, b.len() as u64) as u16;

    prop_assert_eq!(combined, ref_combined);
  }
}

#[test]
fn test_vectors_crc16_ccitt() {
  assert_eq!(Crc16Ccitt::checksum(b"123456789"), 0x906E);
}

#[test]
fn test_vectors_crc16_ibm() {
  assert_eq!(Crc16Ibm::checksum(b"123456789"), 0xBB3D);
}
