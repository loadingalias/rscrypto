//! CRC-16 property tests: cross-library validation.
//!
//! These tests validate our CRC-16 implementations against:
//! 1. The `crc-fast` crate as an external reference

// Proptest uses getcwd() which fails under Miri isolation.
#![cfg(not(miri))]
#![cfg(feature = "checksums")]

use crc_fast::CrcAlgorithm;
use proptest::prelude::*;
use rscrypto::{Checksum, ChecksumCombine, Crc16Ccitt, Crc16Ibm};

fn reference_u16(value: u64) -> u16 {
  u16::try_from(value).expect("CRC-16 reference output must fit in 16 bits")
}

proptest! {
  // Cross-validation against crc-fast-rust

  #[test]
  fn crc16_ccitt_matches_crc_fast_rust(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc16Ccitt::checksum(&data);
    let reference = reference_u16(crc_fast::checksum(CrcAlgorithm::Crc16IbmSdlc, &data));
    prop_assert_eq!(ours, reference);
  }

  #[test]
  fn crc16_ibm_matches_crc_fast_rust(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc16Ibm::checksum(&data);
    let reference = reference_u16(crc_fast::checksum(CrcAlgorithm::Crc16Arc, &data));
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

    prop_assert_eq!(ours.finalize(), reference_u16(reference.finalize()));
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

    prop_assert_eq!(ours.finalize(), reference_u16(reference.finalize()));
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
    let ref_combined = reference_u16(crc_fast::checksum_combine(
      CrcAlgorithm::Crc16IbmSdlc,
      ref_crc_a,
      ref_crc_b,
      u64::try_from(b.len()).expect("test input length must fit in u64"),
    ));

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
    let ref_combined = reference_u16(crc_fast::checksum_combine(
      CrcAlgorithm::Crc16Arc,
      ref_crc_a,
      ref_crc_b,
      u64::try_from(b.len()).expect("test input length must fit in u64"),
    ));

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
