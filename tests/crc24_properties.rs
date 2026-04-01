//! CRC-24 property tests: cross-library validation.
//!
//! These tests validate our CRC-24 implementation against:
//! 1. The `crc` crate as an external reference

// Proptest uses getcwd() which fails under Miri isolation.
#![cfg(not(miri))]
#![cfg(feature = "checksums")]

use crc::Crc as RefCrc;
use proptest::prelude::*;
use rscrypto::{Checksum, ChecksumCombine, Crc24OpenPgp};

const REF_CRC24_OPENPGP: RefCrc<u32> = RefCrc::<u32>::new(&crc::CRC_24_OPENPGP);

proptest! {
  #[test]
  fn crc24_openpgp_matches_crc_crate(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc24OpenPgp::checksum(&data);
    let reference = REF_CRC24_OPENPGP.checksum(&data) & 0x00FF_FFFF;
    prop_assert_eq!(ours, reference);
  }

  #[test]
  fn crc24_openpgp_streaming_matches_crc_crate(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    chunk in 1usize..=257
  ) {
    let mut ours = Crc24OpenPgp::new();
    let mut reference = REF_CRC24_OPENPGP.digest();

    for part in data.chunks(chunk) {
      ours.update(part);
      reference.update(part);
    }

    prop_assert_eq!(ours.finalize(), reference.finalize() & 0x00FF_FFFF);
  }

  #[test]
  fn crc24_openpgp_combine_matches_reference(
    data in proptest::collection::vec(any::<u8>(), 0..=4096),
    split in any::<usize>()
  ) {
    let split = split.strict_rem(data.len().strict_add(1));
    let (a, b) = data.split_at(split);

    let crc_a = Crc24OpenPgp::checksum(a);
    let crc_b = Crc24OpenPgp::checksum(b);
    let combined = Crc24OpenPgp::combine(crc_a, crc_b, b.len());

    let reference = REF_CRC24_OPENPGP.checksum(&data) & 0x00FF_FFFF;
    prop_assert_eq!(combined, reference);
  }
}

#[test]
fn test_vectors_crc24_openpgp() {
  assert_eq!(Crc24OpenPgp::checksum(b"123456789"), 0x0021_CF02);
  assert_eq!(Crc24OpenPgp::checksum(b""), 0x00B7_04CE);
}
