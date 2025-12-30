// Proptest uses getcwd() which fails under Miri isolation.
#![cfg(not(miri))]

extern crate std;

use crc::Crc as RefCrc;
use proptest::prelude::*;

use super::*;

const REF_CRC24_OPENPGP: RefCrc<u32> = RefCrc::<u32>::new(&crc::CRC_24_OPENPGP);

fn crc24_openpgp_reference(data: &[u8]) -> u32 {
  // MSB-first OpenPGP using a 32-bit expanded register (top 24 bits).
  let poly_aligned = crate::common::tables::CRC24_OPENPGP_POLY << 8;
  let mut state: u32 = 0x00B7_04CE << 8;
  for &byte in data {
    state ^= (byte as u32) << 24;
    for _ in 0..8 {
      if state & 0x8000_0000 != 0 {
        state = (state << 1) ^ poly_aligned;
      } else {
        state <<= 1;
      }
    }
  }
  (state >> 8) & 0x00FF_FFFF
}

proptest! {
  #[test]
  fn crc24_openpgp_matches_portable(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc24OpenPgp::checksum(&data);
    let portable = portable::crc24_openpgp_slice8(0x00B7_04CE, &data) & 0x00FF_FFFF;
    prop_assert_eq!(ours, portable);
  }

  #[test]
  fn crc24_openpgp_matches_reference(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc24OpenPgp::checksum(&data);
    let reference = crc24_openpgp_reference(&data);
    prop_assert_eq!(ours, reference);
  }

  #[test]
  fn crc24_openpgp_matches_crc_crate(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc24OpenPgp::checksum(&data);
    let reference = REF_CRC24_OPENPGP.checksum(&data) & 0x00FF_FFFF;
    prop_assert_eq!(ours, reference);
  }

  // NOTE: crc24_openpgp_streaming_and_combine test removed - now covered by unified
  // tests in common/proptests.rs (combine_correctness, chunking_equivalence, resume_correctness)

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

    let reference = crc24_openpgp_reference(&data);
    prop_assert_eq!(combined, reference);
  }
}

#[test]
fn test_vectors_crc24_openpgp() {
  assert_eq!(Crc24OpenPgp::checksum(b"123456789"), 0x0021_CF02);
  assert_eq!(Crc24OpenPgp::checksum(b""), 0x00B7_04CE);
}
