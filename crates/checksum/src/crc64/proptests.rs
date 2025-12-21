extern crate std;

use crc64fast as ref_crc64fast;
use crc64fast_nvme as ref_crc64fast_nvme;
use proptest::prelude::*;

use super::*;

proptest! {
  #[test]
  fn crc64_xz_matches_portable(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc64::checksum(&data);
    let portable = portable::crc64_slice8_xz(!0, &data) ^ !0;
    prop_assert_eq!(ours, portable);
  }

  #[test]
  fn crc64_nvme_matches_portable(data in proptest::collection::vec(any::<u8>(), 0..=4096)) {
    let ours = Crc64Nvme::checksum(&data);
    let portable = portable::crc64_slice8_nvme(!0, &data) ^ !0;
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

  #[test]
  fn crc64_xz_streaming_and_combine(data in proptest::collection::vec(any::<u8>(), 0..=4096), split in any::<usize>(), chunk in 1usize..=257) {
    let split = split % (data.len() + 1);
    let (a, b) = data.split_at(split);

    let oneshot = Crc64::checksum(&data);

    let mut hasher = Crc64::new();
    for part in a.chunks(chunk) {
      hasher.update(part);
    }
    for part in b.chunks(chunk) {
      hasher.update(part);
    }
    prop_assert_eq!(hasher.finalize(), oneshot);

    let crc_a = Crc64::checksum(a);
    let crc_b = Crc64::checksum(b);
    let combined = Crc64::combine(crc_a, crc_b, b.len());
    prop_assert_eq!(combined, oneshot);

    let mut resumed = Crc64::resume(crc_a);
    resumed.update(b);
    prop_assert_eq!(resumed.finalize(), oneshot);
  }

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

  #[test]
  fn crc64_nvme_streaming_and_combine(data in proptest::collection::vec(any::<u8>(), 0..=4096), split in any::<usize>(), chunk in 1usize..=257) {
    let split = split % (data.len() + 1);
    let (a, b) = data.split_at(split);

    let oneshot = Crc64Nvme::checksum(&data);

    let mut hasher = Crc64Nvme::new();
    for part in a.chunks(chunk) {
      hasher.update(part);
    }
    for part in b.chunks(chunk) {
      hasher.update(part);
    }
    prop_assert_eq!(hasher.finalize(), oneshot);

    let crc_a = Crc64Nvme::checksum(a);
    let crc_b = Crc64Nvme::checksum(b);
    let combined = Crc64Nvme::combine(crc_a, crc_b, b.len());
    prop_assert_eq!(combined, oneshot);

    let mut resumed = Crc64Nvme::resume(crc_a);
    resumed.update(b);
    prop_assert_eq!(resumed.finalize(), oneshot);
  }

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
}
