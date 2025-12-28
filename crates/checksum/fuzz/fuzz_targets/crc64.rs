//! Fuzz target for checksum implementations.
//!
//! Tests that:
//! - No panics on arbitrary input
//! - Incremental updates produce same result as one-shot
//! - Resume produces correct results

#![no_main]

use arbitrary::Arbitrary;
use checksum::{Checksum, ChecksumCombine, Crc32, Crc32C, Crc64, Crc64Nvme};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
  data: Vec<u8>,
  split_point: usize,
}

fuzz_target!(|input: Input| {
  let data = &input.data;
  let split = input.split_point % (data.len() + 1);

  // Test CRC64/XZ
  test_crc64_xz(data, split);

  // Test CRC64/NVME
  test_crc64_nvme(data, split);

  // Test CRC32/IEEE
  test_crc32_ieee(data, split);

  // Test CRC32C
  test_crc32c(data, split);
});

fn test_crc64_xz(data: &[u8], split: usize) {
  let oneshot = Crc64::checksum(data);

  let (a, b) = data.split_at(split);
  let mut hasher = Crc64::new();
  hasher.update(a);
  hasher.update(b);
  let incremental = hasher.finalize();

  assert_eq!(oneshot, incremental, "crc64/xz incremental mismatch");

  let crc_a = Crc64::checksum(a);
  let mut resumed = Crc64::resume(crc_a);
  resumed.update(b);
  let resume_result = resumed.finalize();

  assert_eq!(oneshot, resume_result, "crc64/xz resume mismatch");

  let crc_b = Crc64::checksum(b);
  let combined = Crc64::combine(crc_a, crc_b, b.len());

  assert_eq!(oneshot, combined, "crc64/xz combine mismatch");
}

fn test_crc64_nvme(data: &[u8], split: usize) {
  let oneshot = Crc64Nvme::checksum(data);

  let (a, b) = data.split_at(split);
  let mut hasher = Crc64Nvme::new();
  hasher.update(a);
  hasher.update(b);
  let incremental = hasher.finalize();

  assert_eq!(oneshot, incremental, "crc64/nvme incremental mismatch");

  let crc_a = Crc64Nvme::checksum(a);
  let mut resumed = Crc64Nvme::resume(crc_a);
  resumed.update(b);
  let resume_result = resumed.finalize();

  assert_eq!(oneshot, resume_result, "crc64/nvme resume mismatch");

  let crc_b = Crc64Nvme::checksum(b);
  let combined = Crc64Nvme::combine(crc_a, crc_b, b.len());

  assert_eq!(oneshot, combined, "crc64/nvme combine mismatch");
}

fn test_crc32_ieee(data: &[u8], split: usize) {
  let oneshot = Crc32::checksum(data);

  let (a, b) = data.split_at(split);
  let mut hasher = Crc32::new();
  hasher.update(a);
  hasher.update(b);
  let incremental = hasher.finalize();

  assert_eq!(oneshot, incremental, "crc32 incremental mismatch");

  let crc_a = Crc32::checksum(a);
  let mut resumed = Crc32::resume(crc_a);
  resumed.update(b);
  let resume_result = resumed.finalize();

  assert_eq!(oneshot, resume_result, "crc32 resume mismatch");

  let crc_b = Crc32::checksum(b);
  let combined = Crc32::combine(crc_a, crc_b, b.len());

  assert_eq!(oneshot, combined, "crc32 combine mismatch");
}

fn test_crc32c(data: &[u8], split: usize) {
  let oneshot = Crc32C::checksum(data);

  let (a, b) = data.split_at(split);
  let mut hasher = Crc32C::new();
  hasher.update(a);
  hasher.update(b);
  let incremental = hasher.finalize();

  assert_eq!(oneshot, incremental, "crc32c incremental mismatch");

  let crc_a = Crc32C::checksum(a);
  let mut resumed = Crc32C::resume(crc_a);
  resumed.update(b);
  let resume_result = resumed.finalize();

  assert_eq!(oneshot, resume_result, "crc32c resume mismatch");

  let crc_b = Crc32C::checksum(b);
  let combined = Crc32C::combine(crc_a, crc_b, b.len());

  assert_eq!(oneshot, combined, "crc32c combine mismatch");
}
