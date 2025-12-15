//! Fuzz target for CRC64 implementations (XZ and NVME).
//!
//! Tests that:
//! - No panics on arbitrary input
//! - Incremental updates produce same result as one-shot
//! - Resume produces correct results

#![no_main]

use arbitrary::Arbitrary;
use checksum::{Crc64, Crc64Nvme};
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
