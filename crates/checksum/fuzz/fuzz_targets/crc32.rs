//! Fuzz target for CRC32 implementations.
//!
//! Tests that:
//! - Incremental updates produce the same result as one-shot
//! - Resume produces correct results
//! - Combine produces correct results

#![no_main]

use arbitrary::Arbitrary;
use checksum::{Checksum, ChecksumCombine, Crc32, Crc32C};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
  data: Vec<u8>,
  split_point: usize,
}

fuzz_target!(|input: Input| {
  let data = &input.data;
  let split = input
    .split_point
    .strict_rem(data.len().strict_add(1));

  test_crc32_ieee(data, split);
  test_crc32c(data, split);
});

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
