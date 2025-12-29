//! Fuzz target for CRC32 implementations.
//!
//! Tests that:
//! - Incremental updates produce the same result as one-shot
//! - Resume produces correct results
//! - Combine produces correct results

#![no_main]

use arbitrary::Arbitrary;
use checksum::{Checksum, ChecksumCombine, Crc16Ccitt, Crc16Ibm, Crc24OpenPgp, Crc32, Crc32C};
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
  test_crc16_ccitt(data, split);
  test_crc16_ibm(data, split);
  test_crc24_openpgp(data, split);
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

fn test_crc16_ccitt(data: &[u8], split: usize) {
  let oneshot = Crc16Ccitt::checksum(data);

  let (a, b) = data.split_at(split);
  let mut hasher = Crc16Ccitt::new();
  hasher.update(a);
  hasher.update(b);
  let incremental = hasher.finalize();

  assert_eq!(oneshot, incremental, "crc16/ccitt incremental mismatch");

  let crc_a = Crc16Ccitt::checksum(a);
  let mut resumed = Crc16Ccitt::resume(crc_a);
  resumed.update(b);
  let resume_result = resumed.finalize();

  assert_eq!(oneshot, resume_result, "crc16/ccitt resume mismatch");

  let crc_b = Crc16Ccitt::checksum(b);
  let combined = Crc16Ccitt::combine(crc_a, crc_b, b.len());

  assert_eq!(oneshot, combined, "crc16/ccitt combine mismatch");
}

fn test_crc16_ibm(data: &[u8], split: usize) {
  let oneshot = Crc16Ibm::checksum(data);

  let (a, b) = data.split_at(split);
  let mut hasher = Crc16Ibm::new();
  hasher.update(a);
  hasher.update(b);
  let incremental = hasher.finalize();

  assert_eq!(oneshot, incremental, "crc16/ibm incremental mismatch");

  let crc_a = Crc16Ibm::checksum(a);
  let mut resumed = Crc16Ibm::resume(crc_a);
  resumed.update(b);
  let resume_result = resumed.finalize();

  assert_eq!(oneshot, resume_result, "crc16/ibm resume mismatch");

  let crc_b = Crc16Ibm::checksum(b);
  let combined = Crc16Ibm::combine(crc_a, crc_b, b.len());

  assert_eq!(oneshot, combined, "crc16/ibm combine mismatch");
}

fn test_crc24_openpgp(data: &[u8], split: usize) {
  let oneshot = Crc24OpenPgp::checksum(data);

  let (a, b) = data.split_at(split);
  let mut hasher = Crc24OpenPgp::new();
  hasher.update(a);
  hasher.update(b);
  let incremental = hasher.finalize();

  assert_eq!(oneshot, incremental, "crc24/openpgp incremental mismatch");

  let crc_a = Crc24OpenPgp::checksum(a);
  let mut resumed = Crc24OpenPgp::resume(crc_a);
  resumed.update(b);
  let resume_result = resumed.finalize();

  assert_eq!(oneshot, resume_result, "crc24/openpgp resume mismatch");

  let crc_b = Crc24OpenPgp::checksum(b);
  let combined = Crc24OpenPgp::combine(crc_a, crc_b, b.len());

  assert_eq!(oneshot, combined, "crc24/openpgp combine mismatch");
}
