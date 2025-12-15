//! Fuzz target for CRC16 implementations (IBM and CCITT-FALSE).
//!
//! Tests that:
//! - No panics on arbitrary input
//! - Incremental updates produce same result as one-shot
//! - Resume produces correct results

#![no_main]

use arbitrary::Arbitrary;
use checksum::{Crc16CcittFalse, Crc16Ibm};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
  data: Vec<u8>,
  split_point: usize,
}

fuzz_target!(|input: Input| {
  let data = &input.data;
  let split = input.split_point % (data.len() + 1);

  // Test CRC16/IBM
  test_crc16_ibm(data, split);

  // Test CRC16/CCITT-FALSE
  test_crc16_ccitt_false(data, split);
});

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

fn test_crc16_ccitt_false(data: &[u8], split: usize) {
  let oneshot = Crc16CcittFalse::checksum(data);

  let (a, b) = data.split_at(split);
  let mut hasher = Crc16CcittFalse::new();
  hasher.update(a);
  hasher.update(b);
  let incremental = hasher.finalize();

  assert_eq!(oneshot, incremental, "crc16/ccitt-false incremental mismatch");

  let crc_a = Crc16CcittFalse::checksum(a);
  let mut resumed = Crc16CcittFalse::resume(crc_a);
  resumed.update(b);
  let resume_result = resumed.finalize();

  assert_eq!(oneshot, resume_result, "crc16/ccitt-false resume mismatch");

  let crc_b = Crc16CcittFalse::checksum(b);
  let combined = Crc16CcittFalse::combine(crc_a, crc_b, b.len());

  assert_eq!(oneshot, combined, "crc16/ccitt-false combine mismatch");
}
