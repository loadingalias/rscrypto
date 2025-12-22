//! Fuzz target for CRC32 (ISO-HDLC) implementation.
//!
//! Tests that:
//! - No panics on arbitrary input
//! - Incremental updates produce same result as one-shot
//! - Resume produces correct results

#![no_main]

use arbitrary::Arbitrary;
use checksum::{Checksum, ChecksumCombine, Crc32};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
  data: Vec<u8>,
  split_point: usize,
}

fuzz_target!(|input: Input| {
  let data = &input.data;
  let split = input.split_point % (data.len() + 1);

  // One-shot computation
  let oneshot = Crc32::checksum(data);

  // Incremental computation
  let (a, b) = data.split_at(split);
  let mut hasher = Crc32::new();
  hasher.update(a);
  hasher.update(b);
  let incremental = hasher.finalize();

  assert_eq!(oneshot, incremental, "incremental mismatch");

  // Resume computation
  let crc_a = Crc32::checksum(a);
  let mut resumed = Crc32::resume(crc_a);
  resumed.update(b);
  let resume_result = resumed.finalize();

  assert_eq!(oneshot, resume_result, "resume mismatch");

  // Combine computation
  let crc_b = Crc32::checksum(b);
  let combined = Crc32::combine(crc_a, crc_b, b.len());

  assert_eq!(oneshot, combined, "combine mismatch");
});
