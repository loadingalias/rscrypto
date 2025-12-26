//! Fuzz target for streaming CRC API.
//!
//! Tests that arbitrary sequences of update calls produce correct results.

#![no_main]

use arbitrary::Arbitrary;
use checksum::{Checksum, Crc64, Crc64Nvme};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
  data: Vec<u8>,
  /// Chunk sizes for streaming updates
  chunk_sizes: Vec<usize>,
}

fuzz_target!(|input: Input| {
  let data = &input.data;

  // Test with arbitrary chunk sizes
  test_streaming_crc64(data, &input.chunk_sizes);
  test_streaming_crc64_nvme(data, &input.chunk_sizes);
});

fn test_streaming_crc64(data: &[u8], chunk_sizes: &[usize]) {
  let expected = Crc64::checksum(data);

  let mut hasher = Crc64::new();
  let mut offset = 0;
  let mut chunk_idx = 0;

  while offset < data.len() {
    let chunk_size = if chunk_sizes.is_empty() {
      1
    } else {
      (chunk_sizes[chunk_idx % chunk_sizes.len()] % 256).max(1)
    };

    let end = (offset + chunk_size).min(data.len());
    hasher.update(&data[offset..end]);
    offset = end;
    chunk_idx += 1;
  }

  assert_eq!(hasher.finalize(), expected, "crc64 streaming mismatch");
}

fn test_streaming_crc64_nvme(data: &[u8], chunk_sizes: &[usize]) {
  let expected = Crc64Nvme::checksum(data);

  let mut hasher = Crc64Nvme::new();
  let mut offset = 0;
  let mut chunk_idx = 0;

  while offset < data.len() {
    let chunk_size = if chunk_sizes.is_empty() {
      1
    } else {
      (chunk_sizes[chunk_idx % chunk_sizes.len()] % 256).max(1)
    };

    let end = (offset + chunk_size).min(data.len());
    hasher.update(&data[offset..end]);
    offset = end;
    chunk_idx += 1;
  }

  assert_eq!(hasher.finalize(), expected, "crc64/nvme streaming mismatch");
}
