//! Fuzz target for streaming CRC API.
//!
//! Tests that arbitrary sequences of update calls produce correct results.

#![no_main]

use arbitrary::Arbitrary;
use checksum::{Checksum, Crc16Ccitt, Crc16Ibm, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme};
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
  test_streaming_crc32(data, &input.chunk_sizes);
  test_streaming_crc32c(data, &input.chunk_sizes);
  test_streaming_crc16_ccitt(data, &input.chunk_sizes);
  test_streaming_crc16_ibm(data, &input.chunk_sizes);
  test_streaming_crc24_openpgp(data, &input.chunk_sizes);
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
      let idx = chunk_idx.strict_rem(chunk_sizes.len());
      chunk_sizes[idx].strict_rem(256).max(1)
    };

    let end = offset.strict_add(chunk_size).min(data.len());
    hasher.update(&data[offset..end]);
    offset = end;
    chunk_idx = chunk_idx.strict_add(1);
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
      let idx = chunk_idx.strict_rem(chunk_sizes.len());
      chunk_sizes[idx].strict_rem(256).max(1)
    };

    let end = offset.strict_add(chunk_size).min(data.len());
    hasher.update(&data[offset..end]);
    offset = end;
    chunk_idx = chunk_idx.strict_add(1);
  }

  assert_eq!(hasher.finalize(), expected, "crc64/nvme streaming mismatch");
}

fn test_streaming_crc32(data: &[u8], chunk_sizes: &[usize]) {
  let expected = Crc32::checksum(data);

  let mut hasher = Crc32::new();
  let mut offset = 0;
  let mut chunk_idx = 0;

  while offset < data.len() {
    let chunk_size = if chunk_sizes.is_empty() {
      1
    } else {
      let idx = chunk_idx.strict_rem(chunk_sizes.len());
      chunk_sizes[idx].strict_rem(256).max(1)
    };

    let end = offset.strict_add(chunk_size).min(data.len());
    hasher.update(&data[offset..end]);
    offset = end;
    chunk_idx = chunk_idx.strict_add(1);
  }

  assert_eq!(hasher.finalize(), expected, "crc32 streaming mismatch");
}

fn test_streaming_crc32c(data: &[u8], chunk_sizes: &[usize]) {
  let expected = Crc32C::checksum(data);

  let mut hasher = Crc32C::new();
  let mut offset = 0;
  let mut chunk_idx = 0;

  while offset < data.len() {
    let chunk_size = if chunk_sizes.is_empty() {
      1
    } else {
      let idx = chunk_idx.strict_rem(chunk_sizes.len());
      chunk_sizes[idx].strict_rem(256).max(1)
    };

    let end = offset.strict_add(chunk_size).min(data.len());
    hasher.update(&data[offset..end]);
    offset = end;
    chunk_idx = chunk_idx.strict_add(1);
  }

  assert_eq!(hasher.finalize(), expected, "crc32c streaming mismatch");
}

fn test_streaming_crc16_ccitt(data: &[u8], chunk_sizes: &[usize]) {
  let expected = Crc16Ccitt::checksum(data);

  let mut hasher = Crc16Ccitt::new();
  let mut offset = 0;
  let mut chunk_idx = 0;

  while offset < data.len() {
    let chunk_size = if chunk_sizes.is_empty() {
      1
    } else {
      let idx = chunk_idx.strict_rem(chunk_sizes.len());
      chunk_sizes[idx].strict_rem(256).max(1)
    };

    let end = offset.strict_add(chunk_size).min(data.len());
    hasher.update(&data[offset..end]);
    offset = end;
    chunk_idx = chunk_idx.strict_add(1);
  }

  assert_eq!(hasher.finalize(), expected, "crc16/ccitt streaming mismatch");
}

fn test_streaming_crc16_ibm(data: &[u8], chunk_sizes: &[usize]) {
  let expected = Crc16Ibm::checksum(data);

  let mut hasher = Crc16Ibm::new();
  let mut offset = 0;
  let mut chunk_idx = 0;

  while offset < data.len() {
    let chunk_size = if chunk_sizes.is_empty() {
      1
    } else {
      let idx = chunk_idx.strict_rem(chunk_sizes.len());
      chunk_sizes[idx].strict_rem(256).max(1)
    };

    let end = offset.strict_add(chunk_size).min(data.len());
    hasher.update(&data[offset..end]);
    offset = end;
    chunk_idx = chunk_idx.strict_add(1);
  }

  assert_eq!(hasher.finalize(), expected, "crc16/ibm streaming mismatch");
}

fn test_streaming_crc24_openpgp(data: &[u8], chunk_sizes: &[usize]) {
  let expected = Crc24OpenPgp::checksum(data);

  let mut hasher = Crc24OpenPgp::new();
  let mut offset = 0;
  let mut chunk_idx = 0;

  while offset < data.len() {
    let chunk_size = if chunk_sizes.is_empty() {
      1
    } else {
      let idx = chunk_idx.strict_rem(chunk_sizes.len());
      chunk_sizes[idx].strict_rem(256).max(1)
    };

    let end = offset.strict_add(chunk_size).min(data.len());
    hasher.update(&data[offset..end]);
    offset = end;
    chunk_idx = chunk_idx.strict_add(1);
  }

  assert_eq!(hasher.finalize(), expected, "crc24/openpgp streaming mismatch");
}
