//! Fuzz target for CRC combine operations.
//!
//! Tests combine associativity and correctness with multiple splits.

#![no_main]

use arbitrary::Arbitrary;
use checksum::{Checksum, ChecksumCombine, Crc16Ccitt, Crc16Ibm, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
  data: Vec<u8>,
  splits: Vec<usize>,
}

fuzz_target!(|input: Input| {
  let data = &input.data;
  if data.is_empty() {
    return;
  }

  // Normalize splits to valid range and sort
  let max_split = data.len().strict_add(1);
  let mut splits: Vec<usize> = input
    .splits
    .iter()
    .map(|s| (*s).strict_rem(max_split))
    .collect();
  splits.sort();
  splits.dedup();

  // Test CRC64 combine chain
  test_combine_chain_crc64(data, &splits);

  // Test CRC64/NVME combine chain
  test_combine_chain_crc64_nvme(data, &splits);

  // Test CRC32/IEEE combine chain
  test_combine_chain_crc32(data, &splits);

  // Test CRC32C combine chain
  test_combine_chain_crc32c(data, &splits);

  // Test CRC16/CCITT combine chain
  test_combine_chain_crc16_ccitt(data, &splits);

  // Test CRC16/IBM combine chain
  test_combine_chain_crc16_ibm(data, &splits);

  // Test CRC24/OPENPGP combine chain
  test_combine_chain_crc24_openpgp(data, &splits);
});

fn test_combine_chain_crc64(data: &[u8], splits: &[usize]) {
  let expected = Crc64::checksum(data);

  let mut chunks = Vec::new();
  let mut prev = 0;
  for &split in splits {
    if split > prev && split <= data.len() {
      chunks.push(&data[prev..split]);
      prev = split;
    }
  }
  if prev < data.len() {
    chunks.push(&data[prev..]);
  }

  if chunks.is_empty() {
    return;
  }

  let mut combined_crc = Crc64::checksum(chunks[0]);
  for chunk in &chunks[1..] {
    let chunk_crc = Crc64::checksum(chunk);
    combined_crc = Crc64::combine(combined_crc, chunk_crc, chunk.len());
  }

  assert_eq!(combined_crc, expected, "crc64 combine chain mismatch");
}

fn test_combine_chain_crc64_nvme(data: &[u8], splits: &[usize]) {
  let expected = Crc64Nvme::checksum(data);

  let mut chunks = Vec::new();
  let mut prev = 0;
  for &split in splits {
    if split > prev && split <= data.len() {
      chunks.push(&data[prev..split]);
      prev = split;
    }
  }
  if prev < data.len() {
    chunks.push(&data[prev..]);
  }

  if chunks.is_empty() {
    return;
  }

  let mut combined_crc = Crc64Nvme::checksum(chunks[0]);
  for chunk in &chunks[1..] {
    let chunk_crc = Crc64Nvme::checksum(chunk);
    combined_crc = Crc64Nvme::combine(combined_crc, chunk_crc, chunk.len());
  }

  assert_eq!(combined_crc, expected, "crc64/nvme combine chain mismatch");
}

fn test_combine_chain_crc32(data: &[u8], splits: &[usize]) {
  let expected = Crc32::checksum(data);

  let mut chunks = Vec::new();
  let mut prev = 0;
  for &split in splits {
    if split > prev && split <= data.len() {
      chunks.push(&data[prev..split]);
      prev = split;
    }
  }
  if prev < data.len() {
    chunks.push(&data[prev..]);
  }

  if chunks.is_empty() {
    return;
  }

  let mut combined_crc = Crc32::checksum(chunks[0]);
  for chunk in &chunks[1..] {
    let chunk_crc = Crc32::checksum(chunk);
    combined_crc = Crc32::combine(combined_crc, chunk_crc, chunk.len());
  }

  assert_eq!(combined_crc, expected, "crc32 combine chain mismatch");
}

fn test_combine_chain_crc32c(data: &[u8], splits: &[usize]) {
  let expected = Crc32C::checksum(data);

  let mut chunks = Vec::new();
  let mut prev = 0;
  for &split in splits {
    if split > prev && split <= data.len() {
      chunks.push(&data[prev..split]);
      prev = split;
    }
  }
  if prev < data.len() {
    chunks.push(&data[prev..]);
  }

  if chunks.is_empty() {
    return;
  }

  let mut combined_crc = Crc32C::checksum(chunks[0]);
  for chunk in &chunks[1..] {
    let chunk_crc = Crc32C::checksum(chunk);
    combined_crc = Crc32C::combine(combined_crc, chunk_crc, chunk.len());
  }

  assert_eq!(combined_crc, expected, "crc32c combine chain mismatch");
}

fn test_combine_chain_crc16_ccitt(data: &[u8], splits: &[usize]) {
  let expected = Crc16Ccitt::checksum(data);

  let mut chunks = Vec::new();
  let mut prev = 0;
  for &split in splits {
    if split > prev && split <= data.len() {
      chunks.push(&data[prev..split]);
      prev = split;
    }
  }
  if prev < data.len() {
    chunks.push(&data[prev..]);
  }

  if chunks.is_empty() {
    return;
  }

  let mut combined_crc = Crc16Ccitt::checksum(chunks[0]);
  for chunk in &chunks[1..] {
    let chunk_crc = Crc16Ccitt::checksum(chunk);
    combined_crc = Crc16Ccitt::combine(combined_crc, chunk_crc, chunk.len());
  }

  assert_eq!(combined_crc, expected, "crc16/ccitt combine chain mismatch");
}

fn test_combine_chain_crc16_ibm(data: &[u8], splits: &[usize]) {
  let expected = Crc16Ibm::checksum(data);

  let mut chunks = Vec::new();
  let mut prev = 0;
  for &split in splits {
    if split > prev && split <= data.len() {
      chunks.push(&data[prev..split]);
      prev = split;
    }
  }
  if prev < data.len() {
    chunks.push(&data[prev..]);
  }

  if chunks.is_empty() {
    return;
  }

  let mut combined_crc = Crc16Ibm::checksum(chunks[0]);
  for chunk in &chunks[1..] {
    let chunk_crc = Crc16Ibm::checksum(chunk);
    combined_crc = Crc16Ibm::combine(combined_crc, chunk_crc, chunk.len());
  }

  assert_eq!(combined_crc, expected, "crc16/ibm combine chain mismatch");
}

fn test_combine_chain_crc24_openpgp(data: &[u8], splits: &[usize]) {
  let expected = Crc24OpenPgp::checksum(data);

  let mut chunks = Vec::new();
  let mut prev = 0;
  for &split in splits {
    if split > prev && split <= data.len() {
      chunks.push(&data[prev..split]);
      prev = split;
    }
  }
  if prev < data.len() {
    chunks.push(&data[prev..]);
  }

  if chunks.is_empty() {
    return;
  }

  let mut combined_crc = Crc24OpenPgp::checksum(chunks[0]);
  for chunk in &chunks[1..] {
    let chunk_crc = Crc24OpenPgp::checksum(chunk);
    combined_crc = Crc24OpenPgp::combine(combined_crc, chunk_crc, chunk.len());
  }

  assert_eq!(combined_crc, expected, "crc24/openpgp combine chain mismatch");
}
