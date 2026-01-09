//! Parallel checksum computation using combine().
//!
//! CRC checksums are mathematically combinable: given crc(A) and crc(B),
//! we can compute crc(A || B) without having both chunks in memory.
//! This enables efficient parallel processing of large data.
//!
//! Run with: `cargo run --example parallel -p checksum`

use std::thread;

use checksum::{Checksum, ChecksumCombine, Crc32, Crc64};

fn main() {
  println!("=== Parallel Checksum Examples ===\n");

  combine_basics();
  parallel_chunks();
  threaded_example();
}

/// Basic combine() demonstration.
fn combine_basics() {
  println!("--- Combine Basics ---\n");

  let data = b"hello world";
  let (part_a, part_b) = data.split_at(6); // "hello " and "world"

  // Compute checksums of each part independently
  let crc_a = Crc32::checksum(part_a);
  let crc_b = Crc32::checksum(part_b);

  println!("Part A (\"hello \"): 0x{crc_a:08X}");
  println!("Part B (\"world\"):  0x{crc_b:08X}");

  // Combine to get checksum of full data
  // combine(crc_a, crc_b, len_b) = crc(part_a || part_b)
  let combined = Crc32::combine(crc_a, crc_b, part_b.len());
  let expected = Crc32::checksum(data);

  println!("Combined:           0x{combined:08X}");
  println!("Full data checksum: 0x{expected:08X}");
  assert_eq!(combined, expected);
  println!("Match!\n");

  // Works with any number of parts - combine sequentially
  let parts: &[&[u8]] = &[b"one", b"two", b"three"];
  let full: Vec<u8> = parts.iter().flat_map(|p| p.iter().copied()).collect();

  let mut result = Crc64::checksum(parts[0]);
  for part in &parts[1..] {
    let part_crc = Crc64::checksum(part);
    result = Crc64::combine(result, part_crc, part.len());
  }

  println!("Multi-part combine: 0x{result:016X}");
  println!("Full data verify:   0x{:016X}", Crc64::checksum(&full));
  assert_eq!(result, Crc64::checksum(&full));
  println!();
}

/// Processing large data in parallel chunks.
fn parallel_chunks() {
  println!("--- Parallel Chunk Processing ---\n");

  // Simulate large data (in practice, this could be a memory-mapped file)
  let data: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();

  let chunk_size = 250_000; // 4 chunks of 250KB each

  // Sequential: reference result
  let sequential = Crc64::checksum(&data);
  println!("Sequential CRC-64: 0x{sequential:016X}");

  // Parallel: compute each chunk's CRC, then combine
  let chunks: Vec<_> = data.chunks(chunk_size).collect();
  let chunk_crcs: Vec<_> = chunks.iter().map(|c| Crc64::checksum(c)).collect();

  // Combine all chunk CRCs
  let mut parallel = chunk_crcs[0];
  for (crc, chunk) in chunk_crcs[1..].iter().zip(&chunks[1..]) {
    parallel = Crc64::combine(parallel, *crc, chunk.len());
  }

  println!("Parallel CRC-64:   0x{parallel:016X}");
  assert_eq!(sequential, parallel);
  println!("Match! (processed {} chunks)\n", chunks.len());
}

/// Multi-threaded checksum using std::thread.
fn threaded_example() {
  println!("--- Multi-Threaded Example ---\n");

  // Generate test data
  let data: Vec<u8> = (0..4_000_000).map(|i| ((i * 17) % 256) as u8).collect();

  let num_threads = 4;
  let chunk_size = data.len() / num_threads;

  // Sequential reference
  let sequential = Crc64::checksum(&data);
  println!("Sequential: 0x{sequential:016X}");

  // Split data into chunks with their indices
  let chunks: Vec<(usize, &[u8])> = data.chunks(chunk_size).enumerate().collect();

  // Spawn threads to compute each chunk's CRC
  let handles: Vec<_> = chunks
    .into_iter()
    .map(|(idx, chunk)| {
      let chunk = chunk.to_vec(); // Clone for thread ownership
      thread::spawn(move || {
        let crc = Crc64::checksum(&chunk);
        (idx, crc, chunk.len())
      })
    })
    .collect();

  // Collect results in order
  let mut results: Vec<(usize, u64, usize)> = handles
    .into_iter()
    .map(|h| h.join().expect("thread panicked"))
    .collect();
  results.sort_by_key(|(idx, _, _)| *idx);

  // Combine in order
  let mut combined = results[0].1;
  for (_, crc, len) in &results[1..] {
    combined = Crc64::combine(combined, *crc, *len);
  }

  println!("Threaded:   0x{combined:016X}");
  assert_eq!(sequential, combined);
  println!("Match! (used {} threads)\n", num_threads);
}
