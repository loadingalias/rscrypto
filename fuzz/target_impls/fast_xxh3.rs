use core::hash::Hasher;

use rscrypto::{FastHash, Xxh3, Xxh3_128, Xxh3_128Hasher, Xxh3Hasher};
use rscrypto_fuzz::{FuzzInput, some_or_return};

pub fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let seed_bytes: [u8; 8] = some_or_return!(input.bytes());
  let partitions: [u8; 8] = some_or_return!(input.bytes());
  let data = input.rest();
  let seed = u64::from_le_bytes(seed_bytes);

  // Differential: rscrypto ↔ xxhash-rust crate (64-bit)
  {
    let ours = Xxh3::hash_with_seed(seed, data);
    let oracle = xxhash_rust::xxh3::xxh3_64_with_seed(data, seed);
    assert_eq!(ours, oracle, "xxh3-64 oracle mismatch");
  }

  // Arbitrary incremental partitions must equal the one-shot XXH3-64 oracle.
  {
    let mut streamed = Xxh3Hasher::with_seed(seed);
    let mut offset = 0usize;
    let mut partition = 0usize;
    while offset < data.len() {
      let requested = usize::from(partitions[partition & 7]).strict_add(1);
      let end = offset.strict_add(requested).min(data.len());
      streamed.write(&data[offset..end]);
      streamed.write(&[]);
      offset = end;
      partition = partition.strict_add(1);
    }
    assert_eq!(streamed.finish(), xxhash_rust::xxh3::xxh3_64_with_seed(data, seed));

    let mut streamed128 = Xxh3_128Hasher::with_seed(seed);
    let mut offset = 0usize;
    let mut partition = 0usize;
    while offset < data.len() {
      let requested = usize::from(partitions[partition & 7]).strict_add(1);
      let end = offset.strict_add(requested).min(data.len());
      streamed128.write(&data[offset..end]);
      streamed128.write(&[]);
      offset = end;
      partition = partition.strict_add(1);
    }
    assert_eq!(streamed128.finish(), xxhash_rust::xxh3::xxh3_128_with_seed(data, seed));
  }

  // Differential: rscrypto ↔ xxhash-rust crate (128-bit)
  {
    let ours = Xxh3_128::hash_with_seed(seed, data);
    let oracle = xxhash_rust::xxh3::xxh3_128_with_seed(data, seed);
    assert_eq!(ours, oracle, "xxh3-128 oracle mismatch");
  }

  // Default-seed (0) consistency
  {
    let default_64 = Xxh3::hash(data);
    let seeded_64 = Xxh3::hash_with_seed(0, data);
    assert_eq!(default_64, seeded_64, "xxh3-64: default vs seed=0 mismatch");

    let default_128 = Xxh3_128::hash(data);
    let seeded_128 = Xxh3_128::hash_with_seed(0, data);
    assert_eq!(default_128, seeded_128, "xxh3-128: default vs seed=0 mismatch");
  }
}
