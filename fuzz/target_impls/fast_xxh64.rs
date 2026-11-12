use core::hash::Hasher;

use rscrypto::{FastHash, Xxh64, Xxh64Hasher};
use rscrypto_fuzz::{FuzzInput, some_or_return};

pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let seed_bytes: [u8; 8] = some_or_return!(input.bytes());
  let partitions: [u8; 8] = some_or_return!(input.bytes());
  let data = input.rest();
  let seed = u64::from_le_bytes(seed_bytes);

  let oracle = xxhash_rust::xxh64::xxh64(data, seed);
  assert_eq!(Xxh64::hash_with_seed(seed, data), oracle, "xxh64 oracle mismatch");

  // Arbitrary incremental partitions must equal the one-shot oracle. The
  // second pass starts from a reset of the first pass's final state.
  let mut streamed = Xxh64Hasher::with_seed(seed);
  for first_partition in [0, 1] {
    streamed.reset();
    let mut offset = 0usize;
    let mut partition = first_partition;
    while offset < data.len() {
      let requested = usize::from(partitions[partition & 7]).strict_add(1);
      let end = offset.strict_add(requested).min(data.len());
      streamed.write(&data[offset..end]);
      streamed.write(&[]);
      offset = end;
      partition = partition.strict_add(1);
    }
    assert_eq!(streamed.finish(), oracle, "xxh64 streaming mismatch");
  }
}
