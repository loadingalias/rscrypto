use core::hash::Hasher;

use rscrypto::{FastHash, RapidHash, RapidHash128, RapidHashFast64, RapidHashFast128, RapidStreamHasher};
use rscrypto_fuzz::{FuzzInput, some_or_return};

pub fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let seed_bytes: [u8; 8] = some_or_return!(input.bytes());
  let partitions: [u8; 8] = some_or_return!(input.bytes());
  let data = input.rest();
  let seed = u64::from_le_bytes(seed_bytes);

  let secrets = rapidhash::v3::RapidSecrets::seed_cpp(seed);
  let oracle = rapidhash::v3::rapidhash_v3_seeded(data, &secrets);
  assert_eq!(RapidHash::hash_with_seed(seed, data), oracle, "rapidhash-v3 oracle mismatch");

  let mut streamed = RapidStreamHasher::with_seed(seed);
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
  assert_eq!(streamed.finish(), oracle, "rapidhash-v3 streaming mismatch");

  // Property: default seed = seed(0)
  {
    let default_64 = RapidHash::hash(data);
    let seeded_64 = RapidHash::hash_with_seed(0, data);
    assert_eq!(default_64, seeded_64, "rapidhash-64: default vs seed=0");

    let default_128 = RapidHash128::hash(data);
    let seeded_128 = RapidHash128::hash_with_seed(0, data);
    assert_eq!(default_128, seeded_128, "rapidhash-128: default vs seed=0");
  }

  // Property: 128-bit hash embeds 64-bit hash
  // (The high/low 64 bits of rapidhash-128 should relate to rapidhash-64.)
  // Just exercise all variants with the fuzzed seed to shake out panics/UB.
  let _h64 = RapidHash::hash_with_seed(seed, data);
  let _h128 = RapidHash128::hash_with_seed(seed, data);
  let _f64 = RapidHashFast64::hash_with_seed(seed, data);
  let _f128 = RapidHashFast128::hash_with_seed(seed, data);
}
