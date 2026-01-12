#![no_main]

use hashes::fast::{Xxh3_128, Xxh3_64};
use libfuzzer_sys::fuzz_target;
use traits::FastHash as _;

fuzz_target!(|input: &[u8]| {
  let (seed_bytes, data) = input.split_at(core::cmp::min(8, input.len()));
  let mut seed = 0u64;
  for (i, &b) in seed_bytes.iter().enumerate() {
    seed |= (b as u64) << (i * 8);
  }

  let ours64 = Xxh3_64::hash_with_seed(seed, data);
  let ref64 = xxhash_rust::xxh3::xxh3_64_with_seed(data, seed);
  assert_eq!(ours64, ref64);

  let ours128 = Xxh3_128::hash_with_seed(seed, data);
  let ref128 = xxhash_rust::xxh3::xxh3_128_with_seed(data, seed);
  assert_eq!(ours128, ref128);
});

