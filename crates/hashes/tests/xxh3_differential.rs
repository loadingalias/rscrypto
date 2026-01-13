use hashes::fast::{Xxh3_64, Xxh3_128};
use proptest::prelude::*;
use traits::FastHash as _;

fn xxh3_64_ref(seed: u64, data: &[u8]) -> u64 {
  xxhash_rust::xxh3::xxh3_64_with_seed(data, seed)
}

fn xxh3_128_ref(seed: u64, data: &[u8]) -> u128 {
  xxhash_rust::xxh3::xxh3_128_with_seed(data, seed)
}

proptest! {
  #[test]
  fn xxh3_64_matches_xxhash_rust(seed in any::<u64>(), data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    let ours = Xxh3_64::hash_with_seed(seed, &data);
    let expected = xxh3_64_ref(seed, &data);
    prop_assert_eq!(ours, expected);
  }

  #[test]
  fn xxh3_128_matches_xxhash_rust(seed in any::<u64>(), data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    let ours = Xxh3_128::hash_with_seed(seed, &data);
    let expected = xxh3_128_ref(seed, &data);
    prop_assert_eq!(ours, expected);
  }
}
