use hashes::fast::{RapidHash64, RapidHash128};
use proptest::prelude::*;
use traits::FastHash as _;

fn rapidhash64_ref(seed: u64, data: &[u8]) -> u64 {
  let secrets = rapidhash::v3::RapidSecrets::seed_cpp(seed);
  rapidhash::v3::rapidhash_v3_seeded(data, &secrets)
}

fn rapidhash128_ref(seed: u64, data: &[u8]) -> u128 {
  let lo = rapidhash64_ref(seed, data);
  let hi = rapidhash64_ref(seed ^ 0x9E37_79B9_7F4A_7C15, data);
  (lo as u128) | ((hi as u128) << 64)
}

proptest! {
  #[test]
  fn rapidhash64_matches_rapidhash(seed in any::<u64>(), data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    let ours = RapidHash64::hash_with_seed(seed, &data);
    let expected = rapidhash64_ref(seed, &data);
    prop_assert_eq!(ours, expected);
  }

  #[test]
  fn rapidhash128_matches_rapidhash(seed in any::<u64>(), data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    let ours = RapidHash128::hash_with_seed(seed, &data);
    let expected = rapidhash128_ref(seed, &data);
    prop_assert_eq!(ours, expected);
  }
}
