#![cfg(feature = "hashes")]

use proptest::prelude::*;
use rscrypto::{hashes::fast::RapidHash64, traits::FastHash as _};

fn rapidhash64_ref(seed: u64, data: &[u8]) -> u64 {
  let secrets = rapidhash::v3::RapidSecrets::seed_cpp(seed);
  rapidhash::v3::rapidhash_v3_seeded(data, &secrets)
}

proptest! {
  #[test]
  fn rapidhash64_matches_rapidhash(seed in any::<u64>(), data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    let ours = RapidHash64::hash_with_seed(seed, &data);
    let expected = rapidhash64_ref(seed, &data);
    prop_assert_eq!(ours, expected);
  }
}
