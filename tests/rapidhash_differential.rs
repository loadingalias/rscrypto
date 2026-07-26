#![cfg(feature = "rapidhash")]

use core::hash::{BuildHasher, Hasher};

use proptest::prelude::*;
use rscrypto::{RapidHash64, RapidRandomState, RapidSeededState};

fn rapidhash64_ref(seed: u64, data: &[u8]) -> u64 {
  let secrets = rapidhash::v3::RapidSecrets::seed_cpp(seed);
  rapidhash::v3::rapidhash_v3_seeded(data, &secrets)
}

fn hash_bytes(state: &impl BuildHasher, data: &[u8]) -> u64 {
  let mut hasher = state.build_hasher();
  hasher.write(data);
  hasher.finish()
}

#[test]
fn collection_state_schedules_match_rapidhash() {
  let lengths: [usize; 12] = [1, 3, 4, 8, 16, 17, 48, 112, 113, 224, 225, 511];

  for seed in [0, 1, u64::MAX, 0x243f_6a88_85a3_08d3] {
    for len in lengths {
      let data: Vec<u8> = (0..len)
        .map(|index| index.wrapping_mul(131).wrapping_add(17) as u8)
        .collect();

      let deterministic = RapidSeededState::new(seed);
      assert_eq!(
        hash_bytes(&deterministic, &data),
        rapidhash64_ref(seed, &data),
        "deterministic seed={seed:#x}, len={len}"
      );

      let randomized = RapidRandomState::try_new_with(|out| {
        out.copy_from_slice(&seed.to_le_bytes());
        Ok::<_, ()>(())
      })
      .unwrap();
      let secrets = rapidhash::v3::RapidSecrets::seed(seed);
      assert_eq!(
        hash_bytes(&randomized, &data),
        rapidhash::v3::rapidhash_v3_seeded(&data, &secrets),
        "randomized seed={seed:#x}, len={len}"
      );
    }
  }
}

proptest! {
  #[test]
  fn rapidhash64_matches_rapidhash(seed in any::<u64>(), data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    let ours = RapidHash64::hash_with_seed(seed, &data);
    let expected = rapidhash64_ref(seed, &data);
    prop_assert_eq!(ours, expected);
  }
}
