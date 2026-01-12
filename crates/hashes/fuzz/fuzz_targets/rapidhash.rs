#![no_main]

use hashes::fast::{RapidHash128, RapidHash64};
use libfuzzer_sys::fuzz_target;
use traits::FastHash as _;

const RAPIDHASH128_HI_SEED_XOR: u64 = 0x9E37_79B9_7F4A_7C15;

fuzz_target!(|input: &[u8]| {
  let (seed_bytes, data) = input.split_at(core::cmp::min(8, input.len()));
  let mut seed = 0u64;
  for (i, &b) in seed_bytes.iter().enumerate() {
    seed |= (b as u64) << (i * 8);
  }

  let ours = RapidHash64::hash_with_seed(seed, data);
  let secrets = rapidhash::v3::RapidSecrets::seed_cpp(seed);
  let expected = rapidhash::v3::rapidhash_v3_seeded(data, &secrets);
  assert_eq!(ours, expected);

  // RapidHash128 is defined as two independent 64-bit rapidhash outputs with
  // different seeds; assert both halves match the oracle.
  let out128 = RapidHash128::hash_with_seed(seed, data);
  let lo = out128 as u64;
  let hi = (out128 >> 64) as u64;

  let expected_lo = expected;
  let secrets_hi = rapidhash::v3::RapidSecrets::seed_cpp(seed ^ RAPIDHASH128_HI_SEED_XOR);
  let expected_hi = rapidhash::v3::rapidhash_v3_seeded(data, &secrets_hi);

  assert_eq!(lo, expected_lo);
  assert_eq!(hi, expected_hi);
});
