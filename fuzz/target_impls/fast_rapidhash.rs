use core::hash::Hasher;

use rscrypto::{RapidHash64, RapidStreamHasher};
use rscrypto_fuzz::{FuzzInput, some_or_return};

pub fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let seed_bytes: [u8; 8] = some_or_return!(input.bytes());
  let partitions: [u8; 8] = some_or_return!(input.bytes());
  let data = input.rest();
  let seed = u64::from_le_bytes(seed_bytes);

  let secrets = rapidhash::v3::RapidSecrets::seed_cpp(seed);
  let oracle = rapidhash::v3::rapidhash_v3_seeded(data, &secrets);
  assert_eq!(
    RapidHash64::hash_with_seed(seed, data),
    oracle,
    "rapidhash-v3 oracle mismatch"
  );

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

  assert_eq!(
    RapidHash64::hash(data),
    RapidHash64::hash_with_seed(0, data),
    "rapidhash-v3 default vs seed=0"
  );
}
