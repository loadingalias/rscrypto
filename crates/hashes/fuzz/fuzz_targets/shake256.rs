#![no_main]

use hashes::crypto::Shake256;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
  let mut ours = [0u8; 64];
  Shake256::hash_into(data, &mut ours);

  use sha3::digest::{ExtendableOutput, Update, XofReader};
  let mut h = sha3::Shake256::default();
  h.update(data);
  let mut reader = h.finalize_xof();
  let mut expected = [0u8; 64];
  reader.read(&mut expected);

  assert_eq!(ours, expected);
});

