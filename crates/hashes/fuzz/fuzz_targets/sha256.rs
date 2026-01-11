#![no_main]

use hashes::crypto::Sha256;
use libfuzzer_sys::fuzz_target;
use traits::Digest as _;

fuzz_target!(|data: &[u8]| {
  let ours = Sha256::digest(data);

  use sha2::Digest as _;
  let ref_out = sha2::Sha256::digest(data);
  let mut expected = [0u8; 32];
  expected.copy_from_slice(&ref_out);

  assert_eq!(ours, expected);
});

