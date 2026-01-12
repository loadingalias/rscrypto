#![no_main]

use hashes::crypto::Sha512;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
  let ours = Sha512::digest(data);

  use sha2::Digest as _;
  let ref_out = sha2::Sha512::digest(data);
  let mut expected = [0u8; 64];
  expected.copy_from_slice(&ref_out);

  assert_eq!(ours, expected);
});

