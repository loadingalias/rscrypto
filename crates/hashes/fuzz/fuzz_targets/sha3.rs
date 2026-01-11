#![no_main]

use hashes::crypto::{Sha3_256, Sha3_512};
use libfuzzer_sys::fuzz_target;
use traits::Digest as _;

fuzz_target!(|data: &[u8]| {
  let ours_256 = Sha3_256::digest(data);
  let ours_512 = Sha3_512::digest(data);

  use sha3::Digest as _;
  let ref_256 = sha3::Sha3_256::digest(data);
  let ref_512 = sha3::Sha3_512::digest(data);

  let mut exp_256 = [0u8; 32];
  exp_256.copy_from_slice(&ref_256);
  let mut exp_512 = [0u8; 64];
  exp_512.copy_from_slice(&ref_512);

  assert_eq!(ours_256, exp_256);
  assert_eq!(ours_512, exp_512);
});

