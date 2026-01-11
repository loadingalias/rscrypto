#![no_main]

use hashes::crypto::Blake3;
use libfuzzer_sys::fuzz_target;
use traits::Digest as _;

fuzz_target!(|data: &[u8]| {
  let ours = Blake3::digest(data);
  let expected = *blake3::hash(data).as_bytes();
  assert_eq!(ours, expected);
});

