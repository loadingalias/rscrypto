#![no_main]

use hashes::crypto::{Blake2b512, Blake2s256};
use libfuzzer_sys::fuzz_target;
use traits::Digest as _;

fn split_point(input: &[u8]) -> usize {
  if input.is_empty() {
    return 0;
  }
  (input[0] as usize) % (input.len() + 1)
}

fuzz_target!(|input: &[u8]| {
  let split = split_point(input);
  let (a, b) = input.split_at(split);

  {
    let ours = Blake2s256::digest(input);
    let mut h = Blake2s256::new();
    h.update(a);
    h.update(b);
    assert_eq!(ours, h.finalize());

    use blake2::Digest as _;
    let ref_out = blake2::Blake2s256::digest(input);
    let mut expected = [0u8; 32];
    expected.copy_from_slice(&ref_out);
    assert_eq!(ours, expected);
  }

  {
    let ours = Blake2b512::digest(input);
    let mut h = Blake2b512::new();
    h.update(a);
    h.update(b);
    assert_eq!(ours, h.finalize());

    use blake2::Digest as _;
    let ref_out = blake2::Blake2b512::digest(input);
    let mut expected = [0u8; 64];
    expected.copy_from_slice(&ref_out);
    assert_eq!(ours, expected);
  }
});

