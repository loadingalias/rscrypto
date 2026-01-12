#![no_main]

use hashes::crypto::{Sha3_224, Sha3_256, Sha3_384, Sha3_512};
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
    let ours = Sha3_224::digest(input);
    let mut h = Sha3_224::new();
    h.update(a);
    h.update(b);
    assert_eq!(ours, h.finalize());

    use sha3::Digest as _;
    let ref_out = sha3::Sha3_224::digest(input);
    let mut expected = [0u8; 28];
    expected.copy_from_slice(&ref_out);
    assert_eq!(ours, expected);
  }

  {
    let ours = Sha3_256::digest(input);
    let mut h = Sha3_256::new();
    h.update(a);
    h.update(b);
    assert_eq!(ours, h.finalize());

    use sha3::Digest as _;
    let ref_out = sha3::Sha3_256::digest(input);
    let mut expected = [0u8; 32];
    expected.copy_from_slice(&ref_out);
    assert_eq!(ours, expected);
  }

  {
    let ours = Sha3_384::digest(input);
    let mut h = Sha3_384::new();
    h.update(a);
    h.update(b);
    assert_eq!(ours, h.finalize());

    use sha3::Digest as _;
    let ref_out = sha3::Sha3_384::digest(input);
    let mut expected = [0u8; 48];
    expected.copy_from_slice(&ref_out);
    assert_eq!(ours, expected);
  }

  {
    let ours = Sha3_512::digest(input);
    let mut h = Sha3_512::new();
    h.update(a);
    h.update(b);
    assert_eq!(ours, h.finalize());

    use sha3::Digest as _;
    let ref_out = sha3::Sha3_512::digest(input);
    let mut expected = [0u8; 64];
    expected.copy_from_slice(&ref_out);
    assert_eq!(ours, expected);
  }
});

