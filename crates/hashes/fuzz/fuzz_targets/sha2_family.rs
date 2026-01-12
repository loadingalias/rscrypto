#![no_main]

use hashes::crypto::{Sha224, Sha256, Sha384, Sha512, Sha512_224, Sha512_256};
use libfuzzer_sys::fuzz_target;
use traits::Digest as _;

fn split_point(input: &[u8]) -> usize {
  if input.is_empty() {
    return 0;
  }
  (input[0] as usize) % (input.len() + 1)
}

fuzz_target!(|input: &[u8]| {
  // Use a data-dependent split to exercise streaming boundaries, while still
  // hashing the entire `input` buffer.
  let split = split_point(input);
  let (a, b) = input.split_at(split);

  {
    let ours = Sha224::digest(input);
    let mut h = Sha224::new();
    h.update(a);
    h.update(b);
    assert_eq!(ours, h.finalize());

    use sha2::Digest as _;
    let ref_out = sha2::Sha224::digest(input);
    let mut expected = [0u8; 28];
    expected.copy_from_slice(&ref_out);
    assert_eq!(ours, expected);
  }

  {
    let ours = Sha256::digest(input);
    let mut h = Sha256::new();
    h.update(a);
    h.update(b);
    assert_eq!(ours, h.finalize());

    use sha2::Digest as _;
    let ref_out = sha2::Sha256::digest(input);
    let mut expected = [0u8; 32];
    expected.copy_from_slice(&ref_out);
    assert_eq!(ours, expected);
  }

  {
    let ours = Sha384::digest(input);
    let mut h = Sha384::new();
    h.update(a);
    h.update(b);
    assert_eq!(ours, h.finalize());

    use sha2::Digest as _;
    let ref_out = sha2::Sha384::digest(input);
    let mut expected = [0u8; 48];
    expected.copy_from_slice(&ref_out);
    assert_eq!(ours, expected);
  }

  {
    let ours = Sha512::digest(input);
    let mut h = Sha512::new();
    h.update(a);
    h.update(b);
    assert_eq!(ours, h.finalize());

    use sha2::Digest as _;
    let ref_out = sha2::Sha512::digest(input);
    let mut expected = [0u8; 64];
    expected.copy_from_slice(&ref_out);
    assert_eq!(ours, expected);
  }

  {
    let ours = Sha512_224::digest(input);
    let mut h = Sha512_224::new();
    h.update(a);
    h.update(b);
    assert_eq!(ours, h.finalize());

    use sha2::Digest as _;
    let ref_out = sha2::Sha512_224::digest(input);
    let mut expected = [0u8; 28];
    expected.copy_from_slice(&ref_out);
    assert_eq!(ours, expected);
  }

  {
    let ours = Sha512_256::digest(input);
    let mut h = Sha512_256::new();
    h.update(a);
    h.update(b);
    assert_eq!(ours, h.finalize());

    use sha2::Digest as _;
    let ref_out = sha2::Sha512_256::digest(input);
    let mut expected = [0u8; 32];
    expected.copy_from_slice(&ref_out);
    assert_eq!(ours, expected);
  }
});

