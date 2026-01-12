use hashes::crypto::{Sha3_224, Sha3_256, Sha3_384, Sha3_512};
use proptest::prelude::*;
use traits::Digest as _;

fn sha3_224_ref(data: &[u8]) -> [u8; 28] {
  use sha3::Digest as _;
  let out = sha3::Sha3_224::digest(data);
  let mut bytes = [0u8; 28];
  bytes.copy_from_slice(&out);
  bytes
}

fn sha3_256_ref(data: &[u8]) -> [u8; 32] {
  use sha3::Digest as _;
  let out = sha3::Sha3_256::digest(data);
  let mut bytes = [0u8; 32];
  bytes.copy_from_slice(&out);
  bytes
}

fn sha3_384_ref(data: &[u8]) -> [u8; 48] {
  use sha3::Digest as _;
  let out = sha3::Sha3_384::digest(data);
  let mut bytes = [0u8; 48];
  bytes.copy_from_slice(&out);
  bytes
}

fn sha3_512_ref(data: &[u8]) -> [u8; 64] {
  use sha3::Digest as _;
  let out = sha3::Sha3_512::digest(data);
  let mut bytes = [0u8; 64];
  bytes.copy_from_slice(&out);
  bytes
}

proptest! {
  #[test]
  fn sha3_224_matches_sha3_crate(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    prop_assert_eq!(Sha3_224::digest(&data), sha3_224_ref(&data));
  }

  #[test]
  fn sha3_256_matches_sha3_crate(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    prop_assert_eq!(Sha3_256::digest(&data), sha3_256_ref(&data));
  }

  #[test]
  fn sha3_384_matches_sha3_crate(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    prop_assert_eq!(Sha3_384::digest(&data), sha3_384_ref(&data));
  }

  #[test]
  fn sha3_512_matches_sha3_crate(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    prop_assert_eq!(Sha3_512::digest(&data), sha3_512_ref(&data));
  }

  #[test]
  fn sha3_224_streaming_matches_sha3_crate(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    let expected = sha3_224_ref(&data);

    let mut h = Sha3_224::new();
    let mut i = 0usize;
    while i < data.len() {
      let step = (data[i] as usize % 97) + 1;
      let end = core::cmp::min(data.len(), i + step);
      h.update(&data[i..end]);
      i = end;
    }

    prop_assert_eq!(h.finalize(), expected);
  }

  #[test]
  fn sha3_256_streaming_matches_sha3_crate(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    let expected = sha3_256_ref(&data);

    let mut h = Sha3_256::new();
    let mut i = 0usize;
    while i < data.len() {
      let step = (data[i] as usize % 97) + 1;
      let end = core::cmp::min(data.len(), i + step);
      h.update(&data[i..end]);
      i = end;
    }

    prop_assert_eq!(h.finalize(), expected);
  }

  #[test]
  fn sha3_384_streaming_matches_sha3_crate(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    let expected = sha3_384_ref(&data);

    let mut h = Sha3_384::new();
    let mut i = 0usize;
    while i < data.len() {
      let step = (data[i] as usize % 97) + 1;
      let end = core::cmp::min(data.len(), i + step);
      h.update(&data[i..end]);
      i = end;
    }

    prop_assert_eq!(h.finalize(), expected);
  }

  #[test]
  fn sha3_512_streaming_matches_sha3_crate(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    let expected = sha3_512_ref(&data);

    let mut h = Sha3_512::new();
    let mut i = 0usize;
    while i < data.len() {
      let step = (data[i] as usize % 97) + 1;
      let end = core::cmp::min(data.len(), i + step);
      h.update(&data[i..end]);
      i = end;
    }

    prop_assert_eq!(h.finalize(), expected);
  }
}
