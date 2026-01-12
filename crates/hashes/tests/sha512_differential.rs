use hashes::crypto::{Sha384, Sha512, Sha512_224, Sha512_256};
use proptest::prelude::*;
use traits::Digest as _;

fn sha512_ref(data: &[u8]) -> [u8; 64] {
  use sha2::Digest as _;
  let out = sha2::Sha512::digest(data);
  let mut bytes = [0u8; 64];
  bytes.copy_from_slice(&out);
  bytes
}

fn sha384_ref(data: &[u8]) -> [u8; 48] {
  use sha2::Digest as _;
  let out = sha2::Sha384::digest(data);
  let mut bytes = [0u8; 48];
  bytes.copy_from_slice(&out);
  bytes
}

fn sha512_224_ref(data: &[u8]) -> [u8; 28] {
  use sha2::Digest as _;
  let out = sha2::Sha512_224::digest(data);
  let mut bytes = [0u8; 28];
  bytes.copy_from_slice(&out);
  bytes
}

fn sha512_256_ref(data: &[u8]) -> [u8; 32] {
  use sha2::Digest as _;
  let out = sha2::Sha512_256::digest(data);
  let mut bytes = [0u8; 32];
  bytes.copy_from_slice(&out);
  bytes
}

proptest! {
  #[test]
  fn sha512_one_shot_matches_sha2(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    prop_assert_eq!(Sha512::digest(&data), sha512_ref(&data));
  }

  #[test]
  fn sha512_streaming_matches_sha2(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    let expected = sha512_ref(&data);

    let mut h = Sha512::new();
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
  fn sha384_one_shot_matches_sha2(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    prop_assert_eq!(Sha384::digest(&data), sha384_ref(&data));
  }

  #[test]
  fn sha384_streaming_matches_sha2(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    let expected = sha384_ref(&data);

    let mut h = Sha384::new();
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
  fn sha512_224_one_shot_matches_sha2(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    prop_assert_eq!(Sha512_224::digest(&data), sha512_224_ref(&data));
  }

  #[test]
  fn sha512_224_streaming_matches_sha2(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    let expected = sha512_224_ref(&data);

    let mut h = Sha512_224::new();
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
  fn sha512_256_one_shot_matches_sha2(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    prop_assert_eq!(Sha512_256::digest(&data), sha512_256_ref(&data));
  }

  #[test]
  fn sha512_256_streaming_matches_sha2(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    let expected = sha512_256_ref(&data);

    let mut h = Sha512_256::new();
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
