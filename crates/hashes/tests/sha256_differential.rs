use hashes::crypto::{Sha224, Sha256};
use proptest::prelude::*;
use traits::Digest as _;

fn sha2_ref(data: &[u8]) -> [u8; 32] {
  use sha2::Digest as _;
  let out = sha2::Sha256::digest(data);
  let mut bytes = [0u8; 32];
  bytes.copy_from_slice(&out);
  bytes
}

fn sha224_ref(data: &[u8]) -> [u8; 28] {
  use sha2::Digest as _;
  let out = sha2::Sha224::digest(data);
  let mut bytes = [0u8; 28];
  bytes.copy_from_slice(&out);
  bytes
}

proptest! {
  #[test]
  fn sha256_one_shot_matches_sha2(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    prop_assert_eq!(Sha256::digest(&data), sha2_ref(&data));
  }

  #[test]
  fn sha256_streaming_matches_sha2(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    let expected = sha2_ref(&data);

    let mut h = Sha256::new();
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
  fn sha224_one_shot_matches_sha2(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    prop_assert_eq!(Sha224::digest(&data), sha224_ref(&data));
  }

  #[test]
  fn sha224_streaming_matches_sha2(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    let expected = sha224_ref(&data);

    let mut h = Sha224::new();
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
