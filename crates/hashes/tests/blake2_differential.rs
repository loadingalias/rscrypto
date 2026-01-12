use hashes::crypto::{Blake2b512, Blake2s256};
use proptest::prelude::*;
use traits::Digest as _;

fn blake2s256_ref(data: &[u8]) -> [u8; 32] {
  use blake2::Digest as _;
  let out = blake2::Blake2s256::digest(data);
  let mut bytes = [0u8; 32];
  bytes.copy_from_slice(&out);
  bytes
}

fn blake2b512_ref(data: &[u8]) -> [u8; 64] {
  use blake2::Digest as _;
  let out = blake2::Blake2b512::digest(data);
  let mut bytes = [0u8; 64];
  bytes.copy_from_slice(&out);
  bytes
}

proptest! {
  #[test]
  fn blake2s256_one_shot_matches_blake2(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    prop_assert_eq!(Blake2s256::digest(&data), blake2s256_ref(&data));
  }

  #[test]
  fn blake2s256_streaming_matches_blake2(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    let expected = blake2s256_ref(&data);
    let mut h = Blake2s256::new();

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
  fn blake2b512_one_shot_matches_blake2(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    prop_assert_eq!(Blake2b512::digest(&data), blake2b512_ref(&data));
  }

  #[test]
  fn blake2b512_streaming_matches_blake2(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
    let expected = blake2b512_ref(&data);
    let mut h = Blake2b512::new();

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
