use hashes::crypto::Blake3;
use proptest::prelude::*;
use traits::{Digest as _, Xof as _};

fn blake3_ref_hash(data: &[u8]) -> [u8; 32] {
  *blake3::hash(data).as_bytes()
}

fn blake3_ref_keyed(key: &[u8; 32], data: &[u8]) -> [u8; 32] {
  *blake3::keyed_hash(key, data).as_bytes()
}

fn blake3_ref_derive(context: &str, data: &[u8]) -> [u8; 32] {
  blake3::derive_key(context, data)
}

proptest! {
  #[test]
  fn blake3_one_shot_matches_official(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    prop_assert_eq!(Blake3::digest(&data), blake3_ref_hash(&data));
  }

  #[test]
  fn blake3_streaming_matches_official(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    let expected = blake3_ref_hash(&data);

    let mut h = Blake3::new();
    let mut i = 0usize;
    while i < data.len() {
      let step = (data[i] as usize % 251) + 1;
      let end = core::cmp::min(data.len(), i + step);
      h.update(&data[i..end]);
      i = end;
    }

    prop_assert_eq!(h.finalize(), expected);
  }

  #[test]
  fn blake3_xof_matches_official(data in proptest::collection::vec(any::<u8>(), 0..4096), out_len in 0usize..2048) {
    let mut expected = vec![0u8; out_len];
    let mut ref_hasher = blake3::Hasher::new();
    ref_hasher.update(&data);
    ref_hasher.finalize_xof().fill(&mut expected);

    let mut h = Blake3::new();
    h.update(&data);
    let mut xof = h.finalize_xof();
    let mut actual = vec![0u8; out_len];
    xof.squeeze(&mut actual);

    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn blake3_keyed_matches_official(
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    key in any::<[u8; 32]>(),
  ) {
    let expected = blake3_ref_keyed(&key, &data);
    let mut h = Blake3::new_keyed(&key);
    h.update(&data);
    prop_assert_eq!(h.finalize(), expected);
  }

  #[test]
  fn blake3_derive_key_matches_official(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    const CONTEXT: &str = "rscrypto blake3 derive-key test context";

    let expected = blake3_ref_derive(CONTEXT, &data);
    let mut h = Blake3::new_derive_key(CONTEXT);
    h.update(&data);
    prop_assert_eq!(h.finalize(), expected);
  }
}
