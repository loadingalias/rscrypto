#![cfg(feature = "hashes")]

use proptest::prelude::*;
use rscrypto::{
  hashes::crypto::Blake3,
  traits::{Digest as _, Xof as _},
};

fn blake3_ref_hash(data: &[u8]) -> [u8; 32] {
  *blake3::hash(data).as_bytes()
}

fn blake3_ref_xof(data: &[u8], out: &mut [u8]) {
  let mut h = blake3::Hasher::new();
  h.update(data);
  h.finalize_xof().fill(out);
}

fn blake3_ref_keyed(key: &[u8; 32], data: &[u8]) -> [u8; 32] {
  *blake3::keyed_hash(key, data).as_bytes()
}

fn blake3_ref_keyed_xof(key: &[u8; 32], data: &[u8], out: &mut [u8]) {
  let mut h = blake3::Hasher::new_keyed(key);
  h.update(data);
  h.finalize_xof().fill(out);
}

fn blake3_ref_derive(context: &str, data: &[u8]) -> [u8; 32] {
  blake3::derive_key(context, data)
}

fn blake3_ref_derive_xof(context: &str, data: &[u8], out: &mut [u8]) {
  let mut h = blake3::Hasher::new_derive_key(context);
  h.update(data);
  h.finalize_xof().fill(out);
}

fn patterned_bytes(len: usize) -> Vec<u8> {
  (0..len).map(|i| (i % 251) as u8).collect()
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

#[test]
fn blake3_multi_chunk_then_small_tail_matches_official_in_all_modes() {
  const CHUNK_LEN: usize = 1024;
  const TAIL_LEN: usize = 9;
  const XOF_LEN: usize = 96;
  const CONTEXT: &str = "rscrypto blake3 derive-key regression context";

  let data = patterned_bytes(2 * CHUNK_LEN + TAIL_LEN);
  let (first, second) = data.split_at(2 * CHUNK_LEN);
  let key = *b"whats the Elvish word for friend";

  {
    let mut h = Blake3::new();
    h.update(first);
    h.update(second);
    assert_eq!(h.finalize(), blake3_ref_hash(&data));

    let mut expected = [0u8; XOF_LEN];
    blake3_ref_xof(&data, &mut expected);
    let mut actual = [0u8; XOF_LEN];
    h.finalize_xof().squeeze(&mut actual);
    assert_eq!(actual, expected);
  }

  {
    let mut h = Blake3::new_keyed(&key);
    h.update(first);
    h.update(second);
    assert_eq!(h.finalize(), blake3_ref_keyed(&key, &data));

    let mut expected = [0u8; XOF_LEN];
    blake3_ref_keyed_xof(&key, &data, &mut expected);
    let mut actual = [0u8; XOF_LEN];
    h.finalize_xof().squeeze(&mut actual);
    assert_eq!(actual, expected);
  }

  {
    let mut h = Blake3::new_derive_key(CONTEXT);
    h.update(first);
    h.update(second);
    assert_eq!(h.finalize(), blake3_ref_derive(CONTEXT, &data));

    let mut expected = [0u8; XOF_LEN];
    blake3_ref_derive_xof(CONTEXT, &data, &mut expected);
    let mut actual = [0u8; XOF_LEN];
    h.finalize_xof().squeeze(&mut actual);
    assert_eq!(actual, expected);
  }
}
