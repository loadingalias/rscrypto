#![cfg(feature = "blake3")]

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
  (0..len)
    .map(|i| u8::try_from(i % 251).expect("remainder modulo 251 must fit in one byte"))
    .collect()
}

#[test]
fn blake3_digest_const_matches_reference_and_streaming_for_every_one_chunk_length() {
  for len in 0..=1024 {
    let data = patterned_bytes(len);
    let digest = Blake3::digest_const(&data);
    assert_eq!(digest, blake3_ref_hash(&data), "reference mismatch at len={len}");
    let mut streaming = Blake3::new();
    streaming.update(&data);
    assert_eq!(digest, streaming.finalize(), "streaming mismatch at len={len}");
  }
}

#[test]
#[should_panic(expected = "Blake3::digest_const accepts at most 1,024 bytes")]
fn blake3_digest_const_rejects_more_than_one_chunk() {
  core::hint::black_box(Blake3::digest_const(core::hint::black_box(&[0; 1025])));
}

proptest! {
  #[test]
  fn blake3_digest_const_matches_official(data in proptest::collection::vec(any::<u8>(), 0..=1024)) {
    prop_assert_eq!(Blake3::digest_const(&data), blake3_ref_hash(&data));
  }

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
      let step = (usize::from(data[i]) % 251).strict_add(1);
      let end = core::cmp::min(data.len(), i.strict_add(step));
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
