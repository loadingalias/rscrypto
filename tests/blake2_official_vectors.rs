#![cfg(feature = "hashes")]

mod support;

use rscrypto::{Blake2b512, Blake2bKey, Blake2s256, Blake2sKey, Digest};
use support::blobby_compat::BlobIterator;

fn run_blake2_vectors<const OUT: usize>(
  data: &'static [u8],
  name: &str,
  mut one_shot: impl FnMut(&[u8], &[u8]) -> [u8; OUT],
  mut streaming: impl FnMut(&[u8], &[u8]) -> [u8; OUT],
) {
  for (index, row) in BlobIterator::<3>::new(data)
    .expect("blake2 vector corpus must parse")
    .enumerate()
  {
    let [input, key, output] = row.expect("BLAKE2 vector row must decode");

    let actual = one_shot(input, key);
    assert_eq!(
      &actual[..],
      output,
      "{name} oneshot mismatch at case {index} (input_len={}, key_len={})",
      input.len(),
      key.len()
    );

    let streamed = streaming(input, key);
    assert_eq!(
      &streamed[..],
      output,
      "{name} streaming mismatch at case {index} (input_len={}, key_len={})",
      input.len(),
      key.len()
    );
  }
}

#[test]
fn blake2s_official_vectors() {
  let data = include_bytes!("../testdata/blake2/blake2s.blb");
  run_blake2_vectors(
    data,
    "blake2s",
    |input, key| {
      if key.is_empty() {
        Blake2s256::digest(input)
      } else {
        let key = Blake2sKey::new(key).expect("official BLAKE2s key length must be valid");
        Blake2s256::keyed_digest(key, input)
      }
    },
    |input, key| {
      let mut hasher = if key.is_empty() {
        Blake2s256::new()
      } else {
        let key = Blake2sKey::new(key).expect("official BLAKE2s key length must be valid");
        Blake2s256::new_keyed(key)
      };
      for chunk in input.chunks(11) {
        hasher.update(chunk);
      }
      hasher.finalize()
    },
  );
}

#[test]
fn blake2b_official_vectors() {
  let data = include_bytes!("../testdata/blake2/blake2b.blb");
  run_blake2_vectors(
    data,
    "blake2b",
    |input, key| {
      if key.is_empty() {
        Blake2b512::digest(input)
      } else {
        let key = Blake2bKey::new(key).expect("official BLAKE2b key length must be valid");
        Blake2b512::keyed_digest(key, input)
      }
    },
    |input, key| {
      let mut hasher = if key.is_empty() {
        Blake2b512::new()
      } else {
        let key = Blake2bKey::new(key).expect("official BLAKE2b key length must be valid");
        Blake2b512::new_keyed(key)
      };
      for chunk in input.chunks(17) {
        hasher.update(chunk);
      }
      hasher.finalize()
    },
  );
}
