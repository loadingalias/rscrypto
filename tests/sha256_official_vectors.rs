#![cfg(feature = "hashes")]

mod support;

use rscrypto::hashes::crypto::Sha256;
use support::blobby_compat::BlobIterator;

#[test]
fn sha256_official_vectors() {
  let data = include_bytes!("../testdata/sha2/sha256.blb");
  for (i, row) in BlobIterator::<2>::new(data)
    .expect("sha256 vector corpus must parse")
    .enumerate()
  {
    let [input, output] = row.expect("SHA-256 vector row must decode");
    let actual = Sha256::digest(input);
    assert_eq!(
      &actual[..],
      output,
      "sha256 vector mismatch at case {i} (len={})",
      input.len()
    );
  }
}
