#![cfg(feature = "hashes")]

mod support;

use rscrypto::hashes::crypto::Sha256;
use support::blobby_compat::Blob2Iterator;

#[test]
fn sha256_official_vectors() {
  let data = include_bytes!("../testdata/sha2/sha256.blb");
  for (i, row) in Blob2Iterator::new(data)
    .expect("sha256 vector corpus must parse")
    .enumerate()
  {
    let [input, output] = row.unwrap_or_else(|err| panic!("sha256 vector row decode failed at case {i}: {err:?}"));
    let actual = Sha256::digest(input);
    assert_eq!(
      &actual[..],
      output,
      "sha256 vector mismatch at case {i} (len={})",
      input.len()
    );
  }
}
