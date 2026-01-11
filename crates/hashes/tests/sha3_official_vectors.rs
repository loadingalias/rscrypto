use digest::dev::blobby::Blob2Iterator;
use hashes::crypto::{Sha3_256, Sha3_512};
use traits::Digest as _;

#[test]
fn sha3_256_official_vectors() {
  let data = include_bytes!("../testdata/sha3/sha3_256.blb");
  for (i, row) in Blob2Iterator::new(data).unwrap().enumerate() {
    let [input, output] = row.unwrap();
    let actual = Sha3_256::digest(input);
    assert_eq!(
      &actual[..],
      output,
      "sha3-256 vector mismatch at case {i} (len={})",
      input.len()
    );
  }
}

#[test]
fn sha3_512_official_vectors() {
  let data = include_bytes!("../testdata/sha3/sha3_512.blb");
  for (i, row) in Blob2Iterator::new(data).unwrap().enumerate() {
    let [input, output] = row.unwrap();
    let actual = Sha3_512::digest(input);
    assert_eq!(
      &actual[..],
      output,
      "sha3-512 vector mismatch at case {i} (len={})",
      input.len()
    );
  }
}
