use digest::dev::blobby::Blob2Iterator;
use hashes::crypto::Sha256;

#[test]
fn sha256_official_vectors() {
  let data = include_bytes!("../testdata/sha2/sha256.blb");
  let iter = Blob2Iterator::new(data).expect("sha256 vector corpus must parse");
  for (i, row) in iter.enumerate() {
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
