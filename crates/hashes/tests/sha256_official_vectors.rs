use digest::dev::blobby::Blob2Iterator;
use hashes::crypto::Sha256;

#[test]
fn sha256_official_vectors() {
  let data = include_bytes!("../testdata/sha2/sha256.blb");
  for (i, row) in Blob2Iterator::new(data).unwrap().enumerate() {
    let [input, output] = row.unwrap();
    let actual = Sha256::digest(input);
    assert_eq!(
      &actual[..],
      output,
      "sha256 vector mismatch at case {i} (len={})",
      input.len()
    );
  }
}
