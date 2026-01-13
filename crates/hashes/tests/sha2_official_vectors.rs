use digest::dev::blobby::Blob2Iterator;
use hashes::crypto::{Sha224, Sha384, Sha512, Sha512_224, Sha512_256};

fn run_fixed_vectors<const OUT: usize>(data: &'static [u8], name: &str, mut digest: impl FnMut(&[u8]) -> [u8; OUT]) {
  for (i, row) in Blob2Iterator::new(data).unwrap().enumerate() {
    let [input, output] = row.unwrap();
    let actual = digest(input);
    assert_eq!(
      &actual[..],
      output,
      "{name} vector mismatch at case {i} (len={})",
      input.len()
    );
  }
}

#[test]
fn sha224_official_vectors() {
  let data = include_bytes!("../testdata/sha2/sha224.blb");
  run_fixed_vectors(data, "sha224", Sha224::digest);
}

#[test]
fn sha384_official_vectors() {
  let data = include_bytes!("../testdata/sha2/sha384.blb");
  run_fixed_vectors(data, "sha384", Sha384::digest);
}

#[test]
fn sha512_official_vectors() {
  let data = include_bytes!("../testdata/sha2/sha512.blb");
  run_fixed_vectors(data, "sha512", Sha512::digest);
}

#[test]
fn sha512_224_official_vectors() {
  let data = include_bytes!("../testdata/sha2/sha512_224.blb");
  run_fixed_vectors(data, "sha512/224", Sha512_224::digest);
}

#[test]
fn sha512_256_official_vectors() {
  let data = include_bytes!("../testdata/sha2/sha512_256.blb");
  run_fixed_vectors(data, "sha512/256", Sha512_256::digest);
}
