use digest::dev::blobby::Blob2Iterator;
use hashes::{
  Digest,
  crypto::{Sha3_224, Sha3_256, Sha3_384, Sha3_512, Shake128, Shake256},
};

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
fn sha3_256_official_vectors() {
  let data = include_bytes!("../testdata/sha3/sha3_256.blb");
  run_fixed_vectors(data, "sha3-256", Sha3_256::digest);
}

#[test]
fn sha3_512_official_vectors() {
  let data = include_bytes!("../testdata/sha3/sha3_512.blb");
  run_fixed_vectors(data, "sha3-512", Sha3_512::digest);
}

#[test]
fn sha3_224_official_vectors() {
  let data = include_bytes!("../testdata/sha3/sha3_224.blb");
  run_fixed_vectors(data, "sha3-224", Sha3_224::digest);
}

#[test]
fn sha3_384_official_vectors() {
  let data = include_bytes!("../testdata/sha3/sha3_384.blb");
  run_fixed_vectors(data, "sha3-384", Sha3_384::digest);
}

fn run_xof_vectors(data: &'static [u8], name: &str, mut hash_into: impl FnMut(&[u8], &mut [u8])) {
  for (i, row) in Blob2Iterator::new(data).unwrap().enumerate() {
    let [input, output] = row.unwrap();
    let mut out = vec![0u8; output.len()];
    hash_into(input, &mut out);
    assert_eq!(
      &out[..],
      output,
      "{name} vector mismatch at case {i} (len={})",
      input.len()
    );
  }
}

#[test]
fn shake128_official_vectors() {
  let data = include_bytes!("../testdata/sha3/shake128.blb");
  run_xof_vectors(data, "shake128", Shake128::hash_into);
}

#[test]
fn shake256_official_vectors() {
  let data = include_bytes!("../testdata/sha3/shake256.blb");
  run_xof_vectors(data, "shake256", Shake256::hash_into);
}
