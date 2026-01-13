use digest::dev::blobby::Blob2Iterator;
use hashes::crypto::{Blake2b512, Blake2s256};

#[test]
fn blake2b_512_official_vectors() {
  let data = include_bytes!("../testdata/blake2/blake2b_fixed.blb");
  for (i, row) in Blob2Iterator::new(data).unwrap().enumerate() {
    let [input, output] = row.unwrap();
    let actual = Blake2b512::digest(input);
    assert_eq!(
      &actual[..],
      output,
      "blake2b-512 vector mismatch at case {i} (len={})",
      input.len()
    );
  }
}

#[test]
fn blake2s_256_official_vectors() {
  // RustCrypto's upstream BLAKE2s vector corpus is for variable-length output;
  // we filter it down to the fixed 32-byte (256-bit) variant.
  let data = include_bytes!("../testdata/blake2/blake2s_variable.blb");
  let mut matched = 0usize;

  for (i, row) in Blob2Iterator::new(data).unwrap().enumerate() {
    let [input, output] = row.unwrap();
    if output.len() != 32 {
      continue;
    }
    matched += 1;
    let actual = Blake2s256::digest(input);
    assert_eq!(
      &actual[..],
      output,
      "blake2s-256 vector mismatch at case {i} (len={})",
      input.len()
    );
  }

  assert!(
    matched > 0,
    "no 32-byte outputs found in blake2s variable vector corpus"
  );
}
