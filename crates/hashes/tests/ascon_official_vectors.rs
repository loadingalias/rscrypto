use digest::dev::blobby::Blob2Iterator;
use hashes::crypto::{AsconHash256, AsconXof128};
use traits::Digest as _;

#[test]
fn ascon_hash256_official_vectors() {
  let data = include_bytes!("../testdata/ascon/asconhash.blb");
  for (i, row) in Blob2Iterator::new(data).unwrap().enumerate() {
    let [input, output] = row.unwrap();
    let actual = AsconHash256::digest(input);
    assert_eq!(
      &actual[..],
      output,
      "ascon-hash256 vector mismatch at case {i} (len={})",
      input.len()
    );
  }
}

#[test]
fn ascon_xof128_official_vectors() {
  let data = include_bytes!("../testdata/ascon/asconxof.blb");
  for (i, row) in Blob2Iterator::new(data).unwrap().enumerate() {
    let [input, output] = row.unwrap();
    let mut actual = vec![0u8; output.len()];
    AsconXof128::hash_into(input, &mut actual);
    assert_eq!(
      &actual[..],
      output,
      "ascon-xof128 vector mismatch at case {i} (len={})",
      input.len()
    );
  }
}
