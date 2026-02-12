use digest::dev::blobby::Blob2Iterator;
use hashes::crypto::{AsconHash256, AsconXof128};
use traits::Digest as _;

#[test]
fn ascon_hash256_official_vectors() {
  let data = include_bytes!("../testdata/ascon/asconhash.blb");
  let iter = Blob2Iterator::new(data).expect("ascon hash vector corpus must parse");
  for (i, row) in iter.enumerate() {
    let [input, output] =
      row.unwrap_or_else(|err| panic!("ascon-hash256 vector row decode failed at case {i}: {err:?}"));
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
  let iter = Blob2Iterator::new(data).expect("ascon xof vector corpus must parse");
  for (i, row) in iter.enumerate() {
    let [input, output] =
      row.unwrap_or_else(|err| panic!("ascon-xof128 vector row decode failed at case {i}: {err:?}"));
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
