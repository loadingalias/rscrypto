#![cfg(feature = "hashes")]

mod support;

use rscrypto::{
  hashes::crypto::{AsconHash256, AsconXof},
  traits::{Digest as _, Xof as _},
};
use support::blobby_compat::Blob2Iterator;

#[test]
fn ascon_hash256_official_vectors() {
  let data = include_bytes!("../testdata/ascon/asconhash.blb");
  for (i, row) in Blob2Iterator::new(data)
    .expect("ascon hash vector corpus must parse")
    .enumerate()
  {
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
fn ascon_xof_official_vectors() {
  let data = include_bytes!("../testdata/ascon/asconxof.blb");
  for (i, row) in Blob2Iterator::new(data)
    .expect("ascon xof vector corpus must parse")
    .enumerate()
  {
    let [input, output] =
      row.unwrap_or_else(|err| panic!("ascon-xof128 vector row decode failed at case {i}: {err:?}"));
    let mut actual = vec![0u8; output.len()];
    AsconXof::xof(input).squeeze(&mut actual);
    assert_eq!(
      &actual[..],
      output,
      "ascon-xof128 vector mismatch at case {i} (len={})",
      input.len()
    );
  }
}
