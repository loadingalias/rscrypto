#![cfg(feature = "ascon-hash")]

mod support;

use rscrypto::{
  hashes::crypto::{AsconHash256, AsconXof},
  traits::{Digest as _, Xof as _},
};
use support::blobby_compat::BlobIterator;

#[test]
fn ascon_hash256_official_vectors() {
  let data = include_bytes!("../testdata/ascon/asconhash.blb");
  for (i, row) in BlobIterator::<2>::new(data)
    .expect("ascon hash vector corpus must parse")
    .enumerate()
  {
    let [input, output] = row.expect("Ascon-Hash256 vector row must decode");
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
  for (i, row) in BlobIterator::<2>::new(data)
    .expect("ascon xof vector corpus must parse")
    .enumerate()
  {
    let [input, output] = row.expect("Ascon-XOF128 vector row must decode");
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
