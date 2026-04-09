#![cfg(feature = "hashes")]

use proptest::prelude::*;
use rscrypto::{Cshake256, traits::Xof as _};

fn cshake256_ref(function_name: &[u8], customization: &[u8], data: &[u8], out: &mut [u8]) {
  use tiny_keccak::{CShake, Hasher as _};

  let mut hasher = CShake::v256(function_name, customization);
  hasher.update(data);
  hasher.finalize(out);
}

proptest! {
  #[test]
  fn cshake256_one_shot_matches_sha3_crate(
    function_name in proptest::collection::vec(any::<u8>(), 0..32),
    customization in proptest::collection::vec(any::<u8>(), 0..64),
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..1024,
  ) {
    let mut expected = vec![0u8; out_len];
    cshake256_ref(&function_name, &customization, &data, &mut expected);

    let mut actual = vec![0u8; out_len];
    Cshake256::hash_into(&function_name, &customization, &data, &mut actual);

    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn cshake256_streaming_matches_sha3_crate(
    function_name in proptest::collection::vec(any::<u8>(), 0..32),
    customization in proptest::collection::vec(any::<u8>(), 0..64),
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..1024,
  ) {
    let mut expected = vec![0u8; out_len];
    cshake256_ref(&function_name, &customization, &data, &mut expected);

    let mut hasher = Cshake256::new(&function_name, &customization);
    let mut i = 0usize;
    while i < data.len() {
      let step = (data[i] as usize % 97) + 1;
      let end = core::cmp::min(data.len(), i + step);
      hasher.update(&data[i..end]);
      i = end;
    }

    let mut actual = vec![0u8; out_len];
    hasher.finalize_xof().squeeze(&mut actual);
    prop_assert_eq!(actual, expected);
  }
}
