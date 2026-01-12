use hashes::crypto::{CShake128, CShake256};
use proptest::prelude::*;
use traits::Xof as _;

fn cshake128_ref(function_name: &[u8], customization: &[u8], data: &[u8], out: &mut [u8]) {
  use sha3::digest::{ExtendableOutput, Update, XofReader};
  let core = if function_name.is_empty() {
    sha3::CShake128Core::new(customization)
  } else {
    sha3::CShake128Core::new_with_function_name(function_name, customization)
  };
  let mut h = sha3::CShake128::from_core(core);
  h.update(data);
  let mut reader = h.finalize_xof();
  reader.read(out);
}

fn cshake256_ref(function_name: &[u8], customization: &[u8], data: &[u8], out: &mut [u8]) {
  use sha3::digest::{ExtendableOutput, Update, XofReader};
  let core = if function_name.is_empty() {
    sha3::CShake256Core::new(customization)
  } else {
    sha3::CShake256Core::new_with_function_name(function_name, customization)
  };
  let mut h = sha3::CShake256::from_core(core);
  h.update(data);
  let mut reader = h.finalize_xof();
  reader.read(out);
}

proptest! {
  #[test]
  fn cshake128_matches_sha3_crate(
    function_name in proptest::collection::vec(any::<u8>(), 0..64),
    customization in proptest::collection::vec(any::<u8>(), 0..64),
    data in proptest::collection::vec(any::<u8>(), 0..2048),
    out_len in 0usize..1024,
  ) {
    let mut expected = vec![0u8; out_len];
    cshake128_ref(&function_name, &customization, &data, &mut expected);

    let mut actual = vec![0u8; out_len];
    CShake128::hash_into(&function_name, &customization, &data, &mut actual);

    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn cshake128_streaming_matches_sha3_crate(
    function_name in proptest::collection::vec(any::<u8>(), 0..64),
    customization in proptest::collection::vec(any::<u8>(), 0..64),
    data in proptest::collection::vec(any::<u8>(), 0..2048),
    out_len in 0usize..1024,
    split in any::<usize>(),
  ) {
    let mut expected = vec![0u8; out_len];
    cshake128_ref(&function_name, &customization, &data, &mut expected);

    let mut h = CShake128::new_with_function_name(&function_name, &customization);
    let split = split % (data.len() + 1);
    h.update(&data[..split]);
    h.update(&data[split..]);

    let mut xof = h.finalize_xof();
    let mut actual = vec![0u8; out_len];
    xof.squeeze(&mut actual);
    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn cshake256_matches_sha3_crate(
    function_name in proptest::collection::vec(any::<u8>(), 0..64),
    customization in proptest::collection::vec(any::<u8>(), 0..64),
    data in proptest::collection::vec(any::<u8>(), 0..2048),
    out_len in 0usize..1024,
  ) {
    let mut expected = vec![0u8; out_len];
    cshake256_ref(&function_name, &customization, &data, &mut expected);

    let mut actual = vec![0u8; out_len];
    CShake256::hash_into(&function_name, &customization, &data, &mut actual);

    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn cshake256_streaming_matches_sha3_crate(
    function_name in proptest::collection::vec(any::<u8>(), 0..64),
    customization in proptest::collection::vec(any::<u8>(), 0..64),
    data in proptest::collection::vec(any::<u8>(), 0..2048),
    out_len in 0usize..1024,
    split in any::<usize>(),
  ) {
    let mut expected = vec![0u8; out_len];
    cshake256_ref(&function_name, &customization, &data, &mut expected);

    let mut h = CShake256::new_with_function_name(&function_name, &customization);
    let split = split % (data.len() + 1);
    h.update(&data[..split]);
    h.update(&data[split..]);

    let mut xof = h.finalize_xof();
    let mut actual = vec![0u8; out_len];
    xof.squeeze(&mut actual);
    prop_assert_eq!(actual, expected);
  }
}
