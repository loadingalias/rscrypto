use hashes::crypto::{Kmac128, Kmac256};
use proptest::prelude::*;
use traits::Xof as _;

fn kmac128_ref(key: &[u8], customization: &[u8], data: &[u8], out: &mut [u8]) {
  use tiny_keccak::{Hasher, Kmac};
  let mut h = Kmac::v128(key, customization);
  h.update(data);
  h.finalize(out);
}

fn kmac256_ref(key: &[u8], customization: &[u8], data: &[u8], out: &mut [u8]) {
  use tiny_keccak::{Hasher, Kmac};
  let mut h = Kmac::v256(key, customization);
  h.update(data);
  h.finalize(out);
}

fn kmac128_xof_ref(key: &[u8], customization: &[u8], data: &[u8], out: &mut [u8], split: usize) {
  use tiny_keccak::{Hasher, IntoXof, Kmac, Xof};
  let mut h = Kmac::v128(key, customization);
  h.update(data);
  let mut xof = h.into_xof();
  xof.squeeze(&mut out[..split]);
  xof.squeeze(&mut out[split..]);
}

fn kmac256_xof_ref(key: &[u8], customization: &[u8], data: &[u8], out: &mut [u8], split: usize) {
  use tiny_keccak::{Hasher, IntoXof, Kmac, Xof};
  let mut h = Kmac::v256(key, customization);
  h.update(data);
  let mut xof = h.into_xof();
  xof.squeeze(&mut out[..split]);
  xof.squeeze(&mut out[split..]);
}

proptest! {
  #[test]
  fn kmac128_matches_tiny_keccak(
    key in proptest::collection::vec(any::<u8>(), 0..64),
    customization in proptest::collection::vec(any::<u8>(), 0..64),
    data in proptest::collection::vec(any::<u8>(), 0..2048),
    out_len in 0usize..512,
  ) {
    let mut expected = vec![0u8; out_len];
    kmac128_ref(&key, &customization, &data, &mut expected);

    let mut actual = vec![0u8; out_len];
    let mut h = Kmac128::new(&key, &customization);
    h.update(&data);
    h.finalize_into(&mut actual);

    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn kmac256_matches_tiny_keccak(
    key in proptest::collection::vec(any::<u8>(), 0..64),
    customization in proptest::collection::vec(any::<u8>(), 0..64),
    data in proptest::collection::vec(any::<u8>(), 0..2048),
    out_len in 0usize..512,
  ) {
    let mut expected = vec![0u8; out_len];
    kmac256_ref(&key, &customization, &data, &mut expected);

    let mut actual = vec![0u8; out_len];
    let mut h = Kmac256::new(&key, &customization);
    h.update(&data);
    h.finalize_into(&mut actual);

    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn kmac128_xof_matches_tiny_keccak(
    key in proptest::collection::vec(any::<u8>(), 0..64),
    customization in proptest::collection::vec(any::<u8>(), 0..64),
    data in proptest::collection::vec(any::<u8>(), 0..2048),
    out_len in 0usize..512,
    split in any::<usize>(),
  ) {
    let split = split % (out_len + 1);

    let mut expected = vec![0u8; out_len];
    kmac128_xof_ref(&key, &customization, &data, &mut expected, split);

    let mut actual = vec![0u8; out_len];
    let mut h = Kmac128::new(&key, &customization);
    h.update(&data);
    let mut xof = h.finalize_xof();
    xof.squeeze(&mut actual[..split]);
    xof.squeeze(&mut actual[split..]);

    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn kmac256_xof_matches_tiny_keccak(
    key in proptest::collection::vec(any::<u8>(), 0..64),
    customization in proptest::collection::vec(any::<u8>(), 0..64),
    data in proptest::collection::vec(any::<u8>(), 0..2048),
    out_len in 0usize..512,
    split in any::<usize>(),
  ) {
    let split = split % (out_len + 1);

    let mut expected = vec![0u8; out_len];
    kmac256_xof_ref(&key, &customization, &data, &mut expected, split);

    let mut actual = vec![0u8; out_len];
    let mut h = Kmac256::new(&key, &customization);
    h.update(&data);
    let mut xof = h.finalize_xof();
    xof.squeeze(&mut actual[..split]);
    xof.squeeze(&mut actual[split..]);

    prop_assert_eq!(actual, expected);
  }
}
