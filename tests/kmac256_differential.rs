#![cfg(feature = "auth")]

use proptest::prelude::*;
use rscrypto::Kmac256;

fn kmac256_ref(key: &[u8], customization: &[u8], data: &[u8], out: &mut [u8]) {
  use tiny_keccak::Hasher as _;

  let mut kmac = tiny_keccak::Kmac::v256(key, customization);
  kmac.update(data);
  kmac.finalize(out);
}

proptest! {
  #[test]
  fn kmac256_matches_tiny_keccak(
    key in proptest::collection::vec(any::<u8>(), 0..192),
    customization in proptest::collection::vec(any::<u8>(), 0..192),
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..256,
  ) {
    let mut expected = vec![0u8; out_len];
    kmac256_ref(&key, &customization, &data, &mut expected);

    let mut actual = vec![0u8; out_len];
    Kmac256::mac_into(&key, &customization, &data, &mut actual);

    prop_assert_eq!(actual, expected.clone());
    prop_assert_eq!(Kmac256::verify(&key, &customization, &data, &expected), Ok(()));
  }

  #[test]
  fn kmac256_streaming_matches_tiny_keccak(
    key in proptest::collection::vec(any::<u8>(), 0..192),
    customization in proptest::collection::vec(any::<u8>(), 0..192),
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..256,
  ) {
    let mut expected = vec![0u8; out_len];
    kmac256_ref(&key, &customization, &data, &mut expected);

    let mut kmac = Kmac256::new(&key, &customization);
    let mut i = 0usize;
    while i < data.len() {
      let step = (data[i] as usize % 97) + 1;
      let end = core::cmp::min(data.len(), i + step);
      kmac.update(&data[i..end]);
      i = end;
    }

    let mut actual = vec![0u8; out_len];
    kmac.finalize_into(&mut actual);
    prop_assert_eq!(actual, expected);
  }
}

#[test]
fn kmac256_exact_rate_key_bytepad_matches_tiny_keccak() {
  use tiny_keccak::Hasher as _;

  let key = [0x42; 131];
  let mut expected = [0u8; 64];
  let mut oracle = tiny_keccak::Kmac::v256(&key, b"");
  oracle.update(b"");
  oracle.finalize(&mut expected);

  let mut actual = [0u8; 64];
  Kmac256::mac_into(&key, b"", b"", &mut actual);

  assert_eq!(actual, expected);
}
