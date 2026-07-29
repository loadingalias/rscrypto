#![cfg(feature = "kmac")]

use proptest::prelude::*;
use rscrypto::Kmac128;

fn kmac128_ref(key: &[u8], customization: &[u8], data: &[u8], out: &mut [u8]) {
  use tiny_keccak::Hasher as _;

  let mut kmac = tiny_keccak::Kmac::v128(key, customization);
  kmac.update(data);
  kmac.finalize(out);
}

fn encoded_string_len(len: usize) -> usize {
  let bits = len * 8;
  let width = ((usize::BITS - bits.leading_zeros()) as usize).div_ceil(8).max(1);
  1 + width + len
}

fn bytepad_is_aligned(rate: usize, segments: &[usize]) -> bool {
  (2 + segments.iter().map(|&len| encoded_string_len(len)).sum::<usize>()).is_multiple_of(rate)
}

fn decode_hex_32(value: &str) -> [u8; 32] {
  assert_eq!(value.len(), 64);
  let mut out = [0u8; 32];
  for (index, byte) in out.iter_mut().enumerate() {
    let offset = index * 2;
    *byte = u8::from_str_radix(&value[offset..offset + 2], 16).unwrap();
  }
  out
}

proptest! {
  #[test]
  fn kmac128_matches_tiny_keccak(
    key in proptest::collection::vec(any::<u8>(), 0..192),
    customization in proptest::collection::vec(any::<u8>(), 0..192),
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..256,
  ) {
    prop_assume!(!bytepad_is_aligned(168, &[4, customization.len()]));
    prop_assume!(!bytepad_is_aligned(168, &[key.len()]));
    let mut expected = vec![0u8; out_len];
    kmac128_ref(&key, &customization, &data, &mut expected);

    let mut actual = vec![0u8; out_len];
    Kmac128::mac_into(&key, &customization, &data, &mut actual);

    prop_assert_eq!(actual, expected.clone());
    if expected.is_empty() {
      prop_assert!(Kmac128::verify_tag(&key, &customization, &data, &expected).is_err());
    } else {
      prop_assert_eq!(
        Kmac128::verify_tag(&key, &customization, &data, &expected).is_ok(),
        expected.len() >= Kmac128::MIN_AUTH_TAG_SIZE
      );
      prop_assert_eq!(
        Kmac128::verify_tag_primitive(&key, &customization, &data, &expected),
        Ok(())
      );
    }
  }

  #[test]
  fn kmac128_streaming_matches_tiny_keccak(
    key in proptest::collection::vec(any::<u8>(), 0..192),
    customization in proptest::collection::vec(any::<u8>(), 0..192),
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..256,
  ) {
    prop_assume!(!bytepad_is_aligned(168, &[4, customization.len()]));
    prop_assume!(!bytepad_is_aligned(168, &[key.len()]));
    let mut expected = vec![0u8; out_len];
    kmac128_ref(&key, &customization, &data, &mut expected);

    let mut kmac = Kmac128::new(&key, &customization);
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
fn kmac128_exact_rate_key_bytepad_matches_openssl() {
  // OpenSSL 3.6.3 KMAC-128, empty customization/message, 32-byte output.
  let key = [0x42; 163];
  let expected = decode_hex_32("6c9c526b592fdabce43190f544cf5ae6671d223d268eb5a206283df289346fc8");
  let mut actual = [0u8; 32];
  Kmac128::mac_into(&key, b"", b"", &mut actual);
  assert_eq!(actual, expected);
}
