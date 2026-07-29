#![cfg(feature = "hashes")]

use proptest::prelude::*;
use rscrypto::{Cshake128, Cshake256, traits::Xof as _};

fn cshake128_ref(function_name: &[u8], customization: &[u8], data: &[u8], out: &mut [u8]) {
  use tiny_keccak::{CShake, Hasher as _};

  let mut hasher = CShake::v128(function_name, customization);
  hasher.update(data);
  hasher.finalize(out);
}

fn cshake256_ref(function_name: &[u8], customization: &[u8], data: &[u8], out: &mut [u8]) {
  use tiny_keccak::{CShake, Hasher as _};

  let mut hasher = CShake::v256(function_name, customization);
  hasher.update(data);
  hasher.finalize(out);
}

fn encoded_string_len(len: usize) -> usize {
  let bits = len * 8;
  let width = ((usize::BITS - bits.leading_zeros()) as usize).div_ceil(8).max(1);
  1 + width + len
}

fn bytepad_is_aligned(rate: usize, function_name_len: usize, customization_len: usize) -> bool {
  (2 + encoded_string_len(function_name_len) + encoded_string_len(customization_len)).is_multiple_of(rate)
}

fn decode_hex_64(value: &str) -> [u8; 64] {
  assert_eq!(value.len(), 128);
  let mut out = [0u8; 64];
  for (index, byte) in out.iter_mut().enumerate() {
    let offset = index * 2;
    *byte = u8::from_str_radix(&value[offset..offset + 2], 16).unwrap();
  }
  out
}

proptest! {
  #[test]
  fn cshake128_one_shot_matches_tiny_keccak(
    function_name in proptest::collection::vec(any::<u8>(), 0..192),
    customization in proptest::collection::vec(any::<u8>(), 0..192),
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..1024,
  ) {
    prop_assume!(!bytepad_is_aligned(168, function_name.len(), customization.len()));
    let mut expected = vec![0u8; out_len];
    cshake128_ref(&function_name, &customization, &data, &mut expected);

    let mut actual = vec![0u8; out_len];
    Cshake128::hash_into(&function_name, &customization, &data, &mut actual);

    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn cshake128_streaming_matches_tiny_keccak(
    function_name in proptest::collection::vec(any::<u8>(), 0..192),
    customization in proptest::collection::vec(any::<u8>(), 0..192),
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..1024,
  ) {
    prop_assume!(!bytepad_is_aligned(168, function_name.len(), customization.len()));
    let mut expected = vec![0u8; out_len];
    cshake128_ref(&function_name, &customization, &data, &mut expected);

    let mut hasher = Cshake128::new(&function_name, &customization);
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

  #[test]
  fn cshake256_one_shot_matches_tiny_keccak(
    function_name in proptest::collection::vec(any::<u8>(), 0..192),
    customization in proptest::collection::vec(any::<u8>(), 0..192),
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..1024,
  ) {
    prop_assume!(!bytepad_is_aligned(136, function_name.len(), customization.len()));
    let mut expected = vec![0u8; out_len];
    cshake256_ref(&function_name, &customization, &data, &mut expected);

    let mut actual = vec![0u8; out_len];
    Cshake256::hash_into(&function_name, &customization, &data, &mut actual);

    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn cshake256_streaming_matches_tiny_keccak(
    function_name in proptest::collection::vec(any::<u8>(), 0..192),
    customization in proptest::collection::vec(any::<u8>(), 0..192),
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    out_len in 0usize..1024,
  ) {
    prop_assume!(!bytepad_is_aligned(136, function_name.len(), customization.len()));
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

#[test]
fn cshake_bytepad_rate_boundaries_match_go_crypto_sha3() {
  // Generated independently with Go 1.26.5's standard-library
  // crypto/sha3.NewCSHAKE128 and NewCSHAKE256.
  let cases128 = [
    (
      160,
      "99aba19b4ac53e5df9aa569831ee87fc3cb063731b03abd73f3ac54d6b93a437dd17fab8c0961be98b036179f212bba251f8ec0000a3ed1fb5121e9ac1c564bc",
    ),
    (
      161,
      "957923a2379d6fde510a97f8d53af1131f543e8080fcb1a5db4acbb15bea189f6f81a0986827ce523673d9947dec1c0b99edaa443cc492cfbabeccfd951c1299",
    ),
    (
      162,
      "092ddeceaa1c4d98c91bbae1de6c971988f000c257c4c4ed89792698be2145347b9bc3589c8317c9654207f3b8fb543630c91ce227312a48429841161dcac583",
    ),
  ];
  for (name_len, expected) in cases128 {
    let mut actual = [0u8; 64];
    Cshake128::hash_into(&vec![0xff; name_len], b"", b"", &mut actual);
    assert_eq!(actual, decode_hex_64(expected), "cSHAKE128 N={name_len}");
  }

  let cases256 = [
    (
      128,
      "a85b94a121902b2e16fad687bbbc27698f6cb9517f49567d0b925abd93f794408ad99a30c61e626cbd216525505aac7c3cbcbd9fe02ad0381eb2bccf60e8e989",
    ),
    (
      129,
      "495d281b373f64e33ea5e96efc3e13a6da5897397e02cc9f6e5c9f9e03312ff116185a092f41ed5f9bbaf18db0d31d6135cad43308400a07f70d446a630263fe",
    ),
    (
      130,
      "39887e69a44f8e722f6597285f78841ae51ec3c65447f8f7471be1a257003fdf3d550afa356e323c561fb7eb8e9ff25720cf99458734d8bb245376d79f9533a6",
    ),
  ];
  for (name_len, expected) in cases256 {
    let mut actual = [0u8; 64];
    Cshake256::hash_into(&vec![0xff; name_len], b"", b"", &mut actual);
    assert_eq!(actual, decode_hex_64(expected), "cSHAKE256 N={name_len}");
  }
}
