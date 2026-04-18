#![cfg(feature = "hashes")]

use blake2::{
  Blake2b256 as OracleBlake2b256, Blake2b512 as OracleBlake2b512, Blake2bMac, Blake2s128 as OracleBlake2s128,
  Blake2s256 as OracleBlake2s256, Blake2sMac, Digest as _,
};
use digest::typenum::{U16, U32, U64};
use hmac::{Mac as _, digest::KeyInit};
use proptest::{prelude::*, test_runner::Config as ProptestConfig};
use rscrypto::{Blake2b256, Blake2b512, Blake2bParams, Blake2s128, Blake2s256, Blake2sParams, Digest};

type OracleBlake2bMac256 = Blake2bMac<U32>;
type OracleBlake2bMac512 = Blake2bMac<U64>;
type OracleBlake2sMac128 = Blake2sMac<U16>;
type OracleBlake2sMac256 = Blake2sMac<U32>;

fn split_at_ratio(data: &[u8], ratio: u8) -> (&[u8], &[u8]) {
  let idx = if data.is_empty() {
    0
  } else {
    data.len().strict_mul(ratio as usize) / 255
  };
  data.split_at(idx.min(data.len()))
}

fn patterned_input(seed: u8, len: usize) -> Vec<u8> {
  (0..len).map(|i| seed.wrapping_add((i % 251) as u8)).collect()
}

proptest! {
  #![proptest_config(ProptestConfig::with_cases(64))]

  #[test]
  fn blake2b_fixed_outputs_match_rustcrypto(
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    split in any::<u8>(),
    key_len in 1usize..=64,
    tail_len in 0usize..512,
  ) {
    let (left, right) = split_at_ratio(&data, split);
    let key = &patterned_input(0x42, key_len);
    let tail = patterned_input(0xA5, tail_len);

    let expected_256 = OracleBlake2b256::digest(&data);
    let expected_512 = OracleBlake2b512::digest(&data);
    prop_assert_eq!(&Blake2b256::digest(&data)[..], expected_256.as_slice());
    prop_assert_eq!(&Blake2b512::digest(&data)[..], expected_512.as_slice());

    let mut streaming_256 = Blake2b256::new();
    streaming_256.update(left);
    streaming_256.update(right);
    prop_assert_eq!(&streaming_256.finalize()[..], expected_256.as_slice());

    let mut streaming_512 = Blake2b512::new();
    streaming_512.update(left);
    streaming_512.update(right);
    prop_assert_eq!(&streaming_512.finalize()[..], expected_512.as_slice());

    let mut oracle_keyed_256 = OracleBlake2bMac256::new_from_slice(key).unwrap();
    oracle_keyed_256.update(&data);
    let expected_keyed_256 = oracle_keyed_256.finalize().into_bytes();
    prop_assert_eq!(&Blake2b256::keyed_digest(key, &data)[..], &expected_keyed_256[..]);

    let mut oracle_keyed_512 = OracleBlake2bMac512::new_from_slice(key).unwrap();
    oracle_keyed_512.update(&data);
    let expected_keyed_512 = oracle_keyed_512.finalize().into_bytes();
    prop_assert_eq!(&Blake2b512::keyed_digest(key, &data)[..], &expected_keyed_512[..]);

    let mut reset_256 = Blake2b256::new();
    reset_256.update(&data);
    let _ = reset_256.finalize();
    reset_256.reset();
    reset_256.update(&tail);
    prop_assert_eq!(reset_256.finalize(), Blake2b256::digest(&tail));

    let mut reset_512 = Blake2b512::new();
    reset_512.update(&data);
    let _ = reset_512.finalize();
    reset_512.reset();
    reset_512.update(&tail);
    prop_assert_eq!(reset_512.finalize(), Blake2b512::digest(&tail));
  }

  #[test]
  fn blake2s_fixed_outputs_match_rustcrypto(
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    split in any::<u8>(),
    key_len in 1usize..=32,
    tail_len in 0usize..512,
  ) {
    let (left, right) = split_at_ratio(&data, split);
    let key = &patterned_input(0x24, key_len);
    let tail = patterned_input(0x5A, tail_len);

    let expected_128 = OracleBlake2s128::digest(&data);
    let expected_256 = OracleBlake2s256::digest(&data);
    prop_assert_eq!(&Blake2s128::digest(&data)[..], expected_128.as_slice());
    prop_assert_eq!(&Blake2s256::digest(&data)[..], expected_256.as_slice());

    let mut streaming_128 = Blake2s128::new();
    streaming_128.update(left);
    streaming_128.update(right);
    prop_assert_eq!(&streaming_128.finalize()[..], expected_128.as_slice());

    let mut streaming_256 = Blake2s256::new();
    streaming_256.update(left);
    streaming_256.update(right);
    prop_assert_eq!(&streaming_256.finalize()[..], expected_256.as_slice());

    let mut oracle_keyed_128 = OracleBlake2sMac128::new_from_slice(key).unwrap();
    oracle_keyed_128.update(&data);
    let expected_keyed_128 = oracle_keyed_128.finalize().into_bytes();
    prop_assert_eq!(&Blake2s128::keyed_digest(key, &data)[..], &expected_keyed_128[..]);

    let mut oracle_keyed_256 = OracleBlake2sMac256::new_from_slice(key).unwrap();
    oracle_keyed_256.update(&data);
    let expected_keyed_256 = oracle_keyed_256.finalize().into_bytes();
    prop_assert_eq!(&Blake2s256::keyed_digest(key, &data)[..], &expected_keyed_256[..]);

    let mut reset_128 = Blake2s128::new();
    reset_128.update(&data);
    let _ = reset_128.finalize();
    reset_128.reset();
    reset_128.update(&tail);
    prop_assert_eq!(reset_128.finalize(), Blake2s128::digest(&tail));

    let mut reset_256 = Blake2s256::new();
    reset_256.update(&data);
    let _ = reset_256.finalize();
    reset_256.reset();
    reset_256.update(&tail);
    prop_assert_eq!(reset_256.finalize(), Blake2s256::digest(&tail));
  }

  #[test]
  fn blake2b_params_matches_rustcrypto(
    data in proptest::collection::vec(any::<u8>(), 0..2048),
    key_len in 0usize..=64,
    salt_len in 0usize..=16,
    personal_len in 0usize..=16,
  ) {
    let key = patterned_input(0x11, key_len);
    let salt = patterned_input(0x22, salt_len);
    let personal = patterned_input(0x33, personal_len);

    let key_opt: Option<&[u8]> = if key.is_empty() { None } else { Some(&key) };

    // Oracle: Blake2bMac::new_with_salt_and_personal(key, salt, personal).
    // When key=None the MAC reduces to a plain keyless hash with params.
    let mut oracle_256 = OracleBlake2bMac256::new_with_salt_and_personal(key_opt, &salt, &personal).unwrap();
    oracle_256.update(&data);
    let expected_256 = oracle_256.finalize().into_bytes();

    let ours_oneshot_256 = Blake2bParams::new()
      .key(&key)
      .salt(&salt)
      .personal(&personal)
      .hash_256(&data);
    prop_assert_eq!(&ours_oneshot_256[..], &expected_256[..]);

    // Streaming should match too.
    let mut ours_stream_256 = Blake2bParams::new()
      .key(&key)
      .salt(&salt)
      .personal(&personal)
      .build_256();
    ours_stream_256.update(&data);
    prop_assert_eq!(&ours_stream_256.finalize()[..], &expected_256[..]);

    let mut oracle_512 = OracleBlake2bMac512::new_with_salt_and_personal(key_opt, &salt, &personal).unwrap();
    oracle_512.update(&data);
    let expected_512 = oracle_512.finalize().into_bytes();

    let ours_oneshot_512 = Blake2bParams::new()
      .key(&key)
      .salt(&salt)
      .personal(&personal)
      .hash_512(&data);
    prop_assert_eq!(&ours_oneshot_512[..], &expected_512[..]);
  }

  #[test]
  fn blake2s_params_matches_rustcrypto(
    data in proptest::collection::vec(any::<u8>(), 0..2048),
    key_len in 0usize..=32,
    salt_len in 0usize..=8,
    personal_len in 0usize..=8,
  ) {
    let key = patterned_input(0x44, key_len);
    let salt = patterned_input(0x55, salt_len);
    let personal = patterned_input(0x66, personal_len);

    let key_opt: Option<&[u8]> = if key.is_empty() { None } else { Some(&key) };

    let mut oracle_256 = OracleBlake2sMac256::new_with_salt_and_personal(key_opt, &salt, &personal).unwrap();
    oracle_256.update(&data);
    let expected_256 = oracle_256.finalize().into_bytes();

    let ours_oneshot_256 = Blake2sParams::new()
      .key(&key)
      .salt(&salt)
      .personal(&personal)
      .hash_256(&data);
    prop_assert_eq!(&ours_oneshot_256[..], &expected_256[..]);

    let mut ours_stream_256 = Blake2sParams::new()
      .key(&key)
      .salt(&salt)
      .personal(&personal)
      .build_256();
    ours_stream_256.update(&data);
    prop_assert_eq!(&ours_stream_256.finalize()[..], &expected_256[..]);

    let mut oracle_128 = OracleBlake2sMac128::new_with_salt_and_personal(key_opt, &salt, &personal).unwrap();
    oracle_128.update(&data);
    let expected_128 = oracle_128.finalize().into_bytes();

    let ours_oneshot_128 = Blake2sParams::new()
      .key(&key)
      .salt(&salt)
      .personal(&personal)
      .hash_128(&data);
    prop_assert_eq!(&ours_oneshot_128[..], &expected_128[..]);
  }
}
