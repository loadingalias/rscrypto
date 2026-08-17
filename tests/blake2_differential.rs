#![cfg(feature = "hashes")]

use blake2::{
  Blake2b as OracleBlake2b, Blake2b512 as OracleBlake2b512, Blake2bMac, Blake2bVarCore, Blake2s as OracleBlake2s,
  Blake2s256 as OracleBlake2s256, Blake2sMac, Blake2sVarCore,
  digest::{
    Digest as _, Mac as _, Output,
    consts::{U16, U32, U64},
    core_api::{Buffer, UpdateCore, VariableOutputCore},
  },
};
use proptest::{prelude::*, test_runner::Config as ProptestConfig};
use rscrypto::{
  Blake2b256, Blake2b512, Blake2bKey, Blake2bParams, Blake2s128, Blake2s256, Blake2sKey, Blake2sParams, Digest,
};

type OracleBlake2bMac256 = Blake2bMac<U32>;
type OracleBlake2bMac512 = Blake2bMac<U64>;
type OracleBlake2sMac128 = Blake2sMac<U16>;
type OracleBlake2sMac256 = Blake2sMac<U32>;
type OracleBlake2b256 = OracleBlake2b<U32>;
type OracleBlake2s128 = OracleBlake2s<U16>;

fn split_at_ratio(data: &[u8], ratio: u8) -> (&[u8], &[u8]) {
  let idx = if data.is_empty() {
    0
  } else {
    data.len().strict_mul(usize::from(ratio)) / 255
  };
  data.split_at(idx.min(data.len()))
}

fn patterned_input(seed: u8, len: usize) -> Vec<u8> {
  (0..len)
    .map(|i| {
      let offset = u8::try_from(i % 251).expect("remainder modulo 251 must fit in one byte");
      seed.wrapping_add(offset)
    })
    .collect()
}

fn oracle_blake2b_unkeyed<const N: usize>(data: &[u8], salt: &[u8], personal: &[u8]) -> [u8; N] {
  let mut core = Blake2bVarCore::new_with_params(salt, personal, 0, N);
  let mut buffer = Buffer::<Blake2bVarCore>::default();
  buffer.digest_blocks(data, |blocks| core.update_blocks(blocks));
  let mut full = Output::<Blake2bVarCore>::default();
  core.finalize_variable_core(&mut buffer, &mut full);
  let mut out = [0u8; N];
  out.copy_from_slice(&full[..N]);
  out
}

fn oracle_blake2s_unkeyed<const N: usize>(data: &[u8], salt: &[u8], personal: &[u8]) -> [u8; N] {
  let mut core = Blake2sVarCore::new_with_params(salt, personal, 0, N);
  let mut buffer = Buffer::<Blake2sVarCore>::default();
  buffer.digest_blocks(data, |blocks| core.update_blocks(blocks));
  let mut full = Output::<Blake2sVarCore>::default();
  core.finalize_variable_core(&mut buffer, &mut full);
  let mut out = [0u8; N];
  out.copy_from_slice(&full[..N]);
  out
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
    let typed_key = Blake2bKey::new(key).expect("generated BLAKE2b key length must be valid");
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

    let mut oracle_keyed_256 = OracleBlake2bMac256::new_from_slice(key)
      .expect("RustCrypto must accept the generated BLAKE2b-256 key");
    oracle_keyed_256.update(&data);
    let expected_keyed_256 = oracle_keyed_256.finalize().into_bytes();
    prop_assert_eq!(
      &Blake2b256::keyed_digest(typed_key, &data)[..],
      &expected_keyed_256[..]
    );

    let mut oracle_keyed_512 = OracleBlake2bMac512::new_from_slice(key)
      .expect("RustCrypto must accept the generated BLAKE2b-512 key");
    oracle_keyed_512.update(&data);
    let expected_keyed_512 = oracle_keyed_512.finalize().into_bytes();
    prop_assert_eq!(
      &Blake2b512::keyed_digest(typed_key, &data)[..],
      &expected_keyed_512[..]
    );

    let mut reset_256 = Blake2b256::new();
    reset_256.update(&data);
    let _first_digest = reset_256.finalize();
    reset_256.reset();
    reset_256.update(&tail);
    prop_assert_eq!(reset_256.finalize(), Blake2b256::digest(&tail));

    let mut reset_512 = Blake2b512::new();
    reset_512.update(&data);
    let _first_digest = reset_512.finalize();
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
    let typed_key = Blake2sKey::new(key).expect("generated BLAKE2s key length must be valid");
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

    let mut oracle_keyed_128 = OracleBlake2sMac128::new_from_slice(key)
      .expect("RustCrypto must accept the generated BLAKE2s-128 key");
    oracle_keyed_128.update(&data);
    let expected_keyed_128 = oracle_keyed_128.finalize().into_bytes();
    prop_assert_eq!(
      &Blake2s128::keyed_digest(typed_key, &data)[..],
      &expected_keyed_128[..]
    );

    let mut oracle_keyed_256 = OracleBlake2sMac256::new_from_slice(key)
      .expect("RustCrypto must accept the generated BLAKE2s-256 key");
    oracle_keyed_256.update(&data);
    let expected_keyed_256 = oracle_keyed_256.finalize().into_bytes();
    prop_assert_eq!(
      &Blake2s256::keyed_digest(typed_key, &data)[..],
      &expected_keyed_256[..]
    );

    let mut reset_128 = Blake2s128::new();
    reset_128.update(&data);
    let _first_digest = reset_128.finalize();
    reset_128.reset();
    reset_128.update(&tail);
    prop_assert_eq!(reset_128.finalize(), Blake2s128::digest(&tail));

    let mut reset_256 = Blake2s256::new();
    reset_256.update(&data);
    let _first_digest = reset_256.finalize();
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
    let mut salt_field = [0u8; 16];
    salt_field[..salt.len()].copy_from_slice(&salt);
    let mut personal_field = [0u8; 16];
    personal_field[..personal.len()].copy_from_slice(&personal);

    let expected_256: [u8; 32] = if key.is_empty() {
      oracle_blake2b_unkeyed(&data, &salt, &personal)
    } else {
      let mut oracle = OracleBlake2bMac256::new_with_salt_and_personal(&key, &salt, &personal)
        .expect("RustCrypto must accept generated BLAKE2b-256 parameters");
      oracle.update(&data);
      oracle.finalize().into_bytes().into()
    };

    let mut params = Blake2bParams::new().salt(salt_field).personal(personal_field);
    if !key.is_empty() {
      params = params.key(Blake2bKey::new(&key).expect("generated BLAKE2b parameter key must be valid"));
    }
    let ours_oneshot_256 = params.hash_256(&data);
    prop_assert_eq!(&ours_oneshot_256[..], &expected_256[..]);

    // Streaming should match too.
    let mut ours_stream_256 = params.build_256();
    ours_stream_256.update(&data);
    prop_assert_eq!(&ours_stream_256.finalize()[..], &expected_256[..]);

    let expected_512: [u8; 64] = if key.is_empty() {
      oracle_blake2b_unkeyed(&data, &salt, &personal)
    } else {
      let mut oracle = OracleBlake2bMac512::new_with_salt_and_personal(&key, &salt, &personal)
        .expect("RustCrypto must accept generated BLAKE2b-512 parameters");
      oracle.update(&data);
      oracle.finalize().into_bytes().into()
    };

    let ours_oneshot_512 = params.hash_512(&data);
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
    let mut salt_field = [0u8; 8];
    salt_field[..salt.len()].copy_from_slice(&salt);
    let mut personal_field = [0u8; 8];
    personal_field[..personal.len()].copy_from_slice(&personal);

    let expected_256: [u8; 32] = if key.is_empty() {
      oracle_blake2s_unkeyed(&data, &salt, &personal)
    } else {
      let mut oracle = OracleBlake2sMac256::new_with_salt_and_personal(&key, &salt, &personal)
        .expect("RustCrypto must accept generated BLAKE2s-256 parameters");
      oracle.update(&data);
      oracle.finalize().into_bytes().into()
    };

    let mut params = Blake2sParams::new().salt(salt_field).personal(personal_field);
    if !key.is_empty() {
      params = params.key(Blake2sKey::new(&key).expect("generated BLAKE2s parameter key must be valid"));
    }
    let ours_oneshot_256 = params.hash_256(&data);
    prop_assert_eq!(&ours_oneshot_256[..], &expected_256[..]);

    let mut ours_stream_256 = params.build_256();
    ours_stream_256.update(&data);
    prop_assert_eq!(&ours_stream_256.finalize()[..], &expected_256[..]);

    let expected_128: [u8; 16] = if key.is_empty() {
      oracle_blake2s_unkeyed(&data, &salt, &personal)
    } else {
      let mut oracle = OracleBlake2sMac128::new_with_salt_and_personal(&key, &salt, &personal)
        .expect("RustCrypto must accept generated BLAKE2s-128 parameters");
      oracle.update(&data);
      oracle.finalize().into_bytes().into()
    };

    let ours_oneshot_128 = params.hash_128(&data);
    prop_assert_eq!(&ours_oneshot_128[..], &expected_128[..]);
  }
}
