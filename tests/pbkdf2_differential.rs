#![cfg(feature = "pbkdf2")]

use proptest::{prelude::*, test_runner::Config as ProptestConfig};
use rscrypto::{Pbkdf2Sha256, Pbkdf2Sha512};

struct Pbkdf2Sha256Case {
  password: &'static [u8],
  salt: &'static [u8],
  iterations: u32,
  expected: [u8; 64],
}

fn oracle_sha256(password: &[u8], salt: &[u8], iterations: u32, out: &mut [u8]) {
  pbkdf2::pbkdf2_hmac::<sha2_010::Sha256>(password, salt, iterations, out);
}

fn oracle_sha512(password: &[u8], salt: &[u8], iterations: u32, out: &mut [u8]) {
  pbkdf2::pbkdf2_hmac::<sha2_010::Sha512>(password, salt, iterations, out);
}

proptest! {
  #![proptest_config(ProptestConfig::with_cases(64))]

  #[test]
  fn pbkdf2_sha256_matches_rustcrypto_oracle_and_reuse(
    password in proptest::collection::vec(any::<u8>(), 0..192),
    salt in proptest::collection::vec(any::<u8>(), 0..256),
    iterations in 1u32..=64,
    out_len in 1usize..=128,
  ) {
    let mut expected = vec![0u8; out_len];
    oracle_sha256(&password, &salt, iterations, &mut expected);

    let mut oneshot = vec![0u8; out_len];
    Pbkdf2Sha256::derive_key(&password, &salt, iterations, &mut oneshot).unwrap();

    let state = Pbkdf2Sha256::new(&password);
    let mut reused = vec![0u8; out_len];
    state.derive(&salt, iterations, &mut reused).unwrap();

    prop_assert_eq!(oneshot, expected.as_slice());
    prop_assert_eq!(reused, expected.as_slice());
    prop_assert!(state.verify(&salt, iterations, &expected).is_ok());
    prop_assert!(Pbkdf2Sha256::verify_password(&password, &salt, iterations, &expected).is_ok());

    let mut wrong = expected.clone();
    wrong[0] ^= 1;
    prop_assert!(state.verify(&salt, iterations, &wrong).is_err());
    prop_assert!(Pbkdf2Sha256::verify_password(&password, &salt, iterations, &wrong).is_err());
  }

  #[test]
  fn pbkdf2_sha512_matches_rustcrypto_oracle_and_reuse(
    password in proptest::collection::vec(any::<u8>(), 0..256),
    salt in proptest::collection::vec(any::<u8>(), 0..320),
    iterations in 1u32..=64,
    out_len in 1usize..=192,
  ) {
    let mut expected = vec![0u8; out_len];
    oracle_sha512(&password, &salt, iterations, &mut expected);

    let mut oneshot = vec![0u8; out_len];
    Pbkdf2Sha512::derive_key(&password, &salt, iterations, &mut oneshot).unwrap();

    let state = Pbkdf2Sha512::new(&password);
    let mut reused = vec![0u8; out_len];
    state.derive(&salt, iterations, &mut reused).unwrap();

    prop_assert_eq!(oneshot, expected.as_slice());
    prop_assert_eq!(reused, expected.as_slice());
    prop_assert!(state.verify(&salt, iterations, &expected).is_ok());
    prop_assert!(Pbkdf2Sha512::verify_password(&password, &salt, iterations, &expected).is_ok());

    let mut wrong = expected.clone();
    wrong[0] ^= 1;
    prop_assert!(state.verify(&salt, iterations, &wrong).is_err());
    prop_assert!(Pbkdf2Sha512::verify_password(&password, &salt, iterations, &wrong).is_err());
  }
}

#[test]
fn pbkdf2_sha256_rfc7914_vectors_match_rustcrypto() {
  let cases = [
    Pbkdf2Sha256Case {
      password: b"passwd",
      salt: b"salt",
      iterations: 1,
      expected: [
        0x55, 0xac, 0x04, 0x6e, 0x56, 0xe3, 0x08, 0x9f, 0xec, 0x16, 0x91, 0xc2, 0x25, 0x44, 0xb6, 0x05, 0xf9, 0x41,
        0x85, 0x21, 0x6d, 0xde, 0x04, 0x65, 0xe6, 0x8b, 0x9d, 0x57, 0xc2, 0x0d, 0xac, 0xbc, 0x49, 0xca, 0x9c, 0xcc,
        0xf1, 0x79, 0xb6, 0x45, 0x99, 0x16, 0x64, 0xb3, 0x9d, 0x77, 0xef, 0x31, 0x7c, 0x71, 0xb8, 0x45, 0xb1, 0xe3,
        0x0b, 0xd5, 0x09, 0x11, 0x20, 0x41, 0xd3, 0xa1, 0x97, 0x83,
      ],
    },
    Pbkdf2Sha256Case {
      password: b"Password",
      salt: b"NaCl",
      iterations: 80_000,
      expected: [
        0x4d, 0xdc, 0xd8, 0xf6, 0x0b, 0x98, 0xbe, 0x21, 0x83, 0x0c, 0xee, 0x5e, 0xf2, 0x27, 0x01, 0xf9, 0x64, 0x1a,
        0x44, 0x18, 0xd0, 0x4c, 0x04, 0x14, 0xae, 0xff, 0x08, 0x87, 0x6b, 0x34, 0xab, 0x56, 0xa1, 0xd4, 0x25, 0xa1,
        0x22, 0x58, 0x33, 0x54, 0x9a, 0xdb, 0x84, 0x1b, 0x51, 0xc9, 0xb3, 0x17, 0x6a, 0x27, 0x2b, 0xde, 0xbb, 0xa1,
        0xd0, 0x78, 0x47, 0x8f, 0x62, 0xb3, 0x97, 0xf3, 0x3c, 0x8d,
      ],
    },
  ];

  for case in cases {
    let actual = Pbkdf2Sha256::derive_key_array::<64>(case.password, case.salt, case.iterations).unwrap();
    assert_eq!(actual, case.expected);

    let mut oracle = [0u8; 64];
    oracle_sha256(case.password, case.salt, case.iterations, &mut oracle);
    assert_eq!(actual, oracle);
  }
}
