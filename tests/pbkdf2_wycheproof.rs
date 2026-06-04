#![cfg(feature = "pbkdf2")]

use rscrypto::{Pbkdf2Sha256, Pbkdf2Sha512, auth::Pbkdf2Error};
use serde_json::Value;

mod common;
use common::decode_hex_vec;

const PBKDF2_HMAC_SHA256: &str = include_str!("../testdata/auth/wycheproof/pbkdf2_hmacsha256_test.json");
const PBKDF2_HMAC_SHA512: &str = include_str!("../testdata/auth/wycheproof/pbkdf2_hmacsha512_test.json");

type Pbkdf2Derive = fn(&[u8], &[u8], u32, &mut [u8]) -> Result<(), Pbkdf2Error>;
type Pbkdf2Verify = fn(&[u8], &[u8], u32, &[u8]) -> Result<(), rscrypto::VerificationError>;

fn field<'a>(value: &'a Value, name: &str) -> &'a str {
  value[name]
    .as_str()
    .unwrap_or_else(|| panic!("missing string field `{name}`"))
}

fn groups(suite: &Value) -> &[Value] {
  suite["testGroups"]
    .as_array()
    .expect("Wycheproof testGroups must be an array")
}

fn tests(group: &Value) -> &[Value] {
  group["tests"].as_array().expect("Wycheproof tests must be an array")
}

fn run_pbkdf2_suite(
  suite_json: &str,
  algorithm: &str,
  derive: Pbkdf2Derive,
  verify: Pbkdf2Verify,
  expected_valid: usize,
) {
  let suite: Value = serde_json::from_str(suite_json).expect("Wycheproof JSON must parse");
  let mut valid = 0usize;
  let mut first_valid_for_negative_check = None;

  for group in groups(&suite) {
    for test in tests(group) {
      let tc_id = test["tcId"].as_u64().expect("tcId must be numeric");
      let password = decode_hex_vec(field(test, "password"));
      let salt = decode_hex_vec(field(test, "salt"));
      let iterations = test["iterationCount"].as_u64().expect("iterationCount must be numeric") as u32;
      let dk_len = test["dkLen"].as_u64().expect("dkLen must be numeric") as usize;
      let expected_dk = decode_hex_vec(field(test, "dk"));
      assert_eq!(expected_dk.len(), dk_len, "{algorithm} tcId {tc_id} dkLen mismatch");

      match field(test, "result") {
        "valid" => {
          valid += 1;
          let mut actual = vec![0u8; dk_len];
          derive(&password, &salt, iterations, &mut actual)
            .unwrap_or_else(|err| panic!("{algorithm} tcId {tc_id} failed: {err}"));
          assert_eq!(actual, expected_dk, "{algorithm} tcId {tc_id} derived key mismatch");

          if first_valid_for_negative_check.is_none() {
            first_valid_for_negative_check = Some((password, salt, iterations, expected_dk));
          }
        }
        other => panic!("{algorithm} tcId {tc_id} has unsupported result `{other}`"),
      }
    }
  }

  assert_eq!(valid, expected_valid, "{algorithm} Wycheproof coverage count changed");

  let (password, salt, iterations, expected_dk) =
    first_valid_for_negative_check.expect("PBKDF2 suite must contain at least one valid vector");
  assert!(
    verify(&password, &salt, iterations, &expected_dk).is_ok(),
    "{algorithm} rejected a known-good derived key"
  );

  let mut wrong_dk = expected_dk.clone();
  wrong_dk[0] ^= 1;
  assert!(
    verify(&password, &salt, iterations, &wrong_dk).is_err(),
    "{algorithm} accepted a corrupted derived key"
  );
  assert!(
    verify(b"wrong password", &salt, iterations, &expected_dk).is_err(),
    "{algorithm} accepted the wrong password"
  );
}

#[test]
fn pbkdf2_hmac_sha256_wycheproof_vectors() {
  run_pbkdf2_suite(
    PBKDF2_HMAC_SHA256,
    "PBKDF2-HMAC-SHA256",
    Pbkdf2Sha256::derive_key,
    Pbkdf2Sha256::verify_password,
    60,
  );
}

#[test]
fn pbkdf2_hmac_sha512_wycheproof_vectors() {
  run_pbkdf2_suite(
    PBKDF2_HMAC_SHA512,
    "PBKDF2-HMAC-SHA512",
    Pbkdf2Sha512::derive_key,
    Pbkdf2Sha512::verify_password,
    58,
  );
}
