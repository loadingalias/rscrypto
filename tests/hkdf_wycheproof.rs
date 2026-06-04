#![cfg(feature = "hkdf")]

use rscrypto::{HkdfSha256, HkdfSha384, auth::HkdfOutputLengthError};
use serde_json::Value;

mod common;
use common::decode_hex_vec;

const HKDF_SHA256: &str = include_str!("../testdata/auth/wycheproof/hkdf_sha256_test.json");
const HKDF_SHA384: &str = include_str!("../testdata/auth/wycheproof/hkdf_sha384_test.json");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Counts {
  valid: usize,
  invalid: usize,
}

type HkdfDerive = fn(&[u8], &[u8], &[u8], &mut [u8]) -> Result<(), HkdfOutputLengthError>;

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

fn run_hkdf_suite(suite_json: &str, algorithm: &str, derive: HkdfDerive, expected: Counts) {
  let suite: Value = serde_json::from_str(suite_json).expect("Wycheproof JSON must parse");
  let mut counts = Counts { valid: 0, invalid: 0 };

  for group in groups(&suite) {
    for test in tests(group) {
      let tc_id = test["tcId"].as_u64().expect("tcId must be numeric");
      let ikm = decode_hex_vec(field(test, "ikm"));
      let salt = decode_hex_vec(field(test, "salt"));
      let info = decode_hex_vec(field(test, "info"));
      let size = test["size"].as_u64().expect("size must be numeric") as usize;
      let mut okm = vec![0u8; size];

      match field(test, "result") {
        "valid" => {
          counts.valid += 1;
          let expected_okm = decode_hex_vec(field(test, "okm"));
          assert_eq!(expected_okm.len(), size, "{algorithm} tcId {tc_id} okm length mismatch");
          derive(&salt, &ikm, &info, &mut okm).unwrap_or_else(|err| panic!("{algorithm} tcId {tc_id} failed: {err}"));
          assert_eq!(okm, expected_okm, "{algorithm} tcId {tc_id} OKM mismatch");
        }
        "invalid" => {
          counts.invalid += 1;
          assert!(
            derive(&salt, &ikm, &info, &mut okm).is_err(),
            "{algorithm} tcId {tc_id} accepted invalid size"
          );
        }
        other => panic!("{algorithm} tcId {tc_id} has unsupported result `{other}`"),
      }
    }
  }

  assert_eq!(counts, expected, "{algorithm} Wycheproof coverage count changed");
}

#[test]
fn hkdf_sha256_wycheproof_vectors() {
  run_hkdf_suite(
    HKDF_SHA256,
    "HKDF-SHA256",
    HkdfSha256::derive,
    Counts { valid: 83, invalid: 3 },
  );
}

#[test]
fn hkdf_sha384_wycheproof_vectors() {
  run_hkdf_suite(
    HKDF_SHA384,
    "HKDF-SHA384",
    HkdfSha384::derive,
    Counts { valid: 80, invalid: 3 },
  );
}
