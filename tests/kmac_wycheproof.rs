#![cfg(feature = "kmac")]

use rscrypto::Kmac256;
use serde_json::Value;

mod common;
use common::decode_hex_vec;

const KMAC256_NO_CUSTOMIZATION: &str = include_str!("../testdata/auth/wycheproof/kmac256_no_customization_test.json");

#[derive(Debug, PartialEq, Eq)]
struct Counts {
  valid: usize,
  invalid: usize,
}

fn field<'a>(value: &'a Value, name: &str) -> &'a str {
  value
    .get(name)
    .and_then(Value::as_str)
    .expect("Wycheproof string field must exist and contain a string")
}

enum VectorResult {
  Valid,
  Invalid,
}

fn vector_result(test: &Value) -> VectorResult {
  let result = match field(test, "result") {
    "valid" => Some(VectorResult::Valid),
    "invalid" => Some(VectorResult::Invalid),
    _ => None,
  };
  result.expect("KMAC Wycheproof result must be valid or invalid")
}

fn groups(suite: &Value) -> &[Value] {
  suite["testGroups"]
    .as_array()
    .expect("Wycheproof testGroups must be an array")
}

fn tests(group: &Value) -> &[Value] {
  group["tests"].as_array().expect("Wycheproof tests must be an array")
}

#[test]
fn kmac256_no_customization_wycheproof_vectors() {
  let suite: Value = serde_json::from_str(KMAC256_NO_CUSTOMIZATION).expect("Wycheproof JSON must parse");
  let mut counts = Counts { valid: 0, invalid: 0 };

  for group in groups(&suite) {
    for test in tests(group) {
      let tc_id = test["tcId"].as_u64().expect("tcId must be numeric");
      let key = decode_hex_vec(field(test, "key"));
      let msg = decode_hex_vec(field(test, "msg"));
      let tag = decode_hex_vec(field(test, "tag"));

      match vector_result(test) {
        VectorResult::Valid => {
          counts.valid = counts.valid.strict_add(1);
          let mut actual = vec![0u8; tag.len()];
          Kmac256::mac_into(&key, b"", &msg, &mut actual);
          assert_eq!(actual, tag, "KMAC256 tcId {tc_id} tag mismatch");
          Kmac256::verify_tag_primitive(&key, b"", &msg, &tag)
            .expect("KMAC256 must verify a known-valid Wycheproof tag");
        }
        VectorResult::Invalid => {
          counts.invalid = counts.invalid.strict_add(1);
          Kmac256::verify_tag_primitive(&key, b"", &msg, &tag)
            .expect_err("KMAC256 must reject a known-invalid Wycheproof tag");
        }
      }
    }
  }

  assert_eq!(
    counts,
    Counts {
      valid: 99,
      invalid: 162,
    },
    "KMAC256 Wycheproof coverage count changed"
  );
}

#[test]
fn kmac256_rejects_empty_verification_tag() {
  Kmac256::verify_tag(b"key", b"", b"message", b"").expect_err("KMAC256 must reject an empty verification tag");
}
