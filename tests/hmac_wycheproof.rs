#![cfg(feature = "hmac")]

use rscrypto::{HmacSha256, HmacSha384, HmacSha512, Mac};
use serde_json::Value;

mod common;
use common::decode_hex_vec;

const HMAC_SHA256: &str = include_str!("../testdata/auth/wycheproof/hmac_sha256_test.json");
const HMAC_SHA384: &str = include_str!("../testdata/auth/wycheproof/hmac_sha384_test.json");
const HMAC_SHA512: &str = include_str!("../testdata/auth/wycheproof/hmac_sha512_test.json");

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
  result.expect("HMAC Wycheproof result must be valid or invalid")
}

fn groups(suite: &Value) -> &[Value] {
  suite["testGroups"]
    .as_array()
    .expect("Wycheproof testGroups must be an array")
}

fn tests(group: &Value) -> &[Value] {
  group["tests"].as_array().expect("Wycheproof tests must be an array")
}

fn run_hmac_suite<M, const TAG_SIZE: usize>(
  suite_json: &str,
  algorithm: &str,
  full_tag_size_bits: u64,
  expected: Counts,
) where
  M: Mac,
  M::Tag: From<[u8; TAG_SIZE]>,
{
  let suite: Value = serde_json::from_str(suite_json).expect("Wycheproof JSON must parse");
  let mut counts = Counts { valid: 0, invalid: 0 };

  for group in groups(&suite) {
    let tag_size = group["tagSize"].as_u64().expect("tagSize must be numeric");
    if tag_size != full_tag_size_bits {
      continue;
    }

    for test in tests(group) {
      let tc_id = test["tcId"].as_u64().expect("tcId must be numeric");
      let key = decode_hex_vec(field(test, "key"));
      let msg = decode_hex_vec(field(test, "msg"));
      let tag = decode_hex_vec(field(test, "tag"));
      let tag_bytes: [u8; TAG_SIZE] = tag
        .try_into()
        .expect("full-tag HMAC Wycheproof vector must have the algorithm tag length");
      let tag = M::Tag::from(tag_bytes);

      match vector_result(test) {
        VectorResult::Valid => {
          counts.valid = counts.valid.strict_add(1);
          let actual = M::mac(&key, &msg);
          assert_eq!(
            <M::Tag as AsRef<[u8]>>::as_ref(&actual),
            <M::Tag as AsRef<[u8]>>::as_ref(&tag),
            "{algorithm} tcId {tc_id} MAC mismatch"
          );
          M::verify_tag(&key, &msg, &tag).expect("HMAC must verify a known-valid Wycheproof tag");
        }
        VectorResult::Invalid => {
          counts.invalid = counts.invalid.strict_add(1);
          M::verify_tag(&key, &msg, &tag).expect_err("HMAC must reject a known-invalid Wycheproof tag");
        }
      }
    }
  }

  assert_eq!(counts, expected, "{algorithm} Wycheproof coverage count changed");
}

#[test]
fn hmac_sha256_wycheproof_full_tag_vectors() {
  run_hmac_suite::<HmacSha256, 32>(HMAC_SHA256, "HMAC-SHA256", 256, Counts { valid: 33, invalid: 54 });
}

#[test]
fn hmac_sha384_wycheproof_full_tag_vectors() {
  run_hmac_suite::<HmacSha384, 48>(HMAC_SHA384, "HMAC-SHA384", 384, Counts { valid: 33, invalid: 54 });
}

#[test]
fn hmac_sha512_wycheproof_full_tag_vectors() {
  run_hmac_suite::<HmacSha512, 64>(HMAC_SHA512, "HMAC-SHA512", 512, Counts { valid: 33, invalid: 54 });
}
