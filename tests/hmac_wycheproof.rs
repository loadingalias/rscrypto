#![cfg(feature = "hmac")]

use rscrypto::{HmacSha256, HmacSha384, HmacSha512, Mac};
use serde_json::Value;

mod common;
use common::decode_hex_vec;

const HMAC_SHA256: &str = include_str!("../testdata/auth/wycheproof/hmac_sha256_test.json");
const HMAC_SHA384: &str = include_str!("../testdata/auth/wycheproof/hmac_sha384_test.json");
const HMAC_SHA512: &str = include_str!("../testdata/auth/wycheproof/hmac_sha512_test.json");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Counts {
  valid: usize,
  invalid: usize,
}

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

fn run_hmac_suite<M, const TAG_SIZE: usize>(
  suite_json: &str,
  algorithm: &str,
  full_tag_size_bits: u64,
  expected: Counts,
) where
  M: Mac<Tag = [u8; TAG_SIZE]>,
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
      let tag: [u8; TAG_SIZE] = match tag.try_into() {
        Ok(tag) => tag,
        Err(tag) => panic!("{algorithm} tcId {tc_id} tag has wrong length: {}", tag.len()),
      };

      match field(test, "result") {
        "valid" => {
          counts.valid += 1;
          let actual = M::mac(&key, &msg);
          assert_eq!(actual.as_ref(), tag.as_slice(), "{algorithm} tcId {tc_id} MAC mismatch");
          assert!(
            M::verify_tag(&key, &msg, &tag).is_ok(),
            "{algorithm} tcId {tc_id} verify failed"
          );
        }
        "invalid" => {
          counts.invalid += 1;
          assert!(
            M::verify_tag(&key, &msg, &tag).is_err(),
            "{algorithm} tcId {tc_id} accepted an invalid tag"
          );
        }
        other => panic!("{algorithm} tcId {tc_id} has unsupported result `{other}`"),
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
