#![cfg(feature = "ed25519")]

use rscrypto::{Ed25519PublicKey, Ed25519Signature};
use serde_json::Value;

mod common;
#[path = "common/array.rs"]
mod hex_array;
use common::decode_hex_vec;
use hex_array::decode_hex_array;

const ED25519: &str = include_str!("../testdata/auth/wycheproof/ed25519_test.json");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Counts {
  valid: usize,
  invalid: usize,
}

fn field<'a>(value: &'a Value, name: &str) -> &'a str {
  value
    .get(name)
    .expect("Wycheproof field must be present")
    .as_str()
    .expect("Wycheproof field must be a string")
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
fn wycheproof_ed25519_verify_vectors_match_expected_results() {
  let suite: Value = serde_json::from_str(ED25519).expect("Wycheproof Ed25519 JSON must parse");
  assert_eq!(suite["algorithm"].as_str(), Some("EDDSA"));

  let mut counts = Counts { valid: 0, invalid: 0 };
  for group in groups(&suite) {
    assert_eq!(group["type"].as_str(), Some("EddsaVerify"));
    assert_eq!(group["publicKey"]["curve"].as_str(), Some("edwards25519"));
    let public = Ed25519PublicKey::from_bytes(decode_hex_array(field(&group["publicKey"], "pk")));

    for test in tests(group) {
      let message = decode_hex_vec(field(test, "msg"));
      let signature = decode_hex_vec(field(test, "sig"));

      let disposition = field(test, "result");
      assert!(
        matches!(disposition, "valid" | "invalid"),
        "unsupported Wycheproof Ed25519 result `{disposition}`"
      );
      match disposition {
        "valid" => {
          counts.valid = counts.valid.strict_add(1);
          let signature = Ed25519Signature::from_bytes(
            signature
              .try_into()
              .expect("valid Wycheproof Ed25519 signatures must be 64 bytes"),
          );
          public
            .verify(&message, &signature)
            .expect("Wycheproof valid Ed25519 signature must verify");
        }
        "invalid" => {
          counts.invalid = counts.invalid.strict_add(1);
          if signature.len() == Ed25519Signature::LENGTH {
            let signature = Ed25519Signature::from_bytes(
              signature
                .try_into()
                .expect("length-checked Wycheproof Ed25519 signature must fit an array"),
            );
            public
              .verify(&message, &signature)
              .expect_err("Wycheproof invalid Ed25519 signature must be rejected");
          }
        }
        _ => {}
      }
    }
  }

  assert_eq!(counts, Counts { valid: 88, invalid: 62 });
}
