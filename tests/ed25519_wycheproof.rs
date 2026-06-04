#![cfg(feature = "ed25519")]

use rscrypto::{Ed25519PublicKey, Ed25519Signature};
use serde_json::Value;

mod common;
use common::{decode_hex_array, decode_hex_vec};

const ED25519: &str = include_str!("../testdata/auth/wycheproof/ed25519_test.json");

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

      match field(test, "result") {
        "valid" => {
          counts.valid = counts.valid.strict_add(1);
          let signature = Ed25519Signature::from_bytes(
            signature
              .try_into()
              .expect("valid Wycheproof Ed25519 signatures must be 64 bytes"),
          );
          assert!(
            public.verify(&message, &signature).is_ok(),
            "Wycheproof Ed25519 tcId {} rejected a valid signature",
            test["tcId"]
          );
        }
        "invalid" => {
          counts.invalid = counts.invalid.strict_add(1);
          if signature.len() == Ed25519Signature::LENGTH {
            let signature = Ed25519Signature::from_bytes(signature.try_into().unwrap());
            assert!(
              public.verify(&message, &signature).is_err(),
              "Wycheproof Ed25519 tcId {} accepted an invalid signature: {}",
              test["tcId"],
              field(test, "comment")
            );
          }
        }
        other => panic!("unsupported Wycheproof Ed25519 result `{other}`"),
      }
    }
  }

  assert_eq!(counts, Counts { valid: 88, invalid: 62 });
}
