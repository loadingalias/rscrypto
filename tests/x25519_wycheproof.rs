#![cfg(feature = "x25519")]

use rscrypto::{X25519Error, X25519PublicKey, X25519SecretKey};
use serde_json::Value;

mod common;
use common::decode_hex_array;

const X25519: &str = include_str!("../testdata/auth/wycheproof/x25519_test.json");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Counts {
  valid: usize,
  acceptable: usize,
  zero_shared_rejected: usize,
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
fn wycheproof_x25519_vectors_match_or_reject_all_zero_shared_secret() {
  let suite: Value = serde_json::from_str(X25519).expect("Wycheproof X25519 JSON must parse");
  assert_eq!(suite["algorithm"].as_str(), Some("XDH"));

  let mut counts = Counts {
    valid: 0,
    acceptable: 0,
    zero_shared_rejected: 0,
  };

  for group in groups(&suite) {
    assert_eq!(group["type"].as_str(), Some("XdhComp"));
    assert_eq!(group["curve"].as_str(), Some("curve25519"));

    for test in tests(group) {
      let secret = X25519SecretKey::from_bytes(decode_hex_array(field(test, "private")));
      let public = X25519PublicKey::from_bytes(decode_hex_array(field(test, "public")));
      let expected_shared: [u8; 32] = decode_hex_array(field(test, "shared"));
      let result = secret.diffie_hellman(&public);

      match field(test, "result") {
        "valid" => counts.valid = counts.valid.strict_add(1),
        "acceptable" => counts.acceptable = counts.acceptable.strict_add(1),
        other => panic!("unsupported Wycheproof X25519 result `{other}`"),
      }

      if expected_shared == [0u8; 32] {
        counts.zero_shared_rejected = counts.zero_shared_rejected.strict_add(1);
        assert_eq!(
          result,
          Err(X25519Error::new()),
          "Wycheproof X25519 tcId {} must reject all-zero shared secret",
          test["tcId"]
        );
      } else {
        let shared = result.unwrap_or_else(|error| {
          panic!(
            "Wycheproof X25519 tcId {} unexpectedly rejected non-zero shared secret: {error}",
            test["tcId"]
          )
        });
        assert_eq!(
          *shared.as_bytes(),
          expected_shared,
          "Wycheproof X25519 tcId {} shared secret mismatch",
          test["tcId"]
        );
      }
    }
  }

  assert_eq!(
    counts,
    Counts {
      valid: 264,
      acceptable: 254,
      zero_shared_rejected: 31,
    }
  );
}
