#![cfg(feature = "rsa")]

use rscrypto::{RsaPkcs1v15Profile, RsaPssProfile, RsaPublicKey, RsaPublicKeyPolicy};
use serde_json::Value;

const PKCS1_SHA256: &str = include_str!("../testdata/rsa/wycheproof/rsa_signature_2048_sha256_test.json");
const PKCS1_SHA384: &str = include_str!("../testdata/rsa/wycheproof/rsa_signature_2048_sha384_test.json");
const PKCS1_SHA512: &str = include_str!("../testdata/rsa/wycheproof/rsa_signature_2048_sha512_test.json");
const PSS_SHA256: &str = include_str!("../testdata/rsa/wycheproof/rsa_pss_2048_sha256_mgf1_32_test.json");
const PSS_SHA384: &str = include_str!("../testdata/rsa/wycheproof/rsa_pss_2048_sha384_mgf1_48_test.json");
const PSS_SHA512: &str = include_str!("../testdata/rsa/wycheproof/rsa_pss_4096_sha512_mgf1_64_test.json");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExpectedCounts {
  valid: usize,
  acceptable: usize,
  invalid: usize,
}

fn hex_to_vec(hex: &str) -> Vec<u8> {
  assert_eq!(hex.len() % 2, 0);
  let mut out = Vec::with_capacity(hex.len() / 2);
  for chunk in hex.as_bytes().chunks_exact(2) {
    out.push((hex_value(chunk[0]) << 4) | hex_value(chunk[1]));
  }
  out
}

fn hex_value(byte: u8) -> u8 {
  match byte {
    b'0'..=b'9' => byte - b'0',
    b'a'..=b'f' => byte - b'a' + 10,
    b'A'..=b'F' => byte - b'A' + 10,
    _ => panic!("invalid hex digit"),
  }
}

fn groups<'a>(suite: &'a Value, expected_algorithm: &str) -> &'a [Value] {
  assert_eq!(suite["algorithm"].as_str(), Some(expected_algorithm));
  suite["testGroups"]
    .as_array()
    .expect("Wycheproof testGroups must be an array")
}

fn test_cases(group: &Value) -> &[Value] {
  group["tests"].as_array().expect("Wycheproof tests must be an array")
}

fn field<'a>(value: &'a Value, name: &str) -> &'a str {
  value[name]
    .as_str()
    .unwrap_or_else(|| panic!("missing string field `{name}`"))
}

fn assert_pkcs1v15_wycheproof_vectors(
  json: &str,
  profile: RsaPkcs1v15Profile,
  expected_sha: &str,
  expected: ExpectedCounts,
) {
  let suite: Value = serde_json::from_str(json).expect("Wycheproof PKCS1v1.5 JSON must parse");
  let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_small_exponents();
  let mut valid = 0usize;
  let mut invalid = 0usize;
  let mut acceptable = 0usize;

  for group in groups(&suite, "RSASSA-PKCS1-v1_5") {
    assert_eq!(group["sha"].as_str(), Some(expected_sha));
    let public_key = hex_to_vec(field(group, "publicKeyDer"));
    let key = RsaPublicKey::from_spki_der_with_policy(&public_key, &policy).expect("Wycheproof RSA key must parse");
    let mut scratch = key.public_scratch();

    for test in test_cases(group) {
      let msg = hex_to_vec(field(test, "msg"));
      let sig = hex_to_vec(field(test, "sig"));
      let verified = key
        .verify_pkcs1v15_with_scratch(profile, &msg, &sig, &mut scratch)
        .is_ok();

      match field(test, "result") {
        "valid" => {
          valid = valid.strict_add(1);
          assert!(
            verified,
            "Wycheproof PKCS1v1.5 tcId {} rejected valid signature",
            test["tcId"]
          );
        }
        "invalid" => {
          invalid = invalid.strict_add(1);
          assert!(
            !verified,
            "Wycheproof PKCS1v1.5 tcId {} accepted invalid signature: {}",
            test["tcId"],
            field(test, "comment")
          );
        }
        "acceptable" => {
          acceptable = acceptable.strict_add(1);
        }
        other => panic!("unknown Wycheproof result `{other}`"),
      }
    }
  }

  assert_eq!(
    ExpectedCounts {
      valid,
      acceptable,
      invalid
    },
    expected
  );
}

fn assert_pss_wycheproof_vectors(json: &str, profile: RsaPssProfile, expected_sha: &str, expected: ExpectedCounts) {
  let suite: Value = serde_json::from_str(json).expect("Wycheproof PSS JSON must parse");
  let mut valid = 0usize;
  let mut invalid = 0usize;
  let mut acceptable = 0usize;

  for group in groups(&suite, "RSASSA-PSS") {
    assert_eq!(group["sha"].as_str(), Some(expected_sha));
    assert_eq!(group["mgf"].as_str(), Some("MGF1"));
    assert_eq!(group["mgfSha"].as_str(), Some(expected_sha));
    assert_eq!(group["sLen"].as_u64(), Some(profile.digest_len() as u64));
    let public_key = hex_to_vec(field(group, "publicKeyDer"));
    let key = RsaPublicKey::from_spki_der(&public_key).expect("Wycheproof RSA-PSS key must parse");
    let mut scratch = key.public_scratch();

    for test in test_cases(group) {
      let msg = hex_to_vec(field(test, "msg"));
      let sig = hex_to_vec(field(test, "sig"));
      let verified = key.verify_pss_with_scratch(profile, &msg, &sig, &mut scratch).is_ok();

      match field(test, "result") {
        "valid" => {
          valid = valid.strict_add(1);
          assert!(
            verified,
            "Wycheproof PSS tcId {} rejected valid signature",
            test["tcId"]
          );
        }
        "invalid" => {
          invalid = invalid.strict_add(1);
          assert!(
            !verified,
            "Wycheproof PSS tcId {} accepted invalid signature: {}",
            test["tcId"],
            field(test, "comment")
          );
        }
        "acceptable" => {
          acceptable = acceptable.strict_add(1);
        }
        other => panic!("unknown Wycheproof result `{other}`"),
      }
    }
  }

  assert_eq!(
    ExpectedCounts {
      valid,
      acceptable,
      invalid
    },
    expected
  );
}

#[test]
fn wycheproof_pkcs1v15_sha2_vectors_match_expected_results() {
  assert_pkcs1v15_wycheproof_vectors(
    PKCS1_SHA256,
    RsaPkcs1v15Profile::Sha256,
    "SHA-256",
    ExpectedCounts {
      valid: 9,
      acceptable: 1,
      invalid: 249,
    },
  );
  assert_pkcs1v15_wycheproof_vectors(
    PKCS1_SHA384,
    RsaPkcs1v15Profile::Sha384,
    "SHA-384",
    ExpectedCounts {
      valid: 7,
      acceptable: 1,
      invalid: 250,
    },
  );
  assert_pkcs1v15_wycheproof_vectors(
    PKCS1_SHA512,
    RsaPkcs1v15Profile::Sha512,
    "SHA-512",
    ExpectedCounts {
      valid: 8,
      acceptable: 1,
      invalid: 250,
    },
  );
}

#[test]
fn wycheproof_pss_sha2_mgf1_digest_salt_vectors_match_expected_results() {
  assert_pss_wycheproof_vectors(
    PSS_SHA256,
    RsaPssProfile::Sha256,
    "SHA-256",
    ExpectedCounts {
      valid: 63,
      acceptable: 0,
      invalid: 45,
    },
  );
  assert_pss_wycheproof_vectors(
    PSS_SHA384,
    RsaPssProfile::Sha384,
    "SHA-384",
    ExpectedCounts {
      valid: 95,
      acceptable: 0,
      invalid: 46,
    },
  );
  assert_pss_wycheproof_vectors(
    PSS_SHA512,
    RsaPssProfile::Sha512,
    "SHA-512",
    ExpectedCounts {
      valid: 132,
      acceptable: 0,
      invalid: 47,
    },
  );
}
