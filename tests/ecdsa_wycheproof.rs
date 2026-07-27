#![cfg(any(feature = "ecdsa-p256", feature = "ecdsa-p384"))]

#[cfg(feature = "ecdsa-p256")]
use rscrypto::{EcdsaError, EcdsaP256PublicKey, EcdsaP256Signature};
#[cfg(feature = "ecdsa-p384")]
use rscrypto::{EcdsaP384PublicKey, EcdsaP384Signature};
use serde_json::Value;

mod common;
use common::decode_hex_vec;

#[cfg(feature = "ecdsa-p256")]
const P256_DER: &str = include_str!("../testdata/auth/wycheproof/ecdsa_secp256r1_sha256_test.json");
#[cfg(feature = "ecdsa-p384")]
const P384_DER: &str = include_str!("../testdata/auth/wycheproof/ecdsa_secp384r1_sha384_test.json");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Counts {
  valid: usize,
  invalid: usize,
}

struct Suite<'a> {
  json: &'a str,
  curve: &'a str,
  key_size: u64,
  sha: &'a str,
  expected: Counts,
}

fn field<'a>(value: &'a Value, name: &str) -> &'a str {
  value[name]
    .as_str()
    .unwrap_or_else(|| panic!("missing string field `{name}`"))
}

fn assert_der_vectors<Key>(
  spec: Suite<'_>,
  parse_sec1: impl Fn(&[u8]) -> Key,
  parse_spki: impl Fn(&[u8]) -> Key,
  verify_der: impl Fn(&Key, &[u8], &[u8]) -> bool,
) where
  Key: core::fmt::Debug + Eq,
{
  let suite: Value = serde_json::from_str(spec.json).expect("Wycheproof ECDSA JSON must parse");
  assert_eq!(suite["algorithm"].as_str(), Some("ECDSA"));
  assert_eq!(suite["schema"].as_str(), Some("ecdsa_verify_schema_v1.json"));
  assert_eq!(
    suite["numberOfTests"].as_u64(),
    Some((spec.expected.valid + spec.expected.invalid) as u64)
  );

  let groups = suite["testGroups"]
    .as_array()
    .expect("Wycheproof testGroups must be an array");
  let mut counts = Counts { valid: 0, invalid: 0 };

  for group in groups {
    assert_eq!(group["type"].as_str(), Some("EcdsaVerify"));
    assert_eq!(group["sha"].as_str(), Some(spec.sha));
    assert_eq!(group["publicKey"]["curve"].as_str(), Some(spec.curve));
    assert_eq!(group["publicKey"]["keySize"].as_u64(), Some(spec.key_size));

    let public = parse_sec1(&decode_hex_vec(field(&group["publicKey"], "uncompressed")));
    let spki_public = parse_spki(&decode_hex_vec(field(group, "publicKeyDer")));
    assert_eq!(public, spki_public, "SEC1 and SPKI public keys must agree");

    for test in group["tests"].as_array().expect("Wycheproof tests must be an array") {
      let message = decode_hex_vec(field(test, "msg"));
      let signature = decode_hex_vec(field(test, "sig"));
      let verified = verify_der(&public, &message, &signature);

      match field(test, "result") {
        "valid" => {
          counts.valid = counts.valid.strict_add(1);
          assert!(
            verified,
            "Wycheproof ECDSA tcId {} rejected a valid signature: {}",
            test["tcId"],
            field(test, "comment")
          );
        }
        "invalid" => {
          counts.invalid = counts.invalid.strict_add(1);
          assert!(
            !verified,
            "Wycheproof ECDSA tcId {} accepted an invalid signature: {}",
            test["tcId"],
            field(test, "comment")
          );
        }
        other => panic!("unsupported Wycheproof ECDSA result `{other}`"),
      }
    }
  }

  assert_eq!(counts, spec.expected);
}

#[cfg(feature = "ecdsa-p256")]
#[test]
fn wycheproof_p256_der_verification_matches_expected_results() {
  assert_der_vectors(
    Suite {
      json: P256_DER,
      curve: "secp256r1",
      key_size: 256,
      sha: "SHA-256",
      expected: Counts {
        valid: 174,
        invalid: 310,
      },
    },
    |sec1| EcdsaP256PublicKey::from_sec1_bytes(sec1).expect("Wycheproof P-256 SEC1 public key must parse"),
    |spki| EcdsaP256PublicKey::from_spki_der(spki).expect("Wycheproof P-256 SPKI public key must parse"),
    |public, message, der| {
      EcdsaP256Signature::from_der(der).is_ok_and(|signature| public.verify(message, &signature).is_ok())
    },
  );
}

#[cfg(feature = "ecdsa-p256")]
#[test]
fn p256_rejects_a_noncanonical_coordinate_before_field_reduction() {
  let canonical = decode_hex_vec(
    "04bcbb2914c79f045eaa6ecbbc612816b3be5d2d6796707d8125e9f851c18af015\
     000000001352bb4a0fa2ea4cceb9ab63dd684ade5a1127bcf300a698a7193bc2",
  );
  EcdsaP256PublicKey::from_sec1_bytes(&canonical).expect("canonical P-256 public key must parse");

  let noncanonical = decode_hex_vec(
    "04bcbb2914c79f045eaa6ecbbc612816b3be5d2d6796707d8125e9f851c18af015\
     ffffffff1352bb4b0fa2ea4cceb9ab63dd684adf5a1127bcf300a698a7193bc1",
  );
  assert_eq!(
    EcdsaP256PublicKey::from_sec1_bytes(&noncanonical),
    Err(EcdsaError::InvalidPublicKey)
  );
}

#[cfg(feature = "ecdsa-p384")]
#[test]
fn wycheproof_p384_der_verification_matches_expected_results() {
  assert_der_vectors(
    Suite {
      json: P384_DER,
      curve: "secp384r1",
      key_size: 384,
      sha: "SHA-384",
      expected: Counts {
        valid: 194,
        invalid: 310,
      },
    },
    |sec1| EcdsaP384PublicKey::from_sec1_bytes(sec1).expect("Wycheproof P-384 SEC1 public key must parse"),
    |spki| EcdsaP384PublicKey::from_spki_der(spki).expect("Wycheproof P-384 SPKI public key must parse"),
    |public, message, der| {
      EcdsaP384Signature::from_der(der).is_ok_and(|signature| public.verify(message, &signature).is_ok())
    },
  );
}
