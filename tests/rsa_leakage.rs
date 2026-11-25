#![cfg(all(feature = "rsa", feature = "diag", feature = "getrandom"))]

mod common;
use common::decode_hex_vec;

use rscrypto::{RsaPrivateKey, RsaPublicKeyPolicy};
use serde_json::Value;

const OAEP_SHA256: &str = include_str!("../testdata/rsa/wycheproof/rsa_oaep_2048_sha256_mgf1sha256_test.json");

fn legacy_rsa2048_fixture_key() -> RsaPrivateKey {
  let suite: Value = serde_json::from_str(OAEP_SHA256).expect("Wycheproof OAEP JSON must parse");
  let group = suite["testGroups"]
    .as_array()
    .and_then(|groups| groups.first())
    .expect("Wycheproof OAEP group must exist");
  let der = decode_hex_vec(group["privateKeyPkcs8"].as_str().expect("privateKeyPkcs8 must exist"));
  let key = RsaPrivateKey::from_pkcs8_der_with_policy(&der, &RsaPublicKeyPolicy::modulus_2048_to_8192())
    .expect("Wycheproof RSA-2048 private key must parse with explicit legacy policy");
  assert_eq!(key.public_key().modulus_bits(), 2048);
  key
}

#[test]
fn rsa_leakage_fixture_uses_explicit_legacy_rsa2048_policy() {
  let key = legacy_rsa2048_fixture_key();
  assert_eq!(key.public_key().public_exponent().as_u64(), 65_537);
}
