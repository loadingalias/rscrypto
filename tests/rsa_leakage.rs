#![cfg(all(feature = "rsa", feature = "diag", feature = "getrandom"))]

use rscrypto::{RsaPrivateKey, RsaPublicKeyPolicy};
use serde_json::Value;

const OAEP_SHA256: &str = include_str!("../testdata/rsa/wycheproof/rsa_oaep_2048_sha256_mgf1sha256_test.json");

fn hex_to_vec(hex: &str) -> Vec<u8> {
  assert_eq!(hex.len() % 2, 0);
  let mut out = Vec::with_capacity(hex.len() / 2);
  for chunk in hex.as_bytes().as_chunks::<2>().0 {
    let high = hex_value(chunk[0]).expect("leakage fixture must contain hexadecimal digits");
    let low = hex_value(chunk[1]).expect("leakage fixture must contain hexadecimal digits");
    out.push((high << 4) | low);
  }
  out
}

fn hex_value(byte: u8) -> Option<u8> {
  match byte {
    b'0'..=b'9' => Some(byte.strict_sub(b'0')),
    b'a'..=b'f' => Some(byte.strict_sub(b'a').strict_add(10)),
    b'A'..=b'F' => Some(byte.strict_sub(b'A').strict_add(10)),
    _ => None,
  }
}

fn legacy_rsa2048_fixture_key() -> RsaPrivateKey {
  let suite: Value = serde_json::from_str(OAEP_SHA256).expect("Wycheproof OAEP JSON must parse");
  let group = suite["testGroups"]
    .as_array()
    .and_then(|groups| groups.first())
    .expect("Wycheproof OAEP group must exist");
  let der = hex_to_vec(group["privateKeyPkcs8"].as_str().expect("privateKeyPkcs8 must exist"));
  let key = RsaPrivateKey::from_pkcs8_der_with_policy(&der, &RsaPublicKeyPolicy::legacy_verification())
    .expect("Wycheproof RSA-2048 private key must parse with explicit legacy policy");
  assert_eq!(key.public_key().modulus_bits(), 2048);
  key
}

#[test]
fn rsa_leakage_fixture_uses_explicit_legacy_rsa2048_policy() {
  let key = legacy_rsa2048_fixture_key();
  assert_eq!(key.public_key().public_exponent().as_u64(), 65_537);
}
