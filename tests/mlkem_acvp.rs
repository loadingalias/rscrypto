#![cfg(feature = "ml-kem")]

use rscrypto::{Kem, MlKem768, MlKem768Ciphertext, MlKem768DecapsulationKey, MlKem768EncapsulationKey, MlKemError};
use serde_json::Value;

const ACVP_MLKEM768_FIPS203: &str = include_str!("vectors/mlkem768_acvp_fips203.json");

fn acvp_vectors() -> Value {
  serde_json::from_str(ACVP_MLKEM768_FIPS203).unwrap()
}

fn cases<'a>(vectors: &'a Value, group: &str) -> &'a [Value] {
  vectors
    .get(group)
    .and_then(Value::as_array)
    .unwrap_or_else(|| panic!("missing ACVP group {group}"))
}

fn tc_id(test_case: &Value) -> u64 {
  test_case
    .get("tcId")
    .and_then(Value::as_u64)
    .unwrap_or_else(|| panic!("missing ACVP tcId"))
}

fn string_field<'a>(test_case: &'a Value, field: &str) -> &'a str {
  test_case
    .get(field)
    .and_then(Value::as_str)
    .unwrap_or_else(|| panic!("missing ACVP field {field} for tcId {}", tc_id(test_case)))
}

fn bool_field(test_case: &Value, field: &str) -> bool {
  test_case
    .get(field)
    .and_then(Value::as_bool)
    .unwrap_or_else(|| panic!("missing ACVP field {field} for tcId {}", tc_id(test_case)))
}

fn hex_nibble(byte: u8) -> u8 {
  match byte {
    b'0'..=b'9' => byte - b'0',
    b'a'..=b'f' => byte - b'a' + 10,
    b'A'..=b'F' => byte - b'A' + 10,
    _ => panic!("invalid hex byte {byte:#x}"),
  }
}

fn decode_hex(hex: &str) -> Vec<u8> {
  assert_eq!(hex.len() % 2, 0, "hex input must have even length");

  let mut out = Vec::with_capacity(hex.len() / 2);
  for pair in hex.as_bytes().chunks_exact(2) {
    out.push((hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]));
  }
  out
}

fn array_from_hex<const N: usize>(hex: &str) -> [u8; N] {
  let bytes = decode_hex(hex);
  assert_eq!(bytes.len(), N, "unexpected hex length");

  let mut out = [0u8; N];
  out.copy_from_slice(&bytes);
  out
}

fn assert_bytes_eq(actual: &[u8], expected: &[u8], field: &str, tc_id: u64) {
  assert!(
    actual == expected,
    "ACVP ML-KEM-768 FIPS203 tcId {tc_id} {field} mismatch"
  );
}

#[test]
fn mlkem768_matches_acvp_keygen_vectors() {
  let vectors = acvp_vectors();

  for test_case in cases(&vectors, "keyGen") {
    let tc_id = tc_id(test_case);
    let d = array_from_hex::<32>(string_field(test_case, "d"));
    let z = array_from_hex::<32>(string_field(test_case, "z"));
    let mut random = [0u8; MlKem768::KEY_GENERATION_RANDOM_SIZE];
    random[..32].copy_from_slice(&d);
    random[32..].copy_from_slice(&z);

    let (ek, dk) = MlKem768::generate_keypair(|out| {
      out.copy_from_slice(&random);
      Ok::<(), MlKemError>(())
    })
    .unwrap();

    let expected_ek = array_from_hex::<{ MlKem768::ENCAPSULATION_KEY_SIZE }>(string_field(test_case, "ek"));
    let expected_dk = array_from_hex::<{ MlKem768::DECAPSULATION_KEY_SIZE }>(string_field(test_case, "dk"));

    assert_bytes_eq(ek.as_bytes(), &expected_ek, "ek", tc_id);
    assert_bytes_eq(dk.expose_secret().as_bytes(), &expected_dk, "dk", tc_id);
  }
}

#[test]
fn mlkem768_matches_acvp_encapsulation_vectors() {
  let vectors = acvp_vectors();

  for test_case in cases(&vectors, "encapsulation") {
    let tc_id = tc_id(test_case);
    let ek_bytes = decode_hex(string_field(test_case, "ek"));
    let ek = MlKem768EncapsulationKey::try_from_slice(&ek_bytes).unwrap();
    let m = array_from_hex::<{ MlKem768::ENCAPSULATION_RANDOM_SIZE }>(string_field(test_case, "m"));

    let (ciphertext, shared_secret) = MlKem768::encapsulate(&ek, |out| {
      out.copy_from_slice(&m);
      Ok::<(), MlKemError>(())
    })
    .unwrap();

    let expected_c = array_from_hex::<{ MlKem768::CIPHERTEXT_SIZE }>(string_field(test_case, "c"));
    let expected_k = array_from_hex::<{ MlKem768::SHARED_SECRET_SIZE }>(string_field(test_case, "k"));

    assert_bytes_eq(ciphertext.as_bytes(), &expected_c, "c", tc_id);
    assert_bytes_eq(shared_secret.expose_secret().as_bytes(), &expected_k, "k", tc_id);
  }
}

#[test]
fn mlkem768_matches_acvp_decapsulation_vectors() {
  let vectors = acvp_vectors();

  for test_case in cases(&vectors, "decapsulation") {
    let tc_id = tc_id(test_case);
    let dk_bytes = decode_hex(string_field(test_case, "dk"));
    let ciphertext_bytes = decode_hex(string_field(test_case, "c"));
    let dk = MlKem768DecapsulationKey::try_from_slice(&dk_bytes).unwrap();
    let ciphertext = MlKem768Ciphertext::try_from_slice(&ciphertext_bytes).unwrap();
    let shared_secret = MlKem768::decapsulate(&dk, &ciphertext).unwrap();
    let expected_k = array_from_hex::<{ MlKem768::SHARED_SECRET_SIZE }>(string_field(test_case, "k"));

    assert_bytes_eq(shared_secret.expose_secret().as_bytes(), &expected_k, "k", tc_id);
  }
}

#[test]
fn mlkem768_matches_acvp_decapsulation_key_check_vectors() {
  let vectors = acvp_vectors();

  for test_case in cases(&vectors, "decapsulationKeyCheck") {
    let tc_id = tc_id(test_case);
    let dk_bytes = decode_hex(string_field(test_case, "dk"));

    assert_eq!(
      MlKem768DecapsulationKey::try_from_slice(&dk_bytes).is_ok(),
      bool_field(test_case, "testPassed"),
      "ACVP ML-KEM-768 FIPS203 decapsulationKeyCheck tcId {tc_id} mismatch"
    );
  }
}

#[test]
fn mlkem768_matches_acvp_encapsulation_key_check_vectors() {
  let vectors = acvp_vectors();

  for test_case in cases(&vectors, "encapsulationKeyCheck") {
    let tc_id = tc_id(test_case);
    let ek_bytes = decode_hex(string_field(test_case, "ek"));

    assert_eq!(
      MlKem768EncapsulationKey::try_from_slice(&ek_bytes).is_ok(),
      bool_field(test_case, "testPassed"),
      "ACVP ML-KEM-768 FIPS203 encapsulationKeyCheck tcId {tc_id} mismatch"
    );
  }
}
