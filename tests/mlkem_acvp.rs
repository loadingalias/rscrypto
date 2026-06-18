#![cfg(feature = "ml-kem")]

use rscrypto::{
  Kem, MlKem512, MlKem512Ciphertext, MlKem512DecapsulationKey, MlKem512EncapsulationKey, MlKem768, MlKem768Ciphertext,
  MlKem768DecapsulationKey, MlKem768EncapsulationKey, MlKem1024, MlKem1024Ciphertext, MlKem1024DecapsulationKey,
  MlKem1024EncapsulationKey, MlKemError,
};
use serde_json::Value;

// NIST ACVP-Server FIPS 203 ML-KEM gen-val fixtures, pinned at commit
// 15c0f3deeefbfa8cb6cd32a99e1ca3b738c66bf0.
const ACVP_KEYGEN_PROMPT: &str = include_str!("vectors/mlkem_acvp_keygen_fips203_prompt.json");
const ACVP_KEYGEN_EXPECTED: &str = include_str!("vectors/mlkem_acvp_keygen_fips203_expected.json");
const ACVP_ENCAP_DECAP_PROMPT: &str = include_str!("vectors/mlkem_acvp_encapdecap_fips203_prompt.json");
const ACVP_ENCAP_DECAP_EXPECTED: &str = include_str!("vectors/mlkem_acvp_encapdecap_fips203_expected.json");

fn parse_acvp(json: &str) -> Value {
  serde_json::from_str(json).unwrap()
}

fn string_field<'a>(value: &'a Value, field: &str) -> &'a str {
  value
    .get(field)
    .and_then(Value::as_str)
    .unwrap_or_else(|| panic!("missing ACVP string field `{field}`"))
}

fn bool_field(value: &Value, field: &str) -> bool {
  value
    .get(field)
    .and_then(Value::as_bool)
    .unwrap_or_else(|| panic!("missing ACVP bool field `{field}`"))
}

fn u64_field(value: &Value, field: &str) -> u64 {
  value
    .get(field)
    .and_then(Value::as_u64)
    .unwrap_or_else(|| panic!("missing ACVP integer field `{field}`"))
}

fn test_groups(vectors: &Value) -> &[Value] {
  vectors
    .get("testGroups")
    .and_then(Value::as_array)
    .map(Vec::as_slice)
    .expect("ACVP testGroups must be an array")
}

fn test_cases(group: &Value) -> &[Value] {
  group
    .get("tests")
    .and_then(Value::as_array)
    .map(Vec::as_slice)
    .expect("ACVP tests must be an array")
}

fn group_by_id(vectors: &Value, tg_id: u64) -> &Value {
  test_groups(vectors)
    .iter()
    .find(|group| u64_field(group, "tgId") == tg_id)
    .unwrap_or_else(|| panic!("missing ACVP expected group tgId {tg_id}"))
}

fn case_by_id(group: &Value, tc_id: u64) -> &Value {
  test_cases(group)
    .iter()
    .find(|test_case| u64_field(test_case, "tcId") == tc_id)
    .unwrap_or_else(|| panic!("missing ACVP expected tcId {tc_id}"))
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
  assert_eq!(bytes.len(), N, "unexpected ACVP hex length");

  let mut out = [0u8; N];
  out.copy_from_slice(&bytes);
  out
}

fn assert_bytes_eq(actual: &[u8], expected: &[u8], field: &str, parameter_set: &str, tc_id: u64) {
  assert!(
    actual == expected,
    "ACVP {parameter_set} FIPS203 tcId {tc_id} {field} mismatch"
  );
}

macro_rules! assert_keygen_case {
  ($profile:ty, $parameter_set:literal, $prompt:expr, $expected:expr) => {{
    let tc_id = u64_field($prompt, "tcId");
    let d = array_from_hex::<32>(string_field($prompt, "d"));
    let z = array_from_hex::<32>(string_field($prompt, "z"));
    let mut random = [0u8; <$profile>::KEY_GENERATION_RANDOM_SIZE];
    random[..32].copy_from_slice(&d);
    random[32..].copy_from_slice(&z);

    let (encapsulation_key, decapsulation_key) = <$profile>::generate_keypair(|out| {
      out.copy_from_slice(&random);
      Ok::<(), MlKemError>(())
    })
    .unwrap();

    let expected_encapsulation_key =
      array_from_hex::<{ <$profile>::ENCAPSULATION_KEY_SIZE }>(string_field($expected, "ek"));
    let expected_decapsulation_key =
      array_from_hex::<{ <$profile>::DECAPSULATION_KEY_SIZE }>(string_field($expected, "dk"));

    assert_bytes_eq(
      encapsulation_key.as_bytes(),
      &expected_encapsulation_key,
      "ek",
      $parameter_set,
      tc_id,
    );
    assert_bytes_eq(
      decapsulation_key.expose_secret().as_bytes(),
      &expected_decapsulation_key,
      "dk",
      $parameter_set,
      tc_id,
    );
  }};
}

macro_rules! assert_encapsulation_case {
  ($profile:ty, $encapsulation_key:ty, $parameter_set:literal, $prompt:expr, $expected:expr) => {{
    let tc_id = u64_field($prompt, "tcId");
    let encapsulation_key_bytes = decode_hex(string_field($prompt, "ek"));
    let encapsulation_key = <$encapsulation_key>::try_from_slice(&encapsulation_key_bytes).unwrap();
    let random = array_from_hex::<{ <$profile>::ENCAPSULATION_RANDOM_SIZE }>(string_field($prompt, "m"));

    let (ciphertext, shared_secret) = <$profile>::encapsulate(&encapsulation_key, |out| {
      out.copy_from_slice(&random);
      Ok::<(), MlKemError>(())
    })
    .unwrap();

    let expected_ciphertext = array_from_hex::<{ <$profile>::CIPHERTEXT_SIZE }>(string_field($expected, "c"));
    let expected_shared_secret = array_from_hex::<{ <$profile>::SHARED_SECRET_SIZE }>(string_field($expected, "k"));

    assert_bytes_eq(ciphertext.as_bytes(), &expected_ciphertext, "c", $parameter_set, tc_id);
    assert_bytes_eq(
      shared_secret.expose_secret().as_bytes(),
      &expected_shared_secret,
      "k",
      $parameter_set,
      tc_id,
    );
  }};
}

macro_rules! assert_decapsulation_case {
  ($profile:ty, $decapsulation_key:ty, $ciphertext:ty, $parameter_set:literal, $prompt:expr, $expected:expr) => {{
    let tc_id = u64_field($prompt, "tcId");
    let decapsulation_key_bytes = decode_hex(string_field($prompt, "dk"));
    let ciphertext_bytes = decode_hex(string_field($prompt, "c"));
    let decapsulation_key = <$decapsulation_key>::try_from_slice(&decapsulation_key_bytes).unwrap();
    let ciphertext = <$ciphertext>::try_from_slice(&ciphertext_bytes).unwrap();
    let shared_secret = <$profile>::decapsulate(&decapsulation_key, &ciphertext).unwrap();
    let expected_shared_secret = array_from_hex::<{ <$profile>::SHARED_SECRET_SIZE }>(string_field($expected, "k"));

    assert_bytes_eq(
      shared_secret.expose_secret().as_bytes(),
      &expected_shared_secret,
      "k",
      $parameter_set,
      tc_id,
    );
  }};
}

macro_rules! assert_key_check_case {
  ($key:ty, $parameter_set:literal, $prompt:expr, $expected:expr, $field:literal) => {{
    let tc_id = u64_field($prompt, "tcId");
    let key_bytes = decode_hex(string_field($prompt, $field));

    assert_eq!(
      <$key>::try_from_slice(&key_bytes).is_ok(),
      bool_field($expected, "testPassed"),
      "ACVP {parameter_set} FIPS203 tcId {tc_id} {field} key-check mismatch",
      parameter_set = $parameter_set,
      field = $field
    );
  }};
}

#[test]
fn mlkem_matches_acvp_keygen_fips203_vectors() {
  let prompt = parse_acvp(ACVP_KEYGEN_PROMPT);
  let expected = parse_acvp(ACVP_KEYGEN_EXPECTED);

  for prompt_group in test_groups(&prompt) {
    assert_eq!(string_field(prompt_group, "testType"), "AFT");
    let expected_group = group_by_id(&expected, u64_field(prompt_group, "tgId"));

    for prompt_case in test_cases(prompt_group) {
      let expected_case = case_by_id(expected_group, u64_field(prompt_case, "tcId"));

      match string_field(prompt_group, "parameterSet") {
        "ML-KEM-512" => assert_keygen_case!(MlKem512, "ML-KEM-512", prompt_case, expected_case),
        "ML-KEM-768" => assert_keygen_case!(MlKem768, "ML-KEM-768", prompt_case, expected_case),
        "ML-KEM-1024" => assert_keygen_case!(MlKem1024, "ML-KEM-1024", prompt_case, expected_case),
        parameter_set => panic!("unsupported ACVP ML-KEM parameter set `{parameter_set}`"),
      }
    }
  }
}

#[test]
fn mlkem_matches_acvp_encapsulation_fips203_vectors() {
  let prompt = parse_acvp(ACVP_ENCAP_DECAP_PROMPT);
  let expected = parse_acvp(ACVP_ENCAP_DECAP_EXPECTED);

  for prompt_group in test_groups(&prompt) {
    if string_field(prompt_group, "function") != "encapsulation" {
      continue;
    }
    assert_eq!(string_field(prompt_group, "testType"), "AFT");
    let expected_group = group_by_id(&expected, u64_field(prompt_group, "tgId"));

    for prompt_case in test_cases(prompt_group) {
      let expected_case = case_by_id(expected_group, u64_field(prompt_case, "tcId"));

      match string_field(prompt_group, "parameterSet") {
        "ML-KEM-512" => {
          assert_encapsulation_case!(
            MlKem512,
            MlKem512EncapsulationKey,
            "ML-KEM-512",
            prompt_case,
            expected_case
          )
        }
        "ML-KEM-768" => {
          assert_encapsulation_case!(
            MlKem768,
            MlKem768EncapsulationKey,
            "ML-KEM-768",
            prompt_case,
            expected_case
          )
        }
        "ML-KEM-1024" => {
          assert_encapsulation_case!(
            MlKem1024,
            MlKem1024EncapsulationKey,
            "ML-KEM-1024",
            prompt_case,
            expected_case
          )
        }
        parameter_set => panic!("unsupported ACVP ML-KEM parameter set `{parameter_set}`"),
      }
    }
  }
}

#[test]
fn mlkem_matches_acvp_decapsulation_fips203_vectors() {
  let prompt = parse_acvp(ACVP_ENCAP_DECAP_PROMPT);
  let expected = parse_acvp(ACVP_ENCAP_DECAP_EXPECTED);

  for prompt_group in test_groups(&prompt) {
    if string_field(prompt_group, "function") != "decapsulation" {
      continue;
    }
    assert_eq!(string_field(prompt_group, "testType"), "VAL");
    let expected_group = group_by_id(&expected, u64_field(prompt_group, "tgId"));

    for prompt_case in test_cases(prompt_group) {
      let expected_case = case_by_id(expected_group, u64_field(prompt_case, "tcId"));

      match string_field(prompt_group, "parameterSet") {
        "ML-KEM-512" => assert_decapsulation_case!(
          MlKem512,
          MlKem512DecapsulationKey,
          MlKem512Ciphertext,
          "ML-KEM-512",
          prompt_case,
          expected_case
        ),
        "ML-KEM-768" => assert_decapsulation_case!(
          MlKem768,
          MlKem768DecapsulationKey,
          MlKem768Ciphertext,
          "ML-KEM-768",
          prompt_case,
          expected_case
        ),
        "ML-KEM-1024" => assert_decapsulation_case!(
          MlKem1024,
          MlKem1024DecapsulationKey,
          MlKem1024Ciphertext,
          "ML-KEM-1024",
          prompt_case,
          expected_case
        ),
        parameter_set => panic!("unsupported ACVP ML-KEM parameter set `{parameter_set}`"),
      }
    }
  }
}

#[test]
fn mlkem_matches_acvp_decapsulation_key_check_fips203_vectors() {
  let prompt = parse_acvp(ACVP_ENCAP_DECAP_PROMPT);
  let expected = parse_acvp(ACVP_ENCAP_DECAP_EXPECTED);

  for prompt_group in test_groups(&prompt) {
    if string_field(prompt_group, "function") != "decapsulationKeyCheck" {
      continue;
    }
    assert_eq!(string_field(prompt_group, "testType"), "VAL");
    let expected_group = group_by_id(&expected, u64_field(prompt_group, "tgId"));

    for prompt_case in test_cases(prompt_group) {
      let expected_case = case_by_id(expected_group, u64_field(prompt_case, "tcId"));

      match string_field(prompt_group, "parameterSet") {
        "ML-KEM-512" => {
          assert_key_check_case!(MlKem512DecapsulationKey, "ML-KEM-512", prompt_case, expected_case, "dk")
        }
        "ML-KEM-768" => {
          assert_key_check_case!(MlKem768DecapsulationKey, "ML-KEM-768", prompt_case, expected_case, "dk")
        }
        "ML-KEM-1024" => {
          assert_key_check_case!(
            MlKem1024DecapsulationKey,
            "ML-KEM-1024",
            prompt_case,
            expected_case,
            "dk"
          )
        }
        parameter_set => panic!("unsupported ACVP ML-KEM parameter set `{parameter_set}`"),
      }
    }
  }
}

#[test]
fn mlkem_matches_acvp_encapsulation_key_check_fips203_vectors() {
  let prompt = parse_acvp(ACVP_ENCAP_DECAP_PROMPT);
  let expected = parse_acvp(ACVP_ENCAP_DECAP_EXPECTED);

  for prompt_group in test_groups(&prompt) {
    if string_field(prompt_group, "function") != "encapsulationKeyCheck" {
      continue;
    }
    assert_eq!(string_field(prompt_group, "testType"), "VAL");
    let expected_group = group_by_id(&expected, u64_field(prompt_group, "tgId"));

    for prompt_case in test_cases(prompt_group) {
      let expected_case = case_by_id(expected_group, u64_field(prompt_case, "tcId"));

      match string_field(prompt_group, "parameterSet") {
        "ML-KEM-512" => {
          assert_key_check_case!(MlKem512EncapsulationKey, "ML-KEM-512", prompt_case, expected_case, "ek")
        }
        "ML-KEM-768" => {
          assert_key_check_case!(MlKem768EncapsulationKey, "ML-KEM-768", prompt_case, expected_case, "ek")
        }
        "ML-KEM-1024" => {
          assert_key_check_case!(
            MlKem1024EncapsulationKey,
            "ML-KEM-1024",
            prompt_case,
            expected_case,
            "ek"
          )
        }
        parameter_set => panic!("unsupported ACVP ML-KEM parameter set `{parameter_set}`"),
      }
    }
  }
}
