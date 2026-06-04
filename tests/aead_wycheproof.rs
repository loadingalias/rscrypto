#![cfg(feature = "aead")]

use rscrypto::{
  Aegis256, Aegis256Key, Aegis256Tag, Aes128Gcm, Aes128GcmKey, Aes128GcmSiv, Aes128GcmSivKey, Aes128GcmSivTag,
  Aes128GcmTag, Aes256Gcm, Aes256GcmKey, Aes256GcmSiv, Aes256GcmSivKey, Aes256GcmSivTag, Aes256GcmTag,
  ChaCha20Poly1305, ChaCha20Poly1305Key, ChaCha20Poly1305Tag, XChaCha20Poly1305, XChaCha20Poly1305Key,
  XChaCha20Poly1305Tag,
  aead::{Nonce96, Nonce192, Nonce256},
};
use serde_json::Value;

mod common;
use common::decode_hex_vec;

const AES_GCM: &str = include_str!("../testdata/aead/wycheproof/aes_gcm_test.json");
const AES_GCM_SIV: &str = include_str!("../testdata/aead/wycheproof/aes_gcm_siv_test.json");
const CHACHA20_POLY1305: &str = include_str!("../testdata/aead/wycheproof/chacha20_poly1305_test.json");
const XCHACHA20_POLY1305: &str = include_str!("../testdata/aead/wycheproof/xchacha20_poly1305_test.json");
const AEGIS256: &str = include_str!("../testdata/aead/wycheproof/aegis256_test.json");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Counts {
  valid: usize,
  invalid: usize,
}

struct AeadCase {
  tc_id: u64,
  comment: String,
  key: Vec<u8>,
  nonce: Vec<u8>,
  aad: Vec<u8>,
  msg: Vec<u8>,
  ct: Vec<u8>,
  tag: Vec<u8>,
  valid: bool,
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

fn case_from_json(test: &Value) -> AeadCase {
  AeadCase {
    tc_id: test["tcId"].as_u64().expect("tcId must be numeric"),
    comment: field(test, "comment").to_owned(),
    key: decode_hex_vec(field(test, "key")),
    nonce: decode_hex_vec(field(test, "iv")),
    aad: decode_hex_vec(field(test, "aad")),
    msg: decode_hex_vec(field(test, "msg")),
    ct: decode_hex_vec(field(test, "ct")),
    tag: decode_hex_vec(field(test, "tag")),
    valid: match field(test, "result") {
      "valid" => true,
      "invalid" => false,
      other => panic!("unsupported Wycheproof AEAD result `{other}`"),
    },
  }
}

fn assert_open_failure_clears_buffer(case: &AeadCase, buffer: &[u8]) {
  assert!(
    buffer.iter().all(|&byte| byte == 0),
    "Wycheproof tcId {} ({}) failed open but left plaintext/ciphertext material in caller buffer",
    case.tc_id,
    case.comment
  );
}

fn assert_chacha20poly1305_case(case: &AeadCase) {
  let key = ChaCha20Poly1305Key::from_bytes(case.key.as_slice().try_into().unwrap());
  let nonce = Nonce96::from_bytes(case.nonce.as_slice().try_into().unwrap());
  let tag = ChaCha20Poly1305Tag::from_bytes(case.tag.as_slice().try_into().unwrap());
  let cipher = ChaCha20Poly1305::new(&key);
  let mut buffer = case.ct.clone();

  if case.valid {
    cipher.decrypt_in_place(&nonce, &case.aad, &mut buffer, &tag).unwrap();
    assert_eq!(
      buffer, case.msg,
      "Wycheproof ChaCha20-Poly1305 tcId {} opened incorrectly",
      case.tc_id
    );

    let mut sealed = case.msg.clone();
    let actual_tag = cipher.encrypt_in_place(&nonce, &case.aad, &mut sealed).unwrap();
    assert_eq!(
      sealed, case.ct,
      "Wycheproof ChaCha20-Poly1305 tcId {} encrypted incorrectly",
      case.tc_id
    );
    assert_eq!(
      actual_tag.as_bytes(),
      case.tag.as_slice(),
      "Wycheproof ChaCha20-Poly1305 tcId {} tag mismatch",
      case.tc_id
    );
  } else {
    assert!(cipher.decrypt_in_place(&nonce, &case.aad, &mut buffer, &tag).is_err());
    assert_open_failure_clears_buffer(case, &buffer);
  }
}

fn assert_xchacha20poly1305_case(case: &AeadCase) {
  let key = XChaCha20Poly1305Key::from_bytes(case.key.as_slice().try_into().unwrap());
  let nonce = Nonce192::from_bytes(case.nonce.as_slice().try_into().unwrap());
  let tag = XChaCha20Poly1305Tag::from_bytes(case.tag.as_slice().try_into().unwrap());
  let cipher = XChaCha20Poly1305::new(&key);
  let mut buffer = case.ct.clone();

  if case.valid {
    cipher.decrypt_in_place(&nonce, &case.aad, &mut buffer, &tag).unwrap();
    assert_eq!(
      buffer, case.msg,
      "Wycheproof XChaCha20-Poly1305 tcId {} opened incorrectly",
      case.tc_id
    );

    let mut sealed = case.msg.clone();
    let actual_tag = cipher.encrypt_in_place(&nonce, &case.aad, &mut sealed).unwrap();
    assert_eq!(
      sealed, case.ct,
      "Wycheproof XChaCha20-Poly1305 tcId {} encrypted incorrectly",
      case.tc_id
    );
    assert_eq!(
      actual_tag.as_bytes(),
      case.tag.as_slice(),
      "Wycheproof XChaCha20-Poly1305 tcId {} tag mismatch",
      case.tc_id
    );
  } else {
    assert!(cipher.decrypt_in_place(&nonce, &case.aad, &mut buffer, &tag).is_err());
    assert_open_failure_clears_buffer(case, &buffer);
  }
}

fn assert_aes128gcm_case(case: &AeadCase) {
  let key = Aes128GcmKey::from_bytes(case.key.as_slice().try_into().unwrap());
  let nonce = Nonce96::from_bytes(case.nonce.as_slice().try_into().unwrap());
  let tag = Aes128GcmTag::from_bytes(case.tag.as_slice().try_into().unwrap());
  let cipher = Aes128Gcm::new(&key);
  let mut buffer = case.ct.clone();

  if case.valid {
    cipher.decrypt_in_place(&nonce, &case.aad, &mut buffer, &tag).unwrap();
    assert_eq!(
      buffer, case.msg,
      "Wycheproof AES-128-GCM tcId {} opened incorrectly",
      case.tc_id
    );
    let mut sealed = case.msg.clone();
    let actual_tag = cipher.encrypt_in_place(&nonce, &case.aad, &mut sealed).unwrap();
    assert_eq!(
      sealed, case.ct,
      "Wycheproof AES-128-GCM tcId {} encrypted incorrectly",
      case.tc_id
    );
    assert_eq!(
      actual_tag.as_bytes(),
      case.tag.as_slice(),
      "Wycheproof AES-128-GCM tcId {} tag mismatch",
      case.tc_id
    );
  } else {
    assert!(cipher.decrypt_in_place(&nonce, &case.aad, &mut buffer, &tag).is_err());
    assert_open_failure_clears_buffer(case, &buffer);
  }
}

fn assert_aes256gcm_case(case: &AeadCase) {
  let key = Aes256GcmKey::from_bytes(case.key.as_slice().try_into().unwrap());
  let nonce = Nonce96::from_bytes(case.nonce.as_slice().try_into().unwrap());
  let tag = Aes256GcmTag::from_bytes(case.tag.as_slice().try_into().unwrap());
  let cipher = Aes256Gcm::new(&key);
  let mut buffer = case.ct.clone();

  if case.valid {
    cipher.decrypt_in_place(&nonce, &case.aad, &mut buffer, &tag).unwrap();
    assert_eq!(
      buffer, case.msg,
      "Wycheproof AES-256-GCM tcId {} opened incorrectly",
      case.tc_id
    );
    let mut sealed = case.msg.clone();
    let actual_tag = cipher.encrypt_in_place(&nonce, &case.aad, &mut sealed).unwrap();
    assert_eq!(
      sealed, case.ct,
      "Wycheproof AES-256-GCM tcId {} encrypted incorrectly",
      case.tc_id
    );
    assert_eq!(
      actual_tag.as_bytes(),
      case.tag.as_slice(),
      "Wycheproof AES-256-GCM tcId {} tag mismatch",
      case.tc_id
    );
  } else {
    assert!(cipher.decrypt_in_place(&nonce, &case.aad, &mut buffer, &tag).is_err());
    assert_open_failure_clears_buffer(case, &buffer);
  }
}

fn assert_aes128gcmsiv_case(case: &AeadCase) {
  let key = Aes128GcmSivKey::from_bytes(case.key.as_slice().try_into().unwrap());
  let nonce = Nonce96::from_bytes(case.nonce.as_slice().try_into().unwrap());
  let tag = Aes128GcmSivTag::from_bytes(case.tag.as_slice().try_into().unwrap());
  let cipher = Aes128GcmSiv::new(&key);
  let mut buffer = case.ct.clone();

  if case.valid {
    cipher.decrypt_in_place(&nonce, &case.aad, &mut buffer, &tag).unwrap();
    assert_eq!(
      buffer, case.msg,
      "Wycheproof AES-128-GCM-SIV tcId {} opened incorrectly",
      case.tc_id
    );
    let mut sealed = case.msg.clone();
    let actual_tag = cipher.encrypt_in_place(&nonce, &case.aad, &mut sealed).unwrap();
    assert_eq!(
      sealed, case.ct,
      "Wycheproof AES-128-GCM-SIV tcId {} encrypted incorrectly",
      case.tc_id
    );
    assert_eq!(
      actual_tag.as_bytes(),
      case.tag.as_slice(),
      "Wycheproof AES-128-GCM-SIV tcId {} tag mismatch",
      case.tc_id
    );
  } else {
    assert!(cipher.decrypt_in_place(&nonce, &case.aad, &mut buffer, &tag).is_err());
    assert_open_failure_clears_buffer(case, &buffer);
  }
}

fn assert_aes256gcmsiv_case(case: &AeadCase) {
  let key = Aes256GcmSivKey::from_bytes(case.key.as_slice().try_into().unwrap());
  let nonce = Nonce96::from_bytes(case.nonce.as_slice().try_into().unwrap());
  let tag = Aes256GcmSivTag::from_bytes(case.tag.as_slice().try_into().unwrap());
  let cipher = Aes256GcmSiv::new(&key);
  let mut buffer = case.ct.clone();

  if case.valid {
    cipher.decrypt_in_place(&nonce, &case.aad, &mut buffer, &tag).unwrap();
    assert_eq!(
      buffer, case.msg,
      "Wycheproof AES-256-GCM-SIV tcId {} opened incorrectly",
      case.tc_id
    );
    let mut sealed = case.msg.clone();
    let actual_tag = cipher.encrypt_in_place(&nonce, &case.aad, &mut sealed).unwrap();
    assert_eq!(
      sealed, case.ct,
      "Wycheproof AES-256-GCM-SIV tcId {} encrypted incorrectly",
      case.tc_id
    );
    assert_eq!(
      actual_tag.as_bytes(),
      case.tag.as_slice(),
      "Wycheproof AES-256-GCM-SIV tcId {} tag mismatch",
      case.tc_id
    );
  } else {
    assert!(cipher.decrypt_in_place(&nonce, &case.aad, &mut buffer, &tag).is_err());
    assert_open_failure_clears_buffer(case, &buffer);
  }
}

fn assert_aegis256_case(case: &AeadCase) {
  let key = Aegis256Key::from_bytes(case.key.as_slice().try_into().unwrap());
  let nonce = Nonce256::from_bytes(case.nonce.as_slice().try_into().unwrap());
  let tag = Aegis256Tag::from_bytes(case.tag.as_slice().try_into().unwrap());
  let cipher = Aegis256::new(&key);
  let mut buffer = case.ct.clone();

  if case.valid {
    cipher.decrypt_in_place(&nonce, &case.aad, &mut buffer, &tag).unwrap();
    assert_eq!(
      buffer, case.msg,
      "Wycheproof AEGIS-256 tcId {} opened incorrectly",
      case.tc_id
    );
    let mut sealed = case.msg.clone();
    let actual_tag = cipher.encrypt_in_place(&nonce, &case.aad, &mut sealed).unwrap();
    assert_eq!(
      sealed, case.ct,
      "Wycheproof AEGIS-256 tcId {} encrypted incorrectly",
      case.tc_id
    );
    assert_eq!(
      actual_tag.as_bytes(),
      case.tag.as_slice(),
      "Wycheproof AEGIS-256 tcId {} tag mismatch",
      case.tc_id
    );
  } else {
    assert!(cipher.decrypt_in_place(&nonce, &case.aad, &mut buffer, &tag).is_err());
    assert_open_failure_clears_buffer(case, &buffer);
  }
}

fn assert_wycheproof_suite(
  json: &str,
  expected_algorithm: &str,
  expected: Counts,
  applies: impl Fn(&Value) -> bool,
  assert_case: impl Fn(&AeadCase),
) {
  let suite: Value = serde_json::from_str(json).expect("Wycheproof AEAD JSON must parse");
  assert_eq!(suite["algorithm"].as_str(), Some(expected_algorithm));

  let mut counts = Counts { valid: 0, invalid: 0 };
  for group in groups(&suite).iter().filter(|group| applies(group)) {
    assert_eq!(group["type"].as_str(), Some("AeadTest"));
    assert_eq!(group["tagSize"].as_u64(), Some(128));

    for test in tests(group) {
      let case = case_from_json(test);
      if case.valid {
        counts.valid = counts.valid.strict_add(1);
      } else {
        counts.invalid = counts.invalid.strict_add(1);
      }
      assert_case(&case);
    }
  }

  assert_eq!(counts, expected);
}

#[test]
fn wycheproof_aes_gcm_open_vectors_match_expected_results() {
  assert_wycheproof_suite(
    AES_GCM,
    "AES-GCM",
    Counts { valid: 79, invalid: 54 },
    |group| {
      group["ivSize"].as_u64() == Some(96)
        && matches!(group["keySize"].as_u64(), Some(128 | 256))
        && group["tagSize"].as_u64() == Some(128)
    },
    |case| match case.key.len() {
      16 => assert_aes128gcm_case(case),
      32 => assert_aes256gcm_case(case),
      len => panic!("unsupported AES-GCM key length {len}"),
    },
  );
}

#[test]
fn wycheproof_aes_gcm_siv_open_vectors_match_expected_results() {
  assert_wycheproof_suite(
    AES_GCM_SIV,
    "AES-GCM-SIV",
    Counts {
      valid: 136,
      invalid: 66,
    },
    |group| {
      group["ivSize"].as_u64() == Some(96)
        && matches!(group["keySize"].as_u64(), Some(128 | 256))
        && group["tagSize"].as_u64() == Some(128)
    },
    |case| match case.key.len() {
      16 => assert_aes128gcmsiv_case(case),
      32 => assert_aes256gcmsiv_case(case),
      len => panic!("unsupported AES-GCM-SIV key length {len}"),
    },
  );
}

#[test]
fn wycheproof_chacha20poly1305_open_vectors_match_expected_results() {
  assert_wycheproof_suite(
    CHACHA20_POLY1305,
    "CHACHA20-POLY1305",
    Counts {
      valid: 256,
      invalid: 60,
    },
    |group| {
      group["ivSize"].as_u64() == Some(96)
        && group["keySize"].as_u64() == Some(256)
        && group["tagSize"].as_u64() == Some(128)
    },
    assert_chacha20poly1305_case,
  );
}

#[test]
fn wycheproof_xchacha20poly1305_open_vectors_match_expected_results() {
  assert_wycheproof_suite(
    XCHACHA20_POLY1305,
    "XCHACHA20-POLY1305",
    Counts {
      valid: 246,
      invalid: 60,
    },
    |group| {
      group["ivSize"].as_u64() == Some(192)
        && group["keySize"].as_u64() == Some(256)
        && group["tagSize"].as_u64() == Some(128)
    },
    assert_xchacha20poly1305_case,
  );
}

#[test]
fn wycheproof_aegis256_open_vectors_match_expected_results() {
  assert_wycheproof_suite(
    AEGIS256,
    "AEGIS256",
    Counts {
      valid: 360,
      invalid: 112,
    },
    |group| {
      group["ivSize"].as_u64() == Some(256)
        && group["keySize"].as_u64() == Some(256)
        && group["tagSize"].as_u64() == Some(128)
    },
    assert_aegis256_case,
  );
}
