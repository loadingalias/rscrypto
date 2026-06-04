#![cfg(feature = "rsa")]

use rscrypto::{RsaOaepProfile, RsaPkcs1v15Profile, RsaPrivateKey, RsaPssProfile, RsaPublicKey, RsaPublicKeyPolicy};
use serde_json::Value;

const PKCS1_SHA256: &str = include_str!("../testdata/rsa/wycheproof/rsa_signature_2048_sha256_test.json");
const PKCS1_SHA384: &str = include_str!("../testdata/rsa/wycheproof/rsa_signature_2048_sha384_test.json");
const PKCS1_SHA512: &str = include_str!("../testdata/rsa/wycheproof/rsa_signature_2048_sha512_test.json");
const PKCS1_3072_SHA256: &str = include_str!("../testdata/rsa/wycheproof/rsa_signature_3072_sha256_test.json");
const PKCS1_3072_SHA384: &str = include_str!("../testdata/rsa/wycheproof/rsa_signature_3072_sha384_test.json");
const PKCS1_3072_SHA512: &str = include_str!("../testdata/rsa/wycheproof/rsa_signature_3072_sha512_test.json");
const PKCS1_4096_SHA256: &str = include_str!("../testdata/rsa/wycheproof/rsa_signature_4096_sha256_test.json");
const PKCS1_4096_SHA384: &str = include_str!("../testdata/rsa/wycheproof/rsa_signature_4096_sha384_test.json");
const PKCS1_4096_SHA512: &str = include_str!("../testdata/rsa/wycheproof/rsa_signature_4096_sha512_test.json");
const PKCS1_SIG_GEN_2048: &str = include_str!("../testdata/rsa/wycheproof/rsa_pkcs1_2048_sig_gen_test.json");
const PKCS1_SIG_GEN_3072: &str = include_str!("../testdata/rsa/wycheproof/rsa_pkcs1_3072_sig_gen_test.json");
const PKCS1_SIG_GEN_4096: &str = include_str!("../testdata/rsa/wycheproof/rsa_pkcs1_4096_sig_gen_test.json");
const PSS_SHA256: &str = include_str!("../testdata/rsa/wycheproof/rsa_pss_2048_sha256_mgf1_32_test.json");
const PSS_SHA384: &str = include_str!("../testdata/rsa/wycheproof/rsa_pss_2048_sha384_mgf1_48_test.json");
const PSS_3072_SHA256: &str = include_str!("../testdata/rsa/wycheproof/rsa_pss_3072_sha256_mgf1_32_test.json");
const PSS_4096_SHA256: &str = include_str!("../testdata/rsa/wycheproof/rsa_pss_4096_sha256_mgf1_32_test.json");
const PSS_4096_SHA384: &str = include_str!("../testdata/rsa/wycheproof/rsa_pss_4096_sha384_mgf1_48_test.json");
const PSS_4096_SHA512_SALT32: &str = include_str!("../testdata/rsa/wycheproof/rsa_pss_4096_sha512_mgf1_32_test.json");
const PSS_SHA512: &str = include_str!("../testdata/rsa/wycheproof/rsa_pss_4096_sha512_mgf1_64_test.json");
const OAEP_SHA256: &str = include_str!("../testdata/rsa/wycheproof/rsa_oaep_2048_sha256_mgf1sha256_test.json");
const OAEP_SHA384: &str = include_str!("../testdata/rsa/wycheproof/rsa_oaep_2048_sha384_mgf1sha384_test.json");
const OAEP_SHA512: &str = include_str!("../testdata/rsa/wycheproof/rsa_oaep_2048_sha512_mgf1sha512_test.json");
const OAEP_SHA256_MGF1SHA1: &str = include_str!("../testdata/rsa/wycheproof/rsa_oaep_2048_sha256_mgf1sha1_test.json");
const OAEP_SHA384_MGF1SHA1: &str = include_str!("../testdata/rsa/wycheproof/rsa_oaep_2048_sha384_mgf1sha1_test.json");
const OAEP_SHA512_MGF1SHA1: &str = include_str!("../testdata/rsa/wycheproof/rsa_oaep_2048_sha512_mgf1sha1_test.json");
const OAEP_3072_SHA256: &str = include_str!("../testdata/rsa/wycheproof/rsa_oaep_3072_sha256_mgf1sha256_test.json");
const OAEP_3072_SHA256_MGF1SHA1: &str =
  include_str!("../testdata/rsa/wycheproof/rsa_oaep_3072_sha256_mgf1sha1_test.json");
const OAEP_3072_SHA512: &str = include_str!("../testdata/rsa/wycheproof/rsa_oaep_3072_sha512_mgf1sha512_test.json");
const OAEP_3072_SHA512_MGF1SHA1: &str =
  include_str!("../testdata/rsa/wycheproof/rsa_oaep_3072_sha512_mgf1sha1_test.json");
const OAEP_4096_SHA256: &str = include_str!("../testdata/rsa/wycheproof/rsa_oaep_4096_sha256_mgf1sha256_test.json");
const OAEP_4096_SHA256_MGF1SHA1: &str =
  include_str!("../testdata/rsa/wycheproof/rsa_oaep_4096_sha256_mgf1sha1_test.json");
const OAEP_4096_SHA512: &str = include_str!("../testdata/rsa/wycheproof/rsa_oaep_4096_sha512_mgf1sha512_test.json");
const OAEP_4096_SHA512_MGF1SHA1: &str =
  include_str!("../testdata/rsa/wycheproof/rsa_oaep_4096_sha512_mgf1sha1_test.json");
const RSAES_PKCS1_2048: &str = include_str!("../testdata/rsa/wycheproof/rsa_pkcs1_2048_test.json");
const RSAES_PKCS1_3072: &str = include_str!("../testdata/rsa/wycheproof/rsa_pkcs1_3072_test.json");
const RSAES_PKCS1_4096: &str = include_str!("../testdata/rsa/wycheproof/rsa_pkcs1_4096_test.json");

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
  expected_key_size: u64,
  expected_sha: &str,
  expected: ExpectedCounts,
) {
  let suite: Value = serde_json::from_str(json).expect("Wycheproof PKCS1v1.5 JSON must parse");
  let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_small_exponents();
  let mut valid = 0usize;
  let mut invalid = 0usize;
  let mut acceptable = 0usize;

  for group in groups(&suite, "RSASSA-PKCS1-v1_5") {
    assert_eq!(group["keySize"].as_u64(), Some(expected_key_size));
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

fn assert_pss_wycheproof_vectors(
  json: &str,
  profile: RsaPssProfile,
  expected_key_size: u64,
  expected_sha: &str,
  expected_salt_len: usize,
  expected: ExpectedCounts,
) {
  let suite: Value = serde_json::from_str(json).expect("Wycheproof PSS JSON must parse");
  let mut valid = 0usize;
  let mut invalid = 0usize;
  let mut acceptable = 0usize;

  for group in groups(&suite, "RSASSA-PSS") {
    assert_eq!(group["keySize"].as_u64(), Some(expected_key_size));
    assert_eq!(group["sha"].as_str(), Some(expected_sha));
    assert_eq!(group["mgf"].as_str(), Some("MGF1"));
    assert_eq!(group["mgfSha"].as_str(), Some(expected_sha));
    assert_eq!(group["sLen"].as_u64(), Some(expected_salt_len as u64));
    let public_key = hex_to_vec(field(group, "publicKeyDer"));
    let key = RsaPublicKey::from_spki_der(&public_key).expect("Wycheproof RSA-PSS key must parse");
    let mut scratch = key.public_scratch();

    for test in test_cases(group) {
      let msg = hex_to_vec(field(test, "msg"));
      let sig = hex_to_vec(field(test, "sig"));
      let verified = key
        .verify_pss_with_salt_len_and_scratch(profile, expected_salt_len, &msg, &sig, &mut scratch)
        .is_ok();

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

fn pkcs1v15_profile(sha: &str) -> Option<RsaPkcs1v15Profile> {
  match sha {
    "SHA-256" => Some(RsaPkcs1v15Profile::Sha256),
    "SHA-384" => Some(RsaPkcs1v15Profile::Sha384),
    "SHA-512" => Some(RsaPkcs1v15Profile::Sha512),
    _ => None,
  }
}

fn fixed_width_one(len: usize) -> Vec<u8> {
  let mut out = vec![0u8; len];
  *out.last_mut().expect("non-empty RSA modulus") = 1;
  out
}

fn assert_oaep_wycheproof_vectors(
  json: &str,
  profile: RsaOaepProfile,
  expected_key_size: u64,
  expected_sha: &str,
  expected: ExpectedCounts,
) {
  let suite: Value = serde_json::from_str(json).expect("Wycheproof OAEP JSON must parse");
  assert_eq!(suite["algorithm"].as_str(), Some("RSAES-OAEP"));
  assert_eq!(
    suite["numberOfTests"].as_u64(),
    Some(expected.valid.strict_add(expected.invalid) as u64)
  );
  let mut valid = 0usize;
  let mut invalid = 0usize;
  let mut acceptable = 0usize;

  for group in groups(&suite, "RSAES-OAEP") {
    assert_eq!(group["type"].as_str(), Some("RsaesOaepDecrypt"));
    assert_eq!(group["keySize"].as_u64(), Some(expected_key_size));
    assert_eq!(group["sha"].as_str(), Some(expected_sha));
    assert_eq!(group["mgf"].as_str(), Some("MGF1"));
    assert_eq!(group["mgfSha"].as_str(), Some(expected_sha));
    let private_key_der = hex_to_vec(field(group, "privateKeyPkcs8"));
    let key = RsaPrivateKey::from_pkcs8_der(&private_key_der).expect("Wycheproof OAEP private key must parse");
    let blinding_factor = fixed_width_one(key.public_key().modulus().len());
    let blinding_factor_inverse = fixed_width_one(key.public_key().modulus().len());
    let mut scratch = key.private_scratch();
    let mut scratch_valid_checked = false;
    let mut scratch_invalid_checked = false;
    let group_has_valid = test_cases(group).iter().any(|test| field(test, "result") == "valid");
    let group_has_invalid = test_cases(group).iter().any(|test| field(test, "result") == "invalid");

    for test in test_cases(group) {
      let ciphertext = hex_to_vec(field(test, "ct"));
      let label = hex_to_vec(field(test, "label"));
      let mut plaintext = vec![0xa5; key.public_key().modulus().len()];
      let decrypted = key.decrypt_oaep_with_blinding_factor(
        profile,
        &label,
        &ciphertext,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut plaintext,
      );

      match field(test, "result") {
        "valid" => {
          valid = valid.strict_add(1);
          let plaintext_len = decrypted.unwrap_or_else(|error| {
            panic!(
              "Wycheproof OAEP tcId {} rejected valid ciphertext: {error}",
              test["tcId"]
            )
          });
          assert_eq!(
            &plaintext[..plaintext_len],
            hex_to_vec(field(test, "msg")).as_slice(),
            "Wycheproof OAEP tcId {} decrypted to the wrong plaintext",
            test["tcId"]
          );
          if !scratch_valid_checked {
            let mut scratch_plaintext = vec![0u8; key.public_key().modulus().len()];
            let scratch_plaintext_len = key
              .decrypt_oaep_with_blinding_factor_and_scratch(
                profile,
                &label,
                &ciphertext,
                &blinding_factor,
                &blinding_factor_inverse,
                &mut scratch_plaintext,
                &mut scratch,
              )
              .expect("Wycheproof OAEP valid ciphertext must decrypt with caller-owned scratch");
            assert_eq!(
              &scratch_plaintext[..scratch_plaintext_len],
              hex_to_vec(field(test, "msg")).as_slice(),
              "Wycheproof OAEP tcId {} scratch decrypt produced the wrong plaintext",
              test["tcId"]
            );
            scratch_valid_checked = true;
          }
        }
        "invalid" => {
          invalid = invalid.strict_add(1);
          assert!(
            decrypted.is_err(),
            "Wycheproof OAEP tcId {} accepted invalid ciphertext: {}",
            test["tcId"],
            field(test, "comment")
          );
          assert!(
            plaintext.iter().all(|&byte| byte == 0),
            "Wycheproof OAEP tcId {} decrypt must clear plaintext on failure",
            test["tcId"]
          );
          if !scratch_invalid_checked {
            let mut scratch_plaintext = vec![0xa5; key.public_key().modulus().len()];
            assert!(
              key
                .decrypt_oaep_with_blinding_factor_and_scratch(
                  profile,
                  &label,
                  &ciphertext,
                  &blinding_factor,
                  &blinding_factor_inverse,
                  &mut scratch_plaintext,
                  &mut scratch,
                )
                .is_err(),
              "Wycheproof OAEP tcId {} scratch decrypt accepted invalid ciphertext: {}",
              test["tcId"],
              field(test, "comment")
            );
            assert!(
              scratch_plaintext.iter().all(|&byte| byte == 0),
              "Wycheproof OAEP tcId {} scratch decrypt must clear plaintext on failure",
              test["tcId"]
            );
            scratch_invalid_checked = true;
          }
        }
        "acceptable" => {
          acceptable = acceptable.strict_add(1);
        }
        other => panic!("unknown Wycheproof result `{other}`"),
      }
    }
    if group_has_valid {
      assert!(
        scratch_valid_checked,
        "Wycheproof OAEP group for {expected_key_size}/{expected_sha} must exercise a valid scratch decrypt"
      );
    }
    if group_has_invalid {
      assert!(
        scratch_invalid_checked,
        "Wycheproof OAEP group for {expected_key_size}/{expected_sha} must exercise an invalid scratch decrypt"
      );
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

fn assert_oaep_mgf1sha1_vectors_are_rejected(
  json: &str,
  profile: RsaOaepProfile,
  expected_key_size: u64,
  expected_sha: &str,
  expected: ExpectedCounts,
) {
  let suite: Value = serde_json::from_str(json).expect("Wycheproof OAEP MGF mismatch JSON must parse");
  assert_eq!(suite["algorithm"].as_str(), Some("RSAES-OAEP"));
  assert_eq!(
    suite["numberOfTests"].as_u64(),
    Some(expected.valid.strict_add(expected.invalid) as u64)
  );
  let mut valid = 0usize;
  let mut invalid = 0usize;
  let mut acceptable = 0usize;

  for group in groups(&suite, "RSAES-OAEP") {
    assert_eq!(group["type"].as_str(), Some("RsaesOaepDecrypt"));
    assert_eq!(group["keySize"].as_u64(), Some(expected_key_size));
    assert_eq!(group["sha"].as_str(), Some(expected_sha));
    assert_eq!(group["mgf"].as_str(), Some("MGF1"));
    assert_eq!(group["mgfSha"].as_str(), Some("SHA-1"));
    let private_key_der = hex_to_vec(field(group, "privateKeyPkcs8"));
    let key = RsaPrivateKey::from_pkcs8_der(&private_key_der).expect("Wycheproof OAEP private key must parse");
    let blinding_factor = fixed_width_one(key.public_key().modulus().len());
    let blinding_factor_inverse = fixed_width_one(key.public_key().modulus().len());
    let mut scratch = key.private_scratch();

    for test in test_cases(group) {
      let ciphertext = hex_to_vec(field(test, "ct"));
      let label = hex_to_vec(field(test, "label"));
      let mut plaintext = vec![0u8; key.public_key().modulus().len()];
      let decrypted = key.decrypt_oaep_with_blinding_factor(
        profile,
        &label,
        &ciphertext,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut plaintext,
      );
      let mut scratch_plaintext = vec![0xa5; key.public_key().modulus().len()];
      let scratch_decrypted = key.decrypt_oaep_with_blinding_factor_and_scratch(
        profile,
        &label,
        &ciphertext,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut scratch_plaintext,
        &mut scratch,
      );

      match field(test, "result") {
        "valid" => {
          valid = valid.strict_add(1);
          assert!(
            decrypted.is_err(),
            "Wycheproof OAEP tcId {} used unsupported MGF1-SHA1 but decrypted under {expected_sha}",
            test["tcId"]
          );
          assert!(
            scratch_decrypted.is_err(),
            "Wycheproof OAEP tcId {} used unsupported MGF1-SHA1 but scratch decrypted under {expected_sha}",
            test["tcId"]
          );
        }
        "invalid" => {
          invalid = invalid.strict_add(1);
          assert!(
            decrypted.is_err(),
            "Wycheproof OAEP tcId {} accepted invalid MGF1-SHA1 ciphertext: {}",
            test["tcId"],
            field(test, "comment")
          );
          assert!(
            scratch_decrypted.is_err(),
            "Wycheproof OAEP tcId {} scratch accepted invalid MGF1-SHA1 ciphertext: {}",
            test["tcId"],
            field(test, "comment")
          );
        }
        "acceptable" => {
          acceptable = acceptable.strict_add(1);
          assert!(
            decrypted.is_err(),
            "Wycheproof OAEP tcId {} accepted acceptable MGF1-SHA1 ciphertext under unsupported profile",
            test["tcId"]
          );
          assert!(
            scratch_decrypted.is_err(),
            "Wycheproof OAEP tcId {} scratch accepted acceptable MGF1-SHA1 ciphertext under unsupported profile",
            test["tcId"]
          );
        }
        other => panic!("unknown Wycheproof result `{other}`"),
      }
      assert!(
        plaintext.iter().all(|&byte| byte == 0),
        "Wycheproof OAEP tcId {} unsupported-MGF reject must clear plaintext",
        test["tcId"]
      );
      assert!(
        scratch_plaintext.iter().all(|&byte| byte == 0),
        "Wycheproof OAEP tcId {} unsupported-MGF scratch reject must clear plaintext",
        test["tcId"]
      );
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

fn assert_rsaes_pkcs1v15_wycheproof_vectors(json: &str, expected_key_size: u64, expected: ExpectedCounts) {
  let suite: Value = serde_json::from_str(json).expect("Wycheproof RSAES-PKCS1-v1_5 JSON must parse");
  assert_eq!(suite["algorithm"].as_str(), Some("RSAES-PKCS1-v1_5"));
  assert_eq!(
    suite["numberOfTests"].as_u64(),
    Some(expected.valid.strict_add(expected.invalid) as u64)
  );
  let mut valid = 0usize;
  let mut invalid = 0usize;
  let mut acceptable = 0usize;

  for group in groups(&suite, "RSAES-PKCS1-v1_5") {
    assert_eq!(group["type"].as_str(), Some("RsaesPkcs1Decrypt"));
    assert_eq!(group["keySize"].as_u64(), Some(expected_key_size));
    let private_key_der = hex_to_vec(field(group, "privateKeyPkcs8"));
    let key = RsaPrivateKey::from_pkcs8_der(&private_key_der).expect("Wycheproof RSAES-PKCS1-v1_5 key must parse");
    let blinding_factor = fixed_width_one(key.public_key().modulus().len());
    let blinding_factor_inverse = fixed_width_one(key.public_key().modulus().len());
    let mut scratch = key.private_scratch();
    let mut scratch_valid_checked = false;
    let mut scratch_invalid_checked = false;
    let group_has_valid = test_cases(group).iter().any(|test| field(test, "result") == "valid");
    let group_has_invalid = test_cases(group).iter().any(|test| field(test, "result") == "invalid");

    for test in test_cases(group) {
      let ciphertext = hex_to_vec(field(test, "ct"));
      let mut plaintext = vec![0xa5; key.public_key().modulus().len()];
      let decrypted = key.decrypt_pkcs1v15_with_blinding_factor(
        &ciphertext,
        &blinding_factor,
        &blinding_factor_inverse,
        &mut plaintext,
      );

      match field(test, "result") {
        "valid" => {
          valid = valid.strict_add(1);
          let plaintext_len = decrypted.unwrap_or_else(|error| {
            panic!(
              "Wycheproof RSAES-PKCS1-v1_5 tcId {} rejected valid ciphertext: {error}",
              test["tcId"]
            )
          });
          assert_eq!(
            &plaintext[..plaintext_len],
            hex_to_vec(field(test, "msg")).as_slice(),
            "Wycheproof RSAES-PKCS1-v1_5 tcId {} decrypted to the wrong plaintext",
            test["tcId"]
          );
          if !scratch_valid_checked {
            let mut scratch_plaintext = vec![0u8; key.public_key().modulus().len()];
            let scratch_plaintext_len = key
              .decrypt_pkcs1v15_with_blinding_factor_and_scratch(
                &ciphertext,
                &blinding_factor,
                &blinding_factor_inverse,
                &mut scratch_plaintext,
                &mut scratch,
              )
              .expect("Wycheproof RSAES-PKCS1-v1_5 valid ciphertext must decrypt with caller-owned scratch");
            assert_eq!(
              &scratch_plaintext[..scratch_plaintext_len],
              hex_to_vec(field(test, "msg")).as_slice(),
              "Wycheproof RSAES-PKCS1-v1_5 tcId {} scratch decrypt produced the wrong plaintext",
              test["tcId"]
            );
            scratch_valid_checked = true;
          }
        }
        "invalid" => {
          invalid = invalid.strict_add(1);
          assert!(
            decrypted.is_err(),
            "Wycheproof RSAES-PKCS1-v1_5 tcId {} accepted invalid ciphertext: {}",
            test["tcId"],
            field(test, "comment")
          );
          assert!(
            plaintext.iter().all(|&byte| byte == 0),
            "Wycheproof RSAES-PKCS1-v1_5 tcId {} decrypt must clear plaintext on failure",
            test["tcId"]
          );
          if !scratch_invalid_checked {
            let mut scratch_plaintext = vec![0xa5; key.public_key().modulus().len()];
            assert!(
              key
                .decrypt_pkcs1v15_with_blinding_factor_and_scratch(
                  &ciphertext,
                  &blinding_factor,
                  &blinding_factor_inverse,
                  &mut scratch_plaintext,
                  &mut scratch,
                )
                .is_err(),
              "Wycheproof RSAES-PKCS1-v1_5 tcId {} scratch decrypt accepted invalid ciphertext: {}",
              test["tcId"],
              field(test, "comment")
            );
            assert!(
              scratch_plaintext.iter().all(|&byte| byte == 0),
              "Wycheproof RSAES-PKCS1-v1_5 tcId {} scratch decrypt must clear plaintext on failure",
              test["tcId"]
            );
            scratch_invalid_checked = true;
          }
        }
        "acceptable" => {
          acceptable = acceptable.strict_add(1);
        }
        other => panic!("unknown Wycheproof result `{other}`"),
      }
    }
    if group_has_valid {
      assert!(
        scratch_valid_checked,
        "Wycheproof RSAES-PKCS1-v1_5 group for {expected_key_size} must exercise a valid scratch decrypt"
      );
    }
    if group_has_invalid {
      assert!(
        scratch_invalid_checked,
        "Wycheproof RSAES-PKCS1-v1_5 group for {expected_key_size} must exercise an invalid scratch decrypt"
      );
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

fn assert_pkcs1v15_sig_gen_wycheproof_vectors(json: &str, expected_key_size: u64, expected_valid: usize) {
  let suite: Value = serde_json::from_str(json).expect("Wycheproof PKCS1v1.5 SigGen JSON must parse");
  assert_eq!(suite["algorithm"].as_str(), Some("RSASSA-PKCS1-v1_5"));
  let mut valid = 0usize;
  let mut skipped = 0usize;

  for group in groups(&suite, "RSASSA-PKCS1-v1_5") {
    assert_eq!(group["type"].as_str(), Some("RsassaPkcs1Generate"));
    assert_eq!(group["keySize"].as_u64(), Some(expected_key_size));
    let Some(profile) = pkcs1v15_profile(field(group, "sha")) else {
      skipped = skipped.strict_add(test_cases(group).len());
      continue;
    };
    if test_cases(group).iter().all(|test| field(test, "result") != "valid") {
      skipped = skipped.strict_add(test_cases(group).len());
      continue;
    }

    let private_key_der = hex_to_vec(field(group, "privateKeyPkcs8"));
    let key = RsaPrivateKey::from_pkcs8_der_with_policy(&private_key_der, &RsaPublicKeyPolicy::legacy_verification())
      .expect("Wycheproof PKCS1v1.5 private key must parse");
    assert_eq!(key.public_key().modulus_bits() as u64, expected_key_size);
    let blinding_factor = fixed_width_one(key.public_key().modulus().len());
    let blinding_factor_inverse = fixed_width_one(key.public_key().modulus().len());

    for test in test_cases(group) {
      match field(test, "result") {
        "valid" => {
          valid = valid.strict_add(1);
          let message = hex_to_vec(field(test, "msg"));
          let expected_signature = hex_to_vec(field(test, "sig"));
          let mut signature = vec![0u8; key.public_key().modulus().len()];
          let mut scratch_signature = vec![0u8; key.public_key().modulus().len()];
          let mut scratch = key.private_scratch();
          key
            .sign_pkcs1v15_with_blinding_factor(
              profile,
              &message,
              &blinding_factor,
              &blinding_factor_inverse,
              &mut signature,
            )
            .expect("Wycheproof PKCS1v1.5 private signing must succeed");
          assert_eq!(
            signature, expected_signature,
            "Wycheproof PKCS1v1.5 SigGen tcId {} generated the wrong signature",
            test["tcId"]
          );
          key
            .sign_pkcs1v15_with_blinding_factor_and_scratch(
              profile,
              &message,
              &blinding_factor,
              &blinding_factor_inverse,
              &mut scratch_signature,
              &mut scratch,
            )
            .expect("Wycheproof PKCS1v1.5 scratch private signing must succeed");
          assert_eq!(
            scratch_signature, expected_signature,
            "Wycheproof PKCS1v1.5 SigGen tcId {} scratch signing generated the wrong signature",
            test["tcId"]
          );
          key
            .public_key()
            .verify_pkcs1v15(profile, &message, &signature)
            .expect("Wycheproof PKCS1v1.5 generated signature must verify");
        }
        "acceptable" | "invalid" => {
          skipped = skipped.strict_add(1);
        }
        other => panic!("unknown Wycheproof result `{other}`"),
      }
    }
  }

  assert_eq!(valid, expected_valid);
  assert_eq!(
    valid.strict_add(skipped),
    suite["numberOfTests"].as_u64().unwrap() as usize
  );
}

#[test]
fn wycheproof_pkcs1v15_sha2_vectors_match_expected_results() {
  assert_pkcs1v15_wycheproof_vectors(
    PKCS1_SHA256,
    RsaPkcs1v15Profile::Sha256,
    2048,
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
    2048,
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
    2048,
    "SHA-512",
    ExpectedCounts {
      valid: 8,
      acceptable: 1,
      invalid: 250,
    },
  );
  assert_pkcs1v15_wycheproof_vectors(
    PKCS1_3072_SHA256,
    RsaPkcs1v15Profile::Sha256,
    3072,
    "SHA-256",
    ExpectedCounts {
      valid: 8,
      acceptable: 1,
      invalid: 250,
    },
  );
  assert_pkcs1v15_wycheproof_vectors(
    PKCS1_3072_SHA384,
    RsaPkcs1v15Profile::Sha384,
    3072,
    "SHA-384",
    ExpectedCounts {
      valid: 7,
      acceptable: 1,
      invalid: 251,
    },
  );
  assert_pkcs1v15_wycheproof_vectors(
    PKCS1_3072_SHA512,
    RsaPkcs1v15Profile::Sha512,
    3072,
    "SHA-512",
    ExpectedCounts {
      valid: 8,
      acceptable: 1,
      invalid: 251,
    },
  );
  assert_pkcs1v15_wycheproof_vectors(
    PKCS1_4096_SHA256,
    RsaPkcs1v15Profile::Sha256,
    4096,
    "SHA-256",
    ExpectedCounts {
      valid: 7,
      acceptable: 1,
      invalid: 250,
    },
  );
  assert_pkcs1v15_wycheproof_vectors(
    PKCS1_4096_SHA384,
    RsaPkcs1v15Profile::Sha384,
    4096,
    "SHA-384",
    ExpectedCounts {
      valid: 7,
      acceptable: 1,
      invalid: 251,
    },
  );
  assert_pkcs1v15_wycheproof_vectors(
    PKCS1_4096_SHA512,
    RsaPkcs1v15Profile::Sha512,
    4096,
    "SHA-512",
    ExpectedCounts {
      valid: 7,
      acceptable: 1,
      invalid: 251,
    },
  );
}

#[test]
fn wycheproof_pkcs1v15_sha2_sig_gen_vectors_match_expected_signatures() {
  assert_pkcs1v15_sig_gen_wycheproof_vectors(PKCS1_SIG_GEN_2048, 2048, 24);
  assert_pkcs1v15_sig_gen_wycheproof_vectors(PKCS1_SIG_GEN_3072, 3072, 24);
  assert_pkcs1v15_sig_gen_wycheproof_vectors(PKCS1_SIG_GEN_4096, 4096, 24);
}

#[test]
fn wycheproof_rsaes_pkcs1v15_decrypt_vectors_match_expected_results() {
  assert_rsaes_pkcs1v15_wycheproof_vectors(
    RSAES_PKCS1_2048,
    2048,
    ExpectedCounts {
      valid: 42,
      acceptable: 0,
      invalid: 25,
    },
  );
  assert_rsaes_pkcs1v15_wycheproof_vectors(
    RSAES_PKCS1_3072,
    3072,
    ExpectedCounts {
      valid: 41,
      acceptable: 0,
      invalid: 26,
    },
  );
  assert_rsaes_pkcs1v15_wycheproof_vectors(
    RSAES_PKCS1_4096,
    4096,
    ExpectedCounts {
      valid: 41,
      acceptable: 0,
      invalid: 26,
    },
  );
}

#[test]
fn wycheproof_oaep_sha2_decrypt_vectors_match_expected_results() {
  assert_oaep_wycheproof_vectors(
    OAEP_SHA256,
    RsaOaepProfile::Sha256,
    2048,
    "SHA-256",
    ExpectedCounts {
      valid: 18,
      acceptable: 0,
      invalid: 19,
    },
  );
  assert_oaep_wycheproof_vectors(
    OAEP_SHA384,
    RsaOaepProfile::Sha384,
    2048,
    "SHA-384",
    ExpectedCounts {
      valid: 16,
      acceptable: 0,
      invalid: 18,
    },
  );
  assert_oaep_wycheproof_vectors(
    OAEP_SHA512,
    RsaOaepProfile::Sha512,
    2048,
    "SHA-512",
    ExpectedCounts {
      valid: 14,
      acceptable: 0,
      invalid: 19,
    },
  );
  assert_oaep_wycheproof_vectors(
    OAEP_3072_SHA256,
    RsaOaepProfile::Sha256,
    3072,
    "SHA-256",
    ExpectedCounts {
      valid: 18,
      acceptable: 0,
      invalid: 19,
    },
  );
  assert_oaep_wycheproof_vectors(
    OAEP_3072_SHA512,
    RsaOaepProfile::Sha512,
    3072,
    "SHA-512",
    ExpectedCounts {
      valid: 15,
      acceptable: 0,
      invalid: 18,
    },
  );
  assert_oaep_wycheproof_vectors(
    OAEP_4096_SHA256,
    RsaOaepProfile::Sha256,
    4096,
    "SHA-256",
    ExpectedCounts {
      valid: 18,
      acceptable: 0,
      invalid: 19,
    },
  );
  assert_oaep_wycheproof_vectors(
    OAEP_4096_SHA512,
    RsaOaepProfile::Sha512,
    4096,
    "SHA-512",
    ExpectedCounts {
      valid: 17,
      acceptable: 0,
      invalid: 19,
    },
  );
}

#[test]
fn wycheproof_oaep_mgf1sha1_vectors_are_rejected_by_sha2_mgf1sha2_profiles() {
  assert_oaep_mgf1sha1_vectors_are_rejected(
    OAEP_SHA256_MGF1SHA1,
    RsaOaepProfile::Sha256,
    2048,
    "SHA-256",
    ExpectedCounts {
      valid: 13,
      acceptable: 0,
      invalid: 18,
    },
  );
  assert_oaep_mgf1sha1_vectors_are_rejected(
    OAEP_SHA384_MGF1SHA1,
    RsaOaepProfile::Sha384,
    2048,
    "SHA-384",
    ExpectedCounts {
      valid: 13,
      acceptable: 0,
      invalid: 18,
    },
  );
  assert_oaep_mgf1sha1_vectors_are_rejected(
    OAEP_SHA512_MGF1SHA1,
    RsaOaepProfile::Sha512,
    2048,
    "SHA-512",
    ExpectedCounts {
      valid: 13,
      acceptable: 0,
      invalid: 18,
    },
  );
  assert_oaep_mgf1sha1_vectors_are_rejected(
    OAEP_3072_SHA256_MGF1SHA1,
    RsaOaepProfile::Sha256,
    3072,
    "SHA-256",
    ExpectedCounts {
      valid: 13,
      acceptable: 0,
      invalid: 19,
    },
  );
  assert_oaep_mgf1sha1_vectors_are_rejected(
    OAEP_3072_SHA512_MGF1SHA1,
    RsaOaepProfile::Sha512,
    3072,
    "SHA-512",
    ExpectedCounts {
      valid: 13,
      acceptable: 0,
      invalid: 18,
    },
  );
  assert_oaep_mgf1sha1_vectors_are_rejected(
    OAEP_4096_SHA256_MGF1SHA1,
    RsaOaepProfile::Sha256,
    4096,
    "SHA-256",
    ExpectedCounts {
      valid: 13,
      acceptable: 0,
      invalid: 19,
    },
  );
  assert_oaep_mgf1sha1_vectors_are_rejected(
    OAEP_4096_SHA512_MGF1SHA1,
    RsaOaepProfile::Sha512,
    4096,
    "SHA-512",
    ExpectedCounts {
      valid: 13,
      acceptable: 0,
      invalid: 18,
    },
  );
}

#[test]
fn wycheproof_pss_sha2_mgf1_digest_salt_vectors_match_expected_results() {
  assert_pss_wycheproof_vectors(
    PSS_SHA256,
    RsaPssProfile::Sha256,
    2048,
    "SHA-256",
    RsaPssProfile::Sha256.digest_len(),
    ExpectedCounts {
      valid: 63,
      acceptable: 0,
      invalid: 45,
    },
  );
  assert_pss_wycheproof_vectors(
    PSS_SHA384,
    RsaPssProfile::Sha384,
    2048,
    "SHA-384",
    RsaPssProfile::Sha384.digest_len(),
    ExpectedCounts {
      valid: 95,
      acceptable: 0,
      invalid: 46,
    },
  );
  assert_pss_wycheproof_vectors(
    PSS_3072_SHA256,
    RsaPssProfile::Sha256,
    3072,
    "SHA-256",
    RsaPssProfile::Sha256.digest_len(),
    ExpectedCounts {
      valid: 63,
      acceptable: 0,
      invalid: 45,
    },
  );
  assert_pss_wycheproof_vectors(
    PSS_4096_SHA256,
    RsaPssProfile::Sha256,
    4096,
    "SHA-256",
    RsaPssProfile::Sha256.digest_len(),
    ExpectedCounts {
      valid: 63,
      acceptable: 0,
      invalid: 45,
    },
  );
  assert_pss_wycheproof_vectors(
    PSS_4096_SHA384,
    RsaPssProfile::Sha384,
    4096,
    "SHA-384",
    RsaPssProfile::Sha384.digest_len(),
    ExpectedCounts {
      valid: 95,
      acceptable: 0,
      invalid: 46,
    },
  );
  assert_pss_wycheproof_vectors(
    PSS_SHA512,
    RsaPssProfile::Sha512,
    4096,
    "SHA-512",
    RsaPssProfile::Sha512.digest_len(),
    ExpectedCounts {
      valid: 132,
      acceptable: 0,
      invalid: 47,
    },
  );
  assert_pss_wycheproof_vectors(
    PSS_4096_SHA512_SALT32,
    RsaPssProfile::Sha512,
    4096,
    "SHA-512",
    RsaPssProfile::Sha256.digest_len(),
    ExpectedCounts {
      valid: 132,
      acceptable: 0,
      invalid: 45,
    },
  );
}
