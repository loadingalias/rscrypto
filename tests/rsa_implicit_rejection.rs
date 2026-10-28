#![cfg(feature = "rsa")]

//! RSAES-PKCS1-v1_5 implicit rejection against independent expectations.
//!
//! `testdata/rsa/implicit_rejection/` holds Appendix B of the pinned
//! draft-irtf-cfrg-rsa-guidance-10 and OpenSSL 4.0.3 decryptions of seeded
//! random ciphertexts and the Wycheproof invalid-padding cases; see its README.

extern crate alloc;

use alloc::collections::BTreeSet;
use core::ops::Range;
use std::collections::HashMap;

use rscrypto::{
  RsaBlindingPair, RsaKeyError, RsaPrivateKey, RsaPrivateOpError, RsaPublicKeyPolicy,
  auth::rsa::expert::RsaPkcs1v15Decrypt,
};
use serde_json::Value;

const APPENDIX_B: &str = include_str!("../testdata/rsa/implicit_rejection/appendix_b.json");
const OPENSSL: &str = include_str!("../testdata/rsa/implicit_rejection/openssl_4_0_3.json");
const WYCHEPROOF: [(&str, &str); 3] = [
  (
    "rsa_pkcs1_2048_test.json",
    include_str!("../testdata/rsa/wycheproof/rsa_pkcs1_2048_test.json"),
  ),
  (
    "rsa_pkcs1_3072_test.json",
    include_str!("../testdata/rsa/wycheproof/rsa_pkcs1_3072_test.json"),
  ),
  (
    "rsa_pkcs1_4096_test.json",
    include_str!("../testdata/rsa/wycheproof/rsa_pkcs1_4096_test.json"),
  ),
];
const PKCS1V15_OVERHEAD: usize = 11;
/// Appendix B.2's 2049-bit key fails the existing half-modulus-width factor
/// policy. Unit tests cover its depadding below that boundary; these integration
/// tests assert its public import rejection.
const ODD_WIDTH_KEY: &str = "B.2";

fn hex(value: &Value) -> Vec<u8> {
  let text = value.as_str().expect("vector field must be a hex string");
  let (pairs, remainder) = text.as_bytes().as_chunks::<2>();
  assert!(remainder.is_empty(), "hex string must contain whole octets");
  pairs
    .iter()
    .map(|pair| {
      u8::from_str_radix(core::str::from_utf8(pair).expect("ASCII hex"), 16).expect("vector field must be hex")
    })
    .collect()
}

fn try_import(pkcs8: &[u8]) -> Result<RsaPrivateKey, RsaKeyError> {
  RsaPrivateKey::from_pkcs8_der_with_policy(pkcs8, &RsaPublicKeyPolicy::modulus_2048_to_8192())
}

fn import(pkcs8: &[u8]) -> RsaPrivateKey {
  try_import(pkcs8).expect("vector private key must import")
}

/// Import the Appendix B keys by section, requiring the odd-width key's rejection.
fn appendix_b_keys(vectors: &Value) -> HashMap<&str, RsaPrivateKey> {
  let keys = vectors["keys"].as_array().expect("Appendix B keys");
  assert_eq!(keys.len(), 4);
  let mut imported = HashMap::new();
  for vector in keys {
    let section = vector["section"].as_str().expect("key section");
    let pkcs8 = hex(&vector["pkcs8"]);
    if section == ODD_WIDTH_KEY {
      assert_eq!(vector["modulus_bits"].as_u64(), Some(2049));
      assert_eq!(try_import(&pkcs8).err(), Some(RsaKeyError::InvalidModulus));
      continue;
    }
    let key = import(&pkcs8);
    assert_eq!(
      Some(u64::try_from(key.public_key().modulus_bits()).expect("modulus bits fit")),
      vector["modulus_bits"].as_u64()
    );
    imported.insert(section, key);
  }
  imported
}

/// Decrypt through every entry point and require one result, with the rest of
/// the `k - 11` window zero and the bytes after it untouched.
fn decrypt_everywhere(key: &RsaPrivateKey, ciphertext: &[u8], check: impl FnOnce(Result<Vec<u8>, RsaPrivateOpError>)) {
  let k = key.public_key().modulus().len();
  let window = k.strict_sub(PKCS1V15_OVERHEAD);
  let mut one = vec![0u8; k];
  one[k.strict_sub(1)] = 1;
  let blinding = RsaBlindingPair::new(&one, &one);
  let mut scratch = key.private_scratch();
  let mut results: Vec<Result<Vec<u8>, RsaPrivateOpError>> = Vec::new();
  let mut record = |out: &[u8], result: Result<usize, RsaPrivateOpError>| {
    results.push(result.map(|len| {
      assert!(len <= window, "message length {len} exceeds k - 11");
      assert!(
        out[len..window].iter().all(|&byte| byte == 0),
        "the window after the message must be zero"
      );
      assert!(
        out[window..].iter().all(|&byte| byte == 0xa5),
        "bytes after the window must stay untouched"
      );
      out[..len].to_vec()
    }));
    if results.last().is_some_and(Result::is_err) {
      assert!(
        out.iter().all(|&byte| byte == 0),
        "a failed decryption must clear `out`"
      );
    }
  };

  let mut out = vec![0xa5u8; k];
  let result = key.decrypt_pkcs1v15_with_blinding_factor(ciphertext, blinding, &mut out);
  record(&out, result);
  out.fill(0xa5);
  let result = key.decrypt_pkcs1v15_with_blinding_factor_and_scratch(ciphertext, blinding, &mut out, &mut scratch);
  record(&out, result);
  #[cfg(feature = "getrandom")]
  {
    out.fill(0xa5);
    let result = key.decrypt_pkcs1v15(ciphertext, &mut out);
    record(&out, result);
    out.fill(0xa5);
    let result = key.decrypt_pkcs1v15_with_scratch(ciphertext, &mut out, &mut scratch);
    record(&out, result);
  }
  let first = results[0].clone();
  assert!(
    results.iter().all(|result| *result == first),
    "entry points disagree: {results:?}"
  );
  check(first);
}

#[test]
fn appendix_b_vectors_decrypt_through_every_entry_point() {
  let vectors: Value = serde_json::from_str(APPENDIX_B).expect("Appendix B fixture must parse");
  let keys = appendix_b_keys(&vectors);
  let mut cases = 0usize;
  for vector in vectors["keys"].as_array().expect("Appendix B keys") {
    let Some(key) = keys.get(vector["section"].as_str().expect("key section")) else {
      continue;
    };
    for case in vector["cases"].as_array().expect("Appendix B cases") {
      decrypt_everywhere(key, &hex(&case["ciphertext"]), |decrypted| {
        assert_eq!(
          decrypted,
          Ok(hex(&case["message"])),
          "{} {}",
          case["section"],
          case["name"]
        );
      });
      cases += 1;
    }
  }
  assert_eq!(cases, 36, "twelve cases for each of the three importable keys");
}

#[test]
fn random_ciphertexts_match_openssl_implicit_rejection() {
  let appendix: Value = serde_json::from_str(APPENDIX_B).expect("Appendix B fixture must parse");
  let keys = appendix_b_keys(&appendix);
  let oracle: Value = serde_json::from_str(OPENSSL).expect("OpenSSL fixture must parse");
  let cases = oracle["random"].as_array().expect("random cases");
  assert_eq!(cases.len(), 128);
  let mut lengths = BTreeSet::new();
  let mut checked = 0usize;
  for case in cases {
    let Some(key) = keys.get(case["key"].as_str().expect("case key")) else {
      continue;
    };
    checked += 1;
    decrypt_everywhere(key, &hex(&case["ciphertext"]), |decrypted| {
      let message = decrypted.expect("a reduced ciphertext must decrypt");
      assert_eq!(message, hex(&case["message"]), "random ciphertext for {}", case["key"]);
      lengths.insert(message.len());
    });
  }
  assert_eq!(
    checked, 96,
    "32 seeded ciphertexts for each of the three importable keys"
  );
  // The synthetic lengths spread over the admissible range.
  assert!(lengths.len() > 64, "synthetic lengths must vary: {lengths:?}");
}

fn check_wycheproof(suite_index: usize, group_range: Range<usize>, counts: (usize, usize, usize)) {
  let oracle: Value = serde_json::from_str(OPENSSL).expect("OpenSSL fixture must parse");
  let expected: HashMap<(&str, u64), Vec<u8>> = oracle["wycheproof"]
    .as_array()
    .expect("Wycheproof cases")
    .iter()
    .map(|case| {
      (
        (
          case["file"].as_str().expect("file"),
          case["tcId"].as_u64().expect("tcId"),
        ),
        hex(&case["message"]),
      )
    })
    .collect();
  assert_eq!(
    expected.len(),
    59,
    "every OpenSSL padding-failure expectation must remain present"
  );
  let (file, json) = WYCHEPROOF[suite_index];
  let suite: Value = serde_json::from_str(json).expect("Wycheproof fixture must parse");
  let groups = suite["testGroups"].as_array().expect("groups");
  assert_eq!(groups.len(), 33);
  assert_eq!(suite["numberOfTests"].as_u64(), Some(67));
  assert_eq!(
    groups
      .iter()
      .map(|group| group["tests"].as_array().expect("tests").len())
      .sum::<usize>(),
    67
  );
  let (mut valid, mut padding, mut format) = (0usize, 0usize, 0usize);
  for group in groups.get(group_range).expect("complete Wycheproof group range") {
    let key = import(&hex(&group["privateKeyPkcs8"]));
    for test in group["tests"].as_array().expect("tests") {
      let flags: Vec<&str> = test["flags"]
        .as_array()
        .expect("flags")
        .iter()
        .map(|flag| flag.as_str().expect("flag"))
        .collect();
      let id = test["tcId"].as_u64().expect("tcId");
      decrypt_everywhere(&key, &hex(&test["ct"]), |result| {
        if flags.contains(&"InvalidPkcs1Padding") {
          assert_eq!(
            result.expect("a padding failure must decrypt to the synthetic message"),
            expected[&(file, id)],
            "{file} tcId {id}"
          );
          padding = padding.strict_add(1);
        } else if flags.contains(&"InvalidCiphertextFormat") {
          assert!(
            matches!(
              result,
              Err(RsaPrivateOpError::InvalidLength | RsaPrivateOpError::RepresentativeOutOfRange)
            ),
            "{file} tcId {id}: {result:?}"
          );
          format = format.strict_add(1);
        } else {
          valid = valid.strict_add(1);
          assert_eq!(
            result.expect("a valid ciphertext must decrypt"),
            hex(&test["msg"]),
            "{file} tcId {id}"
          );
        }
      });
    }
  }
  assert_eq!((valid, padding, format), counts, "{file}: every selected case must run");
}

// Keep the independent oracle checks within the existing per-test deadline on
// native RISC-V. These disjoint group ranges cover all 201 cases and all 99 keys;
// every selected ciphertext still exercises every decryption entry point.
#[test]
fn wycheproof_2048_matches_openssl_and_preserves_format_errors() {
  check_wycheproof(0, 0..33, (42, 19, 6));
}

#[test]
fn wycheproof_3072_matches_openssl_and_preserves_format_errors() {
  check_wycheproof(1, 0..33, (41, 20, 6));
}

#[test]
fn wycheproof_4096_cases_1_to_35_match_openssl_and_preserve_format_errors() {
  check_wycheproof(2, 0..1, (10, 19, 6));
}

#[test]
fn wycheproof_4096_cases_36_to_39_match_expected_messages() {
  check_wycheproof(2, 1..5, (4, 0, 0));
}

#[test]
fn wycheproof_4096_cases_40_to_43_match_expected_messages() {
  check_wycheproof(2, 5..9, (4, 0, 0));
}

#[test]
fn wycheproof_4096_cases_44_to_47_match_expected_messages() {
  check_wycheproof(2, 9..13, (4, 0, 0));
}

#[test]
fn wycheproof_4096_cases_48_to_51_match_expected_messages() {
  check_wycheproof(2, 13..17, (4, 0, 0));
}

#[test]
fn wycheproof_4096_cases_52_to_55_match_expected_messages() {
  check_wycheproof(2, 17..21, (4, 0, 0));
}

#[test]
fn wycheproof_4096_cases_56_to_59_match_openssl_and_expected_messages() {
  check_wycheproof(2, 21..25, (3, 1, 0));
}

#[test]
fn wycheproof_4096_cases_60_to_63_match_expected_messages() {
  check_wycheproof(2, 25..29, (4, 0, 0));
}

#[test]
fn wycheproof_4096_cases_64_to_67_match_expected_messages() {
  check_wycheproof(2, 29..33, (4, 0, 0));
}
