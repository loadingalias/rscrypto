#![cfg(feature = "rsa")]

extern crate alloc;

use alloc::collections::BTreeMap;

use rscrypto::{
  RsaBlindingPair, RsaKeyError, RsaPkcs1v15Profile, RsaPrivateKey, RsaPrivateKeyParts, RsaPssProfile, RsaPublicKey,
  RsaPublicKeyPolicy, RsaSignatureProfile,
};
use serde_json::Value;

const CAVP_SIGVER_186_3: &str = include_str!("../testdata/rsa/nist_cavp/rsa_sigver_186_3_subset.json");
const CAVP_SIGGEN_186_3_PRIVATE: &str = include_str!("../testdata/rsa/nist_cavp/rsa_siggen_186_3_private_subset.json");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Counts {
  valid: usize,
  invalid: usize,
}

#[derive(Clone, Copy)]
enum SigGenScheme {
  Pkcs1v15,
  Pss,
}

#[derive(Clone, Copy)]
enum CavpResult {
  Pass,
  Fail,
}

fn der_len(len: usize) -> Vec<u8> {
  if len < 128 {
    return vec![u8::try_from(len).expect("short DER length must fit in one byte")];
  }

  let bytes = len.to_be_bytes();
  let first_nonzero = bytes
    .iter()
    .position(|&byte| byte != 0)
    .expect("long DER length must contain a non-zero byte");
  let len_bytes = &bytes[first_nonzero..];
  let mut out = Vec::with_capacity(1usize.strict_add(len_bytes.len()));
  out.push(0x80 | u8::try_from(len_bytes.len()).expect("DER length-of-length must fit in one byte"));
  out.extend_from_slice(len_bytes);
  out
}

fn tlv(tag: u8, value: &[u8]) -> Vec<u8> {
  let encoded_len = der_len(value.len());
  let capacity = 1usize.strict_add(encoded_len.len()).strict_add(value.len());
  let mut out = Vec::with_capacity(capacity);
  out.push(tag);
  out.extend_from_slice(&encoded_len);
  out.extend_from_slice(value);
  out
}

fn integer_unsigned(value: &[u8]) -> Vec<u8> {
  let first_nonzero = value.iter().position(|&byte| byte != 0);
  let value = first_nonzero.map_or(&[0u8][..], |index| &value[index..]);
  let mut encoded = Vec::with_capacity(value.len().strict_add(usize::from(value[0] & 0x80 != 0)));
  if value[0] & 0x80 != 0 {
    encoded.push(0);
  }
  encoded.extend_from_slice(value);
  tlv(0x02, &encoded)
}

fn pkcs1_der(n: &[u8], e: &[u8]) -> Vec<u8> {
  let mut body = Vec::new();
  body.extend_from_slice(&integer_unsigned(n));
  body.extend_from_slice(&integer_unsigned(e));
  tlv(0x30, &body)
}

fn hex_to_vec(hex: &str) -> Vec<u8> {
  let mut padded;
  let hex = if hex.len().is_multiple_of(2) {
    hex
  } else {
    padded = String::with_capacity(hex.len().strict_add(1));
    padded.push('0');
    padded.push_str(hex);
    &padded
  };

  let mut out = Vec::with_capacity(hex.len() / 2);
  for chunk in hex.as_bytes().chunks_exact(2) {
    let high = hex_value(chunk[0]).expect("CAVP fixture must contain hexadecimal digits");
    let low = hex_value(chunk[1]).expect("CAVP fixture must contain hexadecimal digits");
    out.push((high << 4) | low);
  }
  out
}

fn hex_to_canonical_vec(hex: &str) -> Vec<u8> {
  let mut value = hex_to_vec(hex);
  let first_nonzero = value
    .iter()
    .position(|&byte| byte != 0)
    .unwrap_or_else(|| value.len().strict_sub(1));
  if first_nonzero != 0 {
    value.drain(..first_nonzero);
  }
  value
}

fn hex_to_u64(hex: &str) -> u64 {
  let bytes = hex_to_canonical_vec(hex);
  assert!(bytes.len() <= core::mem::size_of::<u64>());
  let mut value = 0u64;
  for byte in bytes {
    value = (value << 8) | u64::from(byte);
  }
  value
}

fn hex_value(byte: u8) -> Option<u8> {
  match byte {
    b'0'..=b'9' => Some(byte.strict_sub(b'0')),
    b'a'..=b'f' => Some(byte.strict_sub(b'a').strict_add(10)),
    b'A'..=b'F' => Some(byte.strict_sub(b'A').strict_add(10)),
    _ => None,
  }
}

fn field<'a>(value: &'a Value, name: &'static str) -> &'a str {
  value[name]
    .as_str()
    .expect("CAVP fixture must contain the requested string field")
}

fn pkcs1_profile(sha: &str) -> RsaPkcs1v15Profile {
  match sha {
    "SHA256" => Some(RsaPkcs1v15Profile::Sha256),
    "SHA384" => Some(RsaPkcs1v15Profile::Sha384),
    "SHA512" => Some(RsaPkcs1v15Profile::Sha512),
    _ => None,
  }
  .expect("CAVP fixture must use a supported PKCS#1 v1.5 hash")
}

fn pss_profile(sha: &str) -> RsaPssProfile {
  match sha {
    "SHA256" => Some(RsaPssProfile::Sha256),
    "SHA384" => Some(RsaPssProfile::Sha384),
    "SHA512" => Some(RsaPssProfile::Sha512),
    _ => None,
  }
  .expect("CAVP fixture must use a supported PSS hash")
}

fn siggen_scheme(scheme: &str) -> SigGenScheme {
  match scheme {
    "pkcs1v15" => Some(SigGenScheme::Pkcs1v15),
    "pss" => Some(SigGenScheme::Pss),
    _ => None,
  }
  .expect("CAVP fixture must use a supported RSA signature scheme")
}

fn cavp_result(result: &str) -> CavpResult {
  match result {
    "P" => Some(CavpResult::Pass),
    "F" => Some(CavpResult::Fail),
    _ => None,
  }
  .expect("CAVP fixture result must be P or F")
}

fn signature_profile(scheme: &str, sha: &str, salt_len: Option<u64>) -> RsaSignatureProfile {
  match siggen_scheme(scheme) {
    SigGenScheme::Pkcs1v15 => RsaSignatureProfile::pkcs1v15(pkcs1_profile(sha)),
    SigGenScheme::Pss => RsaSignatureProfile::pss_with_salt_len(
      pss_profile(sha),
      usize::try_from(salt_len.expect("CAVP PSS case must provide a salt length"))
        .expect("CAVP PSS salt length must fit usize"),
    ),
  }
}

fn cavp_tests(suite: &Value) -> &[Value] {
  suite["tests"].as_array().expect("CAVP test list must be an array")
}

fn fixed_width_one(len: usize) -> Vec<u8> {
  let mut out = vec![0u8; len];
  *out.last_mut().expect("non-empty RSA modulus") = 1;
  out
}

fn private_key_from_cavp_siggen(test: &Value, policy: &RsaPublicKeyPolicy) -> RsaPrivateKey {
  let modulus = hex_to_canonical_vec(field(test, "n"));
  let private_exponent = hex_to_canonical_vec(field(test, "d"));
  let prime_p = hex_to_canonical_vec(field(test, "p"));
  let prime_q = hex_to_canonical_vec(field(test, "q"));
  let exponent_p = hex_to_canonical_vec(field(test, "dp"));
  let exponent_q = hex_to_canonical_vec(field(test, "dq"));
  let coefficient = hex_to_canonical_vec(field(test, "qinv"));

  RsaPrivateKey::from_components_with_policy(
    RsaPrivateKeyParts {
      modulus: &modulus,
      public_exponent: hex_to_u64(field(test, "e")),
      private_exponent: &private_exponent,
      prime_p: &prime_p,
      prime_q: &prime_q,
      exponent_p: &exponent_p,
      exponent_q: &exponent_q,
      coefficient: &coefficient,
    },
    policy,
  )
  .expect("CAVP RSA private-key components must validate")
}

#[test]
fn nist_cavp_odd_public_exponents_require_explicit_policy() {
  let suite: Value = serde_json::from_str(CAVP_SIGVER_186_3).expect("CAVP JSON must parse");
  let test = &cavp_tests(&suite)[0];
  let key_der = pkcs1_der(&hex_to_vec(field(test, "n")), &hex_to_vec(field(test, "e")));

  assert_eq!(
    RsaPublicKey::from_pkcs1_der(&key_der),
    Err(RsaKeyError::InvalidPublicExponent)
  );

  let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_odd_exponents();
  let key = RsaPublicKey::from_pkcs1_der_with_policy(&key_der, &policy)
    .expect("CAVP odd-exponent RSA key must parse under the explicit legacy policy");
  assert_eq!(key.modulus_bits(), 2048);
}

fn expected_siggen_coverage() -> BTreeMap<(String, u64, String), usize> {
  BTreeMap::from([
    (("pkcs1v15".to_owned(), 2048, "SHA256".to_owned()), 1),
    (("pkcs1v15".to_owned(), 2048, "SHA384".to_owned()), 1),
    (("pkcs1v15".to_owned(), 2048, "SHA512".to_owned()), 1),
    (("pkcs1v15".to_owned(), 3072, "SHA256".to_owned()), 1),
    (("pkcs1v15".to_owned(), 3072, "SHA384".to_owned()), 1),
    (("pkcs1v15".to_owned(), 3072, "SHA512".to_owned()), 1),
    (("pss".to_owned(), 2048, "SHA256".to_owned()), 1),
    (("pss".to_owned(), 2048, "SHA384".to_owned()), 1),
    (("pss".to_owned(), 2048, "SHA512".to_owned()), 1),
    (("pss".to_owned(), 3072, "SHA256".to_owned()), 1),
    (("pss".to_owned(), 3072, "SHA384".to_owned()), 1),
    (("pss".to_owned(), 3072, "SHA512".to_owned()), 1),
  ])
}

#[test]
fn nist_cavp_sha2_siggen_private_operations_match_expected_signatures() {
  let suite: Value = serde_json::from_str(CAVP_SIGGEN_186_3_PRIVATE).expect("CAVP SigGen JSON must parse");
  assert_eq!(suite["counts"]["total"].as_u64(), Some(12));
  assert_eq!(suite["counts"]["pkcs1v15"].as_u64(), Some(6));
  assert_eq!(suite["counts"]["pss"].as_u64(), Some(6));
  assert_eq!(suite["source_files"][0].as_str(), Some("SigGen15_186-3.txt"));
  assert_eq!(suite["source_files"][1].as_str(), Some("SigGenPSS_186-3.txt"));

  let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_odd_exponents();
  let mut pkcs1v15 = 0usize;
  let mut pss = 0usize;
  let mut coverage: BTreeMap<(String, u64, String), usize> = BTreeMap::new();

  for test in cavp_tests(&suite) {
    let coverage_key = (
      field(test, "scheme").to_owned(),
      test["mod"].as_u64().expect("CAVP modulus size must be numeric"),
      field(test, "sha").to_owned(),
    );
    coverage
      .entry(coverage_key)
      .and_modify(|count| *count = (*count).strict_add(1))
      .or_insert(1);

    let key = private_key_from_cavp_siggen(test, &policy);
    let message = hex_to_vec(field(test, "msg"));
    let expected_signature = hex_to_vec(field(test, "sig"));
    let mut signature = vec![0u8; key.public_key().modulus().len()];
    let mut scratch_signature = vec![0u8; key.public_key().modulus().len()];
    let mut scratch = key.private_scratch();
    let blinding_factor = fixed_width_one(key.public_key().modulus().len());
    let blinding_factor_inverse = fixed_width_one(key.public_key().modulus().len());

    match siggen_scheme(field(test, "scheme")) {
      SigGenScheme::Pkcs1v15 => {
        pkcs1v15 = pkcs1v15.strict_add(1);
        key
          .sign_pkcs1v15_with_blinding_factor(
            pkcs1_profile(field(test, "sha")),
            &message,
            RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
            &mut signature,
          )
          .expect("CAVP PKCS1v1.5 private signing must succeed");
        assert_eq!(signature, expected_signature, "CAVP PKCS1v1.5 signature mismatch");
        key
          .sign_pkcs1v15_with_blinding_factor_and_scratch(
            pkcs1_profile(field(test, "sha")),
            &message,
            RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
            &mut scratch_signature,
            &mut scratch,
          )
          .expect("CAVP PKCS1v1.5 scratch private signing must succeed");
        assert_eq!(
          scratch_signature, expected_signature,
          "CAVP PKCS1v1.5 scratch signature mismatch"
        );
        key
          .public_key()
          .verify_pkcs1v15(pkcs1_profile(field(test, "sha")), &message, &signature)
          .expect("CAVP PKCS1v1.5 generated signature must verify");
      }
      SigGenScheme::Pss => {
        pss = pss.strict_add(1);
        let salt = hex_to_vec(field(test, "salt"));
        key
          .sign_pss_with_salt_and_blinding_factor(
            pss_profile(field(test, "sha")),
            &message,
            &salt,
            RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
            &mut signature,
          )
          .expect("CAVP PSS private signing must succeed");
        assert_eq!(signature, expected_signature, "CAVP PSS signature mismatch");
        key
          .sign_pss_with_salt_and_blinding_factor_and_scratch(
            pss_profile(field(test, "sha")),
            &message,
            &salt,
            RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
            &mut scratch_signature,
            &mut scratch,
          )
          .expect("CAVP PSS scratch private signing must succeed");
        assert_eq!(
          scratch_signature, expected_signature,
          "CAVP PSS scratch signature mismatch"
        );
        key
          .public_key()
          .verify_signature(
            RsaSignatureProfile::pss_with_salt_len(pss_profile(field(test, "sha")), salt.len()),
            &message,
            &signature,
          )
          .expect("CAVP PSS generated signature must verify");
      }
    }
  }

  assert_eq!(pkcs1v15, 6);
  assert_eq!(pss, 6);
  assert_eq!(coverage, expected_siggen_coverage());
}

#[test]
fn nist_cavp_same_width_public_scratch_rebinds_between_keys() {
  let suite: Value = serde_json::from_str(CAVP_SIGGEN_186_3_PRIVATE).expect("CAVP SigGen JSON must parse");
  let tests = cavp_tests(&suite);
  let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_odd_exponents();
  let key_a = private_key_from_cavp_siggen(&tests[0], &policy);
  let key_b = private_key_from_cavp_siggen(&tests[6], &policy);
  assert_eq!(key_a.public_key().modulus().len(), key_b.public_key().modulus().len());
  assert_ne!(key_a.public_key().modulus(), key_b.public_key().modulus());

  let representative_b = hex_to_vec(field(&tests[6], "sig"));
  let mut expected_b = vec![0u8; key_b.public_key().modulus().len()];
  key_b
    .public_key()
    .public_operation(&representative_b, &mut expected_b)
    .expect("key-B public operation must succeed");

  let mut scratch = key_a.public_key().public_scratch();
  let mut actual_b = vec![0u8; expected_b.len()];
  key_b
    .public_key()
    .public_operation_with_scratch(&representative_b, &mut actual_b, &mut scratch)
    .expect("same-width scratch must rebind from key A to key B");
  assert_eq!(actual_b, expected_b);

  let mut representative_a = vec![0u8; key_a.public_key().modulus().len()];
  *representative_a.last_mut().expect("non-empty RSA modulus") = 2;
  let mut expected_a = vec![0u8; representative_a.len()];
  key_a
    .public_key()
    .public_operation(&representative_a, &mut expected_a)
    .expect("key-A public operation must succeed");

  let mut actual_a = vec![0u8; expected_a.len()];
  key_a
    .public_key()
    .public_operation_with_scratch(&representative_a, &mut actual_a, &mut scratch)
    .expect("same-width scratch must rebind from key B back to key A");
  assert_eq!(actual_a, expected_a);
}

#[test]
fn nist_cavp_same_width_private_scratch_rebinds_between_keys() {
  let suite: Value = serde_json::from_str(CAVP_SIGGEN_186_3_PRIVATE).expect("CAVP SigGen JSON must parse");
  let tests = cavp_tests(&suite);
  let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_odd_exponents();
  let key_a = private_key_from_cavp_siggen(&tests[0], &policy);
  let test_b = &tests[6];
  let key_b = private_key_from_cavp_siggen(test_b, &policy);
  assert_eq!(key_a.public_key().modulus().len(), key_b.public_key().modulus().len());
  assert_ne!(key_a.public_key().modulus(), key_b.public_key().modulus());

  let message = hex_to_vec(field(test_b, "msg"));
  let salt = hex_to_vec(field(test_b, "salt"));
  let expected_signature = hex_to_vec(field(test_b, "sig"));
  let blinding_factor = fixed_width_one(key_b.public_key().modulus().len());
  let blinding_factor_inverse = fixed_width_one(key_b.public_key().modulus().len());
  let mut signature = vec![0u8; expected_signature.len()];
  let mut scratch = key_a.private_scratch();

  key_b
    .sign_pss_with_salt_and_blinding_factor_and_scratch(
      pss_profile(field(test_b, "sha")),
      &message,
      &salt,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut signature,
      &mut scratch,
    )
    .expect("same-width private scratch must rebind from key A to key B");
  assert_eq!(signature, expected_signature);
}

#[cfg(feature = "getrandom")]
#[test]
fn nist_cavp_sha2_siggen_profile_signing_matches_expected_results() {
  let suite: Value = serde_json::from_str(CAVP_SIGGEN_186_3_PRIVATE).expect("CAVP SigGen JSON must parse");
  let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_odd_exponents();
  let mut pkcs1v15 = 0usize;
  let mut pss = 0usize;

  for test in cavp_tests(&suite) {
    let key = private_key_from_cavp_siggen(test, &policy);
    let message = hex_to_vec(field(test, "msg"));
    let expected_signature = hex_to_vec(field(test, "sig"));
    let mut signature = vec![0u8; key.public_key().modulus().len()];
    let mut scratch_signature = vec![0u8; key.public_key().modulus().len()];
    let mut scratch = key.private_scratch();

    match siggen_scheme(field(test, "scheme")) {
      SigGenScheme::Pkcs1v15 => {
        pkcs1v15 = pkcs1v15.strict_add(1);
        let profile = RsaSignatureProfile::pkcs1v15(pkcs1_profile(field(test, "sha")));
        key
          .sign_signature(profile, &message, &mut signature)
          .expect("CAVP PKCS1v1.5 profile signing must succeed");
        assert_eq!(
          signature, expected_signature,
          "CAVP PKCS1v1.5 profile signature mismatch"
        );
        key
          .sign_signature_with_scratch(profile, &message, &mut scratch_signature, &mut scratch)
          .expect("CAVP PKCS1v1.5 scratch profile signing must succeed");
        assert_eq!(
          scratch_signature, expected_signature,
          "CAVP PKCS1v1.5 scratch profile signature mismatch"
        );
      }
      SigGenScheme::Pss => {
        pss = pss.strict_add(1);
        let salt_len = hex_to_vec(field(test, "salt")).len();
        let profile = RsaSignatureProfile::pss_with_salt_len(pss_profile(field(test, "sha")), salt_len);
        key
          .sign_signature(profile, &message, &mut signature)
          .expect("CAVP PSS profile signing must succeed");
        key
          .public_key()
          .verify_signature(profile, &message, &signature)
          .expect("CAVP PSS profile signature must verify");
        key
          .sign_signature_with_scratch(profile, &message, &mut scratch_signature, &mut scratch)
          .expect("CAVP PSS scratch profile signing must succeed");
        key
          .public_key()
          .verify_signature(profile, &message, &scratch_signature)
          .expect("CAVP PSS scratch profile signature must verify");
      }
    }
  }

  assert_eq!(pkcs1v15, 6);
  assert_eq!(pss, 6);
}

fn expected_coverage() -> BTreeMap<(String, u64, String, Option<u64>), usize> {
  BTreeMap::from([
    (("pkcs1v15".to_owned(), 2048, "SHA256".to_owned(), None), 18),
    (("pkcs1v15".to_owned(), 2048, "SHA384".to_owned(), None), 18),
    (("pkcs1v15".to_owned(), 2048, "SHA512".to_owned(), None), 18),
    (("pkcs1v15".to_owned(), 3072, "SHA256".to_owned(), None), 18),
    (("pkcs1v15".to_owned(), 3072, "SHA384".to_owned(), None), 18),
    (("pkcs1v15".to_owned(), 3072, "SHA512".to_owned(), None), 18),
    (("pss".to_owned(), 2048, "SHA256".to_owned(), Some(32)), 18),
    (("pss".to_owned(), 2048, "SHA384".to_owned(), Some(48)), 18),
    (("pss".to_owned(), 2048, "SHA512".to_owned(), Some(64)), 18),
    (("pss".to_owned(), 3072, "SHA256".to_owned(), Some(0)), 18),
    (("pss".to_owned(), 3072, "SHA384".to_owned(), Some(24)), 18),
    (("pss".to_owned(), 3072, "SHA512".to_owned(), Some(0)), 18),
  ])
}

#[test]
fn nist_cavp_supported_sha2_sigver_subset_matches_expected_results() {
  let suite: Value = serde_json::from_str(CAVP_SIGVER_186_3).expect("CAVP JSON must parse");
  assert_eq!(suite["counts"]["total"].as_u64(), Some(216));
  assert_eq!(suite["counts"]["valid"].as_u64(), Some(36));
  assert_eq!(suite["counts"]["invalid"].as_u64(), Some(180));
  assert_eq!(cavp_tests(&suite).len(), 216);

  let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_odd_exponents();
  let mut counts = Counts { valid: 0, invalid: 0 };
  let mut coverage: BTreeMap<(String, u64, String, Option<u64>), usize> = BTreeMap::new();

  for test in cavp_tests(&suite) {
    let scheme = field(test, "scheme");
    let sha = field(test, "sha");
    let modulus_bits = test["mod"].as_u64().expect("CAVP modulus size must be numeric");
    let salt_len = if scheme == "pss" {
      Some(test["salt_len"].as_u64().expect("CAVP PSS salt length must be numeric"))
    } else {
      None
    };
    let coverage_key = (scheme.to_owned(), modulus_bits, sha.to_owned(), salt_len);
    coverage
      .entry(coverage_key)
      .and_modify(|count| *count = (*count).strict_add(1))
      .or_insert(1);

    let key_der = pkcs1_der(&hex_to_vec(field(test, "n")), &hex_to_vec(field(test, "e")));
    let key = RsaPublicKey::from_pkcs1_der_with_policy(&key_der, &policy).expect("CAVP RSA key must parse");
    let mut scratch = key.public_scratch();
    let message = hex_to_vec(field(test, "msg"));
    let signature = hex_to_vec(field(test, "sig"));
    let verified = key
      .verify_signature_with_scratch(
        signature_profile(scheme, sha, salt_len),
        &message,
        &signature,
        &mut scratch,
      )
      .is_ok();

    match cavp_result(field(test, "result")) {
      CavpResult::Pass => {
        counts.valid = counts.valid.strict_add(1);
        assert!(verified, "CAVP tcId {} rejected valid signature", test["tc_id"]);
      }
      CavpResult::Fail => {
        counts.invalid = counts.invalid.strict_add(1);
        assert!(
          !verified,
          "CAVP tcId {} accepted invalid {} signature",
          test["tc_id"], scheme
        );
      }
    }
  }

  assert_eq!(
    counts,
    Counts {
      valid: 36,
      invalid: 180
    }
  );
  assert_eq!(coverage, expected_coverage());
}
