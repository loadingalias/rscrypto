#![cfg(feature = "rsa")]

extern crate alloc;

use alloc::collections::BTreeMap;

use rscrypto::{
  RsaKeyError, RsaPkcs1v15Profile, RsaPrivateKey, RsaPrivateKeyParts, RsaPssProfile, RsaPublicKey, RsaPublicKeyPolicy,
  RsaSignatureProfile,
};
use serde_json::Value;

const CAVP_SIGVER_186_3: &str = include_str!("../testdata/rsa/nist_cavp/rsa_sigver_186_3_subset.json");
const CAVP_SIGGEN_186_3_PRIVATE: &str = include_str!("../testdata/rsa/nist_cavp/rsa_siggen_186_3_private_subset.json");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Counts {
  valid: usize,
  invalid: usize,
}

fn der_len(len: usize) -> Vec<u8> {
  if len < 128 {
    return vec![len as u8];
  }

  let bytes = len.to_be_bytes();
  let first_nonzero = bytes.iter().position(|&byte| byte != 0).unwrap();
  let len_bytes = &bytes[first_nonzero..];
  let mut out = Vec::with_capacity(1 + len_bytes.len());
  out.push(0x80 | len_bytes.len() as u8);
  out.extend_from_slice(len_bytes);
  out
}

fn tlv(tag: u8, value: &[u8]) -> Vec<u8> {
  let mut out = Vec::with_capacity(1 + der_len(value.len()).len() + value.len());
  out.push(tag);
  out.extend_from_slice(&der_len(value.len()));
  out.extend_from_slice(value);
  out
}

fn integer_unsigned(value: &[u8]) -> Vec<u8> {
  let first_nonzero = value.iter().position(|&byte| byte != 0);
  let value = first_nonzero.map_or(&[0u8][..], |index| &value[index..]);
  let mut encoded = Vec::with_capacity(value.len() + usize::from(value[0] & 0x80 != 0));
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
    out.push((hex_value(chunk[0]) << 4) | hex_value(chunk[1]));
  }
  out
}

fn hex_to_canonical_vec(hex: &str) -> Vec<u8> {
  let mut value = hex_to_vec(hex);
  let first_nonzero = value
    .iter()
    .position(|&byte| byte != 0)
    .unwrap_or(value.len().strict_sub(1));
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

fn hex_value(byte: u8) -> u8 {
  match byte {
    b'0'..=b'9' => byte - b'0',
    b'a'..=b'f' => byte - b'a' + 10,
    b'A'..=b'F' => byte - b'A' + 10,
    _ => panic!("invalid hex digit"),
  }
}

fn field<'a>(value: &'a Value, name: &'static str) -> &'a str {
  value[name]
    .as_str()
    .unwrap_or_else(|| panic!("missing string field `{name}`"))
}

fn pkcs1_profile(sha: &str) -> RsaPkcs1v15Profile {
  match sha {
    "SHA256" => RsaPkcs1v15Profile::Sha256,
    "SHA384" => RsaPkcs1v15Profile::Sha384,
    "SHA512" => RsaPkcs1v15Profile::Sha512,
    other => panic!("unsupported CAVP PKCS1v1.5 hash `{other}`"),
  }
}

fn pss_profile(sha: &str) -> RsaPssProfile {
  match sha {
    "SHA256" => RsaPssProfile::Sha256,
    "SHA384" => RsaPssProfile::Sha384,
    "SHA512" => RsaPssProfile::Sha512,
    other => panic!("unsupported CAVP PSS hash `{other}`"),
  }
}

fn signature_profile(scheme: &str, sha: &str, salt_len: Option<u64>) -> RsaSignatureProfile {
  match scheme {
    "pkcs1v15" => RsaSignatureProfile::pkcs1v15(pkcs1_profile(sha)),
    "pss" => RsaSignatureProfile::pss_with_salt_len(pss_profile(sha), salt_len.unwrap() as usize),
    other => panic!("unsupported CAVP RSA signature scheme `{other}`"),
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
  let key = RsaPublicKey::from_pkcs1_der_with_policy(&key_der, &policy).unwrap();
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

    match field(test, "scheme") {
      "pkcs1v15" => {
        pkcs1v15 = pkcs1v15.strict_add(1);
        key
          .sign_pkcs1v15_with_blinding_factor(
            pkcs1_profile(field(test, "sha")),
            &message,
            &blinding_factor,
            &blinding_factor_inverse,
            &mut signature,
          )
          .expect("CAVP PKCS1v1.5 private signing must succeed");
        assert_eq!(signature, expected_signature, "CAVP PKCS1v1.5 signature mismatch");
        key
          .sign_pkcs1v15_with_blinding_factor_and_scratch(
            pkcs1_profile(field(test, "sha")),
            &message,
            &blinding_factor,
            &blinding_factor_inverse,
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
      "pss" => {
        pss = pss.strict_add(1);
        let salt = hex_to_vec(field(test, "salt"));
        key
          .sign_pss_with_salt_and_blinding_factor(
            pss_profile(field(test, "sha")),
            &message,
            &salt,
            &blinding_factor,
            &blinding_factor_inverse,
            &mut signature,
          )
          .expect("CAVP PSS private signing must succeed");
        assert_eq!(signature, expected_signature, "CAVP PSS signature mismatch");
        key
          .sign_pss_with_salt_and_blinding_factor_and_scratch(
            pss_profile(field(test, "sha")),
            &message,
            &salt,
            &blinding_factor,
            &blinding_factor_inverse,
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
      other => panic!("unsupported CAVP SigGen scheme `{other}`"),
    }
  }

  assert_eq!(pkcs1v15, 6);
  assert_eq!(pss, 6);
  assert_eq!(coverage, expected_siggen_coverage());
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

    match field(test, "scheme") {
      "pkcs1v15" => {
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
      "pss" => {
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
      other => panic!("unsupported CAVP SigGen scheme `{other}`"),
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

    match field(test, "result") {
      "P" => {
        counts.valid = counts.valid.strict_add(1);
        assert!(verified, "CAVP tcId {} rejected valid signature", test["tc_id"]);
      }
      "F" => {
        counts.invalid = counts.invalid.strict_add(1);
        assert!(
          !verified,
          "CAVP tcId {} accepted invalid {} signature",
          test["tc_id"], scheme
        );
      }
      other => panic!("unknown CAVP result `{other}`"),
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
