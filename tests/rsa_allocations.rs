#![cfg(feature = "rsa")]

use rsa::{
  PrivateKey as RustCryptoRsaPrivateKey,
  pkcs1::{DecodeRsaPrivateKey, EncodeRsaPrivateKey},
};
use rscrypto::auth::rsa::expert::RsaPkcs1v15Decrypt;
use rscrypto::{
  RsaBlindingPair, RsaEncryptionError, RsaJwtAlgorithm, RsaOaepProfile, RsaPkcs1v15Profile, RsaPrivateKey,
  RsaPssProfile, RsaPublicKey, RsaPublicKeyPolicy, RsaSignatureProfile, RsaX509PublicKey,
};

#[path = "common/allocations.rs"]
mod allocations;
use allocations::{Measurement, measure_allocations};

const ID_RSASSA_PSS_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a];

const MESSAGE_PSS: &[u8] = b"rscrypto RSA-PSS verification fixture";
const MESSAGE_PKCS1V15: &[u8] = b"rscrypto RSA-PKCS1-v1_5 verification fixture";
const X509_PSS_SHA256_ALGORITHM: &[u8] = &[
  0x30, 0x41, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a, 0x30, 0x34, 0xa0, 0x0f, 0x30, 0x0d,
  0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05, 0x00, 0xa1, 0x1c, 0x30, 0x1a, 0x06, 0x09,
  0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x08, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03,
  0x04, 0x02, 0x01, 0x05, 0x00, 0xa2, 0x03, 0x02, 0x01, 0x20,
];
const X509_PSS_SHA256_OVERSIZED_SALT_ALGORITHM: &[u8] = &[
  0x30, 0x42, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a, 0x30, 0x35, 0xa0, 0x0f, 0x30, 0x0d,
  0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05, 0x00, 0xa1, 0x1c, 0x30, 0x1a, 0x06, 0x09,
  0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x08, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03,
  0x04, 0x02, 0x01, 0x05, 0x00, 0xa2, 0x04, 0x02, 0x02, 0x10, 0x00,
];
const X509_SHA256_WITH_RSA_ENCRYPTION: &[u8] = &[
  0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b, 0x05, 0x00,
];
const X509_SHA1_WITH_RSA_ENCRYPTION: &[u8] = &[
  0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x05, 0x05, 0x00,
];
const X509_PSS_DEFAULT_SHA1_ALGORITHM: &[u8] = &[
  0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a, 0x30, 0x00,
];

fn der_len(len: usize) -> Vec<u8> {
  if len < 128 {
    return vec![u8::try_from(len).expect("a short DER length must fit in one byte")];
  }

  let bytes = len.to_be_bytes();
  let first_nonzero = bytes
    .iter()
    .position(|&byte| byte != 0)
    .expect("a DER length of at least 128 must contain a nonzero byte");
  let len_bytes = &bytes[first_nonzero..];
  let mut out = Vec::with_capacity(1usize.strict_add(len_bytes.len()));
  out.push(0x80 | u8::try_from(len_bytes.len()).expect("a usize DER length prefix must fit in one byte"));
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

fn sequence(value: &[u8]) -> Vec<u8> {
  tlv(0x30, value)
}

fn fill_rsa_random_with(byte: u8) -> impl FnMut(&mut [u8]) -> Result<(), RsaEncryptionError> {
  move |out| {
    out.fill(byte);
    Ok(())
  }
}

fn fill_rsa_random_from(bytes: &[u8]) -> impl FnMut(&mut [u8]) -> Result<(), RsaEncryptionError> + '_ {
  let mut offset = 0usize;
  move |out| {
    let end = offset.checked_add(out.len()).ok_or(RsaEncryptionError::InvalidLength)?;
    let Some(random) = bytes.get(offset..end) else {
      return Err(RsaEncryptionError::InvalidLength);
    };
    out.copy_from_slice(random);
    offset = end;
    Ok(())
  }
}

fn oid(value: &[u8]) -> Vec<u8> {
  tlv(0x06, value)
}

fn bit_string(value: &[u8]) -> Vec<u8> {
  tlv(0x03, value)
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

fn algorithm_identifier(algorithm_oid: &[u8], params: Option<&[u8]>) -> Vec<u8> {
  let mut body = Vec::new();
  body.extend_from_slice(&oid(algorithm_oid));
  if let Some(params) = params {
    body.extend_from_slice(params);
  }
  sequence(&body)
}

fn exponent_bytes(value: u64) -> Vec<u8> {
  let bytes = value.to_be_bytes();
  let first_nonzero = bytes
    .iter()
    .position(|&byte| byte != 0)
    .unwrap_or_else(|| bytes.len().strict_sub(1));
  bytes[first_nonzero..].to_vec()
}

fn pkcs1_from_public_key(key: &RsaPublicKey) -> Vec<u8> {
  let mut body = Vec::new();
  body.extend_from_slice(&integer_unsigned(key.modulus()));
  body.extend_from_slice(&integer_unsigned(&exponent_bytes(key.public_exponent().as_u64())));
  sequence(&body)
}

fn spki_for_pkcs1_with_algorithm(pkcs1: &[u8], algorithm: &[u8]) -> Vec<u8> {
  let mut subject_public_key = Vec::with_capacity(1usize.strict_add(pkcs1.len()));
  subject_public_key.push(0);
  subject_public_key.extend_from_slice(pkcs1);

  let mut spki = Vec::new();
  spki.extend_from_slice(algorithm);
  spki.extend_from_slice(&bit_string(&subject_public_key));
  sequence(&spki)
}

fn pss_algorithm_spki_from_rsa_encryption_spki(spki: &[u8]) -> Vec<u8> {
  let public_key = legacy_public_key_from_spki(spki);
  let pkcs1 = pkcs1_from_public_key(&public_key);
  spki_for_pkcs1_with_algorithm(&pkcs1, &algorithm_identifier(ID_RSASSA_PSS_OID, None))
}

fn legacy_public_key_from_spki(spki: &[u8]) -> RsaPublicKey {
  RsaPublicKey::from_spki_der_with_policy(spki, &RsaPublicKeyPolicy::modulus_2048_to_8192())
    .expect("the SPKI fixture must satisfy the legacy verification policy")
}

fn legacy_x509_public_key_from_spki(spki: &[u8]) -> RsaX509PublicKey {
  RsaX509PublicKey::from_spki_der_with_policy(spki, &RsaPublicKeyPolicy::modulus_2048_to_8192())
    .expect("the X.509 SPKI fixture must satisfy the legacy verification policy")
}

fn minimal_tbs_certificate(signature_algorithm_der: &[u8]) -> Vec<u8> {
  let mut tbs = Vec::new();
  tbs.extend_from_slice(&integer_unsigned(&[1]));
  tbs.extend_from_slice(signature_algorithm_der);
  sequence(&tbs)
}

fn x509_certificate(tbs_certificate_der: &[u8], signature_algorithm_der: &[u8], signature: &[u8]) -> Vec<u8> {
  let mut signature_value = Vec::with_capacity(signature.len().strict_add(1));
  signature_value.push(0);
  signature_value.extend_from_slice(signature);

  let mut certificate = Vec::new();
  certificate.extend_from_slice(tbs_certificate_der);
  certificate.extend_from_slice(signature_algorithm_der);
  certificate.extend_from_slice(&bit_string(&signature_value));
  sequence(&certificate)
}

fn pss_spki() -> Vec<u8> {
  include_bytes!("vectors/rsa/rsa2048_pss_spki.der").to_vec()
}

fn pss_signature_sha256() -> Vec<u8> {
  include_bytes!("vectors/rsa/rsa2048_pss_signature_sha256.sig").to_vec()
}

fn pkcs1v15_spki() -> Vec<u8> {
  include_bytes!("vectors/rsa/rsa2048_pkcs1v15_spki.der").to_vec()
}

fn pkcs1v15_signature_sha256() -> Vec<u8> {
  include_bytes!("vectors/rsa/rsa2048_pkcs1v15_signature_sha256.sig").to_vec()
}

fn rustcrypto_fixture_private_key() -> RustCryptoRsaPrivateKey {
  RustCryptoRsaPrivateKey::from_pkcs1_der(include_bytes!("vectors/rsa/rsa2048_private_pkcs1.der"))
    .expect("the independent RSA-2048 fixture must parse")
}

fn private_key() -> RsaPrivateKey {
  let der = rustcrypto_fixture_private_key()
    .to_pkcs1_der()
    .expect("the RustCrypto private-key fixture must encode as PKCS#1");
  RsaPrivateKey::from_pkcs1_der_with_policy(der.as_bytes(), &RsaPublicKeyPolicy::modulus_2048_to_8192())
    .expect("rscrypto must decode the RustCrypto private-key fixture")
}

fn factor_two_and_inverse(modulus: &[u8]) -> (Vec<u8>, Vec<u8>) {
  let mut factor = vec![0u8; modulus.len()];
  if let Some(last) = factor.last_mut() {
    *last = 2;
  }

  let mut plus_one = modulus.to_vec();
  let mut carry = 1u16;
  for byte in plus_one.iter_mut().rev() {
    let sum = u16::from(*byte).strict_add(carry);
    *byte = sum.to_le_bytes()[0];
    carry = sum >> 8;
    if carry == 0 {
      break;
    }
  }
  if carry != 0 {
    plus_one.insert(0, u8::try_from(carry).expect("addition carry must be at most one"));
  }

  let mut quotient = Vec::with_capacity(plus_one.len());
  let mut remainder = 0u16;
  for byte in plus_one {
    let value = (remainder << 8) | u16::from(byte);
    quotient.push(u8::try_from(value / 2).expect("a base-256 long-division digit must fit in one byte"));
    remainder = value % 2;
  }
  let first_nonzero = quotient
    .iter()
    .position(|&byte| byte != 0)
    .unwrap_or_else(|| quotient.len().strict_sub(1));
  let inverse = quotient[first_nonzero..].to_vec();

  let mut inverse_fixed = vec![0u8; modulus.len()];
  inverse_fixed[modulus.len().strict_sub(inverse.len())..].copy_from_slice(&inverse);
  (factor, inverse_fixed)
}

fn public_operation_input() -> Vec<u8> {
  include_bytes!("vectors/rsa/rsa2048_legacy_exponent_input.bin").to_vec()
}

#[test]
fn der_encoders_allocate_only_the_returned_buffer() {
  use core::hint::black_box;

  let private = private_key();
  let public = private.public_key();
  assert_eq!(measure_allocations(|| drop(black_box(public.to_pkcs1_der()))), 1);
  assert_eq!(measure_allocations(|| drop(black_box(public.to_spki_der()))), 1);
  assert_eq!(measure_allocations(|| drop(black_box(private.to_pkcs1_der()))), 1);
  assert_eq!(measure_allocations(|| drop(black_box(private.to_pkcs8_der()))), 1);
}

#[test]
fn reused_scratch_rsa_operations_do_not_allocate() {
  let key = legacy_public_key_from_spki(&pss_spki());
  let input = public_operation_input();
  let mut out = vec![0u8; key.modulus().len()];

  let mut scratch = key.public_scratch();

  key
    .public_operation_with_scratch(&input, &mut out, &mut scratch)
    .expect("the warm-up RSA public operation must succeed");

  let measurement = Measurement::start();
  key
    .public_operation_with_scratch(&input, &mut out, &mut scratch)
    .expect("the measured RSA public operation must succeed");
  assert_eq!(measurement.finish(), 0);

  let sig = pss_signature_sha256();
  let pss_sha256 = RsaSignatureProfile::pss(RsaPssProfile::Sha256);
  let measurement = Measurement::start();
  key
    .verify_pss_with_scratch(RsaPssProfile::Sha256, MESSAGE_PSS, &sig, &mut scratch)
    .expect("scratch-backed PSS verification must succeed");
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  key
    .verify_signature_with_scratch(
      RsaSignatureProfile::pss(RsaPssProfile::Sha256),
      MESSAGE_PSS,
      &sig,
      &mut scratch,
    )
    .expect("typed scratch-backed PSS verification must succeed");
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  key
    .jwt_verifier(RsaJwtAlgorithm::Ps256)
    .verify_with_scratch("PS256", MESSAGE_PSS, &sig, &mut scratch)
    .expect("scratch-backed PS256 verification must succeed");
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  key
    .verify_expected_cose_algorithm_id_with_scratch(-37, -37, pss_sha256, MESSAGE_PSS, &sig, &mut scratch)
    .expect("scratch-backed PS256 COSE verification must succeed");
  assert_eq!(measurement.finish(), 0);

  let seed = [0x42; 32];
  let measurement = Measurement::start();
  key
    .encrypt_oaep_with_random_fill_and_scratch(
      RsaOaepProfile::Sha256,
      b"allocation-oracle",
      b"scratch-backed OAEP encryption",
      &mut out,
      &mut scratch,
      fill_rsa_random_from(&seed),
    )
    .expect("scratch-backed OAEP encryption must succeed");
  assert_eq!(measurement.finish(), 0);

  let x509_key = legacy_x509_public_key_from_spki(&pss_spki());
  let mut scratch = x509_key.public_key().public_scratch();
  let measurement = Measurement::start();
  x509_key
    .verify_signature_from_x509_algorithm_der_with_scratch(X509_PSS_SHA256_ALGORITHM, MESSAGE_PSS, &sig, &mut scratch)
    .expect("scratch-backed X.509 PSS verification must succeed");
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  x509_key
    .verify_expected_tls13_signature_scheme_with_scratch(0x0804, 0x0804, pss_sha256, MESSAGE_PSS, &sig, &mut scratch)
    .expect("scratch-backed TLS 1.3 PSS verification must succeed");
  assert_eq!(measurement.finish(), 0);

  let key = legacy_public_key_from_spki(&pkcs1v15_spki());
  let sig = pkcs1v15_signature_sha256();
  let pkcs1v15_sha256 = RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256);
  let mut scratch = key.public_scratch();
  let measurement = Measurement::start();
  key
    .verify_pkcs1v15_with_scratch(RsaPkcs1v15Profile::Sha256, MESSAGE_PKCS1V15, &sig, &mut scratch)
    .expect("scratch-backed PKCS#1 v1.5 verification must succeed");
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  key
    .verify_signature_with_scratch(
      RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256),
      MESSAGE_PKCS1V15,
      &sig,
      &mut scratch,
    )
    .expect("typed scratch-backed PKCS#1 v1.5 verification must succeed");
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  key
    .jwt_verifier(RsaJwtAlgorithm::Rs256)
    .verify_with_scratch("RS256", MESSAGE_PKCS1V15, &sig, &mut scratch)
    .expect("scratch-backed RS256 verification must succeed");
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  key
    .verify_expected_cose_algorithm_id_with_scratch(-257, -257, pkcs1v15_sha256, MESSAGE_PKCS1V15, &sig, &mut scratch)
    .expect("scratch-backed RS256 COSE verification must succeed");
  assert_eq!(measurement.finish(), 0);

  let message = b"scratch-backed RSAES-PKCS1-v1_5 encryption";
  let mut ciphertext = vec![0u8; key.modulus().len()];
  let measurement = Measurement::start();
  key
    .encrypt_pkcs1v15_with_random_fill_and_scratch(message, &mut ciphertext, &mut scratch, fill_rsa_random_with(0x5d))
    .expect("scratch-backed PKCS#1 v1.5 encryption must succeed");
  assert_eq!(measurement.finish(), 0);

  let x509_key = legacy_x509_public_key_from_spki(&pkcs1v15_spki());
  let mut scratch = x509_key.public_key().public_scratch();
  let measurement = Measurement::start();
  x509_key
    .verify_signature_from_x509_algorithm_der_with_scratch(
      X509_SHA256_WITH_RSA_ENCRYPTION,
      MESSAGE_PKCS1V15,
      &sig,
      &mut scratch,
    )
    .expect("scratch-backed X.509 PKCS#1 v1.5 verification must succeed");
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  x509_key
    .verify_expected_tls_certificate_signature_scheme_with_scratch(
      0x0401,
      0x0401,
      pkcs1v15_sha256,
      MESSAGE_PKCS1V15,
      &sig,
      &mut scratch,
    )
    .expect("scratch-backed TLS certificate PKCS#1 v1.5 verification must succeed");
  assert_eq!(measurement.finish(), 0);

  assert_one_shot_protocol_rejects_fail_before_scratch_allocation();
  let private_key = private_key();
  assert_private_caller_random_rejects_fail_before_scratch_allocation(&private_key);
  #[cfg(feature = "getrandom")]
  assert_private_protocol_signing_rejects_fail_before_entropy_allocation(&private_key);
  assert_private_scratch_operations_do_not_allocate(&private_key);
  #[cfg(feature = "getrandom")]
  assert_rng_private_scratch_operations_do_not_allocate(&private_key);
}

fn assert_private_caller_random_rejects_fail_before_scratch_allocation(key: &RsaPrivateKey) {
  let mut signature = vec![0xa5; key.signature_len()];
  let mut entropy_calls = 0usize;

  let measurement = Measurement::start();
  assert!(
    key
      .sign_signature_with_random_fill(
        RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, usize::MAX),
        b"private caller-random allocation reject",
        &mut signature,
        |_| {
          entropy_calls = entropy_calls.strict_add(1);
          Ok::<(), ()>(())
        },
      )
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);
  assert_eq!(entropy_calls, 0);
  assert!(signature.iter().all(|&byte| byte == 0));
}

#[cfg(feature = "getrandom")]
fn assert_private_protocol_signing_rejects_fail_before_entropy_allocation(key: &RsaPrivateKey) {
  let mut scratch = key.private_scratch();
  let mut signature = vec![0xa5; key.signature_len()];
  let message = b"private protocol signing allocation reject";

  let measurement = Measurement::start();
  assert!(
    key
      .sign_tls13_signature_scheme(0x0401, message, &mut signature)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  let measurement = Measurement::start();
  assert!(
    key
      .sign_tls13_signature_scheme_with_scratch(0x0401, message, &mut signature, &mut scratch)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  let measurement = Measurement::start();
  assert!(
    key
      .sign_tls_certificate_signature_scheme(0x0201, message, &mut signature)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  let measurement = Measurement::start();
  assert!(
    key
      .sign_tls_certificate_signature_scheme_with_scratch(0x0201, message, &mut signature, &mut scratch)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  let measurement = Measurement::start();
  assert!(key.sign_cose_algorithm_id(-7, message, &mut signature).is_err());
  assert_eq!(measurement.finish(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  let measurement = Measurement::start();
  assert!(
    key
      .sign_cose_algorithm_id_with_scratch(-7, message, &mut signature, &mut scratch)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  let measurement = Measurement::start();
  assert!(
    key
      .sign_x509_signature_algorithm_der(X509_SHA1_WITH_RSA_ENCRYPTION, message, &mut signature)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  let measurement = Measurement::start();
  assert!(
    key
      .sign_x509_signature_algorithm_der_with_scratch(
        X509_SHA1_WITH_RSA_ENCRYPTION,
        message,
        &mut signature,
        &mut scratch,
      )
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  let measurement = Measurement::start();
  assert!(
    key
      .sign_signature(
        RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, usize::MAX),
        message,
        &mut signature
      )
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  let measurement = Measurement::start();
  assert!(
    key
      .sign_signature_with_scratch(
        RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, usize::MAX),
        message,
        &mut signature,
        &mut scratch,
      )
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));
}

fn assert_private_scratch_operations_do_not_allocate(key: &RsaPrivateKey) {
  let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
  let mut scratch = key.private_scratch();
  let mut signature = vec![0u8; key.signature_len()];
  let mut decrypted = vec![0u8; key.signature_len()];

  let measurement = Measurement::start();
  key
    .sign_pkcs1v15_with_blinding_factor_and_scratch(
      RsaPkcs1v15Profile::Sha256,
      b"private scratch allocation PKCS1v15",
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut signature,
      &mut scratch,
    )
    .expect("scratch-backed blinded PKCS#1 v1.5 signing must succeed");
  assert_eq!(measurement.finish(), 0);

  let salt = [0x7a; 32];
  let measurement = Measurement::start();
  key
    .sign_pss_with_salt_and_blinding_factor_and_scratch(
      RsaPssProfile::Sha256,
      b"private scratch allocation PSS",
      &salt,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut signature,
      &mut scratch,
    )
    .expect("scratch-backed blinded PSS signing must succeed");
  assert_eq!(measurement.finish(), 0);

  let mut entropy_calls = 0usize;
  let measurement = Measurement::start();
  key
    .sign_pss_with_random_fill_and_scratch(
      RsaPssProfile::Sha256,
      b"private scratch allocation caller-random PSS",
      &mut signature,
      &mut scratch,
      |out| {
        if entropy_calls == 0 {
          out.fill(0x39);
        } else {
          out.copy_from_slice(&blinding_factor);
        }
        entropy_calls = entropy_calls.strict_add(1);
        Ok::<(), ()>(())
      },
    )
    .expect("scratch-backed caller-random PSS signing must succeed");
  assert_eq!(entropy_calls, 3);
  assert_eq!(measurement.finish(), 0);

  let label = b"private-scratch-allocation";
  let plaintext = b"private scratch allocation OAEP";
  let seed = [0x52; 32];
  let mut ciphertext = vec![0u8; key.signature_len()];
  key
    .public_key()
    .encrypt_oaep_with_random_fill(
      RsaOaepProfile::Sha256,
      label,
      plaintext,
      &mut ciphertext,
      fill_rsa_random_from(&seed),
    )
    .expect("OAEP fixture encryption must succeed");

  let measurement = Measurement::start();
  let len = key
    .decrypt_oaep_with_blinding_factor_and_scratch(
      RsaOaepProfile::Sha256,
      label,
      &ciphertext,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut decrypted,
      &mut scratch,
    )
    .expect("scratch-backed blinded OAEP decryption must succeed");
  assert_eq!(&decrypted[..len], plaintext);
  assert_eq!(measurement.finish(), 0);

  let pkcs1v15_plaintext = b"private scratch allocation RSAES-PKCS1-v1_5";
  key
    .public_key()
    .encrypt_pkcs1v15_with_random_fill(pkcs1v15_plaintext, &mut ciphertext, fill_rsa_random_with(0x5d))
    .expect("PKCS#1 v1.5 fixture encryption must succeed");

  let measurement = Measurement::start();
  let len = key
    .decrypt_pkcs1v15_with_blinding_factor_and_scratch(
      &ciphertext,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut decrypted,
      &mut scratch,
    )
    .expect("scratch-backed blinded PKCS#1 v1.5 decryption must succeed");
  assert_eq!(&decrypted[..len], pkcs1v15_plaintext);
  assert_eq!(measurement.finish(), 0);

  // A wrong block type must select the synthetic message, using only scratch.
  let mut invalid_encoded = vec![0x5au8; key.signature_len()];
  invalid_encoded[..2].copy_from_slice(&[0, 1]);
  key
    .public_key()
    .public_operation(&invalid_encoded, &mut ciphertext)
    .expect("a zero-leading encoded block must be below the modulus");
  let measurement = Measurement::start();
  let len = key
    .decrypt_pkcs1v15_with_blinding_factor_and_scratch(
      &ciphertext,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut decrypted,
      &mut scratch,
    )
    .expect("implicit rejection must return a message for an invalid block type");
  assert_ne!(&decrypted[..len], pkcs1v15_plaintext);
  assert_eq!(measurement.finish(), 0);
}

#[cfg(feature = "getrandom")]
fn assert_rng_private_scratch_operations_do_not_allocate(key: &RsaPrivateKey) {
  let mut scratch = key.private_scratch();
  let mut signature = vec![0u8; key.signature_len()];
  let mut decrypted = vec![0u8; key.signature_len()];

  let measurement = Measurement::start();
  key
    .sign_pkcs1v15_with_scratch(
      RsaPkcs1v15Profile::Sha256,
      b"rng private scratch allocation PKCS1v15",
      &mut signature,
      &mut scratch,
    )
    .expect("scratch-backed randomized PKCS#1 v1.5 signing must succeed");
  assert_eq!(measurement.finish(), 0);
  key
    .public_key()
    .verify_pkcs1v15(
      RsaPkcs1v15Profile::Sha256,
      b"rng private scratch allocation PKCS1v15",
      &signature,
    )
    .expect("the randomized PKCS#1 v1.5 signature must verify");

  let measurement = Measurement::start();
  key
    .sign_pss_with_scratch(
      RsaPssProfile::Sha256,
      b"rng private scratch allocation PSS",
      &mut signature,
      &mut scratch,
    )
    .expect("scratch-backed randomized PSS signing must succeed");
  assert_eq!(measurement.finish(), 0);
  key
    .public_key()
    .verify_pss(RsaPssProfile::Sha256, b"rng private scratch allocation PSS", &signature)
    .expect("the randomized PSS signature must verify");

  // A workspace of the key signs without allocation, on refresh and on reuse.
  let mut workspace = key.signing_workspace();
  for message in [
    b"rng signing workspace allocation refresh".as_slice(),
    b"rng signing workspace allocation reuse",
  ] {
    let measurement = Measurement::start();
    key
      .sign_signature_with_workspace(
        RsaSignatureProfile::pss(RsaPssProfile::Sha256),
        message,
        &mut signature,
        &mut workspace,
      )
      .expect("workspace PSS signing must succeed");
    assert_eq!(measurement.finish(), 0);
    key
      .public_key()
      .verify_pss(RsaPssProfile::Sha256, message, &signature)
      .expect("the workspace PSS signature must verify");
  }
  let measurement = Measurement::start();
  key
    .sign_signature_with_workspace(
      RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256),
      b"rng signing workspace allocation PKCS1v15",
      &mut signature,
      &mut workspace,
    )
    .expect("workspace PKCS#1 v1.5 signing must succeed");
  assert_eq!(measurement.finish(), 0);
  key
    .public_key()
    .verify_pkcs1v15(
      RsaPkcs1v15Profile::Sha256,
      b"rng signing workspace allocation PKCS1v15",
      &signature,
    )
    .expect("the workspace PKCS#1 v1.5 signature must verify");

  let label = b"rng-private-scratch-allocation";
  let plaintext = b"rng private scratch allocation OAEP";
  let seed = [0x52; 32];
  let mut ciphertext = vec![0u8; key.signature_len()];
  key
    .public_key()
    .encrypt_oaep_with_random_fill(
      RsaOaepProfile::Sha256,
      label,
      plaintext,
      &mut ciphertext,
      fill_rsa_random_from(&seed),
    )
    .expect("OAEP fixture encryption must succeed");

  let measurement = Measurement::start();
  let len = key
    .decrypt_oaep_with_scratch(RsaOaepProfile::Sha256, label, &ciphertext, &mut decrypted, &mut scratch)
    .expect("scratch-backed randomized OAEP decryption must succeed");
  assert_eq!(&decrypted[..len], plaintext);
  assert_eq!(measurement.finish(), 0);

  let pkcs1v15_plaintext = b"rng private scratch allocation RSAES-PKCS1-v1_5";
  key
    .public_key()
    .encrypt_pkcs1v15_with_random_fill(pkcs1v15_plaintext, &mut ciphertext, fill_rsa_random_with(0x5d))
    .expect("PKCS#1 v1.5 fixture encryption must succeed");

  let measurement = Measurement::start();
  let len = key
    .decrypt_pkcs1v15_with_scratch(&ciphertext, &mut decrypted, &mut scratch)
    .expect("scratch-backed randomized PKCS#1 v1.5 decryption must succeed");
  assert_eq!(&decrypted[..len], pkcs1v15_plaintext);
  assert_eq!(measurement.finish(), 0);
}

fn assert_one_shot_protocol_rejects_fail_before_scratch_allocation() {
  let key = legacy_public_key_from_spki(&pss_spki());
  let sig = pss_signature_sha256();
  let pkcs1_sig = pkcs1v15_signature_sha256();
  let x509_key = legacy_x509_public_key_from_spki(&pss_spki());
  let pss_algorithm_x509_key =
    legacy_x509_public_key_from_spki(&pss_algorithm_spki_from_rsa_encryption_spki(&pss_spki()));

  let measurement = Measurement::start();
  assert!(
    key
      .jwt_verifier(RsaJwtAlgorithm::Ps256)
      .verify("none", MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  assert!(
    key
      .jwt_verifier(RsaJwtAlgorithm::Ps256)
      .verify("PS1", MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  assert!(key.verify_cose_algorithm_id(1, MESSAGE_PSS, &sig).is_err());
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  assert!(key.verify_cose_algorithm_id(-65535, MESSAGE_PSS, &sig).is_err());
  assert_eq!(measurement.finish(), 0);

  for impossible_salt_len in [4096, usize::MAX] {
    let measurement = Measurement::start();
    assert!(
      key
        .verify_pss_with_salt_len(RsaPssProfile::Sha256, impossible_salt_len, MESSAGE_PSS, &sig)
        .is_err()
    );
    assert_eq!(measurement.finish(), 0);

    let measurement = Measurement::start();
    assert!(
      key
        .verify_signature(
          RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, impossible_salt_len),
          MESSAGE_PSS,
          &sig,
        )
        .is_err()
    );
    assert_eq!(measurement.finish(), 0);

    let mut scratch = key.public_scratch();
    let measurement = Measurement::start();
    assert!(
      key
        .verify_signature_with_scratch(
          RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, impossible_salt_len),
          MESSAGE_PSS,
          &sig,
          &mut scratch,
        )
        .is_err()
    );
    assert_eq!(measurement.finish(), 0);
  }

  let measurement = Measurement::start();
  assert!(
    x509_key
      .verify_signature_from_x509_algorithm_der(&[0x30, 0x00], MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  assert!(
    x509_key
      .verify_signature_from_x509_algorithm_der(X509_SHA1_WITH_RSA_ENCRYPTION, MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  assert!(
    x509_key
      .verify_signature_from_x509_algorithm_der(X509_PSS_DEFAULT_SHA1_ALGORITHM, MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  assert!(
    x509_key
      .verify_signature_from_x509_algorithm_der(X509_PSS_SHA256_OVERSIZED_SALT_ALGORITHM, MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  assert!(
    x509_key
      .verify_tls13_signature_scheme(0x0401, MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  assert!(
    x509_key
      .verify_tls13_signature_scheme(0x0809, MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  assert!(
    x509_key
      .verify_tls13_signature_scheme(0x0420, MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  assert!(
    x509_key
      .verify_tls_certificate_signature_scheme(0x0201, MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  assert!(
    pss_algorithm_x509_key
      .verify_tls13_signature_scheme(0x0804, MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  assert!(
    pss_algorithm_x509_key
      .verify_tls_certificate_signature_scheme(0x0401, MESSAGE_PKCS1V15, &pkcs1_sig)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  assert!(
    pss_algorithm_x509_key
      .verify_signature_from_x509_algorithm_der(X509_SHA256_WITH_RSA_ENCRYPTION, MESSAGE_PKCS1V15, &pkcs1_sig,)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);

  let measurement = Measurement::start();
  assert!(x509_key.verify_x509_certificate_signature_der(&[0x30, 0x00]).is_err());
  assert_eq!(measurement.finish(), 0);

  let sha1_certificate = x509_certificate(
    &minimal_tbs_certificate(X509_SHA1_WITH_RSA_ENCRYPTION),
    X509_SHA1_WITH_RSA_ENCRYPTION,
    &sig,
  );
  let measurement = Measurement::start();
  assert!(
    x509_key
      .verify_x509_certificate_signature_der(&sha1_certificate)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);

  let pss_default_sha1_certificate = x509_certificate(
    &minimal_tbs_certificate(X509_PSS_DEFAULT_SHA1_ALGORITHM),
    X509_PSS_DEFAULT_SHA1_ALGORITHM,
    &sig,
  );
  let measurement = Measurement::start();
  assert!(
    x509_key
      .verify_x509_certificate_signature_der(&pss_default_sha1_certificate)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);

  let pss_oversized_salt_certificate = x509_certificate(
    &minimal_tbs_certificate(X509_PSS_SHA256_OVERSIZED_SALT_ALGORITHM),
    X509_PSS_SHA256_OVERSIZED_SALT_ALGORITHM,
    &sig,
  );
  let measurement = Measurement::start();
  assert!(
    x509_key
      .verify_x509_certificate_signature_der(&pss_oversized_salt_certificate)
      .is_err()
  );
  assert_eq!(measurement.finish(), 0);
}
