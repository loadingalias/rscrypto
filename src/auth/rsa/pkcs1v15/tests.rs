use super::*;
use crate::auth::rsa::tests::*;

#[test]
fn pkcs1v15_sha2_digest_info_prefixes_match_rfc8017_der() {
  assert_eq!(
    SHA256_DIGEST_INFO_PREFIX,
    &[
      0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05, 0x00, 0x04, 0x20,
    ],
  );
  assert_eq!(
    SHA384_DIGEST_INFO_PREFIX,
    &[
      0x30, 0x41, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02, 0x05, 0x00, 0x04, 0x30,
    ],
  );
  assert_eq!(
    SHA512_DIGEST_INFO_PREFIX,
    &[
      0x30, 0x51, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x03, 0x05, 0x00, 0x04, 0x40,
    ],
  );
}

#[test]
fn pkcs1v15_encoded_sha256_accepts_exact_der_digest_info() {
  let message = b"rsa-pkcs1v15-encoded-message";
  let encoded = pkcs1v15_encoded_sha256(message, 256);

  assert_eq!(
    verify_pkcs1v15_encoded::<Sha256>(message, &encoded, SHA256_DIGEST_INFO_PREFIX),
    Ok(())
  );
}

#[test]
fn pkcs1v15_encoded_sha256_rejects_short_padding_and_bad_separator() {
  let message = b"rsa-pkcs1v15-encoded-message";
  let digest = Sha256::digest(message);
  let mut short = Vec::new();
  short.push(0);
  short.push(1);
  short.extend(core::iter::repeat_n(0xff, 7));
  short.push(0);
  short.extend_from_slice(SHA256_DIGEST_INFO_PREFIX);
  short.extend_from_slice(&digest);
  assert!(verify_pkcs1v15_encoded::<Sha256>(message, &short, SHA256_DIGEST_INFO_PREFIX).is_err());

  let mut bad_separator = pkcs1v15_encoded_sha256(message, 256);
  let separator = bad_separator
    .len()
    .strict_sub(SHA256_DIGEST_INFO_PREFIX.len())
    .strict_sub(Sha256::OUTPUT_SIZE)
    .strict_sub(1);
  if let Some(byte) = bad_separator.get_mut(separator) {
    *byte = 0xff;
  }
  assert!(verify_pkcs1v15_encoded::<Sha256>(message, &bad_separator, SHA256_DIGEST_INFO_PREFIX).is_err());
}

#[test]
fn pkcs1v15_encoded_sha256_rejects_digest_info_oid_and_digest_mismatches() {
  let message = b"rsa-pkcs1v15-encoded-message";
  let mut bad_oid = pkcs1v15_encoded_sha256(message, 256);
  let digest_info = bad_oid
    .len()
    .strict_sub(SHA256_DIGEST_INFO_PREFIX.len())
    .strict_sub(Sha256::OUTPUT_SIZE);
  if let Some(byte) = bad_oid.get_mut(digest_info.strict_add(14)) {
    *byte ^= 0x01;
  }
  assert!(verify_pkcs1v15_encoded::<Sha256>(message, &bad_oid, SHA256_DIGEST_INFO_PREFIX).is_err());

  let mut bad_digest = pkcs1v15_encoded_sha256(message, 256);
  if let Some(byte) = bad_digest.last_mut() {
    *byte ^= 0x01;
  }
  assert!(verify_pkcs1v15_encoded::<Sha256>(message, &bad_digest, SHA256_DIGEST_INFO_PREFIX).is_err());
}

#[test]
fn pkcs1v15_encoded_sha256_rejects_historical_forgery_shapes() {
  let message = b"rsa-pkcs1v15-encoded-message";
  let digest = Sha256::digest(message);

  let mut early_separator = Vec::new();
  early_separator.push(0);
  early_separator.push(1);
  early_separator.extend(core::iter::repeat_n(0xff, 8));
  early_separator.push(0);
  early_separator.extend_from_slice(SHA256_DIGEST_INFO_PREFIX);
  early_separator.extend_from_slice(&digest);
  early_separator.extend(core::iter::repeat_n(0xaa, 256usize.strict_sub(early_separator.len())));
  assert!(verify_pkcs1v15_encoded::<Sha256>(message, &early_separator, SHA256_DIGEST_INFO_PREFIX).is_err());

  let mut missing_null_prefix = [
    0x30, 0x2f, 0x30, 0x0b, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x04, 0x20,
  ]
  .to_vec();
  missing_null_prefix.extend_from_slice(&digest);
  let mut missing_null = Vec::new();
  missing_null.push(0);
  missing_null.push(1);
  missing_null.extend(core::iter::repeat_n(
    0xff,
    256usize.strict_sub(missing_null_prefix.len()).strict_sub(3),
  ));
  missing_null.push(0);
  missing_null.extend_from_slice(&missing_null_prefix);
  assert!(verify_pkcs1v15_encoded::<Sha256>(message, &missing_null, SHA256_DIGEST_INFO_PREFIX).is_err());

  let mut ber_indefinite_digest_info = Vec::new();
  ber_indefinite_digest_info.extend_from_slice(&[0x30, 0x80]);
  ber_indefinite_digest_info.extend_from_slice(&SHA256_DIGEST_INFO_PREFIX[2..SHA256_DIGEST_INFO_PREFIX.len()]);
  ber_indefinite_digest_info.extend_from_slice(&digest);
  ber_indefinite_digest_info.extend_from_slice(&[0x00, 0x00]);
  let mut ber = Vec::new();
  ber.push(0);
  ber.push(1);
  ber.extend(core::iter::repeat_n(
    0xff,
    256usize.strict_sub(ber_indefinite_digest_info.len()).strict_sub(3),
  ));
  ber.push(0);
  ber.extend_from_slice(&ber_indefinite_digest_info);
  assert!(verify_pkcs1v15_encoded::<Sha256>(message, &ber, SHA256_DIGEST_INFO_PREFIX).is_err());

  let mut padding_hole = pkcs1v15_encoded_sha256(message, 256);
  if let Some(byte) = padding_hole.get_mut(10) {
    *byte = 0;
  }
  assert!(verify_pkcs1v15_encoded::<Sha256>(message, &padding_hole, SHA256_DIGEST_INFO_PREFIX).is_err());
}

#[test]
fn pkcs1v15_encoded_message_oracle_failures_are_opaque() {
  let reject = |result| assert_eq!(result, Err(VerificationError::new()));
  let check =
    |message: &[u8], encoded: &[u8]| verify_pkcs1v15_encoded::<Sha256>(message, encoded, SHA256_DIGEST_INFO_PREFIX);
  let key = RsaPublicKey::from_spki_der_with_policy(
    include_bytes!("../../../../tests/vectors/rsa/rsa2048_pkcs1v15_spki.der"),
    &RsaPublicKeyPolicy::modulus_2048_to_8192(),
  )
  .expect("fixture SPKI key must decode");
  let mut encoded = vec![0u8; key.modulus().len()];
  key
    .public_operation(
      include_bytes!("../../../../tests/vectors/rsa/rsa2048_pkcs1v15_signature_sha256.sig"),
      &mut encoded,
    )
    .expect("the PKCS#1 v1.5 signature fixture must decode to its encoded message");

  check(b"rscrypto RSA-PKCS1-v1_5 verification fixture", &encoded)
    .expect("the valid PKCS#1 v1.5 encoded message must verify");

  let mut bad_prefix = encoded.clone();
  bad_prefix[0] = 0x01;
  reject(check(b"rscrypto RSA-PKCS1-v1_5 verification fixture", &bad_prefix));

  let mut bad_block_type = encoded.clone();
  bad_block_type[1] = 0x02;
  reject(check(b"rscrypto RSA-PKCS1-v1_5 verification fixture", &bad_block_type));

  let mut bad_padding = encoded.clone();
  bad_padding[2] = 0x00;
  reject(check(b"rscrypto RSA-PKCS1-v1_5 verification fixture", &bad_padding));

  let digest_info_len = 19usize.strict_add(32);
  let separator_index = encoded.len().strict_sub(digest_info_len).strict_sub(1);
  let mut bad_separator = encoded.clone();
  bad_separator[separator_index] = 0xff;
  reject(check(b"rscrypto RSA-PKCS1-v1_5 verification fixture", &bad_separator));

  let mut bad_digest_info = encoded.clone();
  bad_digest_info[separator_index.strict_add(1)] ^= 0x01;
  reject(check(b"rscrypto RSA-PKCS1-v1_5 verification fixture", &bad_digest_info));

  let mut bad_digest = encoded.clone();
  *bad_digest
    .last_mut()
    .expect("the PKCS#1 v1.5 encoded message must contain a digest byte") ^= 0x80;
  reject(check(b"rscrypto RSA-PKCS1-v1_5 verification fixture", &bad_digest));

  reject(check(b"wrong message", &encoded));
  reject(check(b"rscrypto RSA-PKCS1-v1_5 verification fixture", &[0, 1, 0xff]));
}
