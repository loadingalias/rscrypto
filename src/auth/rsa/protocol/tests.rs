use super::*;
use crate::auth::rsa::tests::*;

#[cfg(feature = "getrandom")]
#[test]
fn private_key_protocol_signing_helpers_roundtrip_and_reject_confusion() {
  let key = prevalidated_test_private_key();
  let message = b"rscrypto RSA primitive protocol signing helper roundtrip";
  let mut signature = vec![0u8; key.signature_len()];
  let mut scratch = key.private_scratch();

  let x509_pkcs1v15 = algorithm_identifier(SHA384_WITH_RSA_ENCRYPTION_OID, Some(&null()));
  key
    .sign_x509_signature_algorithm_der(&x509_pkcs1v15, message, &mut signature)
    .expect("X.509 PKCS#1 v1.5 algorithm must sign");
  let x509_pkcs1v15_profile = RsaSignatureProfile::from_x509_signature_algorithm_der(&x509_pkcs1v15)
    .expect("X.509 PKCS#1 v1.5 algorithm must parse");
  key
    .public_key()
    .verify_signature(x509_pkcs1v15_profile, message, &signature)
    .expect("X.509 PKCS#1 v1.5 signature must verify");
  key
    .sign_x509_signature_algorithm_der_with_scratch(&x509_pkcs1v15, message, &mut signature, &mut scratch)
    .expect("X.509 PKCS#1 v1.5 algorithm must sign with caller scratch");
  key
    .public_key()
    .verify_signature(x509_pkcs1v15_profile, message, &signature)
    .expect("caller-scratch X.509 PKCS#1 v1.5 signature must verify");

  let x509_pss = x509_pss_algorithm(RsaPssProfile::Sha256, 20);
  key
    .sign_x509_signature_algorithm_der(&x509_pss, message, &mut signature)
    .expect("X.509 PSS algorithm must sign");
  let x509_pss_profile =
    RsaSignatureProfile::from_x509_signature_algorithm_der(&x509_pss).expect("X.509 PSS algorithm must parse");
  key
    .public_key()
    .verify_signature(x509_pss_profile, message, &signature)
    .expect("X.509 PSS signature must verify");
  key
    .sign_x509_signature_algorithm_der_with_scratch(&x509_pss, message, &mut signature, &mut scratch)
    .expect("X.509 PSS algorithm must sign with caller scratch");
  key
    .public_key()
    .verify_signature(x509_pss_profile, message, &signature)
    .expect("caller-scratch X.509 PSS signature must verify");

  key
    .sign_tls13_signature_scheme(0x0804, message, &mut signature)
    .expect("TLS 1.3 rsa_pss_rsae_sha256 scheme must sign");
  key
    .public_key()
    .verify_signature(
      RsaSignatureProfile::from_tls13_signature_scheme(0x0804)
        .expect("TLS 1.3 rsa_pss_rsae_sha256 scheme must map to a profile"),
      message,
      &signature,
    )
    .expect("TLS 1.3 rsa_pss_rsae_sha256 signature must verify");
  key
    .sign_tls13_signature_scheme_with_scratch(0x0804, message, &mut signature, &mut scratch)
    .expect("TLS 1.3 scheme must sign with caller scratch");
  key
    .public_key()
    .verify_signature(
      RsaSignatureProfile::from_tls13_signature_scheme(0x0804)
        .expect("TLS 1.3 rsa_pss_rsae_sha256 scheme must map to a profile"),
      message,
      &signature,
    )
    .expect("caller-scratch TLS 1.3 signature must verify");

  key
    .sign_tls_certificate_signature_scheme(0x0501, message, &mut signature)
    .expect("TLS certificate rsa_pkcs1_sha384 scheme must sign");
  key
    .public_key()
    .verify_signature(
      RsaSignatureProfile::from_tls_certificate_signature_scheme(0x0501)
        .expect("TLS certificate rsa_pkcs1_sha384 scheme must map to a profile"),
      message,
      &signature,
    )
    .expect("TLS certificate rsa_pkcs1_sha384 signature must verify");
  key
    .sign_tls_certificate_signature_scheme_with_scratch(0x0501, message, &mut signature, &mut scratch)
    .expect("TLS certificate scheme must sign with caller scratch");
  key
    .public_key()
    .verify_signature(
      RsaSignatureProfile::from_tls_certificate_signature_scheme(0x0501)
        .expect("TLS certificate rsa_pkcs1_sha384 scheme must map to a profile"),
      message,
      &signature,
    )
    .expect("caller-scratch TLS certificate signature must verify");

  let jwt_algorithm = RsaJwtAlgorithm::Ps512;
  key
    .jwt_signer(jwt_algorithm)
    .try_sign_into(message, &mut signature)
    .expect("JWT PS512 signer must sign");
  key
    .public_key()
    .jwt_verifier(jwt_algorithm)
    .verify("PS512", message, &signature)
    .expect("JWT PS512 signature must verify");
  key
    .sign_signature_with_scratch(jwt_algorithm.signature_profile(), message, &mut signature, &mut scratch)
    .expect("JWT PS512 profile must sign with caller scratch");
  key
    .public_key()
    .jwt_verifier(jwt_algorithm)
    .verify("PS512", message, &signature)
    .expect("caller-scratch JWT PS512 signature must verify");

  key
    .sign_cose_algorithm_id(-257, message, &mut signature)
    .expect("COSE RS256 algorithm must sign");
  key
    .public_key()
    .verify_cose_algorithm_id(-257, message, &signature)
    .expect("COSE RS256 signature must verify");
  key
    .sign_cose_algorithm_id_with_scratch(-257, message, &mut signature, &mut scratch)
    .expect("COSE RS256 algorithm must sign with caller scratch");
  key
    .public_key()
    .verify_cose_algorithm_id(-257, message, &signature)
    .expect("caller-scratch COSE RS256 signature must verify");

  signature.fill(0xa5);
  assert_eq!(
    key.sign_tls13_signature_scheme(0x0401, message, &mut signature),
    Err(RsaPrivateOpError::UnsupportedAlgorithm)
  );
  assert!(is_zero_unsigned_be(&signature));
  signature.fill(0xa5);
  assert_eq!(
    key.sign_tls13_signature_scheme_with_scratch(0x0401, message, &mut signature, &mut scratch),
    Err(RsaPrivateOpError::UnsupportedAlgorithm)
  );
  assert!(is_zero_unsigned_be(&signature));
  signature.fill(0xa5);
  assert_eq!(
    key.sign_tls_certificate_signature_scheme(0x0201, message, &mut signature),
    Err(RsaPrivateOpError::UnsupportedAlgorithm)
  );
  assert!(is_zero_unsigned_be(&signature));
  signature.fill(0xa5);
  assert_eq!(
    key.sign_tls_certificate_signature_scheme_with_scratch(0x0201, message, &mut signature, &mut scratch),
    Err(RsaPrivateOpError::UnsupportedAlgorithm)
  );
  assert!(is_zero_unsigned_be(&signature));
  signature.fill(0xa5);
  assert_eq!(
    key.sign_cose_algorithm_id(-7, message, &mut signature),
    Err(RsaPrivateOpError::UnsupportedAlgorithm)
  );
  assert!(is_zero_unsigned_be(&signature));
  signature.fill(0xa5);
  assert_eq!(
    key.sign_cose_algorithm_id_with_scratch(-7, message, &mut signature, &mut scratch),
    Err(RsaPrivateOpError::UnsupportedAlgorithm)
  );
  assert!(is_zero_unsigned_be(&signature));

  let sha1 = algorithm_identifier(SHA1_WITH_RSA_ENCRYPTION_OID, Some(&null()));
  signature.fill(0xa5);
  assert_eq!(
    key.sign_x509_signature_algorithm_der(&sha1, message, &mut signature),
    Err(RsaPrivateOpError::UnsupportedAlgorithm)
  );
  assert!(is_zero_unsigned_be(&signature));
  signature.fill(0xa5);
  assert_eq!(
    key.sign_x509_signature_algorithm_der_with_scratch(&sha1, message, &mut signature, &mut scratch),
    Err(RsaPrivateOpError::UnsupportedAlgorithm)
  );
  assert!(is_zero_unsigned_be(&signature));
  signature.fill(0xa5);
  assert_eq!(
    key.sign_x509_signature_algorithm_der(&[0x30, 0x00], message, &mut signature),
    Err(RsaPrivateOpError::UnsupportedAlgorithm)
  );
  assert!(is_zero_unsigned_be(&signature));
  signature.fill(0xa5);
  assert_eq!(
    key.sign_x509_signature_algorithm_der_with_scratch(&[0x30, 0x00], message, &mut signature, &mut scratch),
    Err(RsaPrivateOpError::UnsupportedAlgorithm)
  );
  assert!(is_zero_unsigned_be(&signature));
}

#[test]
fn protocol_adapter_mappings_are_explicit_and_reject_confusion() {
  assert_eq!(RsaJwtAlgorithm::Ps256.as_str(), "PS256");
  assert_eq!(
    RsaJwtAlgorithm::Ps256.signature_profile(),
    RsaSignatureProfile::pss(RsaPssProfile::Sha256)
  );
  assert_eq!(RsaJwtAlgorithm::Rs512.as_str(), "RS512");
  assert_eq!(
    RsaJwtAlgorithm::Rs512.signature_profile(),
    RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha512)
  );

  assert_eq!(
    RsaSignatureProfile::from_cose_algorithm_id(-37),
    Ok(RsaSignatureProfile::pss(RsaPssProfile::Sha256))
  );
  assert_eq!(
    RsaSignatureProfile::from_cose_algorithm_id(-259),
    Ok(RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha512))
  );
  for algorithm in [0, 1, -7, -65535, i64::from(i32::MAX)] {
    assert_eq!(
      RsaSignatureProfile::from_cose_algorithm_id(algorithm),
      Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
    );
  }

  assert_eq!(
    RsaSignatureProfile::from_tls13_signature_scheme(0x0804),
    Ok(RsaSignatureProfile::pss(RsaPssProfile::Sha256))
  );
  assert_eq!(
    RsaSignatureProfile::from_tls13_signature_scheme(0x0401),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );
  assert_eq!(
    RsaX509PublicKeyAlgorithm::RsaEncryption.signature_profile_from_tls13_signature_scheme(0x0809),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );
  assert_eq!(
    RsaX509PublicKeyAlgorithm::RsaPss.signature_profile_from_tls_certificate_signature_scheme(0x0401),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );

  let pss_default_sha1 = algorithm_identifier(ID_RSASSA_PSS_OID, Some(&tlv(TAG_SEQUENCE, &[])));
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&pss_default_sha1),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );

  let pkcs1v15 = algorithm_identifier(SHA256_WITH_RSA_ENCRYPTION_OID, Some(&null()));
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&pkcs1v15),
    Ok(RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256))
  );
  let pss = x509_pss_algorithm(RsaPssProfile::Sha256, Sha256::OUTPUT_SIZE);
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&pss),
    Ok(RsaSignatureProfile::pss(RsaPssProfile::Sha256))
  );
  let sha1 = algorithm_identifier(SHA1_WITH_RSA_ENCRYPTION_OID, Some(&null()));
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&sha1),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );
  let absent_params = algorithm_identifier(SHA256_WITH_RSA_ENCRYPTION_OID, None);
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&absent_params),
    Ok(RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256))
  );
}

#[test]
fn x509_certificate_signature_helper_accepts_real_certificate_and_rejects_tamper() {
  let issuer = RsaX509PublicKey::from_spki_der_with_policy(
    &x509_certificate_fixture_public_key(),
    &RsaPublicKeyPolicy::modulus_2048_to_8192(),
  )
  .expect("X.509 issuer fixture must import");
  let certificate = x509_pkcs1v15_certificate_fixture();

  issuer
    .verify_x509_certificate_signature_der(&certificate)
    .expect("fixture certificate signature must verify");

  let mut tampered = certificate;
  if let Some(last) = tampered.last_mut() {
    *last ^= 0x01;
  }
  assert_eq!(
    issuer.verify_x509_certificate_signature_der(&tampered),
    Err(VerificationError::new())
  );
}

#[test]
fn x509_certificate_signature_parser_rejects_trailing_and_noncanonical_der() {
  let certificate = x509_pkcs1v15_certificate_fixture();

  let mut trailing = certificate.clone();
  trailing.extend_from_slice(&null());
  assert_eq!(
    parse_x509_certificate_signature(&trailing).err(),
    Some(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier)
  );

  assert_eq!(
    parse_x509_certificate_signature(&tlv_with_leading_zero_long_len(&certificate)).err(),
    Some(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier)
  );
}
