use super::*;
use crate::auth::rsa::tests::*;

#[cfg(feature = "getrandom")]
#[test]
fn private_key_sign_signature_profile_and_explicit_pss_salt_len_roundtrip() {
  let key = prevalidated_test_private_key();
  let message = b"rscrypto RSA typed signing profile roundtrip";

  let pkcs1_profile = RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha384);
  let mut pkcs1v15_signature = vec![0u8; key.signature_len()];
  key
    .sign_signature(pkcs1_profile, message, &mut pkcs1v15_signature)
    .expect("typed PKCS#1 v1.5 profile must sign");
  key
    .public_key()
    .verify_signature(pkcs1_profile, message, &pkcs1v15_signature)
    .expect("typed PKCS#1 v1.5 signature must verify");

  let pss_zero_salt = RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, 0);
  let mut pss_signature = vec![0u8; key.signature_len()];
  key
    .sign_signature(pss_zero_salt, message, &mut pss_signature)
    .expect("typed zero-salt PSS profile must sign");
  key
    .public_key()
    .verify_signature(pss_zero_salt, message, &pss_signature)
    .expect("typed zero-salt PSS signature must verify");

  key
    .sign_pss_with_salt_len(RsaPssProfile::Sha512, 24, message, &mut pss_signature)
    .expect("explicit-salt-length PSS signing must succeed");
  key
    .public_key()
    .verify_pss_with_salt_len(RsaPssProfile::Sha512, 24, message, &pss_signature)
    .expect("explicit-salt-length PSS signature must verify");

  let mut scratch = key.private_scratch();
  key
    .sign_pss_with_salt_len_and_scratch(RsaPssProfile::Sha384, 16, message, &mut pss_signature, &mut scratch)
    .expect("explicit-salt-length PSS signing with caller scratch must succeed");
  key
    .public_key()
    .verify_pss_with_salt_len(RsaPssProfile::Sha384, 16, message, &pss_signature)
    .expect("caller-scratch explicit-salt-length PSS signature must verify");

  pss_signature.fill(0xa5);
  assert_eq!(
    key.sign_pss_with_salt_len(RsaPssProfile::Sha512, usize::MAX, message, &mut pss_signature),
    Err(RsaPrivateOpError::MessageTooLong)
  );
  assert!(is_zero_unsigned_be(&pss_signature));

  pss_signature.fill(0xa5);
  assert_eq!(
    key.sign_pss_with_salt_len_and_scratch(
      RsaPssProfile::Sha512,
      usize::MAX,
      message,
      &mut pss_signature,
      &mut scratch,
    ),
    Err(RsaPrivateOpError::MessageTooLong)
  );
  assert!(is_zero_unsigned_be(&pss_signature));
}

#[test]
fn pss_encoded_sha256_accepts_valid_encoding() {
  let message = b"rsa-pss-encoded-message";
  let encoded = pss_encoded_sha256(message, 2047, Sha256::OUTPUT_SIZE);

  assert_eq!(verify_pss_encoded::<Sha256>(message, &encoded, 2047), Ok(()));
}

#[test]
fn pss_encoded_sha256_accepts_explicit_zero_salt_and_rejects_leftmost_unused_bit() {
  let message = b"rsa-pss-encoded-message";
  let encoded = pss_encoded_sha256(message, 2047, 0);
  let mut db = vec![0u8; encoded.len()];
  let mut db_mask = vec![0u8; encoded.len()];

  assert_eq!(
    verify_pss_encoded_with_scratch::<Sha256>(message, &encoded, 2047, 0, &mut db, &mut db_mask),
    Ok(())
  );

  let mut bad_unused_bit = encoded;
  bad_unused_bit[0] |= 0x80;
  assert_eq!(
    verify_pss_encoded_with_scratch::<Sha256>(message, &bad_unused_bit, 2047, 0, &mut db, &mut db_mask),
    Err(VerificationError::new())
  );
}

#[test]
fn pss_encoded_sha256_rejects_impossible_em_bits_without_panic() {
  let message = b"rsa-pss-encoded-message";
  let encoded = pss_encoded_sha256(message, 2047, Sha256::OUTPUT_SIZE);
  let mut db = vec![0u8; encoded.len()];
  let mut db_mask = vec![0u8; encoded.len()];

  assert_eq!(
    verify_pss_encoded_with_scratch::<Sha256>(
      message,
      &encoded,
      encoded.len().strict_mul(8).strict_add(1),
      Sha256::OUTPUT_SIZE,
      &mut db,
      &mut db_mask
    ),
    Err(VerificationError::new())
  );
  assert_eq!(
    verify_pss_encoded_with_scratch::<Sha256>(message, &encoded, 0, Sha256::OUTPUT_SIZE, &mut db, &mut db_mask),
    Err(VerificationError::new())
  );
}

#[test]
fn pss_encoded_sha256_rejects_invalid_salt_length_trailer_and_masked_db() {
  let message = b"rsa-pss-encoded-message";

  let short_salt = pss_encoded_sha256(message, 2047, Sha256::OUTPUT_SIZE.strict_sub(1));
  assert!(verify_pss_encoded::<Sha256>(message, &short_salt, 2047).is_err());

  let mut bad_trailer = pss_encoded_sha256(message, 2047, Sha256::OUTPUT_SIZE);
  if let Some(byte) = bad_trailer.last_mut() {
    *byte = 0xbd;
  }
  assert!(verify_pss_encoded::<Sha256>(message, &bad_trailer, 2047).is_err());

  let mut bad_db = pss_encoded_sha256(message, 2047, Sha256::OUTPUT_SIZE);
  if let Some(byte) = bad_db.get_mut(16) {
    *byte ^= 0x40;
  }
  assert!(verify_pss_encoded::<Sha256>(message, &bad_db, 2047).is_err());
}

#[test]
fn pss_encoded_sha256_rejects_message_hash_mismatch() {
  let encoded = pss_encoded_sha256(b"rsa-pss-encoded-message", 2047, Sha256::OUTPUT_SIZE);

  assert!(verify_pss_encoded::<Sha256>(b"wrong message", &encoded, 2047).is_err());
}

#[test]
fn pss_encoded_message_oracle_failures_are_opaque() {
  let reject = |result| assert_eq!(result, Err(VerificationError::new()));
  let key = RsaPublicKey::from_spki_der_with_policy(
    include_bytes!("../../../../tests/vectors/rsa/rsa2048_pss_spki.der"),
    &RsaPublicKeyPolicy::modulus_2048_to_8192(),
  )
  .expect("fixture SPKI key must decode");
  let mut encoded = vec![0u8; key.modulus().len()];
  key
    .public_operation(
      include_bytes!("../../../../tests/vectors/rsa/rsa2048_pss_signature_sha256.sig"),
      &mut encoded,
    )
    .expect("the PSS signature fixture must decode to its encoded message");
  let em_bits = key.modulus_bits().strict_sub(1);

  verify_pss_encoded::<Sha256>(b"rscrypto RSA-PSS verification fixture", &encoded, em_bits)
    .expect("the valid PSS encoded message must verify");

  let mut bad_trailer = encoded.clone();
  *bad_trailer
    .last_mut()
    .expect("the PSS encoded message must contain a trailer byte") ^= 0x01;
  reject(verify_pss_encoded::<Sha256>(
    b"rscrypto RSA-PSS verification fixture",
    &bad_trailer,
    em_bits,
  ));

  let mut bad_unused_bit = encoded.clone();
  bad_unused_bit[0] |= 0x80;
  reject(verify_pss_encoded::<Sha256>(
    b"rscrypto RSA-PSS verification fixture",
    &bad_unused_bit,
    em_bits,
  ));

  let mut bad_masked_db = encoded.clone();
  bad_masked_db[17] ^= 0x40;
  reject(verify_pss_encoded::<Sha256>(
    b"rscrypto RSA-PSS verification fixture",
    &bad_masked_db,
    em_bits,
  ));

  let h_offset = encoded
    .len()
    .strict_sub(RsaPssProfile::Sha256.digest_len())
    .strict_sub(1);
  let mut bad_hash = encoded.clone();
  bad_hash[h_offset] ^= 0x20;
  reject(verify_pss_encoded::<Sha256>(
    b"rscrypto RSA-PSS verification fixture",
    &bad_hash,
    em_bits,
  ));

  let mut short_db = vec![0u8; 8];
  let mut short_mask = vec![0u8; 8];
  reject(verify_pss_encoded_with_scratch::<Sha256>(
    b"rscrypto RSA-PSS verification fixture",
    &encoded,
    em_bits,
    Sha256::OUTPUT_SIZE,
    &mut short_db,
    &mut short_mask,
  ));

  reject(verify_pss_encoded::<Sha256>(b"wrong message", &encoded, em_bits));
  reject(verify_pss_encoded::<Sha256>(
    b"rscrypto RSA-PSS verification fixture",
    &[0xbc],
    8,
  ));
}
