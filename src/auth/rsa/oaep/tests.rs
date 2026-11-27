use super::*;
use crate::auth::rsa::tests::*;

#[test]
fn oaep_decode_rejects_padding_oracle_classes_opaquely() {
  let label = b"rscrypto-oaep-label";
  let message = b"oaep message";
  let seed = [0x3cu8; Sha256::OUTPUT_SIZE];
  let decoded_db = decoded_oaep_sha256_db(label, message, 12);

  let mut valid = masked_oaep_sha256_from_decoded_db(&seed, &decoded_db);
  let mut out = vec![0u8; message.len()];
  let len =
    decode_oaep::<Sha256>(label, &mut valid, &mut out).expect("the canonical OAEP encoded-message fixture must decode");
  assert_eq!(len, message.len());
  assert_eq!(out, message);

  let mut bad_leading = masked_oaep_sha256_from_decoded_db(&seed, &decoded_db);
  bad_leading[0] = 1;
  assert_eq!(
    decode_oaep::<Sha256>(label, &mut bad_leading, &mut out),
    Err(RsaPrivateOpError::DecryptionFailed)
  );

  let mut bad_label_db = decoded_db.clone();
  bad_label_db[0] ^= 0x80;
  let mut bad_label = masked_oaep_sha256_from_decoded_db(&seed, &bad_label_db);
  assert_eq!(
    decode_oaep::<Sha256>(label, &mut bad_label, &mut out),
    Err(RsaPrivateOpError::DecryptionFailed)
  );

  let mut bad_padding_db = decoded_db.clone();
  bad_padding_db[Sha256::OUTPUT_SIZE.strict_add(3)] = 0x7f;
  let mut bad_padding = masked_oaep_sha256_from_decoded_db(&seed, &bad_padding_db);
  assert_eq!(
    decode_oaep::<Sha256>(label, &mut bad_padding, &mut out),
    Err(RsaPrivateOpError::DecryptionFailed)
  );

  let mut missing_separator_db = decoded_db.clone();
  missing_separator_db[Sha256::OUTPUT_SIZE..].fill(0);
  let mut missing_separator = masked_oaep_sha256_from_decoded_db(&seed, &missing_separator_db);
  assert_eq!(
    decode_oaep::<Sha256>(label, &mut missing_separator, &mut out),
    Err(RsaPrivateOpError::DecryptionFailed)
  );

  let mut valid_short_out = masked_oaep_sha256_from_decoded_db(&seed, &decoded_db);
  let mut short_out = vec![0u8; message.len().strict_sub(1)];
  assert_eq!(
    decode_oaep::<Sha256>(label, &mut valid_short_out, &mut short_out),
    Err(RsaPrivateOpError::DecryptionFailed)
  );
}

#[test]
fn decrypt_rejects_output_below_public_capacity_before_decryption() {
  let key = prevalidated_test_private_key();
  let k = key.signature_len();
  let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
  let valid = RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse);
  // 2 * 2 != 1 mod n, so any private operation would reject this pair.
  let invalid = RsaBlindingPair::new(&blinding_factor, &blinding_factor);
  let mut scratch = key.private_scratch();
  let label = b"rscrypto-decrypt-capacity";

  // RFC 8017 sections 7.1.1 and 7.2.1 bound the message at k - 2hLen - 2
  // for OAEP and k - 11 for PKCS#1 v1.5.
  let oaep_max = k.strict_sub(2 * Sha256::OUTPUT_SIZE).strict_sub(2);
  let pkcs1v15_max = k.strict_sub(11);

  let oaep_message = vec![0x3c; oaep_max];
  let mut oaep_valid = vec![0u8; k];
  key
    .public_key()
    .encrypt_oaep_with_random_fill(
      RsaOaepProfile::Sha256,
      label,
      &oaep_message,
      &mut oaep_valid,
      seed_fill(&[0x5e; Sha256::OUTPUT_SIZE]),
    )
    .expect("a maximum-length OAEP message must encrypt");
  let mut oaep_invalid = oaep_valid.clone();
  *oaep_invalid.last_mut().expect("OAEP ciphertext is modulus-sized") ^= 0x01;

  let pkcs1v15_message = vec![0x4d; pkcs1v15_max];
  let mut pkcs1v15_valid = vec![0u8; k];
  key
    .public_key()
    .encrypt_pkcs1v15_with_random_fill(&pkcs1v15_message, &mut pkcs1v15_valid, seed_fill(&[0x6b; 8]))
    .expect("a maximum-length PKCS#1 v1.5 message must encrypt");
  let mut pkcs1v15_invalid = pkcs1v15_valid.clone();
  *pkcs1v15_invalid
    .last_mut()
    .expect("PKCS#1 v1.5 ciphertext is modulus-sized") ^= 0x01;

  let mut oaep = |ciphertext: &[u8], blinding: RsaBlindingPair<'_>, out: &mut [u8]| {
    let profile = RsaOaepProfile::Sha256;
    let mut results = Vec::new();
    results.push(decrypt_into_canary(out, |out| {
      key.decrypt_oaep_with_blinding_factor(profile, label, ciphertext, blinding, out)
    }));
    results.push(decrypt_into_canary(out, |out| {
      key.decrypt_oaep_with_blinding_factor_and_scratch(profile, label, ciphertext, blinding, out, &mut scratch)
    }));
    #[cfg(feature = "getrandom")]
    {
      results.push(decrypt_into_canary(out, |out| {
        key.decrypt_oaep(profile, label, ciphertext, out)
      }));
      results.push(decrypt_into_canary(out, |out| {
        key.decrypt_oaep_with_scratch(profile, label, ciphertext, out, &mut scratch)
      }));
    }
    results
  };

  let mut out = vec![0u8; oaep_max];
  for outcome in oaep(&oaep_valid, valid, &mut out) {
    assert_eq!(outcome, (Ok(oaep_max), true));
    assert_eq!(out, oaep_message);
  }
  let mut short = vec![0u8; oaep_max.strict_sub(1)];
  for ciphertext in [&oaep_valid, &oaep_invalid] {
    for blinding in [valid, invalid] {
      for outcome in oaep(ciphertext, blinding, &mut short) {
        assert_eq!(outcome, (Err(RsaPrivateOpError::InvalidLength), true));
      }
    }
  }

  let mut pkcs1v15 = |ciphertext: &[u8], blinding: RsaBlindingPair<'_>, out: &mut [u8]| {
    let mut results = Vec::new();
    results.push(decrypt_into_canary(out, |out| {
      key.decrypt_pkcs1v15_with_blinding_factor(ciphertext, blinding, out)
    }));
    results.push(decrypt_into_canary(out, |out| {
      key.decrypt_pkcs1v15_with_blinding_factor_and_scratch(ciphertext, blinding, out, &mut scratch)
    }));
    #[cfg(feature = "getrandom")]
    {
      results.push(decrypt_into_canary(out, |out| key.decrypt_pkcs1v15(ciphertext, out)));
      results.push(decrypt_into_canary(out, |out| {
        key.decrypt_pkcs1v15_with_scratch(ciphertext, out, &mut scratch)
      }));
    }
    results
  };

  let mut out = vec![0u8; pkcs1v15_max];
  for outcome in pkcs1v15(&pkcs1v15_valid, valid, &mut out) {
    assert_eq!(outcome, (Ok(pkcs1v15_max), true));
    assert_eq!(out, pkcs1v15_message);
  }
  let mut short = vec![0u8; pkcs1v15_max.strict_sub(1)];
  for ciphertext in [&pkcs1v15_valid, &pkcs1v15_invalid] {
    for blinding in [valid, invalid] {
      for outcome in pkcs1v15(ciphertext, blinding, &mut short) {
        assert_eq!(outcome, (Err(RsaPrivateOpError::InvalidLength), true));
      }
    }
  }
}

#[test]
fn oaep_decrypt_api_rejects_same_width_oracle_classes_opaquely() {
  let key = prevalidated_test_private_key();
  let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
  let label = b"rscrypto-oaep-label";
  let plaintext = b"oaep decrypt oracle regression";

  macro_rules! assert_profile_rejects_opaquely {
    ($profile:expr, $seed:expr) => {{
      let mut ciphertext = vec![0u8; key.signature_len()];
      key
        .public_key()
        .encrypt_oaep_with_random_fill($profile, label, plaintext, &mut ciphertext, seed_fill(&$seed))
        .unwrap();

      let mut out = vec![0u8; key.signature_len()];
      let len = key
        .decrypt_oaep_with_blinding_factor(
          $profile,
          label,
          &ciphertext,
          RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
          &mut out,
        )
        .unwrap();
      assert_eq!(&out[..len], plaintext);

      let mut scratch = key.private_scratch();
      let len = key
        .decrypt_oaep_with_blinding_factor_and_scratch(
          $profile,
          label,
          &ciphertext,
          RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
          &mut out,
          &mut scratch,
        )
        .unwrap();
      assert_eq!(&out[..len], plaintext);

      #[cfg(feature = "getrandom")]
      {
        let len = key.decrypt_oaep($profile, label, &ciphertext, &mut out).unwrap();
        assert_eq!(&out[..len], plaintext);

        let len = key
          .decrypt_oaep_with_scratch($profile, label, &ciphertext, &mut out, &mut scratch)
          .unwrap();
        assert_eq!(&out[..len], plaintext);
      }

      out.fill(0xa5);
      assert_eq!(
        key.decrypt_oaep_with_blinding_factor(
          $profile,
          b"wrong label",
          &ciphertext,
          RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
          &mut out,
        ),
        Err(RsaPrivateOpError::DecryptionFailed)
      );
      assert!(is_zero_unsigned_be(&out));
      out.fill(0xa5);
      assert_eq!(
        key.decrypt_oaep_with_blinding_factor_and_scratch(
          $profile,
          b"wrong label",
          &ciphertext,
          RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
          &mut out,
          &mut scratch,
        ),
        Err(RsaPrivateOpError::DecryptionFailed)
      );
      assert!(is_zero_unsigned_be(&out));

      #[cfg(feature = "getrandom")]
      {
        out.fill(0xa5);
        assert_eq!(
          key.decrypt_oaep($profile, b"wrong label", &ciphertext, &mut out),
          Err(RsaPrivateOpError::DecryptionFailed)
        );
        assert!(is_zero_unsigned_be(&out));
        out.fill(0xa5);
        assert_eq!(
          key.decrypt_oaep_with_scratch($profile, b"wrong label", &ciphertext, &mut out, &mut scratch),
          Err(RsaPrivateOpError::DecryptionFailed)
        );
        assert!(is_zero_unsigned_be(&out));
      }

      let mut tampered_tail = ciphertext.clone();
      *tampered_tail.last_mut().unwrap() ^= 0x01;
      out.fill(0xa5);
      assert_eq!(
        key.decrypt_oaep_with_blinding_factor(
          $profile,
          label,
          &tampered_tail,
          RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
          &mut out,
        ),
        Err(RsaPrivateOpError::DecryptionFailed)
      );
      assert!(is_zero_unsigned_be(&out));
      out.fill(0xa5);
      assert_eq!(
        key.decrypt_oaep_with_blinding_factor_and_scratch(
          $profile,
          label,
          &tampered_tail,
          RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
          &mut out,
          &mut scratch,
        ),
        Err(RsaPrivateOpError::DecryptionFailed)
      );
      assert!(is_zero_unsigned_be(&out));

      #[cfg(feature = "getrandom")]
      {
        out.fill(0xa5);
        assert_eq!(
          key.decrypt_oaep($profile, label, &tampered_tail, &mut out),
          Err(RsaPrivateOpError::DecryptionFailed)
        );
        assert!(is_zero_unsigned_be(&out));
        out.fill(0xa5);
        assert_eq!(
          key.decrypt_oaep_with_scratch($profile, label, &tampered_tail, &mut out, &mut scratch),
          Err(RsaPrivateOpError::DecryptionFailed)
        );
        assert!(is_zero_unsigned_be(&out));
      }

      let zero_representative = vec![0u8; key.signature_len()];
      out.fill(0xa5);
      assert_eq!(
        key.decrypt_oaep_with_blinding_factor(
          $profile,
          label,
          &zero_representative,
          RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
          &mut out,
        ),
        Err(RsaPrivateOpError::DecryptionFailed)
      );
      assert!(is_zero_unsigned_be(&out));
      out.fill(0xa5);
      assert_eq!(
        key.decrypt_oaep_with_blinding_factor_and_scratch(
          $profile,
          label,
          &zero_representative,
          RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
          &mut out,
          &mut scratch,
        ),
        Err(RsaPrivateOpError::DecryptionFailed)
      );
      assert!(is_zero_unsigned_be(&out));

      #[cfg(feature = "getrandom")]
      {
        out.fill(0xa5);
        assert_eq!(
          key.decrypt_oaep($profile, label, &zero_representative, &mut out),
          Err(RsaPrivateOpError::DecryptionFailed)
        );
        assert!(is_zero_unsigned_be(&out));
        out.fill(0xa5);
        assert_eq!(
          key.decrypt_oaep_with_scratch($profile, label, &zero_representative, &mut out, &mut scratch),
          Err(RsaPrivateOpError::DecryptionFailed)
        );
        assert!(is_zero_unsigned_be(&out));
      }
    }};
  }

  assert_profile_rejects_opaquely!(RsaOaepProfile::Sha256, [0x45; Sha256::OUTPUT_SIZE]);
  assert_profile_rejects_opaquely!(RsaOaepProfile::Sha384, [0x46; Sha384::OUTPUT_SIZE]);
  assert_profile_rejects_opaquely!(RsaOaepProfile::Sha512, [0x47; Sha512::OUTPUT_SIZE]);
}
