use super::*;
use crate::auth::rsa::tests::*;

#[test]
fn pkcs1v15_implicit_rejection_selects_the_message_or_the_alternative() {
  const K: usize = 32;
  let alternative = |len: usize| -> Vec<u8> {
    (K.strict_sub(len)..K)
      .map(|index| 0x80u8.wrapping_add(u8::try_from(index).expect("test block fits one octet")))
      .collect()
  };
  let padded = |message: &[u8], window: usize| -> Vec<u8> {
    let mut expected = message.to_vec();
    expected.resize(window, 0);
    expected
  };
  let window = K - RSAES_PKCS1V15_OVERHEAD;

  // EM = 00 02 PS 00 M with an eight-octet PS and a 13-octet M.
  let message = b"thirteen byte";
  let mut encoded = [0x5au8; K];
  encoded[0] = 0;
  encoded[1] = 2;
  encoded[K - message.len() - 1] = 0;
  encoded[K - message.len()..].copy_from_slice(message);
  for alternative_len in [0, 4, window] {
    assert_eq!(
      implicit_rejection_decode(&encoded, alternative_len),
      (message.len(), padded(message, window)),
      "a valid message must be returned whatever the alternative length"
    );
  }

  // Boundary messages: the longest (k - 11) and the empty message.
  let mut longest = [0x5au8; K];
  longest[..2].copy_from_slice(&[0, 2]);
  longest[10] = 0;
  assert_eq!(implicit_rejection_decode(&longest, 4), (window, longest[11..].to_vec()));
  let mut empty = [0x5au8; K];
  empty[..2].copy_from_slice(&[0, 2]);
  empty[K - 1] = 0;
  assert_eq!(implicit_rejection_decode(&empty, 4), (0, vec![0; window]));

  // Each padding rule fails on its own and selects the alternative's last octets.
  let mut first_byte = encoded;
  first_byte[0] = 1;
  let mut second_byte = encoded;
  second_byte[1] = 1;
  let mut missing_separator = encoded;
  missing_separator[2..].fill(0x5a);
  let mut short_padding = encoded;
  short_padding[9] = 0;
  let mut empty_padding = encoded;
  empty_padding[2] = 0;
  for (name, invalid) in [
    ("first octet", first_byte),
    ("block type", second_byte),
    ("missing separator", missing_separator),
    ("seven-octet PS", short_padding),
    ("empty PS", empty_padding),
  ] {
    for alternative_len in [0, 1, 7, window] {
      assert_eq!(
        implicit_rejection_decode(&invalid, alternative_len),
        (alternative_len, padded(&alternative(alternative_len), window)),
        "{name} with alternative length {alternative_len} must return the alternative message"
      );
    }
  }
}

#[test]
fn pkcs1v15_implicit_rejection_length_masks_and_keeps_the_last_fitting_candidate() {
  let candidates = |values: &[u16]| -> [u8; IMPLICIT_REJECTION_CANDIDATE_BYTES] {
    let mut bytes = [0xffu8; IMPLICIT_REJECTION_CANDIDATE_BYTES];
    for (pair, value) in bytes.as_chunks_mut::<2>().0.iter_mut().zip(values) {
      pair.copy_from_slice(&value.to_be_bytes());
    }
    bytes
  };
  // 245 = k - 11 for a 2048-bit modulus has bit length 8: 0x01f5 masks to 0xf5.
  assert_eq!(
    implicit_rejection_length(&candidates(&[]), 245),
    0,
    "0xffff masks to 255 > 245"
  );
  assert_eq!(implicit_rejection_length(&candidates(&[3]), 245), 3);
  assert_eq!(implicit_rejection_length(&candidates(&[3, 0x01f5]), 245), 245);
  assert_eq!(
    implicit_rejection_length(&candidates(&[3, 246]), 245),
    3,
    "246 does not fit"
  );
  assert_eq!(implicit_rejection_length(&candidates(&[0, 9, 0x0110]), 245), 16);
  // 500 has bit length 9: 0x03ff masks to 511, which does not fit; 0x03f4 masks to 500.
  assert_eq!(implicit_rejection_length(&candidates(&[0x03ff]), 500), 0);
  assert_eq!(implicit_rejection_length(&candidates(&[0x03f4]), 500), 500);
}

#[test]
fn pkcs1v15_implicit_rejection_depadding_matches_all_appendix_b_widths() {
  let vectors: serde_json::Value = serde_json::from_str(include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/rsa/implicit_rejection/appendix_b.json"
  )))
  .expect("Appendix B fixture must parse");
  let mut checked = 0;
  for key in vectors["keys"].as_array().expect("Appendix B keys") {
    let der = hex_to_vec(key["pkcs8"].as_str().expect("PKCS #8 hex"));
    // Read the stored exponent without importing a key. In particular, B.2's
    // 2049-bit key is outside the public half-width-factor import policy.
    let parts = parse_pkcs8_private_key_der_parts_with_policy(&der, &RsaPublicKeyPolicy::modulus_2048_to_8192())
      .expect("Appendix B DER structure must parse");
    for case in key["cases"].as_array().expect("Appendix B cases") {
      let ciphertext = hex_to_vec(case["ciphertext"].as_str().expect("ciphertext hex"));
      let encoded = hex_to_vec(case["encoded_message"].as_str().expect("OpenSSL raw RSA output"));
      let expected = hex_to_vec(case["message"].as_str().expect("draft message hex"));
      assert_eq!(encoded.len(), ciphertext.len());
      let exponent = SecretBigEndianInteger::new_fixed_width(parts.private_exponent, encoded.len())
        .expect("the stored exponent must fit the modulus width");
      let mut alternative = vec![0u8; encoded.len()];
      let mut out = vec![0xa5u8; encoded.len()];
      let len =
        decode_pkcs1v15_with_implicit_rejection(exponent.as_bytes(), &ciphertext, &encoded, &mut alternative, &mut out)
          .expect("all Appendix B widths must decode below the import boundary");
      assert_eq!(&out[..len], expected, "{}", case["section"]);
      let window = encoded.len() - RSAES_PKCS1V15_OVERHEAD;
      assert!(out[len..window].iter().all(|&byte| byte == 0));
      assert!(out[window..].iter().all(|&byte| byte == 0xa5));
      assert!(alternative.iter().all(|&byte| byte == 0));
      checked += 1;
    }
  }
  assert_eq!(checked, 48);
}

#[test]
fn pkcs1v15_encrypt_decrypt_hides_padding_failures_behind_implicit_rejection() {
  let key = prevalidated_test_private_key();
  let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
  let plaintext = b"rsaes-pkcs1v15 legacy roundtrip";
  let seed = vec![0xb6; key.signature_len().strict_sub(plaintext.len()).strict_sub(3)];

  let mut ciphertext = vec![0u8; key.signature_len()];
  key
    .public_key()
    .encrypt_pkcs1v15_with_random_fill(plaintext, &mut ciphertext, seed_fill(&seed))
    .expect("valid PKCS#1 v1.5 test plaintext and seed must encrypt");
  let mut out = vec![0u8; key.signature_len()];
  let decrypted_len = key
    .decrypt_pkcs1v15_with_blinding_factor(
      &ciphertext,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut out,
    )
    .expect("valid PKCS#1 v1.5 ciphertext must decrypt with explicit blinding");
  assert_eq!(&out[..decrypted_len], plaintext);

  let mut scratch = key.private_scratch();
  let decrypted_len = key
    .decrypt_pkcs1v15_with_blinding_factor_and_scratch(
      &ciphertext,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut out,
      &mut scratch,
    )
    .expect("valid PKCS#1 v1.5 ciphertext must decrypt with caller scratch");
  assert_eq!(&out[..decrypted_len], plaintext);

  #[cfg(feature = "getrandom")]
  {
    let decrypted_len = key
      .decrypt_pkcs1v15(&ciphertext, &mut out)
      .expect("valid PKCS#1 v1.5 ciphertext must decrypt with random blinding");
    assert_eq!(&out[..decrypted_len], plaintext);
    let decrypted_len = key
      .decrypt_pkcs1v15_with_scratch(&ciphertext, &mut out, &mut scratch)
      .expect("valid PKCS#1 v1.5 ciphertext must decrypt with random blinding and caller scratch");
    assert_eq!(&out[..decrypted_len], plaintext);
  }

  #[cfg(feature = "getrandom")]
  {
    let oversized_plaintext = vec![0x5a; key.signature_len().strict_sub(10)];
    ciphertext.fill(0xa5);
    assert_eq!(
      key.public_key().encrypt_pkcs1v15(&oversized_plaintext, &mut ciphertext),
      Err(RsaEncryptionError::MessageTooLong)
    );
    assert!(is_zero_unsigned_be(&ciphertext));
  }

  let mut bad_block = vec![0u8; key.signature_len()];
  bad_block[1] = 2;
  bad_block[2..10].fill(0xc5);
  bad_block[10] = 0;
  bad_block[11..11usize.strict_add(plaintext.len())].copy_from_slice(plaintext);
  key
    .public_key()
    .public_operation(&bad_block, &mut ciphertext)
    .expect("malformed encoded block must remain a valid RSA representative");
  // Every entry point returns the same synthetic message for a padding
  // failure: it depends only on the key and the ciphertext, not on blinding
  // or scratch. The values are checked against Appendix B and OpenSSL in the
  // integration tests.
  let window = key.signature_len().strict_sub(RSAES_PKCS1V15_OVERHEAD);
  let mut assert_implicit_rejection = |ciphertext: &[u8]| {
    out.fill(0xa5);
    let len = key
      .decrypt_pkcs1v15_with_blinding_factor(
        ciphertext,
        RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
        &mut out,
      )
      .expect("invalid padding must decrypt to the synthetic message");
    assert!(len <= window);
    assert!(out[len..window].iter().all(|&byte| byte == 0));
    assert!(out[window..].iter().all(|&byte| byte == 0xa5));
    assert_ne!(&out[..len], plaintext.as_slice());
    let expected = (len, out.clone());

    out.fill(0xa5);
    let len = key
      .decrypt_pkcs1v15_with_blinding_factor_and_scratch(
        ciphertext,
        RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
        &mut out,
        &mut scratch,
      )
      .expect("invalid padding must decrypt to the synthetic message with caller scratch");
    assert_eq!((len, out.clone()), expected);
    #[cfg(feature = "getrandom")]
    {
      out.fill(0xa5);
      let len = key
        .decrypt_pkcs1v15(ciphertext, &mut out)
        .expect("invalid padding must decrypt to the synthetic message with random blinding");
      assert_eq!((len, out.clone()), expected);
      out.fill(0xa5);
      let len = key
        .decrypt_pkcs1v15_with_scratch(ciphertext, &mut out, &mut scratch)
        .expect("invalid padding must decrypt to the synthetic message with random blinding and scratch");
      assert_eq!((len, out.clone()), expected);
    }
  };

  bad_block[1] = 1;
  key
    .public_key()
    .public_operation(&bad_block, &mut ciphertext)
    .expect("wrong block type must remain a valid RSA representative");
  assert_implicit_rejection(&ciphertext);

  bad_block[1] = 2;
  bad_block[5] = 0;
  key
    .public_key()
    .public_operation(&bad_block, &mut ciphertext)
    .expect("short padding block must remain a valid RSA representative");
  assert_implicit_rejection(&ciphertext);

  bad_block[5] = 0xc5;
  bad_block[10..].fill(0xc5);
  key
    .public_key()
    .public_operation(&bad_block, &mut ciphertext)
    .expect("missing separator block must remain a valid RSA representative");
  assert_implicit_rejection(&ciphertext);
}
