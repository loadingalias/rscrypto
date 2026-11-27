use super::*;
use crate::auth::rsa::tests::*;

#[cfg(feature = "getrandom")]
const CAVP_KEYGEN_186_3_PROBABLE_PRIME: &str = include_str!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/testdata/rsa/nist_cavp/rsa_keygen_186_3_probable_prime_subset.json"
));

#[test]
fn rsa_miri_proof_miller_rabin_covers_small_prime_and_composite_paths() {
  let prime = 1009u16.to_be_bytes();
  assert!(!has_small_prime_factor(&prime));
  assert!(private_import_is_probable_prime(&prime).expect("the small-prime RSA fixture must be accepted for testing"));

  let composite = [0x0f, 0x98, 0xa5];
  assert_eq!(
    u32::from_be_bytes([0, composite[0], composite[1], composite[2]]),
    1009 * 1013
  );
  assert!(!has_small_prime_factor(&composite));
  assert!(
    !private_import_is_probable_prime(&composite)
      .expect("the composite RSA fixture must be evaluated without an arithmetic failure")
  );
}

#[test]
fn pkcs1_private_key_parser_preserves_components_and_public_key() {
  let der = test_pkcs1_private_key();
  let key = parse_pkcs1_private_key_der_with_policy(&der, &RsaPublicKeyPolicy::modulus_2048_to_8192())
    .expect("the canonical PKCS#1 private-key fixture must parse");

  assert_eq!(prevalidated_test_private_key().to_pkcs1_der().as_bytes(), der);

  assert_eq!(key.public_key().modulus(), rsa_private_modulus());
  assert_eq!(key.public_key().public_exponent().as_u64(), 65_537);
  assert_eq!(key.private_exponent.as_bytes(), rsa_private_exponent());
  assert_eq!(key.prime_p.as_bytes(), rsa_private_prime_p());
  assert_eq!(key.prime_q.as_bytes(), rsa_private_prime_q());
  let exponent_p = rsa_private_exponent_p();
  let exponent_q = rsa_private_exponent_q();
  assert_eq!(key.exponent_p.as_bytes().len(), key.prime_p.as_bytes().len());
  assert_eq!(key.exponent_q.as_bytes().len(), key.prime_q.as_bytes().len());
  assert!(key.exponent_p.as_bytes().ends_with(&exponent_p));
  assert!(key.exponent_q.as_bytes().ends_with(&exponent_q));
  assert!(
    key.exponent_p.as_bytes()[..key.exponent_p.as_bytes().len().strict_sub(exponent_p.len())]
      .iter()
      .all(|&byte| byte == 0)
  );
  assert!(
    key.exponent_q.as_bytes()[..key.exponent_q.as_bytes().len().strict_sub(exponent_q.len())]
      .iter()
      .all(|&byte| byte == 0)
  );
  assert_eq!(key.coefficient.as_bytes(), rsa_private_coefficient());
}

#[test]
fn private_key_import_normalizes_crt_exponents_to_factor_width() {
  let prime_p = rsa::BoxedUint::from_be_slice_vartime(&rsa_private_prime_p());
  let prime_q = rsa::BoxedUint::from_be_slice_vartime(&rsa_private_prime_q());
  let public_exponent = 883;
  let oracle = rsa::PrivateKey::from_p_q(prime_p, prime_q, rsa::BoxedUint::from(public_exponent))
    .expect("fixture factors and exponent form an algebraically valid key");
  let modulus = rustcrypto_be(oracle.n());
  let private_exponent = rustcrypto_be(oracle.d());
  let prime_p = rustcrypto_be(&oracle.primes()[0]);
  let prime_q = rustcrypto_be(&oracle.primes()[1]);
  let exponent_p = rustcrypto_be(oracle.dp().expect("RustCrypto CRT exponent"));
  let exponent_q = rustcrypto_be(oracle.dq().expect("RustCrypto CRT exponent"));
  assert!(exponent_p.len() < prime_p.len() || exponent_q.len() < prime_q.len());
  let coefficient = rustcrypto_be(&oracle.crt_coefficient().expect("RustCrypto CRT coefficient"));

  let key = RsaPrivateKey::from_components_with_policy(
    RsaPrivateKeyParts {
      modulus: &modulus,
      public_exponent,
      private_exponent: &private_exponent,
      prime_p: &prime_p,
      prime_q: &prime_q,
      exponent_p: &exponent_p,
      exponent_q: &exponent_q,
      coefficient: &coefficient,
    },
    &RsaPublicKeyPolicy::modulus_2048_to_8192().allow_legacy_odd_exponents(),
  )
  .expect("algebraically valid components");

  assert_eq!(key.components.exponent_p.as_bytes().len(), prime_p.len());
  assert_eq!(key.components.exponent_q.as_bytes().len(), prime_q.len());
  assert!(key.components.exponent_p.as_bytes().ends_with(&exponent_p));
  assert!(key.components.exponent_q.as_bytes().ends_with(&exponent_q));
  assert!(
    key.components.exponent_p.as_bytes()[..prime_p.len().strict_sub(exponent_p.len())]
      .iter()
      .all(|&byte| byte == 0)
  );
  assert!(
    key.components.exponent_q.as_bytes()[..prime_q.len().strict_sub(exponent_q.len())]
      .iter()
      .all(|&byte| byte == 0)
  );
}

#[test]
fn private_key_import_normalizes_a_short_crt_coefficient_without_changing_der_or_arithmetic() {
  use rsa::{pkcs1::EncodeRsaPrivateKey as _, pkcs8::DecodePrivateKey as _};

  // Recombine published Wycheproof factors so q^-1 mod p occupies 127
  // octets, one fewer than p. RustCrypto supplies the independent key, and
  // num-bigint the private-operation expectation. No prime search runs in
  // this test.
  let signing: serde_json::Value = serde_json::from_str(include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/rsa/wycheproof/rsa_pkcs1_2048_sig_gen_test.json"
  )))
  .expect("Wycheproof signing fixture");
  let encryption: serde_json::Value = serde_json::from_str(include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/rsa/wycheproof/rsa_oaep_2048_sha384_mgf1sha1_test.json"
  )))
  .expect("Wycheproof encryption fixture");
  let signing_key = rsa::PrivateKey::from_pkcs8_der(&hex_to_vec(
    signing["testGroups"][0]["privateKeyPkcs8"]
      .as_str()
      .expect("signing key"),
  ))
  .expect("published signing key");
  let encryption_key = rsa::PrivateKey::from_pkcs8_der(&hex_to_vec(
    encryption["testGroups"][0]["privateKeyPkcs8"]
      .as_str()
      .expect("encryption key"),
  ))
  .expect("published encryption key");
  let oracle = rsa::PrivateKey::from_p_q(
    signing_key.primes()[1].clone().get(),
    encryption_key.primes()[0].clone().get(),
    rsa::BoxedUint::from(65_537u32),
  )
  .expect("published factors form a valid RSA key");
  let coefficient = rustcrypto_be(&oracle.crt_coefficient().expect("CRT coefficient"));
  assert_eq!(coefficient.len(), 127);
  let der = oracle.to_pkcs1_der().expect("independent canonical DER");
  let key = RsaPrivateKey::from_pkcs1_der_with_policy(der.as_bytes(), &RsaPublicKeyPolicy::modulus_2048_to_8192())
    .expect("valid recombined RSA-2048 key");
  let retained = key.components.coefficient.as_bytes();
  assert_eq!(retained.len(), 128);
  assert_eq!(retained[0], 0);
  assert_eq!(&retained[1..], coefficient);
  assert_eq!(key.components.private_exponent.as_bytes().len(), key.signature_len());
  assert_eq!(key.to_pkcs1_der().as_bytes(), der.as_bytes());

  let mut input = vec![0u8; key.signature_len()];
  *input.last_mut().expect("RSA input") = 2;
  let expected = BigUint::from(2u8)
    .modpow(
      &BigUint::from_bytes_be(&oracle.d().to_be_bytes()),
      &BigUint::from_bytes_be(&oracle.n().to_be_bytes()),
    )
    .to_bytes_be();
  let mut padded_expected = vec![0u8; key.signature_len().strict_sub(expected.len())];
  padded_expected.extend_from_slice(&expected);
  let (factor, inverse) = factor_two_and_inverse(key.public_key().modulus());
  let blinding = RsaBlindingPair::new(&factor, &inverse);
  let mut out = vec![0u8; key.signature_len()];
  key
    .components
    .private_operation_with_blinding_factor(&input, blinding, &mut out)
    .expect("private operation");
  assert_eq!(out, padded_expected);
  let mut scratch = key.private_scratch();
  scratch.encoded.as_mut_slice().copy_from_slice(&input);
  key
    .components
    .private_operation_with_blinding_factor_and_scratch(blinding, &mut out, &mut scratch)
    .expect("private operation with scratch");
  assert_eq!(out, padded_expected);
}

#[test]
fn private_key_import_rejects_balanced_composite_factor_in_every_format() {
  let one = BigUint::from(1u8);
  let prime_q = rsa::BoxedUint::from_be_slice_vartime(&rsa_private_prime_q());
  let (oracle, composite_p) = (1u64..4096)
    .step_by(2)
    .find_map(|offset| {
      let left = (&one << 512usize) + BigUint::from(offset);
      let right = (&one << 512usize) - BigUint::from(offset);
      let composite_p = left * right;
      let composite_bytes = composite_p.to_bytes_be();
      if unsigned_be_bit_len(&composite_bytes) != 1024 || has_small_prime_factor(&composite_bytes) {
        return None;
      }
      rsa::PrivateKey::from_p_q(
        rsa::BoxedUint::from_be_slice_vartime(&composite_bytes),
        prime_q.clone(),
        rsa::BoxedUint::from(65_537u32),
      )
      .ok()
      .map(|oracle| (oracle, composite_bytes))
    })
    .expect("construct a balanced composite without a small prime divisor");
  let modulus = rustcrypto_be(oracle.n());
  let private_exponent = rustcrypto_be(oracle.d());
  let prime_p = rustcrypto_be(&oracle.primes()[0]);
  let prime_q = rustcrypto_be(&oracle.primes()[1]);
  let exponent_p = rustcrypto_be(oracle.dp().expect("RustCrypto CRT exponent"));
  let exponent_q = rustcrypto_be(oracle.dq().expect("RustCrypto CRT exponent"));
  let coefficient = rustcrypto_be(&oracle.crt_coefficient().expect("RustCrypto CRT coefficient"));
  assert_eq!(unsigned_be_bit_len(&modulus), 2048);
  assert_eq!(unsigned_be_bit_len(&prime_p), 1024);
  assert_eq!(unsigned_be_bit_len(&prime_q), 1024);
  assert_eq!(prime_p, composite_p);
  assert!(!has_small_prime_factor(&prime_p));
  assert!(
    !private_import_is_probable_prime(&prime_p)
      .expect("the balanced composite fixture must be evaluated without an arithmetic failure")
  );

  assert_eq!(
    RsaPrivateKey::from_components_with_policy(
      RsaPrivateKeyParts {
        modulus: &modulus,
        public_exponent: 65_537,
        private_exponent: &private_exponent,
        prime_p: &prime_p,
        prime_q: &prime_q,
        exponent_p: &exponent_p,
        exponent_q: &exponent_q,
        coefficient: &coefficient,
      },
      &RsaPublicKeyPolicy::modulus_2048_to_8192(),
    )
    .err(),
    Some(RsaKeyError::InvalidModulus)
  );

  let pkcs1 = rsa::pkcs1::EncodeRsaPrivateKey::to_pkcs1_der(&oracle).expect("RustCrypto PKCS #1 encoding");
  assert_eq!(
    RsaPrivateKey::from_pkcs1_der_with_policy(pkcs1.as_bytes(), &RsaPublicKeyPolicy::modulus_2048_to_8192()).err(),
    Some(RsaKeyError::InvalidModulus)
  );
  let pkcs8 = rsa::pkcs8::EncodePrivateKey::to_pkcs8_der(&oracle).expect("RustCrypto PKCS #8 encoding");
  assert_eq!(
    RsaPrivateKey::from_pkcs8_der_with_policy(pkcs8.as_bytes(), &RsaPublicKeyPolicy::modulus_2048_to_8192()).err(),
    Some(RsaKeyError::InvalidModulus)
  );
}

#[test]
fn private_key_import_rejects_grossly_unbalanced_factors() {
  let one = BigUint::from(1u8);
  let oversized_composite = (&one << 3008usize) - &one;
  let small_prime = BigUint::from(18_446_744_073_709_551_557u64);
  let public_exponent = BigUint::from(65_537u32);
  let private_exponent = public_exponent
    .modinv(&((&oversized_composite - &one) * (&small_prime - &one)))
    .expect("the public exponent is coprime to the totient");
  let boxed = |value: &BigUint| rsa::BoxedUint::from_be_slice_vartime(&value.to_bytes_be());
  // rsa 0.10 limits each `from_p_q` prime to ceil(bits(n)/2) bits, so this key needs the wide-primes constructor.
  let oracle = rsa::PrivateKey::from_components_with_wide_primes(
    boxed(&(&oversized_composite * &small_prime)),
    boxed(&public_exponent),
    boxed(&private_exponent),
    vec![boxed(&oversized_composite), boxed(&small_prime)],
  )
  .expect("algebraically valid two-factor key");
  let modulus = rustcrypto_be(oracle.n());
  let private_exponent = rustcrypto_be(oracle.d());
  let prime_p = rustcrypto_be(&oracle.primes()[0]);
  let prime_q = rustcrypto_be(&oracle.primes()[1]);
  let exponent_p = rustcrypto_be(oracle.dp().expect("RustCrypto CRT exponent"));
  let exponent_q = rustcrypto_be(oracle.dq().expect("RustCrypto CRT exponent"));
  let coefficient = rustcrypto_be(&oracle.crt_coefficient().expect("RustCrypto CRT coefficient"));
  assert_eq!(unsigned_be_bit_len(&modulus), 3072);
  assert_eq!(unsigned_be_bit_len(&prime_p), 3008);
  assert_eq!(unsigned_be_bit_len(&prime_q), 64);

  assert_eq!(
    RsaPrivateKey::from_components_with_policy(
      RsaPrivateKeyParts {
        modulus: &modulus,
        public_exponent: 65_537,
        private_exponent: &private_exponent,
        prime_p: &prime_p,
        prime_q: &prime_q,
        exponent_p: &exponent_p,
        exponent_q: &exponent_q,
        coefficient: &coefficient,
      },
      &RsaPublicKeyPolicy::modulus_3072_to_8192(),
    )
    .err(),
    Some(RsaKeyError::InvalidModulus)
  );
}

#[test]
fn private_key_components_debug_redacts_secret_material() {
  let key = prevalidated_test_private_key();
  let debug = format!("{:?}", key.components);
  let (blinding_factor, blinding_inverse) = factor_two_and_inverse(key.public_key().modulus());
  let blinding_debug = format!("{:?}", RsaBlindingPair::new(&blinding_factor, &blinding_inverse));
  let parts_debug = format!(
    "{:?}",
    rsa_private_key_parts(
      &rsa_private_modulus(),
      &rsa_private_exponent(),
      &rsa_private_prime_p(),
      &rsa_private_prime_q(),
      &rsa_private_exponent_p(),
      &rsa_private_exponent_q(),
      &rsa_private_coefficient(),
    )
  );

  assert!(debug.contains("modulus_bits"));
  assert!(debug.contains("public_exponent"));
  assert!(debug.contains("private_exponent: \"****\""));
  assert!(debug.contains("prime_p: \"****\""));
  assert!(debug.contains("prime_q: \"****\""));
  assert!(debug.contains("coefficient: \"****\""));
  assert!(parts_debug.contains("modulus_bits"));
  assert!(parts_debug.contains("public_exponent: 65537"));
  assert!(parts_debug.contains("private_exponent: \"****\""));
  assert!(parts_debug.contains("prime_p: \"****\""));
  assert!(parts_debug.contains("prime_q: \"****\""));
  assert!(parts_debug.contains("coefficient: \"****\""));
  assert_eq!(
    blinding_debug,
    "RsaBlindingPair { factor: \"****\", inverse: \"****\" }"
  );
  assert!(!blinding_debug.contains(&format!("{blinding_factor:?}")));
  assert!(!blinding_debug.contains(&format!("{blinding_inverse:?}")));
  for secret in [
    rsa_private_exponent(),
    rsa_private_prime_p(),
    rsa_private_prime_q(),
    rsa_private_exponent_p(),
    rsa_private_exponent_q(),
    rsa_private_coefficient(),
  ] {
    assert!(!debug.contains(&format!("{secret:?}")));
    assert!(!parts_debug.contains(&format!("{secret:?}")));
  }

  assert_eq!(
    format!("{key:?}"),
    format!(
      "RsaPrivateKey {{ public: {:?}, private_components: \"****\", .. }}",
      key.public_key()
    )
  );
  assert_eq!(
    format!("{:?}", key.private_scratch()),
    format!("RsaPrivateScratch {{ bytes: {}, .. }}", key.signature_len())
  );

  // The signing workspace shows only its public width and use count, also
  // while it holds a blinding pair.
  #[cfg(feature = "getrandom")]
  {
    let mut workspace = key.signing_workspace();
    assert_eq!(
      format!("{workspace:?}"),
      format!(
        "RsaSigningWorkspace {{ bytes: {}, remaining: 0, .. }}",
        key.signature_len()
      )
    );
    let mut signature = vec![0u8; key.signature_len()];
    key
      .sign_signature_with_workspace(
        RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256),
        b"rscrypto signing workspace Debug",
        &mut signature,
        &mut workspace,
      )
      .expect("signing through a new workspace must succeed");
    assert!(!is_zero_unsigned_be(workspace.blinding_power.as_slice()));
    assert!(!is_zero_unsigned_be(workspace.blinding_inverse.as_slice()));
    assert_eq!(
      format!("{workspace:?}"),
      format!(
        "RsaSigningWorkspace {{ bytes: {}, remaining: 31, .. }}",
        key.signature_len()
      )
    );
  }
}

#[test]
fn key_der_exports_roundtrip_through_strict_importers() {
  let pkcs1 = test_pkcs1_private_key();
  let key = RsaPrivateKey::from_pkcs1_der_with_policy(&pkcs1, &RsaPublicKeyPolicy::modulus_2048_to_8192())
    .expect("PKCS#1 fixture must import");
  let rsa_algorithm = algorithm_identifier(RSA_ENCRYPTION_OID, Some(&null()));

  assert_eq!(
    key.public_key().to_pkcs1_der(),
    test_pkcs1_public_key(&rsa_private_modulus(), &[0x01, 0x00, 0x01])
  );
  assert_eq!(
    key.to_pkcs8_der().as_bytes(),
    test_pkcs8_private_key(&pkcs1, &rsa_algorithm)
  );

  let exported_pkcs1 = key.to_pkcs1_der();
  let imported_pkcs1 =
    RsaPrivateKey::from_pkcs1_der_with_policy(&exported_pkcs1, &RsaPublicKeyPolicy::modulus_2048_to_8192())
      .expect("exported PKCS#1 private key must re-import");
  assert_eq!(imported_pkcs1.public_key(), key.public_key());

  let exported_pkcs8 = key.to_pkcs8_der();
  let imported_pkcs8 =
    RsaPrivateKey::from_pkcs8_der_with_policy(&exported_pkcs8, &RsaPublicKeyPolicy::modulus_2048_to_8192())
      .expect("exported PKCS#8 private key must re-import");
  assert_eq!(imported_pkcs8.public_key(), key.public_key());

  let public_pkcs1 = key.public_key().to_pkcs1_der();
  let imported_public_pkcs1 =
    RsaPublicKey::from_pkcs1_der_with_policy(&public_pkcs1, &RsaPublicKeyPolicy::modulus_2048_to_8192())
      .expect("exported PKCS#1 public key must re-import");
  assert_eq!(imported_public_pkcs1, *key.public_key());

  let spki = key.public_key().to_spki_der();
  let imported_spki = RsaPublicKey::from_spki_der_with_policy(&spki, &RsaPublicKeyPolicy::modulus_2048_to_8192())
    .expect("exported SPKI public key must re-import");
  assert_eq!(imported_spki, *key.public_key());
}

#[test]
fn raw_public_and_private_components_import_end_to_end() {
  let modulus = rsa_private_modulus();
  let private_exponent = rsa_private_exponent();
  let prime_p = rsa_private_prime_p();
  let prime_q = rsa_private_prime_q();
  let exponent_p = rsa_private_exponent_p();
  let exponent_q = rsa_private_exponent_q();
  let coefficient = rsa_private_coefficient();
  let policy = RsaPublicKeyPolicy::modulus_2048_to_8192();

  let public = RsaPublicKey::from_modulus_exponent_with_policy(&modulus, 65_537, &policy)
    .expect("valid raw public components must import");
  assert_eq!(public.modulus(), modulus);
  assert_eq!(public.public_exponent().as_u64(), 65_537);

  let key = RsaPrivateKey::from_components_with_policy(
    rsa_private_key_parts(
      &modulus,
      &private_exponent,
      &prime_p,
      &prime_q,
      &exponent_p,
      &exponent_q,
      &coefficient,
    ),
    &policy,
  )
  .expect("valid raw private components must import");
  assert_eq!(key.public_key(), &public);

  let message = b"rscrypto raw RSA component import signing roundtrip";
  let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
  let mut signature = vec![0u8; key.signature_len()];
  key
    .sign_pkcs1v15_with_blinding_factor(
      RsaPkcs1v15Profile::Sha256,
      message,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut signature,
    )
    .expect("imported raw private components must sign");
  public
    .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &signature)
    .expect("signature from imported raw components must verify");

  let mut bad_coefficient = coefficient;
  *bad_coefficient
    .last_mut()
    .expect("RSA coefficient fixture must be nonempty") ^= 1;
  assert_eq!(
    RsaPrivateKey::from_components_with_policy(
      rsa_private_key_parts(
        &modulus,
        &private_exponent,
        &prime_p,
        &prime_q,
        &exponent_p,
        &exponent_q,
        &bad_coefficient,
      ),
      &policy,
    )
    .err(),
    Some(RsaKeyError::InvalidModulus)
  );
  assert_eq!(
    RsaPublicKey::from_modulus_exponent_with_policy(&modulus, 17, &policy).err(),
    Some(RsaKeyError::InvalidPublicExponent)
  );
}

#[test]
fn raw_private_components_reject_noncanonical_leading_zeroes() {
  let modulus = rsa_private_modulus();
  let private_exponent = rsa_private_exponent();
  let prime_p = rsa_private_prime_p();
  let prime_q = rsa_private_prime_q();
  let exponent_p = rsa_private_exponent_p();
  let exponent_q = rsa_private_exponent_q();
  let coefficient = rsa_private_coefficient();
  let policy = RsaPublicKeyPolicy::modulus_2048_to_8192();

  let mut bad_private_exponent = private_exponent.clone();
  bad_private_exponent.insert(0, 0);
  assert_eq!(
    RsaPrivateKey::from_components_with_policy(
      rsa_private_key_parts(
        &modulus,
        &bad_private_exponent,
        &prime_p,
        &prime_q,
        &exponent_p,
        &exponent_q,
        &coefficient,
      ),
      &policy,
    )
    .err(),
    Some(RsaKeyError::InvalidModulus)
  );

  let mut bad_prime_p = prime_p.clone();
  bad_prime_p.insert(0, 0);
  assert_eq!(
    RsaPrivateKey::from_components_with_policy(
      rsa_private_key_parts(
        &modulus,
        &private_exponent,
        &bad_prime_p,
        &prime_q,
        &exponent_p,
        &exponent_q,
        &coefficient,
      ),
      &policy,
    )
    .err(),
    Some(RsaKeyError::InvalidModulus)
  );

  let mut bad_prime_q = prime_q.clone();
  bad_prime_q.insert(0, 0);
  assert_eq!(
    RsaPrivateKey::from_components_with_policy(
      rsa_private_key_parts(
        &modulus,
        &private_exponent,
        &prime_p,
        &bad_prime_q,
        &exponent_p,
        &exponent_q,
        &coefficient,
      ),
      &policy,
    )
    .err(),
    Some(RsaKeyError::InvalidModulus)
  );

  let mut bad_exponent_p = exponent_p.clone();
  bad_exponent_p.insert(0, 0);
  assert_eq!(
    RsaPrivateKey::from_components_with_policy(
      rsa_private_key_parts(
        &modulus,
        &private_exponent,
        &prime_p,
        &prime_q,
        &bad_exponent_p,
        &exponent_q,
        &coefficient,
      ),
      &policy,
    )
    .err(),
    Some(RsaKeyError::InvalidModulus)
  );

  let mut bad_exponent_q = exponent_q.clone();
  bad_exponent_q.insert(0, 0);
  assert_eq!(
    RsaPrivateKey::from_components_with_policy(
      rsa_private_key_parts(
        &modulus,
        &private_exponent,
        &prime_p,
        &prime_q,
        &exponent_p,
        &bad_exponent_q,
        &coefficient,
      ),
      &policy,
    )
    .err(),
    Some(RsaKeyError::InvalidModulus)
  );

  let mut bad_coefficient = coefficient;
  bad_coefficient.insert(0, 0);
  assert_eq!(
    RsaPrivateKey::from_components_with_policy(
      rsa_private_key_parts(
        &modulus,
        &private_exponent,
        &prime_p,
        &prime_q,
        &exponent_p,
        &exponent_q,
        &bad_coefficient,
      ),
      &policy,
    )
    .err(),
    Some(RsaKeyError::InvalidModulus)
  );
}

#[cfg(feature = "getrandom")]
#[test]
fn keygen_contract_is_explicitly_fips_186_5_a1_3_probable_prime() {
  assert_eq!(
    RsaPrivateKey::GENERATION_CONTRACT,
    RsaKeyGenerationContract::Fips1865A13ProbablePrime
  );
}

#[cfg(feature = "getrandom")]
#[test]
fn keygen_random_candidate_has_fips_a1_3_raw_shape() {
  let mut drbg = test_keygen_drbg(b"candidate-shape");
  for bits in [2usize, 3, 9, 128, 1024] {
    let candidate = keygen_random_odd_candidate(&mut drbg, bits);
    assert_eq!(candidate.len(), bits.strict_add(7) / 8);
    assert_eq!(candidate.last().copied().unwrap_or_default() & 1, 1);
    assert!(unsigned_be_bit_len(&candidate) <= bits);
  }
}

#[cfg(feature = "getrandom")]
#[test]
fn keygen_fips_a1_3_lower_bound_matches_square_test_at_boundary() {
  let below = [0xb5, 0x04, 0xf3, 0x33, 0xf9, 0xde, 0x64, 0x83];
  let at_or_above = [0xb5, 0x04, 0xf3, 0x33, 0xf9, 0xde, 0x64, 0x84];
  let above = [0xb5, 0x04, 0xf3, 0x33, 0xf9, 0xde, 0x64, 0x85];

  assert!(!keygen_probable_prime_meets_fips_lower_bound(&below, 64));
  assert_eq!(
    keygen_probable_prime_meets_fips_lower_bound(&at_or_above, 64),
    private_import_product_unsigned_be(&at_or_above, &at_or_above)
      .as_ref()
      .is_some_and(|square| unsigned_be_bit_len(square.as_slice()) >= 128)
  );
  assert!(keygen_probable_prime_meets_fips_lower_bound(&above, 64));
}

#[cfg(feature = "getrandom")]
#[test]
fn keygen_generated_prime_satisfies_fips_a1_3_prime_constraints() {
  let mut drbg = test_keygen_drbg(b"generated-prime-constraints");
  let prime = keygen_generate_prime(&mut drbg, 128, 256, None, 256usize.strict_mul(5))
    .expect("deterministic prime search must complete");

  assert!(keygen_probable_prime_meets_fips_lower_bound(&prime, 128));
  assert_eq!(prime.last().copied().unwrap_or_default() & 1, 1);
  assert!(!has_small_prime_factor(&prime));
  assert!(!keygen_conflicts_with_public_exponent(&prime));
  assert!(keygen_is_probable_prime(&mut drbg, &prime).expect("generated prime must pass a valid primality test"));
}

#[cfg(feature = "getrandom")]
#[test]
fn keygen_private_exponent_matches_euclid_and_bigint_inverse() {
  // Oracle: Euclid's algorithm for the LCM and num-bigint for the modular inverse.
  fn gcd(mut a: BigUint, mut b: BigUint) -> BigUint {
    while b != BigUint::ZERO {
      let remainder = &a % &b;
      a = core::mem::replace(&mut b, remainder);
    }
    a
  }
  let pad = |value: &BigUint| {
    let bytes = value.to_bytes_be();
    let mut padded = vec![0u8; 64usize.strict_sub(bytes.len())];
    padded.extend_from_slice(&bytes);
    padded
  };
  let e = BigUint::from(RSA_KEYGEN_PUBLIC_EXPONENT);
  let one = BigUint::from(1u8);
  let minimum = &one << 512usize;
  let mut cases = vec![
    (BigUint::from(6u8), BigUint::from(4u8)),
    (&e * 2u8, BigUint::from(4u8)),
    (&one << 511usize, &one << 511usize),
    ((&one << 511usize) - 2u8, (&one << 511usize) - 2u8),
    ((&one << 400usize) * 3u8, (&one << 300usize) * 5u8),
  ];
  let mut drbg = test_keygen_drbg(b"private-exponent-oracle");
  for _ in 0..48 {
    let mut bytes = [0u8; 24 + 32 + 32 + 1];
    drbg.fill(&mut bytes);
    let common = BigUint::from_bytes_be(&bytes[..24]) << (usize::from(bytes[88]) % 64);
    let a = (&common * BigUint::from_bytes_be(&bytes[24..56])).max(one.clone());
    let b = (&common * BigUint::from_bytes_be(&bytes[56..88])).max(one.clone());
    cases.push((a, b));
  }

  let value = |limbs: &SecretLimbs| {
    BigUint::from_bytes_le(
      &limbs
        .as_slice()
        .iter()
        .flat_map(|limb| limb.to_le_bytes())
        .collect::<Vec<_>>(),
    )
  };
  for (a, b) in cases {
    let lambda = &a * &b / gcd(a.clone(), b.clone());
    let mut actual = keygen_lcm(&limbs_from_be(&pad(&a)), &limbs_from_be(&pad(&b)));
    assert_eq!(value(&actual), lambda, "a = {a:x}, b = {b:x}");
    let inverse = e.modinv(&lambda).unwrap_or_default();
    keygen_invert_public_exponent(actual.as_mut_slice());
    assert_eq!(value(&actual), inverse, "a = {a:x}, b = {b:x}");
    let exported = keygen_private_exponent(&pad(&a), &pad(&b), 1024).map(|d| BigUint::from_bytes_be(d.as_slice()));
    assert_eq!(exported, Some(inverse).filter(|d| *d > minimum));
  }
}

#[cfg(feature = "getrandom")]
#[test]
fn keygen_candidate_shape_helper_tracks_fips_search_acceptance_shape() {
  let mut drbg = test_keygen_drbg(b"accepted-candidate-shape");
  let mut accepted = None;
  while accepted.is_none() {
    let candidate = keygen_random_odd_candidate(&mut drbg, 128);
    if keygen_probable_prime_meets_fips_lower_bound(&candidate, 128) {
      accepted = Some(candidate);
    }
  }
  let candidate = accepted.expect("deterministic search must find a FIPS-shaped candidate");
  assert_eq!(
    unsigned_be_bit_len(&candidate),
    128,
    "accepted FIPS A.1.3 candidate must be full-width"
  );
  assert_eq!(candidate.last().copied().unwrap_or_default() & 1, 1);
}

#[cfg(feature = "getrandom")]
#[test]
fn keygen_prefilter_rejects_small_prime_factors_without_rejecting_larger_prime() {
  assert!(has_small_prime_factor(&999u16.to_be_bytes()));
  assert!(!has_small_prime_factor(&1009u16.to_be_bytes()));
}

#[cfg(feature = "getrandom")]
#[test]
fn keygen_public_exponent_conflict_detects_p_minus_one_not_coprime_to_e() {
  let conflicting = RSA_KEYGEN_PUBLIC_EXPONENT.strict_add(1).to_be_bytes();
  assert!(keygen_conflicts_with_public_exponent(&conflicting));
  assert!(!keygen_conflicts_with_public_exponent(
    &RSA_KEYGEN_PUBLIC_EXPONENT.to_be_bytes()
  ));
}

#[cfg(feature = "getrandom")]
#[test]
fn keygen_miller_rabin_uses_fixed_round_count_above_fips_table_minimums() {
  assert_eq!(keygen_miller_rabin_rounds(1024), 32);
  assert_eq!(keygen_miller_rabin_rounds(1536), 32);
  assert_eq!(keygen_miller_rabin_rounds(2048), 32);

  assert!(keygen_miller_rabin_rounds(1024) >= 5);
  assert!(keygen_miller_rabin_rounds(1536) >= 4);
  assert!(keygen_miller_rabin_rounds(2048) >= 4);
}

#[cfg(feature = "getrandom")]
#[test]
fn keygen_miller_rabin_accepts_prime_and_rejects_composite_for_fixed_bases() {
  let prime = 1009u16.to_be_bytes();
  let prime_modulus = private_component_modulus(&prime).expect("nonzero prime fixture must define a modulus");
  assert!(
    keygen_miller_rabin_accepts_base(&prime_modulus, &[0, 63], 4, &1008u16.to_be_bytes(), &[0, 11])
      .expect("valid prime fixture must admit a Miller-Rabin round"),
    "1009 must pass a direct Miller-Rabin round for base 11"
  );

  let composite = 341u16.to_be_bytes();
  let composite_modulus =
    private_component_modulus(&composite).expect("nonzero composite fixture must define a modulus");
  assert!(
    !keygen_miller_rabin_accepts_base(&composite_modulus, &[0, 85], 2, &340u16.to_be_bytes(), &[0, 2])
      .expect("valid composite fixture must admit a Miller-Rabin round"),
    "341 must fail a direct Miller-Rabin round for base 2"
  );
}

#[cfg(feature = "getrandom")]
#[test]
fn keygen_derives_private_components_from_fixture_primes_end_to_end() {
  let modulus = rsa_private_modulus();
  let components = keygen_build_private_key_from_primes(
    unsigned_be_bit_len(&modulus),
    &RsaPublicKeyPolicy::modulus_2048_to_8192(),
    rsa_private_prime_p(),
    rsa_private_prime_q(),
  )
  .expect("fixture primes must derive without arithmetic failure")
  .expect("fixture primes must produce the requested modulus width");
  let key = RsaPrivateKey { components };
  assert_eq!(key.public_key().modulus(), modulus);
  assert_eq!(key.public_key().public_exponent().as_u64(), RSA_KEYGEN_PUBLIC_EXPONENT);

  let exported = key.to_pkcs1_der();
  let imported = RsaPrivateKey::from_pkcs1_der_with_policy(&exported, &RsaPublicKeyPolicy::modulus_2048_to_8192())
    .expect("generated PKCS#1 private key must re-import");
  assert_eq!(imported.public_key(), key.public_key());

  let message = b"rscrypto generated RSA component signing roundtrip";
  let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
  let mut signature = vec![0u8; key.signature_len()];
  key
    .sign_pkcs1v15_with_blinding_factor(
      RsaPkcs1v15Profile::Sha256,
      message,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut signature,
    )
    .expect("key derived from fixture primes must sign");
  key
    .public_key()
    .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &signature)
    .expect("signature from fixture-prime key must verify");

  let label = b"rscrypto-keygen-oaep";
  let plaintext = b"generated component oaep roundtrip";
  let seed = [0x7b; Sha256::OUTPUT_SIZE];
  let mut ciphertext = vec![0u8; key.signature_len()];
  key
    .public_key()
    .encrypt_oaep_with_random_fill(
      RsaOaepProfile::Sha256,
      label,
      plaintext,
      &mut ciphertext,
      seed_fill(&seed),
    )
    .expect("fixture-prime OAEP inputs must encrypt");
  let mut decrypted = vec![0u8; key.signature_len()];
  let decrypted_len = key
    .decrypt_oaep_with_blinding_factor(
      RsaOaepProfile::Sha256,
      label,
      &ciphertext,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut decrypted,
    )
    .expect("fixture-prime OAEP ciphertext must decrypt");
  assert_eq!(&decrypted[..decrypted_len], plaintext);

  let pkcs1v15_plaintext = b"generated component RSAES-PKCS1-v1_5 roundtrip";
  let pkcs1v15_seed = vec![0x3c; key.signature_len().strict_sub(pkcs1v15_plaintext.len()).strict_sub(3)];
  key
    .public_key()
    .encrypt_pkcs1v15_with_random_fill(pkcs1v15_plaintext, &mut ciphertext, seed_fill(&pkcs1v15_seed))
    .expect("fixture-prime PKCS#1 v1.5 inputs must encrypt");
  let decrypted_len = key
    .decrypt_pkcs1v15_with_blinding_factor(
      &ciphertext,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut decrypted,
    )
    .expect("fixture-prime PKCS#1 v1.5 ciphertext must decrypt");
  assert_eq!(&decrypted[..decrypted_len], pkcs1v15_plaintext);
}

/// Probable primes from a key generation run whose `d` is below `2^(2048 - 8)`, so its
/// modulus-width encoding starts with a zero byte. About 1 key in 20 has such a `d`.
#[cfg(feature = "getrandom")]
#[test]
fn keygen_accepts_a_private_exponent_shorter_than_the_modulus() {
  let prime_p = hex_to_vec(
    &[
      "f3a3c18ce6a9a56d1e388c01d5d2b30a91906629bab8df0f3661560602bdda8463eafbe9116d60d0f6075ee0562b14a0",
      "3ab01cab931eec9947f0289793bd765db08afa99fcbf920debf67c51afdd1c07c1167772d9d67f17e0e5fc94e9e7b5ba",
      "e6dc01dba9bd649ddf2c8066f67f3db42ca88fc80a3ad544ccfcee563fe61099",
    ]
    .concat(),
  );
  let prime_q = hex_to_vec(
    &[
      "fe2f31be4b89c9a7eddcc93f9291687352fb06c640e09d473144306ec53c018bcbcea38eb0cc5bd3084a9e636bc66fb1",
      "319fdb64b281988adef848107020408a1cdf9f18fb10ef5030bd066a9693dcb61eff6686aa11119d54e5f2d1a90623b0",
      "ae2d9c19bb2e2ba05f59dee57367a6d4bebf10717cc00a29d0f2d3bb0199f161",
    ]
    .concat(),
  );
  let oracle = rsa::PrivateKey::from_p_q(
    rsa::BoxedUint::from_be_slice_vartime(&prime_p),
    rsa::BoxedUint::from_be_slice_vartime(&prime_q),
    rsa::BoxedUint::from(RSA_KEYGEN_PUBLIC_EXPONENT),
  )
  .expect("fixture primes form a valid key");
  let private_exponent = rustcrypto_be(oracle.d());
  assert!(
    private_exponent.len() < 2048 / 8,
    "the fixture needs a short private exponent"
  );

  let policy = RsaPublicKeyPolicy::modulus_2048_to_8192();
  let components = keygen_build_private_key_from_primes(2048, &policy, prime_p, prime_q)
    .expect("a short private exponent is valid")
    .expect("fixture primes produce a 2048-bit modulus");
  let key = RsaPrivateKey { components };
  assert_eq!(key.components.private_exponent.as_bytes()[0], 0);
  assert_eq!(
    BigUint::from_bytes_be(key.components.private_exponent.as_bytes()),
    BigUint::from_bytes_be(&private_exponent)
  );
  assert_eq!(key.public_key().modulus(), rustcrypto_be(oracle.n()));

  let imported = RsaPrivateKey::from_pkcs1_der_with_policy(&key.to_pkcs1_der(), &policy)
    .expect("the generated PKCS#1 private key re-imports");
  assert_eq!(imported.public_key(), key.public_key());
  let message = b"rscrypto short private exponent";
  let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
  let mut signature = vec![0u8; key.signature_len()];
  imported
    .sign_pkcs1v15_with_blinding_factor(
      RsaPkcs1v15Profile::Sha256,
      message,
      RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
      &mut signature,
    )
    .expect("the re-imported key signs");
  key
    .public_key()
    .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &signature)
    .expect("the signature verifies");
}

#[cfg(feature = "getrandom")]
#[test]
fn keygen_derives_private_components_from_nist_cavp_probable_primes() {
  let suite: Value = serde_json::from_str(CAVP_KEYGEN_186_3_PROBABLE_PRIME).expect("CAVP keygen JSON must parse");
  assert_eq!(
    suite["source_files"][0].as_str(),
    Some("KeyGen_186-3_RandomProbablyPrime3_3_KAT.txt")
  );
  assert_eq!(suite["counts"]["total"].as_u64(), Some(2));

  let tests = suite["tests"].as_array().expect("CAVP keygen tests must be an array");
  let policy = RsaPublicKeyPolicy::modulus_2048_to_8192();
  let mut covered = Vec::new();

  for test in tests {
    let modulus_bits = usize::try_from(test["mod"].as_u64().expect("CAVP modulus size must be numeric"))
      .expect("CAVP modulus size must fit usize");
    let components = keygen_build_private_key_from_primes(
      modulus_bits,
      &policy,
      hex_to_vec(json_field(test, "p")),
      hex_to_vec(json_field(test, "q")),
    )
    .expect("CAVP probable-prime candidates must derive cleanly")
    .expect("CAVP probable-prime candidates must produce the requested modulus width");
    let key = RsaPrivateKey { components };
    assert_eq!(key.public_key().modulus_bits(), modulus_bits);
    assert_eq!(key.public_key().public_exponent().as_u64(), RSA_KEYGEN_PUBLIC_EXPONENT);
    let mut primality_drbg = test_keygen_drbg(b"cavp-primality");
    assert!(
      keygen_is_probable_prime(&mut primality_drbg, key.components.prime_p.as_bytes())
        .expect("CAVP prime p must admit a primality test")
    );
    assert!(
      keygen_is_probable_prime(&mut primality_drbg, key.components.prime_q.as_bytes())
        .expect("CAVP prime q must admit a primality test")
    );

    let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
    let message = b"rscrypto NIST CAVP keygen candidate private operation";

    let mut pkcs1v15_signature = vec![0u8; key.signature_len()];
    key
      .sign_pkcs1v15_with_blinding_factor(
        RsaPkcs1v15Profile::Sha256,
        message,
        RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
        &mut pkcs1v15_signature,
      )
      .expect("CAVP-derived key must sign PKCS#1 v1.5");
    key
      .public_key()
      .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &pkcs1v15_signature)
      .expect("CAVP-derived PKCS#1 v1.5 signature must verify");

    let salt = [0xa5; Sha256::OUTPUT_SIZE];
    let mut pss_signature = vec![0u8; key.signature_len()];
    key
      .sign_pss_with_salt_and_blinding_factor(
        RsaPssProfile::Sha256,
        message,
        &salt,
        RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
        &mut pss_signature,
      )
      .expect("CAVP-derived key must sign PSS with explicit salt and blinding");
    key
      .public_key()
      .verify_pss(RsaPssProfile::Sha256, message, &pss_signature)
      .expect("CAVP-derived PSS signature must verify");

    let label = b"rscrypto-cavp-keygen-oaep";
    let plaintext = b"NIST CAVP keygen candidate OAEP";
    let seed = [0x3d; Sha256::OUTPUT_SIZE];
    let mut ciphertext = vec![0u8; key.signature_len()];
    key
      .public_key()
      .encrypt_oaep_with_random_fill(
        RsaOaepProfile::Sha256,
        label,
        plaintext,
        &mut ciphertext,
        seed_fill(&seed),
      )
      .expect("CAVP-derived OAEP inputs must encrypt");
    let mut decrypted = vec![0u8; key.signature_len()];
    let decrypted_len = key
      .decrypt_oaep_with_blinding_factor(
        RsaOaepProfile::Sha256,
        label,
        &ciphertext,
        RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
        &mut decrypted,
      )
      .expect("CAVP-derived OAEP ciphertext must decrypt");
    assert_eq!(&decrypted[..decrypted_len], plaintext);

    let pkcs1v15_plaintext = b"NIST CAVP keygen candidate RSAES-PKCS1-v1_5";
    let pkcs1v15_seed = vec![0x69; key.signature_len().strict_sub(pkcs1v15_plaintext.len()).strict_sub(3)];
    key
      .public_key()
      .encrypt_pkcs1v15_with_random_fill(pkcs1v15_plaintext, &mut ciphertext, seed_fill(&pkcs1v15_seed))
      .expect("CAVP-derived PKCS#1 v1.5 inputs must encrypt");
    let decrypted_len = key
      .decrypt_pkcs1v15_with_blinding_factor(
        &ciphertext,
        RsaBlindingPair::new(&blinding_factor, &blinding_factor_inverse),
        &mut decrypted,
      )
      .expect("CAVP-derived PKCS#1 v1.5 ciphertext must decrypt");
    assert_eq!(&decrypted[..decrypted_len], pkcs1v15_plaintext);

    covered.push(modulus_bits);
  }

  assert_eq!(covered, [2048, 3072]);
}

#[cfg(feature = "getrandom")]
#[test]
fn keygen_generate_with_policy_produces_usable_private_key_end_to_end() {
  let key = RsaPrivateKey::generate_with_policy(2048, &RsaPublicKeyPolicy::modulus_2048_to_8192())
    .expect("legacy test policy must generate a 2048-bit key");
  assert_eq!(key.public_key().modulus_bits(), 2048);
  assert_eq!(key.public_key().public_exponent().as_u64(), RSA_KEYGEN_PUBLIC_EXPONENT);

  let pkcs1 = key.to_pkcs1_der();
  let pkcs8 = key.to_pkcs8_der();
  let pkcs1_imported = RsaPrivateKey::from_pkcs1_der_with_policy(&pkcs1, &RsaPublicKeyPolicy::modulus_2048_to_8192())
    .expect("generated PKCS#1 key must re-import");
  let pkcs8_imported = RsaPrivateKey::from_pkcs8_der_with_policy(&pkcs8, &RsaPublicKeyPolicy::modulus_2048_to_8192())
    .expect("generated PKCS#8 key must re-import");
  assert_eq!(pkcs1_imported.public_key(), key.public_key());
  assert_eq!(pkcs8_imported.public_key(), key.public_key());

  macro_rules! assert_generated_key_profile {
    ($pkcs1_profile:expr, $pss_profile:expr, $oaep_profile:expr, $seed:expr, $plaintext:expr) => {{
      let message = b"rscrypto generated RSA key signs end to end";
      let mut pkcs1v15_signature = vec![0u8; key.signature_len()];
      key
        .sign_pkcs1v15($pkcs1_profile, message, &mut pkcs1v15_signature)
        .expect("generated key must sign PKCS#1 v1.5");
      key
        .public_key()
        .verify_pkcs1v15($pkcs1_profile, message, &pkcs1v15_signature)
        .expect("generated-key PKCS#1 v1.5 signature must verify");

      let mut pss_signature = vec![0u8; key.signature_len()];
      key
        .sign_pss($pss_profile, message, &mut pss_signature)
        .expect("generated key must sign PSS");
      key
        .public_key()
        .verify_pss($pss_profile, message, &pss_signature)
        .expect("generated-key PSS signature must verify");

      let label = b"rscrypto-generated-key-oaep";
      let mut ciphertext = vec![0u8; key.signature_len()];
      key
        .public_key()
        .encrypt_oaep_with_random_fill($oaep_profile, label, $plaintext, &mut ciphertext, seed_fill(&$seed))
        .expect("generated-key OAEP inputs must encrypt");
      let mut decrypted = vec![0u8; key.signature_len()];
      let decrypted_len = key
        .decrypt_oaep($oaep_profile, label, &ciphertext, &mut decrypted)
        .expect("generated-key OAEP ciphertext must decrypt");
      assert_eq!(&decrypted[..decrypted_len], $plaintext);

      let pkcs1v15_plaintext = b"generated key RSAES-PKCS1-v1_5";
      let pkcs1v15_seed = vec![0x63; key.signature_len().strict_sub(pkcs1v15_plaintext.len()).strict_sub(3)];
      key
        .public_key()
        .encrypt_pkcs1v15_with_random_fill(pkcs1v15_plaintext, &mut ciphertext, seed_fill(&pkcs1v15_seed))
        .expect("generated-key PKCS#1 v1.5 inputs must encrypt");
      let decrypted_len = key
        .decrypt_pkcs1v15(&ciphertext, &mut decrypted)
        .expect("generated-key PKCS#1 v1.5 ciphertext must decrypt");
      assert_eq!(&decrypted[..decrypted_len], pkcs1v15_plaintext);
    }};
  }

  assert_generated_key_profile!(
    RsaPkcs1v15Profile::Sha256,
    RsaPssProfile::Sha256,
    RsaOaepProfile::Sha256,
    [0x26; Sha256::OUTPUT_SIZE],
    b"generated key sha256 oaep"
  );
  assert_generated_key_profile!(
    RsaPkcs1v15Profile::Sha384,
    RsaPssProfile::Sha384,
    RsaOaepProfile::Sha384,
    [0x38; Sha384::OUTPUT_SIZE],
    b"generated key sha384 oaep"
  );
  assert_generated_key_profile!(
    RsaPkcs1v15Profile::Sha512,
    RsaPssProfile::Sha512,
    RsaOaepProfile::Sha512,
    [0x52; Sha512::OUTPUT_SIZE],
    b"generated key sha512 oaep"
  );
}

#[cfg(feature = "getrandom")]
#[test]
fn keygen_generate_default_modern_key_produces_usable_private_key_end_to_end() {
  let key = RsaPrivateKey::generate(3072).expect("default policy must generate a 3072-bit key");
  assert_eq!(key.public_key().modulus_bits(), 3072);
  assert_eq!(key.public_key().public_exponent().as_u64(), RSA_KEYGEN_PUBLIC_EXPONENT);

  let pkcs1 = key.to_pkcs1_der();
  let pkcs8 = key.to_pkcs8_der();
  assert_eq!(
    RsaPrivateKey::from_pkcs1_der(&pkcs1)
      .expect("generated PKCS#1 key must re-import")
      .public_key(),
    key.public_key()
  );
  assert_eq!(
    RsaPrivateKey::from_pkcs8_der(&pkcs8)
      .expect("generated PKCS#8 key must re-import")
      .public_key(),
    key.public_key()
  );

  let mut scratch = key.private_scratch();
  let message = b"rscrypto generated modern RSA key end to end";
  let mut signature = vec![0u8; key.signature_len()];

  let pkcs1v15_profile = RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256);
  key
    .sign_signature_with_scratch(pkcs1v15_profile, message, &mut signature, &mut scratch)
    .expect("generated modern key must sign PKCS#1 v1.5 with caller scratch");
  key
    .public_key()
    .verify_signature(pkcs1v15_profile, message, &signature)
    .expect("generated modern-key PKCS#1 v1.5 signature must verify");

  let pss_profile = RsaSignatureProfile::pss(RsaPssProfile::Sha256);
  key
    .sign_signature_with_scratch(pss_profile, message, &mut signature, &mut scratch)
    .expect("generated modern key must sign PSS with caller scratch");
  key
    .public_key()
    .verify_signature(pss_profile, message, &signature)
    .expect("generated modern-key PSS signature must verify");

  let label = b"rscrypto-generated-modern-key-oaep";
  let plaintext = b"modern generated key OAEP roundtrip";
  let seed = [0x30; Sha256::OUTPUT_SIZE];
  let mut ciphertext = vec![0u8; key.signature_len()];
  key
    .public_key()
    .encrypt_oaep_with_random_fill(
      RsaOaepProfile::Sha256,
      label,
      plaintext,
      &mut ciphertext,
      seed_fill(&seed),
    )
    .expect("generated modern-key OAEP inputs must encrypt");
  let mut decrypted = vec![0u8; key.signature_len()];
  let decrypted_len = key
    .decrypt_oaep_with_scratch(RsaOaepProfile::Sha256, label, &ciphertext, &mut decrypted, &mut scratch)
    .expect("generated modern-key OAEP ciphertext must decrypt with caller scratch");
  assert_eq!(&decrypted[..decrypted_len], plaintext);

  let pkcs1v15_plaintext = b"modern generated key RSAES-PKCS1-v1_5";
  let pkcs1v15_seed = vec![0x64; key.signature_len().strict_sub(pkcs1v15_plaintext.len()).strict_sub(3)];
  key
    .public_key()
    .encrypt_pkcs1v15_with_random_fill(pkcs1v15_plaintext, &mut ciphertext, seed_fill(&pkcs1v15_seed))
    .expect("generated modern-key PKCS#1 v1.5 inputs must encrypt");
  let decrypted_len = key
    .decrypt_pkcs1v15_with_scratch(&ciphertext, &mut decrypted, &mut scratch)
    .expect("generated modern-key PKCS#1 v1.5 ciphertext must decrypt with caller scratch");
  assert_eq!(&decrypted[..decrypted_len], pkcs1v15_plaintext);
}

#[cfg(feature = "getrandom")]
#[test]
fn keygen_generate_rejects_policy_disallowed_sizes_before_entropy() {
  assert_eq!(
    RsaPrivateKey::generate(2048).err(),
    Some(RsaKeyGenerationError::InvalidModulusBits)
  );
  assert_eq!(
    RsaPrivateKey::generate(3071).err(),
    Some(RsaKeyGenerationError::InvalidModulusBits)
  );
  assert_eq!(
    RsaPrivateKey::generate(8193).err(),
    Some(RsaKeyGenerationError::InvalidModulusBits)
  );

  let impossible_policy = RsaPublicKeyPolicy::modulus_2048_to_8192()
    .with_min_modulus_bits(4096)
    .with_max_modulus_bits(3072);
  assert_eq!(
    RsaPrivateKey::generate_with_policy(4096, &impossible_policy).err(),
    Some(RsaKeyGenerationError::InvalidModulusBits)
  );

  let undersized_policy = RsaPublicKeyPolicy::modulus_2048_to_8192().with_min_modulus_bits(3072);
  assert_eq!(
    RsaPrivateKey::generate_with_policy(2048, &undersized_policy).err(),
    Some(RsaKeyGenerationError::InvalidModulusBits)
  );
}

#[cfg(feature = "getrandom")]
#[test]
fn keygen_build_private_key_from_primes_rejects_duplicate_and_wrong_width_candidates() {
  let modulus_bits = unsigned_be_bit_len(&rsa_private_modulus());
  let policy = RsaPublicKeyPolicy::modulus_2048_to_8192();

  assert!(
    keygen_build_private_key_from_primes(modulus_bits, &policy, rsa_private_prime_p(), rsa_private_prime_p())
      .expect("duplicate prime fixtures must complete validation")
      .is_none()
  );
  assert!(
    keygen_build_private_key_from_primes(
      modulus_bits.strict_sub(1),
      &policy,
      rsa_private_prime_p(),
      rsa_private_prime_q()
    )
    .expect("undersized modulus candidates must complete validation")
    .is_none()
  );
  assert!(
    keygen_build_private_key_from_primes(
      modulus_bits.strict_add(1),
      &policy,
      rsa_private_prime_p(),
      rsa_private_prime_q()
    )
    .expect("oversized modulus candidates must complete validation")
    .is_none()
  );

  let mut close_prime_q = rsa_private_prime_p();
  let mut carry = 2u16;
  for byte in close_prime_q.iter_mut().rev() {
    let sum = u16::from(*byte).strict_add(carry);
    *byte = sum.to_le_bytes()[0];
    carry = sum >> 8;
    if carry == 0 {
      break;
    }
  }
  assert_eq!(carry, 0);
  assert!(
    !keygen_prime_distance_is_sufficient(&rsa_private_prime_p(), &close_prime_q, modulus_bits),
    "key generation must reject dangerously close prime candidates"
  );
  assert!(
    keygen_build_private_key_from_primes(modulus_bits, &policy, rsa_private_prime_p(), close_prime_q)
      .expect("close-prime candidates must complete validation")
      .is_none()
  );

  let mut public_exponent_conflict = rsa_private_prime_p();
  let remainder = unsigned_be_mod_u64(&public_exponent_conflict, RSA_KEYGEN_PUBLIC_EXPONENT);
  let mut delta = (RSA_KEYGEN_PUBLIC_EXPONENT.strict_add(1).strict_sub(remainder)) % RSA_KEYGEN_PUBLIC_EXPONENT;
  for byte in public_exponent_conflict.iter_mut().rev() {
    if delta == 0 {
      break;
    }
    let sum = u64::from(*byte).strict_add(delta & 0xff);
    *byte = sum.to_le_bytes()[0];
    delta = (delta >> 8).strict_add(sum >> 8);
  }
  assert_eq!(delta, 0);
  assert!(
    keygen_conflicts_with_public_exponent(&public_exponent_conflict),
    "test candidate must conflict with RSA key-generation public exponent"
  );
  assert!(
    keygen_build_private_key_from_primes(modulus_bits, &policy, public_exponent_conflict, rsa_private_prime_q())
      .expect("public-exponent-conflicting candidates must complete validation")
      .is_none()
  );
}

#[cfg(feature = "getrandom")]
#[test]
fn keygen_random_prime_search_returns_probable_prime() {
  let mut drbg = test_keygen_drbg(b"random-prime-search");
  let prime = keygen_generate_prime(&mut drbg, 128, 256, None, 256usize.strict_mul(5))
    .expect("deterministic prime search must complete");

  assert_eq!(unsigned_be_bit_len(&prime), 128);
  assert_eq!(prime.last().copied().unwrap_or_default() & 1, 1);
  assert!(!has_small_prime_factor(&prime));
  assert!(!keygen_conflicts_with_public_exponent(&prime));
  assert!(keygen_probable_prime_meets_fips_lower_bound(&prime, 128));
  assert!(keygen_is_probable_prime(&mut drbg, &prime).expect("generated prime must admit a primality test"));
}

#[test]
fn pkcs1_private_key_parser_applies_modulus_policy_before_component_checks() {
  let mut bad_q = rsa_private_prime_q();
  *bad_q.last_mut().expect("prime q fixture must be nonempty") ^= 0x02;
  let der = test_pkcs1_private_key_with_components(
    &[0x01, 0x00, 0x01],
    &rsa_private_exponent(),
    &rsa_private_prime_p(),
    &bad_q,
    &rsa_private_exponent_p(),
    &rsa_private_exponent_q(),
    &rsa_private_coefficient(),
  );
  let policy = RsaPublicKeyPolicy::modulus_2048_to_8192().with_max_modulus_bits(1024);

  assert_eq!(
    parse_pkcs1_private_key_der_with_policy(&der, &policy).err(),
    Some(RsaKeyError::InvalidModulus)
  );
}

#[test]
fn pkcs8_private_key_parser_requires_rsa_algorithm_and_strict_inner_key() {
  let pkcs1 = test_pkcs1_private_key();
  let rsa_algorithm = algorithm_identifier(RSA_ENCRYPTION_OID, Some(&null()));
  let der = test_pkcs8_private_key(&pkcs1, &rsa_algorithm);
  let key = parse_pkcs8_private_key_der_with_policy(&der, &RsaPublicKeyPolicy::modulus_2048_to_8192())
    .expect("valid PKCS#8 fixture must parse");

  assert_eq!(key.public_key().modulus(), rsa_private_modulus());

  let pss_algorithm = algorithm_identifier(ID_RSASSA_PSS_OID, Some(&tlv(TAG_SEQUENCE, &[])));
  let wrong_algorithm = test_pkcs8_private_key(&pkcs1, &pss_algorithm);
  assert_eq!(
    parse_pkcs8_private_key_der_with_policy(&wrong_algorithm, &RsaPublicKeyPolicy::modulus_2048_to_8192()).err(),
    Some(RsaKeyError::UnsupportedAlgorithm)
  );
}

#[test]
fn private_key_parser_rejects_noncanonical_container_lengths_and_attributes() {
  let pkcs1 = test_pkcs1_private_key();
  assert_eq!(
    parse_pkcs1_private_key_der_with_policy(
      &tlv_with_leading_zero_long_len(&pkcs1),
      &RsaPublicKeyPolicy::modulus_2048_to_8192()
    )
    .err(),
    Some(RsaKeyError::MalformedDer)
  );

  let mut pkcs1_body = Vec::new();
  for field in [
    tlv_with_noncanonical_short_len(TAG_INTEGER, &[0]),
    integer_unsigned(&rsa_private_modulus()),
    integer_unsigned(&[0x01, 0x00, 0x01]),
    integer_unsigned(&rsa_private_exponent()),
    integer_unsigned(&rsa_private_prime_p()),
    integer_unsigned(&rsa_private_prime_q()),
    integer_unsigned(&rsa_private_exponent_p()),
    integer_unsigned(&rsa_private_exponent_q()),
    integer_unsigned(&rsa_private_coefficient()),
  ] {
    pkcs1_body.extend_from_slice(&field);
  }
  assert_eq!(
    parse_pkcs1_private_key_der_with_policy(
      &tlv(TAG_SEQUENCE, &pkcs1_body),
      &RsaPublicKeyPolicy::modulus_2048_to_8192()
    )
    .err(),
    Some(RsaKeyError::MalformedDer)
  );

  let rsa_algorithm = algorithm_identifier(RSA_ENCRYPTION_OID, Some(&null()));
  let pkcs8 = test_pkcs8_private_key(&pkcs1, &rsa_algorithm);
  assert_eq!(
    parse_pkcs8_private_key_der_with_policy(
      &tlv_with_leading_zero_long_len(&pkcs8),
      &RsaPublicKeyPolicy::modulus_2048_to_8192()
    )
    .err(),
    Some(RsaKeyError::MalformedDer)
  );

  let mut pkcs8_with_attributes = Vec::new();
  pkcs8_with_attributes.extend_from_slice(&integer_unsigned(&[0]));
  pkcs8_with_attributes.extend_from_slice(&rsa_algorithm);
  pkcs8_with_attributes.extend_from_slice(&tlv(TAG_OCTET_STRING, &pkcs1));
  pkcs8_with_attributes.extend_from_slice(&context_constructed(0, &[]));
  assert_eq!(
    parse_pkcs8_private_key_der_with_policy(
      &tlv(TAG_SEQUENCE, &pkcs8_with_attributes),
      &RsaPublicKeyPolicy::modulus_2048_to_8192()
    )
    .err(),
    Some(RsaKeyError::MalformedDer)
  );
}

#[test]
fn pkcs1_private_key_parser_rejects_multiprime_and_inconsistent_components() {
  let mut multiprime_body = Vec::new();
  multiprime_body.extend_from_slice(&integer_unsigned(&[1]));
  multiprime_body.extend_from_slice(&integer_unsigned(&rsa_private_modulus()));
  multiprime_body.extend_from_slice(&integer_unsigned(&[0x01, 0x00, 0x01]));
  multiprime_body.extend_from_slice(&integer_unsigned(&rsa_private_exponent()));
  multiprime_body.extend_from_slice(&integer_unsigned(&rsa_private_prime_p()));
  multiprime_body.extend_from_slice(&integer_unsigned(&rsa_private_prime_q()));
  multiprime_body.extend_from_slice(&integer_unsigned(&rsa_private_exponent_p()));
  multiprime_body.extend_from_slice(&integer_unsigned(&rsa_private_exponent_q()));
  multiprime_body.extend_from_slice(&integer_unsigned(&rsa_private_coefficient()));
  let multiprime = tlv(TAG_SEQUENCE, &multiprime_body);
  assert_eq!(
    parse_pkcs1_private_key_der_with_policy(&multiprime, &RsaPublicKeyPolicy::modulus_2048_to_8192()).err(),
    Some(RsaKeyError::UnsupportedAlgorithm)
  );

  let mut bad_q = rsa_private_prime_q();
  let last = bad_q.last_mut().expect("prime q fixture must be nonempty");
  *last ^= 0x02;
  let mut inconsistent_body = Vec::new();
  for field in [
    integer_unsigned(&[0]),
    integer_unsigned(&rsa_private_modulus()),
    integer_unsigned(&[0x01, 0x00, 0x01]),
    integer_unsigned(&rsa_private_exponent()),
    integer_unsigned(&rsa_private_prime_p()),
    integer_unsigned(&bad_q),
    integer_unsigned(&rsa_private_exponent_p()),
    integer_unsigned(&rsa_private_exponent_q()),
    integer_unsigned(&rsa_private_coefficient()),
  ] {
    inconsistent_body.extend_from_slice(&field);
  }
  let inconsistent = tlv(TAG_SEQUENCE, &inconsistent_body);
  assert_eq!(
    parse_pkcs1_private_key_der_with_policy(&inconsistent, &RsaPublicKeyPolicy::modulus_2048_to_8192()).err(),
    Some(RsaKeyError::InvalidModulus)
  );
}

#[test]
fn pkcs1_private_key_parser_rejects_crt_congruence_mismatches() {
  let mut bad_exponent_p = rsa_private_exponent_p();
  *bad_exponent_p
    .last_mut()
    .expect("CRT exponent p fixture must be nonempty") ^= 0x02;
  let bad_dp = test_pkcs1_private_key_with_crt(
    &[0x01, 0x00, 0x01],
    &bad_exponent_p,
    &rsa_private_exponent_q(),
    &rsa_private_coefficient(),
  );
  assert_eq!(
    parse_pkcs1_private_key_der_with_policy(&bad_dp, &RsaPublicKeyPolicy::modulus_2048_to_8192()).err(),
    Some(RsaKeyError::InvalidModulus)
  );

  let mut bad_exponent_q = rsa_private_exponent_q();
  *bad_exponent_q
    .last_mut()
    .expect("CRT exponent q fixture must be nonempty") ^= 0x02;
  let bad_dq = test_pkcs1_private_key_with_crt(
    &[0x01, 0x00, 0x01],
    &rsa_private_exponent_p(),
    &bad_exponent_q,
    &rsa_private_coefficient(),
  );
  assert_eq!(
    parse_pkcs1_private_key_der_with_policy(&bad_dq, &RsaPublicKeyPolicy::modulus_2048_to_8192()).err(),
    Some(RsaKeyError::InvalidModulus)
  );

  let mut bad_coefficient = rsa_private_coefficient();
  *bad_coefficient
    .last_mut()
    .expect("CRT coefficient fixture must be nonempty") ^= 0x02;
  let bad_qinv = test_pkcs1_private_key_with_crt(
    &[0x01, 0x00, 0x01],
    &rsa_private_exponent_p(),
    &rsa_private_exponent_q(),
    &bad_coefficient,
  );
  assert_eq!(
    parse_pkcs1_private_key_der_with_policy(&bad_qinv, &RsaPublicKeyPolicy::modulus_2048_to_8192()).err(),
    Some(RsaKeyError::InvalidModulus)
  );
}

#[test]
fn pkcs1_private_key_parser_rejects_invalid_secret_ranges() {
  let policy = RsaPublicKeyPolicy::modulus_2048_to_8192();

  let zero_private_exponent = test_pkcs1_private_key_with_components(
    &[0x01, 0x00, 0x01],
    &[0],
    &rsa_private_prime_p(),
    &rsa_private_prime_q(),
    &rsa_private_exponent_p(),
    &rsa_private_exponent_q(),
    &rsa_private_coefficient(),
  );
  assert_eq!(
    parse_pkcs1_private_key_der_with_policy(&zero_private_exponent, &policy).err(),
    Some(RsaKeyError::InvalidModulus)
  );

  let modulus_sized_private_exponent = test_pkcs1_private_key_with_components(
    &[0x01, 0x00, 0x01],
    &rsa_private_modulus(),
    &rsa_private_prime_p(),
    &rsa_private_prime_q(),
    &rsa_private_exponent_p(),
    &rsa_private_exponent_q(),
    &rsa_private_coefficient(),
  );
  assert_eq!(
    parse_pkcs1_private_key_der_with_policy(&modulus_sized_private_exponent, &policy).err(),
    Some(RsaKeyError::InvalidModulus)
  );

  let zero_prime = test_pkcs1_private_key_with_components(
    &[0x01, 0x00, 0x01],
    &rsa_private_exponent(),
    &[0],
    &rsa_private_prime_q(),
    &rsa_private_exponent_p(),
    &rsa_private_exponent_q(),
    &rsa_private_coefficient(),
  );
  assert_eq!(
    parse_pkcs1_private_key_der_with_policy(&zero_prime, &policy).err(),
    Some(RsaKeyError::InvalidModulus)
  );

  let even_prime = test_pkcs1_private_key_with_components(
    &[0x01, 0x00, 0x01],
    &rsa_private_exponent(),
    &[2],
    &rsa_private_prime_q(),
    &rsa_private_exponent_p(),
    &rsa_private_exponent_q(),
    &rsa_private_coefficient(),
  );
  assert_eq!(
    parse_pkcs1_private_key_der_with_policy(&even_prime, &policy).err(),
    Some(RsaKeyError::InvalidModulus)
  );

  let zero_crt_exponent = test_pkcs1_private_key_with_components(
    &[0x01, 0x00, 0x01],
    &rsa_private_exponent(),
    &rsa_private_prime_p(),
    &rsa_private_prime_q(),
    &[0],
    &rsa_private_exponent_q(),
    &rsa_private_coefficient(),
  );
  assert_eq!(
    parse_pkcs1_private_key_der_with_policy(&zero_crt_exponent, &policy).err(),
    Some(RsaKeyError::InvalidModulus)
  );

  let out_of_range_coefficient = test_pkcs1_private_key_with_components(
    &[0x01, 0x00, 0x01],
    &rsa_private_exponent(),
    &rsa_private_prime_p(),
    &rsa_private_prime_q(),
    &rsa_private_exponent_p(),
    &rsa_private_exponent_q(),
    &rsa_private_prime_p(),
  );
  assert_eq!(
    parse_pkcs1_private_key_der_with_policy(&out_of_range_coefficient, &policy).err(),
    Some(RsaKeyError::InvalidModulus)
  );
}

#[test]
fn pkcs1_private_key_parser_rejects_public_private_exponent_mismatch() {
  let wrong_exponent = test_pkcs1_private_key_with_public_exponent(&[3]);
  assert_eq!(
    parse_pkcs1_private_key_der_with_policy(
      &wrong_exponent,
      &RsaPublicKeyPolicy::modulus_2048_to_8192().allow_legacy_small_exponents()
    )
    .err(),
    Some(RsaKeyError::InvalidModulus)
  );
}

#[test]
fn pkcs1_private_key_parser_rejects_noncanonical_integer_and_trailing_data() {
  let mut noncanonical_exponent_p = rsa_private_exponent_p();
  noncanonical_exponent_p.insert(0, 0);
  let mut noncanonical_body = Vec::new();
  for field in [
    integer_unsigned(&[0]),
    integer_unsigned(&rsa_private_modulus()),
    integer_unsigned(&[0x01, 0x00, 0x01]),
    integer_unsigned(&rsa_private_exponent()),
    integer_unsigned(&rsa_private_prime_p()),
    integer_unsigned(&rsa_private_prime_q()),
    integer_der_value(&noncanonical_exponent_p),
    integer_unsigned(&rsa_private_exponent_q()),
    integer_unsigned(&rsa_private_coefficient()),
  ] {
    noncanonical_body.extend_from_slice(&field);
  }
  let noncanonical = tlv(TAG_SEQUENCE, &noncanonical_body);
  assert_eq!(
    parse_pkcs1_private_key_der_with_policy(&noncanonical, &RsaPublicKeyPolicy::modulus_2048_to_8192()).err(),
    Some(RsaKeyError::MalformedDer)
  );

  let mut trailing = test_pkcs1_private_key();
  trailing.extend_from_slice(&null());
  assert_eq!(
    parse_pkcs1_private_key_der_with_policy(&trailing, &RsaPublicKeyPolicy::modulus_2048_to_8192()).err(),
    Some(RsaKeyError::MalformedDer)
  );
}
