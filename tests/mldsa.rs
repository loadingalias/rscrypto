//! Interoperability and hostile-input tests of the public ML-DSA API.
#![cfg(feature = "ml-dsa")]

use rscrypto::*;
use rustcrypto_ml_dsa as oracle;

macro_rules! reject_signing_failures {
  ($key:expr) => {{
    let digest = [0x42; 32];
    let prehash = MlDsaPrehash::new(MlDsaPrehashAlgorithm::Sha256, &digest).expect("digest length");
    let mut calls = 0usize;
    let mut entropy = |out: &mut [u8]| {
      calls = calls.strict_add(1);
      out[..7].fill(0xff);
      Err(MlDsaError::RandomGenerationFailed)
    };
    assert_eq!(
      $key.sign_with(b"message", &[0; 256], &mut entropy),
      Err(MlDsaError::ContextTooLong)
    );
    assert_eq!(
      $key.sign_prehash_with(prehash, &[0; 256], &mut entropy),
      Err(MlDsaError::ContextTooLong)
    );
    assert_eq!(
      $key.sign_with(b"message", b"context", &mut entropy),
      Err(MlDsaError::RandomGenerationFailed)
    );
    assert_eq!(
      $key.sign_prehash_with(prehash, b"context", &mut entropy),
      Err(MlDsaError::RandomGenerationFailed)
    );
    assert_eq!(calls, 2, "invalid contexts must not consume entropy");
  }};
}

macro_rules! interop {
  ($name:ident, $profile:ident, $secret:ident, $signature:ident, $oracle:ident) => {
    #[test]
    fn $name() {
      for seed_byte in [0, 1, 0x7f, 0xff] {
        let seed = [seed_byte; 32];
        let (public, secret) = $profile::keypair_from_seed(&seed).expect("key generation");
        let external = oracle::ExpandedSigningKey::<oracle::$oracle>::from_seed(&seed.into());
        let external_public = external.verifying_key();
        assert_eq!(public.as_bytes().as_slice(), external_public.encode().as_slice());
        #[expect(
          deprecated,
          reason = "FIPS 204 expanded encoding is the interoperability contract under test"
        )]
        let encoded = external.to_expanded();
        assert_eq!(secret.expose_secret().as_bytes().as_slice(), encoded.as_slice());
        let imported = $secret::try_from_slice(encoded.as_slice()).expect("key import");
        let prepared = imported.prepare().expect("prepare signing key");
        let verifier = public.prepare().expect("prepare public key");
        assert_eq!(imported.public_key(), &public);
        if seed_byte == 0 {
          reject_signing_failures!(secret);
          reject_signing_failures!(prepared);
          // The following independent signature comparisons also prove reuse
          // after context rejection and partially filled entropy failures.
        }
        for (message_len, context_len) in [(0, 0), (135, 1), (136, 254), (137, 255), (4096, 17)] {
          let message = vec![seed_byte; message_len];
          let context = vec![0x55; context_len];
          let ours = secret.sign_deterministic(&message, &context).expect("sign");
          let theirs = external.sign_deterministic(&message, &context).expect("oracle sign");
          assert_eq!(ours.as_bytes().as_slice(), theirs.encode().as_slice());
          assert_eq!(
            prepared
              .sign_deterministic(&message, &context)
              .expect("prepared signing"),
            secret.sign_deterministic(&message, &context).expect("ordinary signing")
          );
          verifier
            .verify_with_context(&message, &context, &ours)
            .expect("prepared verifier");
          assert!(external_public.verify_with_context(&message, &context, &theirs));
          public
            .verify_with_context(
              &message,
              &context,
              &$signature::try_from_slice(theirs.encode().as_slice()).expect("signature import"),
            )
            .expect("verify oracle signature");
          // sign_internal takes the standard's formatted message M', not raw M.
          let prefix = [0, u8::try_from(context_len).expect("bounded context")];
          let random = [seed_byte ^ 0xa5; 32];
          let theirs = external.sign_internal(&[&prefix, &context, &message], &random.into());
          let ours = secret
            .sign_with(&message, &context, |out| {
              out.copy_from_slice(&random);
              Ok(())
            })
            .expect("hedged sign");
          assert_eq!(ours.as_bytes().as_slice(), theirs.encode().as_slice());
          public
            .verify_with_context(&message, &context, &ours)
            .expect("verify hedged signature");
          assert_eq!(
            prepared
              .sign_with(&message, &context, |out| {
                out.copy_from_slice(&random);
                Ok(())
              })
              .expect("prepared hedged signing"),
            ours
          );
          verifier
            .verify_with_context(&message, &context, &ours)
            .expect("prepared hedged verifier");
          assert!(verifier.verify_with_context(b"wrong message", &context, &ours).is_err());
          assert!(public.verify_with_context(&message, b"wrong context", &ours).is_err());
          assert!(public.verify_with_context(b"wrong message", &context, &ours).is_err());
        }
      }
    }
  };
}

interop!(interop_44, MlDsa44, MlDsa44SecretKey, MlDsa44Signature, MlDsa44);
interop!(interop_65, MlDsa65, MlDsa65SecretKey, MlDsa65Signature, MlDsa65);
interop!(interop_87, MlDsa87, MlDsa87SecretKey, MlDsa87Signature, MlDsa87);

#[test]
fn context_and_entropy_failure_leave_key_reusable() {
  let (public, secret) = MlDsa44::keypair_from_seed(&[8; 32]).expect("key generation");
  let expected = secret.sign_deterministic(b"message", b"context").expect("sign");
  let mut calls = 0;
  let error = secret.sign_with(b"message", &[0; 256], |_| {
    calls += 1;
    Ok(())
  });
  assert_eq!(error, Err(MlDsaError::ContextTooLong));
  assert_eq!(calls, 0, "invalid public inputs must not consume entropy");
  let error = secret.sign_with(b"message", b"context", |out| {
    out[..7].fill(0xff);
    Err(MlDsaError::RandomGenerationFailed)
  });
  assert_eq!(error, Err(MlDsaError::RandomGenerationFailed));
  assert_eq!(secret.sign_deterministic(b"message", b"context"), Ok(expected.clone()));
  assert!(public.verify_with_context(b"message", &[0; 256], &expected).is_err());
  MlDsa44::generate_keypair(|out| {
    out[..13].fill(0xff);
    Err(MlDsaError::RandomGenerationFailed)
  })
  .expect_err("malformed input must be rejected");
  assert_eq!(format!("{secret:?}"), "MlDsa44SecretKey(****)");
}

#[test]
fn expanded_secret_import_checks_redundancy_and_noise_range() {
  let (_, secret) = MlDsa44::keypair_from_seed(&[3; 32]).expect("key generation");
  let encoded = secret.expose_secret();
  for offset in [0, 64, 128, 896, 2559] {
    let mut damaged = *encoded.as_bytes();
    damaged[offset] ^= 0x80;
    assert!(MlDsa44SecretKey::try_from_slice(&damaged).is_err(), "offset {offset}");
  }
  let mut damaged = *encoded.as_bytes();
  damaged[128] |= 7; // eta=2 permits only the three-bit encodings 0..=4.
  MlDsa44SecretKey::try_from_slice(&damaged).expect_err("malformed input must be rejected");
  MlDsa44SecretKey::try_from_slice(&encoded.as_bytes()[1..]).expect_err("malformed input must be rejected");
  MlDsa44PublicKey::try_from_slice(&[0; 1311]).expect_err("malformed input must be rejected");
}

#[test]
fn signature_parser_rejects_noncanonical_hints_and_response_bounds() {
  let (_, secret) = MlDsa44::keypair_from_seed(&[9; 32]).expect("key generation");
  let signature = secret.sign_deterministic(b"parser", &[]).expect("sign");
  let hints = 32 + 4 * 576;
  let mut bytes = signature.to_bytes();
  bytes[hints + 80] = 81;
  MlDsa44Signature::try_from_slice(&bytes).expect_err("malformed input must be rejected");
  bytes = signature.to_bytes();
  bytes[hints..].fill(0);
  bytes[hints] = 12;
  bytes[hints + 1] = 12;
  bytes[hints + 80..].fill(2); // Duplicate index in the first polynomial.
  MlDsa44Signature::try_from_slice(&bytes).expect_err("malformed input must be rejected");
  bytes[hints] = 13; // Descending indices are also noncanonical.
  MlDsa44Signature::try_from_slice(&bytes).expect_err("malformed input must be rejected");
  bytes[hints..].fill(0);
  bytes[hints + 79] = 1; // Unused positions must be zero.
  MlDsa44Signature::try_from_slice(&bytes).expect_err("malformed input must be rejected");
  bytes = signature.to_bytes();
  bytes[32] = 0;
  bytes[33] = 0;
  bytes[34] &= 0xfc; // First response = gamma1, outside the accepted norm.
  MlDsa44Signature::try_from_slice(&bytes).expect_err("malformed input must be rejected");
  MlDsa44Signature::try_from_slice(&bytes[1..]).expect_err("malformed input must be rejected");
}

#[test]
fn prehash_identifiers_and_pure_mode_are_separate_domains() {
  let (public, secret) = MlDsa65::keypair_from_seed(&[0x31; 32]).expect("key generation");
  let digest = [0x42; 32];
  let sha2 = MlDsaPrehash::new(MlDsaPrehashAlgorithm::Sha256, &digest).expect("digest length");
  let sha3 = MlDsaPrehash::new(MlDsaPrehashAlgorithm::Sha3_256, &digest).expect("digest length");
  let signature = secret
    .sign_prehash_deterministic(sha2, b"protocol")
    .expect("prehash sign");
  public
    .verify_prehash(sha2, b"protocol", &signature)
    .expect("prehash verify");
  assert!(public.verify_prehash(sha3, b"protocol", &signature).is_err());
  assert!(public.verify_with_context(&digest, b"protocol", &signature).is_err());
  MlDsaPrehash::new(MlDsaPrehashAlgorithm::Sha256, &[0; 31]).expect_err("malformed input must be rejected");
}

#[cfg(feature = "serde")]
#[test]
fn serde_retains_canonical_validation() {
  let (public, secret) = MlDsa87::keypair_from_seed(&[7; 32]).expect("key generation");
  let signature = secret.sign_deterministic(b"serde", &[]).expect("sign");
  let encoded = serde_json::to_vec(&signature).expect("serialize signature");
  let decoded: MlDsa87Signature = serde_json::from_slice(&encoded).expect("deserialize signature");
  assert_eq!(decoded, signature);
  let decoded: MlDsa87PublicKey = serde_json::from_slice(&serde_json::to_vec(&public).expect("serialize public key"))
    .expect("deserialize public key");
  assert_eq!(decoded, public);
  let mut extended = signature.as_bytes().to_vec();
  extended.push(0);
  serde_json::from_slice::<MlDsa87Signature>(&serde_json::to_vec(&extended).expect("serialize extended signature"))
    .expect_err("malformed input must be rejected");
  let mut malformed = signature.to_bytes();
  malformed[4626] = 255;
  serde_json::from_slice::<MlDsa87Signature>(
    &serde_json::to_vec(malformed.as_slice()).expect("serialize malformed signature"),
  )
  .expect_err("malformed input must be rejected");
  #[cfg(feature = "serde-secrets")]
  {
    let encoded = serde_json::to_vec(&secret).expect("explicit secret serialization");
    let decoded: MlDsa87SecretKey = serde_json::from_slice(&encoded).expect("secret import");
    assert_eq!(decoded.public_key(), &public);
    assert_eq!(
      decoded.sign_deterministic(b"serde", &[]).expect("imported signing"),
      signature
    );
    let mut bytes = *secret.expose_secret().as_bytes();
    bytes[64] ^= 1;
    serde_json::from_slice::<MlDsa87SecretKey>(
      &serde_json::to_vec(bytes.as_slice()).expect("malformed private encoding"),
    )
    .expect_err("malformed input must be rejected");
  }
}
