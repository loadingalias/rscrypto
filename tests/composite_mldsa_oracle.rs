//! Composite ML-DSA against independent implementations of
//! draft-ietf-lamps-pq-composite-sigs.
//!
//! - `darkbio-crypto` (`xdsa`) implements `id-MLDSA65-Ed25519-SHA512` with
//!   the draft's raw encodings and the empty context. It signs ML-DSA
//!   deterministically, so deterministic signatures must match byte for byte.
//! - `kryptering` implements `id-MLDSA65-ECDSA-P256-SHA512` with the same
//!   combiner, label, and empty context, and the JOSE encodings of the P-256
//!   component: a raw scalar, an `x || y` point, and an `r || s` signature.
//!   The conversions below change only those component encodings.
//!
//! Both comparators use RustCrypto ML-DSA, Ed25519, and P-256, which are
//! independent of rscrypto's components.
#![cfg(all(feature = "mldsa65-ed25519", feature = "mldsa65-ecdsa-p256"))]

use kryptering::traits::{Signer as _, Verifier as _};
use kryptering::{CompositeMlDsaVariant, SignatureAlgorithm, SoftwareKey, SoftwareSigner, SoftwareVerifier};
use rscrypto::*;

/// Deterministic, distinct key and message material per round.
fn filler(round: u8) -> impl FnMut(&mut [u8]) -> Result<(), CompositeMlDsaError> {
  let mut counter = 0u8;
  move |out| {
    for byte in out.iter_mut() {
      *byte = round.wrapping_mul(31).wrapping_add(counter);
      counter = counter.wrapping_add(17);
    }
    Ok(())
  }
}

/// Messages of representative lengths, including empty and multi-block.
fn messages() -> Vec<Vec<u8>> {
  [0usize, 1, 55, 111, 128, 1000, 4096]
    .iter()
    .map(|&len| (0..len).map(|i| u8::try_from(i % 251).expect("byte")).collect())
    .collect()
}

#[test]
fn mldsa65_ed25519_matches_darkbio_xdsa() {
  for round in 0..6 {
    let (public, secret) = MlDsa65Ed25519::generate_keypair(filler(round)).expect("generate");
    let raw = secret.expose_secret();
    let oracle = darkbio_crypto::xdsa::SecretKey::from_bytes(raw.as_bytes());
    // Both expand the same seeds to the same public key and encodings.
    assert_eq!(oracle.public_key().to_bytes(), public.to_bytes(), "round {round}");
    assert_eq!(oracle.public_key().to_der().as_slice(), public.to_spki_der().as_slice());
    let oracle_public = darkbio_crypto::xdsa::PublicKey::from_bytes(&public.to_bytes()).expect("oracle key");

    for message in messages() {
      let ours = secret
        .sign_with(&message, b"", filler(round.wrapping_add(100)))
        .expect("sign");
      let deterministic = secret.sign_deterministic(&message, b"").expect("sign");
      for signature in [&ours, &deterministic] {
        oracle_public
          .verify(
            &message,
            &darkbio_crypto::xdsa::Signature::from_bytes(signature.as_bytes()),
          )
          .expect("the oracle accepts rscrypto's signature");
      }
      let theirs = oracle.sign(&message).to_bytes();
      // Both sign ML-DSA deterministically, and Ed25519 is deterministic, so
      // the composite signatures are identical.
      assert_eq!(
        deterministic.as_bytes(),
        &theirs,
        "deterministic signature, round {round}"
      );
      let theirs = MlDsa65Ed25519Signature::try_from_slice(&theirs).expect("oracle signature parses");
      public
        .verify(&message, &theirs)
        .expect("rscrypto accepts the oracle's signature");

      let mut altered = message;
      altered.push(0);
      assert!(
        oracle_public
          .verify(&altered, &darkbio_crypto::xdsa::Signature::from_bytes(ours.as_bytes()))
          .is_err()
      );
      assert!(public.verify(&altered, &theirs).is_err());
    }
  }
}

/// rscrypto's raw private key to kryptering's: the seed and the raw scalar.
fn kryptering_private(secret: &MlDsa65EcdsaP256SecretKey) -> Vec<u8> {
  let raw = secret.expose_secret();
  let raw = raw.as_bytes();
  // seed (32) || 30 31 02 01 01 04 20 <scalar> a0 0a ...
  assert_eq!(&raw[32..39], &[0x30, 0x31, 0x02, 0x01, 0x01, 0x04, 0x20]);
  [&raw[..32], &raw[39..71]].concat()
}

/// rscrypto's public key to kryptering's: the uncompressed point without
/// its `04` prefix.
fn kryptering_public(public: &MlDsa65EcdsaP256PublicKey) -> Vec<u8> {
  let bytes = public.to_bytes();
  let (ml_dsa, point) = bytes.split_at(MlDsa65PublicKey::LENGTH);
  assert_eq!(point[0], 0x04);
  [ml_dsa, &point[1..]].concat()
}

/// A DER `Ecdsa-Sig-Value` component to `r || s`, or back.
fn to_raw_ecdsa(signature: &MlDsa65EcdsaP256Signature) -> Vec<u8> {
  let (ml_dsa, der) = signature.as_bytes().split_at(MlDsa65Signature::LENGTH);
  let ecdsa = EcdsaP256Signature::from_der(der).expect("DER component");
  [ml_dsa, ecdsa.as_bytes()].concat()
}

fn from_raw_ecdsa(signature: &[u8]) -> MlDsa65EcdsaP256Signature {
  let (ml_dsa, raw) = signature.split_at(MlDsa65Signature::LENGTH);
  let ecdsa = EcdsaP256Signature::from_bytes(raw.try_into().expect("r || s")).expect("scalars");
  let mut der = [0; EcdsaP256Signature::DER_MAX_LENGTH];
  let der = ecdsa.to_der_into(&mut der);
  MlDsa65EcdsaP256Signature::try_from_slice(&[ml_dsa, der].concat()).expect("composite signature")
}

#[test]
fn mldsa65_ecdsa_p256_matches_kryptering() {
  let variant = CompositeMlDsaVariant::MlDsa65Es256;
  let algorithm = SignatureAlgorithm::CompositeMlDsa(variant);
  for round in 0..6 {
    let (public, secret) = MlDsa65EcdsaP256::generate_keypair(filler(round)).expect("generate");
    let private = kryptering_private(&secret);
    let oracle_public = kryptering_public(&public);
    // kryptering derives the public key from the private key and rejects a
    // mismatch, so this import also checks rscrypto's key expansion.
    let key = SoftwareKey::from_composite_ml_dsa(variant, Some(&private), &oracle_public).expect("oracle key");
    let signer = SoftwareSigner::new(algorithm, key.clone()).expect("oracle signer");
    let verifier = SoftwareVerifier::new(algorithm, key).expect("oracle verifier");

    for message in messages() {
      let ours = secret
        .sign_with(&message, b"", filler(round.wrapping_add(100)))
        .expect("sign");
      let deterministic = secret.sign_deterministic(&message, b"").expect("sign");
      for signature in [&ours, &deterministic] {
        assert!(
          verifier
            .verify(&message, &to_raw_ecdsa(signature))
            .expect("oracle verification"),
          "the oracle accepts rscrypto's signature"
        );
      }
      let theirs = from_raw_ecdsa(&signer.sign(&message).expect("oracle signature"));
      public
        .verify(&message, &theirs)
        .expect("rscrypto accepts the oracle's signature");

      let mut altered = message;
      altered.push(0);
      assert!(!verifier.verify(&altered, &to_raw_ecdsa(&ours)).unwrap_or(false));
      assert!(public.verify(&altered, &theirs).is_err());
    }
  }
}

/// The ML-DSA-65 seed and the P-256 scalar of a composite secret key.
fn components(secret: &MlDsa65EcdsaP256SecretKey) -> ([u8; 32], [u8; 32]) {
  let raw = secret.expose_secret();
  let raw = raw.as_bytes();
  assert_eq!(&raw[32..39], &[0x30, 0x31, 0x02, 0x01, 0x01, 0x04, 0x20]);
  (
    raw[..32].try_into().expect("seed"),
    raw[39..71].try_into().expect("scalar"),
  )
}

#[test]
fn external_signing_matches_one_shot_and_kryptering() {
  use p256::ecdsa::SigningKey as P256OracleSigningKey;
  use p256::ecdsa::signature::hazmat::PrehashSigner as _;
  use rustcrypto_ml_dsa::{ExpandedSigningKey, MlDsa65 as OracleMlDsa65};

  let variant = CompositeMlDsaVariant::MlDsa65Es256;
  let algorithm = SignatureAlgorithm::CompositeMlDsa(variant);
  let label = MlDsa65EcdsaP256::LABEL;
  let ml_dsa_context = [0, u8::try_from(label.len()).expect("label length")];
  for round in 0..3 {
    let (public, secret) = MlDsa65EcdsaP256::generate_keypair(filler(round)).expect("generate");
    let (seed, scalar) = components(&secret);
    let (ml_dsa_public, _) = MlDsa65::keypair_from_seed(&seed).expect("ML-DSA-65 key");
    let ecdsa = EcdsaP256SecretKey::from_bytes(scalar).expect("P-256 key");
    assert_eq!(
      MlDsa65EcdsaP256PublicKey::from_components(&ml_dsa_public, &ecdsa.public_key()),
      public
    );
    // RustCrypto ML-DSA and P-256 stand in for external signers, such as
    // Windows CNG or a TPM.
    let ml_dsa = ExpandedSigningKey::<OracleMlDsa65>::from_seed(&seed.into());
    let tpm = P256OracleSigningKey::from_slice(&scalar).expect("oracle key");
    let oracle_public = kryptering_public(&public);
    let key = SoftwareKey::from_composite_ml_dsa(variant, None, &oracle_public).expect("oracle key");
    let verifier = SoftwareVerifier::new(algorithm, key).expect("oracle verifier");

    for message in messages() {
      let pending = expert::MlDsa65EcdsaP256PendingSignature::new(&message, b"external").expect("begin");
      let half = ml_dsa
        .sign_deterministic(pending.representative(), label)
        .expect("ML-DSA-65 half");
      let ml_dsa_half = MlDsa65Signature::try_from_slice(half.encode().as_slice()).expect("ML-DSA-65 half parses");
      let ecdsa_half = ecdsa.try_sign(pending.representative()).expect("P-256 half");
      assert_eq!(
        pending.finish(&public, &ml_dsa_half, &ecdsa_half).expect("finish"),
        secret.sign_deterministic(&message, b"external").expect("one-shot"),
        "external and one-shot deterministic signing agree"
      );

      // A caller-computed prehash, a hedged ML-DSA-65 half, and a P-256 half
      // over the SHA-256 digest of M'.
      let pending =
        expert::MlDsa65EcdsaP256PendingSignature::from_prehash(&Sha512::digest(&message), b"").expect("begin");
      let random = [round ^ 0xa5; 32];
      let half = ml_dsa.sign_internal(&[&ml_dsa_context, label, pending.representative()], &random.into());
      let ml_dsa_half = MlDsa65Signature::try_from_slice(half.encode().as_slice()).expect("ML-DSA-65 half parses");
      let half: p256::ecdsa::Signature = tpm.sign_prehash(&pending.ecdsa_digest()).expect("digest half");
      let ecdsa_half = EcdsaP256Signature::from_der(half.to_der().as_bytes()).expect("digest half parses");
      let signature = pending.finish(&public, &ml_dsa_half, &ecdsa_half).expect("finish");
      assert!(
        verifier
          .verify(&message, &to_raw_ecdsa(&signature))
          .expect("oracle verification"),
        "the oracle accepts a signature from two external halves"
      );
    }
  }
}
