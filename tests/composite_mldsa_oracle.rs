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
