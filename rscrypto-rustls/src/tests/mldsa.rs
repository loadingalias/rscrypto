use alloc::sync::Arc;
use rustls::pki_types::{PrivateKeyDer, SignatureVerificationAlgorithm, alg_id};
use rustls::sign::CertifiedKey;
use rustls::{Error, SignatureAlgorithm, SignatureScheme};

use crate::{sign, verify};

const MESSAGE: &[u8] = include_bytes!("../../tests/vectors/credentials/message.txt");

struct Fixture {
  scheme: SignatureScheme,
  key_len: usize,
  signature_len: usize,
  key: &'static [u8],
  seed: &'static [u8],
  spki: &'static [u8],
  leaf: &'static [u8],
  signature: &'static [u8],
  context_signature: &'static [u8],
}

fn fixtures() -> [Fixture; 3] {
  macro_rules! fixture {
    ($name:literal, $scheme:ident, $key_len:literal, $signature_len:literal) => {
      Fixture {
        scheme: SignatureScheme::$scheme,
        key_len: $key_len,
        signature_len: $signature_len,
        key: include_bytes!(concat!("../../tests/vectors/credentials/", $name, "-leaf-key.der")),
        seed: include_bytes!(concat!("../../tests/vectors/credentials/", $name, "-leaf-seed-key.der")),
        spki: include_bytes!(concat!("../../tests/vectors/credentials/", $name, "-spki.der")),
        leaf: include_bytes!(concat!("../../tests/vectors/credentials/", $name, "-leaf.der")),
        signature: include_bytes!(concat!("../../tests/vectors/credentials/", $name, ".sig")),
        context_signature: include_bytes!(concat!("../../tests/vectors/credentials/", $name, "-context.sig")),
      }
    };
  }
  [
    fixture!("mldsa44", ML_DSA_44, 1312, 2420),
    fixture!("mldsa65", ML_DSA_65, 1952, 3309),
    fixture!("mldsa87", ML_DSA_87, 2592, 4627),
  ]
}

fn algorithm(scheme: SignatureScheme) -> &'static dyn SignatureVerificationAlgorithm {
  let algorithms = verify::SUPPORTED_ALGORITHMS
    .mapping
    .iter()
    .find(|(name, _)| *name == scheme)
    .expect("ML-DSA scheme")
    .1;
  assert_eq!(algorithms.len(), 1, "one exact parameter set per TLS scheme");
  algorithms[0]
}

#[test]
fn mldsa_verifiers_accept_openssl_and_reject_context_length_and_corruption() {
  for (fixture, identifier) in fixtures()
    .iter()
    .zip([alg_id::ML_DSA_44, alg_id::ML_DSA_65, alg_id::ML_DSA_87])
  {
    let ours = algorithm(fixture.scheme);
    assert_eq!(ours.public_key_alg_id(), identifier);
    assert_eq!(ours.signature_alg_id(), identifier);
    assert_eq!(fixture.signature.len(), fixture.signature_len);
    assert_eq!(fixture.spki.len(), 22 + fixture.key_len);
    let key = &fixture.spki[22..];
    ours
      .verify_signature(key, MESSAGE, fixture.signature)
      .expect("independent OpenSSL signature");
    let mut corrupt = fixture.signature.to_vec();
    corrupt[0] ^= 1;
    let mut trailing = fixture.signature.to_vec();
    trailing.push(0);
    let mut long_key = key.to_vec();
    long_key.push(0);
    for (key, message, signature) in [
      (key, &b"wrong message"[..], fixture.signature),
      (&key[..key.len().strict_sub(1)], MESSAGE, fixture.signature),
      (&long_key, MESSAGE, fixture.signature),
      (key, MESSAGE, &corrupt),
      (key, MESSAGE, &trailing),
      (key, MESSAGE, &fixture.signature[..fixture.signature_len.strict_sub(1)]),
      (key, MESSAGE, fixture.context_signature),
    ] {
      assert!(ours.verify_signature(key, message, signature).is_err());
    }
  }
}

#[test]
fn mldsa_pkcs8_schemes_spki_and_signatures_match_independent_peer() {
  let peer = rustls::crypto::aws_lc_rs::default_provider();
  for fixture in fixtures() {
    for der in [fixture.key, fixture.seed] {
      let key = sign::KEY_PROVIDER
        .load_private_key(PrivateKeyDer::try_from(der.to_vec()).expect("DER"))
        .expect("ML-DSA key");
      assert_eq!(key.algorithm(), SignatureAlgorithm::Unknown(0));
      assert_eq!(key.public_key().expect("SPKI").as_ref(), fixture.spki);
      let certified = CertifiedKey::new(vec![fixture.leaf.to_vec().into()], Arc::clone(&key));
      certified.keys_match().expect("certificate matches imported public key");
      let signer = key
        .choose_scheme(&[SignatureScheme::ED25519, fixture.scheme])
        .expect("exact scheme");
      assert_eq!(signer.scheme(), fixture.scheme);
      let signature = signer.sign(MESSAGE).expect("hedged TLS signing");
      assert_eq!(signature.len(), fixture.signature_len);
      let independent = peer
        .signature_verification_algorithms
        .mapping
        .iter()
        .find(|(scheme, _)| *scheme == fixture.scheme)
        .expect("independent ML-DSA")
        .1[0];
      independent
        .verify_signature(&fixture.spki[22..], MESSAGE, &signature)
        .expect("independent signature verification");
      for other in [
        SignatureScheme::ML_DSA_44,
        SignatureScheme::ML_DSA_65,
        SignatureScheme::ML_DSA_87,
        SignatureScheme::ED25519,
      ] {
        assert_eq!(key.choose_scheme(&[other]).is_some(), other == fixture.scheme);
      }
    }
    let mut trailing = fixture.key.to_vec();
    trailing.push(0);
    for malformed in [
      &fixture.key[..fixture.key.len().strict_sub(1)],
      &trailing,
      &[],
      &[0; 16_385],
    ] {
      let key = PrivateKeyDer::Pkcs8(malformed.to_vec().into());
      assert!(
        matches!(sign::KEY_PROVIDER.load_private_key(key), Err(Error::General(message))
        if message == "invalid or unsupported TLS private key")
      );
    }
  }
}

#[test]
fn mldsa_tls_rejects_hashmldsa_domain() {
  macro_rules! check {
    ($secret:ident, $name:literal, $scheme:ident) => {
      let key = rscrypto::$secret::from_pkcs8_der(include_bytes!(concat!(
        "../../tests/vectors/credentials/",
        $name,
        "-leaf-key.der"
      )))
      .expect("fixture key");
      let prehash =
        rscrypto::MlDsaPrehash::new(rscrypto::MlDsaPrehashAlgorithm::Sha512, &[9; 64]).expect("prehash length");
      let signature = key
        .sign_prehash_deterministic(prehash, b"")
        .expect("HashML-DSA signature");
      assert!(
        algorithm(SignatureScheme::$scheme)
          .verify_signature(key.public_key().as_bytes(), MESSAGE, signature.as_bytes())
          .is_err()
      );
    };
  }
  check!(MlDsa44SecretKey, "mldsa44", ML_DSA_44);
  check!(MlDsa65SecretKey, "mldsa65", ML_DSA_65);
  check!(MlDsa87SecretKey, "mldsa87", ML_DSA_87);
}

#[test]
fn mldsa_import_and_sign_execute_on_a_256_kib_worker_stack() {
  std::thread::Builder::new()
    .name("provider-mldsa-resource".into())
    .stack_size(256 * 1024)
    .spawn(|| {
      for fixture in fixtures() {
        for der in [fixture.key, fixture.seed] {
          let key = sign::KEY_PROVIDER
            .load_private_key(PrivateKeyDer::try_from(der.to_vec()).expect("DER"))
            .expect("key import");
          let signer = key.choose_scheme(&[fixture.scheme]).expect("TLS scheme");
          for _ in 0..32 {
            let signature = signer.sign(MESSAGE).expect("production signing on bounded stack");
            algorithm(fixture.scheme)
              .verify_signature(&fixture.spki[22..], MESSAGE, &signature)
              .expect("signature");
          }
        }
      }
    })
    .expect("resource worker")
    .join()
    .expect("resource worker completed");
}
