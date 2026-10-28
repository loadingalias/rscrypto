use alloc::sync::Arc;
use rustls::Error;
use rustls::crypto::SignatureScheme;
use rustls::pki_types::{
  CertificateDer, InvalidSignature, PrivateKeyDer, PrivatePkcs1KeyDer, PrivatePkcs8KeyDer, PrivateSec1KeyDer, alg_id,
};

use super::tls::{self, Credential};
use crate::{sign, verify};

macro_rules! fixture {
  ($file:literal) => {
    &include_bytes!(concat!("../../tests/vectors/credentials/", $file))[..]
  };
}

const MESSAGE: &[u8] = include_bytes!("../../tests/vectors/credentials/message.txt");

#[test]
fn selected_entropy_controls_key_loading_and_signing_without_fallback() {
  use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
  use rustls::crypto::{GetRandomFailed, KeyProvider, SecureRandom};

  #[derive(Debug)]
  struct Source {
    failing: AtomicBool,
    calls: AtomicUsize,
  }
  impl SecureRandom for Source {
    fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
      self.calls.fetch_add(1, Ordering::Relaxed);
      if self.failing.load(Ordering::Relaxed) {
        let length = output.len().min(17);
        output[..length].fill(0x42);
        Err(GetRandomFailed)
      } else {
        crate::tests::RANDOM.fill(output)
      }
    }
  }
  static SOURCE: Source = Source {
    failing: AtomicBool::new(false),
    calls: AtomicUsize::new(0),
  };
  let provider = sign::Provider::new(&SOURCE);
  let peer = rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER.signature_verification_algorithms;
  for (kind, public) in [
    (Credential::P256, public_bits(fixture!("p256-spki.der"), 65)),
    (Credential::P384, public_bits(fixture!("p384-spki.der"), 97)),
    (Credential::Ed25519, public_bits(fixture!("ed25519-spki.der"), 32)),
    (Credential::Rsa, fixture!("rsa3072-public.der")),
    (Credential::Rsa2048, fixture!("rsa2048-public.der")),
    #[cfg(feature = "ml-dsa")]
    (Credential::MlDsa44, &fixture!("mldsa44-spki.der")[22..]),
    #[cfg(feature = "ml-dsa")]
    (Credential::MlDsa65, &fixture!("mldsa65-spki.der")[22..]),
    #[cfg(feature = "ml-dsa")]
    (Credential::MlDsa87, &fixture!("mldsa87-spki.der")[22..]),
  ] {
    let (_, _, der) = tls::credentials(kind);
    SOURCE.failing.store(true, Ordering::Relaxed);
    let calls = SOURCE.calls.load(Ordering::Relaxed);
    let loaded = provider.load_private_key(PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(der.to_vec())));
    if matches!(kind, Credential::P256 | Credential::P384) {
      assert!(matches!(loaded, Err(Error::FailedToGetRandomBytes)));
      assert_eq!(SOURCE.calls.load(Ordering::Relaxed), calls.strict_add(1));
    } else {
      assert!(loaded.is_ok(), "this key import does not require entropy");
      assert_eq!(SOURCE.calls.load(Ordering::Relaxed), calls);
    }
    SOURCE.failing.store(false, Ordering::Relaxed);
    let key = provider
      .load_private_key(PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(der.to_vec())))
      .expect("key import after entropy recovery");
    let signer = key.choose_scheme(&[kind.scheme()]).expect("fixture scheme");
    SOURCE.failing.store(true, Ordering::Relaxed);
    let calls = SOURCE.calls.load(Ordering::Relaxed);
    let signature = signer.sign(MESSAGE);
    if matches!(kind, Credential::Ed25519) {
      assert_eq!(SOURCE.calls.load(Ordering::Relaxed), calls);
      peer
        .mapping()
        .iter()
        .find(|(scheme, _)| *scheme == kind.scheme())
        .expect("peer scheme")
        .1[0]
        .verify_signature(public, MESSAGE, &signature.expect("Ed25519 needs no entropy"))
        .expect("independent Ed25519 verification");
    } else {
      assert!(matches!(signature, Err(Error::General(message)) if message == "TLS signing failed"));
      assert_eq!(SOURCE.calls.load(Ordering::Relaxed), calls.strict_add(1));
    }
    SOURCE.failing.store(false, Ordering::Relaxed);
    let signature = key
      .choose_scheme(&[kind.scheme()])
      .expect("scheme after recovery")
      .sign(MESSAGE)
      .expect("same key after entropy recovery");
    peer
      .mapping()
      .iter()
      .find(|(scheme, _)| *scheme == kind.scheme())
      .expect("peer scheme")
      .1[0]
      .verify_signature(public, MESSAGE, &signature)
      .expect("independent verification after entropy failure");
  }
}

fn public_bits(spki: &[u8], length: usize) -> &[u8] {
  // These pinned SPKIs end in a fixed-size, byte-aligned public-key BIT STRING.
  // End-to-end certificate tests separately exercise WebPKI's DER interpretation.
  &spki[spki.len().strict_sub(length)..]
}

#[test]
fn all_fourteen_verifiers_accept_independent_signatures_and_reject_corruption() {
  let p256 = public_bits(fixture!("p256-spki.der"), 65);
  let p384 = public_bits(fixture!("p384-spki.der"), 97);
  let ed25519 = public_bits(fixture!("ed25519-spki.der"), 32);
  let rsa = fixture!("rsa3072-public.der");
  let cases = [
    (
      alg_id::ECDSA_P256,
      alg_id::ECDSA_SHA256,
      p256,
      fixture!("p256-sha256.sig"),
    ),
    (
      alg_id::ECDSA_P256,
      alg_id::ECDSA_SHA384,
      p256,
      fixture!("p256-sha384.sig"),
    ),
    (
      alg_id::ECDSA_P384,
      alg_id::ECDSA_SHA256,
      p384,
      fixture!("p384-sha256.sig"),
    ),
    (
      alg_id::ECDSA_P384,
      alg_id::ECDSA_SHA384,
      p384,
      fixture!("p384-sha384.sig"),
    ),
    (alg_id::ED25519, alg_id::ED25519, ed25519, fixture!("ed25519.sig")),
    (
      alg_id::RSA_ENCRYPTION,
      alg_id::RSA_PSS_SHA256,
      rsa,
      fixture!("rsa3072-pss-sha256.sig"),
    ),
    (
      alg_id::RSA_ENCRYPTION,
      alg_id::RSA_PSS_SHA384,
      rsa,
      fixture!("rsa3072-pss-sha384.sig"),
    ),
    (
      alg_id::RSA_ENCRYPTION,
      alg_id::RSA_PSS_SHA512,
      rsa,
      fixture!("rsa3072-pss-sha512.sig"),
    ),
    (
      alg_id::RSA_ENCRYPTION,
      alg_id::RSA_PKCS1_SHA256,
      rsa,
      fixture!("rsa3072-pkcs1-sha256.sig"),
    ),
    (
      alg_id::RSA_ENCRYPTION,
      alg_id::RSA_PKCS1_SHA384,
      rsa,
      fixture!("rsa3072-pkcs1-sha384.sig"),
    ),
    (
      alg_id::RSA_ENCRYPTION,
      alg_id::RSA_PKCS1_SHA512,
      rsa,
      fixture!("rsa3072-pkcs1-sha512.sig"),
    ),
  ];
  assert_eq!(
    verify::ALL_ALGORITHMS.len(),
    14 + if cfg!(feature = "ml-dsa") { 3 } else { 0 }
  );
  let mut exercised = 0_usize;
  for algorithm in verify::ALL_ALGORITHMS {
    if [alg_id::ML_DSA_44, alg_id::ML_DSA_65, alg_id::ML_DSA_87].contains(&algorithm.public_key_alg_id()) {
      continue; // The independent ML-DSA fixtures are exercised in tests::mldsa.
    }
    let signature_algorithm = algorithm.signature_alg_id();
    let signature_id = signature_algorithm.as_ref();
    let (_, _, key, signature) = cases
      .iter()
      .find(|(public, signature, _, _)| {
        *public == algorithm.public_key_alg_id()
          && (signature.as_ref() == signature_id
            || (signature.as_ref().ends_with(&[5, 0])
              && signature.as_ref().strip_suffix(&[5, 0]) == Some(signature_id)))
      })
      .expect("independent fixture for each algorithm identifier");
    let independent = rustls_ring::ALL_VERIFICATION_ALGS
      .iter()
      .find(|peer| {
        peer.public_key_alg_id() == algorithm.public_key_alg_id()
          && peer.signature_alg_id() == algorithm.signature_alg_id()
      })
      .expect("independent algorithm identifier");
    algorithm
      .verify_signature(key, MESSAGE, signature)
      .expect("OpenSSL signature");
    independent
      .verify_signature(key, MESSAGE, signature)
      .expect("ring accepts the same fixture");
    let mut corrupt = signature.to_vec();
    corrupt[0] ^= 1;
    let mut trailing = signature.to_vec();
    trailing.push(0);
    for (key, message, signature) in [
      (*key, &b"changed message"[..], *signature),
      (&key[..key.len().strict_sub(1)], MESSAGE, *signature),
      (*key, MESSAGE, &corrupt),
      (*key, MESSAGE, &trailing),
      (*key, MESSAGE, &signature[..signature.len().strict_sub(1)]),
    ] {
      assert!(matches!(
        algorithm.verify_signature(key, message, signature),
        Err(InvalidSignature)
      ));
    }
    exercised = exercised.strict_add(1);
  }
  assert_eq!(exercised, 14);
  for (scheme, public) in [
    (SignatureScheme::ECDSA_NISTP256_SHA256, alg_id::ECDSA_P256),
    (SignatureScheme::ECDSA_NISTP384_SHA384, alg_id::ECDSA_P384),
  ] {
    let mapped = verify::SUPPORTED_ALGORITHMS
      .mapping()
      .iter()
      .find(|(name, _)| *name == scheme)
      .expect("TLS scheme")
      .1;
    // TLS 1.3 verifies with the first algorithm only, so it never substitutes
    // another curve. TLS 1.2 binds no curve and also accepts the other one
    // with the same hash.
    assert_eq!(mapped[0].public_key_alg_id(), public);
    assert_eq!(mapped.len(), 2);
    assert_ne!(mapped[1].public_key_alg_id(), public);
    assert_eq!(mapped[1].signature_alg_id(), mapped[0].signature_alg_id());
  }
}

#[test]
fn loaded_signing_keys_match_spki_and_independent_verifiers() {
  let peer = rustls_ring::DEFAULT_TLS13_PROVIDER.signature_verification_algorithms;
  for (kind, scheme, spki, public) in [
    (
      Credential::P256,
      SignatureScheme::ECDSA_NISTP256_SHA256,
      fixture!("p256-spki.der"),
      public_bits(fixture!("p256-spki.der"), 65),
    ),
    (
      Credential::P384,
      SignatureScheme::ECDSA_NISTP384_SHA384,
      fixture!("p384-spki.der"),
      public_bits(fixture!("p384-spki.der"), 97),
    ),
    (
      Credential::Ed25519,
      SignatureScheme::ED25519,
      fixture!("ed25519-spki.der"),
      public_bits(fixture!("ed25519-spki.der"), 32),
    ),
    (
      Credential::Rsa,
      SignatureScheme::RSA_PSS_SHA256,
      fixture!("rsa3072-spki.der"),
      fixture!("rsa3072-public.der"),
    ),
    (
      Credential::Rsa2048,
      SignatureScheme::RSA_PSS_SHA256,
      fixture!("rsa2048-spki.der"),
      fixture!("rsa2048-public.der"),
    ),
  ] {
    let (_, cert, private) = tls::credentials(kind);
    let key = crate::tests::key_provider()
      .load_private_key(PrivatePkcs8KeyDer::from(private.to_vec()).into())
      .expect("valid fixture key");
    assert_eq!(key.public_key().expect("SPKI").as_ref(), spki);
    let certified = rustls::crypto::Credentials::new(
      Arc::new(rustls::crypto::Identity::from_cert_chain(vec![CertificateDer::from(cert.to_vec())]).expect("identity")),
      key,
    )
    .expect("certificate and key match");
    assert!(certified.key.choose_scheme(&[]).is_none());
    // Only RSA keys sign PKCS#1 v1.5, for TLS 1.2 peers that offer nothing else.
    assert_eq!(
      certified
        .key
        .choose_scheme(&[SignatureScheme::RSA_PKCS1_SHA256])
        .map(|signer| signer.scheme()),
      matches!(kind, Credential::Rsa | Credential::Rsa2048).then_some(SignatureScheme::RSA_PKCS1_SHA256)
    );
    let schemes = if matches!(kind, Credential::Rsa | Credential::Rsa2048) {
      &[
        SignatureScheme::RSA_PSS_SHA256,
        SignatureScheme::RSA_PSS_SHA384,
        SignatureScheme::RSA_PSS_SHA512,
      ][..]
    } else {
      core::slice::from_ref(&scheme)
    };
    for &scheme in schemes {
      let signer = certified.key.choose_scheme(&[scheme]).expect("supported scheme");
      assert_eq!(signer.scheme(), scheme);
      let signature = signer.sign(MESSAGE).expect("sign");
      let verifier = peer
        .mapping()
        .iter()
        .find(|(name, _)| *name == scheme)
        .expect("independent scheme")
        .1[0];
      verifier
        .verify_signature(public, MESSAGE, &signature)
        .expect("ring verifies provider signature");
      assert!(matches!(
        verifier.verify_signature(public, b"changed", &signature),
        Err(InvalidSignature)
      ));
      if matches!(kind, Credential::Ed25519) {
        assert_eq!(signature, fixture!("ed25519.sig"));
      }
      if matches!(kind, Credential::Rsa | Credential::Rsa2048) {
        assert_ne!(
          signature,
          certified
            .key
            .choose_scheme(&[scheme])
            .expect("scheme")
            .sign(MESSAGE)
            .expect("fresh PSS salt")
        );
      }
    }
    let (_, wrong_cert, _) = tls::credentials(if matches!(kind, Credential::P256) {
      Credential::P384
    } else {
      Credential::P256
    });
    let mismatched = rustls::crypto::Credentials::new(
      Arc::new(
        rustls::crypto::Identity::from_cert_chain(vec![CertificateDer::from(wrong_cert.to_vec())]).expect("identity"),
      ),
      certified.key,
    );
    assert!(matches!(mismatched, Err(Error::InconsistentKeys(_))));
  }
}

#[test]
fn alternate_private_key_encodings_and_invalid_containers() {
  for (private, expected) in [
    (
      PrivateKeyDer::Sec1(PrivateSec1KeyDer::from(fixture!("p256-sec1.der").to_vec())),
      fixture!("p256-spki.der"),
    ),
    (
      PrivateKeyDer::Sec1(PrivateSec1KeyDer::from(fixture!("p384-sec1.der").to_vec())),
      fixture!("p384-spki.der"),
    ),
    (
      PrivateKeyDer::Pkcs1(PrivatePkcs1KeyDer::from(fixture!("rsa3072-pkcs1.der").to_vec())),
      fixture!("rsa3072-spki.der"),
    ),
    (
      PrivateKeyDer::Pkcs1(PrivatePkcs1KeyDer::from(fixture!("rsa2048-pkcs1.der").to_vec())),
      fixture!("rsa2048-spki.der"),
    ),
  ] {
    let key = crate::tests::key_provider()
      .load_private_key(private)
      .expect("alternate encoding");
    assert_eq!(key.public_key().expect("SPKI").as_ref(), expected);
  }
  for kind in [
    Credential::P256,
    Credential::P384,
    Credential::Ed25519,
    Credential::Rsa,
    Credential::Rsa2048,
  ] {
    let private = tls::credentials(kind).2;
    let mut trailing = private.to_vec();
    trailing.push(0);
    for malformed in [&private[..private.len().strict_sub(1)], &trailing, &[], &[0; 16_385]] {
      let error = crate::tests::key_provider()
        .load_private_key(PrivatePkcs8KeyDer::from(malformed.to_vec()).into())
        .expect_err("reject malformed key");
      assert!(matches!(error, Error::General(message) if message == "invalid or unsupported TLS private key"));
    }
  }
}

#[test]
fn rsa_signing_keys_load_from_2048_bits_unless_the_policy_requires_3072() {
  use rustls::crypto::KeyProvider as _;
  use sign::RsaSigningKeySizes;

  let rsa2048 = || {
    PrivateKeyDer::from(PrivatePkcs8KeyDer::from(
      tls::credentials(Credential::Rsa2048).2.to_vec(),
    ))
  };
  let rsa3072 = || PrivateKeyDer::from(PrivatePkcs8KeyDer::from(tls::credentials(Credential::Rsa).2.to_vec()));
  let key = crate::tests::key_provider()
    .load_private_key(rsa2048())
    .expect("default loads RSA-2048");
  // PKCS#1 v1.5 is deterministic, so OpenSSL's signatures are known answers.
  for (scheme, expected) in [
    (SignatureScheme::RSA_PKCS1_SHA256, fixture!("rsa2048-pkcs1-sha256.sig")),
    (SignatureScheme::RSA_PKCS1_SHA384, fixture!("rsa2048-pkcs1-sha384.sig")),
    (SignatureScheme::RSA_PKCS1_SHA512, fixture!("rsa2048-pkcs1-sha512.sig")),
  ] {
    let signature = key
      .choose_scheme(&[scheme])
      .expect("scheme")
      .sign(MESSAGE)
      .expect("sign");
    assert_eq!(signature, expected, "{scheme:?}");
  }
  for (scheme, signature) in [
    (SignatureScheme::RSA_PSS_SHA256, fixture!("rsa2048-pss-sha256.sig")),
    (SignatureScheme::RSA_PSS_SHA384, fixture!("rsa2048-pss-sha384.sig")),
    (SignatureScheme::RSA_PSS_SHA512, fixture!("rsa2048-pss-sha512.sig")),
  ] {
    let verifier = verify::SUPPORTED_ALGORITHMS
      .mapping()
      .iter()
      .find(|(candidate, _)| *candidate == scheme)
      .expect("PSS verifier")
      .1[0];
    verifier
      .verify_signature(fixture!("rsa2048-public.der"), MESSAGE, signature)
      .expect("OpenSSL RSA-2048 signature");
  }
  let strict = sign::Provider::new(&crate::tests::RANDOM).rsa_key_sizes(RsaSigningKeySizes::Bits3072To8192);
  assert!(matches!(
    strict.load_private_key(rsa2048()),
    Err(Error::General(message)) if message == "invalid or unsupported TLS private key"
  ));
  strict.load_private_key(rsa3072()).expect("strict loads RSA-3072");
  sign::Provider::new(&crate::tests::RANDOM)
    .rsa_key_sizes(RsaSigningKeySizes::Bits2048To8192)
    .load_private_key(rsa2048())
    .expect("explicit default loads RSA-2048");
}

#[test]
fn authenticated_handshakes_use_each_credential_in_both_roles() {
  for kind in [
    Credential::P256,
    Credential::P384,
    Credential::Ed25519,
    Credential::Rsa,
    Credential::Rsa2048,
  ] {
    for independent in [
      (|| rustls_ring::DEFAULT_TLS13_PROVIDER),
      (|| rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER),
    ] {
      for (client, server) in [(tls::provider(), independent()), (independent(), tls::provider())] {
        let (client, server) = tls::configs(client, server, kind, false);
        let (mut client, mut server) = tls::connections_from_configs(Arc::new(client), Arc::new(server));
        tls::drive(&mut client, &mut server).expect("authenticated handshake");
        assert!(!client.connection.is_handshaking() && !server.connection.is_handshaking());
        tls::application_exchange(&mut client, &mut server);
      }
    }
  }
}

#[cfg(feature = "getrandom")]
#[test]
fn required_pq_refuses_classical_peers_in_both_roles() {
  for (client, server) in [
    (
      crate::post_quantum_required_provider(),
      rustls_ring::DEFAULT_TLS13_PROVIDER,
    ),
    (
      rustls_ring::DEFAULT_TLS13_PROVIDER,
      crate::post_quantum_required_provider(),
    ),
  ] {
    let (client, server) = tls::configs(client, server, Credential::P256, false);
    let (mut client, mut server) = tls::connections_from_configs(Arc::new(client), Arc::new(server));
    assert!(matches!(
      tls::drive(&mut client, &mut server),
      Err(Error::PeerIncompatible(_))
    ));
    assert!(client.connection.is_handshaking() && server.connection.is_handshaking());
  }
}
