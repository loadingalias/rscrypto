//! Composite ML-DSA TLS signature schemes (draft-ietf-tls-composite-mldsa-00).
//!
//! The draft's Appendix E certificates come from the LAMPS reference
//! implementation and authenticate a client through webpki. The handshake
//! fixtures are built here deterministically: fixed key seeds and
//! deterministic composite signatures over minimal DER certificates.
//! `credential_fixtures_are_reproducible` regenerates them and requires the
//! embedded checked-in files to match; run it with
//! `RSCRYPTO_WRITE_COMPOSITE_CREDENTIALS=1` from the source tree to rewrite them.

use alloc::sync::Arc;
use alloc::vec::Vec;

use rscrypto::{
  CompositeMlDsaError, MlDsa65EcdsaP256, MlDsa65EcdsaP256SecretKey, MlDsa65Ed25519, MlDsa65Ed25519SecretKey,
};
use rustls::crypto::{CipherSuite, Identity, KeyProvider, SignatureScheme, SigningKey};
use rustls::error::{AlertDescription, PeerMisbehaved};
use rustls::pki_types::{
  AlgorithmIdentifier, CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, SignatureVerificationAlgorithm,
};
use rustls::{ClientConfig, Error, RootCertStore, ServerConfig};

use super::tls::{Credential, FixtureTime, configs, connections_from_configs, drive, provider, transfer};
use crate::verify;

/// DER contents of 1.3.6.1.5.5.7.6.`arc`.
fn oid(arc: u8) -> [u8; 8] {
  [0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x06, arc]
}

fn tlv(tag: u8, contents: &[u8]) -> Vec<u8> {
  let mut out = vec![tag];
  let len = contents.len();
  if len < 0x80 {
    out.push(u8::try_from(len).expect("short length"));
  } else if len < 0x100 {
    out.extend([0x81, u8::try_from(len).expect("one-octet length")]);
  } else {
    out.push(0x82);
    out.extend(u16::try_from(len).expect("two-octet length").to_be_bytes());
  }
  out.extend_from_slice(contents);
  out
}

/// One DER element: its contents and the input after it.
fn element(der: &[u8]) -> (&[u8], &[u8]) {
  let (len, header) = if der[1] < 0x80 {
    (usize::from(der[1]), 2usize)
  } else if der[1] == 0x81 {
    (usize::from(der[2]), 3)
  } else {
    assert_eq!(der[1], 0x82, "a DER length of at most two octets");
    (usize::from(u16::from_be_bytes([der[2], der[3]])), 4)
  };
  let end = header.strict_add(len);
  (&der[header..end], &der[end..])
}

/// A composite algorithm of this provider: its kind, its signature scheme,
/// its OID arc, and the draft test case.
#[derive(Clone, Copy)]
enum Kind {
  Ed25519,
  EcdsaP256,
}

impl Kind {
  const ALL: [Self; 2] = [Self::Ed25519, Self::EcdsaP256];

  fn name(self) -> &'static str {
    match self {
      Self::Ed25519 => "mldsa65-ed25519",
      Self::EcdsaP256 => "mldsa65-ecdsa-p256",
    }
  }

  fn arc(self) -> u8 {
    match self {
      Self::Ed25519 => 48,
      Self::EcdsaP256 => 45,
    }
  }

  fn scheme(self) -> SignatureScheme {
    match self {
      Self::Ed25519 => verify::MLDSA65_ED25519_SHA512,
      Self::EcdsaP256 => verify::MLDSA65_ECDSA_SECP256R1_SHA512,
    }
  }

  fn credential(self) -> Credential {
    match self {
      Self::Ed25519 => Credential::MlDsa65Ed25519,
      Self::EcdsaP256 => Credential::MlDsa65EcdsaP256,
    }
  }

  /// The draft test case's self-signed certificate (`x5c`) and composite
  /// PKCS #8 key (`sk_pkcs8`).
  fn draft_case(self) -> (&'static [u8], &'static [u8]) {
    match self {
      Self::Ed25519 => (
        include_bytes!("../../tests/vectors/composite_mldsa/draft19/id-MLDSA65-Ed25519-SHA512-x5c.der"),
        include_bytes!("../../tests/vectors/composite_mldsa/draft19/id-MLDSA65-Ed25519-SHA512-sk_pkcs8.der"),
      ),
      Self::EcdsaP256 => (
        include_bytes!("../../tests/vectors/composite_mldsa/draft19/id-MLDSA65-ECDSA-P256-SHA512-x5c.der"),
        include_bytes!("../../tests/vectors/composite_mldsa/draft19/id-MLDSA65-ECDSA-P256-SHA512-sk_pkcs8.der"),
      ),
    }
  }

  fn algorithm_id(self) -> AlgorithmIdentifier {
    AlgorithmIdentifier::from_slice(Vec::leak(tlv(0x06, &oid(self.arc()))))
  }
}

/// A deterministic signer over certificate bytes.
type Signer = Box<dyn Fn(&[u8]) -> Vec<u8>>;

/// A deterministic composite key: its SubjectPublicKeyInfo, a PKCS #8
/// encoding, and a deterministic signer.
struct Key {
  spki: Vec<u8>,
  pkcs8: Vec<u8>,
  sign: Signer,
}

fn key(kind: Kind, seed: u8) -> Key {
  let fill = move |out: &mut [u8]| -> Result<(), CompositeMlDsaError> {
    for (index, byte) in out.iter_mut().enumerate() {
      *byte = seed ^ u8::try_from(index % 251).expect("byte");
    }
    Ok(())
  };
  match kind {
    Kind::Ed25519 => {
      let (public, secret) = MlDsa65Ed25519::generate_keypair(fill).expect("composite key");
      let mut pkcs8 = vec![0; MlDsa65Ed25519SecretKey::PKCS8_DER_LENGTH];
      secret.to_pkcs8_der_into(pkcs8.as_mut_slice().try_into().expect("PKCS #8 length"));
      Key {
        spki: public.to_spki_der().to_vec(),
        pkcs8,
        sign: Box::new(move |tbs| secret.sign_deterministic(tbs, b"").expect("sign").as_ref().to_vec()),
      }
    }
    Kind::EcdsaP256 => {
      let (public, secret) = MlDsa65EcdsaP256::generate_keypair(fill).expect("composite key");
      let mut pkcs8 = vec![0; MlDsa65EcdsaP256SecretKey::PKCS8_DER_LENGTH];
      secret.to_pkcs8_der_into(pkcs8.as_mut_slice().try_into().expect("PKCS #8 length"));
      Key {
        spki: public.to_spki_der().to_vec(),
        pkcs8,
        sign: Box::new(move |tbs| secret.sign_deterministic(tbs, b"").expect("sign").as_ref().to_vec()),
      }
    }
  }
}

fn name(common_name: &str) -> Vec<u8> {
  // Name ::= SEQUENCE OF RelativeDistinguishedName; CN = 2.5.4.3.
  let attribute = tlv(
    0x30,
    &[tlv(0x06, &[0x55, 0x04, 0x03]), tlv(0x0c, common_name.as_bytes())].concat(),
  );
  tlv(0x30, &tlv(0x31, &attribute))
}

fn extension(oid_contents: &[u8], critical: bool, value: &[u8]) -> Vec<u8> {
  let mut contents = tlv(0x06, oid_contents);
  if critical {
    contents.extend(tlv(0x01, &[0xff]));
  }
  contents.extend(tlv(0x04, value));
  tlv(0x30, &contents)
}

/// A minimal X.509 v3 certificate signed with the issuer's composite key.
fn certificate(kind: Kind, serial: u8, subject: &str, issuer: &str, key: &Key, signer: &Key, ca: bool) -> Vec<u8> {
  let algorithm = tlv(0x30, &tlv(0x06, &oid(kind.arc())));
  let validity = tlv(
    0x30,
    &[tlv(0x17, b"260101000000Z"), tlv(0x17, b"360101000000Z")].concat(),
  );
  let mut extensions = vec![
    // basicConstraints: cA for authorities, absent cA for leaves.
    extension(
      &[0x55, 0x1d, 0x13],
      true,
      &tlv(0x30, if ca { &[0x01, 0x01, 0xff] } else { &[] }),
    ),
    // keyUsage: keyCertSign and cRLSign with digitalSignature for
    // authorities, digitalSignature for leaves.
    extension(
      &[0x55, 0x1d, 0x0f],
      true,
      &if ca {
        tlv(0x03, &[0x01, 0x86])
      } else {
        tlv(0x03, &[0x07, 0x80])
      },
    ),
  ];
  if !ca {
    // subjectAltName dNSName localhost; extKeyUsage serverAuth and clientAuth.
    extensions.push(extension(
      &[0x55, 0x1d, 0x11],
      false,
      &tlv(0x30, &tlv(0x82, b"localhost")),
    ));
    extensions.push(extension(
      &[0x55, 0x1d, 0x25],
      false,
      &tlv(
        0x30,
        &[
          tlv(0x06, &[0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x03, 0x01]),
          tlv(0x06, &[0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x03, 0x02]),
        ]
        .concat(),
      ),
    ));
  }
  let tbs = tlv(
    0x30,
    &[
      tlv(0xa0, &tlv(0x02, &[2])),
      tlv(0x02, &[serial]),
      algorithm.clone(),
      name(issuer),
      validity,
      name(subject),
      key.spki.clone(),
      tlv(0xa3, &tlv(0x30, &extensions.concat())),
    ]
    .concat(),
  );
  let signature = (signer.sign)(&tbs);
  tlv(
    0x30,
    &[tbs, algorithm, tlv(0x03, &[&[0][..], &signature].concat())].concat(),
  )
}

/// The fixture files of one composite: root, leaf, leaf key, and a
/// root–intermediate–leaf chain.
fn credential_files(kind: Kind) -> Vec<(String, Vec<u8>)> {
  let base = kind.arc().strict_mul(4);
  let [root, leaf, intermediate, chain_leaf] = [0, 1, 2, 3].map(|offset| key(kind, base.strict_add(offset)));
  let prefix = kind.name();
  let root_name = format!("{prefix} root");
  let intermediate_name = format!("{prefix} intermediate");
  vec![
    (
      format!("{prefix}-root.der"),
      certificate(kind, 1, &root_name, &root_name, &root, &root, true),
    ),
    (
      format!("{prefix}-leaf.der"),
      certificate(kind, 2, &format!("{prefix} leaf"), &root_name, &leaf, &root, false),
    ),
    (format!("{prefix}-leaf-key.der"), leaf.pkcs8.clone()),
    (
      format!("{prefix}-chain-intermediate.der"),
      certificate(kind, 3, &intermediate_name, &root_name, &intermediate, &root, true),
    ),
    (
      format!("{prefix}-chain-leaf.der"),
      certificate(
        kind,
        4,
        &format!("{prefix} chain leaf"),
        &intermediate_name,
        &chain_leaf,
        &intermediate,
        false,
      ),
    ),
    (format!("{prefix}-chain-leaf-key.der"), chain_leaf.pkcs8.clone()),
  ]
}

/// The checked-in fixtures, embedded at compile time so the comparison does
/// not depend on where a cross-built test archive runs.
macro_rules! checked_in {
  ($($name:literal),* $(,)?) => {
    [$(($name, include_bytes!(concat!("../../tests/vectors/credentials/", $name)) as &[u8])),*]
  };
}
const CHECKED_IN: [(&str, &[u8]); 12] = checked_in!(
  "mldsa65-ed25519-root.der",
  "mldsa65-ed25519-leaf.der",
  "mldsa65-ed25519-leaf-key.der",
  "mldsa65-ed25519-chain-intermediate.der",
  "mldsa65-ed25519-chain-leaf.der",
  "mldsa65-ed25519-chain-leaf-key.der",
  "mldsa65-ecdsa-p256-root.der",
  "mldsa65-ecdsa-p256-leaf.der",
  "mldsa65-ecdsa-p256-leaf-key.der",
  "mldsa65-ecdsa-p256-chain-intermediate.der",
  "mldsa65-ecdsa-p256-chain-leaf.der",
  "mldsa65-ecdsa-p256-chain-leaf-key.der",
);

#[test]
fn credential_fixtures_are_reproducible() {
  let generated: Vec<_> = Kind::ALL.into_iter().flat_map(credential_files).collect();
  assert_eq!(
    generated.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>(),
    CHECKED_IN.map(|(name, _)| name).to_vec(),
    "the generator and the embedded fixture list name the same files"
  );
  // Rewriting needs the source tree, so it resolves the package directory at
  // run time rather than from the build host's compile-time path.
  if std::env::var_os("RSCRYPTO_WRITE_COMPOSITE_CREDENTIALS").is_some() {
    let package = std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR names the package to rewrite");
    let directory = std::path::Path::new(&package).join("tests/vectors/credentials");
    for (name, data) in &generated {
      std::fs::write(directory.join(name), data).expect("write fixture");
    }
    return;
  }
  for ((name, expected), (_, checked_in)) in generated.iter().zip(CHECKED_IN) {
    assert_eq!(checked_in, expected.as_slice(), "{name}");
  }
}

fn algorithm(kind: Kind) -> &'static dyn SignatureVerificationAlgorithm {
  let algorithms = verify::SUPPORTED_ALGORITHMS
    .mapping()
    .iter()
    .find(|(scheme, _)| *scheme == kind.scheme())
    .expect("composite scheme")
    .1;
  assert_eq!(algorithms.len(), 1, "one composite per TLS scheme");
  algorithms[0]
}

#[test]
fn draft_certificate_signatures_verify_through_the_provider() {
  for kind in Kind::ALL {
    let ours = algorithm(kind);
    assert_eq!(ours.public_key_alg_id(), kind.algorithm_id());
    assert_eq!(ours.signature_alg_id(), kind.algorithm_id());
    assert!(
      verify::ALL_ALGORITHMS
        .iter()
        .any(|candidate| candidate.public_key_alg_id() == kind.algorithm_id())
    );
    // Certificate ::= SEQUENCE { tbsCertificate, signatureAlgorithm, signature }.
    let (x5c, _) = kind.draft_case();
    let (certificate, _) = element(x5c);
    let (_, after_tbs) = element(certificate);
    let tbs = &certificate[..certificate.len().strict_sub(after_tbs.len())];
    let (_, after_algorithm) = element(after_tbs);
    let (bits, _) = element(after_algorithm);
    let signature = &bits[1..];
    // tbsCertificate fields: version, serial, signature, issuer, validity,
    // subject, then subjectPublicKeyInfo.
    let (mut fields, _) = element(tbs);
    for _ in 0..6 {
      fields = element(fields).1;
    }
    let (spki, _) = element(fields);
    let (_, key_bits) = element(spki);
    let (key, _) = element(key_bits);
    let key = &key[1..];
    ours
      .verify_signature(key, tbs, signature)
      .expect("draft certificate signature");
    let mut corrupt = signature.to_vec();
    corrupt[0] ^= 1;
    assert!(ours.verify_signature(key, tbs, &corrupt).is_err());
    assert!(ours.verify_signature(key, &tbs[1..], signature).is_err());
    assert!(ours.verify_signature(&key[1..], tbs, signature).is_err());
  }
}

#[test]
fn handshakes_authenticate_with_each_composite() {
  for kind in Kind::ALL {
    for client_auth in [false, true] {
      let (client, server) = configs(provider(), provider(), kind.credential(), client_auth);
      let (mut client, mut server) = connections_from_configs(Arc::new(client), Arc::new(server));
      drive(&mut client, &mut server).expect("composite handshake");
      super::tls::application_exchange(&mut client, &mut server);
      assert_eq!(
        server.connection.handshake_kind(),
        Some(rustls::HandshakeKind::Full),
        "{} client_auth={client_auth}",
        kind.name()
      );
    }
  }
}

fn chain_configs(kind: Kind, compact: bool) -> (ClientConfig, ServerConfig) {
  let files = credential_files(kind);
  let file = |suffix: &str| {
    files
      .iter()
      .find(|(name, _)| name.ends_with(suffix))
      .expect("fixture")
      .1
      .clone()
  };
  let mut roots = RootCertStore::empty();
  roots.add(CertificateDer::from(file("-root.der"))).expect("root");
  let mut server_provider = provider();
  if compact {
    static COMPACT: crate::sign::Provider = crate::sign::Provider::new(&super::RANDOM).compact();
    server_provider.key_provider = &COMPACT;
  }
  let client = ClientConfig::builder_with_details(Arc::new(provider()), Arc::new(FixtureTime))
    .with_root_certificates(roots)
    .with_no_client_auth()
    .expect("client");
  let identity = Identity::from_cert_chain(vec![
    CertificateDer::from(file("-chain-leaf.der")),
    CertificateDer::from(file("-chain-intermediate.der")),
  ])
  .expect("chain");
  let server = ServerConfig::builder_with_details(Arc::new(server_provider), Arc::new(FixtureTime))
    .with_no_client_auth()
    .with_single_cert(
      Arc::new(identity),
      PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(file("-chain-leaf-key.der"))),
    )
    .expect("server");
  (client, server)
}

#[test]
fn chains_verify_and_compact_keys_sign() {
  for kind in Kind::ALL {
    for compact in [false, true] {
      let (client, server) = chain_configs(kind, compact);
      let (mut client, mut server) = connections_from_configs(Arc::new(client), Arc::new(server));
      drive(&mut client, &mut server).expect("composite chain");
      super::tls::application_exchange(&mut client, &mut server);
    }
  }
}

#[test]
fn draft_certificate_authenticates_a_client() {
  // The LAMPS reference implementation signed this self-signed certificate.
  // It has no subjectAltName, so it serves as a client certificate, which
  // webpki checks without a name.
  for kind in Kind::ALL {
    let (x5c, pkcs8) = kind.draft_case();
    let (root, leaf, key) = super::tls::credentials(kind.credential());
    let mut client_roots = RootCertStore::empty();
    client_roots.add(CertificateDer::from(root.to_vec())).expect("root");
    let mut client_cert_roots = RootCertStore::empty();
    client_cert_roots
      .add(CertificateDer::from(x5c))
      .expect("draft certificate");
    let server_provider = Arc::new(provider());
    let verifier = rustls::server::WebPkiClientVerifier::builder(Arc::new(client_cert_roots), &server_provider)
      .build()
      .expect("client verifier");
    let client = ClientConfig::builder_with_details(Arc::new(provider()), Arc::new(FixtureTime))
      .with_root_certificates(client_roots)
      .with_client_auth_cert(
        Arc::new(Identity::from_cert_chain(vec![CertificateDer::from(x5c)]).expect("identity")),
        PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(pkcs8)),
      )
      .expect("client credential");
    let server = ServerConfig::builder_with_details(server_provider, Arc::new(FixtureTime))
      .with_client_cert_verifier(Arc::new(verifier))
      .with_single_cert(
        Arc::new(Identity::from_cert_chain(vec![CertificateDer::from(leaf.to_vec())]).expect("identity")),
        PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key.to_vec())),
      )
      .expect("server");
    let (mut client, mut server) = connections_from_configs(Arc::new(client), Arc::new(server));
    drive(&mut client, &mut server).expect("draft client certificate");
    assert!(server.connection.peer_identity().is_some(), "{}", kind.name());
  }
}

#[test]
fn composite_keys_never_sign_with_a_component_scheme() {
  for kind in Kind::ALL {
    let (_, pkcs8) = kind.draft_case();
    let key = super::key_provider()
      .load_private_key(PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(pkcs8)))
      .expect("draft key");
    let components = [
      SignatureScheme::ML_DSA_65,
      SignatureScheme::ED25519,
      SignatureScheme::ECDSA_NISTP256_SHA256,
      SignatureScheme::RSA_PSS_SHA256,
    ];
    assert!(key.choose_scheme(&components).is_none(), "{}", kind.name());
    let signer = key.choose_scheme(&[kind.scheme()]).expect("composite scheme");
    assert_eq!(signer.scheme(), kind.scheme());
    let signature = signer.sign(b"transcript").expect("sign");
    let spki = key.public_key().expect("SPKI");
    // The composite SPKI header is 21 bytes for both public-key lengths.
    algorithm(kind)
      .verify_signature(&spki.as_ref()[21..], b"transcript", &signature)
      .expect("composite verification");
  }
}

#[test]
fn composite_only_relying_party_refuses_component_peers() {
  for kind in Kind::ALL {
    let mut client_provider = provider();
    client_provider.signature_verification_algorithms = verify::COMPOSITE_ALGORITHMS;
    for credential in [Credential::MlDsa65, Credential::Ed25519, Credential::P256] {
      let (client, server) = configs(client_provider.clone(), provider(), credential, false);
      let (mut client, mut server) = connections_from_configs(Arc::new(client), Arc::new(server));
      assert!(
        drive(&mut client, &mut server).is_err(),
        "a composite-only client must refuse a {credential:?} server"
      );
    }
    let (client, server) = configs(client_provider, provider(), kind.credential(), false);
    let (mut client, mut server) = connections_from_configs(Arc::new(client), Arc::new(server));
    drive(&mut client, &mut server).expect("composite-only client accepts its composite");
  }
}

#[test]
fn servers_without_a_composite_offer_refuse_composite_keys() {
  // A client that does not offer the composite scheme leaves the server no
  // signature it may use: the handshake fails rather than falling back.
  for kind in Kind::ALL {
    let mut client_provider = provider();
    client_provider.signature_verification_algorithms = verify::SUPPORTED_ALGORITHMS;
    let mapping: Vec<_> = verify::SUPPORTED_ALGORITHMS
      .mapping()
      .iter()
      .copied()
      .filter(|(scheme, _)| *scheme != kind.scheme())
      .collect();
    client_provider.signature_verification_algorithms =
      rustls::crypto::WebPkiSupportedAlgorithms::new(verify::ALL_ALGORITHMS, Vec::leak(mapping)).expect("mapping");
    let (client, server) = configs(client_provider, provider(), kind.credential(), false);
    let (mut client, mut server) = connections_from_configs(Arc::new(client), Arc::new(server));
    assert!(drive(&mut client, &mut server).is_err(), "{}", kind.name());
  }
}

#[test]
fn composite_schemes_stay_out_of_tls12() {
  // draft-ietf-tls-composite-mldsa-00 section 2 forbids these schemes in
  // TLS 1.2. Neither codepoint names a TLS 1.2 signature algorithm, so a
  // TLS 1.2-only server with a composite key has no usable signature.
  let mut tls12 = provider();
  tls12.tls13_cipher_suites = alloc::borrow::Cow::Borrowed(&[]);
  tls12.tls12_cipher_suites = alloc::borrow::Cow::Borrowed(crate::cipher_suite::ALL_TLS12_CIPHER_SUITES);
  // Control: the same TLS 1.2 configuration authenticates an ECDSA server.
  let (client, server) = configs(tls12.clone(), tls12.clone(), Credential::P256, false);
  let (mut client, mut server) = connections_from_configs(Arc::new(client), Arc::new(server));
  drive(&mut client, &mut server).expect("TLS 1.2 control");
  for kind in Kind::ALL {
    let (client, server) = configs(tls12.clone(), tls12.clone(), kind.credential(), false);
    let (mut client, mut server) = connections_from_configs(Arc::new(client), Arc::new(server));
    let error = drive(&mut client, &mut server).expect_err("composite in TLS 1.2");
    assert!(
      matches!(
        error,
        rustls::Error::PeerIncompatible(rustls::error::PeerIncompatible::NoSignatureSchemesInCommon)
      ),
      "{}: {error:?}",
      kind.name()
    );
  }
  assert_eq!(verify::MLDSA65_ED25519_SHA512, SignatureScheme(0xfe41));
  assert_eq!(verify::MLDSA65_ECDSA_SECP256R1_SHA512, SignatureScheme(0xfe40));
}

/// A hostile peer's key: it signs each handshake with its composite scheme,
/// whatever the peer offers. A hostile server reports `disguise` to its own
/// stack so that it can select a TLS 1.2 suite.
#[derive(Debug)]
struct ForcedComposite {
  key: alloc::boxed::Box<dyn SigningKey>,
  disguise: Option<SignatureScheme>,
}

impl SigningKey for ForcedComposite {
  fn choose_scheme(&self, _offered: &[SignatureScheme]) -> Option<alloc::boxed::Box<dyn rustls::crypto::Signer>> {
    let signer = self
      .key
      .choose_scheme(&[verify::MLDSA65_ED25519_SHA512, verify::MLDSA65_ECDSA_SECP256R1_SHA512])?;
    Some(match self.disguise {
      Some(reported) => alloc::boxed::Box::new(Disguised { signer, reported }),
      None => signer,
    })
  }

  fn public_key(&self) -> Option<rustls::pki_types::SubjectPublicKeyInfoDer<'_>> {
    self.key.public_key()
  }
}

#[derive(Debug)]
struct Disguised {
  signer: alloc::boxed::Box<dyn rustls::crypto::Signer>,
  reported: SignatureScheme,
}

impl rustls::crypto::Signer for Disguised {
  fn sign(self: alloc::boxed::Box<Self>, message: &[u8]) -> Result<Vec<u8>, rustls::Error> {
    self.signer.sign(message)
  }

  fn scheme(&self) -> SignatureScheme {
    self.reported
  }
}

#[derive(Debug)]
struct ForcedKeys(Option<SignatureScheme>);

impl KeyProvider for ForcedKeys {
  fn load_private_key(&self, key: PrivateKeyDer<'static>) -> Result<alloc::boxed::Box<dyn SigningKey>, rustls::Error> {
    Ok(alloc::boxed::Box::new(ForcedComposite {
      key: super::key_provider().load_private_key(key)?,
      disguise: self.0,
    }))
  }
}

/// Set the signature scheme of the ECDHE ServerKeyExchange in `flight`, one
/// plaintext handshake record, to `scheme`.
fn relabel_server_key_exchange(flight: &mut [u8], scheme: SignatureScheme) {
  assert_eq!(flight[0], 22, "handshake record");
  let record = usize::from(u16::from_be_bytes([flight[3], flight[4]]));
  assert_eq!(record.strict_add(5), flight.len(), "one record");
  let mut at = 5_usize;
  while flight[at] != 12 {
    let length = u32::from_be_bytes([
      0,
      flight[at.strict_add(1)],
      flight[at.strict_add(2)],
      flight[at.strict_add(3)],
    ]);
    at = at
      .strict_add(4)
      .strict_add(usize::try_from(length).expect("handshake length"));
  }
  // ECParameters (curve_type and named_curve), then the opaque point.
  let point = at.strict_add(4).strict_add(3);
  let signed = point.strict_add(1).strict_add(usize::from(flight[point]));
  flight[signed..signed.strict_add(2)].copy_from_slice(&scheme.0.to_be_bytes());
}

#[test]
fn tls12_composite_signatures_get_illegal_parameter() {
  // draft-ietf-tls-composite-mldsa-00 section 2: a peer that receives a
  // composite scheme in a TLS 1.2 ServerKeyExchange or CertificateVerify
  // aborts with illegal_parameter. Each hostile peer holds a composite
  // certificate and sends a valid composite signature.
  static SERVER_KEYS: ForcedKeys = ForcedKeys(Some(SignatureScheme::ECDSA_NISTP256_SHA256));
  static CLIENT_KEYS: ForcedKeys = ForcedKeys(None);
  let hostile = |keys: &'static ForcedKeys| {
    let mut provider = provider();
    provider.tls13_cipher_suites = alloc::borrow::Cow::Borrowed(&[]);
    provider.tls12_cipher_suites = alloc::borrow::Cow::Borrowed(crate::cipher_suite::ALL_TLS12_CIPHER_SUITES);
    provider
      .tls12_cipher_suites
      .to_mut()
      .retain(|suite| suite.common.suite == CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256);
    provider.key_provider = keys;
    provider
  };
  // This provider with TLS 1.3 and TLS 1.2, so its offers carry the composite schemes.
  let ours = || {
    let mut provider = provider();
    provider.tls12_cipher_suites = alloc::borrow::Cow::Borrowed(crate::cipher_suite::ALL_TLS12_CIPHER_SUITES);
    provider
  };
  let illegal_parameter = Error::AlertReceived(AlertDescription::IllegalParameter);
  for kind in Kind::ALL {
    let (client, mut server) = configs(ours(), hostile(&SERVER_KEYS), kind.credential(), false);
    server.max_fragment_size = None;
    let (mut client, mut server) = connections_from_configs(Arc::new(client), Arc::new(server));
    transfer(&mut client, &mut server, |_, _| Ok(())).expect("ClientHello");
    relabel_server_key_exchange(&mut server.output, kind.scheme());
    let error = transfer(&mut server, &mut client, |_, _| Ok(())).expect_err("composite ServerKeyExchange");
    assert_eq!(
      error,
      Error::PeerMisbehaved(PeerMisbehaved::SignedKxWithWrongAlgorithm),
      "{}",
      kind.name()
    );
    let alert = transfer(&mut client, &mut server, |_, _| Ok(())).expect_err("client alert");
    assert_eq!(alert, illegal_parameter, "{}", kind.name());

    let (root, leaf, key) = super::tls::credentials(kind.credential());
    let (server_root, server_leaf, server_key) = super::tls::credentials(Credential::P256);
    let mut client_roots = RootCertStore::empty();
    client_roots.add(CertificateDer::from(root.to_vec())).expect("root");
    let client_roots = Arc::new(client_roots);
    let server_provider = Arc::new(ours());
    let verifiers: [Arc<dyn rustls::server::danger::ClientVerifier>; 2] = [
      Arc::new(
        rustls::server::WebPkiClientVerifier::builder(Arc::clone(&client_roots), &server_provider)
          .build()
          .expect("stock verifier"),
      ),
      Arc::new(verify::PreparedClientVerifier::new(client_roots).expect("prepared verifier")),
    ];
    for verifier in verifiers {
      let mut server_roots = RootCertStore::empty();
      server_roots
        .add(CertificateDer::from(server_root.to_vec()))
        .expect("root");
      let client = ClientConfig::builder_with_details(Arc::new(hostile(&CLIENT_KEYS)), Arc::new(FixtureTime))
        .with_root_certificates(server_roots)
        .with_client_auth_cert(
          Arc::new(Identity::from_cert_chain(vec![CertificateDer::from(leaf.to_vec())]).expect("identity")),
          PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key.to_vec())),
        )
        .expect("client credential");
      let server = ServerConfig::builder_with_details(Arc::clone(&server_provider), Arc::new(FixtureTime))
        .with_client_cert_verifier(verifier)
        .with_single_cert(
          Arc::new(Identity::from_cert_chain(vec![CertificateDer::from(server_leaf.to_vec())]).expect("identity")),
          PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(server_key.to_vec())),
        )
        .expect("server");
      let (mut client, mut server) = connections_from_configs(Arc::new(client), Arc::new(server));
      let error = drive(&mut client, &mut server).expect_err("composite CertificateVerify");
      assert_eq!(
        error,
        Error::PeerMisbehaved(PeerMisbehaved::SignedHandshakeWithUnadvertisedSigScheme),
        "{}",
        kind.name()
      );
      let alert = transfer(&mut server, &mut client, |_, _| Ok(())).expect_err("server alert");
      assert_eq!(alert, illegal_parameter, "{}", kind.name());
    }
  }
}
