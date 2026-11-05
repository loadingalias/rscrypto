//! Composite ML-DSA TLS signature schemes (draft-reddy-tls-composite-mldsa-12).
//!
//! The draft's Appendix E certificates come from the LAMPS reference
//! implementation and authenticate a client through webpki. The handshake
//! fixtures are built here deterministically: fixed key seeds and
//! deterministic composite signatures over minimal DER certificates.
//! `credential_fixtures_are_reproducible` regenerates them and requires the
//! checked-in files to match; set `RSCRYPTO_WRITE_COMPOSITE_CREDENTIALS=1` to
//! rewrite them.

use alloc::sync::Arc;
use alloc::vec::Vec;

use rscrypto::{
  CompositeMlDsaError, MlDsa65EcdsaP256, MlDsa65EcdsaP256SecretKey, MlDsa65Ed25519, MlDsa65Ed25519SecretKey,
};
use rustls::crypto::{Identity, SignatureScheme};
use rustls::pki_types::{
  AlgorithmIdentifier, CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, SignatureVerificationAlgorithm,
};
use rustls::{ClientConfig, RootCertStore, ServerConfig};

use super::tls::{Credential, FixtureTime, configs, connections_from_configs, drive, provider};
use crate::verify;

const VECTORS: &str = include_str!("../../../testdata/composite_mldsa/draft19/vectors.json");

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

fn base64(input: &str) -> Vec<u8> {
  const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  let digits: Vec<u32> = input
    .bytes()
    .filter(|&byte| byte != b'=')
    .map(|byte| u32::try_from(ALPHABET.iter().position(|&c| c == byte).expect("base64 digit")).expect("digit"))
    .collect();
  let mut out = Vec::new();
  for chunk in digits.chunks(4) {
    let word = chunk.iter().enumerate().fold(0u32, |word, (i, &digit)| {
      word | digit << 18usize.strict_sub(i.strict_mul(6))
    });
    out.extend_from_slice(&word.to_be_bytes()[1..chunk.len()]);
  }
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

  fn test_case(self) -> &'static str {
    match self {
      Self::Ed25519 => "id-MLDSA65-Ed25519-SHA512",
      Self::EcdsaP256 => "id-MLDSA65-ECDSA-P256-SHA512",
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

#[test]
fn credential_fixtures_are_reproducible() {
  let directory = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/credentials");
  let write = std::env::var_os("RSCRYPTO_WRITE_COMPOSITE_CREDENTIALS").is_some();
  for kind in Kind::ALL {
    for (name, expected) in credential_files(kind) {
      let path = format!("{directory}/{name}");
      if write {
        std::fs::write(&path, &expected).expect("write fixture");
      }
      assert_eq!(std::fs::read(&path).expect("checked-in fixture"), expected, "{name}");
    }
  }
}

fn draft_case(kind: Kind) -> (Vec<u8>, Vec<u8>) {
  // The fixture is indented JSON with one string field per line.
  let case = VECTORS
    .split("\"tcId\": ")
    .find(|case| case.starts_with(&format!("\"{}\"", kind.test_case())))
    .expect("test case");
  let field = |name: &str| {
    let (_, rest) = case.split_once(&format!("\"{name}\": \"")).expect("field");
    let (value, _) = rest.split_once('"').expect("string end");
    base64(value)
  };
  (field("x5c"), field("sk_pkcs8"))
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
    let (x5c, _) = draft_case(kind);
    let (certificate, _) = element(&x5c);
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
    let (x5c, pkcs8) = draft_case(kind);
    let (root, leaf, key) = super::tls::credentials(kind.credential());
    let mut client_roots = RootCertStore::empty();
    client_roots.add(CertificateDer::from(root.to_vec())).expect("root");
    let mut client_cert_roots = RootCertStore::empty();
    client_cert_roots
      .add(CertificateDer::from(x5c.clone()))
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
    let (_, pkcs8) = draft_case(kind);
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
  // draft-reddy-tls-composite-mldsa-12 section 2 forbids these schemes in
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
  assert_eq!(verify::MLDSA65_ED25519_SHA512, SignatureScheme(0x090b));
  assert_eq!(verify::MLDSA65_ECDSA_SECP256R1_SHA512, SignatureScheme(0xfe40));
}
