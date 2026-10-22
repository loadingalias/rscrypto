//! Prepared-anchor verifiers against rustls's WebPKI verifiers for the same roots.
//!
//! Rustls's verifiers, configured with this crate's algorithms and no
//! revocation lists, are the oracle: every accepted and rejected chain must
//! give the same result and the same error.

use alloc::{format, sync::Arc, vec::Vec};
use core::time::Duration;

use rustls::client::WebPkiServerVerifier;
use rustls::client::danger::{ServerIdentity, ServerVerifier};
use rustls::crypto::{CertificateIdentity, Identity};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::server::WebPkiClientVerifier;
use rustls::server::danger::ClientVerifier;
use rustls::{ClientConfig, RootCertStore, ServerConfig};

use super::tls::{Credential, application_exchange, connections_from_configs, credentials, drive};
use crate::verify::{PreparedClientVerifier, PreparedServerVerifier};

const CREDENTIALS: &[Credential] = &[
  #[cfg(feature = "ecdsa-p256")]
  Credential::P256,
  #[cfg(feature = "ecdsa-p384")]
  Credential::P384,
  #[cfg(feature = "ed25519")]
  Credential::Ed25519,
  #[cfg(feature = "rsa")]
  Credential::Rsa,
  #[cfg(feature = "ml-dsa")]
  Credential::MlDsa65,
];

// 2026-10-22 UTC, inside the fixtures' 2026–2036 validity, and a time after it.
const VALID: UnixTime = UnixTime::since_unix_epoch(Duration::from_secs(1_792_627_200));
const EXPIRED: UnixTime = UnixTime::since_unix_epoch(Duration::from_secs(2_208_988_800));

fn roots(kind: Credential) -> Arc<RootCertStore> {
  let mut roots = RootCertStore::empty();
  roots
    .add(CertificateDer::from(credentials(kind).0.to_vec()))
    .expect("root fixture");
  Arc::new(roots)
}

fn identity(leaf: &[u8]) -> Identity<'static> {
  Identity::X509(CertificateIdentity::new(
    CertificateDer::from(leaf.to_vec()),
    Vec::new(),
  ))
}

/// Another enabled credential's roots, which did not issue `kind`'s leaf.
fn foreign_roots(kind: Credential) -> Option<Arc<RootCertStore>> {
  CREDENTIALS
    .iter()
    .copied()
    .find(|candidate| candidate.scheme() != kind.scheme())
    .map(roots)
}

/// Valid, re-signed, wrong-issuer, malformed and expired inputs for `kind`.
fn cases(kind: Credential) -> Vec<(&'static str, Arc<RootCertStore>, Identity<'static>, UnixTime)> {
  let (_, leaf, _) = credentials(kind);
  let mut corrupted = leaf.to_vec();
  // The certificate's last byte belongs to its signature value.
  *corrupted.last_mut().expect("nonempty leaf") ^= 0x01;
  let mut cases = vec![
    ("valid", roots(kind), identity(leaf), VALID),
    ("bad signature", roots(kind), identity(&corrupted), VALID),
    ("malformed", roots(kind), identity(&leaf[..leaf.len() / 2]), VALID),
    ("expired", roots(kind), identity(leaf), EXPIRED),
    (
      "raw public key",
      roots(kind),
      Identity::RawPublicKey(rustls::pki_types::SubjectPublicKeyInfoDer::from(&[0x30, 0x00][..])),
      VALID,
    ),
  ];
  // A single-credential build has no other issuer.
  if let Some(foreign) = foreign_roots(kind) {
    cases.push(("unknown issuer", foreign, identity(leaf), VALID));
  }
  cases
}

#[test]
fn prepared_server_verifier_matches_webpki_results_and_errors() {
  let provider = super::tls::provider();
  for &kind in CREDENTIALS {
    for (case, roots, identity, now) in cases(kind) {
      let stock = WebPkiServerVerifier::builder(Arc::clone(&roots), &provider)
        .build()
        .expect("stock verifier");
      let prepared = PreparedServerVerifier::new(roots).expect("prepared verifier");
      for name in ["localhost", "example.invalid"] {
        let server_name = ServerName::try_from(name).expect("server name");
        let input = ServerIdentity::new(&identity, &server_name, now);
        assert_eq!(
          format!("{:?}", prepared.verify_identity(&input)),
          format!("{:?}", stock.verify_identity(&input)),
          "{kind:?} {case} {name}"
        );
      }
    }
  }
}

#[derive(Debug)]
struct At(UnixTime);

impl rustls::time_provider::TimeProvider for At {
  fn current_time(&self) -> Option<UnixTime> {
    Some(self.0)
  }
}

/// Run one mutual handshake whose server verifies the client with `verifier`.
fn client_authentication(
  kind: Credential,
  verifier: Arc<dyn ClientVerifier>,
  client_leaf: &[u8],
  server_time: UnixTime,
) -> Result<(), rustls::Error> {
  let (_, leaf, key) = credentials(kind);
  let key = || rustls::pki_types::PrivateKeyDer::try_from(key.to_vec()).expect("key fixture");
  let chain = |certificate: &[u8]| {
    Arc::new(
      rustls::crypto::Identity::from_cert_chain(vec![CertificateDer::from(certificate.to_vec())]).expect("identity"),
    )
  };
  let client = ClientConfig::builder_with_details(Arc::new(super::tls::provider()), Arc::new(At(VALID)))
    .with_root_certificates(roots(kind))
    .with_client_auth_cert(chain(client_leaf), key())
    .expect("client configuration");
  let server = ServerConfig::builder_with_details(Arc::new(super::tls::provider()), Arc::new(At(server_time)))
    .with_client_cert_verifier(verifier)
    .with_single_cert(chain(leaf), key())
    .expect("server configuration");
  let (mut client, mut server) = connections_from_configs(Arc::new(client), Arc::new(server));
  drive(&mut client, &mut server)
}

#[test]
fn prepared_client_verifier_matches_webpki_results_and_errors() {
  // `ClientIdentity` has no public constructor, so each case runs a real
  // handshake twice: once per verifier, with identical client inputs.
  let provider = super::tls::provider();
  for &kind in CREDENTIALS {
    let (_, leaf, _) = credentials(kind);
    let mut corrupted = leaf.to_vec();
    *corrupted.last_mut().expect("nonempty leaf") ^= 0x01;
    let mut cases = vec![
      ("valid", roots(kind), leaf, VALID),
      ("bad signature", roots(kind), corrupted.as_slice(), VALID),
      ("expired", roots(kind), leaf, EXPIRED),
    ];
    // A single-credential build has no other issuer.
    if let Some(foreign) = foreign_roots(kind) {
      cases.push(("unknown issuer", foreign, leaf, VALID));
    }
    for (case, server_roots, client_leaf, server_time) in cases {
      let stock = WebPkiClientVerifier::builder(Arc::clone(&server_roots), &provider)
        .build()
        .expect("stock verifier");
      let prepared = PreparedClientVerifier::new(server_roots).expect("prepared verifier");
      assert_eq!(prepared.root_hint_subjects(), stock.root_hint_subjects());
      assert_eq!(prepared.supported_verify_schemes(), stock.supported_verify_schemes());
      assert_eq!(prepared.client_auth_mandatory(), stock.client_auth_mandatory());
      let expected = client_authentication(kind, Arc::new(stock), client_leaf, server_time);
      let actual = client_authentication(kind, Arc::new(prepared), client_leaf, server_time);
      assert_eq!(format!("{actual:?}"), format!("{expected:?}"), "{kind:?} {case}");
      assert_eq!(actual.is_ok(), case == "valid", "{kind:?} {case}");
    }
  }
}

#[test]
fn ecdsa_and_mldsa_anchor_keys_are_retained() {
  for &kind in CREDENTIALS {
    let expected = match kind {
      #[cfg(feature = "ecdsa-p256")]
      Credential::P256 => 65 + size_of::<rscrypto::EcdsaP256PublicKey>(),
      #[cfg(feature = "ecdsa-p384")]
      Credential::P384 => 97 + size_of::<rscrypto::EcdsaP384PublicKey>(),
      #[cfg(feature = "ml-dsa")]
      Credential::MlDsa65 => {
        1952
          + size_of::<Box<rscrypto::MlDsa65PreparedPublicKeyStorage>>()
          + size_of::<rscrypto::MlDsa65PreparedPublicKeyStorage>()
      }
      // Ed25519, RSA, ML-DSA-44 and ML-DSA-87 anchors are not in `CREDENTIALS`
      // or are not prepared.
      #[cfg(any(feature = "ed25519", feature = "rsa", feature = "ml-dsa"))]
      _ => 0,
    };
    let server = PreparedServerVerifier::new(roots(kind)).expect("prepared verifier");
    let client = PreparedClientVerifier::new(roots(kind)).expect("prepared verifier");
    assert_eq!(server.retained_anchor_bytes(), expected, "{kind:?}");
    assert_eq!(client.retained_anchor_bytes(), expected, "{kind:?}");
  }
  assert!(matches!(
    PreparedServerVerifier::new(Arc::new(RootCertStore::empty())),
    Err(rustls::client::VerifierBuilderError::NoRootAnchors)
  ));
  assert!(matches!(
    PreparedClientVerifier::new(Arc::new(RootCertStore::empty())),
    Err(rustls::server::VerifierBuilderError::NoRootAnchors)
  ));
}

#[test]
fn mutual_handshakes_complete_through_prepared_verifiers() {
  for &kind in CREDENTIALS {
    let (_, leaf, key) = credentials(kind);
    let chain = || {
      Arc::new(rustls::crypto::Identity::from_cert_chain(vec![CertificateDer::from(leaf.to_vec())]).expect("identity"))
    };
    let key = || rustls::pki_types::PrivateKeyDer::try_from(key.to_vec()).expect("key fixture");
    let time = Arc::new(super::tls::FixtureTime);
    let client = ClientConfig::builder_with_details(Arc::new(super::tls::provider()), time.clone())
      .dangerous()
      .with_custom_certificate_verifier(Arc::new(PreparedServerVerifier::new(roots(kind)).expect("verifier")))
      .with_client_auth_cert(chain(), key())
      .expect("client configuration");
    let server = ServerConfig::builder_with_details(Arc::new(super::tls::provider()), time)
      .with_client_cert_verifier(Arc::new(PreparedClientVerifier::new(roots(kind)).expect("verifier")))
      .with_single_cert(chain(), key())
      .expect("server configuration");
    let (mut client, mut server) = connections_from_configs(Arc::new(client), Arc::new(server));
    assert_eq!(format!("{:?}", drive(&mut client, &mut server)), "Ok(())", "{kind:?}");
    application_exchange(&mut client, &mut server);
  }
}
