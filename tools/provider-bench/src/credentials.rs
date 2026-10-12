use alloc::sync::Arc;
use core::time::Duration;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, UnixTime};
use rustls::{ClientConfig, RootCertStore, ServerConfig};

pub(crate) const TIME: u64 = 1_792_627_200;
pub(crate) const ALPN: &[u8] = b"provider-test";

#[expect(
  clippy::panic,
  reason = "the benchmark catalog must select a known credential fixture"
)]
pub(crate) fn scheme(name: &str) -> rustls::crypto::SignatureScheme {
  use rustls::crypto::SignatureScheme;
  match name {
    "p256" => SignatureScheme::ECDSA_NISTP256_SHA256,
    "p384" => SignatureScheme::ECDSA_NISTP384_SHA384,
    "ed25519" => SignatureScheme::ED25519,
    "rsa3072" => SignatureScheme::RSA_PSS_SHA256,
    "mldsa44" => SignatureScheme::ML_DSA_44,
    "mldsa65" => SignatureScheme::ML_DSA_65,
    "mldsa87" => SignatureScheme::ML_DSA_87,
    _ => panic!("unknown credential"),
  }
}

#[derive(Debug)]
struct FixtureTime;

impl rustls::time_provider::TimeProvider for FixtureTime {
  fn current_time(&self) -> Option<UnixTime> {
    Some(UnixTime::since_unix_epoch(Duration::from_secs(TIME)))
  }
}

#[expect(
  clippy::panic,
  reason = "the benchmark catalog must select a known credential fixture"
)]
pub(crate) fn der(name: &str) -> (&'static [u8], &'static [u8], &'static [u8]) {
  macro_rules! files {
    ($name:literal) => {
      (
        include_bytes!(concat!(
          "../../../rscrypto-rustls/tests/vectors/credentials/",
          $name,
          "-root.der"
        ))
        .as_slice(),
        include_bytes!(concat!(
          "../../../rscrypto-rustls/tests/vectors/credentials/",
          $name,
          "-leaf.der"
        ))
        .as_slice(),
        include_bytes!(concat!(
          "../../../rscrypto-rustls/tests/vectors/credentials/",
          $name,
          "-leaf-key.der"
        ))
        .as_slice(),
      )
    };
  }
  match name {
    "p256" => files!("p256"),
    "p384" => files!("p384"),
    "ed25519" => files!("ed25519"),
    "rsa3072" => files!("rsa3072"),
    "mldsa44" => files!("mldsa44"),
    "mldsa65" => files!("mldsa65"),
    "mldsa87" => files!("mldsa87"),
    _ => panic!("unknown credential"),
  }
}

/// Build both endpoints. With `prepared`, the rscrypto provider's prepared-anchor
/// verifiers replace rustls's WebPKI verifiers, with the same signature schemes.
pub(crate) fn configs(
  provider: rustls::crypto::CryptoProvider,
  name: &str,
  prepared: bool,
) -> (ClientConfig, ServerConfig) {
  let (root, leaf, key) = der(name);
  let mut roots = RootCertStore::empty();
  roots.add(CertificateDer::from(root)).expect("root fixture");
  let roots = Arc::new(roots);
  let provider = Arc::new(provider);
  let supported = provider.signature_verification_algorithms;
  let verifier: Arc<dyn rustls::server::danger::ClientVerifier> = if prepared {
    Arc::new(
      rscrypto_rustls::verify::PreparedClientVerifier::with_algorithms(Arc::clone(&roots), supported)
        .expect("client verifier"),
    )
  } else {
    Arc::new(
      rustls::server::WebPkiClientVerifier::builder(Arc::clone(&roots), &provider)
        .build()
        .expect("client verifier"),
    )
  };
  let client = ClientConfig::builder_with_details(provider.clone(), Arc::new(FixtureTime));
  let client = if prepared {
    client.dangerous().with_custom_certificate_verifier(Arc::new(
      rscrypto_rustls::verify::PreparedServerVerifier::with_algorithms(roots, supported).expect("server verifier"),
    ))
  } else {
    client.with_root_certificates(roots)
  };
  let mut client = client
    .with_client_auth_cert(
      Arc::new(rustls::crypto::Identity::from_cert_chain(vec![CertificateDer::from(leaf)]).expect("fixture identity")),
      PrivateKeyDer::try_from(key.to_vec()).expect("DER fixture"),
    )
    .expect("client identity");
  let mut server = ServerConfig::builder_with_details(provider, Arc::new(FixtureTime))
    .with_client_cert_verifier(verifier)
    .with_single_cert(
      Arc::new(rustls::crypto::Identity::from_cert_chain(vec![CertificateDer::from(leaf)]).expect("fixture identity")),
      PrivateKeyDer::try_from(key.to_vec()).expect("DER fixture"),
    )
    .expect("server identity");
  client.alpn_protocols = vec![ALPN.to_vec().into()];
  server.alpn_protocols = client.alpn_protocols.clone();
  (client, server)
}
