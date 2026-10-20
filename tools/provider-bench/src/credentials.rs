use crate::Workload;
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
    "rsa3072" | "rsa4096" => SignatureScheme::RSA_PSS_SHA256,
    "rsa3072-sha384" | "rsa4096-sha384" => SignatureScheme::RSA_PSS_SHA384,
    "rsa3072-sha512" | "rsa4096-sha512" => SignatureScheme::RSA_PSS_SHA512,
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

#[derive(Clone, Copy)]
pub(crate) struct Credential {
  pub root: &'static [u8],
  pub leaf: &'static [u8],
  pub key: &'static [u8],
  pub intermediate: Option<&'static [u8]>,
}

#[expect(clippy::panic, reason = "the catalog selects known credentials")]
pub(crate) fn der(name: &str, workload: Workload) -> Credential {
  macro_rules! file {
    ($name:literal, $suffix:literal) => {
      include_bytes!(concat!(
        "../../../rscrypto-rustls/tests/vectors/credentials/",
        $name,
        $suffix
      ))
      .as_slice()
    };
  }
  macro_rules! files {
    ($name:literal) => {
      Credential {
        root: file!($name, "-root.der"),
        leaf: file!($name, "-leaf.der"),
        key: file!($name, "-leaf-key.der"),
        intermediate: None,
      }
    };
  }
  macro_rules! chain {
    ($name:literal) => {
      Credential {
        intermediate: Some(file!($name, "-intermediate.der")),
        ..files!($name)
      }
    };
  }
  if workload == Workload::Intermediate {
    return match name {
      "p256" => chain!("p256-chain"),
      "p384" => chain!("p384-chain"),
      "ed25519" => chain!("ed25519-chain"),
      "rsa3072" => chain!("rsa3072-chain"),
      "mldsa44" => chain!("mldsa44-chain"),
      "mldsa65" => chain!("mldsa65-chain"),
      "mldsa87" => chain!("mldsa87-chain"),
      _ => panic!("unknown chain credential"),
    };
  }
  match name {
    "p256" => files!("p256"),
    "p384" => files!("p384"),
    "ed25519" => files!("ed25519"),
    "rsa3072" | "rsa3072-sha384" | "rsa3072-sha512" => files!("rsa3072"),
    "rsa4096" | "rsa4096-sha384" | "rsa4096-sha512" => files!("rsa4096"),
    "mldsa44" => files!("mldsa44"),
    "mldsa65" => files!("mldsa65"),
    "mldsa87" => files!("mldsa87"),
    _ => panic!("unknown credential"),
  }
}

/// Build both endpoints. With `prepared`, the rscrypto provider's prepared-anchor
/// verifiers replace rustls's WebPKI verifiers, with the same signature schemes.
pub(crate) fn configs(
  client_provider: rustls::crypto::CryptoProvider,
  server_provider: rustls::crypto::CryptoProvider,
  name: &str,
  prepared: bool,
  workload: Workload,
) -> (ClientConfig, ServerConfig) {
  let Credential {
    root,
    leaf,
    key,
    intermediate,
  } = der(name, workload);
  let mut chain = vec![CertificateDer::from(leaf)];
  chain.extend(intermediate.map(CertificateDer::from));
  let identity = Arc::new(rustls::crypto::Identity::from_cert_chain(chain).expect("fixture identity"));
  let mut roots = RootCertStore::empty();
  roots.add(CertificateDer::from(root)).expect("root fixture");
  let roots = Arc::new(roots);
  let provider = Arc::new(server_provider);
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
  let client = ClientConfig::builder_with_details(Arc::new(client_provider), Arc::new(FixtureTime));
  let client = if prepared {
    client.dangerous().with_custom_certificate_verifier(Arc::new(
      rscrypto_rustls::verify::PreparedServerVerifier::with_algorithms(roots, supported).expect("server verifier"),
    ))
  } else {
    client.with_root_certificates(roots)
  };
  let mut client = if workload == Workload::ServerAuth {
    client.with_no_client_auth().expect("client configuration")
  } else {
    client
      .with_client_auth_cert(
        identity.clone(),
        PrivateKeyDer::try_from(key.to_vec()).expect("DER fixture"),
      )
      .expect("client identity")
  };
  let server = ServerConfig::builder_with_details(provider, Arc::new(FixtureTime));
  let server = if workload == Workload::ServerAuth {
    server.with_no_client_auth()
  } else {
    server.with_client_cert_verifier(verifier)
  };
  let mut server = server
    .with_single_cert(identity, PrivateKeyDer::try_from(key.to_vec()).expect("DER fixture"))
    .expect("server identity");
  client.alpn_protocols = vec![ALPN.to_vec().into()];
  server.alpn_protocols = client.alpn_protocols.clone();
  (client, server)
}
