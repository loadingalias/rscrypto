use alloc::sync::Arc;
use core::time::Duration;
use std::io::{Read, Write};

use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::{ClientConfig, ClientConnection, Connection, Error, RootCertStore, ServerConfig, ServerConnection};

#[derive(Debug)]
struct FixtureTime;

impl rustls::time_provider::TimeProvider for FixtureTime {
  fn current_time(&self) -> Option<UnixTime> {
    // 2026-10-22 UTC, inside the fixed 2026–2036 fixture validity period.
    Some(UnixTime::since_unix_epoch(Duration::from_secs(1_792_627_200)))
  }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Credential {
  #[cfg(feature = "ecdsa-p256")]
  P256,
  #[cfg(feature = "ecdsa-p384")]
  P384,
  #[cfg(feature = "ed25519")]
  Ed25519,
  #[cfg(feature = "rsa")]
  Rsa,
  #[cfg(feature = "ml-dsa")]
  MlDsa44,
  #[cfg(feature = "ml-dsa")]
  MlDsa65,
  #[cfg(feature = "ml-dsa")]
  MlDsa87,
}

impl Credential {
  pub(crate) fn scheme(self) -> rustls::SignatureScheme {
    match self {
      #[cfg(feature = "ecdsa-p256")]
      Self::P256 => rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
      #[cfg(feature = "ecdsa-p384")]
      Self::P384 => rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
      #[cfg(feature = "ed25519")]
      Self::Ed25519 => rustls::SignatureScheme::ED25519,
      #[cfg(feature = "rsa")]
      Self::Rsa => rustls::SignatureScheme::RSA_PSS_SHA256,
      #[cfg(feature = "ml-dsa")]
      Self::MlDsa44 => rustls::SignatureScheme::ML_DSA_44,
      #[cfg(feature = "ml-dsa")]
      Self::MlDsa65 => rustls::SignatureScheme::ML_DSA_65,
      #[cfg(feature = "ml-dsa")]
      Self::MlDsa87 => rustls::SignatureScheme::ML_DSA_87,
    }
  }
}

pub(crate) fn credentials(kind: Credential) -> (&'static [u8], &'static [u8], &'static [u8]) {
  macro_rules! files {
    ($kind:literal) => {
      (
        &include_bytes!(concat!("../../tests/vectors/credentials/", $kind, "-root.der"))[..],
        &include_bytes!(concat!("../../tests/vectors/credentials/", $kind, "-leaf.der"))[..],
        &include_bytes!(concat!("../../tests/vectors/credentials/", $kind, "-leaf-key.der"))[..],
      )
    };
  }
  match kind {
    #[cfg(feature = "ecdsa-p256")]
    Credential::P256 => files!("p256"),
    #[cfg(feature = "ecdsa-p384")]
    Credential::P384 => files!("p384"),
    #[cfg(feature = "ed25519")]
    Credential::Ed25519 => files!("ed25519"),
    #[cfg(feature = "rsa")]
    Credential::Rsa => files!("rsa3072"),
    #[cfg(feature = "ml-dsa")]
    Credential::MlDsa44 => files!("mldsa44"),
    #[cfg(feature = "ml-dsa")]
    Credential::MlDsa65 => files!("mldsa65"),
    #[cfg(feature = "ml-dsa")]
    Credential::MlDsa87 => files!("mldsa87"),
  }
}

pub(crate) fn configs(
  client: CryptoProvider,
  server: CryptoProvider,
  kind: Credential,
  client_auth: bool,
) -> (ClientConfig, ServerConfig) {
  let (root, leaf, key) = credentials(kind);
  let mut roots = RootCertStore::empty();
  roots.add(CertificateDer::from(root.to_vec())).expect("TLS fixture");
  let server_provider = Arc::new(server);
  let verifier =
    rustls::server::WebPkiClientVerifier::builder_with_provider(Arc::new(roots.clone()), server_provider.clone())
      .build()
      .expect("client verifier");
  let client = ClientConfig::builder_with_details(Arc::new(client), Arc::new(FixtureTime))
    .with_protocol_versions(&[&rustls::version::TLS13])
    .expect("TLS fixture")
    .with_root_certificates(roots);
  let mut client = if client_auth {
    client
      .with_client_auth_cert(
        vec![CertificateDer::from(leaf.to_vec())],
        PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key.to_vec())),
      )
      .expect("client credential")
  } else {
    client.with_no_client_auth()
  };
  client.alpn_protocols = vec![b"provider-test".to_vec()];
  client.max_fragment_size = Some(512);
  let server = ServerConfig::builder_with_details(server_provider, Arc::new(FixtureTime))
    .with_protocol_versions(&[&rustls::version::TLS13])
    .expect("TLS fixture");
  let server = if client_auth {
    server.with_client_cert_verifier(verifier)
  } else {
    server.with_no_client_auth()
  };
  let mut server = server
    .with_single_cert(
      vec![CertificateDer::from(leaf.to_vec())],
      PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key.to_vec())),
    )
    .expect("TLS fixture");
  server.alpn_protocols = client.alpn_protocols.clone();
  server.max_fragment_size = Some(512);
  (client, server)
}

pub(crate) fn connections_from_configs(
  client: Arc<ClientConfig>,
  server: Arc<ServerConfig>,
) -> (Connection, Connection) {
  (
    Connection::Client(
      ClientConnection::new(client, ServerName::try_from("localhost").expect("TLS fixture")).expect("TLS fixture"),
    ),
    Connection::Server(ServerConnection::new(server).expect("TLS fixture")),
  )
}

fn transfer(source: &mut Connection, destination: &mut Connection) -> Result<usize, Error> {
  let mut flight = Vec::new();
  while source.wants_write() {
    source.write_tls(&mut flight).expect("write flight");
    if flight.len() > 131_072 {
      return Err(Error::General("test flight exceeded its size bound".into()));
    }
  }
  // Transport fragmentation is independent of the configured record size.
  for chunk in flight.chunks(37) {
    let mut chunk = chunk;
    while !chunk.is_empty() {
      if destination.read_tls(&mut chunk).expect("read flight") == 0 {
        return Err(Error::General("test receiver stalled".into()));
      }
      destination.process_new_packets()?;
    }
  }
  Ok(flight.len())
}

pub(crate) fn drive(client: &mut Connection, server: &mut Connection) -> Result<(), Error> {
  for _ in 0..128 {
    let progress = transfer(client, server)?.strict_add(transfer(server, client)?);
    if !client.wants_write() && !server.wants_write() && !client.is_handshaking() && !server.is_handshaking() {
      return Ok(());
    }
    if progress == 0 {
      return Err(Error::General("test handshake stalled".into()));
    }
  }
  Err(Error::General("test handshake exceeded its flight count".into()))
}

pub(crate) fn application_exchange(client: &mut Connection, server: &mut Connection) {
  let request = b"client application data".repeat(100);
  let response = b"server application data".repeat(100);
  client.writer().write_all(&request).expect("write request");
  server.writer().write_all(&response).expect("write response");
  drive(client, server).expect("application transfer");
  let mut received = vec![0; request.len()];
  server
    .reader()
    .read_exact(&mut received)
    .expect("read application data");
  assert_eq!(received, request);
  received.resize(response.len(), 0);
  client
    .reader()
    .read_exact(&mut received)
    .expect("read application data");
  assert_eq!(received, response);
}
