use alloc::sync::Arc;
use core::time::Duration;
use std::io::{Read, Write};

use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::{ClientConfig, ClientConnection, Connection, Error, RootCertStore, ServerConfig, ServerConnection};

// Only certificate signing and verification use the dev-only peer here.
// This helper is removed when the authentication adapters enter the provider.
pub(crate) fn symmetric_provider() -> CryptoProvider {
  CryptoProvider {
    cipher_suites: crate::cipher_suite::ALL_CIPHER_SUITES.to_vec(),
    kx_groups: crate::kx_group::ALL_KX_GROUPS.to_vec(),
    secure_random: crate::random::SECURE_RANDOM,
    ..rustls::crypto::aws_lc_rs::default_provider()
  }
}

#[derive(Debug)]
struct FixtureTime;

impl rustls::time_provider::TimeProvider for FixtureTime {
  fn current_time(&self) -> Option<UnixTime> {
    // 2026-10-22 UTC, inside the fixed 2026–2036 fixture validity period.
    Some(UnixTime::since_unix_epoch(Duration::from_secs(1_792_627_200)))
  }
}

pub(crate) fn connections(client: CryptoProvider, server: CryptoProvider) -> (Connection, Connection) {
  let mut roots = RootCertStore::empty();
  roots
    .add(CertificateDer::from(
      include_bytes!("../../tests/vectors/credentials/p256-root.der").to_vec(),
    ))
    .expect("TLS fixture");
  let mut client = ClientConfig::builder_with_details(Arc::new(client), Arc::new(FixtureTime))
    .with_protocol_versions(&[&rustls::version::TLS13])
    .expect("TLS fixture")
    .with_root_certificates(roots)
    .with_no_client_auth();
  client.alpn_protocols = vec![b"provider-test".to_vec()];
  client.max_fragment_size = Some(512);
  let mut server = ServerConfig::builder_with_details(Arc::new(server), Arc::new(FixtureTime))
    .with_protocol_versions(&[&rustls::version::TLS13])
    .expect("TLS fixture")
    .with_no_client_auth()
    .with_single_cert(
      vec![CertificateDer::from(
        include_bytes!("../../tests/vectors/credentials/p256-leaf.der").to_vec(),
      )],
      PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(
        include_bytes!("../../tests/vectors/credentials/p256-leaf-key.der").to_vec(),
      )),
    )
    .expect("TLS fixture");
  server.alpn_protocols = client.alpn_protocols.clone();
  server.max_fragment_size = Some(512);
  (
    Connection::Client(
      ClientConnection::new(
        Arc::new(client),
        ServerName::try_from("localhost").expect("TLS fixture"),
      )
      .expect("TLS fixture"),
    ),
    Connection::Server(ServerConnection::new(Arc::new(server)).expect("TLS fixture")),
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
