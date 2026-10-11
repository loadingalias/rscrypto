use alloc::sync::Arc;
use std::io::{Read, Write};

use rustls::crypto::CryptoProvider;
use rustls::server::ProducesTickets;
use rustls::{CipherSuite, Connection, Error, HandshakeKind, NamedGroup, ProtocolVersion};

use super::tls::{self, Credential};

fn select(mut provider: CryptoProvider, group: NamedGroup, suite: CipherSuite) -> CryptoProvider {
  provider.kx_groups.retain(|supported| supported.name() == group);
  provider.cipher_suites.retain(|supported| supported.suite() == suite);
  assert_eq!(provider.kx_groups.len(), 1);
  assert_eq!(provider.cipher_suites.len(), 1);
  provider
}

fn exercise(client: &mut Connection, server: &mut Connection) -> [u8; 64] {
  let label = b"rscrypto provider interop";
  let context = Some(&b"context"[..]);
  assert_eq!(client.alpn_protocol(), Some(&b"provider-test"[..]));
  assert_eq!(server.alpn_protocol(), client.alpn_protocol());
  let exported = client
    .export_keying_material([0; 64], label, context)
    .expect("exporter");
  assert_eq!(
    exported,
    server
      .export_keying_material([0; 64], label, context)
      .expect("peer exporter")
  );
  assert_ne!(
    exported,
    client
      .export_keying_material([0; 64], label, Some(b"other"))
      .expect("different context")
  );
  assert_ne!(
    exported,
    client
      .export_keying_material([0; 64], b"other", context)
      .expect("different label")
  );
  tls::application_exchange(client, server);
  client.refresh_traffic_keys().expect("client KeyUpdate");
  server.refresh_traffic_keys().expect("server KeyUpdate");
  tls::application_exchange(client, server);
  // KeyUpdate changes traffic keys, not the handshake exporter secret.
  assert_eq!(
    exported,
    client
      .export_keying_material([0; 64], label, context)
      .expect("exporter after update")
  );
  assert_eq!(
    exported,
    server
      .export_keying_material([0; 64], label, context)
      .expect("peer exporter after update")
  );
  client.send_close_notify();
  server.send_close_notify();
  tls::drive(client, server).expect("close exchange");
  assert_eq!(client.reader().read(&mut [0; 1]).expect("client EOF"), 0);
  assert_eq!(server.reader().read(&mut [0; 1]).expect("server EOF"), 0);
  exported
}

fn profile_matrix(credential: Credential) {
  type Peer = (fn() -> CryptoProvider, fn() -> Result<Arc<dyn ProducesTickets>, Error>);
  let peers: [Peer; 2] = [
    (
      rustls::crypto::ring::default_provider,
      rustls::crypto::ring::Ticketer::new,
    ),
    (
      rustls::crypto::aws_lc_rs::default_provider,
      rustls::crypto::aws_lc_rs::Ticketer::new,
    ),
  ];
  let mut completed = 0_usize;
  for (peer_provider, peer_ticketer) in peers {
    if !peer_provider()
      .signature_verification_algorithms
      .supported_schemes()
      .contains(&credential.scheme())
    {
      continue; // ring has no ML-DSA; AWS-LC and the OpenSSL matrix cover it.
    }
    for group in crate::kx_group::ALL_KX_GROUPS {
      if !peer_provider().kx_groups.iter().any(|peer| peer.name() == group.name()) {
        continue; // ring has no hybrid group; AWS-LC covers it in both roles.
      }
      for suite in crate::cipher_suite::ALL_CIPHER_SUITES {
        for ours_client in [true, false] {
          for stateless in [false, true] {
            if stateless && !ours_client && !cfg!(feature = "aes-gcm") {
              continue;
            }
            let ours = select(crate::default_provider(), group.name(), suite.suite());
            let peer = select(peer_provider(), group.name(), suite.suite());
            let (client, server) = if ours_client { (ours, peer) } else { (peer, ours) };
            let (client, mut server) = tls::configs(client, server, credential, true);
            assert!(!client.enable_early_data);
            assert_eq!(server.max_early_data_size, 0);
            if stateless {
              server.ticketer = if ours_client {
                peer_ticketer().expect("independent ticketer")
              } else {
                #[cfg(feature = "aes-gcm")]
                {
                  crate::ticket::ticketer().expect("rscrypto ticketer")
                }
                #[cfg(not(feature = "aes-gcm"))]
                {
                  continue;
                }
              };
              server.session_storage = Arc::new(rustls::server::NoServerSessionStorage {});
            }
            let (client_config, server_config) = (Arc::new(client), Arc::new(server));
            let mut previous = None;
            for expected in [HandshakeKind::Full, HandshakeKind::Resumed] {
              let (mut client, mut server) =
                tls::connections_from_configs(client_config.clone(), server_config.clone());
              tls::drive(&mut client, &mut server)
                .map_err(|error| {
                  format!(
                    "{credential:?}/{:?}/{:?}/ours_client={ours_client}/stateless={stateless}/{expected:?}: {error}",
                    group.name(),
                    suite.suite()
                  )
                })
                .expect("profile handshake");
              for connection in [&client, &server] {
                assert_eq!(connection.handshake_kind(), Some(expected));
                assert_eq!(connection.protocol_version(), Some(ProtocolVersion::TLSv1_3));
                assert_eq!(
                  connection.negotiated_cipher_suite().expect("suite").suite(),
                  suite.suite()
                );
                assert_eq!(
                  connection.negotiated_key_exchange_group().expect("group").name(),
                  group.name()
                );
                let certificates = connection.peer_certificates().expect("mutual authentication");
                assert_eq!(certificates.len(), 1);
                assert_eq!(certificates[0].as_ref(), tls::credentials(credential).1);
              }
              let exporter = exercise(&mut client, &mut server);
              assert_ne!(Some(exporter), previous);
              previous = Some(exporter);
              completed = completed.strict_add(1);
            }
          }
        }
      }
    }
  }
  assert!(completed >= 4, "each enabled profile needs both roles and resumption");
  eprintln!("{credential:?}: {completed} authenticated full/resumed connections");
}

#[cfg(feature = "ecdsa-p256")]
#[test]
fn p256_profiles() {
  profile_matrix(Credential::P256);
}

#[cfg(feature = "ecdsa-p384")]
#[test]
fn p384_profiles() {
  profile_matrix(Credential::P384);
}

#[cfg(feature = "ed25519")]
#[test]
fn ed25519_profiles() {
  profile_matrix(Credential::Ed25519);
}

#[cfg(feature = "rsa")]
#[test]
fn rsa_profiles() {
  profile_matrix(Credential::Rsa);
}

#[cfg(feature = "ml-dsa")]
#[test]
fn mldsa44_profiles() {
  profile_matrix(Credential::MlDsa44);
}

#[cfg(feature = "ml-dsa")]
#[test]
fn mldsa65_profiles() {
  profile_matrix(Credential::MlDsa65);
}

#[cfg(feature = "ml-dsa")]
#[test]
fn mldsa87_profiles() {
  profile_matrix(Credential::MlDsa87);
}

#[test]
fn early_data_requires_explicit_enablement_and_obeys_server_rejection() {
  let credential = [
    #[cfg(feature = "ml-dsa")]
    Credential::MlDsa44,
    #[cfg(feature = "ecdsa-p256")]
    Credential::P256,
    #[cfg(feature = "ecdsa-p384")]
    Credential::P384,
    #[cfg(feature = "ed25519")]
    Credential::Ed25519,
    #[cfg(feature = "rsa")]
    Credential::Rsa,
  ][0];
  for ours_client in [true, false] {
    for reject in [false, true] {
      let mut ours = crate::default_provider();
      let mut peer = rustls::crypto::aws_lc_rs::default_provider();
      // This case tests early-data policy. Offer the same first group so a
      // minimal P-256/P-384 build does not introduce an unrelated HRR rejection.
      let group = ours.kx_groups[0].name();
      ours.kx_groups.truncate(1);
      peer.kx_groups.retain(|candidate| candidate.name() == group);
      let (client, server) = if ours_client { (ours, peer) } else { (peer, ours) };
      let (mut client, mut server) = tls::configs(client, server, credential, true);
      assert!(!client.enable_early_data);
      assert_eq!(server.max_early_data_size, 0);
      client.enable_early_data = true;
      server.max_early_data_size = 1024;
      let (client_config, server_config) = (Arc::new(client), Arc::new(server));
      let (mut client, mut server) = tls::connections_from_configs(client_config.clone(), server_config.clone());
      tls::drive(&mut client, &mut server).expect("full handshake establishes PSK");
      assert_eq!(client.handshake_kind(), Some(HandshakeKind::Full));
      let (mut client, mut server) = tls::connections_from_configs(client_config, server_config);
      if let Connection::Client(client) = &mut client {
        let mut early = client.early_data().expect("explicitly enabled early data");
        early
          .write_all(b"replay-sensitive request")
          .expect("bounded early data");
      }
      if reject && let Connection::Server(server) = &mut server {
        server.reject_early_data();
      }
      tls::drive(&mut client, &mut server).expect("resumed handshake");
      assert_eq!(client.handshake_kind(), Some(HandshakeKind::Resumed));
      if let Connection::Client(client) = &client {
        assert_eq!(client.is_early_data_accepted(), !reject);
      }
      if let Connection::Server(server) = &mut server {
        if reject {
          assert!(server.early_data().is_none());
        } else {
          let mut received = Vec::new();
          server
            .early_data()
            .expect("accepted early data")
            .read_to_end(&mut received)
            .expect("read early data");
          assert_eq!(received, b"replay-sensitive request");
        }
      }
      tls::application_exchange(&mut client, &mut server);
    }
  }
}
