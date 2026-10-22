use alloc::sync::Arc;

use rustls::crypto::CipherSuite;
use rustls::crypto::CryptoProvider;
use rustls::crypto::TicketProducer;
use rustls::crypto::kx::NamedGroup;
use rustls::enums::ProtocolVersion;
use rustls::{ClientSide, Error, HandshakeKind, ServerSide};

use super::tls::{self, Credential};

fn select(mut provider: CryptoProvider, group: NamedGroup, suite: CipherSuite) -> CryptoProvider {
  provider
    .kx_groups
    .to_mut()
    .retain(|supported| supported.name() == group);
  provider
    .tls13_cipher_suites
    .to_mut()
    .retain(|supported| supported.common.suite == suite);
  assert_eq!(provider.kx_groups.len(), 1);
  assert_eq!(provider.tls13_cipher_suites.len(), 1);
  provider
}

fn exercise(client: &mut tls::Peer<ClientSide>, server: &mut tls::Peer<ServerSide>) -> [u8; 64] {
  let label = b"rscrypto provider interop";
  let context = Some(&b"context"[..]);
  assert_eq!(client.alpn_protocol().map(AsRef::as_ref), Some(&b"provider-test"[..]));
  assert_eq!(
    server.alpn_protocol().map(AsRef::as_ref),
    client.alpn_protocol().map(AsRef::as_ref)
  );
  let exported = client.export(label, context);
  assert_eq!(exported, server.export(label, context));
  assert_ne!(exported, client.export(label, Some(b"other")));
  assert_ne!(exported, client.export(b"other", context));
  tls::application_exchange(client, server);
  client
    .connection
    .refresh_traffic_keys(&mut client.output)
    .expect("client KeyUpdate");
  server
    .connection
    .refresh_traffic_keys(&mut server.output)
    .expect("server KeyUpdate");
  tls::application_exchange(client, server);
  // KeyUpdate changes traffic keys, not the handshake exporter secret.
  assert_eq!(exported, client.export(label, context));
  assert_eq!(exported, server.export(label, context));
  client.connection.send_close_notify(&mut client.output).expect("close");
  server.connection.send_close_notify(&mut server.output).expect("close");
  tls::drive(client, server).expect("close exchange");
  assert!(client.closed);
  assert!(server.closed);
  exported
}

fn profile_matrix(credential: Credential) {
  type Peer = (fn() -> CryptoProvider, fn() -> Result<Arc<dyn TicketProducer>, Error>);
  let peers: [Peer; 2] = [
    (
      (|| rustls_ring::DEFAULT_TLS13_PROVIDER),
      (|| rustls_ring::DEFAULT_TLS13_PROVIDER.ticketer_factory.ticketer()),
    ),
    (
      (|| rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER),
      (|| rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER.ticketer_factory.ticketer()),
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
    for group in tls::kx_groups() {
      if !peer_provider().kx_groups.iter().any(|peer| peer.name() == group.name()) {
        continue; // ring has no hybrid group; AWS-LC covers it in both roles.
      }
      for suite in crate::cipher_suite::ALL_CIPHER_SUITES {
        for ours_client in [true, false] {
          for stateless in [false, true] {
            if stateless && !ours_client && !cfg!(all(feature = "aes-gcm", feature = "getrandom")) {
              continue;
            }
            let ours = select(tls::provider(), group.name(), suite.common.suite);
            let peer = select(peer_provider(), group.name(), suite.common.suite);
            let (client, server) = if ours_client { (ours, peer) } else { (peer, ours) };
            let (client, mut server) = tls::configs(client, server, credential, true);
            assert!(!client.enable_early_data);
            assert_eq!(server.max_early_data_size, 0);
            if stateless {
              server.ticketer = Some(if ours_client {
                peer_ticketer().expect("independent ticketer")
              } else {
                #[cfg(all(feature = "aes-gcm", feature = "getrandom"))]
                {
                  crate::ticket::ticketer().expect("rscrypto ticketer")
                }
                #[cfg(not(all(feature = "aes-gcm", feature = "getrandom")))]
                {
                  continue;
                }
              });
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
                    suite.common.suite
                  )
                })
                .expect("profile handshake");
              for connection in [&*client, &*server] {
                assert_eq!(connection.handshake_kind(), Some(expected));
                assert_eq!(connection.protocol_version(), Some(ProtocolVersion::TLSv1_3));
                assert_eq!(
                  connection.negotiated_cipher_suite().expect("suite").suite(),
                  suite.common.suite
                );
                assert_eq!(
                  connection.negotiated_key_exchange_group().expect("group").name(),
                  group.name()
                );
                let identity = match connection.peer_identity().expect("mutual authentication").identity() {
                  rustls::crypto::Identity::X509(identity) => Some(identity),
                  _ => None,
                }
                .expect("X.509 identity");
                assert!(identity.intermediates.is_empty());
                assert_eq!(identity.end_entity.as_ref(), tls::credentials(credential).1);
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
      let mut ours = tls::provider();
      let mut peer = rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER;
      // This case tests early-data policy. Offer the same first group so a
      // minimal P-256/P-384 build does not introduce an unrelated HRR rejection.
      let group = ours.kx_groups[0].name();
      ours.kx_groups.to_mut().truncate(1);
      peer.kx_groups.to_mut().retain(|candidate| candidate.name() == group);
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
      let mut rejected = (*server_config).clone();
      if reject {
        rejected.max_early_data_size = 0;
      }
      let (mut client, mut server) = tls::connections_from_configs(client_config, Arc::new(rejected));
      let early = b"replay-sensitive request";
      assert_eq!(
        client
          .connection
          .early_data()
          .expect("explicitly enabled early data")
          .write(early.as_slice().into(), &mut client.output)
          .expect("bounded early data"),
        early.len()
      );
      tls::drive(&mut client, &mut server).expect("resumed handshake");
      assert_eq!(client.handshake_kind(), Some(HandshakeKind::Resumed));
      assert_eq!(client.connection.side().is_early_data_accepted(), !reject);
      if reject {
        assert!(server.early.is_empty());
      } else {
        assert_eq!(server.early, early);
      }
      tls::application_exchange(&mut client, &mut server);
    }
  }
}
