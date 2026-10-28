//! TLS 1.2 handshakes, resumption, refusal, and downgrade protection against independent peers.

use alloc::{borrow::Cow, sync::Arc, vec, vec::Vec};

use rustls::crypto::kx::NamedGroup;
use rustls::crypto::{CipherSuite, CryptoProvider, SignatureScheme, TicketProducer, WebPkiSupportedAlgorithms};
use rustls::enums::ProtocolVersion;
use rustls::error::{PeerIncompatible, PeerMisbehaved};
use rustls::{ClientSide, Error, HandshakeKind, ServerSide, Tls12CipherSuite};

use super::tls::{self, Credential};
use crate::cipher_suite::ALL_TLS12_CIPHER_SUITES;

type Ticketer = fn() -> Result<Arc<dyn TicketProducer>, Error>;
type Independent = (&'static str, fn() -> CryptoProvider, Ticketer);
type Stack = (&'static str, fn() -> CryptoProvider);

// This crate's stateless ticketer needs OS entropy.
#[cfg(all(feature = "aes-gcm", feature = "getrandom"))]
const OURS_TICKETER: Option<Ticketer> = Some(crate::ticket::ticketer);
#[cfg(not(all(feature = "aes-gcm", feature = "getrandom")))]
const OURS_TICKETER: Option<Ticketer> = None;

const PEERS: [Independent; 2] = [
  (
    "ring",
    || rustls_ring::DEFAULT_TLS12_PROVIDER,
    || rustls_ring::DEFAULT_PROVIDER.ticketer_factory.ticketer(),
  ),
  (
    "aws-lc-rs",
    || rustls_aws_lc_rs::DEFAULT_TLS12_PROVIDER,
    || rustls_aws_lc_rs::DEFAULT_PROVIDER.ticketer_factory.ticketer(),
  ),
];

const CLASSICAL: [NamedGroup; 3] = [NamedGroup::X25519, NamedGroup::secp256r1, NamedGroup::secp384r1];

/// This crate's provider as a caller enables TLS 1.2: its suites after TLS 1.3.
fn ours() -> CryptoProvider {
  let mut provider = tls::provider();
  assert!(provider.tls12_cipher_suites.is_empty(), "TLS 1.2 is opt-in");
  provider.tls12_cipher_suites = Cow::Borrowed(ALL_TLS12_CIPHER_SUITES);
  provider
}

/// `provider` restricted to TLS 1.2 with exactly `suite` and `group`.
fn tls12_only(mut provider: CryptoProvider, suite: CipherSuite, group: NamedGroup) -> CryptoProvider {
  provider.tls13_cipher_suites = Cow::Borrowed(&[]);
  provider
    .tls12_cipher_suites
    .to_mut()
    .retain(|candidate| candidate.common.suite == suite);
  provider
    .kx_groups
    .to_mut()
    .retain(|candidate| candidate.name() == group);
  assert_eq!(provider.tls12_cipher_suites.len(), 1);
  assert_eq!(provider.kx_groups.len(), 1);
  provider
}

fn authenticates(suite: &Tls12CipherSuite, credential: Credential) -> bool {
  suite.sign.contains(&credential.scheme())
}

fn configs(
  client: CryptoProvider,
  server: CryptoProvider,
  credential: Credential,
) -> (rustls::ClientConfig, rustls::ServerConfig) {
  let (mut client, mut server) = tls::configs(client, server, credential, true);
  // TLS 1.2 callers must require the extended master secret (RFC 7627).
  client.require_ems = true;
  server.require_ems = true;
  (client, server)
}

/// Exporters agree and are bound to label and context; data flows both ways; both close.
fn exercise(client: &mut tls::Peer<ClientSide>, server: &mut tls::Peer<ServerSide>) -> [u8; 64] {
  let (label, context) = (b"rscrypto provider TLS 1.2", Some(&b"context"[..]));
  let exported = client.export(label, context);
  assert_eq!(exported, server.export(label, context));
  assert_ne!(exported, client.export(label, Some(b"other")));
  assert_ne!(exported, client.export(b"other", context));
  for _ in 0..2 {
    tls::application_exchange(client, server);
  }
  client.connection.send_close_notify(&mut client.output).expect("close");
  server.connection.send_close_notify(&mut server.output).expect("close");
  tls::drive(client, server).expect("close exchange");
  assert!(client.closed && server.closed);
  exported
}

fn matrix(credential: Credential) {
  let mut completed = 0_usize;
  for suite in ALL_TLS12_CIPHER_SUITES
    .iter()
    .copied()
    .filter(|suite| authenticates(suite, credential))
  {
    for (peer_name, peer_provider, peer_ticketer) in PEERS {
      for (group_index, group) in CLASSICAL.into_iter().enumerate() {
        for ours_client in [true, false] {
          // Every group runs with session IDs; one group covers each ticketer.
          for stateless in [false, true] {
            let ticketer = if ours_client {
              Some(peer_ticketer)
            } else {
              OURS_TICKETER
            };
            if stateless && (group_index != 0 || ticketer.is_none()) {
              continue;
            }
            let ours = tls12_only(ours(), suite.common.suite, group);
            let peer = tls12_only(peer_provider(), suite.common.suite, group);
            let (client, server) = if ours_client { (ours, peer) } else { (peer, ours) };
            let (client, mut server) = configs(client, server, credential);
            if stateless {
              server.ticketer = Some(ticketer.expect("selected above")().expect("stateless ticketer"));
              server.session_storage = Arc::new(rustls::server::NoServerSessionStorage {});
            }
            let (client_config, server_config) = (Arc::new(client), Arc::new(server));
            let mut previous = None;
            for expected in [HandshakeKind::Full, HandshakeKind::Resumed] {
              let context = format!(
                "{credential:?}/{:?}/{group:?}/{peer_name}/ours_client={ours_client}/stateless={stateless}/{expected:?}",
                suite.common.suite
              );
              let (mut client, mut server) =
                tls::connections_from_configs(client_config.clone(), server_config.clone());
              tls::drive(&mut client, &mut server)
                .map_err(|error| format!("{context}: {error}"))
                .expect("TLS 1.2 handshake");
              for connection in [&*client, &*server] {
                assert_eq!(connection.handshake_kind(), Some(expected), "{context}");
                assert_eq!(
                  connection.protocol_version(),
                  Some(ProtocolVersion::TLSv1_2),
                  "{context}"
                );
                assert_eq!(connection.extended_main_secret(), Some(true), "{context}");
                assert_eq!(
                  connection.negotiated_cipher_suite().expect("suite").suite(),
                  suite.common.suite
                );
                // An abbreviated TLS 1.2 handshake performs no key exchange.
                assert_eq!(
                  connection.negotiated_key_exchange_group().map(|group| group.name()),
                  (expected == HandshakeKind::Full).then_some(group),
                  "{context}"
                );
                let identity = match connection.peer_identity().expect("mutual authentication").identity() {
                  rustls::crypto::Identity::X509(identity) => Some(identity),
                  _ => None,
                }
                .expect("X.509 identity");
                assert_eq!(identity.end_entity.as_ref(), tls::credentials(credential).1);
              }
              let exported = exercise(&mut client, &mut server);
              assert_ne!(Some(exported), previous, "{context}: resumption derives fresh keys");
              previous = Some(exported);
              completed = completed.strict_add(1);
            }
          }
        }
      }
    }
  }
  // Suites × 2 peers × (3 groups × 2 roles + 2 ticketers) × full and resumed.
  let suites = ALL_TLS12_CIPHER_SUITES
    .iter()
    .filter(|suite| authenticates(suite, credential))
    .count();
  let ticketers: usize = if OURS_TICKETER.is_some() { 2 } else { 1 };
  assert_eq!(
    completed,
    suites.strict_mul(2).strict_mul(ticketers.strict_add(6)).strict_mul(2)
  );
  eprintln!("{credential:?}: {completed} TLS 1.2 full/resumed connections");
}

#[test]
fn tls12_inventory_and_prf_match_independent_peers() {
  // Wire identifiers, PRF hashes, key-block shapes, and record limits promised
  // by the profile, independent of the tables that build the suites.
  let expected = [
    (
      CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
      48,
      (32, 4, 8),
      1 << 24,
    ),
    (
      CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
      32,
      (16, 4, 8),
      1 << 24,
    ),
    (
      CipherSuite::TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
      32,
      (32, 12, 0),
      u64::MAX,
    ),
    (
      CipherSuite::TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
      48,
      (32, 4, 8),
      1 << 24,
    ),
    (
      CipherSuite::TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
      32,
      (16, 4, 8),
      1 << 24,
    ),
    (
      CipherSuite::TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
      32,
      (32, 12, 0),
      u64::MAX,
    ),
  ];
  assert_eq!(ALL_TLS12_CIPHER_SUITES.len(), expected.len());
  for (suite, (id, hash_len, shape, limit)) in ALL_TLS12_CIPHER_SUITES.iter().zip(expected) {
    assert_eq!(suite.common.suite, id);
    assert_eq!(suite.common.hash_provider.output_len(), hash_len);
    let actual = suite.aead_alg.key_block_shape();
    assert_eq!(
      (actual.enc_key_len, actual.fixed_iv_len, actual.explicit_nonce_len),
      shape
    );
    assert_eq!(suite.common.confidentiality_limit, limit);
    assert_eq!(suite.kx, rustls::crypto::kx::KeyExchangeAlgorithm::ECDHE);
    assert_eq!(suite.fips(), rustls::pki_types::FipsStatus::Unvalidated);
    let rsa = format!("{id:?}").contains("_RSA_");
    for scheme in suite.sign {
      let name = format!("{scheme:?}");
      assert_eq!(name.starts_with("RSA_"), rsa, "{id:?} must not admit {name}");
      assert!(!name.contains("SHA1") && !name.contains("NISTP521"), "{id:?}: {name}");
    }
    for (_, peer_provider, _) in PEERS {
      let peer = peer_provider()
        .tls12_cipher_suites
        .iter()
        .copied()
        .find(|peer| peer.common.suite == id)
        .expect("independent suite");
      for secret in [[0; 48], [0x0b; 48]] {
        let (ours, theirs) = (
          suite.prf_provider.new_secret(&secret),
          peer.prf_provider.new_secret(&secret),
        );
        for (label, seed) in [
          (&b"master secret"[..], &[0x5a; 64][..]),
          (&b""[..], &b"\x01"[..]),
          (&b"key expansion"[..], &[0; 200][..]),
        ] {
          for length in [1, 12, 48, 104, 1000] {
            let (mut actual, mut expected) = (vec![0; length], vec![0; length]);
            ours.prf(&mut actual, label, seed);
            theirs.prf(&mut expected, label, seed);
            assert_eq!(actual, expected, "{id:?} label={label:?} length={length}");
          }
        }
      }
    }
  }
}

#[test]
fn p256_tls12_profiles() {
  matrix(Credential::P256);
}

#[test]
fn p384_tls12_profiles() {
  matrix(Credential::P384);
}

#[test]
fn ed25519_tls12_profiles() {
  matrix(Credential::Ed25519);
}

#[test]
fn rsa_tls12_profiles() {
  matrix(Credential::Rsa);
}

#[test]
fn rsa2048_tls12_profiles() {
  matrix(Credential::Rsa2048);
}

/// A peer that verifies, and so advertises, only RSA PKCS#1 v1.5 signatures.
fn pkcs1_only(mut provider: CryptoProvider) -> CryptoProvider {
  let mapping = provider.signature_verification_algorithms.mapping();
  let all = mapping
    .iter()
    .flat_map(|(_, algorithms)| algorithms.iter().copied())
    .collect::<Vec<_>>();
  let pkcs1 = mapping
    .iter()
    .filter(|(scheme, _)| {
      matches!(
        *scheme,
        SignatureScheme::RSA_PKCS1_SHA256 | SignatureScheme::RSA_PKCS1_SHA384 | SignatureScheme::RSA_PKCS1_SHA512
      )
    })
    .copied()
    .collect::<Vec<_>>();
  assert_eq!(pkcs1.len(), 3);
  provider.signature_verification_algorithms =
    WebPkiSupportedAlgorithms::new(all.leak(), pkcs1.leak()).expect("PKCS#1 v1.5 schemes");
  provider
}

#[test]
fn rsa_keys_sign_pkcs1_for_tls12_peers_that_accept_nothing_else() {
  let suite = CipherSuite::TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256;
  for (peer_name, peer_provider, _) in PEERS {
    for ours_client in [true, false] {
      // As client, this crate signs CertificateVerify; as server, ServerKeyExchange.
      let ours = tls12_only(ours(), suite, NamedGroup::X25519);
      let peer = pkcs1_only(tls12_only(peer_provider(), suite, NamedGroup::X25519));
      let (client, server) = if ours_client { (ours, peer) } else { (peer, ours) };
      for kind in [Credential::Rsa, Credential::Rsa2048] {
        let (client, server) = configs(client.clone(), server.clone(), kind);
        let (mut client, mut server) = tls::connections_from_configs(Arc::new(client), Arc::new(server));
        tls::drive(&mut client, &mut server)
          .map_err(|error| format!("{peer_name}/{kind:?}/ours_client={ours_client}: {error}"))
          .expect("PKCS#1 v1.5 handshake");
        assert_eq!(client.protocol_version(), Some(ProtocolVersion::TLSv1_2));
        exercise(&mut client, &mut server);
      }
    }
  }
  // TLS 1.3 never signs PKCS#1 v1.5 (RFC 8446 §4.2.3), so the same peers fail there.
  let tls13 = [
    rustls_ring::DEFAULT_TLS13_PROVIDER,
    rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER,
  ];
  for peer in tls13 {
    for ours_client in [true, false] {
      let (client, server) = if ours_client {
        (tls::provider(), pkcs1_only(peer.clone()))
      } else {
        (pkcs1_only(peer.clone()), tls::provider())
      };
      let (client, server) = tls::configs(client, server, Credential::Rsa, true);
      let (mut client, mut server) = tls::connections_from_configs(Arc::new(client), Arc::new(server));
      assert!(
        tls::drive(&mut client, &mut server).is_err(),
        "TLS 1.3 must not authenticate with PKCS#1 v1.5 (ours_client={ours_client})"
      );
    }
  }
}

#[cfg(feature = "ml-kem")]
#[test]
fn hybrid_groups_never_negotiate_tls12() {
  for (peer_name, peer_provider, _) in PEERS {
    for ours_client in [true, false] {
      // Hybrid first, as in the default provider; the peer speaks only TLS 1.2.
      let ours = ours();
      assert_eq!(ours.kx_groups[0].name(), NamedGroup::X25519MLKEM768);
      let (client, server) = if ours_client {
        (ours, peer_provider())
      } else {
        (peer_provider(), ours)
      };
      let (client, server) = configs(client, server, Credential::P256);
      let (mut client, mut server) = tls::connections_from_configs(Arc::new(client), Arc::new(server));
      tls::drive(&mut client, &mut server)
        .map_err(|error| format!("{peer_name}/ours_client={ours_client}: {error}"))
        .expect("classical TLS 1.2 handshake");
      for connection in [&*client, &*server] {
        assert_eq!(connection.protocol_version(), Some(ProtocolVersion::TLSv1_2));
        let group = connection.negotiated_key_exchange_group().expect("group").name();
        assert!(CLASSICAL.contains(&group), "{peer_name}: {group:?}");
      }
    }
  }
}

#[cfg(all(feature = "ml-kem", feature = "getrandom"))]
#[test]
fn required_hybrid_refuses_tls12_even_with_tls12_suites() {
  for (peer_name, peer_provider, _) in PEERS {
    for ours_client in [true, false] {
      let mut required = crate::post_quantum_required_provider();
      required.tls12_cipher_suites = Cow::Borrowed(ALL_TLS12_CIPHER_SUITES);
      required.key_provider = super::key_provider();
      let (client, server) = if ours_client {
        (required, peer_provider())
      } else {
        (peer_provider(), required)
      };
      let (client, server) = configs(client, server, Credential::P256);
      let (mut client, mut server) = tls::connections_from_configs(Arc::new(client), Arc::new(server));
      let error = tls::drive(&mut client, &mut server).expect_err("no hybrid group exists in TLS 1.2");
      // The TLS 1.2 server finds no group usable with both peers' lists.
      assert!(
        matches!(error, Error::PeerIncompatible(PeerIncompatible::NoKxGroupsInCommon)),
        "{peer_name}/ours_client={ours_client}: {error}"
      );
    }
  }
}

/// Remove extension `kind` from the handshake message of type `message` that
/// begins the first record of `flight`, fixing the record, message, and
/// extension-block lengths. Later messages in the record are kept.
fn strip_extension(flight: &mut Vec<u8>, message: u8, kind: u16) {
  let read16 = |flight: &[u8], at: usize| usize::from(u16::from_be_bytes([flight[at], flight[at.strict_add(1)]]));
  let write16 = |flight: &mut [u8], at: usize, value: usize| {
    flight[at..at.strict_add(2)].copy_from_slice(&u16::try_from(value).expect("16-bit length").to_be_bytes());
  };
  assert_eq!(flight[0], 22, "handshake record");
  assert_eq!(flight[5], message, "first handshake message");
  // legacy_version and random, then session_id.
  let mut at = 9_usize.strict_add(2).strict_add(32);
  at = at.strict_add(1).strict_add(usize::from(flight[at]));
  if message == 1 {
    at = at.strict_add(2).strict_add(read16(flight, at)); // cipher_suites
    at = at.strict_add(1).strict_add(usize::from(flight[at])); // compression_methods
  } else {
    at = at.strict_add(3); // cipher_suite and compression_method
  }
  let extensions = at;
  let end = extensions.strict_add(2).strict_add(read16(flight, extensions));
  let mut cursor = extensions.strict_add(2);
  let mut found = None;
  while found.is_none() && cursor < end {
    let length = 4_usize.strict_add(read16(flight, cursor.strict_add(2)));
    if read16(flight, cursor) == usize::from(kind) {
      found = Some((cursor, length));
    }
    cursor = cursor.strict_add(length);
  }
  let (start, length) = found.expect("extension present");
  flight.drain(start..start.strict_add(length));
  let extensions_len = read16(flight, extensions).strict_sub(length);
  write16(flight, extensions, extensions_len);
  let message_len = u32::from_be_bytes([0, flight[6], flight[7], flight[8]]);
  let message_len = message_len.strict_sub(u32::try_from(length).expect("extension length"));
  flight[6..9].copy_from_slice(&message_len.to_be_bytes()[1..]);
  let record_len = read16(flight, 3).strict_sub(length);
  write16(flight, 3, record_len);
}

#[test]
fn downgrade_sentinel_rejects_a_stripped_supported_versions() {
  // RFC 8446 §4.1.3: a TLS 1.3 server that negotiates TLS 1.2 marks its
  // random, and a TLS 1.3 client rejects the mark.
  let servers: [Stack; 3] = [
    ("rscrypto", ours),
    ("ring", || rustls_ring::DEFAULT_PROVIDER),
    ("aws-lc-rs", || rustls_aws_lc_rs::DEFAULT_PROVIDER),
  ];
  let clients: [Stack; 2] = [("rscrypto", ours), ("aws-lc-rs", || rustls_aws_lc_rs::DEFAULT_PROVIDER)];
  for (client_name, client_provider) in clients {
    for (server_name, server_provider) in servers {
      if client_name != "rscrypto" && server_name != "rscrypto" {
        continue;
      }
      for strip in [false, true] {
        let (mut client, server) = configs(client_provider(), server_provider(), Credential::P256);
        // One ClientHello record, so its extensions can be edited in flight.
        client.max_fragment_size = None;
        let (mut client, mut server) = tls::connections_from_configs(Arc::new(client), Arc::new(server));
        let context = format!("{client_name} client, {server_name} server, strip={strip}");
        if !strip {
          tls::drive(&mut client, &mut server).expect(&context);
          assert_eq!(client.protocol_version(), Some(ProtocolVersion::TLSv1_3), "{context}");
          continue;
        }
        strip_extension(&mut client.output, 1, 0x002b);
        let error = tls::drive(&mut client, &mut server).expect_err(&context);
        assert!(
          matches!(
            error,
            Error::PeerMisbehaved(PeerMisbehaved::AttemptedDowngradeToTls12WhenTls13IsSupported)
          ),
          "{context}: {error}"
        );
      }
    }
  }
}

#[test]
fn tls12_never_derives_a_master_secret_without_ems() {
  let suite = CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256;
  for (peer_name, peer_provider, _) in PEERS {
    for (ours_client, require_ems) in [(true, true), (true, false), (false, true), (false, false)] {
      let ours = tls12_only(ours(), suite, NamedGroup::X25519);
      let peer = tls12_only(peer_provider(), suite, NamedGroup::X25519);
      let (client, server) = if ours_client { (ours, peer) } else { (peer, ours) };
      let (mut client, mut server) = configs(client, server, Credential::P256);
      // Only this provider's side may relax the requirement; the peer's PRF is legacy-capable.
      if ours_client {
        client.require_ems = require_ems;
        server.require_ems = false;
      } else {
        client.require_ems = false;
        server.require_ems = require_ems;
      }
      let (mut client, mut server) = tls::connections_from_configs(Arc::new(client), Arc::new(server));
      let error = if ours_client {
        tls::transfer(&mut client, &mut server, |_, _| Ok(())).expect("ClientHello");
        // An attacker removes the server's extended_master_secret acknowledgement.
        strip_extension(&mut server.output, 2, 0x0017);
        tls::transfer(&mut server, &mut client, |_, _| Ok(()))
          .and_then(|_| tls::drive(&mut client, &mut server))
          .expect_err("a stripped acknowledgement never completes")
      } else {
        // An attacker removes the client's extended_master_secret request.
        strip_extension(&mut client.output, 1, 0x0017);
        tls::drive(&mut client, &mut server).expect_err("a stripped request never completes")
      };
      // With or without `require_ems`, this provider refuses the legacy master
      // secret before using it, as RFC 7627 §5.2 allows.
      assert!(
        matches!(
          error,
          Error::PeerIncompatible(PeerIncompatible::ExtendedMainSecretExtensionRequired)
        ),
        "{peer_name}/ours_client={ours_client}/require_ems={require_ems}: {error}"
      );
    }
  }
}

#[cfg(feature = "quinn")]
#[test]
fn quic_rejects_providers_with_tls12_suites() {
  let (client, server) = tls::configs(ours(), ours(), Credential::P256, false);
  for result in [
    crate::quinn::client_config(client).map(drop),
    crate::quinn::server_config(server).map(drop),
  ] {
    assert!(matches!(result, Err(Error::General(message)) if message == "QUIC requires a TLS 1.3-only provider"));
  }
}
