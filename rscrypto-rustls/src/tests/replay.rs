//! TLS 1.3 handshakes that draw from scoped generator streams.

use alloc::sync::Arc;

use rscrypto::ChaCha20Rng;
use rustls::crypto::CryptoProvider;
use rustls::pki_types::ServerName;
use rustls::{ClientConfig, Connection, Error, ServerConfig};

use super::tls::{Credential, Peer, configs, transfer};
use crate::replay::{ScopedRandom, Stream, scope};

static RANDOM: ScopedRandom = ScopedRandom;
static CLIENT: crate::ClientProvider = crate::ClientProvider::new(&RANDOM);
static KEYS: crate::sign::Provider = crate::sign::Provider::new(&RANDOM);

/// Client and server configurations whose every entropy request goes to [`ScopedRandom`].
fn scoped_configs() -> (Arc<ClientConfig>, Arc<ServerConfig>) {
  let provider = || CryptoProvider {
    key_provider: &KEYS,
    ..CLIENT.provider()
  };
  let (client, server) = configs(provider(), provider(), Credential::Ed25519, false);
  (Arc::new(client), Arc::new(server))
}

fn server_name() -> ServerName<'static> {
  ServerName::try_from("localhost").expect("TLS fixture")
}

/// Every byte that the client and the server of connection `index` under `root` send in a full handshake.
///
/// Each rustls call runs inside the scope of its own side's stream.
fn handshake_transcripts(root: &ChaCha20Rng, index: u64) -> [Vec<u8>; 2] {
  let client_stream = Stream::new(root.child(b"client\0\0", index));
  let server_stream = Stream::new(root.child(b"server\0\0", index));
  let (client_config, server_config) = scoped_configs();
  let mut hello = Vec::new();
  let client = scope(&client_stream, || {
    client_config.connect(server_name()).build(&mut hello)
  })
  .expect("ClientHello");
  let server = scope(&server_stream, || Connection::new(server_config)).expect("server connection");
  let (mut client, mut server) = (Peer::new(client, hello), Peer::new(server, Vec::new()));
  let mut transcripts = [Vec::new(), Vec::new()];
  let mut flights = 0_u8;
  while client.connection.is_handshaking() || server.connection.is_handshaking() || !client.output.is_empty() {
    flights = flights.strict_add(1);
    assert!(flights <= 16, "the handshake exceeded its flight count");
    transcripts[0].extend_from_slice(&client.output);
    scope(&server_stream, || transfer(&mut client, &mut server, |_, _| Ok(()))).expect("client flight");
    transcripts[1].extend_from_slice(&server.output);
    scope(&client_stream, || transfer(&mut server, &mut client, |_, _| Ok(()))).expect("server flight");
  }
  transcripts
}

#[test]
fn scoped_handshakes_replay_byte_for_byte() {
  let root = ChaCha20Rng::from_seed([0x52; ChaCha20Rng::SEED_SIZE]);
  let first = handshake_transcripts(&root, 0);
  let second = handshake_transcripts(&root, 1);
  // The ClientHello random and key shares come from each connection's stream, so another stream changes them.
  assert_ne!(first[0], second[0], "connection 1 has its own client stream");
  assert_ne!(first[1], second[1], "connection 1 has its own server stream");
  assert_eq!(handshake_transcripts(&root, 0), first, "connection 0 replays");
  assert_eq!(handshake_transcripts(&root, 1), second, "connection 1 replays");
}

#[test]
fn a_connection_built_outside_a_scope_fails() {
  let (client_config, _) = scoped_configs();
  let result = client_config.connect(server_name()).build(&mut Vec::new());
  assert!(matches!(result, Err(Error::FailedToGetRandomBytes)), "{result:?}");
}
