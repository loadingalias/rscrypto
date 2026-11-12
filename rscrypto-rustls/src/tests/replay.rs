//! TLS 1.3 handshakes that draw from scoped generator streams.

use alloc::sync::Arc;
use core::time::Duration;
use std::sync::Mutex;

use rscrypto::ChaCha20Rng;
use rustls::crypto::{CryptoProvider, TicketProducer};
use rustls::pki_types::ServerName;
use rustls::{ClientConfig, Connection, Error, ServerConfig};

use super::tls::{Credential, FixtureTime, Peer, configs, transfer};
use crate::replay::{ScopedRandom, Stream, scope};
use crate::ticket::ticketer_with;

static RANDOM: ScopedRandom = ScopedRandom;
static CLIENT: crate::ClientProvider = crate::ClientProvider::new(&RANDOM);
static KEYS: crate::sign::Provider = crate::sign::Provider::new(&RANDOM);

/// Client and server configurations whose every entropy request goes to [`ScopedRandom`].
fn scoped_configs() -> (ClientConfig, ServerConfig) {
  let provider = || CryptoProvider {
    key_provider: &KEYS,
    ..CLIENT.provider()
  };
  configs(provider(), provider(), Credential::Ed25519, false)
}

fn server_name() -> ServerName<'static> {
  ServerName::try_from("localhost").expect("TLS fixture")
}

/// The provider's ticketer, recording each ticket that it issues.
#[derive(Debug)]
struct RecordedTickets {
  ticketer: Arc<dyn TicketProducer>,
  issued: Mutex<Vec<Vec<u8>>>,
}

impl TicketProducer for RecordedTickets {
  fn encrypt(&self, plain: &[u8]) -> Option<Vec<u8>> {
    let ticket = self.ticketer.encrypt(plain)?;
    self.issued.lock().expect("ticket record").push(ticket.clone());
    Some(ticket)
  }

  fn decrypt(&self, cipher: &[u8]) -> Option<Vec<u8>> {
    self.ticketer.decrypt(cipher)
  }

  fn lifetime(&self) -> Duration {
    self.ticketer.lifetime()
  }
}

/// What one full handshake sends.
#[derive(Debug, PartialEq)]
struct Handshake {
  /// Every byte that the client and the server send.
  transcripts: [Vec<u8>; 2],
  /// Each session ticket that the server issues.
  tickets: Vec<Vec<u8>>,
}

/// Run one full handshake between the client and server children of `root` with the given indexes.
///
/// Each rustls call runs inside the scope of its own side's stream.
fn handshake(root: &ChaCha20Rng, client_index: u64, server_index: u64) -> Handshake {
  let client_stream = Stream::new(root.child(b"client\0\0", client_index));
  let server_stream = Stream::new(root.child(b"server\0\0", server_index));
  let (client_config, mut server_config) = scoped_configs();
  // The ticketer draws its first key from the server's stream.
  let ticketer = scope(&server_stream, || ticketer_with(&RANDOM, Arc::new(FixtureTime))).expect("ticketer");
  let recorded = Arc::new(RecordedTickets {
    ticketer,
    issued: Mutex::default(),
  });
  server_config.ticketer = Some(recorded.clone());
  let (client_config, server_config) = (Arc::new(client_config), Arc::new(server_config));

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
  let tickets = core::mem::take(&mut *recorded.issued.lock().expect("ticket record"));
  Handshake { transcripts, tickets }
}

fn root() -> ChaCha20Rng {
  ChaCha20Rng::from_seed([0x52; ChaCha20Rng::SEED_SIZE])
}

#[test]
fn scoped_handshakes_replay_byte_for_byte() {
  let root = root();
  let first = handshake(&root, 0, 0);
  let second = handshake(&root, 1, 1);
  // The ClientHello random and key shares come from each connection's stream, so another stream changes them.
  assert_ne!(
    first.transcripts[0], second.transcripts[0],
    "connection 1 has its own client stream"
  );
  assert_ne!(
    first.transcripts[1], second.transcripts[1],
    "connection 1 has its own server stream"
  );
  assert_eq!(handshake(&root, 0, 0), first, "connection 0 replays");
  assert_eq!(handshake(&root, 1, 1), second, "connection 1 replays");
}

#[test]
fn scoped_ticketers_issue_replayable_tickets() {
  let root = root();
  let first = handshake(&root, 0, 0).tickets;
  assert!(!first.is_empty(), "the server issues tickets");
  assert_eq!(
    handshake(&root, 0, 0).tickets,
    first,
    "the same streams issue the same tickets"
  );
  assert_ne!(
    handshake(&root, 0, 1).tickets,
    first,
    "another server stream issues other tickets"
  );
}

#[test]
fn a_connection_built_outside_a_scope_fails() {
  let (client_config, _) = scoped_configs();
  let result = Arc::new(client_config).connect(server_name()).build(&mut Vec::new());
  assert!(matches!(result, Err(Error::FailedToGetRandomBytes)), "{result:?}");
}
