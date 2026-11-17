//! TLS 1.3 handshakes that draw from scoped generator streams.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;
use std::sync::Mutex;

use rscrypto::ChaCha20Rng;
use rustls::crypto::{CryptoProvider, SecureRandom as _, TicketProducer};
use rustls::pki_types::{ServerName, UnixTime};
use rustls::time_provider::TimeProvider;
use rustls::{ClientConfig, Connection, Error, ServerConfig};

use super::tls::{Credential, Peer, configs_with_time, transfer};
use crate::replay::{ScopedRandom, Stream, scope};
use crate::ticket::ticketer_with_rng;

static RANDOM: ScopedRandom = ScopedRandom;
static CLIENT: crate::ClientProvider = crate::ClientProvider::new(&RANDOM);
static KEYS: crate::sign::Provider = crate::sign::Provider::scoped();

/// The fixture clock's reading: 2026-10-22 UTC, inside the fixture certificates' validity.
const T0: u64 = 1_792_627_200;
/// The ticket key rotation period of [`crate::ticket`].
const ROTATION: u64 = 6 * 60 * 60;

/// A clock that the test moves. The client and the server read it.
#[derive(Debug)]
struct StepTime(AtomicU64);

impl TimeProvider for StepTime {
  fn current_time(&self) -> Option<UnixTime> {
    Some(UnixTime::since_unix_epoch(Duration::from_secs(
      self.0.load(Ordering::Relaxed),
    )))
  }
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

/// The configurations of one client and one server, which any number of connections can share.
struct Endpoints {
  client: Arc<ClientConfig>,
  server: Arc<ServerConfig>,
  tickets: Arc<RecordedTickets>,
  time: Arc<StepTime>,
}

/// Build the endpoints at clock reading `now`.
///
/// The server key loads in a scope of the `setup` child of `root`, where an RSA key seeds its
/// own blinding generator. The ticketer owns the `tickets` child with index `ticketer`. The client
/// stores no sessions, so each connection is a full handshake.
fn endpoints(root: &ChaCha20Rng, credential: Credential, ticketer: u64, now: u64) -> Endpoints {
  let setup = Stream::new(root.child(b"setup\0\0\0", 0));
  let time = Arc::new(StepTime(AtomicU64::new(now)));
  let provider = || CryptoProvider {
    key_provider: &KEYS,
    ..CLIENT.provider()
  };
  let (mut client, mut server) = scope(&setup, || {
    configs_with_time(provider(), provider(), credential, false, time.clone())
  });
  client.resumption = rustls::client::Resumption::disabled();
  let tickets = Arc::new(RecordedTickets {
    ticketer: ticketer_with_rng(root.child(b"tickets\0", ticketer), time.clone()).expect("ticketer"),
    issued: Mutex::default(),
  });
  server.ticketer = Some(tickets.clone());
  Endpoints {
    client: Arc::new(client),
    server: Arc::new(server),
    tickets,
    time,
  }
}

/// The client and server streams of connection `index` under `root`.
fn connection_streams(root: &ChaCha20Rng, index: u64) -> (Stream, Stream) {
  (
    Stream::new(root.child(b"client\0\0", index)),
    Stream::new(root.child(b"server\0\0", index)),
  )
}

/// Run one full handshake over `endpoints`. Each rustls call runs inside the scope of its own
/// side's stream.
fn handshake(endpoints: &Endpoints, client_stream: &Stream, server_stream: &Stream) -> Handshake {
  let mut hello = Vec::new();
  let client = scope(client_stream, || {
    endpoints.client.connect(server_name()).build(&mut hello)
  })
  .expect("ClientHello");
  let server = scope(server_stream, || Connection::new(endpoints.server.clone())).expect("server connection");
  let (mut client, mut server) = (Peer::new(client, hello), Peer::new(server, Vec::new()));
  let mut transcripts = [Vec::new(), Vec::new()];
  let mut flights = 0_u8;
  while client.connection.is_handshaking() || server.connection.is_handshaking() || !client.output.is_empty() {
    flights = flights.strict_add(1);
    assert!(flights <= 16, "the handshake exceeded its flight count");
    transcripts[0].extend_from_slice(&client.output);
    scope(server_stream, || transfer(&mut client, &mut server, |_, _| Ok(()))).expect("client flight");
    transcripts[1].extend_from_slice(&server.output);
    scope(client_stream, || transfer(&mut server, &mut client, |_, _| Ok(()))).expect("server flight");
  }
  let tickets = core::mem::take(&mut *endpoints.tickets.issued.lock().expect("ticket record"));
  Handshake { transcripts, tickets }
}

/// One full Ed25519 handshake of connection `index` over new endpoints with ticketer child
/// `ticketer`.
fn fresh_handshake(root: &ChaCha20Rng, index: u64, ticketer: u64) -> Handshake {
  let (client, server) = connection_streams(root, index);
  handshake(&endpoints(root, Credential::Ed25519, ticketer, T0), &client, &server)
}

/// The next 32 bytes of `stream`, which show how far its connection drew.
fn next(stream: &Stream) -> [u8; 32] {
  let mut bytes = [0; 32];
  scope(stream, || ScopedRandom.fill(&mut bytes)).expect("draw");
  bytes
}

fn root() -> ChaCha20Rng {
  ChaCha20Rng::from_seed([0x52; ChaCha20Rng::SEED_SIZE])
}

/// The clock reading of connection `index` in a shared run: connection 2 comes one rotation
/// period later, so its tickets rotate the ticketer's key.
fn reading(index: u64) -> u64 {
  if index == 2 { T0 + ROTATION } else { T0 }
}

/// Connections 0 to 2 over one set of RSA-2048 endpoints. Each result holds the handshake and the
/// next bytes of the connection's client and server streams.
#[cfg(feature = "rsa")]
fn shared_run(root: &ChaCha20Rng) -> Vec<(Handshake, [u8; 32], [u8; 32])> {
  let shared = endpoints(root, Credential::Rsa2048, 0, T0);
  let run: Vec<_> = (0..3)
    .map(|index| {
      shared.time.0.store(reading(index), Ordering::Relaxed);
      let (client, server) = connection_streams(root, index);
      let handshake = handshake(&shared, &client, &server);
      (handshake, next(&client), next(&server))
    })
    .collect();
  // A ticket starts with its 12-byte nonce, a count of the tickets of one key. So connection 2's
  // first ticket has nonce 0 only under a rotated key.
  let first_nonce = |index: usize| run[index].0.tickets.first().expect("the server issues tickets")[..12].to_vec();
  assert_ne!(first_nonce(1), [0; 12], "connection 1 continues the first key");
  assert_eq!(first_nonce(2), [0; 12], "connection 2 rotates the key");
  run
}

#[test]
fn scoped_handshakes_replay_byte_for_byte() {
  let root = root();
  let first = fresh_handshake(&root, 0, 0);
  let second = fresh_handshake(&root, 1, 0);
  // The ClientHello random and key shares come from each connection's stream, so another stream changes them.
  assert_ne!(
    first.transcripts[0], second.transcripts[0],
    "connection 1 has its own client stream"
  );
  assert_ne!(
    first.transcripts[1], second.transcripts[1],
    "connection 1 has its own server stream"
  );
  assert_eq!(fresh_handshake(&root, 0, 0), first, "connection 0 replays");
  assert_eq!(fresh_handshake(&root, 1, 0), second, "connection 1 replays");
}

#[test]
fn scoped_ticketers_issue_replayable_tickets() {
  let root = root();
  let first = fresh_handshake(&root, 0, 0).tickets;
  assert!(!first.is_empty(), "the server issues tickets");
  assert_eq!(
    fresh_handshake(&root, 0, 0).tickets,
    first,
    "the same generators issue the same tickets"
  );
  assert_ne!(
    fresh_handshake(&root, 0, 1).tickets,
    first,
    "another ticketer generator issues other tickets"
  );
}

/// A shared owner never draws from a connection's stream, so each connection's stream position
/// depends only on its own operations. In the shared run, connection 0 builds the RSA key's
/// signing workspace, connection 1 reuses it, and connection 2 also rotates the ticket key. Alone,
/// each connection gets new owners at its own clock reading, which build a workspace and rotate
/// nothing. A stream draws the same next bytes in both cases only if no blinding refresh and no key
/// rotation drew from it.
#[cfg(feature = "rsa")]
#[test]
fn a_connection_stream_depends_only_on_its_own_operations() {
  let root = root();
  let run = shared_run(&root);
  for index in 0..3 {
    let alone = endpoints(&root, Credential::Rsa2048, 0, reading(index));
    let (client, server) = connection_streams(&root, index);
    handshake(&alone, &client, &server);
    let (_, client_next, server_next) = &run[usize::try_from(index).expect("index")];
    assert_eq!(&next(&client), client_next, "connection {index} client");
    assert_eq!(&next(&server), server_next, "connection {index} server");
  }
}

/// The same generators and the same calls in the same order replay the whole run, including the
/// tickets under the rotated key.
#[cfg(feature = "rsa")]
#[test]
fn shared_endpoints_replay_the_whole_run() {
  assert_eq!(shared_run(&root()), shared_run(&root()));
}

/// A shared owner that draws from [`ScopedRandom`] fails at once, so a wrong configuration never
/// draws from a connection's stream.
#[cfg(feature = "rsa")]
#[test]
fn shared_owners_refuse_scoped_random() {
  use rustls::crypto::{KeyProvider as _, SignatureScheme};
  use rustls::pki_types::{PrivateKeyDer, PrivatePkcs8KeyDer};

  static UNSCOPED: crate::sign::Provider = crate::sign::Provider::new(&RANDOM);
  let sign = |provider: &crate::sign::Provider| {
    provider
      .load_private_key(PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(
        include_bytes!("../../tests/vectors/credentials/rsa2048-leaf-key.der").to_vec(),
      )))
      .expect("RSA-2048 key")
      .choose_scheme(&[SignatureScheme::RSA_PSS_SHA256])
      .expect("RSA-PSS")
      .sign(b"message")
  };
  scope(&Stream::new(root().child(b"server\0\0", 0)), || {
    assert!(
      matches!(
        crate::ticket::ticketer_with(&RANDOM, Arc::new(StepTime(AtomicU64::new(T0)))),
        Err(Error::FailedToGetRandomBytes)
      ),
      "a ticketer over ScopedRandom has no first key"
    );
    assert!(
      matches!(sign(&UNSCOPED), Err(Error::General(message)) if message == "TLS signing failed"),
      "an RSA key over ScopedRandom refuses its blinding refresh"
    );
    sign(&KEYS).expect("a scoped RSA key refreshes from its own generator");
  });
}

#[test]
fn a_connection_built_outside_a_scope_fails() {
  let (client, _) = configs_with_time(
    CLIENT.provider(),
    CryptoProvider {
      key_provider: &KEYS,
      ..CLIENT.provider()
    },
    Credential::Ed25519,
    false,
    Arc::new(StepTime(AtomicU64::new(T0))),
  );
  let result = Arc::new(client).connect(server_name()).build(&mut Vec::new());
  assert!(matches!(result, Err(Error::FailedToGetRandomBytes)), "{result:?}");
}
