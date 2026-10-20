//! Bounded, lossless datagram transport for Quinn's production endpoints.

use alloc::{collections::VecDeque, sync::Arc};
use core::net::SocketAddr;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::time::Duration;
use std::time::Instant;

use bytes::{Bytes, BytesMut};
use quinn_proto::{Connection, ConnectionHandle, DatagramEvent, Endpoint, Event, TokenStore};

use super::{configs, oracle};
use crate::{Workload, credentials, group, suite};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetworkMode {
  Connect,
  Retry,
  AddressToken,
}

/// Counts real cache operations without replacing Quinn's token policy.
#[derive(Debug)]
struct Tokens {
  cache: quinn_proto::TokenMemoryCache,
  inserted: AtomicUsize,
  taken: AtomicUsize,
}

impl TokenStore for Tokens {
  fn insert(&self, name: &str, token: Bytes) {
    self.cache.insert(name, token);
    self.inserted.fetch_add(1, Ordering::Relaxed);
  }
  fn take(&self, name: &str) -> Option<Bytes> {
    let token = self.cache.take(name);
    if token.is_some() {
      self.taken.fetch_add(1, Ordering::Relaxed);
    }
    token
  }
}

pub struct NetworkFixture {
  client: quinn_proto::ClientConfig,
  server: Arc<quinn_proto::ServerConfig>,
  endpoints: [Arc<quinn_proto::EndpointConfig>; 2],
  tokens: Arc<Tokens>,
  mode: NetworkMode,
  cipher: String,
}

impl NetworkFixture {
  pub fn new(backend: &str, cipher: &str, mode: NetworkMode) -> Self {
    let fixture = Self::configured(backend, backend, cipher, mode);
    fixture.check(fixture.connect());
    // Exercise the same adapter and datagram pump with independent crypto peers.
    for (client, server) in [(backend, oracle(backend)), (oracle(backend), backend)] {
      let cross = Self::configured(client, server, cipher, mode);
      cross.check(cross.connect());
    }
    fixture
  }

  fn configured(client_backend: &str, server_backend: &str, cipher: &str, mode: NetworkMode) -> Self {
    let (client, server) = configs(client_backend, cipher);
    let server = if client_backend == server_backend {
      server
    } else {
      configs(server_backend, cipher).1
    };
    let tokens = Arc::new(Tokens {
      cache: quinn_proto::TokenMemoryCache::new(1, 1),
      inserted: AtomicUsize::new(0),
      taken: AtomicUsize::new(0),
    });
    let mut client = rscrypto_rustls::quinn::client_config((*client).clone()).expect("Quinn client config");
    client.token_store(tokens.clone());
    let mut server = rscrypto_rustls::quinn::server_config((*server).clone()).expect("Quinn server config");
    // The approved Quinn dependency enables its shipped BloomTokenLog. Without
    // that feature, its default NoneTokenLog rejects every NEW_TOKEN redemption.
    let mut validation = quinn_proto::ValidationTokenConfig::default();
    validation.sent(u32::from(mode == NetworkMode::AddressToken));
    server.validation_token_config(validation);
    let fixture = Self {
      client,
      server: Arc::new(server),
      tokens,
      mode,
      cipher: cipher.to_owned(),
      endpoints: [
        Arc::new(rscrypto_rustls::quinn::endpoint_config().expect("client endpoint keys")),
        Arc::new(rscrypto_rustls::quinn::endpoint_config().expect("server endpoint keys")),
      ],
    };
    if mode == NetworkMode::AddressToken {
      // Obtain one real token. The one-entry cache retains the newest token
      // after each connection; TLS session resumption remains disabled.
      let seed = fixture.connect_inner(false);
      fixture.check(seed);
      assert_eq!(fixture.tokens.inserted.load(Ordering::Relaxed), 1);
      assert_eq!(fixture.tokens.taken.load(Ordering::Relaxed), 0);
    }
    fixture
  }

  /// Fresh endpoints and connections, lossless UDP copies, cryptography and
  /// immediate post-handshake control traffic. No wall-clock wait or sockets.
  pub fn connect(&self) -> NetworkPair {
    self.connect_inner(self.mode == NetworkMode::AddressToken)
  }

  fn connect_inner(&self, token: bool) -> NetworkPair {
    let before = (
      self.tokens.inserted.load(Ordering::Relaxed),
      self.tokens.taken.load(Ordering::Relaxed),
    );
    let mut pair = NetworkPair::new(self);
    pair.handshake();
    assert_eq!(
      pair.retries,
      usize::from(self.mode == NetworkMode::Retry),
      "exact Retry path"
    );
    assert_eq!(
      pair.validated,
      Some(token || self.mode == NetworkMode::Retry),
      "accepted address validation"
    );
    assert_eq!(
      self.tokens.taken.load(Ordering::Relaxed).strict_sub(before.1),
      usize::from(token),
      "token consumed"
    );
    // At this pinned revision, an already validated Initial queues one token
    // in Connection::new; receipt of the Handshake packet queues another in
    // on_path_validated. The unvalidated seed connection issues only the latter.
    let issued = if self.mode == NetworkMode::AddressToken {
      usize::from(token).strict_add(1)
    } else {
      0
    };
    assert_eq!(
      self.tokens.inserted.load(Ordering::Relaxed).strict_sub(before.0),
      issued,
      "exact production token issuance"
    );
    pair
  }

  fn check(&self, pair: NetworkPair) {
    let mut exported = [[0; 32]; 2];
    for (side, peer) in pair.peers.iter().enumerate() {
      let session = peer.connection.as_ref().expect("connected").1.crypto_session();
      assert!(!session.is_handshaking());
      let data = session
        .handshake_data()
        .expect("handshake data")
        .downcast::<rscrypto_rustls::quinn::HandshakeData>()
        .expect("adapter data");
      assert_eq!(data.protocol, credentials::ALPN);
      assert_eq!(data.cipher_suite, suite(&self.cipher));
      assert_eq!(data.key_exchange_group, Some(group("X25519")));
      assert_eq!(data.handshake_kind, Some(rustls::HandshakeKind::Full));
      if side == 0 {
        let chain = session
          .peer_identity()
          .expect("server identity")
          .downcast::<Vec<rustls::pki_types::CertificateDer<'static>>>()
          .expect("certificate chain");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].as_ref(), credentials::der("p256", Workload::ServerAuth).leaf);
      } else {
        assert!(session.peer_identity().is_none());
      }
      session
        .export_keying_material(&mut exported[side], b"quinn-bench", b"matched")
        .expect("exporter");
    }
    assert_eq!(exported[0], exported[1], "independent endpoint exporters");
    assert!(pair.sent > 0 && pair.sent < 64, "bounded complete handshake");
    assert!(pair.queue.is_empty(), "all immediate datagrams delivered");
  }
}

struct Peer {
  endpoint: Endpoint,
  connection: Option<(ConnectionHandle, Connection)>,
  address: SocketAddr,
  connected: bool,
}

pub struct NetworkPair {
  peers: [Peer; 2],
  now: Instant,
  queue: VecDeque<(usize, BytesMut)>,
  output: Vec<u8>,
  sent: usize,
  retry: bool,
  retries: usize,
  validated: Option<bool>,
}

impl NetworkPair {
  fn new(fixture: &NetworkFixture) -> Self {
    let addresses = [
      "127.0.0.1:44330".parse().expect("address"),
      "127.0.0.1:44331".parse().expect("address"),
    ];
    let now = Instant::now();
    let mut client = Endpoint::new(fixture.endpoints[0].clone(), None, false);
    let connection = client
      .connect(now, fixture.client.clone(), addresses[1], "localhost")
      .expect("connect");
    let server = Endpoint::new(fixture.endpoints[1].clone(), Some(fixture.server.clone()), false);
    Self {
      peers: [
        Peer {
          endpoint: client,
          connection: Some(connection),
          address: addresses[0],
          connected: false,
        },
        Peer {
          endpoint: server,
          connection: None,
          address: addresses[1],
          connected: false,
        },
      ],
      now,
      queue: VecDeque::with_capacity(16),
      output: Vec::with_capacity(2048),
      sent: 0,
      retry: fixture.mode == NetworkMode::Retry,
      retries: 0,
      validated: None,
    }
  }

  fn queue_output(&mut self, source: usize) {
    if self.output.is_empty() {
      return;
    }
    self.sent = self.sent.strict_add(1);
    assert!(self.sent < 64, "bounded handshake datagram count");
    self
      .queue
      .push_back((1usize.strict_sub(source), BytesMut::from(self.output.as_slice())));
    assert!(self.queue.len() < 64, "bounded datagram queue");
  }

  /// Fixed scheduler ticks advance pacing, never real time. Stop at immediate
  /// quiescence after Connected on both peers, not an arbitrary idle-spin count.
  #[expect(clippy::panic, reason = "a lost connection invalidates this benchmark fixture")]
  fn step(&mut self) -> bool {
    self.now = self.now.checked_add(Duration::from_millis(1)).expect("simulated time");
    let before = self.sent;
    for source in 0..2 {
      self.output.clear();
      let peer = &mut self.peers[source];
      if let Some((handle, connection)) = &mut peer.connection {
        if connection.poll_timeout().is_some_and(|t| t <= self.now) {
          connection.handle_timeout(self.now);
        }
        while let Some(event) = connection.poll_endpoint_events() {
          if let Some(event) = peer.endpoint.handle_event(*handle, event) {
            connection.handle_event(event);
          }
        }
        while let Some(event) = connection.poll() {
          match event {
            Event::Connected => peer.connected = true,
            Event::ConnectionLost { reason } => panic!("unexpected connection loss: {reason:?}"),
            _ => {}
          }
        }
        if let Some(transmit) = connection.poll_transmit(self.now, 1, &mut self.output) {
          assert_eq!(transmit.size, self.output.len());
          assert!(transmit.segment_size.is_none(), "one UDP datagram per send");
        }
      }
      self.queue_output(source);
    }
    let received = !self.queue.is_empty();
    while let Some((destination, bytes)) = self.queue.pop_front() {
      self.receive(destination, bytes);
    }
    received || before != self.sent
  }

  fn receive(&mut self, destination: usize, bytes: BytesMut) {
    let remote = self.peers[1usize.strict_sub(destination)].address;
    let peer = &mut self.peers[destination];
    self.output.clear();
    match peer
      .endpoint
      .handle(self.now, remote, None, None, bytes, &mut self.output)
    {
      Some(DatagramEvent::ConnectionEvent(handle, event)) => {
        let (expected, connection) = peer.connection.as_mut().expect("known connection");
        assert_eq!(handle, *expected);
        connection.handle_event(event);
      }
      Some(DatagramEvent::NewConnection(incoming)) => {
        if self.retry && incoming.may_retry() {
          assert_eq!(self.retries, 0, "at most one Retry");
          let transmit = peer.endpoint.retry(incoming, &mut self.output).expect("Retry");
          assert_eq!(transmit.size, self.output.len());
          self.retries = self.retries.strict_add(1);
        } else {
          assert!(peer.connection.is_none(), "one connection per endpoint");
          self.validated = Some(incoming.remote_address_validated());
          peer.connection = Some(
            peer
              .endpoint
              .accept(incoming, self.now, &mut self.output, None)
              .expect("accept"),
          );
        }
      }
      Some(DatagramEvent::Response(transmit)) => assert_eq!(transmit.size, self.output.len()),
      None => {}
    }
    self.queue_output(destination);
  }

  #[expect(
    clippy::panic,
    reason = "a stalled production handshake invalidates this bounded benchmark fixture"
  )]
  fn handshake(&mut self) {
    for _ in 0..1000 {
      let progress = self.step();
      if !progress && self.peers.iter().all(|p| p.connected) {
        return;
      }
    }
    panic!("QUIC connection exceeded the simulated one-second bound")
  }
}

/// Production reset emission by an endpoint that lost a real connection.
/// Reset cryptography belongs to rscrypto on every adapter configuration, so
/// this has one row and no misleading TLS-provider comparisons.
pub struct ResetFixture {
  endpoint: Endpoint,
  inciting: BytesMut,
  remote: SocketAddr,
  now: Instant,
  output: Vec<u8>,
}

impl ResetFixture {
  pub fn checked() -> Self {
    let fixture = NetworkFixture::configured("rscrypto", "rscrypto", "AES128-GCM-SHA256", NetworkMode::Connect);
    let mut pair = fixture.connect();
    // A real stream packet is large enough to permit an amplification-safe reset.
    let client = &mut pair.peers[0].connection.as_mut().expect("client").1;
    let stream = client.streams().open(quinn_proto::Dir::Uni).expect("stream credit");
    assert_eq!(client.send_stream(stream).write(&[0x5a; 256]).expect("write"), 256);
    let mut inciting = Vec::new();
    let transmit = client.poll_transmit(pair.now, 1, &mut inciting).expect("stream packet");
    assert_eq!(transmit.size, inciting.len());
    assert_eq!(inciting[0] & 0x80, 0, "1-RTT short header");
    let mut reset = Self {
      endpoint: Endpoint::new(fixture.endpoints[1].clone(), None, false),
      inciting: BytesMut::from(inciting.as_slice()),
      remote: pair.peers[0].address,
      now: pair.now,
      output: Vec::with_capacity(2048),
    };
    let bytes = BytesMut::from(reset.emit());
    assert!(
      bytes.len() < inciting.len(),
      "reset must be smaller than the triggering datagram"
    );
    // An altered token cannot kill the live peer; the exact emitted token must.
    let mut invalid = bytes.clone();
    *invalid.last_mut().expect("reset token") ^= 1;
    pair.now = reset.now;
    pair.receive(0, invalid);
    let client = &mut pair.peers[0].connection.as_mut().expect("client").1;
    while let Some(event) = client.poll() {
      assert!(
        !matches!(event, Event::ConnectionLost { .. }),
        "unauthenticated reset rejected"
      );
    }
    pair.receive(0, bytes);
    let client = &mut pair.peers[0].connection.as_mut().expect("client").1;
    assert!(
      core::iter::from_fn(|| client.poll()).any(|event| matches!(
        event,
        Event::ConnectionLost {
          reason: quinn_proto::ConnectionError::Reset
        }
      )),
      "peer must authenticate the emitted reset"
    );
    reset
  }

  #[expect(
    clippy::panic,
    reason = "a missing production reset invalidates this benchmark fixture"
  )]
  pub fn emit(&mut self) -> &[u8] {
    // Keep Quinn's production 20 ms rate limit. Simulated time advances without
    // sleeping; the row measures CPU work, not the allowed emission frequency.
    self.now = self.now.checked_add(Duration::from_millis(20)).expect("simulated time");
    self.output.clear();
    let event = self.endpoint.handle(
      self.now,
      self.remote,
      None,
      None,
      self.inciting.clone(),
      &mut self.output,
    );
    let Some(DatagramEvent::Response(transmit)) = event else {
      panic!("stateless reset response required")
    };
    assert_eq!(transmit.size, self.output.len());
    &self.output
  }
}
