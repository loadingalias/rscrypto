//! Bounded in-memory transport driving Quinn's production endpoints and streams.

use super::*;
use crate::tests::tls::{self, Credential};
use alloc::collections::VecDeque;
use core::{net::SocketAddr, time::Duration};
use quinn_proto::{Connection, ConnectionHandle, DatagramEvent, Dir, EcnCodepoint, Endpoint, Event};
use std::time::{Instant, SystemTime};

/// Address-token issue time for simulated runs.
#[derive(Debug)]
struct FixedTime;

impl quinn_proto::TimeSource for FixedTime {
  fn now(&self) -> SystemTime {
    SystemTime::UNIX_EPOCH
      .checked_add(Duration::from_secs(1_792_627_200))
      .expect("fixture time")
  }
}

struct Peer {
  endpoint: Endpoint,
  connection: Option<(ConnectionHandle, Connection)>,
  address: SocketAddr,
  keys: EndpointKeys,
}

#[derive(Clone)]
struct Datagram {
  destination: usize,
  address: SocketAddr,
  remote: SocketAddr,
  ecn: Option<EcnCodepoint>,
  bytes: BytesMut,
}

struct Link {
  peers: [Peer; 2],
  now: Instant,
  queue: VecDeque<Datagram>,
  sent: usize,
  faults: [usize; 4],
  distort: bool,
  retry: bool,
  retries: usize,
  expected_reset: Option<usize>,
  reset_received: bool,
  mark_congestion: bool,
  congestion_marks: [usize; 2],
  /// Every datagram handed to the network, in send order.
  sent_log: Vec<(usize, Vec<u8>)>,
}

impl Link {
  fn new(client: rustls::ClientConfig, server: rustls::ServerConfig, distort: bool, retry: bool) -> Self {
    Self::with_entropy(client, server, distort, retry, &crate::tests::RANDOM)
  }

  /// Build both endpoints with keys and seeds from `random`, so every transport
  /// value comes from the TLS providers' sources and `random`.
  fn with_entropy(
    client: rustls::ClientConfig,
    server: rustls::ServerConfig,
    distort: bool,
    retry: bool,
    random: &dyn rustls::crypto::SecureRandom,
  ) -> Self {
    let addresses = [
      "127.0.0.1:44330".parse().expect("address"),
      "127.0.0.1:44331".parse().expect("address"),
    ];
    let now = Instant::now();
    let keys = [
      EndpointKeys::generate(random).expect("client keys"),
      EndpointKeys::generate(random).expect("server keys"),
    ];
    let client_endpoint = Arc::new(keys[0].endpoint_config(random).expect("client endpoint"));
    let server_endpoint = Arc::new(keys[1].endpoint_config(random).expect("server endpoint"));
    let mut server_config = server_config(server).expect("server config");
    // A fixed issue time keeps address tokens a function of the simulated run.
    server_config.time_source(Arc::new(FixedTime));
    let mut endpoint = Endpoint::new(client_endpoint, None, false);
    let connection = endpoint
      .connect(
        now,
        client_config(client).expect("client config"),
        addresses[1],
        "localhost",
      )
      .expect("start client");
    let server = Endpoint::new(server_endpoint, Some(Arc::new(server_config)), false);
    let [client_keys, server_keys] = keys;
    Self {
      peers: [
        Peer {
          endpoint,
          connection: Some(connection),
          address: addresses[0],
          keys: client_keys,
        },
        Peer {
          endpoint: server,
          connection: None,
          address: addresses[1],
          keys: server_keys,
        },
      ],
      now,
      queue: VecDeque::new(),
      sent: 0,
      faults: [0; 4],
      distort,
      retry,
      retries: 0,
      expected_reset: None,
      reset_received: false,
      mark_congestion: false,
      congestion_marks: [0; 2],
      sent_log: Vec::new(),
    }
  }

  fn send(&mut self, source: usize, address: SocketAddr, ecn: Option<EcnCodepoint>, bytes: &[u8]) {
    self.sent = self.sent.strict_add(1);
    let destination = 1usize.strict_sub(source);
    self.sent_log.push((destination, bytes.to_vec()));
    if address != self.peers[destination].address {
      return;
    }
    if self.distort && self.sent.is_multiple_of(7) {
      self.faults[0] = self.faults[0].strict_add(1);
      return;
    }
    let ecn = if self.mark_congestion && ecn.is_some() {
      self.congestion_marks[source] = self.congestion_marks[source].strict_add(1);
      Some(EcnCodepoint::Ce)
    } else {
      ecn
    };
    let mut datagram = Datagram {
      destination,
      address,
      remote: self.peers[source].address,
      ecn,
      bytes: BytesMut::from(bytes),
    };
    if self.distort && self.sent.is_multiple_of(11) {
      let last = datagram.bytes.len().strict_sub(1);
      datagram.bytes[last] ^= 1;
      self.faults[1] = self.faults[1].strict_add(1);
    }
    if self.distort && self.sent.is_multiple_of(5) {
      self.queue.push_back(datagram.clone());
      self.faults[2] = self.faults[2].strict_add(1);
    }
    self.queue.push_back(datagram);
    assert!(self.queue.len() < 1024, "bounded simulated network");
  }

  fn step(&mut self) {
    self.now = self
      .now
      .checked_add(Duration::from_millis(2))
      .expect("simulated clock bound");
    for source in 0..2 {
      let mut output = Vec::new();
      let mut route = None;
      let peer = &mut self.peers[source];
      if let Some((handle, connection)) = &mut peer.connection {
        if connection.poll_timeout().is_some_and(|deadline| deadline <= self.now) {
          connection.handle_timeout(self.now);
        }
        while let Some(event) = connection.poll_endpoint_events() {
          if let Some(event) = peer.endpoint.handle_event(*handle, event) {
            connection.handle_event(event);
          }
        }
        while let Some(event) = connection.poll() {
          if let Event::ConnectionLost { reason } = event {
            assert_eq!(self.expected_reset, Some(source), "unexpected QUIC failure: {reason:?}");
            assert!(
              matches!(reason, quinn_proto::ConnectionError::Reset),
              "expected authenticated reset: {reason:?}"
            );
            self.reset_received = true;
          }
        }
        // One datagram per endpoint per scheduler step keeps pacing explicit.
        if let Some(transmit) = connection.poll_transmit(self.now, 1, &mut output) {
          assert_eq!(transmit.size, output.len());
          assert!(transmit.segment_size.is_none());
          route = Some((transmit.destination, transmit.ecn));
        }
      }
      if !output.is_empty() {
        let (address, ecn) = route.expect("transmitted datagram has a route");
        self.send(source, address, ecn, &output);
      }
    }
    let datagram = if self.distort && self.queue.len() > 1 && self.sent.is_multiple_of(3) {
      self.faults[3] = self.faults[3].strict_add(1);
      self.queue.pop_back()
    } else {
      self.queue.pop_front()
    };
    let Some(Datagram {
      destination,
      address,
      remote,
      ecn,
      bytes,
    }) = datagram
    else {
      return;
    };
    let peer = &mut self.peers[destination];
    if address != peer.address {
      return;
    }
    let mut response = Vec::new();
    let mut response_route = (remote, None);
    match peer.endpoint.handle(self.now, remote, None, ecn, bytes, &mut response) {
      Some(DatagramEvent::ConnectionEvent(handle, event)) => {
        let (expected, connection) = peer.connection.as_mut().expect("known connection");
        assert_eq!(handle, *expected);
        connection.handle_event(event);
      }
      Some(DatagramEvent::NewConnection(incoming)) => {
        if self.retry && incoming.may_retry() {
          let transmit = peer.endpoint.retry(incoming, &mut response).expect("Retry");
          assert_eq!(transmit.size, response.len());
          self.retries = self.retries.strict_add(1);
          response_route = (transmit.destination, transmit.ecn);
        } else {
          assert!(peer.connection.is_none(), "single connection per endpoint");
          peer.connection = Some(
            peer
              .endpoint
              .accept(incoming, self.now, &mut response, None)
              .expect("accept QUIC"),
          );
        }
      }
      Some(DatagramEvent::Response(transmit)) => {
        assert_eq!(transmit.size, response.len());
        response_route = (transmit.destination, transmit.ecn);
      }
      None => {}
    }
    if !response.is_empty() {
      self.send(destination, response_route.0, response_route.1, &response);
    }
  }

  fn connection(&mut self, side: usize) -> &mut Connection {
    &mut self.peers[side].connection.as_mut().expect("established connection").1
  }

  #[expect(
    clippy::panic,
    reason = "test harness must fail when the handshake exceeds its execution bound"
  )]
  fn handshake(&mut self) {
    for _ in 0..10_000 {
      self.step();
      if self.peers.iter().all(|peer| {
        peer
          .connection
          .as_ref()
          .is_some_and(|(_, connection)| !connection.is_handshaking())
      }) {
        // Complete HANDSHAKE_DONE/ACK delivery before initiating a key update.
        for _ in 0..100 {
          self.step();
        }
        return;
      }
    }
    panic!("QUIC handshake exceeded the simulated 20-second budget");
  }

  #[expect(
    clippy::panic,
    reason = "test harness reports unexpected stream errors or a bounded transfer timeout"
  )]
  fn stream(&mut self, sender: usize) {
    let receiver = 1usize.strict_sub(sender);
    let payload = b"loss-tolerant authenticated QUIC stream\n".repeat(256);
    let id = self.connection(sender).streams().open(Dir::Uni).expect("stream credit");
    assert_eq!(
      self
        .connection(sender)
        .send_stream(id)
        .write(&payload)
        .expect("stream write"),
      payload.len()
    );
    self.connection(sender).send_stream(id).finish().expect("finish");
    let mut received = Vec::new();
    let mut accepted = false;
    for _ in 0..10_000 {
      self.step();
      if !accepted {
        if let Some(incoming) = self.connection(receiver).streams().accept(Dir::Uni) {
          assert_eq!(incoming, id);
          accepted = true;
        } else {
          continue;
        }
      }
      let mut stream = self.connection(receiver).recv_stream(id);
      let mut chunks = stream.read(true).expect("readable stream");
      let mut done = false;
      loop {
        match chunks.next(4096) {
          Ok(Some(chunk)) => received.extend_from_slice(&chunk.bytes),
          Ok(None) => {
            done = true;
            break;
          }
          Err(quinn_proto::ReadError::Blocked) => break,
          Err(error) => panic!("unexpected stream error: {error:?}"),
        }
      }
      let transmit = chunks.finalize();
      // The next scheduler step polls output regardless of whether flow control
      // currently needs a transmission, so consume that notification explicitly.
      let _needs_transmit = transmit.should_transmit();
      if done {
        assert_eq!(received, payload);
        return;
      }
      assert!(
        received.len() <= payload.len(),
        "duplicates must not duplicate stream bytes"
      );
    }
    panic!("QUIC transfer exceeded its simulated 20-second budget");
  }
}

fn exercise(credential: Credential, distort: bool, retry: bool) {
  for suite in crate::cipher_suite::ALL_CIPHER_SUITES {
    let mut provider = tls::provider();
    provider.tls13_cipher_suites = vec![*suite].into();
    let expected_group = provider.kx_groups[0].name();
    let (client, mut server) = tls::configs(provider.clone(), provider, credential, true);
    server.ticketer =
      Some(crate::ticket::ticketer_with(&crate::tests::RANDOM, Arc::new(tls::FixtureTime)).expect("ticket key"));
    let mut link = Link::new(client, server, distort, retry);
    link.handshake();
    let mut exported = [[0; 32]; 2];
    for (side, output) in exported.iter_mut().enumerate() {
      let crypto = link.connection(side).crypto_session();
      let data = crypto
        .handshake_data()
        .expect("handshake data")
        .downcast::<HandshakeData>()
        .expect("our adapter");
      assert_eq!(data.protocol, b"provider-test");
      assert_eq!(data.cipher_suite, suite.common.suite);
      assert_eq!(data.key_exchange_group, Some(expected_group));
      assert!(crypto.peer_identity().is_some(), "mutual authentication");
      crypto
        .export_keying_material(output, b"quinn adapter", b"test context")
        .expect("exporter");
    }
    assert_eq!(exported[0], exported[1]);
    link.stream(0);
    link.connection(0).force_key_update();
    link.stream(0);
    link.connection(1).force_key_update();
    link.stream(1);
    assert_eq!(link.retries > 0, retry);
    if distort {
      assert!(
        link.faults.iter().all(|count| *count > 0),
        "each transport fault ran: {:?}",
        link.faults
      );
    }
  }
}

#[test]
fn quinn_streams_survive_loss_reordering_duplication_and_corruption() {
  exercise(Credential::P256, true, false);
}

#[test]
fn quinn_retry_validates_the_client_before_accepting() {
  exercise(Credential::P256, false, true);
}

#[cfg(feature = "ml-dsa")]
#[test]
fn quinn_mldsa_chain_survives_fragmented_lossy_transport() {
  exercise(Credential::MlDsa87, true, false);
}

#[test]
fn quinn_ecn_feedback_reaches_both_congestion_controllers() {
  let (client, server) = tls::configs(tls::provider(), tls::provider(), Credential::P256, true);
  let mut link = Link::new(client, server, false, false);
  link.handshake();
  let before = [
    link.connection(0).stats().path.congestion_events,
    link.connection(1).stats().path.congestion_events,
  ];
  link.mark_congestion = true;
  link.stream(0);
  link.stream(1);
  for _ in 0..100 {
    link.step();
  }
  for (side, initial) in before.into_iter().enumerate() {
    assert!(link.congestion_marks[side] > 0, "peer sent ECN-capable datagrams");
    assert!(
      link.connection(side).stats().path.congestion_events > initial,
      "authenticated ECN feedback must reach each sender's congestion controller"
    );
  }
}

#[test]
fn quinn_streams_survive_client_address_migration() {
  let (client, server) = tls::configs(tls::provider(), tls::provider(), Credential::P256, true);
  let mut link = Link::new(client, server, false, false);
  link.handshake();
  link.stream(0);
  let migrated: SocketAddr = "127.0.0.2:44332".parse().expect("new client address");
  link.peers[0].address = migrated;
  link.connection(0).local_address_changed();
  link.stream(0);
  link.connection(0).force_key_update();
  link.stream(1);
  for _ in 0..100 {
    link.step();
  }
  assert_eq!(link.connection(1).remote_address(), migrated);
  let client = link.connection(0).stats();
  let server = link.connection(1).stats();
  assert!(server.frame_tx.path_challenge > 0, "server challenged the new path");
  assert!(client.frame_tx.path_response > 0, "client answered the path challenge");
  assert!(
    server.frame_rx.path_response > 0,
    "server authenticated the path response"
  );
}

#[test]
fn quinn_resumes_with_explicit_early_data_and_rejects_it_after_policy_change() {
  for accept in [false, true] {
    let (mut client, mut server) = tls::configs(tls::provider(), tls::provider(), Credential::P256, true);
    client.enable_early_data = true;
    server.max_early_data_size = u32::MAX;
    let mut first = Link::new(client.clone(), server.clone(), false, false);
    first.handshake();
    first.stream(0);
    if !accept {
      server.max_early_data_size = 0;
    }
    let mut resumed = Link::new(client, server, false, false);
    assert!(resumed.connection(0).has_0rtt(), "ticket supplied early packet keys");
    let stream = resumed
      .connection(0)
      .streams()
      .open(Dir::Uni)
      .expect("cached stream credit");
    let payload = b"explicit replay-safe early request";
    assert_eq!(
      resumed
        .connection(0)
        .send_stream(stream)
        .write(payload)
        .expect("early write"),
      payload.len()
    );
    resumed
      .connection(0)
      .send_stream(stream)
      .finish()
      .expect("early finish");
    resumed.handshake();
    assert_eq!(resumed.connection(0).accepted_0rtt(), accept);
    let data = resumed
      .connection(0)
      .crypto_session()
      .handshake_data()
      .expect("resumption data")
      .downcast::<HandshakeData>()
      .expect("adapter");
    assert_eq!(data.handshake_kind, Some(rustls::HandshakeKind::Resumed));
    if accept {
      assert_eq!(resumed.connection(1).streams().accept(Dir::Uni), Some(stream));
      let mut stream = resumed.connection(1).recv_stream(stream);
      let mut chunks = stream.read(true).expect("early receive");
      let mut received = Vec::new();
      while let Some(chunk) = chunks.next(4096).expect("complete early data") {
        received.extend_from_slice(&chunk.bytes);
      }
      let _transmit = chunks.finalize();
      assert_eq!(received, payload);
    } else {
      assert!(
        resumed.connection(1).streams().accept(Dir::Uni).is_none(),
        "rejected early bytes stay unavailable"
      );
    }
    resumed.stream(0);
    resumed.stream(1);
  }
}

#[test]
fn quinn_reset_authenticates_forgotten_connection_in_both_directions() {
  for forgotten in 0..2 {
    let (client, server) = tls::configs(tls::provider(), tls::provider(), Credential::P256, true);
    let mut link = Link::new(client, server, false, false);
    link.handshake();
    link.stream(0);
    link.stream(1);
    let survivor = 1usize.strict_sub(forgotten);
    link.expected_reset = Some(survivor);
    link.peers[forgotten].connection = None;
    // The restarted endpoint keeps its reset key but draws a fresh endpoint seed.
    let restarted = link.peers[forgotten]
      .keys
      .endpoint_config(&crate::tests::RANDOM)
      .expect("restarted endpoint");
    link.peers[forgotten].endpoint = Endpoint::new(Arc::new(restarted), None, false);
    link.queue.clear();
    link.connection(survivor).ping();
    for _ in 0..1000 {
      link.step();
      if link.reset_received {
        break;
      }
    }
    assert!(
      link.reset_received,
      "a reset authenticated with the retained endpoint key terminates the peer"
    );
  }
}

/// Test entropy that replays: a ChaCha20 stream from a fixed seed.
#[derive(Debug)]
struct Replayable(std::sync::Mutex<rscrypto::ChaCha20Rng>);

impl rustls::crypto::SecureRandom for Replayable {
  fn fill(&self, output: &mut [u8]) -> Result<(), rustls::crypto::GetRandomFailed> {
    self.0.lock().expect("test entropy").fill_bytes(output);
    Ok(())
  }
}

/// Every datagram of one handshake, transfer, and key update, from fixed sources.
fn transcript(seed: u8) -> Vec<(usize, Vec<u8>)> {
  let random: &'static Replayable = Box::leak(Box::new(Replayable(std::sync::Mutex::new(
    rscrypto::ChaCha20Rng::from_seed([seed; 32]),
  ))));
  let client: &'static crate::ClientProvider = Box::leak(Box::new(crate::ClientProvider::new(random)));
  let mut provider = client.provider();
  provider.key_provider = Box::leak(Box::new(crate::sign::Provider::new(random)));
  let (client, server) = tls::configs(provider.clone(), provider, Credential::P256, true);
  let mut link = Link::with_entropy(client, server, false, false, random);
  link.handshake();
  link.stream(0);
  link.connection(0).force_key_update();
  link.stream(1);
  assert!(link.sent_log.len() > 20, "the run exchanged packets");
  link.sent_log
}

#[test]
fn quinn_transcript_is_a_function_of_the_caller_sources_and_clock() {
  // TLS, endpoint keys, endpoint and connection generators, connection IDs,
  // and address tokens all draw from the replayed source; the clock is
  // simulated. Any read of OS entropy would make two runs diverge.
  let first = transcript(1);
  assert_eq!(
    first,
    transcript(1),
    "identical sources and clock replay every datagram"
  );
  assert_ne!(first, transcript(2), "the transcript depends on the caller's entropy");
}
