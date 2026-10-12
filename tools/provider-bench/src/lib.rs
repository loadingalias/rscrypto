//! Matched TLS 1.3 connection workloads over an in-process byte transport.
//!
//! Every endpoint calls its real TLS library. Configuration and key import are
//! outside connection timing; connection allocation, cryptography, certificate
//! validation, transport copies, ticket processing, and destruction are inside.

extern crate alloc;

mod openssl_peer;

use alloc::{collections::VecDeque, rc::Rc, sync::Arc};
use core::cell::RefCell;
use std::io::{self, Read, Write};

use rustls::{ClientConfig, ClientSide, Connection, ServerConfig, ServerSide, SideData, Tcp, VecInput};
use serde_json::{Value, json};

mod credentials;

pub const CREDENTIALS: [&str; 7] = ["p256", "p384", "ed25519", "rsa3072", "mldsa44", "mldsa65", "mldsa87"];
pub const GROUPS: [&str; 4] = ["X25519", "secp256r1", "secp384r1", "X25519MLKEM768"];
pub const SUITES: [&str; 3] = ["AES128-GCM-SHA256", "AES256-GCM-SHA384", "CHACHA20-POLY1305-SHA256"];
pub const BACKENDS: [&str; 3] = ["rscrypto", "aws-lc-rs", "openssl"];

#[derive(Clone, Copy, Debug)]
pub struct Profile<'a> {
  pub credential: &'a str,
  pub group: &'a str,
  pub suite: &'a str,
  pub resumed: bool,
}

#[derive(Default)]
struct Queue {
  bytes: VecDeque<u8>,
  written: usize,
}

#[derive(Clone)]
pub(crate) struct Pipe {
  input: Rc<RefCell<Queue>>,
  output: Rc<RefCell<Queue>>,
}

impl Pipe {
  fn pair() -> (Self, Self) {
    let left = Rc::new(RefCell::new(Queue::default()));
    let right = Rc::new(RefCell::new(Queue::default()));
    (
      Self {
        input: left.clone(),
        output: right.clone(),
      },
      Self {
        input: right,
        output: left,
      },
    )
  }

  fn pending(&self) -> bool {
    !self.input.borrow().bytes.is_empty()
  }
  fn written(&self) -> usize {
    self.output.borrow().written
  }
}

impl Read for Pipe {
  fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
    let mut queue = self.input.borrow_mut();
    if queue.bytes.is_empty() {
      return Err(io::ErrorKind::WouldBlock.into());
    }
    let count = output.len().min(queue.bytes.len());
    for byte in &mut output[..count] {
      *byte = queue.bytes.pop_front().expect("queue length");
    }
    Ok(count)
  }
}

impl Write for Pipe {
  fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
    let mut queue = self.output.borrow_mut();
    if queue.bytes.len().strict_add(bytes.len()) > 1_048_576 {
      return Err(io::Error::other("benchmark transport exceeded its byte bound"));
    }
    queue.bytes.extend(bytes);
    queue.written = queue.written.strict_add(bytes.len());
    Ok(bytes.len())
  }
  fn flush(&mut self) -> io::Result<()> {
    Ok(())
  }
}

struct RustlsEndpoint<S: SideData> {
  connection: Connection<S, Tcp>,
  pipe: Pipe,
  input: VecInput,
  output: Vec<u8>,
  plaintext: Vec<u8>,
}

impl<S: SideData> RustlsEndpoint<S> {
  fn new(connection: Connection<S, Tcp>, pipe: Pipe, output: Vec<u8>) -> Self {
    Self {
      connection,
      pipe,
      input: VecInput::default(),
      output,
      plaintext: Vec::new(),
    }
  }
}

#[expect(
  clippy::large_enum_variant,
  reason = "do not add a harness-only allocation to the timed rustls connection"
)]
enum Endpoint<S: SideData> {
  Rustls(RustlsEndpoint<S>),
  OpenSsl(openssl_peer::Endpoint),
}

impl<S: SideData> Drop for Endpoint<S> {
  fn drop(&mut self) {
    match self {
      Self::Rustls(peer) => {
        peer
          .connection
          .send_close_notify(&mut peer.output)
          .expect("close notification");
        peer.pipe.write_all(&peer.output).expect("TLS output");
      }
      Self::OpenSsl(peer) => peer.close(),
    }
  }
}

impl<S: SideData> Endpoint<S> {
  fn pipe(&self) -> &Pipe {
    match self {
      Self::Rustls(peer) => &peer.pipe,
      Self::OpenSsl(peer) => &peer.pipe,
    }
  }
  fn step(&mut self) {
    match self {
      Self::Rustls(peer) => {
        while peer.pipe.pending() {
          assert!(
            peer.input.read(&mut peer.pipe).expect("TLS input") > 0,
            "TLS input stalled"
          );
          peer
            .connection
            .read_tls(&mut peer.input, &mut peer.output)
            .handle_all(&mut peer.plaintext)
            .expect("TLS authentication");
        }
        peer.pipe.write_all(&peer.output).expect("TLS output");
        peer.output.clear();
      }
      Self::OpenSsl(peer) => peer.step(),
    }
  }
  fn handshaking(&self) -> bool {
    match self {
      Self::Rustls(peer) => peer.connection.is_handshaking(),
      Self::OpenSsl(peer) => peer.handshaking(),
    }
  }
  fn resumed(&self) -> bool {
    match self {
      Self::Rustls(peer) => peer.connection.handshake_kind() == Some(rustls::HandshakeKind::Resumed),
      Self::OpenSsl(peer) => peer.resumed(),
    }
  }
  fn exporter(&mut self) -> [u8; 32] {
    let mut bytes = [0; 32];
    match self {
      Self::Rustls(peer) => {
        peer
          .connection
          .exporter()
          .expect("handshake exporter")
          .derive(b"provider-bench", Some(b"matched"), &mut bytes)
          .expect("exporter");
      }
      Self::OpenSsl(peer) => peer.exporter(&mut bytes),
    }
    bytes
  }
  fn send(&mut self, bytes: &[u8]) {
    match self {
      Self::Rustls(peer) => peer
        .connection
        .write(bytes.into(), &mut peer.output)
        .expect("application write"),
      Self::OpenSsl(peer) => peer.send(bytes),
    }
    self.step();
  }
  fn receive(&mut self, bytes: &mut [u8]) {
    match self {
      Self::Rustls(peer) => {
        assert_eq!(peer.plaintext.len(), bytes.len(), "application length");
        bytes.copy_from_slice(&peer.plaintext);
        peer.plaintext.clear();
      }
      Self::OpenSsl(peer) => peer.receive(bytes),
    }
  }
}

/// An authenticated client/server pair. No network delay is simulated here.
pub struct Pair {
  client: Endpoint<ClientSide>,
  server: Endpoint<ServerSide>,
  /// Each nonempty TLS flight in client/server drive order, before application data.
  pub flights: Vec<(bool, usize)>,
}

impl Pair {
  #[expect(
    clippy::panic,
    reason = "a stalled production handshake invalidates this bounded benchmark fixture"
  )]
  fn handshake(mut self, resumed: bool) -> Self {
    for _ in 0..64 {
      fn step<S: SideData>(endpoint: &mut Endpoint<S>, client: bool, flights: &mut Vec<(bool, usize)>) {
        let before = endpoint.pipe().written();
        endpoint.step();
        let size = endpoint.pipe().written().strict_sub(before);
        if size > 0 {
          flights.push((client, size));
        }
      }
      step(&mut self.client, true, &mut self.flights);
      step(&mut self.server, false, &mut self.flights);
      if !self.client.handshaking()
        && !self.server.handshaking()
        && !self.client.pipe().pending()
        && !self.server.pipe().pending()
      {
        assert_eq!(self.client.resumed(), resumed, "client full/resumed workload");
        assert_eq!(self.server.resumed(), resumed, "server full/resumed workload");
        return self;
      }
    }
    panic!("TLS fixture exceeded its flight bound");
  }

  pub fn verify(&mut self) {
    assert_eq!(
      self.client.exporter(),
      self.server.exporter(),
      "independent endpoint exporters"
    );
  }

  /// Transfer application bytes in one direction, including both record endpoints.
  pub fn transfer(&mut self, bytes: &[u8], output: &mut [u8]) {
    assert_eq!(bytes.len(), output.len());
    self.client.send(bytes);
    self.server.step();
    self.server.receive(output);
  }

  pub fn wire_report(&self) -> Value {
    json!({"client_bytes": self.client.pipe().written(), "server_bytes": self.server.pipe().written(),
           "flights": self.flights, "transport": "in-process, no RTT or packet loss"})
  }
}

enum Configs {
  Rustls(Arc<ClientConfig>, Arc<ServerConfig>),
  OpenSsl(openssl_peer::Configs),
}

/// Reusable production TLS configurations and a bounded session cache.
pub struct Fixture {
  configs: Configs,
  resumed: bool,
}

impl Fixture {
  #[expect(clippy::panic, reason = "the benchmark catalog must select known workload values")]
  pub fn new(backend: &str, profile: Profile<'_>) -> Self {
    let configs = if backend == "openssl" {
      Configs::OpenSsl(openssl_peer::Configs::new(profile))
    } else {
      let mut provider = match backend {
        "rscrypto" => rscrypto_rustls::default_provider(),
        "aws-lc-rs" => rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER,
        _ => panic!("unknown benchmark backend"),
      };
      let group = match profile.group {
        "X25519" => rustls::crypto::kx::NamedGroup::X25519,
        "secp256r1" => rustls::crypto::kx::NamedGroup::secp256r1,
        "secp384r1" => rustls::crypto::kx::NamedGroup::secp384r1,
        "X25519MLKEM768" => rustls::crypto::kx::NamedGroup::X25519MLKEM768,
        _ => panic!("unknown group"),
      };
      let suite = match profile.suite {
        "AES128-GCM-SHA256" => rustls::crypto::CipherSuite::TLS13_AES_128_GCM_SHA256,
        "AES256-GCM-SHA384" => rustls::crypto::CipherSuite::TLS13_AES_256_GCM_SHA384,
        "CHACHA20-POLY1305-SHA256" => rustls::crypto::CipherSuite::TLS13_CHACHA20_POLY1305_SHA256,
        _ => panic!("unknown suite"),
      };
      provider.kx_groups.to_mut().retain(|entry| entry.name() == group);
      provider
        .tls13_cipher_suites
        .to_mut()
        .retain(|entry| entry.common.suite == suite);
      let scheme = credentials::scheme(profile.credential);
      let mapping = provider.signature_verification_algorithms.mapping();
      let index = mapping
        .iter()
        .position(|(name, _)| *name == scheme)
        .expect("credential signature scheme");
      let all = if backend == "rscrypto" {
        rscrypto_rustls::verify::ALL_ALGORITHMS
      } else {
        rustls_aws_lc_rs::ALL_VERIFICATION_ALGS
      };
      provider.signature_verification_algorithms =
        rustls::crypto::WebPkiSupportedAlgorithms::new(all, &mapping[index..index.strict_add(1)])
          .expect("selected signature scheme");
      assert_eq!(provider.kx_groups.len(), 1);
      assert_eq!(provider.tls13_cipher_suites.len(), 1);
      let (mut client, mut server) = credentials::configs(provider, profile.credential);
      if profile.resumed {
        server.ticketer = Some(if backend == "rscrypto" {
          rscrypto_rustls::ticket::ticketer().expect("ticket key")
        } else {
          rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER
            .ticketer_factory
            .ticketer()
            .expect("ticket key")
        });
      } else {
        client.resumption = rustls::client::Resumption::disabled();
        server.send_tls13_tickets = rustls::server::Tls13Tickets { default: 0, max: 0 };
      }
      Configs::Rustls(Arc::new(client), Arc::new(server))
    };
    let fixture = Self {
      configs,
      resumed: profile.resumed,
    };
    // Seed the client cache once; every timed resumed connection must consume a
    // real ticket and process a replacement. Never time a disguised full handshake.
    let mut first = fixture.connect_inner(false);
    first.verify();
    fixture
  }

  pub fn connect(&self) -> Pair {
    self.connect_inner(self.resumed)
  }

  fn connect_inner(&self, resumed: bool) -> Pair {
    let (left, right) = Pipe::pair();
    let (client, server) = match &self.configs {
      Configs::Rustls(client, server) => {
        let mut output = Vec::new();
        let client = client
          .connect(rustls::pki_types::ServerName::try_from("localhost").expect("fixture name"))
          .build(&mut output)
          .expect("client configuration");
        let server = Connection::new(server.clone()).expect("server configuration");
        (
          Endpoint::Rustls(RustlsEndpoint::new(client, left, output)),
          Endpoint::Rustls(RustlsEndpoint::new(server, right, Vec::new())),
        )
      }
      Configs::OpenSsl(configs) => {
        let (client, server) = configs.endpoints(left, right, resumed);
        (Endpoint::OpenSsl(client), Endpoint::OpenSsl(server))
      }
    };
    Pair {
      client,
      server,
      flights: Vec::new(),
    }
    .handshake(resumed)
  }
}
