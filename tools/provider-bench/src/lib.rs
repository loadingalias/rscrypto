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

use rustls::{ClientConfig, Connection, ServerConfig};
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

#[expect(
  clippy::large_enum_variant,
  reason = "do not add a harness-only allocation to the timed rustls connection"
)]
enum Endpoint {
  Rustls(Connection, Pipe),
  OpenSsl(openssl_peer::Endpoint),
}

impl Drop for Endpoint {
  fn drop(&mut self) {
    match self {
      Self::Rustls(connection, pipe) => {
        connection.send_close_notify();
        while connection.wants_write() {
          connection.write_tls(pipe).expect("close notification");
        }
      }
      Self::OpenSsl(peer) => peer.close(),
    }
  }
}

impl Endpoint {
  fn pipe(&self) -> &Pipe {
    match self {
      Self::Rustls(_, pipe) => pipe,
      Self::OpenSsl(peer) => &peer.pipe,
    }
  }
  fn step(&mut self) {
    match self {
      Self::Rustls(connection, pipe) => {
        while pipe.pending() {
          connection.read_tls(pipe).expect("TLS input");
          connection.process_new_packets().expect("TLS authentication");
        }
        while connection.wants_write() {
          connection.write_tls(pipe).expect("TLS output");
        }
      }
      Self::OpenSsl(peer) => peer.step(),
    }
  }
  fn handshaking(&self) -> bool {
    match self {
      Self::Rustls(connection, _) => connection.is_handshaking(),
      Self::OpenSsl(peer) => peer.handshaking(),
    }
  }
  fn resumed(&self) -> bool {
    match self {
      Self::Rustls(connection, _) => connection.handshake_kind() == Some(rustls::HandshakeKind::Resumed),
      Self::OpenSsl(peer) => peer.resumed(),
    }
  }
  fn exporter(&self) -> [u8; 32] {
    let mut bytes = [0; 32];
    match self {
      Self::Rustls(connection, _) => {
        connection
          .export_keying_material(&mut bytes, b"provider-bench", Some(b"matched"))
          .expect("exporter");
      }
      Self::OpenSsl(peer) => peer.exporter(&mut bytes),
    }
    bytes
  }
  fn send(&mut self, bytes: &[u8]) {
    match self {
      Self::Rustls(connection, _) => connection.writer().write_all(bytes).expect("application write"),
      Self::OpenSsl(peer) => peer.send(bytes),
    }
    self.step();
  }
  fn receive(&mut self, bytes: &mut [u8]) {
    match self {
      Self::Rustls(connection, _) => connection.reader().read_exact(bytes).expect("application read"),
      Self::OpenSsl(peer) => peer.receive(bytes),
    }
  }
}

/// An authenticated client/server pair. No network delay is simulated here.
pub struct Pair {
  client: Endpoint,
  server: Endpoint,
  /// Each nonempty TLS flight in client/server drive order, before application data.
  pub flights: Vec<(bool, usize)>,
}

impl Pair {
  #[expect(clippy::panic, reason = "a stalled production handshake invalidates this bounded benchmark fixture")]
  fn handshake(mut self, resumed: bool) -> Self {
    for _ in 0..64 {
      for (client, endpoint) in [(true, &mut self.client), (false, &mut self.server)] {
        let before = endpoint.pipe().written();
        endpoint.step();
        let size = endpoint.pipe().written().strict_sub(before);
        if size > 0 {
          self.flights.push((client, size));
        }
      }
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

  pub fn verify(&self) {
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
        "aws-lc-rs" => rustls::crypto::aws_lc_rs::default_provider(),
        _ => panic!("unknown benchmark backend"),
      };
      let group = match profile.group {
        "X25519" => rustls::NamedGroup::X25519,
        "secp256r1" => rustls::NamedGroup::secp256r1,
        "secp384r1" => rustls::NamedGroup::secp384r1,
        "X25519MLKEM768" => rustls::NamedGroup::X25519MLKEM768,
        _ => panic!("unknown group"),
      };
      let suite = match profile.suite {
        "AES128-GCM-SHA256" => rustls::CipherSuite::TLS13_AES_128_GCM_SHA256,
        "AES256-GCM-SHA384" => rustls::CipherSuite::TLS13_AES_256_GCM_SHA384,
        "CHACHA20-POLY1305-SHA256" => rustls::CipherSuite::TLS13_CHACHA20_POLY1305_SHA256,
        _ => panic!("unknown suite"),
      };
      provider.kx_groups.retain(|entry| entry.name() == group);
      provider.cipher_suites.retain(|entry| entry.suite() == suite);
      let scheme = credentials::scheme(profile.credential);
      let mapping = provider.signature_verification_algorithms.mapping;
      let index = mapping
        .iter()
        .position(|(name, _)| *name == scheme)
        .expect("credential signature scheme");
      provider.signature_verification_algorithms.mapping = &mapping[index..index.strict_add(1)];
      assert_eq!(provider.kx_groups.len(), 1);
      assert_eq!(provider.cipher_suites.len(), 1);
      let (mut client, mut server) = credentials::configs(provider, profile.credential);
      if profile.resumed {
        server.ticketer = if backend == "rscrypto" {
          rscrypto_rustls::ticket::ticketer().expect("ticket key")
        } else {
          rustls::crypto::aws_lc_rs::Ticketer::new().expect("ticket key")
        };
      } else {
        client.resumption = rustls::client::Resumption::disabled();
        server.send_tls13_tickets = 0;
      }
      Configs::Rustls(Arc::new(client), Arc::new(server))
    };
    let fixture = Self {
      configs,
      resumed: profile.resumed,
    };
    // Seed the client cache once; every timed resumed connection must consume a
    // real ticket and process a replacement. Never time a disguised full handshake.
    let first = fixture.connect_inner(false);
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
        let client = Connection::Client(
          rustls::ClientConnection::new(
            client.clone(),
            rustls::pki_types::ServerName::try_from("localhost").expect("fixture name"),
          )
          .expect("client configuration"),
        );
        let server = Connection::Server(rustls::ServerConnection::new(server.clone()).expect("server configuration"));
        (Endpoint::Rustls(client, left), Endpoint::Rustls(server, right))
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
