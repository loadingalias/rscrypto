//! Matched TLS 1.2/1.3 connection workloads over an in-process byte transport.
//!
//! Every endpoint calls its real TLS library. Configuration and key import are
//! outside connection timing; connection allocation, cryptography, certificate
//! validation, transport copies, ticket processing, and destruction are inside.

extern crate alloc;

mod openssl_peer;

use alloc::{borrow::Cow, collections::VecDeque, rc::Rc, sync::Arc};
use core::cell::RefCell;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::io::{self, Read, Write};

use rustls::client::{ClientSessionKey, ClientSessionMemoryCache, ClientSessionStore, Resumption, Tls12Resumption};
use rustls::crypto::kx::NamedGroup;
use rustls::crypto::{CipherSuite, Identity};
use rustls::enums::ProtocolVersion;
use rustls::{ClientConfig, ClientSide, Connection, ServerConfig, ServerSide, SideData, Tcp, VecInput};
use serde_json::{Value, json};

mod credentials;
mod provenance;

#[cfg(feature = "quic")]
pub mod quic;

pub use provenance::provenance;

pub const CREDENTIALS: [&str; 7] = ["p256", "p384", "ed25519", "rsa3072", "mldsa44", "mldsa65", "mldsa87"];
pub const TLS12_CREDENTIALS: [&str; 5] = ["p256", "p384", "ed25519", "rsa2048", "rsa3072"];
pub const TLS12_GROUPS: [&str; 3] = ["X25519", "secp256r1", "secp384r1"];
pub const RSA_CREDENTIALS: [&str; 5] = [
  "rsa3072-sha384",
  "rsa3072-sha512",
  "rsa4096",
  "rsa4096-sha384",
  "rsa4096-sha512",
];
pub const GROUPS: [&str; 4] = ["X25519", "secp256r1", "secp384r1", "X25519MLKEM768"];
/// The provider's opt-in groups. They have p256 handshake rows only; the
/// default groups above carry the full credential matrix.
pub const OPT_IN_GROUPS: [&str; 4] = ["secp256r1MLKEM768", "secp384r1MLKEM1024", "MLKEM768", "MLKEM1024"];
pub const SUITES: [&str; 3] = ["AES128-GCM-SHA256", "AES256-GCM-SHA384", "CHACHA20-POLY1305-SHA256"];
/// `rscrypto` installs the provider's prepared-anchor verifiers, its recommended
/// configuration. `rscrypto-webpki` uses rustls's WebPKI verifiers, as `aws-lc-rs`
/// and `ring` do.
pub const BACKENDS: [&str; 5] = ["rscrypto", "rscrypto-webpki", "aws-lc-rs", "ring", "openssl"];

/// Composite ML-DSA credentials (draft-reddy-tls-composite-mldsa-12). No
/// other backend implements these schemes, so only the rscrypto backends run them.
pub const COMPOSITE_CREDENTIALS: [&str; 2] = ["mldsa65-ed25519", "mldsa65-ecdsa-p256"];

/// Whether `backend` implements every algorithm `profile` needs. *ring* has no
/// ML-KEM or ML-DSA, and the pinned aws-lc-rs provider has no
/// SecP384r1MLKEM1024, so those profiles have no row rather than a substitute.
#[must_use]
pub fn supports(backend: &str, profile: Profile<'_>) -> bool {
  if COMPOSITE_CREDENTIALS.contains(&profile.credential) {
    return backend.starts_with("rscrypto");
  }
  match backend {
    "ring" => !(profile.group.contains("MLKEM") || profile.credential.starts_with("mldsa")),
    "aws-lc-rs" => profile.group != "secp384r1MLKEM1024",
    _ => true,
  }
}

/// Largest TLS record plaintext.
const MAX_PLAINTEXT: usize = 16_384;
/// Each direction's preallocated bytes. The largest catalog flight and a full
/// record fit without reallocating, so transport growth is equal on every stack.
const QUEUE_CAPACITY: usize = 32 * 1024;
const QUEUE_BOUND: usize = 1_048_576;

#[derive(Clone, Copy, Debug)]
pub struct Profile<'a> {
  pub credential: &'a str,
  pub group: &'a str,
  pub suite: &'a str,
  pub resumed: bool,
}

/// One change from the original mutually authenticated, leaf-only workload.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Workload {
  #[default]
  Mutual,
  ServerAuth,
  Intermediate,
  Stateful,
  HelloRetryRequest,
  CompactKey,
}

#[expect(clippy::panic, reason = "the benchmark catalog must select known workload values")]
fn canonical(values: &[&'static str], value: &str) -> &'static str {
  values
    .iter()
    .copied()
    .find(|known| *known == value)
    .unwrap_or_else(|| panic!("unknown workload value {value}"))
}

impl Profile<'_> {
  fn canonical(self, version: ProtocolVersion) -> Profile<'static> {
    let credentials = if version == ProtocolVersion::TLSv1_2 {
      TLS12_CREDENTIALS.to_vec()
    } else {
      CREDENTIALS
        .into_iter()
        .chain(RSA_CREDENTIALS)
        .chain(COMPOSITE_CREDENTIALS)
        .collect()
    };
    Profile {
      credential: canonical(&credentials, self.credential),
      group: canonical(
        &if version == ProtocolVersion::TLSv1_2 {
          TLS12_GROUPS.to_vec()
        } else {
          GROUPS.into_iter().chain(OPT_IN_GROUPS).collect()
        },
        self.group,
      ),
      suite: canonical(&SUITES, self.suite),
      resumed: self.resumed,
    }
  }
}

#[expect(clippy::panic, reason = "the benchmark catalog must select known workload values")]
fn suite(name: &str) -> CipherSuite {
  match name {
    "AES128-GCM-SHA256" => CipherSuite::TLS13_AES_128_GCM_SHA256,
    "AES256-GCM-SHA384" => CipherSuite::TLS13_AES_256_GCM_SHA384,
    "CHACHA20-POLY1305-SHA256" => CipherSuite::TLS13_CHACHA20_POLY1305_SHA256,
    _ => panic!("unknown suite"),
  }
}

#[expect(clippy::panic, reason = "the benchmark catalog must select known workload values")]
fn selected_suite(profile: Profile<'_>, version: ProtocolVersion) -> CipherSuite {
  if version == ProtocolVersion::TLSv1_3 {
    return suite(profile.suite);
  }
  // TLS 1.2 Ed25519 authentication uses the ECDHE_ECDSA suite family.
  match (profile.credential.starts_with("rsa"), profile.suite) {
    (false, "AES128-GCM-SHA256") => CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
    (false, "AES256-GCM-SHA384") => CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
    (false, "CHACHA20-POLY1305-SHA256") => CipherSuite::TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
    (true, "AES128-GCM-SHA256") => CipherSuite::TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
    (true, "AES256-GCM-SHA384") => CipherSuite::TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
    (true, "CHACHA20-POLY1305-SHA256") => CipherSuite::TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
    _ => panic!("unknown TLS 1.2 suite"),
  }
}

#[expect(clippy::panic, reason = "the benchmark catalog must select known workload values")]
fn group(name: &str) -> NamedGroup {
  match name {
    "X25519" => NamedGroup::X25519,
    "secp256r1" => NamedGroup::secp256r1,
    "secp384r1" => NamedGroup::secp384r1,
    "X25519MLKEM768" => NamedGroup::X25519MLKEM768,
    "secp256r1MLKEM768" => NamedGroup::secp256r1MLKEM768,
    "secp384r1MLKEM1024" => NamedGroup::secp384r1MLKEM1024,
    "MLKEM768" => NamedGroup::MLKEM768,
    "MLKEM1024" => NamedGroup::MLKEM1024,
    _ => panic!("unknown group"),
  }
}

fn certificate_group(profile: Profile<'_>) -> Option<NamedGroup> {
  match profile.credential {
    "p256" => Some(NamedGroup::secp256r1),
    "p384" => Some(NamedGroup::secp384r1),
    _ => None,
  }
}

struct Queue {
  bytes: VecDeque<u8>,
  written: usize,
}

impl Queue {
  fn new() -> Self {
    Self {
      bytes: VecDeque::with_capacity(QUEUE_CAPACITY),
      written: 0,
    }
  }
}

#[derive(Clone)]
pub(crate) struct Pipe {
  input: Rc<RefCell<Queue>>,
  output: Rc<RefCell<Queue>>,
}

impl Pipe {
  fn pair() -> (Self, Self) {
    let left = Rc::new(RefCell::new(Queue::new()));
    let right = Rc::new(RefCell::new(Queue::new()));
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
    // std copies the ring buffer's front slice with one memcpy, then drains it.
    queue.bytes.read(output)
  }
}

impl Write for Pipe {
  fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
    let mut queue = self.output.borrow_mut();
    if queue.bytes.len().strict_add(bytes.len()) > QUEUE_BOUND {
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

/// Client session storage that counts the TLS 1.3 tickets its client accepts.
///
/// With `inner`, it delegates to rustls's default in-memory store; without it,
/// it stores nothing, as `Resumption::disabled()` does.
#[derive(Debug)]
struct CountingStore {
  inner: Option<ClientSessionMemoryCache>,
  tickets: Arc<AtomicUsize>,
}

impl ClientSessionStore for CountingStore {
  fn set_kx_hint(&self, key: ClientSessionKey<'static>, group: NamedGroup) {
    if let Some(inner) = &self.inner {
      inner.set_kx_hint(key, group);
    }
  }
  fn kx_hint(&self, key: &ClientSessionKey<'_>) -> Option<NamedGroup> {
    self.inner.as_ref()?.kx_hint(key)
  }
  fn set_tls12_session(&self, key: ClientSessionKey<'static>, value: rustls::client::Tls12Session) {
    if let Some(inner) = &self.inner {
      inner.set_tls12_session(key, value);
    }
  }
  fn tls12_session(&self, key: &ClientSessionKey<'_>) -> Option<rustls::client::Tls12Session> {
    self.inner.as_ref()?.tls12_session(key)
  }
  fn remove_tls12_session(&self, key: &ClientSessionKey<'static>) {
    if let Some(inner) = &self.inner {
      inner.remove_tls12_session(key);
    }
  }
  fn insert_tls13_ticket(&self, key: ClientSessionKey<'static>, value: rustls::client::Tls13Session) {
    self.tickets.fetch_add(1, Ordering::Relaxed);
    if let Some(inner) = &self.inner {
      inner.insert_tls13_ticket(key, value);
    }
  }
  fn take_tls13_ticket(&self, key: &ClientSessionKey<'static>) -> Option<rustls::client::Tls13Session> {
    self.inner.as_ref()?.take_tls13_ticket(key)
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
  /// Fail unless this endpoint completed exactly the selected handshake.
  /// rustls reports a HelloRetryRequest as its own kind; the flight count
  /// catches one on every stack.
  fn assert_kind(&self, resumed: bool, retry: bool, side: &str) {
    match self {
      Self::Rustls(peer) => assert_eq!(
        peer.connection.handshake_kind(),
        Some(if resumed {
          rustls::HandshakeKind::Resumed
        } else if retry {
          rustls::HandshakeKind::FullWithHelloRetryRequest
        } else {
          rustls::HandshakeKind::Full
        }),
        "{side} handshake kind"
      ),
      Self::OpenSsl(peer) => assert_eq!(peer.resumed(), resumed, "{side} full/resumed workload"),
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
  /// Check the negotiated suite, group, ALPN and the expected peer identity.
  #[expect(clippy::panic, reason = "a wrong negotiation invalidates this benchmark fixture")]
  fn check(&self, profile: Profile<'_>, workload: Workload, version: ProtocolVersion, resumed: bool, side: &str) {
    let credential = credentials::der(profile.credential, workload);
    let authenticated = side == "client" || workload != Workload::ServerAuth;
    match self {
      Self::Rustls(peer) => {
        let connection = &peer.connection;
        assert_eq!(connection.protocol_version(), Some(version), "{side} protocol version");
        if version == ProtocolVersion::TLSv1_2 {
          assert_eq!(connection.extended_main_secret(), Some(true), "{side} EMS");
        }
        assert_eq!(
          connection
            .negotiated_cipher_suite()
            .map(|negotiated| negotiated.suite()),
          Some(selected_suite(profile, version)),
          "{side} cipher suite"
        );
        assert_eq!(
          connection
            .negotiated_key_exchange_group()
            .map(|negotiated| negotiated.name()),
          (!(version == ProtocolVersion::TLSv1_2 && resumed)).then(|| group(profile.group)),
          "{side} key exchange group"
        );
        assert_eq!(
          connection.alpn_protocol(),
          Some(&rustls::enums::ApplicationProtocol::from(credentials::ALPN.to_vec())),
          "{side} ALPN"
        );
        match connection.peer_identity().map(|identity| identity.identity()) {
          Some(Identity::X509(chain)) => {
            assert!(authenticated, "{side} must have no peer identity");
            assert_eq!(chain.end_entity.as_ref(), credential.leaf, "{side} peer leaf");
            let expected = credential
              .intermediate
              .into_iter()
              .map(rustls::pki_types::CertificateDer::from)
              .collect::<Vec<_>>();
            assert_eq!(chain.intermediates, expected, "{side} intermediate chain");
          }
          None if !authenticated => {}
          _ => panic!("{side} peer certificate identity"),
        }
      }
      Self::OpenSsl(peer) => peer.check(profile, credential, authenticated, version, resumed, side),
    }
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
  /// Copy the decrypted bytes out of the endpoint's plaintext buffer. Every
  /// stack makes two plaintext copies: into that buffer, then into `bytes`.
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
  fn handshake(mut self, resumed: bool, retry: bool, flights: usize) -> Self {
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
        self.client.assert_kind(resumed, retry, "client");
        self.server.assert_kind(resumed, retry, "server");
        assert_eq!(self.flights.len(), flights, "TLS flight count");
        return self;
      }
    }
    panic!("TLS fixture exceeded its flight bound");
  }

  /// Transfer application bytes in one direction, including both record endpoints.
  pub fn transfer(&mut self, bytes: &[u8], output: &mut [u8]) {
    assert_eq!(bytes.len(), output.len());
    for (input, output) in bytes.chunks(MAX_PLAINTEXT).zip(output.chunks_mut(MAX_PLAINTEXT)) {
      self.client.send(input);
      self.server.step();
      self.server.receive(output);
    }
  }

  /// Send both directions before receiving each pair of records. `output` holds
  /// client-to-server bytes followed by server-to-client bytes; inputs differ.
  pub fn transfer_both(&mut self, client_bytes: &[u8], server_bytes: &[u8], output: &mut [u8]) {
    assert_eq!(client_bytes.len(), server_bytes.len());
    assert_eq!(output.len(), client_bytes.len().strict_mul(2));
    let (server_output, client_output) = output.split_at_mut(client_bytes.len());
    for ((client_input, server_input), (server_output, client_output)) in client_bytes
      .chunks(MAX_PLAINTEXT)
      .zip(server_bytes.chunks(MAX_PLAINTEXT))
      .zip(
        server_output
          .chunks_mut(MAX_PLAINTEXT)
          .zip(client_output.chunks_mut(MAX_PLAINTEXT)),
      )
    {
      self.client.send(client_input);
      self.server.send(server_input);
      self.client.step();
      self.server.step();
      self.server.receive(server_output);
      self.client.receive(client_output);
    }
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
  profile: Profile<'static>,
  workload: Workload,
  version: ProtocolVersion,
  /// TLS 1.3 tickets the client has accepted, across all connections.
  tickets: Arc<AtomicUsize>,
}

impl Fixture {
  pub fn new(backend: &str, profile: Profile<'_>) -> Self {
    Self::with_workload(backend, profile, Workload::Mutual)
  }

  pub fn with_workload(backend: &str, profile: Profile<'_>, workload: Workload) -> Self {
    Self::with_version(backend, profile, workload, ProtocolVersion::TLSv1_3)
  }

  /// TLS 1.2 mutual authentication, with session-ID resumption and no tickets.
  pub fn tls12(backend: &str, profile: Profile<'_>) -> Self {
    Self::with_version(backend, profile, Workload::Mutual, ProtocolVersion::TLSv1_2)
  }

  #[expect(clippy::panic, reason = "the benchmark catalog must select known workload values")]
  fn with_version(backend: &str, profile: Profile<'_>, workload: Workload, version: ProtocolVersion) -> Self {
    assert!(supports(backend, profile), "{backend} does not implement this profile");
    assert!(
      workload != Workload::Stateful || profile.resumed,
      "stateful resumption needs a resumed profile"
    );
    assert!(
      workload != Workload::HelloRetryRequest || (!profile.resumed && profile.group == "secp256r1"),
      "retry requests P-256 after X25519"
    );
    assert!(
      workload != Workload::CompactKey
        || (backend.starts_with("rscrypto") && profile.credential.starts_with("mldsa") && !profile.resumed),
      "compact signing is an rscrypto ML-DSA workload"
    );
    let profile = profile.canonical(version);
    let tickets = Arc::new(AtomicUsize::new(0));
    let configs = if backend == "openssl" {
      Configs::OpenSsl(openssl_peer::Configs::new(profile, workload, version, tickets.clone()))
    } else {
      let (mut provider, all, tls12) = match backend {
        "rscrypto" | "rscrypto-webpki" => (
          rscrypto_rustls::default_provider(),
          rscrypto_rustls::verify::ALL_ALGORITHMS,
          rscrypto_rustls::cipher_suite::ALL_TLS12_CIPHER_SUITES,
        ),
        "aws-lc-rs" => (
          rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER,
          rustls_aws_lc_rs::ALL_VERIFICATION_ALGS,
          rustls_aws_lc_rs::ALL_TLS12_CIPHER_SUITES,
        ),
        "ring" => (
          rustls_ring::DEFAULT_TLS13_PROVIDER,
          rustls_ring::ALL_VERIFICATION_ALGS,
          rustls_ring::ALL_TLS12_CIPHER_SUITES,
        ),
        _ => panic!("unknown benchmark backend"),
      };
      // Make each backend's opt-in groups selectable; the retain below keeps one.
      match backend {
        "rscrypto" | "rscrypto-webpki" => provider.kx_groups.to_mut().extend([
          rscrypto_rustls::kx_group::SECP256R1MLKEM768,
          rscrypto_rustls::kx_group::SECP384R1MLKEM1024,
          rscrypto_rustls::kx_group::MLKEM768,
          rscrypto_rustls::kx_group::MLKEM1024,
        ]),
        "aws-lc-rs" => provider.kx_groups = Cow::Borrowed(rustls_aws_lc_rs::ALL_KX_GROUPS),
        _ => {}
      }
      let (group, suite) = (group(profile.group), selected_suite(profile, version));
      let retry_group = provider
        .kx_groups
        .iter()
        .find(|entry| entry.name() == NamedGroup::X25519)
        .copied();
      // RFC 8422 section 5.1 also uses supported_groups for certificate curves.
      // Keep the measured KX first on both peers; verify the actual selection.
      let certificate_group = (version == ProtocolVersion::TLSv1_2)
        .then(|| certificate_group(profile))
        .flatten()
        .filter(|name| *name != group)
        .and_then(|name| provider.kx_groups.iter().find(|entry| entry.name() == name).copied());
      provider.kx_groups.to_mut().retain(|entry| entry.name() == group);
      if let Some(certificate_group) = certificate_group {
        provider.kx_groups.to_mut().push(certificate_group);
      }
      if version == ProtocolVersion::TLSv1_2 {
        provider.tls13_cipher_suites = Cow::Borrowed(&[]);
        provider.tls12_cipher_suites = Cow::Owned(
          tls12
            .iter()
            .copied()
            .filter(|entry| entry.common.suite == suite)
            .collect(),
        );
        assert_eq!(provider.tls12_cipher_suites.len(), 1);
      } else {
        provider
          .tls13_cipher_suites
          .to_mut()
          .retain(|entry| entry.common.suite == suite);
        assert_eq!(provider.tls13_cipher_suites.len(), 1);
      }
      let scheme = credentials::scheme(profile.credential);
      let mapping = provider.signature_verification_algorithms.mapping();
      let index = mapping
        .iter()
        .position(|(name, _)| *name == scheme)
        .expect("credential signature scheme");
      provider.signature_verification_algorithms =
        rustls::crypto::WebPkiSupportedAlgorithms::new(all, &mapping[index..index.strict_add(1)])
          .expect("selected signature scheme");
      assert_eq!(
        provider.kx_groups.len(),
        1_usize.strict_add(usize::from(certificate_group.is_some()))
      );
      let ticketer = provider.ticketer_factory;
      if workload == Workload::CompactKey {
        #[cfg(feature = "ml-dsa")]
        {
          static COMPACT: rscrypto_rustls::sign::Provider =
            rscrypto_rustls::sign::Provider::new(rscrypto_rustls::random::SECURE_RANDOM).compact();
          provider.key_provider = &COMPACT;
        }
        #[cfg(not(feature = "ml-dsa"))]
        panic!("compact ML-DSA requires the ml-dsa feature");
      }
      let mut client_provider = provider.clone();
      if workload == Workload::HelloRetryRequest {
        client_provider
          .kx_groups
          .to_mut()
          .insert(0, retry_group.expect("initial X25519 share"));
      }
      let (mut client, mut server) = credentials::configs(
        client_provider,
        provider,
        profile.credential,
        backend == "rscrypto",
        workload,
      );
      // rustls's default 256-entry store for resumed rows; no storage for full rows.
      let store = Resumption::store(Arc::new(CountingStore {
        inner: profile.resumed.then(|| ClientSessionMemoryCache::new(256)),
        tickets: tickets.clone(),
      }));
      if version == ProtocolVersion::TLSv1_2 {
        client.require_ems = true;
        server.require_ems = true;
        client.resumption = store.tls12_resumption(if profile.resumed {
          Tls12Resumption::SessionIdOnly
        } else {
          Tls12Resumption::Disabled
        });
        server.ticketer = None;
        server.session_storage = if profile.resumed {
          rustls::server::ServerSessionMemoryCache::new(256)
        } else {
          Arc::new(rustls::server::NoServerSessionStorage {})
        };
      } else if profile.resumed {
        client.resumption = store;
        // Each stack's production ticketer. OpenSSL issues at most one ticket
        // after a resumption, so every stack issues exactly one per connection.
        if workload == Workload::Stateful {
          server.ticketer = None;
          server.session_storage = rustls::server::ServerSessionMemoryCache::new(256);
        } else {
          server.ticketer = Some(ticketer.ticketer().expect("ticket key"));
        }
        server.send_tls13_tickets = rustls::server::Tls13Tickets { default: 1, max: 1 };
      } else {
        client.resumption = store.tls12_resumption(Tls12Resumption::Disabled);
        server.send_tls13_tickets = rustls::server::Tls13Tickets { default: 0, max: 0 };
      }
      Configs::Rustls(Arc::new(client), Arc::new(server))
    };
    let fixture = Self {
      configs,
      profile,
      workload,
      version,
      tickets,
    };
    // Seed the client cache once. Every timed connection must then complete the
    // selected handshake kind; a cache miss cannot masquerade as resumption.
    fixture.checked(false);
    fixture
  }

  /// Share Rustls signing keys and trust anchors with a separately seeded ticket store.
  ///
  /// TLS 1.3 consumes a ticket for each connection and keeps a bounded number per
  /// origin. Concurrent resumed observations therefore need independent stores.
  /// Cloning `ClientConfig` retains its Arc-owned key resolver and verifier.
  #[expect(clippy::panic, reason = "this resource boundary observes Rustls configurations only")]
  pub fn fork_with_resumption_cache(&self) -> Self {
    let Configs::Rustls(client, server) = &self.configs else {
      panic!("independent ticket stores require a Rustls fixture");
    };
    let tickets = Arc::new(AtomicUsize::new(0));
    let mut client = (**client).clone();
    client.resumption = Resumption::store(Arc::new(CountingStore {
      inner: self.profile.resumed.then(|| ClientSessionMemoryCache::new(256)),
      tickets: tickets.clone(),
    }))
    .tls12_resumption(if self.version == ProtocolVersion::TLSv1_2 && self.profile.resumed {
      Tls12Resumption::SessionIdOnly
    } else {
      Tls12Resumption::Disabled
    });
    let fixture = Self {
      configs: Configs::Rustls(Arc::new(client), server.clone()),
      profile: self.profile,
      workload: self.workload,
      version: self.version,
      tickets,
    };
    if fixture.profile.resumed {
      fixture.checked(false);
    }
    fixture
  }

  /// One timed-path connection: complete, of the selected kind, with the expected flights.
  pub fn connect(&self) -> Pair {
    self.connect_inner(self.profile.resumed)
  }

  /// A connection like [`Self::connect`], checked outside timing: ticket count,
  /// exporter equality, negotiated suite, group and ALPN, and peer identities.
  pub fn checked_connect(&self) -> Pair {
    self.checked(self.profile.resumed)
  }

  fn checked(&self, resumed: bool) -> Pair {
    let before = self.tickets.load(Ordering::Relaxed);
    let mut pair = self.connect_inner(resumed);
    assert_eq!(
      self.tickets.load(Ordering::Relaxed).strict_sub(before),
      usize::from(self.profile.resumed && self.version == ProtocolVersion::TLSv1_3),
      "TLS 1.3 tickets accepted from one connection"
    );
    assert_eq!(
      pair.client.exporter(),
      pair.server.exporter(),
      "independent endpoint exporters"
    );
    pair
      .client
      .check(self.profile, self.workload, self.version, resumed, "client");
    pair
      .server
      .check(self.profile, self.workload, self.version, resumed, "server");
    pair
  }

  fn connect_inner(&self, resumed: bool) -> Pair {
    self.finish_inner(self.begin_inner(resumed), resumed)
  }

  /// Construct both production connections without driving the handshake.
  ///
  /// The resource observer uses this boundary to hold every admitted handshake
  /// live before releasing its workers. Complete it with [`Self::finish`].
  pub fn begin(&self) -> Pair {
    self.begin_inner(self.profile.resumed)
  }

  fn begin_inner(&self, resumed: bool) -> Pair {
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
  }

  /// Drive a pair created by this fixture through the normal checked flights.
  pub fn finish(&self, pair: Pair) -> Pair {
    self.finish_inner(pair, self.profile.resumed)
  }

  fn finish_inner(&self, pair: Pair, resumed: bool) -> Pair {
    // TLS 1.2 full: CH, server flight, client flight, server Finished.
    // TLS 1.2 resumed: CH, server flight, client Finished; no KX or tickets.
    // TLS 1.3 has three flights plus one for tickets, and two for a retry.
    let flights = if self.version == ProtocolVersion::TLSv1_2 {
      if resumed { 3 } else { 4 }
    } else {
      3_usize.strict_add(usize::from(self.profile.resumed)).strict_add(
        if self.workload == Workload::HelloRetryRequest {
          2
        } else {
          0
        },
      )
    };
    pair.handshake(resumed, self.workload == Workload::HelloRetryRequest, flights)
  }
}
