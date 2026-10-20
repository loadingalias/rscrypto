//! TLS event transport and component fixtures; every cryptographic operation
//! is supplied by a real provider through rustls's QUIC interfaces.

use alloc::sync::Arc;

use rustls::crypto::{Identity, tls13::OkmBlock};
use rustls::quic::{self, KeyChange, QuicEvent};
use rustls::{ClientConfig, ClientSide, Connection, ServerConfig, ServerSide, SideData, VecInput};

use super::{configs, oracle};
use crate::{Workload, credentials, group, suite};

// Valid active_connection_id_limit parameters, distinct to detect swapped peers.
const CLIENT_PARAMS: &[u8] = &[0x0e, 1, 2];
const SERVER_PARAMS: &[u8] = &[0x0e, 1, 3];

pub struct CryptoFixture {
  client: Arc<ClientConfig>,
  server: Arc<ServerConfig>,
  suite: String,
}

pub struct CryptoPair {
  client: Connection<ClientSide, quic::Quic>,
  server: Connection<ServerSide, quic::Quic>,
  keys: [Option<quic::Keys>; 2],
  changes: [usize; 2],
}

impl CryptoFixture {
  pub fn new(backend: &str, cipher: &str) -> Self {
    let (client, server) = configs(backend, cipher);
    let fixture = Self {
      client,
      server,
      suite: cipher.to_owned(),
    };
    fixture.check(fixture.connect());
    // Interoperability is independent of a provider agreeing with itself.
    let (client, server) = configs(oracle(backend), cipher);
    fixture.check(connect(fixture.client.clone(), server));
    fixture.check(connect(client, fixture.server.clone()));
    fixture
  }

  /// Both endpoints, event copies, key derivation and connection allocation.
  /// Dropping the returned pair is part of the caller's timed operation.
  pub fn connect(&self) -> CryptoPair {
    connect(self.client.clone(), self.server.clone())
  }

  #[expect(
    clippy::panic,
    reason = "a missing server identity invalidates this benchmark fixture"
  )]
  fn check(&self, mut pair: CryptoPair) {
    fn check_peer<S: SideData>(peer: &Connection<S, quic::Quic>, cipher: &str) {
      assert!(!peer.is_handshaking());
      assert_eq!(peer.handshake_kind(), Some(rustls::HandshakeKind::Full));
      assert_eq!(peer.negotiated_cipher_suite().map(|s| s.suite()), Some(suite(cipher)));
      assert_eq!(
        peer.negotiated_key_exchange_group().map(|g| g.name()),
        Some(group("X25519"))
      );
      assert_eq!(peer.alpn_protocol(), Some(&credentials::ALPN.to_vec().into()));
    }
    check_peer(&pair.client, &self.suite);
    check_peer(&pair.server, &self.suite);
    assert_eq!(pair.client.quic_transport_parameters(), Some(SERVER_PARAMS));
    assert_eq!(pair.server.quic_transport_parameters(), Some(CLIENT_PARAMS));
    assert_eq!(pair.changes, [2, 2], "Handshake then 1-RTT keys on both peers");
    let Some(Identity::X509(chain)) = pair.client.peer_identity().map(|p| p.identity()) else {
      panic!("authenticated server certificate required")
    };
    assert_eq!(
      chain.end_entity.as_ref(),
      credentials::der("p256", Workload::ServerAuth).leaf
    );
    assert!(pair.server.peer_identity().is_none(), "server-only authentication");
    let mut exported = [[0; 32]; 2];
    pair
      .client
      .exporter()
      .expect("client exporter")
      .derive(b"quic-bench", Some(b"matched"), &mut exported[0])
      .expect("derive");
    pair
      .server
      .exporter()
      .expect("server exporter")
      .derive(b"quic-bench", Some(b"matched"), &mut exported[1])
      .expect("derive");
    assert_eq!(exported[0], exported[1]);
    // Check the traffic keys from the handshake, not just its status flag.
    let client = pair.keys[0].as_ref().expect("client 1-RTT keys");
    let server = pair.keys[1].as_ref().expect("server 1-RTT keys");
    for (send, receive) in [(&client.local, &server.remote), (&server.local, &client.remote)] {
      let mut bytes = b"authenticated QUIC payload".to_vec();
      let tag = send
        .packet
        .encrypt_in_place(0, b"header", &mut bytes, None)
        .expect("seal");
      bytes.extend_from_slice(tag.as_ref());
      assert_eq!(
        receive
          .packet
          .decrypt_in_place(0, b"header", &mut bytes, None)
          .expect("open"),
        b"authenticated QUIC payload"
      );
    }
  }
}

#[expect(
  clippy::panic,
  reason = "a stalled production handshake invalidates this bounded benchmark fixture"
)]
fn connect(client: Arc<ClientConfig>, server: Arc<ServerConfig>) -> CryptoPair {
  let mut pair = CryptoPair {
    client: client
      .connect(rustls::pki_types::ServerName::try_from("localhost").expect("name"))
      .build_quic(quic::Version::V1, CLIENT_PARAMS.to_vec())
      .expect("QUIC client"),
    server: Connection::<ServerSide, quic::Quic>::new_quic(server, quic::Version::V1, SERVER_PARAMS.to_vec())
      .expect("QUIC server"),
    keys: [None, None],
    changes: [0, 0],
  };
  let mut inputs = [VecInput::default(), VecInput::default()];
  for _ in 0..16 {
    let mut progress = false;
    transfer(
      &mut pair.client,
      &mut pair.server,
      &mut inputs[1],
      &mut pair.keys[0],
      &mut pair.changes[0],
      &mut progress,
    );
    transfer(
      &mut pair.server,
      &mut pair.client,
      &mut inputs[0],
      &mut pair.keys[1],
      &mut pair.changes[1],
      &mut progress,
    );
    if !progress && !pair.client.is_handshaking() && !pair.server.is_handshaking() {
      assert!(pair.keys.iter().all(Option::is_some), "complete 1-RTT key schedule");
      return pair;
    }
  }
  panic!("QUIC crypto handshake exceeded its event bound")
}

#[expect(
  clippy::panic,
  reason = "an unexpected QUIC event invalidates this benchmark fixture"
)]
fn transfer<S: SideData, D: SideData>(
  source: &mut Connection<S, quic::Quic>,
  destination: &mut Connection<D, quic::Quic>,
  input: &mut VecInput,
  keys: &mut Option<quic::Keys>,
  changes: &mut usize,
  progress: &mut bool,
) {
  for event in source.events() {
    *progress = true;
    match event {
      QuicEvent::Message(bytes) => {
        assert!(bytes.len() < 131_072, "bounded TLS flight");
        // Never concatenate data across an encryption-level change.
        let mut remaining = bytes.as_slice();
        while !remaining.is_empty() {
          input.read(&mut remaining).expect("bounded TLS input");
          destination.read_hs(input).expect("QUIC handshake message");
        }
      }
      QuicEvent::KeyChange(change) => {
        *changes = changes.strict_add(1);
        match change {
          KeyChange::Handshake { .. } => {}
          KeyChange::OneRtt { keys: next, .. } => *keys = Some(next),
        }
      }
      _ => panic!("unhandled QUIC event"),
    }
  }
}

#[expect(clippy::panic, reason = "the benchmark catalog must select known workload values")]
fn keys(backend: &str, cipher: &str) -> quic::DirectionalKeys {
  let provider = match backend {
    "rscrypto" => rscrypto_rustls::default_provider(),
    "aws-lc-rs" => rustls_aws_lc_rs::DEFAULT_TLS13_PROVIDER,
    "ring" => rustls_ring::DEFAULT_TLS13_PROVIDER,
    _ => panic!("unknown packet provider"),
  };
  let cipher = provider
    .tls13_cipher_suites
    .iter()
    .find(|s| s.common.suite == suite(cipher))
    .expect("suite");
  let secret = OkmBlock::new(&[0x42; 48][..cipher.common.hash_provider.output_len()]);
  let builder = quic::KeyBuilder::new(
    &secret,
    quic::Version::V1,
    cipher.quic.expect("QUIC"),
    cipher.hkdf_provider,
  );
  quic::DirectionalKeys {
    packet: builder.packet_key(),
    header: builder.header_protection_key(),
  }
}

/// In-place 1-RTT payload protection with retained storage and keys. Buffer
/// restoration and packet-number encoding are included in each operation.
pub struct PacketFixture {
  key: Box<dyn quic::PacketKey>,
  plaintext: Vec<u8>,
  ciphertext: Vec<u8>,
  buffer: Vec<u8>,
  next: u64,
}

fn header(number: u64) -> [u8; 13] {
  let mut header = [0x43, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0];
  header[9..].copy_from_slice(
    &u32::try_from(number)
      .expect("four-byte packet number budget")
      .to_be_bytes(),
  );
  header
}

impl PacketFixture {
  pub fn new(backend: &str, cipher: &str, len: usize) -> Self {
    let key = keys(backend, cipher).packet;
    let other = keys(oracle(backend), cipher).packet;
    let plaintext: Vec<u8> = (0..len).map(|i| i.to_le_bytes()[0]).collect();
    let mut ciphertext = plaintext.clone();
    let tag = other
      .encrypt_in_place(1, &header(1), &mut ciphertext, None)
      .expect("independent seal");
    ciphertext.extend_from_slice(tag.as_ref());
    let mut fixture = Self {
      key,
      buffer: vec![0; ciphertext.len()],
      plaintext,
      ciphertext,
      next: 1,
    };
    fixture.seal();
    assert_eq!(fixture.buffer, fixture.ciphertext, "independent ciphertext and tag");
    let expected = fixture.plaintext.clone();
    assert_eq!(fixture.open(), expected, "independent peer ciphertext");
    // Authenticate all inputs; a round-trip alone would miss an ignored AAD or tag.
    for fault in 0..3 {
      let mut bytes = fixture.ciphertext.clone();
      let mut aad = header(1);
      let mut pn = 1;
      match fault {
        0 => *bytes.last_mut().expect("tag") ^= 1,
        1 => aad[1] ^= 1,
        _ => pn = 2,
      }
      assert!(
        fixture.key.decrypt_in_place(pn, &aad, &mut bytes, None).is_err(),
        "authentication fault {fault}"
      );
    }
    fixture
  }

  pub fn seal(&mut self) -> &[u8] {
    let len = self.plaintext.len();
    self.buffer[..len].copy_from_slice(&self.plaintext);
    let aad = header(self.next);
    let tag = self
      .key
      .encrypt_in_place(self.next, &aad, &mut self.buffer[..len], None)
      .expect("seal");
    self.buffer[len..].copy_from_slice(tag.as_ref());
    self.next = self.next.strict_add(1);
    &self.buffer
  }

  pub fn open(&mut self) -> &[u8] {
    self.buffer.copy_from_slice(&self.ciphertext);
    self
      .key
      .decrypt_in_place(1, &header(1), &mut self.buffer, None)
      .expect("open")
  }
}

/// Short-header protection with a four-byte packet number and a 16-byte sample.
pub struct HeaderFixture {
  key: Box<dyn quic::HeaderProtectionKey>,
  protected: (u8, [u8; 4]),
}

const SAMPLE: [u8; 16] = [
  0x9d, 0x3c, 0x91, 0x27, 0x88, 0xa4, 0x11, 0x62, 0x80, 0x43, 0xc0, 0x7b, 0xa5, 0x16, 0x1e, 0xf2,
];
const PLAIN: (u8, [u8; 4]) = (0x43, [1, 2, 3, 4]);

impl HeaderFixture {
  pub fn new(backend: &str, cipher: &str) -> Self {
    let key = keys(backend, cipher).header;
    let other = keys(oracle(backend), cipher).header;
    assert_eq!(key.sample_len(), SAMPLE.len());
    let mut protected = PLAIN;
    other
      .encrypt_in_place(&SAMPLE, &mut protected.0, &mut protected.1)
      .expect("independent mask");
    let fixture = Self { key, protected };
    assert_eq!(fixture.protect(), protected, "independent protected header");
    assert_eq!(fixture.unprotect(), PLAIN, "independent unprotected header");
    fixture
  }

  pub fn protect(&self) -> (u8, [u8; 4]) {
    let mut bytes = PLAIN;
    self
      .key
      .encrypt_in_place(&SAMPLE, &mut bytes.0, &mut bytes.1)
      .expect("header protection");
    bytes
  }

  pub fn unprotect(&self) -> (u8, [u8; 4]) {
    let mut bytes = self.protected;
    self
      .key
      .decrypt_in_place(&SAMPLE, &mut bytes.0, &mut bytes.1)
      .expect("header removal");
    bytes
  }
}
