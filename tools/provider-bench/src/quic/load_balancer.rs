//! QUIC-LB connection-ID fixtures (draft-ietf-quic-load-balancers-22).
//!
//! Each shape is a draft Appendix B.2 row, so the untimed check compares the production result
//! with the draft's bytes.

use rscrypto_rustls::quic::{ConnectionId, FirstOctet, LoadBalancerConfig, LoadBalancerKey, RotationId, ServerId};

const KEY: [u8; 16] = [
  0x8f, 0x95, 0xf0, 0x92, 0x45, 0x76, 0x5f, 0x80, 0x25, 0x69, 0x34, 0xe5, 0x0c, 0x66, 0x20, 0x7f,
];
const SERVER_ID: [u8; 10] = [0xed, 0x79, 0x3a, 0x51, 0xd4, 0x9b, 0x8f, 0x5f, 0xab, 0x65];
const NONCE: [u8; 9] = [0xee, 0x08, 0x0d, 0xbf, 0x48, 0xc0, 0xd1, 0xe5, 0x5d];

/// The B.2 shapes: four-pass with three-pass decode, four-pass with four-pass decode, single
/// pass, and four-pass at an even length. The CID of the 9+9 row encodes rotation 0, whatever its
/// cr_bits column says.
pub const SHAPES: [&str; 4] = ["3+4", "10+5", "8+8", "9+9"];

const IDS: [(&str, &[u8]); 4] = [
  ("3+4", &[0x07, 0x20, 0xb1, 0xd0, 0x7b, 0x35, 0x9d, 0x3c]),
  (
    "10+5",
    &[
      0x2f, 0xcc, 0x38, 0x1b, 0xc7, 0x4c, 0xb4, 0xfb, 0xad, 0x28, 0x23, 0xa3, 0xd1, 0xf8, 0xfe, 0xd2,
    ],
  ),
  (
    "8+8",
    &[
      0x50, 0x4d, 0xd2, 0xd0, 0x5a, 0x7b, 0x0d, 0xe9, 0xb2, 0xb9, 0x90, 0x7a, 0xfb, 0x5e, 0xcf, 0x8c, 0xc3,
    ],
  ),
  (
    "9+9",
    &[
      0x12, 0x57, 0x79, 0xc9, 0xcc, 0x86, 0xbe, 0xb3, 0xa3, 0xa4, 0xa3, 0xca, 0x96, 0xfc, 0xe4, 0xbf, 0xe0, 0xcd, 0xbc,
    ],
  ),
];

/// One retained configuration with the server ID and nonce of its draft row.
pub struct LoadBalancerFixture {
  config: LoadBalancerConfig,
  id: &'static [u8],
}

impl LoadBalancerFixture {
  pub fn new(shape: &str) -> Self {
    let &(_, id) = IDS.iter().find(|(name, _)| *name == shape).expect("catalog shape");
    let (server_id_len, nonce_len) = shape.split_once('+').expect("catalog shape");
    let rotation = RotationId::new(id[0] >> 5).expect("draft rotation");
    let config = LoadBalancerConfig::new(
      rotation,
      server_id_len.parse().expect("server-ID length"),
      nonce_len.parse().expect("nonce length"),
      FirstOctet::Length,
      LoadBalancerKey::from_bytes(KEY),
    )
    .expect("draft lengths");
    let fixture = Self { config, id };
    assert_eq!(fixture.encode().as_slice(), id, "draft connection ID");
    assert_eq!(fixture.decode().as_slice(), fixture.server_id(), "draft server ID");
    fixture
  }

  fn server_id(&self) -> &'static [u8] {
    &SERVER_ID[..self.config.server_id_len()]
  }

  pub fn encode(&self) -> ConnectionId {
    self
      .config
      .encode(self.server_id(), &NONCE[..self.config.nonce_len()], &NoEntropy)
      .expect("configured lengths")
  }

  pub fn decode(&self) -> ServerId {
    self.config.decode(self.id).expect("routable")
  }
}

/// The length-describing first octet draws no entropy, so any draw is a defect.
#[derive(Debug)]
struct NoEntropy;

impl rustls::crypto::SecureRandom for NoEntropy {
  fn fill(&self, _output: &mut [u8]) -> Result<(), rustls::crypto::GetRandomFailed> {
    Err(rustls::crypto::GetRandomFailed)
  }
}
