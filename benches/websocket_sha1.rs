//! RFC 6455 WebSocket accept-digest comparison benchmark.
//!
//! The fixture is the RFC 6455 section 1.3 sample key. Before timing, both rows
//! must return the SHA-1 digest of its published accept value.

#[path = "common/criterion.rs"]
mod bench_config;

use core::hint::black_box;

use bench_config::checked;
use criterion::{Criterion, Throughput};
use rscrypto::hashes::legacy::WebSocketAcceptDigest;
use sha1::{Digest as _, Sha1};

const WEBSOCKET_GUID: &[u8] = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
const REPRESENTATIVE_KEY: &[u8] = b"dGhlIHNhbXBsZSBub25jZQ==";
/// SHA-1 digest whose base64 form is the RFC 6455 accept value `s3pPLMBiTxaQ9kYGzzhZRbK+xOo=`.
const RFC6455_ACCEPT_DIGEST: [u8; 20] = [
  0xb3, 0x7a, 0x4f, 0x2c, 0xc0, 0x62, 0x4f, 0x16, 0x90, 0xf6, 0x46, 0x06, 0xcf, 0x38, 0x59, 0x45, 0xb2, 0xbe, 0xc4,
  0xea,
];

fn websocket_accept_digest(c: &mut Criterion) {
  if !bench_config::selected("websocket-accept-digest/24-byte-key") {
    return;
  }
  let mut group = c.benchmark_group("websocket-accept-digest/24-byte-key");
  group.throughput(Throughput::Bytes(
    REPRESENTATIVE_KEY.len().strict_add(WEBSOCKET_GUID.len()) as u64,
  ));

  group.bench_function(
    "rscrypto",
    checked(
      || WebSocketAcceptDigest::compute(black_box(REPRESENTATIVE_KEY)),
      |digest| assert_eq!(digest.as_ref(), RFC6455_ACCEPT_DIGEST),
    ),
  );

  group.bench_function(
    "rustcrypto",
    checked(
      || {
        let mut sha1 = Sha1::new();
        sha1.update(black_box(REPRESENTATIVE_KEY));
        sha1.update(WEBSOCKET_GUID);
        sha1.finalize()
      },
      |digest| assert_eq!(digest.as_slice(), RFC6455_ACCEPT_DIGEST),
    ),
  );

  group.finish();
}

fn main() {
  bench_config::run(&[websocket_accept_digest]);
}
