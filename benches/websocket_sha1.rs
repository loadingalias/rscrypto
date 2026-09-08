//! RFC 6455 WebSocket accept-digest comparison benchmark.

#[path = "common/criterion.rs"]
mod bench_config;

use core::hint::black_box;

use criterion::{Criterion, Throughput};
use rscrypto::hashes::legacy::WebSocketAcceptDigest;
use sha1::{Digest as _, Sha1};

const WEBSOCKET_GUID: &[u8] = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
const REPRESENTATIVE_KEY: &[u8] = b"dGhlIHNhbXBsZSBub25jZQ==";

fn websocket_accept_digest(c: &mut Criterion) {
  if !bench_config::selected("websocket-accept-digest/24-byte-key") {
    return;
  }
  let mut group = c.benchmark_group("websocket-accept-digest/24-byte-key");
  group.throughput(Throughput::Bytes(
    REPRESENTATIVE_KEY.len().strict_add(WEBSOCKET_GUID.len()) as u64,
  ));

  group.bench_function("rscrypto", |b| {
    b.iter(|| black_box(WebSocketAcceptDigest::compute(black_box(REPRESENTATIVE_KEY))))
  });

  group.bench_function("rustcrypto", |b| {
    b.iter(|| {
      let mut sha1 = Sha1::new();
      sha1.update(black_box(REPRESENTATIVE_KEY));
      sha1.update(WEBSOCKET_GUID);
      black_box(sha1.finalize())
    })
  });

  group.finish();
}

fn main() {
  bench_config::run(&[websocket_accept_digest]);
}
