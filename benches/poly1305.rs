//! Standalone Poly1305 construction, authentication, finalization, and cleanup.

#[path = "common/criterion.rs"]
mod bench_config;

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use dryoc::classic::crypto_onetimeauth::crypto_onetimeauth;
use rscrypto::{Poly1305, Poly1305OneTimeKey};

fn authenticate(c: &mut Criterion) {
  if !bench_config::selected("poly1305/authenticate") {
    return;
  }
  let key = [0x42; 32];
  let mut group = c.benchmark_group("poly1305/authenticate");
  // Empty-message setup; both sides of the 16-byte block boundary; short and
  // sustained block processing. Fixtures and oracle checks stay outside timing.
  for len in [0, 15, 16, 17, 64, 1024, 16384] {
    let data = vec![0xff; len];
    let mut expected = [0; 16];
    crypto_onetimeauth(&mut expected, &data, &key);
    assert_eq!(
      Poly1305::authenticate_once(Poly1305OneTimeKey::from_bytes(key), &data).to_bytes(),
      expected
    );
    if len != 0 {
      group.throughput(Throughput::Bytes(len as u64));
    }
    // Timed: construct and consume a key, authenticate, finalize, clear state,
    // and observe the tag. There is no per-operation allocation or input copy.
    group.bench_with_input(BenchmarkId::new("rscrypto", len), &data, |b, data| {
      b.iter(|| {
        black_box(Poly1305::authenticate_once(
          Poly1305OneTimeKey::from_bytes(*black_box(&key)),
          black_box(data),
        ))
      });
    });
  }
  group.finish();
}

fn main() {
  bench_config::run(&[authenticate]);
}
