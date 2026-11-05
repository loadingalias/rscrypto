//! QUIC v1 handshake, packet-protection, and header-protection workloads.

#[path = "../../../benches/common/criterion.rs"]
#[expect(dead_code, reason = "fixtures are registered lazily through custom routines")]
mod bench_config;

use core::hint::black_box;
use criterion::{Criterion, Throughput};
use rscrypto_provider_bench::{
  SUITES,
  quic::{CRYPTO_BACKENDS, CryptoFixture, HeaderFixture, PacketFixture, TLS_BACKENDS},
};

fn handshakes(c: &mut Criterion) {
  for suite in SUITES {
    let name = format!("quic/handshake/p256/X25519/{suite}/full");
    if !bench_config::selected(&name) {
      continue;
    }
    let mut group = c.benchmark_group(&name);
    group.throughput(Throughput::Elements(1));
    for backend in TLS_BACKENDS {
      if !bench_config::selected(&format!("{name}/{backend}")) {
        continue;
      }
      group.bench_function(
        backend,
        bench_config::checked_routine(
          None,
          |fixture| *fixture = Some(CryptoFixture::new(backend, suite)),
          |b, fixture| b.iter(|| drop(black_box(fixture.as_ref().expect("checked").connect()))),
        ),
      );
    }
    group.finish();
  }
}

fn packets(c: &mut Criterion) {
  for suite in SUITES {
    for len in [1200, 1350] {
      for operation in ["seal", "open"] {
        let name = format!("quic/packet-{operation}/{suite}/{len}");
        if !bench_config::selected(&name) {
          continue;
        }
        let mut group = c.benchmark_group(&name);
        group.throughput(Throughput::Bytes(len as u64));
        for backend in CRYPTO_BACKENDS {
          if !bench_config::selected(&format!("{name}/{backend}")) {
            continue;
          }
          group.bench_function(
            backend,
            bench_config::checked_routine(
              None,
              |fixture| *fixture = Some(PacketFixture::new(backend, suite, len)),
              |b, fixture| {
                let fixture = fixture.as_mut().expect("checked");
                if operation == "seal" {
                  b.iter(|| {
                    black_box(black_box(&mut *fixture).seal());
                  });
                } else {
                  b.iter(|| {
                    black_box(black_box(&mut *fixture).open());
                  });
                }
              },
            ),
          );
        }
        group.finish();
      }
    }
    for operation in ["protect", "unprotect"] {
      let name = format!("quic/header-{operation}/{suite}/short-pn4");
      if !bench_config::selected(&name) {
        continue;
      }
      let mut group = c.benchmark_group(&name);
      group.throughput(Throughput::Elements(1));
      for backend in CRYPTO_BACKENDS {
        if !bench_config::selected(&format!("{name}/{backend}")) {
          continue;
        }
        group.bench_function(
          backend,
          bench_config::checked_routine(
            None,
            |fixture| *fixture = Some(HeaderFixture::new(backend, suite)),
            |b, fixture| {
              let fixture = fixture.as_ref().expect("checked");
              if operation == "protect" {
                b.iter(|| black_box(black_box(fixture).protect()));
              } else {
                b.iter(|| black_box(black_box(fixture).unprotect()));
              }
            },
          ),
        );
      }
      group.finish();
    }
  }
}

fn main() {
  if !std::env::args().any(|arg| arg == "--list" || arg == "--rscrypto-filter-cases") {
    assert!(
      openssl::version::version().starts_with("OpenSSL 4.0.3 "),
      "catalog prerequisite: OpenSSL 4.0.3"
    );
    eprintln!(
      "{}",
      serde_json::json!({ "provider_bench_provenance": rscrypto_provider_bench::provenance() })
    );
  }
  bench_config::run(&[handshakes, packets]);
}
