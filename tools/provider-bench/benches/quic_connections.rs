//! QUIC v1 component and complete Quinn connection workloads.

#[path = "../../../benches/common/criterion.rs"]
#[expect(dead_code, reason = "fixtures are registered lazily through custom routines")]
mod bench_config;

use core::hint::black_box;
use criterion::{Criterion, Throughput};
use rscrypto_provider_bench::{
  SUITES,
  quic::{
    CRYPTO_BACKENDS, CryptoFixture, HeaderFixture, NetworkFixture, NetworkMode, PacketFixture, ResetFixture,
    TLS_BACKENDS,
  },
};

fn handshakes(c: &mut Criterion) {
  for suite in SUITES {
    for (family, mode) in [
      ("quic/handshake", None),
      ("quinn/connect", Some(NetworkMode::Connect)),
      ("quinn/retry", Some(NetworkMode::Retry)),
      ("quinn/address-token", Some(NetworkMode::AddressToken)),
    ] {
      let name = format!("{family}/p256/X25519/{suite}/full");
      if !bench_config::selected(&name) {
        continue;
      }
      let mut group = c.benchmark_group(&name);
      group.throughput(Throughput::Elements(1));
      for backend in TLS_BACKENDS {
        if !bench_config::selected(&format!("{name}/{backend}")) {
          continue;
        }
        if let Some(mode) = mode {
          group.bench_function(
            backend,
            bench_config::checked_routine(
              None,
              |fixture| *fixture = Some(NetworkFixture::new(backend, suite, mode)),
              |b, fixture| b.iter(|| drop(black_box(fixture.as_ref().expect("checked").connect()))),
            ),
          );
        } else {
          group.bench_function(
            backend,
            bench_config::checked_routine(
              None,
              |fixture| *fixture = Some(CryptoFixture::new(backend, suite)),
              |b, fixture| b.iter(|| drop(black_box(fixture.as_ref().expect("checked").connect()))),
            ),
          );
        }
      }
      group.finish();
    }
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

fn resets(c: &mut Criterion) {
  let name = "quinn/stateless-reset/emit/rscrypto";
  if !bench_config::selected(name) {
    return;
  }
  c.bench_function(
    name,
    bench_config::checked_routine(
      None,
      |fixture| *fixture = Some(ResetFixture::checked()),
      |b, fixture| {
        b.iter(|| {
          black_box(fixture.as_mut().expect("checked").emit());
        })
      },
    ),
  );
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
  bench_config::run(&[handshakes, packets, resets]);
}
