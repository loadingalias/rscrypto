//! Full/resumed mutually authenticated TLS, and application records on reused sessions.

#[path = "../../../benches/common/criterion.rs"]
mod bench_config;

use core::hint::black_box;
use criterion::{Criterion, Throughput};
use rscrypto_provider_bench::{BACKENDS, CREDENTIALS, Fixture, GROUPS, Profile, SUITES};

fn connections(c: &mut Criterion) {
  for credential in CREDENTIALS {
    for group in GROUPS {
      for suite in SUITES {
        for resumed in [false, true] {
          let mode = if resumed { "resumed" } else { "full" };
          let name = format!("tls13/handshake/{credential}/{group}/{suite}/{mode}");
          if !bench_config::selected(&name) {
            continue;
          }
          let mut measured = c.benchmark_group(&name);
          measured.throughput(Throughput::Elements(1));
          for backend in BACKENDS {
            if !bench_config::selected(&format!("{name}/{backend}")) {
              continue;
            }
            let fixture = Fixture::new(
              backend,
              Profile {
                credential,
                group,
                suite,
                resumed,
              },
            );
            fixture.connect().verify();
            measured.bench_function(backend, |b| b.iter(|| drop(black_box(fixture.connect()))));
          }
          measured.finish();
        }
      }
    }
  }
}

fn traffic(c: &mut Criterion) {
  for suite in SUITES {
    for size in [1024usize, 16_384] {
      let name = format!("tls13/traffic/{suite}/{size}");
      if !bench_config::selected(&name) {
        continue;
      }
      let mut measured = c.benchmark_group(&name);
      measured.throughput(Throughput::Bytes(u64::try_from(size).expect("record size")));
      for backend in BACKENDS {
        if !bench_config::selected(&format!("{name}/{backend}")) {
          continue;
        }
        let fixture = Fixture::new(
          backend,
          Profile {
            credential: "p256",
            group: "X25519MLKEM768",
            suite,
            resumed: false,
          },
        );
        let mut pair = fixture.connect();
        pair.verify();
        let bytes = vec![0x42; size];
        let mut output = vec![0; size];
        pair.transfer(&bytes, &mut output);
        assert_eq!(bytes, output, "authenticated transfer before timing");
        measured.bench_function(backend, |b| {
          b.iter(|| {
            pair.transfer(black_box(&bytes), black_box(&mut output));
            black_box(&output);
          })
        });
      }
      measured.finish();
    }
  }
}

fn main() {
  bench_config::run(&[connections, traffic]);
}
