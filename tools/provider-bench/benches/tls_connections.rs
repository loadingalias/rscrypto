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
            let profile = Profile {
              credential,
              group,
              suite,
              resumed,
            };
            // Discovery registers names without running routines, so build and
            // check each fixture only when its case runs.
            let mut fixture = None;
            measured.bench_function(backend, |b| {
              let fixture = fixture.get_or_insert_with(|| {
                let fixture = Fixture::new(backend, profile);
                fixture.connect().verify();
                fixture
              });
              b.iter(|| drop(black_box(fixture.connect())));
            });
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
        let bytes = vec![0x42; size];
        let mut output = vec![0; size];
        let mut state = None;
        measured.bench_function(backend, |b| {
          let (_fixture, pair) = state.get_or_insert_with(|| {
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
            pair.transfer(&bytes, &mut output);
            assert_eq!(bytes, output, "authenticated transfer before timing");
            (fixture, pair)
          });
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
