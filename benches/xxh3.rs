//! XXHash3 comparison benchmarks: rscrypto vs xxhash-rust crate.

mod common;

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rscrypto::FastHash;

fn xxh3_64(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("xxh3-64");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::Xxh3::hash(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("xxhash-rust", len), data, |b, d| {
      b.iter(|| black_box(xxhash_rust::xxh3::xxh3_64(black_box(d))))
    });
  }

  g.finish();
}

fn xxh3_128(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("xxh3-128");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::Xxh3_128::hash(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("xxhash-rust", len), data, |b, d| {
      b.iter(|| black_box(xxhash_rust::xxh3::xxh3_128(black_box(d))))
    });
  }

  g.finish();
}

#[cfg(feature = "diag")]
fn xxh3_128_diagnostic(c: &mut Criterion) {
  let inputs: Vec<_> = [1024, 4096, 16384]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect();
  let mut g = c.benchmark_group("xxh3-128/diagnostic");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("auto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::hashes::fast::xxh3::diagnostics::hash128_auto(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("portable", len), data, |b, d| {
      b.iter(|| {
        black_box(rscrypto::hashes::fast::xxh3::diagnostics::hash128_portable(black_box(
          d,
        )))
      })
    });

    #[cfg(target_arch = "aarch64")]
    {
      g.bench_with_input(BenchmarkId::new("neon-prefetch", len), data, |b, d| {
        b.iter(|| {
          black_box(rscrypto::hashes::fast::xxh3::diagnostics::hash128_neon_prefetch(
            black_box(d),
          ))
        })
      });

      g.bench_with_input(BenchmarkId::new("neon-no-prefetch", len), data, |b, d| {
        b.iter(|| {
          black_box(rscrypto::hashes::fast::xxh3::diagnostics::hash128_neon_no_prefetch(
            black_box(d),
          ))
        })
      });
    }

    g.bench_with_input(BenchmarkId::new("xxhash-rust", len), data, |b, d| {
      b.iter(|| black_box(xxhash_rust::xxh3::xxh3_128(black_box(d))))
    });
  }

  g.finish();
}

#[cfg(not(feature = "diag"))]
criterion_group!(benches, xxh3_64, xxh3_128);
#[cfg(feature = "diag")]
criterion_group!(benches, xxh3_64, xxh3_128, xxh3_128_diagnostic);
criterion_main!(benches);
