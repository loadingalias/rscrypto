//! XXHash3 comparison benchmarks: rscrypto vs xxhash-rust crate.

mod common;

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rscrypto::FastHash;

fn xxh3_64(c: &mut Criterion) {
  use core::hash::{BuildHasher as _, Hasher as _};

  let inputs = common::comp_sizes();
  let ahash_state = ahash::RandomState::with_seed(0);
  let foldhash_state = foldhash::fast::FixedState::default();
  let mut g = c.benchmark_group("xxh3-64");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::Xxh3::hash(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("xxhash-rust", len), data, |b, d| {
      b.iter(|| black_box(xxhash_rust::xxh3::xxh3_64(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("gxhash", len), data, |b, d| {
      b.iter(|| black_box(gxhash::gxhash64(black_box(d), 0)))
    });

    g.bench_with_input(BenchmarkId::new("ahash", len), data, |b, d| {
      b.iter(|| {
        let mut h = ahash_state.build_hasher();
        h.write(black_box(d));
        black_box(h.finish())
      })
    });

    g.bench_with_input(BenchmarkId::new("foldhash", len), data, |b, d| {
      b.iter(|| {
        let mut h = foldhash_state.build_hasher();
        h.write(black_box(d));
        black_box(h.finish())
      })
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

    // ahash and foldhash have no native 128-bit output API; gxhash does.
    g.bench_with_input(BenchmarkId::new("gxhash", len), data, |b, d| {
      b.iter(|| black_box(gxhash::gxhash128(black_box(d), 0)))
    });
  }

  g.finish();
}

criterion_group!(benches, xxh3_64, xxh3_128);
criterion_main!(benches);
