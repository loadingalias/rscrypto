//! Fast non-cryptographic hash class benchmarks.

mod common;
mod hash_competitors;

use core::{hash::Hasher, hint::black_box};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

const V3_HI_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

fn fast_hash_64(c: &mut Criterion) {
  use core::hash::BuildHasher as _;

  let inputs = common::comp_sizes();
  let ahash_state = ahash::RandomState::with_seed(0);
  let foldhash_state = foldhash::fast::FixedState::default();
  let mut g = c.benchmark_group("fast-hash-64");

  for (len, data) in &inputs {
    let data = data.as_slice();
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::AesHash64::hash(black_box(d))))
    });

    hash_competitors::bench_gxhash64(&mut g, *len, data);

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

    g.bench_with_input(BenchmarkId::new("rapidhash", len), data, |b, d| {
      b.iter(|| {
        let mut h = rapidhash::fast::RapidHasher::new(0);
        h.write(black_box(d));
        black_box(h.finish())
      })
    });

    g.bench_with_input(BenchmarkId::new("xxhash-rust", len), data, |b, d| {
      b.iter(|| black_box(xxhash_rust::xxh3::xxh3_64(black_box(d))))
    });
  }

  g.finish();
}

fn fast_hash_128(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("fast-hash-128");

  for (len, data) in &inputs {
    let data = data.as_slice();
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::AesHash128::hash(black_box(d))))
    });

    hash_competitors::bench_gxhash128(&mut g, *len, data);

    g.bench_with_input(BenchmarkId::new("rapidhash", len), data, |b, d| {
      b.iter(|| {
        let mut lo_h = rapidhash::fast::RapidHasher::new(0);
        lo_h.write(black_box(d));
        let lo = lo_h.finish() as u128;

        let mut hi_h = rapidhash::fast::RapidHasher::new(V3_HI_SEED);
        hi_h.write(black_box(d));
        let hi = hi_h.finish() as u128;

        black_box(lo | (hi << 64))
      })
    });

    g.bench_with_input(BenchmarkId::new("xxhash-rust", len), data, |b, d| {
      b.iter(|| black_box(xxhash_rust::xxh3::xxh3_128(black_box(d))))
    });
  }

  g.finish();
}

criterion_group!(benches, fast_hash_64, fast_hash_128);
criterion_main!(benches);
