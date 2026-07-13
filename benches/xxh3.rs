//! XXHash3 comparison benchmarks: rscrypto vs xxhash-rust crate.

mod common;

use core::{hash::BuildHasher, hint::black_box};
use std::collections::HashMap;

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

fn xxh3_build_hasher(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("xxh3-buildhasher");
  let ours = rscrypto::Xxh3BuildHasher::new();
  let upstream = xxhash_rust::xxh3::Xxh3DefaultBuilder::new();

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(ours.hash_one(black_box(d.as_slice()))))
    });
    g.bench_with_input(BenchmarkId::new("xxhash-rust", len), data, |b, d| {
      b.iter(|| black_box(upstream.hash_one(black_box(d.as_slice()))))
    });
  }
  g.finish();
}

fn xxh3_hashmap_lookup(c: &mut Criterion) {
  let key = common::random_bytes(32);
  let mut ours = HashMap::with_capacity_and_hasher(1, rscrypto::Xxh3BuildHasher::new());
  let mut upstream = HashMap::with_capacity_and_hasher(1, xxhash_rust::xxh3::Xxh3DefaultBuilder::new());
  ours.insert(key.as_slice(), 1u8);
  upstream.insert(key.as_slice(), 1u8);

  let mut g = c.benchmark_group("xxh3-hashmap/lookup-32");
  g.bench_function("rscrypto", |b| {
    b.iter(|| black_box(ours.get(black_box(key.as_slice()))))
  });
  g.bench_function("xxhash-rust", |b| {
    b.iter(|| black_box(upstream.get(black_box(key.as_slice()))))
  });
  g.finish();
}

criterion_group!(benches, xxh3_64, xxh3_128, xxh3_build_hasher, xxh3_hashmap_lookup);
criterion_main!(benches);
