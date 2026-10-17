//! XXHash3 comparison benchmarks: rscrypto vs xxhash-rust crate.
//!
//! Before timing, every row must return xxhash-rust's hash of its input.

#[path = "common/criterion.rs"]
mod bench_config;

mod common;

use core::{hash::BuildHasher, hint::black_box};
use std::collections::HashMap;

use bench_config::checked;
use criterion::{BenchmarkId, Criterion};
use rscrypto::FastHash;

fn xxh3_64(c: &mut Criterion) {
  if !bench_config::selected("xxh3-64") {
    return;
  }
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("xxh3-64");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = xxhash_rust::xxh3::xxh3_64(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Xxh3::hash(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("xxhash-rust", len),
      checked(
        || xxhash_rust::xxh3::xxh3_64(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
  }

  g.finish();
}

fn xxh3_128(c: &mut Criterion) {
  if !bench_config::selected("xxh3-128") {
    return;
  }
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("xxh3-128");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = xxhash_rust::xxh3::xxh3_128(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Xxh3_128::hash(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("xxhash-rust", len),
      checked(
        || xxhash_rust::xxh3::xxh3_128(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
  }

  g.finish();
}

fn xxh3_build_hasher(c: &mut Criterion) {
  if !bench_config::selected("xxh3-buildhasher") {
    return;
  }
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("xxh3-buildhasher");
  let ours = rscrypto::Xxh3BuildHasher::new();
  let upstream = xxhash_rust::xxh3::Xxh3DefaultBuilder::new();

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    // Both builders stream the slice's length prefix and bytes into seed-0 XXH3-64.
    let expected = upstream.hash_one(data.as_slice());
    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || ours.hash_one(black_box(data.as_slice())),
        |out| assert_eq!(out, expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("xxhash-rust", len),
      checked(
        || upstream.hash_one(black_box(data.as_slice())),
        |out| assert_eq!(out, expected),
      ),
    );
  }
  g.finish();
}

fn xxh3_hashmap_lookup(c: &mut Criterion) {
  if !bench_config::selected("xxh3-hashmap/lookup-32") {
    return;
  }
  let key = common::random_bytes(32);
  let mut ours = HashMap::with_capacity_and_hasher(1, rscrypto::Xxh3BuildHasher::new());
  let mut upstream = HashMap::with_capacity_and_hasher(1, xxhash_rust::xxh3::Xxh3DefaultBuilder::new());
  ours.insert(key.as_slice(), 1u8);
  upstream.insert(key.as_slice(), 1u8);

  let mut g = c.benchmark_group("xxh3-hashmap/lookup-32");
  g.bench_function(
    "rscrypto",
    checked(|| ours.get(black_box(key.as_slice())), |out| assert_eq!(out, Some(&1))),
  );
  g.bench_function(
    "xxhash-rust",
    checked(
      || upstream.get(black_box(key.as_slice())),
      |out| assert_eq!(out, Some(&1)),
    ),
  );
  g.finish();
}

fn main() {
  bench_config::run(&[xxh3_64, xxh3_128, xxh3_build_hasher, xxh3_hashmap_lookup]);
}
