//! RapidHash V3 comparison benchmarks.

mod common;

use core::{
  hash::{BuildHasher, Hasher},
  hint::black_box,
};
use std::collections::HashMap;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn rapidhash_v3_64(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut group = c.benchmark_group("rapidhash-v3-64");

  for (len, data) in &inputs {
    common::set_throughput(&mut group, *len);

    group.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, data| {
      b.iter(|| black_box(rscrypto::RapidHash64::hash(black_box(data))))
    });

    group.bench_with_input(BenchmarkId::new("rapidhash", len), data, |b, data| {
      let secrets = rapidhash::v3::RapidSecrets::seed_cpp(0);
      b.iter(|| black_box(rapidhash::v3::rapidhash_v3_seeded(black_box(data), &secrets)))
    });
  }

  group.finish();
}

fn rapidhash_seeded_state(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut group = c.benchmark_group("rapidhash-buildhasher");
  let ours = rscrypto::RapidSeededState::new(0);
  let upstream = rapidhash::quality::SeedableState::fixed();

  for (len, data) in &inputs {
    common::set_throughput(&mut group, *len);
    group.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, data| {
      b.iter(|| black_box(ours.hash_one(black_box(data.as_slice()))))
    });
    group.bench_with_input(BenchmarkId::new("rapidhash", len), data, |b, data| {
      b.iter(|| black_box(upstream.hash_one(black_box(data.as_slice()))))
    });
  }
  group.finish();
}

fn rapidhash_hashmap_lookup(c: &mut Criterion) {
  let key = common::random_bytes(32);
  let mut ours = HashMap::with_capacity_and_hasher(1, rscrypto::RapidSeededState::new(0));
  let mut upstream = HashMap::with_capacity_and_hasher(1, rapidhash::quality::SeedableState::fixed());
  ours.insert(key.as_slice(), 1u8);
  upstream.insert(key.as_slice(), 1u8);

  let mut group = c.benchmark_group("rapidhash-hashmap/lookup-32");
  group.bench_function("rscrypto", |b| {
    b.iter(|| black_box(ours.get(black_box(key.as_slice()))))
  });
  group.bench_function("rapidhash", |b| {
    b.iter(|| black_box(upstream.get(black_box(key.as_slice()))))
  });
  group.finish();
}

fn rapidhash_streaming(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let secrets = rapidhash::v3::RapidSecrets::seed_cpp(0);

  for (group_name, chunk_size) in [
    ("rapidhash-stream/one-write", usize::MAX),
    ("rapidhash-stream/chunk-64", 64),
  ] {
    let mut group = c.benchmark_group(group_name);
    for (len, data) in &inputs {
      common::set_throughput(&mut group, *len);

      group.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, data| {
        b.iter(|| {
          let mut hasher = rscrypto::RapidStreamHasher::new();
          if chunk_size == usize::MAX {
            hasher.write(black_box(data));
          } else {
            for chunk in data.chunks(chunk_size) {
              hasher.write(black_box(chunk));
            }
          }
          black_box(hasher.finish())
        })
      });

      group.bench_with_input(BenchmarkId::new("rapidhash", len), data, |b, data| {
        b.iter(|| {
          let mut hasher = rapidhash::v3::RapidStreamHasherV3::new(&secrets);
          if chunk_size == usize::MAX {
            hasher.write(black_box(data));
          } else {
            for chunk in data.chunks(chunk_size) {
              hasher.write(black_box(chunk));
            }
          }
          black_box(hasher.finish())
        })
      });
    }
    group.finish();
  }
}

criterion_group!(
  benches,
  rapidhash_v3_64,
  rapidhash_seeded_state,
  rapidhash_hashmap_lookup,
  rapidhash_streaming
);
criterion_main!(benches);
