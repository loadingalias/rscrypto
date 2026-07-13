//! RapidHash comparison benchmarks: rscrypto vs rapidhash crate.
//!
//! Four groups:
//! - `rapidhash-64`: `RapidHashFast64` vs `rapidhash::fast::RapidHasher` — recommended
//!   HashMap-grade fast-path on both sides (V3 core, no avalanche).
//! - `rapidhash-128`: `RapidHashFast128` vs two `rapidhash::fast::RapidHasher` runs composed lo/hi
//!   (competitor has no single 128-bit API for the fast path).
//! - `rapidhash-v3-64`: `RapidHash64` vs `rapidhash::v3::rapidhash_v3_seeded` — C++ bit-compatible
//!   standard variant.
//! - `rapidhash-v3-128`: `RapidHash128` vs composed v3 for the competitor.

mod common;

use core::{
  hash::{BuildHasher, Hasher},
  hint::black_box,
};
use std::collections::HashMap;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rscrypto::FastHash;

const V3_HI_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

fn rapidhash_fast_64(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("rapidhash-64");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::RapidHashFast64::hash(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("rapidhash", len), data, |b, d| {
      b.iter(|| {
        let mut h = rapidhash::fast::RapidHasher::default();
        h.write(black_box(d));
        black_box(h.finish())
      })
    });
  }

  g.finish();
}

fn rapidhash_fast_128(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("rapidhash-128");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::RapidHashFast128::hash(black_box(d))))
    });

    // Competitor has no single 128-bit API on the fast path; compose lo/hi via
    // seed reflection (matches our `RapidHashFast128::hash` construction).
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
  }

  g.finish();
}

fn rapidhash_v3_64(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("rapidhash-v3-64");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(<rscrypto::RapidHash as FastHash>::hash(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("rapidhash", len), data, |b, d| {
      let secrets = rapidhash::v3::RapidSecrets::seed_cpp(0);
      b.iter(|| black_box(rapidhash::v3::rapidhash_v3_seeded(black_box(d), &secrets)))
    });
  }

  g.finish();
}

fn rapidhash_v3_128(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("rapidhash-v3-128");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(<rscrypto::RapidHash128 as FastHash>::hash(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("rapidhash", len), data, |b, d| {
      let lo_secrets = rapidhash::v3::RapidSecrets::seed_cpp(0);
      let hi_secrets = rapidhash::v3::RapidSecrets::seed_cpp(V3_HI_SEED);
      b.iter(|| {
        let lo = rapidhash::v3::rapidhash_v3_seeded(black_box(d), &lo_secrets) as u128;
        let hi = rapidhash::v3::rapidhash_v3_seeded(black_box(d), &hi_secrets) as u128;
        black_box(lo | (hi << 64))
      })
    });
  }

  g.finish();
}

fn rapidhash_build_hasher(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("rapidhash-buildhasher");
  let ours = rscrypto::RapidBuildHasher::new();
  let upstream = rapidhash::fast::SeedableState::fixed();

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(ours.hash_one(black_box(d.as_slice()))))
    });
    g.bench_with_input(BenchmarkId::new("rapidhash", len), data, |b, d| {
      b.iter(|| black_box(upstream.hash_one(black_box(d.as_slice()))))
    });
  }
  g.finish();
}

fn rapidhash_hashmap_lookup(c: &mut Criterion) {
  let key = common::random_bytes(32);
  let mut ours = HashMap::with_capacity_and_hasher(1, rscrypto::RapidBuildHasher::new());
  let mut upstream = HashMap::with_capacity_and_hasher(1, rapidhash::fast::SeedableState::fixed());
  ours.insert(key.as_slice(), 1u8);
  upstream.insert(key.as_slice(), 1u8);

  let mut g = c.benchmark_group("rapidhash-hashmap/lookup-32");
  g.bench_function("rscrypto", |b| {
    b.iter(|| black_box(ours.get(black_box(key.as_slice()))))
  });
  g.bench_function("rapidhash", |b| {
    b.iter(|| black_box(upstream.get(black_box(key.as_slice()))))
  });
  g.finish();
}

fn rapidhash_streaming(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let secrets = rapidhash::v3::RapidSecrets::seed_cpp(0);

  for (group_name, chunk_size) in [
    ("rapidhash-stream/one-write", usize::MAX),
    ("rapidhash-stream/chunk-64", 64),
  ] {
    let mut g = c.benchmark_group(group_name);
    for (len, data) in &inputs {
      common::set_throughput(&mut g, *len);

      g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
        b.iter(|| {
          let mut hasher = rscrypto::RapidStreamHasher::new();
          if chunk_size == usize::MAX {
            hasher.write(black_box(d));
          } else {
            for chunk in d.chunks(chunk_size) {
              hasher.write(black_box(chunk));
            }
          }
          black_box(hasher.finish())
        })
      });

      g.bench_with_input(BenchmarkId::new("rapidhash", len), data, |b, d| {
        b.iter(|| {
          let mut hasher = rapidhash::v3::RapidStreamHasherV3::new(&secrets);
          if chunk_size == usize::MAX {
            hasher.write(black_box(d));
          } else {
            for chunk in d.chunks(chunk_size) {
              hasher.write(black_box(chunk));
            }
          }
          black_box(hasher.finish())
        })
      });
    }
    g.finish();
  }
}

criterion_group!(
  benches,
  rapidhash_fast_64,
  rapidhash_fast_128,
  rapidhash_v3_64,
  rapidhash_v3_128,
  rapidhash_build_hasher,
  rapidhash_hashmap_lookup,
  rapidhash_streaming
);
criterion_main!(benches);
