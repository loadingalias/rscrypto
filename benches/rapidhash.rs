//! RapidHash comparison benchmarks: rscrypto vs rapidhash crate.
//!
//! Primary comparison: our `RapidHashFast` (V3 core, no avalanche) vs the
//! competitor's `fast::RapidHasher` (inner module, no avalanche). Both are
//! the recommended fast-path for HashMap users in their respective crates.

mod common;

use core::{hash::Hasher, hint::black_box};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rscrypto::FastHash;

fn rapidhash_64(c: &mut Criterion) {
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

criterion_group!(benches, rapidhash_64);
criterion_main!(benches);
