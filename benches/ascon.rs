//! Ascon comparison benchmarks: rscrypto vs ascon-hash256 crate.

mod common;

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn ascon_hash256(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("ascon-hash256");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      use rscrypto::Digest as _;
      b.iter(|| black_box(rscrypto::AsconHash256::digest(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("ascon-hash256", len), data, |b, d| {
      b.iter(|| {
        use ascon_hash256::digest::Digest as _;
        black_box(ascon_hash256::AsconHash256::digest(black_box(d)))
      })
    });
  }

  g.finish();
}

fn ascon_hash256_streaming(c: &mut Criterion) {
  let data = common::random_bytes(1048576);
  let mut g = c.benchmark_group("ascon-hash256/streaming");
  g.throughput(criterion::Throughput::Bytes(data.len() as u64));

  for chunk_size in [64, 4096] {
    g.bench_function(format!("rscrypto/{chunk_size}B"), |b| {
      b.iter(|| {
        use rscrypto::Digest;
        let mut h = rscrypto::AsconHash256::new();
        for chunk in data.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        black_box(h.finalize())
      })
    });

    g.bench_function(format!("ascon-hash256/{chunk_size}B"), |b| {
      b.iter(|| {
        use ascon_hash256::digest::Digest;
        let mut h = ascon_hash256::AsconHash256::new();
        for chunk in data.chunks(chunk_size) {
          ascon_hash256::digest::Update::update(&mut h, black_box(chunk));
        }
        black_box(h.finalize())
      })
    });
  }

  g.finish();
}

criterion_group!(benches, ascon_hash256, ascon_hash256_streaming);
criterion_main!(benches);
