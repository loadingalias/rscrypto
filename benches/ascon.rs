//! Ascon benchmarks for rscrypto public APIs.

mod common;

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rscrypto::Xof as _;

fn ascon_hash256(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("ascon-hash256");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      use rscrypto::Digest as _;
      b.iter(|| black_box(rscrypto::AsconHash256::digest(black_box(d))))
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
  }

  g.finish();
}

fn ascon_hash256_many(c: &mut Criterion) {
  const MSG_LEN: usize = 4096;
  const COUNT: usize = 128;

  let slab = common::random_bytes(MSG_LEN * COUNT);
  let inputs: Vec<&[u8]> = slab.chunks(MSG_LEN).collect();
  let mut g = c.benchmark_group("ascon-hash256/many");
  g.throughput(criterion::Throughput::Bytes((MSG_LEN * COUNT) as u64));

  g.bench_function("rscrypto/batch-auto", |b| {
    let mut out = vec![[0u8; 32]; COUNT];
    b.iter(|| {
      rscrypto::AsconHash256::digest_many(black_box(&inputs), black_box(&mut out));
      black_box(out[0])
    })
  });

  g.bench_function("ascon-hash256/scalar-loop", |b| {
    let mut out = vec![[0u8; 32]; COUNT];
    b.iter(|| {
      for (input, slot) in inputs.iter().zip(out.iter_mut()) {
        use rscrypto::Digest as _;
        *slot = rscrypto::AsconHash256::digest(black_box(input));
      }
      black_box(out[0])
    })
  });

  g.finish();
}

fn ascon_xof128_many(c: &mut Criterion) {
  const MSG_LEN: usize = 4096;
  const OUT_LEN: usize = 64;
  const COUNT: usize = 128;

  let slab = common::random_bytes(MSG_LEN * COUNT);
  let inputs: Vec<&[u8]> = slab.chunks(MSG_LEN).collect();
  let mut g = c.benchmark_group("ascon-xof128/many");
  g.throughput(criterion::Throughput::Bytes((MSG_LEN * COUNT) as u64));

  g.bench_function("rscrypto/batch-auto", |b| {
    let mut out = vec![0u8; COUNT * OUT_LEN];
    b.iter(|| {
      rscrypto::AsconXof::hash_many_into(black_box(&inputs), OUT_LEN, black_box(&mut out));
      black_box(out[0])
    })
  });

  g.bench_function("ascon-xof128/scalar-loop", |b| {
    let mut out = vec![0u8; COUNT * OUT_LEN];
    b.iter(|| {
      for (index, input) in inputs.iter().enumerate() {
        let mut hasher = rscrypto::AsconXof::new();
        hasher.update(black_box(input));
        let mut reader = hasher.finalize_xof();
        let base = index * OUT_LEN;
        reader.squeeze(&mut out[base..base + OUT_LEN]);
      }
      black_box(out[0])
    })
  });

  g.finish();
}

criterion_group!(
  benches,
  ascon_hash256,
  ascon_hash256_streaming,
  ascon_hash256_many,
  ascon_xof128_many
);
criterion_main!(benches);
