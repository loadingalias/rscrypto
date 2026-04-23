//! Blake3 comparison benchmarks: rscrypto vs official blake3 crate.

mod common;

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn oneshot(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("blake3");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::Blake3::digest(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("blake3", len), data, |b, d| {
      b.iter(|| black_box(blake3::hash(black_box(d))))
    });
  }

  g.finish();
}

fn keyed(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let key = [0x42u8; 32];
  let mut g = c.benchmark_group("blake3/keyed");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::Blake3::keyed_digest(black_box(&key), black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("blake3", len), data, |b, d| {
      b.iter(|| black_box(blake3::keyed_hash(black_box(&key), black_box(d))))
    });
  }

  g.finish();
}

fn derive_key(c: &mut Criterion) {
  const CONTEXT: &str = "rscrypto benchmark derive-key context";

  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("blake3/derive-key");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::Blake3::derive_key(black_box(CONTEXT), black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("blake3", len), data, |b, d| {
      b.iter(|| black_box(blake3::derive_key(black_box(CONTEXT), black_box(d))))
    });
  }

  g.finish();
}

fn streaming(c: &mut Criterion) {
  let data = common::random_bytes(1048576);
  let mut g = c.benchmark_group("blake3/streaming");
  g.throughput(criterion::Throughput::Bytes(data.len() as u64));

  for chunk_size in [64, 4096, 16384, 65536] {
    g.bench_function(format!("rscrypto/{chunk_size}B"), |b| {
      b.iter(|| {
        use rscrypto::Digest;
        let mut h = rscrypto::Blake3::new();
        for chunk in data.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        black_box(h.finalize())
      })
    });

    g.bench_function(format!("blake3/{chunk_size}B"), |b| {
      b.iter(|| {
        let mut h = blake3::Hasher::new();
        for chunk in data.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        black_box(h.finalize())
      })
    });
  }

  g.finish();
}

fn xof(c: &mut Criterion) {
  const OUT_LEN: usize = 64;

  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("blake3/xof");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        use rscrypto::Xof;

        let mut xof = rscrypto::Blake3::xof(black_box(d));
        let mut out = [0u8; OUT_LEN];
        xof.squeeze(&mut out);
        black_box(out)
      })
    });

    g.bench_with_input(BenchmarkId::new("blake3", len), data, |b, d| {
      b.iter(|| {
        let mut hasher = blake3::Hasher::new();
        hasher.update(black_box(d));
        let mut reader = hasher.finalize_xof();
        let mut out = [0u8; OUT_LEN];
        reader.fill(&mut out);
        black_box(out)
      })
    });
  }

  g.finish();
}

criterion_group!(benches, oneshot, keyed, derive_key, streaming, xof);
criterion_main!(benches);
