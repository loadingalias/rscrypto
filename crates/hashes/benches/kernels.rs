use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hashes::crypto::{Blake3, Sha3_256, Sha3_512, Sha256};
use traits::Digest as _;

fn inputs() -> Vec<(usize, Vec<u8>)> {
  // Sizes chosen to exercise small inputs, one block, and large throughput.
  let sizes = [0usize, 1, 3, 8, 16, 31, 32, 63, 64, 65, 1024, 16 * 1024, 1024 * 1024];
  sizes
    .into_iter()
    .map(|len| {
      let mut v = vec![0u8; len];
      for (i, b) in v.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(31).wrapping_add(7);
      }
      (len, v)
    })
    .collect()
}

fn oneshot(c: &mut Criterion) {
  let inputs = inputs();
  let mut group = c.benchmark_group("hashes/oneshot");

  for (len, data) in &inputs {
    group.throughput(Throughput::Bytes(*len as u64));

    group.bench_with_input(BenchmarkId::new("sha256", len), data, |b, d| {
      b.iter(|| black_box(Sha256::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha3_256", len), data, |b, d| {
      b.iter(|| black_box(Sha3_256::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha3_512", len), data, |b, d| {
      b.iter(|| black_box(Sha3_512::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("blake3", len), data, |b, d| {
      b.iter(|| black_box(Blake3::digest(black_box(d))))
    });
  }

  group.finish();
}

fn streaming(c: &mut Criterion) {
  let mut group = c.benchmark_group("hashes/streaming");
  let data = vec![0u8; 1024 * 1024];
  group.throughput(Throughput::Bytes(data.len() as u64));

  group.bench_function("sha256/64B-chunks", |b| {
    b.iter(|| {
      let mut h = Sha256::new();
      for chunk in black_box(&data).chunks(64) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("sha3_256/136B-chunks", |b| {
    b.iter(|| {
      let mut h = Sha3_256::new();
      for chunk in black_box(&data).chunks(136) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("blake3/1024B-chunks", |b| {
    b.iter(|| {
      let mut h = Blake3::new();
      for chunk in black_box(&data).chunks(1024) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.finish();
}

criterion_group!(benches, oneshot, streaming);
criterion_main!(benches);
