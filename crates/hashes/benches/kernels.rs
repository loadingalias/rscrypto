use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hashes::crypto::{Blake3, Sha3_256, Sha3_512, Sha256};
use traits::Digest as _;

mod common;

fn oneshot(c: &mut Criterion) {
  let inputs = common::sized_inputs();
  let mut group = c.benchmark_group("hashes/oneshot");

  for (len, data) in &inputs {
    common::set_throughput(&mut group, *len);

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
  let data = common::pseudo_random_bytes(1024 * 1024, 0xA11C_E5ED_5EED_0001);
  let data = black_box(data);
  group.throughput(Throughput::Bytes(data.len() as u64));

  group.bench_function("sha256/64B-chunks", |b| {
    b.iter(|| {
      let mut h = Sha256::new();
      for chunk in data.chunks(64) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("sha256/4KiB-chunks", |b| {
    b.iter(|| {
      let mut h = Sha256::new();
      for chunk in data.chunks(4 * 1024) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("sha3_256/64B-chunks", |b| {
    b.iter(|| {
      let mut h = Sha3_256::new();
      for chunk in data.chunks(64) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("sha3_256/4KiB-chunks", |b| {
    b.iter(|| {
      let mut h = Sha3_256::new();
      for chunk in data.chunks(4 * 1024) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("sha3_512/64B-chunks", |b| {
    b.iter(|| {
      let mut h = Sha3_512::new();
      for chunk in data.chunks(64) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("sha3_512/4KiB-chunks", |b| {
    b.iter(|| {
      let mut h = Sha3_512::new();
      for chunk in data.chunks(4 * 1024) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("blake3/64B-chunks", |b| {
    b.iter(|| {
      let mut h = Blake3::new();
      for chunk in data.chunks(64) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.bench_function("blake3/4KiB-chunks", |b| {
    b.iter(|| {
      let mut h = Blake3::new();
      for chunk in data.chunks(4 * 1024) {
        h.update(chunk);
      }
      black_box(h.finalize())
    })
  });

  group.finish();
}

criterion_group!(benches, oneshot, streaming);
criterion_main!(benches);
