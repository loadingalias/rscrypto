use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hashes::crypto::{Blake3, Sha3_256, Sha3_512, Sha256};
use traits::Digest as _;

fn inputs() -> Vec<(usize, Vec<u8>)> {
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

fn comp(c: &mut Criterion) {
  let inputs = inputs();
  let mut group = c.benchmark_group("hashes/comp");

  for (len, data) in &inputs {
    group.throughput(Throughput::Bytes(*len as u64));

    group.bench_with_input(BenchmarkId::new("sha256/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Sha256::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha256/sha2", len), data, |b, d| {
      b.iter(|| {
        use sha2::Digest as _;
        let out = sha2::Sha256::digest(black_box(d));
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("sha3_256/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Sha3_256::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha3_256/sha3", len), data, |b, d| {
      b.iter(|| {
        use sha3::Digest as _;
        let out = sha3::Sha3_256::digest(black_box(d));
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("sha3_512/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Sha3_512::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("sha3_512/sha3", len), data, |b, d| {
      b.iter(|| {
        use sha3::Digest as _;
        let out = sha3::Sha3_512::digest(black_box(d));
        black_box(out)
      })
    });

    group.bench_with_input(BenchmarkId::new("blake3/rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Blake3::digest(black_box(d))))
    });
    group.bench_with_input(BenchmarkId::new("blake3/official", len), data, |b, d| {
      b.iter(|| black_box(blake3::hash(black_box(d))))
    });
  }

  group.finish();
}

criterion_group!(benches, comp);
criterion_main!(benches);
