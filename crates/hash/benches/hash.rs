//! Hash benchmarks
//!
//! Run: `cargo bench -p hash`
//! Native: `RUSTFLAGS='-C target-cpu=native' cargo bench -p hash`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn bench_blake3(c: &mut Criterion) {
  let mut group = c.benchmark_group("blake3");

  for size in [64, 256, 1024, 4096, 16384, 65536, 1048576] {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
      b.iter(|| {
        // TODO: hash::Blake3::digest(data)
        core::hint::black_box(data)
      });
    });
  }

  group.finish();
}

fn bench_sha256(c: &mut Criterion) {
  let mut group = c.benchmark_group("sha256");

  for size in [64, 256, 1024, 4096, 16384, 65536] {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
      b.iter(|| {
        // TODO: hash::Sha256::digest(data)
        core::hint::black_box(data)
      });
    });
  }

  group.finish();
}

criterion_group!(benches, bench_blake3, bench_sha256);
criterion_main!(benches);
