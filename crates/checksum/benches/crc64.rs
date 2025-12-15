//! CRC64 benchmarks (rscrypto implementations only).
//!
//! Run: `cargo bench -p checksum -- crc64`
//! Native: `RUSTFLAGS='-C target-cpu=native' cargo bench -p checksum -- crc64`
//!
//! This benchmarks:
//! - CRC64/XZ (ECMA polynomial) - used by XZ compression, storage systems
//! - CRC64/NVMe - used by NVMe storage protocol

use checksum::{Crc64, Crc64Nvme};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

/// Standard benchmark sizes.
const SIZES: [usize; 7] = [64, 256, 1024, 4096, 16384, 65536, 1048576];

/// Benchmark the CRC64/XZ dispatch path.
fn bench_xz(c: &mut Criterion) {
  let mut group = c.benchmark_group("crc64/xz");
  eprintln!("crc64/xz backend: {}", checksum::crc64::selected_backend());

  for size in SIZES {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
      b.iter(|| core::hint::black_box(Crc64::checksum(data)));
    });
  }

  group.finish();
}

/// Benchmark the CRC64/NVMe dispatch path.
fn bench_nvme(c: &mut Criterion) {
  let mut group = c.benchmark_group("crc64/nvme");
  eprintln!("crc64/nvme backend: {}", checksum::crc64::selected_backend_nvme());

  for size in SIZES {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
      b.iter(|| core::hint::black_box(Crc64Nvme::checksum(data)));
    });
  }

  group.finish();
}

criterion_group!(benches, bench_xz, bench_nvme,);
criterion_main!(benches);
