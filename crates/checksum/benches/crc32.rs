//! CRC32 (ISO-HDLC) benchmarks (rscrypto implementations only).
//!
//! Run: `cargo bench -p checksum -- crc32`
//! Native: `RUSTFLAGS='-C target-cpu=native' cargo bench -p checksum -- crc32`
//!
//! This benchmarks:
//! - Main dispatch path (auto-selects best backend)
//! - Bitwise table-less implementation (for embedded/wasm)

use checksum::Crc32;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

/// Standard benchmark sizes.
const SIZES: [usize; 7] = [64, 256, 1024, 4096, 16384, 65536, 1048576];

/// Smaller sizes for bitwise (embedded/wasm focus).
const BITWISE_SIZES: [usize; 5] = [16, 64, 256, 1024, 4096];

/// Benchmark the main CRC32 dispatch path.
///
/// This uses the automatically-selected best backend for the current platform.
fn bench_dispatch(c: &mut Criterion) {
  let mut group = c.benchmark_group("crc32/dispatch");
  eprintln!("crc32 backend: {}", checksum::crc32::selected_backend());

  for size in SIZES {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
      b.iter(|| core::hint::black_box(Crc32::checksum(data)));
    });
  }

  group.finish();
}

/// Benchmark the table-less bitwise CRC32 implementation.
///
/// Bitwise implementation is optimized for:
/// - Embedded systems with limited memory
/// - WebAssembly targets
/// - Situations where lookup tables are undesirable
fn bench_bitwise(c: &mut Criterion) {
  let mut group = c.benchmark_group("crc32/bitwise");

  for size in BITWISE_SIZES {
    let data = vec![0xABu8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
      b.iter(|| {
        let crc = checksum::bitwise::crc32::compute(0xFFFF_FFFF, data);
        core::hint::black_box(crc ^ 0xFFFF_FFFF)
      });
    });
  }

  group.finish();
}

criterion_group!(benches, bench_dispatch, bench_bitwise,);
criterion_main!(benches);
