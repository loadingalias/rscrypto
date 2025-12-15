//! CRC32-C benchmarks (rscrypto implementations only).
//!
//! Run: `cargo bench -p checksum -- crc32c`
//! Native: `RUSTFLAGS='-C target-cpu=native' cargo bench -p checksum -- crc32c`
//!
//! This benchmarks:
//! - Main dispatch path (auto-selects best backend)
//! - Bitwise table-less implementation (for embedded/wasm)
//! - Hybrid Zen4/Zen5 kernels (when available on AMD hardware)

#![allow(unsafe_code)] // Required for direct kernel benchmarks

use checksum::Crc32c;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

/// Standard benchmark sizes.
const SIZES: [usize; 7] = [64, 256, 1024, 4096, 16384, 65536, 1048576];

/// Smaller sizes for bitwise (embedded/wasm focus).
const BITWISE_SIZES: [usize; 5] = [16, 64, 256, 1024, 4096];

/// Benchmark the main CRC32-C dispatch path.
///
/// This uses the automatically-selected best backend for the current platform.
fn bench_dispatch(c: &mut Criterion) {
  let mut group = c.benchmark_group("crc32c/dispatch");
  eprintln!("crc32c backend: {}", checksum::crc32c::selected_backend());

  for size in SIZES {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
      b.iter(|| core::hint::black_box(Crc32c::checksum(data)));
    });
  }

  group.finish();
}

/// Benchmark the table-less bitwise CRC32-C implementation.
///
/// Bitwise implementation is optimized for:
/// - Embedded systems with limited memory
/// - WebAssembly targets
/// - Situations where lookup tables are undesirable
fn bench_bitwise(c: &mut Criterion) {
  let mut group = c.benchmark_group("crc32c/bitwise");

  for size in BITWISE_SIZES {
    let data = vec![0xABu8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
      b.iter(|| {
        let crc = checksum::bitwise::crc32c::compute(0xFFFF_FFFF, data);
        core::hint::black_box(crc ^ 0xFFFF_FFFF)
      });
    });
  }

  group.finish();
}

/// Benchmark the hybrid Zen4 kernel directly (bypasses dispatch).
///
/// This benchmark only runs when the required CPU features are available.
/// On non-AMD or non-Zen4+ hardware, this benchmark will be skipped.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
fn bench_hybrid_zen4(c: &mut Criterion) {
  // Check if we have the required features at runtime
  let has_features = std::arch::is_x86_feature_detected!("sse4.2")
    && std::arch::is_x86_feature_detected!("avx512f")
    && std::arch::is_x86_feature_detected!("avx512vl")
    && std::arch::is_x86_feature_detected!("avx512bw")
    && std::arch::is_x86_feature_detected!("vpclmulqdq")
    && std::arch::is_x86_feature_detected!("pclmulqdq");

  if !has_features {
    eprintln!("Skipping hybrid_zen4 benchmark: required CPU features not available");
    return;
  }

  let mut group = c.benchmark_group("crc32c/hybrid_zen4");
  eprintln!("Running hybrid_zen4 benchmark (3-way crc32q + vpclmul)");

  // Hybrid requires larger buffers (min 512 bytes)
  for size in [512, 1024, 4096, 16384, 65536, 1048576] {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
      b.iter(|| {
        // SAFETY: We checked for required features above
        let crc = unsafe { checksum::__bench::hybrid::compute_hybrid_zen4_unchecked(0xFFFF_FFFF, data) };
        core::hint::black_box(crc ^ 0xFFFF_FFFF)
      });
    });
  }

  group.finish();
}

/// Benchmark the hybrid Zen5 kernel directly (bypasses dispatch).
///
/// This benchmark only runs when the required CPU features are available.
/// On non-AMD or non-Zen5 hardware, this benchmark will be skipped.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
fn bench_hybrid_zen5(c: &mut Criterion) {
  // Check if we have the required features at runtime
  let has_features = std::arch::is_x86_feature_detected!("sse4.2")
    && std::arch::is_x86_feature_detected!("avx512f")
    && std::arch::is_x86_feature_detected!("avx512vl")
    && std::arch::is_x86_feature_detected!("avx512bw")
    && std::arch::is_x86_feature_detected!("vpclmulqdq")
    && std::arch::is_x86_feature_detected!("pclmulqdq");

  if !has_features {
    eprintln!("Skipping hybrid_zen5 benchmark: required CPU features not available");
    return;
  }

  let mut group = c.benchmark_group("crc32c/hybrid_zen5");
  eprintln!("Running hybrid_zen5 benchmark (7-way crc32q + vpclmul)");

  // Hybrid requires larger buffers (min 512 bytes)
  for size in [512, 1024, 4096, 16384, 65536, 1048576] {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
      b.iter(|| {
        // SAFETY: We checked for required features above
        let crc = unsafe { checksum::__bench::hybrid::compute_hybrid_zen5_unchecked(0xFFFF_FFFF, data) };
        core::hint::black_box(crc ^ 0xFFFF_FFFF)
      });
    });
  }

  group.finish();
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
criterion_group!(
  benches,
  bench_dispatch,
  bench_bitwise,
  bench_hybrid_zen4,
  bench_hybrid_zen5,
);

#[cfg(not(all(feature = "std", target_arch = "x86_64")))]
criterion_group!(benches, bench_dispatch, bench_bitwise,);

criterion_main!(benches);
