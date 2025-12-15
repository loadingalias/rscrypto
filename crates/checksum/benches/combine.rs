//! CRC combine operation benchmarks.
//!
//! Run: `cargo bench -p checksum -- combine`
//!
//! The combine operation computes `crc(A || B)` from `crc(A)`, `crc(B)`, `len(B)`
//! in O(log n) time. This is useful for parallel CRC computation.

use checksum::Crc32c;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

/// Benchmark the CRC32-C combine operation.
///
/// The combine operation is O(log n) in the length, making it efficient
/// for merging independently-computed CRC values.
fn bench_combine_crc32c(c: &mut Criterion) {
  let mut group = c.benchmark_group("combine/crc32c");

  // The combine operation is O(log n) in the length
  for len in [64, 256, 1024, 4096, 16384, 65536] {
    // Throughput isn't really meaningful for combine since it's O(log n),
    // but we include it for consistency
    group.throughput(Throughput::Elements(1));

    group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, &len| {
      let crc_a = 0x1234_5678u32;
      let crc_b = 0x8765_4321u32;
      b.iter(|| core::hint::black_box(Crc32c::combine(crc_a, crc_b, len)));
    });
  }

  group.finish();
}

criterion_group!(benches, bench_combine_crc32c,);
criterion_main!(benches);
