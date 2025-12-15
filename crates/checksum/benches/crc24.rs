//! CRC24 benchmarks (rscrypto implementations only).
//!
//! Run: `cargo bench -p checksum -- crc24`
//!
//! This benchmarks:
//! - CRC24/OpenPGP (polynomial 0x864CFB) - OpenPGP, IETF protocols

use checksum::Crc24;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

/// Standard benchmark sizes.
const SIZES: [usize; 7] = [64, 256, 1024, 4096, 16384, 65536, 1048576];

/// Benchmark the CRC24/OpenPGP dispatch path.
fn bench_openpgp(c: &mut Criterion) {
  let mut group = c.benchmark_group("crc24/openpgp");
  eprintln!("crc24/openpgp backend: {}", checksum::crc24::selected_backend());

  for size in SIZES {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
      b.iter(|| core::hint::black_box(Crc24::checksum(data)));
    });
  }

  group.finish();
}

criterion_group!(benches, bench_openpgp,);
criterion_main!(benches);
