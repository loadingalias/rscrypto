//! CRC16 benchmarks (rscrypto implementations only).
//!
//! Run: `cargo bench -p checksum -- crc16`
//!
//! This benchmarks:
//! - CRC16/IBM (polynomial 0x8005) - Modbus, USB, legacy protocols
//! - CRC16/CCITT-FALSE (polynomial 0x1021) - Bluetooth, SD cards

use checksum::{Crc16CcittFalse, Crc16Ibm};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

/// Standard benchmark sizes.
const SIZES: [usize; 7] = [64, 256, 1024, 4096, 16384, 65536, 1048576];

/// Benchmark the CRC16/IBM dispatch path.
fn bench_ibm(c: &mut Criterion) {
  let mut group = c.benchmark_group("crc16/ibm");
  eprintln!("crc16/ibm backend: {}", checksum::crc16::ibm::selected_backend());

  for size in SIZES {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
      b.iter(|| core::hint::black_box(Crc16Ibm::checksum(data)));
    });
  }

  group.finish();
}

/// Benchmark the CRC16/CCITT-FALSE dispatch path.
fn bench_ccitt_false(c: &mut Criterion) {
  let mut group = c.benchmark_group("crc16/ccitt-false");
  eprintln!(
    "crc16/ccitt-false backend: {}",
    checksum::crc16::ccitt_false::selected_backend()
  );

  for size in SIZES {
    let data = vec![0u8; size];
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
      b.iter(|| core::hint::black_box(Crc16CcittFalse::checksum(data)));
    });
  }

  group.finish();
}

criterion_group!(benches, bench_ibm, bench_ccitt_false,);
criterion_main!(benches);
