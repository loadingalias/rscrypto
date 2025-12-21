use core::hint::black_box;

use checksum::{BufferedCrc64, BufferedCrc64Nvme, Checksum, Crc64, Crc64Nvme};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const SIZES: &[usize] = &[
  0,
  1,
  7,
  8,
  15,
  16,
  31,
  32,
  63,
  64,
  127,
  128,
  255,
  256,
  512,
  1024,
  1792, // 7-way multi-stream threshold (7 * 2 * 128B) for CRC64 folding
  2048,
  4096,
  16 * 1024,
  64 * 1024,
  1024 * 1024,
];

const STREAM_CHUNK_SIZES: &[usize] = &[1, 7, 16, 32, 64, 128, 256, 512, 1024];

fn make_data(len: usize) -> Vec<u8> {
  (0..len)
    .map(|i| (i as u8).wrapping_mul(31).wrapping_add((i >> 8) as u8))
    .collect()
}

fn bench_crc64_xz(c: &mut Criterion) {
  let cfg = Crc64::config();
  let mut group = c.benchmark_group(format!(
    "crc64/xz(force={},eff={},streams={})",
    cfg.requested_force.as_str(),
    cfg.effective_force.as_str(),
    cfg.tunables.streams
  ));

  for &size in SIZES {
    let data = make_data(size);
    if size > 0 {
      group.throughput(Throughput::Bytes(size as u64));
    }

    let kernel = Crc64::kernel_name_for_len(size);
    group.bench_with_input(BenchmarkId::new(format!("oneshot/{kernel}"), size), &data, |b, data| {
      b.iter(|| black_box(Crc64::checksum(black_box(data))));
    });

    for &chunk_size in STREAM_CHUNK_SIZES {
      let kernel = Crc64::kernel_name_for_len(chunk_size);
      group.bench_with_input(
        BenchmarkId::new(format!("stream/chunk-{chunk_size}/{kernel}"), size),
        &data,
        |b, data| {
          b.iter(|| {
            let mut hasher = Crc64::new();
            for chunk in data.chunks(chunk_size) {
              hasher.update(chunk);
            }
            black_box(hasher.finalize());
          });
        },
      );
    }

    for &chunk_size in STREAM_CHUNK_SIZES {
      let kernel = Crc64::kernel_name_for_len(chunk_size);
      group.bench_with_input(
        BenchmarkId::new(format!("buffered/chunk-{chunk_size}/{kernel}"), size),
        &data,
        |b, data| {
          b.iter(|| {
            let mut hasher = BufferedCrc64::new();
            for chunk in data.chunks(chunk_size) {
              hasher.update(chunk);
            }
            black_box(hasher.finalize());
          });
        },
      );
    }
  }

  group.finish();
}

fn bench_crc64_nvme(c: &mut Criterion) {
  let cfg = Crc64Nvme::config();
  let mut group = c.benchmark_group(format!(
    "crc64/nvme(force={},eff={},streams={})",
    cfg.requested_force.as_str(),
    cfg.effective_force.as_str(),
    cfg.tunables.streams
  ));

  for &size in SIZES {
    let data = make_data(size);
    if size > 0 {
      group.throughput(Throughput::Bytes(size as u64));
    }

    let kernel = Crc64Nvme::kernel_name_for_len(size);
    group.bench_with_input(BenchmarkId::new(format!("oneshot/{kernel}"), size), &data, |b, data| {
      b.iter(|| black_box(Crc64Nvme::checksum(black_box(data))));
    });

    for &chunk_size in STREAM_CHUNK_SIZES {
      let kernel = Crc64Nvme::kernel_name_for_len(chunk_size);
      group.bench_with_input(
        BenchmarkId::new(format!("stream/chunk-{chunk_size}/{kernel}"), size),
        &data,
        |b, data| {
          b.iter(|| {
            let mut hasher = Crc64Nvme::new();
            for chunk in data.chunks(chunk_size) {
              hasher.update(chunk);
            }
            black_box(hasher.finalize());
          });
        },
      );
    }

    for &chunk_size in STREAM_CHUNK_SIZES {
      let kernel = Crc64Nvme::kernel_name_for_len(chunk_size);
      group.bench_with_input(
        BenchmarkId::new(format!("buffered/chunk-{chunk_size}/{kernel}"), size),
        &data,
        |b, data| {
          b.iter(|| {
            let mut hasher = BufferedCrc64Nvme::new();
            for chunk in data.chunks(chunk_size) {
              hasher.update(chunk);
            }
            black_box(hasher.finalize());
          });
        },
      );
    }
  }

  group.finish();
}

criterion_group!(benches, bench_crc64_xz, bench_crc64_nvme);
criterion_main!(benches);
