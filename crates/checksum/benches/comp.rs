use core::hint::black_box;
use std::sync::Once;

use checksum::{Checksum, Crc64, Crc64Nvme};
use crc_fast::{CrcAlgorithm as CrcFastAlgorithm, Digest as CrcFastDigest};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

/// Print platform detection info once at benchmark start.
fn print_platform_info() {
  static ONCE: Once = Once::new();
  ONCE.call_once(|| {
    let tune = platform::tune();
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║                   PLATFORM DETECTION INFO                    ║");
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!("║ Platform: {}", platform::describe());
    eprintln!("║ Tune Kind: {:?}", tune.kind());
    eprintln!("║ PCLMUL threshold: {} bytes", tune.pclmul_threshold);
    eprintln!("║ SIMD width: {} bits", tune.effective_simd_width);
    eprintln!("║ Fast wide ops: {}", tune.fast_wide_ops);
    eprintln!("║ Parallel streams: {}", tune.parallel_streams);
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!("║ Kernel selection by size:");
    for &(label, size) in CASES {
      eprintln!("║   {:>3} ({:>7} B): {}", label, size, Crc64::kernel_name_for_len(size));
    }
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
  });
}

const CASES: &[(&str, usize)] = &[
  ("xs", 64),
  ("s", 256),
  ("m", 4 * 1024),
  ("l", 64 * 1024),
  ("xl", 1024 * 1024),
];

fn make_data(len: usize) -> Vec<u8> {
  (0..len)
    .map(|i| (i as u8).wrapping_mul(31).wrapping_add((i >> 8) as u8))
    .collect()
}

fn bench_crc64_xz_comp(c: &mut Criterion) {
  print_platform_info();
  let base_rs = Crc64::new();
  let base_ref = crc64fast::Digest::new();
  let base_fast = CrcFastDigest::new(CrcFastAlgorithm::Crc64Xz);

  let mut group = c.benchmark_group("crc64/xz/compare");
  for &(label, size) in CASES {
    let data = make_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    let kernel = Crc64::kernel_name_for_len(size);
    group.bench_with_input(
      BenchmarkId::new(format!("rscrypto/{kernel}"), label),
      &data,
      |b, data| {
        b.iter(|| {
          let mut hasher = base_rs.clone();
          hasher.update(black_box(data));
          black_box(hasher.finalize());
        });
      },
    );

    group.bench_with_input(BenchmarkId::new("crc64fast/auto", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_ref.clone();
        hasher.write(black_box(data));
        black_box(hasher.sum64());
      });
    });

    group.bench_with_input(BenchmarkId::new("crc-fast/auto", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_fast;
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });
  }
  group.finish();
}

fn bench_crc64_nvme_comp(c: &mut Criterion) {
  let base_rs = Crc64Nvme::new();
  let base_ref = crc64fast_nvme::Digest::new();
  let base_fast = CrcFastDigest::new(CrcFastAlgorithm::Crc64Nvme);

  let mut group = c.benchmark_group("crc64/nvme/compare");
  for &(label, size) in CASES {
    let data = make_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    let kernel = Crc64Nvme::kernel_name_for_len(size);
    group.bench_with_input(
      BenchmarkId::new(format!("rscrypto/{kernel}"), label),
      &data,
      |b, data| {
        b.iter(|| {
          let mut hasher = base_rs.clone();
          hasher.update(black_box(data));
          black_box(hasher.finalize());
        });
      },
    );

    group.bench_with_input(BenchmarkId::new("crc64fast-nvme/auto", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_ref.clone();
        hasher.write(black_box(data));
        black_box(hasher.sum64());
      });
    });

    group.bench_with_input(BenchmarkId::new("crc-fast/auto", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_fast;
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });
  }
  group.finish();
}

criterion_group!(benches, bench_crc64_xz_comp, bench_crc64_nvme_comp);
criterion_main!(benches);
