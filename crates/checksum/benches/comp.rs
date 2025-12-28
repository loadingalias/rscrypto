use core::hint::black_box;
use std::sync::Once;

use checksum::{Checksum, Crc32, Crc32C, Crc64, Crc64Nvme};
use crc_fast::{CrcAlgorithm as CrcFastAlgorithm, Digest as CrcFastDigest};
use crc32c::crc32c as crc32c_oneshot;
use crc32fast::Hasher as Crc32FastHasher;
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
      let crc64 = Crc64::kernel_name_for_len(size);
      let crc64_nvme = Crc64Nvme::kernel_name_for_len(size);
      let crc32 = Crc32::kernel_name_for_len(size);
      let crc32c = Crc32C::kernel_name_for_len(size);
      eprintln!(
        "║   {:>3} ({:>7} B): crc64/xz={crc64}  crc64/nvme={crc64_nvme}  crc32={crc32}  crc32c={crc32c}",
        label, size
      );
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

fn bench_crc32_ieee_comp(c: &mut Criterion) {
  print_platform_info();
  let base_rs = Crc32::new();
  let base_fast = CrcFastDigest::new(CrcFastAlgorithm::Crc32IsoHdlc);
  let base_crc32fast = Crc32FastHasher::new();

  let mut group = c.benchmark_group("crc32/ieee/compare");
  for &(label, size) in CASES {
    let data = make_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    let kernel = Crc32::kernel_name_for_len(size);
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

    group.bench_with_input(BenchmarkId::new("crc-fast/auto", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_fast;
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("crc32fast/auto", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_crc32fast.clone();
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });
  }
  group.finish();
}

fn bench_crc32c_castagnoli_comp(c: &mut Criterion) {
  print_platform_info();
  let base_rs = Crc32C::new();
  let base_fast = CrcFastDigest::new(CrcFastAlgorithm::Crc32Iscsi);

  let mut group = c.benchmark_group("crc32c/castagnoli/compare");
  for &(label, size) in CASES {
    let data = make_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    let kernel = Crc32C::kernel_name_for_len(size);
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

    group.bench_with_input(BenchmarkId::new("crc-fast/auto", label), &data, |b, data| {
      b.iter(|| {
        let mut hasher = base_fast;
        hasher.update(black_box(data));
        black_box(hasher.finalize());
      });
    });

    group.bench_with_input(BenchmarkId::new("crc32c/oneshot", label), &data, |b, data| {
      b.iter(|| black_box(crc32c_oneshot(black_box(data))));
    });
  }
  group.finish();
}

criterion_group!(
  benches,
  bench_crc64_xz_comp,
  bench_crc64_nvme_comp,
  bench_crc32_ieee_comp,
  bench_crc32c_castagnoli_comp
);
criterion_main!(benches);
