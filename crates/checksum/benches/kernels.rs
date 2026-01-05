use core::{hint::black_box, time::Duration};
use std::collections::HashSet;

use checksum::bench;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

mod util;

/// Deduplicate kernel names (some kernel arrays have duplicates for index consistency).
fn dedup_kernels(names: Vec<&'static str>) -> Vec<&'static str> {
  let mut seen = HashSet::new();
  names.into_iter().filter(|&n| seen.insert(n)).collect()
}

fn bench_crc64_xz_kernels(c: &mut Criterion) {
  util::print_platform_info();
  let kernel_names = dedup_kernels(bench::available_crc64_kernels());

  let mut group = c.benchmark_group("kernels/crc64/xz");
  for &(label, size) in util::CASES {
    let data = util::make_alignment_variants(util::make_data(size));
    group.throughput(Throughput::Bytes(size as u64));

    for variant in &data {
      let data = variant.as_slice();
      let param = util::bench_param_label(label, variant.alignment());

      for &name in &kernel_names {
        if name == "reference" {
          continue;
        }
        let Some(kernel) = bench::get_crc64_xz_kernel(name) else {
          panic!("crc64/xz kernel should exist for name={name}");
        };
        let func = kernel.func;
        group.bench_with_input(BenchmarkId::new(kernel.name, &param), &data, |b, data| {
          b.iter(|| {
            let state = black_box(!0u64);
            black_box(func(state, black_box(data)))
          });
        });
      }
    }
  }
  group.finish();
}

fn bench_crc16_ccitt_kernels(c: &mut Criterion) {
  util::print_platform_info();
  let kernel_names = dedup_kernels(bench::available_crc16_kernels());

  let mut group = c.benchmark_group("kernels/crc16/ccitt");
  for &(label, size) in util::CASES {
    let data = util::make_alignment_variants(util::make_data(size));
    group.throughput(Throughput::Bytes(size as u64));

    for variant in &data {
      let data = variant.as_slice();
      let param = util::bench_param_label(label, variant.alignment());

      for &name in &kernel_names {
        if name == "reference" {
          continue;
        }
        let Some(kernel) = bench::get_crc16_ccitt_kernel(name) else {
          panic!("crc16/ccitt kernel should exist for name={name}");
        };
        let func = kernel.func;
        group.bench_with_input(BenchmarkId::new(kernel.name, &param), &data, |b, data| {
          b.iter(|| {
            let state = black_box(!0u16);
            black_box(func(state, black_box(data)))
          });
        });
      }
    }
  }
  group.finish();
}

fn bench_crc16_ibm_kernels(c: &mut Criterion) {
  util::print_platform_info();
  let kernel_names = dedup_kernels(bench::available_crc16_kernels());

  let mut group = c.benchmark_group("kernels/crc16/ibm");
  for &(label, size) in util::CASES {
    let data = util::make_alignment_variants(util::make_data(size));
    group.throughput(Throughput::Bytes(size as u64));

    for variant in &data {
      let data = variant.as_slice();
      let param = util::bench_param_label(label, variant.alignment());

      for &name in &kernel_names {
        if name == "reference" {
          continue;
        }
        let Some(kernel) = bench::get_crc16_ibm_kernel(name) else {
          panic!("crc16/ibm kernel should exist for name={name}");
        };
        let func = kernel.func;
        group.bench_with_input(BenchmarkId::new(kernel.name, &param), &data, |b, data| {
          b.iter(|| {
            let state = black_box(0u16);
            black_box(func(state, black_box(data)))
          });
        });
      }
    }
  }
  group.finish();
}

fn bench_crc24_openpgp_kernels(c: &mut Criterion) {
  util::print_platform_info();
  let kernel_names = dedup_kernels(bench::available_crc24_kernels());

  let mut group = c.benchmark_group("kernels/crc24/openpgp");
  for &(label, size) in util::CASES {
    let data = util::make_alignment_variants(util::make_data(size));
    group.throughput(Throughput::Bytes(size as u64));

    for variant in &data {
      let data = variant.as_slice();
      let param = util::bench_param_label(label, variant.alignment());

      for &name in &kernel_names {
        if name == "reference" {
          continue;
        }
        let Some(kernel) = bench::get_crc24_openpgp_kernel(name) else {
          panic!("crc24/openpgp kernel should exist for name={name}");
        };
        let func = kernel.func;
        group.bench_with_input(BenchmarkId::new(kernel.name, &param), &data, |b, data| {
          b.iter(|| {
            let state = black_box(0u32);
            black_box(func(state, black_box(data)))
          });
        });
      }
    }
  }
  group.finish();
}

fn bench_crc64_nvme_kernels(c: &mut Criterion) {
  util::print_platform_info();
  let kernel_names = dedup_kernels(bench::available_crc64_kernels());

  let mut group = c.benchmark_group("kernels/crc64/nvme");
  for &(label, size) in util::CASES {
    let data = util::make_alignment_variants(util::make_data(size));
    group.throughput(Throughput::Bytes(size as u64));

    for variant in &data {
      let data = variant.as_slice();
      let param = util::bench_param_label(label, variant.alignment());

      for &name in &kernel_names {
        if name == "reference" {
          continue;
        }
        let Some(kernel) = bench::get_crc64_nvme_kernel(name) else {
          panic!("crc64/nvme kernel should exist for name={name}");
        };
        let func = kernel.func;
        group.bench_with_input(BenchmarkId::new(kernel.name, &param), &data, |b, data| {
          b.iter(|| {
            let state = black_box(!0u64);
            black_box(func(state, black_box(data)))
          });
        });
      }
    }
  }
  group.finish();
}

fn bench_crc32_ieee_kernels(c: &mut Criterion) {
  util::print_platform_info();
  let kernel_names = dedup_kernels(bench::available_crc32_ieee_kernels());

  let mut group = c.benchmark_group("kernels/crc32/ieee");
  for &(label, size) in util::CASES {
    let data = util::make_alignment_variants(util::make_data(size));
    group.throughput(Throughput::Bytes(size as u64));

    for variant in &data {
      let data = variant.as_slice();
      let param = util::bench_param_label(label, variant.alignment());

      for &name in &kernel_names {
        if name == "reference" {
          continue;
        }
        let Some(kernel) = bench::get_crc32_ieee_kernel(name) else {
          panic!("crc32/ieee kernel should exist for name={name}");
        };
        let func = kernel.func;
        group.bench_with_input(BenchmarkId::new(kernel.name, &param), &data, |b, data| {
          b.iter(|| {
            let state = black_box(!0u32);
            black_box(func(state, black_box(data)))
          });
        });
      }
    }
  }
  group.finish();
}

fn bench_crc32c_kernels(c: &mut Criterion) {
  util::print_platform_info();
  let kernel_names = dedup_kernels(bench::available_crc32c_kernels());

  let mut group = c.benchmark_group("kernels/crc32c/castagnoli");
  for &(label, size) in util::CASES {
    let data = util::make_alignment_variants(util::make_data(size));
    group.throughput(Throughput::Bytes(size as u64));

    for variant in &data {
      let data = variant.as_slice();
      let param = util::bench_param_label(label, variant.alignment());

      for &name in &kernel_names {
        if name == "reference" {
          continue;
        }
        let Some(kernel) = bench::get_crc32c_kernel(name) else {
          panic!("crc32c kernel should exist for name={name}");
        };
        let func = kernel.func;
        group.bench_with_input(BenchmarkId::new(kernel.name, &param), &data, |b, data| {
          b.iter(|| {
            let state = black_box(!0u32);
            black_box(func(state, black_box(data)))
          });
        });
      }
    }
  }
  group.finish();
}

criterion_group! {
  name = benches;
  config = Criterion::default()
    .measurement_time(Duration::from_secs(3))
    .sample_size(50);
  targets =
    bench_crc16_ccitt_kernels,
    bench_crc16_ibm_kernels,
    bench_crc24_openpgp_kernels,
    bench_crc64_xz_kernels,
    bench_crc64_nvme_kernels,
    bench_crc32_ieee_kernels,
    bench_crc32c_kernels,
}
criterion_main!(benches);
