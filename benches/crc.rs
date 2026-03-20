//! CRC comparison benchmarks: rscrypto vs competitor crates.

mod common;

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rscrypto::Checksum;

fn crc32_ieee(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("crc32");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::Crc32::checksum(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("crc32fast", len), data, |b, d| {
      b.iter(|| black_box(crc32fast::hash(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("crc-fast", len), data, |b, d| {
      b.iter(|| black_box(crc_fast::crc32_iso_hdlc(black_box(d))))
    });
  }

  g.finish();
}

fn crc32c(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("crc32c");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::Crc32C::checksum(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("crc32c", len), data, |b, d| {
      b.iter(|| black_box(crc32c::crc32c(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("crc-fast", len), data, |b, d| {
      b.iter(|| black_box(crc_fast::crc32_iscsi(black_box(d))))
    });
  }

  g.finish();
}

fn crc64_xz(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("crc64-xz");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::Crc64::checksum(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("crc64fast", len), data, |b, d| {
      b.iter(|| {
        let mut h = crc64fast::Digest::new();
        h.write(black_box(d));
        black_box(h.sum64())
      })
    });
  }

  g.finish();
}

fn crc64_nvme(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("crc64-nvme");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::Crc64Nvme::checksum(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("crc-fast", len), data, |b, d| {
      b.iter(|| black_box(crc_fast::crc64_nvme(black_box(d))))
    });
  }

  g.finish();
}

fn crc16(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let crc_algo = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC);
  let mut g = c.benchmark_group("crc16-ccitt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::Crc16Ccitt::checksum(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("crc", len), data, |b, d| {
      b.iter(|| black_box(crc_algo.checksum(black_box(d))))
    });
  }

  g.finish();
}

fn crc24(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let crc_algo = crc::Crc::<u32>::new(&crc::CRC_24_OPENPGP);
  let mut g = c.benchmark_group("crc24-openpgp");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::Crc24OpenPgp::checksum(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("crc", len), data, |b, d| {
      b.iter(|| black_box(crc_algo.checksum(black_box(d))))
    });
  }

  g.finish();
}

criterion_group!(benches, crc32_ieee, crc32c, crc64_xz, crc64_nvme, crc16, crc24);
criterion_main!(benches);
