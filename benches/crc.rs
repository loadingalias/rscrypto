//! CRC comparison benchmarks: rscrypto vs competitor crates.
//!
//! Before timing, every row must return the reference crate's checksum for its input.

#[path = "common/criterion.rs"]
mod bench_config;

mod common;

use core::hint::black_box;

use bench_config::checked;
use criterion::{BenchmarkId, Criterion};
use rscrypto::Checksum;

fn crc32_ieee(c: &mut Criterion) {
  if !bench_config::selected("crc32") {
    return;
  }
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("crc32");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = crc32fast::hash(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Crc32::checksum(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("crc32fast", len),
      checked(|| crc32fast::hash(black_box(data)), |out| assert_eq!(out, expected)),
    );
    g.bench_function(
      BenchmarkId::new("crc-fast", len),
      checked(
        || crc_fast::crc32_iso_hdlc(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
  }

  g.finish();
}

fn crc32c(c: &mut Criterion) {
  if !bench_config::selected("crc32c") {
    return;
  }
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("crc32c");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = crc32c::crc32c(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Crc32C::checksum(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("crc32c", len),
      checked(|| crc32c::crc32c(black_box(data)), |out| assert_eq!(out, expected)),
    );
    g.bench_function(
      BenchmarkId::new("crc-fast", len),
      checked(
        || crc_fast::crc32_iscsi(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
  }

  g.finish();
}

fn crc64_xz(c: &mut Criterion) {
  if !bench_config::selected("crc64-xz") {
    return;
  }
  let inputs = common::comp_sizes();
  let crc64fast = |data: &[u8]| {
    let mut h = crc64fast::Digest::new();
    h.write(data);
    h.sum64()
  };
  let mut g = c.benchmark_group("crc64-xz");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = crc64fast(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Crc64::checksum(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("crc64fast", len),
      checked(|| crc64fast(black_box(data)), |out| assert_eq!(out, expected)),
    );
  }

  g.finish();
}

fn crc64_nvme(c: &mut Criterion) {
  if !bench_config::selected("crc64-nvme") {
    return;
  }
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("crc64-nvme");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = crc_fast::crc64_nvme(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Crc64Nvme::checksum(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("crc-fast", len),
      checked(
        || crc_fast::crc64_nvme(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
  }

  g.finish();
}

fn crc16(c: &mut Criterion) {
  if !bench_config::selected("crc16-ccitt") {
    return;
  }
  let inputs = common::comp_sizes();
  let crc_algo = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC);
  let mut g = c.benchmark_group("crc16-ccitt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = crc_algo.checksum(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Crc16Ccitt::checksum(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("crc", len),
      checked(|| crc_algo.checksum(black_box(data)), |out| assert_eq!(out, expected)),
    );
  }

  g.finish();
}

fn crc16_ibm(c: &mut Criterion) {
  if !bench_config::selected("crc16-ibm") {
    return;
  }
  let inputs = common::comp_sizes();
  let crc_algo = crc::Crc::<u16>::new(&crc::CRC_16_ARC);
  let mut g = c.benchmark_group("crc16-ibm");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = crc_algo.checksum(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Crc16Ibm::checksum(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("crc", len),
      checked(|| crc_algo.checksum(black_box(data)), |out| assert_eq!(out, expected)),
    );
  }

  g.finish();
}

fn crc24(c: &mut Criterion) {
  if !bench_config::selected("crc24-openpgp") {
    return;
  }
  let inputs = common::comp_sizes();
  let crc_algo = crc::Crc::<u32>::new(&crc::CRC_24_OPENPGP);
  let mut g = c.benchmark_group("crc24-openpgp");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = crc_algo.checksum(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Crc24OpenPgp::checksum(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
    g.bench_function(
      BenchmarkId::new("crc", len),
      checked(|| crc_algo.checksum(black_box(data)), |out| assert_eq!(out, expected)),
    );
  }

  g.finish();
}

fn main() {
  bench_config::run(&[crc32_ieee, crc32c, crc64_xz, crc64_nvme, crc16, crc16_ibm, crc24]);
}
