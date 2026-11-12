//! XXH32 and XXH64 comparison benchmarks: rscrypto, xxhash-rust, and twox-hash.
//!
//! Every row hashes with seed zero, the LZ4 and Zstandard frame seed, and must
//! return xxhash-rust's one-shot hash of its input before timing. The
//! `chunk-64` rows write the input to each streaming hasher in 64-byte slices,
//! as a decoder writes decoded data, and include construction and the final
//! digest.

#[path = "common/criterion.rs"]
mod bench_config;

mod common;

use core::{hash::Hasher, hint::black_box};

use bench_config::checked;
use criterion::{BenchmarkId, Criterion};
use rscrypto::FastHash;

/// XXH32 hashes 16-byte stripes from 16 bytes of input; XXH64 32-byte stripes from 32.
const XXH32_BOUNDARIES: [usize; 2] = [15, 16];
const XXH64_BOUNDARIES: [usize; 2] = [31, 32];
const CHUNK: usize = 64;

fn xxh32(c: &mut Criterion) {
  if bench_config::selected("xxh32/") {
    let mut group = c.benchmark_group("xxh32");
    for (len, data) in &common::comp_sizes(&XXH32_BOUNDARIES) {
      common::set_throughput(&mut group, *len);
      let expected = xxhash_rust::xxh32::xxh32(data, 0);
      group.bench_function(
        BenchmarkId::new("rscrypto", len),
        checked(
          || rscrypto::Xxh32::hash(black_box(data)),
          |out| assert_eq!(out, expected),
        ),
      );
      group.bench_function(
        BenchmarkId::new("xxhash-rust", len),
        checked(
          || xxhash_rust::xxh32::xxh32(black_box(data), 0),
          |out| assert_eq!(out, expected),
        ),
      );
      group.bench_function(
        BenchmarkId::new("twox-hash", len),
        checked(
          || twox_hash::XxHash32::oneshot(0, black_box(data)),
          |out| assert_eq!(out, expected),
        ),
      );
    }
    group.finish();
  }

  if bench_config::selected("xxh32-stream/") {
    let mut group = c.benchmark_group("xxh32-stream/chunk-64");
    for (len, data) in &common::comp_sizes(&XXH32_BOUNDARIES) {
      common::set_throughput(&mut group, *len);
      let expected = xxhash_rust::xxh32::xxh32(data, 0);
      group.bench_function(
        BenchmarkId::new("rscrypto", len),
        checked(
          || {
            let mut hasher = rscrypto::Xxh32Hasher::new();
            for chunk in data.chunks(CHUNK) {
              hasher.write(black_box(chunk));
            }
            hasher.finish()
          },
          |out| assert_eq!(out, expected),
        ),
      );
      group.bench_function(
        BenchmarkId::new("xxhash-rust", len),
        checked(
          || {
            let mut hasher = xxhash_rust::xxh32::Xxh32::new(0);
            for chunk in data.chunks(CHUNK) {
              hasher.update(black_box(chunk));
            }
            hasher.digest()
          },
          |out| assert_eq!(out, expected),
        ),
      );
      group.bench_function(
        BenchmarkId::new("twox-hash", len),
        checked(
          || {
            let mut hasher = twox_hash::XxHash32::with_seed(0);
            for chunk in data.chunks(CHUNK) {
              hasher.write(black_box(chunk));
            }
            hasher.finish_32()
          },
          |out| assert_eq!(out, expected),
        ),
      );
    }
    group.finish();
  }
}

fn xxh64(c: &mut Criterion) {
  if bench_config::selected("xxh64/") {
    let mut group = c.benchmark_group("xxh64");
    for (len, data) in &common::comp_sizes(&XXH64_BOUNDARIES) {
      common::set_throughput(&mut group, *len);
      let expected = xxhash_rust::xxh64::xxh64(data, 0);
      group.bench_function(
        BenchmarkId::new("rscrypto", len),
        checked(
          || rscrypto::Xxh64::hash(black_box(data)),
          |out| assert_eq!(out, expected),
        ),
      );
      group.bench_function(
        BenchmarkId::new("xxhash-rust", len),
        checked(
          || xxhash_rust::xxh64::xxh64(black_box(data), 0),
          |out| assert_eq!(out, expected),
        ),
      );
      group.bench_function(
        BenchmarkId::new("twox-hash", len),
        checked(
          || twox_hash::XxHash64::oneshot(0, black_box(data)),
          |out| assert_eq!(out, expected),
        ),
      );
    }
    group.finish();
  }

  if bench_config::selected("xxh64-stream/") {
    let mut group = c.benchmark_group("xxh64-stream/chunk-64");
    for (len, data) in &common::comp_sizes(&XXH64_BOUNDARIES) {
      common::set_throughput(&mut group, *len);
      let expected = xxhash_rust::xxh64::xxh64(data, 0);
      group.bench_function(
        BenchmarkId::new("rscrypto", len),
        checked(
          || {
            let mut hasher = rscrypto::Xxh64Hasher::new();
            for chunk in data.chunks(CHUNK) {
              hasher.write(black_box(chunk));
            }
            hasher.finish()
          },
          |out| assert_eq!(out, expected),
        ),
      );
      group.bench_function(
        BenchmarkId::new("xxhash-rust", len),
        checked(
          || {
            let mut hasher = xxhash_rust::xxh64::Xxh64::new(0);
            for chunk in data.chunks(CHUNK) {
              hasher.update(black_box(chunk));
            }
            hasher.digest()
          },
          |out| assert_eq!(out, expected),
        ),
      );
      group.bench_function(
        BenchmarkId::new("twox-hash", len),
        checked(
          || {
            let mut hasher = twox_hash::XxHash64::with_seed(0);
            for chunk in data.chunks(CHUNK) {
              hasher.write(black_box(chunk));
            }
            hasher.finish()
          },
          |out| assert_eq!(out, expected),
        ),
      );
    }
    group.finish();
  }
}

fn main() {
  bench_config::run(&[xxh32, xxh64]);
}
