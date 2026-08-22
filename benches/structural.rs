//! Stable structural-cost benchmarks for representative public hash and checksum paths.

use core::hint::black_box;

use gungraun::{library_benchmark, library_benchmark_group};
use rscrypto::{Blake3, Checksum, Crc32, Sha256};

static INPUT_64: [u8; 64] = [0x3d; 64];
static INPUT_4K: [u8; 4096] = [0xa7; 4096];
static INPUT_16K: [u8; 16_384] = [0x5c; 16_384];

#[library_benchmark]
#[bench::bytes_64(&INPUT_64)]
#[bench::bytes_4096(&INPUT_4K)]
fn sha256(input: &[u8]) {
  black_box(Sha256::digest(black_box(input)));
}

#[library_benchmark]
#[bench::bytes_4096(&INPUT_4K)]
#[bench::bytes_16384(&INPUT_16K)]
fn blake3(input: &[u8]) {
  black_box(Blake3::digest(black_box(input)));
}

#[library_benchmark]
#[bench::bytes_64(&INPUT_64)]
#[bench::bytes_4096(&INPUT_4K)]
fn crc32(input: &[u8]) {
  black_box(Crc32::checksum(black_box(input)));
}

library_benchmark_group!(name = structural; benchmarks = sha256, blake3, crc32);
gungraun::main!(library_benchmark_groups = structural);
