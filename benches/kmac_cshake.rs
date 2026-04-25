//! KMAC256 / cSHAKE256 comparison benchmarks: rscrypto vs tiny-keccak.
//!
//! tiny-keccak is the canonical reference implementation for the SP 800-185
//! derived functions (cSHAKE, KMAC). We compare one-shot throughput across
//! the standard size matrix with a 32-byte output.

mod common;

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

const KMAC_KEY: &[u8] = b"rscrypto-kmac-bench-key-32bytes";
const CUSTOMIZATION: &[u8] = b"rscrypto-bench";
const FUNCTION_NAME: &[u8] = b"";
const OUTPUT_LEN: usize = 32;

fn kmac256(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("kmac256");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        let out: [u8; OUTPUT_LEN] =
          rscrypto::Kmac256::mac_array(black_box(KMAC_KEY), black_box(CUSTOMIZATION), black_box(d));
        black_box(out)
      })
    });

    g.bench_with_input(BenchmarkId::new("tiny-keccak", len), data, |b, d| {
      b.iter(|| {
        use tiny_keccak::Hasher as _;

        let mut out = [0u8; OUTPUT_LEN];
        let mut kmac = tiny_keccak::Kmac::v256(black_box(KMAC_KEY), black_box(CUSTOMIZATION));
        kmac.update(black_box(d));
        kmac.finalize(&mut out);
        black_box(out)
      })
    });
  }

  g.finish();
}

fn cshake256(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("cshake256");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        let mut out = [0u8; OUTPUT_LEN];
        rscrypto::Cshake256::hash_into(
          black_box(FUNCTION_NAME),
          black_box(CUSTOMIZATION),
          black_box(d),
          &mut out,
        );
        black_box(out)
      })
    });

    g.bench_with_input(BenchmarkId::new("tiny-keccak", len), data, |b, d| {
      b.iter(|| {
        use tiny_keccak::Hasher as _;

        let mut out = [0u8; OUTPUT_LEN];
        let mut cshake = tiny_keccak::CShake::v256(black_box(FUNCTION_NAME), black_box(CUSTOMIZATION));
        cshake.update(black_box(d));
        cshake.finalize(&mut out);
        black_box(out)
      })
    });
  }

  g.finish();
}

criterion_group!(benches, kmac256, cshake256);
criterion_main!(benches);
