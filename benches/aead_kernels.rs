//! Diagnostic AEAD kernel benchmarks.
//!
//! These benches are intentionally outside the production comparison bench so
//! global result tables do not treat kernel-only timings as user-facing AEADs.

mod common;

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

#[cfg(target_arch = "aarch64")]
const KEY_32: [u8; 32] = [0x42u8; 32];
const POLY_KEY: [u8; 32] = [
  0x7b, 0xac, 0x2b, 0x25, 0x2d, 0xb4, 0x47, 0xaf, 0x09, 0xb6, 0x7a, 0x55, 0xa4, 0xe9, 0x55, 0x84, 0x0a, 0xe1, 0xd6,
  0x73, 0x10, 0x75, 0xd9, 0xeb, 0x2a, 0x93, 0x75, 0x78, 0x3e, 0xd5, 0x53, 0xff,
];
#[cfg(target_arch = "aarch64")]
const NONCE_12: [u8; 12] = [0x07u8; 12];
const AAD: &[u8] = b"rscrypto-bench";

#[cfg(target_arch = "aarch64")]
fn chacha20_xor_kernel(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("aead-kernel/chacha20-xor");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    let mut buf = data.clone();
    g.bench_with_input(BenchmarkId::new("aarch64-neon", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        rscrypto::aead::diag_chacha20_xor_keystream_aarch64_neon(
          black_box(&KEY_32),
          black_box(1),
          black_box(&NONCE_12),
          black_box(&mut buf),
        );
        black_box(buf.as_ptr())
      })
    });

    #[cfg(all(feature = "diag", any(target_os = "linux", target_os = "macos")))]
    g.bench_with_input(BenchmarkId::new("aarch64-owned-asm-8block", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        rscrypto::aead::diag_chacha20_xor_keystream_aarch64_owned_asm(
          black_box(&KEY_32),
          black_box(1),
          black_box(&NONCE_12),
          black_box(&mut buf),
        );
        black_box(buf.as_ptr())
      })
    });
  }

  g.finish();
}

#[cfg(not(target_arch = "aarch64"))]
fn chacha20_xor_kernel(_: &mut Criterion) {}

fn poly1305_auth_kernel(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("aead-kernel/poly1305-auth");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("dispatched", len), data, |b, d| {
      b.iter(|| {
        black_box(
          rscrypto::aead::diag_chacha20poly1305_authenticate_aead(black_box(AAD), black_box(d), black_box(&POLY_KEY))
            .unwrap(),
        )
      })
    });

    #[cfg(target_arch = "aarch64")]
    g.bench_with_input(BenchmarkId::new("aarch64-neon-par4", len), data, |b, d| {
      b.iter(|| {
        black_box(
          rscrypto::aead::diag_chacha20poly1305_authenticate_aead_aarch64_neon_par4(
            black_box(AAD),
            black_box(d),
            black_box(&POLY_KEY),
          )
          .unwrap(),
        )
      })
    });

    #[cfg(all(target_arch = "aarch64", any(target_os = "linux", target_os = "macos")))]
    g.bench_with_input(BenchmarkId::new("aarch64-owned-par4-asm", len), data, |b, d| {
      b.iter(|| {
        black_box(
          rscrypto::aead::diag_chacha20poly1305_authenticate_aead_aarch64_owned_par4(
            black_box(AAD),
            black_box(d),
            black_box(&POLY_KEY),
          )
          .unwrap(),
        )
      })
    });
  }

  g.finish();
}

criterion_group!(aead_kernels, chacha20_xor_kernel, poly1305_auth_kernel);
criterion_main!(aead_kernels);
