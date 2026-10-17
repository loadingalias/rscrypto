//! Diagnostic AEAD kernel benchmarks.
//!
//! These benches are intentionally outside the production comparison bench so
//! global result tables do not treat kernel-only timings as user-facing AEADs.
//! Before timing, every row must match RustCrypto's ChaCha20 keystream or
//! ChaCha20-Poly1305 tag for its input.

#[cfg(all(rscrypto_internal, feature = "diag"))]
#[path = "common/criterion.rs"]
#[cfg(all(rscrypto_internal, feature = "diag"))]
mod bench_config;

#[cfg(all(rscrypto_internal, feature = "diag"))]
mod common;

#[cfg(all(rscrypto_internal, feature = "diag"))]
use core::hint::black_box;

#[cfg(all(rscrypto_internal, feature = "diag"))]
use bench_config::checked;
#[cfg(all(rscrypto_internal, feature = "diag"))]
use criterion::{BenchmarkId, Criterion};

#[cfg(target_arch = "aarch64")]
#[cfg(all(rscrypto_internal, feature = "diag"))]
const KEY_32: [u8; 32] = [0x42u8; 32];
#[cfg(all(rscrypto_internal, feature = "diag"))]
const POLY_KEY: [u8; 32] = [
  0x7b, 0xac, 0x2b, 0x25, 0x2d, 0xb4, 0x47, 0xaf, 0x09, 0xb6, 0x7a, 0x55, 0xa4, 0xe9, 0x55, 0x84, 0x0a, 0xe1, 0xd6,
  0x73, 0x10, 0x75, 0xd9, 0xeb, 0x2a, 0x93, 0x75, 0x78, 0x3e, 0xd5, 0x53, 0xff,
];
/// RFC 8439 §2.8.2 key and nonce, whose Poly1305 one-time key is [`POLY_KEY`].
#[cfg(all(rscrypto_internal, feature = "diag"))]
const RFC8439_KEY: [u8; 32] = [
  0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f, 0x90, 0x91, 0x92,
  0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f,
];
#[cfg(all(rscrypto_internal, feature = "diag"))]
const RFC8439_NONCE: [u8; 12] = [0x07, 0x00, 0x00, 0x00, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47];
#[cfg(target_arch = "aarch64")]
#[cfg(all(rscrypto_internal, feature = "diag"))]
const NONCE_12: [u8; 12] = [0x07u8; 12];
#[cfg(all(rscrypto_internal, feature = "diag"))]
const AAD: &[u8] = b"rscrypto-bench";

/// RustCrypto's ChaCha20 keystream for `KEY_32` and `NONCE_12`, from block 1, XORed into `data`.
#[cfg(target_arch = "aarch64")]
#[cfg(all(rscrypto_internal, feature = "diag"))]
fn reference_xor(data: &[u8]) -> Vec<u8> {
  use chacha20::cipher::{KeyIvInit as _, StreamCipher as _, StreamCipherSeek as _};

  let mut output = data.to_vec();
  let mut cipher = chacha20::ChaCha20::new(&KEY_32.into(), &NONCE_12.into());
  cipher.seek(64u64);
  cipher.apply_keystream(&mut output);
  output
}

/// RustCrypto's ChaCha20-Poly1305 tag over `ciphertext` under the RFC 8439 §2.8.2 key and nonce.
#[cfg(all(rscrypto_internal, feature = "diag"))]
fn reference_tag(ciphertext: &[u8]) -> [u8; 16] {
  use chacha20::cipher::{KeyIvInit as _, StreamCipher as _, StreamCipherSeek as _};
  use chacha20poly1305::aead::{AeadInOut as _, KeyInit as _};

  // Sealing `ciphertext` XOR the keystream yields `ciphertext` and its tag.
  let mut buffer = ciphertext.to_vec();
  let mut keystream = chacha20::ChaCha20::new(&RFC8439_KEY.into(), &RFC8439_NONCE.into());
  keystream.seek(64u64);
  keystream.apply_keystream(&mut buffer);
  let tag = chacha20poly1305::ChaCha20Poly1305::new(&RFC8439_KEY.into())
    .encrypt_inout_detached(&RFC8439_NONCE.into(), AAD, buffer.as_mut_slice().into())
    .expect("RFC 8439 reference must seal");
  assert_eq!(buffer, ciphertext, "reference ciphertext");
  <[u8; 16]>::try_from(tag.as_slice()).expect("Poly1305 tags have 16 bytes")
}

#[cfg(target_arch = "aarch64")]
#[cfg(all(rscrypto_internal, feature = "diag"))]
fn chacha20_xor_kernel(c: &mut Criterion) {
  if !bench_config::selected("aead-kernel/chacha20-copy-and-xor") {
    return;
  }
  let inputs = common::comp_sizes(&[]);
  // Timed: restore the input and XOR the keystream into the reused buffer.
  // Untimed: fixture generation and buffer allocation/destruction.
  let mut g = c.benchmark_group("aead-kernel/chacha20-copy-and-xor");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = reference_xor(data);

    g.bench_function(
      BenchmarkId::new("aarch64-neon", len),
      bench_config::checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          // SAFETY: AArch64 NEON is part of this target's compile-time baseline. `comp_sizes` tops out at 1,048,576
          // bytes (16,384 blocks), which fits the `u32` counter range starting at 1.
          unsafe {
            rscrypto::aead::diag_chacha20_xor_keystream_aarch64_neon(
              black_box(&KEY_32),
              black_box(1),
              black_box(&NONCE_12),
              black_box(&mut *buf),
            );
          }
          black_box(buf.as_ptr())
        },
        |_, buf| assert_eq!(*buf, expected),
      ),
    );
  }

  g.finish();
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
fn poly1305_auth_kernel(c: &mut Criterion) {
  if !bench_config::selected("aead-kernel/poly1305-auth") {
    return;
  }
  let inputs = common::comp_sizes(&[]);
  // Timed: authenticate immutable AAD/message bytes and return/drop the tag.
  // Untimed: fixtures; there is no per-iteration message restoration.
  let mut g = c.benchmark_group("aead-kernel/poly1305-auth");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = reference_tag(data);

    g.bench_function(
      BenchmarkId::new("dispatched", len),
      checked(
        || {
          rscrypto::aead::diag_chacha20poly1305_authenticate_aead(black_box(AAD), black_box(data), black_box(&POLY_KEY))
        },
        |tag| assert_eq!(tag, expected),
      ),
    );

    #[cfg(target_arch = "aarch64")]
    g.bench_function(
      BenchmarkId::new("aarch64-neon-par4", len),
      checked(
        || {
          rscrypto::aead::diag_chacha20poly1305_authenticate_aead_aarch64_neon_par4(
            black_box(AAD),
            black_box(data),
            black_box(&POLY_KEY),
          )
        },
        |tag| assert_eq!(tag, expected),
      ),
    );
  }

  g.finish();
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
fn main() {
  bench_config::run(&[
    #[cfg(target_arch = "aarch64")]
    chacha20_xor_kernel,
    poly1305_auth_kernel,
  ]);
}

#[cfg(not(all(rscrypto_internal, feature = "diag")))]
fn main() -> Result<(), &'static str> {
  Err("use just bench --bench aead_kernels to enable internal kernel benchmarks")
}
