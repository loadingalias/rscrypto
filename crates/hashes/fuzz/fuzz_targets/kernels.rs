//! Cross-kernel equivalence fuzzing.
//!
//! Verifies that ALL available kernels on the current platform produce identical
//! results for any input. This is meant to scale as SIMD/HW kernels are added.

#![no_main]

use hashes::__internal::kernel_test::{
  verify_ascon_p12_kernels, verify_blake2b_512_kernels, verify_blake2s_256_kernels, verify_blake3_kernels,
  verify_keccakf1600_kernels, verify_sha224_kernels, verify_sha256_kernels, verify_sha384_kernels, verify_sha512_224_kernels,
  verify_sha512_256_kernels, verify_sha512_kernels,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
  verify_sha224_kernels(data).expect("sha224 kernels should agree");
  verify_sha256_kernels(data).expect("sha256 kernels should agree");
  verify_sha384_kernels(data).expect("sha384 kernels should agree");
  verify_sha512_kernels(data).expect("sha512 kernels should agree");
  verify_sha512_224_kernels(data).expect("sha512-224 kernels should agree");
  verify_sha512_256_kernels(data).expect("sha512-256 kernels should agree");

  verify_blake2b_512_kernels(data).expect("blake2b-512 kernels should agree");
  verify_blake2s_256_kernels(data).expect("blake2s-256 kernels should agree");
  verify_blake3_kernels(data).expect("blake3 kernels should agree");

  verify_keccakf1600_kernels(data).expect("keccakf1600 kernels should agree");
  verify_ascon_p12_kernels(data).expect("ascon p12 kernels should agree");
});

