//! aarch64 assembly backends for BLAKE3 oneshot/chunk hashing.
//!
//! These symbols are namespaced to avoid collisions with the `blake3` crate
//! (used as a dev-dependency oracle in benches/tests).

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::global_asm;

#[cfg(target_os = "linux")]
global_asm!(include_str!("asm/rscrypto_blake3_hash1_chunk_aarch64_unix_linux.s"));

#[cfg(target_os = "macos")]
global_asm!(include_str!("asm/rscrypto_blake3_hash1_chunk_aarch64_apple_darwin.s"));

#[cfg(target_os = "linux")]
unsafe extern "C" {
  pub fn rscrypto_blake3_hash1_chunk_root_aarch64_unix_linux(
    input: *const u8,
    key: *const u32,
    flags: u32,
    out: *mut u8,
  );
  pub fn rscrypto_blake3_hash1_chunk_cv_aarch64_unix_linux(
    input: *const u8,
    key: *const u32,
    counter: u64,
    flags: u32,
    out: *mut u8,
  );

  pub fn rscrypto_blake3_hash1_chunk_state_aarch64_unix_linux(
    input: *const u8,
    key: *const u32,
    counter: u64,
    flags: u32,
    out_cv: *mut u32,
    out_last_block: *mut u8,
  );
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
  pub fn rscrypto_blake3_hash1_chunk_root_aarch64_apple_darwin(
    input: *const u8,
    key: *const u32,
    flags: u32,
    out: *mut u8,
  );
  pub fn rscrypto_blake3_hash1_chunk_cv_aarch64_apple_darwin(
    input: *const u8,
    key: *const u32,
    counter: u64,
    flags: u32,
    out: *mut u8,
  );

  pub fn rscrypto_blake3_hash1_chunk_state_aarch64_apple_darwin(
    input: *const u8,
    key: *const u32,
    counter: u64,
    flags: u32,
    out_cv: *mut u32,
    out_last_block: *mut u8,
  );
}
