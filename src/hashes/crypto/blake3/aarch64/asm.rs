//! aarch64 assembly backends for BLAKE3 oneshot/chunk hashing.
//!
//! These symbols are namespaced to avoid collisions with the `blake3` crate
//! (used as a dev-dependency oracle in benches/tests).
//!
//! Pointer contract:
//! - asm entrypoints assume 8-byte aligned inputs
//! - u32 outputs require 4-byte alignment
//! - last-block byte outputs require 8-byte alignment
//! - unaligned callers must use the NEON paths in `aarch64.rs`.

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::global_asm;

pub(crate) const ASM_ALIGN_INPUT: usize = 8;
pub(crate) const ASM_ALIGN_U32_OUT: usize = 4;
pub(crate) const ASM_ALIGN_LAST_BLOCK_OUT: usize = 8;

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

  pub fn rscrypto_blake3_chunk_compress_blocks_aarch64_unix_linux(
    blocks: *const u8,
    chaining_value: *mut u32,
    chunk_counter: u64,
    flags: u32,
    blocks_compressed: *mut u8,
    num_blocks: usize,
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

  pub fn rscrypto_blake3_chunk_compress_blocks_aarch64_apple_darwin(
    blocks: *const u8,
    chaining_value: *mut u32,
    chunk_counter: u64,
    flags: u32,
    blocks_compressed: *mut u8,
    num_blocks: usize,
  );
}
