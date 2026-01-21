//! Linux-only x86_64 assembly backends for BLAKE3.
//!
//! We keep these symbols namespaced to avoid collisions with the `blake3` crate
//! (used as a dev-dependency oracle in benches/tests).
//!
//! Notes:
//! - These `.s` sources are derived from the upstream BLAKE3 x86_64 assembly, with preprocessor
//!   conditionals removed and symbols renamed.
//! - The implementation is compiled via `global_asm!` (no external objects).

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::global_asm;

global_asm!(include_str!("asm/rscrypto_blake3_avx2_x86-64_unix_linux.s"));
global_asm!(include_str!("asm/rscrypto_blake3_avx512_x86-64_unix_linux.s"));

unsafe extern "C" {
  pub fn rscrypto_blake3_hash_many_avx2(
    inputs: *const *const u8,
    num_inputs: usize,
    blocks: usize,
    key: *const u32,
    counter: u64,
    increment_counter: bool,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: *mut u8,
  );

  pub fn rscrypto_blake3_hash_many_avx512(
    inputs: *const *const u8,
    num_inputs: usize,
    blocks: usize,
    key: *const u32,
    counter: u64,
    increment_counter: bool,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: *mut u8,
  );
}
