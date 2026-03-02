//! x86_64 assembly backends for BLAKE3.
//!
//! These symbols are namespaced to avoid collisions with the `blake3` crate
//! (used as a dev-dependency oracle in benches/tests).
//!
//! Notes:
//! - The `.s` sources are derived from upstream BLAKE3 x86_64 assembly, with preprocessor
//!   conditionals removed and symbols renamed.
//! - The implementation is compiled via `global_asm!` (no external objects).

#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::global_asm;

#[cfg(target_os = "linux")]
global_asm!(include_str!("asm/rscrypto_blake3_avx2_x86-64_unix_linux.s"));
#[cfg(target_os = "linux")]
global_asm!(include_str!("asm/rscrypto_blake3_avx512_x86-64_unix_linux.s"));

#[cfg(target_os = "macos")]
global_asm!(include_str!("asm/rscrypto_blake3_avx2_x86-64_apple_darwin.s"));
#[cfg(target_os = "macos")]
global_asm!(include_str!("asm/rscrypto_blake3_avx512_x86-64_apple_darwin.s"));

#[cfg(target_os = "windows")]
global_asm!(include_str!("asm/rscrypto_blake3_avx2_x86-64_windows_msvc.s"));
#[cfg(target_os = "windows")]
global_asm!(include_str!("asm/rscrypto_blake3_avx512_x86-64_windows_msvc.s"));

// On Windows, we intentionally call the upstream-derived assembly using the
// SysV64 calling convention, to avoid the (very expensive) Windows x64 ABI
// callee-saved XMM/YMM register requirements. This is safe because these
// entrypoints are internal to rscrypto and are only called from our own code.
#[cfg(target_os = "windows")]
unsafe extern "sysv64" {
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

  pub fn rscrypto_blake3_xof_many_avx512(
    cv: *const u32,
    block: *const u8,
    block_len: u8,
    counter: u64,
    flags: u8,
    out: *mut u8,
    outblocks: usize,
  );

  pub fn rscrypto_blake3_compress_in_place_avx512(
    cv: *mut u32,
    block: *const u8,
    counter: u64,
    block_len: u8,
    flags: u8,
  );

  pub fn rscrypto_blake3_compress_xof_avx512(
    cv: *const u32,
    block: *const u8,
    counter: u64,
    block_len: u8,
    flags: u8,
    out: *mut u8,
  );
}

#[cfg(not(target_os = "windows"))]
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

  pub fn rscrypto_blake3_xof_many_avx512(
    cv: *const u32,
    block: *const u8,
    block_len: u8,
    counter: u64,
    flags: u8,
    out: *mut u8,
    outblocks: usize,
  );

  pub fn rscrypto_blake3_compress_in_place_avx512(
    cv: *mut u32,
    block: *const u8,
    counter: u64,
    block_len: u8,
    flags: u8,
  );

  pub fn rscrypto_blake3_compress_xof_avx512(
    cv: *const u32,
    block: *const u8,
    counter: u64,
    block_len: u8,
    flags: u8,
    out: *mut u8,
  );
}

#[inline(always)]
pub(crate) unsafe fn hash_many_avx2(
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
) {
  // SAFETY: callsites validate CPU features and pointer contracts.
  unsafe {
    rscrypto_blake3_hash_many_avx2(
      inputs,
      num_inputs,
      blocks,
      key,
      counter,
      increment_counter,
      flags,
      flags_start,
      flags_end,
      out,
    )
  };
}

#[inline(always)]
pub(crate) unsafe fn hash_many_avx512(
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
) {
  // SAFETY: callsites validate CPU features and pointer contracts.
  unsafe {
    rscrypto_blake3_hash_many_avx512(
      inputs,
      num_inputs,
      blocks,
      key,
      counter,
      increment_counter,
      flags,
      flags_start,
      flags_end,
      out,
    )
  };
}

/// Single-block compress using AVX-512 assembly (CV-only output).
///
/// # Safety
/// Caller must ensure AVX-512 F+VL are available.
#[inline(always)]
pub(crate) unsafe fn compress_in_place_avx512(
  cv: &[u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  debug_assert!(block_len <= u8::MAX as u32);
  debug_assert!(flags <= u8::MAX as u32);
  let mut cv_out = *cv;
  // SAFETY: callsites validate CPU features and pointer contracts.
  unsafe {
    rscrypto_blake3_compress_in_place_avx512(cv_out.as_mut_ptr(), block, counter, block_len as u8, flags as u8);
  }
  cv_out
}

/// Single-block compress with full 64-byte output using AVX-512 assembly.
///
/// # Safety
/// Caller must ensure AVX-512 F+VL are available, out valid for 64 bytes.
#[inline(always)]
pub(crate) unsafe fn compress_xof_avx512(
  cv: &[u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
  out: *mut u8,
) {
  debug_assert!(block_len <= u8::MAX as u32);
  debug_assert!(flags <= u8::MAX as u32);
  // SAFETY: callsites validate CPU features and pointer contracts.
  unsafe {
    rscrypto_blake3_compress_xof_avx512(cv.as_ptr(), block, counter, block_len as u8, flags as u8, out);
  }
}
