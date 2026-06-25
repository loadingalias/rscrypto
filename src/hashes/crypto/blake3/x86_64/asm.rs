//! x86_64 assembly backends for BLAKE3.
//!
//! These symbols are namespaced to avoid collisions with the `blake3` crate
//! (used as a dev-dependency oracle in benches/tests).
//!
//! Notes:
//! - The remaining `.s` sources are derived from upstream BLAKE3 x86_64 AVX2/AVX-512 assembly, with
//!   preprocessor conditionals removed and symbols renamed.
//! - The implementation is compiled via `global_asm!` (no external objects).

#![allow(unsafe_code)]

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

}

/// AVX2 `hash_many` assembly entrypoint.
///
/// # Safety
///
/// The caller must ensure:
/// 1. AVX2 is available on the current CPU.
/// 2. `inputs` points to `num_inputs` readable input pointers.
/// 3. Every input pointer is readable for `blocks * 64` bytes.
/// 4. `key` points to 8 readable little-endian BLAKE3 key words.
/// 5. `out` is writable for `num_inputs * 32` bytes.
/// 6. The output range does not alias any input range or `key`.
/// 7. `num_inputs`, `blocks`, counters, and flags are public values.
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
  // SAFETY: AVX2 `hash_many` FFI call because:
  // 1. The caller upholds this wrapper's CPU-feature contract.
  // 2. The caller upholds the input pointer, key, output, and aliasing contracts.
  // 3. The assembly branches only on public sizes, counters, and flags.
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

/// AVX-512 `hash_many` assembly entrypoint.
///
/// # Safety
///
/// The caller must ensure:
/// 1. AVX-512F, AVX-512VL, AVX-512BW, AVX-512DQ, and AVX-512CD are available on the current CPU.
/// 2. `inputs` points to `num_inputs` readable input pointers.
/// 3. Every input pointer is readable for `blocks * 64` bytes.
/// 4. `key` points to 8 readable little-endian BLAKE3 key words.
/// 5. `out` is writable for `num_inputs * 32` bytes.
/// 6. The output range does not alias any input range or `key`.
/// 7. `num_inputs`, `blocks`, counters, and flags are public values.
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
  // SAFETY: AVX-512 `hash_many` FFI call because:
  // 1. The caller upholds this wrapper's CPU-feature contract.
  // 2. The caller upholds the input pointer, key, output, and aliasing contracts.
  // 3. The assembly branches only on public sizes, counters, and flags.
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

/// AVX-512 root-output/XOF assembly entrypoint.
///
/// # Safety
///
/// The caller must ensure:
/// 1. AVX-512F and AVX-512VL are available on the current CPU.
/// 2. `cv` points to 8 readable little-endian BLAKE3 chaining-value words.
/// 3. `block` points to one readable 64-byte BLAKE3 block.
/// 4. `block_len` and `flags` already fit the assembly ABI.
/// 5. `out` is writable for `outblocks * 64` bytes.
/// 6. The output range does not alias `cv` or `block`.
/// 7. `counter`, `block_len`, `flags`, and `outblocks` are public values.
#[inline(always)]
pub(crate) unsafe fn xof_many_avx512(
  cv: *const u32,
  block: *const u8,
  block_len: u8,
  counter: u64,
  flags: u8,
  out: *mut u8,
  outblocks: usize,
) {
  // SAFETY: AVX-512 XOF FFI call because:
  // 1. The caller upholds this wrapper's CPU-feature contract.
  // 2. The caller upholds the CV, block, output, and aliasing contracts.
  // 3. The assembly branches only on public counters, flags, and output block counts.
  unsafe {
    rscrypto_blake3_xof_many_avx512(cv, block, block_len, counter, flags, out, outblocks);
  }
}

/// Single-block compress using AVX-512 assembly (CV-only output).
///
/// # Safety
///
/// The caller must ensure:
/// 1. AVX-512F and AVX-512VL are available on the current CPU.
/// 2. `block` is readable for 64 bytes.
/// 3. `block_len <= 255` and `flags <= 255`.
/// 4. `block`, `counter`, `block_len`, and `flags` are public values.
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
  // SAFETY: AVX-512 single-block FFI call because:
  // 1. The caller upholds this wrapper's CPU-feature contract.
  // 2. `cv_out` is a local 8-word CV and is writable for the assembly call.
  // 3. The caller guarantees `block` is readable for one BLAKE3 block.
  // 4. Debug assertions document the ABI narrowing for `block_len` and `flags`.
  unsafe {
    rscrypto_blake3_compress_in_place_avx512(cv_out.as_mut_ptr(), block, counter, block_len as u8, flags as u8);
  }
  cv_out
}

/// Single-block compress using AVX-512 assembly, mutating the CV in place.
///
/// # Safety
///
/// The caller must ensure:
/// 1. AVX-512F and AVX-512VL are available on the current CPU.
/// 2. `cv` is the mutable chaining value for this call and does not alias `block`.
/// 3. `block` is readable for 64 bytes.
/// 4. `block_len <= 255` and `flags <= 255`.
/// 5. `block`, `counter`, `block_len`, and `flags` are public values.
#[inline(always)]
pub(crate) unsafe fn compress_in_place_avx512_mut(
  cv: &mut [u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
) {
  debug_assert!(block_len <= u8::MAX as u32);
  debug_assert!(flags <= u8::MAX as u32);
  // SAFETY: AVX-512 in-place single-block FFI call because:
  // 1. The caller upholds this wrapper's CPU-feature contract.
  // 2. `cv` is writable for 8 words and the mutable borrow prevents another Rust alias.
  // 3. The caller guarantees `block` is readable for one BLAKE3 block and does not alias `cv`.
  // 4. Debug assertions document the ABI narrowing for `block_len` and `flags`.
  unsafe {
    rscrypto_blake3_compress_in_place_avx512(cv.as_mut_ptr(), block, counter, block_len as u8, flags as u8);
  }
}
