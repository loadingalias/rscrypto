//! Software prefetch helpers for hash algorithm kernels.
//!
//! This module provides platform-tuned prefetch constants and inline helpers
//! for optimal memory access patterns in hash computation.
//!
//! # Background
//!
//! Modern CPUs have hardware prefetchers that work well for sequential access,
//! but software prefetch hints can still provide 5-15% gains in tight loops by:
//! - Reducing cache miss stalls when hardware prefetch falls behind
//! - Ensuring data arrives in L1 before the CPU needs it
//! - Working better with multi-block hash loops that process larger chunks
//!
//! # Prefetch Distance Tuning
//!
//! The optimal prefetch distance depends on:
//! - Memory latency (~70-100 cycles on modern x86, ~60-80 cycles on ARM)
//! - Loop iteration time (cycles per block processed)
//! - Cache line size (64 bytes on all modern platforms)
//!
//! For hash algorithms processing 64-128 byte blocks at high throughput:
//! - Time per block: ~50-200 cycles depending on algorithm
//! - With 80-cycle memory latency: 1-2 blocks ahead is optimal
//! - Practical value: 256-512 bytes ahead
//!
//! # Usage Pattern
//!
//! ```text
//! use crate::common::prefetch::{prefetch_read, HASH_PREFETCH_DISTANCE};
//!
//! // In a multi-block hash loop:
//! while ptr.add(BLOCK_SIZE) <= end {
//!     // Prefetch 2-4 blocks ahead
//!     prefetch_read(ptr.add(HASH_PREFETCH_DISTANCE));
//!
//!     // Process current block
//!     compress(state, ptr);
//!
//!     ptr = ptr.add(BLOCK_SIZE);
//! }
//! ```

// SAFETY: This module provides low-level prefetch intrinsics that require unsafe.
// Prefetch instructions are hints to the CPU and cannot cause memory unsafety;
// invalid addresses are silently ignored.
#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

// ─────────────────────────────────────────────────────────────────────────────
// Platform-Tuned Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Prefetch distance for hash algorithms on x86_64.
///
/// Tuned for multi-block compression loops processing 64-128 byte blocks.
/// Value: 512 bytes (4-8 blocks ahead).
///
/// # Rationale
/// - Hash compression functions: ~100-300 cycles per block
/// - Memory latency ~80-100 cycles on Zen4/Ice Lake
/// - 100 cycles / 150 cycles per block ≈ 0.7 blocks, but prefetching further ahead amortizes the
///   prefetch instruction overhead
/// - 512 bytes is 8 cache lines, covering multiple iterations
#[cfg(target_arch = "x86_64")]
pub const HASH_PREFETCH_DISTANCE: usize = 512;

/// Prefetch distance for BLAKE3/BLAKE2 chunk processing.
///
/// BLAKE3 processes 1024-byte chunks with 16 blocks of 64 bytes each.
/// Prefetch the next chunk while processing the current one.
#[cfg(target_arch = "x86_64")]
pub const CHUNK_PREFETCH_DISTANCE: usize = 1024;

/// Prefetch distance for SHA-256/SHA-512 block processing.
///
/// SHA processes 64/128-byte blocks with ~64-80 rounds each.
/// More conservative prefetch distance due to longer per-block time.
#[cfg(target_arch = "x86_64")]
pub const SHA_PREFETCH_DISTANCE: usize = 256;

/// Prefetch distance for hash algorithms on ARM64.
///
/// Tuned for Graviton2/3 and Apple Silicon.
/// Value: 384 bytes (~6 cache lines).
///
/// # Rationale
/// - Graviton2: ~60-70 cycle memory latency, excellent hardware prefetch
/// - Apple M1-M3: Best-in-class hardware prefetch, but software hints help
/// - ARM NEON hash kernels process 64-128B blocks
/// - 384 bytes is 6 cache lines, slightly more conservative than x86
#[cfg(target_arch = "aarch64")]
pub const HASH_PREFETCH_DISTANCE: usize = 384;

/// Prefetch distance for BLAKE3/BLAKE2 chunk processing on ARM64.
#[cfg(target_arch = "aarch64")]
pub const CHUNK_PREFETCH_DISTANCE: usize = 768;

/// Prefetch distance for SHA-256/SHA-512 block processing on ARM64.
#[cfg(target_arch = "aarch64")]
pub const SHA_PREFETCH_DISTANCE: usize = 192;

// ─────────────────────────────────────────────────────────────────────────────
// x86-64 Prefetch Intrinsics
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod x86_64_impl {
  use core::arch::x86_64::{_MM_HINT_NTA, _MM_HINT_T0, _MM_HINT_T1, _mm_prefetch};

  /// Prefetch data for read into L1 cache (temporal).
  ///
  /// Use when data will be accessed multiple times or soon after prefetch.
  /// This is the most common choice for hash compression loops.
  ///
  /// # Safety
  ///
  /// The pointer does not need to be valid or aligned. Prefetch is a hint;
  /// invalid addresses are silently ignored by the CPU.
  #[inline(always)]
  pub unsafe fn prefetch_read_l1(ptr: *const u8) {
    _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0);
  }

  /// Prefetch data for read into L2 cache (less temporal).
  ///
  /// Use when data will be accessed once but not immediately.
  /// Can be useful for prefetching further ahead without polluting L1.
  ///
  /// # Safety
  ///
  /// Same as [`prefetch_read_l1`].
  #[inline(always)]
  #[allow(dead_code)]
  pub unsafe fn prefetch_read_l2(ptr: *const u8) {
    _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T1);
  }

  /// Prefetch data for read, non-temporal (streaming).
  ///
  /// Use for data that will be accessed exactly once and should not
  /// pollute the cache hierarchy. Useful for very large inputs.
  ///
  /// # Safety
  ///
  /// Same as [`prefetch_read_l1`].
  #[inline(always)]
  #[allow(dead_code)]
  pub unsafe fn prefetch_read_nta(ptr: *const u8) {
    _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_NTA);
  }

  /// Prefetch for the next iteration of a hash compression loop.
  ///
  /// This is the recommended prefetch pattern for hash algorithms.
  /// Prefetches into L1 at `HASH_PREFETCH_DISTANCE` ahead of current pointer.
  ///
  /// # Safety
  ///
  /// Same as [`prefetch_read_l1`].
  #[inline(always)]
  pub unsafe fn prefetch_hash_block(ptr: *const u8) {
    // Use wrapping pointer arithmetic: prefetch addresses are allowed to be
    // out-of-bounds, but `ptr.add()` would be UB unless in-bounds.
    prefetch_read_l1(ptr.wrapping_add(super::HASH_PREFETCH_DISTANCE));
  }

  /// Prefetch for the next chunk in chunk-based hash algorithms (BLAKE3).
  ///
  /// # Safety
  ///
  /// Same as [`prefetch_read_l1`].
  #[inline(always)]
  #[allow(dead_code)]
  pub unsafe fn prefetch_next_chunk(ptr: *const u8) {
    prefetch_read_l1(ptr.wrapping_add(super::CHUNK_PREFETCH_DISTANCE));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// ARM64 Prefetch Intrinsics
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod aarch64_impl {
  // ARM64 prefetch using PRFM instruction via inline assembly.
  // Rust's core::arch::aarch64 doesn't expose prefetch intrinsics directly,
  // so we use inline assembly for the PRFM (prefetch memory) instruction.

  /// Prefetch data for read into L1 cache (PLDL1KEEP).
  ///
  /// Uses the ARM PRFM instruction with PLDL1KEEP hint:
  /// - PLD = Prefetch for Load
  /// - L1 = Target L1 cache
  /// - KEEP = Temporal (keep in cache)
  ///
  /// # Safety
  ///
  /// The pointer does not need to be valid or aligned. Prefetch is a hint;
  /// invalid addresses are silently ignored by the CPU.
  #[inline(always)]
  pub unsafe fn prefetch_read_l1(ptr: *const u8) {
    // PRFM PLDL1KEEP, [ptr]
    core::arch::asm!(
      "prfm pldl1keep, [{ptr}]",
      ptr = in(reg) ptr,
      options(nostack, preserves_flags)
    );
  }

  /// Prefetch data for read into L2 cache (PLDL2KEEP).
  ///
  /// # Safety
  ///
  /// Same as [`prefetch_read_l1`].
  #[inline(always)]
  #[allow(dead_code)]
  pub unsafe fn prefetch_read_l2(ptr: *const u8) {
    core::arch::asm!(
      "prfm pldl2keep, [{ptr}]",
      ptr = in(reg) ptr,
      options(nostack, preserves_flags)
    );
  }

  /// Prefetch data for read, streaming (PLDL1STRM).
  ///
  /// Use for data that will be accessed exactly once.
  ///
  /// # Safety
  ///
  /// Same as [`prefetch_read_l1`].
  #[inline(always)]
  #[allow(dead_code)]
  pub unsafe fn prefetch_read_stream(ptr: *const u8) {
    core::arch::asm!(
      "prfm pldl1strm, [{ptr}]",
      ptr = in(reg) ptr,
      options(nostack, preserves_flags)
    );
  }

  /// Prefetch for the next iteration of a hash compression loop.
  ///
  /// # Safety
  ///
  /// Same as [`prefetch_read_l1`].
  #[inline(always)]
  pub unsafe fn prefetch_hash_block(ptr: *const u8) {
    prefetch_read_l1(ptr.wrapping_add(super::HASH_PREFETCH_DISTANCE));
  }

  /// Prefetch for the next chunk in chunk-based hash algorithms (BLAKE3).
  ///
  /// # Safety
  ///
  /// Same as [`prefetch_read_l1`].
  #[inline(always)]
  #[allow(dead_code)]
  pub unsafe fn prefetch_next_chunk(ptr: *const u8) {
    prefetch_read_l1(ptr.wrapping_add(super::CHUNK_PREFETCH_DISTANCE));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
pub use aarch64_impl::{prefetch_hash_block, prefetch_next_chunk, prefetch_read_l1, prefetch_read_l2};
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use x86_64_impl::{prefetch_hash_block, prefetch_next_chunk, prefetch_read_l1, prefetch_read_l2};

// Fallback for other architectures (no-op)
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub const HASH_PREFETCH_DISTANCE: usize = 256;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub const CHUNK_PREFETCH_DISTANCE: usize = 512;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub const SHA_PREFETCH_DISTANCE: usize = 128;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
/// No-op prefetch fallback on architectures without explicit prefetch support.
///
/// # Safety
///
/// This function performs no memory access and is always safe to call.
pub unsafe fn prefetch_read_l1(_ptr: *const u8) {}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
/// No-op prefetch fallback on architectures without explicit prefetch support.
///
/// # Safety
///
/// This function performs no memory access and is always safe to call.
pub unsafe fn prefetch_read_l2(_ptr: *const u8) {}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
/// No-op prefetch fallback on architectures without explicit prefetch support.
///
/// # Safety
///
/// This function performs no memory access and is always safe to call.
pub unsafe fn prefetch_hash_block(_ptr: *const u8) {}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
/// No-op prefetch fallback on architectures without explicit prefetch support.
///
/// # Safety
///
/// This function performs no memory access and is always safe to call.
pub unsafe fn prefetch_next_chunk(_ptr: *const u8) {}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn prefetch_constants_are_cache_line_aligned() {
    // Prefetch distances should be multiples of cache line size (64 bytes)
    assert_eq!(HASH_PREFETCH_DISTANCE % 64, 0);
    assert_eq!(CHUNK_PREFETCH_DISTANCE % 64, 0);
    assert_eq!(SHA_PREFETCH_DISTANCE % 64, 0);
  }

  #[test]
  fn prefetch_constants_are_reasonable() {
    // Prefetch distances should be reasonable for hash algorithms
    // Use const blocks to satisfy clippy::assertions_on_constants
    const _: () = assert!(HASH_PREFETCH_DISTANCE >= 128);
    const _: () = assert!(HASH_PREFETCH_DISTANCE <= 1024);
    const _: () = assert!(CHUNK_PREFETCH_DISTANCE >= 256);
    const _: () = assert!(CHUNK_PREFETCH_DISTANCE <= 2048);
  }

  #[test]
  fn prefetch_does_not_crash_on_null() {
    // Prefetch should be safe to call with any pointer, including null.
    // The CPU silently ignores invalid prefetch addresses.
    // SAFETY: the prefetch intrinsics are explicitly documented as safe for any pointer value.
    unsafe {
      prefetch_read_l1(core::ptr::null());
      prefetch_read_l2(core::ptr::null());
      prefetch_hash_block(core::ptr::null());
      prefetch_next_chunk(core::ptr::null());
    }
  }

  #[test]
  fn prefetch_does_not_crash_on_unaligned() {
    let data = [0u8; 256];
    // SAFETY: the prefetch intrinsics are explicitly documented as safe for any pointer value.
    unsafe {
      // Test various unaligned addresses
      prefetch_read_l1(data.as_ptr().add(1));
      prefetch_read_l1(data.as_ptr().add(7));
      prefetch_read_l1(data.as_ptr().add(63));
      prefetch_hash_block(data.as_ptr().add(17));
    }
  }

  #[test]
  fn prefetch_does_not_crash_on_out_of_bounds() {
    // Prefetch beyond buffer bounds should be safe (CPU ignores invalid addresses)
    let data = [0u8; 64];
    // SAFETY: the prefetch intrinsics are explicitly documented as safe for any pointer value.
    unsafe {
      // This prefetches way beyond the buffer, but should be fine
      prefetch_hash_block(data.as_ptr());
      prefetch_next_chunk(data.as_ptr());
    }
  }
}
