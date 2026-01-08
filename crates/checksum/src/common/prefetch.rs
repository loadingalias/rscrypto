//! Software prefetch helpers for SIMD CRC kernels.

// SAFETY: This module provides low-level prefetch intrinsics that require unsafe.
// Prefetch instructions are hints to the CPU and cannot cause memory unsafety;
// invalid addresses are silently ignored.
#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]
//! This module provides platform-tuned prefetch constants and inline helpers
//! for optimal memory access patterns in large-buffer CRC computation.
//!
//! # Background
//!
//! Modern CPUs have hardware prefetchers that work well for sequential access,
//! but software prefetch hints can still provide 5-15% gains in tight loops by:
//! - Reducing cache miss stalls when hardware prefetch falls behind
//! - Ensuring data arrives in L1 before the CPU needs it
//! - Working better with double-unrolled loops that process larger chunks
//!
//! # Prefetch Distance Tuning
//!
//! The optimal prefetch distance depends on:
//! - Memory latency (~70-100 cycles on modern x86, ~60-80 cycles on ARM)
//! - Loop iteration time (cycles per block processed)
//! - Cache line size (64 bytes on all modern platforms)
//!
//! Formula: `prefetch_distance = (memory_latency / cycles_per_block) * block_size`
//!
//! For a kernel processing 256B blocks at ~80 GiB/s on a 4GHz CPU:
//! - Time per block: 256B / 80GiB/s ≈ 3ns ≈ 12 cycles
//! - With 80-cycle memory latency: 80/12 * 256B ≈ 1.7KB
//! - Practical value: 512B-1KB (2-4 blocks ahead)
//!
//! # Usage Pattern
//!
//! ```ignore
//! use crate::common::prefetch::{prefetch_read, LARGE_BLOCK_DISTANCE};
//!
//! // In a double-unrolled loop processing 512B per iteration:
//! while ptr.add(DOUBLE_BLOCK) <= end {
//!     // Prefetch 2 iterations ahead (1KB for 512B blocks)
//!     prefetch_read(ptr.add(LARGE_BLOCK_DISTANCE));
//!
//!     // Process first 256B block
//!     // ... fold operations ...
//!
//!     // Process second 256B block
//!     // ... fold operations ...
//!
//!     ptr = ptr.add(DOUBLE_BLOCK);
//! }
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// Platform-Tuned Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Prefetch distance for large buffer kernels (xl size, 1MB+).
///
/// Tuned for double-unrolled loops processing 512B per iteration.
/// Value: 1024 bytes (2 iterations ahead).
///
/// # Rationale
/// - At 80 GiB/s, 512B takes ~6ns ≈ 24 cycles at 4GHz
/// - Memory latency ~80-100 cycles on Zen4/Ice Lake
/// - 100 cycles / 24 cycles ≈ 4 blocks, but 2 blocks (1KB) is practical sweet spot
/// - Prefetching too far ahead wastes L1 cache space
#[cfg(target_arch = "x86_64")]
pub const LARGE_BLOCK_DISTANCE: usize = 1024;

/// Prefetch distance for medium buffer kernels (m/l size, 4KB-1MB).
///
/// Tuned for single or double-unrolled loops processing 256-512B per iteration.
/// Value: 512 bytes (1-2 iterations ahead).
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub const MEDIUM_BLOCK_DISTANCE: usize = 512;

/// Prefetch distance for large buffer kernels on ARM64.
///
/// Tuned for Graviton2/3 and Apple Silicon.
/// Value: 768 bytes (~2-3 iterations ahead for 256B blocks).
///
/// # Rationale
/// - Graviton2: ~60-70 cycle memory latency, narrower memory bus than x86
/// - Apple M1-M3: Excellent hardware prefetch, but software hints still help
/// - ARM NEON processes 128B blocks, so 768B = 6 blocks ahead
#[cfg(target_arch = "aarch64")]
pub const LARGE_BLOCK_DISTANCE: usize = 768;

/// Prefetch distance for medium buffer kernels on ARM64.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
pub const MEDIUM_BLOCK_DISTANCE: usize = 384;

// ─────────────────────────────────────────────────────────────────────────────
// x86-64 Prefetch Intrinsics
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod x86_64_impl {
  use core::arch::x86_64::{_MM_HINT_NTA, _MM_HINT_T0, _MM_HINT_T1, _mm_prefetch};

  /// Prefetch data for read into L1 cache (temporal).
  ///
  /// Use when data will be accessed multiple times or soon after prefetch.
  /// This is the most common choice for CRC folding loops.
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
  /// pollute the cache hierarchy. Rarely useful for CRC computation.
  ///
  /// # Safety
  ///
  /// Same as [`prefetch_read_l1`].
  #[inline(always)]
  #[allow(dead_code)]
  pub unsafe fn prefetch_read_nta(ptr: *const u8) {
    _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_NTA);
  }

  /// Prefetch for the next iteration of a double-unrolled loop.
  ///
  /// This is the recommended prefetch pattern for large buffer kernels.
  /// Prefetches into L1 at `LARGE_BLOCK_DISTANCE` ahead of current pointer.
  ///
  /// # Safety
  ///
  /// Same as [`prefetch_read_l1`].
  #[inline(always)]
  #[allow(dead_code)]
  pub unsafe fn prefetch_for_next_iteration(ptr: *const u8) {
    // Use wrapping pointer arithmetic: prefetch addresses are allowed to be
    // out-of-bounds, but `ptr.add()` would be UB unless in-bounds.
    prefetch_read_l1(ptr.wrapping_add(super::LARGE_BLOCK_DISTANCE));
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
    // Encoding: PLDL1KEEP = 0b00000 (type=0, target=0, policy=0)
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

  /// Prefetch for the next iteration of a double-unrolled loop.
  ///
  /// # Safety
  ///
  /// Same as [`prefetch_read_l1`].
  #[inline(always)]
  #[allow(dead_code)]
  pub unsafe fn prefetch_for_next_iteration(ptr: *const u8) {
    // Use wrapping pointer arithmetic: prefetch addresses are allowed to be
    // out-of-bounds, but `ptr.add()` would be UB unless in-bounds.
    prefetch_read_l1(ptr.wrapping_add(super::LARGE_BLOCK_DISTANCE));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
pub use aarch64_impl::{prefetch_for_next_iteration, prefetch_read_l1, prefetch_read_l2};
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use x86_64_impl::{prefetch_for_next_iteration, prefetch_read_l1, prefetch_read_l2};

// Fallback for other architectures (no-op)
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub const LARGE_BLOCK_DISTANCE: usize = 512;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub const MEDIUM_BLOCK_DISTANCE: usize = 256;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
pub unsafe fn prefetch_read_l1(_ptr: *const u8) {}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
pub unsafe fn prefetch_read_l2(_ptr: *const u8) {}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
pub unsafe fn prefetch_for_next_iteration(_ptr: *const u8) {}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn prefetch_constants_are_cache_line_aligned() {
    // Prefetch distances should be multiples of cache line size (64 bytes)
    assert_eq!(LARGE_BLOCK_DISTANCE % 64, 0);
    assert_eq!(MEDIUM_BLOCK_DISTANCE % 64, 0);
  }

  #[test]
  fn prefetch_does_not_crash_on_null() {
    // Prefetch should be safe to call with any pointer, including null.
    // The CPU silently ignores invalid prefetch addresses.
    unsafe {
      prefetch_read_l1(core::ptr::null());
      prefetch_read_l2(core::ptr::null());
      prefetch_for_next_iteration(core::ptr::null());
    }
  }

  #[test]
  fn prefetch_does_not_crash_on_unaligned() {
    let data = [0u8; 256];
    unsafe {
      // Test various unaligned addresses
      prefetch_read_l1(data.as_ptr().add(1));
      prefetch_read_l1(data.as_ptr().add(7));
      prefetch_read_l1(data.as_ptr().add(63));
    }
  }
}
