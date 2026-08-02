//! Software prefetch helpers for SIMD CRC kernels.

// SAFETY: This module provides low-level prefetch intrinsics that require unsafe.
// Prefetch instructions are hints to the CPU and cannot cause memory unsafety;
// invalid addresses are silently ignored.
#![allow(unsafe_code)]
#![cfg_attr(
  all(
    target_arch = "aarch64",
    feature = "crc32",
    not(any(feature = "crc16", feature = "crc24", feature = "crc64", test))
  ),
  allow(dead_code, unused_imports)
)]
//! This module provides architecture-specific prefetch distances and inline
//! helpers for large-buffer CRC computation.
//!
//! # Background
//!
//! Software prefetch is only a hint. Whether it helps depends on the
//! microarchitecture, cache state, loop body, and input size. These distances
//! are manually maintained implementation choices, not portable performance
//! guarantees.
//!
//! # Usage Pattern
//!
//! ```text
//! use crate::checksum::common::prefetch::{prefetch_read_l1, LARGE_BLOCK_DISTANCE};
//!
//! // In a double-unrolled loop processing 512B per iteration:
//! while ptr.add(DOUBLE_BLOCK) <= end {
//!     // Prefetch 2 iterations ahead (1KB for 512B blocks)
//!     prefetch_read_l1(ptr.add(LARGE_BLOCK_DISTANCE));
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

// Platform-Tuned Constants

/// Prefetch distance for large buffer kernels (xl size, 1MB+).
///
/// The x86-64 folding loops use this 1,024-byte lookahead.
#[cfg(target_arch = "x86_64")]
pub const LARGE_BLOCK_DISTANCE: usize = 1024;

/// Prefetch distance for large buffer kernels on ARM64.
///
/// The AArch64 folding loops use this 768-byte lookahead.
#[cfg(target_arch = "aarch64")]
pub const LARGE_BLOCK_DISTANCE: usize = 768;

// x86-64 Prefetch Intrinsics

#[cfg(target_arch = "x86_64")]
mod x86_64_impl {
  use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

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
  pub(crate) unsafe fn prefetch_read_l1(ptr: *const u8) {
    // SAFETY: Prefetch is a CPU hint; invalid addresses are silently ignored.
    // The _mm_prefetch intrinsic cannot cause memory unsafety.
    unsafe {
      _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0);
    }
  }
}

// ARM64 Prefetch Intrinsics

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
  pub(crate) unsafe fn prefetch_read_l1(ptr: *const u8) {
    // SAFETY: Inline assembly for PRFM prefetch hint. Prefetch instructions
    // are CPU hints that cannot cause memory unsafety; invalid addresses
    // are silently ignored by the hardware.
    unsafe {
      // PRFM PLDL1KEEP, [ptr]
      // Encoding: PLDL1KEEP = 0b00000 (type=0, target=0, policy=0)
      core::arch::asm!(
        "prfm pldl1keep, [{ptr}]",
        ptr = in(reg) ptr,
        options(nostack, preserves_flags)
      );
    }
  }
}

// Public API

#[cfg(target_arch = "aarch64")]
pub(crate) use aarch64_impl::prefetch_read_l1;
#[cfg(target_arch = "x86_64")]
pub(crate) use x86_64_impl::prefetch_read_l1;

// Fallback for other architectures (no-op)
#[cfg(all(not(any(target_arch = "x86_64", target_arch = "aarch64")), test))]
pub const LARGE_BLOCK_DISTANCE: usize = 512;

#[cfg(all(not(any(target_arch = "x86_64", target_arch = "aarch64")), test))]
#[inline(always)]
/// No-op prefetch fallback used only by tests on unsupported architectures.
///
/// # Safety
///
/// This function performs no memory access and is always safe to call.
pub(crate) unsafe fn prefetch_read_l1(_ptr: *const u8) {}

// Tests

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn prefetch_distance_is_cache_line_aligned() {
    // Prefetch distance should be a multiple of cache line size (64 bytes)
    assert_eq!(LARGE_BLOCK_DISTANCE % 64, 0);
  }

  #[test]
  #[cfg_attr(miri, ignore)] // Miri does not support inline assembly (aarch64 PRFM)
  fn prefetch_does_not_crash_on_null() {
    // Prefetch should be safe to call with any pointer, including null.
    // The CPU silently ignores invalid prefetch addresses.
    // SAFETY: the prefetch intrinsics are explicitly documented as safe for any pointer value.
    unsafe {
      prefetch_read_l1(core::ptr::null());
    }
  }

  #[test]
  #[cfg_attr(miri, ignore)] // Miri does not support inline assembly (aarch64 PRFM)
  fn prefetch_does_not_crash_on_unaligned() {
    let data = [0u8; 256];
    // SAFETY: the prefetch intrinsics are explicitly documented as safe for any pointer value.
    unsafe {
      // Test various unaligned addresses
      prefetch_read_l1(data.as_ptr().add(1));
      prefetch_read_l1(data.as_ptr().add(7));
      prefetch_read_l1(data.as_ptr().add(63));
    }
  }
}
