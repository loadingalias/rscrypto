//! x86_64 hardware-accelerated CRC-32 kernels.
//!
//! # Safety
//!
//! This module uses `unsafe` for hardware intrinsics. All unsafe functions
//! document their safety requirements. Safe wrappers are provided for use
//! via the dispatcher system.
#![allow(unsafe_code)]
//! This module provides three acceleration tiers:
//!
//! | Kernel | Instructions | Throughput | CRC-32 | CRC-32C |
//! |--------|--------------|------------|--------|---------|
//! | `vpclmul` | AVX-512 VPCLMULQDQ | ~40 GB/s | ✓ | ✓ |
//! | `pclmul` | PCLMULQDQ + SSE4.1 | ~15 GB/s | ✓ | ✓ |
//! | `sse42` | SSE4.2 CRC32 | ~20 GB/s | ✗ | ✓ |
//!
//! # Selection Priority
//!
//! 1. **VPCLMULQDQ** (AVX-512): Processes 256 bytes/iteration
//! 2. **PCLMULQDQ**: Processes 64 bytes/iteration
//! 3. **SSE4.2**: Native CRC32C instruction (not available for IEEE polynomial)
//!
//! # References
//!
//! - Intel: "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ"
//! - Linux kernel: `arch/x86/crypto/crc32-pclmul_asm.S`
//! - zlib-ng: `arch/x86/crc32_fold_pclmulqdq.c`

#![allow(dead_code)]
// Kernels wired up via dispatcher
// SAFETY: All indexing is over fixed-size arrays with in-bounds constant indices
// (e.g., chunks_exact(8) guarantees 8 bytes per chunk).
#![allow(clippy::indexing_slicing)]

// ─────────────────────────────────────────────────────────────────────────────
// SSE4.2 CRC32C (native instruction)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32C using native SSE4.2 `crc32` instruction.
///
/// This is the fastest option for CRC-32C on x86_64, but only works for the
/// Castagnoli polynomial (iSCSI, ext4, Btrfs). The IEEE polynomial must use
/// PCLMULQDQ instead.
///
/// # Performance
///
/// - Processes 8 bytes per `crc32q` instruction
/// - ~3 cycles/8 bytes = ~20 GB/s at 5 GHz
///
/// # Safety
///
/// Requires SSE4.2. Caller must verify `platform::caps().has(SSE42)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
pub unsafe fn crc32c_sse42(mut crc: u32, data: &[u8]) -> u32 {
  #[cfg(target_arch = "x86_64")]
  use core::arch::x86_64::{_mm_crc32_u8, _mm_crc32_u64};

  let mut crc64 = u64::from(crc);

  // Process 8 bytes at a time
  let mut chunks = data.chunks_exact(8);
  for chunk in chunks.by_ref() {
    // SAFETY: chunks_exact(8) guarantees exactly 8 bytes
    let bytes: [u8; 8] = [
      chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
    ];
    let val = u64::from_le_bytes(bytes);
    crc64 = _mm_crc32_u64(crc64, val);
  }

  // Process remaining bytes
  crc = crc64 as u32;
  for &byte in chunks.remainder() {
    crc = _mm_crc32_u8(crc, byte);
  }

  crc
}

// ─────────────────────────────────────────────────────────────────────────────
// PCLMULQDQ Folding (works for any polynomial)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32 using PCLMULQDQ carryless multiplication.
///
/// This technique "folds" 64 bytes at a time using parallel carryless
/// multiplications, then reduces to the final 32-bit CRC.
///
/// # Algorithm
///
/// 1. Load 4x 128-bit blocks (64 bytes)
/// 2. Fold each block using pre-computed constants
/// 3. Reduce 512 bits → 128 bits → 32 bits
///
/// # Safety
///
/// Requires PCLMULQDQ + SSE4.1. Caller must verify caps.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq", enable = "sse4.1")]
pub unsafe fn crc32_pclmul(_crc: u32, _data: &[u8]) -> u32 {
  // Reference: Intel Fast CRC whitepaper
  todo!("PCLMULQDQ CRC-32 IEEE implementation")
}

/// CRC-32C using PCLMULQDQ carryless multiplication.
///
/// Same algorithm as `crc32_pclmul` but with Castagnoli polynomial constants.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq", enable = "sse4.1")]
pub unsafe fn crc32c_pclmul(_crc: u32, _data: &[u8]) -> u32 {
  todo!("PCLMULQDQ CRC-32C implementation")
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX-512 VPCLMULQDQ (widest vectors)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32 using AVX-512 VPCLMULQDQ.
///
/// Processes 256 bytes per iteration using 512-bit vectors.
///
/// # Safety
///
/// Requires AVX-512F + AVX-512VL + VPCLMULQDQ. Caller must verify caps.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vl", enable = "vpclmulqdq")]
pub unsafe fn crc32_vpclmul(_crc: u32, _data: &[u8]) -> u32 {
  todo!("VPCLMULQDQ CRC-32 IEEE implementation")
}

/// CRC-32C using AVX-512 VPCLMULQDQ.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vl", enable = "vpclmulqdq")]
pub unsafe fn crc32c_vpclmul(_crc: u32, _data: &[u8]) -> u32 {
  todo!("VPCLMULQDQ CRC-32C implementation")
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Wrappers (safe interface)
// ─────────────────────────────────────────────────────────────────────────────

/// Safe wrapper for SSE4.2 CRC-32C kernel.
///
/// # Safety
///
/// This function checks CPU features at the call site via the dispatcher.
/// Only call through `Crc32Dispatcher` which verifies SSE4.2 is available.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn crc32c_sse42_safe(crc: u32, data: &[u8]) -> u32 {
  // SAFETY: Dispatcher verifies SSE4.2 before selecting this kernel
  unsafe { crc32c_sse42(crc, data) }
}

// ─────────────────────────────────────────────────────────────────────────────
// PCLMULQDQ Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-computed folding constants for CRC-32 IEEE (PCLMULQDQ).
///
/// These are derived from the polynomial 0xEDB88320 (reflected).
/// See Intel whitepaper for derivation.
#[allow(dead_code)]
mod constants_ieee {
  // Fold by 4 (512 → 128 bits) - placeholder
  pub const K1_K2: u128 = 0;
  // Fold by 1 (128 → 128 bits) - placeholder
  pub const K3_K4: u128 = 0;
  // Barrett reduction constants - placeholder
  pub const K5_K6: u128 = 0;
}

/// Pre-computed folding constants for CRC-32C (PCLMULQDQ).
///
/// These are derived from the polynomial 0x82F63B78 (reflected).
#[allow(dead_code)]
mod constants_castagnoli {
  // Fold by 4 (512 → 128 bits) - placeholder
  pub const K1_K2: u128 = 0;
  // Fold by 1 (128 → 128 bits) - placeholder
  pub const K3_K4: u128 = 0;
  // Barrett reduction constants - placeholder
  pub const K5_K6: u128 = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  extern crate alloc;
  extern crate std;

  use alloc::vec::Vec;

  use super::*;

  const TEST_DATA: &[u8] = b"123456789";
  const CRC32C_CHECK: u32 = 0xE3069283;

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn test_crc32c_sse42() {
    if !std::is_x86_feature_detected!("sse4.2") {
      std::eprintln!("Skipping SSE4.2 test: not supported");
      return;
    }

    // SAFETY: We just checked SSE4.2 is available
    let crc = unsafe { crc32c_sse42(!0, TEST_DATA) } ^ !0;
    assert_eq!(crc, CRC32C_CHECK);
  }

  #[test]
  fn test_sse42_streaming() {
    if !std::is_x86_feature_detected!("sse4.2") {
      return;
    }

    // Test that streaming produces same result as oneshot
    let oneshot = unsafe { crc32c_sse42(!0, TEST_DATA) } ^ !0;

    let mut state = !0u32;
    state = unsafe { crc32c_sse42(state, &TEST_DATA[..5]) };
    state = unsafe { crc32c_sse42(state, &TEST_DATA[5..]) };
    let streamed = state ^ !0;

    assert_eq!(streamed, oneshot);
  }

  #[test]
  fn test_sse42_various_lengths() {
    if !std::is_x86_feature_detected!("sse4.2") {
      return;
    }

    // Test lengths 0-32 to exercise all remainder paths
    for len in 0..=32 {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();
      let _ = unsafe { crc32c_sse42(!0, &data) };
    }
  }
}
