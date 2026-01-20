//! Generic kernel selection and dispatch infrastructure for hash algorithms.
//!
//! This module provides shared infrastructure for kernel selection that works
//! across all hash algorithms. The patterns mirror `checksum::common::kernels`
//! to maintain consistency across the rscrypto crate ecosystem.
//!
//! # Kernel Tier System
//!
//! Hash implementations follow a tiered kernel selection model. Higher tiers
//! offer better performance but have stricter hardware requirements.
//!
//! | Tier | Name | Description |
//! |------|------|-------------|
//! | 0 | Reference | Bitwise/textbook - always available, for verification |
//! | 1 | Portable | Optimized scalar - always available, production fallback |
//! | 2 | Hardware | Native hash instructions (SHA-NI, ARMv8 Crypto) |
//! | 3 | SIMD | Vectorized (SSSE3/NEON) - single-block acceleration |
//! | 4 | Wide | Wide SIMD (AVX2/AVX-512/SVE2) - multi-block parallel |
//!
//! ## Tier Availability by Algorithm
//!
//! | Algorithm | Tier 0 | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
//! |-----------|--------|--------|--------|--------|--------|
//! | BLAKE3 | Yes | Yes | No | Yes | Yes |
//! | SHA-256 | Yes | Yes | Yes* | Yes | No |
//! | SHA-512 | Yes | Yes | Yes† | Yes | Yes |
//! | Keccak | Yes | Yes | Yes‡ | Yes | Yes |
//! | BLAKE2b | Yes | Yes | No | Yes | Yes |
//! | BLAKE2s | Yes | Yes | No | Yes | Yes |
//! | Ascon | Yes | Yes | No | Yes | No |
//!
//! \* SHA-NI (x86_64), ARMv8 SHA256 Crypto |
//! † ARMv8.2 SHA512 extension |
//! ‡ ARMv8.2 SHA3 extension
//!
//! # Design Philosophy
//!
//! Rather than using traits (which add vtable overhead) or generics (which
//! complicate the code), we provide:
//!
//! - Shared constants for kernel naming
//! - Parallelism-level mapping (for multi-block kernels)
//! - Name selection helpers that work with static arrays
//!
//! This keeps the code explicit and allows each hash module to define its
//! own kernel arrays and selection logic while reusing common patterns.

// ─────────────────────────────────────────────────────────────────────────────
// Common Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Reference (bitwise/textbook) kernel name - canonical implementation for verification.
pub const REFERENCE: &str = "reference";

/// Portable fallback kernel name (used by all hash algorithms).
pub const PORTABLE: &str = "portable";

// ─────────────────────────────────────────────────────────────────────────────
// Parallelism Mapping
// ─────────────────────────────────────────────────────────────────────────────

/// Map parallelism level to kernel array index.
///
/// Hash kernels that support multi-block parallel processing use this mapping:
/// - Index 0: 1-way (single block)
/// - Index 1: 2-way (AVX2 for 32-bit words, NEON 2-way)
/// - Index 2: 4-way (AVX-512 for 32-bit words, AVX2 for 64-bit words)
/// - Index 3: 8-way (AVX-512 for 64-bit words)
///
/// This provides a consistent mapping across architectures:
/// - x86_64: 1, 2, 4, 8-way depending on vector width and word size
/// - aarch64: 1, 2-way (4-way with SVE2)
///
/// # Arguments
///
/// * `parallelism` - Number of blocks processed in parallel
///
/// # Returns
///
/// Index into the kernel array (0-3).
#[inline]
#[must_use]
pub const fn parallelism_to_index(parallelism: u8) -> usize {
  match parallelism {
    8 => 3,
    4 => 2,
    2 => 1,
    _ => 0, // 1-way or fallback
  }
}

/// Maximum kernel array size for parallelism levels.
///
/// Kernel arrays: `[1-way, 2-way, 4-way, 8-way]`
pub const MAX_PARALLELISM_LEVELS: usize = 4;

// ─────────────────────────────────────────────────────────────────────────────
// Size Thresholds
// ─────────────────────────────────────────────────────────────────────────────

/// Default minimum input size (bytes) to use SIMD kernels.
///
/// Below this threshold, the setup overhead of SIMD instructions typically
/// exceeds the benefit. Actual thresholds are algorithm and platform-specific;
/// use tuning data when available.
pub const DEFAULT_SIMD_THRESHOLD: usize = 64;

/// Default minimum input size (bytes) to use wide SIMD kernels (AVX-512, etc.).
///
/// Wide SIMD requires more data to amortize setup costs. Again, this is a
/// conservative default; use tuning data when available.
pub const DEFAULT_WIDE_SIMD_THRESHOLD: usize = 256;

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Selection Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Calculate the optimal parallelism level for a given buffer length.
///
/// This is a generic helper; specific algorithms may have different thresholds.
///
/// # Arguments
///
/// * `len` - Input buffer length in bytes
/// * `block_size` - Algorithm block size in bytes
/// * `max_parallelism` - Maximum parallelism supported by available kernels
///
/// # Returns
///
/// The parallelism level to use (1, 2, 4, or 8).
#[inline]
#[must_use]
pub const fn select_parallelism(len: usize, block_size: usize, max_parallelism: u8) -> u8 {
  // Need at least 2 blocks for 2-way, 4 for 4-way, etc.
  // Add some headroom (2x) to make parallelism worthwhile
  let blocks_available = len / block_size;

  if max_parallelism >= 8 && blocks_available >= 16 {
    8
  } else if max_parallelism >= 4 && blocks_available >= 8 {
    4
  } else if max_parallelism >= 2 && blocks_available >= 4 {
    2
  } else {
    1
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_parallelism_to_index() {
    assert_eq!(parallelism_to_index(1), 0);
    assert_eq!(parallelism_to_index(2), 1);
    assert_eq!(parallelism_to_index(4), 2);
    assert_eq!(parallelism_to_index(8), 3);
    // Edge cases
    assert_eq!(parallelism_to_index(0), 0);
    assert_eq!(parallelism_to_index(3), 0); // Falls back to 1-way
    assert_eq!(parallelism_to_index(5), 0);
  }

  #[test]
  fn test_select_parallelism() {
    const BLOCK_SIZE: usize = 64; // Typical for SHA-256, BLAKE3

    // Small buffers: 1-way
    assert_eq!(select_parallelism(64, BLOCK_SIZE, 8), 1); // 1 block
    assert_eq!(select_parallelism(128, BLOCK_SIZE, 8), 1); // 2 blocks
    assert_eq!(select_parallelism(192, BLOCK_SIZE, 8), 1); // 3 blocks

    // Medium buffers: 2-way
    assert_eq!(select_parallelism(256, BLOCK_SIZE, 8), 2); // 4 blocks
    assert_eq!(select_parallelism(384, BLOCK_SIZE, 8), 2); // 6 blocks
    assert_eq!(select_parallelism(448, BLOCK_SIZE, 8), 2); // 7 blocks

    // Larger buffers: 4-way
    assert_eq!(select_parallelism(512, BLOCK_SIZE, 8), 4); // 8 blocks
    assert_eq!(select_parallelism(768, BLOCK_SIZE, 8), 4); // 12 blocks
    assert_eq!(select_parallelism(960, BLOCK_SIZE, 8), 4); // 15 blocks

    // Large buffers: 8-way
    assert_eq!(select_parallelism(1024, BLOCK_SIZE, 8), 8); // 16 blocks
    assert_eq!(select_parallelism(4096, BLOCK_SIZE, 8), 8); // 64 blocks

    // Respect max_parallelism
    assert_eq!(select_parallelism(4096, BLOCK_SIZE, 4), 4);
    assert_eq!(select_parallelism(4096, BLOCK_SIZE, 2), 2);
    assert_eq!(select_parallelism(4096, BLOCK_SIZE, 1), 1);
  }

  #[test]
  fn test_constants() {
    assert_eq!(REFERENCE, "reference");
    assert_eq!(PORTABLE, "portable");
    // Use const block to satisfy clippy::assertions_on_constants
    const _: () = assert!(DEFAULT_SIMD_THRESHOLD < DEFAULT_WIDE_SIMD_THRESHOLD);
  }
}
