//! Generic kernel selection and dispatch infrastructure for CRC algorithms.
//!
//! This module provides shared infrastructure for kernel selection that works
//! across all CRC widths (16, 24, 32, 64). The patterns are identical:
//!
//! 1. Check if `len < small_threshold` → use portable
//! 2. Check forced backend override
//! 3. Check CPU capabilities and thresholds → select SIMD tier
//!
//! # Kernel Tier System
//!
//! CRC implementations follow a tiered kernel selection model. Higher tiers
//! offer better performance but have stricter hardware requirements.
//!
//! | Tier | Name | Description |
//! |------|------|-------------|
//! | 0 | Reference | Bitwise implementation - always available, for verification |
//! | 1 | Portable | Table-based slice-by-N - always available, production fallback |
//! | 2 | HW CRC | Native CRC instructions - CRC-32/32C only on x86_64 (SSE4.2), aarch64 (CRC ext) |
//! | 3 | Folding | PCLMUL/PMULL/VPMSUM/VGFM/Zbc - carryless multiply folding |
//! | 4 | Wide | VPCLMUL/EOR3/SVE2/Zvbc - wide SIMD / advanced folding |
//!
//! ## Tier Availability by CRC Width
//!
//! | CRC Width | Tier 0 | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
//! |-----------|--------|--------|--------|--------|--------|
//! | CRC-16 | Yes | Yes | No | Yes | Yes* |
//! | CRC-24 | Yes | Yes | No | Yes | Yes* |
//! | CRC-32/32C | Yes | Yes | Yes | Yes | Yes |
//! | CRC-64 | Yes | Yes | No | Yes | Yes |
//!
//! *CRC-16/CRC-24 Tier 4 is x86_64 VPCLMUL only.
//!
//! # Design Philosophy
//!
//! Rather than using traits (which add vtable overhead) or generics (which
//! complicate the code), we provide:
//!
//! - Shared constants and stream-to-index mapping
//! - Type aliases and dispatch macros for each CRC width
//! - Name selection helpers that work with static arrays
//!
//! This keeps the code explicit and allows each CRC module to define its
//! own kernel arrays and selection logic while reusing common patterns.

// ─────────────────────────────────────────────────────────────────────────────
// Common Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Reference (bitwise) kernel name - canonical implementation for verification.
pub const REFERENCE: &str = "reference/bitwise";

/// Portable fallback kernel name (used by all CRC widths).
#[cfg(any(feature = "crc32", feature = "crc64"))]
pub const PORTABLE_SLICE16: &str = "portable/slice16";

/// Portable slice-by-8 kernel name.
#[cfg(any(feature = "crc16", feature = "crc24"))]
pub const PORTABLE_SLICE8: &str = "portable/slice8";

// ─────────────────────────────────────────────────────────────────────────────
// Stream Mapping
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Stream Selection Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Calculate the maximum stream count supported for a given buffer length.
///
/// Each additional stream level requires roughly 2× the minimum buffer size
/// to amortize the merge overhead.
///
/// # Arguments
///
/// * `len` - Input buffer length
/// * `max_streams` - Maximum streams allowed by config
/// * `fold_bytes` - Bytes per fold block (e.g., 128 for CRC-64)
/// * `stream_thresholds` - Array of `(min_streams, min_bytes)` pairs, ordered descending
///
/// # Returns
///
/// The stream count to use (1, 2, 4, or 7/8 depending on architecture).
#[cfg(test)] // Will be used when CRC modules integrate stream selection
#[inline]
#[must_use]
#[allow(clippy::indexing_slicing)] // Loop guard `i < len()` ensures index is in bounds
pub const fn select_streams(len: usize, max_streams: u8, fold_bytes: usize, stream_thresholds: &[(u8, usize)]) -> u8 {
  // Check thresholds from highest to lowest
  let mut i = 0;
  while i < stream_thresholds.len() {
    let (streams, min_bytes) = stream_thresholds[i];
    if max_streams >= streams && len >= min_bytes {
      return streams;
    }
    i = i.strict_add(1);
  }

  // Default: single stream if buffer is large enough for any folding
  if len >= fold_bytes {
    1
  } else {
    0 // Use portable
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_select_streams() {
    const FOLD_BYTES: usize = 128;
    const THRESHOLDS: &[(u8, usize)] = &[
      (8, 8 * 2 * FOLD_BYTES), // 2048 bytes
      (7, 7 * 2 * FOLD_BYTES), // 1792 bytes
      (4, 4 * 2 * FOLD_BYTES), // 1024 bytes
      (2, 2 * 2 * FOLD_BYTES), // 512 bytes
    ];

    // Below minimum fold threshold
    assert_eq!(select_streams(64, 8, FOLD_BYTES, THRESHOLDS), 0);

    // Above fold threshold but below 2-way threshold
    assert_eq!(select_streams(256, 8, FOLD_BYTES, THRESHOLDS), 1);

    // 2-way threshold
    assert_eq!(select_streams(512, 8, FOLD_BYTES, THRESHOLDS), 2);
    assert_eq!(select_streams(1000, 8, FOLD_BYTES, THRESHOLDS), 2);

    // 4-way threshold
    assert_eq!(select_streams(1024, 8, FOLD_BYTES, THRESHOLDS), 4);
    assert_eq!(select_streams(1500, 8, FOLD_BYTES, THRESHOLDS), 4);

    // 7-way threshold
    assert_eq!(select_streams(1792, 8, FOLD_BYTES, THRESHOLDS), 7);
    assert_eq!(select_streams(2000, 8, FOLD_BYTES, THRESHOLDS), 7);

    // 8-way threshold
    assert_eq!(select_streams(2048, 8, FOLD_BYTES, THRESHOLDS), 8);
    assert_eq!(select_streams(4096, 8, FOLD_BYTES, THRESHOLDS), 8);

    // Respect max_streams config
    assert_eq!(select_streams(4096, 7, FOLD_BYTES, THRESHOLDS), 7);
    assert_eq!(select_streams(4096, 4, FOLD_BYTES, THRESHOLDS), 4);
    assert_eq!(select_streams(4096, 2, FOLD_BYTES, THRESHOLDS), 2);
    assert_eq!(select_streams(4096, 1, FOLD_BYTES, THRESHOLDS), 1);
  }
}
