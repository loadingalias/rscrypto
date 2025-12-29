//! Generic kernel selection and dispatch infrastructure for CRC algorithms.
//!
//! This module provides shared infrastructure for kernel selection that works
//! across all CRC widths (16, 24, 32, 64). The patterns are identical:
//!
//! 1. Check if `len < small_threshold` → use portable
//! 2. Check forced backend override
//! 3. Check CPU capabilities and thresholds → select SIMD tier
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

/// Portable fallback kernel name (used by all CRC widths).
pub const PORTABLE_SLICE16: &str = "portable/slice16";

/// Portable slice-by-8 kernel name.
#[allow(dead_code)] // Used by CRC-32 module for small buffer fallback
pub const PORTABLE_SLICE8: &str = "portable/slice8";

/// Portable slice-by-4 kernel name.
#[cfg(test)] // Will be used when CRC-16 module is added
#[allow(dead_code)]
pub const PORTABLE_SLICE4: &str = "portable/slice4";

// ─────────────────────────────────────────────────────────────────────────────
// Stream Mapping
// ─────────────────────────────────────────────────────────────────────────────

/// Map stream count to kernel array index.
///
/// Kernel arrays are ordered: `[1-way, 2-way, 3/4-way, 7-way, 8-way]`
///
/// This provides a consistent mapping across all architectures:
/// - x86_64: 1, 2, 4, 7, 8-way
/// - aarch64: 1, 2, 3-way (slot 2 used for 3-way)
/// - powerpc64: 1, 2, 4, 8-way
/// - s390x: 1, 2, 4-way
/// - riscv64: 1, 2, 4-way
#[inline]
#[must_use]
pub const fn stream_to_index(streams: u8) -> usize {
  match streams {
    8 => 4,
    7 => 3,
    3 | 4 => 2,
    2 => 1,
    _ => 0,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Name Selection
// ─────────────────────────────────────────────────────────────────────────────

/// Select kernel name from a static array based on stream count.
///
/// # Arguments
///
/// * `names` - Static array of kernel names ordered by stream count
/// * `small` - Optional small-buffer kernel name (used when `len < fold_bytes`)
/// * `streams` - Selected stream count (1, 2, 3, 4, 7, or 8)
/// * `len` - Input buffer length
/// * `fold_bytes` - Minimum bytes for full folding (e.g., 128 for CRC-64)
///
/// # Returns
///
/// The kernel name that would be selected for the given parameters.
#[inline]
#[must_use]
pub fn select_name(
  names: &[&'static str],
  small: Option<&'static str>,
  streams: u8,
  len: usize,
  fold_bytes: usize,
) -> &'static str {
  // Use small kernel for buffers below fold threshold
  if let Some(s) = small
    && len < fold_bytes
  {
    return s;
  }

  // Select from the names array based on stream count
  // Fallback to first element (1-way kernel) if index is out of bounds
  names
    .get(stream_to_index(streams))
    .or_else(|| names.first())
    .copied()
    .unwrap_or(PORTABLE_SLICE16)
}

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
    i += 1;
  }

  // Default: single stream if buffer is large enough for any folding
  if len >= fold_bytes {
    1
  } else {
    0 // Use portable
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch Macros
// ─────────────────────────────────────────────────────────────────────────────
//
// These macros generate type-specific dispatch functions. We use macros
// instead of generics to avoid any runtime overhead and keep the generated
// code optimal for each CRC width.

/// Define dispatch functions for a specific CRC function type.
///
/// This macro generates:
/// - `dispatch_streams`: Dispatch to stream variant based on count
/// - `dispatch_with_small`: Dispatch with small buffer handling
///
/// # Example
///
/// ```rust
/// use checksum::dispatchers::Crc32Fn;
///
/// checksum::define_crc_dispatch!(Crc32Fn, u32);
///
/// fn k(crc: u32, _data: &[u8]) -> u32 {
///   crc.wrapping_add(1)
/// }
///
/// const KERNELS: [Crc32Fn; 5] = [k, k, k, k, k];
/// let out = dispatch_streams(&KERNELS, 1, 0, b"");
/// assert_eq!(out, 1);
/// ```
#[macro_export]
macro_rules! define_crc_dispatch {
  ($fn_type:ty, $state_type:ty) => {
    /// Dispatch to stream variant based on stream count.
    ///
    /// Kernels array: `[1-way, 2-way, 3/4-way, 7-way, 8-way]`
    #[inline]
    #[allow(clippy::indexing_slicing)] // stream_to_index returns 0-4, array is [_; 5]
    pub fn dispatch_streams(kernels: &[$fn_type; 5], streams: u8, crc: $state_type, data: &[u8]) -> $state_type {
      kernels[$crate::__internal::stream_to_index(streams)](crc, data)
    }

    /// Dispatch with small buffer handling.
    #[inline]
    #[allow(dead_code)] // Not all CRC widths/arches use small-buffer kernels yet.
    #[allow(clippy::indexing_slicing)] // stream_to_index returns 0-4, array is [_; 5]
    pub fn dispatch_with_small(
      kernels: &[$fn_type; 5],
      small: $fn_type,
      streams: u8,
      len: usize,
      fold_bytes: usize,
      crc: $state_type,
      data: &[u8],
    ) -> $state_type {
      if len < fold_bytes {
        small(crc, data)
      } else {
        kernels[$crate::__internal::stream_to_index(streams)](crc, data)
      }
    }
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_stream_to_index() {
    assert_eq!(stream_to_index(1), 0);
    assert_eq!(stream_to_index(2), 1);
    assert_eq!(stream_to_index(3), 2);
    assert_eq!(stream_to_index(4), 2);
    assert_eq!(stream_to_index(7), 3);
    assert_eq!(stream_to_index(8), 4);
  }

  #[test]
  fn test_select_name_basic() {
    let names = &[
      "kernel/1way",
      "kernel/2way",
      "kernel/4way",
      "kernel/7way",
      "kernel/8way",
    ];

    assert_eq!(select_name(names, None, 1, 256, 128), "kernel/1way");
    assert_eq!(select_name(names, None, 2, 256, 128), "kernel/2way");
    assert_eq!(select_name(names, None, 4, 256, 128), "kernel/4way");
    assert_eq!(select_name(names, None, 7, 256, 128), "kernel/7way");
    assert_eq!(select_name(names, None, 8, 256, 128), "kernel/8way");
  }

  #[test]
  fn test_select_name_with_small() {
    let names = &["kernel/1way", "kernel/2way"];
    let small = "kernel/small";

    // Below fold threshold: use small kernel
    assert_eq!(select_name(names, Some(small), 1, 64, 128), "kernel/small");

    // At or above fold threshold: use stream kernel
    assert_eq!(select_name(names, Some(small), 1, 128, 128), "kernel/1way");
    assert_eq!(select_name(names, Some(small), 2, 256, 128), "kernel/2way");
  }

  #[test]
  fn test_select_name_fallback() {
    let names = &["kernel/1way"];

    // Out of bounds stream count falls back to first element
    assert_eq!(select_name(names, None, 7, 256, 128), "kernel/1way");

    // Empty names array falls back to portable
    let empty: &[&str] = &[];
    assert_eq!(select_name(empty, None, 1, 256, 128), PORTABLE_SLICE16);
  }

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
