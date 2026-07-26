//! Shared CRC kernel names and selection helpers.
//!
//! Each CRC module owns its capability and threshold policy. This module keeps
//! only the mechanics that are identical across widths.

/// Reference (bitwise) kernel name - canonical implementation for verification.
pub const REFERENCE: &str = "reference/bitwise";

/// Portable fallback kernel name (used by all CRC widths).
#[cfg(any(feature = "crc32", feature = "crc64"))]
pub const PORTABLE_SLICE16: &str = "portable/slice16";

/// Portable slice-by-8 kernel name.
#[cfg(any(feature = "crc16", feature = "crc24"))]
pub const PORTABLE_SLICE8: &str = "portable/slice8";

// TODO(T7): Integrate this selector only after target-native, per-CRC
// benchmarks prove the stream counts at every threshold; otherwise delete it.
/// Select the highest allowed stream count whose byte threshold is met.
///
/// `stream_thresholds` must be ordered from highest to lowest stream count.
#[cfg(test)]
#[inline]
#[must_use]
#[allow(clippy::indexing_slicing)] // Loop guard `i < len()` ensures index is in bounds
pub const fn select_streams(len: usize, max_streams: u8, fold_bytes: usize, stream_thresholds: &[(u8, usize)]) -> u8 {
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
