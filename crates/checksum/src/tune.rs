//! Cached algorithm-family thresholds.
//!
//! This module caches the PCLMUL and hardware CRC thresholds from the
//! platform crate to avoid repeated `OnceLock` lookups in hot paths.
//!
//! # Usage
//!
//! ```rust
//! use checksum::tune;
//!
//! let pclmul = tune::pclmul_threshold();
//! let hwcrc = tune::hwcrc_threshold();
//! assert!(pclmul > 0);
//! assert!(hwcrc > 0);
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// Threshold Caching
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "std")]
mod cache {
  use std::sync::OnceLock;

  static PCLMUL_THRESHOLD: OnceLock<usize> = OnceLock::new();
  static HWCRC_THRESHOLD: OnceLock<usize> = OnceLock::new();

  /// Get the cached PCLMUL/PMULL threshold.
  ///
  /// This is the minimum buffer size where carryless multiply operations
  /// (PCLMULQDQ, VPCLMULQDQ, PMULL) become faster than table-based CRC.
  #[inline]
  pub fn pclmul_threshold() -> usize {
    *PCLMUL_THRESHOLD.get_or_init(|| platform::tune().pclmul_threshold)
  }

  /// Get the cached hardware CRC instruction threshold.
  ///
  /// This is the minimum buffer size where hardware CRC instructions
  /// (SSE4.2 crc32, aarch64 CRC32 extension) become faster than table-based CRC.
  #[inline]
  pub fn hwcrc_threshold() -> usize {
    *HWCRC_THRESHOLD.get_or_init(|| platform::tune().hwcrc_threshold)
  }
}

#[cfg(all(not(feature = "std"), target_has_atomic = "ptr"))]
mod cache {
  use core::sync::atomic::{AtomicUsize, Ordering};

  // Use 0 as sentinel for "not initialized"
  // Actual thresholds are always > 0
  static PCLMUL_THRESHOLD: AtomicUsize = AtomicUsize::new(0);
  static HWCRC_THRESHOLD: AtomicUsize = AtomicUsize::new(0);

  /// Get the cached PCLMUL/PMULL threshold.
  #[inline]
  pub fn pclmul_threshold() -> usize {
    let cached = PCLMUL_THRESHOLD.load(Ordering::Relaxed);
    if cached != 0 {
      return cached;
    }
    let value = platform::tune().pclmul_threshold;
    PCLMUL_THRESHOLD.store(value, Ordering::Relaxed);
    value
  }

  /// Get the cached hardware CRC instruction threshold.
  #[inline]
  pub fn hwcrc_threshold() -> usize {
    let cached = HWCRC_THRESHOLD.load(Ordering::Relaxed);
    if cached != 0 {
      return cached;
    }
    let value = platform::tune().hwcrc_threshold;
    HWCRC_THRESHOLD.store(value, Ordering::Relaxed);
    value
  }
}

#[cfg(all(not(feature = "std"), not(target_has_atomic = "ptr")))]
mod cache {
  /// Get the PCLMUL/PMULL threshold (no caching on this target).
  #[inline]
  pub fn pclmul_threshold() -> usize {
    platform::tune().pclmul_threshold
  }

  /// Get the hardware CRC instruction threshold (no caching on this target).
  #[inline]
  pub fn hwcrc_threshold() -> usize {
    platform::tune().hwcrc_threshold
  }
}

// Re-export the cache functions at module level
pub use cache::*;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_pclmul_threshold_positive() {
    assert!(pclmul_threshold() > 0, "pclmul_threshold should be > 0");
  }

  #[test]
  fn test_hwcrc_threshold_positive() {
    assert!(hwcrc_threshold() > 0, "hwcrc_threshold should be > 0");
  }

  #[test]
  fn test_hwcrc_le_pclmul() {
    assert!(
      hwcrc_threshold() <= pclmul_threshold(),
      "hwcrc_threshold ({}) should be <= pclmul_threshold ({})",
      hwcrc_threshold(),
      pclmul_threshold()
    );
  }

  #[test]
  fn test_thresholds_are_cached() {
    // Call multiple times - should return same values
    let pclmul1 = pclmul_threshold();
    let pclmul2 = pclmul_threshold();
    assert_eq!(pclmul1, pclmul2);

    let hwcrc1 = hwcrc_threshold();
    let hwcrc2 = hwcrc_threshold();
    assert_eq!(hwcrc1, hwcrc2);
  }
}
