//! aarch64 CPU detection and feature identification.
//!
//! This module provides feature detection for ARM processors:
//! - NEON (baseline SIMD)
//! - PMULL (polynomial multiplication for CRC/GHASH)
//! - CRC32 extension (hardware CRC32C)
//! - SHA3 extension (EOR3 for optimized folding)
//!
//! # Microarchitecture-Specific Optimizations
//!
//! | Microarch | PMULL | CRC | SHA3/EOR3 | Optimal CRC Config |
//! |-----------|-------|-----|-----------|-------------------|
//! | Cortex-A53/55 | ✓ | ✓ | ❌ | v12e_v1 |
//! | Cortex-A72/76 | ✓ | ✓ | ❌ | v12e_v1 |
//! | Apple M1+ | ✓ | ✓ | ✓ | v9s3x2e_s3 |
//! | Neoverse N1/V1 | ✓ | ✓ | ✓ | v9s3x2e_s3 |

/// aarch64 feature capability set.
///
/// Unlike x86_64, ARM doesn't have easily identifiable microarchitectures
/// via CPUID. Instead, we detect feature combinations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Features {
  /// NEON (Advanced SIMD) - baseline for all AArch64
  pub neon: bool,

  /// PMULL/PMULL2 (polynomial multiplication) - for CRC/GHASH
  pub pmull: bool,

  /// CRC32 extension (hardware CRC32C instruction)
  pub crc: bool,

  /// SHA3 extension (includes EOR3 for 3-way XOR)
  pub sha3: bool,

  /// AES extension (for AES-NI equivalent)
  pub aes: bool,

  /// SHA2 extension (hardware SHA-256)
  pub sha2: bool,
}

impl Features {
  /// Detect features at runtime (requires `std`).
  #[cfg(feature = "std")]
  #[inline]
  #[must_use]
  pub fn detect() -> Self {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Features> = OnceLock::new();
    *CACHED.get_or_init(Self::detect_uncached)
  }

  /// Detect features without caching.
  #[cfg(feature = "std")]
  #[inline]
  #[must_use]
  pub fn detect_uncached() -> Self {
    Self {
      neon: std::arch::is_aarch64_feature_detected!("neon"),
      pmull: std::arch::is_aarch64_feature_detected!("aes"), // PMULL is part of crypto/aes
      crc: std::arch::is_aarch64_feature_detected!("crc"),
      sha3: std::arch::is_aarch64_feature_detected!("sha3"),
      aes: std::arch::is_aarch64_feature_detected!("aes"),
      sha2: std::arch::is_aarch64_feature_detected!("sha2"),
    }
  }

  /// Returns `true` if PMULL + EOR3 (SHA3) are available for optimal CRC.
  ///
  /// This combination enables the fastest CRC paths on Apple M1+ and
  /// ARM Neoverse processors.
  #[inline]
  #[must_use]
  pub const fn has_pmull_eor3(self) -> bool {
    self.pmull && self.sha3
  }

  /// Returns `true` if hardware CRC32 extension is available.
  ///
  /// Used for CRC32C (Castagnoli polynomial only).
  #[inline]
  #[must_use]
  pub const fn has_crc(self) -> bool {
    self.crc
  }

  /// Returns the recommended small-buffer threshold for SIMD.
  #[inline]
  #[must_use]
  pub const fn simd_threshold(self) -> usize {
    if self.has_pmull_eor3() {
      // Apple M1+ can efficiently use PMULL even for smaller buffers
      64
    } else if self.pmull {
      256
    } else {
      64 // CRC extension is efficient for all sizes
    }
  }
}

impl Default for Features {
  /// Returns a conservative feature set (NEON only).
  #[inline]
  fn default() -> Self {
    Self {
      neon: true, // Always available on AArch64
      pmull: false,
      crc: false,
      sha3: false,
      aes: false,
      sha2: false,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  #[cfg(feature = "std")]
  fn test_detect_runs() {
    let features = Features::detect();
    // NEON is always available on AArch64
    assert!(features.neon);
  }

  #[test]
  fn test_feature_combinations() {
    let full = Features {
      neon: true,
      pmull: true,
      crc: true,
      sha3: true,
      aes: true,
      sha2: true,
    };
    assert!(full.has_pmull_eor3());
    assert!(full.has_crc());

    let no_sha3 = Features { sha3: false, ..full };
    assert!(!no_sha3.has_pmull_eor3());
  }

  #[test]
  fn test_simd_threshold() {
    let with_eor3 = Features {
      neon: true,
      pmull: true,
      sha3: true,
      ..Default::default()
    };
    assert_eq!(with_eor3.simd_threshold(), 64);

    let pmull_only = Features {
      neon: true,
      pmull: true,
      sha3: false,
      ..Default::default()
    };
    assert_eq!(pmull_only.simd_threshold(), 256);
  }
}
