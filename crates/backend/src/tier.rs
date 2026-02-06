//! Kernel acceleration tiers.
//!
//! Tiers represent levels of hardware acceleration, from reference
//! implementations to wide SIMD operations. Higher tiers offer better
//! performance but have stricter hardware requirements.
//!
//! # Tier Overview
//!
//! | Tier | Name | Throughput | Description |
//! |------|------|------------|-------------|
//! | 0 | Reference | ~100 MB/s | Bitwise - always available, for verification |
//! | 1 | Portable | 1-3 GB/s | Table-based slice-by-N - production fallback |
//! | 2 | HwCrc | 15-25 GB/s | Hardware instructions (CRC32, SHA extensions, etc.) |
//! | 3 | Folding | 8-15 GB/s | Carryless multiply (PCLMUL/PMULL/VPMSUM/Zbc) |
//! | 4 | Wide | 20-40 GB/s | Wide SIMD (VPCLMUL/EOR3/SVE2/Zvbc) |

use core::fmt;

use platform::Tune;

/// Tier-level thresholds used by policy dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierThresholds {
  pub small: usize,
  pub fold: usize,
  pub wide: usize,
}

/// Kernel acceleration tier.
///
/// Tiers are ordered from lowest (always available) to highest (best
/// performance, most stringent requirements). Selection policies use
/// tiers to classify available backends and make dispatch decisions.
///
/// # Ordering
///
/// Tiers implement `Ord` with higher tiers being "greater". This allows
/// simple comparisons like `tier >= KernelTier::Folding` to check if
/// SIMD acceleration is available.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum KernelTier {
  /// Tier 0: Bitwise reference implementation.
  ///
  /// Always available, always correct. Used for verification and
  /// audit-critical paths where absolute clarity trumps performance.
  /// Performance: ~100 MB/s.
  #[default]
  Reference = 0,

  /// Tier 1: Portable table-based implementation.
  ///
  /// Always available. Production fallback using slice-by-N lookup tables.
  /// Performance: 1-3 GB/s on modern CPUs.
  Portable = 1,

  /// Tier 2: Native hardware CRC instructions.
  ///
  /// Available on:
  /// - x86_64: SSE4.2 `crc32` (CRC-32C only, poly-locked)
  /// - aarch64: CRC extension (CRC-32 and CRC-32C)
  ///
  /// More generally, this tier is reserved for dedicated ISA extensions that
  /// accelerate an algorithm's inner loop without full SIMD "folding" kernels
  /// (e.g., SHA extensions, CRC32 instructions).
  ///
  /// Performance: 15-25 GB/s for CRC. Availability varies by algorithm.
  HwCrc = 2,

  /// Tier 3: Carryless multiply folding.
  ///
  /// Available on:
  /// - x86_64: PCLMULQDQ
  /// - aarch64: PMULL
  /// - Power: VPMSUMD
  /// - s390x: VGFM
  /// - riscv64: Zbc
  ///
  /// Performance: 8-15 GB/s. Polynomial-agnostic.
  Folding = 3,

  /// Tier 4: Wide SIMD / advanced folding.
  ///
  /// Available on:
  /// - x86_64: VPCLMULQDQ (AVX-512)
  /// - aarch64: PMULL+EOR3 (SHA3), SVE2 PMULL
  /// - riscv64: Zvbc (vector carryless multiply)
  ///
  /// Performance: 20-40 GB/s. Requires modern hardware.
  Wide = 4,
}

impl KernelTier {
  /// Alias for [`KernelTier::HwCrc`] with a generic name.
  ///
  /// Many algorithms use dedicated hardware instruction extensions (not just CRC),
  /// and it is often useful to talk about this tier generically as "hardware".
  #[allow(non_upper_case_globals)]
  pub const Hardware: Self = Self::HwCrc;

  /// Convert to numeric value.
  #[inline]
  #[must_use]
  pub const fn as_u8(self) -> u8 {
    self as u8
  }

  /// Human-readable tier name.
  #[inline]
  #[must_use]
  pub const fn name(self) -> &'static str {
    match self {
      Self::Reference => "reference",
      Self::Portable => "portable",
      Self::HwCrc => "hwcrc",
      Self::Folding => "folding",
      Self::Wide => "wide",
    }
  }

  /// Check if this tier requires runtime capability detection.
  ///
  /// Reference and Portable tiers are always available; higher tiers
  /// require runtime detection to verify hardware support.
  #[inline]
  #[must_use]
  pub const fn requires_runtime_detection(self) -> bool {
    matches!(self, Self::HwCrc | Self::Folding | Self::Wide)
  }

  /// Check if this tier uses SIMD acceleration.
  ///
  /// Currently equivalent to [`requires_runtime_detection`](Self::requires_runtime_detection),
  /// but kept separate for semantic clarity.
  #[inline]
  #[must_use]
  pub const fn is_simd(self) -> bool {
    matches!(self, Self::HwCrc | Self::Folding | Self::Wide)
  }

  /// All tiers in ascending order.
  pub const ALL: [Self; 5] = [Self::Reference, Self::Portable, Self::HwCrc, Self::Folding, Self::Wide];

  /// Build dispatch thresholds for this tier from tune hints.
  #[inline]
  #[must_use]
  pub fn policy_thresholds(self, tune: &Tune) -> TierThresholds {
    match self {
      Self::Reference | Self::Portable => TierThresholds {
        small: usize::MAX,
        fold: usize::MAX,
        wide: usize::MAX,
      },
      Self::HwCrc => TierThresholds {
        small: tune.hwcrc_threshold,
        fold: usize::MAX,
        wide: usize::MAX,
      },
      Self::Folding => TierThresholds {
        small: tune.pclmul_threshold,
        fold: tune.pclmul_threshold,
        wide: usize::MAX,
      },
      Self::Wide => {
        let fold_to_wide = if tune.fast_wide_ops { 512 } else { 2048 };
        TierThresholds {
          small: tune.pclmul_threshold,
          fold: tune.pclmul_threshold,
          wide: fold_to_wide,
        }
      }
    }
  }
}

impl fmt::Display for KernelTier {
  #[inline]
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(self.name())
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn tier_ordering() {
    assert!(KernelTier::Reference < KernelTier::Portable);
    assert!(KernelTier::Portable < KernelTier::HwCrc);
    assert!(KernelTier::HwCrc < KernelTier::Folding);
    assert!(KernelTier::Folding < KernelTier::Wide);
  }

  #[test]
  fn tier_values() {
    assert_eq!(KernelTier::Reference.as_u8(), 0);
    assert_eq!(KernelTier::Portable.as_u8(), 1);
    assert_eq!(KernelTier::HwCrc.as_u8(), 2);
    assert_eq!(KernelTier::Folding.as_u8(), 3);
    assert_eq!(KernelTier::Wide.as_u8(), 4);
  }

  #[test]
  fn tier_names() {
    assert_eq!(KernelTier::Reference.name(), "reference");
    assert_eq!(KernelTier::Portable.name(), "portable");
    assert_eq!(KernelTier::Wide.name(), "wide");
  }

  #[test]
  fn runtime_detection() {
    assert!(!KernelTier::Reference.requires_runtime_detection());
    assert!(!KernelTier::Portable.requires_runtime_detection());
    assert!(KernelTier::HwCrc.requires_runtime_detection());
    assert!(KernelTier::Folding.requires_runtime_detection());
    assert!(KernelTier::Wide.requires_runtime_detection());
  }

  #[test]
  fn default_is_reference() {
    assert_eq!(KernelTier::default(), KernelTier::Reference);
  }
}
