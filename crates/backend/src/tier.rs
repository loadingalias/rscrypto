//! Kernel acceleration tiers.
//!
//! Tiers represent levels of hardware acceleration, from reference
//! implementations to wide SIMD operations. Higher tiers offer better
//! performance but have stricter hardware requirements.
//!
//! # Tier System
//!
//! | Tier | Name | Description |
//! |------|------|-------------|
//! | 0 | Reference | Bitwise implementation - always available, for verification |
//! | 1 | Portable | Table-based slice-by-N - always available, production fallback |
//! | 2 | HwCrc | Native CRC instructions - CRC-32/32C only on x86_64, aarch64 |
//! | 3 | Folding | PCLMUL/PMULL/VPMSUM/VGFM/Zbc - carryless multiply folding |
//! | 4 | Wide | VPCLMUL/EOR3/SVE2/Zvbc - wide SIMD / advanced folding |

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
  /// Performance: 15-25 GB/s. Not available for CRC-16, CRC-24, or CRC-64.
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
  #[inline]
  #[must_use]
  pub const fn is_simd(self) -> bool {
    matches!(self, Self::HwCrc | Self::Folding | Self::Wide)
  }

  /// All tiers in ascending order.
  pub const ALL: [Self; 5] = [Self::Reference, Self::Portable, Self::HwCrc, Self::Folding, Self::Wide];
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
