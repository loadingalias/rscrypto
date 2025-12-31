//! Kernel family identification.
//!
//! Families represent specific backend implementations across all supported
//! architectures. Each family maps to exactly one tier and requires specific
//! CPU capabilities.
//!
//! # Design Philosophy
//!
//! The `KernelFamily` enum is algorithm-agnostic: the same families apply to
//! CRC-64, CRC-32, Blake3, and future algorithms. This enables:
//!
//! - Unified forcing across algorithms (force PCLMUL for all CRCs)
//! - Introspection without allocation
//! - Policy caching with family-level granularity
//! - Consistent dispatch patterns across the codebase

use crate::{
  caps::{Arch, Caps},
  tier::KernelTier,
};

/// Specific backend implementation family.
///
/// Families are algorithm-agnostic identifiers for backend implementations.
/// Each family:
/// - Maps to exactly one [`KernelTier`]
/// - Requires specific CPU capabilities
/// - Is available only on certain architectures
///
/// # Naming Convention
///
/// Family names follow the pattern `{Arch}{Instruction}`:
/// - `X86Pclmul` - x86_64 PCLMULQDQ instruction
/// - `ArmPmull` - aarch64 PMULL instruction
/// - `PowerVpmsum` - powerpc64 VPMSUMD instruction
///
/// # Extensibility
///
/// The enum is `#[non_exhaustive]` to allow adding new families without
/// breaking changes when new architectures or instructions become available.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
#[non_exhaustive]
pub enum KernelFamily {
  // ─── Tier 0: Reference ───────────────────────────────────────────────────
  /// Bitwise reference implementation (all architectures).
  ///
  /// The canonical implementation used for verification and audit.
  /// Always available, no capability requirements.
  #[default]
  Reference = 0,

  // ─── Tier 1: Portable ────────────────────────────────────────────────────
  /// Table-based slice-by-N implementation (all architectures).
  ///
  /// Production fallback using precomputed lookup tables.
  /// Always available, no capability requirements.
  Portable = 1,

  // ─── Tier 2: Hardware CRC ────────────────────────────────────────────────
  /// x86_64 SSE4.2 hardware CRC (CRC-32C only).
  ///
  /// Uses the `crc32` instruction which is polynomial-locked to CRC-32C.
  /// Available on Intel Nehalem+ and AMD Bulldozer+.
  X86Crc32 = 10,

  /// aarch64 CRC extension (CRC-32 and CRC-32C).
  ///
  /// Uses dedicated CRC instructions from the ARMv8 CRC extension.
  /// Available on most ARMv8+ processors.
  ArmCrc32 = 11,

  // ─── Tier 3: Folding ─────────────────────────────────────────────────────
  /// x86_64 PCLMULQDQ carryless multiply.
  ///
  /// 128-bit carry-less multiplication for polynomial folding.
  /// Available on Intel Westmere+ and AMD Bulldozer+.
  X86Pclmul = 20,

  /// aarch64 PMULL carryless multiply.
  ///
  /// 64-bit polynomial multiplication using NEON crypto extensions.
  /// Available on ARMv8+ with crypto extensions.
  ArmPmull = 21,

  /// powerpc64 VPMSUMD vector carryless multiply.
  ///
  /// 128-bit vector polynomial multiply using AltiVec/VSX.
  /// Available on POWER8+.
  PowerVpmsum = 22,

  /// s390x VGFM Galois field multiply.
  ///
  /// Vector Galois field multiplication for CRC folding.
  /// Available on z13+ with vector facility.
  S390xVgfm = 23,

  /// riscv64 Zbc scalar carryless multiply.
  ///
  /// Scalar carry-less multiply from the Zbc extension.
  /// Available on RISC-V processors with Zbc support.
  RiscvZbc = 24,

  // ─── Tier 4: Wide ────────────────────────────────────────────────────────
  /// x86_64 VPCLMULQDQ (AVX-512 carryless multiply).
  ///
  /// 256/512-bit carry-less multiplication for wide folding.
  /// Available on Intel Ice Lake+ and AMD Zen4+.
  X86Vpclmul = 30,

  /// aarch64 PMULL + EOR3 (SHA3 three-way XOR).
  ///
  /// Uses PMULL with EOR3 instruction from SHA3 extension for
  /// more efficient folding. Available on Apple M1+ and Neoverse V1+.
  ArmPmullEor3 = 31,

  /// aarch64 SVE2 PMULL.
  ///
  /// Scalable Vector Extension 2 polynomial multiplication.
  /// Available on Neoverse V2+ and future ARM designs.
  ArmSve2Pmull = 32,

  /// riscv64 Zvbc vector carryless multiply.
  ///
  /// Vector carry-less multiply from the Zvbc extension.
  /// Available on RISC-V processors with vector crypto.
  RiscvZvbc = 33,
}

impl KernelFamily {
  /// Get the tier this family belongs to.
  #[inline]
  #[must_use]
  pub const fn tier(self) -> KernelTier {
    match self {
      Self::Reference => KernelTier::Reference,
      Self::Portable => KernelTier::Portable,
      Self::X86Crc32 | Self::ArmCrc32 => KernelTier::HwCrc,
      Self::X86Pclmul | Self::ArmPmull | Self::PowerVpmsum | Self::S390xVgfm | Self::RiscvZbc => KernelTier::Folding,
      Self::X86Vpclmul | Self::ArmPmullEor3 | Self::ArmSve2Pmull | Self::RiscvZvbc => KernelTier::Wide,
    }
  }

  /// Human-readable family name.
  ///
  /// Returns a path-style name like `"x86_64/pclmul"` that matches
  /// the kernel name convention used in benchmarks and diagnostics.
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Reference => "reference",
      Self::Portable => "portable",
      Self::X86Crc32 => "x86_64/crc32",
      Self::ArmCrc32 => "aarch64/crc32",
      Self::X86Pclmul => "x86_64/pclmul",
      Self::ArmPmull => "aarch64/pmull",
      Self::PowerVpmsum => "powerpc64/vpmsum",
      Self::S390xVgfm => "s390x/vgfm",
      Self::RiscvZbc => "riscv64/zbc",
      Self::X86Vpclmul => "x86_64/vpclmul",
      Self::ArmPmullEor3 => "aarch64/pmull-eor3",
      Self::ArmSve2Pmull => "aarch64/sve2-pmull",
      Self::RiscvZvbc => "riscv64/zvbc",
    }
  }

  /// Check if this family is available on the given architecture.
  ///
  /// Note: This only checks architecture compatibility, not CPU capabilities.
  /// Use [`required_caps`](Self::required_caps) to check actual capability
  /// requirements.
  #[inline]
  #[must_use]
  pub const fn is_available_on(self, arch: Arch) -> bool {
    match self {
      Self::Reference | Self::Portable => true,
      Self::X86Crc32 | Self::X86Pclmul | Self::X86Vpclmul => {
        matches!(arch, Arch::X86_64 | Arch::X86)
      }
      Self::ArmCrc32 | Self::ArmPmull | Self::ArmPmullEor3 | Self::ArmSve2Pmull => {
        matches!(arch, Arch::Aarch64 | Arch::Arm)
      }
      Self::PowerVpmsum => matches!(arch, Arch::Powerpc64),
      Self::S390xVgfm => matches!(arch, Arch::S390x),
      Self::RiscvZbc | Self::RiscvZvbc => matches!(arch, Arch::Riscv64 | Arch::Riscv32),
    }
  }

  /// Get the required capabilities for this family.
  ///
  /// Returns `Caps::NONE` for Reference and Portable families.
  /// For other families, returns the capability mask that must be
  /// satisfied for the family to be usable.
  #[inline]
  #[must_use]
  pub fn required_caps(self) -> Caps {
    use crate::caps::{aarch64, riscv, x86};

    match self {
      Self::Reference | Self::Portable => Caps::NONE,
      Self::X86Crc32 => x86::CRC32C_READY,
      Self::X86Pclmul => x86::PCLMUL_READY,
      Self::X86Vpclmul => x86::VPCLMUL_READY,
      Self::ArmCrc32 => aarch64::CRC_READY,
      Self::ArmPmull => aarch64::PMULL_READY,
      Self::ArmPmullEor3 => aarch64::PMULL_EOR3_READY,
      Self::ArmSve2Pmull => aarch64::SVE2_PMULL.union(aarch64::PMULL_READY),
      Self::PowerVpmsum => platform::caps::powerpc64::VPMSUM_READY,
      Self::S390xVgfm => platform::caps::s390x::VECTOR,
      Self::RiscvZbc => riscv::ZBC,
      Self::RiscvZvbc => riscv::ZVBC,
    }
  }

  /// Check if this family is available given the detected capabilities.
  #[inline]
  #[must_use]
  pub fn is_available(self, caps: Caps) -> bool {
    caps.has(self.required_caps())
  }

  /// Minimum bytes per lane for multi-stream folding.
  ///
  /// Multi-stream kernels divide the buffer into parallel lanes. Each lane
  /// must have enough data to amortize setup costs. This returns the minimum
  /// bytes *per lane* for this family.
  ///
  /// **NOTE**: These are initial conservative defaults. Run tune binaries
  /// to find empirical crossover points for each microarch.
  ///
  /// # Returns
  ///
  /// - `usize::MAX` for Reference/Portable (no multi-stream support)
  /// - Conservative values for SIMD families, to be refined by benchmarking
  #[inline]
  #[must_use]
  pub const fn min_bytes_per_lane(self) -> usize {
    match self {
      // Reference/Portable: no multi-stream, return MAX to disable
      Self::Reference | Self::Portable => usize::MAX,

      // HW CRC: low latency instructions, but memory-bound quickly
      Self::X86Crc32 | Self::ArmCrc32 => 128,

      // Folding tier: ~1 fold block per lane minimum
      Self::X86Pclmul | Self::ArmPmull | Self::PowerVpmsum | Self::S390xVgfm | Self::RiscvZbc => 256,

      // Wide tier: higher setup cost, need more per lane
      Self::X86Vpclmul => 512,
      Self::ArmPmullEor3 => 256, // EOR3 is still 128-bit NEON operations
      Self::ArmSve2Pmull => 512,
      Self::RiscvZvbc => 512,
    }
  }

  /// All families in the given tier.
  #[must_use]
  pub const fn families_in_tier(tier: KernelTier) -> &'static [Self] {
    match tier {
      KernelTier::Reference => &[Self::Reference],
      KernelTier::Portable => &[Self::Portable],
      KernelTier::HwCrc => &[Self::X86Crc32, Self::ArmCrc32],
      KernelTier::Folding => &[
        Self::X86Pclmul,
        Self::ArmPmull,
        Self::PowerVpmsum,
        Self::S390xVgfm,
        Self::RiscvZbc,
      ],
      KernelTier::Wide => &[
        Self::X86Vpclmul,
        Self::ArmPmullEor3,
        Self::ArmSve2Pmull,
        Self::RiscvZvbc,
      ],
    }
  }

  /// Get all families available on the current target architecture.
  ///
  /// Returns families in descending tier order (wide first, reference last)
  /// for use in selection loops.
  #[must_use]
  pub const fn families_for_current_arch() -> &'static [Self] {
    #[cfg(target_arch = "x86_64")]
    {
      &[
        Self::X86Vpclmul,
        Self::X86Pclmul,
        Self::X86Crc32,
        Self::Portable,
        Self::Reference,
      ]
    }
    #[cfg(target_arch = "aarch64")]
    {
      &[
        Self::ArmSve2Pmull,
        Self::ArmPmullEor3,
        Self::ArmPmull,
        Self::ArmCrc32,
        Self::Portable,
        Self::Reference,
      ]
    }
    #[cfg(target_arch = "powerpc64")]
    {
      &[Self::PowerVpmsum, Self::Portable, Self::Reference]
    }
    #[cfg(target_arch = "s390x")]
    {
      &[Self::S390xVgfm, Self::Portable, Self::Reference]
    }
    #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
    {
      &[Self::RiscvZvbc, Self::RiscvZbc, Self::Portable, Self::Reference]
    }
    #[cfg(not(any(
      target_arch = "x86_64",
      target_arch = "aarch64",
      target_arch = "powerpc64",
      target_arch = "s390x",
      target_arch = "riscv64",
      target_arch = "riscv32"
    )))]
    {
      &[Self::Portable, Self::Reference]
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn family_tiers() {
    assert_eq!(KernelFamily::Reference.tier(), KernelTier::Reference);
    assert_eq!(KernelFamily::Portable.tier(), KernelTier::Portable);
    assert_eq!(KernelFamily::X86Crc32.tier(), KernelTier::HwCrc);
    assert_eq!(KernelFamily::ArmCrc32.tier(), KernelTier::HwCrc);
    assert_eq!(KernelFamily::X86Pclmul.tier(), KernelTier::Folding);
    assert_eq!(KernelFamily::ArmPmull.tier(), KernelTier::Folding);
    assert_eq!(KernelFamily::X86Vpclmul.tier(), KernelTier::Wide);
    assert_eq!(KernelFamily::ArmPmullEor3.tier(), KernelTier::Wide);
  }

  #[test]
  fn family_names() {
    assert_eq!(KernelFamily::Reference.as_str(), "reference");
    assert_eq!(KernelFamily::Portable.as_str(), "portable");
    assert_eq!(KernelFamily::X86Pclmul.as_str(), "x86_64/pclmul");
    assert_eq!(KernelFamily::ArmPmull.as_str(), "aarch64/pmull");
    assert_eq!(KernelFamily::X86Vpclmul.as_str(), "x86_64/vpclmul");
  }

  #[test]
  fn portable_always_available() {
    assert!(KernelFamily::Reference.is_available(Caps::NONE));
    assert!(KernelFamily::Portable.is_available(Caps::NONE));
  }

  #[test]
  fn default_is_reference() {
    assert_eq!(KernelFamily::default(), KernelFamily::Reference);
  }

  #[test]
  fn families_in_tier_complete() {
    // Verify all families are categorized
    let mut count: usize = 0;
    for tier in KernelTier::ALL {
      count = count.strict_add(KernelFamily::families_in_tier(tier).len());
    }
    // 2 (ref+portable) + 2 (hwcrc) + 5 (folding) + 4 (wide) = 13
    assert_eq!(count, 13);
  }

  #[test]
  fn current_arch_families_ordered_by_tier() {
    let families = KernelFamily::families_for_current_arch();
    // Verify descending tier order
    for window in families.windows(2) {
      assert!(
        window[0].tier() >= window[1].tier(),
        "{:?} should come after {:?}",
        window[1],
        window[0]
      );
    }
    // Verify portable and reference are always present
    assert!(families.contains(&KernelFamily::Portable));
    assert!(families.contains(&KernelFamily::Reference));
  }

  #[test]
  fn min_bytes_per_lane_values() {
    // Reference/Portable: no multi-stream
    assert_eq!(KernelFamily::Reference.min_bytes_per_lane(), usize::MAX);
    assert_eq!(KernelFamily::Portable.min_bytes_per_lane(), usize::MAX);

    // HW CRC: low values (memory-bound quickly)
    assert_eq!(KernelFamily::X86Crc32.min_bytes_per_lane(), 128);
    assert_eq!(KernelFamily::ArmCrc32.min_bytes_per_lane(), 128);

    // Folding tier: moderate values
    assert_eq!(KernelFamily::X86Pclmul.min_bytes_per_lane(), 256);
    assert_eq!(KernelFamily::ArmPmull.min_bytes_per_lane(), 256);

    // Wide tier: higher values
    assert_eq!(KernelFamily::X86Vpclmul.min_bytes_per_lane(), 512);
    assert_eq!(KernelFamily::ArmSve2Pmull.min_bytes_per_lane(), 512);

    // All SIMD families should have finite min_bytes_per_lane
    for &family in KernelFamily::families_for_current_arch() {
      if family.tier().is_simd() {
        assert!(
          family.min_bytes_per_lane() < usize::MAX,
          "{:?} should have finite min_bytes_per_lane",
          family
        );
      }
    }
  }
}
