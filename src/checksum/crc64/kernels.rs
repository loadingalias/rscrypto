//! Static kernel name tables and selection helpers for CRC64.
//!
//! This module centralizes all CRC64 kernel names and dispatch logic to reduce
//! code duplication across architectures. The same pattern applies to both
//! kernel name selection (for introspection) and actual dispatch.
//!
//! # Kernel Tiers
//!
//! CRC-64 supports Tiers 0, 1, 3, and 4 (no HW CRC instructions exist):
//! - Tier 0 (Reference): Bitwise implementation
//! - Tier 1 (Portable): Slice-by-16 table lookup
//! - Tier 3 (Folding): PCLMUL (x86_64), PMULL (aarch64), VPMSUM (Power), VGFM (s390x), Zbc
//!   (riscv64)
//! - Tier 4 (Wide): VPCLMUL (x86_64), PMULL+EOR3/SVE2 (aarch64), Zvbc (riscv64)
//!
//! # Design
//!
//! - Static arrays for kernel names per tier per architecture
//! - Delegates to `common::kernels` for stream-to-index mapping
//! - CRC64-specific dispatch functions generated via macro

/// Portable fallback kernel name.
pub(in crate::checksum) use kernels::PORTABLE_SLICE16 as PORTABLE;
/// Reference (bitwise) kernel name - always available for force mode.
pub(in crate::checksum) use kernels::REFERENCE;

use crate::checksum::common::kernels;

// Kernel Name Tables (per architecture)

#[cfg(target_arch = "x86_64")]
pub(in crate::checksum) mod x86_64 {
  use super::super::x86_64 as arch;
  use crate::checksum::dispatchers::Crc64Fn;

  // CRC64-XZ Kernel Function Arrays

  /// XZ PCLMUL kernels: [1-way, 2-way, 4-way, 7-way, 8-way]
  pub(in crate::checksum) const XZ_PCLMUL: [Crc64Fn; 5] = [
    arch::crc64_xz_pclmul_safe,
    arch::crc64_xz_pclmul_2way_safe,
    arch::crc64_xz_pclmul_4way_safe,
    arch::crc64_xz_pclmul_7way_safe,
    arch::crc64_xz_pclmul_8way_safe,
  ];
  /// XZ PCLMUL small buffer kernel.
  pub(in crate::checksum) const XZ_PCLMUL_SMALL: Crc64Fn = arch::crc64_xz_pclmul_small_safe;

  /// XZ VPCLMUL kernels: [1-way, 2-way, 4-way, 7-way, 8-way]
  pub(in crate::checksum) const XZ_VPCLMUL: [Crc64Fn; 5] = [
    arch::crc64_xz_vpclmul_safe,
    arch::crc64_xz_vpclmul_2way_safe,
    arch::crc64_xz_vpclmul_4way_safe,
    arch::crc64_xz_vpclmul_7way_safe,
    arch::crc64_xz_vpclmul_8way_safe,
  ];
  /// XZ VPCLMUL 4×512-bit kernel.
  pub(in crate::checksum) const XZ_VPCLMUL_4X512: Crc64Fn = arch::crc64_xz_vpclmul_4x512_safe;

  // CRC64-NVME Kernel Function Arrays

  /// NVME PCLMUL kernels: [1-way, 2-way, 4-way, 7-way, 8-way]
  pub(in crate::checksum) const NVME_PCLMUL: [Crc64Fn; 5] = [
    arch::crc64_nvme_pclmul_safe,
    arch::crc64_nvme_pclmul_2way_safe,
    arch::crc64_nvme_pclmul_4way_safe,
    arch::crc64_nvme_pclmul_7way_safe,
    arch::crc64_nvme_pclmul_8way_safe,
  ];
  /// NVME PCLMUL small buffer kernel.
  pub(in crate::checksum) const NVME_PCLMUL_SMALL: Crc64Fn = arch::crc64_nvme_pclmul_small_safe;

  /// NVME VPCLMUL kernels: [1-way, 2-way, 4-way, 7-way, 8-way]
  pub(in crate::checksum) const NVME_VPCLMUL: [Crc64Fn; 5] = [
    arch::crc64_nvme_vpclmul_safe,
    arch::crc64_nvme_vpclmul_2way_safe,
    arch::crc64_nvme_vpclmul_4way_safe,
    arch::crc64_nvme_vpclmul_7way_safe,
    arch::crc64_nvme_vpclmul_8way_safe,
  ];
}

#[cfg(target_arch = "aarch64")]
pub(in crate::checksum) mod aarch64 {

  use super::super::aarch64 as arch;
  use crate::checksum::dispatchers::Crc64Fn;

  // CRC64-XZ Kernel Function Arrays
  // Note: aarch64 only supports up to 3-way, slots 3-4 are duplicates

  /// XZ PMULL kernels: [1-way, 2-way, 3-way, 3-way(dup), 3-way(dup)]
  pub(in crate::checksum) const XZ_PMULL: [Crc64Fn; 5] = [
    arch::crc64_xz_pmull_safe,
    arch::crc64_xz_pmull_2way_safe,
    arch::crc64_xz_pmull_3way_safe,
    arch::crc64_xz_pmull_3way_safe, // dup for index consistency
    arch::crc64_xz_pmull_3way_safe, // dup for index consistency
  ];
  /// XZ PMULL small buffer kernel.
  pub(in crate::checksum) const XZ_PMULL_SMALL: Crc64Fn = arch::crc64_xz_pmull_small_safe;

  /// XZ PMULL+EOR3 kernels: [1-way, 2-way, 3-way, 3-way(dup), 3-way(dup)]
  pub(in crate::checksum) const XZ_PMULL_EOR3: [Crc64Fn; 5] = [
    arch::crc64_xz_pmull_eor3_safe,
    arch::crc64_xz_pmull_eor3_2way_safe,
    arch::crc64_xz_pmull_eor3_3way_safe,
    arch::crc64_xz_pmull_eor3_3way_safe, // dup for index consistency
    arch::crc64_xz_pmull_eor3_3way_safe, // dup for index consistency
  ];

  #[cfg(feature = "std")]
  /// XZ SVE2 PMULL kernels: [1-way, 2-way, 3-way, 3-way(dup), 3-way(dup)]
  pub(in crate::checksum) const XZ_SVE2_PMULL: [Crc64Fn; 5] = [
    arch::crc64_xz_sve2_pmull_safe,
    arch::crc64_xz_sve2_pmull_2way_safe,
    arch::crc64_xz_sve2_pmull_3way_safe,
    arch::crc64_xz_sve2_pmull_3way_safe, // dup for index consistency
    arch::crc64_xz_sve2_pmull_3way_safe, // dup for index consistency
  ];
  #[cfg(feature = "std")]
  /// XZ SVE2 PMULL small buffer kernel.
  pub(in crate::checksum) const XZ_SVE2_PMULL_SMALL: Crc64Fn = arch::crc64_xz_sve2_pmull_small_safe;

  // CRC64-NVME Kernel Function Arrays

  /// NVME PMULL kernels: [1-way, 2-way, 3-way, 3-way(dup), 3-way(dup)]
  pub(in crate::checksum) const NVME_PMULL: [Crc64Fn; 5] = [
    arch::crc64_nvme_pmull_safe,
    arch::crc64_nvme_pmull_2way_safe,
    arch::crc64_nvme_pmull_3way_safe,
    arch::crc64_nvme_pmull_3way_safe, // dup for index consistency
    arch::crc64_nvme_pmull_3way_safe, // dup for index consistency
  ];
  /// NVME PMULL small buffer kernel.
  pub(in crate::checksum) const NVME_PMULL_SMALL: Crc64Fn = arch::crc64_nvme_pmull_small_safe;

  /// NVME PMULL+EOR3 kernels: [1-way, 2-way, 3-way, 3-way(dup), 3-way(dup)]
  pub(in crate::checksum) const NVME_PMULL_EOR3: [Crc64Fn; 5] = [
    arch::crc64_nvme_pmull_eor3_safe,
    arch::crc64_nvme_pmull_eor3_2way_safe,
    arch::crc64_nvme_pmull_eor3_3way_safe,
    arch::crc64_nvme_pmull_eor3_3way_safe, // dup for index consistency
    arch::crc64_nvme_pmull_eor3_3way_safe, // dup for index consistency
  ];

  #[cfg(feature = "std")]
  /// NVME SVE2 PMULL kernels: [1-way, 2-way, 3-way, 3-way(dup), 3-way(dup)]
  pub(in crate::checksum) const NVME_SVE2_PMULL: [Crc64Fn; 5] = [
    arch::crc64_nvme_sve2_pmull_safe,
    arch::crc64_nvme_sve2_pmull_2way_safe,
    arch::crc64_nvme_sve2_pmull_3way_safe,
    arch::crc64_nvme_sve2_pmull_3way_safe, // dup for index consistency
    arch::crc64_nvme_sve2_pmull_3way_safe, // dup for index consistency
  ];
  #[cfg(feature = "std")]
  /// NVME SVE2 PMULL small buffer kernel.
  pub(in crate::checksum) const NVME_SVE2_PMULL_SMALL: Crc64Fn = arch::crc64_nvme_sve2_pmull_small_safe;
}

#[cfg(target_arch = "powerpc64")]
pub(in crate::checksum) mod power {
  use super::super::power as arch;
  use crate::checksum::dispatchers::Crc64Fn;

  // CRC64-XZ Kernel Function Arrays

  /// XZ VPMSUM kernels: [1-way, 2-way, 4-way, 8-way, 8-way(dup)]
  pub(in crate::checksum) const XZ_VPMSUM: [Crc64Fn; 5] = [
    arch::crc64_xz_vpmsum_safe,
    arch::crc64_xz_vpmsum_2way_safe,
    arch::crc64_xz_vpmsum_4way_safe,
    arch::crc64_xz_vpmsum_8way_safe,
    arch::crc64_xz_vpmsum_8way_safe, // dup for index consistency
  ];

  // CRC64-NVME Kernel Function Arrays

  /// NVME VPMSUM kernels: [1-way, 2-way, 4-way, 8-way, 8-way(dup)]
  pub(in crate::checksum) const NVME_VPMSUM: [Crc64Fn; 5] = [
    arch::crc64_nvme_vpmsum_safe,
    arch::crc64_nvme_vpmsum_2way_safe,
    arch::crc64_nvme_vpmsum_4way_safe,
    arch::crc64_nvme_vpmsum_8way_safe,
    arch::crc64_nvme_vpmsum_8way_safe, // dup for index consistency
  ];
}

#[cfg(target_arch = "s390x")]
pub(in crate::checksum) mod s390x {
  use super::super::s390x as arch;
  use crate::checksum::dispatchers::Crc64Fn;

  // CRC64-XZ Kernel Function Arrays
  // Note: s390x only supports up to 4-way, slots 3-4 are duplicates

  /// XZ VGFM kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)]
  pub(in crate::checksum) const XZ_VGFM: [Crc64Fn; 5] = [
    arch::crc64_xz_vgfm_safe,
    arch::crc64_xz_vgfm_2way_safe,
    arch::crc64_xz_vgfm_4way_safe,
    arch::crc64_xz_vgfm_4way_safe, // dup for index consistency
    arch::crc64_xz_vgfm_4way_safe, // dup for index consistency
  ];

  // CRC64-NVME Kernel Function Arrays

  /// NVME VGFM kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)]
  pub(in crate::checksum) const NVME_VGFM: [Crc64Fn; 5] = [
    arch::crc64_nvme_vgfm_safe,
    arch::crc64_nvme_vgfm_2way_safe,
    arch::crc64_nvme_vgfm_4way_safe,
    arch::crc64_nvme_vgfm_4way_safe, // dup for index consistency
    arch::crc64_nvme_vgfm_4way_safe, // dup for index consistency
  ];
}
