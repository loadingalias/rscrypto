//! Static kernel name tables and selection helpers for CRC64.
//!
//! This module centralizes all CRC64 kernel names and dispatch logic to reduce
//! code duplication across architectures. The same pattern applies to both
//! kernel name selection (for introspection) and actual dispatch.
//!
//! # Design
//!
//! - Static arrays for kernel names per tier per architecture
//! - Delegates to `common::kernels` for stream-to-index mapping
//! - CRC64-specific dispatch functions generated via macro

/// Portable fallback kernel name.
pub use kernels::PORTABLE_SLICE16 as PORTABLE;
/// Re-export select_name from common kernels.
pub use kernels::select_name;

use crate::{common::kernels, dispatchers::Crc64Fn};

// Generate CRC64-specific dispatch functions using the common macro.
crate::define_crc_dispatch!(Crc64Fn, u64);

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Name Tables (per architecture)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
  use super::super::x86_64 as arch;
  use crate::dispatchers::Crc64Fn;

  /// PCLMUL kernel names: [1-way, 2-way, 4-way, 7-way]
  pub const PCLMUL_NAMES: &[&str] = &[
    "x86_64/pclmul",
    "x86_64/pclmul-2way",
    "x86_64/pclmul-4way",
    "x86_64/pclmul-7way",
  ];
  /// PCLMUL small buffer kernel name.
  pub const PCLMUL_SMALL: &str = "x86_64/pclmul-small";

  /// VPCLMUL kernel names: [1-way, 2-way, 4-way, 7-way]
  pub const VPCLMUL_NAMES: &[&str] = &[
    "x86_64/vpclmul",
    "x86_64/vpclmul-2way",
    "x86_64/vpclmul-4way",
    "x86_64/vpclmul-7way",
  ];

  // ─────────────────────────────────────────────────────────────────────────
  // CRC64-XZ Kernel Function Arrays
  // ─────────────────────────────────────────────────────────────────────────

  /// XZ PCLMUL kernels: [1-way, 2-way, 4-way, 7-way]
  pub const XZ_PCLMUL: [Crc64Fn; 4] = [
    arch::crc64_xz_pclmul_safe,
    arch::crc64_xz_pclmul_2way_safe,
    arch::crc64_xz_pclmul_4way_safe,
    arch::crc64_xz_pclmul_7way_safe,
  ];
  /// XZ PCLMUL small buffer kernel.
  pub const XZ_PCLMUL_SMALL: Crc64Fn = arch::crc64_xz_pclmul_small_safe;

  /// XZ VPCLMUL kernels: [1-way, 2-way, 4-way, 7-way]
  pub const XZ_VPCLMUL: [Crc64Fn; 4] = [
    arch::crc64_xz_vpclmul_safe,
    arch::crc64_xz_vpclmul_2way_safe,
    arch::crc64_xz_vpclmul_4way_safe,
    arch::crc64_xz_vpclmul_7way_safe,
  ];

  // ─────────────────────────────────────────────────────────────────────────
  // CRC64-NVME Kernel Function Arrays
  // ─────────────────────────────────────────────────────────────────────────

  /// NVME PCLMUL kernels: [1-way, 2-way, 4-way, 7-way]
  pub const NVME_PCLMUL: [Crc64Fn; 4] = [
    arch::crc64_nvme_pclmul_safe,
    arch::crc64_nvme_pclmul_2way_safe,
    arch::crc64_nvme_pclmul_4way_safe,
    arch::crc64_nvme_pclmul_7way_safe,
  ];
  /// NVME PCLMUL small buffer kernel.
  pub const NVME_PCLMUL_SMALL: Crc64Fn = arch::crc64_nvme_pclmul_small_safe;

  /// NVME VPCLMUL kernels: [1-way, 2-way, 4-way, 7-way]
  pub const NVME_VPCLMUL: [Crc64Fn; 4] = [
    arch::crc64_nvme_vpclmul_safe,
    arch::crc64_nvme_vpclmul_2way_safe,
    arch::crc64_nvme_vpclmul_4way_safe,
    arch::crc64_nvme_vpclmul_7way_safe,
  ];
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
  use super::super::aarch64 as arch;
  use crate::dispatchers::Crc64Fn;

  /// PMULL kernel names: [1-way, 2-way, 3-way]
  pub const PMULL_NAMES: &[&str] = &["aarch64/pmull", "aarch64/pmull-2way", "aarch64/pmull-3way"];
  /// PMULL small buffer kernel name.
  pub const PMULL_SMALL: &str = "aarch64/pmull-small";

  /// PMULL+EOR3 kernel names: [1-way, 2-way, 3-way]
  pub const PMULL_EOR3_NAMES: &[&str] = &[
    "aarch64/pmull-eor3",
    "aarch64/pmull-eor3-2way",
    "aarch64/pmull-eor3-3way",
  ];

  /// SVE2 PMULL kernel names: [1-way, 2-way, 3-way]
  pub const SVE2_PMULL_NAMES: &[&str] = &[
    "aarch64/sve2-pmull",
    "aarch64/sve2-pmull-2way",
    "aarch64/sve2-pmull-3way",
  ];
  /// SVE2 PMULL small buffer kernel name.
  pub const SVE2_PMULL_SMALL: &str = "aarch64/sve2-pmull-small";

  // ─────────────────────────────────────────────────────────────────────────
  // CRC64-XZ Kernel Function Arrays
  // Note: aarch64 only supports up to 3-way, slot 3 is duplicate of 3-way
  // ─────────────────────────────────────────────────────────────────────────

  /// XZ PMULL kernels: [1-way, 2-way, 3-way, 3-way(dup)]
  pub const XZ_PMULL: [Crc64Fn; 4] = [
    arch::crc64_xz_pmull_safe,
    arch::crc64_xz_pmull_2way_safe,
    arch::crc64_xz_pmull_3way_safe,
    arch::crc64_xz_pmull_3way_safe, // dup for index consistency
  ];
  /// XZ PMULL small buffer kernel.
  pub const XZ_PMULL_SMALL: Crc64Fn = arch::crc64_xz_pmull_small_safe;

  /// XZ PMULL+EOR3 kernels: [1-way, 2-way, 3-way, 3-way(dup)]
  pub const XZ_PMULL_EOR3: [Crc64Fn; 4] = [
    arch::crc64_xz_pmull_eor3_safe,
    arch::crc64_xz_pmull_eor3_2way_safe,
    arch::crc64_xz_pmull_eor3_3way_safe,
    arch::crc64_xz_pmull_eor3_3way_safe, // dup for index consistency
  ];

  /// XZ SVE2 PMULL kernels: [1-way, 2-way, 3-way, 3-way(dup)]
  pub const XZ_SVE2_PMULL: [Crc64Fn; 4] = [
    arch::crc64_xz_sve2_pmull_safe,
    arch::crc64_xz_sve2_pmull_2way_safe,
    arch::crc64_xz_sve2_pmull_3way_safe,
    arch::crc64_xz_sve2_pmull_3way_safe, // dup for index consistency
  ];
  /// XZ SVE2 PMULL small buffer kernel.
  pub const XZ_SVE2_PMULL_SMALL: Crc64Fn = arch::crc64_xz_sve2_pmull_small_safe;

  // ─────────────────────────────────────────────────────────────────────────
  // CRC64-NVME Kernel Function Arrays
  // ─────────────────────────────────────────────────────────────────────────

  /// NVME PMULL kernels: [1-way, 2-way, 3-way, 3-way(dup)]
  pub const NVME_PMULL: [Crc64Fn; 4] = [
    arch::crc64_nvme_pmull_safe,
    arch::crc64_nvme_pmull_2way_safe,
    arch::crc64_nvme_pmull_3way_safe,
    arch::crc64_nvme_pmull_3way_safe, // dup for index consistency
  ];
  /// NVME PMULL small buffer kernel.
  pub const NVME_PMULL_SMALL: Crc64Fn = arch::crc64_nvme_pmull_small_safe;

  /// NVME PMULL+EOR3 kernels: [1-way, 2-way, 3-way, 3-way(dup)]
  pub const NVME_PMULL_EOR3: [Crc64Fn; 4] = [
    arch::crc64_nvme_pmull_eor3_safe,
    arch::crc64_nvme_pmull_eor3_2way_safe,
    arch::crc64_nvme_pmull_eor3_3way_safe,
    arch::crc64_nvme_pmull_eor3_3way_safe, // dup for index consistency
  ];

  /// NVME SVE2 PMULL kernels: [1-way, 2-way, 3-way, 3-way(dup)]
  pub const NVME_SVE2_PMULL: [Crc64Fn; 4] = [
    arch::crc64_nvme_sve2_pmull_safe,
    arch::crc64_nvme_sve2_pmull_2way_safe,
    arch::crc64_nvme_sve2_pmull_3way_safe,
    arch::crc64_nvme_sve2_pmull_3way_safe, // dup for index consistency
  ];
  /// NVME SVE2 PMULL small buffer kernel.
  pub const NVME_SVE2_PMULL_SMALL: Crc64Fn = arch::crc64_nvme_sve2_pmull_small_safe;
}

#[cfg(target_arch = "powerpc64")]
pub mod powerpc64 {
  use super::super::powerpc64 as arch;
  use crate::dispatchers::Crc64Fn;

  /// VPMSUM kernel names: [1-way, 2-way, 4-way, 8-way]
  pub const VPMSUM_NAMES: &[&str] = &[
    "powerpc64/vpmsum",
    "powerpc64/vpmsum-2way",
    "powerpc64/vpmsum-4way",
    "powerpc64/vpmsum-8way",
  ];

  // ─────────────────────────────────────────────────────────────────────────
  // CRC64-XZ Kernel Function Arrays
  // ─────────────────────────────────────────────────────────────────────────

  /// XZ VPMSUM kernels: [1-way, 2-way, 4-way, 8-way]
  pub const XZ_VPMSUM: [Crc64Fn; 4] = [
    arch::crc64_xz_vpmsum_safe,
    arch::crc64_xz_vpmsum_2way_safe,
    arch::crc64_xz_vpmsum_4way_safe,
    arch::crc64_xz_vpmsum_8way_safe,
  ];

  // ─────────────────────────────────────────────────────────────────────────
  // CRC64-NVME Kernel Function Arrays
  // ─────────────────────────────────────────────────────────────────────────

  /// NVME VPMSUM kernels: [1-way, 2-way, 4-way, 8-way]
  pub const NVME_VPMSUM: [Crc64Fn; 4] = [
    arch::crc64_nvme_vpmsum_safe,
    arch::crc64_nvme_vpmsum_2way_safe,
    arch::crc64_nvme_vpmsum_4way_safe,
    arch::crc64_nvme_vpmsum_8way_safe,
  ];
}

#[cfg(target_arch = "s390x")]
pub mod s390x {
  use super::super::s390x as arch;
  use crate::dispatchers::Crc64Fn;

  /// VGFM kernel names: [1-way, 2-way, 4-way]
  pub const VGFM_NAMES: &[&str] = &["s390x/vgfm", "s390x/vgfm-2way", "s390x/vgfm-4way"];

  // ─────────────────────────────────────────────────────────────────────────
  // CRC64-XZ Kernel Function Arrays
  // Note: s390x only supports up to 4-way, slot 3 is duplicate of 4-way
  // ─────────────────────────────────────────────────────────────────────────

  /// XZ VGFM kernels: [1-way, 2-way, 4-way, 4-way(dup)]
  pub const XZ_VGFM: [Crc64Fn; 4] = [
    arch::crc64_xz_vgfm_safe,
    arch::crc64_xz_vgfm_2way_safe,
    arch::crc64_xz_vgfm_4way_safe,
    arch::crc64_xz_vgfm_4way_safe, // dup for index consistency
  ];

  // ─────────────────────────────────────────────────────────────────────────
  // CRC64-NVME Kernel Function Arrays
  // ─────────────────────────────────────────────────────────────────────────

  /// NVME VGFM kernels: [1-way, 2-way, 4-way, 4-way(dup)]
  pub const NVME_VGFM: [Crc64Fn; 4] = [
    arch::crc64_nvme_vgfm_safe,
    arch::crc64_nvme_vgfm_2way_safe,
    arch::crc64_nvme_vgfm_4way_safe,
    arch::crc64_nvme_vgfm_4way_safe, // dup for index consistency
  ];
}

#[cfg(target_arch = "riscv64")]
pub mod riscv64 {
  use super::super::riscv64 as arch;
  use crate::dispatchers::Crc64Fn;

  /// Zbc kernel names: [1-way, 2-way, 4-way]
  pub const ZBC_NAMES: &[&str] = &["riscv64/zbc", "riscv64/zbc-2way", "riscv64/zbc-4way"];

  /// Zvbc kernel names: [1-way, 2-way, 4-way]
  pub const ZVBC_NAMES: &[&str] = &["riscv64/zvbc", "riscv64/zvbc-2way", "riscv64/zvbc-4way"];

  // ─────────────────────────────────────────────────────────────────────────
  // CRC64-XZ Kernel Function Arrays
  // Note: riscv64 only supports up to 4-way, slot 3 is duplicate of 4-way
  // ─────────────────────────────────────────────────────────────────────────

  /// XZ Zbc kernels: [1-way, 2-way, 4-way, 4-way(dup)]
  pub const XZ_ZBC: [Crc64Fn; 4] = [
    arch::crc64_xz_zbc_safe,
    arch::crc64_xz_zbc_2way_safe,
    arch::crc64_xz_zbc_4way_safe,
    arch::crc64_xz_zbc_4way_safe, // dup for index consistency
  ];

  /// XZ Zvbc kernels: [1-way, 2-way, 4-way, 4-way(dup)]
  pub const XZ_ZVBC: [Crc64Fn; 4] = [
    arch::crc64_xz_zvbc_safe,
    arch::crc64_xz_zvbc_2way_safe,
    arch::crc64_xz_zvbc_4way_safe,
    arch::crc64_xz_zvbc_4way_safe, // dup for index consistency
  ];

  // ─────────────────────────────────────────────────────────────────────────
  // CRC64-NVME Kernel Function Arrays
  // ─────────────────────────────────────────────────────────────────────────

  /// NVME Zbc kernels: [1-way, 2-way, 4-way, 4-way(dup)]
  pub const NVME_ZBC: [Crc64Fn; 4] = [
    arch::crc64_nvme_zbc_safe,
    arch::crc64_nvme_zbc_2way_safe,
    arch::crc64_nvme_zbc_4way_safe,
    arch::crc64_nvme_zbc_4way_safe, // dup for index consistency
  ];

  /// NVME Zvbc kernels: [1-way, 2-way, 4-way, 4-way(dup)]
  pub const NVME_ZVBC: [Crc64Fn; 4] = [
    arch::crc64_nvme_zvbc_safe,
    arch::crc64_nvme_zvbc_2way_safe,
    arch::crc64_nvme_zvbc_4way_safe,
    arch::crc64_nvme_zvbc_4way_safe, // dup for index consistency
  ];
}
