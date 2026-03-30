//! Static kernel name tables and selection helpers for CRC-16.
//!
//! This module mirrors the CRC-64 layout: centralized kernel names for both
//! introspection and dispatch, with architecture-specific submodules for
//! SIMD kernel function references.
//!
//! # Kernel Tiers
//!
//! CRC-16 supports Tiers 0, 1, 3, and 4 (no HW CRC instructions exist):
//! - Tier 0 (Reference): Bitwise implementation
//! - Tier 1 (Portable): Slice-by-4/8 table lookup
//! - Tier 3 (Folding): PCLMUL (x86_64), PMULL (aarch64), VPMSUM (Power), VGFM (s390x), Zbc
//!   (riscv64)
//! - Tier 4 (Wide): VPCLMUL (x86_64 AVX-512), Zvbc (riscv64)

/// Reference (bitwise) kernel name.
pub use kernels::REFERENCE;

use crate::checksum::common::kernels;

/// Portable slice-by-8 kernel name.
pub const PORTABLE_SLICE8: &str = kernels::PORTABLE_SLICE8;

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Name Tables and Functions (per architecture)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
  use super::super::x86_64 as arch;
  use crate::checksum::dispatchers::Crc16Fn;

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16/CCITT Kernel Functions
  // ─────────────────────────────────────────────────────────────────────────

  /// CCITT PCLMUL kernel.
  pub const CCITT_PCLMUL: [Crc16Fn; 5] = [
    arch::crc16_ccitt_pclmul_safe,
    arch::crc16_ccitt_pclmul_2way_safe,
    arch::crc16_ccitt_pclmul_4way_safe,
    arch::crc16_ccitt_pclmul_7way_safe,
    arch::crc16_ccitt_pclmul_8way_safe,
  ];

  /// CCITT PCLMUL small-buffer kernel.
  pub const CCITT_PCLMUL_SMALL_KERNEL: Crc16Fn = arch::crc16_ccitt_pclmul_small_safe;

  /// CCITT VPCLMUL kernel.
  pub const CCITT_VPCLMUL: [Crc16Fn; 5] = [
    arch::crc16_ccitt_vpclmul_safe,
    arch::crc16_ccitt_vpclmul_2way_safe,
    arch::crc16_ccitt_vpclmul_4way_safe,
    arch::crc16_ccitt_vpclmul_7way_safe,
    arch::crc16_ccitt_vpclmul_8way_safe,
  ];

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16/IBM Kernel Functions
  // ─────────────────────────────────────────────────────────────────────────

  /// IBM PCLMUL kernel.
  pub const IBM_PCLMUL: [Crc16Fn; 5] = [
    arch::crc16_ibm_pclmul_safe,
    arch::crc16_ibm_pclmul_2way_safe,
    arch::crc16_ibm_pclmul_4way_safe,
    arch::crc16_ibm_pclmul_7way_safe,
    arch::crc16_ibm_pclmul_8way_safe,
  ];

  /// IBM PCLMUL small-buffer kernel.
  pub const IBM_PCLMUL_SMALL_KERNEL: Crc16Fn = arch::crc16_ibm_pclmul_small_safe;

  /// IBM VPCLMUL kernel.
  pub const IBM_VPCLMUL: [Crc16Fn; 5] = [
    arch::crc16_ibm_vpclmul_safe,
    arch::crc16_ibm_vpclmul_2way_safe,
    arch::crc16_ibm_vpclmul_4way_safe,
    arch::crc16_ibm_vpclmul_7way_safe,
    arch::crc16_ibm_vpclmul_8way_safe,
  ];
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
  use super::super::aarch64 as arch;
  use crate::checksum::dispatchers::Crc16Fn;

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16/CCITT Kernel Functions
  // ─────────────────────────────────────────────────────────────────────────

  /// CCITT PMULL kernels: [1-way, 2-way, 3-way, 3-way(dup), 3-way(dup)].
  pub const CCITT_PMULL: [Crc16Fn; 5] = [
    arch::crc16_ccitt_pmull_safe,
    arch::crc16_ccitt_pmull_2way_safe,
    arch::crc16_ccitt_pmull_3way_safe,
    arch::crc16_ccitt_pmull_3way_safe, // dup for index consistency
    arch::crc16_ccitt_pmull_3way_safe, // dup for index consistency
  ];

  /// CCITT PMULL+EOR3 kernels: [1-way, 2-way, 3-way, 3-way(dup), 3-way(dup)].
  #[cfg(any(target_os = "linux", target_os = "android"))]
  pub const CCITT_PMULL_EOR3: [Crc16Fn; 5] = [
    arch::crc16_ccitt_pmull_eor3_safe,
    arch::crc16_ccitt_pmull_eor3_2way_safe,
    arch::crc16_ccitt_pmull_eor3_3way_safe,
    arch::crc16_ccitt_pmull_eor3_3way_safe, // dup for index consistency
    arch::crc16_ccitt_pmull_eor3_3way_safe, // dup for index consistency
  ];

  /// CCITT PMULL small-buffer kernel.
  pub const CCITT_PMULL_SMALL_KERNEL: Crc16Fn = arch::crc16_ccitt_pmull_small_safe;

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16/IBM Kernel Functions
  // ─────────────────────────────────────────────────────────────────────────

  /// IBM PMULL kernels: [1-way, 2-way, 3-way, 3-way(dup), 3-way(dup)].
  pub const IBM_PMULL: [Crc16Fn; 5] = [
    arch::crc16_ibm_pmull_safe,
    arch::crc16_ibm_pmull_2way_safe,
    arch::crc16_ibm_pmull_3way_safe,
    arch::crc16_ibm_pmull_3way_safe, // dup for index consistency
    arch::crc16_ibm_pmull_3way_safe, // dup for index consistency
  ];

  /// IBM PMULL+EOR3 kernels: [1-way, 2-way, 3-way, 3-way(dup), 3-way(dup)].
  #[cfg(any(target_os = "linux", target_os = "android"))]
  pub const IBM_PMULL_EOR3: [Crc16Fn; 5] = [
    arch::crc16_ibm_pmull_eor3_safe,
    arch::crc16_ibm_pmull_eor3_2way_safe,
    arch::crc16_ibm_pmull_eor3_3way_safe,
    arch::crc16_ibm_pmull_eor3_3way_safe, // dup for index consistency
    arch::crc16_ibm_pmull_eor3_3way_safe, // dup for index consistency
  ];

  /// IBM PMULL small-buffer kernel.
  pub const IBM_PMULL_SMALL_KERNEL: Crc16Fn = arch::crc16_ibm_pmull_small_safe;
}

#[cfg(target_arch = "powerpc64")]
pub mod power {
  use super::super::power as arch;
  use crate::checksum::dispatchers::Crc16Fn;

  /// CCITT VPMSUM kernels: [1-way, 2-way, 4-way, 8-way, 8-way(dup)].
  pub const CCITT_VPMSUM: [Crc16Fn; 5] = [
    arch::crc16_ccitt_vpmsum_safe,
    arch::crc16_ccitt_vpmsum_2way_safe,
    arch::crc16_ccitt_vpmsum_4way_safe,
    arch::crc16_ccitt_vpmsum_8way_safe,
    arch::crc16_ccitt_vpmsum_8way_safe,
  ];

  /// IBM VPMSUM kernels: [1-way, 2-way, 4-way, 8-way, 8-way(dup)].
  pub const IBM_VPMSUM: [Crc16Fn; 5] = [
    arch::crc16_ibm_vpmsum_safe,
    arch::crc16_ibm_vpmsum_2way_safe,
    arch::crc16_ibm_vpmsum_4way_safe,
    arch::crc16_ibm_vpmsum_8way_safe,
    arch::crc16_ibm_vpmsum_8way_safe,
  ];
}

#[cfg(target_arch = "s390x")]
pub mod s390x {
  use super::super::s390x as arch;
  use crate::checksum::dispatchers::Crc16Fn;

  /// CCITT VGFM kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const CCITT_VGFM: [Crc16Fn; 5] = [
    arch::crc16_ccitt_vgfm_safe,
    arch::crc16_ccitt_vgfm_2way_safe,
    arch::crc16_ccitt_vgfm_4way_safe,
    arch::crc16_ccitt_vgfm_4way_safe,
    arch::crc16_ccitt_vgfm_4way_safe,
  ];

  /// IBM VGFM kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const IBM_VGFM: [Crc16Fn; 5] = [
    arch::crc16_ibm_vgfm_safe,
    arch::crc16_ibm_vgfm_2way_safe,
    arch::crc16_ibm_vgfm_4way_safe,
    arch::crc16_ibm_vgfm_4way_safe,
    arch::crc16_ibm_vgfm_4way_safe,
  ];
}

#[cfg(target_arch = "riscv64")]
pub mod riscv64 {
  use super::super::riscv64 as arch;
  use crate::checksum::dispatchers::Crc16Fn;

  /// Zbc kernel names: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const ZBC_NAMES: &[&str] = &[
    "riscv64/zbc",
    "riscv64/zbc-2way",
    "riscv64/zbc-4way",
    "riscv64/zbc-4way",
    "riscv64/zbc-4way",
  ];

  /// Zvbc kernel names: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const ZVBC_NAMES: &[&str] = &[
    "riscv64/zvbc",
    "riscv64/zvbc-2way",
    "riscv64/zvbc-4way",
    "riscv64/zvbc-4way",
    "riscv64/zvbc-4way",
  ];

  /// CCITT Zbc kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const CCITT_ZBC: [Crc16Fn; 5] = [
    arch::crc16_ccitt_zbc_safe,
    arch::crc16_ccitt_zbc_2way_safe,
    arch::crc16_ccitt_zbc_4way_safe,
    arch::crc16_ccitt_zbc_4way_safe,
    arch::crc16_ccitt_zbc_4way_safe,
  ];

  /// CCITT Zvbc kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const CCITT_ZVBC: [Crc16Fn; 5] = [
    arch::crc16_ccitt_zvbc_safe,
    arch::crc16_ccitt_zvbc_2way_safe,
    arch::crc16_ccitt_zvbc_4way_safe,
    arch::crc16_ccitt_zvbc_4way_safe,
    arch::crc16_ccitt_zvbc_4way_safe,
  ];

  /// IBM Zbc kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const IBM_ZBC: [Crc16Fn; 5] = [
    arch::crc16_ibm_zbc_safe,
    arch::crc16_ibm_zbc_2way_safe,
    arch::crc16_ibm_zbc_4way_safe,
    arch::crc16_ibm_zbc_4way_safe,
    arch::crc16_ibm_zbc_4way_safe,
  ];

  /// IBM Zvbc kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const IBM_ZVBC: [Crc16Fn; 5] = [
    arch::crc16_ibm_zvbc_safe,
    arch::crc16_ibm_zvbc_2way_safe,
    arch::crc16_ibm_zvbc_4way_safe,
    arch::crc16_ibm_zvbc_4way_safe,
    arch::crc16_ibm_zvbc_4way_safe,
  ];
}
