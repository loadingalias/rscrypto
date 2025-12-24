//! Static kernel name tables and selection helpers for CRC-32.
//!
//! This module centralizes all CRC-32 kernel names to reduce
//! code duplication across architectures.

/// Portable fallback kernel name.
pub use kernels::PORTABLE_SLICE16 as PORTABLE;

use crate::common::kernels;

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Name Tables (per architecture)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // Kernels wired up via dispatcher
pub mod x86_64 {
  use super::super::x86_64 as arch;
  use crate::dispatchers::Crc32Fn;

  /// Hardware CRC32C instruction kernel name.
  pub const SSE42_CRC32C: &str = "x86_64/sse4.2-crc32c";

  /// PCLMUL kernel names: [1-way, 2-way, 4-way, 7-way]
  pub const PCLMUL_NAMES: &[&str] = &[
    "x86_64/pclmul",
    "x86_64/pclmul-2way",
    "x86_64/pclmul-4way",
    "x86_64/pclmul-7way",
  ];

  /// PCLMUL 1-way kernel name (for backward compatibility).
  pub const PCLMUL: &str = "x86_64/pclmul";

  /// VPCLMUL kernel names: [1-way, 2-way, 4-way, 7-way]
  pub const VPCLMUL_NAMES: &[&str] = &[
    "x86_64/vpclmul",
    "x86_64/vpclmul-2way",
    "x86_64/vpclmul-4way",
    "x86_64/vpclmul-7way",
  ];

  /// VPCLMUL 1-way kernel name (for backward compatibility).
  pub const VPCLMUL: &str = "x86_64/vpclmul";

  // ─────────────────────────────────────────────────────────────────────────
  // CRC32-IEEE Kernel Function Arrays
  // ─────────────────────────────────────────────────────────────────────────

  /// IEEE PCLMUL kernels: [1-way, 2-way, 4-way, 7-way]
  pub const IEEE_PCLMUL: [Crc32Fn; 4] = [
    arch::crc32_ieee_pclmul_safe,
    arch::crc32_ieee_pclmul_2way_safe,
    arch::crc32_ieee_pclmul_4way_safe,
    arch::crc32_ieee_pclmul_7way_safe,
  ];

  /// IEEE VPCLMUL kernels: [1-way, 2-way, 4-way, 7-way]
  pub const IEEE_VPCLMUL: [Crc32Fn; 4] = [
    arch::crc32_ieee_vpclmul_safe,
    arch::crc32_ieee_vpclmul_2way_safe,
    arch::crc32_ieee_vpclmul_4way_safe,
    arch::crc32_ieee_vpclmul_7way_safe,
  ];

  // ─────────────────────────────────────────────────────────────────────────
  // CRC32C Kernel Function Arrays
  // ─────────────────────────────────────────────────────────────────────────

  /// CRC32C SSE4.2 kernel.
  pub const CRC32C_SSE42: Crc32Fn = arch::crc32c_sse42_safe;

  /// CRC32C PCLMUL kernels: [1-way (fusion), 2-way, 4-way, 7-way]
  pub const CRC32C_PCLMUL: [Crc32Fn; 4] = [
    arch::crc32c_pclmul_safe, // 1-way uses fusion kernel
    arch::crc32c_pclmul_2way_safe,
    arch::crc32c_pclmul_4way_safe,
    arch::crc32c_pclmul_7way_safe,
  ];

  /// CRC32C VPCLMUL kernels: [1-way, 2-way, 4-way, 7-way]
  pub const CRC32C_VPCLMUL: [Crc32Fn; 4] = [
    arch::crc32c_vpclmul_safe,
    arch::crc32c_vpclmul_2way_safe,
    arch::crc32c_vpclmul_4way_safe,
    arch::crc32c_vpclmul_7way_safe,
  ];
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
  /// Hardware CRC32C instruction kernel name.
  pub const ARM_CRC32C: &str = "aarch64/crc32c";
  /// Hardware CRC32 instruction kernel name.
  #[allow(dead_code)]
  pub const ARM_CRC32: &str = "aarch64/crc32";

  /// PMULL 1-way kernel name.
  pub const PMULL: &str = "aarch64/pmull";

  /// PMULL+EOR3 1-way kernel name.
  pub const PMULL_EOR3: &str = "aarch64/pmull-eor3";
}
