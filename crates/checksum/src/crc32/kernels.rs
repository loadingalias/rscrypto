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
pub mod x86_64 {
  /// Hardware CRC32C instruction kernel name.
  pub const SSE42_CRC32C: &str = "x86_64/sse4.2-crc32c";

  /// PCLMUL 1-way kernel name.
  pub const PCLMUL: &str = "x86_64/pclmul";

  /// VPCLMUL 1-way kernel name.
  pub const VPCLMUL: &str = "x86_64/vpclmul";
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
