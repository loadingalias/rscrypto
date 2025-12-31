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
//! - Tier 3 (Folding): PCLMUL (x86_64), PMULL (aarch64)
//! - Tier 4 (Wide): VPCLMUL (x86_64 AVX-512)

/// Reference (bitwise) kernel name.
pub use kernels::REFERENCE;

use crate::common::kernels;
// CRC-16 currently uses single-lane SIMD (no multi-stream support), so the
// dispatch macro is not used. It's included here for structural consistency
// with CRC-64 and to enable future multi-stream support.
#[cfg(any())] // Disabled until multi-stream is implemented
use crate::dispatchers::Crc16Fn;
#[cfg(any())]
crate::define_crc_dispatch!(Crc16Fn, u16);

/// Portable slice-by-4 kernel name.
pub const PORTABLE_SLICE4: &str = kernels::PORTABLE_SLICE4;
/// Portable slice-by-8 kernel name.
pub const PORTABLE_SLICE8: &str = kernels::PORTABLE_SLICE8;
/// Portable auto-selection kernel name (slice4 vs slice8 by length).
#[allow(dead_code)] // Reserved for introspection
pub const PORTABLE_AUTO: &str = "portable/auto";

/// Portable kernel name table (ordered by increasing work per byte).
#[allow(dead_code)]
pub const PORTABLE_NAMES: &[&str] = &[PORTABLE_SLICE4, PORTABLE_SLICE8];

#[inline]
#[must_use]
pub const fn portable_name_for_len(len: usize, slice4_to_slice8: usize) -> &'static str {
  if len < slice4_to_slice8 {
    PORTABLE_SLICE4
  } else {
    PORTABLE_SLICE8
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Name Tables and Functions (per architecture)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
  use super::super::x86_64 as arch;
  use crate::dispatchers::Crc16Fn;

  /// PCLMUL kernel name (SSE4.2 + PCLMULQDQ).
  pub const PCLMUL: &str = "x86_64/pclmul";

  /// VPCLMUL kernel name (AVX-512 VPCLMULQDQ).
  pub const VPCLMUL: &str = "x86_64/vpclmul";

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16/CCITT Kernel Functions
  // ─────────────────────────────────────────────────────────────────────────

  /// CCITT PCLMUL kernel.
  #[allow(dead_code)] // Exposed for kernel ladder consistency; dispatch uses wrapper fn
  pub const CCITT_PCLMUL: Crc16Fn = arch::crc16_ccitt_pclmul_safe;

  /// CCITT VPCLMUL kernel.
  #[allow(dead_code)] // Exposed for kernel ladder consistency; dispatch uses wrapper fn
  pub const CCITT_VPCLMUL: Crc16Fn = arch::crc16_ccitt_vpclmul_safe;

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16/IBM Kernel Functions
  // ─────────────────────────────────────────────────────────────────────────

  /// IBM PCLMUL kernel.
  #[allow(dead_code)] // Exposed for kernel ladder consistency; dispatch uses wrapper fn
  pub const IBM_PCLMUL: Crc16Fn = arch::crc16_ibm_pclmul_safe;

  /// IBM VPCLMUL kernel.
  #[allow(dead_code)] // Exposed for kernel ladder consistency; dispatch uses wrapper fn
  pub const IBM_VPCLMUL: Crc16Fn = arch::crc16_ibm_vpclmul_safe;
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
  use super::super::aarch64 as arch;
  use crate::dispatchers::Crc16Fn;

  /// PMULL kernel name (NEON carryless multiply).
  pub const PMULL: &str = "aarch64/pmull";

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16/CCITT Kernel Functions
  // ─────────────────────────────────────────────────────────────────────────

  /// CCITT PMULL kernel.
  #[allow(dead_code)] // Exposed for kernel ladder consistency; dispatch uses wrapper fn
  pub const CCITT_PMULL: Crc16Fn = arch::crc16_ccitt_pmull_safe;

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16/IBM Kernel Functions
  // ─────────────────────────────────────────────────────────────────────────

  /// IBM PMULL kernel.
  #[allow(dead_code)] // Exposed for kernel ladder consistency; dispatch uses wrapper fn
  pub const IBM_PMULL: Crc16Fn = arch::crc16_ibm_pmull_safe;
}
