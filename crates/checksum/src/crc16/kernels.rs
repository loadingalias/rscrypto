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
//! - Tier 3 (Folding): PCLMUL (x86_64), PMULL (aarch64), VPMSUM (powerpc64), VGFM (s390x), Zbc
//!   (riscv64)
//! - Tier 4 (Wide): VPCLMUL (x86_64 AVX-512), Zvbc (riscv64)

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
  /// PCLMUL kernel names: [1-way, 2-way, 4-way, 7-way, 8-way].
  pub const PCLMUL_NAMES: &[&str] = &[
    "x86_64/pclmul",
    "x86_64/pclmul-2way",
    "x86_64/pclmul-4way",
    "x86_64/pclmul-7way",
    "x86_64/pclmul-8way",
  ];

  /// VPCLMUL kernel name (AVX-512 VPCLMULQDQ).
  pub const VPCLMUL: &str = "x86_64/vpclmul";
  /// VPCLMUL kernel names: [1-way, 2-way, 4-way, 7-way, 8-way].
  pub const VPCLMUL_NAMES: &[&str] = &[
    "x86_64/vpclmul",
    "x86_64/vpclmul-2way",
    "x86_64/vpclmul-4way",
    "x86_64/vpclmul-7way",
    "x86_64/vpclmul-8way",
  ];

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16/CCITT Kernel Functions
  // ─────────────────────────────────────────────────────────────────────────

  /// CCITT PCLMUL kernel.
  #[allow(dead_code)] // Used by bench + policy dispatch.
  pub const CCITT_PCLMUL: [Crc16Fn; 5] = [
    arch::crc16_ccitt_pclmul_safe,
    arch::crc16_ccitt_pclmul_2way_safe,
    arch::crc16_ccitt_pclmul_4way_safe,
    arch::crc16_ccitt_pclmul_7way_safe,
    arch::crc16_ccitt_pclmul_8way_safe,
  ];

  /// CCITT VPCLMUL kernel.
  #[allow(dead_code)] // Used by bench + policy dispatch.
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
  #[allow(dead_code)] // Used by bench + policy dispatch.
  pub const IBM_PCLMUL: [Crc16Fn; 5] = [
    arch::crc16_ibm_pclmul_safe,
    arch::crc16_ibm_pclmul_2way_safe,
    arch::crc16_ibm_pclmul_4way_safe,
    arch::crc16_ibm_pclmul_7way_safe,
    arch::crc16_ibm_pclmul_8way_safe,
  ];

  /// IBM VPCLMUL kernel.
  #[allow(dead_code)] // Used by bench + policy dispatch.
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
  use crate::dispatchers::Crc16Fn;

  /// PMULL kernel name (NEON carryless multiply).
  pub const PMULL: &str = "aarch64/pmull";
  /// PMULL kernel names: [1-way, 2-way, 3-way].
  pub const PMULL_NAMES: &[&str] = &["aarch64/pmull", "aarch64/pmull-2way", "aarch64/pmull-3way"];

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16/CCITT Kernel Functions
  // ─────────────────────────────────────────────────────────────────────────

  /// CCITT PMULL kernels: [1-way, 2-way, 3-way, 3-way(dup), 3-way(dup)].
  #[allow(dead_code)] // Used by bench + future stream dispatch.
  pub const CCITT_PMULL: [Crc16Fn; 5] = [
    arch::crc16_ccitt_pmull_safe,
    arch::crc16_ccitt_pmull_2way_safe,
    arch::crc16_ccitt_pmull_3way_safe,
    arch::crc16_ccitt_pmull_3way_safe, // dup for index consistency
    arch::crc16_ccitt_pmull_3way_safe, // dup for index consistency
  ];

  // ─────────────────────────────────────────────────────────────────────────
  // CRC-16/IBM Kernel Functions
  // ─────────────────────────────────────────────────────────────────────────

  /// IBM PMULL kernels: [1-way, 2-way, 3-way, 3-way(dup), 3-way(dup)].
  #[allow(dead_code)] // Used by bench + future stream dispatch.
  pub const IBM_PMULL: [Crc16Fn; 5] = [
    arch::crc16_ibm_pmull_safe,
    arch::crc16_ibm_pmull_2way_safe,
    arch::crc16_ibm_pmull_3way_safe,
    arch::crc16_ibm_pmull_3way_safe, // dup for index consistency
    arch::crc16_ibm_pmull_3way_safe, // dup for index consistency
  ];
}

#[cfg(target_arch = "powerpc64")]
pub mod powerpc64 {
  use super::super::powerpc64 as arch;
  use crate::dispatchers::Crc16Fn;

  /// VPMSUM kernel name (POWER8+ carryless multiply).
  pub const VPMSUM: &str = "powerpc64/vpmsum";

  /// CCITT VPMSUM kernel.
  #[allow(dead_code)] // Exposed for kernel ladder consistency; dispatch uses wrapper fn
  pub const CCITT_VPMSUM: Crc16Fn = arch::crc16_ccitt_vpmsum_safe;

  /// IBM VPMSUM kernel.
  #[allow(dead_code)] // Exposed for kernel ladder consistency; dispatch uses wrapper fn
  pub const IBM_VPMSUM: Crc16Fn = arch::crc16_ibm_vpmsum_safe;
}

#[cfg(target_arch = "s390x")]
pub mod s390x {
  use super::super::s390x as arch;
  use crate::dispatchers::Crc16Fn;

  /// VGFM kernel name (s390x vector Galois field multiply).
  pub const VGFM: &str = "s390x/vgfm";

  /// CCITT VGFM kernel.
  #[allow(dead_code)] // Exposed for kernel ladder consistency; dispatch uses wrapper fn
  pub const CCITT_VGFM: Crc16Fn = arch::crc16_ccitt_vgfm_safe;

  /// IBM VGFM kernel.
  #[allow(dead_code)] // Exposed for kernel ladder consistency; dispatch uses wrapper fn
  pub const IBM_VGFM: Crc16Fn = arch::crc16_ibm_vgfm_safe;
}

#[cfg(target_arch = "riscv64")]
pub mod riscv64 {
  use super::super::riscv64 as arch;
  use crate::dispatchers::Crc16Fn;

  /// Zbc kernel name (scalar carryless multiply).
  pub const ZBC: &str = "riscv64/zbc";

  /// Zvbc kernel name (vector carryless multiply).
  pub const ZVBC: &str = "riscv64/zvbc";

  /// CCITT Zbc kernel.
  #[allow(dead_code)] // Exposed for kernel ladder consistency; dispatch uses wrapper fn
  pub const CCITT_ZBC: Crc16Fn = arch::crc16_ccitt_zbc_safe;

  /// CCITT Zvbc kernel.
  #[allow(dead_code)] // Exposed for kernel ladder consistency; dispatch uses wrapper fn
  pub const CCITT_ZVBC: Crc16Fn = arch::crc16_ccitt_zvbc_safe;

  /// IBM Zbc kernel.
  #[allow(dead_code)] // Exposed for kernel ladder consistency; dispatch uses wrapper fn
  pub const IBM_ZBC: Crc16Fn = arch::crc16_ibm_zbc_safe;

  /// IBM Zvbc kernel.
  #[allow(dead_code)] // Exposed for kernel ladder consistency; dispatch uses wrapper fn
  pub const IBM_ZVBC: Crc16Fn = arch::crc16_ibm_zvbc_safe;
}
