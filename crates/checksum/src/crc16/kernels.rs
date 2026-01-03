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
  /// PCLMUL small-buffer kernel name.
  pub const PCLMUL_SMALL: &str = "x86_64/pclmul-small";
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

  /// CCITT PCLMUL small-buffer kernel.
  #[allow(dead_code)] // Used by bench + policy dispatch.
  pub const CCITT_PCLMUL_SMALL_KERNEL: Crc16Fn = arch::crc16_ccitt_pclmul_small_safe;

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

  /// IBM PCLMUL small-buffer kernel.
  #[allow(dead_code)] // Used by bench + policy dispatch.
  pub const IBM_PCLMUL_SMALL_KERNEL: Crc16Fn = arch::crc16_ibm_pclmul_small_safe;

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
  /// PMULL small-buffer kernel name.
  pub const PMULL_SMALL: &str = "aarch64/pmull-small";
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

  /// CCITT PMULL small-buffer kernel.
  #[allow(dead_code)] // Used by bench + policy dispatch.
  pub const CCITT_PMULL_SMALL_KERNEL: Crc16Fn = arch::crc16_ccitt_pmull_small_safe;

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

  /// IBM PMULL small-buffer kernel.
  #[allow(dead_code)] // Used by bench + policy dispatch.
  pub const IBM_PMULL_SMALL_KERNEL: Crc16Fn = arch::crc16_ibm_pmull_small_safe;
}

#[cfg(target_arch = "powerpc64")]
pub mod power {
  use super::super::power as arch;
  use crate::dispatchers::Crc16Fn;

  /// VPMSUM kernel name (POWER8+ carryless multiply).
  pub const VPMSUM: &str = "power/vpmsum";

  /// VPMSUM kernel names: [1-way, 2-way, 4-way, 8-way, 8-way(dup)].
  pub const VPMSUM_NAMES: &[&str] = &[
    "power/vpmsum",
    "power/vpmsum-2way",
    "power/vpmsum-4way",
    "power/vpmsum-8way",
    "power/vpmsum-8way",
  ];

  /// CCITT VPMSUM kernels: [1-way, 2-way, 4-way, 8-way, 8-way(dup)].
  #[allow(dead_code)] // Used by bench + policy dispatch.
  pub const CCITT_VPMSUM: [Crc16Fn; 5] = [
    arch::crc16_ccitt_vpmsum_safe,
    arch::crc16_ccitt_vpmsum_2way_safe,
    arch::crc16_ccitt_vpmsum_4way_safe,
    arch::crc16_ccitt_vpmsum_8way_safe,
    arch::crc16_ccitt_vpmsum_8way_safe,
  ];

  /// IBM VPMSUM kernels: [1-way, 2-way, 4-way, 8-way, 8-way(dup)].
  #[allow(dead_code)] // Used by bench + policy dispatch.
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
  use crate::dispatchers::Crc16Fn;

  /// VGFM kernel name (s390x vector Galois field multiply).
  pub const VGFM: &str = "s390x/vgfm";

  /// VGFM kernel names: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const VGFM_NAMES: &[&str] = &[
    "s390x/vgfm",
    "s390x/vgfm-2way",
    "s390x/vgfm-4way",
    "s390x/vgfm-4way",
    "s390x/vgfm-4way",
  ];

  /// CCITT VGFM kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  #[allow(dead_code)] // Used by bench + policy dispatch.
  pub const CCITT_VGFM: [Crc16Fn; 5] = [
    arch::crc16_ccitt_vgfm_safe,
    arch::crc16_ccitt_vgfm_2way_safe,
    arch::crc16_ccitt_vgfm_4way_safe,
    arch::crc16_ccitt_vgfm_4way_safe,
    arch::crc16_ccitt_vgfm_4way_safe,
  ];

  /// IBM VGFM kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  #[allow(dead_code)] // Used by bench + policy dispatch.
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
  use crate::dispatchers::Crc16Fn;

  /// Zbc kernel name (scalar carryless multiply).
  pub const ZBC: &str = "riscv64/zbc";
  /// Zbc kernel names: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const ZBC_NAMES: &[&str] = &[
    "riscv64/zbc",
    "riscv64/zbc-2way",
    "riscv64/zbc-4way",
    "riscv64/zbc-4way",
    "riscv64/zbc-4way",
  ];

  /// Zvbc kernel name (vector carryless multiply).
  pub const ZVBC: &str = "riscv64/zvbc";
  /// Zvbc kernel names: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const ZVBC_NAMES: &[&str] = &[
    "riscv64/zvbc",
    "riscv64/zvbc-2way",
    "riscv64/zvbc-4way",
    "riscv64/zvbc-4way",
    "riscv64/zvbc-4way",
  ];

  /// CCITT Zbc kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  #[allow(dead_code)] // Used by bench + policy dispatch.
  pub const CCITT_ZBC: [Crc16Fn; 5] = [
    arch::crc16_ccitt_zbc_safe,
    arch::crc16_ccitt_zbc_2way_safe,
    arch::crc16_ccitt_zbc_4way_safe,
    arch::crc16_ccitt_zbc_4way_safe,
    arch::crc16_ccitt_zbc_4way_safe,
  ];

  /// CCITT Zvbc kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  #[allow(dead_code)] // Used by bench + policy dispatch.
  pub const CCITT_ZVBC: [Crc16Fn; 5] = [
    arch::crc16_ccitt_zvbc_safe,
    arch::crc16_ccitt_zvbc_2way_safe,
    arch::crc16_ccitt_zvbc_4way_safe,
    arch::crc16_ccitt_zvbc_4way_safe,
    arch::crc16_ccitt_zvbc_4way_safe,
  ];

  /// IBM Zbc kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  #[allow(dead_code)] // Used by bench + policy dispatch.
  pub const IBM_ZBC: [Crc16Fn; 5] = [
    arch::crc16_ibm_zbc_safe,
    arch::crc16_ibm_zbc_2way_safe,
    arch::crc16_ibm_zbc_4way_safe,
    arch::crc16_ibm_zbc_4way_safe,
    arch::crc16_ibm_zbc_4way_safe,
  ];

  /// IBM Zvbc kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  #[allow(dead_code)] // Used by bench + policy dispatch.
  pub const IBM_ZVBC: [Crc16Fn; 5] = [
    arch::crc16_ibm_zvbc_safe,
    arch::crc16_ibm_zvbc_2way_safe,
    arch::crc16_ibm_zvbc_4way_safe,
    arch::crc16_ibm_zvbc_4way_safe,
    arch::crc16_ibm_zvbc_4way_safe,
  ];
}
