//! Static kernel name tables for CRC-24.
//!
//! This mirrors the structure used by CRC-32/CRC-64: keep all kernel names in
//! one place so both introspection and dispatch share identifiers.
//!
//! # Kernel Tiers
//!
//! CRC-24 supports Tiers 0, 1, 3, and 4 (no HW CRC instructions exist):
//! - Tier 0 (Reference): Bitwise implementation
//! - Tier 1 (Portable): Slice-by-4/8 table lookup
//! - Tier 3 (Folding): PCLMUL (x86_64), PMULL (aarch64), VPMSUM (Power), VGFM (s390x), Zbc
//!   (riscv64)
//! - Tier 4 (Wide): VPCLMUL (x86_64 AVX-512), Zvbc (riscv64)
//!
//! ## Why No SIMD Acceleration?
//!
//! CRC-24/OPENPGP uses a non-reflected (MSB-first) polynomial. This means:
//! - Data bits are processed high-to-low instead of low-to-high
//! - Carryless multiply folding requires additional byte-reversal operations
//! - The OpenPGP use case (ASCII armor integrity) doesn't require extreme throughput
//!
//! SIMD acceleration is provided via CLMUL/PMULL folding by internally computing
//! the equivalent reflected CRC-24 over bit-reversed bytes, then converting the
//! state back to the MSB-first OpenPGP representation.

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
  use crate::checksum::dispatchers::Crc24Fn;

  /// OpenPGP PCLMUL kernel.
  pub const OPENPGP_PCLMUL: [Crc24Fn; 5] = [
    arch::crc24_openpgp_pclmul_safe,
    arch::crc24_openpgp_pclmul_2way_safe,
    arch::crc24_openpgp_pclmul_4way_safe,
    arch::crc24_openpgp_pclmul_7way_safe,
    arch::crc24_openpgp_pclmul_8way_safe,
  ];

  /// OpenPGP PCLMUL small-buffer kernel.
  pub const OPENPGP_PCLMUL_SMALL_KERNEL: Crc24Fn = arch::crc24_openpgp_pclmul_small_safe;

  /// OpenPGP VPCLMUL kernel.
  pub const OPENPGP_VPCLMUL: [Crc24Fn; 5] = [
    arch::crc24_openpgp_vpclmul_safe,
    arch::crc24_openpgp_vpclmul_2way_safe,
    arch::crc24_openpgp_vpclmul_4way_safe,
    arch::crc24_openpgp_vpclmul_7way_safe,
    arch::crc24_openpgp_vpclmul_8way_safe,
  ];
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
  use super::super::aarch64 as arch;
  use crate::checksum::dispatchers::Crc24Fn;

  /// OpenPGP PMULL kernels: [1-way, 2-way, 3-way, 3-way(dup), 3-way(dup)].
  pub const OPENPGP_PMULL: [Crc24Fn; 5] = [
    arch::crc24_openpgp_pmull_safe,
    arch::crc24_openpgp_pmull_2way_safe,
    arch::crc24_openpgp_pmull_3way_safe,
    arch::crc24_openpgp_pmull_3way_safe, // dup for index consistency
    arch::crc24_openpgp_pmull_3way_safe, // dup for index consistency
  ];

  /// OpenPGP PMULL small-buffer kernel.
  pub const OPENPGP_PMULL_SMALL_KERNEL: Crc24Fn = arch::crc24_openpgp_pmull_small_safe;
}

#[cfg(target_arch = "powerpc64")]
pub mod power {
  use super::super::power as arch;
  use crate::checksum::dispatchers::Crc24Fn;

  /// OpenPGP VPMSUM kernels: [1-way, 2-way, 4-way, 8-way, 8-way(dup)].
  pub const OPENPGP_VPMSUM: [Crc24Fn; 5] = [
    arch::crc24_openpgp_vpmsum_safe,
    arch::crc24_openpgp_vpmsum_2way_safe,
    arch::crc24_openpgp_vpmsum_4way_safe,
    arch::crc24_openpgp_vpmsum_8way_safe,
    arch::crc24_openpgp_vpmsum_8way_safe, // dup for index consistency
  ];
}

#[cfg(target_arch = "s390x")]
pub mod s390x {
  use super::super::s390x as arch;
  use crate::checksum::dispatchers::Crc24Fn;

  /// OpenPGP VGFM kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const OPENPGP_VGFM: [Crc24Fn; 5] = [
    arch::crc24_openpgp_vgfm_safe,
    arch::crc24_openpgp_vgfm_2way_safe,
    arch::crc24_openpgp_vgfm_4way_safe,
    arch::crc24_openpgp_vgfm_4way_safe, // dup for index consistency
    arch::crc24_openpgp_vgfm_4way_safe, // dup for index consistency
  ];
}

#[cfg(target_arch = "riscv64")]
#[allow(dead_code)]
pub mod riscv64 {
  use super::super::riscv64 as arch;
  use crate::checksum::dispatchers::Crc24Fn;

  /// Zbc kernel names: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const ZBC_NAMES: &[&str] = &[
    "riscv64/zbc",
    "riscv64/zbc-2way",
    "riscv64/zbc-4way",
    "riscv64/zbc-4way", // dup for index consistency
    "riscv64/zbc-4way", // dup for index consistency
  ];

  /// Zvbc kernel names: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const ZVBC_NAMES: &[&str] = &[
    "riscv64/zvbc",
    "riscv64/zvbc-2way",
    "riscv64/zvbc-4way",
    "riscv64/zvbc-4way", // dup for index consistency
    "riscv64/zvbc-4way", // dup for index consistency
  ];

  /// OpenPGP Zbc kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const OPENPGP_ZBC: [Crc24Fn; 5] = [
    arch::crc24_openpgp_zbc_safe,
    arch::crc24_openpgp_zbc_2way_safe,
    arch::crc24_openpgp_zbc_4way_safe,
    arch::crc24_openpgp_zbc_4way_safe, // dup for index consistency
    arch::crc24_openpgp_zbc_4way_safe, // dup for index consistency
  ];

  /// OpenPGP Zvbc kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const OPENPGP_ZVBC: [Crc24Fn; 5] = [
    arch::crc24_openpgp_zvbc_safe,
    arch::crc24_openpgp_zvbc_2way_safe,
    arch::crc24_openpgp_zvbc_4way_safe,
    arch::crc24_openpgp_zvbc_4way_safe, // dup for index consistency
    arch::crc24_openpgp_zvbc_4way_safe, // dup for index consistency
  ];
}
