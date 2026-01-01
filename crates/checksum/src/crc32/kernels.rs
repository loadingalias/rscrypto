//! Static kernel name tables and dispatch helpers for CRC-32.
//!
//! This module mirrors the CRC-64 layout and centralizes all kernel names so
//! both name introspection and dispatch can share the same identifiers.
//!
//! # Kernel Tiers
//!
//! CRC-32 supports all tiers (0-4):
//! - Tier 0 (Reference): Bitwise implementation
//! - Tier 1 (Portable): Slice-by-16 table lookup
//! - Tier 2 (HW CRC): SSE4.2 `crc32` (x86_64), ARMv8 CRC extension (aarch64) - CRC-32C only on x86
//! - Tier 3 (Folding): PCLMUL (x86_64), PMULL (aarch64), VPMSUM (ppc64), VGFM (s390x), Zbc
//!   (riscv64)
//! - Tier 4 (Wide): VPCLMUL (x86_64), PMULL+EOR3/SVE2 (aarch64), Zvbc (riscv64)

/// Portable fallback kernel name.
pub use kernels::PORTABLE_SLICE16 as PORTABLE;
/// Reference (bitwise) kernel name - always available for force mode.
pub use kernels::REFERENCE;

use crate::common::kernels;
#[cfg(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "powerpc64",
  target_arch = "s390x",
  target_arch = "riscv64"
))]
use crate::dispatchers::Crc32Fn;

// Generate CRC32-specific dispatch functions using the common macro.
// Only needed on SIMD architectures where we have multiple kernel tiers.
#[cfg(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "powerpc64",
  target_arch = "s390x",
  target_arch = "riscv64"
))]
crate::define_crc_dispatch!(Crc32Fn, u32);

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Name Tables (per architecture)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
  use super::super::x86_64 as arch;
  use crate::dispatchers::Crc32Fn;

  /// CRC-32 (IEEE) PCLMUL kernel names.
  pub const CRC32_PCLMUL_NAMES: &[&str] = &[
    "x86_64/pclmul",
    "x86_64/pclmul-2way",
    "x86_64/pclmul-4way",
    "x86_64/pclmul-7way",
    "x86_64/pclmul-8way",
  ];
  /// CRC-32 (IEEE) PCLMUL small-buffer kernel name.
  pub const CRC32_PCLMUL_SMALL: &str = "x86_64/pclmul-small";
  /// CRC-32 (IEEE) PCLMUL kernels.
  pub const CRC32_PCLMUL: [Crc32Fn; 5] = [
    arch::crc32_ieee_pclmul_safe,
    arch::crc32_ieee_pclmul_2way_safe,
    arch::crc32_ieee_pclmul_4way_safe,
    arch::crc32_ieee_pclmul_7way_safe,
    arch::crc32_ieee_pclmul_8way_safe,
  ];
  /// CRC-32 (IEEE) PCLMUL small-buffer kernel.
  pub const CRC32_PCLMUL_SMALL_KERNEL: Crc32Fn = arch::crc32_ieee_pclmul_small_safe;

  /// CRC-32 (IEEE) VPCLMUL kernel names.
  pub const CRC32_VPCLMUL_NAMES: &[&str] = &[
    "x86_64/vpclmul",
    "x86_64/vpclmul-2way",
    "x86_64/vpclmul-4way",
    "x86_64/vpclmul-7way",
    "x86_64/vpclmul-8way",
  ];
  /// CRC-32 (IEEE) VPCLMUL small-buffer kernel name.
  #[allow(dead_code)] // Reserved for future wide_small policy field
  pub const CRC32_VPCLMUL_SMALL: &str = "x86_64/vpclmul-small";
  /// CRC-32 (IEEE) VPCLMUL kernels.
  pub const CRC32_VPCLMUL: [Crc32Fn; 5] = [
    arch::crc32_ieee_vpclmul_safe,
    arch::crc32_ieee_vpclmul_2way_safe,
    arch::crc32_ieee_vpclmul_4way_safe,
    arch::crc32_ieee_vpclmul_7way_safe,
    arch::crc32_ieee_vpclmul_8way_safe,
  ];
  /// CRC-32 (IEEE) VPCLMUL small-buffer kernel (falls back to the PCLMUL small-lane kernel).
  #[allow(dead_code)] // Reserved for future wide_small policy field
  pub const CRC32_VPCLMUL_SMALL_KERNEL: Crc32Fn = arch::crc32_ieee_vpclmul_small_safe;

  /// SSE4.2 `crc32` instruction kernel names (CRC-32C only).
  pub const CRC32C_HWCRC_NAMES: &[&str] = &[
    "x86_64/hwcrc",
    "x86_64/hwcrc-2way",
    "x86_64/hwcrc-4way",
    "x86_64/hwcrc-7way",
    "x86_64/hwcrc-8way",
  ];

  /// CRC-32C SSE4.2 kernel function array.
  pub const CRC32C_HWCRC: [Crc32Fn; 5] = [
    arch::crc32c_sse42_safe,
    arch::crc32c_sse42_2way_safe,
    arch::crc32c_sse42_4way_safe,
    arch::crc32c_sse42_7way_safe,
    arch::crc32c_sse42_8way_safe,
  ];

  /// CRC-32C fusion (SSE4.2 + PCLMULQDQ) kernel names.
  pub const CRC32C_FUSION_SSE_NAMES: &[&str] = &[
    "x86_64/fusion-sse-v4s3x3",
    "x86_64/fusion-sse-v4s3x3-2way",
    "x86_64/fusion-sse-v4s3x3-4way",
    "x86_64/fusion-sse-v4s3x3-7way",
    "x86_64/fusion-sse-v4s3x3-8way",
  ];

  /// CRC-32C fusion (SSE4.2 + PCLMULQDQ) kernels.
  pub const CRC32C_FUSION_SSE: [Crc32Fn; 5] = [
    arch::crc32c_iscsi_sse_v4s3x3_safe,
    arch::crc32c_iscsi_sse_v4s3x3_2way_safe,
    arch::crc32c_iscsi_sse_v4s3x3_4way_safe,
    arch::crc32c_iscsi_sse_v4s3x3_7way_safe,
    arch::crc32c_iscsi_sse_v4s3x3_8way_safe,
  ];

  /// CRC-32C fusion (AVX-512 + PCLMULQDQ) kernel names.
  pub const CRC32C_FUSION_AVX512_NAMES: &[&str] = &[
    "x86_64/fusion-avx512-v4s3x3",
    "x86_64/fusion-avx512-v4s3x3-2way",
    "x86_64/fusion-avx512-v4s3x3-4way",
    "x86_64/fusion-avx512-v4s3x3-7way",
    "x86_64/fusion-avx512-v4s3x3-8way",
  ];

  /// CRC-32C fusion (AVX-512 + PCLMULQDQ) kernels.
  pub const CRC32C_FUSION_AVX512: [Crc32Fn; 5] = [
    arch::crc32c_iscsi_avx512_v4s3x3_safe,
    arch::crc32c_iscsi_avx512_v4s3x3_2way_safe,
    arch::crc32c_iscsi_avx512_v4s3x3_4way_safe,
    arch::crc32c_iscsi_avx512_v4s3x3_7way_safe,
    arch::crc32c_iscsi_avx512_v4s3x3_8way_safe,
  ];

  /// CRC-32C fusion (AVX-512 + VPCLMULQDQ) kernel names.
  pub const CRC32C_FUSION_VPCLMUL_NAMES: &[&str] = &[
    "x86_64/fusion-vpclmul-v3x2",
    "x86_64/fusion-vpclmul-v3x2-2way",
    "x86_64/fusion-vpclmul-v3x2-4way",
    "x86_64/fusion-vpclmul-v3x2-7way",
    "x86_64/fusion-vpclmul-v3x2-8way",
  ];

  /// CRC-32C fusion (AVX-512 + VPCLMULQDQ) kernels.
  pub const CRC32C_FUSION_VPCLMUL: [Crc32Fn; 5] = [
    arch::crc32c_iscsi_avx512_vpclmulqdq_v3x2_safe,
    arch::crc32c_iscsi_avx512_vpclmulqdq_v3x2_2way_safe,
    arch::crc32c_iscsi_avx512_vpclmulqdq_v3x2_4way_safe,
    arch::crc32c_iscsi_avx512_vpclmulqdq_v3x2_7way_safe,
    arch::crc32c_iscsi_avx512_vpclmulqdq_v3x2_8way_safe,
  ];
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
  use super::super::aarch64 as arch;
  use crate::dispatchers::Crc32Fn;

  /// ARMv8 CRC extension kernel names.
  pub const CRC32_HWCRC_NAMES: &[&str] = &[
    "aarch64/hwcrc",
    "aarch64/hwcrc-2way",
    "aarch64/hwcrc-3way",
    "aarch64/hwcrc-3way", // dup for index consistency
    "aarch64/hwcrc-3way", // dup for index consistency
  ];

  pub const CRC32C_HWCRC_NAMES: &[&str] = &[
    "aarch64/hwcrc",
    "aarch64/hwcrc-2way",
    "aarch64/hwcrc-3way",
    "aarch64/hwcrc-3way", // dup for index consistency
    "aarch64/hwcrc-3way", // dup for index consistency
  ];

  /// CRC-32 (IEEE) CRC-extension kernel function array.
  pub const CRC32_HWCRC: [Crc32Fn; 5] = [
    arch::crc32_armv8_safe,
    arch::crc32_armv8_2way_safe,
    arch::crc32_armv8_3way_safe,
    arch::crc32_armv8_3way_safe, // dup for index consistency
    arch::crc32_armv8_3way_safe, // dup for index consistency
  ];

  /// CRC-32C (Castagnoli) CRC-extension kernel function array.
  pub const CRC32C_HWCRC: [Crc32Fn; 5] = [
    arch::crc32c_armv8_safe,
    arch::crc32c_armv8_2way_safe,
    arch::crc32c_armv8_3way_safe,
    arch::crc32c_armv8_3way_safe, // dup for index consistency
    arch::crc32c_armv8_3way_safe, // dup for index consistency
  ];

  /// CRC-32 (IEEE) PMULL fusion kernel names.
  pub const CRC32_PMULL_NAMES: &[&str] = &[
    "aarch64/pmull-v9s3x2e-s3",
    "aarch64/pmull-v9s3x2e-s3-2way",
    "aarch64/pmull-v9s3x2e-s3-3way",
    "aarch64/pmull-v9s3x2e-s3-3way", // dup for index consistency
    "aarch64/pmull-v9s3x2e-s3-3way", // dup for index consistency
  ];

  /// CRC-32 (IEEE) PMULL fusion kernels.
  pub const CRC32_PMULL: [Crc32Fn; 5] = [
    arch::crc32_iso_hdlc_pmull_v9s3x2e_s3_safe,
    arch::crc32_iso_hdlc_pmull_2way_safe,
    arch::crc32_iso_hdlc_pmull_3way_safe,
    arch::crc32_iso_hdlc_pmull_3way_safe, // dup for index consistency
    arch::crc32_iso_hdlc_pmull_3way_safe, // dup for index consistency
  ];
  /// CRC-32 (IEEE) PMULL small-buffer kernel.
  pub const CRC32_PMULL_SMALL_KERNEL: Crc32Fn = arch::crc32_iso_hdlc_pmull_small_safe;
  /// CRC-32 PMULL small-buffer kernel name.
  pub const PMULL_SMALL: &str = "aarch64/pmull-small";

  /// CRC-32 (IEEE) PMULL+EOR3 fusion kernel names.
  pub const CRC32_PMULL_EOR3_NAMES: &[&str] = &[
    "aarch64/pmull-eor3-v9s3x2e-s3",
    "aarch64/pmull-eor3-v9s3x2e-s3-2way",
    "aarch64/pmull-eor3-v9s3x2e-s3-3way",
    "aarch64/pmull-eor3-v9s3x2e-s3-3way", // dup for index consistency
    "aarch64/pmull-eor3-v9s3x2e-s3-3way", // dup for index consistency
  ];

  /// CRC-32 (IEEE) PMULL+EOR3 fusion kernels.
  pub const CRC32_PMULL_EOR3: [Crc32Fn; 5] = [
    arch::crc32_iso_hdlc_pmull_eor3_v9s3x2e_s3_safe,
    arch::crc32_iso_hdlc_pmull_eor3_2way_safe,
    arch::crc32_iso_hdlc_pmull_eor3_3way_safe,
    arch::crc32_iso_hdlc_pmull_eor3_3way_safe, // dup for index consistency
    arch::crc32_iso_hdlc_pmull_eor3_3way_safe, // dup for index consistency
  ];

  /// CRC-32C (Castagnoli) PMULL fusion kernel names.
  pub const CRC32C_PMULL_NAMES: &[&str] = &[
    "aarch64/pmull-v9s3x2e-s3",
    "aarch64/pmull-v9s3x2e-s3-2way",
    "aarch64/pmull-v9s3x2e-s3-3way",
    "aarch64/pmull-v9s3x2e-s3-3way", // dup for index consistency
    "aarch64/pmull-v9s3x2e-s3-3way", // dup for index consistency
  ];

  /// CRC-32C (Castagnoli) PMULL fusion kernels.
  pub const CRC32C_PMULL: [Crc32Fn; 5] = [
    arch::crc32c_iscsi_pmull_v9s3x2e_s3_safe,
    arch::crc32c_iscsi_pmull_2way_safe,
    arch::crc32c_iscsi_pmull_3way_safe,
    arch::crc32c_iscsi_pmull_3way_safe, // dup for index consistency
    arch::crc32c_iscsi_pmull_3way_safe, // dup for index consistency
  ];
  /// CRC-32C PMULL small-buffer kernel.
  pub const CRC32C_PMULL_SMALL_KERNEL: Crc32Fn = arch::crc32c_iscsi_pmull_small_safe;

  /// CRC-32C (Castagnoli) PMULL+EOR3 fusion kernel names.
  pub const CRC32C_PMULL_EOR3_NAMES: &[&str] = &[
    "aarch64/pmull-eor3-v9s3x2e-s3",
    "aarch64/pmull-eor3-v9s3x2e-s3-2way",
    "aarch64/pmull-eor3-v9s3x2e-s3-3way",
    "aarch64/pmull-eor3-v9s3x2e-s3-3way", // dup for index consistency
    "aarch64/pmull-eor3-v9s3x2e-s3-3way", // dup for index consistency
  ];

  /// CRC-32C (Castagnoli) PMULL+EOR3 fusion kernels.
  pub const CRC32C_PMULL_EOR3: [Crc32Fn; 5] = [
    arch::crc32c_iscsi_pmull_eor3_v9s3x2e_s3_safe,
    arch::crc32c_iscsi_pmull_eor3_2way_safe,
    arch::crc32c_iscsi_pmull_eor3_3way_safe,
    arch::crc32c_iscsi_pmull_eor3_3way_safe, // dup for index consistency
    arch::crc32c_iscsi_pmull_eor3_3way_safe, // dup for index consistency
  ];

  /// CRC-32 (IEEE) "SVE2 PMULL" tier kernel names.
  pub const CRC32_SVE2_PMULL_NAMES: &[&str] = &[
    "aarch64/sve2-pmull",
    "aarch64/sve2-pmull-2way",
    "aarch64/sve2-pmull-3way",
    "aarch64/sve2-pmull-3way", // dup for index consistency
    "aarch64/sve2-pmull-3way", // dup for index consistency
  ];

  /// CRC-32 (IEEE) "SVE2 PMULL" tier kernels (2/3-way striping).
  pub const CRC32_SVE2_PMULL: [Crc32Fn; 5] = [
    arch::crc32_iso_hdlc_pmull_v12e_v1_safe,
    arch::crc32_iso_hdlc_sve2_pmull_2way_safe,
    arch::crc32_iso_hdlc_sve2_pmull_3way_safe,
    arch::crc32_iso_hdlc_sve2_pmull_3way_safe,
    arch::crc32_iso_hdlc_sve2_pmull_3way_safe,
  ];
  /// CRC-32 "SVE2 PMULL" small-buffer kernel.
  #[allow(dead_code)] // Reserved for future wide_small policy field
  pub const CRC32_SVE2_PMULL_SMALL_KERNEL: Crc32Fn = arch::crc32_iso_hdlc_sve2_pmull_small_safe;
  /// CRC-32 SVE2 PMULL small-buffer kernel name.
  #[allow(dead_code)] // Reserved for future wide_small policy field
  pub const SVE2_PMULL_SMALL: &str = "aarch64/sve2-pmull-small";

  /// CRC-32C (Castagnoli) "SVE2 PMULL" tier kernel names.
  pub const CRC32C_SVE2_PMULL_NAMES: &[&str] = &[
    "aarch64/sve2-pmull",
    "aarch64/sve2-pmull-2way",
    "aarch64/sve2-pmull-3way",
    "aarch64/sve2-pmull-3way", // dup for index consistency
    "aarch64/sve2-pmull-3way", // dup for index consistency
  ];

  /// CRC-32C (Castagnoli) "SVE2 PMULL" tier kernels (2/3-way striping).
  pub const CRC32C_SVE2_PMULL: [Crc32Fn; 5] = [
    arch::crc32c_iscsi_pmull_v12e_v1_safe,
    arch::crc32c_iscsi_sve2_pmull_2way_safe,
    arch::crc32c_iscsi_sve2_pmull_3way_safe,
    arch::crc32c_iscsi_sve2_pmull_3way_safe,
    arch::crc32c_iscsi_sve2_pmull_3way_safe,
  ];
  /// CRC-32C "SVE2 PMULL" small-buffer kernel.
  #[allow(dead_code)] // Reserved for future wide_small policy field
  pub const CRC32C_SVE2_PMULL_SMALL_KERNEL: Crc32Fn = arch::crc32c_iscsi_sve2_pmull_small_safe;
}

#[cfg(target_arch = "powerpc64")]
pub mod powerpc64 {
  use super::super::powerpc64 as arch;
  use crate::dispatchers::Crc32Fn;

  /// VPMSUM kernel names: [1-way, 2-way, 4-way, 8-way, 8-way(dup)].
  pub const CRC32_VPMSUM_NAMES: &[&str] = &[
    "powerpc64/vpmsum",
    "powerpc64/vpmsum-2way",
    "powerpc64/vpmsum-4way",
    "powerpc64/vpmsum-8way",
    "powerpc64/vpmsum-8way",
  ];

  /// CRC-32 (IEEE) VPMSUM kernels: [1-way, 2-way, 4-way, 8-way, 8-way(dup)].
  pub const CRC32_VPMSUM: [Crc32Fn; 5] = [
    arch::crc32_ieee_vpmsum_safe,
    arch::crc32_ieee_vpmsum_2way_safe,
    arch::crc32_ieee_vpmsum_4way_safe,
    arch::crc32_ieee_vpmsum_8way_safe,
    arch::crc32_ieee_vpmsum_8way_safe, // dup for index consistency
  ];

  /// CRC-32C VPMSUM kernel names: [1-way, 2-way, 4-way, 8-way, 8-way(dup)].
  pub const CRC32C_VPMSUM_NAMES: &[&str] = &[
    "powerpc64/vpmsum",
    "powerpc64/vpmsum-2way",
    "powerpc64/vpmsum-4way",
    "powerpc64/vpmsum-8way",
    "powerpc64/vpmsum-8way",
  ];

  /// CRC-32C VPMSUM kernels: [1-way, 2-way, 4-way, 8-way, 8-way(dup)].
  pub const CRC32C_VPMSUM: [Crc32Fn; 5] = [
    arch::crc32c_vpmsum_safe,
    arch::crc32c_vpmsum_2way_safe,
    arch::crc32c_vpmsum_4way_safe,
    arch::crc32c_vpmsum_8way_safe,
    arch::crc32c_vpmsum_8way_safe, // dup for index consistency
  ];
}

#[cfg(target_arch = "s390x")]
pub mod s390x {
  use super::super::s390x as arch;
  use crate::dispatchers::Crc32Fn;

  /// VGFM kernel names: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const CRC32_VGFM_NAMES: &[&str] = &[
    "s390x/vgfm",
    "s390x/vgfm-2way",
    "s390x/vgfm-4way",
    "s390x/vgfm-4way",
    "s390x/vgfm-4way",
  ];

  /// CRC-32 (IEEE) VGFM kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const CRC32_VGFM: [Crc32Fn; 5] = [
    arch::crc32_ieee_vgfm_safe,
    arch::crc32_ieee_vgfm_2way_safe,
    arch::crc32_ieee_vgfm_4way_safe,
    arch::crc32_ieee_vgfm_4way_safe, // dup for index consistency
    arch::crc32_ieee_vgfm_4way_safe, // dup for index consistency
  ];

  /// CRC-32C VGFM kernel names: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const CRC32C_VGFM_NAMES: &[&str] = &[
    "s390x/vgfm",
    "s390x/vgfm-2way",
    "s390x/vgfm-4way",
    "s390x/vgfm-4way",
    "s390x/vgfm-4way",
  ];

  /// CRC-32C VGFM kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const CRC32C_VGFM: [Crc32Fn; 5] = [
    arch::crc32c_vgfm_safe,
    arch::crc32c_vgfm_2way_safe,
    arch::crc32c_vgfm_4way_safe,
    arch::crc32c_vgfm_4way_safe, // dup for index consistency
    arch::crc32c_vgfm_4way_safe, // dup for index consistency
  ];
}

#[cfg(target_arch = "riscv64")]
pub mod riscv64 {
  use super::super::riscv64 as arch;
  use crate::dispatchers::Crc32Fn;

  /// Zbc kernel names: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const CRC32_ZBC_NAMES: &[&str] = &[
    "riscv64/zbc",
    "riscv64/zbc-2way",
    "riscv64/zbc-4way",
    "riscv64/zbc-4way",
    "riscv64/zbc-4way",
  ];

  /// Zvbc kernel names: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const CRC32_ZVBC_NAMES: &[&str] = &[
    "riscv64/zvbc",
    "riscv64/zvbc-2way",
    "riscv64/zvbc-4way",
    "riscv64/zvbc-4way",
    "riscv64/zvbc-4way",
  ];

  /// CRC-32 (IEEE) Zbc kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const CRC32_ZBC: [Crc32Fn; 5] = [
    arch::crc32_ieee_zbc_safe,
    arch::crc32_ieee_zbc_2way_safe,
    arch::crc32_ieee_zbc_4way_safe,
    arch::crc32_ieee_zbc_4way_safe, // dup for index consistency
    arch::crc32_ieee_zbc_4way_safe, // dup for index consistency
  ];

  /// CRC-32 (IEEE) Zvbc kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const CRC32_ZVBC: [Crc32Fn; 5] = [
    arch::crc32_ieee_zvbc_safe,
    arch::crc32_ieee_zvbc_2way_safe,
    arch::crc32_ieee_zvbc_4way_safe,
    arch::crc32_ieee_zvbc_4way_safe, // dup for index consistency
    arch::crc32_ieee_zvbc_4way_safe, // dup for index consistency
  ];

  /// CRC-32C Zbc kernel names: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const CRC32C_ZBC_NAMES: &[&str] = &[
    "riscv64/zbc",
    "riscv64/zbc-2way",
    "riscv64/zbc-4way",
    "riscv64/zbc-4way",
    "riscv64/zbc-4way",
  ];

  /// CRC-32C Zvbc kernel names: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const CRC32C_ZVBC_NAMES: &[&str] = &[
    "riscv64/zvbc",
    "riscv64/zvbc-2way",
    "riscv64/zvbc-4way",
    "riscv64/zvbc-4way",
    "riscv64/zvbc-4way",
  ];

  /// CRC-32C Zbc kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const CRC32C_ZBC: [Crc32Fn; 5] = [
    arch::crc32c_zbc_safe,
    arch::crc32c_zbc_2way_safe,
    arch::crc32c_zbc_4way_safe,
    arch::crc32c_zbc_4way_safe, // dup for index consistency
    arch::crc32c_zbc_4way_safe, // dup for index consistency
  ];

  /// CRC-32C Zvbc kernels: [1-way, 2-way, 4-way, 4-way(dup), 4-way(dup)].
  pub const CRC32C_ZVBC: [Crc32Fn; 5] = [
    arch::crc32c_zvbc_safe,
    arch::crc32c_zvbc_2way_safe,
    arch::crc32c_zvbc_4way_safe,
    arch::crc32c_zvbc_4way_safe, // dup for index consistency
    arch::crc32c_zvbc_4way_safe, // dup for index consistency
  ];
}
