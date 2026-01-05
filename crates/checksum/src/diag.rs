//! Optional diagnostics for kernel selection.
//!
//! This module is behind `cfg(feature = "diag")` and is intended for
//! explainable/debuggable kernel selection without affecting normal builds.

use platform::TuneKind;

/// High-level reason for a selection outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionReason {
  /// Below the algorithm's hard-coded tiny-size threshold (always portable).
  BelowSmallThreshold,
  /// A forced mode was active (kernel selection bypassed normal thresholds).
  Forced,
  /// Below the portable→SIMD transition threshold.
  BelowSimdThreshold,
  /// Normal auto selection.
  Auto,
}

/// CRC-32 polynomial variant (selection diagnostics).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Crc32Polynomial {
  Ieee,
  Castagnoli,
}

/// CRC-64 polynomial variant (selection diagnostics).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Crc64Polynomial {
  Xz,
  Nvme,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc32SelectionDiag {
  pub polynomial: Crc32Polynomial,
  pub len: usize,
  pub tune_kind: TuneKind,
  pub reason: SelectionReason,
  pub effective_force: crate::Crc32Force,
  pub policy_family: &'static str,
  pub selected_kernel: &'static str,
  pub selected_streams: u8,
  pub portable_to_hwcrc: usize,
  pub hwcrc_to_fusion: usize,
  pub fusion_to_avx512: usize,
  pub fusion_to_vpclmul: usize,
  pub min_bytes_per_lane: usize,
  pub memory_bound: bool,
  pub has_hwcrc: bool,
  pub has_fusion: bool,
  pub has_vpclmul: bool,
  pub has_avx512: bool,
  pub has_eor3: bool,
  pub has_sve2: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc64SelectionDiag {
  pub polynomial: Crc64Polynomial,
  pub len: usize,
  pub tune_kind: TuneKind,
  pub reason: SelectionReason,
  pub effective_force: crate::Crc64Force,
  pub policy_family: &'static str,
  pub selected_kernel: &'static str,
  pub selected_streams: u8,
  pub portable_to_clmul: usize,
  pub pclmul_to_vpclmul: usize,
  pub small_kernel_max_bytes: usize,
  pub use_4x512: bool,
  pub min_bytes_per_lane: usize,
}

/// Diagnose CRC-32 (IEEE) selection for `len`.
#[inline]
#[must_use]
pub fn crc32_ieee(len: usize) -> Crc32SelectionDiag {
  crate::crc32::diag_crc32_ieee(len)
}

/// Diagnose CRC-32C (Castagnoli) selection for `len`.
#[inline]
#[must_use]
pub fn crc32c(len: usize) -> Crc32SelectionDiag {
  crate::crc32::diag_crc32c(len)
}

/// Diagnose CRC-64/XZ selection for `len`.
#[inline]
#[must_use]
pub fn crc64_xz(len: usize) -> Crc64SelectionDiag {
  crate::crc64::diag_crc64_xz(len)
}

/// Diagnose CRC-64/NVME selection for `len`.
#[inline]
#[must_use]
pub fn crc64_nvme(len: usize) -> Crc64SelectionDiag {
  crate::crc64::diag_crc64_nvme(len)
}
