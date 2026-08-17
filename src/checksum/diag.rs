//! Optional diagnostics for kernel selection.
//!
//! This module is behind `cfg(feature = "diag")` and is intended for
//! explainable/debuggable kernel selection without affecting normal builds.
//!
//! ```
//! # #[cfg(feature = "crc32")]
//! # {
//! use rscrypto::checksum::diag::{self, Crc32Polynomial};
//! assert_eq!(diag::crc32_ieee(1024).polynomial, Crc32Polynomial::Ieee);
//! # }
//! # #[cfg(feature = "crc64")]
//! # {
//! use rscrypto::checksum::diag::{self, Crc64Polynomial};
//! assert_eq!(diag::crc64_xz(1024).polynomial, Crc64Polynomial::Xz);
//! # }
//! ```

#[cfg(any(feature = "crc32", feature = "crc64"))]
use crate::platform::Arch;

/// High-level reason for a selection outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
#[cfg(feature = "crc32")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Crc32Polynomial {
  /// CRC-32/ISO-HDLC, commonly called CRC-32/IEEE.
  Ieee,
  /// CRC-32C using the Castagnoli polynomial.
  Castagnoli,
}

/// CRC-64 polynomial variant (selection diagnostics).
#[cfg(feature = "crc64")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Crc64Polynomial {
  /// CRC-64/XZ using the reflected ECMA-182 polynomial.
  Xz,
  /// CRC-64/NVME.
  Nvme,
}

/// Snapshot explaining the CRC-32 kernel selected for one input length.
///
/// Capability booleans describe facts reported by the active selection policy;
/// use [`crate::platform`] when direct platform capability detection is needed.
#[cfg(feature = "crc32")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc32SelectionDiag {
  /// Polynomial variant evaluated by the selector.
  pub polynomial: Crc32Polynomial,
  /// Input length supplied to the selector, in bytes.
  pub len: usize,
  /// Detected architecture used to choose the dispatch table.
  pub arch: Arch,
  /// High-level reason the reported kernel was selected.
  pub reason: SelectionReason,
  /// Force request after clamping it to detected platform capabilities.
  pub effective_force: crate::checksum::config::Crc32Force,
  /// Name of the policy family that produced this snapshot.
  pub policy_family: &'static str,
  /// Stable name of the selected kernel.
  pub selected_kernel: &'static str,
  /// Stream count reported by the policy.
  ///
  /// The table-backed policy reports `1`; wider stream selection is encoded in
  /// [`Self::selected_kernel`].
  pub selected_streams: u8,
  /// First size boundary in the active CRC-32 dispatch table.
  ///
  /// The legacy field name does not guarantee that the next tier uses hardware
  /// CRC instructions.
  pub portable_to_hwcrc: usize,
  /// Hardware-CRC-to-fusion boundary reported by the policy.
  ///
  /// The table-backed policy currently reports its first size boundary here.
  pub hwcrc_to_fusion: usize,
  /// Fusion-to-AVX-512 boundary, or [`usize::MAX`] when not modeled separately.
  pub fusion_to_avx512: usize,
  /// Fusion-to-VPCLMUL boundary, or [`usize::MAX`] when not modeled separately.
  pub fusion_to_vpclmul: usize,
  /// Minimum bytes per reported stream, or [`usize::MAX`] when unavailable.
  pub min_bytes_per_lane: usize,
  /// Whether the policy classified this selection as memory-bound.
  pub memory_bound: bool,
  /// Whether the policy reported a hardware CRC tier.
  pub has_hwcrc: bool,
  /// Whether the policy reported a fused CRC/CLMUL tier.
  pub has_fusion: bool,
  /// Whether the policy reported VPCLMUL support.
  pub has_vpclmul: bool,
  /// Whether the policy reported AVX-512 support.
  pub has_avx512: bool,
  /// Whether the policy reported an AArch64 EOR3 tier.
  pub has_eor3: bool,
  /// Whether the policy reported an AArch64 SVE2 tier.
  pub has_sve2: bool,
}

/// Snapshot explaining the CRC-64 kernel selected for one input length.
#[cfg(feature = "crc64")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crc64SelectionDiag {
  /// Polynomial variant evaluated by the selector.
  pub polynomial: Crc64Polynomial,
  /// Input length supplied to the selector, in bytes.
  pub len: usize,
  /// Detected architecture used to choose the dispatch table.
  pub arch: Arch,
  /// High-level reason the reported kernel was selected.
  pub reason: SelectionReason,
  /// Force request after clamping it to detected platform capabilities.
  pub effective_force: crate::checksum::config::Crc64Force,
  /// Name of the policy family that produced this snapshot.
  pub policy_family: &'static str,
  /// Stable name of the selected kernel.
  pub selected_kernel: &'static str,
  /// Stream count reported by the policy.
  ///
  /// The table-backed policy reports `1`; wider stream selection is encoded in
  /// [`Self::selected_kernel`].
  pub selected_streams: u8,
  /// Boundary between portable and carryless-multiply tiers, in bytes.
  pub portable_to_clmul: usize,
  /// Boundary between narrow and wide carryless-multiply tiers, in bytes.
  ///
  /// The legacy field name also represents equivalent non-x86 wide tiers.
  pub pclmul_to_vpclmul: usize,
  /// Largest input size assigned to the small-buffer kernel, in bytes.
  pub small_kernel_max_bytes: usize,
  /// Whether the policy selected the four-lane 512-bit VPCLMUL strategy.
  pub use_4x512: bool,
  /// Minimum bytes per reported stream, or [`usize::MAX`] when unavailable.
  pub min_bytes_per_lane: usize,
}

/// Diagnose CRC-32 (IEEE) selection for `len`.
#[cfg(feature = "crc32")]
#[inline]
#[must_use]
pub fn crc32_ieee(len: usize) -> Crc32SelectionDiag {
  crate::checksum::crc32::diag_crc32_ieee(len)
}

/// Diagnose CRC-32C (Castagnoli) selection for `len`.
#[cfg(feature = "crc32")]
#[inline]
#[must_use]
pub fn crc32c(len: usize) -> Crc32SelectionDiag {
  crate::checksum::crc32::diag_crc32c(len)
}

/// Diagnose CRC-64/XZ selection for `len`.
#[cfg(feature = "crc64")]
#[inline]
#[must_use]
pub fn crc64_xz(len: usize) -> Crc64SelectionDiag {
  crate::checksum::crc64::diag_crc64_xz(len)
}

/// Diagnose CRC-64/NVME selection for `len`.
#[cfg(feature = "crc64")]
#[inline]
#[must_use]
pub fn crc64_nvme(len: usize) -> Crc64SelectionDiag {
  crate::checksum::crc64::diag_crc64_nvme(len)
}
