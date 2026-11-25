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
  /// A forced mode was active (kernel selection bypassed normal thresholds).
  Forced,
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
/// Use [`crate::platform`] for direct platform capability detection.
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
  pub effective_force: crate::checksum::config::CrcForce,
  /// Stable name of the selected kernel.
  pub selected_kernel: &'static str,
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
  pub effective_force: crate::checksum::config::CrcForce,
  /// Stable name of the selected kernel.
  pub selected_kernel: &'static str,
}

/// The selection reason for an effective force.
#[cfg(any(feature = "crc32", feature = "crc64"))]
fn reason(config: crate::checksum::config::CrcConfig) -> SelectionReason {
  if config.effective_force == crate::checksum::config::CrcForce::Auto {
    SelectionReason::Auto
  } else {
    SelectionReason::Forced
  }
}

#[cfg(any(feature = "crc32", feature = "crc64"))]
macro_rules! selection {
  ($crc:ident, $diag:ident, $polynomial:path, $len:expr) => {{
    use crate::checksum::$crc;
    let config = $crc::config();
    $diag {
      polynomial: $polynomial,
      len: $len,
      arch: crate::platform::arch(),
      reason: reason(config),
      effective_force: config.effective_force,
      selected_kernel: $crc::kernel_name_for_len($len),
    }
  }};
}

/// Diagnose CRC-32 (IEEE) selection for `len`.
#[cfg(feature = "crc32")]
#[inline]
#[must_use]
pub fn crc32_ieee(len: usize) -> Crc32SelectionDiag {
  selection!(Crc32, Crc32SelectionDiag, Crc32Polynomial::Ieee, len)
}

/// Diagnose CRC-32C (Castagnoli) selection for `len`.
#[cfg(feature = "crc32")]
#[inline]
#[must_use]
pub fn crc32c(len: usize) -> Crc32SelectionDiag {
  selection!(Crc32C, Crc32SelectionDiag, Crc32Polynomial::Castagnoli, len)
}

/// Diagnose CRC-64/XZ selection for `len`.
#[cfg(feature = "crc64")]
#[inline]
#[must_use]
pub fn crc64_xz(len: usize) -> Crc64SelectionDiag {
  selection!(Crc64, Crc64SelectionDiag, Crc64Polynomial::Xz, len)
}

/// Diagnose CRC-64/NVME selection for `len`.
#[cfg(feature = "crc64")]
#[inline]
#[must_use]
pub fn crc64_nvme(len: usize) -> Crc64SelectionDiag {
  selection!(Crc64Nvme, Crc64SelectionDiag, Crc64Polynomial::Nvme, len)
}
