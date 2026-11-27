//! Kernel dispatch introspection for verifying optimization.
//!
//! This module provides user-friendly APIs to inspect which kernels are selected
//! for the current platform, without impacting hot-path performance.
//!
//! # Examples
//!
//! ```
//! use rscrypto::checksum::introspect::DispatchInfo;
//!
//! // Platform-level info
//! let info = DispatchInfo::current();
//! assert!(!format!("{info}").is_empty());
//! ```

#[cfg(any(feature = "crc16", feature = "crc24", feature = "crc32", feature = "crc64"))]
pub use crate::checksum::crc::dispatch::is_hardware_accelerated;
pub use crate::platform::{DispatchInfo, KernelIntrospect};

/// Returns the kernel name selected for a specific algorithm and buffer size.
///
/// This is useful for verifying size-based kernel transitions.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "crc32")]
/// # {
/// use rscrypto::checksum::{Crc32, introspect::kernel_for};
///
/// // Check kernel selection at different buffer sizes
/// let small = kernel_for::<Crc32>(64);
/// let large = kernel_for::<Crc32>(65536);
/// assert!(!small.is_empty());
/// assert!(!large.is_empty());
/// # }
/// ```
#[inline]
#[must_use]
pub fn kernel_for<T: KernelIntrospect>(len: usize) -> &'static str {
  T::kernel_name_for_len(len)
}

#[cfg(any(feature = "crc32", feature = "crc64"))]
use crate::platform::Arch;

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
  /// Stable name of the selected kernel.
  pub selected_kernel: &'static str,
}

#[cfg(any(feature = "crc32", feature = "crc64"))]
macro_rules! selection {
  ($crc:ident, $diag:ident, $polynomial:path, $len:expr) => {{
    use crate::checksum::$crc;
    $diag {
      polynomial: $polynomial,
      len: $len,
      arch: crate::platform::arch(),
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
