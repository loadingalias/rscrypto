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
