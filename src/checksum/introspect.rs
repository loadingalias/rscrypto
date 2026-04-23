//! Kernel dispatch introspection for verifying optimization.
//!
//! This module provides user-friendly APIs to inspect which kernels are selected
//! for the current platform, without impacting hot-path performance.
//!
//! # Examples
//!
//! ```
//! use rscrypto::checksum::{
//!   Crc64,
//!   introspect::{DispatchInfo, KernelIntrospect},
//! };
//!
//! // Platform-level info
//! let info = DispatchInfo::current();
//! assert!(!format!("{info}").is_empty());
//!
//! // Per-algorithm kernel selection
//! assert!(!Crc64::kernel_name_for_len(1024).is_empty());
//! assert!(!Crc64::kernel_name_for_len(4096).is_empty());
//! ```

pub use crate::{
  checksum::kernel_table::is_hardware_accelerated,
  platform::{DispatchInfo, KernelIntrospect},
};

/// Returns the kernel name selected for a specific algorithm and buffer size.
///
/// This is useful for verifying size-based kernel transitions.
///
/// # Examples
///
/// ```
/// use rscrypto::checksum::{Crc64, introspect::kernel_for};
///
/// // Check kernel selection at different buffer sizes
/// let small = kernel_for::<Crc64>(64);
/// let large = kernel_for::<Crc64>(65536);
/// assert!(!small.is_empty());
/// assert!(!large.is_empty());
/// ```
#[inline]
#[must_use]
pub fn kernel_for<T: KernelIntrospect>(len: usize) -> &'static str {
  T::kernel_name_for_len(len)
}

#[cfg(test)]
mod tests {

  use super::*;

  #[test]
  fn dispatch_info_display_not_empty() {
    let info = DispatchInfo::current();
    let s = alloc::format!("{info}");
    assert!(!s.is_empty());
  }

  #[test]
  fn dispatch_info_debug_not_empty() {
    let info = DispatchInfo::current();
    let s = alloc::format!("{info:?}");
    assert!(!s.is_empty());
    assert!(s.contains("DispatchInfo"));
  }

  #[test]
  fn dispatch_info_is_copy() {
    let info = DispatchInfo::current();
    let copy = info;
    let _ = info; // Use original after copy - proves Copy
    let _ = copy;
  }
}
