//! Kernel dispatch introspection for verifying optimization.
//!
//! This module provides user-friendly APIs to inspect which kernels are selected
//! for the current platform, without impacting hot-path performance.
//!
//! # Examples
//!
//! ```
//! use checksum::{Crc64, DispatchInfo, KernelIntrospect};
//!
//! // Platform-level info
//! let info = DispatchInfo::current();
//! println!("{info}");
//!
//! // Per-algorithm kernel selection
//! println!("CRC-64 backend: {}", Crc64::backend_name());
//! println!("CRC-64 @ 4KB: {}", Crc64::kernel_name_for_len(4096));
//! ```

use core::fmt;

/// Information about the current dispatch configuration.
///
/// This is a zero-allocation wrapper that provides a user-friendly view
/// of the detected CPU capabilities and selected microarchitecture tuning.
///
/// # Examples
///
/// ```
/// use checksum::DispatchInfo;
///
/// let info = DispatchInfo::current();
/// println!("{info}");
/// // Example output: "Caps(aarch64, [AES, PMULL, SHA256, ...]) (Apple M1-M3)"
/// ```
#[derive(Clone, Copy)]
pub struct DispatchInfo {
  platform: platform::Description,
}

impl DispatchInfo {
  /// Returns dispatch info for the current platform.
  ///
  /// This call is cached after the first invocation, so subsequent calls
  /// are essentially free (single atomic load).
  #[inline]
  #[must_use]
  pub fn current() -> Self {
    Self {
      platform: platform::describe(),
    }
  }

  /// Returns the platform description (CPU, features, microarch).
  #[inline]
  #[must_use]
  pub fn platform(&self) -> platform::Description {
    self.platform
  }
}

impl fmt::Display for DispatchInfo {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.platform)
  }
}

impl fmt::Debug for DispatchInfo {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("DispatchInfo")
      .field("platform", &format_args!("{}", self.platform))
      .finish()
  }
}

/// Returns the kernel name selected for a specific algorithm and buffer size.
///
/// This is useful for verifying size-based kernel transitions.
///
/// # Examples
///
/// ```
/// use checksum::{Crc64, kernel_for};
///
/// // Check kernel selection at different buffer sizes
/// let small = kernel_for::<Crc64>(64);
/// let large = kernel_for::<Crc64>(65536);
/// println!("Small buffers: {small}");
/// println!("Large buffers: {large}");
/// ```
#[inline]
#[must_use]
pub fn kernel_for<T: KernelIntrospect>(len: usize) -> &'static str {
  T::kernel_name_for_len(len)
}

/// Trait for types that support kernel introspection.
///
/// This trait is implemented for all CRC types, allowing generic
/// introspection of kernel selection.
pub trait KernelIntrospect {
  /// Returns the kernel name that would be selected for a buffer of `len` bytes.
  ///
  /// The returned string is architecture and size-class specific, e.g.:
  /// - `"aarch64/pmull-eor3-3way"` for large buffers on Apple Silicon
  /// - `"x86_64/vpclmul-4x512"` for large buffers on Zen4
  /// - `"portable/slice16"` on platforms without hardware acceleration
  fn kernel_name_for_len(len: usize) -> &'static str;

  /// Returns the currently selected backend name.
  ///
  /// This reflects the kernel that would be used for a representative
  /// buffer size (typically 1KB), unless a force mode is active.
  fn backend_name() -> &'static str;
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
