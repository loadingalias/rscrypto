//! Backend dispatch introspection for AEAD primitives.
//!
//! This mirrors the checksum/hash introspection surfaces so AEAD dispatch can be
//! audited and benchmarked without reaching into private internals.
//!
//! # Examples
//!
//! ```
//! use rscrypto::aead::{
//!   AeadPrimitive,
//!   introspect::{DispatchInfo, backend_for},
//! };
//!
//! let info = DispatchInfo::current();
//! println!("{info}");
//! println!(
//!   "xchacha20-poly1305: {}",
//!   backend_for(AeadPrimitive::XChaCha20Poly1305)
//! );
//! ```

use core::fmt;

use crate::aead::{AeadPrimitive, select_backend};

/// Zero-allocation view of the current AEAD dispatch environment.
#[derive(Clone, Copy)]
pub struct DispatchInfo {
  platform: crate::platform::Description,
}

impl DispatchInfo {
  /// Returns dispatch info for the current platform.
  #[inline]
  #[must_use]
  pub fn current() -> Self {
    Self {
      platform: crate::platform::describe(),
    }
  }

  /// Returns the platform description used to drive AEAD backend selection.
  #[inline]
  #[must_use]
  pub fn platform(&self) -> crate::platform::Description {
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

/// Returns the backend label selected for `primitive` on the current machine.
#[inline]
#[must_use]
pub fn backend_for(primitive: AeadPrimitive) -> &'static str {
  select_backend(primitive, crate::platform::arch(), crate::platform::caps()).name()
}

#[cfg(test)]
mod tests {
  use super::{DispatchInfo, backend_for};
  use crate::aead::AeadPrimitive;

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
  fn backend_for_current_platform_is_not_empty() {
    assert!(!backend_for(AeadPrimitive::XChaCha20Poly1305).is_empty());
    assert!(!backend_for(AeadPrimitive::ChaCha20Poly1305).is_empty());
  }
}
