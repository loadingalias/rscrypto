//! Backend dispatch introspection for AEAD primitives.
//!
//! This mirrors the checksum/hash introspection surfaces so AEAD dispatch can be
//! audited and benchmarked without reaching into private internals.
//!
//! # Examples
//!
//! ```
//! use rscrypto::aead::introspect::{DispatchInfo, xchacha20poly1305_backend};
//!
//! let info = DispatchInfo::current();
//! assert!(!format!("{info}").is_empty());
//! assert!(!xchacha20poly1305_backend().is_empty());
//! ```

#[cfg(any(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305",
  feature = "aegis256",
  feature = "ascon-aead"
))]
use crate::aead::targets::{AeadPrimitive, select_backend};
pub use crate::platform::DispatchInfo;

#[cfg(any(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305",
  feature = "aegis256",
  feature = "ascon-aead"
))]
#[inline]
fn backend_for(primitive: AeadPrimitive) -> &'static str {
  select_backend(primitive, crate::platform::arch(), crate::platform::caps()).name()
}

/// Returns the backend selected for `Aes256Gcm` on the current machine.
#[cfg(feature = "aes-gcm")]
#[inline]
#[must_use]
pub fn aes256gcm_backend() -> &'static str {
  backend_for(AeadPrimitive::Aes256Gcm)
}

/// Returns the backend selected for `Aes256GcmSiv` on the current machine.
#[cfg(feature = "aes-gcm-siv")]
#[inline]
#[must_use]
pub fn aes256gcmsiv_backend() -> &'static str {
  backend_for(AeadPrimitive::Aes256GcmSiv)
}

/// Returns the backend selected for `ChaCha20Poly1305` on the current machine.
#[cfg(feature = "chacha20poly1305")]
#[inline]
#[must_use]
pub fn chacha20poly1305_backend() -> &'static str {
  backend_for(AeadPrimitive::ChaCha20Poly1305)
}

/// Returns the backend selected for `XChaCha20Poly1305` on the current machine.
#[cfg(feature = "xchacha20poly1305")]
#[inline]
#[must_use]
pub fn xchacha20poly1305_backend() -> &'static str {
  backend_for(AeadPrimitive::XChaCha20Poly1305)
}

/// Returns the backend selected for `Aegis256` on the current machine.
#[cfg(feature = "aegis256")]
#[inline]
#[must_use]
pub fn aegis256_backend() -> &'static str {
  backend_for(AeadPrimitive::Aegis256)
}

/// Returns the backend selected for `AsconAead128` on the current machine.
#[cfg(feature = "ascon-aead")]
#[inline]
#[must_use]
pub fn ascon_aead128_backend() -> &'static str {
  backend_for(AeadPrimitive::AsconAead128)
}

#[cfg(test)]
mod tests {
  use super::DispatchInfo;

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
    #[cfg(feature = "xchacha20poly1305")]
    assert!(!super::xchacha20poly1305_backend().is_empty());
    #[cfg(feature = "chacha20poly1305")]
    assert!(!super::chacha20poly1305_backend().is_empty());
    #[cfg(feature = "aes-gcm")]
    assert!(!super::aes256gcm_backend().is_empty());
    #[cfg(feature = "aes-gcm-siv")]
    assert!(!super::aes256gcmsiv_backend().is_empty());
    #[cfg(feature = "aegis256")]
    assert!(!super::aegis256_backend().is_empty());
    #[cfg(feature = "ascon-aead")]
    assert!(!super::ascon_aead128_backend().is_empty());
  }
}
