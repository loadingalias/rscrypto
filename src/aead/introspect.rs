//! Backend dispatch introspection for AEAD primitives.
//!
//! This mirrors the checksum/hash introspection surfaces so AEAD dispatch can be
//! audited and benchmarked without reaching into private internals.
//!
//! # Examples
//!
//! ```
//! use rscrypto::aead::introspect::DispatchInfo;
//!
//! let info = DispatchInfo::current();
//! assert!(!format!("{info}").is_empty());
//! #[cfg(feature = "xchacha20poly1305")]
//! assert!(!rscrypto::aead::introspect::xchacha20poly1305_backend().is_empty());
//! ```

#[cfg(any(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305",
  feature = "aegis256"
))]
use crate::aead::targets::{AeadPrimitive, select_backend};
pub use crate::platform::DispatchInfo;

#[cfg(any(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305",
  feature = "aegis256"
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

/// Returns the backend selected for `Aes128Gcm` on the current machine.
#[cfg(feature = "aes-gcm")]
#[inline]
#[must_use]
pub fn aes128gcm_backend() -> &'static str {
  backend_for(AeadPrimitive::Aes128Gcm)
}

/// Returns the backend selected for `Aes256GcmSiv` on the current machine.
#[cfg(feature = "aes-gcm-siv")]
#[inline]
#[must_use]
pub fn aes256gcmsiv_backend() -> &'static str {
  backend_for(AeadPrimitive::Aes256GcmSiv)
}

/// Returns the backend selected for `Aes128GcmSiv` on the current machine.
#[cfg(feature = "aes-gcm-siv")]
#[inline]
#[must_use]
pub fn aes128gcmsiv_backend() -> &'static str {
  backend_for(AeadPrimitive::Aes128GcmSiv)
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
  match crate::platform::arch() {
    crate::platform::Arch::Wasm32 | crate::platform::Arch::Wasm64 => "wasm32/portable",
    _ => "portable",
  }
}
