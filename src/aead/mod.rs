//! Authenticated encryption with associated data foundations.
//!
//! This module intentionally starts small:
//!
//! - AEAD-specific error types
//! - explicit nonce wrappers
//! - lane and backend planning for the AEAD rollout
//! - dispatch introspection
//! - the shared [`Aead`] trait re-export
//!
//! Concrete AEAD algorithms will land here on top of the same typed surface.

use core::fmt;

pub use crate::traits::Aead;
use crate::traits::VerificationError;
mod aes;
mod aes256gcmsiv;
mod chacha20;
mod chacha20poly1305;
pub mod introspect;
mod poly1305;
mod polyval;
pub mod targets;
mod xchacha20poly1305;
pub use aes256gcmsiv::{Aes256GcmSiv, Aes256GcmSivKey, Aes256GcmSivTag};
pub use chacha20poly1305::{ChaCha20Poly1305, ChaCha20Poly1305Key, ChaCha20Poly1305Tag};
pub use targets::{AeadBackend, AeadPrimitive, BenchLane, lane_target_backend, select_backend};
pub use xchacha20poly1305::{XChaCha20Poly1305, XChaCha20Poly1305Key, XChaCha20Poly1305Tag};

macro_rules! define_nonce_type {
  ($name:ident, $len:expr, $doc:literal) => {
    #[doc = $doc]
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct $name([u8; Self::LENGTH]);

    impl $name {
      /// Nonce length in bytes.
      pub const LENGTH: usize = $len;

      /// Construct a typed nonce from raw bytes.
      #[inline]
      #[must_use]
      pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
        Self(bytes)
      }

      /// Return the raw nonce bytes.
      #[inline]
      #[must_use]
      pub const fn to_bytes(self) -> [u8; Self::LENGTH] {
        self.0
      }

      /// Borrow the raw nonce bytes.
      #[inline]
      #[must_use]
      pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
        &self.0
      }
    }

    impl Default for $name {
      #[inline]
      fn default() -> Self {
        Self([0u8; Self::LENGTH])
      }
    }

    impl AsRef<[u8]> for $name {
      #[inline]
      fn as_ref(&self) -> &[u8] {
        &self.0
      }
    }

    impl fmt::Debug for $name {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple(stringify!($name)).field(&self.0).finish()
      }
    }
  };
}

define_nonce_type!(Nonce96, 12, "Explicit 96-bit nonce wrapper.");
define_nonce_type!(Nonce128, 16, "Explicit 128-bit nonce wrapper.");
define_nonce_type!(Nonce192, 24, "Explicit 192-bit nonce wrapper.");

/// Combined-buffer length mismatch during AEAD sealing or opening.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AeadBufferError;

impl AeadBufferError {
  /// Construct a new AEAD buffer error.
  #[inline]
  #[must_use]
  pub const fn new() -> Self {
    Self
  }
}

impl Default for AeadBufferError {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl fmt::Display for AeadBufferError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("buffer length mismatch")
  }
}

impl core::error::Error for AeadBufferError {}

/// Combined AEAD open failure.
///
/// Length mismatches and authentication failures are kept distinct so callers
/// can fix their buffer management without guessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenError {
  /// Combined input or output buffers have the wrong length.
  Buffer(AeadBufferError),
  /// Authentication failed.
  Verification(VerificationError),
}

impl OpenError {
  /// Construct an open error for buffer-length mismatches.
  #[inline]
  #[must_use]
  pub const fn buffer() -> Self {
    Self::Buffer(AeadBufferError::new())
  }

  /// Construct an open error for authentication failures.
  #[inline]
  #[must_use]
  pub const fn verification() -> Self {
    Self::Verification(VerificationError::new())
  }
}

impl Default for OpenError {
  #[inline]
  fn default() -> Self {
    Self::buffer()
  }
}

impl fmt::Display for OpenError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Buffer(err) => err.fmt(f),
      Self::Verification(err) => err.fmt(f),
    }
  }
}

impl core::error::Error for OpenError {
  fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
    match self {
      Self::Buffer(err) => Some(err),
      Self::Verification(err) => Some(err),
    }
  }
}

impl From<AeadBufferError> for OpenError {
  #[inline]
  fn from(value: AeadBufferError) -> Self {
    Self::Buffer(value)
  }
}

impl From<VerificationError> for OpenError {
  #[inline]
  fn from(value: VerificationError) -> Self {
    Self::Verification(value)
  }
}

#[cfg(test)]
mod tests {
  use super::{AeadBufferError, Nonce96, Nonce128, Nonce192, OpenError};
  use crate::traits::VerificationError;

  #[test]
  fn nonce_wrappers_round_trip() {
    let nonce96 = Nonce96::from_bytes([0x11; Nonce96::LENGTH]);
    let nonce128 = Nonce128::from_bytes([0x22; Nonce128::LENGTH]);
    let nonce192 = Nonce192::from_bytes([0x33; Nonce192::LENGTH]);

    assert_eq!(nonce96.to_bytes(), [0x11; Nonce96::LENGTH]);
    assert_eq!(nonce128.to_bytes(), [0x22; Nonce128::LENGTH]);
    assert_eq!(nonce192.to_bytes(), [0x33; Nonce192::LENGTH]);
  }

  #[test]
  fn open_error_conversions_preserve_variant() {
    assert_eq!(OpenError::from(AeadBufferError::new()), OpenError::buffer());
    assert_eq!(OpenError::from(VerificationError::new()), OpenError::verification());
  }
}
