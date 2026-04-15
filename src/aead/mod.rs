//! Authenticated encryption with associated data foundations.
//!
//! This module provides the typed AEAD surface for rscrypto:
//!
//! - AEAD-specific error types
//! - explicit nonce wrappers
//! - dispatch introspection
//! - the shared [`Aead`] trait re-export
//! - concrete AEAD implementations on the same typed surface
//!
//! # Quick Start
//!
//! ```rust
//! use rscrypto::{
//!   Aead, ChaCha20Poly1305, ChaCha20Poly1305Key,
//!   aead::{Nonce96, OpenError},
//! };
//!
//! let cipher = ChaCha20Poly1305::new(&ChaCha20Poly1305Key::from_bytes([0x11; 32]));
//! let nonce = Nonce96::from_bytes([0x22; Nonce96::LENGTH]);
//!
//! let mut sealed = [0u8; 4 + ChaCha20Poly1305::TAG_SIZE];
//! cipher.encrypt(&nonce, b"hdr", b"data", &mut sealed)?;
//!
//! let mut opened = [0u8; 4];
//! cipher.decrypt(&nonce, b"hdr", &sealed, &mut opened)?;
//! assert_eq!(&opened, b"data");
//! # Ok::<(), OpenError>(())
//! ```
//!
//! # Feature Selection
//!
//! ```toml
//! [dependencies]
//! # ChaCha20-Poly1305 only
//! rscrypto = { version = "0.1", default-features = false, features = ["chacha20poly1305"] }
//!
//! # All AEADs
//! rscrypto = { version = "0.1", default-features = false, features = ["aead"] }
//! ```
//!
//! # API Conventions
//!
//! - Every cipher uses a typed `*Key`, typed nonce wrapper, and typed `*Tag`.
//! - Combined-buffer helpers use `encrypt` / `decrypt`.
//! - Detached helpers use `encrypt_in_place` / `decrypt_in_place`.
//! - All AEADs implement the shared [`Aead`] trait with the same constructor and operation names.
//!
//! # Error Conventions
//!
//! - Buffer shape mistakes return [`AeadBufferError`].
//! - Combined open failures return [`OpenError`], which preserves whether the failure was a length
//!   mistake or an authentication failure.

use core::fmt;

pub use crate::traits::Aead;
use crate::traits::VerificationError;
#[cfg(feature = "aegis256")]
mod aegis256;
#[cfg(any(feature = "aes-gcm", feature = "aes-gcm-siv"))]
mod aes;
#[cfg(feature = "aes-gcm")]
mod aes256gcm;
#[cfg(feature = "aes-gcm-siv")]
mod aes256gcmsiv;
#[cfg(all(feature = "aegis256", not(any(target_arch = "s390x", target_arch = "riscv64"))))]
mod aes_round;
#[cfg(feature = "ascon-aead")]
mod ascon128;
#[cfg(any(feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
mod chacha20;
#[cfg(feature = "chacha20poly1305")]
mod chacha20poly1305;
#[cfg(feature = "aes-gcm")]
mod ghash;
#[cfg(feature = "diag")]
pub mod introspect;
#[cfg(any(feature = "chacha20poly1305", feature = "xchacha20poly1305"))]
mod poly1305;
#[cfg(any(feature = "aes-gcm", feature = "aes-gcm-siv"))]
mod polyval;
pub mod targets;
#[cfg(feature = "xchacha20poly1305")]
mod xchacha20poly1305;
#[cfg(feature = "aegis256")]
pub use aegis256::{Aegis256, Aegis256Key, Aegis256Tag};
#[cfg(feature = "aes-gcm")]
pub use aes256gcm::{Aes256Gcm, Aes256GcmKey, Aes256GcmTag};
#[cfg(feature = "aes-gcm-siv")]
pub use aes256gcmsiv::{Aes256GcmSiv, Aes256GcmSivKey, Aes256GcmSivTag};
#[cfg(feature = "ascon-aead")]
pub use ascon128::{AsconAead128, AsconAead128Key, AsconAead128Tag};
#[cfg(feature = "chacha20poly1305")]
pub use chacha20poly1305::{ChaCha20Poly1305, ChaCha20Poly1305Key, ChaCha20Poly1305Tag};
pub use targets::{AeadBackend, AeadPrimitive, BenchLane, lane_target_backend, select_backend};
#[cfg(feature = "xchacha20poly1305")]
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
        write!(f, "{}(", stringify!($name))?;
        crate::hex::fmt_hex_lower(&self.0, f)?;
        write!(f, ")")
      }
    }

    impl $name {
      #[doc = concat!(
                                                    "Construct a nonce by filling bytes from the provided closure.\n\n",
                                                    "```rust\n",
                                                    "use rscrypto::aead::",
                                                    stringify!($name),
                                                    ";\n\n",
                                                    "let nonce = ",
                                                    stringify!($name),
                                                    "::generate(|buf| buf.fill(0xA5));\n",
                                                    "assert_eq!(nonce.as_bytes(), &[0xA5; ",
                                                    stringify!($name),
                                                    "::LENGTH]);\n",
                                                    "```"
                                                  )]
      #[inline]
      #[must_use]
      pub fn generate(fill: impl FnOnce(&mut [u8; $len])) -> Self {
        let mut bytes = [0u8; $len];
        fill(&mut bytes);
        Self(bytes)
      }
    }
  };
}

define_nonce_type!(Nonce96, 12, "Explicit 96-bit nonce wrapper.");
define_nonce_type!(Nonce128, 16, "Explicit 128-bit nonce wrapper.");
define_nonce_type!(Nonce192, 24, "Explicit 192-bit nonce wrapper.");
define_nonce_type!(Nonce256, 32, "Explicit 256-bit nonce wrapper.");

impl_hex_fmt!(Nonce96);
impl_hex_fmt!(Nonce128);
impl_hex_fmt!(Nonce192);
impl_hex_fmt!(Nonce256);

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
  use super::{AeadBufferError, Nonce96, Nonce128, Nonce192, Nonce256, OpenError};
  use crate::traits::VerificationError;

  #[test]
  fn nonce_wrappers_round_trip() {
    let nonce96 = Nonce96::from_bytes([0x11; Nonce96::LENGTH]);
    let nonce128 = Nonce128::from_bytes([0x22; Nonce128::LENGTH]);
    let nonce192 = Nonce192::from_bytes([0x33; Nonce192::LENGTH]);

    assert_eq!(nonce96.to_bytes(), [0x11; Nonce96::LENGTH]);
    assert_eq!(nonce128.to_bytes(), [0x22; Nonce128::LENGTH]);
    assert_eq!(nonce192.to_bytes(), [0x33; Nonce192::LENGTH]);

    let nonce256 = Nonce256::from_bytes([0x44; Nonce256::LENGTH]);
    assert_eq!(nonce256.to_bytes(), [0x44; Nonce256::LENGTH]);
  }

  #[test]
  fn open_error_conversions_preserve_variant() {
    assert_eq!(OpenError::from(AeadBufferError::new()), OpenError::buffer());
    assert_eq!(OpenError::from(VerificationError::new()), OpenError::verification());
  }
}
