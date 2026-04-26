//! Authenticated encryption with associated data foundations.
//!
//! This module provides the typed AEAD surface for rscrypto:
//!
//! - AEAD-specific error types
//! - explicit nonce wrappers
//! - deterministic AES-GCM nonce sequencing when `aes-gcm` is enabled
//! - dispatch introspection
//! - the shared [`Aead`] trait re-export
//! - concrete AEAD implementations on the same typed surface
//!
//! # Quick Start
//!
//! ```rust
//! use rscrypto::{Aead, ChaCha20Poly1305, ChaCha20Poly1305Key, aead::Nonce96};
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
//! # Ok::<(), Box<dyn std::error::Error>>(())
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
#[cfg(any(
  feature = "aegis256",
  all(target_arch = "riscv64", any(feature = "aes-gcm", feature = "aes-gcm-siv"))
))]
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
#[cfg(feature = "aes-gcm")]
mod nonce_counter;
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
#[cfg(all(
  feature = "diag",
  target_arch = "aarch64",
  any(feature = "chacha20poly1305", feature = "xchacha20poly1305")
))]
pub use chacha20::diag_chacha20_xor_keystream_aarch64_neon;
// Re-export the per-backend ChaCha20 entry points for forced-kernel
// equivalence tests in `tests/aead_kernel_equivalence.rs`. Mirrors the
// `diag_compress_*` pattern used by Argon2.
#[cfg(all(feature = "diag", any(feature = "chacha20poly1305", feature = "xchacha20poly1305")))]
pub use chacha20::diag_chacha20_xor_keystream_portable;
#[cfg(all(
  feature = "diag",
  target_arch = "powerpc64",
  target_endian = "little",
  any(feature = "chacha20poly1305", feature = "xchacha20poly1305")
))]
pub use chacha20::diag_chacha20_xor_keystream_power_vsx;
#[cfg(all(
  feature = "diag",
  target_arch = "riscv64",
  any(feature = "chacha20poly1305", feature = "xchacha20poly1305")
))]
pub use chacha20::diag_chacha20_xor_keystream_riscv64_vector;
#[cfg(all(
  feature = "diag",
  target_arch = "s390x",
  any(feature = "chacha20poly1305", feature = "xchacha20poly1305")
))]
pub use chacha20::diag_chacha20_xor_keystream_s390x_vector;
#[cfg(all(
  feature = "diag",
  target_arch = "wasm32",
  any(feature = "chacha20poly1305", feature = "xchacha20poly1305")
))]
pub use chacha20::diag_chacha20_xor_keystream_wasm_simd128;
#[cfg(all(
  feature = "diag",
  target_arch = "x86_64",
  any(feature = "chacha20poly1305", feature = "xchacha20poly1305")
))]
pub use chacha20::{diag_chacha20_xor_keystream_x86_avx2, diag_chacha20_xor_keystream_x86_avx512};
#[cfg(feature = "chacha20poly1305")]
pub use chacha20poly1305::{ChaCha20Poly1305, ChaCha20Poly1305Key, ChaCha20Poly1305Tag};
#[cfg(feature = "aes-gcm")]
pub use nonce_counter::{NonceCounter, NonceCounterExhausted, NonceCounterSealError};
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

    impl crate::traits::ConstantTimeEq for $name {
      #[inline]
      fn ct_eq(&self, other: &Self) -> bool {
        crate::traits::ct::constant_time_eq(&self.0, &other.0)
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

      /// Generate a random nonce using the operating system's CSPRNG.
      ///
      /// # Panics
      ///
      /// Panics if the platform entropy source is unavailable.
      #[cfg(feature = "getrandom")]
      #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
      #[inline]
      #[must_use]
      pub fn random() -> Self {
        match Self::try_random() {
          Ok(value) => value,
          Err(e) => panic!("getrandom failed: {e}"),
        }
      }

      /// Try to generate a random nonce from the platform entropy source.
      ///
      /// # Errors
      ///
      /// Returns a getrandom error if the platform entropy source is unavailable.
      #[cfg(feature = "getrandom")]
      #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
      #[inline]
      pub fn try_random() -> Result<Self, getrandom::Error> {
        let mut bytes = [0u8; $len];
        getrandom::fill(&mut bytes).map(|()| Self(bytes))
      }
    }
  };
}

define_nonce_type!(
  Nonce96,
  12,
  "Explicit 96-bit nonce wrapper.

Pairs with `Aes256Gcm`, `Aes256GcmSiv`, and `ChaCha20Poly1305`."
);
define_nonce_type!(
  Nonce128,
  16,
  "Explicit 128-bit nonce wrapper.

Pairs with `AsconAead128`."
);
define_nonce_type!(
  Nonce192,
  24,
  "Explicit 192-bit nonce wrapper.

Pairs with `XChaCha20Poly1305`."
);
define_nonce_type!(
  Nonce256,
  32,
  "Explicit 256-bit nonce wrapper.

Pairs with `Aegis256`."
);

impl_hex_fmt!(Nonce96);
impl_hex_fmt!(Nonce128);
impl_hex_fmt!(Nonce192);
impl_hex_fmt!(Nonce256);
impl_serde_bytes!(Nonce96);
impl_serde_bytes!(Nonce128);
impl_serde_bytes!(Nonce192);
impl_serde_bytes!(Nonce256);

define_unit_error! {
  /// Combined-buffer length mismatch during AEAD sealing or opening.
  pub struct AeadBufferError;
  "buffer length mismatch"
}

/// Combined AEAD seal failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SealError {
  /// Combined input or output buffers have the wrong length.
  Buffer(AeadBufferError),
  /// Input exceeds the algorithm's supported length bound.
  TooLarge,
}

impl SealError {
  /// Construct a seal error for buffer-length mismatches.
  #[inline]
  #[must_use]
  pub const fn buffer() -> Self {
    Self::Buffer(AeadBufferError::new())
  }

  /// Construct a seal error for oversized inputs.
  #[inline]
  #[must_use]
  pub const fn too_large() -> Self {
    Self::TooLarge
  }
}

impl Default for SealError {
  #[inline]
  fn default() -> Self {
    Self::buffer()
  }
}

impl fmt::Display for SealError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Buffer(err) => err.fmt(f),
      Self::TooLarge => f.write_str("input exceeds the algorithm maximum length"),
    }
  }
}

impl core::error::Error for SealError {
  fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
    match self {
      Self::Buffer(err) => Some(err),
      Self::TooLarge => None,
    }
  }
}

impl From<AeadBufferError> for SealError {
  #[inline]
  fn from(value: AeadBufferError) -> Self {
    Self::Buffer(value)
  }
}

/// Combined AEAD open failure.
///
/// Length mismatches and authentication failures are kept distinct so callers
/// can fix their buffer management without guessing. Treat
/// [`OpenError::Verification`] as an opaque authentication failure at trust
/// boundaries; only the buffer-shape variants are intended for caller-visible
/// diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OpenError {
  /// Combined input or output buffers have the wrong length.
  Buffer(AeadBufferError),
  /// Input exceeds the algorithm's supported length bound.
  TooLarge,
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

  /// Construct an open error for oversized inputs.
  #[inline]
  #[must_use]
  pub const fn too_large() -> Self {
    Self::TooLarge
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
      Self::TooLarge => f.write_str("input exceeds the algorithm maximum length"),
      Self::Verification(err) => err.fmt(f),
    }
  }
}

impl core::error::Error for OpenError {
  fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
    match self {
      Self::Buffer(err) => Some(err),
      Self::TooLarge => None,
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

#[cfg_attr(
  not(any(
    feature = "aes-gcm",
    feature = "aes-gcm-siv",
    feature = "chacha20poly1305",
    feature = "xchacha20poly1305"
  )),
  allow(dead_code)
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LengthOverflow;

const _: () = assert!(usize::BITS <= u64::BITS);

#[cfg_attr(
  not(any(
    feature = "aes-gcm",
    feature = "aes-gcm-siv",
    feature = "chacha20poly1305",
    feature = "xchacha20poly1305"
  )),
  allow(dead_code)
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AeadByteLengths {
  aad: u64,
  text: u64,
}

impl AeadByteLengths {
  #[inline]
  pub(crate) const fn from_usize(aad_len: usize, text_len: usize) -> Self {
    Self {
      aad: aad_len as u64,
      text: text_len as u64,
    }
  }

  #[cfg_attr(
    not(any(feature = "chacha20poly1305", feature = "xchacha20poly1305")),
    allow(dead_code)
  )]
  #[inline]
  pub(crate) fn try_new(aad_len: usize, text_len: usize) -> Result<Self, LengthOverflow> {
    Ok(Self::from_usize(aad_len, text_len))
  }

  #[cfg_attr(
    not(any(feature = "chacha20poly1305", feature = "xchacha20poly1305")),
    allow(dead_code)
  )]
  #[inline]
  pub(crate) fn to_le_bytes_block(self) -> [u8; 16] {
    let mut block = [0u8; 16];
    block[0..8].copy_from_slice(&self.aad.to_le_bytes());
    block[8..16].copy_from_slice(&self.text.to_le_bytes());
    block
  }

  #[cfg_attr(not(feature = "aes-gcm-siv"), allow(dead_code))]
  #[inline]
  pub(crate) fn to_le_bits_block(self) -> [u8; 16] {
    let aad_bits = self.aad.strict_mul(8);
    let text_bits = self.text.strict_mul(8);
    let mut block = [0u8; 16];
    block[0..8].copy_from_slice(&aad_bits.to_le_bytes());
    block[8..16].copy_from_slice(&text_bits.to_le_bytes());
    block
  }

  #[cfg_attr(not(feature = "aes-gcm"), allow(dead_code))]
  #[inline]
  pub(crate) fn to_be_bits_block(self) -> [u8; 16] {
    let aad_bits = self.aad.strict_mul(8);
    let text_bits = self.text.strict_mul(8);
    let mut block = [0u8; 16];
    block[0..8].copy_from_slice(&aad_bits.to_be_bytes());
    block[8..16].copy_from_slice(&text_bits.to_be_bytes());
    block
  }
}

#[cfg_attr(
  not(any(
    feature = "aes-gcm",
    feature = "aes-gcm-siv",
    feature = "chacha20poly1305",
    feature = "xchacha20poly1305"
  )),
  allow(dead_code)
)]
#[inline]
pub(crate) fn try_length_as_u64(len: usize) -> Result<u64, LengthOverflow> {
  u64::try_from(len).map_err(|_| LengthOverflow)
}

#[cfg_attr(
  not(any(
    feature = "aes-gcm",
    feature = "aes-gcm-siv",
    feature = "chacha20poly1305",
    feature = "xchacha20poly1305"
  )),
  allow(dead_code)
)]
#[inline]
pub(crate) fn try_bounded_length_as_u64(len: usize, max: u64) -> Result<u64, LengthOverflow> {
  let len = try_length_as_u64(len)?;
  if len > max {
    return Err(LengthOverflow);
  }
  Ok(len)
}

#[cfg_attr(
  not(any(
    feature = "aes-gcm",
    feature = "aes-gcm-siv",
    feature = "chacha20poly1305",
    feature = "xchacha20poly1305"
  )),
  allow(dead_code)
)]
#[inline]
pub(crate) fn seal_bounded_length_as_u64(len: usize, max: u64) -> Result<u64, SealError> {
  try_bounded_length_as_u64(len, max).map_err(|_| SealError::too_large())
}

#[cfg_attr(
  not(any(
    feature = "aes-gcm",
    feature = "aes-gcm-siv",
    feature = "chacha20poly1305",
    feature = "xchacha20poly1305"
  )),
  allow(dead_code)
)]
#[inline]
pub(crate) fn open_bounded_length_as_u64(len: usize, max: u64) -> Result<u64, OpenError> {
  try_bounded_length_as_u64(len, max).map_err(|_| OpenError::too_large())
}

#[cfg(test)]
mod tests {
  use alloc::string::ToString;

  use super::{AeadBufferError, Nonce96, Nonce128, Nonce192, Nonce256, OpenError, SealError};
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

  #[test]
  fn aead_length_error_conventions_remain_stable() {
    assert_eq!(SealError::default(), SealError::buffer());
    assert_eq!(
      SealError::too_large().to_string(),
      "input exceeds the algorithm maximum length"
    );
    assert_eq!(
      OpenError::too_large().to_string(),
      "input exceeds the algorithm maximum length"
    );
  }
}
