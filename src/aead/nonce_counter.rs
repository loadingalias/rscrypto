//! Deterministic AES-GCM nonce sequencing.
//!
//! `NonceCounter<Aes256Gcm>` builds 96-bit nonces as:
//!
//! - 32-bit fixed prefix chosen by the caller
//! - 64-bit big-endian invocation counter
//!
//! This follows the deterministic IV shape from SP 800-38D and removes the
//! easiest nonce-reuse footgun from high-volume AES-GCM usage.
//!
//! ```rust
//! use rscrypto::{Aead, Aes256Gcm, Aes256GcmKey, aead::NonceCounter};
//!
//! let cipher = Aes256Gcm::new(&Aes256GcmKey::from_bytes([0x42; 32]));
//! let mut counter = NonceCounter::<Aes256Gcm>::new(*b"sess");
//!
//! let mut sealed = [0u8; 4 + Aes256Gcm::TAG_SIZE];
//! let nonce = counter.encrypt(&cipher, b"hdr", b"data", &mut sealed)?;
//!
//! let mut opened = [0u8; 4];
//! cipher.decrypt(&nonce, b"hdr", &sealed, &mut opened)?;
//! assert_eq!(&opened, b"data");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use core::{fmt, marker::PhantomData};

use super::{Aes256Gcm, Aes256GcmTag, Nonce96, SealError};

const FIXED_PREFIX_LEN: usize = 4;
const COUNTER_LEN: usize = 8;
const MAX_MESSAGES: u64 = 1u64 << 48;

define_unit_error! {
  /// AES-GCM nonce counter exhausted its deterministic invocation budget.
  pub struct NonceCounterExhausted;
  "AES-GCM nonce counter exhausted"
}

/// AES-GCM sealing failure from [`NonceCounter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NonceCounterSealError {
  /// No fresh nonce remains in the counter.
  Exhausted(NonceCounterExhausted),
  /// AES-GCM sealing itself failed.
  Seal(SealError),
}

impl NonceCounterSealError {
  /// Construct a nonce-exhaustion error.
  #[inline]
  #[must_use]
  pub const fn exhausted() -> Self {
    Self::Exhausted(NonceCounterExhausted::new())
  }

  /// Construct a sealing error.
  #[inline]
  #[must_use]
  pub const fn seal(err: SealError) -> Self {
    Self::Seal(err)
  }
}

impl fmt::Display for NonceCounterSealError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Exhausted(err) => err.fmt(f),
      Self::Seal(err) => err.fmt(f),
    }
  }
}

impl core::error::Error for NonceCounterSealError {
  fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
    match self {
      Self::Exhausted(err) => Some(err),
      Self::Seal(err) => Some(err),
    }
  }
}

impl From<NonceCounterExhausted> for NonceCounterSealError {
  #[inline]
  fn from(value: NonceCounterExhausted) -> Self {
    Self::Exhausted(value)
  }
}

impl From<SealError> for NonceCounterSealError {
  #[inline]
  fn from(value: SealError) -> Self {
    Self::Seal(value)
  }
}

/// Monotonic deterministic nonce generator for AES-GCM.
///
/// The counter is intentionally not `Clone` or `Copy`. One instance owns one
/// nonce stream. If you need restart-safe continuation, persist
/// [`next_counter`](Self::next_counter) and restore with
/// [`with_counter`](Self::with_counter).
pub struct NonceCounter<Cipher> {
  fixed_prefix: [u8; FIXED_PREFIX_LEN],
  next: u64,
  _cipher: PhantomData<fn() -> Cipher>,
}

impl NonceCounter<Aes256Gcm> {
  /// Fixed per-stream prefix length in bytes.
  pub const FIXED_PREFIX_LEN: usize = FIXED_PREFIX_LEN;

  /// Counter field length in bytes.
  pub const COUNTER_LEN: usize = COUNTER_LEN;

  /// Maximum deterministic AES-GCM invocations per key before rotation.
  pub const MAX_MESSAGES: u64 = MAX_MESSAGES;

  /// Start a fresh AES-GCM nonce stream with `fixed_prefix`.
  #[inline]
  #[must_use]
  pub const fn new(fixed_prefix: [u8; FIXED_PREFIX_LEN]) -> Self {
    Self {
      fixed_prefix,
      next: 0,
      _cipher: PhantomData,
    }
  }

  /// Resume an AES-GCM nonce stream from a persisted counter value.
  ///
  /// # Errors
  ///
  /// Returns [`NonceCounterExhausted`] when `next_counter >= MAX_MESSAGES`.
  #[inline]
  pub fn with_counter(fixed_prefix: [u8; FIXED_PREFIX_LEN], next_counter: u64) -> Result<Self, NonceCounterExhausted> {
    if next_counter >= Self::MAX_MESSAGES {
      return Err(NonceCounterExhausted::new());
    }

    Ok(Self {
      fixed_prefix,
      next: next_counter,
      _cipher: PhantomData,
    })
  }

  /// Return the fixed 32-bit prefix.
  #[inline]
  #[must_use]
  pub const fn fixed_prefix(&self) -> [u8; FIXED_PREFIX_LEN] {
    self.fixed_prefix
  }

  /// Return the next 64-bit invocation counter that will be issued.
  #[inline]
  #[must_use]
  pub const fn next_counter(&self) -> u64 {
    self.next
  }

  /// Return how many nonces have already been issued.
  #[inline]
  #[must_use]
  pub const fn issued(&self) -> u64 {
    self.next
  }

  /// Return how many deterministic AES-GCM invocations remain.
  #[inline]
  #[must_use]
  pub const fn remaining(&self) -> u64 {
    Self::MAX_MESSAGES - self.next
  }

  /// Issue the next fresh AES-GCM nonce.
  ///
  /// # Errors
  ///
  /// Returns [`NonceCounterExhausted`] when the counter reaches
  /// [`MAX_MESSAGES`](Self::MAX_MESSAGES).
  #[inline]
  pub fn next_nonce(&mut self) -> Result<Nonce96, NonceCounterExhausted> {
    if self.next >= Self::MAX_MESSAGES {
      return Err(NonceCounterExhausted::new());
    }

    let nonce = Self::build_nonce(self.fixed_prefix, self.next);
    self.next = self.next.strict_add(1);
    Ok(nonce)
  }

  /// Encrypt `buffer` in place with the next fresh nonce.
  ///
  /// The nonce is consumed before sealing so it is never reissued, even if
  /// the cipher returns a buffer or length error.
  #[inline]
  pub fn encrypt_in_place(
    &mut self,
    cipher: &Aes256Gcm,
    aad: &[u8],
    buffer: &mut [u8],
  ) -> Result<(Nonce96, Aes256GcmTag), NonceCounterSealError> {
    let nonce = self.next_nonce()?;
    let tag = cipher.encrypt_in_place(&nonce, aad, buffer)?;
    Ok((nonce, tag))
  }

  /// Encrypt `plaintext` into `out` with the next fresh nonce.
  ///
  /// The returned nonce must be transmitted alongside `out`.
  #[inline]
  pub fn encrypt(
    &mut self,
    cipher: &Aes256Gcm,
    aad: &[u8],
    plaintext: &[u8],
    out: &mut [u8],
  ) -> Result<Nonce96, NonceCounterSealError> {
    let nonce = self.next_nonce()?;
    cipher.encrypt(&nonce, aad, plaintext, out)?;
    Ok(nonce)
  }

  #[inline]
  fn build_nonce(fixed_prefix: [u8; FIXED_PREFIX_LEN], counter: u64) -> Nonce96 {
    let mut bytes = [0u8; Nonce96::LENGTH];
    bytes[..FIXED_PREFIX_LEN].copy_from_slice(&fixed_prefix);
    bytes[FIXED_PREFIX_LEN..].copy_from_slice(&counter.to_be_bytes());
    Nonce96::from_bytes(bytes)
  }
}

#[cfg(test)]
mod tests {
  use super::{Aes256Gcm, NonceCounter, NonceCounterSealError};
  use crate::{
    Aes256GcmKey,
    aead::{Nonce96, SealError},
  };

  #[test]
  fn aes_gcm_nonce_counter_formats_prefix_and_counter() {
    let mut counter = NonceCounter::<Aes256Gcm>::new(*b"conn");

    assert_eq!(
      counter.next_nonce().unwrap(),
      Nonce96::from_bytes([b'c', b'o', b'n', b'n', 0, 0, 0, 0, 0, 0, 0, 0])
    );
    assert_eq!(
      counter.next_nonce().unwrap(),
      Nonce96::from_bytes([b'c', b'o', b'n', b'n', 0, 0, 0, 0, 0, 0, 0, 1])
    );
    assert_eq!(counter.issued(), 2);
  }

  #[test]
  fn aes_gcm_nonce_counter_encrypt_round_trip() {
    let cipher = Aes256Gcm::new(&Aes256GcmKey::from_bytes([0x42; 32]));
    let mut counter = NonceCounter::<Aes256Gcm>::new(*b"sess");

    let mut sealed = [0u8; 4 + Aes256Gcm::TAG_SIZE];
    let nonce = counter.encrypt(&cipher, b"hdr", b"data", &mut sealed).unwrap();

    let mut opened = [0u8; 4];
    cipher.decrypt(&nonce, b"hdr", &sealed, &mut opened).unwrap();
    assert_eq!(&opened, b"data");
    assert_eq!(counter.next_counter(), 1);
  }

  #[test]
  fn aes_gcm_nonce_counter_consumes_nonce_on_seal_error() {
    let cipher = Aes256Gcm::new(&Aes256GcmKey::from_bytes([0x24; 32]));
    let mut counter = NonceCounter::<Aes256Gcm>::new(*b"bufr");
    let mut out = [0u8; 3];

    let err = counter.encrypt(&cipher, b"", b"data", &mut out).unwrap_err();
    assert_eq!(err, NonceCounterSealError::from(SealError::buffer()));
    assert_eq!(counter.next_counter(), 1);
  }

  #[test]
  fn aes_gcm_nonce_counter_exhausts_cleanly() {
    let mut counter =
      NonceCounter::<Aes256Gcm>::with_counter(*b"last", NonceCounter::<Aes256Gcm>::MAX_MESSAGES.strict_sub(1)).unwrap();

    assert!(counter.next_nonce().is_ok());
    assert_eq!(counter.remaining(), 0);
    assert!(counter.next_nonce().is_err());
  }
}
