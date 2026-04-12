//! Internal Ed25519 hash wiring.
//!
//! Ed25519 expands the 32-byte secret key with SHA-512, then splits the digest
//! into the clamped secret scalar bytes and the nonce prefix. This module keeps
//! that flow explicit so later sign / verify code can reuse it without
//! smearing byte-level logic across point arithmetic.

use core::fmt;

use super::{
  Ed25519SecretKey,
  constants::SECRET_KEY_LENGTH,
  point::ExtendedPoint,
  scalar::{Scalar, clamp_secret_scalar, decode_words_le},
};
use crate::{hashes::crypto::Sha512, traits::ct};

/// Expanded secret-key material derived from SHA-512(secret_key).
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct ExpandedSecret {
  scalar_bytes: [u8; SECRET_KEY_LENGTH],
  nonce_prefix: [u8; SECRET_KEY_LENGTH],
}

impl ExpandedSecret {
  /// Expand a 32-byte Ed25519 secret key into clamped scalar bytes and nonce prefix.
  #[must_use]
  pub(crate) fn from_secret_key(secret: &Ed25519SecretKey) -> Self {
    let mut digest = Sha512::digest(secret.as_bytes());
    let (scalar_source, prefix_source) = digest.as_slice().split_at(SECRET_KEY_LENGTH);

    let mut scalar_bytes = [0u8; SECRET_KEY_LENGTH];
    let mut nonce_prefix = [0u8; SECRET_KEY_LENGTH];

    for (dst, src) in scalar_bytes.iter_mut().zip(scalar_source.iter().copied()) {
      *dst = src;
    }
    for (dst, src) in nonce_prefix.iter_mut().zip(prefix_source.iter().copied()) {
      *dst = src;
    }

    ct::zeroize(&mut digest);
    clamp_secret_scalar(&mut scalar_bytes);

    Self {
      scalar_bytes,
      nonce_prefix,
    }
  }

  /// Return the clamped scalar bytes.
  #[inline]
  #[must_use]
  pub(crate) const fn scalar_bytes(&self) -> &[u8; SECRET_KEY_LENGTH] {
    &self.scalar_bytes
  }

  /// Decode the clamped scalar bytes into the portable limb layout.
  #[inline]
  #[must_use]
  pub(crate) fn scalar_words(&self) -> Scalar {
    decode_words_le(&self.scalar_bytes)
  }

  /// Return the nonce prefix bytes used by signing.
  #[inline]
  #[must_use]
  pub(crate) const fn nonce_prefix(&self) -> &[u8; SECRET_KEY_LENGTH] {
    &self.nonce_prefix
  }

  /// Derive the Ed25519 public key bytes from the expanded secret scalar.
  #[must_use]
  pub(crate) fn public_key_bytes(&self) -> [u8; SECRET_KEY_LENGTH] {
    ExtendedPoint::scalar_mul_basepoint(&self.scalar_bytes)
      .to_bytes()
      .unwrap_or_default()
  }
}

impl fmt::Debug for ExpandedSecret {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("ExpandedSecret").finish_non_exhaustive()
  }
}

impl Drop for ExpandedSecret {
  fn drop(&mut self) {
    ct::zeroize(&mut self.scalar_bytes);
    ct::zeroize(&mut self.nonce_prefix);
  }
}

#[cfg(test)]
mod tests {
  use alloc::format;

  use super::ExpandedSecret;
  use crate::{auth::ed25519::Ed25519SecretKey, hashes::crypto::Sha512};

  #[test]
  fn expanded_secret_splits_sha512_digest_into_scalar_and_prefix() {
    let secret = Ed25519SecretKey::from_bytes([0x42; Ed25519SecretKey::LENGTH]);
    let expanded = ExpandedSecret::from_secret_key(&secret);
    let digest = Sha512::digest(secret.as_bytes());
    let (scalar_source, prefix_source) = digest.as_slice().split_at(Ed25519SecretKey::LENGTH);
    let mut expected_scalar = [0u8; Ed25519SecretKey::LENGTH];

    for (dst, src) in expected_scalar.iter_mut().zip(scalar_source.iter().copied()) {
      *dst = src;
    }
    crate::auth::ed25519::scalar::clamp_secret_scalar(&mut expected_scalar);

    assert_eq!(expanded.scalar_bytes(), &expected_scalar);
    assert_ne!(expanded.scalar_bytes()[0] & 0b0000_0111, 0b0000_0111);
    assert_eq!(expanded.scalar_bytes()[31] & 0b0100_0000, 0b0100_0000);
    assert_eq!(expanded.scalar_bytes()[31] & 0b1000_0000, 0);
    assert_eq!(&expanded.nonce_prefix()[..], prefix_source);
  }

  #[test]
  fn expanded_secret_decodes_scalar_words_from_clamped_bytes() {
    let secret = Ed25519SecretKey::from_bytes([0xA5; Ed25519SecretKey::LENGTH]);
    let expanded = ExpandedSecret::from_secret_key(&secret);
    let words = expanded.scalar_words();

    assert_eq!(words.len(), 4);
    assert_ne!(words, [0u64; 4]);
  }

  #[test]
  fn expanded_secret_debug_is_redacted() {
    let secret = Ed25519SecretKey::from_bytes([0x11; Ed25519SecretKey::LENGTH]);
    let expanded = ExpandedSecret::from_secret_key(&secret);

    assert_eq!(format!("{expanded:?}"), "ExpandedSecret { .. }");
  }

  #[test]
  fn expanded_secret_derives_nonzero_public_key_bytes() {
    let secret = Ed25519SecretKey::from_bytes([0x23; Ed25519SecretKey::LENGTH]);
    let expanded = ExpandedSecret::from_secret_key(&secret);
    let public = expanded.public_key_bytes();

    assert_ne!(public, [0u8; Ed25519SecretKey::LENGTH]);
  }
}
