//! Zeroizing owned wrappers for explicit secret extraction.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::fmt;

use crate::traits::ct;

/// Owned secret bytes that zeroize on drop.
///
/// This is the explicit escape hatch for APIs that must hand secret material
/// to external code. Borrowing via `as_bytes()` should stay the default.
pub struct SecretBytes<const N: usize>([u8; N]);

impl<const N: usize> SecretBytes<N> {
  /// Wrapped length in bytes.
  pub const LENGTH: usize = N;

  /// Wrap raw secret bytes.
  #[inline]
  #[must_use]
  pub const fn new(bytes: [u8; N]) -> Self {
    Self(bytes)
  }

  /// Borrow the wrapped bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; N] {
    &self.0
  }

  /// Explicitly extract the raw bytes.
  ///
  /// This copies the secret into a plain array and zeroizes the wrapper's
  /// backing storage before it drops.
  #[inline]
  #[must_use]
  pub fn expose(mut self) -> [u8; N] {
    let bytes = self.0;
    // Belt-and-braces: clear the wrapper storage before `Drop` runs its final wipe.
    ct::zeroize(&mut self.0);
    bytes
  }
}

impl<const N: usize> From<[u8; N]> for SecretBytes<N> {
  #[inline]
  fn from(value: [u8; N]) -> Self {
    Self::new(value)
  }
}

impl<const N: usize> AsRef<[u8]> for SecretBytes<N> {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

impl<const N: usize> fmt::Debug for SecretBytes<N> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("SecretBytes(****)")
  }
}

impl<const N: usize> Drop for SecretBytes<N> {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

/// Owned variable-length secret bytes that zeroize on drop.
///
/// Borrow the contents with [`as_bytes`](Self::as_bytes). Converting back to an
/// ordinary allocation requires the explicitly named
/// [`into_unprotected_vec`](Self::into_unprotected_vec) operation.
#[cfg(feature = "alloc")]
pub struct SecretVec(Vec<u8>);

#[cfg(feature = "alloc")]
impl SecretVec {
  #[inline]
  #[must_use]
  #[cfg(any(feature = "rsa", feature = "diag", test))]
  pub(crate) const fn new(bytes: Vec<u8>) -> Self {
    Self(bytes)
  }

  /// Borrow the protected bytes.
  #[inline]
  #[must_use]
  pub fn as_bytes(&self) -> &[u8] {
    &self.0
  }

  /// Return the protected byte length.
  #[inline]
  #[must_use]
  pub fn len(&self) -> usize {
    self.0.len()
  }

  /// Return whether the protected allocation is empty.
  #[inline]
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.0.is_empty()
  }

  /// Clear the protected bytes without releasing the allocation.
  #[inline(always)]
  pub fn clear(&mut self) {
    ct::zeroize(&mut self.0);
  }

  /// Extract the bytes into an ordinary vector that will not zeroize on drop.
  ///
  /// Use this only at an integration boundary that cannot borrow the protected
  /// bytes. The caller becomes responsible for clearing every resulting copy.
  #[inline]
  #[must_use]
  pub fn into_unprotected_vec(mut self) -> Vec<u8> {
    core::mem::take(&mut self.0)
  }
}

#[cfg(feature = "alloc")]
impl AsRef<[u8]> for SecretVec {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    self.as_bytes()
  }
}

#[cfg(feature = "alloc")]
impl core::ops::Deref for SecretVec {
  type Target = [u8];

  #[inline]
  fn deref(&self) -> &Self::Target {
    self.as_bytes()
  }
}

#[cfg(feature = "alloc")]
impl fmt::Debug for SecretVec {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("SecretVec(****)")
  }
}

#[cfg(feature = "alloc")]
impl Drop for SecretVec {
  fn drop(&mut self) {
    self.clear();
  }
}

#[cfg(any(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305",
  feature = "aegis256",
  feature = "ascon-aead",
  all(feature = "phc-strings", any(feature = "argon2", feature = "scrypt")),
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "ml-kem",
  feature = "poly1305",
  feature = "x25519"
))]
pub(crate) struct ZeroizingBytes<const N: usize>([u8; N]);

#[cfg(any(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305",
  feature = "aegis256",
  feature = "ascon-aead",
  all(feature = "phc-strings", any(feature = "argon2", feature = "scrypt")),
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "ml-kem",
  feature = "poly1305",
  feature = "x25519"
))]
impl<const N: usize> ZeroizingBytes<N> {
  #[inline]
  #[cfg(any(feature = "serde-secrets", feature = "ecdsa-p256", feature = "ecdsa-p384"))]
  pub(crate) const fn new(bytes: [u8; N]) -> Self {
    Self(bytes)
  }

  #[inline]
  pub(crate) const fn zeroed() -> Self {
    Self([0u8; N])
  }

  #[inline]
  pub(crate) const fn as_array(&self) -> &[u8; N] {
    &self.0
  }

  #[inline]
  pub(crate) fn as_mut_array(&mut self) -> &mut [u8; N] {
    &mut self.0
  }
}

#[cfg(any(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "chacha20poly1305",
  feature = "xchacha20poly1305",
  feature = "aegis256",
  feature = "ascon-aead",
  all(feature = "phc-strings", any(feature = "argon2", feature = "scrypt")),
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "ml-kem",
  feature = "poly1305",
  feature = "x25519"
))]
impl<const N: usize> Drop for ZeroizingBytes<N> {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[unsafe(no_mangle)]
#[inline(never)]
pub fn diag_zeroize_fixed_stack(input: [u8; 32]) -> u8 {
  let secret = SecretBytes::new(input);
  core::hint::black_box(secret.as_bytes()[0])
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[unsafe(no_mangle)]
#[inline(never)]
pub fn diag_zeroize_fixed_move(input: [u8; 32]) -> u8 {
  let secret = SecretBytes::new(input);
  let mut exposed = secret.expose();
  let output = core::hint::black_box(exposed[0]);
  ct::zeroize(&mut exposed);
  output
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[unsafe(no_mangle)]
#[inline(never)]
pub fn diag_zeroize_early_return(input: [u8; 32], stop: bool) -> u8 {
  let secret = SecretBytes::new(input);
  if core::hint::black_box(stop) {
    return 0;
  }
  core::hint::black_box(secret.as_bytes()[0])
}

#[cfg(all(feature = "diag", feature = "alloc"))]
#[doc(hidden)]
#[unsafe(no_mangle)]
#[inline(never)]
pub fn diag_zeroize_variable_heap(input: Vec<u8>) -> usize {
  let secret = SecretVec::new(input);
  core::hint::black_box(secret.len())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn secret_bytes_debug_masks_contents() {
    assert_eq!(alloc::format!("{:?}", SecretBytes::new([0x42; 4])), "SecretBytes(****)");
  }

  #[cfg(feature = "alloc")]
  #[test]
  fn secret_vec_requires_explicit_unprotected_extraction() {
    let protected = SecretVec::new(alloc::vec![0x42, 0x24]);
    assert_eq!(protected.as_bytes(), [0x42, 0x24]);
    assert_eq!(alloc::format!("{protected:?}"), "SecretVec(****)");
    assert_eq!(protected.into_unprotected_vec(), [0x42, 0x24]);
  }
}
