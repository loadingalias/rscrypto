//! Zeroizing owned wrappers for explicit secret extraction.

#[cfg(feature = "alloc")]
use alloc::{collections::TryReserveError, string::String, vec::Vec};
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

  /// Construct secret bytes in place with a fallible caller-provided filler.
  ///
  /// The destination is zero-initialized before `fill` runs. If `fill` returns
  /// an error after writing only part of the destination, the complete array is
  /// cleared before this method returns.
  #[inline]
  pub fn try_fill_with<E>(fill: impl FnOnce(&mut [u8; N]) -> Result<(), E>) -> Result<Self, E> {
    let mut secret = Self([0u8; N]);
    fill(&mut secret.0)?;
    Ok(secret)
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

/// Error constructing a [`SecretVec`] in place.
#[cfg(feature = "alloc")]
pub enum SecretVecConstructionError<E> {
  /// Reserving the requested initialized length failed.
  Allocation(TryReserveError),
  /// The caller-provided filler failed.
  Fill(E),
}

#[cfg(feature = "alloc")]
impl<E> fmt::Debug for SecretVecConstructionError<E> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Allocation(_) => f.write_str("Allocation(..)"),
      Self::Fill(_) => f.write_str("Fill(..)"),
    }
  }
}

#[cfg(feature = "alloc")]
impl<E> fmt::Display for SecretVecConstructionError<E> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Allocation(_) => f.write_str("secret-vector allocation failed"),
      Self::Fill(_) => f.write_str("secret-vector filler failed"),
    }
  }
}

#[cfg(feature = "alloc")]
impl<E> core::error::Error for SecretVecConstructionError<E> where E: core::error::Error + 'static {}

#[cfg(feature = "alloc")]
impl SecretVec {
  /// Protect an existing vector without copying or reallocating it.
  #[inline]
  #[must_use]
  pub const fn from_vec(bytes: Vec<u8>) -> Self {
    Self(bytes)
  }

  /// Allocate a zero-initialized secret vector and fill it in place.
  ///
  /// Allocation failure and filler failure remain distinct. If `fill` returns
  /// an error after writing only part of the destination, every byte in the
  /// initialized length is cleared before this method returns. Spare capacity
  /// outside that initialized length is not claimed to be cleared.
  #[inline]
  pub fn try_fill_with<E>(
    len: usize,
    fill: impl FnOnce(&mut [u8]) -> Result<(), E>,
  ) -> Result<Self, SecretVecConstructionError<E>> {
    let mut bytes = Vec::new();
    bytes
      .try_reserve_exact(len)
      .map_err(SecretVecConstructionError::Allocation)?;
    bytes.resize(len, 0);

    let mut secret = Self(bytes);
    fill(&mut secret.0).map_err(SecretVecConstructionError::Fill)?;
    Ok(secret)
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

/// Owned UTF-8 secret that zeroizes its initialized bytes on drop.
///
/// This owner preserves the allocation of an existing [`String`]. Borrowing is
/// explicit, formatting is redacted, and converting back to an ordinary string
/// transfers cleanup responsibility to the caller. Spare allocation capacity
/// outside the initialized string length is not claimed to be cleared.
#[cfg(feature = "alloc")]
pub struct SecretString(String);

#[cfg(feature = "alloc")]
impl SecretString {
  /// Protect an existing string without copying or reallocating it.
  #[inline]
  #[must_use]
  pub const fn from_string(value: String) -> Self {
    Self(value)
  }

  /// Borrow the protected UTF-8 text.
  #[inline]
  #[must_use]
  pub fn as_str(&self) -> &str {
    &self.0
  }

  /// Borrow the protected UTF-8 bytes.
  #[inline]
  #[must_use]
  pub fn as_bytes(&self) -> &[u8] {
    self.0.as_bytes()
  }

  /// Return the protected UTF-8 byte length.
  #[inline]
  #[must_use]
  pub fn len(&self) -> usize {
    self.0.len()
  }

  /// Return whether the protected string is empty.
  #[inline]
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.0.is_empty()
  }

  /// Extract the value into an ordinary string that will not zeroize on drop.
  ///
  /// The caller becomes responsible for clearing the transferred allocation.
  #[inline]
  #[must_use]
  pub fn into_unprotected_string(mut self) -> String {
    core::mem::take(&mut self.0)
  }
}

#[cfg(feature = "alloc")]
impl AsRef<str> for SecretString {
  #[inline]
  fn as_ref(&self) -> &str {
    self.as_str()
  }
}

#[cfg(feature = "alloc")]
impl AsRef<[u8]> for SecretString {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    self.as_bytes()
  }
}

#[cfg(feature = "alloc")]
impl fmt::Debug for SecretString {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("SecretString(****)")
  }
}

#[cfg(feature = "alloc")]
impl Drop for SecretString {
  fn drop(&mut self) {
    let mut bytes = core::mem::take(&mut self.0).into_bytes();
    ct::zeroize(&mut bytes);
  }
}

#[cfg(any(
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "aes-siv",
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
  feature = "aes-siv",
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
  feature = "aes-siv",
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
pub(crate) fn diag_zeroize_fixed_stack(input: [u8; 32]) -> u8 {
  let secret = SecretBytes::new(input);
  core::hint::black_box(secret.as_bytes()[0])
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[unsafe(no_mangle)]
#[inline(never)]
pub(crate) fn diag_zeroize_fixed_move(input: [u8; 32]) -> u8 {
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
pub(crate) fn diag_zeroize_early_return(input: [u8; 32], stop: bool) -> u8 {
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
pub(crate) fn diag_zeroize_variable_heap(input: Vec<u8>) -> usize {
  let secret = SecretVec::from_vec(input);
  core::hint::black_box(secret.len())
}

#[cfg(feature = "diag")]
#[doc(hidden)]
#[unsafe(no_mangle)]
#[inline(never)]
pub(crate) fn diag_zeroize_fixed_fill_error(value: u8) -> bool {
  SecretBytes::<32>::try_fill_with(|bytes| {
    bytes[..16].fill(core::hint::black_box(value));
    Err(())
  })
  .is_err()
}

#[cfg(all(feature = "diag", feature = "alloc"))]
#[doc(hidden)]
#[unsafe(no_mangle)]
#[inline(never)]
pub(crate) fn diag_zeroize_variable_fill_error(len: usize, value: u8) -> bool {
  SecretVec::try_fill_with(core::hint::black_box(len), |bytes| {
    let initialized = bytes.len() / 2;
    bytes[..initialized].fill(core::hint::black_box(value));
    Err(())
  })
  .is_err()
}

#[cfg(all(feature = "diag", feature = "alloc"))]
#[doc(hidden)]
#[unsafe(no_mangle)]
#[inline(never)]
pub(crate) fn diag_zeroize_secret_string(input: String) -> usize {
  let secret = SecretString::from_string(input);
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
    let protected = SecretVec::from_vec(alloc::vec![0x42, 0x24]);
    assert_eq!(protected.as_bytes(), [0x42, 0x24]);
    assert_eq!(alloc::format!("{protected:?}"), "SecretVec(****)");
    assert_eq!(protected.into_unprotected_vec(), [0x42, 0x24]);
  }

  #[test]
  fn secret_bytes_fills_in_place() {
    let protected = SecretBytes::<4>::try_fill_with(|bytes| {
      bytes.copy_from_slice(&[1, 2, 3, 4]);
      Ok::<(), ()>(())
    })
    .expect("successful fixed-size filler must construct a secret owner");
    assert_eq!(protected.as_bytes(), &[1, 2, 3, 4]);
  }

  #[test]
  fn secret_bytes_supports_zero_length_fill() {
    let protected = SecretBytes::<0>::try_fill_with(|bytes| {
      assert!(bytes.is_empty());
      Ok::<(), ()>(())
    })
    .expect("zero-length fixed-size filler must construct a secret owner");
    assert!(protected.as_bytes().is_empty());
  }

  #[test]
  fn secret_bytes_returns_exact_filler_error() {
    let err = SecretBytes::<4>::try_fill_with(|bytes| {
      bytes[..2].copy_from_slice(&[1, 2]);
      Err(17u8)
    })
    .expect_err("filler failure must be returned unchanged");
    assert_eq!(err, 17);
  }

  #[cfg(feature = "alloc")]
  #[test]
  fn secret_vec_preserves_existing_allocation() {
    let mut bytes = alloc::vec![1, 2, 3];
    bytes.reserve_exact(7);
    let pointer = bytes.as_ptr();
    let capacity = bytes.capacity();

    let protected = SecretVec::from_vec(bytes);
    assert_eq!(protected.as_bytes().as_ptr(), pointer);
    assert_eq!(protected.0.capacity(), capacity);

    let unprotected = protected.into_unprotected_vec();
    assert_eq!(unprotected.as_ptr(), pointer);
    assert_eq!(unprotected.capacity(), capacity);
  }

  #[cfg(feature = "alloc")]
  #[test]
  fn secret_vec_fill_distinguishes_failures() {
    let protected = SecretVec::try_fill_with(3, |bytes| {
      assert_eq!(bytes, &[0, 0, 0]);
      bytes.copy_from_slice(&[1, 2, 3]);
      Ok::<(), u8>(())
    })
    .expect("successful variable-size filler must construct a secret owner");
    assert_eq!(protected.as_bytes(), &[1, 2, 3]);

    let err = SecretVec::try_fill_with(4, |bytes| {
      bytes[..2].copy_from_slice(&[1, 2]);
      Err(23u8)
    })
    .expect_err("filler failure must remain distinct");
    assert!(matches!(err, SecretVecConstructionError::Fill(23)));

    let mut called = false;
    let err = SecretVec::try_fill_with(usize::MAX, |_| {
      called = true;
      Ok::<(), u8>(())
    })
    .expect_err("capacity overflow must remain distinct");
    assert!(matches!(err, SecretVecConstructionError::Allocation(_)));
    assert!(!called);
  }

  #[cfg(feature = "alloc")]
  #[test]
  fn secret_vec_supports_empty_fill() {
    let protected = SecretVec::try_fill_with(0, |bytes| {
      assert!(bytes.is_empty());
      Ok::<(), ()>(())
    })
    .expect("zero-length variable-size filler must construct a secret owner");
    assert!(protected.is_empty());
  }

  #[cfg(feature = "alloc")]
  #[test]
  fn secret_string_preserves_utf8_allocation_and_redacts_debug() {
    let mut value = alloc::string::String::from("s3crét");
    value.reserve_exact(11);
    let pointer = value.as_ptr();
    let capacity = value.capacity();

    let protected = SecretString::from_string(value);
    assert_eq!(protected.as_str(), "s3crét");
    assert_eq!(protected.as_bytes(), "s3crét".as_bytes());
    assert_eq!(protected.len(), "s3crét".len());
    assert_eq!(protected.as_bytes().as_ptr(), pointer);
    assert_eq!(protected.0.capacity(), capacity);
    assert_eq!(alloc::format!("{protected:?}"), "SecretString(****)");

    let unprotected = protected.into_unprotected_string();
    assert_eq!(unprotected.as_ptr(), pointer);
    assert_eq!(unprotected.capacity(), capacity);
  }

  #[cfg(feature = "alloc")]
  #[test]
  fn secret_string_supports_empty_utf8() {
    let protected = SecretString::from_string(alloc::string::String::new());
    assert!(protected.is_empty());
    assert_eq!(protected.as_str(), "");
  }
}
