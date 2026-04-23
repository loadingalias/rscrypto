//! Zeroizing owned wrappers for explicit secret extraction.

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

impl<const N: usize> PartialEq for SecretBytes<N> {
  #[inline]
  fn eq(&self, other: &Self) -> bool {
    ct::constant_time_eq(&self.0, &other.0)
  }
}

impl<const N: usize> Eq for SecretBytes<N> {}

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
