//! Error types for cryptographic operations.
//!
//! Minimal, timing-safe error types designed to prevent information leakage.
//! Individual crates may define additional errors as needed.

use core::fmt;

/// Verification failed.
///
/// Returned when cryptographic verification fails (MAC tags, AEAD tags,
/// signatures). Intentionally opaque to prevent timing side-channels.
///
/// # Examples
///
/// ```
/// use traits::VerificationError;
///
/// fn verify(computed: &[u8; 32], expected: &[u8; 32]) -> Result<(), VerificationError> {
///   // Real code: use constant-time comparison
///   if computed == expected {
///     Ok(())
///   } else {
///     Err(VerificationError::new())
///   }
/// }
///
/// let a = [0u8; 32];
/// let b = [1u8; 32];
/// assert!(verify(&a, &b).is_err());
/// ```
///
/// # Security
///
/// This error provides no details about the failure to prevent timing
/// side-channels. The underlying verification should use constant-time
/// comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct VerificationError;

impl VerificationError {
  /// Create a new verification error.
  ///
  /// This is the only way to construct this error from outside the crate,
  /// ensuring forward compatibility if fields are added in the future.
  #[inline]
  #[must_use]
  pub const fn new() -> Self {
    Self
  }
}

impl Default for VerificationError {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl fmt::Display for VerificationError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("verification failed")
  }
}

impl core::error::Error for VerificationError {}

#[cfg(test)]
mod tests {
  extern crate alloc;

  use alloc::{format, string::ToString};
  use core::hash::{Hash, Hasher};

  use super::*;

  // A minimal hasher for testing Hash impl
  struct TestHasher(u64);

  impl Hasher for TestHasher {
    fn finish(&self) -> u64 {
      self.0
    }
    fn write(&mut self, bytes: &[u8]) {
      for &b in bytes {
        self.0 = self.0.wrapping_mul(31).wrapping_add(b as u64);
      }
    }
  }

  #[test]
  fn display_message() {
    assert_eq!(VerificationError::new().to_string(), "verification failed");
  }

  #[test]
  fn debug_impl() {
    let dbg = format!("{:?}", VerificationError::new());
    assert_eq!(dbg, "VerificationError");
  }

  #[test]
  fn is_copy() {
    let e = VerificationError::new();
    let e2 = e; // Copy
    let e3 = e; // Still valid
    assert_eq!(e2, e3);
  }

  #[test]
  fn is_clone() {
    let e = VerificationError::new();
    #[allow(clippy::clone_on_copy)]
    let cloned = e.clone();
    assert_eq!(e, cloned);
  }

  #[test]
  fn equality() {
    let a = VerificationError::new();
    let b = VerificationError::new();
    assert_eq!(a, b);
    assert!(a == b); // PartialEq
  }

  #[test]
  fn hash_consistent() {
    fn hash_one<T: Hash>(t: &T) -> u64 {
      let mut h = TestHasher(0);
      t.hash(&mut h);
      h.finish()
    }

    let a = VerificationError::new();
    let b = VerificationError::new();
    assert_eq!(hash_one(&a), hash_one(&b));
  }

  #[test]
  fn result_ok_path() {
    fn verify_match() -> Result<(), VerificationError> {
      Ok(())
    }
    assert!(verify_match().is_ok());
  }

  #[test]
  fn result_err_path() {
    fn verify_mismatch() -> Result<(), VerificationError> {
      Err(VerificationError::new())
    }
    let err = verify_mismatch().expect_err("verify_mismatch must return VerificationError");
    assert_eq!(err, VerificationError::new());
  }

  #[test]
  fn error_in_result_unwrap_err() {
    fn returns_err() -> Result<(), VerificationError> {
      Err(VerificationError::new())
    }
    let err = returns_err().unwrap_err();
    assert_eq!(err.to_string(), "verification failed");
  }

  #[test]
  fn trait_bounds() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    fn assert_unpin<T: Unpin>() {}

    assert_send::<VerificationError>();
    assert_sync::<VerificationError>();
    assert_unpin::<VerificationError>();
  }

  #[test]
  fn error_trait_impl() {
    use core::error::Error;

    fn assert_error<T: core::error::Error>() {}
    assert_error::<VerificationError>();

    let err = VerificationError::new();
    assert!(err.source().is_none());
  }

  #[test]
  fn default_impl() {
    let err: VerificationError = Default::default();
    assert_eq!(err, VerificationError::new());
  }

  #[test]
  fn size_is_zero() {
    assert_eq!(core::mem::size_of::<VerificationError>(), 0);
  }
}
