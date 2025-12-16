//! Error types for cryptographic operations.
//!
//! Minimal, timing-safe error types. Individual crates define additional
//! errors as needed.

use core::fmt;

/// Verification failed.
///
/// Returned when cryptographic verification fails (MAC tags, AEAD tags,
/// signatures). Intentionally opaque to prevent timing side-channels.
///
/// # Security
///
/// This error provides no details about the failure. The underlying
/// verification uses constant-time comparison.
///
/// # Example
///
/// ```
/// use traits::VerificationError;
///
/// fn verify(computed: &[u8; 32], expected: &[u8; 32]) -> Result<(), VerificationError> {
///   // Real code: use constant-time comparison
///   if computed == expected {
///     Ok(())
///   } else {
///     Err(VerificationError)
///   }
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VerificationError;

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
    assert_eq!(VerificationError.to_string(), "verification failed");
  }

  #[test]
  fn debug_impl() {
    let dbg = format!("{:?}", VerificationError);
    assert_eq!(dbg, "VerificationError");
  }

  #[test]
  fn is_copy() {
    let e = VerificationError;
    let e2 = e; // Copy
    let e3 = e; // Still valid
    assert_eq!(e2, e3);
  }

  #[test]
  fn is_clone() {
    let e = VerificationError;
    #[allow(clippy::clone_on_copy)]
    let cloned = e.clone();
    assert_eq!(e, cloned);
  }

  #[test]
  fn equality() {
    let a = VerificationError;
    let b = VerificationError;
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

    let a = VerificationError;
    let b = VerificationError;
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
      Err(VerificationError)
    }
    let result = verify_mismatch();
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), VerificationError);
  }

  #[test]
  fn error_in_result_unwrap_err() {
    fn returns_err() -> Result<(), VerificationError> {
      Err(VerificationError)
    }
    let err = returns_err().unwrap_err();
    assert_eq!(err.to_string(), "verification failed");
  }

  // Compile-time trait bound checks
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

    // Test Error trait methods
    let err = VerificationError;
    assert!(err.source().is_none()); // No source error
  }

  #[test]
  fn size_is_zero() {
    // VerificationError is a unit struct, should be zero-sized
    assert_eq!(core::mem::size_of::<VerificationError>(), 0);
  }
}
