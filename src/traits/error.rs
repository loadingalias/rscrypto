//! Error types for cryptographic operations.
//!
//! Minimal, opaque error types designed to avoid failure-detail leakage.
//! Individual crates may define additional errors as needed.

define_unit_error! {
  /// Verification failed.
  ///
  /// Returned when cryptographic verification fails (MAC tags, AEAD tags,
  /// signatures). Intentionally opaque so callers cannot distinguish internal
  /// failure causes.
  ///
  /// # Examples
  ///
  /// ```
  /// # #[cfg(feature = "hmac")]
  /// # {
  /// use rscrypto::{HmacSha256, HmacSha256Tag, Mac};
  ///
  /// let expected = HmacSha256Tag::from_bytes([0u8; HmacSha256Tag::LENGTH]);
  /// assert!(HmacSha256::verify_tag(b"key", b"message", &expected).is_err());
  /// # }
  /// ```
  ///
  /// # Security
  ///
  /// This error provides no details about the failure. Treat it as a generic
  /// authentication failure and avoid mapping it to finer-grained protocol
  /// responses that would recreate an oracle. Opacity does not by itself make
  /// success and failure take identical time. The underlying verification
  /// should use a sealed comparison decision, with any generated-code timing
  /// claim bounded by `ct.toml` and matching release evidence.
  #[non_exhaustive]
  pub struct VerificationError;
  "verification failed"
}

#[cfg(test)]
mod tests {

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
