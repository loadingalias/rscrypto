//! Native signing and verification traits.
//!
//! The trait surface is fallible-first and keeps algorithm/profile selection on
//! concrete types. RSA uses profile-bound wrapper types rather than letting a
//! raw key imply a signature scheme.

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

use crate::traits::VerificationError;

/// Fallible signer that returns a typed signature value.
pub trait TrySigner {
  /// Signature value returned by this signer.
  type Signature;

  /// Signing error.
  type Error;

  /// Sign `message`.
  ///
  /// # Errors
  ///
  /// Returns the implementation-specific signing error.
  #[must_use = "signature creation failure must be checked"]
  fn try_sign(&self, message: &[u8]) -> Result<Self::Signature, Self::Error>;
}

/// Verifier for a typed signature value.
pub trait Verifier<Sig: ?Sized> {
  /// Verify `signature` over `message`.
  ///
  /// # Errors
  ///
  /// Returns opaque [`VerificationError`] for any invalid signature.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  fn verify(&self, message: &[u8], signature: &Sig) -> Result<(), VerificationError>;
}

/// Fallible signer for algorithms whose signature length is runtime-sized.
pub trait TrySignerInto {
  /// Signing error.
  type Error;

  /// Return the required signature output length in bytes.
  #[must_use]
  fn signature_len(&self) -> usize;

  /// Sign `message` into `out`.
  ///
  /// # Errors
  ///
  /// Returns the implementation-specific signing error when `out` is the wrong
  /// length or signing fails.
  #[must_use = "signature creation failure must be checked"]
  fn try_sign_into(&self, message: &[u8], out: &mut [u8]) -> Result<(), Self::Error>;

  /// Allocate a signature buffer and sign `message` into it.
  ///
  /// # Errors
  ///
  /// Returns the implementation-specific signing error. The temporary output
  /// buffer is cleared before returning an error.
  #[cfg(feature = "alloc")]
  #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
  #[inline]
  fn try_sign_to_vec(&self, message: &[u8]) -> Result<Vec<u8>, Self::Error> {
    let mut out = vec![0u8; self.signature_len()];
    match self.try_sign_into(message, &mut out) {
      Ok(()) => Ok(out),
      Err(err) => {
        super::ct::zeroize(&mut out);
        Err(err)
      }
    }
  }
}
