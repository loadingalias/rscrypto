//! Message authentication code traits.
//!
//! This trait mirrors [`crate::Digest`]: keyed construction, streaming updates,
//! idempotent finalize, and reset support.

use core::fmt::Debug;

use crate::traits::VerificationError;

/// Message authentication code producing a fixed-size tag.
///
/// This trait is intended for fixed-size algorithms like HMAC-SHA256.
///
/// Each implementation owns a semantic tag type whose [`Eq`] implementation
/// defines verification behavior. rscrypto's built-in MACs compare their
/// fixed-size tag owners without content-dependent exits. External
/// implementations must provide the same property; the trait cannot prove it.
pub trait Mac: Clone {
  /// Tag size in bytes.
  const TAG_SIZE: usize;

  /// The authentication tag type.
  ///
  /// This should be a semantic owner type, not a raw byte array. Its [`Eq`]
  /// implementation is used by the default verification methods and must not
  /// exit based on secret tag contents.
  type Tag: Copy + Eq + Debug + AsRef<[u8]>;

  /// Construct a new MAC state with `key`.
  #[must_use]
  fn new(key: &[u8]) -> Self;

  /// Update the MAC with additional data.
  fn update(&mut self, data: &[u8]);

  /// Update the MAC with multiple non-contiguous buffers.
  #[inline]
  fn update_vectored(&mut self, bufs: &[&[u8]]) {
    for buf in bufs {
      self.update(buf);
    }
  }

  /// Update the MAC with `std::io::IoSlice` buffers.
  #[cfg(feature = "std")]
  #[inline]
  fn update_io_slices(&mut self, bufs: &[std::io::IoSlice<'_>]) {
    for buf in bufs {
      self.update(buf);
    }
  }

  /// Finalize and return the authentication tag.
  ///
  /// This method does not consume the MAC state, allowing further updates if
  /// needed.
  #[must_use]
  fn finalize(&self) -> Self::Tag;

  /// Reset the MAC state to the keyed initial state.
  fn reset(&mut self);

  /// Compute the tag of `data` in one shot.
  #[inline]
  #[must_use]
  fn mac(key: &[u8], data: &[u8]) -> Self::Tag {
    let mut mac = Self::new(key);
    mac.update(data);
    mac.finalize()
  }

  /// Finalize and return the tag as a `Vec<u8>`.
  #[cfg(feature = "alloc")]
  #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
  #[inline]
  #[must_use]
  fn finalize_to_vec(&self) -> alloc::vec::Vec<u8> {
    self.finalize().as_ref().to_vec()
  }

  /// Compute the tag of `data` in one shot, returning a `Vec<u8>`.
  #[cfg(feature = "alloc")]
  #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
  #[inline]
  #[must_use]
  fn mac_to_vec(key: &[u8], data: &[u8]) -> alloc::vec::Vec<u8> {
    Self::mac(key, data).as_ref().to_vec()
  }

  /// Verify `expected` against the current tag using the tag owner's equality.
  #[inline]
  #[must_use = "MAC verification must be checked; a dropped Result silently accepts a forged tag"]
  fn verify(&self, expected: &Self::Tag) -> Result<(), VerificationError> {
    if self.finalize() == *expected {
      Ok(())
    } else {
      Err(VerificationError::new())
    }
  }

  /// Compute and verify a tag in one shot using the tag owner's equality.
  #[inline]
  #[must_use = "MAC verification must be checked; a dropped Result silently accepts a forged tag"]
  fn verify_tag(key: &[u8], data: &[u8], expected: &Self::Tag) -> Result<(), VerificationError> {
    if Self::mac(key, data) == *expected {
      Ok(())
    } else {
      Err(VerificationError::new())
    }
  }
}
