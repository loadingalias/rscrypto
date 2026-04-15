//! Message authentication code traits.
//!
//! This trait mirrors [`crate::Digest`]: keyed construction, streaming updates,
//! idempotent finalize, and reset support.

use core::fmt::Debug;

use crate::traits::{VerificationError, ct};

/// Message authentication code producing a fixed-size tag.
///
/// This trait is intended for fixed-size algorithms like HMAC-SHA256.
///
/// # Examples
///
/// ```
/// use rscrypto::{HmacSha256, Mac};
///
/// let key = b"secret-key";
///
/// // One-shot.
/// let tag = HmacSha256::mac(key, b"hello world");
///
/// // Streaming.
/// let mut mac = HmacSha256::new(key);
/// mac.update(b"hello ");
/// mac.update(b"world");
/// assert_eq!(mac.finalize(), tag);
///
/// // Verify.
/// assert!(HmacSha256::verify_tag(key, b"hello world", &tag).is_ok());
/// ```
pub trait Mac: Clone {
  /// Tag size in bytes.
  const TAG_SIZE: usize;

  /// The authentication tag type.
  ///
  /// Typically `[u8; N]`.
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

  /// Verify `expected` against the current tag in constant time.
  #[inline]
  fn verify(&self, expected: &Self::Tag) -> Result<(), VerificationError> {
    if ct::constant_time_eq(self.finalize().as_ref(), expected.as_ref()) {
      Ok(())
    } else {
      Err(VerificationError::new())
    }
  }

  /// Compute and verify a tag in one shot.
  #[inline]
  fn verify_tag(key: &[u8], data: &[u8], expected: &Self::Tag) -> Result<(), VerificationError> {
    if ct::constant_time_eq(Self::mac(key, data).as_ref(), expected.as_ref()) {
      Ok(())
    } else {
      Err(VerificationError::new())
    }
  }
}
