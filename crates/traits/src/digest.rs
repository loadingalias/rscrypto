//! Cryptographic digest traits.
//!
//! This trait is deliberately shaped like [`crate::Checksum`]: streaming updates,
//! idempotent finalize, and reset support.

use core::fmt::Debug;

/// Cryptographic hash function producing a fixed-size digest.
///
/// This trait is intended for algorithms like SHA-256 and BLAKE3 (hash mode).
pub trait Digest: Clone + Default {
  /// Output size in bytes.
  const OUTPUT_SIZE: usize;

  /// The digest output type.
  ///
  /// Typically `[u8; N]`.
  type Output: Copy + Eq + Debug;

  /// Create a new hasher in its initial state.
  #[must_use]
  fn new() -> Self;

  /// Update the hasher with additional data.
  fn update(&mut self, data: &[u8]);

  /// Update the hasher with multiple non-contiguous buffers.
  #[inline]
  fn update_vectored(&mut self, bufs: &[&[u8]]) {
    for buf in bufs {
      self.update(buf);
    }
  }

  /// Update the hasher with `std::io::IoSlice` buffers.
  #[cfg(feature = "std")]
  #[inline]
  fn update_io_slices(&mut self, bufs: &[std::io::IoSlice<'_>]) {
    for buf in bufs {
      self.update(buf);
    }
  }

  /// Finalize and return the digest.
  ///
  /// This method does not consume the hasher, allowing further updates if needed.
  #[must_use]
  fn finalize(&self) -> Self::Output;

  /// Reset the hasher to its initial state.
  fn reset(&mut self);

  /// Compute the digest of data in one shot.
  #[inline]
  #[must_use]
  fn digest(data: &[u8]) -> Self::Output {
    let mut h = Self::new();
    h.update(data);
    h.finalize()
  }

  /// Compute the digest of multiple buffers in one shot.
  #[inline]
  #[must_use]
  fn digest_vectored(bufs: &[&[u8]]) -> Self::Output {
    let mut h = Self::new();
    h.update_vectored(bufs);
    h.finalize()
  }

  /// Compute the digest of `std::io::IoSlice` buffers in one shot.
  #[cfg(feature = "std")]
  #[inline]
  #[must_use]
  fn digest_io_slices(bufs: &[std::io::IoSlice<'_>]) -> Self::Output {
    let mut h = Self::new();
    h.update_io_slices(bufs);
    h.finalize()
  }
}
