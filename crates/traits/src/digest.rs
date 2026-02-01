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

  /// Wrap a reader to compute digest transparently during I/O.
  ///
  /// # Example
  ///
  /// ```rust,ignore
  /// use hashes::crypto::blake3::Blake3;
  /// use std::fs::File;
  ///
  /// let file = File::open("data.bin")?;
  /// let mut reader = Blake3::reader(file);
  /// std::io::copy(&mut reader, &mut std::io::sink())?;
  /// println!("Digest: {:?}", reader.digest());
  /// ```
  #[cfg(feature = "std")]
  #[inline]
  #[must_use]
  fn reader<R>(inner: R) -> crate::io::DigestReader<R, Self>
  where
    Self: Sized,
  {
    crate::io::DigestReader::new(inner)
  }

  /// Wrap a writer to compute digest transparently during I/O.
  ///
  /// # Example
  ///
  /// ```rust,ignore
  /// use hashes::crypto::blake3::Blake3;
  /// use std::fs::File;
  ///
  /// let file = File::create("output.bin")?;
  /// let mut writer = Blake3::writer(file);
  /// writer.write_all(b"hello world")?;
  /// let (file, digest) = writer.into_parts();
  /// println!("Digest: {:?}", digest);
  /// ```
  #[cfg(feature = "std")]
  #[inline]
  #[must_use]
  fn writer<W>(inner: W) -> crate::io::DigestWriter<W, Self>
  where
    Self: Sized,
  {
    crate::io::DigestWriter::new(inner)
  }
}

// Sealed trait implementations for I/O support
#[cfg(feature = "std")]
impl<T: Digest> crate::io::SealedMarker for T {}

#[cfg(feature = "std")]
impl<T: Digest> crate::io::Hashable for T {
  type Output = T::Output;

  #[inline(always)]
  fn new_hasher() -> Self {
    T::new()
  }

  #[inline(always)]
  fn update(&mut self, data: &[u8]) {
    T::update(self, data);
  }

  #[inline(always)]
  fn finalize(&self) -> Self::Output {
    T::finalize(self)
  }
}
