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
  /// ```rust
  /// # use traits::Digest;
  /// # #[derive(Clone, Default)]
  /// # struct SumDigest(u8);
  /// # impl Digest for SumDigest {
  /// #   const OUTPUT_SIZE: usize = 4;
  /// #   type Output = [u8; 4];
  /// #   fn new() -> Self { Self(0) }
  /// #   fn update(&mut self, data: &[u8]) {
  /// #     self.0 = data.iter().fold(self.0, |acc, &b| acc.wrapping_add(b));
  /// #   }
  /// #   fn finalize(&self) -> Self::Output { [self.0; 4] }
  /// #   fn reset(&mut self) { self.0 = 0; }
  /// # }
  /// # use std::io::Cursor;
  ///
  /// let mut reader = SumDigest::reader(Cursor::new(b"abc".to_vec()));
  /// std::io::copy(&mut reader, &mut std::io::sink())?;
  /// assert_eq!(
  ///   reader.digest(),
  ///   [b'a'.wrapping_add(b'b').wrapping_add(b'c'); 4]
  /// );
  /// # Ok::<(), std::io::Error>(())
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
  /// ```rust
  /// # use traits::Digest;
  /// # #[derive(Clone, Default)]
  /// # struct SumDigest(u8);
  /// # impl Digest for SumDigest {
  /// #   const OUTPUT_SIZE: usize = 4;
  /// #   type Output = [u8; 4];
  /// #   fn new() -> Self { Self(0) }
  /// #   fn update(&mut self, data: &[u8]) {
  /// #     self.0 = data.iter().fold(self.0, |acc, &b| acc.wrapping_add(b));
  /// #   }
  /// #   fn finalize(&self) -> Self::Output { [self.0; 4] }
  /// #   fn reset(&mut self) { self.0 = 0; }
  /// # }
  /// # use std::io::Write;
  ///
  /// let mut writer = SumDigest::writer(Vec::new());
  /// writer.write_all(b"hello world")?;
  /// let (out, digest) = writer.into_parts();
  /// assert_eq!(out, b"hello world".to_vec());
  /// assert_eq!(
  ///   digest,
  ///   [b"hello world"
  ///     .iter()
  ///     .fold(0u8, |acc, &b| acc.wrapping_add(b)); 4]
  /// );
  /// # Ok::<(), std::io::Error>(())
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
