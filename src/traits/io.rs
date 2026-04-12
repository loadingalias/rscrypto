//! I/O adapter support for hashing algorithms.
//!
//! This module provides sealed traits that enable generic I/O adapter implementations
//! for [`Checksum`](crate::Checksum) and [`Digest`](crate::Digest) types.
//!
//! # Design
//!
//! The sealed trait pattern ensures:
//! - New algorithms automatically get I/O support
//! - External types cannot implement these traits (preventing coherence issues)
//! - Future API extensions don't break existing code
//!
//! # Example
//!
//! ```rust
//! # use rscrypto::traits::Checksum;
//! # #[derive(Clone, Default)]
//! # struct Sum(u32);
//! # impl Checksum for Sum {
//! #   const OUTPUT_SIZE: usize = 4;
//! #   type Output = u32;
//! #   fn new() -> Self { Self(0) }
//! #   fn with_initial(initial: Self::Output) -> Self { Self(initial) }
//! #   fn update(&mut self, data: &[u8]) {
//! #     self.0 = data.iter().fold(self.0, |acc, &b| acc.wrapping_add(u32::from(b)));
//! #   }
//! #   fn finalize(&self) -> Self::Output { self.0 }
//! #   fn reset(&mut self) { self.0 = 0; }
//! # }
//! # use std::io::Cursor;
//! let mut reader = Sum::reader(Cursor::new(b"abc".to_vec()));
//! std::io::copy(&mut reader, &mut std::io::sink())?;
//! assert_eq!(
//!   reader.checksum(),
//!   u32::from(b'a') + u32::from(b'b') + u32::from(b'c')
//! );
//! # Ok::<(), std::io::Error>(())
//! ```

/// Sealed trait marker - not implementable outside rscrypto.
///
/// This module prevents external implementations of [`Hashable`](crate::traits::io::Hashable),
/// allowing us to extend the trait with new methods without breaking changes.
mod private {
  /// Sealed trait marker.
  pub trait Sealed {}
}

// Internal re-export for use by sibling modules (checksum, digest).
pub(crate) use private::Sealed as SealedMarker;

/// Trait for types that can be used with I/O adapters.
///
/// This is implemented automatically for all [`Checksum`](crate::Checksum) and
/// [`Digest`](crate::Digest) types. It cannot be implemented manually.
///
/// # Stability
///
/// This trait is sealed - new methods may be added in minor versions.
pub trait Hashable: private::Sealed {
  /// The output type (e.g., `u32` for CRC32, `[u8; 32]` for SHA-256).
  type Output: Copy + core::fmt::Debug;

  /// Create a new hasher in its initial state.
  fn new_hasher() -> Self;

  /// Update the hasher with data.
  fn update(&mut self, data: &[u8]);

  /// Finalize and return the hash/checksum.
  fn finalize(&self) -> Self::Output;
}

/// Marker trait for checksum algorithms.
///
/// This is automatically implemented for all types implementing [`Checksum`](crate::Checksum).
/// It provides the `checksum()` method alias for `finalize()`.
pub trait ChecksumMarker: private::Sealed {
  type Output: Copy + core::fmt::Debug;

  fn checksum(&self) -> Self::Output;
}

/// Marker trait for digest algorithms.
///
/// This is automatically implemented for all types implementing [`Digest`](crate::Digest).
/// It provides the `digest()` method alias for `finalize()`.
pub trait DigestMarker: private::Sealed {
  type Output: Copy + core::fmt::Debug;

  fn digest(&self) -> Self::Output;
}

#[cfg(feature = "std")]
#[inline]
fn read_and_update<R>(inner: &mut R, buf: &mut [u8], mut on_data: impl FnMut(&[u8])) -> std::io::Result<usize>
where
  R: std::io::Read,
{
  let n = inner.read(buf)?;
  if let Some(data) = buf.get(..n) {
    on_data(data);
  }
  Ok(n)
}

#[cfg(feature = "std")]
#[inline]
fn read_vectored_and_update<R>(
  inner: &mut R,
  bufs: &mut [std::io::IoSliceMut<'_>],
  mut on_data: impl FnMut(&[u8]),
) -> std::io::Result<usize>
where
  R: std::io::Read,
{
  let n = inner.read_vectored(bufs)?;
  let mut remaining = n;
  for buf in bufs {
    if remaining == 0 {
      break;
    }
    let to_hash = remaining.min(buf.len());
    if to_hash == 0 {
      continue;
    }
    if let Some(data) = buf.get(..to_hash) {
      on_data(data);
    }
    remaining = remaining.strict_sub(to_hash);
  }
  Ok(n)
}

#[cfg(feature = "std")]
#[inline]
fn write_and_update<W>(inner: &mut W, buf: &[u8], mut on_data: impl FnMut(&[u8])) -> std::io::Result<usize>
where
  W: std::io::Write,
{
  let n = inner.write(buf)?;
  if let Some(data) = buf.get(..n) {
    on_data(data);
  }
  Ok(n)
}

#[cfg(feature = "std")]
#[inline]
fn write_vectored_and_update<W>(
  inner: &mut W,
  bufs: &[std::io::IoSlice<'_>],
  mut on_data: impl FnMut(&[u8]),
) -> std::io::Result<usize>
where
  W: std::io::Write,
{
  let n = inner.write_vectored(bufs)?;
  let mut remaining = n;
  for buf in bufs {
    if remaining == 0 {
      break;
    }
    let to_hash = remaining.min(buf.len());
    if to_hash == 0 {
      continue;
    }
    if let Some(data) = buf.get(..to_hash) {
      on_data(data);
    }
    remaining = remaining.strict_sub(to_hash);
  }
  Ok(n)
}

// ─────────────────────────────────────────────────────────────────────────────
// Checksum I/O Adapters
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps a [`Read`](std::io::Read) and computes a checksum transparently.
///
/// All reads from this type pass through to the inner reader while
/// updating the checksum with the actual bytes read (handling short reads).
///
/// # Type Parameters
///
/// - `R`: The inner reader type
/// - `C`: The checksum algorithm type (e.g., `Crc32C`)
///
/// # Example
///
/// ```rust
/// # use rscrypto::traits::Checksum;
/// # #[derive(Clone, Default)]
/// # struct Sum(u32);
/// # impl Checksum for Sum {
/// #   const OUTPUT_SIZE: usize = 4;
/// #   type Output = u32;
/// #   fn new() -> Self { Self(0) }
/// #   fn with_initial(initial: Self::Output) -> Self { Self(initial) }
/// #   fn update(&mut self, data: &[u8]) {
/// #     self.0 = data.iter().fold(self.0, |acc, &b| acc.wrapping_add(u32::from(b)));
/// #   }
/// #   fn finalize(&self) -> Self::Output { self.0 }
/// #   fn reset(&mut self) { self.0 = 0; }
/// # }
/// # use std::io::Cursor;
/// let mut reader = Sum::reader(Cursor::new(b"abc".to_vec()));
/// std::io::copy(&mut reader, &mut std::io::sink())?;
/// assert_eq!(
///   reader.checksum(),
///   u32::from(b'a') + u32::from(b'b') + u32::from(b'c')
/// );
/// # Ok::<(), std::io::Error>(())
/// ```
#[cfg(feature = "std")]
#[derive(Clone)]
pub struct ChecksumReader<R, C: crate::Checksum> {
  inner: R,
  hasher: C,
}

#[cfg(feature = "std")]
impl<R: core::fmt::Debug, C: crate::Checksum + core::fmt::Debug> core::fmt::Debug for ChecksumReader<R, C> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("ChecksumReader")
      .field("inner", &self.inner)
      .field("hasher", &self.hasher)
      .finish()
  }
}

#[cfg(feature = "std")]
impl<R, C: crate::Checksum> ChecksumReader<R, C> {
  /// Create a new reader wrapper with the default initial state.
  #[inline]
  #[must_use]
  pub fn new(inner: R) -> Self {
    Self {
      inner,
      hasher: C::new(),
    }
  }

  /// Create a new reader wrapper with a custom initial state.
  ///
  /// Useful for resuming a checksum computation from a known state.
  #[inline]
  #[must_use]
  pub fn with_initial(inner: R, initial: C::Output) -> Self {
    Self {
      inner,
      hasher: C::with_initial(initial),
    }
  }

  /// Get the current checksum value.
  ///
  /// This does not consume the reader or finalize the hasher -
  /// further reads will continue updating the checksum.
  #[inline]
  #[must_use]
  pub fn checksum(&self) -> C::Output {
    self.hasher.finalize()
  }

  /// Get a mutable reference to the underlying hasher.
  ///
  /// This allows advanced use cases like manual state manipulation.
  #[inline]
  pub fn hasher_mut(&mut self) -> &mut C {
    &mut self.hasher
  }

  /// Unwrap this `ChecksumReader`, returning the inner reader and the final checksum.
  #[inline]
  pub fn into_parts(self) -> (R, C::Output) {
    (self.inner, self.hasher.finalize())
  }

  /// Unwrap this `ChecksumReader`, returning the inner reader and discarding the checksum.
  #[inline]
  pub fn into_inner(self) -> R {
    self.inner
  }

  /// Get a reference to the inner reader.
  #[inline]
  pub fn inner(&self) -> &R {
    &self.inner
  }

  /// Get a mutable reference to the inner reader.
  #[inline]
  pub fn inner_mut(&mut self) -> &mut R {
    &mut self.inner
  }
}

#[cfg(feature = "std")]
impl<R: std::io::Read, C: crate::Checksum> std::io::Read for ChecksumReader<R, C> {
  #[inline]
  fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
    read_and_update(&mut self.inner, buf, |data| self.hasher.update(data))
  }

  #[inline]
  fn read_vectored(&mut self, bufs: &mut [std::io::IoSliceMut<'_>]) -> std::io::Result<usize> {
    read_vectored_and_update(&mut self.inner, bufs, |data| self.hasher.update(data))
  }
}

/// Wraps a [`Write`](std::io::Write) and computes a checksum transparently.
///
/// All writes to this type pass through to the inner writer while
/// updating the checksum with the bytes being written.
///
/// # Partial Write Semantics
///
/// The checksum is updated with only the bytes the inner writer reports as
/// successfully written. Short writes do not hash unwritten bytes.
///
/// # Type Parameters
///
/// - `W`: The inner writer type
/// - `C`: The checksum algorithm type (e.g., `Crc32C`)
///
/// # Example
///
/// ```rust
/// # use rscrypto::traits::Checksum;
/// # #[derive(Clone, Default)]
/// # struct Sum(u32);
/// # impl Checksum for Sum {
/// #   const OUTPUT_SIZE: usize = 4;
/// #   type Output = u32;
/// #   fn new() -> Self { Self(0) }
/// #   fn with_initial(initial: Self::Output) -> Self { Self(initial) }
/// #   fn update(&mut self, data: &[u8]) {
/// #     self.0 = data.iter().fold(self.0, |acc, &b| acc.wrapping_add(u32::from(b)));
/// #   }
/// #   fn finalize(&self) -> Self::Output { self.0 }
/// #   fn reset(&mut self) { self.0 = 0; }
/// # }
/// # use std::io::Write;
/// let mut writer = Sum::writer(Vec::new());
/// writer.write_all(b"hello world")?;
/// let (out, checksum) = writer.into_parts();
/// assert_eq!(out, b"hello world".to_vec());
/// assert_eq!(
///   checksum,
///   b"hello world"
///     .iter()
///     .fold(0u32, |acc, &b| acc.wrapping_add(u32::from(b)))
/// );
/// # Ok::<(), std::io::Error>(())
/// ```
#[cfg(feature = "std")]
#[derive(Clone)]
pub struct ChecksumWriter<W, C: crate::Checksum> {
  inner: W,
  hasher: C,
}

#[cfg(feature = "std")]
impl<W: core::fmt::Debug, C: crate::Checksum + core::fmt::Debug> core::fmt::Debug for ChecksumWriter<W, C> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("ChecksumWriter")
      .field("inner", &self.inner)
      .field("hasher", &self.hasher)
      .finish()
  }
}

#[cfg(feature = "std")]
impl<W, C: crate::Checksum> ChecksumWriter<W, C> {
  /// Create a new writer wrapper with the default initial state.
  #[inline]
  #[must_use]
  pub fn new(inner: W) -> Self {
    Self {
      inner,
      hasher: C::new(),
    }
  }

  /// Create a new writer wrapper with a custom initial state.
  #[inline]
  #[must_use]
  pub fn with_initial(inner: W, initial: C::Output) -> Self {
    Self {
      inner,
      hasher: C::with_initial(initial),
    }
  }

  /// Get the current checksum value.
  #[inline]
  #[must_use]
  pub fn checksum(&self) -> C::Output {
    self.hasher.finalize()
  }

  /// Get a mutable reference to the underlying hasher.
  #[inline]
  pub fn hasher_mut(&mut self) -> &mut C {
    &mut self.hasher
  }

  /// Unwrap this `ChecksumWriter`, returning the inner writer and the final checksum.
  #[inline]
  pub fn into_parts(self) -> (W, C::Output) {
    (self.inner, self.hasher.finalize())
  }

  /// Unwrap this `ChecksumWriter`, returning the inner writer and discarding the checksum.
  #[inline]
  pub fn into_inner(self) -> W {
    self.inner
  }

  /// Get a reference to the inner writer.
  #[inline]
  pub fn inner(&self) -> &W {
    &self.inner
  }

  /// Get a mutable reference to the inner writer.
  #[inline]
  pub fn inner_mut(&mut self) -> &mut W {
    &mut self.inner
  }
}

#[cfg(feature = "std")]
impl<W: std::io::Write, C: crate::Checksum> std::io::Write for ChecksumWriter<W, C> {
  #[inline]
  fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
    write_and_update(&mut self.inner, buf, |data| self.hasher.update(data))
  }

  #[inline]
  fn flush(&mut self) -> std::io::Result<()> {
    self.inner.flush()
  }

  #[inline]
  fn write_vectored(&mut self, bufs: &[std::io::IoSlice<'_>]) -> std::io::Result<usize> {
    write_vectored_and_update(&mut self.inner, bufs, |data| self.hasher.update(data))
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Digest I/O Adapters
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps a [`Read`](std::io::Read) and computes a digest transparently.
///
/// All reads from this type pass through to the inner reader while
/// updating the digest with the actual bytes read (handling short reads).
///
/// # Type Parameters
///
/// - `R`: The inner reader type
/// - `D`: The digest algorithm type (e.g., `Blake3`)
///
/// # Example
///
/// ```rust
/// # use rscrypto::traits::Digest;
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
/// let mut reader = SumDigest::reader(Cursor::new(b"abc".to_vec()));
/// std::io::copy(&mut reader, &mut std::io::sink())?;
/// assert_eq!(
///   reader.digest(),
///   [b'a'.wrapping_add(b'b').wrapping_add(b'c'); 4]
/// );
/// # Ok::<(), std::io::Error>(())
/// ```
#[cfg(feature = "std")]
#[derive(Clone)]
pub struct DigestReader<R, D: crate::Digest> {
  inner: R,
  hasher: D,
}

#[cfg(feature = "std")]
impl<R: core::fmt::Debug, D: crate::Digest + core::fmt::Debug> core::fmt::Debug for DigestReader<R, D> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("DigestReader")
      .field("inner", &self.inner)
      .field("hasher", &self.hasher)
      .finish()
  }
}

#[cfg(feature = "std")]
impl<R, D: crate::Digest> DigestReader<R, D> {
  /// Create a new reader wrapper with the default initial state.
  #[inline]
  #[must_use]
  pub fn new(inner: R) -> Self {
    Self {
      inner,
      hasher: D::new(),
    }
  }

  /// Get the current digest value.
  ///
  /// This does not consume the reader or finalize the hasher -
  /// further reads will continue updating the digest.
  #[inline]
  #[must_use]
  pub fn digest(&self) -> D::Output {
    self.hasher.finalize()
  }

  /// Get a mutable reference to the underlying hasher.
  ///
  /// This allows advanced use cases like manual state manipulation.
  #[inline]
  pub fn hasher_mut(&mut self) -> &mut D {
    &mut self.hasher
  }

  /// Unwrap this `DigestReader`, returning the inner reader and the final digest.
  #[inline]
  pub fn into_parts(self) -> (R, D::Output) {
    (self.inner, self.hasher.finalize())
  }

  /// Unwrap this `DigestReader`, returning the inner reader and discarding the digest.
  #[inline]
  pub fn into_inner(self) -> R {
    self.inner
  }

  /// Get a reference to the inner reader.
  #[inline]
  pub fn inner(&self) -> &R {
    &self.inner
  }

  /// Get a mutable reference to the inner reader.
  #[inline]
  pub fn inner_mut(&mut self) -> &mut R {
    &mut self.inner
  }
}

#[cfg(feature = "std")]
impl<R: std::io::Read, D: crate::Digest> std::io::Read for DigestReader<R, D> {
  #[inline]
  fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
    read_and_update(&mut self.inner, buf, |data| self.hasher.update(data))
  }

  #[inline]
  fn read_vectored(&mut self, bufs: &mut [std::io::IoSliceMut<'_>]) -> std::io::Result<usize> {
    read_vectored_and_update(&mut self.inner, bufs, |data| self.hasher.update(data))
  }
}

/// Wraps a [`Write`](std::io::Write) and computes a digest transparently.
///
/// All writes to this type pass through to the inner writer while
/// updating the digest with the bytes being written.
///
/// # Partial Write Semantics
///
/// The digest is updated with only the bytes the inner writer reports as
/// successfully written. Short writes do not hash unwritten bytes.
///
/// # Type Parameters
///
/// - `W`: The inner writer type
/// - `D`: The digest algorithm type (e.g., `Blake3`)
///
/// # Example
///
/// ```rust
/// # use rscrypto::traits::Digest;
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
#[derive(Clone)]
pub struct DigestWriter<W, D: crate::Digest> {
  inner: W,
  hasher: D,
}

#[cfg(feature = "std")]
impl<W: core::fmt::Debug, D: crate::Digest + core::fmt::Debug> core::fmt::Debug for DigestWriter<W, D> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("DigestWriter")
      .field("inner", &self.inner)
      .field("hasher", &self.hasher)
      .finish()
  }
}

#[cfg(feature = "std")]
impl<W, D: crate::Digest> DigestWriter<W, D> {
  /// Create a new writer wrapper with the default initial state.
  #[inline]
  #[must_use]
  pub fn new(inner: W) -> Self {
    Self {
      inner,
      hasher: D::new(),
    }
  }

  /// Get the current digest value.
  #[inline]
  #[must_use]
  pub fn digest(&self) -> D::Output {
    self.hasher.finalize()
  }

  /// Get a mutable reference to the underlying hasher.
  #[inline]
  pub fn hasher_mut(&mut self) -> &mut D {
    &mut self.hasher
  }

  /// Unwrap this `DigestWriter`, returning the inner writer and the final digest.
  #[inline]
  pub fn into_parts(self) -> (W, D::Output) {
    (self.inner, self.hasher.finalize())
  }

  /// Unwrap this `DigestWriter`, returning the inner writer and discarding the digest.
  #[inline]
  pub fn into_inner(self) -> W {
    self.inner
  }

  /// Get a reference to the inner writer.
  #[inline]
  pub fn inner(&self) -> &W {
    &self.inner
  }

  /// Get a mutable reference to the inner writer.
  #[inline]
  pub fn inner_mut(&mut self) -> &mut W {
    &mut self.inner
  }
}

#[cfg(feature = "std")]
impl<W: std::io::Write, D: crate::Digest> std::io::Write for DigestWriter<W, D> {
  #[inline]
  fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
    write_and_update(&mut self.inner, buf, |data| self.hasher.update(data))
  }

  #[inline]
  fn flush(&mut self) -> std::io::Result<()> {
    self.inner.flush()
  }

  #[inline]
  fn write_vectored(&mut self, bufs: &[std::io::IoSlice<'_>]) -> std::io::Result<usize> {
    write_vectored_and_update(&mut self.inner, bufs, |data| self.hasher.update(data))
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use std::io::{self, IoSlice, IoSliceMut, Read, Write};

  use super::{ChecksumWriter, DigestWriter};
  use crate::{Checksum, Digest};

  #[derive(Clone, Copy, Debug, Default)]
  struct Sum(u32);

  impl Checksum for Sum {
    const OUTPUT_SIZE: usize = 4;
    type Output = u32;

    fn new() -> Self {
      Self(0)
    }

    fn with_initial(initial: Self::Output) -> Self {
      Self(initial)
    }

    fn update(&mut self, data: &[u8]) {
      self.0 = data.iter().fold(self.0, |acc, &b| acc.wrapping_add(u32::from(b)));
    }

    fn finalize(&self) -> Self::Output {
      self.0
    }

    fn reset(&mut self) {
      self.0 = 0;
    }
  }

  #[derive(Clone, Copy, Debug, Default)]
  struct SumDigest(u8);

  impl Digest for SumDigest {
    const OUTPUT_SIZE: usize = 4;
    type Output = [u8; 4];

    fn new() -> Self {
      Self(0)
    }

    fn update(&mut self, data: &[u8]) {
      self.0 = data.iter().fold(self.0, |acc, &b| acc.wrapping_add(b));
    }

    fn finalize(&self) -> Self::Output {
      [self.0; 4]
    }

    fn reset(&mut self) {
      self.0 = 0;
    }
  }

  #[derive(Debug, Default)]
  struct PartialWriter {
    inner: Vec<u8>,
    max_per_call: usize,
  }

  impl PartialWriter {
    fn new(max_per_call: usize) -> Self {
      Self {
        inner: Vec::new(),
        max_per_call,
      }
    }
  }

  impl Write for PartialWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
      let written = self.max_per_call.min(buf.len());
      self.inner.extend_from_slice(&buf[..written]);
      Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
      Ok(())
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
      let mut remaining = self.max_per_call;
      let mut written = 0usize;

      for buf in bufs {
        let take = remaining.min(buf.len());
        if remaining == 0 {
          break;
        }
        if take == 0 {
          continue;
        }
        self.inner.extend_from_slice(&buf[..take]);
        written = written.strict_add(take);
        remaining = remaining.strict_sub(take);
      }

      Ok(written)
    }
  }

  #[inline]
  fn checksum_sum(data: &[u8]) -> u32 {
    data.iter().fold(0u32, |acc, &b| acc.wrapping_add(u32::from(b)))
  }

  #[inline]
  fn digest_sum(data: &[u8]) -> [u8; 4] {
    let sum = data.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    [sum; 4]
  }

  #[test]
  fn checksum_writer_hashes_only_written_prefix() {
    let mut writer = ChecksumWriter::<_, Sum>::new(PartialWriter::new(4));
    let written = writer.write(b"abcdef").unwrap();
    assert_eq!(written, 4);
    assert_eq!(writer.checksum(), checksum_sum(b"abcd"));

    let (inner, checksum) = writer.into_parts();
    assert_eq!(inner.inner, b"abcd");
    assert_eq!(checksum, checksum_sum(b"abcd"));
  }

  #[test]
  fn checksum_writer_vectored_hashes_only_written_prefix() {
    let mut writer = ChecksumWriter::<_, Sum>::new(PartialWriter::new(5));
    let bufs = [IoSlice::new(b"ab"), IoSlice::new(b"cdef"), IoSlice::new(b"gh")];
    let written = writer.write_vectored(&bufs).unwrap();
    assert_eq!(written, 5);
    assert_eq!(writer.checksum(), checksum_sum(b"abcde"));

    let (inner, checksum) = writer.into_parts();
    assert_eq!(inner.inner, b"abcde");
    assert_eq!(checksum, checksum_sum(b"abcde"));
  }

  #[test]
  fn digest_writer_hashes_only_written_prefix() {
    let mut writer = DigestWriter::<_, SumDigest>::new(PartialWriter::new(3));
    let written = writer.write(b"abcdef").unwrap();
    assert_eq!(written, 3);
    assert_eq!(writer.digest(), digest_sum(b"abc"));

    let (inner, digest) = writer.into_parts();
    assert_eq!(inner.inner, b"abc");
    assert_eq!(digest, digest_sum(b"abc"));
  }

  #[test]
  fn digest_writer_vectored_hashes_only_written_prefix() {
    let mut writer = DigestWriter::<_, SumDigest>::new(PartialWriter::new(6));
    let bufs = [IoSlice::new(b"ab"), IoSlice::new(b"cdef"), IoSlice::new(b"ghij")];
    let written = writer.write_vectored(&bufs).unwrap();
    assert_eq!(written, 6);
    assert_eq!(writer.digest(), digest_sum(b"abcdef"));

    let (inner, digest) = writer.into_parts();
    assert_eq!(inner.inner, b"abcdef");
    assert_eq!(digest, digest_sum(b"abcdef"));
  }

  #[test]
  fn checksum_reader_vectored_skips_empty_buffers() {
    let mut reader = Sum::reader(io::Cursor::new(b"abcde".to_vec()));
    let mut first = [];
    let mut second = [0u8; 2];
    let mut third = [0u8; 3];
    let mut bufs = [
      IoSliceMut::new(&mut first),
      IoSliceMut::new(&mut second),
      IoSliceMut::new(&mut third),
    ];

    let read = reader.read_vectored(&mut bufs).unwrap();
    assert_eq!(read, 5);
    assert_eq!(second, *b"ab");
    assert_eq!(third, *b"cde");
    assert_eq!(reader.checksum(), checksum_sum(b"abcde"));
  }

  #[test]
  fn digest_reader_vectored_skips_empty_buffers() {
    let mut reader = SumDigest::reader(io::Cursor::new(b"abcde".to_vec()));
    let mut first = [];
    let mut second = [0u8; 2];
    let mut third = [0u8; 3];
    let mut bufs = [
      IoSliceMut::new(&mut first),
      IoSliceMut::new(&mut second),
      IoSliceMut::new(&mut third),
    ];

    let read = reader.read_vectored(&mut bufs).unwrap();
    assert_eq!(read, 5);
    assert_eq!(second, *b"ab");
    assert_eq!(third, *b"cde");
    assert_eq!(reader.digest(), digest_sum(b"abcde"));
  }

  #[test]
  fn checksum_writer_vectored_skips_empty_buffers() {
    let mut writer = ChecksumWriter::<_, Sum>::new(PartialWriter::new(5));
    let bufs = [IoSlice::new(b""), IoSlice::new(b"ab"), IoSlice::new(b"cde")];
    let written = writer.write_vectored(&bufs).unwrap();

    assert_eq!(written, 5);
    assert_eq!(writer.checksum(), checksum_sum(b"abcde"));

    let (inner, checksum) = writer.into_parts();
    assert_eq!(inner.inner, b"abcde");
    assert_eq!(checksum, checksum_sum(b"abcde"));
  }

  #[test]
  fn digest_writer_vectored_skips_empty_buffers() {
    let mut writer = DigestWriter::<_, SumDigest>::new(PartialWriter::new(5));
    let bufs = [IoSlice::new(b""), IoSlice::new(b"ab"), IoSlice::new(b"cde")];
    let written = writer.write_vectored(&bufs).unwrap();

    assert_eq!(written, 5);
    assert_eq!(writer.digest(), digest_sum(b"abcde"));

    let (inner, digest) = writer.into_parts();
    assert_eq!(inner.inner, b"abcde");
    assert_eq!(digest, digest_sum(b"abcde"));
  }
}
