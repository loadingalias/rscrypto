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
//! ```rust,ignore
//! use checksum::{Crc32C, ChecksumReader};
//! use std::fs::File;
//!
//! let file = File::open("data.bin")?;
//! let mut reader = Crc32C::reader(file);
//! std::io::copy(&mut reader, &mut std::io::sink())?;
//! let crc = reader.crc();
//! ```

/// Sealed trait marker - not implementable outside rscrypto.
///
/// This module prevents external implementations of [`Hashable`](crate::io::Hashable),
/// allowing us to extend the trait with new methods without breaking changes.
mod private {
  /// Sealed trait marker.
  pub trait Sealed {}
}

// Internal re-export for use by sibling modules (checksum, digest).
// This is #[doc(hidden)] because external users should not implement this trait.
#[doc(hidden)]
pub use private::Sealed as SealedMarker;

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
/// ```rust,ignore
/// use checksum::Crc32C;
/// use std::fs::File;
///
/// let file = File::open("large.iso")?;
/// let mut reader = Crc32C::reader(file);
/// std::io::copy(&mut reader, &mut std::io::sink())?;
/// assert_eq!(reader.crc(), expected_crc);
/// ```
#[cfg(feature = "std")]
#[derive(Clone)]
pub struct ChecksumReader<R, C: crate::Checksum> {
  inner: R,
  hasher: C,
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
  pub fn crc(&self) -> C::Output {
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
    let n = self.inner.read(buf)?;
    if let Some(data) = buf.get(..n) {
      self.hasher.update(data);
    }
    Ok(n)
  }

  #[inline]
  fn read_vectored(&mut self, bufs: &mut [std::io::IoSliceMut<'_>]) -> std::io::Result<usize> {
    let n = self.inner.read_vectored(bufs)?;
    let mut remaining = n;
    for buf in bufs {
      let to_hash = remaining.min(buf.len());
      if to_hash > 0 {
        if let Some(data) = buf.get(..to_hash) {
          self.hasher.update(data);
        }
        remaining -= to_hash;
      } else {
        break;
      }
    }
    Ok(n)
  }
}

/// Wraps a [`Write`](std::io::Write) and computes a checksum transparently.
///
/// All writes to this type pass through to the inner writer while
/// updating the checksum with the bytes being written.
///
/// # Important: Hash-Then-Write Order
///
/// The checksum is updated **before** writing to the inner writer.
/// This ensures that if the write fails, the caller knows exactly
/// what data was hashed vs what was successfully written.
///
/// # Type Parameters
///
/// - `W`: The inner writer type
/// - `C`: The checksum algorithm type (e.g., `Crc32C`)
///
/// # Example
///
/// ```rust,ignore
/// use checksum::Crc32C;
/// use std::fs::File;
///
/// let file = File::create("output.bin")?;
/// let mut writer = Crc32C::writer(file);
/// writer.write_all(b"hello world")?;
/// let (file, crc) = writer.into_parts();
/// println!("Wrote file with CRC: {:08x}", crc);
/// ```
#[cfg(feature = "std")]
#[derive(Clone)]
pub struct ChecksumWriter<W, C: crate::Checksum> {
  inner: W,
  hasher: C,
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
  pub fn crc(&self) -> C::Output {
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
    self.hasher.update(buf);
    self.inner.write(buf)
  }

  #[inline]
  fn flush(&mut self) -> std::io::Result<()> {
    self.inner.flush()
  }

  #[inline]
  fn write_vectored(&mut self, bufs: &[std::io::IoSlice<'_>]) -> std::io::Result<usize> {
    for buf in bufs {
      self.hasher.update(buf);
    }
    self.inner.write_vectored(bufs)
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
/// ```rust,ignore
/// use hashes::crypto::blake3::Blake3;
/// use std::fs::File;
///
/// let file = File::open("large.iso")?;
/// let mut reader = Blake3::reader(file);
/// std::io::copy(&mut reader, &mut std::io::sink())?;
/// assert_eq!(reader.digest(), expected_hash);
/// ```
#[cfg(feature = "std")]
#[derive(Clone)]
pub struct DigestReader<R, D: crate::Digest> {
  inner: R,
  hasher: D,
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
    let n = self.inner.read(buf)?;
    if let Some(data) = buf.get(..n) {
      self.hasher.update(data);
    }
    Ok(n)
  }

  #[inline]
  fn read_vectored(&mut self, bufs: &mut [std::io::IoSliceMut<'_>]) -> std::io::Result<usize> {
    let n = self.inner.read_vectored(bufs)?;
    let mut remaining = n;
    for buf in bufs {
      let to_hash = remaining.min(buf.len());
      if to_hash > 0 {
        if let Some(data) = buf.get(..to_hash) {
          self.hasher.update(data);
        }
        remaining -= to_hash;
      } else {
        break;
      }
    }
    Ok(n)
  }
}

/// Wraps a [`Write`](std::io::Write) and computes a digest transparently.
///
/// All writes to this type pass through to the inner writer while
/// updating the digest with the bytes being written.
///
/// # Important: Hash-Then-Write Order
///
/// The digest is updated **before** writing to the inner writer.
/// This ensures that if the write fails, the caller knows exactly
/// what data was hashed vs what was successfully written.
///
/// # Type Parameters
///
/// - `W`: The inner writer type
/// - `D`: The digest algorithm type (e.g., `Blake3`)
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
/// println!("Blake3: {:?}", digest);
/// ```
#[cfg(feature = "std")]
#[derive(Clone)]
pub struct DigestWriter<W, D: crate::Digest> {
  inner: W,
  hasher: D,
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
    self.hasher.update(buf);
    self.inner.write(buf)
  }

  #[inline]
  fn flush(&mut self) -> std::io::Result<()> {
    self.inner.flush()
  }

  #[inline]
  fn write_vectored(&mut self, bufs: &[std::io::IoSlice<'_>]) -> std::io::Result<usize> {
    for buf in bufs {
      self.hasher.update(buf);
    }
    self.inner.write_vectored(bufs)
  }
}
