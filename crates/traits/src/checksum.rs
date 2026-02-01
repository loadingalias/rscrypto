//! Non-cryptographic checksum traits.
//!
//! Traits for checksum algorithms like CRC32, CRC64, and non-cryptographic hashes.
//!
//! - **Performance**: Zero-cost abstractions, inline-friendly
//! - **Streaming**: Incremental updates for large data
//! - **Parallelism**: Combine operation for parallel chunk processing

use core::fmt::Debug;

/// Non-cryptographic checksum algorithm.
///
/// Provides the core interface for checksum computation with support for
/// incremental updates and streaming data.
///
/// # Usage
///
/// ```rust,ignore
/// use checksum::{Checksum, Crc32C};
///
/// // One-shot (fastest for data already in memory)
/// let crc = Crc32C::checksum(b"hello world");
///
/// // Streaming (for incremental or large data)
/// let mut hasher = Crc32C::new();
/// hasher.update(b"hello ");
/// hasher.update(b"world");
/// let crc = hasher.finalize();
/// ```
///
/// # Implementor Requirements
///
/// - `new()` must return the same state as `Default::default()`
/// - `finalize()` must be idempotent (calling multiple times returns same value)
/// - `reset()` must restore the hasher to its initial state
pub trait Checksum: Clone + Default {
  /// Output size in bytes.
  ///
  /// - CRC32: 4
  /// - CRC64: 8
  /// - CRC16: 2
  const OUTPUT_SIZE: usize;

  /// The checksum output type.
  ///
  /// Typically `u32` for CRC32, `u64` for CRC64, etc.
  type Output: Copy + Eq + Debug + Default;

  /// Create a new hasher with the default initial value.
  #[must_use]
  fn new() -> Self;

  /// Create a new hasher with a custom initial value.
  ///
  /// Useful for resuming a checksum computation or for non-standard initial values.
  #[must_use]
  fn with_initial(initial: Self::Output) -> Self;

  /// Update the hasher with additional data.
  ///
  /// This method can be called multiple times to process data incrementally.
  fn update(&mut self, data: &[u8]);

  /// Update the hasher with multiple non-contiguous buffers.
  ///
  /// Semantics are identical to calling [`update`](Self::update) on each buffer
  /// in order, but implementations may fuse dispatch and reduce per-buffer
  /// overhead.
  #[inline]
  fn update_vectored(&mut self, bufs: &[&[u8]]) {
    for buf in bufs {
      self.update(buf);
    }
  }

  /// Update the hasher with `std::io::IoSlice` buffers.
  ///
  /// This is a convenience for integrating with vectored I/O APIs.
  #[cfg(feature = "std")]
  #[inline]
  fn update_io_slices(&mut self, bufs: &[std::io::IoSlice<'_>]) {
    for buf in bufs {
      self.update(buf);
    }
  }

  /// Finalize and return the checksum.
  ///
  /// This method does not consume the hasher, allowing further updates
  /// if needed (though the result would include all data processed so far).
  #[must_use]
  fn finalize(&self) -> Self::Output;

  /// Reset the hasher to its initial state.
  ///
  /// After calling this, the hasher behaves as if newly constructed.
  fn reset(&mut self);

  /// Compute the checksum of data in one shot.
  ///
  /// This is the fastest path for small to medium data that fits in memory.
  /// For large data or streaming, use [`new`](Self::new) + [`update`](Self::update).
  #[inline]
  #[must_use]
  fn checksum(data: &[u8]) -> Self::Output {
    let mut h = Self::new();
    h.update(data);
    h.finalize()
  }

  /// Compute the checksum of multiple buffers in one shot.
  #[inline]
  #[must_use]
  fn checksum_vectored(bufs: &[&[u8]]) -> Self::Output {
    let mut h = Self::new();
    h.update_vectored(bufs);
    h.finalize()
  }

  /// Compute the checksum of `std::io::IoSlice` buffers in one shot.
  #[cfg(feature = "std")]
  #[inline]
  #[must_use]
  fn checksum_io_slices(bufs: &[std::io::IoSlice<'_>]) -> Self::Output {
    let mut h = Self::new();
    h.update_io_slices(bufs);
    h.finalize()
  }

  /// Wrap a reader to compute checksum transparently during I/O.
  ///
  /// # Example
  ///
  /// ```rust,ignore
  /// use checksum::Crc32C;
  /// use std::fs::File;
  ///
  /// let file = File::open("data.bin")?;
  /// let mut reader = Crc32C::reader(file);
  /// std::io::copy(&mut reader, &mut std::io::sink())?;
  /// println!("CRC: {:08x}", reader.crc());
  /// ```
  #[cfg(feature = "std")]
  #[inline]
  #[must_use]
  fn reader<R>(inner: R) -> crate::io::ChecksumReader<R, Self>
  where
    Self: Sized,
  {
    crate::io::ChecksumReader::new(inner)
  }

  /// Wrap a writer to compute checksum transparently during I/O.
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
  /// println!("CRC: {:08x}", crc);
  /// ```
  #[cfg(feature = "std")]
  #[inline]
  #[must_use]
  fn writer<W>(inner: W) -> crate::io::ChecksumWriter<W, Self>
  where
    Self: Sized,
  {
    crate::io::ChecksumWriter::new(inner)
  }
}

/// Checksums that support parallel computation via combination.
///
/// The combine operation computes `crc(A || B)` from `crc(A)`, `crc(B)`, and `len(B)`
/// in O(log n) time. This enables parallel checksum computation:
///
/// 1. Split data into chunks
/// 2. Compute checksums in parallel
/// 3. Combine results
///
/// # Mathematical Background
///
/// For CRC, this works because:
///
/// ```text
/// crc(A || B) = crc(A) * x^(8*len(B)) mod G(x) XOR crc(B)
/// ```
///
/// The exponentiation uses square-and-multiply for O(log n) complexity.
///
/// # Usage
///
/// ```rust,ignore
/// use checksum::{Checksum, ChecksumCombine, Crc32C};
///
/// let data = b"hello world";
/// let (a, b) = data.split_at(6);
///
/// let crc_a = Crc32C::checksum(a);
/// let crc_b = Crc32C::checksum(b);
///
/// // Combine produces crc(a || b)
/// let combined = Crc32C::combine(crc_a, crc_b, b.len());
/// assert_eq!(combined, Crc32C::checksum(data));
/// ```
pub trait ChecksumCombine: Checksum {
  /// Combine two checksums.
  ///
  /// Given `crc_a = crc(A)` and `crc_b = crc(B)`, computes `crc(A || B)`.
  ///
  /// # Arguments
  ///
  /// * `crc_a` - Checksum of the first part (A)
  /// * `crc_b` - Checksum of the second part (B)
  /// * `len_b` - Length of the second part in bytes
  #[must_use]
  fn combine(crc_a: Self::Output, crc_b: Self::Output, len_b: usize) -> Self::Output;
}
