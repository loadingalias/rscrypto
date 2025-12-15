//! Non-cryptographic checksum traits.
//!
//! Traits for checksum algorithms like CRC32, CRC64, and non-cryptographic hashes.
//!
//! - **Performance**: Zero-cost abstractions, inline-friendly
//! - **Streaming**: Incremental updates for large data
//! - **Parallelism**: Combine operation for parallel chunk processing
//!
//! See [`checksum`](https://docs.rs/checksum) crate for implementations.

use core::fmt::Debug;

/// Non-cryptographic checksum algorithm.
///
/// Provides the core interface for checksum computation with support for
/// incremental updates and streaming data.
///
/// # Implementors
///
/// - [`checksum::Crc32c`] - CRC32-C (Castagnoli)
/// - [`checksum::Crc32`] - CRC32 (ISO-HDLC)
/// - [`checksum::Crc64`] - CRC64 (XZ/ECMA)
///
/// # Usage Pattern
///
/// ```text
/// // One-shot (fastest)
/// let crc = Crc32c::checksum(b"hello world");
///
/// // Streaming
/// let mut hasher = Crc32c::new();
/// hasher.update(b"hello ");
/// hasher.update(b"world");
/// let crc = hasher.finalize();
/// ```
///
/// [`checksum::Crc32c`]: https://docs.rs/checksum/latest/checksum/struct.Crc32c.html
/// [`checksum::Crc32`]: https://docs.rs/checksum/latest/checksum/struct.Crc32.html
/// [`checksum::Crc64`]: https://docs.rs/checksum/latest/checksum/struct.Crc64.html
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
  fn new() -> Self;

  /// Create a new hasher with a custom initial value.
  ///
  /// This is useful for resuming a checksum computation or for
  /// non-standard initial values.
  fn with_initial(initial: Self::Output) -> Self;

  /// Update the hasher with additional data.
  ///
  /// This method can be called multiple times to process data incrementally.
  fn update(&mut self, data: &[u8]);

  /// Finalize and return the checksum.
  ///
  /// This method does not consume the hasher, allowing further updates
  /// if needed (though the result would include all data processed so far).
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
  fn checksum(data: &[u8]) -> Self::Output {
    let mut h = Self::new();
    h.update(data);
    h.finalize()
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
/// # Usage Pattern
///
/// ```text
/// let data = b"hello world";
/// let (a, b) = data.split_at(6);
///
/// let crc_a = Crc32c::checksum(a);
/// let crc_b = Crc32c::checksum(b);
///
/// // Combine produces crc(a || b)
/// let combined = Crc32c::combine(crc_a, crc_b, b.len());
/// assert_eq!(combined, Crc32c::checksum(data));
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
  ///
  /// # Returns
  ///
  /// The checksum of the concatenation `A || B`.
  fn combine(crc_a: Self::Output, crc_b: Self::Output, len_b: usize) -> Self::Output;
}
