//! Parallel checksum computation utilities.
//!
//! This module provides helpers for computing checksums over chunked data,
//! enabling parallel processing with user-provided parallelism (rayon, threads, etc.).
//!
//! # Design
//!
//! The CRC combine operation computes `crc(A || B)` from `crc(A)`, `crc(B)`, and `len(B)`
//! in O(log n) time. This module provides utilities that leverage this property.
//!
//! **Key insight**: This module does NOT add any dependencies. Users bring their own
//! parallelism (rayon, std::thread, tokio, etc.) and use these helpers to combine results.
//!
//! # Example: Manual Parallelism
//!
//! ```
//! use checksum::{Crc32c, parallel::checksum_chunks};
//!
//! let data = b"The quick brown fox jumps over the lazy dog";
//! let chunks: Vec<&[u8]> = data.chunks(16).collect();
//!
//! // Sequential combination (works everywhere including no_std)
//! let crc = checksum_chunks::<Crc32c>(&chunks);
//! assert_eq!(crc, Crc32c::checksum(data));
//! ```
//!
//! # Example: With Rayon (user brings dependency)
//!
//! ```ignore
//! use rayon::prelude::*;
//! use checksum::{Crc32c, parallel::combine_checksums};
//!
//! let chunks: Vec<&[u8]> = large_data.chunks(1024 * 1024).collect();
//!
//! // Compute checksums in parallel
//! let checksums: Vec<(u32, usize)> = chunks
//!     .par_iter()
//!     .map(|chunk| (Crc32c::checksum(chunk), chunk.len()))
//!     .collect();
//!
//! // Combine results
//! let final_crc = combine_checksums::<Crc32c>(&checksums);
//! ```
//!
//! # no_std Support
//!
//! All functions in this module work in `no_std` environments. The only requirement
//! is that the checksum type implements [`ChecksumCombine`].

use traits::ChecksumCombine;

/// Compute checksum over multiple chunks by combining their individual checksums.
///
/// This function computes the checksum of each chunk and combines them using
/// the CRC combine operation. The result is equivalent to computing the checksum
/// of the concatenated data.
///
/// # Complexity
///
/// - Checksum computation: O(total_bytes)
/// - Combine operations: O(n × log(max_chunk_len)) where n = number of chunks
///
/// # Example
///
/// ```
/// use checksum::{Crc32c, parallel::checksum_chunks};
///
/// let data = b"hello world";
/// let chunks: Vec<&[u8]> = data.chunks(4).collect();
///
/// assert_eq!(checksum_chunks::<Crc32c>(&chunks), Crc32c::checksum(data));
/// ```
///
/// # Empty Input
///
/// Returns the checksum of empty data (typically 0 for most CRC variants).
///
/// ```
/// use checksum::{Crc32c, parallel::checksum_chunks};
///
/// let empty: &[&[u8]] = &[];
/// assert_eq!(checksum_chunks::<Crc32c>(empty), Crc32c::checksum(b""));
/// ```
#[inline]
pub fn checksum_chunks<C: ChecksumCombine>(chunks: &[&[u8]]) -> C::Output {
  let Some((first, rest)) = chunks.split_first() else {
    return C::checksum(&[]);
  };

  let mut result = C::checksum(first);
  for chunk in rest {
    let chunk_crc = C::checksum(chunk);
    result = C::combine(result, chunk_crc, chunk.len());
  }

  result
}

/// Combine pre-computed checksums into a single checksum.
///
/// This function takes checksums that were computed separately (potentially in parallel)
/// and combines them into the checksum of the concatenated data.
///
/// # Arguments
///
/// * `checksums` - Slice of (checksum, chunk_length) pairs in order
///
/// # Example
///
/// ```
/// use checksum::{Crc32c, parallel::combine_checksums};
///
/// let data = b"hello world";
/// let (a, b) = data.split_at(6);
///
/// // Compute separately (could be in parallel)
/// let checksums = [
///   (Crc32c::checksum(a), a.len()),
///   (Crc32c::checksum(b), b.len()),
/// ];
///
/// assert_eq!(
///   combine_checksums::<Crc32c>(&checksums),
///   Crc32c::checksum(data)
/// );
/// ```
///
/// # Panics
///
/// Panics if `checksums` is empty. Use [`combine_checksums_or`] for fallible version.
#[inline]
#[allow(clippy::expect_used)] // Intentional panic documented above
pub fn combine_checksums<C: ChecksumCombine>(checksums: &[(C::Output, usize)]) -> C::Output {
  combine_checksums_or::<C>(checksums).expect("checksums slice must not be empty")
}

/// Combine pre-computed checksums, returning `None` if empty.
///
/// This is the fallible version of [`combine_checksums`].
///
/// # Example
///
/// ```
/// use checksum::{Crc32c, parallel::combine_checksums_or};
///
/// let empty: &[(u32, usize)] = &[];
/// assert_eq!(combine_checksums_or::<Crc32c>(empty), None);
///
/// let single = [(0x12345678u32, 100)];
/// assert_eq!(combine_checksums_or::<Crc32c>(&single), Some(0x12345678));
/// ```
#[inline]
pub fn combine_checksums_or<C: ChecksumCombine>(checksums: &[(C::Output, usize)]) -> Option<C::Output> {
  let mut iter = checksums.iter();
  let (first_crc, _) = iter.next()?;
  let mut result = *first_crc;

  for &(crc, len) in iter {
    result = C::combine(result, crc, len);
  }

  Some(result)
}

/// Iterator adapter for computing checksum over chunked data.
///
/// This struct is created by [`checksum_iter`]. See its documentation for more.
pub struct ChecksumIter<I, C> {
  inner: I,
  _marker: core::marker::PhantomData<C>,
}

impl<I, C> ChecksumIter<I, C>
where
  I: Iterator,
  I::Item: AsRef<[u8]>,
  C: ChecksumCombine,
{
  /// Consume the iterator and compute the combined checksum.
  ///
  /// # Example
  ///
  /// ```
  /// use checksum::{Crc32c, parallel::checksum_iter};
  ///
  /// let chunks = vec![b"hello ".to_vec(), b"world".to_vec()];
  /// let crc = checksum_iter::<_, Crc32c>(chunks.iter()).finalize();
  ///
  /// assert_eq!(crc, Crc32c::checksum(b"hello world"));
  /// ```
  #[inline]
  pub fn finalize(self) -> C::Output {
    let mut iter = self.inner;

    let Some(first) = iter.next() else {
      return C::checksum(&[]);
    };

    let first_data = first.as_ref();
    let mut result = C::checksum(first_data);

    for chunk in iter {
      let chunk_data = chunk.as_ref();
      let chunk_crc = C::checksum(chunk_data);
      result = C::combine(result, chunk_crc, chunk_data.len());
    }

    result
  }
}

/// Create a checksum iterator adapter.
///
/// This allows computing checksums over any iterator of byte slices.
///
/// # Example
///
/// ```
/// use checksum::{Crc64, parallel::checksum_iter};
///
/// let data = b"The quick brown fox";
///
/// // From chunks iterator
/// let crc = checksum_iter::<_, Crc64>(data.chunks(8)).finalize();
/// assert_eq!(crc, Crc64::checksum(data));
///
/// // From vec of vecs
/// let chunks: Vec<Vec<u8>> = vec![b"The quick ".to_vec(), b"brown fox".to_vec()];
/// let crc = checksum_iter::<_, Crc64>(chunks.iter()).finalize();
/// assert_eq!(crc, Crc64::checksum(data));
/// ```
#[inline]
pub fn checksum_iter<I, C>(iter: I) -> ChecksumIter<I, C>
where
  I: Iterator,
  I::Item: AsRef<[u8]>,
  C: ChecksumCombine,
{
  ChecksumIter {
    inner: iter,
    _marker: core::marker::PhantomData,
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{Crc16Ibm, Crc24, Crc32, Crc32c, Crc64, Crc64Nvme};

  #[test]
  fn test_checksum_chunks_crc32c() {
    let data = b"The quick brown fox jumps over the lazy dog";

    // Various chunk sizes
    for chunk_size in [1, 2, 3, 4, 7, 8, 16, 32, 64] {
      let chunks: alloc::vec::Vec<&[u8]> = data.chunks(chunk_size).collect();
      assert_eq!(
        checksum_chunks::<Crc32c>(&chunks),
        Crc32c::checksum(data),
        "mismatch at chunk_size={}",
        chunk_size
      );
    }
  }

  #[test]
  fn test_checksum_chunks_crc32() {
    let data = b"hello world";
    let chunks: alloc::vec::Vec<&[u8]> = data.chunks(4).collect();
    assert_eq!(checksum_chunks::<Crc32>(&chunks), Crc32::checksum(data));
  }

  #[test]
  fn test_checksum_chunks_crc64() {
    let data = b"The quick brown fox";
    let chunks: alloc::vec::Vec<&[u8]> = data.chunks(8).collect();
    assert_eq!(checksum_chunks::<Crc64>(&chunks), Crc64::checksum(data));
  }

  #[test]
  fn test_checksum_chunks_crc64_nvme() {
    let data = b"NVMe storage test data";
    let chunks: alloc::vec::Vec<&[u8]> = data.chunks(7).collect();
    assert_eq!(checksum_chunks::<Crc64Nvme>(&chunks), Crc64Nvme::checksum(data));
  }

  #[test]
  fn test_checksum_chunks_crc16() {
    let data = b"Modbus test";
    let chunks: alloc::vec::Vec<&[u8]> = data.chunks(3).collect();
    assert_eq!(checksum_chunks::<Crc16Ibm>(&chunks), Crc16Ibm::checksum(data));
  }

  #[test]
  fn test_checksum_chunks_crc24() {
    let data = b"OpenPGP armor";
    let chunks: alloc::vec::Vec<&[u8]> = data.chunks(5).collect();
    assert_eq!(checksum_chunks::<Crc24>(&chunks), Crc24::checksum(data));
  }

  #[test]
  fn test_checksum_chunks_empty() {
    let empty: &[&[u8]] = &[];
    assert_eq!(checksum_chunks::<Crc32c>(empty), Crc32c::checksum(b""));
  }

  #[test]
  fn test_checksum_chunks_single() {
    let data = b"single chunk";
    let chunks: &[&[u8]] = &[data];
    assert_eq!(checksum_chunks::<Crc32c>(chunks), Crc32c::checksum(data));
  }

  #[test]
  fn test_combine_checksums() {
    let data = b"hello world";
    let (a, b) = data.split_at(6);

    let checksums = [(Crc32c::checksum(a), a.len()), (Crc32c::checksum(b), b.len())];

    assert_eq!(combine_checksums::<Crc32c>(&checksums), Crc32c::checksum(data));
  }

  #[test]
  fn test_combine_checksums_or_empty() {
    let empty: &[(u32, usize)] = &[];
    assert_eq!(combine_checksums_or::<Crc32c>(empty), None);
  }

  #[test]
  fn test_combine_checksums_or_single() {
    let single = [(Crc32c::checksum(b"test"), 4)];
    assert_eq!(combine_checksums_or::<Crc32c>(&single), Some(Crc32c::checksum(b"test")));
  }

  #[test]
  fn test_checksum_iter() {
    let chunks = [b"hello ".as_slice(), b"world".as_slice()];
    let crc = checksum_iter::<_, Crc32c>(chunks.iter().copied()).finalize();
    assert_eq!(crc, Crc32c::checksum(b"hello world"));
  }

  #[test]
  fn test_checksum_iter_owned() {
    extern crate alloc;
    use alloc::vec::Vec;

    let chunks: Vec<Vec<u8>> = alloc::vec![b"The ".to_vec(), b"quick ".to_vec(), b"fox".to_vec()];

    let crc = checksum_iter::<_, Crc32c>(chunks.iter()).finalize();
    assert_eq!(crc, Crc32c::checksum(b"The quick fox"));
  }

  #[test]
  fn test_checksum_iter_empty() {
    let empty: [&[u8]; 0] = [];
    let crc = checksum_iter::<_, Crc32c>(empty.iter().copied()).finalize();
    assert_eq!(crc, Crc32c::checksum(b""));
  }

  // Property: checksum_chunks should always equal direct checksum
  #[test]
  fn test_property_chunks_equal_direct() {
    // Various data patterns
    let patterns: &[&[u8]] = &[
      b"",
      b"a",
      b"ab",
      b"abc",
      b"The quick brown fox jumps over the lazy dog",
      &[0u8; 256],
      &[0xFFu8; 256],
    ];

    for &data in patterns {
      for chunk_size in 1..=16 {
        if chunk_size > data.len() && !data.is_empty() {
          continue;
        }

        let chunks: alloc::vec::Vec<&[u8]> = if data.is_empty() {
          alloc::vec![]
        } else {
          data.chunks(chunk_size).collect()
        };

        assert_eq!(
          checksum_chunks::<Crc32c>(&chunks),
          Crc32c::checksum(data),
          "CRC32c mismatch: data.len()={}, chunk_size={}",
          data.len(),
          chunk_size
        );

        assert_eq!(
          checksum_chunks::<Crc64>(&chunks),
          Crc64::checksum(data),
          "CRC64 mismatch: data.len()={}, chunk_size={}",
          data.len(),
          chunk_size
        );
      }
    }
  }
}
