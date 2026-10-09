//! Bounded input buffering for ordinary and caller-scheduled hashing.

use alloc::{vec, vec::Vec};
use std::io::{self, Read};

use super::Blake3;
use crate::traits::{Digest, ct};

pub(super) const BUFFER_LEN: usize = 1024 * 1024;
const ALIGNMENT: usize = 64;

struct ReadBuffer(Vec<u8>);

impl Drop for ReadBuffer {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

pub(super) fn read_into(
  reader: &mut (impl Read + ?Sized),
  mut remaining: u64,
  mut update: impl FnMut(&[u8]),
) -> io::Result<u64> {
  if remaining == 0 {
    return Ok(0);
  }
  // Leave enough initialized padding to align the usable region without moving
  // input or growing its allocation. Drop clears the entire allocation's length.
  let mut storage = ReadBuffer(vec![0; BUFFER_LEN.strict_add(ALIGNMENT.strict_sub(1))]);
  let offset = storage.0.as_ptr().align_offset(ALIGNMENT);
  let buffer = &mut storage.0[offset..offset.strict_add(BUFFER_LEN)];
  let mut total = 0u64;
  loop {
    let limit = usize::try_from(remaining.min(BUFFER_LEN as u64)).expect("read capacity fits usize");
    let mut filled = 0usize;
    let mut result = Ok(());
    while filled < limit {
      match reader.read(&mut buffer[filled..limit]) {
        Ok(0) => break,
        Ok(count) => filled = filled.strict_add(count),
        Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
        Err(error) => {
          result = Err(error);
          break;
        }
      }
    }
    // Commit successful reads even when the next read failed. The reader has
    // already consumed those bytes, so leaving them buffered would lose input.
    if filled != 0 {
      update(&buffer[..filled]);
    }
    result?;
    let count = u64::try_from(filled).expect("read capacity fits u64");
    total = total.strict_add(count);
    remaining = remaining.strict_sub(count);
    if filled < limit || remaining == 0 {
      return Ok(total);
    }
  }
}

impl Blake3 {
  /// Read through EOF and append the bytes to this hash state.
  ///
  /// Coalesces short reads into an aligned 1 MiB buffer before hashing. This
  /// uses bounded heap storage, which is cleared before deallocation in every
  /// mode. With `parallel`, Linux AArch64 may hash complete buffers
  /// in the current Rayon pool. Keyed hashing, derive-key hashing, and partial
  /// buffers use [`Digest::update`]. Requires `std`; it does not require `parallel`.
  ///
  /// Returns the number of bytes read by this call. Existing input stays in the
  /// state, and further updates and finalization remain available. To read a
  /// fixed range, pass a [`Read::take`] adapter and check the returned count.
  ///
  /// # Errors
  ///
  /// Retries [`io::ErrorKind::Interrupted`]. Returns any other reader error after
  /// absorbing all bytes returned by preceding successful reads, including a
  /// partially filled buffer. The state is not rolled back on failure.
  ///
  /// # Example
  ///
  /// ```
  /// use rscrypto::{Blake3, Digest};
  /// let mut hasher = Blake3::new();
  /// hasher.update(b"prefix:");
  /// assert_eq!(hasher.update_reader(&mut &b"contents"[..])?, 8);
  /// assert_eq!(hasher.finalize(), Blake3::digest(b"prefix:contents"));
  /// # Ok::<(), std::io::Error>(())
  /// ```
  pub fn update_reader(&mut self, reader: &mut (impl Read + ?Sized)) -> io::Result<u64> {
    read_into(reader, u64::MAX, |input| {
      #[cfg(all(
        feature = "parallel",
        target_os = "linux",
        target_endian = "little",
        target_arch = "aarch64"
      ))]
      if self.try_parallel_reader_update(input) {
        return;
      }
      self.update(input);
    })
  }
}
