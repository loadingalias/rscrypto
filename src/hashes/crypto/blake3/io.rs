//! Reader input for ordinary and caller-scheduled hashing.
//!
//! A safe reader needs initialized bytes, and any byte a reader receives may
//! hold input. Storage is therefore zeroed only when a read needs it, and every
//! zeroed byte is cleared on release, so both costs follow the input length.
//! The first read receives 1 KiB, and the buffer doubles each time the reader
//! fills it until it holds one batch. When the reader has filled every
//! initialized byte, a 64-byte read checks for end of input before more storage
//! is zeroed, moved to the heap, or allocated. Inputs that fit the stack buffer
//! never allocate, and the heap buffer is reserved once and never reallocated.
//!
//! A serial hasher receives one update per full buffer. Its buffer starts at
//! 16 KiB and is reused; it doubles only after eight times its size has been
//! read, up to 1 MiB. Short inputs stay in the level-1 data cache, while long
//! inputs and files reach 1 MiB reads and the zeroed bytes stay a small fraction
//! of the input. A hasher that may use threads receives whole 1 MiB buffers.
//! Every update ends at a multiple of the current buffer size within this
//! call's input, so a reader panic leaves at most one buffer unabsorbed.

use alloc::vec::Vec;
use core::mem::MaybeUninit;
use std::io::{self, Read};

use super::Blake3;
use crate::traits::{Digest, ct};

/// The largest read, and the buffer a hasher that may use threads receives whole.
pub(super) const BUFFER_LEN: usize = 1024 * 1024;
/// The first serial buffer size. It stays within a 32 KiB level-1 data cache
/// with room for the source, so copying, zeroing, and clearing it stay cheap.
const SERIAL_BATCH: usize = 16 * 1024;
/// Stack storage, the same size as upstream's reader buffer. Inputs up to this
/// size never allocate, and serial reads stay here through 1 MiB of input, which
/// avoids allocator costs that vary between processes for 1 MiB blocks.
const STACK_LEN: usize = 64 * 1024;
const FIRST_READ: usize = 1024;
/// The end-of-input check, sized to one BLAKE3 block.
const PROBE_LEN: usize = 64;
/// A serial read larger than one batch waits until this many reads of its size
/// have been read.
const GROWTH_DIVISOR: u64 = 8;
const ALIGNMENT: usize = 64;

#[repr(C, align(64))]
struct StackBytes([MaybeUninit<u8>; STACK_LEN + PROBE_LEN]);

struct HeapBytes {
  storage: Vec<u8>,
  /// Initialized padding that aligns the usable region.
  offset: usize,
}

/// Input storage that gives readers only initialized bytes.
///
/// Drop clears every initialized byte, which includes every byte any reader
/// received. Storage is never zeroed ahead of the read that needs it.
struct ReadBuffer {
  stack: StackBytes,
  /// `stack.0[..stack_len]` is initialized; the rest is untouched.
  stack_len: usize,
  heap: Option<HeapBytes>,
}

impl ReadBuffer {
  #[inline(always)]
  fn new() -> Self {
    Self {
      stack: StackBytes([MaybeUninit::uninit(); STACK_LEN + PROBE_LEN]),
      stack_len: 0,
      heap: None,
    }
  }

  #[inline(always)]
  fn stack_bytes(&mut self) -> &mut [u8] {
    let initialized = &mut self.stack.0[..self.stack_len];
    // SAFETY: `stack_len` starts at zero and only `extend_to` advances it,
    // after writing every byte below the new length. Code outside this type
    // writes only through `&mut [u8]`, which keeps bytes initialized.
    unsafe { initialized.assume_init_mut() }
  }

  /// Initialized bytes of the active storage.
  #[inline(always)]
  fn len(&self) -> usize {
    match &self.heap {
      Some(heap) => heap.storage.len().strict_sub(heap.offset),
      None => self.stack_len,
    }
  }

  /// The most bytes the active storage can hold.
  #[inline(always)]
  fn limit(&self) -> usize {
    if self.heap.is_some() {
      BUFFER_LEN
    } else {
      STACK_LEN.strict_add(PROBE_LEN)
    }
  }

  /// Zero the active storage up to `end`.
  #[inline(always)]
  fn extend_to(&mut self, end: usize) {
    debug_assert!(end <= self.limit());
    match &mut self.heap {
      // Within the reserved capacity, so the storage is never reallocated.
      Some(heap) => {
        let len = heap.offset.strict_add(end);
        if len > heap.storage.len() {
          heap.storage.resize(len, 0);
        }
      }
      None => {
        if end > self.stack_len {
          self.stack.0[self.stack_len..end].fill(MaybeUninit::new(0));
          self.stack_len = end;
        }
      }
    }
  }

  #[inline(always)]
  fn bytes(&mut self) -> &mut [u8] {
    if self.heap.is_none() {
      return self.stack_bytes();
    }
    // Check the storage kind before borrowing it: the MSRV cannot retain the conditional heap
    // borrow across the stack branch. No mutation can change the kind before this borrow.
    let heap = self.heap.as_mut().expect("heap storage selected above");
    &mut heap.storage[heap.offset..]
  }

  /// Continue on the heap with the first `keep` stack bytes.
  #[cold]
  #[inline(never)]
  fn promote(&mut self, keep: usize) {
    debug_assert!(self.heap.is_none());
    let mut storage = Vec::<u8>::with_capacity(BUFFER_LEN.strict_add(ALIGNMENT.strict_sub(1)));
    // At run time this is the exact offset; a larger value would only cost alignment.
    let offset = storage.as_ptr().align_offset(ALIGNMENT).min(ALIGNMENT.strict_sub(1));
    storage.resize(offset, 0);
    storage.extend_from_slice(&self.stack_bytes()[..keep]);
    self.heap = Some(HeapBytes { storage, offset });
  }
}

impl Drop for ReadBuffer {
  fn drop(&mut self) {
    ct::zeroize(self.stack_bytes());
    if let Some(heap) = &mut self.heap {
      ct::zeroize(&mut heap.storage);
    }
  }
}

/// Hash one buffer of pending input, if any.
#[inline(always)]
fn absorb(update: &mut impl FnMut(&[u8]), input: &[u8]) {
  if !input.is_empty() {
    update(input);
  }
}

/// Read through EOF or `remaining` bytes, updating once per full buffer.
///
/// Reads coalesce until the buffer holds `batch` bytes, which is
/// [`SERIAL_BATCH`] or [`BUFFER_LEN`]; serial buffers may then grow. Each
/// update receives one whole buffer, or the pending input at the end.
pub(super) fn read_into(
  reader: &mut (impl Read + ?Sized),
  mut remaining: u64,
  batch: usize,
  mut update: impl FnMut(&[u8]),
) -> io::Result<u64> {
  debug_assert!(batch == SERIAL_BATCH || batch == BUFFER_LEN);
  if remaining == 0 {
    return Ok(0);
  }
  let mut buffer = ReadBuffer::new();
  // Coalesce reads until `filled == capacity`.
  let mut capacity = FIRST_READ;
  let mut filled = 0usize;
  let mut total = 0u64;
  // The last new storage was a probe, so the reader has more input.
  let mut probed = false;
  loop {
    let initialized = buffer.len();
    let end = if filled < initialized {
      // Offer initialized storage before zeroing more.
      initialized.min(capacity)
    } else if filled != 0 && !probed {
      // The reader filled every initialized byte. Check for end of input
      // before zeroing, moving, or allocating storage for more.
      probed = true;
      filled.strict_add(PROBE_LEN.min(capacity.strict_sub(filled)))
    } else {
      probed = false;
      if capacity > buffer.limit() {
        // Moving the pending input keeps every update boundary at a multiple
        // of the buffer size, so a failure loses no more than one buffer.
        buffer.promote(filled);
      }
      capacity
    };
    let room = end
      .strict_sub(filled)
      .min(usize::try_from(remaining).unwrap_or(usize::MAX));
    let end = filled.strict_add(room);
    buffer.extend_to(end);
    let window = &mut buffer.bytes()[filled..end];
    let result = loop {
      match reader.read(window) {
        Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
        result => break result,
      }
    };
    let count = match result {
      Ok(0) => {
        absorb(&mut update, &buffer.bytes()[..filled]);
        return Ok(total);
      }
      Ok(count) => count,
      Err(error) => {
        // Commit successful reads. The reader has already consumed those
        // bytes, so leaving them buffered would lose input.
        absorb(&mut update, &buffer.bytes()[..filled]);
        return Err(error);
      }
    };
    let unread = room
      .checked_sub(count)
      .expect("Read::read returned more bytes than the buffer holds");
    filled = end.strict_sub(unread);
    let count = u64::try_from(count).expect("read length fits u64");
    total = total.strict_add(count);
    remaining = remaining.strict_sub(count);
    if remaining == 0 {
      absorb(&mut update, &buffer.bytes()[..filled]);
      return Ok(total);
    }
    if filled < capacity {
      continue;
    }
    if capacity < batch {
      capacity = capacity.strict_mul(2).min(batch);
      continue;
    }
    absorb(&mut update, &buffer.bytes()[..filled]);
    filled = 0;
    // Serial reads grow with the input; whole-buffer reads are already largest.
    let target = usize::try_from(total / GROWTH_DIVISOR).map_or(BUFFER_LEN, |target| target.min(BUFFER_LEN));
    if let Some(log) = target.checked_ilog2() {
      capacity = capacity.max(1usize << log);
    }
  }
}

impl Blake3 {
  /// The update size a reader should give this hasher.
  #[inline]
  pub(super) fn reader_batch(&self, reader_path: bool) -> usize {
    #[cfg(feature = "parallel")]
    if self.reader_may_thread_full_buffers(reader_path) {
      return BUFFER_LEN;
    }
    #[cfg(not(feature = "parallel"))]
    let _ = reader_path;
    SERIAL_BATCH
  }

  /// Read through EOF and append the bytes to this hash state.
  ///
  /// Reads start at 1 KiB and grow as the reader fills them, up to 1 MiB for
  /// long inputs. Storage is zeroed only as reads need it: inputs that fit a
  /// 64 KiB stack buffer never allocate, and longer inputs may continue in one
  /// heap buffer of up to 1 MiB that is allocated once and never resized.
  /// Every byte passed to the reader is cleared before its storage is released,
  /// in every mode, including bytes written by a reader that then fails.
  ///
  /// Each full buffer is one [`Digest::update`]. Serial buffers start at 16 KiB
  /// and grow with the input. With `parallel`, a plain hasher at a 1 MiB
  /// boundary on Linux AArch64, and any hasher whose CPU profile can admit
  /// threads in [`Digest::update`], instead coalesces whole 1 MiB buffers;
  /// Linux AArch64 may hash them in the current Rayon pool. Requires `std`; it
  /// does not require `parallel`.
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
    let batch = self.reader_batch(true);
    read_into(reader, u64::MAX, batch, |input| {
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

#[cfg(test)]
mod tests {
  use alloc::vec::Vec;
  use std::io::{self, Read};

  use super::{BUFFER_LEN, SERIAL_BATCH, STACK_LEN, read_into};

  /// Fills at most `max` bytes of each read and counts the bytes consumed.
  /// It first reads every byte it receives, which a safe reader may do, so
  /// Miri rejects any uninitialized byte in the offered storage.
  struct Source<'a> {
    input: &'a [u8],
    position: usize,
    max: usize,
  }

  impl Read for Source<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
      core::hint::black_box(buffer.iter().fold(0u8, |acc, &byte| acc ^ byte));
      let count = buffer
        .len()
        .min(self.max)
        .min(self.input.len().strict_sub(self.position));
      let end = self.position.strict_add(count);
      buffer[..count].copy_from_slice(&self.input[self.position..end]);
      self.position = end;
      Ok(count)
    }
  }

  /// Runs the production storage, including its initialized-prefix view, with
  /// inputs small enough for Miri. Updates must reproduce the input in order,
  /// every full-buffer update must end at a multiple of its power-of-two
  /// length, and reads must stop at the caller's limit.
  #[test]
  fn miri_reader_storage_preserves_input_and_buffer_boundaries() {
    let input: Vec<u8> = (0..STACK_LEN.strict_add(200))
      .map(|i| u8::try_from(i % 251).expect("byte pattern"))
      .collect();
    const S: usize = SERIAL_BATCH;
    const B: usize = BUFFER_LEN;
    // Under Miri, each transition once: the first reads, the end-of-input
    // check, reuse of initialized bytes after a full serial buffer, and the
    // move to the heap after the check past the stack buffer.
    let cases: Vec<(usize, usize)> = if cfg!(miri) {
      [
        (0, S),
        (1, S),
        (64, S),
        (65, B),
        (1024, S),
        (1088, B),
        (1089, S),
        (S + 1, S),
        (STACK_LEN + 65, B),
      ]
      .to_vec()
    } else {
      [
        0,
        1,
        63,
        64,
        65,
        1023,
        1024,
        1025,
        1088,
        1089,
        4096,
        S,
        S + 1,
        STACK_LEN,
        STACK_LEN + 64,
        STACK_LEN + 65,
      ]
      .into_iter()
      .flat_map(|len| [(len, S), (len, B)])
      .collect()
    };
    for (len, batch) in cases {
      let available = u64::try_from(len).expect("length fits u64");
      // Byte-at-a-time reads and limits keep runs short only for short inputs.
      let short = len <= if cfg!(miri) { 65 } else { 2048 };
      let maxima: &[usize] = if short { &[usize::MAX, 1] } else { &[usize::MAX] };
      let limits: &[u64] = if short { &[u64::MAX, available / 2] } else { &[u64::MAX] };
      for &max in maxima {
        for &limit in limits {
          let mut source = Source {
            input: &input[..len],
            position: 0,
            max,
          };
          let mut updates = Vec::new();
          let read = read_into(&mut source, limit, batch, |part| updates.push(part.to_vec())).expect("in-memory reads");
          let expected = available.min(limit);
          let expected_len = usize::try_from(expected).expect("length fits usize");
          assert_eq!(read, expected, "len={len} batch={batch} max={max} limit={limit}");
          assert_eq!(source.position, expected_len, "no read past the limit");
          assert_eq!(updates.concat(), &input[..expected_len]);
          if let Some((_, full)) = updates.split_last() {
            let mut end = 0usize;
            for part in full {
              end = end.strict_add(part.len());
              assert!(
                part.len().is_power_of_two() && end.is_multiple_of(part.len()),
                "len={len} batch={batch} part={}",
                part.len()
              );
            }
          }
        }
      }
    }
  }
}
