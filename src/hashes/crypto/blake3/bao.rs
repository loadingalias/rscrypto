//! Bao combined-format verification through the public BLAKE3 subtree API.
//!
//! Format: <https://github.com/oconnor663/bao/blob/0.13.1/docs/spec.md>.

use core::{fmt, ops::Range};
use std::io::{self, Read};

use super::{
  Blake3, CHUNK_LEN,
  tree::{Blake3ChainingValue, Blake3Tree},
};
use crate::traits::ct;

// A u64 byte length contains at most 2^54 BLAKE3 chunks. The traversal keeps
// only the right sibling at each depth; its left child stays in a local value.
const MAX_DEPTH: usize = 54;

struct ChunkBuffer([u8; CHUNK_LEN]);

impl Drop for ChunkBuffer {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
  Header,
  Reading,
  Complete,
  Failed,
}

/// Read and verify a Bao combined encoding against a trusted BLAKE3 root.
///
/// Each returned byte has been verified at its position in the original input.
/// A read returns at most one 1,024-byte chunk. Memory use is bounded independently
/// of the encoded length. The decoder uses unkeyed BLAKE3 and requires `std`.
/// Outboard encodings, slices and seeking are not supported.
///
/// Obtain `root` independently of the untrusted encoding. Reading to EOF verifies
/// the whole input, including its length. Stopping early verifies only the bytes
/// returned so far. The unverified length header is never exposed. Empty input
/// is verified before a nonempty read can report EOF. A read into an empty buffer
/// performs no verification. Bytes after the combined encoding remain unread.
///
/// # Errors
///
/// A hash mismatch returns [`io::ErrorKind::InvalidData`]; a truncated header,
/// parent or chunk returns [`io::ErrorKind::UnexpectedEof`]. Interrupted reads
/// are retried. Other reader errors propagate unchanged. Any error permanently
/// stops decoding: subsequent nonempty reads return `InvalidData`. Previously
/// returned bytes remain verified. An error does not modify the caller's buffer.
///
/// # Example
///
/// ```
/// use std::io::Read;
/// use rscrypto::{Blake3, hashes::expert::bao::Decoder};
///
/// // A one-chunk Bao encoding is its length followed by the input bytes.
/// let input = b"verified contents";
/// let root = Blake3::digest(input); // In practice, receive this through a trusted channel.
/// let mut encoded = (input.len() as u64).to_le_bytes().to_vec();
/// encoded.extend_from_slice(input);
/// let mut decoder = Decoder::new(encoded.as_slice(), &root);
/// let mut output = Vec::new();
/// decoder.read_to_end(&mut output)?;
/// assert_eq!(output, input);
/// # Ok::<(), std::io::Error>(())
/// ```
pub struct Decoder<R> {
  inner: R,
  tree: Blake3Tree,
  root: [u8; 32],
  pending: [Option<Blake3ChainingValue>; MAX_DEPTH],
  pending_len: usize,
  chunk: ChunkBuffer,
  available: Range<usize>,
  state: State,
}

impl<R> Decoder<R> {
  /// Wrap an untrusted combined encoding and an independently trusted root.
  ///
  /// Performs no I/O. Verification starts on the first nonempty read.
  #[must_use]
  pub fn new(inner: R, root: &[u8; 32]) -> Self {
    Self {
      inner,
      tree: Blake3Tree::new(),
      root: *root,
      pending: [const { None }; MAX_DEPTH],
      pending_len: 0,
      chunk: ChunkBuffer([0; CHUNK_LEN]),
      available: 0..0,
      state: State::Header,
    }
  }

  /// Return the underlying reader without reading or verifying more input.
  ///
  /// This does not establish that the whole input was verified. Buffered,
  /// already verified bytes that have not been returned are discarded.
  #[must_use]
  pub fn into_inner(self) -> R {
    self.inner
  }
}

impl<R: Read> Decoder<R> {
  fn read_parent(&mut self, offset: u64, len: u64) -> io::Result<(Blake3ChainingValue, Blake3ChainingValue)> {
    let left_len = Blake3Tree::left_subtree_len(len).ok_or_else(invalid_data)?;
    let mut bytes = [0u8; 64];
    self.inner.read_exact(&mut bytes)?;
    let (left_bytes, right_bytes) = bytes.split_at(32);
    let left = self
      .tree
      .chaining_value(
        left_bytes.try_into().expect("left parent half is 32 bytes"),
        offset,
        left_len,
      )
      .map_err(|_| invalid_data())?;
    let right = self
      .tree
      .chaining_value(
        right_bytes.try_into().expect("right parent half is 32 bytes"),
        offset.checked_add(left_len).ok_or_else(invalid_data)?,
        len.strict_sub(left_len),
      )
      .map_err(|_| invalid_data())?;
    Ok((left, right))
  }

  fn push_right(&mut self, right: Blake3ChainingValue) -> io::Result<()> {
    let slot = self.pending.get_mut(self.pending_len).ok_or_else(invalid_data)?;
    *slot = Some(right);
    self.pending_len = self.pending_len.strict_add(1);
    Ok(())
  }

  fn read_chunk(&mut self, mut node: Blake3ChainingValue) -> io::Result<()> {
    while node.len() > CHUNK_LEN as u64 {
      let (left, right) = self.read_parent(node.input_offset(), node.len())?;
      let actual = self.tree.merge(&left, &right).map_err(|_| invalid_data())?;
      if !ct::fixed_eq(actual.as_bytes(), node.as_bytes()).declassify() {
        return Err(invalid_data());
      }
      self.push_right(right)?;
      node = left;
    }
    let len = usize::try_from(node.len()).expect("one chunk fits usize");
    self.inner.read_exact(&mut self.chunk.0[..len])?;
    let mut subtree = self.tree.subtree(node.input_offset()).map_err(|_| invalid_data())?;
    subtree.update(&self.chunk.0[..len]).map_err(|_| invalid_data())?;
    let actual = subtree.finalize().map_err(|_| invalid_data())?;
    if !ct::fixed_eq(actual.as_bytes(), node.as_bytes()).declassify() {
      return Err(invalid_data());
    }
    self.available = 0..len;
    Ok(())
  }

  fn start(&mut self) -> io::Result<()> {
    let mut header = [0u8; 8];
    self.inner.read_exact(&mut header)?;
    let len = u64::from_le_bytes(header);
    if len <= CHUNK_LEN as u64 {
      let len = usize::try_from(len).expect("one chunk fits usize");
      self.inner.read_exact(&mut self.chunk.0[..len])?;
      let actual = Blake3::digest(&self.chunk.0[..len]);
      if !ct::fixed_eq(&actual, &self.root).declassify() {
        return Err(invalid_data());
      }
      self.available = 0..len;
      return Ok(());
    }
    let (left, right) = self.read_parent(0, len)?;
    let actual = self.tree.merge_root(&left, &right).map_err(|_| invalid_data())?;
    if !ct::fixed_eq(&actual, &self.root).declassify() {
      return Err(invalid_data());
    }
    self.push_right(right)?;
    self.read_chunk(left)
  }

  fn fill(&mut self) -> io::Result<()> {
    let previous = self.state;
    // An I/O error or a caught panic must never let a later read skip the node
    // whose bytes the underlying reader already consumed.
    self.state = State::Failed;
    let result = if previous == State::Header {
      self.start()
    } else {
      self.pending_len = self.pending_len.strict_sub(1);
      let node = self.pending[self.pending_len].take().ok_or_else(invalid_data)?;
      self.read_chunk(node)
    };
    if let Err(error) = result {
      ct::zeroize(&mut self.chunk.0);
      self.pending.fill_with(|| None);
      self.pending_len = 0;
      return Err(error);
    }
    self.state = if self.pending_len == 0 {
      State::Complete
    } else {
      State::Reading
    };
    Ok(())
  }
}

impl<R: Read> Read for Decoder<R> {
  fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
    if output.is_empty() {
      return Ok(0);
    }
    if self.state == State::Failed {
      return Err(invalid_data());
    }
    if self.available.is_empty() && self.state != State::Complete {
      self.fill()?;
    }
    let count = output.len().min(self.available.len());
    output[..count].copy_from_slice(&self.chunk.0[self.available.start..self.available.start.strict_add(count)]);
    self.available.start = self.available.start.strict_add(count);
    Ok(count)
  }
}

impl<R> fmt::Debug for Decoder<R> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    // Neither the unauthenticated header nor buffered content is diagnostic data.
    f.debug_struct("Decoder").finish_non_exhaustive()
  }
}

fn invalid_data() -> io::Error {
  io::Error::new(io::ErrorKind::InvalidData, "Bao verification failed")
}
