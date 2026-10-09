//! BLAKE3 subtree hashing: non-root chaining values and their merges.
//!
//! Every chaining value records the input range and mode that produced it,
//! so a merge rejects any pair that is not a valid BLAKE3 parent node instead
//! of returning an unrelated hash.

use super::{
  Blake3, Blake3DeriveContext, Blake3XofReader, CHUNK_LEN, ChunkState, DERIVE_KEY_MATERIAL, IV, KEYED_HASH, OUT_LEN,
  RootEmitState, dispatch, kernels, parent_output, words8_from_le_bytes_32, words8_to_le_bytes,
};
use crate::traits::{Digest as _, ct};

const CHUNK_LEN_U64: u64 = CHUNK_LEN as u64;

/// Invalid subtree position, length, or merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Blake3SubtreeError {
  /// The input offset is not a multiple of the 1,024-byte chunk length.
  UnalignedOffset,
  /// The input would extend past the largest subtree that starts at its offset.
  TooLong,
  /// A subtree needs at least one input byte.
  Empty,
  /// The two chaining values do not form one BLAKE3 parent node.
  InvalidMerge,
  /// A chaining value belongs to a tree with a different mode.
  ModeMismatch,
  /// A merge level needs an even number of child chaining values.
  OddChildCount,
  /// A merge level needs one output slot per pair of child chaining values.
  OutputLengthMismatch,
}

impl core::fmt::Display for Blake3SubtreeError {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.write_str(match self {
      Self::UnalignedOffset => "BLAKE3 subtree offset is not a multiple of 1024 bytes",
      Self::TooLong => "BLAKE3 subtree input exceeds the largest subtree at its offset",
      Self::Empty => "BLAKE3 subtree has no input",
      Self::InvalidMerge => "BLAKE3 chaining values do not form a parent node",
      Self::ModeMismatch => "BLAKE3 chaining value belongs to a tree with a different mode",
      Self::OddChildCount => "BLAKE3 merge level needs an even number of children",
      Self::OutputLengthMismatch => "BLAKE3 merge level output length does not match its parent count",
    })
  }
}

impl core::error::Error for Blake3SubtreeError {}

/// The mode and key of one BLAKE3 hash tree.
///
/// Hash parts of one input as independent subtrees, possibly on different
/// threads or machines, then merge their chaining values into the root. The
/// root equals [`Blake3::digest`], [`Blake3::keyed_digest`], or
/// [`Blake3::derive_key_with`] over the whole input.
///
/// In keyed mode this value holds the key; dropping it clears the key.
///
/// # Examples
///
/// ```
/// use rscrypto::{Blake3, hashes::expert::blake3_tree::Blake3Tree};
///
/// let input = vec![7u8; 5000];
/// let tree = Blake3Tree::new();
/// let split = Blake3Tree::left_subtree_len(input.len() as u64).unwrap() as usize;
///
/// let mut left = tree.subtree(0)?;
/// left.update(&input[..split])?;
/// let mut right = tree.subtree(split as u64)?;
/// right.update(&input[split..])?;
///
/// let root = tree.merge_root(&left.finalize()?, &right.finalize()?)?;
/// assert_eq!(root, Blake3::digest(&input));
/// # Ok::<(), rscrypto::hashes::expert::blake3_tree::Blake3SubtreeError>(())
/// ```
#[derive(Clone)]
pub struct Blake3Tree {
  key_words: [u32; 8],
  flags: u32,
}

impl Blake3Tree {
  /// A tree for the unkeyed hash, [`Blake3::digest`].
  #[must_use]
  pub const fn new() -> Self {
    Self {
      key_words: IV,
      flags: 0,
    }
  }

  /// A tree for the keyed hash, [`Blake3::keyed_digest`].
  #[must_use]
  pub fn keyed(key: &[u8; 32]) -> Self {
    Self {
      key_words: words8_from_le_bytes_32(key),
      flags: KEYED_HASH,
    }
  }

  /// A tree for key derivation, [`Blake3::derive_key_with`].
  #[must_use]
  pub const fn derive_key(context: &Blake3DeriveContext) -> Self {
    Self {
      key_words: context.key_words,
      flags: DERIVE_KEY_MATERIAL,
    }
  }

  /// The length of the left child of an input or subtree of `input_len` bytes.
  ///
  /// Split an input here to hash it as two subtrees; the rest is the right
  /// child. Returns `None` for `input_len <= 1024`: one chunk has no children.
  #[must_use]
  pub const fn left_subtree_len(input_len: u64) -> Option<u64> {
    if input_len <= CHUNK_LEN_U64 {
      return None;
    }
    // The largest power of two strictly below `input_len`; at least one chunk.
    Some(1u64 << input_len.strict_sub(1).ilog2())
  }

  /// Start a subtree whose input begins `input_offset` bytes into the whole input.
  ///
  /// # Errors
  ///
  /// Returns [`Blake3SubtreeError::UnalignedOffset`] unless `input_offset` is
  /// a multiple of 1,024 bytes.
  pub fn subtree(&self, input_offset: u64) -> Result<Blake3Subtree, Blake3SubtreeError> {
    if !input_offset.is_multiple_of(CHUNK_LEN_U64) {
      return Err(Blake3SubtreeError::UnalignedOffset);
    }
    let chunk_counter = input_offset / CHUNK_LEN_U64;
    // A subtree that starts at chunk N > 0 spans at most the largest power of
    // two dividing N; beyond that it would absorb chunks to its left.
    let max_len = (chunk_counter != 0).then(|| CHUNK_LEN_U64 << chunk_counter.trailing_zeros());
    // No input can reach past `u64::MAX` bytes.
    let remaining = u64::MAX.strict_sub(input_offset);
    let max_len = max_len.map_or(remaining, |max| max.min(remaining));

    let mut hasher = Blake3::new_internal(self.key_words, self.flags);
    hasher.chunk_state = ChunkState::new(self.key_words, chunk_counter, self.flags, hasher.chunk_state.kernel_id);
    Ok(Blake3Subtree {
      hasher,
      input_offset,
      len: 0,
      max_len,
    })
  }

  /// Merge two adjacent subtrees into their non-root parent.
  ///
  /// # Errors
  ///
  /// Returns [`Blake3SubtreeError::ModeMismatch`] if either value comes from a
  /// tree with another mode, and [`Blake3SubtreeError::InvalidMerge`] unless
  /// `left` is the complete left child and `right` the right child of one
  /// parent node.
  pub fn merge(
    &self,
    left: &Blake3ChainingValue,
    right: &Blake3ChainingValue,
  ) -> Result<Blake3ChainingValue, Blake3SubtreeError> {
    let len = self.check_merge(left, right)?;
    let (mut left_words, mut right_words) = (left.words(), right.words());
    let mut words = kernels::parent_cv_inline(merge_kernel(), left_words, right_words, self.key_words, self.flags);
    let cv = Blake3ChainingValue {
      bytes: words8_to_le_bytes(&words),
      input_offset: left.input_offset,
      len,
      flags: self.flags,
    };
    ct::zeroize_words_no_fence(&mut left_words);
    ct::zeroize_words_no_fence(&mut right_words);
    ct::zeroize_words_no_fence(&mut words);
    ct::zeroize_fence();
    Ok(cv)
  }

  /// Merge adjacent pairs of child chaining values into non-root parents.
  ///
  /// Writes `merge(&children[2 * i], &children[2 * i + 1])` to `out[i]`.
  /// Each pair must form a valid parent, but separate pairs need not be
  /// adjacent to each other or have the same height. An unpaired right edge
  /// must be carried to the next level by the caller. The final two children
  /// need [`Self::merge_root`] or [`Self::merge_root_xof`] to produce a root.
  ///
  /// This method allocates nothing and uses bounded scratch, cleared before
  /// return or unwind. Existing output values are replaced and cleared by
  /// their destructors. Initialize reusable output slots by cloning existing
  /// chaining values. Empty input and output slices succeed without work.
  ///
  /// # Errors
  ///
  /// Checks the following conditions in order:
  ///
  /// 1. Returns [`Blake3SubtreeError::OddChildCount`] if `children.len()` is odd.
  /// 2. Returns [`Blake3SubtreeError::OutputLengthMismatch`] unless
  ///    `out.len() == children.len() / 2`.
  /// 3. Validates every pair in input order with the same checks as
  ///    [`Self::merge`], returning the first pair's error.
  ///
  /// Any error leaves every output value unchanged.
  ///
  /// # Example
  ///
  /// ```
  /// use rscrypto::{Blake3, hashes::expert::blake3_tree::Blake3Tree};
  ///
  /// let tree = Blake3Tree::new();
  /// let input = [7u8; 4096];
  /// let mut children = Vec::new();
  /// for (index, chunk) in input.chunks(1024).enumerate() {
  ///     let mut subtree = tree.subtree(index as u64 * 1024)?;
  ///     subtree.update(chunk)?;
  ///     children.push(subtree.finalize()?);
  /// }
  /// let mut parents = children[..2].to_vec();
  /// tree.merge_level(&children, &mut parents)?;
  /// assert_eq!(tree.merge_root(&parents[0], &parents[1])?, Blake3::digest(&input));
  /// # Ok::<(), rscrypto::hashes::expert::blake3_tree::Blake3SubtreeError>(())
  /// ```
  pub fn merge_level(
    &self,
    children: &[Blake3ChainingValue],
    out: &mut [Blake3ChainingValue],
  ) -> Result<(), Blake3SubtreeError> {
    if !children.len().is_multiple_of(2) {
      return Err(Blake3SubtreeError::OddChildCount);
    }
    if out.len() != children.len() / 2 {
      return Err(Blake3SubtreeError::OutputLengthMismatch);
    }
    for pair in children.as_chunks::<2>().0 {
      self.check_merge(&pair[0], &pair[1])?;
    }
    if out.is_empty() {
      return Ok(());
    }

    let kernel = dispatch::hasher_dispatch().bulk_kernel_for_update(usize::MAX).id;
    let mut scratch = MergeLevelScratch {
      children: [[0; 8]; MERGE_LEVEL_PARENTS * 2],
      parents: [[0; 8]; MERGE_LEVEL_PARENTS],
    };
    for (children, out) in children
      .chunks(MERGE_LEVEL_PARENTS * 2)
      .zip(out.chunks_mut(MERGE_LEVEL_PARENTS))
    {
      for (child, words) in children.iter().zip(scratch.children.iter_mut()) {
        *words = child.words();
      }
      kernels::parent_cvs_many_from_cvs_inline(
        kernel,
        &scratch.children[..children.len()],
        self.key_words,
        self.flags,
        &mut scratch.parents[..out.len()],
      );
      for ((pair, words), parent) in children.as_chunks::<2>().0.iter().zip(scratch.parents.iter()).zip(out) {
        *parent = Blake3ChainingValue {
          bytes: words8_to_le_bytes(words),
          input_offset: pair[0].input_offset,
          len: pair[0].len.strict_add(pair[1].len),
          flags: self.flags,
        };
      }
    }
    Ok(())
  }

  /// Merge the two children of the root into the 32-byte hash.
  ///
  /// # Errors
  ///
  /// Returns the errors of [`Self::merge`], and
  /// [`Blake3SubtreeError::InvalidMerge`] unless `left` starts at offset 0.
  pub fn merge_root(
    &self,
    left: &Blake3ChainingValue,
    right: &Blake3ChainingValue,
  ) -> Result<[u8; OUT_LEN], Blake3SubtreeError> {
    self.check_root_merge(left, right)?;
    Ok(parent_output(merge_kernel(), left.words(), right.words(), self.key_words, self.flags).root_hash_bytes())
  }

  /// Merge the two children of the root into an extendable output reader.
  ///
  /// # Errors
  ///
  /// Returns the errors of [`Self::merge_root`].
  pub fn merge_root_xof(
    &self,
    left: &Blake3ChainingValue,
    right: &Blake3ChainingValue,
  ) -> Result<Blake3XofReader, Blake3SubtreeError> {
    self.check_root_merge(left, right)?;
    Ok(Blake3XofReader::new(RootEmitState::from_parent(
      merge_kernel(),
      left.words(),
      right.words(),
      self.key_words,
      self.flags,
    )))
  }

  /// Rebuild a chaining value received from elsewhere, such as a remote store.
  ///
  /// The caller asserts that `bytes` is the chaining value of the `len` input
  /// bytes that start at `input_offset`, in this tree's mode. A wrong claim
  /// yields a wrong root, which verification against a trusted root detects.
  ///
  /// # Errors
  ///
  /// Returns [`Blake3SubtreeError::UnalignedOffset`],
  /// [`Blake3SubtreeError::Empty`], or [`Blake3SubtreeError::TooLong`] when
  /// no subtree has that position and length.
  pub fn chaining_value(
    &self,
    bytes: [u8; OUT_LEN],
    input_offset: u64,
    len: u64,
  ) -> Result<Blake3ChainingValue, Blake3SubtreeError> {
    let subtree_max = self.subtree(input_offset)?.max_len;
    if len == 0 {
      return Err(Blake3SubtreeError::Empty);
    }
    if len > subtree_max {
      return Err(Blake3SubtreeError::TooLong);
    }
    Ok(Blake3ChainingValue {
      bytes,
      input_offset,
      len,
      flags: self.flags,
    })
  }

  /// Validate a parent merge and return the parent's input length.
  fn check_merge(&self, left: &Blake3ChainingValue, right: &Blake3ChainingValue) -> Result<u64, Blake3SubtreeError> {
    if left.flags != self.flags || right.flags != self.flags {
      return Err(Blake3SubtreeError::ModeMismatch);
    }
    let len = left
      .len
      .checked_add(right.len)
      .ok_or(Blake3SubtreeError::InvalidMerge)?;
    // The children must be adjacent, `left` must be the complete left child of
    // an input of `len` bytes, and the parent must start at a multiple of its
    // complete size.
    let adjacent = left.input_offset.checked_add(left.len) == Some(right.input_offset);
    let left_is_left_child = Self::left_subtree_len(len) == Some(left.len);
    let parent_aligned = left
      .len
      .checked_mul(2)
      .is_none_or(|parent_span| left.input_offset.is_multiple_of(parent_span));
    if adjacent && left_is_left_child && parent_aligned {
      Ok(len)
    } else {
      Err(Blake3SubtreeError::InvalidMerge)
    }
  }

  fn check_root_merge(
    &self,
    left: &Blake3ChainingValue,
    right: &Blake3ChainingValue,
  ) -> Result<(), Blake3SubtreeError> {
    self.check_merge(left, right)?;
    if left.input_offset == 0 {
      Ok(())
    } else {
      Err(Blake3SubtreeError::InvalidMerge)
    }
  }
}

impl Default for Blake3Tree {
  fn default() -> Self {
    Self::new()
  }
}

impl Drop for Blake3Tree {
  fn drop(&mut self) {
    ct::zeroize_words(&mut self.key_words);
  }
}

impl core::fmt::Debug for Blake3Tree {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Blake3Tree")
      .field("mode", &mode_name(self.flags))
      .finish_non_exhaustive()
  }
}

/// A subtree of one BLAKE3 input at a chunk-aligned offset.
///
/// Start one with [`Blake3Tree::subtree`], feed it its input, and finalize it
/// into a [`Blake3ChainingValue`].
#[derive(Clone)]
pub struct Blake3Subtree {
  hasher: Blake3,
  input_offset: u64,
  len: u64,
  max_len: u64,
}

impl Blake3Subtree {
  /// Absorb more subtree input.
  ///
  /// # Errors
  ///
  /// Returns [`Blake3SubtreeError::TooLong`], and absorbs nothing, if the
  /// subtree would extend past [`Self::max_len`].
  pub fn update(&mut self, input: &[u8]) -> Result<(), Blake3SubtreeError> {
    let len = u64::try_from(input.len())
      .ok()
      .and_then(|input_len| self.len.checked_add(input_len))
      .filter(|&len| len <= self.max_len)
      .ok_or(Blake3SubtreeError::TooLong)?;
    self.hasher.update(input);
    self.len = len;
    Ok(())
  }

  /// Compute this subtree's non-root chaining value.
  ///
  /// # Errors
  ///
  /// Returns [`Blake3SubtreeError::Empty`] if the subtree has no input.
  pub fn finalize(&self) -> Result<Blake3ChainingValue, Blake3SubtreeError> {
    if self.len == 0 {
      return Err(Blake3SubtreeError::Empty);
    }
    let mut words = self.hasher.root_output().chaining_value();
    let cv = Blake3ChainingValue {
      bytes: words8_to_le_bytes(&words),
      input_offset: self.input_offset,
      len: self.len,
      flags: self.hasher.chunk_state.flags,
    };
    ct::zeroize_words(&mut words);
    Ok(cv)
  }

  /// Offset of this subtree's first byte in the whole input.
  #[must_use]
  pub const fn input_offset(&self) -> u64 {
    self.input_offset
  }

  /// Bytes absorbed so far.
  #[must_use]
  pub const fn len(&self) -> u64 {
    self.len
  }

  /// Whether no input has been absorbed.
  #[must_use]
  pub const fn is_empty(&self) -> bool {
    self.len == 0
  }

  /// The most input this subtree accepts.
  ///
  /// A subtree at offset 0 is bounded only by `u64`. Any other subtree spans
  /// at most the largest power-of-two number of chunks that divides its
  /// starting chunk index.
  #[must_use]
  pub const fn max_len(&self) -> u64 {
    self.max_len
  }
}

impl core::fmt::Debug for Blake3Subtree {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Blake3Subtree")
      .field("mode", &mode_name(self.hasher.chunk_state.flags))
      .field("input_offset", &self.input_offset)
      .field("len", &self.len)
      .finish_non_exhaustive()
  }
}

/// The non-root hash of one subtree, with the input range and mode it covers.
///
/// It is never a hash of its input on its own; only [`Blake3Tree::merge_root`]
/// and [`Blake3Tree::merge_root_xof`] produce one. In keyed and derive-key
/// modes it is secret-derived: compare roots, not chaining values, and
/// dropping it clears the bytes.
#[derive(Clone)]
pub struct Blake3ChainingValue {
  bytes: [u8; OUT_LEN],
  input_offset: u64,
  len: u64,
  flags: u32,
}

impl Blake3ChainingValue {
  /// The 32-byte chaining value.
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; OUT_LEN] {
    &self.bytes
  }

  /// Offset of the subtree's first byte in the whole input.
  #[must_use]
  pub const fn input_offset(&self) -> u64 {
    self.input_offset
  }

  /// Length of the subtree's input in bytes.
  #[must_use]
  pub const fn len(&self) -> u64 {
    self.len
  }

  /// Always `false`: a chaining value covers at least one byte.
  #[must_use]
  pub const fn is_empty(&self) -> bool {
    false
  }

  fn words(&self) -> [u32; 8] {
    words8_from_le_bytes_32(&self.bytes)
  }
}

impl Drop for Blake3ChainingValue {
  fn drop(&mut self) {
    ct::zeroize(&mut self.bytes);
  }
}

impl core::fmt::Debug for Blake3ChainingValue {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Blake3ChainingValue")
      .field("mode", &mode_name(self.flags))
      .field("input_offset", &self.input_offset)
      .field("len", &self.len)
      .finish_non_exhaustive()
  }
}

// One group fills the widest parent kernel without allocating per-level storage.
const MERGE_LEVEL_PARENTS: usize = 16;

struct MergeLevelScratch {
  children: [[u32; 8]; MERGE_LEVEL_PARENTS * 2],
  parents: [[u32; 8]; MERGE_LEVEL_PARENTS],
}

impl Drop for MergeLevelScratch {
  fn drop(&mut self) {
    ct::zeroize_words_no_fence(self.children.as_flattened_mut());
    ct::zeroize_words_no_fence(self.parents.as_flattened_mut());
    ct::zeroize_fence();
  }
}

fn merge_kernel() -> kernels::Blake3KernelId {
  dispatch::hasher_dispatch().stream_kernel().id
}

const fn mode_name(flags: u32) -> &'static str {
  match flags {
    KEYED_HASH => "keyed",
    DERIVE_KEY_MATERIAL => "derive-key",
    _ => "hash",
  }
}
