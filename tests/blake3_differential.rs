#![cfg(feature = "blake3")]

use proptest::prelude::*;
use rscrypto::{
  hashes::crypto::{Blake3, Blake3DeriveContext},
  traits::{Digest as _, Xof as _},
};

fn blake3_ref_hash(data: &[u8]) -> [u8; 32] {
  *blake3::hash(data).as_bytes()
}

fn blake3_ref_xof(data: &[u8], out: &mut [u8]) {
  let mut h = blake3::Hasher::new();
  h.update(data);
  h.finalize_xof().fill(out);
}

fn blake3_ref_keyed(key: &[u8; 32], data: &[u8]) -> [u8; 32] {
  *blake3::keyed_hash(key, data).as_bytes()
}

fn blake3_ref_keyed_xof(key: &[u8; 32], data: &[u8], out: &mut [u8]) {
  let mut h = blake3::Hasher::new_keyed(key);
  h.update(data);
  h.finalize_xof().fill(out);
}

fn blake3_ref_derive(context: &str, data: &[u8]) -> [u8; 32] {
  blake3::derive_key(context, data)
}

fn blake3_ref_derive_xof(context: &str, data: &[u8], out: &mut [u8]) {
  let mut h = blake3::Hasher::new_derive_key(context);
  h.update(data);
  h.finalize_xof().fill(out);
}

fn patterned_bytes(len: usize) -> Vec<u8> {
  (0..len)
    .map(|i| u8::try_from(i % 251).expect("remainder modulo 251 must fit in one byte"))
    .collect()
}

#[test]
fn blake3_digest_const_matches_reference_and_streaming_for_every_two_chunk_length() {
  for len in 0..=2048 {
    let data = patterned_bytes(len);
    let digest = Blake3::digest_const(&data);
    assert_eq!(digest, blake3_ref_hash(&data), "reference mismatch at len={len}");
    let mut streaming = Blake3::new();
    streaming.update(&data);
    assert_eq!(digest, streaming.finalize(), "streaming mismatch at len={len}");
  }
}

// Constant evaluation of a full final block, alone and after the multi-block loop.
const FULL_BLOCK_DIGEST: [u8; 32] = Blake3::digest_const(&[0x5a; 64]);
const FULL_CHUNK_DIGEST: [u8; 32] = Blake3::digest_const(&[0x5a; 1024]);
const TWO_CHUNK_DIGEST: [u8; 32] = Blake3::digest_const(&[0x5a; 2048]);
const UNEVEN_TREE_DIGEST: [u8; 32] = Blake3::digest_const(&[0x5a; 3 * 1024 + 17]);

#[test]
fn blake3_digest_const_evaluates_full_final_blocks_at_compile_time() {
  assert_eq!(FULL_BLOCK_DIGEST, blake3_ref_hash(&[0x5a; 64]));
  assert_eq!(FULL_CHUNK_DIGEST, blake3_ref_hash(&[0x5a; 1024]));
  assert_eq!(TWO_CHUNK_DIGEST, blake3_ref_hash(&[0x5a; 2048]));
  assert_eq!(UNEVEN_TREE_DIGEST, blake3_ref_hash(&[0x5a; 3 * 1024 + 17]));
}

#[test]
fn blake3_digest_const_matches_reference_at_tree_boundaries() {
  for chunks in [2usize, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1024] {
    let boundary = chunks.strict_mul(1024);
    for len in [boundary.strict_sub(1), boundary, boundary.strict_add(1)] {
      let data = patterned_bytes(len);
      assert_eq!(Blake3::digest_const(&data), blake3_ref_hash(&data), "len={len}");
    }
  }
}

#[test]
fn blake3_derive_context_const_matches_runtime_for_every_one_chunk_length() {
  for len in 0..=1024 {
    let context: String = patterned_bytes(len)
      .iter()
      .map(|&b| char::from(b'a'.strict_add(b % 26)))
      .collect();
    let prehashed = Blake3DeriveContext::new_const(&context);
    assert_eq!(
      prehashed,
      Blake3DeriveContext::new(&context),
      "context mismatch at len={len}"
    );
    assert_eq!(
      Blake3::derive_key_with(&prehashed, b"key material"),
      blake3_ref_derive(&context, b"key material"),
      "reference mismatch at len={len}"
    );
  }
}

#[test]
#[should_panic(expected = "Blake3DeriveContext::new_const accepts at most 1,024 bytes")]
fn blake3_derive_context_const_rejects_more_than_one_chunk() {
  let context = "a".repeat(1025);
  core::hint::black_box(Blake3DeriveContext::new_const(core::hint::black_box(&context)));
}

proptest! {
  #[test]
  fn blake3_digest_const_matches_official(data in proptest::collection::vec(any::<u8>(), 0..=32769)) {
    prop_assert_eq!(Blake3::digest_const(&data), blake3_ref_hash(&data));
  }

  #[test]
  fn blake3_one_shot_matches_official(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    prop_assert_eq!(Blake3::digest(&data), blake3_ref_hash(&data));
  }

  #[test]
  fn blake3_streaming_matches_official(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    let expected = blake3_ref_hash(&data);

    let mut h = Blake3::new();
    let mut i = 0usize;
    while i < data.len() {
      let step = (usize::from(data[i]) % 251).strict_add(1);
      let end = core::cmp::min(data.len(), i.strict_add(step));
      h.update(&data[i..end]);
      i = end;
    }

    prop_assert_eq!(h.finalize(), expected);
  }

  #[test]
  fn blake3_xof_matches_official(data in proptest::collection::vec(any::<u8>(), 0..4096), out_len in 0usize..2048) {
    let mut expected = vec![0u8; out_len];
    let mut ref_hasher = blake3::Hasher::new();
    ref_hasher.update(&data);
    ref_hasher.finalize_xof().fill(&mut expected);

    let mut h = Blake3::new();
    h.update(&data);
    let mut xof = h.finalize_xof();
    let mut actual = vec![0u8; out_len];
    xof.squeeze(&mut actual);

    prop_assert_eq!(actual, expected);
  }

  #[test]
  fn blake3_keyed_matches_official(
    data in proptest::collection::vec(any::<u8>(), 0..4096),
    key in any::<[u8; 32]>(),
  ) {
    let expected = blake3_ref_keyed(&key, &data);
    let mut h = Blake3::new_keyed(&key);
    h.update(&data);
    prop_assert_eq!(h.finalize(), expected);
  }

  #[test]
  fn blake3_derive_key_matches_official(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
    const CONTEXT: &str = "rscrypto blake3 derive-key test context";

    let expected = blake3_ref_derive(CONTEXT, &data);
    let mut h = Blake3::new_derive_key(CONTEXT);
    h.update(&data);
    prop_assert_eq!(h.finalize(), expected);
  }

  #[test]
  fn blake3_prehashed_derive_key_matches_official(
    context in "[ -~]{0,2100}",
    data in proptest::collection::vec(any::<u8>(), 0..4096),
  ) {
    let prehashed = Blake3DeriveContext::new(&context);
    let expected = blake3_ref_derive(&context, &data);
    prop_assert_eq!(Blake3::derive_key_with(&prehashed, &data), expected);

    let mut h = Blake3::new_derive_key_from(&prehashed);
    h.update(&data);
    prop_assert_eq!(h.finalize(), expected);
  }
}

#[test]
fn blake3_multi_chunk_then_small_tail_matches_official_in_all_modes() {
  const CHUNK_LEN: usize = 1024;
  const TAIL_LEN: usize = 9;
  const XOF_LEN: usize = 96;
  const CONTEXT: &str = "rscrypto blake3 derive-key regression context";

  let data = patterned_bytes(2 * CHUNK_LEN + TAIL_LEN);
  let (first, second) = data.split_at(2 * CHUNK_LEN);
  let key = *b"whats the Elvish word for friend";

  {
    let mut h = Blake3::new();
    h.update(first);
    h.update(second);
    assert_eq!(h.finalize(), blake3_ref_hash(&data));

    let mut expected = [0u8; XOF_LEN];
    blake3_ref_xof(&data, &mut expected);
    let mut actual = [0u8; XOF_LEN];
    h.finalize_xof().squeeze(&mut actual);
    assert_eq!(actual, expected);
  }

  {
    let mut h = Blake3::new_keyed(&key);
    h.update(first);
    h.update(second);
    assert_eq!(h.finalize(), blake3_ref_keyed(&key, &data));

    let mut expected = [0u8; XOF_LEN];
    blake3_ref_keyed_xof(&key, &data, &mut expected);
    let mut actual = [0u8; XOF_LEN];
    h.finalize_xof().squeeze(&mut actual);
    assert_eq!(actual, expected);
  }

  {
    let mut h = Blake3::new_derive_key(CONTEXT);
    h.update(first);
    h.update(second);
    assert_eq!(h.finalize(), blake3_ref_derive(CONTEXT, &data));

    let mut expected = [0u8; XOF_LEN];
    blake3_ref_derive_xof(CONTEXT, &data, &mut expected);
    let mut actual = [0u8; XOF_LEN];
    h.finalize_xof().squeeze(&mut actual);
    assert_eq!(actual, expected);
  }
}

mod subtree {
  use blake3::hazmat::{self, HasherExt as _};
  use proptest::prelude::*;
  use rscrypto::{
    hashes::{
      crypto::{Blake3, Blake3DeriveContext},
      expert::blake3_tree::{Blake3ChainingValue, Blake3SubtreeError, Blake3Tree},
    },
    traits::Xof as _,
  };

  use super::patterned_bytes;

  const CHUNK_LEN: u64 = 1024;
  const KEY: [u8; 32] = *b"whats the Elvish word for friend";
  const CONTEXT: &str = "rscrypto blake3 subtree test context";

  #[derive(Clone, Copy, Debug)]
  enum Mode {
    Hash,
    Keyed,
    DeriveKey,
  }

  const MODES: [Mode; 3] = [Mode::Hash, Mode::Keyed, Mode::DeriveKey];

  fn tree(mode: Mode) -> Blake3Tree {
    match mode {
      Mode::Hash => Blake3Tree::new(),
      Mode::Keyed => Blake3Tree::keyed(&KEY),
      Mode::DeriveKey => Blake3Tree::derive_key(&Blake3DeriveContext::new(CONTEXT)),
    }
  }

  fn reference_hasher(mode: Mode) -> blake3::Hasher {
    match mode {
      Mode::Hash => blake3::Hasher::new(),
      Mode::Keyed => blake3::Hasher::new_keyed(&KEY),
      Mode::DeriveKey => blake3::Hasher::new_derive_key(CONTEXT),
    }
  }

  fn reference_root(mode: Mode, input: &[u8]) -> blake3::Hasher {
    let mut hasher = reference_hasher(mode);
    hasher.update(input);
    hasher
  }

  fn reference_cv(mode: Mode, offset: u64, input: &[u8]) -> [u8; 32] {
    let mut hasher = reference_hasher(mode);
    hasher.set_input_offset(offset);
    hasher.update(input);
    hasher.finalize_non_root()
  }

  fn subtree_cv(tree: &Blake3Tree, offset: u64, input: &[u8]) -> Blake3ChainingValue {
    // One update reaches the multi-chunk paths; uneven pieces cross buffer and
    // chunk boundaries inside the hasher.
    let mut whole = tree.subtree(offset).expect("test offsets are chunk-aligned");
    whole.update(input).expect("test subtrees fit their offset");
    let mut pieces = tree.subtree(offset).expect("test offsets are chunk-aligned");
    for piece in input.chunks(1537) {
      pieces.update(piece).expect("test subtrees fit their offset");
    }
    let cv = whole.finalize().expect("test subtrees are non-empty");
    assert_eq!(
      cv.as_bytes(),
      pieces.finalize().expect("test subtrees are non-empty").as_bytes(),
      "update split mismatch at offset={offset} len={}",
      input.len()
    );
    cv
  }

  /// Hash `input` at `offset` as a subtree split recursively down to `leaf_len`
  /// bytes, checking every chaining value and merge against `blake3::hazmat`.
  fn split_cv(mode: Mode, tree: &Blake3Tree, offset: u64, input: &[u8], leaf_len: u64) -> Blake3ChainingValue {
    let len = input.len() as u64;
    let cv = match Blake3Tree::left_subtree_len(len) {
      Some(left_len) if len > leaf_len => {
        let (left, right) = input.split_at(usize::try_from(left_len).expect("test inputs fit in memory"));
        let left_cv = split_cv(mode, tree, offset, left, leaf_len);
        let right_cv = split_cv(mode, tree, offset.strict_add(left_len), right, leaf_len);
        let merged = tree
          .merge(&left_cv, &right_cv)
          .expect("recursive splits are valid merges");
        assert_eq!(
          merged.as_bytes(),
          &hazmat::merge_subtrees_non_root(left_cv.as_bytes(), right_cv.as_bytes(), hazmat_mode(mode)),
          "merge mismatch for {mode:?} at offset={offset} len={len}"
        );
        merged
      }
      _ => subtree_cv(tree, offset, input),
    };
    assert_eq!(
      cv.as_bytes(),
      &reference_cv(mode, offset, input),
      "chaining value mismatch for {mode:?} at offset={offset} len={len}"
    );
    assert_eq!((cv.input_offset(), cv.len()), (offset, len));
    cv
  }

  fn hazmat_mode(mode: Mode) -> hazmat::Mode<'static> {
    static CONTEXT_KEY: std::sync::OnceLock<[u8; 32]> = std::sync::OnceLock::new();
    match mode {
      Mode::Hash => hazmat::Mode::Hash,
      Mode::Keyed => hazmat::Mode::KeyedHash(&KEY),
      Mode::DeriveKey => {
        hazmat::Mode::DeriveKeyMaterial(CONTEXT_KEY.get_or_init(|| hazmat::hash_derive_key_context(CONTEXT)))
      }
    }
  }

  fn check_split_root(mode: Mode, input: &[u8], leaf_len: u64) {
    let tree = tree(mode);
    let len = input.len() as u64;
    let Some(left_len) = Blake3Tree::left_subtree_len(len) else {
      return;
    };
    let (left, right) = input.split_at(usize::try_from(left_len).expect("test inputs fit in memory"));
    let left_cv = split_cv(mode, &tree, 0, left, leaf_len);
    let right_cv = split_cv(mode, &tree, left_len, right, leaf_len);

    let reference = reference_root(mode, input);
    let root = tree
      .merge_root(&left_cv, &right_cv)
      .expect("the top split is a valid root");
    assert_eq!(
      &root,
      reference.finalize().as_bytes(),
      "root mismatch for {mode:?} len={len}"
    );
    let expected_root = match mode {
      Mode::Hash => Blake3::digest(input),
      Mode::Keyed => Blake3::keyed_digest(&KEY, input).to_bytes(),
      Mode::DeriveKey => Blake3::derive_key_with(&Blake3DeriveContext::new(CONTEXT), input),
    };
    assert_eq!(root, expected_root, "one-shot mismatch for {mode:?} len={len}");

    let mut expected_xof = [0u8; 131];
    reference.finalize_xof().fill(&mut expected_xof);
    let mut xof = [0u8; 131];
    tree
      .merge_root_xof(&left_cv, &right_cv)
      .expect("the top split is a valid root")
      .squeeze(&mut xof);
    assert_eq!(xof, expected_xof, "root XOF mismatch for {mode:?} len={len}");
  }

  #[test]
  fn recursive_splits_match_reference_at_tree_boundaries() {
    for chunks in [2u64, 3, 4, 5, 7, 8, 9, 16, 17, 31, 33] {
      for delta in [-1i64, 0, 1] {
        let len = (chunks * CHUNK_LEN).strict_add_signed(delta);
        let input = patterned_bytes(usize::try_from(len).expect("test inputs fit in memory"));
        for mode in MODES {
          check_split_root(mode, &input, CHUNK_LEN);
        }
      }
    }
  }

  #[test]
  fn wide_subtrees_at_nonzero_offsets_match_reference() {
    // 128- and 256-chunk leaves reach the multi-chunk update paths with a
    // nonzero starting counter.
    let input = patterned_bytes(1024 * 1024 + 77);
    for mode in MODES {
      check_split_root(mode, &input, 256 * CHUNK_LEN);
      check_split_root(mode, &input, 128 * CHUNK_LEN);
    }
  }

  #[test]
  fn subtrees_at_large_offsets_match_reference() {
    let input = patterned_bytes(3 * 1024 + 5);
    for offset_chunks in [1u64 << 20, 1 << 40, 1 << 53, (1 << 54) - 4] {
      let offset = offset_chunks * CHUNK_LEN;
      for mode in MODES {
        let cv = subtree_cv(&tree(mode), offset, &input);
        assert_eq!(
          cv.as_bytes(),
          &reference_cv(mode, offset, &input),
          "{mode:?} at chunk {offset_chunks}"
        );
      }
    }
  }

  #[test]
  fn subtree_rejects_unaligned_offsets_and_empty_input() {
    let tree = Blake3Tree::new();
    assert_eq!(tree.subtree(1).err(), Some(Blake3SubtreeError::UnalignedOffset));
    assert_eq!(tree.subtree(1023).err(), Some(Blake3SubtreeError::UnalignedOffset));
    let empty = tree.subtree(CHUNK_LEN).expect("chunk-aligned offset");
    assert_eq!(empty.finalize().err(), Some(Blake3SubtreeError::Empty));
    assert_eq!(
      tree.chaining_value([0; 32], CHUNK_LEN, 0).err(),
      Some(Blake3SubtreeError::Empty)
    );
  }

  #[test]
  fn subtree_rejects_input_past_its_maximum_without_absorbing_it() {
    let tree = Blake3Tree::new();
    // Chunk 6 starts a subtree of at most two chunks.
    let offset = 6 * CHUNK_LEN;
    let mut subtree = tree.subtree(offset).expect("chunk-aligned offset");
    assert_eq!(subtree.max_len(), 2 * CHUNK_LEN);
    let input = patterned_bytes(2 * 1024);
    subtree.update(&input[..1500]).expect("within the maximum");
    assert_eq!(subtree.update(&[0; 549]).err(), Some(Blake3SubtreeError::TooLong));
    assert_eq!(subtree.len(), 1500);
    subtree.update(&input[1500..]).expect("exactly the maximum");
    assert_eq!(subtree.update(&[0]).err(), Some(Blake3SubtreeError::TooLong));
    let cv = subtree.finalize().expect("non-empty subtree");
    assert_eq!(cv.as_bytes(), &reference_cv(Mode::Hash, offset, &input));

    assert_eq!(
      tree.chaining_value([0; 32], offset, 2 * CHUNK_LEN + 1).err(),
      Some(Blake3SubtreeError::TooLong)
    );
    assert_eq!(Blake3Tree::new().subtree(0).expect("offset 0").max_len(), u64::MAX);
  }

  #[test]
  fn merges_reject_every_invalid_parent() {
    let tree = Blake3Tree::new();
    let cv = |offset_chunks: u64, len: u64| {
      tree
        .chaining_value([0; 32], offset_chunks * CHUNK_LEN, len)
        .expect("valid chaining value position")
    };
    let full = CHUNK_LEN;

    // Valid: chunks 0 and 1, chunks 2 and 3, and a short right edge.
    tree.merge(&cv(0, full), &cv(1, full)).expect("sibling chunks");
    tree.merge(&cv(2, full), &cv(3, 10)).expect("short right edge");
    tree
      .merge_root(&cv(0, 2 * full), &cv(2, full))
      .expect("root over three chunks");

    let invalid = Err(Blake3SubtreeError::InvalidMerge);
    // Not adjacent.
    assert_eq!(tree.merge(&cv(0, full), &cv(2, full)).map(drop), invalid);
    // Swapped children.
    assert_eq!(tree.merge(&cv(1, full), &cv(0, full)).map(drop), invalid);
    // A short left child leaves a gap before the next chunk.
    assert_eq!(tree.merge(&cv(0, 10), &cv(1, full)).map(drop), invalid);
    // A right child larger than its left sibling cannot be constructed.
    assert_eq!(
      tree.chaining_value([0; 32], CHUNK_LEN, 2 * full).err(),
      Some(Blake3SubtreeError::TooLong)
    );
    // Parent not aligned: chunks 1 and 2 are not siblings.
    assert_eq!(tree.merge(&cv(1, full), &cv(2, full)).map(drop), invalid);
    // A root must start at offset 0.
    assert_eq!(tree.merge_root(&cv(2, full), &cv(3, full)).map(drop), invalid);
    assert_eq!(tree.merge_root_xof(&cv(2, full), &cv(3, full)).map(drop), invalid);
    // A partial parent cannot be a left child.
    let partial = tree.merge(&cv(0, 2 * full), &cv(2, full)).expect("right-edge parent");
    assert_eq!(tree.merge(&partial, &cv(3, full)).map(drop), invalid);

    let keyed = Blake3Tree::keyed(&KEY);
    let keyed_cv = keyed.chaining_value([0; 32], CHUNK_LEN, full).expect("valid position");
    assert_eq!(
      tree.merge(&cv(0, full), &keyed_cv).map(drop),
      Err(Blake3SubtreeError::ModeMismatch)
    );
  }

  proptest! {
    #[test]
    fn random_splits_match_reference(len in 1025usize..40_000, leaf_shift in 0u32..5, mode in 0usize..3) {
      let input = patterned_bytes(len);
      check_split_root(MODES[mode], &input, CHUNK_LEN << leaf_shift);
    }
  }
}

#[test]
fn blake3_digest_batch_matches_reference_for_every_one_chunk_length() {
  let data = patterned_bytes(33 * 1025);
  for len in 0..=1025 {
    // 33 inputs fill every lane width and leave a partial group.
    let inputs: Vec<&[u8]> = (0..33).map(|i| &data[i * len..i * len + len]).collect();
    for count in [1, 2, 3, 4, 5, 8, 9, 16, 17, 33] {
      let mut outputs = vec![[0u8; 32]; count];
      Blake3::digest_batch(&inputs[..count], &mut outputs);
      for (i, output) in outputs.iter().enumerate() {
        assert_eq!(*output, blake3_ref_hash(inputs[i]), "len={len} count={count} index={i}");
      }
    }
  }
}

#[test]
#[should_panic(expected = "Blake3::digest_batch needs one output per input")]
fn blake3_digest_batch_rejects_mismatched_outputs() {
  let mut outputs = [[0u8; 32]; 1];
  Blake3::digest_batch(&[b"a", b"b"], &mut outputs);
}

proptest! {
  #[test]
  fn blake3_digest_batch_matches_reference_for_mixed_lengths(
    lens in proptest::collection::vec(prop_oneof![Just(64usize), Just(21), Just(1024), 0usize..1100], 0..40),
  ) {
    let data = patterned_bytes(lens.iter().sum());
    let mut inputs = Vec::with_capacity(lens.len());
    let mut rest = data.as_slice();
    for &len in &lens {
      let (input, next) = rest.split_at(len);
      inputs.push(input);
      rest = next;
    }
    let mut outputs = vec![[0u8; 32]; inputs.len()];
    Blake3::digest_batch(&inputs, &mut outputs);
    for (input, output) in inputs.iter().zip(&outputs) {
      prop_assert_eq!(*output, blake3_ref_hash(input));
    }
  }
}

mod seekable_xof {
  use proptest::prelude::*;
  use rscrypto::{
    hashes::crypto::{Blake3, Blake3XofReader},
    traits::{Digest as _, Xof as _},
  };

  use super::patterned_bytes;

  const KEY: [u8; 32] = *b"rscrypto seekable xof test key!!";
  const CONTEXT: &str = "rscrypto blake3 seekable XOF test context";
  const STREAM_LEN: usize = 640;
  /// First byte of output block 2^32, where the block counter's high word changes.
  const COUNTER_HIGH_WORD: u64 = (1 << 32) * 64;

  #[derive(Clone, Copy, Debug)]
  enum Mode {
    Hash,
    Keyed,
    DeriveKey,
  }

  const MODES: [Mode; 3] = [Mode::Hash, Mode::Keyed, Mode::DeriveKey];

  /// Our reader, built one-shot or by streaming, and the upstream reader for one input.
  fn readers(mode: Mode, data: &[u8], one_shot: bool) -> (Blake3XofReader, blake3::OutputReader) {
    let (ours, mut reference) = match mode {
      Mode::Hash if one_shot => (Blake3::xof(data), blake3::Hasher::new()),
      Mode::Keyed if one_shot => (Blake3::keyed_xof(&KEY, data), blake3::Hasher::new_keyed(&KEY)),
      Mode::Hash => (streamed(Blake3::new(), data), blake3::Hasher::new()),
      Mode::Keyed => (streamed(Blake3::new_keyed(&KEY), data), blake3::Hasher::new_keyed(&KEY)),
      Mode::DeriveKey => (
        streamed(Blake3::new_derive_key(CONTEXT), data),
        blake3::Hasher::new_derive_key(CONTEXT),
      ),
    };
    reference.update(data);
    (ours, reference.finalize_xof())
  }

  fn streamed(mut hasher: Blake3, data: &[u8]) -> Blake3XofReader {
    hasher.update(data);
    hasher.finalize_xof()
  }

  fn upstream_at(mut reference: blake3::OutputReader, position: u64, len: usize) -> Vec<u8> {
    reference.set_position(position);
    let mut out = vec![0u8; len];
    reference.fill(&mut out);
    out
  }

  #[test]
  fn seeks_match_one_long_squeeze_and_upstream() {
    let data = patterned_bytes(5000);
    for mode in MODES {
      // One-chunk, multi-block, and parent-root inputs reach different reader constructors.
      for len in [0, 1, 64, 65, 1024, 1025, 2049, 5000] {
        for one_shot in [true, false] {
          let input = &data[..len];
          let (mut ours, reference) = readers(mode, input, one_shot);
          let mut stream = [0u8; STREAM_LEN];
          ours.squeeze(&mut stream);
          assert_eq!(
            stream[..],
            upstream_at(reference, 0, STREAM_LEN)[..],
            "{mode:?} len={len} one_shot={one_shot}"
          );

          for position in [0, 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 300, 575] {
            for read_len in [0, 1, 31, 32, 33, 64, 65, 200] {
              let end = position + read_len;
              if end > STREAM_LEN {
                continue;
              }
              let (mut reader, _) = readers(mode, input, one_shot);
              reader.set_position(position as u64);
              assert_eq!(reader.position(), position as u64);
              let mut out = vec![0u8; read_len];
              reader.squeeze(&mut out);
              assert_eq!(
                out[..],
                stream[position..end],
                "{mode:?} len={len} one_shot={one_shot} position={position} read_len={read_len}"
              );
              assert_eq!(reader.position(), end as u64);
            }
          }

          // Seeking a used reader backward replays earlier output.
          let mut out = [0u8; 100];
          ours.set_position(17);
          ours.squeeze(&mut out);
          assert_eq!(out[..], stream[17..117], "{mode:?} len={len} one_shot={one_shot}");
        }
      }
    }
  }

  #[test]
  fn large_positions_match_upstream() {
    let data = patterned_bytes(3000);
    for mode in MODES {
      for len in [0, 33, 1024, 3000] {
        for position in [
          COUNTER_HIGH_WORD - 100,
          COUNTER_HIGH_WORD,
          COUNTER_HIGH_WORD + 7,
          1 << 40,
          u64::MAX - 400,
          u64::MAX - 63,
        ] {
          let (mut ours, reference) = readers(mode, &data[..len], true);
          ours.set_position(position);
          assert_eq!(ours.position(), position);
          let mut out = [0u8; 400];
          ours.squeeze(&mut out);
          assert_eq!(
            out[..],
            upstream_at(reference, position, out.len())[..],
            "{mode:?} len={len} position={position}"
          );
        }
      }
    }
  }

  #[test]
  fn position_wraps_after_the_last_u64_offset() {
    let (mut ours, reference) = readers(Mode::Keyed, b"wrap", true);
    ours.set_position(u64::MAX);
    let mut out = [0u8; 130];
    ours.squeeze(&mut out);
    // Byte u64::MAX is the last byte of block 2^58 - 1; the read then continues into block 2^58.
    assert_eq!(out[..], upstream_at(reference, u64::MAX, out.len())[..]);
    assert_eq!(ours.position(), 129);
  }

  proptest! {
    #[test]
    fn seeks_after_partial_reads_match_upstream(
      mode in 0usize..3,
      data in proptest::collection::vec(any::<u8>(), 0..2100),
      one_shot in any::<bool>(),
      first_read in 0usize..200,
      position in prop_oneof![0u64..4096, any::<u64>()],
      read_len in 0usize..600,
    ) {
      let (mut ours, reference) = readers(MODES[mode], &data, one_shot);
      let mut first = vec![0u8; first_read];
      ours.squeeze(&mut first);
      ours.set_position(position);
      let mut out = vec![0u8; read_len];
      ours.squeeze(&mut out);
      prop_assert_eq!(out, upstream_at(reference, position, read_len));
    }
  }
}
