#![cfg(feature = "blake3")]

use blake3::hazmat::{self, HasherExt as _};
use rscrypto::{
  Blake3DeriveContext,
  hashes::expert::blake3_tree::{Blake3ChainingValue, Blake3SubtreeError, Blake3Tree},
};

const CHUNK_LEN: usize = 1024;
const KEY: [u8; 32] = [0x53; 32];
const CONTEXT: &str = "rscrypto BLAKE3 merge level";

#[derive(Clone, Copy, Debug)]
enum Mode {
  Hash,
  Keyed,
  DeriveKey,
}

const MODES: [Mode; 3] = [Mode::Hash, Mode::Keyed, Mode::DeriveKey];

impl Mode {
  fn tree(self) -> Blake3Tree {
    match self {
      Self::Hash => Blake3Tree::new(),
      Self::Keyed => Blake3Tree::keyed(&KEY),
      Self::DeriveKey => Blake3Tree::derive_key(&Blake3DeriveContext::new(CONTEXT)),
    }
  }

  fn hasher(self) -> blake3::Hasher {
    match self {
      Self::Hash => blake3::Hasher::new(),
      Self::Keyed => blake3::Hasher::new_keyed(&KEY),
      Self::DeriveKey => blake3::Hasher::new_derive_key(CONTEXT),
    }
  }

  fn hazmat(self) -> hazmat::Mode<'static> {
    static CONTEXT_KEY: std::sync::OnceLock<[u8; 32]> = std::sync::OnceLock::new();
    match self {
      Self::Hash => hazmat::Mode::Hash,
      Self::Keyed => hazmat::Mode::KeyedHash(&KEY),
      Self::DeriveKey => {
        hazmat::Mode::DeriveKeyMaterial(CONTEXT_KEY.get_or_init(|| hazmat::hash_derive_key_context(CONTEXT)))
      }
    }
  }
}

fn next_random(state: &mut u64) -> u64 {
  *state ^= *state << 13;
  *state ^= *state >> 7;
  *state ^= *state << 17;
  *state
}

fn random_bytes(state: &mut u64, bytes: &mut [u8]) {
  for chunk in bytes.chunks_mut(8) {
    chunk.copy_from_slice(&next_random(state).to_le_bytes()[..chunk.len()]);
  }
}

fn hashed_cv(tree: &Blake3Tree, mode: Mode, offset: u64, input: &[u8]) -> Blake3ChainingValue {
  let mut subtree = tree.subtree(offset).expect("aligned subtree");
  subtree.update(input).expect("input fits subtree");
  let cv = subtree.finalize().expect("nonempty subtree");
  let mut oracle = mode.hasher();
  oracle.set_input_offset(offset).update(input);
  assert_eq!(cv.as_bytes(), &oracle.finalize_non_root(), "leaf at {offset}");
  cv
}

fn snapshot(values: &[Blake3ChainingValue]) -> Vec<([u8; 32], u64, u64)> {
  values
    .iter()
    .map(|cv| (*cv.as_bytes(), cv.input_offset(), cv.len()))
    .collect()
}

/// Check the batch against the serial production operation and the locked
/// upstream primitive. Carry an odd right edge without merging it prematurely.
fn checked_level(tree: &Blake3Tree, mode: Mode, children: &[Blake3ChainingValue]) -> Vec<Blake3ChainingValue> {
  let pairs = children.len() / 2;
  let mut parents = children[..pairs].to_vec();
  tree
    .merge_level(&children[..pairs.strict_mul(2)], &mut parents)
    .expect("valid level");
  for (pair, parent) in children.as_chunks::<2>().0.iter().zip(&parents) {
    let serial = tree.merge(&pair[0], &pair[1]).expect("valid pair");
    let oracle = hazmat::merge_subtrees_non_root(pair[0].as_bytes(), pair[1].as_bytes(), mode.hazmat());
    assert_eq!(parent.as_bytes(), &oracle, "{mode:?} at {}", parent.input_offset());
    assert_eq!(parent.as_bytes(), serial.as_bytes());
    assert_eq!(parent.input_offset(), pair[0].input_offset());
    assert_eq!(parent.len(), pair[0].len().strict_add(pair[1].len()));
  }
  if let Some(last) = children.as_chunks::<2>().1.first() {
    parents.push(last.clone());
  }
  parents
}

#[test]
fn random_hashed_trees_match_pairwise_hazmat_and_whole_input() {
  let mut random = 0x424c_414b_4533_4c56;
  // Cross 4-, 8-, 16-lane boundaries, the 16-parent scratch boundary and
  // multiple scratch groups, with partial final chunks and odd right edges.
  let counts = [
    2usize, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
  ];
  for mode in MODES {
    let tree = mode.tree();
    for leaves in counts {
      for final_len in [1usize, 64, 1023, 1024] {
        let len = leaves.strict_sub(1).strict_mul(CHUNK_LEN).strict_add(final_len);
        let mut input = vec![0; len];
        random_bytes(&mut random, &mut input);
        let mut level: Vec<_> = input
          .chunks(CHUNK_LEN)
          .enumerate()
          .map(|(index, chunk)| {
            let offset = u64::try_from(index.strict_mul(CHUNK_LEN)).expect("test offset fits u64");
            hashed_cv(&tree, mode, offset, chunk)
          })
          .collect();
        while level.len() > 2 {
          level = checked_level(&tree, mode, &level);
        }
        let root = tree.merge_root(&level[0], &level[1]).expect("root pair");
        assert_eq!(&root, mode.hasher().update(&input).finalize().as_bytes());
        // The last non-root merge has distinct flags from the root operation.
        let last = checked_level(&tree, mode, &level);
        assert_eq!(last[0].len(), u64::try_from(len).expect("test length fits u64"));
      }
    }
  }
}

#[test]
fn independently_ordered_pairs_at_mixed_heights_and_large_offsets() {
  let mut random = 0x6c61_6e65_5f70_6169;
  for mode in MODES {
    let tree = mode.tree();
    let mut children = Vec::new();
    for index in 0u64..65 {
      let shift = u32::try_from(next_random(&mut random) % 5).expect("bounded shift");
      let left_len = 1024usize << shift;
      let right_len = usize::try_from(next_random(&mut random) % u64::try_from(left_len).expect("bounded length"))
        .expect("bounded length")
        .strict_add(1);
      // Reverse offsets prove there is no relationship required between pairs.
      let offset = (1u64 << 53).strict_add(64u64.strict_sub(index).strict_mul(1 << 16));
      let mut input = vec![0; left_len.strict_add(right_len)];
      random_bytes(&mut random, &mut input);
      children.push(hashed_cv(&tree, mode, offset, &input[..left_len]));
      children.push(hashed_cv(
        &tree,
        mode,
        offset.strict_add(u64::try_from(left_len).expect("bounded length")),
        &input[left_len..],
      ));
    }
    let parents = checked_level(&tree, mode, &children);
    assert_eq!(parents.len(), 65);

    // Imported CV metadata can describe the largest representable input even
    // when this machine cannot hold that input in memory.
    let largest = [
      tree.chaining_value([0x11; 32], 0, 1 << 63).expect("left half"),
      tree
        .chaining_value([0x22; 32], 1 << 63, (1u64 << 63).strict_sub(1))
        .expect("right half"),
    ];
    assert_eq!(checked_level(&tree, mode, &largest)[0].len(), u64::MAX);
  }
}

#[test]
fn one_k_to_one_m_leaf_builds_match_pairwise_and_hazmat() {
  // Imported deterministic CV leaves exercise the parent API at scale without
  // rehashing GiB of existing leaf code. The smaller test hashes real leaves
  // and anchors complete roots to upstream whole-message hashing.
  for mode in MODES {
    let tree = mode.tree();
    let mut random = 0x7363_616c_655f_6376;
    for shift in [10u32, 12, 14, 16, 18, 20] {
      let leaf_count = 1usize << shift;
      let mut level: Vec<_> = (0..leaf_count)
        .map(|index| {
          let mut bytes = [0; 32];
          random_bytes(&mut random, &mut bytes);
          tree
            .chaining_value(
              bytes,
              u64::try_from(index.strict_mul(CHUNK_LEN)).expect("test offset fits u64"),
              1024,
            )
            .expect("complete leaf")
        })
        .collect();
      while level.len() > 2 {
        level = checked_level(&tree, mode, &level);
      }
      assert_eq!(
        tree.merge_root(&level[0], &level[1]).expect("root pair"),
        *hazmat::merge_subtrees_root(level[0].as_bytes(), level[1].as_bytes(), mode.hazmat()).as_bytes()
      );
      let last = checked_level(&tree, mode, &level);
      assert_eq!(last[0].input_offset(), 0);
      assert_eq!(
        last[0].len(),
        u64::try_from(leaf_count.strict_mul(CHUNK_LEN)).expect("test length")
      );
    }
  }
}

#[test]
fn shape_errors_precede_pair_validation_and_preserve_output() {
  let tree = Blake3Tree::new();
  let keyed = Blake3Tree::keyed(&KEY);
  let invalid = keyed.chaining_value([1; 32], 0, 1024).expect("valid leaf");
  let mut out = vec![tree.chaining_value([0x5a; 32], 4096, 1024).expect("sentinel")];
  let before = snapshot(&out);
  assert_eq!(
    tree.merge_level(core::slice::from_ref(&invalid), &mut out),
    Err(Blake3SubtreeError::OddChildCount)
  );
  assert_eq!(snapshot(&out), before);
  assert_eq!(
    tree.merge_level(&[], &mut out),
    Err(Blake3SubtreeError::OutputLengthMismatch)
  );
  assert_eq!(snapshot(&out), before);
  assert_eq!(
    tree.merge_level(&[invalid.clone(), invalid], &mut []),
    Err(Blake3SubtreeError::OutputLengthMismatch)
  );
  assert_eq!(tree.merge_level(&[], &mut []), Ok(()));
}

#[test]
fn later_invalid_pairs_leave_every_output_unchanged() {
  let tree = Blake3Tree::new();
  let valid: Vec<_> = (0u64..66)
    .map(|index| {
      tree
        .chaining_value([7; 32], index.strict_mul(1024), 1024)
        .expect("valid leaf")
    })
    .collect();
  let mut out = valid[..33].to_vec();
  let before = snapshot(&out);

  // Place each invalid form after one complete scratch group. A one-pass
  // validate-and-write implementation would already have mutated output.
  let invalid_pairs = [
    // Gap, reversed order, partial left child, and misaligned parent.
    ((32 * 1024, 1024), (34 * 1024, 1024)),
    ((33 * 1024, 1024), (32 * 1024, 1024)),
    ((32 * 1024, 17), (33 * 1024, 1024)),
    ((33 * 1024, 1024), (34 * 1024, 1024)),
    // A partial parent cannot be a left child at the next level.
    ((32 * 1024, 3 * 1024), (35 * 1024, 1024)),
    // Individually valid ranges can overflow when combined as an invalid pair.
    ((0, u64::MAX), (0, 1)),
  ];
  for ((left_offset, left_len), (right_offset, right_len)) in invalid_pairs {
    let mut children = valid.clone();
    children[32] = tree
      .chaining_value([1; 32], left_offset, left_len)
      .expect("valid left CV");
    children[33] = tree
      .chaining_value([2; 32], right_offset, right_len)
      .expect("valid right CV");
    assert_eq!(
      tree.merge_level(&children, &mut out),
      Err(Blake3SubtreeError::InvalidMerge)
    );
    assert_eq!(snapshot(&out), before);
  }
  let mut children = valid.clone();
  children[64] = Blake3Tree::keyed(&KEY)
    .chaining_value([3; 32], 64 * 1024, 1024)
    .expect("keyed leaf");
  assert_eq!(
    tree.merge_level(&children, &mut out),
    Err(Blake3SubtreeError::ModeMismatch)
  );
  assert_eq!(snapshot(&out), before);

  // Pair ordering takes precedence over a different error in a later pair.
  children[32] = tree.chaining_value([4; 32], 34 * 1024, 1024).expect("misplaced leaf");
  assert_eq!(
    tree.merge_level(&children, &mut out),
    Err(Blake3SubtreeError::InvalidMerge)
  );
  assert_eq!(snapshot(&out), before);

  // Retrying the same output after rejection succeeds with the valid inputs.
  tree.merge_level(&valid, &mut out).expect("retry with valid level");
  for (pair, parent) in valid.as_chunks::<2>().0.iter().zip(out) {
    assert_eq!(
      parent.as_bytes(),
      tree.merge(&pair[0], &pair[1]).expect("valid pair").as_bytes()
    );
  }
}
