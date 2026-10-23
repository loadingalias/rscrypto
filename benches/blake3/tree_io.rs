//! Reader input and caller-scheduled tree merges against upstream `blake3`.
//!
//! Case IDs: `blake3/<build>/update-reader/<implementation>/<input>`,
//! `blake3/serial/subtree-update-reader/<implementation>/<input>`, and
//! `blake3/merge-level/<implementation>/<parents>`. Every row's output is
//! checked against the other library's output before timing.

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rscrypto::{
  Blake3, Digest as _,
  hashes::expert::blake3_tree::{Blake3ChainingValue, Blake3Tree},
};

use super::bench_config::{checked, checked_with};

/// A short input inside one reader buffer, two inputs that fill rscrypto's
/// 1 MiB coalescing buffer partly and exactly, and an input of eight buffers.
const READER_LENGTHS: [usize; 4] = [4096, 65536, 1048576, 8388608];

/// Hash one in-memory input through each library's reader API.
///
/// rscrypto `Blake3::update_reader` coalesces reads into an aligned 1 MiB heap
/// buffer, cleared before deallocation, and hashes each buffer with `update`.
/// Upstream `Hasher::update_reader` copies through its own buffer into
/// `update`. The reader is a `&[u8]` over the whole input.
///
/// `blake3/serial/update-reader` (build `blake3_serial`): neither row threads.
/// `blake3/parallel/update-reader` (build `blake3`): threads are allowed. With
/// `parallel`, rscrypto hashes complete buffers in the current Rayon pool on
/// little-endian Linux AArch64, and its `update` admits threads on the CPU
/// profiles that enable streaming admission. Upstream has no threaded reader
/// API, so its row is serial in both builds.
///
/// Timed: hasher construction, every read and update, the buffer, and
/// finalization with its 32-byte output. Untimed: the input and expected
/// outputs. Each row must also report the whole input length as read.
pub(super) fn update_reader(c: &mut Criterion) {
  let name = format!("blake3/{}/update-reader", super::BUILD);
  if !super::bench_config::selected(&name) {
    return;
  }

  let mut g = c.benchmark_group(name);
  for len in READER_LENGTHS {
    let data = super::common::random_bytes(len);
    let expected = (
      u64::try_from(len).expect("benchmark length fits u64"),
      *blake3::hash(&data).as_bytes(),
    );
    super::common::set_throughput(&mut g, len);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || {
          let mut hasher = Blake3::new();
          let mut reader = black_box(data.as_slice());
          let read = hasher
            .update_reader(&mut reader)
            .expect("an in-memory reader does not fail");
          (read, hasher.finalize())
        },
        |out| assert_eq!(out, expected, "update_reader at len={len}"),
      ),
    );
    g.bench_function(
      BenchmarkId::new("blake3", len),
      checked(
        || {
          let mut hasher = blake3::Hasher::new();
          hasher
            .update_reader(black_box(data.as_slice()))
            .expect("an in-memory reader does not fail");
          (hasher.count(), *hasher.finalize().as_bytes())
        },
        |out| assert_eq!(out, expected, "upstream update_reader at len={len}"),
      ),
    );
  }
  g.finish();
}

/// Hash one subtree of a larger input from an in-memory reader into its
/// non-root chaining value.
///
/// Each case is a subtree of the named length that starts at the same byte
/// offset as its length, the largest subtree that offset admits. rscrypto
/// calls `Blake3Tree::subtree`, `Blake3Subtree::update_reader` and `finalize`;
/// upstream calls `Hasher::new`, `HasherExt::set_input_offset`,
/// `update_reader` and `finalize_non_root`. Only `blake3_serial` registers this
/// group: with `parallel`, rscrypto's `update` admits threads on some CPU
/// profiles.
///
/// Timed: subtree construction, every read and update, the reader buffer, and
/// the 32-byte chaining value. Untimed: the input and expected values.
pub(super) fn subtree_update_reader(c: &mut Criterion) {
  use blake3::hazmat::HasherExt as _;

  const NAME: &str = "blake3/serial/subtree-update-reader";
  if !super::bench_config::selected(NAME) {
    return;
  }

  let tree = Blake3Tree::new();
  let mut g = c.benchmark_group(NAME);
  for len in [65536usize, 1048576] {
    let data = super::common::random_bytes(len);
    let length = u64::try_from(len).expect("benchmark length fits u64");
    let offset = length;
    let expected = {
      let mut hasher = blake3::Hasher::new();
      hasher.set_input_offset(offset).update(&data);
      (length, hasher.finalize_non_root())
    };
    super::common::set_throughput(&mut g, len);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || {
          let mut subtree = tree.subtree(black_box(offset)).expect("chunk-aligned subtree offset");
          let mut reader = black_box(data.as_slice());
          let read = subtree
            .update_reader(&mut reader)
            .expect("an in-memory reader does not fail");
          (read, *subtree.finalize().expect("a non-empty subtree").as_bytes())
        },
        |out| assert_eq!(out, expected, "subtree update_reader at len={len}"),
      ),
    );
    g.bench_function(
      BenchmarkId::new("blake3", len),
      checked(
        || {
          let mut hasher = blake3::Hasher::new();
          hasher.set_input_offset(black_box(offset));
          hasher
            .update_reader(black_box(data.as_slice()))
            .expect("an in-memory reader does not fail");
          (hasher.count(), hasher.finalize_non_root())
        },
        |out| assert_eq!(out, expected, "upstream subtree update_reader at len={len}"),
      ),
    );
  }
  g.finish();
}

/// Merge adjacent pairs of chunk chaining values into their non-root parents.
///
/// The children are the chaining values of `2 * parents` consecutive chunks,
/// so each pair forms one two-chunk parent. rscrypto calls
/// `Blake3Tree::merge_level` once into reused output slots; it validates every
/// pair's input range and mode, then hashes up to 16 parents per kernel call.
/// Upstream has no level API and no validation: its row calls
/// `hazmat::merge_subtrees_non_root` once per pair into a reused array.
/// Inputs: one pair, one full rscrypto kernel group, and 16 groups. Neither row
/// threads, so only the `blake3` build registers this group.
///
/// Timed: validation, every parent hash, and the writes into the output slots.
/// Untimed: hashing the chunks and the expected parents. Outside timing, the
/// rscrypto row must also reject a pair with its children swapped.
pub(super) fn merge_level(c: &mut Criterion) {
  use blake3::hazmat::{HasherExt as _, Mode, merge_subtrees_non_root};

  const NAME: &str = "blake3/merge-level";
  if !super::bench_config::selected(NAME) {
    return;
  }

  let tree = Blake3Tree::new();
  let mut g = c.benchmark_group(NAME);
  for parents in [1usize, 16, 256] {
    let data = super::common::random_bytes(parents.strict_mul(2048));
    let children: Vec<Blake3ChainingValue> = data
      .chunks(1024)
      .enumerate()
      .map(|(index, chunk)| {
        let offset = u64::try_from(index.strict_mul(1024)).expect("chunk offset fits u64");
        let mut subtree = tree.subtree(offset).expect("chunk-aligned subtree offset");
        subtree.update(chunk).expect("a chunk fits its subtree");
        subtree.finalize().expect("a non-empty subtree")
      })
      .collect();
    let upstream_children: Vec<[u8; 32]> = data
      .chunks(1024)
      .enumerate()
      .map(|(index, chunk)| {
        let offset = u64::try_from(index.strict_mul(1024)).expect("chunk offset fits u64");
        let mut hasher = blake3::Hasher::new();
        hasher.set_input_offset(offset).update(chunk);
        hasher.finalize_non_root()
      })
      .collect();
    assert!(
      children
        .iter()
        .zip(&upstream_children)
        .all(|(ours, upstream)| ours.as_bytes() == upstream),
      "chunk chaining values must agree"
    );
    let expected: Vec<[u8; 32]> = upstream_children
      .as_chunks::<2>()
      .0
      .iter()
      .map(|[left, right]| merge_subtrees_non_root(left, right, Mode::Hash))
      .collect();
    let mut swapped = children[..2].to_vec();
    swapped.swap(0, 1);
    super::common::set_throughput(&mut g, parents.strict_mul(64));

    g.bench_function(
      BenchmarkId::new("rscrypto", parents),
      checked_with(
        children[..parents].to_vec(),
        |out| {
          tree
            .merge_level(black_box(&children), black_box(out.as_mut_slice()))
            .expect("adjacent chunk pairs form parents");
        },
        |(), out| {
          assert!(
            out.iter().map(Blake3ChainingValue::as_bytes).eq(expected.iter()),
            "merge_level parents at {parents}"
          );
          assert!(
            tree.merge_level(&swapped, &mut out.clone()[..1]).is_err(),
            "swapped children must not merge"
          );
        },
      ),
    );
    g.bench_function(
      BenchmarkId::new("blake3", parents),
      checked_with(
        vec![[0u8; 32]; parents],
        |out| {
          for ([left, right], parent) in black_box(&upstream_children)
            .as_chunks::<2>()
            .0
            .iter()
            .zip(out.iter_mut())
          {
            *parent = merge_subtrees_non_root(left, right, Mode::Hash);
          }
          black_box(&*out);
        },
        |(), out| assert_eq!(*out, expected, "upstream parents at {parents}"),
      ),
    );
  }
  g.finish();
}
