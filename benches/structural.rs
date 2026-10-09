//! Stable structural-cost benchmarks for representative public hash and checksum paths.

use core::hint::black_box;

use ::blake3 as upstream_blake3;
use gungraun::{Callgrind, EntryPoint, LibraryBenchmarkConfig, library_benchmark, library_benchmark_group};
use rscrypto::{
  Blake3, Blake3DeriveContext, Blake3KeyedHash, Checksum, Crc32, Digest as _, Sha256, Xof as _,
  hashes::expert::blake3_tree::{Blake3ChainingValue, Blake3Tree},
};

static INPUT_64: [u8; 64] = [0x3d; 64];
static INPUT_4K: [u8; 4096] = [0xa7; 4096];
static INPUT_16K: [u8; 16_384] = [0x5c; 16_384];
const KEY: [u8; 32] = [0x53; 32];
const CONTEXT: &str = "rscrypto BLAKE3 structural counters";

fn blake3_collection() -> LibraryBenchmarkConfig {
  // The outer ID wrapper contains only the benchmark call. Collecting there
  // keeps setup outside the measurement and avoids relying on Callgrind's
  // ARM call/return tracking inside the inlined public operation.
  let mut config = LibraryBenchmarkConfig::default();
  config.tool(Callgrind::default().entry_point(EntryPoint::Custom("*::__gungraun_wrapper_id_mod*::*".to_owned())));
  config
}

fn blake3_fixture_collection() -> LibraryBenchmarkConfig {
  // Explicit client requests exclude the returned fixture's later Drop even
  // when Callgrind's ARM call/return heuristics lose the wrapper boundary.
  let mut config = LibraryBenchmarkConfig::default();
  config.tool(Callgrind::with_args(["collect-atstart=no"]).entry_point(EntryPoint::None));
  config
}

// A call boundary accounts for the final production stores before collection
// turns off, including keyed scratch cleanup in the caller's basic block.
#[inline(never)]
#[cfg_attr(
  not(target_os = "linux"),
  expect(
    clippy::panic,
    reason = "unsupported collection must fail instead of reporting zero counters"
  )
)]
fn toggle_fixture_collection() {
  #[cfg(target_os = "linux")]
  gungraun::client_requests::callgrind::toggle_collect();
  #[cfg(not(target_os = "linux"))]
  panic!("BLAKE3 fixture structural benchmarks require Linux client requests");
}

#[library_benchmark]
#[bench::bytes_64(&INPUT_64)]
#[bench::bytes_4096(&INPUT_4K)]
fn sha256(input: &[u8]) {
  black_box(Sha256::digest(black_box(input)));
}

#[library_benchmark(config = blake3_collection())]
#[bench::bytes_64(&INPUT_64)]
#[bench::bytes_4096(&INPUT_4K)]
#[bench::bytes_16384(&INPUT_16K)]
fn blake3(input: &[u8]) {
  black_box(Blake3::digest(black_box(input)));
}

// Setup validates the same public operation outside collection. The benchmark
// includes construction, updates, finalization and destruction of its hasher.
fn streaming_input(prefix: usize, bulk: usize) -> (&'static [u8], usize) {
  let input = &INPUT_16K[..prefix.strict_add(bulk)];
  let mut hasher = Blake3::new();
  hasher.update(&input[..prefix]);
  hasher.update(&input[prefix..]);
  assert_eq!(&hasher.finalize(), upstream_blake3::hash(input).as_bytes());
  (input, prefix)
}

#[library_benchmark(config = blake3_collection())]
#[bench::prefix_70_bulk_4096(streaming_input(70, 4096))]
#[bench::prefix_24_bulk_3104(streaming_input(24, 3104))]
fn blake3_streaming((input, prefix): (&[u8], usize)) {
  let mut hasher = Blake3::new();
  hasher.update(black_box(&input[..prefix]));
  hasher.update(black_box(&input[prefix..]));
  black_box(hasher.finalize());
}

fn keyed_input(input: &'static [u8]) -> &'static [u8] {
  assert_eq!(
    Blake3::keyed_digest(&KEY, input).as_bytes(),
    upstream_blake3::keyed_hash(&KEY, input).as_bytes()
  );
  input
}

#[library_benchmark(config = blake3_collection())]
#[bench::bytes_64(keyed_input(&INPUT_64))]
#[bench::bytes_4096(keyed_input(&INPUT_4K))]
fn blake3_keyed(input: &[u8]) {
  // The call includes its internal keyed scratch cleanup.
  black_box(Blake3::keyed_digest(black_box(&KEY), black_box(input)));
}

fn keyed_xof_input() -> &'static [u8] {
  let mut output = [0; 256];
  let mut hasher = Blake3::new_keyed(&KEY);
  hasher.update(&INPUT_4K);
  hasher.finalize_xof().squeeze(&mut output);
  let mut expected = [0; 256];
  upstream_blake3::Hasher::new_keyed(&KEY)
    .update(&INPUT_4K)
    .finalize_xof()
    .fill(&mut expected);
  assert_eq!(output, expected);
  &INPUT_4K
}

#[library_benchmark(config = blake3_collection())]
#[bench::input_4096_output_256(keyed_xof_input())]
fn blake3_keyed_xof(input: &[u8]) {
  let mut output = [0; 256];
  let mut hasher = Blake3::new_keyed(black_box(&KEY));
  hasher.update(black_box(input));
  hasher.finalize_xof().squeeze(&mut output);
  black_box(output);
}

struct BatchFixture {
  inputs: [&'static [u8]; 16],
  output: [[u8; 32]; 16],
  keyed_output: [Blake3KeyedHash; 16],
  context: Blake3DeriveContext,
}

fn batch_fixture() -> Box<BatchFixture> {
  let lengths = [
    0, 1, 24, 63, 64, 65, 70, 127, 255, 511, 512, 513, 1000, 1023, 1024, 4097,
  ];
  let mut fixture = Box::new(BatchFixture {
    inputs: lengths.map(|len| &INPUT_16K[..len]),
    output: [[0; 32]; 16],
    keyed_output: core::array::from_fn(|_| Blake3KeyedHash::default()),
    context: Blake3DeriveContext::new(CONTEXT),
  });
  Blake3::digest_batch(&fixture.inputs, &mut fixture.output);
  for (input, output) in fixture.inputs.iter().zip(&fixture.output) {
    assert_eq!(output, upstream_blake3::hash(input).as_bytes());
  }
  Blake3::keyed_digest_batch(&KEY, &fixture.inputs, &mut fixture.keyed_output);
  for (input, output) in fixture.inputs.iter().zip(&fixture.keyed_output) {
    assert_eq!(output.as_bytes(), upstream_blake3::keyed_hash(&KEY, input).as_bytes());
  }
  fixture.context.derive_key_batch(&fixture.inputs, &mut fixture.output);
  for (input, output) in fixture.inputs.iter().zip(&fixture.output) {
    assert_eq!(output, &upstream_blake3::derive_key(CONTEXT, input));
  }
  fixture
}

// Requests enclose the public call, including its internal scratch cleanup.
// The prepared fixture is allocated before collection and dropped after it.
#[library_benchmark(config = blake3_fixture_collection())]
#[bench::mixed_16(batch_fixture())]
fn blake3_batch(mut fixture: Box<BatchFixture>) -> Box<BatchFixture> {
  toggle_fixture_collection();
  Blake3::digest_batch(black_box(&fixture.inputs), black_box(&mut fixture.output));
  toggle_fixture_collection();
  black_box(fixture)
}

#[library_benchmark(config = blake3_fixture_collection())]
#[bench::mixed_16(batch_fixture())]
fn blake3_keyed_batch(mut fixture: Box<BatchFixture>) -> Box<BatchFixture> {
  toggle_fixture_collection();
  Blake3::keyed_digest_batch(
    black_box(&KEY),
    black_box(&fixture.inputs),
    black_box(&mut fixture.keyed_output),
  );
  toggle_fixture_collection();
  black_box(fixture)
}

#[library_benchmark(config = blake3_fixture_collection())]
#[bench::mixed_16(batch_fixture())]
fn blake3_derive_batch(mut fixture: Box<BatchFixture>) -> Box<BatchFixture> {
  toggle_fixture_collection();
  fixture
    .context
    .derive_key_batch(black_box(&fixture.inputs), black_box(&mut fixture.output));
  toggle_fixture_collection();
  black_box(fixture)
}

struct MergeFixture {
  tree: Blake3Tree,
  children: Vec<Blake3ChainingValue>,
  parents: Vec<Blake3ChainingValue>,
}

fn merge_fixture(mode: u8) -> Box<MergeFixture> {
  let context_key = upstream_blake3::hazmat::hash_derive_key_context(CONTEXT);
  let (tree, oracle_mode) = match mode {
    0 => (Blake3Tree::new(), upstream_blake3::hazmat::Mode::Hash),
    1 => (Blake3Tree::keyed(&KEY), upstream_blake3::hazmat::Mode::KeyedHash(&KEY)),
    _ => (
      Blake3Tree::derive_key(&Blake3DeriveContext::new(CONTEXT)),
      upstream_blake3::hazmat::Mode::DeriveKeyMaterial(&context_key),
    ),
  };
  let children: Vec<_> = (0u64..128)
    .map(|index| {
      let mut subtree = tree.subtree(index.strict_mul(1024)).expect("chunk offset");
      subtree.update(&INPUT_4K[..1024]).expect("one chunk");
      subtree.finalize().expect("nonempty chunk")
    })
    .collect();
  let mut parents = children[..64].to_vec();
  tree.merge_level(&children, &mut parents).expect("valid level");
  for (pair, parent) in children.as_chunks::<2>().0.iter().zip(&parents) {
    assert_eq!(
      parent.as_bytes(),
      &upstream_blake3::hazmat::merge_subtrees_non_root(pair[0].as_bytes(), pair[1].as_bytes(), oracle_mode)
    );
  }
  Box::new(MergeFixture {
    tree,
    children,
    parents,
  })
}

#[library_benchmark(config = blake3_fixture_collection())]
#[bench::parents_64_hash(merge_fixture(0))]
#[bench::parents_64_keyed(merge_fixture(1))]
#[bench::parents_64_derive(merge_fixture(2))]
fn blake3_merge_level(mut fixture: Box<MergeFixture>) -> Box<MergeFixture> {
  toggle_fixture_collection();
  fixture
    .tree
    .merge_level(black_box(&fixture.children), black_box(&mut fixture.parents))
    .expect("valid level");
  toggle_fixture_collection();
  black_box(fixture)
}

#[library_benchmark]
#[bench::bytes_64(&INPUT_64)]
#[bench::bytes_4096(&INPUT_4K)]
fn crc32(input: &[u8]) {
  black_box(Crc32::checksum(black_box(input)));
}

library_benchmark_group!(
  name = structural;
  benchmarks = sha256, blake3, crc32, blake3_streaming, blake3_keyed, blake3_keyed_xof,
    blake3_batch, blake3_keyed_batch, blake3_derive_batch, blake3_merge_level
);
gungraun::main!(library_benchmark_groups = structural);
