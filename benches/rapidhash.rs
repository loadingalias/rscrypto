//! RapidHash V3 comparison benchmarks.
//!
//! The byte-hash and streaming rows compute the C++ RapidHash V3 function, and
//! must match upstream before timing. The collection rows compare three
//! different `Hasher`s: rscrypto mixes buffered integers in before each byte
//! write, upstream `fast` keeps them buffered and skips avalanche rounds, and
//! upstream `quality` adds them. `fast` does less work than rscrypto and
//! `quality` more, so a claim cites both. No external specification defines
//! these hashers, so pinned known answers check each one; map rows check their
//! results.

#[path = "common/criterion.rs"]
mod bench_config;

mod common;

use core::{
  hash::{BuildHasher, Hash, Hasher},
  hint::black_box,
};
use std::collections::HashMap;

use bench_config::{checked, checked_routine};
use criterion::{BatchSize, Bencher, BenchmarkId, Criterion, Throughput};

const COLLECTION_KEYS: usize = 4096;

/// `hash_one` of each `comp_sizes` input: (length, rscrypto seed 0, upstream fast, upstream quality).
const BUILDHASHER_KNOWN_ANSWERS: [(usize, [u64; 3]); 11] = [
  (0, [0xd0e9c5b79ce32c48, 0xaeb6b65fe5579182, 0x776d1fb05dc4cd58]),
  (1, [0x0839f6f3c4b1810b, 0xe6c8b4c58a412bc5, 0x81ba20f91701b208]),
  (32, [0xb862adad2a38c136, 0x738c36adc35e63d8, 0xae608ccd13e24cff]),
  (64, [0x039320a8365fa087, 0xf8917d09db6249fe, 0x9add6d5759e16a82]),
  (256, [0x86da9fa3c7337dd6, 0x113d7be7e57e37a9, 0xa57f14fac8aff4d6]),
  (1024, [0x10554b5f3280fe94, 0x13a1e14d0eb4daf9, 0x4d9597a9f753f3d3]),
  (4096, [0xb00a9d42c2127b41, 0x771bd53e47453bb6, 0x7eee20e012e3b9ea]),
  (16384, [0x31737366b033202f, 0xd2cfbbbb55e6f696, 0x5d67ad7813ff4dca]),
  (65536, [0x1f1335da224ba355, 0x68eac8d5bc3f8d28, 0x151ead03daeea079]),
  (262144, [0xeb514753067a56b1, 0x24c55dc4d82e5ba0, 0x3fff6f08bce9ccbf]),
  (1048576, [0xdcb6e7cb1209859c, 0xbf6cd695419cd63f, 0xd2f06304e12d27a3]),
];

/// `hash_one` of the u64, 32-byte array, and string keys: (rscrypto seed 0, upstream fast, upstream quality).
const KEY_KNOWN_ANSWERS: [[u64; 3]; 3] = [
  [0xc6e48685502eadfb, 0xc6e48685502eadfb, 0x65602702090e4f3c],
  [0x4514ea13bedb1d85, 0xc40d77536f2e0d13, 0x9f6af865b04262d7],
  [0xec40254713db16ea, 0xbfe0fcb8368a250c, 0xad6921015e86afce],
];

/// The collection hashers, in row order: rscrypto, then upstream `fast` and `quality`.
fn collection_hashers() -> (
  rscrypto::RapidSeededState,
  rapidhash::fast::SeedableState<'static>,
  rapidhash::quality::SeedableState<'static>,
) {
  (
    rscrypto::RapidSeededState::new(0),
    rapidhash::fast::SeedableState::fixed(),
    rapidhash::quality::SeedableState::fixed(),
  )
}

/// A routine that times `state.hash_one(key)` after it returns its known answer.
fn hash_one_row<S: BuildHasher, K: Hash + Copy>(state: S, key: K, answer: u64) -> impl FnMut(&mut Bencher<'_>) {
  checked(
    move || state.hash_one(black_box(key)),
    move |out| assert_eq!(out, answer),
  )
}

/// A routine that times inserting every key into an empty map with reserved
/// capacity. The check reads every key back. Map allocation and destruction are
/// untimed.
fn insert_row<S: BuildHasher + Copy>(state: S, keys: &[[u8; 32]]) -> impl FnMut(&mut Bencher<'_>) {
  checked_routine(
    (),
    move |_| {
      let mut map = HashMap::with_capacity_and_hasher(COLLECTION_KEYS, state);
      insert_all(&mut map, keys);
      assert_eq!(count_found(&map, keys), COLLECTION_KEYS);
      assert_eq!(map.len(), COLLECTION_KEYS);
    },
    move |b, _| {
      b.iter_batched_ref(
        || HashMap::with_capacity_and_hasher(COLLECTION_KEYS, state),
        |map| insert_all(map, keys),
        BatchSize::LargeInput,
      )
    },
  )
}

fn collection_key(index: u64) -> [u8; 32] {
  let mut state = index;
  let mut key = [0u8; 32];
  for lane in key.as_chunks_mut::<8>().0 {
    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut word = state;
    word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    lane.copy_from_slice(&(word ^ (word >> 31)).to_le_bytes());
  }
  key
}

fn collection_keys(start: u64) -> Vec<[u8; 32]> {
  (0..COLLECTION_KEYS)
    .map(|index| collection_key(start.wrapping_add(index as u64)))
    .collect()
}

fn populated_map<S: BuildHasher>(keys: &[[u8; 32]], state: S) -> HashMap<[u8; 32], usize, S> {
  let mut map = HashMap::with_capacity_and_hasher(keys.len(), state);
  map.extend(keys.iter().copied().enumerate().map(|(value, key)| (key, value)));
  map
}

fn insert_all<S: BuildHasher>(map: &mut HashMap<[u8; 32], usize, S>, keys: &[[u8; 32]]) {
  for (value, key) in black_box(keys).iter().copied().enumerate() {
    black_box(map.insert(key, value));
  }
}

fn count_found<S: BuildHasher>(map: &HashMap<[u8; 32], usize, S>, keys: &[[u8; 32]]) -> usize {
  let mut found = 0usize;
  for key in black_box(keys) {
    found = found.wrapping_add(black_box(map.get(black_box(key))).is_some() as usize);
  }
  found
}

fn rapidhash_v3_64(c: &mut Criterion) {
  if !bench_config::selected("rapidhash-v3-64") {
    return;
  }
  let inputs = common::comp_sizes();
  let secrets = rapidhash::v3::RapidSecrets::seed_cpp(0);
  let mut group = c.benchmark_group("rapidhash-v3-64");

  for (len, data) in &inputs {
    common::set_throughput(&mut group, *len);
    let expected = rapidhash::v3::rapidhash_v3_seeded(data, &secrets);

    group.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::RapidHash64::hash(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
    group.bench_function(
      BenchmarkId::new("rapidhash", len),
      checked(
        || rapidhash::v3::rapidhash_v3_seeded(black_box(data), &secrets),
        |out| assert_eq!(out, expected),
      ),
    );
  }

  group.finish();
}

fn rapidhash_seeded_state(c: &mut Criterion) {
  if !bench_config::selected("rapidhash-buildhasher") {
    return;
  }
  let inputs = common::comp_sizes();
  let (ours, fast, quality) = collection_hashers();
  let mut group = c.benchmark_group("rapidhash-buildhasher");

  for ((len, data), (answer_len, answers)) in inputs.iter().zip(BUILDHASHER_KNOWN_ANSWERS) {
    assert_eq!(*len, answer_len, "known answers follow the size matrix");
    common::set_throughput(&mut group, *len);
    let key = data.as_slice();
    group.bench_function(BenchmarkId::new("rscrypto", len), hash_one_row(ours, key, answers[0]));
    group.bench_function(
      BenchmarkId::new("rapidhash-fast", len),
      hash_one_row(fast, key, answers[1]),
    );
    group.bench_function(
      BenchmarkId::new("rapidhash-quality", len),
      hash_one_row(quality, key, answers[2]),
    );
  }
  group.finish();
}

fn rapidhash_key_types(c: &mut Criterion) {
  if !bench_config::selected("rapidhash-hash-one/") {
    return;
  }
  let (ours, fast, quality) = collection_hashers();
  let [integer_answers, bytes_answers, text_answers] = KEY_KNOWN_ANSWERS;

  let integer = 0xa5c3_17e9_6b4d_2f01u64;
  let mut group = c.benchmark_group("rapidhash-hash-one/u64");
  group.bench_function("rscrypto", hash_one_row(ours, integer, integer_answers[0]));
  group.bench_function("rapidhash-fast", hash_one_row(fast, integer, integer_answers[1]));
  group.bench_function("rapidhash-quality", hash_one_row(quality, integer, integer_answers[2]));
  group.finish();

  let bytes = collection_key(17);
  let mut group = c.benchmark_group("rapidhash-hash-one/array-32");
  group.bench_function("rscrypto", hash_one_row(ours, bytes, bytes_answers[0]));
  group.bench_function("rapidhash-fast", hash_one_row(fast, bytes, bytes_answers[1]));
  group.bench_function("rapidhash-quality", hash_one_row(quality, bytes, bytes_answers[2]));
  group.finish();

  let text = "collection-key";
  let mut group = c.benchmark_group("rapidhash-hash-one/str-14");
  group.bench_function("rscrypto", hash_one_row(ours, text, text_answers[0]));
  group.bench_function("rapidhash-fast", hash_one_row(fast, text, text_answers[1]));
  group.bench_function("rapidhash-quality", hash_one_row(quality, text, text_answers[2]));
  group.finish();
}

fn rapidhash_hashmap_operations(c: &mut Criterion) {
  if !bench_config::selected("rapidhash-hashmap/") {
    return;
  }
  let present = collection_keys(0);
  let absent = collection_keys(COLLECTION_KEYS as u64);
  let (ours, fast, quality) = collection_hashers();

  let mut insert = c.benchmark_group("rapidhash-hashmap/insert-32");
  insert.throughput(Throughput::Elements(COLLECTION_KEYS as u64));
  insert.bench_function("rscrypto", insert_row(ours, &present));
  insert.bench_function("rapidhash-fast", insert_row(fast, &present));
  insert.bench_function("rapidhash-quality", insert_row(quality, &present));
  insert.finish();

  let ours = populated_map(&present, ours);
  let fast = populated_map(&present, fast);
  let quality = populated_map(&present, quality);
  for (group_name, keys, found) in [
    ("rapidhash-hashmap/hit-32", &present, COLLECTION_KEYS),
    ("rapidhash-hashmap/miss-32", &absent, 0),
  ] {
    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Elements(COLLECTION_KEYS as u64));
    group.bench_function(
      "rscrypto",
      checked(|| count_found(&ours, keys), |out| assert_eq!(out, found)),
    );
    group.bench_function(
      "rapidhash-fast",
      checked(|| count_found(&fast, keys), |out| assert_eq!(out, found)),
    );
    group.bench_function(
      "rapidhash-quality",
      checked(|| count_found(&quality, keys), |out| assert_eq!(out, found)),
    );
    group.finish();
  }
}

fn rapidhash_streaming(c: &mut Criterion) {
  if !bench_config::selected("rapidhash-stream/") {
    return;
  }
  let inputs = common::comp_sizes();
  let secrets = rapidhash::v3::RapidSecrets::seed_cpp(0);

  for (group_name, chunk_size) in [
    ("rapidhash-stream/one-write", usize::MAX),
    ("rapidhash-stream/chunk-64", 64),
  ] {
    let mut group = c.benchmark_group(group_name);
    for (len, data) in &inputs {
      common::set_throughput(&mut group, *len);
      let expected = rapidhash::v3::rapidhash_v3_seeded(data, &secrets);

      group.bench_function(
        BenchmarkId::new("rscrypto", len),
        checked(
          || {
            let mut hasher = rscrypto::RapidStreamHasher::new();
            if chunk_size == usize::MAX {
              hasher.write(black_box(data));
            } else {
              for chunk in data.chunks(chunk_size) {
                hasher.write(black_box(chunk));
              }
            }
            hasher.finish()
          },
          |out| assert_eq!(out, expected),
        ),
      );

      group.bench_function(
        BenchmarkId::new("rapidhash", len),
        checked(
          || {
            let mut hasher = rapidhash::v3::RapidStreamHasherV3::new(&secrets);
            if chunk_size == usize::MAX {
              hasher.write(black_box(data));
            } else {
              for chunk in data.chunks(chunk_size) {
                hasher.write(black_box(chunk));
              }
            }
            hasher.finish()
          },
          |out| assert_eq!(out, expected),
        ),
      );
    }
    group.finish();
  }
}

fn main() {
  bench_config::run(&[
    rapidhash_v3_64,
    rapidhash_seeded_state,
    rapidhash_key_types,
    rapidhash_hashmap_operations,
    rapidhash_streaming,
  ]);
}
