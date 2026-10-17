//! RapidHash V3 comparison benchmarks.
//!
//! Before timing, byte-hash and streaming rows must match upstream RapidHash V3.
//! The collection hashers have no external specification and differ by design,
//! so pinned known answers check each library; map rows check their results.

#[path = "common/criterion.rs"]
mod bench_config;

mod common;

use core::{
  hash::{BuildHasher, Hasher},
  hint::black_box,
};
use std::collections::HashMap;

use bench_config::{checked, checked_routine};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};

const COLLECTION_KEYS: usize = 4096;

/// `hash_one` of each `comp_sizes` input: (length, rscrypto seed 0, upstream fixed state).
const BUILDHASHER_KNOWN_ANSWERS: [(usize, u64, u64); 11] = [
  (0, 0xd0e9c5b79ce32c48, 0x776d1fb05dc4cd58),
  (1, 0x0839f6f3c4b1810b, 0x81ba20f91701b208),
  (32, 0xb862adad2a38c136, 0xae608ccd13e24cff),
  (64, 0x039320a8365fa087, 0x9add6d5759e16a82),
  (256, 0x86da9fa3c7337dd6, 0xa57f14fac8aff4d6),
  (1024, 0x10554b5f3280fe94, 0x4d9597a9f753f3d3),
  (4096, 0xb00a9d42c2127b41, 0x7eee20e012e3b9ea),
  (16384, 0x31737366b033202f, 0x5d67ad7813ff4dca),
  (65536, 0x1f1335da224ba355, 0x151ead03daeea079),
  (262144, 0xeb514753067a56b1, 0x3fff6f08bce9ccbf),
  (1048576, 0xdcb6e7cb1209859c, 0xd2f06304e12d27a3),
];

/// `hash_one` of the u64, 32-byte array, and string keys: (rscrypto seed 0, upstream fixed state).
const KEY_KNOWN_ANSWERS: [(u64, u64); 3] = [
  (0xc6e48685502eadfb, 0x65602702090e4f3c),
  (0x4514ea13bedb1d85, 0x9f6af865b04262d7),
  (0xec40254713db16ea, 0xad6921015e86afce),
];

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
  let mut group = c.benchmark_group("rapidhash-buildhasher");
  let ours = rscrypto::RapidSeededState::new(0);
  let upstream = rapidhash::quality::SeedableState::fixed();

  for ((len, data), (answer_len, ours_answer, upstream_answer)) in inputs.iter().zip(BUILDHASHER_KNOWN_ANSWERS) {
    assert_eq!(*len, answer_len, "known answers follow the size matrix");
    common::set_throughput(&mut group, *len);
    group.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || ours.hash_one(black_box(data.as_slice())),
        |out| assert_eq!(out, ours_answer),
      ),
    );
    group.bench_function(
      BenchmarkId::new("rapidhash", len),
      checked(
        || upstream.hash_one(black_box(data.as_slice())),
        |out| assert_eq!(out, upstream_answer),
      ),
    );
  }
  group.finish();
}

fn rapidhash_key_types(c: &mut Criterion) {
  if !bench_config::selected("rapidhash-hash-one/") {
    return;
  }
  let ours = rscrypto::RapidSeededState::new(0);
  let upstream = rapidhash::quality::SeedableState::fixed();
  let integer = 0xa5c3_17e9_6b4d_2f01u64;
  let bytes = collection_key(17);
  let text = "collection-key";
  let [integer_answers, bytes_answers, text_answers] = KEY_KNOWN_ANSWERS;

  let mut integer_group = c.benchmark_group("rapidhash-hash-one/u64");
  integer_group.bench_function(
    "rscrypto",
    checked(
      || ours.hash_one(black_box(integer)),
      |out| assert_eq!(out, integer_answers.0),
    ),
  );
  integer_group.bench_function(
    "rapidhash",
    checked(
      || upstream.hash_one(black_box(integer)),
      |out| assert_eq!(out, integer_answers.1),
    ),
  );
  integer_group.finish();

  let mut bytes_group = c.benchmark_group("rapidhash-hash-one/array-32");
  bytes_group.bench_function(
    "rscrypto",
    checked(
      || ours.hash_one(black_box(bytes)),
      |out| assert_eq!(out, bytes_answers.0),
    ),
  );
  bytes_group.bench_function(
    "rapidhash",
    checked(
      || upstream.hash_one(black_box(bytes)),
      |out| assert_eq!(out, bytes_answers.1),
    ),
  );
  bytes_group.finish();

  let mut text_group = c.benchmark_group("rapidhash-hash-one/str-14");
  text_group.bench_function(
    "rscrypto",
    checked(|| ours.hash_one(black_box(text)), |out| assert_eq!(out, text_answers.0)),
  );
  text_group.bench_function(
    "rapidhash",
    checked(
      || upstream.hash_one(black_box(text)),
      |out| assert_eq!(out, text_answers.1),
    ),
  );
  text_group.finish();
}

fn rapidhash_hashmap_operations(c: &mut Criterion) {
  if !bench_config::selected("rapidhash-hashmap/") {
    return;
  }
  let present = collection_keys(0);
  let absent = collection_keys(COLLECTION_KEYS as u64);
  let ours = populated_map(&present, rscrypto::RapidSeededState::new(0));
  let upstream = populated_map(&present, rapidhash::quality::SeedableState::fixed());

  // Timed: inserting every key into an empty map with reserved capacity.
  // Untimed: map allocation and destruction.
  let mut insert = c.benchmark_group("rapidhash-hashmap/insert-32");
  insert.throughput(Throughput::Elements(COLLECTION_KEYS as u64));
  insert.bench_function(
    "rscrypto",
    checked_routine(
      (),
      |_| {
        let mut map = HashMap::with_capacity_and_hasher(COLLECTION_KEYS, rscrypto::RapidSeededState::new(0));
        insert_all(&mut map, &present);
        assert_eq!(count_found(&map, &present), COLLECTION_KEYS);
        assert_eq!(map.len(), COLLECTION_KEYS);
      },
      |b, _| {
        b.iter_batched_ref(
          || HashMap::with_capacity_and_hasher(COLLECTION_KEYS, rscrypto::RapidSeededState::new(0)),
          |map| insert_all(map, &present),
          BatchSize::LargeInput,
        )
      },
    ),
  );
  insert.bench_function(
    "rapidhash",
    checked_routine(
      (),
      |_| {
        let mut map = HashMap::with_capacity_and_hasher(COLLECTION_KEYS, rapidhash::quality::SeedableState::fixed());
        insert_all(&mut map, &present);
        assert_eq!(count_found(&map, &present), COLLECTION_KEYS);
        assert_eq!(map.len(), COLLECTION_KEYS);
      },
      |b, _| {
        b.iter_batched_ref(
          || HashMap::with_capacity_and_hasher(COLLECTION_KEYS, rapidhash::quality::SeedableState::fixed()),
          |map| insert_all(map, &present),
          BatchSize::LargeInput,
        )
      },
    ),
  );
  insert.finish();

  let mut hit = c.benchmark_group("rapidhash-hashmap/hit-32");
  hit.throughput(Throughput::Elements(COLLECTION_KEYS as u64));
  hit.bench_function(
    "rscrypto",
    checked(
      || count_found(&ours, &present),
      |found| assert_eq!(found, COLLECTION_KEYS),
    ),
  );
  hit.bench_function(
    "rapidhash",
    checked(
      || count_found(&upstream, &present),
      |found| assert_eq!(found, COLLECTION_KEYS),
    ),
  );
  hit.finish();

  let mut miss = c.benchmark_group("rapidhash-hashmap/miss-32");
  miss.throughput(Throughput::Elements(COLLECTION_KEYS as u64));
  miss.bench_function(
    "rscrypto",
    checked(|| count_found(&ours, &absent), |found| assert_eq!(found, 0)),
  );
  miss.bench_function(
    "rapidhash",
    checked(|| count_found(&upstream, &absent), |found| assert_eq!(found, 0)),
  );
  miss.finish();
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
