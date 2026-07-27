//! RapidHash V3 comparison benchmarks.

mod common;

use core::{
  hash::{BuildHasher, Hasher},
  hint::black_box,
};
use std::collections::HashMap;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const COLLECTION_KEYS: usize = 4096;

fn collection_key(index: u64) -> [u8; 32] {
  let mut state = index;
  let mut key = [0u8; 32];
  for lane in key.chunks_exact_mut(8) {
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

fn rapidhash_v3_64(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut group = c.benchmark_group("rapidhash-v3-64");

  for (len, data) in &inputs {
    common::set_throughput(&mut group, *len);

    group.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, data| {
      b.iter(|| black_box(rscrypto::RapidHash64::hash(black_box(data))))
    });

    group.bench_with_input(BenchmarkId::new("rapidhash", len), data, |b, data| {
      let secrets = rapidhash::v3::RapidSecrets::seed_cpp(0);
      b.iter(|| black_box(rapidhash::v3::rapidhash_v3_seeded(black_box(data), &secrets)))
    });
  }

  group.finish();
}

fn rapidhash_seeded_state(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut group = c.benchmark_group("rapidhash-buildhasher");
  let ours = rscrypto::RapidSeededState::new(0);
  let upstream = rapidhash::quality::SeedableState::fixed();

  for (len, data) in &inputs {
    common::set_throughput(&mut group, *len);
    group.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, data| {
      b.iter(|| black_box(ours.hash_one(black_box(data.as_slice()))))
    });
    group.bench_with_input(BenchmarkId::new("rapidhash", len), data, |b, data| {
      b.iter(|| black_box(upstream.hash_one(black_box(data.as_slice()))))
    });
  }
  group.finish();
}

fn rapidhash_key_types(c: &mut Criterion) {
  let ours = rscrypto::RapidSeededState::new(0);
  let upstream = rapidhash::quality::SeedableState::fixed();
  let integer = 0xa5c3_17e9_6b4d_2f01u64;
  let bytes = collection_key(17);
  let text = "collection-key";

  let mut integer_group = c.benchmark_group("rapidhash-hash-one/u64");
  integer_group.bench_function("rscrypto", |b| b.iter(|| black_box(ours.hash_one(black_box(integer)))));
  integer_group.bench_function("rapidhash", |b| {
    b.iter(|| black_box(upstream.hash_one(black_box(integer))))
  });
  integer_group.finish();

  let mut bytes_group = c.benchmark_group("rapidhash-hash-one/array-32");
  bytes_group.bench_function("rscrypto", |b| b.iter(|| black_box(ours.hash_one(black_box(bytes)))));
  bytes_group.bench_function("rapidhash", |b| {
    b.iter(|| black_box(upstream.hash_one(black_box(bytes))))
  });
  bytes_group.finish();

  let mut text_group = c.benchmark_group("rapidhash-hash-one/str-14");
  text_group.bench_function("rscrypto", |b| b.iter(|| black_box(ours.hash_one(black_box(text)))));
  text_group.bench_function("rapidhash", |b| {
    b.iter(|| black_box(upstream.hash_one(black_box(text))))
  });
  text_group.finish();
}

fn rapidhash_hashmap_operations(c: &mut Criterion) {
  let present = collection_keys(0);
  let absent = collection_keys(COLLECTION_KEYS as u64);
  let ours = populated_map(&present, rscrypto::RapidSeededState::new(0));
  let upstream = populated_map(&present, rapidhash::quality::SeedableState::fixed());

  let mut insert = c.benchmark_group("rapidhash-hashmap/insert-32");
  insert.throughput(Throughput::Elements(COLLECTION_KEYS as u64));
  insert.bench_function("rscrypto", |b| {
    b.iter_batched_ref(
      || HashMap::with_capacity_and_hasher(COLLECTION_KEYS, rscrypto::RapidSeededState::new(0)),
      |map| {
        for (value, key) in black_box(present.as_slice()).iter().copied().enumerate() {
          black_box(map.insert(key, value));
        }
      },
      BatchSize::LargeInput,
    )
  });
  insert.bench_function("rapidhash", |b| {
    b.iter_batched_ref(
      || HashMap::with_capacity_and_hasher(COLLECTION_KEYS, rapidhash::quality::SeedableState::fixed()),
      |map| {
        for (value, key) in black_box(present.as_slice()).iter().copied().enumerate() {
          black_box(map.insert(key, value));
        }
      },
      BatchSize::LargeInput,
    )
  });
  insert.finish();

  let mut hit = c.benchmark_group("rapidhash-hashmap/hit-32");
  hit.throughput(Throughput::Elements(COLLECTION_KEYS as u64));
  hit.bench_function("rscrypto", |b| {
    b.iter(|| {
      let mut found = 0usize;
      for key in black_box(present.as_slice()) {
        found = found.wrapping_add(black_box(ours.get(black_box(key))).is_some() as usize);
      }
      black_box(found)
    })
  });
  hit.bench_function("rapidhash", |b| {
    b.iter(|| {
      let mut found = 0usize;
      for key in black_box(present.as_slice()) {
        found = found.wrapping_add(black_box(upstream.get(black_box(key))).is_some() as usize);
      }
      black_box(found)
    })
  });
  hit.finish();

  let mut miss = c.benchmark_group("rapidhash-hashmap/miss-32");
  miss.throughput(Throughput::Elements(COLLECTION_KEYS as u64));
  miss.bench_function("rscrypto", |b| {
    b.iter(|| {
      let mut found = 0usize;
      for key in black_box(absent.as_slice()) {
        found = found.wrapping_add(black_box(ours.get(black_box(key))).is_some() as usize);
      }
      black_box(found)
    })
  });
  miss.bench_function("rapidhash", |b| {
    b.iter(|| {
      let mut found = 0usize;
      for key in black_box(absent.as_slice()) {
        found = found.wrapping_add(black_box(upstream.get(black_box(key))).is_some() as usize);
      }
      black_box(found)
    })
  });
  miss.finish();
}

fn rapidhash_streaming(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let secrets = rapidhash::v3::RapidSecrets::seed_cpp(0);

  for (group_name, chunk_size) in [
    ("rapidhash-stream/one-write", usize::MAX),
    ("rapidhash-stream/chunk-64", 64),
  ] {
    let mut group = c.benchmark_group(group_name);
    for (len, data) in &inputs {
      common::set_throughput(&mut group, *len);

      group.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, data| {
        b.iter(|| {
          let mut hasher = rscrypto::RapidStreamHasher::new();
          if chunk_size == usize::MAX {
            hasher.write(black_box(data));
          } else {
            for chunk in data.chunks(chunk_size) {
              hasher.write(black_box(chunk));
            }
          }
          black_box(hasher.finish())
        })
      });

      group.bench_with_input(BenchmarkId::new("rapidhash", len), data, |b, data| {
        b.iter(|| {
          let mut hasher = rapidhash::v3::RapidStreamHasherV3::new(&secrets);
          if chunk_size == usize::MAX {
            hasher.write(black_box(data));
          } else {
            for chunk in data.chunks(chunk_size) {
              hasher.write(black_box(chunk));
            }
          }
          black_box(hasher.finish())
        })
      });
    }
    group.finish();
  }
}

criterion_group!(
  benches,
  rapidhash_v3_64,
  rapidhash_seeded_state,
  rapidhash_key_types,
  rapidhash_hashmap_operations,
  rapidhash_streaming
);
criterion_main!(benches);
