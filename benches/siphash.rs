//! SipHash-1-3 and SipHash-2-4 comparison benchmarks.
//!
//! The one-shot rows hash one byte string under a fixed key. Their lengths are
//! the shared comparison sizes plus SipHash's 8-byte block boundaries and the
//! 20-byte maximum QUIC connection ID. std's deprecated `SipHasher` is
//! SipHash-2-4; std exposes no keyed SipHash-1-3.
//!
//! The collection rows time the path that a hash table takes:
//! `BuildHasher::hash_one`, the key's `Hash` implementation, and the
//! `Hasher` writes it makes. Every state reads its key at run time, as a
//! table's state does: a key known at compile time would let the compiler
//! fold the first rounds. rscrypto and `siphasher` hash under the fixed key,
//! and their rows must match the `siphasher` result before timing; both feed
//! the same bytes for integer writes on either byte order, so the check holds
//! on every target. std's `RandomState` is SipHash-1-3 under a random key, so
//! its rows check that a second call returns the same hash.

#[path = "common/criterion.rs"]
mod bench_config;

mod common;

use core::hash::{BuildHasher, Hash, Hasher};
use core::hint::black_box;
use core::net::{Ipv6Addr, SocketAddr, SocketAddrV6};
use std::collections::HashMap;
use std::hash::RandomState;

use bench_config::checked;
use criterion::{BenchmarkId, Criterion};

const KEY: [u8; 16] = [
  0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00,
];
const BOUNDARIES: [usize; 5] = [7, 8, 15, 16, 20];
/// Table-key lengths: empty, the 8-byte block boundaries, the 20-byte maximum
/// QUIC connection ID, and short composite keys.
const KEY_LENGTHS: [usize; 10] = [0, 1, 4, 7, 8, 15, 16, 20, 32, 64];
const TABLE_KEYS: usize = 4096;

/// `siphasher`'s SipHash-1-3 under a key read at run time.
#[derive(Clone)]
struct SiphasherState([u8; 16]);

impl BuildHasher for SiphasherState {
  type Hasher = siphasher::sip::SipHasher13;

  fn build_hasher(&self) -> Self::Hasher {
    siphasher::sip::SipHasher13::new_with_key(&self.0)
  }
}

/// The collection hashers, each keyed at run time: rscrypto, `siphasher`, std.
fn collection_states() -> (rscrypto::SipRandomState, SiphasherState, RandomState) {
  let key = black_box(KEY);
  (
    rscrypto::SipRandomState::new(rscrypto::SipHashKey::from_bytes(key)),
    SiphasherState(key),
    RandomState::new(),
  )
}

/// Times `state.hash_one(key)` after checking it: against `expected`, or
/// without one, against a separate untimed call.
fn hash_one_row<S: BuildHasher, K: Hash + Copy>(
  state: S,
  key: K,
  expected: Option<u64>,
) -> impl FnMut(&mut criterion::Bencher<'_>) {
  let expected = expected.unwrap_or_else(|| state.hash_one(key));
  checked(
    move || state.hash_one(black_box(key)),
    move |out| assert_eq!(out, expected),
  )
}

/// A QUIC connection ID of up to 20 bytes; unused bytes stay zero.
#[derive(Clone, Copy, PartialEq, Eq)]
struct ConnectionId {
  len: u8,
  bytes: [u8; 20],
}

impl Hash for ConnectionId {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.bytes[..usize::from(self.len)].hash(state);
  }
}

fn connection_ids() -> Vec<ConnectionId> {
  (0..TABLE_KEYS)
    .map(|index| {
      let bytes = common::random_bytes(index.strict_add(20));
      let mut id = ConnectionId {
        len: 20,
        bytes: [0; 20],
      };
      id.bytes.copy_from_slice(&bytes[index..]);
      // Mix 8- and 20-byte IDs, the common server choices.
      if index % 2 == 0 {
        id.len = 8;
        id.bytes[8..].fill(0);
      }
      id
    })
    .collect()
}

fn lookup_row<S: BuildHasher>(state: S, keys: &[ConnectionId]) -> impl FnMut(&mut criterion::Bencher<'_>) {
  let mut map = HashMap::with_capacity_and_hasher(keys.len(), state);
  map.extend(keys.iter().copied().enumerate().map(|(value, key)| (key, value)));
  let keys = keys.to_vec();
  checked(
    move || {
      let mut found = 0usize;
      for key in black_box(&keys) {
        found = found.wrapping_add(usize::from(black_box(map.get(black_box(key))).is_some()));
      }
      found
    },
    |found| assert_eq!(found, TABLE_KEYS),
  )
}

fn siphash13(c: &mut Criterion) {
  if !bench_config::selected("siphash13") {
    return;
  }
  let key = rscrypto::SipHashKey::from_bytes(KEY);
  let mut group = c.benchmark_group("siphash13");
  for (length, data) in common::comp_sizes(&BOUNDARIES) {
    common::set_throughput(&mut group, length);
    let reference = || {
      let mut hasher = siphasher::sip::SipHasher13::new_with_key(&KEY);
      hasher.write(black_box(&data));
      hasher.finish()
    };
    let expected = reference();
    group.bench_function(
      BenchmarkId::new("rscrypto", length),
      checked(
        || rscrypto::SipHash13::hash(&key, black_box(&data)),
        |out| assert_eq!(out, expected),
      ),
    );
    group.bench_function(
      BenchmarkId::new("siphasher", length),
      checked(reference, |out| assert_eq!(out, expected)),
    );
  }
  group.finish();
}

#[expect(deprecated, reason = "std's SipHasher is SipHash-2-4, the comparison baseline")]
fn siphash24(c: &mut Criterion) {
  if !bench_config::selected("siphash24") {
    return;
  }
  let key = rscrypto::SipHashKey::from_bytes(KEY);
  let (k0, k1) = KEY.split_at(8);
  let (k0, k1) = (
    u64::from_le_bytes(k0.try_into().expect("half")),
    u64::from_le_bytes(k1.try_into().expect("half")),
  );
  let mut group = c.benchmark_group("siphash24");
  for (length, data) in common::comp_sizes(&BOUNDARIES) {
    common::set_throughput(&mut group, length);
    let reference = || {
      let mut hasher = siphasher::sip::SipHasher24::new_with_key(&KEY);
      hasher.write(black_box(&data));
      hasher.finish()
    };
    let expected = reference();
    group.bench_function(
      BenchmarkId::new("rscrypto", length),
      checked(
        || rscrypto::SipHash24::hash(&key, black_box(&data)),
        |out| assert_eq!(out, expected),
      ),
    );
    group.bench_function(
      BenchmarkId::new("siphasher", length),
      checked(reference, |out| assert_eq!(out, expected)),
    );
    group.bench_function(
      BenchmarkId::new("std", length),
      checked(
        || {
          let mut hasher = core::hash::SipHasher::new_with_keys(k0, k1);
          hasher.write(black_box(&data));
          hasher.finish()
        },
        |out| assert_eq!(out, expected),
      ),
    );
  }
  group.finish();
}

fn siphash13_buildhasher(c: &mut Criterion) {
  if !bench_config::selected("siphash13-buildhasher") {
    return;
  }
  let (ours, theirs, std) = collection_states();
  let mut group = c.benchmark_group("siphash13-buildhasher");
  for length in KEY_LENGTHS {
    let data = common::random_bytes(length);
    let key: &[u8] = &data;
    common::set_throughput(&mut group, length);
    let expected = theirs.hash_one(key);
    group.bench_function(
      BenchmarkId::new("rscrypto", length),
      hash_one_row(ours.clone(), key, Some(expected)),
    );
    group.bench_function(
      BenchmarkId::new("siphasher", length),
      hash_one_row(theirs.clone(), key, Some(expected)),
    );
    group.bench_function(BenchmarkId::new("std", length), hash_one_row(std.clone(), key, None));
  }
  group.finish();
}

/// One table key of each shape: a stream ID, a peer address, and a connection ID.
fn siphash13_key_types(c: &mut Criterion) {
  if !bench_config::selected("siphash13-hash-one") {
    return;
  }
  fn rows<K: Hash + Copy>(c: &mut Criterion, name: &str, key: K) {
    let (ours, theirs, std) = collection_states();
    let expected = theirs.hash_one(key);
    let mut group = c.benchmark_group(format!("siphash13-hash-one/{name}"));
    group.bench_function("rscrypto", hash_one_row(ours, key, Some(expected)));
    group.bench_function("siphasher", hash_one_row(theirs, key, Some(expected)));
    group.bench_function("std", hash_one_row(std, key, None));
    group.finish();
  }
  rows(c, "u64", 0x0123_4567_89ab_cdef_u64);
  rows(
    c,
    "socket-v6",
    SocketAddr::V6(SocketAddrV6::new(
      Ipv6Addr::new(0x2001, 0x0db8, 0, 0, 0, 0, 0x8a2e, 0x0370),
      443,
      0,
      0,
    )),
  );
  rows(c, "cid-20", connection_ids()[1]);
}

/// Look up every key of a table of mixed 8- and 20-byte connection IDs.
fn siphash13_hashmap(c: &mut Criterion) {
  if !bench_config::selected("siphash13-hashmap") {
    return;
  }
  let keys = connection_ids();
  let (ours, theirs, std) = collection_states();
  let mut group = c.benchmark_group("siphash13-hashmap/lookup-cid");
  group.bench_function("rscrypto", lookup_row(ours, &keys));
  group.bench_function("siphasher", lookup_row(theirs, &keys));
  group.bench_function("std", lookup_row(std, &keys));
  group.finish();
}

fn main() {
  bench_config::run(&[
    siphash13,
    siphash24,
    siphash13_buildhasher,
    siphash13_key_types,
    siphash13_hashmap,
  ]);
}
