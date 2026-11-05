//! SipHash-1-3 and SipHash-2-4 one-shot comparison benchmarks.
//!
//! Lengths are the shared comparison sizes plus SipHash's 8-byte block
//! boundaries and the 20-byte maximum QUIC connection ID.
//! Every implementation hashes the same bytes under the same key, and each row
//! must match the `siphasher` result before timing. std's deprecated
//! `SipHasher` is SipHash-2-4; std exposes no keyed SipHash-1-3.

#[path = "common/criterion.rs"]
mod bench_config;

mod common;

use core::hash::Hasher;
use core::hint::black_box;

use bench_config::checked;
use criterion::{BenchmarkId, Criterion};

const KEY: [u8; 16] = [
  0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00,
];
const BOUNDARIES: [usize; 5] = [7, 8, 15, 16, 20];

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

fn main() {
  bench_config::run(&[siphash13, siphash24]);
}
