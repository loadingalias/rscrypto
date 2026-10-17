//! Ascon benchmarks for rscrypto public APIs.
//!
//! Before timing, Ascon-Hash256 and Ascon-XOF128 rows must return the ascon-hash
//! crate's output for every message. No comparison library implements
//! Ascon-CXOF128: its rows must reproduce a NIST SP 800-232 reference vector
//! through the timed API and match rscrypto's incremental CXOF for their input.

#[path = "common/criterion.rs"]
mod bench_config;

mod common;

use core::{cell::LazyCell, hint::black_box};

use bench_config::{checked, checked_with};
use criterion::{BenchmarkId, Criterion};
use rscrypto::Xof as _;

/// ascon-hash's Ascon-XOF128 output for `data`, written to `out`.
fn reference_xof(data: &[u8], out: &mut [u8]) {
  use ascon_hash::{ExtendableOutput, Update, XofReader};
  let mut h = ascon_hash::AsconXof128::default();
  h.update(data);
  h.finalize_xof().read(out);
}

/// Message (`00..1f`) and customization (`10..2f`) of Ascon-CXOF128 KAT 1089 in
/// the final reference vectors, `testdata/ascon/asconcxof128.txt`.
fn cxof_kat_1089() -> (Vec<u8>, Vec<u8>) {
  ((0x00u8..0x20).collect(), (0x10u8..0x30).collect())
}

/// Digest of Ascon-CXOF128 KAT 1089.
const CXOF_KAT_1089_MD: [u8; 64] = [
  0xf2, 0x2e, 0x68, 0xaf, 0xdb, 0xc5, 0xce, 0xd4, 0x21, 0xb3, 0x67, 0x9e, 0x86, 0x06, 0x66, 0x6b, 0x1e, 0x9f, 0x08,
  0x0e, 0xf4, 0x2a, 0x70, 0xef, 0x9b, 0xc3, 0x2d, 0x76, 0x4d, 0xdd, 0xf2, 0x7f, 0xcd, 0x87, 0x94, 0x90, 0x42, 0x89,
  0x77, 0xc5, 0xac, 0x7d, 0x7d, 0xb6, 0xcb, 0x16, 0xc9, 0xe2, 0x6a, 0x2c, 0xea, 0x15, 0x86, 0xec, 0x0a, 0xe2, 0xaa,
  0x5c, 0x8b, 0x01, 0x35, 0xcf, 0xc3, 0x8c,
];

fn ascon_hash256(c: &mut Criterion) {
  if !bench_config::selected("ascon-hash256") {
    return;
  }
  let inputs = common::comp_sizes(&[]);
  let mut g = c.benchmark_group("ascon-hash256");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = <ascon_hash::AsconHash256 as ascon_hash::Digest>::digest(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || <rscrypto::AsconHash256 as rscrypto::Digest>::digest(black_box(data)),
        |out| assert_eq!(out.as_slice(), expected.as_slice()),
      ),
    );

    g.bench_function(
      BenchmarkId::new("ascon-hash", len),
      checked(
        || <ascon_hash::AsconHash256 as ascon_hash::Digest>::digest(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
  }

  g.finish();
}

fn ascon_hash256_streaming(c: &mut Criterion) {
  if !bench_config::selected("ascon-hash256/streaming") {
    return;
  }
  let data = common::random_bytes(1048576);
  let expected = <ascon_hash::AsconHash256 as ascon_hash::Digest>::digest(&data);
  let mut g = c.benchmark_group("ascon-hash256/streaming");
  g.throughput(criterion::Throughput::Bytes(data.len() as u64));

  for chunk_size in [64, 4096] {
    g.bench_function(
      format!("rscrypto/{chunk_size}B"),
      checked(
        || {
          use rscrypto::Digest;
          let mut h = rscrypto::AsconHash256::new();
          for chunk in data.chunks(chunk_size) {
            h.update(black_box(chunk));
          }
          h.finalize()
        },
        |out| assert_eq!(out.as_slice(), expected.as_slice()),
      ),
    );
  }

  g.finish();
}

fn ascon_hash256_many(c: &mut Criterion) {
  if !bench_config::selected("ascon-hash256/many") {
    return;
  }
  const MSG_LEN: usize = 4096;
  const COUNT: usize = 128;

  let slab = common::random_bytes(MSG_LEN * COUNT);
  let inputs: Vec<&[u8]> = slab.chunks(MSG_LEN).collect();
  let expected = LazyCell::new(|| {
    inputs
      .iter()
      .map(|input| <[u8; 32]>::from(<ascon_hash::AsconHash256 as ascon_hash::Digest>::digest(input)))
      .collect::<Vec<_>>()
  });
  let mut g = c.benchmark_group("ascon-hash256/many");
  g.throughput(criterion::Throughput::Bytes((MSG_LEN * COUNT) as u64));

  // Timed: hash the fixed message batch into reused output storage.
  // Untimed: message fixtures, ascon-hash reference digests, and output allocation/destruction.
  g.bench_function(
    "rscrypto/batch-auto",
    checked_with(
      vec![[0u8; 32]; COUNT],
      |out| {
        rscrypto::AsconHash256::digest_many(black_box(&inputs), black_box(&mut *out));
        black_box(out[0])
      },
      |_, out| assert_eq!(*out, *expected),
    ),
  );

  // Same message/output boundary as batch-auto, using repeated rscrypto scalar calls.
  g.bench_function(
    "rscrypto/scalar-loop",
    checked_with(
      vec![[0u8; 32]; COUNT],
      |out| {
        for (input, slot) in inputs.iter().zip(out.iter_mut()) {
          use rscrypto::Digest as _;
          *slot = rscrypto::AsconHash256::digest(black_box(input));
        }
        black_box(out[0])
      },
      |_, out| assert_eq!(*out, *expected),
    ),
  );

  g.finish();
}

fn ascon_xof128(c: &mut Criterion) {
  if !bench_config::selected("ascon-xof128") {
    return;
  }
  let inputs = common::comp_sizes(&[]);
  let mut g = c.benchmark_group("ascon-xof128");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let mut expected = [0u8; 32];
    reference_xof(data, &mut expected);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || {
          use rscrypto::Xof;
          let mut h = rscrypto::AsconXof::new();
          h.update(black_box(data));
          let mut xof = h.finalize_xof();
          let mut out = [0u8; 32];
          xof.squeeze(&mut out);
          out
        },
        |out| assert_eq!(out, expected),
      ),
    );

    g.bench_function(
      BenchmarkId::new("ascon-hash", len),
      checked(
        || {
          use ascon_hash::digest::{ExtendableOutput, Update, XofReader};
          let mut h = ascon_hash::AsconXof128::default();
          h.update(black_box(data));
          let mut reader = h.finalize_xof();
          let mut out = [0u8; 32];
          reader.read(&mut out);
          out
        },
        |out| assert_eq!(out, expected),
      ),
    );
  }

  g.finish();
}

fn ascon_xof128_many(c: &mut Criterion) {
  if !bench_config::selected("ascon-xof128/many") {
    return;
  }
  const MSG_LEN: usize = 4096;
  const OUT_LEN: usize = 64;
  const COUNT: usize = 128;

  let slab = common::random_bytes(MSG_LEN * COUNT);
  let inputs: Vec<&[u8]> = slab.chunks(MSG_LEN).collect();
  let expected = LazyCell::new(|| {
    let mut expected = vec![0u8; COUNT * OUT_LEN];
    for (input, out) in inputs.iter().zip(expected.as_chunks_mut::<OUT_LEN>().0) {
      reference_xof(input, out);
    }
    expected
  });
  let mut g = c.benchmark_group("ascon-xof128/many");
  g.throughput(criterion::Throughput::Bytes((MSG_LEN * COUNT) as u64));

  // Timed: hash the fixed message batch into reused output storage.
  // Untimed: message fixtures, ascon-hash reference outputs, and output allocation/destruction.
  g.bench_function(
    "rscrypto/batch-auto",
    checked_with(
      vec![0u8; COUNT * OUT_LEN],
      |out| {
        rscrypto::AsconXof::hash_many_into(black_box(&inputs), OUT_LEN, black_box(&mut *out));
        black_box(out[0])
      },
      |_, out| assert_eq!(*out, *expected),
    ),
  );

  // Same message/output boundary as batch-auto, using repeated rscrypto scalar calls.
  g.bench_function(
    "rscrypto/scalar-loop",
    checked_with(
      vec![0u8; COUNT * OUT_LEN],
      |out| {
        for (index, input) in inputs.iter().enumerate() {
          let mut hasher = rscrypto::AsconXof::new();
          hasher.update(black_box(input));
          let mut reader = hasher.finalize_xof();
          let base = index.strict_mul(OUT_LEN);
          reader.squeeze(&mut out[base..base.strict_add(OUT_LEN)]);
        }
        black_box(out[0])
      },
      |_, out| assert_eq!(*out, *expected),
    ),
  );

  g.finish();
}

fn ascon_cxof128(c: &mut Criterion) {
  if !bench_config::selected("ascon-cxof128") {
    return;
  }
  const CUSTOMIZATION: &[u8] = b"rscrypto-bench";
  const OUT_LEN: usize = 32;

  let inputs = common::comp_sizes(&[]);
  let mut g = c.benchmark_group("ascon-cxof128");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || {
          let mut out = [0u8; OUT_LEN];
          rscrypto::AsconCxof128::hash_into(black_box(CUSTOMIZATION), black_box(data), &mut out)
            .expect("valid Ascon benchmark operation must succeed");
          out
        },
        |out| {
          let (message, customization) = cxof_kat_1089();
          let mut kat = [0u8; 64];
          rscrypto::AsconCxof128::hash_into(&customization, &message, &mut kat)
            .expect("Ascon-CXOF128 KAT customization must be valid");
          assert_eq!(kat, CXOF_KAT_1089_MD, "Ascon-CXOF128 KAT 1089");

          let mut hasher =
            rscrypto::AsconCxof128::new(CUSTOMIZATION).expect("valid Ascon benchmark operation must succeed");
          hasher.update(data);
          let mut incremental = [0u8; OUT_LEN];
          hasher.finalize_xof().squeeze(&mut incremental);
          assert_eq!(out, incremental);
        },
      ),
    );
  }

  g.finish();
}

fn main() {
  bench_config::run(&[
    ascon_hash256,
    ascon_hash256_streaming,
    ascon_hash256_many,
    ascon_xof128,
    ascon_xof128_many,
    ascon_cxof128,
  ]);
}
