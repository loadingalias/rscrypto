//! SHA-3 family comparison benchmarks: rscrypto vs sha3 crate.
//!
//! Before timing, SHA-3 rows must return the sha3 crate's digest and SHAKE rows
//! tiny-keccak's output for their input. Long SHAKE output rows from tiny-keccak
//! must return dryoc's output.

#[path = "common/criterion.rs"]
mod bench_config;

mod common;

use core::hint::black_box;

use bench_config::{checked, checked_with};
use criterion::{BenchmarkId, Criterion};

macro_rules! sha3_oneshot {
  ($fn_name:ident, $group:literal, $ours:ty, $theirs:ty) => {
    fn $fn_name(c: &mut Criterion) {
      if !bench_config::selected($group) {
        return;
      }
      let inputs = common::comp_sizes();
      let mut g = c.benchmark_group($group);

      for (len, data) in &inputs {
        common::set_throughput(&mut g, *len);
        let expected = <$theirs as sha3::Digest>::digest(data);

        g.bench_function(
          BenchmarkId::new("rscrypto", len),
          checked(
            || <$ours as rscrypto::Digest>::digest(black_box(data)),
            |out| assert_eq!(out.as_ref(), expected.as_slice()),
          ),
        );

        g.bench_function(
          BenchmarkId::new("sha3", len),
          checked(
            || <$theirs as sha3::Digest>::digest(black_box(data)),
            |out| assert_eq!(out, expected),
          ),
        );
      }

      g.finish();
    }
  };
}

sha3_oneshot!(sha3_224, "sha3-224", rscrypto::Sha3_224, sha3::Sha3_224);
sha3_oneshot!(sha3_256, "sha3-256", rscrypto::Sha3_256, sha3::Sha3_256);
sha3_oneshot!(sha3_384, "sha3-384", rscrypto::Sha3_384, sha3::Sha3_384);
sha3_oneshot!(sha3_512, "sha3-512", rscrypto::Sha3_512, sha3::Sha3_512);

/// Streams 1 MiB in fixed-size chunks.
macro_rules! sha3_streaming {
  ($fn_name:ident, $group:literal, $ours:ty, $theirs:ty) => {
    fn $fn_name(c: &mut Criterion) {
      if !bench_config::selected($group) {
        return;
      }
      let data = common::random_bytes(1048576);
      let rscrypto_digest = |chunk_size: usize| {
        use rscrypto::Digest;
        let mut h = <$ours>::new();
        for chunk in data.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        h.finalize()
      };
      let sha3_digest = |chunk_size: usize| {
        use sha3::Digest;
        let mut h = <$theirs>::new();
        for chunk in data.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        h.finalize()
      };

      let mut g = c.benchmark_group($group);
      g.throughput(criterion::Throughput::Bytes(data.len() as u64));

      let expected = <$theirs as sha3::Digest>::digest(&data);
      for chunk_size in [64, 4096] {
        g.bench_function(
          format!("rscrypto/{chunk_size}B"),
          checked(
            || rscrypto_digest(chunk_size),
            |out| assert_eq!(out.as_ref(), expected.as_slice()),
          ),
        );

        g.bench_function(
          format!("sha3/{chunk_size}B"),
          checked(|| sha3_digest(chunk_size), |out| assert_eq!(out, expected)),
        );
      }

      g.finish();
    }
  };
}

sha3_streaming!(
  sha3_224_streaming,
  "sha3-224/streaming",
  rscrypto::Sha3_224,
  sha3::Sha3_224
);
sha3_streaming!(
  sha3_256_streaming,
  "sha3-256/streaming",
  rscrypto::Sha3_256,
  sha3::Sha3_256
);
sha3_streaming!(
  sha3_384_streaming,
  "sha3-384/streaming",
  rscrypto::Sha3_384,
  sha3::Sha3_384
);
sha3_streaming!(
  sha3_512_streaming,
  "sha3-512/streaming",
  rscrypto::Sha3_512,
  sha3::Sha3_512
);

fn shake128(c: &mut Criterion) {
  if !bench_config::selected("shake128") {
    return;
  }
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("shake128");

  let tiny_keccak = |data: &[u8]| {
    use tiny_keccak::{Hasher as _, Xof as _};
    let mut hasher = tiny_keccak::Shake::v128();
    hasher.update(data);
    let mut out = [0u8; 32];
    hasher.squeeze(&mut out);
    out
  };

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = tiny_keccak(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || {
          use rscrypto::Xof;
          let mut h = rscrypto::Shake128::new();
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
      BenchmarkId::new("tiny-keccak", len),
      checked(|| tiny_keccak(black_box(data)), |out| assert_eq!(out, expected)),
    );
  }

  g.finish();
}

fn shake256(c: &mut Criterion) {
  if !bench_config::selected("shake256") {
    return;
  }
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("shake256");

  let tiny_keccak = |data: &[u8]| {
    use tiny_keccak::{Hasher as _, Xof as _};
    let mut hasher = tiny_keccak::Shake::v256();
    hasher.update(data);
    let mut out = [0u8; 32];
    hasher.squeeze(&mut out);
    out
  };

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = tiny_keccak(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || {
          use rscrypto::Xof;
          let mut h = rscrypto::Shake256::new();
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
      BenchmarkId::new("tiny-keccak", len),
      checked(|| tiny_keccak(black_box(data)), |out| assert_eq!(out, expected)),
    );
  }

  g.finish();
}

/// Input length for long XOF output: a 32-byte seed, as in key and matrix expansion.
const XOF_SEED_LEN: usize = 32;
/// Output lengths for long XOF output. The one-shot groups cover 32-byte output; these measure the squeeze.
const XOF_OUTPUT_LENGTHS: [usize; 3] = [1024, 16384, 1048576];

/// Absorbs a 32-byte seed and squeezes long output into a reused buffer.
///
/// Timed: state construction, absorption, finalization, the squeeze, and state destruction. The output buffer is
/// allocated outside timing. Every row must return tiny-keccak's output; tiny-keccak's row must return dryoc's.
/// RustCrypto's `sha3` 0.12 has no SHAKE; dryoc's SHAKE runs the `keccak` crate's permutation.
macro_rules! shake_xof_output {
  ($fn_name:ident, $group:literal, $ours:ty, $theirs:ty, $tiny_keccak:path) => {
    fn $fn_name(c: &mut Criterion) {
      if !bench_config::selected($group) {
        return;
      }
      let seed = common::random_bytes(XOF_SEED_LEN);
      let tiny_keccak = |seed: &[u8], out: &mut [u8]| {
        use tiny_keccak::{Hasher as _, Xof as _};
        let mut hasher = $tiny_keccak();
        hasher.update(seed);
        hasher.squeeze(out);
      };
      let dryoc = |seed: &[u8], out: &mut [u8]| {
        let mut hasher = <$theirs>::new();
        hasher.update(seed);
        hasher.finalize().squeeze(out);
      };
      let mut g = c.benchmark_group($group);

      for out_len in XOF_OUTPUT_LENGTHS {
        common::set_throughput(&mut g, out_len);
        let mut expected = vec![0u8; out_len];
        tiny_keccak(&seed, &mut expected);
        let mut dryoc_expected = vec![0u8; out_len];
        dryoc(&seed, &mut dryoc_expected);

        g.bench_function(
          BenchmarkId::new("rscrypto", out_len),
          checked_with(
            vec![0u8; out_len],
            |out| {
              use rscrypto::Xof as _;
              let mut hasher = <$ours>::new();
              hasher.update(black_box(&seed));
              hasher.finalize_xof().squeeze(black_box(out.as_mut_slice()));
            },
            |(), out| assert_eq!(*out, expected),
          ),
        );
        g.bench_function(
          BenchmarkId::new("tiny-keccak", out_len),
          checked_with(
            vec![0u8; out_len],
            |out| tiny_keccak(black_box(&seed), black_box(out.as_mut_slice())),
            |(), out| assert_eq!(*out, dryoc_expected),
          ),
        );
        g.bench_function(
          BenchmarkId::new("dryoc", out_len),
          checked_with(
            vec![0u8; out_len],
            |out| dryoc(black_box(&seed), black_box(out.as_mut_slice())),
            |(), out| assert_eq!(*out, expected),
          ),
        );
      }

      g.finish();
    }
  };
}

shake_xof_output!(
  shake128_xof_output,
  "shake128/xof-output",
  rscrypto::Shake128,
  dryoc::xof::Shake128,
  tiny_keccak::Shake::v128
);
shake_xof_output!(
  shake256_xof_output,
  "shake256/xof-output",
  rscrypto::Shake256,
  dryoc::xof::Shake256,
  tiny_keccak::Shake::v256
);

fn sha3_256_digest_pair(c: &mut Criterion) {
  if !bench_config::selected("sha3-256/digest_pair") {
    return;
  }
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("sha3-256/digest_pair");

  for (len, first) in &inputs {
    // Throughput is 2× because we hash two messages of the same length. Distinct
    // messages let the check catch a swapped or mixed lane.
    g.throughput(criterion::Throughput::Bytes((*len as u64).strict_mul(2)));
    let second: Vec<u8> = first.iter().map(|byte| byte ^ 0xa5).collect();
    let expected = (
      <sha3::Sha3_256 as sha3::Digest>::digest(first),
      <sha3::Sha3_256 as sha3::Digest>::digest(&second),
    );
    let matches = |(a, b): ([u8; 32], [u8; 32])| {
      assert_eq!(a.as_slice(), expected.0.as_slice());
      assert_eq!(b.as_slice(), expected.1.as_slice());
    };

    g.bench_function(
      BenchmarkId::new("pair", len),
      checked(
        || rscrypto::Sha3_256::digest_pair(black_box(first), black_box(&second)),
        matches,
      ),
    );

    g.bench_function(
      BenchmarkId::new("2x_sequential", len),
      checked(
        || {
          use rscrypto::Digest as _;
          let a = rscrypto::Sha3_256::digest(black_box(first));
          let b = rscrypto::Sha3_256::digest(black_box(&second));
          (a, b)
        },
        matches,
      ),
    );
  }

  g.finish();
}

fn main() {
  bench_config::run(&[
    sha3_224,
    sha3_256,
    sha3_384,
    sha3_512,
    sha3_224_streaming,
    sha3_256_streaming,
    sha3_384_streaming,
    sha3_512_streaming,
    shake128,
    shake256,
    shake128_xof_output,
    shake256_xof_output,
    sha3_256_digest_pair,
  ]);
}
