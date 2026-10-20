//! SHA-3 family comparison benchmarks: rscrypto against RustCrypto, AWS-LC,
//! tiny-keccak, and dryoc.
//!
//! RustCrypto's `sha3` and `shake` select the AArch64 SHA3 instructions at run
//! time. AWS-LC runs s2n-bignum Keccak assembly on x86-64 and AArch64 and
//! portable C elsewhere; it is called through aws-lc-rs where that exposes the
//! algorithm and through aws-lc-sys otherwise.
//!
//! Before timing, SHA-3 rows must return the sha3 crate's digest, and SHAKE
//! rows tiny-keccak's output. tiny-keccak's SHAKE rows must return RustCrypto's.

#[path = "common/criterion.rs"]
mod bench_config;

#[path = "common/aws_lc.rs"]
mod aws_lc_support;
use aws_lc_support::aws_lc_bench;

mod common;

use core::hint::black_box;

use bench_config::{checked, checked_with};
use criterion::{BenchmarkId, Criterion};

aws_lc_bench! {
  use core::{ffi::c_uint, ptr};

  use aws_lc_sys as aws_lc;

  /// AWS-LC's SHAKE variants. Each descriptor is an XOF.
  #[derive(Clone, Copy)]
  enum AwsLcShake {
    Shake128,
    Shake256,
  }

  /// Calls AWS-LC's one-call `EVP_Digest`, which writes `*out_size` bytes for an XOF and the digest length for a
  /// fixed-length digest.
  ///
  /// Citations name files under aws-lc-sys 0.46.0's `aws-lc/` tree: `include/openssl/digest.h:159-167` and
  /// `crypto/fipsmodule/digest/digest.c:282-313`. `EVP_Digest` keeps its context on its own stack, and its final
  /// call cleans the context up (`digest.c:90-101`, `digest.c:282-286`).
  ///
  /// # Safety
  ///
  /// `md` must be an AWS-LC digest descriptor that writes at most `out.len()` bytes: an XOF, or a fixed-length
  /// digest of at most `out.len()` bytes.
  unsafe fn aws_lc_digest(md: *const aws_lc::EVP_MD, data: &[u8], out: &mut [u8]) {
    let mut out_size = c_uint::try_from(out.len()).expect("benchmark outputs fit an unsigned int");
    // SAFETY: Hashes `data` into `out` because:
    // 1. `data.as_ptr()` is readable for `data.len()` bytes; AWS-LC keeps no pointer to it.
    // 2. `md` is a non-null, process-static descriptor that writes at most `out.len()` bytes (caller contract), and
    //    `out.as_mut_ptr()` is writable for `out.len()` bytes.
    // 3. `out_size` is non-null and holds `out.len()`, the length an XOF writes (digest.c:293-307).
    // 4. A null `impl_` selects the default implementation.
    let rc = unsafe {
      aws_lc::EVP_Digest(data.as_ptr().cast(), data.len(), out.as_mut_ptr(), &raw mut out_size, md, ptr::null_mut())
    };
    assert_eq!(rc, 1, "AWS-LC EVP_Digest must succeed");
  }

  /// AWS-LC SHA3-224, which aws-lc-rs does not expose.
  fn aws_lc_sha3_224(data: &[u8]) -> [u8; 28] {
    let mut out = [0u8; 28];
    // SAFETY: `EVP_sha3_224` takes no arguments and returns a process-static descriptor whose digest is 28 bytes
    // (`crypto/fipsmodule/digest/digests.c:292-294`), which `out` holds.
    unsafe { aws_lc_digest(aws_lc::EVP_sha3_224(), data, &mut out) };
    out
  }

  /// AWS-LC SHAKE of `data` into all of `out`. aws-lc-rs does not expose SHAKE.
  fn aws_lc_shake(variant: AwsLcShake, data: &[u8], out: &mut [u8]) {
    let md = match variant {
      // SAFETY: Takes no arguments and returns a process-static XOF descriptor (`digests.c:408-411`).
      AwsLcShake::Shake128 => unsafe { aws_lc::EVP_shake128() },
      // SAFETY: Takes no arguments and returns a process-static XOF descriptor (`digests.c:442-445`).
      AwsLcShake::Shake256 => unsafe { aws_lc::EVP_shake256() },
    };
    // SAFETY: `md` is an XOF descriptor (`crypto/fipsmodule/digest/digests.c`), so it writes exactly `out.len()`
    // bytes.
    unsafe { aws_lc_digest(md, data, out) };
  }
}

/// RustCrypto SHAKE of `data` into all of `out`.
fn rustcrypto_shake<H: Default + shake::digest::Update + shake::digest::ExtendableOutput>(data: &[u8], out: &mut [u8]) {
  use shake::digest::XofReader as _;
  let mut hasher = H::default();
  hasher.update(data);
  hasher.finalize_xof().read(out);
}

/// One-shot hashing across the shared size matrix. `$boundaries` adds the
/// longest message whose padding fits one absorption block and the shortest
/// that needs a second: one byte below the rate, and the rate.
/// `$aws_name` and `$aws` give AWS-LC's row: the library name and its one-shot digest of `$data`.
macro_rules! sha3_oneshot {
  (
    $fn_name:ident,
    $group:literal,
    $ours:ty,
    $theirs:ty,
    $boundaries:expr,
    aws: ($aws_name:literal, |$data:ident| $aws:expr) $(,)?
  ) => {
    fn $fn_name(c: &mut Criterion) {
      if !bench_config::selected($group) {
        return;
      }
      let inputs = common::comp_sizes(&$boundaries);
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

        aws_lc_bench! {
          g.bench_function(
            BenchmarkId::new($aws_name, len),
            checked(
              || {
                let $data = black_box(data.as_slice());
                $aws
              },
              |out| assert_eq!(out.as_ref(), expected.as_slice()),
            ),
          );
        }
      }

      g.finish();
    }
  };
}

sha3_oneshot!(
  sha3_224,
  "sha3-224",
  rscrypto::Sha3_224,
  sha3::Sha3_224,
  [143, 144],
  aws: ("aws-lc", |data| aws_lc_sha3_224(data)),
);
sha3_oneshot!(
  sha3_256,
  "sha3-256",
  rscrypto::Sha3_256,
  sha3::Sha3_256,
  [135, 136],
  aws: ("aws-lc-rs", |data| aws_lc_rs::digest::digest(&aws_lc_rs::digest::SHA3_256, data)),
);
sha3_oneshot!(
  sha3_384,
  "sha3-384",
  rscrypto::Sha3_384,
  sha3::Sha3_384,
  [103, 104],
  aws: ("aws-lc-rs", |data| aws_lc_rs::digest::digest(&aws_lc_rs::digest::SHA3_384, data)),
);
sha3_oneshot!(
  sha3_512,
  "sha3-512",
  rscrypto::Sha3_512,
  sha3::Sha3_512,
  [71, 72],
  aws: ("aws-lc-rs", |data| aws_lc_rs::digest::digest(&aws_lc_rs::digest::SHA3_512, data)),
);

/// Streams 1 MiB in fixed-size chunks. `$aws_algorithm` adds an aws-lc-rs `digest::Context` row; aws-lc-rs has no
/// SHA3-224.
macro_rules! sha3_streaming {
  ($fn_name:ident, $group:literal, $ours:ty, $theirs:ty $(, $aws_algorithm:path)? $(,)?) => {
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

        $(
          aws_lc_bench! {
            g.bench_function(
              format!("aws-lc-rs/{chunk_size}B"),
              checked(
                || {
                  let mut context = aws_lc_rs::digest::Context::new(&$aws_algorithm);
                  for chunk in data.chunks(chunk_size) {
                    context.update(black_box(chunk));
                  }
                  context.finish()
                },
                |out| assert_eq!(out.as_ref(), expected.as_slice()),
              ),
            );
          }
        )?
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
  sha3::Sha3_256,
  aws_lc_rs::digest::SHA3_256,
);
sha3_streaming!(
  sha3_384_streaming,
  "sha3-384/streaming",
  rscrypto::Sha3_384,
  sha3::Sha3_384,
  aws_lc_rs::digest::SHA3_384,
);
sha3_streaming!(
  sha3_512_streaming,
  "sha3-512/streaming",
  rscrypto::Sha3_512,
  sha3::Sha3_512,
  aws_lc_rs::digest::SHA3_512,
);

/// Absorbs one buffer and squeezes 32 bytes in one call. `$boundaries` adds the rate boundary: the longest
/// message whose padding fits one block, and the shortest that needs a second. Every row must return tiny-keccak's
/// output; tiny-keccak's row must return RustCrypto's.
macro_rules! shake_oneshot {
  (
    $fn_name:ident,
    $group:literal,
    $ours:ty,
    $rustcrypto:ty,
    $tiny_keccak:path,
    $aws:expr,
    $boundaries:expr $(,)?
  ) => {
    fn $fn_name(c: &mut Criterion) {
      if !bench_config::selected($group) {
        return;
      }
      let inputs = common::comp_sizes(&$boundaries);
      let mut g = c.benchmark_group($group);

      let tiny_keccak = |data: &[u8]| {
        use tiny_keccak::{Hasher as _, Xof as _};
        let mut hasher = $tiny_keccak();
        hasher.update(data);
        let mut out = [0u8; 32];
        hasher.squeeze(&mut out);
        out
      };
      let rustcrypto = |data: &[u8]| {
        let mut out = [0u8; 32];
        rustcrypto_shake::<$rustcrypto>(data, &mut out);
        out
      };

      for (len, data) in &inputs {
        common::set_throughput(&mut g, *len);
        let expected = tiny_keccak(data);
        let rustcrypto_expected = rustcrypto(data);

        g.bench_function(
          BenchmarkId::new("rscrypto", len),
          checked(
            || {
              use rscrypto::Xof;
              let mut h = <$ours>::new();
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
          BenchmarkId::new("rustcrypto", len),
          checked(|| rustcrypto(black_box(data)), |out| assert_eq!(out, expected)),
        );

        aws_lc_bench! {
          g.bench_function(
            BenchmarkId::new("aws-lc", len),
            checked(
              || {
                let mut out = [0u8; 32];
                aws_lc_shake($aws, black_box(data), &mut out);
                out
              },
              |out| assert_eq!(out, expected),
            ),
          );
        }

        g.bench_function(
          BenchmarkId::new("tiny-keccak", len),
          checked(
            || tiny_keccak(black_box(data)),
            |out| assert_eq!(out, rustcrypto_expected),
          ),
        );
      }

      g.finish();
    }
  };
}

shake_oneshot!(
  shake128,
  "shake128",
  rscrypto::Shake128,
  shake::Shake128,
  tiny_keccak::Shake::v128,
  AwsLcShake::Shake128,
  [167, 168],
);
shake_oneshot!(
  shake256,
  "shake256",
  rscrypto::Shake256,
  shake::Shake256,
  tiny_keccak::Shake::v256,
  AwsLcShake::Shake256,
  [135, 136],
);

/// Input length for long XOF output: a 32-byte seed, as in key and matrix expansion.
const XOF_SEED_LEN: usize = 32;
/// Output lengths for long XOF output. The one-shot groups cover 32-byte output; these measure the squeeze.
const XOF_OUTPUT_LENGTHS: [usize; 3] = [1024, 16384, 1048576];

/// Absorbs a 32-byte seed and squeezes long output into a reused buffer.
///
/// Timed: state construction, absorption, finalization, the squeeze, and state destruction. The output buffer is
/// allocated outside timing. Every row must return tiny-keccak's output; tiny-keccak's row must return RustCrypto's.
/// dryoc's SHAKE runs the portable permutation of the `keccak` crate.
macro_rules! shake_xof_output {
  ($fn_name:ident, $group:literal, $ours:ty, $rustcrypto:ty, $theirs:ty, $tiny_keccak:path, $aws:expr) => {
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
        let mut rustcrypto_expected = vec![0u8; out_len];
        rustcrypto_shake::<$rustcrypto>(&seed, &mut rustcrypto_expected);

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
          BenchmarkId::new("rustcrypto", out_len),
          checked_with(
            vec![0u8; out_len],
            |out| rustcrypto_shake::<$rustcrypto>(black_box(&seed), black_box(out.as_mut_slice())),
            |(), out| assert_eq!(*out, expected),
          ),
        );
        aws_lc_bench! {
          g.bench_function(
            BenchmarkId::new("aws-lc", out_len),
            checked_with(
              vec![0u8; out_len],
              |out| aws_lc_shake($aws, black_box(&seed), black_box(out.as_mut_slice())),
              |(), out| assert_eq!(*out, expected),
            ),
          );
        }
        g.bench_function(
          BenchmarkId::new("tiny-keccak", out_len),
          checked_with(
            vec![0u8; out_len],
            |out| tiny_keccak(black_box(&seed), black_box(out.as_mut_slice())),
            |(), out| assert_eq!(*out, rustcrypto_expected),
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
  shake::Shake128,
  dryoc::xof::Shake128,
  tiny_keccak::Shake::v128,
  AwsLcShake::Shake128
);
shake_xof_output!(
  shake256_xof_output,
  "shake256/xof-output",
  rscrypto::Shake256,
  shake::Shake256,
  dryoc::xof::Shake256,
  tiny_keccak::Shake::v256,
  AwsLcShake::Shake256
);

fn sha3_256_digest_pair(c: &mut Criterion) {
  if !bench_config::selected("sha3-256/digest_pair") {
    return;
  }
  let inputs = common::comp_sizes(&[]);
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
