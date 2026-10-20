//! SHA-2 family comparison benchmarks: rscrypto vs sha2 crate.
//!
//! Before timing, every row must return the sha2 crate's digest of its input.

#[path = "common/criterion.rs"]
mod bench_config;

#[path = "common/aws_lc.rs"]
mod aws_lc_support;
use aws_lc_support::aws_lc_bench;

mod common;

use core::hint::black_box;

use bench_config::checked;
use criterion::{BenchmarkId, Criterion};

#[cfg(all(rscrypto_internal, feature = "diag"))]
fn print_sha2_diag_once() {
  use std::sync::Once;

  static ONCE: Once = Once::new();
  ONCE.call_once(|| {
    use rscrypto::{Sha256, hashes::introspect::kernel_for};

    eprintln!("rscrypto-diag sha2 runtime_caps={}", rscrypto::platform::caps());
    eprintln!("rscrypto-diag sha2 static_caps={}", rscrypto::platform::caps_static());
    eprintln!(
      "rscrypto-diag sha2 target_features sha={} sha512={} avx2={} avx512f={}",
      cfg!(target_feature = "sha"),
      cfg!(target_feature = "sha512"),
      cfg!(target_feature = "avx2"),
      cfg!(target_feature = "avx512f")
    );
    eprintln!(
      "rscrypto-diag sha2 sha256_kernel 64={} 4096={} 1048576={}",
      kernel_for::<Sha256>(64),
      kernel_for::<Sha256>(4096),
      kernel_for::<Sha256>(1_048_576)
    );
  });
}

#[cfg(not(all(rscrypto_internal, feature = "diag")))]
#[inline]
fn print_sha2_diag_once() {}

/// One-shot hashing across the shared size matrix. `$boundaries` adds the longest message whose padding fits one compression
/// block and the shortest that needs a second: 55 and 56 bytes for 64-byte
/// blocks, 111 and 112 for 128-byte blocks.
macro_rules! sha2_oneshot {
  ($fn_name:ident, $group:literal, $ours:ty, $theirs:ty, $boundaries:expr) => {
    fn $fn_name(c: &mut Criterion) {
      if !bench_config::selected($group) {
        return;
      }
      print_sha2_diag_once();

      let inputs = common::comp_sizes(&$boundaries);
      let mut g = c.benchmark_group($group);

      for (len, data) in &inputs {
        common::set_throughput(&mut g, *len);
        let expected = <$theirs as sha2::Digest>::digest(data);

        g.bench_function(
          BenchmarkId::new("rscrypto", len),
          checked(
            || <$ours>::digest(black_box(data)),
            |out| assert_eq!(out.as_ref(), expected.as_slice()),
          ),
        );

        g.bench_function(
          BenchmarkId::new("sha2", len),
          checked(
            || <$theirs as sha2::Digest>::digest(black_box(data)),
            |out| assert_eq!(out, expected),
          ),
        );
      }

      g.finish();
    }
  };
}

/// Variant of `sha2_oneshot!` that also benches `aws-lc-rs` and `ring` — both
/// expose SHA-256/384/512 but neither ships SHA-224 or SHA-512/256.
macro_rules! sha2_oneshot_ietf {
  ($fn_name:ident, $group:literal, $ours:ty, $theirs:ty, $boundaries:expr, $aws_alg:expr, $ring_alg:expr) => {
    fn $fn_name(c: &mut Criterion) {
      if !bench_config::selected($group) {
        return;
      }
      print_sha2_diag_once();

      let inputs = common::comp_sizes(&$boundaries);
      let mut g = c.benchmark_group($group);

      for (len, data) in &inputs {
        common::set_throughput(&mut g, *len);
        let expected = <$theirs as sha2::Digest>::digest(data);

        g.bench_function(
          BenchmarkId::new("rscrypto", len),
          checked(
            || <$ours>::digest(black_box(data)),
            |out| assert_eq!(out.as_ref(), expected.as_slice()),
          ),
        );

        g.bench_function(
          BenchmarkId::new("sha2", len),
          checked(
            || <$theirs as sha2::Digest>::digest(black_box(data)),
            |out| assert_eq!(out, expected),
          ),
        );

        aws_lc_bench! {
          g.bench_function(
            BenchmarkId::new("aws-lc-rs", len),
            checked(
              || aws_lc_rs::digest::digest($aws_alg, black_box(data)),
              |out| assert_eq!(out.as_ref(), expected.as_slice()),
            ),
          );
        }

        g.bench_function(
          BenchmarkId::new("ring", len),
          checked(
            || ring::digest::digest($ring_alg, black_box(data)),
            |out| assert_eq!(out.as_ref(), expected.as_slice()),
          ),
        );
      }

      g.finish();
    }
  };
}

sha2_oneshot!(sha224, "sha224", rscrypto::Sha224, sha2::Sha224, [55, 56]);
sha2_oneshot_ietf!(
  sha256,
  "sha256",
  rscrypto::Sha256,
  sha2::Sha256,
  [55, 56],
  &aws_lc_rs::digest::SHA256,
  &ring::digest::SHA256
);
sha2_oneshot_ietf!(
  sha384,
  "sha384",
  rscrypto::Sha384,
  sha2::Sha384,
  [111, 112],
  &aws_lc_rs::digest::SHA384,
  &ring::digest::SHA384
);
sha2_oneshot_ietf!(
  sha512,
  "sha512",
  rscrypto::Sha512,
  sha2::Sha512,
  [111, 112],
  &aws_lc_rs::digest::SHA512,
  &ring::digest::SHA512
);
sha2_oneshot!(
  sha512_256,
  "sha512-256",
  rscrypto::Sha512_256,
  sha2::Sha512_256,
  [111, 112]
);

/// Streams 1 MiB in fixed-size chunks.
macro_rules! sha2_streaming {
  ($fn_name:ident, $group:literal, $ours:ty, $theirs:ty) => {
    fn $fn_name(c: &mut Criterion) {
      if !bench_config::selected($group) {
        return;
      }
      print_sha2_diag_once();

      let data = common::random_bytes(1048576);
      let rscrypto_digest = |chunk_size: usize| {
        let mut h = <$ours as rscrypto::Digest>::new();
        for chunk in data.chunks(chunk_size) {
          rscrypto::Digest::update(&mut h, black_box(chunk));
        }
        rscrypto::Digest::finalize(&h)
      };
      let sha2_digest = |chunk_size: usize| {
        use sha2::Digest;
        let mut h = <$theirs>::new();
        for chunk in data.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        h.finalize()
      };

      let mut g = c.benchmark_group($group);
      g.throughput(criterion::Throughput::Bytes(data.len() as u64));

      let expected = <$theirs as sha2::Digest>::digest(&data);
      for chunk_size in [64, 4096] {
        g.bench_function(
          format!("rscrypto/{chunk_size}B"),
          checked(
            || rscrypto_digest(chunk_size),
            |out| assert_eq!(out.as_ref(), expected.as_slice()),
          ),
        );

        g.bench_function(
          format!("sha2/{chunk_size}B"),
          checked(|| sha2_digest(chunk_size), |out| assert_eq!(out, expected)),
        );
      }

      g.finish();
    }
  };
}

sha2_streaming!(sha224_streaming, "sha224/streaming", rscrypto::Sha224, sha2::Sha224);
sha2_streaming!(sha256_streaming, "sha256/streaming", rscrypto::Sha256, sha2::Sha256);
sha2_streaming!(sha384_streaming, "sha384/streaming", rscrypto::Sha384, sha2::Sha384);
sha2_streaming!(sha512_streaming, "sha512/streaming", rscrypto::Sha512, sha2::Sha512);
sha2_streaming!(
  sha512_256_streaming,
  "sha512-256/streaming",
  rscrypto::Sha512_256,
  sha2::Sha512_256
);

fn sha256_internal(_c: &mut Criterion) {
  if !bench_config::selected("sha256/internal/") {
    return;
  }
  print_sha2_diag_once();

  #[cfg(all(rscrypto_internal, feature = "diag"))]
  {
    use bench_config::checked_with;

    let blocks = common::random_bytes(64 * 16);
    let mut g = _c.benchmark_group("sha256/internal/compress");

    for block_count in [1usize, 2, 16] {
      let len = block_count.strict_mul(64);
      common::set_throughput(&mut g, len);
      let blocks = &blocks[..len];
      let initial = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
      ];
      g.bench_function(
        BenchmarkId::new("selected-kernel", format!("{len}B")),
        checked_with(
          initial,
          |state| {
            rscrypto::hashes::introspect::sha256_compress_blocks_for_bench(black_box(&mut *state), black_box(blocks));
            state[0]
          },
          |_, state| {
            // The check sees the state after two calls; sha2 compresses the same blocks twice.
            let mut expected = initial;
            sha2::block_api::compress256(&mut expected, blocks.as_chunks::<64>().0);
            sha2::block_api::compress256(&mut expected, blocks.as_chunks::<64>().0);
            assert_eq!(*state, expected);
          },
        ),
      );
    }

    g.finish();
  }
}

fn main() {
  bench_config::run(&[
    sha224,
    sha256,
    sha384,
    sha512,
    sha512_256,
    sha224_streaming,
    sha256_streaming,
    sha384_streaming,
    sha512_streaming,
    sha512_256_streaming,
    sha256_internal,
  ]);
}
