//! SHA-2 family comparison benchmarks: rscrypto vs sha2 crate.

mod common;

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

#[cfg(feature = "diag")]
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

#[cfg(not(feature = "diag"))]
#[inline]
fn print_sha2_diag_once() {}

macro_rules! sha2_oneshot {
  ($fn_name:ident, $group:literal, $ours:ty, $theirs:ty) => {
    fn $fn_name(c: &mut Criterion) {
      print_sha2_diag_once();

      let inputs = common::comp_sizes();
      let mut g = c.benchmark_group($group);

      for (len, data) in &inputs {
        common::set_throughput(&mut g, *len);

        g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
          b.iter(|| black_box(<$ours>::digest(black_box(d))))
        });

        g.bench_with_input(BenchmarkId::new("sha2", len), data, |b, d| {
          b.iter(|| {
            use sha2::Digest as _;
            black_box(<$theirs>::digest(black_box(d)))
          })
        });
      }

      g.finish();
    }
  };
}

/// Variant of `sha2_oneshot!` that also benches `aws-lc-rs` and `ring` — both
/// expose SHA-256/384/512 but neither ships SHA-224 or SHA-512/256.
macro_rules! sha2_oneshot_ietf {
  ($fn_name:ident, $group:literal, $ours:ty, $theirs:ty, $aws_alg:expr, $ring_alg:expr) => {
    fn $fn_name(c: &mut Criterion) {
      print_sha2_diag_once();

      let inputs = common::comp_sizes();
      let mut g = c.benchmark_group($group);

      for (len, data) in &inputs {
        common::set_throughput(&mut g, *len);

        g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
          b.iter(|| black_box(<$ours>::digest(black_box(d))))
        });

        g.bench_with_input(BenchmarkId::new("sha2", len), data, |b, d| {
          b.iter(|| {
            use sha2::Digest as _;
            black_box(<$theirs>::digest(black_box(d)))
          })
        });

        g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
          b.iter(|| black_box(aws_lc_rs::digest::digest($aws_alg, black_box(d))))
        });

        g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
          b.iter(|| black_box(ring::digest::digest($ring_alg, black_box(d))))
        });
      }

      g.finish();
    }
  };
}

sha2_oneshot!(sha224, "sha224", rscrypto::Sha224, sha2::Sha224);
sha2_oneshot_ietf!(
  sha256,
  "sha256",
  rscrypto::Sha256,
  sha2::Sha256,
  &aws_lc_rs::digest::SHA256,
  &ring::digest::SHA256
);
sha2_oneshot_ietf!(
  sha384,
  "sha384",
  rscrypto::Sha384,
  sha2::Sha384,
  &aws_lc_rs::digest::SHA384,
  &ring::digest::SHA384
);
sha2_oneshot_ietf!(
  sha512,
  "sha512",
  rscrypto::Sha512,
  sha2::Sha512,
  &aws_lc_rs::digest::SHA512,
  &ring::digest::SHA512
);
sha2_oneshot!(sha512_256, "sha512-256", rscrypto::Sha512_256, sha2::Sha512_256);

macro_rules! sha2_streaming {
  ($fn_name:ident, $group:literal, $ours:ty, $theirs:ty) => {
    fn $fn_name(c: &mut Criterion) {
      print_sha2_diag_once();

      let data = common::random_bytes(1048576);
      let mut g = c.benchmark_group($group);
      g.throughput(criterion::Throughput::Bytes(data.len() as u64));

      for chunk_size in [64, 4096] {
        g.bench_function(format!("rscrypto/{chunk_size}B"), |b| {
          b.iter(|| {
            use rscrypto::Digest;
            let mut h = <$ours>::new();
            for chunk in data.chunks(chunk_size) {
              h.update(black_box(chunk));
            }
            black_box(h.finalize())
          })
        });

        g.bench_function(format!("sha2/{chunk_size}B"), |b| {
          b.iter(|| {
            use sha2::Digest;
            let mut h = <$theirs>::new();
            for chunk in data.chunks(chunk_size) {
              h.update(black_box(chunk));
            }
            black_box(h.finalize())
          })
        });
      }

      g.finish();
    }
  };
}

sha2_streaming!(sha256_streaming, "sha256/streaming", rscrypto::Sha256, sha2::Sha256);
sha2_streaming!(sha512_streaming, "sha512/streaming", rscrypto::Sha512, sha2::Sha512);

fn sha256_internal(c: &mut Criterion) {
  print_sha2_diag_once();

  #[cfg(feature = "diag")]
  {
    let blocks = common::random_bytes(64 * 16);
    let mut g = c.benchmark_group("sha256/internal/compress");

    for block_count in [1usize, 2, 16] {
      let len = block_count * 64;
      common::set_throughput(&mut g, len);
      g.bench_with_input(BenchmarkId::new("selected-kernel", format!("{len}B")), &len, |b, &n| {
        let blocks = &blocks[..n];
        let mut state = [
          0x6a09e667u32,
          0xbb67ae85,
          0x3c6ef372,
          0xa54ff53a,
          0x510e527f,
          0x9b05688c,
          0x1f83d9ab,
          0x5be0cd19,
        ];
        b.iter(|| {
          rscrypto::hashes::introspect::sha256_compress_blocks_for_bench(black_box(&mut state), black_box(blocks));
          black_box(state[0])
        })
      });
    }

    g.finish();
  }

  let data = common::random_bytes(1_048_576);
  let mut g = c.benchmark_group("sha256/internal/public-overhead");

  for chunk_size in [64usize, 4096] {
    g.throughput(criterion::Throughput::Bytes(data.len() as u64));

    g.bench_function(format!("rscrypto-stream-{chunk_size}B"), |b| {
      b.iter(|| {
        use rscrypto::Digest;
        let mut h = rscrypto::Sha256::new();
        for chunk in data.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        black_box(h.finalize())
      })
    });

    g.bench_function(format!("sha2-stream-{chunk_size}B"), |b| {
      b.iter(|| {
        use sha2::Digest;
        let mut h = sha2::Sha256::new();
        for chunk in data.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        black_box(h.finalize())
      })
    });
  }

  for len in [64usize, 4096, 1_048_576] {
    let input = &data[..len];
    common::set_throughput(&mut g, len);

    g.bench_with_input(BenchmarkId::new("rscrypto-digest", len), input, |b, d| {
      b.iter(|| black_box(rscrypto::Sha256::digest(black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("sha2-digest", len), input, |b, d| {
      b.iter(|| {
        use sha2::Digest as _;
        black_box(sha2::Sha256::digest(black_box(d)))
      })
    });
  }

  g.finish();
}

criterion_group!(
  benches,
  sha224,
  sha256,
  sha384,
  sha512,
  sha512_256,
  sha256_streaming,
  sha512_streaming,
  sha256_internal
);
criterion_main!(benches);
