//! SHA-2 family comparison benchmarks: rscrypto vs sha2 crate.

mod common;

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

macro_rules! sha2_oneshot {
  ($fn_name:ident, $group:literal, $ours:ty, $theirs:ty) => {
    fn $fn_name(c: &mut Criterion) {
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

sha2_oneshot!(sha224, "sha224", rscrypto::Sha224, sha2::Sha224);
sha2_oneshot!(sha256, "sha256", rscrypto::Sha256, sha2::Sha256);
sha2_oneshot!(sha384, "sha384", rscrypto::Sha384, sha2::Sha384);
sha2_oneshot!(sha512, "sha512", rscrypto::Sha512, sha2::Sha512);
sha2_oneshot!(sha512_256, "sha512-256", rscrypto::Sha512_256, sha2::Sha512_256);

macro_rules! sha2_streaming {
  ($fn_name:ident, $group:literal, $ours:ty, $theirs:ty) => {
    fn $fn_name(c: &mut Criterion) {
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

criterion_group!(
  benches,
  sha224,
  sha256,
  sha384,
  sha512,
  sha512_256,
  sha256_streaming,
  sha512_streaming
);
criterion_main!(benches);
