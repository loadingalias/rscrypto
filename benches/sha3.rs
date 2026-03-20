//! SHA-3 family comparison benchmarks: rscrypto vs sha3 crate.

mod common;

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

macro_rules! sha3_oneshot {
  ($fn_name:ident, $group:literal, $ours:ty, $theirs:ty) => {
    fn $fn_name(c: &mut Criterion) {
      let inputs = common::comp_sizes();
      let mut g = c.benchmark_group($group);

      for (len, data) in &inputs {
        common::set_throughput(&mut g, *len);

        g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
          use rscrypto::Digest as _;
          b.iter(|| black_box(<$ours>::digest(black_box(d))))
        });

        g.bench_with_input(BenchmarkId::new("sha3", len), data, |b, d| {
          b.iter(|| {
            use sha3::Digest as _;
            black_box(<$theirs>::digest(black_box(d)))
          })
        });
      }

      g.finish();
    }
  };
}

sha3_oneshot!(sha3_224, "sha3-224", rscrypto::Sha3_224, sha3::Sha3_224);
sha3_oneshot!(sha3_256, "sha3-256", rscrypto::Sha3_256, sha3::Sha3_256);
sha3_oneshot!(sha3_384, "sha3-384", rscrypto::Sha3_384, sha3::Sha3_384);
sha3_oneshot!(sha3_512, "sha3-512", rscrypto::Sha3_512, sha3::Sha3_512);

fn sha3_256_streaming(c: &mut Criterion) {
  let data = common::random_bytes(1048576);
  let mut g = c.benchmark_group("sha3-256/streaming");
  g.throughput(criterion::Throughput::Bytes(data.len() as u64));

  for chunk_size in [64, 4096] {
    g.bench_function(format!("rscrypto/{chunk_size}B"), |b| {
      b.iter(|| {
        use rscrypto::Digest;
        let mut h = rscrypto::Sha3_256::new();
        for chunk in data.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        black_box(h.finalize())
      })
    });

    g.bench_function(format!("sha3/{chunk_size}B"), |b| {
      b.iter(|| {
        use sha3::Digest;
        let mut h = sha3::Sha3_256::new();
        for chunk in data.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        black_box(h.finalize())
      })
    });
  }

  g.finish();
}

fn shake128(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("shake128");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        use rscrypto::Xof;
        let mut h = rscrypto::Shake128::new();
        h.update(black_box(d));
        let mut xof = h.finalize_xof();
        let mut out = [0u8; 32];
        xof.squeeze(&mut out);
        black_box(out)
      })
    });

    g.bench_with_input(BenchmarkId::new("sha3", len), data, |b, d| {
      b.iter(|| {
        use sha3::digest::{ExtendableOutput, Update, XofReader};
        let mut hasher = sha3::Shake128::default();
        hasher.update(black_box(d));
        let mut reader = hasher.finalize_xof();
        let mut out = [0u8; 32];
        reader.read(&mut out);
        black_box(out)
      })
    });
  }

  g.finish();
}

fn shake256(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("shake256");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        use rscrypto::Xof;
        let mut h = rscrypto::Shake256::new();
        h.update(black_box(d));
        let mut xof = h.finalize_xof();
        let mut out = [0u8; 32];
        xof.squeeze(&mut out);
        black_box(out)
      })
    });

    g.bench_with_input(BenchmarkId::new("sha3", len), data, |b, d| {
      b.iter(|| {
        use sha3::digest::{ExtendableOutput, Update, XofReader};
        let mut hasher = sha3::Shake256::default();
        hasher.update(black_box(d));
        let mut reader = hasher.finalize_xof();
        let mut out = [0u8; 32];
        reader.read(&mut out);
        black_box(out)
      })
    });
  }

  g.finish();
}

criterion_group!(
  benches,
  sha3_224,
  sha3_256,
  sha3_384,
  sha3_512,
  sha3_256_streaming,
  shake128,
  shake256
);
criterion_main!(benches);
