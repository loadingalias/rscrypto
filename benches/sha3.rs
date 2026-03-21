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

// ---------------------------------------------------------------------------
// Diagnostic: raw permutation isolation (no sponge overhead)
// ---------------------------------------------------------------------------

/// Benchmark the raw Keccak-f[1600] permutation in isolation — no sponge
/// absorb, no padding, no zeroization on Drop. Compares rscrypto's portable
/// kernel against the `keccak` crate's `f1600` to isolate whether the SHA-3
/// gap is in the permutation or the sponge wrapper.
fn keccakf1600_raw(c: &mut Criterion) {
  let mut g = c.benchmark_group("keccakf1600/raw");
  g.throughput(criterion::Throughput::Bytes(200));

  let init_state: [u64; 25] = {
    let bytes = common::random_bytes(200);
    let mut s = [0u64; 25];
    for (i, chunk) in bytes.chunks_exact(8).enumerate() {
      s[i] = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    s
  };

  // rscrypto: dispatched (what the production SHA-3 path actually uses).
  g.bench_function("rscrypto/auto", |b| {
    let bytes: Vec<u8> = init_state.iter().flat_map(|w| w.to_le_bytes()).collect();
    b.iter(|| black_box(rscrypto::hashes::bench::run_auto("keccakf1600", black_box(&bytes))))
  });

  // rscrypto: portable kernel specifically.
  if let Some(k) = rscrypto::hashes::bench::get_kernel("keccakf1600", "portable") {
    g.bench_function("rscrypto/portable", |b| {
      let bytes: Vec<u8> = init_state.iter().flat_map(|w| w.to_le_bytes()).collect();
      b.iter(|| black_box((k.func)(black_box(&bytes))))
    });
  }

  // rscrypto: SHA3 CE kernel (aarch64 only).
  if let Some(k) = rscrypto::hashes::bench::get_kernel("keccakf1600", "aarch64/sha3") {
    g.bench_function("rscrypto/aarch64-sha3", |b| {
      let bytes: Vec<u8> = init_state.iter().flat_map(|w| w.to_le_bytes()).collect();
      b.iter(|| black_box((k.func)(black_box(&bytes))))
    });
  }

  // rscrypto: AVX-512 kernel (x86 only).
  if let Some(k) = rscrypto::hashes::bench::get_kernel("keccakf1600", "x86/avx512") {
    g.bench_function("rscrypto/x86-avx512", |b| {
      let bytes: Vec<u8> = init_state.iter().flat_map(|w| w.to_le_bytes()).collect();
      b.iter(|| black_box((k.func)(black_box(&bytes))))
    });
  }

  // keccak crate's f1600 (the competitor's raw permutation).
  g.bench_function("keccak-crate/f1600", |b| {
    let mut state = init_state;
    b.iter(|| {
      keccak::f1600(black_box(&mut state));
      black_box(&state);
    })
  });

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
  shake256,
  keccakf1600_raw
);
criterion_main!(benches);
