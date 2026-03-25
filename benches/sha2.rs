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

/// Benchmark the raw SHA-512 compress_blocks function in isolation — no padding,
/// no finalization, no dispatch overhead beyond the initial kernel selection.
/// Compares rscrypto's per-kernel compression against the `sha2` crate's
/// `compress512` to isolate whether the SHA-512 gap is in compression or the
/// sponge/padding wrapper.
fn sha512_compress_raw(c: &mut Criterion) {
  const BLOCK_LEN: usize = 128;
  const NUM_BLOCKS: usize = 8;
  const DATA_LEN: usize = BLOCK_LEN * NUM_BLOCKS;

  let data = common::random_bytes(DATA_LEN);
  let mut g = c.benchmark_group("sha512-compress/raw");
  g.throughput(criterion::Throughput::Bytes(DATA_LEN as u64));

  let seed = u64::from_le_bytes(data[..8].try_into().unwrap());

  // rscrypto: dispatched (what the production SHA-512 path actually uses).
  g.bench_function("rscrypto/auto", |b| {
    b.iter(|| black_box(rscrypto::hashes::bench::run_auto("sha512-compress", black_box(&data))))
  });

  // rscrypto: portable kernel specifically.
  if let Some(k) = rscrypto::hashes::bench::get_kernel("sha512-compress", "portable") {
    g.bench_function("rscrypto/portable", |b| {
      b.iter(|| black_box((k.func)(black_box(&data))))
    });
  }

  // rscrypto: AVX2 kernel (x86_64 only).
  if let Some(k) = rscrypto::hashes::bench::get_kernel("sha512-compress", "x86_64/avx2") {
    g.bench_function("rscrypto/x86-avx2", |b| {
      b.iter(|| black_box((k.func)(black_box(&data))))
    });
  }

  // rscrypto: AVX-512VL kernel (x86_64 only).
  if let Some(k) = rscrypto::hashes::bench::get_kernel("sha512-compress", "x86_64/avx512vl") {
    g.bench_function("rscrypto/x86-avx512vl", |b| {
      b.iter(|| black_box((k.func)(black_box(&data))))
    });
  }

  // rscrypto: AVX2 standard-round kernel (non-deferred Σ0, x86_64 only).
  if let Some(k) = rscrypto::hashes::bench::get_kernel("sha512-compress", "x86_64/avx2-std") {
    g.bench_function("rscrypto/x86-avx2-std", |b| {
      b.iter(|| black_box((k.func)(black_box(&data))))
    });
  }

  // rscrypto: AVX-512VL standard-round kernel (non-deferred Σ0, x86_64 only).
  if let Some(k) = rscrypto::hashes::bench::get_kernel("sha512-compress", "x86_64/avx512vl-std") {
    g.bench_function("rscrypto/x86-avx512vl-std", |b| {
      b.iter(|| black_box((k.func)(black_box(&data))))
    });
  }

  // sha2 crate: compress512 — raw compression, apples-to-apples.
  g.bench_function("sha2-crate/compress512", |b| {
    // SAFETY: `data` has exactly DATA_LEN = BLOCK_LEN * NUM_BLOCKS bytes, so
    // from_raw_parts produces a valid &[[u8; BLOCK_LEN]] of NUM_BLOCKS elements.
    let blocks: &[[u8; BLOCK_LEN]] = unsafe { core::slice::from_raw_parts(data.as_ptr().cast(), NUM_BLOCKS) };
    // SAFETY: GenericArray<u8, U128> is #[repr(transparent)] over [u8; 128],
    // so the slice layout is identical.
    let ga_blocks: &[sha2::digest::generic_array::GenericArray<u8, _>] = unsafe {
      &*(blocks as *const [[u8; BLOCK_LEN]]
        as *const [sha2::digest::generic_array::GenericArray<u8, sha2::digest::consts::U128>])
    };
    b.iter(|| {
      let mut state = [seed; 8];
      sha2::compress512(&mut state, black_box(ga_blocks));
      black_box(state)
    })
  });

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
  sha512_compress_raw
);
criterion_main!(benches);
