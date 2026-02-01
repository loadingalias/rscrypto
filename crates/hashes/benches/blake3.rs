//! BLAKE3 dedicated benchmarks.
//!
//! This benchmark file focuses exclusively on BLAKE3 to enable detailed
//! performance analysis and comparison with the official blake3 crate.

use core::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main};
use hashes::{bench as microbench, crypto::Blake3};
use traits::{Digest as _, Xof as _};

mod common;

#[inline]
fn official_hash_bytes(input: &[u8]) -> [u8; 32] {
  *blake3::hash(input).as_bytes()
}

// ─────────────────────────────────────────────────────────────────────────────
// One-shot Comparison Benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn blake3_oneshot_comparison(c: &mut Criterion) {
  let inputs = common::sized_inputs();
  let mut group = c.benchmark_group("blake3/oneshot");
  group.sample_size(40);
  group.warm_up_time(Duration::from_secs(2));
  group.measurement_time(Duration::from_secs(4));
  group.sampling_mode(SamplingMode::Flat);

  for (len, data) in &inputs {
    common::set_throughput(&mut group, *len);

    group.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Blake3::digest(black_box(d))))
    });

    group.bench_with_input(BenchmarkId::new("official", len), data, |b, d| {
      b.iter(|| black_box(official_hash_bytes(black_box(d))))
    });
  }

  group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming Benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn blake3_streaming(c: &mut Criterion) {
  let data_1mb = common::pseudo_random_bytes(1024 * 1024, 0xB1AE_E3B1_A1E3_0001);
  let data_1mb = black_box(data_1mb);

  let mut group = c.benchmark_group("blake3/streaming");
  group.sample_size(30);
  group.warm_up_time(Duration::from_secs(2));
  group.measurement_time(Duration::from_secs(4));
  group.sampling_mode(SamplingMode::Flat);
  group.throughput(Throughput::Bytes(data_1mb.len() as u64));

  // Test various chunk sizes to understand streaming overhead
  for chunk_size in [64, 128, 256, 512, 1024, 4096, 16384, 65536] {
    group.bench_function(format!("rscrypto/{chunk_size}B-chunks"), |b| {
      b.iter(|| {
        let mut h = Blake3::new();
        for chunk in data_1mb.chunks(chunk_size) {
          h.update(chunk);
        }
        black_box(h.finalize())
      })
    });

    group.bench_function(format!("official/{chunk_size}B-chunks"), |b| {
      b.iter(|| {
        let mut h = blake3::Hasher::new();
        for chunk in data_1mb.chunks(chunk_size) {
          h.update(chunk);
        }
        black_box(*h.finalize().as_bytes())
      })
    });
  }

  group.finish();
}

fn blake3_update_overhead(c: &mut Criterion) {
  let data_1mb = common::pseudo_random_bytes(1024 * 1024, 0xB1AE_E3B1_A1E3_0002);
  let data_1mb = black_box(data_1mb);

  let mut group = c.benchmark_group("blake3/update-overhead");
  group.sample_size(30);
  group.warm_up_time(Duration::from_secs(2));
  group.measurement_time(Duration::from_secs(4));
  group.sampling_mode(SamplingMode::Flat);
  group.throughput(Throughput::Bytes(data_1mb.len() as u64));

  for chunk_size in [64, 128, 256, 512, 1024, 4096, 16384, 65536] {
    group.bench_function(format!("{chunk_size}B-chunks"), |b| {
      b.iter(|| {
        let mut h = Blake3::new();
        for chunk in data_1mb.chunks(chunk_size) {
          h.update(chunk);
        }
        black_box(h.finalize())
      })
    });
  }

  group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Parent Folding (CVs -> Root) Benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn blake3_parent_folding_only(c: &mut Criterion) {
  // 1 MiB worth of chaining values (CVs): 32 bytes each -> 32768 CVs.
  let data = common::pseudo_random_bytes(1024 * 1024, 0xB1AE_E3B1_A1E3_7001);
  let data = black_box(data);

  let mut group = c.benchmark_group("blake3/parent-folding");
  group.sample_size(30);
  group.warm_up_time(Duration::from_secs(2));
  group.measurement_time(Duration::from_secs(4));
  group.sampling_mode(SamplingMode::Flat);
  group.throughput(Throughput::Bytes(data.len() as u64));

  group.bench_function("rscrypto/auto", |b| {
    b.iter(|| black_box(microbench::run_auto("blake3-parent-fold", black_box(&data)).unwrap_or(0)))
  });

  for name in [
    "portable",
    #[cfg(target_arch = "x86_64")]
    "x86_64/ssse3",
    #[cfg(target_arch = "x86_64")]
    "x86_64/sse4.1",
    #[cfg(target_arch = "x86_64")]
    "x86_64/avx2",
    #[cfg(target_arch = "x86_64")]
    "x86_64/avx512",
    #[cfg(target_arch = "aarch64")]
    "aarch64/neon",
  ] {
    let Some(k) = microbench::get_kernel("blake3-parent-fold", name) else {
      continue;
    };
    group.bench_function(format!("rscrypto/{}", k.name), |b| {
      b.iter(|| black_box((k.func)(black_box(&data))))
    });
  }

  group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// XOF (Extendable Output) Benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn blake3_xof(c: &mut Criterion) {
  let inputs = common::sized_inputs();
  let mut group = c.benchmark_group("blake3/xof");
  group.sample_size(20);
  group.warm_up_time(Duration::from_secs(1));
  group.measurement_time(Duration::from_secs(3));
  group.sampling_mode(SamplingMode::Flat);

  // Keep the default XOF matrix small; it otherwise dominates total runtime.
  let extended = std::env::var("RSCRYPTO_BLAKE3_BENCH_EXTENDED").is_ok();
  let read_only = extended || std::env::var("RSCRYPTO_BLAKE3_BENCH_READ_ONLY").is_ok();
  let output_sizes: &[usize] = if extended {
    &[32, 64, 128, 256, 512, 1024]
  } else {
    &[32, 1024]
  };
  let input_sizes: &[usize] = if extended { &[] } else { &[1, 64, 1024, 64 * 1024] };

  // Test XOF with various output sizes.
  for &output_size in output_sizes {
    for (len, data) in &inputs {
      if *len == 0 || *len > 64 * 1024 {
        continue; // skip 0B and very large inputs here
      }
      if !extended && !input_sizes.contains(len) {
        continue;
      }

      let name = format!("{len}B-in/{output_size}B-out");
      group.throughput(Throughput::Bytes((*len + output_size) as u64));

      // Mode A: hash + finalize_xof + squeeze
      group.bench_function(format!("rscrypto/init+read/{name}"), |b| {
        let mut out = vec![0u8; output_size];
        b.iter(|| {
          let mut h = Blake3::new();
          h.update(black_box(data));
          let mut xof = h.finalize_xof();
          xof.squeeze(&mut out);
          black_box(&out);
        })
      });

      group.bench_function(format!("official/init+read/{name}"), |b| {
        let mut out = vec![0u8; output_size];
        b.iter(|| {
          let mut h = blake3::Hasher::new();
          h.update(black_box(data));
          let mut reader = h.finalize_xof();
          reader.fill(&mut out);
          black_box(&out);
        })
      });

      if read_only {
        // Mode B: squeeze-only (no hashing); helps attribute differences.
        // Both XOF readers are `Clone`, so we can reset position cheaply.
        group.bench_function(format!("rscrypto/read-only/{name}"), |b| {
          let mut out = vec![0u8; output_size];
          let mut h = Blake3::new();
          h.update(data);
          let base = h.finalize_xof();
          b.iter(|| {
            let mut xof = base.clone();
            xof.squeeze(&mut out);
            black_box(&out);
          })
        });

        group.bench_function(format!("official/read-only/{name}"), |b| {
          let mut out = vec![0u8; output_size];
          let mut h = blake3::Hasher::new();
          h.update(data);
          let base = h.finalize_xof();
          b.iter(|| {
            let mut reader = base.clone();
            reader.fill(&mut out);
            black_box(&out);
          })
        });
      }
    }
  }

  group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// One-shot XOF Benchmarks (avoid streaming hasher setup)
// ─────────────────────────────────────────────────────────────────────────────

fn blake3_xof_oneshot(c: &mut Criterion) {
  if std::env::var("RSCRYPTO_BLAKE3_BENCH_EXTENDED").is_err() {
    return;
  }

  let inputs = common::sized_inputs();
  let mut group = c.benchmark_group("blake3/xof-oneshot");
  group.sample_size(25);
  group.warm_up_time(Duration::from_secs(1));
  group.measurement_time(Duration::from_secs(3));
  group.sampling_mode(SamplingMode::Flat);

  let output_sizes: &[usize] = &[32, 1024];
  for &output_size in output_sizes {
    for (len, data) in &inputs {
      if *len == 0 || *len > 64 * 1024 {
        continue;
      }

      let name = format!("{len}B-in/{output_size}B-out");
      group.throughput(Throughput::Bytes((*len + output_size) as u64));

      group.bench_function(format!("rscrypto/{name}"), |b| {
        let mut out = vec![0u8; output_size];
        b.iter(|| {
          let mut xof = Blake3::xof(black_box(data));
          xof.squeeze(&mut out);
          black_box(&out);
        })
      });

      group.bench_function(format!("official/{name}"), |b| {
        let mut out = vec![0u8; output_size];
        b.iter(|| {
          let mut h = blake3::Hasher::new();
          h.update(black_box(data));
          let mut reader = h.finalize_xof();
          reader.fill(&mut out);
          black_box(&out);
        })
      });
    }
  }

  group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Keyed Hash Benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn blake3_keyed(c: &mut Criterion) {
  let inputs = common::sized_inputs();
  let key: [u8; 32] = *b"rscrypto-blake3-benchmark-key!!_";
  let mut group = c.benchmark_group("blake3/keyed");
  group.sample_size(25);
  group.warm_up_time(Duration::from_secs(1));
  group.measurement_time(Duration::from_secs(3));
  group.sampling_mode(SamplingMode::Flat);

  for (len, data) in &inputs {
    common::set_throughput(&mut group, *len);

    group.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        let mut h = Blake3::new_keyed(&key);
        h.update(black_box(d));
        black_box(h.finalize())
      })
    });

    group.bench_with_input(BenchmarkId::new("official", len), data, |b, d| {
      b.iter(|| black_box(*blake3::keyed_hash(&key, black_box(d)).as_bytes()))
    });
  }

  group.finish();
}

fn blake3_keyed_oneshot(c: &mut Criterion) {
  if std::env::var("RSCRYPTO_BLAKE3_BENCH_EXTENDED").is_err() {
    return;
  }

  let inputs = common::sized_inputs();
  let key: [u8; 32] = *b"rscrypto-blake3-benchmark-key!!_";
  let mut group = c.benchmark_group("blake3/keyed-oneshot");
  group.sample_size(40);
  group.warm_up_time(Duration::from_secs(2));
  group.measurement_time(Duration::from_secs(4));
  group.sampling_mode(SamplingMode::Flat);

  for (len, data) in &inputs {
    common::set_throughput(&mut group, *len);

    group.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Blake3::keyed_digest(&key, black_box(d))))
    });

    group.bench_with_input(BenchmarkId::new("official", len), data, |b, d| {
      b.iter(|| black_box(*blake3::keyed_hash(&key, black_box(d)).as_bytes()))
    });
  }

  group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Derive Key Benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn blake3_derive_key(c: &mut Criterion) {
  let inputs = common::sized_inputs();
  let context = "rscrypto benchmark 2024-01-01 derive key context";
  let mut group = c.benchmark_group("blake3/derive-key");
  group.sample_size(25);
  group.warm_up_time(Duration::from_secs(1));
  group.measurement_time(Duration::from_secs(3));
  group.sampling_mode(SamplingMode::Flat);

  for (len, data) in &inputs {
    common::set_throughput(&mut group, *len);

    group.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        let mut h = Blake3::new_derive_key(context);
        h.update(black_box(d));
        black_box(h.finalize())
      })
    });

    group.bench_with_input(BenchmarkId::new("official", len), data, |b, d| {
      b.iter(|| {
        let mut h = blake3::Hasher::new_derive_key(context);
        h.update(black_box(d));
        black_box(*h.finalize().as_bytes())
      })
    });
  }

  group.finish();
}

fn blake3_derive_key_oneshot(c: &mut Criterion) {
  if std::env::var("RSCRYPTO_BLAKE3_BENCH_EXTENDED").is_err() {
    return;
  }

  let inputs = common::sized_inputs();
  let context = "rscrypto benchmark 2024-01-01 derive key context";
  let mut group = c.benchmark_group("blake3/derive-key-oneshot");
  group.sample_size(40);
  group.warm_up_time(Duration::from_secs(2));
  group.measurement_time(Duration::from_secs(4));
  group.sampling_mode(SamplingMode::Flat);

  for (len, data) in &inputs {
    common::set_throughput(&mut group, *len);

    group.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(Blake3::derive_key(context, black_box(d))))
    });

    group.bench_with_input(BenchmarkId::new("official", len), data, |b, d| {
      b.iter(|| black_box(blake3::derive_key(context, black_box(d))))
    });
  }

  group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Active Kernel Info (reports which kernel is being used)
// ─────────────────────────────────────────────────────────────────────────────

fn blake3_active_kernel(c: &mut Criterion) {
  use hashes::crypto::blake3::dispatch::kernel_name_for_len;

  let inputs = common::sized_inputs();
  let mut group = c.benchmark_group("blake3/active-kernel");
  group.sample_size(15);
  group.warm_up_time(Duration::from_secs(1));
  group.measurement_time(Duration::from_secs(2));
  group.sampling_mode(SamplingMode::Flat);

  // Report which kernel is selected for different sizes
  for (len, data) in &inputs {
    let kernel_name = kernel_name_for_len(*len);
    common::set_throughput(&mut group, *len);

    group.bench_with_input(BenchmarkId::new(kernel_name, len), data, |b, d| {
      b.iter(|| black_box(Blake3::digest(black_box(d))))
    });
  }

  group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Large Input Benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn blake3_large_inputs(c: &mut Criterion) {
  if std::env::var("RSCRYPTO_BLAKE3_BENCH_EXTENDED").is_err() {
    return;
  }

  let mut group = c.benchmark_group("blake3/large");
  group.sample_size(15);
  group.warm_up_time(Duration::from_secs(2));
  group.measurement_time(Duration::from_secs(4));
  group.sampling_mode(SamplingMode::Flat);

  // Test larger inputs where SIMD benefits become more apparent
  for size_mb in [1, 4, 16] {
    let size = size_mb * 1024 * 1024;
    let data = common::pseudo_random_bytes(size, 0x1A56_E1A9_0700 + size_mb as u64);
    let data = black_box(data);

    group.throughput(Throughput::Bytes(size as u64));

    group.bench_function(format!("rscrypto/{size_mb}MB"), |b| {
      b.iter(|| black_box(Blake3::digest(black_box(&data))))
    });

    group.bench_function(format!("official/{size_mb}MB"), |b| {
      b.iter(|| black_box(official_hash_bytes(black_box(&data))))
    });
  }

  group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Latency Benchmarks (small inputs, measure overhead)
// ─────────────────────────────────────────────────────────────────────────────

fn blake3_latency(c: &mut Criterion) {
  if std::env::var("RSCRYPTO_BLAKE3_BENCH_EXTENDED").is_err() {
    return;
  }

  let mut group = c.benchmark_group("blake3/latency");
  group.sample_size(50);
  group.warm_up_time(Duration::from_secs(1));
  group.measurement_time(Duration::from_secs(3));
  group.sampling_mode(SamplingMode::Flat);

  // Very small inputs to measure setup/teardown overhead
  for len in [0, 1, 8, 16, 32, 64] {
    let data = common::pseudo_random_bytes(len, 0x1A7E_AC90_0000 + len as u64);
    let data = black_box(data);

    group.throughput(Throughput::Elements(1));

    group.bench_function(format!("rscrypto/{len}B"), |b| {
      b.iter(|| black_box(Blake3::digest(black_box(&data))))
    });

    group.bench_function(format!("official/{len}B"), |b| {
      b.iter(|| black_box(official_hash_bytes(black_box(&data))))
    });
  }

  group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// XOF Sized Comparison Benchmark (Phase 1 Fix Verification)
// ─────────────────────────────────────────────────────────────────────────────

/// Benchmark to verify the XOF dispatch fix: compares `finalize_xof()` vs `finalize_xof_sized()`
/// on small inputs with large outputs. This is the critical case that showed +500% slowdown.
fn blake3_xof_sized_comparison(c: &mut Criterion) {
  let mut group = c.benchmark_group("blake3/xof-sized-comparison");
  group.sample_size(30);
  group.warm_up_time(Duration::from_secs(2));
  group.measurement_time(Duration::from_secs(4));
  group.sampling_mode(SamplingMode::Flat);

  // These are the critical cases from TASK.md showing +500% slowdown
  let input_sizes: &[usize] = &[1, 64];
  let output_sizes: &[usize] = &[512, 1024];

  for &input_len in input_sizes {
    for &output_size in output_sizes {
      let data = common::pseudo_random_bytes(input_len, 0xB1AE_E3B1_A1E3_0006 ^ (input_len as u64));
      let name = format!("{input_len}B-in/{output_size}B-out");
      group.throughput(Throughput::Bytes((input_len + output_size) as u64));

      // Baseline: standard finalize_xof() - may use portable kernel for small inputs
      group.bench_function(format!("standard/{name}"), |b| {
        let mut out = vec![0u8; output_size];
        b.iter(|| {
          let mut h = Blake3::new();
          h.update(black_box(&data));
          let mut xof = h.finalize_xof();
          xof.squeeze(&mut out);
          black_box(&out);
        })
      });

      // Fixed: finalize_xof_sized() - uses output size to select kernel
      group.bench_function(format!("sized/{name}"), |b| {
        let mut out = vec![0u8; output_size];
        b.iter(|| {
          let mut h = Blake3::new();
          h.update(black_box(&data));
          let mut xof = h.finalize_xof_sized(output_size);
          xof.squeeze(&mut out);
          black_box(&out);
        })
      });

      // Official implementation for comparison
      group.bench_function(format!("official/{name}"), |b| {
        let mut out = vec![0u8; output_size];
        b.iter(|| {
          let mut h = blake3::Hasher::new();
          h.update(black_box(&data));
          let mut reader = h.finalize_xof();
          reader.fill(&mut out);
          black_box(&out);
        })
      });
    }
  }

  group.finish();
}

criterion_group!(
  benches,
  blake3_oneshot_comparison,
  blake3_streaming,
  blake3_update_overhead,
  blake3_parent_folding_only,
  blake3_xof,
  blake3_xof_oneshot,
  blake3_xof_sized_comparison,
  blake3_keyed,
  blake3_keyed_oneshot,
  blake3_derive_key,
  blake3_derive_key_oneshot,
  blake3_active_kernel,
  blake3_large_inputs,
  blake3_latency,
);
criterion_main!(benches);
