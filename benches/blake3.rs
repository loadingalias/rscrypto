//! Blake3 comparison benchmarks: rscrypto vs official blake3 crate.

mod common;

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
#[cfg(feature = "diag")]
use rscrypto::hashes::crypto::blake3::{
  Blake3DiagKernel, diag_blake3_chunk_cvs_with_kernel, diag_blake3_digest_with_kernel, diag_blake3_kernel_available,
  diag_blake3_keyed_digest_with_kernel, diag_blake3_parent_cvs_with_kernel, diag_blake3_streaming_digest_with_kernel,
  diag_blake3_xof_with_kernel,
};

#[cfg(feature = "diag")]
const BLAKE3_CHUNK_LEN: usize = 1024;
#[cfg(feature = "diag")]
const BLAKE3_OUT_LEN: usize = 32;

#[cfg(feature = "diag")]
fn diag_kernels() -> &'static [Blake3DiagKernel] {
  &[
    Blake3DiagKernel::Portable,
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Sse41,
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Avx2,
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Avx2OwnedHashMany,
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Avx512,
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Avx512ExactBlockAsm,
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Avx512OwnedHashMany,
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Avx512OwnedCompress,
    #[cfg(target_arch = "aarch64")]
    Blake3DiagKernel::Aarch64Neon,
  ]
}

#[cfg(feature = "diag")]
fn chunk_tail_diag_kernels() -> &'static [Blake3DiagKernel] {
  &[
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Sse41,
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Avx2,
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Avx2OwnedHashMany,
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Avx2PairChunkTail,
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Avx512,
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Avx512OwnedHashMany,
    #[cfg(target_arch = "aarch64")]
    Blake3DiagKernel::Aarch64Neon,
  ]
}

#[cfg(feature = "diag")]
fn parent_tail_diag_kernels() -> &'static [Blake3DiagKernel] {
  &[
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Sse41,
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Avx2,
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Avx2OwnedHashMany,
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Avx2PairParentTail,
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Avx512,
    #[cfg(target_arch = "x86_64")]
    Blake3DiagKernel::X86Avx512OwnedHashMany,
    #[cfg(target_arch = "aarch64")]
    Blake3DiagKernel::Aarch64Neon,
  ]
}

#[cfg(feature = "diag")]
fn print_blake3_diag_once() {
  use std::sync::Once;

  static ONCE: Once = Once::new();
  ONCE.call_once(|| {
    eprintln!("rscrypto-diag blake3 runtime_caps={}", rscrypto::platform::caps());
    eprintln!("rscrypto-diag blake3 static_caps={}", rscrypto::platform::caps_static());
    eprintln!(
      "rscrypto-diag blake3 target_features sse4.1={} avx2={} avx512f={} avx512vl={} neon={}",
      cfg!(target_feature = "sse4.1"),
      cfg!(target_feature = "avx2"),
      cfg!(target_feature = "avx512f"),
      cfg!(target_feature = "avx512vl"),
      cfg!(target_feature = "neon")
    );
    for &kernel in diag_kernels() {
      eprintln!(
        "rscrypto-diag blake3 kernel {} available={} streaming={}",
        kernel.label(),
        diag_blake3_kernel_available(kernel),
        kernel.supports_streaming()
      );
    }
  });
}

#[cfg(not(feature = "diag"))]
#[inline]
fn print_blake3_diag_once() {}

fn oneshot(c: &mut Criterion) {
  print_blake3_diag_once();

  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("blake3");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::Blake3::digest(black_box(d))))
    });

    #[cfg(feature = "diag")]
    for &kernel in diag_kernels() {
      if !diag_blake3_kernel_available(kernel) {
        continue;
      }
      g.bench_with_input(
        BenchmarkId::new(format!("rscrypto-{}", kernel.label()), len),
        data,
        |b, d| b.iter(|| black_box(diag_blake3_digest_with_kernel(kernel, black_box(d)).unwrap())),
      );
    }

    g.bench_with_input(BenchmarkId::new("blake3", len), data, |b, d| {
      b.iter(|| black_box(blake3::hash(black_box(d))))
    });
  }

  g.finish();
}

fn keyed(c: &mut Criterion) {
  print_blake3_diag_once();

  let inputs = common::comp_sizes();
  let key = [0x42u8; 32];
  let mut g = c.benchmark_group("blake3/keyed");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::Blake3::keyed_digest(black_box(&key), black_box(d))))
    });

    #[cfg(feature = "diag")]
    for &kernel in diag_kernels() {
      if !diag_blake3_kernel_available(kernel) {
        continue;
      }
      g.bench_with_input(
        BenchmarkId::new(format!("rscrypto-{}", kernel.label()), len),
        data,
        |b, d| {
          b.iter(|| black_box(diag_blake3_keyed_digest_with_kernel(kernel, black_box(&key), black_box(d)).unwrap()))
        },
      );
    }

    g.bench_with_input(BenchmarkId::new("blake3", len), data, |b, d| {
      b.iter(|| black_box(blake3::keyed_hash(black_box(&key), black_box(d))))
    });
  }

  g.finish();
}

fn derive_key(c: &mut Criterion) {
  print_blake3_diag_once();

  const CONTEXT: &str = "rscrypto benchmark derive-key context";

  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("blake3/derive-key");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| black_box(rscrypto::Blake3::derive_key(black_box(CONTEXT), black_box(d))))
    });

    g.bench_with_input(BenchmarkId::new("blake3", len), data, |b, d| {
      b.iter(|| black_box(blake3::derive_key(black_box(CONTEXT), black_box(d))))
    });
  }

  g.finish();
}

fn streaming(c: &mut Criterion) {
  print_blake3_diag_once();

  let data = common::random_bytes(1048576);
  let mut g = c.benchmark_group("blake3/streaming");
  g.throughput(criterion::Throughput::Bytes(data.len() as u64));

  for chunk_size in [64, 4096, 16384, 65536] {
    g.bench_function(format!("rscrypto/{chunk_size}B"), |b| {
      b.iter(|| {
        use rscrypto::Digest;
        let mut h = rscrypto::Blake3::new();
        for chunk in data.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        black_box(h.finalize())
      })
    });

    #[cfg(feature = "diag")]
    for &kernel in diag_kernels() {
      if !diag_blake3_kernel_available(kernel) || !kernel.supports_streaming() {
        continue;
      }
      g.bench_function(format!("rscrypto-{}/{chunk_size}B", kernel.label()), |b| {
        b.iter(|| black_box(diag_blake3_streaming_digest_with_kernel(kernel, black_box(&data), chunk_size).unwrap()))
      });
    }

    g.bench_function(format!("blake3/{chunk_size}B"), |b| {
      b.iter(|| {
        let mut h = blake3::Hasher::new();
        for chunk in data.chunks(chunk_size) {
          h.update(black_box(chunk));
        }
        black_box(h.finalize())
      })
    });
  }

  g.finish();
}

fn xof(c: &mut Criterion) {
  print_blake3_diag_once();

  const OUT_LEN: usize = 64;

  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("blake3/xof");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        use rscrypto::Xof;

        let mut xof = rscrypto::Blake3::xof(black_box(d));
        let mut out = [0u8; OUT_LEN];
        xof.squeeze(&mut out);
        black_box(out)
      })
    });

    #[cfg(feature = "diag")]
    for &kernel in diag_kernels() {
      if !diag_blake3_kernel_available(kernel) {
        continue;
      }
      g.bench_with_input(
        BenchmarkId::new(format!("rscrypto-{}", kernel.label()), len),
        data,
        |b, d| {
          b.iter(|| {
            let mut out = [0u8; OUT_LEN];
            diag_blake3_xof_with_kernel(kernel, black_box(d), &mut out).unwrap();
            black_box(out)
          })
        },
      );
    }

    g.bench_with_input(BenchmarkId::new("blake3", len), data, |b, d| {
      b.iter(|| {
        let mut hasher = blake3::Hasher::new();
        hasher.update(black_box(d));
        let mut reader = hasher.finalize_xof();
        let mut out = [0u8; OUT_LEN];
        reader.fill(&mut out);
        black_box(out)
      })
    });
  }

  g.finish();
}

#[cfg(feature = "diag")]
fn xof_output(c: &mut Criterion) {
  use rscrypto::Xof;

  print_blake3_diag_once();

  const INPUT_LEN: usize = 4096;
  const OUTPUT_LENGTHS: [usize; 6] = [64, 128, 256, 512, 1024, 4096];

  let data = common::random_bytes(INPUT_LEN);
  let mut g = c.benchmark_group("blake3/xof-output");

  for out_len in OUTPUT_LENGTHS {
    common::set_throughput(&mut g, out_len);

    g.bench_with_input(BenchmarkId::new("rscrypto", out_len), &out_len, |b, &len| {
      let mut out = vec![0u8; len];
      b.iter(|| {
        let mut xof = rscrypto::Blake3::xof(black_box(&data));
        xof.squeeze(black_box(out.as_mut_slice()));
        black_box(out[0])
      })
    });

    for &kernel in diag_kernels() {
      if !diag_blake3_kernel_available(kernel) {
        continue;
      }
      g.bench_with_input(
        BenchmarkId::new(format!("rscrypto-{}", kernel.label()), out_len),
        &out_len,
        |b, &len| {
          let mut out = vec![0u8; len];
          b.iter(|| {
            diag_blake3_xof_with_kernel(kernel, black_box(&data), black_box(out.as_mut_slice())).unwrap();
            black_box(out[0])
          })
        },
      );
    }

    g.bench_with_input(BenchmarkId::new("blake3", out_len), &out_len, |b, &len| {
      let mut out = vec![0u8; len];
      b.iter(|| {
        let mut hasher = blake3::Hasher::new();
        hasher.update(black_box(&data));
        let mut reader = hasher.finalize_xof();
        reader.fill(black_box(out.as_mut_slice()));
        black_box(out[0])
      })
    });
  }

  g.finish();
}

#[cfg(not(feature = "diag"))]
fn xof_output(_c: &mut Criterion) {}

#[cfg(feature = "diag")]
fn tail_diagnostics(c: &mut Criterion) {
  print_blake3_diag_once();

  let tail_counts = [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

  let mut digest_group = c.benchmark_group("blake3/chunk-tail-digest");
  for chunks in tail_counts.iter().copied() {
    let data = common::random_bytes(chunks * BLAKE3_CHUNK_LEN);
    common::set_throughput(&mut digest_group, data.len());

    digest_group.bench_with_input(BenchmarkId::new("rscrypto", chunks), &data, |b, d| {
      b.iter(|| black_box(rscrypto::Blake3::digest(black_box(d))))
    });

    for &kernel in chunk_tail_diag_kernels() {
      if !diag_blake3_kernel_available(kernel) {
        continue;
      }
      digest_group.bench_with_input(
        BenchmarkId::new(format!("rscrypto-{}", kernel.label()), chunks),
        &data,
        |b, d| b.iter(|| black_box(diag_blake3_digest_with_kernel(kernel, black_box(d)).unwrap())),
      );
    }

    digest_group.bench_with_input(BenchmarkId::new("blake3", chunks), &data, |b, d| {
      b.iter(|| black_box(blake3::hash(black_box(d))))
    });
  }
  digest_group.finish();

  let mut chunk_group = c.benchmark_group("blake3/chunk-tail-cvs");
  for chunks in tail_counts.iter().copied() {
    let data = common::random_bytes(chunks * BLAKE3_CHUNK_LEN);
    common::set_throughput(&mut chunk_group, data.len());

    for &kernel in chunk_tail_diag_kernels() {
      if !diag_blake3_kernel_available(kernel) {
        continue;
      }
      let mut out = vec![0u8; chunks * BLAKE3_OUT_LEN];
      chunk_group.bench_with_input(
        BenchmarkId::new(format!("rscrypto-{}", kernel.label()), chunks),
        &data,
        |b, d| {
          b.iter(|| {
            diag_blake3_chunk_cvs_with_kernel(kernel, black_box(d), black_box(out.as_mut_slice())).unwrap();
            black_box(out[0])
          })
        },
      );
    }
  }
  chunk_group.finish();

  let mut parent_group = c.benchmark_group("blake3/parent-tail-cvs");
  for parents in tail_counts {
    let children = common::random_bytes(parents * 2 * BLAKE3_OUT_LEN);
    common::set_throughput(&mut parent_group, children.len());

    for &kernel in parent_tail_diag_kernels() {
      if !diag_blake3_kernel_available(kernel) {
        continue;
      }
      let mut out = vec![0u8; parents * BLAKE3_OUT_LEN];
      parent_group.bench_with_input(
        BenchmarkId::new(format!("rscrypto-{}", kernel.label()), parents),
        &children,
        |b, d| {
          b.iter(|| {
            diag_blake3_parent_cvs_with_kernel(kernel, black_box(d), black_box(out.as_mut_slice())).unwrap();
            black_box(out[0])
          })
        },
      );
    }
  }
  parent_group.finish();
}

#[cfg(not(feature = "diag"))]
fn tail_diagnostics(_c: &mut Criterion) {}

criterion_group!(
  benches,
  oneshot,
  keyed,
  derive_key,
  streaming,
  xof,
  xof_output,
  tail_diagnostics
);
criterion_main!(benches);
