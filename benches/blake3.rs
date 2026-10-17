//! Blake3 comparison benchmarks: rscrypto vs official blake3 crate.

#[path = "common/criterion.rs"]
mod bench_config;

mod common;

#[path = "blake3/streaming.rs"]
mod streaming_shapes;

#[path = "blake3/leaf_alignment.rs"]
mod leaf_alignment;

#[path = "blake3/mixed.rs"]
mod mixed_batches;

use core::hint::black_box;

use bench_config::{checked, checked_with};
use criterion::{BenchmarkId, Criterion};
#[cfg(all(rscrypto_internal, feature = "diag"))]
use rscrypto::hashes::crypto::blake3::{
  Blake3DiagKernel, diag_blake3_chunk_cvs_with_kernel, diag_blake3_digest_with_kernel, diag_blake3_kernel_available,
  diag_blake3_keyed_digest_with_kernel, diag_blake3_parent_cvs_with_kernel, diag_blake3_streaming_digest_with_kernel,
  diag_blake3_xof_with_kernel,
};

#[cfg(all(rscrypto_internal, feature = "diag"))]
const BLAKE3_CHUNK_LEN: usize = 1024;
#[cfg(all(rscrypto_internal, feature = "diag"))]
const BLAKE3_OUT_LEN: usize = 32;

#[cfg(all(rscrypto_internal, feature = "diag"))]
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

#[cfg(all(rscrypto_internal, feature = "diag"))]
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

#[cfg(all(rscrypto_internal, feature = "diag"))]
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

#[cfg(all(rscrypto_internal, feature = "diag"))]
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

#[cfg(not(all(rscrypto_internal, feature = "diag")))]
#[inline]
fn print_blake3_diag_once() {}

/// Upstream XOF output for `data`, filling `out`.
fn upstream_xof(data: &[u8], out: &mut [u8]) {
  blake3::Hasher::new().update(data).finalize_xof().fill(out);
}

/// Upstream non-root chaining values of every full chunk of `data`.
#[cfg(all(rscrypto_internal, feature = "diag"))]
fn upstream_chunk_cvs(data: &[u8]) -> Vec<u8> {
  use blake3::hazmat::HasherExt as _;

  let mut cvs = Vec::new();
  for (index, chunk) in data.as_chunks::<BLAKE3_CHUNK_LEN>().0.iter().enumerate() {
    let offset = u64::try_from(index.strict_mul(BLAKE3_CHUNK_LEN)).expect("chunk offset fits u64");
    let mut hasher = blake3::Hasher::new();
    hasher.set_input_offset(offset).update(chunk);
    cvs.extend_from_slice(&hasher.finalize_non_root());
  }
  cvs
}

fn oneshot(c: &mut Criterion) {
  if !bench_config::selected("blake3") {
    return;
  }
  print_blake3_diag_once();

  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("blake3");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = *blake3::hash(data).as_bytes();

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Blake3::digest(black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );

    #[cfg(all(rscrypto_internal, feature = "diag"))]
    for &kernel in diag_kernels() {
      if !diag_blake3_kernel_available(kernel) {
        continue;
      }
      g.bench_function(
        BenchmarkId::new(format!("rscrypto-{}", kernel.label()), len),
        checked(
          || {
            diag_blake3_digest_with_kernel(kernel, black_box(data))
              .expect("selected BLAKE3 benchmark kernel must be available")
          },
          |out| assert_eq!(out, expected),
        ),
      );
    }

    g.bench_function(
      BenchmarkId::new("blake3", len),
      checked(
        || blake3::hash(black_box(data)),
        |out| assert_eq!(out.as_bytes(), &expected),
      ),
    );
  }

  g.finish();
}

fn keyed(c: &mut Criterion) {
  if !bench_config::selected("blake3/keyed") {
    return;
  }
  print_blake3_diag_once();

  let inputs = common::comp_sizes();
  let key = [0x42u8; 32];
  let mut g = c.benchmark_group("blake3/keyed");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = *blake3::keyed_hash(&key, data).as_bytes();

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Blake3::keyed_digest(black_box(&key), black_box(data)),
        |out| assert_eq!(out.as_bytes(), &expected),
      ),
    );

    #[cfg(all(rscrypto_internal, feature = "diag"))]
    for &kernel in diag_kernels() {
      if !diag_blake3_kernel_available(kernel) {
        continue;
      }
      g.bench_function(
        BenchmarkId::new(format!("rscrypto-{}", kernel.label()), len),
        checked(
          || {
            diag_blake3_keyed_digest_with_kernel(kernel, black_box(&key), black_box(data))
              .expect("selected BLAKE3 benchmark kernel must be available")
          },
          |out| assert_eq!(out.as_bytes(), &expected),
        ),
      );
    }

    g.bench_function(
      BenchmarkId::new("blake3", len),
      checked(
        || blake3::keyed_hash(black_box(&key), black_box(data)),
        |out| assert_eq!(out.as_bytes(), &expected),
      ),
    );
  }

  g.finish();
}

fn derive_key(c: &mut Criterion) {
  if !bench_config::selected("blake3/derive-key") {
    return;
  }
  print_blake3_diag_once();

  const CONTEXT: &str = "rscrypto benchmark derive-key context";

  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("blake3/derive-key");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let expected = blake3::derive_key(CONTEXT, data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Blake3::derive_key(black_box(CONTEXT), black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );

    g.bench_function(
      BenchmarkId::new("blake3", len),
      checked(
        || blake3::derive_key(black_box(CONTEXT), black_box(data)),
        |out| assert_eq!(out, expected),
      ),
    );
  }

  g.finish();
}

/// Hash 64 independent equal-length messages: one batch call versus one call each.
///
/// 64 messages fill every lane width evenly. 21 bytes is a short fingerprint
/// with a partial final block; 64, 256, and 1,024 bytes end on a block
/// boundary. The `blake3` row hashes one message per call; the reference crate
/// has no batch API.
fn batch(c: &mut Criterion) {
  if !bench_config::selected("blake3/batch") {
    return;
  }
  print_blake3_diag_once();

  const MESSAGES: usize = 64;
  let cases: Vec<(usize, Vec<u8>)> = [21usize, 64, 256, 1024]
    .into_iter()
    .map(|len| (len, common::random_bytes(MESSAGES.strict_mul(len))))
    .collect();

  let mut g = c.benchmark_group("blake3/batch");
  for (len, data) in &cases {
    let len = *len;
    let inputs: Vec<&[u8]> = data.chunks_exact(len).collect();
    let expected: Vec<[u8; 32]> = inputs.iter().map(|input| *blake3::hash(input).as_bytes()).collect();
    common::set_throughput(&mut g, MESSAGES.strict_mul(len));

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        vec![[0u8; 32]; MESSAGES],
        |outputs| rscrypto::Blake3::digest_batch(black_box(&inputs), black_box(&mut **outputs)),
        |(), outputs| assert_eq!(*outputs, expected, "batch output mismatch at len={len}"),
      ),
    );
    g.bench_function(
      BenchmarkId::new("blake3", len),
      checked_with(
        vec![[0u8; 32]; MESSAGES],
        |outputs| {
          for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            *output = *blake3::hash(black_box(input)).as_bytes();
          }
          black_box(&**outputs);
        },
        |(), outputs| assert_eq!(*outputs, expected),
      ),
    );
  }
  g.finish();

  let mut g = c.benchmark_group("blake3/batch-serial");
  for (len, data) in &cases {
    let len = *len;
    let inputs: Vec<&[u8]> = data.chunks_exact(len).collect();
    let expected: Vec<[u8; 32]> = inputs.iter().map(|input| *blake3::hash(input).as_bytes()).collect();
    common::set_throughput(&mut g, MESSAGES.strict_mul(len));

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        vec![[0u8; 32]; MESSAGES],
        |outputs| {
          for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            *output = rscrypto::Blake3::digest(black_box(input));
          }
          black_box(&*outputs);
        },
        |(), outputs| assert_eq!(*outputs, expected),
      ),
    );
  }
  g.finish();
}

fn streaming(c: &mut Criterion) {
  if !bench_config::selected("blake3/streaming") {
    return;
  }
  print_blake3_diag_once();

  let data = common::random_bytes(1048576);
  let expected = *blake3::hash(&data).as_bytes();
  let mut g = c.benchmark_group("blake3/streaming");
  g.throughput(criterion::Throughput::Bytes(data.len() as u64));

  for chunk_size in [64, 4096, 16384, 65536] {
    g.bench_function(
      format!("rscrypto/{chunk_size}B"),
      checked(
        || {
          use rscrypto::Digest;
          let mut h = rscrypto::Blake3::new();
          for chunk in data.chunks(chunk_size) {
            h.update(black_box(chunk));
          }
          h.finalize()
        },
        |out| assert_eq!(out, expected),
      ),
    );

    #[cfg(all(rscrypto_internal, feature = "diag"))]
    for &kernel in diag_kernels() {
      if !diag_blake3_kernel_available(kernel) || !kernel.supports_streaming() {
        continue;
      }
      g.bench_function(
        format!("rscrypto-{}/{chunk_size}B", kernel.label()),
        checked(
          || {
            diag_blake3_streaming_digest_with_kernel(kernel, black_box(&data), chunk_size)
              .expect("selected BLAKE3 benchmark kernel must be available")
          },
          |out| assert_eq!(out, expected),
        ),
      );
    }

    g.bench_function(
      format!("blake3/{chunk_size}B"),
      checked(
        || {
          let mut h = blake3::Hasher::new();
          for chunk in data.chunks(chunk_size) {
            h.update(black_box(chunk));
          }
          h.finalize()
        },
        |out| assert_eq!(out.as_bytes(), &expected),
      ),
    );
  }

  g.finish();
}

fn xof(c: &mut Criterion) {
  if !bench_config::selected("blake3/xof") {
    return;
  }
  print_blake3_diag_once();

  const OUT_LEN: usize = 64;

  let inputs = common::comp_sizes();
  let mut g = c.benchmark_group("blake3/xof");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let mut expected = [0u8; OUT_LEN];
    upstream_xof(data, &mut expected);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || {
          use rscrypto::Xof;

          let mut xof = rscrypto::Blake3::xof(black_box(data));
          let mut out = [0u8; OUT_LEN];
          xof.squeeze(&mut out);
          out
        },
        |out| assert_eq!(out, expected),
      ),
    );

    #[cfg(all(rscrypto_internal, feature = "diag"))]
    for &kernel in diag_kernels() {
      if !diag_blake3_kernel_available(kernel) {
        continue;
      }
      g.bench_function(
        BenchmarkId::new(format!("rscrypto-{}", kernel.label()), len),
        checked(
          || {
            let mut out = [0u8; OUT_LEN];
            diag_blake3_xof_with_kernel(kernel, black_box(data), &mut out)
              .expect("selected BLAKE3 benchmark kernel must be available");
            out
          },
          |out| assert_eq!(out, expected),
        ),
      );
    }

    g.bench_function(
      BenchmarkId::new("blake3", len),
      checked(
        || {
          let mut hasher = blake3::Hasher::new();
          hasher.update(black_box(data));
          let mut reader = hasher.finalize_xof();
          let mut out = [0u8; OUT_LEN];
          reader.fill(&mut out);
          out
        },
        |out| assert_eq!(out, expected),
      ),
    );
  }

  g.finish();
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
fn xof_output(c: &mut Criterion) {
  if !bench_config::selected("blake3/xof-output") {
    return;
  }
  use rscrypto::Xof;

  print_blake3_diag_once();

  const INPUT_LEN: usize = 4096;
  const OUTPUT_LENGTHS: [usize; 6] = [64, 128, 256, 512, 1024, 4096];

  let data = common::random_bytes(INPUT_LEN);
  let mut g = c.benchmark_group("blake3/xof-output");

  for out_len in OUTPUT_LENGTHS {
    common::set_throughput(&mut g, out_len);
    let mut expected = vec![0u8; out_len];
    upstream_xof(&data, &mut expected);

    g.bench_function(
      BenchmarkId::new("rscrypto", out_len),
      checked_with(
        vec![0u8; out_len],
        |out| {
          let mut xof = rscrypto::Blake3::xof(black_box(&data));
          xof.squeeze(black_box(out.as_mut_slice()));
          black_box(out[0])
        },
        |_, out| assert_eq!(*out, expected),
      ),
    );

    for &kernel in diag_kernels() {
      if !diag_blake3_kernel_available(kernel) {
        continue;
      }
      g.bench_function(
        BenchmarkId::new(format!("rscrypto-{}", kernel.label()), out_len),
        checked_with(
          vec![0u8; out_len],
          |out| {
            diag_blake3_xof_with_kernel(kernel, black_box(&data), black_box(out.as_mut_slice()))
              .expect("selected BLAKE3 benchmark kernel must be available");
            black_box(out[0])
          },
          |_, out| assert_eq!(*out, expected),
        ),
      );
    }

    g.bench_function(
      BenchmarkId::new("blake3", out_len),
      checked_with(
        vec![0u8; out_len],
        |out| {
          let mut hasher = blake3::Hasher::new();
          hasher.update(black_box(&data));
          let mut reader = hasher.finalize_xof();
          reader.fill(black_box(out.as_mut_slice()));
          black_box(out[0])
        },
        |_, out| assert_eq!(*out, expected),
      ),
    );
  }

  g.finish();
}

#[cfg(all(rscrypto_internal, feature = "diag"))]
fn tail_diagnostics(c: &mut Criterion) {
  if !bench_config::selected("blake3/") {
    return;
  }
  print_blake3_diag_once();

  let tail_counts = [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

  let mut digest_group = c.benchmark_group("blake3/chunk-tail-digest");
  for chunks in tail_counts.iter().copied() {
    let data = common::random_bytes(chunks.strict_mul(BLAKE3_CHUNK_LEN));
    common::set_throughput(&mut digest_group, data.len());
    let expected = *blake3::hash(&data).as_bytes();

    digest_group.bench_function(
      BenchmarkId::new("rscrypto", chunks),
      checked(
        || rscrypto::Blake3::digest(black_box(&data)),
        |out| assert_eq!(out, expected),
      ),
    );

    for &kernel in chunk_tail_diag_kernels() {
      if !diag_blake3_kernel_available(kernel) {
        continue;
      }
      digest_group.bench_function(
        BenchmarkId::new(format!("rscrypto-{}", kernel.label()), chunks),
        checked(
          || {
            diag_blake3_digest_with_kernel(kernel, black_box(&data))
              .expect("selected BLAKE3 benchmark kernel must be available")
          },
          |out| assert_eq!(out, expected),
        ),
      );
    }

    digest_group.bench_function(
      BenchmarkId::new("blake3", chunks),
      checked(
        || blake3::hash(black_box(&data)),
        |out| assert_eq!(out.as_bytes(), &expected),
      ),
    );
  }
  digest_group.finish();

  let mut chunk_group = c.benchmark_group("blake3/chunk-tail-cvs");
  for chunks in tail_counts.iter().copied() {
    let data = common::random_bytes(chunks.strict_mul(BLAKE3_CHUNK_LEN));
    common::set_throughput(&mut chunk_group, data.len());
    let expected = upstream_chunk_cvs(&data);

    for &kernel in chunk_tail_diag_kernels() {
      if !diag_blake3_kernel_available(kernel) {
        continue;
      }
      chunk_group.bench_function(
        BenchmarkId::new(format!("rscrypto-{}", kernel.label()), chunks),
        checked_with(
          vec![0u8; chunks.strict_mul(BLAKE3_OUT_LEN)],
          |out| {
            diag_blake3_chunk_cvs_with_kernel(kernel, black_box(&data), black_box(out.as_mut_slice()))
              .expect("selected BLAKE3 benchmark kernel must be available");
            black_box(out[0])
          },
          |_, out| assert_eq!(*out, expected),
        ),
      );
    }
  }
  chunk_group.finish();

  let mut parent_group = c.benchmark_group("blake3/parent-tail-cvs");
  for parents in tail_counts {
    let children = common::random_bytes(parents.strict_mul(2).strict_mul(BLAKE3_OUT_LEN));
    common::set_throughput(&mut parent_group, children.len());
    let expected: Vec<u8> = children
      .as_chunks::<64>()
      .0
      .iter()
      .flat_map(|pair| {
        let (left, right) = pair.split_at(BLAKE3_OUT_LEN);
        blake3::hazmat::merge_subtrees_non_root(
          left.try_into().expect("left CV has 32 bytes"),
          right.try_into().expect("right CV has 32 bytes"),
          blake3::hazmat::Mode::Hash,
        )
      })
      .collect();

    for &kernel in parent_tail_diag_kernels() {
      if !diag_blake3_kernel_available(kernel) {
        continue;
      }
      parent_group.bench_function(
        BenchmarkId::new(format!("rscrypto-{}", kernel.label()), parents),
        checked_with(
          vec![0u8; parents.strict_mul(BLAKE3_OUT_LEN)],
          |out| {
            diag_blake3_parent_cvs_with_kernel(kernel, black_box(&children), black_box(out.as_mut_slice()))
              .expect("selected BLAKE3 benchmark kernel must be available");
            black_box(out[0])
          },
          |_, out| assert_eq!(*out, expected),
        ),
      );
    }
  }
  parent_group.finish();
}

fn main() {
  mixed_batches::configure_isa().expect("valid BLAKE3 benchmark dispatch request");
  bench_config::run(&[
    oneshot,
    keyed,
    derive_key,
    batch,
    mixed_batches::bench,
    streaming,
    streaming_shapes::bench,
    streaming_shapes::alignment_bench,
    xof,
    #[cfg(all(rscrypto_internal, feature = "diag"))]
    xof_output,
    #[cfg(all(rscrypto_internal, feature = "diag"))]
    tail_diagnostics,
    #[cfg(all(rscrypto_internal, feature = "diag"))]
    leaf_alignment::bench,
  ]);
}
