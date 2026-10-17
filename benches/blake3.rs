//! BLAKE3 comparison benchmarks: rscrypto against the official `blake3` crate.
//!
//! Two catalog builds share this binary. `blake3_serial` (no `parallel`)
//! registers the single-thread task groups under `blake3/serial/`.
//! `blake3` (with `parallel`) registers the multi-thread-allowed task groups
//! under `blake3/parallel/` and the groups in which no row threads. `main`
//! lists each build's groups. Every rscrypto row is checked against upstream
//! output, and every upstream row against rscrypto output, before timing.

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
use std::io::{self, Read as _};

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

/// Case-ID segment of the task groups that both builds register: `serial`
/// without `parallel` (no row threads), `parallel` with it (threads allowed).
const BUILD: &str = if cfg!(feature = "parallel") {
  "parallel"
} else {
  "serial"
};

const KEY: [u8; 32] = [0x42; 32];
const CONTEXT: &str = "rscrypto benchmark derive-key context";
const XOF_LEN: usize = 64;
/// One byte past one 64-byte block and one 1,024-byte chunk: the shortest messages that need a second compression
/// and a second chunk.
const BLOCK_BOUNDARIES: [usize; 2] = [65, 1025];

/// Hash one message in one call.
///
/// `blake3/serial/hash` (build `blake3_serial`): rscrypto `Blake3::digest`
/// against `blake3::hash`; neither threads. Diagnostic `rscrypto-<kernel>` rows
/// force one kernel through the same one-shot path.
///
/// `blake3/parallel/hash` (build `blake3`): threads are allowed. `Blake3::digest`
/// joins on the current Rayon pool once the cost model of the CPU profile
/// admits the input: from 64 KiB on the AArch64 NEON profile and from 256 KiB
/// on the x86 AVX-512 profile, for example. `blake3::hash` is always serial.
/// `blake3-rayon` calls `Hasher::new().update_rayon(..)` and `finalize`; it
/// joins above the SIMD degree with no size gate. A comparison takes the
/// faster upstream row at each size.
///
/// Timed: the call and its 32-byte output. Untimed: inputs and expected
/// outputs. Neither library keeps state between calls. Both libraries share
/// one global Rayon pool, which starts during an output check, before timing.
fn hash(c: &mut Criterion) {
  let name = format!("blake3/{BUILD}/hash");
  if !bench_config::selected(&name) {
    return;
  }
  print_blake3_diag_once();

  let mut g = c.benchmark_group(name);
  for (len, data) in &common::comp_sizes(&BLOCK_BOUNDARIES) {
    common::set_throughput(&mut g, *len);
    let upstream = *blake3::hash(data).as_bytes();
    let ours = rscrypto::Blake3::digest(data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Blake3::digest(black_box(data)),
        |out| assert_eq!(out, upstream),
      ),
    );

    #[cfg(all(rscrypto_internal, feature = "diag"))]
    if !cfg!(feature = "parallel") {
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
            |out| assert_eq!(out, upstream),
          ),
        );
      }
    }

    g.bench_function(
      BenchmarkId::new("blake3", len),
      checked(
        || blake3::hash(black_box(data)),
        |out| assert_eq!(out.as_bytes(), &ours),
      ),
    );

    if cfg!(feature = "parallel") {
      g.bench_function(
        BenchmarkId::new("blake3-rayon", len),
        checked(
          || blake3::Hasher::new().update_rayon(black_box(data)).finalize(),
          |out| assert_eq!(out.as_bytes(), &ours),
        ),
      );
    }
  }
  g.finish();
}

/// Keyed-hash one message in one call with a constant 32-byte key.
///
/// Builds, threading, rows, and timing match `hash`: rscrypto
/// `Blake3::keyed_digest` against `blake3::keyed_hash`, plus `blake3-rayon`
/// (`Hasher::new_keyed(..).update_rayon(..)`, `finalize`) in
/// `blake3/parallel/keyed`. Neither library keeps key state between calls.
fn keyed(c: &mut Criterion) {
  let name = format!("blake3/{BUILD}/keyed");
  if !bench_config::selected(&name) {
    return;
  }
  print_blake3_diag_once();

  let mut g = c.benchmark_group(name);
  for (len, data) in &common::comp_sizes(&BLOCK_BOUNDARIES) {
    common::set_throughput(&mut g, *len);
    let upstream = *blake3::keyed_hash(&KEY, data).as_bytes();
    let ours = *rscrypto::Blake3::keyed_digest(&KEY, data).as_bytes();

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Blake3::keyed_digest(black_box(&KEY), black_box(data)),
        |out| assert_eq!(out.as_bytes(), &upstream),
      ),
    );

    #[cfg(all(rscrypto_internal, feature = "diag"))]
    if !cfg!(feature = "parallel") {
      for &kernel in diag_kernels() {
        if !diag_blake3_kernel_available(kernel) {
          continue;
        }
        g.bench_function(
          BenchmarkId::new(format!("rscrypto-{}", kernel.label()), len),
          checked(
            || {
              diag_blake3_keyed_digest_with_kernel(kernel, black_box(&KEY), black_box(data))
                .expect("selected BLAKE3 benchmark kernel must be available")
            },
            |out| assert_eq!(out.as_bytes(), &upstream),
          ),
        );
      }
    }

    g.bench_function(
      BenchmarkId::new("blake3", len),
      checked(
        || blake3::keyed_hash(black_box(&KEY), black_box(data)),
        |out| assert_eq!(out.as_bytes(), &ours),
      ),
    );

    if cfg!(feature = "parallel") {
      g.bench_function(
        BenchmarkId::new("blake3-rayon", len),
        checked(
          || {
            blake3::Hasher::new_keyed(black_box(&KEY))
              .update_rayon(black_box(data))
              .finalize()
          },
          |out| assert_eq!(out.as_bytes(), &ours),
        ),
      );
    }
  }
  g.finish();
}

/// Derive a key when each call supplies the context string, so each call
/// hashes the 37-byte context, then the key material.
///
/// rscrypto builds `Blake3DeriveContext::new` and calls `derive_key_with`;
/// `blake3::derive_key` hashes the context internally. Both hashes are timed.
/// `Blake3::derive_key` is left out: its per-thread cache skips the context
/// hash after the first call, which `derive-key/prepared-context` measures.
/// Builds and threading match `hash`; `blake3-rayon` calls
/// `Hasher::new_derive_key(..).update_rayon(..)` and `finalize`.
fn derive_key_context(c: &mut Criterion) {
  let name = format!("blake3/{BUILD}/derive-key/context");
  if !bench_config::selected(&name) {
    return;
  }
  print_blake3_diag_once();

  let mut g = c.benchmark_group(name);
  for (len, data) in &common::comp_sizes(&[]) {
    common::set_throughput(&mut g, *len);
    let upstream = blake3::derive_key(CONTEXT, data);
    let ours = rscrypto::Blake3::derive_key_with(&rscrypto::Blake3DeriveContext::new(CONTEXT), data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || {
          let context = rscrypto::Blake3DeriveContext::new(black_box(CONTEXT));
          rscrypto::Blake3::derive_key_with(&context, black_box(data))
        },
        |out| assert_eq!(out, upstream),
      ),
    );

    g.bench_function(
      BenchmarkId::new("blake3", len),
      checked(
        || blake3::derive_key(black_box(CONTEXT), black_box(data)),
        |out| assert_eq!(out, ours),
      ),
    );

    if cfg!(feature = "parallel") {
      g.bench_function(
        BenchmarkId::new("blake3-rayon", len),
        checked(
          || {
            *blake3::Hasher::new_derive_key(black_box(CONTEXT))
              .update_rayon(black_box(data))
              .finalize()
              .as_bytes()
          },
          |out| assert_eq!(out, ours),
        ),
      );
    }
  }
  g.finish();
}

/// Derive a key from a context hashed once, before timing.
///
/// `rscrypto` passes a prepared `Blake3DeriveContext` to `derive_key_with`.
/// `rscrypto-cached` passes the context string to `Blake3::derive_key`; its
/// per-thread cache already holds that context, so each call compares 37 bytes
/// instead of hashing them. Upstream has no one-shot form for a prepared
/// context: `blake3` builds `hazmat::HasherExt::new_from_context_key`, updates,
/// and finalizes in each timed call. Only `blake3_serial` registers this
/// single-thread group.
fn derive_key_prepared_context(c: &mut Criterion) {
  use blake3::hazmat::HasherExt as _;

  const NAME: &str = "blake3/serial/derive-key/prepared-context";
  if !bench_config::selected(NAME) {
    return;
  }
  print_blake3_diag_once();

  let prepared = rscrypto::Blake3DeriveContext::new(CONTEXT);
  let context_key = blake3::hazmat::hash_derive_key_context(CONTEXT);
  let mut g = c.benchmark_group(NAME);
  for (len, data) in &common::comp_sizes(&[]) {
    common::set_throughput(&mut g, *len);
    let upstream = blake3::derive_key(CONTEXT, data);
    let ours = rscrypto::Blake3::derive_key_with(&prepared, data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto::Blake3::derive_key_with(black_box(&prepared), black_box(data)),
        |out| assert_eq!(out, upstream),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rscrypto-cached", len),
      checked(
        || rscrypto::Blake3::derive_key(black_box(CONTEXT), black_box(data)),
        |out| assert_eq!(out, upstream),
      ),
    );

    g.bench_function(
      BenchmarkId::new("blake3", len),
      checked(
        || {
          let mut hasher = blake3::Hasher::new_from_context_key(black_box(&context_key));
          hasher.update(black_box(data));
          *hasher.finalize().as_bytes()
        },
        |out| assert_eq!(out, ours),
      ),
    );
  }
  g.finish();
}

/// Hash 64 independent equal-length messages: one batch call versus one call each.
///
/// 64 messages fill every lane width evenly. 21 bytes is a short fingerprint
/// with a partial final block; 64, 256, and 1,024 bytes end on a block
/// boundary. `blake3/batch`: rscrypto `Blake3::digest_batch` against one
/// `blake3::hash` call per message; upstream has no batch API. The
/// rscrypto-only `blake3/batch-serial` makes one `Blake3::digest` call per
/// message. Timed: every hash and the writes into reused output storage. No
/// row threads, because every message fits one chunk, so only the `blake3`
/// build registers these groups.
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
    let upstream: Vec<[u8; 32]> = inputs.iter().map(|input| *blake3::hash(input).as_bytes()).collect();
    let ours: Vec<[u8; 32]> = inputs.iter().map(|input| rscrypto::Blake3::digest(input)).collect();
    common::set_throughput(&mut g, MESSAGES.strict_mul(len));

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        vec![[0u8; 32]; MESSAGES],
        |outputs| rscrypto::Blake3::digest_batch(black_box(&inputs), black_box(&mut **outputs)),
        |(), outputs| assert_eq!(*outputs, upstream, "batch output mismatch at len={len}"),
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
        |(), outputs| assert_eq!(*outputs, ours),
      ),
    );
  }
  g.finish();

  let mut g = c.benchmark_group("blake3/batch-serial");
  for (len, data) in &cases {
    let len = *len;
    let inputs: Vec<&[u8]> = data.chunks_exact(len).collect();
    let upstream: Vec<[u8; 32]> = inputs.iter().map(|input| *blake3::hash(input).as_bytes()).collect();
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
        |(), outputs| assert_eq!(*outputs, upstream),
      ),
    );
  }
  g.finish();
}

/// Hash 1 MiB through `update` calls of 64 B to 64 KiB, then finalize.
///
/// Timed: construction, every update, and finalization. No row threads, so
/// only the `blake3` build registers this group. rscrypto's streaming
/// admission needs at least 128 KiB in one update on the CPU profiles that
/// enable it, and is off on AVX-512, NEON, z15, and POWER10 profiles.
/// Upstream `Hasher::update` is serial.
fn streaming(c: &mut Criterion) {
  if !bench_config::selected("blake3/streaming") {
    return;
  }
  print_blake3_diag_once();

  let data = common::random_bytes(1048576);
  let upstream = *blake3::hash(&data).as_bytes();
  let ours = rscrypto::Blake3::digest(&data);
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
        |out| assert_eq!(out, upstream),
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
          |out| assert_eq!(out, upstream),
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
        |out| assert_eq!(out.as_bytes(), &ours),
      ),
    );
  }

  g.finish();
}

/// Absorb one message and squeeze 64 output bytes.
///
/// rscrypto `Blake3::xof` and `squeeze` against upstream
/// `Hasher::new().update(..)`, `finalize_xof`, and `fill`. Builds, threading,
/// and diagnostic rows match `hash`; `blake3-rayon` replaces `update` with
/// `update_rayon`. Timed: reader construction, the squeeze, and the stack
/// output buffer.
fn xof(c: &mut Criterion) {
  use rscrypto::Xof as _;

  let name = format!("blake3/{BUILD}/xof");
  if !bench_config::selected(&name) {
    return;
  }
  print_blake3_diag_once();

  let mut g = c.benchmark_group(name);
  for (len, data) in &common::comp_sizes(&[]) {
    common::set_throughput(&mut g, *len);
    let mut upstream = [0u8; XOF_LEN];
    upstream_xof(data, &mut upstream);
    let mut ours = [0u8; XOF_LEN];
    rscrypto::Blake3::xof(data).squeeze(&mut ours);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || {
          let mut xof = rscrypto::Blake3::xof(black_box(data));
          let mut out = [0u8; XOF_LEN];
          xof.squeeze(&mut out);
          out
        },
        |out| assert_eq!(out, upstream),
      ),
    );

    #[cfg(all(rscrypto_internal, feature = "diag"))]
    if !cfg!(feature = "parallel") {
      for &kernel in diag_kernels() {
        if !diag_blake3_kernel_available(kernel) {
          continue;
        }
        g.bench_function(
          BenchmarkId::new(format!("rscrypto-{}", kernel.label()), len),
          checked(
            || {
              let mut out = [0u8; XOF_LEN];
              diag_blake3_xof_with_kernel(kernel, black_box(data), &mut out)
                .expect("selected BLAKE3 benchmark kernel must be available");
              out
            },
            |out| assert_eq!(out, upstream),
          ),
        );
      }
    }

    g.bench_function(
      BenchmarkId::new("blake3", len),
      checked(
        || {
          let mut hasher = blake3::Hasher::new();
          hasher.update(black_box(data));
          let mut reader = hasher.finalize_xof();
          let mut out = [0u8; XOF_LEN];
          reader.fill(&mut out);
          out
        },
        |out| assert_eq!(out, ours),
      ),
    );

    if cfg!(feature = "parallel") {
      g.bench_function(
        BenchmarkId::new("blake3-rayon", len),
        checked(
          || {
            let mut hasher = blake3::Hasher::new();
            hasher.update_rayon(black_box(data));
            let mut reader = hasher.finalize_xof();
            let mut out = [0u8; XOF_LEN];
            reader.fill(&mut out);
            out
          },
          |out| assert_eq!(out, ours),
        ),
      );
    }
  }
  g.finish();
}

/// Absorb a 32-byte seed and squeeze long output into a reused buffer, on one thread.
///
/// `blake3/serial/xof-output` (build `blake3_serial`): rscrypto `Blake3::xof`
/// and `squeeze` against upstream `Hasher::update`, `finalize_xof`, and `fill`.
/// `xof` covers 64-byte output; these lengths measure the squeeze: 16 output
/// blocks, 256 blocks, and 1 MiB. Timed: absorption, reader construction, and
/// the squeeze. The output buffer is allocated outside timing. Diagnostic
/// `rscrypto-<kernel>` rows force one kernel through the same path.
fn xof_output(c: &mut Criterion) {
  use rscrypto::Xof as _;

  const INPUT_LEN: usize = 32;
  const OUTPUT_LENGTHS: [usize; 3] = [1024, 16384, 1048576];

  let name = format!("blake3/{BUILD}/xof-output");
  if !bench_config::selected(&name) {
    return;
  }
  print_blake3_diag_once();

  let data = common::random_bytes(INPUT_LEN);
  let mut g = c.benchmark_group(name);

  for out_len in OUTPUT_LENGTHS {
    common::set_throughput(&mut g, out_len);
    let mut expected = vec![0u8; out_len];
    upstream_xof(&data, &mut expected);
    let mut ours = vec![0u8; out_len];
    rscrypto::Blake3::xof(&data).squeeze(&mut ours);

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

    #[cfg(all(rscrypto_internal, feature = "diag"))]
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
        |_, out| assert_eq!(*out, ours),
      ),
    );
  }

  g.finish();
}

/// Returns a Bao decoding row. `decode` is the row's timed path: it must decode `encoding` into its buffer and
/// return the original input. Outside timing, it must reject `changed`.
fn bao_decode_row<'a>(
  decode: impl Fn(&[u8], &mut Vec<u8>) -> io::Result<()> + 'a,
  encoding: &'a [u8],
  changed: &'a [u8],
  input: &'a [u8],
) -> impl FnMut(&mut criterion::Bencher<'_>) + 'a {
  checked_with(
    (Vec::with_capacity(input.len()), decode),
    move |(out, decode)| decode(black_box(encoding), out).expect("a valid Bao benchmark encoding must decode"),
    move |(), (out, decode)| {
      assert_eq!(out.as_slice(), input);
      assert!(
        decode(changed, &mut Vec::new()).is_err(),
        "a changed Bao encoding must fail"
      );
    },
  )
}

/// Verify a whole Bao combined encoding from memory against a trusted root, on one thread.
///
/// `blake3/serial/bao-decode` (build `blake3_serial`): rscrypto's streaming
/// `hashes::expert::bao::Decoder` against the format's reference crate, bao
/// 0.13.1. `bao` calls its one-shot `decode`, which returns a new `Vec`;
/// `bao-reader` and `rscrypto` read their decoder to EOF into a reused
/// buffer. Timed: decoder construction, every read and verification, the
/// output, and destruction. The encoding and root are built outside timing.
/// Lengths: one chunk, the first parent node, and 64 KiB and 1 MiB trees. Every
/// row must return the original input and reject the encoding with its last
/// byte changed.
fn bao_decode(c: &mut Criterion) {
  const INPUT_LENGTHS: [usize; 4] = [1024, 1025, 65536, 1048576];

  let name = format!("blake3/{BUILD}/bao-decode");
  if !bench_config::selected(&name) {
    return;
  }

  let mut g = c.benchmark_group(name);
  for len in INPUT_LENGTHS {
    let input = common::random_bytes(len);
    let (encoding, root) = bao::encode::encode(&input);
    let mut changed = encoding.clone();
    *changed.last_mut().expect("a Bao encoding is not empty") ^= 1;
    common::set_throughput(&mut g, len);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      bao_decode_row(
        |encoding, out| {
          out.clear();
          rscrypto::hashes::expert::bao::Decoder::new(encoding, root.as_bytes())
            .read_to_end(out)
            .map(drop)
        },
        &encoding,
        &changed,
        &input,
      ),
    );
    g.bench_function(
      BenchmarkId::new("bao", len),
      bao_decode_row(
        |encoding, out| {
          *out = bao::decode::decode(encoding, &root)?;
          Ok(())
        },
        &encoding,
        &changed,
        &input,
      ),
    );
    g.bench_function(
      BenchmarkId::new("bao-reader", len),
      bao_decode_row(
        |encoding, out| {
          out.clear();
          bao::decode::Decoder::new(encoding, &root).read_to_end(out).map(drop)
        },
        &encoding,
        &changed,
        &input,
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
    let ours = rscrypto::Blake3::digest(&data);

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
        |out| assert_eq!(out.as_bytes(), &ours),
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
  if cfg!(feature = "parallel") {
    // Build `blake3`: threads allowed, plus the groups in which no row threads.
    bench_config::run(&[
      hash,
      keyed,
      derive_key_context,
      xof,
      batch,
      mixed_batches::bench,
      streaming,
      #[cfg(all(rscrypto_internal, feature = "diag"))]
      tail_diagnostics,
      #[cfg(all(rscrypto_internal, feature = "diag"))]
      leaf_alignment::bench,
    ]);
  } else {
    // Build `blake3_serial`: single-thread task groups only.
    bench_config::run(&[
      hash,
      keyed,
      derive_key_context,
      derive_key_prepared_context,
      xof,
      xof_output,
      bao_decode,
      streaming_shapes::bench,
      streaming_shapes::alignment_bench,
    ]);
  }
}
