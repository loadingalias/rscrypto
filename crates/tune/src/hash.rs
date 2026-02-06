//! Generic one-shot tunable for algorithms in the `hashes` crate.

use alloc::{boxed::Box, string::String, vec, vec::Vec};

use hashes::bench::{self, Kernel};
use platform::Caps;
#[cfg(target_arch = "aarch64")]
use platform::caps::aarch64;
#[cfg(target_arch = "x86_64")]
use platform::caps::x86;

use crate::{
  BenchResult, KernelSpec, KernelTier, TunableParam, TuneError, TuningDomain,
  sampler::{Sampler, SamplerConfig},
};

const HASH_PARAMS: &[TunableParam] = &[];

/// Core hash algorithm corpus (one-shot APIs).
pub const HASH_CORE_TUNING_CORPUS: &[(&str, &str)] = &[
  ("sha224", "RSCRYPTO_SHA224"),
  ("sha256", "RSCRYPTO_SHA256"),
  ("sha384", "RSCRYPTO_SHA384"),
  ("sha512", "RSCRYPTO_SHA512"),
  ("sha512-224", "RSCRYPTO_SHA512_224"),
  ("sha512-256", "RSCRYPTO_SHA512_256"),
  ("blake2b-512", "RSCRYPTO_BLAKE2B_512"),
  ("blake2s-256", "RSCRYPTO_BLAKE2S_256"),
  ("sha3-224", "RSCRYPTO_SHA3_224"),
  ("sha3-256", "RSCRYPTO_SHA3_256"),
  ("sha3-384", "RSCRYPTO_SHA3_384"),
  ("sha3-512", "RSCRYPTO_SHA3_512"),
  ("shake128", "RSCRYPTO_SHAKE128"),
  ("shake256", "RSCRYPTO_SHAKE256"),
  ("xxh3", "RSCRYPTO_XXH3"),
  ("rapidhash", "RSCRYPTO_RAPIDHASH"),
  ("siphash", "RSCRYPTO_SIPHASH"),
  ("keccakf1600", "RSCRYPTO_KECCAKF1600"),
  ("ascon-hash256", "RSCRYPTO_ASCON_HASH256"),
  ("ascon-xof128", "RSCRYPTO_ASCON_XOF128"),
];

/// Canonical BLAKE3 tuning corpus.
///
/// Keep all BLAKE3 surfaces in one place so tune registration, apply logic,
/// and target evaluation stay in sync.
pub const BLAKE3_TUNING_CORPUS: &[(&str, &str)] = &[
  ("blake3", "RSCRYPTO_BLAKE3"),
  ("blake3-chunk", "RSCRYPTO_BENCH_BLAKE3_CHUNK"),
  ("blake3-parent", "RSCRYPTO_BENCH_BLAKE3_PARENT"),
  ("blake3-parent-fold", "RSCRYPTO_BENCH_BLAKE3_PARENT_FOLD"),
  ("blake3-stream64", "RSCRYPTO_BENCH_BLAKE3_STREAM64"),
  ("blake3-stream4k", "RSCRYPTO_BENCH_BLAKE3_STREAM4K"),
  ("blake3-stream64-keyed", "RSCRYPTO_BENCH_BLAKE3_STREAM64_KEYED"),
  ("blake3-stream4k-keyed", "RSCRYPTO_BENCH_BLAKE3_STREAM4K_KEYED"),
  ("blake3-stream64-derive", "RSCRYPTO_BENCH_BLAKE3_STREAM64_DERIVE"),
  ("blake3-stream4k-derive", "RSCRYPTO_BENCH_BLAKE3_STREAM4K_DERIVE"),
];

/// Hash kernel-loop microbench corpus.
pub const HASH_MICRO_TUNING_CORPUS: &[(&str, &str)] = &[
  ("sha224-compress", "RSCRYPTO_BENCH_SHA224_COMPRESS"),
  ("sha256-compress", "RSCRYPTO_BENCH_SHA256_COMPRESS"),
  ("sha256-compress-unaligned", "RSCRYPTO_BENCH_SHA256_COMPRESS_UNALIGNED"),
  ("sha384-compress", "RSCRYPTO_BENCH_SHA384_COMPRESS"),
  ("sha512-compress", "RSCRYPTO_BENCH_SHA512_COMPRESS"),
  ("sha512-compress-unaligned", "RSCRYPTO_BENCH_SHA512_COMPRESS_UNALIGNED"),
  ("sha512-224-compress", "RSCRYPTO_BENCH_SHA512_224_COMPRESS"),
  ("sha512-256-compress", "RSCRYPTO_BENCH_SHA512_256_COMPRESS"),
  ("blake2b-512-compress", "RSCRYPTO_BENCH_BLAKE2B_512_COMPRESS"),
  ("blake2s-256-compress", "RSCRYPTO_BENCH_BLAKE2S_256_COMPRESS"),
  ("keccakf1600-permute", "RSCRYPTO_BENCH_KECCAKF1600_PERMUTE"),
];

/// Hash stream-profile corpus (small-update vs large-update).
pub const HASH_STREAM_PROFILE_TUNING_CORPUS: &[(&str, &str)] = &[
  ("sha256-stream64", "RSCRYPTO_BENCH_SHA256_STREAM64"),
  ("sha256-stream4k", "RSCRYPTO_BENCH_SHA256_STREAM4K"),
  ("sha512-stream64", "RSCRYPTO_BENCH_SHA512_STREAM64"),
  ("sha512-stream4k", "RSCRYPTO_BENCH_SHA512_STREAM4K"),
  ("blake2b-512-stream64", "RSCRYPTO_BENCH_BLAKE2B_512_STREAM64"),
  ("blake2b-512-stream4k", "RSCRYPTO_BENCH_BLAKE2B_512_STREAM4K"),
  ("blake2s-256-stream64", "RSCRYPTO_BENCH_BLAKE2S_256_STREAM64"),
  ("blake2s-256-stream4k", "RSCRYPTO_BENCH_BLAKE2S_256_STREAM4K"),
];

/// Returns true if `algo` is in the canonical BLAKE3 tuning corpus.
#[inline]
#[must_use]
pub fn is_blake3_tuning_algo(algo: &str) -> bool {
  BLAKE3_TUNING_CORPUS.iter().any(|(name, _)| *name == algo)
}

#[inline]
#[must_use]
fn is_blake3_stream_tuning_algo(algo: &str) -> bool {
  matches!(
    algo,
    "blake3-stream64"
      | "blake3-stream4k"
      | "blake3-stream64-keyed"
      | "blake3-stream4k-keyed"
      | "blake3-stream64-derive"
      | "blake3-stream4k-derive"
  )
}

fn kernel_specs(algo: &'static str, caps: &Caps) -> Vec<KernelSpec> {
  let mut base = vec![KernelSpec::new("portable", KernelTier::Portable, Caps::NONE)];

  // BLAKE3 primitives: allow forcing SIMD kernels so `rscrypto-tune` can
  // generate real dispatch tables (and validate crossovers) per platform.
  if is_blake3_tuning_algo(algo) {
    #[cfg(target_arch = "x86_64")]
    if caps.has(x86::SSSE3) {
      base.push(KernelSpec::new("x86_64/ssse3", KernelTier::Wide, x86::SSSE3));
    }
    #[cfg(target_arch = "x86_64")]
    if caps.has(x86::SSE41.union(x86::SSSE3)) {
      base.push(KernelSpec::new(
        "x86_64/sse4.1",
        KernelTier::Wide,
        x86::SSE41.union(x86::SSSE3),
      ));
    }
    #[cfg(target_arch = "x86_64")]
    if caps.has(x86::AVX2.union(x86::SSE41).union(x86::SSSE3)) {
      base.push(KernelSpec::new(
        "x86_64/avx2",
        KernelTier::Wide,
        x86::AVX2.union(x86::SSE41).union(x86::SSSE3),
      ));
    }
    #[cfg(target_arch = "x86_64")]
    // Match upstream BLAKE3: AVX-512 backends are gated by `avx512f` + `avx512vl`
    // (plus AVX2/SSE4.1/SSSE3 as baseline). Do not over-gate on BW/DQ here,
    // or Zen-class AVX-512 will be incorrectly excluded from tuning results.
    if caps.has(
      x86::AVX512F
        .union(x86::AVX512VL)
        .union(x86::AVX2)
        .union(x86::SSE41)
        .union(x86::SSSE3),
    ) {
      base.push(KernelSpec::new(
        "x86_64/avx512",
        KernelTier::Wide,
        x86::AVX512F
          .union(x86::AVX512VL)
          .union(x86::AVX2)
          .union(x86::SSE41)
          .union(x86::SSSE3),
      ));
    }
    #[cfg(target_arch = "aarch64")]
    if caps.has(aarch64::NEON) {
      base.push(KernelSpec::new("aarch64/neon", KernelTier::Wide, aarch64::NEON));
    }
  }

  if is_blake3_stream_tuning_algo(algo) {
    // Streaming uses a `(stream, bulk)` kernel pair at runtime. Tune pairs
    // directly so we don't pick two independent winners that regress together.
    let mut out = Vec::with_capacity(base.len().saturating_mul(base.len()));
    for stream in &base {
      for bulk in &base {
        let tier = if stream.name == "portable" && bulk.name == "portable" {
          KernelTier::Portable
        } else {
          KernelTier::Wide
        };
        let name: &'static str = Box::leak(format!("{}+{}", stream.name, bulk.name).into_boxed_str());
        out.push(KernelSpec::new(name, tier, stream.requires.union(bulk.requires)));
      }
    }
    return out;
  }

  base
}

/// One-shot hash tunable backed by `hashes::bench`.
///
/// The benchmark target is algorithm-specific (digest, hash, or primitive),
/// but the tuning interface is uniform and string-keyed.
pub struct HashTunable {
  algo: &'static str,
  env_prefix: &'static str,
  forced_kernel: Option<String>,
  cached_kernel: Option<Kernel>,
  effective_kernel_name: &'static str,
}

impl HashTunable {
  #[must_use]
  pub fn new(algo: &'static str, env_prefix: &'static str) -> Self {
    Self {
      algo,
      env_prefix,
      forced_kernel: None,
      cached_kernel: None,
      effective_kernel_name: "auto",
    }
  }

  fn resolve_kernel(&mut self) {
    if is_blake3_stream_tuning_algo(self.algo) {
      self.cached_kernel = None;
      self.effective_kernel_name = if let Some(name) = self.forced_kernel.as_deref() {
        Box::leak(name.to_string().into_boxed_str())
      } else {
        "auto"
      };
      return;
    }

    if let Some(ref name) = self.forced_kernel {
      self.cached_kernel = bench::get_kernel(self.algo, name);
      self.effective_kernel_name = self.cached_kernel.map(|k| k.name).unwrap_or("auto");
    } else {
      self.cached_kernel = None;
      self.effective_kernel_name = "auto";
    }
  }
}

impl crate::Tunable for HashTunable {
  fn name(&self) -> &'static str {
    self.algo
  }

  fn available_kernels(&self, caps: &Caps) -> Vec<KernelSpec> {
    kernel_specs(self.algo, caps)
  }

  fn force_kernel(&mut self, name: &str) -> Result<(), TuneError> {
    // Validate against available list for the current platform.
    let caps = platform::caps();
    let available = self.available_kernels(&caps);
    if !available.iter().any(|k| k.name == name) {
      return Err(TuneError::KernelNotAvailable("kernel not available on this platform"));
    }

    self.forced_kernel = Some(name.to_string());
    self.resolve_kernel();

    if self.cached_kernel.is_none() && !is_blake3_stream_tuning_algo(self.algo) {
      self.forced_kernel = None;
      return Err(TuneError::KernelNotAvailable(
        "kernel name did not resolve to a bench kernel",
      ));
    }

    Ok(())
  }

  fn force_streams(&mut self, count: u8) -> Result<(), TuneError> {
    if count == 1 {
      return Ok(());
    }
    Err(TuneError::InvalidStreamCount(count))
  }

  fn reset(&mut self) {
    self.forced_kernel = None;
    self.resolve_kernel();
  }

  fn benchmark(&self, data: &[u8], config: &SamplerConfig) -> BenchResult {
    let sampler = Sampler::new(config);

    if is_blake3_stream_tuning_algo(self.algo)
      && let Some(kernel_name) = self.forced_kernel.as_deref()
    {
      let result = sampler.run(data, |buf| {
        let out = bench::run_blake3_stream_forced(self.algo, kernel_name, core::hint::black_box(buf)).unwrap_or(0);
        core::hint::black_box(out);
      });

      return BenchResult {
        kernel: self.effective_kernel_name,
        buffer_size: data.len(),
        iterations: result.iterations,
        bytes_processed: result.bytes_processed,
        throughput_gib_s: result.throughput_gib_s,
        elapsed_secs: result.elapsed_secs,
        sample_count: Some(result.sample_count),
        std_dev: Some(result.std_dev),
        cv: Some(result.cv),
        outliers_rejected: Some(result.outliers_rejected),
        min_throughput_gib_s: Some(result.min_throughput_gib_s),
        max_throughput_gib_s: Some(result.max_throughput_gib_s),
      };
    }

    let (kernel_name, result) = if let Some(kernel) = self.cached_kernel {
      let func = kernel.func;
      let result = sampler.run(data, |buf| {
        core::hint::black_box(func(core::hint::black_box(buf)));
      });
      (kernel.name, result)
    } else {
      let result = sampler.run(data, |buf| {
        let out = bench::run_auto(self.algo, core::hint::black_box(buf)).unwrap_or(0);
        core::hint::black_box(out);
      });
      (
        bench::kernel_name_for_len(self.algo, data.len()).unwrap_or("unknown"),
        result,
      )
    };

    BenchResult {
      kernel: kernel_name,
      buffer_size: data.len(),
      iterations: result.iterations,
      bytes_processed: result.bytes_processed,
      throughput_gib_s: result.throughput_gib_s,
      elapsed_secs: result.elapsed_secs,
      sample_count: Some(result.sample_count),
      std_dev: Some(result.std_dev),
      cv: Some(result.cv),
      outliers_rejected: Some(result.outliers_rejected),
      min_throughput_gib_s: Some(result.min_throughput_gib_s),
      max_throughput_gib_s: Some(result.max_throughput_gib_s),
    }
  }

  fn current_kernel(&self) -> &'static str {
    self.effective_kernel_name
  }

  fn tunable_params(&self) -> &[TunableParam] {
    HASH_PARAMS
  }

  fn env_prefix(&self) -> &'static str {
    self.env_prefix
  }

  fn tuning_domain(&self) -> TuningDomain {
    TuningDomain::Hash
  }

  fn threshold_to_env_suffix(&self, threshold_name: &str) -> Option<&'static str> {
    match (self.algo, threshold_name) {
      ("blake3", "parallel_min_bytes") => Some("PARALLEL_MIN_BYTES"),
      ("blake3", "parallel_min_chunks") => Some("PARALLEL_MIN_CHUNKS"),
      ("blake3", "parallel_max_threads") => Some("PARALLEL_MAX_THREADS"),
      ("blake3", "parallel_spawn_cost_bytes") => Some("PARALLEL_SPAWN_COST_BYTES"),
      ("blake3", "parallel_merge_cost_bytes") => Some("PARALLEL_MERGE_COST_BYTES"),
      ("blake3", "parallel_bytes_per_core_small") => Some("PARALLEL_BYTES_PER_CORE_SMALL"),
      ("blake3", "parallel_bytes_per_core_medium") => Some("PARALLEL_BYTES_PER_CORE_MEDIUM"),
      ("blake3", "parallel_bytes_per_core_large") => Some("PARALLEL_BYTES_PER_CORE_LARGE"),
      ("blake3", "parallel_small_limit_bytes") => Some("PARALLEL_SMALL_LIMIT_BYTES"),
      ("blake3", "parallel_medium_limit_bytes") => Some("PARALLEL_MEDIUM_LIMIT_BYTES"),
      _ => None,
    }
  }
}
