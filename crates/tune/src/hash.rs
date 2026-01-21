//! Generic one-shot tunable for algorithms in the `hashes` crate.

use alloc::{string::String, vec, vec::Vec};

use hashes::bench::{self, Kernel};
use platform::Caps;
#[cfg(target_arch = "aarch64")]
use platform::caps::aarch64;
#[cfg(target_arch = "x86_64")]
use platform::caps::x86;

use crate::{
  BenchResult, KernelSpec, KernelTier, TunableParam, TuneError,
  sampler::{Sampler, SamplerConfig},
};

const HASH_PARAMS: &[TunableParam] = &[];

fn kernel_specs(algo: &'static str, caps: &Caps) -> Vec<KernelSpec> {
  let mut out = vec![KernelSpec::new("portable", KernelTier::Portable, Caps::NONE)];

  // BLAKE3 primitives: allow forcing SIMD kernels so `rscrypto-tune` can
  // generate real dispatch tables (and validate crossovers) per platform.
  if algo == "blake3-chunk" || algo == "blake3-parent" {
    #[cfg(target_arch = "x86_64")]
    if caps.has(x86::SSSE3) {
      out.push(KernelSpec::new("x86_64/ssse3", KernelTier::Wide, x86::SSSE3));
    }
    #[cfg(target_arch = "x86_64")]
    if caps.has(x86::SSE41.union(x86::SSSE3)) {
      out.push(KernelSpec::new(
        "x86_64/sse4.1",
        KernelTier::Wide,
        x86::SSE41.union(x86::SSSE3),
      ));
    }
    #[cfg(target_arch = "x86_64")]
    if caps.has(x86::AVX2.union(x86::SSE41).union(x86::SSSE3)) {
      out.push(KernelSpec::new(
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
      out.push(KernelSpec::new(
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
      out.push(KernelSpec::new("aarch64/neon", KernelTier::Wide, aarch64::NEON));
    }
  }

  out
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

    if self.cached_kernel.is_none() {
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

  fn benchmark(&self, data: &[u8], _iterations: usize) -> BenchResult {
    let config = SamplerConfig::default();
    let sampler = Sampler::new(&config);

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
}
