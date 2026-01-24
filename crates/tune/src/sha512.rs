//! SHA-512 tunable implementation.

use alloc::{string::String, vec, vec::Vec};

use hashes::bench::{self, Sha512Kernel};
use platform::Caps;

use crate::{
  BenchResult, KernelSpec, KernelTier, TunableParam, TuneError,
  sampler::{Sampler, SamplerConfig},
};

const SHA512_PARAMS: &[TunableParam] = &[];

fn sha512_kernel_specs(_caps: &Caps) -> Vec<KernelSpec> {
  vec![KernelSpec::new("portable", KernelTier::Portable, Caps::NONE)]
}

/// Tunable implementation for SHA-512.
pub struct Sha512Tunable {
  forced_kernel: Option<String>,
  cached_kernel: Option<Sha512Kernel>,
  effective_kernel_name: &'static str,
}

impl Sha512Tunable {
  #[must_use]
  pub fn new() -> Self {
    Self {
      forced_kernel: None,
      cached_kernel: None,
      effective_kernel_name: "auto",
    }
  }

  fn resolve_kernel(&mut self) {
    if let Some(ref name) = self.forced_kernel {
      self.cached_kernel = bench::get_sha512_kernel(name);
      self.effective_kernel_name = self.cached_kernel.map(|k| k.name).unwrap_or("auto");
    } else {
      self.cached_kernel = None;
      self.effective_kernel_name = "auto";
    }
  }
}

impl Default for Sha512Tunable {
  fn default() -> Self {
    Self::new()
  }
}

impl crate::Tunable for Sha512Tunable {
  fn name(&self) -> &'static str {
    "sha512"
  }

  fn available_kernels(&self, caps: &Caps) -> Vec<KernelSpec> {
    sha512_kernel_specs(caps)
  }

  fn force_kernel(&mut self, name: &str) -> Result<(), TuneError> {
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

  fn benchmark(&self, data: &[u8], config: &SamplerConfig) -> BenchResult {
    let sampler = Sampler::new(config);

    let (kernel_name, result) = if let Some(kernel) = self.cached_kernel {
      let func = kernel.func;
      let result = sampler.run(data, |buf| {
        core::hint::black_box(func(core::hint::black_box(buf)));
      });
      (kernel.name, result)
    } else {
      let result = sampler.run(data, |buf| {
        core::hint::black_box(hashes::crypto::sha512::dispatch::digest(core::hint::black_box(buf)));
      });
      (hashes::crypto::sha512::dispatch::kernel_name_for_len(data.len()), result)
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
    SHA512_PARAMS
  }

  fn env_prefix(&self) -> &'static str {
    "RSCRYPTO_SHA512"
  }
}
