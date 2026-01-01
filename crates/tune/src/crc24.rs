//! CRC-24 tunable implementations.
//!
//! This module provides [`Tunable`](crate::Tunable) implementations for CRC-24 algorithms:
//! - [`Crc24OpenPgpTunable`] - CRC-24/OpenPGP (RFC 4880)
//!
//! These tunables support direct kernel benchmarking via the `bench` module's
//! kernel lookup functions, allowing the tuning engine to measure specific
//! kernel implementations directly rather than going through the library dispatch.

use alloc::{string::String, vec, vec::Vec};

use checksum::bench::{self, Crc24Kernel};
use platform::Caps;

use crate::{
  BenchResult, KernelSpec, KernelTier, TunableParam, TuneError,
  sampler::{Sampler, SamplerConfig},
};

// ─────────────────────────────────────────────────────────────────────────────
// Tunable Parameters
// ─────────────────────────────────────────────────────────────────────────────

const CRC24_PARAMS: &[TunableParam] = &[
  TunableParam::new(
    "portable_to_clmul",
    "Bytes where SIMD becomes faster than portable",
    32,
    4096,
    128,
  ),
  TunableParam::new("streams", "Number of parallel folding streams", 1, 4, 2),
];

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Specifications
// ─────────────────────────────────────────────────────────────────────────────

fn crc24_kernel_specs(caps: &Caps) -> Vec<KernelSpec> {
  let specs = vec![
    KernelSpec::new("reference", KernelTier::Reference, Caps::NONE),
    KernelSpec::new("portable/slice8", KernelTier::Portable, Caps::NONE),
  ];

  // CRC-24 currently only has portable implementations
  let _ = caps;

  specs
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-24/OpenPGP Tunable
// ─────────────────────────────────────────────────────────────────────────────

/// Tunable implementation for CRC-24/OpenPGP.
///
/// Supports direct kernel benchmarking via `force_kernel()`, allowing
/// measurement of specific kernel implementations.
pub struct Crc24OpenPgpTunable {
  /// Forced kernel name (None = auto).
  forced_kernel: Option<String>,
  /// Forced stream count (None = auto, reserved for future multi-stream).
  forced_streams: Option<u8>,
  /// Cached kernel function (resolved from forced_kernel).
  cached_kernel: Option<Crc24Kernel>,
  /// Effective kernel name for reporting.
  effective_kernel_name: &'static str,
}

impl Crc24OpenPgpTunable {
  /// Create a new CRC-24/OpenPGP tunable in auto mode.
  #[must_use]
  pub fn new() -> Self {
    Self {
      forced_kernel: None,
      forced_streams: None,
      cached_kernel: None,
      effective_kernel_name: "auto",
    }
  }

  /// Resolve the forced kernel to a function pointer.
  fn resolve_kernel(&mut self) {
    if let Some(ref name) = self.forced_kernel {
      if let Some(kernel) = bench::get_crc24_openpgp_kernel(name) {
        self.cached_kernel = Some(kernel);
        self.effective_kernel_name = kernel.name;
      }
    } else {
      self.cached_kernel = None;
      self.effective_kernel_name = "auto";
    }
  }
}

impl Default for Crc24OpenPgpTunable {
  fn default() -> Self {
    Self::new()
  }
}

impl crate::Tunable for Crc24OpenPgpTunable {
  fn name(&self) -> &'static str {
    "crc24-openpgp"
  }

  fn available_kernels(&self, caps: &Caps) -> Vec<KernelSpec> {
    crc24_kernel_specs(caps)
  }

  fn force_kernel(&mut self, name: &str) -> Result<(), TuneError> {
    let caps = platform::caps();
    let available = self.available_kernels(&caps);
    let base_name = name.split('-').next().unwrap_or(name);
    let valid = available.iter().any(|k| {
      k.name == name || k.name == base_name || name.starts_with(k.name) || name == "reference" || name == "portable"
    });
    if !valid {
      return Err(TuneError::KernelNotAvailable("kernel not available on this platform"));
    }
    self.forced_kernel = Some(name.to_string());
    self.resolve_kernel();
    Ok(())
  }

  fn force_streams(&mut self, count: u8) -> Result<(), TuneError> {
    if count == 0 || count > 16 {
      return Err(TuneError::InvalidStreamCount(count));
    }
    self.forced_streams = Some(count);
    self.resolve_kernel();
    Ok(())
  }

  fn reset(&mut self) {
    self.forced_kernel = None;
    self.forced_streams = None;
    self.cached_kernel = None;
    self.effective_kernel_name = "auto";
  }

  fn benchmark(&self, data: &[u8], _iterations: usize) -> BenchResult {
    let config = SamplerConfig::default();
    let sampler = Sampler::new(&config);

    let (kernel_name, result) = if let Some(ref kernel) = self.cached_kernel {
      let func = kernel.func;
      let result = sampler.run(data, |buf| {
        core::hint::black_box(func(core::hint::black_box(0u32), core::hint::black_box(buf)));
      });
      (kernel.name, result)
    } else {
      use checksum::{Checksum, Crc24OpenPgp};

      let result = sampler.run(data, |buf| {
        core::hint::black_box(Crc24OpenPgp::checksum(core::hint::black_box(buf)));
      });
      (Crc24OpenPgp::kernel_name_for_len(data.len()), result)
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
    CRC24_PARAMS
  }

  fn env_prefix(&self) -> &'static str {
    "RSCRYPTO_CRC24"
  }
}
