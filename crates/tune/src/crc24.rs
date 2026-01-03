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

const CRC24_PARAMS: &[TunableParam] = &[TunableParam::new(
  "portable_to_clmul",
  "Bytes where SIMD becomes faster than portable",
  32,
  4096,
  128,
)];

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Specifications
// ─────────────────────────────────────────────────────────────────────────────

/// Map generic threshold names to CRC-24 specific env var suffixes.
///
/// This translates analysis-generated threshold names to the env var names
/// expected by the CRC-24 config module.
fn crc24_threshold_to_env_suffix(threshold_name: &str) -> Option<&'static str> {
  match threshold_name {
    "portable_to_simd" => Some("THRESHOLD_PORTABLE_TO_CLMUL"),
    "slice4_to_slice8" => Some("THRESHOLD_SLICE4_TO_SLICE8"),
    "simd_to_wide" => Some("THRESHOLD_PCLMUL_TO_VPCLMUL"),
    "min_bytes_per_lane" => Some("MIN_BYTES_PER_LANE"),
    "streams" => Some("STREAMS"),
    _ => None,
  }
}

fn crc24_kernel_specs(caps: &Caps) -> Vec<KernelSpec> {
  let mut specs = vec![
    KernelSpec::new("reference", KernelTier::Reference, Caps::NONE),
    KernelSpec::new("portable/slice4", KernelTier::Portable, Caps::NONE),
    KernelSpec::new("portable/slice8", KernelTier::Portable, Caps::NONE),
  ];

  #[cfg(target_arch = "x86_64")]
  {
    use platform::caps::x86;
    if caps.has(x86::PCLMUL_READY) {
      specs.push(KernelSpec::with_streams(
        "x86_64/pclmul",
        KernelTier::Folding,
        x86::PCLMUL_READY,
        1,
        8,
      ));
    }
    if caps.has(x86::VPCLMUL_READY) {
      specs.push(KernelSpec::with_streams(
        "x86_64/vpclmul",
        KernelTier::Wide,
        x86::VPCLMUL_READY,
        1,
        8,
      ));
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    use platform::caps::aarch64;
    if caps.has(aarch64::PMULL_READY) {
      specs.push(KernelSpec::with_streams(
        "aarch64/pmull",
        KernelTier::Folding,
        aarch64::PMULL_READY,
        1,
        3,
      ));
    }
  }

  #[cfg(target_arch = "powerpc64")]
  {
    use platform::caps::power;
    if caps.has(power::VPMSUM_READY) {
      specs.push(KernelSpec::with_streams(
        "power/vpmsum",
        KernelTier::Folding,
        power::VPMSUM_READY,
        1,
        8,
      ));
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    use platform::caps::s390x;
    if caps.has(s390x::VECTOR) {
      specs.push(KernelSpec::with_streams(
        "s390x/vgfm",
        KernelTier::Folding,
        s390x::VECTOR,
        1,
        4,
      ));
    }
  }

  #[cfg(target_arch = "riscv64")]
  {
    use platform::caps::riscv;
    if caps.has(riscv::ZBC) {
      specs.push(KernelSpec::with_streams(
        "riscv64/zbc",
        KernelTier::Folding,
        riscv::ZBC,
        1,
        4,
      ));
    }
    if caps.has(riscv::ZVBC) {
      specs.push(KernelSpec::with_streams(
        "riscv64/zvbc",
        KernelTier::Wide,
        riscv::ZVBC,
        1,
        4,
      ));
    }
  }

  #[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "s390x",
    target_arch = "riscv64"
  )))]
  {
    let _ = caps;
  }

  specs
}

fn kernel_name_with_streams(base: &str, streams: u8) -> &'static str {
  match (base, streams) {
    ("aarch64/pmull", 1) => "aarch64/pmull",
    ("aarch64/pmull", 2) => "aarch64/pmull-2way",
    ("aarch64/pmull", 3) => "aarch64/pmull-3way",

    // Reference and portable don't have stream variants.
    ("reference", _) => "reference",
    ("portable", _) | ("portable/slice4", _) | ("portable/slice8", _) => "portable",

    (b, 1) => Box::leak(b.to_string().into_boxed_str()),
    (b, _) if b.contains("-way") => Box::leak(b.to_string().into_boxed_str()),
    (b, s) => Box::leak(format!("{b}-{s}way").into_boxed_str()),
  }
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
  /// Forced stream count (None = auto).
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
      let full_name = if let Some(streams) = self.forced_streams {
        kernel_name_with_streams(name, streams)
      } else {
        name.as_str()
      };

      if let Some(kernel) = bench::get_crc24_openpgp_kernel(full_name) {
        self.cached_kernel = Some(kernel);
        self.effective_kernel_name = kernel.name;
      } else if let Some(kernel) = bench::get_crc24_openpgp_kernel(name) {
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
    let forced_base = self.forced_kernel.as_deref().and_then(|name| name.split('-').next());

    if forced_base == Some("aarch64/pmull") {
      if count > 3 {
        return Err(TuneError::InvalidStreamCount(count));
      }
    } else if forced_base == Some("x86_64/pclmul") || forced_base == Some("x86_64/vpclmul") {
      if !matches!(count, 1 | 2 | 4 | 7 | 8) {
        return Err(TuneError::InvalidStreamCount(count));
      }
    } else if count != 1 {
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

  fn threshold_to_env_suffix(&self, threshold_name: &str) -> Option<&'static str> {
    crc24_threshold_to_env_suffix(threshold_name)
  }
}
