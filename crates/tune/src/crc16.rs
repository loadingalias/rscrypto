//! CRC-16 tunable implementations.
//!
//! This module provides [`Tunable`](crate::Tunable) implementations for CRC-16 algorithms:
//! - [`Crc16CcittTunable`] - CRC-16/CCITT (X.25, HDLC)
//! - [`Crc16IbmTunable`] - CRC-16/IBM (ARC)
//!
//! These tunables support direct kernel benchmarking via the `bench` module's
//! kernel lookup functions, allowing the tuning engine to measure specific
//! kernel implementations directly rather than going through the library dispatch.

use alloc::{string::String, vec, vec::Vec};

use checksum::bench::{self, Crc16Kernel};
use platform::Caps;

use crate::{
  BenchResult, KernelSpec, KernelTier, TunableParam, TuneError,
  sampler::{Sampler, SamplerConfig},
};

// ─────────────────────────────────────────────────────────────────────────────
// Tunable Parameters
// ─────────────────────────────────────────────────────────────────────────────

const CRC16_PARAMS: &[TunableParam] = &[TunableParam::new(
  "portable_to_clmul",
  "Bytes where SIMD becomes faster than portable",
  32,
  4096,
  128,
)];

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Specifications
// ─────────────────────────────────────────────────────────────────────────────

/// Map generic threshold names to CRC-16 specific env var suffixes.
///
/// This translates analysis-generated threshold names to the env var names
/// expected by the CRC-16 config module.
fn crc16_threshold_to_env_suffix(threshold_name: &str) -> Option<&'static str> {
  match threshold_name {
    "portable_to_simd" => Some("THRESHOLD_PORTABLE_TO_CLMUL"),
    "slice4_to_slice8" => Some("THRESHOLD_SLICE4_TO_SLICE8"),
    "simd_to_wide" => Some("THRESHOLD_PCLMUL_TO_VPCLMUL"),
    "min_bytes_per_lane" => Some("MIN_BYTES_PER_LANE"),
    "streams" => Some("STREAMS"),
    _ => None,
  }
}

fn crc16_kernel_specs(caps: &Caps) -> Vec<KernelSpec> {
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
      specs.push(KernelSpec::new(
        "power/vpmsum",
        KernelTier::Folding,
        power::VPMSUM_READY,
      ));
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    use platform::caps::s390x;
    if caps.has(s390x::VECTOR) {
      specs.push(KernelSpec::new("s390x/vgfm", KernelTier::Folding, s390x::VECTOR));
    }
  }

  #[cfg(target_arch = "riscv64")]
  {
    use platform::caps::riscv;
    if caps.has(riscv::ZBC) {
      specs.push(KernelSpec::new("riscv64/zbc", KernelTier::Folding, riscv::ZBC));
    }
    if caps.has(riscv::ZVBC) {
      specs.push(KernelSpec::new("riscv64/zvbc", KernelTier::Wide, riscv::ZVBC));
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

/// Map a base kernel name and stream count to the full kernel name with suffix.
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

/// Validate stream count for a given kernel base name.
///
/// Different architectures support different stream counts:
/// - aarch64/pmull: 1-3 streams
/// - x86_64 PCLMUL/VPCLMUL: 1, 2, 4, 7, 8 streams
/// - Other kernels: single stream only
fn validate_stream_count(forced_kernel: Option<&str>, count: u8) -> Result<(), TuneError> {
  if count == 0 || count > 16 {
    return Err(TuneError::InvalidStreamCount(count));
  }

  let forced_base = forced_kernel.and_then(|name| name.split('-').next());

  let valid = match forced_base {
    Some("aarch64/pmull") => count <= 3,
    Some("x86_64/pclmul") | Some("x86_64/vpclmul") => matches!(count, 1 | 2 | 4 | 7 | 8),
    Some(_) => count == 1, // Unknown kernel only supports single stream
    None => true,          // No kernel forced, allow any count
  };

  if valid {
    Ok(())
  } else {
    Err(TuneError::InvalidStreamCount(count))
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-16/CCITT Tunable
// ─────────────────────────────────────────────────────────────────────────────

/// Tunable implementation for CRC-16/CCITT.
///
/// Supports direct kernel benchmarking via `force_kernel()`, allowing
/// measurement of specific kernel implementations.
pub struct Crc16CcittTunable {
  /// Forced kernel name (None = auto).
  forced_kernel: Option<String>,
  /// Forced stream count (None = auto).
  forced_streams: Option<u8>,
  /// Cached kernel function (resolved from forced_kernel).
  cached_kernel: Option<Crc16Kernel>,
  /// Effective kernel name for reporting.
  effective_kernel_name: &'static str,
}

impl Crc16CcittTunable {
  /// Create a new CRC-16/CCITT tunable in auto mode.
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

      if let Some(kernel) = bench::get_crc16_ccitt_kernel(full_name) {
        self.cached_kernel = Some(kernel);
        self.effective_kernel_name = kernel.name;
      } else if let Some(kernel) = bench::get_crc16_ccitt_kernel(name) {
        self.cached_kernel = Some(kernel);
        self.effective_kernel_name = kernel.name;
      }
    } else {
      self.cached_kernel = None;
      self.effective_kernel_name = "auto";
    }
  }
}

impl Default for Crc16CcittTunable {
  fn default() -> Self {
    Self::new()
  }
}

impl crate::Tunable for Crc16CcittTunable {
  fn name(&self) -> &'static str {
    "crc16-ccitt"
  }

  fn available_kernels(&self, caps: &Caps) -> Vec<KernelSpec> {
    crc16_kernel_specs(caps)
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
    validate_stream_count(self.forced_kernel.as_deref(), count)?;
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

  fn benchmark(&self, data: &[u8], config: &SamplerConfig) -> BenchResult {
    let sampler = Sampler::new(config);

    let (kernel_name, result) = if let Some(ref kernel) = self.cached_kernel {
      let func = kernel.func;
      let result = sampler.run(data, |buf| {
        core::hint::black_box(func(core::hint::black_box(0u16), core::hint::black_box(buf)));
      });
      (kernel.name, result)
    } else {
      use checksum::{Checksum, Crc16Ccitt};

      let result = sampler.run(data, |buf| {
        core::hint::black_box(Crc16Ccitt::checksum(core::hint::black_box(buf)));
      });
      (Crc16Ccitt::kernel_name_for_len(data.len()), result)
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
    CRC16_PARAMS
  }

  fn env_prefix(&self) -> &'static str {
    "RSCRYPTO_CRC16_CCITT"
  }

  fn threshold_to_env_suffix(&self, threshold_name: &str) -> Option<&'static str> {
    crc16_threshold_to_env_suffix(threshold_name)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-16/IBM Tunable
// ─────────────────────────────────────────────────────────────────────────────

/// Tunable implementation for CRC-16/IBM.
///
/// Supports direct kernel benchmarking via `force_kernel()`, allowing
/// measurement of specific kernel implementations.
pub struct Crc16IbmTunable {
  /// Forced kernel name (None = auto).
  forced_kernel: Option<String>,
  /// Forced stream count (None = auto).
  forced_streams: Option<u8>,
  /// Cached kernel function (resolved from forced_kernel).
  cached_kernel: Option<Crc16Kernel>,
  /// Effective kernel name for reporting.
  effective_kernel_name: &'static str,
}

impl Crc16IbmTunable {
  /// Create a new CRC-16/IBM tunable in auto mode.
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

      if let Some(kernel) = bench::get_crc16_ibm_kernel(full_name) {
        self.cached_kernel = Some(kernel);
        self.effective_kernel_name = kernel.name;
      } else if let Some(kernel) = bench::get_crc16_ibm_kernel(name) {
        self.cached_kernel = Some(kernel);
        self.effective_kernel_name = kernel.name;
      }
    } else {
      self.cached_kernel = None;
      self.effective_kernel_name = "auto";
    }
  }
}

impl Default for Crc16IbmTunable {
  fn default() -> Self {
    Self::new()
  }
}

impl crate::Tunable for Crc16IbmTunable {
  fn name(&self) -> &'static str {
    "crc16-ibm"
  }

  fn available_kernels(&self, caps: &Caps) -> Vec<KernelSpec> {
    crc16_kernel_specs(caps)
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
    validate_stream_count(self.forced_kernel.as_deref(), count)?;
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

  fn benchmark(&self, data: &[u8], config: &SamplerConfig) -> BenchResult {
    let sampler = Sampler::new(config);

    let (kernel_name, result) = if let Some(ref kernel) = self.cached_kernel {
      let func = kernel.func;
      let result = sampler.run(data, |buf| {
        core::hint::black_box(func(core::hint::black_box(0u16), core::hint::black_box(buf)));
      });
      (kernel.name, result)
    } else {
      use checksum::{Checksum, Crc16Ibm};

      let result = sampler.run(data, |buf| {
        core::hint::black_box(Crc16Ibm::checksum(core::hint::black_box(buf)));
      });
      (Crc16Ibm::kernel_name_for_len(data.len()), result)
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
    CRC16_PARAMS
  }

  fn env_prefix(&self) -> &'static str {
    "RSCRYPTO_CRC16_IBM"
  }

  fn threshold_to_env_suffix(&self, threshold_name: &str) -> Option<&'static str> {
    crc16_threshold_to_env_suffix(threshold_name)
  }
}
