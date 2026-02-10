//! CRC-64 tunable implementations.
//!
//! This module provides [`Tunable`](crate::Tunable) implementations for CRC-64 algorithms:
//! - [`Crc64XzTunable`] - CRC-64-XZ (ECMA-182)
//! - [`Crc64NvmeTunable`] - CRC-64-NVME
//!
//! These allow the unified tuning engine to benchmark and tune CRC-64
//! kernels across different platforms.

use alloc::{string::String, vec, vec::Vec};

use checksum::bench::{self, Crc64Kernel};
use platform::Caps;

use crate::{
  BenchResult, KernelSpec, KernelTier, TunableParam, TuneError,
  sampler::{Sampler, SamplerConfig},
};

// ─────────────────────────────────────────────────────────────────────────────
// Common Tunable Parameters
// ─────────────────────────────────────────────────────────────────────────────

/// Map generic threshold names to CRC-64 specific env var suffixes.
///
/// This translates the generic analysis names to the env var names
/// expected by the CRC-64 config module.
pub(crate) fn crc64_threshold_to_env_suffix(threshold_name: &str) -> Option<&'static str> {
  match threshold_name {
    // Portable → SIMD crossover
    "portable_to_simd" => Some("THRESHOLD_PORTABLE_TO_CLMUL"),
    // Small SIMD kernel window (within the SIMD tier)
    "small_kernel_max_bytes" => Some("THRESHOLD_SMALL_KERNEL_MAX_BYTES"),
    // SIMD → Wide crossover (PCLMUL → VPCLMUL on x86, PMULL → PMULL-EOR3 on ARM, etc.)
    "simd_to_wide" => Some("THRESHOLD_PCLMUL_TO_VPCLMUL"),
    // Min bytes per lane (no THRESHOLD_ prefix in config)
    "min_bytes_per_lane" => Some("MIN_BYTES_PER_LANE"),
    // Streams (no THRESHOLD_ prefix)
    "streams" => Some("STREAMS"),
    _ => None,
  }
}

/// Tunable parameters for CRC-64 algorithms.
const CRC64_PARAMS: &[TunableParam] = &[
  TunableParam::new(
    "portable_to_clmul",
    "Bytes where SIMD becomes faster than portable",
    16,
    4096,
    64,
  ),
  TunableParam::new(
    "small_kernel_max_bytes",
    "Max bytes where the small SIMD kernel is preferred",
    16,
    4096,
    512,
  ),
  TunableParam::new(
    "pclmul_to_vpclmul",
    "Bytes where VPCLMUL becomes faster than PCLMUL (x86_64 only)",
    128,
    65536,
    2048,
  ),
  TunableParam::new("streams", "Number of parallel folding streams", 1, 8, 4),
  TunableParam::new(
    "min_bytes_per_lane",
    "Minimum bytes per stream before multi-stream kicks in",
    64,
    65536,
    4096,
  ),
];

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Specifications
// ─────────────────────────────────────────────────────────────────────────────

/// Build available kernel specifications for CRC-64 on the current platform.
fn crc64_kernel_specs(caps: &Caps) -> Vec<KernelSpec> {
  let mut specs = vec![
    KernelSpec::new("reference", KernelTier::Reference, Caps::NONE),
    KernelSpec::new("portable/slice16", KernelTier::Portable, Caps::NONE),
  ];

  #[cfg(target_arch = "x86_64")]
  {
    use platform::caps::x86;
    if caps.has(x86::PCLMUL_READY) {
      specs.push(KernelSpec::new(
        "x86_64/pclmul-small",
        KernelTier::Folding,
        x86::PCLMUL_READY,
      ));
      specs.push(KernelSpec::with_streams(
        "x86_64/pclmul",
        KernelTier::Folding,
        x86::PCLMUL_READY,
        1,
        8,
      ));
    }
    if caps.has(x86::VPCLMUL_READY) {
      specs.push(KernelSpec::new(
        "x86_64/vpclmul-4x512",
        KernelTier::Wide,
        x86::VPCLMUL_READY,
      ));
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
      specs.push(KernelSpec::new(
        "aarch64/pmull-small",
        KernelTier::Folding,
        aarch64::PMULL_READY,
      ));
      specs.push(KernelSpec::with_streams(
        "aarch64/pmull",
        KernelTier::Folding,
        aarch64::PMULL_READY,
        1,
        3,
      ));
    }
    if caps.has(aarch64::PMULL_EOR3_READY) {
      specs.push(KernelSpec::with_streams(
        "aarch64/pmull-eor3",
        KernelTier::Wide,
        aarch64::PMULL_EOR3_READY,
        1,
        3,
      ));
    }
    if caps.has(aarch64::SVE2_PMULL) && caps.has(aarch64::PMULL_READY) {
      specs.push(KernelSpec::new(
        "aarch64/sve2-pmull-small",
        KernelTier::Wide,
        aarch64::SVE2_PMULL,
      ));
      specs.push(KernelSpec::with_streams(
        "aarch64/sve2-pmull",
        KernelTier::Wide,
        aarch64::SVE2_PMULL,
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

/// Get the kernel name with stream suffix (e.g., "x86_64/pclmul-4way").
fn kernel_name_with_streams(base: &str, streams: u8) -> &'static str {
  // Map kernel base names to their stream-suffixed versions
  match (base, streams) {
    // x86_64 PCLMUL
    ("x86_64/pclmul", 1) => "x86_64/pclmul",
    ("x86_64/pclmul", 2) => "x86_64/pclmul-2way",
    ("x86_64/pclmul", 4) => "x86_64/pclmul-4way",
    ("x86_64/pclmul", 7) => "x86_64/pclmul-7way",
    ("x86_64/pclmul", 8) => "x86_64/pclmul-8way",

    // x86_64 VPCLMUL
    ("x86_64/vpclmul", 1) => "x86_64/vpclmul",
    ("x86_64/vpclmul", 2) => "x86_64/vpclmul-2way",
    ("x86_64/vpclmul", 4) => "x86_64/vpclmul-4way",
    ("x86_64/vpclmul", 7) => "x86_64/vpclmul-7way",
    ("x86_64/vpclmul", 8) => "x86_64/vpclmul-8way",

    // aarch64 PMULL
    ("aarch64/pmull", 1) => "aarch64/pmull",
    ("aarch64/pmull", 2) => "aarch64/pmull-2way",
    ("aarch64/pmull", 3) => "aarch64/pmull-3way",

    // aarch64 PMULL+EOR3
    ("aarch64/pmull-eor3", 1) => "aarch64/pmull-eor3",
    ("aarch64/pmull-eor3", 2) => "aarch64/pmull-eor3-2way",
    ("aarch64/pmull-eor3", 3) => "aarch64/pmull-eor3-3way",

    // aarch64 SVE2 PMULL
    ("aarch64/sve2-pmull", 1) => "aarch64/sve2-pmull",
    ("aarch64/sve2-pmull", 2) => "aarch64/sve2-pmull-2way",
    ("aarch64/sve2-pmull", 3) => "aarch64/sve2-pmull-3way",

    // Power VPMSUM
    ("power/vpmsum", 1) => "power/vpmsum",
    ("power/vpmsum", 2) => "power/vpmsum-2way",
    ("power/vpmsum", 4) => "power/vpmsum-4way",
    ("power/vpmsum", 8) => "power/vpmsum-8way",

    // s390x VGFM
    ("s390x/vgfm", 1) => "s390x/vgfm",
    ("s390x/vgfm", 2) => "s390x/vgfm-2way",
    ("s390x/vgfm", 4) => "s390x/vgfm-4way",

    // riscv64 ZBC
    ("riscv64/zbc", 1) => "riscv64/zbc",
    ("riscv64/zbc", 2) => "riscv64/zbc-2way",
    ("riscv64/zbc", 4) => "riscv64/zbc-4way",

    // riscv64 ZVBC
    ("riscv64/zvbc", 1) => "riscv64/zvbc",
    ("riscv64/zvbc", 2) => "riscv64/zvbc-2way",
    ("riscv64/zvbc", 4) => "riscv64/zvbc-4way",

    // Reference and portable don't have stream variants
    ("reference", _) => "reference",
    ("portable", _) | ("portable/slice16", _) => "portable/slice16",

    // Fallback: return base unchanged (this might be a fully-qualified name already)
    _ => {
      // Try to return the original if it's already a complete name
      // This is a bit of a hack, but we can't allocate a String here
      if base.contains("-way") || base == "reference" || base.contains("slice") {
        // Leak a copy since we need 'static
        Box::leak(base.to_string().into_boxed_str())
      } else {
        "unknown"
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-64-XZ Tunable
// ─────────────────────────────────────────────────────────────────────────────

/// Tunable implementation for CRC-64-XZ.
///
/// This allows the tuning engine to benchmark CRC-64-XZ with different
/// kernel and stream configurations.
pub struct Crc64XzTunable {
  /// Forced kernel name (None = auto).
  forced_kernel: Option<String>,
  /// Forced stream count (None = auto).
  forced_streams: Option<u8>,
  /// Cached kernel function (resolved from forced_kernel).
  cached_kernel: Option<Crc64Kernel>,
  /// Effective kernel name for reporting.
  effective_kernel_name: &'static str,
}

impl Crc64XzTunable {
  /// Create a new CRC-64-XZ tunable in auto mode.
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
      // Try to find the kernel with stream suffix
      let full_name = if let Some(streams) = self.forced_streams {
        kernel_name_with_streams(name, streams)
      } else {
        name.as_str()
      };

      if let Some(kernel) = bench::get_crc64_xz_kernel(full_name) {
        self.cached_kernel = Some(kernel);
        self.effective_kernel_name = kernel.name;
      } else if let Some(kernel) = bench::get_crc64_xz_kernel(name) {
        // Fallback to base name
        self.cached_kernel = Some(kernel);
        self.effective_kernel_name = kernel.name;
      }
    } else {
      self.cached_kernel = None;
      self.effective_kernel_name = "auto";
    }
  }
}

impl Default for Crc64XzTunable {
  fn default() -> Self {
    Self::new()
  }
}

impl crate::Tunable for Crc64XzTunable {
  fn name(&self) -> &'static str {
    "crc64-xz"
  }

  fn available_kernels(&self, caps: &Caps) -> Vec<KernelSpec> {
    crc64_kernel_specs(caps)
  }

  fn force_kernel(&mut self, name: &str) -> Result<(), TuneError> {
    // Validate kernel name is in the available list (base name match)
    let caps = platform::caps();
    let available = self.available_kernels(&caps);
    let base_name = name.split('-').next().unwrap_or(name);

    // Check if the base kernel exists or the full name matches
    let valid = available.iter().any(|k| {
      k.name == name || k.name == base_name || name.starts_with(k.name) || name == "reference" || name == "portable"
    });

    if !valid {
      return Err(TuneError::KernelNotAvailable("kernel not available on this platform"));
    }

    self.forced_kernel = Some(name.to_string());
    self.resolve_kernel();
    if self.cached_kernel.is_none() {
      self.forced_kernel = None;
      self.forced_streams = None;
      return Err(TuneError::KernelNotAvailable(
        "kernel name did not resolve to a bench kernel",
      ));
    }
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

  fn benchmark(&self, data: &[u8], config: &SamplerConfig) -> BenchResult {
    let sampler = Sampler::new(config);

    // Use the forced kernel if available, otherwise fall back to auto
    let (kernel_name, result) = if let Some(ref kernel) = self.cached_kernel {
      let func = kernel.func;

      let result = sampler.run(data, |buf| {
        core::hint::black_box(func(core::hint::black_box(!0u64), core::hint::black_box(buf)));
      });
      (kernel.name, result)
    } else {
      // Auto mode: use library dispatch
      use checksum::{Checksum, Crc64Xz};

      let result = sampler.run(data, |buf| {
        core::hint::black_box(Crc64Xz::checksum(core::hint::black_box(buf)));
      });
      (Crc64Xz::kernel_name_for_len(data.len()), result)
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
    CRC64_PARAMS
  }

  fn env_prefix(&self) -> &'static str {
    "RSCRYPTO_CRC64"
  }

  fn threshold_to_env_suffix(&self, threshold_name: &str) -> Option<&'static str> {
    crc64_threshold_to_env_suffix(threshold_name)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-64-NVME Tunable
// ─────────────────────────────────────────────────────────────────────────────

/// Tunable implementation for CRC-64-NVME.
///
/// This allows the tuning engine to benchmark CRC-64-NVME with different
/// kernel and stream configurations.
pub struct Crc64NvmeTunable {
  /// Forced kernel name (None = auto).
  forced_kernel: Option<String>,
  /// Forced stream count (None = auto).
  forced_streams: Option<u8>,
  /// Cached kernel function (resolved from forced_kernel).
  cached_kernel: Option<Crc64Kernel>,
  /// Effective kernel name for reporting.
  effective_kernel_name: &'static str,
}

impl Crc64NvmeTunable {
  /// Create a new CRC-64-NVME tunable in auto mode.
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

      if let Some(kernel) = bench::get_crc64_nvme_kernel(full_name) {
        self.cached_kernel = Some(kernel);
        self.effective_kernel_name = kernel.name;
      } else if let Some(kernel) = bench::get_crc64_nvme_kernel(name) {
        self.cached_kernel = Some(kernel);
        self.effective_kernel_name = kernel.name;
      }
    } else {
      self.cached_kernel = None;
      self.effective_kernel_name = "auto";
    }
  }
}

impl Default for Crc64NvmeTunable {
  fn default() -> Self {
    Self::new()
  }
}

impl crate::Tunable for Crc64NvmeTunable {
  fn name(&self) -> &'static str {
    "crc64-nvme"
  }

  fn available_kernels(&self, caps: &Caps) -> Vec<KernelSpec> {
    crc64_kernel_specs(caps)
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
    if self.cached_kernel.is_none() {
      self.forced_kernel = None;
      self.forced_streams = None;
      return Err(TuneError::KernelNotAvailable(
        "kernel name did not resolve to a bench kernel",
      ));
    }
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

  fn benchmark(&self, data: &[u8], config: &SamplerConfig) -> BenchResult {
    let sampler = Sampler::new(config);

    let (kernel_name, result) = if let Some(ref kernel) = self.cached_kernel {
      let func = kernel.func;

      let result = sampler.run(data, |buf| {
        core::hint::black_box(func(core::hint::black_box(!0u64), core::hint::black_box(buf)));
      });
      (kernel.name, result)
    } else {
      use checksum::{Checksum, Crc64Nvme};

      let result = sampler.run(data, |buf| {
        core::hint::black_box(Crc64Nvme::checksum(core::hint::black_box(buf)));
      });
      (Crc64Nvme::kernel_name_for_len(data.len()), result)
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
    CRC64_PARAMS
  }

  fn env_prefix(&self) -> &'static str {
    "RSCRYPTO_CRC64"
  }

  fn threshold_to_env_suffix(&self, threshold_name: &str) -> Option<&'static str> {
    crc64_threshold_to_env_suffix(threshold_name)
  }
}
