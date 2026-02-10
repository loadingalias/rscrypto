//! CRC-32 tunable implementations.
//!
//! This module provides [`Tunable`](crate::Tunable) implementations for CRC-32 algorithms:
//! - [`Crc32IeeeTunable`] - CRC-32 (IEEE/Ethernet)
//! - [`Crc32cTunable`] - CRC-32C (Castagnoli/iSCSI)
//!
//! These tunables support direct kernel benchmarking via the `bench` module's
//! kernel lookup functions, allowing the tuning engine to measure specific
//! kernel implementations directly rather than going through the library dispatch.

use alloc::{string::String, vec, vec::Vec};

use checksum::bench::{self, Crc32Kernel};
use platform::Caps;

use crate::{
  BenchResult, KernelSpec, KernelTier, TunableParam, TuneError,
  sampler::{Sampler, SamplerConfig},
};

// ─────────────────────────────────────────────────────────────────────────────
// Common Tunable Parameters
// ─────────────────────────────────────────────────────────────────────────────

// Must match `checksum::crc32::policy::CRC32_FOLD_BLOCK_BYTES`.
const CRC32_FOLD_BLOCK_BYTES: usize = 128;

/// Map generic threshold names to CRC-32 specific env var suffixes.
///
/// CRC-32 has a more complex tier system than CRC-64:
/// - Portable → Hardware CRC (on ARM/x86 with CRC32 instruction)
/// - Portable → PCLMUL folding (on x86 without CRC32)
/// - Hardware CRC → Fusion (on Intel x86_64)
/// - Fusion → VPCLMUL (on AVX-512 x86_64)
pub(crate) fn crc32_threshold_to_env_suffix(threshold_name: &str) -> Option<&'static str> {
  match threshold_name {
    "portable_bytewise_to_slice16" => Some("THRESHOLD_PORTABLE_BYTEWISE_TO_SLICE16"),
    // Policy-specific thresholds (match `checksum::crc32::config` env vars).
    "portable_to_hwcrc" => Some("THRESHOLD_PORTABLE_TO_HWCRC"),
    "hwcrc_to_fusion" => Some("THRESHOLD_HWCRC_TO_FUSION"),
    "fusion_to_avx512" => Some("THRESHOLD_FUSION_TO_AVX512"),
    "fusion_to_vpclmul" => Some("THRESHOLD_FUSION_TO_VPCLMUL"),

    // Back-compat with older generic naming used by the engine.
    "portable_to_simd" => Some("THRESHOLD_PORTABLE_TO_HWCRC"),
    "simd_to_wide" => Some("THRESHOLD_HWCRC_TO_FUSION"),
    // Min bytes per lane
    "min_bytes_per_lane" => Some("MIN_BYTES_PER_LANE"),
    // Streams
    "streams" => Some("STREAMS"),
    _ => None,
  }
}

#[inline]
#[must_use]
fn small_kernel_name_for_crc32(name: &str) -> Option<&'static str> {
  if name.starts_with("x86_64/pclmul") {
    return Some("x86_64/pclmul-small");
  }
  if name.starts_with("x86_64/vpclmul") {
    return Some("x86_64/vpclmul-small");
  }
  if name.starts_with("aarch64/sve2-pmull") {
    return Some("aarch64/sve2-pmull-small");
  }
  if name.starts_with("aarch64/pmull") || name.starts_with("aarch64/pmull-eor3") {
    return Some("aarch64/pmull-small");
  }
  None
}

/// Tunable parameters for CRC-32 algorithms.
const CRC32_PARAMS: &[TunableParam] = &[
  TunableParam::new(
    "portable_bytewise_to_slice16",
    "Bytes where slice-by-16 becomes faster than bytewise portable",
    1,
    256,
    64,
  ),
  TunableParam::new(
    "portable_to_hwcrc",
    "Bytes where hardware CRC becomes faster than portable",
    16,
    4096,
    64,
  ),
  TunableParam::new(
    "hwcrc_to_fusion",
    "Bytes where fusion/folding becomes faster than HWCRC",
    64,
    65536,
    1024,
  ),
  TunableParam::new(
    "fusion_to_vpclmul",
    "Bytes where VPCLMUL fusion becomes faster than baseline fusion",
    64,
    1_048_576,
    2048,
  ),
  TunableParam::new(
    "fusion_to_avx512",
    "Bytes where AVX-512 fusion becomes faster than baseline fusion",
    64,
    1_048_576,
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

/// Build available kernel specifications for CRC-32-IEEE on the current platform.
fn crc32_ieee_kernel_specs(caps: &Caps) -> Vec<KernelSpec> {
  let mut specs = vec![
    KernelSpec::new("reference", KernelTier::Reference, Caps::NONE),
    KernelSpec::new("portable/bytewise", KernelTier::Portable, Caps::NONE),
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
        "x86_64/vpclmul-small",
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
    if caps.has(aarch64::CRC_READY) {
      specs.push(KernelSpec::with_streams(
        "aarch64/hwcrc",
        KernelTier::Hardware,
        aarch64::CRC_READY,
        1,
        3,
      ));
    }
    if caps.has(aarch64::PMULL_READY) && caps.has(aarch64::CRC_READY) {
      specs.push(KernelSpec::new(
        "aarch64/pmull-small",
        KernelTier::Folding,
        aarch64::PMULL_READY,
      ));
      specs.push(KernelSpec::with_streams(
        "aarch64/pmull-v9s3x2e-s3",
        KernelTier::Folding,
        aarch64::PMULL_READY,
        1,
        3,
      ));
    }
    if caps.has(aarch64::PMULL_EOR3_READY) && caps.has(aarch64::CRC_READY) {
      specs.push(KernelSpec::with_streams(
        "aarch64/pmull-eor3-v9s3x2e-s3",
        KernelTier::Wide,
        aarch64::PMULL_EOR3_READY,
        1,
        3,
      ));
    }
    if caps.has(aarch64::SVE2_PMULL) && caps.has(aarch64::PMULL_READY) && caps.has(aarch64::CRC_READY) {
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

/// Build available kernel specifications for CRC-32C on the current platform.
fn crc32c_kernel_specs(caps: &Caps) -> Vec<KernelSpec> {
  let mut specs = vec![
    KernelSpec::new("reference", KernelTier::Reference, Caps::NONE),
    KernelSpec::new("portable/bytewise", KernelTier::Portable, Caps::NONE),
    KernelSpec::new("portable/slice16", KernelTier::Portable, Caps::NONE),
  ];

  #[cfg(target_arch = "x86_64")]
  {
    use platform::caps::x86;
    // CRC-32C has hardware CRC on x86_64 (SSE4.2)
    if caps.has(x86::CRC32C_READY) {
      specs.push(KernelSpec::with_streams(
        "x86_64/hwcrc",
        KernelTier::Hardware,
        x86::CRC32C_READY,
        1,
        8,
      ));
    }
    // Fusion kernels (hwcrc + pclmul)
    if caps.has(x86::CRC32C_READY) && caps.has(x86::PCLMUL_READY) {
      specs.push(KernelSpec::with_streams(
        "x86_64/fusion-sse-v4s3x3",
        KernelTier::Folding,
        x86::PCLMUL_READY,
        1,
        8,
      ));
    }
    if caps.has(x86::CRC32C_READY) && caps.has(x86::AVX512_READY) && caps.has(x86::PCLMUL_READY) {
      specs.push(KernelSpec::with_streams(
        "x86_64/fusion-avx512-v4s3x3",
        KernelTier::Wide,
        x86::AVX512_READY,
        1,
        8,
      ));
    }
    if caps.has(x86::CRC32C_READY) && caps.has(x86::VPCLMUL_READY) {
      specs.push(KernelSpec::with_streams(
        "x86_64/fusion-vpclmul-v3x2",
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
    if caps.has(aarch64::CRC_READY) {
      specs.push(KernelSpec::with_streams(
        "aarch64/hwcrc",
        KernelTier::Hardware,
        aarch64::CRC_READY,
        1,
        3,
      ));
    }
    if caps.has(aarch64::PMULL_READY) && caps.has(aarch64::CRC_READY) {
      specs.push(KernelSpec::new(
        "aarch64/pmull-small",
        KernelTier::Folding,
        aarch64::PMULL_READY,
      ));
      specs.push(KernelSpec::with_streams(
        "aarch64/pmull-v9s3x2e-s3",
        KernelTier::Folding,
        aarch64::PMULL_READY,
        1,
        3,
      ));
    }
    if caps.has(aarch64::PMULL_EOR3_READY) && caps.has(aarch64::CRC_READY) {
      specs.push(KernelSpec::with_streams(
        "aarch64/pmull-eor3-v9s3x2e-s3",
        KernelTier::Wide,
        aarch64::PMULL_EOR3_READY,
        1,
        3,
      ));
    }
    if caps.has(aarch64::SVE2_PMULL) && caps.has(aarch64::PMULL_READY) && caps.has(aarch64::CRC_READY) {
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

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32-IEEE Tunable
// ─────────────────────────────────────────────────────────────────────────────

/// Tunable implementation for CRC-32-IEEE.
///
/// Supports direct kernel benchmarking via `force_kernel()`, allowing
/// measurement of specific kernel implementations including PCLMUL and VPCLMUL
/// variants with different stream configurations.
pub struct Crc32IeeeTunable {
  /// Forced kernel name (None = auto).
  forced_kernel: Option<String>,
  /// Forced stream count (None = auto).
  forced_streams: Option<u8>,
  /// Cached kernel function (resolved from forced_kernel).
  cached_kernel: Option<Crc32Kernel>,
  /// Effective kernel name for reporting.
  effective_kernel_name: &'static str,
}

impl Crc32IeeeTunable {
  /// Create a new CRC-32-IEEE tunable in auto mode.
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
      // Try with stream suffix first if streams are forced
      let full_name = if let Some(streams) = self.forced_streams {
        kernel_name_with_streams(name, streams)
      } else {
        name.as_str()
      };

      if let Some(kernel) = bench::get_crc32_ieee_kernel(full_name) {
        self.cached_kernel = Some(kernel);
        self.effective_kernel_name = kernel.name;
      } else if let Some(kernel) = bench::get_crc32_ieee_kernel(name) {
        // Fallback to base name
        self.cached_kernel = Some(kernel);
        self.effective_kernel_name = kernel.name;
      } else {
        self.cached_kernel = None;
        self.effective_kernel_name = "unresolved";
      }
    } else {
      self.cached_kernel = None;
      self.effective_kernel_name = "auto";
    }
  }
}

impl Default for Crc32IeeeTunable {
  fn default() -> Self {
    Self::new()
  }
}

impl crate::Tunable for Crc32IeeeTunable {
  fn name(&self) -> &'static str {
    "crc32-ieee"
  }

  fn available_kernels(&self, caps: &Caps) -> Vec<KernelSpec> {
    crc32_ieee_kernel_specs(caps)
  }

  fn force_kernel(&mut self, name: &str) -> Result<(), TuneError> {
    let caps = platform::caps();
    let available = self.available_kernels(&caps);
    let base_name = strip_stream_suffix(name);

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
      let mut func = kernel.func;
      if data.len() < CRC32_FOLD_BLOCK_BYTES
        && let Some(small_name) = small_kernel_name_for_crc32(kernel.name)
        && let Some(small) = bench::get_crc32_ieee_kernel(small_name)
      {
        func = small.func;
      }
      let result = sampler.run(data, |buf| {
        core::hint::black_box(func(core::hint::black_box(!0u32), core::hint::black_box(buf)));
      });
      (kernel.name, result)
    } else {
      use checksum::{Checksum, Crc32};

      let result = sampler.run(data, |buf| {
        core::hint::black_box(Crc32::checksum(core::hint::black_box(buf)));
      });
      (Crc32::kernel_name_for_len(data.len()), result)
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
    CRC32_PARAMS
  }

  fn env_prefix(&self) -> &'static str {
    "RSCRYPTO_CRC32"
  }

  fn threshold_to_env_suffix(&self, threshold_name: &str) -> Option<&'static str> {
    crc32_threshold_to_env_suffix(threshold_name)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32C Tunable
// ─────────────────────────────────────────────────────────────────────────────

/// Tunable implementation for CRC-32C (Castagnoli).
///
/// Supports direct kernel benchmarking via `force_kernel()`, allowing
/// measurement of specific kernel implementations including hardware CRC,
/// fusion, and VPCLMUL variants with different stream configurations.
pub struct Crc32cTunable {
  /// Forced kernel name (None = auto).
  forced_kernel: Option<String>,
  /// Forced stream count (None = auto).
  forced_streams: Option<u8>,
  /// Cached kernel function (resolved from forced_kernel).
  cached_kernel: Option<Crc32Kernel>,
  /// Effective kernel name for reporting.
  effective_kernel_name: &'static str,
}

impl Crc32cTunable {
  /// Create a new CRC-32C tunable in auto mode.
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
      // Try with stream suffix first if streams are forced
      let full_name = if let Some(streams) = self.forced_streams {
        kernel_name_with_streams(name, streams)
      } else {
        name.as_str()
      };

      if let Some(kernel) = bench::get_crc32c_kernel(full_name) {
        self.cached_kernel = Some(kernel);
        self.effective_kernel_name = kernel.name;
      } else if let Some(kernel) = bench::get_crc32c_kernel(name) {
        // Fallback to base name
        self.cached_kernel = Some(kernel);
        self.effective_kernel_name = kernel.name;
      } else {
        self.cached_kernel = None;
        self.effective_kernel_name = "unresolved";
      }
    } else {
      self.cached_kernel = None;
      self.effective_kernel_name = "auto";
    }
  }
}

impl Default for Crc32cTunable {
  fn default() -> Self {
    Self::new()
  }
}

impl crate::Tunable for Crc32cTunable {
  fn name(&self) -> &'static str {
    "crc32c"
  }

  fn available_kernels(&self, caps: &Caps) -> Vec<KernelSpec> {
    crc32c_kernel_specs(caps)
  }

  fn force_kernel(&mut self, name: &str) -> Result<(), TuneError> {
    let caps = platform::caps();
    let available = self.available_kernels(&caps);
    let base_name = strip_stream_suffix(name);

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
      let mut func = kernel.func;
      if data.len() < CRC32_FOLD_BLOCK_BYTES
        && let Some(small_name) = small_kernel_name_for_crc32(kernel.name)
        && let Some(small) = bench::get_crc32c_kernel(small_name)
      {
        func = small.func;
      }
      let result = sampler.run(data, |buf| {
        core::hint::black_box(func(core::hint::black_box(!0u32), core::hint::black_box(buf)));
      });
      (kernel.name, result)
    } else {
      use checksum::{Checksum, Crc32C};

      let result = sampler.run(data, |buf| {
        core::hint::black_box(Crc32C::checksum(core::hint::black_box(buf)));
      });
      (Crc32C::kernel_name_for_len(data.len()), result)
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
    CRC32_PARAMS
  }

  fn env_prefix(&self) -> &'static str {
    "RSCRYPTO_CRC32C"
  }

  fn threshold_to_env_suffix(&self, threshold_name: &str) -> Option<&'static str> {
    crc32_threshold_to_env_suffix(threshold_name)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Get the kernel name with stream suffix (e.g., "x86_64/pclmul-4way").
fn kernel_name_with_streams(base: &str, streams: u8) -> &'static str {
  if streams <= 1 {
    return match base {
      "portable" | "portable/slice16" => "portable/slice16",
      "portable/bytewise" => "portable/bytewise",
      "reference" | "reference/bitwise" => "reference",
      _ => Box::leak(base.to_string().into_boxed_str()),
    };
  }

  if base == "portable" || base == "portable/slice16" {
    return "portable/slice16";
  }
  if base == "portable/bytewise" {
    return "portable/bytewise";
  }
  if base == "reference" || base == "reference/bitwise" {
    return "reference";
  }

  // If the caller provided a fully qualified stream-suffixed name already,
  // keep it.
  if base.contains("-way") {
    return Box::leak(base.to_string().into_boxed_str());
  }

  Box::leak(format!("{base}-{streams}way").into_boxed_str())
}

#[inline]
#[must_use]
fn strip_stream_suffix(name: &str) -> &str {
  name
    .strip_suffix("-2way")
    .or_else(|| name.strip_suffix("-3way"))
    .or_else(|| name.strip_suffix("-4way"))
    .or_else(|| name.strip_suffix("-7way"))
    .or_else(|| name.strip_suffix("-8way"))
    .unwrap_or(name)
}

#[cfg(test)]
mod tests {
  extern crate std;

  use super::*;

  #[test]
  fn strip_stream_suffix_preserves_versioned_kernel_names() {
    assert_eq!(
      strip_stream_suffix("aarch64/pmull-v9s3x2e-s3-2way"),
      "aarch64/pmull-v9s3x2e-s3"
    );
    assert_eq!(
      strip_stream_suffix("x86_64/fusion-sse-v4s3x3-4way"),
      "x86_64/fusion-sse-v4s3x3"
    );
    assert_eq!(
      strip_stream_suffix("x86_64/fusion-vpclmul-v3x2-8way"),
      "x86_64/fusion-vpclmul-v3x2"
    );
    assert_eq!(
      strip_stream_suffix("aarch64/pmull-v9s3x2e-s3"),
      "aarch64/pmull-v9s3x2e-s3"
    );
  }

  #[test]
  fn kernel_name_with_streams_appends_expected_suffix() {
    assert_eq!(kernel_name_with_streams("portable", 8), "portable/slice16");
    assert_eq!(kernel_name_with_streams("reference", 8), "reference");
    assert_eq!(kernel_name_with_streams("x86_64/pclmul", 4), "x86_64/pclmul-4way");
    assert_eq!(
      kernel_name_with_streams("x86_64/fusion-sse-v4s3x3", 4),
      "x86_64/fusion-sse-v4s3x3-4way"
    );
    assert_eq!(
      kernel_name_with_streams("aarch64/pmull-eor3-v9s3x2e-s3", 3),
      "aarch64/pmull-eor3-v9s3x2e-s3-3way"
    );
  }
}
