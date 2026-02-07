//! CRC-64 implementations.
//!
//! This module provides:
//! - [`Crc64`] - CRC-64-XZ (ECMA-182, used in XZ Utils)
//! - [`Crc64Nvme`] - CRC-64-NVME (NVMe specification)
//!
//! # Hardware Acceleration
//!
//! - x86_64: VPCLMULQDQ / PCLMULQDQ folding
//! - aarch64: PMULL folding
//! - Power: VPMSUMD folding
//! - s390x: VGFM folding
//! - riscv64: ZVBC (RVV vector CLMUL) / Zbc folding
//! - wasm32/wasm64: portable only (no CLMUL)

pub(crate) mod config;
// kernels is pub(crate) but needs internal access from bench module
pub(crate) mod kernels;
// portable is pub(crate) but needs internal access from bench module
pub(crate) mod portable;

#[cfg(feature = "alloc")]
pub mod kernel_test;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

// NOTE: The VPMSUMD CRC64 backend is currently implemented for little-endian
// POWER and supports both endiannesses (big-endian loads are normalized).
#[cfg(target_arch = "powerpc64")]
mod power;

#[cfg(target_arch = "s390x")]
mod s390x;

#[cfg(target_arch = "riscv64")]
mod riscv64;

// Re-export config types for public API (Crc64Force only used internally on SIMD archs)
#[allow(unused_imports)]
pub use config::{Crc64Config, Crc64Force};
// Re-export traits for test module (`use super::*`).
#[allow(unused_imports)]
pub(super) use traits::{Checksum, ChecksumCombine};

#[cfg(any(test, feature = "std"))]
use crate::common::reference::crc64_bitwise;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", test))]
use crate::common::tables::generate_crc64_tables_8;
use crate::common::tables::{CRC64_NVME_POLY, CRC64_XZ_POLY, generate_crc64_tables_16};
#[cfg(feature = "diag")]
use crate::diag::{Crc64Polynomial, Crc64SelectionDiag};

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Name Introspection
// ─────────────────────────────────────────────────────────────────────────────

/// Get the name of the CRC-64/XZ kernel that would be selected for a given buffer length.
///
/// This is primarily for diagnostics and testing. The actual kernel selection
/// happens via the dispatch module's pre-computed tables.
#[inline]
#[must_use]
pub(crate) fn crc64_xz_selected_kernel_name(len: usize) -> &'static str {
  let cfg = config::get();

  // Handle forced modes
  if cfg.effective_force == Crc64Force::Reference {
    return kernels::REFERENCE;
  }
  if cfg.effective_force == Crc64Force::Portable {
    return kernels::PORTABLE;
  }

  // For auto mode, return the specific kernel name from the dispatch table
  let table = crate::dispatch::active_table();
  table.select_set(len).crc64_xz_name
}

/// Get the name of the CRC-64/NVME kernel that would be selected for a given buffer length.
///
/// This is primarily for diagnostics and testing. The actual kernel selection
/// happens via the dispatch module's pre-computed tables.
#[inline]
#[must_use]
pub(crate) fn crc64_nvme_selected_kernel_name(len: usize) -> &'static str {
  let cfg = config::get();

  // Handle forced modes
  if cfg.effective_force == Crc64Force::Reference {
    return kernels::REFERENCE;
  }
  if cfg.effective_force == Crc64Force::Portable {
    return kernels::PORTABLE;
  }

  // For auto mode, return the specific kernel name from the dispatch table
  let table = crate::dispatch::active_table();
  table.select_set(len).crc64_nvme_name
}

// ─────────────────────────────────────────────────────────────────────────────
// Selection Diagnostics
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "diag")]
#[inline]
#[must_use]
pub(crate) fn diag_crc64_xz(len: usize) -> Crc64SelectionDiag {
  let tune = platform::tune();
  let cfg = config::get();
  let selected_kernel = crc64_xz_selected_kernel_name(len);

  // Dispatch handles size-based kernel selection internally
  let reason = if cfg.effective_force != Crc64Force::Auto {
    crate::diag::SelectionReason::Forced
  } else {
    crate::diag::SelectionReason::Auto
  };

  // Thresholds are now baked into dispatch tables; report dispatch boundaries
  let table = crate::dispatch::active_table();

  Crc64SelectionDiag {
    polynomial: Crc64Polynomial::Xz,
    len,
    tune_kind: tune.kind(),
    reason,
    effective_force: cfg.effective_force,
    policy_family: "dispatch",
    selected_kernel,
    selected_streams: 1,                         // Dispatch uses pre-computed optimal kernel
    portable_to_clmul: table.boundaries[0],      // xs_max boundary
    pclmul_to_vpclmul: table.boundaries[2],      // m_max boundary
    small_kernel_max_bytes: table.boundaries[1], // s_max boundary
    use_4x512: false,
    min_bytes_per_lane: usize::MAX,
  }
}

#[cfg(feature = "diag")]
#[inline]
#[must_use]
pub(crate) fn diag_crc64_nvme(len: usize) -> Crc64SelectionDiag {
  let tune = platform::tune();
  let cfg = config::get();
  let selected_kernel = crc64_nvme_selected_kernel_name(len);

  // Dispatch handles size-based kernel selection internally
  let reason = if cfg.effective_force != Crc64Force::Auto {
    crate::diag::SelectionReason::Forced
  } else {
    crate::diag::SelectionReason::Auto
  };

  // Thresholds are now baked into dispatch tables; report dispatch boundaries
  let table = crate::dispatch::active_table();

  Crc64SelectionDiag {
    polynomial: Crc64Polynomial::Nvme,
    len,
    tune_kind: tune.kind(),
    reason,
    effective_force: cfg.effective_force,
    policy_family: "dispatch",
    selected_kernel,
    selected_streams: 1,
    portable_to_clmul: table.boundaries[0],      // xs_max boundary
    pclmul_to_vpclmul: table.boundaries[2],      // m_max boundary
    small_kernel_max_bytes: table.boundaries[1], // s_max boundary
    use_4x512: false,
    min_bytes_per_lane: usize::MAX,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Wrappers
// ─────────────────────────────────────────────────────────────────────────────
//
// These wrap the portable/arch-specific implementations to match the Crc64Fn
// signature. Each wrapper bakes in the appropriate polynomial tables.

/// Portable kernel tables (pre-computed at compile time).
mod kernel_tables {
  use super::*;
  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", test))]
  pub static XZ_TABLES_8: [[u64; 256]; 8] = generate_crc64_tables_8(CRC64_XZ_POLY);
  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", test))]
  pub static NVME_TABLES_8: [[u64; 256]; 8] = generate_crc64_tables_8(CRC64_NVME_POLY);
  pub static XZ_TABLES_16: [[u64; 256]; 16] = generate_crc64_tables_16(CRC64_XZ_POLY);
  pub static NVME_TABLES_16: [[u64; 256]; 16] = generate_crc64_tables_16(CRC64_NVME_POLY);
}

/// CRC-64-XZ portable kernel wrapper.
#[cfg(any(test, feature = "std"))]
#[allow(dead_code)]
fn crc64_xz_portable(crc: u64, data: &[u8]) -> u64 {
  portable::crc64_slice16_xz(crc, data)
}

/// CRC-64-NVME portable kernel wrapper.
#[cfg(any(test, feature = "std"))]
#[allow(dead_code)]
fn crc64_nvme_portable(crc: u64, data: &[u8]) -> u64 {
  portable::crc64_slice16_nvme(crc, data)
}

/// CRC-64-XZ reference (bitwise) kernel wrapper.
///
/// This is the canonical reference implementation - obviously correct,
/// audit-friendly, and used for verification of all optimized paths.
#[cfg(any(test, feature = "std"))]
#[allow(dead_code)]
fn crc64_xz_reference(crc: u64, data: &[u8]) -> u64 {
  crc64_bitwise(CRC64_XZ_POLY, crc, data)
}

/// CRC-64-NVME reference (bitwise) kernel wrapper.
///
/// This is the canonical reference implementation - obviously correct,
/// audit-friendly, and used for verification of all optimized paths.
#[cfg(any(test, feature = "std"))]
#[allow(dead_code)]
fn crc64_nvme_reference(crc: u64, data: &[u8]) -> u64 {
  crc64_bitwise(CRC64_NVME_POLY, crc, data)
}

// Folding parameters (only used on SIMD architectures)
//
// - Small-buffer tier folds one 16B lane at a time.
// - Large-buffer tier folds 8×16B lanes per 128B block.

/// Default SIMD threshold for buffered CRC.
///
/// Buffered CRC uses this to decide when to flush accumulated small updates.
/// The dispatch system handles optimal kernel selection; this is a conservative
/// threshold for buffer flush decisions.
#[cfg(feature = "alloc")]
const CRC64_BUFFERED_THRESHOLD: usize = 64;

// ─────────────────────────────────────────────────────────────────────────────
// Auto Kernels (using new dispatch module)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-64-XZ dispatch - fast path using pre-resolved kernel tables.
///
/// Uses the empirically-optimal kernel for the current platform and buffer size.
/// When `std` is enabled, also respects `RSCRYPTO_CRC64_FORCE` env var for
/// debugging/testing specific kernel paths.
#[inline]
fn crc64_xz_dispatch(crc: u64, data: &[u8]) -> u64 {
  #[cfg(feature = "std")]
  {
    let cfg = config::get();
    if cfg.effective_force == Crc64Force::Reference {
      return crc64_xz_reference(crc, data);
    }
    if cfg.effective_force == Crc64Force::Portable {
      return crc64_xz_portable(crc, data);
    }
  }

  let table = crate::dispatch::active_table();
  let kernel = table.select_set(data.len()).crc64_xz;
  kernel(crc, data)
}

/// CRC-64/XZ vectored dispatch (processes multiple buffers in order).
#[inline]
fn crc64_xz_dispatch_vectored(mut crc: u64, bufs: &[&[u8]]) -> u64 {
  #[cfg(feature = "std")]
  {
    let cfg = config::get();
    if cfg.effective_force == Crc64Force::Reference {
      for &buf in bufs {
        if !buf.is_empty() {
          crc = crc64_xz_reference(crc, buf);
        }
      }
      return crc;
    }
    if cfg.effective_force == Crc64Force::Portable {
      for &buf in bufs {
        if !buf.is_empty() {
          crc = crc64_xz_portable(crc, buf);
        }
      }
      return crc;
    }
  }

  let table = crate::dispatch::active_table();
  let mut last_set: *const crate::dispatch::KernelSet = core::ptr::null();
  let mut kernel = table.xs.crc64_xz;

  for &buf in bufs {
    if buf.is_empty() {
      continue;
    }
    let set = table.select_set(buf.len());
    let set_ptr: *const crate::dispatch::KernelSet = core::ptr::from_ref(set);
    if set_ptr != last_set {
      last_set = set_ptr;
      kernel = set.crc64_xz;
    }
    crc = kernel(crc, buf);
  }

  crc
}

/// CRC-64-NVME dispatch - fast path using pre-resolved kernel tables.
///
/// Uses the empirically-optimal kernel for the current platform and buffer size.
/// When `std` is enabled, also respects `RSCRYPTO_CRC64_FORCE` env var for
/// debugging/testing specific kernel paths.
#[inline]
fn crc64_nvme_dispatch(crc: u64, data: &[u8]) -> u64 {
  #[cfg(feature = "std")]
  {
    let cfg = config::get();
    if cfg.effective_force == Crc64Force::Reference {
      return crc64_nvme_reference(crc, data);
    }
    if cfg.effective_force == Crc64Force::Portable {
      return crc64_nvme_portable(crc, data);
    }
  }

  let table = crate::dispatch::active_table();
  let kernel = table.select_set(data.len()).crc64_nvme;
  kernel(crc, data)
}

/// CRC-64/NVME vectored dispatch (processes multiple buffers in order).
#[inline]
fn crc64_nvme_dispatch_vectored(mut crc: u64, bufs: &[&[u8]]) -> u64 {
  #[cfg(feature = "std")]
  {
    let cfg = config::get();
    if cfg.effective_force == Crc64Force::Reference {
      for &buf in bufs {
        if !buf.is_empty() {
          crc = crc64_nvme_reference(crc, buf);
        }
      }
      return crc;
    }
    if cfg.effective_force == Crc64Force::Portable {
      for &buf in bufs {
        if !buf.is_empty() {
          crc = crc64_nvme_portable(crc, buf);
        }
      }
      return crc;
    }
  }

  let table = crate::dispatch::active_table();
  let mut last_set: *const crate::dispatch::KernelSet = core::ptr::null();
  let mut kernel = table.xs.crc64_nvme;

  for &buf in bufs {
    if buf.is_empty() {
      continue;
    }
    let set = table.select_set(buf.len());
    let set_ptr: *const crate::dispatch::KernelSet = core::ptr::from_ref(set);
    if set_ptr != last_set {
      last_set = set_ptr;
      kernel = set.crc64_nvme;
    }
    crc = kernel(crc, buf);
  }

  crc
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-64 Types
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-64-XZ checksum (ECMA-182).
///
/// Used in XZ Utils, 7-Zip, and other compression tools.
///
/// # Properties
///
/// - **Polynomial**: 0x42F0E1EBA9EA3693 (normal), 0xC96C5795D7870F42 (reflected)
/// - **Initial value**: 0xFFFFFFFFFFFFFFFF
/// - **Final XOR**: 0xFFFFFFFFFFFFFFFF
/// - **Reflect input/output**: Yes
///
/// # Performance Notes
///
/// For optimal throughput, prefer larger updates when possible:
///
/// | Update Size | Path | Notes |
/// |-------------|------|-------|
/// | < 32-128 bytes | Portable slice-by-8 | Threshold varies by CPU |
/// | ≥ 32-128 bytes | SIMD (PCLMULQDQ/PMULL) | Hardware accelerated |
///
/// The exact threshold is microarchitecture-specific:
/// - AMD Zen 4/5: 32 bytes (fast SIMD setup)
/// - Intel SPR: 128 bytes (ZMM warmup overhead)
/// - Apple M1-M5: 48 bytes (efficient PMULL)
///
/// For streaming many small chunks, consider using [`BufferedCrc64Xz`] which
/// accumulates data internally until reaching the SIMD threshold.
///
/// # Examples
///
/// ```rust
/// use checksum::{Checksum, Crc64Xz};
///
/// let crc = Crc64Xz::checksum(b"123456789");
/// assert_eq!(crc, 0x995DC9BBDF1939FA); // "123456789" test vector
/// ```
#[derive(Clone, Copy)]
pub struct Crc64 {
  state: u64,
}

/// Explicit name for the XZ CRC-64 variant (alias of [`Crc64`]).
pub type Crc64Xz = Crc64;

impl Crc64 {
  /// Pre-computed shift-by-8 matrix for combine.
  const SHIFT8_MATRIX: crate::common::combine::Gf2Matrix64 =
    crate::common::combine::generate_shift8_matrix_64(CRC64_XZ_POLY);

  /// Create a hasher to resume from a previous CRC value.
  #[inline]
  #[must_use]
  pub const fn resume(crc: u64) -> Self {
    Self { state: crc ^ !0 }
  }

  /// Get the name of the currently selected backend.
  #[must_use]
  pub fn backend_name() -> &'static str {
    let cfg = config::get();
    match cfg.effective_force {
      Crc64Force::Reference => kernels::REFERENCE,
      Crc64Force::Portable => kernels::PORTABLE,
      _ => crc64_xz_selected_kernel_name(1024), // representative size
    }
  }

  /// Get the effective CRC-64 configuration (force mode).
  #[must_use]
  pub fn config() -> Crc64Config {
    config::get()
  }

  /// Returns the kernel name that the selector would choose for `len`.
  #[must_use]
  pub fn kernel_name_for_len(len: usize) -> &'static str {
    crc64_xz_selected_kernel_name(len)
  }
}

impl traits::Checksum for Crc64 {
  const OUTPUT_SIZE: usize = 8;
  type Output = u64;

  #[inline]
  fn new() -> Self {
    Self { state: !0 }
  }

  #[inline]
  fn with_initial(initial: u64) -> Self {
    Self { state: initial ^ !0 }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.state = crc64_xz_dispatch(self.state, data);
  }

  #[inline]
  fn update_vectored(&mut self, bufs: &[&[u8]]) {
    self.state = crc64_xz_dispatch_vectored(self.state, bufs);
  }

  #[inline]
  fn finalize(&self) -> u64 {
    self.state ^ !0
  }

  #[inline]
  fn reset(&mut self) {
    self.state = !0;
  }
}

impl Default for Crc64 {
  fn default() -> Self {
    <Self as traits::Checksum>::new()
  }
}

impl traits::ChecksumCombine for Crc64 {
  fn combine(crc_a: u64, crc_b: u64, len_b: usize) -> u64 {
    crate::common::combine::combine_crc64(crc_a, crc_b, len_b, Self::SHIFT8_MATRIX)
  }
}

/// CRC-64-NVME checksum.
///
/// Used in the NVMe specification for data integrity.
///
/// # Properties
///
/// - **Polynomial**: 0xAD93D23594C93659 (normal), 0x9A6C9329AC4BC9B5 (reflected)
/// - **Initial value**: 0xFFFFFFFFFFFFFFFF
/// - **Final XOR**: 0xFFFFFFFFFFFFFFFF
/// - **Reflect input/output**: Yes
///
/// # Performance Notes
///
/// For optimal throughput, prefer larger updates when possible:
///
/// | Update Size | Path | Notes |
/// |-------------|------|-------|
/// | < 32-128 bytes | Portable slice-by-8 | Threshold varies by CPU |
/// | ≥ 32-128 bytes | SIMD (PCLMULQDQ/PMULL) | Hardware accelerated |
///
/// The exact threshold is microarchitecture-specific:
/// - AMD Zen 4/5: 32 bytes (fast SIMD setup)
/// - Intel SPR: 128 bytes (ZMM warmup overhead)
/// - Apple M1-M5: 48 bytes (efficient PMULL)
///
/// For streaming many small chunks, consider using [`BufferedCrc64Nvme`] which
/// accumulates data internally until reaching the SIMD threshold.
///
/// # Examples
///
/// ```rust
/// use checksum::{Checksum, Crc64Nvme};
///
/// let crc = Crc64Nvme::checksum(b"123456789");
/// assert_eq!(crc, 0xAE8B14860A799888); // "123456789" test vector
/// ```
#[derive(Clone, Copy)]
pub struct Crc64Nvme {
  state: u64,
}

impl Crc64Nvme {
  /// Pre-computed shift-by-8 matrix for combine.
  const SHIFT8_MATRIX: crate::common::combine::Gf2Matrix64 =
    crate::common::combine::generate_shift8_matrix_64(CRC64_NVME_POLY);

  /// Create a hasher to resume from a previous CRC value.
  #[inline]
  #[must_use]
  pub const fn resume(crc: u64) -> Self {
    Self { state: crc ^ !0 }
  }

  /// Get the name of the currently selected backend.
  #[must_use]
  pub fn backend_name() -> &'static str {
    let cfg = config::get();
    match cfg.effective_force {
      Crc64Force::Reference => kernels::REFERENCE,
      Crc64Force::Portable => kernels::PORTABLE,
      _ => crc64_nvme_selected_kernel_name(1024), // representative size
    }
  }

  /// Get the effective CRC-64 configuration (force mode).
  #[must_use]
  pub fn config() -> Crc64Config {
    config::get()
  }

  /// Returns the kernel name that the selector would choose for `len`.
  #[must_use]
  pub fn kernel_name_for_len(len: usize) -> &'static str {
    crc64_nvme_selected_kernel_name(len)
  }
}

impl traits::Checksum for Crc64Nvme {
  const OUTPUT_SIZE: usize = 8;
  type Output = u64;

  #[inline]
  fn new() -> Self {
    Self { state: !0 }
  }

  #[inline]
  fn with_initial(initial: u64) -> Self {
    Self { state: initial ^ !0 }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.state = crc64_nvme_dispatch(self.state, data);
  }

  #[inline]
  fn update_vectored(&mut self, bufs: &[&[u8]]) {
    self.state = crc64_nvme_dispatch_vectored(self.state, bufs);
  }

  #[inline]
  fn finalize(&self) -> u64 {
    self.state ^ !0
  }

  #[inline]
  fn reset(&mut self) {
    self.state = !0;
  }
}

impl Default for Crc64Nvme {
  fn default() -> Self {
    <Self as traits::Checksum>::new()
  }
}

impl traits::ChecksumCombine for Crc64Nvme {
  fn combine(crc_a: u64, crc_b: u64, len_b: usize) -> u64 {
    crate::common::combine::combine_crc64(crc_a, crc_b, len_b, Self::SHIFT8_MATRIX)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffered CRC-64 Wrappers (generated via macro)
// ─────────────────────────────────────────────────────────────────────────────

/// Buffer size for buffered CRC wrappers.
///
/// Chosen to be larger than the maximum SIMD threshold (256 bytes on Intel SPR)
/// while remaining cache-friendly.
#[cfg(feature = "alloc")]
const BUFFERED_CRC_BUFFER_SIZE: usize = 512;

#[cfg(feature = "alloc")]
define_buffered_crc! {
  /// A buffering wrapper around [`Crc64Xz`] for streaming small chunks.
  ///
  /// Use when you expect many small updates (< 64 bytes). This wrapper
  /// accumulates data internally until reaching the SIMD threshold, then
  /// flushes in batches for optimal throughput.
  ///
  /// # When to Use
  ///
  /// - Processing data from network sockets (typically 1-8 KB reads)
  /// - Streaming parsers that emit small tokens
  /// - Any workload with many small `update()` calls
  ///
  /// For large contiguous buffers, use [`Crc64Xz`] directly.
  ///
  /// # Examples
  ///
  /// ```rust
  /// use checksum::{BufferedCrc64Xz, Checksum};
  ///
  /// let mut hasher = BufferedCrc64Xz::new();
  /// let data = b"The quick brown fox jumps over the lazy dog";
  ///
  /// // Many small updates - internally buffered
  /// for byte in data.iter() {
  ///     hasher.update(&[*byte]);
  /// }
  ///
  /// let crc = hasher.finalize();
  /// assert_eq!(crc, checksum::Crc64Xz::checksum(data));
  /// ```
  pub struct BufferedCrc64<Crc64> {
    buffer_size: BUFFERED_CRC_BUFFER_SIZE,
    threshold_fn: || CRC64_BUFFERED_THRESHOLD,
  }
}

/// Explicit name for the XZ buffered CRC-64 variant (alias of [`BufferedCrc64`]).
#[cfg(feature = "alloc")]
pub type BufferedCrc64Xz = BufferedCrc64;

#[cfg(feature = "alloc")]
define_buffered_crc! {
  /// A buffering wrapper around [`Crc64Nvme`] for streaming small chunks.
  ///
  /// Use when you expect many small updates (< 64 bytes). This wrapper
  /// accumulates data internally until reaching the SIMD threshold, then
  /// flushes in batches for optimal throughput.
  ///
  /// # When to Use
  ///
  /// - Processing NVMe data from network/storage streams
  /// - Any workload with many small `update()` calls
  ///
  /// For large contiguous buffers, use [`Crc64Nvme`] directly.
  ///
  /// # Examples
  ///
  /// ```rust
  /// use checksum::{BufferedCrc64Nvme, Checksum};
  ///
  /// let mut hasher = BufferedCrc64Nvme::new();
  /// let data = b"The quick brown fox jumps over the lazy dog";
  ///
  /// // Many small updates - internally buffered
  /// for byte in data.iter() {
  ///     hasher.update(&[*byte]);
  /// }
  ///
  /// let crc = hasher.finalize();
  /// assert_eq!(crc, checksum::Crc64Nvme::checksum(data));
  /// ```
  pub struct BufferedCrc64Nvme<Crc64Nvme> {
    buffer_size: BUFFERED_CRC_BUFFER_SIZE,
    threshold_fn: || CRC64_BUFFERED_THRESHOLD,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Introspection
// ─────────────────────────────────────────────────────────────────────────────

impl crate::introspect::KernelIntrospect for Crc64 {
  fn kernel_name_for_len(len: usize) -> &'static str {
    Self::kernel_name_for_len(len)
  }

  fn backend_name() -> &'static str {
    Self::backend_name()
  }
}

impl crate::introspect::KernelIntrospect for Crc64Nvme {
  fn kernel_name_for_len(len: usize) -> &'static str {
    Self::kernel_name_for_len(len)
  }

  fn backend_name() -> &'static str {
    Self::backend_name()
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  extern crate std;

  use alloc::vec::Vec;

  use super::*;

  const TEST_DATA: &[u8] = b"123456789";

  #[test]
  fn test_crc64_xz_checksum() {
    // Standard test vector for CRC-64-XZ (ECMA-182)
    let crc = Crc64::checksum(TEST_DATA);
    assert_eq!(crc, 0x995DC9BBDF1939FA);
  }

  #[test]
  fn test_crc64_nvme_checksum() {
    // Standard test vector for CRC-64-NVME (per CRC RevEng catalog)
    // https://reveng.sourceforge.io/crc-catalogue/17plus.htm
    let crc = Crc64Nvme::checksum(TEST_DATA);
    assert_eq!(crc, 0xAE8B14860A799888);
  }

  #[test]
  fn test_crc64_xz_streaming() {
    let oneshot = Crc64::checksum(TEST_DATA);

    let mut hasher = Crc64::new();
    hasher.update(&TEST_DATA[..5]);
    hasher.update(&TEST_DATA[5..]);
    assert_eq!(hasher.finalize(), oneshot);
  }

  #[test]
  fn test_crc64_nvme_streaming() {
    let oneshot = Crc64Nvme::checksum(TEST_DATA);

    let mut hasher = Crc64Nvme::new();
    for chunk in TEST_DATA.chunks(3) {
      hasher.update(chunk);
    }
    assert_eq!(hasher.finalize(), oneshot);
  }

  #[test]
  fn test_crc64_xz_combine() {
    let data = b"hello world";
    let (a, b) = data.split_at(6);

    let crc_a = Crc64::checksum(a);
    let crc_b = Crc64::checksum(b);
    let combined = Crc64::combine(crc_a, crc_b, b.len());

    assert_eq!(combined, Crc64::checksum(data));
  }

  #[test]
  fn test_crc64_nvme_combine() {
    let data = b"hello world";
    let (a, b) = data.split_at(6);

    let crc_a = Crc64Nvme::checksum(a);
    let crc_b = Crc64Nvme::checksum(b);
    let combined = Crc64Nvme::combine(crc_a, crc_b, b.len());

    assert_eq!(combined, Crc64Nvme::checksum(data));
  }

  #[test]
  fn test_crc64_empty() {
    let crc = Crc64::checksum(&[]);
    assert_eq!(crc, 0);

    let crc = Crc64Nvme::checksum(&[]);
    assert_eq!(crc, 0);
  }

  #[test]
  fn test_crc64_combine_all_splits() {
    for split in 0..=TEST_DATA.len() {
      let (a, b) = TEST_DATA.split_at(split);
      let crc_a = Crc64::checksum(a);
      let crc_b = Crc64::checksum(b);
      let combined = Crc64::combine(crc_a, crc_b, b.len());
      assert_eq!(combined, Crc64::checksum(TEST_DATA), "Failed at split {split}");
    }
  }

  #[test]
  fn test_backend_name_not_empty() {
    assert!(!Crc64::backend_name().is_empty());
    assert!(!Crc64Nvme::backend_name().is_empty());
  }

  /// Test various buffer sizes to exercise both portable and SIMD paths.
  ///
  /// This test verifies that the tune-aware dispatch produces correct results
  /// regardless of whether the portable slice-by-8 or SIMD path is selected.
  #[test]
  fn test_crc64_various_lengths() {
    // Generate predictable test data
    let mut data = [0u8; 512];
    for (i, byte) in data.iter_mut().enumerate() {
      *byte = (i as u8).wrapping_mul(17).wrapping_add(i as u8);
    }

    // Test lengths around key thresholds:
    // - 0-15: always portable (below 16B lane minimum)
    // - 16-127: may use portable or small-lane CLMUL/PMULL (tune dependent)
    // - 128+: may use 128B folding (and VPCLMUL on wide-SIMD x86_64)
    let test_lengths = [
      0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 48, 63, 64, 65, 100, 127, 128, 200, 255, 256, 300, 400, 512,
    ];

    for &len in &test_lengths {
      let slice = &data[..len];

      // Compute with one-shot API
      let oneshot = Crc64::checksum(slice);

      // Verify streaming produces same result
      let mut hasher = Crc64::new();
      hasher.update(slice);
      let streamed = hasher.finalize();

      assert_eq!(oneshot, streamed, "Streaming mismatch at length {len}");

      // Verify chunked streaming produces same result
      let mut chunked = Crc64::new();
      for chunk in slice.chunks(37) {
        // Prime-sized chunks
        chunked.update(chunk);
      }
      assert_eq!(oneshot, chunked.finalize(), "Chunked mismatch at length {len}");
    }
  }

  /// Test streaming across size class boundaries.
  ///
  /// This ensures correct state handling when chunks span different
  /// size classes in the dispatch system.
  #[test]
  fn test_crc64_streaming_across_threshold() {
    // Use dispatch table boundaries for threshold
    let table = crate::dispatch::active_table();
    let threshold = table.boundaries[0]; // xs_max boundary

    // Generate test data larger than threshold
    let size = threshold + 128;
    let data: Vec<u8> = (0..size).map(|i| (i as u8).wrapping_mul(31)).collect();

    let oneshot = Crc64::checksum(&data);

    // Stream: small chunk (below threshold), then rest (above threshold)
    let mut hasher = Crc64::new();
    hasher.update(&data[..16]); // Small, uses xs kernel
    hasher.update(&data[16..]); // Large, may use different kernel
    assert_eq!(hasher.finalize(), oneshot, "Small-then-large streaming failed");

    // Stream: large chunk, then small chunk
    let mut hasher = Crc64::new();
    hasher.update(&data[..threshold + 64]); // Large
    hasher.update(&data[threshold + 64..]); // Small remainder
    assert_eq!(hasher.finalize(), oneshot, "Large-then-small streaming failed");

    // Many small chunks
    let mut hasher = Crc64::new();
    for chunk in data.chunks(17) {
      hasher.update(chunk);
    }
    assert_eq!(hasher.finalize(), oneshot, "Many small chunks failed");
  }

  /// Verify backend name reflects the selected kernel.
  #[test]
  fn test_backend_selection() {
    let name = Crc64::backend_name();
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = name;

    #[cfg(target_arch = "x86_64")]
    {
      let caps = platform::caps();
      let cfg = Crc64::config();

      if cfg.effective_force == Crc64Force::Portable {
        assert_eq!(name, "portable/slice16");
      } else if caps.has(platform::caps::x86::PCLMUL_READY) || caps.has(platform::caps::x86::VPCLMUL_READY) {
        // Now returns specific kernel names like "x86_64/vpclmul" or "x86_64/pclmul"
        assert!(name.starts_with("x86_64/"), "expected x86_64 kernel name, got: {name}");
        assert!(
          !name.contains("auto"),
          "expected specific kernel name, not auto: {name}"
        );
      } else {
        assert_eq!(name, "portable/slice16");
      }
    }

    #[cfg(target_arch = "aarch64")]
    {
      let caps = platform::caps();
      let cfg = Crc64::config();

      if cfg.effective_force == Crc64Force::Portable {
        assert_eq!(name, "portable/slice16");
      } else if caps.has(platform::caps::aarch64::PMULL_READY) {
        // Now returns specific kernel names like "aarch64/pmull-eor3" or "aarch64/pmull"
        assert!(
          name.starts_with("aarch64/"),
          "expected aarch64 kernel name, got: {name}"
        );
        assert!(
          !name.contains("auto"),
          "expected specific kernel name, not auto: {name}"
        );
      } else {
        assert_eq!(name, "portable/slice16");
      }
    }

    // kernel_name_for_len now returns specific kernel names from the dispatch table
    let len0_name = Crc64::kernel_name_for_len(0);
    assert!(
      !len0_name.is_empty(),
      "kernel_name_for_len should return a non-empty name"
    );
    assert!(
      !len0_name.contains("auto"),
      "expected specific kernel name, not auto: {len0_name}"
    );
  }

  /// If CRC-64 force env vars are set, assert that they are honored and exercise the tier.
  #[test]
  fn test_crc64_forced_kernel_smoke_from_env() {
    let Ok(force) = std::env::var("RSCRYPTO_CRC64_FORCE") else {
      return;
    };
    let force = force.trim();
    if force.is_empty() {
      return;
    }

    let cfg = Crc64::config();
    let len = 4096usize;

    // Execute both variants to ensure the selected tier doesn't trap.
    let data: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(13)).collect();
    let ours_xz = Crc64::checksum(&data);
    let ours_nvme = Crc64Nvme::checksum(&data);

    // Also validate correctness against the portable reference path.
    let portable_xz = portable::crc64_slice16_xz(!0, &data) ^ !0;
    let portable_nvme = portable::crc64_slice16_nvme(!0, &data) ^ !0;
    assert_eq!(ours_xz, portable_xz, "Forced tier produced incorrect CRC64-XZ");
    assert_eq!(ours_nvme, portable_nvme, "Forced tier produced incorrect CRC64-NVME");

    let kernel = Crc64::kernel_name_for_len(len);

    if force.eq_ignore_ascii_case("portable") {
      assert_eq!(cfg.requested_force, Crc64Force::Portable);
      assert_eq!(kernel, "portable/slice16");
    }

    #[cfg(target_arch = "x86_64")]
    {
      if force.eq_ignore_ascii_case("pclmul") {
        assert_eq!(cfg.requested_force, Crc64Force::Pclmul);
        // Kernel selection is handled by dispatch; just verify it's a pclmul variant
        if cfg.effective_force == Crc64Force::Pclmul {
          assert!(
            kernel.starts_with("x86_64/pclmul"),
            "Expected pclmul kernel, got {kernel}"
          );
        }
        return;
      }

      if force.eq_ignore_ascii_case("vpclmul") {
        assert_eq!(cfg.requested_force, Crc64Force::Vpclmul);
        if cfg.effective_force == Crc64Force::Vpclmul {
          assert!(
            kernel.starts_with("x86_64/vpclmul"),
            "Expected vpclmul kernel, got {kernel}"
          );
        }
      }
    }

    #[cfg(target_arch = "aarch64")]
    {
      if force.eq_ignore_ascii_case("pmull") {
        assert_eq!(cfg.requested_force, Crc64Force::Pmull);
        if cfg.effective_force == Crc64Force::Pmull {
          assert!(
            kernel.starts_with("aarch64/pmull"),
            "Expected pmull kernel, got {kernel}"
          );
        }
      }

      if force.eq_ignore_ascii_case("sve2-pmull")
        || force.eq_ignore_ascii_case("sve2")
        || force.eq_ignore_ascii_case("pmull-sve2")
      {
        assert_eq!(cfg.requested_force, Crc64Force::Sve2Pmull);
        if cfg.effective_force == Crc64Force::Sve2Pmull {
          assert!(
            kernel.starts_with("aarch64/sve2-pmull"),
            "Expected sve2-pmull kernel, got {kernel}"
          );
        }
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Buffered CRC Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_buffered_crc64_xz_matches_unbuffered() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let expected = Crc64::checksum(data);

    let mut buffered = BufferedCrc64::new();
    buffered.update(data);
    assert_eq!(buffered.finalize(), expected);
  }

  #[test]
  fn test_buffered_crc64_xz_single_byte_updates() {
    let data = b"123456789";
    let expected = Crc64::checksum(data);

    let mut buffered = BufferedCrc64::new();
    for byte in data.iter() {
      buffered.update(&[*byte]);
    }
    assert_eq!(buffered.finalize(), expected);
  }

  #[test]
  fn test_buffered_crc64_xz_mixed_sizes() {
    // Generate test data
    let mut data = [0u8; 1024];
    for (i, byte) in data.iter_mut().enumerate() {
      *byte = (i as u8).wrapping_mul(13);
    }
    let expected = Crc64::checksum(&data);

    let mut buffered = BufferedCrc64::new();
    let mut offset = 0;

    // Mix of small and large chunks
    let chunk_sizes = [1, 3, 7, 15, 31, 64, 128, 256, 300, 219];
    for &size in &chunk_sizes {
      let end = (offset + size).min(data.len());
      buffered.update(&data[offset..end]);
      offset = end;
      if offset >= data.len() {
        break;
      }
    }

    assert_eq!(buffered.finalize(), expected);
  }

  #[test]
  fn test_buffered_crc64_nvme_matches_unbuffered() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let expected = Crc64Nvme::checksum(data);

    let mut buffered = BufferedCrc64Nvme::new();
    buffered.update(data);
    assert_eq!(buffered.finalize(), expected);
  }

  #[test]
  fn test_buffered_crc64_nvme_single_byte_updates() {
    let data = b"123456789";
    let expected = Crc64Nvme::checksum(data);

    let mut buffered = BufferedCrc64Nvme::new();
    for byte in data.iter() {
      buffered.update(&[*byte]);
    }
    assert_eq!(buffered.finalize(), expected);
  }

  #[test]
  fn test_buffered_crc64_reset() {
    let data1 = b"hello";
    let data2 = b"world";

    let mut buffered = BufferedCrc64::new();
    buffered.update(data1);
    buffered.reset();
    buffered.update(data2);

    assert_eq!(buffered.finalize(), Crc64::checksum(data2));
  }

  #[test]
  fn test_buffered_crc64_empty() {
    let buffered = BufferedCrc64::new();
    assert_eq!(buffered.finalize(), Crc64::checksum(&[]));
  }

  #[test]
  fn test_buffered_crc64_finalize_is_idempotent() {
    let data = b"test data";
    let mut buffered = BufferedCrc64::new();
    buffered.update(data);

    let crc1 = buffered.finalize();
    let crc2 = buffered.finalize();
    assert_eq!(crc1, crc2, "finalize should be idempotent");
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Cross-Check Tests: Reference Implementation Verification
  // ─────────────────────────────────────────────────────────────────────────────

  /// Cross-check all kernels against the bitwise reference implementation.
  ///
  /// This is the canonical test that verifies correctness of all optimized
  /// implementations against the obviously-correct bitwise reference.
  mod cross_check {
    use alloc::{vec, vec::Vec};

    use super::*;
    use crate::common::{
      reference::crc64_bitwise,
      tables::{CRC64_NVME_POLY, CRC64_XZ_POLY},
    };

    /// Comprehensive test lengths covering all edge cases.
    const TEST_LENGTHS: &[usize] = &[
      // Empty
      0, // Single bytes
      1, // Small lengths (portable path)
      2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, // SIMD lane boundaries (16 bytes)
      16, 17, 31, 32, 33, // Near 64-byte boundaries
      63, 64, 65, // Near 128-byte fold block boundaries
      127, 128, 129, // Near 256-byte boundaries (VPCLMUL threshold)
      255, 256, 257, // Near 512-byte boundaries
      511, 512, 513, // Larger sizes for multi-stream kernels
      1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, // Very large (exercises 8-way and 4x512 kernels)
      8192, 16384, 32768, 65536,
    ];

    /// Prime-sized chunk patterns for streaming tests.
    const STREAMING_CHUNK_SIZES: &[usize] = &[1, 3, 7, 13, 17, 31, 37, 61, 127, 251];

    /// Generate deterministic test data of given length.
    fn generate_test_data(len: usize) -> Vec<u8> {
      (0..len)
        .map(|i| {
          // Use a mixing function to avoid patterns that might accidentally pass
          let i = i as u64;
          ((i.wrapping_mul(2654435761) ^ i.wrapping_mul(0x9E3779B97F4A7C15)) & 0xFF) as u8
        })
        .collect()
    }

    /// Compute CRC-64-XZ using the bitwise reference.
    fn reference_xz(data: &[u8]) -> u64 {
      crc64_bitwise(CRC64_XZ_POLY, !0u64, data) ^ !0u64
    }

    /// Compute CRC-64-NVME using the bitwise reference.
    fn reference_nvme(data: &[u8]) -> u64 {
      crc64_bitwise(CRC64_NVME_POLY, !0u64, data) ^ !0u64
    }

    #[test]
    fn cross_check_xz_all_lengths() {
      for &len in TEST_LENGTHS {
        let data = generate_test_data(len);
        let reference = reference_xz(&data);
        let actual = Crc64::checksum(&data);
        assert_eq!(
          actual, reference,
          "CRC64-XZ mismatch at len={len}: actual={actual:#018X}, reference={reference:#018X}"
        );
      }
    }

    #[test]
    fn cross_check_nvme_all_lengths() {
      for &len in TEST_LENGTHS {
        let data = generate_test_data(len);
        let reference = reference_nvme(&data);
        let actual = Crc64Nvme::checksum(&data);
        assert_eq!(
          actual, reference,
          "CRC64-NVME mismatch at len={len}: actual={actual:#018X}, reference={reference:#018X}"
        );
      }
    }

    #[test]
    fn cross_check_xz_all_single_bytes() {
      // Verify every possible single-byte input
      for byte in 0u8..=255 {
        let data = [byte];
        let reference = reference_xz(&data);
        let actual = Crc64::checksum(&data);
        assert_eq!(actual, reference, "CRC64-XZ single-byte mismatch for byte={byte:#04X}");
      }
    }

    #[test]
    fn cross_check_nvme_all_single_bytes() {
      for byte in 0u8..=255 {
        let data = [byte];
        let reference = reference_nvme(&data);
        let actual = Crc64Nvme::checksum(&data);
        assert_eq!(
          actual, reference,
          "CRC64-NVME single-byte mismatch for byte={byte:#04X}"
        );
      }
    }

    #[test]
    fn cross_check_xz_streaming_all_chunk_sizes() {
      let data = generate_test_data(4096);
      let reference = reference_xz(&data);

      for &chunk_size in STREAMING_CHUNK_SIZES {
        let mut hasher = Crc64::new();
        for chunk in data.chunks(chunk_size) {
          hasher.update(chunk);
        }
        let actual = hasher.finalize();
        assert_eq!(
          actual, reference,
          "CRC64-XZ streaming mismatch with chunk_size={chunk_size}"
        );
      }
    }

    #[test]
    fn cross_check_nvme_streaming_all_chunk_sizes() {
      let data = generate_test_data(4096);
      let reference = reference_nvme(&data);

      for &chunk_size in STREAMING_CHUNK_SIZES {
        let mut hasher = Crc64Nvme::new();
        for chunk in data.chunks(chunk_size) {
          hasher.update(chunk);
        }
        let actual = hasher.finalize();
        assert_eq!(
          actual, reference,
          "CRC64-NVME streaming mismatch with chunk_size={chunk_size}"
        );
      }
    }

    #[test]
    fn cross_check_xz_combine_all_splits() {
      let data = generate_test_data(1024);
      let reference = reference_xz(&data);

      // Test combine at every split point for smaller buffer
      let small_data = &data[..64];
      let small_ref = reference_xz(small_data);

      for split in 0..=small_data.len() {
        let (a, b) = small_data.split_at(split);
        let crc_a = Crc64::checksum(a);
        let crc_b = Crc64::checksum(b);
        let combined = Crc64::combine(crc_a, crc_b, b.len());
        assert_eq!(combined, small_ref, "CRC64-XZ combine mismatch at split={split}");
      }

      // Test strategic splits for larger buffer
      let strategic_splits = [0, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 512, 1024];
      for &split in &strategic_splits {
        if split > data.len() {
          continue;
        }
        let (a, b) = data.split_at(split);
        let crc_a = Crc64::checksum(a);
        let crc_b = Crc64::checksum(b);
        let combined = Crc64::combine(crc_a, crc_b, b.len());
        assert_eq!(
          combined, reference,
          "CRC64-XZ combine mismatch at strategic split={split}"
        );
      }
    }

    #[test]
    fn cross_check_nvme_combine_all_splits() {
      let data = generate_test_data(1024);
      let reference = reference_nvme(&data);

      // Test combine at every split point for smaller buffer
      let small_data = &data[..64];
      let small_ref = reference_nvme(small_data);

      for split in 0..=small_data.len() {
        let (a, b) = small_data.split_at(split);
        let crc_a = Crc64Nvme::checksum(a);
        let crc_b = Crc64Nvme::checksum(b);
        let combined = Crc64Nvme::combine(crc_a, crc_b, b.len());
        assert_eq!(combined, small_ref, "CRC64-NVME combine mismatch at split={split}");
      }

      // Test strategic splits for larger buffer
      let strategic_splits = [0, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 512, 1024];
      for &split in &strategic_splits {
        if split > data.len() {
          continue;
        }
        let (a, b) = data.split_at(split);
        let crc_a = Crc64Nvme::checksum(a);
        let crc_b = Crc64Nvme::checksum(b);
        let combined = Crc64Nvme::combine(crc_a, crc_b, b.len());
        assert_eq!(
          combined, reference,
          "CRC64-NVME combine mismatch at strategic split={split}"
        );
      }
    }

    #[test]
    fn cross_check_xz_unaligned_offsets() {
      // Test with data at various alignment offsets
      let mut buffer = vec![0u8; 4096 + 64];
      for (i, byte) in buffer.iter_mut().enumerate() {
        *byte = (((i as u64).wrapping_mul(17)) & 0xFF) as u8;
      }

      for offset in 0..16 {
        let data = &buffer[offset..offset + 1024];
        let reference = reference_xz(data);
        let actual = Crc64::checksum(data);
        assert_eq!(actual, reference, "CRC64-XZ unaligned mismatch at offset={offset}");
      }
    }

    #[test]
    fn cross_check_nvme_unaligned_offsets() {
      let mut buffer = vec![0u8; 4096 + 64];
      for (i, byte) in buffer.iter_mut().enumerate() {
        *byte = (((i as u64).wrapping_mul(17)) & 0xFF) as u8;
      }

      for offset in 0..16 {
        let data = &buffer[offset..offset + 1024];
        let reference = reference_nvme(data);
        let actual = Crc64Nvme::checksum(data);
        assert_eq!(actual, reference, "CRC64-NVME unaligned mismatch at offset={offset}");
      }
    }

    #[test]
    fn cross_check_xz_byte_at_a_time_streaming() {
      // Most stringent test: update one byte at a time
      let data = generate_test_data(256);
      let reference = reference_xz(&data);

      let mut hasher = Crc64::new();
      for &byte in &data {
        hasher.update(&[byte]);
      }
      let actual = hasher.finalize();
      assert_eq!(actual, reference, "CRC64-XZ byte-at-a-time streaming mismatch");
    }

    #[test]
    fn cross_check_nvme_byte_at_a_time_streaming() {
      let data = generate_test_data(256);
      let reference = reference_nvme(&data);

      let mut hasher = Crc64Nvme::new();
      for &byte in &data {
        hasher.update(&[byte]);
      }
      let actual = hasher.finalize();
      assert_eq!(actual, reference, "CRC64-NVME byte-at-a-time streaming mismatch");
    }

    /// Test that the reference kernel is accessible and works correctly.
    #[test]
    fn cross_check_reference_kernel_accessible() {
      // Force reference kernel via the wrapper functions
      let data = generate_test_data(1024);

      // Compute via reference kernel wrappers
      let xz_ref = crc64_xz_reference(!0u64, &data) ^ !0u64;
      let nvme_ref = crc64_nvme_reference(!0u64, &data) ^ !0u64;

      // Compute via bitwise reference directly
      let xz_direct = reference_xz(&data);
      let nvme_direct = reference_nvme(&data);

      assert_eq!(xz_ref, xz_direct, "XZ reference kernel mismatch");
      assert_eq!(nvme_ref, nvme_direct, "NVME reference kernel mismatch");
    }

    /// Test portable kernel matches reference.
    #[test]
    fn cross_check_portable_matches_reference() {
      for &len in TEST_LENGTHS {
        let data = generate_test_data(len);

        // XZ
        let portable_xz = portable::crc64_slice16_xz(!0u64, &data) ^ !0u64;
        let reference_xz_val = reference_xz(&data);
        assert_eq!(portable_xz, reference_xz_val, "XZ portable mismatch at len={len}");

        // NVME
        let portable_nvme = portable::crc64_slice16_nvme(!0u64, &data) ^ !0u64;
        let reference_nvme_val = reference_nvme(&data);
        assert_eq!(portable_nvme, reference_nvme_val, "NVME portable mismatch at len={len}");
      }
    }
  }
}
