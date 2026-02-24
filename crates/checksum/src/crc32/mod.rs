//! CRC-32 implementations (IEEE and Castagnoli).
//!
//! This module provides:
//! - [`Crc32`] - CRC-32 (IEEE, Ethernet)
//! - [`Crc32C`] - CRC-32C (Castagnoli, iSCSI)
//!
//! # Hardware Acceleration
//!
//! - x86_64: SSE4.2 `crc32` (CRC-32C only)
//! - x86_64: PCLMULQDQ folding (CRC-32 / IEEE)
//! - aarch64: ARMv8 CRC extension (CRC-32 and CRC-32C)
//! - Power: VPMSUMD folding (CRC-32 and CRC-32C)
//! - s390x: VGFM folding (CRC-32 and CRC-32C)
//! - riscv64: ZVBC (RVV vector CLMUL) / Zbc folding (CRC-32 and CRC-32C)
//! - wasm32/wasm64: portable only (no CRC32/CLMUL instructions)
//!
//! # Quick Start
//!
//! ```rust
//! use checksum::{Checksum, ChecksumCombine, Crc32, Crc32C};
//!
//! let data = b"123456789";
//!
//! // One-shot
//! assert_eq!(Crc32::checksum(data), 0xCBF4_3926);
//! assert_eq!(Crc32C::checksum(data), 0xE306_9283);
//!
//! // Streaming
//! let mut hasher = Crc32::new();
//! hasher.update(b"1234");
//! hasher.update(b"56789");
//! assert_eq!(hasher.finalize(), Crc32::checksum(data));
//!
//! // Combine: crc(A || B) == combine(crc(A), crc(B), len(B))
//! let (a, b) = data.split_at(4);
//! let combined = Crc32::combine(Crc32::checksum(a), Crc32::checksum(b), b.len());
//! assert_eq!(combined, Crc32::checksum(data));
//! ```

#[cfg(any(target_arch = "powerpc64", target_arch = "s390x", target_arch = "riscv64"))]
mod clmul;
pub(crate) mod config;
pub(crate) mod kernels;
pub(crate) mod portable;

#[cfg(feature = "alloc")]
pub mod kernel_test;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "powerpc64")]
mod power;

#[cfg(target_arch = "s390x")]
mod s390x;

#[cfg(target_arch = "riscv64")]
mod riscv64;

#[allow(unused_imports)]
pub use config::{Crc32Config, Crc32Force};
#[allow(unused_imports)]
pub(super) use traits::{Checksum, ChecksumCombine};

#[cfg(any(test, feature = "std"))]
use crate::common::reference::crc32_bitwise;
use crate::common::{
  combine::{Gf2Matrix32, generate_shift8_matrix_32},
  tables::{CRC32_IEEE_POLY, CRC32C_POLY, generate_crc32_tables_16},
};
#[cfg(feature = "diag")]
use crate::diag::{Crc32Polynomial, Crc32SelectionDiag};

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Tables (compile-time)
// ─────────────────────────────────────────────────────────────────────────────

/// Portable kernel tables (pre-computed at compile time).
mod kernel_tables {
  use super::*;
  pub static IEEE_TABLES_16: [[u32; 256]; 16] = generate_crc32_tables_16(CRC32_IEEE_POLY);
  pub static CRC32C_TABLES_16: [[u32; 256]; 16] = generate_crc32_tables_16(CRC32C_POLY);
}

/// Block size for CRC-32 folding operations.
#[allow(dead_code)]
pub(crate) const CRC32_FOLD_BLOCK_BYTES: usize = 128;

// ─────────────────────────────────────────────────────────────────────────────
// Portable Kernel Wrappers
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(any(test, feature = "std"))]
#[allow(dead_code)]
fn crc32_portable(crc: u32, data: &[u8]) -> u32 {
  const THRESHOLD: usize = 64;
  if data.len() < THRESHOLD {
    portable::crc32_bytewise_ieee(crc, data)
  } else {
    portable::crc32_slice16_ieee(crc, data)
  }
}

#[cfg(any(test, feature = "std"))]
#[allow(dead_code)]
fn crc32c_portable(crc: u32, data: &[u8]) -> u32 {
  const THRESHOLD: usize = 64;
  if data.len() < THRESHOLD {
    portable::crc32c_bytewise(crc, data)
  } else {
    portable::crc32c_slice16(crc, data)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reference Kernel Wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32 (IEEE) reference (bitwise) kernel wrapper.
///
/// This is the canonical reference implementation - obviously correct,
/// audit-friendly, and used for verification of all optimized paths.
#[cfg(any(test, feature = "std"))]
#[allow(dead_code)]
fn crc32_reference(crc: u32, data: &[u8]) -> u32 {
  crc32_bitwise(CRC32_IEEE_POLY, crc, data)
}

/// CRC-32C (Castagnoli) reference (bitwise) kernel wrapper.
///
/// This is the canonical reference implementation - obviously correct,
/// audit-friendly, and used for verification of all optimized paths.
#[cfg(any(test, feature = "std"))]
#[allow(dead_code)]
fn crc32c_reference(crc: u32, data: &[u8]) -> u32 {
  crc32_bitwise(CRC32C_POLY, crc, data)
}

// Buffered wrappers use this to decide when to flush/process in larger chunks.
//
// We want a value that is:
// - Small enough to avoid excessive buffering latency
// - Large enough to clear the "portable is still fastest" region on each arch
#[cfg(feature = "alloc")]
#[inline]
#[must_use]
#[allow(dead_code)]
fn crc32_buffered_threshold() -> usize {
  crc32_buffered_threshold_impl()
}

#[cfg(all(feature = "alloc", target_arch = "x86_64"))]
#[inline]
#[must_use]
fn crc32_buffered_threshold_impl() -> usize {
  const THRESHOLD: usize = 64;
  THRESHOLD
}

#[cfg(all(feature = "alloc", target_arch = "aarch64"))]
#[inline]
#[must_use]
fn crc32_buffered_threshold_impl() -> usize {
  const THRESHOLD: usize = 64;
  THRESHOLD
}

#[cfg(all(feature = "alloc", not(any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[inline]
#[must_use]
fn crc32_buffered_threshold_impl() -> usize {
  const THRESHOLD: usize = 64;
  THRESHOLD
}

#[cfg(feature = "alloc")]
#[inline]
#[must_use]
#[allow(dead_code)]
fn crc32c_buffered_threshold() -> usize {
  const THRESHOLD: usize = 64;
  THRESHOLD
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Name Introspection
// ─────────────────────────────────────────────────────────────────────────────

/// Get the name of the kernel that would be selected for a given buffer length.
#[inline]
#[must_use]
pub(crate) fn crc32_selected_kernel_name(len: usize) -> &'static str {
  let cfg = config::get();

  if cfg.effective_force == Crc32Force::Reference {
    return kernels::REFERENCE;
  }
  if cfg.effective_force == Crc32Force::Portable {
    return kernels::PORTABLE;
  }

  let table = crate::dispatch::active_table();
  table.select_set(len).crc32_ieee_name
}

/// Get the name of the kernel that would be selected for a given buffer length.
#[inline]
#[must_use]
pub(crate) fn crc32c_selected_kernel_name(len: usize) -> &'static str {
  let cfg = config::get();

  if cfg.effective_force == Crc32Force::Reference {
    return kernels::REFERENCE;
  }
  if cfg.effective_force == Crc32Force::Portable {
    return kernels::PORTABLE;
  }

  let table = crate::dispatch::active_table();
  table.select_set(len).crc32c_name
}

// ─────────────────────────────────────────────────────────────────────────────
// Selection Diagnostics
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "diag")]
#[inline]
#[must_use]
pub(crate) fn diag_crc32_ieee(len: usize) -> Crc32SelectionDiag {
  let tune = platform::tune();
  let cfg = config::get();
  let selected_kernel = crc32_selected_kernel_name(len);

  let reason = if cfg.effective_force != Crc32Force::Auto {
    crate::diag::SelectionReason::Forced
  } else {
    crate::diag::SelectionReason::Auto
  };

  let table = crate::dispatch::active_table();
  let boundary = if !table.boundaries.is_empty() {
    table.boundaries[0]
  } else {
    64
  };

  Crc32SelectionDiag {
    polynomial: Crc32Polynomial::Ieee,
    len,
    tune_kind: tune.kind(),
    reason,
    effective_force: cfg.effective_force,
    policy_family: "dispatch",
    selected_kernel,
    selected_streams: 1,
    portable_to_hwcrc: boundary,
    hwcrc_to_fusion: boundary,
    fusion_to_avx512: usize::MAX,
    fusion_to_vpclmul: usize::MAX,
    min_bytes_per_lane: usize::MAX,
    memory_bound: false,
    has_hwcrc: false,
    has_fusion: false,
    has_vpclmul: false,
    has_avx512: false,
    has_eor3: false,
    has_sve2: false,
  }
}

#[cfg(feature = "diag")]
#[inline]
#[must_use]
pub(crate) fn diag_crc32c(len: usize) -> Crc32SelectionDiag {
  let tune = platform::tune();
  let cfg = config::get();
  let selected_kernel = crc32c_selected_kernel_name(len);

  let reason = if cfg.effective_force != Crc32Force::Auto {
    crate::diag::SelectionReason::Forced
  } else {
    crate::diag::SelectionReason::Auto
  };

  let table = crate::dispatch::active_table();
  let boundary = if !table.boundaries.is_empty() {
    table.boundaries[0]
  } else {
    64
  };

  Crc32SelectionDiag {
    polynomial: Crc32Polynomial::Castagnoli,
    len,
    tune_kind: tune.kind(),
    reason,
    effective_force: cfg.effective_force,
    policy_family: "dispatch",
    selected_kernel,
    selected_streams: 1,
    portable_to_hwcrc: boundary,
    hwcrc_to_fusion: boundary,
    fusion_to_avx512: usize::MAX,
    fusion_to_vpclmul: usize::MAX,
    min_bytes_per_lane: usize::MAX,
    memory_bound: false,
    has_hwcrc: false,
    has_fusion: false,
    has_vpclmul: false,
    has_avx512: false,
    has_eor3: false,
    has_sve2: false,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto Kernels (using new dispatch module)
// ─────────────────────────────────────────────────────────────────────────────

type Crc32DispatchFn = crate::dispatchers::Crc32Fn;
type Crc32DispatchVectoredFn = fn(u32, &[&[u8]]) -> u32;

#[inline]
fn crc32_apply_kernel_vectored(mut crc: u32, bufs: &[&[u8]], kernel: Crc32DispatchFn) -> u32 {
  for &buf in bufs {
    if !buf.is_empty() {
      crc = kernel(crc, buf);
    }
  }
  crc
}

#[inline]
fn crc32_dispatch_auto(crc: u32, data: &[u8]) -> u32 {
  let table = crate::dispatch::active_table();
  let kernel = table.select_set(data.len()).crc32_ieee;
  kernel(crc, data)
}

#[inline]
fn crc32_dispatch_auto_vectored(mut crc: u32, bufs: &[&[u8]]) -> u32 {
  let table = crate::dispatch::active_table();
  let mut last_set: *const crate::dispatch::KernelSet = core::ptr::null();
  let mut kernel = table.xs.crc32_ieee;

  for &buf in bufs {
    if buf.is_empty() {
      continue;
    }
    let set = table.select_set(buf.len());
    let set_ptr: *const crate::dispatch::KernelSet = core::ptr::from_ref(set);
    if set_ptr != last_set {
      last_set = set_ptr;
      kernel = set.crc32_ieee;
    }
    crc = kernel(crc, buf);
  }

  crc
}

#[inline]
fn crc32c_dispatch_auto(crc: u32, data: &[u8]) -> u32 {
  let table = crate::dispatch::active_table();
  let kernel = table.select_set(data.len()).crc32c;
  kernel(crc, data)
}

#[inline]
fn crc32c_dispatch_auto_vectored(mut crc: u32, bufs: &[&[u8]]) -> u32 {
  let table = crate::dispatch::active_table();
  let mut last_set: *const crate::dispatch::KernelSet = core::ptr::null();
  let mut kernel = table.xs.crc32c;

  for &buf in bufs {
    if buf.is_empty() {
      continue;
    }
    let set = table.select_set(buf.len());
    let set_ptr: *const crate::dispatch::KernelSet = core::ptr::from_ref(set);
    if set_ptr != last_set {
      last_set = set_ptr;
      kernel = set.crc32c;
    }
    crc = kernel(crc, buf);
  }

  crc
}

#[cfg(feature = "std")]
#[inline]
fn crc32_dispatch_reference(crc: u32, data: &[u8]) -> u32 {
  crc32_reference(crc, data)
}

#[cfg(feature = "std")]
#[inline]
fn crc32_dispatch_portable(crc: u32, data: &[u8]) -> u32 {
  crc32_portable(crc, data)
}

#[cfg(feature = "std")]
#[inline]
fn crc32_dispatch_reference_vectored(crc: u32, bufs: &[&[u8]]) -> u32 {
  crc32_apply_kernel_vectored(crc, bufs, crc32_reference)
}

#[cfg(feature = "std")]
#[inline]
fn crc32_dispatch_portable_vectored(crc: u32, bufs: &[&[u8]]) -> u32 {
  crc32_apply_kernel_vectored(crc, bufs, crc32_portable)
}

#[cfg(feature = "std")]
#[inline]
fn crc32c_dispatch_reference(crc: u32, data: &[u8]) -> u32 {
  crc32c_reference(crc, data)
}

#[cfg(feature = "std")]
#[inline]
fn crc32c_dispatch_portable(crc: u32, data: &[u8]) -> u32 {
  crc32c_portable(crc, data)
}

#[cfg(feature = "std")]
#[inline]
fn crc32c_dispatch_reference_vectored(crc: u32, bufs: &[&[u8]]) -> u32 {
  crc32_apply_kernel_vectored(crc, bufs, crc32c_reference)
}

#[cfg(feature = "std")]
#[inline]
fn crc32c_dispatch_portable_vectored(crc: u32, bufs: &[&[u8]]) -> u32 {
  crc32_apply_kernel_vectored(crc, bufs, crc32c_portable)
}

#[cfg(feature = "std")]
static CRC32_DISPATCH: backend::OnceCache<Crc32DispatchFn> = backend::OnceCache::new();
#[cfg(feature = "std")]
static CRC32_DISPATCH_VECTORED: backend::OnceCache<Crc32DispatchVectoredFn> = backend::OnceCache::new();
#[cfg(feature = "std")]
static CRC32C_DISPATCH: backend::OnceCache<Crc32DispatchFn> = backend::OnceCache::new();
#[cfg(feature = "std")]
static CRC32C_DISPATCH_VECTORED: backend::OnceCache<Crc32DispatchVectoredFn> = backend::OnceCache::new();

#[cfg(feature = "std")]
#[inline]
fn resolve_crc32_dispatch() -> Crc32DispatchFn {
  match config::get().effective_force {
    Crc32Force::Reference => crc32_dispatch_reference,
    Crc32Force::Portable => crc32_dispatch_portable,
    _ => crc32_dispatch_auto,
  }
}

#[cfg(feature = "std")]
#[inline]
fn resolve_crc32_dispatch_vectored() -> Crc32DispatchVectoredFn {
  match config::get().effective_force {
    Crc32Force::Reference => crc32_dispatch_reference_vectored,
    Crc32Force::Portable => crc32_dispatch_portable_vectored,
    _ => crc32_dispatch_auto_vectored,
  }
}

#[cfg(feature = "std")]
#[inline]
fn resolve_crc32c_dispatch() -> Crc32DispatchFn {
  match config::get().effective_force {
    Crc32Force::Reference => crc32c_dispatch_reference,
    Crc32Force::Portable => crc32c_dispatch_portable,
    _ => crc32c_dispatch_auto,
  }
}

#[cfg(feature = "std")]
#[inline]
fn resolve_crc32c_dispatch_vectored() -> Crc32DispatchVectoredFn {
  match config::get().effective_force {
    Crc32Force::Reference => crc32c_dispatch_reference_vectored,
    Crc32Force::Portable => crc32c_dispatch_portable_vectored,
    _ => crc32c_dispatch_auto_vectored,
  }
}

/// CRC-32 (IEEE) dispatch - hot path uses one-time resolved dispatch function.
#[inline]
fn crc32_dispatch(crc: u32, data: &[u8]) -> u32 {
  #[cfg(feature = "std")]
  {
    let dispatch = CRC32_DISPATCH.get_or_init(resolve_crc32_dispatch);
    dispatch(crc, data)
  }

  #[cfg(not(feature = "std"))]
  {
    crc32_dispatch_auto(crc, data)
  }
}

/// CRC-32 (IEEE) vectored dispatch (processes multiple buffers in order).
#[inline]
fn crc32_dispatch_vectored(crc: u32, bufs: &[&[u8]]) -> u32 {
  #[cfg(feature = "std")]
  {
    let dispatch = CRC32_DISPATCH_VECTORED.get_or_init(resolve_crc32_dispatch_vectored);
    dispatch(crc, bufs)
  }

  #[cfg(not(feature = "std"))]
  {
    crc32_dispatch_auto_vectored(crc, bufs)
  }
}

/// CRC-32C (Castagnoli) dispatch - hot path uses one-time resolved dispatch function.
#[inline]
fn crc32c_dispatch(crc: u32, data: &[u8]) -> u32 {
  #[cfg(feature = "std")]
  {
    let dispatch = CRC32C_DISPATCH.get_or_init(resolve_crc32c_dispatch);
    dispatch(crc, data)
  }

  #[cfg(not(feature = "std"))]
  {
    crc32c_dispatch_auto(crc, data)
  }
}

/// CRC-32C (Castagnoli) vectored dispatch (processes multiple buffers in order).
#[inline]
fn crc32c_dispatch_vectored(crc: u32, bufs: &[&[u8]]) -> u32 {
  #[cfg(feature = "std")]
  {
    let dispatch = CRC32C_DISPATCH_VECTORED.get_or_init(resolve_crc32c_dispatch_vectored);
    dispatch(crc, bufs)
  }

  #[cfg(not(feature = "std"))]
  {
    crc32c_dispatch_auto_vectored(crc, bufs)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32 Types
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32 (IEEE) checksum.
///
/// Used by Ethernet, gzip, zip, PNG, etc.
///
/// # Properties
///
/// - **Polynomial**: 0x04C11DB7 (normal), 0xEDB88320 (reflected)
/// - **Initial value**: 0xFFFFFFFF
/// - **Final XOR**: 0xFFFFFFFF
/// - **Reflect input/output**: Yes
///
/// # Examples
///
/// ```rust
/// use checksum::{Checksum, Crc32};
///
/// let crc = Crc32::checksum(b"123456789");
/// assert_eq!(crc, 0xCBF4_3926);
/// ```
#[derive(Clone, Copy)]
pub struct Crc32 {
  state: u32,
}

/// Explicit name for the IEEE CRC-32 variant (alias of [`Crc32`]).
pub type Crc32Ieee = Crc32;

impl Crc32 {
  /// Pre-computed shift-by-8 matrix for combine.
  const SHIFT8_MATRIX: Gf2Matrix32 = generate_shift8_matrix_32(CRC32_IEEE_POLY);

  /// Create a hasher to resume from a previous CRC value.
  #[inline]
  #[must_use]
  pub const fn resume(crc: u32) -> Self {
    Self { state: crc ^ !0 }
  }

  /// Get the name of the currently selected backend.
  #[must_use]
  pub fn backend_name() -> &'static str {
    let cfg = config::get();
    match cfg.effective_force {
      Crc32Force::Reference => kernels::REFERENCE,
      Crc32Force::Portable => kernels::PORTABLE,
      _ => crc32_selected_kernel_name(1024),
    }
  }

  /// Get the effective CRC-32 configuration (overrides + thresholds).
  #[must_use]
  pub fn config() -> Crc32Config {
    config::get()
  }

  /// Returns the kernel name that the selector would choose for `len`.
  #[must_use]
  pub fn kernel_name_for_len(len: usize) -> &'static str {
    crc32_selected_kernel_name(len)
  }
}

impl traits::Checksum for Crc32 {
  const OUTPUT_SIZE: usize = 4;
  type Output = u32;

  #[inline]
  fn new() -> Self {
    Self { state: !0 }
  }

  #[inline]
  fn with_initial(initial: u32) -> Self {
    Self { state: initial ^ !0 }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.state = crc32_dispatch(self.state, data);
  }

  #[inline]
  fn update_vectored(&mut self, bufs: &[&[u8]]) {
    self.state = crc32_dispatch_vectored(self.state, bufs);
  }

  #[inline]
  fn finalize(&self) -> u32 {
    self.state ^ !0
  }

  #[inline]
  fn reset(&mut self) {
    self.state = !0;
  }
}

impl Default for Crc32 {
  fn default() -> Self {
    <Self as traits::Checksum>::new()
  }
}

impl traits::ChecksumCombine for Crc32 {
  fn combine(crc_a: u32, crc_b: u32, len_b: usize) -> u32 {
    crate::common::combine::combine_crc32(crc_a, crc_b, len_b, Self::SHIFT8_MATRIX)
  }
}

/// CRC-32C (Castagnoli) checksum.
///
/// Used by iSCSI, SCTP, ext4, Btrfs, SSE4.2 `crc32`, etc.
///
/// # Properties
///
/// - **Polynomial**: 0x1EDC6F41 (normal), 0x82F63B78 (reflected)
/// - **Initial value**: 0xFFFFFFFF
/// - **Final XOR**: 0xFFFFFFFF
/// - **Reflect input/output**: Yes
///
/// # Examples
///
/// ```rust
/// use checksum::{Checksum, Crc32C};
///
/// let crc = Crc32C::checksum(b"123456789");
/// assert_eq!(crc, 0xE306_9283);
/// ```
#[derive(Clone, Copy)]
pub struct Crc32C {
  state: u32,
}

/// Explicit name for the Castagnoli CRC-32C variant (alias of [`Crc32C`]).
pub type Crc32Castagnoli = Crc32C;

impl Crc32C {
  /// Pre-computed shift-by-8 matrix for combine.
  const SHIFT8_MATRIX: Gf2Matrix32 = generate_shift8_matrix_32(CRC32C_POLY);

  /// Create a hasher to resume from a previous CRC value.
  #[inline]
  #[must_use]
  pub const fn resume(crc: u32) -> Self {
    Self { state: crc ^ !0 }
  }

  /// Get the name of the currently selected backend.
  #[must_use]
  pub fn backend_name() -> &'static str {
    let cfg = config::get();
    match cfg.effective_force {
      Crc32Force::Reference => kernels::REFERENCE,
      Crc32Force::Portable => kernels::PORTABLE,
      _ => crc32c_selected_kernel_name(1024),
    }
  }

  /// Get the effective CRC-32 configuration (overrides + thresholds).
  #[must_use]
  pub fn config() -> Crc32Config {
    config::get()
  }

  /// Returns the kernel name that the selector would choose for `len`.
  #[must_use]
  pub fn kernel_name_for_len(len: usize) -> &'static str {
    crc32c_selected_kernel_name(len)
  }
}

impl traits::Checksum for Crc32C {
  const OUTPUT_SIZE: usize = 4;
  type Output = u32;

  #[inline]
  fn new() -> Self {
    Self { state: !0 }
  }

  #[inline]
  fn with_initial(initial: u32) -> Self {
    Self { state: initial ^ !0 }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.state = crc32c_dispatch(self.state, data);
  }

  #[inline]
  fn update_vectored(&mut self, bufs: &[&[u8]]) {
    self.state = crc32c_dispatch_vectored(self.state, bufs);
  }

  #[inline]
  fn finalize(&self) -> u32 {
    self.state ^ !0
  }

  #[inline]
  fn reset(&mut self) {
    self.state = !0;
  }
}

impl Default for Crc32C {
  fn default() -> Self {
    <Self as traits::Checksum>::new()
  }
}

impl traits::ChecksumCombine for Crc32C {
  fn combine(crc_a: u32, crc_b: u32, len_b: usize) -> u32 {
    crate::common::combine::combine_crc32(crc_a, crc_b, len_b, Self::SHIFT8_MATRIX)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffered CRC-32 Wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// Buffer size for buffered CRC wrappers.
///
/// Large enough to clear any warmup threshold, small enough to stay cache-friendly.
#[cfg(feature = "alloc")]
const BUFFERED_CRC32_BUFFER_SIZE: usize = 512;

#[cfg(feature = "alloc")]
define_buffered_crc! {
  /// A buffering wrapper around [`Crc32`] for streaming small chunks.
  pub struct BufferedCrc32<Crc32> {
    buffer_size: BUFFERED_CRC32_BUFFER_SIZE,
    threshold_fn: crc32_buffered_threshold,
  }
}

#[cfg(feature = "alloc")]
define_buffered_crc! {
  /// A buffering wrapper around [`Crc32C`] for streaming small chunks.
  pub struct BufferedCrc32C<Crc32C> {
    buffer_size: BUFFERED_CRC32_BUFFER_SIZE,
    threshold_fn: crc32c_buffered_threshold,
  }
}

/// Explicit buffered alias for the IEEE CRC-32 variant (alias of [`BufferedCrc32`]).
#[cfg(feature = "alloc")]
pub type BufferedCrc32Ieee = BufferedCrc32;

/// Explicit buffered alias for the Castagnoli CRC-32C variant (alias of [`BufferedCrc32C`]).
#[cfg(feature = "alloc")]
pub type BufferedCrc32Castagnoli = BufferedCrc32C;

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Introspection
// ─────────────────────────────────────────────────────────────────────────────

impl crate::introspect::KernelIntrospect for Crc32 {
  fn kernel_name_for_len(len: usize) -> &'static str {
    Self::kernel_name_for_len(len)
  }

  fn backend_name() -> &'static str {
    Self::backend_name()
  }
}

impl crate::introspect::KernelIntrospect for Crc32C {
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

  use alloc::{string::String, vec::Vec};

  use super::*;

  const TEST_DATA: &[u8] = b"123456789";

  #[test]
  fn test_crc32_test_vectors() {
    assert_eq!(Crc32::checksum(TEST_DATA), 0xCBF4_3926);
    assert_eq!(Crc32C::checksum(TEST_DATA), 0xE306_9283);
  }

  #[test]
  fn test_crc32_empty_is_zero() {
    assert_eq!(Crc32::checksum(&[]), 0);
    assert_eq!(Crc32C::checksum(&[]), 0);
  }

  #[test]
  fn test_backend_name_not_empty() {
    assert!(!Crc32::backend_name().is_empty());
    assert!(!Crc32C::backend_name().is_empty());
  }

  #[test]
  fn test_kernel_name_for_len_returns_specific_kernel_name() {
    // With the dispatch system, kernel_name_for_len returns the specific kernel name
    // from the dispatch table (e.g., "aarch64/pmull-small", "x86_64/hwcrc").
    let name = Crc32::kernel_name_for_len(0);
    assert!(!name.is_empty());
    // Should be a specific kernel name with arch prefix or portable/reference
    assert!(
      name.contains('/') || name.contains("portable") || name.contains("reference"),
      "Expected specific kernel name, got: {name}"
    );

    // CRC32C should also return a specific kernel name
    let name_c = Crc32C::kernel_name_for_len(0);
    assert!(!name_c.is_empty());
    assert!(
      name_c.contains('/') || name_c.contains("portable") || name_c.contains("reference"),
      "Expected specific kernel name for CRC32C, got: {name_c}"
    );
  }

  #[test]
  fn test_crc32_various_lengths_streaming_matches_oneshot() {
    let mut data = [0u8; 512];
    for (i, b) in data.iter_mut().enumerate() {
      *b = (i as u8).wrapping_mul(17).wrapping_add(i as u8);
    }

    for &len in &[0usize, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 400, 512] {
      let slice = &data[..len];
      let oneshot32 = Crc32::checksum(slice);
      let oneshot32c = Crc32C::checksum(slice);

      let mut s32 = Crc32::new();
      s32.update(slice);
      assert_eq!(s32.finalize(), oneshot32, "crc32 len={len}");

      let mut s32c = Crc32C::new();
      s32c.update(slice);
      assert_eq!(s32c.finalize(), oneshot32c, "crc32c len={len}");

      let mut c32 = Crc32::new();
      for chunk in slice.chunks(37) {
        c32.update(chunk);
      }
      assert_eq!(c32.finalize(), oneshot32, "crc32 chunked len={len}");

      let mut c32c = Crc32C::new();
      for chunk in slice.chunks(37) {
        c32c.update(chunk);
      }
      assert_eq!(c32c.finalize(), oneshot32c, "crc32c chunked len={len}");
    }
  }

  #[cfg(feature = "alloc")]
  #[test]
  fn test_buffered_crc32_matches_unbuffered() {
    let data: Vec<u8> = (0..2048).map(|i| (i as u8).wrapping_mul(31)).collect();
    let expected = Crc32::checksum(&data);

    let mut buffered = BufferedCrc32::new();
    for chunk in data.chunks(3) {
      buffered.update(chunk);
    }
    assert_eq!(buffered.finalize(), expected);
  }

  #[cfg(feature = "alloc")]
  #[test]
  fn test_buffered_crc32c_matches_unbuffered() {
    let data: Vec<u8> = (0..2048).map(|i| (i as u8).wrapping_mul(29).wrapping_add(7)).collect();
    let expected = Crc32C::checksum(&data);

    let mut buffered = BufferedCrc32C::new();
    for chunk in data.chunks(3) {
      buffered.update(chunk);
    }
    assert_eq!(buffered.finalize(), expected);
  }

  #[test]
  fn test_crc32_streaming_across_thresholds() {
    let table = crate::dispatch::active_table();

    // Use boundaries from the active dispatch table
    let mut crc32_thresholds = Vec::with_capacity(table.boundaries.len());
    for &boundary in &table.boundaries {
      if boundary > 0 && boundary <= (1 << 20) {
        crc32_thresholds.push(boundary);
      }
    }

    for &threshold in &crc32_thresholds {
      let size = threshold + 256;
      let data: Vec<u8> = (0..size).map(|i| (i as u8).wrapping_mul(13)).collect();

      let oneshot32 = Crc32::checksum(&data);

      let mut h32 = Crc32::new();
      h32.update(&data[..16]);
      h32.update(&data[16..]);
      assert_eq!(h32.finalize(), oneshot32, "crc32 threshold={threshold}");
    }

    // Same thresholds for CRC32C
    for &threshold in &crc32_thresholds {
      let size = threshold + 256;
      let data: Vec<u8> = (0..size).map(|i| (i as u8).wrapping_mul(13)).collect();

      let oneshot32c = Crc32C::checksum(&data);

      let mut h32c = Crc32C::new();
      h32c.update(&data[..16]);
      h32c.update(&data[16..]);
      assert_eq!(h32c.finalize(), oneshot32c, "crc32c threshold={threshold}");
    }
  }

  /// Smoke test used by CI: validates forced tier selection + correctness.
  ///
  /// Run it in isolation with env set, e.g.:
  /// `RSCRYPTO_CRC32_FORCE=pclmul cargo test -p checksum test_crc32_forced_kernel_smoke_from_env
  /// --lib`
  #[test]
  fn test_crc32_forced_kernel_smoke_from_env() {
    let force = std::env::var("RSCRYPTO_CRC32_FORCE").unwrap_or_else(|_| String::from("auto"));

    if force.trim().is_empty() {
      return;
    }

    let len = 64 * 1024;
    let data: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(31).wrapping_add(7)).collect();
    let expected = portable::crc32_slice16_ieee(!0, &data) ^ !0;
    let got = Crc32::checksum(&data);
    assert_eq!(got, expected);

    let cfg = Crc32::config();
    let kernel = Crc32::kernel_name_for_len(len);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = (cfg, kernel);

    #[cfg(target_arch = "x86_64")]
    {
      let caps = platform::caps();
      // Verify the force mode was recognized - checksum correctness already validated above.
      // Note: kernel_name_for_len returns the dispatcher name (e.g., "x86_64/auto"),
      // not the forced kernel, so we don't assert on kernel name for forced modes.
      let _ = kernel; // silence unused variable warning
      if force.eq_ignore_ascii_case("pclmul") || force.eq_ignore_ascii_case("clmul") {
        assert_eq!(cfg.requested_force, Crc32Force::Pclmul);
        assert!(caps.has(platform::caps::x86::PCLMUL_READY) || cfg.effective_force != Crc32Force::Pclmul);
      }
      if force.eq_ignore_ascii_case("vpclmul") {
        assert_eq!(cfg.requested_force, Crc32Force::Vpclmul);
        assert!(caps.has(platform::caps::x86::VPCLMUL_READY) || cfg.effective_force != Crc32Force::Vpclmul);
      }
    }

    #[cfg(target_arch = "aarch64")]
    {
      let caps = platform::caps();
      // Verify the force mode was recognized - checksum correctness already validated above.
      // Note: kernel_name_for_len returns the policy-based selection, not the forced kernel,
      // so we don't assert on kernel name for forced modes.
      if force.eq_ignore_ascii_case("hwcrc") || force.eq_ignore_ascii_case("crc") {
        assert_eq!(cfg.requested_force, Crc32Force::Hwcrc);
        assert!(caps.has(platform::caps::aarch64::CRC_READY) || cfg.effective_force != Crc32Force::Hwcrc);
      }
      if force.eq_ignore_ascii_case("pmull") {
        assert_eq!(cfg.requested_force, Crc32Force::Pmull);
        assert!(caps.has(platform::caps::aarch64::PMULL_READY) || cfg.effective_force != Crc32Force::Pmull);
      }
      if force.eq_ignore_ascii_case("pmull-eor3") || force.eq_ignore_ascii_case("eor3") {
        assert_eq!(cfg.requested_force, Crc32Force::PmullEor3);
        assert!(caps.has(platform::caps::aarch64::PMULL_EOR3_READY) || cfg.effective_force != Crc32Force::PmullEor3);
      }
      let _ = kernel; // suppress unused warning
    }
  }

  /// Smoke test used by CI: validates forced tier selection + correctness.
  #[test]
  fn test_crc32c_forced_kernel_smoke_from_env() {
    let force = std::env::var("RSCRYPTO_CRC32_FORCE").unwrap_or_else(|_| String::from("auto"));

    if force.trim().is_empty() {
      return;
    }

    let len = 64 * 1024;
    let data: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(29).wrapping_add(7)).collect();
    let expected = portable::crc32c_slice16(!0, &data) ^ !0;
    let got = Crc32C::checksum(&data);
    assert_eq!(got, expected);

    let cfg = Crc32C::config();
    let kernel = Crc32C::kernel_name_for_len(len);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = (cfg, kernel);

    #[cfg(target_arch = "x86_64")]
    {
      let caps = platform::caps();
      // Verify the force mode was recognized - checksum correctness already validated above.
      // Note: kernel_name_for_len returns the dispatcher name (e.g., "x86_64/auto"),
      // not the forced kernel, so we don't assert on kernel name for forced modes.
      let _ = kernel; // silence unused variable warning
      if force.eq_ignore_ascii_case("hwcrc") || force.eq_ignore_ascii_case("crc") {
        assert_eq!(cfg.requested_force, Crc32Force::Hwcrc);
        assert!(caps.has(platform::caps::x86::CRC32C_READY) || cfg.effective_force != Crc32Force::Hwcrc);
      }
      if force.eq_ignore_ascii_case("pclmul") || force.eq_ignore_ascii_case("clmul") {
        assert_eq!(cfg.requested_force, Crc32Force::Pclmul);
        assert!(
          (caps.has(platform::caps::x86::CRC32C_READY) && caps.has(platform::caps::x86::PCLMUL_READY))
            || cfg.effective_force != Crc32Force::Pclmul
        );
      }
      if force.eq_ignore_ascii_case("vpclmul") {
        assert_eq!(cfg.requested_force, Crc32Force::Vpclmul);
        assert!(
          (caps.has(platform::caps::x86::CRC32C_READY) && caps.has(platform::caps::x86::VPCLMUL_READY))
            || cfg.effective_force != Crc32Force::Vpclmul
        );
      }
    }

    #[cfg(target_arch = "aarch64")]
    {
      let caps = platform::caps();
      // Verify the force mode was recognized - checksum correctness already validated above.
      // Note: kernel_name_for_len returns the policy-based selection, not the forced kernel,
      // so we don't assert on kernel name for forced modes.
      if force.eq_ignore_ascii_case("hwcrc") || force.eq_ignore_ascii_case("crc") {
        assert_eq!(cfg.requested_force, Crc32Force::Hwcrc);
        assert!(caps.has(platform::caps::aarch64::CRC_READY) || cfg.effective_force != Crc32Force::Hwcrc);
      }
      if force.eq_ignore_ascii_case("pmull") {
        assert_eq!(cfg.requested_force, Crc32Force::Pmull);
        assert!(caps.has(platform::caps::aarch64::PMULL_READY) || cfg.effective_force != Crc32Force::Pmull);
      }
      if force.eq_ignore_ascii_case("pmull-eor3") || force.eq_ignore_ascii_case("eor3") {
        assert_eq!(cfg.requested_force, Crc32Force::PmullEor3);
        assert!(caps.has(platform::caps::aarch64::PMULL_EOR3_READY) || cfg.effective_force != Crc32Force::PmullEor3);
      }
      let _ = kernel; // suppress unused warning
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Cross-Check Tests: Reference Implementation Verification
  // ─────────────────────────────────────────────────────────────────────────────

  mod cross_check {
    use alloc::{vec, vec::Vec};

    use super::*;
    use crate::common::{
      reference::crc32_bitwise,
      tables::{CRC32_IEEE_POLY, CRC32C_POLY},
    };

    /// Comprehensive test lengths covering all edge cases.
    const TEST_LENGTHS: &[usize] = &[
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256,
      257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8192, 16384, 32768, 65536,
    ];

    const STREAMING_CHUNK_SIZES: &[usize] = &[1, 3, 7, 13, 17, 31, 37, 61, 127, 251];

    fn generate_test_data(len: usize) -> Vec<u8> {
      (0..len)
        .map(|i| {
          let i = i as u64;
          ((i.wrapping_mul(2654435761) ^ i.wrapping_mul(0x9E3779B97F4A7C15)) & 0xFF) as u8
        })
        .collect()
    }

    fn reference_ieee(data: &[u8]) -> u32 {
      crc32_bitwise(CRC32_IEEE_POLY, !0u32, data) ^ !0u32
    }

    fn reference_castagnoli(data: &[u8]) -> u32 {
      crc32_bitwise(CRC32C_POLY, !0u32, data) ^ !0u32
    }

    #[test]
    fn cross_check_ieee_all_lengths() {
      for &len in TEST_LENGTHS {
        let data = generate_test_data(len);
        let reference = reference_ieee(&data);
        let actual = Crc32::checksum(&data);
        assert_eq!(
          actual, reference,
          "CRC32-IEEE mismatch at len={len}: actual={actual:#010X}, reference={reference:#010X}"
        );
      }
    }

    #[test]
    fn cross_check_castagnoli_all_lengths() {
      for &len in TEST_LENGTHS {
        let data = generate_test_data(len);
        let reference = reference_castagnoli(&data);
        let actual = Crc32C::checksum(&data);
        assert_eq!(
          actual, reference,
          "CRC32C mismatch at len={len}: actual={actual:#010X}, reference={reference:#010X}"
        );
      }
    }

    #[test]
    fn cross_check_ieee_all_single_bytes() {
      for byte in 0u8..=255 {
        let data = [byte];
        let reference = reference_ieee(&data);
        let actual = Crc32::checksum(&data);
        assert_eq!(
          actual, reference,
          "CRC32-IEEE single-byte mismatch for byte={byte:#04X}"
        );
      }
    }

    #[test]
    fn cross_check_castagnoli_all_single_bytes() {
      for byte in 0u8..=255 {
        let data = [byte];
        let reference = reference_castagnoli(&data);
        let actual = Crc32C::checksum(&data);
        assert_eq!(actual, reference, "CRC32C single-byte mismatch for byte={byte:#04X}");
      }
    }

    #[test]
    fn cross_check_ieee_streaming_all_chunk_sizes() {
      let data = generate_test_data(4096);
      let reference = reference_ieee(&data);

      for &chunk_size in STREAMING_CHUNK_SIZES {
        let mut hasher = Crc32::new();
        for chunk in data.chunks(chunk_size) {
          hasher.update(chunk);
        }
        let actual = hasher.finalize();
        assert_eq!(
          actual, reference,
          "CRC32-IEEE streaming mismatch with chunk_size={chunk_size}"
        );
      }
    }

    #[test]
    fn cross_check_castagnoli_streaming_all_chunk_sizes() {
      let data = generate_test_data(4096);
      let reference = reference_castagnoli(&data);

      for &chunk_size in STREAMING_CHUNK_SIZES {
        let mut hasher = Crc32C::new();
        for chunk in data.chunks(chunk_size) {
          hasher.update(chunk);
        }
        let actual = hasher.finalize();
        assert_eq!(
          actual, reference,
          "CRC32C streaming mismatch with chunk_size={chunk_size}"
        );
      }
    }

    #[test]
    fn cross_check_ieee_combine_all_splits() {
      let data = generate_test_data(1024);
      let reference = reference_ieee(&data);

      let small_data = &data[..64];
      let small_ref = reference_ieee(small_data);

      for split in 0..=small_data.len() {
        let (a, b) = small_data.split_at(split);
        let crc_a = Crc32::checksum(a);
        let crc_b = Crc32::checksum(b);
        let combined = Crc32::combine(crc_a, crc_b, b.len());
        assert_eq!(combined, small_ref, "CRC32-IEEE combine mismatch at split={split}");
      }

      let strategic_splits = [0, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 512, 1024];
      for &split in &strategic_splits {
        if split > data.len() {
          continue;
        }
        let (a, b) = data.split_at(split);
        let combined = Crc32::combine(Crc32::checksum(a), Crc32::checksum(b), b.len());
        assert_eq!(
          combined, reference,
          "CRC32-IEEE combine mismatch at strategic split={split}"
        );
      }
    }

    #[test]
    fn cross_check_castagnoli_combine_all_splits() {
      let data = generate_test_data(1024);
      let reference = reference_castagnoli(&data);

      let small_data = &data[..64];
      let small_ref = reference_castagnoli(small_data);

      for split in 0..=small_data.len() {
        let (a, b) = small_data.split_at(split);
        let crc_a = Crc32C::checksum(a);
        let crc_b = Crc32C::checksum(b);
        let combined = Crc32C::combine(crc_a, crc_b, b.len());
        assert_eq!(combined, small_ref, "CRC32C combine mismatch at split={split}");
      }

      let strategic_splits = [0, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 512, 1024];
      for &split in &strategic_splits {
        if split > data.len() {
          continue;
        }
        let (a, b) = data.split_at(split);
        let combined = Crc32C::combine(Crc32C::checksum(a), Crc32C::checksum(b), b.len());
        assert_eq!(
          combined, reference,
          "CRC32C combine mismatch at strategic split={split}"
        );
      }
    }

    #[test]
    fn cross_check_ieee_unaligned_offsets() {
      let mut buffer = vec![0u8; 4096 + 64];
      for (i, byte) in buffer.iter_mut().enumerate() {
        *byte = (((i as u64).wrapping_mul(17)) & 0xFF) as u8;
      }

      for offset in 0..16 {
        let data = &buffer[offset..offset + 1024];
        let reference = reference_ieee(data);
        let actual = Crc32::checksum(data);
        assert_eq!(actual, reference, "CRC32-IEEE unaligned mismatch at offset={offset}");
      }
    }

    #[test]
    fn cross_check_castagnoli_unaligned_offsets() {
      let mut buffer = vec![0u8; 4096 + 64];
      for (i, byte) in buffer.iter_mut().enumerate() {
        *byte = (((i as u64).wrapping_mul(17)) & 0xFF) as u8;
      }

      for offset in 0..16 {
        let data = &buffer[offset..offset + 1024];
        let reference = reference_castagnoli(data);
        let actual = Crc32C::checksum(data);
        assert_eq!(actual, reference, "CRC32C unaligned mismatch at offset={offset}");
      }
    }

    #[test]
    fn cross_check_ieee_byte_at_a_time_streaming() {
      let data = generate_test_data(256);
      let reference = reference_ieee(&data);

      let mut hasher = Crc32::new();
      for &byte in &data {
        hasher.update(&[byte]);
      }
      let actual = hasher.finalize();
      assert_eq!(actual, reference, "CRC32-IEEE byte-at-a-time streaming mismatch");
    }

    #[test]
    fn cross_check_castagnoli_byte_at_a_time_streaming() {
      let data = generate_test_data(256);
      let reference = reference_castagnoli(&data);

      let mut hasher = Crc32C::new();
      for &byte in &data {
        hasher.update(&[byte]);
      }
      let actual = hasher.finalize();
      assert_eq!(actual, reference, "CRC32C byte-at-a-time streaming mismatch");
    }

    #[test]
    fn cross_check_reference_kernel_accessible() {
      let data = generate_test_data(1024);

      let ieee_ref = crc32_reference(!0u32, &data) ^ !0u32;
      let ieee_direct = reference_ieee(&data);
      assert_eq!(ieee_ref, ieee_direct, "IEEE reference kernel mismatch");

      let c_ref = crc32c_reference(!0u32, &data) ^ !0u32;
      let c_direct = reference_castagnoli(&data);
      assert_eq!(c_ref, c_direct, "Castagnoli reference kernel mismatch");
    }

    #[test]
    fn cross_check_portable_matches_reference() {
      for &len in TEST_LENGTHS {
        let data = generate_test_data(len);

        let portable_ieee = portable::crc32_slice16_ieee(!0u32, &data) ^ !0u32;
        let reference_ieee_val = reference_ieee(&data);
        assert_eq!(portable_ieee, reference_ieee_val, "IEEE portable mismatch at len={len}");

        let portable_c = portable::crc32c_slice16(!0u32, &data) ^ !0u32;
        let reference_c_val = reference_castagnoli(&data);
        assert_eq!(portable_c, reference_c_val, "Castagnoli portable mismatch at len={len}");
      }
    }
  }
}
