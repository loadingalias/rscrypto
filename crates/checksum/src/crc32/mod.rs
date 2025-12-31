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
//! - powerpc64: VPMSUMD folding (CRC-32 and CRC-32C)
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
mod kernels;
mod policy;
pub(crate) mod portable;
mod tuned_defaults;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "powerpc64")]
mod powerpc64;

#[cfg(target_arch = "s390x")]
mod s390x;

#[cfg(target_arch = "riscv64")]
mod riscv64;

use backend::{PolicyCache, dispatch::Selected};
#[allow(unused_imports)]
pub use config::{Crc32Config, Crc32Force, Crc32Tunables};
#[allow(unused_imports)]
pub(super) use traits::{Checksum, ChecksumCombine};

use crate::{
  common::{
    combine::{Gf2Matrix32, generate_shift8_matrix_32},
    reference::crc32_bitwise,
    tables::{CRC32_IEEE_POLY, CRC32C_POLY, generate_crc32_tables_16},
  },
  dispatchers::{Crc32Dispatcher, Crc32Fn},
};

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Tables (compile-time)
// ─────────────────────────────────────────────────────────────────────────────

/// Portable kernel tables (pre-computed at compile time).
mod kernel_tables {
  use super::*;
  pub static IEEE_TABLES_16: [[u32; 256]; 16] = generate_crc32_tables_16(CRC32_IEEE_POLY);
  pub static CRC32C_TABLES_16: [[u32; 256]; 16] = generate_crc32_tables_16(CRC32C_POLY);
}

// ─────────────────────────────────────────────────────────────────────────────
// Portable Kernel Wrappers
// ─────────────────────────────────────────────────────────────────────────────

fn crc32_portable(crc: u32, data: &[u8]) -> u32 {
  portable::crc32_slice16_ieee(crc, data)
}

fn crc32c_portable(crc: u32, data: &[u8]) -> u32 {
  portable::crc32c_slice16(crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Reference Kernel Wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32 (IEEE) reference (bitwise) kernel wrapper.
///
/// This is the canonical reference implementation - obviously correct,
/// audit-friendly, and used for verification of all optimized paths.
fn crc32_reference(crc: u32, data: &[u8]) -> u32 {
  crc32_bitwise(CRC32_IEEE_POLY, crc, data)
}

/// CRC-32C (Castagnoli) reference (bitwise) kernel wrapper.
///
/// This is the canonical reference implementation - obviously correct,
/// audit-friendly, and used for verification of all optimized paths.
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
  config::get().tunables.hwcrc_to_fusion.max(64)
}

#[cfg(all(feature = "alloc", target_arch = "aarch64"))]
#[inline]
#[must_use]
fn crc32_buffered_threshold_impl() -> usize {
  config::get().tunables.portable_to_hwcrc.max(64)
}

#[cfg(all(feature = "alloc", not(any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[inline]
#[must_use]
fn crc32_buffered_threshold_impl() -> usize {
  config::get().tunables.portable_to_hwcrc.max(64)
}

#[cfg(feature = "alloc")]
#[inline]
#[must_use]
#[allow(dead_code)]
fn crc32c_buffered_threshold() -> usize {
  config::get().tunables.portable_to_hwcrc.max(64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Cached Policy and Kernels (works on both std and no_std)
// ─────────────────────────────────────────────────────────────────────────────

/// Cached IEEE policy and kernels for x86_64.
#[cfg(target_arch = "x86_64")]
static CRC32_IEEE_CACHED: PolicyCache<policy::Crc32Policy, policy::Crc32Kernels> = PolicyCache::new();

/// Cached Castagnoli policy and kernels for x86_64.
#[cfg(target_arch = "x86_64")]
static CRC32C_CACHED: PolicyCache<policy::Crc32Policy, policy::Crc32Kernels> = PolicyCache::new();

/// Cached IEEE policy and kernels for aarch64.
#[cfg(target_arch = "aarch64")]
static CRC32_IEEE_CACHED: PolicyCache<policy::Crc32Policy, policy::Crc32Kernels> = PolicyCache::new();

/// Cached Castagnoli policy and kernels for aarch64.
#[cfg(target_arch = "aarch64")]
static CRC32C_CACHED: PolicyCache<policy::Crc32Policy, policy::Crc32Kernels> = PolicyCache::new();

/// Cached IEEE policy and kernels for powerpc64.
#[cfg(target_arch = "powerpc64")]
static CRC32_IEEE_CACHED: PolicyCache<policy::Crc32Policy, policy::Crc32Kernels> = PolicyCache::new();

/// Cached Castagnoli policy and kernels for powerpc64.
#[cfg(target_arch = "powerpc64")]
static CRC32C_CACHED: PolicyCache<policy::Crc32Policy, policy::Crc32Kernels> = PolicyCache::new();

/// Cached IEEE policy and kernels for s390x.
#[cfg(target_arch = "s390x")]
static CRC32_IEEE_CACHED: PolicyCache<policy::Crc32Policy, policy::Crc32Kernels> = PolicyCache::new();

/// Cached Castagnoli policy and kernels for s390x.
#[cfg(target_arch = "s390x")]
static CRC32C_CACHED: PolicyCache<policy::Crc32Policy, policy::Crc32Kernels> = PolicyCache::new();

/// Cached IEEE policy and kernels for riscv64.
#[cfg(target_arch = "riscv64")]
static CRC32_IEEE_CACHED: PolicyCache<policy::Crc32Policy, policy::Crc32Kernels> = PolicyCache::new();

/// Cached Castagnoli policy and kernels for riscv64.
#[cfg(target_arch = "riscv64")]
static CRC32C_CACHED: PolicyCache<policy::Crc32Policy, policy::Crc32Kernels> = PolicyCache::new();

/// Initialize IEEE policy and kernels for x86_64.
#[cfg(target_arch = "x86_64")]
fn init_ieee_policy() -> (policy::Crc32Policy, policy::Crc32Kernels) {
  let cfg = config::get();
  let caps = platform::caps();
  let tune = platform::tune();
  let pol = policy::Crc32Policy::from_config(&cfg, caps, &tune, policy::Crc32Variant::Ieee);
  let kernels = policy::build_ieee_kernels_x86(&pol, crc32_reference, crc32_portable);
  (pol, kernels)
}

/// Initialize Castagnoli policy and kernels for x86_64.
#[cfg(target_arch = "x86_64")]
fn init_castagnoli_policy() -> (policy::Crc32Policy, policy::Crc32Kernels) {
  let cfg = config::get();
  let caps = platform::caps();
  let tune = platform::tune();
  let pol = policy::Crc32Policy::from_config(&cfg, caps, &tune, policy::Crc32Variant::Castagnoli);
  let kernels = policy::build_castagnoli_kernels_x86(&pol, crc32c_reference, crc32c_portable);
  (pol, kernels)
}

/// Initialize IEEE policy and kernels for aarch64.
#[cfg(target_arch = "aarch64")]
fn init_ieee_policy() -> (policy::Crc32Policy, policy::Crc32Kernels) {
  let cfg = config::get();
  let caps = platform::caps();
  let tune = platform::tune();
  let pol = policy::Crc32Policy::from_config(&cfg, caps, &tune, policy::Crc32Variant::Ieee);
  let kernels = policy::build_ieee_kernels_aarch64(&pol, crc32_reference, crc32_portable);
  (pol, kernels)
}

/// Initialize Castagnoli policy and kernels for aarch64.
#[cfg(target_arch = "aarch64")]
fn init_castagnoli_policy() -> (policy::Crc32Policy, policy::Crc32Kernels) {
  let cfg = config::get();
  let caps = platform::caps();
  let tune = platform::tune();
  let pol = policy::Crc32Policy::from_config(&cfg, caps, &tune, policy::Crc32Variant::Castagnoli);
  let kernels = policy::build_castagnoli_kernels_aarch64(&pol, crc32c_reference, crc32c_portable);
  (pol, kernels)
}

/// Initialize IEEE policy and kernels for powerpc64.
#[cfg(target_arch = "powerpc64")]
fn init_ieee_policy() -> (policy::Crc32Policy, policy::Crc32Kernels) {
  let cfg = config::get();
  let caps = platform::caps();
  let tune = platform::tune();
  let pol = policy::Crc32Policy::from_config(&cfg, caps, &tune, policy::Crc32Variant::Ieee);
  let kernels = policy::build_ieee_kernels_powerpc64(&pol, crc32_reference, crc32_portable);
  (pol, kernels)
}

/// Initialize Castagnoli policy and kernels for powerpc64.
#[cfg(target_arch = "powerpc64")]
fn init_castagnoli_policy() -> (policy::Crc32Policy, policy::Crc32Kernels) {
  let cfg = config::get();
  let caps = platform::caps();
  let tune = platform::tune();
  let pol = policy::Crc32Policy::from_config(&cfg, caps, &tune, policy::Crc32Variant::Castagnoli);
  let kernels = policy::build_castagnoli_kernels_powerpc64(&pol, crc32c_reference, crc32c_portable);
  (pol, kernels)
}

/// Initialize IEEE policy and kernels for s390x.
#[cfg(target_arch = "s390x")]
fn init_ieee_policy() -> (policy::Crc32Policy, policy::Crc32Kernels) {
  let cfg = config::get();
  let caps = platform::caps();
  let tune = platform::tune();
  let pol = policy::Crc32Policy::from_config(&cfg, caps, &tune, policy::Crc32Variant::Ieee);
  let kernels = policy::build_ieee_kernels_s390x(&pol, crc32_reference, crc32_portable);
  (pol, kernels)
}

/// Initialize Castagnoli policy and kernels for s390x.
#[cfg(target_arch = "s390x")]
fn init_castagnoli_policy() -> (policy::Crc32Policy, policy::Crc32Kernels) {
  let cfg = config::get();
  let caps = platform::caps();
  let tune = platform::tune();
  let pol = policy::Crc32Policy::from_config(&cfg, caps, &tune, policy::Crc32Variant::Castagnoli);
  let kernels = policy::build_castagnoli_kernels_s390x(&pol, crc32c_reference, crc32c_portable);
  (pol, kernels)
}

/// Initialize IEEE policy and kernels for riscv64.
#[cfg(target_arch = "riscv64")]
fn init_ieee_policy() -> (policy::Crc32Policy, policy::Crc32Kernels) {
  let cfg = config::get();
  let caps = platform::caps();
  let tune = platform::tune();
  let pol = policy::Crc32Policy::from_config(&cfg, caps, &tune, policy::Crc32Variant::Ieee);
  let kernels = policy::build_ieee_kernels_riscv64(&pol, crc32_reference, crc32_portable);
  (pol, kernels)
}

/// Initialize Castagnoli policy and kernels for riscv64.
#[cfg(target_arch = "riscv64")]
fn init_castagnoli_policy() -> (policy::Crc32Policy, policy::Crc32Kernels) {
  let cfg = config::get();
  let caps = platform::caps();
  let tune = platform::tune();
  let pol = policy::Crc32Policy::from_config(&cfg, caps, &tune, policy::Crc32Variant::Castagnoli);
  let kernels = policy::build_castagnoli_kernels_riscv64(&pol, crc32c_reference, crc32c_portable);
  (pol, kernels)
}

/// Get the kernel name for the selected CRC-32 (IEEE) algorithm.
///
/// Uses the cached policy for efficient kernel name lookup.
/// Caching works on both std (OnceLock) and no_std (atomic state machine).
#[inline]
#[must_use]
#[allow(unused_variables)] // `len` only used on SIMD architectures
pub(crate) fn crc32_selected_kernel_name(len: usize) -> &'static str {
  #[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "s390x",
    target_arch = "riscv64"
  ))]
  {
    let (pol, _) = CRC32_IEEE_CACHED.get_or_init(init_ieee_policy);
    pol.kernel_name(len)
  }

  #[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "s390x",
    target_arch = "riscv64"
  )))]
  {
    kernels::PORTABLE
  }
}

/// Get the kernel name for the selected CRC-32C (Castagnoli) algorithm.
///
/// Uses the cached policy for efficient kernel name lookup.
/// Caching works on both std (OnceLock) and no_std (atomic state machine).
#[inline]
#[must_use]
#[allow(unused_variables)] // `len` only used on SIMD architectures
pub(crate) fn crc32c_selected_kernel_name(len: usize) -> &'static str {
  #[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "s390x",
    target_arch = "riscv64"
  ))]
  {
    let (pol, _) = CRC32C_CACHED.get_or_init(init_castagnoli_policy);
    pol.kernel_name(len)
  }

  #[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "s390x",
    target_arch = "riscv64"
  )))]
  {
    kernels::PORTABLE
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto Kernels (architecture-specific)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32C (Castagnoli) auto-dispatch for x86_64.
///
/// Uses cached policy and kernels for efficient dispatch with minimal branching.
/// Caching works on both std (OnceLock) and no_std (atomic state machine).
#[cfg(target_arch = "x86_64")]
fn crc32c_x86_64_auto(crc: u32, data: &[u8]) -> u32 {
  let (pol, kernels) = CRC32C_CACHED.get_or_init(init_castagnoli_policy);
  policy::policy_dispatch(&pol, &kernels, crc, data)
}

/// CRC-32 (IEEE) auto-dispatch for x86_64.
///
/// Uses cached policy and kernels for efficient dispatch with minimal branching.
/// Caching works on both std (OnceLock) and no_std (atomic state machine).
#[cfg(target_arch = "x86_64")]
fn crc32_x86_64_auto(crc: u32, data: &[u8]) -> u32 {
  let (pol, kernels) = CRC32_IEEE_CACHED.get_or_init(init_ieee_policy);
  policy::policy_dispatch(&pol, &kernels, crc, data)
}

/// CRC-32 (IEEE) auto-dispatch for aarch64.
///
/// Uses cached policy and kernels for efficient dispatch with minimal branching.
/// Caching works on both std (OnceLock) and no_std (atomic state machine).
#[cfg(target_arch = "aarch64")]
fn crc32_aarch64_auto(crc: u32, data: &[u8]) -> u32 {
  let (pol, kernels) = CRC32_IEEE_CACHED.get_or_init(init_ieee_policy);
  policy::policy_dispatch(&pol, &kernels, crc, data)
}

/// CRC-32C (Castagnoli) auto-dispatch for aarch64.
///
/// Uses cached policy and kernels for efficient dispatch with minimal branching.
/// Caching works on both std (OnceLock) and no_std (atomic state machine).
#[cfg(target_arch = "aarch64")]
fn crc32c_aarch64_auto(crc: u32, data: &[u8]) -> u32 {
  let (pol, kernels) = CRC32C_CACHED.get_or_init(init_castagnoli_policy);
  policy::policy_dispatch(&pol, &kernels, crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC32 CLMUL Backends (POWER/s390x/RISC-V)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32 (IEEE) auto-dispatch for powerpc64.
///
/// Uses cached policy and kernels for efficient dispatch with minimal branching.
/// Caching works on both std (OnceLock) and no_std (atomic state machine).
#[cfg(target_arch = "powerpc64")]
fn crc32_powerpc64_auto(crc: u32, data: &[u8]) -> u32 {
  let (pol, kernels) = CRC32_IEEE_CACHED.get_or_init(init_ieee_policy);
  policy::policy_dispatch(&pol, &kernels, crc, data)
}

/// CRC-32C (Castagnoli) auto-dispatch for powerpc64.
///
/// Uses cached policy and kernels for efficient dispatch with minimal branching.
/// Caching works on both std (OnceLock) and no_std (atomic state machine).
#[cfg(target_arch = "powerpc64")]
fn crc32c_powerpc64_auto(crc: u32, data: &[u8]) -> u32 {
  let (pol, kernels) = CRC32C_CACHED.get_or_init(init_castagnoli_policy);
  policy::policy_dispatch(&pol, &kernels, crc, data)
}

/// CRC-32 (IEEE) auto-dispatch for s390x.
///
/// Uses cached policy and kernels for efficient dispatch with minimal branching.
/// Caching works on both std (OnceLock) and no_std (atomic state machine).
#[cfg(target_arch = "s390x")]
fn crc32_s390x_auto(crc: u32, data: &[u8]) -> u32 {
  let (pol, kernels) = CRC32_IEEE_CACHED.get_or_init(init_ieee_policy);
  policy::policy_dispatch(&pol, &kernels, crc, data)
}

/// CRC-32C (Castagnoli) auto-dispatch for s390x.
///
/// Uses cached policy and kernels for efficient dispatch with minimal branching.
/// Caching works on both std (OnceLock) and no_std (atomic state machine).
#[cfg(target_arch = "s390x")]
fn crc32c_s390x_auto(crc: u32, data: &[u8]) -> u32 {
  let (pol, kernels) = CRC32C_CACHED.get_or_init(init_castagnoli_policy);
  policy::policy_dispatch(&pol, &kernels, crc, data)
}

/// CRC-32 (IEEE) auto-dispatch for riscv64.
///
/// Uses cached policy and kernels for efficient dispatch with minimal branching.
/// Caching works on both std (OnceLock) and no_std (atomic state machine).
#[cfg(target_arch = "riscv64")]
fn crc32_riscv64_auto(crc: u32, data: &[u8]) -> u32 {
  let (pol, kernels) = CRC32_IEEE_CACHED.get_or_init(init_ieee_policy);
  policy::policy_dispatch(&pol, &kernels, crc, data)
}

/// CRC-32C (Castagnoli) auto-dispatch for riscv64.
///
/// Uses cached policy and kernels for efficient dispatch with minimal branching.
/// Caching works on both std (OnceLock) and no_std (atomic state machine).
#[cfg(target_arch = "riscv64")]
fn crc32c_riscv64_auto(crc: u32, data: &[u8]) -> u32 {
  let (pol, kernels) = CRC32C_CACHED.get_or_init(init_castagnoli_policy);
  policy::policy_dispatch(&pol, &kernels, crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Selection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn select_crc32() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  // Reference/Portable bypass is fine - no SIMD semantics to preserve
  if cfg.effective_force == Crc32Force::Reference {
    return Selected::new(kernels::REFERENCE, crc32_reference);
  }
  if cfg.effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32_portable);
  }

  // ALL SIMD modes (Auto + forced) go through policy dispatch.
  // The policy respects effective_force via kernel family selection.
  if caps.has(platform::caps::x86::PCLMUL_READY) {
    return Selected::new("x86_64/auto", crc32_x86_64_auto);
  }

  Selected::new(kernels::PORTABLE, crc32_portable)
}

#[cfg(target_arch = "x86_64")]
fn select_crc32c() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  // Reference/Portable bypass is fine - no SIMD semantics to preserve
  if cfg.effective_force == Crc32Force::Reference {
    return Selected::new(kernels::REFERENCE, crc32c_reference);
  }
  if cfg.effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32c_portable);
  }

  // ALL SIMD modes (Auto + forced) go through policy dispatch.
  // The policy respects effective_force via kernel family selection.
  if caps.has(platform::caps::x86::CRC32C_READY) {
    return Selected::new("x86_64/auto", crc32c_x86_64_auto);
  }

  Selected::new(kernels::PORTABLE, crc32c_portable)
}

#[cfg(target_arch = "aarch64")]
fn select_crc32() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  // Reference/Portable bypass is fine - no SIMD semantics to preserve
  if cfg.effective_force == Crc32Force::Reference {
    return Selected::new(kernels::REFERENCE, crc32_reference);
  }
  if cfg.effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32_portable);
  }

  // ALL SIMD modes (Auto + forced) go through policy dispatch.
  // The policy respects effective_force via kernel family selection,
  // and applies memory_bound stream suppression uniformly.
  if caps.has(platform::caps::aarch64::CRC_READY) {
    return Selected::new("aarch64/auto", crc32_aarch64_auto);
  }

  Selected::new(kernels::PORTABLE, crc32_portable)
}

#[cfg(target_arch = "aarch64")]
fn select_crc32c() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  // Reference/Portable bypass is fine - no SIMD semantics to preserve
  if cfg.effective_force == Crc32Force::Reference {
    return Selected::new(kernels::REFERENCE, crc32c_reference);
  }
  if cfg.effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32c_portable);
  }

  // ALL SIMD modes (Auto + forced) go through policy dispatch.
  // The policy respects effective_force via kernel family selection,
  // and applies memory_bound stream suppression uniformly.
  if caps.has(platform::caps::aarch64::CRC_READY) {
    return Selected::new("aarch64/auto", crc32c_aarch64_auto);
  }

  Selected::new(kernels::PORTABLE, crc32c_portable)
}

#[cfg(target_arch = "powerpc64")]
fn select_crc32() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc32Force::Reference {
    return Selected::new(kernels::REFERENCE, crc32_reference);
  }

  if cfg.effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32_portable);
  }

  if caps.has(platform::caps::powerpc64::VPMSUM_READY) {
    return Selected::new("powerpc64/auto", crc32_powerpc64_auto);
  }

  Selected::new(kernels::PORTABLE, crc32_portable)
}

#[cfg(target_arch = "powerpc64")]
fn select_crc32c() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc32Force::Reference {
    return Selected::new(kernels::REFERENCE, crc32c_reference);
  }

  if cfg.effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32c_portable);
  }

  if caps.has(platform::caps::powerpc64::VPMSUM_READY) {
    return Selected::new("powerpc64/auto", crc32c_powerpc64_auto);
  }

  Selected::new(kernels::PORTABLE, crc32c_portable)
}

#[cfg(target_arch = "s390x")]
fn select_crc32() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc32Force::Reference {
    return Selected::new(kernels::REFERENCE, crc32_reference);
  }

  if cfg.effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32_portable);
  }

  if caps.has(platform::caps::s390x::VECTOR) {
    return Selected::new("s390x/auto", crc32_s390x_auto);
  }

  Selected::new(kernels::PORTABLE, crc32_portable)
}

#[cfg(target_arch = "s390x")]
fn select_crc32c() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc32Force::Reference {
    return Selected::new(kernels::REFERENCE, crc32c_reference);
  }

  if cfg.effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32c_portable);
  }

  if caps.has(platform::caps::s390x::VECTOR) {
    return Selected::new("s390x/auto", crc32c_s390x_auto);
  }

  Selected::new(kernels::PORTABLE, crc32c_portable)
}

#[cfg(target_arch = "riscv64")]
fn select_crc32() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc32Force::Reference {
    return Selected::new(kernels::REFERENCE, crc32_reference);
  }

  if cfg.effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32_portable);
  }

  if caps.has(platform::caps::riscv::ZBC) || caps.has(platform::caps::riscv::ZVBC) {
    return Selected::new("riscv64/auto", crc32_riscv64_auto);
  }

  Selected::new(kernels::PORTABLE, crc32_portable)
}

#[cfg(target_arch = "riscv64")]
fn select_crc32c() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc32Force::Reference {
    return Selected::new(kernels::REFERENCE, crc32c_reference);
  }

  if cfg.effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32c_portable);
  }

  if caps.has(platform::caps::riscv::ZBC) || caps.has(platform::caps::riscv::ZVBC) {
    return Selected::new("riscv64/auto", crc32c_riscv64_auto);
  }

  Selected::new(kernels::PORTABLE, crc32c_portable)
}

#[cfg(not(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "powerpc64",
  target_arch = "s390x",
  target_arch = "riscv64"
)))]
fn select_crc32() -> Selected<Crc32Fn> {
  if config::get().effective_force == Crc32Force::Reference {
    return Selected::new(kernels::REFERENCE, crc32_reference);
  }
  Selected::new(kernels::PORTABLE, crc32_portable)
}

#[cfg(not(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "powerpc64",
  target_arch = "s390x",
  target_arch = "riscv64"
)))]
fn select_crc32c() -> Selected<Crc32Fn> {
  if config::get().effective_force == Crc32Force::Reference {
    return Selected::new(kernels::REFERENCE, crc32c_reference);
  }
  Selected::new(kernels::PORTABLE, crc32c_portable)
}

/// Static dispatcher for CRC-32 (IEEE).
static CRC32_DISPATCHER: Crc32Dispatcher = Crc32Dispatcher::new(select_crc32);

/// Static dispatcher for CRC-32C (Castagnoli).
static CRC32C_DISPATCHER: Crc32Dispatcher = Crc32Dispatcher::new(select_crc32c);

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
#[derive(Clone)]
pub struct Crc32 {
  state: u32,
  kernel: Crc32Fn,
  initialized: bool,
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
    Self {
      state: crc ^ !0,
      kernel: crc32_portable,
      initialized: false,
    }
  }

  /// Get the name of the currently selected backend.
  #[must_use]
  pub fn backend_name() -> &'static str {
    CRC32_DISPATCHER.backend_name()
  }

  /// Get the effective CRC-32 configuration (overrides + thresholds).
  #[must_use]
  pub fn config() -> Crc32Config {
    config::get()
  }

  /// Convenience accessor for the active CRC-32 tunables.
  #[must_use]
  pub fn tunables() -> Crc32Tunables {
    Self::config().tunables
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
    Self {
      state: !0,
      kernel: CRC32_DISPATCHER.kernel(),
      initialized: true,
    }
  }

  #[inline]
  fn with_initial(initial: u32) -> Self {
    Self {
      state: initial ^ !0,
      kernel: CRC32_DISPATCHER.kernel(),
      initialized: true,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    if !self.initialized {
      self.kernel = CRC32_DISPATCHER.kernel();
      self.initialized = true;
    }
    self.state = (self.kernel)(self.state, data);
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
#[derive(Clone)]
pub struct Crc32C {
  state: u32,
  kernel: Crc32Fn,
  initialized: bool,
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
    Self {
      state: crc ^ !0,
      kernel: crc32c_portable,
      initialized: false,
    }
  }

  /// Get the name of the currently selected backend.
  #[must_use]
  pub fn backend_name() -> &'static str {
    CRC32C_DISPATCHER.backend_name()
  }

  /// Get the effective CRC-32 configuration (overrides + thresholds).
  #[must_use]
  pub fn config() -> Crc32Config {
    config::get()
  }

  /// Convenience accessor for the active CRC-32 tunables.
  #[must_use]
  pub fn tunables() -> Crc32Tunables {
    Self::config().tunables
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
    Self {
      state: !0,
      kernel: CRC32C_DISPATCHER.kernel(),
      initialized: true,
    }
  }

  #[inline]
  fn with_initial(initial: u32) -> Self {
    Self {
      state: initial ^ !0,
      kernel: CRC32C_DISPATCHER.kernel(),
      initialized: true,
    }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    if !self.initialized {
      self.kernel = CRC32C_DISPATCHER.kernel();
      self.initialized = true;
    }
    self.state = (self.kernel)(self.state, data);
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
  fn test_kernel_name_empty_input_is_portable() {
    assert_eq!(Crc32::kernel_name_for_len(0), "portable/slice16");
    assert_eq!(Crc32C::kernel_name_for_len(0), "portable/slice16");
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
    let cfg = config::get();

    let thresholds = [
      cfg.tunables.portable_to_hwcrc,
      cfg.tunables.hwcrc_to_fusion,
      cfg.tunables.fusion_to_avx512,
      cfg.tunables.fusion_to_vpclmul,
    ];

    for &threshold in &thresholds {
      if threshold == usize::MAX || threshold == 0 || threshold > (1 << 20) {
        continue;
      }

      let size = threshold + 256;
      let data: Vec<u8> = (0..size).map(|i| (i as u8).wrapping_mul(13)).collect();

      let oneshot32 = Crc32::checksum(&data);
      let oneshot32c = Crc32C::checksum(&data);

      let mut h32 = Crc32::new();
      h32.update(&data[..16]);
      h32.update(&data[16..]);
      assert_eq!(h32.finalize(), oneshot32, "crc32 threshold={threshold}");

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

    #[cfg(target_arch = "x86_64")]
    {
      let streams_env = std::env::var("RSCRYPTO_CRC32_STREAMS_CRC32").ok();
      let caps = platform::caps();
      if force.eq_ignore_ascii_case("pclmul") || force.eq_ignore_ascii_case("clmul") {
        assert_eq!(cfg.requested_force, Crc32Force::Pclmul);
        if cfg.effective_force == Crc32Force::Pclmul && caps.has(platform::caps::x86::PCLMUL_READY) {
          if streams_env.is_some() {
            let expected = match x86_streams_for_len(len, cfg.tunables.streams_crc32) {
              8 => "x86_64/pclmul-8way",
              7 => "x86_64/pclmul-7way",
              4 => "x86_64/pclmul-4way",
              2 => "x86_64/pclmul-2way",
              _ => "x86_64/pclmul",
            };
            assert_eq!(kernel, expected);
          } else {
            assert!(kernel.starts_with("x86_64/pclmul"));
          }
        }
      }
      if force.eq_ignore_ascii_case("vpclmul") {
        assert_eq!(cfg.requested_force, Crc32Force::Vpclmul);
        if cfg.effective_force == Crc32Force::Vpclmul && caps.has(platform::caps::x86::VPCLMUL_READY) {
          if streams_env.is_some() {
            let expected = match x86_streams_for_len(len, cfg.tunables.streams_crc32) {
              8 => "x86_64/vpclmul-8way",
              7 => "x86_64/vpclmul-7way",
              4 => "x86_64/vpclmul-4way",
              2 => "x86_64/vpclmul-2way",
              _ => "x86_64/vpclmul",
            };
            assert_eq!(kernel, expected);
          } else {
            assert!(kernel.starts_with("x86_64/vpclmul"));
          }
        }
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

    #[cfg(target_arch = "x86_64")]
    {
      let streams_env = std::env::var("RSCRYPTO_CRC32_STREAMS_CRC32C").ok();
      let caps = platform::caps();
      if force.eq_ignore_ascii_case("hwcrc") || force.eq_ignore_ascii_case("crc") {
        // Verify the force mode was recognized - checksum correctness already validated above.
        // Note: kernel_name_for_len returns the policy-based selection, not the forced kernel,
        // so we don't assert on kernel name here.
        assert_eq!(cfg.requested_force, Crc32Force::Hwcrc);
        assert!(caps.has(platform::caps::x86::CRC32C_READY) || cfg.effective_force != Crc32Force::Hwcrc);
      }
      if force.eq_ignore_ascii_case("pclmul") || force.eq_ignore_ascii_case("clmul") {
        assert_eq!(cfg.requested_force, Crc32Force::Pclmul);
        if cfg.effective_force == Crc32Force::Pclmul
          && caps.has(platform::caps::x86::CRC32C_READY)
          && caps.has(platform::caps::x86::PCLMUL_READY)
        {
          if streams_env.is_some() {
            let expected = match x86_streams_for_len_crc32c(len, cfg.tunables.streams_crc32c) {
              8 => "x86_64/fusion-sse-v4s3x3-8way",
              7 => "x86_64/fusion-sse-v4s3x3-7way",
              4 => "x86_64/fusion-sse-v4s3x3-4way",
              2 => "x86_64/fusion-sse-v4s3x3-2way",
              _ => "x86_64/fusion-sse-v4s3x3",
            };
            assert_eq!(kernel, expected);
          } else {
            assert!(kernel.starts_with("x86_64/fusion-sse-v4s3x3"));
          }
        }
      }
      if force.eq_ignore_ascii_case("vpclmul") {
        assert_eq!(cfg.requested_force, Crc32Force::Vpclmul);
        if cfg.effective_force == Crc32Force::Vpclmul
          && caps.has(platform::caps::x86::CRC32C_READY)
          && caps.has(platform::caps::x86::VPCLMUL_READY)
        {
          if streams_env.is_some() {
            let expected = match x86_streams_for_len_crc32c(len, cfg.tunables.streams_crc32c) {
              8 => "x86_64/fusion-vpclmul-v3x2-8way",
              7 => "x86_64/fusion-vpclmul-v3x2-7way",
              4 => "x86_64/fusion-vpclmul-v3x2-4way",
              2 => "x86_64/fusion-vpclmul-v3x2-2way",
              _ => "x86_64/fusion-vpclmul-v3x2",
            };
            assert_eq!(kernel, expected);
          } else {
            assert!(kernel.starts_with("x86_64/fusion-vpclmul-v3x2"));
          }
        }
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
