//! CRC-32 implementations with hardware acceleration.
//!
//! This module provides two CRC-32 variants:
//! - [`Crc32`]: IEEE 802.3 polynomial (Ethernet, ZIP, PNG)
//! - [`Crc32c`]: Castagnoli polynomial (iSCSI, ext4, Btrfs)
//!
//! # Hardware Acceleration
//!
//! ## x86_64
//!
//! | Feature | Algorithms | Throughput |
//! |---------|------------|------------|
//! | SSE4.2 crc32 | CRC-32C only | ~20 GB/s |
//! | PCLMULQDQ | Both | ~15 GB/s |
//! | VPCLMULQDQ | Both | ~35 GB/s |
//!
//! ## aarch64
//!
//! | Feature | Algorithms | Throughput |
//! |---------|------------|------------|
//! | CRC32 extension | Both | ~20 GB/s |
//! | PMULL | Both | ~12 GB/s |
//! | PMULL+EOR3 | Both | ~15 GB/s |
//!
//! # Example
//!
//! ```ignore
//! use checksum::{Crc32c, Checksum, ChecksumCombine};
//!
//! // One-shot computation
//! let crc = Crc32c::checksum(b"hello world");
//!
//! // Streaming computation
//! let mut hasher = Crc32c::new();
//! hasher.update(b"hello ");
//! hasher.update(b"world");
//! assert_eq!(hasher.finalize(), crc);
//!
//! // Parallel combine
//! let crc_a = Crc32c::checksum(b"hello ");
//! let crc_b = Crc32c::checksum(b"world");
//! let combined = Crc32c::combine(crc_a, crc_b, 5);
//! assert_eq!(combined, crc);
//! ```

pub mod config;
mod kernels;
mod portable;
#[cfg(test)]
mod proptests;
mod tuned_defaults;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

// Re-export config types
use backend::dispatch::Selected;
pub use config::{Crc32Config, Crc32Force, Crc32Tunables};
#[allow(unused_imports)]
use platform::Caps;

use crate::{
  common::tables::{CRC32_IEEE_POLY, CRC32C_POLY},
  dispatchers::{Crc32Dispatcher, Crc32Fn},
};

// ─────────────────────────────────────────────────────────────────────────────
// Auto-Dispatch Functions (Length-Based Kernel Selection)
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32-IEEE portable kernel wrapper.
fn crc32_ieee_portable(crc: u32, data: &[u8]) -> u32 {
  portable::crc32_ieee_slice16(crc, data)
}

/// CRC-32C portable kernel wrapper.
fn crc32c_portable(crc: u32, data: &[u8]) -> u32 {
  portable::crc32c_slice16(crc, data)
}

#[cfg(target_arch = "x86_64")]
fn crc32_ieee_x86_64_auto(crc: u32, data: &[u8]) -> u32 {
  let len = data.len();

  // Tiny regime: avoid any dispatch overhead
  if len < 16 {
    return crc32_ieee_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  // Handle forced backend selection
  match cfg.effective_force {
    Crc32Force::Portable => return crc32_ieee_portable(crc, data),
    Crc32Force::Vpclmul if caps.has(platform::caps::x86::VPCLMUL_READY) => {
      return x86_64::crc32_ieee_vpclmul_safe(crc, data);
    }
    Crc32Force::Vpclmul if caps.has(platform::caps::x86::PCLMUL_READY) => {
      return x86_64::crc32_ieee_pclmul_safe(crc, data);
    }
    Crc32Force::Vpclmul => return crc32_ieee_portable(crc, data),
    Crc32Force::Pclmul if caps.has(platform::caps::x86::PCLMUL_READY) => {
      return x86_64::crc32_ieee_pclmul_safe(crc, data);
    }
    Crc32Force::Pclmul => return crc32_ieee_portable(crc, data),
    _ => {}
  }

  // Auto selection with thresholds
  // CRC32-IEEE has no HW instruction on x86, so the tiers are:
  // portable → PCLMUL → VPCLMUL

  // VPCLMUL tier (if available and above threshold)
  if caps.has(platform::caps::x86::VPCLMUL_READY) && len >= cfg.tunables.pclmul_to_vpclmul {
    return x86_64::crc32_ieee_vpclmul_safe(crc, data);
  }

  // PCLMUL tier (64+ bytes minimum for efficient folding)
  if caps.has(platform::caps::x86::PCLMUL_READY) && len >= 64 {
    return x86_64::crc32_ieee_pclmul_safe(crc, data);
  }

  crc32_ieee_portable(crc, data)
}

#[cfg(target_arch = "x86_64")]
fn crc32c_x86_64_auto(crc: u32, data: &[u8]) -> u32 {
  let len = data.len();

  // Tiny regime: avoid any dispatch overhead
  if len < 8 {
    return crc32c_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  // Handle forced backend selection
  match cfg.effective_force {
    Crc32Force::Portable => return crc32c_portable(crc, data),
    Crc32Force::Vpclmul if caps.has(platform::caps::x86::VPCLMUL_READY) => {
      return x86_64::crc32c_vpclmul_safe(crc, data);
    }
    Crc32Force::Vpclmul if caps.has(platform::caps::x86::PCLMUL_READY) => {
      // Fallback to PCLMUL fusion
      return x86_64::crc32c_pclmul_safe(crc, data);
    }
    Crc32Force::Vpclmul => return crc32c_portable(crc, data),
    Crc32Force::Pclmul if caps.has(platform::caps::x86::PCLMUL_READY) => {
      return x86_64::crc32c_pclmul_safe(crc, data);
    }
    Crc32Force::Pclmul => return crc32c_portable(crc, data),
    Crc32Force::Sse42 if caps.has(platform::caps::x86::SSE42) => {
      return x86_64::crc32c_sse42_safe(crc, data);
    }
    Crc32Force::Sse42 => return crc32c_portable(crc, data),
    _ => {}
  }

  // Auto selection with thresholds for CRC32C:
  // The SSE4.2 crc32 instruction is extremely fast for small/medium buffers.
  // The PCLMUL fusion only wins for larger buffers due to setup overhead.
  // Tiers: portable → SSE4.2 HW → PCLMUL fusion → VPCLMUL

  // Below portable_to_hw: use portable
  if len < cfg.tunables.portable_to_hw {
    return crc32c_portable(crc, data);
  }

  // VPCLMUL tier (if available and above threshold)
  if caps.has(platform::caps::x86::VPCLMUL_READY) && len >= cfg.tunables.pclmul_to_vpclmul {
    return x86_64::crc32c_vpclmul_safe(crc, data);
  }

  // PCLMUL fusion tier (above hw_to_clmul threshold)
  if caps.has(platform::caps::x86::PCLMUL_READY)
    && caps.has(platform::caps::x86::SSE42)
    && len >= cfg.tunables.hw_to_clmul
  {
    return x86_64::crc32c_pclmul_safe(crc, data);
  }

  // SSE4.2 hardware CRC tier (default fast path)
  if caps.has(platform::caps::x86::SSE42) {
    return x86_64::crc32c_sse42_safe(crc, data);
  }

  // Pure PCLMUL (no SSE4.2 fusion) for very rare CPUs
  if caps.has(platform::caps::x86::PCLMUL_READY) {
    return x86_64::crc32c_pclmul_safe(crc, data);
  }

  crc32c_portable(crc, data)
}

#[cfg(target_arch = "aarch64")]
fn crc32_ieee_aarch64_auto(crc: u32, data: &[u8]) -> u32 {
  let len = data.len();

  if len < 8 {
    return crc32_ieee_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  // Handle forced backend selection
  match cfg.effective_force {
    Crc32Force::Portable => return crc32_ieee_portable(crc, data),
    Crc32Force::PmullEor3 if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) => {
      return aarch64::crc32_ieee_pmull_eor3_safe(crc, data);
    }
    Crc32Force::PmullEor3 => return crc32_ieee_portable(crc, data),
    Crc32Force::Pmull if caps.has(platform::caps::aarch64::PMULL_READY) => {
      return aarch64::crc32_ieee_pmull_safe(crc, data);
    }
    Crc32Force::Pmull => return crc32_ieee_portable(crc, data),
    Crc32Force::ArmCrc if caps.has(platform::caps::aarch64::CRC) => {
      return aarch64::crc32_ieee_arm_safe(crc, data);
    }
    Crc32Force::ArmCrc => return crc32_ieee_portable(crc, data),
    _ => {}
  }

  // Auto selection with thresholds
  // CRC32-IEEE on ARM: ARM CRC instruction handles both polynomials
  // Tiers: portable → ARM CRC → PMULL → PMULL+EOR3

  if len < cfg.tunables.portable_to_hw {
    return crc32_ieee_portable(crc, data);
  }

  // PMULL+EOR3 tier (highest throughput, requires large buffers)
  if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) && len >= cfg.tunables.hw_to_clmul {
    return aarch64::crc32_ieee_pmull_eor3_safe(crc, data);
  }

  // PMULL tier (for large buffers)
  if caps.has(platform::caps::aarch64::PMULL_READY) && len >= cfg.tunables.hw_to_clmul {
    return aarch64::crc32_ieee_pmull_safe(crc, data);
  }

  // ARM CRC instruction (fast for small/medium buffers)
  if caps.has(platform::caps::aarch64::CRC) {
    return aarch64::crc32_ieee_arm_safe(crc, data);
  }

  // PMULL fallback (if no ARM CRC)
  if caps.has(platform::caps::aarch64::PMULL_READY) {
    return aarch64::crc32_ieee_pmull_safe(crc, data);
  }

  crc32_ieee_portable(crc, data)
}

#[cfg(target_arch = "aarch64")]
fn crc32c_aarch64_auto(crc: u32, data: &[u8]) -> u32 {
  let len = data.len();

  if len < 8 {
    return crc32c_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  // Handle forced backend selection
  match cfg.effective_force {
    Crc32Force::Portable => return crc32c_portable(crc, data),
    Crc32Force::PmullEor3 if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) => {
      return aarch64::crc32c_pmull_eor3_safe(crc, data);
    }
    Crc32Force::PmullEor3 => return crc32c_portable(crc, data),
    Crc32Force::Pmull if caps.has(platform::caps::aarch64::PMULL_READY) => {
      return aarch64::crc32c_pmull_safe(crc, data);
    }
    Crc32Force::Pmull => return crc32c_portable(crc, data),
    Crc32Force::ArmCrc if caps.has(platform::caps::aarch64::CRC) => {
      return aarch64::crc32c_arm_safe(crc, data);
    }
    Crc32Force::ArmCrc => return crc32c_portable(crc, data),
    _ => {}
  }

  // Auto selection with thresholds
  // CRC32C on ARM: ARM CRC32C instruction is very fast
  // Tiers: portable → ARM CRC32C → PMULL → PMULL+EOR3

  if len < cfg.tunables.portable_to_hw {
    return crc32c_portable(crc, data);
  }

  // PMULL+EOR3 tier (highest throughput, requires large buffers)
  if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) && len >= cfg.tunables.hw_to_clmul {
    return aarch64::crc32c_pmull_eor3_safe(crc, data);
  }

  // PMULL tier (for large buffers)
  if caps.has(platform::caps::aarch64::PMULL_READY) && len >= cfg.tunables.hw_to_clmul {
    return aarch64::crc32c_pmull_safe(crc, data);
  }

  // ARM CRC32C instruction (fast for small/medium buffers)
  if caps.has(platform::caps::aarch64::CRC) {
    return aarch64::crc32c_arm_safe(crc, data);
  }

  // PMULL fallback (if no ARM CRC)
  if caps.has(platform::caps::aarch64::PMULL_READY) {
    return aarch64::crc32c_pmull_safe(crc, data);
  }

  crc32c_portable(crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Selection
// ─────────────────────────────────────────────────────────────────────────────

/// Select the best CRC-32-IEEE kernel for the current platform.
fn select_crc32_ieee() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let config = config::get();

  // Explicit portable override always wins.
  if config.effective_force == Crc32Force::Portable {
    return Selected::new("portable/slice16", crc32_ieee_portable);
  }

  #[cfg(target_arch = "x86_64")]
  {
    use platform::caps::x86;

    // If any SIMD feature is available, use the auto function for length-based dispatch
    if caps.has(x86::PCLMUL_READY) || caps.has(x86::VPCLMUL_READY) {
      return Selected::new("x86_64/auto", crc32_ieee_x86_64_auto);
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    use platform::caps::aarch64 as arm;

    // If any accelerator is available, use the auto function
    if caps.has(arm::CRC) || caps.has(arm::PMULL_READY) {
      return Selected::new("aarch64/auto", crc32_ieee_aarch64_auto);
    }
  }

  Selected::new("portable/slice16", crc32_ieee_portable)
}

/// Select the best CRC-32C kernel for the current platform.
fn select_crc32c() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let config = config::get();

  // Explicit portable override always wins.
  if config.effective_force == Crc32Force::Portable {
    return Selected::new("portable/slice16", crc32c_portable);
  }

  #[cfg(target_arch = "x86_64")]
  {
    use platform::caps::x86;

    // If any accelerator is available, use the auto function for length-based dispatch
    if caps.has(x86::SSE42) || caps.has(x86::PCLMUL_READY) || caps.has(x86::VPCLMUL_READY) {
      return Selected::new("x86_64/auto", crc32c_x86_64_auto);
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    use platform::caps::aarch64 as arm;

    // If any accelerator is available, use the auto function
    if caps.has(arm::CRC) || caps.has(arm::PMULL_READY) {
      return Selected::new("aarch64/auto", crc32c_aarch64_auto);
    }
  }

  Selected::new("portable/slice16", crc32c_portable)
}

// Static dispatchers
static CRC32_IEEE_DISPATCHER: Crc32Dispatcher = Crc32Dispatcher::new(select_crc32_ieee);
static CRC32C_DISPATCHER: Crc32Dispatcher = Crc32Dispatcher::new(select_crc32c);

// ─────────────────────────────────────────────────────────────────────────────
// Type Definitions
// ─────────────────────────────────────────────────────────────────────────────

define_crc32_type!(
  /// CRC-32 (IEEE 802.3) checksum.
  ///
  /// Used by Ethernet, PKZIP, Gzip, PNG, MPEG-2, and many other formats.
  ///
  /// # Polynomial
  ///
  /// - Normal form: `0x04C11DB7`
  /// - Reflected form: `0xEDB88320`
  ///
  /// # Example
  ///
  /// ```ignore
  /// use checksum::{Crc32, Checksum};
  ///
  /// let crc = Crc32::checksum(b"hello world");
  /// assert_eq!(crc, 0x0D4A1185);
  /// ```
  pub struct Crc32 {
    poly: CRC32_IEEE_POLY,
    dispatcher: CRC32_IEEE_DISPATCHER,
  }
);

define_crc32_type!(
  /// CRC-32C (Castagnoli) checksum.
  ///
  /// Used by iSCSI, SCTP, ext4, Btrfs, LevelDB, and modern storage systems.
  /// This polynomial was chosen for its superior error detection properties
  /// and has dedicated hardware support on both x86_64 (SSE4.2) and aarch64.
  ///
  /// # Polynomial
  ///
  /// - Normal form: `0x1EDC6F41`
  /// - Reflected form: `0x82F63B78`
  ///
  /// # Example
  ///
  /// ```ignore
  /// use checksum::{Crc32c, Checksum};
  ///
  /// let crc = Crc32c::checksum(b"hello world");
  /// assert_eq!(crc, 0xC99465AA);
  /// ```
  pub struct Crc32c {
    poly: CRC32C_POLY,
    dispatcher: CRC32C_DISPATCHER,
  }
);

// Type aliases for compatibility
/// Alias for [`Crc32`] (IEEE 802.3 polynomial).
pub type Crc32Ieee = Crc32;

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Name Selection
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the kernel name that would be selected for CRC-32C for a given buffer length.
///
/// This is intended for debugging/benchmarking and does not allocate.
///
/// Note: This defaults to CRC-32C selection logic. For CRC-32-IEEE-specific
/// selection, use [`crc32_ieee_selected_kernel_name`].
#[must_use]
pub fn crc32_selected_kernel_name(len: usize) -> &'static str {
  crc32c_selected_kernel_name(len)
}

/// Returns the kernel name that would be selected for CRC-32C for a given buffer length.
///
/// This is intended for debugging/benchmarking and does not allocate.
#[must_use]
pub fn crc32c_selected_kernel_name(len: usize) -> &'static str {
  let config = config::get();
  let caps = platform::caps();

  crc32c_kernel_name_for_caps_and_len(caps, &config, len)
}

/// Returns the kernel name that would be selected for CRC-32-IEEE for a given buffer length.
///
/// This is intended for debugging/benchmarking and does not allocate.
#[must_use]
#[allow(dead_code)]
pub fn crc32_ieee_selected_kernel_name(len: usize) -> &'static str {
  let config = config::get();
  let caps = platform::caps();

  crc32_ieee_kernel_name_for_caps_and_len(caps, &config, len)
}

#[allow(unused_variables)]
fn crc32c_kernel_name_for_caps_and_len(caps: Caps, config: &Crc32Config, len: usize) -> &'static str {
  #[cfg(target_arch = "x86_64")]
  {
    use platform::caps::x86;

    if config.effective_force == Crc32Force::Portable {
      return kernels::PORTABLE;
    }

    // Tiny regime
    if len < 8 {
      return kernels::PORTABLE;
    }

    // Below portable_to_hw threshold
    if len < config.tunables.portable_to_hw {
      return kernels::PORTABLE;
    }

    // VPCLMUL tier
    if caps.has(x86::VPCLMUL_READY) && len >= config.tunables.pclmul_to_vpclmul {
      return kernels::x86_64::VPCLMUL;
    }

    // PCLMUL fusion tier
    if caps.has(x86::PCLMUL_READY) && caps.has(x86::SSE42) && len >= config.tunables.hw_to_clmul {
      return kernels::x86_64::PCLMUL;
    }

    // SSE4.2 HW CRC tier
    if caps.has(x86::SSE42) {
      return kernels::x86_64::SSE42_CRC32C;
    }

    // Pure PCLMUL fallback
    if caps.has(x86::PCLMUL_READY) {
      return kernels::x86_64::PCLMUL;
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    use platform::caps::aarch64 as arm;

    if config.effective_force == Crc32Force::Portable {
      return kernels::PORTABLE;
    }

    if len < 8 {
      return kernels::PORTABLE;
    }

    if len < config.tunables.portable_to_hw {
      return kernels::PORTABLE;
    }

    // PMULL+EOR3 tier
    if caps.has(arm::PMULL_EOR3_READY) && len >= config.tunables.hw_to_clmul {
      return kernels::aarch64::PMULL_EOR3;
    }

    // PMULL tier
    if caps.has(arm::PMULL_READY) && len >= config.tunables.hw_to_clmul {
      return kernels::aarch64::PMULL;
    }

    // ARM CRC32C tier
    if caps.has(arm::CRC) {
      return kernels::aarch64::ARM_CRC32C;
    }

    // PMULL fallback
    if caps.has(arm::PMULL_READY) {
      return kernels::aarch64::PMULL;
    }
  }

  kernels::PORTABLE
}

#[allow(unused_variables)]
#[allow(dead_code)]
fn crc32_ieee_kernel_name_for_caps_and_len(caps: Caps, config: &Crc32Config, len: usize) -> &'static str {
  #[cfg(target_arch = "x86_64")]
  {
    use platform::caps::x86;

    if config.effective_force == Crc32Force::Portable {
      return kernels::PORTABLE;
    }

    // Tiny regime
    if len < 16 {
      return kernels::PORTABLE;
    }

    // VPCLMUL tier (CRC32-IEEE has no HW instruction, so skip directly to CLMUL)
    if caps.has(x86::VPCLMUL_READY) && len >= config.tunables.pclmul_to_vpclmul {
      return kernels::x86_64::VPCLMUL;
    }

    // PCLMUL tier (64+ bytes)
    if caps.has(x86::PCLMUL_READY) && len >= 64 {
      return kernels::x86_64::PCLMUL;
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    use platform::caps::aarch64 as arm;

    if config.effective_force == Crc32Force::Portable {
      return kernels::PORTABLE;
    }

    if len < 8 {
      return kernels::PORTABLE;
    }

    if len < config.tunables.portable_to_hw {
      return kernels::PORTABLE;
    }

    // PMULL+EOR3 tier
    if caps.has(arm::PMULL_EOR3_READY) && len >= config.tunables.hw_to_clmul {
      return kernels::aarch64::PMULL_EOR3;
    }

    // PMULL tier
    if caps.has(arm::PMULL_READY) && len >= config.tunables.hw_to_clmul {
      return kernels::aarch64::PMULL;
    }

    // ARM CRC32 tier
    if caps.has(arm::CRC) {
      return kernels::aarch64::ARM_CRC32;
    }

    // PMULL fallback
    if caps.has(arm::PMULL_READY) {
      return kernels::aarch64::PMULL;
    }
  }

  kernels::PORTABLE
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  extern crate alloc;
  use alloc::vec::Vec;

  use super::*;
  use crate::{Checksum, ChecksumCombine};

  // Known test vectors
  const CRC32_IEEE_EMPTY: u32 = 0x0000_0000;
  const CRC32_IEEE_HELLO_WORLD: u32 = 0x0D4A_1185;

  const CRC32C_EMPTY: u32 = 0x0000_0000;
  const CRC32C_HELLO_WORLD: u32 = 0xC99465AA;

  #[test]
  fn test_crc32_empty() {
    assert_eq!(Crc32::checksum(b""), CRC32_IEEE_EMPTY);
  }

  #[test]
  fn test_crc32_hello_world() {
    assert_eq!(Crc32::checksum(b"hello world"), CRC32_IEEE_HELLO_WORLD);
  }

  #[test]
  fn test_crc32c_empty() {
    assert_eq!(Crc32c::checksum(b""), CRC32C_EMPTY);
  }

  #[test]
  fn test_crc32c_hello_world() {
    assert_eq!(Crc32c::checksum(b"hello world"), CRC32C_HELLO_WORLD);
  }

  #[test]
  fn test_crc32_streaming() {
    let full = Crc32::checksum(b"hello world");

    let mut hasher = Crc32::new();
    hasher.update(b"hello ");
    hasher.update(b"world");
    assert_eq!(hasher.finalize(), full);
  }

  #[test]
  fn test_crc32c_streaming() {
    let full = Crc32c::checksum(b"hello world");

    let mut hasher = Crc32c::new();
    hasher.update(b"hello ");
    hasher.update(b"world");
    assert_eq!(hasher.finalize(), full);
  }

  #[test]
  fn test_crc32_combine() {
    let full = Crc32::checksum(b"hello world");
    let crc_a = Crc32::checksum(b"hello ");
    let crc_b = Crc32::checksum(b"world");
    let combined = Crc32::combine(crc_a, crc_b, 5);
    assert_eq!(combined, full);
  }

  #[test]
  fn test_crc32c_combine() {
    let full = Crc32c::checksum(b"hello world");
    let crc_a = Crc32c::checksum(b"hello ");
    let crc_b = Crc32c::checksum(b"world");
    let combined = Crc32c::combine(crc_a, crc_b, 5);
    assert_eq!(combined, full);
  }

  #[test]
  fn test_crc32_resume() {
    let data = b"hello world";
    let full = Crc32::checksum(data);

    // Compute in two parts using resume
    let part1 = Crc32::checksum(b"hello ");
    let hasher = Crc32::resume(part1);
    let mut hasher = hasher;
    hasher.update(b"world");
    let resumed = hasher.finalize();

    assert_eq!(resumed, full);
  }

  #[test]
  fn test_crc32c_resume() {
    let data = b"hello world";
    let full = Crc32c::checksum(data);

    let part1 = Crc32c::checksum(b"hello ");
    let hasher = Crc32c::resume(part1);
    let mut hasher = hasher;
    hasher.update(b"world");
    let resumed = hasher.finalize();

    assert_eq!(resumed, full);
  }

  #[test]
  fn test_backend_name() {
    // Just verify it returns something
    let name = Crc32::backend_name();
    assert!(!name.is_empty());

    let name = Crc32c::backend_name();
    assert!(!name.is_empty());
  }

  #[test]
  fn test_various_lengths() {
    // Test various lengths to exercise different code paths
    for len in [0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 512, 1024, 4096] {
      let data: Vec<u8> = (0..len).map(|i| i as u8).collect();

      // Just verify it doesn't panic and returns consistent results
      let crc1 = Crc32c::checksum(&data);
      let crc2 = Crc32c::checksum(&data);
      assert_eq!(crc1, crc2, "Inconsistent result at length {len}");
    }
  }
}
