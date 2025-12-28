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

pub(crate) mod config;
mod kernels;
mod portable;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

use backend::dispatch::Selected;
#[allow(unused_imports)]
pub use config::{Crc32Config, Crc32Force, Crc32Tunables};
#[allow(unused_imports)]
pub(super) use traits::{Checksum, ChecksumCombine};

use crate::{
  common::{
    combine::{Gf2Matrix32, generate_shift8_matrix_32},
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

// Folding block sizing (used by SIMD tiers and stream gating).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const CRC32_FOLD_BLOCK_BYTES: usize = 128;

#[cfg(target_arch = "x86_64")]
#[inline]
const fn x86_streams_for_len(len: usize, streams: u8) -> u8 {
  if streams >= 8 && len >= 8 * 2 * CRC32_FOLD_BLOCK_BYTES {
    return 8;
  }
  if streams >= 7 && len >= 7 * 2 * CRC32_FOLD_BLOCK_BYTES {
    return 7;
  }
  if streams >= 4 && len >= 4 * 2 * CRC32_FOLD_BLOCK_BYTES {
    return 4;
  }
  if streams >= 2 && len >= 2 * 2 * CRC32_FOLD_BLOCK_BYTES {
    return 2;
  }
  1
}

#[cfg(target_arch = "aarch64")]
#[inline]
const fn aarch64_streams_for_len(len: usize, streams: u8) -> u8 {
  // Our aarch64 stream slots are [1, 2, 3] (index 2 is shared by 3/4).
  if streams >= 3 && len >= 3 * 2 * CRC32_FOLD_BLOCK_BYTES {
    return 3;
  }
  if streams >= 2 && len >= 2 * 2 * CRC32_FOLD_BLOCK_BYTES {
    return 2;
  }
  1
}

#[inline]
#[must_use]
pub(crate) fn crc32_selected_kernel_name(len: usize) -> &'static str {
  let cfg = config::get();

  if cfg.effective_force == Crc32Force::Portable {
    return kernels::PORTABLE;
  }

  if cfg.effective_force == Crc32Force::Auto && len < cfg.tunables.portable_to_hwcrc {
    return kernels::PORTABLE;
  }

  #[cfg(target_arch = "x86_64")]
  {
    let caps = platform::caps();
    use kernels::x86_64::*;
    let streams = x86_streams_for_len(len, cfg.tunables.streams);

    match cfg.effective_force {
      Crc32Force::Vpclmul if caps.has(platform::caps::x86::VPCLMUL_READY) => {
        return kernels::select_name(CRC32_VPCLMUL_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      Crc32Force::Vpclmul => return kernels::PORTABLE,
      Crc32Force::Pclmul if caps.has(platform::caps::x86::PCLMUL_READY) => {
        return kernels::select_name(CRC32_PCLMUL_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      Crc32Force::Pclmul => return kernels::PORTABLE,
      _ => {}
    }

    if cfg.effective_force == Crc32Force::Auto
      && len >= cfg.tunables.hwcrc_to_fusion
      && caps.has(platform::caps::x86::PCLMUL_READY)
    {
      if len >= cfg.tunables.fusion_to_vpclmul && caps.has(platform::caps::x86::VPCLMUL_READY) {
        return kernels::select_name(CRC32_VPCLMUL_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      return kernels::select_name(CRC32_PCLMUL_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    let caps = platform::caps();
    use kernels::aarch64::*;
    let streams = aarch64_streams_for_len(len, cfg.tunables.streams);

    match cfg.effective_force {
      Crc32Force::PmullEor3 if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) => {
        return kernels::select_name(CRC32_PMULL_EOR3_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      Crc32Force::Pmull => {
        if caps.has(platform::caps::aarch64::PMULL_READY) {
          return kernels::select_name(CRC32_PMULL_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
        }
        return kernels::PORTABLE;
      }
      Crc32Force::Hwcrc => {
        if caps.has(platform::caps::aarch64::CRC_READY) {
          return kernels::select_name(CRC32_HWCRC_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
        }
        return kernels::PORTABLE;
      }
      _ => {}
    }

    // Auto selection: prefer fusion above threshold, otherwise HWCRC.
    if len >= cfg.tunables.hwcrc_to_fusion {
      if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
        return kernels::select_name(CRC32_PMULL_EOR3_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      if caps.has(platform::caps::aarch64::PMULL_READY) {
        return kernels::select_name(CRC32_PMULL_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
    }

    if caps.has(platform::caps::aarch64::CRC_READY) {
      return kernels::select_name(CRC32_HWCRC_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
    }
  }

  kernels::PORTABLE
}

#[inline]
#[must_use]
pub(crate) fn crc32c_selected_kernel_name(len: usize) -> &'static str {
  let cfg = config::get();
  let caps = platform::caps();

  if cfg.effective_force == Crc32Force::Portable {
    return kernels::PORTABLE;
  }

  if cfg.effective_force == Crc32Force::Auto && len < cfg.tunables.portable_to_hwcrc {
    return kernels::PORTABLE;
  }

  #[cfg(target_arch = "x86_64")]
  {
    use kernels::x86_64::*;
    let streams = x86_streams_for_len(len, cfg.tunables.streams);

    match cfg.effective_force {
      Crc32Force::Vpclmul
        if caps.has(platform::caps::x86::VPCLMUL_READY) && caps.has(platform::caps::x86::CRC32C_READY) =>
      {
        return kernels::select_name(CRC32C_FUSION_VPCLMUL_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      Crc32Force::Pclmul
        if caps.has(platform::caps::x86::PCLMUL_READY) && caps.has(platform::caps::x86::CRC32C_READY) =>
      {
        return kernels::select_name(CRC32C_FUSION_SSE_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      Crc32Force::Hwcrc if caps.has(platform::caps::x86::CRC32C_READY) => {
        return kernels::select_name(CRC32C_HWCRC_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      Crc32Force::Hwcrc => return kernels::PORTABLE,
      _ => {}
    }

    // Auto selection: prefer fusion above threshold.
    if len >= cfg.tunables.hwcrc_to_fusion && caps.has(platform::caps::x86::CRC32C_READY) {
      if caps.has(platform::caps::x86::VPCLMUL_READY) && len >= cfg.tunables.fusion_to_vpclmul {
        return kernels::select_name(CRC32C_FUSION_VPCLMUL_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      if caps.has(platform::caps::x86::AVX512_READY)
        && len >= cfg.tunables.fusion_to_avx512
        && caps.has(platform::caps::x86::PCLMULQDQ)
      {
        return kernels::select_name(CRC32C_FUSION_AVX512_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      if caps.has(platform::caps::x86::PCLMUL_READY) {
        return kernels::select_name(CRC32C_FUSION_SSE_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
    }

    if caps.has(platform::caps::x86::CRC32C_READY) {
      return kernels::select_name(CRC32C_HWCRC_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    use kernels::aarch64::*;
    let streams = aarch64_streams_for_len(len, cfg.tunables.streams);

    match cfg.effective_force {
      Crc32Force::PmullEor3 if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) => {
        return kernels::select_name(CRC32C_PMULL_EOR3_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      Crc32Force::Pmull => {
        if caps.has(platform::caps::aarch64::PMULL_READY) {
          return kernels::select_name(CRC32C_PMULL_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
        }
        return kernels::PORTABLE;
      }
      Crc32Force::Hwcrc => {
        if caps.has(platform::caps::aarch64::CRC_READY) {
          return kernels::select_name(CRC32C_HWCRC_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
        }
        return kernels::PORTABLE;
      }
      _ => {}
    }

    // Auto selection: prefer fusion above threshold, otherwise HWCRC.
    if len >= cfg.tunables.hwcrc_to_fusion {
      if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
        return kernels::select_name(CRC32C_PMULL_EOR3_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      if caps.has(platform::caps::aarch64::PMULL_READY) {
        return kernels::select_name(CRC32C_PMULL_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
    }

    if caps.has(platform::caps::aarch64::CRC_READY) {
      return kernels::select_name(CRC32C_HWCRC_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
    }
  }

  kernels::PORTABLE
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto Kernels (architecture-specific)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn crc32c_x86_64_auto(crc: u32, data: &[u8]) -> u32 {
  let cfg = config::get();
  let caps = platform::caps();
  let len = data.len();

  if cfg.effective_force == Crc32Force::Portable
    || (cfg.effective_force == Crc32Force::Auto && len < cfg.tunables.portable_to_hwcrc)
  {
    return crc32c_portable(crc, data);
  }

  use kernels::x86_64::*;
  let streams = x86_streams_for_len(len, cfg.tunables.streams);

  match cfg.effective_force {
    Crc32Force::Hwcrc => {
      if caps.has(platform::caps::x86::CRC32C_READY) {
        return kernels::dispatch_streams(&CRC32C_HWCRC, streams, crc, data);
      }
      return crc32c_portable(crc, data);
    }
    Crc32Force::Pclmul => {
      if caps.has(platform::caps::x86::CRC32C_READY) && caps.has(platform::caps::x86::PCLMUL_READY) {
        return kernels::dispatch_streams(&CRC32C_FUSION_SSE, streams, crc, data);
      }
      return crc32c_portable(crc, data);
    }
    Crc32Force::Vpclmul => {
      if caps.has(platform::caps::x86::CRC32C_READY) && caps.has(platform::caps::x86::VPCLMUL_READY) {
        return kernels::dispatch_streams(&CRC32C_FUSION_VPCLMUL, streams, crc, data);
      }
      return crc32c_portable(crc, data);
    }
    _ => {}
  }

  if caps.has(platform::caps::x86::CRC32C_READY) {
    if len >= cfg.tunables.hwcrc_to_fusion {
      if caps.has(platform::caps::x86::VPCLMUL_READY) && len >= cfg.tunables.fusion_to_vpclmul {
        return kernels::dispatch_streams(&CRC32C_FUSION_VPCLMUL, streams, crc, data);
      }
      if caps.has(platform::caps::x86::AVX512_READY)
        && len >= cfg.tunables.fusion_to_avx512
        && caps.has(platform::caps::x86::PCLMULQDQ)
      {
        return kernels::dispatch_streams(&CRC32C_FUSION_AVX512, streams, crc, data);
      }
      if caps.has(platform::caps::x86::PCLMUL_READY) {
        return kernels::dispatch_streams(&CRC32C_FUSION_SSE, streams, crc, data);
      }
    }
    return kernels::dispatch_streams(&CRC32C_HWCRC, streams, crc, data);
  }

  crc32c_portable(crc, data)
}

#[cfg(target_arch = "x86_64")]
fn crc32_x86_64_auto(crc: u32, data: &[u8]) -> u32 {
  let cfg = config::get();
  let caps = platform::caps();
  let len = data.len();

  if cfg.effective_force == Crc32Force::Portable {
    return crc32_portable(crc, data);
  }

  use kernels::x86_64::*;
  let streams = x86_streams_for_len(len, cfg.tunables.streams);

  match cfg.effective_force {
    Crc32Force::Vpclmul => {
      if caps.has(platform::caps::x86::VPCLMUL_READY) {
        return kernels::dispatch_streams(&CRC32_VPCLMUL, streams, crc, data);
      }
      return crc32_portable(crc, data);
    }
    Crc32Force::Pclmul => {
      if caps.has(platform::caps::x86::PCLMUL_READY) {
        return kernels::dispatch_streams(&CRC32_PCLMUL, streams, crc, data);
      }
      return crc32_portable(crc, data);
    }
    Crc32Force::Hwcrc => return crc32_portable(crc, data),
    _ => {}
  }

  if len >= cfg.tunables.hwcrc_to_fusion && caps.has(platform::caps::x86::PCLMUL_READY) {
    if len >= cfg.tunables.fusion_to_vpclmul && caps.has(platform::caps::x86::VPCLMUL_READY) {
      return kernels::dispatch_streams(&CRC32_VPCLMUL, streams, crc, data);
    }
    return kernels::dispatch_streams(&CRC32_PCLMUL, streams, crc, data);
  }

  crc32_portable(crc, data)
}

#[cfg(target_arch = "aarch64")]
fn crc32_aarch64_auto(crc: u32, data: &[u8]) -> u32 {
  let cfg = config::get();
  let caps = platform::caps();
  let len = data.len();

  if cfg.effective_force == Crc32Force::Portable
    || (cfg.effective_force == Crc32Force::Auto && len < cfg.tunables.portable_to_hwcrc)
  {
    return crc32_portable(crc, data);
  }

  use kernels::aarch64::*;
  let streams = aarch64_streams_for_len(len, cfg.tunables.streams);

  match cfg.effective_force {
    Crc32Force::PmullEor3 => {
      if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
        return kernels::dispatch_streams(&CRC32_PMULL_EOR3, streams, crc, data);
      }
      return crc32_portable(crc, data);
    }
    Crc32Force::Pmull => {
      if caps.has(platform::caps::aarch64::PMULL_READY) {
        return kernels::dispatch_streams(&CRC32_PMULL, streams, crc, data);
      }
      return crc32_portable(crc, data);
    }
    Crc32Force::Hwcrc => {
      if caps.has(platform::caps::aarch64::CRC_READY) {
        return kernels::dispatch_streams(&CRC32_HWCRC, streams, crc, data);
      }
      return crc32_portable(crc, data);
    }
    _ => {}
  }

  if caps.has(platform::caps::aarch64::CRC_READY) {
    if len >= cfg.tunables.hwcrc_to_fusion {
      if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
        return kernels::dispatch_streams(&CRC32_PMULL_EOR3, streams, crc, data);
      }
      if caps.has(platform::caps::aarch64::PMULL_READY) {
        return kernels::dispatch_streams(&CRC32_PMULL, streams, crc, data);
      }
    }
    return kernels::dispatch_streams(&CRC32_HWCRC, streams, crc, data);
  }

  crc32_portable(crc, data)
}

#[cfg(target_arch = "aarch64")]
fn crc32c_aarch64_auto(crc: u32, data: &[u8]) -> u32 {
  let cfg = config::get();
  let caps = platform::caps();
  let len = data.len();

  if cfg.effective_force == Crc32Force::Portable
    || (cfg.effective_force == Crc32Force::Auto && len < cfg.tunables.portable_to_hwcrc)
  {
    return crc32c_portable(crc, data);
  }

  use kernels::aarch64::*;
  let streams = aarch64_streams_for_len(len, cfg.tunables.streams);

  match cfg.effective_force {
    Crc32Force::PmullEor3 => {
      if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
        return kernels::dispatch_streams(&CRC32C_PMULL_EOR3, streams, crc, data);
      }
      return crc32c_portable(crc, data);
    }
    Crc32Force::Pmull => {
      if caps.has(platform::caps::aarch64::PMULL_READY) {
        return kernels::dispatch_streams(&CRC32C_PMULL, streams, crc, data);
      }
      return crc32c_portable(crc, data);
    }
    Crc32Force::Hwcrc => {
      if caps.has(platform::caps::aarch64::CRC_READY) {
        return kernels::dispatch_streams(&CRC32C_HWCRC, streams, crc, data);
      }
      return crc32c_portable(crc, data);
    }
    _ => {}
  }

  if caps.has(platform::caps::aarch64::CRC_READY) {
    if len >= cfg.tunables.hwcrc_to_fusion {
      if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
        return kernels::dispatch_streams(&CRC32C_PMULL_EOR3, streams, crc, data);
      }
      if caps.has(platform::caps::aarch64::PMULL_READY) {
        return kernels::dispatch_streams(&CRC32C_PMULL, streams, crc, data);
      }
    }
    return kernels::dispatch_streams(&CRC32C_HWCRC, streams, crc, data);
  }

  crc32c_portable(crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Selection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn select_crc32() -> Selected<Crc32Fn> {
  let caps = platform::caps();

  if config::get().effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32_portable);
  }

  if caps.has(platform::caps::x86::PCLMUL_READY) {
    return Selected::new("x86_64/auto", crc32_x86_64_auto);
  }

  Selected::new(kernels::PORTABLE, crc32_portable)
}

#[cfg(target_arch = "x86_64")]
fn select_crc32c() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32c_portable);
  }

  if caps.has(platform::caps::x86::CRC32C_READY) {
    return Selected::new("x86_64/auto", crc32c_x86_64_auto);
  }

  Selected::new(kernels::PORTABLE, crc32c_portable)
}

#[cfg(target_arch = "aarch64")]
fn select_crc32() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32_portable);
  }

  if caps.has(platform::caps::aarch64::CRC_READY) {
    return Selected::new("aarch64/auto", crc32_aarch64_auto);
  }

  Selected::new(kernels::PORTABLE, crc32_portable)
}

#[cfg(target_arch = "aarch64")]
fn select_crc32c() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32c_portable);
  }

  if caps.has(platform::caps::aarch64::CRC_READY) {
    return Selected::new("aarch64/auto", crc32c_aarch64_auto);
  }

  Selected::new(kernels::PORTABLE, crc32c_portable)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn select_crc32() -> Selected<Crc32Fn> {
  Selected::new(kernels::PORTABLE, crc32_portable)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn select_crc32c() -> Selected<Crc32Fn> {
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
#[derive(Clone, Default)]
pub struct Crc32 {
  state: u32,
}

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
    Self { state: !0 }
  }

  #[inline]
  fn with_initial(initial: u32) -> Self {
    Self { state: initial ^ !0 }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.state = CRC32_DISPATCHER.call(self.state, data);
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

impl traits::ChecksumCombine for Crc32 {
  fn combine(crc_a: u32, crc_b: u32, len_b: usize) -> u32 {
    crate::common::combine::combine_crc32(crc_a, crc_b, len_b, Self::SHIFT8_MATRIX)
  }
}

/// CRC-32C (Castagnoli) checksum.
///
/// Used by iSCSI, SCTP, ext4, Btrfs, SSE4.2 `crc32`, etc.
#[derive(Clone, Default)]
pub struct Crc32C {
  state: u32,
}

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
    Self { state: !0 }
  }

  #[inline]
  fn with_initial(initial: u32) -> Self {
    Self { state: initial ^ !0 }
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.state = CRC32C_DISPATCHER.call(self.state, data);
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

impl traits::ChecksumCombine for Crc32C {
  fn combine(crc_a: u32, crc_b: u32, len_b: usize) -> u32 {
    crate::common::combine::combine_crc32(crc_a, crc_b, len_b, Self::SHIFT8_MATRIX)
  }
}

#[cfg(test)]
mod proptests;
