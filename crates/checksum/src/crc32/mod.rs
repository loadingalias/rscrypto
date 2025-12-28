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

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  extern crate std;

  use alloc::{string::String, vec::Vec};

  use super::*;

  #[test]
  fn test_crc32_empty_is_zero() {
    assert_eq!(Crc32::checksum(&[]), 0);
    assert_eq!(Crc32C::checksum(&[]), 0);
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
      let streams_env = std::env::var("RSCRYPTO_CRC32_STREAMS").ok();
      let caps = platform::caps();
      if force.eq_ignore_ascii_case("pclmul") || force.eq_ignore_ascii_case("clmul") {
        assert_eq!(cfg.requested_force, Crc32Force::Pclmul);
        if cfg.effective_force == Crc32Force::Pclmul && caps.has(platform::caps::x86::PCLMUL_READY) {
          if streams_env.is_some() {
            let expected = match x86_streams_for_len(len, cfg.tunables.streams) {
              8 => "x86_64/crc32-pclmul-8way",
              7 => "x86_64/crc32-pclmul-7way",
              4 => "x86_64/crc32-pclmul-4way",
              2 => "x86_64/crc32-pclmul-2way",
              _ => "x86_64/crc32-pclmul",
            };
            assert_eq!(kernel, expected);
          } else {
            assert!(kernel.starts_with("x86_64/crc32-pclmul"));
          }
        }
        return;
      }
      if force.eq_ignore_ascii_case("vpclmul") {
        assert_eq!(cfg.requested_force, Crc32Force::Vpclmul);
        if cfg.effective_force == Crc32Force::Vpclmul && caps.has(platform::caps::x86::VPCLMUL_READY) {
          if streams_env.is_some() {
            let expected = match x86_streams_for_len(len, cfg.tunables.streams) {
              8 => "x86_64/crc32-vpclmul-8way",
              7 => "x86_64/crc32-vpclmul-7way",
              4 => "x86_64/crc32-vpclmul-4way",
              2 => "x86_64/crc32-vpclmul-2way",
              _ => "x86_64/crc32-vpclmul",
            };
            assert_eq!(kernel, expected);
          } else {
            assert!(kernel.starts_with("x86_64/crc32-vpclmul"));
          }
        }
        return;
      }
    }

    #[cfg(target_arch = "aarch64")]
    {
      let caps = platform::caps();
      if force.eq_ignore_ascii_case("hwcrc") || force.eq_ignore_ascii_case("crc") {
        assert_eq!(cfg.requested_force, Crc32Force::Hwcrc);
        if cfg.effective_force == Crc32Force::Hwcrc && caps.has(platform::caps::aarch64::CRC_READY) {
          assert!(kernel.starts_with("aarch64/crc32"));
        }
      }
      if force.eq_ignore_ascii_case("pmull") {
        assert_eq!(cfg.requested_force, Crc32Force::Pmull);
        if cfg.effective_force == Crc32Force::Pmull && caps.has(platform::caps::aarch64::PMULL_READY) {
          assert!(kernel.starts_with("aarch64/crc32-pmull"));
        }
      }
      if force.eq_ignore_ascii_case("pmull-eor3") || force.eq_ignore_ascii_case("eor3") {
        assert_eq!(cfg.requested_force, Crc32Force::PmullEor3);
        if cfg.effective_force == Crc32Force::PmullEor3 && caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
          assert!(kernel.starts_with("aarch64/crc32-pmull-eor3"));
        }
      }
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
      let streams_env = std::env::var("RSCRYPTO_CRC32_STREAMS").ok();
      let caps = platform::caps();
      if force.eq_ignore_ascii_case("hwcrc") || force.eq_ignore_ascii_case("crc") {
        assert_eq!(cfg.requested_force, Crc32Force::Hwcrc);
        if cfg.effective_force == Crc32Force::Hwcrc && caps.has(platform::caps::x86::CRC32C_READY) {
          assert!(kernel.starts_with("x86_64/crc32c"));
        }
        return;
      }
      if force.eq_ignore_ascii_case("pclmul") || force.eq_ignore_ascii_case("clmul") {
        assert_eq!(cfg.requested_force, Crc32Force::Pclmul);
        if cfg.effective_force == Crc32Force::Pclmul
          && caps.has(platform::caps::x86::CRC32C_READY)
          && caps.has(platform::caps::x86::PCLMUL_READY)
        {
          if streams_env.is_some() {
            let expected = match x86_streams_for_len(len, cfg.tunables.streams) {
              8 => "x86_64/crc32c-fusion-v4s3x3-8way",
              7 => "x86_64/crc32c-fusion-v4s3x3-7way",
              4 => "x86_64/crc32c-fusion-v4s3x3-4way",
              2 => "x86_64/crc32c-fusion-v4s3x3-2way",
              _ => "x86_64/crc32c-fusion-v4s3x3",
            };
            assert_eq!(kernel, expected);
          } else {
            assert!(kernel.starts_with("x86_64/crc32c-fusion-v4s3x3"));
          }
        }
        return;
      }
      if force.eq_ignore_ascii_case("vpclmul") {
        assert_eq!(cfg.requested_force, Crc32Force::Vpclmul);
        if cfg.effective_force == Crc32Force::Vpclmul
          && caps.has(platform::caps::x86::CRC32C_READY)
          && caps.has(platform::caps::x86::VPCLMUL_READY)
        {
          if streams_env.is_some() {
            let expected = match x86_streams_for_len(len, cfg.tunables.streams) {
              8 => "x86_64/crc32c-fusion-vpclmul-v3x2-8way",
              7 => "x86_64/crc32c-fusion-vpclmul-v3x2-7way",
              4 => "x86_64/crc32c-fusion-vpclmul-v3x2-4way",
              2 => "x86_64/crc32c-fusion-vpclmul-v3x2-2way",
              _ => "x86_64/crc32c-fusion-vpclmul-v3x2",
            };
            assert_eq!(kernel, expected);
          } else {
            assert!(kernel.starts_with("x86_64/crc32c-fusion-vpclmul-v3x2"));
          }
        }
        return;
      }
    }

    #[cfg(target_arch = "aarch64")]
    {
      let caps = platform::caps();
      if force.eq_ignore_ascii_case("hwcrc") || force.eq_ignore_ascii_case("crc") {
        assert_eq!(cfg.requested_force, Crc32Force::Hwcrc);
        if cfg.effective_force == Crc32Force::Hwcrc && caps.has(platform::caps::aarch64::CRC_READY) {
          assert!(kernel.starts_with("aarch64/crc32c"));
        }
      }
      if force.eq_ignore_ascii_case("pmull") {
        assert_eq!(cfg.requested_force, Crc32Force::Pmull);
        if cfg.effective_force == Crc32Force::Pmull && caps.has(platform::caps::aarch64::PMULL_READY) {
          assert!(kernel.starts_with("aarch64/crc32c-pmull"));
        }
      }
      if force.eq_ignore_ascii_case("pmull-eor3") || force.eq_ignore_ascii_case("eor3") {
        assert_eq!(cfg.requested_force, Crc32Force::PmullEor3);
        if cfg.effective_force == Crc32Force::PmullEor3 && caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
          assert!(kernel.starts_with("aarch64/crc32c-pmull-eor3"));
        }
      }
    }
  }
}

#[cfg(test)]
mod proptests;
