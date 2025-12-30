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
//! - powerpc64le: VPMSUMD folding
//! - s390x: VGFM folding
//! - riscv64: ZVBC (RVV vector CLMUL) / Zbc folding
//! - wasm32/wasm64: portable only (no CLMUL)

pub(crate) mod config;
pub(crate) mod kernels;
pub(crate) mod portable;
mod tuned_defaults;

#[cfg(feature = "alloc")]
pub mod kernel_test;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

// NOTE: The VPMSUMD CRC64 backend is currently implemented for little-endian
// POWER and supports both endiannesses (big-endian loads are normalized).
#[cfg(target_arch = "powerpc64")]
mod powerpc64;

#[cfg(target_arch = "s390x")]
mod s390x;

#[cfg(target_arch = "riscv64")]
mod riscv64;

#[cfg(all(feature = "std", target_arch = "x86_64"))]
use std::sync::OnceLock;

use backend::dispatch::Selected;
// Re-export config types for public API (Crc64Force only used internally on SIMD archs)
#[allow(unused_imports)]
pub use config::{Crc64Config, Crc64Force, Crc64Tunables};
// Re-export traits for test module (`use super::*`).
#[allow(unused_imports)]
pub(super) use traits::{Checksum, ChecksumCombine};

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", test))]
use crate::common::tables::generate_crc64_tables_8;
use crate::{
  common::{
    reference::crc64_bitwise,
    tables::{CRC64_NVME_POLY, CRC64_XZ_POLY, generate_crc64_tables_16},
  },
  dispatchers::{Crc64Dispatcher, Crc64Fn},
};

// ─────────────────────────────────────────────────────────────────────────────
// Cached Dispatch Params (std-only hot path)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
struct Crc64X86Auto {
  portable_to_clmul: usize,
  pclmul_to_vpclmul: usize,
  streams: u8,
  effective_force: Crc64Force,
  has_pclmul: bool,
  has_vpclmul: bool,
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
static CRC64_X86_AUTO: OnceLock<Crc64X86Auto> = OnceLock::new();

#[inline]
#[must_use]
pub(crate) fn crc64_selected_kernel_name(len: usize) -> &'static str {
  #[cfg(target_arch = "x86_64")]
  {
    use kernels::x86_64::*;
    let cfg = config::get();
    let caps = platform::caps();
    let portable_to_clmul = cfg.tunables.portable_to_clmul.max(64);

    // Reference always uses bitwise
    if cfg.effective_force == Crc64Force::Reference {
      return kernels::REFERENCE;
    }

    if len < CRC64_SMALL_LANE_BYTES || cfg.effective_force == Crc64Force::Portable {
      return kernels::PORTABLE;
    }

    // Handle forced backend selection
    match cfg.effective_force {
      Crc64Force::Pclmul if caps.has(platform::caps::x86::PCLMUL_READY) => {
        let streams = x86_pclmul_streams_for_len(len, cfg.tunables.streams);
        return kernels::select_name(PCLMUL_NAMES, Some(PCLMUL_SMALL), streams, len, CRC64_FOLD_BLOCK_BYTES);
      }
      Crc64Force::Pclmul => return kernels::PORTABLE,
      Crc64Force::Vpclmul if caps.has(platform::caps::x86::VPCLMUL_READY) => {
        // Use 4×512 kernel only for very large buffers (higher fixed overhead).
        if len >= CRC64_4X512_MIN_BYTES {
          return VPCLMUL_4X512;
        }
        let streams = x86_vpclmul_streams_for_len(len, cfg.tunables.streams);
        return kernels::select_name(VPCLMUL_NAMES, None, streams, len, CRC64_FOLD_BLOCK_BYTES);
      }
      Crc64Force::Vpclmul if caps.has(platform::caps::x86::PCLMUL_READY) => {
        // Fall back to PCLMUL if VPCLMUL unavailable
        let streams = x86_pclmul_streams_for_len(len, cfg.tunables.streams);
        return kernels::select_name(PCLMUL_NAMES, Some(PCLMUL_SMALL), streams, len, CRC64_FOLD_BLOCK_BYTES);
      }
      Crc64Force::Vpclmul => return kernels::PORTABLE,
      _ => {}
    }

    // Auto selection with thresholds
    if len < portable_to_clmul {
      return kernels::PORTABLE;
    }

    // VPCLMUL tier (if available and above threshold)
    if caps.has(platform::caps::x86::VPCLMUL_READY) && len >= cfg.tunables.pclmul_to_vpclmul {
      // Use 4×512 kernel only for very large buffers (higher fixed overhead).
      if len >= CRC64_4X512_MIN_BYTES {
        return VPCLMUL_4X512;
      }
      let streams = x86_vpclmul_streams_for_len(len, cfg.tunables.streams);
      return kernels::select_name(VPCLMUL_NAMES, None, streams, len, CRC64_FOLD_BLOCK_BYTES);
    }

    // PCLMUL tier
    if caps.has(platform::caps::x86::PCLMUL_READY) {
      let streams = x86_pclmul_streams_for_len(len, cfg.tunables.streams);
      return kernels::select_name(PCLMUL_NAMES, Some(PCLMUL_SMALL), streams, len, CRC64_FOLD_BLOCK_BYTES);
    }

    // Fallback: VPCLMUL without PCLMUL (rare edge case)
    if caps.has(platform::caps::x86::VPCLMUL_READY) {
      // Use 4×512 kernel only for very large buffers (higher fixed overhead).
      if len >= CRC64_4X512_MIN_BYTES {
        return VPCLMUL_4X512;
      }
      let streams = x86_vpclmul_streams_for_len(len, cfg.tunables.streams);
      return kernels::select_name(VPCLMUL_NAMES, None, streams, len, CRC64_FOLD_BLOCK_BYTES);
    }

    kernels::PORTABLE
  }

  #[cfg(target_arch = "aarch64")]
  {
    use kernels::aarch64::*;
    let cfg = config::get();
    let caps = platform::caps();

    // Reference always uses bitwise
    if cfg.effective_force == Crc64Force::Reference {
      return kernels::REFERENCE;
    }

    if len < CRC64_SMALL_LANE_BYTES || cfg.effective_force == Crc64Force::Portable {
      return kernels::PORTABLE;
    }

    // Handle forced backend selection
    match cfg.effective_force {
      Crc64Force::Pmull if caps.has(platform::caps::aarch64::PMULL_READY) => {
        let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
        return kernels::select_name(PMULL_NAMES, Some(PMULL_SMALL), streams, len, CRC64_FOLD_BLOCK_BYTES);
      }
      Crc64Force::Pmull => return kernels::PORTABLE,
      Crc64Force::PmullEor3 if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) => {
        // EOR3 uses PMULL small kernel for small buffers
        let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
        return kernels::select_name(
          PMULL_EOR3_NAMES,
          Some(PMULL_SMALL),
          streams,
          len,
          CRC64_FOLD_BLOCK_BYTES,
        );
      }
      Crc64Force::PmullEor3 => return kernels::PORTABLE,
      Crc64Force::Sve2Pmull
        if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) =>
      {
        let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
        return kernels::select_name(
          SVE2_PMULL_NAMES,
          Some(SVE2_PMULL_SMALL),
          streams,
          len,
          CRC64_FOLD_BLOCK_BYTES,
        );
      }
      Crc64Force::Sve2Pmull => return kernels::PORTABLE,
      _ => {}
    }

    // Auto selection with thresholds
    if len < cfg.tunables.portable_to_clmul {
      return kernels::PORTABLE;
    }

    // EOR3 tier (highest priority, requires large buffer)
    if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) && len >= CRC64_FOLD_BLOCK_BYTES {
      let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
      return kernels::select_name(PMULL_EOR3_NAMES, None, streams, len, CRC64_FOLD_BLOCK_BYTES);
    }

    // SVE2 tier (only for multi-way, falls through for 1-way)
    if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
      let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
      if streams >= 2 {
        return kernels::select_name(SVE2_PMULL_NAMES, None, streams, len, CRC64_FOLD_BLOCK_BYTES);
      }
    }

    // PMULL tier (base tier)
    if caps.has(platform::caps::aarch64::PMULL_READY) {
      let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
      return kernels::select_name(PMULL_NAMES, Some(PMULL_SMALL), streams, len, CRC64_FOLD_BLOCK_BYTES);
    }

    kernels::PORTABLE
  }

  #[cfg(target_arch = "powerpc64")]
  {
    use kernels::powerpc64::*;
    let cfg = config::get();
    let caps = platform::caps();

    // Reference always uses bitwise
    if cfg.effective_force == Crc64Force::Reference {
      return kernels::REFERENCE;
    }

    if len < CRC64_SMALL_LANE_BYTES || cfg.effective_force == Crc64Force::Portable {
      return kernels::PORTABLE;
    }

    // Handle forced backend selection
    if cfg.effective_force == Crc64Force::Vpmsum {
      if caps.has(platform::caps::powerpc64::VPMSUM_READY) {
        let streams = powerpc64_vpmsum_streams_for_len(len, cfg.tunables.streams);
        return kernels::select_name(VPMSUM_NAMES, None, streams, len, CRC64_FOLD_BLOCK_BYTES);
      }
      return kernels::PORTABLE;
    }

    // Auto selection
    if len < cfg.tunables.portable_to_clmul {
      return kernels::PORTABLE;
    }

    if caps.has(platform::caps::powerpc64::VPMSUM_READY) {
      let streams = powerpc64_vpmsum_streams_for_len(len, cfg.tunables.streams);
      return kernels::select_name(VPMSUM_NAMES, None, streams, len, CRC64_FOLD_BLOCK_BYTES);
    }

    kernels::PORTABLE
  }

  #[cfg(target_arch = "s390x")]
  {
    use kernels::s390x::*;
    let cfg = config::get();
    let caps = platform::caps();

    // Reference always uses bitwise
    if cfg.effective_force == Crc64Force::Reference {
      return kernels::REFERENCE;
    }

    if len < CRC64_SMALL_LANE_BYTES || cfg.effective_force == Crc64Force::Portable {
      return kernels::PORTABLE;
    }

    // Handle forced backend selection
    if cfg.effective_force == Crc64Force::Vgfm {
      if caps.has(platform::caps::s390x::VECTOR) {
        let streams = s390x_vgfm_streams_for_len(len, cfg.tunables.streams);
        return kernels::select_name(VGFM_NAMES, None, streams, len, CRC64_FOLD_BLOCK_BYTES);
      }
      return kernels::PORTABLE;
    }

    // Auto selection
    if len < cfg.tunables.portable_to_clmul {
      return kernels::PORTABLE;
    }

    if caps.has(platform::caps::s390x::VECTOR) {
      let streams = s390x_vgfm_streams_for_len(len, cfg.tunables.streams);
      return kernels::select_name(VGFM_NAMES, None, streams, len, CRC64_FOLD_BLOCK_BYTES);
    }

    kernels::PORTABLE
  }

  #[cfg(target_arch = "riscv64")]
  {
    use kernels::riscv64::*;
    let cfg = config::get();
    let caps = platform::caps();

    // Reference always uses bitwise
    if cfg.effective_force == Crc64Force::Reference {
      return kernels::REFERENCE;
    }

    if len < CRC64_SMALL_LANE_BYTES || cfg.effective_force == Crc64Force::Portable {
      return kernels::PORTABLE;
    }

    // Handle forced backend selection
    match cfg.effective_force {
      Crc64Force::Zvbc if caps.has(platform::caps::riscv::ZVBC) => {
        let streams = riscv64_zbc_streams_for_len(len, cfg.tunables.streams);
        return kernels::select_name(ZVBC_NAMES, None, streams, len, CRC64_FOLD_BLOCK_BYTES);
      }
      Crc64Force::Zvbc => return kernels::PORTABLE,
      Crc64Force::Zbc if caps.has(platform::caps::riscv::ZBC) => {
        let streams = riscv64_zbc_streams_for_len(len, cfg.tunables.streams);
        return kernels::select_name(ZBC_NAMES, None, streams, len, CRC64_FOLD_BLOCK_BYTES);
      }
      Crc64Force::Zbc => return kernels::PORTABLE,
      _ => {}
    }

    // Auto selection
    if len < cfg.tunables.portable_to_clmul {
      return kernels::PORTABLE;
    }

    // ZVBC tier (higher priority)
    if caps.has(platform::caps::riscv::ZVBC) {
      let streams = riscv64_zbc_streams_for_len(len, cfg.tunables.streams);
      return kernels::select_name(ZVBC_NAMES, None, streams, len, CRC64_FOLD_BLOCK_BYTES);
    }

    // ZBC tier
    if caps.has(platform::caps::riscv::ZBC) {
      let streams = riscv64_zbc_streams_for_len(len, cfg.tunables.streams);
      return kernels::select_name(ZBC_NAMES, None, streams, len, CRC64_FOLD_BLOCK_BYTES);
    }

    kernels::PORTABLE
  }

  #[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "s390x",
    target_arch = "riscv64"
  )))]
  {
    let _ = len;
    kernels::PORTABLE
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
fn crc64_xz_portable(crc: u64, data: &[u8]) -> u64 {
  portable::crc64_slice16_xz(crc, data)
}

/// CRC-64-NVME portable kernel wrapper.
fn crc64_nvme_portable(crc: u64, data: &[u8]) -> u64 {
  portable::crc64_slice16_nvme(crc, data)
}

/// CRC-64-XZ reference (bitwise) kernel wrapper.
///
/// This is the canonical reference implementation - obviously correct,
/// audit-friendly, and used for verification of all optimized paths.
fn crc64_xz_reference(crc: u64, data: &[u8]) -> u64 {
  crc64_bitwise(CRC64_XZ_POLY, crc, data)
}

/// CRC-64-NVME reference (bitwise) kernel wrapper.
///
/// This is the canonical reference implementation - obviously correct,
/// audit-friendly, and used for verification of all optimized paths.
fn crc64_nvme_reference(crc: u64, data: &[u8]) -> u64 {
  crc64_bitwise(CRC64_NVME_POLY, crc, data)
}

// Folding parameters (only used on SIMD architectures)
//
// - Small-buffer tier folds one 16B lane at a time.
// - Large-buffer tier folds 8×16B lanes per 128B block.
#[cfg(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "powerpc64",
  target_arch = "s390x",
  target_arch = "riscv64"
))]
const CRC64_FOLD_BLOCK_BYTES: usize = 128;
#[cfg(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "powerpc64",
  target_arch = "s390x",
  target_arch = "riscv64"
))]
const CRC64_SMALL_LANE_BYTES: usize = 16;
/// Block size for 4×512-bit VPCLMUL processing (4 × 64 bytes).
#[cfg(target_arch = "x86_64")]
const CRC64_4X512_BLOCK_BYTES: usize = 256;
/// Minimum buffer length where the 4×512-bit VPCLMUL kernel is worthwhile.
#[cfg(target_arch = "x86_64")]
const CRC64_4X512_MIN_BYTES: usize = CRC64_4X512_BLOCK_BYTES.strict_mul(1024);

#[cfg(target_arch = "x86_64")]
#[inline]
const fn x86_pclmul_streams_for_len(len: usize, streams: u8) -> u8 {
  // 8-way is the default (Intel white paper / Linux kernel standard)
  if streams >= 8 && len >= 8 * 2 * CRC64_FOLD_BLOCK_BYTES {
    return 8;
  }
  if streams >= 7 && len >= 7 * 2 * CRC64_FOLD_BLOCK_BYTES {
    return 7;
  }
  if streams >= 4 && len >= 4 * 2 * CRC64_FOLD_BLOCK_BYTES {
    return 4;
  }
  if streams >= 2 && len >= 2 * 2 * CRC64_FOLD_BLOCK_BYTES {
    return 2;
  }
  1
}

#[cfg(target_arch = "x86_64")]
#[inline]
const fn x86_vpclmul_streams_for_len(len: usize, streams: u8) -> u8 {
  // 8-way is the default (Intel white paper / Linux kernel standard)
  if streams >= 8 && len >= 8 * 2 * CRC64_FOLD_BLOCK_BYTES {
    return 8;
  }
  if streams >= 7 && len >= 7 * 2 * CRC64_FOLD_BLOCK_BYTES {
    return 7;
  }
  if streams >= 4 && len >= 4 * 2 * CRC64_FOLD_BLOCK_BYTES {
    return 4;
  }
  if streams >= 2 && len >= 2 * 2 * CRC64_FOLD_BLOCK_BYTES {
    return 2;
  }
  1
}

#[cfg(target_arch = "aarch64")]
const CRC64_PMULL_2WAY_MIN_BYTES: usize = 128 * CRC64_FOLD_BLOCK_BYTES; // 16 KiB

#[cfg(target_arch = "aarch64")]
#[inline]
const fn aarch64_pmull_streams_for_len(len: usize, streams: u8) -> u8 {
  // 3-way is intentionally conservative: it increases register pressure and
  // tends to win only in the large-buffer regime.
  const CRC64_PMULL_3WAY_MIN_BYTES: usize = 256 * CRC64_FOLD_BLOCK_BYTES; // 32 KiB

  if streams >= 3 && len >= CRC64_PMULL_3WAY_MIN_BYTES {
    return 3;
  }
  if streams >= 2 && len >= CRC64_PMULL_2WAY_MIN_BYTES {
    return 2;
  }
  1
}

#[cfg(target_arch = "powerpc64")]
const CRC64_VPMSUM_2WAY_MIN_BYTES: usize = 128 * CRC64_FOLD_BLOCK_BYTES; // 16 KiB

#[cfg(target_arch = "powerpc64")]
#[inline]
const fn powerpc64_vpmsum_streams_for_len(len: usize, streams: u8) -> u8 {
  // POWER benefits from higher ILP, but multi-stream merging has non-trivial
  // setup costs. Keep thresholds conservative by default; users can override
  // via `RSCRYPTO_CRC64_STREAMS` and tune results.
  const CRC64_VPMSUM_4WAY_MIN_BYTES: usize = 2 * CRC64_VPMSUM_2WAY_MIN_BYTES; // 32 KiB
  const CRC64_VPMSUM_8WAY_MIN_BYTES: usize = 4 * CRC64_VPMSUM_2WAY_MIN_BYTES; // 64 KiB

  if streams >= 8 && len >= CRC64_VPMSUM_8WAY_MIN_BYTES {
    return 8;
  }
  if streams >= 4 && len >= CRC64_VPMSUM_4WAY_MIN_BYTES {
    return 4;
  }
  if streams >= 2 && len >= CRC64_VPMSUM_2WAY_MIN_BYTES {
    return 2;
  }
  1
}

#[cfg(target_arch = "s390x")]
const CRC64_VGFM_2WAY_MIN_BYTES: usize = 128 * CRC64_FOLD_BLOCK_BYTES; // 16 KiB

#[cfg(target_arch = "s390x")]
#[inline]
const fn s390x_vgfm_streams_for_len(len: usize, streams: u8) -> u8 {
  // z13+ has 32 vector registers and strong ILP; keep thresholds conservative.
  const CRC64_VGFM_4WAY_MIN_BYTES: usize = 2 * CRC64_VGFM_2WAY_MIN_BYTES; // 32 KiB

  if streams >= 4 && len >= CRC64_VGFM_4WAY_MIN_BYTES {
    return 4;
  }
  if streams >= 2 && len >= CRC64_VGFM_2WAY_MIN_BYTES {
    return 2;
  }
  1
}

#[cfg(target_arch = "riscv64")]
const CRC64_ZBC_2WAY_MIN_BYTES: usize = 128 * CRC64_FOLD_BLOCK_BYTES; // 16 KiB

#[cfg(target_arch = "riscv64")]
#[inline]
const fn riscv64_zbc_streams_for_len(len: usize, streams: u8) -> u8 {
  const CRC64_ZBC_4WAY_MIN_BYTES: usize = 2 * CRC64_ZBC_2WAY_MIN_BYTES; // 32 KiB

  if streams >= 4 && len >= CRC64_ZBC_4WAY_MIN_BYTES {
    return 4;
  }
  if streams >= 2 && len >= CRC64_ZBC_2WAY_MIN_BYTES {
    return 2;
  }
  1
}

// Buffered CRC uses this to decide when to flush accumulated small updates.
// On platforms without a SIMD backend, this effectively becomes `usize::MAX`
// (no early flush beyond the fixed buffer size).
#[allow(dead_code)]
#[inline]
fn crc64_simd_threshold() -> usize {
  config::get().tunables.portable_to_clmul.max(64)
}

#[cfg(target_arch = "x86_64")]
fn crc64_xz_x86_64_auto(crc: u64, data: &[u8]) -> u64 {
  use kernels::x86_64::*;
  let len = data.len();

  // Avoid any dispatch overhead for the tiny regime.
  if len < CRC64_SMALL_LANE_BYTES {
    return crc64_xz_portable(crc, data);
  }

  #[cfg(feature = "std")]
  let params = *CRC64_X86_AUTO.get_or_init(|| {
    let cfg = config::get();
    let caps = platform::caps();
    Crc64X86Auto {
      // Keep this conservative: several x86 µarches lose below ~64B due to setup costs.
      portable_to_clmul: cfg.tunables.portable_to_clmul.max(64),
      pclmul_to_vpclmul: cfg.tunables.pclmul_to_vpclmul,
      streams: cfg.tunables.streams,
      effective_force: cfg.effective_force,
      has_pclmul: caps.has(platform::caps::x86::PCLMUL_READY),
      has_vpclmul: caps.has(platform::caps::x86::VPCLMUL_READY),
    }
  });

  #[cfg(not(feature = "std"))]
  let params = {
    let cfg = config::get();
    let caps = platform::caps();
    Crc64X86Auto {
      portable_to_clmul: cfg.tunables.portable_to_clmul.max(64),
      pclmul_to_vpclmul: cfg.tunables.pclmul_to_vpclmul,
      streams: cfg.tunables.streams,
      effective_force: cfg.effective_force,
      has_pclmul: caps.has(platform::caps::x86::PCLMUL_READY),
      has_vpclmul: caps.has(platform::caps::x86::VPCLMUL_READY),
    }
  };

  // Handle forced backend selection
  match params.effective_force {
    Crc64Force::Reference => {
      return crc64_xz_reference(crc, data);
    }
    Crc64Force::Portable => {
      return crc64_xz_portable(crc, data);
    }
    Crc64Force::Pclmul if params.has_pclmul => {
      let streams = x86_pclmul_streams_for_len(len, params.streams);
      return kernels::dispatch_with_small(
        &XZ_PCLMUL,
        XZ_PCLMUL_SMALL,
        streams,
        len,
        CRC64_FOLD_BLOCK_BYTES,
        crc,
        data,
      );
    }
    Crc64Force::Pclmul => {
      return crc64_xz_portable(crc, data);
    }
    Crc64Force::Vpclmul if params.has_vpclmul => {
      // Use 4×512 kernel only for very large buffers (higher fixed overhead).
      if len >= CRC64_4X512_MIN_BYTES {
        return XZ_VPCLMUL_4X512(crc, data);
      }
      let streams = x86_vpclmul_streams_for_len(len, params.streams);
      return kernels::dispatch_streams(&XZ_VPCLMUL, streams, crc, data);
    }
    Crc64Force::Vpclmul if params.has_pclmul => {
      // Fall back to PCLMUL if VPCLMUL unavailable
      let streams = x86_pclmul_streams_for_len(len, params.streams);
      return kernels::dispatch_with_small(
        &XZ_PCLMUL,
        XZ_PCLMUL_SMALL,
        streams,
        len,
        CRC64_FOLD_BLOCK_BYTES,
        crc,
        data,
      );
    }
    Crc64Force::Vpclmul => {
      return crc64_xz_portable(crc, data);
    }
    _ => {}
  }

  // Auto selection
  if len < params.portable_to_clmul {
    return crc64_xz_portable(crc, data);
  }

  // VPCLMUL tier (if available and above threshold)
  if params.has_vpclmul && len >= params.pclmul_to_vpclmul {
    // Use 4×512 kernel only for very large buffers (higher fixed overhead).
    if len >= CRC64_4X512_MIN_BYTES {
      return XZ_VPCLMUL_4X512(crc, data);
    }
    let streams = x86_vpclmul_streams_for_len(len, params.streams);
    return kernels::dispatch_streams(&XZ_VPCLMUL, streams, crc, data);
  }

  // PCLMUL tier
  if params.has_pclmul {
    let streams = x86_pclmul_streams_for_len(len, params.streams);
    return kernels::dispatch_with_small(
      &XZ_PCLMUL,
      XZ_PCLMUL_SMALL,
      streams,
      len,
      CRC64_FOLD_BLOCK_BYTES,
      crc,
      data,
    );
  }

  // Fallback: VPCLMUL without PCLMUL (rare edge case)
  if params.has_vpclmul {
    // Use 4×512 kernel only for very large buffers (higher fixed overhead).
    if len >= CRC64_4X512_MIN_BYTES {
      return XZ_VPCLMUL_4X512(crc, data);
    }
    let streams = x86_vpclmul_streams_for_len(len, params.streams);
    return kernels::dispatch_streams(&XZ_VPCLMUL, streams, crc, data);
  }

  crc64_xz_portable(crc, data)
}

#[cfg(target_arch = "x86_64")]
fn crc64_nvme_x86_64_auto(crc: u64, data: &[u8]) -> u64 {
  use kernels::x86_64::*;
  let len = data.len();

  if len < CRC64_SMALL_LANE_BYTES {
    return crc64_nvme_portable(crc, data);
  }

  #[cfg(feature = "std")]
  let params = *CRC64_X86_AUTO.get_or_init(|| {
    let cfg = config::get();
    let caps = platform::caps();
    Crc64X86Auto {
      portable_to_clmul: cfg.tunables.portable_to_clmul.max(64),
      pclmul_to_vpclmul: cfg.tunables.pclmul_to_vpclmul,
      streams: cfg.tunables.streams,
      effective_force: cfg.effective_force,
      has_pclmul: caps.has(platform::caps::x86::PCLMUL_READY),
      has_vpclmul: caps.has(platform::caps::x86::VPCLMUL_READY),
    }
  });

  #[cfg(not(feature = "std"))]
  let params = {
    let cfg = config::get();
    let caps = platform::caps();
    Crc64X86Auto {
      portable_to_clmul: cfg.tunables.portable_to_clmul.max(64),
      pclmul_to_vpclmul: cfg.tunables.pclmul_to_vpclmul,
      streams: cfg.tunables.streams,
      effective_force: cfg.effective_force,
      has_pclmul: caps.has(platform::caps::x86::PCLMUL_READY),
      has_vpclmul: caps.has(platform::caps::x86::VPCLMUL_READY),
    }
  };

  // Handle forced backend selection
  match params.effective_force {
    Crc64Force::Reference => {
      return crc64_nvme_reference(crc, data);
    }
    Crc64Force::Portable => {
      return crc64_nvme_portable(crc, data);
    }
    Crc64Force::Pclmul if params.has_pclmul => {
      let streams = x86_pclmul_streams_for_len(len, params.streams);
      return kernels::dispatch_with_small(
        &NVME_PCLMUL,
        NVME_PCLMUL_SMALL,
        streams,
        len,
        CRC64_FOLD_BLOCK_BYTES,
        crc,
        data,
      );
    }
    Crc64Force::Pclmul => {
      return crc64_nvme_portable(crc, data);
    }
    Crc64Force::Vpclmul if params.has_vpclmul => {
      // Use 4×512 kernel only for very large buffers (higher fixed overhead).
      if len >= CRC64_4X512_MIN_BYTES {
        return NVME_VPCLMUL_4X512(crc, data);
      }
      let streams = x86_vpclmul_streams_for_len(len, params.streams);
      return kernels::dispatch_streams(&NVME_VPCLMUL, streams, crc, data);
    }
    Crc64Force::Vpclmul if params.has_pclmul => {
      // Fall back to PCLMUL if VPCLMUL unavailable
      let streams = x86_pclmul_streams_for_len(len, params.streams);
      return kernels::dispatch_with_small(
        &NVME_PCLMUL,
        NVME_PCLMUL_SMALL,
        streams,
        len,
        CRC64_FOLD_BLOCK_BYTES,
        crc,
        data,
      );
    }
    Crc64Force::Vpclmul => {
      return crc64_nvme_portable(crc, data);
    }
    _ => {}
  }

  // Auto selection
  if len < params.portable_to_clmul {
    return crc64_nvme_portable(crc, data);
  }

  // VPCLMUL tier (if available and above threshold)
  if params.has_vpclmul && len >= params.pclmul_to_vpclmul {
    // Use 4×512 kernel only for very large buffers (higher fixed overhead).
    if len >= CRC64_4X512_MIN_BYTES {
      return NVME_VPCLMUL_4X512(crc, data);
    }
    let streams = x86_vpclmul_streams_for_len(len, params.streams);
    return kernels::dispatch_streams(&NVME_VPCLMUL, streams, crc, data);
  }

  // PCLMUL tier
  if params.has_pclmul {
    let streams = x86_pclmul_streams_for_len(len, params.streams);
    return kernels::dispatch_with_small(
      &NVME_PCLMUL,
      NVME_PCLMUL_SMALL,
      streams,
      len,
      CRC64_FOLD_BLOCK_BYTES,
      crc,
      data,
    );
  }

  // Fallback: VPCLMUL without PCLMUL (rare edge case)
  if params.has_vpclmul {
    // Use 4×512 kernel only for very large buffers (higher fixed overhead).
    if len >= CRC64_4X512_MIN_BYTES {
      return NVME_VPCLMUL_4X512(crc, data);
    }
    let streams = x86_vpclmul_streams_for_len(len, params.streams);
    return kernels::dispatch_streams(&NVME_VPCLMUL, streams, crc, data);
  }

  crc64_nvme_portable(crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Selection
// ─────────────────────────────────────────────────────────────────────────────

/// Select the best CRC-64-XZ kernel for the current platform.
#[cfg(target_arch = "x86_64")]
fn select_crc64_xz() -> Selected<Crc64Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  // Explicit reference override always wins.
  if cfg.effective_force == Crc64Force::Reference {
    return Selected::new("reference/bitwise", crc64_xz_reference);
  }

  // Explicit portable override.
  if cfg.effective_force == Crc64Force::Portable {
    return Selected::new("portable/slice16", crc64_xz_portable);
  }

  if caps.has(platform::caps::x86::PCLMUL_READY) || caps.has(platform::caps::x86::VPCLMUL_READY) {
    return Selected::new("x86_64/auto", crc64_xz_x86_64_auto);
  }

  Selected::new("portable/slice16", crc64_xz_portable)
}

#[cfg(target_arch = "powerpc64")]
fn crc64_xz_powerpc64_auto(crc: u64, data: &[u8]) -> u64 {
  use kernels::powerpc64::*;
  let len = data.len();

  if len < CRC64_SMALL_LANE_BYTES {
    return crc64_xz_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  // Handle forced backend selection
  match cfg.effective_force {
    Crc64Force::Reference => return crc64_xz_reference(crc, data),
    Crc64Force::Portable => return crc64_xz_portable(crc, data),
    Crc64Force::Vpmsum if caps.has(platform::caps::powerpc64::VPMSUM_READY) => {
      let streams = powerpc64_vpmsum_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_streams(&XZ_VPMSUM, streams, crc, data);
    }
    Crc64Force::Vpmsum => return crc64_xz_portable(crc, data),
    _ => {}
  }

  // Auto selection
  if len < cfg.tunables.portable_to_clmul {
    return crc64_xz_portable(crc, data);
  }

  if caps.has(platform::caps::powerpc64::VPMSUM_READY) {
    let streams = powerpc64_vpmsum_streams_for_len(len, cfg.tunables.streams);
    return kernels::dispatch_streams(&XZ_VPMSUM, streams, crc, data);
  }

  crc64_xz_portable(crc, data)
}

#[cfg(target_arch = "powerpc64")]
fn crc64_nvme_powerpc64_auto(crc: u64, data: &[u8]) -> u64 {
  use kernels::powerpc64::*;
  let len = data.len();

  if len < CRC64_SMALL_LANE_BYTES {
    return crc64_nvme_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  // Handle forced backend selection
  match cfg.effective_force {
    Crc64Force::Reference => return crc64_nvme_reference(crc, data),
    Crc64Force::Portable => return crc64_nvme_portable(crc, data),
    Crc64Force::Vpmsum if caps.has(platform::caps::powerpc64::VPMSUM_READY) => {
      let streams = powerpc64_vpmsum_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_streams(&NVME_VPMSUM, streams, crc, data);
    }
    Crc64Force::Vpmsum => return crc64_nvme_portable(crc, data),
    _ => {}
  }

  // Auto selection
  if len < cfg.tunables.portable_to_clmul {
    return crc64_nvme_portable(crc, data);
  }

  if caps.has(platform::caps::powerpc64::VPMSUM_READY) {
    let streams = powerpc64_vpmsum_streams_for_len(len, cfg.tunables.streams);
    return kernels::dispatch_streams(&NVME_VPMSUM, streams, crc, data);
  }

  crc64_nvme_portable(crc, data)
}

#[cfg(target_arch = "s390x")]
fn crc64_xz_s390x_auto(crc: u64, data: &[u8]) -> u64 {
  use kernels::s390x::*;
  let len = data.len();

  if len < CRC64_SMALL_LANE_BYTES {
    return crc64_xz_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  // Handle forced backend selection
  match cfg.effective_force {
    Crc64Force::Reference => return crc64_xz_reference(crc, data),
    Crc64Force::Portable => return crc64_xz_portable(crc, data),
    Crc64Force::Vgfm if caps.has(platform::caps::s390x::VECTOR) => {
      let streams = s390x_vgfm_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_streams(&XZ_VGFM, streams, crc, data);
    }
    Crc64Force::Vgfm => return crc64_xz_portable(crc, data),
    _ => {}
  }

  // Auto selection
  if len < cfg.tunables.portable_to_clmul {
    return crc64_xz_portable(crc, data);
  }

  if caps.has(platform::caps::s390x::VECTOR) {
    let streams = s390x_vgfm_streams_for_len(len, cfg.tunables.streams);
    return kernels::dispatch_streams(&XZ_VGFM, streams, crc, data);
  }

  crc64_xz_portable(crc, data)
}

#[cfg(target_arch = "s390x")]
fn crc64_nvme_s390x_auto(crc: u64, data: &[u8]) -> u64 {
  use kernels::s390x::*;
  let len = data.len();

  if len < CRC64_SMALL_LANE_BYTES {
    return crc64_nvme_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  // Handle forced backend selection
  match cfg.effective_force {
    Crc64Force::Reference => return crc64_nvme_reference(crc, data),
    Crc64Force::Portable => return crc64_nvme_portable(crc, data),
    Crc64Force::Vgfm if caps.has(platform::caps::s390x::VECTOR) => {
      let streams = s390x_vgfm_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_streams(&NVME_VGFM, streams, crc, data);
    }
    Crc64Force::Vgfm => return crc64_nvme_portable(crc, data),
    _ => {}
  }

  // Auto selection
  if len < cfg.tunables.portable_to_clmul {
    return crc64_nvme_portable(crc, data);
  }

  if caps.has(platform::caps::s390x::VECTOR) {
    let streams = s390x_vgfm_streams_for_len(len, cfg.tunables.streams);
    return kernels::dispatch_streams(&NVME_VGFM, streams, crc, data);
  }

  crc64_nvme_portable(crc, data)
}

#[cfg(target_arch = "riscv64")]
fn crc64_xz_riscv64_auto(crc: u64, data: &[u8]) -> u64 {
  use kernels::riscv64::*;
  let len = data.len();

  if len < CRC64_SMALL_LANE_BYTES {
    return crc64_xz_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  // Handle forced backend selection
  match cfg.effective_force {
    Crc64Force::Reference => return crc64_xz_reference(crc, data),
    Crc64Force::Portable => return crc64_xz_portable(crc, data),
    Crc64Force::Zvbc if caps.has(platform::caps::riscv::ZVBC) => {
      let streams = riscv64_zbc_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_streams(&XZ_ZVBC, streams, crc, data);
    }
    Crc64Force::Zvbc => return crc64_xz_portable(crc, data),
    Crc64Force::Zbc if caps.has(platform::caps::riscv::ZBC) => {
      let streams = riscv64_zbc_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_streams(&XZ_ZBC, streams, crc, data);
    }
    Crc64Force::Zbc => return crc64_xz_portable(crc, data),
    _ => {}
  }

  // Auto selection
  if len < cfg.tunables.portable_to_clmul {
    return crc64_xz_portable(crc, data);
  }

  // ZVBC tier (higher priority)
  if caps.has(platform::caps::riscv::ZVBC) {
    let streams = riscv64_zbc_streams_for_len(len, cfg.tunables.streams);
    return kernels::dispatch_streams(&XZ_ZVBC, streams, crc, data);
  }

  // ZBC tier
  if caps.has(platform::caps::riscv::ZBC) {
    let streams = riscv64_zbc_streams_for_len(len, cfg.tunables.streams);
    return kernels::dispatch_streams(&XZ_ZBC, streams, crc, data);
  }

  crc64_xz_portable(crc, data)
}

#[cfg(target_arch = "riscv64")]
fn crc64_nvme_riscv64_auto(crc: u64, data: &[u8]) -> u64 {
  use kernels::riscv64::*;
  let len = data.len();

  if len < CRC64_SMALL_LANE_BYTES {
    return crc64_nvme_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  // Handle forced backend selection
  match cfg.effective_force {
    Crc64Force::Reference => return crc64_nvme_reference(crc, data),
    Crc64Force::Portable => return crc64_nvme_portable(crc, data),
    Crc64Force::Zvbc if caps.has(platform::caps::riscv::ZVBC) => {
      let streams = riscv64_zbc_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_streams(&NVME_ZVBC, streams, crc, data);
    }
    Crc64Force::Zvbc => return crc64_nvme_portable(crc, data),
    Crc64Force::Zbc if caps.has(platform::caps::riscv::ZBC) => {
      let streams = riscv64_zbc_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_streams(&NVME_ZBC, streams, crc, data);
    }
    Crc64Force::Zbc => return crc64_nvme_portable(crc, data),
    _ => {}
  }

  // Auto selection
  if len < cfg.tunables.portable_to_clmul {
    return crc64_nvme_portable(crc, data);
  }

  // ZVBC tier (higher priority)
  if caps.has(platform::caps::riscv::ZVBC) {
    let streams = riscv64_zbc_streams_for_len(len, cfg.tunables.streams);
    return kernels::dispatch_streams(&NVME_ZVBC, streams, crc, data);
  }

  // ZBC tier
  if caps.has(platform::caps::riscv::ZBC) {
    let streams = riscv64_zbc_streams_for_len(len, cfg.tunables.streams);
    return kernels::dispatch_streams(&NVME_ZBC, streams, crc, data);
  }

  crc64_nvme_portable(crc, data)
}

#[cfg(target_arch = "aarch64")]
fn crc64_xz_aarch64_auto(crc: u64, data: &[u8]) -> u64 {
  use kernels::aarch64::*;
  let len = data.len();

  if len < CRC64_SMALL_LANE_BYTES {
    return crc64_xz_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  // Handle forced backend selection
  match cfg.effective_force {
    Crc64Force::Reference => return crc64_xz_reference(crc, data),
    Crc64Force::Portable => return crc64_xz_portable(crc, data),
    Crc64Force::PmullEor3 if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) => {
      // EOR3 uses PMULL small kernel for small buffers
      let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_with_small(
        &XZ_PMULL_EOR3,
        XZ_PMULL_SMALL,
        streams,
        len,
        CRC64_FOLD_BLOCK_BYTES,
        crc,
        data,
      );
    }
    Crc64Force::PmullEor3 => return crc64_xz_portable(crc, data),
    Crc64Force::Sve2Pmull
      if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) =>
    {
      let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_with_small(
        &XZ_SVE2_PMULL,
        XZ_SVE2_PMULL_SMALL,
        streams,
        len,
        CRC64_FOLD_BLOCK_BYTES,
        crc,
        data,
      );
    }
    Crc64Force::Sve2Pmull => return crc64_xz_portable(crc, data),
    Crc64Force::Pmull if caps.has(platform::caps::aarch64::PMULL_READY) => {
      let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_with_small(
        &XZ_PMULL,
        XZ_PMULL_SMALL,
        streams,
        len,
        CRC64_FOLD_BLOCK_BYTES,
        crc,
        data,
      );
    }
    Crc64Force::Pmull => return crc64_xz_portable(crc, data),
    _ => {}
  }

  // Auto selection
  if len < cfg.tunables.portable_to_clmul {
    return crc64_xz_portable(crc, data);
  }

  // EOR3 tier (highest priority, requires large buffer)
  if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) && len >= CRC64_FOLD_BLOCK_BYTES {
    let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
    return kernels::dispatch_streams(&XZ_PMULL_EOR3, streams, crc, data);
  }

  // SVE2 tier (only for multi-way, falls through for 1-way)
  if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
    let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
    if streams >= 2 {
      return kernels::dispatch_streams(&XZ_SVE2_PMULL, streams, crc, data);
    }
  }

  // PMULL tier (base tier)
  if caps.has(platform::caps::aarch64::PMULL_READY) {
    let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
    return kernels::dispatch_with_small(
      &XZ_PMULL,
      XZ_PMULL_SMALL,
      streams,
      len,
      CRC64_FOLD_BLOCK_BYTES,
      crc,
      data,
    );
  }

  crc64_xz_portable(crc, data)
}

#[cfg(target_arch = "aarch64")]
fn crc64_nvme_aarch64_auto(crc: u64, data: &[u8]) -> u64 {
  use kernels::aarch64::*;
  let len = data.len();

  if len < CRC64_SMALL_LANE_BYTES {
    return crc64_nvme_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  // Handle forced backend selection
  match cfg.effective_force {
    Crc64Force::Reference => return crc64_nvme_reference(crc, data),
    Crc64Force::Portable => return crc64_nvme_portable(crc, data),
    Crc64Force::PmullEor3 if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) => {
      // EOR3 uses PMULL small kernel for small buffers
      let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_with_small(
        &NVME_PMULL_EOR3,
        NVME_PMULL_SMALL,
        streams,
        len,
        CRC64_FOLD_BLOCK_BYTES,
        crc,
        data,
      );
    }
    Crc64Force::PmullEor3 => return crc64_nvme_portable(crc, data),
    Crc64Force::Sve2Pmull
      if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) =>
    {
      let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_with_small(
        &NVME_SVE2_PMULL,
        NVME_SVE2_PMULL_SMALL,
        streams,
        len,
        CRC64_FOLD_BLOCK_BYTES,
        crc,
        data,
      );
    }
    Crc64Force::Sve2Pmull => return crc64_nvme_portable(crc, data),
    Crc64Force::Pmull if caps.has(platform::caps::aarch64::PMULL_READY) => {
      let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_with_small(
        &NVME_PMULL,
        NVME_PMULL_SMALL,
        streams,
        len,
        CRC64_FOLD_BLOCK_BYTES,
        crc,
        data,
      );
    }
    Crc64Force::Pmull => return crc64_nvme_portable(crc, data),
    _ => {}
  }

  // Auto selection
  if len < cfg.tunables.portable_to_clmul {
    return crc64_nvme_portable(crc, data);
  }

  // EOR3 tier (highest priority, requires large buffer)
  if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) && len >= CRC64_FOLD_BLOCK_BYTES {
    let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
    return kernels::dispatch_streams(&NVME_PMULL_EOR3, streams, crc, data);
  }

  // SVE2 tier (only for multi-way, falls through for 1-way)
  if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
    let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
    if streams >= 2 {
      return kernels::dispatch_streams(&NVME_SVE2_PMULL, streams, crc, data);
    }
  }

  // PMULL tier (base tier)
  if caps.has(platform::caps::aarch64::PMULL_READY) {
    let streams = aarch64_pmull_streams_for_len(len, cfg.tunables.streams);
    return kernels::dispatch_with_small(
      &NVME_PMULL,
      NVME_PMULL_SMALL,
      streams,
      len,
      CRC64_FOLD_BLOCK_BYTES,
      crc,
      data,
    );
  }

  crc64_nvme_portable(crc, data)
}

#[cfg(target_arch = "aarch64")]
fn select_crc64_xz() -> Selected<Crc64Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc64Force::Reference {
    return Selected::new("reference/bitwise", crc64_xz_reference);
  }

  if cfg.effective_force == Crc64Force::Portable {
    return Selected::new("portable/slice16", crc64_xz_portable);
  }

  if caps.has(platform::caps::aarch64::PMULL_READY) {
    return Selected::new("aarch64/auto", crc64_xz_aarch64_auto);
  }

  Selected::new("portable/slice16", crc64_xz_portable)
}

#[cfg(target_arch = "powerpc64")]
fn select_crc64_xz() -> Selected<Crc64Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc64Force::Reference {
    return Selected::new("reference/bitwise", crc64_xz_reference);
  }

  if cfg.effective_force == Crc64Force::Portable {
    return Selected::new("portable/slice16", crc64_xz_portable);
  }

  if caps.has(platform::caps::powerpc64::VPMSUM_READY) {
    return Selected::new("powerpc64/auto", crc64_xz_powerpc64_auto);
  }

  Selected::new("portable/slice16", crc64_xz_portable)
}

#[cfg(target_arch = "s390x")]
fn select_crc64_xz() -> Selected<Crc64Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc64Force::Reference {
    return Selected::new("reference/bitwise", crc64_xz_reference);
  }

  if cfg.effective_force == Crc64Force::Portable {
    return Selected::new("portable/slice16", crc64_xz_portable);
  }

  if caps.has(platform::caps::s390x::VECTOR) {
    return Selected::new("s390x/auto", crc64_xz_s390x_auto);
  }

  Selected::new("portable/slice16", crc64_xz_portable)
}

#[cfg(target_arch = "riscv64")]
fn select_crc64_xz() -> Selected<Crc64Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc64Force::Reference {
    return Selected::new("reference/bitwise", crc64_xz_reference);
  }

  if cfg.effective_force == Crc64Force::Portable {
    return Selected::new("portable/slice16", crc64_xz_portable);
  }

  if caps.has(platform::caps::riscv::ZVBC) || caps.has(platform::caps::riscv::ZBC) {
    return Selected::new("riscv64/auto", crc64_xz_riscv64_auto);
  }

  Selected::new("portable/slice16", crc64_xz_portable)
}

#[cfg(not(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "powerpc64",
  target_arch = "s390x",
  target_arch = "riscv64"
)))]
fn select_crc64_xz() -> Selected<Crc64Fn> {
  if config::get().effective_force == Crc64Force::Reference {
    return Selected::new(kernels::REFERENCE, crc64_xz_reference);
  }
  Selected::new(kernels::PORTABLE, crc64_xz_portable)
}

/// Select the best CRC-64-NVME kernel for the current platform.
#[cfg(target_arch = "x86_64")]
fn select_crc64_nvme() -> Selected<Crc64Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc64Force::Reference {
    return Selected::new("reference/bitwise", crc64_nvme_reference);
  }

  if cfg.effective_force == Crc64Force::Portable {
    return Selected::new("portable/slice16", crc64_nvme_portable);
  }

  if caps.has(platform::caps::x86::PCLMUL_READY) || caps.has(platform::caps::x86::VPCLMUL_READY) {
    return Selected::new("x86_64/auto", crc64_nvme_x86_64_auto);
  }

  Selected::new("portable/slice16", crc64_nvme_portable)
}

#[cfg(target_arch = "aarch64")]
fn select_crc64_nvme() -> Selected<Crc64Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc64Force::Reference {
    return Selected::new("reference/bitwise", crc64_nvme_reference);
  }

  if cfg.effective_force == Crc64Force::Portable {
    return Selected::new("portable/slice16", crc64_nvme_portable);
  }

  if caps.has(platform::caps::aarch64::PMULL_READY) {
    return Selected::new("aarch64/auto", crc64_nvme_aarch64_auto);
  }

  Selected::new("portable/slice16", crc64_nvme_portable)
}

#[cfg(target_arch = "powerpc64")]
fn select_crc64_nvme() -> Selected<Crc64Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc64Force::Reference {
    return Selected::new("reference/bitwise", crc64_nvme_reference);
  }

  if cfg.effective_force == Crc64Force::Portable {
    return Selected::new("portable/slice16", crc64_nvme_portable);
  }

  if caps.has(platform::caps::powerpc64::VPMSUM_READY) {
    return Selected::new("powerpc64/auto", crc64_nvme_powerpc64_auto);
  }

  Selected::new("portable/slice16", crc64_nvme_portable)
}

#[cfg(target_arch = "s390x")]
fn select_crc64_nvme() -> Selected<Crc64Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc64Force::Reference {
    return Selected::new("reference/bitwise", crc64_nvme_reference);
  }

  if cfg.effective_force == Crc64Force::Portable {
    return Selected::new("portable/slice16", crc64_nvme_portable);
  }

  if caps.has(platform::caps::s390x::VECTOR) {
    return Selected::new("s390x/auto", crc64_nvme_s390x_auto);
  }

  Selected::new("portable/slice16", crc64_nvme_portable)
}

#[cfg(target_arch = "riscv64")]
fn select_crc64_nvme() -> Selected<Crc64Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc64Force::Reference {
    return Selected::new("reference/bitwise", crc64_nvme_reference);
  }

  if cfg.effective_force == Crc64Force::Portable {
    return Selected::new("portable/slice16", crc64_nvme_portable);
  }

  if caps.has(platform::caps::riscv::ZVBC) || caps.has(platform::caps::riscv::ZBC) {
    return Selected::new("riscv64/auto", crc64_nvme_riscv64_auto);
  }

  Selected::new("portable/slice16", crc64_nvme_portable)
}

#[cfg(not(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "powerpc64",
  target_arch = "s390x",
  target_arch = "riscv64"
)))]
fn select_crc64_nvme() -> Selected<Crc64Fn> {
  if config::get().effective_force == Crc64Force::Reference {
    return Selected::new(kernels::REFERENCE, crc64_nvme_reference);
  }
  Selected::new(kernels::PORTABLE, crc64_nvme_portable)
}

/// Static dispatcher for CRC-64-XZ.
static CRC64_XZ_DISPATCHER: Crc64Dispatcher = Crc64Dispatcher::new(select_crc64_xz);

/// Static dispatcher for CRC-64-NVME.
static CRC64_NVME_DISPATCHER: Crc64Dispatcher = Crc64Dispatcher::new(select_crc64_nvme);

// ─────────────────────────────────────────────────────────────────────────────
// CRC-64 Types (generated via macro)
// ─────────────────────────────────────────────────────────────────────────────

define_crc64_type! {
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
  /// # Example
  ///
  /// ```rust
  /// use checksum::{Crc64Xz, Checksum};
  ///
  /// let crc = Crc64Xz::checksum(b"123456789");
  /// assert_eq!(crc, 0x995DC9BBDF1939FA); // "123456789" test vector
  /// ```
  pub struct Crc64 {
    poly: CRC64_XZ_POLY,
    dispatcher: CRC64_XZ_DISPATCHER,
  }
}

/// Explicit name for the XZ CRC-64 variant (alias of [`Crc64`]).
pub type Crc64Xz = Crc64;

define_crc64_type! {
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
  /// # Example
  ///
  /// ```rust
  /// use checksum::{Crc64Nvme, Checksum};
  ///
  /// let crc = Crc64Nvme::checksum(b"123456789");
  /// assert_eq!(crc, 0xAE8B14860A799888); // "123456789" test vector
  /// ```
  pub struct Crc64Nvme {
    poly: CRC64_NVME_POLY,
    dispatcher: CRC64_NVME_DISPATCHER,
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
  /// # Example
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
    threshold_fn: crc64_simd_threshold,
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
  /// # Example
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
    threshold_fn: crc64_simd_threshold,
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

  /// Test streaming across the SIMD threshold boundary.
  ///
  /// This ensures correct state handling when the first chunk is below
  /// threshold (uses portable) and later chunks push us above threshold
  /// (switches to SIMD), or vice versa.
  #[test]
  fn test_crc64_streaming_across_threshold() {
    let threshold = Crc64::tunables().portable_to_clmul;
    if threshold == usize::MAX || threshold > (1 << 20) {
      // No meaningful SIMD crossover on this target/preset.
      return;
    }

    // Generate test data larger than threshold
    let size = threshold + 128;
    let data: Vec<u8> = (0..size).map(|i| (i as u8).wrapping_mul(31)).collect();

    let oneshot = Crc64::checksum(&data);

    // Stream: small chunk (below threshold), then rest (above threshold)
    let mut hasher = Crc64::new();
    hasher.update(&data[..16]); // Small, uses portable
    hasher.update(&data[16..]); // Large, may use SIMD
    assert_eq!(hasher.finalize(), oneshot, "Small-then-large streaming failed");

    // Stream: large chunk, then small chunk
    let mut hasher = Crc64::new();
    hasher.update(&data[..threshold + 64]); // Large, uses SIMD
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

    #[cfg(target_arch = "x86_64")]
    {
      let caps = platform::caps();
      let cfg = Crc64::config();

      if cfg.effective_force == Crc64Force::Portable {
        assert_eq!(name, "portable/slice16");
      } else if caps.has(platform::caps::x86::PCLMUL_READY) || caps.has(platform::caps::x86::VPCLMUL_READY) {
        assert_eq!(name, "x86_64/auto");
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
        assert_eq!(name, "aarch64/auto");
      } else {
        assert_eq!(name, "portable/slice16");
      }
    }

    // Introspection should always report portable for empty input.
    assert_eq!(Crc64::kernel_name_for_len(0), "portable/slice16");
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
    let streams_env = std::env::var("RSCRYPTO_CRC64_STREAMS").ok();

    if force.eq_ignore_ascii_case("portable") {
      assert_eq!(cfg.requested_force, Crc64Force::Portable);
      assert_eq!(kernel, "portable/slice16");
      return;
    }

    #[cfg(target_arch = "x86_64")]
    {
      if force.eq_ignore_ascii_case("pclmul") {
        assert_eq!(cfg.requested_force, Crc64Force::Pclmul);
        if cfg.effective_force == Crc64Force::Pclmul {
          if streams_env.is_some() {
            let expected = if len < CRC64_FOLD_BLOCK_BYTES {
              "x86_64/pclmul-small"
            } else {
              match x86_pclmul_streams_for_len(len, cfg.tunables.streams) {
                7 => "x86_64/pclmul-7way",
                4 => "x86_64/pclmul-4way",
                2 => "x86_64/pclmul-2way",
                _ => "x86_64/pclmul",
              }
            };
            assert_eq!(kernel, expected);
          } else {
            assert!(kernel.starts_with("x86_64/pclmul"));
          }
        }
        return;
      }

      if force.eq_ignore_ascii_case("vpclmul") {
        assert_eq!(cfg.requested_force, Crc64Force::Vpclmul);
        if cfg.effective_force == Crc64Force::Vpclmul {
          // For very large buffers, 4x512 is selected regardless of streams setting.
          if len >= CRC64_4X512_MIN_BYTES {
            assert_eq!(kernel, "x86_64/vpclmul-4x512");
          } else if streams_env.is_some() {
            let expected = match x86_vpclmul_streams_for_len(len, cfg.tunables.streams) {
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
      if force.eq_ignore_ascii_case("pmull") {
        assert_eq!(cfg.requested_force, Crc64Force::Pmull);
        if cfg.effective_force == Crc64Force::Pmull {
          assert!(kernel.starts_with("aarch64/pmull"));
        }
      }

      if force.eq_ignore_ascii_case("sve2-pmull")
        || force.eq_ignore_ascii_case("sve2")
        || force.eq_ignore_ascii_case("pmull-sve2")
      {
        assert_eq!(cfg.requested_force, Crc64Force::Sve2Pmull);
        if cfg.effective_force == Crc64Force::Sve2Pmull {
          if streams_env.is_some() {
            let expected = if len < CRC64_FOLD_BLOCK_BYTES {
              "aarch64/sve2-pmull-small"
            } else {
              match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
                3 => "aarch64/sve2-pmull-3way",
                2 => "aarch64/sve2-pmull-2way",
                _ => "aarch64/sve2-pmull",
              }
            };
            assert_eq!(kernel, expected);
          } else {
            assert!(kernel.starts_with("aarch64/sve2-pmull"));
          }
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
