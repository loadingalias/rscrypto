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
mod portable;
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

#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
use std::sync::OnceLock;

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
#[cfg(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "powerpc64",
  target_arch = "s390x",
  target_arch = "riscv64"
))]
const CRC32_FOLD_BLOCK_BYTES: usize = 128;

// Small-buffer tier folds a single 16B lane; large-buffer tiers fold 8×16B lanes per 128B block.
#[cfg(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "powerpc64",
  target_arch = "s390x",
  target_arch = "riscv64"
))]
const CRC32_SMALL_LANE_BYTES: usize = 16;

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

#[cfg(target_arch = "powerpc64")]
const CRC32_POWERPC64_2WAY_MIN_BYTES: usize = 128 * CRC32_FOLD_BLOCK_BYTES; // 16 KiB

#[cfg(target_arch = "powerpc64")]
#[inline]
const fn powerpc64_streams_for_len(len: usize, streams: u8) -> u8 {
  // POWER benefits from higher ILP, but multi-stream merging has setup costs.
  // Keep thresholds conservative by default (CRC64 parity).
  const CRC32_POWERPC64_4WAY_MIN_BYTES: usize = 2 * CRC32_POWERPC64_2WAY_MIN_BYTES; // 32 KiB
  const CRC32_POWERPC64_8WAY_MIN_BYTES: usize = 4 * CRC32_POWERPC64_2WAY_MIN_BYTES; // 64 KiB

  if streams >= 8 && len >= CRC32_POWERPC64_8WAY_MIN_BYTES {
    return 8;
  }
  if streams >= 4 && len >= CRC32_POWERPC64_4WAY_MIN_BYTES {
    return 4;
  }
  if streams >= 2 && len >= CRC32_POWERPC64_2WAY_MIN_BYTES {
    return 2;
  }
  1
}

#[cfg(target_arch = "s390x")]
const CRC32_S390X_2WAY_MIN_BYTES: usize = 128 * CRC32_FOLD_BLOCK_BYTES; // 16 KiB

#[cfg(target_arch = "s390x")]
#[inline]
const fn s390x_streams_for_len(len: usize, streams: u8) -> u8 {
  // z13+ has 32 vector registers and strong ILP; keep thresholds conservative.
  const CRC32_S390X_4WAY_MIN_BYTES: usize = 2 * CRC32_S390X_2WAY_MIN_BYTES; // 32 KiB

  if streams >= 4 && len >= CRC32_S390X_4WAY_MIN_BYTES {
    return 4;
  }
  if streams >= 2 && len >= CRC32_S390X_2WAY_MIN_BYTES {
    return 2;
  }
  1
}

#[cfg(target_arch = "riscv64")]
const CRC32_RISCV64_2WAY_MIN_BYTES: usize = 128 * CRC32_FOLD_BLOCK_BYTES; // 16 KiB

#[cfg(target_arch = "riscv64")]
#[inline]
const fn riscv64_streams_for_len(len: usize, streams: u8) -> u8 {
  const CRC32_RISCV64_4WAY_MIN_BYTES: usize = 2 * CRC32_RISCV64_2WAY_MIN_BYTES; // 32 KiB

  if streams >= 4 && len >= CRC32_RISCV64_4WAY_MIN_BYTES {
    return 4;
  }
  if streams >= 2 && len >= CRC32_RISCV64_2WAY_MIN_BYTES {
    return 2;
  }
  1
}

// ─────────────────────────────────────────────────────────────────────────────
// Cached Dispatch Params (std-only hot path)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "std", target_arch = "aarch64"))]
#[derive(Clone, Copy, Debug)]
struct Crc32Aarch64Auto {
  portable_to_hwcrc: usize,
  hwcrc_to_fusion: usize,
  has_crc: bool,
  has_pmull: bool,
  has_pmull_eor3: bool,
  has_sve2_pmull: bool,
}

#[cfg(all(feature = "std", target_arch = "aarch64"))]
static CRC32_AARCH64_AUTO: OnceLock<Crc32Aarch64Auto> = OnceLock::new();
#[cfg(all(feature = "std", target_arch = "aarch64"))]
static CRC32C_AARCH64_AUTO: OnceLock<Crc32Aarch64Auto> = OnceLock::new();

#[cfg(all(feature = "std", target_arch = "aarch64"))]
static CRC32_AARCH64_STREAMS: OnceLock<u8> = OnceLock::new();
#[cfg(all(feature = "std", target_arch = "aarch64"))]
static CRC32C_AARCH64_STREAMS: OnceLock<u8> = OnceLock::new();

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug)]
struct Crc32X86Auto {
  hwcrc_to_fusion: usize,
  fusion_to_vpclmul: usize,
  streams: u8,
  has_pclmul: bool,
  has_vpclmul: bool,
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug)]
struct Crc32cX86Auto {
  portable_to_hwcrc: usize,
  hwcrc_to_fusion: usize,
  fusion_to_avx512: usize,
  fusion_to_vpclmul: usize,
  streams: u8,
  has_crc32c: bool,
  has_pclmul: bool,
  has_vpclmul: bool,
  has_avx512: bool,
  has_pclmulqdq: bool,
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
static CRC32_X86_AUTO: OnceLock<Crc32X86Auto> = OnceLock::new();
#[cfg(all(feature = "std", target_arch = "x86_64"))]
static CRC32C_X86_AUTO: OnceLock<Crc32cX86Auto> = OnceLock::new();

#[cfg(all(feature = "std", target_arch = "x86_64"))]
static CRC32_X86_STREAMS: OnceLock<u8> = OnceLock::new();
#[cfg(all(feature = "std", target_arch = "x86_64"))]
static CRC32C_X86_STREAMS: OnceLock<u8> = OnceLock::new();

#[inline]
#[must_use]
pub(crate) fn crc32_selected_kernel_name(len: usize) -> &'static str {
  let cfg = config::get();

  #[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "s390x",
    target_arch = "riscv64"
  ))]
  if len < CRC32_SMALL_LANE_BYTES {
    return kernels::PORTABLE;
  }

  if cfg.effective_force == Crc32Force::Portable {
    return kernels::PORTABLE;
  }

  #[cfg(target_arch = "x86_64")]
  {
    let caps = platform::caps();
    use kernels::x86_64::*;
    let streams = x86_streams_for_len(len, cfg.tunables.streams);

    // CRC-32 (IEEE) has no HWCRC tier on x86; use the folding threshold.
    if cfg.effective_force == Crc32Force::Auto && len < cfg.tunables.hwcrc_to_fusion {
      return kernels::PORTABLE;
    }

    match cfg.effective_force {
      Crc32Force::Vpclmul if caps.has(platform::caps::x86::VPCLMUL_READY) => {
        return kernels::select_name(
          CRC32_VPCLMUL_NAMES,
          Some(CRC32_VPCLMUL_SMALL),
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
        );
      }
      Crc32Force::Vpclmul => return kernels::PORTABLE,
      Crc32Force::Pclmul if caps.has(platform::caps::x86::PCLMUL_READY) => {
        return kernels::select_name(
          CRC32_PCLMUL_NAMES,
          Some(CRC32_PCLMUL_SMALL),
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
        );
      }
      Crc32Force::Pclmul => return kernels::PORTABLE,
      _ => {}
    }

    if cfg.effective_force == Crc32Force::Auto
      && len >= cfg.tunables.hwcrc_to_fusion
      && caps.has(platform::caps::x86::PCLMUL_READY)
    {
      if len >= CRC32_FOLD_BLOCK_BYTES
        && len >= cfg.tunables.fusion_to_vpclmul
        && caps.has(platform::caps::x86::VPCLMUL_READY)
      {
        return kernels::select_name(
          CRC32_VPCLMUL_NAMES,
          Some(CRC32_VPCLMUL_SMALL),
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
        );
      }
      return kernels::select_name(
        CRC32_PCLMUL_NAMES,
        Some(CRC32_PCLMUL_SMALL),
        streams,
        len,
        CRC32_FOLD_BLOCK_BYTES,
      );
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    let caps = platform::caps();
    use kernels::aarch64::*;
    // Match the auto-kernel behavior: avoid the current 2/3-way wrappers that
    // use `combine_crc32` (GF(2) matrix exponentiation).
    let streams = 1u8;

    if cfg.effective_force == Crc32Force::Auto && len < cfg.tunables.portable_to_hwcrc {
      return kernels::PORTABLE;
    }

    match cfg.effective_force {
      Crc32Force::PmullEor3
        if caps.has(platform::caps::aarch64::CRC_READY) && caps.has(platform::caps::aarch64::PMULL_EOR3_READY) =>
      {
        return kernels::select_name(
          CRC32_PMULL_EOR3_NAMES,
          Some(CRC32_PMULL_SMALL),
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
        );
      }
      Crc32Force::Pmull => {
        if caps.has(platform::caps::aarch64::CRC_READY) && caps.has(platform::caps::aarch64::PMULL_READY) {
          return kernels::select_name(
            CRC32_PMULL_NAMES,
            Some(CRC32_PMULL_SMALL),
            streams,
            len,
            CRC32_FOLD_BLOCK_BYTES,
          );
        }
        return kernels::PORTABLE;
      }
      Crc32Force::Sve2Pmull
        if caps.has(platform::caps::aarch64::SVE2_PMULL)
          && caps.has(platform::caps::aarch64::PMULL_READY)
          && caps.has(platform::caps::aarch64::CRC_READY) =>
      {
        return kernels::select_name(
          CRC32_SVE2_PMULL_NAMES,
          Some(CRC32_SVE2_PMULL_SMALL),
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
        );
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
    if len >= cfg.tunables.hwcrc_to_fusion && caps.has(platform::caps::aarch64::CRC_READY) {
      if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
        return kernels::select_name(
          CRC32_PMULL_EOR3_NAMES,
          Some(CRC32_PMULL_SMALL),
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
        );
      }
      if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) && streams >= 2
      {
        return kernels::select_name(
          CRC32_SVE2_PMULL_NAMES,
          Some(CRC32_SVE2_PMULL_SMALL),
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
        );
      }
      if caps.has(platform::caps::aarch64::PMULL_READY) {
        return kernels::select_name(
          CRC32_PMULL_NAMES,
          Some(CRC32_PMULL_SMALL),
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
        );
      }
    }

    if caps.has(platform::caps::aarch64::CRC_READY) {
      return kernels::select_name(CRC32_HWCRC_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
    }
  }

  #[cfg(target_arch = "powerpc64")]
  {
    use kernels::powerpc64::*;
    let caps = platform::caps();
    let streams = powerpc64_streams_for_len(len, cfg.tunables.streams);

    match cfg.effective_force {
      Crc32Force::Vpmsum if caps.has(platform::caps::powerpc64::VPMSUM_READY) => {
        return kernels::select_name(CRC32_VPMSUM_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      Crc32Force::Vpmsum => return kernels::PORTABLE,
      _ => {}
    }

    if cfg.effective_force == Crc32Force::Auto && len < cfg.tunables.hwcrc_to_fusion {
      return kernels::PORTABLE;
    }

    if caps.has(platform::caps::powerpc64::VPMSUM_READY) {
      return kernels::select_name(CRC32_VPMSUM_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    use kernels::s390x::*;
    let caps = platform::caps();
    let streams = s390x_streams_for_len(len, cfg.tunables.streams);

    match cfg.effective_force {
      Crc32Force::Vgfm if caps.has(platform::caps::s390x::VECTOR) => {
        return kernels::select_name(CRC32_VGFM_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      Crc32Force::Vgfm => return kernels::PORTABLE,
      _ => {}
    }

    if cfg.effective_force == Crc32Force::Auto && len < cfg.tunables.hwcrc_to_fusion {
      return kernels::PORTABLE;
    }

    if caps.has(platform::caps::s390x::VECTOR) {
      return kernels::select_name(CRC32_VGFM_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
    }
  }

  #[cfg(target_arch = "riscv64")]
  {
    use kernels::riscv64::*;
    let caps = platform::caps();
    let streams = riscv64_streams_for_len(len, cfg.tunables.streams);

    match cfg.effective_force {
      Crc32Force::Zvbc if caps.has(platform::caps::riscv::ZVBC) => {
        return kernels::select_name(CRC32_ZVBC_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      Crc32Force::Zvbc => return kernels::PORTABLE,
      Crc32Force::Zbc if caps.has(platform::caps::riscv::ZBC) => {
        return kernels::select_name(CRC32_ZBC_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      Crc32Force::Zbc => return kernels::PORTABLE,
      _ => {}
    }

    if cfg.effective_force == Crc32Force::Auto && len < cfg.tunables.hwcrc_to_fusion {
      return kernels::PORTABLE;
    }

    if caps.has(platform::caps::riscv::ZVBC) {
      return kernels::select_name(CRC32_ZVBC_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
    }
    if caps.has(platform::caps::riscv::ZBC) {
      return kernels::select_name(CRC32_ZBC_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
    }
  }

  kernels::PORTABLE
}

#[inline]
#[must_use]
pub(crate) fn crc32c_selected_kernel_name(len: usize) -> &'static str {
  let cfg = config::get();
  let caps = platform::caps();

  #[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "s390x",
    target_arch = "riscv64"
  ))]
  if len < CRC32_SMALL_LANE_BYTES {
    return kernels::PORTABLE;
  }

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
    // Similar to x86_64: avoid multi-stream wrappers that combine lanes via
    // `combine_crc32` (GF(2) matrix exponentiation).
    let streams = 1u8;

    match cfg.effective_force {
      Crc32Force::PmullEor3
        if caps.has(platform::caps::aarch64::CRC_READY) && caps.has(platform::caps::aarch64::PMULL_EOR3_READY) =>
      {
        return kernels::select_name(
          CRC32C_PMULL_EOR3_NAMES,
          Some(CRC32C_PMULL_SMALL),
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
        );
      }
      Crc32Force::Pmull => {
        if caps.has(platform::caps::aarch64::CRC_READY) && caps.has(platform::caps::aarch64::PMULL_READY) {
          return kernels::select_name(
            CRC32C_PMULL_NAMES,
            Some(CRC32C_PMULL_SMALL),
            streams,
            len,
            CRC32_FOLD_BLOCK_BYTES,
          );
        }
        return kernels::PORTABLE;
      }
      Crc32Force::Sve2Pmull
        if caps.has(platform::caps::aarch64::SVE2_PMULL)
          && caps.has(platform::caps::aarch64::PMULL_READY)
          && caps.has(platform::caps::aarch64::CRC_READY) =>
      {
        return kernels::select_name(
          CRC32C_SVE2_PMULL_NAMES,
          Some(CRC32C_SVE2_PMULL_SMALL),
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
        );
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
    if len >= cfg.tunables.hwcrc_to_fusion && caps.has(platform::caps::aarch64::CRC_READY) {
      if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
        return kernels::select_name(
          CRC32C_PMULL_EOR3_NAMES,
          Some(CRC32C_PMULL_SMALL),
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
        );
      }
      if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) && streams >= 2
      {
        return kernels::select_name(
          CRC32C_SVE2_PMULL_NAMES,
          Some(CRC32C_SVE2_PMULL_SMALL),
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
        );
      }
      if caps.has(platform::caps::aarch64::PMULL_READY) {
        return kernels::select_name(
          CRC32C_PMULL_NAMES,
          Some(CRC32C_PMULL_SMALL),
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
        );
      }
    }

    if caps.has(platform::caps::aarch64::CRC_READY) {
      return kernels::select_name(CRC32C_HWCRC_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
    }
  }

  #[cfg(target_arch = "powerpc64")]
  {
    use kernels::powerpc64::*;
    let streams = powerpc64_streams_for_len(len, cfg.tunables.streams);

    match cfg.effective_force {
      Crc32Force::Vpmsum if caps.has(platform::caps::powerpc64::VPMSUM_READY) => {
        return kernels::select_name(CRC32C_VPMSUM_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      Crc32Force::Vpmsum => return kernels::PORTABLE,
      _ => {}
    }

    if caps.has(platform::caps::powerpc64::VPMSUM_READY) {
      return kernels::select_name(CRC32C_VPMSUM_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    use kernels::s390x::*;
    let streams = s390x_streams_for_len(len, cfg.tunables.streams);

    match cfg.effective_force {
      Crc32Force::Vgfm if caps.has(platform::caps::s390x::VECTOR) => {
        return kernels::select_name(CRC32C_VGFM_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      Crc32Force::Vgfm => return kernels::PORTABLE,
      _ => {}
    }

    if caps.has(platform::caps::s390x::VECTOR) {
      return kernels::select_name(CRC32C_VGFM_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
    }
  }

  #[cfg(target_arch = "riscv64")]
  {
    use kernels::riscv64::*;
    let streams = riscv64_streams_for_len(len, cfg.tunables.streams);

    match cfg.effective_force {
      Crc32Force::Zvbc if caps.has(platform::caps::riscv::ZVBC) => {
        return kernels::select_name(CRC32C_ZVBC_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      Crc32Force::Zvbc => return kernels::PORTABLE,
      Crc32Force::Zbc if caps.has(platform::caps::riscv::ZBC) => {
        return kernels::select_name(CRC32C_ZBC_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
      }
      Crc32Force::Zbc => return kernels::PORTABLE,
      _ => {}
    }

    if caps.has(platform::caps::riscv::ZVBC) {
      return kernels::select_name(CRC32C_ZVBC_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
    }
    if caps.has(platform::caps::riscv::ZBC) {
      return kernels::select_name(CRC32C_ZBC_NAMES, None, streams, len, CRC32_FOLD_BLOCK_BYTES);
    }
  }

  kernels::PORTABLE
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto Kernels (architecture-specific)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
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

#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn crc32c_x86_64_auto(crc: u32, data: &[u8]) -> u32 {
  let params = *CRC32C_X86_AUTO.get_or_init(|| {
    let cfg = config::get();
    let caps = platform::caps();
    Crc32cX86Auto {
      portable_to_hwcrc: cfg.tunables.portable_to_hwcrc,
      hwcrc_to_fusion: cfg.tunables.hwcrc_to_fusion,
      fusion_to_avx512: cfg.tunables.fusion_to_avx512,
      fusion_to_vpclmul: cfg.tunables.fusion_to_vpclmul,
      streams: cfg.tunables.streams,
      has_crc32c: caps.has(platform::caps::x86::CRC32C_READY),
      has_pclmul: caps.has(platform::caps::x86::PCLMUL_READY),
      has_vpclmul: caps.has(platform::caps::x86::VPCLMUL_READY),
      has_avx512: caps.has(platform::caps::x86::AVX512_READY),
      has_pclmulqdq: caps.has(platform::caps::x86::PCLMULQDQ),
    }
  });
  let len = data.len();

  if !params.has_crc32c || len < params.portable_to_hwcrc {
    return crc32c_portable(crc, data);
  }

  use kernels::x86_64::*;
  let streams = x86_streams_for_len(len, params.streams);

  if len >= params.hwcrc_to_fusion {
    if params.has_vpclmul && len >= params.fusion_to_vpclmul {
      return kernels::dispatch_streams(&CRC32C_FUSION_VPCLMUL, streams, crc, data);
    }
    if params.has_avx512 && len >= params.fusion_to_avx512 && params.has_pclmulqdq {
      return kernels::dispatch_streams(&CRC32C_FUSION_AVX512, streams, crc, data);
    }
    if params.has_pclmul {
      return kernels::dispatch_streams(&CRC32C_FUSION_SSE, streams, crc, data);
    }
  }

  kernels::dispatch_streams(&CRC32C_HWCRC, streams, crc, data)
}

#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
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
        return kernels::dispatch_with_small(
          &CRC32_VPCLMUL,
          CRC32_VPCLMUL_SMALL_KERNEL,
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
          crc,
          data,
        );
      }
      return crc32_portable(crc, data);
    }
    Crc32Force::Pclmul => {
      if caps.has(platform::caps::x86::PCLMUL_READY) {
        return kernels::dispatch_with_small(
          &CRC32_PCLMUL,
          CRC32_PCLMUL_SMALL_KERNEL,
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
          crc,
          data,
        );
      }
      return crc32_portable(crc, data);
    }
    Crc32Force::Hwcrc => return crc32_portable(crc, data),
    _ => {}
  }

  if len >= cfg.tunables.hwcrc_to_fusion && caps.has(platform::caps::x86::PCLMUL_READY) {
    if len >= CRC32_FOLD_BLOCK_BYTES
      && len >= cfg.tunables.fusion_to_vpclmul
      && caps.has(platform::caps::x86::VPCLMUL_READY)
    {
      return kernels::dispatch_with_small(
        &CRC32_VPCLMUL,
        CRC32_VPCLMUL_SMALL_KERNEL,
        streams,
        len,
        CRC32_FOLD_BLOCK_BYTES,
        crc,
        data,
      );
    }
    return kernels::dispatch_with_small(
      &CRC32_PCLMUL,
      CRC32_PCLMUL_SMALL_KERNEL,
      streams,
      len,
      CRC32_FOLD_BLOCK_BYTES,
      crc,
      data,
    );
  }

  crc32_portable(crc, data)
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn crc32_x86_64_auto(crc: u32, data: &[u8]) -> u32 {
  let params = *CRC32_X86_AUTO.get_or_init(|| {
    let cfg = config::get();
    let caps = platform::caps();
    Crc32X86Auto {
      hwcrc_to_fusion: cfg.tunables.hwcrc_to_fusion,
      fusion_to_vpclmul: cfg.tunables.fusion_to_vpclmul,
      streams: cfg.tunables.streams,
      has_pclmul: caps.has(platform::caps::x86::PCLMUL_READY),
      has_vpclmul: caps.has(platform::caps::x86::VPCLMUL_READY),
    }
  });
  let len = data.len();

  if !params.has_pclmul || len < params.hwcrc_to_fusion {
    return crc32_portable(crc, data);
  }

  use kernels::x86_64::*;
  let streams = x86_streams_for_len(len, params.streams);

  if params.has_vpclmul && len >= CRC32_FOLD_BLOCK_BYTES && len >= params.fusion_to_vpclmul {
    return kernels::dispatch_with_small(
      &CRC32_VPCLMUL,
      CRC32_VPCLMUL_SMALL_KERNEL,
      streams,
      len,
      CRC32_FOLD_BLOCK_BYTES,
      crc,
      data,
    );
  }
  kernels::dispatch_with_small(
    &CRC32_PCLMUL,
    CRC32_PCLMUL_SMALL_KERNEL,
    streams,
    len,
    CRC32_FOLD_BLOCK_BYTES,
    crc,
    data,
  )
}

#[cfg(all(target_arch = "aarch64", not(feature = "std")))]
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
  // The aarch64 CRC32 HWCRC/PMULL 2/3-way wrappers currently use `combine_crc32`
  // (GF(2) matrix exponentiation), which is much slower than the underlying
  // 1-way tuned kernels. Clamp streams to 1 until we have CLMUL/PMULL combine.
  let streams = 1u8;

  match cfg.effective_force {
    Crc32Force::PmullEor3 => {
      if caps.has(platform::caps::aarch64::CRC_READY) && caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
        return kernels::dispatch_with_small(
          &CRC32_PMULL_EOR3,
          CRC32_PMULL_SMALL_KERNEL,
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
          crc,
          data,
        );
      }
      return crc32_portable(crc, data);
    }
    Crc32Force::Pmull => {
      if caps.has(platform::caps::aarch64::CRC_READY) && caps.has(platform::caps::aarch64::PMULL_READY) {
        return kernels::dispatch_with_small(
          &CRC32_PMULL,
          CRC32_PMULL_SMALL_KERNEL,
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
          crc,
          data,
        );
      }
      return crc32_portable(crc, data);
    }
    Crc32Force::Sve2Pmull => {
      if caps.has(platform::caps::aarch64::CRC_READY)
        && caps.has(platform::caps::aarch64::SVE2_PMULL)
        && caps.has(platform::caps::aarch64::PMULL_READY)
      {
        return kernels::dispatch_with_small(
          &CRC32_SVE2_PMULL,
          CRC32_SVE2_PMULL_SMALL_KERNEL,
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
          crc,
          data,
        );
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
        return kernels::dispatch_with_small(
          &CRC32_PMULL_EOR3,
          CRC32_PMULL_SMALL_KERNEL,
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
          crc,
          data,
        );
      }
      if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) && streams >= 2
      {
        return kernels::dispatch_with_small(
          &CRC32_SVE2_PMULL,
          CRC32_SVE2_PMULL_SMALL_KERNEL,
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
          crc,
          data,
        );
      }
      if caps.has(platform::caps::aarch64::PMULL_READY) {
        return kernels::dispatch_with_small(
          &CRC32_PMULL,
          CRC32_PMULL_SMALL_KERNEL,
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
          crc,
          data,
        );
      }
    }
    return kernels::dispatch_streams(&CRC32_HWCRC, streams, crc, data);
  }

  crc32_portable(crc, data)
}

#[cfg(all(target_arch = "aarch64", feature = "std"))]
fn crc32_aarch64_auto(crc: u32, data: &[u8]) -> u32 {
  let params = *CRC32_AARCH64_AUTO.get_or_init(|| {
    let cfg = config::get();
    let caps = platform::caps();
    let has_crc = caps.has(platform::caps::aarch64::CRC_READY);
    Crc32Aarch64Auto {
      portable_to_hwcrc: cfg.tunables.portable_to_hwcrc,
      hwcrc_to_fusion: cfg.tunables.hwcrc_to_fusion,
      has_crc,
      has_pmull: has_crc && caps.has(platform::caps::aarch64::PMULL_READY),
      has_pmull_eor3: has_crc && caps.has(platform::caps::aarch64::PMULL_EOR3_READY),
      has_sve2_pmull: has_crc
        && caps.has(platform::caps::aarch64::PMULL_READY)
        && caps.has(platform::caps::aarch64::SVE2_PMULL),
    }
  });
  let len = data.len();

  if !params.has_crc || len < params.portable_to_hwcrc {
    return crc32_portable(crc, data);
  }

  use kernels::aarch64::*;
  // See the `not(feature = "std")` variant above: clamp streams to 1 until
  // multi-stream combine is implemented without `combine_crc32`.
  let streams = 1u8;

  if len >= params.hwcrc_to_fusion {
    if params.has_pmull_eor3 {
      return kernels::dispatch_with_small(
        &CRC32_PMULL_EOR3,
        CRC32_PMULL_SMALL_KERNEL,
        streams,
        len,
        CRC32_FOLD_BLOCK_BYTES,
        crc,
        data,
      );
    }
    if params.has_sve2_pmull && streams >= 2 {
      return kernels::dispatch_with_small(
        &CRC32_SVE2_PMULL,
        CRC32_SVE2_PMULL_SMALL_KERNEL,
        streams,
        len,
        CRC32_FOLD_BLOCK_BYTES,
        crc,
        data,
      );
    }
    if params.has_pmull {
      return kernels::dispatch_with_small(
        &CRC32_PMULL,
        CRC32_PMULL_SMALL_KERNEL,
        streams,
        len,
        CRC32_FOLD_BLOCK_BYTES,
        crc,
        data,
      );
    }
  }

  kernels::dispatch_streams(&CRC32_HWCRC, streams, crc, data)
}

#[cfg(all(target_arch = "aarch64", not(feature = "std")))]
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
  // See `crc32_aarch64_auto`: clamp streams to 1 until multi-stream combine is
  // implemented without `combine_crc32`.
  let streams = 1u8;

  match cfg.effective_force {
    Crc32Force::PmullEor3 => {
      if caps.has(platform::caps::aarch64::CRC_READY) && caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
        return kernels::dispatch_with_small(
          &CRC32C_PMULL_EOR3,
          CRC32C_PMULL_SMALL_KERNEL,
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
          crc,
          data,
        );
      }
      return crc32c_portable(crc, data);
    }
    Crc32Force::Pmull => {
      if caps.has(platform::caps::aarch64::CRC_READY) && caps.has(platform::caps::aarch64::PMULL_READY) {
        return kernels::dispatch_with_small(
          &CRC32C_PMULL,
          CRC32C_PMULL_SMALL_KERNEL,
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
          crc,
          data,
        );
      }
      return crc32c_portable(crc, data);
    }
    Crc32Force::Sve2Pmull => {
      if caps.has(platform::caps::aarch64::CRC_READY)
        && caps.has(platform::caps::aarch64::SVE2_PMULL)
        && caps.has(platform::caps::aarch64::PMULL_READY)
      {
        return kernels::dispatch_with_small(
          &CRC32C_SVE2_PMULL,
          CRC32C_SVE2_PMULL_SMALL_KERNEL,
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
          crc,
          data,
        );
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
        return kernels::dispatch_with_small(
          &CRC32C_PMULL_EOR3,
          CRC32C_PMULL_SMALL_KERNEL,
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
          crc,
          data,
        );
      }
      if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) && streams >= 2
      {
        return kernels::dispatch_with_small(
          &CRC32C_SVE2_PMULL,
          CRC32C_SVE2_PMULL_SMALL_KERNEL,
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
          crc,
          data,
        );
      }
      if caps.has(platform::caps::aarch64::PMULL_READY) {
        return kernels::dispatch_with_small(
          &CRC32C_PMULL,
          CRC32C_PMULL_SMALL_KERNEL,
          streams,
          len,
          CRC32_FOLD_BLOCK_BYTES,
          crc,
          data,
        );
      }
    }
    return kernels::dispatch_streams(&CRC32C_HWCRC, streams, crc, data);
  }

  crc32c_portable(crc, data)
}

#[cfg(all(target_arch = "aarch64", feature = "std"))]
fn crc32c_aarch64_auto(crc: u32, data: &[u8]) -> u32 {
  let params = *CRC32C_AARCH64_AUTO.get_or_init(|| {
    let cfg = config::get();
    let caps = platform::caps();
    let has_crc = caps.has(platform::caps::aarch64::CRC_READY);
    Crc32Aarch64Auto {
      portable_to_hwcrc: cfg.tunables.portable_to_hwcrc,
      hwcrc_to_fusion: cfg.tunables.hwcrc_to_fusion,
      has_crc,
      has_pmull: has_crc && caps.has(platform::caps::aarch64::PMULL_READY),
      has_pmull_eor3: has_crc && caps.has(platform::caps::aarch64::PMULL_EOR3_READY),
      has_sve2_pmull: has_crc
        && caps.has(platform::caps::aarch64::PMULL_READY)
        && caps.has(platform::caps::aarch64::SVE2_PMULL),
    }
  });
  let len = data.len();

  if !params.has_crc || len < params.portable_to_hwcrc {
    return crc32c_portable(crc, data);
  }

  use kernels::aarch64::*;
  // See the `not(feature = "std")` variant above: clamp streams to 1 until
  // multi-stream combine is implemented without `combine_crc32`.
  let streams = 1u8;

  if len >= params.hwcrc_to_fusion {
    if params.has_pmull_eor3 {
      return kernels::dispatch_with_small(
        &CRC32C_PMULL_EOR3,
        CRC32C_PMULL_SMALL_KERNEL,
        streams,
        len,
        CRC32_FOLD_BLOCK_BYTES,
        crc,
        data,
      );
    }
    if params.has_sve2_pmull && streams >= 2 {
      return kernels::dispatch_with_small(
        &CRC32C_SVE2_PMULL,
        CRC32C_SVE2_PMULL_SMALL_KERNEL,
        streams,
        len,
        CRC32_FOLD_BLOCK_BYTES,
        crc,
        data,
      );
    }
    if params.has_pmull {
      return kernels::dispatch_with_small(
        &CRC32C_PMULL,
        CRC32C_PMULL_SMALL_KERNEL,
        streams,
        len,
        CRC32_FOLD_BLOCK_BYTES,
        crc,
        data,
      );
    }
  }

  kernels::dispatch_streams(&CRC32C_HWCRC, streams, crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Forced Kernels (std-only hot path)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(target_arch = "aarch64", feature = "std"))]
fn crc32_aarch64_hwcrc_forced(crc: u32, data: &[u8]) -> u32 {
  use kernels::aarch64::*;
  // Avoid multi-stream wrappers that currently combine via `combine_crc32`.
  let streams = 1u8;
  kernels::dispatch_streams(&CRC32_HWCRC, streams, crc, data)
}

#[cfg(all(target_arch = "aarch64", feature = "std"))]
fn crc32c_aarch64_hwcrc_forced(crc: u32, data: &[u8]) -> u32 {
  use kernels::aarch64::*;
  // Avoid multi-stream wrappers that currently combine via `combine_crc32`.
  let streams = 1u8;
  kernels::dispatch_streams(&CRC32C_HWCRC, streams, crc, data)
}

#[cfg(all(target_arch = "aarch64", feature = "std"))]
fn crc32_aarch64_pmull_forced(crc: u32, data: &[u8]) -> u32 {
  use kernels::aarch64::*;
  let len = data.len();
  // Avoid multi-stream wrappers that currently combine via `combine_crc32`.
  let streams = 1u8;
  kernels::dispatch_with_small(
    &CRC32_PMULL,
    CRC32_PMULL_SMALL_KERNEL,
    streams,
    len,
    CRC32_FOLD_BLOCK_BYTES,
    crc,
    data,
  )
}

#[cfg(all(target_arch = "aarch64", feature = "std"))]
fn crc32_aarch64_pmull_eor3_forced(crc: u32, data: &[u8]) -> u32 {
  use kernels::aarch64::*;
  let len = data.len();
  // Avoid multi-stream wrappers that currently combine via `combine_crc32`.
  let streams = 1u8;
  kernels::dispatch_with_small(
    &CRC32_PMULL_EOR3,
    CRC32_PMULL_SMALL_KERNEL,
    streams,
    len,
    CRC32_FOLD_BLOCK_BYTES,
    crc,
    data,
  )
}

#[cfg(all(target_arch = "aarch64", feature = "std"))]
fn crc32c_aarch64_pmull_forced(crc: u32, data: &[u8]) -> u32 {
  use kernels::aarch64::*;
  let len = data.len();
  // Avoid multi-stream wrappers that currently combine via `combine_crc32`.
  let streams = 1u8;
  kernels::dispatch_with_small(
    &CRC32C_PMULL,
    CRC32C_PMULL_SMALL_KERNEL,
    streams,
    len,
    CRC32_FOLD_BLOCK_BYTES,
    crc,
    data,
  )
}

#[cfg(all(target_arch = "aarch64", feature = "std"))]
fn crc32c_aarch64_pmull_eor3_forced(crc: u32, data: &[u8]) -> u32 {
  use kernels::aarch64::*;
  let len = data.len();
  // Avoid multi-stream wrappers that currently combine via `combine_crc32`.
  let streams = 1u8;
  kernels::dispatch_with_small(
    &CRC32C_PMULL_EOR3,
    CRC32C_PMULL_SMALL_KERNEL,
    streams,
    len,
    CRC32_FOLD_BLOCK_BYTES,
    crc,
    data,
  )
}

#[cfg(all(target_arch = "aarch64", feature = "std"))]
fn crc32_aarch64_sve2_pmull_forced(crc: u32, data: &[u8]) -> u32 {
  use kernels::aarch64::*;
  let len = data.len();
  // Avoid multi-stream wrappers that currently combine via `combine_crc32`.
  let streams = 1u8;
  kernels::dispatch_with_small(
    &CRC32_SVE2_PMULL,
    CRC32_SVE2_PMULL_SMALL_KERNEL,
    streams,
    len,
    CRC32_FOLD_BLOCK_BYTES,
    crc,
    data,
  )
}

#[cfg(all(target_arch = "aarch64", feature = "std"))]
fn crc32c_aarch64_sve2_pmull_forced(crc: u32, data: &[u8]) -> u32 {
  use kernels::aarch64::*;
  let len = data.len();
  // Avoid multi-stream wrappers that currently combine via `combine_crc32`.
  let streams = 1u8;
  kernels::dispatch_with_small(
    &CRC32C_SVE2_PMULL,
    CRC32C_SVE2_PMULL_SMALL_KERNEL,
    streams,
    len,
    CRC32_FOLD_BLOCK_BYTES,
    crc,
    data,
  )
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn crc32_x86_64_pclmul_forced(crc: u32, data: &[u8]) -> u32 {
  use kernels::x86_64::*;
  let len = data.len();
  let streams_cfg = CRC32_X86_STREAMS.get().copied().unwrap_or(1);
  let streams = x86_streams_for_len(len, streams_cfg);
  kernels::dispatch_with_small(
    &CRC32_PCLMUL,
    CRC32_PCLMUL_SMALL_KERNEL,
    streams,
    len,
    CRC32_FOLD_BLOCK_BYTES,
    crc,
    data,
  )
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn crc32_x86_64_vpclmul_forced(crc: u32, data: &[u8]) -> u32 {
  use kernels::x86_64::*;
  let len = data.len();
  let streams_cfg = CRC32_X86_STREAMS.get().copied().unwrap_or(1);
  let streams = x86_streams_for_len(len, streams_cfg);
  kernels::dispatch_with_small(
    &CRC32_VPCLMUL,
    CRC32_VPCLMUL_SMALL_KERNEL,
    streams,
    len,
    CRC32_FOLD_BLOCK_BYTES,
    crc,
    data,
  )
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn crc32c_x86_64_hwcrc_forced(crc: u32, data: &[u8]) -> u32 {
  use kernels::x86_64::*;
  let len = data.len();
  let streams_cfg = CRC32C_X86_STREAMS.get().copied().unwrap_or(1);
  let streams = x86_streams_for_len(len, streams_cfg);
  kernels::dispatch_streams(&CRC32C_HWCRC, streams, crc, data)
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn crc32c_x86_64_pclmul_forced(crc: u32, data: &[u8]) -> u32 {
  use kernels::x86_64::*;
  let len = data.len();
  let streams_cfg = CRC32C_X86_STREAMS.get().copied().unwrap_or(1);
  let streams = x86_streams_for_len(len, streams_cfg);
  kernels::dispatch_streams(&CRC32C_FUSION_SSE, streams, crc, data)
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn crc32c_x86_64_vpclmul_forced(crc: u32, data: &[u8]) -> u32 {
  use kernels::x86_64::*;
  let len = data.len();
  let streams_cfg = CRC32C_X86_STREAMS.get().copied().unwrap_or(1);
  let streams = x86_streams_for_len(len, streams_cfg);
  kernels::dispatch_streams(&CRC32C_FUSION_VPCLMUL, streams, crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// CRC32 CLMUL Backends (POWER/s390x/RISC-V)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "powerpc64")]
fn crc32_powerpc64_auto(crc: u32, data: &[u8]) -> u32 {
  use kernels::powerpc64::*;
  let len = data.len();

  if len < CRC32_SMALL_LANE_BYTES {
    return crc32_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  // Handle forced backend selection
  match cfg.effective_force {
    Crc32Force::Portable => return crc32_portable(crc, data),
    Crc32Force::Vpmsum if caps.has(platform::caps::powerpc64::VPMSUM_READY) => {
      let streams = powerpc64_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_streams(&CRC32_VPMSUM, streams, crc, data);
    }
    Crc32Force::Vpmsum => return crc32_portable(crc, data),
    _ => {}
  }

  // Auto selection
  if cfg.effective_force == Crc32Force::Auto && len < cfg.tunables.hwcrc_to_fusion {
    return crc32_portable(crc, data);
  }

  if caps.has(platform::caps::powerpc64::VPMSUM_READY) {
    let streams = powerpc64_streams_for_len(len, cfg.tunables.streams);
    return kernels::dispatch_streams(&CRC32_VPMSUM, streams, crc, data);
  }

  crc32_portable(crc, data)
}

#[cfg(target_arch = "powerpc64")]
fn crc32c_powerpc64_auto(crc: u32, data: &[u8]) -> u32 {
  use kernels::powerpc64::*;
  let len = data.len();

  if len < CRC32_SMALL_LANE_BYTES {
    return crc32c_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  // Handle forced backend selection
  match cfg.effective_force {
    Crc32Force::Portable => return crc32c_portable(crc, data),
    Crc32Force::Vpmsum if caps.has(platform::caps::powerpc64::VPMSUM_READY) => {
      let streams = powerpc64_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_streams(&CRC32C_VPMSUM, streams, crc, data);
    }
    Crc32Force::Vpmsum => return crc32c_portable(crc, data),
    _ => {}
  }

  // Auto selection
  if cfg.effective_force == Crc32Force::Auto && len < cfg.tunables.portable_to_hwcrc {
    return crc32c_portable(crc, data);
  }

  if caps.has(platform::caps::powerpc64::VPMSUM_READY) {
    let streams = powerpc64_streams_for_len(len, cfg.tunables.streams);
    return kernels::dispatch_streams(&CRC32C_VPMSUM, streams, crc, data);
  }

  crc32c_portable(crc, data)
}

#[cfg(target_arch = "s390x")]
fn crc32_s390x_auto(crc: u32, data: &[u8]) -> u32 {
  use kernels::s390x::*;
  let len = data.len();

  if len < CRC32_SMALL_LANE_BYTES {
    return crc32_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  // Handle forced backend selection
  match cfg.effective_force {
    Crc32Force::Portable => return crc32_portable(crc, data),
    Crc32Force::Vgfm if caps.has(platform::caps::s390x::VECTOR) => {
      let streams = s390x_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_streams(&CRC32_VGFM, streams, crc, data);
    }
    Crc32Force::Vgfm => return crc32_portable(crc, data),
    _ => {}
  }

  // Auto selection
  if cfg.effective_force == Crc32Force::Auto && len < cfg.tunables.hwcrc_to_fusion {
    return crc32_portable(crc, data);
  }

  if caps.has(platform::caps::s390x::VECTOR) {
    let streams = s390x_streams_for_len(len, cfg.tunables.streams);
    return kernels::dispatch_streams(&CRC32_VGFM, streams, crc, data);
  }

  crc32_portable(crc, data)
}

#[cfg(target_arch = "s390x")]
fn crc32c_s390x_auto(crc: u32, data: &[u8]) -> u32 {
  use kernels::s390x::*;
  let len = data.len();

  if len < CRC32_SMALL_LANE_BYTES {
    return crc32c_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  // Handle forced backend selection
  match cfg.effective_force {
    Crc32Force::Portable => return crc32c_portable(crc, data),
    Crc32Force::Vgfm if caps.has(platform::caps::s390x::VECTOR) => {
      let streams = s390x_streams_for_len(len, cfg.tunables.streams);
      return kernels::dispatch_streams(&CRC32C_VGFM, streams, crc, data);
    }
    Crc32Force::Vgfm => return crc32c_portable(crc, data),
    _ => {}
  }

  // Auto selection
  if cfg.effective_force == Crc32Force::Auto && len < cfg.tunables.portable_to_hwcrc {
    return crc32c_portable(crc, data);
  }

  if caps.has(platform::caps::s390x::VECTOR) {
    let streams = s390x_streams_for_len(len, cfg.tunables.streams);
    return kernels::dispatch_streams(&CRC32C_VGFM, streams, crc, data);
  }

  crc32c_portable(crc, data)
}

#[cfg(target_arch = "riscv64")]
fn crc32_riscv64_auto(crc: u32, data: &[u8]) -> u32 {
  use kernels::riscv64::*;
  let len = data.len();

  if len < CRC32_SMALL_LANE_BYTES {
    return crc32_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();
  let streams = riscv64_streams_for_len(len, cfg.tunables.streams);

  // Handle forced backend selection
  match cfg.effective_force {
    Crc32Force::Portable => return crc32_portable(crc, data),
    Crc32Force::Zvbc if caps.has(platform::caps::riscv::ZVBC) => {
      return kernels::dispatch_streams(&CRC32_ZVBC, streams, crc, data);
    }
    Crc32Force::Zvbc => return crc32_portable(crc, data),
    Crc32Force::Zbc if caps.has(platform::caps::riscv::ZBC) => {
      return kernels::dispatch_streams(&CRC32_ZBC, streams, crc, data);
    }
    Crc32Force::Zbc => return crc32_portable(crc, data),
    _ => {}
  }

  // Auto selection
  if cfg.effective_force == Crc32Force::Auto && len < cfg.tunables.hwcrc_to_fusion {
    return crc32_portable(crc, data);
  }

  if caps.has(platform::caps::riscv::ZVBC) {
    return kernels::dispatch_streams(&CRC32_ZVBC, streams, crc, data);
  }
  if caps.has(platform::caps::riscv::ZBC) {
    return kernels::dispatch_streams(&CRC32_ZBC, streams, crc, data);
  }

  crc32_portable(crc, data)
}

#[cfg(target_arch = "riscv64")]
fn crc32c_riscv64_auto(crc: u32, data: &[u8]) -> u32 {
  use kernels::riscv64::*;
  let len = data.len();

  if len < CRC32_SMALL_LANE_BYTES {
    return crc32c_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();
  let streams = riscv64_streams_for_len(len, cfg.tunables.streams);

  // Handle forced backend selection
  match cfg.effective_force {
    Crc32Force::Portable => return crc32c_portable(crc, data),
    Crc32Force::Zvbc if caps.has(platform::caps::riscv::ZVBC) => {
      return kernels::dispatch_streams(&CRC32C_ZVBC, streams, crc, data);
    }
    Crc32Force::Zvbc => return crc32c_portable(crc, data),
    Crc32Force::Zbc if caps.has(platform::caps::riscv::ZBC) => {
      return kernels::dispatch_streams(&CRC32C_ZBC, streams, crc, data);
    }
    Crc32Force::Zbc => return crc32c_portable(crc, data),
    _ => {}
  }

  // Auto selection
  if cfg.effective_force == Crc32Force::Auto && len < cfg.tunables.portable_to_hwcrc {
    return crc32c_portable(crc, data);
  }

  if caps.has(platform::caps::riscv::ZVBC) {
    return kernels::dispatch_streams(&CRC32C_ZVBC, streams, crc, data);
  }
  if caps.has(platform::caps::riscv::ZBC) {
    return kernels::dispatch_streams(&CRC32C_ZBC, streams, crc, data);
  }

  crc32c_portable(crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher Selection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn select_crc32() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32_portable);
  }

  #[cfg(feature = "std")]
  {
    let _ = CRC32_X86_STREAMS.get_or_init(|| cfg.tunables.streams);
    match cfg.effective_force {
      Crc32Force::Auto => {
        if !caps.has(platform::caps::x86::PCLMUL_READY) {
          Selected::new(kernels::PORTABLE, crc32_portable)
        } else {
          let _ = CRC32_X86_AUTO.get_or_init(|| Crc32X86Auto {
            hwcrc_to_fusion: cfg.tunables.hwcrc_to_fusion,
            fusion_to_vpclmul: cfg.tunables.fusion_to_vpclmul,
            streams: cfg.tunables.streams,
            has_pclmul: caps.has(platform::caps::x86::PCLMUL_READY),
            has_vpclmul: caps.has(platform::caps::x86::VPCLMUL_READY),
          });
          Selected::new("x86_64/auto", crc32_x86_64_auto)
        }
      }
      Crc32Force::Pclmul if caps.has(platform::caps::x86::PCLMUL_READY) => {
        Selected::new("x86_64/pclmul", crc32_x86_64_pclmul_forced)
      }
      Crc32Force::Vpclmul if caps.has(platform::caps::x86::VPCLMUL_READY) => {
        Selected::new("x86_64/vpclmul", crc32_x86_64_vpclmul_forced)
      }
      _ => Selected::new(kernels::PORTABLE, crc32_portable),
    }
  }

  #[cfg(not(feature = "std"))]
  {
    if caps.has(platform::caps::x86::PCLMUL_READY) {
      return Selected::new("x86_64/auto", crc32_x86_64_auto);
    }
    Selected::new(kernels::PORTABLE, crc32_portable)
  }
}

#[cfg(target_arch = "x86_64")]
fn select_crc32c() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32c_portable);
  }

  #[cfg(feature = "std")]
  {
    if !caps.has(platform::caps::x86::CRC32C_READY) {
      return Selected::new(kernels::PORTABLE, crc32c_portable);
    }

    let _ = CRC32C_X86_STREAMS.get_or_init(|| cfg.tunables.streams);

    match cfg.effective_force {
      Crc32Force::Auto => {
        let _ = CRC32C_X86_AUTO.get_or_init(|| Crc32cX86Auto {
          portable_to_hwcrc: cfg.tunables.portable_to_hwcrc,
          hwcrc_to_fusion: cfg.tunables.hwcrc_to_fusion,
          fusion_to_avx512: cfg.tunables.fusion_to_avx512,
          fusion_to_vpclmul: cfg.tunables.fusion_to_vpclmul,
          streams: cfg.tunables.streams,
          has_crc32c: caps.has(platform::caps::x86::CRC32C_READY),
          has_pclmul: caps.has(platform::caps::x86::PCLMUL_READY),
          has_vpclmul: caps.has(platform::caps::x86::VPCLMUL_READY),
          has_avx512: caps.has(platform::caps::x86::AVX512_READY),
          has_pclmulqdq: caps.has(platform::caps::x86::PCLMULQDQ),
        });
        Selected::new("x86_64/auto", crc32c_x86_64_auto)
      }
      Crc32Force::Hwcrc if caps.has(platform::caps::x86::CRC32C_READY) => {
        Selected::new("x86_64/hwcrc", crc32c_x86_64_hwcrc_forced)
      }
      Crc32Force::Pclmul
        if caps.has(platform::caps::x86::CRC32C_READY) && caps.has(platform::caps::x86::PCLMUL_READY) =>
      {
        Selected::new("x86_64/fusion-sse", crc32c_x86_64_pclmul_forced)
      }
      Crc32Force::Vpclmul
        if caps.has(platform::caps::x86::CRC32C_READY) && caps.has(platform::caps::x86::VPCLMUL_READY) =>
      {
        Selected::new("x86_64/fusion-vpclmul", crc32c_x86_64_vpclmul_forced)
      }
      _ => Selected::new(kernels::PORTABLE, crc32c_portable),
    }
  }

  #[cfg(not(feature = "std"))]
  {
    if caps.has(platform::caps::x86::CRC32C_READY) {
      return Selected::new("x86_64/auto", crc32c_x86_64_auto);
    }
    Selected::new(kernels::PORTABLE, crc32c_portable)
  }
}

#[cfg(target_arch = "aarch64")]
fn select_crc32() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32_portable);
  }

  #[cfg(feature = "std")]
  {
    if !caps.has(platform::caps::aarch64::CRC_READY) {
      return Selected::new(kernels::PORTABLE, crc32_portable);
    }

    let _ = CRC32_AARCH64_STREAMS.get_or_init(|| cfg.tunables.streams);

    match cfg.effective_force {
      Crc32Force::Auto => Selected::new("aarch64/auto", crc32_aarch64_auto),
      Crc32Force::Hwcrc => Selected::new("aarch64/hwcrc", crc32_aarch64_hwcrc_forced),
      Crc32Force::Pmull if caps.has(platform::caps::aarch64::PMULL_READY) => {
        Selected::new("aarch64/pmull-v12e-v1", crc32_aarch64_pmull_forced)
      }
      Crc32Force::PmullEor3 if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) => {
        Selected::new("aarch64/pmull-eor3-v9s3x2e-s3", crc32_aarch64_pmull_eor3_forced)
      }
      Crc32Force::Sve2Pmull
        if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) =>
      {
        Selected::new("aarch64/sve2-pmull", crc32_aarch64_sve2_pmull_forced)
      }
      _ => Selected::new(kernels::PORTABLE, crc32_portable),
    }
  }

  #[cfg(not(feature = "std"))]
  {
    if caps.has(platform::caps::aarch64::CRC_READY) {
      return Selected::new("aarch64/auto", crc32_aarch64_auto);
    }
    Selected::new(kernels::PORTABLE, crc32_portable)
  }
}

#[cfg(target_arch = "aarch64")]
fn select_crc32c() -> Selected<Crc32Fn> {
  let caps = platform::caps();
  let cfg = config::get();

  if cfg.effective_force == Crc32Force::Portable {
    return Selected::new(kernels::PORTABLE, crc32c_portable);
  }

  #[cfg(feature = "std")]
  {
    if !caps.has(platform::caps::aarch64::CRC_READY) {
      return Selected::new(kernels::PORTABLE, crc32c_portable);
    }

    let _ = CRC32C_AARCH64_STREAMS.get_or_init(|| cfg.tunables.streams);

    match cfg.effective_force {
      Crc32Force::Auto => Selected::new("aarch64/auto", crc32c_aarch64_auto),
      Crc32Force::Hwcrc => Selected::new("aarch64/hwcrc", crc32c_aarch64_hwcrc_forced),
      Crc32Force::Pmull if caps.has(platform::caps::aarch64::PMULL_READY) => {
        Selected::new("aarch64/pmull-v12e-v1", crc32c_aarch64_pmull_forced)
      }
      Crc32Force::PmullEor3 if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) => {
        Selected::new("aarch64/pmull-eor3-v9s3x2e-s3", crc32c_aarch64_pmull_eor3_forced)
      }
      Crc32Force::Sve2Pmull
        if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) =>
      {
        Selected::new("aarch64/sve2-pmull", crc32c_aarch64_sve2_pmull_forced)
      }
      _ => Selected::new(kernels::PORTABLE, crc32c_portable),
    }
  }

  #[cfg(not(feature = "std"))]
  {
    if caps.has(platform::caps::aarch64::CRC_READY) {
      return Selected::new("aarch64/auto", crc32c_aarch64_auto);
    }
    Selected::new(kernels::PORTABLE, crc32c_portable)
  }
}

#[cfg(target_arch = "powerpc64")]
fn select_crc32() -> Selected<Crc32Fn> {
  let caps = platform::caps();

  if config::get().effective_force == Crc32Force::Portable {
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

  if config::get().effective_force == Crc32Force::Portable {
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

  if config::get().effective_force == Crc32Force::Portable {
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

  if config::get().effective_force == Crc32Force::Portable {
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

  if config::get().effective_force == Crc32Force::Portable {
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

  if config::get().effective_force == Crc32Force::Portable {
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
      let streams_env = std::env::var("RSCRYPTO_CRC32_STREAMS").ok();
      let caps = platform::caps();
      if force.eq_ignore_ascii_case("pclmul") || force.eq_ignore_ascii_case("clmul") {
        assert_eq!(cfg.requested_force, Crc32Force::Pclmul);
        if cfg.effective_force == Crc32Force::Pclmul && caps.has(platform::caps::x86::PCLMUL_READY) {
          if streams_env.is_some() {
            let expected = match x86_streams_for_len(len, cfg.tunables.streams) {
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
            let expected = match x86_streams_for_len(len, cfg.tunables.streams) {
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
      if force.eq_ignore_ascii_case("hwcrc") || force.eq_ignore_ascii_case("crc") {
        assert_eq!(cfg.requested_force, Crc32Force::Hwcrc);
        if cfg.effective_force == Crc32Force::Hwcrc && caps.has(platform::caps::aarch64::CRC_READY) {
          assert!(kernel.starts_with("aarch64/hwcrc"));
        }
      }
      if force.eq_ignore_ascii_case("pmull") {
        assert_eq!(cfg.requested_force, Crc32Force::Pmull);
        if cfg.effective_force == Crc32Force::Pmull && caps.has(platform::caps::aarch64::PMULL_READY) {
          assert!(kernel.starts_with("aarch64/pmull"));
        }
      }
      if force.eq_ignore_ascii_case("pmull-eor3") || force.eq_ignore_ascii_case("eor3") {
        assert_eq!(cfg.requested_force, Crc32Force::PmullEor3);
        if cfg.effective_force == Crc32Force::PmullEor3 && caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
          assert!(kernel.starts_with("aarch64/pmull-eor3"));
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
          assert!(kernel.starts_with("x86_64/hwcrc"));
        }
      }
      if force.eq_ignore_ascii_case("pclmul") || force.eq_ignore_ascii_case("clmul") {
        assert_eq!(cfg.requested_force, Crc32Force::Pclmul);
        if cfg.effective_force == Crc32Force::Pclmul
          && caps.has(platform::caps::x86::CRC32C_READY)
          && caps.has(platform::caps::x86::PCLMUL_READY)
        {
          if streams_env.is_some() {
            let expected = match x86_streams_for_len(len, cfg.tunables.streams) {
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
            let expected = match x86_streams_for_len(len, cfg.tunables.streams) {
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
      if force.eq_ignore_ascii_case("hwcrc") || force.eq_ignore_ascii_case("crc") {
        assert_eq!(cfg.requested_force, Crc32Force::Hwcrc);
        if cfg.effective_force == Crc32Force::Hwcrc && caps.has(platform::caps::aarch64::CRC_READY) {
          assert!(kernel.starts_with("aarch64/hwcrc"));
        }
      }
      if force.eq_ignore_ascii_case("pmull") {
        assert_eq!(cfg.requested_force, Crc32Force::Pmull);
        if cfg.effective_force == Crc32Force::Pmull && caps.has(platform::caps::aarch64::PMULL_READY) {
          assert!(kernel.starts_with("aarch64/pmull"));
        }
      }
      if force.eq_ignore_ascii_case("pmull-eor3") || force.eq_ignore_ascii_case("eor3") {
        assert_eq!(cfg.requested_force, Crc32Force::PmullEor3);
        if cfg.effective_force == Crc32Force::PmullEor3 && caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
          assert!(kernel.starts_with("aarch64/pmull-eor3"));
        }
      }
    }
  }
}

#[cfg(test)]
mod proptests;
