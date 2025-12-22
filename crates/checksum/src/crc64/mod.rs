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

mod config;
mod portable;
mod tuned_defaults;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

use backend::dispatch::Selected;
// Re-export config types for public API (Crc64Force only used internally on SIMD archs)
#[allow(unused_imports)]
pub use config::{Crc64Config, Crc64Force, Crc64Tunables};
use traits::{Checksum, ChecksumCombine};

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", test))]
use crate::common::tables::generate_crc64_tables_8;
use crate::{
  common::{
    combine::{Gf2Matrix64, combine_crc64, generate_shift8_matrix_64},
    tables::{CRC64_NVME_POLY, CRC64_XZ_POLY, generate_crc64_tables_16},
  },
  dispatchers::{Crc64Dispatcher, Crc64Fn},
};

#[inline]
#[must_use]
fn crc64_selected_kernel_name(len: usize) -> &'static str {
  #[cfg(target_arch = "x86_64")]
  {
    let cfg = config::get();
    let caps = platform::caps();

    if len < CRC64_SMALL_LANE_BYTES || cfg.effective_force == Crc64Force::Portable {
      return "portable/slice16";
    }

    match cfg.effective_force {
      Crc64Force::Pclmul => {
        if caps.has(platform::caps::x86::PCLMUL_READY) {
          return if len < CRC64_FOLD_BLOCK_BYTES {
            "x86_64/pclmul-small"
          } else {
            match x86_pclmul_streams_for_len(len, cfg.tunables.streams) {
              7 => "x86_64/pclmul-7way",
              4 => "x86_64/pclmul-4way",
              2 => "x86_64/pclmul-2way",
              _ => "x86_64/pclmul",
            }
          };
        }
        return "portable/slice16";
      }
      Crc64Force::Vpclmul => {
        if caps.has(platform::caps::x86::VPCLMUL_READY) {
          return match x86_vpclmul_streams_for_len(len, cfg.tunables.streams) {
            7 => "x86_64/vpclmul-7way",
            4 => "x86_64/vpclmul-4way",
            2 => "x86_64/vpclmul-2way",
            _ => "x86_64/vpclmul",
          };
        }
        // Fall back to PCLMUL if available.
        if caps.has(platform::caps::x86::PCLMUL_READY) {
          return if len < CRC64_FOLD_BLOCK_BYTES {
            "x86_64/pclmul-small"
          } else {
            match x86_pclmul_streams_for_len(len, cfg.tunables.streams) {
              7 => "x86_64/pclmul-7way",
              4 => "x86_64/pclmul-4way",
              2 => "x86_64/pclmul-2way",
              _ => "x86_64/pclmul",
            }
          };
        }
        return "portable/slice16";
      }
      _ => {}
    }

    if len < cfg.tunables.portable_to_clmul {
      return "portable/slice16";
    }

    if caps.has(platform::caps::x86::VPCLMUL_READY) && len >= cfg.tunables.pclmul_to_vpclmul {
      return match x86_vpclmul_streams_for_len(len, cfg.tunables.streams) {
        7 => "x86_64/vpclmul-7way",
        4 => "x86_64/vpclmul-4way",
        2 => "x86_64/vpclmul-2way",
        _ => "x86_64/vpclmul",
      };
    }

    if caps.has(platform::caps::x86::PCLMUL_READY) {
      return if len < CRC64_FOLD_BLOCK_BYTES {
        "x86_64/pclmul-small"
      } else {
        match x86_pclmul_streams_for_len(len, cfg.tunables.streams) {
          7 => "x86_64/pclmul-7way",
          4 => "x86_64/pclmul-4way",
          2 => "x86_64/pclmul-2way",
          _ => "x86_64/pclmul",
        }
      };
    }

    if caps.has(platform::caps::x86::VPCLMUL_READY) {
      return match x86_vpclmul_streams_for_len(len, cfg.tunables.streams) {
        7 => "x86_64/vpclmul-7way",
        4 => "x86_64/vpclmul-4way",
        2 => "x86_64/vpclmul-2way",
        _ => "x86_64/vpclmul",
      };
    }

    "portable/slice16"
  }

  #[cfg(target_arch = "aarch64")]
  {
    let cfg = config::get();
    let caps = platform::caps();

    if len < CRC64_SMALL_LANE_BYTES || cfg.effective_force == Crc64Force::Portable {
      return "portable/slice16";
    }

    match cfg.effective_force {
      Crc64Force::Pmull => {
        if caps.has(platform::caps::aarch64::PMULL_READY) {
          return if len < CRC64_FOLD_BLOCK_BYTES {
            "aarch64/pmull-small"
          } else {
            match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
              3 => "aarch64/pmull-3way",
              2 => "aarch64/pmull-2way",
              _ => "aarch64/pmull",
            }
          };
        }
        return "portable/slice16";
      }
      Crc64Force::PmullEor3 => {
        if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
          return if len < CRC64_FOLD_BLOCK_BYTES {
            "aarch64/pmull-small" // EOR3 only benefits large-buffer folding
          } else {
            match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
              3 => "aarch64/pmull-eor3-3way",
              2 => "aarch64/pmull-eor3-2way",
              _ => "aarch64/pmull-eor3",
            }
          };
        }
        return "portable/slice16";
      }
      Crc64Force::Sve2Pmull => {
        if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
          return if len < CRC64_FOLD_BLOCK_BYTES {
            "aarch64/sve2-pmull-small"
          } else {
            match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
              3 => "aarch64/sve2-pmull-3way",
              2 => "aarch64/sve2-pmull-2way",
              _ => "aarch64/sve2-pmull",
            }
          };
        }
        return "portable/slice16";
      }
      _ => {}
    }

    if len < cfg.tunables.portable_to_clmul {
      return "portable/slice16";
    }

    // Auto selection: prefer EOR3 > SVE2 > PMULL tiers
    if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) && len >= CRC64_FOLD_BLOCK_BYTES {
      return match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
        3 => "aarch64/pmull-eor3-3way",
        2 => "aarch64/pmull-eor3-2way",
        _ => "aarch64/pmull-eor3",
      };
    }

    if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
      match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
        3 => return "aarch64/sve2-pmull-3way",
        2 => return "aarch64/sve2-pmull-2way",
        _ => {}
      }
    }

    if caps.has(platform::caps::aarch64::PMULL_READY) {
      return if len < CRC64_FOLD_BLOCK_BYTES {
        "aarch64/pmull-small"
      } else {
        match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
          3 => "aarch64/pmull-3way",
          2 => "aarch64/pmull-2way",
          _ => "aarch64/pmull",
        }
      };
    }

    "portable/slice16"
  }

  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  {
    let _ = len;
    "portable/slice16"
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

// Folding parameters (only used on SIMD architectures)
//
// - Small-buffer tier folds one 16B lane at a time.
// - Large-buffer tier folds 8×16B lanes per 128B block.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const CRC64_FOLD_BLOCK_BYTES: usize = 128;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const CRC64_SMALL_LANE_BYTES: usize = 16;

#[cfg(target_arch = "x86_64")]
#[inline]
const fn x86_pclmul_streams_for_len(len: usize, streams: u8) -> u8 {
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

// Note: Also matches x86_64-unknown-none where it may be unused (no auto-dispatch on bare metal)
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[allow(dead_code)]
#[inline]
fn crc64_simd_threshold() -> usize {
  config::get().tunables.portable_to_clmul
}

#[cfg(target_arch = "x86_64")]
fn crc64_xz_x86_64_auto(crc: u64, data: &[u8]) -> u64 {
  let len = data.len();

  // Avoid any dispatch overhead for the tiny regime.
  if len < CRC64_SMALL_LANE_BYTES {
    return crc64_xz_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  match cfg.effective_force {
    Crc64Force::Portable => return crc64_xz_portable(crc, data),
    Crc64Force::Pclmul => {
      if caps.has(platform::caps::x86::PCLMUL_READY) {
        if len < CRC64_FOLD_BLOCK_BYTES {
          return x86_64::crc64_xz_pclmul_small_safe(crc, data);
        }
        match x86_pclmul_streams_for_len(len, cfg.tunables.streams) {
          7 => return x86_64::crc64_xz_pclmul_7way_safe(crc, data),
          4 => return x86_64::crc64_xz_pclmul_4way_safe(crc, data),
          2 => return x86_64::crc64_xz_pclmul_2way_safe(crc, data),
          _ => {}
        }
        return x86_64::crc64_xz_pclmul_safe(crc, data);
      }
      return crc64_xz_portable(crc, data);
    }
    Crc64Force::Vpclmul => {
      if caps.has(platform::caps::x86::VPCLMUL_READY) {
        match x86_vpclmul_streams_for_len(len, cfg.tunables.streams) {
          7 => return x86_64::crc64_xz_vpclmul_7way_safe(crc, data),
          4 => return x86_64::crc64_xz_vpclmul_4way_safe(crc, data),
          2 => return x86_64::crc64_xz_vpclmul_2way_safe(crc, data),
          _ => return x86_64::crc64_xz_vpclmul_safe(crc, data),
        }
      }
      // Safe fallback: PCLMUL if available, else portable.
      if caps.has(platform::caps::x86::PCLMUL_READY) {
        if len < CRC64_FOLD_BLOCK_BYTES {
          return x86_64::crc64_xz_pclmul_small_safe(crc, data);
        }
        match x86_pclmul_streams_for_len(len, cfg.tunables.streams) {
          7 => return x86_64::crc64_xz_pclmul_7way_safe(crc, data),
          4 => return x86_64::crc64_xz_pclmul_4way_safe(crc, data),
          2 => return x86_64::crc64_xz_pclmul_2way_safe(crc, data),
          _ => {}
        }
        return x86_64::crc64_xz_pclmul_safe(crc, data);
      }
      return crc64_xz_portable(crc, data);
    }
    _ => {}
  }

  // Auto selection.
  if len < cfg.tunables.portable_to_clmul {
    return crc64_xz_portable(crc, data);
  }

  if caps.has(platform::caps::x86::VPCLMUL_READY) && len >= cfg.tunables.pclmul_to_vpclmul {
    match x86_vpclmul_streams_for_len(len, cfg.tunables.streams) {
      7 => return x86_64::crc64_xz_vpclmul_7way_safe(crc, data),
      4 => return x86_64::crc64_xz_vpclmul_4way_safe(crc, data),
      2 => return x86_64::crc64_xz_vpclmul_2way_safe(crc, data),
      _ => return x86_64::crc64_xz_vpclmul_safe(crc, data),
    }
  }

  if caps.has(platform::caps::x86::PCLMUL_READY) {
    if len < CRC64_FOLD_BLOCK_BYTES {
      return x86_64::crc64_xz_pclmul_small_safe(crc, data);
    }
    match x86_pclmul_streams_for_len(len, cfg.tunables.streams) {
      7 => return x86_64::crc64_xz_pclmul_7way_safe(crc, data),
      4 => return x86_64::crc64_xz_pclmul_4way_safe(crc, data),
      2 => return x86_64::crc64_xz_pclmul_2way_safe(crc, data),
      _ => {}
    }
    return x86_64::crc64_xz_pclmul_safe(crc, data);
  }

  // Unexpected: VPCLMUL without PCLMUL. Still safe to run VPCLMUL.
  if caps.has(platform::caps::x86::VPCLMUL_READY) {
    match x86_vpclmul_streams_for_len(len, cfg.tunables.streams) {
      7 => return x86_64::crc64_xz_vpclmul_7way_safe(crc, data),
      4 => return x86_64::crc64_xz_vpclmul_4way_safe(crc, data),
      2 => return x86_64::crc64_xz_vpclmul_2way_safe(crc, data),
      _ => return x86_64::crc64_xz_vpclmul_safe(crc, data),
    }
  }

  crc64_xz_portable(crc, data)
}

#[cfg(target_arch = "x86_64")]
fn crc64_nvme_x86_64_auto(crc: u64, data: &[u8]) -> u64 {
  let len = data.len();

  if len < CRC64_SMALL_LANE_BYTES {
    return crc64_nvme_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  match cfg.effective_force {
    Crc64Force::Portable => return crc64_nvme_portable(crc, data),
    Crc64Force::Pclmul => {
      if caps.has(platform::caps::x86::PCLMUL_READY) {
        if len < CRC64_FOLD_BLOCK_BYTES {
          return x86_64::crc64_nvme_pclmul_small_safe(crc, data);
        }
        match x86_pclmul_streams_for_len(len, cfg.tunables.streams) {
          7 => return x86_64::crc64_nvme_pclmul_7way_safe(crc, data),
          4 => return x86_64::crc64_nvme_pclmul_4way_safe(crc, data),
          2 => return x86_64::crc64_nvme_pclmul_2way_safe(crc, data),
          _ => {}
        }
        return x86_64::crc64_nvme_pclmul_safe(crc, data);
      }
      return crc64_nvme_portable(crc, data);
    }
    Crc64Force::Vpclmul => {
      if caps.has(platform::caps::x86::VPCLMUL_READY) {
        match x86_vpclmul_streams_for_len(len, cfg.tunables.streams) {
          7 => return x86_64::crc64_nvme_vpclmul_7way_safe(crc, data),
          4 => return x86_64::crc64_nvme_vpclmul_4way_safe(crc, data),
          2 => return x86_64::crc64_nvme_vpclmul_2way_safe(crc, data),
          _ => return x86_64::crc64_nvme_vpclmul_safe(crc, data),
        }
      }
      if caps.has(platform::caps::x86::PCLMUL_READY) {
        if len < CRC64_FOLD_BLOCK_BYTES {
          return x86_64::crc64_nvme_pclmul_small_safe(crc, data);
        }
        match x86_pclmul_streams_for_len(len, cfg.tunables.streams) {
          7 => return x86_64::crc64_nvme_pclmul_7way_safe(crc, data),
          4 => return x86_64::crc64_nvme_pclmul_4way_safe(crc, data),
          2 => return x86_64::crc64_nvme_pclmul_2way_safe(crc, data),
          _ => {}
        }
        return x86_64::crc64_nvme_pclmul_safe(crc, data);
      }
      return crc64_nvme_portable(crc, data);
    }
    _ => {}
  }

  if len < cfg.tunables.portable_to_clmul {
    return crc64_nvme_portable(crc, data);
  }

  if caps.has(platform::caps::x86::VPCLMUL_READY) && len >= cfg.tunables.pclmul_to_vpclmul {
    match x86_vpclmul_streams_for_len(len, cfg.tunables.streams) {
      7 => return x86_64::crc64_nvme_vpclmul_7way_safe(crc, data),
      4 => return x86_64::crc64_nvme_vpclmul_4way_safe(crc, data),
      2 => return x86_64::crc64_nvme_vpclmul_2way_safe(crc, data),
      _ => return x86_64::crc64_nvme_vpclmul_safe(crc, data),
    }
  }

  if caps.has(platform::caps::x86::PCLMUL_READY) {
    if len < CRC64_FOLD_BLOCK_BYTES {
      return x86_64::crc64_nvme_pclmul_small_safe(crc, data);
    }
    match x86_pclmul_streams_for_len(len, cfg.tunables.streams) {
      7 => return x86_64::crc64_nvme_pclmul_7way_safe(crc, data),
      4 => return x86_64::crc64_nvme_pclmul_4way_safe(crc, data),
      2 => return x86_64::crc64_nvme_pclmul_2way_safe(crc, data),
      _ => {}
    }
    return x86_64::crc64_nvme_pclmul_safe(crc, data);
  }

  if caps.has(platform::caps::x86::VPCLMUL_READY) {
    match x86_vpclmul_streams_for_len(len, cfg.tunables.streams) {
      7 => return x86_64::crc64_nvme_vpclmul_7way_safe(crc, data),
      4 => return x86_64::crc64_nvme_vpclmul_4way_safe(crc, data),
      2 => return x86_64::crc64_nvme_vpclmul_2way_safe(crc, data),
      _ => return x86_64::crc64_nvme_vpclmul_safe(crc, data),
    }
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

  // Explicit portable override always wins.
  if config::get().effective_force == Crc64Force::Portable {
    return Selected::new("portable/slice16", crc64_xz_portable);
  }

  if caps.has(platform::caps::x86::PCLMUL_READY) || caps.has(platform::caps::x86::VPCLMUL_READY) {
    return Selected::new("x86_64/auto", crc64_xz_x86_64_auto);
  }

  Selected::new("portable/slice16", crc64_xz_portable)
}

#[cfg(target_arch = "aarch64")]
fn crc64_xz_aarch64_auto(crc: u64, data: &[u8]) -> u64 {
  let len = data.len();
  if len < CRC64_SMALL_LANE_BYTES {
    return crc64_xz_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  match cfg.effective_force {
    Crc64Force::Portable => return crc64_xz_portable(crc, data),
    Crc64Force::PmullEor3 => {
      if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
        if len < CRC64_FOLD_BLOCK_BYTES {
          return aarch64::crc64_xz_pmull_small_safe(crc, data);
        }
        match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
          3 => return aarch64::crc64_xz_pmull_eor3_3way_safe(crc, data),
          2 => return aarch64::crc64_xz_pmull_eor3_2way_safe(crc, data),
          _ => {}
        }
        return aarch64::crc64_xz_pmull_eor3_safe(crc, data);
      }
      return crc64_xz_portable(crc, data);
    }
    Crc64Force::Sve2Pmull => {
      if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
        if len < CRC64_FOLD_BLOCK_BYTES {
          return aarch64::crc64_xz_sve2_pmull_small_safe(crc, data);
        }
        match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
          3 => return aarch64::crc64_xz_sve2_pmull_3way_safe(crc, data),
          2 => return aarch64::crc64_xz_sve2_pmull_2way_safe(crc, data),
          _ => {}
        }
        return aarch64::crc64_xz_sve2_pmull_safe(crc, data);
      }
      return crc64_xz_portable(crc, data);
    }
    Crc64Force::Pmull => {
      if caps.has(platform::caps::aarch64::PMULL_READY) {
        if len < CRC64_FOLD_BLOCK_BYTES {
          return aarch64::crc64_xz_pmull_small_safe(crc, data);
        }
        match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
          3 => return aarch64::crc64_xz_pmull_3way_safe(crc, data),
          2 => return aarch64::crc64_xz_pmull_2way_safe(crc, data),
          _ => {}
        }
        return aarch64::crc64_xz_pmull_safe(crc, data);
      }
      return crc64_xz_portable(crc, data);
    }
    _ => {}
  }

  if len < cfg.tunables.portable_to_clmul {
    return crc64_xz_portable(crc, data);
  }

  // Auto selection: prefer EOR3 > SVE2 > PMULL tiers
  if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) && len >= CRC64_FOLD_BLOCK_BYTES {
    match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
      3 => return aarch64::crc64_xz_pmull_eor3_3way_safe(crc, data),
      2 => return aarch64::crc64_xz_pmull_eor3_2way_safe(crc, data),
      _ => {}
    }
    return aarch64::crc64_xz_pmull_eor3_safe(crc, data);
  }

  if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
    match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
      3 => return aarch64::crc64_xz_sve2_pmull_3way_safe(crc, data),
      2 => return aarch64::crc64_xz_sve2_pmull_2way_safe(crc, data),
      _ => {}
    }
  }

  if caps.has(platform::caps::aarch64::PMULL_READY) {
    if len < CRC64_FOLD_BLOCK_BYTES {
      return aarch64::crc64_xz_pmull_small_safe(crc, data);
    }
    match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
      3 => return aarch64::crc64_xz_pmull_3way_safe(crc, data),
      2 => return aarch64::crc64_xz_pmull_2way_safe(crc, data),
      _ => {}
    }
    return aarch64::crc64_xz_pmull_safe(crc, data);
  }

  crc64_xz_portable(crc, data)
}

#[cfg(target_arch = "aarch64")]
fn crc64_nvme_aarch64_auto(crc: u64, data: &[u8]) -> u64 {
  let len = data.len();
  if len < CRC64_SMALL_LANE_BYTES {
    return crc64_nvme_portable(crc, data);
  }

  let cfg = config::get();
  let caps = platform::caps();

  match cfg.effective_force {
    Crc64Force::Portable => return crc64_nvme_portable(crc, data),
    Crc64Force::PmullEor3 => {
      if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) {
        if len < CRC64_FOLD_BLOCK_BYTES {
          return aarch64::crc64_nvme_pmull_small_safe(crc, data);
        }
        match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
          3 => return aarch64::crc64_nvme_pmull_eor3_3way_safe(crc, data),
          2 => return aarch64::crc64_nvme_pmull_eor3_2way_safe(crc, data),
          _ => {}
        }
        return aarch64::crc64_nvme_pmull_eor3_safe(crc, data);
      }
      return crc64_nvme_portable(crc, data);
    }
    Crc64Force::Sve2Pmull => {
      if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
        if len < CRC64_FOLD_BLOCK_BYTES {
          return aarch64::crc64_nvme_sve2_pmull_small_safe(crc, data);
        }
        match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
          3 => return aarch64::crc64_nvme_sve2_pmull_3way_safe(crc, data),
          2 => return aarch64::crc64_nvme_sve2_pmull_2way_safe(crc, data),
          _ => {}
        }
        return aarch64::crc64_nvme_sve2_pmull_safe(crc, data);
      }
      return crc64_nvme_portable(crc, data);
    }
    Crc64Force::Pmull => {
      if caps.has(platform::caps::aarch64::PMULL_READY) {
        if len < CRC64_FOLD_BLOCK_BYTES {
          return aarch64::crc64_nvme_pmull_small_safe(crc, data);
        }
        match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
          3 => return aarch64::crc64_nvme_pmull_3way_safe(crc, data),
          2 => return aarch64::crc64_nvme_pmull_2way_safe(crc, data),
          _ => {}
        }
        return aarch64::crc64_nvme_pmull_safe(crc, data);
      }
      return crc64_nvme_portable(crc, data);
    }
    _ => {}
  }

  if len < cfg.tunables.portable_to_clmul {
    return crc64_nvme_portable(crc, data);
  }

  // Auto selection: prefer EOR3 > SVE2 > PMULL tiers
  if caps.has(platform::caps::aarch64::PMULL_EOR3_READY) && len >= CRC64_FOLD_BLOCK_BYTES {
    match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
      3 => return aarch64::crc64_nvme_pmull_eor3_3way_safe(crc, data),
      2 => return aarch64::crc64_nvme_pmull_eor3_2way_safe(crc, data),
      _ => {}
    }
    return aarch64::crc64_nvme_pmull_eor3_safe(crc, data);
  }

  if caps.has(platform::caps::aarch64::SVE2_PMULL) && caps.has(platform::caps::aarch64::PMULL_READY) {
    match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
      3 => return aarch64::crc64_nvme_sve2_pmull_3way_safe(crc, data),
      2 => return aarch64::crc64_nvme_sve2_pmull_2way_safe(crc, data),
      _ => {}
    }
  }

  if caps.has(platform::caps::aarch64::PMULL_READY) {
    if len < CRC64_FOLD_BLOCK_BYTES {
      return aarch64::crc64_nvme_pmull_small_safe(crc, data);
    }
    match aarch64_pmull_streams_for_len(len, cfg.tunables.streams) {
      3 => return aarch64::crc64_nvme_pmull_3way_safe(crc, data),
      2 => return aarch64::crc64_nvme_pmull_2way_safe(crc, data),
      _ => {}
    }
    return aarch64::crc64_nvme_pmull_safe(crc, data);
  }

  crc64_nvme_portable(crc, data)
}

#[cfg(target_arch = "aarch64")]
fn select_crc64_xz() -> Selected<Crc64Fn> {
  let caps = platform::caps();

  if config::get().effective_force == Crc64Force::Portable {
    return Selected::new("portable/slice16", crc64_xz_portable);
  }

  if caps.has(platform::caps::aarch64::PMULL_READY) {
    return Selected::new("aarch64/auto", crc64_xz_aarch64_auto);
  }

  Selected::new("portable/slice16", crc64_xz_portable)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn select_crc64_xz() -> Selected<Crc64Fn> {
  Selected::new("portable/slice16", crc64_xz_portable)
}

/// Select the best CRC-64-NVME kernel for the current platform.
#[cfg(target_arch = "x86_64")]
fn select_crc64_nvme() -> Selected<Crc64Fn> {
  let caps = platform::caps();

  if config::get().effective_force == Crc64Force::Portable {
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

  if config::get().effective_force == Crc64Force::Portable {
    return Selected::new("portable/slice16", crc64_nvme_portable);
  }

  if caps.has(platform::caps::aarch64::PMULL_READY) {
    return Selected::new("aarch64/auto", crc64_nvme_aarch64_auto);
  }

  Selected::new("portable/slice16", crc64_nvme_portable)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn select_crc64_nvme() -> Selected<Crc64Fn> {
  Selected::new("portable/slice16", crc64_nvme_portable)
}

/// Static dispatcher for CRC-64-XZ.
static CRC64_XZ_DISPATCHER: Crc64Dispatcher = Crc64Dispatcher::new(select_crc64_xz);

/// Static dispatcher for CRC-64-NVME.
static CRC64_NVME_DISPATCHER: Crc64Dispatcher = Crc64Dispatcher::new(select_crc64_nvme);

// ─────────────────────────────────────────────────────────────────────────────
// CRC-64-XZ (ECMA-182)
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
/// # Example
///
/// ```ignore
/// use checksum::{Crc64Xz, Checksum};
///
/// let crc = Crc64Xz::checksum(b"123456789");
/// assert_eq!(crc, 0x995DC9BBDF1939FA); // "123456789" test vector
/// ```
#[derive(Clone, Default)]
pub struct Crc64 {
  state: u64,
}

impl Crc64 {
  /// Pre-computed shift-by-8 matrix for combine.
  const SHIFT8_MATRIX: Gf2Matrix64 = generate_shift8_matrix_64(CRC64_XZ_POLY);

  /// Create a hasher to resume from a previous CRC value.
  #[inline]
  #[must_use]
  pub const fn resume(crc: u64) -> Self {
    Self { state: crc ^ !0 }
  }

  /// Get the name of the currently selected backend.
  ///
  /// Returns the dispatcher name (e.g., "portable/slice16", "x86_64/auto").
  #[must_use]
  pub fn backend_name() -> &'static str {
    CRC64_XZ_DISPATCHER.backend_name()
  }

  /// Get the effective CRC-64 configuration (overrides + thresholds).
  #[must_use]
  pub fn config() -> Crc64Config {
    config::get()
  }

  /// Convenience accessor for the active CRC-64 tunables.
  #[must_use]
  pub fn tunables() -> Crc64Tunables {
    Self::config().tunables
  }

  /// Returns the kernel name that the selector would choose for `len`.
  ///
  /// This is intended for debugging/benchmarking and does not allocate.
  #[must_use]
  pub fn kernel_name_for_len(len: usize) -> &'static str {
    crc64_selected_kernel_name(len)
  }
}

impl Checksum for Crc64 {
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
    self.state = CRC64_XZ_DISPATCHER.call(self.state, data);
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

impl ChecksumCombine for Crc64 {
  fn combine(crc_a: u64, crc_b: u64, len_b: usize) -> u64 {
    combine_crc64(crc_a, crc_b, len_b, Self::SHIFT8_MATRIX)
  }
}

/// Explicit name for the XZ CRC-64 variant (alias of [`Crc64`]).
pub type Crc64Xz = Crc64;

// ─────────────────────────────────────────────────────────────────────────────
// CRC-64-NVME
// ─────────────────────────────────────────────────────────────────────────────

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
/// ```ignore
/// use checksum::{Crc64Nvme, Checksum};
///
/// let crc = Crc64Nvme::checksum(b"123456789");
/// // NVMe CRC-64 test vector
/// ```
#[derive(Clone, Default)]
pub struct Crc64Nvme {
  state: u64,
}

impl Crc64Nvme {
  /// Pre-computed shift-by-8 matrix for combine.
  const SHIFT8_MATRIX: Gf2Matrix64 = generate_shift8_matrix_64(CRC64_NVME_POLY);

  /// Create a hasher to resume from a previous CRC value.
  #[inline]
  #[must_use]
  pub const fn resume(crc: u64) -> Self {
    Self { state: crc ^ !0 }
  }

  /// Get the name of the currently selected backend.
  ///
  /// Returns the dispatcher name (e.g., "portable/slice16", "x86_64/auto").
  #[must_use]
  pub fn backend_name() -> &'static str {
    CRC64_NVME_DISPATCHER.backend_name()
  }

  /// Get the effective CRC-64 configuration (overrides + thresholds).
  #[must_use]
  pub fn config() -> Crc64Config {
    config::get()
  }

  /// Convenience accessor for the active CRC-64 tunables.
  #[must_use]
  pub fn tunables() -> Crc64Tunables {
    Self::config().tunables
  }

  /// Returns the kernel name that the selector would choose for `len`.
  ///
  /// This is intended for debugging/benchmarking and does not allocate.
  #[must_use]
  pub fn kernel_name_for_len(len: usize) -> &'static str {
    crc64_selected_kernel_name(len)
  }
}

impl Checksum for Crc64Nvme {
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
    self.state = CRC64_NVME_DISPATCHER.call(self.state, data);
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

impl ChecksumCombine for Crc64Nvme {
  fn combine(crc_a: u64, crc_b: u64, len_b: usize) -> u64 {
    combine_crc64(crc_a, crc_b, len_b, Self::SHIFT8_MATRIX)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffered CRC-64 Wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// Buffer size for buffered CRC wrappers.
///
/// Chosen to be larger than the maximum SIMD threshold (256 bytes on Intel SPR)
/// while remaining cache-friendly.
#[cfg(feature = "alloc")]
const BUFFERED_CRC_BUFFER_SIZE: usize = 512;

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
/// ```ignore
/// use checksum::{BufferedCrc64Xz, Checksum};
///
/// let mut hasher = BufferedCrc64Xz::new();
///
/// // Many small updates - internally buffered
/// for byte in data.iter() {
///     hasher.update(&[*byte]);
/// }
///
/// let crc = hasher.finalize();
/// ```
#[cfg(feature = "alloc")]
pub struct BufferedCrc64 {
  inner: Crc64,
  buffer: alloc::boxed::Box<[u8; BUFFERED_CRC_BUFFER_SIZE]>,
  len: usize,
}

#[cfg(feature = "alloc")]
impl BufferedCrc64 {
  /// Create a new buffered CRC-64-XZ hasher.
  #[must_use]
  pub fn new() -> Self {
    Self {
      inner: Crc64::new(),
      buffer: alloc::boxed::Box::new([0u8; BUFFERED_CRC_BUFFER_SIZE]),
      len: 0,
    }
  }

  /// Update the CRC with more data.
  ///
  /// Data is buffered internally until enough accumulates for efficient
  /// SIMD processing.
  #[allow(clippy::indexing_slicing)]
  // Safety: All slice indices are bounds-checked by the algorithm:
  // - self.len < BUFFERED_CRC_BUFFER_SIZE (invariant maintained by this function)
  // - fill = min(input.len(), space), so input[..fill] and buffer[len..len+fill] are valid
  // - aligned <= input.len() by construction
  pub fn update(&mut self, data: &[u8]) {
    let threshold = crc64_simd_threshold();
    let mut input = data;

    // If we have buffered data, try to fill and flush
    if self.len > 0 {
      let space = BUFFERED_CRC_BUFFER_SIZE - self.len;
      let fill = input.len().min(space);
      self.buffer[self.len..self.len + fill].copy_from_slice(&input[..fill]);
      self.len += fill;
      input = &input[fill..];

      // Flush if buffer is full or we have enough for SIMD
      if self.len >= BUFFERED_CRC_BUFFER_SIZE || (self.len >= threshold && input.is_empty()) {
        self.inner.update(&self.buffer[..self.len]);
        self.len = 0;
      }
    }

    // Process large chunks directly
    if input.len() >= threshold {
      // Find largest aligned chunk
      let aligned = (input.len() / threshold) * threshold;
      self.inner.update(&input[..aligned]);
      input = &input[aligned..];
    }

    // Buffer remainder
    if !input.is_empty() {
      self.buffer[..input.len()].copy_from_slice(input);
      self.len = input.len();
    }
  }

  /// Finalize and return the CRC value.
  ///
  /// Flushes any remaining buffered data before computing the final CRC.
  #[must_use]
  #[allow(clippy::indexing_slicing)]
  // Safety: self.len < BUFFERED_CRC_BUFFER_SIZE (invariant)
  pub fn finalize(&self) -> u64 {
    if self.len > 0 {
      // Clone inner to avoid mutating self
      let mut inner = self.inner.clone();
      inner.update(&self.buffer[..self.len]);
      inner.finalize()
    } else {
      self.inner.finalize()
    }
  }

  /// Reset the hasher to initial state.
  pub fn reset(&mut self) {
    self.inner.reset();
    self.len = 0;
  }

  /// Get the name of the currently selected backend.
  #[must_use]
  pub fn backend_name() -> &'static str {
    Crc64::backend_name()
  }
}

#[cfg(feature = "alloc")]
impl Default for BufferedCrc64 {
  fn default() -> Self {
    Self::new()
  }
}

/// Explicit name for the XZ buffered CRC-64 variant (alias of [`BufferedCrc64`]).
#[cfg(feature = "alloc")]
pub type BufferedCrc64Xz = BufferedCrc64;

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
/// ```ignore
/// use checksum::{BufferedCrc64Nvme, Checksum};
///
/// let mut hasher = BufferedCrc64Nvme::new();
///
/// // Many small updates - internally buffered
/// for byte in data.iter() {
///     hasher.update(&[*byte]);
/// }
///
/// let crc = hasher.finalize();
/// ```
#[cfg(feature = "alloc")]
pub struct BufferedCrc64Nvme {
  inner: Crc64Nvme,
  buffer: alloc::boxed::Box<[u8; BUFFERED_CRC_BUFFER_SIZE]>,
  len: usize,
}

#[cfg(feature = "alloc")]
impl BufferedCrc64Nvme {
  /// Create a new buffered CRC-64-NVME hasher.
  #[must_use]
  pub fn new() -> Self {
    Self {
      inner: Crc64Nvme::new(),
      buffer: alloc::boxed::Box::new([0u8; BUFFERED_CRC_BUFFER_SIZE]),
      len: 0,
    }
  }

  /// Update the CRC with more data.
  ///
  /// Data is buffered internally until enough accumulates for efficient
  /// SIMD processing.
  #[allow(clippy::indexing_slicing)]
  // Safety: All slice indices are bounds-checked by the algorithm:
  // - self.len < BUFFERED_CRC_BUFFER_SIZE (invariant maintained by this function)
  // - fill = min(input.len(), space), so input[..fill] and buffer[len..len+fill] are valid
  // - aligned <= input.len() by construction
  pub fn update(&mut self, data: &[u8]) {
    let threshold = crc64_simd_threshold();
    let mut input = data;

    // If we have buffered data, try to fill and flush
    if self.len > 0 {
      let space = BUFFERED_CRC_BUFFER_SIZE - self.len;
      let fill = input.len().min(space);
      self.buffer[self.len..self.len + fill].copy_from_slice(&input[..fill]);
      self.len += fill;
      input = &input[fill..];

      // Flush if buffer is full or we have enough for SIMD
      if self.len >= BUFFERED_CRC_BUFFER_SIZE || (self.len >= threshold && input.is_empty()) {
        self.inner.update(&self.buffer[..self.len]);
        self.len = 0;
      }
    }

    // Process large chunks directly
    if input.len() >= threshold {
      // Find largest aligned chunk
      let aligned = (input.len() / threshold) * threshold;
      self.inner.update(&input[..aligned]);
      input = &input[aligned..];
    }

    // Buffer remainder
    if !input.is_empty() {
      self.buffer[..input.len()].copy_from_slice(input);
      self.len = input.len();
    }
  }

  /// Finalize and return the CRC value.
  ///
  /// Flushes any remaining buffered data before computing the final CRC.
  #[must_use]
  #[allow(clippy::indexing_slicing)]
  // Safety: self.len < BUFFERED_CRC_BUFFER_SIZE (invariant)
  pub fn finalize(&self) -> u64 {
    if self.len > 0 {
      // Clone inner to avoid mutating self
      let mut inner = self.inner.clone();
      inner.update(&self.buffer[..self.len]);
      inner.finalize()
    } else {
      self.inner.finalize()
    }
  }

  /// Reset the hasher to initial state.
  pub fn reset(&mut self) {
    self.inner.reset();
    self.len = 0;
  }

  /// Get the name of the currently selected backend.
  #[must_use]
  pub fn backend_name() -> &'static str {
    Crc64Nvme::backend_name()
  }
}

#[cfg(feature = "alloc")]
impl Default for BufferedCrc64Nvme {
  fn default() -> Self {
    Self::new()
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
    let _ = Crc64::checksum(&data);
    let _ = Crc64Nvme::checksum(&data);

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
          if streams_env.is_some() {
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
}

#[cfg(test)]
mod proptests;
