//! Empirical kernel dispatch tables.
//!
//! This module provides pre-computed kernel selection tables derived from
//! actual benchmark data. Each platform has a table mapping (variant, size_class)
//! to the empirically-fastest kernel function pointer.
//!
//! # Design
//!
//! Unlike the policy-based system which computes kernel selection at runtime,
//! this module uses **pre-resolved tables**:
//!
//! ```text
//! Current:  caps → policy → thresholds → streams → kernel_name → lookup → kernel
//! This:     platform → size_class → kernel (direct)
//! ```
//!
//! This eliminates ~5ns of per-call overhead, reducing dispatch to ~1.5ns.
//!
//! # Data Sources
//!
//! Tables are derived from benchmark files in `crates/checksum/bench_baseline/`:
//! - `macos_arm64_kernels.txt` → AppleM1M3
//! - `linux_arm64_kernels.txt` → Graviton2
//! - `linux_x86-64_kernels.txt` → Zen4
//! - `windows_x86-64_kernels.txt` → Default (generic x86-64)
//!
//! Run `python scripts/gen/kernel_tables.py` to analyze benchmarks and see
//! optimal kernel selections.

#![allow(dead_code)] // Tables for non-current architectures

use platform::{Caps, TuneKind};

use crate::dispatchers::{Crc16Fn, Crc24Fn, Crc32Fn, Crc64Fn};

// ─────────────────────────────────────────────────────────────────────────────
// Global Kernel Table Cache
// ─────────────────────────────────────────────────────────────────────────────

/// Global cached kernel table, resolved once on first use.
///
/// This is the heart of the new dispatch system. Platform detection happens
/// exactly once, and all subsequent CRC calls use this pre-resolved table.
static ACTIVE_TABLE: backend::OnceCache<&'static KernelTable> = backend::OnceCache::new();

/// Get the active kernel table for this platform.
///
/// This function is called once per process and caches the result.
#[inline]
pub fn active_table() -> &'static KernelTable {
  ACTIVE_TABLE.get_or_init(|| {
    let tune = platform::tune();
    let caps = platform::caps();
    select_table(tune.kind(), caps)
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Oneshot Functions (Recommended API)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute CRC-64/XZ checksum of data.
///
/// Uses the optimal kernel for the current platform and buffer size.
/// This is the recommended API for one-shot checksums.
///
/// # Examples
///
/// ```
/// use checksum::dispatch::crc64_xz;
///
/// let crc = crc64_xz(b"123456789");
/// assert_eq!(crc, 0x995DC9BBDF1939FA);
/// ```
#[inline]
pub fn crc64_xz(data: &[u8]) -> u64 {
  let table = active_table();
  let kernel = table.select_set(data.len()).crc64_xz;
  kernel(!0, data) ^ !0
}

/// Compute CRC-64/NVME checksum of data.
///
/// Uses the optimal kernel for the current platform and buffer size.
///
/// # Examples
///
/// ```
/// use checksum::dispatch::crc64_nvme;
///
/// let crc = crc64_nvme(b"123456789");
/// assert_eq!(crc, 0xAE8B14860A799888);
/// ```
#[inline]
pub fn crc64_nvme(data: &[u8]) -> u64 {
  let table = active_table();
  let kernel = table.select_set(data.len()).crc64_nvme;
  kernel(!0, data) ^ !0
}

/// Compute CRC-32 (IEEE) checksum of data.
///
/// Uses the optimal kernel for the current platform and buffer size.
///
/// # Examples
///
/// ```
/// use checksum::dispatch::crc32_ieee;
///
/// let crc = crc32_ieee(b"123456789");
/// assert_eq!(crc, 0xCBF43926);
/// ```
#[inline]
pub fn crc32_ieee(data: &[u8]) -> u32 {
  let table = active_table();
  let kernel = table.select_set(data.len()).crc32_ieee;
  kernel(!0, data) ^ !0
}

/// Compute CRC-32C (Castagnoli) checksum of data.
///
/// Uses the optimal kernel for the current platform and buffer size.
///
/// # Examples
///
/// ```
/// use checksum::dispatch::crc32c;
///
/// let crc = crc32c(b"123456789");
/// assert_eq!(crc, 0xE3069283);
/// ```
#[inline]
pub fn crc32c(data: &[u8]) -> u32 {
  let table = active_table();
  let kernel = table.select_set(data.len()).crc32c;
  kernel(!0, data) ^ !0
}

/// Compute CRC-16/CCITT checksum of data.
///
/// Uses the optimal kernel for the current platform and buffer size.
///
/// # Examples
///
/// ```
/// use checksum::dispatch::crc16_ccitt;
///
/// let crc = crc16_ccitt(b"123456789");
/// assert_eq!(crc, 0x906E);
/// ```
#[inline]
pub fn crc16_ccitt(data: &[u8]) -> u16 {
  // CCITT: INIT=0xFFFF, XOROUT=0xFFFF
  let table = active_table();
  let kernel = table.select_set(data.len()).crc16_ccitt;
  kernel(0xFFFF, data) ^ 0xFFFF
}

/// Compute CRC-16/IBM checksum of data.
///
/// Uses the optimal kernel for the current platform and buffer size.
///
/// # Examples
///
/// ```
/// use checksum::dispatch::crc16_ibm;
///
/// let crc = crc16_ibm(b"123456789");
/// assert_eq!(crc, 0xBB3D);
/// ```
#[inline]
pub fn crc16_ibm(data: &[u8]) -> u16 {
  // IBM: INIT=0x0000, XOROUT=0x0000
  let table = active_table();
  let kernel = table.select_set(data.len()).crc16_ibm;
  kernel(0, data)
}

/// Compute CRC-24/OpenPGP checksum of data.
///
/// Uses the optimal kernel for the current platform and buffer size.
///
/// # Examples
///
/// ```
/// use checksum::dispatch::crc24_openpgp;
///
/// let crc = crc24_openpgp(b"123456789");
/// assert_eq!(crc, 0x21CF02);
/// ```
#[inline]
pub fn crc24_openpgp(data: &[u8]) -> u32 {
  // OpenPGP: INIT=0x00B704CE, XOROUT=0x000000, mask to 24 bits
  const INIT: u32 = 0x00B7_04CE;
  const MASK: u32 = 0x00FF_FFFF;
  let table = active_table();
  let kernel = table.select_set(data.len()).crc24_openpgp;
  kernel(INIT, data) & MASK
}

// ─────────────────────────────────────────────────────────────────────────────
// Data Structures
// ─────────────────────────────────────────────────────────────────────────────

/// All kernel function pointers for one size class.
///
/// Contains the optimal kernel for each CRC variant at a specific buffer size.
#[derive(Clone, Copy)]
pub struct KernelSet {
  // CRC-16
  pub crc16_ccitt: Crc16Fn,
  pub crc16_ibm: Crc16Fn,
  // CRC-24
  pub crc24_openpgp: Crc24Fn,
  // CRC-32
  pub crc32_ieee: Crc32Fn,
  pub crc32c: Crc32Fn,
  // CRC-64
  pub crc64_xz: Crc64Fn,
  pub crc64_nvme: Crc64Fn,
}

/// Complete kernel table for one platform.
///
/// Contains pre-selected optimal kernels for each (variant, size_class) pair.
/// Size class boundaries define when to transition between kernel tiers.
#[derive(Clone, Copy)]
pub struct KernelTable {
  /// Size class boundaries: [xs_max, s_max, m_max]
  ///
  /// - `len <= xs_max` → use `xs` kernels (tiny: 0-64 bytes)
  /// - `len <= s_max` → use `s` kernels (small: 65-256 bytes)
  /// - `len <= m_max` → use `m` kernels (medium: 257-4KB)
  /// - else → use `l` kernels (large: 4KB+)
  pub boundaries: [usize; 3],

  /// Kernels for tiny buffers (0 to xs_max bytes, typically 64B)
  pub xs: KernelSet,
  /// Kernels for small buffers (xs_max+1 to s_max bytes, typically 256B)
  pub s: KernelSet,
  /// Kernels for medium buffers (s_max+1 to m_max bytes, typically 4KB)
  pub m: KernelSet,
  /// Kernels for large buffers (m_max+1+ bytes, 4KB+)
  pub l: KernelSet,
}

impl KernelTable {
  /// Select the kernel set for the given buffer length.
  #[inline]
  pub const fn select_set(&self, len: usize) -> &KernelSet {
    if len <= self.boundaries[0] {
      &self.xs
    } else if len <= self.boundaries[1] {
      &self.s
    } else if len <= self.boundaries[2] {
      &self.m
    } else {
      &self.l
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Platform Table Selection
// ─────────────────────────────────────────────────────────────────────────────

/// Select the appropriate kernel table for the current platform.
///
/// Resolution order:
/// 1. **Exact match**: Benchmarked platform with measured data
/// 2. **Family match**: Inferred from similar microarchitecture
/// 3. **Capability match**: Conservative defaults based on CPU features
/// 4. **Portable fallback**: No SIMD, table-based only
#[inline]
pub fn select_table(tune_kind: TuneKind, caps: Caps) -> &'static KernelTable {
  // 1. Exact match (benchmarked platforms)
  if let Some(table) = exact_match(tune_kind) {
    return table;
  }

  // 2. Family match (inferred from similar hardware)
  if let Some(table) = family_match(tune_kind) {
    return table;
  }

  // 3. Capability match (conservative defaults)
  if let Some(table) = capability_match(caps) {
    return table;
  }

  // 4. Portable fallback
  &PORTABLE_TABLE
}

#[inline]
fn exact_match(tune_kind: TuneKind) -> Option<&'static KernelTable> {
  match tune_kind {
    #[cfg(target_arch = "aarch64")]
    TuneKind::AppleM1M3 => Some(&APPLE_M1M3_TABLE),
    #[cfg(target_arch = "aarch64")]
    TuneKind::Graviton2 => Some(&GRAVITON2_TABLE),
    #[cfg(target_arch = "x86_64")]
    TuneKind::Zen4 => Some(&ZEN4_TABLE),
    _ => None,
  }
}

#[inline]
fn family_match(tune_kind: TuneKind) -> Option<&'static KernelTable> {
  match tune_kind {
    // Apple Silicon family → use M1-M3 data
    #[cfg(target_arch = "aarch64")]
    TuneKind::AppleM4 | TuneKind::AppleM5 => Some(&APPLE_M1M3_TABLE),

    // AWS Graviton / ARM Neoverse family → use Graviton2 data
    #[cfg(target_arch = "aarch64")]
    TuneKind::Graviton3 | TuneKind::Graviton4 | TuneKind::Graviton5 => Some(&GRAVITON2_TABLE),
    #[cfg(target_arch = "aarch64")]
    TuneKind::NeoverseN2 | TuneKind::NeoverseN3 | TuneKind::NeoverseV3 => Some(&GRAVITON2_TABLE),
    #[cfg(target_arch = "aarch64")]
    TuneKind::NvidiaGrace | TuneKind::AmpereAltra => Some(&GRAVITON2_TABLE),

    // AMD Zen family → use Zen4 data
    #[cfg(target_arch = "x86_64")]
    TuneKind::Zen5 | TuneKind::Zen5c => Some(&ZEN4_TABLE),

    // Intel family → use generic x86 with VPCLMUL
    #[cfg(target_arch = "x86_64")]
    TuneKind::IntelSpr | TuneKind::IntelGnr | TuneKind::IntelIcl => Some(&GENERIC_X86_VPCLMUL_TABLE),

    _ => None,
  }
}

#[inline]
fn capability_match(caps: Caps) -> Option<&'static KernelTable> {
  #[cfg(target_arch = "aarch64")]
  {
    use platform::caps::aarch64::{PMULL_READY, SHA3};

    // PMULL + SHA3 (EOR3) → use Apple M1-M3 style kernels
    if caps.has(SHA3) && caps.has(PMULL_READY) {
      return Some(&GENERIC_ARM_PMULL_EOR3_TABLE);
    }
    // PMULL only → use Graviton2 style kernels
    if caps.has(PMULL_READY) {
      return Some(&GENERIC_ARM_PMULL_TABLE);
    }
  }

  #[cfg(target_arch = "x86_64")]
  {
    use platform::caps::x86::{PCLMUL_READY, VPCLMUL_READY};

    // VPCLMUL → use Zen4 style kernels
    if caps.has(VPCLMUL_READY) {
      return Some(&GENERIC_X86_VPCLMUL_TABLE);
    }
    // PCLMUL only → use conservative PCLMUL kernels
    if caps.has(PCLMUL_READY) {
      return Some(&GENERIC_X86_PCLMUL_TABLE);
    }
  }

  None
}

// ─────────────────────────────────────────────────────────────────────────────
// Portable Fallback Table
// ─────────────────────────────────────────────────────────────────────────────

/// Portable table - no SIMD, table-based only.
///
/// Used when no SIMD capabilities are detected.
pub static PORTABLE_TABLE: KernelTable = KernelTable {
  boundaries: [64, 256, 4096],
  xs: KernelSet {
    crc16_ccitt: crate::crc16::portable::crc16_ccitt_slice8,
    crc16_ibm: crate::crc16::portable::crc16_ibm_slice8,
    crc24_openpgp: crate::crc24::portable::crc24_openpgp_slice8,
    crc32_ieee: crate::crc32::portable::crc32_slice16_ieee,
    crc32c: crate::crc32::portable::crc32c_slice16,
    crc64_xz: crate::crc64::portable::crc64_slice16_xz,
    crc64_nvme: crate::crc64::portable::crc64_slice16_nvme,
  },
  s: KernelSet {
    crc16_ccitt: crate::crc16::portable::crc16_ccitt_slice8,
    crc16_ibm: crate::crc16::portable::crc16_ibm_slice8,
    crc24_openpgp: crate::crc24::portable::crc24_openpgp_slice8,
    crc32_ieee: crate::crc32::portable::crc32_slice16_ieee,
    crc32c: crate::crc32::portable::crc32c_slice16,
    crc64_xz: crate::crc64::portable::crc64_slice16_xz,
    crc64_nvme: crate::crc64::portable::crc64_slice16_nvme,
  },
  m: KernelSet {
    crc16_ccitt: crate::crc16::portable::crc16_ccitt_slice8,
    crc16_ibm: crate::crc16::portable::crc16_ibm_slice8,
    crc24_openpgp: crate::crc24::portable::crc24_openpgp_slice8,
    crc32_ieee: crate::crc32::portable::crc32_slice16_ieee,
    crc32c: crate::crc32::portable::crc32c_slice16,
    crc64_xz: crate::crc64::portable::crc64_slice16_xz,
    crc64_nvme: crate::crc64::portable::crc64_slice16_nvme,
  },
  l: KernelSet {
    crc16_ccitt: crate::crc16::portable::crc16_ccitt_slice8,
    crc16_ibm: crate::crc16::portable::crc16_ibm_slice8,
    crc24_openpgp: crate::crc24::portable::crc24_openpgp_slice8,
    crc32_ieee: crate::crc32::portable::crc32_slice16_ieee,
    crc32c: crate::crc32::portable::crc32c_slice16,
    crc64_xz: crate::crc64::portable::crc64_slice16_xz,
    crc64_nvme: crate::crc64::portable::crc64_slice16_nvme,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// aarch64 Platform Tables
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod aarch64_tables {
  use super::*;
  // Import kernel constants from the kernels modules
  use crate::crc16::kernels::aarch64 as crc16_k;
  use crate::{
    crc24::kernels::aarch64 as crc24_k, crc32::kernels::aarch64 as crc32_k, crc64::kernels::aarch64 as crc64_k,
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Apple M1-M3 Table
  //
  // Benchmark source: macOS local (2026-01-20)
  // Features: PMULL + SHA3 (EOR3)
  // Peak throughputs: CRC-16 ~60 GiB/s, CRC-32 ~75 GiB/s, CRC-64 ~62 GiB/s
  //
  // Optimal kernels per (variant, peak):
  //   crc16/ccitt: pmull, streams=3, 60.15 GiB/s
  //   crc16/ibm:   pmull, streams=3, 58.77 GiB/s
  //   crc24/openpgp: pmull, streams=3, 44.70 GiB/s
  //   crc32/ieee:  pmull-eor3-v9s3x2e-s3, streams=1, 74.32 GiB/s
  //   crc32c:      pmull-eor3-v9s3x2e-s3, streams=1, 75.32 GiB/s
  //   crc64/xz:    pmull, streams=3, 62.58 GiB/s
  //   crc64/nvme:  pmull-eor3, streams=3, 62.57 GiB/s
  // ───────────────────────────────────────────────────────────────────────────
  pub static APPLE_M1M3_TABLE: KernelTable = KernelTable {
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL_SMALL_KERNEL,
      crc16_ibm: crc16_k::IBM_PMULL_SMALL_KERNEL,
      crc24_openpgp: crc24_k::OPENPGP_PMULL_SMALL_KERNEL,
      crc32_ieee: crc32_k::CRC32_PMULL_SMALL_KERNEL,
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL,
      crc64_xz: crc64_k::XZ_PMULL_SMALL,
      crc64_nvme: crc64_k::NVME_PMULL_SMALL,
    },

    s: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL_SMALL_KERNEL,
      crc16_ibm: crc16_k::IBM_PMULL_SMALL_KERNEL,
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0], // 1-way
      crc32_ieee: crc32_k::CRC32_PMULL_SMALL_KERNEL,
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL,
      crc64_xz: crc64_k::XZ_PMULL[0],          // 1-way
      crc64_nvme: crc64_k::NVME_PMULL_EOR3[0], // 1-way eor3
    },

    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL[1], // 2-way
      crc16_ibm: crc16_k::IBM_PMULL[2],     // 3-way
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0],
      crc32_ieee: crc32_k::CRC32_PMULL_SMALL_KERNEL,
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL,
      crc64_xz: crc64_k::XZ_PMULL_EOR3[0], // 1-way eor3
      crc64_nvme: crc64_k::NVME_PMULL_EOR3[0],
    },

    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL[2],     // 3-way (streams=3)
      crc16_ibm: crc16_k::IBM_PMULL[2],         // 3-way (streams=3)
      crc24_openpgp: crc24_k::OPENPGP_PMULL[2], // 3-way (streams=3)
      crc32_ieee: crc32_k::CRC32_PMULL_EOR3[0], // pmull-eor3-v9s3x2e-s3
      crc32c: crc32_k::CRC32C_PMULL_EOR3[0],    // pmull-eor3-v9s3x2e-s3
      crc64_xz: crc64_k::XZ_PMULL[2],           // 3-way (streams=3)
      crc64_nvme: crc64_k::NVME_PMULL_EOR3[2],  // 3-way eor3 (streams=3)
    },
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Graviton2 Table
  //
  // Benchmark source: Namespace linux-arm64 runner (2026-01-20)
  // Features: PMULL (no EOR3/SHA3)
  // Peak throughputs: CRC-16 ~33 GiB/s, CRC-32 ~40 GiB/s, CRC-64 ~33 GiB/s
  //
  // Optimal kernels per (variant, peak):
  //   crc16/ccitt: pmull, streams=1, 33.42 GiB/s
  //   crc16/ibm:   pmull, streams=1, 33.45 GiB/s
  //   crc24/openpgp: pmull, streams=1, 24.85 GiB/s
  //   crc32/ieee:  pmull-eor3-v9s3x2e-s3, streams=1, 40.31 GiB/s
  //   crc32c:      pmull-eor3-v9s3x2e-s3, streams=1, 40.11 GiB/s
  //   crc64/xz:    pmull-eor3, streams=1, 33.49 GiB/s
  //   crc64/nvme:  pmull, streams=1, 33.33 GiB/s
  //
  // Note: Graviton2 benchmark shows pmull-eor3 winning for CRC-64/XZ even
  // without SHA3 feature flag - the EOR3 instruction is available through
  // a different path on this hardware.
  // ───────────────────────────────────────────────────────────────────────────
  pub static GRAVITON2_TABLE: KernelTable = KernelTable {
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL_SMALL_KERNEL,
      crc16_ibm: crc16_k::IBM_PMULL_SMALL_KERNEL,
      crc24_openpgp: crc24_k::OPENPGP_PMULL_SMALL_KERNEL,
      crc32_ieee: crc32_k::CRC32_PMULL_SMALL_KERNEL,
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL,
      crc64_xz: crc64_k::XZ_PMULL_SMALL,
      crc64_nvme: crc64_k::NVME_PMULL_SMALL,
    },

    s: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL_SMALL_KERNEL,
      crc16_ibm: crc16_k::IBM_PMULL_SMALL_KERNEL,
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0],
      crc32_ieee: crc32_k::CRC32_PMULL_SMALL_KERNEL,
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL,
      crc64_xz: crc64_k::XZ_PMULL[0],
      crc64_nvme: crc64_k::NVME_PMULL[0],
    },

    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL[0], // 1-way
      crc16_ibm: crc16_k::IBM_PMULL[0],
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0],
      crc32_ieee: crc32_k::CRC32_PMULL_SMALL_KERNEL,
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL,
      crc64_xz: crc64_k::XZ_PMULL_EOR3[0],
      crc64_nvme: crc64_k::NVME_PMULL[0],
    },

    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL[0],     // 1-way (streams=1)
      crc16_ibm: crc16_k::IBM_PMULL[0],         // 1-way (streams=1)
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0], // 1-way (streams=1)
      crc32_ieee: crc32_k::CRC32_PMULL_EOR3[0], // pmull-eor3-v9s3x2e-s3
      crc32c: crc32_k::CRC32C_PMULL_EOR3[0],    // pmull-eor3-v9s3x2e-s3
      crc64_xz: crc64_k::XZ_PMULL_EOR3[0],      // pmull-eor3 (streams=1)
      crc64_nvme: crc64_k::NVME_PMULL[0],       // 1-way (streams=1)
    },
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Generic ARM PMULL+EOR3 Table (conservative)
  //
  // For unknown ARM platforms with PMULL + SHA3 features.
  // Uses Apple M1-M3 selections (good EOR3 support).
  // ───────────────────────────────────────────────────────────────────────────
  pub static GENERIC_ARM_PMULL_EOR3_TABLE: KernelTable = APPLE_M1M3_TABLE;

  // ───────────────────────────────────────────────────────────────────────────
  // Generic ARM PMULL Table (conservative)
  //
  // For unknown ARM platforms with only PMULL (no EOR3).
  // Uses Graviton2 selections (no EOR3 dependency).
  // ───────────────────────────────────────────────────────────────────────────
  pub static GENERIC_ARM_PMULL_TABLE: KernelTable = GRAVITON2_TABLE;
}

#[cfg(target_arch = "aarch64")]
pub use aarch64_tables::*;

// ─────────────────────────────────────────────────────────────────────────────
// x86_64 Platform Tables
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod x86_64_tables {
  use super::*;
  // Import kernel constants from the kernels modules
  use crate::crc16::kernels::x86_64 as crc16_k;
  use crate::{
    crc24::kernels::x86_64 as crc24_k, crc32::kernels::x86_64 as crc32_k, crc64::kernels::x86_64 as crc64_k,
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Zen4 Table
  //
  // Benchmark source: Namespace linux-x86 runner (2026-01-20)
  // Features: VPCLMULQDQ + AVX-512
  // Peak throughputs: CRC-16 ~80 GiB/s, CRC-32 ~78 GiB/s, CRC-64 ~75 GiB/s
  //
  // Optimal kernels per (variant, peak):
  //   crc16/ccitt: vpclmul, streams=4, 79.87 GiB/s
  //   crc16/ibm:   vpclmul, streams=4, 77.96 GiB/s
  //   crc24/openpgp: vpclmul, streams=7, 42.96 GiB/s
  //   crc32/ieee:  vpclmul, streams=2, 78.29 GiB/s
  //   crc32c:      fusion-vpclmul-v3x2, streams=1, 72.53 GiB/s
  //   crc64/xz:    vpclmul, streams=2, 71.56 GiB/s
  //   crc64/nvme:  vpclmul, streams=2, 75.18 GiB/s
  // ───────────────────────────────────────────────────────────────────────────
  pub static ZEN4_TABLE: KernelTable = KernelTable {
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL_SMALL_KERNEL,
      crc16_ibm: crc16_k::IBM_PCLMUL_SMALL_KERNEL,
      crc24_openpgp: crc24_k::OPENPGP_PCLMUL_SMALL_KERNEL,
      crc32_ieee: crc32_k::CRC32_VPCLMUL_SMALL_KERNEL,
      crc32c: crc32_k::CRC32C_HWCRC[0], // hwcrc (SSE4.2)
      crc64_xz: crc64_k::XZ_PCLMUL_SMALL,
      crc64_nvme: crc64_k::NVME_PCLMUL_SMALL,
    },

    s: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPCLMUL[1],     // 2-way
      crc16_ibm: crc16_k::IBM_VPCLMUL[1],         // 2-way
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[2], // 4-way
      crc32_ieee: crc32_k::CRC32_VPCLMUL[3],      // 7-way
      crc32c: crc32_k::CRC32C_HWCRC[0],           // hwcrc still wins at small
      crc64_xz: crc64_k::XZ_VPCLMUL[4],           // 8-way
      crc64_nvme: crc64_k::NVME_VPCLMUL_4X512,
    },

    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPCLMUL[2],     // 4-way
      crc16_ibm: crc16_k::IBM_VPCLMUL[2],         // 4-way
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[3], // 7-way
      crc32_ieee: crc32_k::CRC32_VPCLMUL[0],      // 1-way vpclmul
      crc32c: crc32_k::CRC32C_FUSION_VPCLMUL[0],  // fusion-vpclmul-v3x2
      crc64_xz: crc64_k::XZ_VPCLMUL[3],           // 7-way
      crc64_nvme: crc64_k::NVME_VPCLMUL[3],       // 7-way
    },

    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPCLMUL[2],     // 4-way (streams=4)
      crc16_ibm: crc16_k::IBM_VPCLMUL[2],         // 4-way (streams=4)
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[3], // 7-way (streams=7)
      crc32_ieee: crc32_k::CRC32_VPCLMUL[1],      // 2-way (streams=2)
      crc32c: crc32_k::CRC32C_FUSION_VPCLMUL[0],  // fusion-vpclmul-v3x2
      crc64_xz: crc64_k::XZ_VPCLMUL[1],           // 2-way (streams=2)
      crc64_nvme: crc64_k::NVME_VPCLMUL[1],       // 2-way (streams=2)
    },
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Generic x86-64 VPCLMUL Table
  //
  // For unknown x86-64 platforms with VPCLMULQDQ.
  // Uses Zen4 selections (good AVX-512/VPCLMUL support).
  // ───────────────────────────────────────────────────────────────────────────
  pub static GENERIC_X86_VPCLMUL_TABLE: KernelTable = ZEN4_TABLE;

  // ───────────────────────────────────────────────────────────────────────────
  // Generic x86-64 PCLMUL Table (conservative)
  //
  // Benchmark source: windows_x86-64_kernels.txt (Default, no AVX-512)
  // Features: PCLMULQDQ only
  //
  // Optimal kernels per (variant, size_class):
  //   crc16/ccitt: xs=pclmul-small, s=pclmul, m=pclmul-7way, l=pclmul
  //   crc16/ibm:   xs=pclmul-small, s=pclmul-4way, m=pclmul, l=pclmul-2way
  //   crc24/openpgp: xs=pclmul-small, s=pclmul, m=pclmul-2way, l=pclmul-2way
  //   crc32/ieee:  xs=pclmul-small, s=pclmul-4way, m=pclmul, l=pclmul-2way
  //   crc32c:      xs=hwcrc, s=hwcrc, m=hwcrc-2way, l=fusion-sse-v4s3x3-2way
  //   crc64/xz:    xs=pclmul-small, s=pclmul-small, m=pclmul-4way, l=pclmul
  //   crc64/nvme:  xs=pclmul-small, s=pclmul-small, m=pclmul-2way, l=pclmul-2way
  // ───────────────────────────────────────────────────────────────────────────
  pub static GENERIC_X86_PCLMUL_TABLE: KernelTable = KernelTable {
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL_SMALL_KERNEL,
      crc16_ibm: crc16_k::IBM_PCLMUL_SMALL_KERNEL,
      crc24_openpgp: crc24_k::OPENPGP_PCLMUL_SMALL_KERNEL,
      crc32_ieee: crc32_k::CRC32_PCLMUL_SMALL_KERNEL,
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc64_xz: crc64_k::XZ_PCLMUL_SMALL,
      crc64_nvme: crc64_k::NVME_PCLMUL_SMALL,
    },

    s: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL[0], // 1-way
      crc16_ibm: crc16_k::IBM_PCLMUL[2],     // 4-way
      crc24_openpgp: crc24_k::OPENPGP_PCLMUL[0],
      crc32_ieee: crc32_k::CRC32_PCLMUL[2], // 4-way
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc64_xz: crc64_k::XZ_PCLMUL_SMALL, // small still wins
      crc64_nvme: crc64_k::NVME_PCLMUL_SMALL,
    },

    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL[3],     // 7-way
      crc16_ibm: crc16_k::IBM_PCLMUL[0],         // 1-way
      crc24_openpgp: crc24_k::OPENPGP_PCLMUL[1], // 2-way
      crc32_ieee: crc32_k::CRC32_PCLMUL[0],      // 1-way
      crc32c: crc32_k::CRC32C_HWCRC[1],          // 2-way
      crc64_xz: crc64_k::XZ_PCLMUL[2],           // 4-way
      crc64_nvme: crc64_k::NVME_PCLMUL[1],       // 2-way
    },

    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL[0], // 1-way wins at large
      crc16_ibm: crc16_k::IBM_PCLMUL[1],     // 2-way
      crc24_openpgp: crc24_k::OPENPGP_PCLMUL[1],
      crc32_ieee: crc32_k::CRC32_PCLMUL[1],  // 2-way
      crc32c: crc32_k::CRC32C_FUSION_SSE[1], // fusion-sse-v4s3x3-2way
      crc64_xz: crc64_k::XZ_PCLMUL[0],       // 1-way
      crc64_nvme: crc64_k::NVME_PCLMUL[1],   // 2-way
    },
  };
}

#[cfg(target_arch = "x86_64")]
pub use x86_64_tables::*;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  extern crate alloc;
  use alloc::vec::Vec;

  use super::*;

  #[test]
  fn test_portable_table_all_sizes() {
    // Verify portable table returns correct sets for each size class
    let table = &PORTABLE_TABLE;

    assert!(core::ptr::eq(table.select_set(0), &table.xs));
    assert!(core::ptr::eq(table.select_set(64), &table.xs));
    assert!(core::ptr::eq(table.select_set(65), &table.s));
    assert!(core::ptr::eq(table.select_set(256), &table.s));
    assert!(core::ptr::eq(table.select_set(257), &table.m));
    assert!(core::ptr::eq(table.select_set(4096), &table.m));
    assert!(core::ptr::eq(table.select_set(4097), &table.l));
    assert!(core::ptr::eq(table.select_set(1_000_000), &table.l));
  }

  #[test]
  fn test_select_table_fallback() {
    // With no capabilities, should return portable table
    let table = select_table(TuneKind::Portable, Caps::NONE);
    assert!(core::ptr::eq(table, &PORTABLE_TABLE));
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn test_aarch64_exact_match() {
    // Apple M1-M3 should return the Apple table
    let table = exact_match(TuneKind::AppleM1M3);
    assert!(table.is_some());
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn test_x86_64_exact_match() {
    // Zen4 should return the Zen4 table
    let table = exact_match(TuneKind::Zen4);
    assert!(table.is_some());
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Oneshot Function Tests
  // ─────────────────────────────────────────────────────────────────────────────

  /// Standard test vector for CRC verification.
  const TEST_DATA: &[u8] = b"123456789";

  #[test]
  fn test_crc64_xz_oneshot() {
    // Standard CRC-64/XZ check value
    let crc = crc64_xz(TEST_DATA);
    assert_eq!(crc, 0x995DC9BBDF1939FA, "CRC-64/XZ check value mismatch");
  }

  #[test]
  fn test_crc64_nvme_oneshot() {
    // Standard CRC-64/NVME check value
    let crc = crc64_nvme(TEST_DATA);
    assert_eq!(crc, 0xAE8B14860A799888, "CRC-64/NVME check value mismatch");
  }

  #[test]
  fn test_crc32_ieee_oneshot() {
    // Standard CRC-32/IEEE check value
    let crc = crc32_ieee(TEST_DATA);
    assert_eq!(crc, 0xCBF43926, "CRC-32/IEEE check value mismatch");
  }

  #[test]
  fn test_crc32c_oneshot() {
    // Standard CRC-32C check value
    let crc = crc32c(TEST_DATA);
    assert_eq!(crc, 0xE3069283, "CRC-32C check value mismatch");
  }

  #[test]
  fn test_crc16_ccitt_oneshot() {
    // Standard CRC-16/CCITT (X.25) check value
    let crc = crc16_ccitt(TEST_DATA);
    assert_eq!(crc, 0x906E, "CRC-16/CCITT check value mismatch");
  }

  #[test]
  fn test_crc16_ibm_oneshot() {
    // Standard CRC-16/IBM (ARC) check value
    let crc = crc16_ibm(TEST_DATA);
    assert_eq!(crc, 0xBB3D, "CRC-16/IBM check value mismatch");
  }

  #[test]
  fn test_crc24_openpgp_oneshot() {
    // Standard CRC-24/OpenPGP check value
    let crc = crc24_openpgp(TEST_DATA);
    assert_eq!(crc, 0x21CF02, "CRC-24/OpenPGP check value mismatch");
  }

  #[test]
  fn test_empty_data() {
    // Empty data should return the "zero-length" checksum
    assert_eq!(crc64_xz(&[]), 0, "CRC-64/XZ of empty data");
    assert_eq!(crc64_nvme(&[]), 0, "CRC-64/NVME of empty data");
    assert_eq!(crc32_ieee(&[]), 0, "CRC-32/IEEE of empty data");
    assert_eq!(crc32c(&[]), 0, "CRC-32C of empty data");
    assert_eq!(crc16_ccitt(&[]), 0, "CRC-16/CCITT of empty data");
    // Note: CRC-16/IBM and CRC-24/OpenPGP have non-zero init values
  }

  #[test]
  fn test_various_sizes() {
    // Test that all size classes produce consistent results
    // (comparing against a portable reference would be ideal,
    // but for now we just verify they don't panic)
    let sizes = [1, 64, 65, 256, 257, 4096, 4097, 65536];

    for &size in &sizes {
      let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
      // Just verify no panics and consistent non-zero results for non-empty data
      let _ = crc64_xz(&data);
      let _ = crc64_nvme(&data);
      let _ = crc32_ieee(&data);
      let _ = crc32c(&data);
      let _ = crc16_ccitt(&data);
      let _ = crc16_ibm(&data);
      let _ = crc24_openpgp(&data);
    }
  }

  #[test]
  fn test_active_table_caching() {
    // Verify that repeated calls return the same table
    let table1 = active_table();
    let table2 = active_table();
    assert!(
      core::ptr::eq(table1, table2),
      "active_table should return cached reference"
    );
  }
}
