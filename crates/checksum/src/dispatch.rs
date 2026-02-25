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
//! - `linux_x86-64_zen5_kernels.txt` → Zen5
//! - `linux_x86-64_intel_kernels.txt` → IntelSpr
//!
//! Run `python scripts/gen/kernel_tables.py` to analyze benchmarks and emit a
//! draft table module at `crates/checksum/src/generated/kernel_tables.rs` for
//! inspection. The in-crate dispatch tables in this file are authoritative.

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
// Vectored Oneshot Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute CRC-64/XZ checksum of multiple buffers treated as one concatenated stream.
#[inline]
pub fn crc64_xz_vectored(bufs: &[&[u8]]) -> u64 {
  let table = active_table();
  let mut crc: u64 = !0;
  let mut last_set: *const KernelSet = core::ptr::null();
  let mut kernel = table.xs.crc64_xz;

  for &buf in bufs {
    if buf.is_empty() {
      continue;
    }
    let set = table.select_set(buf.len());
    let set_ptr: *const KernelSet = core::ptr::from_ref(set);
    if set_ptr != last_set {
      last_set = set_ptr;
      kernel = set.crc64_xz;
    }
    crc = kernel(crc, buf);
  }

  crc ^ !0
}

/// Compute CRC-64/NVME checksum of multiple buffers treated as one concatenated stream.
#[inline]
pub fn crc64_nvme_vectored(bufs: &[&[u8]]) -> u64 {
  let table = active_table();
  let mut crc: u64 = !0;
  let mut last_set: *const KernelSet = core::ptr::null();
  let mut kernel = table.xs.crc64_nvme;

  for &buf in bufs {
    if buf.is_empty() {
      continue;
    }
    let set = table.select_set(buf.len());
    let set_ptr: *const KernelSet = core::ptr::from_ref(set);
    if set_ptr != last_set {
      last_set = set_ptr;
      kernel = set.crc64_nvme;
    }
    crc = kernel(crc, buf);
  }

  crc ^ !0
}

/// Compute CRC-32 (IEEE) checksum of multiple buffers treated as one concatenated stream.
#[inline]
pub fn crc32_ieee_vectored(bufs: &[&[u8]]) -> u32 {
  let table = active_table();
  let mut crc: u32 = !0;
  let mut last_set: *const KernelSet = core::ptr::null();
  let mut kernel = table.xs.crc32_ieee;

  for &buf in bufs {
    if buf.is_empty() {
      continue;
    }
    let set = table.select_set(buf.len());
    let set_ptr: *const KernelSet = core::ptr::from_ref(set);
    if set_ptr != last_set {
      last_set = set_ptr;
      kernel = set.crc32_ieee;
    }
    crc = kernel(crc, buf);
  }

  crc ^ !0
}

/// Compute CRC-32C (Castagnoli) checksum of multiple buffers treated as one concatenated stream.
#[inline]
pub fn crc32c_vectored(bufs: &[&[u8]]) -> u32 {
  let table = active_table();
  let mut crc: u32 = !0;
  let mut last_set: *const KernelSet = core::ptr::null();
  let mut kernel = table.xs.crc32c;

  for &buf in bufs {
    if buf.is_empty() {
      continue;
    }
    let set = table.select_set(buf.len());
    let set_ptr: *const KernelSet = core::ptr::from_ref(set);
    if set_ptr != last_set {
      last_set = set_ptr;
      kernel = set.crc32c;
    }
    crc = kernel(crc, buf);
  }

  crc ^ !0
}

/// Compute CRC-16/CCITT checksum of multiple buffers treated as one concatenated stream.
#[inline]
pub fn crc16_ccitt_vectored(bufs: &[&[u8]]) -> u16 {
  let table = active_table();
  let mut crc: u16 = 0xFFFF;
  let mut last_set: *const KernelSet = core::ptr::null();
  let mut kernel = table.xs.crc16_ccitt;

  for &buf in bufs {
    if buf.is_empty() {
      continue;
    }
    let set = table.select_set(buf.len());
    let set_ptr: *const KernelSet = core::ptr::from_ref(set);
    if set_ptr != last_set {
      last_set = set_ptr;
      kernel = set.crc16_ccitt;
    }
    crc = kernel(crc, buf);
  }

  crc ^ 0xFFFF
}

/// Compute CRC-16/IBM checksum of multiple buffers treated as one concatenated stream.
#[inline]
pub fn crc16_ibm_vectored(bufs: &[&[u8]]) -> u16 {
  let table = active_table();
  let mut crc: u16 = 0;
  let mut last_set: *const KernelSet = core::ptr::null();
  let mut kernel = table.xs.crc16_ibm;

  for &buf in bufs {
    if buf.is_empty() {
      continue;
    }
    let set = table.select_set(buf.len());
    let set_ptr: *const KernelSet = core::ptr::from_ref(set);
    if set_ptr != last_set {
      last_set = set_ptr;
      kernel = set.crc16_ibm;
    }
    crc = kernel(crc, buf);
  }

  crc
}

/// Compute CRC-24/OpenPGP checksum of multiple buffers treated as one concatenated stream.
#[inline]
pub fn crc24_openpgp_vectored(bufs: &[&[u8]]) -> u32 {
  const INIT: u32 = 0x00B7_04CE;
  const MASK: u32 = 0x00FF_FFFF;
  let table = active_table();
  let mut crc: u32 = INIT;
  let mut last_set: *const KernelSet = core::ptr::null();
  let mut kernel = table.xs.crc24_openpgp;

  for &buf in bufs {
    if buf.is_empty() {
      continue;
    }
    let set = table.select_set(buf.len());
    let set_ptr: *const KernelSet = core::ptr::from_ref(set);
    if set_ptr != last_set {
      last_set = set_ptr;
      kernel = set.crc24_openpgp;
    }
    crc = kernel(crc, buf);
  }

  crc & MASK
}

/// `std::io::IoSlice` versions of the vectored one-shot APIs.
#[cfg(feature = "std")]
pub mod std_io {
  use super::*;

  /// Compute CRC-64/XZ checksum of `IoSlice` buffers treated as one concatenated stream.
  #[inline]
  pub fn crc64_xz_io_slices(bufs: &[std::io::IoSlice<'_>]) -> u64 {
    let table = active_table();
    let mut crc: u64 = !0;
    let mut last_set: *const KernelSet = core::ptr::null();
    let mut kernel = table.xs.crc64_xz;

    for buf in bufs {
      if buf.is_empty() {
        continue;
      }
      let set = table.select_set(buf.len());
      let set_ptr: *const KernelSet = core::ptr::from_ref(set);
      if set_ptr != last_set {
        last_set = set_ptr;
        kernel = set.crc64_xz;
      }
      crc = kernel(crc, buf);
    }

    crc ^ !0
  }

  /// Compute CRC-64/NVME checksum of `IoSlice` buffers treated as one concatenated stream.
  #[inline]
  pub fn crc64_nvme_io_slices(bufs: &[std::io::IoSlice<'_>]) -> u64 {
    let table = active_table();
    let mut crc: u64 = !0;
    let mut last_set: *const KernelSet = core::ptr::null();
    let mut kernel = table.xs.crc64_nvme;

    for buf in bufs {
      if buf.is_empty() {
        continue;
      }
      let set = table.select_set(buf.len());
      let set_ptr: *const KernelSet = core::ptr::from_ref(set);
      if set_ptr != last_set {
        last_set = set_ptr;
        kernel = set.crc64_nvme;
      }
      crc = kernel(crc, buf);
    }

    crc ^ !0
  }

  /// Compute CRC-32 (IEEE) checksum of `IoSlice` buffers treated as one concatenated stream.
  #[inline]
  pub fn crc32_ieee_io_slices(bufs: &[std::io::IoSlice<'_>]) -> u32 {
    let table = active_table();
    let mut crc: u32 = !0;
    let mut last_set: *const KernelSet = core::ptr::null();
    let mut kernel = table.xs.crc32_ieee;

    for buf in bufs {
      if buf.is_empty() {
        continue;
      }
      let set = table.select_set(buf.len());
      let set_ptr: *const KernelSet = core::ptr::from_ref(set);
      if set_ptr != last_set {
        last_set = set_ptr;
        kernel = set.crc32_ieee;
      }
      crc = kernel(crc, buf);
    }

    crc ^ !0
  }

  /// Compute CRC-32C checksum of `IoSlice` buffers treated as one concatenated stream.
  #[inline]
  pub fn crc32c_io_slices(bufs: &[std::io::IoSlice<'_>]) -> u32 {
    let table = active_table();
    let mut crc: u32 = !0;
    let mut last_set: *const KernelSet = core::ptr::null();
    let mut kernel = table.xs.crc32c;

    for buf in bufs {
      if buf.is_empty() {
        continue;
      }
      let set = table.select_set(buf.len());
      let set_ptr: *const KernelSet = core::ptr::from_ref(set);
      if set_ptr != last_set {
        last_set = set_ptr;
        kernel = set.crc32c;
      }
      crc = kernel(crc, buf);
    }

    crc ^ !0
  }

  /// Compute CRC-16/CCITT checksum of `IoSlice` buffers treated as one concatenated stream.
  #[inline]
  pub fn crc16_ccitt_io_slices(bufs: &[std::io::IoSlice<'_>]) -> u16 {
    let table = active_table();
    let mut crc: u16 = 0xFFFF;
    let mut last_set: *const KernelSet = core::ptr::null();
    let mut kernel = table.xs.crc16_ccitt;

    for buf in bufs {
      if buf.is_empty() {
        continue;
      }
      let set = table.select_set(buf.len());
      let set_ptr: *const KernelSet = core::ptr::from_ref(set);
      if set_ptr != last_set {
        last_set = set_ptr;
        kernel = set.crc16_ccitt;
      }
      crc = kernel(crc, buf);
    }

    crc ^ 0xFFFF
  }

  /// Compute CRC-16/IBM checksum of `IoSlice` buffers treated as one concatenated stream.
  #[inline]
  pub fn crc16_ibm_io_slices(bufs: &[std::io::IoSlice<'_>]) -> u16 {
    let table = active_table();
    let mut crc: u16 = 0;
    let mut last_set: *const KernelSet = core::ptr::null();
    let mut kernel = table.xs.crc16_ibm;

    for buf in bufs {
      if buf.is_empty() {
        continue;
      }
      let set = table.select_set(buf.len());
      let set_ptr: *const KernelSet = core::ptr::from_ref(set);
      if set_ptr != last_set {
        last_set = set_ptr;
        kernel = set.crc16_ibm;
      }
      crc = kernel(crc, buf);
    }

    crc
  }

  /// Compute CRC-24/OpenPGP checksum of `IoSlice` buffers treated as one concatenated stream.
  #[inline]
  pub fn crc24_openpgp_io_slices(bufs: &[std::io::IoSlice<'_>]) -> u32 {
    const INIT: u32 = 0x00B7_04CE;
    const MASK: u32 = 0x00FF_FFFF;
    let table = active_table();
    let mut crc: u32 = INIT;
    let mut last_set: *const KernelSet = core::ptr::null();
    let mut kernel = table.xs.crc24_openpgp;

    for buf in bufs {
      if buf.is_empty() {
        continue;
      }
      let set = table.select_set(buf.len());
      let set_ptr: *const KernelSet = core::ptr::from_ref(set);
      if set_ptr != last_set {
        last_set = set_ptr;
        kernel = set.crc24_openpgp;
      }
      crc = kernel(crc, buf);
    }

    crc & MASK
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data Structures
// ─────────────────────────────────────────────────────────────────────────────

/// All kernel function pointers for one size class.
///
/// Contains the optimal kernel for each CRC variant at a specific buffer size,
/// along with human-readable kernel names for introspection.
#[derive(Clone, Copy)]
pub struct KernelSet {
  // CRC-16
  pub crc16_ccitt: Crc16Fn,
  pub crc16_ccitt_name: &'static str,
  pub crc16_ibm: Crc16Fn,
  pub crc16_ibm_name: &'static str,
  // CRC-24
  pub crc24_openpgp: Crc24Fn,
  pub crc24_openpgp_name: &'static str,
  // CRC-32
  pub crc32_ieee: Crc32Fn,
  pub crc32_ieee_name: &'static str,
  pub crc32c: Crc32Fn,
  pub crc32c_name: &'static str,
  // CRC-64
  pub crc64_xz: Crc64Fn,
  pub crc64_xz_name: &'static str,
  pub crc64_nvme: Crc64Fn,
  pub crc64_nvme_name: &'static str,
}

/// Complete kernel table for one platform.
///
/// Contains pre-selected optimal kernels for each (variant, size_class) pair.
/// Size class boundaries define when to transition between kernel tiers.
#[derive(Clone, Copy)]
pub struct KernelTable {
  /// Required capabilities for *all* kernels referenced by this table.
  ///
  /// `select_table()` must only return a table when `caps.has(requires)` holds.
  pub requires: Caps,

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

  /// Returns `true` if this table uses hardware-accelerated kernels.
  ///
  /// A table is hardware-accelerated if it requires any CPU capabilities
  /// beyond the baseline (i.e., `requires != Caps::NONE`).
  #[inline]
  pub const fn is_hardware_accelerated(&self) -> bool {
    !self.requires.is_empty()
  }
}

/// Returns `true` if the current platform uses hardware-accelerated CRC kernels.
///
/// This is a convenience function that checks whether the active dispatch table
/// requires any SIMD or hardware CRC capabilities.
///
/// # Examples
///
/// ```rust
/// use checksum::dispatch::is_hardware_accelerated;
///
/// if is_hardware_accelerated() {
///   println!("CRC operations are hardware-accelerated on this platform");
/// } else {
///   println!("Using portable (table-based) CRC implementations");
/// }
/// ```
#[inline]
pub fn is_hardware_accelerated() -> bool {
  active_table().is_hardware_accelerated()
}

// ─────────────────────────────────────────────────────────────────────────────
// Platform Table Selection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
#[must_use]
fn prefer_intel_icl_tables(caps: Caps, kind: TuneKind) -> bool {
  // `platform` can report some "AVX-512 Intel server" runners as IntelSpr even
  // when AMX is absent (e.g. ICL-SP class). These CPUs have meaningfully
  // different optimal CRC kernel selection than SPR/EMR-class.
  kind == TuneKind::IntelSpr
    && !caps.has(platform::caps::x86::AMX_TILE)
    && !caps.has(platform::caps::x86::AMX_INT8)
    && !caps.has(platform::caps::x86::AMX_BF16)
    && !caps.has(platform::caps::x86::AMX_FP16)
    && !caps.has(platform::caps::x86::AMX_COMPLEX)
}

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
  if let Some(table) = exact_match(tune_kind, caps) {
    return table;
  }

  // 2. Family match (inferred from similar hardware)
  if let Some(table) = family_match(tune_kind, caps) {
    return table;
  }

  // 3. Capability match (conservative defaults)
  if let Some(table) = capability_match(caps) {
    debug_assert!(caps.has(table.requires), "capability_match returned an unsafe table");
    if caps.has(table.requires) {
      return table;
    }
  }

  // 4. Portable fallback
  &PORTABLE_TABLE
}

#[inline]
fn exact_match(tune_kind: TuneKind, caps: Caps) -> Option<&'static KernelTable> {
  let table: Option<&'static KernelTable> = match tune_kind {
    #[cfg(target_arch = "aarch64")]
    TuneKind::AppleM1M3 => Some(&APPLE_M1M3_TABLE),
    #[cfg(target_arch = "aarch64")]
    TuneKind::Graviton2 => Some(&GRAVITON2_TABLE),
    #[cfg(target_arch = "aarch64")]
    TuneKind::Graviton3 => Some(&GRAVITON3_TABLE),
    #[cfg(target_arch = "x86_64")]
    TuneKind::Zen4 => Some(&ZEN4_TABLE),
    #[cfg(target_arch = "x86_64")]
    TuneKind::Zen5 => Some(&ZEN5_TABLE),
    #[cfg(target_arch = "x86_64")]
    TuneKind::IntelSpr => Some(if prefer_intel_icl_tables(caps, tune_kind) {
      &INTEL_ICL_TABLE
    } else {
      &INTEL_SPR_TABLE
    }),
    #[cfg(target_arch = "s390x")]
    TuneKind::Z13 => Some(&S390X_Z13_TABLE),
    #[cfg(target_arch = "s390x")]
    TuneKind::Z14 => Some(&S390X_Z14_TABLE),
    #[cfg(target_arch = "s390x")]
    TuneKind::Z15 => Some(&S390X_Z15_TABLE),
    #[cfg(target_arch = "powerpc64")]
    TuneKind::Power8 => Some(&POWER8_TABLE),
    #[cfg(target_arch = "powerpc64")]
    TuneKind::Power9 => Some(&POWER9_TABLE),
    #[cfg(target_arch = "powerpc64")]
    TuneKind::Power10 => Some(&POWER10_TABLE),
    _ => None,
  };
  let table = table?;
  if caps.has(table.requires) { Some(table) } else { None }
}

#[inline]
fn family_match(tune_kind: TuneKind, caps: Caps) -> Option<&'static KernelTable> {
  let table: Option<&'static KernelTable> = match tune_kind {
    // Apple Silicon family → use M1-M3 data
    #[cfg(target_arch = "aarch64")]
    TuneKind::AppleM4 | TuneKind::AppleM5 => Some(&APPLE_M1M3_TABLE),

    // AWS Graviton / ARM Neoverse family → use Graviton3 data (better EOR3 tuning)
    #[cfg(target_arch = "aarch64")]
    TuneKind::Graviton4 | TuneKind::Graviton5 => Some(&GRAVITON3_TABLE),
    #[cfg(target_arch = "aarch64")]
    TuneKind::NeoverseN2 | TuneKind::NeoverseN3 | TuneKind::NeoverseV3 => Some(&GRAVITON3_TABLE),
    #[cfg(target_arch = "aarch64")]
    TuneKind::NvidiaGrace | TuneKind::AmpereAltra => Some(&GRAVITON3_TABLE),

    // AMD Zen family → prefer Zen5 table when available
    #[cfg(target_arch = "x86_64")]
    TuneKind::Zen5 | TuneKind::Zen5c => Some(&ZEN5_TABLE),

    // Intel family → prefer Intel SPR table when available
    #[cfg(target_arch = "x86_64")]
    TuneKind::IntelSpr | TuneKind::IntelGnr => Some(if prefer_intel_icl_tables(caps, tune_kind) {
      &INTEL_ICL_TABLE
    } else {
      &INTEL_SPR_TABLE
    }),
    #[cfg(target_arch = "x86_64")]
    TuneKind::IntelIcl => Some(&INTEL_ICL_TABLE),

    _ => None,
  };
  let table = table?;
  if caps.has(table.requires) { Some(table) } else { None }
}

#[inline]
fn capability_match(caps: Caps) -> Option<&'static KernelTable> {
  let _ = caps;

  #[cfg(target_arch = "aarch64")]
  {
    use platform::caps::aarch64::{CRC_READY, PMULL_EOR3_READY, PMULL_READY};

    // PMULL + SHA3 (EOR3) + CRC extension → use EOR3-enabled table
    if caps.has(CRC_READY) && caps.has(PMULL_EOR3_READY) {
      return Some(&GENERIC_ARM_PMULL_EOR3_TABLE);
    }
    // PMULL + CRC extension (no EOR3) → safe PMULL-only table
    if caps.has(CRC_READY) && caps.has(PMULL_READY) {
      return Some(&GENERIC_ARM_PMULL_TABLE);
    }
    // PMULL only (no CRC extension) → accelerate CRC-16/24/64, keep CRC-32 portable
    if caps.has(PMULL_READY) {
      return Some(&GENERIC_ARM_PMULL_NO_CRC_TABLE);
    }
    // CRC extension only → accelerate CRC-32/32C, keep others portable
    if caps.has(CRC_READY) {
      return Some(&GENERIC_ARM_CRC_ONLY_TABLE);
    }
  }

  #[cfg(target_arch = "x86_64")]
  {
    use platform::caps::x86::{CRC32C_READY, PCLMUL_READY, VPCLMUL_READY};

    // VPCLMUL → prefer VPCLMUL tables (CRC32C uses hwcrc/fusion only if SSE4.2 exists)
    if caps.has(VPCLMUL_READY) && caps.has(PCLMUL_READY) && caps.has(CRC32C_READY) {
      return Some(&GENERIC_X86_VPCLMUL_TABLE);
    }
    if caps.has(VPCLMUL_READY) && caps.has(PCLMUL_READY) {
      return Some(&GENERIC_X86_VPCLMUL_NO_CRC32C_TABLE);
    }

    // PCLMUL → use PCLMUL tables (CRC32C uses hwcrc/fusion only if SSE4.2 exists)
    if caps.has(PCLMUL_READY) && caps.has(CRC32C_READY) {
      return Some(&GENERIC_X86_PCLMUL_TABLE);
    }
    if caps.has(PCLMUL_READY) {
      return Some(&GENERIC_X86_PCLMUL_NO_CRC32C_TABLE);
    }

    // SSE4.2 only: accelerate CRC32C, keep other variants portable.
    if caps.has(CRC32C_READY) {
      return Some(&GENERIC_X86_CRC32C_ONLY_TABLE);
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    use platform::caps::s390x::{Z13_READY, Z14_READY, Z15_READY, Z16_READY};
    if caps.has(Z16_READY) {
      return Some(&S390X_Z15_TABLE);
    }
    if caps.has(Z15_READY) {
      return Some(&S390X_Z15_TABLE);
    }
    if caps.has(Z14_READY) {
      return Some(&S390X_Z14_TABLE);
    }
    if caps.has(Z13_READY) {
      return Some(&S390X_Z13_TABLE);
    }
  }

  #[cfg(target_arch = "powerpc64")]
  {
    use platform::caps::power::{POWER9_READY, POWER10_READY, VPMSUM_READY};
    if caps.has(POWER10_READY) {
      return Some(&POWER10_TABLE);
    }
    if caps.has(POWER9_READY) {
      return Some(&POWER9_TABLE);
    }
    if caps.has(VPMSUM_READY) {
      return Some(&POWER8_TABLE);
    }
  }

  #[cfg(target_arch = "riscv64")]
  {
    use platform::caps::riscv::{V, ZBC, ZVBC};
    let v_zvbc = V.union(ZVBC);
    if caps.has(v_zvbc) {
      return Some(&RISCV64_ZVBC_TABLE);
    }
    if caps.has(ZBC) {
      return Some(&RISCV64_ZBC_TABLE);
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
const PORTABLE_SET: KernelSet = KernelSet {
  crc16_ccitt: crate::crc16::portable::crc16_ccitt_slice8,
  crc16_ccitt_name: "portable/slice8",
  crc16_ibm: crate::crc16::portable::crc16_ibm_slice8,
  crc16_ibm_name: "portable/slice8",
  crc24_openpgp: crate::crc24::portable::crc24_openpgp_slice8,
  crc24_openpgp_name: "portable/slice8",
  crc32_ieee: crate::crc32::portable::crc32_slice16_ieee,
  crc32_ieee_name: "portable/slice16",
  crc32c: crate::crc32::portable::crc32c_slice16,
  crc32c_name: "portable/slice16",
  crc64_xz: crate::crc64::portable::crc64_slice16_xz,
  crc64_xz_name: "portable/slice16",
  crc64_nvme: crate::crc64::portable::crc64_slice16_nvme,
  crc64_nvme_name: "portable/slice16",
};

pub static PORTABLE_TABLE: KernelTable = KernelTable {
  requires: Caps::NONE,
  boundaries: [64, 256, 4096],
  xs: PORTABLE_SET,
  s: PORTABLE_SET,
  m: PORTABLE_SET,
  l: PORTABLE_SET,
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
  // AppleM1M3 Table
  //
  // Generated by rscrypto-tune. Do not edit manually.
  // ───────────────────────────────────────────────────────────────────────────
  // AppleM1M3 Table
  //
  // Generated by rscrypto-tune. Do not edit manually.
  // ───────────────────────────────────────────────────────────────────────────
  pub static APPLE_M1M3_TABLE: KernelTable = KernelTable {
    requires: platform::caps::aarch64::CRC_READY
      .union(platform::caps::aarch64::PMULL_EOR3_READY)
      .union(platform::caps::aarch64::PMULL_READY),
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL_SMALL_KERNEL,
      crc16_ccitt_name: "aarch64/pmull-small",
      crc16_ibm: crc16_k::IBM_PMULL_SMALL_KERNEL,
      crc16_ibm_name: "aarch64/pmull-small",
      crc24_openpgp: crc24_k::OPENPGP_PMULL_SMALL_KERNEL,
      crc24_openpgp_name: "aarch64/pmull-small",
      crc32_ieee: crc32_k::CRC32_PMULL_EOR3[1],
      crc32_ieee_name: "aarch64/pmull-eor3-v9s3x2e-s3-2way",
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL,
      crc32c_name: "aarch64/pmull-small",
      crc64_xz: crc64_k::XZ_PMULL_SMALL,
      crc64_xz_name: "aarch64/pmull-small",
      crc64_nvme: crc64_k::NVME_PMULL_SMALL,
      crc64_nvme_name: "aarch64/pmull-small",
    },

    s: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL_SMALL_KERNEL,
      crc16_ccitt_name: "aarch64/pmull-small",
      crc16_ibm: crc16_k::IBM_PMULL_SMALL_KERNEL,
      crc16_ibm_name: "aarch64/pmull-small",
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0],
      crc24_openpgp_name: "aarch64/pmull",
      crc32_ieee: crc32_k::CRC32_PMULL_SMALL_KERNEL,
      crc32_ieee_name: "aarch64/pmull-small",
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL,
      crc32c_name: "aarch64/pmull-small",
      crc64_xz: crc64_k::XZ_PMULL_EOR3[0],
      crc64_xz_name: "aarch64/pmull-eor3",
      crc64_nvme: crc64_k::NVME_PMULL[0],
      crc64_nvme_name: "aarch64/pmull",
    },

    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL[0],
      crc16_ccitt_name: "aarch64/pmull",
      crc16_ibm: crc16_k::IBM_PMULL[0],
      crc16_ibm_name: "aarch64/pmull",
      crc24_openpgp: crc24_k::OPENPGP_PMULL[1],
      crc24_openpgp_name: "aarch64/pmull-2way",
      crc32_ieee: crc32_k::CRC32_PMULL_SMALL_KERNEL,
      crc32_ieee_name: "aarch64/pmull-small",
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL,
      crc32c_name: "aarch64/pmull-small",
      crc64_xz: crc64_k::XZ_PMULL[0],
      crc64_xz_name: "aarch64/pmull",
      crc64_nvme: crc64_k::NVME_PMULL_EOR3[0],
      crc64_nvme_name: "aarch64/pmull-eor3",
    },

    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL[1],
      crc16_ccitt_name: "aarch64/pmull-2way",
      crc16_ibm: crc16_k::IBM_PMULL[1],
      crc16_ibm_name: "aarch64/pmull-2way",
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0],
      crc24_openpgp_name: "aarch64/pmull",
      crc32_ieee: crc32_k::CRC32_PMULL_EOR3[0],
      crc32_ieee_name: "aarch64/pmull-eor3-v9s3x2e-s3",
      crc32c: crc32_k::CRC32C_PMULL_EOR3[0],
      crc32c_name: "aarch64/pmull-eor3-v9s3x2e-s3",
      crc64_xz: crc64_k::XZ_PMULL_EOR3[1],
      crc64_xz_name: "aarch64/pmull-eor3-2way",
      crc64_nvme: crc64_k::NVME_PMULL[1],
      crc64_nvme_name: "aarch64/pmull-2way",
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
    requires: platform::caps::aarch64::CRC_READY.union(platform::caps::aarch64::PMULL_EOR3_READY),
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL_SMALL_KERNEL,
      crc16_ccitt_name: "aarch64/pmull-small",
      crc16_ibm: crc16_k::IBM_PMULL_SMALL_KERNEL,
      crc16_ibm_name: "aarch64/pmull-small",
      crc24_openpgp: crc24_k::OPENPGP_PMULL_SMALL_KERNEL,
      crc24_openpgp_name: "aarch64/pmull-small",
      crc32_ieee: crc32_k::CRC32_PMULL_SMALL_KERNEL, // pmull-small beats hwcrc @ 9.53 GiB/s
      crc32_ieee_name: "aarch64/pmull-small",
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL, // pmull-small beats hwcrc @ 9.54 GiB/s
      crc32c_name: "aarch64/pmull-small",
      crc64_xz: crc64_k::XZ_PMULL_SMALL,
      crc64_xz_name: "aarch64/pmull-small",
      crc64_nvme: crc64_k::NVME_PMULL_SMALL,
      crc64_nvme_name: "aarch64/pmull-small",
    },

    s: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL_SMALL_KERNEL,
      crc16_ccitt_name: "aarch64/pmull-small",
      crc16_ibm: crc16_k::IBM_PMULL_SMALL_KERNEL,
      crc16_ibm_name: "aarch64/pmull-small",
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0], // 1-way @ 11.08 GiB/s
      crc24_openpgp_name: "aarch64/pmull",
      crc32_ieee: crc32_k::CRC32_PMULL_SMALL_KERNEL, // pmull-small @ 13.30 GiB/s
      crc32_ieee_name: "aarch64/pmull-small",
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL, // pmull-small @ 13.30 GiB/s
      crc32c_name: "aarch64/pmull-small",
      crc64_xz: crc64_k::XZ_PMULL[0], // pmull @ 13.60 GiB/s
      crc64_xz_name: "aarch64/pmull",
      crc64_nvme: crc64_k::NVME_PMULL[0], // pmull @ 13.58 GiB/s (bench: pmull beats pmull-eor3)
      crc64_nvme_name: "aarch64/pmull",
    },

    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL[0], // 1-way @ 29.19 GiB/s
      crc16_ccitt_name: "aarch64/pmull",
      crc16_ibm: crc16_k::IBM_PMULL[0], // 1-way @ 29.20 GiB/s
      crc16_ibm_name: "aarch64/pmull",
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0], // 1-way @ 19.95 GiB/s
      crc24_openpgp_name: "aarch64/pmull",
      crc32_ieee: crc32_k::CRC32_PMULL_SMALL_KERNEL, // pmull-small @ 28.62 GiB/s
      crc32_ieee_name: "aarch64/pmull-small",
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL, // pmull-small @ 28.68 GiB/s
      crc32c_name: "aarch64/pmull-small",
      crc64_xz: crc64_k::XZ_PMULL[0], // pmull @ 30.39 GiB/s (pmull-eor3 slower here)
      crc64_xz_name: "aarch64/pmull",
      crc64_nvme: crc64_k::NVME_PMULL[0], // pmull @ 30.45 GiB/s
      crc64_nvme_name: "aarch64/pmull",
    },

    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL[0], // 1-way @ 33.01 GiB/s
      crc16_ccitt_name: "aarch64/pmull",
      crc16_ibm: crc16_k::IBM_PMULL[0], // 1-way @ 32.78 GiB/s
      crc16_ibm_name: "aarch64/pmull",
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0], // 1-way @ 24.88 GiB/s
      crc24_openpgp_name: "aarch64/pmull",
      crc32_ieee: crc32_k::CRC32_PMULL_EOR3[0], // pmull-eor3-v9s3x2e-s3 @ 39.82 GiB/s
      crc32_ieee_name: "aarch64/pmull-eor3-v9s3x2e-s3",
      crc32c: crc32_k::CRC32C_PMULL_EOR3[0], // pmull-eor3-v9s3x2e-s3 @ 39.93 GiB/s
      crc32c_name: "aarch64/pmull-eor3-v9s3x2e-s3",
      crc64_xz: crc64_k::XZ_PMULL_EOR3[0], // pmull-eor3 @ 33.07 GiB/s
      crc64_xz_name: "aarch64/pmull-eor3",
      crc64_nvme: crc64_k::NVME_PMULL[0], // pmull @ 33.08 GiB/s
      crc64_nvme_name: "aarch64/pmull",
    },
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Graviton3 Table
  //
  // Benchmark source: `crates/checksum/bench_baseline/linux_arm64_graviton3_kernels.txt`
  // Features: PMULL + SHA3/EOR3
  // Peak throughputs: CRC-16 ~38 GiB/s, CRC-32 ~46 GiB/s, CRC-64 ~38 GiB/s
  //
  // Key differences vs Graviton2:
  // - Higher throughput (~25% faster across the board)
  // - Different optimal kernel choices for CRC16@s and CRC64/NVME
  // ───────────────────────────────────────────────────────────────────────────
  pub static GRAVITON3_TABLE: KernelTable = KernelTable {
    requires: platform::caps::aarch64::CRC_READY.union(platform::caps::aarch64::PMULL_EOR3_READY),
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL_SMALL_KERNEL, // pmull-small @ 8.01 GiB/s
      crc16_ccitt_name: "aarch64/pmull-small",
      crc16_ibm: crc16_k::IBM_PMULL_SMALL_KERNEL, // pmull-small @ 7.85 GiB/s
      crc16_ibm_name: "aarch64/pmull-small",
      crc24_openpgp: crc24_k::OPENPGP_PMULL_SMALL_KERNEL, // pmull-small @ 6.81 GiB/s
      crc24_openpgp_name: "aarch64/pmull-small",
      crc32_ieee: crc32_k::CRC32_PMULL_SMALL_KERNEL, // pmull-small @ 10.30 GiB/s
      crc32_ieee_name: "aarch64/pmull-small",
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL, // pmull-small @ 12.49 GiB/s
      crc32c_name: "aarch64/pmull-small",
      crc64_xz: crc64_k::XZ_PMULL_SMALL, // pmull-small @ 7.06 GiB/s
      crc64_xz_name: "aarch64/pmull-small",
      crc64_nvme: crc64_k::NVME_PMULL_SMALL, // pmull-small @ 7.06 GiB/s
      crc64_nvme_name: "aarch64/pmull-small",
    },

    s: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL[0], // pmull @ 11.97 GiB/s (G3: pmull beats pmull-small)
      crc16_ccitt_name: "aarch64/pmull",
      crc16_ibm: crc16_k::IBM_PMULL[0], // pmull @ 12.01 GiB/s (G3: pmull beats pmull-small)
      crc16_ibm_name: "aarch64/pmull",
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0], // pmull @ 14.12 GiB/s
      crc24_openpgp_name: "aarch64/pmull",
      crc32_ieee: crc32_k::CRC32_PMULL_SMALL_KERNEL, // pmull-small @ 8.66 GiB/s
      crc32_ieee_name: "aarch64/pmull-small",
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL, // pmull-small @ 8.79 GiB/s
      crc32c_name: "aarch64/pmull-small",
      crc64_xz: crc64_k::XZ_PMULL[0], // pmull @ 18.05 GiB/s
      crc64_xz_name: "aarch64/pmull",
      crc64_nvme: crc64_k::NVME_PMULL[0], // pmull @ 18.01 GiB/s
      crc64_nvme_name: "aarch64/pmull",
    },

    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL[0], // pmull @ 32.80 GiB/s
      crc16_ccitt_name: "aarch64/pmull",
      crc16_ibm: crc16_k::IBM_PMULL[0], // pmull @ 32.79 GiB/s
      crc16_ibm_name: "aarch64/pmull",
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0], // pmull @ 32.78 GiB/s
      crc24_openpgp_name: "aarch64/pmull",
      crc32_ieee: crc32_k::CRC32_PMULL_SMALL_KERNEL, // pmull-small @ 31.76 GiB/s
      crc32_ieee_name: "aarch64/pmull-small",
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL, // pmull-small @ 32.26 GiB/s
      crc32c_name: "aarch64/pmull-small",
      crc64_xz: crc64_k::XZ_PMULL[0], // pmull @ 35.34 GiB/s
      crc64_xz_name: "aarch64/pmull",
      crc64_nvme: crc64_k::NVME_PMULL[1], // pmull-2way @ 34.60 GiB/s (G3: 2way beats 1way)
      crc64_nvme_name: "aarch64/pmull-2way",
    },

    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL[0], // pmull @ 37.89 GiB/s
      crc16_ccitt_name: "aarch64/pmull",
      crc16_ibm: crc16_k::IBM_PMULL[0], // pmull @ 37.92 GiB/s
      crc16_ibm_name: "aarch64/pmull",
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0], // pmull @ 37.90 GiB/s
      crc24_openpgp_name: "aarch64/pmull",
      crc32_ieee: crc32_k::CRC32_PMULL_EOR3[0], // pmull-eor3-v9s3x2e-s3 @ 46.29 GiB/s
      crc32_ieee_name: "aarch64/pmull-eor3-v9s3x2e-s3",
      crc32c: crc32_k::CRC32C_PMULL_EOR3[0], // pmull-eor3-v9s3x2e-s3 @ 46.08 GiB/s
      crc32c_name: "aarch64/pmull-eor3-v9s3x2e-s3",
      crc64_xz: crc64_k::XZ_PMULL_EOR3[0], // pmull-eor3 @ 38.04 GiB/s
      crc64_xz_name: "aarch64/pmull-eor3",
      crc64_nvme: crc64_k::NVME_PMULL_EOR3[0], // pmull-eor3 @ 38.05 GiB/s (G3: eor3 beats pmull)
      crc64_nvme_name: "aarch64/pmull-eor3",
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
  // For unknown ARM platforms with CRC + PMULL but *without* SHA3/EOR3.
  // ───────────────────────────────────────────────────────────────────────────
  pub static GENERIC_ARM_PMULL_TABLE: KernelTable = KernelTable {
    requires: platform::caps::aarch64::CRC_READY.union(platform::caps::aarch64::PMULL_READY),
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL_SMALL_KERNEL,
      crc16_ccitt_name: "aarch64/pmull-small",
      crc16_ibm: crc16_k::IBM_PMULL_SMALL_KERNEL,
      crc16_ibm_name: "aarch64/pmull-small",
      crc24_openpgp: crc24_k::OPENPGP_PMULL_SMALL_KERNEL,
      crc24_openpgp_name: "aarch64/pmull-small",
      crc32_ieee: crc32_k::CRC32_PMULL_SMALL_KERNEL,
      crc32_ieee_name: "aarch64/pmull-small",
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL,
      crc32c_name: "aarch64/pmull-small",
      crc64_xz: crc64_k::XZ_PMULL_SMALL,
      crc64_xz_name: "aarch64/pmull-small",
      crc64_nvme: crc64_k::NVME_PMULL_SMALL,
      crc64_nvme_name: "aarch64/pmull-small",
    },

    s: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL_SMALL_KERNEL,
      crc16_ccitt_name: "aarch64/pmull-small",
      crc16_ibm: crc16_k::IBM_PMULL_SMALL_KERNEL,
      crc16_ibm_name: "aarch64/pmull-small",
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0],
      crc24_openpgp_name: "aarch64/pmull",
      crc32_ieee: crc32_k::CRC32_PMULL_SMALL_KERNEL,
      crc32_ieee_name: "aarch64/pmull-small",
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL,
      crc32c_name: "aarch64/pmull-small",
      crc64_xz: crc64_k::XZ_PMULL[0],
      crc64_xz_name: "aarch64/pmull",
      crc64_nvme: crc64_k::NVME_PMULL[0],
      crc64_nvme_name: "aarch64/pmull",
    },

    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL[0],
      crc16_ccitt_name: "aarch64/pmull",
      crc16_ibm: crc16_k::IBM_PMULL[0],
      crc16_ibm_name: "aarch64/pmull",
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0],
      crc24_openpgp_name: "aarch64/pmull",
      crc32_ieee: crc32_k::CRC32_PMULL_SMALL_KERNEL,
      crc32_ieee_name: "aarch64/pmull-small",
      crc32c: crc32_k::CRC32C_PMULL_SMALL_KERNEL,
      crc32c_name: "aarch64/pmull-small",
      crc64_xz: crc64_k::XZ_PMULL[0],
      crc64_xz_name: "aarch64/pmull",
      crc64_nvme: crc64_k::NVME_PMULL[0],
      crc64_nvme_name: "aarch64/pmull",
    },

    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL[0],
      crc16_ccitt_name: "aarch64/pmull",
      crc16_ibm: crc16_k::IBM_PMULL[0],
      crc16_ibm_name: "aarch64/pmull",
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0],
      crc24_openpgp_name: "aarch64/pmull",
      crc32_ieee: crc32_k::CRC32_PMULL[0],
      crc32_ieee_name: "aarch64/pmull-v9s3x2e-s3",
      crc32c: crc32_k::CRC32C_PMULL[0],
      crc32c_name: "aarch64/pmull-v9s3x2e-s3",
      crc64_xz: crc64_k::XZ_PMULL[0],
      crc64_xz_name: "aarch64/pmull",
      crc64_nvme: crc64_k::NVME_PMULL[0],
      crc64_nvme_name: "aarch64/pmull",
    },
  };

  /// PMULL-only table for platforms without the CRC extension.
  pub static GENERIC_ARM_PMULL_NO_CRC_TABLE: KernelTable = KernelTable {
    requires: platform::caps::aarch64::PMULL_READY,
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL_SMALL_KERNEL,
      crc16_ccitt_name: "aarch64/pmull-small",
      crc16_ibm: crc16_k::IBM_PMULL_SMALL_KERNEL,
      crc16_ibm_name: "aarch64/pmull-small",
      crc24_openpgp: crc24_k::OPENPGP_PMULL_SMALL_KERNEL,
      crc24_openpgp_name: "aarch64/pmull-small",
      crc32_ieee: crate::crc32::portable::crc32_bytewise_ieee,
      crc32_ieee_name: crate::crc32::portable::BYTEWISE_KERNEL_NAME,
      crc32c: crate::crc32::portable::crc32c_bytewise,
      crc32c_name: crate::crc32::portable::BYTEWISE_KERNEL_NAME,
      crc64_xz: crc64_k::XZ_PMULL_SMALL,
      crc64_xz_name: "aarch64/pmull-small",
      crc64_nvme: crc64_k::NVME_PMULL_SMALL,
      crc64_nvme_name: "aarch64/pmull-small",
    },

    s: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL_SMALL_KERNEL,
      crc16_ccitt_name: "aarch64/pmull-small",
      crc16_ibm: crc16_k::IBM_PMULL_SMALL_KERNEL,
      crc16_ibm_name: "aarch64/pmull-small",
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0],
      crc24_openpgp_name: "aarch64/pmull",
      crc32_ieee: crate::crc32::portable::crc32_slice16_ieee,
      crc32_ieee_name: "portable/slice16",
      crc32c: crate::crc32::portable::crc32c_slice16,
      crc32c_name: "portable/slice16",
      crc64_xz: crc64_k::XZ_PMULL[0],
      crc64_xz_name: "aarch64/pmull",
      crc64_nvme: crc64_k::NVME_PMULL[0],
      crc64_nvme_name: "aarch64/pmull",
    },

    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL[0],
      crc16_ccitt_name: "aarch64/pmull",
      crc16_ibm: crc16_k::IBM_PMULL[0],
      crc16_ibm_name: "aarch64/pmull",
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0],
      crc24_openpgp_name: "aarch64/pmull",
      crc32_ieee: crate::crc32::portable::crc32_slice16_ieee,
      crc32_ieee_name: "portable/slice16",
      crc32c: crate::crc32::portable::crc32c_slice16,
      crc32c_name: "portable/slice16",
      crc64_xz: crc64_k::XZ_PMULL[0],
      crc64_xz_name: "aarch64/pmull",
      crc64_nvme: crc64_k::NVME_PMULL[0],
      crc64_nvme_name: "aarch64/pmull",
    },

    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL[0],
      crc16_ccitt_name: "aarch64/pmull",
      crc16_ibm: crc16_k::IBM_PMULL[0],
      crc16_ibm_name: "aarch64/pmull",
      crc24_openpgp: crc24_k::OPENPGP_PMULL[0],
      crc24_openpgp_name: "aarch64/pmull",
      crc32_ieee: crate::crc32::portable::crc32_slice16_ieee,
      crc32_ieee_name: "portable/slice16",
      crc32c: crate::crc32::portable::crc32c_slice16,
      crc32c_name: "portable/slice16",
      crc64_xz: crc64_k::XZ_PMULL[0],
      crc64_xz_name: "aarch64/pmull",
      crc64_nvme: crc64_k::NVME_PMULL[0],
      crc64_nvme_name: "aarch64/pmull",
    },
  };

  /// CRC-only table for platforms without PMULL.
  pub static GENERIC_ARM_CRC_ONLY_TABLE: KernelTable = KernelTable {
    requires: platform::caps::aarch64::CRC_READY,
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crate::crc16::portable::crc16_ccitt_slice8,
      crc16_ccitt_name: "portable/slice8",
      crc16_ibm: crate::crc16::portable::crc16_ibm_slice8,
      crc16_ibm_name: "portable/slice8",
      crc24_openpgp: crate::crc24::portable::crc24_openpgp_slice8,
      crc24_openpgp_name: "portable/slice8",
      crc32_ieee: crc32_k::CRC32_HWCRC[0],
      crc32_ieee_name: "aarch64/hwcrc",
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc32c_name: "aarch64/hwcrc",
      crc64_xz: crate::crc64::portable::crc64_slice16_xz,
      crc64_xz_name: "portable/slice16",
      crc64_nvme: crate::crc64::portable::crc64_slice16_nvme,
      crc64_nvme_name: "portable/slice16",
    },

    s: KernelSet {
      crc16_ccitt: crate::crc16::portable::crc16_ccitt_slice8,
      crc16_ccitt_name: "portable/slice8",
      crc16_ibm: crate::crc16::portable::crc16_ibm_slice8,
      crc16_ibm_name: "portable/slice8",
      crc24_openpgp: crate::crc24::portable::crc24_openpgp_slice8,
      crc24_openpgp_name: "portable/slice8",
      crc32_ieee: crc32_k::CRC32_HWCRC[0],
      crc32_ieee_name: "aarch64/hwcrc",
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc32c_name: "aarch64/hwcrc",
      crc64_xz: crate::crc64::portable::crc64_slice16_xz,
      crc64_xz_name: "portable/slice16",
      crc64_nvme: crate::crc64::portable::crc64_slice16_nvme,
      crc64_nvme_name: "portable/slice16",
    },

    m: KernelSet {
      crc16_ccitt: crate::crc16::portable::crc16_ccitt_slice8,
      crc16_ccitt_name: "portable/slice8",
      crc16_ibm: crate::crc16::portable::crc16_ibm_slice8,
      crc16_ibm_name: "portable/slice8",
      crc24_openpgp: crate::crc24::portable::crc24_openpgp_slice8,
      crc24_openpgp_name: "portable/slice8",
      crc32_ieee: crc32_k::CRC32_HWCRC[0],
      crc32_ieee_name: "aarch64/hwcrc",
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc32c_name: "aarch64/hwcrc",
      crc64_xz: crate::crc64::portable::crc64_slice16_xz,
      crc64_xz_name: "portable/slice16",
      crc64_nvme: crate::crc64::portable::crc64_slice16_nvme,
      crc64_nvme_name: "portable/slice16",
    },

    l: KernelSet {
      crc16_ccitt: crate::crc16::portable::crc16_ccitt_slice8,
      crc16_ccitt_name: "portable/slice8",
      crc16_ibm: crate::crc16::portable::crc16_ibm_slice8,
      crc16_ibm_name: "portable/slice8",
      crc24_openpgp: crate::crc24::portable::crc24_openpgp_slice8,
      crc24_openpgp_name: "portable/slice8",
      crc32_ieee: crc32_k::CRC32_HWCRC[0],
      crc32_ieee_name: "aarch64/hwcrc",
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc32c_name: "aarch64/hwcrc",
      crc64_xz: crate::crc64::portable::crc64_slice16_xz,
      crc64_xz_name: "portable/slice16",
      crc64_nvme: crate::crc64::portable::crc64_slice16_nvme,
      crc64_nvme_name: "portable/slice16",
    },
  };
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
  // Zen5 Table
  //
  // Benchmark source: `crates/checksum/bench_baseline/linux_x86-64_zen5_kernels.txt`
  // Features: VPCLMULQDQ + AVX-512
  //
  // Key differences vs Zen4:
  // - Different optimal multi-stream counts for CRC16/24 kernels
  // - CRC64 (large) prefers VPCLMUL-2way over the 4×512 kernel
  // ───────────────────────────────────────────────────────────────────────────
  // ───────────────────────────────────────────────────────────────────────────
  // Zen5 Table
  //
  // Benchmark source: `crates/checksum/bench_baseline/linux_x86-64_zen5_kernels.txt`
  // Features: VPCLMULQDQ + AVX-512
  // Peak throughputs: CRC-16 ~66 GiB/s, CRC-32C ~66 GiB/s, CRC-64 ~66 GiB/s
  // ───────────────────────────────────────────────────────────────────────────
  pub static ZEN5_TABLE: KernelTable = KernelTable {
    requires: platform::caps::x86::VPCLMUL_READY
      .union(platform::caps::x86::PCLMUL_READY)
      .union(platform::caps::x86::CRC32C_READY),
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL_SMALL_KERNEL, // pclmul-small @ 10.68 GiB/s
      crc16_ccitt_name: "x86_64/pclmul-small",
      crc16_ibm: crc16_k::IBM_PCLMUL_SMALL_KERNEL, // pclmul-small @ 9.89 GiB/s
      crc16_ibm_name: "x86_64/pclmul-small",
      crc24_openpgp: crc24_k::OPENPGP_PCLMUL_SMALL_KERNEL, // pclmul-small @ 8.39 GiB/s
      crc24_openpgp_name: "x86_64/pclmul-small",
      crc32_ieee: crc32_k::CRC32_PCLMUL_SMALL_KERNEL, // pclmul-small @ 9.21 GiB/s
      crc32_ieee_name: "x86_64/pclmul-small",
      crc32c: crc32_k::CRC32C_HWCRC[0], // hwcrc @ 25.33 GiB/s
      crc32c_name: "x86_64/hwcrc",
      crc64_xz: crc64_k::XZ_PCLMUL_SMALL, // pclmul-small @ 7.82 GiB/s
      crc64_xz_name: "x86_64/pclmul-small",
      crc64_nvme: crc64_k::NVME_PCLMUL_SMALL, // pclmul-small @ 7.84 GiB/s
      crc64_nvme_name: "x86_64/pclmul-small",
    },

    s: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPCLMUL[2], // vpclmul-4way @ 18.38 GiB/s
      crc16_ccitt_name: "x86_64/vpclmul-4way",
      crc16_ibm: crc16_k::IBM_VPCLMUL[1], // vpclmul-2way @ 17.52 GiB/s
      crc16_ibm_name: "x86_64/vpclmul-2way",
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[3], // vpclmul-7way @ 17.78 GiB/s
      crc24_openpgp_name: "x86_64/vpclmul-7way",
      crc32_ieee: crc32_k::CRC32_VPCLMUL[0], // vpclmul @ 16.43 GiB/s
      crc32_ieee_name: "x86_64/vpclmul",
      crc32c: crc32_k::CRC32C_HWCRC[0], // hwcrc @ 36.14 GiB/s
      crc32c_name: "x86_64/hwcrc",
      crc64_xz: crc64_k::XZ_VPCLMUL[0], // vpclmul @ 17.57 GiB/s
      crc64_xz_name: "x86_64/vpclmul",
      crc64_nvme: crc64_k::NVME_VPCLMUL[0], // vpclmul @ 17.52 GiB/s
      crc64_nvme_name: "x86_64/vpclmul",
    },

    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPCLMUL[1], // vpclmul-2way @ 56.87 GiB/s
      crc16_ccitt_name: "x86_64/vpclmul-2way",
      crc16_ibm: crc16_k::IBM_VPCLMUL[1], // vpclmul-2way @ 56.59 GiB/s
      crc16_ibm_name: "x86_64/vpclmul-2way",
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[4], // vpclmul-8way @ 53.53 GiB/s
      crc24_openpgp_name: "x86_64/vpclmul-8way",
      crc32_ieee: crc32_k::CRC32_VPCLMUL[1], // vpclmul-2way @ 56.20 GiB/s
      crc32_ieee_name: "x86_64/vpclmul-2way",
      crc32c: crc32_k::CRC32C_FUSION_VPCLMUL[0], // fusion-vpclmul-v3x2 @ 66.24 GiB/s
      crc32c_name: "x86_64/fusion-vpclmul-v3x2",
      crc64_xz: crc64_k::XZ_VPCLMUL[1], // vpclmul-2way @ 56.52 GiB/s
      crc64_xz_name: "x86_64/vpclmul-2way",
      crc64_nvme: crc64_k::NVME_VPCLMUL[1], // vpclmul-2way @ 56.53 GiB/s
      crc64_nvme_name: "x86_64/vpclmul-2way",
    },

    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPCLMUL[2], // vpclmul-4way @ 66.09 GiB/s
      crc16_ccitt_name: "x86_64/vpclmul-4way",
      crc16_ibm: crc16_k::IBM_VPCLMUL[1], // vpclmul-2way @ 66.01 GiB/s
      crc16_ibm_name: "x86_64/vpclmul-2way",
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[4], // vpclmul-8way @ 65.97 GiB/s
      crc24_openpgp_name: "x86_64/vpclmul-8way",
      crc32_ieee: crc32_k::CRC32_VPCLMUL[1], // vpclmul-2way @ 66.35 GiB/s
      crc32_ieee_name: "x86_64/vpclmul-2way",
      crc32c: crc32_k::CRC32C_FUSION_VPCLMUL[0], // fusion-vpclmul-v3x2 @ 66.22 GiB/s
      crc32c_name: "x86_64/fusion-vpclmul-v3x2",
      crc64_xz: crc64_k::XZ_VPCLMUL_4X512, // vpclmul-4x512 @ 64.35 GiB/s
      crc64_xz_name: "x86_64/vpclmul-4x512",
      crc64_nvme: crc64_k::NVME_VPCLMUL[2], // vpclmul-4way @ 66.21 GiB/s
      crc64_nvme_name: "x86_64/vpclmul-4way",
    },
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
    requires: platform::caps::x86::VPCLMUL_READY
      .union(platform::caps::x86::PCLMUL_READY)
      .union(platform::caps::x86::CRC32C_READY),
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL_SMALL_KERNEL,
      crc16_ccitt_name: "x86_64/pclmul-small",
      crc16_ibm: crc16_k::IBM_PCLMUL_SMALL_KERNEL,
      crc16_ibm_name: "x86_64/pclmul-small",
      crc24_openpgp: crc24_k::OPENPGP_PCLMUL_SMALL_KERNEL,
      crc24_openpgp_name: "x86_64/pclmul-small",
      crc32_ieee: crc32_k::CRC32_PCLMUL_SMALL_KERNEL, // pclmul-small @ 9.30 GiB/s
      crc32_ieee_name: "x86_64/pclmul-small",
      crc32c: crc32_k::CRC32C_HWCRC[0], // hwcrc @ 27.73 GiB/s
      crc32c_name: "x86_64/hwcrc",
      crc64_xz: crc64_k::XZ_PCLMUL_SMALL,
      crc64_xz_name: "x86_64/pclmul-small",
      crc64_nvme: crc64_k::NVME_PCLMUL_SMALL,
      crc64_nvme_name: "x86_64/pclmul-small",
    },

    s: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPCLMUL[1], // 2-way @ 18.12 GiB/s
      crc16_ccitt_name: "x86_64/vpclmul-2way",
      crc16_ibm: crc16_k::IBM_VPCLMUL[1], // 2-way @ 19.90 GiB/s
      crc16_ibm_name: "x86_64/vpclmul-2way",
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[3], // 7-way @ 17.28 GiB/s
      crc24_openpgp_name: "x86_64/vpclmul-7way",
      crc32_ieee: crc32_k::CRC32_VPCLMUL[0], // 1-way @ 19.15 GiB/s
      crc32_ieee_name: "x86_64/vpclmul",
      crc32c: crc32_k::CRC32C_HWCRC[0], // hwcrc @ 23.59 GiB/s
      crc32c_name: "x86_64/hwcrc",
      crc64_xz: crc64_k::XZ_VPCLMUL[0], // 1-way @ 19.48 GiB/s
      crc64_xz_name: "x86_64/vpclmul",
      crc64_nvme: crc64_k::NVME_VPCLMUL[0], // 1-way @ 20.70 GiB/s
      crc64_nvme_name: "x86_64/vpclmul",
    },

    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPCLMUL[1], // 2-way @ 61.24 GiB/s
      crc16_ccitt_name: "x86_64/vpclmul-2way",
      crc16_ibm: crc16_k::IBM_VPCLMUL[1], // 2-way @ 64.84 GiB/s
      crc16_ibm_name: "x86_64/vpclmul-2way",
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[2], // 4-way @ 34.72 GiB/s (bench: 4way beats 8way)
      crc24_openpgp_name: "x86_64/vpclmul-4way",
      crc32_ieee: crc32_k::CRC32_VPCLMUL[1], // 2-way @ 64.13 GiB/s
      crc32_ieee_name: "x86_64/vpclmul-2way",
      crc32c: crc32_k::CRC32C_FUSION_VPCLMUL[0], // fusion-vpclmul-v3x2 @ 58.96 GiB/s
      crc32c_name: "x86_64/fusion-vpclmul-v3x2",
      crc64_xz: crc64_k::XZ_VPCLMUL[1], // 2-way @ 66.51 GiB/s
      crc64_xz_name: "x86_64/vpclmul-2way",
      crc64_nvme: crc64_k::NVME_VPCLMUL[1], // 2-way @ 66.51 GiB/s
      crc64_nvme_name: "x86_64/vpclmul-2way",
    },

    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPCLMUL[2], // 4-way @ 73.97 GiB/s
      crc16_ccitt_name: "x86_64/vpclmul-4way",
      crc16_ibm: crc16_k::IBM_VPCLMUL[1], // 2-way @ 78.09 GiB/s
      crc16_ibm_name: "x86_64/vpclmul-2way",
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[2], // 4-way @ 71.94 GiB/s (bench: 4way beats 2way)
      crc24_openpgp_name: "x86_64/vpclmul-4way",
      crc32_ieee: crc32_k::CRC32_VPCLMUL[1], // 2-way @ 72.57 GiB/s
      crc32_ieee_name: "x86_64/vpclmul-2way",
      crc32c: crc32_k::CRC32C_FUSION_VPCLMUL[4], // fusion-vpclmul-v3x2-8way @ 75.16 GiB/s
      crc32c_name: "x86_64/fusion-vpclmul-v3x2-8way",
      crc64_xz: crc64_k::XZ_VPCLMUL[2], // 4-way @ 74.62 GiB/s (bench: 4way beats 4x512)
      crc64_xz_name: "x86_64/vpclmul-4way",
      crc64_nvme: crc64_k::NVME_VPCLMUL[2], // 4-way @ 74.10 GiB/s
      crc64_nvme_name: "x86_64/vpclmul-4way",
    },
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Intel SPR Table
  //
  // Benchmark source: `crates/checksum/bench_baseline/linux_x86-64_intel_kernels.txt`
  // Features: VPCLMULQDQ + AVX-512
  //
  // Intel's optimal selections differ meaningfully from Zen4/Zen5, especially
  // for CRC-64 and small-buffer thresholds.
  // ───────────────────────────────────────────────────────────────────────────
  // IntelSpr Table
  //
  // Generated by rscrypto-tune. Do not edit manually.
  // ───────────────────────────────────────────────────────────────────────────
  pub static INTEL_SPR_TABLE: KernelTable = KernelTable {
    requires: platform::caps::x86::CRC32C_READY
      .union(platform::caps::x86::PCLMUL_READY)
      .union(platform::caps::x86::VPCLMUL_READY),
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL_SMALL_KERNEL,
      crc16_ccitt_name: "x86_64/pclmul-small",
      crc16_ibm: crc16_k::IBM_VPCLMUL[2],
      crc16_ibm_name: "x86_64/vpclmul-4way",
      crc24_openpgp: crc24_k::OPENPGP_PCLMUL[2],
      crc24_openpgp_name: "x86_64/pclmul-4way",
      crc32_ieee: crc32_k::CRC32_PCLMUL_SMALL_KERNEL,
      crc32_ieee_name: "x86_64/pclmul-small",
      crc32c: crc32_k::CRC32C_FUSION_VPCLMUL[0],
      crc32c_name: "x86_64/fusion-vpclmul-v3x2",
      crc64_xz: crc64_k::XZ_PCLMUL_SMALL,
      crc64_xz_name: "x86_64/pclmul-small",
      crc64_nvme: crc64_k::NVME_PCLMUL_SMALL,
      crc64_nvme_name: "x86_64/pclmul-small",
    },

    s: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPCLMUL[3],
      crc16_ccitt_name: "x86_64/vpclmul-7way",
      crc16_ibm: crc16_k::IBM_VPCLMUL[3],
      crc16_ibm_name: "x86_64/vpclmul-7way",
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[0],
      crc24_openpgp_name: "x86_64/vpclmul",
      crc32_ieee: crc32_k::CRC32_VPCLMUL[3],
      crc32_ieee_name: "x86_64/vpclmul-7way",
      crc32c: crc32_k::CRC32C_HWCRC[4],
      crc32c_name: "x86_64/hwcrc-8way",
      crc64_xz: crc64_k::XZ_VPCLMUL[3],
      crc64_xz_name: "x86_64/vpclmul-7way",
      crc64_nvme: crc64_k::NVME_PCLMUL[0],
      crc64_nvme_name: "x86_64/pclmul",
    },

    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPCLMUL[3],
      crc16_ccitt_name: "x86_64/vpclmul-7way",
      crc16_ibm: crc16_k::IBM_VPCLMUL[4],
      crc16_ibm_name: "x86_64/vpclmul-8way",
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[0],
      crc24_openpgp_name: "x86_64/vpclmul",
      crc32_ieee: crc32_k::CRC32_VPCLMUL[3],
      crc32_ieee_name: "x86_64/vpclmul-7way",
      crc32c: crc32_k::CRC32C_FUSION_VPCLMUL[0],
      crc32c_name: "x86_64/fusion-vpclmul-v3x2",
      crc64_xz: crc64_k::XZ_VPCLMUL[4],
      crc64_xz_name: "x86_64/vpclmul-8way",
      crc64_nvme: crc64_k::NVME_VPCLMUL[3],
      crc64_nvme_name: "x86_64/vpclmul-7way",
    },

    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPCLMUL[2],
      crc16_ccitt_name: "x86_64/vpclmul-4way",
      crc16_ibm: crc16_k::IBM_VPCLMUL[1],
      crc16_ibm_name: "x86_64/vpclmul-2way",
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[1],
      crc24_openpgp_name: "x86_64/vpclmul-2way",
      crc32_ieee: crc32_k::CRC32_VPCLMUL[0],
      crc32_ieee_name: "x86_64/vpclmul",
      crc32c: crc32_k::CRC32C_FUSION_VPCLMUL[1],
      crc32c_name: "x86_64/fusion-vpclmul-v3x2-2way",
      crc64_xz: crc64_k::XZ_VPCLMUL_4X512,
      crc64_xz_name: "x86_64/vpclmul-4x512",
      crc64_nvme: crc64_k::NVME_VPCLMUL_4X512,
      crc64_nvme_name: "x86_64/vpclmul-4x512",
    },
  };

  // Intel ICL-SP-ish Table (no AMX)
  //
  // Generated by rscrypto-tune (Intel ICL runner that reported TuneKind::IntelSpr).
  // Selected by `prefer_intel_icl_tables` when AMX is absent.
  pub static INTEL_ICL_TABLE: KernelTable = KernelTable {
    requires: platform::caps::x86::CRC32C_READY
      .union(platform::caps::x86::PCLMUL_READY)
      .union(platform::caps::x86::VPCLMUL_READY),
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL_SMALL_KERNEL,
      crc16_ccitt_name: "x86_64/pclmul-small",
      crc16_ibm: crc16_k::IBM_PCLMUL_SMALL_KERNEL,
      crc16_ibm_name: "x86_64/pclmul-small",
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[0],
      crc24_openpgp_name: "x86_64/vpclmul",
      crc32_ieee: crc32_k::CRC32_VPCLMUL_SMALL_KERNEL,
      crc32_ieee_name: "x86_64/vpclmul-small",
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc32c_name: "x86_64/hwcrc",
      crc64_xz: crc64_k::XZ_PCLMUL_SMALL,
      crc64_xz_name: "x86_64/pclmul-small",
      crc64_nvme: crc64_k::NVME_PCLMUL_SMALL,
      crc64_nvme_name: "x86_64/pclmul-small",
    },

    s: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL[0],
      crc16_ccitt_name: "x86_64/pclmul",
      crc16_ibm: crc16_k::IBM_PCLMUL[0],
      crc16_ibm_name: "x86_64/pclmul",
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[0],
      crc24_openpgp_name: "x86_64/vpclmul",
      crc32_ieee: crc32_k::CRC32_VPCLMUL[1],
      crc32_ieee_name: "x86_64/vpclmul-2way",
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc32c_name: "x86_64/hwcrc",
      crc64_xz: crc64_k::XZ_PCLMUL[0],
      crc64_xz_name: "x86_64/pclmul",
      crc64_nvme: crc64_k::NVME_PCLMUL[0],
      crc64_nvme_name: "x86_64/pclmul",
    },

    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPCLMUL[4],
      crc16_ccitt_name: "x86_64/vpclmul-8way",
      crc16_ibm: crc16_k::IBM_VPCLMUL[4],
      crc16_ibm_name: "x86_64/vpclmul-8way",
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[1],
      crc24_openpgp_name: "x86_64/vpclmul-2way",
      crc32_ieee: crc32_k::CRC32_VPCLMUL[1],
      crc32_ieee_name: "x86_64/vpclmul-2way",
      crc32c: crc32_k::CRC32C_FUSION_VPCLMUL[0],
      crc32c_name: "x86_64/fusion-vpclmul-v3x2",
      crc64_xz: crc64_k::XZ_VPCLMUL[1],
      crc64_xz_name: "x86_64/vpclmul-2way",
      crc64_nvme: crc64_k::NVME_VPCLMUL[1],
      crc64_nvme_name: "x86_64/vpclmul-2way",
    },

    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPCLMUL[1],
      crc16_ccitt_name: "x86_64/vpclmul-2way",
      crc16_ibm: crc16_k::IBM_VPCLMUL[1],
      crc16_ibm_name: "x86_64/vpclmul-2way",
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[4],
      crc24_openpgp_name: "x86_64/vpclmul-8way",
      crc32_ieee: crc32_k::CRC32_VPCLMUL[2],
      crc32_ieee_name: "x86_64/vpclmul-4way",
      crc32c: crc32_k::CRC32C_FUSION_VPCLMUL[0],
      crc32c_name: "x86_64/fusion-vpclmul-v3x2",
      crc64_xz: crc64_k::XZ_VPCLMUL[1],
      crc64_xz_name: "x86_64/vpclmul-2way",
      crc64_nvme: crc64_k::NVME_VPCLMUL[1],
      crc64_nvme_name: "x86_64/vpclmul-2way",
    },
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Generic x86-64 VPCLMUL Table
  //
  // For unknown x86-64 platforms with VPCLMULQDQ.
  // Uses Zen4 selections (good AVX-512/VPCLMUL support).
  // ───────────────────────────────────────────────────────────────────────────
  pub static GENERIC_X86_VPCLMUL_TABLE: KernelTable = ZEN4_TABLE;

  /// VPCLMUL table that never selects SSE4.2 CRC32C instructions/fusion.
  ///
  /// Use on systems with VPCLMUL but without SSE4.2 (`CRC32C_READY`).
  pub static GENERIC_X86_VPCLMUL_NO_CRC32C_TABLE: KernelTable = KernelTable {
    requires: platform::caps::x86::VPCLMUL_READY.union(platform::caps::x86::PCLMUL_READY),
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL_SMALL_KERNEL,
      crc16_ccitt_name: "x86_64/pclmul-small",
      crc16_ibm: crc16_k::IBM_PCLMUL_SMALL_KERNEL,
      crc16_ibm_name: "x86_64/pclmul-small",
      crc24_openpgp: crc24_k::OPENPGP_PCLMUL_SMALL_KERNEL,
      crc24_openpgp_name: "x86_64/pclmul-small",
      crc32_ieee: crc32_k::CRC32_PCLMUL_SMALL_KERNEL,
      crc32_ieee_name: "x86_64/pclmul-small",
      crc32c: crate::crc32::portable::crc32c_bytewise,
      crc32c_name: crate::crc32::portable::BYTEWISE_KERNEL_NAME,
      crc64_xz: crc64_k::XZ_PCLMUL_SMALL,
      crc64_xz_name: "x86_64/pclmul-small",
      crc64_nvme: crc64_k::NVME_PCLMUL_SMALL,
      crc64_nvme_name: "x86_64/pclmul-small",
    },

    s: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPCLMUL[1],
      crc16_ccitt_name: "x86_64/vpclmul-2way",
      crc16_ibm: crc16_k::IBM_VPCLMUL[1],
      crc16_ibm_name: "x86_64/vpclmul-2way",
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[3],
      crc24_openpgp_name: "x86_64/vpclmul-7way",
      crc32_ieee: crc32_k::CRC32_VPCLMUL[0],
      crc32_ieee_name: "x86_64/vpclmul",
      crc32c: crate::crc32::portable::crc32c_slice16,
      crc32c_name: "portable/slice16",
      crc64_xz: crc64_k::XZ_VPCLMUL[0],
      crc64_xz_name: "x86_64/vpclmul",
      crc64_nvme: crc64_k::NVME_VPCLMUL[0],
      crc64_nvme_name: "x86_64/vpclmul",
    },

    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPCLMUL[1],
      crc16_ccitt_name: "x86_64/vpclmul-2way",
      crc16_ibm: crc16_k::IBM_VPCLMUL[1],
      crc16_ibm_name: "x86_64/vpclmul-2way",
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[4],
      crc24_openpgp_name: "x86_64/vpclmul-8way",
      crc32_ieee: crc32_k::CRC32_VPCLMUL[1],
      crc32_ieee_name: "x86_64/vpclmul-2way",
      crc32c: crate::crc32::portable::crc32c_slice16,
      crc32c_name: "portable/slice16",
      crc64_xz: crc64_k::XZ_VPCLMUL[1],
      crc64_xz_name: "x86_64/vpclmul-2way",
      crc64_nvme: crc64_k::NVME_VPCLMUL[1],
      crc64_nvme_name: "x86_64/vpclmul-2way",
    },

    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPCLMUL[2],
      crc16_ccitt_name: "x86_64/vpclmul-4way",
      crc16_ibm: crc16_k::IBM_VPCLMUL[1],
      crc16_ibm_name: "x86_64/vpclmul-2way",
      crc24_openpgp: crc24_k::OPENPGP_VPCLMUL[1],
      crc24_openpgp_name: "x86_64/vpclmul-2way",
      crc32_ieee: crc32_k::CRC32_VPCLMUL[1],
      crc32_ieee_name: "x86_64/vpclmul-2way",
      crc32c: crate::crc32::portable::crc32c_slice16,
      crc32c_name: "portable/slice16",
      crc64_xz: crc64_k::XZ_VPCLMUL_4X512,
      crc64_xz_name: "x86_64/vpclmul-4x512",
      crc64_nvme: crc64_k::NVME_VPCLMUL[2],
      crc64_nvme_name: "x86_64/vpclmul-4way",
    },
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Generic x86-64 PCLMUL Table (conservative)
  //
  // Benchmark source: historical Windows "Default" baseline (no AVX-512).
  // Note: Windows benchmark baselines are no longer tracked in-tree.
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
    requires: platform::caps::x86::PCLMUL_READY.union(platform::caps::x86::CRC32C_READY),
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL_SMALL_KERNEL,
      crc16_ccitt_name: "x86_64/pclmul-small",
      crc16_ibm: crc16_k::IBM_PCLMUL_SMALL_KERNEL,
      crc16_ibm_name: "x86_64/pclmul-small",
      crc24_openpgp: crc24_k::OPENPGP_PCLMUL_SMALL_KERNEL,
      crc24_openpgp_name: "x86_64/pclmul-small",
      crc32_ieee: crc32_k::CRC32_PCLMUL_SMALL_KERNEL,
      crc32_ieee_name: "x86_64/pclmul-small",
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc32c_name: "x86_64/hwcrc",
      crc64_xz: crc64_k::XZ_PCLMUL_SMALL,
      crc64_xz_name: "x86_64/pclmul-small",
      crc64_nvme: crc64_k::NVME_PCLMUL_SMALL,
      crc64_nvme_name: "x86_64/pclmul-small",
    },

    s: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL[0], // 1-way
      crc16_ccitt_name: "x86_64/pclmul",
      crc16_ibm: crc16_k::IBM_PCLMUL[2], // 4-way
      crc16_ibm_name: "x86_64/pclmul-4way",
      crc24_openpgp: crc24_k::OPENPGP_PCLMUL[0],
      crc24_openpgp_name: "x86_64/pclmul",
      crc32_ieee: crc32_k::CRC32_PCLMUL[2], // 4-way
      crc32_ieee_name: "x86_64/pclmul-4way",
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc32c_name: "x86_64/hwcrc",
      crc64_xz: crc64_k::XZ_PCLMUL_SMALL, // small still wins
      crc64_xz_name: "x86_64/pclmul-small",
      crc64_nvme: crc64_k::NVME_PCLMUL_SMALL,
      crc64_nvme_name: "x86_64/pclmul-small",
    },

    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL[3], // 7-way
      crc16_ccitt_name: "x86_64/pclmul-7way",
      crc16_ibm: crc16_k::IBM_PCLMUL[0], // 1-way
      crc16_ibm_name: "x86_64/pclmul",
      crc24_openpgp: crc24_k::OPENPGP_PCLMUL[1], // 2-way
      crc24_openpgp_name: "x86_64/pclmul-2way",
      crc32_ieee: crc32_k::CRC32_PCLMUL[0], // 1-way
      crc32_ieee_name: "x86_64/pclmul",
      crc32c: crc32_k::CRC32C_HWCRC[1], // 2-way
      crc32c_name: "x86_64/hwcrc-2way",
      crc64_xz: crc64_k::XZ_PCLMUL[2], // 4-way
      crc64_xz_name: "x86_64/pclmul-4way",
      crc64_nvme: crc64_k::NVME_PCLMUL[1], // 2-way
      crc64_nvme_name: "x86_64/pclmul-2way",
    },

    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL[0], // 1-way wins at large
      crc16_ccitt_name: "x86_64/pclmul",
      crc16_ibm: crc16_k::IBM_PCLMUL[1], // 2-way
      crc16_ibm_name: "x86_64/pclmul-2way",
      crc24_openpgp: crc24_k::OPENPGP_PCLMUL[1],
      crc24_openpgp_name: "x86_64/pclmul-2way",
      crc32_ieee: crc32_k::CRC32_PCLMUL[1], // 2-way
      crc32_ieee_name: "x86_64/pclmul-2way",
      crc32c: crc32_k::CRC32C_FUSION_SSE[1], // fusion-sse-v4s3x3-2way
      crc32c_name: "x86_64/fusion-sse-v4s3x3-2way",
      crc64_xz: crc64_k::XZ_PCLMUL[0], // 1-way
      crc64_xz_name: "x86_64/pclmul",
      crc64_nvme: crc64_k::NVME_PCLMUL[1], // 2-way
      crc64_nvme_name: "x86_64/pclmul-2way",
    },
  };

  /// PCLMUL table that never selects SSE4.2 CRC32C instructions/fusion.
  ///
  /// Use on systems with PCLMUL but without SSE4.2 (`CRC32C_READY`).
  pub static GENERIC_X86_PCLMUL_NO_CRC32C_TABLE: KernelTable = KernelTable {
    requires: platform::caps::x86::PCLMUL_READY,
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL_SMALL_KERNEL,
      crc16_ccitt_name: "x86_64/pclmul-small",
      crc16_ibm: crc16_k::IBM_PCLMUL_SMALL_KERNEL,
      crc16_ibm_name: "x86_64/pclmul-small",
      crc24_openpgp: crc24_k::OPENPGP_PCLMUL_SMALL_KERNEL,
      crc24_openpgp_name: "x86_64/pclmul-small",
      crc32_ieee: crc32_k::CRC32_PCLMUL_SMALL_KERNEL,
      crc32_ieee_name: "x86_64/pclmul-small",
      crc32c: crate::crc32::portable::crc32c_bytewise,
      crc32c_name: crate::crc32::portable::BYTEWISE_KERNEL_NAME,
      crc64_xz: crc64_k::XZ_PCLMUL_SMALL,
      crc64_xz_name: "x86_64/pclmul-small",
      crc64_nvme: crc64_k::NVME_PCLMUL_SMALL,
      crc64_nvme_name: "x86_64/pclmul-small",
    },

    s: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL[0],
      crc16_ccitt_name: "x86_64/pclmul",
      crc16_ibm: crc16_k::IBM_PCLMUL[2],
      crc16_ibm_name: "x86_64/pclmul-4way",
      crc24_openpgp: crc24_k::OPENPGP_PCLMUL[0],
      crc24_openpgp_name: "x86_64/pclmul",
      crc32_ieee: crc32_k::CRC32_PCLMUL[2],
      crc32_ieee_name: "x86_64/pclmul-4way",
      crc32c: crate::crc32::portable::crc32c_slice16,
      crc32c_name: "portable/slice16",
      crc64_xz: crc64_k::XZ_PCLMUL_SMALL,
      crc64_xz_name: "x86_64/pclmul-small",
      crc64_nvme: crc64_k::NVME_PCLMUL_SMALL,
      crc64_nvme_name: "x86_64/pclmul-small",
    },

    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL[3],
      crc16_ccitt_name: "x86_64/pclmul-7way",
      crc16_ibm: crc16_k::IBM_PCLMUL[0],
      crc16_ibm_name: "x86_64/pclmul",
      crc24_openpgp: crc24_k::OPENPGP_PCLMUL[1],
      crc24_openpgp_name: "x86_64/pclmul-2way",
      crc32_ieee: crc32_k::CRC32_PCLMUL[0],
      crc32_ieee_name: "x86_64/pclmul",
      crc32c: crate::crc32::portable::crc32c_slice16,
      crc32c_name: "portable/slice16",
      crc64_xz: crc64_k::XZ_PCLMUL[2],
      crc64_xz_name: "x86_64/pclmul-4way",
      crc64_nvme: crc64_k::NVME_PCLMUL[1],
      crc64_nvme_name: "x86_64/pclmul-2way",
    },

    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PCLMUL[0],
      crc16_ccitt_name: "x86_64/pclmul",
      crc16_ibm: crc16_k::IBM_PCLMUL[1],
      crc16_ibm_name: "x86_64/pclmul-2way",
      crc24_openpgp: crc24_k::OPENPGP_PCLMUL[1],
      crc24_openpgp_name: "x86_64/pclmul-2way",
      crc32_ieee: crc32_k::CRC32_PCLMUL[1],
      crc32_ieee_name: "x86_64/pclmul-2way",
      crc32c: crate::crc32::portable::crc32c_slice16,
      crc32c_name: "portable/slice16",
      crc64_xz: crc64_k::XZ_PCLMUL[0],
      crc64_xz_name: "x86_64/pclmul",
      crc64_nvme: crc64_k::NVME_PCLMUL[1],
      crc64_nvme_name: "x86_64/pclmul-2way",
    },
  };

  /// SSE4.2-only table: accelerate CRC32C, keep other variants portable.
  pub static GENERIC_X86_CRC32C_ONLY_TABLE: KernelTable = KernelTable {
    requires: platform::caps::x86::CRC32C_READY,
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crate::crc16::portable::crc16_ccitt_slice8,
      crc16_ccitt_name: "portable/slice8",
      crc16_ibm: crate::crc16::portable::crc16_ibm_slice8,
      crc16_ibm_name: "portable/slice8",
      crc24_openpgp: crate::crc24::portable::crc24_openpgp_slice8,
      crc24_openpgp_name: "portable/slice8",
      crc32_ieee: crate::crc32::portable::crc32_bytewise_ieee,
      crc32_ieee_name: crate::crc32::portable::BYTEWISE_KERNEL_NAME,
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc32c_name: "x86_64/hwcrc",
      crc64_xz: crate::crc64::portable::crc64_slice16_xz,
      crc64_xz_name: "portable/slice16",
      crc64_nvme: crate::crc64::portable::crc64_slice16_nvme,
      crc64_nvme_name: "portable/slice16",
    },

    s: KernelSet {
      crc16_ccitt: crate::crc16::portable::crc16_ccitt_slice8,
      crc16_ccitt_name: "portable/slice8",
      crc16_ibm: crate::crc16::portable::crc16_ibm_slice8,
      crc16_ibm_name: "portable/slice8",
      crc24_openpgp: crate::crc24::portable::crc24_openpgp_slice8,
      crc24_openpgp_name: "portable/slice8",
      crc32_ieee: crate::crc32::portable::crc32_slice16_ieee,
      crc32_ieee_name: "portable/slice16",
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc32c_name: "x86_64/hwcrc",
      crc64_xz: crate::crc64::portable::crc64_slice16_xz,
      crc64_xz_name: "portable/slice16",
      crc64_nvme: crate::crc64::portable::crc64_slice16_nvme,
      crc64_nvme_name: "portable/slice16",
    },

    m: KernelSet {
      crc16_ccitt: crate::crc16::portable::crc16_ccitt_slice8,
      crc16_ccitt_name: "portable/slice8",
      crc16_ibm: crate::crc16::portable::crc16_ibm_slice8,
      crc16_ibm_name: "portable/slice8",
      crc24_openpgp: crate::crc24::portable::crc24_openpgp_slice8,
      crc24_openpgp_name: "portable/slice8",
      crc32_ieee: crate::crc32::portable::crc32_slice16_ieee,
      crc32_ieee_name: "portable/slice16",
      crc32c: crc32_k::CRC32C_HWCRC[1],
      crc32c_name: "x86_64/hwcrc-2way",
      crc64_xz: crate::crc64::portable::crc64_slice16_xz,
      crc64_xz_name: "portable/slice16",
      crc64_nvme: crate::crc64::portable::crc64_slice16_nvme,
      crc64_nvme_name: "portable/slice16",
    },

    l: KernelSet {
      crc16_ccitt: crate::crc16::portable::crc16_ccitt_slice8,
      crc16_ccitt_name: "portable/slice8",
      crc16_ibm: crate::crc16::portable::crc16_ibm_slice8,
      crc16_ibm_name: "portable/slice8",
      crc24_openpgp: crate::crc24::portable::crc24_openpgp_slice8,
      crc24_openpgp_name: "portable/slice8",
      crc32_ieee: crate::crc32::portable::crc32_slice16_ieee,
      crc32_ieee_name: "portable/slice16",
      crc32c: crc32_k::CRC32C_HWCRC[1],
      crc32c_name: "x86_64/hwcrc-2way",
      crc64_xz: crate::crc64::portable::crc64_slice16_xz,
      crc64_xz_name: "portable/slice16",
      crc64_nvme: crate::crc64::portable::crc64_slice16_nvme,
      crc64_nvme_name: "portable/slice16",
    },
  };
}

#[cfg(target_arch = "x86_64")]
pub use x86_64_tables::*;

// ─────────────────────────────────────────────────────────────────────────────
// s390x Platform Tables
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "s390x")]
mod s390x_tables {
  use super::*;
  use crate::{
    crc16::kernels::s390x as crc16_k, crc24::kernels::s390x as crc24_k, crc32::kernels::s390x as crc32_k,
    crc64::kernels::s390x as crc64_k,
  };

  pub static S390X_Z13_TABLE: KernelTable = KernelTable {
    requires: platform::caps::s390x::Z13_READY,
    boundaries: [64, 256, 4096],
    xs: PORTABLE_SET,
    s: PORTABLE_SET,
    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VGFM[0],
      crc16_ccitt_name: "s390x/vgfm",
      crc16_ibm: crc16_k::IBM_VGFM[0],
      crc16_ibm_name: "s390x/vgfm",
      crc24_openpgp: crc24_k::OPENPGP_VGFM[0],
      crc24_openpgp_name: "s390x/vgfm",
      crc32_ieee: crc32_k::CRC32_VGFM[0],
      crc32_ieee_name: "s390x/vgfm",
      crc32c: crc32_k::CRC32C_VGFM[0],
      crc32c_name: "s390x/vgfm",
      crc64_xz: crc64_k::XZ_VGFM[0],
      crc64_xz_name: "s390x/vgfm",
      crc64_nvme: crc64_k::NVME_VGFM[0],
      crc64_nvme_name: "s390x/vgfm",
    },
    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VGFM[1],
      crc16_ccitt_name: "s390x/vgfm-2way",
      crc16_ibm: crc16_k::IBM_VGFM[1],
      crc16_ibm_name: "s390x/vgfm-2way",
      crc24_openpgp: crc24_k::OPENPGP_VGFM[1],
      crc24_openpgp_name: "s390x/vgfm-2way",
      crc32_ieee: crc32_k::CRC32_VGFM[1],
      crc32_ieee_name: "s390x/vgfm-2way",
      crc32c: crc32_k::CRC32C_VGFM[1],
      crc32c_name: "s390x/vgfm-2way",
      crc64_xz: crc64_k::XZ_VGFM[1],
      crc64_xz_name: "s390x/vgfm-2way",
      crc64_nvme: crc64_k::NVME_VGFM[1],
      crc64_nvme_name: "s390x/vgfm-2way",
    },
  };

  pub static S390X_Z14_TABLE: KernelTable = KernelTable {
    requires: platform::caps::s390x::Z13_READY,
    boundaries: [64, 128, 4096],
    xs: PORTABLE_SET,
    s: PORTABLE_SET,
    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VGFM[0],
      crc16_ccitt_name: "s390x/vgfm",
      crc16_ibm: crc16_k::IBM_VGFM[0],
      crc16_ibm_name: "s390x/vgfm",
      crc24_openpgp: crc24_k::OPENPGP_VGFM[0],
      crc24_openpgp_name: "s390x/vgfm",
      crc32_ieee: crc32_k::CRC32_VGFM[0],
      crc32_ieee_name: "s390x/vgfm",
      crc32c: crc32_k::CRC32C_VGFM[0],
      crc32c_name: "s390x/vgfm",
      crc64_xz: crc64_k::XZ_VGFM[0],
      crc64_xz_name: "s390x/vgfm",
      crc64_nvme: crc64_k::NVME_VGFM[0],
      crc64_nvme_name: "s390x/vgfm",
    },
    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VGFM[1],
      crc16_ccitt_name: "s390x/vgfm-2way",
      crc16_ibm: crc16_k::IBM_VGFM[1],
      crc16_ibm_name: "s390x/vgfm-2way",
      crc24_openpgp: crc24_k::OPENPGP_VGFM[1],
      crc24_openpgp_name: "s390x/vgfm-2way",
      crc32_ieee: crc32_k::CRC32_VGFM[1],
      crc32_ieee_name: "s390x/vgfm-2way",
      crc32c: crc32_k::CRC32C_VGFM[1],
      crc32c_name: "s390x/vgfm-2way",
      crc64_xz: crc64_k::XZ_VGFM[1],
      crc64_xz_name: "s390x/vgfm-2way",
      crc64_nvme: crc64_k::NVME_VGFM[1],
      crc64_nvme_name: "s390x/vgfm-2way",
    },
  };

  pub static S390X_Z15_TABLE: KernelTable = KernelTable {
    requires: platform::caps::s390x::Z13_READY,
    boundaries: [64, 64, 4096],
    xs: PORTABLE_SET,
    s: PORTABLE_SET,
    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VGFM[0],
      crc16_ccitt_name: "s390x/vgfm",
      crc16_ibm: crc16_k::IBM_VGFM[0],
      crc16_ibm_name: "s390x/vgfm",
      crc24_openpgp: crc24_k::OPENPGP_VGFM[0],
      crc24_openpgp_name: "s390x/vgfm",
      crc32_ieee: crc32_k::CRC32_VGFM[0],
      crc32_ieee_name: "s390x/vgfm",
      crc32c: crc32_k::CRC32C_VGFM[0],
      crc32c_name: "s390x/vgfm",
      crc64_xz: crc64_k::XZ_VGFM[0],
      crc64_xz_name: "s390x/vgfm",
      crc64_nvme: crc64_k::NVME_VGFM[0],
      crc64_nvme_name: "s390x/vgfm",
    },
    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VGFM[1],
      crc16_ccitt_name: "s390x/vgfm-2way",
      crc16_ibm: crc16_k::IBM_VGFM[1],
      crc16_ibm_name: "s390x/vgfm-2way",
      crc24_openpgp: crc24_k::OPENPGP_VGFM[1],
      crc24_openpgp_name: "s390x/vgfm-2way",
      crc32_ieee: crc32_k::CRC32_VGFM[1],
      crc32_ieee_name: "s390x/vgfm-2way",
      crc32c: crc32_k::CRC32C_VGFM[1],
      crc32c_name: "s390x/vgfm-2way",
      crc64_xz: crc64_k::XZ_VGFM[1],
      crc64_xz_name: "s390x/vgfm-2way",
      crc64_nvme: crc64_k::NVME_VGFM[1],
      crc64_nvme_name: "s390x/vgfm-2way",
    },
  };
}

#[cfg(target_arch = "s390x")]
pub use s390x_tables::*;

// ─────────────────────────────────────────────────────────────────────────────
// powerpc64 Platform Tables
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "powerpc64")]
mod power_tables {
  use super::*;
  use crate::{
    crc16::kernels::power as crc16_k, crc24::kernels::power as crc24_k, crc32::kernels::power as crc32_k,
    crc64::kernels::power as crc64_k,
  };

  pub static POWER8_TABLE: KernelTable = KernelTable {
    requires: platform::caps::power::VPMSUM_READY,
    boundaries: [64, 128, 4096],
    xs: PORTABLE_SET,
    s: PORTABLE_SET,
    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPMSUM[0],
      crc16_ccitt_name: "power/vpmsum",
      crc16_ibm: crc16_k::IBM_VPMSUM[0],
      crc16_ibm_name: "power/vpmsum",
      crc24_openpgp: crc24_k::OPENPGP_VPMSUM[0],
      crc24_openpgp_name: "power/vpmsum",
      crc32_ieee: crc32_k::CRC32_VPMSUM[0],
      crc32_ieee_name: "power/vpmsum",
      crc32c: crc32_k::CRC32C_VPMSUM[0],
      crc32c_name: "power/vpmsum",
      crc64_xz: crc64_k::XZ_VPMSUM[0],
      crc64_xz_name: "power/vpmsum",
      crc64_nvme: crc64_k::NVME_VPMSUM[0],
      crc64_nvme_name: "power/vpmsum",
    },
    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPMSUM[1],
      crc16_ccitt_name: "power/vpmsum-2way",
      crc16_ibm: crc16_k::IBM_VPMSUM[1],
      crc16_ibm_name: "power/vpmsum-2way",
      crc24_openpgp: crc24_k::OPENPGP_VPMSUM[1],
      crc24_openpgp_name: "power/vpmsum-2way",
      crc32_ieee: crc32_k::CRC32_VPMSUM[1],
      crc32_ieee_name: "power/vpmsum-2way",
      crc32c: crc32_k::CRC32C_VPMSUM[1],
      crc32c_name: "power/vpmsum-2way",
      crc64_xz: crc64_k::XZ_VPMSUM[1],
      crc64_xz_name: "power/vpmsum-2way",
      crc64_nvme: crc64_k::NVME_VPMSUM[1],
      crc64_nvme_name: "power/vpmsum-2way",
    },
  };

  pub static POWER9_TABLE: KernelTable = KernelTable {
    requires: platform::caps::power::VPMSUM_READY,
    boundaries: [64, 64, 4096],
    xs: PORTABLE_SET,
    s: PORTABLE_SET,
    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPMSUM[0],
      crc16_ccitt_name: "power/vpmsum",
      crc16_ibm: crc16_k::IBM_VPMSUM[0],
      crc16_ibm_name: "power/vpmsum",
      crc24_openpgp: crc24_k::OPENPGP_VPMSUM[0],
      crc24_openpgp_name: "power/vpmsum",
      crc32_ieee: crc32_k::CRC32_VPMSUM[0],
      crc32_ieee_name: "power/vpmsum",
      crc32c: crc32_k::CRC32C_VPMSUM[0],
      crc32c_name: "power/vpmsum",
      crc64_xz: crc64_k::XZ_VPMSUM[0],
      crc64_xz_name: "power/vpmsum",
      crc64_nvme: crc64_k::NVME_VPMSUM[0],
      crc64_nvme_name: "power/vpmsum",
    },
    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPMSUM[2],
      crc16_ccitt_name: "power/vpmsum-4way",
      crc16_ibm: crc16_k::IBM_VPMSUM[2],
      crc16_ibm_name: "power/vpmsum-4way",
      crc24_openpgp: crc24_k::OPENPGP_VPMSUM[2],
      crc24_openpgp_name: "power/vpmsum-4way",
      crc32_ieee: crc32_k::CRC32_VPMSUM[2],
      crc32_ieee_name: "power/vpmsum-4way",
      crc32c: crc32_k::CRC32C_VPMSUM[2],
      crc32c_name: "power/vpmsum-4way",
      crc64_xz: crc64_k::XZ_VPMSUM[2],
      crc64_xz_name: "power/vpmsum-4way",
      crc64_nvme: crc64_k::NVME_VPMSUM[2],
      crc64_nvme_name: "power/vpmsum-4way",
    },
  };

  pub static POWER10_TABLE: KernelTable = KernelTable {
    requires: platform::caps::power::VPMSUM_READY,
    boundaries: [64, 64, 4096],
    xs: PORTABLE_SET,
    s: PORTABLE_SET,
    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPMSUM[0],
      crc16_ccitt_name: "power/vpmsum",
      crc16_ibm: crc16_k::IBM_VPMSUM[0],
      crc16_ibm_name: "power/vpmsum",
      crc24_openpgp: crc24_k::OPENPGP_VPMSUM[0],
      crc24_openpgp_name: "power/vpmsum",
      crc32_ieee: crc32_k::CRC32_VPMSUM[0],
      crc32_ieee_name: "power/vpmsum",
      crc32c: crc32_k::CRC32C_VPMSUM[0],
      crc32c_name: "power/vpmsum",
      crc64_xz: crc64_k::XZ_VPMSUM[0],
      crc64_xz_name: "power/vpmsum",
      crc64_nvme: crc64_k::NVME_VPMSUM[0],
      crc64_nvme_name: "power/vpmsum",
    },
    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_VPMSUM[2],
      crc16_ccitt_name: "power/vpmsum-4way",
      crc16_ibm: crc16_k::IBM_VPMSUM[2],
      crc16_ibm_name: "power/vpmsum-4way",
      crc24_openpgp: crc24_k::OPENPGP_VPMSUM[2],
      crc24_openpgp_name: "power/vpmsum-4way",
      crc32_ieee: crc32_k::CRC32_VPMSUM[2],
      crc32_ieee_name: "power/vpmsum-4way",
      crc32c: crc32_k::CRC32C_VPMSUM[2],
      crc32c_name: "power/vpmsum-4way",
      crc64_xz: crc64_k::XZ_VPMSUM[2],
      crc64_xz_name: "power/vpmsum-4way",
      crc64_nvme: crc64_k::NVME_VPMSUM[2],
      crc64_nvme_name: "power/vpmsum-4way",
    },
  };
}

#[cfg(target_arch = "powerpc64")]
pub use power_tables::*;

// ─────────────────────────────────────────────────────────────────────────────
// riscv64 Platform Tables
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "riscv64")]
mod riscv64_tables {
  use super::*;
  use crate::{
    crc16::kernels::riscv64 as crc16_k, crc24::kernels::riscv64 as crc24_k, crc32::kernels::riscv64 as crc32_k,
    crc64::kernels::riscv64 as crc64_k,
  };

  pub static RISCV64_ZBC_TABLE: KernelTable = KernelTable {
    requires: platform::caps::riscv::ZBC,
    boundaries: [64, 64, 4096],
    xs: PORTABLE_SET,
    s: PORTABLE_SET,
    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_ZBC[0],
      crc16_ccitt_name: "riscv64/zbc",
      crc16_ibm: crc16_k::IBM_ZBC[0],
      crc16_ibm_name: "riscv64/zbc",
      crc24_openpgp: crc24_k::OPENPGP_ZBC[0],
      crc24_openpgp_name: "riscv64/zbc",
      crc32_ieee: crc32_k::CRC32_ZBC[0],
      crc32_ieee_name: "riscv64/zbc",
      crc32c: crc32_k::CRC32C_ZBC[0],
      crc32c_name: "riscv64/zbc",
      crc64_xz: crc64_k::XZ_ZBC[0],
      crc64_xz_name: "riscv64/zbc",
      crc64_nvme: crc64_k::NVME_ZBC[0],
      crc64_nvme_name: "riscv64/zbc",
    },
    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_ZBC[2],
      crc16_ccitt_name: "riscv64/zbc-4way",
      crc16_ibm: crc16_k::IBM_ZBC[2],
      crc16_ibm_name: "riscv64/zbc-4way",
      crc24_openpgp: crc24_k::OPENPGP_ZBC[2],
      crc24_openpgp_name: "riscv64/zbc-4way",
      crc32_ieee: crc32_k::CRC32_ZBC[2],
      crc32_ieee_name: "riscv64/zbc-4way",
      crc32c: crc32_k::CRC32C_ZBC[2],
      crc32c_name: "riscv64/zbc-4way",
      crc64_xz: crc64_k::XZ_ZBC[2],
      crc64_xz_name: "riscv64/zbc-4way",
      crc64_nvme: crc64_k::NVME_ZBC[2],
      crc64_nvme_name: "riscv64/zbc-4way",
    },
  };

  pub static RISCV64_ZVBC_TABLE: KernelTable = KernelTable {
    requires: platform::caps::riscv::V.union(platform::caps::riscv::ZVBC),
    boundaries: [64, 64, 4096],
    xs: PORTABLE_SET,
    s: PORTABLE_SET,
    m: KernelSet {
      crc16_ccitt: crc16_k::CCITT_ZVBC[0],
      crc16_ccitt_name: "riscv64/zvbc",
      crc16_ibm: crc16_k::IBM_ZVBC[0],
      crc16_ibm_name: "riscv64/zvbc",
      crc24_openpgp: crc24_k::OPENPGP_ZVBC[0],
      crc24_openpgp_name: "riscv64/zvbc",
      crc32_ieee: crc32_k::CRC32_ZVBC[0],
      crc32_ieee_name: "riscv64/zvbc",
      crc32c: crc32_k::CRC32C_ZVBC[0],
      crc32c_name: "riscv64/zvbc",
      crc64_xz: crc64_k::XZ_ZVBC[0],
      crc64_xz_name: "riscv64/zvbc",
      crc64_nvme: crc64_k::NVME_ZVBC[0],
      crc64_nvme_name: "riscv64/zvbc",
    },
    l: KernelSet {
      crc16_ccitt: crc16_k::CCITT_ZVBC[2],
      crc16_ccitt_name: "riscv64/zvbc-4way",
      crc16_ibm: crc16_k::IBM_ZVBC[2],
      crc16_ibm_name: "riscv64/zvbc-4way",
      crc24_openpgp: crc24_k::OPENPGP_ZVBC[2],
      crc24_openpgp_name: "riscv64/zvbc-4way",
      crc32_ieee: crc32_k::CRC32_ZVBC[2],
      crc32_ieee_name: "riscv64/zvbc-4way",
      crc32c: crc32_k::CRC32C_ZVBC[2],
      crc32c_name: "riscv64/zvbc-4way",
      crc64_xz: crc64_k::XZ_ZVBC[2],
      crc64_xz_name: "riscv64/zvbc-4way",
      crc64_nvme: crc64_k::NVME_ZVBC[2],
      crc64_nvme_name: "riscv64/zvbc-4way",
    },
  };
}

#[cfg(target_arch = "riscv64")]
pub use riscv64_tables::*;

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
    let table = exact_match(TuneKind::AppleM1M3, APPLE_M1M3_TABLE.requires);
    assert!(table.is_some());
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn test_x86_64_exact_match() {
    // Zen4 should return the Zen4 table.
    assert!(exact_match(TuneKind::Zen4, ZEN4_TABLE.requires).is_some());
    // Zen5 should return the Zen5 table.
    assert!(exact_match(TuneKind::Zen5, ZEN5_TABLE.requires).is_some());
    // IntelSpr is split into:
    // - ICL-SP-ish (no AMX) → INTEL_ICL_TABLE
    // - SPR/EMR-ish (AMX) → INTEL_SPR_TABLE
    let icl_table = exact_match(TuneKind::IntelSpr, INTEL_SPR_TABLE.requires).unwrap();
    assert!(core::ptr::eq(icl_table, &INTEL_ICL_TABLE));

    let spr_caps = INTEL_SPR_TABLE
      .requires
      .union(platform::caps::x86::AMX_TILE)
      .union(platform::caps::x86::AMX_INT8)
      .union(platform::caps::x86::AMX_BF16)
      .union(platform::caps::x86::AMX_FP16)
      .union(platform::caps::x86::AMX_COMPLEX);
    let spr_table = exact_match(TuneKind::IntelSpr, spr_caps).unwrap();
    assert!(core::ptr::eq(spr_table, &INTEL_SPR_TABLE));
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
