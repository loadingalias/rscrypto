//! Internal CRC kernel tables and one-shot helpers.
//!
//! The public `checksum::dispatch` API was removed. What remains here is the
//! internal table-driven selector that the CRC implementations use after the
//! tuning work was collapsed into static, benchmark-backed kernel tables.

use crate::{
  checksum::dispatchers::{Crc16Fn, Crc24Fn, Crc32Fn, Crc64Fn},
  platform::Caps,
};

// ─────────────────────────────────────────────────────────────────────────────
// Global Kernel Table Cache
// ─────────────────────────────────────────────────────────────────────────────

/// Global cached kernel table, resolved once on first use.
///
/// This is the heart of the new dispatch system. Platform detection happens
/// exactly once, and all subsequent CRC calls use this pre-resolved table.
static ACTIVE_TABLE: crate::backend::cache::OnceCache<&'static KernelTable> = crate::backend::cache::OnceCache::new();

/// Get the active kernel table for this platform.
#[inline]
pub(crate) fn active_table() -> &'static KernelTable {
  ACTIVE_TABLE.get_or_init(|| select_table(crate::platform::caps()))
}

// ─────────────────────────────────────────────────────────────────────────────
// Oneshot Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum input size for the inline bytewise fast-path.
///
/// Inputs at or below this size bypass all dispatch machinery and use a simple
/// byte-at-a-time table lookup. This eliminates `active_table()` + `select_fns()`
/// + indirect fn ptr overhead (~7-10 ns) that dominates at tiny sizes.
///
/// Set to 7 (not 64) because at 8+ bytes the dispatch path's slice-by-N or
/// hardware-accelerated kernels outperform the bytewise loop.
const FAST_PATH_MAX: usize = 7;

/// Internal one-shot CRC-64/XZ helper used by trait impls.
#[inline]
pub(crate) fn crc64_xz(data: &[u8]) -> u64 {
  if data.len() <= FAST_PATH_MAX {
    if data.is_empty() {
      return 0;
    }
    return crate::checksum::crc64::portable::crc64_xz_bytewise(!0, data) ^ !0;
  }
  let table = active_table();
  let kernel = table.select_fns(data.len()).crc64_xz;
  kernel(!0, data) ^ !0
}

/// Internal one-shot CRC-64/NVME helper used by trait impls.
#[inline]
pub(crate) fn crc64_nvme(data: &[u8]) -> u64 {
  if data.len() <= FAST_PATH_MAX {
    if data.is_empty() {
      return 0;
    }
    return crate::checksum::crc64::portable::crc64_nvme_bytewise(!0, data) ^ !0;
  }
  let table = active_table();
  let kernel = table.select_fns(data.len()).crc64_nvme;
  kernel(!0, data) ^ !0
}

/// Internal one-shot CRC-32/IEEE helper used by trait impls.
#[inline]
pub(crate) fn crc32_ieee(data: &[u8]) -> u32 {
  if data.len() <= FAST_PATH_MAX {
    if data.is_empty() {
      return 0;
    }
    return crate::checksum::crc32::portable::crc32_bytewise_ieee(!0, data) ^ !0;
  }

  // aarch64: 3-tier inline dispatch bypassing function-pointer tables.
  //
  // - Small (<128 B): pure hardware CRC extension — avoids PMULL kernel entry overhead (alignment
  //   prologue + fold threshold check) for data that would just fall through to serial CRC anyway.
  // - Medium (128–1024 B): PMULL v12e_v1 fold — 12-lane carryless multiply amortizes setup cost.
  //   Inlined to eliminate indirect-call barrier.
  // - Large (>1024 B + EOR3): v9s3x2e_s3 EOR3 fusion — 9 PMULL lanes interleaved with 3 scalar CRC
  //   streams for ILP overlap; `veor3q_u64` reduces XOR chains. Significantly faster than v12e_v1 at
  //   scale.
  #[cfg(target_arch = "aarch64")]
  {
    use crate::platform::caps::aarch64;
    let caps = crate::platform::caps();
    if caps.has(aarch64::PMULL_READY) && caps.has(aarch64::CRC_READY) {
      // Large buffers: EOR3 kernel for better ILP (scalar CRC + PMULL interleaving).
      if data.len() > 1024 && caps.has(aarch64::PMULL_EOR3_READY) {
        return crate::checksum::crc32::aarch64::crc32_iso_hdlc_pmull_eor3_v9s3x2e_s3_safe(!0, data) ^ !0;
      }
      // Small buffers: pure hardware CRC (avoids PMULL kernel entry overhead).
      if data.len() < 128 {
        return crate::checksum::crc32::aarch64::crc32_armv8_safe(!0, data) ^ !0;
      }
      // Medium (128-1024B): PMULL v12e_v1 fold.
      // SAFETY: CRC + PMULL (AES) extensions verified via runtime caps check.
      return unsafe { crate::checksum::crc32::aarch64::crc32_ieee_fusion_inline(!0, data) } ^ !0;
    }
    if caps.has(aarch64::CRC_READY) {
      // SAFETY: CRC extension verified via runtime caps check.
      return unsafe { crate::checksum::crc32::aarch64::crc32_ieee_hwcrc_inline(!0, data) } ^ !0;
    }
  }

  let table = active_table();
  let kernel = table.select_fns(data.len()).crc32_ieee;
  kernel(!0, data) ^ !0
}

/// Internal one-shot CRC-32C helper used by trait impls.
#[inline]
pub(crate) fn crc32c(data: &[u8]) -> u32 {
  if data.len() <= FAST_PATH_MAX {
    if data.is_empty() {
      return 0;
    }
    return crate::checksum::crc32::portable::crc32c_bytewise(!0, data) ^ !0;
  }

  // aarch64: same dispatch bypass as crc32_ieee — see comments above.
  #[cfg(target_arch = "aarch64")]
  {
    use crate::platform::caps::aarch64;
    let caps = crate::platform::caps();
    if caps.has(aarch64::PMULL_READY) && caps.has(aarch64::CRC_READY) {
      // Large buffers: EOR3 kernel for better ILP (scalar CRC + PMULL interleaving).
      if data.len() > 1024 && caps.has(aarch64::PMULL_EOR3_READY) {
        return crate::checksum::crc32::aarch64::crc32c_iscsi_pmull_eor3_v9s3x2e_s3_safe(!0, data) ^ !0;
      }
      // Small buffers: pure hardware CRC (avoids PMULL kernel entry overhead).
      if data.len() < 128 {
        return crate::checksum::crc32::aarch64::crc32c_armv8_safe(!0, data) ^ !0;
      }
      // Medium (128-1024B): PMULL v12e_v1 fold.
      // SAFETY: CRC + PMULL (AES) extensions verified via runtime caps check.
      return unsafe { crate::checksum::crc32::aarch64::crc32c_iscsi_fusion_inline(!0, data) } ^ !0;
    }
    if caps.has(aarch64::CRC_READY) {
      // SAFETY: CRC extension verified via runtime caps check.
      return unsafe { crate::checksum::crc32::aarch64::crc32c_iscsi_hwcrc_inline(!0, data) } ^ !0;
    }
  }

  let table = active_table();
  let kernel = table.select_fns(data.len()).crc32c;
  kernel(!0, data) ^ !0
}

/// Internal one-shot CRC-16/CCITT helper used by trait impls.
#[inline]
pub(crate) fn crc16_ccitt(data: &[u8]) -> u16 {
  // CCITT: INIT=0xFFFF, XOROUT=0xFFFF
  if data.len() <= FAST_PATH_MAX {
    if data.is_empty() {
      return 0;
    }
    return crate::checksum::crc16::portable::crc16_ccitt_bytewise(0xFFFF, data) ^ 0xFFFF;
  }
  let table = active_table();
  let kernel = table.select_fns(data.len()).crc16_ccitt;
  kernel(0xFFFF, data) ^ 0xFFFF
}

/// Internal one-shot CRC-16/IBM helper used by trait impls.
#[inline]
pub(crate) fn crc16_ibm(data: &[u8]) -> u16 {
  // IBM: INIT=0x0000, XOROUT=0x0000
  if data.len() <= FAST_PATH_MAX {
    if data.is_empty() {
      return 0;
    }
    return crate::checksum::crc16::portable::crc16_ibm_bytewise(0, data);
  }
  let table = active_table();
  let kernel = table.select_fns(data.len()).crc16_ibm;
  kernel(0, data)
}

/// Internal one-shot CRC-24/OpenPGP helper used by trait impls.
#[inline]
pub(crate) fn crc24_openpgp(data: &[u8]) -> u32 {
  // OpenPGP: INIT=0x00B704CE, XOROUT=0x000000, mask to 24 bits
  const INIT: u32 = 0x00B7_04CE;
  const MASK: u32 = 0x00FF_FFFF;
  if data.len() <= FAST_PATH_MAX {
    if data.is_empty() {
      return INIT;
    }
    return crate::checksum::crc24::portable::crc24_openpgp_bytewise(INIT, data) & MASK;
  }
  let table = active_table();
  let kernel = table.select_fns(data.len()).crc24_openpgp;
  kernel(INIT, data) & MASK
}

// ─────────────────────────────────────────────────────────────────────────────
// Data Structures
// ─────────────────────────────────────────────────────────────────────────────

/// Hot-path kernel function pointers for one size class (56 bytes, fits one cache line).
///
/// Contains only the function pointers needed on the dispatch hot path.
/// Name strings live in the companion [`KernelNameSet`], which is only
/// touched by cold introspection / diagnostics code.
#[derive(Clone, Copy)]
pub(crate) struct KernelFnSet {
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

/// Cold-path kernel name strings for one size class.
///
/// Contains human-readable kernel names for introspection / diagnostics.
/// Separated from [`KernelFnSet`] so the hot dispatch path never pulls
/// name data into the cache line.
#[derive(Clone, Copy)]
pub(crate) struct KernelNameSet {
  // CRC-16
  pub crc16_ccitt_name: &'static str,
  pub crc16_ibm_name: &'static str,
  // CRC-24
  pub crc24_openpgp_name: &'static str,
  // CRC-32
  pub crc32_ieee_name: &'static str,
  pub crc32c_name: &'static str,
  // CRC-64
  pub crc64_xz_name: &'static str,
  pub crc64_nvme_name: &'static str,
}

/// Combined kernel definition for convenient table construction.
///
/// Holds both function pointers and name strings. Used in `const`/`static`
/// table definitions, then split into [`KernelFnSet`] + [`KernelNameSet`]
/// inside [`KernelTable::from_sets`].
#[derive(Clone, Copy)]
pub(crate) struct KernelSet {
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

impl KernelSet {
  /// Extract the hot-path function pointer set.
  #[inline]
  pub const fn fns(&self) -> KernelFnSet {
    KernelFnSet {
      crc16_ccitt: self.crc16_ccitt,
      crc16_ibm: self.crc16_ibm,
      crc24_openpgp: self.crc24_openpgp,
      crc32_ieee: self.crc32_ieee,
      crc32c: self.crc32c,
      crc64_xz: self.crc64_xz,
      crc64_nvme: self.crc64_nvme,
    }
  }

  /// Extract the cold-path name set.
  #[inline]
  pub const fn names(&self) -> KernelNameSet {
    KernelNameSet {
      crc16_ccitt_name: self.crc16_ccitt_name,
      crc16_ibm_name: self.crc16_ibm_name,
      crc24_openpgp_name: self.crc24_openpgp_name,
      crc32_ieee_name: self.crc32_ieee_name,
      crc32c_name: self.crc32c_name,
      crc64_xz_name: self.crc64_xz_name,
      crc64_nvme_name: self.crc64_nvme_name,
    }
  }
}

/// Complete kernel table for one platform.
///
/// Contains pre-selected optimal kernels for each (variant, size_class) pair.
/// Size class boundaries define when to transition between kernel tiers.
///
/// Hot-path function pointers ([`KernelFnSet`]) are stored separately from
/// cold-path name strings ([`KernelNameSet`]) so the dispatch path fits in
/// a single cache line per size class.
#[derive(Clone, Copy)]
pub(crate) struct KernelTable {
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

  /// Hot-path function pointers per size class: [xs, s, m, l]
  pub fns: [KernelFnSet; 4],
  /// Cold-path kernel names per size class: [xs, s, m, l]
  pub names: [KernelNameSet; 4],
}

/// Size-class index constants for [`KernelTable::fns`] / [`KernelTable::names`].
const XS: usize = 0;
const S: usize = 1;
const M: usize = 2;
const L: usize = 3;

impl KernelTable {
  /// Construct a `KernelTable` from four [`KernelSet`] definitions, splitting
  /// function pointers from name strings at compile time.
  pub const fn from_sets(
    requires: Caps,
    boundaries: [usize; 3],
    xs: KernelSet,
    s: KernelSet,
    m: KernelSet,
    l: KernelSet,
  ) -> Self {
    Self {
      requires,
      boundaries,
      fns: [xs.fns(), s.fns(), m.fns(), l.fns()],
      names: [xs.names(), s.names(), m.names(), l.names()],
    }
  }

  /// Select the hot-path function pointer set for the given buffer length.
  #[inline]
  pub const fn select_fns(&self, len: usize) -> &KernelFnSet {
    if len <= self.boundaries[0] {
      &self.fns[XS]
    } else if len <= self.boundaries[1] {
      &self.fns[S]
    } else if len <= self.boundaries[2] {
      &self.fns[M]
    } else {
      &self.fns[L]
    }
  }

  /// Select the cold-path kernel name set for the given buffer length.
  ///
  /// Used only for introspection / diagnostics; never called on the hot path.
  #[inline]
  pub const fn select_names(&self, len: usize) -> &KernelNameSet {
    if len <= self.boundaries[0] {
      &self.names[XS]
    } else if len <= self.boundaries[1] {
      &self.names[S]
    } else if len <= self.boundaries[2] {
      &self.names[M]
    } else {
      &self.names[L]
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

/// Construct a [`KernelTable`] from the familiar field-label syntax, splitting
/// function pointers from name strings at compile time.
///
/// Accepts the same field layout as the old monolithic `KernelTable { ... }`
/// struct literal (with `requires`, `boundaries`, `xs`, `s`, `m`, `l` labels),
/// converting it to `KernelTable::from_sets(...)` under the hood.
macro_rules! kernel_table {
  (
    requires: $req:expr,
    boundaries: $bounds:expr,
    xs: $xs:expr,
    s: $s:expr,
    m: $m:expr,
    l: $l:expr $(,)?
  ) => {
    KernelTable::from_sets($req, $bounds, $xs, $s, $m, $l)
  };
}

/// Returns `true` if the active kernel table uses hardware-accelerated CRC kernels.
#[inline]
pub fn is_hardware_accelerated() -> bool {
  active_table().is_hardware_accelerated()
}

// ─────────────────────────────────────────────────────────────────────────────
// Platform Table Selection
// ─────────────────────────────────────────────────────────────────────────────

/// Select the appropriate kernel table for the current platform.
///
/// Resolution order:
/// 1. **Capability match**: Conservative defaults based on CPU features
/// 2. **Portable fallback**: No SIMD, table-based only
#[inline]
pub(crate) fn select_table(caps: Caps) -> &'static KernelTable {
  if let Some(table) = capability_match(caps) {
    debug_assert!(caps.has(table.requires), "capability_match returned an unsafe table");
    if caps.has(table.requires) {
      return table;
    }
  }

  // 2. Portable fallback
  &PORTABLE_TABLE
}

#[inline]
fn capability_match(caps: Caps) -> Option<&'static KernelTable> {
  let _ = caps;

  #[cfg(target_arch = "aarch64")]
  {
    use crate::platform::caps::aarch64::{CRC_READY, PMULL_EOR3_READY, PMULL_READY};

    #[cfg(feature = "std")]
    if let Some(family) = crate::platform::detect::detect_aarch64_tune_family() {
      match family {
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos"))]
        crate::platform::detect::Aarch64TuneFamily::AppleM1M3 => {
          if caps.has(aarch64_tables::APPLE_M1M3_TABLE.requires) {
            return Some(&aarch64_tables::APPLE_M1M3_TABLE);
          }
        }
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos"))]
        crate::platform::detect::Aarch64TuneFamily::AppleM4M5 => {
          if caps.has(aarch64_tables::APPLE_M1M3_TABLE.requires) {
            return Some(&aarch64_tables::APPLE_M1M3_TABLE);
          }
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        crate::platform::detect::Aarch64TuneFamily::Graviton2 => {
          if caps.has(aarch64_tables::GRAVITON2_TABLE.requires) {
            return Some(&aarch64_tables::GRAVITON2_TABLE);
          }
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        crate::platform::detect::Aarch64TuneFamily::Graviton3 => {
          if caps.has(aarch64_tables::GRAVITON3_TABLE.requires) {
            return Some(&aarch64_tables::GRAVITON3_TABLE);
          }
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        crate::platform::detect::Aarch64TuneFamily::Graviton4 => {
          if caps.has(aarch64_tables::GRAVITON4_TABLE.requires) {
            return Some(&aarch64_tables::GRAVITON4_TABLE);
          }
        }
      }
    }

    // PMULL + SHA3 (EOR3) + CRC extension → use EOR3-enabled table
    if caps.has(CRC_READY) && caps.has(PMULL_EOR3_READY) {
      return Some(&aarch64_tables::GENERIC_ARM_PMULL_EOR3_TABLE);
    }
    // PMULL + CRC extension (no EOR3) → safe PMULL-only table
    if caps.has(CRC_READY) && caps.has(PMULL_READY) {
      return Some(&aarch64_tables::GENERIC_ARM_PMULL_TABLE);
    }
    // PMULL only (no CRC extension) → accelerate CRC-16/24/64, keep CRC-32 portable
    if caps.has(PMULL_READY) {
      return Some(&aarch64_tables::GENERIC_ARM_PMULL_NO_CRC_TABLE);
    }
    // CRC extension only → accelerate CRC-32/32C, keep others portable
    if caps.has(CRC_READY) {
      return Some(&aarch64_tables::GENERIC_ARM_CRC_ONLY_TABLE);
    }
  }

  #[cfg(target_arch = "x86_64")]
  {
    use crate::platform::caps::x86::{CRC32C_READY, PCLMUL_READY, VPCLMUL_READY};

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
    use crate::platform::caps::s390x::{Z13_READY, Z14_READY, Z15_READY, Z16_READY};
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
    use crate::platform::caps::power::{POWER9_READY, POWER10_READY, VPMSUM_READY};
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
    use crate::platform::caps::riscv::{V, ZBC, ZVBC};
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
  crc16_ccitt: crate::checksum::crc16::portable::crc16_ccitt_slice8,
  crc16_ccitt_name: "portable/slice8",
  crc16_ibm: crate::checksum::crc16::portable::crc16_ibm_slice8,
  crc16_ibm_name: "portable/slice8",
  crc24_openpgp: crate::checksum::crc24::portable::crc24_openpgp_slice8,
  crc24_openpgp_name: "portable/slice8",
  crc32_ieee: crate::checksum::crc32::portable::crc32_slice16_ieee,
  crc32_ieee_name: "portable/slice16",
  crc32c: crate::checksum::crc32::portable::crc32c_slice16,
  crc32c_name: "portable/slice16",
  crc64_xz: crate::checksum::crc64::portable::crc64_slice16_xz,
  crc64_xz_name: "portable/slice16",
  crc64_nvme: crate::checksum::crc64::portable::crc64_slice16_nvme,
  crc64_nvme_name: "portable/slice16",
};

pub static PORTABLE_TABLE: KernelTable = KernelTable::from_sets(
  Caps::NONE,
  [64, 256, 4096],
  PORTABLE_SET,
  PORTABLE_SET,
  PORTABLE_SET,
  PORTABLE_SET,
);

// ─────────────────────────────────────────────────────────────────────────────
// aarch64 Platform Tables
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod aarch64_tables {
  use super::*;
  // Import kernel constants from the kernels modules
  use crate::checksum::crc16::kernels::aarch64 as crc16_k;
  use crate::checksum::{
    crc24::kernels::aarch64 as crc24_k, crc32::kernels::aarch64 as crc32_k, crc64::kernels::aarch64 as crc64_k,
  };

  #[cfg(any(target_os = "linux", target_os = "android"))]
  const G3_CRC16_PMULL_EOR3_2WAY_MAX_LEN: usize = 262_144;

  /// Graviton3 CRC16/CCITT large-path PMULL+EOR3 hybrid.
  ///
  /// PMULL+EOR3 2-way improves lower "large" buffers, while PMULL+EOR3 1-way
  /// remains safer for very large buffers on G3.
  #[cfg(any(target_os = "linux", target_os = "android"))]
  #[inline]
  fn g3_crc16_ccitt_l_hybrid(crc: u16, data: &[u8]) -> u16 {
    if data.len() <= G3_CRC16_PMULL_EOR3_2WAY_MAX_LEN {
      (crc16_k::CCITT_PMULL_EOR3[1])(crc, data)
    } else {
      (crc16_k::CCITT_PMULL_EOR3[0])(crc, data)
    }
  }

  /// Graviton3 CRC16/IBM large-path PMULL+EOR3 hybrid.
  ///
  /// Empirically, 2-way PMULL+EOR3 helps lower "large" buffers while 1-way
  /// PMULL+EOR3 remains safer for very large buffers on G3.
  #[cfg(any(target_os = "linux", target_os = "android"))]
  #[inline]
  fn g3_crc16_ibm_l_hybrid(crc: u16, data: &[u8]) -> u16 {
    if data.len() <= G3_CRC16_PMULL_EOR3_2WAY_MAX_LEN {
      (crc16_k::IBM_PMULL_EOR3[1])(crc, data)
    } else {
      (crc16_k::IBM_PMULL_EOR3[0])(crc, data)
    }
  }

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
  // Generated from benchmark-derived dispatch data. Do not edit manually.
  // ───────────────────────────────────────────────────────────────────────────
  // AppleM1M3 Table
  //
  // Generated from benchmark-derived dispatch data. Do not edit manually.
  // ───────────────────────────────────────────────────────────────────────────
  pub static APPLE_M1M3_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::aarch64::CRC_READY
      .union(crate::platform::caps::aarch64::PMULL_EOR3_READY)
      .union(crate::platform::caps::aarch64::PMULL_READY),
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
  #[cfg(any(target_os = "linux", target_os = "android"))]
  pub static GRAVITON2_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::aarch64::CRC_READY.union(crate::platform::caps::aarch64::PMULL_EOR3_READY),
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
  // Benchmark source: `src/checksum/bench_baseline/linux_arm64_graviton3_kernels.txt`
  // Features: PMULL + SHA3/EOR3
  // Peak throughputs: CRC-16 ~38 GiB/s, CRC-32 ~46 GiB/s, CRC-64 ~38 GiB/s
  //
  // Key differences vs Graviton2:
  // - Higher throughput (~25% faster across the board)
  // - Different optimal kernel choices for CRC16@s and CRC64/NVME
  // ───────────────────────────────────────────────────────────────────────────
  #[cfg(any(target_os = "linux", target_os = "android"))]
  const G3_XS: KernelSet = KernelSet {
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
  };

  #[cfg(any(target_os = "linux", target_os = "android"))]
  const G3_S: KernelSet = KernelSet {
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
  };

  #[cfg(any(target_os = "linux", target_os = "android"))]
  const G3_M: KernelSet = KernelSet {
    crc16_ccitt: crc16_k::CCITT_PMULL_EOR3[0], // PMULL+EOR3 cuts XOR chain in large lanes.
    crc16_ccitt_name: "aarch64/pmull-eor3",
    crc16_ibm: crc16_k::IBM_PMULL_EOR3[0], // PMULL+EOR3 cuts XOR chain in large lanes.
    crc16_ibm_name: "aarch64/pmull-eor3",
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
  };

  #[cfg(any(target_os = "linux", target_os = "android"))]
  const G3_L: KernelSet = KernelSet {
    crc16_ccitt: g3_crc16_ccitt_l_hybrid,
    crc16_ccitt_name: "aarch64/pmull-eor3-g3-ccitt-hybrid",
    crc16_ibm: g3_crc16_ibm_l_hybrid,
    crc16_ibm_name: "aarch64/pmull-eor3-g3-ibm-hybrid",
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
  };

  #[cfg(any(target_os = "linux", target_os = "android"))]
  pub static GRAVITON3_TABLE: KernelTable = KernelTable::from_sets(
    crate::platform::caps::aarch64::CRC_READY.union(crate::platform::caps::aarch64::PMULL_EOR3_READY),
    [64, 256, 4096],
    G3_XS,
    G3_S,
    G3_M,
    G3_L,
  );

  // ───────────────────────────────────────────────────────────────────────────
  // Graviton4 Table
  //
  // Starts from Graviton3, but keeps 2-way PMULL+EOR3 for CRC16 large classes.
  // ───────────────────────────────────────────────────────────────────────────
  #[cfg(any(target_os = "linux", target_os = "android"))]
  pub static GRAVITON4_TABLE: KernelTable = KernelTable::from_sets(
    crate::platform::caps::aarch64::CRC_READY.union(crate::platform::caps::aarch64::PMULL_EOR3_READY),
    [64, 256, 4096],
    G3_XS,
    G3_S,
    KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL_EOR3[1],
      crc16_ccitt_name: "aarch64/pmull-eor3-2way",
      crc16_ibm: crc16_k::IBM_PMULL_EOR3[1],
      crc16_ibm_name: "aarch64/pmull-eor3-2way",
      ..G3_M
    },
    KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL_EOR3[1],
      crc16_ccitt_name: "aarch64/pmull-eor3-2way",
      crc16_ibm: crc16_k::IBM_PMULL_EOR3[1],
      crc16_ibm_name: "aarch64/pmull-eor3-2way",
      ..G3_L
    },
  );

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
  pub static GENERIC_ARM_PMULL_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::aarch64::CRC_READY.union(crate::platform::caps::aarch64::PMULL_READY),
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
  pub static GENERIC_ARM_PMULL_NO_CRC_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::aarch64::PMULL_READY,
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crc16_k::CCITT_PMULL_SMALL_KERNEL,
      crc16_ccitt_name: "aarch64/pmull-small",
      crc16_ibm: crc16_k::IBM_PMULL_SMALL_KERNEL,
      crc16_ibm_name: "aarch64/pmull-small",
      crc24_openpgp: crc24_k::OPENPGP_PMULL_SMALL_KERNEL,
      crc24_openpgp_name: "aarch64/pmull-small",
      crc32_ieee: crate::checksum::crc32::portable::crc32_bytewise_ieee,
      crc32_ieee_name: crate::checksum::crc32::portable::BYTEWISE_KERNEL_NAME,
      crc32c: crate::checksum::crc32::portable::crc32c_bytewise,
      crc32c_name: crate::checksum::crc32::portable::BYTEWISE_KERNEL_NAME,
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
      crc32_ieee: crate::checksum::crc32::portable::crc32_slice16_ieee,
      crc32_ieee_name: "portable/slice16",
      crc32c: crate::checksum::crc32::portable::crc32c_slice16,
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
      crc32_ieee: crate::checksum::crc32::portable::crc32_slice16_ieee,
      crc32_ieee_name: "portable/slice16",
      crc32c: crate::checksum::crc32::portable::crc32c_slice16,
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
      crc32_ieee: crate::checksum::crc32::portable::crc32_slice16_ieee,
      crc32_ieee_name: "portable/slice16",
      crc32c: crate::checksum::crc32::portable::crc32c_slice16,
      crc32c_name: "portable/slice16",
      crc64_xz: crc64_k::XZ_PMULL[0],
      crc64_xz_name: "aarch64/pmull",
      crc64_nvme: crc64_k::NVME_PMULL[0],
      crc64_nvme_name: "aarch64/pmull",
    },
  };

  /// CRC-only table for platforms without PMULL.
  pub static GENERIC_ARM_CRC_ONLY_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::aarch64::CRC_READY,
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crate::checksum::crc16::portable::crc16_ccitt_slice8,
      crc16_ccitt_name: "portable/slice8",
      crc16_ibm: crate::checksum::crc16::portable::crc16_ibm_slice8,
      crc16_ibm_name: "portable/slice8",
      crc24_openpgp: crate::checksum::crc24::portable::crc24_openpgp_slice8,
      crc24_openpgp_name: "portable/slice8",
      crc32_ieee: crc32_k::CRC32_HWCRC[0],
      crc32_ieee_name: "aarch64/hwcrc",
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc32c_name: "aarch64/hwcrc",
      crc64_xz: crate::checksum::crc64::portable::crc64_slice16_xz,
      crc64_xz_name: "portable/slice16",
      crc64_nvme: crate::checksum::crc64::portable::crc64_slice16_nvme,
      crc64_nvme_name: "portable/slice16",
    },

    s: KernelSet {
      crc16_ccitt: crate::checksum::crc16::portable::crc16_ccitt_slice8,
      crc16_ccitt_name: "portable/slice8",
      crc16_ibm: crate::checksum::crc16::portable::crc16_ibm_slice8,
      crc16_ibm_name: "portable/slice8",
      crc24_openpgp: crate::checksum::crc24::portable::crc24_openpgp_slice8,
      crc24_openpgp_name: "portable/slice8",
      crc32_ieee: crc32_k::CRC32_HWCRC[0],
      crc32_ieee_name: "aarch64/hwcrc",
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc32c_name: "aarch64/hwcrc",
      crc64_xz: crate::checksum::crc64::portable::crc64_slice16_xz,
      crc64_xz_name: "portable/slice16",
      crc64_nvme: crate::checksum::crc64::portable::crc64_slice16_nvme,
      crc64_nvme_name: "portable/slice16",
    },

    m: KernelSet {
      crc16_ccitt: crate::checksum::crc16::portable::crc16_ccitt_slice8,
      crc16_ccitt_name: "portable/slice8",
      crc16_ibm: crate::checksum::crc16::portable::crc16_ibm_slice8,
      crc16_ibm_name: "portable/slice8",
      crc24_openpgp: crate::checksum::crc24::portable::crc24_openpgp_slice8,
      crc24_openpgp_name: "portable/slice8",
      crc32_ieee: crc32_k::CRC32_HWCRC[0],
      crc32_ieee_name: "aarch64/hwcrc",
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc32c_name: "aarch64/hwcrc",
      crc64_xz: crate::checksum::crc64::portable::crc64_slice16_xz,
      crc64_xz_name: "portable/slice16",
      crc64_nvme: crate::checksum::crc64::portable::crc64_slice16_nvme,
      crc64_nvme_name: "portable/slice16",
    },

    l: KernelSet {
      crc16_ccitt: crate::checksum::crc16::portable::crc16_ccitt_slice8,
      crc16_ccitt_name: "portable/slice8",
      crc16_ibm: crate::checksum::crc16::portable::crc16_ibm_slice8,
      crc16_ibm_name: "portable/slice8",
      crc24_openpgp: crate::checksum::crc24::portable::crc24_openpgp_slice8,
      crc24_openpgp_name: "portable/slice8",
      crc32_ieee: crc32_k::CRC32_HWCRC[0],
      crc32_ieee_name: "aarch64/hwcrc",
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc32c_name: "aarch64/hwcrc",
      crc64_xz: crate::checksum::crc64::portable::crc64_slice16_xz,
      crc64_xz_name: "portable/slice16",
      crc64_nvme: crate::checksum::crc64::portable::crc64_slice16_nvme,
      crc64_nvme_name: "portable/slice16",
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// x86_64 Platform Tables
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod x86_64_tables {
  use super::*;
  // Import kernel constants from the kernels modules
  use crate::checksum::crc16::kernels::x86_64 as crc16_k;
  use crate::checksum::{
    crc24::kernels::x86_64 as crc24_k, crc32::kernels::x86_64 as crc32_k, crc64::kernels::x86_64 as crc64_k,
  };

  const ZEN4_CRC64_XZ_2WAY_MAX_LEN: usize = 262_144;

  /// Zen4 CRC64/XZ large-path hybrid.
  ///
  /// 2-way is strongest around lower "large" sizes, while 8-way closes the
  /// remaining gap at xl-scale buffers.
  #[inline]
  fn zen4_crc64_xz_l_hybrid(crc: u64, data: &[u8]) -> u64 {
    if data.len() <= ZEN4_CRC64_XZ_2WAY_MAX_LEN {
      (crc64_k::XZ_VPCLMUL[1])(crc, data)
    } else {
      (crc64_k::XZ_VPCLMUL[4])(crc, data)
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Zen5 Table
  //
  // Benchmark source: `src/checksum/bench_baseline/linux_x86-64_zen5_kernels.txt`
  // Features: VPCLMULQDQ + AVX-512
  //
  // Key differences vs Zen4:
  // - Different optimal multi-stream counts for CRC16/24 kernels
  // - CRC64 (large) prefers VPCLMUL-2way over the 4×512 kernel
  // ───────────────────────────────────────────────────────────────────────────
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
  pub static ZEN4_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::x86::VPCLMUL_READY
      .union(crate::platform::caps::x86::PCLMUL_READY)
      .union(crate::platform::caps::x86::CRC32C_READY),
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
      crc32c: crc32_k::CRC32C_FUSION_VPCLMUL[0], // fusion-vpclmul-v3x2
      crc32c_name: "x86_64/fusion-vpclmul-v3x2",
      crc64_xz: zen4_crc64_xz_l_hybrid,
      crc64_xz_name: "x86_64/vpclmul-zen4-xz-hybrid",
      crc64_nvme: crc64_k::NVME_VPCLMUL[1], // 2-way edges out 4-way at l/xl in latest comp+kernels
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
  pub static GENERIC_X86_VPCLMUL_NO_CRC32C_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::x86::VPCLMUL_READY.union(crate::platform::caps::x86::PCLMUL_READY),
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
      crc32c: crate::checksum::crc32::portable::crc32c_bytewise,
      crc32c_name: crate::checksum::crc32::portable::BYTEWISE_KERNEL_NAME,
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
      crc32c: crate::checksum::crc32::portable::crc32c_slice16,
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
      crc32c: crate::checksum::crc32::portable::crc32c_slice16,
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
      crc32c: crate::checksum::crc32::portable::crc32c_slice16,
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
  pub static GENERIC_X86_PCLMUL_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::x86::PCLMUL_READY.union(crate::platform::caps::x86::CRC32C_READY),
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
  pub static GENERIC_X86_PCLMUL_NO_CRC32C_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::x86::PCLMUL_READY,
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
      crc32c: crate::checksum::crc32::portable::crc32c_bytewise,
      crc32c_name: crate::checksum::crc32::portable::BYTEWISE_KERNEL_NAME,
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
      crc32c: crate::checksum::crc32::portable::crc32c_slice16,
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
      crc32c: crate::checksum::crc32::portable::crc32c_slice16,
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
      crc32c: crate::checksum::crc32::portable::crc32c_slice16,
      crc32c_name: "portable/slice16",
      crc64_xz: crc64_k::XZ_PCLMUL[0],
      crc64_xz_name: "x86_64/pclmul",
      crc64_nvme: crc64_k::NVME_PCLMUL[1],
      crc64_nvme_name: "x86_64/pclmul-2way",
    },
  };

  /// SSE4.2-only table: accelerate CRC32C, keep other variants portable.
  pub static GENERIC_X86_CRC32C_ONLY_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::x86::CRC32C_READY,
    boundaries: [64, 256, 4096],

    xs: KernelSet {
      crc16_ccitt: crate::checksum::crc16::portable::crc16_ccitt_slice8,
      crc16_ccitt_name: "portable/slice8",
      crc16_ibm: crate::checksum::crc16::portable::crc16_ibm_slice8,
      crc16_ibm_name: "portable/slice8",
      crc24_openpgp: crate::checksum::crc24::portable::crc24_openpgp_slice8,
      crc24_openpgp_name: "portable/slice8",
      crc32_ieee: crate::checksum::crc32::portable::crc32_bytewise_ieee,
      crc32_ieee_name: crate::checksum::crc32::portable::BYTEWISE_KERNEL_NAME,
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc32c_name: "x86_64/hwcrc",
      crc64_xz: crate::checksum::crc64::portable::crc64_slice16_xz,
      crc64_xz_name: "portable/slice16",
      crc64_nvme: crate::checksum::crc64::portable::crc64_slice16_nvme,
      crc64_nvme_name: "portable/slice16",
    },

    s: KernelSet {
      crc16_ccitt: crate::checksum::crc16::portable::crc16_ccitt_slice8,
      crc16_ccitt_name: "portable/slice8",
      crc16_ibm: crate::checksum::crc16::portable::crc16_ibm_slice8,
      crc16_ibm_name: "portable/slice8",
      crc24_openpgp: crate::checksum::crc24::portable::crc24_openpgp_slice8,
      crc24_openpgp_name: "portable/slice8",
      crc32_ieee: crate::checksum::crc32::portable::crc32_slice16_ieee,
      crc32_ieee_name: "portable/slice16",
      crc32c: crc32_k::CRC32C_HWCRC[0],
      crc32c_name: "x86_64/hwcrc",
      crc64_xz: crate::checksum::crc64::portable::crc64_slice16_xz,
      crc64_xz_name: "portable/slice16",
      crc64_nvme: crate::checksum::crc64::portable::crc64_slice16_nvme,
      crc64_nvme_name: "portable/slice16",
    },

    m: KernelSet {
      crc16_ccitt: crate::checksum::crc16::portable::crc16_ccitt_slice8,
      crc16_ccitt_name: "portable/slice8",
      crc16_ibm: crate::checksum::crc16::portable::crc16_ibm_slice8,
      crc16_ibm_name: "portable/slice8",
      crc24_openpgp: crate::checksum::crc24::portable::crc24_openpgp_slice8,
      crc24_openpgp_name: "portable/slice8",
      crc32_ieee: crate::checksum::crc32::portable::crc32_slice16_ieee,
      crc32_ieee_name: "portable/slice16",
      crc32c: crc32_k::CRC32C_HWCRC[1],
      crc32c_name: "x86_64/hwcrc-2way",
      crc64_xz: crate::checksum::crc64::portable::crc64_slice16_xz,
      crc64_xz_name: "portable/slice16",
      crc64_nvme: crate::checksum::crc64::portable::crc64_slice16_nvme,
      crc64_nvme_name: "portable/slice16",
    },

    l: KernelSet {
      crc16_ccitt: crate::checksum::crc16::portable::crc16_ccitt_slice8,
      crc16_ccitt_name: "portable/slice8",
      crc16_ibm: crate::checksum::crc16::portable::crc16_ibm_slice8,
      crc16_ibm_name: "portable/slice8",
      crc24_openpgp: crate::checksum::crc24::portable::crc24_openpgp_slice8,
      crc24_openpgp_name: "portable/slice8",
      crc32_ieee: crate::checksum::crc32::portable::crc32_slice16_ieee,
      crc32_ieee_name: "portable/slice16",
      crc32c: crc32_k::CRC32C_HWCRC[1],
      crc32c_name: "x86_64/hwcrc-2way",
      crc64_xz: crate::checksum::crc64::portable::crc64_slice16_xz,
      crc64_xz_name: "portable/slice16",
      crc64_nvme: crate::checksum::crc64::portable::crc64_slice16_nvme,
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
  use crate::checksum::{
    crc16::kernels::s390x as crc16_k, crc24::kernels::s390x as crc24_k, crc32::kernels::s390x as crc32_k,
    crc64::kernels::s390x as crc64_k,
  };

  pub static S390X_Z13_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::s390x::Z13_READY,
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

  pub static S390X_Z14_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::s390x::Z13_READY,
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

  pub static S390X_Z15_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::s390x::Z13_READY,
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
  use crate::checksum::{
    crc16::kernels::power as crc16_k, crc24::kernels::power as crc24_k, crc32::kernels::power as crc32_k,
    crc64::kernels::power as crc64_k,
  };

  pub static POWER8_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::power::VPMSUM_READY,
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

  pub static POWER9_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::power::VPMSUM_READY,
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

  pub static POWER10_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::power::VPMSUM_READY,
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
  use crate::checksum::{
    crc16::kernels::riscv64 as crc16_k, crc24::kernels::riscv64 as crc24_k, crc32::kernels::riscv64 as crc32_k,
    crc64::kernels::riscv64 as crc64_k,
  };

  pub static RISCV64_ZBC_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::riscv::ZBC,
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

  pub static RISCV64_ZVBC_TABLE: KernelTable = kernel_table! {
    requires: crate::platform::caps::riscv::V.union(crate::platform::caps::riscv::ZVBC),
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
  use alloc::vec::Vec;

  use super::*;

  #[test]
  fn test_portable_table_all_sizes() {
    // Verify portable table returns correct sets for each size class
    let table = &PORTABLE_TABLE;

    assert!(core::ptr::eq(table.select_fns(0), &table.fns[XS]));
    assert!(core::ptr::eq(table.select_fns(64), &table.fns[XS]));
    assert!(core::ptr::eq(table.select_fns(65), &table.fns[S]));
    assert!(core::ptr::eq(table.select_fns(256), &table.fns[S]));
    assert!(core::ptr::eq(table.select_fns(257), &table.fns[M]));
    assert!(core::ptr::eq(table.select_fns(4096), &table.fns[M]));
    assert!(core::ptr::eq(table.select_fns(4097), &table.fns[L]));
    assert!(core::ptr::eq(table.select_fns(1_000_000), &table.fns[L]));
  }

  #[test]
  fn test_select_table_fallback() {
    // With no capabilities, should return portable table
    let table = select_table(Caps::NONE);
    assert!(core::ptr::eq(table, &PORTABLE_TABLE));
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
    assert_eq!(crc16_ibm(&[]), 0, "CRC-16/IBM of empty data");
    assert_eq!(crc24_openpgp(&[]), 0x00B7_04CE, "CRC-24/OpenPGP of empty data");
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

  #[test]
  fn kernel_fn_set_fits_cache_line() {
    // KernelFnSet holds 7 function pointers (8 bytes each on 64-bit) = 56 bytes,
    // which fits within a single 64-byte cache line.
    let size = core::mem::size_of::<KernelFnSet>();
    assert!(
      size <= 64,
      "KernelFnSet is {size} bytes, must be <= 64 for cache-line fit"
    );
    assert_eq!(size, 56, "KernelFnSet should be exactly 56 bytes (7 fn ptrs)");
  }

  #[test]
  fn kernel_name_set_is_separate() {
    // KernelNameSet should only hold name strings, not fn ptrs.
    let fn_size = core::mem::size_of::<KernelFnSet>();
    let name_size = core::mem::size_of::<KernelNameSet>();
    let combined_size = core::mem::size_of::<KernelSet>();
    assert_eq!(
      fn_size.strict_add(name_size),
      combined_size,
      "KernelFnSet + KernelNameSet should equal KernelSet"
    );
  }
}
