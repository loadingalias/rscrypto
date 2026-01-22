//! Tuned dispatch tables for BLAKE3.
//!
//! This module is the integration point for `rscrypto-tune --apply`.
//! The tuning engine updates the per-`TuneKind` tables in this file.

use platform::TuneKind;

pub use super::kernels::Blake3KernelId as KernelId;

pub const DEFAULT_BOUNDARIES: [usize; 3] = [64, 256, 4096];

// A conservative "best available" SIMD kernel per target architecture.
//
// The runtime dispatcher (`dispatch::resolve`) will still validate CPU feature
// availability and fall back to Portable when needed.
#[cfg(target_arch = "x86_64")]
const SIMD_KERNEL: KernelId = KernelId::X86Avx2;
#[cfg(target_arch = "aarch64")]
const SIMD_KERNEL: KernelId = KernelId::Aarch64Neon;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const SIMD_KERNEL: KernelId = KernelId::Portable;

// AVX-512 is not always the best default on x86_64 (notably on some AMD CPUs),
// so we only opt into it for tune kinds where it is known to win.
#[cfg(target_arch = "x86_64")]
const AVX512_KERNEL: KernelId = KernelId::X86Avx512;

const DEFAULT_XS: KernelId = KernelId::Portable;
const DEFAULT_S: KernelId = KernelId::Portable;
const DEFAULT_M: KernelId = SIMD_KERNEL;
const DEFAULT_L: KernelId = SIMD_KERNEL;

#[inline]
#[must_use]
const fn default_kind_table() -> DispatchTable {
  DispatchTable {
    boundaries: DEFAULT_BOUNDARIES,
    xs: DEFAULT_XS,
    s: DEFAULT_S,
    m: DEFAULT_M,
    l: DEFAULT_L,
  }
}

#[derive(Clone, Copy, Debug)]
pub struct DispatchTable {
  pub boundaries: [usize; 3],
  pub xs: KernelId,
  pub s: KernelId,
  pub m: KernelId,
  pub l: KernelId,
}

impl DispatchTable {
  #[inline]
  #[must_use]
  pub const fn kernel_for_len(&self, len: usize) -> KernelId {
    let [xs_max, s_max, m_max] = self.boundaries;
    if len <= xs_max {
      self.xs
    } else if len <= s_max {
      self.s
    } else if len <= m_max {
      self.m
    } else {
      self.l
    }
  }
}

pub static DEFAULT_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Portable,
  s: KernelId::Portable,
  m: KernelId::Portable,
  l: KernelId::Portable,
};

// Custom Table
pub static CUSTOM_TABLE: DispatchTable = DEFAULT_TABLE;
// Default Table
pub static DEFAULT_KIND_TABLE: DispatchTable = default_kind_table();
// Portable Table
pub static PORTABLE_TABLE: DispatchTable = DEFAULT_TABLE;
// Zen4 Table
#[cfg(target_arch = "x86_64")]
pub static ZEN4_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: DEFAULT_XS,
  s: DEFAULT_S,
  m: SIMD_KERNEL,
  // Zen4 can be slower with AVX-512 (frequency/µarch effects). Keep AVX2 as
  // the default until per-runner tuning proves otherwise.
  l: SIMD_KERNEL,
};
#[cfg(not(target_arch = "x86_64"))]
pub static ZEN4_TABLE: DispatchTable = default_kind_table();
// Zen5 Table
#[cfg(target_arch = "x86_64")]
pub static ZEN5_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: DEFAULT_XS,
  s: DEFAULT_S,
  m: SIMD_KERNEL,
  l: AVX512_KERNEL,
};
#[cfg(not(target_arch = "x86_64"))]
pub static ZEN5_TABLE: DispatchTable = default_kind_table();
// Zen5c Table
#[cfg(target_arch = "x86_64")]
pub static ZEN5C_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: DEFAULT_XS,
  s: DEFAULT_S,
  m: SIMD_KERNEL,
  l: AVX512_KERNEL,
};
#[cfg(not(target_arch = "x86_64"))]
pub static ZEN5C_TABLE: DispatchTable = default_kind_table();
// IntelSpr Table
#[cfg(target_arch = "x86_64")]
pub static INTELSPR_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: DEFAULT_XS,
  s: DEFAULT_S,
  m: AVX512_KERNEL,
  l: AVX512_KERNEL,
};
#[cfg(not(target_arch = "x86_64"))]
pub static INTELSPR_TABLE: DispatchTable = default_kind_table();
// IntelGnr Table
#[cfg(target_arch = "x86_64")]
pub static INTELGNR_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: DEFAULT_XS,
  s: DEFAULT_S,
  m: AVX512_KERNEL,
  l: AVX512_KERNEL,
};
#[cfg(not(target_arch = "x86_64"))]
pub static INTELGNR_TABLE: DispatchTable = default_kind_table();
// IntelIcl Table
pub static INTELICL_TABLE: DispatchTable = default_kind_table();
// AppleM1M3 Table
pub static APPLEM1M3_TABLE: DispatchTable = default_kind_table();
// AppleM4 Table
pub static APPLEM4_TABLE: DispatchTable = default_kind_table();
// AppleM5 Table
pub static APPLEM5_TABLE: DispatchTable = default_kind_table();
// Graviton2 Table
pub static GRAVITON2_TABLE: DispatchTable = default_kind_table();
// Graviton3 Table
pub static GRAVITON3_TABLE: DispatchTable = default_kind_table();
// Graviton4 Table
pub static GRAVITON4_TABLE: DispatchTable = default_kind_table();
// Graviton5 Table
pub static GRAVITON5_TABLE: DispatchTable = default_kind_table();
// NeoverseN2 Table
pub static NEOVERSEN2_TABLE: DispatchTable = default_kind_table();
// NeoverseN3 Table
pub static NEOVERSEN3_TABLE: DispatchTable = default_kind_table();
// NeoverseV3 Table
pub static NEOVERSEV3_TABLE: DispatchTable = default_kind_table();
// NvidiaGrace Table
pub static NVIDIAGRACE_TABLE: DispatchTable = default_kind_table();
// AmpereAltra Table
pub static AMPEREALTRA_TABLE: DispatchTable = default_kind_table();
// Aarch64Pmull Table
pub static AARCH64PMULL_TABLE: DispatchTable = default_kind_table();
// Z13 Table
pub static Z13_TABLE: DispatchTable = default_kind_table();
// Z14 Table
pub static Z14_TABLE: DispatchTable = default_kind_table();
// Z15 Table
pub static Z15_TABLE: DispatchTable = default_kind_table();
// Power7 Table
pub static POWER7_TABLE: DispatchTable = default_kind_table();
// Power8 Table
pub static POWER8_TABLE: DispatchTable = default_kind_table();
// Power9 Table
pub static POWER9_TABLE: DispatchTable = default_kind_table();
// Power10 Table
pub static POWER10_TABLE: DispatchTable = default_kind_table();

#[inline]
#[must_use]
pub fn select_table(kind: TuneKind) -> &'static DispatchTable {
  match kind {
    TuneKind::Custom => &CUSTOM_TABLE,
    TuneKind::Default => &DEFAULT_KIND_TABLE,
    TuneKind::Portable => &PORTABLE_TABLE,
    TuneKind::Zen4 => &ZEN4_TABLE,
    TuneKind::Zen5 => &ZEN5_TABLE,
    TuneKind::Zen5c => &ZEN5C_TABLE,
    TuneKind::IntelSpr => &INTELSPR_TABLE,
    TuneKind::IntelGnr => &INTELGNR_TABLE,
    TuneKind::IntelIcl => &INTELICL_TABLE,
    TuneKind::AppleM1M3 => &APPLEM1M3_TABLE,
    TuneKind::AppleM4 => &APPLEM4_TABLE,
    TuneKind::AppleM5 => &APPLEM5_TABLE,
    TuneKind::Graviton2 => &GRAVITON2_TABLE,
    TuneKind::Graviton3 => &GRAVITON3_TABLE,
    TuneKind::Graviton4 => &GRAVITON4_TABLE,
    TuneKind::Graviton5 => &GRAVITON5_TABLE,
    TuneKind::NeoverseN2 => &NEOVERSEN2_TABLE,
    TuneKind::NeoverseN3 => &NEOVERSEN3_TABLE,
    TuneKind::NeoverseV3 => &NEOVERSEV3_TABLE,
    TuneKind::NvidiaGrace => &NVIDIAGRACE_TABLE,
    TuneKind::AmpereAltra => &AMPEREALTRA_TABLE,
    TuneKind::Aarch64Pmull => &AARCH64PMULL_TABLE,
    TuneKind::Z13 => &Z13_TABLE,
    TuneKind::Z14 => &Z14_TABLE,
    TuneKind::Z15 => &Z15_TABLE,
    TuneKind::Power7 => &POWER7_TABLE,
    TuneKind::Power8 => &POWER8_TABLE,
    TuneKind::Power9 => &POWER9_TABLE,
    TuneKind::Power10 => &POWER10_TABLE,
  }
}
