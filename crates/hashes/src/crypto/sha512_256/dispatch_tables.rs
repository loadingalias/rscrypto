//! Tuned dispatch tables for SHA-512/256.
//!
//! This module is the integration point for `rscrypto-tune --apply`.
//! The tuning engine updates the per-`TuneKind` tables in this file.

use platform::TuneKind;

pub use super::kernels::Sha512_256KernelId as KernelId;

pub const DEFAULT_BOUNDARIES: [usize; 3] = [64, 256, 4096];

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
pub static DEFAULT_KIND_TABLE: DispatchTable = DEFAULT_TABLE;
// Portable Table
pub static PORTABLE_TABLE: DispatchTable = DEFAULT_TABLE;
// Zen4 Table
pub static ZEN4_TABLE: DispatchTable = DEFAULT_TABLE;
// Zen5 Table
pub static ZEN5_TABLE: DispatchTable = DEFAULT_TABLE;
// Zen5c Table
pub static ZEN5C_TABLE: DispatchTable = DEFAULT_TABLE;
// IntelSpr Table
pub static INTELSPR_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Portable,
  s: KernelId::Portable,
  m: KernelId::Portable,
  l: KernelId::Portable,
};

// IntelGnr Table
pub static INTELGNR_TABLE: DispatchTable = DEFAULT_TABLE;
// IntelIcl Table
pub static INTELICL_TABLE: DispatchTable = DEFAULT_TABLE;
// AppleM1M3 Table
pub static APPLEM1M3_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Portable,
  s: KernelId::Portable,
  m: KernelId::Portable,
  l: KernelId::Portable,
};

// AppleM4 Table
pub static APPLEM4_TABLE: DispatchTable = DEFAULT_TABLE;
// AppleM5 Table
pub static APPLEM5_TABLE: DispatchTable = DEFAULT_TABLE;
// Graviton2 Table
pub static GRAVITON2_TABLE: DispatchTable = DEFAULT_TABLE;
// Graviton3 Table
pub static GRAVITON3_TABLE: DispatchTable = DEFAULT_TABLE;
// Graviton4 Table
pub static GRAVITON4_TABLE: DispatchTable = DEFAULT_TABLE;
// Graviton5 Table
pub static GRAVITON5_TABLE: DispatchTable = DEFAULT_TABLE;
// NeoverseN2 Table
pub static NEOVERSEN2_TABLE: DispatchTable = DEFAULT_TABLE;
// NeoverseN3 Table
pub static NEOVERSEN3_TABLE: DispatchTable = DEFAULT_TABLE;
// NeoverseV3 Table
pub static NEOVERSEV3_TABLE: DispatchTable = DEFAULT_TABLE;
// NvidiaGrace Table
pub static NVIDIAGRACE_TABLE: DispatchTable = DEFAULT_TABLE;
// AmpereAltra Table
pub static AMPEREALTRA_TABLE: DispatchTable = DEFAULT_TABLE;
// Aarch64Pmull Table
pub static AARCH64PMULL_TABLE: DispatchTable = DEFAULT_TABLE;
// Z13 Table
pub static Z13_TABLE: DispatchTable = DEFAULT_TABLE;
// Z14 Table
pub static Z14_TABLE: DispatchTable = DEFAULT_TABLE;
// Z15 Table
pub static Z15_TABLE: DispatchTable = DEFAULT_TABLE;
// Power7 Table
pub static POWER7_TABLE: DispatchTable = DEFAULT_TABLE;
// Power8 Table
pub static POWER8_TABLE: DispatchTable = DEFAULT_TABLE;
// Power9 Table
pub static POWER9_TABLE: DispatchTable = DEFAULT_TABLE;
// Power10 Table
pub static POWER10_TABLE: DispatchTable = DEFAULT_TABLE;

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
