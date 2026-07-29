//! Tuned dispatch tables for Ascon permutation.
//!
//! This table controls which `permute_12` kernel is used by Ascon hash and XOF.

pub use super::kernels::AsconPermute12KernelId as KernelId;
use crate::platform::Caps;

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
  #[allow(dead_code)]
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

#[cfg(target_arch = "aarch64")]
pub static AARCH64_NEON_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  // The single-state policy stays scalar: duplicating each of the five state
  // words across NEON lanes does not add independent work. The NEON x2 batch
  // path is wired separately.
  xs: KernelId::Portable,
  s: KernelId::Portable,
  m: KernelId::Portable,
  l: KernelId::Portable,
};

#[cfg(target_arch = "x86_64")]
pub static X86_AVX2_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  // The single-state policy stays scalar: broadcasting each state word across
  // four AVX2 lanes does not add independent work. The AVX2 x4 batch path is
  // wired separately.
  xs: KernelId::Portable,
  s: KernelId::Portable,
  m: KernelId::Portable,
  l: KernelId::Portable,
};

#[cfg(target_arch = "x86_64")]
pub static X86_AVX512_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  // The single-state policy stays scalar: broadcasting each state word across
  // eight AVX-512 lanes does not add independent work. The AVX-512 x8 batch
  // path is wired separately.
  xs: KernelId::Portable,
  s: KernelId::Portable,
  m: KernelId::Portable,
  l: KernelId::Portable,
};

#[inline]
#[must_use]
pub fn select_runtime_table(#[allow(unused_variables)] caps: Caps) -> &'static DispatchTable {
  #[cfg(target_arch = "aarch64")]
  {
    use crate::platform::caps::aarch64;
    if caps.has(aarch64::NEON) {
      return &AARCH64_NEON_TABLE;
    }
  }

  #[cfg(target_arch = "x86_64")]
  {
    use crate::platform::caps::x86;
    if caps.has(x86::AVX512F.union(x86::AVX512VL)) {
      return &X86_AVX512_TABLE;
    }
    if caps.has(x86::AVX2) {
      return &X86_AVX2_TABLE;
    }
  }

  &DEFAULT_TABLE
}
