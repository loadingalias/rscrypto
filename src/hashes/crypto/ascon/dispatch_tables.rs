//! Tuned dispatch tables for Ascon permutation.
//!
//! This table controls which `permute_12` kernel is used by Ascon hash and XOF.

pub(crate) use super::kernels::AsconPermute12KernelId as KernelId;
use crate::platform::Caps;

#[cfg(feature = "diag")]
pub(crate) const DEFAULT_BOUNDARIES: [usize; 3] = [64, 256, 4096];

#[derive(Clone, Copy, Debug)]
pub(crate) struct DispatchTable {
  #[cfg(feature = "diag")]
  pub boundaries: [usize; 3],
  pub xs: KernelId,
  #[cfg(feature = "diag")]
  pub s: KernelId,
  #[cfg(feature = "diag")]
  pub m: KernelId,
  #[cfg(feature = "diag")]
  pub l: KernelId,
}

pub(crate) static DEFAULT_TABLE: DispatchTable = DispatchTable {
  #[cfg(feature = "diag")]
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Portable,
  #[cfg(feature = "diag")]
  s: KernelId::Portable,
  #[cfg(feature = "diag")]
  m: KernelId::Portable,
  #[cfg(feature = "diag")]
  l: KernelId::Portable,
};

#[cfg(target_arch = "aarch64")]
pub(crate) static AARCH64_NEON_TABLE: DispatchTable = DispatchTable {
  #[cfg(feature = "diag")]
  boundaries: DEFAULT_BOUNDARIES,
  // The single-state policy stays scalar: duplicating each of the five state
  // words across NEON lanes does not add independent work. The NEON x2 batch
  // path is wired separately.
  xs: KernelId::Portable,
  #[cfg(feature = "diag")]
  s: KernelId::Portable,
  #[cfg(feature = "diag")]
  m: KernelId::Portable,
  #[cfg(feature = "diag")]
  l: KernelId::Portable,
};

#[cfg(target_arch = "x86_64")]
pub(crate) static X86_AVX2_TABLE: DispatchTable = DispatchTable {
  #[cfg(feature = "diag")]
  boundaries: DEFAULT_BOUNDARIES,
  // The single-state policy stays scalar: broadcasting each state word across
  // four AVX2 lanes does not add independent work. The AVX2 x4 batch path is
  // wired separately.
  xs: KernelId::Portable,
  #[cfg(feature = "diag")]
  s: KernelId::Portable,
  #[cfg(feature = "diag")]
  m: KernelId::Portable,
  #[cfg(feature = "diag")]
  l: KernelId::Portable,
};

#[cfg(target_arch = "x86_64")]
pub(crate) static X86_AVX512_TABLE: DispatchTable = DispatchTable {
  #[cfg(feature = "diag")]
  boundaries: DEFAULT_BOUNDARIES,
  // The single-state policy stays scalar: broadcasting each state word across
  // eight AVX-512 lanes does not add independent work. The AVX-512 x8 batch
  // path is wired separately.
  xs: KernelId::Portable,
  #[cfg(feature = "diag")]
  s: KernelId::Portable,
  #[cfg(feature = "diag")]
  m: KernelId::Portable,
  #[cfg(feature = "diag")]
  l: KernelId::Portable,
};

#[inline]
#[must_use]
pub(crate) fn select_runtime_table(caps: Caps) -> &'static DispatchTable {
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

  let _ = caps;
  &DEFAULT_TABLE
}
