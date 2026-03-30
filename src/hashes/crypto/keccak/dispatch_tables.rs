//! Tuned dispatch tables for Keccak-f[1600].
//!
//! This table controls which permutation kernel is used by SHA-3/SHAKE and
//! SP800-185 derived constructions.

pub use super::kernels::Keccakf1600KernelId as KernelId;
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

pub static DEFAULT_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Portable,
  s: KernelId::Portable,
  m: KernelId::Portable,
  l: KernelId::Portable,
};

// The SHA3 CE kernel for single-state is slower than portable on both
// Apple Silicon and Neoverse V1/V2 due to FMOV domain-crossing overhead
// between GPR↔NEON per SHA3 CE instruction. SHA3 CE only wins for the
// 2-state interleaved path (permute_x2) where both NEON lanes carry work.
#[cfg(target_arch = "aarch64")]
pub static AARCH64_SHA3_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
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
    if caps.has(aarch64::SHA3) {
      return &AARCH64_SHA3_TABLE;
    }
  }

  #[allow(unreachable_code)]
  &DEFAULT_TABLE
}
