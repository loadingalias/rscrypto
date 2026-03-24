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

impl DispatchTable {
  #[inline]
  #[must_use]
  #[allow(dead_code)]
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

// The x86-64 AVX2 and AVX-512 chi-only SIMD kernels are slower than the
// optimized portable (array-based) permutation on every tested platform:
//
//   Zen5:  portable 241 ns  vs  AVX-512 332 ns  (+38%)
//   SPR:   portable 361 ns  vs  AVX-512 453 ns  (+26%)
//   Zen4:  portable 342 ns  vs  AVX-512 373 ns  (+9%)
//   ICL:   portable 411 ns  vs  AVX-512 449 ns  (+9%)
//
// The SIMD kernels only accelerate the χ step but pay for load/store
// traffic to move the 25-lane state between GPR and SIMD domains. The
// array-based portable rewrite reduced register pressure enough that the
// scalar-only path wins outright. Route all dispatch to Portable.
#[cfg(target_arch = "x86_64")]
pub static X86_PORTABLE_TABLE: DispatchTable = DispatchTable {
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

  #[cfg(target_arch = "x86_64")]
  {
    // Portable is faster than both AVX2 and AVX-512 chi-only kernels on
    // all tested x86-64 microarchitectures (see comment above).
    return &X86_PORTABLE_TABLE;
  }

  #[allow(unreachable_code)]
  &DEFAULT_TABLE
}
