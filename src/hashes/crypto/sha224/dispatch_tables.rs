//! Tuned dispatch tables for SHA-224.
//!
//! Mirrors SHA-256 — same compression function, same HW accel for all sizes.

pub use super::kernels::Sha224KernelId as KernelId;
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

#[cfg(target_arch = "x86_64")]
pub static X86_SHA_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::X86Sha,
  s: KernelId::X86Sha,
  m: KernelId::X86Sha,
  l: KernelId::X86Sha,
};

#[cfg(target_arch = "aarch64")]
pub static AARCH64_SHA2_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Aarch64Sha2,
  s: KernelId::Aarch64Sha2,
  m: KernelId::Aarch64Sha2,
  l: KernelId::Aarch64Sha2,
};

#[inline]
#[must_use]
pub fn select_runtime_table(#[allow(unused_variables)] caps: Caps) -> &'static DispatchTable {
  #[cfg(target_arch = "x86_64")]
  {
    use crate::platform::caps::x86;
    if caps.has(x86::SHA) {
      return &X86_SHA_TABLE;
    }
  }
  #[cfg(target_arch = "aarch64")]
  {
    use crate::platform::caps::aarch64;
    if caps.has(aarch64::SHA2) {
      return &AARCH64_SHA2_TABLE;
    }
  }
  &DEFAULT_TABLE
}
