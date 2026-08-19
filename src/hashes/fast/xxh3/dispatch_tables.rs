//! Tuned dispatch tables for XXH3 (**NOT CRYPTO**).
//!
//! This module is the checked-in runtime table used by capability-driven dispatch.

pub(crate) use super::kernels::Xxh3KernelId as KernelId;
use crate::platform::Caps;

#[derive(Clone, Copy, Debug)]
pub(crate) struct DispatchTable {
  pub long: KernelId,
}

pub(crate) static DEFAULT_TABLE: DispatchTable = DispatchTable {
  long: KernelId::Portable,
};

// Platform-specific tables

/// x86-64 with AVX-512F: single-iteration per stripe.
#[cfg(target_arch = "x86_64")]
pub(crate) static AVX512_TABLE: DispatchTable = DispatchTable { long: KernelId::Avx512 };

/// x86-64 with AVX2 (no AVX-512): two iterations per stripe.
#[cfg(target_arch = "x86_64")]
pub(crate) static AVX2_TABLE: DispatchTable = DispatchTable { long: KernelId::Avx2 };

/// aarch64 with NEON: four iterations per stripe.
#[cfg(target_arch = "aarch64")]
pub(crate) static NEON_TABLE: DispatchTable = DispatchTable { long: KernelId::Neon };

/// POWER8+ with VSX: four iterations per stripe (128-bit vectors).
#[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
pub(crate) static VSX_TABLE: DispatchTable = DispatchTable { long: KernelId::Vsx };

/// s390x z13+ with z/Vector: four iterations per stripe (128-bit vectors).
#[cfg(target_arch = "s390x")]
pub(crate) static ZVECTOR_TABLE: DispatchTable = DispatchTable { long: KernelId::Vector };

#[inline]
#[must_use]
pub(crate) fn select_runtime_table(caps: Caps) -> &'static DispatchTable {
  let _ = caps;
  #[cfg(target_arch = "x86_64")]
  {
    // Prefer AVX-512 over AVX2 when available.
    if caps.has(crate::platform::caps::x86::AVX512F) {
      return &AVX512_TABLE;
    }
    if caps.has(crate::platform::caps::x86::AVX2) {
      return &AVX2_TABLE;
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    // NEON is always available on aarch64 (baseline ISA).
    if caps.has(crate::platform::caps::aarch64::NEON) {
      return &NEON_TABLE;
    }
  }

  #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
  {
    if caps.has(super::kernels::required_caps(KernelId::Vsx)) {
      return &VSX_TABLE;
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    if caps.has(crate::platform::caps::s390x::VECTOR) {
      return &ZVECTOR_TABLE;
    }
  }

  // The retired RVV implementation lost to portable scalar at 256 B–64 KiB
  // on SpacemiT K1.

  &DEFAULT_TABLE
}

#[cfg(all(test, target_arch = "powerpc64", target_endian = "little"))]
mod tests {
  use super::*;
  use crate::platform::caps::power;

  #[test]
  fn power_table_requires_every_target_feature() {
    let required = power::ALTIVEC | power::VSX | power::POWER8_VECTOR;
    assert_eq!(select_runtime_table(required).long, KernelId::Vsx);
    for missing in [power::ALTIVEC, power::VSX, power::POWER8_VECTOR] {
      assert_eq!(
        select_runtime_table(required.difference(missing)).long,
        KernelId::Portable
      );
    }
  }
}
