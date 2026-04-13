//! Tuned dispatch tables for XXH3 (**NOT CRYPTO**).
//!
//! This module is the checked-in runtime table used by capability-driven dispatch.

pub use super::kernels::Xxh3KernelId as KernelId;
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

// ─────────────────────────────────────────────────────────────────────────────
// Platform-specific tables
// ─────────────────────────────────────────────────────────────────────────────

/// x86-64 with AVX-512F: single-iteration per stripe.
#[cfg(target_arch = "x86_64")]
pub static AVX512_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Avx512,
  s: KernelId::Avx512,
  m: KernelId::Avx512,
  l: KernelId::Avx512,
};

/// x86-64 with AVX2 (no AVX-512): two iterations per stripe.
#[cfg(target_arch = "x86_64")]
pub static AVX2_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Avx2,
  s: KernelId::Avx2,
  m: KernelId::Avx2,
  l: KernelId::Avx2,
};

/// aarch64 with NEON: four iterations per stripe.
#[cfg(target_arch = "aarch64")]
pub static NEON_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Neon,
  s: KernelId::Neon,
  m: KernelId::Neon,
  l: KernelId::Neon,
};

/// POWER8+ with VSX: four iterations per stripe (128-bit vectors).
#[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
pub static VSX_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Vsx,
  s: KernelId::Vsx,
  m: KernelId::Vsx,
  l: KernelId::Vsx,
};

/// s390x z13+ with z/Vector: four iterations per stripe (128-bit vectors).
#[cfg(target_arch = "s390x")]
pub static ZVECTOR_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Vector,
  s: KernelId::Vector,
  m: KernelId::Vector,
  l: KernelId::Vector,
};

/// RISC-V with V extension: four iterations per stripe (VL=2 × u64).
///
/// Currently unused — the RVV kernel is slower than portable scalar on the
/// in-order SpacemiT K1 at 256 B–64 KiB.  Retained for future OoO cores.
#[cfg(target_arch = "riscv64")]
#[allow(dead_code)]
pub static RVV_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Rvv,
  s: KernelId::Rvv,
  m: KernelId::Rvv,
  l: KernelId::Rvv,
};

#[inline]
#[must_use]
pub fn select_runtime_table(caps: Caps) -> &'static DispatchTable {
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
    if caps.has(crate::platform::caps::power::POWER8_VECTOR) {
      return &VSX_TABLE;
    }
  }

  #[cfg(target_arch = "s390x")]
  {
    if caps.has(crate::platform::caps::s390x::VECTOR) {
      return &ZVECTOR_TABLE;
    }
  }

  // RISC-V: RVV kernel is slower than portable scalar at medium sizes on
  // in-order cores (SpacemiT K1).  Fall through to DEFAULT_TABLE (portable)
  // until out-of-order RISC-V targets are available.

  &DEFAULT_TABLE
}
