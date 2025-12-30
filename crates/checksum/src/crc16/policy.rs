//! CRC-16 specific policy computation and kernel dispatch.
//!
//! This module bridges the generic [`SelectionPolicy`] from `backend` with
//! CRC-16 specific configuration and kernel tables.
//!
//! # Design
//!
//! CRC-16 is simpler than CRC-64/CRC-32:
//! - No hardware CRC instructions for 16-bit
//! - No multi-stream support (single-lane SIMD)
//! - Two variants: CCITT (X25) and IBM (ARC)
//! - SIMD tiers: PCLMUL/PMULL (folding), VPCLMUL (wide)
//!
//! The policy is computed once from:
//! - [`Crc16Config`] - user overrides and tunables
//! - [`Caps`] - detected CPU capabilities
//! - [`Tune`] - microarchitecture-specific hints
//!
//! The resulting [`Crc16Policy`] is cached and used for all dispatch decisions,
//! eliminating per-call branching on capabilities and thresholds.

use backend::{KernelFamily, SelectionPolicy};
use platform::{Caps, Tune};

use super::{
  config::{Crc16Config, Crc16Force},
  kernels,
};
use crate::dispatchers::Crc16Fn;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Minimum bytes for SIMD dispatch (below this, portable is always faster).
#[allow(dead_code)] // Reserved for future small-buffer optimization
pub const CRC16_SMALL_THRESHOLD: usize = 16;

// ─────────────────────────────────────────────────────────────────────────────
// Crc16Variant - Distinguishes CCITT from IBM
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-16 polynomial variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Crc16Variant {
  /// CRC-16/X25 (CCITT, IBM-SDLC)
  Ccitt,
  /// CRC-16/ARC (IBM)
  Ibm,
}

// ─────────────────────────────────────────────────────────────────────────────
// Crc16Policy
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-16 specific selection policy.
///
/// Wraps the generic [`SelectionPolicy`] with CRC-16 specific configuration
/// and provides dispatch methods for both CCITT and IBM variants.
#[derive(Clone, Copy, Debug)]
pub struct Crc16Policy {
  /// Generic selection policy (family, thresholds).
  #[allow(dead_code)] // Kept for API uniformity with CRC-64/CRC-32
  pub inner: SelectionPolicy,
  /// Effective force mode after clamping to capabilities.
  pub effective_force: Crc16Force,
  /// CRC-16 variant (CCITT or IBM).
  #[allow(dead_code)] // Kept for future use and API uniformity
  pub variant: Crc16Variant,
  /// Threshold for slice4→slice8 transition.
  pub slice4_to_slice8: usize,
  /// Threshold for portable→clmul transition.
  pub portable_to_clmul: usize,
  /// Threshold for clmul→vpclmul transition (x86_64 only).
  #[allow(dead_code)] // Only used on x86_64
  pub clmul_to_vpclmul: usize,
  /// Whether carryless multiply is available.
  pub has_clmul: bool,
  /// Whether wide carryless multiply is available (x86_64 VPCLMUL).
  #[allow(dead_code)] // Only used on x86_64
  pub has_vpclmul: bool,
}

impl Crc16Policy {
  /// Create a policy from CRC-16 configuration and platform detection.
  #[must_use]
  pub fn from_config(cfg: &Crc16Config, caps: Caps, tune: &Tune, variant: Crc16Variant) -> Self {
    // Detect available features
    let (has_clmul, has_vpclmul) = Self::detect_features(caps, tune);

    // Map Crc16Force to ForceMode if explicitly set
    let inner = if let Some(family) = cfg.effective_force.to_family() {
      SelectionPolicy::with_family(caps, tune, family)
    } else {
      SelectionPolicy::from_platform(caps, tune)
    };

    // Compute VPCLMUL threshold
    let clmul_to_vpclmul = Self::compute_vpclmul_threshold(tune);

    Self {
      inner,
      effective_force: cfg.effective_force,
      variant,
      slice4_to_slice8: cfg.tunables.slice4_to_slice8,
      portable_to_clmul: cfg.tunables.portable_to_clmul,
      clmul_to_vpclmul,
      has_clmul,
      has_vpclmul,
    }
  }

  /// Detect available SIMD features.
  #[inline]
  fn detect_features(caps: Caps, tune: &Tune) -> (bool, bool) {
    #[cfg(target_arch = "x86_64")]
    {
      let has_clmul = caps.has(platform::caps::x86::PCLMUL_READY);
      let has_vpclmul =
        caps.has(platform::caps::x86::VPCLMUL_READY) && tune.fast_wide_ops && tune.effective_simd_width >= 512;
      (has_clmul, has_vpclmul)
    }
    #[cfg(target_arch = "aarch64")]
    {
      let has_clmul = caps.has(platform::caps::aarch64::PMULL_READY);
      let _ = tune;
      (has_clmul, false) // No wide tier on aarch64 for CRC-16
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      let _ = (caps, tune);
      (false, false)
    }
  }

  /// Compute VPCLMUL threshold.
  #[inline]
  fn compute_vpclmul_threshold(tune: &Tune) -> usize {
    let simd_bytes = (tune.effective_simd_width as usize).strict_div(8).max(1);
    simd_bytes.strict_mul(4).max(tune.pclmul_threshold).max(1)
  }

  /// Get the selected kernel family.
  #[inline]
  #[must_use]
  #[allow(dead_code)] // For API uniformity with CRC-64/CRC-32
  pub const fn family(&self) -> KernelFamily {
    self.inner.family()
  }

  /// Check if SIMD should be used for this length.
  #[inline]
  #[must_use]
  pub fn should_use_simd(&self, len: usize) -> bool {
    self.has_clmul && len >= self.portable_to_clmul
  }

  /// Get kernel name for the given length (for introspection).
  #[must_use]
  pub fn kernel_name(&self, len: usize) -> &'static str {
    match self.effective_force {
      Crc16Force::Reference => return kernels::REFERENCE,
      Crc16Force::Slice4 => return kernels::PORTABLE_SLICE4,
      Crc16Force::Slice8 => return kernels::PORTABLE_SLICE8,
      Crc16Force::Portable => return self.portable_name_for_len(len),
      _ => {}
    }

    if !self.should_use_simd(len) {
      return self.portable_name_for_len(len);
    }

    self.kernel_name_for_family(len)
  }

  #[inline]
  fn portable_name_for_len(&self, len: usize) -> &'static str {
    kernels::portable_name_for_len(len, self.slice4_to_slice8)
  }

  #[cfg(target_arch = "x86_64")]
  fn kernel_name_for_family(&self, len: usize) -> &'static str {
    if self.has_vpclmul && len >= self.clmul_to_vpclmul {
      kernels::x86_64::VPCLMUL
    } else {
      kernels::x86_64::PCLMUL
    }
  }

  #[cfg(target_arch = "aarch64")]
  fn kernel_name_for_family(&self, _len: usize) -> &'static str {
    kernels::aarch64::PMULL
  }

  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  fn kernel_name_for_family(&self, len: usize) -> &'static str {
    self.portable_name_for_len(len)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Crc16Kernels - Pre-resolved kernel table
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-resolved kernel function pointers for a CRC-16 variant.
///
/// This eliminates runtime kernel selection by caching all needed
/// function pointers at policy initialization time.
#[derive(Clone, Copy, Debug)]
pub struct Crc16Kernels {
  /// Reference (bitwise) kernel - always available.
  pub reference: Crc16Fn,
  /// Portable slice-by-4 kernel - always available.
  pub slice4: Crc16Fn,
  /// Portable slice-by-8 kernel - always available.
  pub slice8: Crc16Fn,
  /// CLMUL/PMULL kernel (if available).
  pub clmul: Option<Crc16Fn>,
  /// VPCLMUL kernel (x86_64 only, if available).
  #[allow(dead_code)] // Only used on x86_64
  pub vpclmul: Option<Crc16Fn>,
}

impl Crc16Kernels {
  /// Create a portable-only kernel table.
  #[allow(dead_code)] // Kept for API uniformity with CRC-64/CRC-32
  #[must_use]
  pub const fn portable_only(reference: Crc16Fn, slice4: Crc16Fn, slice8: Crc16Fn) -> Self {
    Self {
      reference,
      slice4,
      slice8,
      clmul: None,
      vpclmul: None,
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch Function
// ─────────────────────────────────────────────────────────────────────────────

/// Dispatch CRC-16 computation using pre-computed policy and kernels.
///
/// This is the new hot path - policy and kernels are pre-computed,
/// so dispatch is just threshold checks and function pointer calls.
#[inline]
pub fn policy_dispatch(policy: &Crc16Policy, kernels: &Crc16Kernels, crc: u16, data: &[u8]) -> u16 {
  let len = data.len();

  // Handle forced modes
  match policy.effective_force {
    Crc16Force::Reference => return (kernels.reference)(crc, data),
    Crc16Force::Slice4 => return (kernels.slice4)(crc, data),
    Crc16Force::Slice8 => return (kernels.slice8)(crc, data),
    Crc16Force::Portable => return portable_auto(policy, kernels, crc, data),
    _ => {}
  }

  // Fast path: tiny buffers or no SIMD
  if !policy.should_use_simd(len) {
    return portable_auto(policy, kernels, crc, data);
  }

  // SIMD dispatch
  policy_dispatch_simd(policy, kernels, crc, data)
}

#[inline]
fn portable_auto(policy: &Crc16Policy, kernels: &Crc16Kernels, crc: u16, data: &[u8]) -> u16 {
  if data.len() < policy.slice4_to_slice8 {
    (kernels.slice4)(crc, data)
  } else {
    (kernels.slice8)(crc, data)
  }
}

#[cfg(target_arch = "x86_64")]
fn policy_dispatch_simd(policy: &Crc16Policy, kernels: &Crc16Kernels, crc: u16, data: &[u8]) -> u16 {
  let len = data.len();

  // VPCLMUL tier
  if let Some(vpclmul) = kernels.vpclmul
    && policy.has_vpclmul
    && len >= policy.clmul_to_vpclmul
  {
    return (vpclmul)(crc, data);
  }

  // PCLMUL tier
  if let Some(clmul) = kernels.clmul {
    return (clmul)(crc, data);
  }

  portable_auto(policy, kernels, crc, data)
}

#[cfg(target_arch = "aarch64")]
fn policy_dispatch_simd(policy: &Crc16Policy, kernels: &Crc16Kernels, crc: u16, data: &[u8]) -> u16 {
  // PMULL tier
  if let Some(clmul) = kernels.clmul {
    return (clmul)(crc, data);
  }

  portable_auto(policy, kernels, crc, data)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn policy_dispatch_simd(policy: &Crc16Policy, kernels: &Crc16Kernels, crc: u16, data: &[u8]) -> u16 {
  portable_auto(policy, kernels, crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Builders (per architecture)
// ─────────────────────────────────────────────────────────────────────────────

/// Build CCITT kernel table for x86_64.
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn build_ccitt_kernels_x86(
  policy: &Crc16Policy,
  reference: Crc16Fn,
  slice4: Crc16Fn,
  slice8: Crc16Fn,
) -> Crc16Kernels {
  use super::x86_64;

  Crc16Kernels {
    reference,
    slice4,
    slice8,
    clmul: if policy.has_clmul {
      Some(x86_64::crc16_ccitt_pclmul_safe)
    } else {
      None
    },
    vpclmul: if policy.has_vpclmul {
      Some(x86_64::crc16_ccitt_vpclmul_safe)
    } else {
      None
    },
  }
}

/// Build IBM kernel table for x86_64.
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn build_ibm_kernels_x86(
  policy: &Crc16Policy,
  reference: Crc16Fn,
  slice4: Crc16Fn,
  slice8: Crc16Fn,
) -> Crc16Kernels {
  use super::x86_64;

  Crc16Kernels {
    reference,
    slice4,
    slice8,
    clmul: if policy.has_clmul {
      Some(x86_64::crc16_ibm_pclmul_safe)
    } else {
      None
    },
    vpclmul: if policy.has_vpclmul {
      Some(x86_64::crc16_ibm_vpclmul_safe)
    } else {
      None
    },
  }
}

/// Build CCITT kernel table for aarch64.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn build_ccitt_kernels_aarch64(
  policy: &Crc16Policy,
  reference: Crc16Fn,
  slice4: Crc16Fn,
  slice8: Crc16Fn,
) -> Crc16Kernels {
  use super::aarch64;

  Crc16Kernels {
    reference,
    slice4,
    slice8,
    clmul: if policy.has_clmul {
      Some(aarch64::crc16_ccitt_pmull_safe)
    } else {
      None
    },
    vpclmul: None,
  }
}

/// Build IBM kernel table for aarch64.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn build_ibm_kernels_aarch64(
  policy: &Crc16Policy,
  reference: Crc16Fn,
  slice4: Crc16Fn,
  slice8: Crc16Fn,
) -> Crc16Kernels {
  use super::aarch64;

  Crc16Kernels {
    reference,
    slice4,
    slice8,
    clmul: if policy.has_clmul {
      Some(aarch64::crc16_ibm_pmull_safe)
    } else {
      None
    },
    vpclmul: None,
  }
}

/// Build kernel table for non-SIMD architectures.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[must_use]
pub fn build_ccitt_kernels_generic(reference: Crc16Fn, slice4: Crc16Fn, slice8: Crc16Fn) -> Crc16Kernels {
  Crc16Kernels::portable_only(reference, slice4, slice8)
}

/// Build kernel table for non-SIMD architectures.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[must_use]
pub fn build_ibm_kernels_generic(reference: Crc16Fn, slice4: Crc16Fn, slice8: Crc16Fn) -> Crc16Kernels {
  Crc16Kernels::portable_only(reference, slice4, slice8)
}
