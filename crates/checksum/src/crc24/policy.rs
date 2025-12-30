//! CRC-24 specific policy computation and kernel dispatch.
//!
//! This module bridges the generic [`SelectionPolicy`] from `backend` with
//! CRC-24 specific configuration and kernel tables.
//!
//! # Design
//!
//! CRC-24 is currently portable-only:
//! - No hardware CRC instructions for 24-bit
//! - No SIMD implementations yet (future: PCLMUL/PMULL)
//! - Single variant: OpenPGP
//!
//! The policy module is structured for future SIMD support.

use backend::{KernelFamily, SelectionPolicy};
use platform::{Caps, Tune};

use super::{
  config::{Crc24Config, Crc24Force},
  kernels,
};
use crate::dispatchers::Crc24Fn;

// ─────────────────────────────────────────────────────────────────────────────
// Crc24Policy
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-24 specific selection policy.
///
/// Wraps the generic [`SelectionPolicy`] with CRC-24 specific configuration.
/// Currently portable-only, structured for future SIMD support.
#[derive(Clone, Copy, Debug)]
pub struct Crc24Policy {
  /// Generic selection policy (family, thresholds).
  #[allow(dead_code)] // Kept for API uniformity with CRC-64/CRC-32 and future SIMD
  pub inner: SelectionPolicy,
  /// Effective force mode after clamping to capabilities.
  pub effective_force: Crc24Force,
  /// Threshold for slice4→slice8 transition.
  pub slice4_to_slice8: usize,
}

impl Crc24Policy {
  /// Create a policy from CRC-24 configuration and platform detection.
  #[must_use]
  pub fn from_config(cfg: &Crc24Config, caps: Caps, tune: &Tune) -> Self {
    // Map Crc24Force to ForceMode if explicitly set
    let inner = if let Some(family) = cfg.effective_force.to_family() {
      SelectionPolicy::with_family(caps, tune, family)
    } else {
      // Portable-only for now
      SelectionPolicy::portable()
    };

    Self {
      inner,
      effective_force: cfg.effective_force,
      slice4_to_slice8: cfg.tunables.slice4_to_slice8,
    }
  }

  /// Get the selected kernel family.
  ///
  /// For CRC-24, this is always Portable or Reference (no SIMD).
  #[inline]
  #[must_use]
  #[allow(dead_code)] // For API uniformity with CRC-64/CRC-32
  pub const fn family(&self) -> KernelFamily {
    self.inner.family()
  }

  /// Get kernel name for the given length (for introspection).
  #[must_use]
  pub fn kernel_name(&self, len: usize) -> &'static str {
    match self.effective_force {
      Crc24Force::Reference => kernels::REFERENCE,
      Crc24Force::Slice4 => kernels::PORTABLE_SLICE4,
      Crc24Force::Slice8 => kernels::PORTABLE_SLICE8,
      _ => kernels::portable_name_for_len(len, self.slice4_to_slice8),
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Crc24Kernels - Pre-resolved kernel table
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-resolved kernel function pointers for CRC-24.
///
/// Currently portable-only, structured for future SIMD support.
#[derive(Clone, Copy, Debug)]
pub struct Crc24Kernels {
  /// Reference (bitwise) kernel - always available.
  pub reference: Crc24Fn,
  /// Portable slice-by-4 kernel - always available.
  pub slice4: Crc24Fn,
  /// Portable slice-by-8 kernel - always available.
  pub slice8: Crc24Fn,
}

impl Crc24Kernels {
  /// Create a kernel table.
  #[must_use]
  pub const fn new(reference: Crc24Fn, slice4: Crc24Fn, slice8: Crc24Fn) -> Self {
    Self {
      reference,
      slice4,
      slice8,
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch Function
// ─────────────────────────────────────────────────────────────────────────────

/// Dispatch CRC-24 computation using pre-computed policy and kernels.
///
/// This is the hot path - policy and kernels are pre-computed,
/// so dispatch is just threshold checks and function pointer calls.
#[inline]
pub fn policy_dispatch(policy: &Crc24Policy, kernels: &Crc24Kernels, crc: u32, data: &[u8]) -> u32 {
  // Handle forced modes
  match policy.effective_force {
    Crc24Force::Reference => return (kernels.reference)(crc, data),
    Crc24Force::Slice4 => return (kernels.slice4)(crc, data),
    Crc24Force::Slice8 => return (kernels.slice8)(crc, data),
    _ => {}
  }

  // Portable auto selection
  if data.len() < policy.slice4_to_slice8 {
    (kernels.slice4)(crc, data)
  } else {
    (kernels.slice8)(crc, data)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Builder
// ─────────────────────────────────────────────────────────────────────────────

/// Build OpenPGP kernel table.
#[must_use]
pub fn build_openpgp_kernels(reference: Crc24Fn, slice4: Crc24Fn, slice8: Crc24Fn) -> Crc24Kernels {
  Crc24Kernels::new(reference, slice4, slice8)
}
