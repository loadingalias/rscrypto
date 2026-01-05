//! CRC-64 specific policy computation and kernel dispatch.
//!
//! This module bridges the generic [`SelectionPolicy`] from `backend` with
//! CRC-64 specific configuration and kernel tables.
//!
//! # Design
//!
//! The policy is computed once from:
//! - [`Crc64Config`] - user overrides and tunables
//! - [`Caps`] - detected CPU capabilities
//! - [`Tune`] - microarchitecture-specific hints
//!
//! The resulting [`Crc64Policy`] is cached and used for all dispatch decisions,
//! eliminating per-call branching on capabilities and thresholds.

use backend::{KernelFamily, KernelTier, SelectionPolicy};
#[cfg(feature = "diag")]
use platform::TuneKind;
use platform::{Caps, Tune};

use super::{
  config::{Crc64Config, Crc64Force},
  kernels,
};
#[cfg(feature = "diag")]
use crate::diag::{Crc64Polynomial, Crc64SelectionDiag, SelectionReason};
use crate::{common::kernels::stream_to_index, dispatchers::Crc64Fn};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Minimum bytes for SIMD dispatch (below this, portable is always faster).
pub const CRC64_SMALL_THRESHOLD: usize = 16;

/// Block size for CRC-64 folding operations.
#[allow(dead_code)] // Retained as a semantic baseline constant for tuning docs.
pub const CRC64_FOLD_BLOCK_BYTES: usize = 128;

/// Upper bound for selecting the "small" SIMD kernel.
///
/// The small kernels fold 16-byte lanes with minimal setup cost and can be
/// faster than full 128-byte folding for small-to-medium inputs.
pub const CRC64_SMALL_KERNEL_MAX_BYTES_DEFAULT: usize = 512;

/// Minimum bytes for 4×512 VPCLMUL kernel (very high setup cost).
pub const CRC64_4X512_MIN_BYTES: usize = 8192;

// ─────────────────────────────────────────────────────────────────────────────
// Crc64Policy
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-64 specific selection policy.
///
/// Wraps the generic [`SelectionPolicy`] with CRC-64 specific configuration
/// and provides dispatch methods for both XZ and NVME variants.
#[derive(Clone, Copy, Debug)]
pub struct Crc64Policy {
  /// Generic selection policy (family, thresholds, streams).
  pub inner: SelectionPolicy,
  /// Effective force mode after clamping to capabilities.
  pub effective_force: Crc64Force,
  /// Whether VPCLMUL 4×512 kernel should be used for very large buffers.
  /// Only used on x86_64 with VPCLMUL support.
  #[allow(dead_code)] // Only used on x86_64
  pub use_4x512: bool,
  /// Minimum bytes per lane for multi-stream folding.
  ///
  /// This is resolved from tunables (if set) or the kernel family default.
  pub min_bytes_per_lane: usize,
  /// Upper bound for selecting the small-buffer SIMD kernel.
  pub small_kernel_max_bytes: usize,
}

impl Crc64Policy {
  /// Create a policy from CRC-64 configuration and platform detection.
  #[must_use]
  pub fn from_config(
    cfg: &Crc64Config,
    tunables: super::config::Crc64VariantTunables,
    caps: Caps,
    tune: &Tune,
  ) -> Self {
    // Map Crc64Force to ForceMode if explicitly set
    let mut inner = if let Some(family) = cfg.effective_force.to_family() {
      SelectionPolicy::with_family(caps, tune, family)
    } else {
      // Auto selection with CRC-64 specific thresholds
      let mut policy = SelectionPolicy::from_platform(caps, tune);
      // Apply CRC-64 specific thresholds from tunables
      Self::apply_tunables(&mut policy, tunables);
      policy
    };

    // Apply tuned stream preference (still capped by the architecture limit).
    inner.cap_max_streams(tunables.streams);

    // Check if 4×512 VPCLMUL is worthwhile
    let use_4x512 = inner.family() == KernelFamily::X86Vpclmul && tune.fast_wide_ops;

    // Resolve min_bytes_per_lane: tunables override > family default
    let min_bytes_per_lane = tunables
      .min_bytes_per_lane
      .unwrap_or_else(|| inner.family().min_bytes_per_lane());

    let small_kernel_max_bytes = tunables.small_kernel_max_bytes.max(CRC64_SMALL_THRESHOLD);

    Self {
      inner,
      effective_force: cfg.effective_force,
      use_4x512,
      min_bytes_per_lane,
      small_kernel_max_bytes,
    }
  }

  /// Apply CRC-64 specific tunables to a policy.
  fn apply_tunables(policy: &mut SelectionPolicy, tunables: super::config::Crc64VariantTunables) {
    // Override thresholds from config if set
    policy.small_threshold = tunables.portable_to_clmul.max(CRC64_SMALL_THRESHOLD);

    // VPCLMUL threshold
    if policy.tier() == KernelTier::Wide {
      policy.wide_threshold = tunables.pclmul_to_vpclmul;
    }
  }

  /// Get the selected kernel family.
  #[inline]
  #[must_use]
  pub const fn family(&self) -> KernelFamily {
    self.inner.family()
  }

  /// Get the kernel tier.
  #[inline]
  #[must_use]
  #[allow(dead_code)] // Used for introspection
  pub const fn tier(&self) -> KernelTier {
    self.inner.tier()
  }

  /// Check if SIMD should be used for this buffer length.
  #[inline]
  #[must_use]
  pub fn should_use_simd(&self, len: usize) -> bool {
    self.inner.should_use_simd(len)
  }

  /// Get the optimal stream count for this buffer length.
  ///
  /// Uses the algorithm-specific `min_bytes_per_lane` to determine when
  /// multi-stream processing is worthwhile.
  #[inline]
  #[must_use]
  pub fn streams_for_len(&self, len: usize) -> u8 {
    self.inner.streams_for_len_with_min(len, self.min_bytes_per_lane)
  }

  /// Get the kernel name for this policy and buffer length.
  #[must_use]
  pub fn kernel_name(&self, len: usize) -> &'static str {
    if len < CRC64_SMALL_THRESHOLD {
      return kernels::PORTABLE;
    }

    if !self.should_use_simd(len) {
      return kernels::PORTABLE;
    }

    match self.family() {
      KernelFamily::Reference => kernels::REFERENCE,
      KernelFamily::Portable => kernels::PORTABLE,

      #[cfg(target_arch = "x86_64")]
      KernelFamily::X86Vpclmul => {
        if len < self.small_kernel_max_bytes {
          return kernels::x86_64::PCLMUL_SMALL;
        }
        if self.use_4x512 && len >= CRC64_4X512_MIN_BYTES {
          return kernels::x86_64::VPCLMUL_4X512;
        }
        select_kernel_name_for_len(self, len, kernels::x86_64::VPCLMUL_NAMES)
      }

      #[cfg(target_arch = "x86_64")]
      KernelFamily::X86Pclmul => {
        if len < self.small_kernel_max_bytes {
          kernels::x86_64::PCLMUL_SMALL
        } else {
          select_kernel_name_for_len(self, len, kernels::x86_64::PCLMUL_NAMES)
        }
      }

      #[cfg(target_arch = "aarch64")]
      KernelFamily::ArmSve2Pmull => {
        if len < self.small_kernel_max_bytes {
          return kernels::aarch64::SVE2_PMULL_SMALL;
        }
        select_kernel_name_for_len(self, len, kernels::aarch64::SVE2_PMULL_NAMES)
      }

      #[cfg(target_arch = "aarch64")]
      KernelFamily::ArmPmullEor3 => {
        if len < self.small_kernel_max_bytes {
          kernels::aarch64::PMULL_SMALL
        } else {
          select_kernel_name_for_len(self, len, kernels::aarch64::PMULL_EOR3_NAMES)
        }
      }

      #[cfg(target_arch = "aarch64")]
      KernelFamily::ArmPmull => {
        if len < self.small_kernel_max_bytes {
          kernels::aarch64::PMULL_SMALL
        } else {
          select_kernel_name_for_len(self, len, kernels::aarch64::PMULL_NAMES)
        }
      }

      #[cfg(target_arch = "powerpc64")]
      KernelFamily::PowerVpmsum => select_kernel_name_for_len(self, len, kernels::power::VPMSUM_NAMES),

      #[cfg(target_arch = "s390x")]
      KernelFamily::S390xVgfm => select_kernel_name_for_len(self, len, kernels::s390x::VGFM_NAMES),

      #[cfg(target_arch = "riscv64")]
      KernelFamily::RiscvZvbc => select_kernel_name_for_len(self, len, kernels::riscv64::ZVBC_NAMES),

      #[cfg(target_arch = "riscv64")]
      KernelFamily::RiscvZbc => select_kernel_name_for_len(self, len, kernels::riscv64::ZBC_NAMES),

      // Fallback for families not applicable to current arch
      _ => kernels::PORTABLE,
    }
  }

  #[cfg(feature = "diag")]
  #[inline]
  #[must_use]
  pub fn diag(&self, len: usize, polynomial: Crc64Polynomial, tune_kind: TuneKind) -> Crc64SelectionDiag {
    let selected_kernel = self.kernel_name(len);
    let selected_streams = if selected_kernel.contains("small") {
      1
    } else {
      self.streams_for_len(len)
    };

    let reason = if len < CRC64_SMALL_THRESHOLD {
      SelectionReason::BelowSmallThreshold
    } else if self.effective_force != Crc64Force::Auto {
      SelectionReason::Forced
    } else if !self.should_use_simd(len) {
      SelectionReason::BelowSimdThreshold
    } else {
      SelectionReason::Auto
    };

    Crc64SelectionDiag {
      polynomial,
      len,
      tune_kind,
      reason,
      effective_force: self.effective_force,
      policy_family: self.inner.name(),
      selected_kernel,
      selected_streams,
      portable_to_clmul: self.inner.small_threshold,
      pclmul_to_vpclmul: self.inner.wide_threshold,
      small_kernel_max_bytes: self.small_kernel_max_bytes,
      use_4x512: self.use_4x512,
      min_bytes_per_lane: self.min_bytes_per_lane,
    }
  }
}

#[inline]
#[must_use]
fn select_kernel_name_for_streams(names: &[&'static str], streams: u8) -> &'static str {
  let idx = stream_to_index(streams);
  names.get(idx).or(names.first()).copied().unwrap_or(kernels::PORTABLE)
}

#[inline]
#[must_use]
fn select_kernel_name_for_len(policy: &Crc64Policy, len: usize, names: &[&'static str]) -> &'static str {
  select_kernel_name_for_streams(names, policy.streams_for_len(len))
}

// ─────────────────────────────────────────────────────────────────────────────
// Crc64Kernels - Pre-resolved kernel table
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-resolved kernel function pointers for a CRC-64 variant.
///
/// This eliminates runtime kernel array lookups by caching all needed
/// function pointers at policy initialization time.
// Reserved for future pre-resolved dispatch API.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct Crc64Kernels {
  /// Reference (bitwise) kernel - always available.
  pub reference: Crc64Fn,
  /// Portable (slice-by-16) kernel - always available.
  pub portable: Crc64Fn,
  /// Primary family kernels indexed by stream count.
  /// Layout: [1-way, 2-way, 4-way, 7-way, 8-way]
  pub primary: [Crc64Fn; 5],
  /// Small-buffer kernel for primary family (< fold block).
  pub primary_small: Option<Crc64Fn>,
  /// Wide family kernels (VPCLMUL/SVE2).
  pub wide: Option<[Crc64Fn; 5]>,
  /// Extra-wide kernel (4×512 VPCLMUL).
  pub wide_4x512: Option<Crc64Fn>,
}

impl Crc64Kernels {
  /// Create a portable-only kernel table.
  // Reserved for future pre-resolved dispatch API.
  #[allow(dead_code)]
  #[must_use]
  pub const fn portable_only(reference: Crc64Fn, portable: Crc64Fn) -> Self {
    Self {
      reference,
      portable,
      primary: [portable; 5],
      primary_small: None,
      wide: None,
      wide_4x512: None,
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch Function
// ─────────────────────────────────────────────────────────────────────────────

/// Dispatch CRC-64 computation using pre-computed policy and kernels.
///
/// This is the new hot path - policy and kernels are pre-computed,
/// so dispatch is just threshold checks and function pointer calls.
// Reserved for future pre-resolved dispatch API.
#[allow(dead_code)]
#[inline]
pub fn policy_dispatch(policy: &Crc64Policy, kernels: &Crc64Kernels, crc: u64, data: &[u8]) -> u64 {
  let len = data.len();

  // Fast path: tiny buffers
  if len < CRC64_SMALL_THRESHOLD {
    return (kernels.portable)(crc, data);
  }

  // Handle forced modes
  match policy.effective_force {
    Crc64Force::Reference => return (kernels.reference)(crc, data),
    Crc64Force::Portable => return (kernels.portable)(crc, data),
    _ => {}
  }

  // Below SIMD threshold: use portable
  if !policy.should_use_simd(len) {
    return (kernels.portable)(crc, data);
  }

  // Small-buffer SIMD kernel: prefer this before wide-tier selection.
  if len < policy.small_kernel_max_bytes
    && let Some(small) = kernels.primary_small
  {
    return (small)(crc, data);
  }

  // Wide tier (VPCLMUL/SVE2)
  if let Some(ref wide) = kernels.wide
    && policy.inner.should_use_wide(len)
  {
    // 4×512 for very large buffers
    if let Some(wide_4x512) = kernels.wide_4x512
      && len >= CRC64_4X512_MIN_BYTES
    {
      return (wide_4x512)(crc, data);
    }
    let streams = policy.streams_for_len(len);
    let idx = stream_to_index(streams);
    return (wide.get(idx).copied().unwrap_or(wide[0]))(crc, data);
  }

  // Primary tier (PCLMUL/PMULL/etc)
  let streams = policy.streams_for_len(len);

  let idx = stream_to_index(streams);
  (kernels.primary.get(idx).copied().unwrap_or(kernels.primary[0]))(crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Builders (per architecture)
// ─────────────────────────────────────────────────────────────────────────────

/// Build XZ kernel table for the current architecture.
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn build_xz_kernels_arch(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  build_xz_kernels_x86(policy, reference, portable)
}

/// Build NVME kernel table for the current architecture.
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn build_nvme_kernels_arch(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  build_nvme_kernels_x86(policy, reference, portable)
}

/// Build XZ kernel table for the current architecture.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn build_xz_kernels_arch(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  build_xz_kernels_aarch64(policy, reference, portable)
}

/// Build NVME kernel table for the current architecture.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn build_nvme_kernels_arch(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  build_nvme_kernels_aarch64(policy, reference, portable)
}

/// Build XZ kernel table for the current architecture.
#[cfg(target_arch = "powerpc64")]
#[must_use]
pub fn build_xz_kernels_arch(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  build_xz_kernels_power(policy, reference, portable)
}

/// Build NVME kernel table for the current architecture.
#[cfg(target_arch = "powerpc64")]
#[must_use]
pub fn build_nvme_kernels_arch(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  build_nvme_kernels_power(policy, reference, portable)
}

/// Build XZ kernel table for the current architecture.
#[cfg(target_arch = "s390x")]
#[must_use]
pub fn build_xz_kernels_arch(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  build_xz_kernels_s390x(policy, reference, portable)
}

/// Build NVME kernel table for the current architecture.
#[cfg(target_arch = "s390x")]
#[must_use]
pub fn build_nvme_kernels_arch(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  build_nvme_kernels_s390x(policy, reference, portable)
}

/// Build XZ kernel table for the current architecture.
#[cfg(target_arch = "riscv64")]
#[must_use]
pub fn build_xz_kernels_arch(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  build_xz_kernels_riscv64(policy, reference, portable)
}

/// Build NVME kernel table for the current architecture.
#[cfg(target_arch = "riscv64")]
#[must_use]
pub fn build_nvme_kernels_arch(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  build_nvme_kernels_riscv64(policy, reference, portable)
}

/// Build XZ kernel table for the selected family.
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn build_xz_kernels_x86(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  use kernels::x86_64::*;

  match policy.family() {
    KernelFamily::X86Vpclmul => Crc64Kernels {
      reference,
      portable,
      primary: XZ_PCLMUL,
      primary_small: Some(XZ_PCLMUL_SMALL),
      wide: Some(XZ_VPCLMUL),
      wide_4x512: if policy.use_4x512 { Some(XZ_VPCLMUL_4X512) } else { None },
    },
    KernelFamily::X86Pclmul => Crc64Kernels {
      reference,
      portable,
      primary: XZ_PCLMUL,
      primary_small: Some(XZ_PCLMUL_SMALL),
      wide: None,
      wide_4x512: None,
    },
    _ => Crc64Kernels::portable_only(reference, portable),
  }
}

/// Build NVME kernel table for the selected family.
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn build_nvme_kernels_x86(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  use kernels::x86_64::*;

  match policy.family() {
    KernelFamily::X86Vpclmul => Crc64Kernels {
      reference,
      portable,
      primary: NVME_PCLMUL,
      primary_small: Some(NVME_PCLMUL_SMALL),
      wide: Some(NVME_VPCLMUL),
      wide_4x512: if policy.use_4x512 {
        Some(NVME_VPCLMUL_4X512)
      } else {
        None
      },
    },
    KernelFamily::X86Pclmul => Crc64Kernels {
      reference,
      portable,
      primary: NVME_PCLMUL,
      primary_small: Some(NVME_PCLMUL_SMALL),
      wide: None,
      wide_4x512: None,
    },
    _ => Crc64Kernels::portable_only(reference, portable),
  }
}

/// Build XZ kernel table for aarch64.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn build_xz_kernels_aarch64(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  use kernels::aarch64::*;

  match policy.family() {
    KernelFamily::ArmSve2Pmull => Crc64Kernels {
      reference,
      portable,
      primary: XZ_PMULL,
      primary_small: Some(XZ_SVE2_PMULL_SMALL),
      wide: Some(XZ_SVE2_PMULL),
      wide_4x512: None,
    },
    KernelFamily::ArmPmullEor3 => Crc64Kernels {
      reference,
      portable,
      primary: XZ_PMULL,
      primary_small: Some(XZ_PMULL_SMALL),
      wide: Some(XZ_PMULL_EOR3),
      wide_4x512: None,
    },
    KernelFamily::ArmPmull => Crc64Kernels {
      reference,
      portable,
      primary: XZ_PMULL,
      primary_small: Some(XZ_PMULL_SMALL),
      wide: None,
      wide_4x512: None,
    },
    _ => Crc64Kernels::portable_only(reference, portable),
  }
}

/// Build NVME kernel table for aarch64.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn build_nvme_kernels_aarch64(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  use kernels::aarch64::*;

  match policy.family() {
    KernelFamily::ArmSve2Pmull => Crc64Kernels {
      reference,
      portable,
      primary: NVME_PMULL,
      primary_small: Some(NVME_SVE2_PMULL_SMALL),
      wide: Some(NVME_SVE2_PMULL),
      wide_4x512: None,
    },
    KernelFamily::ArmPmullEor3 => Crc64Kernels {
      reference,
      portable,
      primary: NVME_PMULL,
      primary_small: Some(NVME_PMULL_SMALL),
      wide: Some(NVME_PMULL_EOR3),
      wide_4x512: None,
    },
    KernelFamily::ArmPmull => Crc64Kernels {
      reference,
      portable,
      primary: NVME_PMULL,
      primary_small: Some(NVME_PMULL_SMALL),
      wide: None,
      wide_4x512: None,
    },
    _ => Crc64Kernels::portable_only(reference, portable),
  }
}

/// Build XZ kernel table for Power.
#[cfg(target_arch = "powerpc64")]
#[must_use]
pub fn build_xz_kernels_power(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  use kernels::power::*;

  match policy.family() {
    KernelFamily::PowerVpmsum => Crc64Kernels {
      reference,
      portable,
      primary: XZ_VPMSUM,
      primary_small: None,
      wide: None,
      wide_4x512: None,
    },
    _ => Crc64Kernels::portable_only(reference, portable),
  }
}

/// Build NVME kernel table for Power.
#[cfg(target_arch = "powerpc64")]
#[must_use]
pub fn build_nvme_kernels_power(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  use kernels::power::*;

  match policy.family() {
    KernelFamily::PowerVpmsum => Crc64Kernels {
      reference,
      portable,
      primary: NVME_VPMSUM,
      primary_small: None,
      wide: None,
      wide_4x512: None,
    },
    _ => Crc64Kernels::portable_only(reference, portable),
  }
}

/// Build XZ kernel table for s390x.
#[cfg(target_arch = "s390x")]
#[must_use]
pub fn build_xz_kernels_s390x(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  use kernels::s390x::*;

  match policy.family() {
    KernelFamily::S390xVgfm => Crc64Kernels {
      reference,
      portable,
      primary: XZ_VGFM,
      primary_small: None,
      wide: None,
      wide_4x512: None,
    },
    _ => Crc64Kernels::portable_only(reference, portable),
  }
}

/// Build NVME kernel table for s390x.
#[cfg(target_arch = "s390x")]
#[must_use]
pub fn build_nvme_kernels_s390x(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  use kernels::s390x::*;

  match policy.family() {
    KernelFamily::S390xVgfm => Crc64Kernels {
      reference,
      portable,
      primary: NVME_VGFM,
      primary_small: None,
      wide: None,
      wide_4x512: None,
    },
    _ => Crc64Kernels::portable_only(reference, portable),
  }
}

/// Build XZ kernel table for riscv64.
#[cfg(target_arch = "riscv64")]
#[must_use]
pub fn build_xz_kernels_riscv64(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  use kernels::riscv64::*;

  match policy.family() {
    KernelFamily::RiscvZvbc => Crc64Kernels {
      reference,
      portable,
      primary: XZ_ZBC,
      primary_small: None,
      wide: Some(XZ_ZVBC),
      wide_4x512: None,
    },
    KernelFamily::RiscvZbc => Crc64Kernels {
      reference,
      portable,
      primary: XZ_ZBC,
      primary_small: None,
      wide: None,
      wide_4x512: None,
    },
    _ => Crc64Kernels::portable_only(reference, portable),
  }
}

/// Build NVME kernel table for riscv64.
#[cfg(target_arch = "riscv64")]
#[must_use]
pub fn build_nvme_kernels_riscv64(policy: &Crc64Policy, reference: Crc64Fn, portable: Crc64Fn) -> Crc64Kernels {
  use kernels::riscv64::*;

  match policy.family() {
    KernelFamily::RiscvZvbc => Crc64Kernels {
      reference,
      portable,
      primary: NVME_ZBC,
      primary_small: None,
      wide: Some(NVME_ZVBC),
      wide_4x512: None,
    },
    KernelFamily::RiscvZbc => Crc64Kernels {
      reference,
      portable,
      primary: NVME_ZBC,
      primary_small: None,
      wide: None,
      wide_4x512: None,
    },
    _ => Crc64Kernels::portable_only(reference, portable),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_policy_portable_fallback() {
    let cfg = Crc64Config {
      requested_force: Crc64Force::Portable,
      effective_force: Crc64Force::Portable,
      tunables: super::super::config::Crc64Tunables {
        xz: super::super::config::Crc64VariantTunables {
          small_kernel_max_bytes: CRC64_SMALL_KERNEL_MAX_BYTES_DEFAULT,
          portable_to_clmul: 64,
          pclmul_to_vpclmul: 512,
          streams: 4,
          min_bytes_per_lane: None,
        },
        nvme: super::super::config::Crc64VariantTunables {
          small_kernel_max_bytes: CRC64_SMALL_KERNEL_MAX_BYTES_DEFAULT,
          portable_to_clmul: 64,
          pclmul_to_vpclmul: 512,
          streams: 4,
          min_bytes_per_lane: None,
        },
      },
    };

    let caps = Caps::NONE;
    let tune = Tune::DEFAULT;
    let policy = Crc64Policy::from_config(&cfg, cfg.tunables.xz, caps, &tune);

    assert_eq!(policy.family(), KernelFamily::Portable);
    assert_eq!(policy.effective_force, Crc64Force::Portable);
  }

  #[test]
  fn test_policy_reference_forced() {
    let cfg = Crc64Config {
      requested_force: Crc64Force::Reference,
      effective_force: Crc64Force::Reference,
      tunables: super::super::config::Crc64Tunables {
        xz: super::super::config::Crc64VariantTunables {
          small_kernel_max_bytes: CRC64_SMALL_KERNEL_MAX_BYTES_DEFAULT,
          portable_to_clmul: 64,
          pclmul_to_vpclmul: 512,
          streams: 4,
          min_bytes_per_lane: None,
        },
        nvme: super::super::config::Crc64VariantTunables {
          small_kernel_max_bytes: CRC64_SMALL_KERNEL_MAX_BYTES_DEFAULT,
          portable_to_clmul: 64,
          pclmul_to_vpclmul: 512,
          streams: 4,
          min_bytes_per_lane: None,
        },
      },
    };

    let caps = Caps::NONE;
    let tune = Tune::DEFAULT;
    let policy = Crc64Policy::from_config(&cfg, cfg.tunables.xz, caps, &tune);

    assert_eq!(policy.family(), KernelFamily::Reference);
  }

  #[test]
  fn test_kernel_name_portable() {
    let cfg = Crc64Config {
      requested_force: Crc64Force::Portable,
      effective_force: Crc64Force::Portable,
      tunables: super::super::config::Crc64Tunables {
        xz: super::super::config::Crc64VariantTunables {
          small_kernel_max_bytes: CRC64_SMALL_KERNEL_MAX_BYTES_DEFAULT,
          portable_to_clmul: 64,
          pclmul_to_vpclmul: 512,
          streams: 4,
          min_bytes_per_lane: None,
        },
        nvme: super::super::config::Crc64VariantTunables {
          small_kernel_max_bytes: CRC64_SMALL_KERNEL_MAX_BYTES_DEFAULT,
          portable_to_clmul: 64,
          pclmul_to_vpclmul: 512,
          streams: 4,
          min_bytes_per_lane: None,
        },
      },
    };

    let caps = Caps::NONE;
    let tune = Tune::DEFAULT;
    let policy = Crc64Policy::from_config(&cfg, cfg.tunables.xz, caps, &tune);

    assert_eq!(policy.kernel_name(100), kernels::PORTABLE);
    assert_eq!(policy.kernel_name(10000), kernels::PORTABLE);
  }
}
