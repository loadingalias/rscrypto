//! CRC-32 specific policy computation and kernel dispatch.
//!
//! This module bridges the generic [`SelectionPolicy`] from `backend` with
//! CRC-32 specific configuration and kernel tables.
//!
//! # Design
//!
//! CRC-32 has more complexity than CRC-64:
//! - Two variants: IEEE (Ethernet) and Castagnoli (iSCSI)
//! - Hardware CRC tier: x86_64 (CRC-32C only), aarch64 (both variants)
//! - Fusion kernels: Combining HWCRC + carryless multiply for best performance
//!
//! The policy is computed once from:
//! - [`Crc32Config`] - user overrides and tunables
//! - [`Caps`] - detected CPU capabilities
//! - [`Tune`] - microarchitecture-specific hints
//!
//! The resulting [`Crc32Policy`] is cached and used for all dispatch decisions,
//! eliminating per-call branching on capabilities and thresholds.

use backend::{KernelFamily, KernelTier, SelectionPolicy};
use platform::{Caps, Tune};

use super::{
  config::{Crc32Config, Crc32Force},
  kernels,
};
use crate::{common::kernels::stream_to_index, dispatchers::Crc32Fn};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Minimum bytes for SIMD dispatch (below this, portable is always faster).
pub const CRC32_SMALL_THRESHOLD: usize = 16;

/// Block size for CRC-32 folding operations.
pub const CRC32_FOLD_BLOCK_BYTES: usize = 128;

// ─────────────────────────────────────────────────────────────────────────────
// Crc32Variant - Distinguishes IEEE from Castagnoli
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32 polynomial variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Crc32Variant {
  /// CRC-32 IEEE (Ethernet, HDLC, etc.)
  Ieee,
  /// CRC-32C Castagnoli (iSCSI, ext4, etc.)
  Castagnoli,
}

// ─────────────────────────────────────────────────────────────────────────────
// Crc32Policy
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-32 specific selection policy.
///
/// Wraps the generic [`SelectionPolicy`] with CRC-32 specific configuration
/// and provides dispatch methods for both IEEE and Castagnoli variants.
#[derive(Clone, Copy, Debug)]
pub struct Crc32Policy {
  /// Generic selection policy (family, thresholds, streams).
  pub inner: SelectionPolicy,
  /// Effective force mode after clamping to capabilities.
  pub effective_force: Crc32Force,
  /// CRC-32 variant (IEEE or Castagnoli).
  pub variant: Crc32Variant,
  /// Threshold for portable→hwcrc transition.
  pub portable_to_hwcrc: usize,
  /// Threshold for hwcrc→fusion transition.
  pub hwcrc_to_fusion: usize,
  /// Threshold for fusion→vpclmul transition (x86_64 only).
  #[allow(dead_code)] // Only used on x86_64
  pub fusion_to_vpclmul: usize,
  /// Threshold for fusion→avx512 transition (x86_64 CRC-32C only).
  #[allow(dead_code)] // Only used on x86_64
  pub fusion_to_avx512: usize,
  /// Whether hardware CRC is available for this variant.
  pub has_hwcrc: bool,
  /// Whether fusion kernels are available for this variant.
  pub has_fusion: bool,
  /// Whether VPCLMUL fusion is available (x86_64 only).
  #[allow(dead_code)] // Only used on x86_64
  pub has_vpclmul: bool,
  /// Whether AVX-512 fusion is available (x86_64 CRC-32C only).
  #[allow(dead_code)] // Only used on x86_64
  pub has_avx512: bool,
  /// Minimum bytes per lane for multi-stream folding.
  ///
  /// This is resolved from tunables (if set) or the kernel family default.
  pub min_bytes_per_lane: usize,
  /// Whether this variant is memory-bandwidth limited on this microarch.
  ///
  /// When true, multi-stream processing is suppressed for HWCRC and fusion
  /// kernels (both use hardware CRC as the combine step).
  pub memory_bound: bool,
}

impl Crc32Policy {
  /// Create a policy from CRC-32 configuration and platform detection.
  #[must_use]
  pub fn from_config(cfg: &Crc32Config, caps: Caps, tune: &Tune, variant: Crc32Variant) -> Self {
    // Detect available features
    let (has_hwcrc, has_fusion, has_vpclmul, has_avx512) = Self::detect_features(caps, variant);

    // Map Crc32Force to ForceMode if explicitly set
    let inner = if let Some(family) = cfg.effective_force.to_family() {
      SelectionPolicy::with_family(caps, tune, family)
    } else {
      // Auto selection
      let mut policy = SelectionPolicy::from_platform(caps, tune);
      // Apply CRC-32 specific thresholds from tunables
      policy.small_threshold = cfg.tunables.portable_to_hwcrc.max(CRC32_SMALL_THRESHOLD);
      policy
    };

    // Resolve min_bytes_per_lane: tunables override > family default
    let min_bytes_per_lane = match variant {
      Crc32Variant::Ieee => cfg.tunables.min_bytes_per_lane_crc32,
      Crc32Variant::Castagnoli => cfg.tunables.min_bytes_per_lane_crc32c,
    }
    .unwrap_or_else(|| inner.family().min_bytes_per_lane());

    // Memory-bound heuristic: suppress multi-stream for CRC-32C when HWCRC
    // is bandwidth-limited. This applies to both pure HWCRC AND fusion kernels
    // (fusion still uses HWCRC as the combine step, so it's also bandwidth-limited).
    let memory_bound = variant == Crc32Variant::Castagnoli && tune.memory_bound_hwcrc && has_hwcrc;

    Self {
      inner,
      effective_force: cfg.effective_force,
      variant,
      portable_to_hwcrc: cfg.tunables.portable_to_hwcrc,
      hwcrc_to_fusion: cfg.tunables.hwcrc_to_fusion,
      fusion_to_vpclmul: cfg.tunables.fusion_to_vpclmul,
      fusion_to_avx512: cfg.tunables.fusion_to_avx512,
      has_hwcrc,
      has_fusion,
      has_vpclmul,
      has_avx512,
      min_bytes_per_lane,
      memory_bound,
    }
  }

  /// Detect available features for a given variant.
  #[allow(unused_variables)] // caps only used on specific archs
  fn detect_features(caps: Caps, variant: Crc32Variant) -> (bool, bool, bool, bool) {
    #[cfg(target_arch = "x86_64")]
    {
      let has_hwcrc = variant == Crc32Variant::Castagnoli && caps.has(platform::caps::x86::CRC32C_READY);
      let has_pclmul = caps.has(platform::caps::x86::PCLMUL_READY);
      let has_vpclmul = caps.has(platform::caps::x86::VPCLMUL_READY);
      let has_avx512 = caps.has(platform::caps::x86::AVX512_READY) && caps.has(platform::caps::x86::PCLMULQDQ);

      // For IEEE: no HWCRC, just PCLMUL folding
      // For Castagnoli: HWCRC + fusion (HWCRC + PCLMUL)
      let has_fusion = match variant {
        Crc32Variant::Ieee => has_pclmul,                    // IEEE uses pure PCLMUL folding
        Crc32Variant::Castagnoli => has_hwcrc && has_pclmul, // CRC-32C needs both
      };

      (has_hwcrc, has_fusion, has_vpclmul, has_avx512)
    }

    #[cfg(target_arch = "aarch64")]
    {
      let has_hwcrc = caps.has(platform::caps::aarch64::CRC_READY);
      let has_pmull = caps.has(platform::caps::aarch64::PMULL_READY);
      let has_fusion = has_hwcrc && has_pmull;

      (has_hwcrc, has_fusion, false, false)
    }

    #[cfg(target_arch = "powerpc64")]
    {
      let has_vpmsum = caps.has(platform::caps::powerpc64::VPMSUM_READY);
      (false, has_vpmsum, false, false)
    }

    #[cfg(target_arch = "s390x")]
    {
      let has_vgfm = caps.has(platform::caps::s390x::VECTOR);
      (false, has_vgfm, false, false)
    }

    #[cfg(target_arch = "riscv64")]
    {
      let has_zvbc = caps.has(platform::caps::riscv::ZVBC);
      let has_zbc = caps.has(platform::caps::riscv::ZBC);
      (false, has_zbc || has_zvbc, has_zvbc, false)
    }

    #[cfg(not(any(
      target_arch = "x86_64",
      target_arch = "aarch64",
      target_arch = "powerpc64",
      target_arch = "s390x",
      target_arch = "riscv64"
    )))]
    {
      (false, false, false, false)
    }
  }

  /// Get the kernel tier.
  #[inline]
  #[must_use]
  #[allow(dead_code)] // Used for introspection
  pub const fn tier(&self) -> KernelTier {
    self.inner.tier()
  }

  /// Get the selected kernel family.
  #[inline]
  #[must_use]
  #[allow(dead_code)] // For API uniformity with CRC-64
  pub const fn family(&self) -> KernelFamily {
    self.inner.family()
  }

  /// Check if SIMD should be used for this buffer length.
  #[inline]
  #[must_use]
  pub fn should_use_simd(&self, len: usize) -> bool {
    len >= self.portable_to_hwcrc
  }

  /// Get the optimal stream count for this buffer length.
  ///
  /// Uses the algorithm-specific `min_bytes_per_lane` to determine when
  /// multi-stream processing is worthwhile. When `memory_bound` is set
  /// (CRC-32C on bandwidth-limited microarchs), returns 1 to suppress
  /// parallelization.
  #[inline]
  #[must_use]
  pub fn streams_for_len(&self, len: usize) -> u8 {
    // Memory-bound: suppress multi-stream (HWCRC already saturates bandwidth)
    if self.memory_bound {
      return 1;
    }
    self.inner.streams_for_len_with_min(len, self.min_bytes_per_lane)
  }

  /// Get the kernel name for this policy and buffer length.
  #[must_use]
  pub fn kernel_name(&self, len: usize) -> &'static str {
    if len < CRC32_SMALL_THRESHOLD {
      return kernels::PORTABLE;
    }

    if !self.should_use_simd(len) {
      return kernels::PORTABLE;
    }

    match self.effective_force {
      Crc32Force::Reference => return kernels::REFERENCE,
      Crc32Force::Portable => return kernels::PORTABLE,
      _ => {}
    }

    // Dispatch based on architecture and variant
    self.kernel_name_for_family(len)
  }

  #[cfg(target_arch = "x86_64")]
  fn kernel_name_for_family(&self, len: usize) -> &'static str {
    use kernels::x86_64::*;
    let streams = self.streams_for_len(len);
    let idx = stream_to_index(streams);

    match self.variant {
      Crc32Variant::Ieee => {
        // IEEE: no HWCRC on x86, just PCLMUL/VPCLMUL
        if !self.has_fusion || len < self.hwcrc_to_fusion {
          return kernels::PORTABLE;
        }
        if self.has_vpclmul && len >= self.fusion_to_vpclmul {
          return CRC32_VPCLMUL_NAMES.get(idx).copied().unwrap_or(kernels::PORTABLE);
        }
        CRC32_PCLMUL_NAMES.get(idx).copied().unwrap_or(kernels::PORTABLE)
      }
      Crc32Variant::Castagnoli => {
        if !self.has_hwcrc {
          return kernels::PORTABLE;
        }
        if len >= self.hwcrc_to_fusion && self.has_fusion {
          if self.has_vpclmul && len >= self.fusion_to_vpclmul {
            return CRC32C_FUSION_VPCLMUL_NAMES
              .get(idx)
              .copied()
              .unwrap_or(kernels::PORTABLE);
          }
          if self.has_avx512 && len >= self.fusion_to_avx512 {
            return CRC32C_FUSION_AVX512_NAMES
              .get(idx)
              .copied()
              .unwrap_or(kernels::PORTABLE);
          }
          return CRC32C_FUSION_SSE_NAMES.get(idx).copied().unwrap_or(kernels::PORTABLE);
        }
        CRC32C_HWCRC_NAMES.get(idx).copied().unwrap_or(kernels::PORTABLE)
      }
    }
  }

  #[cfg(target_arch = "aarch64")]
  fn kernel_name_for_family(&self, len: usize) -> &'static str {
    use kernels::aarch64::*;
    let streams = self.streams_for_len(len);
    let idx = stream_to_index(streams);

    if !self.has_hwcrc {
      return kernels::PORTABLE;
    }

    let (hwcrc_names, fusion_names) = match self.variant {
      Crc32Variant::Ieee => (CRC32_HWCRC_NAMES, CRC32_PMULL_NAMES),
      Crc32Variant::Castagnoli => (CRC32C_HWCRC_NAMES, CRC32C_PMULL_NAMES),
    };

    if len >= self.hwcrc_to_fusion && self.has_fusion {
      return fusion_names.get(idx).copied().unwrap_or(kernels::PORTABLE);
    }
    hwcrc_names.get(idx).copied().unwrap_or(kernels::PORTABLE)
  }

  #[cfg(target_arch = "powerpc64")]
  fn kernel_name_for_family(&self, len: usize) -> &'static str {
    use kernels::powerpc64::*;
    let streams = self.streams_for_len(len);
    let idx = stream_to_index(streams);

    if !self.has_fusion || len < self.hwcrc_to_fusion {
      return kernels::PORTABLE;
    }

    let names = match self.variant {
      Crc32Variant::Ieee => CRC32_VPMSUM_NAMES,
      Crc32Variant::Castagnoli => CRC32C_VPMSUM_NAMES,
    };
    names.get(idx).copied().unwrap_or(kernels::PORTABLE)
  }

  #[cfg(target_arch = "s390x")]
  fn kernel_name_for_family(&self, len: usize) -> &'static str {
    use kernels::s390x::*;
    let streams = self.streams_for_len(len);
    let idx = stream_to_index(streams);

    if !self.has_fusion || len < self.hwcrc_to_fusion {
      return kernels::PORTABLE;
    }

    let names = match self.variant {
      Crc32Variant::Ieee => CRC32_VGFM_NAMES,
      Crc32Variant::Castagnoli => CRC32C_VGFM_NAMES,
    };
    names.get(idx).copied().unwrap_or(kernels::PORTABLE)
  }

  #[cfg(target_arch = "riscv64")]
  fn kernel_name_for_family(&self, len: usize) -> &'static str {
    use kernels::riscv64::*;
    let streams = self.streams_for_len(len);
    let idx = stream_to_index(streams);

    if !self.has_fusion || len < self.hwcrc_to_fusion {
      return kernels::PORTABLE;
    }

    let (zbc_names, zvbc_names) = match self.variant {
      Crc32Variant::Ieee => (CRC32_ZBC_NAMES, CRC32_ZVBC_NAMES),
      Crc32Variant::Castagnoli => (CRC32C_ZBC_NAMES, CRC32C_ZVBC_NAMES),
    };

    if self.has_vpclmul {
      return zvbc_names.get(idx).copied().unwrap_or(kernels::PORTABLE);
    }
    zbc_names.get(idx).copied().unwrap_or(kernels::PORTABLE)
  }

  #[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "s390x",
    target_arch = "riscv64"
  )))]
  fn kernel_name_for_family(&self, _len: usize) -> &'static str {
    kernels::PORTABLE
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Crc32Kernels - Pre-resolved kernel table
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-resolved kernel function pointers for a CRC-32 variant.
///
/// This eliminates runtime kernel array lookups by caching all needed
/// function pointers at policy initialization time.
#[derive(Clone, Copy, Debug)]
pub struct Crc32Kernels {
  /// Reference (bitwise) kernel - always available.
  pub reference: Crc32Fn,
  /// Portable (slice-by-16) kernel - always available.
  pub portable: Crc32Fn,
  /// Hardware CRC kernels indexed by stream count (aarch64, x86_64 CRC-32C only).
  pub hwcrc: Option<[Crc32Fn; 5]>,
  /// Primary SIMD kernels indexed by stream count.
  /// For x86_64 IEEE: PCLMUL, for CRC-32C: fusion (HWCRC + PCLMUL).
  pub primary: [Crc32Fn; 5],
  /// Small-buffer kernel for primary family (< fold block).
  pub primary_small: Option<Crc32Fn>,
  /// Wide family kernels (VPCLMUL, AVX-512 fusion, etc.).
  #[allow(dead_code)] // Only used on x86_64 for VPCLMUL
  pub wide: Option<[Crc32Fn; 5]>,
}

impl Crc32Kernels {
  /// Create a portable-only kernel table.
  #[must_use]
  pub const fn portable_only(reference: Crc32Fn, portable: Crc32Fn) -> Self {
    Self {
      reference,
      portable,
      hwcrc: None,
      primary: [portable; 5],
      primary_small: None,
      wide: None,
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch Function
// ─────────────────────────────────────────────────────────────────────────────

/// Dispatch CRC-32 computation using pre-computed policy and kernels.
///
/// This is the new hot path - policy and kernels are pre-computed,
/// so dispatch is just threshold checks and function pointer calls.
#[inline]
pub fn policy_dispatch(policy: &Crc32Policy, kernels: &Crc32Kernels, crc: u32, data: &[u8]) -> u32 {
  let len = data.len();

  // Fast path: tiny buffers
  if len < CRC32_SMALL_THRESHOLD {
    return (kernels.portable)(crc, data);
  }

  // Handle forced modes
  match policy.effective_force {
    Crc32Force::Reference => return (kernels.reference)(crc, data),
    Crc32Force::Portable => return (kernels.portable)(crc, data),
    _ => {}
  }

  // Below SIMD threshold: use portable
  if !policy.should_use_simd(len) {
    return (kernels.portable)(crc, data);
  }

  // Architecture-specific dispatch
  policy_dispatch_simd(policy, kernels, crc, data)
}

#[cfg(target_arch = "x86_64")]
fn policy_dispatch_simd(policy: &Crc32Policy, kernels: &Crc32Kernels, crc: u32, data: &[u8]) -> u32 {
  let len = data.len();
  let streams = policy.streams_for_len(len);
  let idx = stream_to_index(streams);

  match policy.variant {
    Crc32Variant::Ieee => {
      // IEEE: no HWCRC on x86, just PCLMUL/VPCLMUL
      if !policy.has_fusion || len < policy.hwcrc_to_fusion {
        return (kernels.portable)(crc, data);
      }

      // Wide tier (VPCLMUL)
      if let Some(ref wide) = kernels.wide
        && policy.has_vpclmul
        && len >= policy.fusion_to_vpclmul
      {
        return (wide.get(idx).copied().unwrap_or(wide[0]))(crc, data);
      }

      // Small buffer kernel
      if len < CRC32_FOLD_BLOCK_BYTES
        && let Some(small) = kernels.primary_small
      {
        return (small)(crc, data);
      }

      // Primary PCLMUL
      (kernels.primary.get(idx).copied().unwrap_or(kernels.primary[0]))(crc, data)
    }
    Crc32Variant::Castagnoli => {
      if !policy.has_hwcrc {
        return (kernels.portable)(crc, data);
      }

      // Fusion tier (HWCRC + PCLMUL)
      if len >= policy.hwcrc_to_fusion && policy.has_fusion {
        // VPCLMUL fusion
        if let Some(ref wide) = kernels.wide
          && policy.has_vpclmul
          && len >= policy.fusion_to_vpclmul
        {
          return (wide.get(idx).copied().unwrap_or(wide[0]))(crc, data);
        }

        // Primary fusion (SSE or AVX-512)
        return (kernels.primary.get(idx).copied().unwrap_or(kernels.primary[0]))(crc, data);
      }

      // HWCRC tier
      if let Some(ref hwcrc) = kernels.hwcrc {
        return (hwcrc.get(idx).copied().unwrap_or(hwcrc[0]))(crc, data);
      }
      (kernels.portable)(crc, data)
    }
  }
}

#[cfg(target_arch = "aarch64")]
fn policy_dispatch_simd(policy: &Crc32Policy, kernels: &Crc32Kernels, crc: u32, data: &[u8]) -> u32 {
  let len = data.len();
  let streams = policy.streams_for_len(len);
  let idx = stream_to_index(streams);

  if !policy.has_hwcrc {
    return (kernels.portable)(crc, data);
  }

  // Fusion tier (CRC + PMULL)
  if len >= policy.hwcrc_to_fusion && policy.has_fusion {
    // Small buffer kernel
    if len < CRC32_FOLD_BLOCK_BYTES
      && let Some(small) = kernels.primary_small
    {
      return (small)(crc, data);
    }
    return (kernels.primary.get(idx).copied().unwrap_or(kernels.primary[0]))(crc, data);
  }

  // HWCRC tier
  if let Some(ref hwcrc) = kernels.hwcrc {
    return (hwcrc.get(idx).copied().unwrap_or(hwcrc[0]))(crc, data);
  }
  (kernels.portable)(crc, data)
}

#[cfg(target_arch = "powerpc64")]
fn policy_dispatch_simd(policy: &Crc32Policy, kernels: &Crc32Kernels, crc: u32, data: &[u8]) -> u32 {
  let len = data.len();

  if !policy.has_fusion || len < policy.hwcrc_to_fusion {
    return (kernels.portable)(crc, data);
  }

  let streams = policy.streams_for_len(len);
  let idx = stream_to_index(streams);
  (kernels.primary.get(idx).copied().unwrap_or(kernels.primary[0]))(crc, data)
}

#[cfg(target_arch = "s390x")]
fn policy_dispatch_simd(policy: &Crc32Policy, kernels: &Crc32Kernels, crc: u32, data: &[u8]) -> u32 {
  let len = data.len();

  if !policy.has_fusion || len < policy.hwcrc_to_fusion {
    return (kernels.portable)(crc, data);
  }

  let streams = policy.streams_for_len(len);
  let idx = stream_to_index(streams);
  (kernels.primary.get(idx).copied().unwrap_or(kernels.primary[0]))(crc, data)
}

#[cfg(target_arch = "riscv64")]
fn policy_dispatch_simd(policy: &Crc32Policy, kernels: &Crc32Kernels, crc: u32, data: &[u8]) -> u32 {
  let len = data.len();

  if !policy.has_fusion || len < policy.hwcrc_to_fusion {
    return (kernels.portable)(crc, data);
  }

  let streams = policy.streams_for_len(len);
  let idx = stream_to_index(streams);

  // Wide tier (ZVBC)
  if let Some(ref wide) = kernels.wide
    && policy.has_vpclmul
  {
    return (wide.get(idx).copied().unwrap_or(wide[0]))(crc, data);
  }

  // Primary tier (ZBC)
  (kernels.primary.get(idx).copied().unwrap_or(kernels.primary[0]))(crc, data)
}

#[cfg(not(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "powerpc64",
  target_arch = "s390x",
  target_arch = "riscv64"
)))]
fn policy_dispatch_simd(_policy: &Crc32Policy, kernels: &Crc32Kernels, crc: u32, data: &[u8]) -> u32 {
  (kernels.portable)(crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Builders (per architecture)
// ─────────────────────────────────────────────────────────────────────────────

/// Build IEEE kernel table for x86_64.
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn build_ieee_kernels_x86(policy: &Crc32Policy, reference: Crc32Fn, portable: Crc32Fn) -> Crc32Kernels {
  use kernels::x86_64::*;

  if !policy.has_fusion {
    return Crc32Kernels::portable_only(reference, portable);
  }

  Crc32Kernels {
    reference,
    portable,
    hwcrc: None, // No HWCRC for IEEE on x86
    primary: CRC32_PCLMUL,
    primary_small: Some(CRC32_PCLMUL_SMALL_KERNEL),
    wide: if policy.has_vpclmul { Some(CRC32_VPCLMUL) } else { None },
  }
}

/// Build Castagnoli kernel table for x86_64.
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn build_castagnoli_kernels_x86(policy: &Crc32Policy, reference: Crc32Fn, portable: Crc32Fn) -> Crc32Kernels {
  use kernels::x86_64::*;

  if !policy.has_hwcrc {
    return Crc32Kernels::portable_only(reference, portable);
  }

  // Select primary fusion tier based on available features
  let (primary, wide) = if policy.has_avx512 {
    // AVX-512 path: use AVX-512 fusion as primary, VPCLMUL fusion as wide
    if policy.has_vpclmul {
      (CRC32C_FUSION_AVX512, Some(CRC32C_FUSION_VPCLMUL))
    } else {
      (CRC32C_FUSION_AVX512, None)
    }
  } else if policy.has_fusion {
    // SSE path: use SSE fusion as primary, VPCLMUL as wide
    if policy.has_vpclmul {
      (CRC32C_FUSION_SSE, Some(CRC32C_FUSION_VPCLMUL))
    } else {
      (CRC32C_FUSION_SSE, None)
    }
  } else {
    // No fusion: HWCRC only
    (CRC32C_HWCRC, None)
  };

  Crc32Kernels {
    reference,
    portable,
    hwcrc: Some(CRC32C_HWCRC),
    primary,
    primary_small: None, // Fusion kernels don't have small-buffer variants
    wide,
  }
}

/// Build IEEE kernel table for aarch64.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn build_ieee_kernels_aarch64(policy: &Crc32Policy, reference: Crc32Fn, portable: Crc32Fn) -> Crc32Kernels {
  use kernels::aarch64::*;

  if !policy.has_hwcrc {
    return Crc32Kernels::portable_only(reference, portable);
  }

  Crc32Kernels {
    reference,
    portable,
    hwcrc: Some(CRC32_HWCRC),
    primary: if policy.has_fusion { CRC32_PMULL } else { CRC32_HWCRC },
    primary_small: if policy.has_fusion {
      Some(CRC32_PMULL_SMALL_KERNEL)
    } else {
      None
    },
    wide: None,
  }
}

/// Build Castagnoli kernel table for aarch64.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn build_castagnoli_kernels_aarch64(policy: &Crc32Policy, reference: Crc32Fn, portable: Crc32Fn) -> Crc32Kernels {
  use kernels::aarch64::*;

  if !policy.has_hwcrc {
    return Crc32Kernels::portable_only(reference, portable);
  }

  Crc32Kernels {
    reference,
    portable,
    hwcrc: Some(CRC32C_HWCRC),
    primary: if policy.has_fusion { CRC32C_PMULL } else { CRC32C_HWCRC },
    primary_small: if policy.has_fusion {
      Some(CRC32C_PMULL_SMALL_KERNEL)
    } else {
      None
    },
    wide: None,
  }
}

/// Build IEEE kernel table for powerpc64.
#[cfg(target_arch = "powerpc64")]
#[must_use]
pub fn build_ieee_kernels_powerpc64(policy: &Crc32Policy, reference: Crc32Fn, portable: Crc32Fn) -> Crc32Kernels {
  use kernels::powerpc64::*;

  if !policy.has_fusion {
    return Crc32Kernels::portable_only(reference, portable);
  }

  Crc32Kernels {
    reference,
    portable,
    hwcrc: None,
    primary: CRC32_VPMSUM,
    primary_small: None,
    wide: None,
  }
}

/// Build Castagnoli kernel table for powerpc64.
#[cfg(target_arch = "powerpc64")]
#[must_use]
pub fn build_castagnoli_kernels_powerpc64(policy: &Crc32Policy, reference: Crc32Fn, portable: Crc32Fn) -> Crc32Kernels {
  use kernels::powerpc64::*;

  if !policy.has_fusion {
    return Crc32Kernels::portable_only(reference, portable);
  }

  Crc32Kernels {
    reference,
    portable,
    hwcrc: None,
    primary: CRC32C_VPMSUM,
    primary_small: None,
    wide: None,
  }
}

/// Build IEEE kernel table for s390x.
#[cfg(target_arch = "s390x")]
#[must_use]
pub fn build_ieee_kernels_s390x(policy: &Crc32Policy, reference: Crc32Fn, portable: Crc32Fn) -> Crc32Kernels {
  use kernels::s390x::*;

  if !policy.has_fusion {
    return Crc32Kernels::portable_only(reference, portable);
  }

  Crc32Kernels {
    reference,
    portable,
    hwcrc: None,
    primary: CRC32_VGFM,
    primary_small: None,
    wide: None,
  }
}

/// Build Castagnoli kernel table for s390x.
#[cfg(target_arch = "s390x")]
#[must_use]
pub fn build_castagnoli_kernels_s390x(policy: &Crc32Policy, reference: Crc32Fn, portable: Crc32Fn) -> Crc32Kernels {
  use kernels::s390x::*;

  if !policy.has_fusion {
    return Crc32Kernels::portable_only(reference, portable);
  }

  Crc32Kernels {
    reference,
    portable,
    hwcrc: None,
    primary: CRC32C_VGFM,
    primary_small: None,
    wide: None,
  }
}

/// Build IEEE kernel table for riscv64.
#[cfg(target_arch = "riscv64")]
#[must_use]
pub fn build_ieee_kernels_riscv64(policy: &Crc32Policy, reference: Crc32Fn, portable: Crc32Fn) -> Crc32Kernels {
  use kernels::riscv64::*;

  if !policy.has_fusion {
    return Crc32Kernels::portable_only(reference, portable);
  }

  Crc32Kernels {
    reference,
    portable,
    hwcrc: None,
    primary: CRC32_ZBC,
    primary_small: None,
    wide: if policy.has_vpclmul { Some(CRC32_ZVBC) } else { None },
  }
}

/// Build Castagnoli kernel table for riscv64.
#[cfg(target_arch = "riscv64")]
#[must_use]
pub fn build_castagnoli_kernels_riscv64(policy: &Crc32Policy, reference: Crc32Fn, portable: Crc32Fn) -> Crc32Kernels {
  use kernels::riscv64::*;

  if !policy.has_fusion {
    return Crc32Kernels::portable_only(reference, portable);
  }

  Crc32Kernels {
    reference,
    portable,
    hwcrc: None,
    primary: CRC32C_ZBC,
    primary_small: None,
    wide: if policy.has_vpclmul { Some(CRC32C_ZVBC) } else { None },
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
    let cfg = Crc32Config {
      requested_force: Crc32Force::Portable,
      effective_force: Crc32Force::Portable,
      tunables: super::super::config::Crc32Tunables {
        portable_to_hwcrc: 64,
        hwcrc_to_fusion: 256,
        fusion_to_avx512: 1024,
        fusion_to_vpclmul: 2048,
        streams_crc32: 4,
        streams_crc32c: 4,
        min_bytes_per_lane_crc32: None,
        min_bytes_per_lane_crc32c: None,
      },
    };

    let caps = Caps::NONE;
    let tune = Tune::DEFAULT;
    let policy = Crc32Policy::from_config(&cfg, caps, &tune, Crc32Variant::Ieee);

    assert_eq!(policy.effective_force, Crc32Force::Portable);
  }

  #[test]
  fn test_kernel_name_portable() {
    let cfg = Crc32Config {
      requested_force: Crc32Force::Portable,
      effective_force: Crc32Force::Portable,
      tunables: super::super::config::Crc32Tunables {
        portable_to_hwcrc: 64,
        hwcrc_to_fusion: 256,
        fusion_to_avx512: 1024,
        fusion_to_vpclmul: 2048,
        streams_crc32: 4,
        streams_crc32c: 4,
        min_bytes_per_lane_crc32: None,
        min_bytes_per_lane_crc32c: None,
      },
    };

    let caps = Caps::NONE;
    let tune = Tune::DEFAULT;
    let policy = Crc32Policy::from_config(&cfg, caps, &tune, Crc32Variant::Ieee);

    assert_eq!(policy.kernel_name(100), kernels::PORTABLE);
    assert_eq!(policy.kernel_name(10000), kernels::PORTABLE);
  }
}
