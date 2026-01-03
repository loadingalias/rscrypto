//! CRC-24 specific policy computation and kernel dispatch.
//!
//! This module bridges the generic [`SelectionPolicy`] from `backend` with
//! CRC-24 specific configuration and kernel tables.
//!
//! # Design
//!
//! CRC-24 has a single variant (OpenPGP) and supports carryless-multiply
//! acceleration (PCLMULQDQ/PMULL) using the same width32 folding strategy as
//! CRC-16.

use backend::{KernelFamily, SelectionPolicy};
use platform::{Caps, Tune};

use super::{
  config::{Crc24Config, Crc24Force},
  kernels,
};
use crate::{common::kernels::stream_to_index, dispatchers::Crc24Fn};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Minimum bytes for SIMD dispatch (below this, portable is always faster).
#[allow(dead_code)] // Reserved for future small-buffer optimization
pub const CRC24_SMALL_THRESHOLD: usize = 16;

/// Block size for CRC-24 folding operations.
pub const CRC24_FOLD_BLOCK_BYTES: usize = 128;

// ─────────────────────────────────────────────────────────────────────────────
// Crc24Policy
// ─────────────────────────────────────────────────────────────────────────────

/// CRC-24 specific selection policy.
///
/// Wraps the generic [`SelectionPolicy`] with CRC-24 specific configuration.
#[derive(Clone, Copy, Debug)]
pub struct Crc24Policy {
  /// Generic selection policy (family, thresholds).
  #[allow(dead_code)] // Kept for API uniformity with CRC-64/CRC-32
  pub inner: SelectionPolicy,
  /// Effective force mode after clamping to capabilities.
  pub effective_force: Crc24Force,
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
  /// Minimum bytes per lane for multi-stream folding.
  pub min_bytes_per_lane: usize,
}

impl Crc24Policy {
  /// Create a policy from CRC-24 configuration and platform detection.
  #[must_use]
  pub fn from_config(cfg: &Crc24Config, caps: Caps, tune: &Tune) -> Self {
    let (has_clmul, has_vpclmul) = Self::detect_features(caps, tune);

    let mut inner = if let Some(family) = cfg.effective_force.to_family() {
      SelectionPolicy::with_family(caps, tune, family)
    } else {
      SelectionPolicy::from_platform(caps, tune)
    };

    // CRC-24 uses its own portable→SIMD threshold.
    inner.small_threshold = cfg.tunables.portable_to_clmul.max(CRC24_SMALL_THRESHOLD);

    // Apply tuned stream preference (still capped by the architecture limit).
    inner.cap_max_streams(cfg.tunables.streams);

    // Clamp to architectures where CRC-24 multi-stream kernels are implemented.
    if !Self::supports_multi_stream_kernels() {
      inner.cap_max_streams(1);
    }

    let min_bytes_per_lane = cfg
      .tunables
      .min_bytes_per_lane
      .unwrap_or_else(|| inner.family().min_bytes_per_lane());

    Self {
      inner,
      effective_force: cfg.effective_force,
      slice4_to_slice8: cfg.tunables.slice4_to_slice8,
      portable_to_clmul: cfg.tunables.portable_to_clmul,
      clmul_to_vpclmul: cfg.tunables.pclmul_to_vpclmul,
      has_clmul,
      has_vpclmul,
      min_bytes_per_lane,
    }
  }

  #[inline]
  const fn supports_multi_stream_kernels() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
      true
    }
    #[cfg(target_arch = "aarch64")]
    {
      true
    }
    #[cfg(target_arch = "powerpc64")]
    {
      true
    }
    #[cfg(target_arch = "s390x")]
    {
      true
    }
    #[cfg(target_arch = "riscv64")]
    {
      true
    }
    #[cfg(not(any(
      target_arch = "x86_64",
      target_arch = "aarch64",
      target_arch = "powerpc64",
      target_arch = "s390x",
      target_arch = "riscv64"
    )))]
    {
      false
    }
  }

  #[inline]
  fn detect_features(caps: Caps, tune: &Tune) -> (bool, bool) {
    #[cfg(target_arch = "x86_64")]
    {
      let _ = tune;
      (
        caps.has(platform::caps::x86::PCLMUL_READY),
        caps.has(platform::caps::x86::VPCLMUL_READY),
      )
    }
    #[cfg(target_arch = "aarch64")]
    {
      let _ = tune;
      (caps.has(platform::caps::aarch64::PMULL_READY), false)
    }
    #[cfg(target_arch = "powerpc64")]
    {
      let _ = tune;
      (caps.has(platform::caps::power::VPMSUM_READY), false)
    }
    #[cfg(target_arch = "s390x")]
    {
      let _ = tune;
      (caps.has(platform::caps::s390x::VECTOR), false)
    }
    #[cfg(target_arch = "riscv64")]
    {
      use platform::caps::riscv;
      let has_zbc = caps.has(riscv::ZBC);
      let has_zvbc = caps.has(riscv::ZVBC) && tune.effective_simd_width >= 256;
      (has_zbc || has_zvbc, has_zvbc)
    }
    #[cfg(not(any(
      target_arch = "x86_64",
      target_arch = "aarch64",
      target_arch = "powerpc64",
      target_arch = "s390x",
      target_arch = "riscv64"
    )))]
    {
      let _ = (caps, tune);
      (false, false)
    }
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

  /// Get the optimal stream count for this buffer length.
  #[inline]
  #[must_use]
  pub fn streams_for_len(&self, len: usize) -> u8 {
    if !Self::supports_multi_stream_kernels() {
      return 1;
    }
    self.inner.streams_for_len_with_min(len, self.min_bytes_per_lane)
  }

  /// Get kernel name for the given length (for introspection).
  #[must_use]
  pub fn kernel_name(&self, len: usize) -> &'static str {
    match self.effective_force {
      Crc24Force::Reference => return kernels::REFERENCE,
      Crc24Force::Slice4 => return kernels::PORTABLE_SLICE4,
      Crc24Force::Slice8 => return kernels::PORTABLE_SLICE8,
      Crc24Force::Portable => return self.portable_name_for_len(len),
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
    if len < CRC24_FOLD_BLOCK_BYTES {
      return kernels::x86_64::PCLMUL_SMALL;
    }
    let streams = self.streams_for_len(len);
    let idx = stream_to_index(streams);
    if self.has_vpclmul && len >= self.clmul_to_vpclmul {
      kernels::x86_64::VPCLMUL_NAMES
        .get(idx)
        .or(kernels::x86_64::VPCLMUL_NAMES.first())
        .copied()
        .unwrap_or(kernels::x86_64::VPCLMUL)
    } else {
      kernels::x86_64::PCLMUL_NAMES
        .get(idx)
        .or(kernels::x86_64::PCLMUL_NAMES.first())
        .copied()
        .unwrap_or(kernels::x86_64::PCLMUL)
    }
  }

  #[cfg(target_arch = "aarch64")]
  fn kernel_name_for_family(&self, len: usize) -> &'static str {
    if len < CRC24_FOLD_BLOCK_BYTES {
      return kernels::aarch64::PMULL_SMALL;
    }
    let streams = self.streams_for_len(len);
    let idx = stream_to_index(streams);
    kernels::aarch64::PMULL_NAMES
      .get(idx)
      .or(kernels::aarch64::PMULL_NAMES.first())
      .copied()
      .unwrap_or(kernels::aarch64::PMULL)
  }

  #[cfg(target_arch = "powerpc64")]
  fn kernel_name_for_family(&self, len: usize) -> &'static str {
    let streams = self.streams_for_len(len);
    let idx = stream_to_index(streams);
    kernels::power::VPMSUM_NAMES
      .get(idx)
      .or(kernels::power::VPMSUM_NAMES.first())
      .copied()
      .unwrap_or(kernels::power::VPMSUM)
  }

  #[cfg(target_arch = "s390x")]
  fn kernel_name_for_family(&self, len: usize) -> &'static str {
    let streams = self.streams_for_len(len);
    let idx = stream_to_index(streams);
    kernels::s390x::VGFM_NAMES
      .get(idx)
      .or(kernels::s390x::VGFM_NAMES.first())
      .copied()
      .unwrap_or(kernels::s390x::VGFM)
  }

  #[cfg(target_arch = "riscv64")]
  fn kernel_name_for_family(&self, len: usize) -> &'static str {
    let streams = self.streams_for_len(len);
    let idx = stream_to_index(streams);
    if self.has_vpclmul && len >= self.clmul_to_vpclmul {
      kernels::riscv64::ZVBC_NAMES
        .get(idx)
        .or(kernels::riscv64::ZVBC_NAMES.first())
        .copied()
        .unwrap_or(kernels::riscv64::ZVBC)
    } else {
      kernels::riscv64::ZBC_NAMES
        .get(idx)
        .or(kernels::riscv64::ZBC_NAMES.first())
        .copied()
        .unwrap_or(kernels::riscv64::ZBC)
    }
  }

  #[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "s390x",
    target_arch = "riscv64"
  )))]
  fn kernel_name_for_family(&self, len: usize) -> &'static str {
    self.portable_name_for_len(len)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Crc24Kernels - Pre-resolved kernel table
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-resolved kernel function pointers for CRC-24.
#[derive(Clone, Copy, Debug)]
pub struct Crc24Kernels {
  /// Reference (bitwise) kernel - always available.
  pub reference: Crc24Fn,
  /// Portable slice-by-4 kernel - always available.
  pub slice4: Crc24Fn,
  /// Portable slice-by-8 kernel - always available.
  pub slice8: Crc24Fn,
  /// CLMUL/PMULL kernel array (if available), indexed by `stream_to_index`.
  pub clmul: Option<[Crc24Fn; 5]>,
  /// Small-buffer kernel for the CLMUL/PMULL family (< fold block).
  pub clmul_small: Option<Crc24Fn>,
  /// VPCLMUL kernel (x86_64 only, if available).
  #[allow(dead_code)] // Only used on x86_64
  pub vpclmul: Option<[Crc24Fn; 5]>,
}

impl Crc24Kernels {
  /// Create a portable-only kernel table.
  #[must_use]
  #[allow(dead_code)] // Only used on non-SIMD architectures
  pub const fn portable_only(reference: Crc24Fn, slice4: Crc24Fn, slice8: Crc24Fn) -> Self {
    Self {
      reference,
      slice4,
      slice8,
      clmul: None,
      clmul_small: None,
      vpclmul: None,
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
  let len = data.len();

  // Fast path: tiny buffers
  if len < CRC24_SMALL_THRESHOLD {
    return portable_auto(policy, kernels, crc, data);
  }

  // Handle forced modes
  match policy.effective_force {
    Crc24Force::Reference => return (kernels.reference)(crc, data),
    Crc24Force::Slice4 => return (kernels.slice4)(crc, data),
    Crc24Force::Slice8 => return (kernels.slice8)(crc, data),
    Crc24Force::Portable => return portable_auto(policy, kernels, crc, data),
    Crc24Force::Clmul => return policy_dispatch_simd(policy, kernels, crc, data),
    _ => {}
  }

  if !policy.should_use_simd(len) {
    return portable_auto(policy, kernels, crc, data);
  }

  policy_dispatch_simd(policy, kernels, crc, data)
}

#[inline]
fn portable_auto(policy: &Crc24Policy, kernels: &Crc24Kernels, crc: u32, data: &[u8]) -> u32 {
  if data.len() < policy.slice4_to_slice8 {
    (kernels.slice4)(crc, data)
  } else {
    (kernels.slice8)(crc, data)
  }
}

#[cfg(target_arch = "x86_64")]
fn policy_dispatch_simd(policy: &Crc24Policy, kernels: &Crc24Kernels, crc: u32, data: &[u8]) -> u32 {
  let len = data.len();

  if len < CRC24_FOLD_BLOCK_BYTES
    && let Some(small) = kernels.clmul_small
  {
    return small(crc, data);
  }

  if policy.has_vpclmul
    && len >= policy.clmul_to_vpclmul
    && let Some(vpclmul) = kernels.vpclmul
  {
    let streams = policy.streams_for_len(len);
    let idx = stream_to_index(streams);
    if let Some(&func) = vpclmul.get(idx) {
      return func(crc, data);
    }
    if let Some(&func) = vpclmul.first() {
      return func(crc, data);
    }
  }

  if let Some(clmul) = kernels.clmul {
    let streams = policy.streams_for_len(len);
    let idx = stream_to_index(streams);
    if let Some(&func) = clmul.get(idx) {
      return func(crc, data);
    }
    if let Some(&func) = clmul.first() {
      return func(crc, data);
    }
  }

  portable_auto(policy, kernels, crc, data)
}

#[cfg(target_arch = "aarch64")]
fn policy_dispatch_simd(policy: &Crc24Policy, kernels: &Crc24Kernels, crc: u32, data: &[u8]) -> u32 {
  if data.len() < CRC24_FOLD_BLOCK_BYTES
    && let Some(small) = kernels.clmul_small
  {
    return small(crc, data);
  }

  if let Some(clmul) = kernels.clmul {
    let streams = policy.streams_for_len(data.len());
    let idx = stream_to_index(streams);
    if let Some(&func) = clmul.get(idx) {
      return func(crc, data);
    }
    if let Some(&func) = clmul.first() {
      return func(crc, data);
    }
  }
  portable_auto(policy, kernels, crc, data)
}

#[cfg(target_arch = "powerpc64")]
fn policy_dispatch_simd(policy: &Crc24Policy, kernels: &Crc24Kernels, crc: u32, data: &[u8]) -> u32 {
  if let Some(clmul) = kernels.clmul {
    let streams = policy.streams_for_len(data.len());
    let idx = stream_to_index(streams);
    if let Some(&func) = clmul.get(idx) {
      return func(crc, data);
    }
    if let Some(&func) = clmul.first() {
      return func(crc, data);
    }
  }
  portable_auto(policy, kernels, crc, data)
}

#[cfg(target_arch = "s390x")]
fn policy_dispatch_simd(policy: &Crc24Policy, kernels: &Crc24Kernels, crc: u32, data: &[u8]) -> u32 {
  if let Some(clmul) = kernels.clmul {
    let streams = policy.streams_for_len(data.len());
    let idx = stream_to_index(streams);
    if let Some(&func) = clmul.get(idx) {
      return func(crc, data);
    }
    if let Some(&func) = clmul.first() {
      return func(crc, data);
    }
  }
  portable_auto(policy, kernels, crc, data)
}

#[cfg(target_arch = "riscv64")]
fn policy_dispatch_simd(policy: &Crc24Policy, kernels: &Crc24Kernels, crc: u32, data: &[u8]) -> u32 {
  if policy.has_vpclmul
    && data.len() >= policy.clmul_to_vpclmul
    && let Some(vpclmul) = kernels.vpclmul
  {
    let streams = policy.streams_for_len(data.len());
    let idx = stream_to_index(streams);
    if let Some(&func) = vpclmul.get(idx) {
      return func(crc, data);
    }
    if let Some(&func) = vpclmul.first() {
      return func(crc, data);
    }
  }

  if let Some(clmul) = kernels.clmul {
    let streams = policy.streams_for_len(data.len());
    let idx = stream_to_index(streams);
    if let Some(&func) = clmul.get(idx) {
      return func(crc, data);
    }
    if let Some(&func) = clmul.first() {
      return func(crc, data);
    }
  }

  portable_auto(policy, kernels, crc, data)
}

#[cfg(not(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "powerpc64",
  target_arch = "s390x",
  target_arch = "riscv64"
)))]
fn policy_dispatch_simd(policy: &Crc24Policy, kernels: &Crc24Kernels, crc: u32, data: &[u8]) -> u32 {
  portable_auto(policy, kernels, crc, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Builder
// ─────────────────────────────────────────────────────────────────────────────

/// Build OpenPGP kernel table for x86_64.
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn build_openpgp_kernels_x86(
  policy: &Crc24Policy,
  reference: Crc24Fn,
  slice4: Crc24Fn,
  slice8: Crc24Fn,
) -> Crc24Kernels {
  Crc24Kernels {
    reference,
    slice4,
    slice8,
    clmul: if policy.has_clmul {
      Some(kernels::x86_64::OPENPGP_PCLMUL)
    } else {
      None
    },
    clmul_small: if policy.has_clmul {
      Some(kernels::x86_64::OPENPGP_PCLMUL_SMALL_KERNEL)
    } else {
      None
    },
    vpclmul: if policy.has_vpclmul {
      Some(kernels::x86_64::OPENPGP_VPCLMUL)
    } else {
      None
    },
  }
}

/// Build OpenPGP kernel table for aarch64.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn build_openpgp_kernels_aarch64(
  policy: &Crc24Policy,
  reference: Crc24Fn,
  slice4: Crc24Fn,
  slice8: Crc24Fn,
) -> Crc24Kernels {
  use super::aarch64;

  Crc24Kernels {
    reference,
    slice4,
    slice8,
    clmul: if policy.has_clmul {
      Some([
        aarch64::crc24_openpgp_pmull_safe,
        aarch64::crc24_openpgp_pmull_2way_safe,
        aarch64::crc24_openpgp_pmull_3way_safe,
        aarch64::crc24_openpgp_pmull_3way_safe,
        aarch64::crc24_openpgp_pmull_3way_safe,
      ])
    } else {
      None
    },
    clmul_small: if policy.has_clmul {
      Some(kernels::aarch64::OPENPGP_PMULL_SMALL_KERNEL)
    } else {
      None
    },
    vpclmul: None,
  }
}

/// Build OpenPGP kernel table for Power.
#[cfg(target_arch = "powerpc64")]
#[must_use]
pub fn build_openpgp_kernels_power(
  policy: &Crc24Policy,
  reference: Crc24Fn,
  slice4: Crc24Fn,
  slice8: Crc24Fn,
) -> Crc24Kernels {
  Crc24Kernels {
    reference,
    slice4,
    slice8,
    clmul: if policy.has_clmul {
      Some(kernels::power::OPENPGP_VPMSUM)
    } else {
      None
    },
    clmul_small: None,
    vpclmul: None,
  }
}

/// Build OpenPGP kernel table for s390x.
#[cfg(target_arch = "s390x")]
#[must_use]
pub fn build_openpgp_kernels_s390x(
  policy: &Crc24Policy,
  reference: Crc24Fn,
  slice4: Crc24Fn,
  slice8: Crc24Fn,
) -> Crc24Kernels {
  Crc24Kernels {
    reference,
    slice4,
    slice8,
    clmul: if policy.has_clmul {
      Some(kernels::s390x::OPENPGP_VGFM)
    } else {
      None
    },
    clmul_small: None,
    vpclmul: None,
  }
}

/// Build OpenPGP kernel table for riscv64.
#[cfg(target_arch = "riscv64")]
#[must_use]
pub fn build_openpgp_kernels_riscv64(
  policy: &Crc24Policy,
  reference: Crc24Fn,
  slice4: Crc24Fn,
  slice8: Crc24Fn,
) -> Crc24Kernels {
  use platform::caps::riscv;

  let caps = platform::caps();

  Crc24Kernels {
    reference,
    slice4,
    slice8,
    clmul: if policy.has_clmul && caps.has(riscv::ZBC) {
      Some(kernels::riscv64::OPENPGP_ZBC)
    } else {
      None
    },
    clmul_small: None,
    vpclmul: if policy.has_vpclmul && caps.has(riscv::ZVBC) {
      Some(kernels::riscv64::OPENPGP_ZVBC)
    } else {
      None
    },
  }
}

/// Build OpenPGP kernel table for non-SIMD architectures.
#[cfg(not(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "powerpc64",
  target_arch = "s390x",
  target_arch = "riscv64"
)))]
#[must_use]
pub fn build_openpgp_kernels_generic(reference: Crc24Fn, slice4: Crc24Fn, slice8: Crc24Fn) -> Crc24Kernels {
  Crc24Kernels::portable_only(reference, slice4, slice8)
}
