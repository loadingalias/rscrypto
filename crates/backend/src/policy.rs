//! Pre-computed kernel selection policies.
//!
//! A `SelectionPolicy` captures all dispatch decisions at initialization time,
//! eliminating per-call capability checks and threshold comparisons.
//!
//! # Design Goals
//!
//! 1. **Zero per-call overhead**: Policy is computed once and cached
//! 2. **Branchless stream selection**: `streams_for_len()` uses arithmetic
//! 3. **no_std compatible**: All types are `Copy` and heap-free
//! 4. **Algorithm-agnostic**: Same policy structure works for CRC/hash/AEAD
//!
//! See `checksum` crate for usage examples.

use platform::{Caps, Tune};

use crate::{
  family::{KernelFamily, KernelSubfamily},
  tier::KernelTier,
};

// ─────────────────────────────────────────────────────────────────────────────
// SelectionPolicy
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-computed kernel selection policy.
///
/// Policies are computed once from `Caps` + `Tune` and cached for the process
/// lifetime. They encode:
/// - Which kernel family and subfamily to use
/// - Thresholds for tier transitions (e.g., portable → folding)
/// - Per-lane stream count selection based on `min_bytes_per_lane`
///
/// # Per-Lane Stream Selection
///
/// Unlike fixed total-length thresholds, stream selection uses:
/// `streams = max(s) where len / s >= min_bytes_per_lane`
///
/// This ensures each parallel lane has sufficient data to amortize setup.
///
/// # Cache-Friendliness
///
/// The struct fits in one cache line. All fields are accessed together
/// during dispatch, maximizing spatial locality.
///
/// # no_std Support
///
/// Policies can be constructed at compile time for embedded/no_std targets
/// where runtime detection isn't available.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectionPolicy {
  /// Selected kernel family for this policy.
  family: KernelFamily,

  /// Subfamily with algorithm-specific variation details.
  ///
  /// This captures whether the kernel uses hardware CRC, wide operations,
  /// multi-stream processing, etc. See [`KernelSubfamily`] for details.
  subfamily: KernelSubfamily,

  /// Cached capabilities (for diagnostics / introspection).
  caps: Caps,

  /// Minimum bytes where SIMD becomes faster than portable.
  ///
  /// Below this threshold, always use the portable kernel.
  pub small_threshold: usize,

  /// Minimum bytes where folding tier becomes optimal.
  ///
  /// At or above this threshold, use PCLMUL/PMULL kernels.
  pub fold_threshold: usize,

  /// Minimum bytes where wide tier becomes optimal.
  ///
  /// At or above this threshold, use VPCLMUL/EOR3/SVE2 kernels.
  /// Set to `usize::MAX` to disable wide tier.
  pub wide_threshold: usize,

  /// Maximum parallel streams for multi-way folding.
  ///
  /// Capped by architecture limits:
  /// - x86_64: 8-way
  /// - aarch64: 3-way
  /// - Power: 8-way
  /// - s390x: 4-way
  /// - riscv64: 4-way
  pub max_streams: u8,

  /// Minimum bytes per lane for multi-stream to be worthwhile.
  ///
  /// Stream selection uses: `streams = max(s) where len / s >= min_bytes_per_lane`.
  /// This ensures each parallel lane has enough data to amortize setup costs.
  ///
  /// Initialized from `KernelFamily::min_bytes_per_lane()` but can be overridden
  /// by algorithm-specific tunables.
  pub min_bytes_per_lane: usize,
}

impl SelectionPolicy {
  // ─────────────────────────────────────────────────────────────────────────
  // Construction
  // ─────────────────────────────────────────────────────────────────────────

  /// Create a portable-only policy (no SIMD).
  ///
  /// This is the compile-time fallback for no_std targets without
  /// runtime capability detection.
  #[inline]
  #[must_use]
  pub const fn portable() -> Self {
    Self {
      family: KernelFamily::Portable,
      subfamily: KernelSubfamily::portable(),
      caps: Caps::NONE,
      small_threshold: usize::MAX, // Never transition to SIMD
      fold_threshold: usize::MAX,
      wide_threshold: usize::MAX,
      max_streams: 1,
      min_bytes_per_lane: usize::MAX,
    }
  }

  /// Create a reference-only policy (bitwise implementation).
  ///
  /// Used when absolute correctness verification is required.
  #[inline]
  #[must_use]
  pub const fn reference() -> Self {
    Self {
      family: KernelFamily::Reference,
      subfamily: KernelSubfamily::reference(),
      caps: Caps::NONE,
      small_threshold: usize::MAX,
      fold_threshold: usize::MAX,
      wide_threshold: usize::MAX,
      max_streams: 1,
      min_bytes_per_lane: usize::MAX,
    }
  }

  /// Construct a policy from detected capabilities and tuning hints.
  ///
  /// This performs the one-time analysis to select the optimal family
  /// and compute all thresholds.
  #[must_use]
  pub fn from_platform(caps: Caps, tune: &Tune) -> Self {
    // Select the best available family
    let family = Self::select_family(caps, tune);

    // Compute subfamily based on family and tier
    let subfamily = Self::compute_subfamily(family);

    // Compute thresholds based on tune preset
    let (small_threshold, fold_threshold, wide_threshold) = Self::compute_thresholds(family, tune);

    // Compute stream parameters for multi-way folding
    let max_streams = Self::compute_max_streams(family, tune);
    let min_bytes_per_lane = family.min_bytes_per_lane();

    Self {
      family,
      subfamily,
      caps,
      small_threshold,
      fold_threshold,
      wide_threshold,
      max_streams,
      min_bytes_per_lane,
    }
  }

  /// Construct a policy with a specific family (for forcing).
  ///
  /// The family is validated against capabilities - if unavailable,
  /// falls back to the best available alternative.
  #[must_use]
  pub fn with_family(caps: Caps, tune: &Tune, family: KernelFamily) -> Self {
    let effective_family = if family.is_available(caps) {
      family
    } else {
      Self::select_family(caps, tune)
    };

    let subfamily = Self::compute_subfamily(effective_family);
    let (small_threshold, fold_threshold, wide_threshold) = Self::compute_thresholds(effective_family, tune);
    let max_streams = Self::compute_max_streams(effective_family, tune);
    let min_bytes_per_lane = effective_family.min_bytes_per_lane();

    Self {
      family: effective_family,
      subfamily,
      caps,
      small_threshold,
      fold_threshold,
      wide_threshold,
      max_streams,
      min_bytes_per_lane,
    }
  }

  /// Construct a policy with a specific subfamily (for algorithm-specific policies).
  ///
  /// This allows algorithm-specific code to set exact subfamily configuration.
  #[must_use]
  pub fn with_subfamily(caps: Caps, tune: &Tune, subfamily: KernelSubfamily) -> Self {
    let family = subfamily.family;
    let (small_threshold, fold_threshold, wide_threshold) = Self::compute_thresholds(family, tune);
    let max_streams = Self::compute_max_streams(family, tune);
    let min_bytes_per_lane = family.min_bytes_per_lane();

    Self {
      family,
      subfamily,
      caps,
      small_threshold,
      fold_threshold,
      wide_threshold,
      max_streams,
      min_bytes_per_lane,
    }
  }

  /// Compute the default subfamily for a given family.
  ///
  /// Algorithm-specific policies may override this with more specific subfamilies.
  fn compute_subfamily(family: KernelFamily) -> KernelSubfamily {
    match family.tier() {
      KernelTier::Reference => KernelSubfamily::reference(),
      KernelTier::Portable => KernelSubfamily::portable(),
      KernelTier::HwCrc => KernelSubfamily::hwcrc(family),
      KernelTier::Folding => KernelSubfamily::folding(family),
      KernelTier::Wide => KernelSubfamily::wide(family, false), // Algorithm can override uses_hwcrc
    }
  }

  /// Select the best kernel family for the given capabilities and tune.
  fn select_family(caps: Caps, tune: &Tune) -> KernelFamily {
    // Try families in descending tier order
    for &family in KernelFamily::families_for_current_arch() {
      // Skip reference - we want the best *usable* family
      if family == KernelFamily::Reference {
        continue;
      }

      // Check if family is available
      if !family.is_available(caps) {
        continue;
      }

      // For wide tier, check if wide ops are worth it
      if family.tier() == KernelTier::Wide && tune.effective_simd_width < 256 {
        continue;
      }

      return family;
    }

    // Portable is always available
    KernelFamily::Portable
  }

  /// Compute tier transition thresholds.
  fn compute_thresholds(family: KernelFamily, tune: &Tune) -> (usize, usize, usize) {
    let tier = family.tier();

    match tier {
      KernelTier::Reference | KernelTier::Portable => (usize::MAX, usize::MAX, usize::MAX),
      KernelTier::HwCrc => {
        // HW CRC is effective even for tiny buffers
        (tune.hwcrc_threshold, usize::MAX, usize::MAX)
      }
      KernelTier::Folding => {
        // Folding needs setup, use pclmul_threshold
        (tune.pclmul_threshold, tune.pclmul_threshold, usize::MAX)
      }
      KernelTier::Wide => {
        // Wide tier: different threshold for folding->wide transition
        let fold_to_wide = if tune.fast_wide_ops { 512 } else { 2048 };
        (tune.pclmul_threshold, tune.pclmul_threshold, fold_to_wide)
      }
    }
  }

  /// Compute maximum stream count for multi-way folding.
  ///
  /// Returns the architecture-specific limit capped by tune's parallel_streams.
  fn compute_max_streams(family: KernelFamily, tune: &Tune) -> u8 {
    let tier = family.tier();

    if !matches!(tier, KernelTier::HwCrc | KernelTier::Folding | KernelTier::Wide) {
      return 1;
    }

    // Architecture-specific stream limits
    let arch_max: u8 = match family {
      KernelFamily::ArmPmull | KernelFamily::ArmPmullEor3 | KernelFamily::ArmSve2Pmull | KernelFamily::ArmCrc32 => 3,
      KernelFamily::S390xVgfm => 4,
      KernelFamily::RiscvZbc | KernelFamily::RiscvZvbc => 4,
      _ => 8, // x86, powerpc
    };

    tune.parallel_streams.min(arch_max)
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Accessors
  // ─────────────────────────────────────────────────────────────────────────

  /// Get the selected kernel family.
  #[inline]
  #[must_use]
  pub const fn family(&self) -> KernelFamily {
    self.family
  }

  /// Get the selected kernel subfamily.
  ///
  /// The subfamily provides algorithm-specific details like whether
  /// hardware CRC is used, wide operations are enabled, etc.
  #[inline]
  #[must_use]
  pub const fn subfamily(&self) -> KernelSubfamily {
    self.subfamily
  }

  /// Get the tier of the selected family.
  #[inline]
  #[must_use]
  pub const fn tier(&self) -> KernelTier {
    self.family.tier()
  }

  /// Get the cached capabilities.
  #[inline]
  #[must_use]
  pub const fn caps(&self) -> Caps {
    self.caps
  }

  /// Check if this policy uses hardware CRC instructions.
  #[inline]
  #[must_use]
  pub const fn uses_hwcrc(&self) -> bool {
    self.subfamily.uses_hwcrc
  }

  /// Check if this policy uses wide (256/512-bit) operations.
  #[inline]
  #[must_use]
  pub const fn uses_wide(&self) -> bool {
    self.subfamily.uses_wide
  }

  /// Check if this policy supports multi-stream processing.
  #[inline]
  #[must_use]
  pub const fn supports_multi_stream(&self) -> bool {
    self.subfamily.multi_stream
  }

  /// Human-readable policy description.
  #[inline]
  #[must_use]
  pub const fn name(&self) -> &'static str {
    self.family.as_str()
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Stream Selection
  // ─────────────────────────────────────────────────────────────────────────

  /// Select stream count for the given buffer length.
  ///
  /// Uses per-lane logic: `streams = max(s) where len / s >= min_bytes_per_lane`.
  /// This ensures each parallel lane has enough data to amortize setup costs.
  #[inline]
  #[must_use]
  pub fn streams_for_len(&self, len: usize) -> u8 {
    self.streams_for_len_with_min(len, self.min_bytes_per_lane)
  }

  /// Select stream count with a custom `min_bytes_per_lane`.
  ///
  /// This is the core per-lane stream selection logic. Algorithm-specific
  /// policies can use this to override the default `min_bytes_per_lane`.
  ///
  /// # Algorithm
  ///
  /// Find the maximum stream count `s` from a family-specific list such that:
  /// - `s <= max_streams` (architecture limit)
  /// - `len / s >= min_bytes_per_lane` (each lane has enough data)
  ///
  /// Falls back to 1-way if no stream count satisfies the requirement.
  #[inline]
  #[must_use]
  pub fn streams_for_len_with_min(&self, len: usize, min_bytes_per_lane: usize) -> u8 {
    // Early exit for single-stream policies
    if self.max_streams == 1 {
      return 1;
    }

    // If buffer is smaller than min_bytes_per_lane, always 1-way
    if len < min_bytes_per_lane {
      return 1;
    }

    // Stream counts supported by the selected backend family.
    //
    // This avoids selecting unsupported levels (e.g. 3-way on x86_64, 7-way on Power),
    // while still letting `max_streams` cap per-arch limits and tunables.
    const STREAM_LEVELS_X86: [u8; 5] = [8, 7, 4, 2, 1];
    const STREAM_LEVELS_ARM: [u8; 3] = [3, 2, 1];
    const STREAM_LEVELS_POWER: [u8; 4] = [8, 4, 2, 1];
    const STREAM_LEVELS_4WAY: [u8; 3] = [4, 2, 1];
    const STREAM_LEVELS_NONE: [u8; 1] = [1];

    let stream_levels: &[u8] = match self.family {
      KernelFamily::X86Crc32 | KernelFamily::X86Pclmul | KernelFamily::X86Vpclmul => &STREAM_LEVELS_X86,
      KernelFamily::ArmCrc32 | KernelFamily::ArmPmull | KernelFamily::ArmPmullEor3 | KernelFamily::ArmSve2Pmull => {
        &STREAM_LEVELS_ARM
      }
      KernelFamily::PowerVpmsum => &STREAM_LEVELS_POWER,
      KernelFamily::S390xVgfm | KernelFamily::RiscvZbc | KernelFamily::RiscvZvbc => &STREAM_LEVELS_4WAY,
      _ => &STREAM_LEVELS_NONE,
    };

    for &s in stream_levels {
      // Skip if above architecture limit
      if s > self.max_streams {
        continue;
      }
      // Check if each lane gets at least min_bytes_per_lane
      if len.strict_div(s as usize) >= min_bytes_per_lane {
        return s;
      }
    }

    1
  }

  /// Check if wide tier should be used for this length.
  #[inline]
  #[must_use]
  pub fn should_use_wide(&self, len: usize) -> bool {
    self.tier() == KernelTier::Wide && len >= self.wide_threshold
  }

  /// Check if SIMD should be used for this length.
  #[inline]
  #[must_use]
  pub fn should_use_simd(&self, len: usize) -> bool {
    len >= self.small_threshold
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Force Mode Application
  // ─────────────────────────────────────────────────────────────────────────

  /// Apply a force mode, returning a new policy.
  ///
  /// The force is clamped to available capabilities - forcing an unavailable
  /// family results in the best available alternative.
  #[must_use]
  pub fn with_force(&self, force: ForceMode, tune: &Tune) -> Self {
    match force {
      ForceMode::Auto => *self,
      ForceMode::Reference => Self::reference(),
      ForceMode::Portable => Self::portable(),
      ForceMode::Tier(tier) => self.force_to_tier(tier, tune),
      ForceMode::Family(family) => Self::with_family(self.caps, tune, family),
    }
  }

  fn force_to_tier(&self, tier: KernelTier, tune: &Tune) -> Self {
    // Find best available family in the requested tier
    let families = KernelFamily::families_in_tier(tier);

    for &family in families {
      if family.is_available(self.caps) {
        return Self::with_family(self.caps, tune, family);
      }
    }

    // Fall back to portable if tier unavailable
    Self::portable()
  }

  /// Cap the maximum number of parallel streams used by this policy.
  ///
  /// This allows algorithm-specific tuning to set the maximum stream count used
  /// by dispatch (while still respecting architecture limits and available tiers).
  #[inline]
  pub fn cap_max_streams(&mut self, cap: u8) {
    // Reference/Portable families cannot multi-stream.
    if matches!(self.family, KernelFamily::Reference | KernelFamily::Portable) {
      self.max_streams = 1;
      return;
    }

    let cap = cap.max(1);

    // Architecture-specific stream limits.
    let arch_max: u8 = match self.family {
      KernelFamily::ArmPmull | KernelFamily::ArmPmullEor3 | KernelFamily::ArmSve2Pmull | KernelFamily::ArmCrc32 => 3,
      KernelFamily::S390xVgfm => 4,
      KernelFamily::RiscvZbc | KernelFamily::RiscvZvbc => 4,
      _ => 8, // x86, power
    };

    self.max_streams = cap.min(arch_max);
  }
}

impl Default for SelectionPolicy {
  /// Default policy uses portable fallback.
  #[inline]
  fn default() -> Self {
    Self::portable()
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// ForceMode
// ─────────────────────────────────────────────────────────────────────────────

/// Force mode for overriding automatic kernel selection.
///
/// This replaces algorithm-specific force enums (like `Crc64Force`) with
/// a unified system that works across all algorithms.
///
/// # Safety
///
/// Force modes that specify unavailable families or tiers are safe:
/// the policy will fall back to the best available alternative.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ForceMode {
  /// Automatic selection based on capabilities and tuning (default).
  #[default]
  Auto,

  /// Force reference (bitwise) implementation.
  ///
  /// Used for verification and audit-critical paths.
  Reference,

  /// Force portable (table-based) implementation.
  ///
  /// Used for testing or when SIMD causes issues.
  Portable,

  /// Force to a specific tier.
  ///
  /// The best available family within the tier is selected.
  Tier(KernelTier),

  /// Force to a specific family.
  ///
  /// Falls back to automatic selection if the family is unavailable.
  Family(KernelFamily),
}

impl ForceMode {
  /// Parse from string (for env var support).
  ///
  /// Accepts case-insensitive names:
  /// - `"auto"` → `Auto`
  /// - `"reference"`, `"bitwise"` → `Reference`
  /// - `"portable"`, `"table"` → `Portable`
  /// - `"pclmul"`, `"vpclmul"`, etc. → `Family(...)`
  #[must_use]
  pub fn parse(s: &str) -> Option<Self> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("auto") {
      return Some(Self::Auto);
    }
    if s.eq_ignore_ascii_case("reference") || s.eq_ignore_ascii_case("bitwise") {
      return Some(Self::Reference);
    }
    if s.eq_ignore_ascii_case("portable") || s.eq_ignore_ascii_case("table") {
      return Some(Self::Portable);
    }

    // Family names
    if s.eq_ignore_ascii_case("pclmul") {
      return Some(Self::Family(KernelFamily::X86Pclmul));
    }
    if s.eq_ignore_ascii_case("vpclmul") {
      return Some(Self::Family(KernelFamily::X86Vpclmul));
    }
    if s.eq_ignore_ascii_case("pmull") {
      return Some(Self::Family(KernelFamily::ArmPmull));
    }
    if s.eq_ignore_ascii_case("pmull-eor3") || s.eq_ignore_ascii_case("pmulleor3") {
      return Some(Self::Family(KernelFamily::ArmPmullEor3));
    }
    if s.eq_ignore_ascii_case("sve2-pmull") || s.eq_ignore_ascii_case("sve2pmull") {
      return Some(Self::Family(KernelFamily::ArmSve2Pmull));
    }
    if s.eq_ignore_ascii_case("vpmsum") {
      return Some(Self::Family(KernelFamily::PowerVpmsum));
    }
    if s.eq_ignore_ascii_case("vgfm") {
      return Some(Self::Family(KernelFamily::S390xVgfm));
    }
    if s.eq_ignore_ascii_case("zbc") {
      return Some(Self::Family(KernelFamily::RiscvZbc));
    }
    if s.eq_ignore_ascii_case("zvbc") {
      return Some(Self::Family(KernelFamily::RiscvZvbc));
    }

    // Tier names
    if s.eq_ignore_ascii_case("hwcrc") || s.eq_ignore_ascii_case("hw-crc") {
      return Some(Self::Tier(KernelTier::HwCrc));
    }
    if s.eq_ignore_ascii_case("folding") {
      return Some(Self::Tier(KernelTier::Folding));
    }
    if s.eq_ignore_ascii_case("wide") {
      return Some(Self::Tier(KernelTier::Wide));
    }

    None
  }

  /// Check if this force mode requires SIMD.
  #[inline]
  #[must_use]
  pub const fn requires_simd(self) -> bool {
    match self {
      Self::Auto => false, // May or may not use SIMD
      Self::Reference | Self::Portable => false,
      Self::Tier(tier) => tier.is_simd(),
      Self::Family(family) => family.tier().is_simd(),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn portable_policy() {
    let policy = SelectionPolicy::portable();
    assert_eq!(policy.family(), KernelFamily::Portable);
    assert_eq!(policy.tier(), KernelTier::Portable);
    assert!(!policy.should_use_simd(1000));
    assert_eq!(policy.streams_for_len(10000), 1);
  }

  #[test]
  fn reference_policy() {
    let policy = SelectionPolicy::reference();
    assert_eq!(policy.family(), KernelFamily::Reference);
    assert_eq!(policy.tier(), KernelTier::Reference);
  }

  #[test]
  fn streams_for_len_per_lane() {
    // Test per-lane stream selection with min_bytes_per_lane = 256
    let policy = SelectionPolicy {
      family: KernelFamily::X86Pclmul,
      subfamily: KernelSubfamily::folding(KernelFamily::X86Pclmul),
      caps: Caps::NONE,
      small_threshold: 64,
      fold_threshold: 64,
      wide_threshold: usize::MAX,
      max_streams: 8,
      min_bytes_per_lane: 256,
    };

    // len=256: 256/8=32 < 256, 256/7=36 < 256, ..., 256/1=256 >= 256 → 1-way
    assert_eq!(policy.streams_for_len(256), 1);

    // len=512: 512/2=256 >= 256 → 2-way
    assert_eq!(policy.streams_for_len(512), 2);

    // len=1024: 1024/4=256 >= 256 → 4-way
    assert_eq!(policy.streams_for_len(1024), 4);

    // len=1792: 1792/7=256 >= 256 → 7-way
    assert_eq!(policy.streams_for_len(1792), 7);

    // len=2048: 2048/8=256 >= 256 → 8-way
    assert_eq!(policy.streams_for_len(2048), 8);

    // len=4096: 4096/8=512 >= 256 → 8-way
    assert_eq!(policy.streams_for_len(4096), 8);
  }

  #[test]
  fn streams_capped_by_max() {
    let policy = SelectionPolicy {
      family: KernelFamily::ArmPmull,
      subfamily: KernelSubfamily::folding(KernelFamily::ArmPmull),
      caps: Caps::NONE,
      small_threshold: 64,
      fold_threshold: 64,
      wide_threshold: usize::MAX,
      max_streams: 3, // aarch64 limit
      min_bytes_per_lane: 256,
    };

    // Should cap at 3 even with large buffer
    // len=10000: could be 8-way but max_streams=3, so 3-way
    assert_eq!(policy.streams_for_len(10000), 3);
  }

  #[test]
  fn streams_for_len_with_custom_min() {
    let policy = SelectionPolicy {
      family: KernelFamily::X86Vpclmul,
      subfamily: KernelSubfamily::wide(KernelFamily::X86Vpclmul, false),
      caps: Caps::NONE,
      small_threshold: 64,
      fold_threshold: 64,
      wide_threshold: 512,
      max_streams: 8,
      min_bytes_per_lane: 512, // Wide tier default
    };

    // Use custom min_bytes_per_lane of 128 (algorithm override)
    // len=1024: 1024/8=128 >= 128 → 8-way with custom min
    assert_eq!(policy.streams_for_len_with_min(1024, 128), 8);

    // But with default min_bytes_per_lane=512:
    // len=1024: 1024/2=512 >= 512 → 2-way
    assert_eq!(policy.streams_for_len(1024), 2);
  }

  #[test]
  fn streams_below_min_returns_one() {
    let policy = SelectionPolicy {
      family: KernelFamily::X86Pclmul,
      subfamily: KernelSubfamily::folding(KernelFamily::X86Pclmul),
      caps: Caps::NONE,
      small_threshold: 64,
      fold_threshold: 64,
      wide_threshold: usize::MAX,
      max_streams: 8,
      min_bytes_per_lane: 256,
    };

    // Buffer smaller than min_bytes_per_lane always returns 1
    assert_eq!(policy.streams_for_len(100), 1);
    assert_eq!(policy.streams_for_len(255), 1);
  }

  #[test]
  fn force_mode_parsing() {
    assert_eq!(ForceMode::parse("auto"), Some(ForceMode::Auto));
    assert_eq!(ForceMode::parse("AUTO"), Some(ForceMode::Auto));
    assert_eq!(ForceMode::parse("reference"), Some(ForceMode::Reference));
    assert_eq!(ForceMode::parse("bitwise"), Some(ForceMode::Reference));
    assert_eq!(ForceMode::parse("portable"), Some(ForceMode::Portable));
    assert_eq!(ForceMode::parse("table"), Some(ForceMode::Portable));
    assert_eq!(
      ForceMode::parse("pclmul"),
      Some(ForceMode::Family(KernelFamily::X86Pclmul))
    );
    assert_eq!(
      ForceMode::parse("VPCLMUL"),
      Some(ForceMode::Family(KernelFamily::X86Vpclmul))
    );
    assert_eq!(ForceMode::parse("invalid"), None);
  }

  #[test]
  fn default_is_portable() {
    assert_eq!(SelectionPolicy::default().family(), KernelFamily::Portable);
    assert_eq!(ForceMode::default(), ForceMode::Auto);
  }
}
