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
//! # Usage
//!
//! ```ignore
//! // Compute policy once at initialization
//! let policy = SelectionPolicy::from_platform(caps, &tune);
//!
//! // Per-call dispatch (fast path)
//! if policy.should_use_simd(len) {
//!     let streams = policy.streams_for_len(len);
//!     simd_kernel(streams, data)
//! } else {
//!     portable_kernel(data)
//! }
//! ```

use platform::{Caps, Tune};

use crate::{family::KernelFamily, tier::KernelTier};

// ─────────────────────────────────────────────────────────────────────────────
// SelectionPolicy
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-computed kernel selection policy.
///
/// Policies are computed once from `Caps` + `Tune` and cached for the process
/// lifetime. They encode:
/// - Which kernel family to use
/// - Thresholds for tier transitions (e.g., portable → folding)
/// - Stream count selection for different buffer sizes
///
/// # Cache-Friendliness
///
/// The struct is 56 bytes, fitting in one cache line. All fields are
/// accessed together during dispatch, maximizing spatial locality.
///
/// # no_std Support
///
/// Policies can be constructed at compile time for embedded/no_std targets
/// where runtime detection isn't available.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectionPolicy {
  /// Selected kernel family for this policy.
  family: KernelFamily,

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
  /// - powerpc64: 8-way
  /// - s390x: 4-way
  /// - riscv64: 4-way
  pub max_streams: u8,

  /// Pre-computed stream thresholds: [8-way, 7-way, 4-way, 2-way].
  ///
  /// Bytes required for each stream level. Used by `streams_for_len()`.
  /// Higher stream counts require larger buffers to amortize setup cost.
  stream_thresholds: [usize; 4],
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
      caps: Caps::NONE,
      small_threshold: usize::MAX, // Never transition to SIMD
      fold_threshold: usize::MAX,
      wide_threshold: usize::MAX,
      max_streams: 1,
      stream_thresholds: [usize::MAX; 4],
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
      caps: Caps::NONE,
      small_threshold: usize::MAX,
      fold_threshold: usize::MAX,
      wide_threshold: usize::MAX,
      max_streams: 1,
      stream_thresholds: [usize::MAX; 4],
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

    // Compute thresholds based on tune preset
    let (small_threshold, fold_threshold, wide_threshold) = Self::compute_thresholds(family, tune);

    // Compute stream thresholds for multi-way folding
    let (max_streams, stream_thresholds) = Self::compute_stream_params(family, tune);

    Self {
      family,
      caps,
      small_threshold,
      fold_threshold,
      wide_threshold,
      max_streams,
      stream_thresholds,
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

    let (small_threshold, fold_threshold, wide_threshold) = Self::compute_thresholds(effective_family, tune);
    let (max_streams, stream_thresholds) = Self::compute_stream_params(effective_family, tune);

    Self {
      family: effective_family,
      caps,
      small_threshold,
      fold_threshold,
      wide_threshold,
      max_streams,
      stream_thresholds,
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

  /// Compute stream parameters for multi-way folding.
  fn compute_stream_params(family: KernelFamily, tune: &Tune) -> (u8, [usize; 4]) {
    let tier = family.tier();

    if !matches!(tier, KernelTier::Folding | KernelTier::Wide) {
      return (1, [usize::MAX; 4]);
    }

    // Base fold block size (128 bytes for most CRC folding)
    const FOLD_BLOCK: usize = 128;

    // Architecture-specific stream limits
    let arch_max: u8 = match family {
      KernelFamily::ArmPmull | KernelFamily::ArmPmullEor3 | KernelFamily::ArmSve2Pmull => 3,
      KernelFamily::S390xVgfm => 4,
      KernelFamily::RiscvZbc | KernelFamily::RiscvZvbc => 4,
      _ => 8, // x86, powerpc
    };

    let max_streams = tune.parallel_streams.min(arch_max);

    // Compute thresholds: 2× fold block per stream level
    // These are minimum buffer sizes to use each stream count
    let thresholds = [
      8usize.strict_mul(2).strict_mul(FOLD_BLOCK), // 8-way: 2048 bytes
      7usize.strict_mul(2).strict_mul(FOLD_BLOCK), // 7-way: 1792 bytes
      4usize.strict_mul(2).strict_mul(FOLD_BLOCK), // 4-way: 1024 bytes
      2usize.strict_mul(2).strict_mul(FOLD_BLOCK), // 2-way: 512 bytes
    ];

    (max_streams, thresholds)
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
  /// This is the hot path optimization - uses arithmetic instead of branches
  /// to minimize branch misprediction overhead.
  ///
  /// # Algorithm
  ///
  /// Counts how many stream thresholds are satisfied and maps to stream count:
  /// - 4 thresholds met → 8-way
  /// - 3 thresholds met → 7-way
  /// - 2 thresholds met → 4-way
  /// - 1 threshold met → 2-way
  /// - 0 thresholds met → 1-way
  #[inline]
  #[must_use]
  pub fn streams_for_len(&self, len: usize) -> u8 {
    // Early exit for non-folding tiers
    if self.max_streams == 1 {
      return 1;
    }

    let t = &self.stream_thresholds;

    // Branchless: count how many thresholds are satisfied
    // Each comparison produces 0 or 1
    let m0 = (len >= t[0]) as usize; // 8-way eligible
    let m1 = (len >= t[1]) as usize; // 7-way eligible
    let m2 = (len >= t[2]) as usize; // 4-way eligible
    let m3 = (len >= t[3]) as usize; // 2-way eligible

    // Map count to stream value: [1, 2, 4, 7, 8]
    const STREAM_MAP: [u8; 5] = [1, 2, 4, 7, 8];

    // Index = m0 + m1 + m2 + m3 (0-4)
    let idx = m0.wrapping_add(m1).wrapping_add(m2).wrapping_add(m3);

    // SAFETY: idx is in 0..=4, STREAM_MAP has 5 elements
    // Use get().copied().unwrap_or(1) for bounds safety without panicking
    STREAM_MAP.get(idx).copied().unwrap_or(1).min(self.max_streams)
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
  fn streams_for_len_basic() {
    let policy = SelectionPolicy {
      family: KernelFamily::X86Pclmul,
      caps: Caps::NONE,
      small_threshold: 64,
      fold_threshold: 64,
      wide_threshold: usize::MAX,
      max_streams: 8,
      stream_thresholds: [2048, 1792, 1024, 512],
    };

    // Below 2-way threshold
    assert_eq!(policy.streams_for_len(256), 1);

    // 2-way (512+ bytes)
    assert_eq!(policy.streams_for_len(512), 2);
    assert_eq!(policy.streams_for_len(768), 2);

    // 4-way (1024+ bytes)
    assert_eq!(policy.streams_for_len(1024), 4);
    assert_eq!(policy.streams_for_len(1500), 4);

    // 7-way (1792+ bytes)
    assert_eq!(policy.streams_for_len(1792), 7);
    assert_eq!(policy.streams_for_len(2000), 7);

    // 8-way (2048+ bytes)
    assert_eq!(policy.streams_for_len(2048), 8);
    assert_eq!(policy.streams_for_len(10000), 8);
  }

  #[test]
  fn streams_capped_by_max() {
    let policy = SelectionPolicy {
      family: KernelFamily::ArmPmull,
      caps: Caps::NONE,
      small_threshold: 64,
      fold_threshold: 64,
      wide_threshold: usize::MAX,
      max_streams: 3, // aarch64 limit
      stream_thresholds: [2048, 1792, 1024, 512],
    };

    // Should cap at 3 even with large buffer
    assert_eq!(policy.streams_for_len(10000), 3);
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
