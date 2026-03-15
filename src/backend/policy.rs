//! Pre-computed kernel selection policies.
//!
//! A [`SelectionPolicy`] captures all dispatch structure at initialization time,
//! eliminating per-call capability checks and leaving size crossover policy to
//! algorithm crates.
//!
//! # Design Goals
//!
//! 1. **Zero per-call overhead**: Policy is computed once and cached
//! 2. **Backend owns structure**: Family legality, subfamily shape, and stream topology live here
//! 3. **Algorithm owns tuning**: Thresholds and lane crossover policy are explicit inputs
//! 4. **no_std compatible**: All types are `Copy` and heap-free
//!
//! # Usage
//!
//! ```text
//! let policy = SelectionPolicy::from_platform(caps);
//! let policy = policy.with_tunables(PolicyTunables::conservative_for(policy.family()));
//! let streams = policy.streams_for_len(buffer.len());
//! ```

use crate::{
  backend::{
    family::{KernelFamily, KernelSubfamily},
    tier::KernelTier,
  },
  platform::Caps,
};

/// Algorithm-owned dispatch tuning layered on top of a structural policy.
///
/// The backend can provide conservative fallbacks, but algorithm crates should
/// prefer measured values derived from their own benchmarks and dispatch tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyTunables {
  /// Minimum bytes where SIMD becomes faster than portable.
  pub small_threshold: usize,
  /// Minimum bytes where folding becomes worthwhile.
  pub fold_threshold: usize,
  /// Minimum bytes where the wide tier becomes worthwhile.
  pub wide_threshold: usize,
  /// Minimum bytes per lane required before enabling multi-stream processing.
  pub min_bytes_per_lane: usize,
}

impl PolicyTunables {
  /// Disable all size-based tier transitions and multi-streaming.
  pub const DISABLED: Self = Self {
    small_threshold: usize::MAX,
    fold_threshold: usize::MAX,
    wide_threshold: usize::MAX,
    min_bytes_per_lane: usize::MAX,
  };

  /// Conservative backend fallback for callers that do not have measured data yet.
  #[inline]
  #[must_use]
  pub const fn conservative_for(family: KernelFamily) -> Self {
    match family {
      KernelFamily::Reference | KernelFamily::Portable => Self::DISABLED,
      KernelFamily::X86Crc32 | KernelFamily::ArmCrc32 => Self {
        small_threshold: 16,
        fold_threshold: usize::MAX,
        wide_threshold: usize::MAX,
        min_bytes_per_lane: 128,
      },
      KernelFamily::X86Pclmul
      | KernelFamily::ArmPmull
      | KernelFamily::PowerVpmsum
      | KernelFamily::S390xVgfm
      | KernelFamily::RiscvZbc => Self {
        small_threshold: 64,
        fold_threshold: 64,
        wide_threshold: usize::MAX,
        min_bytes_per_lane: 256,
      },
      KernelFamily::X86Vpclmul => Self {
        small_threshold: 64,
        fold_threshold: 64,
        wide_threshold: 512,
        min_bytes_per_lane: 512,
      },
      KernelFamily::ArmPmullEor3 => Self {
        small_threshold: 64,
        fold_threshold: 64,
        wide_threshold: 512,
        min_bytes_per_lane: 256,
      },
      KernelFamily::ArmSve2Pmull | KernelFamily::RiscvZvbc => Self {
        small_threshold: 64,
        fold_threshold: 64,
        wide_threshold: 512,
        min_bytes_per_lane: 512,
      },
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SelectionPolicy
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-computed kernel selection policy.
///
/// Policies are computed once from `Caps` and cached for the process
/// lifetime. They encode:
///
/// - Which kernel family and subfamily to use
/// - Architecture stream topology and legality limits
/// - Optional algorithm-owned thresholds for tier transitions and multi-streaming
///
/// # Stream Selection
///
/// Stream count is selected per-lane: `streams = max(s) where len / s >= min_bytes_per_lane`.
/// This ensures each parallel lane has enough data to amortize setup costs.
///
/// # no_std Support
///
/// Policies can be constructed at compile time for embedded targets where
/// runtime detection is unavailable.
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
  /// This is algorithm-owned tuning. Structural constructors leave it disabled
  /// until the caller applies measured values or an explicit conservative fallback.
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
      small_threshold: PolicyTunables::DISABLED.small_threshold,
      fold_threshold: PolicyTunables::DISABLED.fold_threshold,
      wide_threshold: PolicyTunables::DISABLED.wide_threshold,
      max_streams: 1,
      min_bytes_per_lane: PolicyTunables::DISABLED.min_bytes_per_lane,
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
      small_threshold: PolicyTunables::DISABLED.small_threshold,
      fold_threshold: PolicyTunables::DISABLED.fold_threshold,
      wide_threshold: PolicyTunables::DISABLED.wide_threshold,
      max_streams: 1,
      min_bytes_per_lane: PolicyTunables::DISABLED.min_bytes_per_lane,
    }
  }

  /// Construct a policy from detected capabilities.
  ///
  /// This performs the one-time analysis to select the best legal family and
  /// subfamily. Algorithm-owned thresholds remain disabled until callers apply
  /// explicit tunables.
  #[must_use]
  pub fn from_platform(caps: Caps) -> Self {
    let family = KernelFamily::select_for_platform(caps);
    let subfamily = Self::compute_subfamily(family);
    Self::from_parts(caps, family, subfamily, PolicyTunables::DISABLED)
  }

  /// Construct a policy with a specific family (for forcing).
  ///
  /// The family is validated against capabilities - if unavailable,
  /// falls back to the best available alternative.
  #[must_use]
  pub fn with_family(caps: Caps, family: KernelFamily) -> Self {
    let effective_family = if family.is_available(caps) {
      family
    } else {
      KernelFamily::select_for_platform(caps)
    };
    let subfamily = Self::compute_subfamily(effective_family);
    Self::from_parts(caps, effective_family, subfamily, PolicyTunables::DISABLED)
  }

  /// Construct a policy with a specific subfamily (for algorithm-specific policies).
  ///
  /// This allows algorithm-specific code to set exact subfamily configuration.
  #[must_use]
  pub fn with_subfamily(caps: Caps, subfamily: KernelSubfamily) -> Self {
    let family = subfamily.family;
    Self::from_parts(caps, family, subfamily, PolicyTunables::DISABLED)
  }

  fn from_parts(caps: Caps, family: KernelFamily, subfamily: KernelSubfamily, tunables: PolicyTunables) -> Self {
    let mut policy = Self {
      family,
      subfamily,
      caps,
      small_threshold: PolicyTunables::DISABLED.small_threshold,
      fold_threshold: PolicyTunables::DISABLED.fold_threshold,
      wide_threshold: PolicyTunables::DISABLED.wide_threshold,
      max_streams: family.arch_max_streams(),
      min_bytes_per_lane: PolicyTunables::DISABLED.min_bytes_per_lane,
    };
    policy.apply_tunables(tunables);
    policy
  }

  fn apply_tunables(&mut self, tunables: PolicyTunables) {
    if matches!(self.family, KernelFamily::Reference | KernelFamily::Portable) {
      self.small_threshold = PolicyTunables::DISABLED.small_threshold;
      self.fold_threshold = PolicyTunables::DISABLED.fold_threshold;
      self.wide_threshold = PolicyTunables::DISABLED.wide_threshold;
      self.max_streams = 1;
      self.min_bytes_per_lane = PolicyTunables::DISABLED.min_bytes_per_lane;
      return;
    }

    self.small_threshold = tunables.small_threshold;
    self.fold_threshold = tunables.fold_threshold;
    self.wide_threshold = tunables.wide_threshold;
    self.max_streams = self.family.arch_max_streams();
    self.min_bytes_per_lane = tunables.min_bytes_per_lane;
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

  /// Get the algorithm-owned tuning currently layered onto this policy.
  #[inline]
  #[must_use]
  pub const fn tunables(&self) -> PolicyTunables {
    PolicyTunables {
      small_threshold: self.small_threshold,
      fold_threshold: self.fold_threshold,
      wide_threshold: self.wide_threshold,
      min_bytes_per_lane: self.min_bytes_per_lane,
    }
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
  /// Finds the maximum stream count `s` such that `len / s >= min_bytes_per_lane`
  /// and `s <= max_streams`. Falls back to 1-way if no count satisfies these.
  #[inline]
  #[must_use]
  pub fn streams_for_len_with_min(&self, len: usize, min_bytes_per_lane: usize) -> u8 {
    if self.max_streams == 1 || len < min_bytes_per_lane {
      return 1;
    }

    for &s in self.family.stream_levels() {
      if s <= self.max_streams && len.strict_div(s as usize) >= min_bytes_per_lane {
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

  /// Apply explicit algorithm-owned tuning to this policy.
  ///
  /// Structural fields (family, subfamily, architecture stream limits) stay in
  /// `backend`; size crossover policy remains the caller's responsibility.
  #[inline]
  #[must_use]
  pub fn with_tunables(mut self, tunables: PolicyTunables) -> Self {
    self.apply_tunables(tunables);
    self
  }

  /// Apply backend conservative fallbacks explicitly.
  ///
  /// This exists for algorithms that have not yet landed measured tables.
  #[inline]
  #[must_use]
  pub fn with_conservative_tunables(self) -> Self {
    self.with_tunables(PolicyTunables::conservative_for(self.family))
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Force Mode Application
  // ─────────────────────────────────────────────────────────────────────────

  /// Apply a force mode, returning a new policy.
  ///
  /// The force is clamped to available capabilities - forcing an unavailable
  /// family results in the best available alternative.
  #[must_use]
  pub fn with_force(&self, force: ForceMode) -> Self {
    match force {
      ForceMode::Auto => *self,
      ForceMode::Reference => Self::reference(),
      ForceMode::Portable => Self::portable(),
      ForceMode::Tier(tier) => self.force_to_tier(tier),
      ForceMode::Family(family) => self.retarget(family),
    }
  }

  fn force_to_tier(&self, tier: KernelTier) -> Self {
    if let Some(family) = KernelFamily::best_available_in_tier(tier, self.caps) {
      return self.retarget(family);
    }

    // Fall back to portable if tier unavailable
    Self::portable()
  }

  fn retarget(&self, family: KernelFamily) -> Self {
    let effective_family = if family.is_available(self.caps) {
      family
    } else {
      KernelFamily::select_for_platform(self.caps)
    };
    let subfamily = Self::compute_subfamily(effective_family);
    Self::from_parts(self.caps, effective_family, subfamily, self.tunables())
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
    let arch_max = self.family.arch_max_streams();

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
/// A unified system for forcing specific backends across all algorithms.
/// Force modes that specify unavailable families or tiers are safe: the
/// policy falls back to the best available alternative.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ForceMode {
  /// Automatic selection based on capabilities (default).
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
  /// - `"auto"` - Automatic selection
  /// - `"reference"`, `"bitwise"` - Reference implementation
  /// - `"portable"`, `"table"` - Portable fallback
  /// - `"pclmul"`, `"vpclmul"`, `"pmull"`, etc. - Specific families
  /// - `"hwcrc"`, `"folding"`, `"wide"` - Specific tiers
  #[must_use]
  pub fn parse(s: &str) -> Option<Self> {
    let s = s.trim();

    // Mode names (case-insensitive)
    if s.eq_ignore_ascii_case("auto") {
      return Some(Self::Auto);
    }
    if s.eq_ignore_ascii_case("reference") || s.eq_ignore_ascii_case("bitwise") {
      return Some(Self::Reference);
    }
    if s.eq_ignore_ascii_case("portable") || s.eq_ignore_ascii_case("table") {
      return Some(Self::Portable);
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

    // Family names
    Self::parse_family(s).map(Self::Family)
  }

  /// Parse a family name (helper for `parse`).
  fn parse_family(s: &str) -> Option<KernelFamily> {
    // x86_64 families
    if s.eq_ignore_ascii_case("pclmul") {
      return Some(KernelFamily::X86Pclmul);
    }
    if s.eq_ignore_ascii_case("vpclmul") {
      return Some(KernelFamily::X86Vpclmul);
    }

    // aarch64 families
    if s.eq_ignore_ascii_case("pmull") {
      return Some(KernelFamily::ArmPmull);
    }
    if s.eq_ignore_ascii_case("pmull-eor3") || s.eq_ignore_ascii_case("pmulleor3") {
      return Some(KernelFamily::ArmPmullEor3);
    }
    if s.eq_ignore_ascii_case("sve2-pmull") || s.eq_ignore_ascii_case("sve2pmull") {
      return Some(KernelFamily::ArmSve2Pmull);
    }

    // Other architectures
    if s.eq_ignore_ascii_case("vpmsum") {
      return Some(KernelFamily::PowerVpmsum);
    }
    if s.eq_ignore_ascii_case("vgfm") {
      return Some(KernelFamily::S390xVgfm);
    }
    if s.eq_ignore_ascii_case("zbc") {
      return Some(KernelFamily::RiscvZbc);
    }
    if s.eq_ignore_ascii_case("zvbc") {
      return Some(KernelFamily::RiscvZvbc);
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
  #[cfg(target_arch = "aarch64")]
  fn arm_pmull_eor3_is_selected_when_available() {
    let caps = crate::platform::caps::aarch64::PMULL_EOR3_READY;
    let policy = SelectionPolicy::from_platform(caps);
    assert_eq!(policy.family(), KernelFamily::ArmPmullEor3);
    assert_eq!(policy.tunables(), PolicyTunables::DISABLED);
    assert_eq!(policy.max_streams, 3);
  }

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
  fn conservative_tunables_are_explicit() {
    let policy = SelectionPolicy::with_subfamily(Caps::NONE, KernelSubfamily::folding(KernelFamily::X86Pclmul));
    assert_eq!(policy.tunables(), PolicyTunables::DISABLED);
    assert!(!policy.should_use_simd(4096));
    assert_eq!(policy.streams_for_len(4096), 1);

    let tuned = policy.with_conservative_tunables();
    assert_eq!(
      tuned.tunables(),
      PolicyTunables {
        small_threshold: 64,
        fold_threshold: 64,
        wide_threshold: usize::MAX,
        min_bytes_per_lane: 256,
      }
    );
    assert!(tuned.should_use_simd(4096));
    assert_eq!(tuned.streams_for_len(4096), 8);
  }

  #[test]
  fn force_preserves_algorithm_tuning() {
    #[cfg(target_arch = "x86_64")]
    let (source_subfamily, target_family, caps, expected_max_streams) = (
      KernelSubfamily::wide(KernelFamily::X86Vpclmul, false),
      KernelFamily::X86Pclmul,
      crate::platform::caps::x86::VPCLMUL_READY,
      8,
    );
    #[cfg(target_arch = "aarch64")]
    let (source_subfamily, target_family, caps, expected_max_streams) = (
      KernelSubfamily::wide(KernelFamily::ArmPmullEor3, false),
      KernelFamily::ArmPmull,
      crate::platform::caps::aarch64::PMULL_EOR3_READY,
      3,
    );
    #[cfg(target_arch = "powerpc64")]
    let (source_subfamily, target_family, caps, expected_max_streams) = (
      KernelSubfamily::folding(KernelFamily::PowerVpmsum),
      KernelFamily::PowerVpmsum,
      crate::platform::caps::power::VPMSUM_READY,
      8,
    );
    #[cfg(target_arch = "s390x")]
    let (source_subfamily, target_family, caps, expected_max_streams) = (
      KernelSubfamily::folding(KernelFamily::S390xVgfm),
      KernelFamily::S390xVgfm,
      crate::platform::caps::s390x::VECTOR,
      4,
    );
    #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
    let (source_subfamily, target_family, caps, expected_max_streams) = (
      KernelSubfamily::wide(KernelFamily::RiscvZvbc, false),
      KernelFamily::RiscvZbc,
      crate::platform::caps::riscv::ZVBC,
      4,
    );
    #[cfg(not(any(
      target_arch = "x86_64",
      target_arch = "aarch64",
      target_arch = "powerpc64",
      target_arch = "s390x",
      target_arch = "riscv64",
      target_arch = "riscv32"
    )))]
    let (source_subfamily, target_family, caps, expected_max_streams) =
      (KernelSubfamily::portable(), KernelFamily::Portable, Caps::NONE, 1);

    let tuned = SelectionPolicy::with_subfamily(caps, source_subfamily).with_tunables(PolicyTunables {
      small_threshold: 96,
      fold_threshold: 128,
      wide_threshold: 1024,
      min_bytes_per_lane: 384,
    });

    let forced = tuned.with_force(ForceMode::Family(target_family));

    assert_eq!(forced.family(), target_family);
    assert_eq!(forced.tunables(), tuned.tunables());
    assert_eq!(forced.max_streams, expected_max_streams);
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
