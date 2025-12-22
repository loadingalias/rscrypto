//! Microarchitecture-derived tuning hints.
//!
//! `Tune` answers: "What should I *prefer* on this machine?"
//!
//! Unlike `CpuCaps` (which describes what's *possible*), `Tune` describes
//! what's *optimal*. This includes:
//!
//! - SIMD threshold (minimum buffer size for SIMD to be worthwhile)
//! - Strategy preferences (hybrid vs pure folding, etc.)
//! - Microarch-specific knobs
//!
//! # Design
//!
//! Tuning hints are derived from:
//! 1. Detected microarchitecture (x86_64: family/model, aarch64: feature combo)
//! 2. Known performance characteristics from benchmarks
//!
//! # Usage
//!
//! ```ignore
//! let tune = platform::tune();
//!
//! if data.len() < tune.simd_threshold {
//!     // Use scalar/small-buffer handler
//! } else {
//!     // Use SIMD kernel
//! }
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// TuneKind: Identity discriminant for O(1) name lookup
// ─────────────────────────────────────────────────────────────────────────────

/// Identifies which microarchitecture tuning preset is in use.
///
/// This discriminant enables O(1) `name()` lookup and disambiguates
/// presets that happen to have identical tuning values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TuneKind {
  Custom = 0,
  Default,
  Portable,
  // x86_64
  Zen4,
  Zen5,
  Zen5c,
  IntelSpr,
  IntelGnr,
  IntelIcl,
  // Apple Silicon
  AppleM1M3,
  AppleM4,
  AppleM5,
  // AWS Graviton
  Graviton2,
  Graviton3,
  Graviton4,
  Graviton5,
  // ARM Neoverse
  NeoverseN2,
  NeoverseN3,
  NeoverseV3,
  // Server ARM CPUs
  NvidiaGrace,
  AmpereAltra,
  Aarch64Pmull,
  // s390x (IBM Z)
  Z13,
  Z14,
  Z15,
  // PowerPC64
  Power7,
  Power8,
  Power9,
  Power10,
}

impl TuneKind {
  /// Returns the human-readable name for this tuning preset.
  #[must_use]
  pub const fn name(self) -> &'static str {
    match self {
      Self::Custom => "Custom",
      Self::Default => "Default",
      Self::Portable => "Portable",
      Self::Zen4 => "Zen4",
      Self::Zen5 => "Zen5",
      Self::Zen5c => "Zen5c",
      Self::IntelSpr => "Intel SPR",
      Self::IntelGnr => "Intel GNR",
      Self::IntelIcl => "Intel ICL",
      Self::AppleM1M3 => "Apple M1-M3",
      Self::AppleM4 => "Apple M4",
      Self::AppleM5 => "Apple M5",
      Self::Graviton2 => "Graviton 2",
      Self::Graviton3 => "Graviton 3",
      Self::Graviton4 => "Graviton 4",
      Self::Graviton5 => "Graviton 5",
      Self::NeoverseN2 => "Neoverse N2",
      Self::NeoverseN3 => "Neoverse N3",
      Self::NeoverseV3 => "Neoverse V3",
      Self::NvidiaGrace => "NVIDIA Grace",
      Self::AmpereAltra => "Ampere Altra",
      Self::Aarch64Pmull => "AArch64 PMULL",
      Self::Z13 => "IBM z13",
      Self::Z14 => "IBM z14",
      Self::Z15 => "IBM z15",
      Self::Power7 => "POWER7",
      Self::Power8 => "POWER8",
      Self::Power9 => "POWER9",
      Self::Power10 => "POWER10",
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tune: Microarchitecture tuning hints
// ─────────────────────────────────────────────────────────────────────────────

/// Microarchitecture-derived tuning hints for any workload.
///
/// Unlike [`Caps`](crate::Caps) (which describes what's *possible*), `Tune` describes
/// what's *optimal*. These hints guide algorithm selection and threshold decisions.
///
/// # Design
///
/// Tuning hints are derived from:
/// 1. Detected microarchitecture (x86_64: family/model, aarch64: feature combo)
/// 2. Known performance characteristics from benchmarks
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tune {
  /// Which preset this tuning came from (enables O(1) name lookup).
  pub kind: TuneKind,

  // ─── Universal SIMD Hints ───
  /// Minimum bytes where SIMD becomes faster than scalar.
  ///
  /// Below this threshold, scalar code or hardware instructions
  /// (when available) are often faster due to SIMD setup overhead.
  ///
  /// Typical values:
  /// - AMD Zen 4/5: 64 bytes (very low ZMM warmup)
  /// - Intel SPR/ICL: 256 bytes (higher ZMM warmup latency)
  /// - Apple M-series: 64-256 bytes (depends on path)
  pub simd_threshold: usize,

  // ─── CRC-Specific Thresholds ───
  /// Minimum bytes where PCLMUL/PMULL operations become faster than table CRC.
  ///
  /// PCLMULQDQ (x86_64), VPCLMULQDQ (AVX-512), and PMULL (aarch64) use
  /// carryless multiplication for CRC-64 computation. These have higher
  /// setup overhead than hardware CRC instructions.
  ///
  /// Typical values:
  /// - AMD Zen 4/5: 64 bytes (fast PCLMUL)
  /// - Intel: 128-256 bytes
  /// - Apple M-series: 64 bytes (fast PMULL)
  /// - aarch64 without PMULL: usize::MAX (use table fallback)
  pub pclmul_threshold: usize,

  /// Minimum bytes where hardware CRC instructions become faster than table CRC.
  ///
  /// SSE4.2 crc32 (x86_64) and aarch64 CRC32 extension have very low overhead
  /// and are effective even for small buffers. Used for CRC-32 and CRC-32C.
  ///
  /// Typical values:
  /// - x86_64 with SSE4.2: 8 bytes
  /// - aarch64 with CRC32: 8 bytes
  /// - Without hardware CRC: usize::MAX (use table fallback)
  pub hwcrc_threshold: usize,

  /// Maximum effective SIMD width in bits (128/256/512).
  ///
  /// Even if wider SIMD is available, some CPUs prefer narrower operations.
  /// For example, Ice Lake often prefers 256-bit even with AVX-512 support.
  ///
  /// Typical values:
  /// - AMD Zen 4/5: 512 (native AVX-512)
  /// - Intel ICL: 256 (throttling concerns)
  /// - Intel SPR: 512 (no throttling)
  /// - AArch64 NEON: 128
  /// - AArch64 SVE: matches `sve_vlen`
  pub effective_simd_width: u16,

  /// Whether wide registers (ZMM/SVE) have fast warmup.
  ///
  /// Intel CPUs have ~2000ns warmup when first using ZMM registers.
  /// AMD Zen 4/5 have ~60ns warmup.
  pub fast_wide_ops: bool,

  // ─── Parallelism Hints ───
  /// Recommended parallel instruction streams for throughput.
  ///
  /// This indicates how many independent instruction streams the CPU
  /// can execute efficiently in parallel (ILP capacity).
  ///
  /// Typical values:
  /// - Zen 5: 7 (exceptional ILP)
  /// - Zen 4/Intel: 3
  /// - Portable: 1
  pub parallel_streams: u8,

  /// Whether interleaving scalar+SIMD improves throughput.
  ///
  /// True on AMD Zen4/5 for many workloads where scalar and SIMD
  /// execution units can be utilized simultaneously.
  pub prefer_hybrid: bool,

  // ─── Memory Hints ───
  /// L1 cache line size in bytes.
  ///
  /// Typical values:
  /// - x86_64: 64
  /// - AArch64: 64 (Apple: 128 for efficiency cores)
  pub cache_line: u8,

  /// Prefetch distance hint in bytes (0 = let hardware decide).
  ///
  /// When non-zero, suggests how far ahead to prefetch data.
  /// Useful for memory-bound streaming workloads.
  pub prefetch_distance: u16,

  // ─── Architecture-Specific ───
  /// SVE vector length in bits (0 = no SVE).
  ///
  /// Common values:
  /// - 0: No SVE support
  /// - 128: Graviton 4, Neoverse N2 (smaller SVE for more cores)
  /// - 256: Graviton 3, Neoverse V1
  /// - 512: Some HPC implementations
  pub sve_vlen: u16,

  /// SME tile size in bits (0 = no SME).
  ///
  /// Apple M4 has SME with Streaming SVE mode but not full SVE.
  /// When non-zero, indicates SME is available with this tile dimension.
  pub sme_tile: u16,
}

impl Tune {
  /// Conservative defaults for unknown CPUs.
  pub const DEFAULT: Self = Self {
    kind: TuneKind::Default,
    simd_threshold: 256,
    pclmul_threshold: 256,
    hwcrc_threshold: 16,
    effective_simd_width: 128,
    fast_wide_ops: false,
    parallel_streams: 3,
    prefer_hybrid: false,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // x86_64 Presets
  // ─────────────────────────────────────────────────────────────────────────────

  /// Tuning for AMD Zen 4.
  pub const ZEN4: Self = Self {
    kind: TuneKind::Zen4,
    simd_threshold: 64,
    pclmul_threshold: 64,
    hwcrc_threshold: 8,
    effective_simd_width: 512,
    fast_wide_ops: true,
    parallel_streams: 3,
    prefer_hybrid: true,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };

  /// Tuning for AMD Zen 5.
  ///
  /// Zen 5 has significantly improved ILP capacity with 8-wide dispatch/rename
  /// (up from 6-wide in Zen 4). The parallel_streams value of 7 reflects the
  /// practical ILP width observed for integer operations like crc32q, where
  /// 7-way parallelism has been measured in practice. While the frontend is
  /// 8-wide, not all instruction types can sustain 8-way execution due to
  /// execution port constraints.
  pub const ZEN5: Self = Self {
    kind: TuneKind::Zen5,
    simd_threshold: 64,
    pclmul_threshold: 64,
    hwcrc_threshold: 8,
    effective_simd_width: 512,
    fast_wide_ops: true,
    parallel_streams: 7, // Zen 5 has 8-wide dispatch, but 7-way ILP for integer ops
    prefer_hybrid: true,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };

  /// Tuning for AMD Zen 5c (compact/efficiency variant).
  ///
  /// Zen 5c is a density-optimized variant of Zen 5, ~25% smaller die area
  /// than full Zen 5. Used in EPYC 9005 series (up to 192 cores) and Strix Point
  /// APUs (hybrid Zen 5 + Zen 5c). Has same ISA support as Zen 5 (AVX-512,
  /// VPCLMULQDQ) but slightly different performance characteristics due to
  /// optimizations for density. Conservative parallel_streams value accounts
  /// for potential execution resource differences.
  pub const ZEN5C: Self = Self {
    kind: TuneKind::Zen5c,
    simd_threshold: 64,
    pclmul_threshold: 64,
    hwcrc_threshold: 8,
    effective_simd_width: 512,
    fast_wide_ops: true,
    parallel_streams: 6, // Conservative, Zen 5c is optimized for density
    prefer_hybrid: true,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };

  /// Tuning for Intel Sapphire Rapids / Emerald Rapids.
  ///
  /// Note: Granite Rapids now has its own preset (INTEL_GNR) due to
  /// AMX enhancements and different characteristics.
  pub const INTEL_SPR: Self = Self {
    kind: TuneKind::IntelSpr,
    simd_threshold: 256,
    pclmul_threshold: 128,
    hwcrc_threshold: 8,
    effective_simd_width: 512,
    fast_wide_ops: false,
    parallel_streams: 3,
    prefer_hybrid: false,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };

  /// Tuning for Intel Granite Rapids (Xeon 6).
  ///
  /// Granite Rapids (6th gen Xeon, launched Sept 2024) features Redwood Cove
  /// P-cores with up to 128 cores. Key improvements over SPR:
  /// - Enhanced AMX with FP16 and Complex operations (AMX_FP16, AMX_COMPLEX)
  /// - AVX-512-FP16 support
  /// - APX (Advanced Performance Extensions) - 32 GPRs on select SKUs
  /// - Integrated accelerators: QAT, DSA, IAA
  /// - 2MB L2 per core, improved ILP with 8-wide decode
  ///
  /// Tuning is similar to SPR but accounts for improved execution resources.
  pub const INTEL_GNR: Self = Self {
    kind: TuneKind::IntelGnr,
    simd_threshold: 256,
    pclmul_threshold: 128,
    hwcrc_threshold: 8,
    effective_simd_width: 512,
    fast_wide_ops: false, // Still has ZMM warmup, though improved
    parallel_streams: 4,  // Improved over SPR due to enhanced execution
    prefer_hybrid: false,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };

  /// Tuning for Intel Ice Lake.
  ///
  /// Note: Ice Lake prefers 256-bit operations despite having AVX-512,
  /// due to frequency throttling concerns.
  pub const INTEL_ICL: Self = Self {
    kind: TuneKind::IntelIcl,
    simd_threshold: 256,
    pclmul_threshold: 128,
    hwcrc_threshold: 8,
    effective_simd_width: 256, // ICL prefers 256-bit even with AVX-512
    fast_wide_ops: false,
    parallel_streams: 3,
    prefer_hybrid: false,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // Apple Silicon Presets
  // ─────────────────────────────────────────────────────────────────────────────

  /// Tuning for Apple M1/M2/M3 (PMULL+EOR3, no SVE, no SME).
  pub const APPLE_M1_M3: Self = Self {
    kind: TuneKind::AppleM1M3,
    simd_threshold: 64,
    pclmul_threshold: 64,
    hwcrc_threshold: 8,
    effective_simd_width: 128,
    fast_wide_ops: true, // Apple Silicon has efficient NEON
    parallel_streams: 3,
    prefer_hybrid: false,
    cache_line: 64, // P-cores use 128, but 64 is safer default
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };

  /// Tuning for Apple M4 (PMULL+EOR3, SME with Streaming SVE, no full SVE).
  pub const APPLE_M4: Self = Self {
    kind: TuneKind::AppleM4,
    simd_threshold: 64,
    pclmul_threshold: 64,
    hwcrc_threshold: 8,
    effective_simd_width: 128,
    fast_wide_ops: true,
    parallel_streams: 3,
    prefer_hybrid: false,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 256, // M4 has SME with 256-bit tiles
  };

  /// Tuning for Apple M5 (PMULL+EOR3, SME2p1).
  ///
  /// Released October 2025. Codename: Hidra (base), Sotra (Pro/Max).
  /// Features SME2p1, SMEB16B16, SMEF16F16 per LLVM commit f85494f6afeb.
  /// Slightly higher parallelism than M4 due to architectural improvements.
  pub const APPLE_M5: Self = Self {
    kind: TuneKind::AppleM5,
    simd_threshold: 64,
    pclmul_threshold: 64,
    hwcrc_threshold: 8,
    effective_simd_width: 128,
    fast_wide_ops: true,
    parallel_streams: 4, // M5 has improved parallelism
    prefer_hybrid: false,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 256, // M5 has SME2p1 with 256-bit tiles
  };

  /// Alias: Apple M-series (M1-M3).
  pub const APPLE_M: Self = Self::APPLE_M1_M3;

  // ─────────────────────────────────────────────────────────────────────────────
  // AWS Graviton Presets
  // ─────────────────────────────────────────────────────────────────────────────

  /// Tuning for AWS Graviton 2 (Neoverse N1, no SVE).
  pub const GRAVITON2: Self = Self {
    kind: TuneKind::Graviton2,
    simd_threshold: 64,
    pclmul_threshold: 64,
    hwcrc_threshold: 8,
    effective_simd_width: 128,
    fast_wide_ops: true,
    parallel_streams: 3,
    prefer_hybrid: false,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };

  /// Tuning for AWS Graviton 3 (Neoverse V1, 256-bit SVE).
  pub const GRAVITON3: Self = Self {
    kind: TuneKind::Graviton3,
    simd_threshold: 64,
    pclmul_threshold: 64,
    hwcrc_threshold: 8,
    effective_simd_width: 256,
    fast_wide_ops: true,
    parallel_streams: 3,
    prefer_hybrid: false,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 256,
    sme_tile: 0,
  };

  /// Tuning for AWS Graviton 4 (Neoverse V2, 128-bit SVE for more cores).
  pub const GRAVITON4: Self = Self {
    kind: TuneKind::Graviton4,
    simd_threshold: 64,
    pclmul_threshold: 64,
    hwcrc_threshold: 8,
    effective_simd_width: 128,
    fast_wide_ops: true,
    parallel_streams: 3,
    prefer_hybrid: false,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 128,
    sme_tile: 0,
  };

  /// Tuning for AWS Graviton 5 (Neoverse V3, 128-bit SVE2).
  /// Released late 2025, uses Poseidon cores with enhanced ILP.
  pub const GRAVITON5: Self = Self {
    kind: TuneKind::Graviton5,
    simd_threshold: 64,
    pclmul_threshold: 64,
    hwcrc_threshold: 8,
    effective_simd_width: 128,
    fast_wide_ops: true,
    parallel_streams: 4, // V3 has improved parallelism over V2
    prefer_hybrid: false,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 128,
    sme_tile: 0,
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // ARM Neoverse Presets
  // ─────────────────────────────────────────────────────────────────────────────

  /// Tuning for ARM Neoverse N2 (128-bit SVE).
  pub const NEOVERSE_N2: Self = Self {
    kind: TuneKind::NeoverseN2,
    simd_threshold: 64,
    pclmul_threshold: 64,
    hwcrc_threshold: 8,
    effective_simd_width: 128,
    fast_wide_ops: true,
    parallel_streams: 3,
    prefer_hybrid: false,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 128,
    sme_tile: 0,
  };

  /// Tuning for ARM Neoverse N3 (Hermes, Armv9.2-A with 128-bit SVE2).
  ///
  /// N3 is ARM's efficiency-focused core with Armv9.2-A, offering triple
  /// the ML performance of N2 while maintaining excellent power efficiency.
  /// Features 32-64KB L1 cache and 128KB-2MB L2 with ECC.
  pub const NEOVERSE_N3: Self = Self {
    kind: TuneKind::NeoverseN3,
    simd_threshold: 64,
    pclmul_threshold: 64,
    hwcrc_threshold: 8,
    effective_simd_width: 128,
    fast_wide_ops: true,
    parallel_streams: 3,
    prefer_hybrid: false,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 128, // 128-bit SVE2
    sme_tile: 0,
  };

  /// Tuning for ARM Neoverse V3 (Poseidon, 128-bit SVE2).
  pub const NEOVERSE_V3: Self = Self {
    kind: TuneKind::NeoverseV3,
    simd_threshold: 64,
    pclmul_threshold: 64,
    hwcrc_threshold: 8,
    effective_simd_width: 128,
    fast_wide_ops: true,
    parallel_streams: 4,
    prefer_hybrid: false,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 128,
    sme_tile: 0,
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // Server ARM CPU Presets
  // ─────────────────────────────────────────────────────────────────────────────

  /// Tuning for NVIDIA Grace CPU (Neoverse V2-based, 72 cores, 128-bit SVE2).
  ///
  /// Grace features 72 Armv9 Neoverse V2 cores with 4×128-bit SIMD pipelines
  /// supporting both NEON and SVE2. Each core has 64KB L1I + 64KB L1D,
  /// 1MB L2, and shares 177MB L3 cache. Optimized for HPC and AI workloads
  /// with 480GB LPDDR5X and 546GB/s memory bandwidth.
  pub const NVIDIA_GRACE: Self = Self {
    kind: TuneKind::NvidiaGrace,
    simd_threshold: 64,
    pclmul_threshold: 64,
    hwcrc_threshold: 8,
    effective_simd_width: 128,
    fast_wide_ops: true,
    parallel_streams: 4, // 4×128-bit SIMD ALU pipelines per core
    prefer_hybrid: false,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 128, // Neoverse V2 with 128-bit SVE2
    sme_tile: 0,
  };

  /// Tuning for Ampere Altra (Neoverse N1-based, ARMv8.2+, no SVE).
  ///
  /// Altra features up to 128 Neoverse N1 cores (ARMv8.2-A) with 64KB L1I + 64KB L1D
  /// and 1MB private L2 per core, plus up to 32MB distributed L3. Built on TSMC 7nm,
  /// it offers excellent single-threaded performance and power efficiency.
  /// No SVE support - relies on NEON for SIMD.
  pub const AMPERE_ALTRA: Self = Self {
    kind: TuneKind::AmpereAltra,
    simd_threshold: 64,
    pclmul_threshold: 64,
    hwcrc_threshold: 8,
    effective_simd_width: 128,
    fast_wide_ops: true,
    parallel_streams: 3,
    prefer_hybrid: false,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 0, // No SVE - Neoverse N1 is ARMv8.2-A only
    sme_tile: 0,
  };

  /// Tuning for generic aarch64 with PMULL (no SVE).
  pub const AARCH64_PMULL: Self = Self {
    kind: TuneKind::Aarch64Pmull,
    simd_threshold: 256,
    pclmul_threshold: 128,
    hwcrc_threshold: 8,
    effective_simd_width: 128,
    fast_wide_ops: false,
    parallel_streams: 3,
    prefer_hybrid: false,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // s390x (IBM Z) Presets
  // ─────────────────────────────────────────────────────────────────────────────

  /// Tuning for IBM z13 (base vector facility).
  ///
  /// z13 introduced 128-bit SIMD with 32 vector registers and two independent
  /// SIMD units. First generation with SMT-2 support.
  /// Note: z/Architecture uses 256-byte cache lines, but cache_line is u8, so we use 128.
  pub const Z13: Self = Self {
    kind: TuneKind::Z13,
    simd_threshold: 256,
    pclmul_threshold: 256,       // CRC64 VGFM folding crossover (z13+ vector)
    hwcrc_threshold: usize::MAX, // No hardware CRC on s390x
    effective_simd_width: 128,
    fast_wide_ops: true,
    parallel_streams: 2, // Two SIMD units
    prefer_hybrid: false,
    cache_line: 128, // z/Architecture uses 256-byte lines, capped to u8::MAX/2
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };

  /// Tuning for IBM z14 (vector enhancements 1).
  ///
  /// z14 added enhanced SIMD instructions and improved crypto (6x faster than z13).
  /// Note: z/Architecture uses 256-byte cache lines, but cache_line is u8, so we use 128.
  pub const Z14: Self = Self {
    kind: TuneKind::Z14,
    simd_threshold: 128,
    pclmul_threshold: 128,       // CRC64 VGFM folding crossover (z14+)
    hwcrc_threshold: usize::MAX, // No hardware CRC on s390x
    effective_simd_width: 128,
    fast_wide_ops: true,
    parallel_streams: 2,
    prefer_hybrid: false,
    cache_line: 128,
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };

  /// Tuning for IBM z15 (vector enhancements 2).
  ///
  /// z15 added DEFLATE compression acceleration and enhanced sort.
  /// 14% single-thread improvement over z14.
  /// Note: z/Architecture uses 256-byte cache lines, but cache_line is u8, so we use 128.
  pub const Z15: Self = Self {
    kind: TuneKind::Z15,
    simd_threshold: 64,
    pclmul_threshold: 64,        // CRC64 VGFM folding crossover (z15+)
    hwcrc_threshold: usize::MAX, // No hardware CRC on s390x
    effective_simd_width: 128,
    fast_wide_ops: true,
    parallel_streams: 2,
    prefer_hybrid: false,
    cache_line: 128,
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // PowerPC64 Presets
  // ─────────────────────────────────────────────────────────────────────────────

  /// Tuning for IBM POWER7 (VSX).
  ///
  /// POWER7 introduced VSX (Vector-Scalar Extension) with 64 vector registers.
  pub const POWER7: Self = Self {
    kind: TuneKind::Power7,
    simd_threshold: 256,
    pclmul_threshold: usize::MAX, // No PCLMUL on POWER
    hwcrc_threshold: usize::MAX,  // No hardware CRC on POWER
    effective_simd_width: 128,
    fast_wide_ops: true,
    parallel_streams: 2,
    prefer_hybrid: false,
    cache_line: 128, // POWER uses 128-byte cache lines
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };

  /// Tuning for IBM POWER8 (power8-vector + crypto).
  ///
  /// POWER8 added two fully symmetric vector pipelines and a dedicated crypto pipeline
  /// (AES, GCM, SHA-2). First with hardware transactional memory.
  pub const POWER8: Self = Self {
    kind: TuneKind::Power8,
    simd_threshold: 128,
    pclmul_threshold: 128,       // VPMSUMD folding crossover (POWER8+)
    hwcrc_threshold: usize::MAX, // No hardware CRC on POWER
    effective_simd_width: 128,
    fast_wide_ops: true,
    parallel_streams: 2, // Two vector pipelines
    prefer_hybrid: false,
    cache_line: 128,
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };

  /// Tuning for IBM POWER9 (power9-vector).
  ///
  /// POWER9 added on-chip compression and cryptography co-processors.
  pub const POWER9: Self = Self {
    kind: TuneKind::Power9,
    simd_threshold: 64,
    pclmul_threshold: 64,        // VPMSUMD folding crossover (POWER9+)
    hwcrc_threshold: usize::MAX, // No hardware CRC on POWER
    effective_simd_width: 128,
    fast_wide_ops: true,
    parallel_streams: 4, // Improved execution resources
    prefer_hybrid: false,
    cache_line: 128,
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };

  /// Tuning for IBM POWER10 (power10-vector).
  ///
  /// POWER10 has 4x crypto engines per core (2.5x faster AES/SHA), Matrix Math Assist (MMA)
  /// for AI workloads, and improved inferencing (5-20x over POWER9).
  pub const POWER10: Self = Self {
    kind: TuneKind::Power10,
    simd_threshold: 64,
    pclmul_threshold: 64,        // VPMSUMD folding crossover (POWER10+)
    hwcrc_threshold: usize::MAX, // No hardware CRC on POWER
    effective_simd_width: 128,
    fast_wide_ops: true,
    parallel_streams: 4,
    prefer_hybrid: false,
    cache_line: 128,
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // Fallback Presets
  // ─────────────────────────────────────────────────────────────────────────────

  /// Tuning for portable/scalar code.
  ///
  /// Uses conservative simd_threshold (256) matching DEFAULT - while SIMD won't
  /// be selected for truly portable code (no SIMD caps), this ensures consistent
  /// behavior if someone overrides to PORTABLE on a SIMD-capable system.
  pub const PORTABLE: Self = Self {
    kind: TuneKind::Portable,
    simd_threshold: 256,
    pclmul_threshold: usize::MAX, // No PCLMUL for portable
    hwcrc_threshold: usize::MAX,  // No hardware CRC for portable
    effective_simd_width: 0,      // No SIMD
    fast_wide_ops: false,
    parallel_streams: 1,
    prefer_hybrid: false,
    cache_line: 64,
    prefetch_distance: 0,
    sve_vlen: 0,
    sme_tile: 0,
  };
}

impl Default for Tune {
  #[inline]
  fn default() -> Self {
    Self::DEFAULT
  }
}

impl core::fmt::Display for Tune {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(
      f,
      "Tune({}, simd_threshold={}, width={}, streams={}, hybrid={}, fast_wide={}, cache={})",
      self.kind.name(),
      self.simd_threshold,
      self.effective_simd_width,
      self.parallel_streams,
      self.prefer_hybrid,
      self.fast_wide_ops,
      self.cache_line
    )?;
    if self.sve_vlen > 0 {
      write!(f, ", sve={}", self.sve_vlen)?;
    }
    if self.sme_tile > 0 {
      write!(f, ", sme={}", self.sme_tile)?;
    }
    Ok(())
  }
}

impl Tune {
  /// Returns the kind discriminant for this tuning configuration.
  #[inline]
  #[must_use]
  pub const fn kind(&self) -> TuneKind {
    self.kind
  }

  /// Returns a descriptive name for this tuning configuration.
  ///
  /// O(1) lookup via the stored `TuneKind` discriminant.
  #[inline]
  #[must_use]
  pub const fn name(&self) -> &'static str {
    self.kind.name()
  }
}

#[cfg(test)]
extern crate alloc;

#[cfg(test)]
mod tests {
  use super::*;

  // ─────────────────────────────────────────────────────────────────────────────
  // Basic Tune Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_tune_defaults() {
    let tune = Tune::default();
    assert_eq!(tune.simd_threshold, 256);
    assert_eq!(tune.effective_simd_width, 128);
    assert!(!tune.prefer_hybrid);
    assert_eq!(tune.parallel_streams, 3);
    assert_eq!(tune.cache_line, 64);
    assert_eq!(tune.sve_vlen, 0);
    assert_eq!(tune.sme_tile, 0);
    assert_eq!(tune, Tune::DEFAULT);
  }

  #[test]
  fn test_tune_zen5() {
    let tune = Tune::ZEN5;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.effective_simd_width, 512);
    assert!(tune.prefer_hybrid);
    assert_eq!(tune.parallel_streams, 7);
    assert!(tune.fast_wide_ops);
    assert_eq!(tune.cache_line, 64);
    assert_eq!(tune.sve_vlen, 0);
    assert_eq!(tune.sme_tile, 0);
  }

  #[test]
  fn test_tune_zen4() {
    let tune = Tune::ZEN4;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.effective_simd_width, 512);
    assert!(tune.prefer_hybrid);
    assert_eq!(tune.parallel_streams, 3);
    assert!(tune.fast_wide_ops);
    assert_eq!(tune.sve_vlen, 0);
    assert_eq!(tune.sme_tile, 0);
  }

  #[test]
  fn test_tune_zen5c() {
    let tune = Tune::ZEN5C;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.effective_simd_width, 512);
    assert!(tune.prefer_hybrid);
    assert_eq!(tune.parallel_streams, 6); // Conservative for density-optimized
    assert!(tune.fast_wide_ops);
    assert_eq!(tune.cache_line, 64);
    assert_eq!(tune.sve_vlen, 0);
    assert_eq!(tune.sme_tile, 0);
    assert_eq!(tune.kind(), TuneKind::Zen5c);
    assert_eq!(tune.name(), "Zen5c");
  }

  #[test]
  fn test_tune_intel_spr() {
    let tune = Tune::INTEL_SPR;
    assert_eq!(tune.simd_threshold, 256);
    assert_eq!(tune.effective_simd_width, 512);
    assert!(!tune.prefer_hybrid);
    assert!(!tune.fast_wide_ops);
    assert_eq!(tune.sve_vlen, 0);
  }

  #[test]
  fn test_tune_intel_gnr() {
    let tune = Tune::INTEL_GNR;
    assert_eq!(tune.simd_threshold, 256);
    assert_eq!(tune.effective_simd_width, 512);
    assert!(!tune.prefer_hybrid);
    assert_eq!(tune.parallel_streams, 4); // Improved over SPR
    assert!(!tune.fast_wide_ops);
    assert_eq!(tune.cache_line, 64);
    assert_eq!(tune.sve_vlen, 0);
    assert_eq!(tune.sme_tile, 0);
    assert_eq!(tune.kind(), TuneKind::IntelGnr);
    assert_eq!(tune.name(), "Intel GNR");
  }

  #[test]
  fn test_tune_intel_icl() {
    let tune = Tune::INTEL_ICL;
    assert_eq!(tune.simd_threshold, 256);
    assert_eq!(tune.effective_simd_width, 256); // ICL prefers 256-bit
    assert!(!tune.prefer_hybrid);
    assert_eq!(tune.parallel_streams, 3);
    assert!(!tune.fast_wide_ops);
  }

  #[test]
  fn test_tune_apple_m1_m3() {
    let tune = Tune::APPLE_M1_M3;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.effective_simd_width, 128);
    assert!(!tune.prefer_hybrid);
    assert!(tune.fast_wide_ops); // Apple Silicon has efficient NEON
    assert_eq!(tune.sve_vlen, 0);
    assert_eq!(tune.sme_tile, 0);
    // APPLE_M is an alias
    assert_eq!(Tune::APPLE_M, Tune::APPLE_M1_M3);
  }

  #[test]
  fn test_tune_apple_m4() {
    let tune = Tune::APPLE_M4;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.effective_simd_width, 128);
    assert!(!tune.prefer_hybrid);
    assert!(tune.fast_wide_ops);
    assert_eq!(tune.sve_vlen, 0); // M4 has SME, not full SVE
    assert_eq!(tune.sme_tile, 256); // M4 has SME with 256-bit tiles
  }

  #[test]
  fn test_tune_graviton2() {
    let tune = Tune::GRAVITON2;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.effective_simd_width, 128);
    assert_eq!(tune.sve_vlen, 0); // Neoverse N1 has no SVE
    assert_eq!(tune.sme_tile, 0);
  }

  #[test]
  fn test_tune_graviton3() {
    let tune = Tune::GRAVITON3;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.effective_simd_width, 256);
    assert_eq!(tune.sve_vlen, 256); // Neoverse V1 has 256-bit SVE
    assert_eq!(tune.sme_tile, 0);
  }

  #[test]
  fn test_tune_graviton4() {
    let tune = Tune::GRAVITON4;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.effective_simd_width, 128);
    assert_eq!(tune.sve_vlen, 128); // Neoverse V2 reduced to 128-bit
    assert_eq!(tune.sme_tile, 0);
  }

  #[test]
  fn test_tune_graviton5() {
    let tune = Tune::GRAVITON5;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.effective_simd_width, 128);
    assert_eq!(tune.sve_vlen, 128); // Neoverse V3 with 128-bit SVE2
    assert_eq!(tune.sme_tile, 0);
    assert_eq!(tune.parallel_streams, 4); // V3 has improved parallelism
    assert!(tune.fast_wide_ops);
    assert_eq!(tune.kind(), TuneKind::Graviton5);
    assert_eq!(tune.name(), "Graviton 5");
  }

  #[test]
  fn test_tune_neoverse_n2() {
    let tune = Tune::NEOVERSE_N2;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.effective_simd_width, 128);
    assert_eq!(tune.sve_vlen, 128);
    assert_eq!(tune.sme_tile, 0);
  }

  #[test]
  fn test_tune_neoverse_n3() {
    let tune = Tune::NEOVERSE_N3;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.effective_simd_width, 128);
    assert_eq!(tune.sve_vlen, 128); // Neoverse N3 (Hermes) 128-bit SVE2
    assert_eq!(tune.sme_tile, 0);
    assert_eq!(tune.parallel_streams, 3);
    assert!(tune.fast_wide_ops);
    assert_eq!(tune.kind(), TuneKind::NeoverseN3);
    assert_eq!(tune.name(), "Neoverse N3");
  }

  #[test]
  fn test_tune_neoverse_v3() {
    let tune = Tune::NEOVERSE_V3;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.effective_simd_width, 128);
    assert_eq!(tune.sve_vlen, 128); // Neoverse V3 (Poseidon) 128-bit SVE2
    assert_eq!(tune.sme_tile, 0);
    assert_eq!(tune.parallel_streams, 4); // Improved parallelism
    assert!(tune.fast_wide_ops);
    assert_eq!(tune.kind(), TuneKind::NeoverseV3);
    assert_eq!(tune.name(), "Neoverse V3");
  }

  #[test]
  fn test_tune_aarch64_pmull() {
    let tune = Tune::AARCH64_PMULL;
    assert_eq!(tune.simd_threshold, 256);
    assert_eq!(tune.effective_simd_width, 128);
    assert!(!tune.prefer_hybrid);
    assert_eq!(tune.sve_vlen, 0);
  }

  #[test]
  fn test_tune_nvidia_grace() {
    let tune = Tune::NVIDIA_GRACE;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.effective_simd_width, 128);
    assert_eq!(tune.sve_vlen, 128); // Neoverse V2-based with 128-bit SVE2
    assert_eq!(tune.sme_tile, 0);
    assert_eq!(tune.parallel_streams, 4); // 4×128-bit SIMD pipelines
    assert!(tune.fast_wide_ops);
    assert_eq!(tune.cache_line, 64);
    assert_eq!(tune.kind(), TuneKind::NvidiaGrace);
    assert_eq!(tune.name(), "NVIDIA Grace");
  }

  #[test]
  fn test_tune_ampere_altra() {
    let tune = Tune::AMPERE_ALTRA;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.effective_simd_width, 128);
    assert_eq!(tune.sve_vlen, 0); // No SVE - Neoverse N1 (ARMv8.2-A)
    assert_eq!(tune.sme_tile, 0);
    assert_eq!(tune.parallel_streams, 3);
    assert!(tune.fast_wide_ops);
    assert_eq!(tune.cache_line, 64);
    assert_eq!(tune.kind(), TuneKind::AmpereAltra);
    assert_eq!(tune.name(), "Ampere Altra");
  }

  #[test]
  fn test_tune_portable() {
    let tune = Tune::PORTABLE;
    assert_eq!(tune.simd_threshold, 256); // Conservative, matches DEFAULT
    assert_eq!(tune.effective_simd_width, 0); // No SIMD
    assert!(!tune.prefer_hybrid);
    assert_eq!(tune.parallel_streams, 1);
    assert!(!tune.fast_wide_ops);
    assert_eq!(tune.cache_line, 64);
    assert_eq!(tune.sve_vlen, 0);
    assert_eq!(tune.sme_tile, 0);
  }

  #[test]
  fn test_tune_z13() {
    let tune = Tune::Z13;
    assert_eq!(tune.simd_threshold, 256);
    assert_eq!(tune.effective_simd_width, 128);
    assert!(tune.fast_wide_ops);
    assert_eq!(tune.parallel_streams, 2); // Two SIMD units
    assert!(!tune.prefer_hybrid);
    assert_eq!(tune.cache_line, 128); // z/Arch uses 256-byte, capped
    assert_eq!(tune.kind(), TuneKind::Z13);
    assert_eq!(tune.name(), "IBM z13");
  }

  #[test]
  fn test_tune_z15() {
    let tune = Tune::Z15;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.effective_simd_width, 128);
    assert!(tune.fast_wide_ops);
    assert_eq!(tune.parallel_streams, 2);
    assert_eq!(tune.kind(), TuneKind::Z15);
    assert_eq!(tune.name(), "IBM z15");
  }

  #[test]
  fn test_tune_power8() {
    let tune = Tune::POWER8;
    assert_eq!(tune.simd_threshold, 128);
    assert_eq!(tune.effective_simd_width, 128);
    assert!(tune.fast_wide_ops);
    assert_eq!(tune.parallel_streams, 2); // Two vector pipelines
    assert!(!tune.prefer_hybrid);
    assert_eq!(tune.cache_line, 128); // POWER uses 128-byte lines
    assert_eq!(tune.kind(), TuneKind::Power8);
    assert_eq!(tune.name(), "POWER8");
  }

  #[test]
  fn test_tune_power10() {
    let tune = Tune::POWER10;
    assert_eq!(tune.simd_threshold, 64);
    assert_eq!(tune.effective_simd_width, 128);
    assert!(tune.fast_wide_ops);
    assert_eq!(tune.parallel_streams, 4);
    assert_eq!(tune.cache_line, 128);
    assert_eq!(tune.kind(), TuneKind::Power10);
    assert_eq!(tune.name(), "POWER10");
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Tune Name Identification (O(1) via TuneKind discriminant)
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_tune_name_presets() {
    // Each preset now returns its distinct name via TuneKind discriminant
    assert_eq!(Tune::ZEN5.name(), "Zen5");
    assert_eq!(Tune::ZEN5C.name(), "Zen5c");
    assert_eq!(Tune::ZEN4.name(), "Zen4");
    assert_eq!(Tune::INTEL_SPR.name(), "Intel SPR");
    assert_eq!(Tune::INTEL_GNR.name(), "Intel GNR");
    assert_eq!(Tune::INTEL_ICL.name(), "Intel ICL");
    assert_eq!(Tune::APPLE_M1_M3.name(), "Apple M1-M3");
    assert_eq!(Tune::APPLE_M4.name(), "Apple M4");
    assert_eq!(Tune::APPLE_M5.name(), "Apple M5");
    assert_eq!(Tune::GRAVITON2.name(), "Graviton 2");
    assert_eq!(Tune::GRAVITON3.name(), "Graviton 3");
    assert_eq!(Tune::GRAVITON4.name(), "Graviton 4");
    assert_eq!(Tune::GRAVITON5.name(), "Graviton 5");
    assert_eq!(Tune::NEOVERSE_N2.name(), "Neoverse N2");
    assert_eq!(Tune::NEOVERSE_N3.name(), "Neoverse N3");
    assert_eq!(Tune::NEOVERSE_V3.name(), "Neoverse V3");
    assert_eq!(Tune::NVIDIA_GRACE.name(), "NVIDIA Grace");
    assert_eq!(Tune::AMPERE_ALTRA.name(), "Ampere Altra");
    assert_eq!(Tune::AARCH64_PMULL.name(), "AArch64 PMULL");
    assert_eq!(Tune::Z13.name(), "IBM z13");
    assert_eq!(Tune::Z14.name(), "IBM z14");
    assert_eq!(Tune::Z15.name(), "IBM z15");
    assert_eq!(Tune::POWER7.name(), "POWER7");
    assert_eq!(Tune::POWER8.name(), "POWER8");
    assert_eq!(Tune::POWER9.name(), "POWER9");
    assert_eq!(Tune::POWER10.name(), "POWER10");
    assert_eq!(Tune::PORTABLE.name(), "Portable");
    assert_eq!(Tune::DEFAULT.name(), "Default");
  }

  #[test]
  fn test_tune_kind_accessor() {
    assert_eq!(Tune::ZEN5.kind(), TuneKind::Zen5);
    assert_eq!(Tune::INTEL_SPR.kind(), TuneKind::IntelSpr);
    assert_eq!(Tune::APPLE_M4.kind(), TuneKind::AppleM4);
    assert_eq!(Tune::GRAVITON3.kind(), TuneKind::Graviton3);
  }

  #[test]
  fn test_tune_custom_construction() {
    // Users can construct custom Tune values directly using struct syntax
    let custom = Tune {
      kind: TuneKind::Custom,
      simd_threshold: 128,
      pclmul_threshold: 128,
      hwcrc_threshold: 8,
      effective_simd_width: 256,
      fast_wide_ops: true,
      parallel_streams: 5,
      prefer_hybrid: true,
      cache_line: 64,
      prefetch_distance: 512,
      sve_vlen: 256,
      sme_tile: 128,
    };
    assert_eq!(custom.name(), "Custom");
    assert_eq!(custom.kind(), TuneKind::Custom);
    assert_eq!(custom.simd_threshold, 128);
    assert_eq!(custom.effective_simd_width, 256);
    assert!(custom.fast_wide_ops);
    assert_eq!(custom.parallel_streams, 5);
    assert!(custom.prefer_hybrid);
    assert_eq!(custom.cache_line, 64);
    assert_eq!(custom.prefetch_distance, 512);
    assert_eq!(custom.sve_vlen, 256);
    assert_eq!(custom.sme_tile, 128);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Display Formatting
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_tune_display() {
    let tune = Tune::ZEN5;
    let s = alloc::format!("{tune}");
    assert!(s.contains("Zen5"));
    assert!(s.contains("simd_threshold=64"));
    assert!(s.contains("width=512"));
    assert!(s.contains("streams=7"));
    assert!(s.contains("hybrid=true"));
    assert!(s.contains("fast_wide=true"));
    assert!(s.contains("cache=64"));
    // SVE and SME are 0, so they shouldn't appear in condensed output
    assert!(!s.contains("sve="));
    assert!(!s.contains("sme="));
  }

  #[test]
  fn test_tune_display_graviton3() {
    let tune = Tune::GRAVITON3;
    let s = alloc::format!("{tune}");
    assert!(s.contains("Graviton 3"));
    assert!(s.contains("width=256"));
    assert!(s.contains("sve=256"));
    // No SME, so shouldn't appear
    assert!(!s.contains("sme="));
  }

  #[test]
  fn test_tune_display_apple_m4() {
    let tune = Tune::APPLE_M4;
    let s = alloc::format!("{tune}");
    assert!(s.contains("Apple M4"));
    // No SVE, so shouldn't appear
    assert!(!s.contains("sve="));
    // Has SME
    assert!(s.contains("sme=256"));
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // All Presets Have Reasonable Values
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_all_presets_have_reasonable_values() {
    let presets = [
      Tune::DEFAULT,
      Tune::ZEN4,
      Tune::ZEN5,
      Tune::ZEN5C,
      Tune::INTEL_SPR,
      Tune::INTEL_GNR,
      Tune::INTEL_ICL,
      Tune::APPLE_M1_M3,
      Tune::APPLE_M4,
      Tune::APPLE_M5,
      Tune::GRAVITON2,
      Tune::GRAVITON3,
      Tune::GRAVITON4,
      Tune::GRAVITON5,
      Tune::NEOVERSE_N2,
      Tune::NEOVERSE_N3,
      Tune::NEOVERSE_V3,
      Tune::NVIDIA_GRACE,
      Tune::AMPERE_ALTRA,
      Tune::AARCH64_PMULL,
      Tune::Z13,
      Tune::Z14,
      Tune::Z15,
      Tune::POWER7,
      Tune::POWER8,
      Tune::POWER9,
      Tune::POWER10,
      Tune::PORTABLE,
    ];

    for tune in presets {
      // All thresholds should be positive
      assert!(tune.simd_threshold > 0, "simd_threshold should be > 0");
      // Parallel streams should be at least 1
      assert!(tune.parallel_streams >= 1, "parallel_streams should be >= 1");
      // Threshold should be a power of 2 or common value
      assert!(
        tune.simd_threshold == 64 || tune.simd_threshold == 128 || tune.simd_threshold == 256,
        "unexpected simd_threshold: {}",
        tune.simd_threshold
      );
      // Effective SIMD width should be 0, 128, 256, or 512
      assert!(
        tune.effective_simd_width == 0
          || tune.effective_simd_width == 128
          || tune.effective_simd_width == 256
          || tune.effective_simd_width == 512,
        "unexpected effective_simd_width: {}",
        tune.effective_simd_width
      );
      // SVE vector length should be 0, 128, 256, or 512
      assert!(
        tune.sve_vlen == 0 || tune.sve_vlen == 128 || tune.sve_vlen == 256 || tune.sve_vlen == 512,
        "unexpected sve_vlen: {}",
        tune.sve_vlen
      );
      // Cache line should be 64 or 128
      assert!(
        tune.cache_line == 64 || tune.cache_line == 128,
        "unexpected cache_line: {}",
        tune.cache_line
      );
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Equality and Clone
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_tune_equality() {
    assert_eq!(Tune::ZEN5, Tune::ZEN5);
    assert_ne!(Tune::ZEN5, Tune::ZEN4);
    assert_ne!(Tune::INTEL_SPR, Tune::APPLE_M);
    assert_ne!(Tune::GRAVITON3, Tune::GRAVITON4); // Different SVE vector lengths
    assert_ne!(Tune::APPLE_M1_M3, Tune::APPLE_M4); // Different SME support
  }

  #[test]
  fn test_tune_clone() {
    let tune = Tune::ZEN5;
    let cloned = tune;
    assert_eq!(tune, cloned);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // SVE and SME Field Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_sve_vlen_differentiation() {
    // Graviton 3 and 4 differ only by SVE vector length
    assert_eq!(Tune::GRAVITON3.sve_vlen, 256);
    assert_eq!(Tune::GRAVITON4.sve_vlen, 128);
    assert_ne!(Tune::GRAVITON3, Tune::GRAVITON4);
  }

  #[test]
  fn test_sme_differentiation() {
    // Apple M1-M3 and M4 differ by SME support
    // Use const blocks for compile-time assertions on const values
    const { assert!(Tune::APPLE_M1_M3.sme_tile == 0) };
    const { assert!(Tune::APPLE_M4.sme_tile == 256) };
    assert_ne!(Tune::APPLE_M1_M3, Tune::APPLE_M4);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // New Field Tests
  // ─────────────────────────────────────────────────────────────────────────────

  #[test]
  fn test_effective_simd_width() {
    // Zen 4/5 use full 512-bit
    assert_eq!(Tune::ZEN4.effective_simd_width, 512);
    assert_eq!(Tune::ZEN5.effective_simd_width, 512);

    // Intel SPR can use full 512-bit
    assert_eq!(Tune::INTEL_SPR.effective_simd_width, 512);

    // Intel ICL prefers 256-bit despite having AVX-512
    assert_eq!(Tune::INTEL_ICL.effective_simd_width, 256);

    // AArch64 uses 128-bit NEON
    assert_eq!(Tune::APPLE_M1_M3.effective_simd_width, 128);
    assert_eq!(Tune::GRAVITON2.effective_simd_width, 128);

    // Graviton 3 uses 256-bit SVE
    assert_eq!(Tune::GRAVITON3.effective_simd_width, 256);

    // Portable has no SIMD
    assert_eq!(Tune::PORTABLE.effective_simd_width, 0);
  }

  #[test]
  fn test_cache_line_sizes() {
    // All current presets use 64-byte cache lines
    assert_eq!(Tune::ZEN5.cache_line, 64);
    assert_eq!(Tune::INTEL_SPR.cache_line, 64);
    assert_eq!(Tune::APPLE_M1_M3.cache_line, 64);
    assert_eq!(Tune::GRAVITON3.cache_line, 64);
  }

  #[test]
  fn test_prefetch_distance_default() {
    // All current presets let hardware decide prefetch
    assert_eq!(Tune::ZEN5.prefetch_distance, 0);
    assert_eq!(Tune::INTEL_SPR.prefetch_distance, 0);
    assert_eq!(Tune::GRAVITON3.prefetch_distance, 0);
  }
}
