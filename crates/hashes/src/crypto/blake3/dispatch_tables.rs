//! Tuned dispatch tables for BLAKE3.
//!
//! This module is the integration point for offline dispatch-table generation tooling.
//! The tuning engine updates compact per-family profiles in this file.

use platform::TuneKind;

pub use super::kernels::Blake3KernelId as KernelId;

pub const DEFAULT_BOUNDARIES: [usize; 3] = [64, 256, 4096];

// Architecture-specific thresholds for when to switch from table bulk kernel to size-class
// selection. These are tuned based on SIMD width and latency characteristics.
#[cfg(target_arch = "x86_64")]
const THRESHOLD_AVX512: usize = 4 * 1024; // AVX-512 is very fast, switch early
#[cfg(target_arch = "x86_64")]
const THRESHOLD_AVX2: usize = 8 * 1024; // AVX2 - used for fallback/default x86_64 profile
#[cfg(target_arch = "aarch64")]
const THRESHOLD_NEON: usize = 16 * 1024; // NEON - slightly higher due to different characteristics
const THRESHOLD_PORTABLE: usize = 32 * 1024; // Conservative for scalar

const DEFAULT_PAR_SPAWN_COST_BYTES: usize = 24 * 1024;
const DEFAULT_PAR_MERGE_COST_BYTES: usize = 16 * 1024;
const DEFAULT_PAR_BYTES_PER_CORE_SMALL: usize = 256 * 1024;
const DEFAULT_PAR_BYTES_PER_CORE_MEDIUM: usize = 128 * 1024;
const DEFAULT_PAR_BYTES_PER_CORE_LARGE: usize = 64 * 1024;
const DEFAULT_PAR_SMALL_LIMIT_BYTES: usize = 256 * 1024;
const DEFAULT_PAR_MEDIUM_LIMIT_BYTES: usize = 2 * 1024 * 1024;

/// Parallel hashing policy for large inputs (std-only).
///
/// The scheduler treats this as an explicit cost model:
/// - fixed terms: `spawn_cost_bytes`, `merge_cost_bytes`
/// - work terms: `bytes_per_core_*` for small/medium/large payload classes
#[derive(Clone, Copy, Debug)]
pub struct ParallelTable {
  /// Minimum total input bytes before parallel hashing is considered.
  pub min_bytes: usize,
  /// Minimum number of full chunks to commit in one batch.
  pub min_chunks: usize,
  /// Maximum total threads to use (including the caller thread).
  ///
  /// - `0` means "no explicit cap" (use all available hardware threads).
  /// - `1` disables parallel hashing.
  pub max_threads: u8,
  /// Per-worker scheduling overhead, represented as equivalent input bytes.
  pub spawn_cost_bytes: usize,
  /// Fixed reduction/finalization overhead, represented as equivalent bytes.
  pub merge_cost_bytes: usize,
  /// Minimum effective bytes/core for small payloads.
  pub bytes_per_core_small: usize,
  /// Minimum effective bytes/core for medium payloads.
  pub bytes_per_core_medium: usize,
  /// Minimum effective bytes/core for large payloads.
  pub bytes_per_core_large: usize,
  /// Upper bound (inclusive) for the small payload class.
  pub small_limit_bytes: usize,
  /// Upper bound (inclusive) for the medium payload class.
  pub medium_limit_bytes: usize,
}

/// Streaming dispatch preferences.
///
/// `stream` is used for the per-block / ChunkState hot path (many small updates).
/// `bulk` is used for multi-chunk hashing (`hash_many_contiguous`) and batched parent reductions.
/// `bulk_sizeclass_threshold` is the minimum input length to use size-class-based bulk kernel
/// selection instead of the table's default bulk kernel. This is architecture-specific.
#[derive(Clone, Copy, Debug)]
pub struct StreamingTable {
  pub stream: KernelId,
  pub bulk: KernelId,
  pub bulk_sizeclass_threshold: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct DispatchTable {
  pub boundaries: [usize; 3],
  pub xs: KernelId,
  pub s: KernelId,
  pub m: KernelId,
  pub l: KernelId,
}

impl DispatchTable {
  #[inline]
  #[must_use]
  pub const fn kernel_for_len(&self, len: usize) -> KernelId {
    let [xs_max, s_max, m_max] = self.boundaries;
    if len <= xs_max {
      self.xs
    } else if len <= s_max {
      self.s
    } else if len <= m_max {
      self.m
    } else {
      self.l
    }
  }
}

/// Compact profile for one microarchitecture family.
#[derive(Clone, Copy, Debug)]
pub struct FamilyProfile {
  pub dispatch: DispatchTable,
  pub streaming: StreamingTable,
  pub parallel: ParallelTable,
  pub streaming_parallel: ParallelTable,
}

// A conservative "best available" SIMD kernel per target architecture.
//
// The runtime dispatcher (`dispatch::resolve`) will still validate CPU feature
// availability and fall back to Portable when needed.
#[cfg(target_arch = "x86_64")]
const SIMD_KERNEL: KernelId = KernelId::X86Avx2;
#[cfg(target_arch = "aarch64")]
const SIMD_KERNEL: KernelId = KernelId::Aarch64Neon;
#[cfg(target_arch = "s390x")]
const SIMD_KERNEL: KernelId = KernelId::S390xVector;
#[cfg(target_arch = "powerpc64")]
const SIMD_KERNEL: KernelId = KernelId::PowerVsx;
#[cfg(target_arch = "riscv64")]
const SIMD_KERNEL: KernelId = KernelId::RiscvV;
#[cfg(not(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "s390x",
  target_arch = "powerpc64",
  target_arch = "riscv64"
)))]
const SIMD_KERNEL: KernelId = KernelId::Portable;

#[cfg(target_arch = "x86_64")]
const AVX512_KERNEL: KernelId = KernelId::X86Avx512;

#[cfg(target_arch = "s390x")]
const S390X_VECTOR_KERNEL: KernelId = KernelId::S390xVector;
#[cfg(not(target_arch = "s390x"))]
const S390X_VECTOR_KERNEL: KernelId = KernelId::Portable;

#[cfg(target_arch = "powerpc64")]
const POWER_VSX_KERNEL: KernelId = KernelId::PowerVsx;
#[cfg(not(target_arch = "powerpc64"))]
const POWER_VSX_KERNEL: KernelId = KernelId::Portable;

#[cfg(target_arch = "x86_64")]
const DEFAULT_XS: KernelId = KernelId::X86Sse41;
#[cfg(target_arch = "aarch64")]
const DEFAULT_XS: KernelId = KernelId::Aarch64Neon;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const DEFAULT_XS: KernelId = KernelId::Portable;

#[cfg(target_arch = "x86_64")]
const DEFAULT_S: KernelId = KernelId::X86Sse41;
#[cfg(target_arch = "aarch64")]
const DEFAULT_S: KernelId = KernelId::Aarch64Neon;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const DEFAULT_S: KernelId = KernelId::Portable;

const DEFAULT_M: KernelId = SIMD_KERNEL;
const DEFAULT_L: KernelId = SIMD_KERNEL;

#[cfg(target_arch = "x86_64")]
const DEFAULT_STREAM_KERNEL: KernelId = KernelId::X86Sse41;
#[cfg(target_arch = "aarch64")]
const DEFAULT_STREAM_KERNEL: KernelId = KernelId::Aarch64Neon;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const DEFAULT_STREAM_KERNEL: KernelId = KernelId::Portable;

const DEFAULT_BULK_KERNEL: KernelId = SIMD_KERNEL;

#[derive(Clone, Copy, Debug)]
struct ParallelCostModel {
  spawn_cost_bytes: usize,
  merge_cost_bytes: usize,
  bytes_per_core_small: usize,
  bytes_per_core_medium: usize,
  bytes_per_core_large: usize,
  small_limit_bytes: usize,
  medium_limit_bytes: usize,
}

#[inline]
#[must_use]
const fn parallel_cost_model(
  spawn_cost_bytes: usize,
  merge_cost_bytes: usize,
  bytes_per_core_small: usize,
  bytes_per_core_medium: usize,
  bytes_per_core_large: usize,
  small_limit_bytes: usize,
  medium_limit_bytes: usize,
) -> ParallelCostModel {
  ParallelCostModel {
    spawn_cost_bytes,
    merge_cost_bytes,
    bytes_per_core_small,
    bytes_per_core_medium,
    bytes_per_core_large,
    small_limit_bytes,
    medium_limit_bytes,
  }
}

#[inline]
#[must_use]
const fn parallel_table(
  min_bytes: usize,
  min_chunks: usize,
  max_threads: u8,
  cost: ParallelCostModel,
) -> ParallelTable {
  ParallelTable {
    min_bytes,
    min_chunks,
    max_threads,
    spawn_cost_bytes: cost.spawn_cost_bytes,
    merge_cost_bytes: cost.merge_cost_bytes,
    bytes_per_core_small: cost.bytes_per_core_small,
    bytes_per_core_medium: cost.bytes_per_core_medium,
    bytes_per_core_large: cost.bytes_per_core_large,
    small_limit_bytes: cost.small_limit_bytes,
    medium_limit_bytes: cost.medium_limit_bytes,
  }
}

macro_rules! parallel_costs {
  (
    $min_bytes:expr,
    $min_chunks:expr,
    $max_threads:expr,
    $spawn_cost_bytes:expr,
    $merge_cost_bytes:expr,
    $bytes_per_core_small:expr,
    $bytes_per_core_medium:expr,
    $bytes_per_core_large:expr,
    $small_limit_bytes:expr,
    $medium_limit_bytes:expr $(,)?
  ) => {
    parallel_table(
      $min_bytes,
      $min_chunks,
      $max_threads,
      parallel_cost_model(
        $spawn_cost_bytes,
        $merge_cost_bytes,
        $bytes_per_core_small,
        $bytes_per_core_medium,
        $bytes_per_core_large,
        $small_limit_bytes,
        $medium_limit_bytes,
      ),
    )
  };
}

#[inline]
#[must_use]
const fn default_parallel_costs(min_bytes: usize, min_chunks: usize, max_threads: u8) -> ParallelTable {
  parallel_table(
    min_bytes,
    min_chunks,
    max_threads,
    parallel_cost_model(
      DEFAULT_PAR_SPAWN_COST_BYTES,
      DEFAULT_PAR_MERGE_COST_BYTES,
      DEFAULT_PAR_BYTES_PER_CORE_SMALL,
      DEFAULT_PAR_BYTES_PER_CORE_MEDIUM,
      DEFAULT_PAR_BYTES_PER_CORE_LARGE,
      DEFAULT_PAR_SMALL_LIMIT_BYTES,
      DEFAULT_PAR_MEDIUM_LIMIT_BYTES,
    ),
  )
}

#[inline]
#[must_use]
const fn scalar_profile_parallel(
  min_bytes: usize,
  min_chunks: usize,
  max_threads: u8,
  generation: u8,
) -> ParallelTable {
  match generation {
    0 => parallel_costs!(
      min_bytes,
      min_chunks,
      max_threads,
      64 * 1024,
      48 * 1024,
      384 * 1024,
      256 * 1024,
      192 * 1024,
      512 * 1024,
      4 * 1024 * 1024,
    ),
    1 => parallel_costs!(
      min_bytes,
      min_chunks,
      max_threads,
      56 * 1024,
      40 * 1024,
      320 * 1024,
      224 * 1024,
      160 * 1024,
      384 * 1024,
      3 * 1024 * 1024,
    ),
    2 => parallel_costs!(
      min_bytes,
      min_chunks,
      max_threads,
      48 * 1024,
      32 * 1024,
      256 * 1024,
      192 * 1024,
      128 * 1024,
      320 * 1024,
      3 * 1024 * 1024,
    ),
    3 => parallel_costs!(
      min_bytes,
      min_chunks,
      max_threads,
      40 * 1024,
      28 * 1024,
      256 * 1024,
      160 * 1024,
      96 * 1024,
      256 * 1024,
      2 * 1024 * 1024,
    ),
    _ => parallel_costs!(
      min_bytes,
      min_chunks,
      max_threads,
      32 * 1024,
      24 * 1024,
      224 * 1024,
      128 * 1024,
      80 * 1024,
      256 * 1024,
      2 * 1024 * 1024,
    ),
  }
}

#[inline]
#[must_use]
const fn default_kind_table() -> DispatchTable {
  DispatchTable {
    boundaries: DEFAULT_BOUNDARIES,
    xs: DEFAULT_XS,
    s: DEFAULT_S,
    m: DEFAULT_M,
    l: DEFAULT_L,
  }
}

// Default threshold matches the default SIMD tier for each architecture.
#[cfg(target_arch = "x86_64")]
const DEFAULT_BULK_THRESHOLD: usize = THRESHOLD_AVX2;
#[cfg(target_arch = "aarch64")]
const DEFAULT_BULK_THRESHOLD: usize = THRESHOLD_NEON;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const DEFAULT_BULK_THRESHOLD: usize = THRESHOLD_PORTABLE;

#[inline]
#[must_use]
const fn default_kind_streaming_table() -> StreamingTable {
  StreamingTable {
    stream: DEFAULT_STREAM_KERNEL,
    bulk: DEFAULT_BULK_KERNEL,
    bulk_sizeclass_threshold: DEFAULT_BULK_THRESHOLD,
  }
}

#[inline]
#[must_use]
const fn default_kind_parallel_table() -> ParallelTable {
  default_parallel_costs(128 * 1024, 64, 0)
}

#[inline]
#[must_use]
const fn default_kind_streaming_parallel_table() -> ParallelTable {
  default_kind_parallel_table()
}

#[inline]
#[must_use]
const fn default_kind_profile() -> FamilyProfile {
  FamilyProfile {
    dispatch: default_kind_table(),
    streaming: default_kind_streaming_table(),
    parallel: default_kind_parallel_table(),
    streaming_parallel: default_kind_streaming_parallel_table(),
  }
}

#[inline]
#[must_use]
const fn portable_profile() -> FamilyProfile {
  FamilyProfile {
    dispatch: DispatchTable {
      boundaries: DEFAULT_BOUNDARIES,
      xs: KernelId::Portable,
      s: KernelId::Portable,
      m: KernelId::Portable,
      l: KernelId::Portable,
    },
    streaming: StreamingTable {
      stream: KernelId::Portable,
      bulk: KernelId::Portable,
      bulk_sizeclass_threshold: THRESHOLD_PORTABLE,
    },
    parallel: default_parallel_costs(128 * 1024, 64, 0),
    streaming_parallel: default_parallel_costs(128 * 1024, 64, 0),
  }
}

pub static DEFAULT_TABLE: DispatchTable = portable_profile().dispatch;
pub static DEFAULT_STREAMING_TABLE: StreamingTable = portable_profile().streaming;
pub static DEFAULT_PARALLEL_TABLE: ParallelTable = portable_profile().parallel;
pub static DEFAULT_STREAMING_PARALLEL_TABLE: ParallelTable = portable_profile().streaming_parallel;

// Family Profile: CUSTOM
pub static PROFILE_CUSTOM: FamilyProfile = portable_profile();
// Family Profile: DEFAULT_KIND
pub static PROFILE_DEFAULT_KIND: FamilyProfile = default_kind_profile();
// Family Profile: PORTABLE
pub static PROFILE_PORTABLE: FamilyProfile = portable_profile();

// Family Profile: X86_ZEN4
#[cfg(target_arch = "x86_64")]
pub static PROFILE_X86_ZEN4: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: [64, 1024, 4096],
    xs: KernelId::X86Avx512,
    s: KernelId::X86Avx2,
    m: KernelId::X86Avx512,
    l: KernelId::X86Avx512,
  },
  streaming: StreamingTable {
    stream: KernelId::X86Sse41,
    bulk: KernelId::X86Avx512,
    bulk_sizeclass_threshold: THRESHOLD_AVX512,
  },
  parallel: ParallelTable {
    min_bytes: 65536,
    min_chunks: 64,
    max_threads: 8,
    spawn_cost_bytes: 24576,
    merge_cost_bytes: 16384,
    bytes_per_core_small: 24576,
    bytes_per_core_medium: 107520,
    bytes_per_core_large: 1025024,
    small_limit_bytes: 262144,
    medium_limit_bytes: 2097152,
  },
  streaming_parallel: ParallelTable {
    min_bytes: 0,
    min_chunks: 0,
    max_threads: 1,
    spawn_cost_bytes: 24576,
    merge_cost_bytes: 16384,
    bytes_per_core_small: 262144,
    bytes_per_core_medium: 131072,
    bytes_per_core_large: 65536,
    small_limit_bytes: 262144,
    medium_limit_bytes: 2097152,
  },
};
#[cfg(not(target_arch = "x86_64"))]
pub static PROFILE_X86_ZEN4: FamilyProfile = default_kind_profile();
// Family Profile: X86_ZEN5
#[cfg(target_arch = "x86_64")]
pub static PROFILE_X86_ZEN5: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: [64, 1024, 4096],
    xs: KernelId::X86Sse41,
    s: KernelId::X86Avx512,
    m: KernelId::X86Avx512,
    l: KernelId::X86Avx512,
  },
  streaming: StreamingTable {
    stream: KernelId::X86Sse41,
    bulk: KernelId::X86Avx512,
    bulk_sizeclass_threshold: THRESHOLD_AVX512,
  },
  parallel: ParallelTable {
    min_bytes: 65536,
    min_chunks: 64,
    max_threads: 8,
    spawn_cost_bytes: 24576,
    merge_cost_bytes: 16384,
    bytes_per_core_small: 24576,
    bytes_per_core_medium: 107520,
    bytes_per_core_large: 1025024,
    small_limit_bytes: 262144,
    medium_limit_bytes: 2097152,
  },
  streaming_parallel: ParallelTable {
    min_bytes: 0,
    min_chunks: 0,
    max_threads: 1,
    spawn_cost_bytes: 24576,
    merge_cost_bytes: 16384,
    bytes_per_core_small: 262144,
    bytes_per_core_medium: 131072,
    bytes_per_core_large: 65536,
    small_limit_bytes: 262144,
    medium_limit_bytes: 2097152,
  },
};
#[cfg(not(target_arch = "x86_64"))]
pub static PROFILE_X86_ZEN5: FamilyProfile = default_kind_profile();
// Family Profile: X86_ZEN5C
#[cfg(target_arch = "x86_64")]
pub static PROFILE_X86_ZEN5C: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: DEFAULT_BOUNDARIES,
    xs: DEFAULT_XS,
    s: DEFAULT_S,
    m: SIMD_KERNEL,
    l: AVX512_KERNEL,
  },
  streaming: StreamingTable {
    stream: DEFAULT_STREAM_KERNEL,
    bulk: AVX512_KERNEL,
    bulk_sizeclass_threshold: THRESHOLD_AVX512,
  },
  parallel: parallel_costs!(
    80 * 1024,
    40,
    0,
    18 * 1024,
    12 * 1024,
    176 * 1024,
    104 * 1024,
    52 * 1024,
    256 * 1024,
    2 * 1024 * 1024,
  ),
  streaming_parallel: parallel_costs!(
    80 * 1024,
    40,
    0,
    18 * 1024,
    12 * 1024,
    176 * 1024,
    104 * 1024,
    52 * 1024,
    256 * 1024,
    2 * 1024 * 1024,
  ),
};
#[cfg(not(target_arch = "x86_64"))]
pub static PROFILE_X86_ZEN5C: FamilyProfile = default_kind_profile();

// Family Profile: X86_INTEL_SPR
#[cfg(target_arch = "x86_64")]
pub static PROFILE_X86_INTEL_SPR: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: [64, 1024, 4096],
    xs: KernelId::X86Avx512,
    s: KernelId::X86Avx512,
    m: KernelId::X86Avx512,
    l: KernelId::X86Avx512,
  },
  streaming: StreamingTable {
    stream: KernelId::X86Sse41,
    bulk: KernelId::X86Avx512,
    bulk_sizeclass_threshold: THRESHOLD_AVX512,
  },
  parallel: ParallelTable {
    min_bytes: 65536,
    min_chunks: 64,
    max_threads: 8,
    spawn_cost_bytes: 24576,
    merge_cost_bytes: 131072,
    bytes_per_core_small: 32768,
    bytes_per_core_medium: 224256,
    bytes_per_core_large: 1010688,
    small_limit_bytes: 1048576,
    medium_limit_bytes: 4194304,
  },
  streaming_parallel: ParallelTable {
    min_bytes: 0,
    min_chunks: 0,
    max_threads: 1,
    spawn_cost_bytes: 24576,
    merge_cost_bytes: 16384,
    bytes_per_core_small: 262144,
    bytes_per_core_medium: 131072,
    bytes_per_core_large: 65536,
    small_limit_bytes: 262144,
    medium_limit_bytes: 2097152,
  },
};
#[cfg(not(target_arch = "x86_64"))]
pub static PROFILE_X86_INTEL_SPR: FamilyProfile = default_kind_profile();
// Family Profile: X86_INTEL_GNR
#[cfg(target_arch = "x86_64")]
pub static PROFILE_X86_INTEL_GNR: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: DEFAULT_BOUNDARIES,
    xs: DEFAULT_XS,
    s: DEFAULT_S,
    m: AVX512_KERNEL,
    l: AVX512_KERNEL,
  },
  streaming: StreamingTable {
    stream: DEFAULT_STREAM_KERNEL,
    bulk: AVX512_KERNEL,
    bulk_sizeclass_threshold: THRESHOLD_AVX512,
  },
  parallel: parallel_costs!(
    96 * 1024,
    48,
    0,
    20 * 1024,
    14 * 1024,
    208 * 1024,
    120 * 1024,
    60 * 1024,
    256 * 1024,
    2 * 1024 * 1024,
  ),
  streaming_parallel: parallel_costs!(
    96 * 1024,
    48,
    0,
    20 * 1024,
    14 * 1024,
    208 * 1024,
    120 * 1024,
    60 * 1024,
    256 * 1024,
    2 * 1024 * 1024,
  ),
};
#[cfg(not(target_arch = "x86_64"))]
pub static PROFILE_X86_INTEL_GNR: FamilyProfile = default_kind_profile();

// Family Profile: X86_INTEL_ICL
#[cfg(target_arch = "x86_64")]
pub static PROFILE_X86_INTEL_ICL: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: [64, 1024, 4096],
    xs: KernelId::X86Avx512,
    s: KernelId::X86Avx512,
    m: KernelId::X86Avx512,
    l: KernelId::X86Avx512,
  },
  streaming: StreamingTable {
    stream: KernelId::X86Sse41,
    bulk: KernelId::X86Avx512,
    bulk_sizeclass_threshold: THRESHOLD_AVX512,
  },
  parallel: ParallelTable {
    min_bytes: 65536,
    min_chunks: 64,
    max_threads: 8,
    spawn_cost_bytes: 24576,
    merge_cost_bytes: 16384,
    bytes_per_core_small: 24576,
    bytes_per_core_medium: 107520,
    bytes_per_core_large: 1025024,
    small_limit_bytes: 262144,
    medium_limit_bytes: 2097152,
  },
  streaming_parallel: ParallelTable {
    min_bytes: 0,
    min_chunks: 0,
    max_threads: 1,
    spawn_cost_bytes: 24576,
    merge_cost_bytes: 16384,
    bytes_per_core_small: 262144,
    bytes_per_core_medium: 131072,
    bytes_per_core_large: 65536,
    small_limit_bytes: 262144,
    medium_limit_bytes: 2097152,
  },
};
#[cfg(not(target_arch = "x86_64"))]
pub static PROFILE_X86_INTEL_ICL: FamilyProfile = default_kind_profile();
// Family Profile: AARCH64_APPLE_M1M3
#[cfg(target_arch = "aarch64")]
pub static PROFILE_AARCH64_APPLE_M1M3: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: [64, 4095, 4096],
    xs: KernelId::Portable,
    s: KernelId::Portable,
    m: KernelId::Aarch64Neon,
    l: KernelId::Aarch64Neon,
  },
  streaming: StreamingTable {
    stream: KernelId::Portable,
    bulk: KernelId::Aarch64Neon,
    bulk_sizeclass_threshold: THRESHOLD_NEON,
  },
  parallel: ParallelTable {
    min_bytes: 98304,
    min_chunks: 96,
    max_threads: 12,
    spawn_cost_bytes: 24576,
    merge_cost_bytes: 16384,
    bytes_per_core_small: 19660,
    bytes_per_core_medium: 107520,
    bytes_per_core_large: 1025024,
    small_limit_bytes: 262144,
    medium_limit_bytes: 2097152,
  },
  streaming_parallel: ParallelTable {
    min_bytes: 0,
    min_chunks: 0,
    max_threads: 1,
    spawn_cost_bytes: 24576,
    merge_cost_bytes: 16384,
    bytes_per_core_small: 262144,
    bytes_per_core_medium: 131072,
    bytes_per_core_large: 65536,
    small_limit_bytes: 262144,
    medium_limit_bytes: 2097152,
  },
};
#[cfg(not(target_arch = "aarch64"))]
pub static PROFILE_AARCH64_APPLE_M1M3: FamilyProfile = default_kind_profile();
// Family Profile: AARCH64_APPLE_M4
#[cfg(target_arch = "aarch64")]
pub static PROFILE_AARCH64_APPLE_M4: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: DEFAULT_BOUNDARIES,
    xs: KernelId::Portable,
    s: KernelId::Portable,
    m: KernelId::Aarch64Neon,
    l: KernelId::Aarch64Neon,
  },
  streaming: StreamingTable {
    stream: KernelId::Portable,
    bulk: KernelId::Aarch64Neon,
    bulk_sizeclass_threshold: THRESHOLD_NEON,
  },
  parallel: parallel_costs!(
    64 * 1024,
    32,
    12,
    8 * 1024,
    8 * 1024,
    96 * 1024,
    80 * 1024,
    48 * 1024,
    256 * 1024,
    2 * 1024 * 1024,
  ),
  streaming_parallel: parallel_costs!(
    64 * 1024,
    32,
    12,
    8 * 1024,
    8 * 1024,
    96 * 1024,
    80 * 1024,
    48 * 1024,
    256 * 1024,
    2 * 1024 * 1024,
  ),
};
#[cfg(not(target_arch = "aarch64"))]
pub static PROFILE_AARCH64_APPLE_M4: FamilyProfile = default_kind_profile();

// Family Profile: AARCH64_APPLE_M5
#[cfg(target_arch = "aarch64")]
pub static PROFILE_AARCH64_APPLE_M5: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: DEFAULT_BOUNDARIES,
    xs: KernelId::Portable,
    s: KernelId::Portable,
    m: KernelId::Aarch64Neon,
    l: KernelId::Aarch64Neon,
  },
  streaming: StreamingTable {
    stream: KernelId::Portable,
    bulk: KernelId::Aarch64Neon,
    bulk_sizeclass_threshold: THRESHOLD_NEON,
  },
  parallel: parallel_costs!(
    64 * 1024,
    32,
    16,
    8 * 1024,
    8 * 1024,
    96 * 1024,
    72 * 1024,
    40 * 1024,
    256 * 1024,
    2 * 1024 * 1024,
  ),
  streaming_parallel: parallel_costs!(
    64 * 1024,
    32,
    16,
    8 * 1024,
    8 * 1024,
    96 * 1024,
    72 * 1024,
    40 * 1024,
    256 * 1024,
    2 * 1024 * 1024,
  ),
};
#[cfg(not(target_arch = "aarch64"))]
pub static PROFILE_AARCH64_APPLE_M5: FamilyProfile = default_kind_profile();

// Family Profile: AARCH64_GRAVITON2
#[cfg(target_arch = "aarch64")]
pub static PROFILE_AARCH64_GRAVITON2: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: DEFAULT_BOUNDARIES,
    xs: KernelId::Aarch64Neon,
    s: KernelId::Aarch64Neon,
    m: KernelId::Aarch64Neon,
    l: KernelId::Aarch64Neon,
  },
  streaming: StreamingTable {
    stream: KernelId::Portable,
    bulk: KernelId::Aarch64Neon,
    bulk_sizeclass_threshold: THRESHOLD_NEON,
  },
  parallel: parallel_costs!(
    128 * 1024,
    64,
    16,
    24 * 1024,
    16 * 1024,
    256 * 1024,
    160 * 1024,
    96 * 1024,
    256 * 1024,
    2 * 1024 * 1024,
  ),
  streaming_parallel: parallel_costs!(
    128 * 1024,
    64,
    16,
    24 * 1024,
    16 * 1024,
    256 * 1024,
    160 * 1024,
    96 * 1024,
    256 * 1024,
    2 * 1024 * 1024,
  ),
};
#[cfg(not(target_arch = "aarch64"))]
pub static PROFILE_AARCH64_GRAVITON2: FamilyProfile = default_kind_profile();

// Family Profile: AARCH64_SERVER_NEON
#[cfg(target_arch = "aarch64")]
pub static PROFILE_AARCH64_SERVER_NEON: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: [64, 4095, 4096],
    // Graviton-class attribution shows that for short inputs (<4KiB), the
    // portable path has lower latency than the NEON one-shot path.
    xs: KernelId::Portable,
    s: KernelId::Portable,
    m: KernelId::Aarch64Neon,
    l: KernelId::Aarch64Neon,
  },
  streaming: StreamingTable {
    // Match the short-input policy used by oneshot dispatch: stay portable for
    // stream setup/finalize-sensitive small updates and switch bulk to NEON.
    stream: KernelId::Portable,
    bulk: KernelId::Aarch64Neon,
    bulk_sizeclass_threshold: THRESHOLD_NEON,
  },
  parallel: ParallelTable {
    min_bytes: 65536,
    min_chunks: 64,
    max_threads: 8,
    spawn_cost_bytes: 24576,
    merge_cost_bytes: 16384,
    bytes_per_core_small: 16384,
    bytes_per_core_medium: 107520,
    bytes_per_core_large: 1025024,
    small_limit_bytes: 262144,
    medium_limit_bytes: 2097152,
  },
  streaming_parallel: ParallelTable {
    min_bytes: 0,
    min_chunks: 0,
    max_threads: 1,
    spawn_cost_bytes: 24576,
    merge_cost_bytes: 16384,
    bytes_per_core_small: 262144,
    bytes_per_core_medium: 131072,
    bytes_per_core_large: 65536,
    small_limit_bytes: 262144,
    medium_limit_bytes: 2097152,
  },
};
#[cfg(not(target_arch = "aarch64"))]
pub static PROFILE_AARCH64_SERVER_NEON: FamilyProfile = default_kind_profile();
// Family Profile: Z13
pub static PROFILE_Z13: FamilyProfile = FamilyProfile {
  dispatch: default_kind_table(),
  streaming: StreamingTable {
    stream: KernelId::Portable,
    bulk: S390X_VECTOR_KERNEL,
    bulk_sizeclass_threshold: THRESHOLD_PORTABLE,
  },
  parallel: scalar_profile_parallel(256 * 1024, 128, 8, 0),
  streaming_parallel: scalar_profile_parallel(256 * 1024, 128, 8, 0),
};

// Family Profile: Z14
pub static PROFILE_Z14: FamilyProfile = FamilyProfile {
  dispatch: default_kind_table(),
  streaming: StreamingTable {
    stream: KernelId::Portable,
    bulk: S390X_VECTOR_KERNEL,
    bulk_sizeclass_threshold: THRESHOLD_PORTABLE,
  },
  parallel: scalar_profile_parallel(192 * 1024, 96, 8, 1),
  streaming_parallel: scalar_profile_parallel(192 * 1024, 96, 8, 1),
};

// Family Profile: Z15
pub static PROFILE_Z15: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: [64, 256, 4096],
    xs: KernelId::Portable,
    s: KernelId::Portable,
    m: S390X_VECTOR_KERNEL,
    l: S390X_VECTOR_KERNEL,
  },
  streaming: StreamingTable {
    stream: KernelId::Portable,
    bulk: S390X_VECTOR_KERNEL,
    bulk_sizeclass_threshold: THRESHOLD_PORTABLE,
  },
  parallel: ParallelTable {
    min_bytes: 65536,
    min_chunks: 64,
    max_threads: 4,
    spawn_cost_bytes: 24576,
    merge_cost_bytes: 16384,
    bytes_per_core_small: 32768,
    bytes_per_core_medium: 370688,
    bytes_per_core_large: 65536,
    small_limit_bytes: 262144,
    medium_limit_bytes: 8388608,
  },
  streaming_parallel: ParallelTable {
    min_bytes: 0,
    min_chunks: 0,
    max_threads: 1,
    spawn_cost_bytes: 24576,
    merge_cost_bytes: 16384,
    bytes_per_core_small: 262144,
    bytes_per_core_medium: 131072,
    bytes_per_core_large: 65536,
    small_limit_bytes: 262144,
    medium_limit_bytes: 2097152,
  },
};
// Family Profile: POWER7
pub static PROFILE_POWER7: FamilyProfile = FamilyProfile {
  dispatch: default_kind_table(),
  streaming: StreamingTable {
    stream: KernelId::Portable,
    bulk: POWER_VSX_KERNEL,
    bulk_sizeclass_threshold: THRESHOLD_PORTABLE,
  },
  parallel: scalar_profile_parallel(256 * 1024, 128, 8, 0),
  streaming_parallel: scalar_profile_parallel(256 * 1024, 128, 8, 0),
};

// Family Profile: POWER8
pub static PROFILE_POWER8: FamilyProfile = FamilyProfile {
  dispatch: default_kind_table(),
  streaming: StreamingTable {
    stream: KernelId::Portable,
    bulk: POWER_VSX_KERNEL,
    bulk_sizeclass_threshold: THRESHOLD_PORTABLE,
  },
  parallel: scalar_profile_parallel(192 * 1024, 96, 8, 1),
  streaming_parallel: scalar_profile_parallel(192 * 1024, 96, 8, 1),
};

// Family Profile: POWER9
pub static PROFILE_POWER9: FamilyProfile = FamilyProfile {
  dispatch: default_kind_table(),
  streaming: StreamingTable {
    stream: KernelId::Portable,
    bulk: POWER_VSX_KERNEL,
    bulk_sizeclass_threshold: THRESHOLD_PORTABLE,
  },
  parallel: scalar_profile_parallel(128 * 1024, 64, 16, 3),
  streaming_parallel: scalar_profile_parallel(128 * 1024, 64, 16, 3),
};

// Family Profile: POWER10
pub static PROFILE_POWER10: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: [64, 256, 4096],
    xs: KernelId::Portable,
    s: POWER_VSX_KERNEL,
    m: POWER_VSX_KERNEL,
    l: POWER_VSX_KERNEL,
  },
  streaming: StreamingTable {
    stream: KernelId::Portable,
    bulk: POWER_VSX_KERNEL,
    bulk_sizeclass_threshold: THRESHOLD_PORTABLE,
  },
  parallel: ParallelTable {
    min_bytes: 65536,
    min_chunks: 64,
    max_threads: 4,
    spawn_cost_bytes: 24576,
    merge_cost_bytes: 16384,
    bytes_per_core_small: 32768,
    bytes_per_core_medium: 370688,
    bytes_per_core_large: 65536,
    small_limit_bytes: 262144,
    medium_limit_bytes: 8388608,
  },
  streaming_parallel: ParallelTable {
    min_bytes: 0,
    min_chunks: 0,
    max_threads: 1,
    spawn_cost_bytes: 24576,
    merge_cost_bytes: 16384,
    bytes_per_core_small: 262144,
    bytes_per_core_medium: 131072,
    bytes_per_core_large: 65536,
    small_limit_bytes: 262144,
    medium_limit_bytes: 2097152,
  },
};

#[inline]
#[must_use]
fn select_profile(kind: TuneKind) -> &'static FamilyProfile {
  match kind {
    TuneKind::Custom => &PROFILE_CUSTOM,
    TuneKind::Default => &PROFILE_DEFAULT_KIND,
    TuneKind::Portable => &PROFILE_PORTABLE,
    TuneKind::Zen4 => &PROFILE_X86_ZEN4,
    TuneKind::Zen5 => &PROFILE_X86_ZEN5,
    TuneKind::Zen5c => &PROFILE_X86_ZEN5C,
    TuneKind::IntelSpr => &PROFILE_X86_INTEL_SPR,
    TuneKind::IntelGnr => &PROFILE_X86_INTEL_GNR,
    TuneKind::IntelIcl => &PROFILE_X86_INTEL_ICL,
    TuneKind::AppleM1M3 => &PROFILE_AARCH64_APPLE_M1M3,
    TuneKind::AppleM4 => &PROFILE_AARCH64_APPLE_M4,
    TuneKind::AppleM5 => &PROFILE_AARCH64_APPLE_M5,
    TuneKind::Graviton2 => &PROFILE_AARCH64_GRAVITON2,
    TuneKind::Graviton3
    | TuneKind::Graviton4
    | TuneKind::Graviton5
    | TuneKind::NeoverseN2
    | TuneKind::NeoverseN3
    | TuneKind::NeoverseV3
    | TuneKind::NvidiaGrace
    | TuneKind::AmpereAltra
    | TuneKind::Aarch64Pmull => &PROFILE_AARCH64_SERVER_NEON,
    TuneKind::Z13 => &PROFILE_Z13,
    TuneKind::Z14 => &PROFILE_Z14,
    TuneKind::Z15 => &PROFILE_Z15,
    TuneKind::Power7 => &PROFILE_POWER7,
    TuneKind::Power8 => &PROFILE_POWER8,
    TuneKind::Power9 => &PROFILE_POWER9,
    TuneKind::Power10 => &PROFILE_POWER10,
  }
}

#[inline]
#[must_use]
pub fn select_table(kind: TuneKind) -> &'static DispatchTable {
  &select_profile(kind).dispatch
}

#[inline]
#[must_use]
pub fn select_streaming_table(kind: TuneKind) -> &'static StreamingTable {
  &select_profile(kind).streaming
}

#[inline]
#[must_use]
pub fn select_parallel_table(kind: TuneKind) -> &'static ParallelTable {
  &select_profile(kind).parallel
}

#[inline]
#[must_use]
pub fn select_streaming_parallel_table(kind: TuneKind) -> &'static ParallelTable {
  &select_profile(kind).streaming_parallel
}
