//! Tuned dispatch tables for BLAKE3.
//!
//! This module is the integration point for `rscrypto-tune --apply`.
//! The tuning engine updates compact per-family profiles in this file.

use platform::TuneKind;

pub use super::kernels::Blake3KernelId as KernelId;

pub const DEFAULT_BOUNDARIES: [usize; 3] = [64, 256, 4096];

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
#[derive(Clone, Copy, Debug)]
pub struct StreamingTable {
  pub stream: KernelId,
  pub bulk: KernelId,
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
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const SIMD_KERNEL: KernelId = KernelId::Portable;

#[cfg(target_arch = "x86_64")]
const AVX512_KERNEL: KernelId = KernelId::X86Avx512;

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

#[inline]
#[must_use]
const fn default_parallel_costs(min_bytes: usize, min_chunks: usize, max_threads: u8) -> ParallelTable {
  ParallelTable {
    min_bytes,
    min_chunks,
    max_threads,
    spawn_cost_bytes: DEFAULT_PAR_SPAWN_COST_BYTES,
    merge_cost_bytes: DEFAULT_PAR_MERGE_COST_BYTES,
    bytes_per_core_small: DEFAULT_PAR_BYTES_PER_CORE_SMALL,
    bytes_per_core_medium: DEFAULT_PAR_BYTES_PER_CORE_MEDIUM,
    bytes_per_core_large: DEFAULT_PAR_BYTES_PER_CORE_LARGE,
    small_limit_bytes: DEFAULT_PAR_SMALL_LIMIT_BYTES,
    medium_limit_bytes: DEFAULT_PAR_MEDIUM_LIMIT_BYTES,
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

#[inline]
#[must_use]
const fn default_kind_streaming_table() -> StreamingTable {
  StreamingTable {
    stream: DEFAULT_STREAM_KERNEL,
    bulk: DEFAULT_BULK_KERNEL,
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
    boundaries: DEFAULT_BOUNDARIES,
    xs: KernelId::X86Sse41,
    s: KernelId::X86Sse41,
    m: KernelId::X86Avx512,
    l: KernelId::X86Avx512,
  },
  streaming: StreamingTable {
    stream: KernelId::X86Sse41,
    bulk: KernelId::X86Avx2,
  },
  parallel: default_kind_parallel_table(),
  streaming_parallel: default_kind_streaming_parallel_table(),
};
#[cfg(not(target_arch = "x86_64"))]
pub static PROFILE_X86_ZEN4: FamilyProfile = default_kind_profile();

// Family Profile: X86_ZEN5
#[cfg(target_arch = "x86_64")]
pub static PROFILE_X86_ZEN5: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: DEFAULT_BOUNDARIES,
    xs: KernelId::X86Sse41,
    s: KernelId::X86Sse41,
    m: KernelId::X86Avx512,
    l: KernelId::X86Avx512,
  },
  streaming: StreamingTable {
    stream: KernelId::X86Sse41,
    bulk: KernelId::X86Avx512,
  },
  parallel: default_kind_parallel_table(),
  streaming_parallel: default_kind_streaming_parallel_table(),
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
  },
  parallel: default_kind_parallel_table(),
  streaming_parallel: default_kind_streaming_parallel_table(),
};
#[cfg(not(target_arch = "x86_64"))]
pub static PROFILE_X86_ZEN5C: FamilyProfile = default_kind_profile();

// Family Profile: X86_INTEL_SPR
#[cfg(target_arch = "x86_64")]
pub static PROFILE_X86_INTEL_SPR: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: DEFAULT_BOUNDARIES,
    xs: KernelId::X86Sse41,
    s: KernelId::X86Sse41,
    m: KernelId::X86Avx512,
    l: KernelId::X86Avx512,
  },
  streaming: StreamingTable {
    stream: KernelId::X86Sse41,
    bulk: KernelId::X86Avx512,
  },
  parallel: default_kind_parallel_table(),
  streaming_parallel: default_kind_streaming_parallel_table(),
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
  },
  parallel: default_kind_parallel_table(),
  streaming_parallel: default_kind_streaming_parallel_table(),
};
#[cfg(not(target_arch = "x86_64"))]
pub static PROFILE_X86_INTEL_GNR: FamilyProfile = default_kind_profile();

// Family Profile: X86_INTEL_ICL
#[cfg(target_arch = "x86_64")]
pub static PROFILE_X86_INTEL_ICL: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: DEFAULT_BOUNDARIES,
    xs: KernelId::X86Sse41,
    s: KernelId::X86Sse41,
    m: KernelId::X86Avx512,
    l: KernelId::X86Avx512,
  },
  streaming: StreamingTable {
    stream: KernelId::X86Sse41,
    bulk: KernelId::X86Avx2,
  },
  parallel: default_kind_parallel_table(),
  streaming_parallel: default_kind_streaming_parallel_table(),
};
#[cfg(not(target_arch = "x86_64"))]
pub static PROFILE_X86_INTEL_ICL: FamilyProfile = default_kind_profile();

// Family Profile: AARCH64_APPLE_M1M3
#[cfg(target_arch = "aarch64")]
pub static PROFILE_AARCH64_APPLE_M1M3: FamilyProfile = FamilyProfile {
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
  },
  parallel: default_parallel_costs(64 * 1024, 63, 8),
  streaming_parallel: default_parallel_costs(64 * 1024, 63, 8),
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
  },
  parallel: default_kind_parallel_table(),
  streaming_parallel: default_kind_streaming_parallel_table(),
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
  },
  parallel: default_kind_parallel_table(),
  streaming_parallel: default_kind_streaming_parallel_table(),
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
    stream: KernelId::Aarch64Neon,
    bulk: KernelId::Aarch64Neon,
  },
  parallel: default_kind_parallel_table(),
  streaming_parallel: default_kind_streaming_parallel_table(),
};
#[cfg(not(target_arch = "aarch64"))]
pub static PROFILE_AARCH64_GRAVITON2: FamilyProfile = default_kind_profile();

// Family Profile: AARCH64_SERVER_NEON
#[cfg(target_arch = "aarch64")]
pub static PROFILE_AARCH64_SERVER_NEON: FamilyProfile = FamilyProfile {
  dispatch: DispatchTable {
    boundaries: DEFAULT_BOUNDARIES,
    xs: KernelId::Aarch64Neon,
    s: KernelId::Aarch64Neon,
    m: KernelId::Aarch64Neon,
    l: KernelId::Aarch64Neon,
  },
  streaming: StreamingTable {
    stream: KernelId::Aarch64Neon,
    bulk: KernelId::Aarch64Neon,
  },
  parallel: default_kind_parallel_table(),
  streaming_parallel: default_kind_streaming_parallel_table(),
};
#[cfg(not(target_arch = "aarch64"))]
pub static PROFILE_AARCH64_SERVER_NEON: FamilyProfile = default_kind_profile();

// Family Profile: SCALAR_LEGACY
pub static PROFILE_SCALAR_LEGACY: FamilyProfile = FamilyProfile {
  dispatch: default_kind_table(),
  streaming: StreamingTable {
    stream: KernelId::Portable,
    bulk: KernelId::Portable,
  },
  parallel: default_kind_parallel_table(),
  streaming_parallel: default_kind_streaming_parallel_table(),
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
    TuneKind::Z13
    | TuneKind::Z14
    | TuneKind::Z15
    | TuneKind::Power7
    | TuneKind::Power8
    | TuneKind::Power9
    | TuneKind::Power10 => &PROFILE_SCALAR_LEGACY,
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
