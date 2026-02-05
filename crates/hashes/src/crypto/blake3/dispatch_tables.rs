//! Tuned dispatch tables for BLAKE3.
//!
//! This module is the integration point for `rscrypto-tune --apply`.
//! The tuning engine updates the per-`TuneKind` tables in this file.

use platform::TuneKind;

pub use super::kernels::Blake3KernelId as KernelId;

pub const DEFAULT_BOUNDARIES: [usize; 3] = [64, 256, 4096];

/// Parallel hashing policy for large inputs (std-only).
///
/// This controls when BLAKE3 will automatically use multi-core hashing and how
/// many threads it will use at most.
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

// AVX-512 is not always the best default on x86_64 (notably on some AMD CPUs),
// so we only opt into it for tune kinds where it is known to win.
#[cfg(target_arch = "x86_64")]
const AVX512_KERNEL: KernelId = KernelId::X86Avx512;

// Tiny/short inputs are frequently dominated by per-block compression. Default
// to the stable streaming kernel (rather than Portable or a throughput-only
// kernel) to avoid tiny-input cliffs (notably keyed/derive).
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

// Conservative streaming defaults:
// - `stream`: prefer a stable per-block SIMD kernel (SSE4.1 on x86_64) for many small updates
// - `bulk`: prefer the throughput kernel (AVX2 on x86_64) for hashing many chunks
#[cfg(target_arch = "x86_64")]
const DEFAULT_STREAM_KERNEL: KernelId = KernelId::X86Sse41;
#[cfg(target_arch = "aarch64")]
const DEFAULT_STREAM_KERNEL: KernelId = KernelId::Aarch64Neon;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const DEFAULT_STREAM_KERNEL: KernelId = KernelId::Portable;

const DEFAULT_BULK_KERNEL: KernelId = SIMD_KERNEL;

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
  // Fire earlier than the old 512KiB/256-chunk gate so users get automatic
  // multi-core speedups on typical "large buffer" workloads without manual
  // tuning.
  ParallelTable {
    min_bytes: 128 * 1024,
    min_chunks: 64,
    max_threads: 0,
  }
}

#[inline]
#[must_use]
const fn default_kind_streaming_parallel_table() -> ParallelTable {
  default_kind_parallel_table()
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

pub static DEFAULT_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Portable,
  s: KernelId::Portable,
  m: KernelId::Portable,
  l: KernelId::Portable,
};

pub static DEFAULT_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Portable,
  bulk: KernelId::Portable,
};

pub static DEFAULT_PARALLEL_TABLE: ParallelTable = ParallelTable {
  min_bytes: 128 * 1024,
  min_chunks: 64,
  max_threads: 0,
};
pub static DEFAULT_STREAMING_PARALLEL_TABLE: ParallelTable = DEFAULT_PARALLEL_TABLE;

// Custom Table
pub static CUSTOM_TABLE: DispatchTable = DEFAULT_TABLE;
// Custom Streaming Table
pub static CUSTOM_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// Custom Parallel Table
pub static CUSTOM_PARALLEL_TABLE: ParallelTable = DEFAULT_PARALLEL_TABLE;
// Custom Streaming Parallel Table
pub static CUSTOM_STREAMING_PARALLEL_TABLE: ParallelTable = DEFAULT_STREAMING_PARALLEL_TABLE;
// Default Table
pub static DEFAULT_KIND_TABLE: DispatchTable = default_kind_table();
// Default Streaming Table
pub static DEFAULT_KIND_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// Default Parallel Table
pub static DEFAULT_KIND_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// Default Streaming Parallel Table
pub static DEFAULT_KIND_STREAMING_PARALLEL_TABLE: ParallelTable = default_kind_streaming_parallel_table();
// Portable Table
pub static PORTABLE_TABLE: DispatchTable = DEFAULT_TABLE;
// Portable Streaming Table
pub static PORTABLE_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// Portable Parallel Table
pub static PORTABLE_PARALLEL_TABLE: ParallelTable = DEFAULT_PARALLEL_TABLE;
// Portable Streaming Parallel Table
pub static PORTABLE_STREAMING_PARALLEL_TABLE: ParallelTable = PORTABLE_PARALLEL_TABLE;
// Zen4 Table
#[cfg(target_arch = "x86_64")]
pub static ZEN4_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::X86Sse41,
  s: KernelId::X86Sse41,
  m: KernelId::X86Avx512,
  l: KernelId::X86Avx512,
};
#[cfg(not(target_arch = "x86_64"))]
pub static ZEN4_TABLE: DispatchTable = default_kind_table();

// Zen4 Streaming Table
#[cfg(target_arch = "x86_64")]
pub static ZEN4_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::X86Sse41,
  bulk: KernelId::X86Avx2,
};
#[cfg(not(target_arch = "x86_64"))]
pub static ZEN4_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// Zen4 Parallel Table
pub static ZEN4_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// Zen5 Table
#[cfg(target_arch = "x86_64")]
pub static ZEN5_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::X86Sse41,
  s: KernelId::X86Sse41,
  m: KernelId::X86Avx512,
  l: KernelId::X86Avx512,
};
#[cfg(not(target_arch = "x86_64"))]
pub static ZEN5_TABLE: DispatchTable = default_kind_table();

// Zen5 Streaming Table
#[cfg(target_arch = "x86_64")]
pub static ZEN5_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::X86Sse41,
  bulk: KernelId::X86Avx512,
};
#[cfg(not(target_arch = "x86_64"))]
pub static ZEN5_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// Zen5 Parallel Table
pub static ZEN5_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// Zen5c Table
#[cfg(target_arch = "x86_64")]
pub static ZEN5C_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: DEFAULT_XS,
  s: DEFAULT_S,
  m: SIMD_KERNEL,
  l: AVX512_KERNEL,
};
#[cfg(not(target_arch = "x86_64"))]
pub static ZEN5C_TABLE: DispatchTable = default_kind_table();

// Zen5c Streaming Table
#[cfg(target_arch = "x86_64")]
pub static ZEN5C_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: DEFAULT_STREAM_KERNEL,
  bulk: AVX512_KERNEL,
};
#[cfg(not(target_arch = "x86_64"))]
pub static ZEN5C_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// Zen5c Parallel Table
pub static ZEN5C_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// IntelSpr Table
#[cfg(target_arch = "x86_64")]
pub static INTELSPR_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::X86Sse41,
  s: KernelId::X86Sse41,
  m: KernelId::X86Avx512,
  l: KernelId::X86Avx512,
};
#[cfg(not(target_arch = "x86_64"))]
pub static INTELSPR_TABLE: DispatchTable = default_kind_table();

// IntelSpr Streaming Table
#[cfg(target_arch = "x86_64")]
pub static INTELSPR_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::X86Sse41,
  bulk: KernelId::X86Avx512,
};
#[cfg(not(target_arch = "x86_64"))]
pub static INTELSPR_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// IntelSpr Parallel Table
pub static INTELSPR_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// IntelGnr Parallel Table
pub static INTELGNR_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// IntelIcl Parallel Table
pub static INTELICL_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// AppleM1M3 Parallel Table
pub static APPLEM1M3_PARALLEL_TABLE: ParallelTable = ParallelTable {
  min_bytes: 65536,
  min_chunks: 63,
  max_threads: 8,
};

// AppleM4 Parallel Table
pub static APPLEM4_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// AppleM5 Parallel Table
pub static APPLEM5_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// Graviton2 Parallel Table
pub static GRAVITON2_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// Graviton3 Parallel Table
pub static GRAVITON3_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// Graviton4 Parallel Table
pub static GRAVITON4_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// Graviton5 Parallel Table
pub static GRAVITON5_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// NeoverseN2 Parallel Table
pub static NEOVERSEN2_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// NeoverseN3 Parallel Table
pub static NEOVERSEN3_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// NeoverseV3 Parallel Table
pub static NEOVERSEV3_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// NvidiaGrace Parallel Table
pub static NVIDIAGRACE_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// AmpereAltra Parallel Table
pub static AMPEREALTRA_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// Aarch64Pmull Parallel Table
pub static AARCH64PMULL_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// Z13 Parallel Table
pub static Z13_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// Z14 Parallel Table
pub static Z14_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// Z15 Parallel Table
pub static Z15_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// Power7 Parallel Table
pub static POWER7_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// Power8 Parallel Table
pub static POWER8_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// Power9 Parallel Table
pub static POWER9_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();
// Power10 Parallel Table
pub static POWER10_PARALLEL_TABLE: ParallelTable = default_kind_parallel_table();

// Zen4 Streaming Parallel Table
pub static ZEN4_STREAMING_PARALLEL_TABLE: ParallelTable = ZEN4_PARALLEL_TABLE;
// Zen5 Streaming Parallel Table
pub static ZEN5_STREAMING_PARALLEL_TABLE: ParallelTable = ZEN5_PARALLEL_TABLE;
// Zen5c Streaming Parallel Table
pub static ZEN5C_STREAMING_PARALLEL_TABLE: ParallelTable = ZEN5C_PARALLEL_TABLE;
// IntelSpr Streaming Parallel Table
pub static INTELSPR_STREAMING_PARALLEL_TABLE: ParallelTable = INTELSPR_PARALLEL_TABLE;
// IntelGnr Streaming Parallel Table
pub static INTELGNR_STREAMING_PARALLEL_TABLE: ParallelTable = INTELGNR_PARALLEL_TABLE;
// IntelIcl Streaming Parallel Table
pub static INTELICL_STREAMING_PARALLEL_TABLE: ParallelTable = INTELICL_PARALLEL_TABLE;
// AppleM1M3 Streaming Parallel Table
pub static APPLEM1M3_STREAMING_PARALLEL_TABLE: ParallelTable = APPLEM1M3_PARALLEL_TABLE;
// AppleM4 Streaming Parallel Table
pub static APPLEM4_STREAMING_PARALLEL_TABLE: ParallelTable = APPLEM4_PARALLEL_TABLE;
// AppleM5 Streaming Parallel Table
pub static APPLEM5_STREAMING_PARALLEL_TABLE: ParallelTable = APPLEM5_PARALLEL_TABLE;
// Graviton2 Streaming Parallel Table
pub static GRAVITON2_STREAMING_PARALLEL_TABLE: ParallelTable = GRAVITON2_PARALLEL_TABLE;
// Graviton3 Streaming Parallel Table
pub static GRAVITON3_STREAMING_PARALLEL_TABLE: ParallelTable = GRAVITON3_PARALLEL_TABLE;
// Graviton4 Streaming Parallel Table
pub static GRAVITON4_STREAMING_PARALLEL_TABLE: ParallelTable = GRAVITON4_PARALLEL_TABLE;
// Graviton5 Streaming Parallel Table
pub static GRAVITON5_STREAMING_PARALLEL_TABLE: ParallelTable = GRAVITON5_PARALLEL_TABLE;
// NeoverseN2 Streaming Parallel Table
pub static NEOVERSEN2_STREAMING_PARALLEL_TABLE: ParallelTable = NEOVERSEN2_PARALLEL_TABLE;
// NeoverseN3 Streaming Parallel Table
pub static NEOVERSEN3_STREAMING_PARALLEL_TABLE: ParallelTable = NEOVERSEN3_PARALLEL_TABLE;
// NeoverseV3 Streaming Parallel Table
pub static NEOVERSEV3_STREAMING_PARALLEL_TABLE: ParallelTable = NEOVERSEV3_PARALLEL_TABLE;
// NvidiaGrace Streaming Parallel Table
pub static NVIDIAGRACE_STREAMING_PARALLEL_TABLE: ParallelTable = NVIDIAGRACE_PARALLEL_TABLE;
// AmpereAltra Streaming Parallel Table
pub static AMPEREALTRA_STREAMING_PARALLEL_TABLE: ParallelTable = AMPEREALTRA_PARALLEL_TABLE;
// Aarch64Pmull Streaming Parallel Table
pub static AARCH64PMULL_STREAMING_PARALLEL_TABLE: ParallelTable = AARCH64PMULL_PARALLEL_TABLE;
// Z13 Streaming Parallel Table
pub static Z13_STREAMING_PARALLEL_TABLE: ParallelTable = Z13_PARALLEL_TABLE;
// Z14 Streaming Parallel Table
pub static Z14_STREAMING_PARALLEL_TABLE: ParallelTable = Z14_PARALLEL_TABLE;
// Z15 Streaming Parallel Table
pub static Z15_STREAMING_PARALLEL_TABLE: ParallelTable = Z15_PARALLEL_TABLE;
// Power7 Streaming Parallel Table
pub static POWER7_STREAMING_PARALLEL_TABLE: ParallelTable = POWER7_PARALLEL_TABLE;
// Power8 Streaming Parallel Table
pub static POWER8_STREAMING_PARALLEL_TABLE: ParallelTable = POWER8_PARALLEL_TABLE;
// Power9 Streaming Parallel Table
pub static POWER9_STREAMING_PARALLEL_TABLE: ParallelTable = POWER9_PARALLEL_TABLE;
// Power10 Streaming Parallel Table
pub static POWER10_STREAMING_PARALLEL_TABLE: ParallelTable = POWER10_PARALLEL_TABLE;

// IntelGnr Table
#[cfg(target_arch = "x86_64")]
pub static INTELGNR_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: DEFAULT_XS,
  s: DEFAULT_S,
  m: AVX512_KERNEL,
  l: AVX512_KERNEL,
};
#[cfg(not(target_arch = "x86_64"))]
pub static INTELGNR_TABLE: DispatchTable = default_kind_table();

// IntelGnr Streaming Table
#[cfg(target_arch = "x86_64")]
pub static INTELGNR_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: DEFAULT_STREAM_KERNEL,
  bulk: AVX512_KERNEL,
};
#[cfg(not(target_arch = "x86_64"))]
pub static INTELGNR_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// IntelIcl Table
#[cfg(target_arch = "x86_64")]
pub static INTELICL_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::X86Sse41,
  s: KernelId::X86Sse41,
  m: KernelId::X86Avx512,
  l: KernelId::X86Avx512,
};
#[cfg(not(target_arch = "x86_64"))]
pub static INTELICL_TABLE: DispatchTable = default_kind_table();
// IntelIcl Streaming Table
#[cfg(target_arch = "x86_64")]
pub static INTELICL_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::X86Sse41,
  bulk: KernelId::X86Avx2,
};
#[cfg(not(target_arch = "x86_64"))]
pub static INTELICL_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// AppleM1M3 Table
#[cfg(target_arch = "aarch64")]
pub static APPLEM1M3_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Portable,
  s: KernelId::Portable,
  m: KernelId::Aarch64Neon,
  l: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static APPLEM1M3_TABLE: DispatchTable = default_kind_table();

// AppleM1M3 Streaming Table
#[cfg(target_arch = "aarch64")]
pub static APPLEM1M3_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Aarch64Neon,
  bulk: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static APPLEM1M3_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;

// AppleM4 Table
#[cfg(target_arch = "aarch64")]
pub static APPLEM4_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Portable,
  s: KernelId::Portable,
  m: KernelId::Aarch64Neon,
  l: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static APPLEM4_TABLE: DispatchTable = default_kind_table();
// AppleM4 Streaming Table
#[cfg(target_arch = "aarch64")]
pub static APPLEM4_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Aarch64Neon,
  bulk: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static APPLEM4_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// AppleM5 Table
#[cfg(target_arch = "aarch64")]
pub static APPLEM5_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Portable,
  s: KernelId::Portable,
  m: KernelId::Aarch64Neon,
  l: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static APPLEM5_TABLE: DispatchTable = default_kind_table();
// AppleM5 Streaming Table
#[cfg(target_arch = "aarch64")]
pub static APPLEM5_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Aarch64Neon,
  bulk: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static APPLEM5_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// Graviton2 Table
#[cfg(target_arch = "aarch64")]
pub static GRAVITON2_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Aarch64Neon,
  s: KernelId::Aarch64Neon,
  m: KernelId::Aarch64Neon,
  l: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static GRAVITON2_TABLE: DispatchTable = default_kind_table();
// Graviton2 Streaming Table
#[cfg(target_arch = "aarch64")]
pub static GRAVITON2_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Aarch64Neon,
  bulk: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static GRAVITON2_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// Graviton3 Table
#[cfg(target_arch = "aarch64")]
pub static GRAVITON3_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Aarch64Neon,
  s: KernelId::Aarch64Neon,
  m: KernelId::Aarch64Neon,
  l: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static GRAVITON3_TABLE: DispatchTable = default_kind_table();
// Graviton3 Streaming Table
#[cfg(target_arch = "aarch64")]
pub static GRAVITON3_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Aarch64Neon,
  bulk: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static GRAVITON3_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// Graviton4 Table
#[cfg(target_arch = "aarch64")]
pub static GRAVITON4_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Aarch64Neon,
  s: KernelId::Aarch64Neon,
  m: KernelId::Aarch64Neon,
  l: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static GRAVITON4_TABLE: DispatchTable = default_kind_table();
// Graviton4 Streaming Table
#[cfg(target_arch = "aarch64")]
pub static GRAVITON4_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Aarch64Neon,
  bulk: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static GRAVITON4_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// Graviton5 Table
#[cfg(target_arch = "aarch64")]
pub static GRAVITON5_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Aarch64Neon,
  s: KernelId::Aarch64Neon,
  m: KernelId::Aarch64Neon,
  l: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static GRAVITON5_TABLE: DispatchTable = default_kind_table();
// Graviton5 Streaming Table
#[cfg(target_arch = "aarch64")]
pub static GRAVITON5_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Aarch64Neon,
  bulk: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static GRAVITON5_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// NeoverseN2 Table
#[cfg(target_arch = "aarch64")]
pub static NEOVERSEN2_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Aarch64Neon,
  s: KernelId::Aarch64Neon,
  m: KernelId::Aarch64Neon,
  l: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static NEOVERSEN2_TABLE: DispatchTable = default_kind_table();
// NeoverseN2 Streaming Table
#[cfg(target_arch = "aarch64")]
pub static NEOVERSEN2_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Aarch64Neon,
  bulk: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static NEOVERSEN2_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// NeoverseN3 Table
#[cfg(target_arch = "aarch64")]
pub static NEOVERSEN3_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Aarch64Neon,
  s: KernelId::Aarch64Neon,
  m: KernelId::Aarch64Neon,
  l: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static NEOVERSEN3_TABLE: DispatchTable = default_kind_table();
// NeoverseN3 Streaming Table
#[cfg(target_arch = "aarch64")]
pub static NEOVERSEN3_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Aarch64Neon,
  bulk: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static NEOVERSEN3_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// NeoverseV3 Table
#[cfg(target_arch = "aarch64")]
pub static NEOVERSEV3_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Aarch64Neon,
  s: KernelId::Aarch64Neon,
  m: KernelId::Aarch64Neon,
  l: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static NEOVERSEV3_TABLE: DispatchTable = default_kind_table();
// NeoverseV3 Streaming Table
#[cfg(target_arch = "aarch64")]
pub static NEOVERSEV3_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Aarch64Neon,
  bulk: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static NEOVERSEV3_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// NvidiaGrace Table
#[cfg(target_arch = "aarch64")]
pub static NVIDIAGRACE_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Aarch64Neon,
  s: KernelId::Aarch64Neon,
  m: KernelId::Aarch64Neon,
  l: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static NVIDIAGRACE_TABLE: DispatchTable = default_kind_table();
// NvidiaGrace Streaming Table
#[cfg(target_arch = "aarch64")]
pub static NVIDIAGRACE_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Aarch64Neon,
  bulk: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static NVIDIAGRACE_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// AmpereAltra Table
#[cfg(target_arch = "aarch64")]
pub static AMPEREALTRA_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Aarch64Neon,
  s: KernelId::Aarch64Neon,
  m: KernelId::Aarch64Neon,
  l: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static AMPEREALTRA_TABLE: DispatchTable = default_kind_table();
// AmpereAltra Streaming Table
#[cfg(target_arch = "aarch64")]
pub static AMPEREALTRA_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Aarch64Neon,
  bulk: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static AMPEREALTRA_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// Aarch64Pmull Table
#[cfg(target_arch = "aarch64")]
pub static AARCH64PMULL_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Aarch64Neon,
  s: KernelId::Aarch64Neon,
  m: KernelId::Aarch64Neon,
  l: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static AARCH64PMULL_TABLE: DispatchTable = default_kind_table();
// Aarch64Pmull Streaming Table
#[cfg(target_arch = "aarch64")]
pub static AARCH64PMULL_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Aarch64Neon,
  bulk: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static AARCH64PMULL_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// Z13 Table
pub static Z13_TABLE: DispatchTable = default_kind_table();
// Z13 Streaming Table
pub static Z13_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Portable,
  bulk: KernelId::Portable,
};
// Z14 Table
pub static Z14_TABLE: DispatchTable = default_kind_table();
// Z14 Streaming Table
pub static Z14_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Portable,
  bulk: KernelId::Portable,
};
// Z15 Table
pub static Z15_TABLE: DispatchTable = default_kind_table();
// Z15 Streaming Table
pub static Z15_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Portable,
  bulk: KernelId::Portable,
};
// Power7 Table
pub static POWER7_TABLE: DispatchTable = default_kind_table();
// Power7 Streaming Table
pub static POWER7_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Portable,
  bulk: KernelId::Portable,
};
// Power8 Table
pub static POWER8_TABLE: DispatchTable = default_kind_table();
// Power8 Streaming Table
pub static POWER8_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Portable,
  bulk: KernelId::Portable,
};
// Power9 Table
pub static POWER9_TABLE: DispatchTable = default_kind_table();
// Power9 Streaming Table
pub static POWER9_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Portable,
  bulk: KernelId::Portable,
};
// Power10 Table
pub static POWER10_TABLE: DispatchTable = default_kind_table();
// Power10 Streaming Table
pub static POWER10_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Portable,
  bulk: KernelId::Portable,
};

#[inline]
#[must_use]
pub fn select_table(kind: TuneKind) -> &'static DispatchTable {
  match kind {
    TuneKind::Custom => &CUSTOM_TABLE,
    TuneKind::Default => &DEFAULT_KIND_TABLE,
    TuneKind::Portable => &PORTABLE_TABLE,
    TuneKind::Zen4 => &ZEN4_TABLE,
    TuneKind::Zen5 => &ZEN5_TABLE,
    TuneKind::Zen5c => &ZEN5C_TABLE,
    TuneKind::IntelSpr => &INTELSPR_TABLE,
    TuneKind::IntelGnr => &INTELGNR_TABLE,
    TuneKind::IntelIcl => &INTELICL_TABLE,
    TuneKind::AppleM1M3 => &APPLEM1M3_TABLE,
    TuneKind::AppleM4 => &APPLEM4_TABLE,
    TuneKind::AppleM5 => &APPLEM5_TABLE,
    TuneKind::Graviton2 => &GRAVITON2_TABLE,
    TuneKind::Graviton3 => &GRAVITON3_TABLE,
    TuneKind::Graviton4 => &GRAVITON4_TABLE,
    TuneKind::Graviton5 => &GRAVITON5_TABLE,
    TuneKind::NeoverseN2 => &NEOVERSEN2_TABLE,
    TuneKind::NeoverseN3 => &NEOVERSEN3_TABLE,
    TuneKind::NeoverseV3 => &NEOVERSEV3_TABLE,
    TuneKind::NvidiaGrace => &NVIDIAGRACE_TABLE,
    TuneKind::AmpereAltra => &AMPEREALTRA_TABLE,
    TuneKind::Aarch64Pmull => &AARCH64PMULL_TABLE,
    TuneKind::Z13 => &Z13_TABLE,
    TuneKind::Z14 => &Z14_TABLE,
    TuneKind::Z15 => &Z15_TABLE,
    TuneKind::Power7 => &POWER7_TABLE,
    TuneKind::Power8 => &POWER8_TABLE,
    TuneKind::Power9 => &POWER9_TABLE,
    TuneKind::Power10 => &POWER10_TABLE,
  }
}

#[inline]
#[must_use]
pub fn select_streaming_table(kind: TuneKind) -> &'static StreamingTable {
  match kind {
    TuneKind::Custom => &CUSTOM_STREAMING_TABLE,
    TuneKind::Default => &DEFAULT_KIND_STREAMING_TABLE,
    TuneKind::Portable => &PORTABLE_STREAMING_TABLE,
    TuneKind::Zen4 => &ZEN4_STREAMING_TABLE,
    TuneKind::Zen5 => &ZEN5_STREAMING_TABLE,
    TuneKind::Zen5c => &ZEN5C_STREAMING_TABLE,
    TuneKind::IntelSpr => &INTELSPR_STREAMING_TABLE,
    TuneKind::IntelGnr => &INTELGNR_STREAMING_TABLE,
    TuneKind::IntelIcl => &INTELICL_STREAMING_TABLE,
    TuneKind::AppleM1M3 => &APPLEM1M3_STREAMING_TABLE,
    TuneKind::AppleM4 => &APPLEM4_STREAMING_TABLE,
    TuneKind::AppleM5 => &APPLEM5_STREAMING_TABLE,
    TuneKind::Graviton2 => &GRAVITON2_STREAMING_TABLE,
    TuneKind::Graviton3 => &GRAVITON3_STREAMING_TABLE,
    TuneKind::Graviton4 => &GRAVITON4_STREAMING_TABLE,
    TuneKind::Graviton5 => &GRAVITON5_STREAMING_TABLE,
    TuneKind::NeoverseN2 => &NEOVERSEN2_STREAMING_TABLE,
    TuneKind::NeoverseN3 => &NEOVERSEN3_STREAMING_TABLE,
    TuneKind::NeoverseV3 => &NEOVERSEV3_STREAMING_TABLE,
    TuneKind::NvidiaGrace => &NVIDIAGRACE_STREAMING_TABLE,
    TuneKind::AmpereAltra => &AMPEREALTRA_STREAMING_TABLE,
    TuneKind::Aarch64Pmull => &AARCH64PMULL_STREAMING_TABLE,
    TuneKind::Z13 => &Z13_STREAMING_TABLE,
    TuneKind::Z14 => &Z14_STREAMING_TABLE,
    TuneKind::Z15 => &Z15_STREAMING_TABLE,
    TuneKind::Power7 => &POWER7_STREAMING_TABLE,
    TuneKind::Power8 => &POWER8_STREAMING_TABLE,
    TuneKind::Power9 => &POWER9_STREAMING_TABLE,
    TuneKind::Power10 => &POWER10_STREAMING_TABLE,
  }
}

#[inline]
#[must_use]
pub fn select_parallel_table(kind: TuneKind) -> &'static ParallelTable {
  match kind {
    TuneKind::Custom => &CUSTOM_PARALLEL_TABLE,
    TuneKind::Default => &DEFAULT_KIND_PARALLEL_TABLE,
    TuneKind::Portable => &PORTABLE_PARALLEL_TABLE,
    TuneKind::Zen4 => &ZEN4_PARALLEL_TABLE,
    TuneKind::Zen5 => &ZEN5_PARALLEL_TABLE,
    TuneKind::Zen5c => &ZEN5C_PARALLEL_TABLE,
    TuneKind::IntelSpr => &INTELSPR_PARALLEL_TABLE,
    TuneKind::IntelGnr => &INTELGNR_PARALLEL_TABLE,
    TuneKind::IntelIcl => &INTELICL_PARALLEL_TABLE,
    TuneKind::AppleM1M3 => &APPLEM1M3_PARALLEL_TABLE,
    TuneKind::AppleM4 => &APPLEM4_PARALLEL_TABLE,
    TuneKind::AppleM5 => &APPLEM5_PARALLEL_TABLE,
    TuneKind::Graviton2 => &GRAVITON2_PARALLEL_TABLE,
    TuneKind::Graviton3 => &GRAVITON3_PARALLEL_TABLE,
    TuneKind::Graviton4 => &GRAVITON4_PARALLEL_TABLE,
    TuneKind::Graviton5 => &GRAVITON5_PARALLEL_TABLE,
    TuneKind::NeoverseN2 => &NEOVERSEN2_PARALLEL_TABLE,
    TuneKind::NeoverseN3 => &NEOVERSEN3_PARALLEL_TABLE,
    TuneKind::NeoverseV3 => &NEOVERSEV3_PARALLEL_TABLE,
    TuneKind::NvidiaGrace => &NVIDIAGRACE_PARALLEL_TABLE,
    TuneKind::AmpereAltra => &AMPEREALTRA_PARALLEL_TABLE,
    TuneKind::Aarch64Pmull => &AARCH64PMULL_PARALLEL_TABLE,
    TuneKind::Z13 => &Z13_PARALLEL_TABLE,
    TuneKind::Z14 => &Z14_PARALLEL_TABLE,
    TuneKind::Z15 => &Z15_PARALLEL_TABLE,
    TuneKind::Power7 => &POWER7_PARALLEL_TABLE,
    TuneKind::Power8 => &POWER8_PARALLEL_TABLE,
    TuneKind::Power9 => &POWER9_PARALLEL_TABLE,
    TuneKind::Power10 => &POWER10_PARALLEL_TABLE,
  }
}

#[inline]
#[must_use]
pub fn select_streaming_parallel_table(kind: TuneKind) -> &'static ParallelTable {
  match kind {
    TuneKind::Custom => &CUSTOM_STREAMING_PARALLEL_TABLE,
    TuneKind::Default => &DEFAULT_KIND_STREAMING_PARALLEL_TABLE,
    TuneKind::Portable => &PORTABLE_STREAMING_PARALLEL_TABLE,
    TuneKind::Zen4 => &ZEN4_STREAMING_PARALLEL_TABLE,
    TuneKind::Zen5 => &ZEN5_STREAMING_PARALLEL_TABLE,
    TuneKind::Zen5c => &ZEN5C_STREAMING_PARALLEL_TABLE,
    TuneKind::IntelSpr => &INTELSPR_STREAMING_PARALLEL_TABLE,
    TuneKind::IntelGnr => &INTELGNR_STREAMING_PARALLEL_TABLE,
    TuneKind::IntelIcl => &INTELICL_STREAMING_PARALLEL_TABLE,
    TuneKind::AppleM1M3 => &APPLEM1M3_STREAMING_PARALLEL_TABLE,
    TuneKind::AppleM4 => &APPLEM4_STREAMING_PARALLEL_TABLE,
    TuneKind::AppleM5 => &APPLEM5_STREAMING_PARALLEL_TABLE,
    TuneKind::Graviton2 => &GRAVITON2_STREAMING_PARALLEL_TABLE,
    TuneKind::Graviton3 => &GRAVITON3_STREAMING_PARALLEL_TABLE,
    TuneKind::Graviton4 => &GRAVITON4_STREAMING_PARALLEL_TABLE,
    TuneKind::Graviton5 => &GRAVITON5_STREAMING_PARALLEL_TABLE,
    TuneKind::NeoverseN2 => &NEOVERSEN2_STREAMING_PARALLEL_TABLE,
    TuneKind::NeoverseN3 => &NEOVERSEN3_STREAMING_PARALLEL_TABLE,
    TuneKind::NeoverseV3 => &NEOVERSEV3_STREAMING_PARALLEL_TABLE,
    TuneKind::NvidiaGrace => &NVIDIAGRACE_STREAMING_PARALLEL_TABLE,
    TuneKind::AmpereAltra => &AMPEREALTRA_STREAMING_PARALLEL_TABLE,
    TuneKind::Aarch64Pmull => &AARCH64PMULL_STREAMING_PARALLEL_TABLE,
    TuneKind::Z13 => &Z13_STREAMING_PARALLEL_TABLE,
    TuneKind::Z14 => &Z14_STREAMING_PARALLEL_TABLE,
    TuneKind::Z15 => &Z15_STREAMING_PARALLEL_TABLE,
    TuneKind::Power7 => &POWER7_STREAMING_PARALLEL_TABLE,
    TuneKind::Power8 => &POWER8_STREAMING_PARALLEL_TABLE,
    TuneKind::Power9 => &POWER9_STREAMING_PARALLEL_TABLE,
    TuneKind::Power10 => &POWER10_STREAMING_PARALLEL_TABLE,
  }
}
