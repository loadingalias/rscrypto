//! Tuned dispatch tables for BLAKE3.
//!
//! This module is the integration point for `rscrypto-tune --apply`.
//! The tuning engine updates the per-`TuneKind` tables in this file.

use platform::TuneKind;

pub use super::kernels::Blake3KernelId as KernelId;

pub const DEFAULT_BOUNDARIES: [usize; 3] = [64, 256, 4096];

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
const DEFAULT_XS: KernelId = KernelId::Portable;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const DEFAULT_XS: KernelId = KernelId::Portable;

#[cfg(target_arch = "x86_64")]
const DEFAULT_S: KernelId = KernelId::X86Sse41;
#[cfg(target_arch = "aarch64")]
const DEFAULT_S: KernelId = KernelId::Portable;
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

// Custom Table
pub static CUSTOM_TABLE: DispatchTable = DEFAULT_TABLE;
// Custom Streaming Table
pub static CUSTOM_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
// Default Table
pub static DEFAULT_KIND_TABLE: DispatchTable = default_kind_table();
// Default Streaming Table
pub static DEFAULT_KIND_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// Portable Table
pub static PORTABLE_TABLE: DispatchTable = DEFAULT_TABLE;
// Portable Streaming Table
pub static PORTABLE_STREAMING_TABLE: StreamingTable = DEFAULT_STREAMING_TABLE;
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
pub static ZEN4_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
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
pub static ZEN5_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
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
pub static ZEN5C_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
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
pub static INTELSPR_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();

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
pub static INTELGNR_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
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
  bulk: KernelId::X86Avx512,
};
#[cfg(not(target_arch = "x86_64"))]
pub static INTELICL_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
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
  stream: KernelId::Portable,
  bulk: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static APPLEM1M3_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();

// AppleM4 Table
pub static APPLEM4_TABLE: DispatchTable = default_kind_table();
// AppleM4 Streaming Table
pub static APPLEM4_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// AppleM5 Table
pub static APPLEM5_TABLE: DispatchTable = default_kind_table();
// AppleM5 Streaming Table
pub static APPLEM5_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// Graviton2 Table
pub static GRAVITON2_TABLE: DispatchTable = default_kind_table();
// Graviton2 Streaming Table
pub static GRAVITON2_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// Graviton3 Table
pub static GRAVITON3_TABLE: DispatchTable = default_kind_table();
// Graviton3 Streaming Table
pub static GRAVITON3_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// Graviton4 Table
pub static GRAVITON4_TABLE: DispatchTable = default_kind_table();
// Graviton4 Streaming Table
pub static GRAVITON4_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// Graviton5 Table
pub static GRAVITON5_TABLE: DispatchTable = default_kind_table();
// Graviton5 Streaming Table
pub static GRAVITON5_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// NeoverseN2 Table
pub static NEOVERSEN2_TABLE: DispatchTable = default_kind_table();
// NeoverseN2 Streaming Table
pub static NEOVERSEN2_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// NeoverseN3 Table
pub static NEOVERSEN3_TABLE: DispatchTable = default_kind_table();
// NeoverseN3 Streaming Table
pub static NEOVERSEN3_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// NeoverseV3 Table
pub static NEOVERSEV3_TABLE: DispatchTable = default_kind_table();
// NeoverseV3 Streaming Table
pub static NEOVERSEV3_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// NvidiaGrace Table
#[cfg(target_arch = "aarch64")]
pub static NVIDIAGRACE_TABLE: DispatchTable = DispatchTable {
  boundaries: DEFAULT_BOUNDARIES,
  xs: KernelId::Aarch64Neon,
  s: KernelId::Portable,
  m: KernelId::Aarch64Neon,
  l: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static NVIDIAGRACE_TABLE: DispatchTable = default_kind_table();
// NvidiaGrace Streaming Table
#[cfg(target_arch = "aarch64")]
pub static NVIDIAGRACE_STREAMING_TABLE: StreamingTable = StreamingTable {
  stream: KernelId::Portable,
  bulk: KernelId::Aarch64Neon,
};
#[cfg(not(target_arch = "aarch64"))]
pub static NVIDIAGRACE_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// AmpereAltra Table
pub static AMPEREALTRA_TABLE: DispatchTable = default_kind_table();
// AmpereAltra Streaming Table
pub static AMPEREALTRA_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// Aarch64Pmull Table
pub static AARCH64PMULL_TABLE: DispatchTable = default_kind_table();
// Aarch64Pmull Streaming Table
pub static AARCH64PMULL_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// Z13 Table
pub static Z13_TABLE: DispatchTable = default_kind_table();
// Z13 Streaming Table
pub static Z13_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// Z14 Table
pub static Z14_TABLE: DispatchTable = default_kind_table();
// Z14 Streaming Table
pub static Z14_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// Z15 Table
pub static Z15_TABLE: DispatchTable = default_kind_table();
// Z15 Streaming Table
pub static Z15_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// Power7 Table
pub static POWER7_TABLE: DispatchTable = default_kind_table();
// Power7 Streaming Table
pub static POWER7_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// Power8 Table
pub static POWER8_TABLE: DispatchTable = default_kind_table();
// Power8 Streaming Table
pub static POWER8_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// Power9 Table
pub static POWER9_TABLE: DispatchTable = default_kind_table();
// Power9 Streaming Table
pub static POWER9_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();
// Power10 Table
pub static POWER10_TABLE: DispatchTable = default_kind_table();
// Power10 Streaming Table
pub static POWER10_STREAMING_TABLE: StreamingTable = default_kind_streaming_table();

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
