use backend::OnceCache;
use platform::Caps;
#[cfg(target_arch = "x86_64")]
use platform::TuneKind;
#[cfg(target_arch = "x86_64")]
use platform::caps::x86;

use super::{
  dispatch_tables::{DispatchTable, ParallelTable, StreamingTable},
  kernels::{Blake3KernelId, Kernel, kernel, required_caps},
};
use crate::crypto::dispatch_util::SizeClassDispatch;

#[cfg(target_arch = "x86_64")]
#[inline]
#[must_use]
fn prefer_intel_icl_tables(caps: Caps, kind: TuneKind) -> bool {
  // `platform` currently groups many AVX-512 Intel server CPUs under the
  // IntelSpr preset. BLAKE3 kernel selection is sensitive to AVX-512 warmup
  // overhead, so we split "SPR-like" into:
  // - IntelSpr: AMX-capable (SPR/EMR-class)
  // - IntelIcl: no-AMX (ICL-SP and older AVX-512 servers)
  kind == TuneKind::IntelSpr
    && !caps.has(x86::AMX_TILE)
    && !caps.has(x86::AMX_INT8)
    && !caps.has(x86::AMX_BF16)
    && !caps.has(x86::AMX_FP16)
    && !caps.has(x86::AMX_COMPLEX)
}

#[derive(Clone, Copy)]
struct Entry {
  kernel: Kernel,
}

#[derive(Clone, Copy)]
struct ActiveDispatch {
  boundaries: [usize; 3],
  xs: Entry,
  s: Entry,
  m: Entry,
  l: Entry,
}

#[derive(Clone, Copy)]
struct ResolvedDispatch {
  active: ActiveDispatch,
  streaming: StreamingDispatch,
  parallel: ParallelDispatch,
  simd_threshold: usize,
}

static RESOLVED: OnceCache<ResolvedDispatch> = OnceCache::new();

#[derive(Clone, Copy)]
pub(crate) struct StreamingDispatch {
  pub(crate) stream: Kernel,
  pub(crate) bulk: Kernel,
}

#[derive(Clone, Copy)]
pub(crate) struct ParallelDispatch {
  pub(crate) oneshot: ParallelTable,
  pub(crate) streaming: ParallelTable,
}

#[inline]
#[must_use]
fn effective_tune_kind(caps: Caps, tune: platform::Tune) -> platform::TuneKind {
  #[cfg(target_arch = "x86_64")]
  {
    if prefer_intel_icl_tables(caps, tune.kind) {
      TuneKind::IntelIcl
    } else {
      tune.kind
    }
  }
  #[cfg(not(target_arch = "x86_64"))]
  {
    let _ = caps;
    tune.kind
  }
}

#[inline]
#[must_use]
fn resolve(id: Blake3KernelId, caps: Caps) -> Blake3KernelId {
  // Tables express *preferences*. Here we enforce correctness (required CPU
  // features) and apply a conservative, architecture-aware fallback order.
  //
  // Important: avoid a "missing feature => Portable" cliff when a higher-tier
  // kernel is requested (e.g. AVX-512) but a lower-tier kernel (e.g. AVX2)
  // would work.
  match id {
    Blake3KernelId::Portable => Blake3KernelId::Portable,
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx512 => {
      if caps.has(required_caps(Blake3KernelId::X86Avx512)) {
        Blake3KernelId::X86Avx512
      } else if caps.has(required_caps(Blake3KernelId::X86Avx2)) {
        Blake3KernelId::X86Avx2
      } else if caps.has(required_caps(Blake3KernelId::X86Sse41)) {
        Blake3KernelId::X86Sse41
      } else if caps.has(required_caps(Blake3KernelId::X86Ssse3)) {
        Blake3KernelId::X86Ssse3
      } else {
        Blake3KernelId::Portable
      }
    }
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx2 => {
      if caps.has(required_caps(Blake3KernelId::X86Avx2)) {
        Blake3KernelId::X86Avx2
      } else if caps.has(required_caps(Blake3KernelId::X86Sse41)) {
        Blake3KernelId::X86Sse41
      } else if caps.has(required_caps(Blake3KernelId::X86Ssse3)) {
        Blake3KernelId::X86Ssse3
      } else {
        Blake3KernelId::Portable
      }
    }
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Sse41 => {
      if caps.has(required_caps(Blake3KernelId::X86Sse41)) {
        Blake3KernelId::X86Sse41
      } else if caps.has(required_caps(Blake3KernelId::X86Ssse3)) {
        Blake3KernelId::X86Ssse3
      } else {
        Blake3KernelId::Portable
      }
    }
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Ssse3 => {
      if caps.has(required_caps(Blake3KernelId::X86Ssse3)) {
        Blake3KernelId::X86Ssse3
      } else {
        Blake3KernelId::Portable
      }
    }
    #[cfg(target_arch = "aarch64")]
    Blake3KernelId::Aarch64Neon => {
      if caps.has(required_caps(Blake3KernelId::Aarch64Neon)) {
        Blake3KernelId::Aarch64Neon
      } else {
        Blake3KernelId::Portable
      }
    }
  }
}

#[inline]
#[must_use]
fn resolved() -> ResolvedDispatch {
  RESOLVED.get_or_init(|| {
    let det = platform::get();
    let caps = det.caps;
    let tune = det.tune;
    let kind = effective_tune_kind(caps, tune);

    let table: &'static DispatchTable = super::dispatch_tables::select_table(kind);
    let stream_table: &'static StreamingTable = super::dispatch_tables::select_streaming_table(kind);
    let oneshot_parallel_table: &'static ParallelTable = super::dispatch_tables::select_parallel_table(kind);
    let streaming_parallel_table: &'static ParallelTable =
      super::dispatch_tables::select_streaming_parallel_table(kind);

    let xs_id = resolve(table.xs, caps);
    let s_id = resolve(table.s, caps);
    let m_id = resolve(table.m, caps);
    let l_id = resolve(table.l, caps);

    let stream_id = resolve(stream_table.stream, caps);
    let bulk_id = resolve(stream_table.bulk, caps);

    ResolvedDispatch {
      active: ActiveDispatch {
        boundaries: table.boundaries,
        xs: Entry { kernel: kernel(xs_id) },
        s: Entry { kernel: kernel(s_id) },
        m: Entry { kernel: kernel(m_id) },
        l: Entry { kernel: kernel(l_id) },
      },
      streaming: StreamingDispatch {
        stream: kernel(stream_id),
        bulk: kernel(bulk_id),
      },
      parallel: ParallelDispatch {
        oneshot: *oneshot_parallel_table,
        streaming: *streaming_parallel_table,
      },
      simd_threshold: tune.simd_threshold,
    }
  })
}

#[inline]
#[must_use]
fn active() -> ActiveDispatch {
  resolved().active
}

#[inline]
#[must_use]
fn active_streaming() -> StreamingDispatch {
  resolved().streaming
}

#[inline]
#[must_use]
fn active_parallel() -> ParallelDispatch {
  resolved().parallel
}

#[inline]
#[must_use]
fn select(d: &ActiveDispatch, len: usize) -> Entry {
  let [xs_max, s_max, m_max] = d.boundaries;
  if len <= xs_max {
    d.xs
  } else if len <= s_max {
    d.s
  } else if len <= m_max {
    d.m
  } else {
    d.l
  }
}

#[inline]
#[must_use]
pub fn kernel_name_for_len(len: usize) -> &'static str {
  let d = active();
  select(&d, len).kernel.name
}

#[inline]
#[must_use]
pub fn digest(data: &[u8]) -> [u8; 32] {
  let d = active();
  let kernel = select(&d, data.len()).kernel;

  super::digest_oneshot(kernel, super::IV, 0, data)
}

#[inline]
#[must_use]
pub fn xof(data: &[u8]) -> super::Blake3Xof {
  let d = active();
  let kernel = select(&d, data.len()).kernel;
  let output = super::root_output_oneshot(kernel, super::IV, 0, data);
  super::Blake3Xof::new(output)
}

#[inline]
#[must_use]
pub(crate) fn kernel_dispatch() -> SizeClassDispatch<Kernel> {
  let d = active();
  SizeClassDispatch {
    boundaries: d.boundaries,
    xs: d.xs.kernel,
    s: d.s.kernel,
    m: d.m.kernel,
    l: d.l.kernel,
  }
}

#[inline]
#[must_use]
pub(crate) fn streaming_dispatch() -> StreamingDispatch {
  active_streaming()
}

#[inline]
#[must_use]
pub(crate) fn parallel_dispatch() -> ParallelDispatch {
  active_parallel()
}

#[inline]
#[must_use]
pub(crate) fn streaming_simd_threshold() -> usize {
  resolved().simd_threshold
}

#[inline]
#[must_use]
pub fn streaming_kernel_names() -> (&'static str, &'static str) {
  let d = active_streaming();
  (d.stream.name, d.bulk.name)
}

// ─── Bench-only introspection ────────────────────────────────────────────────

/// BLAKE3 spec flag: keyed hash mode (bit 4).
#[doc(hidden)]
pub const FLAGS_KEYED_HASH: u32 = 1 << 4;

/// BLAKE3 spec flag: derive-key-material mode (bit 6).
#[doc(hidden)]
pub const FLAGS_DERIVE_KEY_MATERIAL: u32 = 1 << 6;

/// Dispatch decisions that `Blake3::update()` would make for a single call
/// with the given flags and input length.
///
/// Bench-only introspection hook — not part of the stable API.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct StreamingDispatchInfo {
  pub stream_kernel: &'static str,
  pub bulk_kernel: &'static str,
  pub parallel_min_bytes: usize,
  pub parallel_min_chunks: usize,
  pub parallel_max_threads: u8,
  pub would_parallelize: bool,
  pub parallel_threads: usize,
}

/// Minimum committed full chunks per worker thread for streaming-update
/// parallelism in `Blake3::update_with`.
pub(crate) const STREAM_PARALLEL_MIN_CHUNKS_PER_THREAD: usize = 16;

/// Return the dispatch decisions that `Blake3::update()` would make for a
/// single call with the given `flags` and `input_len`.
///
/// Bench-only introspection hook — not part of the stable API.
#[doc(hidden)]
#[must_use]
pub fn streaming_dispatch_info(flags: u32, input_len: usize) -> StreamingDispatchInfo {
  let ptable = active_parallel().streaming;

  // Mirror the lazy-SIMD gate in `update()`: plain hashes stay portable for
  // tiny inputs below the platform's simd_threshold.
  let is_plain = flags == 0;
  let (stream_kernel, bulk_kernel) = if is_plain && input_len < streaming_simd_threshold() {
    ("portable", "portable")
  } else {
    let sd = active_streaming();
    let bulk = if input_len >= 8 * 1024 {
      kernel_dispatch().select(input_len).name
    } else {
      sd.bulk.name
    };
    (sd.stream.name, bulk)
  };

  // Mirror `parallel_policy_threads` + the per-thread work guard
  // from `update_with`.
  const CHUNK_LEN: usize = 1024;

  let commit_chunks = input_len / CHUNK_LEN;
  let (would_parallelize, parallel_threads) =
    parallel_threads_for(&ptable, input_len, commit_chunks, STREAM_PARALLEL_MIN_CHUNKS_PER_THREAD);

  StreamingDispatchInfo {
    stream_kernel,
    bulk_kernel,
    parallel_min_bytes: ptable.min_bytes,
    parallel_min_chunks: ptable.min_chunks,
    parallel_max_threads: ptable.max_threads,
    would_parallelize,
    parallel_threads,
  }
}

/// Compute (would_parallelize, thread_count) mirroring `parallel_policy_threads`
/// + the `MIN_CHUNKS_PER_THREAD` streaming guard.
fn parallel_threads_for(
  table: &ParallelTable,
  input_bytes: usize,
  commit_chunks: usize,
  min_chunks_per_thread: usize,
) -> (bool, usize) {
  if table.max_threads == 1 || table.min_bytes == usize::MAX {
    return (false, 1);
  }
  if input_bytes < table.min_bytes || commit_chunks < table.min_chunks {
    return (false, 1);
  }

  #[cfg(feature = "std")]
  {
    let Ok(ap) = std::thread::available_parallelism() else {
      return (false, 1);
    };
    let mut threads = ap.get();
    if table.max_threads != 0 {
      threads = threads.min(table.max_threads as usize);
    }
    if threads <= 1 {
      return (false, 1);
    }
    // Streaming guard: need at least min_chunks_per_thread chunks per thread
    let max_by_work = commit_chunks / min_chunks_per_thread;
    if max_by_work <= 1 {
      return (false, 1);
    }
    threads = threads.min(max_by_work);
    (true, threads)
  }
  #[cfg(not(feature = "std"))]
  {
    let _ = min_chunks_per_thread;
    (false, 1)
  }
}
