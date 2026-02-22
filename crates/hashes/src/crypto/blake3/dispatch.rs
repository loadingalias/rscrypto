use backend::OnceCache;
use platform::Caps;
#[cfg(target_arch = "x86_64")]
use platform::TuneKind;
#[cfg(target_arch = "x86_64")]
use platform::caps::x86;

#[cfg(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "s390x",
  target_arch = "powerpc64",
  target_arch = "riscv64"
))]
use super::kernels::required_caps;
use super::{
  dispatch_tables::{DispatchTable, ParallelTable, StreamingTable},
  kernels::{Blake3KernelId, Kernel, kernel},
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
  hasher: HasherDispatch,
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
  pub(crate) keyed_oneshot: ParallelTable,
  pub(crate) derive_oneshot: ParallelTable,
  pub(crate) xof: ParallelTable,
  pub(crate) keyed_xof: ParallelTable,
  pub(crate) derive_xof: ParallelTable,
  pub(crate) streaming: ParallelTable,
  pub(crate) keyed_streaming: ParallelTable,
  pub(crate) derive_streaming: ParallelTable,
}

// Note: STREAMING_BULK_SIZECLASS_MIN_LEN is now table-driven per profile.
// See StreamingTable::bulk_sizeclass_threshold in dispatch_tables.rs

/// Immutable per-hasher dispatch snapshot.
///
/// This is resolved once from platform caps/tune and can be copied into each
/// hasher, avoiding repeated global dispatch lookups in hot update/finalize
/// paths.
#[derive(Clone, Copy)]
pub(crate) struct HasherDispatch {
  size_classes: SizeClassDispatch<Kernel>,
  plain_size_classes: SizeClassDispatch<Kernel>,
  stream_kernel: Kernel,
  table_bulk_kernel: Kernel,
  parallel_streaming: ParallelTable,
  simd_threshold: usize,
  bulk_sizeclass_threshold: usize,
}

impl HasherDispatch {
  #[inline]
  #[must_use]
  pub(crate) fn stream_kernel(self) -> Kernel {
    self.stream_kernel
  }

  #[inline]
  #[must_use]
  pub(crate) fn bulk_kernel_for_update(self, input_len: usize) -> Kernel {
    if input_len >= self.bulk_sizeclass_threshold {
      self.size_classes.select(input_len)
    } else {
      self.table_bulk_kernel
    }
  }

  #[inline]
  #[must_use]
  pub(crate) fn size_class_kernel(self, len: usize) -> Kernel {
    self.size_classes.select(len)
  }

  #[inline]
  #[must_use]
  pub(crate) fn size_class_kernel_plain(self, len: usize) -> Kernel {
    self.plain_size_classes.select(len)
  }

  #[inline]
  #[must_use]
  pub(crate) fn should_defer_simd(self, buffered_len: usize, incoming_len: usize) -> bool {
    buffered_len.saturating_add(incoming_len) < self.simd_threshold
  }

  #[inline]
  #[must_use]
  pub(crate) fn parallel_streaming(self) -> ParallelTable {
    self.parallel_streaming
  }

  #[inline]
  #[must_use]
  pub(crate) fn bulk_sizeclass_threshold(self) -> usize {
    self.bulk_sizeclass_threshold
  }
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
  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  let _ = caps;

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
      } else {
        Blake3KernelId::Portable
      }
    }
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Sse41 => {
      if caps.has(required_caps(Blake3KernelId::X86Sse41)) {
        Blake3KernelId::X86Sse41
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
    #[cfg(target_arch = "s390x")]
    Blake3KernelId::S390xVector => {
      if caps.has(required_caps(Blake3KernelId::S390xVector)) {
        Blake3KernelId::S390xVector
      } else {
        Blake3KernelId::Portable
      }
    }
    #[cfg(target_arch = "powerpc64")]
    Blake3KernelId::PowerVsx => {
      if caps.has(required_caps(Blake3KernelId::PowerVsx)) {
        Blake3KernelId::PowerVsx
      } else {
        Blake3KernelId::Portable
      }
    }
    #[cfg(target_arch = "riscv64")]
    Blake3KernelId::RiscvV => {
      if caps.has(required_caps(Blake3KernelId::RiscvV)) {
        Blake3KernelId::RiscvV
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

    let active = ActiveDispatch {
      boundaries: table.boundaries,
      xs: Entry { kernel: kernel(xs_id) },
      s: Entry { kernel: kernel(s_id) },
      m: Entry { kernel: kernel(m_id) },
      l: Entry { kernel: kernel(l_id) },
    };
    let streaming = StreamingDispatch {
      stream: kernel(stream_id),
      bulk: kernel(bulk_id),
    };
    let oneshot_base = *oneshot_parallel_table;
    let streaming_base = *streaming_parallel_table;
    // Keep runtime policy selection table-driven only. Mode-specific parallel
    // behavior must come from tuned/applied tables, not post-hoc heuristics.
    let parallel = ParallelDispatch {
      oneshot: oneshot_base,
      keyed_oneshot: oneshot_base,
      derive_oneshot: oneshot_base,
      xof: oneshot_base,
      keyed_xof: oneshot_base,
      derive_xof: oneshot_base,
      streaming: streaming_base,
      keyed_streaming: streaming_base,
      derive_streaming: streaming_base,
    };
    let size_classes = SizeClassDispatch {
      boundaries: active.boundaries,
      xs: active.xs.kernel,
      s: active.s.kernel,
      m: active.m.kernel,
      l: active.l.kernel,
    };
    let plain_boundaries = {
      #[cfg(target_arch = "x86_64")]
      {
        let mut b = active.boundaries;
        // Plain-mode short policy split on AVX-512 Intel server families:
        // keep AVX2 for 256/512, but move 1024 to AVX-512.
        if matches!(kind, TuneKind::IntelSpr | TuneKind::IntelIcl)
          && b[1] >= 1024
          && active.s.kernel.id == Blake3KernelId::X86Avx2
          && active.m.kernel.id == Blake3KernelId::X86Avx512
        {
          b[1] = 512;
        }
        b
      }
      #[cfg(not(target_arch = "x86_64"))]
      {
        active.boundaries
      }
    };
    let plain_size_classes = SizeClassDispatch {
      boundaries: plain_boundaries,
      xs: active.xs.kernel,
      s: active.s.kernel,
      m: active.m.kernel,
      l: active.l.kernel,
    };
    let hasher = HasherDispatch {
      size_classes,
      plain_size_classes,
      stream_kernel: streaming.stream,
      table_bulk_kernel: streaming.bulk,
      parallel_streaming: parallel.streaming,
      simd_threshold: tune.simd_threshold,
      bulk_sizeclass_threshold: stream_table.bulk_sizeclass_threshold,
    };

    ResolvedDispatch {
      active,
      streaming,
      parallel,
      simd_threshold: tune.simd_threshold,
      hasher,
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
  let output = super::root_output_oneshot(kernel, super::IV, 0, super::policy_kind_from_flags(0, true), data);
  super::Blake3Xof::new(output, hasher_dispatch())
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
pub(crate) fn hasher_dispatch() -> HasherDispatch {
  resolved().hasher
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
  pub parallel_spawn_cost_bytes: usize,
  pub parallel_merge_cost_bytes: usize,
  pub parallel_bytes_per_core_small: usize,
  pub parallel_bytes_per_core_medium: usize,
  pub parallel_bytes_per_core_large: usize,
  pub parallel_small_limit_bytes: usize,
  pub parallel_medium_limit_bytes: usize,
  pub would_parallelize: bool,
  pub parallel_threads: usize,
}

/// Return the dispatch decisions that `Blake3::update()` would make for a
/// single call with the given `flags` and `input_len`.
///
/// Bench-only introspection hook — not part of the stable API.
#[doc(hidden)]
#[must_use]
pub fn streaming_dispatch_info(flags: u32, input_len: usize) -> StreamingDispatchInfo {
  let pd = active_parallel();
  let ptable = match flags {
    FLAGS_KEYED_HASH => pd.keyed_streaming,
    FLAGS_DERIVE_KEY_MATERIAL => pd.derive_streaming,
    _ => pd.streaming,
  };

  // Mirror the lazy-SIMD gate in `update()`: plain hashes stay portable for
  // tiny inputs below the platform's simd_threshold.
  let is_plain = flags == 0;
  let hd = hasher_dispatch();
  let (stream_kernel, bulk_kernel) = if is_plain && input_len < streaming_simd_threshold() {
    ("portable", "portable")
  } else {
    let sd = active_streaming();
    let bulk = if input_len >= hd.bulk_sizeclass_threshold() {
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
  let (would_parallelize, parallel_threads) = parallel_threads_for(&ptable, input_len, commit_chunks);

  StreamingDispatchInfo {
    stream_kernel,
    bulk_kernel,
    parallel_min_bytes: ptable.min_bytes,
    parallel_min_chunks: ptable.min_chunks,
    parallel_max_threads: ptable.max_threads,
    parallel_spawn_cost_bytes: ptable.spawn_cost_bytes,
    parallel_merge_cost_bytes: ptable.merge_cost_bytes,
    parallel_bytes_per_core_small: ptable.bytes_per_core_small,
    parallel_bytes_per_core_medium: ptable.bytes_per_core_medium,
    parallel_bytes_per_core_large: ptable.bytes_per_core_large,
    parallel_small_limit_bytes: ptable.small_limit_bytes,
    parallel_medium_limit_bytes: ptable.medium_limit_bytes,
    would_parallelize,
    parallel_threads,
  }
}

#[inline]
#[must_use]
fn bytes_per_core_for_payload(table: &ParallelTable, input_bytes: usize) -> usize {
  if input_bytes <= table.small_limit_bytes {
    table.bytes_per_core_small
  } else if input_bytes <= table.medium_limit_bytes {
    table.bytes_per_core_medium
  } else {
    table.bytes_per_core_large
  }
}

/// Compute (would_parallelize, thread_count) mirroring runtime policy checks.
fn parallel_threads_for(table: &ParallelTable, input_bytes: usize, commit_chunks: usize) -> (bool, usize) {
  if table.max_threads == 1 {
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
    let bytes_per_core = bytes_per_core_for_payload(table, input_bytes);
    let mut candidate = threads.min(commit_chunks);
    while candidate > 1 {
      let chunk_depth = (commit_chunks.max(2)).ilog2() as usize;
      let thread_depth = (candidate.max(2)).ilog2() as usize;
      let merge_divisor = 1 + chunk_depth + thread_depth + 1;
      let merge_cost = table.merge_cost_bytes.saturating_add(merge_divisor - 1) / merge_divisor;
      let spawn_cost = table.spawn_cost_bytes.saturating_mul(candidate - 1);
      let work_cost = bytes_per_core.saturating_mul(candidate);
      let fitted_required = merge_cost.saturating_add(spawn_cost).saturating_add(work_cost);
      let required = fitted_required.saturating_mul(15).saturating_add(10 - 1) / 10;
      if input_bytes >= required {
        return (true, candidate);
      }
      candidate -= 1;
    }
    (false, 1)
  }
  #[cfg(not(feature = "std"))]
  {
    (false, 1)
  }
}
