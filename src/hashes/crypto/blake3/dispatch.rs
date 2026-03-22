#[cfg(feature = "parallel")]
use super::dispatch_tables::ParallelTable;
#[cfg(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "s390x",
  target_arch = "powerpc64",
  target_arch = "riscv64"
))]
use super::kernels::required_caps;
use super::{
  dispatch_tables::{DispatchTable, StreamingTable},
  kernels::{Blake3KernelId, Kernel, kernel},
};
#[cfg(target_arch = "x86_64")]
use crate::platform::caps::x86;
use crate::{backend::OnceCache, hashes::crypto::dispatch_util::SizeClassDispatch, platform::Caps};

#[cfg(target_arch = "x86_64")]
#[inline]
#[must_use]
fn allow_avx2_hash_many_one_chunk_fast_path(caps: Caps) -> bool {
  caps.has(x86::AVX512_READY)
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
  #[cfg(feature = "parallel")]
  parallel: ParallelDispatch,
  hasher: HasherDispatch,
  #[cfg(target_arch = "x86_64")]
  avx2_hash_many_one_chunk_fast_path: bool,
  #[cfg(target_arch = "x86_64")]
  avx2_available: bool,
}

static RESOLVED: OnceCache<ResolvedDispatch> = OnceCache::new();

#[derive(Clone, Copy)]
pub(crate) struct StreamingDispatch {
  pub(crate) stream: Kernel,
  pub(crate) bulk: Kernel,
}

#[derive(Clone, Copy)]
#[cfg(feature = "parallel")]
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
/// This is resolved once from platform caps and dispatch tables and can be copied into each
/// hasher, avoiding repeated global dispatch lookups in hot update/finalize
/// paths.
#[derive(Clone, Copy)]
pub(crate) struct HasherDispatch {
  size_classes: SizeClassDispatch<Kernel>,
  stream_kernel: Kernel,
  table_bulk_kernel: Kernel,
  bulk_sizeclass_threshold: usize,
}

impl HasherDispatch {
  #[inline]
  #[must_use]
  pub(crate) fn stream_kernel(&self) -> Kernel {
    self.stream_kernel
  }

  #[inline]
  #[must_use]
  pub(crate) fn bulk_kernel_for_update(&self, input_len: usize) -> Kernel {
    if input_len >= self.bulk_sizeclass_threshold {
      self.size_classes.select(input_len)
    } else {
      self.table_bulk_kernel
    }
  }

  #[inline]
  #[must_use]
  pub(crate) fn size_class_kernel(&self, len: usize) -> Kernel {
    self.size_classes.select(len)
  }

  #[inline]
  #[must_use]
  #[cfg(feature = "parallel")]
  pub(crate) fn bulk_sizeclass_threshold(&self) -> usize {
    self.bulk_sizeclass_threshold
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
    let caps = crate::hashes::util::dispatch_caps();

    let table: &'static DispatchTable = super::dispatch_tables::select_table_for_caps(caps);
    let stream_table: &'static StreamingTable = super::dispatch_tables::select_streaming_table_for_caps(caps);
    #[cfg(feature = "parallel")]
    let oneshot_parallel_table: &'static ParallelTable = super::dispatch_tables::select_parallel_table_for_caps(caps);
    #[cfg(feature = "parallel")]
    let streaming_parallel_table: &'static ParallelTable =
      super::dispatch_tables::select_streaming_parallel_table_for_caps(caps);

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
    #[cfg(feature = "parallel")]
    let oneshot_base = *oneshot_parallel_table;
    #[cfg(feature = "parallel")]
    let streaming_base = *streaming_parallel_table;
    let size_classes = SizeClassDispatch {
      boundaries: active.boundaries,
      xs: active.xs.kernel,
      s: active.s.kernel,
      m: active.m.kernel,
      l: active.l.kernel,
    };
    let hasher = HasherDispatch {
      size_classes,
      stream_kernel: streaming.stream,
      table_bulk_kernel: streaming.bulk,
      bulk_sizeclass_threshold: stream_table.bulk_sizeclass_threshold,
    };

    ResolvedDispatch {
      active,
      streaming,
      #[cfg(feature = "parallel")]
      parallel: ParallelDispatch {
        oneshot: oneshot_base,
        keyed_oneshot: oneshot_base,
        derive_oneshot: oneshot_base,
        xof: oneshot_base,
        keyed_xof: oneshot_base,
        derive_xof: oneshot_base,
        streaming: streaming_base,
        keyed_streaming: streaming_base,
        derive_streaming: streaming_base,
      },
      hasher,
      #[cfg(target_arch = "x86_64")]
      avx2_hash_many_one_chunk_fast_path: allow_avx2_hash_many_one_chunk_fast_path(caps),
      #[cfg(target_arch = "x86_64")]
      avx2_available: caps.has(required_caps(Blake3KernelId::X86Avx2)),
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

#[cfg(feature = "parallel")]
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
pub(crate) fn kernel_for_len(len: usize) -> Kernel {
  let d = active();
  select(&d, len).kernel
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

  // Lean path for single-chunk inputs: directly construct Blake3Xof without
  // going through root_output_oneshot / single_chunk_output / OutputState.
  if data.len() <= super::CHUNK_LEN {
    return super::xof_oneshot_single_chunk(kernel, super::IV, 0, data);
  }

  let output = super::root_output_oneshot(
    kernel,
    super::IV,
    0,
    super::control::policy_kind_from_flags(0, true),
    data,
  );
  super::Blake3Xof::from_output(output)
}

#[inline]
#[must_use]
#[cfg(any(test, feature = "std"))]
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

#[cfg(feature = "parallel")]
#[inline]
#[must_use]
pub(crate) fn parallel_dispatch() -> ParallelDispatch {
  active_parallel()
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[must_use]
pub(crate) fn avx2_hash_many_one_chunk_fast_path() -> bool {
  resolved().avx2_hash_many_one_chunk_fast_path
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[must_use]
pub(crate) fn avx2_available() -> bool {
  resolved().avx2_available
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
#[cfg(feature = "parallel")]
pub fn streaming_dispatch_info(flags: u32, input_len: usize) -> StreamingDispatchInfo {
  use super::control;
  let hd = hasher_dispatch();
  let sd = active_streaming();
  let bulk_kernel = if input_len >= hd.bulk_sizeclass_threshold() {
    kernel_dispatch().select(input_len).name
  } else {
    sd.bulk.name
  };
  let stream_kernel = sd.stream.name;

  // Mirror `parallel_policy_threads` + the per-thread work guard
  // from `update_with`.
  const CHUNK_LEN: usize = 1024;

  let mode = control::streaming_policy_kind_from_flags(flags);
  let ptable = control::resolved_parallel_policy(mode);
  let commit_chunks = input_len / CHUNK_LEN;
  let decision = control::parallel_admission_decision(mode, ptable, input_len, commit_chunks, commit_chunks);

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
    would_parallelize: decision.would_parallelize,
    parallel_threads: decision.threads,
  }
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
  use super::streaming_dispatch_info;
  use crate::hashes::crypto::blake3::{CHUNK_LEN, DERIVE_KEY_MATERIAL, KEYED_HASH, control};

  #[test]
  fn streaming_dispatch_info_matches_runtime_parallel_admission() {
    for &flags in &[0u32, KEYED_HASH, DERIVE_KEY_MATERIAL] {
      for &input_len in &[
        0usize,
        1,
        CHUNK_LEN - 1,
        CHUNK_LEN,
        CHUNK_LEN * 2,
        4 * 1024,
        16 * 1024,
        64 * 1024,
        1024 * 1024,
      ] {
        let info = streaming_dispatch_info(flags, input_len);
        let commit_chunks = input_len / CHUNK_LEN;
        let expected = control::streaming_parallel_threads_for_flags(flags, input_len, commit_chunks, commit_chunks);

        assert_eq!(
          info.would_parallelize,
          expected.is_some(),
          "parallel admission mismatch for flags={flags:#x}, len={input_len}"
        );
        assert_eq!(
          info.parallel_threads,
          expected.unwrap_or(1),
          "thread count mismatch for flags={flags:#x}, len={input_len}"
        );
      }
    }
  }
}
