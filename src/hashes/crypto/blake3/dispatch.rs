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
use crate::{backend::cache::OnceCache, hashes::crypto::dispatch_util::SizeClassDispatch, platform::Caps};

#[cfg(target_arch = "x86_64")]
#[inline]
#[must_use]
fn allow_avx2_hash_many_one_chunk_fast_path(caps: Caps) -> bool {
  caps.has(x86::AVX512_READY) && !caps.has(x86::INTEL_SAPPHIRE_RAPIDS)
}

/// Return the configured four-block policy class for x86-64.
#[cfg(target_arch = "x86_64")]
#[inline]
#[must_use]
fn is_wide_pipeline_for_hash_many(caps: Caps) -> bool {
  // Zen 5 and Intel AVX-512 use the wide-pipeline policy; earlier AMD uses
  // the alternate four-block policy.
  if caps.has(x86::AMD) {
    caps.has(x86::AMD_ZEN5)
  } else {
    // Intel: all AVX-512 without AMX (ICL, TGL) are wide-pipeline.
    true
  }
}

#[derive(Clone, Copy)]
struct ActiveDispatch {
  size_classes: SizeClassDispatch<Blake3KernelId>,
}

static ACTIVE: OnceCache<ActiveDispatch> = OnceCache::new();
static HASHER: OnceCache<HasherDispatch> = OnceCache::new();

#[cfg(feature = "parallel")]
static PARALLEL: OnceCache<ParallelDispatch> = OnceCache::new();

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct X86Policy {
  avx2_hash_many_one_chunk_fast_path: bool,
  hash_many_wide_pipeline: bool,
  #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
  avx2_available: bool,
}

#[cfg(target_arch = "x86_64")]
static X86_POLICY: OnceCache<X86Policy> = OnceCache::new();

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
fn active() -> ActiveDispatch {
  ACTIVE.get_or_init(|| {
    let caps = crate::platform::caps();
    let table: &'static DispatchTable = super::dispatch_tables::select_table_for_caps(caps);
    ActiveDispatch {
      size_classes: SizeClassDispatch {
        boundaries: table.boundaries,
        xs: resolve(table.xs, caps),
        s: resolve(table.s, caps),
        m: resolve(table.m, caps),
        l: resolve(table.l, caps),
      },
    }
  })
}

#[cfg(feature = "parallel")]
#[inline]
#[must_use]
fn active_parallel() -> ParallelDispatch {
  PARALLEL.get_or_init(|| {
    let caps = crate::platform::caps();
    let oneshot = *super::dispatch_tables::select_parallel_table_for_caps(caps);
    let streaming = *super::dispatch_tables::select_streaming_parallel_table_for_caps(caps);
    ParallelDispatch {
      oneshot,
      keyed_oneshot: oneshot,
      derive_oneshot: oneshot,
      xof: oneshot,
      keyed_xof: oneshot,
      derive_xof: oneshot,
      streaming,
      keyed_streaming: streaming,
      derive_streaming: streaming,
    }
  })
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[must_use]
fn x86_policy() -> X86Policy {
  X86_POLICY.get_or_init(|| {
    let caps = crate::platform::caps();
    X86Policy {
      avx2_hash_many_one_chunk_fast_path: allow_avx2_hash_many_one_chunk_fast_path(caps),
      hash_many_wide_pipeline: is_wide_pipeline_for_hash_many(caps),
      #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
      avx2_available: caps.has(required_caps(Blake3KernelId::X86Avx2)),
    }
  })
}

#[inline]
#[must_use]
pub(crate) fn size_class_kernel(len: usize) -> Kernel {
  kernel(active().size_classes.select(len))
}

#[cfg(feature = "diag")]
#[inline]
#[must_use]
pub(crate) fn kernel_name_for_len(len: usize) -> &'static str {
  size_class_kernel(len).name
}

#[inline]
#[must_use]
pub(crate) fn xof(data: &[u8]) -> super::Blake3XofReader {
  let kernel = size_class_kernel(data.len());

  // Lean path for single-chunk inputs: directly construct Blake3XofReader without
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
  super::Blake3XofReader::from_output(output)
}

#[inline]
#[must_use]
pub(crate) fn hasher_dispatch() -> HasherDispatch {
  HASHER.get_or_init(|| {
    let caps = crate::platform::caps();
    let active = active();
    let stream_table: &'static StreamingTable = super::dispatch_tables::select_streaming_table_for_caps(caps);
    let ids = active.size_classes;
    HasherDispatch {
      size_classes: SizeClassDispatch {
        boundaries: ids.boundaries,
        xs: kernel(ids.xs),
        s: kernel(ids.s),
        m: kernel(ids.m),
        l: kernel(ids.l),
      },
      stream_kernel: kernel(resolve(stream_table.stream, caps)),
      table_bulk_kernel: kernel(resolve(stream_table.bulk, caps)),
      bulk_sizeclass_threshold: stream_table.bulk_sizeclass_threshold,
    }
  })
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
  x86_policy().avx2_hash_many_one_chunk_fast_path
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[must_use]
pub(crate) fn hash_many_wide_pipeline() -> bool {
  x86_policy().hash_many_wide_pipeline
}

#[cfg(all(
  target_arch = "x86_64",
  any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[inline]
#[must_use]
pub(crate) fn avx2_available() -> bool {
  x86_policy().avx2_available
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn one_shot_and_hasher_size_classes_match() {
    let hasher = hasher_dispatch();
    for len in [0, 1, 63, 64, 65, 256, 257, 4096, 4097, usize::MAX] {
      assert_eq!(size_class_kernel(len).id, hasher.size_class_kernel(len).id);
    }
  }

  #[cfg(target_arch = "x86_64")]
  const ALL_AMX: Caps = x86::AMX_TILE
    .union(x86::AMX_BF16)
    .union(x86::AMX_INT8)
    .union(x86::AMX_FP16)
    .union(x86::AMX_COMPLEX);

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn sapphire_rapids_shortcut_policy_does_not_depend_on_amx_permission() {
    let sapphire_rapids = x86::AVX512_READY | x86::INTEL_SAPPHIRE_RAPIDS;
    assert!(!allow_avx2_hash_many_one_chunk_fast_path(sapphire_rapids));
    assert!(!allow_avx2_hash_many_one_chunk_fast_path(sapphire_rapids | ALL_AMX));

    assert!(allow_avx2_hash_many_one_chunk_fast_path(x86::AVX512_READY));
    assert!(allow_avx2_hash_many_one_chunk_fast_path(x86::AVX512_READY | ALL_AMX));
  }
}
