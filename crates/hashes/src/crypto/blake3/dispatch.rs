use backend::OnceCache;
use platform::Caps;
#[cfg(target_arch = "x86_64")]
use platform::TuneKind;
#[cfg(target_arch = "x86_64")]
use platform::caps::x86;

use super::{
  dispatch_tables::{DispatchTable, StreamingTable},
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

static ACTIVE: OnceCache<ActiveDispatch> = OnceCache::new();
static ACTIVE_STREAMING: OnceCache<StreamingDispatch> = OnceCache::new();

#[derive(Clone, Copy)]
pub(crate) struct StreamingDispatch {
  pub(crate) stream: Kernel,
  pub(crate) bulk: Kernel,
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
fn active() -> ActiveDispatch {
  ACTIVE.get_or_init(|| {
    let tune = platform::tune();
    let caps = platform::caps();
    let kind = {
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
        tune.kind
      }
    };
    let table: &'static DispatchTable = super::dispatch_tables::select_table(kind);

    let xs_id = resolve(table.xs, caps);
    let s_id = resolve(table.s, caps);
    let m_id = resolve(table.m, caps);
    let l_id = resolve(table.l, caps);

    ActiveDispatch {
      boundaries: table.boundaries,
      xs: Entry { kernel: kernel(xs_id) },
      s: Entry { kernel: kernel(s_id) },
      m: Entry { kernel: kernel(m_id) },
      l: Entry { kernel: kernel(l_id) },
    }
  })
}

#[inline]
#[must_use]
fn active_streaming() -> StreamingDispatch {
  ACTIVE_STREAMING.get_or_init(|| {
    let tune = platform::tune();
    let caps = platform::caps();
    let kind = {
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
        tune.kind
      }
    };
    let table: &'static StreamingTable = super::dispatch_tables::select_streaming_table(kind);

    let stream_id = resolve(table.stream, caps);
    let bulk_id = resolve(table.bulk, caps);

    StreamingDispatch {
      stream: kernel(stream_id),
      bulk: kernel(bulk_id),
    }
  })
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
pub fn streaming_kernel_names() -> (&'static str, &'static str) {
  let d = active_streaming();
  (d.stream.name, d.bulk.name)
}
