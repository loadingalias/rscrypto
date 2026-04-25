#![cfg_attr(test, allow(dead_code))]

use super::{
  dispatch_tables::DispatchTable,
  kernels::{AsconPermute12KernelId, permute_fn, required_caps},
};
#[cfg(any(test, feature = "std"))]
use crate::hashes::crypto::dispatch_util::SizeClassDispatch;
use crate::{backend::cache::OnceCache, platform::Caps};

type PermuteFn = fn(&mut [u64; 5]);

#[derive(Clone, Copy)]
struct ActiveDispatch {
  boundaries: [usize; 3],
  xs: PermuteFn,
  s: PermuteFn,
  m: PermuteFn,
  l: PermuteFn,
  xs_name: &'static str,
  s_name: &'static str,
  m_name: &'static str,
  l_name: &'static str,
}

static ACTIVE: OnceCache<ActiveDispatch> = OnceCache::new();

#[inline]
#[must_use]
fn resolve(id: AsconPermute12KernelId, caps: Caps) -> AsconPermute12KernelId {
  if caps.has(required_caps(id)) {
    id
  } else {
    AsconPermute12KernelId::Portable
  }
}

#[inline]
#[must_use]
fn active() -> ActiveDispatch {
  ACTIVE.get_or_init(|| {
    let caps = crate::platform::caps();
    let table: &'static DispatchTable = super::dispatch_tables::select_runtime_table(caps);

    let xs_id = resolve(table.xs, caps);
    let s_id = resolve(table.s, caps);
    let m_id = resolve(table.m, caps);
    let l_id = resolve(table.l, caps);

    ActiveDispatch {
      boundaries: table.boundaries,
      xs: permute_fn(xs_id),
      s: permute_fn(s_id),
      m: permute_fn(m_id),
      l: permute_fn(l_id),
      xs_name: xs_id.as_str(),
      s_name: s_id.as_str(),
      m_name: m_id.as_str(),
      l_name: l_id.as_str(),
    }
  })
}

#[inline]
#[must_use]
fn select(d: &ActiveDispatch, len: usize) -> (PermuteFn, &'static str) {
  let [xs_max, s_max, m_max] = d.boundaries;
  if len <= xs_max {
    (d.xs, d.xs_name)
  } else if len <= s_max {
    (d.s, d.s_name)
  } else if len <= m_max {
    (d.m, d.m_name)
  } else {
    (d.l, d.l_name)
  }
}

#[cfg(any(test, feature = "diag"))]
#[inline]
#[must_use]
pub fn kernel_name_for_len(len: usize) -> &'static str {
  let d = active();
  select(&d, len).1
}

/// Apply the configured Ascon permutation kernel for a specific workload size.
///
/// The `len` parameter is a hint representing the total amount of work (bytes)
/// associated with the sponge operation. This allows tuned size-class tables to
/// take effect for one-shot and long-running streaming workloads.
///
/// Production code uses `InlinePermuter` directly; this is retained for the
/// test/bench harness.
#[inline]
#[allow(dead_code)]
pub fn permute_12_for_len(state: &mut [u64; 5], len: usize) {
  let d = active();
  (select(&d, len).0)(state);
}

#[inline]
#[must_use]
#[cfg(any(test, feature = "std"))]
pub(crate) fn permute_dispatch() -> SizeClassDispatch<PermuteFn> {
  let d = active();
  SizeClassDispatch {
    boundaries: d.boundaries,
    xs: d.xs,
    s: d.s,
    m: d.m,
    l: d.l,
  }
}

#[cfg(any(test, feature = "std"))]
#[inline]
#[must_use]
pub(crate) fn scalar_kernel_id() -> AsconPermute12KernelId {
  let caps = crate::platform::caps();
  let table: &'static DispatchTable = super::dispatch_tables::select_runtime_table(caps);
  resolve(table.xs, caps)
}

#[cfg(any(test, feature = "std"))]
#[inline]
#[must_use]
pub(crate) fn batch_kernel_id_for_count(_count: usize) -> AsconPermute12KernelId {
  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
  let caps = crate::platform::caps();

  #[cfg(target_arch = "x86_64")]
  {
    use crate::platform::caps::x86;
    if _count >= 8 && caps.has(x86::AVX512F.union(x86::AVX512VL)) {
      return AsconPermute12KernelId::X86Avx512;
    }
    if _count >= 4 && caps.has(x86::AVX2) {
      return AsconPermute12KernelId::X86Avx2;
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    use crate::platform::caps::aarch64;
    if _count >= 2 && caps.has(aarch64::NEON) {
      return AsconPermute12KernelId::Aarch64Neon;
    }
  }

  scalar_kernel_id()
}

#[cfg(any(test, feature = "std"))]
#[inline]
#[must_use]
pub(crate) fn batch_fallback_kernel_id(requested: AsconPermute12KernelId, _count: usize) -> AsconPermute12KernelId {
  match requested {
    AsconPermute12KernelId::Portable => AsconPermute12KernelId::Portable,
    #[cfg(target_arch = "aarch64")]
    AsconPermute12KernelId::Aarch64Neon => {
      if _count >= 2 {
        AsconPermute12KernelId::Aarch64Neon
      } else {
        scalar_kernel_id()
      }
    }
    #[cfg(target_arch = "x86_64")]
    AsconPermute12KernelId::X86Avx2 => {
      if _count >= 4 {
        AsconPermute12KernelId::X86Avx2
      } else {
        scalar_kernel_id()
      }
    }
    #[cfg(target_arch = "x86_64")]
    AsconPermute12KernelId::X86Avx512 => {
      if _count >= 8 {
        AsconPermute12KernelId::X86Avx512
      } else {
        batch_kernel_id_for_count(_count)
      }
    }
  }
}
