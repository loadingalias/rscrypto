use backend::OnceCache;
use platform::Caps;

use super::{
  dispatch_tables::DispatchTable,
  kernels::{AsconPermute12KernelId, permute_fn, required_caps},
};
use crate::crypto::dispatch_util::SizeClassDispatch;

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
    let tune = platform::tune();
    let caps = platform::caps();
    let table: &'static DispatchTable = super::dispatch_tables::select_table(tune.kind);

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
#[inline]
pub fn permute_12_for_len(state: &mut [u64; 5], len: usize) {
  let d = active();
  (select(&d, len).0)(state);
}

#[inline]
#[must_use]
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

#[inline]
pub fn permute_12(state: &mut [u64; 5]) {
  let d = active();
  (d.l)(state);
}
