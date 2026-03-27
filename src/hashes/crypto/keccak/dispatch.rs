use super::{
  dispatch_tables::DispatchTable,
  kernels::{Keccakf1600KernelId, permute_fn, required_caps},
};
use crate::{backend::cache::OnceCache, hashes::crypto::dispatch_util::SizeClassDispatch, platform::Caps};

type PermuteFn = fn(&mut [u64; 25]);

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
fn resolve(id: Keccakf1600KernelId, caps: Caps) -> Keccakf1600KernelId {
  if caps.has(required_caps(id)) {
    id
  } else {
    Keccakf1600KernelId::Portable
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

#[inline]
#[must_use]
pub fn kernel_name_for_len(len: usize) -> &'static str {
  #[cfg(target_arch = "s390x")]
  {
    use crate::platform::caps::s390x;
    if crate::platform::caps().has(s390x::MSA8) {
      return "s390x-kimd";
    }
  }
  let d = active();
  select(&d, len).1
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
