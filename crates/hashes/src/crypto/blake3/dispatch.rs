use backend::OnceCache;
use platform::Caps;

use super::{
  dispatch_tables::DispatchTable,
  kernels::{Blake3KernelId, Kernel, kernel, required_caps},
};
use crate::crypto::dispatch_util::SizeClassDispatch;

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

#[inline]
#[must_use]
fn resolve(id: Blake3KernelId, caps: Caps) -> Blake3KernelId {
  if caps.has(required_caps(id)) {
    id
  } else {
    Blake3KernelId::Portable
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
      xs: Entry { kernel: kernel(xs_id) },
      s: Entry { kernel: kernel(s_id) },
      m: Entry { kernel: kernel(m_id) },
      l: Entry { kernel: kernel(l_id) },
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
  use traits::Digest;
  let mut h = super::Blake3::default();
  h.update(data);
  h.finalize()
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
