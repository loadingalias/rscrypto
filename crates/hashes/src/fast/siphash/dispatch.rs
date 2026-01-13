use backend::OnceCache;
use platform::Caps;

use super::{
  dispatch_tables::DispatchTable,
  kernels::{SipHashKernelId, hash13_fn, hash24_fn, required_caps},
};

type HashFn = fn([u64; 2], &[u8]) -> u64;

#[derive(Clone, Copy)]
struct ActiveDispatch {
  boundaries: [usize; 3],
  xs13: HashFn,
  s13: HashFn,
  m13: HashFn,
  l13: HashFn,
  xs24: HashFn,
  s24: HashFn,
  m24: HashFn,
  l24: HashFn,
  xs_name: &'static str,
  s_name: &'static str,
  m_name: &'static str,
  l_name: &'static str,
}

static ACTIVE: OnceCache<ActiveDispatch> = OnceCache::new();

#[inline]
#[must_use]
fn resolve(id: SipHashKernelId, caps: Caps) -> SipHashKernelId {
  if caps.has(required_caps(id)) {
    id
  } else {
    SipHashKernelId::Portable
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
      xs13: hash13_fn(xs_id),
      s13: hash13_fn(s_id),
      m13: hash13_fn(m_id),
      l13: hash13_fn(l_id),
      xs24: hash24_fn(xs_id),
      s24: hash24_fn(s_id),
      m24: hash24_fn(m_id),
      l24: hash24_fn(l_id),
      xs_name: xs_id.as_str(),
      s_name: s_id.as_str(),
      m_name: m_id.as_str(),
      l_name: l_id.as_str(),
    }
  })
}

#[inline]
#[must_use]
fn select13(d: &ActiveDispatch, len: usize) -> (HashFn, &'static str) {
  let [xs_max, s_max, m_max] = d.boundaries;
  if len <= xs_max {
    (d.xs13, d.xs_name)
  } else if len <= s_max {
    (d.s13, d.s_name)
  } else if len <= m_max {
    (d.m13, d.m_name)
  } else {
    (d.l13, d.l_name)
  }
}

#[inline]
#[must_use]
fn select24(d: &ActiveDispatch, len: usize) -> (HashFn, &'static str) {
  let [xs_max, s_max, m_max] = d.boundaries;
  if len <= xs_max {
    (d.xs24, d.xs_name)
  } else if len <= s_max {
    (d.s24, d.s_name)
  } else if len <= m_max {
    (d.m24, d.m_name)
  } else {
    (d.l24, d.l_name)
  }
}

#[inline]
#[must_use]
pub fn kernel_name_for_len(len: usize) -> &'static str {
  let d = active();
  select13(&d, len).1
}

#[inline]
#[must_use]
pub fn hash13_with_seed(seed: [u64; 2], data: &[u8]) -> u64 {
  let d = active();
  let (f, _) = select13(&d, data.len());
  f(seed, data)
}

#[inline]
#[must_use]
pub fn hash24_with_seed(seed: [u64; 2], data: &[u8]) -> u64 {
  let d = active();
  let (f, _) = select24(&d, data.len());
  f(seed, data)
}
