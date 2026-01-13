use backend::OnceCache;
use platform::Caps;

use super::{
  dispatch_tables::DispatchTable,
  kernels::{Xxh3KernelId, hash64_fn, hash128_fn, required_caps},
};

type Hash64Fn = fn(&[u8], u64) -> u64;
type Hash128Fn = fn(&[u8], u64) -> u128;

#[derive(Clone, Copy)]
struct ActiveDispatch {
  boundaries: [usize; 3],
  xs64: Hash64Fn,
  s64: Hash64Fn,
  m64: Hash64Fn,
  l64: Hash64Fn,
  xs128: Hash128Fn,
  s128: Hash128Fn,
  m128: Hash128Fn,
  l128: Hash128Fn,
  xs_name: &'static str,
  s_name: &'static str,
  m_name: &'static str,
  l_name: &'static str,
}

static ACTIVE: OnceCache<ActiveDispatch> = OnceCache::new();

#[inline]
#[must_use]
fn resolve(id: Xxh3KernelId, caps: Caps) -> Xxh3KernelId {
  if caps.has(required_caps(id)) {
    id
  } else {
    Xxh3KernelId::Portable
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
      xs64: hash64_fn(xs_id),
      s64: hash64_fn(s_id),
      m64: hash64_fn(m_id),
      l64: hash64_fn(l_id),
      xs128: hash128_fn(xs_id),
      s128: hash128_fn(s_id),
      m128: hash128_fn(m_id),
      l128: hash128_fn(l_id),
      xs_name: xs_id.as_str(),
      s_name: s_id.as_str(),
      m_name: m_id.as_str(),
      l_name: l_id.as_str(),
    }
  })
}

#[inline]
#[must_use]
fn select64(d: &ActiveDispatch, len: usize) -> (Hash64Fn, &'static str) {
  let [xs_max, s_max, m_max] = d.boundaries;
  if len <= xs_max {
    (d.xs64, d.xs_name)
  } else if len <= s_max {
    (d.s64, d.s_name)
  } else if len <= m_max {
    (d.m64, d.m_name)
  } else {
    (d.l64, d.l_name)
  }
}

#[inline]
#[must_use]
fn select128(d: &ActiveDispatch, len: usize) -> (Hash128Fn, &'static str) {
  let [xs_max, s_max, m_max] = d.boundaries;
  if len <= xs_max {
    (d.xs128, d.xs_name)
  } else if len <= s_max {
    (d.s128, d.s_name)
  } else if len <= m_max {
    (d.m128, d.m_name)
  } else {
    (d.l128, d.l_name)
  }
}

#[inline]
#[must_use]
pub fn kernel_name_for_len(len: usize) -> &'static str {
  let d = active();
  select64(&d, len).1
}

#[inline]
#[must_use]
pub fn hash64_with_seed(seed: u64, data: &[u8]) -> u64 {
  let d = active();
  let (f, _) = select64(&d, data.len());
  f(data, seed)
}

#[inline]
#[must_use]
pub fn hash128_with_seed(seed: u64, data: &[u8]) -> u128 {
  let d = active();
  let (f, _) = select128(&d, data.len());
  f(data, seed)
}
