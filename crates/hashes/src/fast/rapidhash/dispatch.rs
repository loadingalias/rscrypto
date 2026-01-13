use backend::OnceCache;
use platform::Caps;

use super::{
  dispatch_tables::DispatchTable,
  kernels::{RapidHashKernelId, hash64_fn, required_caps},
};

type Hash64Fn = fn(&[u8], u64) -> u64;

#[derive(Clone, Copy)]
struct ActiveDispatch {
  boundaries: [usize; 3],
  xs: Hash64Fn,
  s: Hash64Fn,
  m: Hash64Fn,
  l: Hash64Fn,
  xs_name: &'static str,
  s_name: &'static str,
  m_name: &'static str,
  l_name: &'static str,
}

static ACTIVE: OnceCache<ActiveDispatch> = OnceCache::new();

#[inline]
#[must_use]
fn resolve(id: RapidHashKernelId, caps: Caps) -> RapidHashKernelId {
  if caps.has(required_caps(id)) {
    id
  } else {
    RapidHashKernelId::Portable
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
      xs: hash64_fn(xs_id),
      s: hash64_fn(s_id),
      m: hash64_fn(m_id),
      l: hash64_fn(l_id),
      xs_name: xs_id.as_str(),
      s_name: s_id.as_str(),
      m_name: m_id.as_str(),
      l_name: l_id.as_str(),
    }
  })
}

#[inline]
#[must_use]
fn select(d: &ActiveDispatch, len: usize) -> (Hash64Fn, &'static str) {
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

#[inline]
#[must_use]
pub fn hash64_with_seed(seed: u64, data: &[u8]) -> u64 {
  let d = active();
  let (f, _) = select(&d, data.len());
  f(data, seed)
}

#[inline]
#[must_use]
pub fn hash128_with_seed(seed: u64, data: &[u8]) -> u128 {
  let d = active();
  let (f, _) = select(&d, data.len());
  // rapidhash defines a 64-bit output; derive 128-bit from two independent seeds.
  let lo = f(data, seed);
  let hi = f(data, seed ^ 0x9E37_79B9_7F4A_7C15);
  (lo as u128) | ((hi as u128) << 64)
}
