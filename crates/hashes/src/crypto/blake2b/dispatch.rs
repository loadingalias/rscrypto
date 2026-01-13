use backend::OnceCache;
use platform::Caps;

use super::{
  BLOCK_LEN, Blake2b512,
  dispatch_tables::DispatchTable,
  kernels::{Blake2b512KernelId, CompressFn, compress_fn, required_caps},
};
use crate::crypto::dispatch_util::SizeClassDispatch;

#[derive(Clone, Copy)]
struct Entry {
  compress: CompressFn,
  name: &'static str,
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
fn resolve(id: Blake2b512KernelId, caps: Caps) -> Blake2b512KernelId {
  if caps.has(required_caps(id)) {
    id
  } else {
    Blake2b512KernelId::Portable
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
      xs: Entry {
        compress: compress_fn(xs_id),
        name: xs_id.as_str(),
      },
      s: Entry {
        compress: compress_fn(s_id),
        name: s_id.as_str(),
      },
      m: Entry {
        compress: compress_fn(m_id),
        name: m_id.as_str(),
      },
      l: Entry {
        compress: compress_fn(l_id),
        name: l_id.as_str(),
      },
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
  select(&d, len).name
}

#[inline]
#[must_use]
pub fn digest(data: &[u8]) -> [u8; 64] {
  use traits::Digest;
  let mut h = Blake2b512::default();
  h.update(data);
  h.finalize()
}

#[inline]
pub fn compress(h: &mut [u64; 8], blocks: &[u8], bytes_hashed: &mut u128, is_last: bool, last_block_len: u32) {
  if blocks.is_empty() {
    return;
  }
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
  if is_last {
    debug_assert_eq!(blocks.len(), BLOCK_LEN);
    debug_assert!(last_block_len as usize <= BLOCK_LEN);
  }
  let d = active();
  let entry = select(&d, blocks.len());
  (entry.compress)(h, blocks, bytes_hashed, is_last, last_block_len);
}

#[inline]
#[must_use]
pub(crate) fn compress_dispatch() -> SizeClassDispatch<CompressFn> {
  let d = active();
  SizeClassDispatch {
    boundaries: d.boundaries,
    xs: d.xs.compress,
    s: d.s.compress,
    m: d.m.compress,
    l: d.l.compress,
  }
}
