use backend::OnceCache;
use platform::Caps;

use super::{
  BLOCK_LEN, Sha512_256,
  dispatch_tables::DispatchTable,
  kernels::{CompressBlocksFn, Sha512_256KernelId, compress_blocks_fn, required_caps},
};
use crate::crypto::dispatch_util::SizeClassDispatch;

#[derive(Clone, Copy)]
struct Entry {
  compress_blocks: CompressBlocksFn,
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
fn resolve(id: Sha512_256KernelId, caps: Caps) -> Sha512_256KernelId {
  if caps.has(required_caps(id)) {
    id
  } else {
    Sha512_256KernelId::Portable
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
        compress_blocks: compress_blocks_fn(xs_id),
        name: xs_id.as_str(),
      },
      s: Entry {
        compress_blocks: compress_blocks_fn(s_id),
        name: s_id.as_str(),
      },
      m: Entry {
        compress_blocks: compress_blocks_fn(m_id),
        name: m_id.as_str(),
      },
      l: Entry {
        compress_blocks: compress_blocks_fn(l_id),
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
pub fn digest(data: &[u8]) -> [u8; 32] {
  use traits::Digest;
  let mut h = Sha512_256::default();
  h.update(data);
  h.finalize()
}

#[inline]
pub fn compress_blocks(state: &mut [u64; 8], blocks: &[u8]) {
  if blocks.is_empty() {
    return;
  }
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
  let d = active();
  let entry = select(&d, blocks.len());
  (entry.compress_blocks)(state, blocks);
}

#[inline]
#[must_use]
pub(crate) fn compress_dispatch() -> SizeClassDispatch<CompressBlocksFn> {
  let d = active();
  SizeClassDispatch {
    boundaries: d.boundaries,
    xs: d.xs.compress_blocks,
    s: d.s.compress_blocks,
    m: d.m.compress_blocks,
    l: d.l.compress_blocks,
  }
}
