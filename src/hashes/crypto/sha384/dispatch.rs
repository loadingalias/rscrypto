use super::{
  BLOCK_LEN,
  dispatch_tables::DispatchTable,
  kernels::{CompressBlocksFn, Sha384KernelId, compress_blocks_fn, required_caps},
};
use crate::{backend::cache::OnceCache, hashes::crypto::dispatch_util::SizeClassDispatch, platform::Caps};

#[derive(Clone, Copy)]
struct Entry {
  compress_blocks: CompressBlocksFn,
  #[cfg(any(test, feature = "diag"))]
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
fn resolve(id: Sha384KernelId, caps: Caps) -> Sha384KernelId {
  if caps.has(required_caps(id)) {
    id
  } else {
    Sha384KernelId::Portable
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
      xs: Entry {
        compress_blocks: compress_blocks_fn(xs_id),
        #[cfg(any(test, feature = "diag"))]
        name: xs_id.as_str(),
      },
      s: Entry {
        compress_blocks: compress_blocks_fn(s_id),
        #[cfg(any(test, feature = "diag"))]
        name: s_id.as_str(),
      },
      m: Entry {
        compress_blocks: compress_blocks_fn(m_id),
        #[cfg(any(test, feature = "diag"))]
        name: m_id.as_str(),
      },
      l: Entry {
        compress_blocks: compress_blocks_fn(l_id),
        #[cfg(any(test, feature = "diag"))]
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

#[cfg(any(test, feature = "diag"))]
#[inline]
#[must_use]
pub fn kernel_name_for_len(len: usize) -> &'static str {
  let d = active();
  select(&d, len).name
}

#[inline]
#[must_use]
pub fn digest(data: &[u8]) -> [u8; 48] {
  let d = active();
  let compress = select(&d, data.len()).compress_blocks;
  digest_oneshot(data, compress)
}

/// Oneshot digest: processes directly from the input slice without streaming
/// state. Constructs final padded block(s) on the stack. Truncates to 48 bytes.
#[inline]
fn digest_oneshot(data: &[u8], compress_blocks: CompressBlocksFn) -> [u8; 48] {
  let mut state = super::H0;

  let (blocks, rest) = data.as_chunks::<BLOCK_LEN>();
  if !blocks.is_empty() {
    compress_blocks(&mut state, &data[..blocks.len().strict_mul(BLOCK_LEN)]);
  }

  let total_bits = (data.len() as u128) << 3;

  let mut block = [0u8; BLOCK_LEN];
  block[..rest.len()].copy_from_slice(rest);
  block[rest.len()] = 0x80;

  if rest.len() >= 112 {
    compress_blocks(&mut state, &block);
    block = [0u8; BLOCK_LEN];
  }

  block[112..128].copy_from_slice(&total_bits.to_be_bytes());
  compress_blocks(&mut state, &block);

  let mut out = [0u8; 48];
  for (chunk, &word) in out.chunks_exact_mut(8).zip(state.iter()) {
    chunk.copy_from_slice(&word.to_be_bytes());
  }
  out
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
