use super::{
  dispatch_tables::DispatchTable,
  kernels::{Xxh3KernelId, hash64_fn, hash128_fn, required_caps},
};
use crate::{backend::cache::OnceCache, platform::Caps};

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
    let caps = crate::platform::caps();
    let table: &'static DispatchTable = super::dispatch_tables::select_runtime_table(caps);

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

/// Flat size-based dispatch matching xxhash-rust's branch structure.
///
/// Every SIMD kernel delegates inputs ≤ 240 B back to the same portable scalar
/// functions, so we dispatch directly to the sub-functions here — eliminating
/// the intermediate `xxh3_64_with_seed` call and its redundant ≤`MID_SIZE_MAX`
/// guard branch.
///
/// The long-path fallback is `#[cold]` to keep the inlined code small.
#[inline(always)]
#[must_use]
pub fn hash64_with_seed(seed: u64, data: &[u8]) -> u64 {
  if data.len() <= 16 {
    return super::xxh3_64_0to16(data, seed, &super::DEFAULT_SECRET);
  }
  if data.len() <= 128 {
    return super::xxh3_64_7to128(data, seed, &super::DEFAULT_SECRET);
  }
  if data.len() <= super::MID_SIZE_MAX {
    return super::xxh3_64_129to240(data, seed, &super::DEFAULT_SECRET);
  }
  hash64_long(seed, data)
}

/// Long-path dispatch (>240B).
///
/// When target SIMD features are known at compile time (e.g., `-C target-cpu=native`),
/// calls the SIMD kernel directly — matching xxhash-rust's zero-overhead compile-time
/// dispatch model. This eliminates the `OnceCache` load + indirect function pointer
/// call that otherwise dominates at near-boundary sizes (256B).
///
/// Falls back to runtime dispatch when features are unknown at compile time.
#[cold]
#[inline(never)]
fn hash64_long(seed: u64, data: &[u8]) -> u64 {
  // Tier 1: compile-time dispatch — dedicated long entry points skip ≤240B
  // branches that are guaranteed dead at this call site.
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  {
    return super::x86_64_avx512::xxh3_64_long(data, seed);
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
  {
    return super::x86_64_avx2::xxh3_64_long(data, seed);
  }

  #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
  {
    return super::aarch64_neon::xxh3_64_long(data, seed);
  }

  // Tier 2: runtime dispatch (generic binary without known SIMD features).
  #[allow(unreachable_code)]
  {
    let d = active();
    let (f, _) = select64(&d, data.len());
    f(data, seed)
  }
}

/// Direct portable scalar path — no dispatch, no capability check.
/// For benchmarking only: isolates algorithm codegen from dispatch overhead.
#[inline(always)]
#[must_use]
pub fn hash64_portable(seed: u64, data: &[u8]) -> u64 {
  super::xxh3_64_with_seed(data, seed)
}

/// See [`hash64_with_seed`] for the dispatch rationale.
#[inline(always)]
#[must_use]
pub fn hash128_with_seed(seed: u64, data: &[u8]) -> u128 {
  if data.len() <= 16 {
    return super::xxh3_128_0to16(data, seed, &super::DEFAULT_SECRET);
  }
  if data.len() <= 128 {
    return super::xxh3_128_7to128(data, seed, &super::DEFAULT_SECRET);
  }
  if data.len() <= super::MID_SIZE_MAX {
    return super::xxh3_128_129to240(data, seed, &super::DEFAULT_SECRET);
  }
  hash128_long(seed, data)
}

/// See [`hash64_long`] for the compile-time dispatch rationale.
#[cold]
#[inline(never)]
fn hash128_long(seed: u64, data: &[u8]) -> u128 {
  // Tier 1: compile-time dispatch (dedicated long entry points).
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  {
    return super::x86_64_avx512::xxh3_128_long(data, seed);
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
  {
    return super::x86_64_avx2::xxh3_128_long(data, seed);
  }

  #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
  {
    return super::aarch64_neon::xxh3_128_long(data, seed);
  }

  // Tier 2: runtime dispatch.
  #[allow(unreachable_code)]
  {
    let d = active();
    let (f, _) = select128(&d, data.len());
    f(data, seed)
  }
}
