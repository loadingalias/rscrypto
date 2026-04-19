#[cfg(any(test, feature = "diag"))]
use super::kernels::hash64_fn;
use super::{
  dispatch_tables::DispatchTable,
  kernels::{Xxh3KernelId, hash64_long_fn, hash128_long_fn, required_caps},
};
use crate::{backend::cache::OnceCache, platform::Caps};

type Hash64Fn = fn(&[u8], u64) -> u64;
type Hash128Fn = fn(&[u8], u64) -> u128;

#[derive(Clone, Copy)]
struct ActiveDispatch {
  /// Long-path-only entry for 64-bit hash (>240B, no redundant length checks).
  long64: Hash64Fn,
  /// Long-path-only entry for 128-bit hash (>240B, no redundant length checks).
  long128: Hash128Fn,
  #[cfg(any(test, feature = "diag"))]
  boundaries: [usize; 3],
  #[cfg(any(test, feature = "diag"))]
  xs64: Hash64Fn,
  #[cfg(any(test, feature = "diag"))]
  s64: Hash64Fn,
  #[cfg(any(test, feature = "diag"))]
  m64: Hash64Fn,
  #[cfg(any(test, feature = "diag"))]
  l64: Hash64Fn,
  #[cfg(any(test, feature = "diag"))]
  xs_name: &'static str,
  #[cfg(any(test, feature = "diag"))]
  s_name: &'static str,
  #[cfg(any(test, feature = "diag"))]
  m_name: &'static str,
  #[cfg(any(test, feature = "diag"))]
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

    #[cfg(any(test, feature = "diag"))]
    let xs_id = resolve(table.xs, caps);
    #[cfg(any(test, feature = "diag"))]
    let s_id = resolve(table.s, caps);
    #[cfg(any(test, feature = "diag"))]
    let m_id = resolve(table.m, caps);
    let l_id = resolve(table.l, caps);

    ActiveDispatch {
      long64: hash64_long_fn(l_id),
      long128: hash128_long_fn(l_id),
      #[cfg(any(test, feature = "diag"))]
      boundaries: table.boundaries,
      #[cfg(any(test, feature = "diag"))]
      xs64: hash64_fn(xs_id),
      #[cfg(any(test, feature = "diag"))]
      s64: hash64_fn(s_id),
      #[cfg(any(test, feature = "diag"))]
      m64: hash64_fn(m_id),
      #[cfg(any(test, feature = "diag"))]
      l64: hash64_fn(l_id),
      #[cfg(any(test, feature = "diag"))]
      xs_name: xs_id.as_str(),
      #[cfg(any(test, feature = "diag"))]
      s_name: s_id.as_str(),
      #[cfg(any(test, feature = "diag"))]
      m_name: m_id.as_str(),
      #[cfg(any(test, feature = "diag"))]
      l_name: l_id.as_str(),
    }
  })
}

#[cfg(any(test, feature = "diag"))]
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

#[cfg(any(test, feature = "diag"))]
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
/// The long-path fallback is `#[cold]` to keep the caller's ≤240B paths tight
/// in the µop cache.
#[inline(always)]
#[must_use]
pub fn hash64_with_seed(seed: u64, data: &[u8]) -> u64 {
  if data.is_empty() {
    // Bypass the 0..16B branch ladder on the hottest small-input case.
    return super::xxh3_64_0to16(data, seed, &super::DEFAULT_SECRET);
  }
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
/// Falls back to runtime dispatch when features are unknown at compile time,
/// using the dedicated long-path entry point that skips redundant ≤240B length
/// checks in the kernel.
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

  // RISC-V: RVV kernel is slower than portable scalar at 256 B–64 KiB on
  // in-order cores (SpacemiT K1).  Use the portable long path directly,
  // bypassing OnceCache + indirect call overhead that the in-order pipeline
  // pays dearly for (atomic acquire fence + unpredicted indirect branch).
  #[cfg(target_arch = "riscv64")]
  {
    return super::xxh3_64_long(data, seed);
  }

  // Tier 2: runtime dispatch — dedicated long-path fn pointer, no redundant
  // length checks.
  #[allow(unreachable_code)]
  {
    let d = active();
    (d.long64)(data, seed)
  }
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

  // RISC-V: see hash64_long comment.
  #[cfg(target_arch = "riscv64")]
  {
    return super::xxh3_128_long(data, seed);
  }

  // Tier 2: runtime dispatch — dedicated long-path fn pointer.
  #[allow(unreachable_code)]
  {
    let d = active();
    (d.long128)(data, seed)
  }
}
