#![cfg_attr(test, allow(dead_code))]

use super::{
  dispatch_tables::DispatchTable,
  kernels::{
    StreamAccumulateFn, Xxh3KernelId, hash64_long_fn, hash128_long_fn, required_caps,
    stream_accumulate_fn as kernel_stream_accumulate_fn,
  },
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
  stream_accumulate: StreamAccumulateFn,
  #[cfg(any(test, feature = "diag"))]
  long_id: Xxh3KernelId,
}

static ACTIVE: OnceCache<ActiveDispatch> = OnceCache::new();

#[cfg(target_arch = "x86_64")]
// Zen5 CI shows AVX-512 startup cost losing badly for XXH3-64 at 256B/1KiB,
// while larger buffers amortize it. Keep this limited to 64-bit long paths.
const ZEN5_XXH3_64_AVX2_LONG_MAX: usize = 1024;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn use_zen5_xxh3_64_avx2_short_long(len: usize) -> bool {
  len <= ZEN5_XXH3_64_AVX2_LONG_MAX
    && crate::platform::caps().has(crate::platform::caps::x86::AMD_ZEN5 | crate::platform::caps::x86::AVX2)
}

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

    let long_id = resolve(table.long, caps);

    ActiveDispatch {
      long64: hash64_long_fn(long_id),
      long128: hash128_long_fn(long_id),
      stream_accumulate: kernel_stream_accumulate_fn(long_id),
      #[cfg(any(test, feature = "diag"))]
      long_id,
    }
  })
}

#[inline]
pub(crate) fn stream_accumulate_fn() -> StreamAccumulateFn {
  active().stream_accumulate
}

#[cfg(any(test, feature = "diag"))]
#[inline]
#[must_use]
fn kernel_id64_for_len(long_id: Xxh3KernelId, caps: Caps, len: usize) -> Xxh3KernelId {
  if len <= super::MID_SIZE_MAX {
    return Xxh3KernelId::Portable;
  }

  #[cfg(target_arch = "x86_64")]
  if len <= ZEN5_XXH3_64_AVX2_LONG_MAX
    && caps.has(crate::platform::caps::x86::AMD_ZEN5 | crate::platform::caps::x86::AVX2)
  {
    return Xxh3KernelId::Avx2;
  }

  let _ = caps;
  long_id
}

#[cfg(any(test, feature = "diag"))]
#[inline]
#[must_use]
pub fn kernel_name64_for_len(len: usize) -> &'static str {
  let d = active();
  kernel_id64_for_len(d.long_id, crate::platform::caps(), len).as_str()
}

#[cfg(any(test, feature = "diag"))]
#[inline]
#[must_use]
pub fn kernel_name128_for_len(len: usize) -> &'static str {
  if len <= super::MID_SIZE_MAX {
    Xxh3KernelId::Portable.as_str()
  } else {
    active().long_id.as_str()
  }
}

/// Flat size-based dispatch matching xxhash-rust's branch structure.
///
/// Every SIMD kernel delegates inputs ≤ 240 B back to the same portable scalar
/// functions, so we dispatch directly to the sub-functions here — eliminating
/// redundant ≤`MID_SIZE_MAX` checks in the long-path kernels.
///
/// Compile-time long dispatch inlines to a direct kernel call. Runtime fallback
/// stays out-of-line so ≤240B paths do not carry the cache/feature lookup.
#[inline(always)]
#[must_use]
pub fn hash64(data: &[u8]) -> u64 {
  let len = data.len();
  if len == 0 {
    return super::XXH3_64_EMPTY_DEFAULT;
  }
  if len <= 16 {
    return super::xxh3_64_0to16(data, 0, &super::DEFAULT_SECRET);
  }
  if len == 64 {
    return super::xxh3_64_64(data, 0, &super::DEFAULT_SECRET);
  }
  if len == 32 {
    return super::xxh3_64_32(data, 0, &super::DEFAULT_SECRET);
  }
  if len <= 128 {
    return super::xxh3_64_7to128(data, 0, &super::DEFAULT_SECRET);
  }
  if len <= super::MID_SIZE_MAX {
    return super::xxh3_64_129to240(data, 0, &super::DEFAULT_SECRET);
  }
  hash64_long_default(data)
}

#[inline(always)]
#[must_use]
pub fn hash128(data: &[u8]) -> u128 {
  let len = data.len();
  if len == 0 {
    return super::XXH3_128_EMPTY_DEFAULT;
  }
  if len <= 16 {
    return super::xxh3_128_0to16(data, 0, &super::DEFAULT_SECRET);
  }
  if len == 64 {
    return super::xxh3_128_64(data, 0, &super::DEFAULT_SECRET);
  }
  if len == 32 {
    return super::xxh3_128_32(data, 0, &super::DEFAULT_SECRET);
  }
  if len <= 128 {
    return super::xxh3_128_7to128(data, 0, &super::DEFAULT_SECRET);
  }
  if len <= super::MID_SIZE_MAX {
    return super::xxh3_128_129to240(data, 0, &super::DEFAULT_SECRET);
  }
  hash128_long_default(data)
}

#[inline(always)]
#[must_use]
pub fn hash64_with_seed(seed: u64, data: &[u8]) -> u64 {
  let len = data.len();
  if len == 0 && seed == 0 {
    return super::XXH3_64_EMPTY_DEFAULT;
  }
  if len <= 16 {
    return super::xxh3_64_0to16(data, seed, &super::DEFAULT_SECRET);
  }
  if len == 64 {
    return super::xxh3_64_64(data, seed, &super::DEFAULT_SECRET);
  }
  if len == 32 {
    return super::xxh3_64_32(data, seed, &super::DEFAULT_SECRET);
  }
  if len <= 128 {
    return super::xxh3_64_7to128(data, seed, &super::DEFAULT_SECRET);
  }
  if len <= super::MID_SIZE_MAX {
    return super::xxh3_64_129to240(data, seed, &super::DEFAULT_SECRET);
  }
  hash64_long(seed, data)
}

#[inline(always)]
fn hash64_long_default(data: &[u8]) -> u64 {
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx2"))]
  {
    if use_zen5_xxh3_64_avx2_short_long(data.len()) {
      return super::x86_64_avx2::xxh3_64_long_default(data);
    }
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  {
    return super::x86_64_avx512::xxh3_64_long_default(data);
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
  {
    return super::x86_64_avx2::xxh3_64_long_default(data);
  }

  #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
  {
    return super::aarch64_neon::xxh3_64_long_default(data);
  }

  #[cfg(target_arch = "riscv64")]
  {
    return super::xxh3_64_long_default(data);
  }

  #[allow(unreachable_code)]
  {
    let d = active();
    (d.long64)(data, 0)
  }
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
#[inline(always)]
fn hash64_long(seed: u64, data: &[u8]) -> u64 {
  // Tier 1: compile-time dispatch — dedicated long entry points skip ≤240B
  // branches that are guaranteed dead at this call site.
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx2"))]
  {
    if use_zen5_xxh3_64_avx2_short_long(data.len()) {
      return super::x86_64_avx2::xxh3_64_long(data, seed);
    }
  }

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

  // The retired RVV path lost to portable at 256 B–64 KiB on SpacemiT K1.
  // Bypass runtime dispatch on RISC-V.
  #[cfg(target_arch = "riscv64")]
  {
    return super::xxh3_64_long(data, seed);
  }

  // Tier 2: runtime dispatch — dedicated long-path fn pointer, no redundant
  // length checks.
  #[allow(unreachable_code)]
  hash64_long_runtime(seed, data)
}

#[inline(never)]
fn hash64_long_runtime(seed: u64, data: &[u8]) -> u64 {
  #[cfg(target_arch = "x86_64")]
  {
    if use_zen5_xxh3_64_avx2_short_long(data.len()) {
      return super::x86_64_avx2::xxh3_64_long(data, seed);
    }
  }

  let d = active();
  (d.long64)(data, seed)
}

/// See [`hash64_with_seed`] for the dispatch rationale.
#[inline(always)]
#[must_use]
pub fn hash128_with_seed(seed: u64, data: &[u8]) -> u128 {
  let len = data.len();
  if len == 0 && seed == 0 {
    return super::XXH3_128_EMPTY_DEFAULT;
  }
  if len <= 16 {
    return super::xxh3_128_0to16(data, seed, &super::DEFAULT_SECRET);
  }
  if len == 64 {
    return super::xxh3_128_64(data, seed, &super::DEFAULT_SECRET);
  }
  if len == 32 {
    return super::xxh3_128_32(data, seed, &super::DEFAULT_SECRET);
  }
  if len <= 128 {
    return super::xxh3_128_7to128(data, seed, &super::DEFAULT_SECRET);
  }
  if len <= super::MID_SIZE_MAX {
    return super::xxh3_128_129to240(data, seed, &super::DEFAULT_SECRET);
  }
  hash128_long(seed, data)
}

#[inline(always)]
fn hash128_long_default(data: &[u8]) -> u128 {
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  {
    return super::x86_64_avx512::xxh3_128_long_default(data);
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
  {
    return super::x86_64_avx2::xxh3_128_long_default(data);
  }

  #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
  {
    return super::aarch64_neon::xxh3_128_long_default(data);
  }

  #[cfg(target_arch = "riscv64")]
  {
    return super::xxh3_128_long_default(data);
  }

  #[allow(unreachable_code)]
  {
    let d = active();
    (d.long128)(data, 0)
  }
}

/// See [`hash64_long`] for the compile-time dispatch rationale.
#[inline(always)]
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
  hash128_long_runtime(seed, data)
}

#[inline(never)]
fn hash128_long_runtime(seed: u64, data: &[u8]) -> u128 {
  let d = active();
  (d.long128)(data, seed)
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
  use super::*;
  use crate::platform::caps::x86;

  #[test]
  fn diagnostic_kernel_ids_match_short_and_zen5_production_policy() {
    for len in [0, 16, 64, 128, 240] {
      assert_eq!(
        kernel_id64_for_len(Xxh3KernelId::Avx512, x86::AVX512F, len),
        Xxh3KernelId::Portable
      );
    }

    let zen5 = x86::AMD_ZEN5 | x86::AVX2 | x86::AVX512F;
    for len in [241, 256, 1024] {
      assert_eq!(kernel_id64_for_len(Xxh3KernelId::Avx512, zen5, len), Xxh3KernelId::Avx2);
    }
    assert_eq!(
      kernel_id64_for_len(Xxh3KernelId::Avx512, zen5, 1025),
      Xxh3KernelId::Avx512
    );
    assert_eq!(
      kernel_id64_for_len(Xxh3KernelId::Avx512, x86::AVX512F, 241),
      Xxh3KernelId::Avx512
    );
  }
}
