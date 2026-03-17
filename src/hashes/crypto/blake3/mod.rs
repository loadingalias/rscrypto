//! BLAKE3 (hash + XOF).
//!
//! This is a portable, dependency-free implementation suitable for `no_std`.

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![allow(clippy::indexing_slicing)] // Audited fixed-size parsing + perf-critical inner loops.

use core::{cmp::min, mem::MaybeUninit, ptr};
#[cfg(feature = "std")]
extern crate alloc;
#[cfg(feature = "std")]
use alloc::string::{String, ToString};
#[cfg(feature = "std")]
use core::cell::RefCell;
#[cfg(feature = "std")]
use std::sync::{Mutex, OnceLock};
#[cfg(feature = "parallel")]
use std::thread;
#[cfg(feature = "std")]
use std::thread_local;

use crate::traits::{Digest, Xof};

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;
#[doc(hidden)]
pub mod dispatch;
#[doc(hidden)]
pub mod dispatch_tables;
pub(crate) mod kernels;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64;

use self::kernels::Kernel;

const OUT_LEN: usize = 32;
const KEY_LEN: usize = 32;
const BLOCK_LEN: usize = 64;
const CHUNK_LEN: usize = 1024;
const OUTPUT_BLOCK_LEN: usize = 2 * OUT_LEN;
// Max CV stack depth for incremental hashing.
//
// BLAKE3 chunk size is 1024 bytes. If we cap the total input length at `u64`
// bytes, then the maximum number of chunks is `2^64 / 2^10 = 2^54`, i.e. a
// 54-level binary reduction tree. Lazy merging needs one extra slot (the new
// CV sits unmerged on top until the next push).
const CV_STACK_LEN: usize = 55;

type CvBytes = [u8; OUT_LEN];

mod join {
  pub(super) trait Join {
    fn join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
    where
      A: FnOnce() -> RA + Send,
      B: FnOnce() -> RB + Send,
      RA: Send,
      RB: Send;
  }

  pub(super) enum SerialJoin {}

  impl Join for SerialJoin {
    #[inline]
    fn join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
    where
      A: FnOnce() -> RA + Send,
      B: FnOnce() -> RB + Send,
      RA: Send,
      RB: Send,
    {
      (oper_a(), oper_b())
    }
  }

  #[cfg(feature = "parallel")]
  pub(super) enum RayonJoin {}

  #[cfg(feature = "parallel")]
  impl Join for RayonJoin {
    #[inline]
    fn join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
    where
      A: FnOnce() -> RA + Send,
      B: FnOnce() -> RB + Send,
      RA: Send,
      RB: Send,
    {
      rayon::join(oper_a, oper_b)
    }
  }
}

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;
const ROOT: u32 = 1 << 3;
const KEYED_HASH: u32 = 1 << 4;
const DERIVE_KEY_CONTEXT: u32 = 1 << 5;
const DERIVE_KEY_MATERIAL: u32 = 1 << 6;

const IV: [u32; 8] = [
  0x6A09_E667,
  0xBB67_AE85,
  0x3C6E_F372,
  0xA54F_F53A,
  0x510E_527F,
  0x9B05_688C,
  0x1F83_D9AB,
  0x5BE0_CD19,
];

#[cfg(feature = "std")]
static PARALLEL_OVERRIDE: OnceLock<Mutex<Option<ParallelPolicyOverride>>> = OnceLock::new();
#[cfg(feature = "parallel")]
static AVAILABLE_PARALLELISM: OnceLock<Option<usize>> = OnceLock::new();

#[cfg(feature = "std")]
#[inline]
fn compute_derive_context_key_words(context: &str) -> [u32; 8] {
  let context_bytes = context.as_bytes();
  let kernel_ctx = dispatch::hasher_dispatch().size_class_kernel(context_bytes.len());
  digest_oneshot_words(kernel_ctx, IV, DERIVE_KEY_CONTEXT, context_bytes)
}

#[cfg(feature = "std")]
#[inline]
fn derive_context_key_words_cached(context: &str) -> [u32; 8] {
  // Hot path: same context repeated on the same thread.
  if let Some(words) = DERIVE_KEY_CONTEXT_LOCAL_CACHE.with(|slot| {
    let borrowed = slot.borrow();
    borrowed
      .as_ref()
      .and_then(|(cached_context, cached_words)| (cached_context.as_str() == context).then_some(*cached_words))
  }) {
    return words;
  }

  let computed = compute_derive_context_key_words(context);
  DERIVE_KEY_CONTEXT_LOCAL_CACHE.with(|slot| {
    *slot.borrow_mut() = Some((context.to_string(), computed));
  });
  computed
}

#[cfg(feature = "std")]
#[inline]
fn keyed_words_cached(key: &[u8; KEY_LEN]) -> [u32; 8] {
  if let Some(words) = KEYED_WORDS_LOCAL_CACHE.with(|slot| {
    let borrowed = slot.borrow();
    borrowed
      .as_ref()
      .and_then(|(cached_key, cached_words)| (cached_key == key).then_some(*cached_words))
  }) {
    return words;
  }

  let words = words8_from_le_bytes_32(key);
  KEYED_WORDS_LOCAL_CACHE.with(|slot| {
    *slot.borrow_mut() = Some((*key, words));
  });
  words
}

#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug)]
pub struct ParallelPolicyOverride {
  pub oneshot: dispatch_tables::ParallelTable,
  pub keyed_oneshot: dispatch_tables::ParallelTable,
  pub derive_oneshot: dispatch_tables::ParallelTable,
  pub xof: dispatch_tables::ParallelTable,
  pub keyed_xof: dispatch_tables::ParallelTable,
  pub derive_xof: dispatch_tables::ParallelTable,
  pub streaming: dispatch_tables::ParallelTable,
  pub keyed_streaming: dispatch_tables::ParallelTable,
  pub derive_streaming: dispatch_tables::ParallelTable,
}

#[cfg(feature = "std")]
impl ParallelPolicyOverride {
  #[inline]
  fn new(table: dispatch_tables::ParallelTable) -> Self {
    Self {
      oneshot: table,
      keyed_oneshot: table,
      derive_oneshot: table,
      xof: table,
      keyed_xof: table,
      derive_xof: table,
      streaming: table,
      keyed_streaming: table,
      derive_streaming: table,
    }
  }
}

#[derive(Clone, Copy)]
enum ParallelPolicyKind {
  Oneshot,
  KeyedOneshot,
  DeriveOneshot,
  Xof,
  KeyedXof,
  DeriveXof,
  #[cfg(feature = "parallel")]
  Update,
  #[cfg(feature = "parallel")]
  KeyedUpdate,
  #[cfg(feature = "parallel")]
  DeriveUpdate,
}

#[cfg(feature = "parallel")]
#[inline]
fn parallel_override() -> Option<ParallelPolicyOverride> {
  PARALLEL_OVERRIDE
    .get_or_init(|| Mutex::new(None))
    .lock()
    .ok()
    .and_then(|g| *g)
}

#[cfg(feature = "parallel")]
#[inline]
fn available_parallelism_cached() -> Option<usize> {
  *AVAILABLE_PARALLELISM.get_or_init(|| {
    let std_threads = thread::available_parallelism().ok().map(|v| v.get()).unwrap_or(0);
    #[cfg(feature = "parallel")]
    let rayon_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let rayon_threads = 0usize;
    let threads = core::cmp::max(std_threads, rayon_threads);
    (threads != 0).then_some(threads)
  })
}

#[cfg(feature = "parallel")]
#[inline]
fn parallel_policy_threads_with_admission(
  mode: ParallelPolicyKind,
  input_bytes: usize,
  admission_full_chunks: usize,
  commit_full_chunks: usize,
) -> Option<usize> {
  let policy = if let Some(override_policy) = parallel_override() {
    match mode {
      ParallelPolicyKind::Oneshot => override_policy.oneshot,
      ParallelPolicyKind::KeyedOneshot => override_policy.keyed_oneshot,
      ParallelPolicyKind::DeriveOneshot => override_policy.derive_oneshot,
      ParallelPolicyKind::Xof => override_policy.xof,
      ParallelPolicyKind::KeyedXof => override_policy.keyed_xof,
      ParallelPolicyKind::DeriveXof => override_policy.derive_xof,
      ParallelPolicyKind::Update => override_policy.streaming,
      ParallelPolicyKind::KeyedUpdate => override_policy.keyed_streaming,
      ParallelPolicyKind::DeriveUpdate => override_policy.derive_streaming,
    }
  } else {
    let dispatch_policy = dispatch::parallel_dispatch();
    match mode {
      ParallelPolicyKind::Oneshot => dispatch_policy.oneshot,
      ParallelPolicyKind::KeyedOneshot => dispatch_policy.keyed_oneshot,
      ParallelPolicyKind::DeriveOneshot => dispatch_policy.derive_oneshot,
      ParallelPolicyKind::Xof => dispatch_policy.xof,
      ParallelPolicyKind::KeyedXof => dispatch_policy.keyed_xof,
      ParallelPolicyKind::DeriveXof => dispatch_policy.derive_xof,
      ParallelPolicyKind::Update => dispatch_policy.streaming,
      ParallelPolicyKind::KeyedUpdate => dispatch_policy.keyed_streaming,
      ParallelPolicyKind::DeriveUpdate => dispatch_policy.derive_streaming,
    }
  };

  parallel_threads_for_policy(mode, policy, input_bytes, admission_full_chunks, commit_full_chunks)
}

#[cfg(feature = "parallel")]
#[inline]
fn parallel_threads_for_policy(
  mode: ParallelPolicyKind,
  policy: dispatch_tables::ParallelTable,
  input_bytes: usize,
  admission_full_chunks: usize,
  commit_full_chunks: usize,
) -> Option<usize> {
  if policy.max_threads == 1 {
    return None;
  }

  if input_bytes < policy.min_bytes || admission_full_chunks < policy.min_chunks {
    return None;
  }

  let mut threads = available_parallelism_cached()?;
  if policy.max_threads != 0 {
    threads = threads.min(policy.max_threads as usize);
  }

  if threads <= 1 {
    return None;
  }

  let bytes_per_core = if input_bytes <= policy.small_limit_bytes {
    policy.bytes_per_core_small
  } else if input_bytes <= policy.medium_limit_bytes {
    policy.bytes_per_core_medium
  } else {
    policy.bytes_per_core_large
  };

  let mut candidate = threads.min(commit_full_chunks);
  while candidate > 1 {
    let merge_divisor = parallel_merge_divisor(mode, commit_full_chunks, candidate);
    let merge_cost = policy.merge_cost_bytes.saturating_add(merge_divisor - 1) / merge_divisor;
    let spawn_cost = policy.spawn_cost_bytes.saturating_mul(candidate - 1);
    let work_cost = bytes_per_core.saturating_mul(candidate);
    let fitted_required = merge_cost.saturating_add(spawn_cost).saturating_add(work_cost);
    let required = scale_parallel_required_bytes(mode, fitted_required, input_bytes);
    if input_bytes >= required {
      return Some(candidate);
    }
    candidate -= 1;
  }
  None
}

#[cfg(feature = "parallel")]
#[inline]
fn parallel_merge_divisor(mode: ParallelPolicyKind, commit_full_chunks: usize, threads: usize) -> usize {
  let chunk_depth = (commit_full_chunks.max(2)).ilog2() as usize;
  let thread_depth = (threads.max(2)).ilog2() as usize;
  let mode_bias = match mode {
    ParallelPolicyKind::Oneshot | ParallelPolicyKind::KeyedOneshot | ParallelPolicyKind::DeriveOneshot => 2,
    ParallelPolicyKind::Update | ParallelPolicyKind::KeyedUpdate | ParallelPolicyKind::DeriveUpdate => 1,
    ParallelPolicyKind::Xof | ParallelPolicyKind::KeyedXof | ParallelPolicyKind::DeriveXof => 3,
  };
  1 + chunk_depth + thread_depth + mode_bias
}

#[cfg(feature = "parallel")]
#[inline]
fn scale_parallel_required_bytes(mode: ParallelPolicyKind, required: usize, input_bytes: usize) -> usize {
  let (num, den) = match mode {
    // Fitted from crossover data: one-shot/XOF benefit earlier than update.
    ParallelPolicyKind::Oneshot | ParallelPolicyKind::KeyedOneshot | ParallelPolicyKind::DeriveOneshot => {
      if input_bytes >= (64 * 1024) {
        (10usize, 10usize)
      } else {
        (12usize, 10usize)
      }
    }
    ParallelPolicyKind::Update | ParallelPolicyKind::KeyedUpdate | ParallelPolicyKind::DeriveUpdate => {
      (15usize, 10usize)
    }
    ParallelPolicyKind::Xof | ParallelPolicyKind::KeyedXof | ParallelPolicyKind::DeriveXof => {
      if input_bytes >= (64 * 1024) {
        (9usize, 10usize)
      } else {
        (12usize, 10usize)
      }
    }
  };
  required.saturating_mul(num).saturating_add(den - 1) / den
}

#[cfg(feature = "std")]
thread_local! {
  static DERIVE_KEY_CONTEXT_LOCAL_CACHE: RefCell<Option<(String, [u32; 8])>> = const { RefCell::new(None) };
  static KEYED_WORDS_LOCAL_CACHE: RefCell<Option<([u8; KEY_LEN], [u32; 8])>> = const { RefCell::new(None) };
}

/// Benchmark-only hooks (not part of the stable API).
#[cfg(feature = "std")]
#[doc(hidden)]
pub mod __bench {
  // Re-export types needed for policy construction
  use super::{Mutex, PARALLEL_OVERRIDE, dispatch_tables};
  pub use super::{ParallelPolicyOverride, dispatch_tables::ParallelTable};

  #[derive(Debug)]
  pub struct Blake3ParallelOverrideGuard {
    prev: Option<ParallelPolicyOverride>,
  }

  impl Drop for Blake3ParallelOverrideGuard {
    fn drop(&mut self) {
      let lock: &Mutex<Option<ParallelPolicyOverride>> = PARALLEL_OVERRIDE.get_or_init(|| Mutex::new(None));
      if let Ok(mut g) = lock.lock() {
        *g = self.prev;
      }
    }
  }

  /// Override the active BLAKE3 parallel policy for the duration of the guard.
  ///
  /// This sets all variants to the same table. For per-variant control,
  /// use `override_blake3_parallel_policy_full()`.
  ///
  /// This is used by internal benchmarks to compare single-thread and
  /// multi-thread behavior without relying on user-facing environment variables.
  #[must_use]
  pub fn override_blake3_parallel_policy(table: dispatch_tables::ParallelTable) -> Blake3ParallelOverrideGuard {
    override_blake3_parallel_policy_full(ParallelPolicyOverride::new(table))
  }

  /// Override the active BLAKE3 parallel policy with per-variant control.
  ///
  /// This allows independent control of:
  /// - oneshot: regular one-shot hashing (`digest`, `xof`)
  /// - keyed_oneshot: keyed one-shot hashing (`keyed_digest`, `keyed_xof`)
  /// - derive_oneshot: derive-key one-shot hashing (`derive_key`)
  /// - streaming: regular streaming updates
  /// - keyed_streaming: keyed streaming updates
  /// - derive_streaming: derive-key streaming updates
  ///
  /// This is used by internal benchmarks for fine-grained policy overrides.
  #[must_use]
  pub fn override_blake3_parallel_policy_full(policy: ParallelPolicyOverride) -> Blake3ParallelOverrideGuard {
    let lock: &Mutex<Option<ParallelPolicyOverride>> = PARALLEL_OVERRIDE.get_or_init(|| Mutex::new(None));
    let mut prev = None;
    if let Ok(mut g) = lock.lock() {
      prev = *g;
      *g = Some(policy);
    };
    Blake3ParallelOverrideGuard { prev }
  }

  // NOTE: std parallelism is implemented via Rayon; there is no longer an
  // internal subtree scheduler to override.
}

/// BLAKE3 message schedule.
///
/// `MSG_SCHEDULE[round][i]` gives the index of the message word to use.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) const MSG_SCHEDULE: [[usize; 16]; 7] = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
  [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
  [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
  [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
  [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
  [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

#[inline(always)]
fn uninit_cv_stack() -> [MaybeUninit<[u32; 8]>; CV_STACK_LEN] {
  // SAFETY: An uninitialized `[MaybeUninit<_>; N]` is valid, because
  // `MaybeUninit<T>` permits any bit pattern.
  unsafe { MaybeUninit::<[MaybeUninit<[u32; 8]>; CV_STACK_LEN]>::uninit().assume_init() }
}

#[inline(always)]
fn words8_from_le_bytes_32(bytes: &[u8; 32]) -> [u32; 8] {
  if cfg!(target_endian = "little") {
    // SAFETY: `bytes` is exactly 32 bytes, and `read_unaligned` supports the
    // 1-byte alignment of `[u8; 32]`.
    unsafe { ptr::read_unaligned(bytes.as_ptr().cast::<[u32; 8]>()) }
  } else {
    let src = bytes.as_ptr() as *const u32;
    // SAFETY: `bytes` is exactly 32 bytes; `read_unaligned` supports the
    // 1-byte alignment of `[u8; 32]`.
    unsafe {
      [
        u32::from_le(ptr::read_unaligned(src.add(0))),
        u32::from_le(ptr::read_unaligned(src.add(1))),
        u32::from_le(ptr::read_unaligned(src.add(2))),
        u32::from_le(ptr::read_unaligned(src.add(3))),
        u32::from_le(ptr::read_unaligned(src.add(4))),
        u32::from_le(ptr::read_unaligned(src.add(5))),
        u32::from_le(ptr::read_unaligned(src.add(6))),
        u32::from_le(ptr::read_unaligned(src.add(7))),
      ]
    }
  }
}

#[inline(always)]
fn words16_from_le_bytes_64(bytes: &[u8; 64]) -> [u32; 16] {
  if cfg!(target_endian = "little") {
    // SAFETY: `bytes` is exactly 64 bytes, and `read_unaligned` supports the
    // 1-byte alignment of `[u8; 64]`.
    unsafe { ptr::read_unaligned(bytes.as_ptr().cast::<[u32; 16]>()) }
  } else {
    let src = bytes.as_ptr() as *const u32;
    // SAFETY: `bytes` is exactly 64 bytes; `read_unaligned` supports the
    // 1-byte alignment of `[u8; 64]`.
    unsafe {
      [
        u32::from_le(ptr::read_unaligned(src.add(0))),
        u32::from_le(ptr::read_unaligned(src.add(1))),
        u32::from_le(ptr::read_unaligned(src.add(2))),
        u32::from_le(ptr::read_unaligned(src.add(3))),
        u32::from_le(ptr::read_unaligned(src.add(4))),
        u32::from_le(ptr::read_unaligned(src.add(5))),
        u32::from_le(ptr::read_unaligned(src.add(6))),
        u32::from_le(ptr::read_unaligned(src.add(7))),
        u32::from_le(ptr::read_unaligned(src.add(8))),
        u32::from_le(ptr::read_unaligned(src.add(9))),
        u32::from_le(ptr::read_unaligned(src.add(10))),
        u32::from_le(ptr::read_unaligned(src.add(11))),
        u32::from_le(ptr::read_unaligned(src.add(12))),
        u32::from_le(ptr::read_unaligned(src.add(13))),
        u32::from_le(ptr::read_unaligned(src.add(14))),
        u32::from_le(ptr::read_unaligned(src.add(15))),
      ]
    }
  }
}

#[inline(always)]
fn pow2_floor(n: usize) -> usize {
  debug_assert!(n != 0);
  1usize << (usize::BITS - 1 - n.leading_zeros())
}

#[inline]
fn reduce_power_of_two_chunk_cvs(kernel: Kernel, key_words: [u32; 8], flags: u32, cvs: &[[u32; 8]]) -> [u32; 8] {
  debug_assert!(cvs.len().is_power_of_two());
  debug_assert!(cvs.len() <= 16);

  if cvs.len() == 1 {
    return cvs[0];
  }

  let mut cur = [[0u32; 8]; 16];
  let mut next = [[0u32; 8]; 16];
  cur[..cvs.len()].copy_from_slice(cvs);
  let mut cur_len = cvs.len();

  while cur_len > 1 {
    let pairs = cur_len / 2;
    debug_assert!(pairs <= 8);
    kernels::parent_cvs_many_from_cvs_inline(kernel.id, &cur[..2 * pairs], key_words, flags, &mut next[..pairs]);
    cur[..pairs].copy_from_slice(&next[..pairs]);
    cur_len = pairs;
  }

  cur[0]
}

#[cfg(not(target_endian = "little"))]
#[inline]
fn reduce_power_of_two_chunk_cvs_bytes(kernel: Kernel, key_words: [u32; 8], flags: u32, cvs: &[CvBytes]) -> CvBytes {
  debug_assert!(cvs.len().is_power_of_two());
  debug_assert!(cvs.len() <= 16);

  if cvs.len() == 1 {
    return cvs[0];
  }

  let mut cur = [[0u8; OUT_LEN]; 16];
  let mut next = [[0u8; OUT_LEN]; 16];
  cur[..cvs.len()].copy_from_slice(cvs);
  let mut cur_len = cvs.len();

  while cur_len > 1 {
    let pairs = cur_len / 2;
    debug_assert!(pairs <= 8);
    kernels::parent_cvs_many_from_bytes_inline(kernel.id, &cur[..2 * pairs], key_words, flags, &mut next[..pairs]);
    cur[..pairs].copy_from_slice(&next[..pairs]);
    cur_len = pairs;
  }

  cur[0]
}

#[cfg(target_endian = "little")]
#[inline]
fn add_chunk_cvs_batched(
  kernel: Kernel,
  stack: &mut [MaybeUninit<[u32; 8]>; CV_STACK_LEN],
  stack_len: &mut usize,
  base_counter: u64,
  cvs: &[[u32; 8]],
  key_words: [u32; 8],
  flags: u32,
) {
  if cvs.is_empty() {
    return;
  }

  debug_assert!(cvs.len() <= 16);

  #[inline]
  fn push_stack(stack: &mut [MaybeUninit<[u32; 8]>; CV_STACK_LEN], len: &mut usize, cv: [u32; 8]) {
    stack[*len].write(cv);
    *len += 1;
  }

  #[inline]
  fn pop_stack(stack: &mut [MaybeUninit<[u32; 8]>; CV_STACK_LEN], len: &mut usize) -> [u32; 8] {
    *len -= 1;
    // SAFETY: `len` tracks the number of initialized entries.
    unsafe { stack[*len].assume_init_read() }
  }

  let mut offset = 0usize;
  let mut chunk_counter = base_counter;

  while offset < cvs.len() {
    let remaining = cvs.len() - offset;
    let mut size = pow2_floor(remaining);

    let aligned_max = if chunk_counter == 0 {
      usize::MAX
    } else {
      let tz = chunk_counter.trailing_zeros() as usize;
      if tz >= (usize::BITS as usize) {
        usize::MAX
      } else {
        1usize << tz
      }
    };

    size = size.min(aligned_max).min(remaining);
    debug_assert!(size.is_power_of_two());

    let subtree_cv = reduce_power_of_two_chunk_cvs(kernel, key_words, flags, &cvs[offset..offset + size]);
    chunk_counter = chunk_counter.wrapping_add(size as u64);

    // Merge this subtree into the global stack. Because `size` is a power of two
    // that divides `base_counter + offset`, there are no pending nodes below
    // this subtree's level, and popping proceeds in correct level order.
    let level = size.trailing_zeros();
    let mut total = chunk_counter >> level;
    let mut cv = subtree_cv;
    while total & 1 == 0 {
      cv = kernels::parent_cv_inline(kernel.id, pop_stack(stack, stack_len), cv, key_words, flags);
      total >>= 1;
    }
    push_stack(stack, stack_len, cv);

    offset += size;
  }
}

#[cfg(not(target_endian = "little"))]
#[inline]
fn add_chunk_cvs_batched_bytes(
  kernel: Kernel,
  stack: &mut [MaybeUninit<[u32; 8]>; CV_STACK_LEN],
  stack_len: &mut usize,
  base_counter: u64,
  cvs: &[CvBytes],
  key_words: [u32; 8],
  flags: u32,
) {
  if cvs.is_empty() {
    return;
  }

  debug_assert!(cvs.len() <= 16);

  #[inline]
  fn push_stack(stack: &mut [MaybeUninit<[u32; 8]>; CV_STACK_LEN], len: &mut usize, cv: [u32; 8]) {
    stack[*len].write(cv);
    *len += 1;
  }

  #[inline]
  fn pop_stack(stack: &mut [MaybeUninit<[u32; 8]>; CV_STACK_LEN], len: &mut usize) -> [u32; 8] {
    *len -= 1;
    // SAFETY: `len` tracks the number of initialized entries.
    unsafe { stack[*len].assume_init_read() }
  }

  let mut offset = 0usize;
  let mut chunk_counter = base_counter;

  while offset < cvs.len() {
    let remaining = cvs.len() - offset;
    let mut size = pow2_floor(remaining);

    let aligned_max = if chunk_counter == 0 {
      usize::MAX
    } else {
      let tz = chunk_counter.trailing_zeros() as usize;
      if tz >= (usize::BITS as usize) {
        usize::MAX
      } else {
        1usize << tz
      }
    };

    size = size.min(aligned_max).min(remaining);
    debug_assert!(size.is_power_of_two());

    let subtree_cv_bytes = reduce_power_of_two_chunk_cvs_bytes(kernel, key_words, flags, &cvs[offset..offset + size]);
    let subtree_cv = words8_from_le_bytes_32(&subtree_cv_bytes);
    chunk_counter = chunk_counter.wrapping_add(size as u64);

    // Merge this subtree into the global stack. Because `size` is a power of two
    // that divides `base_counter + offset`, there are no pending nodes below
    // this subtree's level, and popping proceeds in correct level order.
    let level = size.trailing_zeros();
    let mut total = chunk_counter >> level;
    let mut cv = subtree_cv;
    while total & 1 == 0 {
      cv = kernels::parent_cv_inline(kernel.id, pop_stack(stack, stack_len), cv, key_words, flags);
      total >>= 1;
    }
    push_stack(stack, stack_len, cv);

    offset += size;
  }
}

#[inline]
fn compress(chaining_value: &[u32; 8], block_words: &[u32; 16], counter: u64, block_len: u32, flags: u32) -> [u32; 16] {
  let m0 = block_words[0];
  let m1 = block_words[1];
  let m2 = block_words[2];
  let m3 = block_words[3];
  let m4 = block_words[4];
  let m5 = block_words[5];
  let m6 = block_words[6];
  let m7 = block_words[7];
  let m8 = block_words[8];
  let m9 = block_words[9];
  let m10 = block_words[10];
  let m11 = block_words[11];
  let m12 = block_words[12];
  let m13 = block_words[13];
  let m14 = block_words[14];
  let m15 = block_words[15];

  let counter_low = counter as u32;
  let counter_high = (counter >> 32) as u32;
  let mut v0 = chaining_value[0];
  let mut v1 = chaining_value[1];
  let mut v2 = chaining_value[2];
  let mut v3 = chaining_value[3];
  let mut v4 = chaining_value[4];
  let mut v5 = chaining_value[5];
  let mut v6 = chaining_value[6];
  let mut v7 = chaining_value[7];
  let mut v8 = IV[0];
  let mut v9 = IV[1];
  let mut v10 = IV[2];
  let mut v11 = IV[3];
  let mut v12 = counter_low;
  let mut v13 = counter_high;
  let mut v14 = block_len;
  let mut v15 = flags;

  macro_rules! g {
    ($a:ident, $b:ident, $c:ident, $d:ident, $mx:expr, $my:expr) => {{
      $a = $a.wrapping_add($b).wrapping_add($mx);
      $d = ($d ^ $a).rotate_right(16);
      $c = $c.wrapping_add($d);
      $b = ($b ^ $c).rotate_right(12);
      $a = $a.wrapping_add($b).wrapping_add($my);
      $d = ($d ^ $a).rotate_right(8);
      $c = $c.wrapping_add($d);
      $b = ($b ^ $c).rotate_right(7);
    }};
  }

  // One full BLAKE3 round, with an explicit message schedule. This lets the
  // compiler keep `v0..v15` and `m0..m15` in registers without indirect
  // indexing in the hottest loop.
  macro_rules! round {
    (
      $m0:expr, $m1:expr, $m2:expr, $m3:expr, $m4:expr, $m5:expr, $m6:expr, $m7:expr,
      $m8:expr, $m9:expr, $m10:expr, $m11:expr, $m12:expr, $m13:expr, $m14:expr, $m15:expr
    ) => {{
      g!(v0, v4, v8, v12, $m0, $m1);
      g!(v1, v5, v9, v13, $m2, $m3);
      g!(v2, v6, v10, v14, $m4, $m5);
      g!(v3, v7, v11, v15, $m6, $m7);

      g!(v0, v5, v10, v15, $m8, $m9);
      g!(v1, v6, v11, v12, $m10, $m11);
      g!(v2, v7, v8, v13, $m12, $m13);
      g!(v3, v4, v9, v14, $m14, $m15);
    }};
  }

  // Per-round schedules for the 7-round BLAKE3 compression function.
  round!(m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);
  round!(m2, m6, m3, m10, m7, m0, m4, m13, m1, m11, m12, m5, m9, m14, m15, m8);
  round!(m3, m4, m10, m12, m13, m2, m7, m14, m6, m5, m9, m0, m11, m15, m8, m1);
  round!(m10, m7, m12, m9, m14, m3, m13, m15, m4, m0, m11, m2, m5, m8, m1, m6);
  round!(m12, m13, m9, m11, m15, m10, m14, m8, m7, m2, m5, m3, m0, m1, m6, m4);
  round!(m9, m14, m11, m5, m8, m12, m15, m1, m13, m3, m0, m10, m2, m6, m4, m7);
  round!(m11, m15, m5, m0, m1, m9, m8, m6, m14, m10, m2, m12, m3, m4, m7, m13);

  v0 ^= v8;
  v1 ^= v9;
  v2 ^= v10;
  v3 ^= v11;
  v4 ^= v12;
  v5 ^= v13;
  v6 ^= v14;
  v7 ^= v15;

  v8 ^= chaining_value[0];
  v9 ^= chaining_value[1];
  v10 ^= chaining_value[2];
  v11 ^= chaining_value[3];
  v12 ^= chaining_value[4];
  v13 ^= chaining_value[5];
  v14 ^= chaining_value[6];
  v15 ^= chaining_value[7];

  [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15]
}

#[inline]
fn left_subtree_len_bytes(input_len: usize) -> usize {
  debug_assert!(input_len > CHUNK_LEN);
  let full_chunks = input_len / CHUNK_LEN;
  let has_partial = !input_len.is_multiple_of(CHUNK_LEN);
  let total_chunks = full_chunks + has_partial as usize;
  debug_assert!(total_chunks >= 2);

  let left_chunks = if total_chunks.is_power_of_two() {
    total_chunks / 2
  } else {
    pow2_floor(total_chunks)
  };
  debug_assert!(left_chunks >= 1 && left_chunks < total_chunks);
  left_chunks * CHUNK_LEN
}

#[inline]
fn compress_parents_parallel_bytes(
  kernel: Kernel,
  child_cvs: &[CvBytes],
  key_words: [u32; 8],
  flags: u32,
  out: &mut [CvBytes],
) -> usize {
  debug_assert!(child_cvs.len() >= 2);
  debug_assert!(out.len() >= child_cvs.len().div_ceil(2));

  let pairs = child_cvs.len() / 2;
  kernels::parent_cvs_many_from_bytes_inline(kernel.id, &child_cvs[..pairs * 2], key_words, flags, &mut out[..pairs]);
  if (child_cvs.len() & 1) == 1 {
    out[pairs] = child_cvs[child_cvs.len() - 1];
    pairs + 1
  } else {
    pairs
  }
}

#[inline]
fn compress_subtree_wide_bytes<J: join::Join>(
  kernel: Kernel,
  key_words: [u32; 8],
  chunk_counter: u64,
  flags: u32,
  input: &[u8],
  out: &mut [CvBytes],
  par_budget: usize,
) -> usize {
  debug_assert!(!input.is_empty());

  let simd_degree = kernel.id.simd_degree();
  let max_leaf_bytes = simd_degree * CHUNK_LEN;
  if input.len() <= max_leaf_bytes {
    let chunks_exact = input.chunks_exact(CHUNK_LEN);
    let full_chunks = chunks_exact.len();
    debug_assert!(full_chunks <= simd_degree);
    debug_assert!(out.len() >= full_chunks + (!chunks_exact.remainder().is_empty()) as usize);

    if full_chunks != 0 {
      // SAFETY: `input` has at least `full_chunks * CHUNK_LEN` bytes and
      // `out` has at least `full_chunks * OUT_LEN` bytes.
      unsafe {
        (kernel.hash_many_contiguous)(
          input.as_ptr(),
          full_chunks,
          &key_words,
          chunk_counter,
          flags,
          out.as_mut_ptr().cast::<u8>(),
        );
      }
    }

    let mut out_len = full_chunks;
    let rem = chunks_exact.remainder();
    if !rem.is_empty() {
      let cv_words =
        single_chunk_output(kernel, key_words, chunk_counter + full_chunks as u64, flags, rem).chaining_value();
      out[out_len] = words8_to_le_bytes(&cv_words);
      out_len += 1;
    }

    return out_len;
  }

  debug_assert!(out.len() >= simd_degree.max(2));
  let left_len = left_subtree_len_bytes(input.len());
  let (left, right) = input.split_at(left_len);
  let right_chunk_counter = chunk_counter + (left.len() / CHUNK_LEN) as u64;

  const MAX_SIMD_DEGREE: usize = 16;
  let mut cv_array = [[0u8; OUT_LEN]; 2 * MAX_SIMD_DEGREE];

  let simd = simd_degree;
  let degree = if simd == 1 && left.len() == CHUNK_LEN {
    1
  } else {
    simd.max(2)
  };
  let (left_out, right_out) = cv_array.split_at_mut(degree);

  let next_budget = par_budget.saturating_sub(1);
  let (left_n, right_n) = if par_budget == 0 {
    (
      compress_subtree_wide_bytes::<join::SerialJoin>(kernel, key_words, chunk_counter, flags, left, left_out, 0),
      compress_subtree_wide_bytes::<join::SerialJoin>(
        kernel,
        key_words,
        right_chunk_counter,
        flags,
        right,
        right_out,
        0,
      ),
    )
  } else {
    J::join(
      || compress_subtree_wide_bytes::<J>(kernel, key_words, chunk_counter, flags, left, left_out, next_budget),
      || {
        compress_subtree_wide_bytes::<J>(
          kernel,
          key_words,
          right_chunk_counter,
          flags,
          right,
          right_out,
          next_budget,
        )
      },
    )
  };

  debug_assert_eq!(left_n, degree);
  debug_assert!(right_n >= 1 && right_n <= left_n);

  if left_n == 1 {
    out[0] = left_out[0];
    out[1] = right_out[0];
    return 2;
  }

  let num_children = left_n + right_n;
  compress_parents_parallel_bytes(kernel, &cv_array[..num_children], key_words, flags, out)
}

#[inline]
fn compress_subtree_to_parent_node_bytes<J: join::Join>(
  kernel: Kernel,
  key_words: [u32; 8],
  chunk_counter: u64,
  flags: u32,
  input: &[u8],
  par_budget: usize,
) -> [u8; BLOCK_LEN] {
  debug_assert!(input.len() > CHUNK_LEN);

  const MAX_SIMD_DEGREE: usize = 16;
  let mut cv_array = [[0u8; OUT_LEN]; MAX_SIMD_DEGREE];
  let mut num_cvs = compress_subtree_wide_bytes::<J>(
    kernel,
    key_words,
    chunk_counter,
    flags,
    input,
    &mut cv_array,
    par_budget,
  );
  debug_assert!(num_cvs >= 2);

  let mut out_array = [[0u8; OUT_LEN]; MAX_SIMD_DEGREE / 2];
  while num_cvs > 2 {
    let cv_slice = &cv_array[..num_cvs];
    let new_n = compress_parents_parallel_bytes(kernel, cv_slice, key_words, flags, &mut out_array);
    cv_array[..new_n].copy_from_slice(&out_array[..new_n]);
    num_cvs = new_n;
  }

  let mut out = [0u8; BLOCK_LEN];
  out[..OUT_LEN].copy_from_slice(&cv_array[0]);
  out[OUT_LEN..].copy_from_slice(&cv_array[1]);
  out
}

#[cfg(feature = "parallel")]
#[inline]
fn root_output_oneshot_join_parallel(
  kernel: Kernel,
  key_words: [u32; 8],
  flags: u32,
  input: &[u8],
  threads: usize,
) -> OutputState {
  debug_assert!(input.len() > CHUNK_LEN);
  debug_assert!(threads > 1);

  // Cap Rayon recursion depth to approximately match `threads` leaves.
  let depth = (usize::BITS - 1 - threads.leading_zeros()) as usize;
  let budget = depth.max(1);

  let parent_block =
    compress_subtree_to_parent_node_bytes::<join::RayonJoin>(kernel, key_words, 0, flags, input, budget);
  let block_words = words16_from_le_bytes_64(&parent_block);
  OutputState {
    kernel_id: kernel.id,
    input_chaining_value: key_words,
    block_words,
    counter: 0,
    block_len: BLOCK_LEN as u32,
    flags: PARENT | flags,
  }
}

#[inline(always)]
fn first_8_words(words: [u32; 16]) -> [u32; 8] {
  // SAFETY: fixed-size arrays
  [
    words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7],
  ]
}

#[inline(always)]
fn words8_to_le_bytes(words: &[u32; 8]) -> [u8; OUT_LEN] {
  let mut out = [0u8; OUT_LEN];
  if cfg!(target_endian = "little") {
    // SAFETY: `words` is 8 u32s = 32 bytes, and `out` is 32 bytes.
    unsafe { ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), out.as_mut_ptr(), OUT_LEN) };
  } else {
    for (i, word) in words.iter().copied().enumerate() {
      let offset = i * 4;
      out[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
    }
  }
  out
}

#[inline(always)]
fn words16_to_le_bytes(words: &[u32; 16]) -> [u8; 2 * OUT_LEN] {
  let mut out = [0u8; 2 * OUT_LEN];
  if cfg!(target_endian = "little") {
    // SAFETY: `words` is 16 u32s = 64 bytes, and `out` is 64 bytes.
    unsafe { ptr::copy_nonoverlapping(words.as_ptr() as *const u8, out.as_mut_ptr(), 2 * OUT_LEN) };
  } else {
    for (i, word) in words.iter().copied().enumerate() {
      let offset = i * 4;
      out[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
    }
  }
  out
}

#[derive(Clone, Copy)]
struct OutputState {
  kernel_id: kernels::Blake3KernelId,
  input_chaining_value: [u32; 8],
  block_words: [u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
}

impl OutputState {
  #[inline]
  fn chaining_value(&self) -> [u32; 8] {
    first_8_words(kernels::compress_block_inline(
      self.kernel_id,
      &self.input_chaining_value,
      &self.block_words,
      self.counter,
      self.block_len,
      self.flags,
    ))
  }

  #[inline]
  fn root_hash_words(&self) -> [u32; 8] {
    first_8_words(kernels::compress_block_inline(
      self.kernel_id,
      &self.input_chaining_value,
      &self.block_words,
      0,
      self.block_len,
      self.flags | ROOT,
    ))
  }

  #[inline]
  fn root_hash_bytes(&self) -> [u8; OUT_LEN] {
    words8_to_le_bytes(&self.root_hash_words())
  }

  #[inline]
  fn into_root_emit_state(self) -> RootEmitState {
    debug_assert!(self.block_len <= u8::MAX as u32);
    debug_assert!(self.flags <= u8::MAX as u32);
    RootEmitState {
      kernel_id: self.kernel_id,
      input_chaining_value: self.input_chaining_value,
      block_words: self.block_words,
      counter: self.counter,
      block_len: self.block_len as u8,
      flags: self.flags as u8,
    }
  }
}

#[derive(Clone, Copy)]
struct RootEmitState {
  input_chaining_value: [u32; 8],
  block_words: [u32; 16],
  counter: u64,
  block_len: u8,
  flags: u8,
  kernel_id: kernels::Blake3KernelId,
}

impl RootEmitState {
  #[inline]
  fn from_parent(
    kernel_id: kernels::Blake3KernelId,
    left_child_cv: [u32; 8],
    right_child_cv: [u32; 8],
    key_words: [u32; 8],
    flags: u32,
  ) -> Self {
    let mut block_words = [0u32; 16];
    block_words[..8].copy_from_slice(&left_child_cv);
    block_words[8..].copy_from_slice(&right_child_cv);
    Self {
      kernel_id,
      input_chaining_value: key_words,
      block_words,
      counter: 0,
      block_len: BLOCK_LEN as u8,
      flags: (PARENT | flags) as u8,
    }
  }

  #[inline]
  fn emit_one_block(&self, out: &mut [u8; OUTPUT_BLOCK_LEN]) {
    kernels::root_output_block_words(
      self.kernel_id,
      &self.input_chaining_value,
      &self.block_words,
      self.counter,
      u32::from(self.block_len),
      u32::from(self.flags) | ROOT,
      out,
    );
  }

  #[inline]
  fn emit_blocks_into(&self, out: &mut [u8]) {
    kernels::root_output_blocks_bytes_into_inline(
      self.kernel_id,
      &self.input_chaining_value,
      &self.block_words,
      self.counter,
      u32::from(self.block_len),
      u32::from(self.flags) | ROOT,
      out,
    );
  }
}

#[inline]
fn compress_node_chaining_value(
  kernel_id: kernels::Blake3KernelId,
  input_chaining_value: [u32; 8],
  block: &[u8; BLOCK_LEN],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  let block_words = words16_from_le_bytes_64(block);
  first_8_words(kernels::compress_block_inline(
    kernel_id,
    &input_chaining_value,
    &block_words,
    counter,
    block_len,
    flags,
  ))
}

#[derive(Clone, Copy)]
struct ChunkState {
  kernel_id: kernels::Blake3KernelId,
  chaining_value: [u32; 8],
  chunk_counter: u64,
  block: [u8; BLOCK_LEN],
  block_len: u8,
  blocks_compressed: u8,
  flags: u32,
}

impl ChunkState {
  #[inline]
  fn new(key_words: [u32; 8], chunk_counter: u64, flags: u32, kernel_id: kernels::Blake3KernelId) -> Self {
    Self {
      kernel_id,
      chaining_value: key_words,
      chunk_counter,
      block: [0u8; BLOCK_LEN],
      block_len: 0,
      blocks_compressed: 0,
      flags,
    }
  }

  #[inline]
  fn len(&self) -> usize {
    BLOCK_LEN * self.blocks_compressed as usize + self.block_len as usize
  }

  #[inline]
  fn start_flag(&self) -> u32 {
    if self.blocks_compressed == 0 { CHUNK_START } else { 0 }
  }

  #[inline]
  fn absorb_exact_one_chunk_fallback(&mut self, input: &[u8]) {
    debug_assert_eq!(self.blocks_compressed, 0);
    debug_assert_eq!(self.block_len, 0);
    debug_assert_eq!(input.len(), CHUNK_LEN);

    kernels::chunk_compress_blocks_inline(
      self.kernel_id,
      &mut self.chaining_value,
      self.chunk_counter,
      self.flags,
      &mut self.blocks_compressed,
      &input[..CHUNK_LEN - BLOCK_LEN],
    );
    debug_assert_eq!(self.blocks_compressed, 15);
    self.block.copy_from_slice(&input[CHUNK_LEN - BLOCK_LEN..]);
    self.block_len = BLOCK_LEN as u8;
  }

  #[inline]
  fn absorb_exact_one_chunk(&mut self, input: &[u8]) {
    debug_assert_eq!(self.blocks_compressed, 0);
    debug_assert_eq!(self.block_len, 0);
    debug_assert_eq!(input.len(), CHUNK_LEN);
    let key = self.chaining_value;
    let (cv, last_block) = absorb_exact_one_chunk_state(self.kernel_id, input, key, self.chunk_counter, self.flags);
    self.chaining_value = cv;
    self.block = last_block;
    self.block_len = BLOCK_LEN as u8;
    self.blocks_compressed = 15;
  }

  #[inline]
  fn update(&mut self, input: &[u8]) {
    if input.is_empty() {
      return;
    }

    if self.block_len == 0 && self.blocks_compressed == 0 {
      if input.len() == CHUNK_LEN {
        if should_use_exact_one_chunk_fast_path(self.kernel_id) {
          self.absorb_exact_one_chunk(input);
          return;
        }

        self.absorb_exact_one_chunk_fallback(input);
        return;
      }

      if input.len() <= BLOCK_LEN {
        self.block[..input.len()].copy_from_slice(input);
        self.block_len = input.len() as u8;
        return;
      }
    }

    self.update_general(input);
  }

  fn update_general(&mut self, mut input: &[u8]) {
    // Phase 1: if we already have a buffered (partial or full) block, fill it
    // (or, if it's already full, compress it) before touching the caller
    // slice. This keeps the hot "many full blocks" path branch-light.
    if self.block_len != 0 {
      let want = BLOCK_LEN - self.block_len as usize;
      let take = min(want, input.len());
      self.block[self.block_len as usize..][..take].copy_from_slice(&input[..take]);
      self.block_len = self.block_len.strict_add(take as u8);
      input = &input[take..];

      // If the caller ended mid-block, we're done. Note that this also covers
      // the (rare) case where we just filled the final block of a chunk
      // (blocks_compressed == 15), in which case the full block must remain
      // buffered until output().
      if input.is_empty() {
        return;
      }

      if self.block_len as usize == BLOCK_LEN {
        debug_assert!(
          self.blocks_compressed < 15,
          "last chunk block stays buffered until output()"
        );
        kernels::chunk_compress_blocks_inline(
          self.kernel_id,
          &mut self.chaining_value,
          self.chunk_counter,
          self.flags,
          &mut self.blocks_compressed,
          &self.block,
        );
        self.block_len = 0;
        self.block = [0u8; BLOCK_LEN];
      }
    }

    // Phase 2: we are block-aligned. Compress as many full blocks as possible
    // directly from the caller slice, leaving exactly one full block buffered
    // when the caller ends on a block boundary.
    debug_assert_eq!(self.block_len, 0);
    while !input.is_empty() {
      // Once we've compressed 15 blocks, the final block stays buffered.
      if self.blocks_compressed == 15 {
        debug_assert!(input.len() <= BLOCK_LEN);
        self.block[..input.len()].copy_from_slice(input);
        self.block_len = input.len() as u8;
        return;
      }

      let full_blocks = input.len() / BLOCK_LEN;
      if full_blocks != 0 {
        let max_blocks = 15usize - self.blocks_compressed as usize;
        let mut blocks_to_compress = full_blocks.min(max_blocks);

        // If we'd consume the entire input as full blocks, leave one block
        // buffered so finalize can apply CHUNK_END if the caller stops here.
        if input.len().is_multiple_of(BLOCK_LEN) && blocks_to_compress == full_blocks && blocks_to_compress > 0 {
          blocks_to_compress = blocks_to_compress.strict_sub(1);
        }

        if blocks_to_compress != 0 {
          let bytes = blocks_to_compress * BLOCK_LEN;
          kernels::chunk_compress_blocks_inline(
            self.kernel_id,
            &mut self.chaining_value,
            self.chunk_counter,
            self.flags,
            &mut self.blocks_compressed,
            &input[..bytes],
          );
          input = &input[bytes..];
          continue;
        }
      }

      // Remainder: buffer <= 64 bytes and return.
      let take = min(BLOCK_LEN, input.len());
      self.block[..take].copy_from_slice(&input[..take]);
      self.block_len = take as u8;
      return;
    }
  }

  #[inline]
  fn output_chaining_value(&self) -> [u32; 8] {
    compress_node_chaining_value(
      self.kernel_id,
      self.chaining_value,
      &self.block,
      self.chunk_counter,
      self.block_len as u32,
      self.flags | self.start_flag() | CHUNK_END,
    )
  }

  #[inline]
  fn root_emit_state(&self) -> RootEmitState {
    debug_assert_eq!(
      self.chunk_counter, 0,
      "single-message frontier root emit state must be chunk-counter zero"
    );
    RootEmitState {
      kernel_id: self.kernel_id,
      input_chaining_value: self.chaining_value,
      block_words: words16_from_le_bytes_64(&self.block),
      counter: self.chunk_counter,
      block_len: self.block_len,
      flags: (self.flags | self.start_flag() | CHUNK_END) as u8,
    }
  }

  #[inline]
  fn output(&self) -> OutputState {
    OutputState {
      kernel_id: self.kernel_id,
      input_chaining_value: self.chaining_value,
      block_words: words16_from_le_bytes_64(&self.block),
      counter: self.chunk_counter,
      block_len: self.block_len as u32,
      flags: self.flags | self.start_flag() | CHUNK_END,
    }
  }
}

#[inline]
fn absorb_exact_one_chunk_state(
  kernel_id: kernels::Blake3KernelId,
  input: &[u8],
  key: [u32; 8],
  counter: u64,
  flags: u32,
) -> ([u32; 8], [u8; BLOCK_LEN]) {
  debug_assert_eq!(input.len(), CHUNK_LEN);

  #[cfg(target_arch = "x86_64")]
  {
    let (prefix_blocks, remainder) = input[..CHUNK_LEN - BLOCK_LEN].as_chunks::<BLOCK_LEN>();
    debug_assert!(remainder.is_empty());

    match kernel_id {
      kernels::Blake3KernelId::X86Sse41 | kernels::Blake3KernelId::X86Avx2 => {
        let mut cv = key;
        for (idx, block) in prefix_blocks.iter().enumerate() {
          let block_flags = flags | if idx == 0 { CHUNK_START } else { 0 };
          // SAFETY: dispatch validates SSE4.1 for both kernels, and `block` is a
          // readable 64-byte buffer.
          unsafe {
            x86_64::compress_in_place_sse41_bytes(&mut cv, block.as_ptr(), counter, BLOCK_LEN as u32, block_flags)
          };
        }
        let mut last_block = [0u8; BLOCK_LEN];
        last_block.copy_from_slice(&input[CHUNK_LEN - BLOCK_LEN..]);
        return (cv, last_block);
      }
      kernels::Blake3KernelId::X86Avx512 => {
        let mut cv = key;
        for (idx, block) in prefix_blocks.iter().enumerate() {
          let block_flags = flags | if idx == 0 { CHUNK_START } else { 0 };
          #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
          {
            // SAFETY: dispatch validates AVX-512 availability, and `block` is a
            // readable 64-byte buffer.
            unsafe {
              x86_64::asm::compress_in_place_avx512_mut(&mut cv, block.as_ptr(), counter, BLOCK_LEN as u32, block_flags)
            };
          }
          #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
          {
            // SAFETY: dispatch validates AVX-512 availability, and `block` is a
            // readable 64-byte buffer.
            unsafe {
              x86_64::compress_in_place_avx512_bytes(&mut cv, block.as_ptr(), counter, BLOCK_LEN as u32, block_flags)
            };
          }
        }
        let mut last_block = [0u8; BLOCK_LEN];
        last_block.copy_from_slice(&input[CHUNK_LEN - BLOCK_LEN..]);
        return (cv, last_block);
      }
      _ => {}
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    if kernel_id == kernels::Blake3KernelId::Aarch64Neon {
      let mut cv = [0u32; 8];
      let mut last_block = [0u8; BLOCK_LEN];

      // SAFETY: the caller only uses this path for the selected NEON kernel,
      // and `input` is exactly one full chunk.
      unsafe {
        aarch64::chunk_state_one_chunk_aarch64_out(
          input.as_ptr(),
          &key,
          counter,
          flags,
          cv.as_mut_ptr(),
          last_block.as_mut_ptr(),
        );
      }

      return (cv, last_block);
    }
  }

  let mut cv = key;
  let mut blocks_compressed = 0u8;
  kernels::chunk_compress_blocks_inline(
    kernel_id,
    &mut cv,
    counter,
    flags,
    &mut blocks_compressed,
    &input[..CHUNK_LEN - BLOCK_LEN],
  );
  debug_assert_eq!(blocks_compressed, 15);
  let mut last_block = [0u8; BLOCK_LEN];
  last_block.copy_from_slice(&input[CHUNK_LEN - BLOCK_LEN..]);
  (cv, last_block)
}

#[inline]
fn should_use_exact_one_chunk_fast_path(_kernel_id: kernels::Blake3KernelId) -> bool {
  #[cfg(target_arch = "x86_64")]
  {
    if matches!(
      _kernel_id,
      kernels::Blake3KernelId::X86Ssse3
        | kernels::Blake3KernelId::X86Sse41
        | kernels::Blake3KernelId::X86Avx2
        | kernels::Blake3KernelId::X86Avx512
    ) {
      return true;
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    if _kernel_id == kernels::Blake3KernelId::Aarch64Neon {
      return true;
    }
  }

  false
}

#[inline]
fn parent_output(
  kernel_id: kernels::Blake3KernelId,
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> OutputState {
  let mut block_words = [0u32; 16];
  block_words[..8].copy_from_slice(&left_child_cv);
  block_words[8..].copy_from_slice(&right_child_cv);
  OutputState {
    kernel_id,
    input_chaining_value: key_words,
    block_words,
    counter: 0,
    block_len: BLOCK_LEN as u32,
    flags: PARENT | flags,
  }
}

#[inline]
fn single_chunk_output(
  kernel: Kernel,
  key_words: [u32; 8],
  chunk_counter: u64,
  flags: u32,
  input: &[u8],
) -> OutputState {
  debug_assert!(input.len() <= CHUNK_LEN);

  // aarch64 fast path for a full chunk.
  //
  // This is used by oneshot XOF (and any other oneshot path that needs an OutputState)
  // to avoid per-block compression for 1024B inputs.
  #[cfg(target_arch = "aarch64")]
  {
    if kernel.id == kernels::Blake3KernelId::Aarch64Neon && input.len() == CHUNK_LEN {
      let mut cv_words = [0u32; 8];
      let mut last_block = [0u8; BLOCK_LEN];

      // SAFETY: `input` is exactly one full chunk, and this kernel is only selected
      // when its required CPU features are available.
      unsafe {
        aarch64::chunk_state_one_chunk_aarch64_out(
          input.as_ptr(),
          &key_words,
          chunk_counter,
          flags,
          cv_words.as_mut_ptr(),
          last_block.as_mut_ptr(),
        );
      }

      let block_words = words16_from_le_bytes_64(&last_block);
      return OutputState {
        kernel_id: kernel.id,
        input_chaining_value: cv_words,
        block_words,
        counter: chunk_counter,
        block_len: BLOCK_LEN as u32,
        flags: flags | CHUNK_END,
      };
    }
  }

  // We always emit an OutputState for the last chunk block, even when the
  // input is empty (which is treated as a single, 0-length block).
  let blocks = core::cmp::max(1usize, input.len().div_ceil(BLOCK_LEN));
  let (full_blocks, last_len) = if input.is_empty() {
    (0usize, 0usize)
  } else if input.len().is_multiple_of(BLOCK_LEN) {
    (blocks - 1, BLOCK_LEN)
  } else {
    (blocks - 1, input.len() % BLOCK_LEN)
  };

  let mut chaining_value = key_words;
  let mut blocks_compressed: u8 = 0;
  let full_bytes = full_blocks * BLOCK_LEN;
  (kernel.chunk_compress_blocks)(
    &mut chaining_value,
    chunk_counter,
    flags,
    &mut blocks_compressed,
    &input[..full_bytes],
  );

  let block_words = if cfg!(target_endian = "little") {
    let mut out = [0u32; 16];
    if !input.is_empty() {
      let offset = full_blocks * BLOCK_LEN;
      // SAFETY: `out` is 64 bytes, and `last_len <= 64`.
      unsafe {
        ptr::copy_nonoverlapping(input.as_ptr().add(offset), out.as_mut_ptr().cast::<u8>(), last_len);
      }
    }
    out
  } else {
    let mut last_block = [0u8; BLOCK_LEN];
    if !input.is_empty() {
      let offset = full_blocks * BLOCK_LEN;
      last_block[..last_len].copy_from_slice(&input[offset..offset + last_len]);
    }
    words16_from_le_bytes_64(&last_block)
  };
  let start = if blocks_compressed == 0 { CHUNK_START } else { 0 };

  OutputState {
    kernel_id: kernel.id,
    input_chaining_value: chaining_value,
    block_words,
    counter: chunk_counter,
    block_len: last_len as u32,
    flags: flags | start | CHUNK_END,
  }
}

#[inline]
fn root_output_oneshot(
  kernel: Kernel,
  key_words: [u32; 8],
  flags: u32,
  mode: ParallelPolicyKind,
  input: &[u8],
) -> OutputState {
  // Fast path for <= 1 chunk (root is the chunk itself).
  if input.len() <= CHUNK_LEN {
    return single_chunk_output(kernel, key_words, 0, flags, input);
  }

  let full_chunks = input.len() / CHUNK_LEN;
  let remainder = input.len() % CHUNK_LEN;
  const MAX_SIMD_DEGREE: usize = 16;
  const FAST_TREE_MAX_CHUNKS: usize = 128;

  #[cfg(not(feature = "parallel"))]
  let _ = mode;
  #[cfg(feature = "parallel")]
  {
    // Large input throughput: compute full-chunk CVs in parallel, but keep the
    // tree-reduction logic identical (canonical BLAKE3 shape).
    //
    // This is intentionally conservative to avoid overhead on latency-critical
    // small inputs (including keyed/derive).
    let commit_full_chunks = if remainder == 0 { full_chunks - 1 } else { full_chunks };
    if let Some(threads) = parallel_policy_threads_with_admission(mode, input.len(), full_chunks, commit_full_chunks) {
      return root_output_oneshot_join_parallel(kernel, key_words, flags, input, threads);
    }
  }

  // Small exact trees are common in oneshot benchmarks (4KiB/16KiB). For these,
  // bypass the generic CV-stack builder and reduce the leaves directly.
  if remainder == 0 && full_chunks.is_power_of_two() {
    if full_chunks <= MAX_SIMD_DEGREE {
      #[cfg(target_endian = "little")]
      {
        let mut cur = [[0u32; 8]; MAX_SIMD_DEGREE];
        let mut next = [[0u32; 8]; MAX_SIMD_DEGREE / 2];

        // SAFETY: input has exactly `full_chunks * CHUNK_LEN` bytes and `cur` has
        // `full_chunks` CV slots (`full_chunks * OUT_LEN` bytes).
        unsafe {
          (kernel.hash_many_contiguous)(
            input.as_ptr(),
            full_chunks,
            &key_words,
            0,
            flags,
            cur.as_mut_ptr().cast::<u8>(),
          )
        };

        let mut cur_len = full_chunks;
        let mut cur_is_cur = true;
        while cur_len > 2 {
          let pairs = cur_len / 2;
          if cur_is_cur {
            kernels::parent_cvs_many_from_cvs_inline(kernel.id, &cur[..cur_len], key_words, flags, &mut next[..pairs]);
          } else {
            kernels::parent_cvs_many_from_cvs_inline(kernel.id, &next[..cur_len], key_words, flags, &mut cur[..pairs]);
          }
          cur_is_cur = !cur_is_cur;
          cur_len = pairs;
        }
        if cur_is_cur {
          return parent_output(kernel.id, cur[0], cur[1], key_words, flags);
        }
        return parent_output(kernel.id, next[0], next[1], key_words, flags);
      }

      #[cfg(not(target_endian = "little"))]
      {
        let mut cur = [[0u8; OUT_LEN]; MAX_SIMD_DEGREE];
        let mut next = [[0u8; OUT_LEN]; MAX_SIMD_DEGREE / 2];

        // SAFETY: input has exactly `full_chunks * CHUNK_LEN` bytes and `cur` has
        // `full_chunks` CV slots (`full_chunks * OUT_LEN` bytes).
        unsafe {
          (kernel.hash_many_contiguous)(
            input.as_ptr(),
            full_chunks,
            &key_words,
            0,
            flags,
            cur.as_mut_ptr().cast::<u8>(),
          )
        };

        let mut cur_len = full_chunks;
        let mut cur_is_cur = true;
        while cur_len > 2 {
          let pairs = cur_len / 2;
          if cur_is_cur {
            kernels::parent_cvs_many_from_bytes_inline(
              kernel.id,
              &cur[..cur_len],
              key_words,
              flags,
              &mut next[..pairs],
            );
          } else {
            kernels::parent_cvs_many_from_bytes_inline(
              kernel.id,
              &next[..cur_len],
              key_words,
              flags,
              &mut cur[..pairs],
            );
          }
          cur_is_cur = !cur_is_cur;
          cur_len = pairs;
        }
        let final_cvs: &[[u8; OUT_LEN]] = if cur_is_cur { &cur[..] } else { &next[..] };
        let left = words8_from_le_bytes_32(&final_cvs[0]);
        let right = words8_from_le_bytes_32(&final_cvs[1]);
        return parent_output(kernel.id, left, right, key_words, flags);
      }
    }

    if full_chunks <= FAST_TREE_MAX_CHUNKS {
      #[cfg(target_endian = "little")]
      {
        let mut cur = [[0u32; 8]; FAST_TREE_MAX_CHUNKS];
        let mut next = [[0u32; 8]; FAST_TREE_MAX_CHUNKS / 2];

        // SAFETY: input has exactly `full_chunks * CHUNK_LEN` bytes and `cur` has
        // `full_chunks` CV slots (`full_chunks * OUT_LEN` bytes).
        unsafe {
          (kernel.hash_many_contiguous)(
            input.as_ptr(),
            full_chunks,
            &key_words,
            0,
            flags,
            cur.as_mut_ptr().cast::<u8>(),
          )
        };

        let mut cur_len = full_chunks;
        let mut cur_is_cur = true;
        while cur_len > 2 {
          let pairs = cur_len / 2;
          if cur_is_cur {
            kernels::parent_cvs_many_from_cvs_inline(kernel.id, &cur[..cur_len], key_words, flags, &mut next[..pairs]);
          } else {
            kernels::parent_cvs_many_from_cvs_inline(kernel.id, &next[..cur_len], key_words, flags, &mut cur[..pairs]);
          }
          cur_is_cur = !cur_is_cur;
          cur_len = pairs;
        }
        if cur_is_cur {
          return parent_output(kernel.id, cur[0], cur[1], key_words, flags);
        }
        return parent_output(kernel.id, next[0], next[1], key_words, flags);
      }

      #[cfg(not(target_endian = "little"))]
      {
        let mut cur = [[0u8; OUT_LEN]; FAST_TREE_MAX_CHUNKS];
        let mut next = [[0u8; OUT_LEN]; FAST_TREE_MAX_CHUNKS / 2];

        // SAFETY: input has exactly `full_chunks * CHUNK_LEN` bytes and `cur` has
        // `full_chunks` CV slots (`full_chunks * OUT_LEN` bytes).
        unsafe {
          (kernel.hash_many_contiguous)(
            input.as_ptr(),
            full_chunks,
            &key_words,
            0,
            flags,
            cur.as_mut_ptr().cast::<u8>(),
          )
        };

        let mut cur_len = full_chunks;
        let mut cur_is_cur = true;
        while cur_len > 2 {
          let pairs = cur_len / 2;
          if cur_is_cur {
            kernels::parent_cvs_many_from_bytes_inline(
              kernel.id,
              &cur[..cur_len],
              key_words,
              flags,
              &mut next[..pairs],
            );
          } else {
            kernels::parent_cvs_many_from_bytes_inline(
              kernel.id,
              &next[..cur_len],
              key_words,
              flags,
              &mut cur[..pairs],
            );
          }
          cur_is_cur = !cur_is_cur;
          cur_len = pairs;
        }
        let final_cvs: &[[u8; OUT_LEN]] = if cur_is_cur { &cur[..] } else { &next[..] };
        let left = words8_from_le_bytes_32(&final_cvs[0]);
        let right = words8_from_le_bytes_32(&final_cvs[1]);
        return parent_output(kernel.id, left, right, key_words, flags);
      }
    }
  }

  // Local CV stack to avoid constructing a full streaming hasher.
  let mut cv_stack: [MaybeUninit<[u32; 8]>; CV_STACK_LEN] = uninit_cv_stack();
  let mut cv_stack_len = 0usize;
  let right_cv = {
    #[cfg(target_endian = "little")]
    {
      let mut cvs = [[0u32; 8]; MAX_SIMD_DEGREE];
      let mut last_full_chunk_cv: Option<[u32; 8]> = None;

      let mut chunk_counter = 0u64;
      let mut offset = 0usize;
      while chunk_counter < full_chunks as u64 {
        let remaining = (full_chunks as u64 - chunk_counter) as usize;
        let batch = core::cmp::min(remaining, MAX_SIMD_DEGREE);
        debug_assert!(batch != 0);

        // SAFETY: `offset` is within `input`, and `cvs` has at least `batch`
        // CV slots (`batch * OUT_LEN` bytes).
        unsafe {
          (kernel.hash_many_contiguous)(
            input.as_ptr().add(offset),
            batch,
            &key_words,
            chunk_counter,
            flags,
            cvs.as_mut_ptr().cast::<u8>(),
          )
        };

        let mut commit = batch;
        if remainder == 0 && chunk_counter + batch as u64 == full_chunks as u64 {
          last_full_chunk_cv = Some(cvs[batch - 1]);
          commit -= 1;
        }

        if commit != 0 {
          add_chunk_cvs_batched(
            kernel,
            &mut cv_stack,
            &mut cv_stack_len,
            chunk_counter,
            &cvs[..commit],
            key_words,
            flags,
          );
        }

        chunk_counter += batch as u64;
        offset += batch * CHUNK_LEN;
      }

      if remainder != 0 {
        let chunk_bytes = &input[full_chunks * CHUNK_LEN..];
        single_chunk_output(kernel, key_words, full_chunks as u64, flags, chunk_bytes).chaining_value()
      } else {
        // `input.len() > CHUNK_LEN` implies there are at least 2 chunks total,
        // so the root output is always derived from parent nodes.
        match last_full_chunk_cv {
          Some(cv) => cv,
          None => unreachable!("missing last full chunk cv"),
        }
      }
    }

    #[cfg(not(target_endian = "little"))]
    {
      let mut cvs = [[0u8; OUT_LEN]; MAX_SIMD_DEGREE];
      let mut last_full_chunk_cv: Option<CvBytes> = None;

      let mut chunk_counter = 0u64;
      let mut offset = 0usize;
      while chunk_counter < full_chunks as u64 {
        let remaining = (full_chunks as u64 - chunk_counter) as usize;
        let batch = core::cmp::min(remaining, MAX_SIMD_DEGREE);
        debug_assert!(batch != 0);

        // SAFETY: `offset` is within `input`, and `cvs` is large enough for `batch`.
        unsafe {
          (kernel.hash_many_contiguous)(
            input.as_ptr().add(offset),
            batch,
            &key_words,
            chunk_counter,
            flags,
            cvs.as_mut_ptr().cast::<u8>(),
          )
        };

        let mut commit = batch;
        if remainder == 0 && chunk_counter + batch as u64 == full_chunks as u64 {
          last_full_chunk_cv = Some(cvs[batch - 1]);
          commit -= 1;
        }

        if commit != 0 {
          add_chunk_cvs_batched_bytes(
            kernel,
            &mut cv_stack,
            &mut cv_stack_len,
            chunk_counter,
            &cvs[..commit],
            key_words,
            flags,
          );
        }

        chunk_counter += batch as u64;
        offset += batch * CHUNK_LEN;
      }

      if remainder != 0 {
        let chunk_bytes = &input[full_chunks * CHUNK_LEN..];
        single_chunk_output(kernel, key_words, full_chunks as u64, flags, chunk_bytes).chaining_value()
      } else {
        // `input.len() > CHUNK_LEN` implies there are at least 2 chunks total, so
        // the root output is always derived from parent nodes rather than a chunk.
        match last_full_chunk_cv {
          Some(cv) => words8_from_le_bytes_32(&cv),
          None => unreachable!("missing last full chunk cv"),
        }
      }
    }
  };

  let mut parent_nodes_remaining = cv_stack_len;
  debug_assert!(parent_nodes_remaining > 0);
  parent_nodes_remaining -= 1;
  // SAFETY: `cv_stack_len` tracks the number of initialized entries.
  let left = unsafe { cv_stack[parent_nodes_remaining].assume_init_read() };
  let mut output = parent_output(kernel.id, left, right_cv, key_words, flags);
  while parent_nodes_remaining > 0 {
    parent_nodes_remaining -= 1;
    // SAFETY: `cv_stack_len` tracks the number of initialized entries.
    let left = unsafe { cv_stack[parent_nodes_remaining].assume_init_read() };
    output = parent_output(kernel.id, left, output.chaining_value(), key_words, flags);
  }

  output
}

#[inline]
fn digest_oneshot_words(kernel: Kernel, key_words: [u32; 8], flags: u32, input: &[u8]) -> [u32; 8] {
  // Ultra-fast path for tiny inputs (≤64B): use unified helper
  if input.len() <= BLOCK_LEN {
    return hash_tiny_to_root_words(kernel, key_words, flags, input);
  }

  // Fast path for single-chunk inputs (≤1024B): use platform-specific helpers
  #[cfg(target_arch = "x86_64")]
  {
    if input.len() <= CHUNK_LEN {
      match kernel.id {
        kernels::Blake3KernelId::X86Sse41 | kernels::Blake3KernelId::X86Avx2 | kernels::Blake3KernelId::X86Avx512 => {
          // SAFETY: x86 SIMD availability is validated by dispatch before selecting these kernels.
          return unsafe { digest_one_chunk_root_hash_words_x86(kernel, key_words, flags, input) };
        }
        _ => {}
      }
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    if input.len() <= CHUNK_LEN && kernel.id == kernels::Blake3KernelId::Aarch64Neon {
      // SAFETY: aarch64 NEON availability is validated by dispatch before selecting this kernel.
      return unsafe { digest_one_chunk_root_hash_words_aarch64(kernel, key_words, flags, input) };
    }
  }

  if input.len() <= CHUNK_LEN {
    return digest_one_chunk_root_hash_words_generic(kernel, key_words, flags, input);
  }

  // Fallback: keep the large-input path in a cold function to avoid
  // inflating short-input codegen in this hot entry point.
  digest_oneshot_words_fallback(kernel, key_words, flags, input)
}

#[cold]
#[inline(never)]
fn digest_oneshot_words_fallback(kernel: Kernel, key_words: [u32; 8], flags: u32, input: &[u8]) -> [u32; 8] {
  let mode = policy_kind_from_flags(flags, false); // is_xof = false
  let output = root_output_oneshot(kernel, key_words, flags, mode, input);
  output.root_hash_words()
}

#[inline]
fn policy_kind_from_flags(flags: u32, is_xof: bool) -> ParallelPolicyKind {
  match (is_xof, flags) {
    (false, 0) => ParallelPolicyKind::Oneshot,
    (false, KEYED_HASH) => ParallelPolicyKind::KeyedOneshot,
    (false, DERIVE_KEY_MATERIAL) => ParallelPolicyKind::DeriveOneshot,
    (true, 0) => ParallelPolicyKind::Xof,
    (true, KEYED_HASH) => ParallelPolicyKind::KeyedXof,
    (true, DERIVE_KEY_MATERIAL) => ParallelPolicyKind::DeriveXof,
    _ => ParallelPolicyKind::Oneshot, // Fallback for unknown flag combinations
  }
}

#[inline]
fn digest_oneshot(kernel: Kernel, key_words: [u32; 8], flags: u32, input: &[u8]) -> [u8; OUT_LEN] {
  #[cfg(target_arch = "aarch64")]
  {
    if kernel.id == kernels::Blake3KernelId::Aarch64Neon && input.len() == CHUNK_LEN {
      // SAFETY: aarch64 NEON is validated by dispatch before selecting this kernel.
      return unsafe { aarch64::root_hash_one_chunk_root_aarch64(input.as_ptr(), &key_words, flags) };
    }
  }

  words8_to_le_bytes(&digest_oneshot_words(kernel, key_words, flags, input))
}

#[inline]
fn digest_public_oneshot(key_words: [u32; 8], flags: u32, input: &[u8]) -> [u8; OUT_LEN] {
  let kernel = dispatch::hasher_dispatch().size_class_kernel(input.len());
  digest_oneshot(kernel, key_words, flags, input)
}

#[derive(Clone)]
pub struct Blake3 {
  bulk_kernel_id: kernels::Blake3KernelId,
  chunk_state: ChunkState,
  key_words: [u32; 8],
  cv_stack: [MaybeUninit<[u32; 8]>; CV_STACK_LEN],
  cv_stack_len: u8,
}

impl core::fmt::Debug for Blake3 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Blake3").finish_non_exhaustive()
  }
}

impl Default for Blake3 {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl Blake3 {
  /// Compute the hash of `data` in one shot.
  ///
  /// This selects the best available kernel for the current platform and input
  /// length (cached after first use).
  #[inline]
  #[must_use]
  pub fn digest(data: &[u8]) -> [u8; OUT_LEN] {
    digest_public_oneshot(IV, 0, data)
  }

  /// Compute the XOF output state of `data` in one shot.
  ///
  /// This avoids constructing a full streaming hasher. It's useful when you
  /// immediately want to squeeze output without incremental updates.
  #[inline]
  #[must_use]
  pub fn xof(data: &[u8]) -> Blake3Xof {
    dispatch::xof(data)
  }

  /// Compute the keyed hash of `data` in one shot.
  ///
  /// This uses dedicated fast paths for tiny inputs (≤64B) to minimize dispatch
  /// and finalize overhead, which is critical for latency-sensitive keyed hashing.
  #[inline]
  #[must_use]
  pub fn keyed_digest(key: &[u8; KEY_LEN], data: &[u8]) -> [u8; OUT_LEN] {
    #[cfg(feature = "std")]
    let key_words = keyed_words_cached(key);
    #[cfg(not(feature = "std"))]
    let key_words = words8_from_le_bytes_32(key);
    digest_public_oneshot(key_words, KEYED_HASH, data)
  }

  /// Compute the keyed XOF output state of `data` in one shot.
  #[inline]
  #[must_use]
  pub fn keyed_xof(key: &[u8; KEY_LEN], data: &[u8]) -> Blake3Xof {
    let key_words = words8_from_le_bytes_32(key);
    let plan = dispatch::hasher_dispatch();
    let kernel = plan.size_class_kernel(data.len());
    Blake3Xof::from_output(root_output_oneshot(
      kernel,
      key_words,
      KEYED_HASH,
      policy_kind_from_flags(KEYED_HASH, true),
      data,
    ))
  }

  /// Compute the derived key for `key_material` under `context`, in one shot.
  ///
  /// This uses context caching to avoid re-hashing the same context string
  /// repeatedly, which is a common pattern in practice. Tiny key material
  /// inputs (≤64B) also use dedicated fast paths to minimize latency.
  #[inline]
  #[must_use]
  pub fn derive_key(context: &str, key_material: &[u8]) -> [u8; OUT_LEN] {
    let context_key_words = {
      #[cfg(feature = "std")]
      {
        derive_context_key_words_cached(context)
      }
      #[cfg(not(feature = "std"))]
      {
        let context_bytes = context.as_bytes();
        let kernel_ctx = dispatch::hasher_dispatch().size_class_kernel(context_bytes.len());
        digest_oneshot_words(kernel_ctx, IV, DERIVE_KEY_CONTEXT, context_bytes)
      }
    };

    digest_public_oneshot(context_key_words, DERIVE_KEY_MATERIAL, key_material)
  }

  /// One-shot hash using an explicitly selected kernel.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn digest_with_kernel_id(id: kernels::Blake3KernelId, data: &[u8]) -> [u8; OUT_LEN] {
    let kernel = kernels::kernel(id);
    digest_oneshot(kernel, IV, 0, data)
  }

  /// One-shot keyed hash using an explicitly selected kernel.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn keyed_digest_with_kernel_id(id: kernels::Blake3KernelId, key: &[u8; 32], data: &[u8]) -> [u8; OUT_LEN] {
    let kernel = kernels::kernel(id);
    let key_words = words8_from_le_bytes_32(key);
    digest_oneshot(kernel, key_words, KEYED_HASH, data)
  }

  /// One-shot derive-key hash using an explicitly selected kernel.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn derive_key_with_kernel_id(
    id: kernels::Blake3KernelId,
    context: &str,
    key_material: &[u8],
  ) -> [u8; OUT_LEN] {
    let kernel = kernels::kernel(id);
    let context_key_words = digest_oneshot_words(kernel, IV, DERIVE_KEY_CONTEXT, context.as_bytes());
    digest_oneshot(kernel, context_key_words, DERIVE_KEY_MATERIAL, key_material)
  }

  #[inline]
  #[cfg(any(test, feature = "std"))]
  fn stream_chunks_pattern_with_kernel_pair_and_state(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_pattern: &[usize],
    key_words: [u32; 8],
    flags: u32,
    xof_mode: bool,
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    let mut h = Self::new_internal_with(key_words, flags, stream_id);
    if chunk_pattern.is_empty() {
      h.update_with(data, stream_id, bulk_id);
    } else {
      let mut offset = 0usize;
      let mut idx = 0usize;
      while offset < data.len() {
        let step = chunk_pattern[idx % chunk_pattern.len()].max(1);
        let end = offset.saturating_add(step).min(data.len());
        h.update_with(&data[offset..end], stream_id, bulk_id);
        offset = end;
        idx = idx.saturating_add(1);
      }
    }

    if xof_mode {
      let mut xof = h.finalize_xof();
      let mut out = [0u8; OUT_LEN];
      xof.squeeze(&mut out);
      out
    } else {
      h.finalize()
    }
  }

  #[inline]
  #[cfg(any(test, feature = "std"))]
  fn stream_chunks_with_kernel_pair_and_state(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_size: usize,
    key_words: [u32; 8],
    flags: u32,
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    Self::stream_chunks_pattern_with_kernel_pair_and_state(
      stream_id,
      bulk_id,
      core::slice::from_ref(&chunk_size),
      key_words,
      flags,
      false,
      data,
    )
  }

  /// Streaming hash with explicit `(stream, bulk)` kernel IDs.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn stream_chunks_with_kernel_pair_id(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_size: usize,
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    Self::stream_chunks_with_kernel_pair_and_state(stream_id, bulk_id, chunk_size, IV, 0, data)
  }

  /// Keyed streaming hash with explicit `(stream, bulk)` kernel IDs.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn stream_chunks_keyed_with_kernel_pair_id(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_size: usize,
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    const STREAM_BENCH_KEY: [u8; KEY_LEN] = [
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12,
      0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
    ];
    let key_words = words8_from_le_bytes_32(&STREAM_BENCH_KEY);
    Self::stream_chunks_with_kernel_pair_and_state(stream_id, bulk_id, chunk_size, key_words, KEYED_HASH, data)
  }

  /// Derive-key-material streaming hash with explicit `(stream, bulk)` kernel IDs.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn stream_chunks_derive_with_kernel_pair_id(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_size: usize,
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    const STREAM_BENCH_CONTEXT: &str = "rscrypto-blake3-stream-bench";
    let stream = kernels::kernel(stream_id);
    let context_key_words = digest_oneshot_words(stream, IV, DERIVE_KEY_CONTEXT, STREAM_BENCH_CONTEXT.as_bytes());
    Self::stream_chunks_with_kernel_pair_and_state(
      stream_id,
      bulk_id,
      chunk_size,
      context_key_words,
      DERIVE_KEY_MATERIAL,
      data,
    )
  }

  /// Streaming XOF with explicit `(stream, bulk)` kernel IDs.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn stream_chunks_xof_with_kernel_pair_id(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_size: usize,
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    Self::stream_chunks_pattern_with_kernel_pair_and_state(
      stream_id,
      bulk_id,
      core::slice::from_ref(&chunk_size),
      IV,
      0,
      true,
      data,
    )
  }

  /// Streaming mixed-pattern hash with explicit `(stream, bulk)` kernel IDs.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn stream_chunks_mixed_with_kernel_pair_id(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_pattern: &[usize],
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    Self::stream_chunks_pattern_with_kernel_pair_and_state(stream_id, bulk_id, chunk_pattern, IV, 0, false, data)
  }

  /// Keyed streaming mixed-pattern hash with explicit `(stream, bulk)` kernel IDs.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn stream_chunks_mixed_keyed_with_kernel_pair_id(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_pattern: &[usize],
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    const STREAM_BENCH_KEY: [u8; KEY_LEN] = [
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12,
      0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
    ];
    let key_words = words8_from_le_bytes_32(&STREAM_BENCH_KEY);
    Self::stream_chunks_pattern_with_kernel_pair_and_state(
      stream_id,
      bulk_id,
      chunk_pattern,
      key_words,
      KEYED_HASH,
      false,
      data,
    )
  }

  /// Derive-key-material streaming mixed-pattern hash with explicit `(stream, bulk)` kernel IDs.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn stream_chunks_mixed_derive_with_kernel_pair_id(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_pattern: &[usize],
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    const STREAM_BENCH_CONTEXT: &str = "rscrypto-blake3-stream-bench";
    let stream = kernels::kernel(stream_id);
    let context_key_words = digest_oneshot_words(stream, IV, DERIVE_KEY_CONTEXT, STREAM_BENCH_CONTEXT.as_bytes());
    Self::stream_chunks_pattern_with_kernel_pair_and_state(
      stream_id,
      bulk_id,
      chunk_pattern,
      context_key_words,
      DERIVE_KEY_MATERIAL,
      false,
      data,
    )
  }

  /// Streaming mixed-pattern XOF with explicit `(stream, bulk)` kernel IDs.
  ///
  /// This is crate-internal glue for `hashes::bench`.
  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn stream_chunks_mixed_xof_with_kernel_pair_id(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_pattern: &[usize],
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    Self::stream_chunks_pattern_with_kernel_pair_and_state(stream_id, bulk_id, chunk_pattern, IV, 0, true, data)
  }

  #[inline]
  #[must_use]
  #[cfg(any(test, feature = "std"))]
  pub(crate) fn digest_portable(data: &[u8]) -> [u8; OUT_LEN] {
    let kernel = kernels::kernel(kernels::Blake3KernelId::Portable);
    if data.len() <= CHUNK_LEN {
      let output = single_chunk_output(kernel, IV, 0, 0, data);
      output.root_hash_bytes()
    } else {
      let mut h = Self::new_internal_with(IV, 0, kernels::Blake3KernelId::Portable);
      h.update_with(
        data,
        kernels::Blake3KernelId::Portable,
        kernels::Blake3KernelId::Portable,
      );
      h.finalize()
    }
  }

  #[inline]
  fn new_internal(key_words: [u32; 8], flags: u32) -> Self {
    #[cfg(feature = "std")]
    let kernel_id = dispatch::stream_kernel_id_ref();
    #[cfg(not(feature = "std"))]
    let kernel_id = dispatch::hasher_dispatch().stream_kernel().id;
    Self::new_internal_with(key_words, flags, kernel_id)
  }

  #[inline]
  fn new_internal_with(key_words: [u32; 8], flags: u32, kernel_id: kernels::Blake3KernelId) -> Self {
    Self {
      bulk_kernel_id: kernel_id,
      chunk_state: ChunkState::new(key_words, 0, flags, kernel_id),
      key_words,
      cv_stack: uninit_cv_stack(),
      cv_stack_len: 0,
    }
  }

  #[cfg(feature = "parallel")]
  #[inline]
  fn streaming_parallel_threads(
    &self,
    input_bytes: usize,
    admission_full_chunks: usize,
    commit_full_chunks: usize,
  ) -> Option<usize> {
    let (policy, mode) = if let Some(override_policy) = parallel_override() {
      let mode = match self.chunk_state.flags {
        0 => ParallelPolicyKind::Update,
        KEYED_HASH => ParallelPolicyKind::KeyedUpdate,
        DERIVE_KEY_MATERIAL => ParallelPolicyKind::DeriveUpdate,
        _ => ParallelPolicyKind::Update,
      };
      let policy = match mode {
        ParallelPolicyKind::Update => override_policy.streaming,
        ParallelPolicyKind::KeyedUpdate => override_policy.keyed_streaming,
        ParallelPolicyKind::DeriveUpdate => override_policy.derive_streaming,
        _ => override_policy.streaming,
      };
      (policy, mode)
    } else {
      (
        dispatch::hasher_dispatch().parallel_streaming(),
        ParallelPolicyKind::Update,
      )
    };
    parallel_threads_for_policy(mode, policy, input_bytes, admission_full_chunks, commit_full_chunks)
  }

  fn update_with(
    &mut self,
    mut input: &[u8],
    stream_kernel_id: kernels::Blake3KernelId,
    bulk_kernel_id: kernels::Blake3KernelId,
  ) {
    self.bulk_kernel_id = bulk_kernel_id;
    self.chunk_state.kernel_id = stream_kernel_id;

    // Phase 1: finish partial chunk if any.
    if self.chunk_state.len() > 0 {
      let want = CHUNK_LEN.strict_sub(self.chunk_state.len());
      let take = min(want, input.len());
      self.chunk_state.update(&input[..take]);
      input = &input[take..];
      if input.is_empty() {
        return;
      }
      // Chunk is full — eagerly merge its CV into the tree.
      let chunk_cv = self.chunk_state.output().chaining_value();
      let total_chunks = self.chunk_state.chunk_counter.strict_add(1);
      self.add_cv_to_tree(chunk_cv, total_chunks, 0);
      self.chunk_state = ChunkState::new(
        self.key_words,
        total_chunks,
        self.chunk_state.flags,
        self.chunk_state.kernel_id,
      );
    }

    // Phase 2: power-of-2 subtree loop.
    // Invariant: always leave at least CHUNK_LEN bytes for Phase 3 so that
    // chunk_state is never empty at finalization. This avoids needing a
    // pending_chunk_cv mechanism.
    let kernel = kernels::kernel(bulk_kernel_id);
    while input.len() > CHUNK_LEN {
      debug_assert_eq!(self.chunk_state.len(), 0);
      let mut subtree_len = pow2_floor(input.len());
      subtree_len = subtree_len.max(CHUNK_LEN);

      // If the subtree would consume ALL remaining input, halve it
      // so that Phase 3 always has data to buffer.
      if subtree_len >= input.len() {
        subtree_len /= 2;
        if subtree_len < CHUNK_LEN {
          break;
        }
      }

      // Shrink until aligned with the byte count so far.
      let count_so_far = self.chunk_state.chunk_counter.strict_mul(CHUNK_LEN as u64);
      while (subtree_len as u64).strict_sub(1) & count_so_far != 0 {
        subtree_len /= 2;
      }
      let subtree_chunks = (subtree_len / CHUNK_LEN) as u64;

      if subtree_len <= CHUNK_LEN {
        debug_assert_eq!(subtree_len, CHUNK_LEN);
        let cv = single_chunk_output(
          kernel,
          self.key_words,
          self.chunk_state.chunk_counter,
          self.chunk_state.flags,
          &input[..CHUNK_LEN],
        )
        .chaining_value();
        let total = self.chunk_state.chunk_counter.strict_add(1);
        self.add_cv_to_tree(cv, total, 0);
      } else {
        // Subtree compression — the high-performance path.
        #[cfg(feature = "parallel")]
        let cv_pair = {
          let budget = self.parallel_budget_for_subtree(input.len());
          if budget > 0 {
            compress_subtree_to_parent_node_bytes::<join::RayonJoin>(
              kernel,
              self.key_words,
              self.chunk_state.chunk_counter,
              self.chunk_state.flags,
              &input[..subtree_len],
              budget,
            )
          } else {
            compress_subtree_to_parent_node_bytes::<join::SerialJoin>(
              kernel,
              self.key_words,
              self.chunk_state.chunk_counter,
              self.chunk_state.flags,
              &input[..subtree_len],
              0,
            )
          }
        };
        #[cfg(not(feature = "parallel"))]
        let cv_pair = compress_subtree_to_parent_node_bytes::<join::SerialJoin>(
          kernel,
          self.key_words,
          self.chunk_state.chunk_counter,
          self.chunk_state.flags,
          &input[..subtree_len],
          0,
        );
        // SAFETY: cv_pair is [u8; 64]; first 32 bytes are the left CV, next 32 are the right.
        let left_cv = words8_from_le_bytes_32(unsafe { &*(cv_pair.as_ptr() as *const [u8; 32]) });
        // SAFETY: cv_pair is [u8; 64]; bytes [32..64] are the right CV, pointer arithmetic is in-bounds.
        let right_cv = words8_from_le_bytes_32(unsafe { &*(cv_pair.as_ptr().add(OUT_LEN) as *const [u8; 32]) });
        // Compute the subtree's parent CV and eagerly merge it into the tree.
        let parent_cv = kernels::parent_cv_inline(kernel.id, left_cv, right_cv, self.key_words, self.chunk_state.flags);
        let total = self.chunk_state.chunk_counter.strict_add(subtree_chunks);
        self.add_cv_to_tree(parent_cv, total, subtree_chunks.trailing_zeros());
      }
      self.chunk_state.chunk_counter = self.chunk_state.chunk_counter.strict_add(subtree_chunks);
      input = &input[subtree_len..];
    }

    // Phase 3: remaining bytes → buffer in chunk_state.
    // This always runs when input > CHUNK_LEN was provided, ensuring
    // chunk_state is non-empty at finalization time.
    if !input.is_empty() {
      debug_assert_eq!(self.chunk_state.len(), 0);
      debug_assert!(input.len() <= CHUNK_LEN);
      self.chunk_state.update(input);
    }
  }

  /// Construct a new hasher for the keyed hash function.
  #[must_use]
  #[inline]
  pub fn new_keyed(key: &[u8; KEY_LEN]) -> Self {
    let key_words = words8_from_le_bytes_32(key);
    Self::new_internal(key_words, KEYED_HASH)
  }

  /// Construct a new hasher for the key derivation function.
  #[must_use]
  #[inline]
  pub fn new_derive_key(context: &str) -> Self {
    #[cfg(feature = "std")]
    let key_words = derive_context_key_words_cached(context);
    #[cfg(not(feature = "std"))]
    let key_words = {
      let context_bytes = context.as_bytes();
      let kernel_ctx = dispatch::hasher_dispatch().size_class_kernel(context_bytes.len());
      digest_oneshot_words(kernel_ctx, IV, DERIVE_KEY_CONTEXT, context_bytes)
    };
    Self::new_internal(key_words, DERIVE_KEY_MATERIAL)
  }

  #[inline]
  fn push_stack(&mut self, cv: [u32; 8]) {
    self.cv_stack[self.cv_stack_len as usize].write(cv);
    self.cv_stack_len = self.cv_stack_len.wrapping_add(1);
  }

  #[inline]
  fn pop_stack(&mut self) -> [u32; 8] {
    self.cv_stack_len = self.cv_stack_len.wrapping_sub(1);
    // SAFETY: `cv_stack_len` tracks the number of initialized entries.
    unsafe { self.cv_stack[self.cv_stack_len as usize].assume_init_read() }
  }

  /// Eagerly merge `new_cv` into the CV stack.
  ///
  /// `total_chunks` is the 1-based total number of committed chunks after this
  /// CV. `subtree_level` is `subtree_chunks.trailing_zeros()` — 0 for a single
  /// chunk, 1 for a 2-chunk subtree parent, etc.
  fn add_cv_to_tree(&mut self, mut new_cv: [u32; 8], total_chunks: u64, subtree_level: u32) {
    let mut shifted = total_chunks >> subtree_level;
    while shifted & 1 == 0 {
      new_cv = kernels::parent_cv_inline(
        self.chunk_state.kernel_id,
        self.pop_stack(),
        new_cv,
        self.key_words,
        self.chunk_state.flags,
      );
      shifted >>= 1;
    }
    self.push_stack(new_cv);
  }

  #[cfg(feature = "parallel")]
  #[inline]
  fn parallel_budget_for_subtree(&self, input_bytes: usize) -> usize {
    let full_chunks = input_bytes / CHUNK_LEN;
    if full_chunks <= 1 {
      return 0;
    }
    let commit = full_chunks;
    match self.streaming_parallel_threads(input_bytes, full_chunks, commit) {
      Some(threads) if threads > 1 => {
        let depth = (usize::BITS.strict_sub(1).strict_sub(threads.leading_zeros())) as usize;
        depth.max(1)
      }
      _ => 0,
    }
  }

  #[inline(never)]
  fn update_digest_slow(&mut self, input: &[u8]) {
    if self.chunk_state.len() == CHUNK_LEN {
      let chunk_cv = self.chunk_state.output().chaining_value();
      let total_chunks = self.chunk_state.chunk_counter.strict_add(1);
      self.add_cv_to_tree(chunk_cv, total_chunks, 0);
      self.chunk_state = ChunkState::new(
        self.key_words,
        total_chunks,
        self.chunk_state.flags,
        self.chunk_state.kernel_id,
      );
    }

    if self.chunk_state.len().strict_add(input.len()) <= CHUNK_LEN {
      self.chunk_state.update(input);
      return;
    }

    let plan = dispatch::hasher_dispatch();
    let stream = self.chunk_state.kernel_id;
    let bulk = plan.bulk_kernel_for_update(input.len()).id;
    self.update_with(input, stream, bulk);
  }

  fn root_output(&self) -> OutputState {
    if self.cv_stack_len == 0 {
      return self.chunk_state.output();
    }

    // Fold the eagerly-merged stack with the chunk state output (right-to-left).
    let mut parent_nodes_remaining = self.cv_stack_len as usize;
    let mut output = self.chunk_state.output();

    while parent_nodes_remaining > 0 {
      parent_nodes_remaining = parent_nodes_remaining.strict_sub(1);
      // SAFETY: `cv_stack_len` tracks the number of initialized entries.
      let left = unsafe { *self.cv_stack[parent_nodes_remaining].assume_init_ref() };
      output = parent_output(
        self.chunk_state.kernel_id,
        left,
        output.chaining_value(),
        self.key_words,
        self.chunk_state.flags,
      );
    }
    output
  }

  #[inline]
  fn root_emit_state(&self) -> RootEmitState {
    if self.cv_stack_len == 0 {
      return self.chunk_state.root_emit_state();
    }

    self.root_emit_state_slow()
  }

  #[inline(never)]
  fn root_emit_state_slow(&self) -> RootEmitState {
    let mut parent_nodes_remaining = self.cv_stack_len as usize;
    let mut current_cv = self.chunk_state.output_chaining_value();

    while parent_nodes_remaining > 0 {
      parent_nodes_remaining = parent_nodes_remaining.strict_sub(1);
      // SAFETY: `cv_stack_len` tracks the number of initialized entries.
      let left = unsafe { *self.cv_stack[parent_nodes_remaining].assume_init_ref() };
      if parent_nodes_remaining == 0 {
        return RootEmitState::from_parent(
          self.chunk_state.kernel_id,
          left,
          current_cv,
          self.key_words,
          self.chunk_state.flags,
        );
      }
      current_cv = kernels::parent_cv_inline(
        self.chunk_state.kernel_id,
        left,
        current_cv,
        self.key_words,
        self.chunk_state.flags,
      );
    }

    unreachable!("root emit state must return from chunk or parent root path");
  }

  /// Finalize into an extendable output state (XOF).
  #[must_use]
  #[inline]
  pub fn finalize_xof(&self) -> Blake3Xof {
    if self.cv_stack_len == 0 {
      return Blake3Xof::new(self.chunk_state.root_emit_state());
    }

    Blake3Xof::new(self.root_emit_state())
  }

  /// Finalize into an extendable output state (XOF) with a size hint.
  ///
  /// This currently behaves identically to [`Self::finalize_xof`]. The size
  /// hint is accepted for API compatibility.
  ///
  /// # Arguments
  /// * `expected_output_bytes` - Expected output length hint (currently unused).
  ///
  /// # Example
  /// ```rust
  /// use rscrypto::{Digest as _, Xof, hashes::crypto::Blake3};
  ///
  /// let mut hasher = Blake3::new();
  /// hasher.update(b"small input");
  /// let mut xof = hasher.finalize_xof_sized(1024);
  /// let mut out = [0u8; 1024];
  /// xof.squeeze(&mut out);
  /// assert_ne!(out, [0u8; 1024]);
  /// ```
  #[must_use]
  #[inline]
  pub fn finalize_xof_sized(&self, expected_output_bytes: usize) -> Blake3Xof {
    let _ = expected_output_bytes;
    self.finalize_xof()
  }
}

impl Drop for Blake3 {
  fn drop(&mut self) {
    for word in self.key_words.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    for word in self.chunk_state.chaining_value.iter_mut() {
      // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
      unsafe { core::ptr::write_volatile(word, 0) };
    }
    crate::traits::ct::zeroize(&mut self.chunk_state.block);
    for slot in self.cv_stack.iter_mut() {
      let slot_ptr = slot as *mut MaybeUninit<[u32; 8]> as *mut u8;
      for i in 0..core::mem::size_of::<[u32; 8]>() {
        // SAFETY: slot_ptr is valid for size_of::<[u32; 8]>() bytes.
        unsafe { core::ptr::write_volatile(slot_ptr.add(i), 0) };
      }
    }
    #[cfg(feature = "std")]
    if self.chunk_state.flags & KEYED_HASH != 0 {
      KEYED_WORDS_LOCAL_CACHE.with(|slot| {
        if let Some((ref mut key_bytes, ref mut words)) = *slot.borrow_mut() {
          crate::traits::ct::zeroize(key_bytes);
          for w in words.iter_mut() {
            // SAFETY: w is a valid, aligned, dereferenceable pointer to initialized memory.
            unsafe { core::ptr::write_volatile(w, 0) };
          }
        }
        *slot.borrow_mut() = None;
      });
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

impl Digest for Blake3 {
  const OUTPUT_SIZE: usize = OUT_LEN;
  type Output = [u8; OUT_LEN];

  #[inline]
  fn digest(data: &[u8]) -> Self::Output {
    Blake3::digest(data)
  }

  #[inline]
  fn new() -> Self {
    Self::new_internal(IV, 0)
  }

  #[inline]
  fn update(&mut self, input: &[u8]) {
    if input.is_empty() {
      return;
    }

    if self.chunk_state.len() != CHUNK_LEN && self.chunk_state.len().strict_add(input.len()) <= CHUNK_LEN {
      self.chunk_state.update(input);
      return;
    }

    self.update_digest_slow(input);
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    // Single-chunk fast path: compute root bytes directly from the current
    // chunk tail. This avoids OutputState construction and bytes<->words
    // conversion on short streaming finalization.
    if self.chunk_state.chunk_counter == 0 && self.cv_stack_len == 0 {
      let block_len = self.chunk_state.block_len as usize;

      let add_chunk_start = self.chunk_state.blocks_compressed == 0;
      let cv = self.chunk_state.chaining_value;
      let kernel = kernels::kernel(self.chunk_state.kernel_id);

      if block_len == BLOCK_LEN {
        let out_words = compress_chunk_tail_to_root_words(
          kernel,
          cv,
          &self.chunk_state.block,
          block_len,
          self.chunk_state.flags,
          add_chunk_start,
        );
        return words8_to_le_bytes(&out_words);
      }

      let mut block = self.chunk_state.block;
      block[block_len..].fill(0);
      let out_words =
        compress_chunk_tail_to_root_words(kernel, cv, &block, block_len, self.chunk_state.flags, add_chunk_start);
      return words8_to_le_bytes(&out_words);
    }

    // If the caller never provided any input (or only provided empty updates),
    // we still want to use the tuned streaming kernel rather than staying
    // pinned to the portable default.
    let output = if self.chunk_state.chunk_counter == 0 && self.chunk_state.len() == 0 && self.cv_stack_len == 0 {
      single_chunk_output(
        kernels::kernel(self.chunk_state.kernel_id),
        self.key_words,
        0,
        self.chunk_state.flags,
        &[],
      )
    } else {
      self.root_output()
    };
    output.root_hash_bytes()
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::new_internal(self.key_words, self.chunk_state.flags);
  }
}

#[derive(Clone)]
pub struct Blake3Xof {
  root: RootEmitState,
  position_within_block: u8,
}

impl core::fmt::Debug for Blake3Xof {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Blake3Xof").finish_non_exhaustive()
  }
}

impl Blake3Xof {
  #[inline]
  fn new(root: RootEmitState) -> Self {
    Self {
      root,
      position_within_block: 0,
    }
  }

  #[inline]
  fn from_output(output: OutputState) -> Self {
    Self::new(output.into_root_emit_state())
  }

  #[inline]
  fn fill_one_block(&mut self, out: &mut &mut [u8]) {
    let mut block = [0u8; OUTPUT_BLOCK_LEN];
    self.root.emit_one_block(&mut block);
    let output_bytes = &block[self.position_within_block as usize..];
    let take = min(out.len(), output_bytes.len());
    out[..take].copy_from_slice(&output_bytes[..take]);
    self.position_within_block += take as u8;
    if self.position_within_block == OUTPUT_BLOCK_LEN as u8 {
      self.root.counter = self.root.counter.wrapping_add(1);
      self.position_within_block = 0;
    }
    *out = &mut core::mem::take(out)[take..];
  }

  fn squeeze_general(&mut self, out: &mut &mut [u8]) {
    if self.position_within_block != 0 {
      self.fill_one_block(out);
    }

    let full = out.len() / OUTPUT_BLOCK_LEN * OUTPUT_BLOCK_LEN;
    if full != 0 {
      let blocks = (full / OUTPUT_BLOCK_LEN) as u64;
      self.root.emit_blocks_into(&mut out[..full]);
      self.root.counter = self.root.counter.wrapping_add(blocks);
      *out = &mut core::mem::take(out)[full..];
    }

    if !out.is_empty() {
      self.fill_one_block(out);
    }
  }
}

impl Xof for Blake3Xof {
  #[inline]
  fn squeeze(&mut self, mut out: &mut [u8]) {
    if out.is_empty() {
      return;
    }

    if out.len() < OUTPUT_BLOCK_LEN {
      self.fill_one_block(&mut out);
      return;
    }

    Blake3Xof::squeeze_general(self, &mut out);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unified tiny-input fast path helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compress a single block directly to the root hash words (8 words).
///
/// This is the unified fast path for tiny inputs (≤64B) used by both streaming
/// finalize() and oneshot operations. It bypasses the generic OutputState
/// machinery to minimize latency for keyed/derive modes.
///
/// # Arguments
/// * `kernel` - The kernel to use (must be a SIMD kernel on supported platforms)
/// * `cv` - The chaining value (key words for keyed mode, IV for plain)
/// * `block` - The input block (must be ≤64 bytes, zero-padded if shorter)
/// * `block_len` - Actual length of data in block (0-64)
/// * `flags` - Mode flags (KEYED_HASH, DERIVE_KEY_MATERIAL, etc.)
///
/// # Returns
/// The root hash as 8 u32 words (little-endian).
#[inline]
#[must_use]
fn compress_chunk_tail_to_root_words(
  kernel: Kernel,
  cv: [u32; 8],
  block: &[u8; BLOCK_LEN],
  block_len: usize,
  flags: u32,
  add_chunk_start: bool,
) -> [u32; 8] {
  let mut final_flags = flags | CHUNK_END | ROOT;
  if add_chunk_start {
    final_flags |= CHUNK_START;
  }

  #[cfg(target_arch = "x86_64")]
  {
    match kernel.id {
      kernels::Blake3KernelId::X86Ssse3
      | kernels::Blake3KernelId::X86Sse41
      | kernels::Blake3KernelId::X86Avx2
      | kernels::Blake3KernelId::X86Avx512 => {
        // SAFETY: dispatch validates required CPU features before selecting
        // each x86 kernel; `block` is a readable 64-byte buffer.
        return unsafe { (kernel.x86_compress_cv_bytes)(&cv, block.as_ptr(), 0, block_len as u32, final_flags) };
      }
      _ => {}
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    if kernel.id == kernels::Blake3KernelId::Aarch64Neon {
      // SAFETY: NEON availability validated by dispatch
      return unsafe { aarch64::compress_cv_neon_bytes(&cv, block.as_ptr(), 0, block_len as u32, final_flags) };
    }
  }

  // Portable fallback
  let block_words = words16_from_le_bytes_64(block);
  first_8_words((kernel.compress)(&cv, &block_words, 0, block_len as u32, final_flags))
}

/// Hash tiny input (≤64B) directly to the root hash bytes.
///
/// This is the primary entry point for tiny-input keyed/derive hashing.
/// It handles input padding and dispatches to the appropriate kernel.
#[inline]
#[must_use]
fn hash_tiny_to_root_words(kernel: Kernel, key_words: [u32; 8], flags: u32, input: &[u8]) -> [u32; 8] {
  debug_assert!(input.len() <= BLOCK_LEN);

  let mut block = [0u8; BLOCK_LEN];
  block[..input.len()].copy_from_slice(input);

  compress_chunk_tail_to_root_words(kernel, key_words, &block, input.len(), flags, true)
}

/// Hash one chunk-or-less input directly to root words.
///
/// This shared path is used by architectures without dedicated one-chunk
/// assembly/intrinsics helpers.
#[inline]
#[must_use]
fn digest_one_chunk_root_hash_words_generic(kernel: Kernel, key_words: [u32; 8], flags: u32, input: &[u8]) -> [u32; 8] {
  debug_assert!(input.len() <= CHUNK_LEN);

  let (full_blocks, last_len) = if input.is_empty() {
    (0usize, 0usize)
  } else {
    let rem = input.len() % BLOCK_LEN;
    if rem == 0 {
      (input.len() / BLOCK_LEN - 1, BLOCK_LEN)
    } else {
      (input.len() / BLOCK_LEN, rem)
    }
  };

  let mut cv = key_words;
  let mut blocks_compressed = 0u8;
  let full_bytes = full_blocks * BLOCK_LEN;
  kernels::chunk_compress_blocks_inline(
    kernel.id,
    &mut cv,
    0,
    flags,
    &mut blocks_compressed,
    &input[..full_bytes],
  );

  let start = if full_blocks == 0 { CHUNK_START } else { 0 };
  let final_flags = flags | start | CHUNK_END | ROOT;

  if last_len == BLOCK_LEN && !input.is_empty() {
    let offset = full_blocks * BLOCK_LEN;
    // SAFETY: `offset + BLOCK_LEN <= input.len()` by construction.
    let block_words = unsafe { words16_from_le_bytes_64(&*input.as_ptr().add(offset).cast::<[u8; BLOCK_LEN]>()) };
    return first_8_words((kernel.compress)(&cv, &block_words, 0, BLOCK_LEN as u32, final_flags));
  }

  let mut final_block = [0u8; BLOCK_LEN];
  if last_len != 0 {
    let offset = full_blocks * BLOCK_LEN;
    // SAFETY: `last_len < BLOCK_LEN` here, and source range is in-bounds.
    unsafe { ptr::copy_nonoverlapping(input.as_ptr().add(offset), final_block.as_mut_ptr(), last_len) };
  }

  let final_words = words16_from_le_bytes_64(&final_block);
  first_8_words((kernel.compress)(&cv, &final_words, 0, last_len as u32, final_flags))
}

// ─────────────────────────────────────────────────────────────────────────────
// x86_64 tiny one-shot helpers (keyed/derive sensitive)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
#[must_use]
fn use_avx2_hash_many_one_chunk_fast_path() -> bool {
  dispatch::avx2_hash_many_one_chunk_fast_path()
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn digest_one_chunk_root_hash_words_x86(
  kernel: Kernel,
  key_words: [u32; 8],
  flags: u32,
  input: &[u8],
) -> [u32; 8] {
  debug_assert!(input.len() <= CHUNK_LEN);

  // AVX2 exact-block one-chunk fast path. Keep this on a narrow allowlist
  // only where CI has shown the path helps short inputs.
  #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
  if kernel.id == kernels::Blake3KernelId::X86Avx2
    && use_avx2_hash_many_one_chunk_fast_path()
    && !input.is_empty()
    && input.len().is_multiple_of(BLOCK_LEN)
  {
    let blocks = input.len() / BLOCK_LEN;
    debug_assert!((1..=CHUNK_LEN / BLOCK_LEN).contains(&blocks));
    let flags_start = flags | CHUNK_START;
    let flags_end = flags | CHUNK_END | ROOT;
    debug_assert!(flags <= u8::MAX as u32);
    debug_assert!(flags_start <= u8::MAX as u32);
    debug_assert!(flags_end <= u8::MAX as u32);
    let input_ptrs = [input.as_ptr()];
    let mut out = [0u8; OUT_LEN];
    // SAFETY: AVX2 dispatch selected this kernel; input is one contiguous
    // full-chunk-or-less buffer; output points to one OUT_LEN digest lane.
    unsafe {
      x86_64::asm::hash_many_avx2(
        input_ptrs.as_ptr(),
        1,
        blocks,
        key_words.as_ptr(),
        0,
        false,
        flags as u8,
        flags_start as u8,
        flags_end as u8,
        out.as_mut_ptr(),
      );
    }
    return words8_from_le_bytes_32(&out);
  }

  #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
  if kernel.id == kernels::Blake3KernelId::X86Avx512 && !input.is_empty() && input.len().is_multiple_of(BLOCK_LEN) {
    let blocks = input.len() / BLOCK_LEN;
    debug_assert!((1..=CHUNK_LEN / BLOCK_LEN).contains(&blocks));
    let flags_start = flags | CHUNK_START;
    let flags_end = flags | CHUNK_END | ROOT;
    debug_assert!(flags <= u8::MAX as u32);
    debug_assert!(flags_start <= u8::MAX as u32);
    debug_assert!(flags_end <= u8::MAX as u32);
    let input_ptrs = [input.as_ptr()];
    let mut out = [0u8; OUT_LEN];
    // SAFETY: AVX-512 dispatch selected this kernel; input is one contiguous
    // full-chunk-or-less buffer; output points to one OUT_LEN digest lane.
    unsafe {
      x86_64::asm::hash_many_avx512(
        input_ptrs.as_ptr(),
        1,
        blocks,
        key_words.as_ptr(),
        0,
        false,
        flags as u8,
        flags_start as u8,
        flags_end as u8,
        out.as_mut_ptr(),
      );
    }
    return words8_from_le_bytes_32(&out);
  }

  // Keep one final block for CHUNK_END|ROOT. For aligned inputs we process
  // `len/BLOCK_LEN - 1` full blocks and finish on the last full block.
  let (full_blocks, last_len) = if input.is_empty() {
    (0usize, 0usize)
  } else {
    let rem = input.len() % BLOCK_LEN;
    if rem == 0 {
      (input.len() / BLOCK_LEN - 1, BLOCK_LEN)
    } else {
      (input.len() / BLOCK_LEN, rem)
    }
  };

  // Hash all full blocks except the final block, updating the CV. This keeps
  // ROOT out of the dependency chain until the last compress.
  let mut cv = key_words;
  let full_bytes = full_blocks * BLOCK_LEN;
  if full_blocks != 0 {
    let first_block_ptr = input.as_ptr();
    let first_flags = flags | CHUNK_START;
    // SAFETY: `input` covers at least one full 64-byte block here, and this
    // helper is only entered for the selected x86 SIMD kernels.
    cv = unsafe {
      match kernel.id {
        kernels::Blake3KernelId::X86Sse41 => {
          x86_64::compress_cv_sse41_bytes(&cv, first_block_ptr, 0, BLOCK_LEN as u32, first_flags)
        }
        kernels::Blake3KernelId::X86Avx2 => {
          x86_64::compress_cv_avx2_bytes(&cv, first_block_ptr, 0, BLOCK_LEN as u32, first_flags)
        }
        kernels::Blake3KernelId::X86Avx512 => {
          #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
          {
            x86_64::asm::compress_in_place_avx512(&cv, first_block_ptr, 0, BLOCK_LEN as u32, first_flags)
          }
          #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
          {
            x86_64::compress_cv_avx512_bytes(&cv, first_block_ptr, 0, BLOCK_LEN as u32, first_flags)
          }
        }
        _ => unreachable!(),
      }
    };

    if full_blocks > 1 {
      let mut blocks_compressed = 1u8;
      (kernel.chunk_compress_blocks)(&mut cv, 0, flags, &mut blocks_compressed, &input[BLOCK_LEN..full_bytes]);
    }
  }

  let start = if full_blocks == 0 { CHUNK_START } else { 0 };
  let final_flags = flags | start | CHUNK_END | ROOT;

  if last_len == BLOCK_LEN && !input.is_empty() {
    // SAFETY: `full_blocks * BLOCK_LEN + BLOCK_LEN <= input.len()`.
    let block_ptr = unsafe { input.as_ptr().add(full_blocks * BLOCK_LEN) };
    // SAFETY: x86 dispatch selected this function only for x86 SIMD kernels.
    return unsafe {
      match kernel.id {
        kernels::Blake3KernelId::X86Sse41 => {
          x86_64::compress_cv_sse41_bytes(&cv, block_ptr, 0, BLOCK_LEN as u32, final_flags)
        }
        kernels::Blake3KernelId::X86Avx2 => {
          x86_64::compress_cv_avx2_bytes(&cv, block_ptr, 0, BLOCK_LEN as u32, final_flags)
        }
        kernels::Blake3KernelId::X86Avx512 => {
          #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
          {
            x86_64::asm::compress_in_place_avx512(&cv, block_ptr, 0, BLOCK_LEN as u32, final_flags)
          }
          #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
          {
            x86_64::compress_cv_avx512_bytes(&cv, block_ptr, 0, BLOCK_LEN as u32, final_flags)
          }
        }
        _ => unreachable!(),
      }
    };
  }

  // Partial final block (including empty): pad to 64 bytes.
  let mut padded = [0u8; BLOCK_LEN];
  if last_len != 0 {
    let offset = full_blocks * BLOCK_LEN;
    // SAFETY: `padded` is 64 bytes, and `last_len < 64` here.
    unsafe { ptr::copy_nonoverlapping(input.as_ptr().add(offset), padded.as_mut_ptr(), last_len) };
  }
  let block_ptr = padded.as_ptr();

  // SAFETY: x86 dispatch selected this function only for x86 SIMD kernels.
  unsafe {
    match kernel.id {
      kernels::Blake3KernelId::X86Sse41 => {
        x86_64::compress_cv_sse41_bytes(&cv, block_ptr, 0, last_len as u32, final_flags)
      }
      kernels::Blake3KernelId::X86Avx2 => {
        x86_64::compress_cv_avx2_bytes(&cv, block_ptr, 0, last_len as u32, final_flags)
      }
      kernels::Blake3KernelId::X86Avx512 => {
        #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
        {
          x86_64::asm::compress_in_place_avx512(&cv, block_ptr, 0, last_len as u32, final_flags)
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
          x86_64::compress_cv_avx512_bytes(&cv, block_ptr, 0, last_len as u32, final_flags)
        }
      }
      _ => unreachable!(),
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// aarch64 tiny one-shot helpers (keyed/derive sensitive)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn digest_one_chunk_root_hash_words_aarch64(
  kernel: Kernel,
  key_words: [u32; 8],
  flags: u32,
  input: &[u8],
) -> [u32; 8] {
  debug_assert!(input.len() <= CHUNK_LEN);
  debug_assert_eq!(kernel.id, kernels::Blake3KernelId::Aarch64Neon);

  // Keep one final block for CHUNK_END|ROOT. For aligned inputs we process
  // `len/BLOCK_LEN - 1` full blocks and finish on the last full block.
  let (full_blocks, last_len) = if input.is_empty() {
    (0usize, 0usize)
  } else {
    let rem = input.len() % BLOCK_LEN;
    if rem == 0 {
      (input.len() / BLOCK_LEN - 1, BLOCK_LEN)
    } else {
      (input.len() / BLOCK_LEN, rem)
    }
  };

  // Hash all full blocks except the final block, updating the CV. This keeps
  // ROOT out of the dependency chain until the last compress.
  let mut cv = key_words;
  let full_bytes = full_blocks * BLOCK_LEN;
  if full_blocks != 0 {
    // SAFETY: `input` covers at least one full 64-byte block here, and this
    // helper is only selected when NEON support has already been validated.
    cv = unsafe { aarch64::compress_cv_neon_bytes(&cv, input.as_ptr(), 0, BLOCK_LEN as u32, flags | CHUNK_START) };
    if full_blocks > 1 {
      let mut blocks_compressed: u8 = 1;
      kernels::chunk_compress_blocks_inline(
        kernel.id,
        &mut cv,
        0,
        flags,
        &mut blocks_compressed,
        &input[BLOCK_LEN..full_bytes],
      );
    }
  }

  let start = if full_blocks == 0 { CHUNK_START } else { 0 };
  let final_flags = flags | start | CHUNK_END | ROOT;

  // For partial blocks (including empty), pad to 64 bytes.
  let mut padded = [0u8; BLOCK_LEN];
  let block_ptr = if last_len == BLOCK_LEN && !input.is_empty() {
    // SAFETY: `full_blocks * BLOCK_LEN + BLOCK_LEN <= input.len()`.
    unsafe { input.as_ptr().add(full_blocks * BLOCK_LEN) }
  } else {
    if last_len != 0 {
      let offset = full_blocks * BLOCK_LEN;
      // SAFETY: `padded` is 64 bytes, and `last_len < 64` here.
      unsafe { ptr::copy_nonoverlapping(input.as_ptr().add(offset), padded.as_mut_ptr(), last_len) };
    }
    padded.as_ptr()
  };

  // SAFETY: `block_ptr` points to 64 bytes (either into `input` or `padded`),
  // and dispatch only selects NEON when the required CPU features are present.
  unsafe { aarch64::compress_cv_neon_bytes(&cv, block_ptr, 0, last_len as u32, final_flags) }
}

#[cfg(feature = "std")]
pub(crate) mod kernel_test;

#[cfg(test)]
mod tests {
  use super::{Blake3, OUT_LEN};
  use crate::traits::{Digest, Xof};

  #[test]
  fn xof_sized_matches_standard() {
    use alloc::vec;

    // Test that finalize_xof_sized produces identical results to finalize_xof
    let test_cases = [
      (1usize, 1024usize),
      (64, 1024),
      (64, 512),
      (0, 512),
      (1024, 256),
      (4096, 1024),
    ];

    for (input_len, output_len) in test_cases {
      let input = vec![0xabu8; input_len];
      let mut out_standard = vec![0u8; output_len];
      let mut out_sized = vec![0u8; output_len];

      // Standard finalize_xof
      let mut h1 = Blake3::new();
      h1.update(&input);
      let mut xof1 = h1.finalize_xof();
      xof1.squeeze(&mut out_standard);

      // New finalize_xof_sized
      let mut h2 = Blake3::new();
      h2.update(&input);
      let mut xof2 = h2.finalize_xof_sized(output_len);
      xof2.squeeze(&mut out_sized);

      assert_eq!(
        out_standard, out_sized,
        "XOF outputs should match for {}B input -> {}B output",
        input_len, output_len
      );
    }
  }

  #[test]
  fn xof_repeated_small_squeezes_match_single_read() {
    use alloc::vec;

    for input_len in [0usize, 1, 64, 1024, 4096] {
      let input = vec![0x5au8; input_len];

      let mut single = Blake3::new();
      single.update(&input);
      let mut single_xof = single.finalize_xof();
      let mut expected = [0u8; 96];
      single_xof.squeeze(&mut expected);

      let mut stepped = Blake3::new();
      stepped.update(&input);
      let mut stepped_xof = stepped.finalize_xof();
      let mut actual = [0u8; 96];
      let mut off = 0usize;
      for step in [7usize, 9, 16, 32, 5, 27] {
        let end = off + step;
        stepped_xof.squeeze(&mut actual[off..end]);
        off = end;
      }

      assert_eq!(actual, expected, "xof squeeze mismatch for input_len={input_len}");
    }
  }

  const KEY: &[u8; 32] = b"whats the Elvish word for friend";
  const CONTEXT: &str = "BLAKE3 2019-12-27 16:29:52 test vectors context";

  fn hex_to_bytes(hex: &str, out: &mut [u8]) {
    assert_eq!(hex.len(), out.len() * 2);
    for (i, chunk) in hex.as_bytes().chunks_exact(2).enumerate() {
      let hi = (chunk[0] as char).to_digit(16).unwrap();
      let lo = (chunk[1] as char).to_digit(16).unwrap();
      out[i] = ((hi << 4) | lo) as u8;
    }
  }

  fn input_pattern(len: usize) -> alloc::vec::Vec<u8> {
    let mut v = alloc::vec::Vec::with_capacity(len);
    for i in 0..len {
      v.push((i % 251) as u8);
    }
    v
  }

  extern crate alloc;

  #[test]
  fn official_vectors_len0_hash_and_xof_prefix() {
    let mut hasher = Blake3::new();
    hasher.update(&input_pattern(0));

    let expected_hash_hex = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262";
    let mut expected_hash = [0u8; OUT_LEN];
    hex_to_bytes(expected_hash_hex, &mut expected_hash);
    assert_eq!(hasher.finalize(), expected_hash);

    let expected_xof_prefix_hex = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262e00f03e7b69af26b7faaf09fcd333050338ddfe085b8cc869ca98b206c08243a26f5487789e8f660afe6c99ef9e0c52b92e7393024a80459cf91f476f9ffdbda7001c22e159b402631f277ca96f2defdf1078282314e763699a31c5363165421cce14d";
    let mut expected_xof_prefix = [0u8; 131];
    hex_to_bytes(expected_xof_prefix_hex, &mut expected_xof_prefix);

    let mut xof = hasher.finalize_xof();
    let mut out = [0u8; 131];
    xof.squeeze(&mut out);
    assert_eq!(out, expected_xof_prefix);
  }

  #[test]
  fn official_vectors_len0_keyed_and_derive() {
    let mut keyed = Blake3::new_keyed(KEY);
    keyed.update(&input_pattern(0));
    let expected_keyed_hex = "92b2b75604ed3c761f9d6f62392c8a9227ad0ea3f09573e783f1498a4ed60d26";
    let mut expected_keyed = [0u8; OUT_LEN];
    hex_to_bytes(expected_keyed_hex, &mut expected_keyed);
    assert_eq!(keyed.finalize(), expected_keyed);

    let mut dk = Blake3::new_derive_key(CONTEXT);
    dk.update(&input_pattern(0));
    let expected_dk_hex = "2cc39783c223154fea8dfb7c1b1660f2ac2dcbd1c1de8277b0b0dd39b7e50d7d";
    let mut expected_dk = [0u8; OUT_LEN];
    hex_to_bytes(expected_dk_hex, &mut expected_dk);
    assert_eq!(dk.finalize(), expected_dk);
  }

  #[test]
  fn keyed_and_derive_streaming_tiny_matches_oneshot() {
    const KEY: &[u8; 32] = b"whats the Elvish word for friend";
    const CONTEXT: &str = "BLAKE3 2019-12-27 16:29:52 test vectors context";

    for len in [0usize, 1, 3, 8, 16, 31, 32, 63, 64, 65, 127, 255, 1024] {
      let input = input_pattern(len);

      // Keyed: streaming vs one-shot.
      let expected_keyed = Blake3::keyed_digest(KEY, &input);
      for split in [1usize, 7, 13, 64, 255, 1024] {
        let mut h = Blake3::new_keyed(KEY);
        for chunk in input.chunks(split) {
          h.update(chunk);
        }
        assert_eq!(
          h.finalize(),
          expected_keyed,
          "keyed streaming mismatch len={len} split={split}"
        );
      }

      // Derive: streaming vs one-shot.
      let expected_derive = Blake3::derive_key(CONTEXT, &input);
      for split in [1usize, 7, 13, 64, 255, 1024] {
        let mut h = Blake3::new_derive_key(CONTEXT);
        for chunk in input.chunks(split) {
          h.update(chunk);
        }
        assert_eq!(
          h.finalize(),
          expected_derive,
          "derive streaming mismatch len={len} split={split}"
        );
      }
    }
  }

  #[test]
  fn short_xof_read32_matches_official_for_all_modes() {
    for len in [1usize, 64, 1024] {
      let input = input_pattern(len);

      let mut plain = Blake3::new();
      plain.update(&input);
      let mut plain_xof = plain.finalize_xof();
      let mut plain_out = [0u8; 32];
      plain_xof.squeeze(&mut plain_out);
      let mut plain_expected = [0u8; 32];
      let mut plain_ref = blake3::Hasher::new();
      plain_ref.update(&input);
      plain_ref.finalize_xof().fill(&mut plain_expected);
      assert_eq!(plain_out, plain_expected, "plain short xof mismatch len={len}");

      let mut keyed = Blake3::new_keyed(KEY);
      keyed.update(&input);
      let mut keyed_xof = keyed.finalize_xof();
      let mut keyed_out = [0u8; 32];
      keyed_xof.squeeze(&mut keyed_out);
      let mut keyed_expected = [0u8; 32];
      let mut keyed_ref = blake3::Hasher::new_keyed(KEY);
      keyed_ref.update(&input);
      keyed_ref.finalize_xof().fill(&mut keyed_expected);
      assert_eq!(keyed_out, keyed_expected, "keyed short xof mismatch len={len}");

      let mut derive = Blake3::new_derive_key(CONTEXT);
      derive.update(&input);
      let mut derive_xof = derive.finalize_xof();
      let mut derive_out = [0u8; 32];
      derive_xof.squeeze(&mut derive_out);
      let mut derive_expected = [0u8; 32];
      let mut derive_ref = blake3::Hasher::new_derive_key(CONTEXT);
      derive_ref.update(&input);
      derive_ref.finalize_xof().fill(&mut derive_expected);
      assert_eq!(derive_out, derive_expected, "derive short xof mismatch len={len}");
    }
  }
}
