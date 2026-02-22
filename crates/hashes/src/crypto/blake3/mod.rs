//! BLAKE3 (hash + XOF).
//!
//! This is a portable, dependency-free implementation suitable for `no_std`.

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![allow(clippy::indexing_slicing)] // Audited fixed-size parsing + perf-critical inner loops.

use core::{cmp::min, mem::MaybeUninit, ptr, slice};
#[cfg(feature = "std")]
extern crate alloc;
#[cfg(feature = "std")]
use alloc::string::{String, ToString};
#[cfg(feature = "std")]
use core::cell::RefCell;
#[cfg(all(feature = "std", test))]
use core::{
  panic::AssertUnwindSafe,
  sync::atomic::{AtomicBool, Ordering},
};
#[cfg(all(feature = "std", test))]
use std::panic;
#[cfg(feature = "std")]
use std::sync::{Mutex, OnceLock};
#[cfg(feature = "std")]
use std::thread;
#[cfg(feature = "std")]
use std::thread_local;

use traits::{Digest, Xof};

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
// 54-level binary reduction tree.
const CV_STACK_LEN: usize = 54;

type CvBytes = [u8; OUT_LEN];

#[cfg(feature = "std")]
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

  pub(super) enum RayonJoin {}

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
#[cfg(feature = "std")]
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
  Update,
  KeyedUpdate,
  DeriveUpdate,
}

#[cfg(feature = "std")]
#[inline]
fn parallel_override() -> Option<ParallelPolicyOverride> {
  PARALLEL_OVERRIDE
    .get_or_init(|| Mutex::new(None))
    .lock()
    .ok()
    .and_then(|g| *g)
}

#[cfg(feature = "std")]
#[inline]
fn available_parallelism_cached() -> Option<usize> {
  *AVAILABLE_PARALLELISM.get_or_init(|| thread::available_parallelism().ok().map(|v| v.get()))
}

#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
#[inline]
fn scale_parallel_required_bytes(mode: ParallelPolicyKind, required: usize, input_bytes: usize) -> usize {
  let (num, den) = match mode {
    // Fitted from crossover data: one-shot/XOF benefit earlier than update.
    ParallelPolicyKind::Oneshot | ParallelPolicyKind::KeyedOneshot | ParallelPolicyKind::DeriveOneshot => {
      if input_bytes >= (1024 * 1024) {
        (10usize, 10usize)
      } else {
        (12usize, 10usize)
      }
    }
    ParallelPolicyKind::Update | ParallelPolicyKind::KeyedUpdate | ParallelPolicyKind::DeriveUpdate => {
      (15usize, 10usize)
    }
    ParallelPolicyKind::Xof | ParallelPolicyKind::KeyedXof | ParallelPolicyKind::DeriveXof => (12usize, 10usize),
  };
  required.saturating_mul(num).saturating_add(den - 1) / den
}

#[cfg(feature = "std")]
thread_local! {
  static BLAKE3_SUBTREE_SCRATCH0: RefCell<alloc::vec::Vec<[u32; 8]>> = const { RefCell::new(alloc::vec::Vec::new()) };
  static BLAKE3_SUBTREE_SCRATCH1: RefCell<alloc::vec::Vec<[u32; 8]>> = const { RefCell::new(alloc::vec::Vec::new()) };
  static DERIVE_KEY_CONTEXT_LOCAL_CACHE: RefCell<Option<(String, [u32; 8])>> = const { RefCell::new(None) };
  static KEYED_WORDS_LOCAL_CACHE: RefCell<Option<([u8; KEY_LEN], [u32; 8])>> = const { RefCell::new(None) };
}

#[cfg(feature = "std")]
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);

// SAFETY: This is only used to pass disjoint slice partitions into Rayon tasks.
#[cfg(feature = "std")]
unsafe impl<T> Send for SendPtr<T> {}
#[cfg(feature = "std")]
// SAFETY: This is only used to pass disjoint slice partitions into Rayon tasks.
unsafe impl<T> Sync for SendPtr<T> {}

#[cfg(feature = "std")]
impl<T> SendPtr<T> {
  #[inline]
  #[must_use]
  fn get(self) -> *mut T {
    self.0
  }
}

#[cfg(feature = "std")]
#[inline]
#[must_use]
fn thread_range(thread_index: usize, threads_total: usize, total: usize) -> (usize, usize) {
  debug_assert!(thread_index < threads_total);
  let start = (thread_index * total) / threads_total;
  let end = ((thread_index + 1) * total) / threads_total;
  (start, end)
}

#[cfg(all(feature = "std", test))]
static FORCE_PARALLEL_PANIC: AtomicBool = AtomicBool::new(false);

#[cfg(all(feature = "std", test))]
struct ForceParallelPanicGuard {
  prev: bool,
}

#[cfg(all(feature = "std", test))]
impl Drop for ForceParallelPanicGuard {
  fn drop(&mut self) {
    FORCE_PARALLEL_PANIC.store(self.prev, Ordering::Relaxed);
  }
}

#[cfg(all(feature = "std", test))]
#[inline]
#[must_use]
fn force_parallel_panic(enabled: bool) -> ForceParallelPanicGuard {
  let prev = FORCE_PARALLEL_PANIC.swap(enabled, Ordering::Relaxed);
  ForceParallelPanicGuard { prev }
}

#[cfg(all(feature = "std", test))]
#[inline]
fn maybe_force_parallel_panic() {
  if FORCE_PARALLEL_PANIC.load(Ordering::Relaxed) {
    panic!("forced parallel panic (test-only)");
  }
}

#[cfg(feature = "std")]
#[inline]
fn run_parallel_task(task: impl FnOnce()) -> bool {
  #[cfg(test)]
  {
    panic::catch_unwind(AssertUnwindSafe(|| {
      maybe_force_parallel_panic();
      task();
    }))
    .is_ok()
  }
  #[cfg(not(test))]
  {
    task();
    true
  }
}

#[cfg(feature = "std")]
#[inline]
fn with_subtree_scratch<R>(subtree_chunks: usize, f: impl FnOnce(&mut [[u32; 8]], &mut [[u32; 8]]) -> R) -> R {
  BLAKE3_SUBTREE_SCRATCH0.with(|s0| {
    BLAKE3_SUBTREE_SCRATCH1.with(|s1| {
      let mut scratch0 = s0.borrow_mut();
      let mut scratch1 = s1.borrow_mut();
      if scratch0.len() != subtree_chunks {
        scratch0.resize(subtree_chunks, [0u32; 8]);
      }
      if scratch1.len() != subtree_chunks {
        scratch1.resize(subtree_chunks, [0u32; 8]);
      }
      f(scratch0.as_mut_slice(), scratch1.as_mut_slice())
    })
  })
}

#[inline]
fn reduce_power_of_two_cvs_in_place(
  kernel: Kernel,
  key_words: [u32; 8],
  flags: u32,
  scratch0: &mut [[u32; 8]],
  scratch1: &mut [[u32; 8]],
) -> [u32; 8] {
  debug_assert_eq!(scratch0.len(), scratch1.len());
  debug_assert!(scratch0.len().is_power_of_two());

  if scratch0.len() == 1 {
    return scratch0[0];
  }

  let mut cur_len = scratch0.len();
  let mut cur_is_0 = true;

  while cur_len > 1 {
    let pairs = cur_len / 2;
    debug_assert!(pairs != 0);
    if cur_is_0 {
      kernels::parent_cvs_many_from_cvs_inline(
        kernel.id,
        &scratch0[..2 * pairs],
        key_words,
        flags,
        &mut scratch1[..pairs],
      );
      cur_is_0 = false;
    } else {
      kernels::parent_cvs_many_from_cvs_inline(
        kernel.id,
        &scratch1[..2 * pairs],
        key_words,
        flags,
        &mut scratch0[..pairs],
      );
      cur_is_0 = true;
    }
    cur_len = pairs;
  }

  if cur_is_0 { scratch0[0] } else { scratch1[0] }
}

#[cfg(feature = "std")]
#[inline]
fn hash_full_chunks_cvs_serial(
  kernel: Kernel,
  key_words: [u32; 8],
  flags: u32,
  base_counter: u64,
  input: &[u8],
  out: &mut [[u32; 8]],
) {
  debug_assert_eq!(input.len(), out.len() * CHUNK_LEN);

  const MAX_SIMD_DEGREE: usize = 16;

  let mut written = 0usize;
  let mut chunk_counter = base_counter;
  let mut input_ptr = input.as_ptr();

  #[cfg(target_endian = "little")]
  {
    while written < out.len() {
      let remaining = out.len() - written;
      let batch = remaining.min(MAX_SIMD_DEGREE);

      // SAFETY:
      // - `input_ptr` points into `input`, which is at least `batch * CHUNK_LEN` bytes.
      // - `out[written..]` has at least `batch` CV slots (= `batch * OUT_LEN` bytes).
      let out_ptr = unsafe { out.as_mut_ptr().add(written).cast::<u8>() };
      // SAFETY: pointers and lengths are validated in the comments above.
      unsafe {
        (kernel.hash_many_contiguous)(input_ptr, batch, &key_words, chunk_counter, flags, out_ptr);
      }

      written += batch;
      chunk_counter = chunk_counter.wrapping_add(batch as u64);
      // SAFETY: advancing within `input` by whole chunks.
      unsafe {
        input_ptr = input_ptr.add(batch * CHUNK_LEN);
      }
    }
  }

  #[cfg(not(target_endian = "little"))]
  let mut out_buf = [0u8; OUT_LEN * MAX_SIMD_DEGREE];

  #[cfg(not(target_endian = "little"))]
  while written < out.len() {
    let remaining = out.len() - written;
    let batch = remaining.min(MAX_SIMD_DEGREE);

    // SAFETY:
    // - `input_ptr` points into `input`, which is at least `batch * CHUNK_LEN` bytes.
    // - `out_buf` is `OUT_LEN * batch` bytes.
    unsafe {
      (kernel.hash_many_contiguous)(input_ptr, batch, &key_words, chunk_counter, flags, out_buf.as_mut_ptr());
    }

    for i in 0..batch {
      let offset = i * OUT_LEN;
      // SAFETY: `out_buf` is `OUT_LEN * MAX_SIMD_DEGREE` bytes, and `i < batch <= MAX_SIMD_DEGREE`.
      let cv = unsafe { words8_from_le_bytes_32(&*(out_buf.as_ptr().add(offset) as *const [u8; OUT_LEN])) };
      out[written + i] = cv;
    }

    written += batch;
    chunk_counter = chunk_counter.wrapping_add(batch as u64);
    // SAFETY: advancing within `input` by whole chunks.
    unsafe {
      input_ptr = input_ptr.add(batch * CHUNK_LEN);
    }
  }
}

#[cfg(feature = "std")]
fn hash_full_chunks_cvs_parallel_rayon(
  kernel: Kernel,
  key_words: [u32; 8],
  flags: u32,
  base_counter: u64,
  input: &[u8],
  out: &mut [[u32; 8]],
  threads_total: usize,
) {
  debug_assert_eq!(input.len(), out.len() * CHUNK_LEN);

  if threads_total <= 1 || out.len() < 2 {
    hash_full_chunks_cvs_serial(kernel, key_words, flags, base_counter, input, out);
    return;
  }

  let threads_total = threads_total.min(out.len()).max(1);
  #[cfg(test)]
  let failed = AtomicBool::new(false);
  let out_ptr = SendPtr(out.as_mut_ptr());
  let out_len = out.len();

  rayon::scope(|s| {
    #[cfg(test)]
    let failed = &failed;
    for t in 1..threads_total {
      let (start, end) = thread_range(t, threads_total, out_len);
      if start == end {
        continue;
      }
      let input = &input[start * CHUNK_LEN..end * CHUNK_LEN];
      let counter = base_counter.wrapping_add(start as u64);
      s.spawn(move |_| {
        #[cfg(test)]
        let ok = run_parallel_task(|| {
          // SAFETY: `out_ptr` is valid for `out_len` elements and this task's
          // range is disjoint from every other task.
          let out = unsafe { slice::from_raw_parts_mut(out_ptr.get().add(start), end - start) };
          hash_full_chunks_cvs_serial(kernel, key_words, flags, counter, input, out);
        });
        #[cfg(test)]
        if !ok {
          failed.store(true, Ordering::Relaxed);
        }
        #[cfg(not(test))]
        {
          run_parallel_task(|| {
            // SAFETY: `out_ptr` is valid for `out_len` elements and this task's
            // range is disjoint from every other task.
            let out = unsafe { slice::from_raw_parts_mut(out_ptr.get().add(start), end - start) };
            hash_full_chunks_cvs_serial(kernel, key_words, flags, counter, input, out);
          });
        }
      });
    }

    let (start, end) = thread_range(0, threads_total, out_len);
    #[cfg(test)]
    let ok = run_parallel_task(|| {
      hash_full_chunks_cvs_serial(
        kernel,
        key_words,
        flags,
        base_counter.wrapping_add(start as u64),
        &input[start * CHUNK_LEN..end * CHUNK_LEN],
        // SAFETY: disjoint partition for thread 0.
        unsafe { slice::from_raw_parts_mut(out_ptr.get().add(start), end - start) },
      );
    });
    #[cfg(test)]
    if !ok {
      failed.store(true, Ordering::Relaxed);
    }
    #[cfg(not(test))]
    {
      run_parallel_task(|| {
        hash_full_chunks_cvs_serial(
          kernel,
          key_words,
          flags,
          base_counter.wrapping_add(start as u64),
          &input[start * CHUNK_LEN..end * CHUNK_LEN],
          // SAFETY: disjoint partition for thread 0.
          unsafe { slice::from_raw_parts_mut(out_ptr.get().add(start), end - start) },
        );
      });
    }
  });

  #[cfg(test)]
  if failed.load(Ordering::Relaxed) {
    hash_full_chunks_cvs_serial(kernel, key_words, flags, base_counter, input, out);
  }
}

#[cfg(feature = "std")]
fn parent_cvs_many_from_cvs_parallel_rayon(
  kernel: Kernel,
  key_words: [u32; 8],
  flags: u32,
  children: &[[u32; 8]],
  out: &mut [[u32; 8]],
  threads_total: usize,
) {
  debug_assert_eq!(children.len(), out.len() * 2);

  if threads_total <= 1 || out.len() < 2 {
    kernels::parent_cvs_many_from_cvs_inline(kernel.id, children, key_words, flags, out);
    return;
  }

  let pairs = out.len();
  let threads_total = threads_total.min(pairs).max(1);
  #[cfg(test)]
  let failed = AtomicBool::new(false);
  let out_ptr = SendPtr(out.as_mut_ptr());

  rayon::scope(|s| {
    #[cfg(test)]
    let failed = &failed;
    for t in 1..threads_total {
      let (start, end) = thread_range(t, threads_total, pairs);
      if start == end {
        continue;
      }
      let children = &children[2 * start..2 * end];
      s.spawn(move |_| {
        #[cfg(test)]
        let ok = run_parallel_task(|| {
          // SAFETY: `out_ptr` is valid for `pairs` outputs and this task's
          // range is disjoint from every other task.
          let out = unsafe { slice::from_raw_parts_mut(out_ptr.get().add(start), end - start) };
          kernels::parent_cvs_many_from_cvs_inline(kernel.id, children, key_words, flags, out);
        });
        #[cfg(test)]
        if !ok {
          failed.store(true, Ordering::Relaxed);
        }
        #[cfg(not(test))]
        {
          run_parallel_task(|| {
            // SAFETY: `out_ptr` is valid for `pairs` outputs and this task's
            // range is disjoint from every other task.
            let out = unsafe { slice::from_raw_parts_mut(out_ptr.get().add(start), end - start) };
            kernels::parent_cvs_many_from_cvs_inline(kernel.id, children, key_words, flags, out);
          });
        }
      });
    }

    let (start, end) = thread_range(0, threads_total, pairs);
    #[cfg(test)]
    let ok = run_parallel_task(|| {
      kernels::parent_cvs_many_from_cvs_inline(
        kernel.id,
        &children[2 * start..2 * end],
        key_words,
        flags,
        // SAFETY: disjoint partition for thread 0.
        unsafe { slice::from_raw_parts_mut(out_ptr.get().add(start), end - start) },
      );
    });
    #[cfg(test)]
    if !ok {
      failed.store(true, Ordering::Relaxed);
    }
    #[cfg(not(test))]
    {
      run_parallel_task(|| {
        kernels::parent_cvs_many_from_cvs_inline(
          kernel.id,
          &children[2 * start..2 * end],
          key_words,
          flags,
          // SAFETY: disjoint partition for thread 0.
          unsafe { slice::from_raw_parts_mut(out_ptr.get().add(start), end - start) },
        );
      });
    }
  });

  #[cfg(test)]
  if failed.load(Ordering::Relaxed) {
    kernels::parent_cvs_many_from_cvs_inline(kernel.id, children, key_words, flags, out);
  }
}

#[cfg(feature = "std")]
#[inline]
fn hash_power_of_two_subtree_roots_serial(
  kernel: Kernel,
  key_words: [u32; 8],
  flags: u32,
  base_counter: u64,
  input: &[u8],
  subtree_chunks: usize,
  out: &mut [[u32; 8]],
) {
  debug_assert!(subtree_chunks.is_power_of_two());
  debug_assert_ne!(subtree_chunks, 0);
  debug_assert_eq!(input.len(), out.len() * subtree_chunks * CHUNK_LEN);

  with_subtree_scratch(subtree_chunks, |scratch0, scratch1| {
    for (i, slot) in out.iter_mut().enumerate() {
      let chunk_base = base_counter.wrapping_add((i * subtree_chunks) as u64);
      let bytes = &input[i * subtree_chunks * CHUNK_LEN..(i + 1) * subtree_chunks * CHUNK_LEN];
      hash_full_chunks_cvs_serial(kernel, key_words, flags, chunk_base, bytes, scratch0);
      *slot = reduce_power_of_two_cvs_in_place(kernel, key_words, flags, scratch0, scratch1);
    }
  });
}

#[cfg(feature = "std")]
struct SubtreeRootsRequest<'a> {
  kernel: Kernel,
  key_words: [u32; 8],
  flags: u32,
  base_counter: u64,
  input: &'a [u8],
  subtree_chunks: usize,
  out: &'a mut [[u32; 8]],
  threads_total: usize,
}

#[cfg(feature = "std")]
fn hash_power_of_two_subtree_roots_parallel_rayon(req: SubtreeRootsRequest<'_>) {
  let SubtreeRootsRequest {
    kernel,
    key_words,
    flags,
    base_counter,
    input,
    subtree_chunks,
    out,
    threads_total,
  } = req;
  debug_assert!(subtree_chunks.is_power_of_two());
  debug_assert_ne!(subtree_chunks, 0);
  debug_assert_eq!(input.len(), out.len() * subtree_chunks * CHUNK_LEN);

  if threads_total <= 1 || out.len() < 2 {
    hash_power_of_two_subtree_roots_serial(kernel, key_words, flags, base_counter, input, subtree_chunks, out);
    return;
  }

  let threads_total = threads_total.min(out.len()).max(1);
  #[cfg(test)]
  let failed = AtomicBool::new(false);
  let out_ptr = SendPtr(out.as_mut_ptr());
  let out_len = out.len();

  rayon::scope(|s| {
    #[cfg(test)]
    let failed = &failed;
    for t in 1..threads_total {
      let (start, end) = thread_range(t, threads_total, out_len);
      if start == end {
        continue;
      }
      let input = &input[start * subtree_chunks * CHUNK_LEN..end * subtree_chunks * CHUNK_LEN];
      let counter = base_counter.wrapping_add((start * subtree_chunks) as u64);

      s.spawn(move |_| {
        #[cfg(test)]
        let ok = run_parallel_task(|| {
          with_subtree_scratch(subtree_chunks, |scratch0, scratch1| {
            // SAFETY: `out_ptr` is valid for `out_len` outputs and this task's
            // range is disjoint from every other task.
            let out = unsafe { slice::from_raw_parts_mut(out_ptr.get().add(start), end - start) };
            for (i, slot) in out.iter_mut().enumerate() {
              let chunk_base = counter + (i * subtree_chunks) as u64;
              let bytes = &input[i * subtree_chunks * CHUNK_LEN..(i + 1) * subtree_chunks * CHUNK_LEN];
              hash_full_chunks_cvs_serial(kernel, key_words, flags, chunk_base, bytes, scratch0);
              *slot = reduce_power_of_two_cvs_in_place(kernel, key_words, flags, scratch0, scratch1);
            }
          });
        });
        #[cfg(test)]
        if !ok {
          failed.store(true, Ordering::Relaxed);
        }
        #[cfg(not(test))]
        {
          run_parallel_task(|| {
            with_subtree_scratch(subtree_chunks, |scratch0, scratch1| {
              // SAFETY: `out_ptr` is valid for `out_len` outputs and this task's
              // range is disjoint from every other task.
              let out = unsafe { slice::from_raw_parts_mut(out_ptr.get().add(start), end - start) };
              for (i, slot) in out.iter_mut().enumerate() {
                let chunk_base = counter + (i * subtree_chunks) as u64;
                let bytes = &input[i * subtree_chunks * CHUNK_LEN..(i + 1) * subtree_chunks * CHUNK_LEN];
                hash_full_chunks_cvs_serial(kernel, key_words, flags, chunk_base, bytes, scratch0);
                *slot = reduce_power_of_two_cvs_in_place(kernel, key_words, flags, scratch0, scratch1);
              }
            });
          });
        }
      });
    }

    let (start, end) = thread_range(0, threads_total, out_len);
    #[cfg(test)]
    let ok = run_parallel_task(|| {
      with_subtree_scratch(subtree_chunks, |scratch0, scratch1| {
        let counter = base_counter.wrapping_add((start * subtree_chunks) as u64);
        // SAFETY: disjoint partition for thread 0.
        let out = unsafe { slice::from_raw_parts_mut(out_ptr.get().add(start), end - start) };
        for (i, slot) in out.iter_mut().enumerate() {
          let chunk_base = counter + (i * subtree_chunks) as u64;
          let bytes = &input[(start + i) * subtree_chunks * CHUNK_LEN..(start + i + 1) * subtree_chunks * CHUNK_LEN];
          hash_full_chunks_cvs_serial(kernel, key_words, flags, chunk_base, bytes, scratch0);
          *slot = reduce_power_of_two_cvs_in_place(kernel, key_words, flags, scratch0, scratch1);
        }
      });
    });
    #[cfg(test)]
    if !ok {
      failed.store(true, Ordering::Relaxed);
    }
    #[cfg(not(test))]
    {
      run_parallel_task(|| {
        with_subtree_scratch(subtree_chunks, |scratch0, scratch1| {
          let counter = base_counter.wrapping_add((start * subtree_chunks) as u64);
          // SAFETY: disjoint partition for thread 0.
          let out = unsafe { slice::from_raw_parts_mut(out_ptr.get().add(start), end - start) };
          for (i, slot) in out.iter_mut().enumerate() {
            let chunk_base = counter + (i * subtree_chunks) as u64;
            let bytes = &input[(start + i) * subtree_chunks * CHUNK_LEN..(start + i + 1) * subtree_chunks * CHUNK_LEN];
            hash_full_chunks_cvs_serial(kernel, key_words, flags, chunk_base, bytes, scratch0);
            *slot = reduce_power_of_two_cvs_in_place(kernel, key_words, flags, scratch0, scratch1);
          }
        });
      });
    }
  });

  #[cfg(test)]
  if failed.load(Ordering::Relaxed) {
    hash_power_of_two_subtree_roots_serial(kernel, key_words, flags, base_counter, input, subtree_chunks, out);
  }
}

/// Tuning-only hooks (not part of the stable API).
#[cfg(feature = "std")]
#[doc(hidden)]
pub mod tune {
  // Re-export types needed for policy construction
  use super::{Mutex, PARALLEL_OVERRIDE, dispatch, dispatch_tables};
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
  /// This is used by legacy tuning tooling to benchmark single-thread vs multi-thread
  /// behavior without relying on user-facing environment variables.
  #[must_use]
  pub fn override_blake3_parallel_policy(table: dispatch_tables::ParallelTable) -> Blake3ParallelOverrideGuard {
    override_blake3_parallel_policy_full(ParallelPolicyOverride::new(table))
  }

  /// Override the active BLAKE3 parallel policy with per-variant control.
  ///
  /// This allows independent tuning of:
  /// - oneshot: regular one-shot hashing (`digest`, `xof`)
  /// - keyed_oneshot: keyed one-shot hashing (`keyed_digest`, `keyed_xof`)
  /// - derive_oneshot: derive-key one-shot hashing (`derive_key`)
  /// - streaming: regular streaming updates
  /// - keyed_streaming: keyed streaming updates
  /// - derive_streaming: derive-key streaming updates
  ///
  /// This is used by legacy tuning tooling for fine-grained threshold tuning.
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

  /// Return the current SIMD threshold for streaming operations.
  ///
  /// This is used by legacy tuning tooling to ensure consistent SIMD threshold
  /// tuning across all operation variants.
  #[inline]
  #[must_use]
  pub fn streaming_simd_threshold() -> usize {
    dispatch::streaming_simd_threshold()
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
    let src = bytes.as_ptr() as *const u32;
    // SAFETY: `bytes` is exactly 32 bytes; `read_unaligned` supports the
    // 1-byte alignment of `[u8; 32]`.
    unsafe {
      [
        ptr::read_unaligned(src.add(0)),
        ptr::read_unaligned(src.add(1)),
        ptr::read_unaligned(src.add(2)),
        ptr::read_unaligned(src.add(3)),
        ptr::read_unaligned(src.add(4)),
        ptr::read_unaligned(src.add(5)),
        ptr::read_unaligned(src.add(6)),
        ptr::read_unaligned(src.add(7)),
      ]
    }
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
    let src = bytes.as_ptr() as *const u32;
    // SAFETY: `bytes` is exactly 64 bytes; `read_unaligned` supports the
    // 1-byte alignment of `[u8; 64]`.
    unsafe {
      [
        ptr::read_unaligned(src.add(0)),
        ptr::read_unaligned(src.add(1)),
        ptr::read_unaligned(src.add(2)),
        ptr::read_unaligned(src.add(3)),
        ptr::read_unaligned(src.add(4)),
        ptr::read_unaligned(src.add(5)),
        ptr::read_unaligned(src.add(6)),
        ptr::read_unaligned(src.add(7)),
        ptr::read_unaligned(src.add(8)),
        ptr::read_unaligned(src.add(9)),
        ptr::read_unaligned(src.add(10)),
        ptr::read_unaligned(src.add(11)),
        ptr::read_unaligned(src.add(12)),
        ptr::read_unaligned(src.add(13)),
        ptr::read_unaligned(src.add(14)),
        ptr::read_unaligned(src.add(15)),
      ]
    }
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

#[cfg(feature = "std")]
#[inline]
fn reduce_power_of_two_chunk_cvs_any(
  kernel: Kernel,
  key_words: [u32; 8],
  flags: u32,
  cvs: &[[u32; 8]],
  threads_total: usize,
) -> [u32; 8] {
  debug_assert!(cvs.len().is_power_of_two());

  if cvs.len() == 1 {
    return cvs[0];
  }
  if cvs.len() <= 16 {
    // Small subtrees are faster to fold serially (and avoid allocation / thread coordination).
    return reduce_power_of_two_chunk_cvs(kernel, key_words, flags, cvs);
  }

  let threads_total = threads_total.max(1);

  let mut buf0 = alloc::vec![[0u32; 8]; cvs.len() / 2];
  let mut buf1 = alloc::vec![[0u32; 8]; cvs.len() / 2];

  enum Cur<'a> {
    Input(&'a [[u32; 8]]),
    Buf0,
    Buf1,
  }

  let mut cur = Cur::Input(cvs);
  let mut cur_len = cvs.len();

  loop {
    let pairs = cur_len / 2;
    debug_assert!(pairs != 0);

    // For large levels, parallelize parent folding. For small levels, the SIMD
    // parent kernel is faster than coordinating threads.
    //
    // Note: at 1 MiB (1024 chunks), the first reduction level is 512 pairs. If
    // we require a huge minimum here, we'd never parallelize parent folding on
    // common “large input” sizes.
    //
    // 256 pairs = 512 child CVs. That's enough work (and memory traffic) to
    // amortize Rayon scheduling overhead on typical desktop/server CPUs.
    const MIN_PAIRS_FOR_PARALLEL: usize = 256;

    let out0: [u32; 8] = match cur {
      Cur::Input(children) => {
        let out = &mut buf0[..pairs];
        let children = &children[..2 * pairs];
        if threads_total > 1 && pairs >= MIN_PAIRS_FOR_PARALLEL {
          parent_cvs_many_from_cvs_parallel_rayon(kernel, key_words, flags, children, out, threads_total);
        } else {
          kernels::parent_cvs_many_from_cvs_inline(kernel.id, children, key_words, flags, out);
        }
        cur = Cur::Buf0;
        out[0]
      }
      Cur::Buf0 => {
        let out = &mut buf1[..pairs];
        let children = &buf0[..2 * pairs];
        if threads_total > 1 && pairs >= MIN_PAIRS_FOR_PARALLEL {
          parent_cvs_many_from_cvs_parallel_rayon(kernel, key_words, flags, children, out, threads_total);
        } else {
          kernels::parent_cvs_many_from_cvs_inline(kernel.id, children, key_words, flags, out);
        }
        cur = Cur::Buf1;
        out[0]
      }
      Cur::Buf1 => {
        let out = &mut buf0[..pairs];
        let children = &buf1[..2 * pairs];
        if threads_total > 1 && pairs >= MIN_PAIRS_FOR_PARALLEL {
          parent_cvs_many_from_cvs_parallel_rayon(kernel, key_words, flags, children, out, threads_total);
        } else {
          kernels::parent_cvs_many_from_cvs_inline(kernel.id, children, key_words, flags, out);
        }
        cur = Cur::Buf0;
        out[0]
      }
    };

    if pairs == 1 {
      return out0;
    }
    cur_len = pairs;
  }
}

#[cfg_attr(not(target_endian = "little"), allow(dead_code))]
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

#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
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

  let max_leaf_bytes = kernel.simd_degree * CHUNK_LEN;
  if input.len() <= max_leaf_bytes {
    let chunks_exact = input.chunks_exact(CHUNK_LEN);
    let full_chunks = chunks_exact.len();
    debug_assert!(full_chunks <= kernel.simd_degree);
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

  debug_assert!(out.len() >= kernel.simd_degree.max(2));
  let left_len = left_subtree_len_bytes(input.len());
  let (left, right) = input.split_at(left_len);
  let right_chunk_counter = chunk_counter + (left.len() / CHUNK_LEN) as u64;

  const MAX_SIMD_DEGREE: usize = 16;
  let mut cv_array = [[0u8; OUT_LEN]; 2 * MAX_SIMD_DEGREE];

  let simd = kernel.simd_degree;
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

#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
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
    kernel,
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
  kernel: Kernel,
  input_chaining_value: [u32; 8],
  block_words: [u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
}

impl OutputState {
  #[inline]
  fn chaining_value(&self) -> [u32; 8] {
    first_8_words((self.kernel.compress)(
      &self.input_chaining_value,
      &self.block_words,
      self.counter,
      self.block_len,
      self.flags,
    ))
  }

  #[inline]
  fn root_hash_words(&self) -> [u32; 8] {
    first_8_words((self.kernel.compress)(
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
  fn root_output_blocks_into(&self, mut output_block_counter: u64, mut out: &mut [u8]) {
    debug_assert!(out.len().is_multiple_of(OUTPUT_BLOCK_LEN));
    let flags = self.flags | ROOT;

    while !out.is_empty() {
      let blocks_remaining = out.len() / OUTPUT_BLOCK_LEN;
      #[cfg(not(target_arch = "x86_64"))]
      let _ = blocks_remaining;

      #[cfg(target_arch = "x86_64")]
      {
        match self.kernel.id {
          kernels::Blake3KernelId::X86Avx512 if blocks_remaining >= 16 => {
            // SAFETY: required CPU features are validated by dispatch before
            // selecting this kernel, and `out` has at least 16 blocks.
            unsafe {
              x86_64::avx512::root_output_blocks16(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(16);
            out = &mut out[16 * OUTPUT_BLOCK_LEN..];
            continue;
          }
          kernels::Blake3KernelId::X86Avx512 if blocks_remaining >= 8 => {
            // SAFETY: AVX-512 implies AVX2 on the platforms we care about, and
            // dispatch only selects AVX-512 when the required caps are present.
            unsafe {
              x86_64::avx2::root_output_blocks8(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(8);
            out = &mut out[8 * OUTPUT_BLOCK_LEN..];
            continue;
          }
          kernels::Blake3KernelId::X86Avx512 if blocks_remaining >= 4 => {
            // SAFETY: AVX-512 implies SSE4.1 on the platforms we care about, and
            // dispatch only selects AVX-512 when the required caps are present.
            unsafe {
              x86_64::sse41::root_output_blocks4(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(4);
            out = &mut out[4 * OUTPUT_BLOCK_LEN..];
            continue;
          }
          kernels::Blake3KernelId::X86Avx512 if blocks_remaining >= 2 => {
            // SAFETY: AVX-512 is available, and dispatch validates CPU features.
            // On ASM platforms this uses AVX-512 asm; on others it falls back to SSE4.1.
            unsafe {
              x86_64::avx512::root_output_blocks2(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(2);
            out = &mut out[2 * OUTPUT_BLOCK_LEN..];
            continue;
          }
          kernels::Blake3KernelId::X86Avx512 if blocks_remaining >= 1 => {
            // SAFETY: AVX-512 is available, and dispatch validates CPU features.
            // On ASM platforms this uses AVX-512 asm; on others it falls back to SSE4.1.
            unsafe {
              x86_64::avx512::root_output_blocks1(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(1);
            out = &mut out[OUTPUT_BLOCK_LEN..];
            continue;
          }
          kernels::Blake3KernelId::X86Avx2 if blocks_remaining >= 8 => {
            // SAFETY: required CPU features are validated by dispatch before
            // selecting this kernel, and `out` has at least 8 blocks.
            unsafe {
              x86_64::avx2::root_output_blocks8(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(8);
            out = &mut out[8 * OUTPUT_BLOCK_LEN..];
            continue;
          }
          kernels::Blake3KernelId::X86Avx2 if blocks_remaining >= 4 => {
            // SAFETY: AVX2 implies SSE4.1 on the platforms we care about, and
            // dispatch only selects AVX2 when the required caps are present.
            unsafe {
              x86_64::sse41::root_output_blocks4(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(4);
            out = &mut out[4 * OUTPUT_BLOCK_LEN..];
            continue;
          }
          kernels::Blake3KernelId::X86Avx2 if blocks_remaining >= 2 => {
            // SAFETY: AVX2 implies SSE4.1 on the platforms we care about.
            unsafe {
              x86_64::avx2::root_output_blocks2(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(2);
            out = &mut out[2 * OUTPUT_BLOCK_LEN..];
            continue;
          }
          kernels::Blake3KernelId::X86Avx2 if blocks_remaining >= 1 => {
            // SAFETY: AVX2 implies SSE4.1 on the platforms we care about.
            unsafe {
              x86_64::avx2::root_output_blocks1(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(1);
            out = &mut out[OUTPUT_BLOCK_LEN..];
            continue;
          }
          kernels::Blake3KernelId::X86Sse41 if blocks_remaining >= 4 => {
            // SAFETY: required CPU features are validated by dispatch before
            // selecting this kernel, and `out` has at least 4 blocks.
            unsafe {
              x86_64::sse41::root_output_blocks4(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(4);
            out = &mut out[4 * OUTPUT_BLOCK_LEN..];
            continue;
          }
          kernels::Blake3KernelId::X86Sse41 if blocks_remaining >= 2 => {
            // SAFETY: required CPU features are validated by dispatch.
            unsafe {
              x86_64::sse41::root_output_blocks2(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(2);
            out = &mut out[2 * OUTPUT_BLOCK_LEN..];
            continue;
          }
          kernels::Blake3KernelId::X86Sse41 if blocks_remaining >= 1 => {
            // SAFETY: required CPU features are validated by dispatch.
            unsafe {
              x86_64::sse41::root_output_blocks1(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                flags,
                out.as_mut_ptr(),
              );
            }
            output_block_counter = output_block_counter.wrapping_add(1);
            out = &mut out[OUTPUT_BLOCK_LEN..];
            continue;
          }
          _ => {}
        }
      }

      #[cfg(target_arch = "aarch64")]
      {
        if self.kernel.id == kernels::Blake3KernelId::Aarch64Neon && blocks_remaining >= 4 {
          // SAFETY: required CPU features are validated by dispatch before
          // selecting this kernel, and `out` has at least 4 blocks.
          unsafe {
            aarch64::root_output_blocks4_neon(
              &self.input_chaining_value,
              &self.block_words,
              output_block_counter,
              self.block_len,
              flags,
              out.as_mut_ptr(),
            );
          }
          output_block_counter = output_block_counter.wrapping_add(4);
          out = &mut out[4 * OUTPUT_BLOCK_LEN..];
          continue;
        }
      }

      // Scalar fallback: generate one block at a time.
      let words = (self.kernel.compress)(
        &self.input_chaining_value,
        &self.block_words,
        output_block_counter,
        self.block_len,
        flags,
      );
      out[..OUTPUT_BLOCK_LEN].copy_from_slice(&words16_to_le_bytes(&words));
      output_block_counter = output_block_counter.wrapping_add(1);
      out = &mut out[OUTPUT_BLOCK_LEN..];
    }
  }
}

#[derive(Clone, Copy)]
struct ChunkState {
  kernel: Kernel,
  chaining_value: [u32; 8],
  chunk_counter: u64,
  block: [u8; BLOCK_LEN],
  block_len: u8,
  blocks_compressed: u8,
  flags: u32,
}

impl ChunkState {
  #[inline]
  fn new(key_words: [u32; 8], chunk_counter: u64, flags: u32, kernel: Kernel) -> Self {
    Self {
      kernel,
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

  fn update(&mut self, mut input: &[u8]) {
    // Streaming fast path: when we receive exactly one whole chunk at a chunk
    // boundary, compute the internal state in one shot.
    #[cfg(target_arch = "aarch64")]
    {
      if self.kernel.id == kernels::Blake3KernelId::Aarch64Neon
        && self.blocks_compressed == 0
        && self.block_len == 0
        && input.len() == CHUNK_LEN
      {
        let mut cv_words = [0u32; 8];
        let mut last_block = [0u8; BLOCK_LEN];

        // SAFETY: input is exactly one full chunk.
        unsafe {
          aarch64::chunk_state_one_chunk_aarch64_out(
            input.as_ptr(),
            &self.chaining_value,
            self.chunk_counter,
            self.flags,
            cv_words.as_mut_ptr(),
            last_block.as_mut_ptr(),
          );
        }

        self.chaining_value = cv_words;
        self.block = last_block;
        self.block_len = BLOCK_LEN as u8;
        self.blocks_compressed = 15;
        return;
      }
    }

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
          self.kernel.id,
          &mut self.chaining_value,
          self.chunk_counter,
          self.flags,
          &mut self.blocks_compressed,
          &self.block,
        );
        self.block_len = 0;
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
            self.kernel.id,
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
  fn output(&self) -> OutputState {
    let mut block = self.block;
    if self.block_len as usize != BLOCK_LEN {
      block[self.block_len as usize..].fill(0);
    }
    let block_words = words16_from_le_bytes_64(&block);
    OutputState {
      kernel: self.kernel,
      input_chaining_value: self.chaining_value,
      block_words,
      counter: self.chunk_counter,
      block_len: self.block_len as u32,
      flags: self.flags | self.start_flag() | CHUNK_END,
    }
  }
}

#[inline]
fn parent_output(
  kernel: Kernel,
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> OutputState {
  let mut block_words = [0u32; 16];
  block_words[..8].copy_from_slice(&left_child_cv);
  block_words[8..].copy_from_slice(&right_child_cv);
  OutputState {
    kernel,
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
        kernel,
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
    kernel,
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

  #[cfg(not(feature = "std"))]
  let _ = mode;
  #[cfg(feature = "std")]
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
  if remainder == 0 && full_chunks.is_power_of_two() && full_chunks <= FAST_TREE_MAX_CHUNKS {
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
      while cur_len > 2 {
        let pairs = cur_len / 2;
        kernels::parent_cvs_many_from_cvs_inline(kernel.id, &cur[..cur_len], key_words, flags, &mut next[..pairs]);
        cur[..pairs].copy_from_slice(&next[..pairs]);
        cur_len = pairs;
      }
      return parent_output(kernel, cur[0], cur[1], key_words, flags);
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
      while cur_len > 2 {
        let pairs = cur_len / 2;
        kernels::parent_cvs_many_from_bytes_inline(kernel.id, &cur[..cur_len], key_words, flags, &mut next[..pairs]);
        cur[..pairs].copy_from_slice(&next[..pairs]);
        cur_len = pairs;
      }
      let left = words8_from_le_bytes_32(&cur[0]);
      let right = words8_from_le_bytes_32(&cur[1]);
      return parent_output(kernel, left, right, key_words, flags);
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
  let mut output = parent_output(kernel, left, right_cv, key_words, flags);
  while parent_nodes_remaining > 0 {
    parent_nodes_remaining -= 1;
    // SAFETY: `cv_stack_len` tracks the number of initialized entries.
    let left = unsafe { cv_stack[parent_nodes_remaining].assume_init_read() };
    output = parent_output(kernel, left, output.chaining_value(), key_words, flags);
  }

  output
}

#[cfg(feature = "std")]
#[inline]
fn hash_one_full_chunk_cv(
  kernel: Kernel,
  key_words: [u32; 8],
  flags: u32,
  chunk_counter: u64,
  chunk: &[u8],
) -> [u32; 8] {
  debug_assert_eq!(chunk.len(), CHUNK_LEN);

  let mut out = [0u8; OUT_LEN];
  // SAFETY: `chunk` is exactly one full chunk, and `out` is OUT_LEN bytes.
  unsafe { (kernel.hash_many_contiguous)(chunk.as_ptr(), 1, &key_words, chunk_counter, flags, out.as_mut_ptr()) };
  words8_from_le_bytes_32(&out)
}

#[cfg(feature = "std")]
#[inline]
fn hash_full_chunks_cvs_scoped(
  kernel: Kernel,
  key_words: [u32; 8],
  flags: u32,
  base_counter: u64,
  input: &[u8],
  out: &mut [[u32; 8]],
  mut threads_total: usize,
) {
  let out_len = out.len();
  debug_assert_eq!(input.len(), out_len * CHUNK_LEN);

  if threads_total <= 1 || out_len < 2 {
    hash_full_chunks_cvs_parallel_rayon(kernel, key_words, flags, base_counter, input, out, 1);
    return;
  }

  threads_total = threads_total.min(out_len);
  hash_full_chunks_cvs_parallel_rayon(kernel, key_words, flags, base_counter, input, out, threads_total);
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

  // Fallback: construct the root output and extract the root hash words.
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
  dispatch_plan: dispatch::HasherDispatch,
  kernel: Kernel,
  bulk_kernel: Kernel,
  chunk_state: ChunkState,
  pending_chunk_cv: Option<[u32; 8]>,
  key_words: [u32; 8],
  cv_stack: [MaybeUninit<[u32; 8]>; CV_STACK_LEN],
  cv_stack_len: u8,
  #[cfg(feature = "std")]
  parallel_roots_scratch: alloc::vec::Vec<[u32; 8]>,
  #[cfg(feature = "std")]
  parallel_leaf_cvs_scratch: alloc::vec::Vec<[u32; 8]>,
  flags: u32,
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
    Blake3Xof::new(
      root_output_oneshot(
        kernel,
        key_words,
        KEYED_HASH,
        policy_kind_from_flags(KEYED_HASH, true),
        data,
      ),
      plan,
    )
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
  /// This is crate-internal glue for `hashes::bench` / legacy tuning tooling.
  #[inline]
  #[must_use]
  pub(crate) fn digest_with_kernel_id(id: kernels::Blake3KernelId, data: &[u8]) -> [u8; OUT_LEN] {
    let kernel = kernels::kernel(id);
    digest_oneshot(kernel, IV, 0, data)
  }

  /// One-shot keyed hash using an explicitly selected kernel.
  ///
  /// This is crate-internal glue for `hashes::bench` / legacy tuning tooling.
  #[inline]
  #[must_use]
  pub(crate) fn keyed_digest_with_kernel_id(id: kernels::Blake3KernelId, key: &[u8; 32], data: &[u8]) -> [u8; OUT_LEN] {
    let kernel = kernels::kernel(id);
    let key_words = words8_from_le_bytes_32(key);
    digest_oneshot(kernel, key_words, KEYED_HASH, data)
  }

  /// One-shot derive-key hash using an explicitly selected kernel.
  ///
  /// This is crate-internal glue for `hashes::bench` / legacy tuning tooling.
  #[inline]
  #[must_use]
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
  fn stream_chunks_pattern_with_kernel_pair_and_state(
    stream_id: kernels::Blake3KernelId,
    bulk_id: kernels::Blake3KernelId,
    chunk_pattern: &[usize],
    key_words: [u32; 8],
    flags: u32,
    xof_mode: bool,
    data: &[u8],
  ) -> [u8; OUT_LEN] {
    let stream = kernels::kernel(stream_id);
    let bulk = kernels::kernel(bulk_id);
    let mut h = Self::new_internal_with(key_words, flags, stream);
    if chunk_pattern.is_empty() {
      h.update_with(data, stream, bulk);
    } else {
      let mut offset = 0usize;
      let mut idx = 0usize;
      while offset < data.len() {
        let step = chunk_pattern[idx % chunk_pattern.len()].max(1);
        let end = offset.saturating_add(step).min(data.len());
        h.update_with(&data[offset..end], stream, bulk);
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
  /// This is crate-internal glue for `hashes::bench` / legacy tuning tooling.
  #[inline]
  #[must_use]
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
  /// This is crate-internal glue for `hashes::bench` / legacy tuning tooling.
  #[inline]
  #[must_use]
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
  /// This is crate-internal glue for `hashes::bench` / legacy tuning tooling.
  #[inline]
  #[must_use]
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
  /// This is crate-internal glue for `hashes::bench` / legacy tuning tooling.
  #[inline]
  #[must_use]
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
  /// This is crate-internal glue for `hashes::bench` / legacy tuning tooling.
  #[inline]
  #[must_use]
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
  /// This is crate-internal glue for `hashes::bench` / legacy tuning tooling.
  #[inline]
  #[must_use]
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
  /// This is crate-internal glue for `hashes::bench` / legacy tuning tooling.
  #[inline]
  #[must_use]
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
  /// This is crate-internal glue for `hashes::bench` / legacy tuning tooling.
  #[inline]
  #[must_use]
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
  pub(crate) fn digest_portable(data: &[u8]) -> [u8; OUT_LEN] {
    let kernel = kernels::kernel(kernels::Blake3KernelId::Portable);
    if data.len() <= CHUNK_LEN {
      let output = single_chunk_output(kernel, IV, 0, 0, data);
      output.root_hash_bytes()
    } else {
      let mut h = Self::new_internal_with(IV, 0, kernel);
      h.update_with(data, kernel, kernel);
      h.finalize()
    }
  }

  #[inline]
  fn new_internal(key_words: [u32; 8], flags: u32) -> Self {
    let kernel = kernels::kernel(kernels::Blake3KernelId::Portable);
    Self::new_internal_with(key_words, flags, kernel)
  }

  #[inline]
  fn new_internal_with(key_words: [u32; 8], flags: u32, kernel: Kernel) -> Self {
    let dispatch_plan = dispatch::hasher_dispatch();
    Self {
      dispatch_plan,
      kernel,
      bulk_kernel: kernel,
      chunk_state: ChunkState::new(key_words, 0, flags, kernel),
      pending_chunk_cv: None,
      key_words,
      cv_stack: uninit_cv_stack(),
      cv_stack_len: 0,
      #[cfg(feature = "std")]
      parallel_roots_scratch: alloc::vec::Vec::new(),
      #[cfg(feature = "std")]
      parallel_leaf_cvs_scratch: alloc::vec::Vec::new(),
      flags,
    }
  }

  #[cfg(feature = "std")]
  #[inline]
  fn streaming_parallel_threads(
    &self,
    input_bytes: usize,
    admission_full_chunks: usize,
    commit_full_chunks: usize,
  ) -> Option<usize> {
    let (policy, mode) = if let Some(override_policy) = parallel_override() {
      let mode = match self.flags {
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
      (self.dispatch_plan.parallel_streaming(), ParallelPolicyKind::Update)
    };
    parallel_threads_for_policy(mode, policy, input_bytes, admission_full_chunks, commit_full_chunks)
  }

  #[cfg(feature = "std")]
  #[cold]
  #[inline(never)]
  fn commit_parallel_batch(
    &mut self,
    batch_input: &[u8],
    base_counter: u64,
    batch: usize,
    commit: usize,
    threads: usize,
    keep_last_full_chunk: bool,
  ) {
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

    let mut stack_len = self.cv_stack_len as usize;
    let mut counter = base_counter;
    let mut offset_chunks = 0usize;
    let mut remaining_commit = commit;

    self.parallel_roots_scratch.clear();
    self.parallel_leaf_cvs_scratch.clear();

    while remaining_commit != 0 {
      let mut size = pow2_floor(remaining_commit);

      let aligned_max = if counter == 0 {
        usize::MAX
      } else {
        let tz = counter.trailing_zeros() as usize;
        if tz >= (usize::BITS as usize) {
          usize::MAX
        } else {
          1usize << tz
        }
      };
      size = size.min(aligned_max).min(remaining_commit);
      debug_assert!(size.is_power_of_two());

      let bytes_base = offset_chunks * CHUNK_LEN;
      let subtree_bytes = size * CHUNK_LEN;
      let subtree_input = &batch_input[bytes_base..bytes_base + subtree_bytes];

      // Fast path: hash this power-of-two subtree without materializing all
      // leaf CVs. Fallback: hash leaf CVs directly and reduce.
      let subtree_cv = if size >= threads && (counter == 0 || (counter & (size as u64 - 1)) == 0) {
        const MAX_SUBTREE_CHUNKS: usize = 1 << 12; // 4096 chunks = 4 MiB per subtree

        let mut subtree_chunks = size / threads;
        subtree_chunks = subtree_chunks.max(1);
        subtree_chunks = pow2_floor(subtree_chunks);
        subtree_chunks = subtree_chunks.min(MAX_SUBTREE_CHUNKS).min(size);

        let roots_len = size / subtree_chunks;
        debug_assert!(roots_len.is_power_of_two());
        debug_assert_eq!(roots_len * subtree_chunks, size);

        self.parallel_roots_scratch.resize(roots_len, [0u32; 8]);
        hash_power_of_two_subtree_roots_parallel_rayon(SubtreeRootsRequest {
          kernel: self.bulk_kernel,
          key_words: self.key_words,
          flags: self.flags,
          base_counter: counter,
          input: subtree_input,
          subtree_chunks,
          out: &mut self.parallel_roots_scratch,
          threads_total: threads,
        });

        reduce_power_of_two_chunk_cvs_any(
          self.bulk_kernel,
          self.key_words,
          self.flags,
          &self.parallel_roots_scratch,
          threads,
        )
      } else {
        self.parallel_leaf_cvs_scratch.resize(size, [0u32; 8]);
        hash_full_chunks_cvs_scoped(
          self.bulk_kernel,
          self.key_words,
          self.flags,
          counter,
          subtree_input,
          &mut self.parallel_leaf_cvs_scratch,
          threads,
        );
        reduce_power_of_two_chunk_cvs_any(
          self.bulk_kernel,
          self.key_words,
          self.flags,
          &self.parallel_leaf_cvs_scratch,
          threads,
        )
      };

      counter = counter.wrapping_add(size as u64);
      let level = size.trailing_zeros();
      let mut total = counter >> level;
      let mut cv = subtree_cv;
      while total & 1 == 0 {
        cv = kernels::parent_cv_inline(
          self.bulk_kernel.id,
          pop_stack(&mut self.cv_stack, &mut stack_len),
          cv,
          self.key_words,
          self.flags,
        );
        total >>= 1;
      }
      push_stack(&mut self.cv_stack, &mut stack_len, cv);

      offset_chunks += size;
      remaining_commit -= size;
    }

    self.cv_stack_len = stack_len as u8;

    let new_counter = base_counter + batch as u64;
    self.chunk_state = ChunkState::new(self.key_words, new_counter, self.flags, self.kernel);
    if keep_last_full_chunk {
      let last = &batch_input[(batch - 1) * CHUNK_LEN..batch * CHUNK_LEN];
      self.pending_chunk_cv = Some(hash_one_full_chunk_cv(
        self.bulk_kernel,
        self.key_words,
        self.flags,
        new_counter - 1,
        last,
      ));
    }
  }

  #[cfg(feature = "std")]
  #[cold]
  #[inline(never)]
  fn try_parallel_update_batch(&mut self, input: &[u8]) -> Option<usize> {
    // Large updates: multi-thread full-chunk hashing.
    //
    // Prefer subtree-root batching for large, chunk-aligned power-of-two
    // subtrees. This avoids materializing all leaf CVs and reduces memory
    // traffic, while preserving the canonical BLAKE3 tree shape.
    const MAX_PASS_CHUNKS: usize = 1 << 16; // 64 MiB input per pass

    if self.chunk_state.len() != 0 || input.len() <= CHUNK_LEN {
      return None;
    }
    let full_chunks = input.len() / CHUNK_LEN;
    if full_chunks <= 1 {
      return None;
    }

    let batch = core::cmp::min(full_chunks, MAX_PASS_CHUNKS);
    let base_counter = self.chunk_state.chunk_counter;

    let keep_last_full_chunk = input.len().is_multiple_of(CHUNK_LEN) && batch == full_chunks;
    let commit = if keep_last_full_chunk { batch - 1 } else { batch };
    if commit == 0 {
      return None;
    }

    let bytes = batch * CHUNK_LEN;
    let threads = self.streaming_parallel_threads(bytes, batch, commit)?;
    if threads <= 1 {
      return None;
    }

    let batch_input = &input[..bytes];
    self.commit_parallel_batch(batch_input, base_counter, batch, commit, threads, keep_last_full_chunk);
    Some(bytes)
  }

  #[inline(never)]
  fn try_simd_update_batch(&mut self, input: &[u8]) -> Option<usize> {
    if self.chunk_state.len() != 0 || self.bulk_kernel.simd_degree <= 1 || input.len() <= CHUNK_LEN {
      return None;
    }

    let full_chunks = input.len() / CHUNK_LEN;
    if full_chunks <= 1 {
      return None;
    }

    const MAX_SIMD_DEGREE: usize = 16;
    let batch = core::cmp::min(full_chunks, MAX_SIMD_DEGREE);
    if batch == 0 {
      return None;
    }

    let mut out_buf = [0u8; OUT_LEN * MAX_SIMD_DEGREE];
    let base_counter = self.chunk_state.chunk_counter;
    // SAFETY: `input` has at least `batch * CHUNK_LEN` bytes, `out_buf`
    // has at least `batch * OUT_LEN` bytes, and this kernel was selected
    // only when its required CPU features are available.
    unsafe {
      (self.bulk_kernel.hash_many_contiguous)(
        input.as_ptr(),
        batch,
        &self.key_words,
        base_counter,
        self.flags,
        out_buf.as_mut_ptr(),
      )
    };

    let keep_last_full_chunk = input.len().is_multiple_of(CHUNK_LEN) && batch == full_chunks;
    let commit = if keep_last_full_chunk { batch - 1 } else { batch };
    if commit != 0 {
      // SAFETY: `out_buf` stores `batch` contiguous CV outputs, and
      // `commit <= batch <= MAX_SIMD_DEGREE`.
      let cvs_bytes: &[[u8; OUT_LEN]] =
        unsafe { slice::from_raw_parts(out_buf.as_ptr().cast::<[u8; OUT_LEN]>(), commit) };
      let mut stack_len = self.cv_stack_len as usize;
      add_chunk_cvs_batched_bytes(
        self.bulk_kernel,
        &mut self.cv_stack,
        &mut stack_len,
        base_counter,
        cvs_bytes,
        self.key_words,
        self.flags,
      );
      self.cv_stack_len = stack_len as u8;
    }

    let new_counter = base_counter + batch as u64;
    self.chunk_state = ChunkState::new(self.key_words, new_counter, self.flags, self.kernel);
    if keep_last_full_chunk {
      let offset = (batch - 1) * OUT_LEN;
      // SAFETY: `out_buf` is `OUT_LEN * MAX_SIMD_DEGREE`, and `offset`
      // is `(batch - 1) * OUT_LEN` with `batch <= MAX_SIMD_DEGREE`.
      let cv = unsafe { words8_from_le_bytes_32(&*(out_buf.as_ptr().add(offset) as *const [u8; OUT_LEN])) };
      self.pending_chunk_cv = Some(cv);
    }

    Some(batch * CHUNK_LEN)
  }

  fn update_with(&mut self, mut input: &[u8], stream_kernel: Kernel, bulk_kernel: Kernel) {
    self.kernel = stream_kernel;
    self.bulk_kernel = bulk_kernel;
    self.chunk_state.kernel = stream_kernel;

    // If the previous update ended exactly on a chunk boundary, we may have
    // stored the last full chunk's CV instead of keeping a fully-buffered
    // `ChunkState`. As soon as more input arrives, that chunk is no longer
    // terminal and can be committed to the tree.
    if !input.is_empty()
      && let Some(cv) = self.pending_chunk_cv.take()
    {
      let total_chunks = self.chunk_state.chunk_counter;
      self.add_chunk_chaining_value(cv, total_chunks);
    }

    // When we're at a chunk boundary and we have more than one whole chunk
    // available, use the kernel's multi-chunk primitive to hash whole chunks
    // directly and feed their chaining values into the tree.
    //
    // Important: we always leave at least one full chunk (or partial) to be
    // processed by `ChunkState::update`, so that the streaming state retains
    // the "buffer the last block" invariant needed when the caller stops at
    // a block boundary and later calls `finalize`.
    while !input.is_empty() {
      if self.chunk_state.len() == CHUNK_LEN {
        let chunk_cv = self.chunk_state.output().chaining_value();
        let total_chunks = self.chunk_state.chunk_counter + 1;
        self.add_chunk_chaining_value(chunk_cv, total_chunks);
        self.chunk_state = ChunkState::new(self.key_words, total_chunks, self.flags, self.kernel);
      }

      #[cfg(feature = "std")]
      {
        if self.chunk_state.len() == 0
          && input.len() > CHUNK_LEN
          && let Some(consumed) = self.try_parallel_update_batch(input)
        {
          input = &input[consumed..];
          continue;
        }
      }

      if let Some(consumed) = self.try_simd_update_batch(input) {
        input = &input[consumed..];
        continue;
      }

      let want = CHUNK_LEN - self.chunk_state.len();
      let take = min(want, input.len());
      self.chunk_state.update(&input[..take]);
      input = &input[take..];
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

  fn add_chunk_chaining_value(&mut self, mut new_cv: [u32; 8], mut total_chunks: u64) {
    while total_chunks & 1 == 0 {
      new_cv = kernels::parent_cv_inline(self.kernel.id, self.pop_stack(), new_cv, self.key_words, self.flags);
      total_chunks >>= 1;
    }
    self.push_stack(new_cv);
  }

  fn root_output(&self) -> OutputState {
    let mut parent_nodes_remaining = self.cv_stack_len as usize;
    let mut output = if let Some(right_cv) = self.pending_chunk_cv {
      debug_assert!(
        parent_nodes_remaining > 0,
        "pending full chunk implies multi-chunk input"
      );
      parent_nodes_remaining -= 1;
      // SAFETY: `cv_stack_len` tracks the number of initialized entries.
      let left = unsafe { *self.cv_stack[parent_nodes_remaining].assume_init_ref() };
      parent_output(self.kernel, left, right_cv, self.key_words, self.flags)
    } else {
      self.chunk_state.output()
    };

    while parent_nodes_remaining > 0 {
      parent_nodes_remaining -= 1;
      // SAFETY: `cv_stack_len` tracks the number of initialized entries.
      let left = unsafe { *self.cv_stack[parent_nodes_remaining].assume_init_ref() };
      output = parent_output(self.kernel, left, output.chaining_value(), self.key_words, self.flags);
    }
    output
  }

  /// Finalize into an extendable output state (XOF).
  #[must_use]
  #[inline]
  pub fn finalize_xof(&self) -> Blake3Xof {
    // Mirror `finalize()` for the empty-input case: don't stay pinned to the
    // portable default when we could use the tuned streaming kernel.
    if self.chunk_state.chunk_counter == 0
      && self.chunk_state.len() == 0
      && self.cv_stack_len == 0
      && self.pending_chunk_cv.is_none()
    {
      Blake3Xof::new(
        single_chunk_output(self.dispatch_plan.stream_kernel(), self.key_words, 0, self.flags, &[]),
        self.dispatch_plan,
      )
    } else {
      Blake3Xof::new(self.root_output(), self.dispatch_plan)
    }
  }

  /// Finalize into an extendable output state (XOF) with output-size-aware kernel selection.
  ///
  /// This is the world-class XOF entry point: it uses the expected output size to select
  /// the optimal kernel. For small inputs with large expected outputs, this ensures we
  /// use SIMD kernels for the XOF squeeze phase, avoiding the "+500% XOF slowdown" that
  /// occurs when tiny inputs pin us to portable kernels.
  ///
  /// # Arguments
  /// * `expected_output_bytes` - The expected number of bytes to squeeze from this XOF. This is
  ///   used as a hint for kernel selection. The actual squeezed amount can differ.
  ///
  /// # Example
  /// ```rust
  /// use hashes::crypto::Blake3;
  /// use traits::{Digest as _, Xof};
  ///
  /// let mut hasher = Blake3::new();
  /// hasher.update(b"small input");
  /// // Even though input is small, we'll use SIMD for the 1KB output
  /// let mut xof = hasher.finalize_xof_sized(1024);
  /// let mut out = [0u8; 1024];
  /// xof.squeeze(&mut out);
  /// assert_ne!(out, [0u8; 1024]);
  /// ```
  #[must_use]
  #[inline]
  pub fn finalize_xof_sized(&self, expected_output_bytes: usize) -> Blake3Xof {
    // World-class XOF: select kernel based on output size, not just input size.
    // This is the fix for the "+500% XOF slowdown" on x86_64.
    let kernel = if expected_output_bytes >= 512 {
      // For large expected outputs, force SIMD kernel regardless of input size.
      // The throughput benefit outweighs any setup overhead.
      self.dispatch_plan.size_class_kernel(expected_output_bytes)
    } else {
      // For small outputs, use streaming dispatch (input-size-aware).
      self.dispatch_plan.stream_kernel()
    };

    // Re-create the root output with the selected kernel for optimal squeeze performance.
    let output = if self.chunk_state.chunk_counter == 0
      && self.chunk_state.len() == 0
      && self.cv_stack_len == 0
      && self.pending_chunk_cv.is_none()
    {
      // Empty input: single chunk output
      single_chunk_output(kernel, self.key_words, 0, self.flags, &[])
    } else if self.chunk_state.chunk_counter == 0 && self.pending_chunk_cv.is_none() {
      // Single chunk input: re-create output with the SIMD kernel
      let mut temp_chunk_state = ChunkState::new(self.key_words, 0, self.flags, kernel);
      // Copy the accumulated state
      temp_chunk_state.chaining_value = self.chunk_state.chaining_value;
      temp_chunk_state.block = self.chunk_state.block;
      temp_chunk_state.block_len = self.chunk_state.block_len;
      temp_chunk_state.blocks_compressed = self.chunk_state.blocks_compressed;
      temp_chunk_state.output()
    } else {
      // Multi-chunk: use root output (tree structure is already built)
      // But we need to rebuild parent outputs with the selected kernel
      self.root_output_with_kernel(kernel)
    };

    Blake3Xof::new_with_kernel(output, kernel, self.dispatch_plan)
  }

  /// Get root output with a specific kernel (for XOF kernel switching).
  #[inline]
  fn root_output_with_kernel(&self, kernel: Kernel) -> OutputState {
    let mut parent_nodes_remaining = self.cv_stack_len as usize;
    let mut output = if let Some(right_cv) = self.pending_chunk_cv {
      debug_assert!(
        parent_nodes_remaining > 0,
        "pending full chunk implies multi-chunk input"
      );
      parent_nodes_remaining -= 1;
      // SAFETY: `cv_stack_len` tracks the number of initialized entries.
      let left = unsafe { *self.cv_stack[parent_nodes_remaining].assume_init_ref() };
      parent_output(kernel, left, right_cv, self.key_words, self.flags)
    } else {
      // Re-create chunk state output with the new kernel
      let mut temp_chunk_state = ChunkState::new(self.key_words, self.chunk_state.chunk_counter, self.flags, kernel);
      temp_chunk_state.chaining_value = self.chunk_state.chaining_value;
      temp_chunk_state.block = self.chunk_state.block;
      temp_chunk_state.block_len = self.chunk_state.block_len;
      temp_chunk_state.blocks_compressed = self.chunk_state.blocks_compressed;
      temp_chunk_state.output()
    };

    while parent_nodes_remaining > 0 {
      parent_nodes_remaining -= 1;
      // SAFETY: `cv_stack_len` tracks the number of initialized entries.
      let left = unsafe { *self.cv_stack[parent_nodes_remaining].assume_init_ref() };
      output = parent_output(kernel, left, output.chaining_value(), self.key_words, self.flags);
    }
    output
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

    let first_update = self.chunk_state.chunk_counter == 0 && self.cv_stack_len == 0 && self.pending_chunk_cv.is_none();

    // Ultra-tiny first update fast path: avoid `update_with` dispatch and loop
    // machinery when we can satisfy the call with a single block copy.
    if first_update
      && self.chunk_state.blocks_compressed == 0
      && self.chunk_state.block_len == 0
      && input.len() <= BLOCK_LEN
    {
      self.chunk_state.block[..input.len()].copy_from_slice(input);
      self.chunk_state.block_len = input.len() as u8;
      return;
    }

    // Streaming has two competing goals:
    // - Be very fast for large updates (file hashing, storage, etc.).
    // - Avoid SIMD setup overhead for tiny one-off updates.
    //
    // We start in portable mode and only "lock in" streaming dispatch once we
    // have enough data to amortize kernel setup costs. This applies uniformly
    // to all modes (plain, keyed, derive) - the defer heuristic improves
    // small-input latency across the board.
    if first_update
      && self.chunk_state.blocks_compressed == 0
      && self.chunk_state.block_len == 0
      && self.flags == 0
      && input.len() == CHUNK_LEN
      && let Some(kernel) = self.dispatch_plan.plain_first_update_1024_kernel()
    {
      // Intel server-only narrow override: allow plain 1024B first updates to
      // use AVX-512 where lane data shows it wins.
      self.kernel = kernel;
      self.bulk_kernel = kernel;
      self.chunk_state.kernel = kernel;
      self.chunk_state.update(input);
      return;
    }

    if first_update
      && self.chunk_state.blocks_compressed == 0
      && self.chunk_state.block_len == 0
      && input.len() <= CHUNK_LEN
    {
      let stream = if self.kernel.id == kernels::Blake3KernelId::Portable
        && self
          .dispatch_plan
          .should_defer_simd(self.chunk_state.len(), input.len())
      {
        self.kernel
      } else {
        self.dispatch_plan.stream_kernel()
      };
      let bulk = self.dispatch_plan.bulk_kernel_for_update(input.len());
      self.kernel = stream;
      self.bulk_kernel = bulk;
      self.chunk_state.kernel = stream;
      self.chunk_state.update(input);
      return;
    }

    let stream = self.dispatch_plan.stream_kernel();
    let bulk = self.dispatch_plan.bulk_kernel_for_update(input.len());

    self.update_with(input, stream, bulk);
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    // Single-chunk fast path: compute root bytes directly from the current
    // chunk tail. This avoids OutputState construction and bytes<->words
    // conversion on short streaming finalization.
    if self.chunk_state.chunk_counter == 0 && self.cv_stack_len == 0 && self.pending_chunk_cv.is_none() {
      let block_len = self.chunk_state.block_len as usize;

      let add_chunk_start = self.chunk_state.blocks_compressed == 0;
      let cv = self.chunk_state.chaining_value;

      if block_len == BLOCK_LEN {
        let out_words = compress_chunk_tail_to_root_words(
          self.kernel,
          cv,
          &self.chunk_state.block,
          block_len,
          self.flags,
          add_chunk_start,
        );
        return words8_to_le_bytes(&out_words);
      }

      let mut block = self.chunk_state.block;
      block[block_len..].fill(0);
      let out_words =
        compress_chunk_tail_to_root_words(self.kernel, cv, &block, block_len, self.flags, add_chunk_start);
      return words8_to_le_bytes(&out_words);
    }

    // If the caller never provided any input (or only provided empty updates),
    // we still want to use the tuned streaming kernel rather than staying
    // pinned to the portable default.
    let output = if self.chunk_state.chunk_counter == 0
      && self.chunk_state.len() == 0
      && self.cv_stack_len == 0
      && self.pending_chunk_cv.is_none()
    {
      single_chunk_output(self.dispatch_plan.stream_kernel(), self.key_words, 0, self.flags, &[])
    } else {
      self.root_output()
    };
    output.root_hash_bytes()
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::new_internal(self.key_words, self.flags);
  }
}

#[derive(Clone)]
pub struct Blake3Xof {
  output: OutputState,
  block_counter: u64,
  buf: [u8; OUTPUT_BLOCK_LEN],
  buf_pos: usize,
  /// Kernel used for XOF squeeze operations. Stored to enable dynamic upgrade
  /// when large squeezes are requested with a suboptimal kernel.
  kernel: Kernel,
  dispatch_plan: dispatch::HasherDispatch,
}

impl Blake3Xof {
  #[inline]
  fn new(output: OutputState, dispatch_plan: dispatch::HasherDispatch) -> Self {
    let kernel = output.kernel;
    Self {
      output,
      block_counter: 0,
      buf: [0u8; OUTPUT_BLOCK_LEN],
      buf_pos: OUTPUT_BLOCK_LEN,
      kernel,
      dispatch_plan,
    }
  }

  #[inline]
  fn new_with_kernel(output: OutputState, kernel: Kernel, dispatch_plan: dispatch::HasherDispatch) -> Self {
    Self {
      output,
      block_counter: 0,
      buf: [0u8; OUTPUT_BLOCK_LEN],
      buf_pos: OUTPUT_BLOCK_LEN,
      kernel,
      dispatch_plan,
    }
  }

  #[inline]
  fn refill(&mut self) {
    self.output.root_output_blocks_into(self.block_counter, &mut self.buf);
    self.block_counter = self.block_counter.wrapping_add(1);
    self.buf_pos = 0;
  }
}

impl Xof for Blake3Xof {
  fn squeeze(&mut self, mut out: &mut [u8]) {
    if out.is_empty() {
      return;
    }

    // Fast path: the first 32 output bytes are the root hash. Avoid generating
    // a full 64-byte output block (and touching the internal buffer) when the
    // caller only needs up to one hash output.
    if self.block_counter == 0 && self.buf_pos == self.buf.len() && out.len() <= OUT_LEN {
      let rh = self.output.root_hash_bytes();
      out.copy_from_slice(&rh[..out.len()]);
      return;
    }

    // World-class: XOF output generation is dominated by the squeeze phase, not the
    // input hashing phase. Tiny inputs can select a "streaming" kernel (e.g. SSE4.1)
    // that is great for per-block updates but suboptimal for generating many output
    // blocks. For large squeezes, upgrade to the best available bulk kernel.
    //
    // This is the runtime safety net: even if the caller didn't use
    // `finalize_xof_sized()`, we can still avoid being pinned to a small-input
    // kernel for a large output request.
    const LARGE_SQUEEZE_THRESHOLD: usize = 512;
    if out.len() >= LARGE_SQUEEZE_THRESHOLD {
      let desired = self.dispatch_plan.size_class_kernel(out.len());
      if desired.id != self.kernel.id {
        self.kernel = desired;
        self.output.kernel = desired;
      }
    }

    // Drain any buffered bytes first.
    if self.buf_pos != self.buf.len() {
      let take = min(self.buf.len() - self.buf_pos, out.len());
      out[..take].copy_from_slice(&self.buf[self.buf_pos..self.buf_pos + take]);
      self.buf_pos += take;
      out = &mut out[take..];
      if out.is_empty() {
        return;
      }
    }

    // Generate any remaining full output blocks directly into the caller
    // buffer (lets the kernel choose its best batch size).
    let full = out.len() / OUTPUT_BLOCK_LEN * OUTPUT_BLOCK_LEN;
    if full != 0 {
      let blocks = (full / OUTPUT_BLOCK_LEN) as u64;
      self
        .output
        .root_output_blocks_into(self.block_counter, &mut out[..full]);
      self.block_counter = self.block_counter.wrapping_add(blocks);
      out = &mut out[full..];
    }

    // Tail: refill once and copy the remaining bytes.
    if !out.is_empty() {
      self.refill();
      let take = out.len();
      out.copy_from_slice(&self.buf[..take]);
      self.buf_pos = take;
    }
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

// ─────────────────────────────────────────────────────────────────────────────
// x86_64 tiny one-shot helpers (keyed/derive sensitive)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn digest_one_chunk_root_hash_words_x86(
  kernel: Kernel,
  key_words: [u32; 8],
  flags: u32,
  input: &[u8],
) -> [u32; 8] {
  debug_assert!(input.len() <= CHUNK_LEN);

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
  let mut blocks_compressed = 0u8;
  let full_bytes = full_blocks * BLOCK_LEN;
  (kernel.chunk_compress_blocks)(&mut cv, 0, flags, &mut blocks_compressed, &input[..full_bytes]);

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
  let mut blocks_compressed: u8 = 0;
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
  use traits::{Digest, Xof};

  use super::{Blake3, OUT_LEN};

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

  #[cfg(feature = "std")]
  mod parallel_pipeline {
    extern crate alloc;

    use traits::{Digest, Xof};

    use super::{
      super::{
        IV, KEYED_HASH, SubtreeRootsRequest, force_parallel_panic, hash_full_chunks_cvs_serial,
        hash_power_of_two_subtree_roots_parallel_rayon, hash_power_of_two_subtree_roots_serial,
        reduce_power_of_two_chunk_cvs_any, words8_from_le_bytes_32,
      },
      Blake3, CONTEXT, KEY, OUT_LEN, input_pattern,
    };
    use crate::crypto::blake3::{CHUNK_LEN, dispatch, dispatch_tables, tune};

    #[inline]
    fn always_parallel_table(max_threads: u8) -> dispatch_tables::ParallelTable {
      dispatch_tables::ParallelTable {
        min_bytes: 0,
        min_chunks: 0,
        max_threads,
        spawn_cost_bytes: 0,
        merge_cost_bytes: 0,
        bytes_per_core_small: 0,
        bytes_per_core_medium: 0,
        bytes_per_core_large: 0,
        small_limit_bytes: 0,
        medium_limit_bytes: 0,
      }
    }

    #[inline]
    fn never_parallel_table() -> dispatch_tables::ParallelTable {
      // `max_threads == 1` disables parallel hashing (even when thresholds are met).
      always_parallel_table(1)
    }

    #[inline]
    fn finalize_digest_and_xof_prefix(h: &Blake3) -> ([u8; OUT_LEN], [u8; 64]) {
      let digest = h.finalize();
      let mut xof = h.finalize_xof();
      let mut prefix = [0u8; 64];
      xof.squeeze(&mut prefix);
      (digest, prefix)
    }

    fn run_case_streaming(
      make_hasher: impl Fn() -> Blake3,
      prefix: &[u8],
      payload: &[u8],
    ) -> ([u8; OUT_LEN], [u8; 64]) {
      let mut h = make_hasher();
      for chunk in prefix.chunks(CHUNK_LEN) {
        h.update(chunk);
      }
      h.update(payload);
      finalize_digest_and_xof_prefix(&h)
    }

    #[test]
    fn streaming_parallel_matches_serial() {
      let Ok(ap) = std::thread::available_parallelism() else {
        return;
      };
      if ap.get() <= 1 {
        return;
      }

      let payload_lens = [
        64 * 1024,       // 64 KiB (exactly chunk-aligned)
        128 * 1024 + 17, // 128 KiB + 17B (forces remainder path)
      ];

      let max_prefix_chunks = 63usize;
      let max_payload_len = *payload_lens.iter().max().unwrap();
      let input = input_pattern(max_prefix_chunks * CHUNK_LEN + max_payload_len);

      let prefix_chunks_cases = [0usize, 1, 7, 15, 16, 63];
      let mut thread_caps = [2usize, 4, 8];

      // Cap threads at runtime to keep tests robust on small machines.
      for t in &mut thread_caps {
        *t = (*t).min(ap.get()).max(2);
      }

      for &payload_len in &payload_lens {
        for &prefix_chunks in &prefix_chunks_cases {
          let offset = prefix_chunks * CHUNK_LEN;
          let prefix = &input[..offset];
          let payload = &input[offset..offset + payload_len];
          let full_input = &input[..offset + payload_len];

          // Serial baseline (forced single-thread).
          let serial_regular = {
            let _serial_policy = tune::override_blake3_parallel_policy(never_parallel_table());
            run_case_streaming(Blake3::new, prefix, payload)
          };

          // Sanity: one-shot should match streaming in serial mode.
          {
            let _serial_policy = tune::override_blake3_parallel_policy(never_parallel_table());
            assert_eq!(
              serial_regular.0,
              Blake3::digest(full_input),
              "serial oneshot mismatch prefix_chunks={prefix_chunks} payload_len={payload_len}"
            );
          }

          for &threads_total in &thread_caps {
            let threads_total = threads_total as u8;
            let _policy = tune::override_blake3_parallel_policy(always_parallel_table(threads_total));

            let got_regular = run_case_streaming(Blake3::new, prefix, payload);
            assert_eq!(
              got_regular, serial_regular,
              "parallel regular mismatch prefix_chunks={prefix_chunks} payload_len={payload_len} \
               threads_total={threads_total}"
            );
          }
        }
      }
    }

    #[test]
    fn keyed_and_derive_parallel_matches_serial_subset() {
      let Ok(ap) = std::thread::available_parallelism() else {
        return;
      };
      if ap.get() <= 1 {
        return;
      }

      let payload_len = 64 * 1024 + 17;
      let max_prefix_chunks = 63usize;
      let input = input_pattern(max_prefix_chunks * CHUNK_LEN + payload_len);

      let prefix_chunks_cases = [0usize, 15, 63];
      let mut thread_caps = [2usize, 8];
      for t in &mut thread_caps {
        *t = (*t).min(ap.get()).max(2);
      }

      for &prefix_chunks in &prefix_chunks_cases {
        let offset = prefix_chunks * CHUNK_LEN;
        let prefix = &input[..offset];
        let payload = &input[offset..offset + payload_len];
        let full_input = &input[..offset + payload_len];

        let (serial_keyed, serial_derive) = {
          let _serial_policy = tune::override_blake3_parallel_policy(never_parallel_table());
          let keyed = run_case_streaming(|| Blake3::new_keyed(KEY), prefix, payload);
          let derive = run_case_streaming(|| Blake3::new_derive_key(CONTEXT), prefix, payload);
          (keyed, derive)
        };

        {
          let _serial_policy = tune::override_blake3_parallel_policy(never_parallel_table());
          assert_eq!(
            serial_keyed.0,
            Blake3::keyed_digest(KEY, full_input),
            "serial keyed oneshot mismatch prefix_chunks={prefix_chunks}"
          );
          assert_eq!(
            serial_derive.0,
            Blake3::derive_key(CONTEXT, full_input),
            "serial derive oneshot mismatch prefix_chunks={prefix_chunks}"
          );
        }

        for &threads_total in &thread_caps {
          let threads_total = threads_total as u8;
          let _policy = tune::override_blake3_parallel_policy(always_parallel_table(threads_total));

          let got_keyed = run_case_streaming(|| Blake3::new_keyed(KEY), prefix, payload);
          let got_derive = run_case_streaming(|| Blake3::new_derive_key(CONTEXT), prefix, payload);
          assert_eq!(
            got_keyed, serial_keyed,
            "parallel keyed mismatch prefix_chunks={prefix_chunks} threads_total={threads_total}"
          );
          assert_eq!(
            got_derive, serial_derive,
            "parallel derive mismatch prefix_chunks={prefix_chunks} threads_total={threads_total}"
          );
        }
      }
    }

    #[test]
    fn subtree_roots_parallel_matches_serial_matrix() {
      let Ok(ap) = std::thread::available_parallelism() else {
        return;
      };
      if ap.get() <= 1 {
        return;
      }

      let kernels = {
        let kd = dispatch::kernel_dispatch();
        [kd.xs, kd.l]
      };

      let modes = [
        ("regular", IV, 0u32),
        ("keyed", words8_from_le_bytes_32(KEY), KEYED_HASH),
      ];

      let mut thread_caps = [2usize, 8];
      for t in &mut thread_caps {
        *t = (*t).min(ap.get()).max(2);
      }

      // Total chunks chosen to keep runtime reasonable while still exercising
      // lots of (roots_len, subtree_chunks) decompositions.
      let total_chunks_cases = [64usize, 1024]; // 64 KiB and 1 MiB

      for kernel in kernels {
        for (mode_name, key_words, flags) in modes {
          for total_chunks in total_chunks_cases {
            let total_bytes = total_chunks * CHUNK_LEN;
            let input = input_pattern(total_bytes);

            // `subtree_chunks` is a power-of-two divisor of `total_chunks`.
            let mut subtree_chunks = 1usize;
            while subtree_chunks <= total_chunks {
              debug_assert!(subtree_chunks.is_power_of_two());
              debug_assert_eq!(total_chunks % subtree_chunks, 0);
              let roots_len = total_chunks / subtree_chunks;
              debug_assert!(roots_len.is_power_of_two());

              let mut serial = alloc::vec![[0u32; 8]; roots_len];
              let mut parallel = alloc::vec![[0u32; 8]; roots_len];

              let wrap_base = {
                let total = total_chunks as u64;
                let mut base = u64::MAX.wrapping_sub(total.wrapping_sub(1));
                let align = subtree_chunks as u64;
                if align > 1 {
                  base &= !(align - 1);
                }
                base
              };

              let base_counters = [0u64, (subtree_chunks as u64).wrapping_mul(7), wrap_base];

              for &base_counter in &base_counters {
                hash_power_of_two_subtree_roots_serial(
                  kernel,
                  key_words,
                  flags,
                  base_counter,
                  &input,
                  subtree_chunks,
                  &mut serial,
                );

                for &threads_total in &thread_caps {
                  parallel.fill([0u32; 8]);
                  hash_power_of_two_subtree_roots_parallel_rayon(SubtreeRootsRequest {
                    kernel,
                    key_words,
                    flags,
                    base_counter,
                    input: &input,
                    subtree_chunks,
                    out: &mut parallel,
                    threads_total,
                  });

                  assert_eq!(
                    parallel, serial,
                    "subtree roots mismatch mode={mode_name} total_chunks={total_chunks} \
                     subtree_chunks={subtree_chunks} roots_len={roots_len} base_counter={base_counter} \
                     threads_total={threads_total}"
                  );
                }
              }

              subtree_chunks *= 2;
            }
          }
        }
      }
    }

    #[test]
    fn parent_fold_parallel_matches_serial_matrix() {
      let Ok(ap) = std::thread::available_parallelism() else {
        return;
      };
      if ap.get() <= 1 {
        return;
      }

      let kernels = {
        let kd = dispatch::kernel_dispatch();
        [kd.xs, kd.l]
      };

      let mut thread_caps = [2usize, 8];
      for t in &mut thread_caps {
        *t = (*t).min(ap.get()).max(2);
      }

      // Keep cases small-ish; we mainly want to shake out parent-fold correctness.
      let chunk_counts = [2usize, 8, 64, 1024];

      for kernel in kernels {
        for &chunks in &chunk_counts {
          debug_assert!(chunks.is_power_of_two());

          let input = input_pattern(chunks * CHUNK_LEN);
          let mut cvs = alloc::vec![[0u32; 8]; chunks];

          let base_counters = [
            0u64,
            (chunks as u64).wrapping_mul(3),
            u64::MAX.wrapping_sub(chunks as u64),
          ];
          for &base_counter in &base_counters {
            hash_full_chunks_cvs_serial(kernel, IV, 0, base_counter, &input, &mut cvs);
            let serial = reduce_power_of_two_chunk_cvs_any(kernel, IV, 0, &cvs, 1);

            for &threads_total in &thread_caps {
              let parallel = reduce_power_of_two_chunk_cvs_any(kernel, IV, 0, &cvs, threads_total);
              assert_eq!(
                parallel, serial,
                "parent fold mismatch chunks={chunks} base_counter={base_counter} threads_total={threads_total}"
              );
            }
          }
        }
      }
    }

    #[test]
    fn parallel_fallback_on_forced_panic_is_correct() {
      let Ok(ap) = std::thread::available_parallelism() else {
        return;
      };
      if ap.get() <= 1 {
        return;
      }

      let input = input_pattern(1024 * CHUNK_LEN);
      let expected = {
        let _serial_policy = tune::override_blake3_parallel_policy(never_parallel_table());
        Blake3::digest(&input)
      };

      let _force = force_parallel_panic(true);
      let _parallel_policy = tune::override_blake3_parallel_policy(always_parallel_table(ap.get() as u8));
      let got = Blake3::digest(&input);
      assert_eq!(got, expected, "forced-panic fallback digest mismatch");
    }
  }
}
