//! BLAKE3 (hash + XOF).
//!
//! This is a portable, dependency-free implementation suitable for `no_std`.

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![allow(clippy::indexing_slicing)] // Audited fixed-size parsing + perf-critical inner loops.

#[cfg(any(feature = "parallel", not(target_endian = "little")))]
use core::slice;
use core::{cmp::min, mem::MaybeUninit, ptr};
#[cfg(feature = "std")]
extern crate alloc;
#[cfg(feature = "std")]
use core::cell::RefCell;
#[cfg(all(feature = "std", test, feature = "parallel"))]
use core::{
  panic::AssertUnwindSafe,
  sync::atomic::{AtomicBool, Ordering},
};
#[cfg(all(feature = "std", test, feature = "parallel"))]
use std::panic;
#[cfg(feature = "std")]
use std::thread_local;

use crate::traits::{Digest, Xof};

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;
#[cfg(any(test, feature = "std"))]
mod bench;
mod control;
#[doc(hidden)]
pub mod dispatch;
#[doc(hidden)]
pub mod dispatch_tables;
pub(crate) mod kernels;
#[cfg(feature = "parallel")]
mod parallel;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64;
#[cfg(feature = "std")]
pub use self::bench::__bench;
use self::{control::ParallelPolicyKind, kernels::Kernel};

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

#[cfg(any(feature = "parallel", not(target_endian = "little")))]
type CvBytes = [u8; OUT_LEN];

#[cfg(feature = "parallel")]
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
thread_local! {
  static BLAKE3_SUBTREE_SCRATCH0: RefCell<alloc::vec::Vec<[u32; 8]>> = const { RefCell::new(alloc::vec::Vec::new()) };
  static BLAKE3_SUBTREE_SCRATCH1: RefCell<alloc::vec::Vec<[u32; 8]>> = const { RefCell::new(alloc::vec::Vec::new()) };
}

#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);

// SAFETY: Callers must ensure no two tasks access overlapping ranges through the pointer.
#[cfg(feature = "parallel")]
unsafe impl<T> Send for SendPtr<T> {}
#[cfg(feature = "parallel")]
// SAFETY: Callers must ensure no two tasks access overlapping ranges through the pointer.
unsafe impl<T> Sync for SendPtr<T> {}

#[cfg(feature = "parallel")]
impl<T> SendPtr<T> {
  #[inline]
  #[must_use]
  fn get(self) -> *mut T {
    self.0
  }
}

#[cfg(feature = "parallel")]
#[inline]
#[must_use]
fn thread_range(thread_index: usize, threads_total: usize, total: usize) -> (usize, usize) {
  debug_assert!(thread_index < threads_total);
  let start = thread_index.strict_mul(total).strict_div(threads_total);
  let end = thread_index.strict_add(1).strict_mul(total).strict_div(threads_total);
  (start, end)
}

#[cfg(all(feature = "std", test, feature = "parallel"))]
static FORCE_PARALLEL_PANIC: AtomicBool = AtomicBool::new(false);

#[cfg(all(feature = "std", test, feature = "parallel"))]
struct ForceParallelPanicGuard {
  prev: bool,
}

#[cfg(all(feature = "std", test, feature = "parallel"))]
impl Drop for ForceParallelPanicGuard {
  fn drop(&mut self) {
    FORCE_PARALLEL_PANIC.store(self.prev, Ordering::Relaxed);
  }
}

#[cfg(all(feature = "std", test, feature = "parallel"))]
#[inline]
#[must_use]
fn force_parallel_panic(enabled: bool) -> ForceParallelPanicGuard {
  let prev = FORCE_PARALLEL_PANIC.swap(enabled, Ordering::Relaxed);
  ForceParallelPanicGuard { prev }
}

#[cfg(all(feature = "std", test, feature = "parallel"))]
#[inline]
fn maybe_force_parallel_panic() {
  if FORCE_PARALLEL_PANIC.load(Ordering::Relaxed) {
    panic!("forced parallel panic (test-only)");
  }
}

#[cfg(feature = "parallel")]
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

#[cfg(feature = "parallel")]
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

#[cfg(feature = "parallel")]
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

#[cfg(feature = "parallel")]
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

#[cfg(feature = "parallel")]
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

#[cfg(feature = "parallel")]
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

#[cfg(feature = "parallel")]
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

#[cfg(feature = "parallel")]
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

#[cfg(feature = "parallel")]
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

#[cfg(any(target_endian = "little", feature = "parallel"))]
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

/// Reduce a power-of-2 slice of CVs down to exactly 2 using SIMD parent
/// compression.  The returned `(left, right)` pair represents the left and
/// right halves of the subtree.  The final parent compression (which may need
/// the ROOT flag) is deferred to the caller.
#[cfg(target_endian = "little")]
#[inline]
fn reduce_subtree_to_pair(
  kernel: Kernel,
  key_words: [u32; 8],
  flags: u32,
  cvs: &mut [[u32; 8]],
) -> ([u32; 8], [u32; 8]) {
  let n = cvs.len();
  debug_assert!(n >= 2);
  debug_assert!(n.is_power_of_two());

  if n == 2 {
    return (cvs[0], cvs[1]);
  }

  // Ping-pong reduction: cvs <-> next, halving each iteration.
  // `cvs` is reused as one buffer; `next` is stack-allocated.
  const HALF_MAX: usize = 64;
  let mut next = [[0u32; 8]; HALF_MAX];
  let mut cur_len = n;
  let mut in_next = false; // false = current data is in cvs, true = in next

  while cur_len > 2 {
    let pairs = cur_len / 2;
    if !in_next {
      kernels::parent_cvs_many_from_cvs_inline(kernel.id, &cvs[..cur_len], key_words, flags, &mut next[..pairs]);
    } else {
      kernels::parent_cvs_many_from_cvs_inline(kernel.id, &next[..cur_len], key_words, flags, &mut cvs[..pairs]);
    }
    in_next = !in_next;
    cur_len = pairs;
  }

  if !in_next { (cvs[0], cvs[1]) } else { (next[0], next[1]) }
}

#[cfg(feature = "parallel")]
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

    // Helper: fold one reduction level, parallel (rayon) or serial.
    macro_rules! fold_level {
      ($children:expr, $out:expr) => {
        #[cfg(feature = "parallel")]
        if threads_total > 1 && pairs >= MIN_PAIRS_FOR_PARALLEL {
          parent_cvs_many_from_cvs_parallel_rayon(kernel, key_words, flags, $children, $out, threads_total);
        } else {
          kernels::parent_cvs_many_from_cvs_inline(kernel.id, $children, key_words, flags, $out);
        }
        #[cfg(not(feature = "parallel"))]
        kernels::parent_cvs_many_from_cvs_inline(kernel.id, $children, key_words, flags, $out);
      };
    }

    let out0: [u32; 8] = match cur {
      Cur::Input(children) => {
        let out = &mut buf0[..pairs];
        let children = &children[..2 * pairs];
        fold_level!(children, out);
        cur = Cur::Buf0;
        out[0]
      }
      Cur::Buf0 => {
        let out = &mut buf1[..pairs];
        let children = &buf0[..2 * pairs];
        fold_level!(children, out);
        cur = Cur::Buf1;
        out[0]
      }
      Cur::Buf1 => {
        let out = &mut buf0[..pairs];
        let children = &buf1[..2 * pairs];
        fold_level!(children, out);
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
    debug_assert!(!input.is_empty(), "caller checks for empty input");

    // Fast path A: buffer input into an empty block at any blocks_compressed.
    if self.block_len == 0 {
      if input.len() <= BLOCK_LEN {
        self.block[..input.len()].copy_from_slice(input);
        self.block_len = input.len() as u8;
        return;
      }

      // Exact full-chunk fast path (only pristine state).
      if self.blocks_compressed == 0 && input.len() == CHUNK_LEN {
        if should_use_exact_one_chunk_fast_path(self.kernel_id) {
          self.absorb_exact_one_chunk(input);
          return;
        }

        self.absorb_exact_one_chunk_fallback(input);
        return;
      }
    }

    // Fast path B: compress full buffered block, buffer the incoming one.
    // Calls assembly directly, bypassing the multi-block wrappers and their
    // #[target_feature] boundaries. This matches the official blake3 crate's
    // call depth: one match + one extern "C" assembly call.
    if self.block_len as usize == BLOCK_LEN && self.blocks_compressed < 15 && input.len() == BLOCK_LEN {
      let start = if self.blocks_compressed == 0 { CHUNK_START } else { 0 };
      // SAFETY: kernel_id was validated at construction to match available CPU features.
      unsafe {
        kernels::compress_block_asm_inline(
          self.kernel_id,
          &mut self.chaining_value,
          &self.block,
          self.chunk_counter,
          self.flags | start,
        );
      }
      self.blocks_compressed = self.blocks_compressed.wrapping_add(1);
      self.block.copy_from_slice(input);
      // block_len stays BLOCK_LEN — no zero-fill needed
      return;
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
        let start = if self.blocks_compressed == 0 { CHUNK_START } else { 0 };
        // SAFETY: kernel_id was validated at construction to match available CPU features.
        unsafe {
          kernels::compress_block_asm_inline(
            self.kernel_id,
            &mut self.chaining_value,
            &self.block,
            self.chunk_counter,
            self.flags | start,
          );
        }
        self.blocks_compressed = self.blocks_compressed.wrapping_add(1);
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
          let bytes = blocks_to_compress.strict_mul(BLOCK_LEN);
          let (block_slices, _) = input[..bytes].as_chunks::<BLOCK_LEN>();
          for block_bytes in block_slices {
            let start = if self.blocks_compressed == 0 { CHUNK_START } else { 0 };
            // SAFETY: kernel_id was validated at construction to match available CPU features.
            unsafe {
              kernels::compress_block_asm_inline(
                self.kernel_id,
                &mut self.chaining_value,
                block_bytes,
                self.chunk_counter,
                self.flags | start,
              );
            }
            self.blocks_compressed = self.blocks_compressed.wrapping_add(1);
          }
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
    if let Some(threads) =
      control::parallel_policy_threads_with_admission(mode, input.len(), full_chunks, commit_full_chunks)
    {
      return parallel::root_output_oneshot_join_parallel(kernel, key_words, flags, input, threads);
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
        let remaining = (full_chunks as u64).strict_sub(chunk_counter) as usize;
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
        if remainder == 0 && chunk_counter.strict_add(batch as u64) == full_chunks as u64 {
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

        chunk_counter = chunk_counter.strict_add(batch as u64);
        offset = offset.strict_add(batch.strict_mul(CHUNK_LEN));
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
        let remaining = (full_chunks as u64).strict_sub(chunk_counter) as usize;
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
        if remainder == 0 && chunk_counter.strict_add(batch as u64) == full_chunks as u64 {
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

        chunk_counter = chunk_counter.strict_add(batch as u64);
        offset = offset.strict_add(batch.strict_mul(CHUNK_LEN));
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

#[cfg(feature = "parallel")]
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

#[cfg(feature = "parallel")]
#[inline]
fn hash_full_chunks_cvs_scoped(
  kernel: Kernel,
  key_words: [u32; 8],
  flags: u32,
  base_counter: u64,
  input: &[u8],
  out: &mut [[u32; 8]],
  threads_total: usize,
) {
  let out_len = out.len();
  debug_assert_eq!(input.len(), out_len * CHUNK_LEN);

  #[cfg(feature = "parallel")]
  {
    let mut threads_total = threads_total;
    if threads_total <= 1 || out_len < 2 {
      hash_full_chunks_cvs_parallel_rayon(kernel, key_words, flags, base_counter, input, out, 1);
      return;
    }
    threads_total = threads_total.min(out_len);
    hash_full_chunks_cvs_parallel_rayon(kernel, key_words, flags, base_counter, input, out, threads_total);
  }

  #[cfg(not(feature = "parallel"))]
  {
    let _ = threads_total;
    hash_full_chunks_cvs_serial(kernel, key_words, flags, base_counter, input, out);
  }
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
  let mode = control::policy_kind_from_flags(flags, false);
  let output = root_output_oneshot(kernel, key_words, flags, mode, input);
  output.root_hash_words()
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
  dispatch: dispatch::HasherDispatch,
  bulk_kernel_id: kernels::Blake3KernelId,
  #[cfg(feature = "parallel")]
  parallel_batch_scratch: parallel::ParallelBatchScratch,
  chunk_state: ChunkState,
  pending_chunk_cv: Option<[u32; 8]>,
  /// Number of chunks the pending CV represents. 1 for a single-chunk pending
  /// (from the parallel path), or a power-of-2 > 1 for a subtree right-half
  /// pending (from the subtree update path). Only meaningful when
  /// `pending_chunk_cv` is `Some`.
  pending_cv_chunks: u64,
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
    let key_words = control::keyed_words_cached(key);
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
    if data.len() <= CHUNK_LEN {
      return xof_oneshot_single_chunk(kernel, key_words, KEYED_HASH, data);
    }
    Blake3Xof::from_output(root_output_oneshot(
      kernel,
      key_words,
      KEYED_HASH,
      control::policy_kind_from_flags(KEYED_HASH, true),
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
        control::derive_context_key_words_cached(context)
      }
      #[cfg(not(feature = "std"))]
      {
        control::derive_context_key_words(context)
      }
    };

    digest_public_oneshot(context_key_words, DERIVE_KEY_MATERIAL, key_material)
  }

  #[inline]
  fn new_internal(key_words: [u32; 8], flags: u32) -> Self {
    let dispatch = dispatch::hasher_dispatch();
    Self::new_internal_with_dispatch(key_words, flags, dispatch.stream_kernel().id, dispatch)
  }

  #[inline]
  fn new_internal_with(key_words: [u32; 8], flags: u32, kernel_id: kernels::Blake3KernelId) -> Self {
    Self::new_internal_with_dispatch(key_words, flags, kernel_id, dispatch::hasher_dispatch())
  }

  #[inline]
  fn new_internal_with_dispatch(
    key_words: [u32; 8],
    flags: u32,
    kernel_id: kernels::Blake3KernelId,
    dispatch: dispatch::HasherDispatch,
  ) -> Self {
    Self {
      dispatch,
      bulk_kernel_id: kernel_id,
      #[cfg(feature = "parallel")]
      parallel_batch_scratch: parallel::ParallelBatchScratch::default(),
      chunk_state: ChunkState::new(key_words, 0, flags, kernel_id),
      pending_chunk_cv: None,
      pending_cv_chunks: 0,
      key_words,
      cv_stack: uninit_cv_stack(),
      cv_stack_len: 0,
    }
  }

  /// Process the largest aligned power-of-2 subtree from the input using SIMD
  /// `hash_many`, then reduce the resulting CVs with SIMD parent compression
  /// and merge into the CV stack.
  ///
  /// This follows the canonical BLAKE3 streaming strategy: the subtree size is
  /// chosen so that `chunk_counter % subtree_chunks == 0`, preserving the
  /// correct Merkle tree shape.
  ///
  /// Returns the number of input bytes consumed, or `None` if the subtree
  /// path is not applicable.
  #[cfg(target_endian = "little")]
  fn try_subtree_update(&mut self, input: &[u8]) -> Option<usize> {
    if self.chunk_state.len() != 0 || self.bulk_kernel_id.simd_degree() <= 1 || input.len() <= CHUNK_LEN {
      return None;
    }

    let chunk_counter = self.chunk_state.chunk_counter;
    let full_chunks = input.len() / CHUNK_LEN;
    if full_chunks <= 1 {
      return None;
    }

    // Compute the largest power-of-2 subtree aligned with the chunk counter.
    // Alignment guarantees the canonical BLAKE3 tree shape: the subtree must
    // evenly divide the total number of chunks processed so far.
    let mut subtree_chunks = pow2_floor(full_chunks);
    while subtree_chunks > 1 && (subtree_chunks as u64).strict_sub(1) & chunk_counter != 0 {
      subtree_chunks /= 2;
    }
    if subtree_chunks <= 1 {
      return None;
    }

    // Cap at our stack buffer size. MAX_SUBTREE_CHUNKS is a power of 2 that
    // divides any larger aligned subtree, so alignment is preserved.
    const MAX_SUBTREE_CHUNKS: usize = 128;
    if subtree_chunks > MAX_SUBTREE_CHUNKS {
      subtree_chunks = MAX_SUBTREE_CHUNKS;
    }

    let subtree_len = subtree_chunks.strict_mul(CHUNK_LEN);
    let flags = self.chunk_state.flags;
    let kernel_id = self.chunk_state.kernel_id;
    let kernel = kernels::kernel(self.bulk_kernel_id);

    // Hash all chunks in the subtree into uninitialized scratch. The kernel
    // writes every committed CV slot, so pre-zeroing the whole buffer is waste.
    let mut cvs: [MaybeUninit<[u32; 8]>; MAX_SUBTREE_CHUNKS] =
      // SAFETY: `[MaybeUninit<T>; N]` may be left uninitialized because
      // `MaybeUninit<T>` permits any bit pattern.
      unsafe { MaybeUninit::<[MaybeUninit<[u32; 8]>; MAX_SUBTREE_CHUNKS]>::uninit().assume_init() };
    // SAFETY: `input` has at least `subtree_len` bytes, `cvs` has
    // `subtree_chunks` CV slots, and the kernel's CPU features are available.
    unsafe {
      kernels::hash_many_contiguous_inline(
        self.bulk_kernel_id,
        input.as_ptr(),
        subtree_chunks,
        &self.key_words,
        chunk_counter,
        flags,
        cvs.as_mut_ptr().cast::<u8>(),
      )
    };
    // SAFETY: `hash_many_contiguous_inline` wrote exactly `subtree_chunks` CVs.
    let cvs = unsafe { core::slice::from_raw_parts_mut(cvs.as_mut_ptr().cast::<[u32; 8]>(), subtree_chunks) };

    let keep_last = input.len().is_multiple_of(CHUNK_LEN) && subtree_chunks == full_chunks;

    if keep_last {
      // Terminal batch: the entire remaining input fits in this subtree.
      // Reduce to 2 CVs so the final parent compression can be deferred to
      // finalize() for ROOT flag application.
      let half = (subtree_chunks / 2) as u64;
      let (left_cv, right_cv) = reduce_subtree_to_pair(kernel, self.key_words, flags, cvs);

      // Push the left subtree onto the stack.
      self.add_subtree_cv(left_cv, half);

      // Advance counter for the pending right half and store it.
      let full_counter = self.chunk_state.chunk_counter.strict_add(half);
      self.chunk_state = ChunkState::new(self.key_words, full_counter, flags, kernel_id);
      self.pending_chunk_cv = Some(right_cv);
      self.pending_cv_chunks = half;
    } else {
      // Non-terminal: more input follows. Reduce to 1 CV and push.
      let (left_cv, right_cv) = reduce_subtree_to_pair(kernel, self.key_words, flags, cvs);
      let subtree_cv = kernels::parent_cv_inline(kernel.id, left_cv, right_cv, self.key_words, flags);
      self.add_subtree_cv(subtree_cv, subtree_chunks as u64);
    }

    Some(subtree_len)
  }

  #[cfg(not(target_endian = "little"))]
  fn try_simd_update_batch(&mut self, input: &[u8]) -> Option<usize> {
    if self.chunk_state.len() != 0 || self.bulk_kernel_id.simd_degree() <= 1 || input.len() <= CHUNK_LEN {
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
      kernels::hash_many_contiguous_inline(
        self.bulk_kernel_id,
        input.as_ptr(),
        batch,
        &self.key_words,
        base_counter,
        self.chunk_state.flags,
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
        kernels::kernel(self.bulk_kernel_id),
        &mut self.cv_stack,
        &mut stack_len,
        base_counter,
        cvs_bytes,
        self.key_words,
        self.chunk_state.flags,
      );
      self.cv_stack_len = stack_len as u8;
    }

    let new_counter = base_counter.strict_add(batch as u64);
    self.chunk_state = ChunkState::new(
      self.key_words,
      new_counter,
      self.chunk_state.flags,
      self.chunk_state.kernel_id,
    );
    if keep_last_full_chunk {
      let offset = batch.strict_sub(1).strict_mul(OUT_LEN);
      // SAFETY: `out_buf` is `OUT_LEN * MAX_SIMD_DEGREE`, and `offset`
      // is `(batch - 1) * OUT_LEN` with `batch <= MAX_SIMD_DEGREE`.
      let cv = unsafe { words8_from_le_bytes_32(&*(out_buf.as_ptr().add(offset) as *const [u8; OUT_LEN])) };
      self.pending_chunk_cv = Some(cv);
      self.pending_cv_chunks = 1;
    }

    Some(batch.strict_mul(CHUNK_LEN))
  }

  fn update_with(
    &mut self,
    mut input: &[u8],
    stream_kernel_id: kernels::Blake3KernelId,
    bulk_kernel_id: kernels::Blake3KernelId,
  ) {
    self.bulk_kernel_id = bulk_kernel_id;
    self.chunk_state.kernel_id = stream_kernel_id;

    // If the previous update ended exactly on a chunk boundary, we may have
    // stored the last full chunk's CV instead of keeping a fully-buffered
    // `ChunkState`. As soon as more input arrives, that chunk is no longer
    // terminal and can be committed to the tree.
    if !input.is_empty() {
      self.commit_pending_chunk_cv();
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
        self.advance_full_chunk();
      }

      #[cfg(feature = "parallel")]
      {
        if self.chunk_state.len() == 0
          && input.len() > CHUNK_LEN
          && let Some(consumed) = self.try_parallel_update_batch(input)
        {
          input = &input[consumed..];
          continue;
        }
      }

      #[cfg(target_endian = "little")]
      if let Some(consumed) = self.try_subtree_update(input) {
        input = &input[consumed..];
        continue;
      }

      #[cfg(not(target_endian = "little"))]
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
    let key_words = control::derive_context_key_words_cached(context);
    #[cfg(not(feature = "std"))]
    let key_words = control::derive_context_key_words(context);
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
      new_cv = kernels::parent_cv_inline(
        self.chunk_state.kernel_id,
        self.pop_stack(),
        new_cv,
        self.key_words,
        self.chunk_state.flags,
      );
      total_chunks >>= 1;
    }
    self.push_stack(new_cv);
  }

  /// Merge a CV that represents `num_chunks` chunks into the CV stack.
  ///
  /// The chunk counter must ALREADY include these chunks (i.e. the counter
  /// was advanced before calling this method). `num_chunks` must be a
  /// power of 2.
  ///
  /// For `num_chunks == 1` this is equivalent to `add_chunk_chaining_value`.
  /// For larger powers of 2 it correctly places the subtree CV at the right
  /// level in the Merkle tree, merging with existing stack entries as needed.
  fn merge_cv_into_stack(&mut self, mut cv: [u32; 8], num_chunks: u64) {
    debug_assert!(num_chunks.is_power_of_two());
    let level = num_chunks.trailing_zeros();
    let mut total = self.chunk_state.chunk_counter >> level;
    while total & 1 == 0 {
      cv = kernels::parent_cv_inline(
        self.bulk_kernel_id,
        self.pop_stack(),
        cv,
        self.key_words,
        self.chunk_state.flags,
      );
      total >>= 1;
    }
    self.push_stack(cv);
  }

  /// Push a reduced subtree CV onto the stack at the correct tree level,
  /// then advance the chunk counter.
  ///
  /// `subtree_chunks` must be a power of 2 and must be aligned with the
  /// current chunk counter (i.e. `chunk_counter % subtree_chunks == 0`).
  #[cfg(target_endian = "little")]
  fn add_subtree_cv(&mut self, cv: [u32; 8], subtree_chunks: u64) {
    debug_assert!(subtree_chunks.is_power_of_two());
    let new_counter = self.chunk_state.chunk_counter.strict_add(subtree_chunks);
    self.chunk_state = ChunkState::new(
      self.key_words,
      new_counter,
      self.chunk_state.flags,
      self.chunk_state.kernel_id,
    );
    self.merge_cv_into_stack(cv, subtree_chunks);
  }

  #[inline]
  fn commit_pending_chunk_cv(&mut self) {
    if let Some(cv) = self.pending_chunk_cv.take() {
      let chunks = self.pending_cv_chunks;
      self.pending_cv_chunks = 0;
      self.merge_cv_into_stack(cv, chunks);
    }
  }

  #[inline]
  fn advance_full_chunk(&mut self) {
    debug_assert_eq!(self.chunk_state.len(), CHUNK_LEN);
    let chunk_cv = self.chunk_state.output().chaining_value();
    let total_chunks = self.chunk_state.chunk_counter.strict_add(1);
    self.add_chunk_chaining_value(chunk_cv, total_chunks);
    self.chunk_state = ChunkState::new(
      self.key_words,
      total_chunks,
      self.chunk_state.flags,
      self.chunk_state.kernel_id,
    );
  }

  #[inline(never)]
  fn update_digest_slow(&mut self, input: &[u8]) {
    self.commit_pending_chunk_cv();

    if self.chunk_state.len() == CHUNK_LEN {
      self.advance_full_chunk();
    }

    if self.chunk_state.len() + input.len() <= CHUNK_LEN {
      self.chunk_state.update(input);
      return;
    }

    let stream = self.chunk_state.kernel_id;
    let bulk = self.dispatch.bulk_kernel_for_update(input.len()).id;
    self.update_with(input, stream, bulk);
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
      parent_output(
        self.chunk_state.kernel_id,
        left,
        right_cv,
        self.key_words,
        self.chunk_state.flags,
      )
    } else {
      self.chunk_state.output()
    };

    while parent_nodes_remaining > 0 {
      parent_nodes_remaining -= 1;
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
    if self.cv_stack_len == 0 && self.pending_chunk_cv.is_none() {
      return self.chunk_state.root_emit_state();
    }

    self.root_emit_state_slow()
  }

  #[inline(never)]
  fn root_emit_state_slow(&self) -> RootEmitState {
    let mut parent_nodes_remaining = self.cv_stack_len as usize;
    let mut current_cv = if let Some(right_cv) = self.pending_chunk_cv {
      debug_assert!(
        parent_nodes_remaining > 0,
        "pending full chunk implies multi-chunk input"
      );
      right_cv
    } else {
      self.chunk_state.output_chaining_value()
    };

    while parent_nodes_remaining > 0 {
      parent_nodes_remaining -= 1;
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
    if self.cv_stack_len == 0 && self.pending_chunk_cv.is_none() {
      return Blake3Xof::new(self.chunk_state.root_emit_state());
    }

    Blake3Xof::new(self.root_emit_state())
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
    for slot in self.cv_stack[..self.cv_stack_len as usize].iter_mut() {
      // SAFETY: cv_stack[..cv_stack_len] entries are initialized.
      let words = unsafe { slot.assume_init_mut() };
      for word in words.iter_mut() {
        // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
        unsafe { core::ptr::write_volatile(word, 0) };
      }
    }
    if let Some(ref mut cv) = self.pending_chunk_cv {
      for word in cv.iter_mut() {
        // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
        unsafe { core::ptr::write_volatile(word, 0) };
      }
    }
    #[cfg(feature = "std")]
    if self.chunk_state.flags & KEYED_HASH != 0 {
      control::clear_keyed_words_cache();
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

    // Fast path: input fits entirely within the current chunk.
    // Safe even when pending_chunk_cv is Some — the pending CV is for an
    // earlier chunk and will be committed when we cross the chunk boundary
    // in update_digest_slow().
    //
    // Uses subtraction instead of addition to avoid an overflow-checking
    // branch from strict_add. The cs_len < CHUNK_LEN guard proves the
    // strict_sub cannot underflow, letting LLVM eliminate the panic path.
    let cs_len = self.chunk_state.len();
    if cs_len < CHUNK_LEN && input.len() <= CHUNK_LEN.strict_sub(cs_len) {
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
    if self.chunk_state.chunk_counter == 0 && self.cv_stack_len == 0 && self.pending_chunk_cv.is_none() {
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
    let output = if self.chunk_state.chunk_counter == 0
      && self.chunk_state.len() == 0
      && self.cv_stack_len == 0
      && self.pending_chunk_cv.is_none()
    {
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

/// Lean XOF oneshot for single-chunk inputs (≤ 1024 bytes).
///
/// Constructs `Blake3Xof` directly without going through `root_output_oneshot`
/// or `single_chunk_output`. For inputs ≤ 64B, no compression is needed at all.
/// For 65–1024B, full blocks are compressed directly via the kernel's compress
/// function pointer, avoiding the indirect `chunk_compress_blocks` call.
#[inline]
fn xof_oneshot_single_chunk(kernel: Kernel, key_words: [u32; 8], flags: u32, input: &[u8]) -> Blake3Xof {
  debug_assert!(input.len() <= CHUNK_LEN);

  // Tiny input (≤ 64B): zero-pad into block_words, no compression needed.
  if input.len() <= BLOCK_LEN {
    let block_words = if cfg!(target_endian = "little") {
      let mut out = [0u32; 16];
      if !input.is_empty() {
        // SAFETY: `out` is 64 bytes, and `input.len() <= 64`.
        unsafe {
          ptr::copy_nonoverlapping(input.as_ptr(), out.as_mut_ptr().cast::<u8>(), input.len());
        }
      }
      out
    } else {
      let mut block = [0u8; BLOCK_LEN];
      block[..input.len()].copy_from_slice(input);
      words16_from_le_bytes_64(&block)
    };

    return Blake3Xof::new(RootEmitState {
      kernel_id: kernel.id,
      input_chaining_value: key_words,
      block_words,
      counter: 0,
      block_len: input.len() as u8,
      flags: (flags | CHUNK_START | CHUNK_END) as u8,
    });
  }

  // Single-chunk input (65–1024B): compress full blocks, store last block.
  let rem = input.len() % BLOCK_LEN;
  let (full_blocks, last_len) = if rem == 0 {
    (input.len() / BLOCK_LEN - 1, BLOCK_LEN)
  } else {
    (input.len() / BLOCK_LEN, rem)
  };

  let mut cv = key_words;

  // Compress full non-final blocks using the kernel's compress function pointer
  // directly (no indirect chunk_compress_blocks call, no kernel-ID match).
  for i in 0..full_blocks {
    let offset = i * BLOCK_LEN;
    let block_flags = flags | if i == 0 { CHUNK_START } else { 0 };
    // SAFETY: `offset + BLOCK_LEN <= input.len()` by construction.
    let block_words = unsafe { words16_from_le_bytes_64(&*input.as_ptr().add(offset).cast::<[u8; BLOCK_LEN]>()) };
    cv = first_8_words((kernel.compress)(&cv, &block_words, 0, BLOCK_LEN as u32, block_flags));
  }

  // Build the last block's words.
  let start = if full_blocks == 0 { CHUNK_START } else { 0 };
  let block_words = if cfg!(target_endian = "little") {
    let mut out = [0u32; 16];
    let offset = full_blocks * BLOCK_LEN;
    // SAFETY: `last_len <= BLOCK_LEN` and source range is in-bounds.
    unsafe {
      ptr::copy_nonoverlapping(input.as_ptr().add(offset), out.as_mut_ptr().cast::<u8>(), last_len);
    }
    out
  } else {
    let mut block = [0u8; BLOCK_LEN];
    let offset = full_blocks * BLOCK_LEN;
    block[..last_len].copy_from_slice(&input[offset..offset + last_len]);
    words16_from_le_bytes_64(&block)
  };

  Blake3Xof::new(RootEmitState {
    kernel_id: kernel.id,
    input_chaining_value: cv,
    block_words,
    counter: 0,
    block_len: last_len as u8,
    flags: (flags | start | CHUNK_END) as u8,
  })
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

  // Process full non-final blocks. Use the compress function pointer directly
  // instead of chunk_compress_blocks_inline to avoid the kernel-ID match
  // dispatch on every call — the Kernel struct already carries the resolved
  // function pointer.
  for i in 0..full_blocks {
    let offset = i * BLOCK_LEN;
    let block_flags = flags | if i == 0 { CHUNK_START } else { 0 };
    // SAFETY: `offset + BLOCK_LEN <= input.len()` by construction.
    let block_words = unsafe { words16_from_le_bytes_64(&*input.as_ptr().add(offset).cast::<[u8; BLOCK_LEN]>()) };
    cv = first_8_words((kernel.compress)(&cv, &block_words, 0, BLOCK_LEN as u32, block_flags));
  }

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
#[must_use]
fn use_avx512_four_block_avx2_fast_path() -> bool {
  dispatch::avx2_available() && dispatch::avx2_hash_many_one_chunk_fast_path()
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
  // On wide-pipeline CPUs (Zen 5, ICL) require ≥ 8 blocks (512 B): at 4 blocks
  // the hash_many SIMD setup cost dominates. On narrow-pipeline AMD (Zen 4) the
  // hash_many path is beneficial even at 4 blocks.
  #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
  if kernel.id == kernels::Blake3KernelId::X86Avx2
    && use_avx2_hash_many_one_chunk_fast_path()
    && !input.is_empty()
    && input.len().is_multiple_of(BLOCK_LEN)
    && (!dispatch::hash_many_wide_pipeline() || input.len() >= BLOCK_LEN.strict_mul(8))
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

  // Same platform-aware threshold as the AVX2 path above.
  #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
  if kernel.id == kernels::Blake3KernelId::X86Avx512
    && !input.is_empty()
    && input.len().is_multiple_of(BLOCK_LEN)
    && (!dispatch::hash_many_wide_pipeline() || input.len() >= BLOCK_LEN.strict_mul(8))
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
    if blocks == 4 && use_avx512_four_block_avx2_fast_path() {
      // For the exact 4-block one-chunk case (4096B input), AVX2's internal
      // tail cascade is a better fit than the AVX-512 4-lane sub-degree path.
      // Keep this heuristic narrow so 64..1024B stay on the proven AVX-512
      // exact-block path.
      // SAFETY: this branch is only enabled when AVX2 is available per
      // dispatch; input is one contiguous full-chunk-or-less buffer; output
      // points to one OUT_LEN digest lane.
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

  #[test]
  fn xof_two_32byte_reads_match_single_64byte_read() {
    for len in [0usize, 1, 64, 1024] {
      let input = input_pattern(len);

      let mut stepped_hasher = Blake3::new();
      stepped_hasher.update(&input);
      let mut stepped_xof = stepped_hasher.finalize_xof();
      let mut stepped = [0u8; 64];
      stepped_xof.squeeze(&mut stepped[..32]);
      stepped_xof.squeeze(&mut stepped[32..]);

      let mut single_hasher = Blake3::new();
      single_hasher.update(&input);
      let mut single_xof = single_hasher.finalize_xof();
      let mut single = [0u8; 64];
      single_xof.squeeze(&mut single);

      assert_eq!(stepped, single, "xof 32+32 mismatch len={len}");
    }
  }

  #[cfg(feature = "parallel")]
  mod parallel_pipeline {
    extern crate alloc;

    use super::{
      super::{
        IV, KEYED_HASH, SubtreeRootsRequest, force_parallel_panic, hash_full_chunks_cvs_serial,
        hash_power_of_two_subtree_roots_parallel_rayon, hash_power_of_two_subtree_roots_serial,
        reduce_power_of_two_chunk_cvs_any, words8_from_le_bytes_32,
      },
      Blake3, CONTEXT, KEY, OUT_LEN, input_pattern,
    };
    use crate::{
      hashes::crypto::blake3::{__bench, CHUNK_LEN, dispatch, dispatch_tables},
      traits::{Digest, Xof},
    };

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
            let _serial_policy = __bench::override_blake3_parallel_policy(never_parallel_table());
            run_case_streaming(Blake3::new, prefix, payload)
          };

          // Sanity: one-shot should match streaming in serial mode.
          {
            let _serial_policy = __bench::override_blake3_parallel_policy(never_parallel_table());
            assert_eq!(
              serial_regular.0,
              Blake3::digest(full_input),
              "serial oneshot mismatch prefix_chunks={prefix_chunks} payload_len={payload_len}"
            );
          }

          for &threads_total in &thread_caps {
            let threads_total = threads_total as u8;
            let _policy = __bench::override_blake3_parallel_policy(always_parallel_table(threads_total));

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
          let _serial_policy = __bench::override_blake3_parallel_policy(never_parallel_table());
          let keyed = run_case_streaming(|| Blake3::new_keyed(KEY), prefix, payload);
          let derive = run_case_streaming(|| Blake3::new_derive_key(CONTEXT), prefix, payload);
          (keyed, derive)
        };

        {
          let _serial_policy = __bench::override_blake3_parallel_policy(never_parallel_table());
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
          let _policy = __bench::override_blake3_parallel_policy(always_parallel_table(threads_total));

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

                #[cfg(feature = "parallel")]
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
        let _serial_policy = __bench::override_blake3_parallel_policy(never_parallel_table());
        Blake3::digest(&input)
      };

      let _force = force_parallel_panic(true);
      let _parallel_policy = __bench::override_blake3_parallel_policy(always_parallel_table(ap.get() as u8));
      let got = Blake3::digest(&input);
      assert_eq!(got, expected, "forced-panic fallback digest mismatch");
    }
  }
}
