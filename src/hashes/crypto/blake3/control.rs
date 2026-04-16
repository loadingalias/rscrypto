#[cfg(feature = "std")]
use alloc::string::{String, ToString};
#[cfg(feature = "std")]
use core::cell::RefCell;
#[cfg(feature = "parallel")]
use std::sync::OnceLock;
#[cfg(feature = "parallel")]
use std::thread;

#[cfg(feature = "parallel")]
use super::dispatch_tables;
use super::{DERIVE_KEY_CONTEXT, DERIVE_KEY_MATERIAL, IV, KEYED_HASH, digest_oneshot_words, dispatch};
#[cfg(feature = "std")]
use super::{KEY_LEN, words8_from_le_bytes_32};

#[cfg(feature = "std")]
thread_local! {
  static DERIVE_KEY_CONTEXT_LOCAL_CACHE: RefCell<Option<(String, [u32; 8])>> = const { RefCell::new(None) };
  static KEYED_WORDS_LOCAL_CACHE: RefCell<Option<([u8; KEY_LEN], [u32; 8])>> = const { RefCell::new(None) };
}

#[cfg(feature = "std")]
#[inline]
fn compute_derive_context_key_words(context: &str) -> [u32; 8] {
  let context_bytes = context.as_bytes();
  let kernel_ctx = dispatch::hasher_dispatch().size_class_kernel(context_bytes.len());
  digest_oneshot_words(kernel_ctx, IV, DERIVE_KEY_CONTEXT, context_bytes)
}

#[inline]
#[cfg_attr(feature = "std", allow(dead_code))]
pub(super) fn derive_context_key_words(context: &str) -> [u32; 8] {
  let context_bytes = context.as_bytes();
  let kernel_ctx = dispatch::hasher_dispatch().size_class_kernel(context_bytes.len());
  digest_oneshot_words(kernel_ctx, IV, DERIVE_KEY_CONTEXT, context_bytes)
}

#[cfg(feature = "std")]
#[inline]
pub(super) fn derive_context_key_words_cached(context: &str) -> [u32; 8] {
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
pub(super) fn keyed_words_cached(key: &[u8; KEY_LEN]) -> [u32; 8] {
  if let Some(words) = KEYED_WORDS_LOCAL_CACHE.with(|slot| {
    let borrowed = slot.borrow();
    borrowed.as_ref().and_then(|(cached_key, cached_words)| {
      crate::traits::ct::constant_time_eq(cached_key, key).then_some(*cached_words)
    })
  }) {
    return words;
  }

  let words = words8_from_le_bytes_32(key);
  KEYED_WORDS_LOCAL_CACHE.with(|slot| {
    *slot.borrow_mut() = Some((*key, words));
  });
  words
}

#[inline]
pub(super) fn policy_kind_from_flags(flags: u32, is_xof: bool) -> ParallelPolicyKind {
  match (is_xof, flags) {
    (false, 0) => ParallelPolicyKind::Oneshot,
    (false, KEYED_HASH) => ParallelPolicyKind::KeyedOneshot,
    (false, DERIVE_KEY_MATERIAL) => ParallelPolicyKind::DeriveOneshot,
    (true, 0) => ParallelPolicyKind::Xof,
    (true, KEYED_HASH) => ParallelPolicyKind::KeyedXof,
    (true, DERIVE_KEY_MATERIAL) => ParallelPolicyKind::DeriveXof,
    _ => ParallelPolicyKind::Oneshot,
  }
}

#[cfg(feature = "std")]
pub(super) fn clear_keyed_words_cache() {
  KEYED_WORDS_LOCAL_CACHE.with(|slot| {
    if let Some((ref mut key_bytes, ref mut words)) = *slot.borrow_mut() {
      crate::traits::ct::zeroize(key_bytes);
      for w in words.iter_mut() {
        // SAFETY: `w` is a valid, aligned, dereferenceable pointer to initialized memory.
        unsafe { core::ptr::write_volatile(w, 0) };
      }
    }
    *slot.borrow_mut() = None;
  });
}

#[derive(Clone, Copy)]
pub(super) enum ParallelPolicyKind {
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ParallelAdmissionDecision {
  pub would_parallelize: bool,
  pub threads: usize,
}

#[cfg(feature = "parallel")]
static AVAILABLE_PARALLELISM: OnceLock<Option<usize>> = OnceLock::new();

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
pub(super) fn streaming_policy_kind_from_flags(flags: u32) -> ParallelPolicyKind {
  match flags {
    0 => ParallelPolicyKind::Update,
    KEYED_HASH => ParallelPolicyKind::KeyedUpdate,
    DERIVE_KEY_MATERIAL => ParallelPolicyKind::DeriveUpdate,
    _ => ParallelPolicyKind::Update,
  }
}

#[cfg(feature = "parallel")]
#[inline]
pub(super) fn resolved_parallel_policy(mode: ParallelPolicyKind) -> dispatch_tables::ParallelTable {
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
}

#[cfg(feature = "parallel")]
#[inline]
pub(super) fn parallel_policy_threads_with_admission(
  mode: ParallelPolicyKind,
  input_bytes: usize,
  admission_full_chunks: usize,
  commit_full_chunks: usize,
) -> Option<usize> {
  let decision = parallel_admission_decision(
    mode,
    resolved_parallel_policy(mode),
    input_bytes,
    admission_full_chunks,
    commit_full_chunks,
  );
  decision.would_parallelize.then_some(decision.threads)
}

#[cfg(feature = "parallel")]
#[inline]
pub(super) fn streaming_parallel_threads_for_flags(
  flags: u32,
  input_bytes: usize,
  admission_full_chunks: usize,
  commit_full_chunks: usize,
) -> Option<usize> {
  let mode = streaming_policy_kind_from_flags(flags);
  let decision = parallel_admission_decision(
    mode,
    resolved_parallel_policy(mode),
    input_bytes,
    admission_full_chunks,
    commit_full_chunks,
  );
  decision.would_parallelize.then_some(decision.threads)
}

#[cfg(feature = "parallel")]
#[inline]
#[allow(clippy::manual_saturating_arithmetic)] // Explicit clamp semantics are preferred here.
fn clamp_add_usize(lhs: usize, rhs: usize) -> usize {
  lhs.checked_add(rhs).unwrap_or(usize::MAX)
}

#[cfg(feature = "parallel")]
#[inline]
#[allow(clippy::manual_saturating_arithmetic)] // Explicit clamp semantics are preferred here.
fn clamp_mul_usize(lhs: usize, rhs: usize) -> usize {
  lhs.checked_mul(rhs).unwrap_or(usize::MAX)
}

#[cfg(feature = "parallel")]
#[inline]
fn ceil_div_usize(value: usize, divisor: usize) -> usize {
  let adjusted = clamp_add_usize(value, divisor.strict_sub(1));
  adjusted / divisor
}

#[cfg(feature = "parallel")]
#[inline]
pub(super) fn parallel_admission_decision(
  mode: ParallelPolicyKind,
  policy: dispatch_tables::ParallelTable,
  input_bytes: usize,
  admission_full_chunks: usize,
  commit_full_chunks: usize,
) -> ParallelAdmissionDecision {
  if policy.max_threads == 1 {
    return ParallelAdmissionDecision {
      would_parallelize: false,
      threads: 1,
    };
  }

  if input_bytes < policy.min_bytes || admission_full_chunks < policy.min_chunks {
    return ParallelAdmissionDecision {
      would_parallelize: false,
      threads: 1,
    };
  }

  let Some(mut threads) = available_parallelism_cached() else {
    return ParallelAdmissionDecision {
      would_parallelize: false,
      threads: 1,
    };
  };
  if policy.max_threads != 0 {
    threads = threads.min(policy.max_threads as usize);
  }

  if threads <= 1 {
    return ParallelAdmissionDecision {
      would_parallelize: false,
      threads: 1,
    };
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
    let merge_cost = ceil_div_usize(policy.merge_cost_bytes, merge_divisor);
    let spawn_cost = clamp_mul_usize(policy.spawn_cost_bytes, candidate.strict_sub(1));
    let work_cost = clamp_mul_usize(bytes_per_core, candidate);
    let fitted_required = clamp_add_usize(clamp_add_usize(merge_cost, spawn_cost), work_cost);
    let required = scale_parallel_required_bytes(mode, fitted_required, input_bytes);
    if input_bytes >= required {
      return ParallelAdmissionDecision {
        would_parallelize: true,
        threads: candidate,
      };
    }
    candidate -= 1;
  }
  ParallelAdmissionDecision {
    would_parallelize: false,
    threads: 1,
  }
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
  ceil_div_usize(clamp_mul_usize(required, num), den)
}
