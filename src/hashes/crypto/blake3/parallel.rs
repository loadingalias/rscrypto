#![cfg(feature = "parallel")]

use core::mem::MaybeUninit;

use super::{
  BLOCK_LEN, Blake3, CHUNK_LEN, CV_STACK_LEN, ChunkState, CvBytes, OUT_LEN, OutputState, control,
  hash_full_chunks_cvs_scoped, hash_one_full_chunk_cv, hash_power_of_two_subtree_roots_parallel_rayon, join, kernels,
  pow2_floor, reduce_power_of_two_chunk_cvs_any, single_chunk_output, words8_to_le_bytes, words16_from_le_bytes_64,
};
use crate::hashes::crypto::blake3::kernels::Kernel;

struct ParallelBatchSpec<'a> {
  input: &'a [u8],
  base_counter: u64,
  batch_chunks: usize,
  commit_chunks: usize,
  threads: usize,
  keep_last_full_chunk: bool,
}

#[derive(Clone, Default)]
pub(super) struct ParallelBatchScratch {
  roots: alloc::vec::Vec<[u32; 8]>,
  leaf_cvs: alloc::vec::Vec<[u32; 8]>,
}

impl ParallelBatchScratch {
  #[inline]
  fn clear(&mut self) {
    self.roots.clear();
    self.leaf_cvs.clear();
  }
}

impl Blake3 {
  #[inline]
  fn streaming_parallel_threads(
    &self,
    input_bytes: usize,
    admission_full_chunks: usize,
    commit_full_chunks: usize,
  ) -> Option<usize> {
    control::streaming_parallel_threads_for_flags(
      self.chunk_state.flags,
      input_bytes,
      admission_full_chunks,
      commit_full_chunks,
    )
  }

  #[cold]
  #[inline(never)]
  fn commit_parallel_batch(&mut self, batch: ParallelBatchSpec<'_>, scratch: &mut ParallelBatchScratch) {
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

    let batch_input = batch.input;
    let base_counter = batch.base_counter;
    let commit = batch.commit_chunks;
    let threads = batch.threads;

    let mut stack_len = self.cv_stack_len as usize;
    let mut counter = base_counter;
    let mut offset_chunks = 0usize;
    let mut remaining_commit = commit;

    scratch.clear();

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
      let bulk_kernel = kernels::kernel(self.bulk_kernel_id);

      let subtree_cv = if size >= threads && (counter == 0 || (counter & (size as u64 - 1)) == 0) {
        const MAX_SUBTREE_CHUNKS: usize = 1 << 12;

        let mut subtree_chunks = size / threads;
        subtree_chunks = subtree_chunks.max(1);
        subtree_chunks = pow2_floor(subtree_chunks);
        subtree_chunks = subtree_chunks.min(MAX_SUBTREE_CHUNKS).min(size);

        let roots_len = size / subtree_chunks;
        debug_assert!(roots_len.is_power_of_two());
        debug_assert_eq!(roots_len * subtree_chunks, size);

        scratch.roots.resize(roots_len, [0u32; 8]);
        hash_power_of_two_subtree_roots_parallel_rayon(super::SubtreeRootsRequest {
          kernel: bulk_kernel,
          key_words: self.key_words,
          flags: self.chunk_state.flags,
          base_counter: counter,
          input: subtree_input,
          subtree_chunks,
          out: &mut scratch.roots,
          threads_total: threads,
        });

        reduce_power_of_two_chunk_cvs_any(
          bulk_kernel,
          self.key_words,
          self.chunk_state.flags,
          &scratch.roots,
          threads,
        )
      } else {
        scratch.leaf_cvs.resize(size, [0u32; 8]);
        hash_full_chunks_cvs_scoped(
          bulk_kernel,
          self.key_words,
          self.chunk_state.flags,
          counter,
          subtree_input,
          &mut scratch.leaf_cvs,
          threads,
        );
        reduce_power_of_two_chunk_cvs_any(
          bulk_kernel,
          self.key_words,
          self.chunk_state.flags,
          &scratch.leaf_cvs,
          threads,
        )
      };

      counter = counter.wrapping_add(size as u64);
      let level = size.trailing_zeros();
      let mut total = counter >> level;
      let mut cv = subtree_cv;
      while total & 1 == 0 {
        cv = kernels::parent_cv_inline(
          self.bulk_kernel_id,
          pop_stack(&mut self.cv_stack, &mut stack_len),
          cv,
          self.key_words,
          self.chunk_state.flags,
        );
        total >>= 1;
      }
      push_stack(&mut self.cv_stack, &mut stack_len, cv);

      offset_chunks = offset_chunks.strict_add(size);
      remaining_commit -= size;
    }

    self.cv_stack_len = stack_len as u8;

    let new_counter = base_counter.strict_add(batch.batch_chunks as u64);
    self.chunk_state = ChunkState::new(
      self.key_words,
      new_counter,
      self.chunk_state.flags,
      self.chunk_state.kernel_id,
    );
    if batch.keep_last_full_chunk {
      let last_chunk_idx = batch.batch_chunks.strict_sub(1);
      let last_start = last_chunk_idx.strict_mul(CHUNK_LEN);
      let last_end = batch.batch_chunks.strict_mul(CHUNK_LEN);
      let last = &batch_input[last_start..last_end];
      self.pending_chunk_cv = Some(hash_one_full_chunk_cv(
        kernels::kernel(self.bulk_kernel_id),
        self.key_words,
        self.chunk_state.flags,
        new_counter.strict_sub(1),
        last,
      ));
      self.pending_cv_chunks = 1;
    }
  }

  #[cold]
  #[inline(never)]
  pub(super) fn try_parallel_update_batch(&mut self, input: &[u8]) -> Option<usize> {
    const MAX_PASS_CHUNKS: usize = 1 << 16;

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

    let mut scratch = core::mem::take(&mut self.parallel_batch_scratch);
    self.commit_parallel_batch(
      ParallelBatchSpec {
        input: &input[..bytes],
        base_counter,
        batch_chunks: batch,
        commit_chunks: commit,
        threads,
        keep_last_full_chunk,
      },
      &mut scratch,
    );
    self.parallel_batch_scratch = scratch;
    Some(bytes)
  }
}

#[inline]
fn left_subtree_len_bytes(input_len: usize) -> usize {
  debug_assert!(input_len > CHUNK_LEN);
  let full_chunks = input_len / CHUNK_LEN;
  let has_partial = !input_len.is_multiple_of(CHUNK_LEN);
  let total_chunks = full_chunks.strict_add(usize::from(has_partial));
  debug_assert!(total_chunks >= 2);

  let left_chunks = if total_chunks.is_power_of_two() {
    total_chunks / 2
  } else {
    pow2_floor(total_chunks)
  };
  debug_assert!(left_chunks >= 1 && left_chunks < total_chunks);
  left_chunks.strict_mul(CHUNK_LEN)
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
    pairs.strict_add(1)
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
  let max_leaf_bytes = simd_degree.strict_mul(CHUNK_LEN);
  if input.len() <= max_leaf_bytes {
    let chunks_exact = input.chunks_exact(CHUNK_LEN);
    let full_chunks = chunks_exact.len();
    debug_assert!(full_chunks <= simd_degree);
    debug_assert!(out.len() >= full_chunks.strict_add(usize::from(!chunks_exact.remainder().is_empty())));

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
      let cv_words = single_chunk_output(
        kernel,
        key_words,
        chunk_counter.strict_add(full_chunks as u64),
        flags,
        rem,
      )
      .chaining_value();
      out[out_len] = words8_to_le_bytes(&cv_words);
      out_len = out_len.strict_add(1);
    }

    return out_len;
  }

  debug_assert!(out.len() >= simd_degree.max(2));
  let left_len = left_subtree_len_bytes(input.len());
  let (left, right) = input.split_at(left_len);
  let right_chunk_counter = chunk_counter.strict_add((left.len() / CHUNK_LEN) as u64);

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

  let num_children = left_n.strict_add(right_n);
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

#[inline]
pub(super) fn root_output_oneshot_join_parallel(
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
    flags: super::PARENT | flags,
  }
}
