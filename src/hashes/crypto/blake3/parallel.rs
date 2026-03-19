#![cfg(feature = "parallel")]

use core::mem::MaybeUninit;

use super::{
  Blake3, CHUNK_LEN, CV_STACK_LEN, ChunkState, control, hash_full_chunks_cvs_scoped, hash_one_full_chunk_cv,
  hash_power_of_two_subtree_roots_parallel_rayon, kernels, pow2_floor, reduce_power_of_two_chunk_cvs_any,
};

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

      offset_chunks += size;
      remaining_commit -= size;
    }

    self.cv_stack_len = stack_len as u8;

    let new_counter = base_counter + batch.batch_chunks as u64;
    self.chunk_state = ChunkState::new(
      self.key_words,
      new_counter,
      self.chunk_state.flags,
      self.chunk_state.kernel_id,
    );
    if batch.keep_last_full_chunk {
      let last_chunk_idx = batch.batch_chunks - 1;
      let last = &batch_input[last_chunk_idx * CHUNK_LEN..batch.batch_chunks * CHUNK_LEN];
      self.pending_chunk_cv = Some(hash_one_full_chunk_cv(
        kernels::kernel(self.bulk_kernel_id),
        self.key_words,
        self.chunk_state.flags,
        new_counter - 1,
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
