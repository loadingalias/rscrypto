use platform::Caps;
#[cfg(target_arch = "aarch64")]
use platform::caps::aarch64;
#[cfg(target_arch = "x86_64")]
use platform::caps::x86;

use super::{BLOCK_LEN, CHUNK_LEN, CHUNK_START, OUT_LEN, PARENT, first_8_words, words16_from_le_bytes_64};

// ─────────────────────────────────────────────────────────────────────────────
// Kernel function types
// ─────────────────────────────────────────────────────────────────────────────

/// Core compression function (single block).
pub(crate) type CompressFn = fn(&[u32; 8], &[u32; 16], u64, u32, u32) -> [u32; 16];

/// Chunk compression: process multiple 64-byte blocks within a chunk.
pub(crate) type ChunkCompressBlocksFn = fn(&mut [u32; 8], u64, u32, &mut u8, &[u8]);

/// Parent CV computation from two child CVs.
pub(crate) type ParentCvFn = fn([u32; 8], [u32; 8], [u32; 8], u32) -> [u32; 8];

/// Multi-chunk parallel hashing (THE KEY FUNCTION for large inputs).
/// Hashes multiple independent chunks in parallel using SIMD.
///
/// # Arguments
/// * `inputs` - Slice of input chunk data (all same length)
/// * `key` - Key/IV words
/// * `counter` - Starting counter value
/// * `increment_counter` - Whether to increment counter for each chunk
/// * `flags` - Base flags
/// * `flags_start` - CHUNK_START flag
/// * `flags_end` - CHUNK_END flag
/// * `out` - Output buffer (OUT_LEN bytes per chunk)
pub(crate) type HashManyFn = fn(
  inputs: &[&[u8]],
  key: &[u32; 8],
  counter: u64,
  increment_counter: bool,
  flags: u32,
  flags_start: u32,
  flags_end: u32,
  out: &mut [u8],
);

/// Multi-chunk hashing for contiguous input (hot path for large inputs).
///
/// Hashes `num_chunks` contiguous `CHUNK_LEN`-byte chunks from `input`, writing
/// `OUT_LEN * num_chunks` bytes to `out`.
///
/// # Safety
///
/// - `input` must point to at least `CHUNK_LEN * num_chunks` readable bytes.
/// - `out` must point to at least `OUT_LEN * num_chunks` writable bytes.
pub(crate) type HashManyContiguousFn =
  unsafe fn(input: *const u8, num_chunks: usize, key: &[u32; 8], counter: u64, flags: u32, out: *mut u8);

// ─────────────────────────────────────────────────────────────────────────────
// Kernel struct
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct Kernel {
  pub(crate) id: Blake3KernelId,
  /// Compress single block (for streaming/small inputs).
  pub(crate) compress: CompressFn,
  /// Compress multiple blocks within a chunk.
  pub(crate) chunk_compress_blocks: ChunkCompressBlocksFn,
  /// Compute parent CV from two child CVs.
  pub(crate) parent_cv: ParentCvFn,
  /// Hash many chunks in parallel (for large inputs).
  pub(crate) hash_many: HashManyFn,
  /// Hash many contiguous full chunks (for throughput, no pointer chasing).
  pub(crate) hash_many_contiguous: HashManyContiguousFn,
  /// SIMD degree: 1 for portable, 4 for NEON/SSE4.1, 8 for AVX2, 16 for AVX-512.
  pub(crate) simd_degree: usize,
  /// Kernel name for debugging/tuning.
  pub(crate) name: &'static str,
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel IDs
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Blake3KernelId {
  Portable = 0,
  #[cfg(target_arch = "x86_64")]
  X86Ssse3 = 1,
  #[cfg(target_arch = "x86_64")]
  X86Sse41 = 2,
  #[cfg(target_arch = "x86_64")]
  X86Avx2 = 3,
  #[cfg(target_arch = "x86_64")]
  X86Avx512 = 4,
  #[cfg(target_arch = "aarch64")]
  Aarch64Neon = 5,
}

pub const ALL: &[Blake3KernelId] = &[
  Blake3KernelId::Portable,
  #[cfg(target_arch = "x86_64")]
  Blake3KernelId::X86Ssse3,
  #[cfg(target_arch = "x86_64")]
  Blake3KernelId::X86Sse41,
  #[cfg(target_arch = "x86_64")]
  Blake3KernelId::X86Avx2,
  #[cfg(target_arch = "x86_64")]
  Blake3KernelId::X86Avx512,
  #[cfg(target_arch = "aarch64")]
  Blake3KernelId::Aarch64Neon,
];

impl Blake3KernelId {
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
      #[cfg(target_arch = "x86_64")]
      Self::X86Ssse3 => "x86_64/ssse3",
      #[cfg(target_arch = "x86_64")]
      Self::X86Sse41 => "x86_64/sse4.1",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx2 => "x86_64/avx2",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx512 => "x86_64/avx512",
      #[cfg(target_arch = "aarch64")]
      Self::Aarch64Neon => "aarch64/neon",
    }
  }

  /// Returns the SIMD degree for this kernel.
  #[inline]
  #[must_use]
  pub const fn simd_degree(self) -> usize {
    match self {
      Self::Portable => 1,
      #[cfg(target_arch = "x86_64")]
      Self::X86Ssse3 => 1, // Single-block only (no multi-lane hash_many)
      #[cfg(target_arch = "x86_64")]
      Self::X86Sse41 => 4,
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx2 => 8,
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx512 => 16,
      #[cfg(target_arch = "aarch64")]
      Self::Aarch64Neon => 4, // NEON processes 4 lanes
    }
  }
}

#[allow(dead_code)]
#[must_use]
pub fn id_from_name(name: &str) -> Option<Blake3KernelId> {
  match name {
    "portable" => Some(Blake3KernelId::Portable),
    #[cfg(target_arch = "x86_64")]
    "x86_64/ssse3" => Some(Blake3KernelId::X86Ssse3),
    #[cfg(target_arch = "x86_64")]
    "x86_64/sse4.1" => Some(Blake3KernelId::X86Sse41),
    #[cfg(target_arch = "x86_64")]
    "x86_64/avx2" => Some(Blake3KernelId::X86Avx2),
    #[cfg(target_arch = "x86_64")]
    "x86_64/avx512" => Some(Blake3KernelId::X86Avx512),
    #[cfg(target_arch = "aarch64")]
    "aarch64/neon" => Some(Blake3KernelId::Aarch64Neon),
    _ => None,
  }
}

#[must_use]
pub(crate) fn kernel(id: Blake3KernelId) -> Kernel {
  match id {
    Blake3KernelId::Portable => Kernel {
      id,
      compress: super::compress,
      chunk_compress_blocks: chunk_compress_blocks_portable,
      parent_cv: parent_cv_portable,
      hash_many: hash_many_portable,
      hash_many_contiguous: hash_many_contiguous_portable,
      simd_degree: 1,
      name: id.as_str(),
    },
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Ssse3 => Kernel {
      id,
      compress: compress_ssse3_wrapper,
      chunk_compress_blocks: chunk_compress_blocks_ssse3_wrapper,
      parent_cv: parent_cv_ssse3_wrapper,
      hash_many: hash_many_ssse3_wrapper,
      hash_many_contiguous: hash_many_contiguous_ssse3_wrapper,
      simd_degree: 1,
      name: id.as_str(),
    },
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Sse41 => Kernel {
      id,
      compress: compress_sse41_wrapper,
      chunk_compress_blocks: chunk_compress_blocks_sse41_wrapper,
      parent_cv: parent_cv_sse41_wrapper,
      hash_many: hash_many_sse41_wrapper,
      hash_many_contiguous: hash_many_contiguous_sse41_wrapper,
      simd_degree: 4,
      name: id.as_str(),
    },
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx2 => Kernel {
      id,
      compress: compress_sse41_wrapper,
      chunk_compress_blocks: chunk_compress_blocks_sse41_wrapper,
      parent_cv: parent_cv_sse41_wrapper,
      hash_many: hash_many_avx2_wrapper,
      hash_many_contiguous: hash_many_contiguous_avx2_wrapper,
      simd_degree: 8,
      name: id.as_str(),
    },
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx512 => Kernel {
      id,
      compress: compress_sse41_wrapper,
      chunk_compress_blocks: chunk_compress_blocks_sse41_wrapper,
      parent_cv: parent_cv_sse41_wrapper,
      hash_many: hash_many_avx512_wrapper,
      hash_many_contiguous: hash_many_contiguous_avx512_wrapper,
      simd_degree: 16,
      name: id.as_str(),
    },
    #[cfg(target_arch = "aarch64")]
    Blake3KernelId::Aarch64Neon => Kernel {
      id,
      // On aarch64, the 4-way NEON `hash_many`/`hash_many_contiguous` kernels
      // are the throughput workhorses. For single-block / streaming chunk
      // compression, the portable scalar compressor is competitive (and avoids
      // NEON message permutation overhead).
      compress: super::compress,
      chunk_compress_blocks: chunk_compress_blocks_portable,
      parent_cv: parent_cv_portable,
      hash_many: hash_many_neon_wrapper,
      hash_many_contiguous: hash_many_contiguous_neon_wrapper,
      simd_degree: 4,
      name: id.as_str(),
    },
  }
}

#[inline(always)]
pub(crate) fn chunk_compress_blocks_inline(
  id: Blake3KernelId,
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  match id {
    Blake3KernelId::Portable => {
      chunk_compress_blocks_portable(chaining_value, chunk_counter, flags, blocks_compressed, blocks)
    }
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Ssse3 => {
      chunk_compress_blocks_ssse3_wrapper(chaining_value, chunk_counter, flags, blocks_compressed, blocks)
    }
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Sse41 => {
      chunk_compress_blocks_sse41_wrapper(chaining_value, chunk_counter, flags, blocks_compressed, blocks)
    }
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx2 => {
      chunk_compress_blocks_sse41_wrapper(chaining_value, chunk_counter, flags, blocks_compressed, blocks)
    }
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx512 => {
      chunk_compress_blocks_sse41_wrapper(chaining_value, chunk_counter, flags, blocks_compressed, blocks)
    }
    #[cfg(target_arch = "aarch64")]
    Blake3KernelId::Aarch64Neon => {
      chunk_compress_blocks_portable(chaining_value, chunk_counter, flags, blocks_compressed, blocks)
    }
  }
}

#[inline(always)]
pub(crate) fn parent_cv_inline(
  id: Blake3KernelId,
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  match id {
    Blake3KernelId::Portable => parent_cv_portable(left_child_cv, right_child_cv, key_words, flags),
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Ssse3 => parent_cv_ssse3_wrapper(left_child_cv, right_child_cv, key_words, flags),
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Sse41 => parent_cv_sse41_wrapper(left_child_cv, right_child_cv, key_words, flags),
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx2 => parent_cv_sse41_wrapper(left_child_cv, right_child_cv, key_words, flags),
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx512 => parent_cv_sse41_wrapper(left_child_cv, right_child_cv, key_words, flags),
    #[cfg(target_arch = "aarch64")]
    Blake3KernelId::Aarch64Neon => parent_cv_portable(left_child_cv, right_child_cv, key_words, flags),
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Blake3KernelId) -> Caps {
  match id {
    Blake3KernelId::Portable => Caps::NONE,
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Ssse3 => x86::SSSE3,
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Sse41 => x86::SSE41.union(x86::SSSE3),
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx2 => x86::AVX2.union(x86::SSE41).union(x86::SSSE3),
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx512 => x86::AVX512_READY.union(x86::AVX2).union(x86::SSE41).union(x86::SSSE3),
    #[cfg(target_arch = "aarch64")]
    Blake3KernelId::Aarch64Neon => aarch64::NEON,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Portable implementations
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn chunk_compress_blocks_portable(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);

  // Hot path for streaming callers that feed one full block at a time.
  if blocks.len() == BLOCK_LEN {
    // SAFETY: `blocks` is exactly one block, and `[u8; BLOCK_LEN]` has 1-byte alignment.
    let block_bytes: &[u8; BLOCK_LEN] = unsafe { &*(blocks.as_ptr().cast()) };
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    let block_words = words16_from_le_bytes_64(block_bytes);
    *chaining_value = first_8_words((super::compress)(
      chaining_value,
      &block_words,
      chunk_counter,
      BLOCK_LEN as u32,
      flags | start,
    ));
    *blocks_compressed = blocks_compressed.wrapping_add(1);
    return;
  }

  let (block_slices, remainder) = blocks.as_chunks::<BLOCK_LEN>();
  debug_assert!(remainder.is_empty());
  for block_bytes in block_slices {
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    let block_words = words16_from_le_bytes_64(block_bytes);
    *chaining_value = first_8_words((super::compress)(
      chaining_value,
      &block_words,
      chunk_counter,
      BLOCK_LEN as u32,
      flags | start,
    ));
    *blocks_compressed = blocks_compressed.wrapping_add(1);
  }
}

#[inline]
fn parent_cv_portable(left_child_cv: [u32; 8], right_child_cv: [u32; 8], key_words: [u32; 8], flags: u32) -> [u32; 8] {
  let mut block_words = [0u32; 16];
  block_words[..8].copy_from_slice(&left_child_cv);
  block_words[8..].copy_from_slice(&right_child_cv);
  first_8_words((super::compress)(
    &key_words,
    &block_words,
    0,
    BLOCK_LEN as u32,
    PARENT | flags,
  ))
}

unsafe fn hash_many_contiguous_portable(
  input: *const u8,
  num_chunks: usize,
  key: &[u32; 8],
  counter: u64,
  flags: u32,
  out: *mut u8,
) {
  debug_assert!(num_chunks != 0);

  for chunk_idx in 0..num_chunks {
    let chunk_counter = counter.wrapping_add(chunk_idx as u64);
    let mut cv = *key;

    for block_idx in 0..(CHUNK_LEN / BLOCK_LEN) {
      let mut block = [0u8; BLOCK_LEN];
      // SAFETY: caller guarantees `input` is valid for `num_chunks * CHUNK_LEN`.
      unsafe {
        let src = input.add(chunk_idx * CHUNK_LEN + block_idx * BLOCK_LEN);
        core::ptr::copy_nonoverlapping(src, block.as_mut_ptr(), BLOCK_LEN);
      }
      let block_words = words16_from_le_bytes_64(&block);

      let start = if block_idx == 0 { CHUNK_START } else { 0 };
      let end = if block_idx + 1 == (CHUNK_LEN / BLOCK_LEN) {
        super::CHUNK_END
      } else {
        0
      };
      let block_flags = flags | start | end;
      cv = first_8_words(super::compress(
        &cv,
        &block_words,
        chunk_counter,
        BLOCK_LEN as u32,
        block_flags,
      ));
    }

    for (j, &word) in cv.iter().enumerate() {
      let bytes = word.to_le_bytes();
      // SAFETY: caller guarantees out is valid for `num_chunks * OUT_LEN`.
      unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), out.add(chunk_idx * OUT_LEN + j * 4), 4) };
    }
  }
}

/// Portable hash_many: processes chunks one at a time (no SIMD parallelism).
#[allow(clippy::too_many_arguments)]
fn hash_many_portable(
  inputs: &[&[u8]],
  key: &[u32; 8],
  mut counter: u64,
  increment_counter: bool,
  flags: u32,
  flags_start: u32,
  flags_end: u32,
  out: &mut [u8],
) {
  debug_assert!(!inputs.is_empty());
  debug_assert_eq!(out.len(), inputs.len() * OUT_LEN);

  let mut out_offset = 0;

  for input in inputs {
    if input.len() == CHUNK_LEN {
      // Fast path: a full chunk, no padding/length handling.
      // SAFETY: `input` is exactly one full chunk and `out` has `OUT_LEN` bytes.
      unsafe { hash_many_contiguous_portable(input.as_ptr(), 1, key, counter, flags, out[out_offset..].as_mut_ptr()) };
      out_offset += OUT_LEN;
      if increment_counter {
        counter += 1;
      }
      continue;
    }

    let mut cv = *key;
    let num_blocks = input.len().div_ceil(BLOCK_LEN);

    for block_idx in 0..num_blocks {
      let block_offset = block_idx * BLOCK_LEN;
      let is_first = block_idx == 0;
      let is_last = block_idx == num_blocks - 1;

      let block_end = core::cmp::min(block_offset + BLOCK_LEN, input.len());
      let block_len = (block_end - block_offset) as u32;

      // Prepare block (pad if necessary)
      let mut block = [0u8; BLOCK_LEN];
      block[..block_len as usize].copy_from_slice(&input[block_offset..block_end]);

      let block_words = words16_from_le_bytes_64(&block);

      let mut block_flags = flags;
      if is_first {
        block_flags |= flags_start;
      }
      if is_last {
        block_flags |= flags_end;
      }

      cv = first_8_words(super::compress(&cv, &block_words, counter, block_len, block_flags));
    }

    // Write output CV as little-endian bytes
    for (j, &word) in cv.iter().enumerate() {
      let bytes = word.to_le_bytes();
      out[out_offset + j * 4..out_offset + (j + 1) * 4].copy_from_slice(&bytes);
    }
    out_offset += OUT_LEN;

    if increment_counter {
      counter += 1;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// x86_64 SSSE3 wrappers
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn compress_ssse3_wrapper(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  // SAFETY: This function is only called when SSSE3 is available (checked by dispatch).
  unsafe { super::x86_64::compress_ssse3(chaining_value, block_words, counter, block_len, flags) }
}

#[cfg(target_arch = "x86_64")]
fn chunk_compress_blocks_ssse3_wrapper(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  // SAFETY: This function is only called when SSSE3 is available (checked by dispatch).
  unsafe { super::x86_64::chunk_compress_blocks_ssse3(chaining_value, chunk_counter, flags, blocks_compressed, blocks) }
}

#[cfg(target_arch = "x86_64")]
fn parent_cv_ssse3_wrapper(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  // SAFETY: This function is only called when SSSE3 is available (checked by dispatch).
  unsafe { super::x86_64::parent_cv_ssse3(left_child_cv, right_child_cv, key_words, flags) }
}

// ─────────────────────────────────────────────────────────────────────────────
// x86_64 SSE4.1 wrappers (single-block / streaming hot paths)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn compress_sse41_wrapper(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  // SAFETY: This function is only called when SSE4.1 is available (checked by dispatch).
  unsafe { super::x86_64::sse41::compress(chaining_value, block_words, counter, block_len, flags) }
}

#[cfg(target_arch = "x86_64")]
fn chunk_compress_blocks_sse41_wrapper(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
  let (block_slices, remainder) = blocks.as_chunks::<BLOCK_LEN>();
  debug_assert!(remainder.is_empty());

  for block_bytes in block_slices {
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    let block_words = words16_from_le_bytes_64(block_bytes);
    *chaining_value = first_8_words(compress_sse41_wrapper(
      chaining_value,
      &block_words,
      chunk_counter,
      BLOCK_LEN as u32,
      flags | start,
    ));
    *blocks_compressed = blocks_compressed.wrapping_add(1);
  }
}

#[cfg(target_arch = "x86_64")]
fn parent_cv_sse41_wrapper(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  let mut block_words = [0u32; 16];
  block_words[..8].copy_from_slice(&left_child_cv);
  block_words[8..].copy_from_slice(&right_child_cv);
  first_8_words(compress_sse41_wrapper(
    &key_words,
    &block_words,
    0,
    BLOCK_LEN as u32,
    PARENT | flags,
  ))
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn hash_many_ssse3_wrapper(
  inputs: &[&[u8]],
  key: &[u32; 8],
  counter: u64,
  increment_counter: bool,
  flags: u32,
  flags_start: u32,
  flags_end: u32,
  out: &mut [u8],
) {
  // SSSE3 currently only accelerates the single-block path. Fall back to the
  // portable multi-chunk implementation.
  hash_many_portable(
    inputs,
    key,
    counter,
    increment_counter,
    flags,
    flags_start,
    flags_end,
    out,
  )
}

#[cfg(target_arch = "x86_64")]
unsafe fn hash_many_contiguous_ssse3_wrapper(
  input: *const u8,
  num_chunks: usize,
  key: &[u32; 8],
  counter: u64,
  flags: u32,
  out: *mut u8,
) {
  // For now, process chunks serially using the SSSE3 single-block compressor.
  // SAFETY: This function is only called when SSSE3 is available (checked by dispatch).
  unsafe { super::x86_64::hash_many_contiguous_ssse3(input, num_chunks, key, counter, flags, out) }
}

// ─────────────────────────────────────────────────────────────────────────────
// x86_64 SSE4.1/AVX2/AVX-512 wrappers (throughput kernels)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn hash_many_sse41_wrapper(
  inputs: &[&[u8]],
  key: &[u32; 8],
  mut counter: u64,
  increment_counter: bool,
  flags: u32,
  flags_start: u32,
  flags_end: u32,
  out: &mut [u8],
) {
  debug_assert_eq!(out.len(), inputs.len() * OUT_LEN);

  // This kernel only accelerates full chunks (the only shape used in library
  // throughput paths). Anything else falls back to portable.
  if inputs.iter().any(|s| s.len() != CHUNK_LEN) {
    hash_many_portable(
      inputs,
      key,
      counter,
      increment_counter,
      flags,
      flags_start,
      flags_end,
      out,
    );
    return;
  }

  let mut in_idx = 0usize;
  let mut out_off = 0usize;
  while in_idx + super::x86_64::sse41::DEGREE <= inputs.len() {
    let ptrs = [
      inputs[in_idx].as_ptr(),
      inputs[in_idx + 1].as_ptr(),
      inputs[in_idx + 2].as_ptr(),
      inputs[in_idx + 3].as_ptr(),
    ];
    // SAFETY: dispatch validates CPU caps; inputs are full chunks; out is sized.
    unsafe {
      super::x86_64::sse41::hash4(
        &ptrs,
        CHUNK_LEN / BLOCK_LEN,
        key,
        counter,
        increment_counter,
        flags,
        flags_start,
        flags_end,
        out.as_mut_ptr().add(out_off),
      )
    };
    if increment_counter {
      counter = counter.wrapping_add(super::x86_64::sse41::DEGREE as u64);
    }
    in_idx += super::x86_64::sse41::DEGREE;
    out_off += super::x86_64::sse41::DEGREE * OUT_LEN;
  }

  if in_idx < inputs.len() {
    hash_many_portable(
      &inputs[in_idx..],
      key,
      counter,
      increment_counter,
      flags,
      flags_start,
      flags_end,
      &mut out[out_off..],
    );
  }
}

#[cfg(target_arch = "x86_64")]
unsafe fn hash_many_contiguous_sse41_wrapper(
  input: *const u8,
  mut num_chunks: usize,
  key: &[u32; 8],
  mut counter: u64,
  flags: u32,
  mut out: *mut u8,
) {
  let mut input = input;
  while num_chunks >= super::x86_64::sse41::DEGREE {
    // SAFETY: `num_chunks >= DEGREE` means there are at least `DEGREE` full
    // chunks remaining. The caller guarantees `input` is valid for the full
    // input buffer, so each `input.add(k * CHUNK_LEN)` stays in-bounds.
    let ptrs = unsafe {
      [
        input,
        input.add(CHUNK_LEN),
        input.add(2 * CHUNK_LEN),
        input.add(3 * CHUNK_LEN),
      ]
    };
    // SAFETY: dispatch selects this kernel only when SSE4.1 is available; the
    // caller guarantees `input`/`out` cover the full `num_chunks` buffer.
    unsafe {
      super::x86_64::sse41::hash4(
        &ptrs,
        CHUNK_LEN / BLOCK_LEN,
        key,
        counter,
        true,
        flags,
        CHUNK_START,
        super::CHUNK_END,
        out,
      );
      input = input.add(super::x86_64::sse41::DEGREE * CHUNK_LEN);
      out = out.add(super::x86_64::sse41::DEGREE * OUT_LEN);
    }
    counter = counter.wrapping_add(super::x86_64::sse41::DEGREE as u64);
    num_chunks -= super::x86_64::sse41::DEGREE;
  }

  if num_chunks != 0 {
    // SAFETY: the remaining chunk pointers are in-bounds for the caller's
    // `input`/`out` buffer, and the portable implementation writes `OUT_LEN`
    // bytes per chunk.
    unsafe { hash_many_contiguous_portable(input, num_chunks, key, counter, flags, out) };
  }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn hash_many_avx2_wrapper(
  inputs: &[&[u8]],
  key: &[u32; 8],
  mut counter: u64,
  increment_counter: bool,
  flags: u32,
  flags_start: u32,
  flags_end: u32,
  out: &mut [u8],
) {
  debug_assert_eq!(out.len(), inputs.len() * OUT_LEN);

  if inputs.iter().any(|s| s.len() != CHUNK_LEN) {
    hash_many_portable(
      inputs,
      key,
      counter,
      increment_counter,
      flags,
      flags_start,
      flags_end,
      out,
    );
    return;
  }

  let mut in_idx = 0usize;
  let mut out_off = 0usize;
  while in_idx + super::x86_64::avx2::DEGREE <= inputs.len() {
    let ptrs = [
      inputs[in_idx].as_ptr(),
      inputs[in_idx + 1].as_ptr(),
      inputs[in_idx + 2].as_ptr(),
      inputs[in_idx + 3].as_ptr(),
      inputs[in_idx + 4].as_ptr(),
      inputs[in_idx + 5].as_ptr(),
      inputs[in_idx + 6].as_ptr(),
      inputs[in_idx + 7].as_ptr(),
    ];
    // SAFETY: dispatch validates CPU caps; inputs are full chunks; out is sized.
    unsafe {
      super::x86_64::avx2::hash8(
        &ptrs,
        CHUNK_LEN / BLOCK_LEN,
        key,
        counter,
        increment_counter,
        flags,
        flags_start,
        flags_end,
        out.as_mut_ptr().add(out_off),
      )
    };
    if increment_counter {
      counter = counter.wrapping_add(super::x86_64::avx2::DEGREE as u64);
    }
    in_idx += super::x86_64::avx2::DEGREE;
    out_off += super::x86_64::avx2::DEGREE * OUT_LEN;
  }

  if in_idx < inputs.len() {
    hash_many_sse41_wrapper(
      &inputs[in_idx..],
      key,
      counter,
      increment_counter,
      flags,
      flags_start,
      flags_end,
      &mut out[out_off..],
    );
  }
}

#[cfg(target_arch = "x86_64")]
unsafe fn hash_many_contiguous_avx2_wrapper(
  input: *const u8,
  mut num_chunks: usize,
  key: &[u32; 8],
  mut counter: u64,
  flags: u32,
  mut out: *mut u8,
) {
  let mut input = input;
  while num_chunks >= super::x86_64::avx2::DEGREE {
    // SAFETY: `num_chunks >= DEGREE` means there are at least `DEGREE` full
    // chunks remaining. The caller guarantees `input` is valid for the full
    // input buffer, so each `input.add(k * CHUNK_LEN)` stays in-bounds.
    let ptrs = unsafe {
      [
        input,
        input.add(CHUNK_LEN),
        input.add(2 * CHUNK_LEN),
        input.add(3 * CHUNK_LEN),
        input.add(4 * CHUNK_LEN),
        input.add(5 * CHUNK_LEN),
        input.add(6 * CHUNK_LEN),
        input.add(7 * CHUNK_LEN),
      ]
    };
    // SAFETY: dispatch selects this kernel only when AVX2 is available; the
    // caller guarantees `input`/`out` cover the full `num_chunks` buffer.
    unsafe {
      super::x86_64::avx2::hash8(
        &ptrs,
        CHUNK_LEN / BLOCK_LEN,
        key,
        counter,
        true,
        flags,
        CHUNK_START,
        super::CHUNK_END,
        out,
      );
      input = input.add(super::x86_64::avx2::DEGREE * CHUNK_LEN);
      out = out.add(super::x86_64::avx2::DEGREE * OUT_LEN);
    }
    counter = counter.wrapping_add(super::x86_64::avx2::DEGREE as u64);
    num_chunks -= super::x86_64::avx2::DEGREE;
  }

  if num_chunks != 0 {
    // SAFETY: the remaining chunk pointers are in-bounds for the caller's
    // `input`/`out` buffer, and SSE4.1 is always available when AVX2 is.
    unsafe { hash_many_contiguous_sse41_wrapper(input, num_chunks, key, counter, flags, out) };
  }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn hash_many_avx512_wrapper(
  inputs: &[&[u8]],
  key: &[u32; 8],
  counter: u64,
  increment_counter: bool,
  flags: u32,
  flags_start: u32,
  flags_end: u32,
  out: &mut [u8],
) {
  // Slice-of-slices hashing is still handled by the AVX2/SSE4.1 kernels.
  hash_many_avx2_wrapper(
    inputs,
    key,
    counter,
    increment_counter,
    flags,
    flags_start,
    flags_end,
    out,
  )
}

#[cfg(target_arch = "x86_64")]
unsafe fn hash_many_contiguous_avx512_wrapper(
  input: *const u8,
  mut num_chunks: usize,
  key: &[u32; 8],
  mut counter: u64,
  flags: u32,
  mut out: *mut u8,
) {
  let mut input = input;
  while num_chunks >= super::x86_64::avx512::DEGREE {
    // SAFETY: dispatch selects this kernel only when AVX-512 is available; the
    // caller guarantees `input`/`out` cover the full `num_chunks` buffer.
    unsafe {
      super::x86_64::avx512::hash16_contiguous(input, key, counter, flags, out);
      input = input.add(super::x86_64::avx512::DEGREE * CHUNK_LEN);
      out = out.add(super::x86_64::avx512::DEGREE * OUT_LEN);
    }
    counter = counter.wrapping_add(super::x86_64::avx512::DEGREE as u64);
    num_chunks -= super::x86_64::avx512::DEGREE;
  }

  if num_chunks != 0 {
    // SAFETY: the remaining chunk pointers are in-bounds for the caller's
    // `input`/`out` buffer, and AVX2 is available when AVX-512 is.
    unsafe { hash_many_contiguous_avx2_wrapper(input, num_chunks, key, counter, flags, out) };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// aarch64 NEON wrappers
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
fn compress_neon_wrapper(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  // SAFETY: This function is only called when NEON is available (checked by dispatch).
  unsafe { super::aarch64::compress_neon(chaining_value, block_words, counter, block_len, flags) }
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
fn chunk_compress_blocks_neon_wrapper(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  // SAFETY: This function is only called when NEON is available (checked by dispatch).
  unsafe { super::aarch64::chunk_compress_blocks_neon(chaining_value, chunk_counter, flags, blocks_compressed, blocks) }
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
fn parent_cv_neon_wrapper(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  // SAFETY: This function is only called when NEON is available (checked by dispatch).
  unsafe { super::aarch64::parent_cv_neon(left_child_cv, right_child_cv, key_words, flags) }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn hash_many_neon_wrapper(
  inputs: &[&[u8]],
  key: &[u32; 8],
  counter: u64,
  increment_counter: bool,
  flags: u32,
  flags_start: u32,
  flags_end: u32,
  out: &mut [u8],
) {
  // SAFETY: This function is only called when NEON is available (checked by dispatch).
  unsafe {
    super::aarch64::hash_many_neon(
      inputs,
      key,
      counter,
      increment_counter,
      flags,
      flags_start,
      flags_end,
      out,
    )
  }
}

#[cfg(target_arch = "aarch64")]
unsafe fn hash_many_contiguous_neon_wrapper(
  input: *const u8,
  num_chunks: usize,
  key: &[u32; 8],
  counter: u64,
  flags: u32,
  out: *mut u8,
) {
  // SAFETY: This function is only called when NEON is available (checked by dispatch).
  unsafe { super::aarch64::hash_many_contiguous_neon(input, num_chunks, key, counter, flags, out) }
}
