use platform::Caps;
#[cfg(target_arch = "aarch64")]
use platform::caps::aarch64;
#[cfg(target_arch = "x86_64")]
use platform::caps::x86;

#[cfg(target_arch = "x86_64")]
use super::words8_from_le_bytes_32;
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
pub(crate) struct Kernel {
  pub(crate) id: Blake3KernelId,
  /// Compress single block (for streaming/small inputs).
  pub(crate) compress: CompressFn,
  /// Compress multiple blocks within a chunk.
  #[allow(dead_code)] // Kept for bench-level introspection / future refactors.
  pub(crate) chunk_compress_blocks: ChunkCompressBlocksFn,
  /// Compute parent CV from two child CVs.
  #[allow(dead_code)] // Kept for bench-level introspection / future refactors.
  pub(crate) parent_cv: ParentCvFn,
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

#[must_use]
pub(crate) fn kernel(id: Blake3KernelId) -> Kernel {
  match id {
    Blake3KernelId::Portable => Kernel {
      id,
      compress: super::compress,
      chunk_compress_blocks: chunk_compress_blocks_portable,
      parent_cv: parent_cv_portable,
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
      hash_many_contiguous: hash_many_contiguous_ssse3_wrapper,
      simd_degree: 1,
      name: id.as_str(),
    },
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Sse41 => Kernel {
      id,
      // Per-block hot paths: keep a distinct SSE4.1 entrypoint for dispatch,
      // but forward to the SSSE3 row-wise compressor (SSE4.1 implies SSSE3 for
      // the platforms we care about, and the dispatcher enforces it).
      compress: compress_sse41_wrapper,
      chunk_compress_blocks: chunk_compress_blocks_sse41_wrapper,
      parent_cv: parent_cv_sse41_wrapper,
      hash_many_contiguous: hash_many_contiguous_sse41_wrapper,
      simd_degree: 4,
      name: id.as_str(),
    },
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx2 => Kernel {
      id,
      // AVX2 accelerates multi-chunk hashing, and we also use an AVX2-enabled
      // per-block compressor to avoid AVX<->SSE transition penalties in mixed
      // streaming workloads.
      compress: compress_avx2_wrapper,
      chunk_compress_blocks: chunk_compress_blocks_avx2_wrapper,
      parent_cv: parent_cv_avx2_wrapper,
      hash_many_contiguous: hash_many_contiguous_avx2_wrapper,
      simd_degree: 8,
      name: id.as_str(),
    },
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx512 => Kernel {
      id,
      // AVX-512 accelerates contiguous multi-chunk hashing; per-block compression
      // uses an AVX-512-enabled per-block entrypoint to avoid transition
      // penalties in mixed workloads.
      compress: compress_avx512_wrapper,
      chunk_compress_blocks: chunk_compress_blocks_avx512_wrapper,
      parent_cv: parent_cv_avx512_wrapper,
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
      chunk_compress_blocks: chunk_compress_blocks_neon_wrapper,
      parent_cv: parent_cv_portable,
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
      chunk_compress_blocks_avx2_wrapper(chaining_value, chunk_counter, flags, blocks_compressed, blocks)
    }
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx512 => {
      chunk_compress_blocks_avx512_wrapper(chaining_value, chunk_counter, flags, blocks_compressed, blocks)
    }
    #[cfg(target_arch = "aarch64")]
    Blake3KernelId::Aarch64Neon => {
      chunk_compress_blocks_neon_wrapper(chaining_value, chunk_counter, flags, blocks_compressed, blocks)
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
    Blake3KernelId::X86Avx2 => parent_cv_avx2_wrapper(left_child_cv, right_child_cv, key_words, flags),
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx512 => parent_cv_avx512_wrapper(left_child_cv, right_child_cv, key_words, flags),
    #[cfg(target_arch = "aarch64")]
    Blake3KernelId::Aarch64Neon => parent_cv_portable(left_child_cv, right_child_cv, key_words, flags),
  }
}

/// Compute many parent CVs from pre-packed parent blocks.
///
/// Each entry in `parents` is a `[u32; 16]` message block representing
/// `[left_cv (8 words), right_cv (8 words)]`.
#[inline]
pub(crate) fn parent_cvs_many_inline(
  id: Blake3KernelId,
  parents: &[[u32; 16]],
  key_words: [u32; 8],
  flags: u32,
  out: &mut [[u32; 8]],
) {
  debug_assert_eq!(parents.len(), out.len());
  if parents.is_empty() {
    return;
  }

  match id {
    Blake3KernelId::Portable => {}
    #[cfg(target_arch = "aarch64")]
    Blake3KernelId::Aarch64Neon => {}
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Ssse3 => {}
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Sse41 => {
      let mut i = 0usize;
      let parent_flags = PARENT | flags;

      while parents.len() - i >= super::x86_64::sse41::DEGREE {
        let ptrs: [*const u8; super::x86_64::sse41::DEGREE] = [
          parents[i].as_ptr().cast(),
          parents[i + 1].as_ptr().cast(),
          parents[i + 2].as_ptr().cast(),
          parents[i + 3].as_ptr().cast(),
        ];

        let mut tmp = [0u8; super::x86_64::sse41::DEGREE * OUT_LEN];
        // SAFETY: `id` is SSE4.1, so SSE4.1 is available; each pointer is
        // valid for one 64-byte block, and `tmp` is large enough.
        unsafe {
          super::x86_64::sse41::hash4(&ptrs, 1, &key_words, 0, false, parent_flags, 0, 0, tmp.as_mut_ptr());
        }

        for lane in 0..super::x86_64::sse41::DEGREE {
          // SAFETY: `tmp` is `DEGREE * OUT_LEN` bytes, and `lane < DEGREE`.
          let bytes: &[u8; OUT_LEN] = unsafe { &*(tmp.as_ptr().add(lane * OUT_LEN).cast()) };
          out[i + lane] = words8_from_le_bytes_32(bytes);
        }

        i += super::x86_64::sse41::DEGREE;
      }

      let rem = parents.len() - i;
      if rem != 0 {
        // Hash a 4-lane batch with duplicated final pointers and copy only the
        // required outputs.
        let last_ptr: *const u8 = parents[parents.len() - 1].as_ptr().cast();
        let ptrs: [*const u8; super::x86_64::sse41::DEGREE] = [
          parents[i].as_ptr().cast(),
          if rem > 1 {
            parents[i + 1].as_ptr().cast()
          } else {
            last_ptr
          },
          if rem > 2 {
            parents[i + 2].as_ptr().cast()
          } else {
            last_ptr
          },
          last_ptr,
        ];

        let mut tmp = [0u8; super::x86_64::sse41::DEGREE * OUT_LEN];
        // SAFETY: SSE4.1 is available for this wrapper; pointers and outputs are valid.
        unsafe {
          super::x86_64::sse41::hash4(&ptrs, 1, &key_words, 0, false, parent_flags, 0, 0, tmp.as_mut_ptr());
        }

        for lane in 0..rem {
          // SAFETY: `lane < rem <= DEGREE`, so this is in-bounds.
          let bytes: &[u8; OUT_LEN] = unsafe { &*(tmp.as_ptr().add(lane * OUT_LEN).cast()) };
          out[i + lane] = words8_from_le_bytes_32(bytes);
        }
      }
      return;
    }
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx512 => {
      let parent_flags = PARENT | flags;

      #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
      {
        // Use the upstream-grade AVX-512 asm `hash_many` backend for parent
        // folding too. This avoids an AVX-512 -> AVX2 downgrade in the tree
        // reduction path, which shows up clearly in large-input benchmarks.
        const DEGREE: usize = 16;
        debug_assert!(parents.len() <= 16, "parent folding batches are capped at 16");

        let mut i = 0usize;
        while parents.len() - i >= DEGREE {
          let ptrs: [*const u8; DEGREE] = [
            parents[i].as_ptr().cast(),
            parents[i + 1].as_ptr().cast(),
            parents[i + 2].as_ptr().cast(),
            parents[i + 3].as_ptr().cast(),
            parents[i + 4].as_ptr().cast(),
            parents[i + 5].as_ptr().cast(),
            parents[i + 6].as_ptr().cast(),
            parents[i + 7].as_ptr().cast(),
            parents[i + 8].as_ptr().cast(),
            parents[i + 9].as_ptr().cast(),
            parents[i + 10].as_ptr().cast(),
            parents[i + 11].as_ptr().cast(),
            parents[i + 12].as_ptr().cast(),
            parents[i + 13].as_ptr().cast(),
            parents[i + 14].as_ptr().cast(),
            parents[i + 15].as_ptr().cast(),
          ];

          let mut tmp = [0u8; DEGREE * OUT_LEN];
          // SAFETY: `id` is AVX-512, so required AVX-512 features are present
          // per dispatch. Each pointer is valid for one 64-byte parent block,
          // and `tmp` is large enough for `DEGREE` outputs.
          unsafe {
            super::x86_64::asm::rscrypto_blake3_hash_many_avx512(
              ptrs.as_ptr(),
              DEGREE,
              1,
              key_words.as_ptr(),
              0,
              false,
              parent_flags as u8,
              0,
              0,
              tmp.as_mut_ptr(),
            );
          }

          for lane in 0..DEGREE {
            // SAFETY: `tmp` is `DEGREE * OUT_LEN` bytes, and `lane < DEGREE`.
            let bytes: &[u8; OUT_LEN] = unsafe { &*(tmp.as_ptr().add(lane * OUT_LEN).cast()) };
            out[i + lane] = words8_from_le_bytes_32(bytes);
          }

          i += DEGREE;
        }

        let rem = parents.len() - i;
        if rem != 0 {
          let last_ptr: *const u8 = parents[parents.len() - 1].as_ptr().cast();
          let mut ptrs = [last_ptr; DEGREE];
          for lane in 0..rem {
            ptrs[lane] = parents[i + lane].as_ptr().cast();
          }

          let mut tmp = [0u8; DEGREE * OUT_LEN];
          // SAFETY: same as the main loop; we duplicate the final pointer to
          // fill the 16-lane batch and only read the first `rem` outputs.
          unsafe {
            super::x86_64::asm::rscrypto_blake3_hash_many_avx512(
              ptrs.as_ptr(),
              DEGREE,
              1,
              key_words.as_ptr(),
              0,
              false,
              parent_flags as u8,
              0,
              0,
              tmp.as_mut_ptr(),
            );
          }

          for lane in 0..rem {
            // SAFETY: `tmp` is `DEGREE * OUT_LEN` bytes, and `lane < rem <= DEGREE`.
            let bytes: &[u8; OUT_LEN] = unsafe { &*(tmp.as_ptr().add(lane * OUT_LEN).cast()) };
            out[i + lane] = words8_from_le_bytes_32(bytes);
          }
        }

        return;
      }

      #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
      {
        // Non-Linux fallback: treat AVX-512 as AVX2 for parent folding.
        let mut i = 0usize;
        while parents.len() - i >= super::x86_64::avx2::DEGREE {
          let ptrs: [*const u8; super::x86_64::avx2::DEGREE] = [
            parents[i].as_ptr().cast(),
            parents[i + 1].as_ptr().cast(),
            parents[i + 2].as_ptr().cast(),
            parents[i + 3].as_ptr().cast(),
            parents[i + 4].as_ptr().cast(),
            parents[i + 5].as_ptr().cast(),
            parents[i + 6].as_ptr().cast(),
            parents[i + 7].as_ptr().cast(),
          ];

          let mut tmp = [0u8; super::x86_64::avx2::DEGREE * OUT_LEN];
          // SAFETY: `id` is AVX-512 and the caller ensured required CPU
          // features. The AVX2 routine is safe to call (AVX-512 implies AVX2 per
          // dispatch), all pointers are valid for one block, and `tmp` is large
          // enough for `DEGREE` outputs.
          unsafe {
            super::x86_64::avx2::hash8(&ptrs, 1, &key_words, 0, false, parent_flags, 0, 0, tmp.as_mut_ptr());
          }

          for lane in 0..super::x86_64::avx2::DEGREE {
            // SAFETY: `tmp` is `DEGREE * OUT_LEN` bytes, and `lane < DEGREE`.
            let bytes: &[u8; OUT_LEN] = unsafe { &*(tmp.as_ptr().add(lane * OUT_LEN).cast()) };
            out[i + lane] = words8_from_le_bytes_32(bytes);
          }

          i += super::x86_64::avx2::DEGREE;
        }

        let rem = parents.len() - i;
        if rem != 0 {
          let last_ptr: *const u8 = parents[parents.len() - 1].as_ptr().cast();
          let ptrs: [*const u8; super::x86_64::avx2::DEGREE] = [
            parents[i].as_ptr().cast(),
            if rem > 1 {
              parents[i + 1].as_ptr().cast()
            } else {
              last_ptr
            },
            if rem > 2 {
              parents[i + 2].as_ptr().cast()
            } else {
              last_ptr
            },
            if rem > 3 {
              parents[i + 3].as_ptr().cast()
            } else {
              last_ptr
            },
            if rem > 4 {
              parents[i + 4].as_ptr().cast()
            } else {
              last_ptr
            },
            if rem > 5 {
              parents[i + 5].as_ptr().cast()
            } else {
              last_ptr
            },
            if rem > 6 {
              parents[i + 6].as_ptr().cast()
            } else {
              last_ptr
            },
            last_ptr,
          ];

          let mut tmp = [0u8; super::x86_64::avx2::DEGREE * OUT_LEN];
          // SAFETY: AVX2 is available per dispatch; pointers are valid for one
          // block each, and `tmp` is large enough for the outputs we read.
          unsafe {
            super::x86_64::avx2::hash8(&ptrs, 1, &key_words, 0, false, parent_flags, 0, 0, tmp.as_mut_ptr());
          }

          for lane in 0..rem {
            // SAFETY: `tmp` is `DEGREE * OUT_LEN` bytes, and `lane < rem <= DEGREE`.
            let bytes: &[u8; OUT_LEN] = unsafe { &*(tmp.as_ptr().add(lane * OUT_LEN).cast()) };
            out[i + lane] = words8_from_le_bytes_32(bytes);
          }
        }
        return;
      }
    }
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx2 => {
      let mut i = 0usize;
      let parent_flags = PARENT | flags;

      while parents.len() - i >= super::x86_64::avx2::DEGREE {
        let ptrs: [*const u8; super::x86_64::avx2::DEGREE] = [
          parents[i].as_ptr().cast(),
          parents[i + 1].as_ptr().cast(),
          parents[i + 2].as_ptr().cast(),
          parents[i + 3].as_ptr().cast(),
          parents[i + 4].as_ptr().cast(),
          parents[i + 5].as_ptr().cast(),
          parents[i + 6].as_ptr().cast(),
          parents[i + 7].as_ptr().cast(),
        ];

        let mut tmp = [0u8; super::x86_64::avx2::DEGREE * OUT_LEN];
        // SAFETY: `id` is AVX2/AVX-512, so AVX2 is available; pointers and outputs are valid.
        unsafe {
          super::x86_64::avx2::hash8(&ptrs, 1, &key_words, 0, false, parent_flags, 0, 0, tmp.as_mut_ptr());
        }

        for lane in 0..super::x86_64::avx2::DEGREE {
          // SAFETY: `tmp` is `DEGREE * OUT_LEN` bytes, and `lane < DEGREE`.
          let bytes: &[u8; OUT_LEN] = unsafe { &*(tmp.as_ptr().add(lane * OUT_LEN).cast()) };
          out[i + lane] = words8_from_le_bytes_32(bytes);
        }

        i += super::x86_64::avx2::DEGREE;
      }

      let rem = parents.len() - i;
      if rem != 0 {
        // Hash an 8-lane batch with duplicated final pointers and copy only the
        // required outputs.
        let last_ptr: *const u8 = parents[parents.len() - 1].as_ptr().cast();
        let ptrs: [*const u8; super::x86_64::avx2::DEGREE] = [
          parents[i].as_ptr().cast(),
          if rem > 1 {
            parents[i + 1].as_ptr().cast()
          } else {
            last_ptr
          },
          if rem > 2 {
            parents[i + 2].as_ptr().cast()
          } else {
            last_ptr
          },
          if rem > 3 {
            parents[i + 3].as_ptr().cast()
          } else {
            last_ptr
          },
          if rem > 4 {
            parents[i + 4].as_ptr().cast()
          } else {
            last_ptr
          },
          if rem > 5 {
            parents[i + 5].as_ptr().cast()
          } else {
            last_ptr
          },
          if rem > 6 {
            parents[i + 6].as_ptr().cast()
          } else {
            last_ptr
          },
          last_ptr,
        ];

        let mut tmp = [0u8; super::x86_64::avx2::DEGREE * OUT_LEN];
        // SAFETY: AVX2 is available for this wrapper; pointers and outputs are valid.
        unsafe {
          super::x86_64::avx2::hash8(&ptrs, 1, &key_words, 0, false, parent_flags, 0, 0, tmp.as_mut_ptr());
        }

        for lane in 0..rem {
          // SAFETY: `lane < rem <= DEGREE`, so this is in-bounds.
          let bytes: &[u8; OUT_LEN] = unsafe { &*(tmp.as_ptr().add(lane * OUT_LEN).cast()) };
          out[i + lane] = words8_from_le_bytes_32(bytes);
        }
      }
      return;
    }
  }

  // Portable/scalar fallback.
  for (i, parent_words) in parents.iter().enumerate() {
    out[i] = first_8_words((super::compress)(
      &key_words,
      parent_words,
      0,
      BLOCK_LEN as u32,
      PARENT | flags,
    ));
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
    // Important: the upstream BLAKE3 asm backend only requires AVX-512 F + VL.
    //
    // However, our non-asm *intrinsics* backend uses AVX-512DQ-only lane
    // extract/insert helpers. On targets where we don't ship the asm backend,
    // we must require AVX-512DQ for correctness.
    Blake3KernelId::X86Avx512 => {
      #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
      {
        x86::AVX512F
          .union(x86::AVX512VL)
          .union(x86::AVX2)
          .union(x86::SSE41)
          .union(x86::SSSE3)
      }
      #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
      {
        x86::AVX512F
          .union(x86::AVX512VL)
          .union(x86::AVX512DQ)
          .union(x86::AVX2)
          .union(x86::SSE41)
          .union(x86::SSSE3)
      }
    }
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
// x86_64 AVX2 wrappers (single-block / streaming hot paths)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn compress_avx2_wrapper(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  // SAFETY: This function is only called when AVX2 is available (checked by dispatch).
  unsafe { super::x86_64::compress_avx2(chaining_value, block_words, counter, block_len, flags) }
}

#[cfg(target_arch = "x86_64")]
fn chunk_compress_blocks_avx2_wrapper(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  // SAFETY: This function is only called when AVX2 is available (checked by dispatch).
  unsafe { super::x86_64::chunk_compress_blocks_avx2(chaining_value, chunk_counter, flags, blocks_compressed, blocks) }
}

#[cfg(target_arch = "x86_64")]
fn parent_cv_avx2_wrapper(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  // SAFETY: This function is only called when AVX2 is available (checked by dispatch).
  unsafe { super::x86_64::parent_cv_avx2(left_child_cv, right_child_cv, key_words, flags) }
}

// ─────────────────────────────────────────────────────────────────────────────
// x86_64 AVX-512 wrappers (single-block / streaming hot paths)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn compress_avx512_wrapper(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  // SAFETY: This function is only called when AVX-512 is available (checked by dispatch).
  unsafe { super::x86_64::compress_avx512(chaining_value, block_words, counter, block_len, flags) }
}

#[cfg(target_arch = "x86_64")]
fn chunk_compress_blocks_avx512_wrapper(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  // SAFETY: This function is only called when AVX-512 is available (checked by dispatch).
  unsafe {
    super::x86_64::chunk_compress_blocks_avx512(chaining_value, chunk_counter, flags, blocks_compressed, blocks)
  }
}

#[cfg(target_arch = "x86_64")]
fn parent_cv_avx512_wrapper(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  // SAFETY: This function is only called when AVX-512 is available (checked by dispatch).
  unsafe { super::x86_64::parent_cv_avx512(left_child_cv, right_child_cv, key_words, flags) }
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
  unsafe { super::x86_64::compress_sse41(chaining_value, block_words, counter, block_len, flags) }
}

#[cfg(target_arch = "x86_64")]
fn chunk_compress_blocks_sse41_wrapper(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  // SAFETY: This function is only called when SSE4.1 is available (checked by dispatch).
  unsafe { super::x86_64::chunk_compress_blocks_sse41(chaining_value, chunk_counter, flags, blocks_compressed, blocks) }
}

#[cfg(target_arch = "x86_64")]
fn parent_cv_sse41_wrapper(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  // SAFETY: This function is only called when SSE4.1 is available (checked by dispatch).
  unsafe { super::x86_64::parent_cv_sse41(left_child_cv, right_child_cv, key_words, flags) }
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
    // For small tails (1–3 chunks), avoid falling back to portable. Duplicate
    // the final chunk pointer into unused lanes and only copy the needed
    // outputs.
    // SAFETY: `num_chunks != 0`, and `input` is valid for `num_chunks * CHUNK_LEN` bytes.
    let last = unsafe { input.add((num_chunks - 1) * CHUNK_LEN) };
    // SAFETY: all pointers are within the caller-provided `input` buffer.
    let ptrs = unsafe {
      [
        input,
        if num_chunks > 1 { input.add(CHUNK_LEN) } else { last },
        if num_chunks > 2 { input.add(2 * CHUNK_LEN) } else { last },
        last,
      ]
    };

    let mut tmp = [0u8; super::x86_64::sse41::DEGREE * OUT_LEN];
    // SAFETY: SSE4.1 is available for this wrapper, `ptrs` are in-bounds for
    // full chunks, and `tmp`/`out` are large enough for the copied outputs.
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
        tmp.as_mut_ptr(),
      );
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), out, num_chunks * OUT_LEN);
    }
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
  #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
  let mut processed_any = false;
  while num_chunks >= super::x86_64::avx2::DEGREE {
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
      processed_any = true;
    }
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
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    // SAFETY: this wrapper is only selected when AVX2 is available (checked by
    // dispatch), `ptrs` point to `DEGREE` in-bounds chunk inputs, and `out` is
    // valid for `DEGREE * OUT_LEN` bytes.
    unsafe {
      debug_assert!(flags <= u8::MAX as u32);
      super::x86_64::asm::rscrypto_blake3_hash_many_avx2(
        ptrs.as_ptr(),
        super::x86_64::avx2::DEGREE,
        CHUNK_LEN / BLOCK_LEN,
        key.as_ptr(),
        counter,
        true,
        flags as u8,
        CHUNK_START as u8,
        super::CHUNK_END as u8,
        out,
      );
      input = input.add(super::x86_64::avx2::DEGREE * CHUNK_LEN);
      out = out.add(super::x86_64::avx2::DEGREE * OUT_LEN);
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    unsafe {
      // SAFETY: dispatch selects this kernel only when AVX2 is available; the
      // caller guarantees `input`/`out` cover the full `num_chunks` buffer.
      // SAFETY: AVX2 is available for this wrapper, `ptrs` are in-bounds for
      // full chunks, and `out` is large enough for `DEGREE * OUT_LEN` bytes.
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
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    {
      debug_assert!(num_chunks < super::x86_64::avx2::DEGREE);
      debug_assert!(flags <= u8::MAX as u32);
      // Use the upstream-grade AVX2 asm `hash_many` backend for sub-degree
      // tails (1–7 chunks). Passing `num_inputs = num_chunks` avoids wasting
      // lanes (and avoids falling back to SSE4.1 on Intel/Zen5 where that is
      // measurably slower for mid-size streaming updates like 4KiB).
      //
      // SAFETY: This wrapper is only selected when AVX2 is available (checked
      // by dispatch). `input` is valid for `num_chunks * CHUNK_LEN` bytes, so
      // each `input.add(i * CHUNK_LEN)` stays in-bounds. `out` is valid for
      // `num_chunks * OUT_LEN` bytes.
      let mut ptrs = [input; super::x86_64::avx2::DEGREE];
      for (i, ptr) in ptrs.iter_mut().enumerate().take(num_chunks) {
        // SAFETY: `i < num_chunks` and the caller guarantees `input` is valid
        // for `num_chunks * CHUNK_LEN` bytes.
        *ptr = unsafe { input.add(i * CHUNK_LEN) };
      }
      // SAFETY: AVX2 is available for this kernel per dispatch. `ptrs` points
      // to `num_chunks` valid chunk inputs, and `out` is valid for
      // `num_chunks * OUT_LEN` bytes.
      unsafe {
        super::x86_64::asm::rscrypto_blake3_hash_many_avx2(
          ptrs.as_ptr(),
          num_chunks,
          CHUNK_LEN / BLOCK_LEN,
          key.as_ptr(),
          counter,
          true,
          flags as u8,
          CHUNK_START as u8,
          super::CHUNK_END as u8,
          out,
        );
      }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
      // For small totals (<= 4 chunks), avoid wasting AVX2 lanes by delegating
      // to the 4-way SSE4.1 backend. This is a key lever for closing gaps at
      // 2–4KiB oneshot sizes and mid-size streaming updates on non-Linux OSes.
      //
      // Only do this when we haven't executed any AVX2 code yet, to avoid
      // AVX->SSE transition penalties in the common "large input with tail"
      // case.
      if !processed_any && num_chunks <= super::x86_64::sse41::DEGREE {
        // SAFETY: AVX2 dispatch implies SSE4.1 + SSSE3 (see `required_caps`),
        // and `input`/`out` are valid for `num_chunks` full chunks.
        unsafe { hash_many_contiguous_sse41_wrapper(input, num_chunks, key, counter, flags, out) };
        return;
      }

      // Non-Linux fallback: hash an 8-lane batch with duplicated final pointers
      // and copy only the needed outputs.
      // SAFETY: `num_chunks != 0`, and `input` is valid for `num_chunks * CHUNK_LEN` bytes.
      let last = unsafe { input.add((num_chunks - 1) * CHUNK_LEN) };
      // SAFETY: all pointers are within the caller-provided `input` buffer.
      let ptrs = unsafe {
        [
          input,
          if num_chunks > 1 { input.add(CHUNK_LEN) } else { last },
          if num_chunks > 2 { input.add(2 * CHUNK_LEN) } else { last },
          if num_chunks > 3 { input.add(3 * CHUNK_LEN) } else { last },
          if num_chunks > 4 { input.add(4 * CHUNK_LEN) } else { last },
          if num_chunks > 5 { input.add(5 * CHUNK_LEN) } else { last },
          if num_chunks > 6 { input.add(6 * CHUNK_LEN) } else { last },
          last,
        ]
      };

      let mut tmp = [0u8; super::x86_64::avx2::DEGREE * OUT_LEN];
      // SAFETY: AVX2 is available for this wrapper, `ptrs` are in-bounds for
      // full chunks, and `tmp`/`out` are large enough for the copied outputs.
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
          tmp.as_mut_ptr(),
        );
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), out, num_chunks * OUT_LEN);
      }
    }
  }
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
  #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
  let mut processed_any = false;
  while num_chunks >= super::x86_64::avx512::DEGREE {
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
      processed_any = true;
    }
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    // SAFETY: this wrapper is only selected when AVX-512 is available (checked
    // by dispatch), the constructed `ptrs` all stay in-bounds for full chunks,
    // and `out` is valid for `DEGREE * OUT_LEN` bytes.
    unsafe {
      debug_assert!(flags <= u8::MAX as u32);
      let ptrs = [
        input,
        input.add(CHUNK_LEN),
        input.add(2 * CHUNK_LEN),
        input.add(3 * CHUNK_LEN),
        input.add(4 * CHUNK_LEN),
        input.add(5 * CHUNK_LEN),
        input.add(6 * CHUNK_LEN),
        input.add(7 * CHUNK_LEN),
        input.add(8 * CHUNK_LEN),
        input.add(9 * CHUNK_LEN),
        input.add(10 * CHUNK_LEN),
        input.add(11 * CHUNK_LEN),
        input.add(12 * CHUNK_LEN),
        input.add(13 * CHUNK_LEN),
        input.add(14 * CHUNK_LEN),
        input.add(15 * CHUNK_LEN),
      ];
      super::x86_64::asm::rscrypto_blake3_hash_many_avx512(
        ptrs.as_ptr(),
        super::x86_64::avx512::DEGREE,
        CHUNK_LEN / BLOCK_LEN,
        key.as_ptr(),
        counter,
        true,
        flags as u8,
        CHUNK_START as u8,
        super::CHUNK_END as u8,
        out,
      );
      input = input.add(super::x86_64::avx512::DEGREE * CHUNK_LEN);
      out = out.add(super::x86_64::avx512::DEGREE * OUT_LEN);
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    unsafe {
      // SAFETY: dispatch selects this kernel only when AVX-512 is available;
      // the caller guarantees `input`/`out` cover the full `num_chunks` buffer.
      super::x86_64::avx512::hash16_contiguous(input, key, counter, flags, out);
      input = input.add(super::x86_64::avx512::DEGREE * CHUNK_LEN);
      out = out.add(super::x86_64::avx512::DEGREE * OUT_LEN);
    }
    counter = counter.wrapping_add(super::x86_64::avx512::DEGREE as u64);
    num_chunks -= super::x86_64::avx512::DEGREE;
  }

  if num_chunks != 0 {
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    {
      debug_assert!(num_chunks < super::x86_64::avx512::DEGREE);
      debug_assert!(flags <= u8::MAX as u32);
      // Use the upstream-grade AVX-512 asm `hash_many` backend for the
      // sub-degree tail (1–15 chunks). Passing `num_inputs = num_chunks`
      // prevents an AVX-512 -> AVX2 -> SSE4.1 downgrade on short totals and
      // is a key lever for closing streaming gaps at 4KiB/16KiB update sizes
      // on Intel and Zen5-class CPUs.
      //
      // SAFETY: This wrapper is only selected when the AVX-512 kernel is
      // available per dispatch. `input`/`out` cover `num_chunks` full chunks.
      let mut ptrs = [input; super::x86_64::avx512::DEGREE];
      for (i, ptr) in ptrs.iter_mut().enumerate().take(num_chunks) {
        // SAFETY: `i < num_chunks` and the caller guarantees `input` is valid
        // for `num_chunks * CHUNK_LEN` bytes.
        *ptr = unsafe { input.add(i * CHUNK_LEN) };
      }
      // SAFETY: AVX-512 is available for this kernel per dispatch. `ptrs`
      // points to `num_chunks` valid chunk inputs, and `out` is valid for
      // `num_chunks * OUT_LEN` bytes.
      unsafe {
        super::x86_64::asm::rscrypto_blake3_hash_many_avx512(
          ptrs.as_ptr(),
          num_chunks,
          CHUNK_LEN / BLOCK_LEN,
          key.as_ptr(),
          counter,
          true,
          flags as u8,
          CHUNK_START as u8,
          super::CHUNK_END as u8,
          out,
        );
      }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
      if !processed_any {
        // For small totals (e.g. <= 15 chunks), delegate to the AVX2 wrapper.
        // It will pick the best strategy for < 8 chunks without wasting lanes.
        //
        // SAFETY: the remaining chunk pointers are in-bounds for the caller's
        // `input`/`out` buffer, and AVX2 is available when AVX-512 is.
        unsafe { hash_many_contiguous_avx2_wrapper(input, num_chunks, key, counter, flags, out) };
        return;
      }

      // We've already executed AVX-512 code. Keep the tail in AVX2 (which is
      // always available with AVX-512) to avoid AVX->SSE transition penalties,
      // and because this tail is at most 15 chunks.
      if num_chunks >= super::x86_64::avx2::DEGREE {
        // SAFETY: `num_chunks >= 8` means there are at least 8 full chunks
        // remaining, all within the caller-provided buffer.
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
        // SAFETY: AVX2 is available when AVX-512 is; `ptrs` are in-bounds for
        // full chunks; and `out` is large enough for `8 * OUT_LEN` bytes.
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
        }
        // SAFETY: we just hashed exactly `DEGREE` full chunks, and the caller
        // guarantees the overall `input`/`out` buffers are large enough.
        input = unsafe { input.add(super::x86_64::avx2::DEGREE * CHUNK_LEN) };
        // SAFETY: we just wrote exactly `DEGREE` outputs, and the caller
        // guarantees the overall `out` buffer is large enough.
        out = unsafe { out.add(super::x86_64::avx2::DEGREE * OUT_LEN) };
        counter = counter.wrapping_add(super::x86_64::avx2::DEGREE as u64);
        num_chunks -= super::x86_64::avx2::DEGREE;
      }

      if num_chunks != 0 {
        // For tails (1–7 chunks), hash an 8-lane batch with duplicated final
        // pointers and copy only the needed outputs.
        // SAFETY: `num_chunks != 0`, and `input` is valid for `num_chunks * CHUNK_LEN` bytes.
        let last = unsafe { input.add((num_chunks - 1) * CHUNK_LEN) };
        // SAFETY: all pointers are within the caller-provided `input` buffer.
        let ptrs = unsafe {
          [
            input,
            if num_chunks > 1 { input.add(CHUNK_LEN) } else { last },
            if num_chunks > 2 { input.add(2 * CHUNK_LEN) } else { last },
            if num_chunks > 3 { input.add(3 * CHUNK_LEN) } else { last },
            if num_chunks > 4 { input.add(4 * CHUNK_LEN) } else { last },
            if num_chunks > 5 { input.add(5 * CHUNK_LEN) } else { last },
            if num_chunks > 6 { input.add(6 * CHUNK_LEN) } else { last },
            last,
          ]
        };

        let mut tmp = [0u8; super::x86_64::avx2::DEGREE * OUT_LEN];
        // SAFETY: AVX2 is available here, `ptrs` are in-bounds for full chunks,
        // and `tmp`/`out` are large enough for the copied outputs.
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
            tmp.as_mut_ptr(),
          );
          core::ptr::copy_nonoverlapping(tmp.as_ptr(), out, num_chunks * OUT_LEN);
        }
      }
    }
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
