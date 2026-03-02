use platform::Caps;
#[cfg(target_arch = "aarch64")]
use platform::caps::aarch64;
#[cfg(target_arch = "powerpc64")]
use platform::caps::power;
#[cfg(target_arch = "riscv64")]
use platform::caps::riscv;
#[cfg(target_arch = "s390x")]
use platform::caps::s390x;
#[cfg(target_arch = "x86_64")]
use platform::caps::x86;

use super::{
  BLOCK_LEN, CHUNK_LEN, CHUNK_START, OUT_LEN, PARENT, first_8_words, words8_from_le_bytes_32, words16_from_le_bytes_64,
};

// ─────────────────────────────────────────────────────────────────────────────
// Kernel function types
// ─────────────────────────────────────────────────────────────────────────────

/// Core compression function (single block).
pub(crate) type CompressFn = fn(&[u32; 8], &[u32; 16], u64, u32, u32) -> [u32; 16];

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

/// Compress one or more full 64-byte blocks into a chaining value.
pub(crate) type ChunkCompressBlocksFn = fn(&mut [u32; 8], u64, u32, &mut u8, &[u8]);

/// x86-only final-block compressor from a raw 64-byte pointer.
#[cfg(target_arch = "x86_64")]
pub(crate) type X86CompressCvBytesFn = unsafe fn(&[u32; 8], *const u8, u64, u32, u32) -> [u32; 8];

// ─────────────────────────────────────────────────────────────────────────────
// Kernel struct
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub(crate) struct Kernel {
  pub(crate) id: Blake3KernelId,
  /// Compress single block (for streaming/small inputs).
  pub(crate) compress: CompressFn,
  /// Compress full blocks (used by tiny/oneshot paths to avoid per-call id dispatch).
  pub(crate) chunk_compress_blocks: ChunkCompressBlocksFn,
  /// Hash many contiguous full chunks (for throughput, no pointer chasing).
  pub(crate) hash_many_contiguous: HashManyContiguousFn,
  /// x86-only final-block compressor from bytes.
  #[cfg(target_arch = "x86_64")]
  pub(crate) x86_compress_cv_bytes: X86CompressCvBytesFn,
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
  #[cfg(target_arch = "s390x")]
  S390xVector = 6,
  #[cfg(target_arch = "powerpc64")]
  PowerVsx = 7,
  #[cfg(target_arch = "riscv64")]
  RiscvV = 8,
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
  #[cfg(target_arch = "s390x")]
  Blake3KernelId::S390xVector,
  #[cfg(target_arch = "powerpc64")]
  Blake3KernelId::PowerVsx,
  #[cfg(target_arch = "riscv64")]
  Blake3KernelId::RiscvV,
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
      #[cfg(target_arch = "s390x")]
      Self::S390xVector => "s390x/vector",
      #[cfg(target_arch = "powerpc64")]
      Self::PowerVsx => "powerpc64/vsx",
      #[cfg(target_arch = "riscv64")]
      Self::RiscvV => "riscv64/v",
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
      #[cfg(target_arch = "s390x")]
      Self::S390xVector => 4,
      #[cfg(target_arch = "powerpc64")]
      Self::PowerVsx => 4,
      #[cfg(target_arch = "riscv64")]
      Self::RiscvV => 4,
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
      hash_many_contiguous: hash_many_contiguous_portable,
      #[cfg(target_arch = "x86_64")]
      x86_compress_cv_bytes: x86_compress_cv_portable_wrapper,
      simd_degree: 1,
      name: id.as_str(),
    },
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Ssse3 => Kernel {
      id,
      compress: compress_ssse3_wrapper,
      chunk_compress_blocks: chunk_compress_blocks_ssse3_wrapper,
      hash_many_contiguous: hash_many_contiguous_ssse3_wrapper,
      x86_compress_cv_bytes: x86_compress_cv_portable_wrapper,
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
      hash_many_contiguous: hash_many_contiguous_sse41_wrapper,
      x86_compress_cv_bytes: x86_compress_cv_sse41_wrapper,
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
      hash_many_contiguous: hash_many_contiguous_avx2_wrapper,
      x86_compress_cv_bytes: x86_compress_cv_avx2_wrapper,
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
      hash_many_contiguous: hash_many_contiguous_avx512_wrapper,
      x86_compress_cv_bytes: x86_compress_cv_avx512_wrapper,
      simd_degree: 16,
      name: id.as_str(),
    },
    #[cfg(target_arch = "aarch64")]
    Blake3KernelId::Aarch64Neon => Kernel {
      id,
      // On aarch64, the 4-way NEON `hash_many`/`hash_many_contiguous` kernels
      // are the throughput workhorses. We also need a NEON single-block
      // compressor to avoid a "last block is scalar" cliff on 1-chunk inputs
      // (e.g. 1024B oneshot) and XOF/output-generation workloads.
      compress: compress_neon_wrapper,
      chunk_compress_blocks: chunk_compress_blocks_neon_wrapper,
      hash_many_contiguous: hash_many_contiguous_neon_wrapper,
      simd_degree: 4,
      name: id.as_str(),
    },
    #[cfg(target_arch = "s390x")]
    Blake3KernelId::S390xVector => Kernel {
      id,
      compress: compress_s390x_vector_wrapper,
      chunk_compress_blocks: chunk_compress_blocks_s390x_vector_wrapper,
      hash_many_contiguous: hash_many_contiguous_s390x_vector_wrapper,
      simd_degree: 4,
      name: id.as_str(),
    },
    #[cfg(target_arch = "powerpc64")]
    Blake3KernelId::PowerVsx => Kernel {
      id,
      compress: compress_power_vsx_wrapper,
      chunk_compress_blocks: chunk_compress_blocks_power_vsx_wrapper,
      hash_many_contiguous: hash_many_contiguous_power_vsx_wrapper,
      simd_degree: 4,
      name: id.as_str(),
    },
    #[cfg(target_arch = "riscv64")]
    Blake3KernelId::RiscvV => Kernel {
      id,
      compress: compress_riscv_v_wrapper,
      chunk_compress_blocks: chunk_compress_blocks_riscv_v_wrapper,
      hash_many_contiguous: hash_many_contiguous_riscv_v_wrapper,
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
    #[cfg(target_arch = "s390x")]
    Blake3KernelId::S390xVector => {
      chunk_compress_blocks_s390x_vector_wrapper(chaining_value, chunk_counter, flags, blocks_compressed, blocks)
    }
    #[cfg(target_arch = "powerpc64")]
    Blake3KernelId::PowerVsx => {
      chunk_compress_blocks_power_vsx_wrapper(chaining_value, chunk_counter, flags, blocks_compressed, blocks)
    }
    #[cfg(target_arch = "riscv64")]
    Blake3KernelId::RiscvV => {
      chunk_compress_blocks_riscv_v_wrapper(chaining_value, chunk_counter, flags, blocks_compressed, blocks)
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
    Blake3KernelId::Aarch64Neon => parent_cv_neon_wrapper(left_child_cv, right_child_cv, key_words, flags),
    #[cfg(target_arch = "s390x")]
    Blake3KernelId::S390xVector => parent_cv_s390x_vector_wrapper(left_child_cv, right_child_cv, key_words, flags),
    #[cfg(target_arch = "powerpc64")]
    Blake3KernelId::PowerVsx => parent_cv_power_vsx_wrapper(left_child_cv, right_child_cv, key_words, flags),
    #[cfg(target_arch = "riscv64")]
    Blake3KernelId::RiscvV => parent_cv_riscv_v_wrapper(left_child_cv, right_child_cv, key_words, flags),
  }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn parent_block_ptrs<const DEGREE: usize, PtrAt>(
  start: usize,
  rem: usize,
  total: usize,
  ptr_at: &mut PtrAt,
) -> [*const u8; DEGREE]
where
  PtrAt: FnMut(usize) -> *const u8,
{
  debug_assert!(rem != 0);
  debug_assert!(rem <= DEGREE);
  let last_ptr = ptr_at(total - 1);
  let mut ptrs = [last_ptr; DEGREE];
  let mut lane = 0usize;
  while lane < rem {
    ptrs[lane] = ptr_at(start + lane);
    lane += 1;
  }
  ptrs
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn reduce_parent_blocks_lanes<const DEGREE: usize, PtrAt, HashMany, Sink>(
  count: usize,
  mut ptr_at: PtrAt,
  mut hash_many: HashMany,
  mut sink: Sink,
) where
  PtrAt: FnMut(usize) -> *const u8,
  HashMany: FnMut(&[*const u8; DEGREE], usize, &mut [[u8; OUT_LEN]; DEGREE]),
  Sink: FnMut(usize, &[u8; OUT_LEN]),
{
  let mut tmp = [[0u8; OUT_LEN]; DEGREE];
  let mut i = 0usize;
  while i < count {
    let rem = core::cmp::min(DEGREE, count - i);
    let ptrs = parent_block_ptrs::<DEGREE, _>(i, rem, count, &mut ptr_at);
    hash_many(&ptrs, rem, &mut tmp);
    let mut lane = 0usize;
    while lane < rem {
      sink(i + lane, &tmp[lane]);
      lane += 1;
    }
    i += rem;
  }
}

/// Compute many parent CVs from adjacent child CV pairs.
///
/// `children` is interpreted as `[left0, right0, left1, right1, ...]`.
#[inline]
pub(crate) fn parent_cvs_many_from_cvs_inline(
  id: Blake3KernelId,
  children: &[[u32; 8]],
  key_words: [u32; 8],
  flags: u32,
  out: &mut [[u32; 8]],
) {
  debug_assert_eq!(children.len(), out.len() * 2);
  if out.is_empty() {
    return;
  }

  if id == Blake3KernelId::Portable {
    for (pair, out_cv) in children.chunks_exact(2).zip(out.iter_mut()) {
      *out_cv = parent_cv_inline(id, pair[0], pair[1], key_words, flags);
    }
    return;
  }

  #[cfg(target_endian = "little")]
  {
    // Fast path on little-endian targets: SIMD parent kernels naturally produce
    // little-endian bytes; reinterpreting avoids bytes<->words churn.
    // SAFETY: `[u32; 8]` and `[u8; OUT_LEN]` are layout-compatible (32 bytes),
    // and both slices keep identical element counts.
    let children_bytes: &[[u8; OUT_LEN]] =
      unsafe { core::slice::from_raw_parts(children.as_ptr().cast(), children.len()) };
    // SAFETY: same layout argument as above for mutable outputs.
    let out_bytes: &mut [[u8; OUT_LEN]] =
      unsafe { core::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), out.len()) };
    parent_cvs_many_from_bytes_inline(id, children_bytes, key_words, flags, out_bytes);
  }

  #[cfg(not(target_endian = "little"))]
  {
    // Big-endian fallback keeps explicit LE conversion.
    for (pair, out_cv) in children.chunks_exact(2).zip(out.iter_mut()) {
      *out_cv = parent_cv_inline(id, pair[0], pair[1], key_words, flags);
    }
  }
}

/// Compute many parent CVs from adjacent child CV pairs (byte representation).
///
/// `children` is interpreted as `[left0, right0, left1, right1, ...]`.
///
/// This is the preferred representation for bulk hashing, because it avoids
/// repeated bytes<->words conversions when interfacing with the `hash_many`
/// primitives (which natively operate on bytes).
#[inline]
pub(crate) fn parent_cvs_many_from_bytes_inline(
  id: Blake3KernelId,
  children: &[[u8; OUT_LEN]],
  key_words: [u32; 8],
  flags: u32,
  out: &mut [[u8; OUT_LEN]],
) {
  debug_assert_eq!(children.len(), out.len() * 2);
  if out.is_empty() {
    return;
  }

  match id {
    Blake3KernelId::Portable => {}
    #[cfg(target_arch = "aarch64")]
    Blake3KernelId::Aarch64Neon => {
      // SAFETY: NEON is available per dispatch; this helper handles all tail
      // cases without alignment constraints.
      unsafe { super::aarch64::parent_cvs_many_neon(children, key_words, flags, out) };
      return;
    }
    #[cfg(target_arch = "s390x")]
    Blake3KernelId::S390xVector => {
      parent_cvs_many4_simd(children, key_words, flags, out);
      return;
    }
    #[cfg(target_arch = "powerpc64")]
    Blake3KernelId::PowerVsx => {
      parent_cvs_many4_simd(children, key_words, flags, out);
      return;
    }
    #[cfg(target_arch = "riscv64")]
    Blake3KernelId::RiscvV => {
      parent_cvs_many4_simd(children, key_words, flags, out);
      return;
    }
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Ssse3 => {}
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Sse41 => {
      let parent_flags = PARENT | flags;
      reduce_parent_blocks_lanes::<{ super::x86_64::sse41::DEGREE }, _, _, _>(
        out.len(),
        |idx| children[2 * idx].as_ptr(),
        |ptrs, _rem, tmp| {
          // SAFETY: SSE4.1 is available per dispatch; pointers and outputs are valid.
          unsafe {
            super::x86_64::sse41::hash4(
              ptrs,
              1,
              &key_words,
              0,
              false,
              parent_flags,
              0,
              0,
              tmp.as_mut_ptr().cast::<u8>(),
            );
          }
        },
        |idx, bytes| out[idx] = *bytes,
      );
      return;
    }
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx512 => {
      let parent_flags = PARENT | flags;

      #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
      {
        const DEGREE: usize = 16;
        debug_assert!(parent_flags <= u8::MAX as u32);
        reduce_parent_blocks_lanes::<DEGREE, _, _, _>(
          out.len(),
          |idx| children[2 * idx].as_ptr(),
          |ptrs, rem, tmp| {
            // SAFETY: AVX-512 is available per dispatch; pointers are valid for
            // one parent block each, `rem <= DEGREE`, and output is large enough.
            unsafe {
              super::x86_64::asm::hash_many_avx512(
                ptrs.as_ptr(),
                rem,
                1,
                key_words.as_ptr(),
                0,
                false,
                parent_flags as u8,
                0,
                0,
                tmp.as_mut_ptr().cast::<u8>(),
              );
            }
          },
          |idx, bytes| out[idx] = *bytes,
        );
        return;
      }

      #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
      {
        // Keep AVX-512 semantics on non-asm targets as well: run the parent
        // fold through the AVX-512 per-parent entrypoint instead of delegating
        // into AVX2.
        for (pair, out_cv) in children.chunks_exact(2).zip(out.iter_mut()) {
          let left = words8_from_le_bytes_32(&pair[0]);
          let right = words8_from_le_bytes_32(&pair[1]);
          *out_cv = super::words8_to_le_bytes(&parent_cv_inline(
            Blake3KernelId::X86Avx512,
            left,
            right,
            key_words,
            flags,
          ));
        }
        return;
      }
    }
    #[cfg(target_arch = "x86_64")]
    Blake3KernelId::X86Avx2 => {
      let parent_flags = PARENT | flags;
      reduce_parent_blocks_lanes::<{ super::x86_64::avx2::DEGREE }, _, _, _>(
        out.len(),
        |idx| children[2 * idx].as_ptr(),
        |ptrs, rem, tmp| {
          #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
          {
            debug_assert!(parent_flags <= u8::MAX as u32);
            // SAFETY: `id` is AVX2, so required AVX2 features are present per
            // dispatch. Each pointer is valid for one 64-byte parent block.
            // `rem <= DEGREE`, and `tmp` is large enough.
            unsafe {
              super::x86_64::asm::hash_many_avx2(
                ptrs.as_ptr(),
                rem,
                1,
                key_words.as_ptr(),
                0,
                false,
                parent_flags as u8,
                0,
                0,
                tmp.as_mut_ptr().cast::<u8>(),
              );
            }
          }

          #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
          {
            // SAFETY: AVX2 is available for this wrapper; pointers and outputs are valid.
            unsafe {
              super::x86_64::avx2::hash8(
                ptrs,
                1,
                &key_words,
                0,
                false,
                parent_flags,
                0,
                0,
                tmp.as_mut_ptr().cast::<u8>(),
              )
            };
          }
        },
        |idx, bytes| out[idx] = *bytes,
      );
      return;
    }
  }

  // Scalar fallback.
  for (pair, out_cv) in children.chunks_exact(2).zip(out.iter_mut()) {
    let left = words8_from_le_bytes_32(&pair[0]);
    let right = words8_from_le_bytes_32(&pair[1]);
    *out_cv = super::words8_to_le_bytes(&parent_cv_inline(id, left, right, key_words, flags));
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
  // `[u32; 16]` is exactly two adjacent `[u32; 8]` children.
  // SAFETY: each `[u32; 16]` element is contiguous and exactly two `[u32; 8]`
  // elements, so the cast preserves length and alignment invariants.
  let children: &[[u32; 8]] = unsafe { core::slice::from_raw_parts(parents.as_ptr().cast(), parents.len() * 2) };
  parent_cvs_many_from_cvs_inline(id, children, key_words, flags, out);
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
    #[cfg(target_arch = "s390x")]
    Blake3KernelId::S390xVector => s390x::VECTOR,
    #[cfg(target_arch = "powerpc64")]
    Blake3KernelId::PowerVsx => power::VSX,
    #[cfg(target_arch = "riscv64")]
    Blake3KernelId::RiscvV => riscv::V,
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
      // SAFETY: caller guarantees `input` is valid for `num_chunks * CHUNK_LEN`.
      let block_words = unsafe {
        let src = input.add(chunk_idx * CHUNK_LEN + block_idx * BLOCK_LEN);
        // SAFETY:
        // - Caller guarantees `src` is valid for `BLOCK_LEN` bytes.
        // - `[u8; BLOCK_LEN]` has alignment 1, so this reference doesn't assume any alignment of the input
        //   pointer.
        words16_from_le_bytes_64(&*src.cast::<[u8; BLOCK_LEN]>())
      };

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

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64", target_arch = "riscv64"))]
#[inline(always)]
unsafe fn write_cv_words(out: *mut u8, cv: &[u32; 8]) {
  for (j, &word) in cv.iter().enumerate() {
    let bytes = word.to_le_bytes();
    // SAFETY: caller guarantees one full CV output is writable at `out`.
    unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), out.add(j * 4), 4) };
  }
}

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64", target_arch = "riscv64"))]
#[inline(always)]
fn rot_lanes_left_1(v: core::simd::u32x4) -> core::simd::u32x4 {
  core::simd::simd_swizzle!(v, [1, 2, 3, 0])
}

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64", target_arch = "riscv64"))]
#[inline(always)]
fn rot_lanes_left_2(v: core::simd::u32x4) -> core::simd::u32x4 {
  core::simd::simd_swizzle!(v, [2, 3, 0, 1])
}

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64", target_arch = "riscv64"))]
#[inline(always)]
fn rot_lanes_left_3(v: core::simd::u32x4) -> core::simd::u32x4 {
  core::simd::simd_swizzle!(v, [3, 0, 1, 2])
}

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64", target_arch = "riscv64"))]
#[inline(always)]
fn rotr32<const N: u32>(v: core::simd::u32x4) -> core::simd::u32x4 {
  debug_assert!(N > 0 && N < 32);
  let s0 = core::simd::u32x4::splat(N);
  let s1 = core::simd::u32x4::splat(32 - N);
  (v >> s0) | (v << s1)
}

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64", target_arch = "riscv64"))]
#[inline(always)]
fn load_msg_vec_const<const I0: usize, const I1: usize, const I2: usize, const I3: usize>(
  block_words: &[u32; 16],
) -> core::simd::u32x4 {
  const {
    assert!(I0 < 16);
    assert!(I1 < 16);
    assert!(I2 < 16);
    assert!(I3 < 16);
  }
  core::simd::u32x4::from_array([block_words[I0], block_words[I1], block_words[I2], block_words[I3]])
}

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64", target_arch = "riscv64"))]
const MSG_SCHEDULE_SIMD: [[usize; 16]; 7] = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
  [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
  [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
  [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
  [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
  [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64", target_arch = "riscv64"))]
#[inline(always)]
fn compress_simd_leaf(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  type Vec4 = core::simd::u32x4;

  let mut row0 = Vec4::from_array([
    chaining_value[0],
    chaining_value[1],
    chaining_value[2],
    chaining_value[3],
  ]);
  let mut row1 = Vec4::from_array([
    chaining_value[4],
    chaining_value[5],
    chaining_value[6],
    chaining_value[7],
  ]);
  let mut row2 = Vec4::from_array([super::IV[0], super::IV[1], super::IV[2], super::IV[3]]);
  let mut row3 = Vec4::from_array([counter as u32, (counter >> 32) as u32, block_len, flags]);

  macro_rules! g {
    ($mx:expr, $my:expr) => {{
      row0 += row1;
      row0 += $mx;
      row3 ^= row0;
      row3 = rotr32::<16>(row3);
      row2 += row3;
      row1 ^= row2;
      row1 = rotr32::<12>(row1);
      row0 += row1;
      row0 += $my;
      row3 ^= row0;
      row3 = rotr32::<8>(row3);
      row2 += row3;
      row1 ^= row2;
      row1 = rotr32::<7>(row1);
    }};
  }

  macro_rules! round {
    (
      $c0:literal, $c1:literal, $c2:literal, $c3:literal, $c4:literal, $c5:literal, $c6:literal, $c7:literal,
      $d0:literal, $d1:literal, $d2:literal, $d3:literal, $d4:literal, $d5:literal, $d6:literal, $d7:literal
    ) => {{
      let mx0 = load_msg_vec_const::<$c0, $c2, $c4, $c6>(block_words);
      let my0 = load_msg_vec_const::<$c1, $c3, $c5, $c7>(block_words);
      let mx1 = load_msg_vec_const::<$d0, $d2, $d4, $d6>(block_words);
      let my1 = load_msg_vec_const::<$d1, $d3, $d5, $d7>(block_words);
      g!(mx0, my0);
      row1 = rot_lanes_left_1(row1);
      row2 = rot_lanes_left_2(row2);
      row3 = rot_lanes_left_3(row3);
      g!(mx1, my1);
      row1 = rot_lanes_left_3(row1);
      row2 = rot_lanes_left_2(row2);
      row3 = rot_lanes_left_1(row3);
    }};
  }

  round!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  round!(2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8);
  round!(3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1);
  round!(10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6);
  round!(12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4);
  round!(9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7);
  round!(11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13);

  let cv_lo = Vec4::from_array([
    chaining_value[0],
    chaining_value[1],
    chaining_value[2],
    chaining_value[3],
  ]);
  let cv_hi = Vec4::from_array([
    chaining_value[4],
    chaining_value[5],
    chaining_value[6],
    chaining_value[7],
  ]);
  row0 ^= row2;
  row1 ^= row3;
  row2 ^= cv_lo;
  row3 ^= cv_hi;

  let mut out = [0u32; 16];
  out[..4].copy_from_slice(&row0.to_array());
  out[4..8].copy_from_slice(&row1.to_array());
  out[8..12].copy_from_slice(&row2.to_array());
  out[12..16].copy_from_slice(&row3.to_array());
  out
}

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64", target_arch = "riscv64"))]
#[inline(always)]
fn load_msg_lanes4_contiguous(base: *const u8, block_offset: usize) -> [core::simd::u32x4; 16] {
  // SAFETY: caller provides 4 contiguous full chunks; each block load is in-bounds.
  let b0 = unsafe { words16_from_le_bytes_64(&*base.add(block_offset).cast::<[u8; BLOCK_LEN]>()) };
  // SAFETY: see above.
  let b1 = unsafe { words16_from_le_bytes_64(&*base.add(CHUNK_LEN + block_offset).cast::<[u8; BLOCK_LEN]>()) };
  // SAFETY: see above.
  let b2 = unsafe { words16_from_le_bytes_64(&*base.add(2 * CHUNK_LEN + block_offset).cast::<[u8; BLOCK_LEN]>()) };
  // SAFETY: see above.
  let b3 = unsafe { words16_from_le_bytes_64(&*base.add(3 * CHUNK_LEN + block_offset).cast::<[u8; BLOCK_LEN]>()) };
  let mut out = [core::simd::u32x4::splat(0); 16];
  let mut i = 0usize;
  while i < 16 {
    out[i] = core::simd::u32x4::from_array([b0[i], b1[i], b2[i], b3[i]]);
    i += 1;
  }
  out
}

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64", target_arch = "riscv64"))]
#[inline(always)]
fn g4_simd(
  v: &mut [core::simd::u32x4; 16],
  a: usize,
  b: usize,
  c: usize,
  d: usize,
  mx: core::simd::u32x4,
  my: core::simd::u32x4,
) {
  v[a] += v[b];
  v[a] += mx;
  v[d] ^= v[a];
  v[d] = rotr32::<16>(v[d]);
  v[c] += v[d];
  v[b] ^= v[c];
  v[b] = rotr32::<12>(v[b]);
  v[a] += v[b];
  v[a] += my;
  v[d] ^= v[a];
  v[d] = rotr32::<8>(v[d]);
  v[c] += v[d];
  v[b] ^= v[c];
  v[b] = rotr32::<7>(v[b]);
}

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64", target_arch = "riscv64"))]
#[inline(always)]
fn round4_simd(v: &mut [core::simd::u32x4; 16], m: &[core::simd::u32x4; 16], r: usize) {
  let s = &MSG_SCHEDULE_SIMD[r];
  g4_simd(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
  g4_simd(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
  g4_simd(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
  g4_simd(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
  g4_simd(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
  g4_simd(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
  g4_simd(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
  g4_simd(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
}

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64", target_arch = "riscv64"))]
#[inline(always)]
fn load_parent_msg_lanes4(children: &[[u8; OUT_LEN]], base: usize, rem: usize) -> [core::simd::u32x4; 16] {
  type Vec4 = core::simd::u32x4;
  debug_assert!(rem > 0 && rem <= 4);
  let last = base + rem - 1;
  let idx0 = base;
  let idx1 = if rem > 1 { base + 1 } else { last };
  let idx2 = if rem > 2 { base + 2 } else { last };
  let idx3 = if rem > 3 { base + 3 } else { last };

  let l0 = words8_from_le_bytes_32(&children[2 * idx0]);
  let r0 = words8_from_le_bytes_32(&children[2 * idx0 + 1]);
  let l1 = words8_from_le_bytes_32(&children[2 * idx1]);
  let r1 = words8_from_le_bytes_32(&children[2 * idx1 + 1]);
  let l2 = words8_from_le_bytes_32(&children[2 * idx2]);
  let r2 = words8_from_le_bytes_32(&children[2 * idx2 + 1]);
  let l3 = words8_from_le_bytes_32(&children[2 * idx3]);
  let r3 = words8_from_le_bytes_32(&children[2 * idx3 + 1]);

  let mut msg = [Vec4::splat(0); 16];
  let mut w = 0usize;
  while w < 8 {
    msg[w] = Vec4::from_array([l0[w], l1[w], l2[w], l3[w]]);
    msg[w + 8] = Vec4::from_array([r0[w], r1[w], r2[w], r3[w]]);
    w += 1;
  }
  msg
}

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64", target_arch = "riscv64"))]
#[inline(always)]
fn parent_cvs_many4_simd(children: &[[u8; OUT_LEN]], key_words: [u32; 8], flags: u32, out: &mut [[u8; OUT_LEN]]) {
  type Vec4 = core::simd::u32x4;
  debug_assert_eq!(children.len(), out.len() * 2);
  if out.is_empty() {
    return;
  }
  let parent_flags = PARENT | flags;

  let mut i = 0usize;
  while i < out.len() {
    let rem = core::cmp::min(4, out.len() - i);
    let msg = load_parent_msg_lanes4(children, i, rem);

    let mut v = [
      Vec4::splat(key_words[0]),
      Vec4::splat(key_words[1]),
      Vec4::splat(key_words[2]),
      Vec4::splat(key_words[3]),
      Vec4::splat(key_words[4]),
      Vec4::splat(key_words[5]),
      Vec4::splat(key_words[6]),
      Vec4::splat(key_words[7]),
      Vec4::splat(super::IV[0]),
      Vec4::splat(super::IV[1]),
      Vec4::splat(super::IV[2]),
      Vec4::splat(super::IV[3]),
      Vec4::splat(0),
      Vec4::splat(0),
      Vec4::splat(BLOCK_LEN as u32),
      Vec4::splat(parent_flags),
    ];

    round4_simd(&mut v, &msg, 0);
    round4_simd(&mut v, &msg, 1);
    round4_simd(&mut v, &msg, 2);
    round4_simd(&mut v, &msg, 3);
    round4_simd(&mut v, &msg, 4);
    round4_simd(&mut v, &msg, 5);
    round4_simd(&mut v, &msg, 6);

    let mut x = [Vec4::splat(0); 8];
    let mut j = 0usize;
    while j < 8 {
      x[j] = v[j] ^ v[j + 8];
      j += 1;
    }
    let xa0 = x[0].to_array();
    let xa1 = x[1].to_array();
    let xa2 = x[2].to_array();
    let xa3 = x[3].to_array();
    let xa4 = x[4].to_array();
    let xa5 = x[5].to_array();
    let xa6 = x[6].to_array();
    let xa7 = x[7].to_array();

    let mut lane = 0usize;
    while lane < rem {
      out[i + lane] = super::words8_to_le_bytes(&[
        xa0[lane], xa1[lane], xa2[lane], xa3[lane], xa4[lane], xa5[lane], xa6[lane], xa7[lane],
      ]);
      lane += 1;
    }
    i += rem;
  }
}

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64", target_arch = "riscv64"))]
#[inline(always)]
unsafe fn hash4_contiguous_full_chunks_simd(input: *const u8, key: &[u32; 8], counter: u64, flags: u32, out: *mut u8) {
  type Vec4 = core::simd::u32x4;
  let mut h = [
    Vec4::splat(key[0]),
    Vec4::splat(key[1]),
    Vec4::splat(key[2]),
    Vec4::splat(key[3]),
    Vec4::splat(key[4]),
    Vec4::splat(key[5]),
    Vec4::splat(key[6]),
    Vec4::splat(key[7]),
  ];
  let counter_low = Vec4::from_array([
    counter as u32,
    counter.wrapping_add(1) as u32,
    counter.wrapping_add(2) as u32,
    counter.wrapping_add(3) as u32,
  ]);
  let counter_high = Vec4::from_array([
    (counter >> 32) as u32,
    (counter.wrapping_add(1) >> 32) as u32,
    (counter.wrapping_add(2) >> 32) as u32,
    (counter.wrapping_add(3) >> 32) as u32,
  ]);

  let mut block_idx = 0usize;
  while block_idx < (CHUNK_LEN / BLOCK_LEN) {
    let msg = load_msg_lanes4_contiguous(input, block_idx * BLOCK_LEN);
    let block_flags = flags
      | if block_idx == 0 { CHUNK_START } else { 0 }
      | if block_idx + 1 == (CHUNK_LEN / BLOCK_LEN) {
        super::CHUNK_END
      } else {
        0
      };
    let mut v = [
      h[0],
      h[1],
      h[2],
      h[3],
      h[4],
      h[5],
      h[6],
      h[7],
      Vec4::splat(super::IV[0]),
      Vec4::splat(super::IV[1]),
      Vec4::splat(super::IV[2]),
      Vec4::splat(super::IV[3]),
      counter_low,
      counter_high,
      Vec4::splat(BLOCK_LEN as u32),
      Vec4::splat(block_flags),
    ];
    round4_simd(&mut v, &msg, 0);
    round4_simd(&mut v, &msg, 1);
    round4_simd(&mut v, &msg, 2);
    round4_simd(&mut v, &msg, 3);
    round4_simd(&mut v, &msg, 4);
    round4_simd(&mut v, &msg, 5);
    round4_simd(&mut v, &msg, 6);
    let mut i = 0usize;
    while i < 8 {
      h[i] = v[i] ^ v[i + 8];
      i += 1;
    }
    block_idx += 1;
  }

  let h0 = h[0].to_array();
  let h1 = h[1].to_array();
  let h2 = h[2].to_array();
  let h3 = h[3].to_array();
  let h4 = h[4].to_array();
  let h5 = h[5].to_array();
  let h6 = h[6].to_array();
  let h7 = h[7].to_array();

  let mut lane = 0usize;
  while lane < 4 {
    // SAFETY: caller guarantees `out` is valid for 4 contiguous `OUT_LEN` outputs.
    let dst = unsafe { out.add(lane * OUT_LEN) };
    let words = [
      h0[lane], h1[lane], h2[lane], h3[lane], h4[lane], h5[lane], h6[lane], h7[lane],
    ];
    let mut word = 0usize;
    while word < 8 {
      let bytes = words[word].to_le_bytes();
      // SAFETY: caller guarantees out is valid for 4 CV outputs.
      unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), dst.add(word * 4), 4) };
      word += 1;
    }
    lane += 1;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// s390x vector wrappers
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "s390x")]
#[target_feature(enable = "vector")]
unsafe fn compress_s390x_vector(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  compress_simd_leaf(chaining_value, block_words, counter, block_len, flags)
}

#[cfg(target_arch = "s390x")]
fn compress_s390x_vector_wrapper(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  // SAFETY: dispatch requires `s390x::VECTOR` before selecting this kernel.
  unsafe { compress_s390x_vector(chaining_value, block_words, counter, block_len, flags) }
}

#[cfg(target_arch = "s390x")]
#[target_feature(enable = "vector")]
unsafe fn chunk_compress_blocks_s390x_vector(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);

  // On IBM Z, the vector per-block wrapper has measurable fixed overhead on
  // short one-chunk bodies (e.g. 256B/1024B kernel-ab cases). Route those
  // small batches through the portable scalar loop and keep the vector path
  // for larger batches where it clearly wins.
  const SHORT_BATCH_PORTABLE_MAX_BLOCKS: usize = (CHUNK_LEN / BLOCK_LEN) - 1; // 15
  let num_blocks = blocks.len() / BLOCK_LEN;
  if num_blocks <= SHORT_BATCH_PORTABLE_MAX_BLOCKS {
    chunk_compress_blocks_portable(chaining_value, chunk_counter, flags, blocks_compressed, blocks);
    return;
  }

  if blocks.len() == BLOCK_LEN {
    // SAFETY: `blocks` is exactly one block, and `[u8; BLOCK_LEN]` has 1-byte alignment.
    let block_bytes: &[u8; BLOCK_LEN] = unsafe { &*(blocks.as_ptr().cast()) };
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    let block_words = words16_from_le_bytes_64(block_bytes);
    // SAFETY: `compress_s390x_vector` is gated by the current function's vector target feature.
    *chaining_value = first_8_words(unsafe {
      compress_s390x_vector(
        chaining_value,
        &block_words,
        chunk_counter,
        BLOCK_LEN as u32,
        flags | start,
      )
    });
    *blocks_compressed = blocks_compressed.wrapping_add(1);
    return;
  }

  let (block_slices, remainder) = blocks.as_chunks::<BLOCK_LEN>();
  debug_assert!(remainder.is_empty());
  for block_bytes in block_slices {
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    let block_words = words16_from_le_bytes_64(block_bytes);
    // SAFETY: `compress_s390x_vector` is gated by the current function's vector target feature.
    *chaining_value = first_8_words(unsafe {
      compress_s390x_vector(
        chaining_value,
        &block_words,
        chunk_counter,
        BLOCK_LEN as u32,
        flags | start,
      )
    });
    *blocks_compressed = blocks_compressed.wrapping_add(1);
  }
}

#[cfg(target_arch = "s390x")]
fn chunk_compress_blocks_s390x_vector_wrapper(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  // SAFETY: dispatch requires `s390x::VECTOR` before selecting this kernel.
  unsafe { chunk_compress_blocks_s390x_vector(chaining_value, chunk_counter, flags, blocks_compressed, blocks) }
}

#[cfg(target_arch = "s390x")]
#[target_feature(enable = "vector")]
unsafe fn parent_cv_s390x_vector(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  let mut block_words = [0u32; 16];
  block_words[..8].copy_from_slice(&left_child_cv);
  block_words[8..].copy_from_slice(&right_child_cv);
  first_8_words(compress_simd_leaf(
    &key_words,
    &block_words,
    0,
    BLOCK_LEN as u32,
    PARENT | flags,
  ))
}

#[cfg(target_arch = "s390x")]
fn parent_cv_s390x_vector_wrapper(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  // SAFETY: dispatch requires `s390x::VECTOR` before selecting this kernel.
  unsafe { parent_cv_s390x_vector(left_child_cv, right_child_cv, key_words, flags) }
}

#[cfg(target_arch = "s390x")]
#[target_feature(enable = "vector")]
unsafe fn hash_one_chunk_s390x_vector(input: *const u8, key: &[u32; 8], counter: u64, flags: u32, out: *mut u8) {
  let mut cv = *key;
  let mut blocks_compressed = 0u8;
  let body_len = CHUNK_LEN - BLOCK_LEN;
  // SAFETY: caller guarantees one full chunk is readable from `input`.
  let body = unsafe { core::slice::from_raw_parts(input, body_len) };
  // SAFETY: caller upholds the chunk pointer/length precondition.
  unsafe { chunk_compress_blocks_s390x_vector(&mut cv, counter, flags, &mut blocks_compressed, body) };

  // SAFETY: caller guarantees one full chunk is readable from `input`.
  let tail_words = unsafe {
    let tail_ptr = input.add(body_len);
    words16_from_le_bytes_64(&*tail_ptr.cast::<[u8; BLOCK_LEN]>())
  };
  let start = if blocks_compressed == 0 { CHUNK_START } else { 0 };
  let tail_flags = flags | start | super::CHUNK_END;
  // SAFETY: `compress_s390x_vector` is gated by the current function's vector target feature.
  cv = first_8_words(unsafe { compress_s390x_vector(&cv, &tail_words, counter, BLOCK_LEN as u32, tail_flags) });
  // SAFETY: caller guarantees one full CV output is writable at `out`.
  unsafe { write_cv_words(out, &cv) };
}

#[cfg(target_arch = "s390x")]
#[target_feature(enable = "vector")]
unsafe fn hash_many_contiguous_s390x_vector(
  input: *const u8,
  num_chunks: usize,
  key: &[u32; 8],
  counter: u64,
  flags: u32,
  out: *mut u8,
) {
  debug_assert!(num_chunks != 0);

  let mut idx = 0usize;
  while idx + 4 <= num_chunks {
    // SAFETY: `idx + 4 <= num_chunks` guarantees four full contiguous chunks and outputs remain.
    unsafe {
      hash4_contiguous_full_chunks_simd(
        input.add(idx * CHUNK_LEN),
        key,
        counter.wrapping_add(idx as u64),
        flags,
        out.add(idx * OUT_LEN),
      );
    }
    idx += 4;
  }

  while idx < num_chunks {
    // SAFETY: `idx < num_chunks` guarantees one full chunk/output remains.
    unsafe {
      hash_one_chunk_s390x_vector(
        input.add(idx * CHUNK_LEN),
        key,
        counter.wrapping_add(idx as u64),
        flags,
        out.add(idx * OUT_LEN),
      );
    }
    idx += 1;
  }
}

#[cfg(target_arch = "s390x")]
unsafe fn hash_many_contiguous_s390x_vector_wrapper(
  input: *const u8,
  num_chunks: usize,
  key: &[u32; 8],
  counter: u64,
  flags: u32,
  out: *mut u8,
) {
  // SAFETY: dispatch requires `s390x::VECTOR` before selecting this kernel.
  unsafe { hash_many_contiguous_s390x_vector(input, num_chunks, key, counter, flags, out) }
}

// ─────────────────────────────────────────────────────────────────────────────
// powerpc64 VSX wrappers
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "powerpc64")]
#[target_feature(enable = "vsx")]
unsafe fn compress_power_vsx(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  compress_simd_leaf(chaining_value, block_words, counter, block_len, flags)
}

#[cfg(target_arch = "powerpc64")]
fn compress_power_vsx_wrapper(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  // SAFETY: dispatch requires `power::VSX` before selecting this kernel.
  unsafe { compress_power_vsx(chaining_value, block_words, counter, block_len, flags) }
}

#[cfg(target_arch = "powerpc64")]
#[target_feature(enable = "vsx")]
unsafe fn chunk_compress_blocks_power_vsx(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);

  if blocks.len() == BLOCK_LEN {
    // SAFETY: `blocks` is exactly one block, and `[u8; BLOCK_LEN]` has 1-byte alignment.
    let block_bytes: &[u8; BLOCK_LEN] = unsafe { &*(blocks.as_ptr().cast()) };
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    let block_words = words16_from_le_bytes_64(block_bytes);
    // SAFETY: `compress_power_vsx` is gated by the current function's VSX target feature.
    *chaining_value = first_8_words(unsafe {
      compress_power_vsx(
        chaining_value,
        &block_words,
        chunk_counter,
        BLOCK_LEN as u32,
        flags | start,
      )
    });
    *blocks_compressed = blocks_compressed.wrapping_add(1);
    return;
  }

  let (block_slices, remainder) = blocks.as_chunks::<BLOCK_LEN>();
  debug_assert!(remainder.is_empty());
  for block_bytes in block_slices {
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    let block_words = words16_from_le_bytes_64(block_bytes);
    // SAFETY: `compress_power_vsx` is gated by the current function's VSX target feature.
    *chaining_value = first_8_words(unsafe {
      compress_power_vsx(
        chaining_value,
        &block_words,
        chunk_counter,
        BLOCK_LEN as u32,
        flags | start,
      )
    });
    *blocks_compressed = blocks_compressed.wrapping_add(1);
  }
}

#[cfg(target_arch = "powerpc64")]
fn chunk_compress_blocks_power_vsx_wrapper(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  // SAFETY: dispatch requires `power::VSX` before selecting this kernel.
  unsafe { chunk_compress_blocks_power_vsx(chaining_value, chunk_counter, flags, blocks_compressed, blocks) }
}

#[cfg(target_arch = "powerpc64")]
#[target_feature(enable = "vsx")]
unsafe fn parent_cv_power_vsx(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  let mut block_words = [0u32; 16];
  block_words[..8].copy_from_slice(&left_child_cv);
  block_words[8..].copy_from_slice(&right_child_cv);
  first_8_words(compress_simd_leaf(
    &key_words,
    &block_words,
    0,
    BLOCK_LEN as u32,
    PARENT | flags,
  ))
}

#[cfg(target_arch = "powerpc64")]
fn parent_cv_power_vsx_wrapper(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  // SAFETY: dispatch requires `power::VSX` before selecting this kernel.
  unsafe { parent_cv_power_vsx(left_child_cv, right_child_cv, key_words, flags) }
}

#[cfg(target_arch = "powerpc64")]
#[target_feature(enable = "vsx")]
unsafe fn hash_one_chunk_power_vsx(input: *const u8, key: &[u32; 8], counter: u64, flags: u32, out: *mut u8) {
  let mut cv = *key;
  let mut blocks_compressed = 0u8;
  let body_len = CHUNK_LEN - BLOCK_LEN;
  // SAFETY: caller guarantees one full chunk is readable from `input`.
  let body = unsafe { core::slice::from_raw_parts(input, body_len) };
  // SAFETY: caller upholds the chunk pointer/length precondition.
  unsafe { chunk_compress_blocks_power_vsx(&mut cv, counter, flags, &mut blocks_compressed, body) };

  // SAFETY: caller guarantees one full chunk is readable from `input`.
  let tail_words = unsafe {
    let tail_ptr = input.add(body_len);
    words16_from_le_bytes_64(&*tail_ptr.cast::<[u8; BLOCK_LEN]>())
  };
  let start = if blocks_compressed == 0 { CHUNK_START } else { 0 };
  let tail_flags = flags | start | super::CHUNK_END;
  // SAFETY: `compress_power_vsx` is gated by the current function's VSX target feature.
  cv = first_8_words(unsafe { compress_power_vsx(&cv, &tail_words, counter, BLOCK_LEN as u32, tail_flags) });
  // SAFETY: caller guarantees one full CV output is writable at `out`.
  unsafe { write_cv_words(out, &cv) };
}

#[cfg(target_arch = "powerpc64")]
#[target_feature(enable = "vsx")]
unsafe fn hash_many_contiguous_power_vsx(
  input: *const u8,
  num_chunks: usize,
  key: &[u32; 8],
  counter: u64,
  flags: u32,
  out: *mut u8,
) {
  debug_assert!(num_chunks != 0);

  let mut idx = 0usize;
  while idx + 4 <= num_chunks {
    // SAFETY: `idx + 4 <= num_chunks` guarantees four full contiguous chunks and outputs remain.
    unsafe {
      hash4_contiguous_full_chunks_simd(
        input.add(idx * CHUNK_LEN),
        key,
        counter.wrapping_add(idx as u64),
        flags,
        out.add(idx * OUT_LEN),
      );
    }
    idx += 4;
  }

  while idx < num_chunks {
    // SAFETY: `idx < num_chunks` guarantees one full chunk/output remains.
    unsafe {
      hash_one_chunk_power_vsx(
        input.add(idx * CHUNK_LEN),
        key,
        counter.wrapping_add(idx as u64),
        flags,
        out.add(idx * OUT_LEN),
      );
    }
    idx += 1;
  }
}

#[cfg(target_arch = "powerpc64")]
unsafe fn hash_many_contiguous_power_vsx_wrapper(
  input: *const u8,
  num_chunks: usize,
  key: &[u32; 8],
  counter: u64,
  flags: u32,
  out: *mut u8,
) {
  // SAFETY: dispatch requires `power::VSX` before selecting this kernel.
  unsafe { hash_many_contiguous_power_vsx(input, num_chunks, key, counter, flags, out) }
}

// ─────────────────────────────────────────────────────────────────────────────
// riscv64 RVV wrappers
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "riscv64")]
#[inline(always)]
fn compress_simd_leaf_riscv(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  type Vec4 = core::simd::u32x4;

  let mut msg = [Vec4::splat(0); 16];
  let mut w = 0usize;
  while w < 16 {
    msg[w] = Vec4::splat(block_words[w]);
    w += 1;
  }

  let mut v = [
    Vec4::splat(chaining_value[0]),
    Vec4::splat(chaining_value[1]),
    Vec4::splat(chaining_value[2]),
    Vec4::splat(chaining_value[3]),
    Vec4::splat(chaining_value[4]),
    Vec4::splat(chaining_value[5]),
    Vec4::splat(chaining_value[6]),
    Vec4::splat(chaining_value[7]),
    Vec4::splat(super::IV[0]),
    Vec4::splat(super::IV[1]),
    Vec4::splat(super::IV[2]),
    Vec4::splat(super::IV[3]),
    Vec4::splat(counter as u32),
    Vec4::splat((counter >> 32) as u32),
    Vec4::splat(block_len),
    Vec4::splat(flags),
  ];

  round4_simd(&mut v, &msg, 0);
  round4_simd(&mut v, &msg, 1);
  round4_simd(&mut v, &msg, 2);
  round4_simd(&mut v, &msg, 3);
  round4_simd(&mut v, &msg, 4);
  round4_simd(&mut v, &msg, 5);
  round4_simd(&mut v, &msg, 6);

  let x0 = (v[0] ^ v[8]).to_array();
  let x1 = (v[1] ^ v[9]).to_array();
  let x2 = (v[2] ^ v[10]).to_array();
  let x3 = (v[3] ^ v[11]).to_array();
  let x4 = (v[4] ^ v[12]).to_array();
  let x5 = (v[5] ^ v[13]).to_array();
  let x6 = (v[6] ^ v[14]).to_array();
  let x7 = (v[7] ^ v[15]).to_array();
  let y0 = (v[8] ^ Vec4::splat(chaining_value[0])).to_array();
  let y1 = (v[9] ^ Vec4::splat(chaining_value[1])).to_array();
  let y2 = (v[10] ^ Vec4::splat(chaining_value[2])).to_array();
  let y3 = (v[11] ^ Vec4::splat(chaining_value[3])).to_array();
  let y4 = (v[12] ^ Vec4::splat(chaining_value[4])).to_array();
  let y5 = (v[13] ^ Vec4::splat(chaining_value[5])).to_array();
  let y6 = (v[14] ^ Vec4::splat(chaining_value[6])).to_array();
  let y7 = (v[15] ^ Vec4::splat(chaining_value[7])).to_array();

  [
    x0[0], x1[0], x2[0], x3[0], x4[0], x5[0], x6[0], x7[0], y0[0], y1[0], y2[0], y3[0], y4[0], y5[0], y6[0], y7[0],
  ]
}

#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "v")]
unsafe fn compress_riscv_v(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  compress_simd_leaf_riscv(chaining_value, block_words, counter, block_len, flags)
}

#[cfg(target_arch = "riscv64")]
fn compress_riscv_v_wrapper(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  // SAFETY: dispatch requires `riscv::V` before selecting this kernel.
  unsafe { compress_riscv_v(chaining_value, block_words, counter, block_len, flags) }
}

#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "v")]
unsafe fn chunk_compress_blocks_riscv_v(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);

  if blocks.len() == BLOCK_LEN {
    // SAFETY: `blocks` is exactly one block, and `[u8; BLOCK_LEN]` has 1-byte alignment.
    let block_bytes: &[u8; BLOCK_LEN] = unsafe { &*(blocks.as_ptr().cast()) };
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    let block_words = words16_from_le_bytes_64(block_bytes);
    *chaining_value = first_8_words(unsafe {
      compress_riscv_v(
        chaining_value,
        &block_words,
        chunk_counter,
        BLOCK_LEN as u32,
        flags | start,
      )
    });
    *blocks_compressed = blocks_compressed.wrapping_add(1);
    return;
  }

  let (block_slices, remainder) = blocks.as_chunks::<BLOCK_LEN>();
  debug_assert!(remainder.is_empty());
  for block_bytes in block_slices {
    let start = if *blocks_compressed == 0 { CHUNK_START } else { 0 };
    let block_words = words16_from_le_bytes_64(block_bytes);
    *chaining_value = first_8_words(unsafe {
      compress_riscv_v(
        chaining_value,
        &block_words,
        chunk_counter,
        BLOCK_LEN as u32,
        flags | start,
      )
    });
    *blocks_compressed = blocks_compressed.wrapping_add(1);
  }
}

#[cfg(target_arch = "riscv64")]
fn chunk_compress_blocks_riscv_v_wrapper(
  chaining_value: &mut [u32; 8],
  chunk_counter: u64,
  flags: u32,
  blocks_compressed: &mut u8,
  blocks: &[u8],
) {
  // SAFETY: dispatch requires `riscv::V` before selecting this kernel.
  unsafe { chunk_compress_blocks_riscv_v(chaining_value, chunk_counter, flags, blocks_compressed, blocks) }
}

#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "v")]
unsafe fn parent_cv_riscv_v(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  let mut block_words = [0u32; 16];
  block_words[..8].copy_from_slice(&left_child_cv);
  block_words[8..].copy_from_slice(&right_child_cv);
  first_8_words(compress_simd_leaf(
    &key_words,
    &block_words,
    0,
    BLOCK_LEN as u32,
    PARENT | flags,
  ))
}

#[cfg(target_arch = "riscv64")]
fn parent_cv_riscv_v_wrapper(
  left_child_cv: [u32; 8],
  right_child_cv: [u32; 8],
  key_words: [u32; 8],
  flags: u32,
) -> [u32; 8] {
  // SAFETY: dispatch requires `riscv::V` before selecting this kernel.
  unsafe { parent_cv_riscv_v(left_child_cv, right_child_cv, key_words, flags) }
}

#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "v")]
unsafe fn hash_one_chunk_riscv_v(input: *const u8, key: &[u32; 8], counter: u64, flags: u32, out: *mut u8) {
  let mut cv = *key;
  let mut blocks_compressed = 0u8;
  let body_len = CHUNK_LEN - BLOCK_LEN;
  // SAFETY: caller guarantees one full chunk is readable from `input`.
  let body = unsafe { core::slice::from_raw_parts(input, body_len) };
  // SAFETY: caller upholds the chunk pointer/length precondition.
  unsafe { chunk_compress_blocks_riscv_v(&mut cv, counter, flags, &mut blocks_compressed, body) };

  // SAFETY: caller guarantees one full chunk is readable from `input`.
  let tail_words = unsafe {
    let tail_ptr = input.add(body_len);
    words16_from_le_bytes_64(&*tail_ptr.cast::<[u8; BLOCK_LEN]>())
  };
  let start = if blocks_compressed == 0 { CHUNK_START } else { 0 };
  let tail_flags = flags | start | super::CHUNK_END;
  cv = first_8_words(unsafe { compress_riscv_v(&cv, &tail_words, counter, BLOCK_LEN as u32, tail_flags) });
  // SAFETY: caller guarantees one full CV output is writable at `out`.
  unsafe { write_cv_words(out, &cv) };
}

#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "v")]
unsafe fn hash_many_contiguous_riscv_v(
  input: *const u8,
  num_chunks: usize,
  key: &[u32; 8],
  counter: u64,
  flags: u32,
  out: *mut u8,
) {
  debug_assert!(num_chunks != 0);

  let mut idx = 0usize;
  while idx + 4 <= num_chunks {
    // SAFETY: `idx + 4 <= num_chunks` guarantees four full contiguous chunks and outputs remain.
    unsafe {
      hash4_contiguous_full_chunks_simd(
        input.add(idx * CHUNK_LEN),
        key,
        counter.wrapping_add(idx as u64),
        flags,
        out.add(idx * OUT_LEN),
      );
    }
    idx += 4;
  }

  while idx < num_chunks {
    // SAFETY: `idx < num_chunks` guarantees one full chunk/output remains.
    unsafe {
      hash_one_chunk_riscv_v(
        input.add(idx * CHUNK_LEN),
        key,
        counter.wrapping_add(idx as u64),
        flags,
        out.add(idx * OUT_LEN),
      );
    }
    idx += 1;
  }
}

#[cfg(target_arch = "riscv64")]
unsafe fn hash_many_contiguous_riscv_v_wrapper(
  input: *const u8,
  num_chunks: usize,
  key: &[u32; 8],
  counter: u64,
  flags: u32,
  out: *mut u8,
) {
  // SAFETY: dispatch requires `riscv::V` before selecting this kernel.
  unsafe { hash_many_contiguous_riscv_v(input, num_chunks, key, counter, flags, out) }
}

// ─────────────────────────────────────────────────────────────────────────────
// x86_64 SSSE3 wrappers
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn x86_compress_cv_portable_wrapper(
  cv: &[u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  // SAFETY: caller guarantees `block` points to a readable 64-byte block.
  let block_words = unsafe { words16_from_le_bytes_64(&*block.cast::<[u8; BLOCK_LEN]>()) };
  first_8_words(super::compress(cv, &block_words, counter, block_len, flags))
}

#[cfg(target_arch = "x86_64")]
unsafe fn x86_compress_cv_sse41_wrapper(
  cv: &[u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  // SAFETY: dispatch guarantees SSE4.1 availability; caller guarantees `block` is readable for 64
  // bytes.
  unsafe { super::x86_64::compress_cv_sse41_bytes(cv, block, counter, block_len, flags) }
}

#[cfg(target_arch = "x86_64")]
unsafe fn x86_compress_cv_avx2_wrapper(
  cv: &[u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  // SAFETY: dispatch guarantees AVX2 availability; caller guarantees `block` is readable for 64
  // bytes.
  unsafe { super::x86_64::compress_cv_avx2_bytes(cv, block, counter, block_len, flags) }
}

#[cfg(target_arch = "x86_64")]
unsafe fn x86_compress_cv_avx512_wrapper(
  cv: &[u32; 8],
  block: *const u8,
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 8] {
  #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
  {
    // SAFETY: dispatch guarantees AVX-512 availability; caller guarantees `block` is readable for 64
    // bytes.
    unsafe { super::x86_64::asm::compress_in_place_avx512(cv, block, counter, block_len, flags) }
  }
  #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
  {
    // SAFETY: dispatch guarantees AVX-512 availability; caller guarantees `block` is readable for 64
    // bytes.
    unsafe { super::x86_64::compress_cv_avx512_bytes(cv, block, counter, block_len, flags) }
  }
}

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
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    // SAFETY: this wrapper is only selected when AVX2 is available (checked by
    // dispatch), `ptrs` point to `DEGREE` in-bounds chunk inputs, and `out` is
    // valid for `DEGREE * OUT_LEN` bytes.
    unsafe {
      debug_assert!(flags <= u8::MAX as u32);
      super::x86_64::asm::hash_many_avx2(
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
        super::x86_64::asm::hash_many_avx2(
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
  while num_chunks >= super::x86_64::avx512::DEGREE {
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
      super::x86_64::asm::hash_many_avx512(
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
        super::x86_64::asm::hash_many_avx512(
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
      debug_assert!(num_chunks < super::x86_64::avx512::DEGREE);
      // Keep the AVX-512 hierarchy strict on non-asm targets too: materialize
      // a full 16-lane contiguous batch and ignore excess outputs.
      let mut tmp_input = [0u8; super::x86_64::avx512::DEGREE * CHUNK_LEN];
      let mut tmp_out = [0u8; super::x86_64::avx512::DEGREE * OUT_LEN];
      for i in 0..num_chunks {
        // SAFETY: `i < num_chunks`, and the caller guarantees `input` is valid
        // for `num_chunks * CHUNK_LEN` bytes. `tmp_input` is sized for
        // `DEGREE * CHUNK_LEN`, so each destination lane is in-bounds.
        unsafe {
          core::ptr::copy_nonoverlapping(
            input.add(i * CHUNK_LEN),
            tmp_input.as_mut_ptr().add(i * CHUNK_LEN),
            CHUNK_LEN,
          );
        }
      }

      let last_src_offset = (num_chunks - 1) * CHUNK_LEN;
      for i in num_chunks..super::x86_64::avx512::DEGREE {
        // SAFETY: `last_src_offset` points to a previously materialized lane in
        // `tmp_input`, and destination lane `i` is within the fixed-size buffer.
        unsafe {
          core::ptr::copy_nonoverlapping(
            tmp_input.as_ptr().add(last_src_offset),
            tmp_input.as_mut_ptr().add(i * CHUNK_LEN),
            CHUNK_LEN,
          );
        }
      }

      // SAFETY: AVX-512 is available per dispatch, and temporary buffers are
      // valid for the full 16-lane contiguous contract.
      unsafe {
        super::x86_64::avx512::hash16_contiguous(tmp_input.as_ptr(), key, counter, flags, tmp_out.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp_out.as_ptr(), out, num_chunks * OUT_LEN);
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
