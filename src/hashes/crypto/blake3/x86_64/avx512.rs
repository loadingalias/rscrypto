//! BLAKE3 x86_64 AVX-512 throughput kernel (16-way).

use core::arch::x86_64::*;

use super::{
  super::{BLOCK_LEN, BLOCK_LEN_U32, CHUNK_LEN, IV, MSG_SCHEDULE, OUT_LEN},
  HashManyRequest,
  avx2::transpose8x8,
  counter_high, counter_low,
};

pub(crate) const DEGREE: usize = 16;

/// # Safety
///
/// AVX-512F must be available.
#[inline(always)]
unsafe fn add(a: __m512i, b: __m512i) -> __m512i {
  // SAFETY: Caller guarantees the required AVX-512 feature set for this backend.
  unsafe { _mm512_add_epi32(a, b) }
}

/// # Safety
///
/// AVX-512F must be available.
#[inline(always)]
unsafe fn xor(a: __m512i, b: __m512i) -> __m512i {
  // SAFETY: Caller guarantees the required AVX-512 feature set for this backend.
  unsafe { _mm512_xor_si512(a, b) }
}

/// # Safety
///
/// AVX-512F must be available.
#[inline(always)]
unsafe fn set1(x: u32) -> __m512i {
  // SAFETY: Caller guarantees the required AVX-512 feature set for this backend.
  unsafe { _mm512_set1_epi32(x.cast_signed()) }
}

/// # Safety
///
/// AVX2 must be available and `src` must be readable for 32 bytes.
#[inline(always)]
unsafe fn loadu256(src: *const u8) -> __m256i {
  // SAFETY: Caller guarantees `src` is valid to read 32 bytes and has enabled this AVX-512 backend.
  unsafe { _mm256_loadu_si256(src.cast()) }
}

/// # Safety
///
/// AVX2 must be available and `dest` must be writable for 32 bytes.
#[inline(always)]
unsafe fn storeu256(src: __m256i, dest: *mut u8) {
  // SAFETY: Caller guarantees `dest` is valid to write 32 bytes and has enabled this AVX-512 backend.
  unsafe { _mm256_storeu_si256(dest.cast(), src) }
}

/// # Safety
///
/// AVX-512F must be available.
#[inline(always)]
unsafe fn rot16(x: __m512i) -> __m512i {
  // 32-bit rotate by 16. Prefer `vprold` over shift/or.
  // SAFETY: Caller guarantees the required AVX-512 feature set for this backend.
  unsafe { _mm512_rol_epi32(x, 16) }
}

/// # Safety
///
/// AVX-512F must be available.
#[inline(always)]
unsafe fn rot12(x: __m512i) -> __m512i {
  // Rotate right by 12 == rotate left by 20.
  // SAFETY: Caller guarantees the required AVX-512 feature set for this backend.
  unsafe { _mm512_rol_epi32(x, 20) }
}

/// # Safety
///
/// AVX-512F must be available.
#[inline(always)]
unsafe fn rot8(x: __m512i) -> __m512i {
  // Rotate right by 8 == rotate left by 24.
  // SAFETY: Caller guarantees the required AVX-512 feature set for this backend.
  unsafe { _mm512_rol_epi32(x, 24) }
}

/// # Safety
///
/// AVX-512F must be available.
#[inline(always)]
unsafe fn rot7(x: __m512i) -> __m512i {
  // Rotate right by 7 == rotate left by 25.
  // SAFETY: Caller guarantees the required AVX-512 feature set for this backend.
  unsafe { _mm512_rol_epi32(x, 25) }
}

/// # Safety
///
/// AVX-512F must be available and `r` must be in `0..7`.
#[inline(always)]
unsafe fn round(v: &mut [__m512i; 16], m: &[__m512i; 16], r: usize) {
  // SAFETY: Caller guarantees this AVX-512 backend is active; all vector lanes are local registers.
  unsafe {
    v[0] = add(v[0], m[MSG_SCHEDULE[r][0]]);
    v[1] = add(v[1], m[MSG_SCHEDULE[r][2]]);
    v[2] = add(v[2], m[MSG_SCHEDULE[r][4]]);
    v[3] = add(v[3], m[MSG_SCHEDULE[r][6]]);
    v[0] = add(v[0], v[4]);
    v[1] = add(v[1], v[5]);
    v[2] = add(v[2], v[6]);
    v[3] = add(v[3], v[7]);
    v[12] = xor(v[12], v[0]);
    v[13] = xor(v[13], v[1]);
    v[14] = xor(v[14], v[2]);
    v[15] = xor(v[15], v[3]);
    v[12] = rot16(v[12]);
    v[13] = rot16(v[13]);
    v[14] = rot16(v[14]);
    v[15] = rot16(v[15]);
    v[8] = add(v[8], v[12]);
    v[9] = add(v[9], v[13]);
    v[10] = add(v[10], v[14]);
    v[11] = add(v[11], v[15]);
    v[4] = xor(v[4], v[8]);
    v[5] = xor(v[5], v[9]);
    v[6] = xor(v[6], v[10]);
    v[7] = xor(v[7], v[11]);
    v[4] = rot12(v[4]);
    v[5] = rot12(v[5]);
    v[6] = rot12(v[6]);
    v[7] = rot12(v[7]);
    v[0] = add(v[0], m[MSG_SCHEDULE[r][1]]);
    v[1] = add(v[1], m[MSG_SCHEDULE[r][3]]);
    v[2] = add(v[2], m[MSG_SCHEDULE[r][5]]);
    v[3] = add(v[3], m[MSG_SCHEDULE[r][7]]);
    v[0] = add(v[0], v[4]);
    v[1] = add(v[1], v[5]);
    v[2] = add(v[2], v[6]);
    v[3] = add(v[3], v[7]);
    v[12] = xor(v[12], v[0]);
    v[13] = xor(v[13], v[1]);
    v[14] = xor(v[14], v[2]);
    v[15] = xor(v[15], v[3]);
    v[12] = rot8(v[12]);
    v[13] = rot8(v[13]);
    v[14] = rot8(v[14]);
    v[15] = rot8(v[15]);
    v[8] = add(v[8], v[12]);
    v[9] = add(v[9], v[13]);
    v[10] = add(v[10], v[14]);
    v[11] = add(v[11], v[15]);
    v[4] = xor(v[4], v[8]);
    v[5] = xor(v[5], v[9]);
    v[6] = xor(v[6], v[10]);
    v[7] = xor(v[7], v[11]);
    v[4] = rot7(v[4]);
    v[5] = rot7(v[5]);
    v[6] = rot7(v[6]);
    v[7] = rot7(v[7]);

    v[0] = add(v[0], m[MSG_SCHEDULE[r][8]]);
    v[1] = add(v[1], m[MSG_SCHEDULE[r][10]]);
    v[2] = add(v[2], m[MSG_SCHEDULE[r][12]]);
    v[3] = add(v[3], m[MSG_SCHEDULE[r][14]]);
    v[0] = add(v[0], v[5]);
    v[1] = add(v[1], v[6]);
    v[2] = add(v[2], v[7]);
    v[3] = add(v[3], v[4]);
    v[15] = xor(v[15], v[0]);
    v[12] = xor(v[12], v[1]);
    v[13] = xor(v[13], v[2]);
    v[14] = xor(v[14], v[3]);
    v[15] = rot16(v[15]);
    v[12] = rot16(v[12]);
    v[13] = rot16(v[13]);
    v[14] = rot16(v[14]);
    v[10] = add(v[10], v[15]);
    v[11] = add(v[11], v[12]);
    v[8] = add(v[8], v[13]);
    v[9] = add(v[9], v[14]);
    v[5] = xor(v[5], v[10]);
    v[6] = xor(v[6], v[11]);
    v[7] = xor(v[7], v[8]);
    v[4] = xor(v[4], v[9]);
    v[5] = rot12(v[5]);
    v[6] = rot12(v[6]);
    v[7] = rot12(v[7]);
    v[4] = rot12(v[4]);
    v[0] = add(v[0], m[MSG_SCHEDULE[r][9]]);
    v[1] = add(v[1], m[MSG_SCHEDULE[r][11]]);
    v[2] = add(v[2], m[MSG_SCHEDULE[r][13]]);
    v[3] = add(v[3], m[MSG_SCHEDULE[r][15]]);
    v[0] = add(v[0], v[5]);
    v[1] = add(v[1], v[6]);
    v[2] = add(v[2], v[7]);
    v[3] = add(v[3], v[4]);
    v[15] = xor(v[15], v[0]);
    v[12] = xor(v[12], v[1]);
    v[13] = xor(v[13], v[2]);
    v[14] = xor(v[14], v[3]);
    v[15] = rot8(v[15]);
    v[12] = rot8(v[12]);
    v[13] = rot8(v[13]);
    v[14] = rot8(v[14]);
    v[10] = add(v[10], v[15]);
    v[11] = add(v[11], v[12]);
    v[8] = add(v[8], v[13]);
    v[9] = add(v[9], v[14]);
    v[5] = xor(v[5], v[10]);
    v[6] = xor(v[6], v[11]);
    v[7] = xor(v[7], v[8]);
    v[4] = xor(v[4], v[9]);
    v[5] = rot7(v[5]);
    v[6] = rot7(v[6]);
    v[7] = rot7(v[7]);
    v[4] = rot7(v[4]);
  }
}

/// # Safety
///
/// AVX-512F must be available.
#[inline(always)]
unsafe fn counter_vec(counter: u64, increment_counter: bool) -> (__m512i, __m512i) {
  let mask = if increment_counter { !0u64 } else { 0u64 };
  // SAFETY: Counter vectors are assembled with this backend's AVX-512 intrinsics only.
  unsafe {
    let lo = _mm512_setr_epi32(
      counter_low(counter).cast_signed(),
      counter_low(counter.wrapping_add(mask & 1)).cast_signed(),
      counter_low(counter.wrapping_add(mask & 2)).cast_signed(),
      counter_low(counter.wrapping_add(mask & 3)).cast_signed(),
      counter_low(counter.wrapping_add(mask & 4)).cast_signed(),
      counter_low(counter.wrapping_add(mask & 5)).cast_signed(),
      counter_low(counter.wrapping_add(mask & 6)).cast_signed(),
      counter_low(counter.wrapping_add(mask & 7)).cast_signed(),
      counter_low(counter.wrapping_add(mask & 8)).cast_signed(),
      counter_low(counter.wrapping_add(mask & 9)).cast_signed(),
      counter_low(counter.wrapping_add(mask & 10)).cast_signed(),
      counter_low(counter.wrapping_add(mask & 11)).cast_signed(),
      counter_low(counter.wrapping_add(mask & 12)).cast_signed(),
      counter_low(counter.wrapping_add(mask & 13)).cast_signed(),
      counter_low(counter.wrapping_add(mask & 14)).cast_signed(),
      counter_low(counter.wrapping_add(mask & 15)).cast_signed(),
    );
    let hi = _mm512_setr_epi32(
      counter_high(counter).cast_signed(),
      counter_high(counter.wrapping_add(mask & 1)).cast_signed(),
      counter_high(counter.wrapping_add(mask & 2)).cast_signed(),
      counter_high(counter.wrapping_add(mask & 3)).cast_signed(),
      counter_high(counter.wrapping_add(mask & 4)).cast_signed(),
      counter_high(counter.wrapping_add(mask & 5)).cast_signed(),
      counter_high(counter.wrapping_add(mask & 6)).cast_signed(),
      counter_high(counter.wrapping_add(mask & 7)).cast_signed(),
      counter_high(counter.wrapping_add(mask & 8)).cast_signed(),
      counter_high(counter.wrapping_add(mask & 9)).cast_signed(),
      counter_high(counter.wrapping_add(mask & 10)).cast_signed(),
      counter_high(counter.wrapping_add(mask & 11)).cast_signed(),
      counter_high(counter.wrapping_add(mask & 12)).cast_signed(),
      counter_high(counter.wrapping_add(mask & 13)).cast_signed(),
      counter_high(counter.wrapping_add(mask & 14)).cast_signed(),
      counter_high(counter.wrapping_add(mask & 15)).cast_signed(),
    );
    (lo, hi)
  }
}

/// # Safety
///
/// AVX2 must be available and every input must be readable for a complete
/// block starting at `block_offset`.
#[inline(always)]
unsafe fn transpose_msg_vecs8(inputs: &[*const u8; 8], block_offset: usize) -> [__m256i; 16] {
  // SAFETY: Caller guarantees each input points to at least one full block at `block_offset`.
  unsafe {
    let stride = 4usize.strict_mul(8);
    let mut half0 = [
      loadu256(inputs[0].add(block_offset)),
      loadu256(inputs[1].add(block_offset)),
      loadu256(inputs[2].add(block_offset)),
      loadu256(inputs[3].add(block_offset)),
      loadu256(inputs[4].add(block_offset)),
      loadu256(inputs[5].add(block_offset)),
      loadu256(inputs[6].add(block_offset)),
      loadu256(inputs[7].add(block_offset)),
    ];
    let mut half1 = [
      loadu256(inputs[0].add(block_offset.strict_add(stride))),
      loadu256(inputs[1].add(block_offset.strict_add(stride))),
      loadu256(inputs[2].add(block_offset.strict_add(stride))),
      loadu256(inputs[3].add(block_offset.strict_add(stride))),
      loadu256(inputs[4].add(block_offset.strict_add(stride))),
      loadu256(inputs[5].add(block_offset.strict_add(stride))),
      loadu256(inputs[6].add(block_offset.strict_add(stride))),
      loadu256(inputs[7].add(block_offset.strict_add(stride))),
    ];

    for &input in inputs.iter() {
      _mm_prefetch(
        input.wrapping_add(block_offset.strict_add(256)).cast::<i8>(),
        _MM_HINT_T0,
      );
    }

    transpose8x8(&mut half0);
    transpose8x8(&mut half1);

    [
      half0[0], half0[1], half0[2], half0[3], half0[4], half0[5], half0[6], half0[7], half1[0], half1[1], half1[2],
      half1[3], half1[4], half1[5], half1[6], half1[7],
    ]
  }
}

/// # Safety
///
/// AVX-512F, AVX-512DQ, and AVX2 must be available. Every input must be
/// readable for a complete block starting at `block_offset`.
#[inline(always)]
unsafe fn transpose_msg_vecs16(inputs: &[*const u8; 16], block_offset: usize) -> [__m512i; 16] {
  // SAFETY: Caller guarantees each input points to at least one full block at `block_offset`.
  unsafe {
    let lo_ptrs = [
      inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7],
    ];
    let hi_ptrs = [
      inputs[8], inputs[9], inputs[10], inputs[11], inputs[12], inputs[13], inputs[14], inputs[15],
    ];

    let lo = transpose_msg_vecs8(&lo_ptrs, block_offset);
    let hi = transpose_msg_vecs8(&hi_ptrs, block_offset);

    let mut out = [set1(0); 16];
    for i in 0..16 {
      let mut v = _mm512_castsi256_si512(lo[i]);
      v = _mm512_inserti64x4(v, hi[i], 1);
      out[i] = v;
    }

    out
  }
}

/// Owned Rust-intrinsic implementation of the 16-way contiguous hash-many kernel.
///
/// This stays callable on platforms where production dispatch still prefers
/// assembly, so diagnostic benches can measure the owned candidate directly.
///
/// # Safety
///
/// Caller must ensure AVX-512F + AVX-512VL + AVX-512DQ + AVX2 are available,
/// every input pointer is valid for `blocks * BLOCK_LEN` readable bytes, and
/// `out` is valid for `DEGREE * OUT_LEN` writable bytes.
#[target_feature(enable = "avx512f,avx512vl,avx512dq,avx2")]
pub(crate) unsafe fn hash16_owned(request: HashManyRequest<'_, DEGREE>) {
  // SAFETY: this function's contract is `hash16_owned_with_last_block_len`'s contract with a full
  // final block, which `BLOCK_LEN_U32` states.
  unsafe { hash16_owned_with_last_block_len(request, BLOCK_LEN_U32) }
}

/// Like [`hash16_owned`], but the final block of every input holds `last_block_len` bytes.
///
/// # Safety
///
/// As for [`hash16_owned`]: every input pointer stays readable for `blocks * BLOCK_LEN`
/// bytes, including all of its final block. `last_block_len` must be in
/// `1..=BLOCK_LEN`; BLAKE3 also requires the final block's bytes past
/// `last_block_len` to be zero for the output to be correct.
#[target_feature(enable = "avx512f,avx512vl,avx512dq,avx2")]
#[inline]
pub(crate) unsafe fn hash16_owned_with_last_block_len(
  HashManyRequest {
    inputs,
    blocks,
    key,
    counter,
    increment_counter,
    flags,
    flags_start,
    flags_end,
    out,
  }: HashManyRequest<'_, DEGREE>,
  last_block_len: u32,
) {
  // SAFETY: 16-way AVX-512 BLAKE3 contiguous hash-many because:
  // 1. The caller guarantees AVX-512F/VL/DQ plus AVX2 availability.
  // 2. Each input pointer is readable for `blocks * BLOCK_LEN` bytes.
  // 3. `out` is writable for `DEGREE * OUT_LEN` bytes.
  // 4. All lane pointers and stores are bounded by fixed-size local arrays.
  unsafe {
    let full_block_len_vec = set1(BLOCK_LEN_U32);
    let last_block_len_vec = set1(last_block_len);
    let iv0 = set1(IV[0]);
    let iv1 = set1(IV[1]);
    let iv2 = set1(IV[2]);
    let iv3 = set1(IV[3]);

    let mut h_vecs = [
      set1(key[0]),
      set1(key[1]),
      set1(key[2]),
      set1(key[3]),
      set1(key[4]),
      set1(key[5]),
      set1(key[6]),
      set1(key[7]),
    ];

    let (counter_low_vec, counter_high_vec) = counter_vec(counter, increment_counter);

    for block in 0..blocks {
      let mut block_flags = flags;
      if block == 0 {
        block_flags |= flags_start;
      }
      if block.strict_add(1) == blocks {
        block_flags |= flags_end;
      }

      let block_flags_vec = set1(block_flags);
      let block_len_vec = if block.strict_add(1) == blocks {
        last_block_len_vec
      } else {
        full_block_len_vec
      };

      let m = transpose_msg_vecs16(inputs, block.strict_mul(BLOCK_LEN));

      let mut v = [
        h_vecs[0],
        h_vecs[1],
        h_vecs[2],
        h_vecs[3],
        h_vecs[4],
        h_vecs[5],
        h_vecs[6],
        h_vecs[7],
        iv0,
        iv1,
        iv2,
        iv3,
        counter_low_vec,
        counter_high_vec,
        block_len_vec,
        block_flags_vec,
      ];

      round(&mut v, &m, 0);
      round(&mut v, &m, 1);
      round(&mut v, &m, 2);
      round(&mut v, &m, 3);
      round(&mut v, &m, 4);
      round(&mut v, &m, 5);
      round(&mut v, &m, 6);

      h_vecs[0] = xor(v[0], v[8]);
      h_vecs[1] = xor(v[1], v[9]);
      h_vecs[2] = xor(v[2], v[10]);
      h_vecs[3] = xor(v[3], v[11]);
      h_vecs[4] = xor(v[4], v[12]);
      h_vecs[5] = xor(v[5], v[13]);
      h_vecs[6] = xor(v[6], v[14]);
      h_vecs[7] = xor(v[7], v[15]);
    }

    // Convert word-major vectors into `[chunk][word]` order without scatter.
    let mut lo = [_mm256_setzero_si256(); 8];
    let mut hi = [_mm256_setzero_si256(); 8];
    for i in 0..8 {
      lo[i] = _mm512_castsi512_si256(h_vecs[i]);
      hi[i] = _mm512_extracti64x4_epi64(h_vecs[i], 1);
    }

    transpose8x8(&mut lo);
    transpose8x8(&mut hi);

    for chunk in 0..8 {
      storeu256(lo[chunk], out.add(chunk.strict_mul(OUT_LEN)));
      storeu256(hi[chunk], out.add(chunk.strict_add(8).strict_mul(OUT_LEN)));
    }
  }
}

/// Owned Rust-intrinsic implementation of the 16-way contiguous chunk kernel.
///
/// This is optimized for the contiguous chunk hashing hot path, where inputs
/// are arranged as `CHUNK_LEN`-byte blocks back-to-back.
///
/// # Safety
///
/// Caller must ensure AVX-512F + AVX-512VL + AVX-512DQ + AVX2 are available,
/// and `input`/`out` are valid for `DEGREE * CHUNK_LEN` and
/// `DEGREE * OUT_LEN` bytes respectively.
#[target_feature(enable = "avx512f,avx512vl,avx512dq,avx2")]
pub(crate) unsafe fn hash16_contiguous_owned(input: *const u8, key: &[u32; 8], counter: u64, flags: u32, out: *mut u8) {
  // SAFETY: Forward to the generic owned AVX-512 hash-many implementation because:
  // 1. The caller guarantees AVX-512F/VL/DQ plus AVX2 availability.
  // 2. The constructed lane pointers stay within the contiguous 16-chunk input.
  // 3. `out` is writable for `DEGREE * OUT_LEN` bytes.
  // 4. `hash16_owned` receives a full 16-lane batch and the chunk start/end flags.
  unsafe {
    let inputs = [
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
    hash16_owned(HashManyRequest {
      inputs: &inputs,
      blocks: CHUNK_LEN / BLOCK_LEN,
      key,
      counter,
      increment_counter: true,
      flags,
      flags_start: super::super::CHUNK_START,
      flags_end: super::super::CHUNK_END,
      out,
    });
  }
}

/// Compress one block of 16 lanes into their chaining values.
///
/// # Safety
///
/// AVX-512F, AVX-512DQ, and AVX2 must be available. Every input must be
/// readable for the complete block at `block`.
#[inline(always)]
unsafe fn chunk_block16(
  h_vecs: &mut [__m512i; 8],
  inputs: &[*const u8; DEGREE],
  block: usize,
  flags: u32,
  counter_low_vec: __m512i,
  counter_high_vec: __m512i,
) {
  // SAFETY: the caller establishes the target features and readable blocks.
  unsafe {
    let mut block_flags = flags;
    if block == 0 {
      block_flags |= super::super::CHUNK_START;
    }
    if block.strict_add(1) == CHUNK_LEN / BLOCK_LEN {
      block_flags |= super::super::CHUNK_END;
    }
    let m = transpose_msg_vecs16(inputs, block.strict_mul(BLOCK_LEN));
    let mut v = [
      h_vecs[0],
      h_vecs[1],
      h_vecs[2],
      h_vecs[3],
      h_vecs[4],
      h_vecs[5],
      h_vecs[6],
      h_vecs[7],
      set1(IV[0]),
      set1(IV[1]),
      set1(IV[2]),
      set1(IV[3]),
      counter_low_vec,
      counter_high_vec,
      set1(BLOCK_LEN_U32),
      set1(block_flags),
    ];
    round(&mut v, &m, 0);
    round(&mut v, &m, 1);
    round(&mut v, &m, 2);
    round(&mut v, &m, 3);
    round(&mut v, &m, 4);
    round(&mut v, &m, 5);
    round(&mut v, &m, 6);
    for word in 0..8 {
      h_vecs[word] = xor(v[word], v[word.strict_add(8)]);
    }
  }
}

/// Finish complete chunks while retaining the following chunk's unfinished state.
///
/// Lanes before `whole = input.len() / CHUNK_LEN` receive complete chunk CVs.
/// Lane `whole` receives the partial chunk's CV before its last buffered block;
/// the return value counts the blocks already compressed there. The caller
/// retains the remaining one to 64 bytes. Later lanes are left unchanged.
///
/// # Safety
///
/// Caller must ensure AVX-512F + AVX-512VL + AVX-512DQ + AVX2 are available.
/// `input` must contain 1..=15 complete chunks followed by 1..=1023 bytes, and
/// `counter` must permit `whole` successors. The input and output borrows must
/// be disjoint. `key` and `flags` define the same mode as the caller's chunk state.
#[target_feature(enable = "avx512f,avx512vl,avx512dq,avx2")]
pub(crate) unsafe fn hash_chunks_and_partial16(
  input: &[u8],
  key: &[u32; 8],
  counter: u64,
  flags: u32,
  out: &mut [[u32; 8]; DEGREE],
) -> u8 {
  let whole = input.len().strict_div(CHUNK_LEN);
  let partial_len = input.len().strict_rem(CHUNK_LEN);
  debug_assert!((1..DEGREE).contains(&whole));
  debug_assert!(partial_len != 0);
  debug_assert!(
    counter
      .checked_add(u64::try_from(whole).expect("BLAKE3 lane count fits in u64"))
      .is_some()
  );
  let partial_blocks = partial_len.strict_sub(1).strict_div(BLOCK_LEN);
  out[whole] = *key;

  // SAFETY: the caller establishes AVX-512F/VL/DQ plus AVX2, lengths and
  // counter bounds. Lanes before `whole` read complete chunks. Lane `whole`
  // reads only the blocks preceding its last buffered block; later blocks and
  // unused lanes reread the first complete chunk, and their CVs are never
  // stored. Each store targets a distinct `[u32; 8]` output below `whole`.
  unsafe {
    let base = input.as_ptr();
    let mut inputs = [base; DEGREE];
    for (lane, ptr) in inputs.iter_mut().enumerate().take(whole) {
      *ptr = base.add(lane.strict_mul(CHUNK_LEN));
    }
    inputs[whole] = base.add(whole.strict_mul(CHUNK_LEN));
    let (counter_low_vec, counter_high_vec) = counter_vec(counter, true);
    let mut h_vecs = [
      set1(key[0]),
      set1(key[1]),
      set1(key[2]),
      set1(key[3]),
      set1(key[4]),
      set1(key[5]),
      set1(key[6]),
      set1(key[7]),
    ];

    for block in 0..partial_blocks {
      chunk_block16(&mut h_vecs, &inputs, block, flags, counter_low_vec, counter_high_vec);
    }
    if partial_blocks != 0 {
      let partial_lane = _mm512_set1_epi32(i32::try_from(whole).expect("BLAKE3 lane index fits in i32"));
      for (word, saved) in out[whole].iter_mut().enumerate() {
        let lane = _mm512_permutexvar_epi32(partial_lane, h_vecs[word]);
        *saved = _mm_cvtsi128_si32(_mm512_castsi512_si128(lane)).cast_unsigned();
      }
    }
    inputs[whole] = base;
    for block in partial_blocks..CHUNK_LEN / BLOCK_LEN {
      chunk_block16(&mut h_vecs, &inputs, block, flags, counter_low_vec, counter_high_vec);
    }

    // Convert word-major vectors into `[chunk][word]` order without scatter.
    let mut lo = [_mm256_setzero_si256(); 8];
    let mut hi = [_mm256_setzero_si256(); 8];
    for i in 0..8 {
      lo[i] = _mm512_castsi512_si256(h_vecs[i]);
      hi[i] = _mm512_extracti64x4_epi64(h_vecs[i], 1);
    }
    transpose8x8(&mut lo);
    transpose8x8(&mut hi);
    for (chunk, cv) in out.iter_mut().enumerate().take(whole) {
      let words = if chunk < 8 { lo[chunk] } else { hi[chunk.strict_sub(8)] };
      storeu256(words, cv.as_mut_ptr().cast::<u8>());
    }
  }
  u8::try_from(partial_blocks).expect("unfinished BLAKE3 chunk has fewer than sixteen compressed blocks")
}

/// Finish three chunks and retain the fourth chunk's unfinished state with
/// AVX-512VL rotates.
///
/// See [`super::sse41::hash3_and_partial4`] for the output contract.
///
/// # Safety
///
/// AVX-512F, AVX-512VL, SSE4.1, and SSSE3 must be available. The remaining
/// contract is that of [`super::sse41::hash3_and_partial4`].
#[target_feature(enable = "avx512f,avx512vl,sse4.1,ssse3")]
pub(crate) unsafe fn hash3_and_partial4_avx512vl(
  input: &[u8],
  key: &[u32; 8],
  counter: u64,
  flags: u32,
  out: &mut [[u32; 8]; DEGREE],
) -> u8 {
  // SAFETY: this function enables a superset of the body's SSE4.1/SSSE3
  // requirement, and the caller upholds its input, counter and output contract.
  unsafe { super::sse41::hash3_and_partial4(input, key, counter, flags, out) }
}

/// Hash 16 contiguous independent inputs in parallel.
///
/// This is optimized for the contiguous chunk hashing hot path, where inputs
/// are arranged as `CHUNK_LEN`-byte blocks back-to-back.
///
/// # Safety
///
/// Caller must ensure AVX-512 is available, and `input`/`out` are valid for
/// `DEGREE * CHUNK_LEN` and `DEGREE * OUT_LEN` bytes respectively.
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
#[target_feature(enable = "avx512f,avx512vl,avx512dq,avx2")]
pub(crate) unsafe fn hash16_contiguous(input: *const u8, key: &[u32; 8], counter: u64, flags: u32, out: *mut u8) {
  // SAFETY: Forwarding to the owned AVX-512 implementation because:
  // 1. This function has the same AVX-512F/VL/DQ + AVX2 target-feature requirement.
  // 2. The caller's pointer/output contract is identical to `hash16_contiguous_owned`.
  unsafe { hash16_contiguous_owned(input, key, counter, flags, out) }
}

/// Generate 16 root output blocks (64 bytes each) in parallel.
///
/// Each lane uses an independent `output_block_counter` (`counter + lane`), but
/// shares the same `chaining_value`, `block_words`, `block_len`, and `flags`.
///
/// # Safety
/// Caller must ensure AVX-512 is available and that `out` is valid for `16 * 64`
/// writable bytes.
#[target_feature(enable = "avx512f,avx512vl,avx2")]
pub(crate) unsafe fn root_output_blocks16(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
  out: *mut u8,
) {
  #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
  {
    let block_len = u8::try_from(block_len).expect("BLAKE3 block length must fit the assembly ABI");
    let flags = u8::try_from(flags).expect("BLAKE3 flags must fit the assembly ABI");
    // SAFETY: AVX-512 XOF assembly call because:
    // 1. This target-feature function requires the AVX-512 features used by the wrapper.
    // 2. `chaining_value` and `block_words` are fixed-size readable arrays.
    // 3. The caller guarantees `out` is writable for `16 * 64` bytes.
    // 4. `block_len` and `flags` were converted without loss to the assembly ABI types.
    // 5. Counters, block length, flags, and output block count are public values.
    unsafe {
      super::asm::xof_many_avx512(
        chaining_value.as_ptr(),
        block_words.as_ptr().cast(),
        block_len,
        counter,
        flags,
        out,
        16,
      );
    }
  }

  #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
  // SAFETY: Running the intrinsic AVX-512 root-output implementation because this function's
  // target-feature contract provides AVX-512F/VL and AVX2, and the caller provides 16 writable
  // output blocks.
  unsafe {
    let cv_vecs = [
      set1(chaining_value[0]),
      set1(chaining_value[1]),
      set1(chaining_value[2]),
      set1(chaining_value[3]),
      set1(chaining_value[4]),
      set1(chaining_value[5]),
      set1(chaining_value[6]),
      set1(chaining_value[7]),
    ];

    let m = [
      set1(block_words[0]),
      set1(block_words[1]),
      set1(block_words[2]),
      set1(block_words[3]),
      set1(block_words[4]),
      set1(block_words[5]),
      set1(block_words[6]),
      set1(block_words[7]),
      set1(block_words[8]),
      set1(block_words[9]),
      set1(block_words[10]),
      set1(block_words[11]),
      set1(block_words[12]),
      set1(block_words[13]),
      set1(block_words[14]),
      set1(block_words[15]),
    ];

    let (counter_low_vec, counter_high_vec) = counter_vec(counter, true);
    let block_len_vec = set1(block_len);
    let flags_vec = set1(flags);

    let iv0 = set1(IV[0]);
    let iv1 = set1(IV[1]);
    let iv2 = set1(IV[2]);
    let iv3 = set1(IV[3]);

    let mut v = [
      cv_vecs[0],
      cv_vecs[1],
      cv_vecs[2],
      cv_vecs[3],
      cv_vecs[4],
      cv_vecs[5],
      cv_vecs[6],
      cv_vecs[7],
      iv0,
      iv1,
      iv2,
      iv3,
      counter_low_vec,
      counter_high_vec,
      block_len_vec,
      flags_vec,
    ];

    round(&mut v, &m, 0);
    round(&mut v, &m, 1);
    round(&mut v, &m, 2);
    round(&mut v, &m, 3);
    round(&mut v, &m, 4);
    round(&mut v, &m, 5);
    round(&mut v, &m, 6);

    let out_words = [
      xor(v[0], v[8]),
      xor(v[1], v[9]),
      xor(v[2], v[10]),
      xor(v[3], v[11]),
      xor(v[4], v[12]),
      xor(v[5], v[13]),
      xor(v[6], v[14]),
      xor(v[7], v[15]),
      xor(v[8], cv_vecs[0]),
      xor(v[9], cv_vecs[1]),
      xor(v[10], cv_vecs[2]),
      xor(v[11], cv_vecs[3]),
      xor(v[12], cv_vecs[4]),
      xor(v[13], cv_vecs[5]),
      xor(v[14], cv_vecs[6]),
      xor(v[15], cv_vecs[7]),
    ];

    let mut lo0 = [_mm256_setzero_si256(); 8];
    let mut hi0 = [_mm256_setzero_si256(); 8];
    let mut lo1 = [_mm256_setzero_si256(); 8];
    let mut hi1 = [_mm256_setzero_si256(); 8];

    for i in 0..8 {
      lo0[i] = _mm512_castsi512_si256(out_words[i]);
      hi0[i] = _mm512_extracti64x4_epi64(out_words[i], 1);
      lo1[i] = _mm512_castsi512_si256(out_words[i.strict_add(8)]);
      hi1[i] = _mm512_extracti64x4_epi64(out_words[i.strict_add(8)], 1);
    }

    transpose8x8(&mut lo0);
    transpose8x8(&mut hi0);
    transpose8x8(&mut lo1);
    transpose8x8(&mut hi1);

    for lane in 0usize..8 {
      let base = out.add(lane.strict_mul(64));
      storeu256(lo0[lane], base);
      storeu256(lo1[lane], base.add(32));
    }
    for lane in 0usize..8 {
      let base = out.add(lane.strict_add(8).strict_mul(64));
      storeu256(hi0[lane], base);
      storeu256(hi1[lane], base.add(32));
    }
  }
}

/// Generate one or more root output blocks with consecutive counters.
///
/// # Safety
/// Caller must ensure AVX-512 is available and that `out` is valid for
/// `blocks * 64` writable bytes.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
pub(crate) unsafe fn root_output_blocks(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
  out: *mut u8,
  blocks: usize,
) {
  debug_assert!(blocks != 0);
  let block_len = u8::try_from(block_len).expect("BLAKE3 block length must fit the assembly ABI");
  let flags = u8::try_from(flags).expect("BLAKE3 flags must fit the assembly ABI");
  // SAFETY: AVX-512 XOF assembly call because:
  // 1. Dispatch only selects this function for the AVX-512 kernel.
  // 2. The caller guarantees `out` is writable for `blocks * 64` bytes.
  // 3. `block_words` is a readable 64-byte block and `chaining_value` has 8 words.
  // 4. `block_len` and `flags` were converted without loss to the assembly ABI types.
  // 5. Counters, block length, flags, and output block count are public values.
  unsafe {
    super::asm::xof_many_avx512(
      chaining_value.as_ptr(),
      block_words.as_ptr().cast(),
      block_len,
      counter,
      flags,
      out,
      blocks,
    );
  }
}

/// Generate 1 root output block (64 bytes).
/// Delegates to the SSE4.1 row-wise emitter for the single-block case.
///
/// # Safety
/// Caller must ensure AVX-512 is available and that `out` is valid for `64` writable bytes.
#[target_feature(enable = "avx512f,avx512vl,avx2")]
pub(crate) unsafe fn root_output_blocks1(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
  out: *mut u8,
) {
  // SAFETY: AVX-512 implies the delegated SSE4.1 path is legal on x86_64, and
  // callers guarantee that `out` is valid for one 64-byte output block.
  unsafe { super::sse41::root_output_blocks1(chaining_value, block_words, counter, block_len, flags, out) }
}

/// Generate 2 root output blocks (128 bytes) with consecutive counters.
/// Delegates to the SSE4.1 row-wise emitter for the two-block case.
///
/// # Safety
/// Caller must ensure AVX-512 is available and that `out` is valid for `128` writable bytes.
#[target_feature(enable = "avx512f,avx512vl,avx2")]
pub(crate) unsafe fn root_output_blocks2(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
  out: *mut u8,
) {
  // SAFETY: AVX-512 implies the delegated SSE4.1 path is legal on x86_64, and
  // callers guarantee that `out` is valid for two 64-byte output blocks.
  unsafe { super::sse41::root_output_blocks2(chaining_value, block_words, counter, block_len, flags, out) }
}

/// Compress one BLAKE3 block with a latency-oriented schedule.
///
/// This uses the same dependency-chain schedule as the SSE4.1/AVX2 single-block
/// path (no 16-lane broadcast + lane extraction), while keeping this entrypoint
/// AVX-512-gated for mixed-workload dispatching.
///
/// # Safety
/// Caller must ensure AVX-512F + AVX-512VL + AVX2 + SSE4.1 + SSSE3 are available.
#[target_feature(enable = "avx512f,avx512vl,avx2,sse4.1,ssse3")]
pub(crate) unsafe fn compress_block(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  // SAFETY: AVX-512/AVX2/SSE4.1/SSSE3 intrinsics are available via this function's
  // #[target_feature] attribute. Pointer accesses are to valid fixed-size array references.
  unsafe {
    let m0 = _mm_loadu_si128(block_words.as_ptr().cast());
    let m1 = _mm_loadu_si128(block_words.as_ptr().add(4).cast());
    let m2 = _mm_loadu_si128(block_words.as_ptr().add(8).cast());
    let m3 = _mm_loadu_si128(block_words.as_ptr().add(12).cast());
    let [mut row0, mut row1, mut row2, mut row3] =
      super::compress_pre_sse41_impl(chaining_value, [m0, m1, m2, m3], counter, block_len, flags);

    let cv_lo = _mm_loadu_si128(chaining_value.as_ptr().cast());
    let cv_hi = _mm_loadu_si128(chaining_value.as_ptr().add(4).cast());

    row0 = _mm_xor_si128(row0, row2);
    row1 = _mm_xor_si128(row1, row3);
    row2 = _mm_xor_si128(row2, cv_lo);
    row3 = _mm_xor_si128(row3, cv_hi);

    let mut out = [0u32; 16];
    _mm_storeu_si128(out.as_mut_ptr().cast(), row0);
    _mm_storeu_si128(out.as_mut_ptr().add(4).cast(), row1);
    _mm_storeu_si128(out.as_mut_ptr().add(8).cast(), row2);
    _mm_storeu_si128(out.as_mut_ptr().add(12).cast(), row3);
    out
  }
}
