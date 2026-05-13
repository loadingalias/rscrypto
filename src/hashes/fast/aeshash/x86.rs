#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{
  BLOCK_SIZE, GX_KEYS, HASH64_GX_LONG_THRESHOLD, KEYS, LONG_CHUNK_SIZE, RAPID_HI_SEED, len_hi, len_lo, rapid_mix,
  read_u64_le, seed_hi, seed_lo,
};

/// 32-byte 64-bit hash with the x86 AES-NI backend.
///
/// # Safety
///
/// Caller must ensure the current CPU supports AES-NI and SSE2.
#[target_feature(enable = "aes", enable = "sse2")]
pub(super) unsafe fn hash64_32(seed: u64, data: &[u8]) -> u64 {
  debug_assert_eq!(data.len(), 2 * BLOCK_SIZE);
  // SAFETY: this whole function is entered only with AES-NI + SSE2 available.
  unsafe { hash64_32_inner(seed, data) }
}

/// 32-byte 64-bit hash with compile-time AES-NI + SSE2 enabled.
///
/// # Safety
///
/// Caller must ensure the current compilation unit and CPU support AES-NI and SSE2.
#[inline(always)]
pub(super) unsafe fn hash64_32_static(seed: u64, data: &[u8]) -> u64 {
  debug_assert_eq!(data.len(), 2 * BLOCK_SIZE);
  // SAFETY: caller guarantees AES-NI + SSE2 availability.
  unsafe { hash64_32_inner(seed, data) }
}

/// Folded 64-bit hash with the x86 AES-NI backend.
///
/// # Safety
///
/// Caller must ensure the current CPU supports AES-NI and SSE2.
#[target_feature(enable = "aes", enable = "sse2")]
pub(super) unsafe fn hash64_fold(seed: u64, data: &[u8]) -> u64 {
  debug_assert_eq!(data.len(), 4 * BLOCK_SIZE);
  // SAFETY: this whole function is entered only with AES-NI + SSE2 available.
  unsafe { hash64_fold_inner(seed, data) }
}

/// Folded 64-bit hash with compile-time AES-NI + SSE2 enabled.
///
/// # Safety
///
/// Caller must ensure the current compilation unit and CPU support AES-NI and SSE2.
#[inline(always)]
pub(super) unsafe fn hash64_fold_static(seed: u64, data: &[u8]) -> u64 {
  debug_assert_eq!(data.len(), 4 * BLOCK_SIZE);
  // SAFETY: caller guarantees AES-NI + SSE2 availability.
  unsafe { hash64_fold_inner(seed, data) }
}

/// Low-lane 64-bit hash with the x86 AES-NI backend.
///
/// # Safety
///
/// Caller must ensure the current CPU supports AES-NI and SSE2.
#[target_feature(enable = "aes", enable = "sse2")]
pub(super) unsafe fn hash64_low(seed: u64, data: &[u8]) -> u64 {
  debug_assert!(super::hash64_low_lane_len(data.len()));
  // SAFETY: this whole function is entered only with AES-NI + SSE2 available.
  unsafe { hash64_low_inner(seed, data) }
}

/// Low-lane 64-bit hash with compile-time AES-NI + SSE2 enabled.
///
/// # Safety
///
/// Caller must ensure the current compilation unit and CPU support AES-NI and SSE2.
#[inline(always)]
pub(super) unsafe fn hash64_low_static(seed: u64, data: &[u8]) -> u64 {
  debug_assert!(super::hash64_low_lane_len(data.len()));
  // SAFETY: caller guarantees AES-NI + SSE2 availability.
  unsafe { hash64_low_inner(seed, data) }
}

/// Hash with the x86 AES-NI backend.
///
/// # Safety
///
/// Caller must ensure the current CPU supports AES-NI and SSE2.
#[target_feature(enable = "aes", enable = "sse2")]
pub(super) unsafe fn hash128(seed: u64, data: &[u8]) -> u128 {
  // SAFETY: this whole function is entered only with AES-NI + SSE2 available.
  unsafe { hash128_inner(seed, data) }
}

/// Hash with compile-time AES-NI + SSE2 enabled.
///
/// # Safety
///
/// Caller must ensure the current compilation unit and CPU support AES-NI and SSE2.
#[inline(always)]
pub(super) unsafe fn hash128_static(seed: u64, data: &[u8]) -> u128 {
  // SAFETY: caller guarantees AES-NI + SSE2 availability.
  unsafe { hash128_inner(seed, data) }
}

#[inline(always)]
unsafe fn hash64_32_inner(seed: u64, data: &[u8]) -> u64 {
  debug_assert_eq!(data.len(), 2 * BLOCK_SIZE);
  let a = read_u64_le(data, 0) ^ read_u64_le(data, 2 * BLOCK_SIZE - 16) ^ seed.rotate_left(17);
  let b = read_u64_le(data, 8) ^ read_u64_le(data, 2 * BLOCK_SIZE - 8) ^ RAPID_HI_SEED ^ (2 * BLOCK_SIZE as u64);
  rapid_mix(a, b)
}

#[inline(always)]
unsafe fn hash64_fold_inner(seed: u64, data: &[u8]) -> u64 {
  debug_assert_eq!(data.len(), 4 * BLOCK_SIZE);
  let ptr = data.as_ptr();
  // SAFETY: four-block cross-lane AES fold because:
  // 1. The exact-length guard proves `ptr..ptr+64` is readable.
  // 2. The caller guarantees AES-NI + SSE2 for unaligned loads, XOR, lane rotation, and AES.
  // 3. Cross-folding the high lane into the low lane matches the portable exact-64 implementation.
  let h = unsafe {
    let mix01 = _mm_xor_si128(load_ptr(ptr), load_ptr(ptr.add(BLOCK_SIZE)));
    let mix23 = _mm_xor_si128(load_ptr(ptr.add(2 * BLOCK_SIZE)), load_ptr(ptr.add(3 * BLOCK_SIZE)));
    let mix = _mm_xor_si128(mix01, mix23);
    let folded = _mm_xor_si128(mix, rot8(mix));
    aesenclast(folded, len_block(seed, 4 * BLOCK_SIZE))
  };
  // SAFETY: caller guarantees SSE2 for low-lane extraction.
  unsafe { to_u64(h) }
}

#[inline(always)]
unsafe fn hash64_low_inner(seed: u64, data: &[u8]) -> u64 {
  debug_assert!(super::hash64_low_lane_len(data.len()));
  if data.len() >= HASH64_GX_LONG_THRESHOLD {
    // SAFETY: caller guarantees AES-NI + SSE2 for the gx-style long compressor and lane extraction.
    let h = unsafe { gx_finalize(aesenc(gx_compress_all(data), gx_seed(seed))) };
    // SAFETY: caller guarantees SSE2 for low-lane extraction.
    return unsafe { to_u64(h) };
  }
  // SAFETY: caller guarantees AES-NI + SSE2 for compression and lane extraction.
  unsafe { to_u64(hash_vector(seed, data)) }
}

#[inline(always)]
unsafe fn hash128_inner(seed: u64, data: &[u8]) -> u128 {
  if data.len() <= BLOCK_SIZE {
    return super::short_hash128(seed, data);
  }
  // SAFETY: caller guarantees AES-NI + SSE2 for compression and extraction.
  unsafe { to_u128(hash_vector(seed, data)) }
}

#[inline(always)]
unsafe fn hash_vector(seed: u64, data: &[u8]) -> __m128i {
  debug_assert!(data.len() > BLOCK_SIZE);
  if data.len() <= 2 * BLOCK_SIZE {
    // SAFETY: caller guarantees AES-NI + SSE2 for all helper calls.
    return unsafe { hash_17to32(seed, data) };
  }
  if data.len() <= 4 * BLOCK_SIZE {
    // SAFETY: caller guarantees AES-NI + SSE2 for all helper calls.
    return unsafe { hash_33to64(seed, data) };
  }
  // SAFETY: caller guarantees AES-NI + SSE2 for all helper calls.
  unsafe { hash_long(seed, data) }
}

#[inline(always)]
unsafe fn hash_17to32(seed: u64, data: &[u8]) -> __m128i {
  // SAFETY: caller guarantees SSE2 for vector construction and AES-NI for AES rounds.
  let s0 = unsafe { seed_block(seed, 0) };

  let tail = if data.len() == 2 * BLOCK_SIZE {
    // SAFETY: length check proves the second full block is readable.
    unsafe { load_ptr(data.as_ptr().add(BLOCK_SIZE)) }
  } else {
    // SAFETY: partial block is copied into a stack-backed 16-byte buffer.
    unsafe { load_partial(&data[BLOCK_SIZE..], data.len()) }
  };

  // SAFETY: first full block is readable because len > 16; AES-NI + SSE2 are available.
  let h = unsafe {
    aesenc(
      _mm_xor_si128(load_ptr(data.as_ptr()), s0),
      _mm_xor_si128(tail, len_block(seed, data.len())),
    )
  };
  h
}

#[inline(always)]
unsafe fn hash_33to64(seed: u64, data: &[u8]) -> __m128i {
  let ptr = data.as_ptr();

  if data.len() == 4 * BLOCK_SIZE {
    // SAFETY: exact length proves four full blocks are readable.
    return unsafe { hash_64(seed, ptr) };
  }

  if data.len() <= 3 * BLOCK_SIZE {
    // SAFETY: caller guarantees SSE2 for vector construction and key loads.
    let (s1, k1) = unsafe { (seed_block(seed, 1), key(1)) };
    let tail = if data.len() == 3 * BLOCK_SIZE {
      // SAFETY: length check proves the third full block is readable.
      unsafe { load_ptr(ptr.add(2 * BLOCK_SIZE)) }
    } else {
      // SAFETY: partial block is copied into a stack-backed 16-byte buffer.
      unsafe { load_partial(&data[2 * BLOCK_SIZE..], data.len()) }
    };
    // SAFETY: the first two full blocks are readable because len > 32; AES-NI + SSE2 are available.
    let h0 = unsafe { aesenc(_mm_xor_si128(load_ptr(ptr), tail), len_block(seed, data.len())) };
    // SAFETY: the second full block is readable because len > 32; AES-NI + SSE2 are available.
    let h1 = unsafe { aesenc(_mm_xor_si128(load_ptr(ptr.add(BLOCK_SIZE)), s1), k1) };
    // SAFETY: caller guarantees SSE2 for XOR and extraction.
    return unsafe { _mm_xor_si128(h0, h1) };
  }

  let tail = if data.len() == 4 * BLOCK_SIZE {
    // SAFETY: length check proves the fourth full block is readable.
    unsafe { load_ptr(ptr.add(3 * BLOCK_SIZE)) }
  } else {
    // SAFETY: partial block is copied into a stack-backed 16-byte buffer.
    unsafe { load_partial(&data[3 * BLOCK_SIZE..], data.len()) }
  };
  // SAFETY: caller guarantees SSE2 for vector construction.
  let s0 = unsafe { seed_block(seed, 0) };
  // SAFETY: the first and third full blocks are readable because len > 48; AES-NI + SSE2 are
  // available.
  let h0 = unsafe { aesenc(_mm_xor_si128(load_ptr(ptr), load_ptr(ptr.add(2 * BLOCK_SIZE))), s0) };
  // SAFETY: the second full block is readable because len > 48; AES-NI + SSE2 are available.
  let h1 = unsafe {
    aesenc(
      _mm_xor_si128(load_ptr(ptr.add(BLOCK_SIZE)), tail),
      len_block(seed, data.len()),
    )
  };
  // SAFETY: caller guarantees AES-NI + SSE2 for finalization and extraction.
  unsafe { aesenclast(_mm_xor_si128(h0, h1), key(7)) }
}

#[inline(always)]
unsafe fn hash_64(seed: u64, ptr: *const u8) -> __m128i {
  // SAFETY: caller guarantees SSE2 and four readable blocks.
  let a = unsafe { hash_64_mix(ptr) };
  let b = unsafe { _mm_xor_si128(seed_block(seed, 0), len_block(seed, 4 * BLOCK_SIZE)) };
  // SAFETY: caller guarantees AES-NI + SSE2.
  unsafe { aesenc(a, b) }
}

#[inline(always)]
unsafe fn hash_64_mix(ptr: *const u8) -> __m128i {
  // SAFETY: caller guarantees four readable blocks.
  let (v0, v1, v2, v3) = unsafe {
    (
      load_ptr(ptr),
      load_ptr(ptr.add(BLOCK_SIZE)),
      load_ptr(ptr.add(2 * BLOCK_SIZE)),
      load_ptr(ptr.add(3 * BLOCK_SIZE)),
    )
  };
  // SAFETY: caller guarantees SSE2 for byte rotation and vector construction.
  unsafe { _mm_xor_si128(_mm_xor_si128(v0, rot3(v1)), _mm_xor_si128(rot7(v2), rot11(v3))) }
}

#[inline(always)]
unsafe fn hash_long(seed: u64, data: &[u8]) -> __m128i {
  if data.len() >= LONG_CHUNK_SIZE && data.len() % LONG_CHUNK_SIZE == 0 {
    // SAFETY: caller guarantees AES-NI + SSE2 and the length check guarantees full 8-block chunks.
    return unsafe { fast_aligned_hash128(seed, data) };
  }
  // SAFETY: caller guarantees AES-NI + SSE2 for the compression schedule.
  let compressed = unsafe { gx_compress_all(data) };
  // SAFETY: caller guarantees AES-NI + SSE2 for finalization.
  unsafe { gx_finalize(aesenc(compressed, gx_seed(seed))) }
}

#[inline(always)]
unsafe fn fast_aligned_hash128(seed: u64, data: &[u8]) -> __m128i {
  let ptr = data.as_ptr();
  let mut offset = 0usize;
  // SAFETY: caller guarantees SSE2 for key and seed construction.
  let mut lane = unsafe { _mm_xor_si128(gx_seed(seed), gx_key(0)) };

  while offset < data.len() {
    // SAFETY: caller guarantees `data.len()` is a multiple of 128, so every load in this chunk is
    // a full in-bounds block.
    let (v0, v1, v2, v3, v4, v5, v6, v7) = unsafe {
      (
        load_ptr(ptr.add(offset)),
        load_ptr(ptr.add(offset + 16)),
        load_ptr(ptr.add(offset + 32)),
        load_ptr(ptr.add(offset + 48)),
        load_ptr(ptr.add(offset + 64)),
        load_ptr(ptr.add(offset + 80)),
        load_ptr(ptr.add(offset + 96)),
        load_ptr(ptr.add(offset + 112)),
      )
    };

    // SAFETY: caller guarantees AES-NI + SSE2.
    let a = unsafe { aesenc(_mm_xor_si128(v0, v4), _mm_xor_si128(v1, v5)) };
    let b = unsafe { aesenc(_mm_xor_si128(v2, v6), _mm_xor_si128(v3, v7)) };
    // SAFETY: caller guarantees SSE2 for vector construction.
    let tweak = unsafe { _mm_set1_epi32(((offset as u32) ^ (data.len() as u32)) as i32) };
    // SAFETY: caller guarantees AES-NI + SSE2.
    lane = unsafe { aesenclast(_mm_xor_si128(_mm_xor_si128(a, b), tweak), lane) };

    offset += LONG_CHUNK_SIZE;
  }

  // SAFETY: caller guarantees AES-NI + SSE2 for finalization and extraction.
  unsafe { gx_finalize(lane) }
}

#[inline(always)]
unsafe fn gx_compress_all(data: &[u8]) -> __m128i {
  let len = data.len();
  debug_assert!(len > 3 * BLOCK_SIZE);
  let ptr = data.as_ptr();
  let mut offset;
  let mut hash_vector;

  let extra = len % BLOCK_SIZE;
  if extra == 0 {
    // SAFETY: len > 48, so the first full block is readable.
    hash_vector = unsafe { load_ptr(ptr) };
    offset = BLOCK_SIZE;
  } else {
    // SAFETY: copies the partial prefix into a stack-backed 16-byte buffer.
    hash_vector = unsafe { gx_partial(&data[..extra], extra) };
    offset = extra;
  }

  // SAFETY: len > 48 leaves at least one full block after the prefix handling above.
  let mut v0 = unsafe { load_ptr(ptr.add(offset)) };
  offset += BLOCK_SIZE;

  if len > 2 * BLOCK_SIZE {
    // SAFETY: gxhash prefix schedule guarantees this full block is readable for len > 32.
    let v = unsafe { load_ptr(ptr.add(offset)) };
    offset += BLOCK_SIZE;
    // SAFETY: caller guarantees AES-NI + SSE2.
    v0 = unsafe { aesenc(v0, v) };

    if len > 3 * BLOCK_SIZE {
      // SAFETY: gxhash prefix schedule guarantees this full block is readable for len > 48.
      let v = unsafe { load_ptr(ptr.add(offset)) };
      offset += BLOCK_SIZE;
      // SAFETY: caller guarantees AES-NI + SSE2.
      v0 = unsafe { aesenc(v0, v) };

      if len > 4 * BLOCK_SIZE {
        // SAFETY: caller guarantees AES-NI + SSE2 for the remaining compression schedule.
        hash_vector = unsafe { gx_compress_many(data, offset, hash_vector, len) };
      }
    }
  }
  // SAFETY: caller guarantees AES-NI + SSE2 for final prefix merge.
  unsafe { aesenclast(hash_vector, aesenc(aesenc(v0, gx_key(0)), gx_key(4))) }
}

#[inline(always)]
unsafe fn gx_compress_many(data: &[u8], mut offset: usize, mut hash_vector: __m128i, len: usize) -> __m128i {
  let remaining = len - offset;
  let unrollable_blocks = remaining / LONG_CHUNK_SIZE * 8;
  let remaining = remaining - unrollable_blocks * BLOCK_SIZE;
  let scalar_end = offset + remaining;
  let ptr = data.as_ptr();

  while offset < scalar_end {
    // SAFETY: scalar prefix only covers full blocks inside `data`.
    let block = unsafe { load_ptr(ptr.add(offset)) };
    // SAFETY: caller guarantees AES-NI + SSE2.
    hash_vector = unsafe { aesenc(hash_vector, block) };
    offset += BLOCK_SIZE;
  }

  // SAFETY: caller guarantees AES-NI + SSE2 for the 8-block loop.
  unsafe { gx_compress_8(data, offset, hash_vector, len) }
}

#[inline(always)]
unsafe fn gx_compress_8(data: &[u8], mut offset: usize, hash_vector: __m128i, len: usize) -> __m128i {
  // SAFETY: caller guarantees SSE2 for vector construction.
  let mut t1 = unsafe { _mm_setzero_si128() };
  let mut t2 = unsafe { _mm_setzero_si128() };
  let mut lane1 = hash_vector;
  let mut lane2 = hash_vector;
  let ptr = data.as_ptr();
  let k0 = unsafe { gx_key(0) };
  let k4 = unsafe { gx_key(4) };

  while offset < len {
    // SAFETY: `offset` is aligned to the 8-block suffix selected by `gx_compress_many`, so all
    // eight loads are full in-bounds blocks.
    let (v0, v1, v2, v3, v4, v5, v6, v7) = unsafe {
      (
        load_ptr(ptr.add(offset)),
        load_ptr(ptr.add(offset + 16)),
        load_ptr(ptr.add(offset + 32)),
        load_ptr(ptr.add(offset + 48)),
        load_ptr(ptr.add(offset + 64)),
        load_ptr(ptr.add(offset + 80)),
        load_ptr(ptr.add(offset + 96)),
        load_ptr(ptr.add(offset + 112)),
      )
    };
    offset += LONG_CHUNK_SIZE;

    // SAFETY: caller guarantees AES-NI + SSE2.
    let mut tmp1 = unsafe { aesenc(v0, v2) };
    let mut tmp2 = unsafe { aesenc(v1, v3) };
    tmp1 = unsafe { aesenc(tmp1, v4) };
    tmp2 = unsafe { aesenc(tmp2, v5) };
    tmp1 = unsafe { aesenc(tmp1, v6) };
    tmp2 = unsafe { aesenc(tmp2, v7) };

    // SAFETY: caller guarantees SSE2 for wrapping byte addition.
    t1 = unsafe { _mm_add_epi8(t1, k0) };
    t2 = unsafe { _mm_add_epi8(t2, k4) };
    // SAFETY: caller guarantees AES-NI + SSE2.
    lane1 = unsafe { aesenclast(aesenc(tmp1, t1), lane1) };
    lane2 = unsafe { aesenclast(aesenc(tmp2, t2), lane2) };
  }

  // SAFETY: caller guarantees SSE2 for wrapping byte addition and vector construction.
  let len_vec = unsafe { _mm_set1_epi32(len as i32) };
  lane1 = unsafe { _mm_add_epi8(lane1, len_vec) };
  lane2 = unsafe { _mm_add_epi8(lane2, len_vec) };
  // SAFETY: caller guarantees AES-NI + SSE2.
  unsafe { aesenc(lane1, lane2) }
}

#[inline(always)]
unsafe fn gx_finalize(hash: __m128i) -> __m128i {
  // SAFETY: caller guarantees AES-NI + SSE2.
  let hash = unsafe { aesenc(hash, gx_key(0)) };
  let hash = unsafe { aesenc(hash, gx_key(4)) };
  unsafe { aesenclast(hash, gx_key(8)) }
}

#[inline(always)]
unsafe fn gx_partial(data: &[u8], len: usize) -> __m128i {
  debug_assert!(data.len() <= BLOCK_SIZE);
  let mut block = [0u8; BLOCK_SIZE];
  block[..data.len()].copy_from_slice(data);
  // SAFETY: `block` is a stack-backed 16-byte buffer and caller guarantees SSE2.
  let partial = unsafe { _mm_loadu_si128(block.as_ptr() as *const __m128i) };
  // SAFETY: caller guarantees SSE2 for wrapping byte addition.
  unsafe { _mm_add_epi8(partial, _mm_set1_epi8(len as i8)) }
}

#[inline(always)]
unsafe fn gx_seed(seed: u64) -> __m128i {
  // SAFETY: caller guarantees SSE2 for vector construction.
  unsafe { _mm_set1_epi64x(seed as i64) }
}

#[inline(always)]
unsafe fn gx_key(offset: usize) -> __m128i {
  debug_assert!(offset + 4 <= GX_KEYS.len());
  // SAFETY: `offset + 4 <= GX_KEYS.len()` and caller guarantees SSE2 for the load.
  unsafe { _mm_loadu_si128(GX_KEYS.as_ptr().add(offset) as *const __m128i) }
}

#[inline(always)]
unsafe fn seed_block(seed: u64, lane: u64) -> __m128i {
  // SAFETY: caller guarantees SSE2 for vector construction.
  unsafe { u64x2(seed_lo(seed, lane), seed_hi(seed, lane)) }
}

#[inline(always)]
unsafe fn len_block(seed: u64, len: usize) -> __m128i {
  // SAFETY: caller guarantees SSE2 for vector construction.
  unsafe { u64x2(len_lo(seed, len), len_hi(seed, len)) }
}

#[inline(always)]
unsafe fn u64x2(lo: u64, hi: u64) -> __m128i {
  // SAFETY: caller guarantees SSE2 for vector construction.
  unsafe { _mm_set_epi64x(hi as i64, lo as i64) }
}

#[inline(always)]
unsafe fn key(index: usize) -> __m128i {
  // SAFETY: `KEYS[index]` is a 16-byte constant and callers only pass 0..8.
  unsafe { _mm_loadu_si128(KEYS[index].as_ptr() as *const __m128i) }
}

#[inline(always)]
unsafe fn load_ptr(ptr: *const u8) -> __m128i {
  // SAFETY: caller guarantees `ptr` points to at least 16 readable bytes.
  unsafe { _mm_loadu_si128(ptr as *const __m128i) }
}

#[inline(always)]
unsafe fn load_partial(data: &[u8], total_len: usize) -> __m128i {
  debug_assert!(data.len() <= BLOCK_SIZE);
  let mut block = [0u8; BLOCK_SIZE];
  block[..data.len()].copy_from_slice(data);
  if data.len() < BLOCK_SIZE {
    block[data.len()] = 0x80;
  }
  let len = (total_len as u64).to_le_bytes();
  for i in 0..8 {
    block[8 + i] ^= len[i];
  }
  // SAFETY: `block` is a stack-backed 16-byte buffer.
  unsafe { _mm_loadu_si128(block.as_ptr() as *const __m128i) }
}

#[inline(always)]
unsafe fn rot3(v: __m128i) -> __m128i {
  // SAFETY: caller guarantees SSE2 for shift/or intrinsics.
  unsafe { _mm_or_si128(_mm_srli_si128::<3>(v), _mm_slli_si128::<13>(v)) }
}

#[inline(always)]
unsafe fn rot7(v: __m128i) -> __m128i {
  // SAFETY: caller guarantees SSE2 for shift/or intrinsics.
  unsafe { _mm_or_si128(_mm_srli_si128::<7>(v), _mm_slli_si128::<9>(v)) }
}

#[inline(always)]
unsafe fn rot8(v: __m128i) -> __m128i {
  // SAFETY: caller guarantees SSE2 for shift/or intrinsics.
  unsafe { _mm_or_si128(_mm_srli_si128::<8>(v), _mm_slli_si128::<8>(v)) }
}

#[inline(always)]
unsafe fn rot11(v: __m128i) -> __m128i {
  // SAFETY: caller guarantees SSE2 for shift/or intrinsics.
  unsafe { _mm_or_si128(_mm_srli_si128::<11>(v), _mm_slli_si128::<5>(v)) }
}

#[inline(always)]
unsafe fn aesenc(data: __m128i, key: __m128i) -> __m128i {
  // SAFETY: caller guarantees AES-NI for AES round intrinsics.
  unsafe { _mm_aesenc_si128(data, key) }
}

#[inline(always)]
unsafe fn aesenclast(data: __m128i, key: __m128i) -> __m128i {
  // SAFETY: caller guarantees AES-NI for AES round intrinsics.
  unsafe { _mm_aesenclast_si128(data, key) }
}

#[inline(always)]
unsafe fn to_u128(v: __m128i) -> u128 {
  // SAFETY: caller guarantees SSE2 for extracting the low lane.
  let lo = unsafe { _mm_cvtsi128_si64(v) as u64 };
  #[cfg(target_arch = "x86_64")]
  {
    // SAFETY: caller guarantees SSE2; extracting the high lane via shift avoids a stack store.
    let hi = unsafe { _mm_cvtsi128_si64(_mm_srli_si128::<8>(v)) as u64 };
    return ((hi as u128) << 64) | lo as u128;
  }
  #[cfg(target_arch = "x86")]
  {
    let mut out = [0u8; BLOCK_SIZE];
    // SAFETY: `out` is a stack-backed 16-byte buffer.
    unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, v) };
    u128::from_le_bytes(out)
  }
}

#[inline(always)]
unsafe fn to_u64(v: __m128i) -> u64 {
  // SAFETY: caller guarantees SSE2 for extracting the low lane.
  unsafe { _mm_cvtsi128_si64(v) as u64 }
}
