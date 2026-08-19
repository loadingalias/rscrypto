//! BLAKE3 x86_64 AVX2 throughput kernel (8-way).

// Supported OS builds still keep the assembly ABI around for sub-degree tails
// and parent reductions. The owned intrinsic body is also used directly for
// full contiguous batches and by diagnostic benches.

use core::arch::x86_64::*;

#[cfg(any(feature = "diag", target_os = "linux", target_os = "macos", target_os = "windows"))]
use super::super::{CHUNK_END, CHUNK_LEN, CHUNK_START, OUT_LEN, PARENT};
use super::{
  super::{BLOCK_LEN, BLOCK_LEN_U32, IV, MSG_SCHEDULE},
  HashManyRequest, counter_high, counter_low,
};

pub(crate) const DEGREE: usize = 8;

/// # Safety
///
/// AVX2 must be available and `src` must be readable for 32 bytes.
#[inline(always)]
unsafe fn loadu(src: *const u8) -> __m256i {
  // SAFETY: Unaligned AVX2 load because:
  // 1. The caller guarantees `src` is readable for 32 bytes.
  // 2. `_mm256_loadu_si256` accepts arbitrary byte alignment.
  // 3. The caller only reaches this helper with AVX2 available.
  unsafe { _mm256_loadu_si256(src.cast()) }
}

/// # Safety
///
/// AVX2 must be available and `dest` must be writable for 32 bytes.
#[inline(always)]
unsafe fn storeu(src: __m256i, dest: *mut u8) {
  // SAFETY: Unaligned AVX2 store because:
  // 1. The caller guarantees `dest` is writable for 32 bytes.
  // 2. `_mm256_storeu_si256` accepts arbitrary byte alignment.
  // 3. The caller only reaches this helper with AVX2 available.
  unsafe { _mm256_storeu_si256(dest.cast(), src) }
}

/// # Safety
///
/// AVX2 must be available.
#[inline(always)]
unsafe fn add(a: __m256i, b: __m256i) -> __m256i {
  // SAFETY: AVX2 lane add because:
  // 1. `a` and `b` are local vector values.
  // 2. The caller only reaches this helper with AVX2 available.
  unsafe { _mm256_add_epi32(a, b) }
}

/// # Safety
///
/// AVX2 must be available.
#[inline(always)]
unsafe fn xor(a: __m256i, b: __m256i) -> __m256i {
  // SAFETY: AVX2 lane xor because:
  // 1. `a` and `b` are local vector values.
  // 2. The caller only reaches this helper with AVX2 available.
  unsafe { _mm256_xor_si256(a, b) }
}

/// # Safety
///
/// AVX2 must be available.
#[inline(always)]
unsafe fn set1(x: u32) -> __m256i {
  // SAFETY: AVX2 scalar broadcast because:
  // 1. Any `u32` bit pattern is valid lane data.
  // 2. The caller only reaches this helper with AVX2 available.
  unsafe { _mm256_set1_epi32(x.cast_signed()) }
}

/// # Safety
///
/// AVX2 must be available.
#[inline(always)]
unsafe fn set8(lanes: [u32; DEGREE]) -> __m256i {
  // SAFETY: AVX2 lane construction because:
  // 1. Any `u32` bit pattern is valid lane data after reinterpretation as `i32`.
  // 2. The caller only reaches this helper with AVX2 available.
  unsafe {
    _mm256_setr_epi32(
      lanes[0].cast_signed(),
      lanes[1].cast_signed(),
      lanes[2].cast_signed(),
      lanes[3].cast_signed(),
      lanes[4].cast_signed(),
      lanes[5].cast_signed(),
      lanes[6].cast_signed(),
      lanes[7].cast_signed(),
    )
  }
}

/// # Safety
///
/// AVX2 must be available.
#[inline(always)]
unsafe fn rot12(x: __m256i) -> __m256i {
  // SAFETY: AVX2 rotate-right-by-12 sequence because:
  // 1. The shifts and or operate only on local vector lanes.
  // 2. The caller only reaches this helper with AVX2 available.
  unsafe { _mm256_or_si256(_mm256_srli_epi32(x, 12), _mm256_slli_epi32(x, 20)) }
}

/// # Safety
///
/// AVX2 must be available.
#[inline(always)]
unsafe fn rot7(x: __m256i) -> __m256i {
  // SAFETY: AVX2 rotate-right-by-7 sequence because:
  // 1. The shifts and or operate only on local vector lanes.
  // 2. The caller only reaches this helper with AVX2 available.
  unsafe { _mm256_or_si256(_mm256_srli_epi32(x, 7), _mm256_slli_epi32(x, 25)) }
}

/// # Safety
///
/// AVX2 must be available and `r` must be in `0..7`.
#[inline(always)]
unsafe fn round(v: &mut [__m256i; 16], m: &[__m256i; 16], r: usize, rot16_mask: __m256i, rot8_mask: __m256i) {
  // SAFETY: One AVX2 BLAKE3 round because:
  // 1. The caller only reaches this helper with AVX2 available.
  // 2. `v` and `m` are fixed-size 16-register arrays.
  // 3. `r` is supplied by callers in the BLAKE3 round range 0..7, so every message-schedule index is
  //    valid.
  // 4. All operations stay within local vector registers.
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
    v[12] = _mm256_shuffle_epi8(v[12], rot16_mask);
    v[13] = _mm256_shuffle_epi8(v[13], rot16_mask);
    v[14] = _mm256_shuffle_epi8(v[14], rot16_mask);
    v[15] = _mm256_shuffle_epi8(v[15], rot16_mask);
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
    v[12] = _mm256_shuffle_epi8(v[12], rot8_mask);
    v[13] = _mm256_shuffle_epi8(v[13], rot8_mask);
    v[14] = _mm256_shuffle_epi8(v[14], rot8_mask);
    v[15] = _mm256_shuffle_epi8(v[15], rot8_mask);
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
    v[15] = _mm256_shuffle_epi8(v[15], rot16_mask);
    v[12] = _mm256_shuffle_epi8(v[12], rot16_mask);
    v[13] = _mm256_shuffle_epi8(v[13], rot16_mask);
    v[14] = _mm256_shuffle_epi8(v[14], rot16_mask);
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
    v[15] = _mm256_shuffle_epi8(v[15], rot8_mask);
    v[12] = _mm256_shuffle_epi8(v[12], rot8_mask);
    v[13] = _mm256_shuffle_epi8(v[13], rot8_mask);
    v[14] = _mm256_shuffle_epi8(v[14], rot8_mask);
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
/// AVX2 must be available.
#[inline(always)]
unsafe fn interleave128(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
  // SAFETY: AVX2 128-bit lane interleave because:
  // 1. The caller only reaches this helper with AVX2 available.
  // 2. `a` and `b` are local vector values.
  // 3. The permutes operate only within vector registers.
  unsafe {
    (
      _mm256_permute2x128_si256(a, b, 0x20),
      _mm256_permute2x128_si256(a, b, 0x31),
    )
  }
}

/// # Safety
///
/// AVX2 must be available.
#[inline(always)]
pub(super) unsafe fn transpose8x8(vecs: &mut [__m256i; 8]) {
  // SAFETY: AVX2 8x8 register transpose because:
  // 1. The caller only reaches this helper with AVX2 available.
  // 2. `vecs` is a valid fixed-size 8-register array.
  // 3. The unpack and permute operations stay within vector registers.
  unsafe {
    let ab_0145 = _mm256_unpacklo_epi32(vecs[0], vecs[1]);
    let ab_2367 = _mm256_unpackhi_epi32(vecs[0], vecs[1]);
    let cd_0145 = _mm256_unpacklo_epi32(vecs[2], vecs[3]);
    let cd_2367 = _mm256_unpackhi_epi32(vecs[2], vecs[3]);
    let ef_0145 = _mm256_unpacklo_epi32(vecs[4], vecs[5]);
    let ef_2367 = _mm256_unpackhi_epi32(vecs[4], vecs[5]);
    let gh_0145 = _mm256_unpacklo_epi32(vecs[6], vecs[7]);
    let gh_2367 = _mm256_unpackhi_epi32(vecs[6], vecs[7]);

    let abcd_04 = _mm256_unpacklo_epi64(ab_0145, cd_0145);
    let abcd_15 = _mm256_unpackhi_epi64(ab_0145, cd_0145);
    let abcd_26 = _mm256_unpacklo_epi64(ab_2367, cd_2367);
    let abcd_37 = _mm256_unpackhi_epi64(ab_2367, cd_2367);
    let efgh_04 = _mm256_unpacklo_epi64(ef_0145, gh_0145);
    let efgh_15 = _mm256_unpackhi_epi64(ef_0145, gh_0145);
    let efgh_26 = _mm256_unpacklo_epi64(ef_2367, gh_2367);
    let efgh_37 = _mm256_unpackhi_epi64(ef_2367, gh_2367);

    let (abcdefgh_0, abcdefgh_4) = interleave128(abcd_04, efgh_04);
    let (abcdefgh_1, abcdefgh_5) = interleave128(abcd_15, efgh_15);
    let (abcdefgh_2, abcdefgh_6) = interleave128(abcd_26, efgh_26);
    let (abcdefgh_3, abcdefgh_7) = interleave128(abcd_37, efgh_37);

    vecs[0] = abcdefgh_0;
    vecs[1] = abcdefgh_1;
    vecs[2] = abcdefgh_2;
    vecs[3] = abcdefgh_3;
    vecs[4] = abcdefgh_4;
    vecs[5] = abcdefgh_5;
    vecs[6] = abcdefgh_6;
    vecs[7] = abcdefgh_7;
  }
}

/// # Safety
///
/// AVX2 must be available and every input must be readable for a complete
/// block starting at `block_offset`.
#[inline(always)]
unsafe fn transpose_msg_vecs(inputs: &[*const u8; DEGREE], block_offset: usize) -> [__m256i; 16] {
  // SAFETY: AVX2 message transpose because:
  // 1. The caller guarantees each input is readable for one full BLAKE3 block at `block_offset`.
  // 2. The two 32-byte loads cover exactly that 64-byte block.
  // 3. The prefetch address is a hint computed with wrapping pointer arithmetic and is not
  //    dereferenced by Rust.
  // 4. The transpose operates only on local vector registers.
  unsafe {
    let stride = 4usize.strict_mul(DEGREE);
    let mut half0 = [
      loadu(inputs[0].add(block_offset)),
      loadu(inputs[1].add(block_offset)),
      loadu(inputs[2].add(block_offset)),
      loadu(inputs[3].add(block_offset)),
      loadu(inputs[4].add(block_offset)),
      loadu(inputs[5].add(block_offset)),
      loadu(inputs[6].add(block_offset)),
      loadu(inputs[7].add(block_offset)),
    ];
    let mut half1 = [
      loadu(inputs[0].add(block_offset.strict_add(stride))),
      loadu(inputs[1].add(block_offset.strict_add(stride))),
      loadu(inputs[2].add(block_offset.strict_add(stride))),
      loadu(inputs[3].add(block_offset.strict_add(stride))),
      loadu(inputs[4].add(block_offset.strict_add(stride))),
      loadu(inputs[5].add(block_offset.strict_add(stride))),
      loadu(inputs[6].add(block_offset.strict_add(stride))),
      loadu(inputs[7].add(block_offset.strict_add(stride))),
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
/// AVX2 must be available.
#[inline(always)]
unsafe fn load_counters(counter: u64, increment_counter: bool) -> (__m256i, __m256i) {
  let mask = if increment_counter { !0u64 } else { 0u64 };
  // SAFETY: AVX2 counter vector construction because:
  // 1. Counter arithmetic is intentionally wrapping per BLAKE3.
  // 2. Any resulting `u32` bit pattern is valid lane data.
  // 3. The caller only reaches this helper with AVX2 available.
  unsafe {
    (
      set8([
        counter_low(counter),
        counter_low(counter.wrapping_add(mask & 1)),
        counter_low(counter.wrapping_add(mask & 2)),
        counter_low(counter.wrapping_add(mask & 3)),
        counter_low(counter.wrapping_add(mask & 4)),
        counter_low(counter.wrapping_add(mask & 5)),
        counter_low(counter.wrapping_add(mask & 6)),
        counter_low(counter.wrapping_add(mask & 7)),
      ]),
      set8([
        counter_high(counter),
        counter_high(counter.wrapping_add(mask & 1)),
        counter_high(counter.wrapping_add(mask & 2)),
        counter_high(counter.wrapping_add(mask & 3)),
        counter_high(counter.wrapping_add(mask & 4)),
        counter_high(counter.wrapping_add(mask & 5)),
        counter_high(counter.wrapping_add(mask & 6)),
        counter_high(counter.wrapping_add(mask & 7)),
      ]),
    )
  }
}

#[cfg(any(feature = "diag", target_os = "linux", target_os = "macos", target_os = "windows"))]
macro_rules! avx2_shuffle {
  ($z:expr, $y:expr, $x:expr, $w:expr) => {
    ($z << 6) | ($y << 4) | ($x << 2) | $w
  };
}

#[cfg(any(feature = "diag", target_os = "linux", target_os = "macos", target_os = "windows"))]
macro_rules! shuffle2 {
  ($a:expr, $b:expr, $c:expr) => {
    _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps($a), _mm256_castsi256_ps($b), $c))
  };
}

/// # Safety
///
/// AVX2 must be available, and both pointers must be readable for 16 bytes
/// starting at `offset`.
#[cfg(any(feature = "diag", target_os = "linux", target_os = "macos", target_os = "windows"))]
#[inline(always)]
unsafe fn load2x128(lo: *const u8, hi: *const u8, offset: usize) -> __m256i {
  // SAFETY: Loading two 128-bit halves into one YMM register because:
  // 1. The caller guarantees both parent-block pointers are readable for `offset + 16` bytes.
  // 2. `_mm_loadu_si128` accepts arbitrary byte alignment.
  // 3. The caller only reaches this helper with AVX2 available.
  unsafe {
    let lo = _mm_loadu_si128(lo.add(offset).cast());
    let hi = _mm_loadu_si128(hi.add(offset).cast());
    _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1)
  }
}

/// # Safety
///
/// AVX2 must be available.
#[cfg(any(feature = "diag", target_os = "linux", target_os = "macos", target_os = "windows"))]
#[inline(always)]
unsafe fn set2x128(row: __m128i) -> __m256i {
  // SAFETY: Duplicating one 128-bit row into both halves because:
  // 1. `row` is a local vector value.
  // 2. The caller only reaches this helper with AVX2 available.
  unsafe { _mm256_broadcastsi128_si256(row) }
}

/// # Safety
///
/// AVX2 must be available.
#[cfg(any(feature = "diag", target_os = "linux", target_os = "macos", target_os = "windows"))]
#[inline(always)]
unsafe fn g1_2(
  row0: &mut __m256i,
  row1: &mut __m256i,
  row2: &mut __m256i,
  row3: &mut __m256i,
  m: __m256i,
  rot16_mask: __m256i,
) {
  // SAFETY: Two-lane AVX2 G round half because:
  // 1. The caller only reaches this helper with AVX2 available.
  // 2. All inputs are local YMM registers; the operation is lane-local.
  // 3. `rot16_mask` is the fixed byte-shuffle mask for rotate-right-by-16.
  unsafe {
    *row0 = add(add(*row0, m), *row1);
    *row3 = xor(*row3, *row0);
    *row3 = _mm256_shuffle_epi8(*row3, rot16_mask);
    *row2 = add(*row2, *row3);
    *row1 = xor(*row1, *row2);
    *row1 = rot12(*row1);
  }
}

/// # Safety
///
/// AVX2 must be available.
#[cfg(any(feature = "diag", target_os = "linux", target_os = "macos", target_os = "windows"))]
#[inline(always)]
unsafe fn g2_2(
  row0: &mut __m256i,
  row1: &mut __m256i,
  row2: &mut __m256i,
  row3: &mut __m256i,
  m: __m256i,
  rot8_mask: __m256i,
) {
  // SAFETY: Two-lane AVX2 G round half because:
  // 1. The caller only reaches this helper with AVX2 available.
  // 2. All inputs are local YMM registers; the operation is lane-local.
  // 3. `rot8_mask` is the fixed byte-shuffle mask for rotate-right-by-8.
  unsafe {
    *row0 = add(add(*row0, m), *row1);
    *row3 = xor(*row3, *row0);
    *row3 = _mm256_shuffle_epi8(*row3, rot8_mask);
    *row2 = add(*row2, *row3);
    *row1 = xor(*row1, *row2);
    *row1 = rot7(*row1);
  }
}

/// # Safety
///
/// AVX2 must be available.
#[cfg(any(feature = "diag", target_os = "linux", target_os = "macos", target_os = "windows"))]
#[inline(always)]
unsafe fn diagonalize_2(row0: &mut __m256i, row2: &mut __m256i, row3: &mut __m256i) {
  // SAFETY: Two-lane AVX2 diagonalization because:
  // 1. The caller only reaches this helper with AVX2 available.
  // 2. `_mm256_shuffle_epi32` shuffles within each independent 128-bit lane.
  unsafe {
    *row0 = _mm256_shuffle_epi32(*row0, avx2_shuffle!(2, 1, 0, 3));
    *row3 = _mm256_shuffle_epi32(*row3, avx2_shuffle!(1, 0, 3, 2));
    *row2 = _mm256_shuffle_epi32(*row2, avx2_shuffle!(0, 3, 2, 1));
  }
}

/// # Safety
///
/// AVX2 must be available.
#[cfg(any(feature = "diag", target_os = "linux", target_os = "macos", target_os = "windows"))]
#[inline(always)]
unsafe fn undiagonalize_2(row0: &mut __m256i, row2: &mut __m256i, row3: &mut __m256i) {
  // SAFETY: Two-lane AVX2 undiagonalization because:
  // 1. The caller only reaches this helper with AVX2 available.
  // 2. `_mm256_shuffle_epi32` shuffles within each independent 128-bit lane.
  unsafe {
    *row0 = _mm256_shuffle_epi32(*row0, avx2_shuffle!(0, 3, 2, 1));
    *row3 = _mm256_shuffle_epi32(*row3, avx2_shuffle!(1, 0, 3, 2));
    *row2 = _mm256_shuffle_epi32(*row2, avx2_shuffle!(2, 1, 0, 3));
  }
}

/// # Safety
///
/// AVX2 must be available.
#[cfg(any(feature = "diag", target_os = "linux", target_os = "macos", target_os = "windows"))]
#[inline(always)]
unsafe fn compress2_pre(
  [mut row0, mut row1, mut row2, mut row3]: [__m256i; 4],
  [mut m0, mut m1, mut m2, mut m3]: [__m256i; 4],
) -> [__m256i; 4] {
  // SAFETY: Two-lane AVX2 compression preimage because:
  // 1. The caller only reaches this helper with AVX2 available.
  // 2. `row0..row3` are the two independent compression states.
  // 3. `m0..m3` are the two 64-byte message blocks.
  // 4. Every shuffle is lane-local, so the two compressions do not mix.
  unsafe {
    let rot16_mask = _mm256_setr_epi8(
      2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
    );
    let rot8_mask = _mm256_setr_epi8(
      1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12,
    );

    let mut t0;
    let mut t1;
    let mut t2;
    let mut t3;
    let mut tt;

    t0 = shuffle2!(m0, m1, avx2_shuffle!(2, 0, 2, 0));
    g1_2(&mut row0, &mut row1, &mut row2, &mut row3, t0, rot16_mask);
    t1 = shuffle2!(m0, m1, avx2_shuffle!(3, 1, 3, 1));
    g2_2(&mut row0, &mut row1, &mut row2, &mut row3, t1, rot8_mask);
    diagonalize_2(&mut row0, &mut row2, &mut row3);
    t2 = shuffle2!(m2, m3, avx2_shuffle!(2, 0, 2, 0));
    t2 = _mm256_shuffle_epi32(t2, avx2_shuffle!(2, 1, 0, 3));
    g1_2(&mut row0, &mut row1, &mut row2, &mut row3, t2, rot16_mask);
    t3 = shuffle2!(m2, m3, avx2_shuffle!(3, 1, 3, 1));
    t3 = _mm256_shuffle_epi32(t3, avx2_shuffle!(2, 1, 0, 3));
    g2_2(&mut row0, &mut row1, &mut row2, &mut row3, t3, rot8_mask);
    undiagonalize_2(&mut row0, &mut row2, &mut row3);
    m0 = t0;
    m1 = t1;
    m2 = t2;
    m3 = t3;

    macro_rules! next_round_update {
      () => {{
        t0 = shuffle2!(m0, m1, avx2_shuffle!(3, 1, 1, 2));
        t0 = _mm256_shuffle_epi32(t0, avx2_shuffle!(0, 3, 2, 1));
        g1_2(&mut row0, &mut row1, &mut row2, &mut row3, t0, rot16_mask);
        t1 = shuffle2!(m2, m3, avx2_shuffle!(3, 3, 2, 2));
        tt = _mm256_shuffle_epi32(m0, avx2_shuffle!(0, 0, 3, 3));
        t1 = _mm256_blend_epi16(tt, t1, 0xCC);
        g2_2(&mut row0, &mut row1, &mut row2, &mut row3, t1, rot8_mask);
        diagonalize_2(&mut row0, &mut row2, &mut row3);
        t2 = _mm256_unpacklo_epi64(m3, m1);
        tt = _mm256_blend_epi16(t2, m2, 0xC0);
        t2 = _mm256_shuffle_epi32(tt, avx2_shuffle!(1, 3, 2, 0));
        g1_2(&mut row0, &mut row1, &mut row2, &mut row3, t2, rot16_mask);
        t3 = _mm256_unpackhi_epi32(m1, m3);
        tt = _mm256_unpacklo_epi32(m2, t3);
        t3 = _mm256_shuffle_epi32(tt, avx2_shuffle!(0, 1, 3, 2));
        g2_2(&mut row0, &mut row1, &mut row2, &mut row3, t3, rot8_mask);
        undiagonalize_2(&mut row0, &mut row2, &mut row3);
        m0 = t0;
        m1 = t1;
        m2 = t2;
        m3 = t3;
      }};
    }

    macro_rules! next_round_final {
      () => {{
        t0 = shuffle2!(m0, m1, avx2_shuffle!(3, 1, 1, 2));
        t0 = _mm256_shuffle_epi32(t0, avx2_shuffle!(0, 3, 2, 1));
        g1_2(&mut row0, &mut row1, &mut row2, &mut row3, t0, rot16_mask);
        t1 = shuffle2!(m2, m3, avx2_shuffle!(3, 3, 2, 2));
        tt = _mm256_shuffle_epi32(m0, avx2_shuffle!(0, 0, 3, 3));
        t1 = _mm256_blend_epi16(tt, t1, 0xCC);
        g2_2(&mut row0, &mut row1, &mut row2, &mut row3, t1, rot8_mask);
        diagonalize_2(&mut row0, &mut row2, &mut row3);
        t2 = _mm256_unpacklo_epi64(m3, m1);
        tt = _mm256_blend_epi16(t2, m2, 0xC0);
        t2 = _mm256_shuffle_epi32(tt, avx2_shuffle!(1, 3, 2, 0));
        g1_2(&mut row0, &mut row1, &mut row2, &mut row3, t2, rot16_mask);
        t3 = _mm256_unpackhi_epi32(m1, m3);
        tt = _mm256_unpacklo_epi32(m2, t3);
        t3 = _mm256_shuffle_epi32(tt, avx2_shuffle!(0, 1, 3, 2));
        g2_2(&mut row0, &mut row1, &mut row2, &mut row3, t3, rot8_mask);
        undiagonalize_2(&mut row0, &mut row2, &mut row3);
      }};
    }

    next_round_update!();
    next_round_update!();
    next_round_update!();
    next_round_update!();
    next_round_update!();
    next_round_final!();

    [row0, row1, row2, row3]
  }
}

/// # Safety
///
/// AVX2 must be available.
#[cfg(any(feature = "diag", target_os = "linux", target_os = "macos", target_os = "windows"))]
#[inline(always)]
unsafe fn iv_row2x128() -> __m256i {
  // SAFETY: Duplicating the fixed BLAKE3 IV row because:
  // 1. The caller only reaches this helper with AVX2 available.
  // 2. Any `u32` bit pattern is valid vector lane data after reinterpretation.
  unsafe {
    set2x128(_mm_setr_epi32(
      IV[0].cast_signed(),
      IV[1].cast_signed(),
      IV[2].cast_signed(),
      IV[3].cast_signed(),
    ))
  }
}

/// # Safety
///
/// AVX2 must be available and `out` must be writable for two chaining values.
#[cfg(any(feature = "diag", target_os = "linux", target_os = "macos", target_os = "windows"))]
#[inline(always)]
unsafe fn store2_cvs(row0: __m256i, row1: __m256i, out: *mut u8) {
  // SAFETY: Storing two 32-byte CVs from two independent 128-bit lanes because:
  // 1. The caller guarantees `out` is writable for two OUT_LEN-byte CV outputs.
  // 2. The low 128-bit halves hold the first CV words; the high halves hold the second CV words.
  // 3. `_mm_storeu_si128` accepts arbitrary byte alignment.
  unsafe {
    _mm_storeu_si128(out.cast(), _mm256_castsi256_si128(row0));
    _mm_storeu_si128(out.add(16).cast(), _mm256_castsi256_si128(row1));
    _mm_storeu_si128(out.add(OUT_LEN).cast(), _mm256_extracti128_si256(row0, 1));
    _mm_storeu_si128(
      out.add(OUT_LEN.strict_add(16)).cast(),
      _mm256_extracti128_si256(row1, 1),
    );
  }
}

/// # Safety
///
/// AVX2 must be available.
#[cfg(any(feature = "diag", target_os = "linux", target_os = "macos", target_os = "windows"))]
#[inline(always)]
unsafe fn compress2_parent_pre(
  key: &[u32; 8],
  m0: __m256i,
  m1: __m256i,
  m2: __m256i,
  m3: __m256i,
  flags: u32,
) -> [__m256i; 4] {
  // SAFETY: Parent-specific two-lane setup because:
  // 1. The caller only reaches this helper with AVX2 available.
  // 2. `key` is a fixed 8-word array.
  // 3. `m0..m3` are the two 64-byte parent blocks.
  unsafe {
    compress2_pre(
      [
        set2x128(_mm_loadu_si128(key.as_ptr().cast())),
        set2x128(_mm_loadu_si128(key.as_ptr().add(4).cast())),
        iv_row2x128(),
        set2x128(_mm_setr_epi32(
          0,
          0,
          BLOCK_LEN_U32.cast_signed(),
          (PARENT | flags).cast_signed(),
        )),
      ],
      [m0, m1, m2, m3],
    )
  }
}

/// Owned AVX2 reducer for exactly two parent blocks.
///
/// # Safety
///
/// The caller must ensure:
/// 1. AVX2 is available on the current CPU.
/// 2. `parents[0]` and `parents[1]` are each readable for one 64-byte parent block.
/// 3. `out` is writable for two 32-byte CV outputs.
#[cfg(any(feature = "diag", target_os = "linux", target_os = "macos", target_os = "windows"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn parent_cv2_owned(parents: &[*const u8; 2], key: &[u32; 8], flags: u32, out: *mut u8) {
  // SAFETY: Two-parent AVX2 CV reduction because:
  // 1. This function requires AVX2 at the target-feature boundary.
  // 2. The caller guarantees both parent blocks are readable for 64 bytes.
  // 3. The caller guarantees `out` is writable for two OUT_LEN-byte CVs.
  unsafe {
    let [mut row0, mut row1, row2, row3] = compress2_parent_pre(
      key,
      load2x128(parents[0], parents[1], 0),
      load2x128(parents[0], parents[1], 16),
      load2x128(parents[0], parents[1], 32),
      load2x128(parents[0], parents[1], 48),
      flags,
    );
    row0 = xor(row0, row2);
    row1 = xor(row1, row3);
    store2_cvs(row0, row1, out);
  }
}

/// Owned AVX2 reducer for exactly two contiguous full chunks.
///
/// # Safety
///
/// The caller must ensure:
/// 1. AVX2 is available on the current CPU.
/// 2. `input` is readable for two full BLAKE3 chunks.
/// 3. `out` is writable for two 32-byte CV outputs.
#[cfg(any(feature = "diag", target_os = "linux", target_os = "macos", target_os = "windows"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn hash2_chunks_owned(input: *const u8, key: &[u32; 8], counter: u64, flags: u32, out: *mut u8) {
  // SAFETY: Two-chunk AVX2 CV reduction because:
  // 1. This function requires AVX2 at the target-feature boundary.
  // 2. The caller guarantees two full chunks are readable from `input`.
  // 3. The caller guarantees `out` is writable for two OUT_LEN-byte CVs.
  // 4. Each 128-bit half tracks one chunk independently.
  unsafe {
    let input1 = input.add(CHUNK_LEN);
    let mut row0 = set2x128(_mm_loadu_si128(key.as_ptr().cast()));
    let mut row1 = set2x128(_mm_loadu_si128(key.as_ptr().add(4).cast()));

    let counter0_low = counter_low(counter).cast_signed();
    let counter0_high = counter_high(counter).cast_signed();
    let counter1 = counter.wrapping_add(1);
    let counter1_low = counter_low(counter1).cast_signed();
    let counter1_high = counter_high(counter1).cast_signed();
    let block_len = BLOCK_LEN_U32.cast_signed();

    for block_idx in 0..(CHUNK_LEN / BLOCK_LEN) {
      let mut block_flags = flags;
      if block_idx == 0 {
        block_flags |= CHUNK_START;
      }
      if block_idx.strict_add(1) == CHUNK_LEN / BLOCK_LEN {
        block_flags |= CHUNK_END;
      }

      let row2 = iv_row2x128();
      let row3 = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_setr_epi32(
          counter0_low,
          counter0_high,
          block_len,
          block_flags.cast_signed(),
        )),
        _mm_setr_epi32(counter1_low, counter1_high, block_len, block_flags.cast_signed()),
        1,
      );
      let offset = block_idx.strict_mul(BLOCK_LEN);
      let [mut v0, mut v1, v2, v3] = compress2_pre(
        [row0, row1, row2, row3],
        [
          load2x128(input, input1, offset),
          load2x128(input, input1, offset.strict_add(16)),
          load2x128(input, input1, offset.strict_add(32)),
          load2x128(input, input1, offset.strict_add(48)),
        ],
      );
      v0 = xor(v0, v2);
      v1 = xor(v1, v3);
      row0 = v0;
      row1 = v1;
    }

    store2_cvs(row0, row1, out);
  }
}

/// Owned Rust-intrinsic implementation of the 8-way hash-many kernel.
///
/// This stays callable on platforms where production dispatch still prefers
/// assembly, so diagnostic benches can measure the owned candidate directly.
///
/// # Safety
///
/// Caller must ensure AVX2 is available, every input pointer is valid for
/// `blocks * BLOCK_LEN` readable bytes, and `out` is valid for
/// `DEGREE * OUT_LEN` writable bytes.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn hash8_owned(
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
) {
  // SAFETY: 8-way AVX2 BLAKE3 hash-many because:
  // 1. The caller guarantees AVX2 availability for this target-feature function.
  // 2. Each input pointer is readable for `blocks * BLOCK_LEN` bytes.
  // 3. `out` is writable for `DEGREE * OUT_LEN` bytes.
  // 4. All pointer offsets are bounded by the fixed BLAKE3 block/chunk sizes.
  unsafe {
    let rot16_mask = _mm256_setr_epi8(
      2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
    );
    let rot8_mask = _mm256_setr_epi8(
      1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12,
    );

    let block_len_vec = set1(BLOCK_LEN_U32);
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

    let (counter_low_vec, counter_high_vec) = load_counters(counter, increment_counter);

    for block in 0..blocks {
      let mut block_flags = flags;
      if block == 0 {
        block_flags |= flags_start;
      }
      if block.strict_add(1) == blocks {
        block_flags |= flags_end;
      }

      let block_flags_vec = set1(block_flags);
      let msg_vecs = transpose_msg_vecs(inputs, block.strict_mul(BLOCK_LEN));

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

      round(&mut v, &msg_vecs, 0, rot16_mask, rot8_mask);
      round(&mut v, &msg_vecs, 1, rot16_mask, rot8_mask);
      round(&mut v, &msg_vecs, 2, rot16_mask, rot8_mask);
      round(&mut v, &msg_vecs, 3, rot16_mask, rot8_mask);
      round(&mut v, &msg_vecs, 4, rot16_mask, rot8_mask);
      round(&mut v, &msg_vecs, 5, rot16_mask, rot8_mask);
      round(&mut v, &msg_vecs, 6, rot16_mask, rot8_mask);

      h_vecs[0] = xor(v[0], v[8]);
      h_vecs[1] = xor(v[1], v[9]);
      h_vecs[2] = xor(v[2], v[10]);
      h_vecs[3] = xor(v[3], v[11]);
      h_vecs[4] = xor(v[4], v[12]);
      h_vecs[5] = xor(v[5], v[13]);
      h_vecs[6] = xor(v[6], v[14]);
      h_vecs[7] = xor(v[7], v[15]);
    }

    // Unlike SSE4.1, this transpose yields output vecs already ordered by word.
    transpose8x8(&mut h_vecs);

    let stride = 4usize.strict_mul(DEGREE);
    storeu(h_vecs[0], out);
    storeu(h_vecs[1], out.add(stride));
    storeu(h_vecs[2], out.add(2usize.strict_mul(stride)));
    storeu(h_vecs[3], out.add(3usize.strict_mul(stride)));
    storeu(h_vecs[4], out.add(4usize.strict_mul(stride)));
    storeu(h_vecs[5], out.add(5usize.strict_mul(stride)));
    storeu(h_vecs[6], out.add(6usize.strict_mul(stride)));
    storeu(h_vecs[7], out.add(7usize.strict_mul(stride)));
  }
}

/// Hash `DEGREE` independent inputs in parallel.
///
/// # Safety
///
/// Caller must ensure AVX2 is available and that all input pointers are valid
/// for `blocks * BLOCK_LEN` bytes.
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn hash8(request: HashManyRequest<'_, DEGREE>) {
  // SAFETY: Forwarding to the owned AVX2 implementation because:
  // 1. This function has the same AVX2 target-feature requirement.
  // 2. The caller's pointer/output contract is identical to `hash8_owned`.
  unsafe { hash8_owned(request) }
}

/// Generate 8 root output blocks (64 bytes each) in parallel.
///
/// Each lane uses an independent `output_block_counter` (`counter + lane`), but
/// shares the same `chaining_value`, `block_words`, `block_len`, and `flags`.
///
/// # Safety
/// Caller must ensure AVX2 is available and that `out` is valid for `8 * 64`
/// writable bytes.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn root_output_blocks8(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
  out: *mut u8,
) {
  // SAFETY: AVX2 8-block root output because:
  // 1. The caller guarantees AVX2 availability for this target-feature function.
  // 2. `out` is writable for 8 full 64-byte root output blocks.
  // 3. `chaining_value` and `block_words` are fixed-size references with all indexed words present.
  // 4. All pointer writes are bounded by `lane < DEGREE` and the 64-byte per-lane stride.
  unsafe {
    let rot16_mask = _mm256_setr_epi8(
      2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
    );
    let rot8_mask = _mm256_setr_epi8(
      1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12,
    );

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

    let msg_vecs = [
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

    let (counter_low_vec, counter_high_vec) = load_counters(counter, true);
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

    round(&mut v, &msg_vecs, 0, rot16_mask, rot8_mask);
    round(&mut v, &msg_vecs, 1, rot16_mask, rot8_mask);
    round(&mut v, &msg_vecs, 2, rot16_mask, rot8_mask);
    round(&mut v, &msg_vecs, 3, rot16_mask, rot8_mask);
    round(&mut v, &msg_vecs, 4, rot16_mask, rot8_mask);
    round(&mut v, &msg_vecs, 5, rot16_mask, rot8_mask);
    round(&mut v, &msg_vecs, 6, rot16_mask, rot8_mask);

    let mut out_lo = [
      xor(v[0], v[8]),
      xor(v[1], v[9]),
      xor(v[2], v[10]),
      xor(v[3], v[11]),
      xor(v[4], v[12]),
      xor(v[5], v[13]),
      xor(v[6], v[14]),
      xor(v[7], v[15]),
    ];
    let mut out_hi = [
      xor(v[8], cv_vecs[0]),
      xor(v[9], cv_vecs[1]),
      xor(v[10], cv_vecs[2]),
      xor(v[11], cv_vecs[3]),
      xor(v[12], cv_vecs[4]),
      xor(v[13], cv_vecs[5]),
      xor(v[14], cv_vecs[6]),
      xor(v[15], cv_vecs[7]),
    ];

    transpose8x8(&mut out_lo);
    transpose8x8(&mut out_hi);

    for lane in 0..DEGREE {
      let base = out.add(lane.strict_mul(64));
      storeu(out_lo[lane], base);
      storeu(out_hi[lane], base.add(32));
    }
  }
}

/// Generate 1 root output block (64 bytes).
/// Delegates to SSE4.1 implementation (AVX2 is overkill for single block).
///
/// # Safety
/// Caller must ensure AVX2 is available and that `out` is valid for `64` writable bytes.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn root_output_blocks1(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
  out: *mut u8,
) {
  // AVX2 implies SSE4.1, so delegate to the SSE4.1 implementation
  // SAFETY: Delegating to the SSE4.1 root-output helper because:
  // 1. The caller guarantees AVX2 availability.
  // 2. AVX2 implies SSE4.1 on x86_64.
  // 3. The caller's output-buffer contract is identical to the delegated helper.
  unsafe { super::sse41::root_output_blocks1(chaining_value, block_words, counter, block_len, flags, out) }
}

/// Generate 2 root output blocks (128 bytes) with consecutive counters.
/// Delegates to SSE4.1 implementation.
///
/// # Safety
/// Caller must ensure AVX2 is available and that `out` is valid for `128` writable bytes.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn root_output_blocks2(
  chaining_value: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
  out: *mut u8,
) {
  // AVX2 implies SSE4.1, so delegate to the SSE4.1 implementation
  // SAFETY: Delegating to the SSE4.1 root-output helper because:
  // 1. The caller guarantees AVX2 availability.
  // 2. AVX2 implies SSE4.1 on x86_64.
  // 3. The caller's output-buffer contract is identical to the delegated helper.
  unsafe { super::sse41::root_output_blocks2(chaining_value, block_words, counter, block_len, flags, out) }
}
