use core::arch::x86_64::{
  __m512i, _mm_loadu_si128, _mm512_add_epi32, _mm512_broadcast_i32x4, _mm512_loadu_si512, _mm512_mask_set1_epi8,
  _mm512_mask_storeu_epi8, _mm512_maskz_loadu_epi8, _mm512_rol_epi32, _mm512_set1_epi32, _mm512_setr_epi32,
  _mm512_setzero_si512, _mm512_shuffle_epi32, _mm512_shuffle_i32x4, _mm512_storeu_si512, _mm512_unpackhi_epi32,
  _mm512_unpackhi_epi64, _mm512_unpacklo_epi32, _mm512_unpacklo_epi64, _mm512_xor_si512,
};

use super::{BLOCK_SIZE, CONSTANTS, KEY_SIZE, NONCE_SIZE, load_u32_le};
use crate::backend::stack::scrub_dead_stack_avx512;

/// Dead stack cleared after [`columns`]. Its linked frame holds spilled
/// keystream on the reviewed targets and must fit within this bound.
const COLUMN_STACK_SCRUB_WORDS: usize = 160;

/// Generate and XOR a ChaCha20 stream with the AVX-512 kernel.
///
/// # Safety
///
/// The caller must ensure that AVX512F, AVX512VL, AVX512BW, and AVX512DQ are available and that `buffer`'s 64-byte
/// block count fits the counter range starting at `initial_counter`.
#[inline]
pub(super) unsafe fn xor_keystream(
  key: &[u8; KEY_SIZE],
  initial_counter: u32,
  nonce: &[u8; NONCE_SIZE],
  buffer: &mut [u8],
) {
  // SAFETY: Production validates the counter range and detects AVX512F+VL+BW+DQ; direct test and diagnostic callers
  // establish the same conditions.
  unsafe { xor_keystream_impl(key, initial_counter, nonce, buffer) }
}

/// Generate and XOR a ChaCha20 stream with the AVX-512 16-way kernel; fewer
/// than nine final blocks take the row kernel.
///
/// # Safety
///
/// The caller must ensure that AVX512F, AVX512VL, AVX512BW, and AVX512DQ are available and that the number of
/// 64-byte blocks in `buffer` does not exhaust the 32-bit block counter starting at `initial_counter`.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
unsafe fn xor_keystream_impl(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
  let source = Source {
    head: buffer.as_ptr(),
    head_len: buffer.len(),
    tail: &[],
  };
  // SAFETY: the caller establishes the features and the counter range; the
  // source and destination are the same exclusively borrowed buffer.
  unsafe {
    xor_from(
      key,
      nonce,
      initial_counter,
      source,
      buffer.as_mut_ptr(),
      buffer.len(),
      None,
    )
  };
}

/// Sixteen keystream blocks, counters `counter..counter + 16`, one per register.
///
/// # Safety
///
/// The caller must ensure that AVX512F is available.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn column16(key: &[u8; KEY_SIZE], counter: u32, nonce: &[u8; NONCE_SIZE]) -> [__m512i; 16] {
  let mut x0 = _mm512_set1_epi32(0x6170_7865u32.cast_signed());
  let mut x1 = _mm512_set1_epi32(0x3320_646eu32.cast_signed());
  let mut x2 = _mm512_set1_epi32(0x7962_2d32u32.cast_signed());
  let mut x3 = _mm512_set1_epi32(0x6b20_6574u32.cast_signed());
  let mut x4 = _mm512_set1_epi32(load_u32_le(&key[0..4]).cast_signed());
  let mut x5 = _mm512_set1_epi32(load_u32_le(&key[4..8]).cast_signed());
  let mut x6 = _mm512_set1_epi32(load_u32_le(&key[8..12]).cast_signed());
  let mut x7 = _mm512_set1_epi32(load_u32_le(&key[12..16]).cast_signed());
  let mut x8 = _mm512_set1_epi32(load_u32_le(&key[16..20]).cast_signed());
  let mut x9 = _mm512_set1_epi32(load_u32_le(&key[20..24]).cast_signed());
  let mut x10 = _mm512_set1_epi32(load_u32_le(&key[24..28]).cast_signed());
  let mut x11 = _mm512_set1_epi32(load_u32_le(&key[28..32]).cast_signed());
  let mut x12 = _mm512_setr_epi32(
    counter.cast_signed(),
    counter.wrapping_add(1).cast_signed(),
    counter.wrapping_add(2).cast_signed(),
    counter.wrapping_add(3).cast_signed(),
    counter.wrapping_add(4).cast_signed(),
    counter.wrapping_add(5).cast_signed(),
    counter.wrapping_add(6).cast_signed(),
    counter.wrapping_add(7).cast_signed(),
    counter.wrapping_add(8).cast_signed(),
    counter.wrapping_add(9).cast_signed(),
    counter.wrapping_add(10).cast_signed(),
    counter.wrapping_add(11).cast_signed(),
    counter.wrapping_add(12).cast_signed(),
    counter.wrapping_add(13).cast_signed(),
    counter.wrapping_add(14).cast_signed(),
    counter.wrapping_add(15).cast_signed(),
  );
  let mut x13 = _mm512_set1_epi32(load_u32_le(&nonce[0..4]).cast_signed());
  let mut x14 = _mm512_set1_epi32(load_u32_le(&nonce[4..8]).cast_signed());
  let mut x15 = _mm512_set1_epi32(load_u32_le(&nonce[8..12]).cast_signed());

  let o0 = x0;
  let o1 = x1;
  let o2 = x2;
  let o3 = x3;
  let o4 = x4;
  let o5 = x5;
  let o6 = x6;
  let o7 = x7;
  let o8 = x8;
  let o9 = x9;
  let o10 = x10;
  let o11 = x11;
  let o12 = x12;
  let o13 = x13;
  let o14 = x14;
  let o15 = x15;

  let mut round = 0usize;
  while round < 10 {
    quarter_round(&mut x0, &mut x4, &mut x8, &mut x12);
    quarter_round(&mut x1, &mut x5, &mut x9, &mut x13);
    quarter_round(&mut x2, &mut x6, &mut x10, &mut x14);
    quarter_round(&mut x3, &mut x7, &mut x11, &mut x15);

    quarter_round(&mut x0, &mut x5, &mut x10, &mut x15);
    quarter_round(&mut x1, &mut x6, &mut x11, &mut x12);
    quarter_round(&mut x2, &mut x7, &mut x8, &mut x13);
    quarter_round(&mut x3, &mut x4, &mut x9, &mut x14);

    round = round.strict_add(1);
  }

  x0 = _mm512_add_epi32(x0, o0);
  x1 = _mm512_add_epi32(x1, o1);
  x2 = _mm512_add_epi32(x2, o2);
  x3 = _mm512_add_epi32(x3, o3);
  x4 = _mm512_add_epi32(x4, o4);
  x5 = _mm512_add_epi32(x5, o5);
  x6 = _mm512_add_epi32(x6, o6);
  x7 = _mm512_add_epi32(x7, o7);
  x8 = _mm512_add_epi32(x8, o8);
  x9 = _mm512_add_epi32(x9, o9);
  x10 = _mm512_add_epi32(x10, o10);
  x11 = _mm512_add_epi32(x11, o11);
  x12 = _mm512_add_epi32(x12, o12);
  x13 = _mm512_add_epi32(x13, o13);
  x14 = _mm512_add_epi32(x14, o14);
  x15 = _mm512_add_epi32(x15, o15);

  // 16×16 u32 matrix transpose: convert from word-major (x_i = word i for all
  // 16 blocks) to block-major (each register = one complete 64-byte block).
  //
  // Stage 1: 32-bit interleave — pairwise unpack adjacent state-word registers.
  let s1_0 = _mm512_unpacklo_epi32(x0, x1);
  let s1_1 = _mm512_unpackhi_epi32(x0, x1);
  let s1_2 = _mm512_unpacklo_epi32(x2, x3);
  let s1_3 = _mm512_unpackhi_epi32(x2, x3);
  let s1_4 = _mm512_unpacklo_epi32(x4, x5);
  let s1_5 = _mm512_unpackhi_epi32(x4, x5);
  let s1_6 = _mm512_unpacklo_epi32(x6, x7);
  let s1_7 = _mm512_unpackhi_epi32(x6, x7);
  let s1_8 = _mm512_unpacklo_epi32(x8, x9);
  let s1_9 = _mm512_unpackhi_epi32(x8, x9);
  let s1_10 = _mm512_unpacklo_epi32(x10, x11);
  let s1_11 = _mm512_unpackhi_epi32(x10, x11);
  let s1_12 = _mm512_unpacklo_epi32(x12, x13);
  let s1_13 = _mm512_unpackhi_epi32(x12, x13);
  let s1_14 = _mm512_unpacklo_epi32(x14, x15);
  let s1_15 = _mm512_unpackhi_epi32(x14, x15);

  // Stage 2: 64-bit interleave — pair up stage-1 results to get 4 consecutive
  // words from one block in each 128-bit lane.
  let s2_0 = _mm512_unpacklo_epi64(s1_0, s1_2);
  let s2_1 = _mm512_unpackhi_epi64(s1_0, s1_2);
  let s2_2 = _mm512_unpacklo_epi64(s1_1, s1_3);
  let s2_3 = _mm512_unpackhi_epi64(s1_1, s1_3);
  let s2_4 = _mm512_unpacklo_epi64(s1_4, s1_6);
  let s2_5 = _mm512_unpackhi_epi64(s1_4, s1_6);
  let s2_6 = _mm512_unpacklo_epi64(s1_5, s1_7);
  let s2_7 = _mm512_unpackhi_epi64(s1_5, s1_7);
  let s2_8 = _mm512_unpacklo_epi64(s1_8, s1_10);
  let s2_9 = _mm512_unpackhi_epi64(s1_8, s1_10);
  let s2_10 = _mm512_unpacklo_epi64(s1_9, s1_11);
  let s2_11 = _mm512_unpackhi_epi64(s1_9, s1_11);
  let s2_12 = _mm512_unpacklo_epi64(s1_12, s1_14);
  let s2_13 = _mm512_unpackhi_epi64(s1_12, s1_14);
  let s2_14 = _mm512_unpacklo_epi64(s1_13, s1_15);
  let s2_15 = _mm512_unpackhi_epi64(s1_13, s1_15);

  // Stage 3: 128-bit lane shuffle — assemble complete 64-byte blocks from
  // four word-groups distributed across s2 registers.
  //
  // After stage 2, s2[m] lane L holds 4 consecutive words of block (4L + offset).
  // We combine word-groups {0-3, 4-7, 8-11, 12-15} for each block using
  // shuffle_i32x4 which selects 128-bit lanes from two source registers.
  //
  // For blocks {0,4,8,12} — from s2[0], s2[4], s2[8], s2[12]:
  let ab_04_lo = _mm512_shuffle_i32x4::<0b_01_00_01_00>(s2_0, s2_4);
  let cd_04_lo = _mm512_shuffle_i32x4::<0b_01_00_01_00>(s2_8, s2_12);
  let ab_04_hi = _mm512_shuffle_i32x4::<0b_11_10_11_10>(s2_0, s2_4);
  let cd_04_hi = _mm512_shuffle_i32x4::<0b_11_10_11_10>(s2_8, s2_12);
  let blk0 = _mm512_shuffle_i32x4::<0b_10_00_10_00>(ab_04_lo, cd_04_lo);
  let blk4 = _mm512_shuffle_i32x4::<0b_11_01_11_01>(ab_04_lo, cd_04_lo);
  let blk8 = _mm512_shuffle_i32x4::<0b_10_00_10_00>(ab_04_hi, cd_04_hi);
  let blk12 = _mm512_shuffle_i32x4::<0b_11_01_11_01>(ab_04_hi, cd_04_hi);

  // For blocks {1,5,9,13} — from s2[1], s2[5], s2[9], s2[13]:
  let ab_15_lo = _mm512_shuffle_i32x4::<0b_01_00_01_00>(s2_1, s2_5);
  let cd_15_lo = _mm512_shuffle_i32x4::<0b_01_00_01_00>(s2_9, s2_13);
  let ab_15_hi = _mm512_shuffle_i32x4::<0b_11_10_11_10>(s2_1, s2_5);
  let cd_15_hi = _mm512_shuffle_i32x4::<0b_11_10_11_10>(s2_9, s2_13);
  let blk1 = _mm512_shuffle_i32x4::<0b_10_00_10_00>(ab_15_lo, cd_15_lo);
  let blk5 = _mm512_shuffle_i32x4::<0b_11_01_11_01>(ab_15_lo, cd_15_lo);
  let blk9 = _mm512_shuffle_i32x4::<0b_10_00_10_00>(ab_15_hi, cd_15_hi);
  let blk13 = _mm512_shuffle_i32x4::<0b_11_01_11_01>(ab_15_hi, cd_15_hi);

  // For blocks {2,6,10,14} — from s2[2], s2[6], s2[10], s2[14]:
  let ab_26_lo = _mm512_shuffle_i32x4::<0b_01_00_01_00>(s2_2, s2_6);
  let cd_26_lo = _mm512_shuffle_i32x4::<0b_01_00_01_00>(s2_10, s2_14);
  let ab_26_hi = _mm512_shuffle_i32x4::<0b_11_10_11_10>(s2_2, s2_6);
  let cd_26_hi = _mm512_shuffle_i32x4::<0b_11_10_11_10>(s2_10, s2_14);
  let blk2 = _mm512_shuffle_i32x4::<0b_10_00_10_00>(ab_26_lo, cd_26_lo);
  let blk6 = _mm512_shuffle_i32x4::<0b_11_01_11_01>(ab_26_lo, cd_26_lo);
  let blk10 = _mm512_shuffle_i32x4::<0b_10_00_10_00>(ab_26_hi, cd_26_hi);
  let blk14 = _mm512_shuffle_i32x4::<0b_11_01_11_01>(ab_26_hi, cd_26_hi);

  // For blocks {3,7,11,15} — from s2[3], s2[7], s2[11], s2[15]:
  let ab_37_lo = _mm512_shuffle_i32x4::<0b_01_00_01_00>(s2_3, s2_7);
  let cd_37_lo = _mm512_shuffle_i32x4::<0b_01_00_01_00>(s2_11, s2_15);
  let ab_37_hi = _mm512_shuffle_i32x4::<0b_11_10_11_10>(s2_3, s2_7);
  let cd_37_hi = _mm512_shuffle_i32x4::<0b_11_10_11_10>(s2_11, s2_15);
  let blk3 = _mm512_shuffle_i32x4::<0b_10_00_10_00>(ab_37_lo, cd_37_lo);
  let blk7 = _mm512_shuffle_i32x4::<0b_11_01_11_01>(ab_37_lo, cd_37_lo);
  let blk11 = _mm512_shuffle_i32x4::<0b_10_00_10_00>(ab_37_hi, cd_37_hi);
  let blk15 = _mm512_shuffle_i32x4::<0b_11_01_11_01>(ab_37_hi, cd_37_hi);

  [
    blk0, blk1, blk2, blk3, blk4, blk5, blk6, blk7, blk8, blk9, blk10, blk11, blk12, blk13, blk14, blk15,
  ]
}

#[inline(always)]
fn quarter_round(a: &mut __m512i, b: &mut __m512i, c: &mut __m512i, d: &mut __m512i) {
  // SAFETY: this helper is only called from the AVX-512 kernel, so all
  // `_mm512_*` intrinsics are valid for the current compilation target.
  unsafe {
    *a = _mm512_add_epi32(*a, *b);
    *d = _mm512_rol_epi32::<16>(_mm512_xor_si512(*d, *a));
    *c = _mm512_add_epi32(*c, *d);
    *b = _mm512_rol_epi32::<12>(_mm512_xor_si512(*b, *c));
    *a = _mm512_add_epi32(*a, *b);
    *d = _mm512_rol_epi32::<8>(_mm512_xor_si512(*d, *a));
    *c = _mm512_add_epi32(*c, *d);
    *b = _mm512_rol_epi32::<7>(_mm512_xor_si512(*b, *c));
  }
}

/// Message bytes for the masked kernels: `head[..head_len]`, then the bytes of
/// every `tail` span.
#[derive(Clone, Copy)]
pub(crate) struct Source<'a> {
  pub(crate) head: *const u8,
  pub(crate) head_len: usize,
  pub(crate) tail: &'a [&'a [u8]],
}

impl Source<'_> {
  /// Message bytes `offset..offset + 64`, zero past the message.
  ///
  /// # Safety
  ///
  /// The caller must ensure that AVX512F and AVX512BW are available and that `head` is valid for `head_len` reads.
  #[target_feature(enable = "avx512f,avx512bw")]
  #[inline]
  pub(crate) unsafe fn chunk(&self, offset: usize) -> __m512i {
    let mut lanes = if offset.strict_add(64) <= self.head_len {
      // SAFETY: bytes `offset..offset + 64` lie inside `head`.
      unsafe { _mm512_loadu_si512(self.head.add(offset).cast()) }
    } else if offset < self.head_len {
      // SAFETY: the mask selects only bytes `offset..head_len`; masked-off bytes are not accessed.
      unsafe {
        _mm512_maskz_loadu_epi8(
          byte_mask(self.head_len.strict_sub(offset)),
          self.head.add(offset).cast(),
        )
      }
    } else {
      _mm512_setzero_si512()
    };
    if offset.strict_add(64) > self.head_len {
      let mut position = self.head_len;
      for byte in self.tail.iter().flat_map(|span| span.iter()) {
        if let Some(lane) = position.checked_sub(offset).filter(|lane| *lane < 64) {
          lanes = _mm512_mask_set1_epi8(lanes, 1u64 << lane, i8::from_ne_bytes([*byte]));
        }
        position = position.strict_add(1);
      }
    }
    lanes
  }
}

/// The first `len` bytes of a 64-byte register, `len <= 64`.
#[inline(always)]
pub(crate) fn byte_mask(len: usize) -> u64 {
  if len >= 64 {
    u64::MAX
  } else {
    (1u64 << len).wrapping_sub(1)
  }
}

/// `G` groups of four keystream blocks in row form; lane `j` of group `g`
/// takes counter `counters[4·g + j]`. Each group returns block-major: register
/// `j` holds the 64 bytes of lane `j`'s block.
///
/// Four blocks share one dependency chain per group, so one or two groups
/// finish in about the latency of one block.
///
/// # Safety
///
/// The caller must ensure that AVX512F is available and `G <= 4`.
// Always inlined, so the keystream returns in registers rather than through
// memory; every caller establishes AVX512F.
#[inline(always)]
pub(crate) unsafe fn rows<const G: usize>(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; NONCE_SIZE],
  counters: &[u32; 16],
) -> [[__m512i; 4]; G] {
  // SAFETY: this helper inlines into a caller that establishes AVX512F, and
  // the loads read inside `key` and `CONSTANTS`.
  unsafe {
    // `key` is 32 initialized bytes and `CONSTANTS` is 16.
    let (row0, row1, row2) = (
      _mm512_broadcast_i32x4(_mm_loadu_si128(CONSTANTS.as_ptr().cast())),
      _mm512_broadcast_i32x4(_mm_loadu_si128(key.as_ptr().cast())),
      _mm512_broadcast_i32x4(_mm_loadu_si128(key.as_ptr().add(16).cast())),
    );
    let n0 = load_u32_le(&nonce[0..4]).cast_signed();
    let n1 = load_u32_le(&nonce[4..8]).cast_signed();
    let n2 = load_u32_le(&nonce[8..12]).cast_signed();
    let mut row3 = [_mm512_setzero_si512(); G];
    for (group, row) in row3.iter_mut().enumerate() {
      let c = |lane: usize| counters[group.strict_mul(4).strict_add(lane)].cast_signed();
      *row = _mm512_setr_epi32(c(0), n0, n1, n2, c(1), n0, n1, n2, c(2), n0, n1, n2, c(3), n0, n1, n2);
    }
    let initial = row3;
    let mut a = [row0; G];
    let mut b = [row1; G];
    let mut c = [row2; G];
    let mut d = row3;
    for _ in 0..10 {
      for group in 0..G {
        quarter_round(&mut a[group], &mut b[group], &mut c[group], &mut d[group]);
        // Rotate rows so the next quarter round runs on the diagonals.
        b[group] = _mm512_shuffle_epi32::<0x39>(b[group]);
        c[group] = _mm512_shuffle_epi32::<0x4e>(c[group]);
        d[group] = _mm512_shuffle_epi32::<0x93>(d[group]);
      }
      for group in 0..G {
        quarter_round(&mut a[group], &mut b[group], &mut c[group], &mut d[group]);
        b[group] = _mm512_shuffle_epi32::<0x93>(b[group]);
        c[group] = _mm512_shuffle_epi32::<0x4e>(c[group]);
        d[group] = _mm512_shuffle_epi32::<0x39>(d[group]);
      }
    }
    let mut out = [[_mm512_setzero_si512(); 4]; G];
    for (group, blocks) in out.iter_mut().enumerate() {
      let a = _mm512_add_epi32(a[group], row0);
      let b = _mm512_add_epi32(b[group], row1);
      let c = _mm512_add_epi32(c[group], row2);
      let d = _mm512_add_epi32(d[group], initial[group]);
      // Transpose 128-bit lanes: block j is [a_j, b_j, c_j, d_j].
      let ab_lo = _mm512_shuffle_i32x4::<0x44>(a, b);
      let ab_hi = _mm512_shuffle_i32x4::<0xee>(a, b);
      let cd_lo = _mm512_shuffle_i32x4::<0x44>(c, d);
      let cd_hi = _mm512_shuffle_i32x4::<0xee>(c, d);
      *blocks = [
        _mm512_shuffle_i32x4::<0x88>(ab_lo, cd_lo),
        _mm512_shuffle_i32x4::<0xdd>(ab_lo, cd_lo),
        _mm512_shuffle_i32x4::<0x88>(ab_hi, cd_hi),
        _mm512_shuffle_i32x4::<0xdd>(ab_hi, cd_hi),
      ];
    }
    out
  }
}

/// XOR `keystream` into message bytes `offset..` and store them in `dst[..len]`.
///
/// # Safety
///
/// The caller must ensure that AVX512F and AVX512BW are available, `offset < len`, and that `dst` is valid for `len`
/// writes and equal to the source's head or disjoint from the source.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn store(source: &Source<'_>, dst: *mut u8, len: usize, offset: usize, keystream: __m512i) {
  // SAFETY: the caller's contract covers both ranges; the mask selects only
  // bytes `offset..min(offset + 64, len)` of `dst`.
  unsafe {
    let text = _mm512_xor_si512(source.chunk(offset), keystream);
    if offset.strict_add(64) <= len {
      _mm512_storeu_si512(dst.add(offset).cast(), text);
    } else {
      _mm512_mask_storeu_epi8(dst.add(offset).cast(), byte_mask(len.strict_sub(offset)), text);
    }
  }
}

/// The column batches of [`xor_from`]: sixteen blocks at a time from block
/// `first` while more than eight blocks remain, counting block 0 when `extra`
/// is 1. Returns the message blocks written.
///
/// The batch's transposed keystream spills, so this runs out of line and its
/// caller scrubs the released frame.
///
/// # Safety
///
/// The contract of [`xor_from`].
#[inline(never)]
#[expect(
  clippy::needless_range_loop,
  reason = "constant indices keep the keystream in registers; iterators leave it in stack temporaries"
)]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
unsafe fn columns(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; NONCE_SIZE],
  first: u32,
  source: &Source<'_>,
  dst: *mut u8,
  len: usize,
  extra: usize,
) -> usize {
  let blocks = len.div_ceil(BLOCK_SIZE);
  let mut done = 0usize;
  let mut counter = first;
  while blocks.strict_sub(done).strict_add(extra) > 8 {
    // SAFETY: the caller establishes the features and ranges.
    let keystream = unsafe { column16(key, counter, nonce) };
    // Constant indices keep the keystream in registers where the allocator
    // can; an iterator over the array leaves it in a stack temporary.
    for index in 0..16 {
      let offset = done.strict_add(index).strict_mul(BLOCK_SIZE);
      if offset < len {
        // SAFETY: `offset < len` and the caller's contract covers both ranges.
        unsafe { store(source, dst, len, offset, keystream[index]) };
      }
    }
    done = done.strict_add(16).min(blocks);
    counter = counter.wrapping_add(16);
  }
  done
}

/// XOR the keystream from block `first` into the `len`-byte message of
/// `source`, writing `dst`. With `poly_key`, also write the first 32 bytes of
/// block 0, the ChaCha20-Poly1305 one-time key, there in the final pass; the
/// key reaches no other memory.
///
/// Full batches of sixteen blocks take the column kernel. A final group of at
/// most eight blocks, counting block 0 when requested, takes the row kernel;
/// a longer final group takes one masked column batch.
///
/// # Safety
///
/// The caller must ensure that AVX512F, AVX512VL, AVX512BW, and AVX512DQ are available, that the source's head is
/// valid for its reads, that `dst` is valid for `len` writes and either equals the head or is disjoint from the source,
/// that the source holds `len` bytes, and that the message's blocks fit the counter range from `first`.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
pub(crate) unsafe fn xor_from(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; NONCE_SIZE],
  first: u32,
  source: Source<'_>,
  dst: *mut u8,
  len: usize,
  poly_key: Option<&mut [u8; 32]>,
) {
  let blocks = len.div_ceil(BLOCK_SIZE);
  let extra = usize::from(poly_key.is_some());
  let poly_key = poly_key.map_or(core::ptr::null_mut(), |key| key.as_mut_ptr());
  let mut done = 0usize;
  if blocks.strict_add(extra) > 8 {
    // SAFETY: the caller establishes the features and ranges.
    done = unsafe { columns(key, nonce, first, &source, dst, len, extra) };
    // SAFETY: the caller establishes AVX512F.
    unsafe { scrub_dead_stack_avx512::<COLUMN_STACK_SCRUB_WORDS>() };
  }
  // Column batches cover sixteen blocks each, so this is their next counter
  // whenever message blocks remain.
  let counter = first.wrapping_add(u32::try_from(done).unwrap_or(u32::MAX));
  let rest = blocks.strict_sub(done);
  if rest.strict_add(extra) == 0 {
    return;
  }
  // Lane 0 takes block 0 for the key when requested; the message blocks follow.
  let mut counters = [0u32; 16];
  for block in 0..rest {
    counters[block.strict_add(extra)] = counter.wrapping_add(u32::try_from(block).unwrap_or(u32::MAX));
  }
  macro_rules! finish {
    ($groups:literal) => {{
      // SAFETY: the caller establishes the features; `$groups <= 2`.
      let keystream = unsafe { rows::<$groups>(key, nonce, &counters) };
      // Constant indices keep the keystream in registers; an iterator over
      // the nested arrays leaves it in a stack temporary.
      for group in 0usize..$groups {
        for lane in 0..4 {
          let index = group.strict_mul(4).strict_add(lane);
          let block = keystream[group][lane];
          if !poly_key.is_null() && index == 0 {
            // SAFETY: `poly_key` came from a 32-byte exclusive borrow; the mask
            // selects the first 32 bytes.
            unsafe { _mm512_mask_storeu_epi8(poly_key.cast(), byte_mask(32), block) };
          } else if let Some(offset) = index.checked_sub(extra).filter(|block| *block < rest) {
            let offset = done.strict_add(offset).strict_mul(BLOCK_SIZE);
            // SAFETY: `offset < len` and the caller's contract covers both ranges.
            unsafe { store(&source, dst, len, offset, block) };
          }
        }
      }
    }};
  }
  if rest.strict_add(extra) <= 4 {
    finish!(1);
  } else {
    finish!(2);
  }
}
