//! AES-GCM message tails and short messages on x86-64 with VAES and AVX-512.
//!
//! [`seal`] and [`open`] finish a message: they take the GHASH state after the
//! AAD and any preceding batches, then process the remaining bytes in passes of
//! eight blocks (two registers), and the length block and the tag in the
//! final pass. Each pass runs its counter blocks, and the final pass also J0,
//! through one interleaved VAES pass. Input is read and output written with
//! byte-masked loads and stores, and GHASH folds each pass with one reduction.
//!
//! A pass holds at most two keystream registers, so keystream, plaintext, and
//! the encrypted J0 stay in registers; the final open compares the tag in
//! registers before it writes plaintext. Lengths are public and select only
//! masks, register counts, and power windows. The compiler still spills some
//! round keys and H powers, so each worker runs out of line and a fixed
//! dead-stack scrub follows it from the same frame; `just stack-frames` checks
//! that the linked worker fits within it.

use core::arch::x86_64::*;

use super::super::polyval;
use crate::backend::stack::scrub_dead_stack_avx512;

/// Most plaintext bytes that may follow the leading input span of a seal.
pub(crate) const MAX_TAIL: usize = 16;

/// Bytes in one full pass: eight blocks in two registers.
const PASS: usize = 128;

/// Dead stack cleared after each worker. Its linked frame holds spilled round
/// keys and H powers on the reviewed targets and must fit within this bound.
const STACK_SCRUB_WORDS: usize = 40;

/// Byte mask of the 64-byte register at `offset` within a `len`-byte range.
#[inline(always)]
fn chunk_mask(len: usize, offset: usize) -> u64 {
  let left = len.saturating_sub(offset);
  if left >= 64 {
    u64::MAX
  } else {
    (1u64 << left).wrapping_sub(1)
  }
}

/// Message bytes: `src[..src_len]`, then the bytes of every `tail` span.
struct Source<'a> {
  src: *const u8,
  src_len: usize,
  tail: &'a [&'a [u8]],
}

impl Source<'_> {
  /// Message bytes `offset..offset + 64`, zero past the message.
  ///
  /// # Safety
  /// The caller must establish AVX-512F and AVX-512BW, and `src` valid for
  /// `src_len` reads.
  #[inline(always)]
  unsafe fn chunk(&self, offset: usize) -> __m512i {
    // SAFETY: this helper inlines into a caller that establishes its target
    // features. Each load selects only bytes `offset..min(offset + 64, src_len)`
    // of `src`; masked-off bytes are not accessed.
    unsafe {
      let mut lanes = if offset.strict_add(64) <= self.src_len {
        _mm512_loadu_si512(self.src.add(offset).cast())
      } else if offset < self.src_len {
        _mm512_maskz_loadu_epi8(chunk_mask(self.src_len, offset), self.src.add(offset).cast())
      } else {
        _mm512_setzero_si512()
      };
      if offset.strict_add(64) > self.src_len {
        let mut position = self.src_len;
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
}

/// Counter blocks `first..first + 4G` over the nonce in `template`.
///
/// # Safety
/// The caller must establish AVX-512F and AVX-512BW.
#[inline(always)]
unsafe fn counters<const G: usize>(template: __m128i, first: u32) -> [__m512i; G] {
  // SAFETY: this helper inlines into a caller that establishes its target features.
  unsafe {
    let template = _mm512_broadcast_i32x4(template);
    // Byte-reverse each 32-bit word: GCM's counter is the big-endian final word.
    let bswap = _mm512_broadcast_i32x4(_mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3));
    let lanes = _mm512_set_epi32(3, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
    let mut blocks = [_mm512_setzero_si512(); G];
    for (group, block) in (0u32..).zip(blocks.iter_mut()) {
      // Lane `j` of register `group` takes counter `first + 4·group + j`. The
      // caller's length bound keeps every message counter within 32 bits.
      let words = _mm512_add_epi32(
        _mm512_set1_epi32(first.wrapping_add(group.strict_mul(4)).cast_signed()),
        lanes,
      );
      *block = _mm512_mask_mov_epi32(template, 0x8888, _mm512_shuffle_epi8(words, bswap));
    }
    blocks
  }
}

/// Encrypt the counter registers and, with `TAG`, the J0 block interleaved with them.
///
/// # Safety
/// The caller must establish AES, SSE2, AVX-512F, and VAES.
#[inline(always)]
unsafe fn encrypt<const N: usize, const G: usize, const TAG: bool>(
  rk: &[__m128i; N],
  mut state: [__m512i; G],
  mut j0: __m128i,
) -> ([__m512i; G], __m128i) {
  // SAFETY: this helper inlines into a caller that establishes its target
  // features; every round key is an initialized element of `rk`.
  unsafe {
    // An opaque schedule pointer keeps each pass loading its round keys, as the
    // assembly kernels do, instead of hoisting fifteen broadcast keys out of
    // the pass loop and spilling some to the stack.
    let rk = core::hint::black_box(rk);
    let key = |index: usize| rk[index];
    let last = N.strict_sub(1);
    let k0 = key(0);
    let round = _mm512_broadcast_i32x4(k0);
    for lanes in &mut state {
      *lanes = _mm512_xor_si512(*lanes, round);
    }
    if TAG {
      j0 = _mm_xor_si128(j0, k0);
    }
    for index in 1..last {
      let k = key(index);
      let round = _mm512_broadcast_i32x4(k);
      for lanes in &mut state {
        *lanes = _mm512_aesenc_epi128(*lanes, round);
      }
      if TAG {
        j0 = _mm_aesenc_si128(j0, k);
      }
    }
    let k = key(last);
    let round = _mm512_broadcast_i32x4(k);
    for lanes in &mut state {
      *lanes = _mm512_aesenclast_epi128(*lanes, round);
    }
    if TAG {
      j0 = _mm_aesenclast_si128(j0, k);
    }
    (state, j0)
  }
}

/// GHASH after `acc` over the first `count <= 16` big-endian lanes of `lanes`.
///
/// # Safety
/// The caller must establish AVX-512F/VL/BW/DQ, VPCLMULQDQ, PCLMULQDQ, and SSE2,
/// and every lane at or after `count` must be zero.
#[inline(always)]
unsafe fn aggregate(acc: u128, lanes: &[__m512i; 4], count: usize, h_powers_rev_16: &[u128; 16]) -> u128 {
  // SAFETY: the caller establishes the features and the zero lanes.
  unsafe { polyval::x86_aggregate_prefix_be_lanes_inline(acc, h_powers_rev_16, lanes, count) }
}

/// GHASH after `acc` over the `blocks` ciphertext blocks of `ciphertext`, which
/// are zero after the final message byte, then the length block.
///
/// # Safety
/// The features of [`aggregate`], with `blocks <= 4 * G <= 8`.
#[inline(always)]
unsafe fn ghash_final<const G: usize>(
  acc: u128,
  ciphertext: &[__m512i; G],
  blocks: usize,
  length_block: &[u8; 16],
  h_powers_rev_16: &[u128; 16],
) -> u128 {
  // SAFETY: this helper inlines into a caller that establishes its target
  // features; `length_block` is sixteen initialized bytes.
  unsafe {
    let zero = _mm512_setzero_si512();
    let mut lanes = [zero; 4];
    for (lane, register) in lanes.iter_mut().zip(ciphertext) {
      *lane = *register;
    }
    let length = _mm_loadu_si128(length_block.as_ptr().cast());
    // The length block follows the final ciphertext block, at lane `blocks`.
    let (group, lane) = (blocks.strict_div(4), blocks.strict_rem(4));
    let dwords = 0xfu16 << lane.strict_mul(4);
    for (index, register) in lanes.iter_mut().enumerate() {
      let mask = if index == group { dwords } else { 0 };
      *register = _mm512_mask_broadcast_i32x4(*register, mask, length);
    }
    aggregate(acc, &lanes, blocks.strict_add(1), h_powers_rev_16)
  }
}

/// `E(J0) ⊕ GHASH` in memory byte order.
///
/// # Safety
/// The caller must establish SSE2 and SSSE3.
#[inline(always)]
unsafe fn tag_vector(acc: u128, tag_mask: __m128i) -> __m128i {
  // SAFETY: this helper inlines into a caller that establishes its target
  // features; `acc` is sixteen initialized bytes. GHASH holds the accumulator
  // as the big-endian integer of its block bytes.
  unsafe {
    let reverse = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let acc = _mm_shuffle_epi8(_mm_loadu_si128(core::ptr::from_ref(&acc).cast()), reverse);
    _mm_xor_si128(acc, tag_mask)
  }
}

/// Seal one full pass at byte `start` and fold its ciphertext into `acc`.
///
/// # Safety
/// The contract of [`seal`], with `start + PASS < len`.
#[expect(
  clippy::too_many_arguments,
  reason = "a kernel boundary: key, nonce, counter, GHASH state, ranges and tables"
)]
#[expect(
  clippy::needless_range_loop,
  reason = "constant indices keep the lanes in registers; iterator zips spill them"
)]
#[inline(always)]
unsafe fn seal_pass<const N: usize>(
  rk: &[__m128i; N],
  template: __m128i,
  counter: u32,
  acc: u128,
  source: &Source<'_>,
  dst: *mut u8,
  start: usize,
  h_powers_rev_16: &[u128; 16],
) -> u128 {
  // SAFETY: this helper inlines into a caller that establishes its target
  // features; every byte of the pass lies inside the message and `dst`.
  unsafe {
    let (mut lanes, _) = encrypt::<N, 2, false>(rk, counters::<2>(template, counter), template);
    for group in 0usize..2 {
      let offset = start.strict_add(group.strict_mul(64));
      lanes[group] = _mm512_xor_si512(source.chunk(offset), lanes[group]);
      _mm512_storeu_si512(dst.add(offset).cast(), lanes[group]);
    }
    let zero = _mm512_setzero_si512();
    aggregate(acc, &[lanes[0], lanes[1], zero, zero], 8, h_powers_rev_16)
  }
}

/// The final seal pass over bytes `start..len`, `G` registers, and the tag.
///
/// # Safety
/// The contract of [`seal`], with `(len - start).div_ceil(64) == G <= 2`.
#[expect(
  clippy::too_many_arguments,
  reason = "a kernel boundary: key, nonce, counter, GHASH state, ranges and tables"
)]
#[expect(
  clippy::needless_range_loop,
  reason = "constant indices keep the lanes in registers; iterator zips spill them"
)]
#[inline(always)]
unsafe fn seal_final<const N: usize, const G: usize>(
  rk: &[__m128i; N],
  template: __m128i,
  tag_mask: __m128i,
  counter: u32,
  acc: u128,
  source: &Source<'_>,
  dst: *mut u8,
  start: usize,
  len: usize,
  length_block: &[u8; 16],
  h_powers_rev_16: &[u128; 16],
) -> [u8; 16] {
  // SAFETY: this helper inlines into a caller that establishes its target
  // features; each mask selects only bytes inside the message and `dst`.
  unsafe {
    let (mut lanes, _) = encrypt::<N, G, false>(rk, counters::<G>(template, counter), template);
    for group in 0..G {
      let offset = start.strict_add(group.strict_mul(64));
      let mask = chunk_mask(len, offset);
      lanes[group] = _mm512_maskz_mov_epi8(mask, _mm512_xor_si512(source.chunk(offset), lanes[group]));
      _mm512_mask_storeu_epi8(dst.add(offset).cast(), mask, lanes[group]);
    }
    let acc = ghash_final::<G>(
      acc,
      &lanes,
      len.strict_sub(start).div_ceil(16),
      length_block,
      h_powers_rev_16,
    );
    let mut tag = [0u8; 16];
    _mm_storeu_si128(tag.as_mut_ptr().cast(), tag_vector(acc, tag_mask));
    tag
  }
}

/// Open one full pass at byte `start` in place and fold its ciphertext into
/// `acc`. The plaintext is written before the final tag check, as in the batch
/// kernels; [`open`] clears every byte on a mismatch.
///
/// # Safety
/// The contract of [`open`], with `start + PASS < len`.
#[expect(
  clippy::needless_range_loop,
  reason = "constant indices keep the lanes in registers; iterator zips spill them"
)]
#[inline(always)]
unsafe fn open_pass<const N: usize>(
  rk: &[__m128i; N],
  template: __m128i,
  counter: u32,
  acc: u128,
  data: *mut u8,
  start: usize,
  h_powers_rev_16: &[u128; 16],
) -> u128 {
  // SAFETY: this helper inlines into a caller that establishes its target
  // features; every byte of the pass lies inside `data`.
  unsafe {
    let mut ciphertext = [_mm512_setzero_si512(); 2];
    for group in 0usize..2 {
      ciphertext[group] = _mm512_loadu_si512(data.add(start.strict_add(group.strict_mul(64))).cast());
    }
    let zero = _mm512_setzero_si512();
    let acc = aggregate(acc, &[ciphertext[0], ciphertext[1], zero, zero], 8, h_powers_rev_16);
    let (keystream, _) = encrypt::<N, 2, false>(rk, counters::<2>(template, counter), template);
    for group in 0usize..2 {
      _mm512_storeu_si512(
        data.add(start.strict_add(group.strict_mul(64))).cast(),
        _mm512_xor_si512(ciphertext[group], keystream[group]),
      );
    }
    acc
  }
}

/// The final open pass over bytes `start..len`: the tag is checked in
/// registers before the plaintext is written.
///
/// # Safety
/// The contract of [`open`], with `(len - start).div_ceil(64) == G <= 2`.
#[expect(
  clippy::too_many_arguments,
  reason = "a kernel boundary: key, nonce, counter, GHASH state, ranges and tables"
)]
#[expect(
  clippy::needless_range_loop,
  reason = "constant indices keep the lanes in registers; iterator zips spill them"
)]
#[inline(always)]
unsafe fn open_final<const N: usize, const G: usize>(
  rk: &[__m128i; N],
  template: __m128i,
  tag_mask: __m128i,
  counter: u32,
  acc: u128,
  data: *mut u8,
  start: usize,
  len: usize,
  length_block: &[u8; 16],
  h_powers_rev_16: &[u128; 16],
  expected: &[u8; 16],
) -> bool {
  // SAFETY: this helper inlines into a caller that establishes its target
  // features; each mask selects only bytes inside `data[..len]`.
  unsafe {
    let mut ciphertext = [_mm512_setzero_si512(); G];
    for group in 0..G {
      let offset = start.strict_add(group.strict_mul(64));
      ciphertext[group] = _mm512_maskz_loadu_epi8(chunk_mask(len, offset), data.add(offset).cast());
    }
    let acc = ghash_final::<G>(
      acc,
      &ciphertext,
      len.strict_sub(start).div_ceil(16),
      length_block,
      h_powers_rev_16,
    );
    let (keystream, _) = encrypt::<N, G, false>(rk, counters::<G>(template, counter), template);
    // One vector test on the complete difference; the result is public.
    let difference = _mm_xor_si128(tag_vector(acc, tag_mask), _mm_loadu_si128(expected.as_ptr().cast()));
    let matched = _mm_testz_si128(difference, difference) != 0;
    // All-ones on a match, zero otherwise: the stores below do not branch.
    let keep = _mm512_set1_epi64(i64::from(matched).wrapping_neg());
    for group in 0..G {
      let offset = start.strict_add(group.strict_mul(64));
      let text = _mm512_and_si512(_mm512_xor_si512(ciphertext[group], keystream[group]), keep);
      _mm512_mask_storeu_epi8(data.add(offset).cast(), chunk_mask(len, offset), text);
    }
    matched
  }
}

/// Seal the final bytes of a message, `src[..src_len]` followed by the bytes of
/// `tail`, into `dst`, and return the tag.
///
/// `first` is the counter of the first of these bytes' blocks (2 for a whole
/// message), and `acc` is GHASH over the AAD and every preceding block.
///
/// # Safety
/// The caller must establish AES, SSE2, SSSE3, AVX-512F/VL/BW/DQ, VAES,
/// VPCLMULQDQ, and PCLMULQDQ. `src` is valid for `src_len` reads; `dst` is
/// valid for writes of the whole range and either equals `src` or does not
/// overlap it or any tail span. The tail holds at most [`MAX_TAIL`] bytes, the
/// counters of the range's blocks fit in 32 bits, and `length_block` encodes
/// the AAD and whole-message bit lengths.
#[expect(
  clippy::too_many_arguments,
  reason = "a kernel boundary: key, nonce, counter, GHASH state, ranges and tables"
)]
// Out of line: the assembly batch calls before it must not carry hoisted
// round keys or keystream across their clobbers, and the scrub after it
// overwrites the frame the worker released.
#[inline(never)]
#[target_feature(enable = "aes,sse2,ssse3,avx512f,avx512vl,avx512bw,avx512dq,vaes,vpclmulqdq,pclmulqdq")]
unsafe fn seal_worker<const N: usize>(
  rk: &[__m128i; N],
  j0: &[u8; 16],
  first: u32,
  mut acc: u128,
  src: *const u8,
  src_len: usize,
  tail: &[&[u8]],
  dst: *mut u8,
  length_block: &[u8; 16],
  h_powers_rev_16: &[u128; 16],
) -> [u8; 16] {
  let source = Source { src, src_len, tail };
  let len = tail.iter().fold(src_len, |len, span| len.strict_add(span.len()));
  // SAFETY: `j0` is sixteen initialized bytes.
  let template = unsafe { _mm_loadu_si128(j0.as_ptr().cast()) };
  // E(J0) first, in one 128-bit chain, so no later pass carries 128-bit round keys.
  // SAFETY: the caller establishes AES and the AVX-512 features.
  let (_, tag_mask) = unsafe { encrypt::<N, 0, true>(rk, [], template) };
  let (mut start, mut counter) = (0usize, first);
  // Full passes leave a final pass of at most two registers.
  while len.strict_sub(start) > PASS {
    // SAFETY: the caller's contract covers the pass.
    acc = unsafe { seal_pass::<N>(rk, template, counter, acc, &source, dst, start, h_powers_rev_16) };
    start = start.strict_add(PASS);
    counter = counter.strict_add(8);
  }
  macro_rules! last {
    ($g:literal) => {
      // SAFETY: the caller's contract covers the final pass of `$g` registers.
      unsafe {
        seal_final::<N, $g>(
          rk,
          template,
          tag_mask,
          counter,
          acc,
          &source,
          dst,
          start,
          len,
          length_block,
          h_powers_rev_16,
        )
      }
    };
  }
  match len.strict_sub(start).div_ceil(64) {
    0 => last!(0),
    1 => last!(1),
    _ => last!(2),
  }
}

/// Open the final `len` bytes of a message in place if the tag equals `expected`.
///
/// Returns whether the tag matched. On a match `data` holds the plaintext;
/// otherwise every byte of `data` is zero. The final pass writes its plaintext
/// only after the tag comparison. `first` and `acc` are as for [`seal`].
///
/// # Safety
/// The caller must establish the features of [`seal`]. `data` is valid for
/// `len` reads and writes, the counters of its blocks fit in 32 bits, and
/// `length_block` encodes the AAD and whole-message bit lengths.
#[expect(
  clippy::too_many_arguments,
  reason = "a kernel boundary: key, nonce, counter, GHASH state, ranges and tables"
)]
// Out of line: the assembly batch calls before it must not carry hoisted
// round keys or keystream across their clobbers, and the scrub after it
// overwrites the frame the worker released.
#[inline(never)]
#[target_feature(enable = "aes,sse2,ssse3,avx512f,avx512vl,avx512bw,avx512dq,vaes,vpclmulqdq,pclmulqdq")]
unsafe fn open_worker<const N: usize>(
  rk: &[__m128i; N],
  j0: &[u8; 16],
  first: u32,
  mut acc: u128,
  data: *mut u8,
  len: usize,
  length_block: &[u8; 16],
  h_powers_rev_16: &[u128; 16],
  expected: &[u8; 16],
) -> bool {
  // SAFETY: `j0` is sixteen initialized bytes.
  let template = unsafe { _mm_loadu_si128(j0.as_ptr().cast()) };
  // SAFETY: the caller establishes AES and the AVX-512 features.
  let (_, tag_mask) = unsafe { encrypt::<N, 0, true>(rk, [], template) };
  let (mut start, mut counter) = (0usize, first);
  while len.strict_sub(start) > PASS {
    // SAFETY: the caller's contract covers the pass.
    acc = unsafe { open_pass::<N>(rk, template, counter, acc, data, start, h_powers_rev_16) };
    start = start.strict_add(PASS);
    counter = counter.strict_add(8);
  }
  macro_rules! last {
    ($g:literal) => {
      // SAFETY: the caller's contract covers the final pass of `$g` registers.
      unsafe {
        open_final::<N, $g>(
          rk,
          template,
          tag_mask,
          counter,
          acc,
          data,
          start,
          len,
          length_block,
          h_powers_rev_16,
          expected,
        )
      }
    };
  }
  let matched = match len.strict_sub(start).div_ceil(64) {
    0 => last!(0),
    1 => last!(1),
    _ => last!(2),
  };
  if !matched {
    // The final pass zeroed its range; clear the passes before it.
    // SAFETY: `data` is valid for `start <= len` writes and no other borrow of it is live.
    crate::traits::ct::zeroize(unsafe { core::slice::from_raw_parts_mut(data, start) });
  }
  matched
}

/// [`seal_worker`], then the dead-stack scrub from the same frame.
///
/// # Safety
/// The contract of [`seal_worker`].
#[expect(
  clippy::too_many_arguments,
  reason = "a kernel boundary: key, nonce, counter, GHASH state, ranges and tables"
)]
#[inline(always)]
pub(crate) unsafe fn seal<const N: usize>(
  rk: &[__m128i; N],
  j0: &[u8; 16],
  first: u32,
  acc: u128,
  src: *const u8,
  src_len: usize,
  tail: &[&[u8]],
  dst: *mut u8,
  length_block: &[u8; 16],
  h_powers_rev_16: &[u128; 16],
) -> [u8; 16] {
  // SAFETY: the caller upholds the worker's contract.
  let tag = unsafe {
    seal_worker(
      rk,
      j0,
      first,
      acc,
      src,
      src_len,
      tail,
      dst,
      length_block,
      h_powers_rev_16,
    )
  };
  // SAFETY: the caller establishes AVX512F.
  unsafe { scrub_dead_stack_avx512::<STACK_SCRUB_WORDS>() };
  tag
}

/// [`open_worker`], then the dead-stack scrub from the same frame.
///
/// # Safety
/// The contract of [`open_worker`].
#[expect(
  clippy::too_many_arguments,
  reason = "a kernel boundary: key, nonce, counter, GHASH state, ranges and tables"
)]
#[inline(always)]
pub(crate) unsafe fn open<const N: usize>(
  rk: &[__m128i; N],
  j0: &[u8; 16],
  first: u32,
  acc: u128,
  data: *mut u8,
  len: usize,
  length_block: &[u8; 16],
  h_powers_rev_16: &[u128; 16],
  expected: &[u8; 16],
) -> bool {
  // SAFETY: the caller upholds the worker's contract.
  let matched = unsafe { open_worker(rk, j0, first, acc, data, len, length_block, h_powers_rev_16, expected) };
  // SAFETY: the caller establishes AVX512F.
  unsafe { scrub_dead_stack_avx512::<STACK_SCRUB_WORDS>() };
  matched
}
