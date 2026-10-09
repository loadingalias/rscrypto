//! SIMD128 BLAKE3 from the portable compression equations.
//!
//! The module is compiled only when the artifact requires SIMD128. Each vector
//! holds the same word from four independent chunks or parent blocks.
//! Loads and stores accept byte-aligned input; no relaxed-SIMD instructions are used.

use core::arch::wasm32::{
  i8x16_shuffle, i32x4_shuffle, u32x4, u32x4_add, u32x4_shl, u32x4_shr, u32x4_splat, v128, v128_load, v128_or,
  v128_store, v128_xor,
};

use super::{
  BLOCK_LEN, CHUNK_END, CHUNK_LEN, CHUNK_START, DERIVE_KEY_MATERIAL, IV, KEYED_HASH, MSG_SCHEDULE, OUT_LEN, PARENT,
  ROOT, first_8_words, words16_from_le_bytes_64,
};
use crate::traits::ct;

const DEGREE: usize = 4;

#[inline(always)]
fn secret(flags: u32) -> bool {
  flags & (KEYED_HASH | DERIVE_KEY_MATERIAL) != 0
}

/// Clear explicit vector scratch; the caller supplies one fence after all scratch.
#[inline(always)]
fn clear_vectors(vectors: &mut [v128]) {
  for vector in vectors {
    // SAFETY: the exclusive reference is aligned and writable for one v128.
    unsafe { core::ptr::write_volatile(vector, u32x4_splat(0)) };
  }
}

#[inline(always)]
fn rotate_right<const N: u32>(value: v128) -> v128 {
  v128_or(u32x4_shr(value, N), u32x4_shl(value, 32u32.strict_sub(N)))
}

#[inline(always)]
fn g(mut a: v128, mut b: v128, mut c: v128, mut d: v128, x: v128, y: v128) -> (v128, v128, v128, v128) {
  // i32x4.add is addition modulo 2^32, matching the portable wrapping_add.
  a = u32x4_add(u32x4_add(a, b), x);
  d = v128_xor(d, a);
  d = i8x16_shuffle::<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>(d, d);
  c = u32x4_add(c, d);
  b = rotate_right::<12>(v128_xor(b, c));
  a = u32x4_add(u32x4_add(a, b), y);
  d = v128_xor(d, a);
  d = i8x16_shuffle::<1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12>(d, d);
  c = u32x4_add(c, d);
  b = rotate_right::<7>(v128_xor(b, c));
  (a, b, c, d)
}

#[inline]
pub(super) fn chunk_compress_blocks(cv: &mut [u32; 8], counter: u64, flags: u32, count: &mut u8, blocks: &[u8]) {
  let (blocks, tail) = blocks.as_chunks::<BLOCK_LEN>();
  debug_assert!(tail.is_empty());
  let mut words = [0u32; 16];
  let mut output = [0u32; 16];
  for block in blocks {
    words = words16_from_le_bytes_64(block);
    let start = if *count == 0 { CHUNK_START } else { 0 };
    output = super::compress(cv, &words, counter, 64, flags | start);
    *cv = first_8_words(output);
    *count = count.strict_add(1);
  }
  if secret(flags) {
    ct::zeroize_words_no_fence(&mut words);
    ct::zeroize_words_no_fence(&mut output);
    ct::zeroize_fence();
  }
}

#[inline]
pub(super) fn parent_cv(mut left: [u32; 8], mut right: [u32; 8], mut key: [u32; 8], flags: u32) -> [u32; 8] {
  let mut block = [0u32; 16];
  block[..8].copy_from_slice(&left);
  block[8..].copy_from_slice(&right);
  let mut output = super::compress(&key, &block, 0, 64, flags | PARENT);
  let cv = first_8_words(output);
  if secret(flags) {
    ct::zeroize_words_no_fence(&mut left);
    ct::zeroize_words_no_fence(&mut right);
    ct::zeroize_words_no_fence(&mut key);
    ct::zeroize_words_no_fence(&mut block);
    ct::zeroize_words_no_fence(&mut output);
    ct::zeroize_fence();
  }
  cv
}

#[inline(always)]
fn round_lanes<const ROUND: usize>(v: &mut [v128; 16], m: &[v128; 16]) {
  let s = MSG_SCHEDULE[ROUND];
  macro_rules! mix {
    ($a:literal, $b:literal, $c:literal, $d:literal, $x:literal, $y:literal) => {
      (v[$a], v[$b], v[$c], v[$d]) = g(v[$a], v[$b], v[$c], v[$d], m[s[$x]], m[s[$y]]);
    };
  }
  mix!(0, 4, 8, 12, 0, 1);
  mix!(1, 5, 9, 13, 2, 3);
  mix!(2, 6, 10, 14, 4, 5);
  mix!(3, 7, 11, 15, 6, 7);
  mix!(0, 5, 10, 15, 8, 9);
  mix!(1, 6, 11, 12, 10, 11);
  mix!(2, 7, 8, 13, 12, 13);
  mix!(3, 4, 9, 14, 14, 15);
}

/// Transpose four rows of four words, without changing byte order.
#[inline(always)]
fn transpose(a: v128, b: v128, c: v128, d: v128) -> (v128, v128, v128, v128) {
  let ab0 = i32x4_shuffle::<0, 4, 1, 5>(a, b);
  let ab1 = i32x4_shuffle::<2, 6, 3, 7>(a, b);
  let cd0 = i32x4_shuffle::<0, 4, 1, 5>(c, d);
  let cd1 = i32x4_shuffle::<2, 6, 3, 7>(c, d);
  (
    i32x4_shuffle::<0, 1, 4, 5>(ab0, cd0),
    i32x4_shuffle::<2, 3, 6, 7>(ab0, cd0),
    i32x4_shuffle::<0, 1, 4, 5>(ab1, cd1),
    i32x4_shuffle::<2, 3, 6, 7>(ab1, cd1),
  )
}

/// Compress four equal-length chunks or four parent blocks.
///
/// # Safety
///
/// Each input pointer must address `len` readable bytes, with `len` in
/// `1..=CHUNK_LEN`. For parents, `len` must be `BLOCK_LEN` and every counter
/// must be zero. Inputs may overlap each other, but not `out`.
unsafe fn hash4<const IS_PARENT: bool, const IS_ROOT: bool>(
  inputs: [*const u8; DEGREE],
  len: usize,
  key: &[u32; 8],
  counters: [u64; DEGREE],
  flags: u32,
  out: &mut [[u8; OUT_LEN]; DEGREE],
) {
  debug_assert!((1..=CHUNK_LEN).contains(&len));
  debug_assert!(!IS_PARENT || (len == BLOCK_LEN && counters == [0; DEGREE]));
  // Specialize root batches separately from full-chunk chaining-value calls.
  let flags = if IS_ROOT { flags | ROOT } else { flags };
  let mut state = [u32x4_splat(0); 16];
  // The first eight state words carry the CV between blocks.
  for (word, &key_word) in state.iter_mut().zip(key) {
    *word = u32x4_splat(key_word);
  }
  let mut message = [u32x4_splat(0); 16];
  let mut padded = [[0u8; BLOCK_LEN]; DEGREE];
  let low = counters.map(|counter| u32::try_from(counter & u64::from(u32::MAX)).expect("low word fits"));
  let high = counters.map(|counter| u32::try_from(counter >> 32).expect("high word fits"));
  let blocks = len.div_ceil(BLOCK_LEN);
  for block in 0..blocks {
    let offset = block.strict_mul(BLOCK_LEN);
    let take = len.strict_sub(offset).min(BLOCK_LEN);
    let mut ptrs = inputs;
    for (lane, ptr) in ptrs.iter_mut().enumerate() {
      // SAFETY: offset is inside each len-byte input by the block loop bound.
      *ptr = unsafe { ptr.add(offset) };
      if take != BLOCK_LEN {
        // SAFETY: exactly take remaining bytes are readable; the separate
        // zero-initialized lane has room for them and the rest stays zero.
        unsafe { core::ptr::copy_nonoverlapping(*ptr, padded[lane].as_mut_ptr(), take) };
        *ptr = padded[lane].as_ptr();
      }
    }
    for word in (0usize..16).step_by(4) {
      let byte = word.strict_mul(4);
      // SAFETY: each pointer addresses a full readable block, either in the
      // input or padded scratch. byte is 0, 16, 32, or 48. Loads are 1-aligned.
      let (a, b, c, d) = unsafe {
        transpose(
          v128_load(ptrs[0].add(byte).cast()),
          v128_load(ptrs[1].add(byte).cast()),
          v128_load(ptrs[2].add(byte).cast()),
          v128_load(ptrs[3].add(byte).cast()),
        )
      };
      message[word] = a;
      message[word.strict_add(1)] = b;
      message[word.strict_add(2)] = c;
      message[word.strict_add(3)] = d;
    }
    for (dst, &iv) in state[8..12].iter_mut().zip(&IV) {
      *dst = u32x4_splat(iv);
    }
    state[12] = u32x4(low[0], low[1], low[2], low[3]);
    state[13] = u32x4(high[0], high[1], high[2], high[3]);
    state[14] = u32x4_splat(u32::try_from(take).expect("block length fits"));
    let block_flags = if IS_PARENT {
      flags | PARENT
    } else {
      (flags & !ROOT)
        | if block == 0 { CHUNK_START } else { 0 }
        | if block.strict_add(1) == blocks {
          CHUNK_END | (flags & ROOT)
        } else {
          0
        }
    };
    state[15] = u32x4_splat(block_flags);
    round_lanes::<0>(&mut state, &message);
    round_lanes::<1>(&mut state, &message);
    round_lanes::<2>(&mut state, &message);
    round_lanes::<3>(&mut state, &message);
    round_lanes::<4>(&mut state, &message);
    round_lanes::<5>(&mut state, &message);
    round_lanes::<6>(&mut state, &message);
    let (cv, upper) = state.split_at_mut(8);
    for (word, upper_word) in cv.iter_mut().zip(upper) {
      *word = v128_xor(*word, *upper_word);
    }
  }
  for word in [0usize, 4] {
    let (a, b, c, d) = transpose(
      state[word],
      state[word.strict_add(1)],
      state[word.strict_add(2)],
      state[word.strict_add(3)],
    );
    let byte = word.strict_mul(4);
    // SAFETY: each output is 32 writable bytes; the two iterations write the
    // disjoint 16-byte halves. v128_store accepts their byte alignment.
    unsafe {
      v128_store(out[0].as_mut_ptr().add(byte).cast(), a);
      v128_store(out[1].as_mut_ptr().add(byte).cast(), b);
      v128_store(out[2].as_mut_ptr().add(byte).cast(), c);
      v128_store(out[3].as_mut_ptr().add(byte).cast(), d);
    }
  }
  if secret(flags) {
    clear_vectors(&mut state);
    clear_vectors(&mut message);
    if !IS_PARENT && !len.is_multiple_of(BLOCK_LEN) {
      // Only a partial final block populates padded; complete blocks leave it zero.
      ct::zeroize_no_fence(padded.as_flattened_mut());
    }
    ct::zeroize_fence();
  }
}

/// Hash complete contiguous chunks into their chaining values.
///
/// # Safety
///
/// `input` must address `num_chunks * CHUNK_LEN` readable bytes; `out` must
/// address `num_chunks * OUT_LEN` writable bytes. The two ranges must not overlap.
pub(super) unsafe fn hash_many_contiguous(
  input: *const u8,
  num_chunks: usize,
  key: &[u32; 8],
  counter: u64,
  flags: u32,
  out: *mut u8,
) {
  let mut outputs = [[0u8; OUT_LEN]; DEGREE];
  let mut done = 0usize;
  while done < num_chunks {
    let take = num_chunks.strict_sub(done).min(DEGREE);
    // SAFETY: done < num_chunks, so this starts a readable full chunk.
    let first = unsafe { input.add(done.strict_mul(CHUNK_LEN)) };
    let mut inputs = [first; DEGREE];
    for (lane, ptr) in inputs.iter_mut().take(take).enumerate() {
      // SAFETY: done + lane < num_chunks. Unused lanes repeat the first chunk.
      *ptr = unsafe { first.add(lane.strict_mul(CHUNK_LEN)) };
    }
    let base = counter.wrapping_add(u64::try_from(done).expect("chunk index fits"));
    let counters = [base, base.wrapping_add(1), base.wrapping_add(2), base.wrapping_add(3)];
    // SAFETY: every lane addresses a full chunk and outputs is separate scratch.
    unsafe { hash4::<false, false>(inputs, CHUNK_LEN, key, counters, flags, &mut outputs) };
    // SAFETY: the caller provides the complete output range; done + take is
    // bounded by num_chunks, and scratch cannot alias that range.
    unsafe {
      core::ptr::copy_nonoverlapping(
        outputs.as_ptr().cast::<u8>(),
        out.add(done.strict_mul(OUT_LEN)),
        take.strict_mul(OUT_LEN),
      )
    };
    done = done.strict_add(take);
  }
  if secret(flags) {
    ct::zeroize(outputs.as_flattened_mut());
  }
}

/// Hash four unkeyed, equal-length, one-chunk messages to root digests.
///
/// # Safety
///
/// Each pointer must address `len` readable bytes, `len` must be in
/// `1..=CHUNK_LEN`, and the inputs must not overlap `out`.
pub(super) unsafe fn hash4_roots(inputs: [*const u8; DEGREE], len: usize, out: &mut [[u8; OUT_LEN]; DEGREE]) {
  // SAFETY: the caller establishes the hash4 input and output contract.
  unsafe { hash4::<false, true>(inputs, len, &IV, [0; DEGREE], 0, out) };
}

/// Reduce four adjacent child pairs directly into their parent CVs.
pub(super) fn parent_cvs4(
  children: &[[u8; OUT_LEN]; 2 * DEGREE],
  key: &[u32; 8],
  flags: u32,
  out: &mut [[u8; OUT_LEN]; DEGREE],
) {
  let base = children.as_ptr().cast::<u8>();
  let inputs = core::array::from_fn(|lane: usize| {
    // SAFETY: children contains four contiguous 64-byte pairs. The offset
    // selects the beginning of one pair within that same borrowed array.
    unsafe { base.add(lane.strict_mul(BLOCK_LEN)) }
  });
  // SAFETY: each input covers one complete parent block. The exclusive output
  // borrow cannot overlap children, and parent counters are all zero.
  unsafe { hash4::<true, false>(inputs, BLOCK_LEN, key, [0; DEGREE], flags, out) };
}
