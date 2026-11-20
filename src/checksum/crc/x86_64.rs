//! x86-64 carryless-multiply CRC engine (PCLMULQDQ, and VPCLMULQDQ with AVX-512).
//!
//! One engine serves every CRC width and both bit orders; see the parent module for the 64-bit
//! domain. The VPCLMULQDQ tier folds 64-byte vectors and finishes with the 16-byte engine, which
//! it compiles with AVX-512 so that LLVM merges each three-way XOR into one `VPTERNLOGQ`.

use core::arch::x86_64::{
  __m128i, __m512i, _mm_and_si128, _mm_clmulepi64_si128, _mm_crc32_u8, _mm_crc32_u64, _mm_cvtsi128_si64,
  _mm_loadu_si128, _mm_set_epi64x, _mm_shuffle_epi8, _mm_unpackhi_epi64, _mm_xor_si128, _mm512_broadcast_i32x4,
  _mm512_castsi512_si128, _mm512_clmulepi64_epi128, _mm512_extracti32x4_epi32, _mm512_inserti32x4, _mm512_loadu_si512,
  _mm512_setzero_si512, _mm512_shuffle_epi8, _mm512_xor_si512,
};

use super::{
  Algo, Hw, Keys, MASK, Word,
  dispatch::{CrcForce, Kernel, fold_with},
  shift_by,
};

/// Lanes in the 16-byte main loop: eight lanes, 128 bytes per iteration.
const LANES: usize = 8;

/// 64-byte vectors in the VPCLMULQDQ main loop: 256 bytes per iteration.
const WIDE_LANES: usize = 4;

/// Indices that reverse the bytes of a 16-byte block.
const REVERSE: [u8; 16] = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];

/// # Safety
///
/// The CPU must support SSE2.
#[inline]
#[target_feature(enable = "sse2")]
fn load(bytes: &[u8; 16]) -> __m128i {
  // SAFETY: `bytes` is 16 readable bytes, and `_mm_loadu_si128` needs no alignment.
  unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) }
}

/// Loads a block as a 128-bit polynomial lane in the engine's bit order.
///
/// # Safety
///
/// The CPU must support SSE2, SSSE3.
#[inline]
#[target_feature(enable = "sse2,ssse3")]
fn load_lane<const REFLECT: bool>(bytes: &[u8; 16]) -> __m128i {
  if REFLECT {
    load(bytes)
  } else {
    _mm_shuffle_epi8(load(bytes), load(&REVERSE))
  }
}

/// # Safety
///
/// The CPU must support SSE2.
#[inline]
#[target_feature(enable = "sse2")]
fn pair(lo: u64, hi: u64) -> __m128i {
  _mm_set_epi64x(hi.cast_signed(), lo.cast_signed())
}

/// # Safety
///
/// The CPU must support SSE2.
#[inline]
#[target_feature(enable = "sse2")]
fn key(k: &[u64; 2]) -> __m128i {
  pair(k[0], k[1])
}

/// # Safety
///
/// The CPU must support SSE2.
#[inline]
#[target_feature(enable = "sse2")]
fn lo(v: __m128i) -> u64 {
  _mm_cvtsi128_si64(v).cast_unsigned()
}

/// # Safety
///
/// The CPU must support SSE2.
#[inline]
#[target_feature(enable = "sse2")]
fn hi(v: __m128i) -> u64 {
  _mm_cvtsi128_si64(_mm_unpackhi_epi64(v, v)).cast_unsigned()
}

/// `clmul(a, b)` as a 128-bit lane.
///
/// # Safety
///
/// The CPU must support SSE2, PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
fn clmul(a: u64, b: u64) -> __m128i {
  _mm_clmulepi64_si128::<0x00>(pair(a, 0), pair(b, 0))
}

/// Folds `x` by the distance of `k`: `clmul(x.lo, k.lo) ^ clmul(x.hi, k.hi)`.
///
/// # Safety
///
/// The CPU must support SSE2, PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
fn fold(x: __m128i, k: __m128i) -> __m128i {
  _mm_xor_si128(_mm_clmulepi64_si128::<0x00>(x, k), _mm_clmulepi64_si128::<0x11>(x, k))
}

/// Folds `x` by the distance of `k` and adds `data`.
///
/// # Safety
///
/// The CPU must support SSE2, PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
fn fold_add(x: __m128i, k: __m128i, data: __m128i) -> __m128i {
  _mm_xor_si128(fold(x, k), data)
}

/// Folds in the final partial block of `data`, then reduces to the 64-bit CRC state.
///
/// # Safety
///
/// The CPU must support SSE2, SSSE3, PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2,ssse3,pclmulqdq")]
fn finish<const REFLECT: bool>(mut x: __m128i, data: &[u8], keys: &Keys) -> u64 {
  let r = data.len().strict_rem(16);
  if r != 0 {
    let last = &data[data.len().strict_sub(16)..].as_chunks::<16>().0[0];
    let masked = _mm_and_si128(load(last), load(&MASK[r..].as_chunks::<16>().0[0]));
    let lane = if REFLECT {
      masked
    } else {
      _mm_shuffle_epi8(masked, load(&REVERSE))
    };
    x = fold_add(x, key(&keys.tail[r]), lane);
  }
  barrett::<REFLECT>(fold(x, key(&keys.tail[8])), keys)
}

/// Barrett reduction of a 128-bit product modulo `G'`.
///
/// # Safety
///
/// The CPU must support SSE2, PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
fn barrett<const REFLECT: bool>(y: __m128i, keys: &Keys) -> u64 {
  if REFLECT {
    let q = lo(clmul(lo(y), keys.mu));
    hi(y) ^ hi(clmul(q, keys.poly)) ^ (q & keys.top)
  } else {
    let q = hi(y) ^ hi(clmul(hi(y), keys.mu));
    lo(y) ^ lo(clmul(q, keys.poly))
  }
}

/// Folds eight consecutive lanes into one: lane `j` moves by `7 - j` blocks.
///
/// # Safety
///
/// The CPU must support SSE2, PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
fn merge8(acc: &[__m128i; 8], keys: &Keys) -> __m128i {
  let a = fold_add(acc[0], key(&keys.fold[6]), fold(acc[1], key(&keys.fold[5])));
  let b = fold_add(acc[2], key(&keys.fold[4]), fold(acc[3], key(&keys.fold[3])));
  let c = fold_add(acc[4], key(&keys.fold[2]), fold(acc[5], key(&keys.fold[1])));
  let d = fold_add(acc[6], key(&keys.fold[0]), acc[7]);
  _mm_xor_si128(_mm_xor_si128(a, b), _mm_xor_si128(c, d))
}

/// Appends whole blocks to a lane that precedes them: each moves by its distance from the last.
///
/// # Safety
///
/// The CPU must support SSE2, SSSE3, PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2,ssse3,pclmulqdq")]
fn append<const REFLECT: bool>(x: __m128i, rest: &[[u8; 16]], keys: &Keys) -> __m128i {
  let Some((last, middle)) = rest.split_last() else {
    return x;
  };
  let n = rest.len();
  let mut sum = fold_add(x, key(&keys.fold[n.strict_sub(1)]), load_lane::<REFLECT>(last));
  for (j, block) in middle.iter().enumerate() {
    sum = fold_add(
      load_lane::<REFLECT>(block),
      key(&keys.fold[n.strict_sub(2).strict_sub(j)]),
      sum,
    );
  }
  sum
}

/// Updates a 64-bit-domain CRC state with at least 16 bytes, 16 bytes per lane.
///
/// # Safety
///
/// The CPU must support SSE2, SSSE3, PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2,ssse3,pclmulqdq")]
fn engine<const REFLECT: bool>(state: u64, data: &[u8], keys: &Keys) -> u64 {
  let (blocks, _) = data.as_chunks::<16>();
  let inject = if REFLECT { pair(state, 0) } else { pair(0, state) };
  let (x, rest) = if let Some((first, rest)) = blocks.split_first_chunk::<LANES>() {
    let mut acc = first.map(|block| load_lane::<REFLECT>(&block));
    acc[0] = _mm_xor_si128(acc[0], inject);
    let k = key(&keys.fold[LANES.strict_sub(1)]);
    let (groups, rest) = rest.as_chunks::<LANES>();
    for group in groups {
      for (lane, block) in acc.iter_mut().zip(group) {
        *lane = fold_add(*lane, k, load_lane::<REFLECT>(block));
      }
    }
    (merge8(&acc, keys), rest)
  } else {
    let (first, rest) = blocks.split_at(1);
    (_mm_xor_si128(load_lane::<REFLECT>(&first[0]), inject), rest)
  };
  finish::<REFLECT>(append::<REFLECT>(x, rest, keys), data, keys)
}

/// PCLMULQDQ tier.
///
/// # Safety
///
/// The CPU must support SSSE3 and PCLMULQDQ, and `data` must hold at least 16 bytes.
#[target_feature(enable = "sse2,ssse3,pclmulqdq")]
pub(in crate::checksum) unsafe fn pclmul<const REFLECT: bool>(state: u64, data: &[u8], keys: &Keys) -> u64 {
  engine::<REFLECT>(state, data, keys)
}

/// # Safety
///
/// The CPU must support AVX-512F, AVX-512BW.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn load_wide<const REFLECT: bool>(bytes: &[u8; 64], reverse: __m512i) -> __m512i {
  // SAFETY: `bytes` is 64 readable bytes, and `_mm512_loadu_si512` needs no alignment.
  let v = unsafe { _mm512_loadu_si512(bytes.as_ptr().cast()) };
  if REFLECT { v } else { _mm512_shuffle_epi8(v, reverse) }
}

/// # Safety
///
/// The CPU must support AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
fn key_wide(k: &[u64; 2]) -> __m512i {
  _mm512_broadcast_i32x4(key(k))
}

/// Folds each 16-byte lane of `x` by the distance of `k`.
///
/// # Safety
///
/// The CPU must support AVX-512F, VPCLMULQDQ.
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
fn fold_wide(x: __m512i, k: __m512i) -> __m512i {
  _mm512_xor_si512(
    _mm512_clmulepi64_epi128::<0x00>(x, k),
    _mm512_clmulepi64_epi128::<0x11>(x, k),
  )
}

/// Folds each 16-byte lane of `x` by the distance of `k` and adds `data`.
///
/// # Safety
///
/// The CPU must support AVX-512F, VPCLMULQDQ.
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
fn fold_add_wide(x: __m512i, k: __m512i, data: __m512i) -> __m512i {
  _mm512_xor_si512(fold_wide(x, k), data)
}

/// VPCLMULQDQ tier.
///
/// # Safety
///
/// The CPU must support AVX-512 F, VL, and BW, VPCLMULQDQ, PCLMULQDQ, and SSSE3, and `data` must
/// hold at least 16 bytes.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,vpclmulqdq,sse2,ssse3,pclmulqdq")]
pub(in crate::checksum) unsafe fn vpclmul<const REFLECT: bool>(state: u64, data: &[u8], keys: &Keys) -> u64 {
  let (vectors, rest) = data.as_chunks::<64>();
  let Some((first, more)) = vectors.split_first_chunk::<WIDE_LANES>() else {
    return engine::<REFLECT>(state, data, keys);
  };
  let reverse = _mm512_broadcast_i32x4(load(&REVERSE));
  let inject = if REFLECT { pair(state, 0) } else { pair(0, state) };
  let mut acc = [_mm512_setzero_si512(); WIDE_LANES];
  for (lane, vector) in acc.iter_mut().zip(first) {
    *lane = load_wide::<REFLECT>(vector, reverse);
  }
  acc[0] = _mm512_xor_si512(acc[0], _mm512_inserti32x4::<0>(_mm512_setzero_si512(), inject));
  let k = key_wide(&keys.fold[WIDE_LANES.strict_mul(4).strict_sub(1)]);
  let (groups, more) = more.as_chunks::<WIDE_LANES>();
  for group in groups {
    for (lane, vector) in acc.iter_mut().zip(group) {
      *lane = fold_add_wide(*lane, k, load_wide::<REFLECT>(vector, reverse));
    }
  }
  let mut z = merge_wide(&acc, keys);
  // Up to three 64-byte vectors follow; fold them in one at a time.
  let k1 = key_wide(&keys.fold[3]);
  for vector in more {
    z = fold_add_wide(z, k1, load_wide::<REFLECT>(vector, reverse));
  }
  let (blocks, _) = rest.as_chunks::<16>();
  finish::<REFLECT>(append::<REFLECT>(split_wide(z, keys), blocks, keys), data, keys)
}

/// Folds four consecutive 64-byte lanes into one: lane `j` moves by `3 - j` vectors.
///
/// # Safety
///
/// The CPU must support AVX-512F, VPCLMULQDQ.
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
fn merge_wide(acc: &[__m512i; WIDE_LANES], keys: &Keys) -> __m512i {
  let k3 = key_wide(&keys.fold[11]);
  let k2 = key_wide(&keys.fold[7]);
  let k1 = key_wide(&keys.fold[3]);
  _mm512_xor_si512(
    fold_add_wide(acc[0], k3, fold_wide(acc[1], k2)),
    fold_add_wide(acc[2], k1, acc[3]),
  )
}

/// Folds the four blocks of a 64-byte lane into one 16-byte lane.
///
/// # Safety
///
/// The CPU must support AVX-512F, SSE2, PCLMULQDQ.
#[inline]
#[target_feature(enable = "avx512f,sse2,pclmulqdq")]
fn split_wide(z: __m512i, keys: &Keys) -> __m128i {
  fold_add(
    _mm512_castsi512_si128(z),
    key(&keys.fold[2]),
    fold_add(
      _mm512_extracti32x4_epi32::<1>(z),
      key(&keys.fold[1]),
      fold_add(
        _mm512_extracti32x4_epi32::<2>(z),
        key(&keys.fold[0]),
        _mm512_extracti32x4_epi32::<3>(z),
      ),
    ),
  )
}

/// CRC-32C of eight bytes with the SSE4.2 instruction.
///
/// # Safety
///
/// The CPU must support SSE4.2.
#[inline]
#[target_feature(enable = "sse4.2")]
fn crc32c_word(crc: u32, word: &[u8; 8]) -> u32 {
  super::low32(_mm_crc32_u64(u64::from(crc), u64::from_le_bytes(*word)))
}

/// CRC-32C with the SSE4.2 instruction, eight bytes per instruction.
///
/// # Safety
///
/// The CPU must support SSE4.2.
#[inline]
#[target_feature(enable = "sse4.2")]
fn crc32c_hw(mut crc: u32, data: &[u8]) -> u32 {
  let (words, tail) = data.as_chunks::<8>();
  for word in words {
    crc = crc32c_word(crc, word);
  }
  for &byte in tail {
    crc = _mm_crc32_u8(crc, byte);
  }
  crc
}

/// Inputs below this length run the CRC-32C instruction alone.
const HW_ONLY_MAX: usize = 255;

/// Fused steps in the shortest fused round.
const FUSED_MIN_STEPS: usize = 8;

/// One step of the three CRC-32C streams: 16 bytes each.
///
/// # Safety
///
/// The CPU must support SSE4.2.
#[inline]
#[target_feature(enable = "sse4.2")]
fn streams_step(c: [u32; 3], words: [&[[u8; 8]]; 3], i: usize) -> [u32; 3] {
  let w = i.strict_mul(2);
  let step = |crc: u32, words: &[[u8; 8]]| crc32c_word(crc32c_word(crc, &words[w]), &words[w.strict_add(1)]);
  [step(c[0], words[0]), step(c[1], words[1]), step(c[2], words[2])]
}

/// Joins a fused round: the register after the vector part and after each stream.
///
/// `crc = s_v * x^(48n) ^ s_1 * x^(32n) ^ s_2 * x^(16n) ^ s_3`, where each multiplier is a combine
/// power because `16n` is a power of two. The CRC of a lane's 16 bytes from a zero register is the
/// lane times `x^32 mod P`, which is the register after the vector part.
///
/// # Safety
///
/// The CPU must support SSE2, SSE4.2, PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2,sse4.2,pclmulqdq")]
fn join(x: __m128i, c: [u32; 3], n: usize, keys: &Keys, powers: &[u64; 64]) -> u64 {
  let vector = crc32c_word(crc32c_word(0, &lo(x).to_le_bytes()), &hi(x).to_le_bytes());
  let j = n.trailing_zeros() as usize;
  let (p16, p32) = (powers[j.strict_add(4)], powers[j.strict_add(5)]);
  let multiply = |value: u64, power: u64| barrett::<true>(clmul(value, power), keys);
  multiply(multiply(u64::from(vector), p32), p16)
    ^ multiply(u64::from(c[0]), p32)
    ^ multiply(u64::from(c[1]), p16)
    ^ u64::from(c[2])
}

/// Splits a round into its vector part and the three 16n-byte streams after it.
#[inline(always)]
fn split_round(round: &[u8], vector_len: usize, n: usize) -> (&[u8], [&[[u8; 8]]; 3]) {
  let (vector, scalar) = round.split_at(vector_len);
  let (words, _) = scalar.as_chunks::<8>();
  let (one, rest) = words.split_at(n.strict_mul(2));
  let (two, three) = rest.split_at(n.strict_mul(2));
  assert!(three.len() == n.strict_mul(2));
  (vector, [one, two, three])
}

/// One fused round of `n` steps: eight 16-byte lanes over `128 * n` bytes, then three streams.
///
/// # Safety
///
/// The CPU must support SSE2, SSSE3, SSE4.2, PCLMULQDQ.
#[inline]
#[target_feature(enable = "sse2,ssse3,sse4.2,pclmulqdq")]
fn fused_round(state: u64, round: &[u8], n: usize, keys: &Keys, powers: &[u64; 64]) -> u64 {
  let (vector, words) = split_round(round, n.strict_mul(128), n);
  let (groups, _) = vector.as_chunks::<16>().0.as_chunks::<LANES>();
  assert!(groups.len() == n);
  let mut acc = groups[0].map(|block| load_lane::<true>(&block));
  acc[0] = _mm_xor_si128(acc[0], pair(state, 0));
  let mut c = streams_step([0; 3], words, 0);
  let k = key(&keys.fold[LANES.strict_sub(1)]);
  let mut i = 1usize;
  while i < n {
    for (lane, block) in acc.iter_mut().zip(&groups[i]) {
      *lane = fold_add(*lane, k, load_lane::<true>(block));
    }
    c = streams_step(c, words, i);
    i = i.strict_add(1);
  }
  join(merge8(&acc, keys), c, n, keys, powers)
}

/// One fused round of `n` steps: four 64-byte lanes over `256 * n` bytes, then three streams.
///
/// # Safety
///
/// The CPU must support AVX-512F, AVX-512VL, AVX-512BW, VPCLMULQDQ, SSE2, SSSE3, SSE4.2, PCLMULQDQ.
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,vpclmulqdq,sse2,ssse3,sse4.2,pclmulqdq")]
fn fused_round_wide(state: u64, round: &[u8], n: usize, keys: &Keys, powers: &[u64; 64]) -> u64 {
  let (vector, words) = split_round(round, n.strict_mul(256), n);
  let (groups, _) = vector.as_chunks::<64>().0.as_chunks::<WIDE_LANES>();
  assert!(groups.len() == n);
  let mut acc = groups[0].map(|block| load_wide::<true>(&block, _mm512_setzero_si512()));
  acc[0] = _mm512_xor_si512(acc[0], _mm512_inserti32x4::<0>(_mm512_setzero_si512(), pair(state, 0)));
  let mut c = streams_step([0; 3], words, 0);
  let k = key_wide(&keys.fold[WIDE_LANES.strict_mul(4).strict_sub(1)]);
  let mut i = 1usize;
  while i < n {
    for (lane, vector) in acc.iter_mut().zip(&groups[i]) {
      *lane = fold_add_wide(*lane, k, load_wide::<true>(vector, _mm512_setzero_si512()));
    }
    c = streams_step(c, words, i);
    i = i.strict_add(1);
  }
  join(split_wide(merge_wide(&acc, keys), keys), c, n, keys, powers)
}

/// CRC-32C: the instruction for short inputs, fused rounds for long ones, and folding for the rest.
///
/// # Safety
///
/// The CPU must support SSE4.2, SSSE3, and PCLMULQDQ, and the VPCLMULQDQ tier's features when
/// `WIDE`.
#[inline(always)]
unsafe fn fused<const WIDE: bool>(mut state: u64, mut data: &[u8], keys: &Keys, powers: &[u64; 64]) -> u64 {
  // SAFETY: the caller guarantees the target features.
  unsafe {
    if data.len() <= HW_ONLY_MAX {
      return u64::from(crc32c_hw(super::low32(state), data));
    }
    let step = if WIDE { 256 + 48 } else { 128 + 48 };
    loop {
      let steps = data.len().strict_div(step);
      if steps < FUSED_MIN_STEPS {
        break;
      }
      let n = 1usize << steps.ilog2();
      let (round, rest) = data.split_at(n.strict_mul(step));
      state = if WIDE {
        fused_round_wide(state, round, n, keys, powers)
      } else {
        fused_round(state, round, n, keys, powers)
      };
      data = rest;
    }
    if data.len() < 16 {
      u64::from(crc32c_hw(super::low32(state), data))
    } else if WIDE {
      vpclmul::<true>(state, data, keys)
    } else {
      engine::<true>(state, data, keys)
    }
  }
}

/// PCLMULQDQ tier for CRC-32C, fused with the SSE4.2 instruction.
///
/// # Safety
///
/// The CPU must support SSE4.2, SSSE3, and PCLMULQDQ.
#[target_feature(enable = "sse2,ssse3,sse4.2,pclmulqdq")]
unsafe fn pclmul_crc32c(state: u64, data: &[u8], keys: &Keys, powers: &[u64; 64]) -> u64 {
  // SAFETY: this function enables the features.
  unsafe { fused::<false>(state, data, keys, powers) }
}

/// VPCLMULQDQ tier for CRC-32C, fused with the SSE4.2 instruction.
///
/// # Safety
///
/// The CPU must support the VPCLMULQDQ tier's features and SSE4.2.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,vpclmulqdq,sse2,ssse3,sse4.2,pclmulqdq")]
unsafe fn vpclmul_crc32c(state: u64, data: &[u8], keys: &Keys, powers: &[u64; 64]) -> u64 {
  // SAFETY: this function enables the features.
  unsafe { fused::<true>(state, data, keys, powers) }
}

/// A fused tier for CRC-32C. The selector runs it only on a CPU with the tier's features and
/// SSE4.2.
fn fused_kernel<A: Algo, const WIDE: bool>(register: A::W, data: &[u8]) -> A::W {
  let crc = A::CRC;
  let state = register.widen();
  // SAFETY: the selector checks the features.
  let state = unsafe {
    if WIDE {
      vpclmul_crc32c(state, data, &crc.keys, &crc.powers)
    } else {
      pclmul_crc32c(state, data, &crc.keys, &crc.powers)
    }
  };
  A::W::narrow(state)
}

/// Hwcrc tier. The selector runs it only for CRC-32C, on a CPU with SSE4.2.
fn hwcrc_kernel<A: Algo>(register: A::W, data: &[u8]) -> A::W {
  // SAFETY: the selector checks SSE4.2 and that `A` is CRC-32C.
  let crc = unsafe { crc32c_hw(super::low32(register.widen()), data) };
  A::W::narrow(u64::from(crc))
}

/// PCLMULQDQ tier.
fn pclmul_kernel<A: Algo>(register: A::W, data: &[u8]) -> A::W {
  // SAFETY: the selector checks PCLMULQDQ and SSSE3.
  unsafe { fold_with::<A>(register, data, pclmul::<true>, pclmul::<false>) }
}

/// VPCLMULQDQ tier.
fn vpclmul_kernel<A: Algo>(register: A::W, data: &[u8]) -> A::W {
  // SAFETY: the selector checks VPCLMULQDQ, AVX-512, PCLMULQDQ, and SSSE3.
  unsafe { fold_with::<A>(register, data, vpclmul::<true>, vpclmul::<false>) }
}

/// The kernel of an x86-64 tier for CRC `A`. The caller has checked that the CPU runs `tier`.
pub(in crate::checksum) fn kernel<A: Algo>(tier: CrcForce) -> Kernel<A::W> {
  let fuse = A::HW == Hw::Crc32c && crate::platform::caps().has(crate::platform::caps::x86::CRC32C_READY);
  match (tier, fuse) {
    (CrcForce::Hwcrc, _) => hwcrc_kernel::<A>,
    (CrcForce::Vpclmul, true) => fused_kernel::<A, true>,
    (CrcForce::Vpclmul, false) => vpclmul_kernel::<A>,
    (_, true) => fused_kernel::<A, false>,
    (_, false) => pclmul_kernel::<A>,
  }
}

/// Multiplies an engine-domain value by `x^(8 * len)`, for `combine`.
///
/// # Safety
///
/// The CPU must support PCLMULQDQ.
#[target_feature(enable = "sse2,pclmulqdq")]
pub(in crate::checksum) unsafe fn shift<const REFLECT: bool>(
  value: u64,
  len: usize,
  powers: &[u64; 64],
  keys: &Keys,
) -> u64 {
  shift_by(value, len, powers, |value, power| {
    barrett::<REFLECT>(clmul(value, power), keys)
  })
}

/// `combine` multiplier of the PCLMULQDQ tiers. The selector runs it only on a CPU with PCLMULQDQ.
pub(in crate::checksum) fn shift_kernel<A: Algo>(value: u64, len: usize) -> u64 {
  let crc = A::CRC;
  // SAFETY: the selector checks PCLMULQDQ.
  unsafe {
    if crc.spec.reflected {
      shift::<true>(value, len, &crc.powers, &crc.keys)
    } else {
      shift::<false>(value, len, &crc.powers, &crc.keys)
    }
  }
}
