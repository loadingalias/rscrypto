//! Eight-lane AVX-512 IFMA Poly1305 bulk kernel.
//!
//! Values use radix 2^44: limbs of 44, 44 and 42 bits for bits 0-43, 44-87 and 88-129. A
//! product term of weight 2^132 folds back multiplied by 20, because `2^130 ≡ 5`. Each 52-bit
//! multiply-add contributes the low or the high 52 bits of one limb product.
//!
//! Each pass takes a sixteen-block group into eight lanes, two blocks per lane: lane `j` updates as
//! `h·r^k + m_{16g+j}·r^8 + m_{16g+8+j}`, so one carry pass serves sixteen blocks. Thirty-two-block
//! groups run two such chains with `k = 32`, which hides the latency of each chain's dependent
//! multiply-adds and carries. The first pass has no running product to multiply, so it needs only
//! `r^8` while the higher powers are computed. After the last group, each lane multiplies by the
//! power that completes its blocks' weights and the lanes are summed. A remaining sixteen-block
//! group takes one single-chain pass, and fewer than sixteen blocks take one more pass whose lane
//! `i` (counting `i` up to 15 across two vectors) multiplies by `r^(k-i)`. The running accumulator
//! enters lane 0 with the first block of each pass. Only the public block count selects the passes.

use core::arch::x86_64::*;

use super::scalar64::State;
use crate::traits::ct;

/// Fewest blocks for which the kernel's power setup beats the scalar loop.
pub(crate) const MIN_BLOCKS: usize = 24;

const MASK44: u64 = (1 << 44) - 1;
const MASK42: u64 = (1 << 42) - 1;

#[inline(always)]
fn low_u64(value: u128) -> u64 {
  let [b0, b1, b2, b3, b4, b5, b6, b7, ..] = value.to_le_bytes();
  u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7])
}

/// A residue in radix 2^44.
#[derive(Clone, Copy, Default)]
struct Limbs([u64; 3]);

impl Limbs {
  /// Splits a radix-2^64 value with a top word below 2^4.
  #[inline(always)]
  fn from_words(w0: u64, w1: u64, w2: u64) -> Self {
    debug_assert!(w2 < 16);
    Self([w0 & MASK44, ((w0 >> 44) | (w1 << 20)) & MASK44, (w1 >> 24) | (w2 << 40)])
  }

  /// `(self·r) mod (2^130 - 5)`, partially reduced: limbs below 2^44, 2^44 + 2^8, and 2^42.
  ///
  /// Both operands hold limbs below 2^45.
  #[inline(always)]
  fn mul(self, r: Self) -> Self {
    let [h0, h1, h2] = self.0.map(u128::from);
    let [r0, r1, r2] = r.0.map(u128::from);
    // 20·r limbs are below 2^50; each product is below 2^95 and each column below 2^97.
    let s1 = r1.wrapping_mul(20);
    let s2 = r2.wrapping_mul(20);
    let d0 = h0
      .wrapping_mul(r0)
      .wrapping_add(h1.wrapping_mul(s2))
      .wrapping_add(h2.wrapping_mul(s1));
    let d1 = h0
      .wrapping_mul(r1)
      .wrapping_add(h1.wrapping_mul(r0))
      .wrapping_add(h2.wrapping_mul(s2));
    let d2 = h0
      .wrapping_mul(r2)
      .wrapping_add(h1.wrapping_mul(r1))
      .wrapping_add(h2.wrapping_mul(r0));
    let t0 = low_u64(d0) & MASK44;
    let d1 = d1.wrapping_add(d0 >> 44);
    let t1 = low_u64(d1) & MASK44;
    let d2 = d2.wrapping_add(d1 >> 44);
    let t2 = low_u64(d2) & MASK42;
    // d2 < 2^97, so five times the excess d2 >> 42 stays below 2^58.
    let t0 = t0.wrapping_add(low_u64(d2 >> 42).wrapping_mul(5));
    let t1 = t1.wrapping_add(t0 >> 44);
    Self([t0 & MASK44, t1, t2])
  }

  /// Carries limbs below 2^56 into partially reduced form.
  #[inline(always)]
  fn carry(self) -> Self {
    let [l0, l1, l2] = self.0;
    let l1 = l1.wrapping_add(l0 >> 44);
    let l2 = l2.wrapping_add(l1 >> 44);
    let l0 = (l0 & MASK44).wrapping_add((l2 >> 42).wrapping_mul(5));
    let l1 = (l1 & MASK44).wrapping_add(l0 >> 44);
    Self([l0 & MASK44, l1, l2 & MASK42])
  }

  /// Converts partially reduced limbs to the radix-2^64 accumulator form with a top word <= 4.
  #[inline(always)]
  fn to_words(self) -> [u64; 3] {
    let [l0, l1, l2] = self.0;
    // l0 < 2^44, l1 < 2^45, l2 < 2^42: the value is below 2^130 + 2^89.
    let low = u128::from(l0).wrapping_add(u128::from(l1) << 44);
    let (low, carry) = low.overflowing_add(u128::from(l2 & ((1 << 40) - 1)) << 88);
    let top = (l2 >> 40).wrapping_add(u64::from(carry));
    let [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15] = low.to_le_bytes();
    [
      u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]),
      u64::from_le_bytes([b8, b9, b10, b11, b12, b13, b14, b15]),
      top,
    ]
  }
}

/// Eight lanes of radix-2^44 limbs.
#[derive(Clone, Copy)]
struct Lanes([__m512i; 3]);

/// A lane multiplier `r` and its limbs 1 and 2 times 20.
#[derive(Clone, Copy)]
struct Multiplier {
  r: Lanes,
  s1: __m512i,
  s2: __m512i,
}

/// Lane-wise `x·20` for limbs below 2^47.
///
/// # Safety
///
/// Calling it outside an AVX-512F-enabled function requires AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
fn times20(x: __m512i) -> __m512i {
  _mm512_add_epi64(_mm512_slli_epi64::<4>(x), _mm512_slli_epi64::<2>(x))
}

impl Multiplier {
  /// # Safety
  ///
  /// Calling it outside an AVX-512F-enabled function requires AVX-512F.
  #[inline]
  #[target_feature(enable = "avx512f")]
  fn new(r: Lanes) -> Self {
    Self {
      r,
      s1: times20(r.0[1]),
      s2: times20(r.0[2]),
    }
  }
}

/// Column sums of radix-2^44 products, each kept as its low and high 52-bit halves:
/// column `k` is `lo[k] + hi[k]·2^52`.
#[derive(Clone, Copy)]
struct Columns {
  lo: [__m512i; 3],
  hi: [__m512i; 3],
}

/// # Safety
///
/// Calling it outside an AVX-512F-enabled function requires AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
fn zero_columns() -> Columns {
  let zero = _mm512_setzero_si512();
  Columns {
    lo: [zero; 3],
    hi: [zero; 3],
  }
}

/// Adds the column sums of `h·r` in every lane, with `2^132 ≡ 20` folding the terms of limbs 1 and 2.
///
/// `h` and `m.r` hold limbs below 2^45. Each call adds three values below 2^52 to every low half and
/// three values below 2^42 to every high half.
///
/// # Safety
///
/// Calling it outside an AVX-512F and AVX-512 IFMA function requires both features.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn accumulate(c: Columns, h: Lanes, m: &Multiplier) -> Columns {
  let [h0, h1, h2] = h.0;
  let [r0, r1, r2] = m.r.0;
  let (s1, s2) = (m.s1, m.s2);
  let [l0, l1, l2] = c.lo;
  let [g0, g1, g2] = c.hi;
  Columns {
    lo: [
      _mm512_madd52lo_epu64(_mm512_madd52lo_epu64(_mm512_madd52lo_epu64(l0, h0, r0), h1, s2), h2, s1),
      _mm512_madd52lo_epu64(_mm512_madd52lo_epu64(_mm512_madd52lo_epu64(l1, h0, r1), h1, r0), h2, s2),
      _mm512_madd52lo_epu64(_mm512_madd52lo_epu64(_mm512_madd52lo_epu64(l2, h0, r2), h1, r1), h2, r0),
    ],
    hi: [
      _mm512_madd52hi_epu64(_mm512_madd52hi_epu64(_mm512_madd52hi_epu64(g0, h0, r0), h1, s2), h2, s1),
      _mm512_madd52hi_epu64(_mm512_madd52hi_epu64(_mm512_madd52hi_epu64(g1, h0, r1), h1, r0), h2, s2),
      _mm512_madd52hi_epu64(_mm512_madd52hi_epu64(_mm512_madd52hi_epu64(g2, h0, r2), h1, r1), h2, r0),
    ],
  }
}

/// Carries column sums into partially reduced limbs below 2^44, 2^44 + 2^9, and 2^42.
///
/// Columns from at most three [`accumulate`] calls keep every low half below 2^56 and every high half
/// below 2^45, so each 64-bit lane sum below stays far from overflow.
///
/// # Safety
///
/// Calling it outside an AVX-512F-enabled function requires AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
fn carry(c: Columns) -> Lanes {
  let [d0_lo, d1_lo, d2_lo] = c.lo;
  let [d0_hi, d1_hi, d2_hi] = c.hi;
  let mask44 = _mm512_set1_epi64(MASK44.cast_signed());
  let mask42 = _mm512_set1_epi64(MASK42.cast_signed());
  // Limb 0 keeps 44 bits; the rest of the column, `low >> 44` plus `high << 8`, carries on.
  let t0 = _mm512_and_si512(d0_lo, mask44);
  let c = _mm512_add_epi64(_mm512_srli_epi64::<44>(d0_lo), _mm512_slli_epi64::<8>(d0_hi));
  let d1 = _mm512_add_epi64(d1_lo, c);
  let t1 = _mm512_and_si512(d1, mask44);
  let c = _mm512_add_epi64(_mm512_srli_epi64::<44>(d1), _mm512_slli_epi64::<8>(d1_hi));
  let d2 = _mm512_add_epi64(d2_lo, c);
  let t2 = _mm512_and_si512(d2, mask42);
  // Limb 2 keeps 42 bits; the excess has weight 2^130 and folds into limb 0 times 5. The excess is
  // below 2^51, so limb 0 stays below 2^54 before its own carry.
  let c = _mm512_add_epi64(_mm512_srli_epi64::<42>(d2), _mm512_slli_epi64::<10>(d2_hi));
  let t0 = _mm512_add_epi64(t0, _mm512_add_epi64(_mm512_slli_epi64::<2>(c), c));
  let t1 = _mm512_add_epi64(t1, _mm512_srli_epi64::<44>(t0));
  Lanes([_mm512_and_si512(t0, mask44), t1, t2])
}

/// `h·r` in every lane, partially reduced, with the column sums and carries of [`Limbs::mul`].
///
/// `h` and `m.r` hold limbs below 2^45.
///
/// # Safety
///
/// Calling it outside an AVX-512F and AVX-512 IFMA function requires both features.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn multiply(h: Lanes, m: &Multiplier) -> Lanes {
  carry(accumulate(zero_columns(), h, m))
}

/// `h·r + x·q` in every lane with one carry pass, partially reduced.
///
/// All four operands hold limbs below 2^45. The `x·q` products come first because they do not
/// depend on the running accumulator.
///
/// # Safety
///
/// Calling it outside an AVX-512F and AVX-512 IFMA function requires both features.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn multiply2(h: Lanes, r: &Multiplier, x: Lanes, q: &Multiplier) -> Lanes {
  carry(accumulate(accumulate(zero_columns(), x, q), h, r))
}

/// `h·r + x·q + y·p` in every lane with one carry pass, partially reduced.
///
/// All six operands hold limbs below 2^45; three accumulations keep every low half below 2^56 and
/// every high half below 2^45.
///
/// # Safety
///
/// Calling it outside an AVX-512F and AVX-512 IFMA function requires both features.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn multiply3(h: Lanes, r: &Multiplier, x: Lanes, q: &Multiplier, y: Lanes, p: &Multiplier) -> Lanes {
  carry(accumulate(accumulate(accumulate(zero_columns(), y, p), x, q), h, r))
}

/// Splits block halves into limbs; `low` and `high` hold each lane's first and second eight bytes.
///
/// `hibit` is `2^40` in each lane that holds a full block and zero elsewhere.
///
/// # Safety
///
/// Calling it outside an AVX-512F-enabled function requires AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
fn split(low: __m512i, high: __m512i, hibit: __m512i) -> Lanes {
  let mask44 = _mm512_set1_epi64(MASK44.cast_signed());
  Lanes([
    _mm512_and_si512(low, mask44),
    _mm512_and_si512(
      _mm512_or_si512(_mm512_srli_epi64::<44>(low), _mm512_slli_epi64::<20>(high)),
      mask44,
    ),
    _mm512_or_si512(_mm512_srli_epi64::<24>(high), hibit),
  ])
}

/// Even qwords are each block's first half, odd qwords its second half.
///
/// # Safety
///
/// Calling it outside an AVX-512F-enabled function requires AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
fn deinterleave(first: __m512i, second: __m512i) -> (__m512i, __m512i) {
  (
    _mm512_permutex2var_epi64(first, _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), second),
    _mm512_permutex2var_epi64(first, _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15), second),
  )
}

/// Message limbs of eight consecutive blocks, lane `j` holding block `j`.
///
/// # Safety
///
/// Calling it outside an AVX-512F-enabled function requires AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
fn load8(blocks: &[[u8; 16]; 8]) -> Lanes {
  let bytes = blocks.as_flattened();
  // SAFETY: `bytes` is 128 initialized bytes; the two unaligned loads read exactly them.
  let (first, second) = unsafe {
    (
      _mm512_loadu_si512(bytes.as_ptr().cast()),
      _mm512_loadu_si512(bytes.as_ptr().add(64).cast()),
    )
  };
  let (low, high) = deinterleave(first, second);
  split(low, high, _mm512_set1_epi64(1 << 40))
}

/// Message limbs of one to eight consecutive blocks; lanes past the last block are zero.
///
/// # Safety
///
/// Calling it outside an AVX-512F-enabled function requires AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
fn load_partial(blocks: &[[u8; 16]]) -> Lanes {
  let count = blocks.len();
  debug_assert!((1..=8).contains(&count));
  #[expect(clippy::cast_ptr_alignment, reason = "the masked loads below are unaligned")]
  let qwords = blocks.as_flattened().as_ptr().cast::<i64>();
  let words = count.strict_mul(2);
  let first_mask = low_bits_mask(words.min(8));
  // SAFETY: the mask enables only qwords below `2·count`, all within `blocks`; masked-off lanes are not
  // accessed and read as zero.
  let first = unsafe { _mm512_maskz_loadu_epi64(first_mask, qwords) };
  let second = if words > 8 {
    // SAFETY: qword 8 lies within `blocks` because it holds more than four blocks, and the mask enables
    // only qwords below `2·count`.
    unsafe { _mm512_maskz_loadu_epi64(low_bits_mask(words.strict_sub(8)), qwords.add(8)) }
  } else {
    _mm512_setzero_si512()
  };
  let (low, high) = deinterleave(first, second);
  split(low, high, _mm512_maskz_set1_epi64(low_bits_mask(count), 1 << 40))
}

/// A mask of the low `count` lanes, `count <= 8`.
#[inline(always)]
fn low_bits_mask(count: usize) -> __mmask8 {
  debug_assert!(count <= 8);
  let all: u16 = 0x00ff;
  let [mask, _] = (all >> 8usize.strict_sub(count)).to_le_bytes();
  mask
}

/// Lane 0 holds `limbs`; the other lanes are zero.
///
/// # Safety
///
/// Calling it outside an AVX-512F-enabled function requires AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
fn lane0(limbs: Limbs) -> Lanes {
  let [l0, l1, l2] = limbs.0;
  Lanes([
    _mm512_maskz_set1_epi64(1, l0.cast_signed()),
    _mm512_maskz_set1_epi64(1, l1.cast_signed()),
    _mm512_maskz_set1_epi64(1, l2.cast_signed()),
  ])
}

/// Lane `j` holds limb `index` of `values[j]`.
///
/// # Safety
///
/// Calling it outside an AVX-512F-enabled function requires AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
fn column(values: &[Limbs; 8], index: usize) -> __m512i {
  let [v0, v1, v2, v3, v4, v5, v6, v7] = values.map(|value| value.0[index].cast_signed());
  _mm512_setr_epi64(v0, v1, v2, v3, v4, v5, v6, v7)
}

/// Lane `j` holds `values[j]`.
///
/// # Safety
///
/// Calling it outside an AVX-512F-enabled function requires AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
fn per_lane(values: &[Limbs; 8]) -> Lanes {
  Lanes([column(values, 0), column(values, 1), column(values, 2)])
}

/// `limbs` in every lane.
///
/// # Safety
///
/// Calling it outside an AVX-512F-enabled function requires AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
fn broadcast(limbs: Limbs) -> Lanes {
  let [l0, l1, l2] = limbs.0;
  Lanes([
    _mm512_set1_epi64(l0.cast_signed()),
    _mm512_set1_epi64(l1.cast_signed()),
    _mm512_set1_epi64(l2.cast_signed()),
  ])
}

/// Lane `j` takes lane `index[j]` of the sixteen lanes `high ++ low`.
///
/// # Safety
///
/// Calling it outside an AVX-512F-enabled function requires AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
fn select(high: Lanes, low: Lanes, index: __m512i) -> Lanes {
  Lanes([
    _mm512_permutex2var_epi64(high.0[0], index, low.0[0]),
    _mm512_permutex2var_epi64(high.0[1], index, low.0[1]),
    _mm512_permutex2var_epi64(high.0[2], index, low.0[2]),
  ])
}

/// # Safety
///
/// Calling it outside an AVX-512F-enabled function requires AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
fn add(x: Lanes, y: Lanes) -> Lanes {
  Lanes([
    _mm512_add_epi64(x.0[0], y.0[0]),
    _mm512_add_epi64(x.0[1], y.0[1]),
    _mm512_add_epi64(x.0[2], y.0[2]),
  ])
}

/// The sum over lanes of each limb column, carried into partially reduced form.
///
/// # Safety
///
/// Calling it outside an AVX-512F-enabled function requires AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
fn sum_lanes(lanes: Lanes) -> Limbs {
  // Eight lanes of limbs below 2^45 sum to below 2^48.
  Limbs([
    _mm512_reduce_add_epi64(lanes.0[0]).cast_unsigned(),
    _mm512_reduce_add_epi64(lanes.0[1]).cast_unsigned(),
    _mm512_reduce_add_epi64(lanes.0[2]).cast_unsigned(),
  ])
  .carry()
}

/// Lane `j` of the remainder pass multiplies by `r^(count - j)`; `table` lane `t` of `high ++ low` is
/// `r^(16 - t)`.
///
/// # Safety
///
/// Calling it outside an AVX-512F-enabled function requires AVX-512F.
#[inline]
#[target_feature(enable = "avx512f")]
fn remainder_index(count: usize, offset: usize) -> __m512i {
  let base = i64::try_from(16usize.strict_sub(count).strict_add(offset)).unwrap_or_default();
  _mm512_add_epi64(_mm512_set1_epi64(base), _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7))
}

/// Clears stack copies of key powers. Copies held only in vector registers are not cleared.
fn clear_powers(powers: &mut [Limbs]) {
  for limbs in powers {
    ct::zeroize_words_no_fence(&mut limbs.0);
  }
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

/// Lanes `(m_a + extra)·r^8 + m_b` for the two eight-block halves `m_a`, `m_b` of `blocks`.
///
/// # Safety
///
/// Calling it outside an AVX-512F and AVX-512 IFMA function requires both features.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn first_pass(blocks: &[[u8; 16]; 16], extra: Lanes, step8: &Multiplier) -> Lanes {
  let (halves, _) = blocks.as_chunks::<8>();
  add(multiply(add(load8(&halves[0]), extra), step8), load8(&halves[1]))
}

/// Lanes `h·r^k + m_a·r^8 + m_b` for the two eight-block halves of `blocks`, where `step` holds `r^k`.
///
/// # Safety
///
/// Calling it outside an AVX-512F and AVX-512 IFMA function requires both features.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn next_pass(h: Lanes, step: &Multiplier, blocks: &[[u8; 16]; 16], step8: &Multiplier) -> Lanes {
  let (halves, _) = blocks.as_chunks::<8>();
  add(multiply2(h, step, load8(&halves[0]), step8), load8(&halves[1]))
}

/// Fewest blocks for which the two-chain loop's extra power setup pays.
const TWO_CHAIN_MIN_BLOCKS: usize = 128;

/// Absorbs full blocks, each with its 2^128 bit, into `state`.
///
/// `fuse_final` folds the final per-lane multiply into the last single-chain pass. That removes one
/// multiply from the dependency chain but adds two power multiplies, so it pays on cores whose vector
/// adds and shifts have two-cycle latency and four issue ports, and costs on cores with one-cycle
/// latency and two 512-bit ports.
///
/// # Safety
///
/// The caller must establish AVX-512F and AVX-512 IFMA.
#[target_feature(enable = "avx512f,avx512ifma")]
pub(crate) unsafe fn absorb_blocks(state: &mut State, blocks: &[[u8; 16]], fuse_final: bool) {
  if blocks.is_empty() {
    return;
  }
  let (pairs, rest): (&[[[u8; 16]; 32]], &[[u8; 16]]) = if blocks.len() >= TWO_CHAIN_MIN_BLOCKS {
    blocks.as_chunks::<32>()
  } else {
    (&[], blocks)
  };
  let (sixteens, rest) = rest.as_chunks::<16>();
  let [r0, r1] = state.r;
  // powers[j] = r^(8-j); `squares` holds r^16 and r^32. Each is partially reduced. Products by the
  // clamped `r` take the shorter radix-2^64 scalar chain; squarings of unclamped powers take radix
  // 2^44.
  let mut powers = [Limbs::default(); 8];
  let mut squares = [Limbs::default(); 2];
  let r2 = state.times_r([r0, r1, 0]);
  let r3 = state.times_r(r2);
  let r4 = state.times_r(r3);
  let r5 = state.times_r(r4);
  let r6 = state.times_r(r5);
  for (power, words) in powers[2..].iter_mut().zip([r6, r5, r4, r3, r2, [r0, r1, 0]]) {
    let [w0, w1, w2] = words;
    *power = Limbs::from_words(w0, w1, w2);
  }
  let r8 = powers[4].mul(powers[4]);
  powers[0] = r8;
  powers[1] = powers[4].mul(powers[5]);
  // Lane j of `low` is r^(8-j) and of `high` r^(16-j); only the fused pass and remainders of more
  // than eight blocks need `high`.
  let low = per_lane(&powers);
  let step8 = Multiplier::new(broadcast(r8));
  let high = if (fuse_final && !sixteens.is_empty()) || rest.len() > 8 {
    multiply(low, &step8)
  } else {
    low
  };

  let [h0, h1, h2] = state.h;
  let mut h = Limbs::from_words(h0, h1, h2);
  if let Some((first, others)) = pairs.split_first() {
    // Two chains: chain `a` takes blocks 0-15 of each 32-block group and chain `b` blocks 16-31.
    // Chain `b`'s blocks sit sixteen positions later, so its lanes finish with r^(8-j) and chain
    // `a`'s with r^(24-j).
    let (groups, _) = first.as_chunks::<16>();
    let mut a = first_pass(&groups[0], lane0(h), &step8);
    let mut b = first_pass(&groups[1], Lanes([_mm512_setzero_si512(); 3]), &step8);
    squares[0] = r8.mul(r8);
    if !others.is_empty() {
      squares[1] = squares[0].mul(squares[0]);
      let step32 = Multiplier::new(broadcast(squares[1]));
      for pair in others {
        let (groups, _) = pair.as_chunks::<16>();
        a = next_pass(a, &step32, &groups[0], &step8);
        b = next_pass(b, &step32, &groups[1], &step8);
      }
    }
    let mid = multiply(low, &Multiplier::new(broadcast(squares[0])));
    h = sum_lanes(multiply2(a, &Multiplier::new(mid), b, &Multiplier::new(low)));
  }

  if !fuse_final && let Some((first, others)) = sixteens.split_first() {
    // One chain; the lanes finish with a separate multiply by r^(8-j).
    let mut lanes = first_pass(first, lane0(h), &step8);
    if !others.is_empty() {
      squares[0] = r8.mul(r8);
      let step16 = Multiplier::new(broadcast(squares[0]));
      for group in others {
        lanes = next_pass(lanes, &step16, group, &step8);
      }
    }
    h = sum_lanes(multiply(lanes, &Multiplier::new(low)));
  } else if let Some((last, leading)) = sixteens.split_last() {
    // One chain. The last group multiplies each lane by the powers that complete its weights:
    // r^(24-j) for the running product, r^(16-j) and r^(8-j) for the group's two blocks in lane j.
    let (halves, _) = last.as_chunks::<8>();
    let (first_half, second_half) = (load8(&halves[0]), load8(&halves[1]));
    h = if let Some((first, middle)) = leading.split_first() {
      let mut lanes = first_pass(first, lane0(h), &step8);
      squares[0] = r8.mul(r8);
      let step16 = Multiplier::new(broadcast(squares[0]));
      for group in middle {
        lanes = next_pass(lanes, &step16, group, &step8);
      }
      let top = multiply(low, &step16);
      sum_lanes(multiply3(
        lanes,
        &Multiplier::new(top),
        first_half,
        &Multiplier::new(high),
        second_half,
        &Multiplier::new(low),
      ))
    } else {
      sum_lanes(multiply2(
        add(first_half, lane0(h)),
        &Multiplier::new(high),
        second_half,
        &Multiplier::new(low),
      ))
    };
  }

  if !rest.is_empty() {
    let tail = rest.len();
    let (first, second) = rest.split_at(tail.min(8));
    let a = multiply(
      add(lane0(h), load_partial(first)),
      &Multiplier::new(select(high, low, remainder_index(tail, 0))),
    );
    let sum = if second.is_empty() {
      a
    } else {
      add(
        a,
        multiply(
          load_partial(second),
          &Multiplier::new(select(high, low, remainder_index(tail, 8))),
        ),
      )
    };
    h = sum_lanes(sum);
  }

  state.h = h.to_words();
  clear_powers(&mut powers);
  clear_powers(&mut squares);
}
