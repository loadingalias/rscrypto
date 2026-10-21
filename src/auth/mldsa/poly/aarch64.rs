//! Four-lane NTT transforms and challenge scans for the macOS and Linux AArch64 NEON baseline.
//!
//! Products and elementwise operations apply the portable canonical Montgomery arithmetic to
//! lanes. The transforms use signed Barrett multiplication by the plain residue of each portable
//! Montgomery constant, with lazily reduced lanes, and reduce to the same canonical residues as
//! the portable transforms; no lane layout or secret owner changes.

use core::arch::aarch64::*;

use super::{FIRST_FACTOR, INV_N, LAST_FACTOR, N, NEG_Q_INVERSE, Poly, Q, R2, ROOTS, public_montgomery};
use crate::auth::mldsa::sampling::{CHALLENGE_BYTES, CHALLENGE_WINDOWS, uniform_portable};

/// One masked subtraction for lanes in [0, 2q).
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn reduce(x: uint32x4_t) -> uint32x4_t {
  let q = vdupq_n_u32(Q);
  vsubq_u32(x, vandq_u32(vcgeq_u32(x, q), q))
}

/// Lane-wise portable Montgomery multiplication, with both inputs below 2q.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn multiply(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
  // The low product and correction are arithmetic modulo 2^32. The widened
  // products and their sum are below 2^56, as proved by super::montgomery.
  let m = vmulq_u32(vmulq_u32(a, b), vdupq_n_u32(NEG_Q_INVERSE));
  let lo = vmull_u32(vget_low_u32(a), vget_low_u32(b));
  let hi = vmull_high_u32(a, b);
  let lo = vmlal_n_u32(lo, vget_low_u32(m), Q);
  let hi = vmlal_high_n_u32(hi, m, Q);
  reduce(vcombine_u32(vshrn_n_u64::<32>(lo), vshrn_n_u64::<32>(hi)))
}

/// A constant multiplier for [`barrett_lane`]: the centered residue `value` in `(-q/2, q/2]`
/// and `quotient = round(value·2^31 / q)`.
#[derive(Clone, Copy)]
struct Barrett {
  value: i32,
  quotient: i32,
}

/// The Barrett multiplier for the Montgomery constant `montgomery`: multiplying by the plain
/// residue `montgomery·2^-32 mod q` equals the portable Montgomery multiplication modulo q.
#[expect(
  clippy::cast_possible_truncation,
  reason = "the rounded quotient has magnitude at most 2^30"
)]
const fn barrett(montgomery: u32) -> Barrett {
  let plain = public_montgomery(montgomery, 1);
  let value = if plain > Q / 2 {
    plain.cast_signed().wrapping_sub(Q.cast_signed())
  } else {
    plain.cast_signed()
  };
  let q = Q as i64;
  let half = if value < 0 { (q / 2).strict_neg() } else { q / 2 };
  let quotient = (value as i64).strict_mul(1 << 31).strict_add(half).strict_div(q);
  Barrett {
    value,
    quotient: quotient as i32,
  }
}

/// Four Barrett multipliers in lane order, loaded as `(values, quotients)` vectors.
#[derive(Clone, Copy)]
struct Lanes {
  values: [i32; 4],
  quotients: [i32; 4],
}

const fn lanes(constants: [u32; 4]) -> Lanes {
  let mut lanes = Lanes {
    values: [0; 4],
    quotients: [0; 4],
  };
  let mut lane = 0usize;
  while lane < 4 {
    let multiplier = barrett(constants[lane]);
    lanes.values[lane] = multiplier.value;
    lanes.quotients[lane] = multiplier.quotient;
    lane = lane.strict_add(1);
  }
  lanes
}

/// `ROOTS[k]` for the forward transform, `q - ROOTS[k]` for the inverse.
const fn root(k: usize, inverse: bool) -> u32 {
  if inverse { Q.strict_sub(ROOTS[k]) } else { ROOTS[k] }
}

/// Width 128, 64 and 32 multipliers: the Montgomery conversion factors `R2` and
/// `M(R2, ζ1)` followed by roots 2..8.
const FORWARD_OUTER: [Lanes; 2] = [
  lanes([R2, FIRST_FACTOR, ROOTS[2], ROOTS[3]]),
  lanes([ROOTS[4], ROOTS[5], ROOTS[6], ROOTS[7]]),
];

/// Inverse width 32 and 64 multipliers, then the normalizing factors `n⁻¹` and
/// `M(q - ζ1, n⁻¹)` of width 128.
const INVERSE_OUTER: [Lanes; 2] = [
  lanes([root(7, true), root(6, true), root(5, true), root(4, true)]),
  lanes([root(3, true), root(2, true), INV_N, LAST_FACTOR]),
];

/// Multipliers for one 32-coefficient block of the block pass.
#[derive(Clone, Copy)]
struct Block {
  /// Forward: the width 16 root, both width 8 roots and padding. Inverse: both width 8
  /// roots, the width 16 root and padding.
  wide: Lanes,
  /// The four width 4 roots in pass order.
  narrow: Lanes,
  /// Per eight-coefficient pair: width 2 roots, lanes 0-1 the first and lanes 2-3 the next.
  width_2: [Lanes; 4],
  /// Per eight-coefficient pair: lane i holds the i-th width 1 root.
  width_1: [Lanes; 4],
}

const FORWARD_BLOCKS: [Block; 8] = blocks(false);
const INVERSE_BLOCKS: [Block; 8] = blocks(true);

const fn blocks(inverse: bool) -> [Block; 8] {
  let empty = lanes([0; 4]);
  let mut table = [Block {
    wide: empty,
    narrow: empty,
    width_2: [empty; 4],
    width_1: [empty; 4],
  }; 8];
  let mut w = 0usize;
  while w < 8 {
    let half = w.strict_mul(2);
    let quarter = w.strict_mul(4);
    if inverse {
      table[w].wide = lanes([
        root(31usize.strict_sub(half), true),
        root(30usize.strict_sub(half), true),
        root(15usize.strict_sub(w), true),
        0,
      ]);
      table[w].narrow = lanes([
        root(63usize.strict_sub(quarter), true),
        root(62usize.strict_sub(quarter), true),
        root(61usize.strict_sub(quarter), true),
        root(60usize.strict_sub(quarter), true),
      ]);
    } else {
      table[w].wide = lanes([
        ROOTS[8usize.strict_add(w)],
        ROOTS[16usize.strict_add(half)],
        ROOTS[17usize.strict_add(half)],
        0,
      ]);
      table[w].narrow = lanes([
        ROOTS[32usize.strict_add(quarter)],
        ROOTS[33usize.strict_add(quarter)],
        ROOTS[34usize.strict_add(quarter)],
        ROOTS[35usize.strict_add(quarter)],
      ]);
    }
    let mut pair = 0usize;
    while pair < 4 {
      let block = quarter.strict_add(pair);
      let mut width_2 = [0u32; 4];
      let mut width_1 = [0u32; 4];
      let mut lane = 0usize;
      while lane < 4 {
        let (two, one) = if inverse {
          (
            127usize.strict_sub(block.strict_mul(2)).strict_sub(lane / 2),
            255usize.strict_sub(block.strict_mul(4)).strict_sub(lane),
          )
        } else {
          (
            64usize.strict_add(block.strict_mul(2)).strict_add(lane / 2),
            128usize.strict_add(block.strict_mul(4)).strict_add(lane),
          )
        };
        width_2[lane] = root(two, inverse);
        width_1[lane] = root(one, inverse);
        lane = lane.strict_add(1);
      }
      table[w].width_2[pair] = lanes(width_2);
      table[w].width_1[pair] = lanes(width_1);
      pair = pair.strict_add(1);
    }
    w = w.strict_add(1);
  }
  table
}

/// Load a lane group's `(values, quotients)`.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn load_lanes(lanes: &Lanes) -> (int32x4_t, int32x4_t) {
  // SAFETY: Each array is four initialized i32 values; the loads read exactly them.
  unsafe { (vld1q_s32(lanes.values.as_ptr()), vld1q_s32(lanes.quotients.as_ptr())) }
}

/// Barrett multiplication `a·c mod q` lane by lane, with `c` given per lane.
///
/// `t = round(a·quotient / 2^31)` differs from `a·c / q` by at most `1/2 + |a|/2^32`, so
/// `a·c - t·q` lies in `(-q, q)` for `|a| < 2^31` and its low 32 bits are exact. The rounding
/// doubled multiply saturates only for two `i32::MIN` operands; quotients are at most 2^30.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn barrett_mul(a: int32x4_t, (values, quotients): (int32x4_t, int32x4_t)) -> int32x4_t {
  let t = vqrdmulhq_s32(a, quotients);
  vmlsq_s32(vmulq_s32(a, values), t, vdupq_n_s32(Q.cast_signed()))
}

/// [`barrett_mul`] by lane `LANE` of `(values, quotients)` in every lane.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn barrett_lane<const LANE: i32>(a: int32x4_t, (values, quotients): (int32x4_t, int32x4_t)) -> int32x4_t {
  let t = vqrdmulhq_laneq_s32::<LANE>(a, quotients);
  vmlsq_s32(vmulq_laneq_s32::<LANE>(a, values), t, vdupq_n_s32(Q.cast_signed()))
}

/// Signed reduction `x - ⌊(x + 2^22) / 2^23⌋·q`, below `0.52·q` for lanes below `16·q` and
/// inside `(-q, q)` for lanes below `9·q`; `t·q` is shifts because `q = 2^23 - 2^13 + 1`.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn reduce32(x: int32x4_t) -> int32x4_t {
  let t = vshrq_n_s32::<23>(vaddq_s32(x, vdupq_n_s32(1 << 22)));
  let tq = vaddq_s32(vsubq_s32(vshlq_n_s32::<23>(t), vshlq_n_s32::<13>(t)), t);
  vsubq_s32(x, tq)
}

/// The canonical residue in `[0, q)` of a lane below `9·q` in magnitude.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn canonical(x: int32x4_t) -> uint32x4_t {
  let r = reduce32(x);
  vreinterpretq_u32_s32(vaddq_s32(
    r,
    vandq_s32(vshrq_n_s32::<31>(r), vdupq_n_s32(Q.cast_signed())),
  ))
}

/// The canonical residue in `[0, q)` of a lane in `(-q, q)`, such as a Barrett product.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn canonical_product(x: int32x4_t) -> uint32x4_t {
  vreinterpretq_u32_s32(vaddq_s32(
    x,
    vandq_s32(vshrq_n_s32::<31>(x), vdupq_n_s32(Q.cast_signed())),
  ))
}

/// Cooley–Tukey butterfly `(x[a], x[b]) -> (x[a] + x[b]·ζ, x[a] - x[b]·ζ)` by lane `LANE`
/// of `zetas`, without reduction.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn forward<const LANE: i32>(x: &mut [int32x4_t; 8], a: usize, b: usize, zetas: (int32x4_t, int32x4_t)) {
  let t = barrett_lane::<LANE>(x[b], zetas);
  x[b] = vsubq_s32(x[a], t);
  x[a] = vaddq_s32(x[a], t);
}

/// Gentleman–Sande butterfly `(x[a], x[b]) -> (x[a] + x[b], (x[a] - x[b])·ζ)` by lane `LANE`
/// of `zetas`, without reducing the sum.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn inverse<const LANE: i32>(x: &mut [int32x4_t; 8], a: usize, b: usize, zetas: (int32x4_t, int32x4_t)) {
  let difference = vsubq_s32(x[a], x[b]);
  x[a] = vaddq_s32(x[a], x[b]);
  x[b] = barrett_lane::<LANE>(difference, zetas);
}

/// Load the eight vectors `first + i·stride` of the polynomial.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn load_eight(chunks: &[[u32; 4]], first: usize, stride: usize) -> [int32x4_t; 8] {
  let mut out = [vdupq_n_s32(0); 8];
  for (i, lanes) in (0usize..).zip(out.iter_mut()) {
    // SAFETY: Each chunk is four initialized u32 lanes; the load reads exactly them.
    *lanes = vreinterpretq_s32_u32(unsafe { vld1q_u32(chunks[first.strict_add(i.strict_mul(stride))].as_ptr()) });
  }
  out
}

/// Store eight vectors to `first + i·stride`.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn store_eight(chunks: &mut [[u32; 4]], first: usize, stride: usize, values: &[uint32x4_t; 8]) {
  for (i, &lanes) in (0usize..).zip(values.iter()) {
    // SAFETY: Each chunk is four uniquely borrowed u32 lanes; the store writes exactly them.
    unsafe { vst1q_u32(chunks[first.strict_add(i.strict_mul(stride))].as_mut_ptr(), lanes) };
  }
}

/// Reinterpret eight signed vectors as unsigned.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn unsigned_eight(values: &[int32x4_t; 8]) -> [uint32x4_t; 8] {
  let mut out = [vdupq_n_u32(0); 8];
  for (result, &lanes) in out.iter_mut().zip(values) {
    *result = vreinterpretq_u32_s32(lanes);
  }
  out
}

/// FIPS 204 Algorithm 41 with conversion to Montgomery form, equal to the portable transform.
///
/// Every multiplication is a Barrett multiplication by the plain residue that equals the
/// portable Montgomery multiplier, so each stage computes the same values modulo q. Lanes are
/// signed and lazily reduced: each product lies in `(-q, q)` and each stage adds at most `q`,
/// so every lane stays below `9·q` and one final reduction returns canonical residues. The
/// first pass runs widths 128, 64 and 32 over eight vectors 32 coefficients apart; the second
/// runs widths 16, 8 and 4 across one 32-coefficient block and widths 2 and 1 inside vector
/// pairs. Each pass keeps eight coefficient vectors and its multipliers in registers.
///
/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(super) unsafe fn ntt(poly: &mut Poly) {
  let chunks = poly.0.as_chunks_mut::<4>().0;
  let outer = load_lanes(&FORWARD_OUTER[0]);
  let roots = load_lanes(&FORWARD_OUTER[1]);
  for v in 0..8usize {
    let mut x = load_eight(chunks, v, 8);
    // Width 128, fused with the Montgomery conversion: M(a, R2) and M(b, M(R2, ζ1)).
    for i in 0..4usize {
      let left = barrett_lane::<0>(x[i], outer);
      let right = barrett_lane::<1>(x[i.strict_add(4)], outer);
      x[i] = vaddq_s32(left, right);
      x[i.strict_add(4)] = vsubq_s32(left, right);
    }
    forward::<2>(&mut x, 0, 2, outer);
    forward::<2>(&mut x, 1, 3, outer);
    forward::<3>(&mut x, 4, 6, outer);
    forward::<3>(&mut x, 5, 7, outer);
    forward::<0>(&mut x, 0, 1, roots);
    forward::<1>(&mut x, 2, 3, roots);
    forward::<2>(&mut x, 4, 5, roots);
    forward::<3>(&mut x, 6, 7, roots);
    store_eight(chunks, v, 8, &unsigned_eight(&x));
  }
  for (w, block) in FORWARD_BLOCKS.iter().enumerate() {
    let mut x = load_eight(chunks, w.strict_mul(8), 1);
    let wide = load_lanes(&block.wide);
    let narrow = load_lanes(&block.narrow);
    for i in 0..4usize {
      forward::<0>(&mut x, i, i.strict_add(4), wide);
    }
    forward::<1>(&mut x, 0, 2, wide);
    forward::<1>(&mut x, 1, 3, wide);
    forward::<2>(&mut x, 4, 6, wide);
    forward::<2>(&mut x, 5, 7, wide);
    forward::<0>(&mut x, 0, 1, narrow);
    forward::<1>(&mut x, 2, 3, narrow);
    forward::<2>(&mut x, 4, 5, narrow);
    forward::<3>(&mut x, 6, 7, narrow);
    let [a0, a1] = forward_pair(x[0], x[1], &block.width_2[0], &block.width_1[0]);
    let [b0, b1] = forward_pair(x[2], x[3], &block.width_2[1], &block.width_1[1]);
    let [c0, c1] = forward_pair(x[4], x[5], &block.width_2[2], &block.width_1[2]);
    let [d0, d1] = forward_pair(x[6], x[7], &block.width_2[3], &block.width_1[3]);
    store_eight(chunks, w.strict_mul(8), 1, &[a0, a1, b0, b1, c0, c1, d0, d1]);
  }
}

/// Widths 2 and 1 inside one eight-coefficient pair of vectors, with canonical output in
/// coefficient order.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn forward_pair(first: int32x4_t, second: int32x4_t, width_2: &Lanes, width_1: &Lanes) -> [uint32x4_t; 2] {
  let a = vreinterpretq_s64_s32(first);
  let b = vreinterpretq_s64_s32(second);
  let left = vreinterpretq_s32_s64(vzip1q_s64(a, b));
  let right = barrett_mul(vreinterpretq_s32_s64(vzip2q_s64(a, b)), load_lanes(width_2));
  let sums = vaddq_s32(left, right);
  let differences = vsubq_s32(left, right);
  let left = vtrn1q_s32(sums, differences);
  let right = barrett_mul(vtrn2q_s32(sums, differences), load_lanes(width_1));
  // Interleave the width-1 outputs back into coefficient order.
  let sums = canonical(vaddq_s32(left, right));
  let differences = canonical(vsubq_s32(left, right));
  [vzip1q_u32(sums, differences), vzip2q_u32(sums, differences)]
}

/// Widths 1 and 2 inside one eight-coefficient pair of vectors, leaving the lanes in
/// coefficient order for the width 4 stage.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn inverse_pair(first: int32x4_t, second: int32x4_t, width_2: &Lanes, width_1: &Lanes) -> [int32x4_t; 2] {
  let even = vuzp1q_s32(first, second);
  let odd = vuzp2q_s32(first, second);
  let sums = vaddq_s32(even, odd);
  let products = barrett_mul(vsubq_s32(even, odd), load_lanes(width_1));
  let left = vtrn1q_s32(sums, products);
  let right = vtrn2q_s32(sums, products);
  let sums = vreinterpretq_s64_s32(vaddq_s32(left, right));
  let products = vreinterpretq_s64_s32(barrett_mul(vsubq_s32(left, right), load_lanes(width_2)));
  [
    vreinterpretq_s32_s64(vzip1q_s64(sums, products)),
    vreinterpretq_s32_s64(vzip2q_s64(sums, products)),
  ]
}

/// FIPS 204 Algorithm 42 with conversion out of Montgomery form, equal to the portable
/// transform.
///
/// The passes mirror [`ntt`], with the same Barrett multipliers. Sums double at each stage,
/// so after width 8 the sum vectors (below `16·q`) are reduced and every lane lies below
/// `0.53·q`; the remaining stages stay below `8.5·q`, and the last stage multiplies both
/// outputs by `n⁻¹`-scaled constants, leaving `(-q, q)` for the final conditional addition.
///
/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(super) unsafe fn inverse_ntt(poly: &mut Poly) {
  let chunks = poly.0.as_chunks_mut::<4>().0;
  for (w, block) in INVERSE_BLOCKS.iter().enumerate() {
    let loaded = load_eight(chunks, w.strict_mul(8), 1);
    let [a0, a1] = inverse_pair(loaded[0], loaded[1], &block.width_2[0], &block.width_1[0]);
    let [b0, b1] = inverse_pair(loaded[2], loaded[3], &block.width_2[1], &block.width_1[1]);
    let [c0, c1] = inverse_pair(loaded[4], loaded[5], &block.width_2[2], &block.width_1[2]);
    let [d0, d1] = inverse_pair(loaded[6], loaded[7], &block.width_2[3], &block.width_1[3]);
    let mut x = [a0, a1, b0, b1, c0, c1, d0, d1];
    let wide = load_lanes(&block.wide);
    let narrow = load_lanes(&block.narrow);
    inverse::<0>(&mut x, 0, 1, narrow);
    inverse::<1>(&mut x, 2, 3, narrow);
    inverse::<2>(&mut x, 4, 5, narrow);
    inverse::<3>(&mut x, 6, 7, narrow);
    inverse::<0>(&mut x, 0, 2, wide);
    inverse::<0>(&mut x, 1, 3, wide);
    inverse::<1>(&mut x, 4, 6, wide);
    inverse::<1>(&mut x, 5, 7, wide);
    // After width 8, vectors 0 and 4 hold sums below 16·q and vectors 1 and 5 sums below 2·q;
    // the others are fresh products below q. Reducing the sums leaves every lane below 0.53·q.
    for i in [0usize, 1, 4, 5] {
      x[i] = reduce32(x[i]);
    }
    for i in 0..4usize {
      inverse::<2>(&mut x, i, i.strict_add(4), wide);
    }
    store_eight(chunks, w.strict_mul(8), 1, &unsigned_eight(&x));
  }
  let roots = load_lanes(&INVERSE_OUTER[0]);
  let outer = load_lanes(&INVERSE_OUTER[1]);
  for v in 0..8usize {
    let mut x = load_eight(chunks, v, 8);
    inverse::<0>(&mut x, 0, 1, roots);
    inverse::<1>(&mut x, 2, 3, roots);
    inverse::<2>(&mut x, 4, 5, roots);
    inverse::<3>(&mut x, 6, 7, roots);
    inverse::<0>(&mut x, 0, 2, outer);
    inverse::<0>(&mut x, 1, 3, outer);
    inverse::<1>(&mut x, 4, 6, outer);
    inverse::<1>(&mut x, 5, 7, outer);
    // Width 128, fused with normalization: M(a + b, n⁻¹) and M(a - b, M(q - ζ1, n⁻¹)).
    // Both inputs are below 17·q in magnitude, so each product already lies in (-q, q).
    let mut out = [vdupq_n_u32(0); 8];
    for i in 0..4usize {
      let (a, b) = (x[i], x[i.strict_add(4)]);
      out[i] = canonical_product(barrett_lane::<2>(vaddq_s32(a, b), outer));
      out[i.strict_add(4)] = canonical_product(barrett_lane::<3>(vsubq_s32(a, b), outer));
    }
    store_eight(chunks, v, 8, &out);
  }
}

/// Write the product of canonical Montgomery residues as canonical output.
///
/// # Safety
/// The caller must establish NEON support. Every coefficient must be below q.
#[target_feature(enable = "neon")]
pub(super) unsafe fn product(out: &mut [u32; N], a: &[u32; N], b: &[u32; N]) {
  for ((out, a), b) in out
    .as_chunks_mut::<4>()
    .0
    .iter_mut()
    .zip(a.as_chunks::<4>().0)
    .zip(b.as_chunks::<4>().0)
  {
    // SAFETY: Each reference covers four initialized u32 lanes; NEON loads
    // need only u32 alignment. The mutable output cannot alias either input.
    // N is divisible by four, so there is no tail. All offsets are public.
    unsafe { vst1q_u32(out.as_mut_ptr(), multiply(vld1q_u32(a.as_ptr()), vld1q_u32(b.as_ptr()))) };
  }
}

/// Write `Σ a[j]·b[j]·2^-32 mod q` for canonical residues, one Montgomery reduction per
/// coefficient.
///
/// Each 64-bit lane sum is below `7·q² < 2^49`, so adding `m·q < 2^55` cannot overflow and the
/// reduced value is below `2^17 + q < 2q`. The result equals the sum of the per-column
/// Montgomery products modulo q, so the canonical output matches [`accumulate_product`] over
/// the same columns.
///
/// # Safety
/// The caller must establish NEON support. Every coefficient must be below q.
#[target_feature(enable = "neon")]
pub(super) unsafe fn row_product<const L: usize>(out: &mut [u32; N], a: &[[u32; N]; L], b: &[Poly; L]) {
  const { assert!(L <= 7) };
  let q = vdupq_n_u32(Q);
  let negative_inverse = vdupq_n_u32(NEG_Q_INVERSE);
  let a: [&[[u32; 4]]; L] = core::array::from_fn(|j| a[j].as_chunks::<4>().0);
  let b: [&[[u32; 4]]; L] = core::array::from_fn(|j| b[j].0.as_chunks::<4>().0);
  for (lane, out) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
    let mut low = vdupq_n_u64(0);
    let mut high = vdupq_n_u64(0);
    for (a, b) in a.iter().zip(&b) {
      // SAFETY: Each chunk is four initialized u32 lanes; the loads read exactly them.
      let (x, y) = unsafe { (vld1q_u32(a[lane].as_ptr()), vld1q_u32(b[lane].as_ptr())) };
      low = vmlal_u32(low, vget_low_u32(x), vget_low_u32(y));
      high = vmlal_high_u32(high, x, y);
    }
    let m = vmulq_u32(vcombine_u32(vmovn_u64(low), vmovn_u64(high)), negative_inverse);
    let low = vmlal_u32(low, vget_low_u32(m), vget_low_u32(q));
    let high = vmlal_high_u32(high, m, q);
    let reduced = reduce(vcombine_u32(vshrn_n_u64::<32>(low), vshrn_n_u64::<32>(high)));
    // SAFETY: `out` is four uniquely borrowed u32 lanes; the store writes exactly them.
    unsafe { vst1q_u32(out.as_mut_ptr(), reduced) };
  }
}

/// Add a product of canonical Montgomery residues to canonical output.
///
/// # Safety
/// The caller must establish NEON support. Every coefficient must be below q.
#[inline(never)]
#[target_feature(enable = "neon")]
pub(super) unsafe fn accumulate_product(out: &mut [u32; N], a: &[u32; N], b: &[u32; N]) {
  for ((out, a), b) in out
    .as_chunks_mut::<4>()
    .0
    .iter_mut()
    .zip(a.as_chunks::<4>().0)
    .zip(b.as_chunks::<4>().0)
  {
    // SAFETY: Each reference covers four initialized u32 lanes; NEON loads
    // need only u32 alignment. The mutable output cannot alias either input.
    // N is divisible by four, so there is no tail. All offsets are public.
    unsafe {
      let product = multiply(vld1q_u32(a.as_ptr()), vld1q_u32(b.as_ptr()));
      let result = reduce(vaddq_u32(vld1q_u32(out.as_ptr()), product));
      vst1q_u32(out.as_mut_ptr(), result);
    }
  }
}

/// Append the Montgomery form of each three-byte candidate below q, in order, until `out`
/// holds N coefficients, as `sampling::uniform_portable` does, eight candidates per step.
///
/// Matrix candidates come from the public rho seed. A group with a rejected candidate, or one
/// that would pass N, takes the scalar authority, which applies the same order and bound.
///
/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(in crate::auth::mldsa) unsafe fn uniform(bytes: &[u8], out: &mut [u32; N], mut count: usize) -> usize {
  let (groups, tail) = bytes.as_chunks::<24>();
  let q = vdupq_n_u32(Q);
  let r2 = vdupq_n_u32(R2);
  for (index, group) in groups.iter().enumerate() {
    if count > N.strict_sub(8) {
      return uniform_portable(&bytes[index.strict_mul(24)..], out, count);
    }
    // SAFETY: `group` is 24 initialized bytes; the de-interleaving load reads exactly them.
    let planes = unsafe { vld3_u8(group.as_ptr()) };
    // Little-endian 23-bit candidates: byte 0, byte 1 and the low seven bits of byte 2.
    let low = vorrq_u16(vmovl_u8(planes.0), vshlq_n_u16::<8>(vmovl_u8(planes.1)));
    let high = vmovl_u8(vand_u8(planes.2, vdup_n_u8(0x7f)));
    let first = vreinterpretq_u32_u16(vzip1q_u16(low, high));
    let second = vreinterpretq_u32_u16(vzip2q_u16(low, high));
    if vminvq_u32(vandq_u32(vcltq_u32(first, q), vcltq_u32(second, q))) == u32::MAX {
      let (slots, _) = out[count..count.strict_add(8)].as_chunks_mut::<4>();
      // SAFETY: Each slot is four uniquely borrowed u32 lanes; each store writes exactly them.
      // Accepted candidates and R2 are below q, as `multiply` requires.
      unsafe {
        vst1q_u32(slots[0].as_mut_ptr(), multiply(first, r2));
        vst1q_u32(slots[1].as_mut_ptr(), multiply(second, r2));
      }
      count = count.strict_add(8);
    } else {
      count = uniform_portable(group, out, count);
    }
  }
  uniform_portable(tail, out, count)
}

/// Fixed-work search for the first eligible challenge byte.
///
/// Returns the minimum of `position << 8 | byte` over positions at or after
/// `offset` whose byte is at most `limit`, or `u32::MAX` when none qualifies,
/// as `sampling::first_challenge_candidate` does. Every challenge position fits
/// a byte, so one scan finds the first eligible position and a second extracts
/// its byte. Every window is read twice, and neither address nor control flow
/// depends on a byte, offset or limit.
///
/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(in crate::auth::mldsa) unsafe fn challenge_first_candidate(
  bytes: &[u8; CHALLENGE_BYTES],
  offset: u32,
  limit: u32,
) -> u32 {
  const { assert!(CHALLENGE_BYTES <= 255) };
  // Offsets are at most 221 and limits at most 255. Masking first makes each
  // conversion provably in range, so no check branches on the secret offset.
  let offset = vdupq_n_u8(u8::try_from(offset & 0xff).expect("masked offsets fit u8"));
  let limit = vdupq_n_u8(u8::try_from(limit & 0xff).expect("masked limits fit u8"));
  let lanes: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
  // SAFETY: The load reads exactly the sixteen initialized lanes of `lanes`.
  let lanes = unsafe { vld1q_u8(lanes.as_ptr()) };
  let mut windows = [(vdupq_n_u8(0), vdupq_n_u8(0)); CHALLENGE_WINDOWS.len()];
  // No position reaches 0xff, so a minimum of 0xff means that no byte qualifies.
  let mut first = vdupq_n_u8(u8::MAX);
  for (start, window) in CHALLENGE_WINDOWS.into_iter().zip(&mut windows) {
    let candidates: &[u8; 16] = bytes[start..start.strict_add(16)]
      .try_into()
      .expect("challenge windows have sixteen bytes");
    // SAFETY: The load reads exactly the sixteen initialized bytes of `candidates`.
    let candidates = unsafe { vld1q_u8(candidates.as_ptr()) };
    let positions = vaddq_u8(
      vdupq_n_u8(u8::try_from(start).expect("challenge positions fit u8")),
      lanes,
    );
    let eligible = vandq_u8(vcgeq_u8(positions, offset), vcleq_u8(candidates, limit));
    first = vminq_u8(first, vbslq_u8(eligible, positions, vdupq_n_u8(u8::MAX)));
    *window = (positions, candidates);
  }
  let position = vminvq_u8(first);
  let target = vdupq_n_u8(position);
  // Overlapping windows repeat a position with the same byte, so OR keeps it exact.
  let mut found = vdupq_n_u8(0);
  for (positions, candidates) in windows {
    found = vorrq_u8(found, vandq_u8(candidates, vceqq_u8(positions, target)));
  }
  let position = u32::from(position);
  let first = (position << 8) | u32::from(vmaxvq_u8(found));
  // Eligible positions are at most 220, so a minimum position of 0xff means none.
  let none = (position ^ 0xff).wrapping_sub(1) >> 31;
  first | super::opaque_mask(0u32.wrapping_sub(none))
}

/// Write `sign` at byte coefficient `j` and return the byte it replaces.
///
/// Every coefficient byte is read and written; the index comparison selects lanes
/// without a secret-dependent address, as the portable placement scan does.
///
/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(in crate::auth::mldsa) unsafe fn challenge_place_bytes(coefficients: &mut [u8; N], j: u32, sign: u8) -> u8 {
  // Placement indices are below 256; masking makes the conversion branch-free.
  let target = vdupq_n_u8(u8::try_from(j & 0xff).expect("masked indices fit u8"));
  let sign = vdupq_n_u8(sign);
  let lanes: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
  // SAFETY: The load reads exactly the sixteen initialized lanes of `lanes`.
  let mut index = unsafe { vld1q_u8(lanes.as_ptr()) };
  let mut previous = vdupq_n_u8(0);
  for chunk in coefficients.as_chunks_mut::<16>().0 {
    // SAFETY: Each load and store covers exactly one uniquely borrowed sixteen-byte chunk.
    // N has no vector tail.
    unsafe {
      let value = vld1q_u8(chunk.as_ptr());
      let hit = vceqq_u8(index, target);
      previous = vorrq_u8(previous, vandq_u8(value, hit));
      vst1q_u8(chunk.as_mut_ptr(), vbslq_u8(hit, sign, value));
    }
    index = vaddq_u8(index, vdupq_n_u8(16));
  }
  // At most one lane holds the replaced byte; every other lane is zero.
  vmaxvq_u8(previous)
}

/// Map challenge bytes `0`, `1` and `0xff` to the canonical residues `0`, `1` and `q - 1`.
///
/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(in crate::auth::mldsa) unsafe fn challenge_expand(bytes: &[u8; N], out: &mut [u32; N]) {
  let q = vdupq_n_s32(Q.cast_signed());
  for (chunk, out) in bytes.as_chunks::<16>().0.iter().zip(out.as_chunks_mut::<16>().0) {
    // SAFETY: The load reads exactly the sixteen initialized bytes of `chunk`.
    let signed = vreinterpretq_s8_u8(unsafe { vld1q_u8(chunk.as_ptr()) });
    let halves = [vmovl_s8(vget_low_s8(signed)), vmovl_high_s8(signed)];
    let mut words = [vdupq_n_s32(0); 4];
    for (pair, half) in words.as_chunks_mut::<2>().0.iter_mut().zip(halves) {
      pair[0] = vmovl_s16(vget_low_s16(half));
      pair[1] = vmovl_high_s16(half);
    }
    for (slot, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(words) {
      let canonical = vaddq_s32(word, vandq_s32(vshrq_n_s32::<31>(word), q));
      // SAFETY: Each slot is four uniquely borrowed u32 lanes; the store writes exactly them.
      unsafe { vst1q_u32(slot.as_mut_ptr(), vreinterpretq_u32_s32(canonical)) };
    }
  }
}

/// Lane-wise canonical addition: inputs below q, output below q.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn add_canonical(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
  // a + b < 2q; when it is below q, subtracting q wraps above 2^31 and the unsigned minimum keeps it.
  let sum = vaddq_u32(a, b);
  vminq_u32(sum, vsubq_u32(sum, vdupq_n_u32(Q)))
}

/// Lane-wise canonical subtraction: inputs below q, output below q.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn sub_canonical(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
  let q = vdupq_n_u32(Q);
  let difference = vsubq_u32(vaddq_u32(a, q), b);
  vminq_u32(difference, vsubq_u32(difference, q))
}

/// Lane-wise FIPS 204 HighBits of canonical coefficients; see `x86_64::high_bits_lanes`.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn high_bits_lanes(x: uint32x4_t, gamma2: u32) -> uint32x4_t {
  let rounded = vaddq_u32(x, vdupq_n_u32(gamma2.strict_sub(1)));
  let (quotient, modulus) = if gamma2 == 95_232 {
    (vshrq_n_u32::<19>(vmulq_n_u32(vshrq_n_u32::<11>(rounded), 5_638)), 44)
  } else {
    (vshrq_n_u32::<25>(vmulq_n_u32(vshrq_n_u32::<9>(rounded), 32_801)), 16)
  };
  vbicq_u32(quotient, vceqq_u32(quotient, vdupq_n_u32(modulus)))
}

/// `a[i] = UseHint(hints[i], a[i])` (FIPS 204 Algorithm 40) for canonical coefficients and
/// hints of zero or one, as `super::use_hint` computes. Inputs are public verification data.
///
/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(super) unsafe fn use_hints(a: &mut [u32; N], hints: &[u32; N], gamma2: u32) {
  let modulus = vdupq_n_u32(if gamma2 == 95_232 { 44 } else { 16 });
  let one = vdupq_n_u32(1);
  let step = gamma2.strict_mul(2);
  for (a, hints) in a.as_chunks_mut::<4>().0.iter_mut().zip(hints.as_chunks::<4>().0) {
    // SAFETY: Each chunk is four initialized u32 lanes; loads and stores cover exactly them.
    unsafe {
      let x = vld1q_u32(a.as_ptr());
      let high = high_bits_lanes(x, gamma2);
      // The high part times 2·gamma2 is below q, so the remainder is canonical.
      let low = sub_canonical(x, vmulq_n_u32(high, step));
      // A centered remainder above zero: 1 <= low <= q/2.
      let positive = vcltq_u32(vsubq_u32(low, one), vdupq_n_u32(Q / 2));
      let up = vaddq_u32(high, one);
      let up = vbicq_u32(up, vceqq_u32(up, modulus));
      let down = vbslq_u32(vceqzq_u32(high), vsubq_u32(modulus, one), vsubq_u32(high, one));
      let adjusted = vbslq_u32(positive, up, down);
      let hinted = vtstq_u32(vld1q_u32(hints.as_ptr()), one);
      vst1q_u32(a.as_mut_ptr(), vbslq_u32(hinted, adjusted, high));
    }
  }
}

/// `a[i] = a[i] + b[i] mod q` for canonical coefficients.
///
/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(super) unsafe fn add_assign(a: &mut [u32; N], b: &[u32; N]) {
  for (a, b) in a.as_chunks_mut::<4>().0.iter_mut().zip(b.as_chunks::<4>().0) {
    // SAFETY: Each chunk is four initialized u32 lanes; loads and stores cover exactly them.
    unsafe {
      vst1q_u32(
        a.as_mut_ptr(),
        add_canonical(vld1q_u32(a.as_ptr()), vld1q_u32(b.as_ptr())),
      )
    };
  }
}

/// `a[i] = a[i] - b[i] mod q` for canonical coefficients.
///
/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(super) unsafe fn sub_assign(a: &mut [u32; N], b: &[u32; N]) {
  for (a, b) in a.as_chunks_mut::<4>().0.iter_mut().zip(b.as_chunks::<4>().0) {
    // SAFETY: Each chunk is four initialized u32 lanes; loads and stores cover exactly them.
    unsafe {
      vst1q_u32(
        a.as_mut_ptr(),
        sub_canonical(vld1q_u32(a.as_ptr()), vld1q_u32(b.as_ptr())),
      )
    };
  }
}

/// One when any centered coefficient has magnitude at least `bound` (`bound < q/2`), else zero;
/// see `x86_64::exceeds_bound`.
///
/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(super) unsafe fn exceeds_bound(a: &[u32; N], bound: u32) -> u32 {
  let offset = vdupq_n_u32(bound);
  let limit = vdupq_n_u32(Q.strict_sub(bound.strict_mul(2)));
  let mut exceeded = vdupq_n_u32(0);
  for chunk in a.as_chunks::<4>().0 {
    // SAFETY: `chunk` is four initialized u32 lanes; the load reads exactly them.
    let shifted = vsubq_u32(unsafe { vld1q_u32(chunk.as_ptr()) }, offset);
    exceeded = vorrq_u32(exceeded, vcleq_u32(shifted, limit));
  }
  u32::from(vmaxvq_u32(exceeded) != 0)
}

/// `out[i] = HighBits(a[i])` for canonical coefficients.
///
/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(super) unsafe fn high_bits_into(a: &[u32; N], gamma2: u32, out: &mut [u32; N]) {
  for (a, out) in a.as_chunks::<4>().0.iter().zip(out.as_chunks_mut::<4>().0) {
    // SAFETY: Each chunk is four initialized u32 lanes; loads and stores cover exactly them.
    unsafe { vst1q_u32(out.as_mut_ptr(), high_bits_lanes(vld1q_u32(a.as_ptr()), gamma2)) };
  }
}

/// `out[i] = LowBits(a[i]) mod q`, the canonical low part of Decompose.
///
/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(super) unsafe fn low_bits_into(a: &[u32; N], gamma2: u32, out: &mut [u32; N]) {
  let alpha = gamma2.strict_mul(2);
  for (a, out) in a.as_chunks::<4>().0.iter().zip(out.as_chunks_mut::<4>().0) {
    // SAFETY: Each chunk is four initialized u32 lanes; loads and stores cover exactly them.
    unsafe {
      let x = vld1q_u32(a.as_ptr());
      // HighBits is at most 15 (γ2 = 261888) or 43 (γ2 = 95232), so high·2γ2 is below q.
      let scaled = vmulq_n_u32(high_bits_lanes(x, gamma2), alpha);
      vst1q_u32(out.as_mut_ptr(), sub_canonical(x, scaled));
    }
  }
}

/// Replace `r[i]` with the hint bit `HighBits(r[i]) != HighBits(r[i] + t[i])` and return their
/// count.
///
/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(super) unsafe fn make_hints(r: &mut [u32; N], t: &[u32; N], gamma2: u32) -> usize {
  let one = vdupq_n_u32(1);
  let mut count = vdupq_n_u32(0);
  for (r, t) in r.as_chunks_mut::<4>().0.iter_mut().zip(t.as_chunks::<4>().0) {
    // SAFETY: Each chunk is four initialized u32 lanes; loads and stores cover exactly them.
    unsafe {
      let x = vld1q_u32(r.as_ptr());
      let shifted = add_canonical(x, vld1q_u32(t.as_ptr()));
      let same = vceqq_u32(high_bits_lanes(x, gamma2), high_bits_lanes(shifted, gamma2));
      let hint = vbicq_u32(one, same);
      count = vaddq_u32(count, hint);
      vst1q_u32(r.as_mut_ptr(), hint);
    }
  }
  // Each lane counts at most N / 4 hints.
  usize::try_from(vaddvq_u32(count)).expect("at most N hints")
}
