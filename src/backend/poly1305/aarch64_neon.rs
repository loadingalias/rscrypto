//! Four-lane NEON Poly1305 bulk kernel.
//!
//! Values use radix 2^26 with five limbs per lane. Each iteration takes eight blocks: block
//! `8g + j` enters lane `j` for `j < 4`, and block `8g + 4 + j` joins lane `j` after the multiply.
//! Lane `j` therefore updates as `h·r^8 + m_{8g+j}·r^4 + m_{8g+4+j}`, which keeps two blocks per lane
//! in one carry pass. After the last group, lane `j` multiplies by `r^(4-j)` and the lanes are
//! summed. The running accumulator enters lane 0 with the first block. Remaining blocks fewer than
//! eight take the scalar path. Only the public block count selects the passes.
//!
//! Each `umlal` multiplies two 32-bit limbs into a 64-bit column, so a block costs 12.5 multiply
//! instructions. An empty `asm!` keeps each column's accumulation in program order, which keeps the
//! loop free of spills. The bounds stated at each carry keep every narrowing exact.

use core::arch::aarch64::*;

use super::{
  radix26::{Limbs, MASK26, carry_columns, from_words, mul, to_words},
  scalar64::{State, absorb_block, load_u64_le, normalize, power, sum, times},
};
use crate::traits::ct;

/// Fewest blocks for which the kernel's power setup beats the scalar loop.
pub(crate) const MIN_BLOCKS: usize = 24;

/// Fewest blocks for which [`absorb_blocks_hybrid`]'s extra power setup pays.
pub(crate) const HYBRID_MIN_BLOCKS: usize = 512;

/// Five limb vectors; vector `k` holds limb `k` of four lanes.
#[derive(Clone, Copy)]
struct Lanes([uint32x4_t; 5]);

/// Five radix-2^26 columns, each as low lanes 0-1 and high lanes 2-3.
type Columns = [uint64x2_t; 10];

#[inline(always)]
fn zero_columns() -> Columns {
  // SAFETY: NEON is a compile-time target feature wherever this module compiles, and the
  // intrinsic is register-only.
  [unsafe { vdupq_n_u64(0) }; 10]
}

/// One multiplier for every lane, packed for by-element multiplies:
/// `[r0, r1, r2, r3]`, `[r4, 5r1, 5r2, 5r3]`, `[5r4, 0, 0, 0]`.
#[derive(Clone, Copy)]
struct Packed([uint32x4_t; 3]);

/// Lane `j` holds limb `k` of `values[j]` in vector `k`; `fivefold` holds the limbs times five.
#[derive(Clone, Copy)]
struct PerLane {
  limbs: Lanes,
  fivefold: Lanes,
}

#[inline(always)]
fn vector(lanes: [u32; 4]) -> uint32x4_t {
  // SAFETY: NEON is a compile-time target feature wherever this module compiles, and the
  // intrinsics below are register-only.
  unsafe {
    let [l0, l1, l2, l3] = lanes.map(u64::from);
    vcombine_u32(vcreate_u32((l1 << 32) | l0), vcreate_u32((l3 << 32) | l2))
  }
}

#[inline(always)]
fn pack(r: Limbs) -> Packed {
  let [r0, r1, r2, r3, r4] = r;
  // Power limbs are at most 2^26, so five times each fits in a lane.
  let [s1, s2, s3, s4] = [r1, r2, r3, r4].map(|limb| limb.wrapping_mul(5));
  Packed([
    vector([r0, r1, r2, r3]),
    vector([r4, s1, s2, s3]),
    vector([s4, 0, 0, 0]),
  ])
}

#[inline(always)]
fn per_lane(values: [Limbs; 4]) -> PerLane {
  let [v0, v1, v2, v3] = values;
  let limb = |k: usize| [v0[k], v1[k], v2[k], v3[k]];
  let fivefold = |k: usize| limb(k).map(|value| value.wrapping_mul(5));
  PerLane {
    limbs: Lanes([
      vector(limb(0)),
      vector(limb(1)),
      vector(limb(2)),
      vector(limb(3)),
      vector(limb(4)),
    ]),
    fivefold: Lanes([
      vector(fivefold(0)),
      vector(fivefold(1)),
      vector(fivefold(2)),
      vector(fivefold(3)),
      vector(fivefold(4)),
    ]),
  }
}

/// Lane 0 holds `limbs`; the other lanes are zero.
///
#[inline(always)]
fn lane0(limbs: Limbs) -> Lanes {
  let [l0, l1, l2, l3, l4] = limbs;
  Lanes([
    vector([l0, 0, 0, 0]),
    vector([l1, 0, 0, 0]),
    vector([l2, 0, 0, 0]),
    vector([l3, 0, 0, 0]),
    vector([l4, 0, 0, 0]),
  ])
}

/// Message limbs of four consecutive blocks with their 2^128 bit, lane `j` holding block `j`.
#[inline(always)]
fn load4(blocks: &[[u8; 16]; 4]) -> Lanes {
  // SAFETY: NEON is a compile-time target feature wherever this module compiles, and the
  // only memory access, `vld1q_u8_x4`, reads the 64 initialized bytes of `blocks` with byte alignment.
  unsafe {
    let bytes = vld1q_u8_x4(blocks.as_flattened().as_ptr());
    let b0 = vreinterpretq_u32_u8(bytes.0);
    let b1 = vreinterpretq_u32_u8(bytes.1);
    let b2 = vreinterpretq_u32_u8(bytes.2);
    let b3 = vreinterpretq_u32_u8(bytes.3);
    // Transpose so that vector `i` holds little-endian word `i` of each block.
    let t0 = vtrn1q_u32(b0, b1);
    let t1 = vtrn2q_u32(b0, b1);
    let t2 = vtrn1q_u32(b2, b3);
    let t3 = vtrn2q_u32(b2, b3);
    let w0 = vreinterpretq_u32_u64(vtrn1q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
    let w2 = vreinterpretq_u32_u64(vtrn2q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
    let w1 = vreinterpretq_u32_u64(vtrn1q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
    let w3 = vreinterpretq_u32_u64(vtrn2q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
    let mask = vdupq_n_u32(MASK26);
    Lanes([
      vandq_u32(w0, mask),
      vandq_u32(vsliq_n_u32::<6>(vshrq_n_u32::<26>(w0), w1), mask),
      vandq_u32(vsliq_n_u32::<12>(vshrq_n_u32::<20>(w1), w2), mask),
      vandq_u32(vsliq_n_u32::<18>(vshrq_n_u32::<14>(w2), w3), mask),
      vorrq_u32(vshrq_n_u32::<8>(w3), vdupq_n_u32(1 << 24)),
    ])
  }
}

#[inline(always)]
fn add(x: Lanes, y: Lanes) -> Lanes {
  // SAFETY: NEON is a compile-time target feature wherever this module compiles, and the
  // intrinsics below are register-only.
  unsafe {
    let [x0, x1, x2, x3, x4] = x.0;
    let [y0, y1, y2, y3, y4] = y.0;
    Lanes([
      vaddq_u32(x0, y0),
      vaddq_u32(x1, y1),
      vaddq_u32(x2, y2),
      vaddq_u32(x3, y3),
      vaddq_u32(x4, y4),
    ])
  }
}

/// Adds `x·p[LANE]` (by element) into column `col`.
#[inline(always)]
fn opaque(x: uint64x2_t) -> uint64x2_t {
  let mut x = x;
  // SAFETY: the template is empty; it only hides the value from reassociation.
  unsafe { core::arch::asm!("/* {0:v} */", inout(vreg) x, options(pure, nomem, nostack, preserves_flags)) };
  x
}

macro_rules! by_element {
  ($columns:ident, $low:literal, $high:literal, $x:expr, $p:expr, $lane:literal) => {
    $columns[$low] = opaque(vmlal_laneq_u32::<$lane>($columns[$low], vget_low_u32($x), $p));
    $columns[$high] = opaque(vmlal_high_laneq_u32::<$lane>($columns[$high], $x, $p));
  };
}

/// Adds `x·r` into the columns, with `r` the same in every lane.
///
#[inline(always)]
fn multiply_add(columns: &mut Columns, x: Lanes, r: Packed) {
  // SAFETY: NEON is a compile-time target feature wherever this module compiles, and the
  // intrinsics below are register-only.
  unsafe {
    let [x0, x1, x2, x3, x4] = x.0;
    let [p0, p1, p2] = r.0;
    // Column 0: x0·r0 + x1·5r4 + x2·5r3 + x3·5r2 + x4·5r1.
    by_element!(columns, 0, 1, x0, p0, 0);
    by_element!(columns, 0, 1, x1, p2, 0);
    by_element!(columns, 0, 1, x2, p1, 3);
    by_element!(columns, 0, 1, x3, p1, 2);
    by_element!(columns, 0, 1, x4, p1, 1);
    // Column 1: x0·r1 + x1·r0 + x2·5r4 + x3·5r3 + x4·5r2.
    by_element!(columns, 2, 3, x0, p0, 1);
    by_element!(columns, 2, 3, x1, p0, 0);
    by_element!(columns, 2, 3, x2, p2, 0);
    by_element!(columns, 2, 3, x3, p1, 3);
    by_element!(columns, 2, 3, x4, p1, 2);
    // Column 2: x0·r2 + x1·r1 + x2·r0 + x3·5r4 + x4·5r3.
    by_element!(columns, 4, 5, x0, p0, 2);
    by_element!(columns, 4, 5, x1, p0, 1);
    by_element!(columns, 4, 5, x2, p0, 0);
    by_element!(columns, 4, 5, x3, p2, 0);
    by_element!(columns, 4, 5, x4, p1, 3);
    // Column 3: x0·r3 + x1·r2 + x2·r1 + x3·r0 + x4·5r4.
    by_element!(columns, 6, 7, x0, p0, 3);
    by_element!(columns, 6, 7, x1, p0, 2);
    by_element!(columns, 6, 7, x2, p0, 1);
    by_element!(columns, 6, 7, x3, p0, 0);
    by_element!(columns, 6, 7, x4, p2, 0);
    // Column 4: x0·r4 + x1·r3 + x2·r2 + x3·r1 + x4·r0.
    by_element!(columns, 8, 9, x0, p1, 0);
    by_element!(columns, 8, 9, x1, p0, 3);
    by_element!(columns, 8, 9, x2, p0, 2);
    by_element!(columns, 8, 9, x3, p0, 1);
    by_element!(columns, 8, 9, x4, p0, 0);
  }
}

/// Adds `x·y` lane-wise into column `col`.
macro_rules! lane_wise {
  ($columns:ident, $low:literal, $high:literal, $x:expr, $y:expr) => {
    $columns[$low] = vmlal_u32($columns[$low], vget_low_u32($x), vget_low_u32($y));
    $columns[$high] = vmlal_high_u32($columns[$high], $x, $y);
  };
}

/// The columns of `x·r` with a different `r` in each lane.
///
#[inline(always)]
fn multiply_lanes(x: Lanes, r: &PerLane) -> Columns {
  // SAFETY: NEON is a compile-time target feature wherever this module compiles, and the
  // intrinsics below are register-only.
  unsafe {
    let [x0, x1, x2, x3, x4] = x.0;
    let [r0, r1, r2, r3, r4] = r.limbs.0;
    let [_, s1, s2, s3, s4] = r.fivefold.0;
    let mut columns = zero_columns();
    lane_wise!(columns, 0, 1, x0, r0);
    lane_wise!(columns, 0, 1, x1, s4);
    lane_wise!(columns, 0, 1, x2, s3);
    lane_wise!(columns, 0, 1, x3, s2);
    lane_wise!(columns, 0, 1, x4, s1);
    lane_wise!(columns, 2, 3, x0, r1);
    lane_wise!(columns, 2, 3, x1, r0);
    lane_wise!(columns, 2, 3, x2, s4);
    lane_wise!(columns, 2, 3, x3, s3);
    lane_wise!(columns, 2, 3, x4, s2);
    lane_wise!(columns, 4, 5, x0, r2);
    lane_wise!(columns, 4, 5, x1, r1);
    lane_wise!(columns, 4, 5, x2, r0);
    lane_wise!(columns, 4, 5, x3, s4);
    lane_wise!(columns, 4, 5, x4, s3);
    lane_wise!(columns, 6, 7, x0, r3);
    lane_wise!(columns, 6, 7, x1, r2);
    lane_wise!(columns, 6, 7, x2, r1);
    lane_wise!(columns, 6, 7, x3, r0);
    lane_wise!(columns, 6, 7, x4, s4);
    lane_wise!(columns, 8, 9, x0, r4);
    lane_wise!(columns, 8, 9, x1, r3);
    lane_wise!(columns, 8, 9, x2, r2);
    lane_wise!(columns, 8, 9, x3, r1);
    lane_wise!(columns, 8, 9, x4, r0);
    columns
  }
}

/// Low 26 bits and the 32-bit carry `>> 26` of one column, as four lanes.
///
/// The column must be below 2^58 so that the narrowed carry is exact.
///
#[inline(always)]
fn split_column(low: uint64x2_t, high: uint64x2_t) -> (uint32x4_t, uint32x4_t) {
  // SAFETY: NEON is a compile-time target feature wherever this module compiles, and the
  // intrinsics below are register-only.
  unsafe {
    let bits = vandq_u32(
      vuzp1q_u32(vreinterpretq_u32_u64(low), vreinterpretq_u32_u64(high)),
      vdupq_n_u32(MASK26),
    );
    let carry = vshrn_high_n_u64::<26>(vshrn_n_u64::<26>(low), high);
    (bits, carry)
  }
}

#[inline(always)]
fn times5(x: uint32x4_t) -> uint32x4_t {
  // SAFETY: NEON is a compile-time target feature wherever this module compiles, and the
  // intrinsics below are register-only.
  unsafe { vaddq_u32(x, vshlq_n_u32::<2>(x)) }
}

/// Carries the columns plus `extra` into limbs below 2^26 + 2^8.
///
/// Columns below 2^57.5, the last column below 2^55.4, and `extra` limbs below 2^26 keep every lane
/// of the first pass below 2^31.7, so the narrowed carries and the fivefold carry are exact.
///
#[inline(always)]
fn carry(columns: &Columns, extra: Lanes) -> Lanes {
  // SAFETY: NEON is a compile-time target feature wherever this module compiles, and the
  // intrinsics below are register-only.
  unsafe {
    let [e0, e1, e2, e3, e4] = extra.0;
    let (l0, c0) = split_column(columns[0], columns[1]);
    let (l1, c1) = split_column(columns[2], columns[3]);
    let (l2, c2) = split_column(columns[4], columns[5]);
    let (l3, c3) = split_column(columns[6], columns[7]);
    let (l4, c4) = split_column(columns[8], columns[9]);
    let t0 = vaddq_u32(vaddq_u32(l0, e0), times5(c4));
    let t1 = vaddq_u32(vaddq_u32(l1, e1), c0);
    let t2 = vaddq_u32(vaddq_u32(l2, e2), c1);
    let t3 = vaddq_u32(vaddq_u32(l3, e3), c2);
    let t4 = vaddq_u32(vaddq_u32(l4, e4), c3);
    let mask = vdupq_n_u32(MASK26);
    Lanes([
      vaddq_u32(vandq_u32(t0, mask), times5(vshrq_n_u32::<26>(t4))),
      vsraq_n_u32::<26>(vandq_u32(t1, mask), t0),
      vsraq_n_u32::<26>(vandq_u32(t2, mask), t1),
      vsraq_n_u32::<26>(vandq_u32(t3, mask), t2),
      vsraq_n_u32::<26>(vandq_u32(t4, mask), t3),
    ])
  }
}

/// Each column summed over the four lanes.
///
#[inline(always)]
fn sum_lanes(columns: &Columns) -> [u64; 5] {
  // SAFETY: NEON is a compile-time target feature wherever this module compiles, and the
  // intrinsics below are register-only.
  unsafe {
    // Four lanes of columns below 2^57 sum to below 2^59.
    [
      vaddvq_u64(vaddq_u64(columns[0], columns[1])),
      vaddvq_u64(vaddq_u64(columns[2], columns[3])),
      vaddvq_u64(vaddq_u64(columns[4], columns[5])),
      vaddvq_u64(vaddq_u64(columns[6], columns[7])),
      vaddvq_u64(vaddq_u64(columns[8], columns[9])),
    ]
  }
}

/// Absorbs full blocks, each with its 2^128 bit, into `state`.
///
/// The module compiles only where NEON is a compile-time target feature, so no runtime capability
/// obligation remains. It stays out of line so that the scalar path keeps a small frame.
#[inline(never)]
pub(crate) fn absorb_blocks(state: &mut State, blocks: &[[u8; 16]]) {
  let (groups, rest) = blocks.as_chunks::<8>();
  if let Some((first, others)) = groups.split_first() {
    let [r0, r1] = state.r;
    // powers = [r, r^2, r^3, r^4, r^8]. Products by the clamped `r` take the shorter radix-2^64
    // scalar chain; the squaring of r^4 takes radix 2^26.
    let mut chain = [[r0, r1, 0]; 4];
    for index in 1..chain.len() {
      chain[index] = state.times_r(chain[index.strict_sub(1)]);
    }
    let mut powers = [[0u32; 5]; 5];
    for (power, words) in powers.iter_mut().zip(&chain) {
      let [w0, w1, w2] = *words;
      *power = from_words(w0, w1, w2);
    }
    powers[4] = mul(powers[3], powers[3]);
    let r4 = pack(powers[3]);
    let r8 = pack(powers[4]);
    let finish = per_lane([powers[3], powers[2], powers[1], powers[0]]);

    let [h0, h1, h2] = state.h;
    let (quads, _) = first.as_chunks::<4>();
    let mut columns = zero_columns();
    multiply_add(&mut columns, add(load4(&quads[0]), lane0(from_words(h0, h1, h2))), r4);
    let mut h = carry(&columns, load4(&quads[1]));
    for group in others {
      let (quads, _) = group.as_chunks::<4>();
      let mut columns = zero_columns();
      multiply_add(&mut columns, load4(&quads[0]), r4);
      multiply_add(&mut columns, h, r8);
      h = carry(&columns, load4(&quads[1]));
    }
    state.h = to_words(carry_columns(sum_lanes(&multiply_lanes(h, &finish))));
    for limbs in &mut powers {
      ct::zeroize_words_no_fence(limbs);
    }
    for words in &mut chain {
      ct::zeroize_words_no_fence(words);
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
  state.absorb_blocks(rest);
}

/// Two blocks of one scalar chain.
#[inline(always)]
fn chain_step(acc: [u64; 3], pair: &[[u8; 16]; 2], r: [u64; 2], s1: u64) -> [u64; 3] {
  let [r0, r1] = r;
  let [first, second] = pair;
  let acc = absorb_block(
    acc,
    load_u64_le(&first[0..8]),
    load_u64_le(&first[8..16]),
    1,
    r0,
    r1,
    s1,
  );
  absorb_block(
    acc,
    load_u64_le(&second[0..8]),
    load_u64_le(&second[8..16]),
    1,
    r0,
    r1,
    s1,
  )
}

/// Absorbs full blocks with NEON lanes and four scalar chains running at once.
///
/// The vector pipes and the integer multipliers are separate resources, so the loop gives each
/// eight-block NEON step eight scalar blocks as well. The NEON lanes take the first half of the
/// blocks, and four scalar chains take the second half as four contiguous regions; the running
/// accumulator enters NEON lane 0. With `t` scalar blocks after the NEON part, the NEON lanes finish
/// with `r^(4-j+t)`, and the chains combine as in [`State::absorb_blocks_wide`]. The two parts then add.
/// Only the public block count selects the split.
#[inline(never)]
pub(crate) fn absorb_blocks_hybrid(state: &mut State, blocks: &[[u8; 16]]) {
  let iterations = blocks.len() / 16;
  if iterations == 0 {
    absorb_blocks(state, blocks);
    return;
  }
  let region = iterations.strict_mul(2);
  let (front, back) = blocks.split_at(iterations.strict_mul(8));
  let (groups, _) = front.as_chunks::<8>();
  let (q0, rest) = back.split_at(region);
  let (q1, rest) = rest.split_at(region);
  let (q2, q3) = rest.split_at(region);
  let extra = q3.len().strict_sub(region);
  let (q0, _) = q0.as_chunks::<2>();
  let (q1, _) = q1.as_chunks::<2>();
  let (q2, _) = q2.as_chunks::<2>();
  let (q3, tail) = q3.split_at(region);
  let (q3, _) = q3.as_chunks::<2>();

  let r = state.r;
  let s1 = state.s1();
  // powers = [r, r^2, r^3, r^4, r^8]; `spans` = [r^l, r^(l+e), r^(4l+e)] for region length `l` and
  // `e` extra blocks; `finals` = r^(4-j+4l+e) for lane j. All are cleared before return.
  let mut chain = [[r[0], r[1], 0]; 4];
  for index in 1..chain.len() {
    chain[index] = state.times_r(chain[index.strict_sub(1)]);
  }
  let mut powers = [[0u32; 5]; 5];
  for (power, words) in powers.iter_mut().zip(&chain) {
    let [w0, w1, w2] = *words;
    *power = from_words(w0, w1, w2);
  }
  powers[4] = mul(powers[3], powers[3]);
  let r4 = pack(powers[3]);
  let r8 = pack(powers[4]);
  let mut spans = [power(powers[0], region); 3];
  let quad = mul(mul(spans[0], spans[0]), mul(spans[0], spans[0]));
  if extra == 0 {
    spans[2] = quad;
  } else {
    let remainder = power(powers[0], extra);
    spans[1] = mul(spans[0], remainder);
    spans[2] = mul(quad, remainder);
  }
  let mut finals = [powers[3], powers[2], powers[1], powers[0]].map(|power| mul(power, spans[2]));
  let finish = per_lane(finals);

  let [h0, h1, h2] = state.h;
  let mut acc = [[0u64; 3]; 4];
  let mut lanes = None;
  for (group, (((a, b), c), d)) in groups.iter().zip(q0.iter().zip(q1).zip(q2).zip(q3)) {
    let (quads, _) = group.as_chunks::<4>();
    let mut columns = zero_columns();
    lanes = Some(match lanes {
      None => {
        multiply_add(&mut columns, add(load4(&quads[0]), lane0(from_words(h0, h1, h2))), r4);
        carry(&columns, load4(&quads[1]))
      }
      Some(h) => {
        multiply_add(&mut columns, load4(&quads[0]), r4);
        multiply_add(&mut columns, h, r8);
        carry(&columns, load4(&quads[1]))
      }
    });
    acc[0] = chain_step(acc[0], a, r, s1);
    acc[1] = chain_step(acc[1], b, r, s1);
    acc[2] = chain_step(acc[2], c, r, s1);
    acc[3] = chain_step(acc[3], d, r, s1);
  }
  for block in tail {
    acc[3] = absorb_block(
      acc[3],
      load_u64_le(&block[0..8]),
      load_u64_le(&block[8..16]),
      1,
      r[0],
      r[1],
      s1,
    );
  }

  let vector_part = match lanes {
    Some(h) => to_words(carry_columns(sum_lanes(&multiply_lanes(h, &finish)))),
    None => [0; 3],
  };
  let [a, b, c, d] = acc.map(normalize);
  let scalar_part = sum(times(sum(times(sum(times(a, spans[0]), b), spans[0]), c), spans[1]), d);
  state.h = sum(vector_part, scalar_part);

  for limbs in powers.iter_mut().chain(&mut spans).chain(&mut finals) {
    ct::zeroize_words_no_fence(limbs);
  }
  for words in chain.iter_mut().chain(&mut acc) {
    ct::zeroize_words_no_fence(words);
  }
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}
