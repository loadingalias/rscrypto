//! Four-lane NTT transforms and challenge scans for the macOS and Linux AArch64 NEON baseline.
//!
//! Products and elementwise operations apply the portable canonical Montgomery arithmetic to
//! lanes. The transforms use signed Montgomery multiplication with lazily reduced lanes and
//! reduce to the same canonical residues as the portable transforms; no lane layout or secret
//! owner changes.

use core::arch::aarch64::*;

use super::{FIRST_FACTOR, INV_N, LAST_FACTOR, N, NEG_Q_INVERSE, Poly, Q, R2, ROOTS};
use crate::auth::mldsa::sampling::{CHALLENGE_BYTES, CHALLENGE_WINDOWS};

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

/// `q⁻¹ mod 2^32`, as a signed lane.
const QINV: i32 = 58_728_449;

/// A signed Montgomery multiplier: the centered residue `value` in `(-q/2, q/2]` and
/// `twisted = value·q⁻¹ mod 2^32`.
#[derive(Clone, Copy)]
struct Twiddle {
  value: i32,
  twisted: i32,
}

const fn twiddle(x: u32) -> Twiddle {
  let value = if x > Q / 2 {
    x.cast_signed().wrapping_sub(Q.cast_signed())
  } else {
    x.cast_signed()
  };
  Twiddle {
    value,
    twisted: value.wrapping_mul(QINV),
  }
}

/// Forward multipliers `ROOTS[k]` and inverse multipliers `q - ROOTS[k]`.
const FORWARD: [Twiddle; N] = twiddles(false);
const INVERSE: [Twiddle; N] = twiddles(true);

const fn twiddles(inverse: bool) -> [Twiddle; N] {
  let mut table = [Twiddle { value: 0, twisted: 0 }; N];
  let mut k = 1usize;
  while k < N {
    table[k] = twiddle(if inverse { Q.strict_sub(ROOTS[k]) } else { ROOTS[k] });
    k = k.strict_add(1);
  }
  table
}

/// Lane multipliers for the in-pair width-2 and width-1 stages, one `(values, twisted)` row per
/// eight-coefficient block, in the lane order the pair schedule uses.
type LaneTable = ([[i32; 4]; N / 8], [[i32; 4]; N / 8]);

const fn lane_table(inverse: bool, width_one: bool) -> LaneTable {
  let table = if inverse { INVERSE } else { FORWARD };
  let mut values = [[0i32; 4]; N / 8];
  let mut twisted = [[0i32; 4]; N / 8];
  let mut block = 0usize;
  while block < N / 8 {
    let mut lane = 0usize;
    while lane < 4 {
      let root = match (inverse, width_one) {
        // Width 2: lanes 0-1 use the block's first root, lanes 2-3 the next.
        (false, false) => 64usize.strict_add(block.strict_mul(2)).strict_add(lane / 2),
        (true, false) => 127usize.strict_sub(block.strict_mul(2)).strict_sub(lane / 2),
        // Width 1: lane i holds the block's i-th butterfly.
        (false, true) => 128usize.strict_add(block.strict_mul(4)).strict_add(lane),
        (true, true) => 255usize.strict_sub(block.strict_mul(4)).strict_sub(lane),
      };
      values[block][lane] = table[root].value;
      twisted[block][lane] = table[root].twisted;
      lane = lane.strict_add(1);
    }
    block = block.strict_add(1);
  }
  (values, twisted)
}

const FORWARD_WIDTH_2: LaneTable = lane_table(false, false);
const FORWARD_WIDTH_1: LaneTable = lane_table(false, true);
const INVERSE_WIDTH_2: LaneTable = lane_table(true, false);
const INVERSE_WIDTH_1: LaneTable = lane_table(true, true);
const R2_TWIDDLE: Twiddle = twiddle(R2);
const FIRST_TWIDDLE: Twiddle = twiddle(FIRST_FACTOR);
const INV_N_TWIDDLE: Twiddle = twiddle(INV_N);
const LAST_TWIDDLE: Twiddle = twiddle(LAST_FACTOR);

/// Signed Montgomery multiplication `a·value·2^-32 mod q` lane by lane.
///
/// With `m = a·twisted mod 2^32`, `a·value - m·q` is an exact multiple of 2^32, so the doubled
/// high halves of `a·value` and `m·q` differ by exactly twice the quotient, which lies in
/// `(-q, q)` for `|value| <= q/2`. Saturation needs both doubled-multiply operands at `i32::MIN`,
/// which `q` never is.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn mont_mul(a: int32x4_t, value: int32x4_t, twisted: int32x4_t) -> int32x4_t {
  let high = vqdmulhq_s32(a, value);
  let low = vmulq_s32(a, twisted);
  vhsubq_s32(high, vqdmulhq_s32(low, vdupq_n_s32(Q.cast_signed())))
}

/// [`mont_mul`] by one multiplier in every lane.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn mont_mul_broadcast(a: int32x4_t, multiplier: Twiddle) -> int32x4_t {
  mont_mul(a, vdupq_n_s32(multiplier.value), vdupq_n_s32(multiplier.twisted))
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

/// Cooley–Tukey butterfly `(a, b) -> (a + b·ζ, a - b·ζ)`, without reduction.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn forward_butterfly(a: &mut int32x4_t, b: &mut int32x4_t, zeta: Twiddle) {
  let t = mont_mul_broadcast(*b, zeta);
  *b = vsubq_s32(*a, t);
  *a = vaddq_s32(*a, t);
}

/// Gentleman–Sande butterfly `(a, b) -> (a + b, (a - b)·ζ)`, without reducing the sum.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn inverse_butterfly(a: &mut int32x4_t, b: &mut int32x4_t, zeta: Twiddle) {
  let difference = vsubq_s32(*a, *b);
  *a = vaddq_s32(*a, *b);
  *b = mont_mul_broadcast(difference, zeta);
}

/// Row `block` of a lane table.
///
/// # Safety
/// Calling outside a NEON-enabled function requires NEON support.
#[inline]
#[target_feature(enable = "neon")]
fn lane_row(table: &LaneTable, block: usize) -> (int32x4_t, int32x4_t) {
  // SAFETY: Each row is four initialized i32 values; the loads read exactly them.
  unsafe { (vld1q_s32(table.0[block].as_ptr()), vld1q_s32(table.1[block].as_ptr())) }
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
/// Lanes are signed and lazily reduced: each Montgomery product lies in `(-q, q)` and each stage
/// adds at most `q`, so every lane stays below `9·q` and one final reduction returns canonical
/// residues. The first pass runs widths 128, 64 and 32 over eight vectors 32 coefficients apart;
/// the second runs widths 16, 8 and 4 across one 32-coefficient block and widths 2 and 1 inside
/// vector pairs. Each pass keeps eight coefficient vectors and its temporaries in registers.
///
/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(super) unsafe fn ntt(poly: &mut Poly) {
  let chunks = poly.0.as_chunks_mut::<4>().0;
  for v in 0..8usize {
    let mut x = load_eight(chunks, v, 8);
    let (low, high) = x.split_at_mut(4);
    // Width 128, fused with the Montgomery conversion: M(a, R2) and M(b, M(R2, ζ1)).
    for (a, b) in low.iter_mut().zip(high) {
      let left = mont_mul_broadcast(*a, R2_TWIDDLE);
      let right = mont_mul_broadcast(*b, FIRST_TWIDDLE);
      *a = vaddq_s32(left, right);
      *b = vsubq_s32(left, right);
    }
    for (first, root) in [(0usize, 2usize), (4, 3)] {
      for k in first..first.strict_add(2) {
        let (a, b) = x.split_at_mut(k.strict_add(2));
        forward_butterfly(&mut a[k], &mut b[0], FORWARD[root]);
      }
    }
    for (pair, root) in (0usize..4).zip(4usize..8) {
      let (a, b) = x.split_at_mut(pair.strict_mul(2).strict_add(1));
      forward_butterfly(&mut a[pair.strict_mul(2)], &mut b[0], FORWARD[root]);
    }
    store_eight(chunks, v, 8, &unsigned_eight(&x));
  }
  for w in 0..8usize {
    let mut x = load_eight(chunks, w.strict_mul(8), 1);
    let (low, high) = x.split_at_mut(4);
    for (a, b) in low.iter_mut().zip(high) {
      forward_butterfly(a, b, FORWARD[8usize.strict_add(w)]);
    }
    for (first, root) in [(0usize, 16usize), (4, 17)] {
      let root = root.strict_add(w.strict_mul(2));
      for k in first..first.strict_add(2) {
        let (a, b) = x.split_at_mut(k.strict_add(2));
        forward_butterfly(&mut a[k], &mut b[0], FORWARD[root]);
      }
    }
    for pair in 0..4usize {
      let (a, b) = x.split_at_mut(pair.strict_mul(2).strict_add(1));
      forward_butterfly(
        &mut a[pair.strict_mul(2)],
        &mut b[0],
        FORWARD[32usize.strict_add(w.strict_mul(4)).strict_add(pair)],
      );
    }
    // Widths 2 and 1 inside each eight-coefficient pair of vectors.
    let mut out = [vdupq_n_u32(0); 8];
    for (pair, (lanes, result)) in x.as_chunks::<2>().0.iter().zip(out.as_chunks_mut::<2>().0).enumerate() {
      let block = w.strict_mul(4).strict_add(pair);
      let a = vreinterpretq_s64_s32(lanes[0]);
      let b = vreinterpretq_s64_s32(lanes[1]);
      let (value, twisted) = lane_row(&FORWARD_WIDTH_2, block);
      let left = vreinterpretq_s32_s64(vzip1q_s64(a, b));
      let right = mont_mul(vreinterpretq_s32_s64(vzip2q_s64(a, b)), value, twisted);
      let sums = vaddq_s32(left, right);
      let differences = vsubq_s32(left, right);
      let (value, twisted) = lane_row(&FORWARD_WIDTH_1, block);
      let left = vtrn1q_s32(sums, differences);
      let right = mont_mul(vtrn2q_s32(sums, differences), value, twisted);
      // Interleave the width-1 outputs back into coefficient order.
      let sums = canonical(vaddq_s32(left, right));
      let differences = canonical(vsubq_s32(left, right));
      result[0] = vzip1q_u32(sums, differences);
      result[1] = vzip2q_u32(sums, differences);
    }
    store_eight(chunks, w.strict_mul(8), 1, &out);
  }
}

/// FIPS 204 Algorithm 42 with conversion out of Montgomery form, equal to the portable
/// transform.
///
/// The passes mirror [`ntt`]. Sums double at each stage, so after width 8 (below `16·q`) every
/// lane is reduced to below `0.52·q`; the remaining stages stay below `8.4·q`, and the last
/// stage multiplies both outputs by `n⁻¹`-scaled constants, leaving `(-q, q)` before the final
/// reduction.
///
/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(super) unsafe fn inverse_ntt(poly: &mut Poly) {
  let chunks = poly.0.as_chunks_mut::<4>().0;
  for w in 0..8usize {
    let loaded = load_eight(chunks, w.strict_mul(8), 1);
    let mut x = [vdupq_n_s32(0); 8];
    // Widths 1 and 2 inside each eight-coefficient pair of vectors.
    for (pair, (lanes, result)) in loaded
      .as_chunks::<2>()
      .0
      .iter()
      .zip(x.as_chunks_mut::<2>().0)
      .enumerate()
    {
      let block = w.strict_mul(4).strict_add(pair);
      let even = vuzp1q_s32(lanes[0], lanes[1]);
      let odd = vuzp2q_s32(lanes[0], lanes[1]);
      let (value, twisted) = lane_row(&INVERSE_WIDTH_1, block);
      let sums = vaddq_s32(even, odd);
      let products = mont_mul(vsubq_s32(even, odd), value, twisted);
      let left = vtrn1q_s32(sums, products);
      let right = vtrn2q_s32(sums, products);
      let (value, twisted) = lane_row(&INVERSE_WIDTH_2, block);
      let sums = vreinterpretq_s64_s32(vaddq_s32(left, right));
      let products = vreinterpretq_s64_s32(mont_mul(vsubq_s32(left, right), value, twisted));
      result[0] = vreinterpretq_s32_s64(vzip1q_s64(sums, products));
      result[1] = vreinterpretq_s32_s64(vzip2q_s64(sums, products));
    }
    for pair in 0..4usize {
      let (a, b) = x.split_at_mut(pair.strict_mul(2).strict_add(1));
      inverse_butterfly(
        &mut a[pair.strict_mul(2)],
        &mut b[0],
        INVERSE[63usize.strict_sub(w.strict_mul(4)).strict_sub(pair)],
      );
    }
    for (first, root) in [(0usize, 31usize), (4, 30)] {
      let root = root.strict_sub(w.strict_mul(2));
      for k in first..first.strict_add(2) {
        let (a, b) = x.split_at_mut(k.strict_add(2));
        inverse_butterfly(&mut a[k], &mut b[0], INVERSE[root]);
      }
    }
    for lanes in &mut x {
      *lanes = reduce32(*lanes);
    }
    let (low, high) = x.split_at_mut(4);
    for (a, b) in low.iter_mut().zip(high) {
      inverse_butterfly(a, b, INVERSE[15usize.strict_sub(w)]);
    }
    store_eight(chunks, w.strict_mul(8), 1, &unsigned_eight(&x));
  }
  for v in 0..8usize {
    let mut x = load_eight(chunks, v, 8);
    for (pair, root) in (0usize..4).zip([7usize, 6, 5, 4]) {
      let (a, b) = x.split_at_mut(pair.strict_mul(2).strict_add(1));
      inverse_butterfly(&mut a[pair.strict_mul(2)], &mut b[0], INVERSE[root]);
    }
    for (first, root) in [(0usize, 3usize), (4, 2)] {
      for k in first..first.strict_add(2) {
        let (a, b) = x.split_at_mut(k.strict_add(2));
        inverse_butterfly(&mut a[k], &mut b[0], INVERSE[root]);
      }
    }
    // Width 128, fused with normalization: M(a + b, n⁻¹) and M(a - b, M(q - ζ1, n⁻¹)).
    let mut out = [vdupq_n_u32(0); 8];
    let (low, high) = x.split_at(4);
    let (out_low, out_high) = out.split_at_mut(4);
    for (((&a, &b), result_a), result_b) in low.iter().zip(high).zip(out_low).zip(out_high) {
      *result_a = canonical(mont_mul_broadcast(vaddq_s32(a, b), INV_N_TWIDDLE));
      *result_b = canonical(mont_mul_broadcast(vsubq_s32(a, b), LAST_TWIDDLE));
    }
    store_eight(chunks, v, 8, &out);
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

/// Fixed-work search for the first eligible challenge byte.
///
/// Returns the minimum of `position << 8 | byte` over positions at or after
/// `offset` whose byte is at most `limit`, or `u32::MAX` when none qualifies,
/// as `sampling::first_challenge_candidate` does. Every window is read, and
/// neither address nor control flow depends on a byte, offset or limit.
///
/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(in crate::auth::mldsa) unsafe fn challenge_first_candidate(
  bytes: &[u8; CHALLENGE_BYTES],
  offset: u32,
  limit: u32,
) -> u32 {
  // Offsets are at most 221 and limits at most 255. Masking first makes each
  // conversion provably in range, so no check branches on the secret offset.
  let offset = vdupq_n_u16(u16::try_from(offset & 0xffff).expect("masked offsets fit u16"));
  let limit = vdupq_n_u16(u16::try_from(limit & 0xffff).expect("masked limits fit u16"));
  let lanes: [u16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
  // SAFETY: The load reads exactly the eight initialized lanes of `lanes`.
  let lanes = unsafe { vld1q_u16(lanes.as_ptr()) };
  let mut first = vdupq_n_u16(u16::MAX);
  for start in CHALLENGE_WINDOWS {
    let window: &[u8; 16] = bytes[start..start.strict_add(16)]
      .try_into()
      .expect("challenge windows have sixteen bytes");
    // SAFETY: The load reads exactly the sixteen initialized bytes of `window`.
    let window = unsafe { vld1q_u8(window.as_ptr()) };
    let start = u16::try_from(start).expect("challenge positions fit u16");
    for (half, candidates) in [(0, vmovl_u8(vget_low_u8(window))), (8, vmovl_high_u8(window))] {
      let positions = vaddq_u16(vdupq_n_u16(start.strict_add(half)), lanes);
      let eligible = vandq_u16(vcgeq_u16(positions, offset), vcleq_u16(candidates, limit));
      let packed = vorrq_u16(vshlq_n_u16::<8>(positions), candidates);
      first = vminq_u16(first, vorrq_u16(packed, vmvnq_u16(eligible)));
    }
  }
  let first = u32::from(vminvq_u16(first));
  // An eligible packed value is at most 220 << 8 | 255, so 0xffff means none.
  let none = (first ^ 0xffff).wrapping_sub(1) >> 31;
  first | super::opaque_mask(0u32.wrapping_sub(none))
}

/// Write `sign` at coefficient `j` and return the coefficient it replaces.
///
/// Every coefficient is read and written; the index comparison selects lanes
/// without a secret-dependent address, as the portable placement scan does.
///
/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(in crate::auth::mldsa) unsafe fn challenge_place(coefficients: &mut [u32; N], j: u32, sign: u32) -> u32 {
  let target = vdupq_n_u32(j);
  let sign = vdupq_n_u32(sign);
  let lanes: [u32; 4] = [0, 1, 2, 3];
  // SAFETY: The load reads exactly the four initialized lanes of `lanes`.
  let mut index = unsafe { vld1q_u32(lanes.as_ptr()) };
  let mut previous = vdupq_n_u32(0);
  for chunk in coefficients.as_chunks_mut::<4>().0 {
    // SAFETY: Each load and store covers exactly one uniquely borrowed
    // four-lane chunk and needs only u32 alignment. N has no vector tail.
    unsafe {
      let value = vld1q_u32(chunk.as_ptr());
      let hit = vceqq_u32(index, target);
      previous = vorrq_u32(previous, vandq_u32(value, hit));
      vst1q_u32(chunk.as_mut_ptr(), vbslq_u32(hit, sign, value));
    }
    index = vaddq_u32(index, vdupq_n_u32(4));
  }
  let previous = vorr_u32(vget_low_u32(previous), vget_high_u32(previous));
  vget_lane_u32::<0>(previous) | vget_lane_u32::<1>(previous)
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
