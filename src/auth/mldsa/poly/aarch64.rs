//! Four-lane NTT transforms and challenge scans for the macOS and Linux AArch64 NEON baseline.
//!
//! Keep the portable canonical representation and Montgomery arithmetic. Every
//! stage returns coefficients below q; no lane layout or secret owner changes.

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

/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(super) unsafe fn ntt(poly: &mut Poly) {
  let q = vdupq_n_u32(Q);
  let (left, right) = poly.0.split_at_mut(N / 2);
  for (a, b) in left.as_chunks_mut::<4>().0.iter_mut().zip(right.as_chunks_mut::<4>().0) {
    // SAFETY: Disjoint initialized four-u32 arrays, requiring only u32
    // alignment. The caller establishes NEON; pointers remain within each array.
    unsafe {
      let x = multiply(vld1q_u32(a.as_ptr()), vdupq_n_u32(R2));
      let y = multiply(vld1q_u32(b.as_ptr()), vdupq_n_u32(FIRST_FACTOR));
      vst1q_u32(a.as_mut_ptr(), reduce(vaddq_u32(x, y)));
      vst1q_u32(b.as_mut_ptr(), reduce(vsubq_u32(vaddq_u32(x, q), y)));
    }
  }
  let mut root = 2usize;
  let mut width = N / 4;
  while width >= 4 {
    for block in poly.0.chunks_exact_mut(width.strict_mul(2)) {
      let zeta = vdupq_n_u32(ROOTS[root]);
      root = root.strict_add(1);
      let (left, right) = block.split_at_mut(width);
      for (a, b) in left.as_chunks_mut::<4>().0.iter_mut().zip(right.as_chunks_mut::<4>().0) {
        // SAFETY: Each disjoint array contains four initialized u32 values.
        // Loads/stores require u32 alignment and never escape the unique borrow.
        unsafe {
          let x = vld1q_u32(a.as_ptr());
          let y = multiply(vld1q_u32(b.as_ptr()), zeta);
          vst1q_u32(a.as_mut_ptr(), reduce(vaddq_u32(x, y)));
          vst1q_u32(b.as_mut_ptr(), reduce(vsubq_u32(vaddq_u32(x, q), y)));
        }
      }
    }
    width >>= 1;
  }
  // Fuse widths two and one. Each vector initially combines two independent
  // width-two butterflies; transposition then supplies the adjacent pairs.
  for (index, block) in poly.0.as_chunks_mut::<8>().0.iter_mut().enumerate() {
    let root2 = (N / 4).strict_add(index.strict_mul(2));
    let root1 = (N / 2).strict_add(index.strict_mul(4));
    let zeta2 = vcombine_u32(vdup_n_u32(ROOTS[root2]), vdup_n_u32(ROOTS[root2.strict_add(1)]));
    // SAFETY: Eight initialized, uniquely borrowed u32 values. Two four-lane
    // loads and the eight-lane structure store stay within block. root1 is
    // 128..=252, so the four root loads stay within ROOTS. NEON is established.
    unsafe {
      let a = vreinterpretq_u64_u32(vld1q_u32(block.as_ptr()));
      let b = vreinterpretq_u64_u32(vld1q_u32(block.as_ptr().add(4)));
      let x = vreinterpretq_u32_u64(vzip1q_u64(a, b));
      let y = multiply(vreinterpretq_u32_u64(vzip2q_u64(a, b)), zeta2);
      let sums = reduce(vaddq_u32(x, y));
      let differences = reduce(vsubq_u32(vaddq_u32(x, q), y));
      let x = vtrn1q_u32(sums, differences);
      let y = multiply(vtrn2q_u32(sums, differences), vld1q_u32(ROOTS.as_ptr().add(root1)));
      let sums = reduce(vaddq_u32(x, y));
      let differences = reduce(vsubq_u32(vaddq_u32(x, q), y));
      vst2q_u32(block.as_mut_ptr(), uint32x4x2_t(sums, differences));
    }
  }
}

/// # Safety
/// The caller must establish NEON support.
#[target_feature(enable = "neon")]
pub(super) unsafe fn inverse_ntt(poly: &mut Poly) {
  // Fuse widths one and two in each eight-coefficient block. The first
  // deinterleave supplies four adjacent butterflies; transposition then pairs
  // their outputs for width two. No intermediate polynomial pass is needed.
  for (index, block) in poly.0.as_chunks_mut::<8>().0.iter_mut().enumerate() {
    let root1 = N.strict_sub(index.strict_mul(4)).strict_sub(4);
    let root2 = (N / 2).strict_sub(index.strict_mul(2)).strict_sub(2);
    let q = vdupq_n_u32(Q);
    let zeta2 = vcombine_u32(
      vdup_n_u32(Q.strict_sub(ROOTS[root2.strict_add(1)])),
      vdup_n_u32(Q.strict_sub(ROOTS[root2])),
    );
    // SAFETY: `block` is eight initialized, uniquely borrowed u32 values.
    // root1 ranges from 252 down to 128, so four root loads stay within ROOTS.
    // The structure load/store uses only u32 alignment and pointers do not
    // escape. The caller establishes NEON before entering this function.
    unsafe {
      let roots = vld1q_u32(ROOTS.as_ptr().add(root1));
      let reversed = vrev64q_u32(roots);
      let zeta1 = vsubq_u32(q, vextq_u32::<2>(reversed, reversed));
      let values = vld2q_u32(block.as_ptr());
      let sums = reduce(vaddq_u32(values.0, values.1));
      let products = multiply(zeta1, vsubq_u32(vaddq_u32(values.0, q), values.1));
      let left = vtrn1q_u32(sums, products);
      let right = vtrn2q_u32(sums, products);
      let sums = reduce(vaddq_u32(left, right));
      let products = multiply(zeta2, vsubq_u32(vaddq_u32(left, q), right));
      let sums = vreinterpretq_u64_u32(sums);
      let products = vreinterpretq_u64_u32(products);
      vst1q_u32(block.as_mut_ptr(), vreinterpretq_u32_u64(vzip1q_u64(sums, products)));
      vst1q_u32(
        block.as_mut_ptr().add(4),
        vreinterpretq_u32_u64(vzip2q_u64(sums, products)),
      );
    }
  }
  let mut root = N / 4;
  let mut width = 4usize;
  while width < N / 2 {
    // SAFETY: This function's caller establishes NEON; the stage borrows the
    // same initialized canonical polynomial. Width and root are public bounds.
    root = unsafe { inverse_stage(poly, width, root) };
    width = width.strict_mul(2);
  }
  let (left, right) = poly.0.split_at_mut(N / 2);
  for (a, b) in left.as_chunks_mut::<4>().0.iter_mut().zip(right.as_chunks_mut::<4>().0) {
    // SAFETY: Disjoint four-u32 arrays, initialized and aligned for u32, with
    // no pointer escape. Compile-time NEON is required by the module gate.
    unsafe {
      let x = vld1q_u32(a.as_ptr());
      let y = vld1q_u32(b.as_ptr());
      let difference = vsubq_u32(vaddq_u32(x, vdupq_n_u32(Q)), y);
      vst1q_u32(a.as_mut_ptr(), multiply(vaddq_u32(x, y), vdupq_n_u32(INV_N)));
      vst1q_u32(b.as_mut_ptr(), multiply(difference, vdupq_n_u32(LAST_FACTOR)));
    }
  }
}

// Keep Linux's generic CPU optimizer from unrolling a complete middle stage
// into enough live coefficient vectors to spill outside the polynomial owner.
// macOS retains its existing inline schedule.
/// # Safety
/// NEON must be available; width is 4, 8, 16, 32, or 64 and root starts at
/// N / width. Coefficients are canonical and all access stays in poly.
#[cfg_attr(target_os = "linux", inline(never))]
#[cfg_attr(not(target_os = "linux"), inline)]
#[target_feature(enable = "neon")]
unsafe fn inverse_stage(poly: &mut Poly, width: usize, mut root: usize) -> usize {
  // Advance by addition: out of line, `chunks_exact_mut` divides by the
  // runtime block length, and the CT gate rejects variable-latency division.
  let step = width.strict_mul(2);
  let mut start = 0;
  while start < N {
    let end = start.strict_add(step);
    let block = &mut poly.0[start..end];
    start = end;
    root = root.strict_sub(1);
    let zeta = vdupq_n_u32(Q.strict_sub(ROOTS[root]));
    let (left, right) = block.split_at_mut(width);
    for (a, b) in left.as_chunks_mut::<4>().0.iter_mut().zip(right.as_chunks_mut::<4>().0) {
      // SAFETY: Each disjoint array contains four initialized u32 lanes.
      // NEON loads/stores require only u32 alignment. Pointers do not escape;
      // this module requires macOS/Linux AArch64 and compile-time NEON support.
      unsafe {
        let x = vld1q_u32(a.as_ptr());
        let y = vld1q_u32(b.as_ptr());
        let difference = vsubq_u32(vaddq_u32(x, vdupq_n_u32(Q)), y);
        vst1q_u32(a.as_mut_ptr(), reduce(vaddq_u32(x, y)));
        vst1q_u32(b.as_mut_ptr(), multiply(zeta, difference));
      }
    }
  }
  root
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
