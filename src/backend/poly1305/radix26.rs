//! Scalar radix-2^26 helpers for key powers and for combining multi-chain results.
//!
//! Powers of the key use five limbs of 26 bits. The helpers convert between this form and the
//! radix-2^64 accumulator, multiply modulo 2^130 - 5, and carry column sums.

pub(super) const MASK26: u32 = 0x03ff_ffff;

pub(super) type Limbs = [u32; 5];

#[inline(always)]
fn low_u32(value: u64) -> u32 {
  let [b0, b1, b2, b3, ..] = value.to_le_bytes();
  u32::from_le_bytes([b0, b1, b2, b3])
}

/// Splits a radix-2^64 value with a top word <= 4 into five limbs; the top limb is below 2^27.
#[inline(always)]
pub(super) fn from_words(w0: u64, w1: u64, w2: u64) -> Limbs {
  debug_assert!(w2 <= 4);
  [
    low_u32(w0) & MASK26,
    low_u32(w0 >> 26) & MASK26,
    low_u32((w0 >> 52) | (w1 << 12)) & MASK26,
    low_u32(w1 >> 14) & MASK26,
    low_u32((w1 >> 40) | (w2 << 24)),
  ]
}

/// Carries five radix-2^26 columns below 2^62 into limbs below 2^26, except limb 2 at most 2^26.
#[inline(always)]
pub(super) fn carry_columns(d: [u64; 5]) -> Limbs {
  let [d0, d1, d2, d3, d4] = d;
  let mask = u64::from(MASK26);
  let d1 = d1.wrapping_add(d0 >> 26);
  let d2 = d2.wrapping_add(d1 >> 26);
  let d3 = d3.wrapping_add(d2 >> 26);
  let d4 = d4.wrapping_add(d3 >> 26);
  // d4 >> 26 < 2^36, so five times it fits beside limb 0.
  let l0 = (d0 & mask).wrapping_add((d4 >> 26).wrapping_mul(5));
  let l1 = (d1 & mask).wrapping_add(l0 >> 26);
  let l2 = (d2 & mask).wrapping_add(l1 >> 26);
  [
    low_u32(l0 & mask),
    low_u32(l1 & mask),
    low_u32(l2),
    low_u32(d3 & mask),
    low_u32(d4 & mask),
  ]
}

/// `(a·b) mod (2^130 - 5)` on scalar limbs below 2^27.
#[inline(always)]
pub(super) fn mul(a: Limbs, b: Limbs) -> Limbs {
  let [a0, a1, a2, a3, a4] = a.map(u64::from);
  let [b0, b1, b2, b3, b4] = b.map(u64::from);
  let [s1, s2, s3, s4] = [b1, b2, b3, b4].map(|limb| limb.wrapping_mul(5));
  // Each product is below 2^57 and each column of five below 2^60.
  let dot = |x: [u64; 5], y: [u64; 5]| {
    x.into_iter()
      .zip(y)
      .fold(0u64, |sum, (left, right)| sum.wrapping_add(left.wrapping_mul(right)))
  };
  let a = [a0, a1, a2, a3, a4];
  carry_columns([
    dot(a, [b0, s4, s3, s2, s1]),
    dot(a, [b1, b0, s4, s3, s2]),
    dot(a, [b2, b1, b0, s4, s3]),
    dot(a, [b3, b2, b1, b0, s4]),
    dot(a, [b4, b3, b2, b1, b0]),
  ])
}

/// Converts limbs from [`carry_columns`] to the radix-2^64 accumulator form with a top word <= 4.
#[inline(always)]
pub(super) fn to_words(limbs: Limbs) -> [u64; 3] {
  let top = u64::from(limbs[4] >> 24);
  let [l0, l1, l2, l3, l4] = limbs.map(u128::from);
  // l2 <= 2^26 and the other limbs are below 2^26, so the low part is below 2^128 + 2^105 and the
  // top word, at most 3 plus a carry, stays at most 4.
  let low = l0.wrapping_add(l1 << 26).wrapping_add(l2 << 52).wrapping_add(l3 << 78);
  let (low, carry) = low.overflowing_add((l4 & ((1 << 24) - 1)) << 104);
  let [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15] = low.to_le_bytes();
  [
    u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]),
    u64::from_le_bytes([b8, b9, b10, b11, b12, b13, b14, b15]),
    top.wrapping_add(u64::from(carry)),
  ]
}
