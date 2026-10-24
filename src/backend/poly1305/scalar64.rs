//! Radix-2^64 scalar Poly1305 arithmetic for 64-bit targets.
//!
//! A residue is `h0 + h1·2^64 + h2·2^128`, with `h2 <= 4` between blocks. Clamping keeps
//! `r0, r1 < 2^60` and makes `r1` a multiple of four, so a product term of weight 2^128 folds
//! exactly: `h1·r1·2^128 = h1·(r1/4)·2^130 ≡ h1·s1 (mod 2^130 - 5)` with `s1 = 5·r1/4`.
//! One block costs four 64×64→128 products and two 64-bit products. Between blocks the accumulator
//! stays unreduced, and the fold of its bits above 2^130 joins the next message addition. Each step
//! states the bound that keeps its wrapping arithmetic from wrapping.

use super::radix26;
use crate::traits::ct;

const R0_CLAMP: u64 = 0x0fff_fffc_0fff_ffff;
const R1_CLAMP: u64 = 0x0fff_fffc_0fff_fffc;

#[inline(always)]
pub(super) fn load_u64_le(bytes: &[u8]) -> u64 {
  let mut word = [0u8; 8];
  word.copy_from_slice(bytes);
  u64::from_le_bytes(word)
}

#[inline(always)]
fn low(value: u128) -> u64 {
  let [b0, b1, b2, b3, b4, b5, b6, b7, ..] = value.to_le_bytes();
  u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7])
}

#[inline(always)]
fn high(value: u128) -> u64 {
  low(value >> 64)
}

/// The exact 128-bit product.
#[inline(always)]
fn wide(a: u64, b: u64) -> u128 {
  u128::from(a).wrapping_mul(u128::from(b))
}

/// The little-endian value of at most eight bytes.
///
/// Only the public length selects the loads.
#[inline(always)]
fn partial_word(bytes: &[u8]) -> u64 {
  if let Some(word) = bytes.first_chunk::<8>() {
    return u64::from_le_bytes(*word);
  }
  let mut value = 0u64;
  let mut shift = 0u32;
  let mut rest = bytes;
  if let Some((word, tail)) = rest.split_first_chunk::<4>() {
    value = u64::from(u32::from_le_bytes(*word));
    shift = 32;
    rest = tail;
  }
  if let Some((word, tail)) = rest.split_first_chunk::<2>() {
    value |= u64::from(u16::from_le_bytes(*word)) << shift;
    shift = shift.strict_add(16);
    rest = tail;
  }
  if let Some(&byte) = rest.first() {
    value |= u64::from(byte) << shift;
  }
  value
}

/// The two words of a final fragment of 1 to 15 bytes followed by a byte of one and zeros.
///
/// Building the words in registers leaves no stack copy of the fragment to clear.
#[inline(always)]
fn fragment_words(fragment: &[u8]) -> (u64, u64) {
  let len = fragment.len();
  debug_assert!((1..16).contains(&len));
  let (first, second) = fragment.split_at(len.min(8));
  let w0 = partial_word(first);
  let w1 = partial_word(second);
  if len < 8 {
    (w0 | (1u64 << len.strict_mul(8)), 0)
  } else {
    (w0, w1 | (1u64 << len.strict_sub(8).strict_mul(8)))
  }
}

/// Folds the bits at and above 2^130 of an unreduced accumulator back multiplied by five.
///
/// The accumulator is `a0 + a1·2^64 + top·2^128` with `top < 2^63.3`; the result has a top word <= 4.
#[inline(always)]
pub(super) fn normalize(acc: [u64; 3]) -> [u64; 3] {
  let [a0, a1, top] = acc;
  // 5·(top >> 2) < 2^63.6.
  let fold = (top >> 2).wrapping_add(top & !3);
  let (h0, carry0) = a0.overflowing_add(fold);
  let (h1, carry1) = a1.overflowing_add(u64::from(carry0));
  [h0, h1, (top & 3).wrapping_add(u64::from(carry1))]
}

/// `(acc + m + hibit·2^128)·r` as an unreduced accumulator for [`normalize`] or the next block.
///
/// `acc` is `a0 + a1·2^64 + top·2^128` with `top < 2^63.3`; a normalized value qualifies. The fold of
/// `top` joins the message addition, so the carries of the message words leave the dependency chain
/// through the multiply. `hibit <= 1`, and `(r0, r1, s1)` come from a clamped key.
#[inline(always)]
pub(super) fn absorb_block(acc: [u64; 3], m0: u64, m1: u64, hibit: u64, r0: u64, r1: u64, s1: u64) -> [u64; 3] {
  let [a0, a1, top] = acc;
  debug_assert!(top >> 63 == 0 && hibit <= 1);

  // 5·(top >> 2) < 2^63.6.
  let fold = (top >> 2).wrapping_add(top & !3);
  let (v0, carry0) = a0.overflowing_add(m0);
  let (v1, carry1) = a1.overflowing_add(m1);
  let (v1, carry2) = v1.overflowing_add(u64::from(carry0));
  // At most one of the two carries out of the middle word is set, so v2 <= 3 + 1 + 1.
  let v2 = (top & 3)
    .wrapping_add(hibit)
    .wrapping_add(u64::from(carry1))
    .wrapping_add(u64::from(carry2));
  let (t0, carry3) = v0.overflowing_add(fold);
  let (t1, carry4) = v1.overflowing_add(u64::from(carry3));
  // t2 <= 6.
  let t2 = v2.wrapping_add(u64::from(carry4));

  // t0·r0 < 2^124 and t1·s1 < 1.25·2^124, so d0 < 2^126.
  let d0 = wide(t0, r0).wrapping_add(wide(t1, s1));
  // t0·r1, t1·r0 < 2^124 and t2·s1 < 6·1.25·2^60 < 2^63, so d1 < 2^125 + 2^63.
  let d1 = wide(t0, r1)
    .wrapping_add(wide(t1, r0))
    .wrapping_add(u128::from(t2.wrapping_mul(s1)));
  // t2·r0 < 6·2^60.
  let d2 = t2.wrapping_mul(r0);

  // d0 >> 64 < 2^62 keeps d1 below 2^126.
  let d1 = d1.wrapping_add(u128::from(high(d0)));
  // (d1 >> 64) < 2^62 and d2 < 2^62.6, so the top word is below 2^63.2.
  [low(d0), low(d1), high(d1).wrapping_add(d2)]
}

/// Independent Horner chains in [`State::absorb_blocks_wide`].
const CHAINS: usize = 4;

/// Fewest blocks for which [`State::absorb_blocks_wide`] beats one chain.
pub(super) const WIDE_MIN_BLOCKS: usize = 48;

/// `base^exponent` for `exponent >= 1`, by squaring and multiplying on the public exponent bits.
#[inline(always)]
pub(super) fn power(base: radix26::Limbs, exponent: usize) -> radix26::Limbs {
  debug_assert!(exponent >= 1);
  let mut result = base;
  for bit in (0..exponent.ilog2()).rev() {
    result = radix26::mul(result, result);
    if (exponent >> bit) & 1 == 1 {
      result = radix26::mul(result, base);
    }
  }
  result
}

/// `x·y` for a normalized `x`, normalized.
#[inline(always)]
pub(super) fn times(x: [u64; 3], y: radix26::Limbs) -> [u64; 3] {
  let [x0, x1, x2] = x;
  radix26::to_words(radix26::mul(radix26::from_words(x0, x1, x2), y))
}

/// `x + y` for normalized operands, normalized.
#[inline(always)]
pub(super) fn sum(x: [u64; 3], y: [u64; 3]) -> [u64; 3] {
  let (s0, carry0) = x[0].overflowing_add(y[0]);
  let (s1, carry1) = x[1].overflowing_add(y[1]);
  let (s1, carry2) = s1.overflowing_add(u64::from(carry0));
  // Top words <= 4 each and at most one carry: the top stays below 10.
  let s2 = x[2]
    .wrapping_add(y[2])
    .wrapping_add(u64::from(carry1))
    .wrapping_add(u64::from(carry2));
  normalize([s0, s1, s2])
}

/// Scalar Poly1305 state: accumulator, clamped key, and additive pad. It clears itself on drop.
pub(crate) struct State {
  pub(super) h: [u64; 3],
  pub(super) r: [u64; 2],
  pad: [u64; 2],
}

impl Drop for State {
  fn drop(&mut self) {
    ct::zeroize_words_no_fence(&mut self.h);
    ct::zeroize_words_no_fence(&mut self.r);
    ct::zeroize_words_no_fence(&mut self.pad);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

impl State {
  #[inline(always)]
  pub(crate) fn new(key: &[u8; 32]) -> Self {
    Self {
      h: [0; 3],
      r: [load_u64_le(&key[0..8]) & R0_CLAMP, load_u64_le(&key[8..16]) & R1_CLAMP],
      pad: [load_u64_le(&key[16..24]), load_u64_le(&key[24..32])],
    }
  }

  /// `s1 = 5·r1/4`, exact because clamping clears the low two bits of `r1`.
  #[inline(always)]
  pub(super) fn s1(&self) -> u64 {
    self.r[1].wrapping_add(self.r[1] >> 2)
  }

  /// `value·r` for a value with a top word <= 4, normalized; used to derive key powers.
  #[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_endian = "little", target_feature = "neon")
  ))]
  #[inline(always)]
  pub(super) fn times_r(&self, value: [u64; 3]) -> [u64; 3] {
    let [r0, r1] = self.r;
    normalize(absorb_block(value, 0, 0, 0, r0, r1, self.s1()))
  }

  /// Absorbs full blocks, each with its 2^128 bit, one after another.
  #[inline(always)]
  pub(crate) fn absorb_blocks(&mut self, blocks: &[[u8; 16]]) {
    let [r0, r1] = self.r;
    let s1 = self.s1();
    let mut acc = self.h;
    for block in blocks {
      acc = absorb_block(
        acc,
        load_u64_le(&block[0..8]),
        load_u64_le(&block[8..16]),
        1,
        r0,
        r1,
        s1,
      );
    }
    self.h = normalize(acc);
  }

  /// Absorbs full blocks with four independent Horner chains over contiguous regions.
  ///
  /// One chain's blocks depend on each other through the multiply, so a single chain runs at the
  /// multiplier's latency. Four chains over four regions keep the multipliers busy instead. Chain
  /// `c` sums its region's blocks with the powers of `r` that end at the region's last block; the
  /// running accumulator enters chain 0. The chains then combine as
  /// `((a·r^l + b)·r^l + c)·r^(l+e) + d`, where `l` is the length of the first three regions and the
  /// last region holds `e < 4` more blocks. Only the public block count selects the split.
  #[inline]
  pub(crate) fn absorb_blocks_wide(&mut self, blocks: &[[u8; 16]]) {
    let len = blocks.len() / CHAINS;
    if len == 0 {
      self.absorb_blocks(blocks);
      return;
    }
    let [r0, r1] = self.r;
    let s1 = self.s1();
    // `powers` holds r, r^l, and r^(l+e); they are cleared before return.
    let mut powers = [radix26::from_words(r0, r1, 0); 3];
    powers[1] = power(powers[0], len);
    let (first, rest) = blocks.split_at(len);
    let (second, rest) = rest.split_at(len);
    let (third, fourth) = rest.split_at(len);
    let extra = fourth.len().strict_sub(len);
    powers[2] = if extra == 0 {
      powers[1]
    } else {
      radix26::mul(powers[1], power(powers[0], extra))
    };

    let mut acc = [self.h, [0; 3], [0; 3], [0; 3]];
    for (((a, b), c), d) in first.iter().zip(second).zip(third).zip(fourth) {
      acc[0] = absorb_block(acc[0], load_u64_le(&a[0..8]), load_u64_le(&a[8..16]), 1, r0, r1, s1);
      acc[1] = absorb_block(acc[1], load_u64_le(&b[0..8]), load_u64_le(&b[8..16]), 1, r0, r1, s1);
      acc[2] = absorb_block(acc[2], load_u64_le(&c[0..8]), load_u64_le(&c[8..16]), 1, r0, r1, s1);
      acc[3] = absorb_block(acc[3], load_u64_le(&d[0..8]), load_u64_le(&d[8..16]), 1, r0, r1, s1);
    }
    for d in &fourth[len..] {
      acc[3] = absorb_block(acc[3], load_u64_le(&d[0..8]), load_u64_le(&d[8..16]), 1, r0, r1, s1);
    }

    let [a, b, c, d] = acc.map(normalize);
    let ab = sum(times(a, powers[1]), b);
    let abc = sum(times(ab, powers[1]), c);
    self.h = sum(times(abc, powers[2]), d);
    for limbs in &mut powers {
      ct::zeroize_words_no_fence(limbs);
    }
    for words in &mut acc {
      ct::zeroize_words_no_fence(words);
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }

  /// Absorbs a final fragment of 1 to 15 message bytes, framed with a byte of one and zeros and
  /// without the 2^128 bit.
  #[inline(always)]
  pub(crate) fn absorb_final_bytes(&mut self, fragment: &[u8]) {
    let [r0, r1] = self.r;
    let (m0, m1) = fragment_words(fragment);
    self.h = normalize(absorb_block(self.h, m0, m1, 0, r0, r1, self.s1()));
  }

  /// Absorbs a caller-framed final fragment: its bytes, a byte of one, then zeros, without the 2^128 bit.
  #[inline(always)]
  pub(crate) fn absorb_final_fragment(&mut self, block: &[u8; 16]) {
    let [r0, r1] = self.r;
    self.h = normalize(absorb_block(
      self.h,
      load_u64_le(&block[0..8]),
      load_u64_le(&block[8..16]),
      0,
      r0,
      r1,
      self.s1(),
    ));
  }

  /// `(h mod (2^130 - 5) + pad) mod 2^128`, without consuming the owner.
  #[inline(always)]
  pub(crate) fn tag(&self) -> [u8; 16] {
    let [h0, h1, h2] = self.h;
    debug_assert!(h2 <= 4);
    // h < 5·2^128 < 2·p, so one conditional subtraction of p is a full reduction. With
    // g = h + 5, g >= 2^130 exactly when h >= p, and then g - 2^130 = h - p.
    let (g0, carry) = h0.overflowing_add(5);
    let (g1, carry) = h1.overflowing_add(u64::from(carry));
    // g2 <= 5, so g2 >> 2 is zero or one.
    let g2 = h2.wrapping_add(u64::from(carry));
    let take_g = (g2 >> 2).wrapping_neg();
    let f0 = (h0 & !take_g) | (g0 & take_g);
    let f1 = (h1 & !take_g) | (g1 & take_g);

    let (t0, carry) = f0.overflowing_add(self.pad[0]);
    let t1 = f1.wrapping_add(self.pad[1]).wrapping_add(u64::from(carry));
    let mut tag = [0u8; 16];
    tag[0..8].copy_from_slice(&t0.to_le_bytes());
    tag[8..16].copy_from_slice(&t1.to_le_bytes());
    tag
  }
}
