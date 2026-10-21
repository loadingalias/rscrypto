//! P-256 arithmetic for public inputs: ECDSA verification and public-key comb tables.
//!
//! Peer keys, signatures, digests and the tables derived from them are public, so this module
//! may branch on point values and scalar digits. Its field and point operations are specialized
//! to P-256 and inlined into the comb loops, instead of dispatching each field operation through
//! the generic modulus. It keeps the generic arithmetic's Montgomery representation (`R = 2^256`)
//! and point formulas, so tables match limb for limb and verification makes the same decisions.

use super::{
  Affine, COMB_TABLE_SIZE, COMB_WIDTH, FieldElement, Jacobian, P256_FIELD, P256_FIELD_MODULUS, P256_SIGNING_COMB_ROWS,
  P256_SIGNING_COMB_WIDTH, P256_SIGNING_GENERATOR_COMB_X, P256_SIGNING_GENERATOR_COMB_Y, Uint, comb_digit, comb_rows,
};

/// A P-256 field element in Montgomery form, as canonical limbs below p.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Fe([u64; 4]);

#[inline(always)]
fn adc(lhs: u64, rhs: u64, carry: u64) -> (u64, u64) {
  let (sum, overflow0) = lhs.overflowing_add(rhs);
  let (sum, overflow1) = sum.overflowing_add(carry);
  (sum, u64::from(overflow0 | overflow1))
}

#[inline(always)]
fn sbb(lhs: u64, rhs: u64, borrow: u64) -> (u64, u64) {
  let (difference, overflow0) = lhs.overflowing_sub(rhs);
  let (difference, overflow1) = difference.overflowing_sub(borrow);
  (difference, u64::from(overflow0 | overflow1))
}

/// `acc + lhs·rhs + carry` as `(low, high)`; the sum is at most `2^128 - 1`.
#[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
#[inline(always)]
fn mac(acc: u64, lhs: u64, rhs: u64, carry: u64) -> (u64, u64) {
  let (low, high) = super::split_u128(u128::from(lhs).strict_mul(u128::from(rhs)));
  let (low, carry0) = adc(low, acc, 0);
  let (low, carry1) = adc(low, carry, 0);
  (low, high.wrapping_add(carry0).wrapping_add(carry1))
}

impl Fe {
  const ZERO: Self = Self([0; 4]);

  #[inline(always)]
  fn from_element(value: FieldElement<4>) -> Self {
    Self(value.value.0)
  }

  #[inline(always)]
  fn to_element(self) -> FieldElement<4> {
    FieldElement::from_montgomery(Uint(self.0), &P256_FIELD_MODULUS)
  }

  #[inline(always)]
  fn is_zero(self) -> bool {
    self.0 == [0; 4]
  }

  #[inline(always)]
  fn add(self, rhs: Self) -> Self {
    let [a0, a1, a2, a3] = self.0;
    let [b0, b1, b2, b3] = rhs.0;
    let (s0, carry) = adc(a0, b0, 0);
    let (s1, carry) = adc(a1, b1, carry);
    let (s2, carry) = adc(a2, b2, carry);
    let (s3, carry) = adc(a3, b3, carry);
    Self::reduce_once([s0, s1, s2, s3, carry])
  }

  #[inline(always)]
  fn sub(self, rhs: Self) -> Self {
    let [a0, a1, a2, a3] = self.0;
    let [b0, b1, b2, b3] = rhs.0;
    let (d0, borrow) = sbb(a0, b0, 0);
    let (d1, borrow) = sbb(a1, b1, borrow);
    let (d2, borrow) = sbb(a2, b2, borrow);
    let (d3, borrow) = sbb(a3, b3, borrow);
    // Add p back when the difference wrapped.
    let mask = 0u64.wrapping_sub(borrow);
    let p = P256_FIELD.0;
    let (r0, carry) = adc(d0, p[0] & mask, 0);
    let (r1, carry) = adc(d1, p[1] & mask, carry);
    let (r2, carry) = adc(d2, p[2] & mask, carry);
    let (r3, _) = adc(d3, p[3] & mask, carry);
    Self([r0, r1, r2, r3])
  }

  #[inline(always)]
  fn double(self) -> Self {
    self.add(self)
  }

  #[inline(always)]
  fn triple(self) -> Self {
    self.double().add(self)
  }

  /// Subtract p once from a value below 2p.
  #[inline(always)]
  fn reduce_once(limbs: [u64; 5]) -> Self {
    let p = P256_FIELD.0;
    let (w0, borrow) = sbb(limbs[0], p[0], 0);
    let (w1, borrow) = sbb(limbs[1], p[1], borrow);
    let (w2, borrow) = sbb(limbs[2], p[2], borrow);
    let (w3, borrow) = sbb(limbs[3], p[3], borrow);
    let (_, borrow) = sbb(limbs[4], 0, borrow);
    // Additions overflow p about half the time, so select without a branch.
    let keep = 0u64.wrapping_sub(borrow);
    Self([
      w0 ^ ((w0 ^ limbs[0]) & keep),
      w1 ^ ((w1 ^ limbs[1]) & keep),
      w2 ^ ((w2 ^ limbs[2]) & keep),
      w3 ^ ((w3 ^ limbs[3]) & keep),
    ])
  }

  #[inline(always)]
  fn mul(self, rhs: Self) -> Self {
    #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
    {
      Self(aarch64::mul(self.0, rhs.0))
    }
    #[cfg(not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri))))]
    {
      self.mul_portable(rhs)
    }
  }

  #[inline(always)]
  fn square(self) -> Self {
    #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
    {
      Self(aarch64::square(self.0))
    }
    #[cfg(not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri))))]
    {
      self.square_portable()
    }
  }

  #[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
  #[inline(always)]
  fn mul_portable(self, rhs: Self) -> Self {
    let [a0, a1, a2, a3] = self.0;
    let [b0, b1, b2, b3] = rhs.0;
    let (w0, carry) = mac(0, a0, b0, 0);
    let (w1, carry) = mac(0, a0, b1, carry);
    let (w2, carry) = mac(0, a0, b2, carry);
    let (w3, w4) = mac(0, a0, b3, carry);

    let (w1, carry) = mac(w1, a1, b0, 0);
    let (w2, carry) = mac(w2, a1, b1, carry);
    let (w3, carry) = mac(w3, a1, b2, carry);
    let (w4, w5) = mac(w4, a1, b3, carry);

    let (w2, carry) = mac(w2, a2, b0, 0);
    let (w3, carry) = mac(w3, a2, b1, carry);
    let (w4, carry) = mac(w4, a2, b2, carry);
    let (w5, w6) = mac(w5, a2, b3, carry);

    let (w3, carry) = mac(w3, a3, b0, 0);
    let (w4, carry) = mac(w4, a3, b1, carry);
    let (w5, carry) = mac(w5, a3, b2, carry);
    let (w6, w7) = mac(w6, a3, b3, carry);

    Self::reduce([w0, w1, w2, w3, w4, w5, w6, w7])
  }

  #[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
  #[inline(always)]
  fn square_portable(self) -> Self {
    let [a0, a1, a2, a3] = self.0;
    let (w1, carry) = mac(0, a0, a1, 0);
    let (w2, carry) = mac(0, a0, a2, carry);
    let (w3, w4) = mac(0, a0, a3, carry);
    let (w3, carry) = mac(w3, a1, a2, 0);
    let (w4, w5) = mac(w4, a1, a3, carry);
    let (w5, w6) = mac(w5, a2, a3, 0);

    // The cross terms are below 2^448, so the doubled sum keeps its top bit in w7.
    let w7 = w6 >> 63;
    let w6 = (w6 << 1) | (w5 >> 63);
    let w5 = (w5 << 1) | (w4 >> 63);
    let w4 = (w4 << 1) | (w3 >> 63);
    let w3 = (w3 << 1) | (w2 >> 63);
    let w2 = (w2 << 1) | (w1 >> 63);
    let w1 = w1 << 1;

    let (s0_low, s0_high) = super::split_u128(u128::from(a0).strict_mul(u128::from(a0)));
    let (s1_low, s1_high) = super::split_u128(u128::from(a1).strict_mul(u128::from(a1)));
    let (s2_low, s2_high) = super::split_u128(u128::from(a2).strict_mul(u128::from(a2)));
    let (s3_low, s3_high) = super::split_u128(u128::from(a3).strict_mul(u128::from(a3)));
    let (w1, carry) = adc(w1, s0_high, 0);
    let (w2, carry) = adc(w2, s1_low, carry);
    let (w3, carry) = adc(w3, s1_high, carry);
    let (w4, carry) = adc(w4, s2_low, carry);
    let (w5, carry) = adc(w5, s2_high, carry);
    let (w6, carry) = adc(w6, s3_low, carry);
    let (w7, _) = adc(w7, s3_high, carry);

    Self::reduce([s0_low, w1, w2, w3, w4, w5, w6, w7])
  }

  #[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
  /// Montgomery-reduce a 512-bit product: `-p⁻¹ mod 2^64 = 1`, so each step's factor is the
  /// low limb, and `p[1] = 2^32 - 1`, `p[2] = 0` and `p[3] = 2^64 - 2^32 + 1` turn its products
  /// into shifts and additions, as in `super::montgomery_reduce_p256_field`.
  #[inline(always)]
  fn reduce(limbs: [u64; 8]) -> Self {
    let [r0, r1, r2, r3, r4, r5, r6, r7] = limbs;

    let (r1, carry) = limb_1_plus_value(r1, r0);
    let (r2, carry) = adc(r2, 0, carry);
    let (r3, carry) = limb_3(r3, r0, carry);
    let (r4, carry2) = adc(r4, 0, carry);

    let (r2, carry) = limb_1_plus_value(r2, r1);
    let (r3, carry) = adc(r3, 0, carry);
    let (r4, carry) = limb_3(r4, r1, carry);
    let (r5, carry2) = adc(r5, carry2, carry);

    let (r3, carry) = limb_1_plus_value(r3, r2);
    let (r4, carry) = adc(r4, 0, carry);
    let (r5, carry) = limb_3(r5, r2, carry);
    let (r6, carry2) = adc(r6, carry2, carry);

    let (r4, carry) = limb_1_plus_value(r4, r3);
    let (r5, carry) = adc(r5, 0, carry);
    let (r6, carry) = limb_3(r6, r3, carry);
    let (r7, r8) = adc(r7, carry2, carry);

    Self::reduce_once([r4, r5, r6, r7, r8])
  }
}

/// AArch64 field multiplication and squaring with flag-chained carries.
///
/// The portable [`Fe::mul`] and [`Fe::square`] are the authority; these compute the same
/// canonical Montgomery products. The four product rows are independent carry chains summed as a
/// tree, and each reduction round's carry joins the next round's high word. That word,
/// `⌊m·(2^64 - 2^32 + 1) / 2^64⌋`, is at most `2^64 - 2^32`, so adding a carry cannot overflow.
#[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
mod aarch64 {
  use core::arch::asm;

  /// `a·b·2^-256 mod p` for canonical inputs.
  #[inline(always)]
  pub(super) fn mul(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let (r0, r1, r2, r3): (u64, u64, u64, u64);
    // SAFETY: The block reads and writes only its register operands, uses base A64 integer
    // instructions that need no target feature, and touches neither memory nor the stack.
    unsafe {
      asm!(
      // Row 0: s0..s4 = a·b0.
      "mul {s0}, {a0}, {b0}",
      "umulh {t0}, {a0}, {b0}",
      "mul {s1}, {a1}, {b0}",
      "umulh {t1}, {a1}, {b0}",
      "mul {s2}, {a2}, {b0}",
      "umulh {t2}, {a2}, {b0}",
      "mul {s3}, {a3}, {b0}",
      "umulh {s4}, {a3}, {b0}",
      "adds {s1}, {s1}, {t0}",
      "adcs {s2}, {s2}, {t1}",
      "adcs {s3}, {s3}, {t2}",
      "adc {s4}, {s4}, xzr",
      // Row 1: r0..r4 = a·b1.
      "mul {r0}, {a0}, {b1}",
      "umulh {u0}, {a0}, {b1}",
      "mul {r1}, {a1}, {b1}",
      "umulh {u1}, {a1}, {b1}",
      "mul {r2}, {a2}, {b1}",
      "umulh {u2}, {a2}, {b1}",
      "mul {r3}, {a3}, {b1}",
      "umulh {r4}, {a3}, {b1}",
      "adds {r1}, {r1}, {u0}",
      "adcs {r2}, {r2}, {u1}",
      "adcs {r3}, {r3}, {u2}",
      "adc {r4}, {r4}, xzr",
      // Rows 0 and 1: s0..s5 = s + r·2^64.
      "adds {s1}, {s1}, {r0}",
      "adcs {s2}, {s2}, {r1}",
      "adcs {s3}, {s3}, {r2}",
      "adcs {s4}, {s4}, {r3}",
      "adc {s5}, {r4}, xzr",
      // Row 2: r0..r4 = a·b2.
      "mul {r0}, {a0}, {b2}",
      "umulh {t0}, {a0}, {b2}",
      "mul {r1}, {a1}, {b2}",
      "umulh {t1}, {a1}, {b2}",
      "mul {r2}, {a2}, {b2}",
      "umulh {t2}, {a2}, {b2}",
      "mul {r3}, {a3}, {b2}",
      "umulh {r4}, {a3}, {b2}",
      "adds {r1}, {r1}, {t0}",
      "adcs {r2}, {r2}, {t1}",
      "adcs {r3}, {r3}, {t2}",
      "adc {r4}, {r4}, xzr",
      // Row 3: u0..u4 = a·b3.
      "mul {u0}, {a0}, {b3}",
      "umulh {t0}, {a0}, {b3}",
      "mul {u1}, {a1}, {b3}",
      "umulh {t1}, {a1}, {b3}",
      "mul {u2}, {a2}, {b3}",
      "umulh {t2}, {a2}, {b3}",
      "mul {u3}, {a3}, {b3}",
      "umulh {u4}, {a3}, {b3}",
      "adds {u1}, {u1}, {t0}",
      "adcs {u2}, {u2}, {t1}",
      "adcs {u3}, {u3}, {t2}",
      "adc {u4}, {u4}, xzr",
      // Rows 2 and 3: r0..r4, t0 = r + u·2^64.
      "adds {r1}, {r1}, {u0}",
      "adcs {r2}, {r2}, {u1}",
      "adcs {r3}, {r3}, {u2}",
      "adcs {r4}, {r4}, {u3}",
      "adc {t0}, {u4}, xzr",
      // Product: s0..s5, u0, u1 = s + r·2^128.
      "adds {s2}, {s2}, {r0}",
      "adcs {s3}, {s3}, {r1}",
      "adcs {s4}, {s4}, {r2}",
      "adcs {s5}, {s5}, {r3}",
      "adcs {u0}, {r4}, xzr",
      "adc {u1}, {t0}, xzr",
      // Reduction 0: m = s0; add m·2^32 at limb 1 and m·(2^64 - 2^32 + 1) at limb 3.
      "lsl {t0}, {s0}, #32",
      "lsr {t1}, {s0}, #32",
      "subs {t2}, {s0}, {t0}",
      "sbc {r0}, {s0}, {t1}",
      "adds {s1}, {s1}, {t0}",
      "adcs {s2}, {s2}, {t1}",
      "adcs {s3}, {s3}, {t2}",
      "adcs {s4}, {s4}, {r0}",
      "adc {r1}, xzr, xzr",
      // Reduction 1: m = s1; the previous carry joins the high word, which stays below 2^64.
      "lsl {t0}, {s1}, #32",
      "lsr {t1}, {s1}, #32",
      "subs {t2}, {s1}, {t0}",
      "sbc {r0}, {s1}, {t1}",
      "add {r0}, {r0}, {r1}",
      "adds {s2}, {s2}, {t0}",
      "adcs {s3}, {s3}, {t1}",
      "adcs {s4}, {s4}, {t2}",
      "adcs {s5}, {s5}, {r0}",
      "adc {r2}, xzr, xzr",
      // Reduction 2: m = s2.
      "lsl {t0}, {s2}, #32",
      "lsr {t1}, {s2}, #32",
      "subs {t2}, {s2}, {t0}",
      "sbc {r0}, {s2}, {t1}",
      "add {r0}, {r0}, {r2}",
      "adds {s3}, {s3}, {t0}",
      "adcs {s4}, {s4}, {t1}",
      "adcs {s5}, {s5}, {t2}",
      "adcs {u0}, {u0}, {r0}",
      "adc {r3}, xzr, xzr",
      // Reduction 3: m = s3, leaving s4, s5, u0, u1, r4 below 2p.
      "lsl {t0}, {s3}, #32",
      "lsr {t1}, {s3}, #32",
      "subs {t2}, {s3}, {t0}",
      "sbc {r0}, {s3}, {t1}",
      "add {r0}, {r0}, {r3}",
      "adds {s4}, {s4}, {t0}",
      "adcs {s5}, {s5}, {t1}",
      "adcs {u0}, {u0}, {t2}",
      "adcs {u1}, {u1}, {r0}",
      "adc {r4}, xzr, xzr",
      // Subtract p unless that borrows.
      "mov {t1}, #0xffffffff",
      "mov {t3}, #0xffffffff00000001",
      "adds {t0}, {s4}, #1",
      "sbcs {t1}, {s5}, {t1}",
      "sbcs {t2}, {u0}, xzr",
      "sbcs {t3}, {u1}, {t3}",
      "sbcs xzr, {r4}, xzr",
      "csel {s4}, {s4}, {t0}, lo",
      "csel {s5}, {s5}, {t1}, lo",
      "csel {u0}, {u0}, {t2}, lo",
      "csel {u1}, {u1}, {t3}, lo",
        a0 = in(reg) a[0], a1 = in(reg) a[1], a2 = in(reg) a[2], a3 = in(reg) a[3],
        b0 = in(reg) b[0], b1 = in(reg) b[1], b2 = in(reg) b[2], b3 = in(reg) b[3],
        s0 = out(reg) _, s1 = out(reg) _, s2 = out(reg) _, s3 = out(reg) _,
        s4 = out(reg) r0, s5 = out(reg) r1, u0 = out(reg) r2, u1 = out(reg) r3,
        u2 = out(reg) _, u3 = out(reg) _, u4 = out(reg) _,
        r0 = out(reg) _, r1 = out(reg) _, r2 = out(reg) _, r3 = out(reg) _, r4 = out(reg) _,
        t0 = out(reg) _, t1 = out(reg) _, t2 = out(reg) _, t3 = out(reg) _,
        options(pure, nomem, nostack),
      );
    }
    [r0, r1, r2, r3]
  }

  /// `a²·2^-256 mod p` for a canonical input.
  #[inline(always)]
  pub(super) fn square(a: [u64; 4]) -> [u64; 4] {
    let (r0, r1, r2, r3): (u64, u64, u64, u64);
    // SAFETY: As for `mul`: register operands only, base A64 integer instructions, no memory or
    // stack access. The input registers are reused as scratch after their last read.
    unsafe {
      asm!(
      // Cross row a0·[a1, a2, a3] at limbs 1..4.
      "mul {x1}, {a0}, {a1}",
      "umulh {t0}, {a0}, {a1}",
      "mul {x2}, {a0}, {a2}",
      "umulh {t1}, {a0}, {a2}",
      "mul {x3}, {a0}, {a3}",
      "umulh {x4}, {a0}, {a3}",
      "adds {x2}, {x2}, {t0}",
      "adcs {x3}, {x3}, {t1}",
      "adc {x4}, {x4}, xzr",
      // Cross row a1·[a2, a3] at limbs 3..5.
      "mul {y3}, {a1}, {a2}",
      "umulh {t0}, {a1}, {a2}",
      "mul {y4}, {a1}, {a3}",
      "umulh {y5}, {a1}, {a3}",
      "adds {y4}, {y4}, {t0}",
      "adc {y5}, {y5}, xzr",
      // Cross product a2·a3 at limbs 5..6.
      "mul {z5}, {a2}, {a3}",
      "umulh {z6}, {a2}, {a3}",
      // Cross sum at limbs 1..6, carry into limb 7.
      "adds {x3}, {x3}, {y3}",
      "adcs {x4}, {x4}, {y4}",
      "adcs {y5}, {y5}, {z5}",
      "adc {z6}, {z6}, xzr",
      // Double the cross sum.
      "lsr {c7}, {z6}, #63",
      "extr {z6}, {z6}, {y5}, #63",
      "extr {y5}, {y5}, {x4}, #63",
      "extr {x4}, {x4}, {x3}, #63",
      "extr {x3}, {x3}, {x2}, #63",
      "extr {x2}, {x2}, {x1}, #63",
      "lsl {x1}, {x1}, #1",
      // Squares a_i² on the diagonal.
      "mul {d0}, {a0}, {a0}",
      "umulh {t0}, {a0}, {a0}",
      "mul {t1}, {a1}, {a1}",
      "umulh {y3}, {a1}, {a1}",
      "mul {y4}, {a2}, {a2}",
      "umulh {z5}, {a2}, {a2}",
      "mul {a0}, {a3}, {a3}",
      "umulh {a1}, {a3}, {a3}",
      // Product: d0, x1..x4, y5, z6, c7.
      "adds {x1}, {x1}, {t0}",
      "adcs {x2}, {x2}, {t1}",
      "adcs {x3}, {x3}, {y3}",
      "adcs {x4}, {x4}, {y4}",
      "adcs {y5}, {y5}, {z5}",
      "adcs {z6}, {z6}, {a0}",
      "adc {c7}, {c7}, {a1}",
      // Reduction 0: m = d0, as in the multiplication.
      "lsl {t0}, {d0}, #32",
      "lsr {t1}, {d0}, #32",
      "subs {y3}, {d0}, {t0}",
      "sbc {y4}, {d0}, {t1}",
      "adds {x1}, {x1}, {t0}",
      "adcs {x2}, {x2}, {t1}",
      "adcs {x3}, {x3}, {y3}",
      "adcs {x4}, {x4}, {y4}",
      "adc {z5}, xzr, xzr",
      // Reduction 1.
      "lsl {t0}, {x1}, #32",
      "lsr {t1}, {x1}, #32",
      "subs {y3}, {x1}, {t0}",
      "sbc {y4}, {x1}, {t1}",
      "add {y4}, {y4}, {z5}",
      "adds {x2}, {x2}, {t0}",
      "adcs {x3}, {x3}, {t1}",
      "adcs {x4}, {x4}, {y3}",
      "adcs {y5}, {y5}, {y4}",
      "adc {a2}, xzr, xzr",
      // Reduction 2.
      "lsl {t0}, {x2}, #32",
      "lsr {t1}, {x2}, #32",
      "subs {y3}, {x2}, {t0}",
      "sbc {y4}, {x2}, {t1}",
      "add {y4}, {y4}, {a2}",
      "adds {x3}, {x3}, {t0}",
      "adcs {x4}, {x4}, {t1}",
      "adcs {y5}, {y5}, {y3}",
      "adcs {z6}, {z6}, {y4}",
      "adc {a3}, xzr, xzr",
      // Reduction 3.
      "lsl {t0}, {x3}, #32",
      "lsr {t1}, {x3}, #32",
      "subs {y3}, {x3}, {t0}",
      "sbc {y4}, {x3}, {t1}",
      "add {y4}, {y4}, {a3}",
      "adds {x4}, {x4}, {t0}",
      "adcs {y5}, {y5}, {t1}",
      "adcs {z6}, {z6}, {y3}",
      "adcs {c7}, {c7}, {y4}",
      "adc {a0}, xzr, xzr",
      // Subtract p unless that borrows.
      "mov {t1}, #0xffffffff",
      "mov {a1}, #0xffffffff00000001",
      "adds {t0}, {x4}, #1",
      "sbcs {t1}, {y5}, {t1}",
      "sbcs {y3}, {z6}, xzr",
      "sbcs {y4}, {c7}, {a1}",
      "sbcs xzr, {a0}, xzr",
      "csel {x4}, {x4}, {t0}, lo",
      "csel {y5}, {y5}, {t1}, lo",
      "csel {z6}, {z6}, {y3}, lo",
      "csel {c7}, {c7}, {y4}, lo",
        a0 = inout(reg) a[0] => _, a1 = inout(reg) a[1] => _,
        a2 = inout(reg) a[2] => _, a3 = inout(reg) a[3] => _,
        d0 = out(reg) _, x1 = out(reg) _, x2 = out(reg) _, x3 = out(reg) _,
        x4 = out(reg) r0, y5 = out(reg) r1, z6 = out(reg) r2, c7 = out(reg) r3,
        y3 = out(reg) _, y4 = out(reg) _, z5 = out(reg) _,
        t0 = out(reg) _, t1 = out(reg) _,
        options(pure, nomem, nostack),
      );
    }
    [r0, r1, r2, r3]
  }
}

/// `acc + value·p[1] + value`, which is `acc + value·2^32`.
#[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
#[inline(always)]
fn limb_1_plus_value(acc: u64, value: u64) -> (u64, u64) {
  let (low, carry) = adc(acc, value << 32, 0);
  (low, (value >> 32).wrapping_add(carry))
}

/// `acc + value·p[3] + carry` for `p[3] = 2^64 - 2^32 + 1`; the sum is below 2^128.
#[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
#[inline(always)]
fn limb_3(acc: u64, value: u64, carry: u64) -> (u64, u64) {
  // value·p[3] = value·2^64 - value·2^32 + value, as a 128-bit (high, low) pair.
  let (product_low, borrow) = sbb(0, value << 32, 0);
  let product_high = value.wrapping_sub(value >> 32).wrapping_sub(borrow);
  let (product_low, product_carry) = adc(product_low, value, 0);
  let product_high = product_high.wrapping_add(product_carry);
  let (result, carry0) = adc(product_low, acc, 0);
  let (result, carry1) = adc(result, carry, 0);
  (result, product_high.wrapping_add(carry0).wrapping_add(carry1))
}

#[derive(Clone, Copy)]
struct Point {
  x: Fe,
  y: Fe,
  z: Fe,
  infinity: bool,
}

#[derive(Clone, Copy)]
struct AffinePoint {
  x: Fe,
  y: Fe,
}

impl AffinePoint {
  #[inline(always)]
  fn from_affine(point: Affine<4>) -> Self {
    Self {
      x: Fe::from_element(point.x),
      y: Fe::from_element(point.y),
    }
  }

  #[inline(always)]
  fn to_affine(self) -> Affine<4> {
    Affine {
      x: self.x.to_element(),
      y: self.y.to_element(),
    }
  }
}

impl Point {
  const INFINITY: Self = Self {
    x: Fe::ZERO,
    y: Fe::ZERO,
    z: Fe::ZERO,
    infinity: true,
  };

  #[inline(always)]
  fn from_affine(point: AffinePoint) -> Self {
    Self {
      x: point.x,
      y: point.y,
      z: Fe(FIELD_ONE),
      infinity: false,
    }
  }

  /// `2·self` with `a = -3` (dbl-2001-b), as `Jacobian::double` computes it.
  #[inline(always)]
  fn double(self) -> Self {
    if self.infinity || self.y.is_zero() {
      return Self::INFINITY;
    }
    let delta = self.z.square();
    let gamma = self.y.square();
    let beta = self.x.mul(gamma);
    let alpha = self.x.sub(delta).mul(self.x.add(delta)).triple();
    let x3 = alpha.square().sub(beta.double().double().double());
    let z3 = self.y.add(self.z).square().sub(gamma).sub(delta);
    let y3 = alpha
      .mul(beta.double().double().sub(x3))
      .sub(gamma.square().double().double().double());
    Self {
      x: x3,
      y: y3,
      z: z3,
      infinity: false,
    }
  }

  /// `self + rhs` for affine `rhs` (madd-2007-bl), as `Jacobian::add_mixed` computes it,
  /// including the doubling and inverse cases.
  #[inline(always)]
  fn add_mixed(self, rhs: AffinePoint) -> Self {
    if self.infinity {
      return Self::from_affine(rhs);
    }
    let z1z1 = self.z.square();
    let u2 = rhs.x.mul(z1z1);
    let s2 = rhs.y.mul(self.z).mul(z1z1);
    let h = u2.sub(self.x);
    let r = s2.sub(self.y).double();
    if h.is_zero() {
      return if r.is_zero() { self.double() } else { Self::INFINITY };
    }
    let hh = h.square();
    let i = hh.double().double();
    let j = h.mul(i);
    let v = self.x.mul(i);
    let x3 = r.square().sub(j).sub(v.double());
    let y3 = r.mul(v.sub(x3)).sub(self.y.mul(j).double());
    let z3 = self.z.add(h).square().sub(z1z1).sub(hh);
    Self {
      x: x3,
      y: y3,
      z: z3,
      infinity: false,
    }
  }

  /// `self + rhs` for Jacobian `rhs` (add-2007-bl), including the infinity, doubling and inverse
  /// cases.
  fn add(self, rhs: Self) -> Self {
    if self.infinity {
      return rhs;
    }
    if rhs.infinity {
      return self;
    }
    let z1z1 = self.z.square();
    let z2z2 = rhs.z.square();
    let u1 = self.x.mul(z2z2);
    let u2 = rhs.x.mul(z1z1);
    let s1 = self.y.mul(rhs.z).mul(z2z2);
    let s2 = rhs.y.mul(self.z).mul(z1z1);
    let h = u2.sub(u1);
    let r = s2.sub(s1).double();
    if h.is_zero() {
      return if r.is_zero() { self.double() } else { Self::INFINITY };
    }
    let i = h.double().square();
    let j = h.mul(i);
    let v = u1.mul(i);
    let x3 = r.square().sub(j).sub(v.double());
    let y3 = r.mul(v.sub(x3)).sub(s1.mul(j).double());
    let z3 = self.z.add(rhs.z).square().sub(z1z1).sub(z2z2).mul(h);
    Self {
      x: x3,
      y: y3,
      z: z3,
      infinity: false,
    }
  }

  #[inline(always)]
  fn to_jacobian(self) -> Jacobian<4> {
    Jacobian {
      x: self.x.to_element(),
      y: self.y.to_element(),
      z: self.z.to_element(),
      infinity: self.infinity,
    }
  }
}

/// One in Montgomery form: `2^256 mod p`.
const FIELD_ONE: [u64; 4] = [
  0x0000_0000_0000_0001,
  0xffff_ffff_0000_0000,
  0xffff_ffff_ffff_ffff,
  0x0000_0000_ffff_fffe,
];

/// Affine forms of finite points, with one field inversion for the whole table.
fn normalize<const N: usize>(table: [Point; N]) -> [AffinePoint; N] {
  let mut prefixes = [Fe(FIELD_ONE); N];
  let mut acc = Fe(FIELD_ONE);
  for (prefix, point) in prefixes.iter_mut().zip(&table) {
    *prefix = acc;
    acc = acc.mul(point.z);
  }
  let mut acc_inverse = Fe::from_element(acc.to_element().inv());
  let mut out = [AffinePoint {
    x: Fe::ZERO,
    y: Fe::ZERO,
  }; N];
  for ((point, prefix), output) in table.iter().zip(prefixes).zip(out.iter_mut()).rev() {
    let z_inverse = acc_inverse.mul(prefix);
    acc_inverse = acc_inverse.mul(point.z);
    let z2 = z_inverse.square();
    let z3 = z2.mul(z_inverse);
    *output = AffinePoint {
      x: point.x.mul(z2),
      y: point.y.mul(z3),
    };
  }
  out
}

/// The public-key comb table of `super::precompute_comb_table`: entry `mask` is the sum of the
/// columns `2^(64·c)·point` whose bit `c` is set, and entry zero is `point`.
pub(super) fn comb_table(point: Affine<4>) -> [Affine<4>; COMB_TABLE_SIZE] {
  let rows = comb_rows::<4>();
  let base = AffinePoint::from_affine(point);
  let mut columns = [Point::from_affine(base); COMB_WIDTH];
  let mut current = Point::from_affine(base);
  for column in columns.iter_mut().skip(1) {
    for _ in 0..rows {
      current = current.double();
    }
    *column = current;
  }
  let columns = normalize(columns);

  let mut table = [Point::from_affine(base); COMB_TABLE_SIZE];
  for (mask, entry) in table.iter_mut().enumerate().skip(1) {
    let mut acc = Point::INFINITY;
    for (column, &column_point) in columns.iter().enumerate() {
      if mask & (1usize << column) != 0 {
        acc = acc.add_mixed(column_point);
      }
    }
    *entry = acc;
  }
  normalize(table).map(AffinePoint::to_affine)
}

/// `generator_scalar·G + point_scalar·Q` with the signing generator comb and the public-key comb
/// table of Q, using the digits of `super::scalar_mul_two`.
///
/// The generator and key terms accumulate in separate chains, combined by one addition, so the
/// core overlaps their doublings and additions instead of serializing every addition on one
/// accumulator.
pub(super) fn scalar_mul_two(
  generator_scalar: Uint<4>,
  point_scalar: Uint<4>,
  point_table: &[Affine<4>; COMB_TABLE_SIZE],
) -> Jacobian<4> {
  let rows = comb_rows::<4>();
  let table = point_table.map(AffinePoint::from_affine);
  let mut generator = Point::INFINITY;
  let mut key = Point::INFINITY;
  for row in (0..rows).rev() {
    if row < P256_SIGNING_COMB_ROWS {
      generator = generator.double();
      let digit = comb_digit(generator_scalar, row, P256_SIGNING_COMB_ROWS, P256_SIGNING_COMB_WIDTH);
      if digit != 0 {
        generator = generator.add_mixed(AffinePoint {
          x: Fe(P256_SIGNING_GENERATOR_COMB_X[digit].0),
          y: Fe(P256_SIGNING_GENERATOR_COMB_Y[digit].0),
        });
      }
    }
    key = key.double();
    let digit = comb_digit(point_scalar, row, rows, COMB_WIDTH);
    if digit != 0 {
      key = key.add_mixed(table[digit]);
    }
  }
  generator.add(key).to_jacobian()
}

#[cfg(test)]
mod tests {
  use super::{Fe, P256_FIELD};

  #[test]
  fn dispatched_field_arithmetic_matches_portable_authority() {
    // Zero, one, p - 1 and values just below p in each limb, plus generated residues.
    let p = P256_FIELD.0;
    let edges = [
      [0u64; 4],
      [1, 0, 0, 0],
      [p[0] - 1, p[1], p[2], p[3]],
      [p[0], p[1] - 1, p[2], p[3]],
      [p[0], p[1], p[2], p[3] - 1],
      [u64::MAX, u64::MAX, u64::MAX, p[3] - 1],
      [0, 0, 0, p[3] - 1],
    ];
    let mut state = 0x6a09_e667_f3bc_c908u64;
    let mut next = || {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      state
    };
    let mut generated = || {
      let value = [next(), next(), next(), next()];
      Fe([value[0], value[1], value[2], value[3].min(p[3] - 1)])
    };
    let count = if cfg!(miri) { 16 } else { 20_000 };
    let mut previous = Fe(edges[2]);
    for index in 0..count + edges.len() {
      let a = edges.get(index).map_or_else(&mut generated, |&edge| Fe(edge));
      for b in [previous, Fe(edges[index % edges.len()])] {
        assert!(a.mul(b) == a.mul_portable(b), "{:x?} · {:x?}", a.0, b.0);
      }
      assert!(a.square() == a.square_portable(), "{:x?}²", a.0);
      assert!(a.square() == a.mul_portable(a), "{:x?}² as a product", a.0);
      previous = a;
    }
  }
}
