//! The P-384 field for public-input verification, in the generic Montgomery representation
//! (`R = 2^384`), for [`super::verify`]'s point layer.
//!
//! AArch64 uses the owned register-only kernels below. x86-64 Linux multiplies with the platform
//! kernels that the generic P-384 arithmetic already uses. Other targets use owned portable Rust.

use super::{Modulus, P384_FIELD_MODULUS, verify::VerifyField};

/// `p = 2^384 - 2^128 - 2^96 + 2^32 - 1`, least-significant limb first.
#[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
const P: [u64; 6] = [
  0x0000_0000_ffff_ffff,
  0xffff_ffff_0000_0000,
  0xffff_ffff_ffff_fffe,
  u64::MAX,
  u64::MAX,
  u64::MAX,
];

/// One in Montgomery form: `2^384 mod p = 2^128 + 2^96 - 2^32 + 1`.
const FIELD_ONE: [u64; 6] = [0xffff_ffff_0000_0001, 0x0000_0000_ffff_ffff, 1, 0, 0, 0];

/// A P-384 field element in Montgomery form, as canonical limbs below p.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Fe([u64; 6]);

#[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
#[inline(always)]
fn adc(lhs: u64, rhs: u64, carry: u64) -> (u64, u64) {
  let (sum, overflow0) = lhs.overflowing_add(rhs);
  let (sum, overflow1) = sum.overflowing_add(carry);
  (sum, u64::from(overflow0 | overflow1))
}

#[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
#[inline(always)]
fn sbb(lhs: u64, rhs: u64, borrow: u64) -> (u64, u64) {
  let (difference, overflow0) = lhs.overflowing_sub(rhs);
  let (difference, overflow1) = difference.overflowing_sub(borrow);
  (difference, u64::from(overflow0 | overflow1))
}

/// `acc + lhs·rhs + carry` as `(low, high)`; the sum is at most `2^128 - 1`. Every input here is
/// public, so the plain wide multiplication serves every target.
#[cfg(any(
  test,
  not(any(
    all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
    all(
      target_arch = "x86_64",
      target_os = "linux",
      not(feature = "portable-only"),
      not(miri)
    )
  ))
))]
#[inline(always)]
fn mac(acc: u64, lhs: u64, rhs: u64, carry: u64) -> (u64, u64) {
  let bytes = u128::from(lhs).strict_mul(u128::from(rhs)).to_le_bytes();
  let (low, high) = bytes.split_at(8);
  let low = u64::from_le_bytes(low.try_into().expect("eight low bytes"));
  let high = u64::from_le_bytes(high.try_into().expect("eight high bytes"));
  let (low, carry0) = adc(low, acc, 0);
  let (low, carry1) = adc(low, carry, 0);
  (low, high.wrapping_add(carry0).wrapping_add(carry1))
}

impl Fe {
  #[inline(always)]
  fn add(self, rhs: Self) -> Self {
    #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
    {
      Self(aarch64::add(self.0, rhs.0))
    }
    #[cfg(not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri))))]
    {
      self.add_portable(rhs)
    }
  }

  #[inline(always)]
  fn sub(self, rhs: Self) -> Self {
    #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
    {
      Self(aarch64::sub(self.0, rhs.0))
    }
    #[cfg(not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri))))]
    {
      self.sub_portable(rhs)
    }
  }

  #[inline(always)]
  fn mul(self, rhs: Self) -> Self {
    #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
    {
      Self(aarch64::mul(self.0, rhs.0))
    }
    #[cfg(all(
      target_arch = "x86_64",
      target_os = "linux",
      not(feature = "portable-only"),
      not(miri)
    ))]
    {
      Self(super::ecdsa_platform_asm::p384_field_mul(&self.0, &rhs.0))
    }
    #[cfg(not(any(
      all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
      all(
        target_arch = "x86_64",
        target_os = "linux",
        not(feature = "portable-only"),
        not(miri)
      )
    )))]
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
    #[cfg(all(
      target_arch = "x86_64",
      target_os = "linux",
      not(feature = "portable-only"),
      not(miri)
    ))]
    {
      Self(super::ecdsa_platform_asm::p384_field_square(&self.0))
    }
    #[cfg(not(any(
      all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
      all(
        target_arch = "x86_64",
        target_os = "linux",
        not(feature = "portable-only"),
        not(miri)
      )
    )))]
    {
      self.square_portable()
    }
  }

  /// Subtract p once from a value below 2p, held as six limbs and a top bit.
  #[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
  #[inline(always)]
  fn reduce_once(limbs: [u64; 6], top: u64) -> Self {
    let mut reduced = [0u64; 6];
    let mut borrow = 0u64;
    for ((out, &limb), &modulus) in reduced.iter_mut().zip(&limbs).zip(&P) {
      (*out, borrow) = sbb(limb, modulus, borrow);
    }
    let (_, borrow) = sbb(top, 0, borrow);
    // Additions overflow p about half the time, so select without a branch.
    let keep = 0u64.wrapping_sub(borrow);
    let mut out = [0u64; 6];
    for ((out, &limb), &reduced) in out.iter_mut().zip(&limbs).zip(&reduced) {
      *out = reduced ^ ((reduced ^ limb) & keep);
    }
    Self(out)
  }

  #[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
  #[inline(always)]
  fn add_portable(self, rhs: Self) -> Self {
    let mut sum = [0u64; 6];
    let mut carry = 0u64;
    for ((out, &a), &b) in sum.iter_mut().zip(&self.0).zip(&rhs.0) {
      (*out, carry) = adc(a, b, carry);
    }
    Self::reduce_once(sum, carry)
  }

  #[cfg(any(test, not(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))))]
  #[inline(always)]
  fn sub_portable(self, rhs: Self) -> Self {
    let mut difference = [0u64; 6];
    let mut borrow = 0u64;
    for ((out, &a), &b) in difference.iter_mut().zip(&self.0).zip(&rhs.0) {
      (*out, borrow) = sbb(a, b, borrow);
    }
    // Add p back when the difference wrapped.
    let mask = 0u64.wrapping_sub(borrow);
    let mut carry = 0u64;
    for (out, &modulus) in difference.iter_mut().zip(&P) {
      (*out, carry) = adc(*out, modulus & mask, carry);
    }
    Self(difference)
  }

  /// Montgomery multiplication `a·b·2^-384 mod p`: the full product by operand scanning, then
  /// [`Self::reduce`].
  #[cfg(any(
    test,
    not(any(
      all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
      all(
        target_arch = "x86_64",
        target_os = "linux",
        not(feature = "portable-only"),
        not(miri)
      )
    ))
  ))]
  #[inline(always)]
  fn mul_portable(self, rhs: Self) -> Self {
    let mut product = [0u64; 12];
    for (row, &b) in rhs.0.iter().enumerate() {
      let mut carry = 0u64;
      for (limb, &a) in product[row..].iter_mut().zip(&self.0) {
        (*limb, carry) = mac(*limb, a, b, carry);
      }
      // Earlier rows end below limb `row + 6`.
      product[row.strict_add(6)] = carry;
    }
    Self::reduce(product)
  }

  /// Montgomery squaring `a²·2^-384 mod p`: the 15 cross products once, doubled, plus the six
  /// diagonal squares, then [`Self::reduce`].
  #[cfg(any(
    test,
    not(any(
      all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
      all(
        target_arch = "x86_64",
        target_os = "linux",
        not(feature = "portable-only"),
        not(miri)
      )
    ))
  ))]
  #[inline(always)]
  fn square_portable(self) -> Self {
    let a = self.0;
    let mut product = [0u64; 12];
    // Row i adds a_i·a_j for j > i at limbs 2i + 1 through i + 5, and its carry starts limb i + 6.
    for (row, &ai) in a.iter().enumerate().take(5) {
      let next = row.strict_add(1);
      let mut carry = 0u64;
      for (limb, &aj) in product[row.strict_add(next)..].iter_mut().zip(&a[next..]) {
        (*limb, carry) = mac(*limb, ai, aj, carry);
      }
      product[row.strict_add(6)] = carry;
    }

    // The cross sum is below 2^767, so doubling it fits twelve limbs.
    let mut high_bit = 0u64;
    for limb in &mut product {
      let next = *limb >> 63;
      *limb = (*limb << 1) | high_bit;
      high_bit = next;
    }

    let mut carry = 0u64;
    for ([low, high], &ai) in product.as_chunks_mut::<2>().0.iter_mut().zip(&a) {
      let (square_low, square_high) = mac(0, ai, ai, 0);
      (*low, carry) = adc(*low, square_low, carry);
      (*high, carry) = adc(*high, square_high, carry);
    }
    Self::reduce(product)
  }

  /// Montgomery-reduce a product `T < p²` held in twelve limbs.
  ///
  /// With `T = T_hi·2^384 + T_lo`, the result is `T_hi + (T_lo + M·p)/2^384`, where six steps
  /// choose `M` to clear `T_lo` one limb at a time. The reduced low half is at most p and `T_hi`
  /// is below p, so one final subtraction suffices.
  ///
  /// Write `p = 2^384 - c` with `c = 2^128 + 2^96 - 2^32 + 1`; then `-p⁻¹ mod 2^64 = 2^32 + 1`.
  /// For a low limb `w`, the factor is `m = w + (w << 32)`, and the low limb of `m·c` is `w`.
  /// With `x = m << 32` and `y = m >> 32`, the rest of `m·c` is
  /// `(x - y - [x > m]) + 2^64·(m + y)`, so each step subtracts that from the next limbs and
  /// adds `m` at `2^384`, without a multiplication.
  #[cfg(any(
    test,
    not(any(
      all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)),
      all(
        target_arch = "x86_64",
        target_os = "linux",
        not(feature = "portable-only"),
        not(miri)
      )
    ))
  ))]
  #[inline(always)]
  fn reduce(product: [u64; 12]) -> Self {
    let [t0, t1, t2, t3, t4, t5, h0, h1, h2, h3, h4, h5] = product;
    // A step maps a running value `r < 2^385` to `(r + m·p)/2^64 < 2^321 + p < 2^385`, so it
    // fits six limbs and a top bit.
    let mut r = [t0, t1, t2, t3, t4, t5];
    let mut top = 0u64;
    for _ in 0..6 {
      let [w, r1, r2, r3, r4, r5] = r;
      let m = w.wrapping_add(w << 32);
      let x = m << 32;
      let y = m >> 32;
      let (d1, borrow) = sbb(x, y, u64::from(x > m));
      // A borrow needs `x = 0`, which leaves `y` positive.
      let (d2, d3) = adc(m, y.strict_sub(borrow), 0);

      // `(r - w)/2^64 + m·2^320 - (d1 + d2·2^64 + d3·2^128)`.
      let (n5, carry) = adc(top, m, 0);
      let (n0, borrow) = sbb(r1, d1, 0);
      let (n1, borrow) = sbb(r2, d2, borrow);
      let (n2, borrow) = sbb(r3, d3, borrow);
      let (n3, borrow) = sbb(r4, 0, borrow);
      let (n4, borrow) = sbb(r5, 0, borrow);
      let (n5, borrow) = sbb(n5, 0, borrow);
      // `r + m·p` is not negative, so the final borrow cannot exceed the carry.
      top = carry.strict_sub(borrow);
      r = [n0, n1, n2, n3, n4, n5];
    }

    // The reduced low half is at most p and `T_hi` is below p, so the sum is below 2p.
    let mut sum = [0u64; 6];
    let mut carry = 0u64;
    for ((out, &low), &high) in sum.iter_mut().zip(&r).zip(&[h0, h1, h2, h3, h4, h5]) {
      (*out, carry) = adc(low, high, carry);
    }
    Self::reduce_once(sum, carry.strict_add(top))
  }
}

impl VerifyField<6> for Fe {
  const ZERO: Self = Self([0; 6]);
  const ONE: Self = Self(FIELD_ONE);
  const MODULUS: &'static Modulus<6> = &P384_FIELD_MODULUS;

  #[inline(always)]
  fn from_limbs(limbs: [u64; 6]) -> Self {
    Self(limbs)
  }

  #[inline(always)]
  fn limbs(self) -> [u64; 6] {
    self.0
  }

  #[inline(always)]
  fn add(self, rhs: Self) -> Self {
    Self::add(self, rhs)
  }

  #[inline(always)]
  fn sub(self, rhs: Self) -> Self {
    Self::sub(self, rhs)
  }

  #[inline(always)]
  fn mul(self, rhs: Self) -> Self {
    Self::mul(self, rhs)
  }

  #[inline(always)]
  fn square(self) -> Self {
    Self::square(self)
  }
}

#[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
mod aarch64 {
  //! Register-only A64 kernels for the P-384 verification field.
  //!
  //! Montgomery reduction uses the shape of p. Write `p = 2^384 - c` with
  //! `c = 2^128 + 2^96 - 2^32 + 1`; then `-p⁻¹ mod 2^64 = 2^32 + 1`. For a low limb `w`, the
  //! multiplier is `m = w + (w << 32)`, and `w + m·p` clears that limb. With `x = m << 32` and
  //! `y = m >> 32`, the remaining part of `m·c` above the cleared limb is
  //! `(x - y - [x > m]) + 2^64·(m + y)`, so one step subtracts that from the next limbs and
  //! adds `m` five limbs higher, without a multiplication.

  use core::arch::asm;

  /// `a + b mod p` for canonical inputs: a flag-chained sum, then `p` subtracted unless that
  /// borrows.
  #[inline(always)]
  pub(super) fn add(a: [u64; 6], b: [u64; 6]) -> [u64; 6] {
    let (r0, r1, r2, r3, r4, r5): (u64, u64, u64, u64, u64, u64);
    // SAFETY: Register operands only, base A64 integer instructions, no memory or stack access.
    unsafe {
      asm!(
        "adds {a0}, {a0}, {b0}",
        "adcs {a1}, {a1}, {b1}",
        "adcs {a2}, {a2}, {b2}",
        "adcs {a3}, {a3}, {b3}",
        "adcs {a4}, {a4}, {b4}",
        "adcs {a5}, {a5}, {b5}",
        "cset {c}, cs",
        // p = [2^32 - 1, 2^64 - 2^32, 2^64 - 2, 2^64 - 1, 2^64 - 1, 2^64 - 1].
        "mov {b0}, #0xffffffff",
        "mov {b1}, #0xffffffff00000000",
        "mov {b2}, #0xfffffffffffffffe",
        "mov {b3}, #-1",
        "subs {t0}, {a0}, {b0}",
        "sbcs {t1}, {a1}, {b1}",
        "sbcs {t2}, {a2}, {b2}",
        "sbcs {t3}, {a3}, {b3}",
        "sbcs {t4}, {a4}, {b3}",
        "sbcs {t5}, {a5}, {b3}",
        "sbcs xzr, {c}, xzr",
        "csel {a0}, {a0}, {t0}, lo",
        "csel {a1}, {a1}, {t1}, lo",
        "csel {a2}, {a2}, {t2}, lo",
        "csel {a3}, {a3}, {t3}, lo",
        "csel {a4}, {a4}, {t4}, lo",
        "csel {a5}, {a5}, {t5}, lo",
        a0 = inout(reg) a[0] => r0, a1 = inout(reg) a[1] => r1, a2 = inout(reg) a[2] => r2,
        a3 = inout(reg) a[3] => r3, a4 = inout(reg) a[4] => r4, a5 = inout(reg) a[5] => r5,
        b0 = inout(reg) b[0] => _, b1 = inout(reg) b[1] => _, b2 = inout(reg) b[2] => _,
        b3 = inout(reg) b[3] => _, b4 = in(reg) b[4], b5 = in(reg) b[5],
        c = out(reg) _,
        t0 = out(reg) _, t1 = out(reg) _, t2 = out(reg) _, t3 = out(reg) _, t4 = out(reg) _, t5 = out(reg) _,
        options(pure, nomem, nostack),
      );
    }
    [r0, r1, r2, r3, r4, r5]
  }

  /// `a - b mod p` for canonical inputs: a flag-chained difference, then `p` added back under
  /// the borrow mask.
  #[inline(always)]
  pub(super) fn sub(a: [u64; 6], b: [u64; 6]) -> [u64; 6] {
    let (r0, r1, r2, r3, r4, r5): (u64, u64, u64, u64, u64, u64);
    // SAFETY: Register operands only, base A64 integer instructions, no memory or stack access.
    unsafe {
      asm!(
        "subs {a0}, {a0}, {b0}",
        "sbcs {a1}, {a1}, {b1}",
        "sbcs {a2}, {a2}, {b2}",
        "sbcs {a3}, {a3}, {b3}",
        "sbcs {a4}, {a4}, {b4}",
        "sbcs {a5}, {a5}, {b5}",
        "csetm {m}, lo",
        // p & mask, limb by limb: m >> 32, m << 32, m & (2^64 - 2), m, m, m.
        "lsr {t0}, {m}, #32",
        "lsl {t1}, {m}, #32",
        "and {t2}, {m}, #0xfffffffffffffffe",
        "adds {a0}, {a0}, {t0}",
        "adcs {a1}, {a1}, {t1}",
        "adcs {a2}, {a2}, {t2}",
        "adcs {a3}, {a3}, {m}",
        "adcs {a4}, {a4}, {m}",
        "adc {a5}, {a5}, {m}",
        a0 = inout(reg) a[0] => r0, a1 = inout(reg) a[1] => r1, a2 = inout(reg) a[2] => r2,
        a3 = inout(reg) a[3] => r3, a4 = inout(reg) a[4] => r4, a5 = inout(reg) a[5] => r5,
        b0 = in(reg) b[0], b1 = in(reg) b[1], b2 = in(reg) b[2],
        b3 = in(reg) b[3], b4 = in(reg) b[4], b5 = in(reg) b[5],
        m = out(reg) _, t0 = out(reg) _, t1 = out(reg) _, t2 = out(reg) _,
        options(pure, nomem, nostack),
      );
    }
    [r0, r1, r2, r3, r4, r5]
  }

  /// `a·b·2^-384 mod p` for canonical inputs, by operand scanning with one reduction step per
  /// row. The running value stays below 2p in seven limbs that rotate through `s0..s7`.
  #[inline(always)]
  pub(super) fn mul(a: [u64; 6], b: [u64; 6]) -> [u64; 6] {
    let (r0, r1, r2, r3, r4, r5): (u64, u64, u64, u64, u64, u64);
    // SAFETY: The block reads and writes only its register operands, uses base A64 integer
    // instructions that need no target feature, and touches neither memory nor the stack.
    unsafe {
      asm!(
        // Row 0: s0..s6 = a·b0.
        "mul {s0}, {a0}, {b0}",
        "mul {s1}, {a1}, {b0}",
        "mul {s2}, {a2}, {b0}",
        "mul {s3}, {a3}, {b0}",
        "mul {s4}, {a4}, {b0}",
        "mul {s5}, {a5}, {b0}",
        "umulh {t0}, {a0}, {b0}",
        "umulh {t1}, {a1}, {b0}",
        "umulh {t2}, {a2}, {b0}",
        "umulh {t3}, {a3}, {b0}",
        "umulh {t4}, {a4}, {b0}",
        "umulh {t5}, {a5}, {b0}",
        "adds {s1}, {s1}, {t0}",
        "adcs {s2}, {s2}, {t1}",
        "adcs {s3}, {s3}, {t2}",
        "adcs {s4}, {s4}, {t3}",
        "adcs {s5}, {s5}, {t4}",
        "adc {s6}, {t5}, xzr",
        // Reduction 0: add m·p and drop the zero low limb.
        "add {t0}, {s0}, {s0}, lsl #32",
        "lsl {t1}, {t0}, #32",
        "lsr {t2}, {t0}, #32",
        "cmp {t0}, {t1}",
        "sbcs {t3}, {t1}, {t2}",
        "sbc {t2}, {t2}, xzr",
        "adds {t4}, {t0}, {t2}",
        "cset {t5}, cs",
        "adds {s6}, {s6}, {t0}",
        "cset {s7}, cs",
        "subs {s1}, {s1}, {t3}",
        "sbcs {s2}, {s2}, {t4}",
        "sbcs {s3}, {s3}, {t5}",
        "sbcs {s4}, {s4}, xzr",
        "sbcs {s5}, {s5}, xzr",
        "sbcs {s6}, {s6}, xzr",
        "sbc {s7}, {s7}, xzr",
        // Row 1: add a·b1; the top limb stays below 2^64.
        "mul {t0}, {a0}, {b1}",
        "mul {t1}, {a1}, {b1}",
        "mul {t2}, {a2}, {b1}",
        "mul {t3}, {a3}, {b1}",
        "mul {t4}, {a4}, {b1}",
        "mul {t5}, {a5}, {b1}",
        "adds {s1}, {s1}, {t0}",
        "adcs {s2}, {s2}, {t1}",
        "adcs {s3}, {s3}, {t2}",
        "adcs {s4}, {s4}, {t3}",
        "adcs {s5}, {s5}, {t4}",
        "adcs {s6}, {s6}, {t5}",
        "adc {s7}, {s7}, xzr",
        "umulh {t0}, {a0}, {b1}",
        "umulh {t1}, {a1}, {b1}",
        "umulh {t2}, {a2}, {b1}",
        "umulh {t3}, {a3}, {b1}",
        "umulh {t4}, {a4}, {b1}",
        "umulh {t5}, {a5}, {b1}",
        "adds {s2}, {s2}, {t0}",
        "adcs {s3}, {s3}, {t1}",
        "adcs {s4}, {s4}, {t2}",
        "adcs {s5}, {s5}, {t3}",
        "adcs {s6}, {s6}, {t4}",
        "adcs {s7}, {s7}, {t5}",
        "adc {s0}, xzr, xzr",
        // Reduction 1: add m·p and drop the zero low limb.
        "add {t0}, {s1}, {s1}, lsl #32",
        "lsl {t1}, {t0}, #32",
        "lsr {t2}, {t0}, #32",
        "cmp {t0}, {t1}",
        "sbcs {t3}, {t1}, {t2}",
        "sbc {t2}, {t2}, xzr",
        "adds {t4}, {t0}, {t2}",
        "cset {t5}, cs",
        "adds {s7}, {s7}, {t0}",
        "adc {s0}, {s0}, xzr",
        "subs {s2}, {s2}, {t3}",
        "sbcs {s3}, {s3}, {t4}",
        "sbcs {s4}, {s4}, {t5}",
        "sbcs {s5}, {s5}, xzr",
        "sbcs {s6}, {s6}, xzr",
        "sbcs {s7}, {s7}, xzr",
        "sbc {s0}, {s0}, xzr",
        // Row 2: add a·b2; the top limb stays below 2^64.
        "mul {t0}, {a0}, {b2}",
        "mul {t1}, {a1}, {b2}",
        "mul {t2}, {a2}, {b2}",
        "mul {t3}, {a3}, {b2}",
        "mul {t4}, {a4}, {b2}",
        "mul {t5}, {a5}, {b2}",
        "adds {s2}, {s2}, {t0}",
        "adcs {s3}, {s3}, {t1}",
        "adcs {s4}, {s4}, {t2}",
        "adcs {s5}, {s5}, {t3}",
        "adcs {s6}, {s6}, {t4}",
        "adcs {s7}, {s7}, {t5}",
        "adc {s0}, {s0}, xzr",
        "umulh {t0}, {a0}, {b2}",
        "umulh {t1}, {a1}, {b2}",
        "umulh {t2}, {a2}, {b2}",
        "umulh {t3}, {a3}, {b2}",
        "umulh {t4}, {a4}, {b2}",
        "umulh {t5}, {a5}, {b2}",
        "adds {s3}, {s3}, {t0}",
        "adcs {s4}, {s4}, {t1}",
        "adcs {s5}, {s5}, {t2}",
        "adcs {s6}, {s6}, {t3}",
        "adcs {s7}, {s7}, {t4}",
        "adcs {s0}, {s0}, {t5}",
        "adc {s1}, xzr, xzr",
        // Reduction 2: add m·p and drop the zero low limb.
        "add {t0}, {s2}, {s2}, lsl #32",
        "lsl {t1}, {t0}, #32",
        "lsr {t2}, {t0}, #32",
        "cmp {t0}, {t1}",
        "sbcs {t3}, {t1}, {t2}",
        "sbc {t2}, {t2}, xzr",
        "adds {t4}, {t0}, {t2}",
        "cset {t5}, cs",
        "adds {s0}, {s0}, {t0}",
        "adc {s1}, {s1}, xzr",
        "subs {s3}, {s3}, {t3}",
        "sbcs {s4}, {s4}, {t4}",
        "sbcs {s5}, {s5}, {t5}",
        "sbcs {s6}, {s6}, xzr",
        "sbcs {s7}, {s7}, xzr",
        "sbcs {s0}, {s0}, xzr",
        "sbc {s1}, {s1}, xzr",
        // Row 3: add a·b3; the top limb stays below 2^64.
        "mul {t0}, {a0}, {b3}",
        "mul {t1}, {a1}, {b3}",
        "mul {t2}, {a2}, {b3}",
        "mul {t3}, {a3}, {b3}",
        "mul {t4}, {a4}, {b3}",
        "mul {t5}, {a5}, {b3}",
        "adds {s3}, {s3}, {t0}",
        "adcs {s4}, {s4}, {t1}",
        "adcs {s5}, {s5}, {t2}",
        "adcs {s6}, {s6}, {t3}",
        "adcs {s7}, {s7}, {t4}",
        "adcs {s0}, {s0}, {t5}",
        "adc {s1}, {s1}, xzr",
        "umulh {t0}, {a0}, {b3}",
        "umulh {t1}, {a1}, {b3}",
        "umulh {t2}, {a2}, {b3}",
        "umulh {t3}, {a3}, {b3}",
        "umulh {t4}, {a4}, {b3}",
        "umulh {t5}, {a5}, {b3}",
        "adds {s4}, {s4}, {t0}",
        "adcs {s5}, {s5}, {t1}",
        "adcs {s6}, {s6}, {t2}",
        "adcs {s7}, {s7}, {t3}",
        "adcs {s0}, {s0}, {t4}",
        "adcs {s1}, {s1}, {t5}",
        "adc {s2}, xzr, xzr",
        // Reduction 3: add m·p and drop the zero low limb.
        "add {t0}, {s3}, {s3}, lsl #32",
        "lsl {t1}, {t0}, #32",
        "lsr {t2}, {t0}, #32",
        "cmp {t0}, {t1}",
        "sbcs {t3}, {t1}, {t2}",
        "sbc {t2}, {t2}, xzr",
        "adds {t4}, {t0}, {t2}",
        "cset {t5}, cs",
        "adds {s1}, {s1}, {t0}",
        "adc {s2}, {s2}, xzr",
        "subs {s4}, {s4}, {t3}",
        "sbcs {s5}, {s5}, {t4}",
        "sbcs {s6}, {s6}, {t5}",
        "sbcs {s7}, {s7}, xzr",
        "sbcs {s0}, {s0}, xzr",
        "sbcs {s1}, {s1}, xzr",
        "sbc {s2}, {s2}, xzr",
        // Row 4: add a·b4; the top limb stays below 2^64.
        "mul {t0}, {a0}, {b4}",
        "mul {t1}, {a1}, {b4}",
        "mul {t2}, {a2}, {b4}",
        "mul {t3}, {a3}, {b4}",
        "mul {t4}, {a4}, {b4}",
        "mul {t5}, {a5}, {b4}",
        "adds {s4}, {s4}, {t0}",
        "adcs {s5}, {s5}, {t1}",
        "adcs {s6}, {s6}, {t2}",
        "adcs {s7}, {s7}, {t3}",
        "adcs {s0}, {s0}, {t4}",
        "adcs {s1}, {s1}, {t5}",
        "adc {s2}, {s2}, xzr",
        "umulh {t0}, {a0}, {b4}",
        "umulh {t1}, {a1}, {b4}",
        "umulh {t2}, {a2}, {b4}",
        "umulh {t3}, {a3}, {b4}",
        "umulh {t4}, {a4}, {b4}",
        "umulh {t5}, {a5}, {b4}",
        "adds {s5}, {s5}, {t0}",
        "adcs {s6}, {s6}, {t1}",
        "adcs {s7}, {s7}, {t2}",
        "adcs {s0}, {s0}, {t3}",
        "adcs {s1}, {s1}, {t4}",
        "adcs {s2}, {s2}, {t5}",
        "adc {s3}, xzr, xzr",
        // Reduction 4: add m·p and drop the zero low limb.
        "add {t0}, {s4}, {s4}, lsl #32",
        "lsl {t1}, {t0}, #32",
        "lsr {t2}, {t0}, #32",
        "cmp {t0}, {t1}",
        "sbcs {t3}, {t1}, {t2}",
        "sbc {t2}, {t2}, xzr",
        "adds {t4}, {t0}, {t2}",
        "cset {t5}, cs",
        "adds {s2}, {s2}, {t0}",
        "adc {s3}, {s3}, xzr",
        "subs {s5}, {s5}, {t3}",
        "sbcs {s6}, {s6}, {t4}",
        "sbcs {s7}, {s7}, {t5}",
        "sbcs {s0}, {s0}, xzr",
        "sbcs {s1}, {s1}, xzr",
        "sbcs {s2}, {s2}, xzr",
        "sbc {s3}, {s3}, xzr",
        // Row 5: add a·b5; the top limb stays below 2^64.
        "mul {t0}, {a0}, {b5}",
        "mul {t1}, {a1}, {b5}",
        "mul {t2}, {a2}, {b5}",
        "mul {t3}, {a3}, {b5}",
        "mul {t4}, {a4}, {b5}",
        "mul {t5}, {a5}, {b5}",
        "adds {s5}, {s5}, {t0}",
        "adcs {s6}, {s6}, {t1}",
        "adcs {s7}, {s7}, {t2}",
        "adcs {s0}, {s0}, {t3}",
        "adcs {s1}, {s1}, {t4}",
        "adcs {s2}, {s2}, {t5}",
        "adc {s3}, {s3}, xzr",
        "umulh {t0}, {a0}, {b5}",
        "umulh {t1}, {a1}, {b5}",
        "umulh {t2}, {a2}, {b5}",
        "umulh {t3}, {a3}, {b5}",
        "umulh {t4}, {a4}, {b5}",
        "umulh {t5}, {a5}, {b5}",
        "adds {s6}, {s6}, {t0}",
        "adcs {s7}, {s7}, {t1}",
        "adcs {s0}, {s0}, {t2}",
        "adcs {s1}, {s1}, {t3}",
        "adcs {s2}, {s2}, {t4}",
        "adcs {s3}, {s3}, {t5}",
        "adc {s4}, xzr, xzr",
        // Reduction 5: add m·p and drop the zero low limb.
        "add {t0}, {s5}, {s5}, lsl #32",
        "lsl {t1}, {t0}, #32",
        "lsr {t2}, {t0}, #32",
        "cmp {t0}, {t1}",
        "sbcs {t3}, {t1}, {t2}",
        "sbc {t2}, {t2}, xzr",
        "adds {t4}, {t0}, {t2}",
        "cset {t5}, cs",
        "adds {s3}, {s3}, {t0}",
        "adc {s4}, {s4}, xzr",
        "subs {s6}, {s6}, {t3}",
        "sbcs {s7}, {s7}, {t4}",
        "sbcs {s0}, {s0}, {t5}",
        "sbcs {s1}, {s1}, xzr",
        "sbcs {s2}, {s2}, xzr",
        "sbcs {s3}, {s3}, xzr",
        "sbc {s4}, {s4}, xzr",
        // Subtract p unless that borrows.
        "mov {t0}, #0xffffffff",
        "mov {t1}, #0xffffffff00000000",
        "mov {t2}, #0xfffffffffffffffe",
        "mov {t3}, #-1",
        "subs {b0}, {s6}, {t0}",
        "sbcs {b1}, {s7}, {t1}",
        "sbcs {b2}, {s0}, {t2}",
        "sbcs {b3}, {s1}, {t3}",
        "sbcs {b4}, {s2}, {t3}",
        "sbcs {b5}, {s3}, {t3}",
        "sbcs xzr, {s4}, xzr",
        "csel {s6}, {s6}, {b0}, lo",
        "csel {s7}, {s7}, {b1}, lo",
        "csel {s0}, {s0}, {b2}, lo",
        "csel {s1}, {s1}, {b3}, lo",
        "csel {s2}, {s2}, {b4}, lo",
        "csel {s3}, {s3}, {b5}, lo",
        a0 = in(reg) a[0], a1 = in(reg) a[1], a2 = in(reg) a[2],
        a3 = in(reg) a[3], a4 = in(reg) a[4], a5 = in(reg) a[5],
        b0 = inout(reg) b[0] => _, b1 = inout(reg) b[1] => _, b2 = inout(reg) b[2] => _,
        b3 = inout(reg) b[3] => _, b4 = inout(reg) b[4] => _, b5 = inout(reg) b[5] => _,
        s6 = out(reg) r0, s7 = out(reg) r1, s0 = out(reg) r2, s1 = out(reg) r3, s2 = out(reg) r4, s3 = out(reg) r5,
        s4 = out(reg) _, s5 = out(reg) _,
        t0 = out(reg) _, t1 = out(reg) _, t2 = out(reg) _, t3 = out(reg) _, t4 = out(reg) _, t5 = out(reg) _,
        options(pure, nomem, nostack),
      );
    }
    [r0, r1, r2, r3, r4, r5]
  }

  /// `a²·2^-384 mod p` for a canonical input: the doubled cross products and the diagonal give
  /// the twelve-limb square; six reduction steps fold its low half to at most p, and the high
  /// half (below p) is added before one conditional subtraction.
  #[inline(always)]
  pub(super) fn square(a: [u64; 6]) -> [u64; 6] {
    let (r0, r1, r2, r3, r4, r5): (u64, u64, u64, u64, u64, u64);
    // SAFETY: As for `mul`: register operands only, base A64 integer instructions, no memory or
    // stack access. The input registers are reused as scratch after their last read.
    unsafe {
      asm!(
        // Cross row a0·[a1..a5] at limbs 1..6.
        "mul {x1}, {a0}, {a1}",
        "mul {x2}, {a0}, {a2}",
        "mul {x3}, {a0}, {a3}",
        "mul {x4}, {a0}, {a4}",
        "mul {x5}, {a0}, {a5}",
        "umulh {t0}, {a0}, {a1}",
        "umulh {t1}, {a0}, {a2}",
        "umulh {t2}, {a0}, {a3}",
        "umulh {t3}, {a0}, {a4}",
        "umulh {t4}, {a0}, {a5}",
        "adds {x2}, {x2}, {t0}",
        "adcs {x3}, {x3}, {t1}",
        "adcs {x4}, {x4}, {t2}",
        "adcs {x5}, {x5}, {t3}",
        "adc {x6}, {t4}, xzr",
        // Cross row a1·[a2, a3, a4, a5] at limbs 3..7.
        "mul {t0}, {a1}, {a2}",
        "mul {t1}, {a1}, {a3}",
        "mul {t2}, {a1}, {a4}",
        "mul {t3}, {a1}, {a5}",
        "adds {x3}, {x3}, {t0}",
        "adcs {x4}, {x4}, {t1}",
        "adcs {x5}, {x5}, {t2}",
        "adcs {x6}, {x6}, {t3}",
        "adc {x7}, xzr, xzr",
        "umulh {t0}, {a1}, {a2}",
        "umulh {t1}, {a1}, {a3}",
        "umulh {t2}, {a1}, {a4}",
        "umulh {t3}, {a1}, {a5}",
        "adds {x4}, {x4}, {t0}",
        "adcs {x5}, {x5}, {t1}",
        "adcs {x6}, {x6}, {t2}",
        "adc {x7}, {x7}, {t3}",
        // Cross row a2·[a3, a4, a5] at limbs 5..8.
        "mul {t0}, {a2}, {a3}",
        "mul {t1}, {a2}, {a4}",
        "mul {t2}, {a2}, {a5}",
        "adds {x5}, {x5}, {t0}",
        "adcs {x6}, {x6}, {t1}",
        "adcs {x7}, {x7}, {t2}",
        "adc {x8}, xzr, xzr",
        "umulh {t0}, {a2}, {a3}",
        "umulh {t1}, {a2}, {a4}",
        "umulh {t2}, {a2}, {a5}",
        "adds {x6}, {x6}, {t0}",
        "adcs {x7}, {x7}, {t1}",
        "adc {x8}, {x8}, {t2}",
        // Cross row a3·[a4, a5] at limbs 7..9.
        "mul {t0}, {a3}, {a4}",
        "mul {t1}, {a3}, {a5}",
        "adds {x7}, {x7}, {t0}",
        "adcs {x8}, {x8}, {t1}",
        "adc {x9}, xzr, xzr",
        "umulh {t0}, {a3}, {a4}",
        "umulh {t1}, {a3}, {a5}",
        "adds {x8}, {x8}, {t0}",
        "adc {x9}, {x9}, {t1}",
        // Cross row a4·[a5] at limbs 9..10.
        "mul {t0}, {a4}, {a5}",
        "adds {x9}, {x9}, {t0}",
        "adc {x10}, xzr, xzr",
        "umulh {t0}, {a4}, {a5}",
        "add {x10}, {x10}, {t0}",
        // Double the cross sum.
        "lsr {x11}, {x10}, #63",
        "extr {x10}, {x10}, {x9}, #63",
        "extr {x9}, {x9}, {x8}, #63",
        "extr {x8}, {x8}, {x7}, #63",
        "extr {x7}, {x7}, {x6}, #63",
        "extr {x6}, {x6}, {x5}, #63",
        "extr {x5}, {x5}, {x4}, #63",
        "extr {x4}, {x4}, {x3}, #63",
        "extr {x3}, {x3}, {x2}, #63",
        "extr {x2}, {x2}, {x1}, #63",
        "lsl {x1}, {x1}, #1",
        // Add the squares a_i² on the diagonal; t0 holds limb 0.
        "mul {t0}, {a0}, {a0}",
        "mul {t1}, {a1}, {a1}",
        "mul {t2}, {a2}, {a2}",
        "mul {t3}, {a3}, {a3}",
        "mul {t4}, {a4}, {a4}",
        "mul {t5}, {a5}, {a5}",
        "umulh {a0}, {a0}, {a0}",
        "umulh {a1}, {a1}, {a1}",
        "umulh {a2}, {a2}, {a2}",
        "umulh {a3}, {a3}, {a3}",
        "umulh {a4}, {a4}, {a4}",
        "umulh {a5}, {a5}, {a5}",
        "adds {x1}, {x1}, {a0}",
        "adcs {x2}, {x2}, {t1}",
        "adcs {x3}, {x3}, {a1}",
        "adcs {x4}, {x4}, {t2}",
        "adcs {x5}, {x5}, {a2}",
        "adcs {x6}, {x6}, {t3}",
        "adcs {x7}, {x7}, {a3}",
        "adcs {x8}, {x8}, {t4}",
        "adcs {x9}, {x9}, {a4}",
        "adcs {x10}, {x10}, {t5}",
        "adc {x11}, {x11}, {a5}",
        // Low-half reduction 0: the window stays below 2^384.
        "add {a0}, {t0}, {t0}, lsl #32",
        "lsl {t2}, {a0}, #32",
        "lsr {t3}, {a0}, #32",
        "cmp {a0}, {t2}",
        "sbcs {t4}, {t2}, {t3}",
        "sbc {t3}, {t3}, xzr",
        "adds {t5}, {a0}, {t3}",
        "cset {t1}, cs",
        "subs {x1}, {x1}, {t4}",
        "sbcs {x2}, {x2}, {t5}",
        "sbcs {x3}, {x3}, {t1}",
        "sbcs {x4}, {x4}, xzr",
        "sbcs {x5}, {x5}, xzr",
        "sbc {t0}, {a0}, xzr",
        // Low-half reduction 1: the window stays below 2^384.
        "add {a0}, {x1}, {x1}, lsl #32",
        "lsl {t2}, {a0}, #32",
        "lsr {t3}, {a0}, #32",
        "cmp {a0}, {t2}",
        "sbcs {t4}, {t2}, {t3}",
        "sbc {t3}, {t3}, xzr",
        "adds {t5}, {a0}, {t3}",
        "cset {t1}, cs",
        "subs {x2}, {x2}, {t4}",
        "sbcs {x3}, {x3}, {t5}",
        "sbcs {x4}, {x4}, {t1}",
        "sbcs {x5}, {x5}, xzr",
        "sbcs {t0}, {t0}, xzr",
        "sbc {x1}, {a0}, xzr",
        // Low-half reduction 2: the window stays below 2^384.
        "add {a0}, {x2}, {x2}, lsl #32",
        "lsl {t2}, {a0}, #32",
        "lsr {t3}, {a0}, #32",
        "cmp {a0}, {t2}",
        "sbcs {t4}, {t2}, {t3}",
        "sbc {t3}, {t3}, xzr",
        "adds {t5}, {a0}, {t3}",
        "cset {t1}, cs",
        "subs {x3}, {x3}, {t4}",
        "sbcs {x4}, {x4}, {t5}",
        "sbcs {x5}, {x5}, {t1}",
        "sbcs {t0}, {t0}, xzr",
        "sbcs {x1}, {x1}, xzr",
        "sbc {x2}, {a0}, xzr",
        // Low-half reduction 3: the window stays below 2^384.
        "add {a0}, {x3}, {x3}, lsl #32",
        "lsl {t2}, {a0}, #32",
        "lsr {t3}, {a0}, #32",
        "cmp {a0}, {t2}",
        "sbcs {t4}, {t2}, {t3}",
        "sbc {t3}, {t3}, xzr",
        "adds {t5}, {a0}, {t3}",
        "cset {t1}, cs",
        "subs {x4}, {x4}, {t4}",
        "sbcs {x5}, {x5}, {t5}",
        "sbcs {t0}, {t0}, {t1}",
        "sbcs {x1}, {x1}, xzr",
        "sbcs {x2}, {x2}, xzr",
        "sbc {x3}, {a0}, xzr",
        // Low-half reduction 4: the window stays below 2^384.
        "add {a0}, {x4}, {x4}, lsl #32",
        "lsl {t2}, {a0}, #32",
        "lsr {t3}, {a0}, #32",
        "cmp {a0}, {t2}",
        "sbcs {t4}, {t2}, {t3}",
        "sbc {t3}, {t3}, xzr",
        "adds {t5}, {a0}, {t3}",
        "cset {t1}, cs",
        "subs {x5}, {x5}, {t4}",
        "sbcs {t0}, {t0}, {t5}",
        "sbcs {x1}, {x1}, {t1}",
        "sbcs {x2}, {x2}, xzr",
        "sbcs {x3}, {x3}, xzr",
        "sbc {x4}, {a0}, xzr",
        // Low-half reduction 5: the window stays below 2^384.
        "add {a0}, {x5}, {x5}, lsl #32",
        "lsl {t2}, {a0}, #32",
        "lsr {t3}, {a0}, #32",
        "cmp {a0}, {t2}",
        "sbcs {t4}, {t2}, {t3}",
        "sbc {t3}, {t3}, xzr",
        "adds {t5}, {a0}, {t3}",
        "cset {t1}, cs",
        "subs {t0}, {t0}, {t4}",
        "sbcs {x1}, {x1}, {t5}",
        "sbcs {x2}, {x2}, {t1}",
        "sbcs {x3}, {x3}, xzr",
        "sbcs {x4}, {x4}, xzr",
        "sbc {x5}, {a0}, xzr",
        // Add the high half; the sum stays below 2p.
        "adds {t0}, {t0}, {x6}",
        "adcs {x1}, {x1}, {x7}",
        "adcs {x2}, {x2}, {x8}",
        "adcs {x3}, {x3}, {x9}",
        "adcs {x4}, {x4}, {x10}",
        "adcs {x5}, {x5}, {x11}",
        "cset {t1}, cs",
        // Subtract p unless that borrows.
        "mov {t2}, #0xffffffff",
        "mov {t3}, #0xffffffff00000000",
        "mov {t4}, #0xfffffffffffffffe",
        "mov {t5}, #-1",
        "subs {x6}, {t0}, {t2}",
        "sbcs {x7}, {x1}, {t3}",
        "sbcs {x8}, {x2}, {t4}",
        "sbcs {x9}, {x3}, {t5}",
        "sbcs {x10}, {x4}, {t5}",
        "sbcs {x11}, {x5}, {t5}",
        "sbcs xzr, {t1}, xzr",
        "csel {t0}, {t0}, {x6}, lo",
        "csel {x1}, {x1}, {x7}, lo",
        "csel {x2}, {x2}, {x8}, lo",
        "csel {x3}, {x3}, {x9}, lo",
        "csel {x4}, {x4}, {x10}, lo",
        "csel {x5}, {x5}, {x11}, lo",
        a0 = inout(reg) a[0] => _, a1 = inout(reg) a[1] => _, a2 = inout(reg) a[2] => _,
        a3 = inout(reg) a[3] => _, a4 = inout(reg) a[4] => _, a5 = inout(reg) a[5] => _,
        t0 = out(reg) r0, x1 = out(reg) r1, x2 = out(reg) r2, x3 = out(reg) r3, x4 = out(reg) r4, x5 = out(reg) r5,
        x6 = out(reg) _, x7 = out(reg) _, x8 = out(reg) _, x9 = out(reg) _, x10 = out(reg) _, x11 = out(reg) _,
        t1 = out(reg) _, t2 = out(reg) _, t3 = out(reg) _, t4 = out(reg) _, t5 = out(reg) _,
        options(pure, nomem, nostack),
      );
    }
    [r0, r1, r2, r3, r4, r5]
  }
}

#[cfg(test)]
mod tests {
  use super::super::{FieldElement, P384_FIELD_MODULUS, Uint, add_p384_field, montgomery_mul, sub_p384_field};
  use super::{FIELD_ONE, Fe, P};

  #[test]
  fn field_arithmetic_matches_generic_authority() {
    assert_eq!(FieldElement::one(&P384_FIELD_MODULUS).value.0, FIELD_ONE);
    let edges = [
      [0u64; 6],
      [1, 0, 0, 0, 0, 0],
      // Times the previous edge, its low limb is 2^32, which takes the reduction's borrow.
      [1 << 32, 0, 0, 0, 0, 0],
      [P[0] - 1, P[1], P[2], P[3], P[4], P[5]],
      [P[0], P[1] - 1, P[2], P[3], P[4], P[5]],
      [u64::MAX, u64::MAX, P[2] - 1, P[3], P[4], P[5]],
      [0, 0, 0, 0, 0, P[5] - 1],
      [P[0] - 2, P[1], P[2], P[3], P[4], P[5]],
      [u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX >> 1],
    ];
    let mut state = 0xbb67_ae85_84ca_a73bu64;
    let mut next = || {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      state
    };
    let mut generated = || {
      let mut value: [u64; 6] = core::array::from_fn(|_| next());
      value[5] = value[5].min(P[5] - 1);
      value
    };
    let count = if cfg!(miri) { 16 } else { 5_000 };
    let mut previous = edges[3];
    for index in 0..count + edges.len() {
      let a = edges.get(index).copied().unwrap_or_else(&mut generated);
      for b in [previous, edges[index % edges.len()]] {
        let (fa, fb) = (Fe(a), Fe(b));
        let product = montgomery_mul(Uint(a), Uint(b), &P384_FIELD_MODULUS).0;
        assert_eq!(fa.mul(fb).0, product, "{a:x?} · {b:x?}");
        assert_eq!(fa.mul_portable(fb).0, product, "portable {a:x?} · {b:x?}");
        assert_eq!(fa.add(fb).0, add_p384_field(a, b).0, "{a:x?} + {b:x?}");
        assert_eq!(
          fa.add_portable(fb).0,
          add_p384_field(a, b).0,
          "portable {a:x?} + {b:x?}"
        );
        assert_eq!(fa.sub(fb).0, sub_p384_field(a, b).0, "{a:x?} - {b:x?}");
        assert_eq!(
          fa.sub_portable(fb).0,
          sub_p384_field(a, b).0,
          "portable {a:x?} - {b:x?}"
        );
        assert_eq!(fb.sub(fa).0, sub_p384_field(b, a).0, "{b:x?} - {a:x?}");
      }
      let square = montgomery_mul(Uint(a), Uint(a), &P384_FIELD_MODULUS).0;
      assert_eq!(Fe(a).square().0, square, "{a:x?}²");
      assert_eq!(Fe(a).square_portable().0, square, "portable {a:x?}²");
      previous = a;
    }
  }
}
