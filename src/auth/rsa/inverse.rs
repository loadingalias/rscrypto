//! Fixed-width modular inverses.

use super::*;

pub(super) fn private_modular_inverse_odd(
  modulus: &RsaPublicModulus,
  input: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPrivateInverseScratch,
) -> Result<(), RsaPrivateOpError> {
  crate::traits::ct::with_data_independent_timing(|| {
    let bytes = modulus.bytes.len();
    if input.len() != bytes || out.len() != bytes || modulus.limbs.first().copied().unwrap_or(0) & 1 == 0 {
      ct::zeroize(out);
      return Err(RsaPrivateOpError::InvalidLength);
    }
    if !private_valid_blinding_factor(input, &modulus.bytes) {
      ct::zeroize(out);
      return Err(RsaPrivateOpError::InvalidBlindingFactor);
    }

    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
    let result = private_modular_inverse_odd_i31(modulus, input, out, scratch);
    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
    let result = private_modular_inverse_odd_s62(modulus, input, out, scratch);

    scratch.clear();
    if result.is_err() {
      ct::zeroize(out);
    }
    result
  })
}

/// Invert a valid blinding factor with the 31-bit batched extended binary GCD.
#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
fn private_modular_inverse_odd_i31(
  modulus: &RsaPublicModulus,
  input: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPrivateI31InverseScratch,
) -> Result<(), RsaPrivateOpError> {
  let RsaPrivateI31InverseWorkspace {
    a,
    b,
    u,
    v,
    modulus: modulus_i31,
  } = scratch.workspace(modulus.bits)?;
  private_be_to_i31(input, a);
  private_be_to_i31(&modulus.bytes, modulus_i31);
  b.copy_from_slice(modulus_i31);
  u.fill(0);
  v.fill(0);
  *u.first_mut().ok_or(RsaPrivateOpError::InvalidScratch)? = 1;

  // The extended-binary-GCD state maintains:
  //   a ≡ input * u (mod modulus)
  //   b ≡ input * v (mod modulus)
  // starting from (a, b, u, v) = (input, modulus, 1, 0). Each batch
  // simulates 31 binary steps from fixed low/high approximations, then
  // applies the resulting matrix to the full-width values. The approximation
  // guarantees at least 30 effective reductions per batch. Binary GCD needs
  // at most 2*k - 2 reductions for a k-bit odd modulus, so this public-width
  // schedule includes one conservative batch beyond 2*k/30.
  let batch_count = modulus.bits.strict_mul(2).strict_add(30) / 30;
  for _ in 0..batch_count {
    let (pa, pb, qa, qb) = private_i31_reduction_factors(a, b);
    let signs = private_i31_co_reduce(a, b, pa, pb, qa, qb);
    let a_negative = i64::from(signs & 1);
    let b_negative = i64::from((signs >> 1) & 1);
    let pa = private_i31_cond_negate_factor(pa, a_negative);
    let pb = private_i31_cond_negate_factor(pb, a_negative);
    let qa = private_i31_cond_negate_factor(qa, b_negative);
    let qb = private_i31_cond_negate_factor(qb, b_negative);
    private_i31_co_reduce_mod(
      u,
      v,
      pa,
      pb,
      qa,
      qb,
      modulus_i31,
      private_i31_low_u32(modulus.n0) & PRIVATE_I31_MASK,
    );
  }

  let mut gcd_difference = (a.first().copied().unwrap_or(0) | b.first().copied().unwrap_or(0)) ^ 1;
  for index in 1..a.len() {
    gcd_difference |= a[index] | b[index];
    u[index] |= v[index];
  }
  if ct_nonzero_u64(u64::from(gcd_difference)) != 0 {
    return Err(RsaPrivateOpError::InvalidBlindingFactor);
  }
  u[0] |= v[0];
  private_i31_to_be(u, out);
  Ok(())
}

/// Invert a valid blinding factor with batches of 62 Bernstein–Yang divsteps.
///
/// The state follows Bernstein and Yang, "Fast constant-time gcd computation and modular
/// inversion" (2019): `f` and `g` start at the modulus and the factor, and `d·x ≡ f`,
/// `e·x ≡ g (mod n)` hold throughout, so `f = ±1` at the end gives `x⁻¹ = ±d`. Theorem 11.2
/// bounds the public divstep count. Every batch, update and normalization runs a fixed number
/// of limb steps, and selections use masks.
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
fn private_modular_inverse_odd_s62(
  modulus: &RsaPublicModulus,
  input: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPrivateS62InverseScratch,
) -> Result<(), RsaPrivateOpError> {
  let RsaPrivateS62InverseWorkspace { f, g, d, e, modulus: n } = scratch.workspace(modulus.bits)?;
  private_be_to_s62(&modulus.bytes, n);
  f.copy_from_slice(n);
  private_be_to_s62(input, g);
  d.fill(0);
  e.fill(0);
  *e.first_mut().ok_or(RsaPrivateOpError::InvalidScratch)? = 1;
  // `n0` is `-n⁻¹ mod 2^64`.
  let n_inverse = modulus.n0.wrapping_neg() & PRIVATE_S62_MASK;

  // Theorem 11.2: for odd f > g >= 0 below 2^bits, divsteps^m(1, f, g) has g = 0 once
  // m >= floor((49 * bits + 57) / 17) for bits >= 46, or floor((49 * bits + 80) / 17) below.
  let bits = modulus.bits;
  let divsteps = bits.strict_mul(49).strict_add(if bits >= 46 { 57 } else { 80 }) / 17;
  let mut delta = 1i64;
  for _ in 0..divsteps.div_ceil(62) {
    let (next_delta, transition) = private_s62_divsteps(delta, f[0], g[0]);
    delta = next_delta;
    private_s62_update_de(d, e, transition, n, n_inverse);
    private_s62_update_fg(f, g, transition);
  }

  // f is now ±gcd(x, n). Lower limbs are 62-bit and the top limb is signed, so -1 is all
  // lower limbs at the mask with a top limb of -1.
  let last = f.len().strict_sub(1);
  let mut not_one = f[0] ^ 1;
  let mut not_minus_one = f[0] ^ PRIVATE_S62_MASK;
  for &limb in &f[1..last] {
    not_one |= limb;
    not_minus_one |= limb ^ PRIVATE_S62_MASK;
  }
  not_one |= f[last];
  not_minus_one |= f[last] ^ u64::MAX;
  let is_minus_one = ct_nonzero_u64(not_minus_one) ^ 1;
  if (ct_nonzero_u64(not_one) ^ 1) | is_minus_one == 0 {
    return Err(RsaPrivateOpError::InvalidBlindingFactor);
  }

  // d is in (-2n, n); multiply by the sign of f, then map into [0, n).
  private_s62_cond_negate(d, 0u64.wrapping_sub(is_minus_one));
  for _ in 0..2 {
    let negative = (d[last].cast_signed() >> 63).cast_unsigned();
    private_s62_add_masked(d, n, negative);
  }
  private_s62_sub(d, n);
  let negative = (d[last].cast_signed() >> 63).cast_unsigned();
  private_s62_add_masked(d, n, negative);
  private_s62_to_be(d, out);
  Ok(())
}

/// Signed 62-bit limbs for a `bit_len`-bit modulus, plus one limb of sign headroom.
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
pub(super) fn private_s62_limb_count(bit_len: usize) -> usize {
  bit_len.div_ceil(62).strict_add(1)
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
#[inline(always)]
fn private_s62_low_u64(value: i128) -> u64 {
  let bytes = value.to_le_bytes();
  u64::from_le_bytes([
    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
  ])
}

/// Load big-endian bytes into non-negative 62-bit limbs.
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
fn private_be_to_s62(input: &[u8], out: &mut [u64]) {
  out.fill(0);
  let mut accumulator = 0u128;
  let mut accumulator_bits = 0usize;
  let mut limb_index = 0usize;
  for &byte in input.iter().rev() {
    accumulator |= u128::from(byte) << accumulator_bits;
    accumulator_bits = accumulator_bits.strict_add(8);
    if accumulator_bits >= 62 {
      out[limb_index] = private_s62_low_u64(accumulator.cast_signed()) & PRIVATE_S62_MASK;
      limb_index = limb_index.strict_add(1);
      accumulator >>= 62;
      accumulator_bits = accumulator_bits.strict_sub(62);
    }
  }
  if limb_index < out.len() {
    out[limb_index] = private_s62_low_u64(accumulator.cast_signed());
  }
}

/// Store non-negative 62-bit limbs that fit `out` as big-endian bytes.
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
fn private_s62_to_be(input: &[u64], out: &mut [u8]) {
  let mut accumulator = 0u128;
  let mut accumulator_bits = 0usize;
  let mut limb_index = 0usize;
  for byte in out.iter_mut().rev() {
    if accumulator_bits < 8 {
      let limb = input.get(limb_index).copied().unwrap_or(0);
      accumulator |= u128::from(limb) << accumulator_bits;
      accumulator_bits = accumulator_bits.strict_add(62);
      limb_index = limb_index.strict_add(1);
    }
    *byte = accumulator.to_le_bytes()[0];
    accumulator >>= 8;
    accumulator_bits = accumulator_bits.strict_sub(8);
  }
}

/// Apply 62 divsteps to the low 62 bits of `f` and `g`.
///
/// Returns the next `delta` and the transition `[u, v, q, r]`: after the batch,
/// `2^62·f' = u·f + v·g` and `2^62·g' = q·f + r·g`. Each step reads only bit 0 of `f` and `g`,
/// which the low 62 bits determine for 62 steps. The row of `f` doubles every step, so
/// `|u| + |v| <= 2^62` and `|q| + |r| <= 2^62`.
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
fn private_s62_divsteps(mut delta: i64, f0: u64, g0: u64) -> (i64, [i64; 4]) {
  let mut f = f0.cast_signed();
  let mut g = g0.cast_signed();
  let (mut u, mut v, mut q, mut r) = (1i64, 0i64, 0i64, 1i64);
  for _ in 0..62 {
    // All ones when delta > 0, and when g is odd. |delta| stays below the step count.
    let positive = delta.wrapping_neg() >> 63;
    let odd = 0i64.wrapping_sub(g & 1);
    // When g is odd, add f's row to g's row, negated when delta > 0.
    g = g.wrapping_add((f ^ positive).wrapping_sub(positive) & odd);
    q = q.wrapping_add((u ^ positive).wrapping_sub(positive) & odd);
    r = r.wrapping_add((v ^ positive).wrapping_sub(positive) & odd);
    // When both held, the old g (now f + g) becomes f and delta becomes 1 - delta.
    let swap = positive & odd;
    delta = (delta ^ swap).wrapping_sub(swap).wrapping_add(1);
    f = f.wrapping_add(g & swap);
    u = u.wrapping_add(q & swap);
    v = v.wrapping_add(r & swap);
    // g is even; halve it and double f's row.
    g >>= 1;
    u <<= 1;
    v <<= 1;
  }
  (delta, [u, v, q, r])
}

/// Replace `(f, g)` by `((u·f + v·g) / 2^62, (q·f + r·g) / 2^62)`; both divisions are exact.
///
/// Lower limbs are below 2^62 and the transition rows sum to at most 2^62 in magnitude, so each
/// accumulator stays below 2^126.
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
fn private_s62_update_fg(f: &mut [u64], g: &mut [u64], transition: [i64; 4]) {
  let [u, v, q, r] = transition.map(i128::from);
  let last = f.len().strict_sub(1);
  let (f0, g0) = (i128::from(f[0].cast_signed()), i128::from(g[0].cast_signed()));
  let mut cf = u.wrapping_mul(f0).wrapping_add(v.wrapping_mul(g0));
  let mut cg = q.wrapping_mul(f0).wrapping_add(r.wrapping_mul(g0));
  debug_assert_eq!(private_s62_low_u64(cf) & PRIVATE_S62_MASK, 0);
  debug_assert_eq!(private_s62_low_u64(cg) & PRIVATE_S62_MASK, 0);
  cf >>= 62;
  cg >>= 62;
  for index in 1..f.len() {
    let (fi, gi) = (i128::from(f[index].cast_signed()), i128::from(g[index].cast_signed()));
    cf = cf.wrapping_add(u.wrapping_mul(fi)).wrapping_add(v.wrapping_mul(gi));
    cg = cg.wrapping_add(q.wrapping_mul(fi)).wrapping_add(r.wrapping_mul(gi));
    f[index.strict_sub(1)] = private_s62_low_u64(cf) & PRIVATE_S62_MASK;
    g[index.strict_sub(1)] = private_s62_low_u64(cg) & PRIVATE_S62_MASK;
    cf >>= 62;
    cg >>= 62;
  }
  f[last] = private_s62_low_u64(cf);
  g[last] = private_s62_low_u64(cg);
}

/// Replace `(d, e)` by `((u·d + v·e) / 2^62, (q·d + r·e) / 2^62) mod n`.
///
/// For inputs in `(-2n, n)` the outputs stay in `(-2n, n)`: `md` and `me` start at the
/// transition rows of the negative inputs, then subtract the residue that makes each numerator
/// divisible by 2^62 (`n_inverse = n⁻¹ mod 2^62`). Accumulators stay below 2^127.
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
fn private_s62_update_de(d: &mut [u64], e: &mut [u64], transition: [i64; 4], modulus: &[u64], n_inverse: u64) {
  let [u, v, q, r] = transition;
  let last = d.len().strict_sub(1);
  let d_negative = d[last].cast_signed() >> 63;
  let e_negative = e[last].cast_signed() >> 63;
  let mut md = (u & d_negative).wrapping_add(v & e_negative);
  let mut me = (q & d_negative).wrapping_add(r & e_negative);
  let (d0, e0) = (d[0].cast_signed(), e[0].cast_signed());
  let cd = u.wrapping_mul(d0).wrapping_add(v.wrapping_mul(e0)).cast_unsigned();
  let ce = q.wrapping_mul(d0).wrapping_add(r.wrapping_mul(e0)).cast_unsigned();
  md = md.wrapping_sub((n_inverse.wrapping_mul(cd).wrapping_add(md.cast_unsigned()) & PRIVATE_S62_MASK).cast_signed());
  me = me.wrapping_sub((n_inverse.wrapping_mul(ce).wrapping_add(me.cast_unsigned()) & PRIVATE_S62_MASK).cast_signed());

  let [u, v, q, r, md, me] = [u, v, q, r, md, me].map(i128::from);
  let (d0, e0, n0) = (i128::from(d0), i128::from(e0), i128::from(modulus[0].cast_signed()));
  let mut cd = u
    .wrapping_mul(d0)
    .wrapping_add(v.wrapping_mul(e0))
    .wrapping_add(md.wrapping_mul(n0));
  let mut ce = q
    .wrapping_mul(d0)
    .wrapping_add(r.wrapping_mul(e0))
    .wrapping_add(me.wrapping_mul(n0));
  debug_assert_eq!(private_s62_low_u64(cd) & PRIVATE_S62_MASK, 0);
  debug_assert_eq!(private_s62_low_u64(ce) & PRIVATE_S62_MASK, 0);
  cd >>= 62;
  ce >>= 62;
  for index in 1..d.len() {
    let di = i128::from(d[index].cast_signed());
    let ei = i128::from(e[index].cast_signed());
    let ni = i128::from(modulus[index].cast_signed());
    cd = cd
      .wrapping_add(u.wrapping_mul(di))
      .wrapping_add(v.wrapping_mul(ei))
      .wrapping_add(md.wrapping_mul(ni));
    ce = ce
      .wrapping_add(q.wrapping_mul(di))
      .wrapping_add(r.wrapping_mul(ei))
      .wrapping_add(me.wrapping_mul(ni));
    d[index.strict_sub(1)] = private_s62_low_u64(cd) & PRIVATE_S62_MASK;
    e[index.strict_sub(1)] = private_s62_low_u64(ce) & PRIVATE_S62_MASK;
    cd >>= 62;
    ce >>= 62;
  }
  d[last] = private_s62_low_u64(cd);
  e[last] = private_s62_low_u64(ce);
}

/// Negate `value` when `mask` is all ones, keeping lower limbs in 62 bits.
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
fn private_s62_cond_negate(value: &mut [u64], mask: u64) {
  let last = value.len().strict_sub(1);
  let mut carry = 0i64;
  for limb in &mut value[..last] {
    let negated = carry.wrapping_sub(limb.cast_signed());
    carry = negated >> 62;
    *limb = (*limb & !mask) | (negated.cast_unsigned() & PRIVATE_S62_MASK & mask);
  }
  let negated = carry.wrapping_sub(value[last].cast_signed());
  value[last] = (value[last] & !mask) | (negated.cast_unsigned() & mask);
}

/// Add `addend & mask` to `value`, keeping lower limbs in 62 bits.
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
fn private_s62_add_masked(value: &mut [u64], addend: &[u64], mask: u64) {
  let last = value.len().strict_sub(1);
  let mut carry = 0i64;
  for (limb, &add) in value[..last].iter_mut().zip(addend) {
    let sum = limb
      .cast_signed()
      .wrapping_add((add & mask).cast_signed())
      .wrapping_add(carry);
    carry = sum >> 62;
    *limb = sum.cast_unsigned() & PRIVATE_S62_MASK;
  }
  value[last] = value[last]
    .cast_signed()
    .wrapping_add((addend[last] & mask).cast_signed())
    .wrapping_add(carry)
    .cast_unsigned();
}

/// Subtract `subtrahend` from `value`, keeping lower limbs in 62 bits.
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
fn private_s62_sub(value: &mut [u64], subtrahend: &[u64]) {
  let last = value.len().strict_sub(1);
  let mut carry = 0i64;
  for (limb, &sub) in value[..last].iter_mut().zip(subtrahend) {
    let difference = limb.cast_signed().wrapping_sub(sub.cast_signed()).wrapping_add(carry);
    carry = difference >> 62;
    *limb = difference.cast_unsigned() & PRIVATE_S62_MASK;
  }
  value[last] = value[last]
    .cast_signed()
    .wrapping_sub(subtrahend[last].cast_signed())
    .wrapping_add(carry)
    .cast_unsigned();
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
#[inline(always)]
const fn private_i31_low_u32(value: u64) -> u32 {
  let bytes = value.to_le_bytes();
  u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
#[inline(always)]
fn private_i31_cond_negate_factor(value: i64, negate: i64) -> i64 {
  let mask = 0i64.wrapping_sub(negate);
  (value ^ mask).wrapping_add(negate)
}

#[cfg(any(test, target_arch = "riscv32"))]
#[inline(never)]
fn private_i31_ct_mul_u64_low(lhs: u64, rhs: u64) -> u64 {
  let mut product = 0u64;
  let mut multiplicand = lhs;
  let mut multiplier = rhs;
  for _ in 0..u64::BITS {
    // Keep LLVM from recognizing this fixed-work product and lowering it back
    // to a target multiply. The CT artifact gate independently rejects scalar
    // multiply in RV32 evidence closures.
    let selected_bit = core::hint::black_box(multiplier & 1);
    product = product.wrapping_add(multiplicand & 0u64.wrapping_sub(selected_bit));
    multiplicand <<= 1;
    multiplier >>= 1;
  }
  product
}

#[cfg(any(test, target_arch = "riscv32"))]
#[inline(never)]
fn private_i31_ct_mul_u32_low(lhs: u32, rhs: u32) -> u32 {
  let mut product = 0u32;
  let mut multiplicand = lhs;
  let mut multiplier = rhs;
  for _ in 0..u32::BITS {
    let selected_bit = core::hint::black_box(multiplier & 1);
    product = product.wrapping_add(multiplicand & 0u32.wrapping_sub(selected_bit));
    multiplicand <<= 1;
    multiplier >>= 1;
  }
  product
}

/// RV64 low product of the operands with their top bits forced set.
///
/// Some RV64 multipliers finish early for operands with leading zeros. With padding bits `a` and
/// `b`, `(lhs + 2^63·a)(rhs + 2^63·b) ≡ lhs·rhs + 2^63·(a·rhs + b·lhs) (mod 2^64)`, and adding
/// `2^63` flips the top bit, so one XOR recovers the product.
#[cfg(any(test, target_arch = "riscv64"))]
#[inline(always)]
fn private_i31_mul_u64_low_riscv64(lhs: u64, rhs: u64) -> u64 {
  const HIGH_BIT: u64 = 1 << 63;
  let lhs_padding = (lhs >> 63) ^ 1;
  let rhs_padding = (rhs >> 63) ^ 1;
  let product = core::hint::black_box(lhs | HIGH_BIT).wrapping_mul(core::hint::black_box(rhs | HIGH_BIT));
  product ^ ((((lhs_padding & rhs) ^ (rhs_padding & lhs)) & 1) << 63)
}

/// The low product of secret-fed words in the 31-bit inverse.
///
/// RV32 uses the fixed-work multiplier and RV64 the normalized hardware multiplier; IBM Z uses
/// the hardware multiplier, as the RSA exponentiation does. Native timing evidence qualifies
/// each target.
#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
#[inline(always)]
pub(super) fn private_i31_mul_u64_low(lhs: u64, rhs: u64) -> u64 {
  #[cfg(target_arch = "riscv32")]
  {
    private_i31_ct_mul_u64_low(lhs, rhs)
  }

  #[cfg(target_arch = "riscv64")]
  {
    private_i31_mul_u64_low_riscv64(lhs, rhs)
  }

  #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
  {
    lhs.wrapping_mul(rhs)
  }
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
#[inline(always)]
fn private_i31_mul_u32_low(lhs: u32, rhs: u32) -> u32 {
  #[cfg(target_arch = "riscv32")]
  {
    private_i31_ct_mul_u32_low(lhs, rhs)
  }

  #[cfg(target_arch = "riscv64")]
  {
    private_i31_low_u32(private_i31_mul_u64_low_riscv64(u64::from(lhs), u64::from(rhs)))
  }

  #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
  {
    lhs.wrapping_mul(rhs)
  }
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
#[inline]
pub(super) fn private_i31_word_count(bit_len: usize) -> usize {
  bit_len.strict_add(30) / 31
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
fn private_be_to_i31(input: &[u8], out: &mut [u32]) {
  out.fill(0);
  let mut accumulator = 0u64;
  let mut accumulator_bits = 0usize;
  let mut word_index = 0usize;
  for &byte in input.iter().rev() {
    accumulator |= u64::from(byte) << accumulator_bits;
    accumulator_bits = accumulator_bits.strict_add(8);
    if accumulator_bits >= 31 {
      out[word_index] = private_i31_low_u32(accumulator) & PRIVATE_I31_MASK;
      word_index = word_index.strict_add(1);
      accumulator >>= 31;
      accumulator_bits = accumulator_bits.strict_sub(31);
    }
  }
  if word_index < out.len() {
    out[word_index] = private_i31_low_u32(accumulator);
  }
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
fn private_i31_to_be(input: &[u32], out: &mut [u8]) {
  let mut accumulator = 0u64;
  let mut accumulator_bits = 0usize;
  let mut word_index = 0usize;
  for byte in out.iter_mut().rev() {
    if accumulator_bits < 8 {
      let word = input.get(word_index).copied().unwrap_or(0);
      accumulator |= u64::from(word) << accumulator_bits;
      accumulator_bits = accumulator_bits.strict_add(31);
      word_index = word_index.strict_add(1);
    }
    *byte = accumulator.to_le_bytes()[0];
    accumulator >>= 8;
    accumulator_bits = accumulator_bits.strict_sub(8);
  }
}

/*
 * The batched extended-binary-GCD structure below is derived from BearSSL's
 * i31 modular-division implementation:
 * https://www.bearssl.org/gitweb/?p=BearSSL;a=blob;f=src/int/i31_moddiv.c
 *
 * Copyright (c) 2018 Thomas Pornin <pornin@bolet.org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
fn private_i31_reduction_factors(a: &[u32], b: &[u32]) -> (i64, i64, i64, i64) {
  debug_assert_eq!(a.len(), b.len());
  let mut first_mask = u32::MAX;
  let mut second_mask = u32::MAX;
  let mut a_top = 0u32;
  let mut a_next = 0u32;
  let mut b_top = 0u32;
  let mut b_next = 0u32;
  for index in (0..a.len()).rev() {
    let a_word = a[index];
    let b_word = b[index];
    a_top ^= (a_top ^ a_word) & first_mask;
    a_next ^= (a_next ^ a_word) & second_mask;
    b_top ^= (b_top ^ b_word) & first_mask;
    b_next ^= (b_next ^ b_word) & second_mask;
    second_mask = first_mask;
    first_mask &= ((a_word | b_word).wrapping_add(PRIVATE_I31_MASK) >> 31).wrapping_sub(1);
  }

  a_next |= a_top & second_mask;
  a_top &= !second_mask;
  b_next |= b_top & second_mask;
  b_top &= !second_mask;
  let mut a_high = (u64::from(a_top) << 31).wrapping_add(u64::from(a_next));
  let mut b_high = (u64::from(b_top) << 31).wrapping_add(u64::from(b_next));
  let mut a_low = a.first().copied().unwrap_or(0);
  let mut b_low = b.first().copied().unwrap_or(0);
  let mut pa = 1i64;
  let mut pb = 0i64;
  let mut qa = 0i64;
  let mut qb = 1i64;

  for bit in 0..31 {
    let difference = b_high.wrapping_sub(a_high);
    let a_greater = private_i31_low_u32((difference ^ ((a_high ^ b_high) & (a_high ^ difference))) >> 63);
    let a_odd = (a_low >> bit) & 1;
    let b_odd = (b_low >> bit) & 1;
    let subtract_b_from_a = a_odd & b_odd & a_greater;
    let subtract_a_from_b = a_odd & b_odd & (a_greater ^ 1);
    let shift_a = subtract_b_from_a | (a_odd ^ 1);

    let a_mask_u32 = 0u32.wrapping_sub(subtract_b_from_a);
    let a_mask_u64 = 0u64.wrapping_sub(u64::from(subtract_b_from_a));
    let a_mask_i64 = 0i64.wrapping_sub(i64::from(subtract_b_from_a));
    a_low = a_low.wrapping_sub(b_low & a_mask_u32);
    a_high = a_high.wrapping_sub(b_high & a_mask_u64);
    pa = pa.wrapping_sub(qa & a_mask_i64);
    pb = pb.wrapping_sub(qb & a_mask_i64);

    let b_mask_u32 = 0u32.wrapping_sub(subtract_a_from_b);
    let b_mask_u64 = 0u64.wrapping_sub(u64::from(subtract_a_from_b));
    let b_mask_i64 = 0i64.wrapping_sub(i64::from(subtract_a_from_b));
    b_low = b_low.wrapping_sub(a_low & b_mask_u32);
    b_high = b_high.wrapping_sub(a_high & b_mask_u64);
    qa = qa.wrapping_sub(pa & b_mask_i64);
    qb = qb.wrapping_sub(pb & b_mask_i64);

    let keep_a_mask_u32 = shift_a.wrapping_sub(1);
    let keep_a_mask_i64 = i64::from(shift_a).wrapping_sub(1);
    a_low = a_low.wrapping_add(a_low & keep_a_mask_u32);
    pa = pa.wrapping_add(pa & keep_a_mask_i64);
    pb = pb.wrapping_add(pb & keep_a_mask_i64);
    let shift_a_mask_u64 = 0u64.wrapping_sub(u64::from(shift_a));
    a_high ^= (a_high ^ (a_high >> 1)) & shift_a_mask_u64;

    let keep_b_mask_u32 = 0u32.wrapping_sub(shift_a);
    let keep_b_mask_i64 = 0i64.wrapping_sub(i64::from(shift_a));
    b_low = b_low.wrapping_add(b_low & keep_b_mask_u32);
    qa = qa.wrapping_add(qa & keep_b_mask_i64);
    qb = qb.wrapping_add(qb & keep_b_mask_i64);
    let shift_b_mask_u64 = u64::from(shift_a).wrapping_sub(1);
    b_high ^= (b_high ^ (b_high >> 1)) & shift_b_mask_u64;
  }

  (pa, pb, qa, qb)
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
fn private_i31_co_reduce(a: &mut [u32], b: &mut [u32], pa: i64, pb: i64, qa: i64, qb: i64) -> u32 {
  debug_assert_eq!(a.len(), b.len());
  let mut a_carry = 0i64;
  let mut b_carry = 0i64;
  for index in 0..a.len() {
    let a_word = u64::from(a[index]);
    let b_word = u64::from(b[index]);
    let a_value = private_i31_mul_u64_low(a_word, pa.cast_unsigned())
      .wrapping_add(private_i31_mul_u64_low(b_word, pb.cast_unsigned()))
      .wrapping_add(a_carry.cast_unsigned());
    let b_value = private_i31_mul_u64_low(a_word, qa.cast_unsigned())
      .wrapping_add(private_i31_mul_u64_low(b_word, qb.cast_unsigned()))
      .wrapping_add(b_carry.cast_unsigned());
    if index > 0 {
      a[index.strict_sub(1)] = private_i31_low_u32(a_value) & PRIVATE_I31_MASK;
      b[index.strict_sub(1)] = private_i31_low_u32(b_value) & PRIVATE_I31_MASK;
    }
    a_carry = ((a_value >> 31) ^ (1u64 << 32)).wrapping_sub(1u64 << 32).cast_signed();
    b_carry = ((b_value >> 31) ^ (1u64 << 32)).wrapping_sub(1u64 << 32).cast_signed();
  }
  let last = a.len().strict_sub(1);
  a[last] = private_i31_low_u32(a_carry.cast_unsigned());
  b[last] = private_i31_low_u32(b_carry.cast_unsigned());
  let a_negative = private_i31_low_u32(a_carry.cast_unsigned() >> 63);
  let b_negative = private_i31_low_u32(b_carry.cast_unsigned() >> 63);
  private_i31_cond_negate(a, a_negative);
  private_i31_cond_negate(b, b_negative);
  a_negative | (b_negative << 1)
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
fn private_i31_cond_negate(value: &mut [u32], negate: u32) {
  let mut carry = negate;
  let mask = 0u32.wrapping_sub(negate) >> 1;
  for word in value {
    let negated = (*word ^ mask).wrapping_add(carry);
    *word = negated & PRIVATE_I31_MASK;
    carry = negated >> 31;
  }
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
#[expect(
  clippy::too_many_arguments,
  reason = "the two-row batched co-reduction matrix and immutable modulus are one arithmetic operation"
)]
fn private_i31_co_reduce_mod(
  u: &mut [u32],
  v: &mut [u32],
  pa: i64,
  pb: i64,
  qa: i64,
  qb: i64,
  modulus: &[u32],
  modulus_inverse: u32,
) {
  debug_assert_eq!(u.len(), v.len());
  debug_assert_eq!(u.len(), modulus.len());
  let u_factor = private_i31_mul_u32_low(
    private_i31_mul_u32_low(u[0], private_i31_low_u32(pa.cast_unsigned()))
      .wrapping_add(private_i31_mul_u32_low(v[0], private_i31_low_u32(pb.cast_unsigned()))),
    modulus_inverse,
  ) & PRIVATE_I31_MASK;
  let v_factor = private_i31_mul_u32_low(
    private_i31_mul_u32_low(u[0], private_i31_low_u32(qa.cast_unsigned()))
      .wrapping_add(private_i31_mul_u32_low(v[0], private_i31_low_u32(qb.cast_unsigned()))),
    modulus_inverse,
  ) & PRIVATE_I31_MASK;
  let mut u_carry = 0i64;
  let mut v_carry = 0i64;
  for index in 0..u.len() {
    let u_word = u64::from(u[index]);
    let v_word = u64::from(v[index]);
    let modulus_word = u64::from(modulus[index]);
    let u_value = private_i31_mul_u64_low(u_word, pa.cast_unsigned())
      .wrapping_add(private_i31_mul_u64_low(v_word, pb.cast_unsigned()))
      .wrapping_add(private_i31_mul_u64_low(modulus_word, u64::from(u_factor)))
      .wrapping_add(u_carry.cast_unsigned());
    let v_value = private_i31_mul_u64_low(u_word, qa.cast_unsigned())
      .wrapping_add(private_i31_mul_u64_low(v_word, qb.cast_unsigned()))
      .wrapping_add(private_i31_mul_u64_low(modulus_word, u64::from(v_factor)))
      .wrapping_add(v_carry.cast_unsigned());
    if index > 0 {
      u[index.strict_sub(1)] = private_i31_low_u32(u_value) & PRIVATE_I31_MASK;
      v[index.strict_sub(1)] = private_i31_low_u32(v_value) & PRIVATE_I31_MASK;
    }
    u_carry = ((u_value >> 31) ^ (1u64 << 32)).wrapping_sub(1u64 << 32).cast_signed();
    v_carry = ((v_value >> 31) ^ (1u64 << 32)).wrapping_sub(1u64 << 32).cast_signed();
  }
  let last = u.len().strict_sub(1);
  u[last] = private_i31_low_u32(u_carry.cast_unsigned());
  v[last] = private_i31_low_u32(v_carry.cast_unsigned());
  private_i31_finish_mod(u, modulus, private_i31_low_u32(u_carry.cast_unsigned() >> 63));
  private_i31_finish_mod(v, modulus, private_i31_low_u32(v_carry.cast_unsigned() >> 63));
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
fn private_i31_finish_mod(value: &mut [u32], modulus: &[u32], negative: u32) {
  debug_assert_eq!(value.len(), modulus.len());
  let mut borrow = 0u32;
  for index in 0..value.len() {
    borrow = value[index].wrapping_sub(modulus[index]).wrapping_sub(borrow) >> 31;
  }

  let negative_mask = 0u32.wrapping_sub(negative) >> 1;
  let modulus_mask = 0u32.wrapping_sub(negative | 1u32.wrapping_sub(borrow));
  let mut carry = negative;
  for index in 0..value.len() {
    let modulus_word = (modulus[index] ^ negative_mask) & modulus_mask;
    let reduced = value[index].wrapping_sub(modulus_word).wrapping_sub(carry);
    value[index] = reduced & PRIVATE_I31_MASK;
    carry = reduced >> 31;
  }
}

#[cfg(test)]
mod tests;
