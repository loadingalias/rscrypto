//! Fixed-iteration scalar inversion for the NIST curve orders.
//!
//! This is the single-divstep Bernstein--Yang construction specialized to the
//! P-256 and P-384 scalar moduli. It deliberately uses 32-bit digits carried in
//! 64-bit temporaries: every add and subtract then has a representable widened
//! result, so targets whose 64-bit carry lowering is unsuitable for secret
//! arithmetic do not need target-specific condition-code tricks here.
//!
//! The transition and iteration bounds follow Bernstein and Yang, "Fast
//! constant-time gcd computation and modular inversion". The correction
//! constants are independently reproducible as `2^(-steps) * R mod n`, where
//! `R = 2^bits`; their values agree with fiat-crypto's generated divstep
//! precomputation for these exact moduli.

use super::{Modulus, SecretScalar, Uint, is_p256_order_modulus, is_p384_order_modulus, montgomery_mul};
use crate::traits::ct;

const MAX_DIGITS: usize = 12;
const MAX_SIGNED_DIGITS: usize = MAX_DIGITS + 1;

#[inline(always)]
fn low_u32(value: u64) -> u32 {
  let [b0, b1, b2, b3, _, _, _, _] = value.to_le_bytes();
  u32::from_le_bytes([b0, b1, b2, b3])
}

#[inline(always)]
fn low_u32_signed(value: i64) -> u32 {
  let [b0, b1, b2, b3, _, _, _, _] = value.to_le_bytes();
  u32::from_le_bytes([b0, b1, b2, b3])
}

// floor((49 * 256 + 57) / 17) and floor((49 * 384 + 57) / 17).
// These public, modulus-specific counts are the proven single-divstep bounds.
const P256_DIVSTEPS: usize = 741;
const P384_DIVSTEPS: usize = 1_110;

const P256_MONTGOMERY_ONE: [u32; MAX_DIGITS] = [
  0x039c_daaf,
  0x0c46_353d,
  0x58e8_617b,
  0x4319_0552,
  0x0000_0000,
  0x0000_0000,
  0xffff_ffff,
  0x0000_0000,
  0,
  0,
  0,
  0,
];

const P384_MONTGOMERY_ONE: [u32; MAX_DIGITS] = [
  0x333a_d68d,
  0x1313_e695,
  0xb74f_5885,
  0xa7e5_f24d,
  0x0bc8_d220,
  0x389c_b27e,
  0,
  0,
  0,
  0,
  0,
  0,
];

const P256_DIVSTEP_CORRECTION_MONTGOMERY: [u32; MAX_DIGITS] = [
  0xb7fc_fbb5,
  0xd739_262f,
  0x2007_4414,
  0x8ac6_f75d,
  0xb5e3_c256,
  0xc674_28bf,
  0xeda7_aedf,
  0x4449_62f2,
  0,
  0,
  0,
  0,
];

const P384_DIVSTEP_CORRECTION_MONTGOMERY: [u32; MAX_DIGITS] = [
  0xe604_5b6a,
  0x4958_9ae0,
  0x8700_40ed,
  0x3c9a_5352,
  0x977d_c242,
  0xdacb_097e,
  0xd1ec_be36,
  0xb5ab_30a6,
  0x1f95_9973,
  0x97d7_a108,
  0xd271_92bc,
  0x2ba0_12f8,
];

struct DivstepState {
  delta: u32,
  f: [u32; MAX_SIGNED_DIGITS],
  g: [u32; MAX_SIGNED_DIGITS],
  v: [u32; MAX_DIGITS],
  r: [u32; MAX_DIGITS],
  next_f: [u32; MAX_SIGNED_DIGITS],
  next_g: [u32; MAX_SIGNED_DIGITS],
  next_v: [u32; MAX_DIGITS],
  next_r: [u32; MAX_DIGITS],
}

impl Drop for DivstepState {
  fn drop(&mut self) {
    ct::zeroize_words_no_fence(core::slice::from_mut(&mut self.delta));
    ct::zeroize_words_no_fence(&mut self.f);
    ct::zeroize_words_no_fence(&mut self.g);
    ct::zeroize_words_no_fence(&mut self.v);
    ct::zeroize_words_no_fence(&mut self.r);
    ct::zeroize_words_no_fence(&mut self.next_f);
    ct::zeroize_words_no_fence(&mut self.next_g);
    ct::zeroize_words_no_fence(&mut self.next_v);
    ct::zeroize_words_no_fence(&mut self.next_r);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

pub(super) fn invert_order_montgomery<const L: usize>(value: Uint<L>, modulus: &'static Modulus<L>) -> Option<Uint<L>> {
  if is_p256_order_modulus(modulus) {
    return Some(invert_montgomery(
      value,
      modulus,
      8,
      P256_DIVSTEPS,
      P256_MONTGOMERY_ONE,
      P256_DIVSTEP_CORRECTION_MONTGOMERY,
    ));
  }
  if is_p384_order_modulus(modulus) {
    return Some(invert_montgomery(
      value,
      modulus,
      12,
      P384_DIVSTEPS,
      P384_MONTGOMERY_ONE,
      P384_DIVSTEP_CORRECTION_MONTGOMERY,
    ));
  }
  None
}

fn invert_montgomery<const L: usize>(
  value: Uint<L>,
  modulus: &'static Modulus<L>,
  digits: usize,
  steps: usize,
  montgomery_one: [u32; MAX_DIGITS],
  correction_montgomery: [u32; MAX_DIGITS],
) -> Uint<L> {
  debug_assert_eq!(digits, L.strict_mul(2));

  let modulus_digits = uint_to_digits(modulus.value);
  let mut state = DivstepState {
    delta: 1,
    f: [0u32; MAX_SIGNED_DIGITS],
    g: [0u32; MAX_SIGNED_DIGITS],
    v: [0u32; MAX_DIGITS],
    r: montgomery_one,
    next_f: [0u32; MAX_SIGNED_DIGITS],
    next_g: [0u32; MAX_SIGNED_DIGITS],
    next_v: [0u32; MAX_DIGITS],
    next_r: [0u32; MAX_DIGITS],
  };
  state.f[..digits].copy_from_slice(&modulus_digits[..digits]);
  let mut value_digits = uint_to_digits(value);
  state.g[..digits].copy_from_slice(&value_digits[..digits]);
  ct::zeroize_words_no_fence(&mut value_digits);

  for _ in 0..steps {
    divstep(&mut state, &modulus_digits, digits);
  }

  // The divstep invariant gives v * value = f * 2^steps (mod n).
  // At the fixed bound f is -1 or 1 for every non-zero scalar. Normalize
  // its sign and multiply by 2^-steps in the Montgomery domain.
  let f_negative = state.f[digits] >> 31;
  sub_mod(
    &mut state.next_v,
    &[0u32; MAX_DIGITS],
    &state.v,
    &modulus_digits,
    digits,
  );
  select_digits(&mut state.next_r, &state.v, &state.next_v, f_negative, digits);

  let normalized = SecretScalar::new(digits_to_uint::<L>(&state.next_r, digits));
  let correction = digits_to_uint::<L>(&correction_montgomery, digits);
  montgomery_mul(normalized.value(), correction, modulus)
}

#[inline(always)]
fn divstep(state: &mut DivstepState, modulus: &[u32; MAX_DIGITS], digits: usize) {
  let signed_digits = digits.strict_add(1);
  let g_odd = state.g[0] & 1;
  let delta_positive = state.delta.wrapping_neg() >> 31;
  let swap = delta_positive & g_odd;

  let delta_plus_one = state.delta.wrapping_add(1);
  let one_minus_delta = 1u32.wrapping_sub(state.delta);
  state.delta = select_word(delta_plus_one, one_minus_delta, swap);

  select_digits(&mut state.next_f, &state.f, &state.g, swap, signed_digits);

  twos_complement(&mut state.next_g, &state.f, signed_digits);
  select_digits_in_place(&mut state.next_g, &state.g, swap, signed_digits);
  let selected_odd = state.next_g[0] & 1;
  add_masked_in_place(&mut state.next_g, &state.next_f, selected_odd, signed_digits);
  arithmetic_shr1(&mut state.next_g, signed_digits);

  select_digits(&mut state.next_v, &state.v, &state.r, swap, digits);
  double_mod_in_place(&mut state.next_v, modulus, digits);

  sub_mod(&mut state.next_r, &[0u32; MAX_DIGITS], &state.v, modulus, digits);
  select_digits_in_place(&mut state.next_r, &state.r, swap, digits);
  let mut selected_vr = [0u32; MAX_DIGITS];
  select_digits(&mut selected_vr, &state.v, &state.r, swap, digits);
  add_masked_mod_in_place(&mut state.next_r, &selected_vr, selected_odd, modulus, digits);
  ct::zeroize_words_no_fence(&mut selected_vr);

  core::mem::swap(&mut state.f, &mut state.next_f);
  core::mem::swap(&mut state.g, &mut state.next_g);
  core::mem::swap(&mut state.v, &mut state.next_v);
  core::mem::swap(&mut state.r, &mut state.next_r);
}

#[inline(always)]
fn select_word(lhs: u32, rhs: u32, choice: u32) -> u32 {
  let mask = 0u32.wrapping_sub(choice & 1);
  lhs ^ (mask & (lhs ^ rhs))
}

#[inline(always)]
fn select_digits<const N: usize>(out: &mut [u32; N], lhs: &[u32; N], rhs: &[u32; N], choice: u32, digits: usize) {
  let mask = 0u32.wrapping_sub(choice & 1);
  for i in 0..digits {
    out[i] = lhs[i] ^ (mask & (lhs[i] ^ rhs[i]));
  }
}

#[inline(always)]
fn select_digits_in_place<const N: usize>(out: &mut [u32; N], lhs: &[u32; N], choice: u32, digits: usize) {
  let mask = 0u32.wrapping_sub(choice & 1);
  for i in 0..digits {
    out[i] = lhs[i] ^ (mask & (lhs[i] ^ out[i]));
  }
}

#[inline(always)]
fn twos_complement<const N: usize>(out: &mut [u32; N], value: &[u32; N], digits: usize) {
  let mut carry = 1u64;
  for i in 0..digits {
    let sum = u64::from(!value[i]).strict_add(carry);
    out[i] = low_u32(sum);
    carry = sum >> 32;
  }
}

#[inline(always)]
fn add_masked_in_place<const N: usize>(acc: &mut [u32; N], rhs: &[u32; N], choice: u32, digits: usize) {
  let mask = 0u32.wrapping_sub(choice & 1);
  let mut carry = 0u64;
  for i in 0..digits {
    let sum = u64::from(acc[i]).strict_add(u64::from(rhs[i] & mask)).strict_add(carry);
    acc[i] = low_u32(sum);
    carry = sum >> 32;
  }
}

#[inline(always)]
fn arithmetic_shr1<const N: usize>(value: &mut [u32; N], digits: usize) {
  let sign = value[digits.strict_sub(1)] & 0x8000_0000;
  let mut carry = sign;
  for limb in value[..digits].iter_mut().rev() {
    let next = *limb << 31;
    *limb = (*limb >> 1) | carry;
    carry = next;
  }
}

#[inline(always)]
fn double_mod_in_place(value: &mut [u32; MAX_DIGITS], modulus: &[u32; MAX_DIGITS], digits: usize) {
  let mut copy = *value;
  add_mod(value, &copy, &copy, modulus, digits);
  ct::zeroize_words_no_fence(&mut copy);
}

#[inline(always)]
fn add_masked_mod_in_place(
  value: &mut [u32; MAX_DIGITS],
  rhs: &[u32; MAX_DIGITS],
  choice: u32,
  modulus: &[u32; MAX_DIGITS],
  digits: usize,
) {
  let mut selected = [0u32; MAX_DIGITS];
  select_digits(&mut selected, &[0u32; MAX_DIGITS], rhs, choice, digits);
  let mut copy = *value;
  add_mod(value, &copy, &selected, modulus, digits);
  ct::zeroize_words_no_fence(&mut copy);
  ct::zeroize_words_no_fence(&mut selected);
}

#[inline(always)]
fn add_mod(
  out: &mut [u32; MAX_DIGITS],
  lhs: &[u32; MAX_DIGITS],
  rhs: &[u32; MAX_DIGITS],
  modulus: &[u32; MAX_DIGITS],
  digits: usize,
) {
  let mut sum = [0u32; MAX_DIGITS];
  let mut reduced = [0u32; MAX_DIGITS];
  let mut carry = 0u64;
  for i in 0..digits {
    let word = u64::from(lhs[i]).strict_add(u64::from(rhs[i])).strict_add(carry);
    sum[i] = low_u32(word);
    carry = word >> 32;
  }

  let borrow = sub_digits(&mut reduced, &sum, modulus, digits);
  let use_reduced = low_u32(carry) | (borrow ^ 1);
  select_digits(out, &sum, &reduced, use_reduced, digits);
  ct::zeroize_words_no_fence(&mut sum);
  ct::zeroize_words_no_fence(&mut reduced);
}

#[inline(always)]
fn sub_mod(
  out: &mut [u32; MAX_DIGITS],
  lhs: &[u32; MAX_DIGITS],
  rhs: &[u32; MAX_DIGITS],
  modulus: &[u32; MAX_DIGITS],
  digits: usize,
) {
  let mut diff = [0u32; MAX_DIGITS];
  let borrow = sub_digits(&mut diff, lhs, rhs, digits);
  let mask = 0u32.wrapping_sub(borrow);
  let mut carry = 0u64;
  for i in 0..digits {
    let word = u64::from(diff[i])
      .strict_add(u64::from(modulus[i] & mask))
      .strict_add(carry);
    out[i] = low_u32(word);
    carry = word >> 32;
  }
  ct::zeroize_words_no_fence(&mut diff);
}

#[inline(always)]
fn sub_digits<const N: usize>(out: &mut [u32; N], lhs: &[u32; N], rhs: &[u32; N], digits: usize) -> u32 {
  let mut borrow = 0i64;
  for i in 0..digits {
    let diff = i64::from(lhs[i]).strict_sub(i64::from(rhs[i])).strict_sub(borrow);
    out[i] = low_u32_signed(diff);
    borrow = (diff >> 63) & 1;
  }
  low_u32_signed(borrow)
}

fn uint_to_digits<const L: usize>(value: Uint<L>) -> [u32; MAX_DIGITS] {
  let mut out = [0u32; MAX_DIGITS];
  let (pairs, remainder) = out.as_chunks_mut::<2>();
  debug_assert!(remainder.is_empty());
  for (pair, limb) in pairs.iter_mut().zip(value.0) {
    pair[0] = low_u32(limb);
    pair[1] = low_u32(limb >> 32);
  }
  out
}

fn digits_to_uint<const L: usize>(digits_value: &[u32; MAX_DIGITS], digits: usize) -> Uint<L> {
  debug_assert_eq!(digits, L.strict_mul(2));
  let mut out = [0u64; L];
  let (pairs, remainder) = digits_value[..digits].as_chunks::<2>();
  debug_assert!(remainder.is_empty());
  for (limb, pair) in out.iter_mut().zip(pairs) {
    *limb = u64::from(pair[0]) | (u64::from(pair[1]) << 32);
  }
  Uint(out)
}
