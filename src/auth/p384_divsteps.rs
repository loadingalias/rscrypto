//! Constant-time P-384 field inversion by Bernstein--Yang divsteps.
//!
//! This computes the same result as the portable Fermat chain in
//! `FieldElement::invert`: the Montgomery-form inverse of a Montgomery-form
//! input, and zero for zero. It follows Bernstein and Yang, "Fast
//! constant-time gcd computation and modular inversion", with the original
//! divstep and its proven bound of `floor((49 * 384 + 57) / 17) = 1110`
//! iterations for 384-bit inputs. Eighteen batches of 62 divsteps run 1116.
//!
//! Each batch derives a 2x2 transition matrix from the low 64 bits of `f` and
//! `g`, which stay exact for 62 steps. After `i` steps the matrix has
//! `|u| + |v| <= 2^i` and `|q| + |r| <= 2^i`, because every step either
//! doubles a row or adds one row to the other. The matrix is then applied to
//! the signed radix-2^62 values `f` and `g`, where the low 62 bits cancel,
//! and to `d` and `e` modulo `p`, where a multiple of `p` makes the division
//! by 2^62 exact. Starting from `f = p`, `g = x`, `d = 0`, and `e = R^2 mod p`
//! keeps `f = d * x * R^-2` and `g = e * x * R^-2` modulo `p`. At the end
//! `g = 0` and `f = +-1` for nonzero `x`, so `+-d` is `x^-1 * R^2`, the
//! Montgomery form of the inverse of the Montgomery-form input.
//!
//! Every loop has a public, fixed trip count, and every secret-dependent
//! choice is a mask. The divstep loop is AArch64 or x86-64 inline assembly
//! so that its masks stay integer ALU operations without compiler-inserted
//! selects. Like the
//! Fermat chain, this does not clear its intermediate values.

const M62: u64 = (1 << 62) - 1;
const LIMBS62: usize = 7;
const BATCHES: usize = 18;
/// Also the literal trip count in the `divsteps` assembly.
const STEPS_PER_BATCH: usize = 62;

const _: () = assert!(BATCHES * STEPS_PER_BATCH >= 1110);

/// Signed radix-2^62 integer: limbs 0..6 hold 62 bits and limb 6 is signed.
type Signed62 = [i64; LIMBS62];

/// `p` in radix 2^62.
const MODULUS: Signed62 = to_signed62(&super::FIELD_MODULUS.0);
/// `p^-1 mod 2^62`.
const MODULUS_INV62: u64 = 0x3fff_fffe_ffff_ffff;

const _: () = assert!(MODULUS_INV62.wrapping_mul(super::FIELD_MODULUS.0[0]) & M62 == 1);

/// A batch transition matrix `[[u, v], [q, r]]` scaled by 2^62.
struct Transition {
  u: i64,
  v: i64,
  q: i64,
  r: i64,
}

const fn to_signed62(value: &[u64; 6]) -> Signed62 {
  let mut out = [0i64; LIMBS62];
  let mut index = 0;
  while index < LIMBS62 {
    let bit = index.strict_mul(62);
    let limb = bit / 64;
    let shift = bit % 64;
    let mut word = value[limb] >> shift;
    if shift > 2 && limb < 5 {
      word |= value[limb.strict_add(1)] << (64usize.strict_sub(shift));
    }
    out[index] = (word & M62).cast_signed();
    index = index.strict_add(1);
  }
  out
}

/// Convert a value in `[0, 2^384)` with normalized limbs back to 64-bit limbs.
///
/// Output limb `j` starts at bit `64j = 62 * (j + j / 31) + 2j mod 62`; for
/// `j < 6` the offset `2j` is at most 10, so two radix-2^62 limbs cover it.
fn from_signed62(value: &Signed62) -> [u64; 6] {
  let mut out = [0u64; 6];
  for (index, limb) in out.iter_mut().enumerate() {
    let bit = index.strict_mul(64);
    let low = bit / 62;
    let shift = bit % 62;
    *limb =
      (value[low].cast_unsigned() >> shift) | (value[low.strict_add(1)].cast_unsigned() << (62usize.strict_sub(shift)));
  }
  out
}

/// Return the low 64 bits of `value`.
#[inline(always)]
fn low_u64(value: i128) -> u64 {
  let [b0, b1, b2, b3, b4, b5, b6, b7, ..] = value.to_le_bytes();
  u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7])
}

/// Return `value` as `i64` when it is known to fit.
#[inline(always)]
fn fitting_i64(value: i128) -> i64 {
  debug_assert!(i64::try_from(value).is_ok());
  low_u64(value).cast_signed()
}

/// Run 62 divsteps on the low 64 bits of `f` and `g`.
///
/// Swap and negation are expressed as `(x ^ mask) - mask` and `x & mask`, so
/// the loop has no data-dependent branch or select. On AArch64 its integer
/// ALU instructions leave NZCV alone apart from the public loop counter; on
/// x86-64 only that counter's `DEC`/`JNZ` consumes the flags.
/// The module bounds keep `|u|, |v|, |q|, |r| <= 2^62` and `|delta| < 1200`,
/// so the 64-bit register arithmetic never wraps for those values; `f` and
/// `g` are deliberately tracked modulo 2^64.
#[inline(always)]
fn divsteps(mut delta: i64, f: u64, g: u64) -> (i64, Transition) {
  let (u, v, q, r): (i64, i64, i64, i64);
  // SAFETY: The block accesses no memory and uses no stack. It uses only base
  // A64 integer instructions, which every aarch64 target provides. Every
  // register it writes is a declared output, and NZCV is clobbered by default.
  #[cfg(target_arch = "aarch64")]
  unsafe {
    core::arch::asm!(
      "mov {u}, #1",
      "mov {v}, #0",
      "mov {q}, #0",
      "mov {r}, #1",
      "mov {count}, #62",
      "2:",
      // swap = all ones when delta > 0; odd = all ones when g is odd.
      "neg {swap}, {delta}",
      "asr {swap}, {swap}, #63",
      "sbfx {odd}, {g}, #0, #1",
      // g += +-f, q += +-u, r += +-v when g is odd.
      "eor {t1}, {f}, {swap}",
      "sub {t1}, {t1}, {swap}",
      "and {t1}, {t1}, {odd}",
      "add {g}, {g}, {t1}",
      "eor {t2}, {u}, {swap}",
      "sub {t2}, {t2}, {swap}",
      "and {t2}, {t2}, {odd}",
      "add {q}, {q}, {t2}",
      "eor {t3}, {v}, {swap}",
      "sub {t3}, {t3}, {swap}",
      "and {t3}, {t3}, {odd}",
      "add {r}, {r}, {t3}",
      // On a swap: delta = 1 - delta and the new g row moves into f.
      "and {swap}, {swap}, {odd}",
      "eor {delta}, {delta}, {swap}",
      "sub {delta}, {delta}, {swap}",
      "add {delta}, {delta}, #1",
      "and {t1}, {g}, {swap}",
      "add {f}, {f}, {t1}",
      "and {t2}, {q}, {swap}",
      "add {u}, {u}, {t2}",
      "and {t3}, {r}, {swap}",
      "add {v}, {v}, {t3}",
      // g is even; doubling u and v keeps the matrix integral.
      "lsr {g}, {g}, #1",
      "lsl {u}, {u}, #1",
      "lsl {v}, {v}, #1",
      "subs {count}, {count}, #1",
      "b.ne 2b",
      delta = inout(reg) delta,
      f = inout(reg) f => _,
      g = inout(reg) g => _,
      u = out(reg) u,
      v = out(reg) v,
      q = out(reg) q,
      r = out(reg) r,
      count = out(reg) _,
      swap = out(reg) _,
      odd = out(reg) _,
      t1 = out(reg) _,
      t2 = out(reg) _,
      t3 = out(reg) _,
      options(nomem, nostack),
    );
  }
  // SAFETY: The block accesses no memory and uses no stack. It uses only
  // baseline x86-64 integer instructions. Every register it writes is a
  // declared output, and flags are clobbered by default.
  #[cfg(target_arch = "x86_64")]
  unsafe {
    core::arch::asm!(
      "mov {u}, 1",
      "xor {v:e}, {v:e}",
      "xor {q:e}, {q:e}",
      "mov {r}, 1",
      "mov {count:e}, 62",
      "2:",
      "mov {swap}, {delta}",
      "neg {swap}",
      "sar {swap}, 63",
      "mov {odd}, {g}",
      "and {odd}, 1",
      "neg {odd}",
      "mov {t1}, {f}",
      "xor {t1}, {swap}",
      "sub {t1}, {swap}",
      "and {t1}, {odd}",
      "add {g}, {t1}",
      "mov {t2}, {u}",
      "xor {t2}, {swap}",
      "sub {t2}, {swap}",
      "and {t2}, {odd}",
      "add {q}, {t2}",
      "mov {t3}, {v}",
      "xor {t3}, {swap}",
      "sub {t3}, {swap}",
      "and {t3}, {odd}",
      "add {r}, {t3}",
      "and {swap}, {odd}",
      "xor {delta}, {swap}",
      "sub {delta}, {swap}",
      "add {delta}, 1",
      "mov {t1}, {g}",
      "and {t1}, {swap}",
      "add {f}, {t1}",
      "mov {t2}, {q}",
      "and {t2}, {swap}",
      "add {u}, {t2}",
      "mov {t3}, {r}",
      "and {t3}, {swap}",
      "add {v}, {t3}",
      "shr {g}, 1",
      "add {u}, {u}",
      "add {v}, {v}",
      "dec {count:e}",
      "jnz 2b",
      delta = inout(reg) delta,
      f = inout(reg) f => _,
      g = inout(reg) g => _,
      u = out(reg) u,
      v = out(reg) v,
      q = out(reg) q,
      r = out(reg) r,
      count = out(reg) _,
      swap = out(reg) _,
      odd = out(reg) _,
      t1 = out(reg) _,
      t2 = out(reg) _,
      t3 = out(reg) _,
      options(nomem, nostack),
    );
  }
  (delta, Transition { u, v, q, r })
}

#[inline(always)]
fn wide(left: i64, right: i64) -> i128 {
  i128::from(left).strict_mul(i128::from(right))
}

/// Replace `(f, g)` with `(u * f + v * g, q * f + r * g) / 2^62`.
///
/// Each limb sum stays below `2^126` in magnitude because matrix rows have
/// absolute sum at most `2^62` and limbs are below `2^62`.
fn update_fg(f: &mut Signed62, g: &mut Signed62, t: &Transition) {
  let mut cf = wide(t.u, f[0]).strict_add(wide(t.v, g[0]));
  let mut cg = wide(t.q, f[0]).strict_add(wide(t.r, g[0]));
  debug_assert!(low_u64(cf) & M62 == 0 && low_u64(cg) & M62 == 0);
  cf >>= 62;
  cg >>= 62;
  for index in 1..LIMBS62 {
    cf = cf.strict_add(wide(t.u, f[index])).strict_add(wide(t.v, g[index]));
    cg = cg.strict_add(wide(t.q, f[index])).strict_add(wide(t.r, g[index]));
    f[index.strict_sub(1)] = (low_u64(cf) & M62).cast_signed();
    g[index.strict_sub(1)] = (low_u64(cg) & M62).cast_signed();
    cf >>= 62;
    cg >>= 62;
  }
  f[LIMBS62 - 1] = fitting_i64(cf);
  g[LIMBS62 - 1] = fitting_i64(cg);
}

/// Replace `(d, e)` with `(u * d + v * e, q * d + r * e) / 2^62 mod p`.
///
/// Inputs and outputs lie in `(-2p, p)`. Adding `p` times each coefficient of
/// a negative input first maps both inputs into `(-p, p)`; the matrix row
/// then gives a value in `(-2^62 * p, 2^62 * p)`, and subtracting a multiple
/// of `p` below `2^62` that clears the low 62 bits leaves `(-2p, p)` after
/// the exact division.
fn update_de(d: &mut Signed62, e: &mut Signed62, t: &Transition) {
  let sign_d = d[LIMBS62 - 1] >> 63;
  let sign_e = e[LIMBS62 - 1] >> 63;
  let mut md = (t.u & sign_d).strict_add(t.v & sign_e);
  let mut me = (t.q & sign_d).strict_add(t.r & sign_e);
  let mut cd = wide(t.u, d[0]).strict_add(wide(t.v, e[0]));
  let mut ce = wide(t.q, d[0]).strict_add(wide(t.r, e[0]));
  md = md.strict_sub((MODULUS_INV62.wrapping_mul(low_u64(cd)).wrapping_add(md.cast_unsigned()) & M62).cast_signed());
  me = me.strict_sub((MODULUS_INV62.wrapping_mul(low_u64(ce)).wrapping_add(me.cast_unsigned()) & M62).cast_signed());
  cd = cd.strict_add(wide(MODULUS[0], md));
  ce = ce.strict_add(wide(MODULUS[0], me));
  debug_assert!(low_u64(cd) & M62 == 0 && low_u64(ce) & M62 == 0);
  cd >>= 62;
  ce >>= 62;
  for index in 1..LIMBS62 {
    cd = cd
      .strict_add(wide(t.u, d[index]))
      .strict_add(wide(t.v, e[index]))
      .strict_add(wide(MODULUS[index], md));
    ce = ce
      .strict_add(wide(t.q, d[index]))
      .strict_add(wide(t.r, e[index]))
      .strict_add(wide(MODULUS[index], me));
    d[index.strict_sub(1)] = (low_u64(cd) & M62).cast_signed();
    e[index.strict_sub(1)] = (low_u64(ce) & M62).cast_signed();
    cd >>= 62;
    ce >>= 62;
  }
  d[LIMBS62 - 1] = fitting_i64(cd);
  e[LIMBS62 - 1] = fitting_i64(ce);
}

/// Add `p` to `value` when it is negative, renormalizing limbs.
fn add_modulus_if_negative(value: &mut Signed62) {
  let mask = value[LIMBS62 - 1] >> 63;
  let mut carry = 0i64;
  for (limb, modulus) in value.iter_mut().zip(MODULUS).take(LIMBS62 - 1) {
    carry = carry.strict_add(*limb).strict_add(modulus & mask);
    *limb = carry & M62.cast_signed();
    carry >>= 62;
  }
  value[LIMBS62 - 1] = value[LIMBS62 - 1]
    .strict_add(MODULUS[LIMBS62 - 1] & mask)
    .strict_add(carry);
}

/// Negate `value` when `mask` is all ones, renormalizing limbs.
fn negate_masked(value: &mut Signed62, mask: i64) {
  let mut carry = 0i64;
  for limb in value.iter_mut().take(LIMBS62 - 1) {
    carry = carry.strict_add((*limb ^ mask).strict_sub(mask));
    *limb = carry & M62.cast_signed();
    carry >>= 62;
  }
  value[LIMBS62 - 1] = (value[LIMBS62 - 1] ^ mask).strict_sub(mask).strict_add(carry);
}

/// Return the Montgomery-form inverse of a canonical Montgomery-form value,
/// or zero for zero.
pub(super) fn invert_montgomery(value: &[u64; 6]) -> [u64; 6] {
  let mut f = MODULUS;
  let mut g = to_signed62(value);
  let mut d = [0i64; LIMBS62];
  let mut e = to_signed62(&super::FIELD_R2.0);
  let mut delta = 1i64;
  for _ in 0..BATCHES {
    let f_low = f[0].cast_unsigned() | (f[1].cast_unsigned() << 62);
    let g_low = g[0].cast_unsigned() | (g[1].cast_unsigned() << 62);
    let (next_delta, transition) = divsteps(delta, f_low, g_low);
    delta = next_delta;
    update_fg(&mut f, &mut g, &transition);
    update_de(&mut d, &mut e, &transition);
  }
  // `f = +-1`, or `p` for a zero input whose `d` is zero. Map `d` from
  // `(-2p, p)` to `[0, p)` and apply the sign of `f`.
  add_modulus_if_negative(&mut d);
  negate_masked(&mut d, f[LIMBS62 - 1] >> 63);
  add_modulus_if_negative(&mut d);
  from_signed62(&d)
}
