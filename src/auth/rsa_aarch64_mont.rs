//! Owned AArch64 Montgomery arithmetic for RSA widths that are multiples of four limbs.
//!
//! Products and Montgomery reduction both run through one register-strip kernel: each limb of
//! one operand is loaded once and multiplied by four limbs of the other, and the four-limb window
//! of the running sum stays in registers. Reduction adds `m·n` four multipliers at a time, with
//! the multipliers computed from the four lowest limbs of the running value.
//!
//! Every loop bound and address depends only on the public width; secret limbs reach only
//! multiplications, flag-chained additions, and the branch-free final subtraction.

use core::arch::asm;

use super::split_u128;

/// Whether the kernels support `words` limbs.
#[inline]
pub(super) fn supports(words: usize) -> bool {
  words >= 8 && words.is_multiple_of(4)
}

/// Scratch limbs the kernels need for `words`-limb operands.
#[inline]
pub(super) fn scratch_words(words: usize) -> usize {
  words.strict_mul(2)
}

/// `t[0..2n] = 2·t[0..2n] + Σ a_i²·2^(128·i)`; the result must fit `2n` limbs.
///
/// # Safety
///
/// `n` must be a nonzero even count, `a` valid for reading `n` limbs, and `t` valid for reading
/// and writing `2n` limbs.
#[inline(always)]
unsafe fn double_add_squares(t: *mut u64, a: *const u64, n: usize) {
  // SAFETY: The caller guarantees the reads of `a[0..n]` and the reads and writes of `t[0..2n]`;
  // the block touches no other memory and no stack. The carry crosses iterations in the flags, so
  // the loop counts down with SUB and CBNZ, which leave them intact.
  unsafe {
    asm!(
      "mov {prev}, xzr",
      "cmn xzr, xzr",
      "2:",
      "ldp {x0}, {x1}, [{t}]",
      "ldp {x2}, {x3}, [{t}, #16]",
      "ldp {ai}, {aj}, [{a}], #16",
      "extr {d0}, {x0}, {prev}, #63",
      "extr {d1}, {x1}, {x0}, #63",
      "extr {d2}, {x2}, {x1}, #63",
      "extr {d3}, {x3}, {x2}, #63",
      "mov {prev}, {x3}",
      "mul {lo}, {ai}, {ai}",
      "umulh {hi}, {ai}, {ai}",
      "mul {lo2}, {aj}, {aj}",
      "umulh {hi2}, {aj}, {aj}",
      "adcs {d0}, {d0}, {lo}",
      "adcs {d1}, {d1}, {hi}",
      "adcs {d2}, {d2}, {lo2}",
      "adcs {d3}, {d3}, {hi2}",
      "stp {d0}, {d1}, [{t}], #16",
      "stp {d2}, {d3}, [{t}], #16",
      "sub {n}, {n}, #2",
      "cbnz {n}, 2b",
      t = inout(reg) t => _,
      a = inout(reg) a => _,
      n = inout(reg) n => _,
      prev = out(reg) _, x0 = out(reg) _, x1 = out(reg) _, x2 = out(reg) _, x3 = out(reg) _,
      ai = out(reg) _, aj = out(reg) _,
      d0 = out(reg) _, d1 = out(reg) _, d2 = out(reg) _, d3 = out(reg) _,
      lo = out(reg) _, hi = out(reg) _, lo2 = out(reg) _, hi2 = out(reg) _,
      options(nostack),
    );
  }
}

/// The four limbs of `limbs` from `start`.
#[inline(always)]
fn quad(limbs: &[u64], start: usize) -> [u64; 4] {
  let mut out = [0u64; 4];
  out.copy_from_slice(&limbs[start..start.strict_add(4)]);
  out
}

/// `-modulus⁻¹ mod 2^256` from `n0 = -modulus⁻¹ mod 2^64`, by two Newton steps on the inverse.
pub(super) fn n0_256(modulus: &[u64], n0: u64) -> [u64; 4] {
  // x·(2 - modulus·x) doubles the correct low bits of an inverse x.
  let low = quad(modulus, 0);
  let mut x = [n0.wrapping_neg(), 0, 0, 0];
  for _ in 0..2 {
    let product = low_mul(&low, &x);
    let mut two_minus = [0u64; 4];
    let mut borrow = 0u64;
    for (out, (&two, &limb)) in two_minus.iter_mut().zip([2u64, 0, 0, 0].iter().zip(&product)) {
      let (difference, first) = two.overflowing_sub(limb);
      let (difference, second) = difference.overflowing_sub(borrow);
      *out = difference;
      borrow = u64::from(first | second);
    }
    x = low_mul(&x, &two_minus);
  }
  // Negate modulo 2^256.
  let mut out = [0u64; 4];
  let mut borrow = 0u64;
  for (out, &limb) in out.iter_mut().zip(&x) {
    let (difference, first) = 0u64.overflowing_sub(limb);
    let (difference, second) = difference.overflowing_sub(borrow);
    *out = difference;
    borrow = u64::from(first | second);
  }
  out
}

/// `a·b mod 2^256`.
fn low_mul(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
  let mut out = [0u64; 4];
  for (i, &ai) in a.iter().enumerate() {
    let mut carry = 0u64;
    for (j, &bj) in b.iter().enumerate().take(4usize.strict_sub(i)) {
      let index = i.strict_add(j);
      let sum = u128::from(out[index])
        .strict_add(u128::from(ai).strict_mul(u128::from(bj)))
        .strict_add(u128::from(carry));
      (out[index], carry) = split_u128(sum);
    }
  }
  out
}

/// `out = t·2^(-64·n) mod modulus` for `t < modulus·2^(64·n)` held in `t[0..2n]`; `k` is
/// `-modulus⁻¹ mod 2^256`.
///
/// Each strip takes four multipliers `m = (t[s..s + 4]·k) mod 2^256` from one 4×4 low product and
/// adds `m·modulus` at limb `s`, clearing those four limbs; the strip's carry joins the next
/// strip's window end. A strip's eighth step finishes limbs `s + 4..s + 8`, so the next strip's
/// multipliers are computed there and parked in `params`, off the strip boundary's critical path. The final subtraction writes `t[n..2n] - modulus` to `out`, then keeps it
/// when that did not borrow or the carry is set, and otherwise copies `t[n..2n]` back.
fn reduce(out: &mut [u64], t: &mut [u64], modulus: &[u64], k: &[u64; 4]) {
  let n = modulus.len();
  assert!(supports(n) && out.len() == n && t.len() >= n.strict_mul(2));
  // k, n, out, then the four multipliers of the next strip.
  let mut params = [k[0], k[1], k[2], k[3], n as u64, out.as_mut_ptr() as u64, 0, 0, 0, 0];
  // SAFETY: Every strip reads and writes `t[s..s + n + 4]` for `s` below `n` in steps of four,
  // inside `t[0..2n]`, and reads `modulus[0..n]`; its lookahead reads `t[s + 4..s + 8]`, also
  // inside `t[0..2n]`, and writes `params[6..10]`. The final subtraction reads `t[n..2n]` and
  // `modulus[0..n]` and writes `out[0..n]`, whose pointer and length `params` carries; `out` does
  // not overlap `t`. The block uses no stack. `n` is a multiple of four of at least eight, so the
  // first segment's eight steps exist and the second segment's count is checked before its loop.
  unsafe {
    asm!(
      // Reduction strips: carry 0, strip base tp, n/4 strips.
      "mov {carry}, xzr",
      // m = (w·k) mod 2^256 for the four limbs w at tp + 0, stored for the strip that starts there.
      "ldp {aj}, {tj}, [{tp}, #0]",
      "ldp {l2}, {l3}, [{tp}, #16]",
      "ldp {h0}, {h1}, [{prm}]",
      "mul {l0}, {aj}, {h0}",
      "umulh {l1}, {aj}, {h0}",
      "str {l0}, [{prm}, #48]",
      "mul {h2}, {aj}, {h1}",
      "adds {l1}, {l1}, {h2}",
      "cset {h3}, cs",
      "mul {h2}, {tj}, {h0}",
      "adds {l1}, {l1}, {h2}",
      "cinc {h3}, {h3}, cs",
      "str {l1}, [{prm}, #56]",
      "umulh {l0}, {aj}, {h1}",
      "adds {l0}, {l0}, {h3}",
      "cset {l1}, cs",
      "umulh {h2}, {tj}, {h0}",
      "adds {l0}, {l0}, {h2}",
      "cinc {l1}, {l1}, cs",
      "mul {h2}, {tj}, {h1}",
      "adds {l0}, {l0}, {h2}",
      "cinc {l1}, {l1}, cs",
      "mul {h3}, {l2}, {h0}",
      "adds {l0}, {l0}, {h3}",
      "cinc {l1}, {l1}, cs",
      "ldr {h2}, [{prm}, #16]",
      "mul {h3}, {aj}, {h2}",
      "adds {l0}, {l0}, {h3}",
      "cinc {l1}, {l1}, cs",
      "str {l0}, [{prm}, #64]",
      "umulh {h3}, {aj}, {h2}",
      "add {l1}, {l1}, {h3}",
      "madd {l1}, {tj}, {h2}, {l1}",
      "umulh {h3}, {tj}, {h1}",
      "add {l1}, {l1}, {h3}",
      "madd {l1}, {l2}, {h1}, {l1}",
      "umulh {h3}, {l2}, {h0}",
      "add {l1}, {l1}, {h3}",
      "madd {l1}, {l3}, {h0}, {l1}",
      "ldr {h2}, [{prm}, #24]",
      "madd {l1}, {aj}, {h2}, {l1}",
      "str {l1}, [{prm}, #72]",
      "2:",
      "ldp {b0}, {b1}, [{prm}, #48]",
      "ldp {b2}, {b3}, [{prm}, #64]",
      "mov {t}, {tp}",
      "mov {a}, {np}",
      // The window p0..p3 holds the pending sum for positions j..j + 3.
      "mov {p0}, xzr",
      "mov {p1}, xzr",
      "mov {p2}, xzr",
      "mov {p3}, xzr",
      "mov {cnt}, #8",
      "3:",
      "ldr {aj}, [{a}], #8",
      "ldr {tj}, [{t}]",
      "mul {l0}, {aj}, {b0}",
      "mul {l1}, {aj}, {b1}",
      "mul {l2}, {aj}, {b2}",
      "mul {l3}, {aj}, {b3}",
      "umulh {h0}, {aj}, {b0}",
      "umulh {h1}, {aj}, {b1}",
      "umulh {h2}, {aj}, {b2}",
      "umulh {h3}, {aj}, {b3}",
      "adds {l0}, {l0}, {tj}",
      "adcs {l1}, {l1}, {h0}",
      "adcs {l2}, {l2}, {h1}",
      "adcs {l3}, {l3}, {h2}",
      "adc {h3}, {h3}, xzr",
      "adds {l0}, {l0}, {p0}",
      "str {l0}, [{t}], #8",
      "adcs {p0}, {l1}, {p1}",
      "adcs {p1}, {l2}, {p2}",
      "adcs {p2}, {l3}, {p3}",
      "adc {p3}, {h3}, xzr",
      "subs {cnt}, {cnt}, #1",
      "b.ne 3b",
      // m = (w·k) mod 2^256 for the four limbs w at tp + 32, stored for the strip that starts there.
      "ldp {aj}, {tj}, [{tp}, #32]",
      "ldp {l2}, {l3}, [{tp}, #48]",
      "ldp {h0}, {h1}, [{prm}]",
      "mul {l0}, {aj}, {h0}",
      "umulh {l1}, {aj}, {h0}",
      "str {l0}, [{prm}, #48]",
      "mul {h2}, {aj}, {h1}",
      "adds {l1}, {l1}, {h2}",
      "cset {h3}, cs",
      "mul {h2}, {tj}, {h0}",
      "adds {l1}, {l1}, {h2}",
      "cinc {h3}, {h3}, cs",
      "str {l1}, [{prm}, #56]",
      "umulh {l0}, {aj}, {h1}",
      "adds {l0}, {l0}, {h3}",
      "cset {l1}, cs",
      "umulh {h2}, {tj}, {h0}",
      "adds {l0}, {l0}, {h2}",
      "cinc {l1}, {l1}, cs",
      "mul {h2}, {tj}, {h1}",
      "adds {l0}, {l0}, {h2}",
      "cinc {l1}, {l1}, cs",
      "mul {h3}, {l2}, {h0}",
      "adds {l0}, {l0}, {h3}",
      "cinc {l1}, {l1}, cs",
      "ldr {h2}, [{prm}, #16]",
      "mul {h3}, {aj}, {h2}",
      "adds {l0}, {l0}, {h3}",
      "cinc {l1}, {l1}, cs",
      "str {l0}, [{prm}, #64]",
      "umulh {h3}, {aj}, {h2}",
      "add {l1}, {l1}, {h3}",
      "madd {l1}, {tj}, {h2}, {l1}",
      "umulh {h3}, {tj}, {h1}",
      "add {l1}, {l1}, {h3}",
      "madd {l1}, {l2}, {h1}, {l1}",
      "umulh {h3}, {l2}, {h0}",
      "add {l1}, {l1}, {h3}",
      "madd {l1}, {l3}, {h0}, {l1}",
      "ldr {h2}, [{prm}, #24]",
      "madd {l1}, {aj}, {h2}, {l1}",
      "str {l1}, [{prm}, #72]",
      "ldr {cnt}, [{prm}, #32]",
      "subs {cnt}, {cnt}, #8",
      "b.eq 7f",
      "6:",
      "ldr {aj}, [{a}], #8",
      "ldr {tj}, [{t}]",
      "mul {l0}, {aj}, {b0}",
      "mul {l1}, {aj}, {b1}",
      "mul {l2}, {aj}, {b2}",
      "mul {l3}, {aj}, {b3}",
      "umulh {h0}, {aj}, {b0}",
      "umulh {h1}, {aj}, {b1}",
      "umulh {h2}, {aj}, {b2}",
      "umulh {h3}, {aj}, {b3}",
      "adds {l0}, {l0}, {tj}",
      "adcs {l1}, {l1}, {h0}",
      "adcs {l2}, {l2}, {h1}",
      "adcs {l3}, {l3}, {h2}",
      "adc {h3}, {h3}, xzr",
      "adds {l0}, {l0}, {p0}",
      "str {l0}, [{t}], #8",
      "adcs {p0}, {l1}, {p1}",
      "adcs {p1}, {l2}, {p2}",
      "adcs {p2}, {l3}, {p3}",
      "adc {p3}, {h3}, xzr",
      "subs {cnt}, {cnt}, #1",
      "b.ne 6b",
      "7:",
      // t[n..n + 4] += window + carry; the carry out of position n + 3 replaces it.
      "ldp {l0}, {l1}, [{t}]",
      "ldp {l2}, {l3}, [{t}, #16]",
      "adds {l0}, {l0}, {p0}",
      "adcs {l1}, {l1}, {p1}",
      "adcs {l2}, {l2}, {p2}",
      "adcs {l3}, {l3}, {p3}",
      "cset {h0}, cs",
      "adds {l0}, {l0}, {carry}",
      "adcs {l1}, {l1}, xzr",
      "adcs {l2}, {l2}, xzr",
      "adcs {l3}, {l3}, xzr",
      "cinc {carry}, {h0}, cs",
      "stp {l0}, {l1}, [{t}]",
      "stp {l2}, {l3}, [{t}, #16]",
      "add {tp}, {tp}, #32",
      "subs {outer}, {outer}, #1",
      "b.ne 2b",
      // out = t[n..2n] + carry·2^(64n) minus n when that does not borrow; tp is t + n.
      "ldr {cnt}, [{prm}, #32]",
      "lsr {cnt}, {cnt}, #1",
      "mov {t}, {tp}",
      "mov {a}, {np}",
      "ldr {aj}, [{prm}, #40]",
      "cmp xzr, xzr",
      "4:",
      "ldp {l0}, {l1}, [{t}], #16",
      "ldp {h0}, {h1}, [{a}], #16",
      "sbcs {l0}, {l0}, {h0}",
      "sbcs {l1}, {l1}, {h1}",
      "stp {l0}, {l1}, [{aj}], #16",
      "sub {cnt}, {cnt}, #1",
      "cbnz {cnt}, 4b",
      // Keep the difference when it did not borrow or the carry is set.
      "cset {h2}, cs",
      "orr {h2}, {h2}, {carry}",
      "ldr {cnt}, [{prm}, #32]",
      "lsr {cnt}, {cnt}, #1",
      "mov {t}, {tp}",
      "ldr {aj}, [{prm}, #40]",
      "cmp {h2}, #0",
      "5:",
      "ldp {l0}, {l1}, [{t}], #16",
      "ldp {h0}, {h1}, [{aj}]",
      "csel {l0}, {h0}, {l0}, ne",
      "csel {l1}, {h1}, {l1}, ne",
      "stp {l0}, {l1}, [{aj}], #16",
      "sub {cnt}, {cnt}, #1",
      "cbnz {cnt}, 5b",
      tp = inout(reg) t.as_mut_ptr() => _,
      outer = inout(reg) n / 4 => _,
      prm = in(reg) params.as_mut_ptr(),
      np = in(reg) modulus.as_ptr(),
      carry = out(reg) _, t = out(reg) _, a = out(reg) _, cnt = out(reg) _,
      b0 = out(reg) _, b1 = out(reg) _, b2 = out(reg) _, b3 = out(reg) _,
      aj = out(reg) _, tj = out(reg) _,
      l0 = out(reg) _, l1 = out(reg) _, l2 = out(reg) _, l3 = out(reg) _,
      h0 = out(reg) _, h1 = out(reg) _, h2 = out(reg) _, h3 = out(reg) _,
      p0 = out(reg) _, p1 = out(reg) _, p2 = out(reg) _, p3 = out(reg) _,
      options(nostack),
    );
  }
}

/// `t[0..2n] = a·b`.
fn product(t: &mut [u64], a: &[u64], b: &[u64]) {
  let n = a.len();
  assert!(supports(n) && b.len() == n && t.len() >= n.strict_mul(2));
  let t = &mut t[..n.strict_mul(2)];
  t.fill(0);
  let params = [0u64, 0, 0, 0, n as u64];
  // SAFETY: Every strip reads `b[s..s + 4]` and `a[0..n]` and reads and writes `t[s..s + n + 4]`
  // for `s` below `n` in steps of four, inside `t[0..2n]`. Each partial product fits its limbs,
  // so the strip carries stay zero. The block uses no stack, and `n` is a nonzero multiple of
  // four.
  unsafe {
    asm!(
      // Product strips: each adds a·b[s..s + 4] at limb s; the carries stay zero.
      "mov {carry}, xzr",
      "2:",
      "ldp {b0}, {b1}, [{bp}], #16",
      "ldp {b2}, {b3}, [{bp}], #16",
      "mov {t}, {tp}",
      "mov {a}, {np}",
      "ldr {cnt}, [{prm}, #32]",
      // The window p0..p3 holds the pending sum for positions j..j + 3.
      "mov {p0}, xzr",
      "mov {p1}, xzr",
      "mov {p2}, xzr",
      "mov {p3}, xzr",
      "3:",
      "ldr {aj}, [{a}], #8",
      "ldr {tj}, [{t}]",
      "mul {l0}, {aj}, {b0}",
      "mul {l1}, {aj}, {b1}",
      "mul {l2}, {aj}, {b2}",
      "mul {l3}, {aj}, {b3}",
      "umulh {h0}, {aj}, {b0}",
      "umulh {h1}, {aj}, {b1}",
      "umulh {h2}, {aj}, {b2}",
      "umulh {h3}, {aj}, {b3}",
      "adds {l0}, {l0}, {tj}",
      "adcs {l1}, {l1}, {h0}",
      "adcs {l2}, {l2}, {h1}",
      "adcs {l3}, {l3}, {h2}",
      "adc {h3}, {h3}, xzr",
      "adds {l0}, {l0}, {p0}",
      "str {l0}, [{t}], #8",
      "adcs {p0}, {l1}, {p1}",
      "adcs {p1}, {l2}, {p2}",
      "adcs {p2}, {l3}, {p3}",
      "adc {p3}, {h3}, xzr",
      "subs {cnt}, {cnt}, #1",
      "b.ne 3b",
      // t[n..n + 4] += window + carry; the carry out of position n + 3 replaces it.
      "ldp {l0}, {l1}, [{t}]",
      "ldp {l2}, {l3}, [{t}, #16]",
      "adds {l0}, {l0}, {p0}",
      "adcs {l1}, {l1}, {p1}",
      "adcs {l2}, {l2}, {p2}",
      "adcs {l3}, {l3}, {p3}",
      "cset {h0}, cs",
      "adds {l0}, {l0}, {carry}",
      "adcs {l1}, {l1}, xzr",
      "adcs {l2}, {l2}, xzr",
      "adcs {l3}, {l3}, xzr",
      "cinc {carry}, {h0}, cs",
      "stp {l0}, {l1}, [{t}]",
      "stp {l2}, {l3}, [{t}, #16]",
      "add {tp}, {tp}, #32",
      "subs {outer}, {outer}, #1",
      "b.ne 2b",
      tp = inout(reg) t.as_mut_ptr() => _,
      outer = inout(reg) n / 4 => _,
      prm = in(reg) params.as_ptr(),
      np = in(reg) a.as_ptr(),
      bp = inout(reg) b.as_ptr() => _,
      carry = out(reg) _, t = out(reg) _, a = out(reg) _, cnt = out(reg) _,
      b0 = out(reg) _, b1 = out(reg) _, b2 = out(reg) _, b3 = out(reg) _,
      aj = out(reg) _, tj = out(reg) _,
      l0 = out(reg) _, l1 = out(reg) _, l2 = out(reg) _, l3 = out(reg) _,
      h0 = out(reg) _, h1 = out(reg) _, h2 = out(reg) _, h3 = out(reg) _,
      p0 = out(reg) _, p1 = out(reg) _, p2 = out(reg) _, p3 = out(reg) _,
      options(nostack),
    );
  }
}

/// `out = a·b·2^(-64·n) mod modulus` for `a, b < modulus`; `k` is `-modulus⁻¹ mod 2^256`.
pub(super) fn mul(out: &mut [u64], a: &[u64], b: &[u64], modulus: &[u64], k: &[u64; 4], t: &mut [u64]) {
  let n = modulus.len();
  assert!(supports(n) && a.len() == n && b.len() == n && out.len() == n);
  let t = &mut t[..scratch_words(n)];
  product(t, a, b);
  reduce(out, t, modulus, k);
}

/// `left = left·right·2^(-64·n) mod modulus` for `left, right < modulus`; `k` is
/// `-modulus⁻¹ mod 2^256`.
pub(super) fn mul_in_place_left(left: &mut [u64], right: &[u64], modulus: &[u64], k: &[u64; 4], t: &mut [u64]) {
  let n = modulus.len();
  assert!(supports(n) && left.len() == n && right.len() == n);
  let t = &mut t[..scratch_words(n)];
  product(t, left, right);
  // The product no longer reads `left`, so the reduction writes the result there.
  reduce(left, t, modulus, k);
}

/// `value = value²·2^(-64·n) mod modulus` for `value < modulus`; `k` is `-modulus⁻¹ mod 2^256`.
///
/// The cross products `a_i·a_j` for `i < j` come first: each 4×4 diagonal block's six products
/// fill limbs `2s..2s + 8` of `t`, which also clears it, and each block's rows times the limbs
/// after the block are added by the strip kernel. Doubling and the diagonal squares follow, then
/// the reduction.
pub(super) fn square_in_place(value: &mut [u64], modulus: &[u64], k: &[u64; 4], t: &mut [u64]) {
  let n = modulus.len();
  assert!(supports(n) && value.len() == n);
  let t = &mut t[..scratch_words(n)];
  // SAFETY: The diagonal blocks read `value[0..n]` and write `t[0..2n]`, eight limbs per block.
  // Rectangle strip s reads `value[s..n]` and reads and writes `t[2s + 4..s + n + 4]`, inside
  // `t[0..2n]`, and the final carry touches `t[2n - 4..2n]`. The block uses no stack, and `n` is
  // a multiple of four of at least eight, so every counter starts nonzero.
  unsafe {
    asm!(
      // Diagonal blocks: limb 2s is zero and limbs 2s + 1..2s + 8 hold the block's cross products.
      "mov {t}, {tw}",
      "mov {a}, {ab}",
      "lsr {cnt}, {len}, #2",
      "2:",
      "ldp {b0}, {b1}, [{a}], #16",
      "ldp {b2}, {b3}, [{a}], #16",
      "mul {l1}, {b0}, {b1}",
      "umulh {aj}, {b0}, {b1}",
      "mul {l2}, {b0}, {b2}",
      "umulh {tj}, {b0}, {b2}",
      "mul {l3}, {b0}, {b3}",
      "umulh {h0}, {b0}, {b3}",
      "adds {l2}, {l2}, {aj}",
      "adcs {l3}, {l3}, {tj}",
      "adc {h0}, {h0}, xzr",
      "mul {p0}, {b1}, {b2}",
      "umulh {aj}, {b1}, {b2}",
      "mul {p1}, {b1}, {b3}",
      "umulh {p2}, {b1}, {b3}",
      "adds {p1}, {p1}, {aj}",
      "adc {p2}, {p2}, xzr",
      "mul {h1}, {b2}, {b3}",
      "umulh {h2}, {b2}, {b3}",
      "adds {l3}, {l3}, {p0}",
      "adcs {h0}, {h0}, {p1}",
      "adcs {p2}, {p2}, {h1}",
      "adcs {h2}, {h2}, xzr",
      "cset {h3}, cs",
      "stp xzr, {l1}, [{t}], #16",
      "stp {l2}, {l3}, [{t}], #16",
      "stp {h0}, {p2}, [{t}], #16",
      "stp {h2}, {h3}, [{t}], #16",
      "subs {cnt}, {cnt}, #1",
      "b.ne 2b",
      // Rectangles: block s's rows times a[s + 4..n] at limb 2s + 4; each carry joins the next window.
      "mov {carry}, xzr",
      "add {tw}, {tw}, #32",
      "add {ab}, {ab}, #32",
      "sub {len}, {len}, #4",
      "8:",
      "ldp {b0}, {b1}, [{ab}, #-32]",
      "ldp {b2}, {b3}, [{ab}, #-16]",
      "mov {t}, {tw}",
      "mov {a}, {ab}",
      "mov {cnt}, {len}",
      // The window p0..p3 holds the pending sum for positions j..j + 3.
      "mov {p0}, xzr",
      "mov {p1}, xzr",
      "mov {p2}, xzr",
      "mov {p3}, xzr",
      "3:",
      "ldr {aj}, [{a}], #8",
      "ldr {tj}, [{t}]",
      "mul {l0}, {aj}, {b0}",
      "mul {l1}, {aj}, {b1}",
      "mul {l2}, {aj}, {b2}",
      "mul {l3}, {aj}, {b3}",
      "umulh {h0}, {aj}, {b0}",
      "umulh {h1}, {aj}, {b1}",
      "umulh {h2}, {aj}, {b2}",
      "umulh {h3}, {aj}, {b3}",
      "adds {l0}, {l0}, {tj}",
      "adcs {l1}, {l1}, {h0}",
      "adcs {l2}, {l2}, {h1}",
      "adcs {l3}, {l3}, {h2}",
      "adc {h3}, {h3}, xzr",
      "adds {l0}, {l0}, {p0}",
      "str {l0}, [{t}], #8",
      "adcs {p0}, {l1}, {p1}",
      "adcs {p1}, {l2}, {p2}",
      "adcs {p2}, {l3}, {p3}",
      "adc {p3}, {h3}, xzr",
      "subs {cnt}, {cnt}, #1",
      "b.ne 3b",
      // t[n..n + 4] += window + carry; the carry out of position n + 3 replaces it.
      "ldp {l0}, {l1}, [{t}]",
      "ldp {l2}, {l3}, [{t}, #16]",
      "adds {l0}, {l0}, {p0}",
      "adcs {l1}, {l1}, {p1}",
      "adcs {l2}, {l2}, {p2}",
      "adcs {l3}, {l3}, {p3}",
      "cset {h0}, cs",
      "adds {l0}, {l0}, {carry}",
      "adcs {l1}, {l1}, xzr",
      "adcs {l2}, {l2}, xzr",
      "adcs {l3}, {l3}, xzr",
      "cinc {carry}, {h0}, cs",
      "stp {l0}, {l1}, [{t}]",
      "stp {l2}, {l3}, [{t}, #16]",
      "add {tw}, {tw}, #64",
      "add {ab}, {ab}, #32",
      "subs {len}, {len}, #4",
      "b.ne 8b",
      // The last carry belongs at limb 2n - 4, where tw now points; the cross sum fits 2n limbs.
      "ldp {l0}, {l1}, [{tw}]",
      "ldp {l2}, {l3}, [{tw}, #16]",
      "adds {l0}, {l0}, {carry}",
      "adcs {l1}, {l1}, xzr",
      "adcs {l2}, {l2}, xzr",
      "adc {l3}, {l3}, xzr",
      "stp {l0}, {l1}, [{tw}]",
      "stp {l2}, {l3}, [{tw}, #16]",
      tw = inout(reg) t.as_mut_ptr() => _,
      ab = inout(reg) value.as_ptr() => _,
      len = inout(reg) n => _,
      carry = out(reg) _, t = out(reg) _, a = out(reg) _, cnt = out(reg) _,
      b0 = out(reg) _, b1 = out(reg) _, b2 = out(reg) _, b3 = out(reg) _,
      aj = out(reg) _, tj = out(reg) _,
      l0 = out(reg) _, l1 = out(reg) _, l2 = out(reg) _, l3 = out(reg) _,
      h0 = out(reg) _, h1 = out(reg) _, h2 = out(reg) _, h3 = out(reg) _,
      p0 = out(reg) _, p1 = out(reg) _, p2 = out(reg) _, p3 = out(reg) _,
      options(nostack),
    );
  }
  // SAFETY: `t` holds `2n` limbs and `value` holds `n`, a nonzero multiple of four.
  unsafe { double_add_squares(t.as_mut_ptr(), value.as_ptr(), n) };
  // The square no longer reads `value`, so the reduction writes the result there.
  reduce(value, t, modulus, k);
}
