//! AVX-512 IFMA modular exponentiation for RSA CRT primes.
//!
//! Residues use radix 2^52: `digits` 52-bit digits in the low bits of `V` vectors of eight
//! `u64` lanes (`8·V >= digits`; unused high lanes stay zero). One almost-Montgomery
//! multiplication (AMM) computes `a·b·R⁻¹ mod m` for `R = 2^(52·digits)`; inputs below `2·m`
//! give an output below `2·m`, because `4·m < R`. The word-serial schedule follows Gueron and
//! Krasnov, "Accelerating big integer arithmetic using Intel IFMA extensions" (ARITH 2016): for
//! each digit `b_i`, add the low 52-bit halves of `a·b_i` and `m·q`, drop one digit, then add
//! the high halves.
//!
//! Digit 0 is tracked exactly in a scalar register, so each step's reduction factor does not wait
//! for a vector multiply. Each digit step is still a serial chain, so the CRT pair runs the two
//! primes' multiplications digit by digit in one loop: their chains are independent and overlap.
//!
//! Loop counts, addresses and lane operations depend only on the public digit count. The secret
//! exponent selects table entries through lane masks, and carries are normalized with lane
//! masks. `rsa.rs` owns validation, dispatch and the portable fallback, which computes the same
//! values.

use core::arch::x86_64::*;

use crate::platform::{caps, caps::x86};

const MASK52: u64 = u64::MAX >> 12;
const WINDOW_ENTRIES: usize = 16;

/// Digit and vector counts for a prime of `limbs` 64-bit limbs, or `None` without a kernel.
///
/// `52·digits >= 64·limbs + 2`, so `4·m < R`.
#[inline]
pub(super) fn shape(limbs: usize) -> Option<(usize, usize)> {
  if !caps().has(x86::AVX512F) || !caps().has(x86::AVX512IFMA) {
    return None;
  }
  match limbs {
    16 => Some((20, 3)),
    24 => Some((30, 4)),
    32 => Some((40, 5)),
    _ => None,
  }
}

/// `u64` workspace words for one prime: the window table, multiplier, modulus and one.
fn prime_workspace_words(vectors: usize) -> usize {
  vectors.strict_mul(8).strict_mul(WINDOW_ENTRIES.strict_add(3))
}

/// `u64` workspace words for the widest kernel prime of at most `max_limbs` limbs, or zero.
///
/// Private scratch is sized by the public modulus width, about twice each CRT prime's.
#[inline]
pub(super) fn max_workspace_words(max_limbs: usize) -> usize {
  [32usize, 24, 16]
    .into_iter()
    .find(|&limbs| limbs <= max_limbs && shape(limbs).is_some())
    .map_or(0, workspace_words)
}

/// `u64` workspace words the CRT pair needs for `limbs`-limb primes; one prime uses half.
#[inline]
pub(super) fn workspace_words(limbs: usize) -> usize {
  let vectors = match limbs {
    16 => 3usize,
    24 => 4,
    32 => 5,
    _ => return 0,
  };
  prime_workspace_words(vectors).strict_mul(2)
}

/// One prime's exponentiation operands.
pub(super) struct Operands<'a> {
  /// Fully reduced result, little-endian 64-bit limbs.
  pub(super) out: &'a mut [u64],
  /// Base below `m`.
  pub(super) base: &'a [u64],
  /// Big-endian exponent bytes, most significant first.
  pub(super) exponent: &'a [u8],
  /// Odd prime modulus.
  pub(super) m: &'a [u64],
  /// `R² mod m` for `R = 2^(52·digits)`, fully reduced.
  pub(super) r2: &'a [u64],
  /// `-m⁻¹ mod 2^52`.
  pub(super) k0: u64,
}

/// Compute `base^exponent mod m` for one prime.
///
/// # Safety
/// The caller must establish AVX-512F and AVX-512 IFMA, pass `shape(m.len())`, and provide at
/// least `workspace_words(m.len()) / 2` workspace words.
pub(super) unsafe fn exponentiate(operands: Operands<'_>, (digits, vectors): (usize, usize), workspace: &mut [u64]) {
  // SAFETY: The caller establishes AVX-512F and AVX-512 IFMA, which every arm requires.
  unsafe {
    match vectors {
      3 => exponentiate_one::<3>(operands, digits, workspace),
      4 => exponentiate_one::<4>(operands, digits, workspace),
      _ => exponentiate_one::<5>(operands, digits, workspace),
    }
  }
}

/// Compute both CRT exponentiations, `base^exponent mod m` for each prime, in one interleaved
/// schedule. Both primes have the same limb count and both exponents the same length.
///
/// # Safety
/// The caller must establish AVX-512F and AVX-512 IFMA, pass `shape(m.len())` for the shared limb
/// count, and provide at least `workspace_words(m.len())` workspace words.
pub(super) unsafe fn exponentiate_pair(
  operands: [Operands<'_>; 2],
  (digits, vectors): (usize, usize),
  workspace: &mut [u64],
) {
  // SAFETY: The caller establishes AVX-512F and AVX-512 IFMA, which every arm requires.
  unsafe {
    match vectors {
      3 => exponentiate_two::<3>(operands, digits, workspace),
      4 => exponentiate_two::<4>(operands, digits, workspace),
      _ => exponentiate_two::<5>(operands, digits, workspace),
    }
  }
}

/// Split little-endian 64-bit limbs into normalized 52-bit digits, zero-filling `out`.
fn to_digits(limbs: &[u64], out: &mut [u64]) {
  out.fill(0);
  let mut accumulator = 0u128;
  let mut bits = 0u32;
  let mut digit = 0usize;
  for &limb in limbs {
    accumulator |= u128::from(limb) << bits;
    bits = bits.strict_add(64);
    while bits >= 52 {
      out[digit] = low_u64(accumulator) & MASK52;
      digit = digit.strict_add(1);
      accumulator >>= 52;
      bits = bits.strict_sub(52);
    }
  }
  if digit < out.len() {
    out[digit] = low_u64(accumulator);
  }
}

/// Join normalized 52-bit digits into little-endian 64-bit limbs.
fn from_digits(digits: &[u64], out: &mut [u64]) {
  let mut accumulator = 0u128;
  let mut bits = 0u32;
  let mut digit = 0usize;
  for limb in out.iter_mut() {
    while bits < 64 {
      let value = digits.get(digit).copied().unwrap_or(0);
      accumulator |= u128::from(value) << bits;
      bits = bits.strict_add(52);
      digit = digit.strict_add(1);
    }
    *limb = low_u64(accumulator);
    accumulator >>= 64;
    bits = bits.strict_sub(64);
  }
}

#[inline(always)]
fn low_u64(value: u128) -> u64 {
  let bytes = value.to_le_bytes();
  u64::from_le_bytes([
    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
  ])
}

/// Load the first `8·V` words of `digits` into vectors.
///
/// # Safety
/// Calling outside an AVX-512F-enabled function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn load<const V: usize>(digits: &[u64]) -> [__m512i; V] {
  let chunks = digits.as_chunks::<8>().0;
  let mut out = [_mm512_setzero_si512(); V];
  for (lanes, chunk) in out.iter_mut().zip(chunks) {
    // SAFETY: `chunk` is eight initialized words; the unaligned load reads exactly them.
    *lanes = unsafe { _mm512_loadu_si512(chunk.as_ptr().cast()) };
  }
  debug_assert!(chunks.len() >= V, "vector load needs 8·V words");
  out
}

/// Store vectors into the first `8·V` words of `digits`.
///
/// # Safety
/// Calling outside an AVX-512F-enabled function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn store<const V: usize>(value: &[__m512i; V], digits: &mut [u64]) {
  let chunks = digits.as_chunks_mut::<8>().0;
  debug_assert!(chunks.len() >= V, "vector store needs 8·V words");
  for (chunk, &lanes) in chunks.iter_mut().zip(value) {
    // SAFETY: `chunk` is eight uniquely borrowed words; the unaligned store writes exactly them.
    unsafe { _mm512_storeu_si512(chunk.as_mut_ptr().cast(), lanes) };
  }
}

/// One AMM digit step: add `a·b_digit` and `m·q`, drop the zero digit, and carry into the next.
///
/// `low` holds digit 0 exactly in a scalar register, so `q` comes from scalar products and does
/// not wait for the vector multiply; vector lane 0 is a shadow that the shift discards. After
/// the shift, `low` takes the carry out of digit 0 plus the new lane 0 before its high halves,
/// which `low`'s full scalar products already include. Each lane gains at most four 52-bit
/// halves per digit, so after at most 40 digits it stays below `(4·40 + 1)·2^52 < 2^60`, and
/// `low` stays below `2^60 + 2^54`.
///
/// # Safety
/// Calling outside an AVX-512F and AVX-512 IFMA function requires both features.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn amm_step<const V: usize>(
  acc: &mut [__m512i; V],
  low: &mut u64,
  a: &[__m512i; V],
  b_digit: u64,
  modulus: &[__m512i; V],
  (a0, m0, k0): (u64, u64, u64),
) {
  let zero = _mm512_setzero_si512();
  let b_lane = _mm512_set1_epi64(b_digit.cast_signed());
  // Digit 0 stays below `2^61 + 2^105`, so these operations cannot wrap; the non-panicking forms
  // keep secret-fed overflow branches out of release code, as in the ECDSA limb arithmetic.
  let digit = u128::from(*low).wrapping_add(u128::from(a0).wrapping_mul(u128::from(b_digit)));
  let q = low_u64(digit).wrapping_mul(k0) & MASK52;
  let digit = digit.wrapping_add(u128::from(m0).wrapping_mul(u128::from(q)));
  debug_assert_eq!(low_u64(digit) & MASK52, 0, "q clears digit 0");
  let carry = low_u64(digit >> 52);
  let q_lane = _mm512_set1_epi64(q.cast_signed());
  for ((lanes, &a_lanes), &m_lanes) in acc.iter_mut().zip(a).zip(modulus) {
    *lanes = _mm512_madd52lo_epu64(*lanes, a_lanes, b_lane);
    *lanes = _mm512_madd52lo_epu64(*lanes, m_lanes, q_lane);
  }
  // Drop digit 0: lane j takes lane j + 1, across vector boundaries.
  let mut next = zero;
  for lanes in acc.iter_mut().rev() {
    let shifted = _mm512_alignr_epi64::<1>(next, *lanes);
    next = *lanes;
    *lanes = shifted;
  }
  *low = carry.wrapping_add(lane0(acc[0]));
  // The high halves weigh one digit more, which is now the same lane.
  for ((lanes, &a_lanes), &m_lanes) in acc.iter_mut().zip(a).zip(modulus) {
    *lanes = _mm512_madd52hi_epu64(*lanes, a_lanes, b_lane);
    *lanes = _mm512_madd52hi_epu64(*lanes, m_lanes, q_lane);
  }
}

/// Lane 0 of `lanes`.
///
/// # Safety
/// Calling outside an AVX-512F-enabled function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn lane0(lanes: __m512i) -> u64 {
  _mm_cvtsi128_si64(_mm512_castsi512_si128(lanes)).cast_unsigned()
}

/// Replace the shadow lane 0 with the exact digit 0 and normalize.
///
/// # Safety
/// Calling outside an AVX-512F-enabled function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn finish_amm<const V: usize>(mut acc: [__m512i; V], low: u64) -> [__m512i; V] {
  acc[0] = _mm512_mask_set1_epi64(acc[0], 1, low.cast_signed());
  normalize(&acc)
}

/// Propagate carries so every lane is a 52-bit digit. Lanes are below 2^61 and the value is
/// below `2^(52·8·V)`, so no carry leaves the top lane.
///
/// One shift moves every lane's carry (below 2^9) one lane up, leaving sums below `2^52 + 2^9`.
/// Such a lane then carries out at most one: when it is at least 2^52 (generate), or when it is
/// exactly `2^52 - 1` and receives a carry (propagate). Treating the generate and propagate lane
/// masks as integers, `((generate << 1) + propagate) ^ propagate` marks every lane that receives
/// a carry, rippling through propagating runs.
///
/// # Safety
/// Calling outside an AVX-512F-enabled function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn normalize<const V: usize>(value: &[__m512i; V]) -> [__m512i; V] {
  let mask = _mm512_set1_epi64(MASK52.cast_signed());
  let mut sum = [_mm512_setzero_si512(); V];
  let mut below = _mm512_setzero_si512();
  let mut generate = 0u64;
  let mut propagate = 0u64;
  // The vector count is a compile-time kernel shape. Indexing that fixed range exposes it to
  // LLVM; the zipped iterator otherwise retains its state machine and spills on every AMM.
  for index in 0..V {
    let lanes = value[index];
    let carries = _mm512_srli_epi64::<52>(lanes);
    // Lane j takes lane j - 1's carry; lane 0 takes the top carry of the vector below.
    sum[index] = _mm512_add_epi64(_mm512_and_si512(lanes, mask), _mm512_alignr_epi64::<7>(carries, below));
    below = carries;
    let offset = index.strict_mul(8);
    generate |= u64::from(_mm512_cmpgt_epu64_mask(sum[index], mask)) << offset;
    propagate |= u64::from(_mm512_cmpeq_epu64_mask(sum[index], mask)) << offset;
  }
  let carry_in = (generate << 1).wrapping_add(propagate) ^ propagate;
  let carry_lanes = carry_in.to_le_bytes();
  let one = _mm512_set1_epi64(1);
  for index in 0..V {
    sum[index] = _mm512_and_si512(
      _mm512_mask_add_epi64(sum[index], carry_lanes[index], sum[index], one),
      mask,
    );
  }
  sum
}

/// One AMM: `a·b·R⁻¹ mod m`, below `2·m`, with normalized digits.
///
/// `a` and `b` hold normalized digits of values below `2·m`; `b_digits` is `b` by digit.
///
/// # Safety
/// Calling outside an AVX-512F and AVX-512 IFMA function requires both features.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn amm<const V: usize>(
  a: &[__m512i; V],
  b_digits: &[u64],
  modulus: &[__m512i; V],
  (m0, k0): (u64, u64),
  digits: usize,
) -> [__m512i; V] {
  let mut acc = [_mm512_setzero_si512(); V];
  let mut low = 0u64;
  let constants = (lane0(a[0]), m0, k0);
  for &b_digit in &b_digits[..digits] {
    amm_step(&mut acc, &mut low, a, b_digit, modulus, constants);
  }
  finish_amm(acc, low)
}

/// Two independent AMMs, one per prime, interleaved digit by digit.
///
/// # Safety
/// Calling outside an AVX-512F and AVX-512 IFMA function requires both features.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn amm_pair<const V: usize>(
  a: [&[__m512i; V]; 2],
  b_digits: [&[u64]; 2],
  modulus: [&[__m512i; V]; 2],
  constants: [(u64, u64); 2],
  digits: usize,
) -> [[__m512i; V]; 2] {
  let zero = _mm512_setzero_si512();
  let mut first = [zero; V];
  let mut second = [zero; V];
  let (mut first_low, mut second_low) = (0u64, 0u64);
  let first_constants = (lane0(a[0][0]), constants[0].0, constants[0].1);
  let second_constants = (lane0(a[1][0]), constants[1].0, constants[1].1);
  for (&b_first, &b_second) in b_digits[0][..digits].iter().zip(&b_digits[1][..digits]) {
    amm_step(&mut first, &mut first_low, a[0], b_first, modulus[0], first_constants);
    amm_step(
      &mut second,
      &mut second_low,
      a[1],
      b_second,
      modulus[1],
      second_constants,
    );
  }
  [finish_amm(first, first_low), finish_amm(second, second_low)]
}

/// One prime's workspace regions and converted constants.
struct Prime<'a, const V: usize> {
  table: &'a mut [u64],
  multiplier: &'a mut [u64],
  one: &'a mut [u64],
  modulus: [__m512i; V],
  constants: (u64, u64),
}

impl<'a, const V: usize> Prime<'a, V> {
  /// Split one prime's workspace and convert its modulus.
  ///
  /// # Safety
  /// Calling outside an AVX-512F-enabled function requires AVX-512F support.
  #[inline]
  #[target_feature(enable = "avx512f")]
  fn new(workspace: &'a mut [u64], m: &[u64], k0: u64) -> Self {
    let width = V.strict_mul(8);
    let (table, rest) = workspace.split_at_mut(WINDOW_ENTRIES.strict_mul(width));
    let (multiplier, rest) = rest.split_at_mut(width);
    let (modulus_digits, rest) = rest.split_at_mut(width);
    let one = &mut rest[..width];
    to_digits(m, modulus_digits);
    let modulus = load::<V>(modulus_digits);
    let constants = (modulus_digits[0], k0);
    one.fill(0);
    one[0] = 1;
    Self {
      table,
      multiplier,
      one,
      modulus,
      constants,
    }
  }

  /// Select `table[window]` into the multiplier by reading every entry under lane masks.
  ///
  /// Every entry is loaded without a mask. Otherwise LLVM folds the load into the merge-masked
  /// OR, and a masked memory operand need not access its masked-off lanes, so the secret window
  /// could decide which table lines are read.
  ///
  /// # Safety
  /// Calling outside an AVX-512F-enabled function requires AVX-512F support.
  #[inline]
  #[target_feature(enable = "avx512f")]
  fn select(&mut self, window: u8) {
    let target = _mm512_set1_epi64(i64::from(window));
    let mut selected = [_mm512_setzero_si512(); V];
    for (entry, values) in (0i64..).zip(self.table.chunks_exact(V.strict_mul(8))) {
      let hit = _mm512_cmpeq_epi64_mask(_mm512_set1_epi64(entry), target);
      for (lanes, mut value) in selected.iter_mut().zip(load::<V>(values)) {
        // SAFETY: the empty template emits no instruction and touches no memory, stack, or
        // flags; this function enables AVX-512F, which `zmm_reg` requires. The barrier makes
        // `value` an opaque register, so its table load stays a separate unmasked load.
        unsafe { core::arch::asm!("/* {0} */", inout(zmm_reg) value, options(nomem, nostack, preserves_flags)) };
        *lanes = _mm512_mask_or_epi64(*lanes, hit, *lanes, value);
      }
    }
    store(&selected, self.multiplier);
  }

  /// Leave the Montgomery domain and write the fully reduced result.
  ///
  /// `acc·R⁻¹` is at most `m`, so one masked subtraction of `m` finishes the reduction.
  ///
  /// # Safety
  /// Calling outside an AVX-512F and AVX-512 IFMA function requires both features.
  #[inline]
  #[target_feature(enable = "avx512f,avx512ifma")]
  fn finish(&mut self, acc: &[__m512i; V], digits: usize, out: &mut [u64]) {
    let result = amm(acc, self.one, &self.modulus, self.constants, digits);
    let modulus = self.modulus;
    store(&conditional_subtract(&result, &modulus), self.multiplier);
    from_digits(&self.multiplier[..digits], out);
    self.multiplier.fill(0);
  }
}

/// Fill `table[k] = base^k·R mod m` (almost reduced) from `R² mod m`, and return `R mod m`.
///
/// # Safety
/// Calling outside an AVX-512F and AVX-512 IFMA function requires both features.
#[inline]
#[target_feature(enable = "avx512f,avx512ifma")]
fn build_table<const V: usize>(prime: &mut Prime<'_, V>, base: &[u64], r2: &[u64], digits: usize) -> [__m512i; V] {
  let width = V.strict_mul(8);
  to_digits(r2, prime.multiplier);
  let r2_lanes = load::<V>(prime.multiplier);
  let one_mont = amm(
    &load::<V>(prime.one),
    prime.multiplier,
    &prime.modulus,
    prime.constants,
    digits,
  );
  to_digits(base, prime.multiplier);
  let base_mont = amm(&r2_lanes, prime.multiplier, &prime.modulus, prime.constants, digits);
  let mut entries = prime.table.chunks_exact_mut(width);
  if let Some(entry) = entries.next() {
    store(&one_mont, entry);
  }
  store(&base_mont, prime.multiplier);
  if let Some(entry) = entries.next() {
    store(&base_mont, entry);
  }
  let mut previous = base_mont;
  for entry in entries {
    previous = amm(&previous, prime.multiplier, &prime.modulus, prime.constants, digits);
    store(&previous, entry);
  }
  one_mont
}

/// [`exponentiate`] for `V` vectors per residue.
///
/// # Safety
/// Calling outside an AVX-512F and AVX-512 IFMA function requires both features.
#[target_feature(enable = "avx512f,avx512ifma")]
fn exponentiate_one<const V: usize>(operands: Operands<'_>, digits: usize, workspace: &mut [u64]) {
  let words = prime_workspace_words(V);
  let mut prime = Prime::<V>::new(&mut workspace[..words], operands.m, operands.k0);
  let mut acc = build_table(&mut prime, operands.base, operands.r2, digits);
  // Fixed four-bit windows over every exponent byte, most significant first.
  for &byte in operands.exponent {
    for window in [byte >> 4, byte & 0x0f] {
      for _ in 0..4 {
        store(&acc, prime.multiplier);
        acc = amm(&acc, prime.multiplier, &prime.modulus, prime.constants, digits);
      }
      prime.select(window);
      acc = amm(&acc, prime.multiplier, &prime.modulus, prime.constants, digits);
    }
  }
  prime.finish(&acc, digits, operands.out);
}

/// [`exponentiate_pair`] for `V` vectors per residue.
///
/// # Safety
/// Calling outside an AVX-512F and AVX-512 IFMA function requires both features.
#[target_feature(enable = "avx512f,avx512ifma")]
fn exponentiate_two<const V: usize>(operands: [Operands<'_>; 2], digits: usize, workspace: &mut [u64]) {
  let words = prime_workspace_words(V);
  let [first_operands, second_operands] = operands;
  let (first_workspace, rest) = workspace.split_at_mut(words);
  let mut first = Prime::<V>::new(first_workspace, first_operands.m, first_operands.k0);
  let mut second = Prime::<V>::new(&mut rest[..words], second_operands.m, second_operands.k0);
  let mut acc = [
    build_table(&mut first, first_operands.base, first_operands.r2, digits),
    build_table(&mut second, second_operands.base, second_operands.r2, digits),
  ];
  // Fixed four-bit windows over every exponent byte, most significant first.
  for (&first_byte, &second_byte) in first_operands.exponent.iter().zip(second_operands.exponent) {
    for shift in [4u32, 0] {
      for _ in 0..4 {
        store(&acc[0], first.multiplier);
        store(&acc[1], second.multiplier);
        acc = amm_pair(
          [&acc[0], &acc[1]],
          [&*first.multiplier, &*second.multiplier],
          [&first.modulus, &second.modulus],
          [first.constants, second.constants],
          digits,
        );
      }
      first.select((first_byte >> shift) & 0x0f);
      second.select((second_byte >> shift) & 0x0f);
      acc = amm_pair(
        [&acc[0], &acc[1]],
        [&*first.multiplier, &*second.multiplier],
        [&first.modulus, &second.modulus],
        [first.constants, second.constants],
        digits,
      );
    }
  }
  first.finish(&acc[0], digits, first_operands.out);
  second.finish(&acc[1], digits, second_operands.out);
}

/// Replace `value` by `value - m` when `value >= m`, selecting the difference under a mask.
///
/// Both operands hold normalized digits. A lane borrows when its digit is below the modulus digit
/// (generate) and passes a borrow on when they are equal (propagate); the same mask arithmetic as
/// [`normalize`] gives every lane's borrow and the borrow out of the top lane, which is set
/// exactly when `value < m`.
///
/// # Safety
/// Calling outside an AVX-512F-enabled function requires AVX-512F support.
#[inline]
#[target_feature(enable = "avx512f")]
fn conditional_subtract<const V: usize>(value: &[__m512i; V], m: &[__m512i; V]) -> [__m512i; V] {
  let mask = _mm512_set1_epi64(MASK52.cast_signed());
  let one = _mm512_set1_epi64(1);
  let mut generate = 0u64;
  let mut propagate = 0u64;
  for (index, (&lanes, &modulus)) in value.iter().zip(m).enumerate() {
    let offset = index.strict_mul(8);
    generate |= u64::from(_mm512_cmplt_epu64_mask(lanes, modulus)) << offset;
    propagate |= u64::from(_mm512_cmpeq_epu64_mask(lanes, modulus)) << offset;
  }
  let borrow_in = (generate << 1).wrapping_add(propagate) ^ propagate;
  let borrow_lanes = borrow_in.to_le_bytes();
  // All ones when a borrow leaves the top lane, so the original value is kept.
  let keep = 0u8.wrapping_sub((borrow_in >> V.strict_mul(8)).to_le_bytes()[0] & 1);
  let mut result = [_mm512_setzero_si512(); V];
  for (((out, &lanes), &modulus), &borrow) in result.iter_mut().zip(value).zip(m).zip(&borrow_lanes) {
    let difference = _mm512_sub_epi64(lanes, modulus);
    let difference = _mm512_and_si512(_mm512_mask_sub_epi64(difference, borrow, difference, one), mask);
    *out = _mm512_mask_blend_epi64(keep, difference, lanes);
  }
  result
}
