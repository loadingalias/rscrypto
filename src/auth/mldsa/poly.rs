//! Arithmetic in FIPS 204's ring `Z_q[X]/(X^256 + 1)`.
//!
//! Coefficients are canonical residues at operation boundaries. Forward NTT
//! additionally converts to Montgomery representation; inverse NTT returns
//! ordinary residues. Products in the NTT domain therefore need exactly one
//! Montgomery multiplication.

use crate::traits::ct;

#[cfg(all(
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  target_feature = "neon",
  not(feature = "portable-only")
))]
mod aarch64;

#[cfg(all(
  target_arch = "x86_64",
  target_os = "linux",
  not(miri),
  not(feature = "portable-only")
))]
mod x86_64;

pub(super) const N: usize = 256;
pub(super) const Q: u32 = 8_380_417;
const R2: u32 = 2_365_951; // 2^64 mod q.
const NEG_Q_INVERSE: u32 = 4_236_238_847; // -q^-1 mod 2^32.
const INV_N: u32 = 8_347_681; // 256^-1 mod q.

pub(super) struct Poly(pub(super) [u32; N]);

impl Poly {
  pub(super) const fn zero() -> Self {
    Self([0; N])
  }

  pub(super) fn copy_from(&mut self, other: &Self) {
    self.0.copy_from_slice(&other.0);
  }

  /// FIPS 204 Algorithm 41, with Montgomery-domain butterfly operands.
  pub(super) fn ntt(&mut self) {
    #[cfg(all(
      target_arch = "x86_64",
      target_os = "linux",
      not(miri),
      not(feature = "portable-only")
    ))]
    if crate::platform::caps().has(crate::platform::caps::x86::AVX2) {
      // SAFETY: Cached capability detection establishes CPU and OS AVX2 support.
      // Fixed-size references provide initialized arrays and disjoint output.
      unsafe {
        x86_64::ntt(self);
      }
      return;
    }
    #[cfg(all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      target_feature = "neon",
      not(feature = "portable-only")
    ))]
    {
      // SAFETY: This branch requires compile-time NEON on macOS/Linux AArch64.
      unsafe {
        aarch64::ntt(self);
      }
    }
    #[cfg(not(all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      target_feature = "neon",
      not(feature = "portable-only")
    )))]
    {
      self.ntt_portable();
    }
  }

  #[cfg(any(
    test,
    not(all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      target_feature = "neon",
      not(feature = "portable-only")
    ))
  ))]
  fn ntt_portable(&mut self) {
    // Fuse conversion with the first butterfly: M(M(b, R2), zeta)
    // equals M(b, M(R2, zeta)). Both outputs remain in Montgomery form.
    // This removes 128 Montgomery multiplications and a full-array pass.
    const FIRST_FACTOR: u32 = montgomery(R2, ROOTS[1]);
    let (left, right) = self.0.split_at_mut(N / 2);
    for (a, b) in left.iter_mut().zip(right) {
      let x = to_montgomery(*a);
      let y = montgomery(*b, FIRST_FACTOR);
      *a = add(x, y);
      *b = sub(x, y);
    }
    let mut root = 2usize;
    let mut width = N / 4;
    while width != 0 {
      for block in self.0.chunks_exact_mut(width.strict_mul(2)) {
        let zeta = ROOTS[root];
        root = root.strict_add(1);
        let (left, right) = block.split_at_mut(width);
        for (a, b) in left.iter_mut().zip(right) {
          let t = montgomery(zeta, *b);
          *b = sub(*a, t);
          *a = add(*a, t);
        }
      }
      width >>= 1;
    }
  }

  /// FIPS 204 Algorithm 42, including conversion out of Montgomery form.
  pub(super) fn inverse_ntt(&mut self) {
    #[cfg(all(
      target_arch = "x86_64",
      target_os = "linux",
      not(miri),
      not(feature = "portable-only")
    ))]
    if crate::platform::caps().has(crate::platform::caps::x86::AVX2) {
      // SAFETY: Cached capability detection establishes CPU and OS AVX2 support.
      // Fixed-size references provide initialized arrays and disjoint output.
      unsafe {
        x86_64::inverse_ntt(self);
      }
      return;
    }
    #[cfg(all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      target_feature = "neon",
      not(feature = "portable-only")
    ))]
    {
      // SAFETY: This branch requires compile-time NEON on macOS/Linux AArch64.
      unsafe {
        aarch64::inverse_ntt(self);
      }
    }
    #[cfg(not(all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      target_feature = "neon",
      not(feature = "portable-only")
    )))]
    {
      self.inverse_ntt_portable();
    }
  }

  #[cfg(any(
    test,
    not(all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      target_feature = "neon",
      not(feature = "portable-only")
    ))
  ))]
  fn inverse_ntt_portable(&mut self) {
    let mut root = N;
    let mut width = 1usize;
    let mut bound = Q;
    while width < N / 2 {
      for block in self.0.chunks_exact_mut(width.strict_mul(2)) {
        root = root.strict_sub(1);
        let zeta = Q.strict_sub(ROOTS[root]);
        let (left, right) = block.split_at_mut(width);
        for (a, b) in left.iter_mut().zip(right) {
          // At entry, a,b < bound = width*q. The sum and biased difference
          // are below 2*bound; the Montgomery output is below 2q. Deferring
          // canonical reduction therefore preserves the next stage's bound.
          let difference = a.strict_add(bound).strict_sub(*b);
          *a = a.strict_add(*b);
          *b = montgomery_unreduced(zeta, difference);
        }
      }
      width = width.strict_mul(2);
      bound = bound.strict_mul(2);
    }
    // Fuse the last butterfly with normalization. Its right operand otherwise
    // undergoes two Montgomery multiplications: M(M(zeta, a-b), INV_N).
    // M(a-b, M(zeta, INV_N)) is identical modulo q. The precomputed right
    // factor is an ordinary residue, so both outputs leave Montgomery form.
    const LAST_FACTOR: u32 = montgomery(Q.strict_sub(ROOTS[1]), INV_N);
    let (left, right) = self.0.split_at_mut(N / 2);
    for (a, b) in left.iter_mut().zip(right) {
      // Here a,b < 128q. Both operands stay below 256q < 2^31, and the
      // final Montgomery products restore canonical ordinary residues.
      let difference = a.strict_add(bound).strict_sub(*b);
      *a = montgomery(a.strict_add(*b), INV_N);
      *b = montgomery(difference, LAST_FACTOR);
    }
  }

  pub(super) fn product(&mut self, a: &Self, b: &[u32; N]) {
    for ((out, &a), &b) in self.0.iter_mut().zip(&a.0).zip(b) {
      *out = montgomery(a, b);
    }
  }

  pub(super) fn accumulate_product(&mut self, a: &[u32; N], b: &Self) {
    #[cfg(all(
      target_arch = "x86_64",
      target_os = "linux",
      not(miri),
      not(feature = "portable-only")
    ))]
    if crate::platform::caps().has(crate::platform::caps::x86::AVX2) {
      // SAFETY: Cached capability detection establishes CPU and OS AVX2 support.
      // Fixed-size references provide initialized arrays and disjoint output.
      unsafe {
        x86_64::accumulate_product(&mut self.0, a, &b.0);
      }
      return;
    }
    #[cfg(all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      target_feature = "neon",
      not(feature = "portable-only")
    ))]
    {
      // SAFETY: Compile-time NEON is required on this macOS/Linux AArch64 path.
      // Fixed-size references provide initialized, disjoint output and inputs.
      unsafe {
        aarch64::accumulate_product(&mut self.0, a, &b.0);
      }
    }
    #[cfg(not(all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      target_feature = "neon",
      not(feature = "portable-only")
    )))]
    {
      for ((out, &a), &b) in self.0.iter_mut().zip(a).zip(&b.0) {
        *out = add(*out, montgomery(a, b));
      }
    }
  }

  pub(super) fn add_assign(&mut self, rhs: &Self) {
    for (a, &b) in self.0.iter_mut().zip(&rhs.0) {
      *a = add(*a, b);
    }
  }

  pub(super) fn sub_assign(&mut self, rhs: &Self) {
    for (a, &b) in self.0.iter_mut().zip(&rhs.0) {
      *a = sub(*a, b);
    }
  }

  /// Aggregate the norm decision without returning at the first coefficient.
  pub(super) fn exceeds_bound(&self, bound: u32) -> u32 {
    let mut invalid = 0;
    for &x in &self.0 {
      let negative = u32::from(x > Q / 2);
      let magnitude = select(x, Q.strict_sub(x), negative);
      invalid |= u32::from(magnitude >= bound);
    }
    invalid
  }
}

impl Drop for Poly {
  fn drop(&mut self) {
    ct::zeroize_words(&mut self.0);
  }
}

#[inline]
pub(super) const fn select(a: u32, b: u32, bit: u32) -> u32 {
  let mask = 0u32.wrapping_sub(bit);
  a ^ ((a ^ b) & mask)
}

/// Reduce an input in [0, 2q) with one masked subtraction.
#[inline]
const fn reduce(x: u32) -> u32 {
  let difference = x.wrapping_sub(Q);
  difference.wrapping_add(0u32.wrapping_sub(difference >> 31) & Q)
}

#[inline]
pub(super) const fn add(a: u32, b: u32) -> u32 {
  reduce(a.strict_add(b))
}

#[inline]
pub(super) const fn sub(a: u32, b: u32) -> u32 {
  reduce(a.strict_add(Q).strict_sub(b))
}

/// Return a canonical Montgomery product for a*b < 256q^2.
#[inline]
pub(super) const fn montgomery(a: u32, b: u32) -> u32 {
  reduce(montgomery_unreduced(a, b))
}

/// Return a Montgomery product below 2q for a*b < 256q^2.
/// Since 256q < 2^31 and m < 2^32, t + mq < 2^56 cannot overflow.
/// Its quotient is below q + 256q^2/2^32 < 2q, so one subtraction canonicalizes it.
#[inline]
#[expect(
  clippy::cast_possible_truncation,
  reason = "low word is arithmetic modulo 2^32; the shifted quotient is below 2q"
)]
const fn montgomery_unreduced(a: u32, b: u32) -> u32 {
  let t = (a as u64).strict_mul(b as u64);
  let m = (t as u32).wrapping_mul(NEG_Q_INVERSE);
  (t.strict_add((m as u64).strict_mul(Q as u64)) >> 32) as u32
}

#[inline]
pub(super) const fn to_montgomery(x: u32) -> u32 {
  montgomery(x, R2)
}

/// Generate FIPS 204 roots from zeta=1753 and BitRev8, not an imported table.
const ROOTS: [u32; N] = {
  let mut roots = [0; N];
  let mut i = 0usize;
  while i < N {
    let mut exponent = i.reverse_bits() >> usize::BITS.strict_sub(8);
    let mut power = to_montgomery(1753);
    let mut value = to_montgomery(1);
    while exponent != 0 {
      if exponent & 1 != 0 {
        value = montgomery(value, power);
      }
      power = montgomery(power, power);
      exponent >>= 1;
    }
    roots[i] = value;
    i = i.strict_add(1);
  }
  roots
};

/// Algorithm 35. The low component is returned as a canonical residue.
pub(super) fn power2_round(x: u32) -> (u32, u32) {
  let high = x.strict_add(4095) >> 13;
  (high, sub(x, high << 13))
}

/// Algorithm 36. Public parameter selection gives constant divisors.
pub(super) fn decompose(x: u32, gamma2: u32) -> (u32, u32) {
  let high = high_bits(x, gamma2);
  (high, sub(x, high.strict_mul(gamma2.strict_mul(2))))
}

/// High component of Algorithm 36 for canonical coefficients.
/// Callers needing only this component need not compute the low remainder.
#[inline]
pub(super) fn high_bits(x: u32, gamma2: u32) -> u32 {
  let rounded = x.strict_add(gamma2).strict_sub(1);
  let (quotient, modulus) = if gamma2 == 95_232 {
    (rounded / 190_464, 44)
  } else {
    (rounded / 523_776, 16)
  };
  select(quotient, 0, u32::from(quotient == modulus))
}

/// Algorithm 40; inputs are public verification data.
pub(super) fn use_hint(x: u32, hint: bool, gamma2: u32) -> u32 {
  let (high, low) = decompose(x, gamma2);
  if !hint {
    return high;
  }
  let modulus = if gamma2 == 95_232 { 44 } else { 16 };
  if low != 0 && low <= Q / 2 {
    if high.strict_add(1) == modulus {
      0
    } else {
      high.strict_add(1)
    }
  } else if high == 0 {
    modulus.strict_sub(1)
  } else {
    high.strict_sub(1)
  }
}

#[cfg(test)]
mod tests {
  use super::{N, Poly, Q, montgomery};

  // Force a valid u32-aligned address that is not SIMD-aligned. The portable
  // differential must catch a backend that accidentally requires aligned loads.
  #[cfg(any(
    all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      target_feature = "neon",
      not(feature = "portable-only")
    ),
    all(
      target_arch = "x86_64",
      target_os = "linux",
      not(miri),
      not(feature = "portable-only")
    )
  ))]
  #[repr(C, align(32))]
  struct Unaligned {
    padding: u32,
    poly: Poly,
  }

  #[cfg(any(
    all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      target_feature = "neon",
      not(feature = "portable-only")
    ),
    all(
      target_arch = "x86_64",
      target_os = "linux",
      not(miri),
      not(feature = "portable-only")
    )
  ))]
  #[test]
  fn ntt_accelerated_matches_portable() {
    #[cfg(target_arch = "x86_64")]
    if !crate::platform::caps().has(crate::platform::caps::x86::AVX2) {
      return;
    }
    // Canonical extremes, lane-alternating extremes, sparse basis vectors,
    // and dense deterministic inputs expose root order and lane mixups.
    let mut state = 0x6d6c_6473u32;
    for case in 0..324 {
      let mut expected = Poly::zero();
      for (i, x) in expected.0.iter_mut().enumerate() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *x = match case {
          0 => 0,
          1 => Q - 1,
          2 => {
            if i % 2 == 0 {
              Q - 1
            } else {
              0
            }
          }
          3 => {
            if i % 2 == 0 {
              0
            } else {
              Q - 1
            }
          }
          4..260 => {
            if i == case - 4 {
              Q - 1
            } else {
              0
            }
          }
          _ => state % Q,
        };
      }
      let mut storage = Unaligned {
        padding: 0,
        poly: Poly::zero(),
      };
      let actual = &mut storage.poly;
      actual.copy_from(&expected);
      let mut forward_storage = Unaligned {
        padding: 0,
        poly: Poly::zero(),
      };
      let forward = &mut forward_storage.poly;
      forward.copy_from(&expected);
      let mut forward_expected = Poly::zero();
      forward_expected.copy_from(&expected);
      forward_expected.ntt_portable();
      forward.ntt();
      assert_eq!(forward.0, forward_expected.0, "forward NTT case {case}");
      expected.inverse_ntt_portable();
      actual.inverse_ntt();
      assert_eq!(actual.0, expected.0, "inverse NTT case {case}");
    }
  }

  #[cfg(any(
    all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      target_feature = "neon",
      not(feature = "portable-only")
    ),
    all(
      target_arch = "x86_64",
      target_os = "linux",
      not(miri),
      not(feature = "portable-only")
    )
  ))]
  #[test]
  fn accumulation_accelerated_matches_portable() {
    #[cfg(target_arch = "x86_64")]
    if !crate::platform::caps().has(crate::platform::caps::x86::AVX2) {
      return;
    }
    let mut state = 0x91e1_0da5u32;
    for case in 0..68 {
      let mut a = Poly::zero();
      let mut b = Poly::zero();
      let mut storage = Unaligned {
        padding: 0,
        poly: Poly::zero(),
      };
      let actual = &mut storage.poly;
      let mut expected = Poly::zero();
      for i in 0..N {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        a.0[i] = if case == 0 {
          0
        } else if case == 1 {
          Q - 1
        } else {
          state % Q
        };
        state = state.rotate_left(7).wrapping_add(0x9e37_79b9);
        b.0[i] = if case < 2 { Q - 1 } else { state % Q };
        actual.0[i] = if i % 2 == 0 { Q - 1 } else { 0 };
        expected.0[i] = super::add(actual.0[i], montgomery(a.0[i], b.0[i]));
      }
      actual.accumulate_product(&a.0, &b);
      assert_eq!(actual.0, expected.0, "accumulation case {case}");
    }
  }

  #[test]
  fn montgomery_accepts_unreduced_butterfly_operands() {
    // Independent integer-modulo oracle; 2^32 * R_INVERSE = 1 mod q.
    const R_INVERSE: u64 = 8_265_825;
    let modulus = u64::from(Q);
    assert_eq!((1u64 << 32).strict_mul(R_INVERSE) % modulus, 1);
    let check = |a, b| {
      let product = u64::from(a).strict_mul(u64::from(b)) % modulus;
      let expected = product.strict_mul(R_INVERSE) % modulus;
      let actual = montgomery(a, b);
      assert!(actual < Q);
      assert_eq!(u64::from(actual), expected, "operands {a}, {b}");
      let unreduced = super::montgomery_unreduced(a, b);
      assert!(unreduced < Q.strict_mul(2));
      assert_eq!(u64::from(unreduced) % modulus, expected, "unreduced operands {a}, {b}");
    };
    let boundaries = [0, 1, Q.strict_sub(1), Q, Q.strict_add(1), Q.strict_mul(2).strict_sub(1)];
    for a in boundaries {
      for b in boundaries {
        check(a, b);
      }
    }
    // Deferred inverse reductions allow one operand near each growing stage
    // bound. The other operand is a canonical root or normalization factor.
    for multiple in [4, 8, 16, 32, 64, 128, 256] {
      let limit = Q.strict_mul(multiple);
      for wide in [
        limit.strict_sub(1),
        limit.strict_sub(Q),
        limit.strict_sub(Q).strict_add(1),
      ] {
        for canonical in [0, 1, Q / 2, Q.strict_sub(1)] {
          check(canonical, wide);
          check(wide, canonical);
        }
      }
    }
  }

  #[test]
  fn ntt_product_matches_schoolbook_at_residue_boundaries() {
    let boundaries = [0, 1, Q / 2, (Q / 2).strict_add(1), Q.strict_sub(2), Q.strict_sub(1)];
    for case in 0..3 {
      let a: [u32; N] = core::array::from_fn(|i| match case {
        0 => Q.strict_sub(1),
        1 => boundaries[i % boundaries.len()],
        _ => u32::from(i == N.strict_sub(1)),
      });
      let b: [u32; N] = core::array::from_fn(|i| match case {
        0 => Q.strict_sub(1),
        1 => boundaries[i.strict_mul(5).strict_add(3) % boundaries.len()],
        _ => u32::from(i == 1),
      });
      // Independent schoolbook multiplication in Z_q[X]/(X^256 + 1).
      // Each accumulator is bounded by 256*(q-1)^2, below i64::MAX.
      let mut expected = [0i64; N];
      for (i, &a) in a.iter().enumerate() {
        for (j, &b) in b.iter().enumerate() {
          let product = i64::from(a).strict_mul(i64::from(b));
          let index = i.strict_add(j);
          if index < N {
            expected[index] = expected[index].strict_add(product);
          } else {
            let index = index.strict_sub(N);
            expected[index] = expected[index].strict_sub(product);
          }
        }
      }
      let expected = expected.map(|x| u32::try_from(x.rem_euclid(i64::from(Q))).expect("canonical oracle residue"));
      let mut a = Poly(a);
      let mut b = Poly(b);
      a.ntt();
      b.ntt();
      let mut product = Poly::zero();
      product.product(&a, &b.0);
      product.inverse_ntt();
      assert_eq!(product.0, expected, "residue-boundary case {case}");
    }
  }
}
