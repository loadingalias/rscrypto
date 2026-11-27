use super::*;
use crate::auth::rsa::tests::*;

#[cfg(feature = "getrandom")]
#[test]
fn private_key_random_blinding_factor_has_valid_crt_inverse() {
  let key = prevalidated_test_private_key();
  let blinding = key
    .components
    .random_blinding_factor()
    .expect("test key must yield an invertible random blinding factor");

  let mut check = vec![0u8; key.signature_len()];
  mod_mul_representatives(
    &key.public_key().modulus,
    blinding.factor(),
    blinding.inverse(),
    &mut check,
  )
  .expect("blinding factor and inverse must multiply modulo n");

  assert!(check[..check.len().strict_sub(1)].iter().all(|&byte| byte == 0));
  assert_eq!(check.last().copied(), Some(1));
}

#[test]
fn private_i31_fixed_work_multiply_matches_wrapping_products() {
  const U64_EDGES: [u64; 8] = [0, 1, 2, 0xffff_ffff, 1u64 << 32, 1u64 << 63, u64::MAX - 1, u64::MAX];
  const U32_EDGES: [u32; 7] = [0, 1, 2, 0x7fff_ffff, 1u32 << 31, u32::MAX - 1, u32::MAX];

  for lhs in U64_EDGES {
    for rhs in U64_EDGES {
      assert_eq!(private_i31_ct_mul_u64_low(lhs, rhs), lhs.wrapping_mul(rhs));
    }
  }
  for lhs in U32_EDGES {
    for rhs in U32_EDGES {
      assert_eq!(private_i31_ct_mul_u32_low(lhs, rhs), lhs.wrapping_mul(rhs));
    }
  }

  let mut lhs = 0x243f_6a88_85a3_08d3u64;
  let mut rhs = 0x1319_8a2e_0370_7344u64;
  for _ in 0..1024 {
    lhs = lhs.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1);
    rhs = rhs.wrapping_mul(0xd134_2543_de82_ef95).wrapping_add(1);
    let lhs_low = private_i31_low_u32(lhs);
    let rhs_low = private_i31_low_u32(rhs);
    assert_eq!(private_i31_ct_mul_u64_low(lhs, rhs), lhs.wrapping_mul(rhs));
    assert_eq!(
      private_i31_ct_mul_u32_low(lhs_low, rhs_low),
      lhs_low.wrapping_mul(rhs_low)
    );
  }
}

#[test]
fn private_i31_riscv64_multiply_matches_wrapping_products() {
  // Each operand's top bit set and clear, with both low-bit parities, covers every correction.
  const EDGES: [u64; 10] = [
    0,
    1,
    2,
    3,
    0xffff_ffff,
    (1u64 << 63) - 1,
    1u64 << 63,
    (1u64 << 63) | 1,
    u64::MAX - 1,
    u64::MAX,
  ];
  for lhs in EDGES {
    for rhs in EDGES {
      assert_eq!(
        private_i31_mul_u64_low_riscv64(lhs, rhs),
        lhs.wrapping_mul(rhs),
        "{lhs:x} * {rhs:x}"
      );
    }
  }

  let mut lhs = 0x243f_6a88_85a3_08d3u64;
  let mut rhs = 0x1319_8a2e_0370_7344u64;
  for _ in 0..4096 {
    lhs = lhs.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1);
    rhs = rhs.wrapping_mul(0xd134_2543_de82_ef95).wrapping_add(1);
    assert_eq!(
      private_i31_mul_u64_low_riscv64(lhs, rhs),
      lhs.wrapping_mul(rhs),
      "{lhs:x} * {rhs:x}"
    );
    // Sign-extended 31-bit factors, as the inverse passes them.
    let factor = (lhs >> 33).cast_signed().wrapping_sub(1 << 30).cast_unsigned();
    assert_eq!(private_i31_mul_u64_low_riscv64(rhs, factor), rhs.wrapping_mul(factor));
  }
}

#[test]
fn private_modular_inverse_matches_exhaustive_small_domain() {
  for modulus_value in (3u16..=255).step_by(2) {
    let modulus_byte = u8::try_from(modulus_value).expect("the exhaustive modulus is at most one byte");
    let modulus_bytes = [modulus_byte];
    let modulus = RsaPublicModulus::new(&modulus_bytes, unsigned_be_bit_len(&modulus_bytes));
    let mut scratch = RsaPrivateInverseScratch::new(modulus.bits);

    for input_value in 1u16..modulus_value {
      let input = [u8::try_from(input_value).expect("the exhaustive input is at most one byte")];
      let mut out = [0xa5];
      let expected = (1u16..modulus_value).find(|candidate| input_value.strict_mul(*candidate) % modulus_value == 1);
      let result = private_modular_inverse_odd(&modulus, &input, &mut out, &mut scratch);

      if let Some(expected) = expected {
        assert_eq!(result, Ok(()), "inverse rejected for {input_value} mod {modulus_value}");
        assert_eq!(
          u16::from(out[0]),
          expected,
          "wrong inverse for {input_value} mod {modulus_value}"
        );
      } else {
        assert_eq!(
          result,
          Err(RsaPrivateOpError::InvalidBlindingFactor),
          "non-invertible input accepted for {input_value} mod {modulus_value}"
        );
        assert_eq!(out, [0]);
      }
      assert!(scratch.is_cleared());
    }
  }
}

// Widths straddle the 46-bit divstep bound and every 62-bit limb boundary; random odd
// moduli also produce non-invertible factors, whose rejection must match.
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
#[test]
fn private_s62_inverse_matches_i31_inverse() {
  let widths: &[usize] = if cfg!(miri) {
    &[9, 64]
  } else {
    &[
      9, 45, 46, 47, 61, 62, 63, 64, 65, 123, 124, 125, 1024, 2047, 2048, 3072, 4096, 8192,
    ]
  };
  let mut state = 0x9e37_79b9_7f4a_7c15u64;
  let mut random_bytes = |len: usize| {
    (0..len)
      .map(|_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state.to_le_bytes()[0]
      })
      .collect::<Vec<u8>>()
  };
  for &bits in widths {
    let len = bits.div_ceil(8);
    let mut modulus_bytes = random_bytes(len);
    let top_bits = bits.strict_sub(len.strict_sub(1).strict_mul(8));
    modulus_bytes[0] =
      (modulus_bytes[0] & (u8::MAX >> (8usize.strict_sub(top_bits)))) | (1 << (top_bits.strict_sub(1)));
    *modulus_bytes.last_mut().expect("modulus width must be nonzero") |= 1;
    let modulus = RsaPublicModulus::new(&modulus_bytes, unsigned_be_bit_len(&modulus_bytes));
    assert_eq!(modulus.bits, bits);
    let mut s62_scratch = RsaPrivateS62InverseScratch::new(bits);
    let mut i31_scratch = RsaPrivateI31InverseScratch::new(bits);

    let mut modulus_minus_one = modulus_bytes.clone();
    *modulus_minus_one.last_mut().expect("modulus width must be nonzero") &= !1;
    let mut one = vec![0u8; len];
    *one.last_mut().expect("modulus width must be nonzero") = 1;
    let mut factors = vec![one, modulus_minus_one];
    for _ in 0..6 {
      let mut factor = random_bytes(len);
      factor[0] &= modulus_bytes[0] >> 1;
      factors.push(factor);
    }
    for factor in factors {
      if !private_valid_blinding_factor(&factor, &modulus_bytes) {
        continue;
      }
      let mut s62 = vec![0xa5u8; len];
      let mut i31 = vec![0x5au8; len];
      let s62_result = private_modular_inverse_odd_s62(&modulus, &factor, &mut s62, &mut s62_scratch);
      let i31_result = private_modular_inverse_odd_i31(&modulus, &factor, &mut i31, &mut i31_scratch);
      assert_eq!(s62_result, i31_result, "{bits}-bit result");
      if s62_result.is_ok() {
        assert_eq!(s62, i31, "{bits}-bit inverse");
      }
      s62_scratch.clear();
      i31_scratch.clear();
      assert!(s62_scratch.is_cleared() && i31_scratch.is_cleared());
    }
  }
}

#[test]
fn private_modular_inverse_matches_biguint_for_full_width_inputs() {
  let modulus_bytes = rsa_private_modulus();
  let modulus = RsaPublicModulus::new(&modulus_bytes, unsigned_be_bit_len(&modulus_bytes));
  let modulus_oracle = BigUint::from_bytes_be(&modulus_bytes);
  let mut scratch = RsaPrivateInverseScratch::new(modulus.bits);
  let mut factor = vec![0u8; modulus_bytes.len()];
  let mut inverse = vec![0u8; modulus_bytes.len()];
  let mut state = 0x243f_6a88_85a3_08d3u64;

  for _ in 0..64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    let candidate = state | 1;
    factor.fill(0);
    let candidate_start = factor.len().strict_sub(8);
    factor[candidate_start..].copy_from_slice(&candidate.to_be_bytes());

    private_modular_inverse_odd(&modulus, &factor, &mut inverse, &mut scratch)
      .expect("a nonzero 64-bit factor cannot share either 1024-bit prime factor");
    let product = BigUint::from(candidate) * BigUint::from_bytes_be(&inverse);
    assert_eq!(product % &modulus_oracle, BigUint::from(1u8));
    assert!(scratch.is_cleared());
  }

  // Exercise modulus-width factors independently of the small-candidate
  // cases above. Since the RSA modulus is odd, every n - 2^k candidate is
  // coprime to n and therefore has an inverse.
  for sample in 0usize..64 {
    let shift = sample.strict_mul(modulus.bits.strict_sub(2)) / 63;
    let candidate = &modulus_oracle - (BigUint::from(1u8) << shift);
    let candidate_bytes = candidate.to_bytes_be();
    factor.fill(0);
    let candidate_start = factor.len().strict_sub(candidate_bytes.len());
    factor[candidate_start..].copy_from_slice(&candidate_bytes);

    private_modular_inverse_odd(&modulus, &factor, &mut inverse, &mut scratch)
      .expect("n - 2^k must be invertible modulo an odd RSA modulus");
    let product = &candidate * BigUint::from_bytes_be(&inverse);
    assert_eq!(product % &modulus_oracle, BigUint::from(1u8));
    assert!(scratch.is_cleared());
  }

  factor.copy_from_slice(&modulus_bytes);
  let mut borrow = 1u16;
  for byte in factor.iter_mut().rev() {
    let difference = u16::from(*byte).wrapping_sub(borrow);
    *byte = difference.to_le_bytes()[0];
    borrow = u16::from(difference > 0xff);
  }
  private_modular_inverse_odd(&modulus, &factor, &mut inverse, &mut scratch)
    .expect("n - 1 must be self-inverse modulo n");
  assert_eq!(inverse, factor);

  factor.fill(0);
  let prime = rsa_private_prime_p();
  let prime_start = factor.len().strict_sub(prime.len());
  factor[prime_start..].copy_from_slice(&prime);
  inverse.fill(0xa5);
  assert_eq!(
    private_modular_inverse_odd(&modulus, &factor, &mut inverse, &mut scratch),
    Err(RsaPrivateOpError::InvalidBlindingFactor)
  );
  assert!(inverse.iter().all(|&byte| byte == 0));
  assert!(scratch.is_cleared());
}

#[cfg(feature = "getrandom")]
#[test]
fn private_key_blinding_inverse_rejects_non_invertible_factor() {
  let key = prevalidated_test_private_key();
  let mut factor = vec![0u8; key.signature_len()];
  left_pad_be(&rsa_private_prime_p(), &mut factor).expect("prime fixture must fit the RSA representative width");
  let mut inverse = vec![0u8; key.signature_len()];

  assert_eq!(
    key
      .components
      .blinding_factor_inverse_with(&factor, &mut inverse, getrandom::fill),
    Err(RsaPrivateOpError::InvalidBlindingFactor)
  );
}

#[test]
fn masked_blinding_inverse_matches_the_unmasked_inverse_and_clears_the_mask() {
  let key = prevalidated_test_private_key();
  let len = key.signature_len();
  let modulus = key.public_key().modulus().to_vec();
  let mut factor = vec![0u8; len];
  factor[len.strict_sub(8)..].copy_from_slice(&0x9e37_79b9_7f4a_7c15u64.to_be_bytes());
  let mut expected = vec![0u8; len];
  let mut inverse_scratch = RsaPrivateInverseScratch::new(key.components.public.modulus.bits);
  private_modular_inverse_odd(
    &key.components.public.modulus,
    &factor,
    &mut expected,
    &mut inverse_scratch,
  )
  .expect("an odd 64-bit factor is invertible modulo the fixture modulus");

  let mut one = vec![0u8; len];
  one[len.strict_sub(1)] = 1;
  // The modulus is odd, so n - 1 only changes the last byte.
  let mut minus_one = modulus.clone();
  minus_one[len.strict_sub(1)] = minus_one[len.strict_sub(1)].strict_sub(1);
  let mut random = (0..len)
    .map(|index| {
      u8::try_from(index & 0xff)
        .expect("masked to one byte")
        .wrapping_mul(0x5b)
        ^ 0xa7
    })
    .collect::<Vec<_>>();
  random[0] &= 0x7f;
  for mask in [&one, &minus_one, &random] {
    let mut scratch = key.private_scratch();
    scratch.blinding_factor.as_mut_slice().copy_from_slice(&factor);
    // The first draw equals the modulus, so it is rejected and redrawn.
    let mut draws = [modulus.as_slice(), mask.as_slice()].into_iter();
    key
      .components
      .blinding_factor_inverse_into_scratch(&mut scratch, |out: &mut [u8]| {
        out.copy_from_slice(draws.next().expect("two draws"));
        Ok::<(), ()>(())
      })
      .expect("masked inverse");
    assert_eq!(scratch.blinding_inverse.as_slice(), expected.as_slice());
    assert!(scratch.inverse_mask.as_slice().iter().all(|&byte| byte == 0));
    assert!(scratch.checked.as_slice().iter().all(|&byte| byte == 0));
  }

  let mut scratch = key.private_scratch();
  scratch.blinding_factor.as_mut_slice().copy_from_slice(&factor);
  assert_eq!(
    key
      .components
      .blinding_factor_inverse_into_scratch(&mut scratch, |out: &mut [u8]| {
        out.fill(0x42);
        Err(())
      }),
    Err(RsaPrivateOpError::EntropyUnavailable)
  );
  assert!(scratch.inverse_mask.as_slice().iter().all(|&byte| byte == 0));
  assert!(scratch.blinding_inverse.as_slice().iter().all(|&byte| byte == 0));
}
