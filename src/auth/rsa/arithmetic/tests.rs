use super::*;
use crate::auth::rsa::tests::*;

#[cfg(all(
  target_arch = "aarch64",
  any(target_os = "linux", target_os = "macos"),
  not(feature = "portable-only"),
  not(miri)
))]
macro_rules! assert_aarch64_rsa_montgomery_backend_matches_portable {
  ($backend:ident) => {
    for words in [16usize, 24, 32, 48, 64, 128] {
      assert!(rsa_aarch64_mont::supports(words));
      let modulus = rsa_montgomery_test_modulus(words);
      let a = rsa_montgomery_test_limbs(words, 0x243f_6a88_85a3_08d3);
      let b = rsa_montgomery_test_limbs(words, 0x1319_8a2e_0370_7344);
      let scratch_words = $backend::bignum_mont_scratch_words(words);
      assert!(scratch_words >= rsa_aarch64_mont::scratch_words(words));
      let mut asm_t = vec![0u64; scratch_words];
      let mut portable_t = vec![0u64; words.strict_mul(2).strict_add(2)];
      let mut asm_out = vec![0u64; words];
      let mut portable_out = vec![0u64; words];

      rsa_aarch64_mont::mul(&mut asm_out, &a, &b, &modulus.limbs, &modulus.n0_256, &mut asm_t);
      mont_mul_cios_portable(&mut portable_out, &a, &b, &modulus, &mut portable_t);
      assert_limbs_match_as_bytes("mont_mul", words, &asm_out, &portable_out);

      asm_t.fill(0);
      portable_t.fill(0);
      let mut asm_square = a.clone();
      rsa_aarch64_mont::square_in_place(&mut asm_square, &modulus.limbs, &modulus.n0_256, &mut asm_t);
      mont_mul_cios_portable(&mut portable_out, &a, &a, &modulus, &mut portable_t);
      assert_limbs_match_as_bytes("mont_square_in_place", words, &asm_square, &portable_out);

      asm_t.fill(0);
      portable_t.fill(0);
      let mut asm_left = a.clone();
      rsa_aarch64_mont::mul_in_place_left(&mut asm_left, &b, &modulus.limbs, &modulus.n0_256, &mut asm_t);
      mont_mul_cios_portable(&mut portable_out, &a, &b, &modulus, &mut portable_t);
      assert_limbs_match_as_bytes("mont_mul_in_place_left", words, &asm_left, &portable_out);

      if words < 32 {
        continue;
      }

      asm_t.fill(0);
      portable_t.fill(0);
      if words == 32 {
        $backend::mont_reduce_cios_32(&mut asm_out, &a, &modulus.limbs, modulus.n0, &mut asm_t);
      } else {
        $backend::mont_reduce_cios_words(&mut asm_out, &a, &modulus.limbs, modulus.n0, words, &mut asm_t);
      }
      mont_reduce_cios_portable(&mut portable_out, &a, &modulus, &mut portable_t);
      assert_limbs_match_as_bytes("mont_reduce", words, &asm_out, &portable_out);
    }
  };
}

#[test]
fn private_mod_subtraction_handles_borrow_equal_and_no_borrow() {
  for (left, right, modulus, expected) in [
    ([0x03], [0x01], [0x0b], [0x02]),
    ([0x01], [0x03], [0x0b], [0x09]),
    ([0x03], [0x03], [0x0b], [0x00]),
    ([0x00], [0x00], [0x01], [0x00]),
  ] {
    let mut out = [0u8; 1];
    private_sub_mod_unsigned_be_to_fixed(&left, &right, &modulus, &mut out)
      .expect("the bounded modular-subtraction fixture must be valid");
    assert_eq!(out, expected);
  }

  let mut out = [0u8; 2];
  private_sub_mod_unsigned_be_to_fixed(&[0x00, 0x01], &[0x01, 0x00], &[0x01, 0x01], &mut out)
    .expect("the borrow-propagation modular-subtraction fixture must be valid");
  assert_eq!(out, [0x00, 0x02]);
}

#[test]
fn public_montgomery_threshold_uses_cios_through_rsa8192() {
  let rsa4096 = RsaPublicModulus::new(&[0xff; 512], 4096);
  let rsa4160 = RsaPublicModulus::new(&[0xff; 520], 4160);
  let rsa6144 = RsaPublicModulus::new(&[0xff; 768], 6144);
  let rsa8128 = RsaPublicModulus::new(&[0xff; 1016], 8128);
  let rsa8192 = RsaPublicModulus::new(&[0xff; 1024], 8192);

  assert!(use_public_cios_montgomery(&rsa4096));
  assert!(use_public_cios_montgomery(&rsa4160));
  assert!(use_public_cios_montgomery(&rsa6144));
  assert!(use_public_cios_montgomery(&rsa8128));
  assert!(use_public_cios_montgomery(&rsa8192));
}

#[test]
fn shared_montgomery_threshold_keeps_product_for_gap_widths() {
  let rsa4096 = RsaPublicModulus::new(&[0xff; 512], 4096);
  let rsa4160 = RsaPublicModulus::new(&[0xff; 520], 4160);
  let rsa6144 = RsaPublicModulus::new(&[0xff; 768], 6144);
  let rsa8128 = RsaPublicModulus::new(&[0xff; 1016], 8128);
  let rsa8192 = RsaPublicModulus::new(&[0xff; 1024], 8192);

  assert!(use_cios_montgomery(&rsa4096));
  assert!(!use_cios_montgomery(&rsa4160));
  assert!(!use_cios_montgomery(&rsa6144));
  assert!(!use_cios_montgomery(&rsa8128));
  assert!(use_cios_montgomery(&rsa8192));
}

#[test]
fn portable_montgomery_kernels_match_big_integer_products() {
  fn big(limbs: &[u64]) -> BigUint {
    let bytes: Vec<u8> = limbs.iter().flat_map(|limb| limb.to_le_bytes()).collect();
    BigUint::from_bytes_le(&bytes)
  }

  #[cfg(all(target_arch = "powerpc64", not(feature = "portable-only"), not(miri)))]
  {
    // Exercise every combination of zero, the sign boundary and full carry words in the
    // native multiply-accumulate. BigUint supplies the full sum independently of its XER chain.
    let edges = [0, 1, 1u64 << 63, u64::MAX - 1, u64::MAX];
    for acc in edges {
      for a in edges {
        for b in edges {
          for carry in edges {
            let (low, high) = mont_mac(acc, a, b, carry);
            assert_eq!(big(&[low, high]), big(&[a]) * big(&[b]) + big(&[acc]) + big(&[carry]));
          }
        }
      }
    }
  }

  let mut state = 0x2545_f491_4f6c_dd1du64;
  let mut next = move || {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
  };
  let widths: &[usize] = if cfg!(miri) {
    &[1, 3]
  } else {
    &[1, 2, 3, 5, 8, 16, 17, 32, 48, 64, 128]
  };
  for &words in widths {
    for case in 0..6 {
      let mut bytes: Vec<u8> = (0..words.strict_mul(8)).map(|_| next().to_le_bytes()[0]).collect();
      if case == 0 {
        bytes.fill(0xff);
      }
      bytes[0] |= 0x80;
      *bytes.last_mut().expect("nonzero width") |= 1;
      let modulus = RsaPublicModulus::new(&bytes, words.strict_mul(64));
      let n = big(&modulus.limbs);
      let r = BigUint::from(1u8) << words.strict_mul(64);
      // Zero, n - 1, and random values below n.
      let mut below = |kind: usize| -> Vec<u64> {
        match kind {
          0 => vec![0u64; words],
          1 => {
            let mut value = modulus.limbs.to_vec();
            value[0] ^= 1;
            value
          }
          _ => {
            let mut value: Vec<u64> = (0..words).map(|_| next()).collect();
            let top = words.strict_sub(1);
            value[top] %= modulus.limbs[top];
            value
          }
        }
      };
      for (left_kind, right_kind) in [(0, 2), (1, 1), (1, 2), (2, 1), (2, 2), (2, 2)] {
        let a = below(left_kind);
        let b = below(right_kind);
        let mut t = vec![0u64; words.strict_mul(2).strict_add(2)];
        let mut out = vec![0u64; words];
        let product = |value: &[u64]| -> BigUint { big(value) * &r % &n };

        mont_mul_cios_portable(&mut out, &a, &b, &modulus, &mut t);
        assert!(big(&out) < n, "mul result below n for {words} limbs");
        assert_eq!(product(&out), big(&a) * big(&b) % &n, "mul for {words} limbs");

        let mut squared = a.clone();
        mont_square_portable_in_place(&mut squared, &modulus, &mut t);
        assert!(big(&squared) < n, "square result below n for {words} limbs");
        assert_eq!(product(&squared), big(&a) * big(&a) % &n, "square for {words} limbs");

        mont_reduce_cios_portable(&mut out, &a, &modulus, &mut t);
        assert!(big(&out) < n, "reduce result below n for {words} limbs");
        assert_eq!(product(&out), big(&a) % &n, "reduce for {words} limbs");

        if words == 16 {
          // Exercise the production specialization on every test host. A fixed-width loop
          // must preserve full carries and overwrite the complete wide result.
          mont_mul_cios_portable_at_width::<16>(&mut out, &a, &b, &modulus, &mut t);
          assert!(big(&out) < n, "fixed-width mul result below n");
          assert_eq!(product(&out), big(&a) * big(&b) % &n, "fixed-width mul");
          let wide = &mut t[..32];
          square_wide_portable_at_width::<16>(wide, &a);
          assert_eq!(big(wide), big(&a) * big(&a), "fixed-width wide square");
          mont_reduce_wide_portable_at_width::<16>(&mut out, wide, &modulus);
          assert!(big(&out) < n, "fixed-width square result below n");
          assert_eq!(product(&out), big(&a) * big(&a) % &n, "fixed-width square");
        }

        // The kernels for widths without CIOS.
        mont_mul(&mut out, &a, &b, &modulus, &mut t);
        assert!(big(&out) < n, "wide mul result below n for {words} limbs");
        assert_eq!(product(&out), big(&a) * big(&b) % &n, "wide mul for {words} limbs");

        let mut squared = a.clone();
        mont_square_product(&mut squared, &modulus, &mut t);
        assert!(big(&squared) < n, "wide square result below n for {words} limbs");
        assert_eq!(
          product(&squared),
          big(&a) * big(&a) % &n,
          "wide square for {words} limbs"
        );

        mont_reduce(&mut out, &a, &modulus, &mut t);
        assert!(big(&out) < n, "wide reduce result below n for {words} limbs");
        assert_eq!(product(&out), big(&a) % &n, "wide reduce for {words} limbs");
      }
    }
  }
}

#[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
#[test]
fn aarch64_owned_montgomery_matches_portable_on_random_and_edge_operands() {
  let mut state = 0x9e37_79b9_7f4a_7c15u64;
  let mut next = move || {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
  };
  for words in [8usize, 12, 16, 20, 24, 32, 48, 64, 128] {
    for case in 0..24 {
      let mut bytes: Vec<u8> = (0..words.strict_mul(8)).map(|_| next().to_le_bytes()[0]).collect();
      if case == 0 {
        bytes.fill(0xff);
      }
      bytes[0] |= 0x80;
      *bytes.last_mut().expect("nonzero width") |= 1;
      let modulus = RsaPublicModulus::new(&bytes, words.strict_mul(64));
      // Operands below the modulus: zero, modulus - 1, and random values with a smaller top limb.
      let mut below = |kind: usize| -> Vec<u64> {
        match kind {
          0 => vec![0u64; words],
          1 => {
            let mut value = modulus.limbs.to_vec();
            value[0] ^= 1;
            value
          }
          _ => {
            let mut value: Vec<u64> = (0..words).map(|_| next()).collect();
            let top = words.strict_sub(1);
            value[top] %= modulus.limbs[top];
            value
          }
        }
      };
      for (left_kind, right_kind) in [(0, 2), (1, 1), (1, 2), (2, 1), (2, 2), (2, 2)] {
        let a = below(left_kind);
        let b = below(right_kind);
        let mut t = vec![0u64; words.strict_mul(2).strict_add(2)];
        let mut portable_t = t.clone();
        let mut expected = vec![0u64; words];
        let mut out = vec![0u64; words];

        rsa_aarch64_mont::mul(&mut out, &a, &b, &modulus.limbs, &modulus.n0_256, &mut t);
        mont_mul_cios_portable(&mut expected, &a, &b, &modulus, &mut portable_t);
        assert_limbs_match_as_bytes("owned mul", words, &out, &expected);

        let mut left = a.clone();
        rsa_aarch64_mont::mul_in_place_left(&mut left, &b, &modulus.limbs, &modulus.n0_256, &mut t);
        assert_limbs_match_as_bytes("owned mul_in_place_left", words, &left, &expected);

        let mut squared = a.clone();
        rsa_aarch64_mont::square_in_place(&mut squared, &modulus.limbs, &modulus.n0_256, &mut t);
        mont_mul_cios_portable(&mut expected, &a, &a, &modulus, &mut portable_t);
        assert_limbs_match_as_bytes("owned square_in_place", words, &squared, &expected);
      }
    }
  }
}

#[cfg(all(
  target_arch = "aarch64",
  target_os = "linux",
  not(feature = "portable-only"),
  not(miri)
))]
#[test]
fn aarch64_linux_rsa_montgomery_asm_matches_portable_across_supported_widths() {
  assert_aarch64_rsa_montgomery_backend_matches_portable!(rsa_aarch64_asm);
}

#[cfg(all(
  target_arch = "aarch64",
  target_os = "macos",
  not(feature = "portable-only"),
  not(miri)
))]
#[test]
fn aarch64_macos_rsa_montgomery_asm_matches_portable_across_supported_widths() {
  assert_aarch64_rsa_montgomery_backend_matches_portable!(rsa_aarch64_asm);
}

// x86-64 BMI2/ADX discovery is `std`-only: without it `caps()` reports just
// the compile-time feature set, so the owned kernel is unreachable and there
// is no second implementation to differentially test. Compile this evidence
// test exactly when the kernel can be observed as available, and keep the
// assertion hard inside it.
#[cfg(all(
  target_arch = "x86_64",
  not(feature = "portable-only"),
  not(miri),
  any(feature = "std", all(target_feature = "bmi2", target_feature = "adx"))
))]
#[test]
fn x86_64_rsa_montgomery_kernel_matches_portable_across_supported_widths() {
  for words in [16usize, 24, 32, 48, 64, 128] {
    assert!(
      rsa_x86_64_mont::supports(words),
      "RSA x86-64 Montgomery evidence requires BMI2 and ADX"
    );

    let modulus = rsa_montgomery_test_modulus(words);
    let a = rsa_montgomery_test_limbs(words, 0x243f_6a88_85a3_08d3);
    let b = rsa_montgomery_test_limbs(words, 0x1319_8a2e_0370_7344);
    let r2 = public_montgomery_r2(&modulus);
    let mut kernel_t = vec![0u64; rsa_x86_64_mont::scratch_words(words)];
    let mut portable_t = vec![0u64; words.strict_mul(2).strict_add(2)];
    let mut kernel_out = vec![0u64; words];
    let mut portable_out = vec![0u64; words];

    rsa_x86_64_mont::mul(&mut kernel_out, &a, &b, &modulus.limbs, modulus.n0, &mut kernel_t);
    mont_mul_cios_portable(&mut portable_out, &a, &b, &modulus, &mut portable_t);
    assert_limbs_match_as_bytes("mont_mul", words, &kernel_out, &portable_out);

    let mut kernel_square = a.clone();
    rsa_x86_64_mont::square_in_place(&mut kernel_square, &modulus.limbs, modulus.n0, &mut kernel_t);
    mont_mul_cios_portable(&mut portable_out, &a, &a, &modulus, &mut portable_t);
    assert_limbs_match_as_bytes("mont_square_in_place", words, &kernel_square, &portable_out);

    let mut kernel_left = a.clone();
    rsa_x86_64_mont::mul_in_place_left(&mut kernel_left, &b, &modulus.limbs, modulus.n0, &mut kernel_t);
    mont_mul_cios_portable(&mut portable_out, &a, &b, &modulus, &mut portable_t);
    assert_limbs_match_as_bytes("mont_mul_in_place_left", words, &kernel_left, &portable_out);

    rsa_x86_64_mont::mul(&mut kernel_out, &a, &r2, &modulus.limbs, modulus.n0, &mut kernel_t);
    let mut kernel_acc = kernel_out.clone();
    for _ in 0..16 {
      rsa_x86_64_mont::square_in_place(&mut kernel_acc, &modulus.limbs, modulus.n0, &mut kernel_t);
    }
    rsa_x86_64_mont::mul_in_place_left(&mut kernel_acc, &kernel_out, &modulus.limbs, modulus.n0, &mut kernel_t);
    mont_reduce_cios_portable(&mut kernel_out, &kernel_acc, &modulus, &mut portable_t);

    let mut portable_acc = vec![0u64; words];
    public_e65537_cios_portable(&mut portable_out, &a, &r2, &mut portable_acc, &modulus, &mut portable_t);
    assert_limbs_match_as_bytes("public_e65537", words, &kernel_out, &portable_out);
  }
}

#[cfg(all(
  target_arch = "x86_64",
  not(feature = "portable-only"),
  not(miri),
  any(feature = "std", all(target_feature = "bmi2", target_feature = "adx"))
))]
#[test]
fn x86_64_rsa_montgomery_kernel_matches_portable_on_random_and_edge_operands() {
  let mut state = 0x9e37_79b9_7f4a_7c15u64;
  let mut next = move || {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
  };
  for words in [8usize, 16, 24, 32, 40, 48, 64, 128] {
    assert!(
      rsa_x86_64_mont::supports(words),
      "the kernel must support {words} limbs"
    );
    for case in 0..24 {
      let mut bytes: Vec<u8> = (0..words.strict_mul(8)).map(|_| next().to_le_bytes()[0]).collect();
      if case == 0 {
        bytes.fill(0xff);
      }
      bytes[0] |= 0x80;
      *bytes.last_mut().expect("nonzero width") |= 1;
      let modulus = RsaPublicModulus::new(&bytes, words.strict_mul(64));
      // Operands below the modulus: zero, modulus - 1, and random values with a smaller top limb.
      let mut below = |kind: usize| -> Vec<u64> {
        match kind {
          0 => vec![0u64; words],
          1 => {
            let mut value = modulus.limbs.to_vec();
            value[0] ^= 1;
            value
          }
          _ => {
            let mut value: Vec<u64> = (0..words).map(|_| next()).collect();
            let top = words.strict_sub(1);
            value[top] %= modulus.limbs[top];
            value
          }
        }
      };
      for (left_kind, right_kind) in [(0, 2), (1, 1), (1, 2), (2, 1), (2, 2), (2, 2)] {
        let a = below(left_kind);
        let b = below(right_kind);
        let mut t = vec![0u64; rsa_x86_64_mont::scratch_words(words)];
        let mut portable_t = vec![0u64; words.strict_mul(2).strict_add(2)];
        let mut expected = vec![0u64; words];
        let mut out = vec![0u64; words];

        rsa_x86_64_mont::mul(&mut out, &a, &b, &modulus.limbs, modulus.n0, &mut t);
        mont_mul_cios_portable(&mut expected, &a, &b, &modulus, &mut portable_t);
        assert_limbs_match_as_bytes("owned mul", words, &out, &expected);

        let mut left = a.clone();
        rsa_x86_64_mont::mul_in_place_left(&mut left, &b, &modulus.limbs, modulus.n0, &mut t);
        assert_limbs_match_as_bytes("owned mul_in_place_left", words, &left, &expected);

        let mut squared = a.clone();
        rsa_x86_64_mont::square_in_place(&mut squared, &modulus.limbs, modulus.n0, &mut t);
        mont_mul_cios_portable(&mut expected, &a, &a, &modulus, &mut portable_t);
        assert_limbs_match_as_bytes("owned square_in_place", words, &squared, &expected);
      }
    }
  }
}

#[test]
fn double_mod_in_place_matches_reference_edges() {
  let cases: [(&[u64], &[u64]); 5] = [
    (&[0x0000_0000_0000_0004], &[0x0000_0000_0000_000b]),
    (&[0x0000_0000_0000_0006], &[0x0000_0000_0000_000b]),
    (&[0x8000_0000_0000_0000], &[0xffff_ffff_ffff_fffb]),
    (&[0xffff_ffff_ffff_fffb], &[0xffff_ffff_ffff_fffd]),
    (
      &[0xffff_ffff_ffff_ffff, 0x7fff_ffff_ffff_ffff],
      &[0xffff_ffff_ffff_fffd, 0xffff_ffff_ffff_ffff],
    ),
  ];

  for (value, modulus) in cases {
    let mut optimized = value.to_vec();
    let mut reference = value.to_vec();
    double_mod_in_place(&mut optimized, modulus);
    reference_double_mod_in_place(&mut reference, modulus);
    assert_eq!(
      optimized, reference,
      "doubling mismatch for value={value:x?} modulus={modulus:x?}"
    );
  }
}

#[test]
fn pow2_mod_direct_prefix_matches_naive_doubling() {
  let mut sparse_modulus = vec![0u8; 256];
  sparse_modulus[0] = 0x80;
  sparse_modulus[255] = 0x01;
  let dense_modulus = vec![0xff; 256];
  let mut unaligned_modulus = vec![0u8; 257];
  unaligned_modulus[0] = 0x01;
  unaligned_modulus[1] = 0x80;
  unaligned_modulus[256] = 0x01;

  for modulus_bytes in [&sparse_modulus, &dense_modulus, &unaligned_modulus] {
    let modulus = limbs_from_be(modulus_bytes);
    let modulus_bits = limb_bit_len(&modulus);
    let mut bit_counts = vec![
      0,
      1,
      63,
      64,
      65,
      modulus_bits.saturating_sub(2),
      modulus_bits.saturating_sub(1),
    ];
    bit_counts.extend([modulus_bits, modulus_bits.strict_add(1), modulus.len().strict_mul(128)]);

    for bits in bit_counts {
      let mut optimized = vec![0u64; modulus.len()];
      let mut naive = vec![0u64; modulus.len()];
      pow2_mod_into(&mut optimized, bits, &modulus);
      naive[0] = 1;
      for _ in 0..bits {
        reference_double_mod_in_place(&mut naive, &modulus);
      }
      assert_eq!(optimized, naive, "pow2_mod_into mismatch at {bits} bits");
    }
  }
}

#[test]
fn public_montgomery_r2_matches_doubling_definition() {
  // Widths cover every Montgomery assembly width, the portable CIOS widths around them,
  // and the product-scanning path above 128 limbs. Short top limbs exercise extra doublings.
  let widths: &[usize] = if cfg!(miri) {
    &[1, 2, 3]
  } else {
    &[1, 2, 3, 15, 16, 24, 31, 32, 33, 48, 64, 65, 127, 128, 129]
  };
  let mut state = 0x6a09_e667_f3bc_c908u64;
  for &words in widths {
    for top in [0xffu8, 0x80, 0x01] {
      let mut bytes = vec![0u8; words.strict_mul(8)];
      for byte in &mut bytes {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = state.to_le_bytes()[0];
      }
      bytes[0] = top;
      *bytes.last_mut().expect("modulus width must be nonzero") |= 1;
      let modulus = RsaPublicModulus::new(&bytes, unsigned_be_bit_len(&bytes));

      let mut expected = vec![0u64; words];
      expected[0] = 1;
      for _ in 0..words.strict_mul(128) {
        reference_double_mod_in_place(&mut expected, &modulus.limbs);
      }

      assert_eq!(
        &*public_montgomery_r2(&modulus),
        expected.as_slice(),
        "{words} limbs, top byte {top:#04x}"
      );
      let precomputed = RsaPublicModulus::new_with_montgomery_r2(&bytes, modulus.bits);
      assert_eq!(
        precomputed.montgomery_r2(),
        Some(expected.as_slice()),
        "precomputed R² for {words} limbs"
      );
    }
  }
}

#[test]
fn private_montgomery_reduction_matches_bit_serial_reduction() {
  // Widths cover the Montgomery assembly widths, portable CIOS widths, and the
  // product-scanning widths. Values stay below modulus·R, the CRT input bound.
  let widths: &[usize] = if cfg!(miri) {
    &[1, 2]
  } else {
    &[1, 2, 3, 16, 24, 31, 32, 33, 48, 64, 65, 128, 129]
  };
  let mut state = 0xbb67_ae85_84ca_a73bu64;
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
  for &words in widths {
    let limb_bytes = words.strict_mul(8);
    let mut modulus_bytes = random_bytes(limb_bytes);
    modulus_bytes[0] |= 0x80;
    *modulus_bytes.last_mut().expect("modulus width must be nonzero") |= 1;
    let modulus = private_component_modulus(&modulus_bytes).expect("odd test modulus must import");
    let mut scratch = RsaPrivateExponentScratch::new(words);
    let reduce_slow = |value: &[u8], scratch: &mut RsaPrivateExponentScratch| {
      let mut out = vec![0u8; limb_bytes];
      private_import_unsigned_be_mod_to_fixed(value, &modulus, &mut out, scratch)
        .expect("bit-serial reduction must succeed");
      out
    };

    let mut modulus_minus_one = modulus_bytes.clone();
    *modulus_minus_one.last_mut().expect("modulus width must be nonzero") &= !1;
    let mut cases = vec![
      vec![0u8; limb_bytes.strict_mul(2)],
      [modulus_minus_one.clone(), vec![0xff; limb_bytes]].concat(),
      [modulus_minus_one, vec![0; limb_bytes]].concat(),
      modulus_bytes,
      vec![0xff; limb_bytes],
      vec![1],
      [vec![0x01, 0x02, 0x03], random_bytes(limb_bytes)].concat(),
    ];
    for _ in 0..4 {
      let high = reduce_slow(&random_bytes(limb_bytes), &mut scratch);
      cases.push([high, random_bytes(limb_bytes)].concat());
    }

    for value in cases {
      let mut fast = vec![0u8; limb_bytes];
      private_reduce_unsigned_be_mod_to_fixed(&value, limb_bytes, &modulus, &mut fast, &mut scratch)
        .expect("Montgomery reduction must succeed");
      assert_eq!(
        fast,
        reduce_slow(&value, &mut scratch),
        "{words} limbs, {} value bytes",
        value.len()
      );
    }
  }
}

#[test]
fn public_operation_matches_independent_rsa2048_e65537_vector() {
  let modulus = hex_to_vec(
    "\
ef8bb02b8e4aec1abc6fac7a0d6fb1f2649bb86a1567423fee4a194a250461a9db702558e92e52cc\
907963d84731a7adaf4c609e1b7c7d7c187099a43857f7628f5d20416fcb48987c9d6f12cfc6bc\
260c9b5506be3fe3cd218ddb37ef5b30feb16172a9832312726ed135c0540ef9d3229b87b5566f\
3355c90f301b856aa822878269806079ab7267cdc6c7403d7be3fa652065b2d39f2dbf9fb61ed9\
71fee37432ebe31d9aa465dbae96b0edd5ffddf1b49e03346a02290fed1e4e31f6b3b6e1f839f\
d5add90a8a212c10dd997b0a4efcb3df990808509dcb28c504e0649827a83ffd864395d1f62f2\
9a004f44423a44b07de943a60fba844a9da3603ce5c5",
  );
  let input = hex_to_vec(
    "\
3450869c4ccbee98815e55cb42f2dd85a3427d3f65e33d29352293e18cde9582a9fbc54b440984\
1ba8d931a9a9411192516a9fbd3a7b886e7f8b8f3f7bb5403309eee9d7234df0b5934e18a1dc\
9e3b568a3fab6947cefe50500abcbda19fd9ab7b7e90a95801e36a020ba79bdc94346198d98131\
6864a06a43448b62acb7a8472661323175f04c5e447d0017e4073efc55f59f79f34aaa3be8ae7\
0d26db78b25e9dfb23856d1b1e024aedfcfd649d209412c0c80832ca3466965eeff539afe791f\
451b554e212cff4d92466438062c5202169b0adf0c95b7d3d31414602cf9d185252b550cc2e8f\
5be08b7fc71f51210ff88363badadfaf5c2915c3a10b2389e",
  );
  let expected = hex_to_vec(
    "\
020d016f2b1394b5ec00d4ddd1725435747a31fd4b2489fad76060b68b2259089d304b1d3c98\
e0a343c4b313d15b6022b0400dde22538f30e8474d483189c04ece5acc8aaf1481b362c2d2fe7\
fa853e856a0aba66cc47cf9e59052fdd4c5f4155bcc2a3f3330e2c48b7f45d1e66d8cd04829c\
0ba2e598569b4eeb8538c3cdf8e02c838d04bdc661b5d8c5291b0feebf284eb9deea03dd0226\
bdb322e180a6ab522ee40a02a0daf41094a2938d39698ab16381ed4d3ddd01bd05a8aa9113d8\
ec34e8c72cc58fd5324fbe1ddd9714909caedfaa38706cfa66d9bc1026ba3ec1188092392a54a\
3e94bf239ee74517b71ec2464551f8174dbd0f3952ffb41070c754",
  );
  let key = RsaPublicKey::from_pkcs1_der_with_policy(
    &test_pkcs1_public_key(&modulus, &[0x01, 0x00, 0x01]),
    &RsaPublicKeyPolicy::modulus_2048_to_8192(),
  )
  .expect("independent-vector public key must import");
  let mut out = vec![0u8; key.modulus().len()];
  let mut scratch = key.public_scratch();

  key
    .public_operation_with_scratch(&input, &mut out, &mut scratch)
    .expect("independent-vector representative must exponentiate");

  assert_eq!(out, expected);
}
