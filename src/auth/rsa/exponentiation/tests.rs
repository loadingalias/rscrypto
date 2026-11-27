use super::*;
use crate::auth::rsa::tests::*;

#[test]
fn rsa_miri_proof_selects_every_window_across_private_limb_and_scratch_layouts() {
  const MARKER: u64 = 0xa5a5_5a5a_c3c3_3c3c;

  // Include the AVX2 prefix boundary before the supported default CRT widths and their tails.
  for limbs in [7, 8, 9].into_iter().chain(rsa_miri_private_limb_widths()) {
    for capacity in [limbs, limbs.strict_mul(2)] {
      let mut scratch = RsaPrivateExponentScratch::new(capacity);
      rsa_miri_fill_exponent_scratch(&mut scratch, MARKER);

      {
        let workspace = scratch
          .workspace(limbs)
          .expect("valid RSA private-exponent scratch must bind to its limb width");
        assert_eq!(workspace.t.len(), limbs.strict_mul(2).strict_add(2));
        assert_eq!(workspace.representative.len(), limbs);
        assert_eq!(workspace.one.len(), limbs);
        assert_eq!(workspace.base.len(), limbs);
        assert_eq!(workspace.acc.len(), limbs);
        assert_eq!(workspace.squared.len(), limbs);
        assert_eq!(workspace.multiplied.len(), limbs);
        assert_eq!(workspace.selected.len(), limbs);
        assert_eq!(workspace.reduced.len(), limbs);
        assert_eq!(
          workspace.table.len(),
          limbs.strict_mul(PRIVATE_FIXED_WINDOW_TABLE_ENTRIES)
        );

        for window in 0..PRIVATE_FIXED_WINDOW_TABLE_ENTRIES {
          let start = window.strict_mul(limbs);
          for (limb, slot) in workspace.table[start..start.strict_add(limbs)].iter_mut().enumerate() {
            *slot = rsa_miri_table_limb(limbs, window, limb);
          }
        }

        let window_count = u8::try_from(PRIVATE_FIXED_WINDOW_TABLE_ENTRIES)
          .expect("the RSA fixed-window table entry count must fit in one byte");
        for window in 0..window_count {
          workspace.selected.fill(MARKER);
          private_select_window_power(workspace.selected, workspace.table, window);
          let start = usize::from(window).strict_mul(limbs);
          assert_eq!(
            workspace.selected,
            &workspace.table[start..start.strict_add(limbs)],
            "wrong fixed-window row for {limbs} limbs, capacity {capacity}, window {window}"
          );
        }
        for window in [16, 31, 240, 255] {
          workspace.selected.fill(MARKER);
          private_select_window_power(workspace.selected, workspace.table, window);
          assert!(
            workspace.selected.iter().all(|&limb| limb == 0),
            "out-of-range window {window}"
          );
        }
      }

      assert!(scratch.selected.as_slice()[limbs..].iter().all(|&limb| limb == MARKER));
      let active_table_len = limbs.strict_mul(PRIVATE_FIXED_WINDOW_TABLE_ENTRIES);
      assert!(
        scratch.table.as_slice()[active_table_len..]
          .iter()
          .all(|&limb| limb == MARKER)
      );

      scratch.clear();
      rsa_miri_assert_exponent_scratch_zeroed(&scratch);
    }

    let mut undersized = RsaPrivateExponentScratch::new(limbs.strict_sub(1));
    assert!(matches!(
      undersized.workspace(limbs),
      Err(RsaPrivateOpError::InvalidScratch)
    ));

    let mut short_table = RsaPrivateExponentScratch::new(limbs);
    let _ = short_table.table.limbs.pop();
    assert!(matches!(
      short_table.workspace(limbs),
      Err(RsaPrivateOpError::InvalidScratch)
    ));
  }
}

// Private scratch is sized by the public modulus width. Its IFMA workspace must still cover
// each CRT prime width, or signing silently keeps the generic exponentiation.
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "sse2",
  not(feature = "portable-only"),
  not(miri)
))]
#[test]
fn ifma_workspace_covers_crt_prime_widths() {
  for prime_limbs in [16usize, 24, 32] {
    if rsa_x86_64_ifma::shape(prime_limbs).is_none() {
      continue;
    }
    let scratch = RsaPrivateExponentScratch::new(prime_limbs.strict_mul(2));
    assert!(
      scratch.ifma.as_slice().len() >= rsa_x86_64_ifma::workspace_words(prime_limbs),
      "{prime_limbs}-limb primes"
    );
  }
}

// The IFMA kernel replaces the CRT exponentiation on AVX-512 IFMA hosts; the portable
// fixed-window exponentiation is the oracle. Hosts without IFMA have no second
// implementation to compare, so the cases run only where the kernel is selected.
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "sse2",
  not(feature = "portable-only"),
  not(miri)
))]
#[test]
fn ifma_exponentiation_matches_portable_exponentiation() {
  let mut state = 0x7f4a_7c15_9e37_79b9u64;
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
  for limbs in [16usize, 24, 32] {
    if rsa_x86_64_ifma::shape(limbs).is_none() {
      continue;
    }
    let bytes = limbs.strict_mul(8);
    for top in [0xffu8, 0x80] {
      let mut modulus_bytes = random_bytes(bytes);
      modulus_bytes[0] |= top;
      *modulus_bytes.last_mut().expect("modulus width must be nonzero") |= 1;
      let modulus = private_component_modulus(&modulus_bytes).expect("odd test modulus must import");
      let mut scratch = RsaPrivateExponentScratch::new(limbs);
      let mut modulus_minus_one = modulus_bytes.clone();
      *modulus_minus_one.last_mut().expect("modulus width must be nonzero") &= !1;
      let mut one = vec![0u8; bytes];
      *one.last_mut().expect("modulus width must be nonzero") = 1;
      let mut bases = vec![vec![0u8; bytes], one, modulus_minus_one];
      let mut exponents = vec![vec![0u8; bytes], vec![0xff; bytes]];
      for _ in 0..3 {
        let mut base = random_bytes(bytes);
        base[0] &= modulus_bytes[0] >> 1;
        bases.push(base);
        exponents.push(random_bytes(bytes));
      }
      for base in &bases {
        for exponent in &exponents {
          let mut ifma = vec![0u8; bytes];
          let mut portable = vec![0u8; bytes];
          private_exponentiate_representative_with_scratch(&modulus, exponent, base, &mut ifma, &mut scratch)
            .expect("IFMA exponentiation must succeed");
          private_exponentiate_representative(&modulus, exponent, base, &mut portable)
            .expect("portable exponentiation must succeed");
          assert_eq!(ifma, portable, "{limbs} limbs");
        }
      }
    }
  }
}

// The CRT pair interleaves two primes of one width; each half must equal the portable
// exponentiation of its own prime, including when the halves' inputs differ in kind.
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "sse2",
  not(feature = "portable-only"),
  not(miri)
))]
#[test]
fn ifma_crt_pair_matches_portable_exponentiation() {
  let mut state = 0x2545_f491_4f6c_dd1du64;
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
  for limbs in [16usize, 24, 32] {
    if rsa_x86_64_ifma::shape(limbs).is_none() {
      continue;
    }
    let bytes = limbs.strict_mul(8);
    let mut moduli = Vec::new();
    for top in [0xffu8, 0x80] {
      let mut modulus_bytes = random_bytes(bytes);
      modulus_bytes[0] |= top;
      *modulus_bytes.last_mut().expect("modulus width must be nonzero") |= 1;
      moduli.push(modulus_bytes);
    }
    let components = moduli
      .iter()
      .map(|bytes| private_component_modulus(bytes).expect("odd test modulus must import"))
      .collect::<Vec<_>>();
    let mut operands = Vec::new();
    for modulus_bytes in &moduli {
      let mut modulus_minus_one = modulus_bytes.clone();
      *modulus_minus_one.last_mut().expect("modulus width must be nonzero") &= !1;
      let mut base = random_bytes(bytes);
      base[0] &= modulus_bytes[0] >> 1;
      let mut bases = vec![vec![0u8; bytes], modulus_minus_one, base];
      let mut exponents = vec![vec![0u8; bytes], vec![0xff; bytes], random_bytes(bytes)];
      // Pair every base kind with every exponent kind across the two halves.
      bases.rotate_left(operands.len());
      exponents.rotate_left(operands.len());
      operands.push((bases, exponents));
    }
    let mut scratch = RsaPrivateExponentScratch::new(limbs);
    for (base_p, base_q) in operands[0].0.iter().zip(&operands[1].0) {
      for (exponent_p, exponent_q) in operands[0].1.iter().zip(&operands[1].1) {
        let mut pair_p = vec![0u8; bytes];
        let mut pair_q = vec![0u8; bytes];
        private_exponentiate_crt_with_scratch(
          (&components[0], exponent_p, base_p, &mut pair_p),
          (&components[1], exponent_q, base_q, &mut pair_q),
          &mut scratch,
        )
        .expect("IFMA CRT pair must succeed");
        let mut portable_p = vec![0u8; bytes];
        let mut portable_q = vec![0u8; bytes];
        private_exponentiate_representative(&components[0], exponent_p, base_p, &mut portable_p)
          .expect("portable exponentiation must succeed");
        private_exponentiate_representative(&components[1], exponent_q, base_q, &mut portable_q)
          .expect("portable exponentiation must succeed");
        assert_eq!(pair_p, portable_p, "{limbs} limbs, first prime");
        assert_eq!(pair_q, portable_q, "{limbs} limbs, second prime");
      }
    }
  }
}

// The NEON selector takes each limb's low and high 32 bits from the entries the two halves of
// `windows` name; the paired engine keeps the two CRT primes in those halves.
#[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
#[test]
fn neon_window_select_takes_each_half_from_its_window() {
  let entry_limbs = 8usize;
  let table: Vec<u64> = (0..PRIVATE_FIXED_WINDOW_TABLE_ENTRIES.strict_mul(entry_limbs))
    .map(|index| {
      let index = index as u64;
      (index.wrapping_mul(0x9e37_79b9) << 32) | index.wrapping_mul(0x85eb_ca6b) & 0xffff_ffff
    })
    .collect();
  for low in 0..16u64 {
    for high in 0..16u64 {
      let mut out = [u64::MAX; 8];
      // SAFETY: Two chunks: `out` has eight writable limbs, and `table` holds sixteen
      // eight-limb entries, so the reads of eight limbs at `8·i` stay inside it.
      unsafe { private_select_window_power_neon(out.as_mut_ptr(), table.as_ptr(), entry_limbs, 2, low | high << 32) };
      for (limb, &value) in out.iter().enumerate() {
        let from_low = table[usize::try_from(low)
          .expect("window fits usize")
          .strict_mul(entry_limbs)
          .strict_add(limb)];
        let from_high = table[usize::try_from(high)
          .expect("window fits usize")
          .strict_mul(entry_limbs)
          .strict_add(limb)];
        assert_eq!(
          value,
          (from_low & 0xffff_ffff) | (from_high & !0xffff_ffff),
          "windows {low}, {high}, limb {limb}"
        );
      }
    }
  }
}

#[cfg(all(
  target_arch = "aarch64",
  target_vendor = "apple",
  not(feature = "portable-only"),
  not(miri)
))]
#[test]
fn neon_pair_workspace_covers_crt_prime_widths() {
  for prime_limbs in [16usize, 24, 32] {
    let scratch = RsaPrivateExponentScratch::new(prime_limbs.strict_mul(2));
    assert!(
      scratch.pair.as_slice().len() >= rsa_aarch64_pair::workspace_words(prime_limbs),
      "{prime_limbs}-limb primes"
    );
    assert!(
      rsa_aarch64_pair::shape(prime_limbs).is_some(),
      "{prime_limbs}-limb primes"
    );
  }
}

// The paired engine runs the two primes in the two vector lanes; each lane must equal the
// portable exponentiation of its own prime, with every base kind against every exponent kind
// across the lanes, and with moduli at both ends of the width so the 2^(28·D) > 4·m bound is
// exercised.
#[cfg(all(
  target_arch = "aarch64",
  target_vendor = "apple",
  not(feature = "portable-only"),
  not(miri)
))]
#[test]
fn neon_crt_pair_matches_portable_exponentiation() {
  let mut state = 0x6a09_e667_f3bc_c909u64;
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
  for limbs in [16usize, 24, 32] {
    let bytes = limbs.strict_mul(8);
    let mut moduli = Vec::new();
    for top in [0xffu8, 0x80] {
      let mut modulus_bytes = random_bytes(bytes);
      modulus_bytes[0] |= top;
      if top == 0x80 {
        modulus_bytes[0] &= 0x80;
      }
      *modulus_bytes.last_mut().expect("modulus width must be nonzero") |= 1;
      moduli.push(modulus_bytes);
    }
    let components = moduli
      .iter()
      .map(|bytes| private_component_modulus(bytes).expect("odd test modulus must import"))
      .collect::<Vec<_>>();
    let mut operands = Vec::new();
    for modulus_bytes in &moduli {
      let mut modulus_minus_one = modulus_bytes.clone();
      *modulus_minus_one.last_mut().expect("modulus width must be nonzero") &= !1;
      let mut one = vec![0u8; bytes];
      *one.last_mut().expect("modulus width must be nonzero") = 1;
      let mut base = random_bytes(bytes);
      base[0] &= modulus_bytes[0] >> 1;
      let mut bases = vec![vec![0u8; bytes], modulus_minus_one, one, base];
      let mut exponents = vec![
        vec![0u8; bytes],
        vec![0xff; bytes],
        random_bytes(bytes),
        random_bytes(bytes),
      ];
      bases.rotate_left(operands.len());
      exponents.rotate_left(operands.len().strict_mul(2));
      operands.push((bases, exponents));
    }
    let mut scratch = RsaPrivateExponentScratch::new(limbs.strict_mul(2));
    assert!(private_crt_pair_neon_shape(&components[0], &components[1], &scratch).is_some());
    for (base_p, base_q) in operands[0].0.iter().zip(&operands[1].0) {
      for (exponent_p, exponent_q) in operands[0].1.iter().zip(&operands[1].1) {
        let mut pair_p = vec![0u8; bytes];
        let mut pair_q = vec![0u8; bytes];
        private_exponentiate_crt_with_scratch(
          (&components[0], exponent_p, base_p, &mut pair_p),
          (&components[1], exponent_q, base_q, &mut pair_q),
          &mut scratch,
        )
        .expect("NEON CRT pair must succeed");
        let mut portable_p = vec![0u8; bytes];
        let mut portable_q = vec![0u8; bytes];
        private_exponentiate_representative(&components[0], exponent_p, base_p, &mut portable_p)
          .expect("portable exponentiation must succeed");
        private_exponentiate_representative(&components[1], exponent_q, base_q, &mut portable_q)
          .expect("portable exponentiation must succeed");
        assert_eq!(pair_p, portable_p, "{limbs} limbs, first prime");
        assert_eq!(pair_q, portable_q, "{limbs} limbs, second prime");
      }
    }
  }
}

#[cfg(any(
  all(
    target_arch = "aarch64",
    target_os = "linux",
    not(feature = "portable-only"),
    not(miri)
  ),
  target_arch = "powerpc64",
  target_arch = "s390x",
  target_arch = "riscv64"
))]
#[test]
fn five_bit_crt_exponentiation_matches_big_integer_power() {
  for words in [8usize, 16, 20, 24, 32] {
    let bytes = words.strict_mul(8);
    let mut modulus_bytes = vec![0xffu8; bytes];
    modulus_bytes[0] = 0xbf;
    let modulus = private_component_modulus(&modulus_bytes).expect("odd modulus imports");
    let reference_modulus = BigUint::from_bytes_be(&modulus_bytes);
    let mut scratch = RsaPrivateExponentScratch::new(words.strict_mul(2));
    assert!(scratch.table.as_slice().len() >= words.strict_mul(32));
    for pattern in [0u8, 0xff, 0xa5, 0x81] {
      let exponent = vec![pattern; bytes];
      for base in [0u8, 7] {
        let mut input = vec![0u8; bytes];
        input[bytes.strict_sub(1)] = base;
        let mut out = vec![0u8; bytes];
        private_exponentiate_representative_with_scratch(&modulus, &exponent, &input, &mut out, &mut scratch)
          .expect("five-bit exponentiation succeeds");
        let expected = BigUint::from_bytes_be(&input).modpow(&BigUint::from_bytes_be(&exponent), &reference_modulus);
        assert_eq!(
          BigUint::from_bytes_be(&out),
          expected,
          "{words} words, {pattern:#x}, base {base}"
        );
      }
    }
    if words == 16 {
      // Alternate both halves to expose a stale contribution or a bad 16-entry offset.
      for window in (0u8..16).flat_map(|low| [low, 31u8.strict_sub(low)]) {
        let mut exponent = vec![0u8; bytes];
        exponent[bytes.strict_sub(1)] = window;
        let mut input = vec![0u8; bytes];
        input[bytes.strict_sub(1)] = 7;
        let mut out = vec![0u8; bytes];
        private_exponentiate_representative_with_scratch(&modulus, &exponent, &input, &mut out, &mut scratch)
          .expect("every five-bit window succeeds");
        let expected = BigUint::from(7u8).modpow(&BigUint::from(window), &reference_modulus);
        assert_eq!(BigUint::from_bytes_be(&out), expected, "window {window}");
      }
    }
  }
}
