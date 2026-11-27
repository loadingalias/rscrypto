//! Private fixed-window exponentiation and target-specific selection.

use super::*;

pub(super) fn private_exponentiate_representative(
  modulus: &RsaPublicModulus,
  exponent: &[u8],
  input: &[u8],
  out: &mut [u8],
) -> Result<(), RsaPrivateOpError> {
  crate::traits::ct::with_data_independent_timing(|| {
    let bytes = modulus.bytes.len();
    let limbs = modulus.limbs.len();
    if input.len() != bytes || out.len() != bytes || exponent.len() != bytes {
      return Err(RsaPrivateOpError::InvalidLength);
    }
    let r2 = modulus
      .montgomery_r2()
      .ok_or(RsaPrivateOpError::RepresentativeOutOfRange)?;

    let mut t = SecretLimbs::zeroed(limbs.strict_mul(2).strict_add(2));
    let mut representative = SecretLimbs::zeroed(limbs);
    let mut one = SecretLimbs::zeroed(limbs);
    let mut base = SecretLimbs::zeroed(limbs);
    let mut acc = SecretLimbs::zeroed(limbs);
    let mut squared = SecretLimbs::zeroed(limbs);
    let mut multiplied = SecretLimbs::zeroed(limbs);
    let mut selected = SecretLimbs::zeroed(limbs);
    let mut reduced = SecretLimbs::zeroed(limbs);

    limbs_from_be_into(input, representative.as_mut_slice());
    if !ct_limbs_lt(representative.as_slice(), &modulus.limbs) {
      return Err(RsaPrivateOpError::RepresentativeOutOfRange);
    }

    one.as_mut_slice()[0] = 1;
    private_mont_mul(
      base.as_mut_slice(),
      representative.as_slice(),
      r2,
      modulus,
      t.as_mut_slice(),
    );
    private_mont_mul(acc.as_mut_slice(), one.as_slice(), r2, modulus, t.as_mut_slice());

    let mut table = private_fixed_window_table(base.as_slice(), acc.as_slice(), modulus, t.as_mut_slice());
    let mut index = 0;
    while index < bytes {
      let exponent_byte = exponent[index];
      private_exponentiate_window(
        &table,
        exponent_byte >> 4,
        modulus,
        RsaPrivateWindowWorkspace {
          t: t.as_mut_slice(),
          acc: acc.as_mut_slice(),
          squared: squared.as_mut_slice(),
          multiplied: multiplied.as_mut_slice(),
          selected: selected.as_mut_slice(),
        },
      );
      private_exponentiate_window(
        &table,
        exponent_byte & 0x0f,
        modulus,
        RsaPrivateWindowWorkspace {
          t: t.as_mut_slice(),
          acc: acc.as_mut_slice(),
          squared: squared.as_mut_slice(),
          multiplied: multiplied.as_mut_slice(),
          selected: selected.as_mut_slice(),
        },
      );
      index = index.strict_add(1);
    }

    ct::zeroize_words(table.as_mut_slice());
    private_mont_reduce(reduced.as_mut_slice(), acc.as_slice(), modulus, t.as_mut_slice());
    limbs_to_be(reduced.as_slice(), out);
    Ok(())
  })
}

/// IFMA workspace words for CRT primes of up to `max_limb_count` limbs; zero without the backend.
pub(super) fn private_ifma_workspace_words(max_limb_count: usize) -> usize {
  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(feature = "portable-only"),
    not(miri)
  ))]
  let words = rsa_x86_64_ifma::max_workspace_words(max_limb_count);
  #[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(feature = "portable-only"),
    not(miri)
  )))]
  let words = {
    let _ = max_limb_count;
    0
  };
  words
}

/// Paired NEON workspace words for CRT primes of up to `max_limb_count` limbs; zero without the
/// engine.
pub(super) fn private_pair_workspace_words(max_limb_count: usize) -> usize {
  #[cfg(all(
    target_arch = "aarch64",
    target_vendor = "apple",
    not(feature = "portable-only"),
    not(miri)
  ))]
  let words = rsa_aarch64_pair::max_workspace_words(max_limb_count);
  #[cfg(not(all(
    target_arch = "aarch64",
    target_vendor = "apple",
    not(feature = "portable-only"),
    not(miri)
  )))]
  let words = {
    let _ = max_limb_count;
    0
  };
  words
}

/// Both CRT exponentiations, `input^exponent mod prime` for each component.
///
/// Equal-width primes with an IFMA kernel run as one interleaved pair, and on Apple AArch64 in the
/// two lanes of the paired NEON engine. Otherwise each component runs
/// [`private_exponentiate_representative_with_scratch`] in turn. Every path validates the first
/// component, then the second, so they return the same errors.
pub(super) fn private_exponentiate_crt_with_scratch(
  first: RsaCrtExponentiation<'_>,
  second: RsaCrtExponentiation<'_>,
  scratch: &mut RsaPrivateExponentScratch,
) -> Result<(), RsaPrivateOpError> {
  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(feature = "portable-only"),
    not(miri)
  ))]
  if let Some(shape) = private_crt_pair_ifma_shape(first.0, second.0, scratch) {
    return private_exponentiate_crt_pair_ifma(first, second, shape, scratch);
  }
  #[cfg(all(
    target_arch = "aarch64",
    target_vendor = "apple",
    not(feature = "portable-only"),
    not(miri)
  ))]
  if let Some(digits) = private_crt_pair_neon_shape(first.0, second.0, scratch) {
    return private_exponentiate_crt_pair_neon(first, second, digits, scratch);
  }
  let (modulus, exponent, input, out) = first;
  private_exponentiate_representative_with_scratch(modulus, exponent, input, out, scratch)?;
  let (modulus, exponent, input, out) = second;
  private_exponentiate_representative_with_scratch(modulus, exponent, input, out, scratch)
}

/// The IFMA kernel shape for a CRT pair, when both primes have one width that the kernel supports
/// and `scratch` holds the pair workspace.
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "sse2",
  not(feature = "portable-only"),
  not(miri)
))]
pub(super) fn private_crt_pair_ifma_shape(
  first: &RsaPublicModulus,
  second: &RsaPublicModulus,
  scratch: &RsaPrivateExponentScratch,
) -> Option<(usize, usize)> {
  if first.limbs.len() != second.limbs.len() || first.bytes.len() != second.bytes.len() {
    return None;
  }
  let shape = rsa_x86_64_ifma::shape(first.limbs.len())?;
  (scratch.ifma.as_slice().len() >= rsa_x86_64_ifma::workspace_words(first.limbs.len())).then_some(shape)
}

/// The CRT pair on the IFMA kernel. Both primes have `shape`'s limb count and byte width.
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "sse2",
  not(feature = "portable-only"),
  not(miri)
))]
fn private_exponentiate_crt_pair_ifma(
  first: RsaCrtExponentiation<'_>,
  second: RsaCrtExponentiation<'_>,
  shape: (usize, usize),
  scratch: &mut RsaPrivateExponentScratch,
) -> Result<(), RsaPrivateOpError> {
  crate::traits::ct::with_data_independent_timing(|| {
    let (modulus_p, exponent_p, input_p, out_p) = first;
    let (modulus_q, exponent_q, input_q, out_q) = second;
    let bytes = modulus_p.bytes.len();
    let limbs = modulus_p.limbs.len();
    if input_p.len() != bytes || out_p.len() != bytes || exponent_p.len() != bytes {
      return Err(RsaPrivateOpError::InvalidLength);
    }
    let r2_p = modulus_p
      .montgomery_r2()
      .ok_or(RsaPrivateOpError::RepresentativeOutOfRange)?;
    let RsaPrivateExponentWorkspace {
      representative: base_p,
      reduced: r2_ifma_p,
      acc: result_p,
      base: base_q,
      squared: r2_ifma_q,
      multiplied: result_q,
      ifma,
      ..
    } = scratch.workspace(limbs)?;
    limbs_from_be_into(input_p, base_p);
    if !ct_limbs_lt(base_p, &modulus_p.limbs) {
      return Err(RsaPrivateOpError::RepresentativeOutOfRange);
    }
    if input_q.len() != bytes || out_q.len() != bytes || exponent_q.len() != bytes {
      return Err(RsaPrivateOpError::InvalidLength);
    }
    let r2_q = modulus_q
      .montgomery_r2()
      .ok_or(RsaPrivateOpError::RepresentativeOutOfRange)?;
    limbs_from_be_into(input_q, base_q);
    if !ct_limbs_lt(base_q, &modulus_q.limbs) {
      return Err(RsaPrivateOpError::RepresentativeOutOfRange);
    }

    // R = 2^(52·digits) for the radix-2^52 kernel: R² mod m is the cached 2^(128·limbs) mod m
    // doubled 2·(52·digits - 64·limbs) times.
    let doublings = shape.0.strict_mul(52).strict_sub(limbs.strict_mul(64)).strict_mul(2);
    for (r2_ifma, r2, modulus) in [(&mut *r2_ifma_p, r2_p, modulus_p), (&mut *r2_ifma_q, r2_q, modulus_q)] {
      r2_ifma.copy_from_slice(r2);
      for _ in 0..doublings {
        double_mod_in_place(r2_ifma, &modulus.limbs);
      }
    }
    // SAFETY: `shape` returned Some for the shared limb count, so AVX-512F and AVX-512 IFMA are
    // present and the width has a kernel. Every limb slice has `limbs` limbs, both exponents have
    // `bytes` bytes, and the caller checked that the workspace has `workspace_words(limbs)` words.
    unsafe {
      rsa_x86_64_ifma::exponentiate_pair(
        [
          rsa_x86_64_ifma::Operands {
            out: &mut *result_p,
            base: base_p,
            exponent: exponent_p,
            m: &modulus_p.limbs,
            r2: r2_ifma_p,
            k0: modulus_p.n0 & (u64::MAX >> 12),
          },
          rsa_x86_64_ifma::Operands {
            out: &mut *result_q,
            base: base_q,
            exponent: exponent_q,
            m: &modulus_q.limbs,
            r2: r2_ifma_q,
            k0: modulus_q.n0 & (u64::MAX >> 12),
          },
        ],
        shape,
        ifma,
      );
    }
    limbs_to_be(result_p, out_p);
    limbs_to_be(result_q, out_q);
    Ok(())
  })
}

/// The paired NEON digit count for a CRT pair, when both primes have one width that the engine
/// supports and `scratch` holds the pair workspace.
#[cfg(all(
  target_arch = "aarch64",
  target_vendor = "apple",
  not(feature = "portable-only"),
  not(miri)
))]
pub(super) fn private_crt_pair_neon_shape(
  first: &RsaPublicModulus,
  second: &RsaPublicModulus,
  scratch: &RsaPrivateExponentScratch,
) -> Option<usize> {
  if first.limbs.len() != second.limbs.len() || first.bytes.len() != second.bytes.len() {
    return None;
  }
  let digits = rsa_aarch64_pair::shape(first.limbs.len())?;
  (scratch.pair.as_slice().len() >= rsa_aarch64_pair::workspace_words(first.limbs.len())).then_some(digits)
}

/// The CRT pair on the paired NEON engine. Both primes have the limb count and byte width that
/// gave `digits`.
#[cfg(all(
  target_arch = "aarch64",
  target_vendor = "apple",
  not(feature = "portable-only"),
  not(miri)
))]
fn private_exponentiate_crt_pair_neon(
  first: RsaCrtExponentiation<'_>,
  second: RsaCrtExponentiation<'_>,
  digits: usize,
  scratch: &mut RsaPrivateExponentScratch,
) -> Result<(), RsaPrivateOpError> {
  crate::traits::ct::with_data_independent_timing(|| {
    let (modulus_p, exponent_p, input_p, out_p) = first;
    let (modulus_q, exponent_q, input_q, out_q) = second;
    let bytes = modulus_p.bytes.len();
    let limbs = modulus_p.limbs.len();
    if input_p.len() != bytes || out_p.len() != bytes || exponent_p.len() != bytes {
      return Err(RsaPrivateOpError::InvalidLength);
    }
    let r2_p = modulus_p
      .montgomery_r2()
      .ok_or(RsaPrivateOpError::RepresentativeOutOfRange)?;
    let RsaPrivateExponentWorkspace {
      representative: base_p,
      reduced: r2_pair_p,
      acc: result_p,
      base: base_q,
      squared: r2_pair_q,
      multiplied: result_q,
      pair,
      ..
    } = scratch.workspace(limbs)?;
    limbs_from_be_into(input_p, base_p);
    if !ct_limbs_lt(base_p, &modulus_p.limbs) {
      return Err(RsaPrivateOpError::RepresentativeOutOfRange);
    }
    if input_q.len() != bytes || out_q.len() != bytes || exponent_q.len() != bytes {
      return Err(RsaPrivateOpError::InvalidLength);
    }
    let r2_q = modulus_q
      .montgomery_r2()
      .ok_or(RsaPrivateOpError::RepresentativeOutOfRange)?;
    limbs_from_be_into(input_q, base_q);
    if !ct_limbs_lt(base_q, &modulus_q.limbs) {
      return Err(RsaPrivateOpError::RepresentativeOutOfRange);
    }

    // R = 2^(28·digits) for the paired engine: R² mod m is the cached 2^(128·limbs) mod m doubled
    // 2·(28·digits - 64·limbs) times.
    let doublings = digits.strict_mul(28).strict_sub(limbs.strict_mul(64)).strict_mul(2);
    for (r2_pair, r2, modulus) in [(&mut *r2_pair_p, r2_p, modulus_p), (&mut *r2_pair_q, r2_q, modulus_q)] {
      r2_pair.copy_from_slice(r2);
      for _ in 0..doublings {
        double_mod_in_place(r2_pair, &modulus.limbs);
      }
    }
    rsa_aarch64_pair::exponentiate_pair(
      [
        rsa_aarch64_pair::Operands {
          out: &mut *result_p,
          base: base_p,
          exponent: exponent_p,
          m: &modulus_p.limbs,
          r2: r2_pair_p,
          n0: modulus_p.n0,
        },
        rsa_aarch64_pair::Operands {
          out: &mut *result_q,
          base: base_q,
          exponent: exponent_q,
          m: &modulus_q.limbs,
          r2: r2_pair_q,
          n0: modulus_q.n0,
        },
      ],
      digits,
      pair,
    );
    limbs_to_be(result_p, out_p);
    limbs_to_be(result_q, out_q);
    Ok(())
  })
}

pub(super) fn private_exponentiate_representative_with_scratch(
  modulus: &RsaPublicModulus,
  exponent: &[u8],
  input: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPrivateExponentScratch,
) -> Result<(), RsaPrivateOpError> {
  crate::traits::ct::with_data_independent_timing(|| {
    let bytes = modulus.bytes.len();
    let limbs = modulus.limbs.len();
    if input.len() != bytes || out.len() != bytes || exponent.len() != bytes {
      return Err(RsaPrivateOpError::InvalidLength);
    }
    let r2 = modulus
      .montgomery_r2()
      .ok_or(RsaPrivateOpError::RepresentativeOutOfRange)?;
    // A balanced CRT prime's 32-entry table fits the existing allocation. Check actual capacity
    // so direct exponentiation and wider components retain the 16-entry fallback when necessary.
    #[cfg(all(
      target_arch = "aarch64",
      target_os = "linux",
      not(feature = "portable-only"),
      not(miri)
    ))]
    let five_bit_supported = rsa_aarch64_mont::supports(limbs);
    #[cfg(not(all(
      target_arch = "aarch64",
      target_os = "linux",
      not(feature = "portable-only"),
      not(miri)
    )))]
    let five_bit_supported = cfg!(any(
      target_arch = "powerpc64",
      target_arch = "s390x",
      target_arch = "riscv64"
    ));
    let use_five_bit = five_bit_supported && scratch.table.as_slice().len() >= limbs.strict_mul(32);
    let RsaPrivateExponentWorkspace {
      t,
      representative,
      one,
      base,
      acc,
      squared,
      multiplied,
      selected,
      reduced,
      table,
      ifma: scratch_ifma,
      #[cfg(all(
        target_arch = "aarch64",
        target_vendor = "apple",
        not(feature = "portable-only"),
        not(miri)
      ))]
        pair: _,
    } = if use_five_bit {
      scratch.workspace_with_table_entries::<32>(limbs)?
    } else {
      scratch.workspace(limbs)?
    };
    #[cfg(not(all(
      target_arch = "x86_64",
      target_feature = "sse2",
      not(feature = "portable-only"),
      not(miri)
    )))]
    let _ = scratch_ifma;

    limbs_from_be_into(input, representative);
    if !ct_limbs_lt(representative, &modulus.limbs) {
      return Err(RsaPrivateOpError::RepresentativeOutOfRange);
    }

    #[cfg(all(
      target_arch = "x86_64",
      target_feature = "sse2",
      not(feature = "portable-only"),
      not(miri)
    ))]
    if let Some(shape) = rsa_x86_64_ifma::shape(limbs)
      && scratch_ifma.len() >= rsa_x86_64_ifma::workspace_words(limbs)
    {
      // R = 2^(52·digits) for the radix-2^52 kernel: R² mod m is the cached 2^(128·limbs) mod m
      // doubled 2·(52·digits - 64·limbs) times.
      reduced.copy_from_slice(r2);
      for _ in 0..shape.0.strict_mul(52).strict_sub(limbs.strict_mul(64)).strict_mul(2) {
        double_mod_in_place(reduced, &modulus.limbs);
      }
      // SAFETY: `shape` returned Some, so AVX-512F and AVX-512 IFMA are present and the limb count
      // has a kernel. Every slice has `limbs` limbs, the exponent has `bytes` bytes, and the
      // workspace has at least `workspace_words(limbs)` words.
      unsafe {
        rsa_x86_64_ifma::exponentiate(
          rsa_x86_64_ifma::Operands {
            out: &mut *acc,
            base: representative,
            exponent,
            m: &modulus.limbs,
            r2: reduced,
            k0: modulus.n0 & (u64::MAX >> 12),
          },
          shape,
          scratch_ifma,
        );
      }
      limbs_to_be(acc, out);
      return Ok(());
    }

    one.fill(0);
    one[0] = 1;
    private_mont_mul(base, representative, r2, modulus, t);
    private_mont_mul(acc, one, r2, modulus, t);

    if use_five_bit {
      private_fixed_window_table_into::<32>(table, base, acc, modulus, t);
      private_exponentiate_five_bit(
        table,
        exponent,
        modulus,
        RsaPrivateWindowWorkspace {
          t: &mut *t,
          acc: &mut *acc,
          squared: &mut *squared,
          multiplied: &mut *multiplied,
          selected: &mut *selected,
        },
      );
      private_mont_reduce(reduced, acc, modulus, t);
      limbs_to_be(reduced, out);
      return Ok(());
    }

    private_fixed_window_table_into::<PRIVATE_FIXED_WINDOW_TABLE_ENTRIES>(table, base, acc, modulus, t);
    let mut index = 0;
    while index < bytes {
      let exponent_byte = exponent[index];
      private_exponentiate_window(
        table,
        exponent_byte >> 4,
        modulus,
        RsaPrivateWindowWorkspace {
          t: &mut *t,
          acc: &mut *acc,
          squared: &mut *squared,
          multiplied: &mut *multiplied,
          selected: &mut *selected,
        },
      );
      private_exponentiate_window(
        table,
        exponent_byte & 0x0f,
        modulus,
        RsaPrivateWindowWorkspace {
          t: &mut *t,
          acc: &mut *acc,
          squared: &mut *squared,
          multiplied: &mut *multiplied,
          selected: &mut *selected,
        },
      );
      index = index.strict_add(1);
    }

    private_mont_reduce(reduced, acc, modulus, t);
    limbs_to_be(reduced, out);
    Ok(())
  })
}

fn private_fixed_window_table(
  base: &[u64],
  one_montgomery: &[u64],
  modulus: &RsaPublicModulus,
  t: &mut [u64],
) -> Vec<u64> {
  let limbs = base.len();
  let mut table = vec![0u64; PRIVATE_FIXED_WINDOW_TABLE_ENTRIES.strict_mul(limbs)];
  private_fixed_window_table_into::<PRIVATE_FIXED_WINDOW_TABLE_ENTRIES>(&mut table, base, one_montgomery, modulus, t);
  table
}

fn private_fixed_window_table_into<const ENTRIES: usize>(
  table: &mut [u64],
  base: &[u64],
  one_montgomery: &[u64],
  modulus: &RsaPublicModulus,
  t: &mut [u64],
) {
  let limbs = base.len();
  assert_eq!(table.len(), ENTRIES.strict_mul(limbs));
  table[..limbs].copy_from_slice(one_montgomery);
  table[limbs..limbs.strict_mul(2)].copy_from_slice(base);

  for index in 2..ENTRIES {
    let previous_start = index.strict_sub(1).strict_mul(limbs);
    let current_start = index.strict_mul(limbs);
    let (previous, current_and_rest) = table.split_at_mut(current_start);
    let previous = &previous[previous_start..previous_start.strict_add(limbs)];
    let current = &mut current_and_rest[..limbs];
    private_mont_mul(current, previous, base, modulus, t);
  }
}

fn private_exponentiate_window(
  table: &[u64],
  window: u8,
  modulus: &RsaPublicModulus,
  workspace: RsaPrivateWindowWorkspace<'_>,
) {
  let RsaPrivateWindowWorkspace {
    t,
    acc,
    squared,
    multiplied,
    selected,
  } = workspace;
  for _ in 0..4 {
    private_mont_square_in_place(acc, squared, modulus, t);
  }

  private_select_window_power(selected, table, window);
  private_mont_mul(multiplied, acc, selected, modulus, t);
  acc.copy_from_slice(multiplied);
}

/// Exponentiate with fixed five-bit windows, padding only the public leading partial window.
/// All 32 table entries are scanned on every step, including zero windows.
fn private_exponentiate_five_bit(
  table: &[u64],
  exponent: &[u8],
  modulus: &RsaPublicModulus,
  mut workspace: RsaPrivateWindowWorkspace<'_>,
) {
  let bits = exponent.len().strict_mul(8);
  let first_bits = (bits.strict_sub(1) % 5).strict_add(1);
  let first = exponent[0] >> 8usize.strict_sub(first_bits);
  private_exponentiate_window_five_bit(table, first, modulus, &mut workspace);
  let mut offset = first_bits;
  while offset < bits {
    let index = offset / 8;
    let pair = (u16::from(exponent[index]) << 8) | u16::from(exponent.get(index.strict_add(1)).copied().unwrap_or(0));
    let window = u8::try_from((pair >> 11usize.strict_sub(offset % 8)) & 31).expect("five-bit window fits u8");
    private_exponentiate_window_five_bit(table, window, modulus, &mut workspace);
    offset = offset.strict_add(5);
  }
}

fn private_exponentiate_window_five_bit(
  table: &[u64],
  window: u8,
  modulus: &RsaPublicModulus,
  workspace: &mut RsaPrivateWindowWorkspace<'_>,
) {
  let RsaPrivateWindowWorkspace {
    t,
    acc,
    squared,
    multiplied,
    selected,
  } = workspace;
  for _ in 0..5 {
    private_mont_square_in_place(acc, squared, modulus, t);
  }
  let (low, high) = table.split_at(selected.len().strict_mul(PRIVATE_FIXED_WINDOW_TABLE_ENTRIES));
  private_select_window_power(selected, low, window);
  // Modulo-256 subtraction maps low-half windows outside 0..16. Both selectors always run;
  // exactly one half contributes, and no secret value controls a branch or table address.
  private_select_window_power(multiplied, high, window.wrapping_sub(16));
  for (out, &upper) in selected.iter_mut().zip(multiplied.iter()) {
    *out |= upper;
  }
  private_mont_mul(multiplied, acc, selected, modulus, t);
  acc.copy_from_slice(multiplied);
}

#[inline(always)]
pub(super) fn private_select_window_power(out: &mut [u64], table: &[u64], window: u8) {
  let limbs = out.len();
  debug_assert_eq!(table.len(), limbs.strict_mul(PRIVATE_FIXED_WINDOW_TABLE_ENTRIES));
  #[cfg(all(target_arch = "aarch64", not(feature = "portable-only"), not(miri)))]
  let vector_limbs = {
    let vector_limbs = limbs & !3;
    if vector_limbs != 0 {
      let entries = &table[..PRIVATE_FIXED_WINDOW_TABLE_ENTRIES
        .strict_sub(1)
        .strict_mul(limbs)
        .strict_add(vector_limbs)];
      // SAFETY: `out` has `vector_limbs` writable limbs, a nonzero multiple of four. `entries`
      // ends after limb `vector_limbs - 1` of entry 15, so every read of four limbs at
      // `limbs·i + offset` for i in 0..16 and offset below `vector_limbs` stays inside it.
      unsafe {
        private_select_window_power_neon(
          out.as_mut_ptr(),
          entries.as_ptr(),
          limbs,
          vector_limbs.strict_div(4),
          u64::from(window).strict_mul(0x1_0000_0001),
        );
      }
    }
    vector_limbs
  };
  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(feature = "portable-only"),
    not(miri)
  ))]
  let vector_limbs = if crate::platform::caps().has(crate::platform::caps::x86::AVX2) {
    let vector_limbs = limbs & !7;
    if vector_limbs != 0 {
      // SAFETY: AVX2 is present. The complete table holds sixteen rows of limbs words, and
      // the output prefix fits a row. Distinct borrows keep table and output disjoint.
      unsafe { private_select_window_power_avx2(&mut out[..vector_limbs], table, window) };
    }
    vector_limbs
  } else {
    0
  };
  #[cfg(not(all(
    any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "sse2")),
    not(feature = "portable-only"),
    not(miri)
  )))]
  let vector_limbs = 0;
  if vector_limbs != limbs {
    private_select_window_power_portable(out, table, vector_limbs, window);
  }
}

/// Scan every entry, selecting eight limbs at a time through two vector accumulators.
///
/// # Safety
/// AVX2 must be present. `out.len()` must be a nonzero multiple of eight. The complete table
/// must contain sixteen equal-width rows; the output prefix must fit one row.
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "sse2",
  not(feature = "portable-only"),
  not(miri)
))]
#[target_feature(enable = "avx2")]
unsafe fn private_select_window_power_avx2(out: &mut [u64], table: &[u64], window: u8) {
  use core::arch::x86_64::*;

  let entry_limbs = table.len() / PRIVATE_FIXED_WINDOW_TABLE_ENTRIES;
  assert!(out.len() <= entry_limbs);
  let mut target = _mm256_set1_epi64x(i64::from(window));
  // SAFETY: AVX2 is enabled and this initialized register is an opaque identity. The empty
  // template changes no memory, stack or flags. Keep the compare in vector lanes; otherwise
  // LLVM expands the uniform target into a scalar boolean and several broadcasts per entry.
  unsafe { core::arch::asm!("/* {0} */", inout(ymm_reg) target, options(nomem, nostack, preserves_flags)) };
  for (chunk_index, chunk) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
    let offset = chunk_index.strict_mul(8);
    let mut low = _mm256_setzero_si256();
    let mut high = _mm256_setzero_si256();
    let mut entry_vector = _mm256_setzero_si256();
    let one = _mm256_set1_epi64x(1);
    let mut rows = table;
    // Keep the public scan count opaque so LLVM retains the loop instead of unrolling it
    // and spilling its masks and table limbs. The register barriers below protect the loads.
    for _ in 0..core::hint::black_box(PRIVATE_FIXED_WINDOW_TABLE_ENTRIES) {
      let (row, remaining) = rows.split_at(entry_limbs);
      rows = remaining;
      // SAFETY: The caller provides eight initialized table limbs at this public entry and
      // chunk offset. Both four-limb loads accept unaligned addresses. The empty register
      // barrier touches no memory, stack or flags, and this function enables AVX2. It keeps
      // the loads unconditional instead of letting LLVM fold them into secret-masked reads.
      let (mut lower, mut upper) = unsafe {
        let pointer = row.as_ptr().add(offset);
        (
          _mm256_loadu_si256(pointer.cast()),
          _mm256_loadu_si256(pointer.add(4).cast()),
        )
      };
      // SAFETY: These initialized vector registers are opaque identities; no other state changes.
      unsafe {
        core::arch::asm!("/* {0} {1} {2} */", inout(ymm_reg) lower, inout(ymm_reg) upper, inout(ymm_reg) entry_vector,
          options(nomem, nostack, preserves_flags));
      }
      let mask = _mm256_cmpeq_epi64(target, entry_vector);
      entry_vector = _mm256_add_epi64(entry_vector, one);
      low = _mm256_or_si256(low, _mm256_and_si256(lower, mask));
      high = _mm256_or_si256(high, _mm256_and_si256(upper, mask));
    }
    // SAFETY: chunk contains eight writable limbs, so both four-limb unaligned stores fit.
    unsafe {
      _mm256_storeu_si256(chunk.as_mut_ptr().cast(), low);
      _mm256_storeu_si256(chunk.as_mut_ptr().add(4).cast(), high);
    }
  }
}

/// Select the first `4·chunks` limbs of the table entries named by `windows` into `out`. The low
/// 32 bits of every limb come from entry `windows mod 2^32` and the high 32 bits from entry
/// `windows >> 32`; the single-modulus path repeats one window in both halves to select whole
/// limbs, and the paired NEON engine keeps the two CRT primes in the two halves. The sixteen
/// entry masks come from 32-bit lane compares; each four-limb chunk then reads the same four limbs
/// of all sixteen entries in a fixed order and merges them under those masks into four
/// accumulators. The secret windows decide only mask bits, never an address or a branch.
///
/// # Safety
///
/// `chunks` must be nonzero, `out` valid for writing `4·chunks` limbs, and `table` valid for
/// reading `4·chunks` limbs at `entry_limbs·i` for every `i` in `0..16`.
#[cfg(all(target_arch = "aarch64", not(miri), any(not(feature = "portable-only"), test)))]
#[inline(always)]
pub(super) unsafe fn private_select_window_power_neon(
  out: *mut u64,
  table: *const u64,
  entry_limbs: usize,
  chunks: usize,
  windows: u64,
) {
  // SAFETY: The caller guarantees a nonzero chunk count, the writes at `out`, and the reads at byte
  // stride `entry_limbs·8` from `table`; the block touches no other memory and no stack. LDP and
  // STP of Q registers need only the eight-byte alignment of `u64`. NEON is part of the A64
  // baseline that every AArch64 Rust target enables.
  unsafe {
    core::arch::asm!(
      "dup {w:v}.2d, {windows}",
      "movi {index:v}.2d, #0",
      "mov {tmp:w}, #1",
      "dup {one:v}.4s, {tmp:w}",
      "cmeq {m0:v}.4s, {index:v}.4s, {w:v}.4s",
      "add {index:v}.4s, {index:v}.4s, {one:v}.4s",
      "cmeq {m1:v}.4s, {index:v}.4s, {w:v}.4s",
      "add {index:v}.4s, {index:v}.4s, {one:v}.4s",
      "cmeq {m2:v}.4s, {index:v}.4s, {w:v}.4s",
      "add {index:v}.4s, {index:v}.4s, {one:v}.4s",
      "cmeq {m3:v}.4s, {index:v}.4s, {w:v}.4s",
      "add {index:v}.4s, {index:v}.4s, {one:v}.4s",
      "cmeq {m4:v}.4s, {index:v}.4s, {w:v}.4s",
      "add {index:v}.4s, {index:v}.4s, {one:v}.4s",
      "cmeq {m5:v}.4s, {index:v}.4s, {w:v}.4s",
      "add {index:v}.4s, {index:v}.4s, {one:v}.4s",
      "cmeq {m6:v}.4s, {index:v}.4s, {w:v}.4s",
      "add {index:v}.4s, {index:v}.4s, {one:v}.4s",
      "cmeq {m7:v}.4s, {index:v}.4s, {w:v}.4s",
      "add {index:v}.4s, {index:v}.4s, {one:v}.4s",
      "cmeq {m8:v}.4s, {index:v}.4s, {w:v}.4s",
      "add {index:v}.4s, {index:v}.4s, {one:v}.4s",
      "cmeq {m9:v}.4s, {index:v}.4s, {w:v}.4s",
      "add {index:v}.4s, {index:v}.4s, {one:v}.4s",
      "cmeq {m10:v}.4s, {index:v}.4s, {w:v}.4s",
      "add {index:v}.4s, {index:v}.4s, {one:v}.4s",
      "cmeq {m11:v}.4s, {index:v}.4s, {w:v}.4s",
      "add {index:v}.4s, {index:v}.4s, {one:v}.4s",
      "cmeq {m12:v}.4s, {index:v}.4s, {w:v}.4s",
      "add {index:v}.4s, {index:v}.4s, {one:v}.4s",
      "cmeq {m13:v}.4s, {index:v}.4s, {w:v}.4s",
      "add {index:v}.4s, {index:v}.4s, {one:v}.4s",
      "cmeq {m14:v}.4s, {index:v}.4s, {w:v}.4s",
      "add {index:v}.4s, {index:v}.4s, {one:v}.4s",
      "cmeq {m15:v}.4s, {index:v}.4s, {w:v}.4s",
      "2:",
      "mov {entry}, {table}",
      "ldp {e0:q}, {e1:q}, [{entry}]",
      "add {entry}, {entry}, {stride}",
      "and {lo0:v}.16b, {e0:v}.16b, {m0:v}.16b",
      "and {hi0:v}.16b, {e1:v}.16b, {m0:v}.16b",
      "ldp {e0:q}, {e1:q}, [{entry}]",
      "add {entry}, {entry}, {stride}",
      "and {lo1:v}.16b, {e0:v}.16b, {m1:v}.16b",
      "and {hi1:v}.16b, {e1:v}.16b, {m1:v}.16b",
      "ldp {e0:q}, {e1:q}, [{entry}]",
      "add {entry}, {entry}, {stride}",
      "and {lo2:v}.16b, {e0:v}.16b, {m2:v}.16b",
      "and {hi2:v}.16b, {e1:v}.16b, {m2:v}.16b",
      "ldp {e0:q}, {e1:q}, [{entry}]",
      "add {entry}, {entry}, {stride}",
      "and {lo3:v}.16b, {e0:v}.16b, {m3:v}.16b",
      "and {hi3:v}.16b, {e1:v}.16b, {m3:v}.16b",
      "ldp {e0:q}, {e1:q}, [{entry}]",
      "add {entry}, {entry}, {stride}",
      "bit {lo0:v}.16b, {e0:v}.16b, {m4:v}.16b",
      "bit {hi0:v}.16b, {e1:v}.16b, {m4:v}.16b",
      "ldp {e0:q}, {e1:q}, [{entry}]",
      "add {entry}, {entry}, {stride}",
      "bit {lo1:v}.16b, {e0:v}.16b, {m5:v}.16b",
      "bit {hi1:v}.16b, {e1:v}.16b, {m5:v}.16b",
      "ldp {e0:q}, {e1:q}, [{entry}]",
      "add {entry}, {entry}, {stride}",
      "bit {lo2:v}.16b, {e0:v}.16b, {m6:v}.16b",
      "bit {hi2:v}.16b, {e1:v}.16b, {m6:v}.16b",
      "ldp {e0:q}, {e1:q}, [{entry}]",
      "add {entry}, {entry}, {stride}",
      "bit {lo3:v}.16b, {e0:v}.16b, {m7:v}.16b",
      "bit {hi3:v}.16b, {e1:v}.16b, {m7:v}.16b",
      "ldp {e0:q}, {e1:q}, [{entry}]",
      "add {entry}, {entry}, {stride}",
      "bit {lo0:v}.16b, {e0:v}.16b, {m8:v}.16b",
      "bit {hi0:v}.16b, {e1:v}.16b, {m8:v}.16b",
      "ldp {e0:q}, {e1:q}, [{entry}]",
      "add {entry}, {entry}, {stride}",
      "bit {lo1:v}.16b, {e0:v}.16b, {m9:v}.16b",
      "bit {hi1:v}.16b, {e1:v}.16b, {m9:v}.16b",
      "ldp {e0:q}, {e1:q}, [{entry}]",
      "add {entry}, {entry}, {stride}",
      "bit {lo2:v}.16b, {e0:v}.16b, {m10:v}.16b",
      "bit {hi2:v}.16b, {e1:v}.16b, {m10:v}.16b",
      "ldp {e0:q}, {e1:q}, [{entry}]",
      "add {entry}, {entry}, {stride}",
      "bit {lo3:v}.16b, {e0:v}.16b, {m11:v}.16b",
      "bit {hi3:v}.16b, {e1:v}.16b, {m11:v}.16b",
      "ldp {e0:q}, {e1:q}, [{entry}]",
      "add {entry}, {entry}, {stride}",
      "bit {lo0:v}.16b, {e0:v}.16b, {m12:v}.16b",
      "bit {hi0:v}.16b, {e1:v}.16b, {m12:v}.16b",
      "ldp {e0:q}, {e1:q}, [{entry}]",
      "add {entry}, {entry}, {stride}",
      "bit {lo1:v}.16b, {e0:v}.16b, {m13:v}.16b",
      "bit {hi1:v}.16b, {e1:v}.16b, {m13:v}.16b",
      "ldp {e0:q}, {e1:q}, [{entry}]",
      "add {entry}, {entry}, {stride}",
      "bit {lo2:v}.16b, {e0:v}.16b, {m14:v}.16b",
      "bit {hi2:v}.16b, {e1:v}.16b, {m14:v}.16b",
      "ldp {e0:q}, {e1:q}, [{entry}]",
      "bit {lo3:v}.16b, {e0:v}.16b, {m15:v}.16b",
      "bit {hi3:v}.16b, {e1:v}.16b, {m15:v}.16b",
      "orr {lo0:v}.16b, {lo0:v}.16b, {lo1:v}.16b",
      "orr {lo2:v}.16b, {lo2:v}.16b, {lo3:v}.16b",
      "orr {hi0:v}.16b, {hi0:v}.16b, {hi1:v}.16b",
      "orr {hi2:v}.16b, {hi2:v}.16b, {hi3:v}.16b",
      "orr {lo0:v}.16b, {lo0:v}.16b, {lo2:v}.16b",
      "orr {hi0:v}.16b, {hi0:v}.16b, {hi2:v}.16b",
      "stp {lo0:q}, {hi0:q}, [{out}], #32",
      "add {table}, {table}, #32",
      "subs {chunks}, {chunks}, #1",
      "b.ne 2b",
      windows = in(reg) windows,
      table = inout(reg) table => _,
      stride = in(reg) entry_limbs.strict_mul(8),
      out = inout(reg) out => _,
      chunks = inout(reg) chunks => _,
      entry = out(reg) _,
      tmp = out(reg) _,
      w = out(vreg) _, index = out(vreg) _, one = out(vreg) _,
      m0 = out(vreg) _, m1 = out(vreg) _, m2 = out(vreg) _, m3 = out(vreg) _,
      m4 = out(vreg) _, m5 = out(vreg) _, m6 = out(vreg) _, m7 = out(vreg) _,
      m8 = out(vreg) _, m9 = out(vreg) _, m10 = out(vreg) _, m11 = out(vreg) _,
      m12 = out(vreg) _, m13 = out(vreg) _, m14 = out(vreg) _, m15 = out(vreg) _,
      e0 = out(vreg) _, e1 = out(vreg) _,
      lo0 = out(vreg) _, lo1 = out(vreg) _, lo2 = out(vreg) _, lo3 = out(vreg) _,
      hi0 = out(vreg) _, hi1 = out(vreg) _, hi2 = out(vreg) _, hi3 = out(vreg) _,
      options(nostack),
    );
  }
}

/// Select limbs `start..` of table entry `window` into `out` with masked scalar merges.
#[inline(always)]
fn private_select_window_power_portable(out: &mut [u64], table: &[u64], start: usize, window: u8) {
  let limbs = out.len();
  let out = &mut out[start..];
  out.fill(0);
  let entry_count =
    u8::try_from(PRIVATE_FIXED_WINDOW_TABLE_ENTRIES).expect("private fixed-window table entry count must fit u8");
  for index in 0..entry_count {
    let entry_start = usize::from(index).strict_mul(limbs);
    let entry = &table[entry_start.strict_add(start)..entry_start.strict_add(limbs)];
    let mask = core::hint::black_box(private_choice_eq_mask_u8(window, index));
    for (dst, limb) in out.iter_mut().zip(entry) {
      // SAFETY: Volatile table read is used as a compiler barrier because:
      // 1. `limb` is a valid shared reference to one fixed-window table limb.
      // 2. `u64` is `Copy`, so the volatile read does not create ownership aliasing.
      // 3. Every table limb is read unconditionally; the secret window affects only masks.
      let limb = unsafe { core::ptr::read_volatile(limb) };
      // The output starts at zero and at most one entry matches, so no earlier value needs replacing.
      *dst |= limb & mask;
    }
  }
}

fn private_choice_eq_mask_u8(left: u8, right: u8) -> u64 {
  let diff = u64::from(left ^ right);
  let is_zero = ((diff | diff.wrapping_neg()) >> 63) ^ 1;
  0u64.wrapping_sub(is_zero)
}

#[cfg(test)]
mod tests;

#[cfg(all(test, not(miri), feature = "portable-only", feature = "rsa"))]
mod binsec {
  use super::*;
  use crate::binsec::*;
  use core::ptr;

  #[unsafe(no_mangle)]
  #[inline(never)]
  pub(super) extern "C" fn ct_binsec_rsa_private_select_window_power_4() -> ! {
    // SAFETY: This pointer references a fixed harness global with static storage.
    let table = unsafe { array_from_global(&raw const CT_BINSEC_RSA_WINDOW_TABLE) };
    // SAFETY: This pointer references a fixed harness global with static storage.
    let window = unsafe { ptr::read_volatile(ptr::addr_of!(CT_BINSEC_RSA_WINDOW)) };
    let mut limbs = [0u64; 4];
    private_select_window_power(&mut limbs, &table, window);

    let mut acc = 0u64;
    for limb in limbs {
      acc ^= limb;
    }
    let folded = acc | (acc >> 8) | (acc >> 16) | (acc >> 24) | (acc >> 32) | (acc >> 40) | (acc >> 48) | (acc >> 56);
    ct_binsec_done(folded.to_le_bytes()[0])
  }

  #[cfg(target_arch = "aarch64")]
  #[unsafe(no_mangle)]
  #[inline(never)]
  pub(super) extern "C" fn ct_binsec_rsa_private_select_window_power_4_neon() -> ! {
    // SAFETY: This pointer references a fixed harness global with static storage.
    let table = unsafe { array_from_global(&raw const CT_BINSEC_RSA_WINDOW_TABLE) };
    // SAFETY: This pointer references a fixed harness global with static storage.
    let window = unsafe { ptr::read_volatile(ptr::addr_of!(CT_BINSEC_RSA_WINDOW)) };
    let mut limbs = [0u64; 4];
    // SAFETY: Sixteen four-limb table entries cover each load. The output holds four limbs.
    unsafe {
      private_select_window_power_neon(
        limbs.as_mut_ptr(),
        table.as_ptr(),
        4,
        1,
        u64::from(window).strict_mul(0x1_0000_0001),
      )
    };

    let mut acc = 0u64;
    for limb in limbs {
      acc ^= limb;
    }
    let folded = acc | (acc >> 8) | (acc >> 16) | (acc >> 24) | (acc >> 32) | (acc >> 40) | (acc >> 48) | (acc >> 56);
    ct_binsec_done(folded.to_le_bytes()[0])
  }
}
