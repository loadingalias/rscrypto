use super::*;

/// Apply the production RSA private fixed-window exponentiation loop with reusable scratch.
///
/// This diagnostic-only primitive uses the first precomputed CRT component
/// modulus from `key`, matching the steady-state production path without
/// timing component-modulus construction or heap allocation. It is raw
/// modular arithmetic without padding or blinding and must not be used as a
/// cryptographic protocol API.
///
/// # Errors
///
/// Returns [`RsaPrivateOpError`] if `scratch` has another public modulus width,
/// if `exponent`, `input`, or `out` is not exactly the first CRT component
/// width, or if `input` is outside that component modulus.
pub(crate) fn diag_rsa_private_exponentiate_fixed_width_with_scratch(
  key: &RsaPrivateKey,
  exponent: &[u8],
  input: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPrivateScratch,
) -> Result<(), RsaPrivateOpError> {
  let result = scratch.ensure_len(key.signature_len()).and_then(|()| {
    private_exponentiate_representative_with_scratch(
      &key.components.prime_p_modulus,
      exponent,
      input,
      out,
      &mut scratch.exponent_scratch,
    )
  });
  clear_output_on_error(result, out)
}

/// Apply the production RSA CRT exponentiation pair with reusable scratch.
///
/// This diagnostic-only primitive raises `inputs[0]` and `inputs[1]` to `exponents[0]` and
/// `exponents[1]` modulo the key's two CRT primes, through the same dispatch as private
/// operations: on x86-64 with AVX-512F and AVX-512 IFMA, equal-width primes run the interleaved
/// IFMA kernel, and otherwise each prime runs the fixed-window loop in turn. It is raw modular
/// arithmetic without padding, blinding, or recombination and must not be used as a
/// cryptographic protocol API. Both outputs are cleared on error.
///
/// # Errors
///
/// Returns [`RsaPrivateOpError`] if `scratch` has another public modulus width, if an exponent,
/// input, or output is not exactly its prime's width, or if an input is outside its prime.
pub(crate) fn diag_rsa_private_exponentiate_crt_pair_with_scratch(
  key: &RsaPrivateKey,
  exponents: [&[u8]; 2],
  inputs: [&[u8]; 2],
  outs: [&mut [u8]; 2],
  scratch: &mut RsaPrivateScratch,
) -> Result<(), RsaPrivateOpError> {
  let [out_p, out_q] = outs;
  let result = scratch.ensure_len(key.signature_len()).and_then(|()| {
    private_exponentiate_crt_with_scratch(
      (&key.components.prime_p_modulus, exponents[0], inputs[0], &mut *out_p),
      (&key.components.prime_q_modulus, exponents[1], inputs[1], &mut *out_q),
      &mut scratch.exponent_scratch,
    )
  });
  if result.is_err() {
    ct::zeroize(out_p);
    ct::zeroize(out_q);
  }
  result
}

/// Report whether private operations with `key` and `scratch` run the CRT pair on the AVX-512
/// IFMA kernel on this host.
#[must_use]
pub(crate) fn diag_rsa_private_crt_pair_uses_ifma(key: &RsaPrivateKey, scratch: &RsaPrivateScratch) -> bool {
  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(feature = "portable-only"),
    not(miri)
  ))]
  let uses_ifma = private_crt_pair_ifma_shape(
    &key.components.prime_p_modulus,
    &key.components.prime_q_modulus,
    &scratch.exponent_scratch,
  )
  .is_some();
  #[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(feature = "portable-only"),
    not(miri)
  )))]
  let uses_ifma = {
    let _ = (key, scratch);
    false
  };
  uses_ifma
}

/// Derive a blinding-factor inverse for RSA side-channel diagnostics.
///
/// The production masked inverse draws its inverse mask from OS randomness.
/// blinding material internally.
#[cfg(feature = "getrandom")]
pub(crate) fn diag_rsa_blinding_factor_inverse(
  key: &RsaPrivateKey,
  factor: &[u8],
  out: &mut [u8],
) -> Result<(), RsaPrivateOpError> {
  key
    .components
    .blinding_factor_inverse_with(factor, out, getrandom::fill)
}

/// Derive a blinding-factor inverse with reusable scratch for side-channel diagnostics.
///
/// This diagnostic-only leaf exercises the production masked modular inverse without
/// per-call allocation; `mask` stands in for the inverse mask that production draws
/// from its entropy source. Normal callers should use the signing and decryption
/// APIs, which generate and clear blinding material internally.
#[unsafe(no_mangle)]
#[inline(never)]
pub(crate) fn diag_rsa_blinding_factor_inverse_with_scratch(
  key: &RsaPrivateKey,
  factor: &[u8],
  mask: &[u8],
  out: &mut [u8],
  scratch: &mut RsaPrivateScratch,
) -> Result<(), RsaPrivateOpError> {
  let len = key.signature_len();
  let result = scratch.ensure_len(len).and_then(|()| {
    if factor.len() != len || mask.len() != len || out.len() != len {
      return Err(RsaPrivateOpError::InvalidLength);
    }
    scratch.blinding_factor.as_mut_slice().copy_from_slice(factor);
    key
      .components
      .blinding_factor_inverse_into_scratch(scratch, |random: &mut [u8]| {
        random.copy_from_slice(mask);
        Ok::<(), core::convert::Infallible>(())
      })?;
    out.copy_from_slice(scratch.blinding_inverse.as_slice());
    Ok(())
  });
  scratch.clear();
  clear_output_on_error(result, out)
}

#[inline(always)]
/// Run the eight-limb RSA private selector, including its native AVX2 path when available.
pub(crate) fn diag_rsa_private_select_window_power_8(table: &[u64; 128], window: u8) -> [u64; 8] {
  let mut out = [0u64; 8];
  private_select_window_power(&mut out, table, window);
  out
}

/// Report whether private operations with `key` and `scratch` run the CRT pair on the paired
/// NEON engine on this host.
#[must_use]
pub(crate) fn diag_rsa_private_crt_pair_uses_neon_pair(key: &RsaPrivateKey, scratch: &RsaPrivateScratch) -> bool {
  #[cfg(all(
    target_arch = "aarch64",
    target_vendor = "apple",
    not(feature = "portable-only"),
    not(miri)
  ))]
  let uses_pair = private_crt_pair_neon_shape(
    &key.components.prime_p_modulus,
    &key.components.prime_q_modulus,
    &scratch.exponent_scratch,
  )
  .is_some();
  #[cfg(not(all(
    target_arch = "aarch64",
    target_vendor = "apple",
    not(feature = "portable-only"),
    not(miri)
  )))]
  let uses_pair = {
    let _ = (key, scratch);
    false
  };
  uses_pair
}

pub(crate) use super::component_test::check as diag_rsa_private_component_validation_32;

#[test]

fn diagnostic_private_exponentiation_matches_independent_fixed_width_reference() {
  let len = 128;
  let modulus = vec![0xa5; len];
  let mut input = vec![0u8; len];
  *input.last_mut().expect("an RSA representative must be nonempty") = 2;

  let mut leading_zero_exponent = vec![0x5a; len];
  leading_zero_exponent[0] = 0;
  let mut full_width_exponent = leading_zero_exponent.clone();
  full_width_exponent[0] = 1;

  for exponent in [full_width_exponent, leading_zero_exponent] {
    let mut out = vec![0u8; len];
    private_exponentiate_representative(
      &private_component_modulus(&modulus).expect("fixture has an odd modulus"),
      &exponent,
      &input,
      &mut out,
    )
    .expect("fixed-width diagnostic exponentiation must accept the valid representative");
    let reference =
      BigUint::from_bytes_be(&input).modpow(&BigUint::from_bytes_be(&exponent), &BigUint::from_bytes_be(&modulus));
    assert_eq!(BigUint::from_bytes_be(&out), reference);
  }

  use num_bigint::BigUint;
  use rsa::{pkcs1::DecodeRsaPrivateKey as _, pkcs8::EncodePrivateKey as _, traits::PrivateKeyParts as _};
  let rustcrypto_key =
    rsa::PrivateKey::from_pkcs1_der(include_bytes!("../../../tests/vectors/rsa/rsa2048_private_pkcs1.der"))
      .expect("RSA-2048 PKCS #1 fixture");
  let component_modulus = BigUint::from_bytes_be(&rustcrypto_key.primes()[0].to_be_bytes());
  let private_der = rustcrypto_key
    .to_pkcs8_der()
    .expect("the fixed private key must encode as PKCS#8");
  let key =
    RsaPrivateKey::from_pkcs8_der_with_policy(private_der.as_bytes(), &RsaPublicKeyPolicy::modulus_2048_to_8192())
      .expect("the fixed private key must import for reusable-scratch diagnostics");
  let component_len = component_modulus.to_bytes_be().len();
  let mut component_input = vec![0u8; component_len];
  *component_input
    .last_mut()
    .expect("an RSA component representative must be nonempty") = 2;
  let mut component_exponent = vec![0x5a; component_len];
  component_exponent[0] = 1;
  let mut component_out = vec![0u8; component_len];
  let mut scratch = key.private_scratch();

  diag_rsa_private_exponentiate_fixed_width_with_scratch(
    &key,
    &component_exponent,
    &component_input,
    &mut component_out,
    &mut scratch,
  )
  .expect("reusable-scratch diagnostic exponentiation must accept the valid representative");
  let reference =
    BigUint::from_bytes_be(&component_input).modpow(&BigUint::from_bytes_be(&component_exponent), &component_modulus);
  assert_eq!(BigUint::from_bytes_be(&component_out), reference);
}

#[test]
fn private_crt_pair_diagnostic_matches_independent_modpow() {
  use num_bigint::BigUint;
  use rsa::{pkcs1::DecodeRsaPrivateKey as _, pkcs8::EncodePrivateKey as _, traits::PrivateKeyParts as _};
  let rustcrypto_key =
    rsa::PrivateKey::from_pkcs1_der(include_bytes!("../../../tests/vectors/rsa/rsa2048_private_pkcs1.der"))
      .expect("RSA-2048 PKCS #1 fixture");
  let primes = [
    BigUint::from_bytes_be(&rustcrypto_key.primes()[0].to_be_bytes()),
    BigUint::from_bytes_be(&rustcrypto_key.primes()[1].to_be_bytes()),
  ];
  let private_der = rustcrypto_key
    .to_pkcs8_der()
    .expect("the fixed private key must encode as PKCS#8");
  let key =
    RsaPrivateKey::from_pkcs8_der_with_policy(private_der.as_bytes(), &RsaPublicKeyPolicy::modulus_2048_to_8192())
      .expect("the fixed private key must import for CRT pair diagnostics");
  let len = primes[0].to_bytes_be().len();
  assert_eq!(primes[1].to_bytes_be().len(), len, "the fixture primes share one width");
  let mut scratch = key.private_scratch();
  // Production dispatch runs these cases on the interleaved IFMA pair kernel exactly when the host
  // has AVX-512F and AVX-512 IFMA.
  #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
  let ifma_host = std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512ifma");
  #[cfg(not(all(target_arch = "x86_64", not(feature = "portable-only"))))]
  let ifma_host = false;
  assert_eq!(diag_rsa_private_crt_pair_uses_ifma(&key, &scratch), ifma_host);
  // And on the paired NEON engine on every Apple AArch64 host.
  let apple_host = cfg!(all(
    target_arch = "aarch64",
    target_vendor = "apple",
    not(feature = "portable-only"),
    not(miri)
  ));
  assert_eq!(diag_rsa_private_crt_pair_uses_neon_pair(&key, &scratch), apple_host);
  let pattern = |seed: usize| -> Vec<u8> {
    (0..len)
      .map(|index| u8::try_from(index.strict_mul(31).strict_add(seed.strict_mul(97)) % 251).expect("below 251"))
      .collect()
  };
  for seed in 0..4 {
    let exponents = [pattern(seed), pattern(seed.strict_add(5))];
    // Clearing the top two bits keeps each input below its 1024-bit prime.
    let inputs = [pattern(seed.strict_add(9)), pattern(seed.strict_add(13))].map(|mut input| {
      input[0] &= 0x3f;
      input
    });
    let mut outputs = [vec![0u8; len], vec![0u8; len]];
    let [out_p, out_q] = &mut outputs;
    diag_rsa_private_exponentiate_crt_pair_with_scratch(
      &key,
      [&exponents[0], &exponents[1]],
      [&inputs[0], &inputs[1]],
      [out_p.as_mut_slice(), out_q.as_mut_slice()],
      &mut scratch,
    )
    .expect("the CRT pair must accept inputs below both primes");
    for index in 0..2 {
      let expected =
        BigUint::from_bytes_be(&inputs[index]).modpow(&BigUint::from_bytes_be(&exponents[index]), &primes[index]);
      assert_eq!(
        BigUint::from_bytes_be(&outputs[index]),
        expected,
        "prime {index}, seed {seed}"
      );
    }
  }

  // An input equal to the second prime is rejected after the first prime's checks, and both
  // outputs are cleared.
  let exponent = pattern(1);
  let mut outputs = [vec![0xa5u8; len], vec![0xa5u8; len]];
  let [out_p, out_q] = &mut outputs;
  assert_eq!(
    diag_rsa_private_exponentiate_crt_pair_with_scratch(
      &key,
      [&exponent, &exponent],
      [&pattern(2), &primes[1].to_bytes_be()],
      [out_p.as_mut_slice(), out_q.as_mut_slice()],
      &mut scratch,
    ),
    Err(RsaPrivateOpError::RepresentativeOutOfRange)
  );
  assert!(outputs.iter().flatten().all(|&byte| byte == 0));
}

#[test]
fn private_component_validation_leaf_requires_every_check() {
  let mut upper = [0u8; 32];
  upper[0] = 0x80;
  let other = [0x22u8; 32];
  let mut component = [0x11u8; 32];
  component[31] = 0x13;
  assert_eq!(super::component_test::check(&component, &upper, &other), 1);

  // Each failed check alone rejects the component.
  let mut leading_zero = component;
  leading_zero[0] = 0;
  assert_eq!(super::component_test::check(&leading_zero, &upper, &other), 0);
  let mut even = component;
  even[31] = 0x12;
  assert_eq!(super::component_test::check(&even, &upper, &other), 0);
  assert_eq!(super::component_test::check(&component, &upper, &component), 0);

  // The bound comparison is strict, and the first differing byte decides it.
  let mut at_bound = upper;
  at_bound[31] = 1;
  let mut bound = at_bound;
  assert_eq!(super::component_test::check(&at_bound, &bound, &other), 0);
  bound[31] = 3;
  assert_eq!(super::component_test::check(&at_bound, &bound, &other), 1);
  let mut low_first_byte = [0xffu8; 32];
  low_first_byte[0] = 0x7f;
  assert_eq!(super::component_test::check(&low_first_byte, &upper, &other), 1);
  let mut high_first_byte = [0u8; 32];
  high_first_byte[0] = 0x80;
  high_first_byte[31] = 1;
  bound = upper;
  bound[1] = 0xff;
  assert_eq!(super::component_test::check(&high_first_byte, &bound, &other), 1);
  high_first_byte[1] = 0xff;
  high_first_byte[2] = 1;
  assert_eq!(super::component_test::check(&high_first_byte, &bound, &other), 0);
}
