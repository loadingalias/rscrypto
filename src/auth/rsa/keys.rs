//! Key import, DER encoding, key generation, and validation.

use super::*;

impl RsaPrivateKey {
  /// Generate a new RSA private key with public exponent `65537`.
  ///
  /// New key material uses the default policy: 3072 through 8192 modulus
  /// bits. Use imported legacy RSA-2048 keys only for compatibility.
  ///
  /// Generation follows [`Self::GENERATION_CONTRACT`]: FIPS 186-5 Appendix
  /// A.1.3 probable-prime RSA generation in code. This is not a CMVP/FIPS 140-3
  /// validation claim.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyGenerationError::InvalidModulusBits`] unless
  /// `modulus_bits` is even and in `3072..=8192`. Returns another
  /// [`RsaKeyGenerationError`] if entropy is unavailable or bounded prime
  /// search fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  pub fn generate(modulus_bits: usize) -> Result<Self, RsaKeyGenerationError> {
    Self::generate_with_policy(modulus_bits, &RsaPublicKeyPolicy::modulus_3072_to_8192())
  }

  /// Generate a new RSA private key with an explicit modulus policy.
  ///
  /// The generated exponent is always `65537`; the policy only controls
  /// accepted modulus size. This is mainly for tests and deployments that must
  /// mint RSA-2048 compatibility keys.
  ///
  /// Generation follows [`Self::GENERATION_CONTRACT`]: FIPS 186-5 Appendix
  /// A.1.3 probable-prime RSA generation in code. This is not a CMVP/FIPS 140-3
  /// validation claim.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyGenerationError`] if the requested size is outside policy,
  /// entropy is unavailable, or bounded prime search fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  pub fn generate_with_policy(modulus_bits: usize, policy: &RsaPublicKeyPolicy) -> Result<Self, RsaKeyGenerationError> {
    generate_rsa_private_key(modulus_bits, policy).map(|components| Self { components })
  }

  /// Parse a PKCS #1 `RSAPrivateKey` DER object with the default policy.
  ///
  /// The default policy accepts RSA-3072 through RSA-8192 with exponent
  /// `65537`. Use [`RsaPublicKeyPolicy::modulus_2048_to_8192`] with
  /// [`Self::from_pkcs1_der_with_policy`] only for deployed RSA-2048
  /// compatibility material.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the DER is malformed or the key components are
  /// inconsistent with RSA private-key policy, including factor width or
  /// probable-prime validation.
  pub fn from_pkcs1_der(der: &[u8]) -> Result<Self, RsaKeyError> {
    Self::from_pkcs1_der_with_policy(der, &RsaPublicKeyPolicy::default())
  }

  /// Parse a PKCS #1 `RSAPrivateKey` DER object with an explicit public-key policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if parsing, factor-strength validation, or
  /// component consistency validation fails.
  pub fn from_pkcs1_der_with_policy(der: &[u8], policy: &RsaPublicKeyPolicy) -> Result<Self, RsaKeyError> {
    parse_pkcs1_private_key_der_with_policy(der, policy).map(|components| Self { components })
  }

  /// Parse a PKCS #8 `PrivateKeyInfo` DER object with the default policy.
  ///
  /// The default policy accepts RSA-3072 through RSA-8192 with exponent
  /// `65537`. Use [`RsaPublicKeyPolicy::modulus_2048_to_8192`] with
  /// [`Self::from_pkcs8_der_with_policy`] only for deployed RSA-2048
  /// compatibility material.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the DER is malformed, not an RSA private key,
  /// or the embedded PKCS #1 private key fails factor-strength or consistency
  /// validation.
  pub fn from_pkcs8_der(der: &[u8]) -> Result<Self, RsaKeyError> {
    Self::from_pkcs8_der_with_policy(der, &RsaPublicKeyPolicy::default())
  }

  /// Parse a PKCS #8 `PrivateKeyInfo` DER object with an explicit public-key policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if parsing, factor-strength validation, or
  /// component consistency validation fails.
  pub fn from_pkcs8_der_with_policy(der: &[u8], policy: &RsaPublicKeyPolicy) -> Result<Self, RsaKeyError> {
    parse_pkcs8_private_key_der_with_policy(der, policy).map(|components| Self { components })
  }

  /// Build an RSA private key from canonical unsigned big-endian CRT components.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the components violate RSA private-key policy
  /// or fail factor-strength or consistency checks.
  pub fn from_components(parts: RsaPrivateKeyParts<'_>) -> Result<Self, RsaKeyError> {
    Self::from_components_with_policy(parts, &RsaPublicKeyPolicy::default())
  }

  /// Build an RSA private key from CRT components with an explicit public-key policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the components violate `policy` or fail
  /// factor-strength or consistency checks.
  pub fn from_components_with_policy(
    parts: RsaPrivateKeyParts<'_>,
    policy: &RsaPublicKeyPolicy,
  ) -> Result<Self, RsaKeyError> {
    let public_exponent = parse_public_exponent(&parts.public_exponent.to_be_bytes(), policy)?;
    let components = RsaPrivateKeyDerComponents {
      modulus: parts.modulus,
      public_exponent,
      private_exponent: parts.private_exponent,
      prime_p: parts.prime_p,
      prime_q: parts.prime_q,
      exponent_p: parts.exponent_p,
      exponent_q: parts.exponent_q,
      coefficient: parts.coefficient,
    };
    private_key_components_from_parts(&components, policy).map(|components| Self { components })
  }

  /// Borrow the validated RSA public key corresponding to this private key.
  #[inline]
  #[must_use]
  pub fn public_key(&self) -> &RsaPublicKey {
    self.components.public_key()
  }

  /// Encode this private key as a PKCS #1 `RSAPrivateKey` DER object.
  ///
  /// The returned allocation zeroizes on drop. Borrow it with
  /// [`SecretVec::as_bytes`]; extracting an ordinary `Vec<u8>` requires
  /// [`SecretVec::into_unprotected_vec`].
  #[must_use]
  pub fn to_pkcs1_der(&self) -> SecretVec {
    self.components.encode_der(false)
  }

  /// Encode this private key as a PKCS #8 `PrivateKeyInfo` DER object.
  ///
  /// The returned allocation zeroizes on drop. Borrow it with
  /// [`SecretVec::as_bytes`]; extracting an ordinary `Vec<u8>` requires
  /// [`SecretVec::into_unprotected_vec`].
  #[must_use]
  pub fn to_pkcs8_der(&self) -> SecretVec {
    self.components.encode_der(true)
  }

  /// Return the fixed signature length for this key.
  #[inline]
  #[must_use]
  pub fn signature_len(&self) -> usize {
    self.public_key().modulus().len()
  }
}

impl RsaPrivateKeyComponents {
  #[inline]
  pub(super) fn public_key(&self) -> &RsaPublicKey {
    &self.public
  }

  fn encode_der(&self, pkcs8: bool) -> SecretVec {
    let public_exponent = self.public.public_exponent().as_u64().to_be_bytes();
    let values: [&[u8]; 9] = [
      &[0],
      self.public.modulus(),
      &public_exponent,
      self.private_exponent.as_bytes(),
      self.prime_p.as_bytes(),
      self.prime_q.as_bytes(),
      self.exponent_p.as_bytes(),
      self.exponent_q.as_bytes(),
      self.coefficient.as_bytes(),
    ];
    let body_len = der_integer_sequence_body_len(&values);
    let pkcs1_len = der_tlv_len(body_len);
    let pkcs8_body_len = der_integer_tlv_len(&[0])
      .strict_add(RSA_ENCRYPTION_ALGORITHM.len())
      .strict_add(der_tlv_len(pkcs1_len));
    let len = if pkcs8 { der_tlv_len(pkcs8_body_len) } else { pkcs1_len };
    SecretVec::build_with_capacity(len, |der| {
      if pkcs8 {
        der_push_header(TAG_SEQUENCE, pkcs8_body_len, der);
        der_push_integer_unsigned(&[0], der);
        der.extend_from_slice(RSA_ENCRYPTION_ALGORITHM);
        der_push_header(TAG_OCTET_STRING, pkcs1_len, der);
      }
      der_push_integer_sequence(&values, body_len, der);
    })
  }
}

impl RsaPublicKey {
  /// Build an RSA public key from canonical unsigned big-endian components
  /// with the default policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the modulus or exponent violates the default
  /// policy.
  pub fn from_modulus_exponent(modulus: &[u8], public_exponent: u64) -> Result<Self, RsaKeyError> {
    Self::from_modulus_exponent_with_policy(modulus, public_exponent, &RsaPublicKeyPolicy::default())
  }

  /// Build an RSA public key from components with an explicit policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the modulus or exponent violates `policy`.
  pub fn from_modulus_exponent_with_policy(
    modulus: &[u8],
    public_exponent: u64,
    policy: &RsaPublicKeyPolicy,
  ) -> Result<Self, RsaKeyError> {
    if policy.min_modulus_bits > policy.max_modulus_bits {
      return Err(RsaKeyError::InvalidModulus);
    }

    let exponent = parse_public_exponent(&public_exponent.to_be_bytes(), policy)?;
    let modulus_bits = validate_modulus(modulus, policy)?;
    Ok(Self {
      modulus: RsaPublicModulus::new(modulus, modulus_bits),
      exponent,
    })
  }

  /// Parse an RSA `SubjectPublicKeyInfo` DER object with the default policy.
  ///
  /// The default policy accepts RSA-3072 through RSA-8192 with exponent
  /// `65537`. Use [`RsaPublicKeyPolicy::modulus_2048_to_8192`] with
  /// [`Self::from_spki_der_with_policy`] only for deployed RSA-2048
  /// compatibility material.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the DER is malformed, the algorithm identifier
  /// is not `rsaEncryption` with NULL parameters, or the embedded key violates
  /// RSA public-key policy.
  pub fn from_spki_der(der: &[u8]) -> Result<Self, RsaKeyError> {
    Self::from_spki_der_with_policy(der, &RsaPublicKeyPolicy::default())
  }

  /// Parse an RSA `SubjectPublicKeyInfo` DER object with an explicit policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if parsing or validation fails.
  pub fn from_spki_der_with_policy(der: &[u8], policy: &RsaPublicKeyPolicy) -> Result<Self, RsaKeyError> {
    let (algorithm, public_key_der) = parse_spki_der(der)?;
    parse_rsa_algorithm_identifier(algorithm)?;
    Self::from_pkcs1_der_with_policy(public_key_der, policy)
  }

  /// Parse a PKCS #1 `RSAPublicKey` DER object with the default policy.
  ///
  /// The default policy accepts RSA-3072 through RSA-8192 with exponent
  /// `65537`. Use [`RsaPublicKeyPolicy::modulus_2048_to_8192`] with
  /// [`Self::from_pkcs1_der_with_policy`] only for deployed RSA-2048
  /// compatibility material.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the DER is malformed or the key violates RSA
  /// public-key policy.
  pub fn from_pkcs1_der(der: &[u8]) -> Result<Self, RsaKeyError> {
    Self::from_pkcs1_der_with_policy(der, &RsaPublicKeyPolicy::default())
  }

  /// Parse a PKCS #1 `RSAPublicKey` DER object with an explicit policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if parsing or validation fails.
  pub fn from_pkcs1_der_with_policy(der: &[u8], policy: &RsaPublicKeyPolicy) -> Result<Self, RsaKeyError> {
    let (modulus, modulus_bits, exponent) = parse_pkcs1_public_key_der_parts(der, policy)?;
    let modulus = RsaPublicModulus::new(modulus, modulus_bits);

    Ok(Self { modulus, exponent })
  }

  /// Borrow the canonical unsigned big-endian modulus bytes.
  #[inline]
  #[must_use]
  pub fn modulus(&self) -> &[u8] {
    &self.modulus.bytes
  }

  /// Return the modulus bit length.
  #[inline]
  #[must_use]
  pub const fn modulus_bits(&self) -> usize {
    self.modulus.bits
  }

  /// Return the validated public exponent.
  #[inline]
  #[must_use]
  pub const fn public_exponent(&self) -> RsaPublicExponent {
    self.exponent
  }

  /// Encode this public key as a PKCS #1 `RSAPublicKey` DER object.
  #[must_use]
  pub fn to_pkcs1_der(&self) -> Vec<u8> {
    self.encode_der(false)
  }

  /// Encode this public key as an X.509 `SubjectPublicKeyInfo` DER object.
  #[must_use]
  pub fn to_spki_der(&self) -> Vec<u8> {
    self.encode_der(true)
  }

  fn encode_der(&self, spki: bool) -> Vec<u8> {
    let exponent = self.public_exponent().as_u64().to_be_bytes();
    let values = [self.modulus(), exponent.as_slice()];
    let body_len = der_integer_sequence_body_len(&values);
    let pkcs1_len = der_tlv_len(body_len);
    let mut der = if spki {
      let bits_len = pkcs1_len.strict_add(1);
      let spki_body_len = RSA_ENCRYPTION_ALGORITHM.len().strict_add(der_tlv_len(bits_len));
      let mut der = Vec::with_capacity(der_tlv_len(spki_body_len));
      der_push_header(TAG_SEQUENCE, spki_body_len, &mut der);
      der.extend_from_slice(RSA_ENCRYPTION_ALGORITHM);
      der_push_header(TAG_BIT_STRING, bits_len, &mut der);
      der.push(0);
      der
    } else {
      Vec::with_capacity(pkcs1_len)
    };
    der_push_integer_sequence(&values, body_len, &mut der);
    der
  }
}

fn der_push_header(tag: u8, len: usize, der: &mut Vec<u8>) {
  der.push(tag);
  der_push_len(len, der);
}

fn der_integer_sequence_body_len(values: &[&[u8]]) -> usize {
  values
    .iter()
    .fold(0usize, |len, value| len.strict_add(der_integer_tlv_len(value)))
}

fn der_push_integer_sequence(values: &[&[u8]], body_len: usize, der: &mut Vec<u8>) {
  der_push_header(TAG_SEQUENCE, body_len, der);
  for value in values {
    der_push_integer_unsigned(value, der);
  }
}

fn der_integer_tlv_len(value: &[u8]) -> usize {
  let first_nonzero = value.iter().position(|&byte| byte != 0);
  let value = first_nonzero.and_then(|index| value.get(index..)).unwrap_or(&[0]);
  let content_len = value
    .len()
    .strict_add(usize::from(value.first().is_some_and(|byte| byte & 0x80 != 0)));
  der_tlv_len(content_len)
}

fn der_push_integer_unsigned(value: &[u8], out: &mut Vec<u8>) {
  let first_nonzero = value.iter().position(|&byte| byte != 0);
  let value = first_nonzero.and_then(|index| value.get(index..)).unwrap_or(&[0]);
  let needs_sign_padding = value.first().is_some_and(|byte| byte & 0x80 != 0);
  let content_len = value.len().strict_add(usize::from(needs_sign_padding));
  der_push_header(TAG_INTEGER, content_len, out);
  if needs_sign_padding {
    out.push(0);
  }
  out.extend_from_slice(value);
}

fn der_tlv_len(value_len: usize) -> usize {
  1usize.strict_add(der_len_len(value_len)).strict_add(value_len)
}

fn der_len_len(len: usize) -> usize {
  if len < 128 {
    return 1;
  }
  let significant = core::mem::size_of::<usize>().strict_sub((len.leading_zeros() as usize) / 8);
  1usize.strict_add(significant)
}

fn der_push_len(len: usize, out: &mut Vec<u8>) {
  if len < 128 {
    out.push(u8::try_from(len).expect("a short DER length must fit in one byte"));
    return;
  }

  let bytes = len.to_be_bytes();
  let first_nonzero = bytes
    .iter()
    .position(|&byte| byte != 0)
    .unwrap_or_else(|| bytes.len().strict_sub(1));
  let len_bytes = &bytes[first_nonzero..];
  out.push(0x80 | u8::try_from(len_bytes.len()).expect("a usize DER length prefix must fit in one byte"));
  out.extend_from_slice(len_bytes);
}

fn parse_rsa_algorithm_identifier(der: &[u8]) -> Result<(), RsaKeyError> {
  let mut reader = DerReader::new(der);
  let oid = reader.read_primitive(TAG_OBJECT_IDENTIFIER)?;
  if oid != RSA_ENCRYPTION_OID {
    return Err(RsaKeyError::UnsupportedAlgorithm);
  }

  let null = reader.read_primitive(TAG_NULL)?;
  if !null.is_empty() {
    return Err(RsaKeyError::MalformedDer);
  }
  reader.finish()
}

fn parse_pkcs8_private_key_der_with_policy(
  der: &[u8],
  policy: &RsaPublicKeyPolicy,
) -> Result<RsaPrivateKeyComponents, RsaKeyError> {
  let components = parse_pkcs8_private_key_der_parts_with_policy(der, policy)?;
  private_key_components_from_parts(&components, policy)
}

pub(super) fn parse_pkcs8_private_key_der_parts_with_policy<'a>(
  der: &'a [u8],
  policy: &RsaPublicKeyPolicy,
) -> Result<RsaPrivateKeyDerComponents<'a>, RsaKeyError> {
  let mut root = DerReader::new(der);
  let private_key_info = root.read_constructed(TAG_SEQUENCE)?;
  root.finish()?;

  let mut private_key_info = DerReader::new(private_key_info);
  read_zero_version(private_key_info.read_primitive(TAG_INTEGER)?)?;
  parse_rsa_algorithm_identifier(private_key_info.read_constructed(TAG_SEQUENCE)?)?;
  let private_key = private_key_info.read_primitive(TAG_OCTET_STRING)?;
  private_key_info.finish()?;

  parse_pkcs1_private_key_der_parts_with_policy(private_key, policy)
}

fn parse_pkcs1_private_key_der_with_policy(
  der: &[u8],
  policy: &RsaPublicKeyPolicy,
) -> Result<RsaPrivateKeyComponents, RsaKeyError> {
  let components = parse_pkcs1_private_key_der_parts_with_policy(der, policy)?;
  private_key_components_from_parts(&components, policy)
}

fn parse_pkcs1_private_key_der_parts_with_policy<'a>(
  der: &'a [u8],
  policy: &RsaPublicKeyPolicy,
) -> Result<RsaPrivateKeyDerComponents<'a>, RsaKeyError> {
  if policy.min_modulus_bits > policy.max_modulus_bits {
    return Err(RsaKeyError::InvalidModulus);
  }

  let mut root = DerReader::new(der);
  let private_key = root.read_constructed(TAG_SEQUENCE)?;
  root.finish()?;

  let mut private_key = DerReader::new(private_key);
  read_zero_version(private_key.read_primitive(TAG_INTEGER)?)?;
  let components = RsaPrivateKeyDerComponents {
    modulus: read_positive_integer(private_key.read_primitive(TAG_INTEGER)?)?,
    public_exponent: parse_public_exponent(read_positive_integer(private_key.read_primitive(TAG_INTEGER)?)?, policy)?,
    private_exponent: read_positive_integer(private_key.read_primitive(TAG_INTEGER)?)?,
    prime_p: read_positive_integer(private_key.read_primitive(TAG_INTEGER)?)?,
    prime_q: read_positive_integer(private_key.read_primitive(TAG_INTEGER)?)?,
    exponent_p: read_positive_integer(private_key.read_primitive(TAG_INTEGER)?)?,
    exponent_q: read_positive_integer(private_key.read_primitive(TAG_INTEGER)?)?,
    coefficient: read_positive_integer(private_key.read_primitive(TAG_INTEGER)?)?,
  };
  private_key.finish()?;

  Ok(components)
}

fn private_key_components_from_parts(
  components: &RsaPrivateKeyDerComponents<'_>,
  policy: &RsaPublicKeyPolicy,
) -> Result<RsaPrivateKeyComponents, RsaKeyError> {
  if policy.min_modulus_bits > policy.max_modulus_bits {
    return Err(RsaKeyError::InvalidModulus);
  }

  let modulus_bits = validate_private_key_import(components, policy)?;
  private_key_components_from_prevalidated_parts(components, modulus_bits)
}

pub(super) fn private_key_components_from_prevalidated_parts(
  components: &RsaPrivateKeyDerComponents<'_>,
  modulus_bits: usize,
) -> Result<RsaPrivateKeyComponents, RsaKeyError> {
  let public = RsaPublicKey {
    modulus: RsaPublicModulus::new_with_montgomery_r2(components.modulus, modulus_bits),
    exponent: components.public_exponent,
  };
  let prime_p_modulus = private_component_modulus(components.prime_p).map_err(|_| RsaKeyError::InvalidModulus)?;
  let prime_q_modulus = private_component_modulus(components.prime_q).map_err(|_| RsaKeyError::InvalidModulus)?;

  Ok(RsaPrivateKeyComponents {
    public,
    private_exponent: SecretBigEndianInteger::new_fixed_width(components.private_exponent, components.modulus.len())?,
    prime_p: SecretBigEndianInteger::new(components.prime_p)?,
    prime_q: SecretBigEndianInteger::new(components.prime_q)?,
    prime_p_modulus,
    prime_q_modulus,
    exponent_p: SecretBigEndianInteger::new_fixed_width(components.exponent_p, components.prime_p.len())?,
    exponent_q: SecretBigEndianInteger::new_fixed_width(components.exponent_q, components.prime_q.len())?,
    coefficient: SecretBigEndianInteger::new_fixed_width(components.coefficient, components.prime_p.len())?,
  })
}

fn validate_private_key_import(
  components: &RsaPrivateKeyDerComponents<'_>,
  policy: &RsaPublicKeyPolicy,
) -> Result<usize, RsaKeyError> {
  let modulus_bits = validate_modulus(components.modulus, policy)?;
  validate_private_prime_factor(components.prime_p)?;
  validate_private_prime_factor(components.prime_q)?;
  if !is_canonical_positive_unsigned_be(components.private_exponent) {
    return Err(RsaKeyError::InvalidModulus);
  }

  let factor_bits = modulus_bits.strict_add(1) / 2;
  if unsigned_be_bit_len(components.prime_p) != factor_bits || unsigned_be_bit_len(components.prime_q) != factor_bits {
    return Err(RsaKeyError::InvalidModulus);
  }

  validate_private_key_components(components)?;
  if !private_import_is_probable_prime(components.prime_p)? || !private_import_is_probable_prime(components.prime_q)? {
    return Err(RsaKeyError::InvalidModulus);
  }
  Ok(modulus_bits)
}

fn read_zero_version(bytes: &[u8]) -> Result<(), RsaKeyError> {
  let version = read_positive_integer(bytes)?;
  if version == [0] {
    Ok(())
  } else {
    Err(RsaKeyError::UnsupportedAlgorithm)
  }
}

fn validate_private_key_components(components: &RsaPrivateKeyDerComponents<'_>) -> Result<(), RsaKeyError> {
  let RsaPrivateKeyDerComponents {
    modulus,
    public_exponent,
    private_exponent,
    prime_p,
    prime_q,
    exponent_p,
    exponent_q,
    coefficient,
  } = *components;

  // Import checks that `d` is canonical. Key generation passes `d` at the modulus width, so a
  // short `d` starts with zero bytes.
  if is_zero_unsigned_be(private_exponent) || !ct_unsigned_be_lt_public_shape(private_exponent, modulus) {
    return Err(RsaKeyError::InvalidModulus);
  }
  validate_private_prime_factor(prime_p)?;
  validate_private_prime_factor(prime_q)?;
  if ct_slices_eq_public_shape(prime_p, prime_q) || !product_matches_unsigned_be_fixed(prime_p, prime_q, modulus) {
    return Err(RsaKeyError::InvalidModulus);
  }
  validate_private_crt_component(exponent_p, prime_p)?;
  validate_private_crt_component(exponent_q, prime_q)?;
  validate_private_crt_component(coefficient, prime_p)?;

  let mut p_minus_one = SecretBigEndianBuffer::zeroed(prime_p.len());
  let mut q_minus_one = SecretBigEndianBuffer::zeroed(prime_q.len());
  private_import_decrement_unsigned_be_to_fixed(prime_p, p_minus_one.as_mut_slice())?;
  private_import_decrement_unsigned_be_to_fixed(prime_q, q_minus_one.as_mut_slice())?;

  let mut d_mod_p_minus_one = SecretBigEndianBuffer::zeroed(p_minus_one.as_slice().len());
  private_import_unsigned_be_mod_to_len(
    private_exponent,
    p_minus_one.as_slice(),
    d_mod_p_minus_one.as_mut_slice(),
  )?;
  if !ct_eq_left_padded_unsigned_be(exponent_p, d_mod_p_minus_one.as_slice()) {
    return Err(RsaKeyError::InvalidModulus);
  }
  let mut d_mod_q_minus_one = SecretBigEndianBuffer::zeroed(q_minus_one.as_slice().len());
  private_import_unsigned_be_mod_to_len(
    private_exponent,
    q_minus_one.as_slice(),
    d_mod_q_minus_one.as_mut_slice(),
  )?;
  if !ct_eq_left_padded_unsigned_be(exponent_q, d_mod_q_minus_one.as_slice()) {
    return Err(RsaKeyError::InvalidModulus);
  }
  let public_exponent = public_exponent.as_u64().to_be_bytes();
  let mut e_times_d = SecretBigEndianBuffer::zeroed(public_exponent.len().strict_add(private_exponent.len()));
  private_import_product_unsigned_be_to_fixed(&public_exponent, private_exponent, e_times_d.as_mut_slice())?;
  let mut e_times_d_mod_p_minus_one = SecretBigEndianBuffer::zeroed(p_minus_one.as_slice().len());
  private_import_unsigned_be_mod_to_len(
    e_times_d.as_slice(),
    p_minus_one.as_slice(),
    e_times_d_mod_p_minus_one.as_mut_slice(),
  )?;
  if !ct_eq_left_padded_unsigned_be(&[1], e_times_d_mod_p_minus_one.as_slice()) {
    return Err(RsaKeyError::InvalidModulus);
  }
  let mut e_times_d_mod_q_minus_one = SecretBigEndianBuffer::zeroed(q_minus_one.as_slice().len());
  private_import_unsigned_be_mod_to_len(
    e_times_d.as_slice(),
    q_minus_one.as_slice(),
    e_times_d_mod_q_minus_one.as_mut_slice(),
  )?;
  if !ct_eq_left_padded_unsigned_be(&[1], e_times_d_mod_q_minus_one.as_slice()) {
    return Err(RsaKeyError::InvalidModulus);
  }

  let mut q_times_coefficient = SecretBigEndianBuffer::zeroed(prime_q.len().strict_add(coefficient.len()));
  private_import_product_unsigned_be_to_fixed(prime_q, coefficient, q_times_coefficient.as_mut_slice())?;
  let mut q_times_coefficient_mod_p = SecretBigEndianBuffer::zeroed(prime_p.len());
  private_import_unsigned_be_mod_to_len(
    q_times_coefficient.as_slice(),
    prime_p,
    q_times_coefficient_mod_p.as_mut_slice(),
  )?;
  if !ct_eq_left_padded_unsigned_be(&[1], q_times_coefficient_mod_p.as_slice()) {
    return Err(RsaKeyError::InvalidModulus);
  }

  Ok(())
}

fn validate_private_prime_factor(bytes: &[u8]) -> Result<(), RsaKeyError> {
  let Some(&last) = bytes.last() else {
    return Err(RsaKeyError::InvalidModulus);
  };
  if !is_canonical_positive_unsigned_be(bytes) || bytes == [1] || last & 1 == 0 {
    return Err(RsaKeyError::InvalidModulus);
  }
  Ok(())
}

fn private_import_is_probable_prime(candidate: &[u8]) -> Result<bool, RsaKeyError> {
  if has_small_prime_factor(candidate) {
    return Ok(false);
  }

  let mut n_minus_one = SecretBigEndianBuffer::zeroed(candidate.len());
  private_import_decrement_unsigned_be_to_fixed(candidate, n_minus_one.as_mut_slice())?;
  let mut odd_part = SecretBigEndianBuffer::zeroed(candidate.len());
  odd_part.as_mut_slice().copy_from_slice(n_minus_one.as_slice());
  let mut powers_of_two = 0usize;
  while odd_part.as_slice().last().is_some_and(|byte| byte & 1 == 0) {
    shift_right_one_unsigned_be(odd_part.as_mut_slice());
    powers_of_two = powers_of_two.strict_add(1);
  }

  let modulus = private_component_modulus(candidate).map_err(|_| RsaKeyError::InvalidModulus)?;
  for base_value in RSA_IMPORT_MILLER_RABIN_BASES {
    let mut base = vec![0u8; candidate.len()];
    let base_bytes = base_value.to_be_bytes();
    let offset = base.len().strict_sub(base_bytes.len());
    let Some(base_suffix) = base.get_mut(offset..) else {
      return Err(RsaKeyError::InvalidModulus);
    };
    base_suffix.copy_from_slice(&base_bytes);
    let accepted = miller_rabin_accepts_base(
      &modulus,
      odd_part.as_slice(),
      powers_of_two,
      n_minus_one.as_slice(),
      &base,
    )
    .map_err(|_| RsaKeyError::InvalidModulus)?;
    ct::zeroize(&mut base);
    if !accepted {
      return Ok(false);
    }
  }

  Ok(true)
}

fn shift_right_one_unsigned_be(bytes: &mut [u8]) {
  let mut carry = 0u8;
  for byte in bytes {
    let next_carry = *byte & 1;
    *byte = (*byte >> 1) | (carry << 7);
    carry = next_carry;
  }
}

fn validate_private_crt_component(component: &[u8], upper_bound: &[u8]) -> Result<(), RsaKeyError> {
  if !is_canonical_positive_unsigned_be(component) || !ct_unsigned_be_lt_public_shape(component, upper_bound) {
    return Err(RsaKeyError::InvalidModulus);
  }
  Ok(())
}

#[cfg(feature = "getrandom")]
impl RsaKeygenDrbg {
  fn from_os_entropy() -> Result<Self, RsaKeyGenerationError> {
    let mut seed = [0u8; RSA_KEYGEN_DRBG_ENTROPY_BYTES + RSA_KEYGEN_DRBG_NONCE_BYTES];
    if getrandom::fill(&mut seed).is_err() {
      ct::zeroize(&mut seed);
      return Err(RsaKeyGenerationError::EntropyUnavailable);
    }

    let drbg = Self::new(&seed, RSA_KEYGEN_DRBG_PERSONALIZATION);
    ct::zeroize(&mut seed);
    Ok(drbg)
  }

  pub(super) fn new(entropy_and_nonce: &[u8], personalization: &[u8]) -> Self {
    let mut drbg = Self {
      key: [0u8; RSA_KEYGEN_DRBG_KEY_BYTES],
      value: [1u8; RSA_KEYGEN_DRBG_OUT_BYTES],
    };
    drbg.update(&[entropy_and_nonce, personalization]);
    drbg
  }

  pub(super) fn fill(&mut self, out: &mut [u8]) {
    for chunk in out.chunks_mut(RSA_KEYGEN_DRBG_OUT_BYTES) {
      self.value = hmac_sha256_parts(&self.key, [self.value.as_slice()]);
      chunk.copy_from_slice(&self.value[..chunk.len()]);
    }
    self.update(&[]);
  }

  pub(super) fn update(&mut self, seed_material: &[&[u8]]) {
    self.key = keygen_drbg_update_key(&self.key, &self.value, 0, seed_material);
    self.value = hmac_sha256_parts(&self.key, [self.value.as_slice()]);

    if seed_material.iter().all(|part| part.is_empty()) {
      return;
    }

    self.key = keygen_drbg_update_key(&self.key, &self.value, 1, seed_material);
    self.value = hmac_sha256_parts(&self.key, [self.value.as_slice()]);
  }
}

#[cfg(feature = "getrandom")]
impl Drop for RsaKeygenDrbg {
  fn drop(&mut self) {
    ct::zeroize(&mut self.key);
    ct::zeroize(&mut self.value);
  }
}

#[cfg(feature = "getrandom")]
fn keygen_drbg_update_key(
  key: &[u8; RSA_KEYGEN_DRBG_KEY_BYTES],
  value: &[u8; RSA_KEYGEN_DRBG_OUT_BYTES],
  domain: u8,
  seed_material: &[&[u8]],
) -> [u8; RSA_KEYGEN_DRBG_OUT_BYTES] {
  let domain = [domain];
  hmac_sha256_parts(
    key,
    [value.as_slice(), &domain]
      .into_iter()
      .chain(seed_material.iter().copied()),
  )
}

#[cfg(feature = "getrandom")]
fn generate_rsa_private_key(
  modulus_bits: usize,
  policy: &RsaPublicKeyPolicy,
) -> Result<RsaPrivateKeyComponents, RsaKeyGenerationError> {
  if policy.min_modulus_bits > policy.max_modulus_bits
    || modulus_bits < policy.min_modulus_bits
    || modulus_bits > policy.max_modulus_bits
    || modulus_bits < MIN_RSA_MODULUS_BITS
    || !modulus_bits.is_multiple_of(2)
    || !policy.exponent_policy.accepts(RSA_KEYGEN_PUBLIC_EXPONENT)
    || !keygen_public_exponent_is_fips_valid(RSA_KEYGEN_PUBLIC_EXPONENT)
  {
    return Err(RsaKeyGenerationError::InvalidModulusBits);
  }

  let mut drbg = RsaKeygenDrbg::from_os_entropy()?;
  let prime_p_bits = modulus_bits / 2;
  let prime_q_bits = modulus_bits / 2;
  for _ in 0..RSA_KEYGEN_PAIR_ATTEMPTS {
    let prime_p = keygen_generate_prime(&mut drbg, prime_p_bits, modulus_bits, None, modulus_bits.strict_mul(5))?;
    let prime_q = match keygen_generate_prime(
      &mut drbg,
      prime_q_bits,
      modulus_bits,
      Some(prime_p.as_slice()),
      modulus_bits.strict_mul(10),
    ) {
      Ok(prime_q) => prime_q,
      Err(err) => {
        let mut prime_p = prime_p;
        ct::zeroize(&mut prime_p);
        return Err(err);
      }
    };
    if let Some(components) = keygen_build_private_key_from_primes(modulus_bits, policy, prime_p, prime_q)? {
      return Ok(components);
    }
  }

  Err(RsaKeyGenerationError::PrimeSearchFailed)
}

#[cfg(feature = "getrandom")]
const fn keygen_public_exponent_is_fips_valid(exponent: u64) -> bool {
  exponent > (1u64 << 16) && exponent % 2 == 1
}

#[cfg(feature = "getrandom")]
fn keygen_build_private_key_from_primes(
  modulus_bits: usize,
  policy: &RsaPublicKeyPolicy,
  prime_p: Vec<u8>,
  prime_q: Vec<u8>,
) -> Result<Option<RsaPrivateKeyComponents>, RsaKeyGenerationError> {
  let prime_p = SecretBigEndianBuffer::new(prime_p);
  let prime_q = SecretBigEndianBuffer::new(prime_q);

  if prime_p.as_slice() == prime_q.as_slice() {
    return Ok(None);
  }
  if keygen_conflicts_with_public_exponent(prime_p.as_slice())
    || keygen_conflicts_with_public_exponent(prime_q.as_slice())
  {
    return Ok(None);
  }
  if !keygen_prime_distance_is_sufficient(prime_p.as_slice(), prime_q.as_slice(), modulus_bits) {
    return Ok(None);
  }

  let modulus = private_import_product_unsigned_be(prime_p.as_slice(), prime_q.as_slice())
    .ok_or(RsaKeyGenerationError::ArithmeticFailure)?;
  if unsigned_be_bit_len(modulus.as_slice()) != modulus_bits {
    return Ok(None);
  }

  let p_minus_one =
    private_import_decrement_unsigned_be(prime_p.as_slice()).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
  let q_minus_one =
    private_import_decrement_unsigned_be(prime_q.as_slice()).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
  let Some(private_exponent) = keygen_private_exponent(p_minus_one.as_slice(), q_minus_one.as_slice(), modulus_bits)
  else {
    return Ok(None);
  };
  let exponent_p = private_import_unsigned_be_mod(private_exponent.as_slice(), p_minus_one.as_slice());
  let exponent_q = private_import_unsigned_be_mod(private_exponent.as_slice(), q_minus_one.as_slice());
  let coefficient = SecretBigEndianBuffer::new(keygen_prime_mod_inverse(prime_q.as_slice(), prime_p.as_slice())?);

  let public_exponent = RsaPublicExponent(RSA_KEYGEN_PUBLIC_EXPONENT);
  let generated = RsaPrivateKeyDerComponents {
    modulus: modulus.as_slice(),
    public_exponent,
    private_exponent: private_exponent.as_slice(),
    prime_p: prime_p.as_slice(),
    prime_q: prime_q.as_slice(),
    exponent_p: exponent_p.as_slice(),
    exponent_q: exponent_q.as_slice(),
    coefficient: coefficient.as_slice(),
  };
  validate_private_key_components(&generated).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
  let checked_modulus_bits =
    validate_modulus(modulus.as_slice(), policy).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
  let public = RsaPublicKey {
    modulus: RsaPublicModulus::new_with_montgomery_r2(modulus.as_slice(), checked_modulus_bits),
    exponent: public_exponent,
  };
  let prime_p_modulus =
    private_component_modulus(prime_p.as_slice()).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
  let prime_q_modulus =
    private_component_modulus(prime_q.as_slice()).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
  let prime_p_bytes = prime_p_modulus.bytes.len();
  let prime_q_bytes = prime_q_modulus.bytes.len();
  let modulus_bytes = public.modulus().len();

  Ok(Some(RsaPrivateKeyComponents {
    public,
    private_exponent: SecretBigEndianInteger::new_fixed_width(private_exponent.as_slice(), modulus_bytes)
      .map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?,
    prime_p: SecretBigEndianInteger::from_vec(prime_p.into_vec())
      .map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?,
    prime_q: SecretBigEndianInteger::from_vec(prime_q.into_vec())
      .map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?,
    prime_p_modulus,
    prime_q_modulus,
    exponent_p: SecretBigEndianInteger::new_fixed_width(exponent_p.as_slice(), prime_p_bytes)
      .map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?,
    exponent_q: SecretBigEndianInteger::new_fixed_width(exponent_q.as_slice(), prime_q_bytes)
      .map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?,
    coefficient: SecretBigEndianInteger::new_fixed_width(coefficient.as_slice(), prime_p_bytes)
      .map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?,
  }))
}

#[cfg(feature = "getrandom")]
fn keygen_generate_prime(
  drbg: &mut RsaKeygenDrbg,
  bits: usize,
  modulus_bits: usize,
  other_prime: Option<&[u8]>,
  tested_candidate_limit: usize,
) -> Result<Vec<u8>, RsaKeyGenerationError> {
  if bits < 2 {
    return Err(RsaKeyGenerationError::InvalidModulusBits);
  }

  let mut tested_candidates = 0usize;
  while tested_candidates < tested_candidate_limit {
    let candidate = keygen_random_odd_candidate(drbg, bits);
    if !keygen_probable_prime_meets_fips_lower_bound(&candidate, bits)
      || other_prime.is_some_and(|prime| !keygen_prime_distance_is_sufficient(&candidate, prime, modulus_bits))
    {
      let mut candidate = candidate;
      ct::zeroize(&mut candidate);
      continue;
    }

    tested_candidates = tested_candidates.strict_add(1);
    // Trial division rejects obvious composites before the expensive B.3 Miller-Rabin work.
    if has_small_prime_factor(&candidate) || keygen_conflicts_with_public_exponent(&candidate) {
      let mut candidate = candidate;
      ct::zeroize(&mut candidate);
      continue;
    }
    match keygen_is_probable_prime(drbg, &candidate) {
      Ok(true) => return Ok(candidate),
      Ok(false) => {
        let mut candidate = candidate;
        ct::zeroize(&mut candidate);
      }
      Err(err) => {
        let mut candidate = candidate;
        ct::zeroize(&mut candidate);
        return Err(err);
      }
    }
  }

  Err(RsaKeyGenerationError::PrimeSearchFailed)
}

#[cfg(feature = "getrandom")]
const fn keygen_miller_rabin_rounds(_candidate_bits: usize) -> usize {
  RSA_KEYGEN_MILLER_RABIN_ROUNDS
}

#[cfg(feature = "getrandom")]
fn keygen_random_odd_candidate(drbg: &mut RsaKeygenDrbg, bits: usize) -> Vec<u8> {
  let len = bits.strict_add(7) / 8;
  let mut candidate = vec![0u8; len];
  drbg.fill(&mut candidate);
  keygen_mask_unused_top_bits(&mut candidate, bits);
  if let Some(last) = candidate.last_mut() {
    *last |= 1;
  }
  candidate
}

#[cfg(feature = "getrandom")]
fn keygen_probable_prime_meets_fips_lower_bound(candidate: &[u8], bits: usize) -> bool {
  if bits.is_multiple_of(8)
    && let Some(prefix) = candidate.get(..RSA_KEYGEN_SQRT2_HALF_TOP64.len())
  {
    match prefix.cmp(&RSA_KEYGEN_SQRT2_HALF_TOP64) {
      core::cmp::Ordering::Greater => return true,
      core::cmp::Ordering::Less => return false,
      core::cmp::Ordering::Equal => {}
    }
  }

  private_import_product_unsigned_be(candidate, candidate)
    .as_ref()
    .is_some_and(|square| unsigned_be_bit_len(square.as_slice()) >= bits.strict_mul(2))
}

#[cfg(feature = "getrandom")]
fn keygen_mask_unused_top_bits(bytes: &mut [u8], bits: usize) {
  let used_top_bits = bits % 8;
  if used_top_bits == 0 {
    return;
  }
  if let Some(first) = bytes.first_mut() {
    *first &= u8::MAX >> 8usize.strict_sub(used_top_bits);
  }
}

fn has_small_prime_factor(candidate: &[u8]) -> bool {
  RSA_SMALL_PRIMES
    .iter()
    .any(|&prime| unsigned_be_mod_u64(candidate, u64::from(prime)) == 0)
}

#[cfg(feature = "getrandom")]
fn keygen_conflicts_with_public_exponent(candidate: &[u8]) -> bool {
  unsigned_be_mod_u64(candidate, RSA_KEYGEN_PUBLIC_EXPONENT) == 1
}

#[cfg(feature = "getrandom")]
fn keygen_prime_distance_is_sufficient(prime_p: &[u8], prime_q: &[u8], modulus_bits: usize) -> bool {
  let Some(min_distance_bits) = (modulus_bits / 2).checked_sub(RSA_KEYGEN_MIN_PRIME_DISTANCE_SECURITY_MARGIN_BITS)
  else {
    return true;
  };

  let (larger, smaller) = match unsigned_be_cmp(prime_p, prime_q) {
    core::cmp::Ordering::Less => (prime_q, prime_p),
    core::cmp::Ordering::Equal => return false,
    core::cmp::Ordering::Greater => (prime_p, prime_q),
  };
  let Ok(distance) = private_sub_unsigned_be_to_len(larger, smaller, larger.len()) else {
    return false;
  };
  unsigned_be_bit_len(distance.as_slice()) > min_distance_bits
}

#[cfg(feature = "getrandom")]
fn keygen_is_probable_prime(drbg: &mut RsaKeygenDrbg, candidate: &[u8]) -> Result<bool, RsaKeyGenerationError> {
  let n_minus_one =
    private_import_decrement_unsigned_be(candidate).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
  let mut n_minus_one_fixed = vec![0u8; candidate.len()];
  if let Err(err) = keygen_left_pad(n_minus_one.as_slice(), &mut n_minus_one_fixed) {
    ct::zeroize(&mut n_minus_one_fixed);
    return Err(err);
  }

  let mut odd_part = n_minus_one.as_slice().to_vec();
  let mut powers_of_two = 0usize;
  while odd_part.last().is_some_and(|byte| byte & 1 == 0) {
    shift_right_one_unsigned_be(&mut odd_part);
    odd_part = keygen_canonical_vec(odd_part);
    powers_of_two = powers_of_two.strict_add(1);
  }
  let odd_part = SecretBigEndianBuffer::new(odd_part);
  let mut odd_part_fixed = SecretBigEndianBuffer::zeroed(candidate.len());
  if let Err(err) = keygen_left_pad(odd_part.as_slice(), odd_part_fixed.as_mut_slice()) {
    ct::zeroize(&mut n_minus_one_fixed);
    return Err(err);
  }

  let result = (|| {
    let modulus = private_component_modulus(candidate).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
    for _ in 0..keygen_miller_rabin_rounds(unsigned_be_bit_len(candidate)) {
      let mut base = keygen_random_miller_rabin_base(drbg, candidate, &n_minus_one_fixed)?;
      let accepted = keygen_miller_rabin_accepts_base(
        &modulus,
        odd_part_fixed.as_slice(),
        powers_of_two,
        &n_minus_one_fixed,
        &base,
      );
      ct::zeroize(&mut base);
      if !accepted? {
        return Ok(false);
      }
    }
    Ok(true)
  })();

  ct::zeroize(&mut n_minus_one_fixed);
  result
}

#[cfg(feature = "getrandom")]
fn keygen_random_miller_rabin_base(
  drbg: &mut RsaKeygenDrbg,
  candidate: &[u8],
  n_minus_one_fixed: &[u8],
) -> Result<Vec<u8>, RsaKeyGenerationError> {
  let mut n_minus_two = n_minus_one_fixed.to_vec();
  keygen_sub_one_fixed(&mut n_minus_two)?;
  let mut two = vec![0u8; candidate.len()];
  if let Some(last) = two.last_mut() {
    *last = 2;
  }

  for _ in 0..128 {
    let mut base = vec![0u8; candidate.len()];
    drbg.fill(&mut base);
    keygen_mask_unused_top_bits(&mut base, unsigned_be_bit_len(candidate));
    if unsigned_be_cmp(&base, &two) != core::cmp::Ordering::Less
      && unsigned_be_cmp(&base, &n_minus_two) != core::cmp::Ordering::Greater
    {
      ct::zeroize(&mut n_minus_two);
      ct::zeroize(&mut two);
      return Ok(base);
    }
    ct::zeroize(&mut base);
  }

  ct::zeroize(&mut n_minus_two);
  ct::zeroize(&mut two);
  Err(RsaKeyGenerationError::PrimeSearchFailed)
}

#[cfg(feature = "getrandom")]
fn keygen_miller_rabin_accepts_base(
  modulus: &RsaPublicModulus,
  odd_part: &[u8],
  powers_of_two: usize,
  n_minus_one_fixed: &[u8],
  base: &[u8],
) -> Result<bool, RsaKeyGenerationError> {
  miller_rabin_accepts_base(modulus, odd_part, powers_of_two, n_minus_one_fixed, base)
    .map_err(|_| RsaKeyGenerationError::ArithmeticFailure)
}

fn miller_rabin_accepts_base(
  modulus: &RsaPublicModulus,
  odd_part: &[u8],
  powers_of_two: usize,
  n_minus_one_fixed: &[u8],
  base: &[u8],
) -> Result<bool, RsaPrivateOpError> {
  let mut x = vec![0u8; modulus.bytes.len()];
  if let Err(err) = private_exponentiate_representative(modulus, odd_part, base, &mut x) {
    ct::zeroize(&mut x);
    return Err(err);
  }

  let mut accepted = is_one_fixed(&x) || ct::public_len_eq(&x, n_minus_one_fixed).declassify();
  for _ in 1..powers_of_two {
    if accepted {
      break;
    }
    let mut squared = vec![0u8; modulus.bytes.len()];
    if let Err(err) = mod_mul_representatives(modulus, &x, &x, &mut squared) {
      ct::zeroize(&mut squared);
      ct::zeroize(&mut x);
      return Err(err);
    }
    x.copy_from_slice(&squared);
    ct::zeroize(&mut squared);
    accepted = ct::public_len_eq(&x, n_minus_one_fixed).declassify();
  }

  ct::zeroize(&mut x);
  Ok(accepted)
}

#[cfg(feature = "getrandom")]
fn keygen_prime_mod_inverse(value: &[u8], prime_modulus: &[u8]) -> Result<Vec<u8>, RsaKeyGenerationError> {
  let modulus = private_component_modulus(prime_modulus).map_err(|_| RsaKeyGenerationError::ArithmeticFailure)?;
  let value_mod = private_import_unsigned_be_mod(value, prime_modulus);
  let mut value_fixed = vec![0u8; prime_modulus.len()];
  if let Err(err) = keygen_left_pad(value_mod.as_slice(), &mut value_fixed) {
    ct::zeroize(&mut value_fixed);
    return Err(err);
  }

  let mut p_minus_two = SecretBigEndianBuffer::new(prime_modulus.to_vec());
  keygen_sub_one_fixed(p_minus_two.as_mut_slice())?;
  keygen_sub_one_fixed(p_minus_two.as_mut_slice())?;
  let mut inverse = vec![0u8; prime_modulus.len()];
  let result =
    if private_exponentiate_representative(&modulus, p_minus_two.as_slice(), &value_fixed, &mut inverse).is_ok() {
      Ok(keygen_canonical_vec(inverse))
    } else {
      ct::zeroize(&mut inverse);
      Err(RsaKeyGenerationError::ArithmeticFailure)
    };
  ct::zeroize(&mut value_fixed);
  result
}

/// Return the FIPS 186-5 private exponent `d = e⁻¹ mod lcm(p - 1, q - 1)` as a modulus-width
/// big-endian value, or `None` when `d ≤ 2^(modulus_bits / 2)`.
///
/// The work depends only on the public widths of `p - 1` and `q - 1`.
#[cfg(feature = "getrandom")]
fn keygen_private_exponent(
  p_minus_one: &[u8],
  q_minus_one: &[u8],
  modulus_bits: usize,
) -> Option<SecretBigEndianBuffer> {
  let limbs = p_minus_one.len().max(q_minus_one.len()).div_ceil(8);
  let mut a = SecretLimbs::zeroed(limbs);
  let mut b = SecretLimbs::zeroed(limbs);
  limbs_from_be_into(p_minus_one, a.as_mut_slice());
  limbs_from_be_into(q_minus_one, b.as_mut_slice());
  let mut private_exponent = keygen_lcm(a.as_slice(), b.as_slice());
  keygen_invert_public_exponent(private_exponent.as_mut_slice());

  let half = modulus_bits / 2;
  let mut minimum = vec![0u64; private_exponent.as_slice().len()];
  minimum[half / 64] = 1 << (half % 64);
  if !ct_limbs_lt(&minimum, private_exponent.as_slice()) {
    return None;
  }
  let mut out = SecretBigEndianBuffer::zeroed(modulus_bits.div_ceil(8));
  limbs_to_be(private_exponent.as_slice(), out.as_mut_slice());
  Some(out)
}

/// Return `lcm(a, b)` in `2·L` limbs for nonzero `a` and `b` of `L` limbs each.
///
/// Write `gcd(a, b) = 2^s·h` with `h` odd, and `a = 2^s·a'`. Then `lcm(a, b) = (a' / h)·b`.
/// Loop counts and indices depend only on `L`, and each choice is a mask.
#[cfg(feature = "getrandom")]
fn keygen_lcm(a: &[u64], b: &[u64]) -> SecretLimbs {
  let limbs = a.len();
  let bits = limbs.strict_mul(64);
  let mut u = SecretLimbs::from_limbs(a);
  let mut v = SecretLimbs::from_limbs(b);
  let mut t = SecretLimbs::zeroed(limbs);
  let (u, v, t) = (u.as_mut_slice(), v.as_mut_slice(), t.as_mut_slice());
  for _ in 0..bits {
    let both_even = !(u[0] | v[0]) & 1;
    keygen_ct_halve(u, both_even);
    keygen_ct_halve(v, both_even);
  }
  let reduced_a = SecretLimbs::from_limbs(u);

  // Stein's algorithm keeps `v` odd. Each step can replace `u` by `|u - v|` and `v` by
  // `min(u, v)`, then halves `u`. The bit lengths of `u` and `v` lose at least one bit in
  // total, so `2·bits` steps end with `u = 0` and `v = h`.
  keygen_ct_swap(u, v, !v[0] & 1);
  for _ in 0..bits.strict_mul(2) {
    let u_odd = u[0] & 1;
    t.copy_from_slice(u);
    keygen_ct_swap(u, v, u_odd & sub_modulus_in_place(t, v));
    t.copy_from_slice(u);
    sub_modulus_in_place(t, v);
    keygen_ct_swap(u, t, u_odd);
    keygen_ct_halve(u, 1);
  }

  // Long division of `a'` by `h`. The zero `u` holds the remainder.
  let mut quotient = SecretLimbs::zeroed(limbs);
  let quotient_limbs = quotient.as_mut_slice();
  for bit in (0..bits).rev() {
    let mut carry = (reduced_a.as_slice()[bit / 64] >> (bit % 64)) & 1;
    for limb in u.iter_mut() {
      let next = *limb >> 63;
      *limb = (*limb << 1) | carry;
      carry = next;
    }
    t.copy_from_slice(u);
    let take = carry | (sub_modulus_in_place(t, v) ^ 1);
    keygen_ct_swap(u, t, take);
    quotient_limbs[bit / 64] |= take << (bit % 64);
  }

  let mut lcm = SecretLimbs::zeroed(limbs.strict_mul(2));
  let product = lcm.as_mut_slice();
  for (index, &quotient_limb) in quotient_limbs.iter().enumerate() {
    let mut carry = 0u64;
    for (slot, &b_limb) in product[index..].iter_mut().zip(b) {
      (*slot, carry) = mont_mac(*slot, quotient_limb, b_limb, carry);
    }
    product[index.strict_add(limbs)] = carry;
  }
  lcm
}

/// Replace `lambda` with `e⁻¹ mod lambda` for the public exponent `e`, or with 0 when `e`
/// divides `lambda`.
///
/// Write `lambda = e·Q + r`. The `k < e` with `k·r ≡ -1 (mod e)` gives
/// `e⁻¹ mod lambda = (1 + k·lambda) / e = k·Q + (1 + k·r) / e`. A masked scan over each `k`
/// finds `k` and `(1 + k·r) / e`, so no step divides by a secret value.
#[cfg(feature = "getrandom")]
fn keygen_invert_public_exponent(lambda: &mut [u64]) {
  const E: u64 = RSA_KEYGEN_PUBLIC_EXPONENT;
  let mut remainder = 0u64;
  for limb in lambda.iter_mut().rev() {
    let mut quotient = 0u64;
    for bit in (0..64).rev() {
      remainder = (remainder << 1) | ((*limb >> bit) & 1);
      let (reduced, borrow) = remainder.overflowing_sub(E);
      let take = u64::from(borrow) ^ 1;
      remainder ^= (remainder ^ reduced) & keygen_ct_mask(take);
      quotient = (quotient << 1) | take;
    }
    *limb = quotient;
  }

  // `residue = k·r mod e` and `wraps = ⌊k·r / e⌋` stay below `2·e`, so no addition wraps.
  let (mut residue, mut wraps, mut multiplier, mut addend) = (0u64, 0u64, 0u64, 0u64);
  for k in 1..E {
    residue = residue.wrapping_add(remainder);
    let (reduced, borrow) = residue.overflowing_sub(E);
    let wrap = u64::from(borrow) ^ 1;
    residue ^= (residue ^ reduced) & keygen_ct_mask(wrap);
    wraps = wraps.wrapping_add(wrap);
    let found = keygen_ct_mask(ct_nonzero_u64(residue ^ (E - 1)) ^ 1);
    multiplier |= k & found;
    addend |= wraps.wrapping_add(1) & found;
  }

  let mut carry = addend;
  for limb in lambda {
    (*limb, carry) = mont_mac(0, *limb, multiplier, carry);
  }
}

#[cfg(feature = "getrandom")]
fn keygen_ct_mask(choice: u64) -> u64 {
  // SECURITY: An opaque mask keeps LLVM from turning a secret selection into a branch.
  core::hint::black_box(0u64.wrapping_sub(choice & 1))
}

/// Swap `left` and `right` when `choice` is 1.
#[cfg(feature = "getrandom")]
fn keygen_ct_swap(left: &mut [u64], right: &mut [u64], choice: u64) {
  let mask = keygen_ct_mask(choice);
  for (left, right) in left.iter_mut().zip(right) {
    let difference = (*left ^ *right) & mask;
    *left ^= difference;
    *right ^= difference;
  }
}

/// Halve `value` when `choice` is 1.
#[cfg(feature = "getrandom")]
fn keygen_ct_halve(value: &mut [u64], choice: u64) {
  let mask = keygen_ct_mask(choice);
  let mut above = 0u64;
  for limb in value.iter_mut().rev() {
    let original = *limb;
    *limb ^= (original ^ ((original >> 1) | (above << 63))) & mask;
    above = original;
  }
}

#[cfg(feature = "getrandom")]
fn keygen_sub_one_fixed(bytes: &mut [u8]) -> Result<(), RsaKeyGenerationError> {
  for byte in bytes.iter_mut().rev() {
    if *byte == 0 {
      *byte = 0xff;
    } else {
      *byte = byte.strict_sub(1);
      return Ok(());
    }
  }
  Err(RsaKeyGenerationError::ArithmeticFailure)
}

fn is_one_fixed(bytes: &[u8]) -> bool {
  bytes.last() == Some(&1)
    && bytes
      .get(..bytes.len().saturating_sub(1))
      .is_some_and(|prefix| prefix.iter().all(|&byte| byte == 0))
}

#[cfg(feature = "getrandom")]
fn keygen_left_pad(src: &[u8], out: &mut [u8]) -> Result<(), RsaKeyGenerationError> {
  if src.len() > out.len() {
    return Err(RsaKeyGenerationError::ArithmeticFailure);
  }
  out.fill(0);
  let offset = out.len().strict_sub(src.len());
  let Some(dst) = out.get_mut(offset..) else {
    return Err(RsaKeyGenerationError::ArithmeticFailure);
  };
  dst.copy_from_slice(src);
  Ok(())
}

#[cfg(feature = "getrandom")]
fn keygen_canonical_vec(bytes: Vec<u8>) -> Vec<u8> {
  let canonical = private_import_canonical_unsigned_be(bytes);
  canonical.as_slice().to_vec()
}

pub(super) fn unsigned_be_bit_len(bytes: &[u8]) -> usize {
  let Some(first_nonzero) = bytes.iter().position(|&byte| byte != 0) else {
    return 0;
  };
  let significant = bytes.get(first_nonzero..).unwrap_or_default();
  let Some(&first) = significant.first() else {
    return 0;
  };
  significant
    .len()
    .strict_sub(1)
    .strict_mul(8)
    .strict_add(8usize.strict_sub(first.leading_zeros() as usize))
}

fn unsigned_be_mod_u64(bytes: &[u8], modulus: u64) -> u64 {
  let mut remainder = 0u128;
  let modulus = u128::from(modulus);
  for &byte in bytes {
    remainder = (remainder << 8).strict_add(u128::from(byte)).rem_euclid(modulus);
  }
  split_u128(remainder).0
}

pub(super) fn parse_spki_der(der: &[u8]) -> Result<(&[u8], &[u8]), RsaKeyError> {
  let mut root = DerReader::new(der);
  let spki = root.read_constructed(TAG_SEQUENCE)?;
  root.finish()?;

  let mut spki = DerReader::new(spki);
  let algorithm = spki.read_constructed(TAG_SEQUENCE)?;
  let subject_public_key = spki.read_primitive(TAG_BIT_STRING)?;
  spki.finish()?;

  let (&unused_bits, public_key_der) = subject_public_key.split_first().ok_or(RsaKeyError::MalformedDer)?;
  if unused_bits != 0 || public_key_der.is_empty() {
    return Err(RsaKeyError::MalformedDer);
  }

  Ok((algorithm, public_key_der))
}

fn parse_pkcs1_public_key_der_parts<'a>(
  der: &'a [u8],
  policy: &RsaPublicKeyPolicy,
) -> Result<(&'a [u8], usize, RsaPublicExponent), RsaKeyError> {
  if policy.min_modulus_bits > policy.max_modulus_bits {
    return Err(RsaKeyError::InvalidModulus);
  }

  let mut root = DerReader::new(der);
  let public_key = root.read_constructed(TAG_SEQUENCE)?;
  root.finish()?;

  let mut public_key = DerReader::new(public_key);
  let modulus = read_positive_integer(public_key.read_primitive(TAG_INTEGER)?)?;
  let exponent = parse_public_exponent(read_positive_integer(public_key.read_primitive(TAG_INTEGER)?)?, policy)?;
  public_key.finish()?;

  let modulus_bits = validate_modulus(modulus, policy)?;
  Ok((modulus, modulus_bits, exponent))
}

pub(super) fn read_positive_integer(bytes: &[u8]) -> Result<&[u8], RsaKeyError> {
  let (&first, rest) = bytes.split_first().ok_or(RsaKeyError::MalformedDer)?;

  if first == 0 {
    let Some((&next, _)) = rest.split_first() else {
      return Ok(bytes);
    };
    if next & 0x80 == 0 {
      return Err(RsaKeyError::MalformedDer);
    }
    return Ok(rest);
  }

  if first & 0x80 != 0 {
    return Err(RsaKeyError::MalformedDer);
  }

  Ok(bytes)
}

fn parse_public_exponent(bytes: &[u8], policy: &RsaPublicKeyPolicy) -> Result<RsaPublicExponent, RsaKeyError> {
  if bytes.len() > core::mem::size_of::<u64>() {
    return Err(RsaKeyError::InvalidPublicExponent);
  }

  let mut value = 0u64;
  for &byte in bytes {
    value = (value << 8) | u64::from(byte);
  }

  if value < 3 || value & 1 == 0 || !policy.exponent_policy.accepts(value) {
    return Err(RsaKeyError::InvalidPublicExponent);
  }

  Ok(RsaPublicExponent(value))
}

pub(super) fn validate_modulus(modulus: &[u8], policy: &RsaPublicKeyPolicy) -> Result<usize, RsaKeyError> {
  let (&first, _) = modulus.split_first().ok_or(RsaKeyError::InvalidModulus)?;
  let Some(&last) = modulus.last() else {
    return Err(RsaKeyError::InvalidModulus);
  };

  if first == 0 || last & 1 == 0 {
    return Err(RsaKeyError::InvalidModulus);
  }

  let modulus_bits = modulus
    .len()
    .strict_sub(1)
    .strict_mul(8)
    .strict_add(8usize.strict_sub(first.leading_zeros() as usize));

  if modulus_bits < policy.min_modulus_bits || modulus_bits > policy.max_modulus_bits {
    return Err(RsaKeyError::InvalidModulus);
  }

  Ok(modulus_bits)
}

pub(super) fn ct_slices_eq_public_shape(left: &[u8], right: &[u8]) -> bool {
  ct::public_len_eq(left, right).declassify()
}

fn product_matches_unsigned_be_fixed(left: &[u8], right: &[u8], expected: &[u8]) -> bool {
  let mut product = vec![0u8; expected.len()];
  let imported = private_import_product_unsigned_be_to_fixed(left, right, &mut product).is_ok();
  let matched = imported & ct::public_len_eq(&product, expected).declassify();
  ct::zeroize(&mut product);
  matched
}

fn ct_eq_left_padded_unsigned_be(value: &[u8], expected: &[u8]) -> bool {
  if value.len() > expected.len() {
    return false;
  }
  let mut padded = vec![0u8; expected.len()];
  let offset = expected.len().strict_sub(value.len());
  let Some(dst) = padded.get_mut(offset..) else {
    return false;
  };
  dst.copy_from_slice(value);
  let eq = ct::public_len_eq(&padded, expected);
  ct::zeroize(&mut padded);
  eq.declassify()
}

#[cfg(feature = "getrandom")]
fn private_import_product_unsigned_be(left: &[u8], right: &[u8]) -> Option<SecretBigEndianBuffer> {
  let left = SecretLimbs::from_be(left);
  let right = SecretLimbs::from_be(right);
  let mut product = SecretLimbs::zeroed(left.as_slice().len().strict_add(right.as_slice().len()));

  for (left_index, &left_limb) in left.as_slice().iter().enumerate() {
    let mut carry = 0u64;
    for (right_index, &right_limb) in right.as_slice().iter().enumerate() {
      let index = left_index.strict_add(right_index);
      let limb = product.as_mut_slice().get_mut(index)?;
      let acc = u128::from(*limb)
        .strict_add(u128::from(left_limb).strict_mul(u128::from(right_limb)))
        .strict_add(u128::from(carry));
      let (acc_low, acc_high) = split_u128(acc);
      *limb = acc_low;
      carry = acc_high;
    }

    let index = left_index.strict_add(right.as_slice().len());
    for limb in product.as_mut_slice().get_mut(index..)?.iter_mut() {
      let acc = u128::from(*limb).strict_add(u128::from(carry));
      let (acc_low, acc_high) = split_u128(acc);
      *limb = acc_low;
      carry = acc_high;
    }
    if carry != 0 {
      return None;
    }
  }

  Some(private_import_limbs_to_canonical_be(product.as_slice()))
}

fn private_import_product_unsigned_be_to_fixed(left: &[u8], right: &[u8], out: &mut [u8]) -> Result<(), RsaKeyError> {
  let left_limb_count = left.len().strict_add(7) / 8;
  let right_limb_count = right.len().strict_add(7) / 8;
  let out_limb_count = out.len().strict_add(7) / 8;
  if left_limb_count.strict_add(right_limb_count) > out_limb_count {
    return Err(RsaKeyError::InvalidModulus);
  }

  let left = SecretLimbs::from_be(left);
  let right = SecretLimbs::from_be(right);
  let mut product = SecretLimbs::zeroed(out_limb_count);

  for (left_index, &left_limb) in left.as_slice().iter().enumerate() {
    let mut carry = 0u64;
    for (right_index, &right_limb) in right.as_slice().iter().enumerate() {
      let index = left_index.strict_add(right_index);
      let Some(limb) = product.as_mut_slice().get_mut(index) else {
        return Err(RsaKeyError::InvalidModulus);
      };
      let acc = u128::from(*limb)
        .strict_add(u128::from(left_limb).strict_mul(u128::from(right_limb)))
        .strict_add(u128::from(carry));
      let (acc_low, acc_high) = split_u128(acc);
      *limb = acc_low;
      carry = acc_high;
    }

    let index = left_index.strict_add(right.as_slice().len());
    let Some(carry_limbs) = product.as_mut_slice().get_mut(index..) else {
      return Err(RsaKeyError::InvalidModulus);
    };
    for limb in carry_limbs {
      let acc = u128::from(*limb).strict_add(u128::from(carry));
      let (acc_low, acc_high) = split_u128(acc);
      *limb = acc_low;
      carry = acc_high;
    }
    if carry != 0 {
      return Err(RsaKeyError::InvalidModulus);
    }
  }

  let mut full = vec![0u8; out_limb_count.strict_mul(8)];
  limbs_to_be(product.as_slice(), &mut full);
  let excess = full.len().strict_sub(out.len());
  let Some(prefix) = full.get(..excess) else {
    ct::zeroize(&mut full);
    return Err(RsaKeyError::InvalidModulus);
  };
  if !is_zero_unsigned_be(prefix) {
    ct::zeroize(&mut full);
    return Err(RsaKeyError::InvalidModulus);
  }
  let Some(suffix) = full.get(excess..) else {
    ct::zeroize(&mut full);
    return Err(RsaKeyError::InvalidModulus);
  };
  out.copy_from_slice(suffix);
  ct::zeroize(&mut full);
  Ok(())
}

#[cfg(feature = "getrandom")]
fn private_import_decrement_unsigned_be(bytes: &[u8]) -> Result<SecretBigEndianBuffer, RsaKeyError> {
  let mut out = vec![0u8; bytes.len()];
  private_import_decrement_unsigned_be_to_fixed(bytes, &mut out)?;
  Ok(private_import_canonical_unsigned_be(out))
}

fn private_import_decrement_unsigned_be_to_fixed(bytes: &[u8], out: &mut [u8]) -> Result<(), RsaKeyError> {
  if bytes.is_empty() || bytes.len() != out.len() || is_zero_unsigned_be(bytes) || bytes == [1] {
    return Err(RsaKeyError::InvalidModulus);
  }

  out.copy_from_slice(bytes);
  let mut borrow = 1u8;
  for byte in out.iter_mut().rev() {
    let (difference, overflow) = byte.overflowing_sub(borrow);
    *byte = difference;
    borrow = u8::from(overflow);
  }

  if borrow == 0 {
    Ok(())
  } else {
    Err(RsaKeyError::InvalidModulus)
  }
}

#[cfg(feature = "getrandom")]
fn private_import_unsigned_be_mod(value: &[u8], modulus: &[u8]) -> SecretBigEndianBuffer {
  if is_zero_unsigned_be(modulus) {
    return SecretBigEndianBuffer::new(vec![0]);
  }

  let modulus = SecretLimbs::from_be(modulus);
  let mut remainder = SecretLimbs::zeroed(modulus.as_slice().len());
  for &byte in value {
    for bit in (0..8).rev() {
      double_mod_in_place(remainder.as_mut_slice(), modulus.as_slice());
      add_bit_mod_in_place(remainder.as_mut_slice(), modulus.as_slice(), (byte >> bit) & 1);
    }
  }

  private_import_limbs_to_canonical_be(remainder.as_slice())
}

#[inline(never)]
pub(super) fn private_import_unsigned_be_mod_to_len(
  value: &[u8],
  modulus: &[u8],
  out: &mut [u8],
) -> Result<(), RsaKeyError> {
  if modulus.is_empty() || modulus.len() != out.len() || is_zero_unsigned_be(modulus) {
    return Err(RsaKeyError::InvalidModulus);
  }

  let modulus = SecretLimbs::from_be(modulus);
  let mut remainder = SecretLimbs::zeroed(modulus.as_slice().len());
  for &byte in value {
    for bit in (0..8).rev() {
      double_mod_in_place(remainder.as_mut_slice(), modulus.as_slice());
      add_bit_mod_in_place(remainder.as_mut_slice(), modulus.as_slice(), (byte >> bit) & 1);
    }
  }

  limbs_to_be(remainder.as_slice(), out);
  Ok(())
}

pub(super) fn private_import_unsigned_be_mod_to_fixed(
  value: &[u8],
  modulus: &RsaPublicModulus,
  out: &mut [u8],
  scratch: &mut RsaPrivateExponentScratch,
) -> Result<(), RsaPrivateOpError> {
  if out.len() != modulus.bytes.len() {
    return Err(RsaPrivateOpError::InvalidLength);
  }
  let workspace = scratch.workspace(modulus.limbs.len())?;

  let remainder = workspace.representative;
  remainder.fill(0);
  for &byte in value {
    for bit in (0..8).rev() {
      double_mod_in_place(remainder, &modulus.limbs);
      add_bit_mod_in_place(remainder, &modulus.limbs, (byte >> bit) & 1);
    }
  }

  limbs_to_be(remainder, out);
  Ok(())
}

#[cfg(feature = "getrandom")]
fn private_import_limbs_to_canonical_be(limbs: &[u64]) -> SecretBigEndianBuffer {
  let mut bytes = vec![0u8; limbs.len().strict_mul(8)];
  limbs_to_be(limbs, &mut bytes);
  private_import_canonical_unsigned_be(bytes)
}

#[cfg(feature = "getrandom")]
pub(super) fn private_import_canonical_unsigned_be(mut bytes: Vec<u8>) -> SecretBigEndianBuffer {
  let first_nonzero = bytes.iter().position(|&byte| byte != 0);
  let canonical = match first_nonzero {
    Some(0) => core::mem::take(&mut bytes),
    Some(index) => bytes.get(index..).map_or_else(|| vec![0], <[u8]>::to_vec),
    None => vec![0],
  };
  ct::zeroize(&mut bytes);
  SecretBigEndianBuffer::new(canonical)
}

impl MalformedDer for RsaKeyError {
  const MALFORMED_DER: Self = Self::MalformedDer;
}

#[cfg(test)]
mod tests;
