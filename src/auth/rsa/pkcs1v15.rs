//! PKCS #1 v1.5 signature encoding and public-key encryption.

use super::*;

impl RsaPrivateKey {
  /// Sign a message using RSASSA-PKCS1-v1_5 and OS-backed blinding.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, the key is too
  /// small for the selected profile, or the post-signing public fault check
  /// fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pkcs1v15(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    self.sign_signature(RsaSignatureProfile::pkcs1v15(profile), message, out)
  }

  /// Sign a message using RSASSA-PKCS1-v1_5, OS-backed blinding, and caller-owned scratch.
  ///
  /// Reusing scratch avoids top-level private-operation temporary allocation
  /// after setup.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, `scratch` was
  /// allocated for a different modulus width, the key is too small for the
  /// selected profile, or the post-signing public fault check fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pkcs1v15_with_scratch(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    self.sign_signature_with_scratch(RsaSignatureProfile::pkcs1v15(profile), message, out, scratch)
  }

  /// Sign a message using RSASSA-PKCS1-v1_5 with caller-supplied blinding.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if lengths are invalid, the blinding pair is
  /// invalid, the key is too small, or the post-signing public fault check fails.
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pkcs1v15_with_blinding_factor(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let result = self
      .components
      .sign_pkcs1v15_with_blinding_factor(profile, message, blinding, out);
    clear_output_on_error(result, out)
  }

  /// Sign using RSASSA-PKCS1-v1_5 with caller-supplied blinding and scratch.
  ///
  /// Reusing scratch avoids top-level private-operation temporary allocation
  /// after setup. The blinding-pair requirements are the same as
  /// [`Self::sign_pkcs1v15_with_blinding_factor`].
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if lengths are invalid, `scratch` was
  /// allocated for a different modulus width, the blinding pair is invalid, the
  /// key is too small, or the post-signing public fault check fails.
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pkcs1v15_with_blinding_factor_and_scratch(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let result = self
      .components
      .sign_pkcs1v15_with_blinding_factor_and_scratch(profile, message, blinding, out, scratch);
    clear_output_on_error(result, out)
  }
}

impl RsaPrivateKeyComponents {
  pub(super) fn sign_pkcs1v15_with_blinding_factor(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let mut encoded = vec![0u8; self.public.modulus().len()];
    let result = match profile {
      RsaPkcs1v15Profile::Sha256 => encode_pkcs1v15::<Sha256>(message, SHA256_DIGEST_INFO_PREFIX, &mut encoded),
      RsaPkcs1v15Profile::Sha384 => encode_pkcs1v15::<Sha384>(message, SHA384_DIGEST_INFO_PREFIX, &mut encoded),
      RsaPkcs1v15Profile::Sha512 => encode_pkcs1v15::<Sha512>(message, SHA512_DIGEST_INFO_PREFIX, &mut encoded),
    }
    .and_then(|()| self.sign_encoded_message_with_blinding_factor(&encoded, blinding, out));
    ct::zeroize(&mut encoded);
    result
  }

  pub(super) fn sign_pkcs1v15_with_blinding_factor_and_scratch(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    scratch.ensure_len(self.public.modulus().len())?;
    let result = match profile {
      RsaPkcs1v15Profile::Sha256 => {
        encode_pkcs1v15::<Sha256>(message, SHA256_DIGEST_INFO_PREFIX, scratch.encoded.as_mut_slice())
      }
      RsaPkcs1v15Profile::Sha384 => {
        encode_pkcs1v15::<Sha384>(message, SHA384_DIGEST_INFO_PREFIX, scratch.encoded.as_mut_slice())
      }
      RsaPkcs1v15Profile::Sha512 => {
        encode_pkcs1v15::<Sha512>(message, SHA512_DIGEST_INFO_PREFIX, scratch.encoded.as_mut_slice())
      }
    }
    .and_then(|()| self.sign_encoded_message_with_blinding_factor_and_scratch(blinding, out, scratch));
    scratch.clear();
    result
  }

  pub(super) fn sign_pkcs1v15_with_stored_blinding_and_scratch(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    self
      .encode_pkcs1v15_into_scratch(profile, message, scratch)
      .and_then(|()| self.sign_encoded_message_with_stored_blinding_and_scratch(out, scratch))
  }

  /// Write the EMSA-PKCS1-v1_5 encoding of `message` into `scratch.encoded`.
  pub(super) fn encode_pkcs1v15_into_scratch(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    scratch.ensure_len(self.public.modulus().len())?;
    match profile {
      RsaPkcs1v15Profile::Sha256 => {
        encode_pkcs1v15::<Sha256>(message, SHA256_DIGEST_INFO_PREFIX, scratch.encoded.as_mut_slice())
      }
      RsaPkcs1v15Profile::Sha384 => {
        encode_pkcs1v15::<Sha384>(message, SHA384_DIGEST_INFO_PREFIX, scratch.encoded.as_mut_slice())
      }
      RsaPkcs1v15Profile::Sha512 => {
        encode_pkcs1v15::<Sha512>(message, SHA512_DIGEST_INFO_PREFIX, scratch.encoded.as_mut_slice())
      }
    }
  }
}

impl RsaPublicKey {
  /// Encrypt a message using legacy RSAES-PKCS1-v1_5 and OS-backed randomness.
  ///
  /// Prefer OAEP for new protocols. `out` must be exactly
  /// [`Self::modulus`]`.len()` bytes.
  ///
  /// # Errors
  ///
  /// Returns [`RsaEncryptionError`] if entropy is unavailable, the message is
  /// too long for this key, `out` has the wrong length, or the RSA public
  /// operation fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA encryption failure must be checked; a dropped Result silently discards ciphertext"]
  pub fn encrypt_pkcs1v15(&self, message: &[u8], out: &mut [u8]) -> Result<(), RsaEncryptionError> {
    self.encrypt_pkcs1v15_with_random_fill(message, out, |random| {
      getrandom::fill(random).map_err(|_| RsaEncryptionError::EntropyUnavailable)
    })
  }

  /// Encrypt using legacy RSAES-PKCS1-v1_5, OS-backed randomness, and scratch.
  ///
  /// Reusing scratch avoids heap allocation after setup. Prefer OAEP for new
  /// protocols.
  ///
  /// # Errors
  ///
  /// Returns [`RsaEncryptionError`] if entropy is unavailable, lengths are
  /// invalid, the message is too long, or the RSA public operation fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA encryption failure must be checked; a dropped Result silently discards ciphertext"]
  pub fn encrypt_pkcs1v15_with_scratch(
    &self,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaEncryptionError> {
    self.encrypt_pkcs1v15_with_random_fill_and_scratch(message, out, scratch, |random| {
      getrandom::fill(random).map_err(|_| RsaEncryptionError::EntropyUnavailable)
    })
  }

  /// Encrypt using legacy RSAES-PKCS1-v1_5 and caller-supplied randomness.
  ///
  /// Prefer OAEP for new protocols. `fill_random` must write fresh,
  /// unpredictable bytes on every call; deterministic or repeated output makes
  /// RSA encryption deterministic. The closure may be called more than once
  /// when zero bytes are sampled for the PKCS #1 v1.5 padding string.
  ///
  /// # Errors
  ///
  /// Returns [`RsaEncryptionError`] if the random fill closure fails, lengths
  /// are invalid, the message is too long, or the RSA public operation fails.
  #[must_use = "RSA encryption failure must be checked; a dropped Result silently discards ciphertext"]
  pub fn encrypt_pkcs1v15_with_random_fill(
    &self,
    message: &[u8],
    out: &mut [u8],
    fill_random: impl FnMut(&mut [u8]) -> Result<(), RsaEncryptionError>,
  ) -> Result<(), RsaEncryptionError> {
    let mut scratch = RsaPublicScratch::new(self);
    self.encrypt_pkcs1v15_with_random_fill_and_scratch(message, out, &mut scratch, fill_random)
  }

  /// Encrypt using legacy RSAES-PKCS1-v1_5, caller-supplied randomness, and scratch.
  ///
  /// Reusing scratch avoids heap allocation after setup. `fill_random` must
  /// write fresh, unpredictable bytes on every call; deterministic or repeated
  /// output makes RSA encryption deterministic.
  ///
  /// # Errors
  ///
  /// Returns [`RsaEncryptionError`] if the random fill closure fails, lengths
  /// are invalid, the message is too long, or the RSA public operation fails.
  #[must_use = "RSA encryption failure must be checked; a dropped Result silently discards ciphertext"]
  pub fn encrypt_pkcs1v15_with_random_fill_and_scratch(
    &self,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
    mut fill_random: impl FnMut(&mut [u8]) -> Result<(), RsaEncryptionError>,
  ) -> Result<(), RsaEncryptionError> {
    if out.len() != self.modulus().len()
      || scratch.limb_count != self.modulus.limbs.len()
      || scratch.byte_count != self.modulus().len()
    {
      return clear_output_on_error(Err(RsaEncryptionError::InvalidLength), out);
    }
    let ps_len = match pkcs1v15_encryption_padding_len(self.modulus().len(), message.len()) {
      Ok(ps_len) => ps_len,
      Err(error) => return clear_output_on_error(Err(error), out),
    };
    let (arithmetic_scratch, encoded, seed, _) = scratch.split_all();
    let result = match seed.get_mut(..ps_len) {
      Some(padding_seed) => fill_pkcs1v15_nonzero_padding_with(padding_seed, &mut fill_random)
        .and_then(|()| encode_pkcs1v15_encryption_with_seed(message, padding_seed, encoded))
        .and_then(|()| {
          self
            .modulus
            .public_operation_by_division(self.exponent, encoded, out, arithmetic_scratch)
            .map_err(|_| RsaEncryptionError::PublicOperationFailed)
        }),
      None => Err(RsaEncryptionError::InvalidLength),
    };
    ct::zeroize(encoded);
    ct::zeroize(seed);
    clear_output_on_error(result, out)
  }

  /// Verify an RSASSA-PKCS1-v1_5 signature under a typed SHA-2 profile.
  ///
  /// # Errors
  ///
  /// Returns an opaque [`VerificationError`] for every invalid signature or
  /// malformed encoded message. The `DigestInfo` must be exact DER for the
  /// selected SHA-2 profile; BER leniency, trailing data, short padding, and
  /// OID/hash mismatches are rejected.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_pkcs1v15(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), VerificationError> {
    let mut scratch = RsaPublicScratch::new(self);
    self.verify_pkcs1v15_with_scratch(profile, message, signature, &mut scratch)
  }

  /// Verify an RSASSA-PKCS1-v1_5 signature with caller-owned RSA scratch space.
  ///
  /// # Errors
  ///
  /// Returns an opaque [`VerificationError`] for every invalid signature.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_pkcs1v15_with_scratch(
    &self,
    profile: RsaPkcs1v15Profile,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    let (arithmetic_scratch, encoded, _, _) = scratch.split_all();
    self
      .modulus
      .public_operation_by_division(self.exponent, signature, encoded, arithmetic_scratch)
      .map_err(|_| VerificationError::new())?;

    match profile {
      RsaPkcs1v15Profile::Sha256 => verify_pkcs1v15_encoded::<Sha256>(message, encoded, SHA256_DIGEST_INFO_PREFIX),
      RsaPkcs1v15Profile::Sha384 => verify_pkcs1v15_encoded::<Sha384>(message, encoded, SHA384_DIGEST_INFO_PREFIX),
      RsaPkcs1v15Profile::Sha512 => verify_pkcs1v15_encoded::<Sha512>(message, encoded, SHA512_DIGEST_INFO_PREFIX),
    }
  }
}

fn encode_pkcs1v15<D>(message: &[u8], digest_info_prefix: &[u8], out: &mut [u8]) -> Result<(), RsaPrivateOpError>
where
  D: Digest,
{
  let digest = D::digest(message);
  let digest_info_len = digest_info_prefix.len().strict_add(D::OUTPUT_SIZE);
  let minimum_len = 11usize.strict_add(digest_info_len);
  if out.len() < minimum_len {
    return Err(RsaPrivateOpError::MessageTooLong);
  }

  let ps_len = out.len().strict_sub(digest_info_len).strict_sub(3);
  out[0] = 0;
  out[1] = 1;
  out[2..2usize.strict_add(ps_len)].fill(0xff);
  let separator = 2usize.strict_add(ps_len);
  out[separator] = 0;
  let digest_info = separator.strict_add(1);
  out[digest_info..digest_info.strict_add(digest_info_prefix.len())].copy_from_slice(digest_info_prefix);
  out[digest_info.strict_add(digest_info_prefix.len())..].copy_from_slice(digest.as_ref());
  Ok(())
}

fn pkcs1v15_encryption_padding_len(key_len: usize, message_len: usize) -> Result<usize, RsaEncryptionError> {
  key_len
    .checked_sub(message_len)
    .and_then(|len| len.checked_sub(3))
    .filter(|&ps_len| ps_len >= 8)
    .ok_or(RsaEncryptionError::MessageTooLong)
}

fn fill_pkcs1v15_nonzero_padding_with(
  out: &mut [u8],
  mut fill_random: impl FnMut(&mut [u8]) -> Result<(), RsaEncryptionError>,
) -> Result<(), RsaEncryptionError> {
  fill_random(out)?;
  for byte in out.iter_mut() {
    let mut retries = 0usize;
    while *byte == 0 {
      if retries == PKCS1V15_NONZERO_PADDING_MAX_RETRIES {
        return Err(RsaEncryptionError::EntropyUnavailable);
      }
      fill_random(core::slice::from_mut(byte))?;
      retries = retries.strict_add(1);
    }
  }
  Ok(())
}

fn encode_pkcs1v15_encryption_with_seed(message: &[u8], seed: &[u8], out: &mut [u8]) -> Result<(), RsaEncryptionError> {
  let ps_len = pkcs1v15_encryption_padding_len(out.len(), message.len())?;
  if seed.len() != ps_len {
    return Err(RsaEncryptionError::InvalidLength);
  }
  if seed.contains(&0) {
    return Err(RsaEncryptionError::InvalidLength);
  }

  out[0] = 0;
  out[1] = 2;
  out[2..2usize.strict_add(ps_len)].copy_from_slice(seed);
  let separator = 2usize.strict_add(ps_len);
  out[separator] = 0;
  out[separator.strict_add(1)..].copy_from_slice(message);
  Ok(())
}

pub(super) fn verify_pkcs1v15_encoded<D>(
  message: &[u8],
  encoded: &[u8],
  digest_info_prefix: &[u8],
) -> Result<(), VerificationError>
where
  D: Digest,
{
  let digest = D::digest(message);
  let digest_info_len = digest_info_prefix.len().strict_add(D::OUTPUT_SIZE);
  if encoded.len() < 11usize.strict_add(digest_info_len) {
    return Err(VerificationError::new());
  }

  let separator_index = encoded.len().strict_sub(digest_info_len).strict_sub(1);
  let padding = encoded.get(2..separator_index).ok_or_else(VerificationError::new)?;
  let digest_info = encoded
    .get(separator_index.strict_add(1)..)
    .ok_or_else(VerificationError::new)?;
  let (prefix, value) = digest_info.split_at(digest_info_prefix.len());

  let mut valid = encoded.first().copied() == Some(0x00);
  valid &= encoded.get(1).copied() == Some(0x01);
  valid &= padding.len() >= 8;
  for &byte in padding {
    valid &= byte == 0xff;
  }
  valid &= encoded.get(separator_index).copied() == Some(0x00);
  valid &= ct::public_len_eq(prefix, digest_info_prefix).declassify();
  valid &= ct::public_len_eq(value, digest.as_ref()).declassify();

  if valid { Ok(()) } else { Err(VerificationError::new()) }
}

#[cfg(test)]
mod tests;
