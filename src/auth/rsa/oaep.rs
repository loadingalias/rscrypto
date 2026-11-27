//! OAEP encoding, encryption, and decryption.

use super::*;

/// RSAES-OAEP padding bytes for `profile`: `2 * hLen + 2` (RFC 8017 7.1.1).
const fn rsaes_oaep_overhead(profile: RsaOaepProfile) -> usize {
  profile.digest_len().strict_mul(2).strict_add(2)
}

impl RsaPrivateKey {
  /// Decrypt an RSAES-OAEP ciphertext using OS-backed blinding.
  ///
  /// `out` must hold at least `k - 2 * profile.digest_len() - 2` bytes for a
  /// `k`-byte modulus, the longest message the profile admits. A shorter `out`
  /// fails with [`RsaPrivateOpError::InvalidLength`] before decryption, whatever
  /// the ciphertext. On success the plaintext is `out[..len]`, where `len` is the
  /// returned length. On failure `out` is zeroed.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, the ciphertext is
  /// malformed, or OAEP decoding fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA decryption failure must be checked; a dropped Result silently discards plaintext"]
  pub fn decrypt_oaep(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    ciphertext: &[u8],
    out: &mut [u8],
  ) -> Result<usize, RsaPrivateOpError> {
    self.reject_short_decryption_output(rsaes_oaep_overhead(profile), out)?;
    let result = self.components.random_blinding_factor().and_then(|blinding| {
      self.components.decrypt_oaep_with_blinding_factor(
        profile,
        label,
        ciphertext,
        RsaBlindingPair::trusted(blinding.factor(), blinding.inverse()),
        out,
      )
    });
    clear_output_on_error(result, out)
  }

  /// Decrypt an RSAES-OAEP ciphertext using OS-backed blinding and caller-owned scratch.
  ///
  /// Reusing scratch avoids top-level private-operation temporary allocation
  /// after setup.
  ///
  /// `out` must hold at least `k - 2 * profile.digest_len() - 2` bytes for a
  /// `k`-byte modulus, the longest message the profile admits. A shorter `out`
  /// fails with [`RsaPrivateOpError::InvalidLength`] before decryption, whatever
  /// the ciphertext. On success the plaintext is `out[..len]`, where `len` is the
  /// returned length. On failure `out` is zeroed.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, `scratch` was
  /// allocated for a different modulus width, the ciphertext is malformed, or
  /// OAEP decoding fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA decryption failure must be checked; a dropped Result silently discards plaintext"]
  pub fn decrypt_oaep_with_scratch(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    ciphertext: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError> {
    self.reject_short_decryption_output(rsaes_oaep_overhead(profile), out)?;
    let result = self
      .components
      .random_blinding_factor_into_scratch(scratch)
      .and_then(|()| {
        self
          .components
          .decrypt_oaep_with_stored_blinding_and_scratch(profile, label, ciphertext, out, scratch)
      });
    scratch.clear();
    clear_decryption_output_on_error(result, out)
  }

  /// Decrypt an RSAES-OAEP ciphertext with caller-supplied blinding.
  ///
  /// The decrypted plaintext is written to `out`, and the plaintext length is
  /// returned. The label must match the label used by the encrypting side.
  ///
  /// `out` must hold at least `k - 2 * profile.digest_len() - 2` bytes for a
  /// `k`-byte modulus, the longest message the profile admits. A shorter `out`
  /// fails with [`RsaPrivateOpError::InvalidLength`] before decryption, whatever
  /// the ciphertext. On success the plaintext is `out[..len]`, where `len` is the
  /// returned length. On failure `out` is zeroed.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if lengths are invalid, the blinding pair is
  /// invalid, the ciphertext representative is out of range, or OAEP decoding
  /// fails.
  #[must_use = "RSA decryption failure must be checked; a dropped Result silently discards plaintext"]
  pub fn decrypt_oaep_with_blinding_factor(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    ciphertext: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
  ) -> Result<usize, RsaPrivateOpError> {
    self.reject_short_decryption_output(rsaes_oaep_overhead(profile), out)?;
    let result = self
      .components
      .decrypt_oaep_with_blinding_factor(profile, label, ciphertext, blinding, out);
    clear_decryption_output_on_error(result, out)
  }

  /// Decrypt RSAES-OAEP with caller-supplied blinding and scratch.
  ///
  /// Reusing scratch avoids top-level private-operation temporary allocation
  /// after setup. The blinding-pair requirements are the same as
  /// [`Self::decrypt_oaep_with_blinding_factor`].
  ///
  /// `out` must hold at least `k - 2 * profile.digest_len() - 2` bytes for a
  /// `k`-byte modulus, the longest message the profile admits. A shorter `out`
  /// fails with [`RsaPrivateOpError::InvalidLength`] before decryption, whatever
  /// the ciphertext. On success the plaintext is `out[..len]`, where `len` is the
  /// returned length. On failure `out` is zeroed.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if lengths are invalid, `scratch` was
  /// allocated for a different modulus width, the blinding pair is invalid, the
  /// ciphertext representative is out of range, or OAEP decoding fails.
  #[must_use = "RSA decryption failure must be checked; a dropped Result silently discards plaintext"]
  pub fn decrypt_oaep_with_blinding_factor_and_scratch(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    ciphertext: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError> {
    self.reject_short_decryption_output(rsaes_oaep_overhead(profile), out)?;
    let result = self
      .components
      .decrypt_oaep_with_blinding_factor_and_scratch(profile, label, ciphertext, blinding, out, scratch);
    clear_decryption_output_on_error(result, out)
  }
}

impl RsaPrivateKeyComponents {
  pub(super) fn decrypt_oaep_with_blinding_factor(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    ciphertext: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
  ) -> Result<usize, RsaPrivateOpError> {
    let mut encoded = vec![0u8; self.public.modulus().len()];
    let result = self
      .private_operation_with_blinding_factor(ciphertext, blinding, &mut encoded)
      .and_then(|()| match profile {
        RsaOaepProfile::Sha256 => decode_oaep::<Sha256>(label, &mut encoded, out),
        RsaOaepProfile::Sha384 => decode_oaep::<Sha384>(label, &mut encoded, out),
        RsaOaepProfile::Sha512 => decode_oaep::<Sha512>(label, &mut encoded, out),
      });
    ct::zeroize(&mut encoded);
    clear_decryption_output_on_error(result, out)
  }

  #[cfg(feature = "getrandom")]
  fn decrypt_oaep_with_stored_blinding_and_scratch(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    ciphertext: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError> {
    let result = scratch.ensure_len(self.public.modulus().len()).and_then(|()| {
      if ciphertext.len() != scratch.encoded.as_slice().len() {
        Err(RsaPrivateOpError::InvalidLength)
      } else {
        scratch.encoded.as_mut_slice().copy_from_slice(ciphertext);
        self
          .private_operation_from_scratch_encoded_with_stored_blinding(scratch)
          .map(|()| {
            scratch
              .encoded
              .as_mut_slice()
              .copy_from_slice(scratch.blinded_private_result.as_slice())
          })
          .and_then(|()| match profile {
            RsaOaepProfile::Sha256 => decode_oaep_with_masks::<Sha256>(
              label,
              scratch.encoded.as_mut_slice(),
              out,
              scratch.blinding_power.as_mut_slice(),
              scratch.blinded.as_mut_slice(),
            ),
            RsaOaepProfile::Sha384 => decode_oaep_with_masks::<Sha384>(
              label,
              scratch.encoded.as_mut_slice(),
              out,
              scratch.blinding_power.as_mut_slice(),
              scratch.blinded.as_mut_slice(),
            ),
            RsaOaepProfile::Sha512 => decode_oaep_with_masks::<Sha512>(
              label,
              scratch.encoded.as_mut_slice(),
              out,
              scratch.blinding_power.as_mut_slice(),
              scratch.blinded.as_mut_slice(),
            ),
          })
      }
    });
    clear_decryption_output_on_error(result, out)
  }

  pub(super) fn decrypt_oaep_with_blinding_factor_and_scratch(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    ciphertext: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<usize, RsaPrivateOpError> {
    let result = scratch.ensure_len(self.public.modulus().len()).and_then(|()| {
      if ciphertext.len() != scratch.encoded.as_slice().len() {
        Err(RsaPrivateOpError::InvalidLength)
      } else {
        scratch.encoded.as_mut_slice().copy_from_slice(ciphertext);
        self
          .private_operation_from_scratch_encoded(blinding, scratch)
          .map(|()| {
            scratch
              .encoded
              .as_mut_slice()
              .copy_from_slice(scratch.blinded_private_result.as_slice())
          })
          .and_then(|()| match profile {
            RsaOaepProfile::Sha256 => decode_oaep_with_masks::<Sha256>(
              label,
              scratch.encoded.as_mut_slice(),
              out,
              scratch.blinding_power.as_mut_slice(),
              scratch.blinded.as_mut_slice(),
            ),
            RsaOaepProfile::Sha384 => decode_oaep_with_masks::<Sha384>(
              label,
              scratch.encoded.as_mut_slice(),
              out,
              scratch.blinding_power.as_mut_slice(),
              scratch.blinded.as_mut_slice(),
            ),
            RsaOaepProfile::Sha512 => decode_oaep_with_masks::<Sha512>(
              label,
              scratch.encoded.as_mut_slice(),
              out,
              scratch.blinding_power.as_mut_slice(),
              scratch.blinded.as_mut_slice(),
            ),
          })
      }
    });
    scratch.clear();
    clear_decryption_output_on_error(result, out)
  }
}

impl RsaPublicKey {
  /// Encrypt a message using RSAES-OAEP and OS-backed randomness.
  ///
  /// `out` must be exactly [`Self::modulus`]`.len()` bytes. The label must
  /// match the label supplied to OAEP decryption.
  ///
  /// # Errors
  ///
  /// Returns [`RsaEncryptionError`] if entropy is unavailable, the message is
  /// too long for this key/profile, `out` has the wrong length, or the RSA
  /// public operation fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA encryption failure must be checked; a dropped Result silently discards ciphertext"]
  pub fn encrypt_oaep(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    message: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaEncryptionError> {
    self.encrypt_oaep_with_random_fill(profile, label, message, out, |random| {
      getrandom::fill(random).map_err(|_| RsaEncryptionError::EntropyUnavailable)
    })
  }

  /// Encrypt a message using RSAES-OAEP, OS-backed randomness, and caller-owned scratch.
  ///
  /// Reusing scratch avoids heap allocation after setup. The label must match
  /// the label supplied to OAEP decryption.
  ///
  /// # Errors
  ///
  /// Returns [`RsaEncryptionError`] if entropy is unavailable, the message is
  /// too long for this key/profile, `out` has the wrong length, `scratch` was
  /// allocated for a different modulus width, or the RSA public operation fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA encryption failure must be checked; a dropped Result silently discards ciphertext"]
  pub fn encrypt_oaep_with_scratch(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), RsaEncryptionError> {
    self.encrypt_oaep_with_random_fill_and_scratch(profile, label, message, out, scratch, |random| {
      getrandom::fill(random).map_err(|_| RsaEncryptionError::EntropyUnavailable)
    })
  }

  /// Encrypt a message using RSAES-OAEP and caller-supplied randomness.
  ///
  /// `fill_random` must write fresh, unpredictable bytes on every call;
  /// deterministic or repeated output makes RSA encryption deterministic. The
  /// label must match the label supplied to OAEP decryption.
  ///
  /// # Errors
  ///
  /// Returns [`RsaEncryptionError`] if the random fill closure fails, the
  /// message is too long for this key/profile, `out` has the wrong length, or
  /// the RSA public operation fails.
  #[must_use = "RSA encryption failure must be checked; a dropped Result silently discards ciphertext"]
  pub fn encrypt_oaep_with_random_fill(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
    message: &[u8],
    out: &mut [u8],
    fill_random: impl FnMut(&mut [u8]) -> Result<(), RsaEncryptionError>,
  ) -> Result<(), RsaEncryptionError> {
    let mut scratch = RsaPublicScratch::new(self);
    self.encrypt_oaep_with_random_fill_and_scratch(profile, label, message, out, &mut scratch, fill_random)
  }

  /// Encrypt a message using RSAES-OAEP, caller-supplied randomness, and scratch.
  ///
  /// Reusing scratch avoids heap allocation after setup. `fill_random` must
  /// write fresh, unpredictable bytes on every call; deterministic or repeated
  /// output makes RSA encryption deterministic. The label must match the label
  /// supplied to OAEP decryption.
  ///
  /// # Errors
  ///
  /// Returns [`RsaEncryptionError`] if the random fill closure fails, the
  /// message is too long for this key/profile, `out` has the wrong length,
  /// `scratch` was allocated for a different modulus width, or the RSA public
  /// operation fails.
  #[must_use = "RSA encryption failure must be checked; a dropped Result silently discards ciphertext"]
  pub fn encrypt_oaep_with_random_fill_and_scratch(
    &self,
    profile: RsaOaepProfile,
    label: &[u8],
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

    let mut seed = [0u8; Sha512::OUTPUT_SIZE];
    let seed_len = profile.digest_len();
    let Some(seed_bytes) = seed.get_mut(..seed_len) else {
      return clear_output_on_error(Err(RsaEncryptionError::InvalidLength), out);
    };
    let result = fill_random(seed_bytes).and_then(|()| {
      let (arithmetic_scratch, encoded, seed_mask, db_mask) = scratch.split_all();
      let result = match profile {
        RsaOaepProfile::Sha256 => {
          encode_oaep_with_masks::<Sha256>(label, message, seed_bytes, encoded, db_mask, seed_mask)
        }
        RsaOaepProfile::Sha384 => {
          encode_oaep_with_masks::<Sha384>(label, message, seed_bytes, encoded, db_mask, seed_mask)
        }
        RsaOaepProfile::Sha512 => {
          encode_oaep_with_masks::<Sha512>(label, message, seed_bytes, encoded, db_mask, seed_mask)
        }
      }
      .and_then(|()| {
        self
          .modulus
          .public_operation_by_division(self.exponent, encoded, out, arithmetic_scratch)
          .map_err(|_| RsaEncryptionError::PublicOperationFailed)
      });
      ct::zeroize(encoded);
      ct::zeroize(seed_mask);
      ct::zeroize(db_mask);
      result
    });
    ct::zeroize(&mut seed);
    clear_output_on_error(result, out)
  }
}

fn encode_oaep_with_masks<D>(
  label: &[u8],
  message: &[u8],
  seed: &[u8],
  out: &mut [u8],
  db_mask: &mut [u8],
  seed_mask: &mut [u8],
) -> Result<(), RsaEncryptionError>
where
  D: Digest,
{
  let h_len = D::OUTPUT_SIZE;
  if seed.len() != h_len || out.len() < h_len.strict_mul(2).strict_add(2) {
    return Err(RsaEncryptionError::InvalidLength);
  }
  let db_len = out.len().strict_sub(h_len).strict_sub(1);
  let Some(db_mask) = db_mask.get_mut(..db_len) else {
    return Err(RsaEncryptionError::InvalidLength);
  };
  let Some(seed_mask) = seed_mask.get_mut(..h_len) else {
    return Err(RsaEncryptionError::InvalidLength);
  };
  let ps_len = out
    .len()
    .checked_sub(message.len())
    .and_then(|len| len.checked_sub(h_len.strict_mul(2)))
    .and_then(|len| len.checked_sub(2))
    .ok_or(RsaEncryptionError::MessageTooLong)?;

  let label_hash = D::digest(label);
  let (leading, rest) = out.split_at_mut(1);
  leading[0] = 0;
  let (masked_seed, masked_db) = rest.split_at_mut(h_len);
  masked_seed.copy_from_slice(seed);
  masked_db[..h_len].copy_from_slice(label_hash.as_ref());
  masked_db[h_len..h_len.strict_add(ps_len)].fill(0);
  masked_db[h_len.strict_add(ps_len)] = 1;
  masked_db[h_len.strict_add(ps_len).strict_add(1)..].copy_from_slice(message);

  mgf1::<D>(seed, db_mask);
  for (byte, mask) in masked_db.iter_mut().zip(db_mask.iter().copied()) {
    *byte ^= mask;
  }

  mgf1::<D>(masked_db, seed_mask);
  for (byte, mask) in masked_seed.iter_mut().zip(seed_mask.iter().copied()) {
    *byte ^= mask;
  }
  Ok(())
}

fn decode_oaep<D>(label: &[u8], encoded: &mut [u8], out: &mut [u8]) -> Result<usize, RsaPrivateOpError>
where
  D: Digest,
{
  let h_len = D::OUTPUT_SIZE;
  let db_mask_len = encoded
    .len()
    .checked_sub(h_len)
    .and_then(|len| len.checked_sub(1))
    .ok_or(RsaPrivateOpError::DecryptionFailed)?;
  let mut seed_mask = vec![0u8; h_len];
  let mut db_mask = vec![0u8; db_mask_len];
  let result = decode_oaep_with_masks::<D>(label, encoded, out, &mut seed_mask, &mut db_mask);
  ct::zeroize(&mut seed_mask);
  ct::zeroize(&mut db_mask);
  result
}

fn decode_oaep_with_masks<D>(
  label: &[u8],
  encoded: &mut [u8],
  out: &mut [u8],
  seed_mask: &mut [u8],
  db_mask: &mut [u8],
) -> Result<usize, RsaPrivateOpError>
where
  D: Digest,
{
  let h_len = D::OUTPUT_SIZE;
  if encoded.len() < h_len.strict_mul(2).strict_add(2) {
    return Err(RsaPrivateOpError::DecryptionFailed);
  }
  let db_len = encoded.len().strict_sub(h_len).strict_sub(1);
  let Some(seed_mask) = seed_mask.get_mut(..h_len) else {
    return Err(RsaPrivateOpError::InvalidLength);
  };
  let Some(db_mask) = db_mask.get_mut(..db_len) else {
    return Err(RsaPrivateOpError::InvalidLength);
  };

  let label_hash = D::digest(label);
  let (leading, rest) = encoded.split_at_mut(1);
  let (masked_seed, masked_db) = rest.split_at_mut(h_len);
  let mut bad = u8::from(leading.first().copied() != Some(0));

  mgf1::<D>(masked_db, seed_mask);
  for (byte, mask) in masked_seed.iter_mut().zip(seed_mask.iter().copied()) {
    *byte ^= mask;
  }

  mgf1::<D>(masked_seed, db_mask);
  for (byte, mask) in masked_db.iter_mut().zip(db_mask.iter().copied()) {
    *byte ^= mask;
  }

  let (decoded_label_hash, rest) = masked_db.split_at(h_len);
  bad |= (!ct::public_len_eq(decoded_label_hash, label_hash.as_ref())).into_u8();

  let mut seen_separator = 0u8;
  let mut separator = 0usize;
  for (index, &byte) in rest.iter().enumerate() {
    let before_separator = seen_separator ^ 1;
    // SECURITY: Keep the octet classes opaque so LLVM cannot branch on the
    // decoded data block, as it did for the PKCS#1 v1.5 separator scan on RISC-V.
    let is_zero = core::hint::black_box(u8::from(byte == 0));
    let is_one = core::hint::black_box(u8::from(byte == 1));
    let separator_at_index = before_separator & is_one;
    let invalid_before_separator = before_separator & ((is_zero | is_one) ^ 1);
    bad |= invalid_before_separator;
    separator = ct_select_usize(separator, index, separator_at_index);
    seen_separator |= separator_at_index;
  }
  bad |= seen_separator ^ 1;

  if bad != 0 {
    return Err(RsaPrivateOpError::DecryptionFailed);
  }
  let message = &rest[separator.strict_add(1)..];
  if out.len() < message.len() {
    return Err(RsaPrivateOpError::DecryptionFailed);
  }
  out[..message.len()].copy_from_slice(message);
  Ok(message.len())
}

#[cfg(test)]
mod tests;
