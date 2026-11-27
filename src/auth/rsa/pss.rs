//! PSS encoding, signing, and verification.

use super::*;

impl RsaPrivateKey {
  /// Sign a message using RSASSA-PSS and caller-provided salt/blinding entropy.
  ///
  /// The salt length is the selected profile's digest length. The entropy
  /// request and failure contracts match
  /// [`Self::sign_signature_with_random_fill`].
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, lengths are
  /// invalid, the key is too small for the selected profile, no valid blinding
  /// factor is found within the retry bound, or the post-signing public fault
  /// check fails.
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pss_with_random_fill<E>(
    &self,
    profile: RsaPssProfile,
    message: &[u8],
    out: &mut [u8],
    fill_random: impl FnMut(&mut [u8]) -> Result<(), E>,
  ) -> Result<(), RsaPrivateOpError> {
    self.sign_signature_with_random_fill(RsaSignatureProfile::pss(profile), message, out, fill_random)
  }

  /// Sign using RSASSA-PSS, caller-provided entropy, and reusable scratch.
  ///
  /// The salt length is the selected profile's digest length. This path
  /// performs no steady-state allocation after `scratch` setup.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] under the same conditions as
  /// [`Self::sign_signature_with_random_fill_and_scratch`].
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pss_with_random_fill_and_scratch<E>(
    &self,
    profile: RsaPssProfile,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
    fill_random: impl FnMut(&mut [u8]) -> Result<(), E>,
  ) -> Result<(), RsaPrivateOpError> {
    self.sign_signature_with_random_fill_and_scratch(
      RsaSignatureProfile::pss(profile),
      message,
      out,
      scratch,
      fill_random,
    )
  }

  /// Sign a message using RSASSA-PSS and OS-backed random salt/blinding.
  ///
  /// The generated salt length is the selected profile's digest length.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, the key is too
  /// small for the selected profile, or the post-signing public fault check
  /// fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pss(&self, profile: RsaPssProfile, message: &[u8], out: &mut [u8]) -> Result<(), RsaPrivateOpError> {
    self.sign_pss_with_random_fill(profile, message, out, getrandom::fill)
  }

  /// Sign a message using RSASSA-PSS, OS-backed random salt/blinding, and caller-owned scratch.
  ///
  /// The generated salt length is the selected profile's digest length.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, `scratch` was
  /// allocated for a different modulus width, the key is too small for the
  /// selected profile, or the post-signing public fault check fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pss_with_scratch(
    &self,
    profile: RsaPssProfile,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    self.sign_pss_with_random_fill_and_scratch(profile, message, out, scratch, getrandom::fill)
  }

  /// Sign a message using RSASSA-PSS with an explicit random salt length.
  ///
  /// Use this after parsing protocol parameters that permit a salt length other
  /// than the digest length. TLS 1.3-style signatures should prefer
  /// [`Self::sign_pss`] or [`Self::sign_signature`].
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, the key is too
  /// small for the requested salt length, or the post-signing public fault
  /// check fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pss_with_salt_len(
    &self,
    profile: RsaPssProfile,
    salt_len: usize,
    message: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    self.sign_signature_with_random_fill(
      RsaSignatureProfile::pss_with_salt_len(profile, salt_len),
      message,
      out,
      getrandom::fill,
    )
  }

  /// Sign a message using RSASSA-PSS with an explicit random salt length and caller-owned scratch.
  ///
  /// Use this after parsing protocol parameters that permit a salt length other
  /// than the digest length. TLS 1.3-style signatures should prefer
  /// [`Self::sign_pss_with_scratch`] or [`Self::sign_signature_with_scratch`].
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, `scratch` was
  /// allocated for a different modulus width, the key is too small for the
  /// requested salt length, or the post-signing public fault check fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pss_with_salt_len_and_scratch(
    &self,
    profile: RsaPssProfile,
    salt_len: usize,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    self.sign_signature_with_random_fill_and_scratch(
      RsaSignatureProfile::pss_with_salt_len(profile, salt_len),
      message,
      out,
      scratch,
      getrandom::fill,
    )
  }

  /// Sign a message using RSASSA-PSS with explicit salt and caller-supplied blinding.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if lengths are invalid, the blinding pair is
  /// invalid, the key is too small for the salt/profile, or the post-signing
  /// public fault check fails.
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pss_with_salt_and_blinding_factor(
    &self,
    profile: RsaPssProfile,
    message: &[u8],
    salt: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let result = self
      .components
      .sign_pss_with_salt_and_blinding_factor(profile, message, salt, blinding, out);
    clear_output_on_error(result, out)
  }

  /// Sign using RSASSA-PSS with explicit salt, caller blinding, and scratch.
  ///
  /// Reusing scratch avoids top-level private-operation temporary allocation
  /// after setup. The salt and blinding-pair requirements are the same as
  /// [`Self::sign_pss_with_salt_and_blinding_factor`].
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if lengths are invalid, `scratch` was
  /// allocated for a different modulus width, the blinding pair is invalid, the
  /// key is too small for the salt/profile, or the post-signing public fault
  /// check fails.
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_pss_with_salt_and_blinding_factor_and_scratch(
    &self,
    profile: RsaPssProfile,
    message: &[u8],
    salt: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let result = self
      .components
      .sign_pss_with_salt_and_blinding_factor_and_scratch(profile, message, salt, blinding, out, scratch);
    clear_output_on_error(result, out)
  }
}

impl RsaPrivateKeyComponents {
  pub(super) fn sign_pss_with_salt_and_blinding_factor(
    &self,
    profile: RsaPssProfile,
    message: &[u8],
    salt: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let em_bits = self.public.modulus_bits().strict_sub(1);
    let em_len = em_bits.strict_add(7) / 8;
    let mut encoded = vec![0u8; self.public.modulus().len()];
    let leading = encoded.len().strict_sub(em_len);
    let Some(encoded_message) = encoded.get_mut(leading..) else {
      ct::zeroize(&mut encoded);
      return Err(RsaPrivateOpError::MessageTooLong);
    };
    let result = match profile {
      RsaPssProfile::Sha256 => encode_pss::<Sha256>(message, salt, em_bits, encoded_message),
      RsaPssProfile::Sha384 => encode_pss::<Sha384>(message, salt, em_bits, encoded_message),
      RsaPssProfile::Sha512 => encode_pss::<Sha512>(message, salt, em_bits, encoded_message),
    }
    .and_then(|()| self.sign_encoded_message_with_blinding_factor(&encoded, blinding, out));
    ct::zeroize(&mut encoded);
    result
  }

  pub(super) fn sign_pss_with_salt_and_blinding_factor_and_scratch(
    &self,
    profile: RsaPssProfile,
    message: &[u8],
    salt: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    scratch.ensure_len(self.public.modulus().len())?;
    let em_bits = self.public.modulus_bits().strict_sub(1);
    let em_len = em_bits.strict_add(7) / 8;
    let leading = scratch.encoded.as_slice().len().strict_sub(em_len);
    let Some(encoded_message) = scratch.encoded.as_mut_slice().get_mut(leading..) else {
      scratch.clear();
      return Err(RsaPrivateOpError::MessageTooLong);
    };
    let result = match profile {
      RsaPssProfile::Sha256 => encode_pss_with_mask::<Sha256>(
        message,
        salt,
        em_bits,
        encoded_message,
        scratch.blinding_power.as_mut_slice(),
      ),
      RsaPssProfile::Sha384 => encode_pss_with_mask::<Sha384>(
        message,
        salt,
        em_bits,
        encoded_message,
        scratch.blinding_power.as_mut_slice(),
      ),
      RsaPssProfile::Sha512 => encode_pss_with_mask::<Sha512>(
        message,
        salt,
        em_bits,
        encoded_message,
        scratch.blinding_power.as_mut_slice(),
      ),
    }
    .and_then(|()| self.sign_encoded_message_with_blinding_factor_and_scratch(blinding, out, scratch));
    scratch.clear();
    result
  }

  pub(super) fn sign_pss_with_stored_salt_and_blinding_and_scratch(
    &self,
    profile: RsaPssProfile,
    message: &[u8],
    salt_len: usize,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    self
      .encode_pss_with_stored_salt_into_scratch(profile, message, salt_len, scratch)
      .and_then(|()| self.sign_encoded_message_with_stored_blinding_and_scratch(out, scratch))
  }

  /// Write the EMSA-PSS encoding of `message` with the first `salt_len` bytes
  /// of `scratch.salt` into `scratch.encoded`.
  ///
  /// The MGF1 mask uses `scratch.blinding_power` as workspace, so any blinding
  /// power must be written after encoding.
  pub(super) fn encode_pss_with_stored_salt_into_scratch(
    &self,
    profile: RsaPssProfile,
    message: &[u8],
    salt_len: usize,
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    scratch.ensure_len(self.public.modulus().len())?;
    let em_bits = self.public.modulus_bits().strict_sub(1);
    let em_len = em_bits.strict_add(7) / 8;
    let leading = scratch.encoded.as_slice().len().strict_sub(em_len);
    let salt = scratch
      .salt
      .as_slice()
      .get(..salt_len)
      .ok_or(RsaPrivateOpError::MessageTooLong)?;
    let encoded_message = scratch
      .encoded
      .as_mut_slice()
      .get_mut(leading..)
      .ok_or(RsaPrivateOpError::MessageTooLong)?;
    match profile {
      RsaPssProfile::Sha256 => encode_pss_with_mask::<Sha256>(
        message,
        salt,
        em_bits,
        encoded_message,
        scratch.blinding_power.as_mut_slice(),
      ),
      RsaPssProfile::Sha384 => encode_pss_with_mask::<Sha384>(
        message,
        salt,
        em_bits,
        encoded_message,
        scratch.blinding_power.as_mut_slice(),
      ),
      RsaPssProfile::Sha512 => encode_pss_with_mask::<Sha512>(
        message,
        salt,
        em_bits,
        encoded_message,
        scratch.blinding_power.as_mut_slice(),
      ),
    }
  }
}

impl RsaPublicKey {
  /// Verify an RSASSA-PSS signature under a typed SHA-2 profile.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] for every invalid signature, malformed PSS
  /// encoding, unsupported representative, or policy mismatch. The error is
  /// intentionally opaque; protocol adapters must not split it into finer
  /// externally visible failure classes.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_pss(&self, profile: RsaPssProfile, message: &[u8], signature: &[u8]) -> Result<(), VerificationError> {
    let mut scratch = RsaPublicScratch::new(self);
    self.verify_pss_with_scratch(profile, message, signature, &mut scratch)
  }

  /// Verify an RSASSA-PSS signature with an explicit salt length.
  ///
  /// Use this when a protocol adapter has parsed RSASSA-PSS parameters that
  /// specify a salt length. Prefer [`Self::verify_pss`] for TLS 1.3-style
  /// profiles where the salt length must equal the digest length.
  ///
  /// # Errors
  ///
  /// Returns an opaque [`VerificationError`] for any invalid signature or
  /// impossible salt-length/profile combination.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_pss_with_salt_len(
    &self,
    profile: RsaPssProfile,
    salt_len: usize,
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), VerificationError> {
    if !self.pss_salt_len_is_possible(profile, salt_len) {
      return Err(VerificationError::new());
    }
    let mut scratch = RsaPublicScratch::new(self);
    self.verify_pss_with_salt_len_and_scratch(profile, salt_len, message, signature, &mut scratch)
  }

  /// Verify an RSASSA-PSS signature with caller-owned RSA scratch space.
  ///
  /// This avoids repeated RSA arithmetic scratch allocation in steady-state
  /// verification loops. PSS decoding still uses bounded temporary buffers
  /// sized by the modulus length.
  ///
  /// # Errors
  ///
  /// Returns an opaque [`VerificationError`] for any invalid signature.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_pss_with_scratch(
    &self,
    profile: RsaPssProfile,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    self.verify_pss_with_salt_len_and_scratch(profile, profile.digest_len(), message, signature, scratch)
  }

  /// Verify an RSASSA-PSS signature with explicit salt length and caller-owned scratch.
  ///
  /// # Errors
  ///
  /// Returns an opaque [`VerificationError`] for any invalid signature.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_pss_with_salt_len_and_scratch(
    &self,
    profile: RsaPssProfile,
    salt_len: usize,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    if !self.pss_salt_len_is_possible(profile, salt_len) {
      return Err(VerificationError::new());
    }
    let (arithmetic_scratch, encoded, db, db_mask) = scratch.split_all();
    self
      .modulus
      .public_operation_by_division(self.exponent, signature, encoded, arithmetic_scratch)
      .map_err(|_| VerificationError::new())?;

    let em_bits = self.modulus_bits().strict_sub(1);
    let em_len = em_bits.strict_add(7) / 8;
    let leading = encoded.len().strict_sub(em_len);
    let Some(prefix) = encoded.get(..leading) else {
      return Err(VerificationError::new());
    };
    if prefix.iter().any(|&byte| byte != 0) {
      return Err(VerificationError::new());
    }
    let Some(encoded) = encoded.get(leading..) else {
      return Err(VerificationError::new());
    };

    match profile {
      RsaPssProfile::Sha256 => {
        verify_pss_encoded_with_scratch::<Sha256>(message, encoded, em_bits, salt_len, db, db_mask)
      }
      RsaPssProfile::Sha384 => {
        verify_pss_encoded_with_scratch::<Sha384>(message, encoded, em_bits, salt_len, db, db_mask)
      }
      RsaPssProfile::Sha512 => {
        verify_pss_encoded_with_scratch::<Sha512>(message, encoded, em_bits, salt_len, db, db_mask)
      }
    }
  }

  /// Return `true` when this key is large enough for the PSS hash and salt length.
  #[inline]
  #[must_use]
  pub fn pss_salt_len_is_possible(&self, profile: RsaPssProfile, salt_len: usize) -> bool {
    let em_bits = self.modulus_bits().strict_sub(1);
    let em_len = em_bits.strict_add(7) / 8;
    profile
      .digest_len()
      .checked_add(salt_len)
      .and_then(|len| len.checked_add(2))
      .is_some_and(|minimum_len| em_len >= minimum_len)
  }
}

fn encode_pss<D>(message: &[u8], salt: &[u8], em_bits: usize, out: &mut [u8]) -> Result<(), RsaPrivateOpError>
where
  D: Digest,
{
  let h_len = D::OUTPUT_SIZE;
  let db_len = out
    .len()
    .checked_sub(h_len)
    .and_then(|len| len.checked_sub(1))
    .ok_or(RsaPrivateOpError::MessageTooLong)?;
  let mut db_mask = vec![0u8; db_len];
  let result = encode_pss_with_mask::<D>(message, salt, em_bits, out, &mut db_mask);
  ct::zeroize(&mut db_mask);
  result
}

fn encode_pss_with_mask<D>(
  message: &[u8],
  salt: &[u8],
  em_bits: usize,
  out: &mut [u8],
  db_mask: &mut [u8],
) -> Result<(), RsaPrivateOpError>
where
  D: Digest,
{
  let h_len = D::OUTPUT_SIZE;
  let em_len = out.len();
  let minimum_len = h_len
    .checked_add(salt.len())
    .and_then(|len| len.checked_add(2))
    .ok_or(RsaPrivateOpError::MessageTooLong)?;
  if em_len < minimum_len {
    return Err(RsaPrivateOpError::MessageTooLong);
  }

  let unused_bits = em_len
    .checked_mul(8)
    .and_then(|bits| bits.checked_sub(em_bits))
    .ok_or(RsaPrivateOpError::MessageTooLong)?;
  if unused_bits >= 8 {
    return Err(RsaPrivateOpError::MessageTooLong);
  }

  let m_hash = D::digest(message);
  let mut h = D::new();
  h.update(&[0u8; 8]);
  h.update(m_hash.as_ref());
  h.update(salt);
  let h = h.finalize();

  let db_len = em_len.strict_sub(h_len).strict_sub(1);
  let Some(db_mask) = db_mask.get_mut(..db_len) else {
    return Err(RsaPrivateOpError::InvalidLength);
  };
  let ps_len = db_len.strict_sub(salt.len()).strict_sub(1);
  let (db, h_and_trailer) = out.split_at_mut(db_len);
  db[..ps_len].fill(0);
  db[ps_len] = 1;
  db[ps_len.strict_add(1)..].copy_from_slice(salt);

  mgf1::<D>(h.as_ref(), db_mask);
  for (byte, mask) in db.iter_mut().zip(db_mask.iter().copied()) {
    *byte ^= mask;
  }
  if unused_bits > 0
    && let Some(first) = db.first_mut()
  {
    *first &= 0xffu8 >> unused_bits;
  }

  let (h_out, trailer) = h_and_trailer.split_at_mut(h_len);
  h_out.copy_from_slice(h.as_ref());
  trailer[0] = 0xbc;
  Ok(())
}

pub(super) fn verify_pss_encoded_with_scratch<D>(
  message: &[u8],
  encoded: &[u8],
  em_bits: usize,
  salt_len: usize,
  db: &mut [u8],
  db_mask: &mut [u8],
) -> Result<(), VerificationError>
where
  D: Digest,
{
  let h_len = D::OUTPUT_SIZE;
  let em_len = encoded.len();
  let min_len = h_len
    .checked_add(salt_len)
    .and_then(|len| len.checked_add(2))
    .ok_or_else(VerificationError::new)?;
  if em_len < min_len {
    return Err(VerificationError::new());
  }
  if encoded.last().copied() != Some(0xbc) {
    return Err(VerificationError::new());
  }

  let db_len = em_len.strict_sub(h_len).strict_sub(1);
  let (masked_db, h_and_trailer) = encoded.split_at(db_len);
  let h = h_and_trailer.get(..h_len).ok_or_else(VerificationError::new)?;
  let db = db.get_mut(..db_len).ok_or_else(VerificationError::new)?;
  let db_mask = db_mask.get_mut(..db_len).ok_or_else(VerificationError::new)?;

  let unused_bits = em_len
    .checked_mul(8)
    .and_then(|bits| bits.checked_sub(em_bits))
    .ok_or_else(VerificationError::new)?;
  if unused_bits >= 8 {
    return Err(VerificationError::new());
  }
  if unused_bits > 0 {
    let mask = 0xffu8 << 8usize.strict_sub(unused_bits);
    if masked_db.first().copied().unwrap_or(0) & mask != 0 {
      return Err(VerificationError::new());
    }
  }

  db.copy_from_slice(masked_db);
  db_mask.fill(0);
  mgf1::<D>(h, db_mask);
  for (dst, mask) in db.iter_mut().zip(db_mask.iter().copied()) {
    *dst ^= mask;
  }
  if unused_bits > 0 {
    let mask = 0xffu8 >> unused_bits;
    if let Some(first) = db.first_mut() {
      *first &= mask;
    }
  }

  let ps_len = em_len.strict_sub(h_len).strict_sub(salt_len).strict_sub(2);
  if db.get(..ps_len).is_none_or(|ps| ps.iter().any(|&byte| byte != 0)) {
    return Err(VerificationError::new());
  }
  if db.get(ps_len).copied() != Some(0x01) {
    return Err(VerificationError::new());
  }
  let salt_start = ps_len.strict_add(1);
  let salt = db.get(salt_start..).ok_or_else(VerificationError::new)?;
  if salt.len() != salt_len {
    return Err(VerificationError::new());
  }

  let m_hash = D::digest(message);
  let mut verifier = D::new();
  verifier.update(&[0u8; 8]);
  verifier.update(m_hash.as_ref());
  verifier.update(salt);
  let expected_h = verifier.finalize();

  if ct::public_len_eq(expected_h.as_ref(), h).declassify() {
    Ok(())
  } else {
    Err(VerificationError::new())
  }
}

#[cfg(test)]
mod tests;
