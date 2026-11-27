//! Blinding, scratch ownership, private operations, and the CRT fault check.

use super::*;

impl RsaPrivateKey {
  /// Reject `out` shorter than the longest message the padding admits.
  ///
  /// The check depends only on public lengths and runs before entropy use or
  /// private arithmetic. A check against the decoded message length would
  /// distinguish valid from invalid padding.
  pub(super) fn reject_short_decryption_output(
    &self,
    overhead: usize,
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    if out.len() < self.signature_len().saturating_sub(overhead) {
      ct::zeroize(out);
      return Err(RsaPrivateOpError::InvalidLength);
    }
    Ok(())
  }
}

impl RsaPrivateKey {
  /// Key-generation contract used by [`Self::generate`] and
  /// [`Self::generate_with_policy`].
  ///
  /// The contract is intentionally explicit: generated keys use probable-prime
  /// RSA generation following FIPS 186-5 Appendix A.1.3 in code. This is not a
  /// CMVP/FIPS 140-3 validation claim.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  pub const GENERATION_CONTRACT: RsaKeyGenerationContract = RsaKeyGenerationContract::Fips1865A13ProbablePrime;

  /// Allocate reusable scratch space for private operations.
  ///
  /// Use this with the caller-random or explicit-blinding `*_and_scratch`
  /// methods for steady-state private operations without top-level temporary
  /// buffer allocation.
  #[inline]
  #[must_use]
  pub fn private_scratch(&self) -> RsaPrivateScratch {
    RsaPrivateScratch::new(self)
  }
}

impl RsaPrivateScratch {
  /// Allocate scratch space for `key`.
  #[must_use]
  pub fn new(key: &RsaPrivateKey) -> Self {
    Self::new_components(&key.components)
  }

  fn new_components(components: &RsaPrivateKeyComponents) -> Self {
    let len = components.public.modulus().len();
    let mut one = SecretBigEndianBuffer::zeroed(len);
    if let Some(last) = one.as_mut_slice().last_mut() {
      *last = 1;
    }
    Self {
      encoded: SecretBigEndianBuffer::zeroed(len),
      salt: SecretBigEndianBuffer::zeroed(len),
      blinding_factor: SecretBigEndianBuffer::zeroed(len),
      blinding_inverse: SecretBigEndianBuffer::zeroed(len),
      inverse_mask: SecretBigEndianBuffer::zeroed(len),
      blinding_power: SecretBigEndianBuffer::zeroed(len),
      blinded: SecretBigEndianBuffer::zeroed(len),
      blinded_private_result: SecretBigEndianBuffer::zeroed(len),
      checked: SecretBigEndianBuffer::zeroed(len),
      one,
      public_scratch: RsaPublicScratch::new(&components.public),
      mul_scratch: RsaPrivateMulScratch::new(components.public.modulus.limbs.len()),
      inverse_scratch: RsaPrivateInverseScratch::new(components.public.modulus.bits),
      exponent_scratch: RsaPrivateExponentScratch::new(components.public.modulus.limbs.len()),
    }
  }

  pub(super) fn ensure_len(&self, len: usize) -> Result<(), RsaPrivateOpError> {
    if self.encoded.as_slice().len() == len
      && self.salt.as_slice().len() == len
      && self.blinding_factor.as_slice().len() == len
      && self.blinding_inverse.as_slice().len() == len
      && self.inverse_mask.as_slice().len() == len
      && self.blinding_power.as_slice().len() == len
      && self.blinded.as_slice().len() == len
      && self.blinded_private_result.as_slice().len() == len
      && self.checked.as_slice().len() == len
      && self.one.as_slice().len() == len
    {
      Ok(())
    } else {
      Err(RsaPrivateOpError::InvalidScratch)
    }
  }

  pub(super) fn set_one(&mut self) -> Result<(), RsaPrivateOpError> {
    ct::zeroize(self.one.as_mut_slice());
    let Some(last) = self.one.as_mut_slice().last_mut() else {
      return Err(RsaPrivateOpError::InvalidScratch);
    };
    *last = 1;
    Ok(())
  }

  pub(super) fn clear(&mut self) {
    ct::zeroize(self.encoded.as_mut_slice());
    ct::zeroize(self.salt.as_mut_slice());
    ct::zeroize(self.blinding_factor.as_mut_slice());
    ct::zeroize(self.blinding_inverse.as_mut_slice());
    ct::zeroize(self.inverse_mask.as_mut_slice());
    ct::zeroize(self.blinding_power.as_mut_slice());
    ct::zeroize(self.blinded.as_mut_slice());
    ct::zeroize(self.blinded_private_result.as_mut_slice());
    ct::zeroize(self.checked.as_mut_slice());
    ct::zeroize(self.one.as_mut_slice());
    ct::zeroize_words(&mut self.public_scratch.limbs[..]);
    ct::zeroize(&mut self.public_scratch.bytes[..]);
    self.mul_scratch.clear();
    self.inverse_scratch.clear();
    self.exponent_scratch.clear();
  }
}

impl fmt::Debug for RsaPrivateScratch {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("RsaPrivateScratch")
      .field("bytes", &self.encoded.as_slice().len())
      .finish_non_exhaustive()
  }
}

impl Drop for RsaPrivateScratch {
  fn drop(&mut self) {
    self.clear();
  }
}

impl RsaPrivateKeyComponents {
  #[cfg(feature = "getrandom")]
  pub(super) fn random_blinding_factor(&self) -> Result<RsaBlindingFactor, RsaPrivateOpError> {
    self.random_blinding_factor_with(getrandom::fill)
  }

  #[cfg(feature = "getrandom")]
  fn random_blinding_factor_with<E>(
    &self,
    mut fill_random: impl FnMut(&mut [u8]) -> Result<(), E>,
  ) -> Result<RsaBlindingFactor, RsaPrivateOpError> {
    let len = self.public.modulus().len();
    for _ in 0..128 {
      let mut factor = vec![0u8; len];
      if fill_random(&mut factor).is_err() {
        ct::zeroize(&mut factor);
        return Err(RsaPrivateOpError::EntropyUnavailable);
      }
      if is_zero_unsigned_be(&factor) || unsigned_be_cmp(&factor, self.public.modulus()) != core::cmp::Ordering::Less {
        ct::zeroize(&mut factor);
        continue;
      }

      let mut inverse = vec![0u8; len];
      match self.blinding_factor_inverse_with(&factor, &mut inverse, &mut fill_random) {
        Ok(()) => {
          let mut check = vec![0u8; len];
          if mod_mul_representatives(&self.public.modulus, &factor, &inverse, &mut check).is_ok()
            && check.last() == Some(&1)
          {
            let is_one = check
              .get(..check.len().strict_sub(1))
              .is_some_and(|prefix| prefix.iter().all(|&byte| byte == 0));
            if is_one {
              return Ok(RsaBlindingFactor {
                factor: SecretBigEndianBuffer::new(factor),
                inverse: SecretBigEndianBuffer::new(inverse),
              });
            }
          }
          ct::zeroize(&mut factor);
          ct::zeroize(&mut inverse);
        }
        Err(error) => {
          ct::zeroize(&mut factor);
          ct::zeroize(&mut inverse);
          // A failed mask draw is an entropy failure, not a rejected candidate.
          if error == RsaPrivateOpError::EntropyUnavailable {
            return Err(error);
          }
        }
      }
    }
    Err(RsaPrivateOpError::InvalidBlindingFactor)
  }

  #[cfg(feature = "getrandom")]
  pub(super) fn random_blinding_factor_into_scratch(
    &self,
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    self.random_blinding_factor_into_scratch_with(scratch, getrandom::fill)
  }

  pub(super) fn random_blinding_factor_into_scratch_with<E>(
    &self,
    scratch: &mut RsaPrivateScratch,
    mut fill_random: impl FnMut(&mut [u8]) -> Result<(), E>,
  ) -> Result<(), RsaPrivateOpError> {
    let len = self.public.modulus().len();
    scratch.ensure_len(len)?;
    for _ in 0..128 {
      if fill_random(scratch.blinding_factor.as_mut_slice()).is_err() {
        ct::zeroize(scratch.blinding_factor.as_mut_slice());
        return Err(RsaPrivateOpError::EntropyUnavailable);
      }
      if !private_valid_blinding_factor(scratch.blinding_factor.as_slice(), self.public.modulus()) {
        ct::zeroize(scratch.blinding_factor.as_mut_slice());
        continue;
      }

      scratch.blinding_inverse.as_mut_slice().fill(0);
      match self.blinding_factor_inverse_into_scratch(scratch, &mut fill_random) {
        Ok(()) => {
          mod_mul_representatives_with_scratch(
            &self.public.modulus,
            scratch.blinding_factor.as_slice(),
            scratch.blinding_inverse.as_slice(),
            scratch.checked.as_mut_slice(),
            &mut scratch.mul_scratch,
          )?;
          if scratch.checked.as_slice().last() == Some(&1) {
            let is_one = scratch
              .checked
              .as_slice()
              .get(..scratch.checked.as_slice().len().strict_sub(1))
              .is_some_and(|prefix| prefix.iter().all(|&byte| byte == 0));
            if is_one {
              return Ok(());
            }
          }
          ct::zeroize(scratch.blinding_factor.as_mut_slice());
          ct::zeroize(scratch.blinding_inverse.as_mut_slice());
          ct::zeroize(scratch.checked.as_mut_slice());
        }
        Err(error) => {
          ct::zeroize(scratch.blinding_factor.as_mut_slice());
          ct::zeroize(scratch.blinding_inverse.as_mut_slice());
          ct::zeroize(scratch.checked.as_mut_slice());
          // A failed mask draw is an entropy failure, not a rejected candidate.
          if error == RsaPrivateOpError::EntropyUnavailable {
            return Err(error);
          }
        }
      }
    }
    Err(RsaPrivateOpError::InvalidBlindingFactor)
  }

  #[cfg(feature = "getrandom")]
  pub(super) fn blinding_factor_inverse_with<E>(
    &self,
    factor: &[u8],
    out: &mut [u8],
    fill_random: impl FnMut(&mut [u8]) -> Result<(), E>,
  ) -> Result<(), RsaPrivateOpError> {
    let n_len = self.public.modulus().len();
    if factor.len() != n_len || out.len() != n_len {
      return Err(RsaPrivateOpError::InvalidLength);
    }

    let mut scratch = RsaPrivateScratch::new_components(self);
    scratch.blinding_factor.as_mut_slice().copy_from_slice(factor);
    let result = self
      .blinding_factor_inverse_into_scratch(&mut scratch, fill_random)
      .map(|()| {
        out.copy_from_slice(scratch.blinding_inverse.as_slice());
      });
    scratch.clear();
    result
  }

  /// Invert `scratch.blinding_factor` into `scratch.blinding_inverse` through a fresh
  /// multiplicative mask drawn from `fill_random`.
  ///
  /// The modular inverse runs on the Montgomery product `factor · mask · R⁻¹ mod n`, which is
  /// uniform and independent of the factor, and the Montgomery product of its result with the
  /// mask is the factor's inverse: `(factor · mask · R⁻¹)⁻¹ · mask · R⁻¹ = factor⁻¹`. The
  /// inverse's running time may therefore depend on its input without depending on the factor:
  /// native DudeCT measured such an input dependence for the 62-bit inverse on AArch64
  /// (Constant-Time 39717348652). The mask is cleared on every return.
  pub(super) fn blinding_factor_inverse_into_scratch<E>(
    &self,
    scratch: &mut RsaPrivateScratch,
    mut fill_random: impl FnMut(&mut [u8]) -> Result<(), E>,
  ) -> Result<(), RsaPrivateOpError> {
    let n_len = self.public.modulus().len();
    scratch.ensure_len(n_len)?;
    let mut mask_valid = false;
    for _ in 0..128 {
      if fill_random(scratch.inverse_mask.as_mut_slice()).is_err() {
        ct::zeroize(scratch.inverse_mask.as_mut_slice());
        return Err(RsaPrivateOpError::EntropyUnavailable);
      }
      if private_valid_blinding_factor(scratch.inverse_mask.as_slice(), self.public.modulus()) {
        mask_valid = true;
        break;
      }
    }
    let result = if mask_valid {
      self.masked_inverse_steps(scratch)
    } else {
      Err(RsaPrivateOpError::InvalidBlindingFactor)
    };
    ct::zeroize(scratch.inverse_mask.as_mut_slice());
    ct::zeroize(scratch.checked.as_mut_slice());
    if result.is_err() {
      ct::zeroize(scratch.blinding_inverse.as_mut_slice());
    }
    result
  }

  /// `blinding_inverse = (factor · mask · R⁻¹)⁻¹ · mask · R⁻¹ mod n`, with `checked` as the masked value.
  fn masked_inverse_steps(&self, scratch: &mut RsaPrivateScratch) -> Result<(), RsaPrivateOpError> {
    // The products read the secret factor and mask, so they run under DIT like the inverse.
    crate::traits::ct::with_data_independent_timing(|| {
      mont_mul_representatives_with_scratch(
        &self.public.modulus,
        scratch.blinding_factor.as_slice(),
        scratch.inverse_mask.as_slice(),
        scratch.checked.as_mut_slice(),
        &mut scratch.mul_scratch,
      )?;
      private_modular_inverse_odd(
        &self.public.modulus,
        scratch.checked.as_slice(),
        scratch.blinding_inverse.as_mut_slice(),
        &mut scratch.inverse_scratch,
      )?;
      mont_mul_representatives_with_scratch(
        &self.public.modulus,
        scratch.blinding_inverse.as_slice(),
        scratch.inverse_mask.as_slice(),
        scratch.checked.as_mut_slice(),
        &mut scratch.mul_scratch,
      )?;
      scratch
        .blinding_inverse
        .as_mut_slice()
        .copy_from_slice(scratch.checked.as_slice());
      Ok(())
    })
  }

  pub(super) fn sign_encoded_message_with_blinding_factor(
    &self,
    encoded: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let len = self.public.modulus().len();
    if encoded.len() != len || out.len() != len {
      return Err(RsaPrivateOpError::InvalidLength);
    }

    self.private_operation_with_blinding_factor(encoded, blinding, out)
  }

  pub(super) fn sign_encoded_message_with_blinding_factor_and_scratch(
    &self,
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let len = self.public.modulus().len();
    if out.len() != len {
      return Err(RsaPrivateOpError::InvalidLength);
    }

    self.private_operation_with_blinding_factor_and_scratch(blinding, out, scratch)
  }

  pub(super) fn sign_encoded_message_with_stored_blinding_and_scratch(
    &self,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let len = self.public.modulus().len();
    if out.len() != len {
      return Err(RsaPrivateOpError::InvalidLength);
    }

    self
      .private_operation_from_scratch_encoded_with_stored_blinding(scratch)
      .map(|()| {
        out.copy_from_slice(scratch.blinded_private_result.as_slice());
      })
  }

  pub(super) fn private_operation_with_blinding_factor(
    &self,
    input: &[u8],
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let len = self.public.modulus().len();
    if input.len() != len || out.len() != len || blinding.factor.len() != len || blinding.inverse.len() != len {
      return Err(RsaPrivateOpError::InvalidLength);
    }

    let mut scratch = RsaPrivateScratch::new_components(self);
    scratch.encoded.as_mut_slice().copy_from_slice(input);
    let result = self.private_operation_with_blinding_factor_and_scratch(blinding, out, &mut scratch);
    scratch.clear();
    result
  }

  pub(super) fn private_operation_with_blinding_factor_and_scratch(
    &self,
    blinding: RsaBlindingPair<'_>,
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    if out.len() != self.public.modulus().len() {
      return Err(RsaPrivateOpError::InvalidLength);
    }

    self
      .private_operation_from_scratch_encoded(blinding, scratch)
      .map(|()| {
        out.copy_from_slice(scratch.blinded_private_result.as_slice());
      })
  }

  pub(super) fn private_operation_from_scratch_encoded(
    &self,
    blinding: RsaBlindingPair<'_>,
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let len = self.public.modulus().len();
    if scratch.encoded.as_slice().len() != len || blinding.factor.len() != len || blinding.inverse.len() != len {
      return Err(RsaPrivateOpError::InvalidLength);
    }
    scratch.ensure_len(len)?;

    if blinding.validate {
      scratch.set_one()?;
      mod_mul_representatives_with_scratch(
        &self.public.modulus,
        blinding.factor,
        blinding.inverse,
        scratch.checked.as_mut_slice(),
        &mut scratch.mul_scratch,
      )?;
      if !ct::public_len_eq(scratch.checked.as_slice(), scratch.one.as_slice()).declassify() {
        return Err(RsaPrivateOpError::InvalidBlindingFactor);
      }
    }

    self
      .public
      .prepared_public_operation_with_scratch(
        blinding.factor,
        scratch.blinding_power.as_mut_slice(),
        &mut scratch.public_scratch,
      )
      .map_err(|_| RsaPrivateOpError::InvalidBlindingFactor)?;

    mod_mul_representatives_with_scratch(
      &self.public.modulus,
      scratch.encoded.as_slice(),
      scratch.blinding_power.as_slice(),
      scratch.blinded.as_mut_slice(),
      &mut scratch.mul_scratch,
    )?;
    self.private_operation_crt_from_blinded_scratch(scratch)?;
    mod_mul_representatives_with_scratch(
      &self.public.modulus,
      scratch.blinded_private_result.as_slice(),
      blinding.inverse,
      scratch.blinded.as_mut_slice(),
      &mut scratch.mul_scratch,
    )?;

    self
      .public
      .prepared_public_operation_with_scratch(
        scratch.blinded.as_slice(),
        scratch.checked.as_mut_slice(),
        &mut scratch.public_scratch,
      )
      .map_err(|_| RsaPrivateOpError::FaultCheckFailed)?;
    if ct::public_len_eq(scratch.checked.as_slice(), scratch.encoded.as_slice()).declassify() {
      scratch
        .blinded_private_result
        .as_mut_slice()
        .copy_from_slice(scratch.blinded.as_slice());
      Ok(())
    } else {
      Err(RsaPrivateOpError::FaultCheckFailed)
    }
  }

  pub(super) fn private_operation_from_scratch_encoded_with_stored_blinding(
    &self,
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let len = self.public.modulus().len();
    if scratch.encoded.as_slice().len() != len
      || scratch.blinding_factor.as_slice().len() != len
      || scratch.blinding_inverse.as_slice().len() != len
    {
      return Err(RsaPrivateOpError::InvalidLength);
    }
    scratch.ensure_len(len)?;

    self
      .public
      .prepared_public_operation_with_scratch(
        scratch.blinding_factor.as_slice(),
        scratch.blinding_power.as_mut_slice(),
        &mut scratch.public_scratch,
      )
      .map_err(|_| RsaPrivateOpError::InvalidBlindingFactor)?;
    self.private_operation_from_scratch_encoded_with_stored_blinding_power(scratch)
  }

  /// Blind `scratch.encoded` with `scratch.blinding_power`, run the CRT private
  /// operation, unblind with `scratch.blinding_inverse`, and check the result
  /// with the public operation.
  ///
  /// The caller guarantees that the blinding power is `rᵉ mod n` for the
  /// factor `r` whose inverse is `scratch.blinding_inverse`; the fault check
  /// rejects any other pair. The result is in `scratch.blinded_private_result`.
  fn private_operation_from_scratch_encoded_with_stored_blinding_power(
    &self,
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let len = self.public.modulus().len();
    if scratch.encoded.as_slice().len() != len
      || scratch.blinding_power.as_slice().len() != len
      || scratch.blinding_inverse.as_slice().len() != len
    {
      return Err(RsaPrivateOpError::InvalidLength);
    }
    scratch.ensure_len(len)?;

    mod_mul_representatives_with_scratch(
      &self.public.modulus,
      scratch.encoded.as_slice(),
      scratch.blinding_power.as_slice(),
      scratch.blinded.as_mut_slice(),
      &mut scratch.mul_scratch,
    )?;
    self.private_operation_crt_from_blinded_scratch(scratch)?;
    mod_mul_representatives_with_scratch(
      &self.public.modulus,
      scratch.blinded_private_result.as_slice(),
      scratch.blinding_inverse.as_slice(),
      scratch.blinded.as_mut_slice(),
      &mut scratch.mul_scratch,
    )?;

    self
      .public
      .prepared_public_operation_with_scratch(
        scratch.blinded.as_slice(),
        scratch.checked.as_mut_slice(),
        &mut scratch.public_scratch,
      )
      .map_err(|_| RsaPrivateOpError::FaultCheckFailed)?;
    if ct::public_len_eq(scratch.checked.as_slice(), scratch.encoded.as_slice()).declassify() {
      scratch
        .blinded_private_result
        .as_mut_slice()
        .copy_from_slice(scratch.blinded.as_slice());
      Ok(())
    } else {
      Err(RsaPrivateOpError::FaultCheckFailed)
    }
  }

  fn private_operation_crt_from_blinded_scratch(
    &self,
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let n_len = self.public.modulus().len();
    if scratch.blinded.as_slice().len() != n_len || scratch.blinded_private_result.as_slice().len() != n_len {
      return Err(RsaPrivateOpError::InvalidLength);
    }

    let prime_p = self.prime_p.as_bytes();
    let prime_q = self.prime_q.as_bytes();
    let modulus_p = &self.prime_p_modulus;
    let modulus_q = &self.prime_q_modulus;
    scratch
      .exponent_scratch
      .ensure_limb_count(core::cmp::max(modulus_p.limbs.len(), modulus_q.limbs.len()))?;

    {
      let representative_p = scratch
        .blinding_power
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?;
      private_reduce_unsigned_be_mod_to_fixed(
        scratch.blinded.as_slice(),
        prime_q.len(),
        modulus_p,
        representative_p,
        &mut scratch.exponent_scratch,
      )?;
    }
    {
      let representative_q = scratch
        .checked
        .as_mut_slice()
        .get_mut(..prime_q.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?;
      private_reduce_unsigned_be_mod_to_fixed(
        scratch.blinded.as_slice(),
        prime_p.len(),
        modulus_q,
        representative_q,
        &mut scratch.exponent_scratch,
      )?;
    }

    private_exponentiate_crt_with_scratch(
      (
        modulus_p,
        self.exponent_p.as_bytes(),
        scratch
          .blinding_power
          .as_slice()
          .get(..prime_p.len())
          .ok_or(RsaPrivateOpError::InvalidScratch)?,
        scratch
          .blinded_private_result
          .as_mut_slice()
          .get_mut(..prime_p.len())
          .ok_or(RsaPrivateOpError::InvalidScratch)?,
      ),
      (
        modulus_q,
        self.exponent_q.as_bytes(),
        scratch
          .checked
          .as_slice()
          .get(..prime_q.len())
          .ok_or(RsaPrivateOpError::InvalidScratch)?,
        scratch
          .blinded
          .as_mut_slice()
          .get_mut(..prime_q.len())
          .ok_or(RsaPrivateOpError::InvalidScratch)?,
      ),
      &mut scratch.exponent_scratch,
    )?;

    private_reduce_unsigned_be_mod_to_fixed(
      scratch
        .blinded
        .as_slice()
        .get(..prime_q.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      prime_q.len(),
      modulus_p,
      scratch
        .blinding_power
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      &mut scratch.exponent_scratch,
    )?;
    private_sub_mod_unsigned_be_to_fixed(
      scratch
        .blinded_private_result
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .blinding_power
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      prime_p,
      scratch
        .checked
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
    )?;

    left_pad_be(
      self.coefficient.as_bytes(),
      scratch
        .blinding_power
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
    )?;
    mod_mul_representatives_with_scratch(
      modulus_p,
      scratch
        .blinding_power
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .checked
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .blinded_private_result
        .as_mut_slice()
        .get_mut(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      &mut scratch.mul_scratch,
    )?;

    private_product_add_unsigned_be_to_fixed(
      prime_q,
      scratch
        .blinded_private_result
        .as_slice()
        .get(..prime_p.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch
        .blinded
        .as_slice()
        .get(..prime_q.len())
        .ok_or(RsaPrivateOpError::InvalidScratch)?,
      scratch.checked.as_mut_slice(),
      &mut scratch.mul_scratch,
    )?;
    scratch
      .blinded_private_result
      .as_mut_slice()
      .copy_from_slice(scratch.checked.as_slice());
    Ok(())
  }
}

impl RsaSigningWorkspace {
  pub(super) fn new(key: &RsaPrivateKey) -> Self {
    let public = key.public_key();
    let len = public.modulus().len();
    Self {
      scratch: key.private_scratch(),
      modulus: public.modulus().into(),
      public_exponent: public.public_exponent().as_u64(),
      blinding_power: SecretBigEndianBuffer::zeroed(len),
      blinding_inverse: SecretBigEndianBuffer::zeroed(len),
      remaining: 0,
      origin: ProcessOrigin::current(),
    }
  }

  /// Whether `public` is the key that created this workspace. For valid RSA
  /// keys, one modulus and public exponent name one key.
  fn belongs_to(&self, public: &RsaPublicKey) -> bool {
    self.public_exponent == public.public_exponent().as_u64() && *self.modulus == *public.modulus()
  }

  fn clear_pair(&mut self) {
    ct::zeroize(self.blinding_power.as_mut_slice());
    ct::zeroize(self.blinding_inverse.as_mut_slice());
    self.remaining = 0;
  }
}

impl fmt::Debug for RsaSigningWorkspace {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("RsaSigningWorkspace")
      .field("bytes", &self.blinding_power.as_slice().len())
      .field("remaining", &self.remaining)
      .finish_non_exhaustive()
  }
}

impl Drop for RsaSigningWorkspace {
  fn drop(&mut self) {
    self.clear_pair();
  }
}

impl RsaPrivateKey {
  /// Allocate a signing workspace that belongs to this key.
  ///
  /// The first signature through the workspace draws a new blinding factor.
  /// See [`RsaSigningWorkspace`] for the reuse schedule and its limits.
  #[inline]
  #[must_use]
  pub fn signing_workspace(&self) -> RsaSigningWorkspace {
    RsaSigningWorkspace::new(self)
  }

  /// Sign using a typed RSA signature profile, OS-backed randomness, and a
  /// signing workspace.
  ///
  /// This is [`Self::sign_signature_with_random_fill_and_workspace`] with
  /// `getrandom` as both entropy sources.
  ///
  /// # Errors
  ///
  /// The same as [`Self::sign_signature_with_random_fill_and_workspace`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_signature_with_workspace(
    &self,
    profile: RsaSignatureProfile,
    message: &[u8],
    out: &mut [u8],
    workspace: &mut RsaSigningWorkspace,
  ) -> Result<(), RsaPrivateOpError> {
    self.sign_signature_with_random_fill_and_workspace(
      profile,
      message,
      out,
      workspace,
      getrandom::fill,
      getrandom::fill,
    )
  }

  /// Sign using a typed RSA signature profile, caller-provided entropy, and a
  /// signing workspace.
  ///
  /// A PSS profile requests its salt length from `fill_salt` once in every
  /// call. PKCS#1 v1.5 never calls `fill_salt`. Blinding comes from
  /// `workspace`, as [`RsaSigningWorkspace`] describes: a refresh requests a
  /// blinding candidate and an inverse mask from `fill_blinding`, with the
  /// bounds, order, and error normalization of
  /// [`Self::sign_signature_with_random_fill`]. Every other call squares the
  /// cached pair and calls neither source for blinding. Signing does not
  /// allocate. The two sources can have different owners: the salt belongs to
  /// the signature, and the blinding pair belongs to the workspace. `out` is
  /// cleared on every error. The workspace scratch is cleared after every call,
  /// and every error also clears the pair, so the next call refreshes it.
  ///
  /// # Security
  ///
  /// The entropy contract of [`Self::sign_signature_with_random_fill`]
  /// applies to every requested byte.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::InvalidScratch`] if `workspace` belongs to
  /// another key; this check comes first and calls no source. Otherwise
  /// returns [`RsaPrivateOpError`] if `out` has the wrong length, the key is
  /// too small for the selected profile, either source fails, a refresh finds
  /// no valid blinding factor within the retry bound, or the post-signing
  /// public fault check fails.
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_signature_with_random_fill_and_workspace<E, F>(
    &self,
    profile: RsaSignatureProfile,
    message: &[u8],
    out: &mut [u8],
    workspace: &mut RsaSigningWorkspace,
    fill_salt: impl FnOnce(&mut [u8]) -> Result<(), E>,
    fill_blinding: impl FnMut(&mut [u8]) -> Result<(), F>,
  ) -> Result<(), RsaPrivateOpError> {
    crate::traits::ct::with_data_independent_timing(|| {
      let result = if workspace.belongs_to(self.public_key()) {
        self.sign_with_workspace(profile, message, out, workspace, fill_salt, fill_blinding)
      } else {
        Err(RsaPrivateOpError::InvalidScratch)
      };
      workspace.scratch.clear();
      match &result {
        Ok(remaining) => workspace.remaining = *remaining,
        Err(_) => workspace.clear_pair(),
      }
      clear_output_on_error(result.map(|_| ()), out)
    })
  }

  /// Encode and sign `message` into `out` with the next blinding pair of
  /// `workspace`, which belongs to this key.
  ///
  /// On success, returns the signatures left in the epoch, which the caller
  /// commits to `workspace.remaining`.
  fn sign_with_workspace<E, F>(
    &self,
    profile: RsaSignatureProfile,
    message: &[u8],
    out: &mut [u8],
    workspace: &mut RsaSigningWorkspace,
    fill_salt: impl FnOnce(&mut [u8]) -> Result<(), E>,
    fill_blinding: impl FnMut(&mut [u8]) -> Result<(), F>,
  ) -> Result<u8, RsaPrivateOpError> {
    if out.len() != self.public_key().modulus().len() {
      return Err(RsaPrivateOpError::InvalidLength);
    }
    let scratch = &mut workspace.scratch;
    match profile {
      RsaSignatureProfile::Pss { profile, salt_len } => {
        if !self.public_key().pss_salt_len_is_possible(profile, salt_len) {
          return Err(RsaPrivateOpError::MessageTooLong);
        }
        {
          let salt = scratch
            .salt
            .as_mut_slice()
            .get_mut(..salt_len)
            .ok_or(RsaPrivateOpError::MessageTooLong)?;
          fill_salt(salt).map_err(|_| RsaPrivateOpError::EntropyUnavailable)?;
        }
        self
          .components
          .encode_pss_with_stored_salt_into_scratch(profile, message, salt_len, scratch)?;
      }
      RsaSignatureProfile::Pkcs1v15(profile) => {
        self
          .components
          .encode_pkcs1v15_into_scratch(profile, message, scratch)?;
      }
    }
    let remaining = self
      .components
      .workspace_blinding_into_scratch(workspace, fill_blinding)?;
    let scratch = &mut workspace.scratch;
    self
      .components
      .private_operation_from_scratch_encoded_with_stored_blinding_power(scratch)?;
    out.copy_from_slice(scratch.blinded_private_result.as_slice());
    Ok(remaining)
  }
}

impl RsaPrivateKeyComponents {
  /// Advance the pair of `workspace` and copy it into its scratch
  /// `blinding_power` and `blinding_inverse`.
  ///
  /// A refresh draws `r` and `r⁻¹` through
  /// [`Self::random_blinding_factor_into_scratch_with`], which checks
  /// `r · r⁻¹ = 1`, and computes `rᵉ` with the public operation. Otherwise
  /// both cached values are squared with the private modular multiplication.
  /// Taking the use count first leaves the workspace in the refresh state until
  /// the caller commits the returned count, so any failure or unwind before the
  /// commit forces the next call to refresh.
  fn workspace_blinding_into_scratch<F>(
    &self,
    workspace: &mut RsaSigningWorkspace,
    fill_blinding: impl FnMut(&mut [u8]) -> Result<(), F>,
  ) -> Result<u8, RsaPrivateOpError> {
    let RsaSigningWorkspace {
      scratch,
      blinding_power,
      blinding_inverse,
      remaining,
      origin,
      ..
    } = workspace;
    let left = core::mem::take(remaining);
    let left = if left == 0 || !origin.is_current() {
      self.random_blinding_factor_into_scratch_with(scratch, fill_blinding)?;
      self
        .public
        .prepared_public_operation_with_scratch(
          scratch.blinding_factor.as_slice(),
          scratch.blinding_power.as_mut_slice(),
          &mut scratch.public_scratch,
        )
        .map_err(|_| RsaPrivateOpError::InvalidBlindingFactor)?;
      *origin = ProcessOrigin::current();
      RSA_SIGNING_WORKSPACE_EPOCH_USES
    } else {
      mod_mul_representatives_with_scratch(
        &self.public.modulus,
        blinding_power.as_slice(),
        blinding_power.as_slice(),
        scratch.blinding_power.as_mut_slice(),
        &mut scratch.mul_scratch,
      )?;
      mod_mul_representatives_with_scratch(
        &self.public.modulus,
        blinding_inverse.as_slice(),
        blinding_inverse.as_slice(),
        scratch.blinding_inverse.as_mut_slice(),
        &mut scratch.mul_scratch,
      )?;
      left
    };
    blinding_power
      .as_mut_slice()
      .copy_from_slice(scratch.blinding_power.as_slice());
    blinding_inverse
      .as_mut_slice()
      .copy_from_slice(scratch.blinding_inverse.as_slice());
    Ok(left.strict_sub(1))
  }
}

#[cfg(feature = "getrandom")]
impl RsaBlindingFactor {
  #[inline]
  pub(super) fn factor(&self) -> &[u8] {
    self.factor.as_slice()
  }

  #[inline]
  pub(super) fn inverse(&self) -> &[u8] {
    self.inverse.as_slice()
  }
}

impl fmt::Debug for RsaBlindingPair<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("RsaBlindingPair")
      .field("factor", &"****")
      .field("inverse", &"****")
      .finish()
  }
}

impl<'a> RsaBlindingPair<'a> {
  /// Borrow a factor and its claimed modular inverse.
  ///
  /// Validation requires the target RSA modulus and is therefore deferred to
  /// the private operation that consumes this value.
  #[inline]
  #[must_use]
  pub const fn new(factor: &'a [u8], inverse: &'a [u8]) -> Self {
    Self {
      factor,
      inverse,
      validate: true,
    }
  }

  #[cfg(feature = "getrandom")]
  #[inline]
  pub(super) const fn trusted(factor: &'a [u8], inverse: &'a [u8]) -> Self {
    Self {
      factor,
      inverse,
      validate: false,
    }
  }
}

pub(super) fn private_scratch_prefix(limbs: &mut SecretLimbs, len: usize) -> Result<&mut [u64], RsaPrivateOpError> {
  limbs
    .as_mut_slice()
    .get_mut(..len)
    .ok_or(RsaPrivateOpError::InvalidScratch)
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
pub(super) fn private_i31_scratch_prefix(words: &mut SecretI31, len: usize) -> Result<&mut [u32], RsaPrivateOpError> {
  words
    .as_mut_slice()
    .get_mut(..len)
    .ok_or(RsaPrivateOpError::InvalidScratch)
}

#[cfg(test)]
mod tests;
