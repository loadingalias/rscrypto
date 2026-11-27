use super::*;

/// Secret intermediates of one production signature, for phase timing probes.
#[derive(Clone, Copy)]
pub(crate) struct DiagEd25519SignInputs {
  /// SHA-512 of the nonce prefix and message, before reduction.
  pub nonce_digest: [u8; 64],
  /// Reduced nonce scalar `r`.
  pub nonce: [u8; SECRET_KEY_LENGTH],
  /// Clamped secret scalar `a`.
  pub secret_scalar: [u8; SECRET_KEY_LENGTH],
  /// Reduced challenge scalar `k`.
  pub challenge: [u8; SECRET_KEY_LENGTH],
}

impl core::fmt::Debug for DiagEd25519SignInputs {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("DiagEd25519SignInputs").finish_non_exhaustive()
  }
}

/// Derive the signing intermediates `sign` computes for `secret` and `message`.
#[must_use]
pub(crate) fn diag_ed25519_sign_inputs(secret: &Ed25519SecretKey, message: &[u8]) -> DiagEd25519SignInputs {
  let expanded = hash::ExpandedSecret::from_secret_key(secret);
  let mut nonce_hasher = Sha512::new();
  nonce_hasher.update(expanded.nonce_prefix());
  nonce_hasher.update(message);
  let nonce_digest = nonce_hasher.finalize_secret();
  let nonce = scalar::to_bytes(&scalar::reduce_64_bytes_mod_order_secret(&nonce_digest));
  let r_encoded = basepoint_mul_encoded_dispatch(&nonce);
  let challenge_digest = hash_challenge(&r_encoded, &expanded.public_key_bytes(), message);
  DiagEd25519SignInputs {
    nonce_digest,
    nonce,
    secret_scalar: *expanded.scalar_bytes(),
    challenge: scalar::to_bytes(&scalar::reduce_bytes_mod_order(&challenge_digest)),
  }
}

/// Execute the production signing nonce reduction.
#[must_use]
pub(crate) fn diag_ed25519_sign_nonce_reduce(nonce_digest: &[u8; 64]) -> [u8; SECRET_KEY_LENGTH] {
  crate::traits::ct::with_data_independent_timing(|| {
    scalar::to_bytes(&scalar::reduce_64_bytes_mod_order_secret(nonce_digest))
  })
}

/// Execute the production signing commitment `R = r·B`.
#[must_use]
pub(crate) fn diag_ed25519_sign_commitment(nonce: &[u8; SECRET_KEY_LENGTH]) -> [u8; PUBLIC_KEY_LENGTH] {
  crate::traits::ct::with_data_independent_timing(|| basepoint_mul_encoded_dispatch(nonce))
}

/// Execute the production signing response `S = k·a + r (mod L)`.
#[must_use]
pub(crate) fn diag_ed25519_sign_response(inputs: &DiagEd25519SignInputs) -> [u8; SECRET_KEY_LENGTH] {
  crate::traits::ct::with_data_independent_timing(|| {
    let challenge = scalar::decode_words_le(&inputs.challenge);
    let secret_scalar = scalar::decode_words_le(&inputs.secret_scalar);
    let nonce = scalar::decode_words_le(&inputs.nonce);
    scalar::to_bytes(&scalar::mul_add_mod_secret(&challenge, &secret_scalar, &nonce))
  })
}
