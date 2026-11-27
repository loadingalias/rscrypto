use super::*;

/// Reduce a wide P-256 nonce candidate to a nonzero scalar and return its limbs.
#[cfg(feature = "ecdsa-p256")]
pub(crate) fn diag_ecdsa_p256_reduce_wide_order_limb_digest(wide: [u8; 64]) -> [u64; 4] {
  let wide = ZeroizingBytes::new(wide);
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P256_ORDER_MODULUS));
  nonce.value().0
}

/// Derive and invert the deterministic P-256 nonce through the caller-blinded s390x path.
#[cfg(feature = "ecdsa-p256")]
pub(crate) fn diag_ecdsa_p256_nonce_inverse_blinded_limb_digest(
  secret: [u8; 32],
  blind: [u8; 64],
  message: &[u8],
) -> [u64; 4] {
  let secret = ZeroizingBytes::new(secret);
  let blind = ZeroizingBytes::new(blind);
  let digest = Sha256::digest(message);
  let mut wide = ZeroizingBytes::zeroed();
  hmac_expand_p256(secret.as_array(), &digest, wide.as_mut_array());
  let (_, nonce_blind) = signing_blinds(blind.as_array(), &P256_ORDER_MODULUS);
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P256_ORDER_MODULUS));
  blinded_nonce_inverse_montgomery(&P256, &nonce, &nonce_blind).value().0
}

/// Reduce a wide P-384 nonce candidate to a nonzero scalar and return its limbs.
#[cfg(feature = "ecdsa-p384")]
pub(crate) fn diag_ecdsa_p384_reduce_wide_order_limb_digest(wide: [u8; 96]) -> [u64; 6] {
  let wide = ZeroizingBytes::new(wide);
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P384_ORDER_MODULUS));
  nonce.value().0
}

/// Derive and invert the deterministic P-384 nonce through the caller-blinded s390x path.
#[cfg(feature = "ecdsa-p384")]
pub(crate) fn diag_ecdsa_p384_nonce_inverse_blinded_limb_digest(
  secret: [u8; 48],
  blind: [u8; 96],
  message: &[u8],
) -> [u64; 6] {
  let secret = ZeroizingBytes::new(secret);
  let blind = ZeroizingBytes::new(blind);
  let digest = Sha384::digest(message);
  let mut wide = ZeroizingBytes::zeroed();
  hmac_expand_p384(secret.as_array(), &digest, wide.as_mut_array());
  let (_, nonce_blind) = signing_blinds(blind.as_array(), &P384_ORDER_MODULUS);
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P384_ORDER_MODULUS));
  blinded_nonce_inverse_montgomery(&P384, &nonce, &nonce_blind).value().0
}
