//! Ed25519 signing and verification with typed keys and signatures.
//!
//! # Quick Start
//!
//! ```rust
//! use rscrypto::auth::ed25519::{Ed25519Keypair, Ed25519SecretKey};
//!
//! let secret = Ed25519SecretKey::from_bytes([7u8; Ed25519SecretKey::LENGTH]);
//! let keypair = Ed25519Keypair::from_secret_key(secret.clone());
//! let public = keypair.public_key();
//! let sig = secret.sign(b"auth");
//!
//! assert_eq!(secret.as_bytes().len(), 32);
//! assert_eq!(public.as_bytes().len(), 32);
//! assert_eq!(sig.as_bytes().len(), 64);
//! assert!(public.verify(b"auth", &sig).is_ok());
//! ```
//!
//! The public API keeps secret keys, public keys, signatures, and keypairs as
//! distinct types instead of passing raw byte slices through signing and
//! verification calls.

use core::fmt;

use crate::{
  hashes::crypto::Sha512,
  traits::{Digest, VerificationError, ct},
};

mod constants;
pub(crate) mod field;
pub(crate) mod hash;
pub(crate) mod point;
pub(crate) mod scalar;

use self::constants::{PUBLIC_KEY_LENGTH, SECRET_KEY_LENGTH, SIGNATURE_LENGTH};

// Keep the planned internal layout explicit at compile time.
const _: usize = core::mem::size_of::<field::FieldElement>();
const _: usize = core::mem::size_of::<hash::ExpandedSecret>();
const _: usize = core::mem::size_of::<point::ExtendedPoint>();
const _: usize = core::mem::size_of::<scalar::Scalar>();
const _: field::FieldElement = field::FieldElement::ONE;
const _: fn([u64; constants::FIELD_LIMBS]) -> field::FieldElement = field::FieldElement::from_limbs;
const _: fn(u64) -> field::FieldElement = field::FieldElement::from_small;
const _: fn(&field::FieldElement) -> &[u64; constants::FIELD_LIMBS] = field::FieldElement::limbs;
const _: fn(&field::FieldElement, &field::FieldElement) -> field::FieldElement = field::FieldElement::add;
const _: fn(&field::FieldElement, &field::FieldElement) -> field::FieldElement = field::FieldElement::sub;
const _: fn(&field::FieldElement, &field::FieldElement) -> field::FieldElement = field::FieldElement::mul;
const _: fn(&field::FieldElement) -> field::FieldElement = field::FieldElement::square;
const _: fn(&field::FieldElement) -> field::FieldElement = field::FieldElement::neg;
const _: fn(&field::FieldElement) -> field::FieldElement = field::FieldElement::invert;
const _: fn(&field::FieldElement) -> field::FieldElement = field::FieldElement::normalize;
const _: fn(&[u8; 32]) -> Option<field::FieldElement> = field::FieldElement::from_bytes;
const _: fn(field::FieldElement) -> [u8; 32] = field::FieldElement::to_bytes;
const _: fn(&field::FieldElement) -> bool = field::FieldElement::is_zero;
const _: fn(&field::FieldElement) -> bool = field::FieldElement::is_negative;
const _: fn(&field::FieldElement) -> Option<field::FieldElement> = field::FieldElement::sqrt;
const _: point::ExtendedPoint = point::ExtendedPoint::identity();
const _: fn(field::FieldElement, field::FieldElement) -> point::ExtendedPoint = point::ExtendedPoint::from_affine;
const _: fn(&point::ExtendedPoint, &point::ExtendedPoint) -> point::ExtendedPoint = point::ExtendedPoint::add;
const _: fn(&point::ExtendedPoint) -> point::ExtendedPoint = point::ExtendedPoint::double;
const _: fn(point::ExtendedPoint) -> Option<[u8; 32]> = point::ExtendedPoint::to_bytes;
const _: fn(&[u8; 32]) -> Option<point::ExtendedPoint> = point::ExtendedPoint::from_bytes;
const _: fn() -> point::ExtendedPoint = point::ExtendedPoint::basepoint;
const _: fn(&point::ExtendedPoint, &[u8; 32]) -> point::ExtendedPoint = point::ExtendedPoint::scalar_mul;
const _: fn(&[u8; 32]) -> point::ExtendedPoint = point::ExtendedPoint::scalar_mul_basepoint;
const _: fn(&point::ExtendedPoint) -> point::ExtendedPoint = point::ExtendedPoint::mul_by_cofactor;
const _: fn(point::ExtendedPoint) -> Option<(field::FieldElement, field::FieldElement)> =
  point::ExtendedPoint::to_affine;
const _: fn(
  &point::ExtendedPoint,
) -> (
  &field::FieldElement,
  &field::FieldElement,
  &field::FieldElement,
  &field::FieldElement,
) = point::ExtendedPoint::components;
const _: fn(&Ed25519SecretKey) -> hash::ExpandedSecret = hash::ExpandedSecret::from_secret_key;
const _: fn(&hash::ExpandedSecret) -> &[u8; SECRET_KEY_LENGTH] = hash::ExpandedSecret::scalar_bytes;
const _: fn(&hash::ExpandedSecret) -> scalar::Scalar = hash::ExpandedSecret::scalar_words;
const _: fn(&hash::ExpandedSecret) -> &[u8; SECRET_KEY_LENGTH] = hash::ExpandedSecret::nonce_prefix;
const _: fn(&hash::ExpandedSecret) -> [u8; SECRET_KEY_LENGTH] = hash::ExpandedSecret::public_key_bytes;
const _: fn(&mut [u8; SECRET_KEY_LENGTH]) = scalar::clamp_secret_scalar;
const _: fn(&[u8; SECRET_KEY_LENGTH]) -> scalar::Scalar = scalar::decode_words_le;
const _: fn(&[u8; SECRET_KEY_LENGTH]) -> Option<scalar::Scalar> = scalar::from_canonical_bytes;
const _: fn(&scalar::Scalar) -> [u8; SECRET_KEY_LENGTH] = scalar::to_bytes;
const _: fn(&[u8]) -> scalar::Scalar = scalar::reduce_bytes_mod_order;
const _: fn(&scalar::Scalar, &scalar::Scalar) -> scalar::Scalar = scalar::add_mod;
const _: fn(&scalar::Scalar, &scalar::Scalar) -> scalar::Scalar = scalar::mul_mod;
const _: fn(&scalar::Scalar, &scalar::Scalar, &scalar::Scalar) -> scalar::Scalar = scalar::mul_add_mod;

/// Ed25519 secret key bytes.
///
/// Provides typed signing and public-key derivation instead of vague `&[u8]`
/// parameters at the call site.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Ed25519SecretKey([u8; Self::LENGTH]);

impl Ed25519SecretKey {
  /// Secret key length in bytes.
  pub const LENGTH: usize = SECRET_KEY_LENGTH;

  /// Construct a secret key from its byte representation.
  #[inline]
  #[must_use]
  pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
    Self(bytes)
  }

  /// Return the secret key bytes.
  #[inline]
  #[must_use]
  pub fn to_bytes(&self) -> [u8; Self::LENGTH] {
    self.0
  }

  /// Borrow the secret key bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }

  /// Derive the matching Ed25519 public key.
  #[must_use]
  pub fn public_key(&self) -> Ed25519PublicKey {
    let expanded = hash::ExpandedSecret::from_secret_key(self);
    Ed25519PublicKey::from_bytes(expanded.public_key_bytes())
  }

  /// Sign a message with this secret key.
  #[must_use]
  pub fn sign(&self, message: &[u8]) -> Ed25519Signature {
    let public = self.public_key();
    sign_with_secret(self, &public, message)
  }
}

impl Default for Ed25519SecretKey {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl fmt::Debug for Ed25519SecretKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Ed25519SecretKey").finish_non_exhaustive()
  }
}

impl Drop for Ed25519SecretKey {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

/// Ed25519 public key bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ed25519PublicKey([u8; Self::LENGTH]);

impl Ed25519PublicKey {
  /// Public key length in bytes.
  pub const LENGTH: usize = PUBLIC_KEY_LENGTH;

  /// Construct a public key from its byte representation.
  #[inline]
  #[must_use]
  pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
    Self(bytes)
  }

  /// Return the public key bytes.
  #[inline]
  #[must_use]
  pub const fn to_bytes(self) -> [u8; Self::LENGTH] {
    self.0
  }

  /// Borrow the public key bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }
}

impl Default for Ed25519PublicKey {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl fmt::Debug for Ed25519PublicKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_tuple("Ed25519PublicKey").field(&self.0).finish()
  }
}

impl Ed25519PublicKey {
  /// Verify a message/signature pair against this public key.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the public key bytes are invalid, the
  /// signature is non-canonical, or the signature does not match `message`.
  pub fn verify(&self, message: &[u8], signature: &Ed25519Signature) -> Result<(), VerificationError> {
    verify(message, self, signature)
  }
}

/// Ed25519 signature bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ed25519Signature([u8; Self::LENGTH]);

impl Ed25519Signature {
  /// Signature length in bytes.
  pub const LENGTH: usize = SIGNATURE_LENGTH;

  /// Construct a signature from its byte representation.
  #[inline]
  #[must_use]
  pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
    Self(bytes)
  }

  /// Return the signature bytes.
  #[inline]
  #[must_use]
  pub const fn to_bytes(self) -> [u8; Self::LENGTH] {
    self.0
  }

  /// Borrow the signature bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }
}

impl Default for Ed25519Signature {
  #[inline]
  fn default() -> Self {
    Self([0u8; Self::LENGTH])
  }
}

impl fmt::Debug for Ed25519Signature {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_tuple("Ed25519Signature").field(&self.0).finish()
  }
}

/// Ed25519 keypair with typed secret and public halves.
#[derive(Clone, PartialEq, Eq)]
pub struct Ed25519Keypair {
  secret: Ed25519SecretKey,
  public: Ed25519PublicKey,
}

impl Ed25519Keypair {
  /// Derive a keypair from a secret key.
  #[must_use]
  pub fn from_secret_key(secret: Ed25519SecretKey) -> Self {
    let public = secret.public_key();
    Self { secret, public }
  }

  /// Borrow the secret key.
  #[must_use]
  pub const fn secret_key(&self) -> &Ed25519SecretKey {
    &self.secret
  }

  /// Return the public key.
  #[must_use]
  pub const fn public_key(&self) -> Ed25519PublicKey {
    self.public
  }

  /// Sign a message with the keypair secret key.
  #[must_use]
  pub fn sign(&self, message: &[u8]) -> Ed25519Signature {
    sign_with_secret(&self.secret, &self.public, message)
  }
}

impl fmt::Debug for Ed25519Keypair {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Ed25519Keypair")
      .field("public", &self.public)
      .finish_non_exhaustive()
  }
}

/// Verify a message/signature pair against an Ed25519 public key.
///
/// # Errors
///
/// Returns [`VerificationError`] if the public key bytes are invalid, the
/// signature is non-canonical, or the signature does not match `message`.
pub fn verify(
  message: &[u8],
  public_key: &Ed25519PublicKey,
  signature: &Ed25519Signature,
) -> Result<(), VerificationError> {
  let (r_bytes, s_bytes) = split_signature(signature);
  let r_point = point::ExtendedPoint::from_bytes(&r_bytes).ok_or(VerificationError::new())?;
  let a_point = point::ExtendedPoint::from_bytes(public_key.as_bytes()).ok_or(VerificationError::new())?;
  let s = scalar::from_canonical_bytes(&s_bytes).ok_or(VerificationError::new())?;

  let mut challenge_hasher = Sha512::new();
  challenge_hasher.update(&r_bytes);
  challenge_hasher.update(public_key.as_bytes());
  challenge_hasher.update(message);
  let challenge_digest = challenge_hasher.finalize();
  let challenge = scalar::reduce_bytes_mod_order(&challenge_digest);
  let neg_challenge = scalar::negate_mod(&challenge);
  let neg_challenge_bytes = scalar::to_bytes(&neg_challenge);
  let s_canonical = scalar::to_bytes(&s);

  // Straus/Shamir: compute [s]B + [-h]A in one interleaved 256-bit scan,
  // then cofactor-clear and compare to [8]R.
  let combined =
    point::ExtendedPoint::straus_basepoint_vartime(&s_canonical, &neg_challenge_bytes, &a_point).mul_by_cofactor();
  let expected = r_point.mul_by_cofactor();

  match (combined.to_bytes(), expected.to_bytes()) {
    (Some(left), Some(right)) if ct::constant_time_eq(&left, &right) => Ok(()),
    _ => Err(VerificationError::new()),
  }
}

#[must_use]
fn sign_with_secret(secret: &Ed25519SecretKey, public: &Ed25519PublicKey, message: &[u8]) -> Ed25519Signature {
  let expanded = hash::ExpandedSecret::from_secret_key(secret);
  let secret_scalar = scalar::reduce_bytes_mod_order(expanded.scalar_bytes());

  let mut nonce_hasher = Sha512::new();
  nonce_hasher.update(expanded.nonce_prefix());
  nonce_hasher.update(message);
  let nonce_digest = nonce_hasher.finalize();
  let nonce_scalar = scalar::reduce_bytes_mod_order(&nonce_digest);
  let nonce_bytes = scalar::to_bytes(&nonce_scalar);
  let r_encoded = point::ExtendedPoint::scalar_mul_basepoint(&nonce_bytes)
    .to_bytes()
    .unwrap_or_default();

  let mut challenge_hasher = Sha512::new();
  challenge_hasher.update(&r_encoded);
  challenge_hasher.update(public.as_bytes());
  challenge_hasher.update(message);
  let challenge_digest = challenge_hasher.finalize();
  let challenge = scalar::reduce_bytes_mod_order(&challenge_digest);
  let s = scalar::mul_add_mod(&challenge, &secret_scalar, &nonce_scalar);
  let s_bytes = scalar::to_bytes(&s);

  assemble_signature(&r_encoded, &s_bytes)
}

#[must_use]
fn assemble_signature(r_bytes: &[u8; 32], s_bytes: &[u8; 32]) -> Ed25519Signature {
  let mut bytes = [0u8; Ed25519Signature::LENGTH];
  let (r_dst, s_dst) = bytes.as_mut_slice().split_at_mut(SECRET_KEY_LENGTH);

  for (dst, src) in r_dst.iter_mut().zip(r_bytes.iter().copied()) {
    *dst = src;
  }
  for (dst, src) in s_dst.iter_mut().zip(s_bytes.iter().copied()) {
    *dst = src;
  }

  Ed25519Signature::from_bytes(bytes)
}

#[must_use]
fn split_signature(signature: &Ed25519Signature) -> ([u8; 32], [u8; 32]) {
  let mut r_bytes = [0u8; 32];
  let mut s_bytes = [0u8; 32];
  let (r_src, s_src) = signature.as_bytes().as_slice().split_at(SECRET_KEY_LENGTH);

  for (dst, src) in r_bytes.iter_mut().zip(r_src.iter().copied()) {
    *dst = src;
  }
  for (dst, src) in s_bytes.iter_mut().zip(s_src.iter().copied()) {
    *dst = src;
  }

  (r_bytes, s_bytes)
}

#[cfg(test)]
mod tests {
  use super::{
    Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature, constants, field, hash, point, scalar, verify,
  };

  #[test]
  fn internal_layout_matches_phase_2b_plan() {
    assert_eq!(constants::FIELD_LIMBS, 5);
    assert_eq!(constants::SCALAR_LIMBS, 4);
    assert_eq!(
      core::mem::size_of::<field::FieldElement>(),
      5 * core::mem::size_of::<u64>()
    );
    assert_eq!(core::mem::size_of::<scalar::Scalar>(), 4 * core::mem::size_of::<u64>());
    assert_eq!(
      core::mem::size_of::<point::ExtendedPoint>(),
      4 * core::mem::size_of::<field::FieldElement>()
    );
  }

  #[test]
  fn secret_key_roundtrips_bytes() {
    let key = Ed25519SecretKey::from_bytes([7u8; Ed25519SecretKey::LENGTH]);
    assert_eq!(key.to_bytes(), [7u8; Ed25519SecretKey::LENGTH]);
    assert_eq!(key.as_bytes(), &[7u8; Ed25519SecretKey::LENGTH]);
  }

  #[test]
  fn public_key_roundtrips_bytes() {
    let key = Ed25519PublicKey::from_bytes([9u8; Ed25519PublicKey::LENGTH]);
    assert_eq!(key.to_bytes(), [9u8; Ed25519PublicKey::LENGTH]);
    assert_eq!(key.as_bytes(), &[9u8; Ed25519PublicKey::LENGTH]);
  }

  #[test]
  fn signature_roundtrips_bytes() {
    let sig = Ed25519Signature::from_bytes([3u8; Ed25519Signature::LENGTH]);
    assert_eq!(sig.to_bytes(), [3u8; Ed25519Signature::LENGTH]);
    assert_eq!(sig.as_bytes(), &[3u8; Ed25519Signature::LENGTH]);
  }

  #[test]
  fn secret_key_debug_is_redacted() {
    let dbg = format!("{:?}", Ed25519SecretKey::default());
    assert_eq!(dbg, "Ed25519SecretKey { .. }");
  }

  #[test]
  fn secret_key_expansion_produces_distinct_scalar_and_nonce_material() {
    let secret = Ed25519SecretKey::from_bytes([5u8; Ed25519SecretKey::LENGTH]);
    let expanded = hash::ExpandedSecret::from_secret_key(&secret);

    assert_ne!(expanded.scalar_bytes(), expanded.nonce_prefix());
  }

  fn decode_hex<const N: usize>(hex: &str) -> [u8; N] {
    let bytes = hex.as_bytes();
    let mut out = [0u8; N];

    for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
      *dst = match chunk[0] {
        b'0'..=b'9' => chunk[0] - b'0',
        b'a'..=b'f' => chunk[0] - b'a' + 10,
        b'A'..=b'F' => chunk[0] - b'A' + 10,
        _ => panic!("invalid hex"),
      } << 4
        | match chunk[1] {
          b'0'..=b'9' => chunk[1] - b'0',
          b'a'..=b'f' => chunk[1] - b'a' + 10,
          b'A'..=b'F' => chunk[1] - b'A' + 10,
          _ => panic!("invalid hex"),
        };
    }

    out
  }

  #[test]
  fn secret_key_derives_public_key_from_rfc8032_vector_1() {
    let secret = Ed25519SecretKey::from_bytes(decode_hex(
      "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
    ));
    let expected = Ed25519PublicKey::from_bytes(decode_hex(
      "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
    ));

    assert_eq!(secret.public_key(), expected);
  }

  #[test]
  fn keypair_signs_and_public_key_verifies() {
    let secret = Ed25519SecretKey::from_bytes([0x55; Ed25519SecretKey::LENGTH]);
    let keypair = Ed25519Keypair::from_secret_key(secret);
    let message = b"rscrypto-ed25519";
    let signature = keypair.sign(message);

    assert!(keypair.public_key().verify(message, &signature).is_ok());
    assert!(verify(message, &keypair.public_key(), &signature).is_ok());
  }

  #[test]
  fn signing_matches_rfc8032_vector_1() {
    let secret = Ed25519SecretKey::from_bytes(decode_hex(
      "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
    ));
    let keypair = Ed25519Keypair::from_secret_key(secret);
    let signature = keypair.sign(b"");
    let expected = Ed25519Signature::from_bytes(decode_hex(
      "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155\
       5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
    ));

    assert_eq!(signature, expected);
  }

  #[test]
  fn verify_rejects_modified_message() {
    let secret = Ed25519SecretKey::from_bytes([0x77; Ed25519SecretKey::LENGTH]);
    let keypair = Ed25519Keypair::from_secret_key(secret);
    let signature = keypair.sign(b"message");

    assert!(keypair.public_key().verify(b"message!", &signature).is_err());
  }

  #[test]
  fn verify_rejects_noncanonical_s_component() {
    let public = Ed25519PublicKey::from_bytes(decode_hex(
      "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
    ));
    let signature = Ed25519Signature::from_bytes(decode_hex(
      "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155\
       edd3f55c1a631258d69cf7a2def9de1400000000000000000000000000000010",
    ));

    assert!(public.verify(b"", &signature).is_err());
  }
}
