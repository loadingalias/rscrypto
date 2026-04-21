//! X25519 Diffie-Hellman key exchange (RFC 7748).
//!
//! # Quick Start
//!
//! ```rust
//! use rscrypto::{X25519PublicKey, X25519SecretKey};
//!
//! let alice = X25519SecretKey::from_bytes([7u8; X25519SecretKey::LENGTH]);
//! let bob = X25519SecretKey::from_bytes([9u8; X25519SecretKey::LENGTH]);
//!
//! let alice_public = alice.public_key();
//! let bob_public = bob.public_key();
//!
//! let alice_shared = alice.diffie_hellman(&bob_public)?;
//! let bob_shared = bob.diffie_hellman(&alice_public)?;
//!
//! assert_eq!(alice_shared, bob_shared);
//! assert_eq!(alice_public.as_bytes().len(), X25519PublicKey::LENGTH);
//! # Ok::<(), rscrypto::auth::X25519Error>(())
//! ```

use core::{
  fmt,
  hash::{Hash, Hasher},
};

use crate::{
  auth::curve25519_edwards,
  backend::curve25519::{FieldElement, clamp_secret_scalar},
  traits::ct,
};

const POINT_LENGTH: usize = 32;
const RADIX_BITS: u32 = 51;
const MASK51: u64 = (1u64 << RADIX_BITS) - 1;
const BASEPOINT_BYTES: [u8; POINT_LENGTH] = {
  let mut bytes = [0u8; POINT_LENGTH];
  bytes[0] = 9;
  bytes
};
const A24: FieldElement = FieldElement::from_small(121665);

/// X25519 key agreement failed because the derived secret was all-zero.
///
/// This occurs when the peer input is a low-order point. RFC 7748 allows
/// implementations to detect this by OR-ing the shared secret bytes together
/// and aborting if the result is zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct X25519Error;

impl X25519Error {
  /// Construct a new X25519 agreement error.
  #[inline]
  #[must_use]
  pub const fn new() -> Self {
    Self
  }
}

impl Default for X25519Error {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl fmt::Display for X25519Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("x25519 shared secret is all-zero")
  }
}

impl core::error::Error for X25519Error {}

/// X25519 secret scalar bytes.
#[derive(Clone)]
pub struct X25519SecretKey([u8; Self::LENGTH]);

impl PartialEq for X25519SecretKey {
  fn eq(&self, other: &Self) -> bool {
    ct::constant_time_eq(&self.0, &other.0)
  }
}

impl Eq for X25519SecretKey {}

impl X25519SecretKey {
  /// Secret key length in bytes.
  pub const LENGTH: usize = POINT_LENGTH;

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

  /// Construct a secret key by filling bytes from the provided closure.
  ///
  /// ```rust
  /// # use rscrypto::X25519SecretKey;
  /// let sk = X25519SecretKey::generate(|buf| buf.fill(0xA5));
  /// assert_eq!(sk.as_bytes(), &[0xA5; X25519SecretKey::LENGTH]);
  /// ```
  #[inline]
  #[must_use]
  pub fn generate(fill: impl FnOnce(&mut [u8; Self::LENGTH])) -> Self {
    let mut bytes = [0u8; Self::LENGTH];
    fill(&mut bytes);
    Self(bytes)
  }

  impl_getrandom!();

  /// Derive the matching public key.
  #[must_use]
  pub fn public_key(&self) -> X25519PublicKey {
    public_key_from_scalar(&self.clamped_scalar_bytes())
  }

  /// Compute an X25519 shared secret with `public`.
  ///
  /// # Errors
  ///
  /// Returns [`X25519Error`] when the derived shared secret is all-zero,
  /// which indicates a low-order peer input.
  pub fn diffie_hellman(&self, public: &X25519PublicKey) -> Result<X25519SharedSecret, X25519Error> {
    X25519SharedSecret::diffie_hellman(self, public)
  }

  #[inline]
  #[must_use]
  fn clamped_scalar_bytes(&self) -> [u8; Self::LENGTH] {
    let mut scalar = self.0;
    clamp_secret_scalar(&mut scalar);
    scalar
  }
}

impl fmt::Debug for X25519SecretKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("X25519SecretKey(****)")
  }
}

impl_hex_fmt_secret!(X25519SecretKey);
impl_serde_bytes!(X25519SecretKey);

impl Drop for X25519SecretKey {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

impl_ct_eq!(X25519SecretKey);

/// X25519 public key bytes.
#[derive(Clone, Copy)]
pub struct X25519PublicKey {
  bytes: [u8; Self::LENGTH],
  u: FieldElement,
}

impl X25519PublicKey {
  /// Public key length in bytes.
  pub const LENGTH: usize = POINT_LENGTH;

  /// Construct a public key from its byte representation.
  ///
  /// The input bytes are preserved exactly for serialization, while internal
  /// arithmetic reduces the u-coordinate modulo `2^255 - 19` per RFC 7748.
  #[must_use]
  pub fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
    Self {
      u: decode_u_coordinate(&bytes),
      bytes,
    }
  }

  /// Return the canonical X25519 basepoint `u = 9`.
  #[inline]
  #[must_use]
  pub fn basepoint() -> Self {
    Self::from_bytes(BASEPOINT_BYTES)
  }

  #[inline]
  #[must_use]
  fn from_u(u: FieldElement) -> Self {
    Self { bytes: u.to_bytes(), u }
  }

  /// Return the public key bytes.
  #[inline]
  #[must_use]
  pub const fn to_bytes(self) -> [u8; Self::LENGTH] {
    self.bytes
  }

  /// Borrow the public key bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.bytes
  }
}

impl PartialEq for X25519PublicKey {
  fn eq(&self, other: &Self) -> bool {
    self.bytes == other.bytes
  }
}

impl Eq for X25519PublicKey {}

impl Hash for X25519PublicKey {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.bytes.hash(state);
  }
}

impl fmt::Debug for X25519PublicKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "X25519PublicKey(")?;
    crate::hex::fmt_hex_lower(&self.bytes, f)?;
    write!(f, ")")
  }
}

impl_hex_fmt!(X25519PublicKey);
impl_serde_bytes!(X25519PublicKey);

impl_ct_eq!(X25519PublicKey, bytes);

impl From<&X25519SecretKey> for X25519PublicKey {
  #[inline]
  fn from(secret: &X25519SecretKey) -> Self {
    secret.public_key()
  }
}

impl From<X25519SecretKey> for X25519PublicKey {
  #[inline]
  fn from(secret: X25519SecretKey) -> Self {
    secret.public_key()
  }
}

/// X25519 shared secret bytes.
#[derive(Clone)]
pub struct X25519SharedSecret([u8; Self::LENGTH]);

impl PartialEq for X25519SharedSecret {
  fn eq(&self, other: &Self) -> bool {
    ct::constant_time_eq(&self.0, &other.0)
  }
}

impl Eq for X25519SharedSecret {}

impl X25519SharedSecret {
  /// Shared secret length in bytes.
  pub const LENGTH: usize = POINT_LENGTH;

  /// Construct a shared secret from its byte representation.
  #[inline]
  #[must_use]
  pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
    Self(bytes)
  }

  /// Return the shared secret bytes.
  #[inline]
  #[must_use]
  pub fn to_bytes(&self) -> [u8; Self::LENGTH] {
    self.0
  }

  /// Borrow the shared secret bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }

  /// Compute an X25519 shared secret with `secret` and `public`.
  ///
  /// # Errors
  ///
  /// Returns [`X25519Error`] when the derived shared secret is all-zero,
  /// which indicates a low-order peer input.
  pub fn diffie_hellman(secret: &X25519SecretKey, public: &X25519PublicKey) -> Result<Self, X25519Error> {
    let shared = montgomery_ladder(&secret.clamped_scalar_bytes(), &public.u).to_bytes();
    if is_all_zero(&shared) {
      Err(X25519Error::new())
    } else {
      Ok(Self(shared))
    }
  }
}

impl fmt::Debug for X25519SharedSecret {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("X25519SharedSecret(****)")
  }
}

impl_hex_fmt_secret!(X25519SharedSecret);
impl_serde_bytes!(X25519SharedSecret);

impl Drop for X25519SharedSecret {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

impl_ct_eq!(X25519SharedSecret);

#[allow(clippy::indexing_slicing)]
#[must_use]
fn montgomery_ladder(scalar_bytes: &[u8; POINT_LENGTH], u: &FieldElement) -> FieldElement {
  let x1 = *u;
  let mut x2 = FieldElement::ONE;
  let mut z2 = FieldElement::ZERO;
  let mut x3 = x1;
  let mut z3 = FieldElement::ONE;
  let mut swap = 0u8;
  let mut bit = 255usize;

  while bit > 0 {
    bit = bit.strict_sub(1);

    let byte_index = bit >> 3;
    let bit_index = bit & 7;
    let bit_value = (scalar_bytes[byte_index] >> bit_index) & 1;

    swap ^= bit_value;
    FieldElement::conditional_swap(&mut x2, &mut x3, swap);
    FieldElement::conditional_swap(&mut z2, &mut z3, swap);
    swap = bit_value;

    let a = x2.add(&z2);
    let aa = a.square();
    let b = x2.sub(&z2);
    let bb = b.square();
    let e = aa.sub(&bb);
    let c = x3.add(&z3);
    let d = x3.sub(&z3);
    let da = d.mul(&a);
    let cb = c.mul(&b);
    let da_plus_cb = da.add(&cb);
    let da_minus_cb = da.sub(&cb);
    let da_minus_cb_sq = da_minus_cb.square();

    x3 = da_plus_cb.square();
    z3 = x1.mul(&da_minus_cb_sq);
    x2 = aa.mul(&bb);
    z2 = e.mul(&aa.add(&A24.mul(&e)));
  }

  FieldElement::conditional_swap(&mut x2, &mut x3, swap);
  FieldElement::conditional_swap(&mut z2, &mut z3, swap);

  x2.mul(&z2.invert())
}

#[inline]
#[must_use]
fn public_key_from_scalar(scalar_bytes: &[u8; POINT_LENGTH]) -> X25519PublicKey {
  let point = curve25519_edwards::basepoint_mul_dispatch(scalar_bytes);
  X25519PublicKey::from_u(point.to_montgomery_u())
}

#[must_use]
fn is_all_zero(bytes: &[u8; POINT_LENGTH]) -> bool {
  let mut acc = 0u8;
  for &byte in bytes {
    acc |= byte;
  }
  core::hint::black_box(acc) == 0
}

#[must_use]
fn decode_u_coordinate(bytes: &[u8; POINT_LENGTH]) -> FieldElement {
  let mut canonical = *bytes;
  canonical[POINT_LENGTH - 1] &= 0x7f;

  let mut acc = 0u128;
  let mut acc_bits = 0u32;
  let mut byte_iter = canonical.iter().copied();
  let mut limbs = [0u64; 5];

  for limb in &mut limbs {
    while acc_bits < RADIX_BITS {
      let Some(byte) = byte_iter.next() else {
        break;
      };
      acc |= u128::from(byte) << acc_bits;
      acc_bits = acc_bits.wrapping_add(8);
    }

    *limb = (acc & u128::from(MASK51)) as u64;
    acc >>= RADIX_BITS;
    acc_bits = acc_bits.wrapping_sub(RADIX_BITS);
  }

  FieldElement::from_limbs(limbs).normalize()
}
