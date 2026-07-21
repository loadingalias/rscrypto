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
//! assert!(alice_shared.ct_eq(&bob_shared).declassify());
//! assert_eq!(alice_public.as_bytes().len(), X25519PublicKey::LENGTH);
//! # Ok::<(), rscrypto::auth::X25519Error>(())
//! ```
//!
//! # Post-Quantum Migration
//!
//! X25519 is a classical key-exchange primitive. For systems with a
//! long-lived confidentiality requirement, plan a hybrid migration path
//! instead of treating X25519 as the end state. The repository roadmap
//! tracks `ML-KEM` as the post-quantum key-establishment direction.

use core::{
  fmt,
  hash::{Hash, Hasher},
};

#[cfg(any(
  not(any(
    all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      not(feature = "portable-only")
    ),
    all(target_arch = "x86_64", target_os = "linux", not(feature = "portable-only"))
  )),
  miri,
  test
))]
use crate::auth::curve25519_edwards;
#[cfg(any(
  not(any(
    all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      not(feature = "portable-only")
    ),
    all(target_arch = "x86_64", target_os = "linux", not(feature = "portable-only"))
  )),
  miri,
  test
))]
use crate::backend::curve25519::FieldElement;
use crate::{SecretBytes, secret::ZeroizingBytes, traits::ct};

const POINT_LENGTH: usize = 32;
#[cfg(any(
  not(any(
    all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      not(feature = "portable-only")
    ),
    all(target_arch = "x86_64", target_os = "linux", not(feature = "portable-only"))
  )),
  miri,
  test
))]
const RADIX_BITS: u32 = 51;
#[cfg(any(
  not(any(
    all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      not(feature = "portable-only")
    ),
    all(target_arch = "x86_64", target_os = "linux", not(feature = "portable-only"))
  )),
  miri,
  test
))]
const MASK51: u64 = (1u64 << RADIX_BITS) - 1;
const BASEPOINT_BYTES: [u8; POINT_LENGTH] = {
  let mut bytes = [0u8; POINT_LENGTH];
  bytes[0] = 9;
  bytes
};
#[cfg(any(
  not(any(
    all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      not(feature = "portable-only")
    ),
    all(target_arch = "x86_64", target_os = "linux", not(feature = "portable-only"))
  )),
  miri,
  test
))]
const A24: FieldElement = FieldElement::from_small(121665);

#[cfg(all(
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(feature = "portable-only"),
  not(miri)
))]
mod aarch64_asm;
#[cfg(all(
  target_arch = "x86_64",
  target_os = "linux",
  not(feature = "portable-only"),
  not(miri)
))]
mod x86_64_asm;

#[cfg(all(
  target_arch = "aarch64",
  any(target_os = "macos", target_os = "linux"),
  not(feature = "portable-only"),
  not(miri)
))]
use aarch64_asm as platform_asm;
#[cfg(all(
  target_arch = "x86_64",
  target_os = "linux",
  not(feature = "portable-only"),
  not(miri)
))]
use x86_64_asm as platform_asm;

define_unit_error! {
  /// X25519 key agreement failed because the derived secret was all-zero.
  ///
  /// This occurs when the peer input is a low-order point. RFC 7748 allows
  /// implementations to detect this by OR-ing the shared secret bytes together
  /// and aborting if the result is zero.
  pub struct X25519Error;
  "x25519 shared secret is all-zero"
}

/// X25519 secret scalar bytes.
pub struct X25519SecretKey([u8; Self::LENGTH]);

impl X25519SecretKey {
  /// Secret key length in bytes.
  pub const LENGTH: usize = POINT_LENGTH;

  /// Compare two secret keys without exposing a branchable boolean.
  #[inline]
  pub fn ct_eq(&self, other: &Self) -> ct::CtDecision {
    ct::fixed_eq(&self.0, &other.0)
  }

  /// Construct a secret key from its byte representation.
  #[inline]
  #[must_use]
  pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
    Self(bytes)
  }

  /// Explicitly extract the secret key bytes into a zeroizing wrapper.
  #[inline]
  #[must_use]
  pub fn expose_secret(&self) -> SecretBytes<{ Self::LENGTH }> {
    SecretBytes::new(self.0)
  }

  /// Explicitly duplicate this secret key.
  #[inline]
  #[must_use]
  pub const fn duplicate_secret(&self) -> Self {
    Self(self.0)
  }

  /// Borrow the secret key bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }

  /// Try to construct a secret key by filling bytes from the provided closure.
  ///
  /// The closure may be called exactly once and must completely fill the
  /// buffer or return an error. Any partially filled buffer is cleared before
  /// returning an error.
  #[inline]
  pub fn try_generate_with<E>(mut fill: impl FnMut(&mut [u8]) -> Result<(), E>) -> Result<Self, E> {
    let mut bytes = ZeroizingBytes::zeroed();
    fill(bytes.as_mut_array())?;
    Ok(Self(*bytes.as_array()))
  }

  /// Try to generate a secret key from the platform entropy source.
  ///
  /// # Errors
  ///
  /// Returns a getrandom error if the platform entropy source is unavailable.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[inline]
  pub fn try_generate() -> Result<Self, getrandom::Error> {
    Self::try_generate_with(getrandom::fill)
  }

  /// Construct a secret key by filling bytes from the provided closure.
  ///
  /// Compatibility name for caller-filled generation. Prefer
  /// [`Self::try_generate_with`] when the entropy source can fail; this method
  /// remains supported until the newer name has shipped for one release.
  ///
  /// ```rust
  /// # use rscrypto::X25519SecretKey;
  /// let sk = X25519SecretKey::generate(|buf| buf.fill(0xA5));
  /// assert_eq!(sk.as_bytes(), &[0xA5; X25519SecretKey::LENGTH]);
  /// ```
  #[inline]
  #[must_use]
  pub fn generate(fill: impl FnOnce(&mut [u8; Self::LENGTH])) -> Self {
    let mut bytes = ZeroizingBytes::zeroed();
    fill(bytes.as_mut_array());
    Self(*bytes.as_array())
  }

  impl_getrandom!();

  /// Derive the matching public key.
  #[must_use]
  pub fn public_key(&self) -> X25519PublicKey {
    #[cfg(any(
      all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux"),
        not(feature = "portable-only"),
        not(miri)
      ),
      all(
        target_arch = "x86_64",
        target_os = "linux",
        not(feature = "portable-only"),
        not(miri)
      )
    ))]
    {
      X25519PublicKey::from_bytes(platform_asm::x25519_base(&self.0))
    }

    #[cfg(not(any(
      all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux"),
        not(feature = "portable-only"),
        not(miri)
      ),
      all(
        target_arch = "x86_64",
        target_os = "linux",
        not(feature = "portable-only"),
        not(miri)
      )
    )))]
    {
      self.with_clamped_scalar(public_key_from_scalar)
    }
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

  #[allow(dead_code)]
  #[cfg(any(
    not(any(
      all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux"),
        not(feature = "portable-only")
      ),
      all(target_arch = "x86_64", target_os = "linux", not(feature = "portable-only"))
    )),
    miri,
    test
  ))]
  fn with_clamped_scalar<R>(&self, f: impl FnOnce(&[u8; Self::LENGTH]) -> R) -> R {
    let mut scalar = ClampedScalar(self.0);
    crate::backend::curve25519::clamp_secret_scalar(&mut scalar.0);
    f(&scalar.0)
  }
}

#[cfg(any(
  not(any(
    all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      not(feature = "portable-only")
    ),
    all(target_arch = "x86_64", target_os = "linux", not(feature = "portable-only"))
  )),
  miri,
  test
))]
struct ClampedScalar([u8; POINT_LENGTH]);

#[cfg(any(
  not(any(
    all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      not(feature = "portable-only")
    ),
    all(target_arch = "x86_64", target_os = "linux", not(feature = "portable-only"))
  )),
  miri,
  test
))]
impl Drop for ClampedScalar {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

impl fmt::Debug for X25519SecretKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("X25519SecretKey(****)")
  }
}

impl_hex_fmt_secret!(X25519SecretKey);
impl_serde_secret_bytes!(X25519SecretKey);

impl Drop for X25519SecretKey {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

/// X25519 public key bytes.
#[derive(Clone, Copy)]
pub struct X25519PublicKey {
  bytes: [u8; Self::LENGTH],
  #[allow(dead_code)]
  #[cfg(any(
    not(any(
      all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux"),
        not(feature = "portable-only")
      ),
      all(target_arch = "x86_64", target_os = "linux", not(feature = "portable-only"))
    )),
    miri,
    test
  ))]
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
      bytes,
      #[cfg(any(
        not(any(
          all(
            target_arch = "aarch64",
            any(target_os = "macos", target_os = "linux"),
            not(feature = "portable-only")
          ),
          all(target_arch = "x86_64", target_os = "linux", not(feature = "portable-only"))
        )),
        miri,
        test
      ))]
      u: decode_u_coordinate(&bytes),
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
  #[allow(dead_code)]
  #[cfg(any(
    not(any(
      all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux"),
        not(feature = "portable-only")
      ),
      all(target_arch = "x86_64", target_os = "linux", not(feature = "portable-only"))
    )),
    miri,
    test
  ))]
  fn from_u(u: FieldElement) -> Self {
    Self {
      bytes: u.to_bytes(),
      #[cfg(any(
        not(any(
          all(
            target_arch = "aarch64",
            any(target_os = "macos", target_os = "linux"),
            not(feature = "portable-only")
          ),
          all(target_arch = "x86_64", target_os = "linux", not(feature = "portable-only"))
        )),
        miri,
        test
      ))]
      u,
    }
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
pub struct X25519SharedSecret([u8; Self::LENGTH]);

impl X25519SharedSecret {
  /// Shared secret length in bytes.
  pub const LENGTH: usize = POINT_LENGTH;

  /// Compare two shared secrets without exposing a branchable boolean.
  #[inline]
  pub fn ct_eq(&self, other: &Self) -> ct::CtDecision {
    ct::fixed_eq(&self.0, &other.0)
  }

  /// Construct a shared secret from its byte representation.
  #[inline]
  #[must_use]
  pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
    Self(bytes)
  }

  /// Explicitly extract the shared secret bytes into a zeroizing wrapper.
  #[inline]
  #[must_use]
  pub fn expose_secret(&self) -> SecretBytes<{ Self::LENGTH }> {
    SecretBytes::new(self.0)
  }

  /// Explicitly duplicate this shared secret.
  #[inline]
  #[must_use]
  pub const fn duplicate_secret(&self) -> Self {
    Self(self.0)
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
    #[cfg(any(
      all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux"),
        not(feature = "portable-only"),
        not(miri)
      ),
      all(
        target_arch = "x86_64",
        target_os = "linux",
        not(feature = "portable-only"),
        not(miri)
      )
    ))]
    let mut shared = platform_asm::x25519(&secret.0, &public.bytes);

    #[cfg(not(any(
      all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux"),
        not(feature = "portable-only"),
        not(miri)
      ),
      all(
        target_arch = "x86_64",
        target_os = "linux",
        not(feature = "portable-only"),
        not(miri)
      )
    )))]
    let mut shared = secret.with_clamped_scalar(|scalar| montgomery_ladder(scalar, &public.u).to_bytes());

    let result = if is_all_zero(&shared) {
      Err(X25519Error::new())
    } else {
      Ok(Self(shared))
    };
    ct::zeroize(&mut shared);
    result
  }
}

impl fmt::Debug for X25519SharedSecret {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("X25519SharedSecret(****)")
  }
}

impl_hex_fmt_secret!(X25519SharedSecret);
impl_serde_secret_bytes!(X25519SharedSecret);

impl Drop for X25519SharedSecret {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

#[allow(clippy::indexing_slicing)]
#[must_use]
#[cfg(any(
  not(any(
    all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      not(feature = "portable-only")
    ),
    all(target_arch = "x86_64", target_os = "linux", not(feature = "portable-only"))
  )),
  miri,
  test
))]
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
#[cfg(any(
  not(any(
    all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      not(feature = "portable-only")
    ),
    all(target_arch = "x86_64", target_os = "linux", not(feature = "portable-only"))
  )),
  miri,
  test
))]
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
#[cfg(any(
  not(any(
    all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      not(feature = "portable-only")
    ),
    all(target_arch = "x86_64", target_os = "linux", not(feature = "portable-only"))
  )),
  miri,
  test
))]
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

#[cfg(test)]
mod tests {
  #[cfg(miri)]
  #[test]
  fn miri_uses_portable_x25519_path() {
    use super::{X25519PublicKey, X25519SecretKey, decode_u_coordinate, montgomery_ladder, public_key_from_scalar};

    let secret = X25519SecretKey::from_bytes([7u8; X25519SecretKey::LENGTH]);
    let expected_public = secret.with_clamped_scalar(|scalar| public_key_from_scalar(scalar).to_bytes());
    assert_eq!(secret.public_key().to_bytes(), expected_public);

    let public = X25519PublicKey::basepoint();
    let expected_shared = secret
      .with_clamped_scalar(|scalar| montgomery_ladder(scalar, &decode_u_coordinate(public.as_bytes())).to_bytes());
    let shared = secret.diffie_hellman(&public).unwrap();
    assert_eq!(shared.as_bytes(), &expected_shared);
  }

  #[cfg(all(
    not(miri),
    any(
      all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux"),
        not(feature = "portable-only")
      ),
      all(target_arch = "x86_64", target_os = "linux", not(feature = "portable-only"))
    )
  ))]
  mod asm {
    use super::super::{
      POINT_LENGTH, X25519PublicKey, X25519SecretKey, decode_u_coordinate, montgomery_ladder, platform_asm,
      public_key_from_scalar,
    };

    fn scalar(seed: u8) -> [u8; POINT_LENGTH] {
      let mut out = [0u8; POINT_LENGTH];
      for (index, byte) in out.iter_mut().enumerate() {
        *byte = seed.wrapping_mul(37).wrapping_add((index as u8).wrapping_mul(19));
      }
      out
    }

    fn peer(seed: u8) -> [u8; POINT_LENGTH] {
      let mut out = [0u8; POINT_LENGTH];
      for (index, byte) in out.iter_mut().enumerate() {
        *byte = seed
          .wrapping_mul(53)
          .wrapping_add((index as u8).wrapping_mul(11))
          .wrapping_add(7);
      }
      out[POINT_LENGTH - 1] |= 0x80;
      out
    }

    #[test]
    fn fixed_base_asm_matches_portable_path() {
      for seed in 0u8..32 {
        let secret = X25519SecretKey::from_bytes(scalar(seed));
        let expected = secret.with_clamped_scalar(|scalar| public_key_from_scalar(scalar).to_bytes());
        let actual = platform_asm::x25519_base(secret.as_bytes());

        assert_eq!(actual, expected, "fixed-base mismatch for seed {seed}");
      }
    }

    #[test]
    fn ladder_asm_matches_portable_path() {
      for seed in 0u8..32 {
        let secret = X25519SecretKey::from_bytes(scalar(seed));
        let public = X25519PublicKey::from_bytes(peer(seed));
        let expected = secret
          .with_clamped_scalar(|scalar| montgomery_ladder(scalar, &decode_u_coordinate(public.as_bytes())).to_bytes());
        let actual = platform_asm::x25519(secret.as_bytes(), public.as_bytes());

        assert_eq!(actual, expected, "ladder mismatch for seed {seed}");
      }
    }
  }
}
