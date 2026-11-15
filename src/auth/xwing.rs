//! X-Wing: the hybrid KEM of ML-KEM-768 and X25519.
//!
//! This module implements
//! [`draft-connolly-cfrg-xwing-kem-11`](https://datatracker.ietf.org/doc/draft-connolly-cfrg-xwing-kem/11/).
//! `draft-irtf-cfrg-concrete-hybrid-kems-04` specifies the same construction
//! as MLKEM768-X25519. Its HPKE KEM identifier is `0x647A`.
//!
//! The components are rscrypto's ML-KEM-768, X25519, SHAKE256, and SHA3-256.
//! This module adds only the key expansion, the combiner, and the encodings.
//! The decapsulation key is a 32-byte seed. SHAKE256 expands it to 96 bytes:
//! the ML-KEM-768 seed `d || z`, then the X25519 secret key. The shared secret
//! is `SHA3-256(ss_M || ss_X || ct_X || pk_X || "\.//^\")`, where `M` marks the
//! ML-KEM-768 part and `X` the X25519 part.
//!
//! # Difference from draft -11
//!
//! Encapsulation and decapsulation reject an all-zero X25519 shared secret.
//! Draft -11 does not specify this check, and RFC 7748 makes it optional. This
//! module follows X-Wing pull request 39 at commit `782c880e`, which makes the
//! check mandatory, as RFC 9180 does for X25519 in HPKE. X25519 clamps its
//! scalar, so the result is all-zero exactly when the other party's X25519
//! input is a point of small order. The rejection therefore depends only on the
//! public encapsulation key or ciphertext. Every other ciphertext decapsulates
//! to a shared secret, with ML-KEM implicit rejection.
//!
//! # Encodings
//!
//! Keys and ciphertexts are the fixed-length byte strings of the draft, which
//! are also their HPKE serializations. An encapsulation key is the ML-KEM-768
//! encapsulation key followed by the X25519 public key. A ciphertext is the
//! ML-KEM-768 ciphertext followed by the X25519 ephemeral public key. The
//! draft's X.509 encodings are not implemented. A prepared decapsulation key
//! holds the expanded component keys and has no encoding, because the draft
//! forbids the transfer of an expanded key between implementations.
//!
//! ```
//! # #[cfg(feature = "x-wing")] {
//! use rscrypto::{Kem, XWing};
//!
//! // A fixed filler keeps the example reproducible. Use fresh randomness
//! // for real keys.
//! let fill = |out: &mut [u8]| {
//!   out.fill(7);
//!   Ok(())
//! };
//! let (encapsulation_key, decapsulation_key) = XWing::generate_keypair(fill)?;
//! let (ciphertext, sent) = XWing::encapsulate(&encapsulation_key, fill)?;
//!
//! // Prepare the key one time to decapsulate many ciphertexts.
//! let prepared = decapsulation_key.prepare();
//! let received = prepared.decapsulate(&ciphertext)?;
//! assert!(sent.ct_eq(&received).declassify());
//! # }
//! # Ok::<(), rscrypto::XWingError>(())
//! ```

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(feature = "alloc")]
use core::alloc::Allocator;
use core::{error::Error, fmt};

use crate::{
  MlKem768Ciphertext, MlKem768EncapsulationKey, MlKem768PreparedDecapsulationKey, MlKem768PreparedEncapsulationKey,
  MlKem768SharedSecret, SecretBytes, X25519PublicKey, X25519SecretKey, X25519SharedSecret,
  hashes::crypto::keccak::secret,
  secret::ZeroizingBytes,
  traits::{
    Kem,
    ct::{self, DataIndependentTiming},
  },
};

const SEED_SIZE: usize = 32;
const ML_KEM_SEED_SIZE: usize = 64;
const EXPANDED_SIZE: usize = ML_KEM_SEED_SIZE + X25519SecretKey::LENGTH;
const ML_KEM_ENCAPSULATION_KEY_SIZE: usize = MlKem768EncapsulationKey::LENGTH;
const ML_KEM_CIPHERTEXT_SIZE: usize = MlKem768Ciphertext::LENGTH;
const ML_KEM_COINS_SIZE: usize = 32;
const ENCAPSULATION_RANDOM_SIZE: usize = ML_KEM_COINS_SIZE + X25519SecretKey::LENGTH;
/// The embedded encapsulation key's offset in an ML-KEM-768 decapsulation key.
const ML_KEM_EMBEDDED_KEY_OFFSET: usize = 1152;
/// `XWingLabel`, the ASCII string `\.//^\`.
const LABEL: &[u8; 6] = b"\\.//^\\";

/// X-Wing operation error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum XWingError {
  /// The caller-provided random source failed.
  RandomGenerationFailed,
  /// The encapsulation key has the wrong length, fails the FIPS 203 modulus
  /// check, or has an X25519 part of small order.
  InvalidEncapsulationKey,
  /// The decapsulation key has the wrong length.
  InvalidDecapsulationKey,
  /// The ciphertext has the wrong length or an X25519 part of small order.
  InvalidCiphertext,
}

impl fmt::Display for XWingError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::RandomGenerationFailed => "X-Wing random generation failed",
      Self::InvalidEncapsulationKey => "X-Wing encapsulation key failed validation",
      Self::InvalidDecapsulationKey => "X-Wing decapsulation key has the wrong length",
      Self::InvalidCiphertext => "X-Wing ciphertext failed validation",
    })
  }
}

impl Error for XWingError {}

macro_rules! define_public_bytes {
  ($name:ident, $len:expr, $doc:literal) => {
    #[doc = $doc]
    #[derive(Clone, PartialEq, Eq, Hash)]
    pub struct $name([u8; Self::LENGTH]);

    impl $name {
      /// Length in bytes.
      pub const LENGTH: usize = $len;

      /// Wrap raw bytes without validation.
      #[inline]
      #[must_use]
      pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
        Self(bytes)
      }

      /// Return the wrapped bytes.
      #[inline]
      #[must_use]
      pub const fn to_bytes(&self) -> [u8; Self::LENGTH] {
        self.0
      }

      /// Borrow the wrapped bytes.
      #[inline]
      #[must_use]
      pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
        &self.0
      }
    }

    impl AsRef<[u8]> for $name {
      #[inline]
      fn as_ref(&self) -> &[u8] {
        &self.0
      }
    }

    impl fmt::Debug for $name {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", stringify!($name))?;
        crate::hex::fmt_hex_lower(&self.0, f)?;
        write!(f, ")")
      }
    }

    impl_hex_fmt!($name);
    impl_serde_bytes!($name);
  };
}

macro_rules! define_secret_bytes {
  ($name:ident, $doc:literal) => {
    #[doc = $doc]
    pub struct $name([u8; Self::LENGTH]);

    impl $name {
      /// Length in bytes.
      pub const LENGTH: usize = 32;

      /// Compare two secret values without exposing a branchable boolean.
      #[inline]
      pub fn ct_eq(&self, other: &Self) -> ct::CtDecision {
        ct::fixed_eq(&self.0, &other.0)
      }

      /// Wrap raw bytes. The caller retains responsibility for clearing `bytes`.
      #[inline]
      #[must_use]
      pub const fn from_bytes(bytes: [u8; Self::LENGTH]) -> Self {
        Self(bytes)
      }

      /// Explicitly extract the secret bytes into a zeroizing wrapper.
      #[inline]
      #[must_use]
      pub fn expose_secret(&self) -> SecretBytes<{ Self::LENGTH }> {
        SecretBytes::new(self.0)
      }

      /// Explicitly duplicate this secret value.
      #[inline]
      #[must_use]
      pub const fn duplicate_secret(&self) -> Self {
        Self(self.0)
      }

      /// Borrow the secret bytes.
      #[inline]
      #[must_use]
      pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
        &self.0
      }
    }

    impl AsRef<[u8]> for $name {
      #[inline]
      fn as_ref(&self) -> &[u8] {
        &self.0
      }
    }

    impl fmt::Debug for $name {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(****)", stringify!($name))
      }
    }

    impl Drop for $name {
      fn drop(&mut self) {
        ct::zeroize(&mut self.0);
      }
    }

    impl_hex_fmt_secret!($name);
    impl_serde_secret_bytes!($name);
  };
}

define_public_bytes!(
  XWingEncapsulationKey,
  ML_KEM_ENCAPSULATION_KEY_SIZE + X25519PublicKey::LENGTH,
  "X-Wing encapsulation key: the ML-KEM-768 encapsulation key, then the X25519 public key."
);
define_public_bytes!(
  XWingCiphertext,
  ML_KEM_CIPHERTEXT_SIZE + X25519PublicKey::LENGTH,
  "X-Wing ciphertext: the ML-KEM-768 ciphertext, then the X25519 ephemeral public key."
);
define_secret_bytes!(
  XWingDecapsulationKey,
  "X-Wing decapsulation key: the 32-byte seed that determines the key pair.\n\nNot `Clone` or `Copy`; zeroized on drop; `Debug` is redacted."
);
define_secret_bytes!(
  XWingSharedSecret,
  "X-Wing shared secret.\n\nNot `Clone` or `Copy`; zeroized on drop; `Debug` is redacted."
);

#[cfg(all(feature = "hpke", feature = "chacha20poly1305"))]
impl XWingDecapsulationKey {
  /// Borrow the seed mutably, so that HPKE derives a key in place and leaves
  /// no moved copy.
  pub(crate) const fn as_mut_bytes(&mut self) -> &mut [u8; Self::LENGTH] {
    &mut self.0
  }
}

/// X-Wing KEM marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct XWing;

impl XWing {
  /// Encapsulation key size in bytes.
  pub const ENCAPSULATION_KEY_SIZE: usize = XWingEncapsulationKey::LENGTH;

  /// Decapsulation key size in bytes.
  pub const DECAPSULATION_KEY_SIZE: usize = XWingDecapsulationKey::LENGTH;

  /// Ciphertext size in bytes.
  pub const CIPHERTEXT_SIZE: usize = XWingCiphertext::LENGTH;

  /// Shared-secret size in bytes.
  pub const SHARED_SECRET_SIZE: usize = XWingSharedSecret::LENGTH;

  /// Random bytes consumed by key generation: the decapsulation key.
  pub const KEY_GENERATION_RANDOM_SIZE: usize = SEED_SIZE;

  /// Random bytes consumed by encapsulation: the ML-KEM-768 coins, then the
  /// X25519 ephemeral secret key, as in `EncapsulateDerand` of the draft.
  pub const ENCAPSULATION_RANDOM_SIZE: usize = ENCAPSULATION_RANDOM_SIZE;

  /// Generate a key pair from the platform entropy source.
  ///
  /// # Errors
  ///
  /// Returns [`XWingError::RandomGenerationFailed`] if the entropy source is unavailable.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  pub fn try_generate_keypair() -> Result<(XWingEncapsulationKey, XWingDecapsulationKey), XWingError> {
    <Self as Kem>::generate_keypair(os_random)
  }

  /// Encapsulate to `encapsulation_key` with platform entropy.
  ///
  /// # Errors
  ///
  /// As [`Kem::encapsulate`], and [`XWingError::RandomGenerationFailed`] if
  /// the entropy source is unavailable.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  pub fn try_encapsulate(
    encapsulation_key: &XWingEncapsulationKey,
  ) -> Result<(XWingCiphertext, XWingSharedSecret), XWingError> {
    <Self as Kem>::encapsulate(encapsulation_key, os_random)
  }
}

impl Kem for XWing {
  const ENCAPSULATION_KEY_SIZE: usize = Self::ENCAPSULATION_KEY_SIZE;
  const DECAPSULATION_KEY_SIZE: usize = Self::DECAPSULATION_KEY_SIZE;
  const CIPHERTEXT_SIZE: usize = Self::CIPHERTEXT_SIZE;
  const SHARED_SECRET_SIZE: usize = Self::SHARED_SECRET_SIZE;

  type EncapsulationKey = XWingEncapsulationKey;
  type DecapsulationKey = XWingDecapsulationKey;
  type Ciphertext = XWingCiphertext;
  type SharedSecret = XWingSharedSecret;
  type KeyGenerationError = XWingError;
  type EncapsulationError = XWingError;
  type DecapsulationError = XWingError;

  /// `GenerateKeyPair` of the draft: 32 random bytes are the decapsulation key.
  fn generate_keypair(
    fill_random: impl FnMut(&mut [u8]) -> Result<(), Self::KeyGenerationError>,
  ) -> Result<(Self::EncapsulationKey, Self::DecapsulationKey), Self::KeyGenerationError> {
    let decapsulation_key = XWingDecapsulationKey::generate(fill_random)?;
    Ok((decapsulation_key.encapsulation_key(), decapsulation_key))
  }

  /// `Encapsulate` of the draft. See [`XWingPreparedEncapsulationKey::encapsulate`].
  fn encapsulate(
    encapsulation_key: &Self::EncapsulationKey,
    fill_random: impl FnMut(&mut [u8]) -> Result<(), Self::EncapsulationError>,
  ) -> Result<(Self::Ciphertext, Self::SharedSecret), Self::EncapsulationError> {
    encapsulation_key.prepare()?.encapsulate(fill_random)
  }

  /// `Decapsulate` of the draft, which expands the key on each call. See
  /// [`XWingPreparedDecapsulationKey::decapsulate`].
  fn decapsulate(
    decapsulation_key: &Self::DecapsulationKey,
    ciphertext: &Self::Ciphertext,
  ) -> Result<Self::SharedSecret, Self::DecapsulationError> {
    let mut prepared = XWingPreparedDecapsulationKey::unexpanded();
    prepared.expand(decapsulation_key);
    prepared.decapsulate(ciphertext)
  }
}

impl XWingEncapsulationKey {
  /// Parse and validate an encapsulation key from raw bytes. See [`Self::validate`].
  ///
  /// # Errors
  ///
  /// Returns [`XWingError::InvalidEncapsulationKey`] for a wrong length or a
  /// failed FIPS 203 modulus check.
  pub fn try_from_slice(bytes: &[u8]) -> Result<Self, XWingError> {
    let key = Self(bytes.try_into().map_err(|_| XWingError::InvalidEncapsulationKey)?);
    key.validate()?;
    Ok(key)
  }

  /// Validate the ML-KEM-768 part with the FIPS 203 modulus check.
  ///
  /// Every X25519 part passes. Encapsulation rejects an X25519 part of small
  /// order, because only its all-zero shared secret identifies it.
  ///
  /// # Errors
  ///
  /// Returns [`XWingError::InvalidEncapsulationKey`] for a failed modulus check.
  pub fn validate(&self) -> Result<(), XWingError> {
    self
      .ml_kem()
      .validate()
      .map_err(|_| XWingError::InvalidEncapsulationKey)
  }

  /// Validate one time and prepare this key for repeated encapsulation.
  ///
  /// # Errors
  ///
  /// As [`Self::validate`].
  pub fn prepare(&self) -> Result<XWingPreparedEncapsulationKey, XWingError> {
    Ok(XWingPreparedEncapsulationKey {
      ml_kem: MlKem768PreparedEncapsulationKey::try_from(self.ml_kem())
        .map_err(|_| XWingError::InvalidEncapsulationKey)?,
      x25519: X25519PublicKey::from_bytes(*self.x25519()),
    })
  }

  fn ml_kem(&self) -> MlKem768EncapsulationKey {
    MlKem768EncapsulationKey::from_bytes(*self.0.first_chunk().expect("the ML-KEM part fits the key"))
  }

  fn x25519(&self) -> &[u8; X25519PublicKey::LENGTH] {
    self.0.last_chunk().expect("the X25519 part fits the key")
  }
}

impl XWingCiphertext {
  /// Parse a ciphertext from raw bytes.
  ///
  /// # Errors
  ///
  /// Returns [`XWingError::InvalidCiphertext`] for a wrong length.
  pub fn try_from_slice(bytes: &[u8]) -> Result<Self, XWingError> {
    Ok(Self(bytes.try_into().map_err(|_| XWingError::InvalidCiphertext)?))
  }

  fn ml_kem(&self) -> &[u8; ML_KEM_CIPHERTEXT_SIZE] {
    self.0.first_chunk().expect("the ML-KEM part fits the ciphertext")
  }

  fn x25519(&self) -> &[u8; X25519PublicKey::LENGTH] {
    self.0.last_chunk().expect("the X25519 part fits the ciphertext")
  }
}

impl XWingDecapsulationKey {
  /// Parse a decapsulation key from raw bytes. Every 32-byte string is a
  /// valid key. The caller retains responsibility for clearing `bytes`.
  ///
  /// # Errors
  ///
  /// Returns [`XWingError::InvalidDecapsulationKey`] for a wrong length.
  pub fn try_from_slice(bytes: &[u8]) -> Result<Self, XWingError> {
    if bytes.len() != Self::LENGTH {
      return Err(XWingError::InvalidDecapsulationKey);
    }
    let mut key = Self([0; Self::LENGTH]);
    key.0.copy_from_slice(bytes);
    Ok(key)
  }

  /// Generate a key from 32 bytes of `fill_random`.
  ///
  /// The callback must fill the entire buffer with fresh cryptographic
  /// randomness. Failure clears a partially filled buffer and returns no key.
  ///
  /// # Errors
  ///
  /// Returns the error from `fill_random`.
  pub fn generate(mut fill_random: impl FnMut(&mut [u8]) -> Result<(), XWingError>) -> Result<Self, XWingError> {
    let mut key = Self([0; Self::LENGTH]);
    fill_random(&mut key.0)?;
    Ok(key)
  }

  /// Generate a key from the platform entropy source.
  ///
  /// # Errors
  ///
  /// Returns [`XWingError::RandomGenerationFailed`] if the entropy source is unavailable.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  pub fn try_generate() -> Result<Self, XWingError> {
    Self::generate(os_random)
  }

  /// Derive the encapsulation key. This expands the key, as [`Self::prepare`] does.
  #[must_use]
  pub fn encapsulation_key(&self) -> XWingEncapsulationKey {
    // Expand in this frame: a prepared key returned by value would leave a
    // moved-from copy of the expanded secrets.
    let mut prepared = XWingPreparedDecapsulationKey::unexpanded();
    prepared.expand(self);
    prepared.encapsulation_key()
  }

  /// Expand this key one time for repeated decapsulation.
  ///
  /// The prepared key holds the expanded ML-KEM-768 and X25519 keys, about
  /// 11 KiB, and clears them on drop.
  #[must_use]
  pub fn prepare(&self) -> XWingPreparedDecapsulationKey {
    let mut prepared = XWingPreparedDecapsulationKey::unexpanded();
    prepared.expand(self);
    prepared
  }

  /// Like [`Self::prepare`], with the prepared key in memory from `alloc`.
  ///
  /// The key is expanded directly into its allocation, so moving the box
  /// moves only a pointer and leaves no by-value copy of the expanded
  /// secrets. The box clears them on drop. Allocation failure is handled as
  /// by [`Box::new_in`].
  #[cfg(feature = "alloc")]
  #[must_use]
  pub fn prepare_in<A: Allocator>(&self, alloc: A) -> Box<XWingPreparedDecapsulationKey, A> {
    let mut prepared = Box::new_in(XWingPreparedDecapsulationKey::unexpanded(), alloc);
    prepared.expand(self);
    prepared
  }
}

/// Validated, reusable X-Wing encapsulation key.
#[derive(Clone)]
pub struct XWingPreparedEncapsulationKey {
  ml_kem: MlKem768PreparedEncapsulationKey,
  x25519: X25519PublicKey,
}

impl XWingPreparedEncapsulationKey {
  /// Return the encapsulation key.
  #[must_use]
  pub fn encapsulation_key(&self) -> XWingEncapsulationKey {
    encapsulation_key(self.ml_kem.as_bytes(), &self.x25519)
  }

  /// `Encapsulate` of the draft, with 64 bytes from `fill_random`.
  ///
  /// The first 32 bytes are the ML-KEM-768 coins, and the last 32 bytes are
  /// the X25519 ephemeral secret key. The callback must fill the entire
  /// buffer with fresh cryptographic randomness.
  ///
  /// # Errors
  ///
  /// Returns the error from `fill_random`, or
  /// [`XWingError::InvalidEncapsulationKey`] when the X25519 shared secret is
  /// all-zero. That occurs exactly when the X25519 part of the key has small
  /// order.
  pub fn encapsulate(
    &self,
    fill_random: impl FnMut(&mut [u8]) -> Result<(), XWingError>,
  ) -> Result<(XWingCiphertext, XWingSharedSecret), XWingError> {
    let mut shared = XWingSharedSecret([0; XWingSharedSecret::LENGTH]);
    let ciphertext = self.encapsulate_into(fill_random, &mut shared.0)?;
    Ok((ciphertext, shared))
  }

  /// [`Self::encapsulate`], writing the shared secret into `shared`, so that
  /// no returned owner is moved.
  pub(crate) fn encapsulate_into(
    &self,
    mut fill_random: impl FnMut(&mut [u8]) -> Result<(), XWingError>,
    shared: &mut [u8; XWingSharedSecret::LENGTH],
  ) -> Result<XWingCiphertext, XWingError> {
    let _dit = DataIndependentTiming::enter();
    let mut random = ZeroizingBytes::<ENCAPSULATION_RANDOM_SIZE>::zeroed();
    fill_random(random.as_mut_array())?;
    let (coins, x25519_secret) = random
      .as_array()
      .split_first_chunk::<ML_KEM_COINS_SIZE>()
      .expect("the coins fit the encapsulation randomness");
    let mut ephemeral = X25519SecretKey::from_bytes([0; X25519SecretKey::LENGTH]);
    ephemeral.as_mut_bytes().copy_from_slice(x25519_secret);

    let mut x25519_shared = ZeroizingBytes::<{ X25519SharedSecret::LENGTH }>::zeroed();
    ephemeral
      .diffie_hellman_into(&self.x25519, x25519_shared.as_mut_array())
      .map_err(|_| XWingError::InvalidEncapsulationKey)?;
    let x25519_ciphertext = ephemeral.public_key();
    let (ml_kem_ciphertext, ml_kem_shared) = self.ml_kem.encapsulate_internal(coins);

    let mut ciphertext = [0u8; XWingCiphertext::LENGTH];
    let (ml_kem_part, x25519_part) = ciphertext.split_at_mut(ML_KEM_CIPHERTEXT_SIZE);
    ml_kem_part.copy_from_slice(ml_kem_ciphertext.as_bytes());
    x25519_part.copy_from_slice(x25519_ciphertext.as_bytes());
    combine(
      &ml_kem_shared,
      x25519_shared.as_array(),
      x25519_ciphertext.as_bytes(),
      self.x25519.as_bytes(),
      shared,
    );
    Ok(XWingCiphertext(ciphertext))
  }
}

impl fmt::Debug for XWingPreparedEncapsulationKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("XWingPreparedEncapsulationKey(")?;
    crate::hex::fmt_hex_lower(self.ml_kem.as_bytes(), f)?;
    crate::hex::fmt_hex_lower(self.x25519.as_bytes(), f)?;
    f.write_str(")")
  }
}

/// Expanded, reusable X-Wing decapsulation key.
///
/// It holds the decapsulation key and the expanded ML-KEM-768 and X25519
/// keys. It has no encoding: the draft forbids the transfer of an expanded key
/// between implementations. Zeroized on drop; `Debug` is redacted.
pub struct XWingPreparedDecapsulationKey {
  key: XWingDecapsulationKey,
  ml_kem: MlKem768PreparedDecapsulationKey,
  x25519: X25519SecretKey,
  x25519_public: X25519PublicKey,
}

impl XWingPreparedDecapsulationKey {
  /// Return the decapsulation key.
  #[must_use]
  pub const fn decapsulation_key(&self) -> &XWingDecapsulationKey {
    &self.key
  }

  /// Return the encapsulation key.
  #[must_use]
  pub fn encapsulation_key(&self) -> XWingEncapsulationKey {
    let embedded = self.ml_kem.as_bytes()[ML_KEM_EMBEDDED_KEY_OFFSET..]
      .first_chunk()
      .expect("an ML-KEM-768 decapsulation key embeds its encapsulation key");
    encapsulation_key(embedded, &self.x25519_public)
  }

  /// Explicitly duplicate this prepared key and its expanded secrets.
  #[must_use]
  pub fn duplicate_secret(&self) -> Self {
    Self {
      key: self.key.duplicate_secret(),
      ml_kem: self.ml_kem.duplicate_secret(),
      x25519: self.x25519.duplicate_secret(),
      x25519_public: self.x25519_public,
    }
  }

  /// `Decapsulate` of the draft.
  ///
  /// A ciphertext with a modified ML-KEM-768 part decapsulates to a
  /// pseudorandom shared secret (ML-KEM implicit rejection). The key does not
  /// change, also after a failure.
  ///
  /// # Errors
  ///
  /// Returns [`XWingError::InvalidCiphertext`] when the X25519 shared secret
  /// is all-zero. That occurs exactly when the X25519 part of the ciphertext
  /// has small order, so the error depends only on the public ciphertext.
  pub fn decapsulate(&self, ciphertext: &XWingCiphertext) -> Result<XWingSharedSecret, XWingError> {
    let mut shared = XWingSharedSecret([0; XWingSharedSecret::LENGTH]);
    self.decapsulate_into(ciphertext, &mut shared.0)?;
    Ok(shared)
  }

  /// [`Self::decapsulate`], writing the shared secret into `shared`, so that
  /// no returned owner is moved.
  pub(crate) fn decapsulate_into(
    &self,
    ciphertext: &XWingCiphertext,
    shared: &mut [u8; XWingSharedSecret::LENGTH],
  ) -> Result<(), XWingError> {
    let _dit = DataIndependentTiming::enter();
    let x25519_ciphertext = ciphertext.x25519();
    let mut x25519_shared = ZeroizingBytes::<{ X25519SharedSecret::LENGTH }>::zeroed();
    self
      .x25519
      .diffie_hellman_into(
        &X25519PublicKey::from_bytes(*x25519_ciphertext),
        x25519_shared.as_mut_array(),
      )
      .map_err(|_| XWingError::InvalidCiphertext)?;
    let ml_kem_shared = self.ml_kem.decapsulate_internal(ciphertext.ml_kem());
    combine(
      &ml_kem_shared,
      x25519_shared.as_array(),
      x25519_ciphertext,
      self.x25519_public.as_bytes(),
      shared,
    );
    Ok(())
  }

  /// All-zero owner that [`Self::expand`] fills in place.
  pub(crate) fn unexpanded() -> Self {
    Self {
      key: XWingDecapsulationKey([0; XWingDecapsulationKey::LENGTH]),
      ml_kem: MlKem768PreparedDecapsulationKey::unexpanded(),
      x25519: X25519SecretKey::from_bytes([0; X25519SecretKey::LENGTH]),
      x25519_public: X25519PublicKey::from_bytes([0; X25519PublicKey::LENGTH]),
    }
  }

  /// `expandDecapsulationKey` of the draft, written in place.
  pub(crate) fn expand(&mut self, key: &XWingDecapsulationKey) {
    let _dit = DataIndependentTiming::enter();
    let mut expanded = ZeroizingBytes::<EXPANDED_SIZE>::zeroed();
    secret::shake256(&[key.as_bytes().as_slice()], expanded.as_mut_array());
    let (ml_kem_seed, x25519_secret) = expanded
      .as_array()
      .split_first_chunk::<ML_KEM_SEED_SIZE>()
      .expect("the ML-KEM seed fits the expansion");
    self.ml_kem.expand_seed(ml_kem_seed);
    self.x25519.as_mut_bytes().copy_from_slice(x25519_secret);
    self.x25519_public = self.x25519.public_key();
    self.key.0 = key.0;
  }
}

impl fmt::Debug for XWingPreparedDecapsulationKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("XWingPreparedDecapsulationKey(****)")
  }
}

fn encapsulation_key(ml_kem: &[u8; ML_KEM_ENCAPSULATION_KEY_SIZE], x25519: &X25519PublicKey) -> XWingEncapsulationKey {
  let mut bytes = [0u8; XWingEncapsulationKey::LENGTH];
  let (ml_kem_part, x25519_part) = bytes.split_at_mut(ML_KEM_ENCAPSULATION_KEY_SIZE);
  ml_kem_part.copy_from_slice(ml_kem);
  x25519_part.copy_from_slice(x25519.as_bytes());
  XWingEncapsulationKey(bytes)
}

/// `Combiner` of the draft, written into `shared`.
fn combine(
  ml_kem_shared: &MlKem768SharedSecret,
  x25519_shared: &[u8; X25519SharedSecret::LENGTH],
  x25519_ciphertext: &[u8; X25519PublicKey::LENGTH],
  x25519_public: &[u8; X25519PublicKey::LENGTH],
  shared: &mut [u8; XWingSharedSecret::LENGTH],
) {
  secret::sha3_256(
    &[
      ml_kem_shared.as_bytes(),
      x25519_shared,
      x25519_ciphertext,
      x25519_public,
      LABEL,
    ],
    shared,
  );
}

#[cfg(feature = "getrandom")]
fn os_random(out: &mut [u8]) -> Result<(), XWingError> {
  getrandom::fill(out).map_err(|_| XWingError::RandomGenerationFailed)
}

#[cfg(test)]
mod tests {
  use serde_json::Value;

  use super::XWingDecapsulationKey;
  use crate::MlKem768Seed;

  fn hex<const N: usize>(case: &Value, name: &str) -> [u8; N] {
    let text = case[name].as_str().expect("hex vector field");
    let mut out = [0u8; N];
    crate::hex::from_hex(text, &mut out).expect("vector hex");
    out
  }

  /// The expansion feeds its first 64 bytes to ML-KEM-768 as `d || z` and its
  /// last 32 bytes to X25519. The public key and a valid shared secret do not
  /// depend on `z`, so only these intermediate values detect a wrong split.
  #[test]
  fn expansion_matches_concrete_hybrid_kem_intermediates() {
    let cases: Value = serde_json::from_str(include_str!(
      "../../testdata/xwing/concrete-hybrid-kems-04-mlkem768-x25519.json"
    ))
    .expect("concrete hybrid KEM vectors");
    for case in cases.as_array().expect("vector list") {
      let prepared = XWingDecapsulationKey::from_bytes(hex(case, "decapsulation_key")).prepare();
      let (_, ml_kem) = MlKem768Seed::from_bytes(hex(case, "decapsulation_key_pq")).keypair();
      assert!(prepared.ml_kem.decapsulation_key().ct_eq(&ml_kem).declassify());
      assert_eq!(prepared.x25519.as_bytes(), &hex::<32>(case, "decapsulation_key_t"));
    }
  }
}
